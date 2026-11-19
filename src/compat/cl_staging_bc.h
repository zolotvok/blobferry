// SPDX-License-Identifier: GPL-3.0-or-later
// cl_staging_bc.h: am_drv_alloc/free (core, used by RT emu + cover trim), staging pool, BC compute init. Included by mali_compat_layer.c.
// MALI_COMPAT_ADDRMAP: the layer's other driver allocations (cover-trim buffer, RT emulation rings/scratch) go through
// these two: L with the capture address (when the memlog has the query, i.e. MEMLOG=1), R before the free. Key unset:
// the same driver call, one bool test.
static VkResult am_drv_alloc(struct DeviceData *d, const char *purpose, const VkMemoryAllocateInfo *ai, VkDeviceMemory *m) {
    VkResult r = d->fpAllocateMemory(d->device, ai, NULL, m);
    if (g_addrmap) {
        uint32_t heap = ai->memoryTypeIndex < VK_MAX_MEMORY_TYPES ? d->type_heap[ai->memoryTypeIndex] : BUDGET_NO_HEAP;
        uint64_t va = 0;
        if (r == VK_SUCCESS && g_memlog == 1 && d->fpGetDeviceMemoryOpaqueCaptureAddress) {
            VkDeviceMemoryOpaqueCaptureAddressInfo ci = {VK_STRUCTURE_TYPE_DEVICE_MEMORY_OPAQUE_CAPTURE_ADDRESS_INFO, NULL, *m};
            va = d->fpGetDeviceMemoryOpaqueCaptureAddress(d->device, &ci);
        }
        am_layer_alloc(purpose, r, r == VK_SUCCESS ? *m : VK_NULL_HANDLE, ai->allocationSize, ai->memoryTypeIndex, heap, va, 0, 0);
    }
    return r;
}
static void am_drv_free(struct DeviceData *d, const char *purpose, VkDeviceMemory m) {
    if (g_addrmap && m) am_layer_free(purpose, m, AM_NA, ML_NONE, BUDGET_NO_HEAP, 0);
    d->fpFreeMemory(d->device, m, NULL);
}

static void destroy_staging(struct DeviceData *d, struct StagingResource *r) {
    if (g_bcr && r->addr) bcr_free(r->addr, r->size, false); // MALI_COMPAT_BCD_RING
    if (r->buffer) d->fpDestroyBuffer(d->device, r->buffer, NULL);
    if (r->memory) {
        if (g_addrmap) am_layer_free("bc-staging", r->memory, r->mem_size, r->type, r->heap, r->addr);
        d->fpFreeMemory(d->device, r->memory, NULL);
        if (r->mem_size) budget_sub(r->heap, r->mem_size);
    }
}

// Back into the device's pool, or freed when the pool is full.
static void release_staging(struct DeviceData *d, struct StagingResource *r) {
    struct StagingPoolNode *n = NULL;
    pthread_mutex_lock(&g_staging_pool_mutex);
    if (r->size && d->staging_pool_bytes + r->size <= STAGING_POOL_CAP && (n = malloc(sizeof *n))) {
        n->r = *r;
        n->next = d->staging_pool;
        d->staging_pool = n;
        d->staging_pool_bytes += r->size;
    }
    pthread_mutex_unlock(&g_staging_pool_mutex);
    if (!n) destroy_staging(d, r);
    else if (g_bcr) bcr_free(r->addr, r->size, true); // MALI_COMPAT_BCD_RING
}

static uint64_t now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}

// A staging buffer of at least `size` bytes with a device address: a pooled one of the same bucket, or a new one.
static VkResult acquire_staging(struct DeviceData *d, VkDeviceSize size, struct StagingResource *out) {
    VkDeviceSize bucket = STAGING_MIN;
    while (bucket < size) bucket <<= 1;
    pthread_mutex_lock(&g_staging_pool_mutex);
    for (struct StagingPoolNode **pp = &d->staging_pool; *pp; pp = &(*pp)->next) {
        if ((*pp)->r.size != bucket) continue;
        struct StagingPoolNode *n = *pp;
        *pp = n->next;
        d->staging_pool_bytes -= bucket;
        pthread_mutex_unlock(&g_staging_pool_mutex);
        *out = n->r;
        free(n);
        if (g_stats) atomic_fetch_add(&g_st_reused, 1);
        return VK_SUCCESS;
    }
    pthread_mutex_unlock(&g_staging_pool_mutex);

    *out = (struct StagingResource){.size = bucket};
    VkBufferCreateInfo b_ci = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, NULL, 0, bucket,
                               VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT};
    VkResult r = d->fpCreateBuffer(d->device, &b_ci, NULL, &out->buffer);
    if (r == VK_SUCCESS) {
        VkMemoryRequirements mem_req = {0};
        d->fpGetBufferMemoryRequirements(d->device, out->buffer, &mem_req);
        uint32_t type = d->device_local_memory_index;
        if (!(mem_req.memoryTypeBits & (1u << type))) type = (uint32_t)__builtin_ctz(mem_req.memoryTypeBits);
        VkMemoryAllocateFlagsInfo flags_info = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO, NULL, VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT};
        VkMemoryAllocateInfo a_info = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, &flags_info, mem_req.size, type};
        r = d->fpAllocateMemory(d->device, &a_info, NULL, &out->memory);
        if (r == VK_SUCCESS) {
            out->mem_size = mem_req.size;
            out->heap = type < VK_MAX_MEMORY_TYPES ? d->type_heap[type] : BUDGET_NO_HEAP;
            out->type = type;
            budget_add(out->heap, out->mem_size);
        } else if (g_addrmap) {
            am_layer_alloc("bc-staging", r, VK_NULL_HANDLE, mem_req.size, type, BUDGET_NO_HEAP, 0, 0, 0);
        }
    }
    if (r == VK_SUCCESS) r = d->fpBindBufferMemory(d->device, out->buffer, out->memory, 0);
    if (r != VK_SUCCESS) {
        if (g_addrmap && out->memory) // allocated, bind failed: L without an address, then destroy_staging's R
            am_layer_alloc("bc-staging", VK_SUCCESS, out->memory, out->mem_size, out->type, out->heap, 0,
                           (uint64_t)(uintptr_t)out->buffer, bucket);
        destroy_staging(d, out);
        return r;
    }
    VkBufferDeviceAddressInfo bda = {VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO, NULL, out->buffer};
    out->addr = d->fpGetBufferDeviceAddress(d->device, &bda);
    if (g_addrmap) // va = the staging buffer's device address (bound at offset 0 = the allocation's base)
        am_layer_alloc("bc-staging", VK_SUCCESS, out->memory, out->mem_size, out->type, out->heap, out->addr,
                       (uint64_t)(uintptr_t)out->buffer, bucket);
    if (g_stats) atomic_fetch_add(&g_st_new, 1);
    return VK_SUCCESS;
}

static void destroy_staging_pool(struct DeviceData *d) {
    pthread_mutex_lock(&g_staging_pool_mutex);
    struct StagingPoolNode *n = d->staging_pool;
    d->staging_pool = NULL;
    d->staging_pool_bytes = 0;
    pthread_mutex_unlock(&g_staging_pool_mutex);
    while (n) {
        struct StagingPoolNode *next = n->next;
        destroy_staging(d, &n->r);
        free(n);
        n = next;
    }
}

static void rt_reset_cmd_state(VkCommandBuffer cmd, bool remove);

// Begin/reset: staging buffers back to the pool, tracked state cleared; the entry stays. Free: the entry goes.
static void reset_cmd_data(struct DeviceData *dev_data, VkCommandBuffer cmd, bool remove) {
    if (g_rt_pipeline) rt_reset_cmd_state(cmd, remove);
    pthread_mutex_lock(&g_cmdbuf_mutex);
    struct CmdBufData **pp = &g_cmdbufs[handle_bucket((uint64_t)(uintptr_t)cmd, CMDBUF_BUCKETS)];
    while (*pp && (*pp)->cmdbuf != cmd) pp = &(*pp)->next;
    struct CmdBufData *e = *pp;
    if (e) staged_unlink(e);
    if (e && remove) {
        *pp = e->next;
        atomic_fetch_add_explicit(&g_cmdbuf_gen, 1, memory_order_release);
    }
    pthread_mutex_unlock(&g_cmdbuf_mutex);
    if (!e) return;
    for (uint32_t j = 0; j < e->staging_count; j++) release_staging(dev_data, &e->staging_resources[j]);
    e->staging_count = 0;
    e->compute_pipeline = VK_NULL_HANDLE;
    e->descriptor_buffers = false;
    e->push_count = 0;
    e->cpd.layout = VK_NULL_HANDLE;
    e->gpd.layout = VK_NULL_HANDLE;
    e->gfx_pipeline = VK_NULL_HANDLE;
    e->gfx_flags = 0;
    e->gt_rec = -1;
    e->gt_src = VK_NULL_HANDLE;
    e->gt_sdraws = e->gt_svcount = e->gt_sicount = 0;
    memset(&e->gt_scnt, 0, sizeof e->gt_scnt);
    e->gt_spipe = VK_NULL_HANDLE;
    e->gt_ssrc = VK_NULL_HANDLE;
    e->async_skip = false;
    memset(&e->dc, 0, sizeof e->dc);
    memset(&e->ct, 0, sizeof e->ct);
    if (g_dpd_on) memset(&e->dpd, 0, sizeof e->dpd);
    if (remove) {
        free(e->staging_resources);
        free(e);
    }
}

// Memory budget tracking: the old report (bytes ever allocated, every heap). It stays the reported value while
// MALI_COMPAT_BUDGET_MB is unset (deployed 2dfc35b8 behaviour); the live per-heap accounting is g_heap_usage.
static _Atomic uint64_t g_allocated_bytes = 0;

// -----------------------------------------------------------------------------
// BC Compute Pipeline Initializer
// -----------------------------------------------------------------------------
static VkResult init_bc_compute(struct DeviceData *d) {
    if (d->bc_init_done) return VK_SUCCESS;

    // Find a memory type with DEVICE_LOCAL
    VkPhysicalDeviceMemoryProperties mem_props;
    d->instance_data->fpGetPhysicalDeviceMemoryProperties(d->physical_device, &mem_props);
    d->device_local_memory_index = 0;
    for (uint32_t i = 0; i < mem_props.memoryTypeCount; i++) {
        if (mem_props.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) {
            d->device_local_memory_index = i;
            break;
        }
    }

    VkPushConstantRange push_range = {
        .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        .offset = 0,
        .size = sizeof(struct panvk_bc_decode_push), // 48 bytes
    };

    VkPipelineLayoutCreateInfo pl_info = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = 0,
        .pushConstantRangeCount = 1,
        .pPushConstantRanges = &push_range,
    };

    VkResult res = d->fpCreatePipelineLayout(d->device, &pl_info, NULL, &d->bc_layout);
    if (res != VK_SUCCESS) {
        fprintf(stderr, "mali-compat: BC decode: pipeline layout failed (%d); BC textures stay undecoded\n", res);
        return res;
    }

    static const struct {
        const uint32_t *code;
        size_t size;
    } spv[4] = {
        { panvk_bc_s3tc_spv, sizeof(panvk_bc_s3tc_spv) },
        { panvk_bc_rgtc_spv, sizeof(panvk_bc_rgtc_spv) },
        { panvk_bc_bc6_spv, sizeof(panvk_bc_bc6_spv) },
        { panvk_bc_bc7_spv, sizeof(panvk_bc_bc7_spv) },
    };

    for (int i = 0; i < 4; i++) {
        VkShaderModuleCreateInfo sm_info = {
            .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
            .codeSize = spv[i].size,
            .pCode = spv[i].code,
        };
        VkShaderModule mod = VK_NULL_HANDLE;
        res = d->fpCreateShaderModule(d->device, &sm_info, NULL, &mod);
        if (res != VK_SUCCESS) {
            fprintf(stderr, "mali-compat: BC decode: shader module %d failed (%d); BC textures stay undecoded\n", i, res);
            return res;
        }

        VkComputePipelineCreateInfo cp_info = {
            .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
            .stage = {
                .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                .stage = VK_SHADER_STAGE_COMPUTE_BIT,
                .module = mod,
                .pName = "main",
            },
            .layout = d->bc_layout,
        };

        res = d->fpCreateComputePipelines(d->device, VK_NULL_HANDLE, 1, &cp_info, NULL, &d->bc_pipelines[i]);
        if (res == VK_SUCCESS && d->descriptor_buffer) {
            // A pipeline without DESCRIPTOR_BUFFER_BIT must not be used where descriptor buffers are bound.
            cp_info.flags = VK_PIPELINE_CREATE_DESCRIPTOR_BUFFER_BIT_EXT;
            res = d->fpCreateComputePipelines(d->device, VK_NULL_HANDLE, 1, &cp_info, NULL, &d->bc_pipelines_db[i]);
        }
        d->fpDestroyShaderModule(d->device, mod, NULL);
        if (res != VK_SUCCESS) {
            fprintf(stderr, "mali-compat: BC decode: compute pipeline %d failed (%d); BC textures stay undecoded\n", i, res);
            return res;
        }
    }

    d->bc_init_done = true;
    LOG("initialized PanVK BC compute pipelines successfully");
    return VK_SUCCESS;
}

// -----------------------------------------------------------------------------
// Placed memory tracking & cache maintenance forward definitions for RT emu
// -----------------------------------------------------------------------------
struct MemInfo {
    VkDeviceMemory memory;
    VkDeviceSize size;
    uint32_t type;
    void *orig;   // driver's mapping address while placed, else NULL
    void *placed; // where the mapping lives for the app
    size_t map_size;
    uint32_t heap; // counted in g_heap_usage[heap] (BUDGET_NO_HEAP = not counted)
    uint64_t va;   // memlog: last mapped base (pointer - offset), 0 = never mapped
    uint64_t cva;  // MALI_COMPAT_ADDRMAP: the capture address of the A event, 0 = none
    struct MemInfo *next;
};
#define MEM_BUCKETS 4096
static struct MemInfo *g_mem[MEM_BUCKETS];
static pthread_mutex_t g_mem_mutex = PTHREAD_MUTEX_INITIALIZER;

static uint32_t mem_bucket(VkDeviceMemory m) {
    uint64_t v = (uint64_t)(uintptr_t)m;
    return (uint32_t)((v ^ (v >> 17) ^ (v >> 33)) % MEM_BUCKETS);
}

// Called with g_mem_mutex held.
static struct MemInfo *mem_find(VkDeviceMemory m) {
    for (struct MemInfo *e = g_mem[mem_bucket(m)]; e; e = e->next)
        if (e->memory == m) return e;
    return NULL;
}

static void cache_maintenance(void *addr, VkDeviceSize size, bool invalidate) {
#if defined(__aarch64__)
    uint64_t ctr;
    __asm__ volatile("mrs %0, ctr_el0" : "=r"(ctr));
    uintptr_t line = 4u << ((ctr >> 16) & 0xf); // DminLine, in bytes
    uintptr_t p = (uintptr_t)addr & ~(line - 1), end = (uintptr_t)addr + size;
    for (; p < end; p += line) {
        if (invalidate) __asm__ volatile("dc civac, %0" ::"r"(p) : "memory");
        else __asm__ volatile("dc cvac, %0" ::"r"(p) : "memory");
    }
    __asm__ volatile("dsb ish" ::: "memory");
#else
    (void)addr, (void)size, (void)invalidate; // only the host test harness builds this file elsewhere
#endif
}


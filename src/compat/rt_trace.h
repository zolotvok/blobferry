// SPDX-License-Identifier: GPL-3.0-or-later
// rt_trace.h: ring buffer, push constants, TraceRays + indirect, FEX thunk patch, transfer buffers. Included by mali_rt_emu.h.
/* ── TraceRays Dispatch Execution & Persistent Ring Buffer ─────────────────── */

#define RT_RING_SIZE (4u * 1024u * 1024u)
#define RT_RING_SLOT_SIZE 256u
#define RT_RING_NUM_SLOTS (RT_RING_SIZE / RT_RING_SLOT_SIZE) /* 16384 slots */

static bool rt_ensure_ring_buffer(struct DeviceData *d) {
    if (!d) return false;
    if (d->rt_ring_bda != 0ULL && d->rt_ring_ptr != NULL) return true;

    pthread_mutex_lock(&d->bc_lock);
    if (d->rt_ring_bda != 0ULL && d->rt_ring_ptr != NULL) {
        pthread_mutex_unlock(&d->bc_lock);
        return true;
    }

    if (!d->fpCreateBuffer || !d->fpAllocateMemory || !d->fpBindBufferMemory ||
        !d->fpGetBufferDeviceAddress || !d->fpMapMemory) {
        pthread_mutex_unlock(&d->bc_lock);
        return false;
    }

    VkBufferCreateInfo bci = {
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .size  = RT_RING_SIZE,
        .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT |
                 VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
    };
    VkBuffer buf = VK_NULL_HANDLE;
    if (d->fpCreateBuffer(d->device, &bci, NULL, &buf) != VK_SUCCESS) {
        pthread_mutex_unlock(&d->bc_lock);
        return false;
    }

    VkMemoryRequirements req = {0};
    d->fpGetBufferMemoryRequirements(d->device, buf, &req);
    uint32_t mask = req.memoryTypeBits;
    if (mask & d->coherent_types) {
        mask &= d->coherent_types;
    }
    uint32_t mtype = mask ? (uint32_t)__builtin_ctz(mask) : 0u;

    VkMemoryAllocateFlagsInfo mafi = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO,
        .flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT,
    };
    VkMemoryAllocateInfo mai = {
        .sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        .pNext           = &mafi,
        .allocationSize  = req.size < RT_RING_SIZE ? RT_RING_SIZE : req.size,
        .memoryTypeIndex = mtype,
    };
    VkDeviceMemory mem = VK_NULL_HANDLE;
    if (am_drv_alloc(d, "rt-ring", &mai, &mem) != VK_SUCCESS) {
        d->fpDestroyBuffer(d->device, buf, NULL);
        pthread_mutex_unlock(&d->bc_lock);
        return false;
    }

    if (d->fpBindBufferMemory(d->device, buf, mem, 0) != VK_SUCCESS) {
        am_drv_free(d, "rt-ring", mem);
        d->fpDestroyBuffer(d->device, buf, NULL);
        pthread_mutex_unlock(&d->bc_lock);
        return false;
    }

    void *ptr = NULL;
    if (d->fpMapMemory(d->device, mem, 0, RT_RING_SIZE, 0, &ptr) != VK_SUCCESS || !ptr) {
        am_drv_free(d, "rt-ring", mem);
        d->fpDestroyBuffer(d->device, buf, NULL);
        pthread_mutex_unlock(&d->bc_lock);
        return false;
    }
    memset(ptr, 0, RT_RING_SIZE);

    VkBufferDeviceAddressInfo bdai = {
        .sType  = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO,
        .buffer = buf,
    };
    VkDeviceAddress bda = d->fpGetBufferDeviceAddress(d->device, &bdai);
    if (!bda) {
        d->fpUnmapMemory(d->device, mem);
        am_drv_free(d, "rt-ring", mem);
        d->fpDestroyBuffer(d->device, buf, NULL);
        pthread_mutex_unlock(&d->bc_lock);
        return false;
    }

    d->rt_ring_buf = buf;
    d->rt_ring_mem = mem;
    d->rt_ring_bda = bda;
    d->rt_ring_ptr = ptr;
    d->rt_ring_size = mai.allocationSize;
    rt_acct_add(&g_acct_ring, (int64_t)mai.allocationSize);
    atomic_store(&d->rt_ring_offset, 0u);
    pthread_mutex_unlock(&d->bc_lock);
    COMPAT_INFO("rt_ensure_ring_buffer: 4MB persistent ring allocated at BDA 0x%lx host=%p",
                (unsigned long)bda, ptr);
    return true;
}

static VkDeviceAddress rt_ring_alloc(struct DeviceData *d,
                                     uint32_t byte_size,
                                     uint32_t align,
                                     void **out_ptr,
                                     uint32_t *out_offset) {
    if (!rt_ensure_ring_buffer(d)) return 0ULL;
    (void)align;
    uint32_t slots_needed = (byte_size + RT_RING_SLOT_SIZE - 1u) / RT_RING_SLOT_SIZE;
    if (slots_needed == 0u) slots_needed = 1u;
    /* last slot = guard counter (RT_GUARD_SLOT_OFF), never handed out */
    if (slots_needed > RT_RING_NUM_SLOTS - 1u) return 0ULL;
    uint32_t max_first_slot = RT_RING_NUM_SLOTS - 1u - slots_needed;
    uint32_t raw_slot = atomic_fetch_add(&d->rt_ring_offset, slots_needed);
    uint32_t slot = raw_slot % (max_first_slot + 1u);
    uint32_t off = slot * RT_RING_SLOT_SIZE;
    if (out_ptr) {
        *out_ptr = (void *)((uintptr_t)d->rt_ring_ptr + off);
    }
    if (out_offset) {
        *out_offset = off;
    }
    return d->rt_ring_bda + (VkDeviceAddress)off;
}

static void rt_guard_log(struct DeviceData *d) {
    static _Atomic int64_t last_ns = 0;
    static uint32_t last_val = 0;
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    int64_t now = (int64_t)ts.tv_sec * 1000000000LL + ts.tv_nsec;
    int64_t prev = atomic_load(&last_ns);
    if (now - prev < 1000000000LL) return;
    if (!atomic_compare_exchange_strong(&last_ns, &prev, now)) return;
    uint32_t v = *(volatile uint32_t *)((uint8_t *)d->rt_ring_ptr + RT_GUARD_SLOT_OFF);
    COMPAT_INFO("rt_guard: %u rays cut as miss after %u ray-query iterations (+%u in the last 1 s)",
                v, g_rt_guard_max, v - last_val);
    last_val = v;
}

static VkDeviceAddress rt_get_zero_bda(struct DeviceData *d) {
    if (!d) return 0ULL;
    if (d->rt_zero_bda != 0ULL) return d->rt_zero_bda;

    pthread_mutex_lock(&d->bc_lock);
    if (d->rt_zero_bda != 0ULL) {
        pthread_mutex_unlock(&d->bc_lock);
        return d->rt_zero_bda;
    }

    if (!d->fpCreateBuffer || !d->fpAllocateMemory || !d->fpBindBufferMemory ||
        !d->fpGetBufferDeviceAddress) {
        pthread_mutex_unlock(&d->bc_lock);
        return 0ULL;
    }

    VkBufferCreateInfo bci = {
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .size  = 4096,
        .usage = VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
    };
    VkBuffer buf = VK_NULL_HANDLE;
    if (d->fpCreateBuffer(d->device, &bci, NULL, &buf) != VK_SUCCESS) {
        pthread_mutex_unlock(&d->bc_lock);
        return 0ULL;
    }

    VkMemoryRequirements req = {0};
    d->fpGetBufferMemoryRequirements(d->device, buf, &req);
    uint32_t mask = req.memoryTypeBits;
    if (mask & d->coherent_types) {
        mask &= d->coherent_types;
    }
    uint32_t mtype = mask ? (uint32_t)__builtin_ctz(mask) : 0u;

    VkMemoryAllocateFlagsInfo mafi = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO,
        .flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT,
    };
    VkMemoryAllocateInfo mai = {
        .sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        .pNext           = &mafi,
        .allocationSize  = req.size,
        .memoryTypeIndex = mtype,
    };
    VkDeviceMemory mem = VK_NULL_HANDLE;
    if (am_drv_alloc(d, "rt-zero", &mai, &mem) != VK_SUCCESS) {
        d->fpDestroyBuffer(d->device, buf, NULL);
        pthread_mutex_unlock(&d->bc_lock);
        return 0ULL;
    }

    if (d->fpBindBufferMemory(d->device, buf, mem, 0) != VK_SUCCESS) {
        am_drv_free(d, "rt-zero", mem);
        d->fpDestroyBuffer(d->device, buf, NULL);
        pthread_mutex_unlock(&d->bc_lock);
        return 0ULL;
    }

    if (d->fpMapMemory && d->fpUnmapMemory) {
        void *ptr = NULL;
        if (d->fpMapMemory(d->device, mem, 0, 4096, 0, &ptr) == VK_SUCCESS && ptr) {
            memset(ptr, 0, 4096);
            d->fpUnmapMemory(d->device, mem);
        }
    }

    VkBufferDeviceAddressInfo bdai = {
        .sType  = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO,
        .buffer = buf,
    };
    VkDeviceAddress bda = d->fpGetBufferDeviceAddress(d->device, &bdai);

    d->rt_zero_buf = buf;
    d->rt_zero_mem = mem;
    d->rt_zero_bda = bda;
    d->rt_zero_size = mai.allocationSize;
    rt_acct_add(&g_acct_zero, (int64_t)mai.allocationSize);

    COMPAT_INFO("RT_EMU: initialized persistent 4KB zero scratch buffer: bda=0x%llx", (unsigned long long)bda);
    pthread_mutex_unlock(&d->bc_lock);
    return bda;
}

static VkDeviceAddress rt_upload_user_push_constants(struct DeviceData *d,
                                                     VkCommandBuffer cmd,
                                                     const struct RtCmdState *cs) {
    (void)cmd;
    if (!cs || cs->user_pc_dirty_size == 0) {
        return rt_get_zero_bda(d);
    }
    if (g_debug && d && d->rt_ring_ptr) rt_guard_log(d);
    void *ptr = NULL;
    uint32_t off = 0;
    VkDeviceAddress bda = rt_ring_alloc(d, 256, 256, &ptr, &off);
    if (bda && d->fpCmdUpdateBuffer && d->rt_ring_buf && d->fpCmdPipelineBarrier &&
        !getenv("MALI_COMPAT_RT_PARAM_HOSTWRITE")) {
        /* param-fix 11-08: the per-dispatch params (vkd3d root constants + root descriptor VAs, incl. the
         * TLAS VA) are written by the GPU in command-stream order right before this dispatch, instead of a
         * host memcpy at record time into a slot that the ring hands out again later (07cd745).  The data
         * now travels inside the command buffer: a resubmitted / late-submitted / still-pending command
         * buffer always reads its own params, no matter how often the ring wrapped. */
        d->fpCmdUpdateBuffer(cmd, d->rt_ring_buf, (VkDeviceSize)off, 256, cs->user_push_constants);
        VkMemoryBarrier ub = {
            .sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
            .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
            .dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_UNIFORM_READ_BIT,
        };
        d->fpCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                0, 1, &ub, 0, NULL, 0, NULL);
        return bda;
    }
    if (bda && ptr) {   /* fallback / A-B lever MALI_COMPAT_RT_PARAM_HOSTWRITE=1: old host write */
        memcpy(ptr, cs->user_push_constants, 256);
        return bda;
    }
    return rt_get_zero_bda(d);
}

static void rt_prepare_and_bind_megakernel(struct DeviceData *d,
                                           VkCommandBuffer cmd,
                                           struct RtCmdState *cs,
                                           struct RtPipeline *rp,
                                           const struct RtDispatchPushConstants *pc) {
    /* Bind compiled megakernel to COMPUTE bind point */
    d->fpCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, rp->compute_pipeline);

    /* If descriptor buffer offsets were set on RAY_TRACING_KHR, ensure COMPUTE has them */
    if (cs && cs->has_rt_desc_buf_offsets && d->fpCmdSetDescriptorBufferOffsetsEXT) {
        d->fpCmdSetDescriptorBufferOffsetsEXT(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                              cs->rt_db_layout ? cs->rt_db_layout : rp->layout,
                                              cs->rt_db_first_set, cs->rt_db_set_count,
                                              cs->rt_db_buf_indices, cs->rt_db_offsets);
    }

    /* Push 96-byte RtDispatchPushConstants */
    VkShaderStageFlags pc_stages = rt_get_layout_pc_stages(rp->layout);
    d->fpCmdPushConstants(cmd, rp->layout, pc_stages, 0, sizeof(*pc), pc);
}

static void rt_restore_post_trace_state(struct DeviceData *d,
                                        VkCommandBuffer cmd,
                                        struct RtCmdState *cs,
                                        struct RtPipeline *rp) {
    if (!cs) return;
    /* Restore user push constants (first 96 bytes) if dirty */
    if (cs->user_pc_dirty_size > 0) {
        VkShaderStageFlags pc_stages = rt_get_layout_pc_stages(rp->layout);
        d->fpCmdPushConstants(cmd, rp->layout, pc_stages, 0,
                              sizeof(struct RtDispatchPushConstants), cs->user_push_constants);
    }
    /* Restore previous compute pipeline if one was bound */
    if (cs->bound_comp_pipeline) {
        d->fpCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, cs->bound_comp_pipeline);
    }
}

/* split-1112: the select pipeline (one per device, like rt_ind_pipeline) */
static void rt_ensure_split_select_pipeline(struct DeviceData *d) {
    if (__atomic_load_n(&d->rt_split_pipeline, __ATOMIC_ACQUIRE)) return;
    pthread_mutex_lock(&d->bc_lock);
    if (d->rt_split_pipeline || d->rt_split_failed) {
        pthread_mutex_unlock(&d->bc_lock);
        return;
    }
    d->rt_split_failed = true;   /* one attempt; cleared on success */
    size_t wc = 0;
#ifdef RT_SPLIT_TEST_NOSELECT   /* PC test build only: exercise the fallback (every child at full size) */
    uint32_t *spv = NULL;
    (void)rt_build_split_select_spirv;
#else
    uint32_t *spv = rt_build_split_select_spirv(&wc);
#endif
    if (spv && wc > 5) {
        VkPushConstantRange pcr = { .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT, .offset = 0,
                                    .size = sizeof(struct RtSplitSelectPC) };
        VkPipelineLayoutCreateInfo plci = { .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
                                            .pushConstantRangeCount = 1, .pPushConstantRanges = &pcr };
        if (!d->rt_split_layout && d->fpCreatePipelineLayout(d->device, &plci, NULL, &d->rt_split_layout) != VK_SUCCESS)
            d->rt_split_layout = VK_NULL_HANDLE;
        if (d->rt_split_layout) {
            VkShaderModuleCreateInfo smci = { .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
                                              .codeSize = wc * sizeof(uint32_t), .pCode = spv };
            VkShaderModule mod = VK_NULL_HANDLE;
            if (d->fpCreateShaderModule(d->device, &smci, NULL, &mod) == VK_SUCCESS) {
                VkComputePipelineCreateInfo cpci = {
                    .sType  = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
                    .stage  = { .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                                .stage = VK_SHADER_STAGE_COMPUTE_BIT, .module = mod, .pName = "main" },
                    .layout = d->rt_split_layout,
                };
                VkPipeline p = VK_NULL_HANDLE;
                if (d->fpCreateComputePipelines(d->device, VK_NULL_HANDLE, 1, &cpci, NULL, &p) == VK_SUCCESS && p) {
                    __atomic_store_n(&d->rt_split_pipeline, p, __ATOMIC_RELEASE);
                    d->rt_split_failed = false;
                }
                d->fpDestroyShaderModule(d->device, mod, NULL);
            }
        }
    }
    if (spv && (!d->rt_split_pipeline || g_debug) && rt_env()->dump_spv) {   /* failure or DEBUG: for spirv-val */
        const char *dir = getenv("MALI_COMPAT_DUMP_RT_SPV");
        char path[512];
        snprintf(path, sizeof path, "%s/rt_split_select%s.spv", (dir && dir[0]) ? dir : "/tmp",
                 d->rt_split_pipeline ? "" : "_failed");
        FILE *fp = fopen(path, "wb");
        if (fp) { fwrite(spv, 4, wc, fp); fclose(fp); }
    }
    free(spv);
    LOG("rt_split: select pipeline %s (%zu words)", d->rt_split_pipeline ? "created" : "FAILED -> every child is "
        "dispatched at full size (each runs only its own raygen)", wc);
    pthread_mutex_unlock(&d->bc_lock);
}

/* split-1112: record one emulated trace of a split pipeline. Called where the unsplit path binds the megakernel and
 * dispatches (after the pre-trace barrier, before the post-trace barrier); w/h/dpt = direct launch size (indirect: 0,
 * the select kernel reads the command). */
static void rt_ensure_indirect_setup_pipeline(struct DeviceData *d);
static void rt_split_dispatch(struct DeviceData *d, VkCommandBuffer cmd, struct RtCmdState *cs, struct RtPipeline *rp,
                              const struct RtDispatchPushConstants *pc, uint32_t w, uint32_t h, uint32_t dpt) {
    atomic_fetch_add_explicit(&g_split_traces, 1, memory_order_relaxed);
    rt_ensure_split_select_pipeline(d);
    uint32_t off = 0;
    VkDeviceAddress dst = d->rt_split_pipeline ? rt_ring_alloc(d, RT_SPLIT_MAX * 16u, 16u, NULL, &off) : 0ULL;
    VkDeviceAddress zero = rt_get_zero_bda(d);
    if (d->rt_split_pipeline && d->fpCmdDispatchIndirect && d->rt_ring_buf && dst && zero) {
        struct RtSplitSelectPC sp;
        memset(&sp, 0, sizeof sp);
        sp.rgen_base = pc->rgen_base;
        sp.dst_bda   = dst;
        sp.ind_bda   = (pc->indirect_mode & 0xffu) ? pc->indirect_bda : 0ULL;
        sp.zero_bda  = zero;
        sp.mode      = pc->indirect_mode & 0xffu;
        sp.count     = rp->split_n;
        sp.w = w; sp.h = h; sp.d = dpt;
        if (pc->indirect_mode & RT_PC_LOCAL1D) {   /* local1d-1112: the select kernel computes ((w+7)>>3, (h+3)>>2, d):
                                                    * feed it values that give (ceil(w/32), h, d) */
            sp.w = ((w + 31u) / 32u) * 8u;
            sp.h = h * 4u;
        }
        sp.rgen_size = pc->rgen_size;
        for (uint32_t k = 0; k < rp->split_n; k++) sp.ids[k] = rp->split_rgen[k];
        d->fpCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, d->rt_split_pipeline);
        d->fpCmdPushConstants(cmd, d->rt_split_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof sp, &sp);
        d->fpCmdDispatch(cmd, 1, 1, 1);
        VkMemoryBarrier mb_bar = { .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER, .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
                                   .dstAccessMask = VK_ACCESS_INDIRECT_COMMAND_READ_BIT };
        d->fpCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT,
                                0, 1, &mb_bar, 0, NULL, 0, NULL);
        for (uint32_t k = 0; k < rp->split_n; k++) {
            rt_prepare_and_bind_megakernel(d, cmd, cs, rp->split[k], pc);
            d->fpCmdDispatchIndirect(cmd, d->rt_ring_buf, (VkDeviceSize)off + 16u * k);
        }
        atomic_fetch_add_explicit(&g_split_select, 1, memory_order_relaxed);
        return;
    }
    /* fallback: every child at the launch size; a child's megakernel runs only its own raygen (its switch has one
     * case), so exactly one of them does work - correct, only slower. Indirect: the launch size from the unsplit path's
     * indirect setup kernel (Codex 1111-201541 #2), its old 256x256 fallback only when that kernel is missing too. */
    uint32_t ind_off = 0;
    VkDeviceAddress ind_dst = 0ULL;
    const uint32_t ind_mode = pc->indirect_mode & 0xffu;   /* local1d-1112: without the RT_PC_LOCAL1D flag bit */
    if (ind_mode) {
        rt_ensure_indirect_setup_pipeline(d);
        if (d->rt_ind_pipeline && d->fpCmdDispatchIndirect && d->rt_ring_buf)
            ind_dst = rt_ring_alloc(d, 64, 64, NULL, &ind_off);
        if (ind_dst) {
            uint64_t setup_pc[2] = { pc->indirect_bda + (ind_mode == 2 ? 88ULL : 0ULL), ind_dst };
            d->fpCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, d->rt_ind_pipeline);
            d->fpCmdPushConstants(cmd, d->rt_ind_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, 16, setup_pc);
            d->fpCmdDispatch(cmd, 1, 1, 1);
            VkMemoryBarrier ib = { .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER, .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
                                   .dstAccessMask = VK_ACCESS_INDIRECT_COMMAND_READ_BIT };
            d->fpCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT,
                                    0, 1, &ib, 0, NULL, 0, NULL);
        }
    }
    const bool l1d = (pc->indirect_mode & RT_PC_LOCAL1D) != 0;
    for (uint32_t k = 0; k < rp->split_n; k++) {
        rt_prepare_and_bind_megakernel(d, cmd, cs, rp->split[k], pc);
        if (l1d) d->fpCmdDispatch(cmd, (w + 31u) / 32u, h, dpt);
        else if (!ind_mode) d->fpCmdDispatch(cmd, (w + 7u) / 8u, (h + 3u) / 4u, dpt);
        else if (ind_dst) d->fpCmdDispatchIndirect(cmd, d->rt_ring_buf, (VkDeviceSize)ind_off);
        else d->fpCmdDispatch(cmd, 256, 256, 1);
    }
    atomic_fetch_add_explicit(&g_split_fallback, 1, memory_order_relaxed);
}

static VKAPI_ATTR void VKAPI_CALL
compat_CmdTraceRaysKHR(
    VkCommandBuffer                             commandBuffer,
    const VkStridedDeviceAddressRegionKHR      *pRaygenShaderBindingTable,
    const VkStridedDeviceAddressRegionKHR      *pMissShaderBindingTable,
    const VkStridedDeviceAddressRegionKHR      *pHitShaderBindingTable,
    const VkStridedDeviceAddressRegionKHR      *pCallableShaderBindingTable,
    uint32_t                                    width,
    uint32_t                                    height,
    uint32_t                                    depth)
{
    rt_patch_fex_thunks();
    COMPAT_TRACE("compat_CmdTraceRaysKHR: cmd=%p width=%u height=%u depth=%u",
                 (void *)commandBuffer, width, height, depth);
    rt_note_trace();   /* MALI_COMPAT_RT_BUILD_LOG: marks the first recorded TraceRays */
    struct DeviceData *d = get_device_data_by_cmd(commandBuffer);
    rt_report(d, false);   /* round 4: timed rt_mem/vk_mem/rt_rays (also when this trace is skipped) */
    if (width == 0 || height == 0 || depth == 0) {
        atomic_fetch_add_explicit(&g_ray_skip_zero, 1, memory_order_relaxed);
        return;
    }
    if (!d) { atomic_fetch_add_explicit(&g_ray_skip_nostate, 1, memory_order_relaxed); return; }
    struct RtCmdState *cs = rt_get_cmd_state(commandBuffer, false);
    if (!cs || !cs->bound_rt_pipeline) { atomic_fetch_add_explicit(&g_ray_skip_nostate, 1, memory_order_relaxed); return; }
    if (rt_forced_trace_skip()) return;
    if (!rt_pipe_ready(cs->bound_rt_pipeline)) { rt_count_skip(cs->bound_rt_pipeline); return; }
    struct RtPipeline *rp = cs->bound_rt_pipeline;

    struct RtDispatchPushConstants pc;
    memset(&pc, 0, sizeof(pc));
    if (pRaygenShaderBindingTable) {
        VkDeviceAddress addr = pRaygenShaderBindingTable->deviceAddress;
        if (rt_is_invalid_gpu_bda(addr) || addr == 0ULL) {
            pc.rgen_base   = rt_get_zero_bda(d);
            pc.rgen_stride = 0;
            pc.rgen_size   = 0;
        } else {
            pc.rgen_base   = addr;
            pc.rgen_stride = (uint32_t)pRaygenShaderBindingTable->stride;
            pc.rgen_size   = (uint32_t)pRaygenShaderBindingTable->size;
        }
    }
    if (pMissShaderBindingTable) {
        VkDeviceAddress addr = pMissShaderBindingTable->deviceAddress;
        if (rt_is_invalid_gpu_bda(addr) || addr == 0ULL) {
            pc.miss_base   = rt_get_zero_bda(d);
            pc.miss_stride = 0;
            pc.miss_size   = 0;
        } else {
            pc.miss_base   = addr;
            pc.miss_stride = (uint32_t)pMissShaderBindingTable->stride;
            pc.miss_size   = (uint32_t)pMissShaderBindingTable->size;
        }
    }
    if (pHitShaderBindingTable) {
        VkDeviceAddress addr = pHitShaderBindingTable->deviceAddress;
        if (rt_is_invalid_gpu_bda(addr) || addr == 0ULL) {
            pc.hit_base    = rt_get_zero_bda(d);
            pc.hit_stride  = 0;
            pc.hit_size    = 0;
        } else {
            pc.hit_base    = addr;
            pc.hit_stride  = (uint32_t)pHitShaderBindingTable->stride;
            pc.hit_size    = (uint32_t)pHitShaderBindingTable->size;
        }
    }
    if (pCallableShaderBindingTable) {
        VkDeviceAddress addr = pCallableShaderBindingTable->deviceAddress;
        if (rt_is_invalid_gpu_bda(addr) || addr == 0ULL) {
            pc.call_base   = rt_get_zero_bda(d);
            pc.call_stride = 0;
            pc.call_size   = 0;
        } else {
            pc.call_base   = addr;
            pc.call_stride = (uint32_t)pCallableShaderBindingTable->stride;
            pc.call_size   = (uint32_t)pCallableShaderBindingTable->size;
        }
    }
    pc.launch_width    = width;
    pc.launch_height   = height;
    pc.launch_depth    = depth;
    pc.indirect_mode   = 0u;
    pc.push_consts_bda = rt_upload_user_push_constants(d, commandBuffer, cs);
    pc.indirect_bda    = 0ULL;

    static uint32_t g_trace_log_count = 0;
    uint32_t trace_idx = g_trace_log_count++;
    if (trace_idx < 30 || (trace_idx % 120) == 0) {
        COMPAT_INFO("CmdTraceRays #%u: dim=%ux%ux%u rgen=%lx:sz%u:str%u miss=%lx:sz%u:str%u hit=%lx:sz%u:str%u call=%lx:sz%u:str%u pc_bda=%lx",
                    trace_idx, width, height, depth,
                    (unsigned long)pc.rgen_base, pc.rgen_size, pc.rgen_stride,
                    (unsigned long)pc.miss_base, pc.miss_size, pc.miss_stride,
                    (unsigned long)pc.hit_base, pc.hit_size, pc.hit_stride,
                    (unsigned long)pc.call_base, pc.call_size, pc.call_stride,
                    (unsigned long)pc.push_consts_bda);
    }

    if (d->fpCmdPipelineBarrier) {
        VkMemoryBarrier full_barrier = {
            .sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
            .srcAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT |
                             VK_ACCESS_HOST_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT |
                             VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR,
            .dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT |
                             VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT |
                             VK_ACCESS_UNIFORM_READ_BIT | VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR,
        };
        d->fpCmdPipelineBarrier(commandBuffer,
                                VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                                VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                                0,
                                1, &full_barrier,
                                0, NULL,
                                0, NULL);
    }

    /* local1d-1112: a direct trace with height < 4 runs 32 consecutive x per workgroup (see RT_PC_LOCAL1D) */
    const bool l1d = rt_env()->local1d && height < 4u;
    if (l1d) {
        pc.indirect_mode |= RT_PC_LOCAL1D;
        atomic_fetch_add_explicit(&g_local1d_traces, 1, memory_order_relaxed);
    }
    if (rp->split_n) {
        rt_split_dispatch(d, commandBuffer, cs, rp, &pc, width, height, depth);   /* split-1112 */
    } else {
        rt_prepare_and_bind_megakernel(d, commandBuffer, cs, rp, &pc);
        uint32_t gx = l1d ? (width + 31u) / 32u : (width + 7u) / 8u;
        uint32_t gy = l1d ? height : (height + 3u) / 4u;
        uint32_t gz = depth;
        d->fpCmdDispatch(commandBuffer, gx, gy, gz);
    }
    {   /* round 4 ray counter: launched megakernel invocations (primary dispatch), not ray-query traversals */
        const uint64_t rays = (uint64_t)width * height * depth;
        const unsigned c = rt_ray_class(rp->stage_count);
        atomic_fetch_add_explicit(&g_ray_traces, 1, memory_order_relaxed);
        atomic_fetch_add_explicit(&g_ray_rays, rays, memory_order_relaxed);
        atomic_fetch_add_explicit(&g_ray_cls_tr[c], 1, memory_order_relaxed);
        atomic_fetch_add_explicit(&g_ray_cls_rays[c], rays, memory_order_relaxed);
    }

    /* Post-trace memory barrier: ensure compute writes to storage buffers/images
     * are coherent for subsequent compute dispatches and graphics reads */
    if (d->fpCmdPipelineBarrier) {
        VkMemoryBarrier post_bar = {
            .sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
            .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_MEMORY_WRITE_BIT,
            .dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT |
                             VK_ACCESS_UNIFORM_READ_BIT | VK_ACCESS_MEMORY_READ_BIT,
        };
        d->fpCmdPipelineBarrier(commandBuffer,
                                VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                                VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                                0,
                                1, &post_bar,
                                0, NULL,
                                0, NULL);
    }

    rt_restore_post_trace_state(d, commandBuffer, cs, rp);
}

static void rt_ensure_indirect_setup_pipeline(struct DeviceData *d) {
    if (d->rt_ind_pipeline) return;
    pthread_mutex_lock(&d->bc_lock);
    if (d->rt_ind_pipeline) {
        pthread_mutex_unlock(&d->bc_lock);
        return;
    }
    size_t wc = 0;
    uint32_t *spv = rt_build_indirect_setup_spirv(&wc);
    if (spv && wc > 5) {
        VkPushConstantRange pcr = {
            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
            .offset     = 0,
            .size       = 16,
        };
        VkPipelineLayoutCreateInfo plci = {
            .sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
            .pushConstantRangeCount = 1,
            .pPushConstantRanges    = &pcr,
        };
        if (d->fpCreatePipelineLayout(d->device, &plci, NULL, &d->rt_ind_layout) == VK_SUCCESS) {
            VkShaderModuleCreateInfo smci = {
                .sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
                .codeSize = wc * sizeof(uint32_t),
                .pCode    = spv,
            };
            VkShaderModule mod = VK_NULL_HANDLE;
            if (d->fpCreateShaderModule(d->device, &smci, NULL, &mod) == VK_SUCCESS) {
                VkComputePipelineCreateInfo cpci = {
                    .sType  = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
                    .stage  = {
                        .sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                        .stage  = VK_SHADER_STAGE_COMPUTE_BIT,
                        .module = mod,
                        .pName  = "main",
                    },
                    .layout = d->rt_ind_layout,
                };
                d->fpCreateComputePipelines(d->device, VK_NULL_HANDLE, 1, &cpci, NULL, &d->rt_ind_pipeline);
                d->fpDestroyShaderModule(d->device, mod, NULL);
            }
        }
    }
    free(spv);
    pthread_mutex_unlock(&d->bc_lock);
}

static void rt_dispatch_indirect_common(
    VkCommandBuffer                             commandBuffer,
    const VkStridedDeviceAddressRegionKHR      *pRaygenShaderBindingTable,
    const VkStridedDeviceAddressRegionKHR      *pMissShaderBindingTable,
    const VkStridedDeviceAddressRegionKHR      *pHitShaderBindingTable,
    const VkStridedDeviceAddressRegionKHR      *pCallableShaderBindingTable,
    VkDeviceAddress                             indirectDeviceAddress,
    uint32_t                                    indirect_mode)
{
    rt_note_trace();
    struct DeviceData *d = get_device_data_by_cmd(commandBuffer);
    rt_report(d, false);   /* round 4 */
    if (!indirectDeviceAddress || rt_is_invalid_gpu_bda(indirectDeviceAddress)) {
        atomic_fetch_add_explicit(&g_ray_skip_zero, 1, memory_order_relaxed);
        return;
    }
    if (!d) { atomic_fetch_add_explicit(&g_ray_skip_nostate, 1, memory_order_relaxed); return; }
    struct RtCmdState *cs = rt_get_cmd_state(commandBuffer, false);
    if (!cs || !cs->bound_rt_pipeline) { atomic_fetch_add_explicit(&g_ray_skip_nostate, 1, memory_order_relaxed); return; }
    if (rt_forced_trace_skip()) return;
    if (!rt_pipe_ready(cs->bound_rt_pipeline)) { rt_count_skip(cs->bound_rt_pipeline); return; }
    struct RtPipeline *rp = cs->bound_rt_pipeline;

    struct RtDispatchPushConstants pc;
    memset(&pc, 0, sizeof(pc));
    if (pRaygenShaderBindingTable) {
        VkDeviceAddress addr = pRaygenShaderBindingTable->deviceAddress;
        if (rt_is_invalid_gpu_bda(addr) || addr == 0ULL) {
            pc.rgen_base   = rt_get_zero_bda(d);
            pc.rgen_stride = 0;
            pc.rgen_size   = 0;
        } else {
            pc.rgen_base   = addr;
            pc.rgen_stride = (uint32_t)pRaygenShaderBindingTable->stride;
            pc.rgen_size   = (uint32_t)pRaygenShaderBindingTable->size;
        }
    }
    if (pMissShaderBindingTable) {
        VkDeviceAddress addr = pMissShaderBindingTable->deviceAddress;
        if (rt_is_invalid_gpu_bda(addr) || addr == 0ULL) {
            pc.miss_base   = rt_get_zero_bda(d);
            pc.miss_stride = 0;
            pc.miss_size   = 0;
        } else {
            pc.miss_base   = addr;
            pc.miss_stride = (uint32_t)pMissShaderBindingTable->stride;
            pc.miss_size   = (uint32_t)pMissShaderBindingTable->size;
        }
    }
    if (pHitShaderBindingTable) {
        VkDeviceAddress addr = pHitShaderBindingTable->deviceAddress;
        if (rt_is_invalid_gpu_bda(addr) || addr == 0ULL) {
            pc.hit_base    = rt_get_zero_bda(d);
            pc.hit_stride  = 0;
            pc.hit_size    = 0;
        } else {
            pc.hit_base    = addr;
            pc.hit_stride  = (uint32_t)pHitShaderBindingTable->stride;
            pc.hit_size    = (uint32_t)pHitShaderBindingTable->size;
        }
    }
    if (pCallableShaderBindingTable) {
        VkDeviceAddress addr = pCallableShaderBindingTable->deviceAddress;
        if (rt_is_invalid_gpu_bda(addr) || addr == 0ULL) {
            pc.call_base   = rt_get_zero_bda(d);
            pc.call_stride = 0;
            pc.call_size   = 0;
        } else {
            pc.call_base   = addr;
            pc.call_stride = (uint32_t)pCallableShaderBindingTable->stride;
            pc.call_size   = (uint32_t)pCallableShaderBindingTable->size;
        }
    }
    pc.indirect_mode   = indirect_mode;
    pc.push_consts_bda = rt_upload_user_push_constants(d, commandBuffer, cs);
    pc.indirect_bda    = indirectDeviceAddress;

    static uint32_t g_ind_trace_log_count = 0;
    if (g_ind_trace_log_count < 20 || (g_ind_trace_log_count % 300) == 0) {
        g_ind_trace_log_count++;
        COMPAT_INFO("CmdTraceRaysIndirect (mode %u) #%u: ind_bda=%lx rgen=%lx:sz%u miss=%lx:sz%u hit=%lx:sz%u pc_bda=%lx",
                    indirect_mode, g_ind_trace_log_count,
                    (unsigned long)indirectDeviceAddress,
                    (unsigned long)pc.rgen_base, pc.rgen_size,
                    (unsigned long)pc.miss_base, pc.miss_size,
                    (unsigned long)pc.hit_base, pc.hit_size,
                    (unsigned long)pc.push_consts_bda);
    }

    rt_ensure_indirect_setup_pipeline(d);

    uint32_t ind_off = 0;
    void *ind_ptr = NULL;
    VkDeviceAddress dst_bda = rt_ring_alloc(d, 64, 64, &ind_ptr, &ind_off);
    if (d->fpCmdPipelineBarrier) {
        VkMemoryBarrier full_barrier = {
            .sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
            .srcAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT |
                             VK_ACCESS_HOST_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT |
                             VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR,
            .dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT |
                             VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT |
                             VK_ACCESS_UNIFORM_READ_BIT | VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR,
        };
        d->fpCmdPipelineBarrier(commandBuffer,
                                VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                                VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                                0,
                                1, &full_barrier,
                                0, NULL,
                                0, NULL);
    }

    if (rp->split_n) {
        rt_split_dispatch(d, commandBuffer, cs, rp, &pc, 0u, 0u, 0u);   /* split-1112: reads the command on the GPU */
    } else if (d->rt_ind_pipeline && d->fpCmdDispatchIndirect && dst_bda && d->rt_ring_buf) {
        uint64_t src_bda = indirectDeviceAddress + (indirect_mode == 2 ? 88ULL : 0ULL);
        uint64_t setup_pc[2] = { src_bda, dst_bda };

        d->fpCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, d->rt_ind_pipeline);
        d->fpCmdPushConstants(commandBuffer, d->rt_ind_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, 16, setup_pc);
        d->fpCmdDispatch(commandBuffer, 1, 1, 1);

        VkMemoryBarrier mb_bar = {
            .sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
            .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
            .dstAccessMask = VK_ACCESS_INDIRECT_COMMAND_READ_BIT,
        };
        d->fpCmdPipelineBarrier(commandBuffer,
                                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT,
                                0, 1, &mb_bar, 0, NULL, 0, NULL);

        rt_prepare_and_bind_megakernel(d, commandBuffer, cs, rp, &pc);
        d->fpCmdDispatchIndirect(commandBuffer, d->rt_ring_buf, (VkDeviceSize)ind_off);
    } else {
        rt_prepare_and_bind_megakernel(d, commandBuffer, cs, rp, &pc);
        d->fpCmdDispatch(commandBuffer, 256, 256, 1);
    }

    if (d->fpCmdPipelineBarrier) {
        VkMemoryBarrier post_bar = {
            .sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
            .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_MEMORY_WRITE_BIT,
            .dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT |
                             VK_ACCESS_UNIFORM_READ_BIT | VK_ACCESS_MEMORY_READ_BIT,
        };
        d->fpCmdPipelineBarrier(commandBuffer,
                                VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                                VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                                0, 1, &post_bar, 0, NULL, 0, NULL);
    }
    atomic_fetch_add_explicit(&g_ray_ind, 1, memory_order_relaxed);   /* dims are GPU-side: not in rays */

    rt_restore_post_trace_state(d, commandBuffer, cs, rp);
}

static VKAPI_ATTR void VKAPI_CALL
compat_CmdTraceRaysIndirectKHR(
    VkCommandBuffer                             commandBuffer,
    const VkStridedDeviceAddressRegionKHR      *pRaygenShaderBindingTable,
    const VkStridedDeviceAddressRegionKHR      *pMissShaderBindingTable,
    const VkStridedDeviceAddressRegionKHR      *pHitShaderBindingTable,
    const VkStridedDeviceAddressRegionKHR      *pCallableShaderBindingTable,
    VkDeviceAddress                             indirectDeviceAddress)
{
    rt_dispatch_indirect_common(commandBuffer, pRaygenShaderBindingTable,
                                pMissShaderBindingTable, pHitShaderBindingTable,
                                pCallableShaderBindingTable, indirectDeviceAddress, 1u);
}

static VKAPI_ATTR void VKAPI_CALL
compat_CmdTraceRaysIndirect2KHR(
    VkCommandBuffer                             commandBuffer,
    VkDeviceAddress                             indirectDeviceAddress)
{
    rt_dispatch_indirect_common(commandBuffer, NULL, NULL, NULL, NULL,
                                indirectDeviceAddress, 2u);
}

/* ── Deferred Host Operations (Synchronous Host Fallback) ─────────────────── */

static VKAPI_ATTR VkResult VKAPI_CALL
compat_CreateDeferredHostOperationsKHR(
    VkDevice                                    device,
    const VkAllocationCallbacks                *pAllocator,
    VkDeferredOperationKHR                     *pDeferredOperation)
{
    (void)pAllocator;
    if (!pDeferredOperation) return VK_ERROR_INITIALIZATION_FAILED;
    struct RtDeferredOp *op = (struct RtDeferredOp *)calloc(1, sizeof(*op));
    if (!op) return VK_ERROR_OUT_OF_HOST_MEMORY;
    op->magic  = 0x5254444Fu; /* 'RTDO' */
    op->device = device;
    op->result = VK_SUCCESS;
    VkDeferredOperationKHR handle = (VkDeferredOperationKHR)(uintptr_t)op;
    unsigned b = rt_bucket((uintptr_t)handle);
    pthread_mutex_lock(&g_rt_lock);
    op->next = g_rt_deferred_ops[b];
    g_rt_deferred_ops[b] = op;
    pthread_mutex_unlock(&g_rt_lock);
    *pDeferredOperation = handle;
    return VK_SUCCESS;
}

static VKAPI_ATTR void VKAPI_CALL
compat_DestroyDeferredHostOperationKHR(
    VkDevice                                    device,
    VkDeferredOperationKHR                      operation,
    const VkAllocationCallbacks                *pAllocator)
{
    (void)device; (void)pAllocator;
    if (!operation) return;
    unsigned b = rt_bucket((uintptr_t)operation);
    pthread_mutex_lock(&g_rt_lock);
    struct RtDeferredOp **pp = &g_rt_deferred_ops[b];
    while (*pp) {
        if ((VkDeferredOperationKHR)(uintptr_t)(*pp) == operation) {
            struct RtDeferredOp *victim = *pp;
            *pp = victim->next;
            pthread_mutex_unlock(&g_rt_lock);
            free(victim);
            return;
        }
        pp = &(*pp)->next;
    }
    pthread_mutex_unlock(&g_rt_lock);
}

static VKAPI_ATTR uint32_t VKAPI_CALL
compat_GetDeferredHostOperationMaxConcurrencyKHR(
    VkDevice                                    device,
    VkDeferredOperationKHR                      operation)
{
    (void)device; (void)operation;
    return 1u;
}

static VKAPI_ATTR VkResult VKAPI_CALL
compat_GetDeferredHostOperationResultKHR(
    VkDevice                                    device,
    VkDeferredOperationKHR                      operation)
{
    (void)device; (void)operation;
    return VK_SUCCESS;
}

static VKAPI_ATTR VkResult VKAPI_CALL
compat_DeferredHostOperationJoinKHR(
    VkDevice                                    device,
    VkDeferredOperationKHR                      operation)
{
    (void)device; (void)operation;
    return VK_SUCCESS;
}

/* ── FEX Thunk Function Pointer Patching ─────────────────────────────────── */

static VKAPI_ATTR VkResult VKAPI_CALL compat_CreateAccelerationStructureKHR(VkDevice, const VkAccelerationStructureCreateInfoKHR*, const VkAllocationCallbacks*, VkAccelerationStructureKHR*);
static VKAPI_ATTR void VKAPI_CALL compat_DestroyAccelerationStructureKHR(VkDevice, VkAccelerationStructureKHR, const VkAllocationCallbacks*);
static VKAPI_ATTR void VKAPI_CALL compat_CmdBuildAccelerationStructuresKHR(VkCommandBuffer, uint32_t, const VkAccelerationStructureBuildGeometryInfoKHR*, const VkAccelerationStructureBuildRangeInfoKHR* const*);
static VKAPI_ATTR void VKAPI_CALL compat_CmdCopyAccelerationStructureKHR(VkCommandBuffer, const VkCopyAccelerationStructureInfoKHR*);
static VKAPI_ATTR void VKAPI_CALL compat_CmdWriteAccelerationStructuresPropertiesKHR(VkCommandBuffer, uint32_t, const VkAccelerationStructureKHR*, VkQueryType, VkQueryPool, uint32_t);
static VKAPI_ATTR VkDeviceAddress VKAPI_CALL compat_GetAccelerationStructureDeviceAddressKHR(VkDevice, const VkAccelerationStructureDeviceAddressInfoKHR*);
static VKAPI_ATTR void VKAPI_CALL compat_GetAccelerationStructureBuildSizesKHR(VkDevice, VkAccelerationStructureBuildTypeKHR, const VkAccelerationStructureBuildGeometryInfoKHR*, const uint32_t*, VkAccelerationStructureBuildSizesInfoKHR*);

struct FexThunkMapping {
    const char *sym_name;
    uintptr_t fallback_offset;
    void *func_ptr;
};

struct FexThunkFinder {
    uintptr_t base_addr;
    char path[512];
    bool found;
};

static int rt_fex_phdr_cb(struct dl_phdr_info *info, size_t size, void *data) {
    (void)size;
    struct FexThunkFinder *finder = (struct FexThunkFinder *)data;
    if (info->dlpi_name && strstr(info->dlpi_name, "libvulkan-host.so")) {
        finder->base_addr = (uintptr_t)info->dlpi_addr;
        strncpy(finder->path, info->dlpi_name, sizeof(finder->path) - 1);
        finder->found = true;
        return 1;
    }
    return 0;
}

static void rt_patch_fex_thunks(void) {
    /* rt-gate-1112: only for the RT emulation. Without MALI_COMPAT_RT_PIPELINE the FEX thunk table keeps what FEX
     * resolved (MALI_COMPAT_AS_FIX alone reaches the AS hooks through vkGetDeviceProcAddr, as every layer hook). */
    if (!g_rt_pipeline) return;
    static _Atomic int s_patched = 0;
    if (atomic_exchange(&s_patched, 1) != 0) return;

    struct FexThunkFinder finder = {0};
    dl_iterate_phdr(rt_fex_phdr_cb, &finder);

    if (!finder.found) {
        FILE *fp = fopen("/proc/self/maps", "r");
        if (fp) {
            char line[1024];
            while (fgets(line, sizeof(line), fp)) {
                if (strstr(line, "libvulkan-host.so")) {
                    uintptr_t start = 0;
                    if (sscanf(line, "%lx-", &start) == 1) {
                        finder.base_addr = start;
                        char *slash = strchr(line, '/');
                        if (slash) {
                            char *nl = strchr(slash, '\n');
                            if (nl) *nl = '\0';
                            strncpy(finder.path, slash, sizeof(finder.path) - 1);
                        }
                        finder.found = true;
                        break;
                    }
                }
            }
            fclose(fp);
        }
    }

    if (!finder.found) {
        COMPAT_INFO("FEX thunk patch: libvulkan-host.so not found in process (native run)");
        return;
    }

    COMPAT_INFO("FEX thunk patch: detected libvulkan-host.so at base %p (path: %s)",
                (void *)finder.base_addr, finder.path);

    struct FexThunkMapping targets[] = {
        { "fexldr_ptr_libvulkan_vkGetRayTracingShaderGroupHandlesKHR", 0xbb098, (void *)compat_GetRayTracingShaderGroupHandlesKHR },
        { "fexldr_ptr_libvulkan_vkGetRayTracingShaderGroupHandlesNV",  0xbb0a0, (void *)compat_GetRayTracingShaderGroupHandlesKHR },
        { "fexldr_ptr_libvulkan_vkCmdTraceRaysKHR",                    0xbb8d0, (void *)compat_CmdTraceRaysKHR },
        { "fexldr_ptr_libvulkan_vkCreateRayTracingPipelinesKHR",        0xbb8d8, (void *)compat_CreateRayTracingPipelinesKHR },
        { "fexldr_ptr_libvulkan_vkGetRayTracingCaptureReplayShaderGroupHandlesKHR", 0xbb8e0, (void *)compat_GetRayTracingCaptureReplayShaderGroupHandlesKHR },
        { "fexldr_ptr_libvulkan_vkCmdTraceRaysIndirectKHR",            0xbb8e8, (void *)compat_CmdTraceRaysIndirectKHR },
        { "fexldr_ptr_libvulkan_vkCmdTraceRaysIndirect2KHR",           0xbacc0, (void *)compat_CmdTraceRaysIndirect2KHR },
        { "fexldr_ptr_libvulkan_vkGetRayTracingShaderGroupStackSizeKHR", 0xbb8f0, (void *)compat_GetRayTracingShaderGroupStackSizeKHR },
        { "fexldr_ptr_libvulkan_vkCmdSetRayTracingPipelineStackSizeKHR", 0xbb8f8, (void *)compat_CmdSetRayTracingPipelineStackSizeKHR },
        { "fexldr_ptr_libvulkan_vkCreateAccelerationStructureKHR",     0, (void *)compat_CreateAccelerationStructureKHR },
        { "fexldr_ptr_libvulkan_vkDestroyAccelerationStructureKHR",    0, (void *)compat_DestroyAccelerationStructureKHR },
        { "fexldr_ptr_libvulkan_vkCmdBuildAccelerationStructuresKHR",  0, (void *)compat_CmdBuildAccelerationStructuresKHR },
        { "fexldr_ptr_libvulkan_vkCmdCopyAccelerationStructureKHR",    0, (void *)compat_CmdCopyAccelerationStructureKHR },
        { "fexldr_ptr_libvulkan_vkCmdWriteAccelerationStructuresPropertiesKHR", 0, (void *)compat_CmdWriteAccelerationStructuresPropertiesKHR },
        { "fexldr_ptr_libvulkan_vkGetAccelerationStructureDeviceAddressKHR", 0, (void *)compat_GetAccelerationStructureDeviceAddressKHR },
        { "fexldr_ptr_libvulkan_vkGetAccelerationStructureBuildSizesKHR",    0, (void *)compat_GetAccelerationStructureBuildSizesKHR },
    };
    size_t num_targets = sizeof(targets) / sizeof(targets[0]);

    const char *open_path = finder.path[0] ? finder.path : "/usr/local/lib/fex-emu/HostThunks/libvulkan-host.so";
    int fd = open(open_path, O_RDONLY);
    if (fd >= 0) {
        struct stat st;
        if (fstat(fd, &st) == 0 && st.st_size > (off_t)sizeof(Elf64_Ehdr)) {
            void *map = mmap(NULL, st.st_size, PROT_READ, MAP_SHARED, fd, 0);
            if (map != MAP_FAILED) {
                Elf64_Ehdr *ehdr = (Elf64_Ehdr *)map;
                if (ehdr->e_ident[EI_MAG0] == ELFMAG0 && ehdr->e_ident[EI_MAG1] == ELFMAG1 &&
                    ehdr->e_ident[EI_MAG2] == ELFMAG2 && ehdr->e_ident[EI_MAG3] == ELFMAG3) {
                    Elf64_Shdr *shdrs = (Elf64_Shdr *)((uint8_t *)map + ehdr->e_shoff);
                    Elf64_Shdr *symtab = NULL, *strtab = NULL;
                    for (int i = 0; i < ehdr->e_shnum; i++) {
                        if (shdrs[i].sh_type == SHT_SYMTAB && shdrs[i].sh_link < ehdr->e_shnum) {
                            symtab = &shdrs[i];
                            strtab = &shdrs[shdrs[i].sh_link];
                            break;
                        }
                    }
                    if (symtab && strtab) {
                        const char *strpool = (const char *)map + strtab->sh_offset;
                        Elf64_Sym *syms = (Elf64_Sym *)((uint8_t *)map + symtab->sh_offset);
                        int count = symtab->sh_size / sizeof(Elf64_Sym);
                        for (int i = 0; i < count; i++) {
                            const char *sym_name = strpool + syms[i].st_name;
                            for (size_t t = 0; t < num_targets; t++) {
                                if (strstr(sym_name, targets[t].sym_name)) {
                                    targets[t].fallback_offset = syms[i].st_value;
                                    break;
                                }
                            }
                        }
                        COMPAT_INFO("FEX thunk patch: resolved symbol offsets dynamically from %s", open_path);
                    }
                }
                munmap(map, st.st_size);
            }
        }
        close(fd);
    }

    for (size_t t = 0; t < num_targets; t++) {
        if (!targets[t].fallback_offset) continue;
        void **slot = (void **)(finder.base_addr + targets[t].fallback_offset);
        void *old_val = *slot;
        *slot = targets[t].func_ptr;
        COMPAT_INFO("FEX thunk patched: %s @ %p [offset 0x%lx] (%p -> %p)",
                    targets[t].sym_name, (void *)slot, (unsigned long)targets[t].fallback_offset, old_val, targets[t].func_ptr);
    }
}

/* ── Layer Integration Wrappers ───────────────────────────────────────────── */

static void rt_reset_cmd_state(VkCommandBuffer cmd, bool remove) {
    if (!cmd) return;
    unsigned b = rt_bucket((uintptr_t)cmd);
    pthread_mutex_lock(&g_rt_lock);
    struct RtCmdState **pp = &g_rt_cmd_states[b];
    while (*pp) {
        if ((*pp)->cmd == cmd) {
            struct RtCmdState *s = *pp;
            if (remove) {
                *pp = s->next;
                pthread_mutex_unlock(&g_rt_lock);
                rt_acct_sub(&g_acct_cmdst, (int64_t)sizeof(*s));
                free(s);
                return;
            }
            struct RtCmdState *next = s->next;
            memset(s, 0, sizeof(*s));
            s->cmd  = cmd;
            s->next = next;
            pthread_mutex_unlock(&g_rt_lock);
            return;
        }
        pp = &(*pp)->next;
    }
    pthread_mutex_unlock(&g_rt_lock);
}

static bool rt_cmd_bind_pipeline(VkCommandBuffer commandBuffer,
                                 VkPipelineBindPoint pipelineBindPoint,
                                 VkPipeline pipeline) {
    if (pipelineBindPoint == VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR) {
        struct RtCmdState *cs = rt_get_cmd_state(commandBuffer, true);
        struct RtPipeline *rp = rt_lookup_pipeline(pipeline);
        if (cs) cs->bound_rt_pipeline = rp;
        if (rp && rt_mega_ready(rp)) {
            struct DeviceData *d = get_device_data_by_cmd(commandBuffer);
            if (d && d->fpCmdBindPipeline) {
                d->fpCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, rp->compute_pipeline);
            }
        }
        return true;
    }
    if (pipelineBindPoint == VK_PIPELINE_BIND_POINT_COMPUTE) {
        struct RtCmdState *cs = rt_get_cmd_state(commandBuffer, true);
        if (cs) cs->bound_comp_pipeline = pipeline;
    }
    return false;
}

static void rt_cmd_push_constants(VkCommandBuffer commandBuffer,
                                  VkPipelineLayout layout,
                                  VkShaderStageFlags *inout_stages,
                                  uint32_t offset,
                                  uint32_t size,
                                  const void *pValues) {
    (void)layout;
    if (pValues && offset < 256) {
        struct RtCmdState *cs = rt_get_cmd_state(commandBuffer, true);
        if (cs) {
            uint32_t copy_len = (offset + size <= 256) ? size : (256 - offset);
            memcpy(cs->user_push_constants + offset, pValues, copy_len);
            if (offset + copy_len > cs->user_pc_dirty_size) {
                cs->user_pc_dirty_size = offset + copy_len;
            }
        }
    }
    if (inout_stages && (*inout_stages & RT_SHADER_STAGE_MASK)) {
        *inout_stages = (*inout_stages & ~RT_SHADER_STAGE_MASK) | VK_SHADER_STAGE_COMPUTE_BIT;
    }
}

static VKAPI_ATTR void VKAPI_CALL
compat_DestroyShaderModule(
    VkDevice                                    device,
    VkShaderModule                              shaderModule,
    const VkAllocationCallbacks                *pAllocator)
{
    if (g_rt_pipeline && rt_destroy_shader_module(shaderModule)) return;
    struct DeviceData *d = get_device_data(device);
    if (d && d->fpDestroyShaderModule) {
        d->fpDestroyShaderModule(device, shaderModule, pAllocator);
    }
}

static VKAPI_ATTR VkResult VKAPI_CALL
compat_RtCreateDescriptorSetLayout(
    VkDevice                                    device,
    const VkDescriptorSetLayoutCreateInfo      *pCreateInfo,
    const VkAllocationCallbacks                *pAllocator,
    VkDescriptorSetLayout                      *pSetLayout)
{
    struct DeviceData *d = get_device_data(device);
    if (!d || !d->fpCreateDescriptorSetLayout || !pCreateInfo) return VK_ERROR_INITIALIZATION_FAILED;

    VkDescriptorSetLayoutCreateInfo mod_ci = *pCreateInfo;
    VkDescriptorSetLayoutBinding stack_bindings[64];
    VkDescriptorSetLayoutBinding *bindings = NULL;

    bool has_rt_stages = false;
    for (uint32_t i = 0; i < pCreateInfo->bindingCount; i++) {
        if (pCreateInfo->pBindings[i].stageFlags & RT_SHADER_STAGE_MASK) {
            has_rt_stages = true;
            break;
        }
    }

    if (has_rt_stages && pCreateInfo->bindingCount > 0) {
        bindings = (pCreateInfo->bindingCount <= 64)
            ? stack_bindings
            : (VkDescriptorSetLayoutBinding *)malloc(pCreateInfo->bindingCount * sizeof(*bindings));
        if (bindings) {
            for (uint32_t i = 0; i < pCreateInfo->bindingCount; i++) {
                bindings[i] = pCreateInfo->pBindings[i];
                if (bindings[i].stageFlags & RT_SHADER_STAGE_MASK) {
                    bindings[i].stageFlags = (bindings[i].stageFlags & ~RT_SHADER_STAGE_MASK) |
                                             VK_SHADER_STAGE_COMPUTE_BIT;
                }
            }
            mod_ci.pBindings = bindings;
        }
    }

    VkResult res = d->fpCreateDescriptorSetLayout(device, &mod_ci, pAllocator, pSetLayout);
    if (bindings && bindings != stack_bindings) free(bindings);
    return res;
}

static VKAPI_ATTR VkResult VKAPI_CALL
compat_RtCreatePipelineLayout(
    VkDevice                                    device,
    const VkPipelineLayoutCreateInfo           *pCreateInfo,
    const VkAllocationCallbacks                *pAllocator,
    VkPipelineLayout                           *pPipelineLayout)
{
    struct DeviceData *d = get_device_data(device);
    if (!d || !d->fpCreatePipelineLayout || !pCreateInfo) return VK_ERROR_INITIALIZATION_FAILED;

    VkPipelineLayoutCreateInfo mod_ci = *pCreateInfo;
    VkPushConstantRange ranges[16];
    uint32_t rc = 0;
    VkShaderStageFlags combined_stages = VK_SHADER_STAGE_COMPUTE_BIT;
    uint32_t max_end = sizeof(struct RtDispatchPushConstants); /* 96 bytes minimum */
    if (max_end < 128) max_end = 128;

    for (uint32_t i = 0; i < pCreateInfo->pushConstantRangeCount && i < 15; i++) {
        VkShaderStageFlags st = pCreateInfo->pPushConstantRanges[i].stageFlags;
        if (st & RT_SHADER_STAGE_MASK) {
            st = (st & ~RT_SHADER_STAGE_MASK) | VK_SHADER_STAGE_COMPUTE_BIT;
        }
        combined_stages |= st;
        uint32_t end = pCreateInfo->pPushConstantRanges[i].offset + pCreateInfo->pPushConstantRanges[i].size;
        if (end > max_end && end <= 256) max_end = end;
    }

    /* Collapse into a single unified push constant range covering offset 0..max_end for combined_stages */
    ranges[0].stageFlags = combined_stages;
    ranges[0].offset     = 0;
    ranges[0].size       = max_end;
    rc = 1;

    mod_ci.pushConstantRangeCount = rc;
    mod_ci.pPushConstantRanges    = ranges;

    VkResult res = d->fpCreatePipelineLayout(device, &mod_ci, pAllocator, pPipelineLayout);
    if (res != VK_SUCCESS && pCreateInfo->pushConstantRangeCount > 0) {
        /* Fallback: keep original ranges with RT stage flags translated to COMPUTE_BIT */
        rc = pCreateInfo->pushConstantRangeCount <= 16 ? pCreateInfo->pushConstantRangeCount : 16;
        for (uint32_t i = 0; i < rc; i++) {
            ranges[i] = pCreateInfo->pPushConstantRanges[i];
            if (ranges[i].stageFlags & RT_SHADER_STAGE_MASK) {
                ranges[i].stageFlags = (ranges[i].stageFlags & ~RT_SHADER_STAGE_MASK) | VK_SHADER_STAGE_COMPUTE_BIT;
            }
            if (ranges[i].offset == 0 && ranges[i].size < sizeof(struct RtDispatchPushConstants)) {
                ranges[i].size = sizeof(struct RtDispatchPushConstants);
            }
            ranges[i].stageFlags |= VK_SHADER_STAGE_COMPUTE_BIT;
        }
        mod_ci.pushConstantRangeCount = rc;
        mod_ci.pPushConstantRanges    = ranges;
        res = d->fpCreatePipelineLayout(device, &mod_ci, pAllocator, pPipelineLayout);
    }
    if (res == VK_SUCCESS && pPipelineLayout && *pPipelineLayout) {
        rt_record_layout_pc_stages(*pPipelineLayout, combined_stages);
    }
    return res;
}

static VKAPI_ATTR void VKAPI_CALL
compat_RtCmdBindDescriptorSets(
    VkCommandBuffer                             commandBuffer,
    VkPipelineBindPoint                         pipelineBindPoint,
    VkPipelineLayout                            layout,
    uint32_t                                    firstSet,
    uint32_t                                    descriptorSetCount,
    const VkDescriptorSet                      *pDescriptorSets,
    uint32_t                                    dynamicOffsetCount,
    const uint32_t                             *pDynamicOffsets)
{
    struct DeviceData *d = get_device_data_by_cmd(commandBuffer);
    if (!d || !d->fpCmdBindDescriptorSets) return;
    if (pipelineBindPoint == VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR) {
        pipelineBindPoint = VK_PIPELINE_BIND_POINT_COMPUTE;
    }
    d->fpCmdBindDescriptorSets(commandBuffer, pipelineBindPoint, layout,
                               firstSet, descriptorSetCount, pDescriptorSets,
                               dynamicOffsetCount, pDynamicOffsets);
}

static VKAPI_ATTR void VKAPI_CALL
compat_RtCmdSetDescriptorBufferOffsetsEXT(
    VkCommandBuffer                             commandBuffer,
    VkPipelineBindPoint                         pipelineBindPoint,
    VkPipelineLayout                            layout,
    uint32_t                                    firstSet,
    uint32_t                                    setCount,
    const uint32_t                             *pBufferIndices,
    const VkDeviceSize                         *pOffsets)
{
    struct DeviceData *d = get_device_data_by_cmd(commandBuffer);
    if (!d || !d->fpCmdSetDescriptorBufferOffsetsEXT) return;
    if (pipelineBindPoint == VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR) {
        struct RtCmdState *cs = rt_get_cmd_state(commandBuffer, true);
        if (cs && pBufferIndices && pOffsets) {
            cs->has_rt_desc_buf_offsets = true;
            cs->rt_db_layout            = layout;
            cs->rt_db_first_set         = firstSet;
            cs->rt_db_set_count         = setCount <= 8 ? setCount : 8;
            memcpy(cs->rt_db_buf_indices, pBufferIndices, cs->rt_db_set_count * sizeof(uint32_t));
            memcpy(cs->rt_db_offsets, pOffsets, cs->rt_db_set_count * sizeof(VkDeviceSize));
        }
        pipelineBindPoint = VK_PIPELINE_BIND_POINT_COMPUTE;
    }
    d->fpCmdSetDescriptorBufferOffsetsEXT(commandBuffer, pipelineBindPoint, layout,
                                          firstSet, setCount, pBufferIndices, pOffsets);
}

/* ── Acceleration Structure Lifetime & Mutation Tracking ─────────────────── */

struct RtAsTrack {
    VkAccelerationStructureKHR orig_handle;
    VkAccelerationStructureKHR active_handle;
    VkDevice                   device;
    VkBuffer                   buffer;
    VkDeviceSize               offset;
    VkDeviceSize               size;
    VkAccelerationStructureTypeKHR created_type;
    VkAccelerationStructureTypeKHR built_type;
    VkDeviceSize               buf_rest;      /* 1110b, MALI_COMPAT_DEBUG only: buffer memreq size - offset (0 = unknown) */
    /* 1110b: driver handles replaced on a type change (layer-created ones, NULL allocator). Destroying one at
     * replacement time is not safe (a submitted command may still use it); the app's vkDestroyAccelerationStructure
     * guarantees every command using this AS completed, so they are destroyed there. */
    VkAccelerationStructureKHR *retired;
    uint32_t                   n_retired, cap_retired;
    /* lead-1110c */
    VkDeviceSize               as_size;       /* driver's accelerationStructureSize of the last BUILD into it (or the
                                               * source's after a copy); 0 = unknown -> a CLONE from it goes to the driver */
    VkAccelerationStructureTypeKHR active_type; /* creation type of active_handle (GENERIC or explicit) */
    bool                       xfer_ok;       /* its VkBuffer was created through compat_CreateBuffer with TRANSFER_SRC+DST */
    bool                       refit_ready;   /* false after a copy into it until a BUILD (MALI_COMPAT_RT_CLONE_REFIT) */
    /* round 4 */
    VkDeviceSize               build_size;    /* driver's accelerationStructureSize of the last BUILD of the content it
                                               * holds (propagated by every copy, also COMPACT): what a re-BUILD of this
                                               * content needs. as_size is the byte extent a CLONE copies (0 after a
                                               * COMPACT: the compacted size is not known to the layer). Codex r3 #1/#2 */
    int8_t                     acct_cls;      /* class counted in g_acct_as[] (-1 = not counted) */
    VkDeviceSize               acct_drv;      /* build_size counted in g_acct_as_drv[acct_cls] */
    size_t                     acct_host;     /* host bytes counted in g_acct_tracks */
    struct RtAsTrack          *next;
};
/* round 4: (re)counts a track in the AS classes after its built type or build size changed. Caller holds g_rt_as_lock
 * (or owns the unlinked track). gone=true: removes it entirely (destroy). */
static void rt_as_acct_locked(struct RtAsTrack *t, bool gone) {
    const int ncls = gone ? -1 : rt_as_cls(t->built_type);
    const VkDeviceSize ndrv = gone ? 0 : t->build_size;
    if (t->acct_cls == ncls && t->acct_drv == ndrv) return;
    if (t->acct_cls >= 0) {
        rt_acct_sub(&g_acct_as[t->acct_cls], (int64_t)t->size);
        if (t->acct_drv) { rt_acct_sub(&g_acct_as_drv[t->acct_cls], (int64_t)t->acct_drv);
                           rt_acct_sub(&g_acct_as_drv_all, (int64_t)t->acct_drv); }
    }
    if (ncls >= 0) {
        rt_acct_add(&g_acct_as[ncls], (int64_t)t->size);
        if (ndrv) { rt_acct_add(&g_acct_as_drv[ncls], (int64_t)ndrv); rt_acct_add(&g_acct_as_drv_all, (int64_t)ndrv); }
    }
    if (t->acct_cls < 0 && ncls >= 0) rt_acct_add(&g_acct_as_all, (int64_t)t->size);
    else if (t->acct_cls >= 0 && ncls < 0) rt_acct_sub(&g_acct_as_all, (int64_t)t->size);
    t->acct_cls = (int8_t)ncls;
    t->acct_drv = ndrv;
}
static void rt_as_acct_host(struct RtAsTrack *t) {   /* track struct + retired-handle array */
    size_t now = sizeof(*t) + (size_t)t->cap_retired * sizeof(VkAccelerationStructureKHR);
    if (!t->acct_host) rt_acct_add(&g_acct_tracks, (int64_t)now);
    else rt_acct_adj(&g_acct_tracks, (int64_t)now - (int64_t)t->acct_host);
    t->acct_host = now;
}

/* lead-1110c: VkBuffers created through compat_CreateBuffer with ACCELERATION_STRUCTURE_STORAGE usage, to which the
 * layer added TRANSFER_SRC|TRANSFER_DST (needed for the layer's vkCmdCopyBuffer CLONE). Open-addressing set,
 * removed in compat_RtDestroyBuffer. An AS track records at CreateAS whether its buffer is in here (xfer_ok).
 * round 4: a map now: EVERY AS-storage buffer is recorded with its created size (g_acct_asbuf) and whether it has both
 * transfer bits (xfer); rt_xfer_buf_has() answers only for xfer entries. */
struct RtXbuf { uint64_t key; uint64_t size; bool xfer; };
static pthread_mutex_t g_rt_xbuf_lock = PTHREAD_MUTEX_INITIALIZER;
static struct RtXbuf  *g_rt_xbuf;
static size_t          g_rt_xbuf_cap, g_rt_xbuf_n;
static inline size_t rt_xbuf_slot(uint64_t k, size_t cap) {
    k ^= k >> 33; k *= 0xff51afd7ed558ccdULL; k ^= k >> 33;
    return (size_t)k & (cap - 1);
}
static void rt_as_buf_add(VkBuffer b, VkDeviceSize size, bool xfer) {
    uint64_t k = (uint64_t)(uintptr_t)b;
    if (!k) return;
    pthread_mutex_lock(&g_rt_xbuf_lock);
    if ((g_rt_xbuf_n + 1) * 2 > g_rt_xbuf_cap) {
        size_t nc = g_rt_xbuf_cap ? g_rt_xbuf_cap * 2 : 1024;
        struct RtXbuf *nt = (struct RtXbuf *)calloc(nc, sizeof(*nt));
        if (!nt) {   /* not recorded: a CLONE through this buffer falls back to the driver (reason=usage) */
            pthread_mutex_unlock(&g_rt_xbuf_lock);
            return;
        }
        for (size_t i = 0; i < g_rt_xbuf_cap; i++) {
            if (!g_rt_xbuf[i].key) continue;
            size_t j = rt_xbuf_slot(g_rt_xbuf[i].key, nc);
            while (nt[j].key) j = (j + 1) & (nc - 1);
            nt[j] = g_rt_xbuf[i];
        }
        free(g_rt_xbuf);
        rt_acct_adj(&g_acct_bufmap, (int64_t)((nc - g_rt_xbuf_cap) * sizeof(*nt)));
        g_rt_xbuf = nt;
        g_rt_xbuf_cap = nc;
    }
    size_t i = rt_xbuf_slot(k, g_rt_xbuf_cap);
    while (g_rt_xbuf[i].key && g_rt_xbuf[i].key != k) i = (i + 1) & (g_rt_xbuf_cap - 1);
    if (g_rt_xbuf[i].key == k) {   /* a stale entry of a reused handle (destroy not seen): replace */
        rt_acct_sub(&g_acct_asbuf, (int64_t)g_rt_xbuf[i].size);
    } else {
        g_rt_xbuf_n++;
    }
    g_rt_xbuf[i] = (struct RtXbuf){ k, size, xfer };
    rt_acct_add(&g_acct_asbuf, (int64_t)size);
    pthread_mutex_unlock(&g_rt_xbuf_lock);
}
static bool rt_xfer_buf_has(VkBuffer b) {
    uint64_t k = (uint64_t)(uintptr_t)b;
    bool found = false;
    if (!k) return false;
    pthread_mutex_lock(&g_rt_xbuf_lock);
    if (g_rt_xbuf_cap) {
        size_t i = rt_xbuf_slot(k, g_rt_xbuf_cap);
        while (g_rt_xbuf[i].key && g_rt_xbuf[i].key != k) i = (i + 1) & (g_rt_xbuf_cap - 1);
        found = g_rt_xbuf[i].key == k && g_rt_xbuf[i].xfer;
    }
    pthread_mutex_unlock(&g_rt_xbuf_lock);
    return found;
}
static void rt_xfer_buf_remove(VkBuffer b) {
    uint64_t k = (uint64_t)(uintptr_t)b;
    if (!k) return;
    pthread_mutex_lock(&g_rt_xbuf_lock);
    if (g_rt_xbuf_cap) {
        const size_t mask = g_rt_xbuf_cap - 1;
        size_t i = rt_xbuf_slot(k, g_rt_xbuf_cap);
        while (g_rt_xbuf[i].key && g_rt_xbuf[i].key != k) i = (i + 1) & mask;
        if (g_rt_xbuf[i].key == k) {   /* backward-shift deletion keeps every probe chain intact */
            rt_acct_sub(&g_acct_asbuf, (int64_t)g_rt_xbuf[i].size);
            size_t j = i;
            for (;;) {
                j = (j + 1) & mask;
                if (!g_rt_xbuf[j].key) break;
                size_t h = rt_xbuf_slot(g_rt_xbuf[j].key, g_rt_xbuf_cap);
                bool movable = (j > i) ? (h <= i || h > j) : (h <= i && h > j);
                if (movable) { g_rt_xbuf[i] = g_rt_xbuf[j]; i = j; }
            }
            g_rt_xbuf[i] = (struct RtXbuf){ 0, 0, false };
            g_rt_xbuf_n--;
        }
    }
    pthread_mutex_unlock(&g_rt_xbuf_lock);
}

#define RT_AS_TYPE_UNBUILT ((VkAccelerationStructureTypeKHR)0x7fffffff)
#define RT_AS_BUCKETS 1024
static struct RtAsTrack *g_rt_as_tracks[RT_AS_BUCKETS];
static pthread_mutex_t   g_rt_as_lock = PTHREAD_MUTEX_INITIALIZER;

static inline unsigned rt_as_hash(uint64_t v) {
    v ^= v >> 16;
    v ^= v >> 32;
    return (unsigned)(v % RT_AS_BUCKETS);
}

static struct RtAsTrack *rt_as_lookup_locked(VkAccelerationStructureKHR as) {
    if (!as) return NULL;
    unsigned b = rt_as_hash((uint64_t)(uintptr_t)as);
    for (struct RtAsTrack *t = g_rt_as_tracks[b]; t; t = t->next) {
        if (t->orig_handle == as || t->active_handle == as) return t;
    }
    for (unsigned i = 0; i < RT_AS_BUCKETS; i++) {
        if (i == b) continue;
        for (struct RtAsTrack *t = g_rt_as_tracks[i]; t; t = t->next) {
            if (t->orig_handle == as || t->active_handle == as) return t;
        }
    }
    return NULL;
}

static VKAPI_ATTR VkResult VKAPI_CALL
compat_CreateAccelerationStructureKHR(
    VkDevice                                    device,
    const VkAccelerationStructureCreateInfoKHR *pCreateInfo,
    const VkAllocationCallbacks                *pAllocator,
    VkAccelerationStructureKHR                 *pAccelerationStructure)
{
    struct DeviceData *d = get_device_data(device);
    if (!d || !d->fpCreateAccelerationStructureKHR) return VK_ERROR_INITIALIZATION_FAILED;

    VkResult res = d->fpCreateAccelerationStructureKHR(device, pCreateInfo, pAllocator, pAccelerationStructure);
    if (res == VK_SUCCESS && pAccelerationStructure && *pAccelerationStructure) {
        struct RtAsTrack *track = (struct RtAsTrack *)calloc(1, sizeof(*track));
        if (track) {
            track->orig_handle   = *pAccelerationStructure;
            track->active_handle = *pAccelerationStructure;
            track->device        = device;
            if (pCreateInfo) {
                track->buffer       = pCreateInfo->buffer;
                track->offset       = pCreateInfo->offset;
                track->size         = pCreateInfo->size;
                track->created_type = pCreateInfo->type;
            }
            track->built_type    = RT_AS_TYPE_UNBUILT; /* unbuilt */
            track->active_type   = track->created_type;
            track->refit_ready   = true;
            track->xfer_ok       = rt_xfer_buf_has(track->buffer);
            track->acct_cls      = -1;
            rt_as_acct_locked(track, false);   /* not linked yet: no lock needed */
            rt_as_acct_host(track);
            if (g_debug && pCreateInfo && pCreateInfo->buffer && d->fpGetBufferMemoryRequirements) {
                VkMemoryRequirements mr = {0};
                d->fpGetBufferMemoryRequirements(device, pCreateInfo->buffer, &mr);
                track->buf_rest = mr.size > pCreateInfo->offset ? mr.size - pCreateInfo->offset : 0;
            }

            pthread_mutex_lock(&g_rt_as_lock);
            unsigned b = rt_as_hash((uint64_t)(uintptr_t)track->orig_handle);
            track->next = g_rt_as_tracks[b];
            g_rt_as_tracks[b] = track;
            pthread_mutex_unlock(&g_rt_as_lock);
            COMPAT_INFO("CreateAS: handle=%p type=%u buf=%p off=%llu sz=%llu",
                        (void *)*pAccelerationStructure,
                        pCreateInfo ? pCreateInfo->type : 0,
                        pCreateInfo ? (void *)pCreateInfo->buffer : NULL,
                        pCreateInfo ? (unsigned long long)pCreateInfo->offset : 0ULL,
                        pCreateInfo ? (unsigned long long)pCreateInfo->size : 0ULL);
        } else {
            /* round 4 (Codex r3 #6): an untracked AS defeats every check that needs its track (UPD_SRC, CLONE size,
             * type recreation); without host memory for the track the creation fails instead */
            if (d->fpDestroyAccelerationStructureKHR)
                d->fpDestroyAccelerationStructureKHR(device, *pAccelerationStructure, pAllocator);
            *pAccelerationStructure = VK_NULL_HANDLE;
            LOG("RT_EMU: CreateAS: out of host memory for the AS track -> VK_ERROR_OUT_OF_HOST_MEMORY");
            return VK_ERROR_OUT_OF_HOST_MEMORY;
        }
    }
    return res;
}

/* 1110b: destroys the track's driver handles: the active one, the app's original (if a type change replaced it) and
 * every intermediate retired one. Layer-created handles were created with a NULL allocator. */
static void rt_as_destroy_track(struct DeviceData *d, VkDevice device, struct RtAsTrack *victim,
                                const VkAllocationCallbacks *pAllocator) {
    if (d->fpDestroyAccelerationStructureKHR) {
        bool replaced = victim->active_handle != victim->orig_handle;
        d->fpDestroyAccelerationStructureKHR(device, victim->active_handle, replaced ? NULL : pAllocator);
        if (replaced && victim->orig_handle)
            d->fpDestroyAccelerationStructureKHR(device, victim->orig_handle, pAllocator);
        for (uint32_t i = 0; i < victim->n_retired; i++)
            d->fpDestroyAccelerationStructureKHR(device, victim->retired[i], NULL);
    }
    rt_as_acct_locked(victim, true);   /* unlinked: owned by this thread */
    if (victim->acct_host) rt_acct_sub(&g_acct_tracks, (int64_t)victim->acct_host);
    free(victim->retired);
    free(victim);
}

static VKAPI_ATTR void VKAPI_CALL
compat_DestroyAccelerationStructureKHR(
    VkDevice                                    device,
    VkAccelerationStructureKHR                  accelerationStructure,
    const VkAllocationCallbacks                *pAllocator)
{
    struct DeviceData *d = get_device_data(device);
    if (!d || !accelerationStructure) return;

    pthread_mutex_lock(&g_rt_as_lock);
    unsigned b = rt_as_hash((uint64_t)(uintptr_t)accelerationStructure);
    for (unsigned k = 0; k <= RT_AS_BUCKETS; k++) {
        unsigned bi = k == 0 ? b : k - 1;      /* own bucket first, then all others */
        if (k > 0 && bi == b) continue;
        for (struct RtAsTrack **pp = &g_rt_as_tracks[bi]; *pp; pp = &(*pp)->next) {
            if ((*pp)->orig_handle == accelerationStructure || (*pp)->active_handle == accelerationStructure) {
                struct RtAsTrack *victim = *pp;
                *pp = victim->next;
                pthread_mutex_unlock(&g_rt_as_lock);
                rt_as_destroy_track(d, device, victim, pAllocator);
                return;
            }
        }
    }
    pthread_mutex_unlock(&g_rt_as_lock);

    if (d->fpDestroyAccelerationStructureKHR) {
        d->fpDestroyAccelerationStructureKHR(device, accelerationStructure, pAllocator);
    }
}


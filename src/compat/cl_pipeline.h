// SPDX-License-Identifier: GPL-3.0-or-later
// cl_pipeline.h: descriptor pools, shader modules, FAIL_LOG, SG_LOG, PIPELINE_CACHE, ASYNC_PSO, pipelines. Included by mali_compat_layer.c.
// -----------------------------------------------------------------------------
// Device Level Interceptions
// -----------------------------------------------------------------------------
// A failing shader module / compute pipeline leaves its SPIR-V as /root/mali-fail-<hash>.spv (first 32 only).
// [game/doom_eternal, Claude Opus 5.5] 2026-11-06
// MALI_COMPAT_DESC_POOL_SCALE=<n> (2..64): every vkCreateDescriptorPool gets maxSets and each descriptor count x n.
// DOOM Eternal (idDescriptorSetFrameAllocator, DOOMEternalx64vk.exe 0x141c17da0) creates one pool per frame bank with
// maxSets 512 and {UB 512, SSBO 192, storage image 16, sampled image 512, sampler 16, UTB 16, STB 16}, but its own
// allocator admits up to 4096 sets and 8x those counts per bank (limits 0x143985e54.., checked at 0x141c18639..86af).
// Desktop drivers do not enforce pool type counts; the vendor driver does, vkAllocateDescriptorSets fails and the
// engine asserts (int3 at 0x141c18809). Costs only pool memory. [game/doom_eternal, Claude Opus 5.5] 2026-11-06
static uint32_t g_desc_pool_scale;
// MALI_COMPAT_DESC_POOL_PAD=<n> (1..64): every core descriptor type that a vkCreateDescriptorPool does not list at all is
// added with n x maxSets descriptors. Quake II RTX <= 1.8.0 (vertex_buffer.c vkpt_vertex_buffer_create) puts a
// UNIFORM_BUFFER binding (sun color UBO) in its vbo set layout but gives the pool only a STORAGE_BUFFER size (fixed
// upstream in 1.8.1). Desktop drivers allocate anyway; the vendor driver refuses the set (NULL), the release build
// ignores the VkResult and vkUpdateDescriptorSets(dstSet = NULL) faults at NULL+8. Scaling cannot help (0 x n = 0).
// [keeper, Claude Opus 5.5] 2026-11-07
static uint32_t g_desc_pool_pad;
static int g_fail_lvl; // MALI_COMPAT_FAIL_LOG level (1 = failures, 2 = also the first 16 calls of each descriptor entry)
static void fl_pool_log(const VkDescriptorPoolCreateInfo *ci, VkResult r);
static VKAPI_ATTR VkResult VKAPI_CALL compat_CreateDescriptorPool(VkDevice device, const VkDescriptorPoolCreateInfo *pCreateInfo,
    const VkAllocationCallbacks *pAllocator, VkDescriptorPool *pPool) {
    struct DeviceData *d = get_device_data(device);
    VkDescriptorPoolCreateInfo ci = *pCreateInfo;
    VkDescriptorPoolSize sizes[64];
    uint32_t k = g_desc_pool_scale;
    uint32_t added = 0;
    if ((k > 1 || g_desc_pool_pad) && ci.poolSizeCount <= 64 - 11) {
        for (uint32_t i = 0; i < ci.poolSizeCount; i++) {
            sizes[i] = pCreateInfo->pPoolSizes[i];
            // inline uniform block counts are bytes: keep them (the IUB pNext struct caps the bindings anyway)
            if (k > 1 && sizes[i].type != VK_DESCRIPTOR_TYPE_INLINE_UNIFORM_BLOCK) sizes[i].descriptorCount *= k;
        }
        if (k > 1) ci.maxSets *= k;
        if (g_desc_pool_pad) // core types 0..10 (SAMPLER .. INPUT_ATTACHMENT) the app left out
            for (int t = VK_DESCRIPTOR_TYPE_SAMPLER; t <= VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT; t++) {
                bool have = false;
                for (uint32_t i = 0; i < pCreateInfo->poolSizeCount; i++) if ((int)pCreateInfo->pPoolSizes[i].type == t) have = true;
                if (!have) { sizes[ci.poolSizeCount + added] = (VkDescriptorPoolSize){(VkDescriptorType)t, g_desc_pool_pad * ci.maxSets}; added++; }
            }
        ci.poolSizeCount += added;
        ci.pPoolSizes = sizes;
    }
    VkResult r = d->fpCreateDescriptorPool(device, &ci, pAllocator, pPool);
    VkResult r_scaled = r;
    static atomic_uint logged;
    if ((k > 1 || g_desc_pool_pad) && atomic_fetch_add(&logged, 1) < 4)
        fprintf(stderr, "mali-compat: desc-pool-scale x%u pad %u: maxSets %u -> %u, %u sizes (+%u missing types), flags 0x%x, result %d\n", k,
                g_desc_pool_pad, pCreateInfo->maxSets, ci.maxSets, pCreateInfo->poolSizeCount, added, (unsigned)ci.flags, r);
    bool fell_back = false;
    if (r != VK_SUCCESS && (k > 1 || added)) { // fall back to the app's own sizes rather than fail
        r = d->fpCreateDescriptorPool(device, pCreateInfo, pAllocator, pPool);
        fell_back = true;
        fprintf(stderr, "mali-compat: desc-pool-scale: scaled pool failed, app sizes -> %d\n", r);
    }
    if (g_fail_lvl) fl_pool_log(pCreateInfo, r);
    if (g_addrmap) { // N marker: the app's request -> what the driver got (app sizes again after a fallback)
        const VkDescriptorPoolCreateInfo *got = fell_back ? pCreateInfo : &ci;
        char sz[400];
        size_t o = 0;
        sz[0] = 0;
        for (uint32_t i = 0; i < got->poolSizeCount && o < sizeof sz - 32; i++) {
            uint32_t req = i < pCreateInfo->poolSizeCount ? pCreateInfo->pPoolSizes[i].descriptorCount : 0;
            o += (size_t)snprintf(sz + o, sizeof sz - o, "%s%d:%u->%u", i ? "," : "", got->pPoolSizes[i].type, req,
                                  got->pPoolSizes[i].descriptorCount);
        }
        const VkDescriptorPoolInlineUniformBlockCreateInfo *iub =
            find_struct(pCreateInfo->pNext, VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_INLINE_UNIFORM_BLOCK_CREATE_INFO);
        am_emit('N', r == VK_SUCCESS && pPool ? (uint64_t)*pPool : 0, AM_NA, ML_NONE, BUDGET_NO_HEAP, AM_NA, 0,
                "what=dpool-create maxSets=%u->%u flags=0x%x sizes=%s iub=%d scale=%u result=%d scaled_result=%d", pCreateInfo->maxSets,
                got->maxSets, (unsigned)got->flags, sz, iub ? (int)iub->maxInlineUniformBlockBindings : -1, k, r, r_scaled);
    }
    return r;
}

static bool g_fail_log; // MALI_COMPAT_FAIL_LOG=1: also hook vkCreateComputePipelines to log its failures

// ---- MALI_COMPAT_FAIL_LOG descriptor hooks (Quake II RTX native: vkUpdateDescriptorSets got dstSet = NULL in
// vkpt_vertex_buffer_create, so its set layout or set allocation failed silently; Q2RTX release builds drop the
// VkResult). FAIL_LOG=1: every non-success VkResult of vkCreateDescriptorSetLayout / vkAllocateDescriptorSets /
// vkCreateDescriptorPool / vkCreatePipelineLayout, and a "success" that leaves a NULL handle, is printed with its
// binding list / pool sizes / set layouts. FAIL_LOG=2: also the first 16 calls of each, successes included.
// Not active while MALI_COMPAT_DESC_CAPTURE owns these entries.
// MALI_COMPAT_DSL_STAGE_RETRY=1: a failed set layout is retried once with VK_SHADER_STAGE_ALL bindings reduced to
// graphics + compute + ray tracing stages (0x3f3f; drops mesh/task/subpass-shading/cluster bits).
// [keeper, Claude Opus 5.5] 2026-11-07
static bool g_dsl_stage_retry;
static bool fl_want(VkResult r, atomic_uint *n) {
    if (r != VK_SUCCESS) return true;
    return g_fail_lvl >= 2 && atomic_fetch_add(n, 1) < 16;
}
static void fl_dsl_print(const char *tag, const VkDescriptorSetLayoutCreateInfo *ci, VkResult r, uint64_t h) {
    const VkDescriptorSetLayoutBindingFlagsCreateInfo *bf = NULL;
    for (const VkBaseInStructure *e = ci->pNext; e; e = e->pNext)
        if (e->sType == VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO) bf = (const void *)e;
    char buf[2048]; int o = 0;
    for (uint32_t i = 0; i < ci->bindingCount && i < 40 && o < (int)sizeof buf - 64; i++) {
        const VkDescriptorSetLayoutBinding *b = &ci->pBindings[i];
        o += snprintf(buf + o, sizeof buf - o, " [b%u t%d n%u s0x%x%s f0x%x]", b->binding, (int)b->descriptorType,
                      b->descriptorCount, (unsigned)b->stageFlags, b->pImmutableSamplers ? " imm" : "",
                      bf && i < bf->bindingCount && bf->pBindingFlags ? (unsigned)bf->pBindingFlags[i] : 0u);
    }
    fprintf(stderr, "mali-compat: fail-log %s vkCreateDescriptorSetLayout result %d handle 0x%llx flags 0x%x pNext %s, %u bindings:%s\n",
            tag, r, (unsigned long long)h, (unsigned)ci->flags, ci->pNext ? (bf ? "binding-flags" : "other") : "none",
            ci->bindingCount, buf);
}
static VKAPI_ATTR VkResult VKAPI_CALL compat_FlCreateDescriptorSetLayout(VkDevice device, const VkDescriptorSetLayoutCreateInfo *ci,
                                                                          const VkAllocationCallbacks *a, VkDescriptorSetLayout *l) {
    struct DeviceData *d = get_device_data(device);
    static atomic_uint n;
    VkResult r = d->fpCreateDescriptorSetLayout(device, ci, a, l);
    bool null_ok = r == VK_SUCCESS && l && *l == VK_NULL_HANDLE;
    if (fl_want(r, &n) || null_ok) fl_dsl_print(null_ok ? "NULL-ON-SUCCESS" : "", ci, r, l ? (uint64_t)*l : 0);
    if (r != VK_SUCCESS && g_dsl_stage_retry && ci->bindingCount <= 64) {
        VkDescriptorSetLayoutBinding b[64];
        VkDescriptorSetLayoutCreateInfo c2 = *ci;
        bool changed = false;
        for (uint32_t i = 0; i < ci->bindingCount; i++) {
            b[i] = ci->pBindings[i];
            if (b[i].stageFlags & ~0x3f3fu) { b[i].stageFlags &= 0x3f3fu; changed = true; }
        }
        if (changed) {
            c2.pBindings = b;
            r = d->fpCreateDescriptorSetLayout(device, &c2, a, l);
            fl_dsl_print("stage-retry", &c2, r, l ? (uint64_t)*l : 0);
        }
    }
    return r;
}
static VKAPI_ATTR VkResult VKAPI_CALL compat_FlAllocateDescriptorSets(VkDevice device, const VkDescriptorSetAllocateInfo *ai,
                                                                       VkDescriptorSet *sets) {
    static atomic_uint n;
    VkResult r = get_device_data(device)->fpAllocateDescriptorSets(device, ai, sets);
    bool null_ok = false;
    if (r == VK_SUCCESS)
        for (uint32_t i = 0; i < ai->descriptorSetCount; i++) if (!sets[i]) null_ok = true;
    if (fl_want(r, &n) || null_ok) {
        char buf[512]; int o = 0;
        for (uint32_t i = 0; i < ai->descriptorSetCount && i < 16 && o < (int)sizeof buf - 48; i++)
            o += snprintf(buf + o, sizeof buf - o, " L0x%llx->S0x%llx", (unsigned long long)ai->pSetLayouts[i],
                          (unsigned long long)(r == VK_SUCCESS ? sets[i] : 0));
        fprintf(stderr, "mali-compat: fail-log %svkAllocateDescriptorSets result %d pool 0x%llx count %u pNext %s:%s\n",
                null_ok ? "NULL-ON-SUCCESS " : "", r, (unsigned long long)ai->descriptorPool, ai->descriptorSetCount,
                ai->pNext ? "yes" : "none", buf);
    }
    return r;
}
static void fl_pool_log(const VkDescriptorPoolCreateInfo *ci, VkResult r) {
    static atomic_uint n;
    if (!fl_want(r, &n)) return;
    char buf[1024]; int o = 0;
    for (uint32_t i = 0; i < ci->poolSizeCount && i < 24 && o < (int)sizeof buf - 32; i++)
        o += snprintf(buf + o, sizeof buf - o, " t%d:%u", (int)ci->pPoolSizes[i].type, ci->pPoolSizes[i].descriptorCount);
    fprintf(stderr, "mali-compat: fail-log vkCreateDescriptorPool result %d (app values) maxSets %u flags 0x%x pNext %s sizes:%s\n",
            r, ci->maxSets, (unsigned)ci->flags, ci->pNext ? "yes" : "none", buf);
}
static VKAPI_ATTR VkResult VKAPI_CALL compat_FlCreatePipelineLayout(VkDevice device, const VkPipelineLayoutCreateInfo *ci,
                                                                     const VkAllocationCallbacks *a, VkPipelineLayout *l) {
    static atomic_uint n;
    VkResult r = get_device_data(device)->fpCreatePipelineLayout(device, ci, a, l);
    if (fl_want(r, &n)) {
        char buf[512]; int o = 0;
        for (uint32_t i = 0; i < ci->setLayoutCount && i < 16 && o < (int)sizeof buf - 24; i++)
            o += snprintf(buf + o, sizeof buf - o, " 0x%llx", (unsigned long long)ci->pSetLayouts[i]);
        for (uint32_t i = 0; i < ci->pushConstantRangeCount && i < 8 && o < (int)sizeof buf - 40; i++)
            o += snprintf(buf + o, sizeof buf - o, " pc[s0x%x %u+%u]", (unsigned)ci->pPushConstantRanges[i].stageFlags,
                          ci->pPushConstantRanges[i].offset, ci->pPushConstantRanges[i].size);
        fprintf(stderr, "mali-compat: fail-log vkCreatePipelineLayout result %d handle 0x%llx sets %u:%s\n", r,
                (unsigned long long)(r == VK_SUCCESS ? *l : 0), ci->setLayoutCount, buf);
    }
    return r;
}
static void fail_spv_dump(const uint32_t *code, size_t bytes, uint32_t hash) {
    static atomic_uint n;
    if (!code || !bytes || atomic_fetch_add(&n, 1) >= 32) return;
    char path[64];
    snprintf(path, sizeof path, "/root/mali-fail-%08x.spv", hash);
    FILE *f = fopen(path, "wb");
    if (f) { fwrite(code, 1, bytes, f); fclose(f); }
}

// ---- MALI_COMPAT_SG_LOG=1 (diagnostic, default off; docs/games/doom_eternal.md 2026-11-13). DOOM Eternal demands subgroup
// support in the VERTEX stage; this driver has it only in FRAGMENT + COMPUTE (real stages 0x3f30) and FAKE_SUBGROUP reports
// the vertex bit anyway. This key answers "does any vertex/tess/geometry shader really use subgroup instructions, and which
// fragment/compute shaders do": every created shader module is scanned (read only, the module is passed on unchanged) for
// OpGroupNonUniform* (333..366, RotateKHR 4431) and the Subgroup* built-ins. One line per module that has any (first 256),
// those modules dumped to /root/mali-sg-<hash>.spv (first 24, plus up to 16 more with a faked stage), totals every 10 s on
// the pc2 log thread. Counters are atomics; nothing else is touched. [game/doom_eternal, Claude Fable 5.2] 2026-11-13
// =2 (diag-1113): as 1, and EVERY module with subgroup use is dumped, fragment ones included (each distinct hash once,
// cap SGL_DUMP2_MAX files, one line when the cap is hit); the module lines run up to 4096; the SPV_KHR_shader_ballot /
// SPV_KHR_subgroup_vote opcodes (4421 4422 4428-4430 4432) and quad-control 5110/5111 are counted too (=1 does not).
// [game/doom_eternal keeper, Claude Opus 5.5] 2026-11-13
#define SGL_DUMP2_MAX 128
#define SGL_MODELS 7 // 0 vertex, 1 tess control, 2 tess eval, 3 geometry, 4 fragment, 5 compute, 6 any other
static int g_sgl_on;
static atomic_ullong g_sgl_mod[SGL_MODELS], g_sgl_sg[SGL_MODELS], g_sgl_total, g_sgl_total_sg;
static void sgl_init(void) {
    const char *e = getenv("MALI_COMPAT_SG_LOG");
    g_sgl_on = e && atoi(e) > 0 ? (atoi(e) >= 2 ? 2 : 1) : 0;
    if (g_sgl_on)
        fprintf(stderr, "mali-compat: sg-log on (MALI_COMPAT_SG_LOG): shader modules are scanned for subgroup instructions "
                "(read only); FAKED-STAGE = a vertex/tess/geometry module, where this driver has no subgroup support\n");
    if (g_sgl_on >= 2)
        fprintf(stderr, "mali-compat: sg-log level 2: every module with subgroup use is dumped to /root/mali-sg-<hash>.spv "
                "(each hash once, cap %d files), module lines up to 4096, KHR ballot/vote + quad-control opcodes counted\n",
                SGL_DUMP2_MAX);
}
static int sgl_on(void) {
    static pthread_once_t once = PTHREAD_ONCE_INIT;
    pthread_once(&once, sgl_init);
    return g_sgl_on;
}
// MALI_COMPAT_SG_LOG=2: true when this hash is to be written (first time seen, below the cap).
static bool sgl_dump2(uint32_t hash) {
    static pthread_mutex_t m = PTHREAD_MUTEX_INITIALIZER;
    static uint32_t done[SGL_DUMP2_MAX], n;
    static bool capped;
    bool fresh = true, cap_line = false;
    pthread_mutex_lock(&m);
    for (uint32_t i = 0; i < n && fresh; i++) fresh = done[i] != hash;
    if (fresh && n == SGL_DUMP2_MAX) { fresh = false; cap_line = !capped; capped = true; }
    else if (fresh) done[n++] = hash;
    pthread_mutex_unlock(&m);
    if (cap_line)
        fprintf(stderr, "mali-compat: sg-log: dump cap of %d files reached at spv %08x; later modules are not dumped\n",
                SGL_DUMP2_MAX, hash);
    return fresh;
}
static void sgl_scan(const uint32_t *code, size_t bytes) {
    size_t n = bytes / 4;
    if (!code || n < 5 || code[0] != 0x07230203u) return;
    uint32_t models = 0, ops = 0, builtins = 0, opc[35] = {0}; // opc[0..33] = 333..366, opc[34] = RotateKHR
    static const uint16_t sgl_x[8] = {4421, 4422, 4428, 4429, 4430, 4432, 5110, 5111}; // =2 only: KHR ballot/vote, quad ctl
    uint32_t opx[8] = {0};
    for (size_t i = 5; i < n;) {
        uint32_t wc = code[i] >> 16, op = code[i] & 0xffffu;
        if (!wc || wc > n - i) break;
        if (op == 15 && wc >= 2) models |= 1u << (code[i + 1] <= 5 ? code[i + 1] : 6); // OpEntryPoint
        else if (op >= 333 && op <= 366) { ops++; opc[op - 333]++; }
        else if (op == 4431) { ops++; opc[34]++; }
        else if (g_sgl_on >= 2 && ((op >= 4421 && op <= 4432) || op == 5110 || op == 5111))
            for (int j = 0; j < 8; j++) { if (sgl_x[j] == op) { ops++; opx[j]++; } }
        else if ((op == 71 && wc >= 4 && code[i + 2] == 11) || (op == 72 && wc >= 5 && code[i + 3] == 11)) { // (Member)Decorate BuiltIn
            uint32_t b = code[i + (op == 71 ? 3 : 4)];
            if (b == 36 || b == 38 || b == 40 || b == 41) builtins |= 1u << (b - 36); // Size, NumSubgroups, Id, LocalInvocationId
            else if (b >= 4416 && b <= 4420) builtins |= 1u << (8 + b - 4416);      // Eq/Ge/Gt/Le/Lt mask
        }
        i += wc;
    }
    bool sg = ops || builtins;
    for (int m = 0; m < SGL_MODELS; m++)
        if (models & (1u << m)) {
            atomic_fetch_add_explicit(&g_sgl_mod[m], 1, memory_order_relaxed);
            if (sg) atomic_fetch_add_explicit(&g_sgl_sg[m], 1, memory_order_relaxed);
        }
    atomic_fetch_add_explicit(&g_sgl_total, 1, memory_order_relaxed);
    if (!sg) return;
    unsigned long long k = atomic_fetch_add_explicit(&g_sgl_total_sg, 1, memory_order_relaxed);
    bool faked = (models & 0x0fu) != 0;
    static atomic_uint lines, dumps, fdumps;
    uint32_t hash = spv_hash(code, bytes);
    if (atomic_fetch_add(&lines, 1) < (g_sgl_on >= 2 ? 4096u : 256u)) {
        char l[512];
        int w = 0;
        for (int j = 0; j < 35 && w < (int)sizeof l - 16; j++)
            if (opc[j]) w += snprintf(l + w, sizeof l - w, " %d:%u", j < 34 ? 333 + j : 4431, opc[j]);
        for (int j = 0; j < 8 && w < (int)sizeof l - 16; j++)
            if (opx[j]) w += snprintf(l + w, sizeof l - w, " %d:%u", sgl_x[j], opx[j]);
        if (!w) l[0] = 0;
        fprintf(stderr, "mali-compat: sg-log: #%llu spv %08x %zu B models 0x%02x ops %u [%s ] builtins 0x%04x%s\n", k, hash, bytes,
                models, ops, l, builtins, faked ? " FAKED-STAGE" : "");
    }
    bool dump;
    if (g_sgl_on >= 2) dump = sgl_dump2(hash);
    else {
        dump = atomic_fetch_add(&dumps, 1) < 24;
        if (!dump && faked) dump = atomic_fetch_add(&fdumps, 1) < 16;
    }
    if (dump) {
        char path[64];
        snprintf(path, sizeof path, "/root/mali-sg-%08x.spv", hash);
        FILE *f = fopen(path, "wb");
        if (f) { fwrite(code, 1, bytes, f); fclose(f); }
    }
}
// Called from the pc2 log thread only (every 10 s).
static void sgl_summary(void) {
    static unsigned long long last; // that thread only
    unsigned long long t = atomic_load_explicit(&g_sgl_total, memory_order_relaxed);
    if (!sgl_on() || t == last) return;
    last = t;
    unsigned long long m[SGL_MODELS], g[SGL_MODELS];
    for (int i = 0; i < SGL_MODELS; i++) {
        m[i] = atomic_load_explicit(&g_sgl_mod[i], memory_order_relaxed);
        g[i] = atomic_load_explicit(&g_sgl_sg[i], memory_order_relaxed);
    }
    fprintf(stderr, "mali-compat: sg-log (pid %d): modules %llu, with subgroup use %llu | per stage with/all: vertex %llu/%llu "
            "tessc %llu/%llu tesse %llu/%llu geom %llu/%llu fragment %llu/%llu compute %llu/%llu other %llu/%llu\n", (int)getpid(), t,
            atomic_load_explicit(&g_sgl_total_sg, memory_order_relaxed), g[0], m[0], g[1], m[1], g[2], m[2], g[3], m[3], g[4],
            m[4], g[5], m[5], g[6], m[6]);
}

static VKAPI_ATTR VkResult VKAPI_CALL compat_CreateShaderModule(
    VkDevice device, const VkShaderModuleCreateInfo *pCreateInfo,
    const VkAllocationCallbacks *pAllocator, VkShaderModule *pShaderModule) {
    if (g_rt_pipeline && pCreateInfo && pShaderModule &&
        rt_is_ray_tracing_spirv(pCreateInfo->pCode, pCreateInfo->codeSize)) {
        *pShaderModule = rt_register_shader_module(device, pCreateInfo->pCode, pCreateInfo->codeSize);
        return *pShaderModule ? VK_SUCCESS : VK_ERROR_OUT_OF_HOST_MEMORY;
    }
    struct DeviceData *d = get_device_data(device);
    uint32_t hash = 0, mflags = 0, cnt[4];
    if (pCreateInfo && sgl_on()) sgl_scan(pCreateInfo->pCode, pCreateInfo->codeSize); // MALI_COMPAT_SG_LOG, read only
    if (g_gpu_time || g_dump_hash_count || g_vs_scan || g_dc_on || g_ct) {
        hash = spv_hash(pCreateInfo->pCode, pCreateInfo->codeSize);
        spv_dump_if_wanted(pCreateInfo->pCode, pCreateInfo->codeSize, hash);
    } else if (g_db_hash) {
        hash = spv_hash(pCreateInfo->pCode, pCreateInfo->codeSize); // disp-barrier: the app's code, before any patch
    }
    if (g_vs_scan && spv_vs_stores(pCreateInfo->pCode, pCreateInfo->codeSize, cnt)) { // KZ2, before any patch
        mflags = PF_VSW;
        vs_store_note(hash, cnt, false);
    }
    size_t patched_bytes = 0;
    uint32_t *patched = compat_patch_spirv(pCreateInfo->pCode, pCreateInfo->codeSize, &patched_bytes);

    VkResult r;
    if (patched) {
        VkShaderModuleCreateInfo mod_ci = *pCreateInfo;
        mod_ci.pCode = patched;
        mod_ci.codeSize = patched_bytes;
        r = d->fpCreateShaderModule(device, &mod_ci, pAllocator, pShaderModule);
        if (r == VK_SUCCESS && g_dump_inflight && pShaderModule) inflight_module_put(*pShaderModule, patched, patched_bytes);
        free(patched);
    } else {
        r = d->fpCreateShaderModule(device, pCreateInfo, pAllocator, pShaderModule);
        if (r == VK_SUCCESS && g_dump_inflight && pShaderModule) inflight_module_put(*pShaderModule, pCreateInfo->pCode, pCreateInfo->codeSize);
    }
    if (r != VK_SUCCESS) { // [game/doom_eternal, Claude Opus 5.5] 2026-11-06: failures were silent; keep the SPIR-V
        uint32_t fh = spv_hash(pCreateInfo->pCode, pCreateInfo->codeSize);
        fprintf(stderr, "mali-compat: vkCreateShaderModule FAILED: %d (spv %08x, %zu B, patched %d)\n", r, fh,
                pCreateInfo->codeSize, patched_bytes != 0);
        fail_spv_dump(pCreateInfo->pCode, pCreateInfo->codeSize, fh);
    }
    if (r == VK_SUCCESS && (g_gpu_time || g_dc_on || g_ct || g_db_hash) && pShaderModule) hash_put(g_module_hash, (uint64_t)*pShaderModule, &hash, NULL, 1, mflags);
    if (r == VK_SUCCESS && g_fragdepth_fix && pShaderModule && spv_writes_fragdepth(pCreateInfo->pCode, pCreateInfo->codeSize))
        fragdepth_module_put(*pShaderModule, pCreateInfo->pCode, pCreateInfo->codeSize);
    return r;
}

// Debug aid: with /root/mali-spv-inflight present, each thread writes the pipeline it is about to create (flags,
// stage pNext types, inline SPIR-V) to /root/mali-spv-inflight/<tid>-*, overwriting the previous one. A thread that
// dies inside the driver leaves the pipeline that killed it. [Claude Opus 5.5]
#include <sys/syscall.h>
static void dump_inflight_stage(FILE *f, long tid, uint32_t i, uint32_t st, const VkPipelineShaderStageCreateInfo *sg) {
    fprintf(f, "  stage 0x%x module %p name %s pNext:", sg->stage, (void *)sg->module, sg->pName ? sg->pName : "?");
    const uint32_t *code = NULL;
    size_t bytes = 0;
    for (const VkBaseInStructure *c = sg->pNext; c; c = c->pNext) {
        fprintf(f, " %d", c->sType);
        if (c->sType == VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO)
            code = ((const VkShaderModuleCreateInfo *)c)->pCode, bytes = ((const VkShaderModuleCreateInfo *)c)->codeSize;
    }
    if (!code && sg->module) {
        const struct CodeNode *n = inflight_module_get(sg->module);
        if (n) code = n->code, bytes = n->bytes;
    }
    if (code) {
        char sp[112];
        snprintf(sp, sizeof sp, "/root/mali-spv-inflight/%ld-p%u-s%u.spv", tid, i, st);
        FILE *o = fopen(sp, "wb");
        if (o) { fwrite(code, 1, bytes, o); fclose(o); }
        fprintf(f, " spv %zu bytes", bytes);
    }
    if (sg->pSpecializationInfo) fprintf(f, " spec %u entries", sg->pSpecializationInfo->mapEntryCount);
    fprintf(f, "\n");
}

// the pipeline got through the driver: remove its record, so after a crash only the pipelines in flight are left
static void dump_inflight_done(void) {
    if (!g_dump_inflight) return;
    char path[96];
    snprintf(path, sizeof path, "/root/mali-spv-inflight/%ld-info.txt", (long)syscall(SYS_gettid));
    unlink(path);
}

static void dump_inflight_compute(uint32_t count, const VkComputePipelineCreateInfo *infos) {
    if (!g_dump_inflight) return;
    long tid = syscall(SYS_gettid);
    char path[96];
    snprintf(path, sizeof path, "/root/mali-spv-inflight/%ld-info.txt", tid);
    FILE *f = fopen(path, "w");
    if (!f) return;
    for (uint32_t i = 0; i < count; i++) {
        fprintf(f, "compute pipeline %u: flags 0x%x layout %p\n", i, infos[i].flags, (void *)infos[i].layout);
        dump_inflight_stage(f, tid, i, 0, &infos[i].stage);
    }
    fclose(f);
}

static void dump_inflight_pipeline(uint32_t count, const VkGraphicsPipelineCreateInfo *infos) {
    if (!g_dump_inflight) return;
    long tid = syscall(SYS_gettid);
    char path[96];
    snprintf(path, sizeof path, "/root/mali-spv-inflight/%ld-info.txt", tid);
    FILE *f = fopen(path, "w");
    if (!f) return;
    for (uint32_t i = 0; i < count; i++) {
        fprintf(f, "pipeline %u: flags 0x%x stages %u layout %p renderPass %p pNext:", i, infos[i].flags,
                infos[i].stageCount, (void *)infos[i].layout, (void *)infos[i].renderPass);
        for (const VkBaseInStructure *c = infos[i].pNext; c; c = c->pNext) fprintf(f, " %d", c->sType);
        fprintf(f, "\n");
        for (uint32_t st = 0; st < infos[i].stageCount; st++) dump_inflight_stage(f, tid, i, st, &infos[i].pStages[st]);
    }
    fclose(f);
}

// Debug aid: with /root/mali-pipelines.txt present, one line per graphics pipeline: library flags, stages, vertex
// formats, raster/depth state and attachment formats. [Claude Opus 5.5]
static void log_pipeline(const VkGraphicsPipelineCreateInfo *ci, VkResult res) {
    static pthread_mutex_t m = PTHREAD_MUTEX_INITIALIZER;
    char line[2048];
    int n = snprintf(line, sizeof line, "res %d flags 0x%x stages", res, ci->flags);
    for (uint32_t i = 0; i < ci->stageCount && n < 1800; i++) {
        uint32_t words = 0, hash = 2166136261u;
        for (const VkBaseInStructure *c = ci->pStages[i].pNext; c; c = c->pNext)
            if (c->sType == VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO) {
                const VkShaderModuleCreateInfo *sm = (const VkShaderModuleCreateInfo *)c;
                words = (uint32_t)(sm->codeSize / 4);
                for (uint32_t w = 0; w < words; w++) hash = (hash ^ sm->pCode[w]) * 16777619u;
            }
        n += snprintf(line + n, sizeof line - n, " %x:%u:%08x", ci->pStages[i].stage, words, hash);
    }
    const VkGraphicsPipelineLibraryCreateInfoEXT *lib = find_struct(ci->pNext, VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_LIBRARY_CREATE_INFO_EXT);
    if (lib) n += snprintf(line + n, sizeof line - n, " gpl 0x%x", lib->flags);
    const VkPipelineLibraryCreateInfoKHR *libs = find_struct(ci->pNext, VK_STRUCTURE_TYPE_PIPELINE_LIBRARY_CREATE_INFO_KHR);
    if (libs) n += snprintf(line + n, sizeof line - n, " links %u", libs->libraryCount);
    if (ci->pVertexInputState && n < 1800) {
        n += snprintf(line + n, sizeof line - n, " vtx");
        for (uint32_t i = 0; i < ci->pVertexInputState->vertexAttributeDescriptionCount && n < 1800; i++)
            n += snprintf(line + n, sizeof line - n, " %u:%d", ci->pVertexInputState->pVertexAttributeDescriptions[i].location,
                          ci->pVertexInputState->pVertexAttributeDescriptions[i].format);
    }
    if (ci->pRasterizationState)
        n += snprintf(line + n, sizeof line - n, " rast clamp%u discard%u cull%u front%u", ci->pRasterizationState->depthClampEnable,
                      ci->pRasterizationState->rasterizerDiscardEnable, ci->pRasterizationState->cullMode,
                      ci->pRasterizationState->frontFace);
    if (ci->pDepthStencilState)
        n += snprintf(line + n, sizeof line - n, " ds test%u write%u op%d", ci->pDepthStencilState->depthTestEnable,
                      ci->pDepthStencilState->depthWriteEnable, ci->pDepthStencilState->depthCompareOp);
    const VkPipelineRenderingCreateInfo *ri = find_struct(ci->pNext, VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO);
    if (ri) {
        n += snprintf(line + n, sizeof line - n, " rt");
        for (uint32_t i = 0; i < ri->colorAttachmentCount && n < 1900; i++)
            n += snprintf(line + n, sizeof line - n, " %d", ri->pColorAttachmentFormats[i]);
        n += snprintf(line + n, sizeof line - n, " d%d s%d", ri->depthAttachmentFormat, ri->stencilAttachmentFormat);
    }
    if (ci->pDynamicState) n += snprintf(line + n, sizeof line - n, " dyn %u", ci->pDynamicState->dynamicStateCount);
    pthread_mutex_lock(&m);
    FILE *f = fopen("/root/mali-pipelines.txt", "a");
    if (f) { fprintf(f, "%s\n", line); fclose(f); }
    pthread_mutex_unlock(&m);
}

// MALI_COMPAT_PIPELINE_CACHE=1: a persistent VkPipelineCache for the driver's compiled pipelines. Under libhybris the
// vendor driver keeps no disk cache, so every launch recompiled all of HZD's ~30k pipelines (~10 min of loading) while
// vkd3d's own cache was already complete. The layer loads /root/.cache/mali-compat/<comm>.pcache into one cache per
// process, passes it for every pipeline it creates (the app's cache is in-memory only), and writes it back once
// pipeline creation has been idle for 20 s, every minute during long compiles, and at device destroy.
// Caution (HZD, 2026-10-29): entries loaded from disk gave wrong pipelines (black menu, no loading bar) and the file
// grew every launch (145 -> 345 MB), so the vendor's cache keys are not stable across runs. =2 keeps only the
// in-process sharing.
// [Claude Opus 5.5] 2026-10-29
static bool g_pcache_on;
static bool g_pcache_mem_only; // MALI_COMPAT_PIPELINE_CACHE=2: one cache per compile thread, never loaded or saved
static pthread_mutex_t g_pc_mutex = PTHREAD_MUTEX_INITIALIZER;
static struct DeviceData *g_pc_dev;
static VkPipelineCache g_pc;
static char g_pc_path[160];
static _Atomic uint64_t g_pc_new, g_pc_last_ns;
// =2: one cache per thread instead of one per process (see pc_for)
#define PC_THREAD_MAX 64
static VkPipelineCache g_pc_thread[PC_THREAD_MAX];
static _Atomic uint32_t g_pc_thread_n;
static __thread VkPipelineCache tl_pc;
static __thread struct DeviceData *tl_pc_dev;

static void pc_save_locked(void) {
    if (g_pcache_mem_only || !g_pc_dev || !g_pc || !atomic_load(&g_pc_new)) return;
    size_t size = 0;
    if (g_pc_dev->fpGetPipelineCacheData(g_pc_dev->device, g_pc, &size, NULL) != VK_SUCCESS || !size) return;
    void *data = malloc(size);
    if (!data) return;
    uint64_t n = atomic_exchange(&g_pc_new, 0);
    if (g_pc_dev->fpGetPipelineCacheData(g_pc_dev->device, g_pc, &size, data) == VK_SUCCESS) {
        char tmp[176];
        snprintf(tmp, sizeof tmp, "%s.tmp", g_pc_path);
        FILE *f = fopen(tmp, "wb");
        if (f && fwrite(data, 1, size, f) == size && fclose(f) == 0 && rename(tmp, g_pc_path) == 0)
            fprintf(stderr, "mali-compat: pipeline cache saved: %zu bytes (+%llu pipelines)\n", size, (unsigned long long)n);
        else {
            if (f) fclose(f);
            fprintf(stderr, "mali-compat: pipeline cache save to %s failed: %s\n", g_pc_path, strerror(errno));
        }
    }
    free(data);
}

static void *pc_saver(void *arg) {
    (void)arg;
    for (;;) {
        sleep(5);
        // after 20 s without new pipelines, or every minute during a long compile (games get killed, not closed)
        static uint64_t last_save;
        uint64_t now = now_ns(), n = atomic_load(&g_pc_new);
        if (n && (now - atomic_load(&g_pc_last_ns) > 20000000000ull || (n >= 500 && now - last_save > 60000000000ull))) {
            last_save = now;
            pthread_mutex_lock(&g_pc_mutex);
            pc_save_locked();
            pthread_mutex_unlock(&g_pc_mutex);
        }
    }
    return NULL;
}

static void pc_init(struct DeviceData *d) {
    if (!g_pcache_on || !d->fpCreatePipelineCache || !d->fpGetPipelineCacheData) return;
    pthread_mutex_lock(&g_pc_mutex);
    if (g_pc_dev) { pthread_mutex_unlock(&g_pc_mutex); return; } // one cache per process (the first live device)
    char comm[32] = "unknown";
    FILE *f = fopen("/proc/self/comm", "r");
    if (f) {
        if (fgets(comm, sizeof comm, f)) comm[strcspn(comm, "\n")] = 0;
        fclose(f);
    }
    for (char *c = comm; *c; c++)
        if (*c == '/' || *c == ' ' || *c == '\\') *c = '_';
    mkdir("/root/.cache", 0755);
    mkdir("/root/.cache/mali-compat", 0755);
    snprintf(g_pc_path, sizeof g_pc_path, "/root/.cache/mali-compat/%s.pcache", comm);
    void *data = NULL;
    size_t size = 0;
    if (!g_pcache_mem_only && (f = fopen(g_pc_path, "rb"))) {
        if (fseek(f, 0, SEEK_END) == 0 && (long)(size = (size_t)ftell(f)) > 0 && (data = malloc(size))) {
            rewind(f);
            if (fread(data, 1, size, f) != size) { free(data); data = NULL; size = 0; }
        }
        fclose(f);
    }
    VkPipelineCacheCreateInfo ci = {VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO, NULL, 0, data ? size : 0, data};
    VkResult r = d->fpCreatePipelineCache(d->device, &ci, NULL, &g_pc);
    if (r != VK_SUCCESS && data) { // a cache from another driver build: start empty
        ci.initialDataSize = 0, ci.pInitialData = NULL;
        r = d->fpCreatePipelineCache(d->device, &ci, NULL, &g_pc);
    }
    free(data);
    if (r == VK_SUCCESS) {
        g_pc_dev = d;
        fprintf(stderr, "mali-compat: pipeline cache %s: loaded %zu bytes\n", g_pc_path, size);
        static bool started;
        pthread_t t;
        if (!started && !g_pcache_mem_only && pthread_create(&t, NULL, pc_saver, NULL) == 0) { pthread_detach(t); started = true; }
    } else g_pc = VK_NULL_HANDLE;
    pthread_mutex_unlock(&g_pc_mutex);
}

static void pc_destroy(struct DeviceData *d) {
    pthread_mutex_lock(&g_pc_mutex);
    if (g_pc_dev == d) {
        pc_save_locked();
        for (uint32_t i = 0; i < atomic_load(&g_pc_thread_n) && i < PC_THREAD_MAX; i++)
            if (g_pc_thread[i]) { d->fpDestroyPipelineCache(d->device, g_pc_thread[i], NULL); g_pc_thread[i] = VK_NULL_HANDLE; }
        atomic_store(&g_pc_thread_n, 0);
        d->fpDestroyPipelineCache(d->device, g_pc, NULL);
        g_pc = VK_NULL_HANDLE;
        g_pc_dev = NULL;
    }
    pthread_mutex_unlock(&g_pc_mutex);
}

// Only the game's background compile threads use the shared cache: with one cache the driver serialises compiles on
// it, and a pipeline the render thread needs then waited behind thousands of background compiles (HZD's menu froze
// at 0 fps, 2026-10-29 test 6). MALI_COMPAT_PIPELINE_CACHE_THREADS=<comm prefixes, comma-separated> (default "AsyncPso", HZD's compile
// queue; "*" = every thread). "*" broke HZD's start (2026-10-30): with a cache on the main thread vkd3d's first compute
// pipelines crashed (C0000005 in vkCreateComputePipelines, or FEX "Unhandled non-JIT atomic" after a 2 min hang).
static const char *g_pc_threads = "AsyncPso";
static __thread int tl_pc_use; // 0 unknown, 1 shared cache, 2 the app's
static bool pc_thread_ok(void) {
    if (!tl_pc_use) {
        char comm[32] = "";
        FILE *f = fopen("/proc/thread-self/comm", "r");
        if (f) {
            if (fgets(comm, sizeof comm, f)) comm[strcspn(comm, "\n")] = 0;
            fclose(f);
        }
        // a comma-separated list of prefixes (the Witcher compiles on several named threads)
        tl_pc_use = 2;
        for (const char *p = g_pc_threads; *p; ) {
            size_t n = strcspn(p, ",");
            if ((n == 1 && *p == '*') || (n && !strncmp(comm, p, n))) { tl_pc_use = 1; break; }
            p += n + (p[n] == ',');
        }
        // once per thread: which threads create pipelines, to pick the prefixes for a new game
        fprintf(stderr, "mali-compat: pipeline cache: thread '%s' %s\n", comm, tl_pc_use == 1 ? "uses it" : "skipped");
    }
    return tl_pc_use == 1;
}

// =2: one cache per thread instead of one per process. Six compile threads inserting into one shared cache spun at
// full CPU without progress (HZD stuck in the menu or intro whenever a pipeline was missing, tests 3/6/7).

// the cache to create pipelines with: the layer's own when it is on for this device and thread
static VkPipelineCache pc_for(struct DeviceData *d, VkPipelineCache app, uint32_t count) {
    if (g_pc_dev != d || !g_pc || !pc_thread_ok()) return app;
    if (g_pcache_mem_only) {
        if (tl_pc_dev != d || !tl_pc) {
            uint32_t i = atomic_fetch_add(&g_pc_thread_n, 1);
            VkPipelineCacheCreateInfo ci = {VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO};
            if (i >= PC_THREAD_MAX || d->fpCreatePipelineCache(d->device, &ci, NULL, &tl_pc) != VK_SUCCESS) return app;
            g_pc_thread[i] = tl_pc;
            tl_pc_dev = d;
        }
        return tl_pc;
    }
    atomic_fetch_add(&g_pc_new, count);
    atomic_store(&g_pc_last_ns, now_ns());
    return g_pc;
}

// -----------------------------------------------------------------------------
// Asynchronous graphics pipeline compiles (MALI_COMPAT_ASYNC_PSO=<ms>). [Claude Opus 5.5] 2026-10-30
// The vendor compiler takes 4-9 s on some pipelines (The Witcher 3 DX11), and DXVK creates them on its CS thread when
// a draw first needs them, so the game stops for that long. With the switch, every graphics pipeline is deep-copied
// and compiled on worker threads; the caller waits up to <ms> milliseconds. A compile that finishes in time returns
// the real pipeline as before. A slower one returns a stand-in handle: binding it binds the real pipeline once it is
// done, and until then the draws recorded with it are dropped (the object appears a moment later). Pipelines that
// capture with transform feedback, pipeline libraries, derivatives, shader modules (not inline SPIR-V) and create
// infos with structures the copy does not know are compiled the normal way. MALI_COMPAT_ASYNC_THREADS sets the worker
// count (default 2).
// -----------------------------------------------------------------------------
static uint32_t g_async_ms;      // 0 = off
static uint32_t g_async_threads = 2;
static _Atomic uint64_t g_async_jobs, g_async_pending, g_async_skipped, g_async_failed;

struct ArBlk { struct ArBlk *next; size_t used, cap; unsigned char data[]; };
struct Arena { struct ArBlk *head; bool fail; };
static void *ar_alloc(struct Arena *a, size_t n) {
    n = (n + 15) & ~(size_t)15;
    struct ArBlk *b = a->head;
    if (!b || b->used + n > b->cap) {
        size_t cap = n > 65536 ? n : 65536;
        if (!(b = malloc(sizeof *b + cap))) { a->fail = true; return NULL; }
        b->next = a->head; b->used = 0; b->cap = cap; a->head = b;
    }
    void *p = b->data + b->used;
    b->used += n;
    return p;
}
static const void *ar_dup(struct Arena *a, const void *src, size_t n) {
    if (!src || !n) return NULL;
    void *p = ar_alloc(a, n);
    if (p) memcpy(p, src, n);
    return p;
}
static void ar_free(struct Arena *a) {
    for (struct ArBlk *b = a->head, *n; b; b = n) { n = b->next; free(b); }
    a->head = NULL;
}

// One pNext chain. Unknown structures fail the copy (the pipeline is then compiled synchronously); creation feedback
// (written by the driver, the app's memory may be gone by then) and object names are left out.
static const void *async_copy_chain(struct Arena *a, const void *chain) {
    const void *head = NULL;
    VkBaseOutStructure *tail = NULL;
    for (const VkBaseInStructure *s = chain; s; s = s->pNext) {
        size_t size = 0;
        switch (s->sType) {
        case VK_STRUCTURE_TYPE_PIPELINE_CREATION_FEEDBACK_CREATE_INFO:
        case VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT: continue;
        case VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO: size = sizeof(VkPipelineRenderingCreateInfo); break;
        case VK_STRUCTURE_TYPE_PIPELINE_CREATE_FLAGS_2_CREATE_INFO: size = sizeof(VkPipelineCreateFlags2CreateInfoKHR); break;
        case VK_STRUCTURE_TYPE_PIPELINE_ROBUSTNESS_CREATE_INFO: size = sizeof(VkPipelineRobustnessCreateInfoEXT); break;
        case VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO: size = sizeof(VkShaderModuleCreateInfo); break;
        case VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO:
            size = sizeof(VkPipelineShaderStageRequiredSubgroupSizeCreateInfo); break;
        case VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_DIVISOR_STATE_CREATE_INFO:
            size = sizeof(VkPipelineVertexInputDivisorStateCreateInfoEXT); break;
        case VK_STRUCTURE_TYPE_PIPELINE_TESSELLATION_DOMAIN_ORIGIN_STATE_CREATE_INFO:
            size = sizeof(VkPipelineTessellationDomainOriginStateCreateInfo); break;
        case VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_DEPTH_CLIP_CONTROL_CREATE_INFO_EXT:
            size = sizeof(VkPipelineViewportDepthClipControlCreateInfoEXT); break;
        case VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_DEPTH_CLIP_STATE_CREATE_INFO_EXT:
            size = sizeof(VkPipelineRasterizationDepthClipStateCreateInfoEXT); break;
        case VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_STREAM_CREATE_INFO_EXT:
            size = sizeof(VkPipelineRasterizationStateStreamCreateInfoEXT); break;
        case VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_LINE_STATE_CREATE_INFO:
            size = sizeof(VkPipelineRasterizationLineStateCreateInfoEXT); break;
        case VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_PROVOKING_VERTEX_STATE_CREATE_INFO_EXT:
            size = sizeof(VkPipelineRasterizationProvokingVertexStateCreateInfoEXT); break;
        case VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_CONSERVATIVE_STATE_CREATE_INFO_EXT:
            size = sizeof(VkPipelineRasterizationConservativeStateCreateInfoEXT); break;
        case VK_STRUCTURE_TYPE_PIPELINE_COLOR_WRITE_CREATE_INFO_EXT: size = sizeof(VkPipelineColorWriteCreateInfoEXT); break;
        default: a->fail = true; return NULL;
        }
        VkBaseOutStructure *c = (VkBaseOutStructure *)ar_dup(a, s, size);
        if (!c) { a->fail = true; return NULL; }
        c->pNext = NULL;
        switch (s->sType) {
        case VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO: {
            VkPipelineRenderingCreateInfo *r = (VkPipelineRenderingCreateInfo *)c;
            r->pColorAttachmentFormats = ar_dup(a, r->pColorAttachmentFormats, r->colorAttachmentCount * sizeof(VkFormat));
            break;
        }
        case VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO: {
            VkShaderModuleCreateInfo *m = (VkShaderModuleCreateInfo *)c;
            m->pCode = ar_dup(a, m->pCode, m->codeSize);
            break;
        }
        case VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_DIVISOR_STATE_CREATE_INFO: {
            VkPipelineVertexInputDivisorStateCreateInfoEXT *d = (VkPipelineVertexInputDivisorStateCreateInfoEXT *)c;
            d->pVertexBindingDivisors = ar_dup(a, d->pVertexBindingDivisors,
                                               d->vertexBindingDivisorCount * sizeof(VkVertexInputBindingDivisorDescriptionEXT));
            break;
        }
        case VK_STRUCTURE_TYPE_PIPELINE_COLOR_WRITE_CREATE_INFO_EXT: {
            VkPipelineColorWriteCreateInfoEXT *w = (VkPipelineColorWriteCreateInfoEXT *)c;
            w->pColorWriteEnables = ar_dup(a, w->pColorWriteEnables, w->attachmentCount * sizeof(VkBool32));
            break;
        }
        default: break;
        }
        if (tail) tail->pNext = c; else head = c;
        tail = c;
    }
    return head;
}

#define AR_STATE(field, type)                                                         \
    if (s->field) {                                                                   \
        type *x = (type *)ar_dup(a, s->field, sizeof(type));                          \
        if (!x) return false;                                                         \
        x->pNext = async_copy_chain(a, s->field->pNext);                              \
        d->field = x;                                                                 \
    }

// Deep copy of a graphics pipeline create info into the arena; false if it cannot be copied.
static bool async_copy_ci(struct Arena *a, const VkGraphicsPipelineCreateInfo *s, VkGraphicsPipelineCreateInfo *d) {
    *d = *s;
    d->pNext = async_copy_chain(a, s->pNext);
    bool tess = false;
    VkPipelineShaderStageCreateInfo *st = s->stageCount ? ar_alloc(a, s->stageCount * sizeof *st) : NULL;
    for (uint32_t i = 0; st && i < s->stageCount; i++) {
        st[i] = s->pStages[i];
        st[i].pNext = async_copy_chain(a, s->pStages[i].pNext);
        if (s->pStages[i].pName) st[i].pName = ar_dup(a, s->pStages[i].pName, strlen(s->pStages[i].pName) + 1);
        if (s->pStages[i].pSpecializationInfo) {
            const VkSpecializationInfo *si = s->pStages[i].pSpecializationInfo;
            VkSpecializationInfo *ci = (VkSpecializationInfo *)ar_dup(a, si, sizeof *si);
            if (ci) {
                ci->pMapEntries = ar_dup(a, si->pMapEntries, si->mapEntryCount * sizeof(VkSpecializationMapEntry));
                ci->pData = ar_dup(a, si->pData, si->dataSize);
            }
            st[i].pSpecializationInfo = ci;
        }
        if (s->pStages[i].stage & (VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT | VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT)) tess = true;
    }
    d->pStages = st;
    AR_STATE(pVertexInputState, VkPipelineVertexInputStateCreateInfo)
    if (s->pVertexInputState) {
        VkPipelineVertexInputStateCreateInfo *v = (VkPipelineVertexInputStateCreateInfo *)d->pVertexInputState;
        v->pVertexBindingDescriptions = ar_dup(a, v->pVertexBindingDescriptions,
                                               v->vertexBindingDescriptionCount * sizeof(VkVertexInputBindingDescription));
        v->pVertexAttributeDescriptions = ar_dup(a, v->pVertexAttributeDescriptions,
                                                 v->vertexAttributeDescriptionCount * sizeof(VkVertexInputAttributeDescription));
    }
    AR_STATE(pInputAssemblyState, VkPipelineInputAssemblyStateCreateInfo)
    if (tess) { AR_STATE(pTessellationState, VkPipelineTessellationStateCreateInfo) } else d->pTessellationState = NULL;
    AR_STATE(pViewportState, VkPipelineViewportStateCreateInfo)
    if (s->pViewportState) {
        VkPipelineViewportStateCreateInfo *v = (VkPipelineViewportStateCreateInfo *)d->pViewportState;
        v->pViewports = ar_dup(a, v->pViewports, v->viewportCount * sizeof(VkViewport));
        v->pScissors = ar_dup(a, v->pScissors, v->scissorCount * sizeof(VkRect2D));
    }
    AR_STATE(pRasterizationState, VkPipelineRasterizationStateCreateInfo)
    AR_STATE(pMultisampleState, VkPipelineMultisampleStateCreateInfo)
    if (s->pMultisampleState && s->pMultisampleState->pSampleMask) {
        VkPipelineMultisampleStateCreateInfo *m = (VkPipelineMultisampleStateCreateInfo *)d->pMultisampleState;
        m->pSampleMask = ar_dup(a, m->pSampleMask, ((m->rasterizationSamples + 31) / 32) * sizeof(VkSampleMask));
    }
    AR_STATE(pDepthStencilState, VkPipelineDepthStencilStateCreateInfo)
    AR_STATE(pColorBlendState, VkPipelineColorBlendStateCreateInfo)
    if (s->pColorBlendState) {
        VkPipelineColorBlendStateCreateInfo *c = (VkPipelineColorBlendStateCreateInfo *)d->pColorBlendState;
        c->pAttachments = ar_dup(a, c->pAttachments, c->attachmentCount * sizeof(VkPipelineColorBlendAttachmentState));
    }
    AR_STATE(pDynamicState, VkPipelineDynamicStateCreateInfo)
    if (s->pDynamicState) {
        VkPipelineDynamicStateCreateInfo *y = (VkPipelineDynamicStateCreateInfo *)d->pDynamicState;
        y->pDynamicStates = ar_dup(a, y->pDynamicStates, y->dynamicStateCount * sizeof(VkDynamicState));
    }
    return !a->fail;
}
#undef AR_STATE

// Can this pipeline be compiled in the background? Inline SPIR-V for every stage and no transform feedback (its
// draws produce data later work reads, so they must not be dropped), no libraries or derivatives.
static bool async_eligible(const VkGraphicsPipelineCreateInfo *ci) {
    if (ci->flags & (VK_PIPELINE_CREATE_LIBRARY_BIT_KHR | VK_PIPELINE_CREATE_DERIVATIVE_BIT |
                     VK_PIPELINE_CREATE_FAIL_ON_PIPELINE_COMPILE_REQUIRED_BIT)) return false;
    if (find_struct(ci->pNext, VK_STRUCTURE_TYPE_PIPELINE_LIBRARY_CREATE_INFO_KHR) ||
        find_struct(ci->pNext, VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_LIBRARY_CREATE_INFO_EXT)) return false;
    for (uint32_t i = 0; i < ci->stageCount; i++) {
        const VkShaderModuleCreateInfo *sm = find_struct(ci->pStages[i].pNext, VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO);
        if (!sm || ci->pStages[i].module) return false;
        size_t n = sm->codeSize / 4;
        for (size_t o = 5; o < n;) {
            uint32_t wc = sm->pCode[o] >> 16, op = sm->pCode[o] & 0xffff;
            if (!wc || o + wc > n) return false;
            if (op == 17 && wc >= 2 && sm->pCode[o + 1] == 53) return false; // OpCapability TransformFeedback
            if (op == 54) break;                                         // first function: past the declarations
            o += wc;
        }
    }
    return true;
}

enum { AJ_PENDING, AJ_DONE, AJ_FAILED };
struct AsyncJob {
    pthread_mutex_t m;
    pthread_cond_t cv;
    int state;
    VkResult result;
    bool detached, destroy; // the caller returned the stand-in / the app destroyed it before the compile ended
    _Atomic(VkPipeline) real;
    struct DeviceData *d;
    VkPipelineCache cache;
    struct Arena arena;
    VkGraphicsPipelineCreateInfo ci;
    uint32_t hash[3];
    struct AsyncJob *next_job;  // work queue
    struct AsyncJob *next_live; // stand-in table
};

static pthread_mutex_t g_async_qm = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t g_async_qcv = PTHREAD_COND_INITIALIZER;
static struct AsyncJob *g_async_qhead, *g_async_qtail;
static bool g_async_started;

// stand-in handle -> job; read on every pipeline bind
#define ASYNC_BUCKETS 4096
static struct AsyncJob *g_async_live[ASYNC_BUCKETS];
static pthread_rwlock_t g_async_rw = PTHREAD_RWLOCK_INITIALIZER;
static _Atomic uint32_t g_async_live_n; // fast path: no stand-ins, no lookup

static struct AsyncJob *async_lookup(VkPipeline p) {
    if (!atomic_load(&g_async_live_n) || !p) return NULL;
    pthread_rwlock_rdlock(&g_async_rw);
    struct AsyncJob *j = g_async_live[handle_bucket((uint64_t)(uintptr_t)p, ASYNC_BUCKETS)];
    while (j && (VkPipeline)(uintptr_t)j != p) j = j->next_live;
    pthread_rwlock_unlock(&g_async_rw);
    return j;
}

static void async_free_job(struct AsyncJob *j) {
    ar_free(&j->arena);
    pthread_mutex_destroy(&j->m);
    pthread_cond_destroy(&j->cv);
    free(j);
}

static void *async_worker(void *arg) {
    (void)arg;
    pthread_setname_np(pthread_self(), "mali-async-pso");
    for (;;) {
        pthread_mutex_lock(&g_async_qm);
        while (!g_async_qhead) pthread_cond_wait(&g_async_qcv, &g_async_qm);
        struct AsyncJob *j = g_async_qhead;
        g_async_qhead = j->next_job;
        if (!g_async_qhead) g_async_qtail = NULL;
        pthread_mutex_unlock(&g_async_qm);

        VkPipeline p = VK_NULL_HANDLE;
        uint64_t t0 = now_ns();
        VkResult r = j->d->fpCreateGraphicsPipelines(j->d->device, j->cache, 1, &j->ci, NULL, &p);
        uint64_t dt = now_ns() - t0;
        if (g_gpu_time) pcstat_add(1, dt);
        if (dt > 1000000000ull)
            fprintf(stderr, "mali-compat: slow pipeline compile %.0f ms (async): %08x %08x %08x\n", dt / 1e6, j->hash[0],
                    j->hash[1], j->hash[2]);
        ar_free(&j->arena); // the copy is no longer needed
        pthread_mutex_lock(&j->m);
        if (r != VK_SUCCESS) { p = VK_NULL_HANDLE; atomic_fetch_add(&g_async_failed, 1); }
        atomic_store(&j->real, p);
        j->result = r;
        j->state = r == VK_SUCCESS ? AJ_DONE : AJ_FAILED;
        bool gone = j->detached && j->destroy;
        pthread_cond_broadcast(&j->cv);
        pthread_mutex_unlock(&j->m);
        if (gone) {
            if (p) j->d->fpDestroyPipeline(j->d->device, p, NULL);
            async_free_job(j);
        }
    }
    return NULL;
}

// Compile one pipeline through the workers. Returns true and sets *out (real pipeline or stand-in); false if the
// create info cannot be copied (the caller compiles it itself).
static bool async_create(struct DeviceData *d, VkPipelineCache cache, const VkGraphicsPipelineCreateInfo *ci,
                         const uint32_t *hash, VkPipeline *out, VkResult *res) {
    struct AsyncJob *j = calloc(1, sizeof *j);
    if (!j) return false;
    if (!async_copy_ci(&j->arena, ci, &j->ci)) { ar_free(&j->arena); free(j); return false; }
    pthread_mutex_init(&j->m, NULL);
    pthread_cond_init(&j->cv, NULL);
    j->d = d;
    j->cache = cache;
    if (hash) memcpy(j->hash, hash, sizeof j->hash);
    pthread_mutex_lock(&g_async_qm);
    if (!g_async_started) {
        for (uint32_t i = 0; i < g_async_threads; i++) {
            pthread_t t;
            if (pthread_create(&t, NULL, async_worker, NULL) == 0) pthread_detach(t);
        }
        g_async_started = true;
    }
    if (g_async_qtail) g_async_qtail->next_job = j; else g_async_qhead = j;
    g_async_qtail = j;
    pthread_cond_signal(&g_async_qcv);
    pthread_mutex_unlock(&g_async_qm);
    atomic_fetch_add(&g_async_jobs, 1);

    struct timespec until;
    clock_gettime(CLOCK_REALTIME, &until);
    until.tv_nsec += (long)g_async_ms * 1000000L;
    until.tv_sec += until.tv_nsec / 1000000000L;
    until.tv_nsec %= 1000000000L;
    pthread_mutex_lock(&j->m);
    while (j->state == AJ_PENDING && pthread_cond_timedwait(&j->cv, &j->m, &until) == 0) {}
    if (j->state != AJ_PENDING) {
        // finished in time: the real pipeline, as without the switch
        VkPipeline p = atomic_load(&j->real);
        VkResult r = j->result;
        pthread_mutex_unlock(&j->m);
        async_free_job(j);
        *out = p;
        *res = r;
        return true;
    }
    j->detached = true;
    pthread_rwlock_wrlock(&g_async_rw);
    uint32_t b = handle_bucket((uint64_t)(uintptr_t)j, ASYNC_BUCKETS);
    j->next_live = g_async_live[b];
    g_async_live[b] = j;
    atomic_fetch_add(&g_async_live_n, 1);
    pthread_rwlock_unlock(&g_async_rw);
    pthread_mutex_unlock(&j->m);
    atomic_fetch_add(&g_async_pending, 1);
    *out = (VkPipeline)(uintptr_t)j;
    *res = VK_SUCCESS;
    return true;
}

// vkDestroyPipeline for a stand-in: the real pipeline goes with it, now or when its compile ends.
static bool async_destroy(VkPipeline p) {
    struct AsyncJob *j = async_lookup(p);
    if (!j) return false;
    pthread_rwlock_wrlock(&g_async_rw);
    struct AsyncJob **pp = &g_async_live[handle_bucket((uint64_t)(uintptr_t)p, ASYNC_BUCKETS)];
    while (*pp && *pp != j) pp = &(*pp)->next_live;
    if (*pp) { *pp = j->next_live; atomic_fetch_sub(&g_async_live_n, 1); }
    pthread_rwlock_unlock(&g_async_rw);
    pthread_mutex_lock(&j->m);
    if (j->state == AJ_PENDING) { j->destroy = true; pthread_mutex_unlock(&j->m); return true; }
    VkPipeline real = atomic_load(&j->real);
    pthread_mutex_unlock(&j->m);
    if (real) j->d->fpDestroyPipeline(j->d->device, real, NULL);
    async_free_job(j);
    return true;
}

static VKAPI_ATTR void VKAPI_CALL compat_DestroyPipeline(VkDevice device, VkPipeline pipeline, const VkAllocationCallbacks *a) {
    if (g_rt_pipeline && rt_destroy_pipeline(get_device_data(device), pipeline, a)) return;
    if (async_destroy(pipeline)) return;
    get_device_data(device)->fpDestroyPipeline(device, pipeline, a);
}

static VKAPI_ATTR VkResult VKAPI_CALL compat_CreateGraphicsPipelines(
    VkDevice device, VkPipelineCache pipelineCache, uint32_t createInfoCount,
    const VkGraphicsPipelineCreateInfo *pCreateInfos, const VkAllocationCallbacks *pAllocator,
    VkPipeline *pPipelines) {
    struct DeviceData *d = get_device_data(device);

    VkGraphicsPipelineCreateInfo *modified = (VkGraphicsPipelineCreateInfo *)malloc(
        createInfoCount * sizeof(VkGraphicsPipelineCreateInfo));
    if (!modified) return VK_ERROR_OUT_OF_HOST_MEMORY;
    memcpy(modified, pCreateInfos, createInfoCount * sizeof(VkGraphicsPipelineCreateInfo));

    VkPipelineRasterizationStateCreateInfo *rast_copies = (VkPipelineRasterizationStateCreateInfo *)malloc(
        createInfoCount * sizeof(VkPipelineRasterizationStateCreateInfo));

    // GPU timing: the stage hashes, taken before the clip/cull patch below rewrites inline SPIR-V
    uint32_t (*gt_hash)[6] = (g_gpu_time || g_dump_hash_count || g_ct) ? calloc(createInfoCount, sizeof *gt_hash) : NULL;
    uint32_t *gt_n = gt_hash ? calloc(createInfoCount, sizeof *gt_n) : NULL;
    for (uint32_t i = 0; gt_n && i < createInfoCount; i++)
        gt_n[i] = stage_hashes(pCreateInfos[i].pStages, pCreateInfos[i].stageCount, gt_hash[i], gt_hash[i] + 3);

    // KZ1/KZ2 (HZD statics): pipeline flags for the per-pass eq/dw/vsw counters (GPU timing), the depth EQUAL
    // replacement (MALI_COMPAT_DEPTH_EQUAL) and the VS-stage store scan, all on the app's state and SPIR-V (before the
    // patches below). The static depth state only: vkd3d keeps test, write and op static. [keeper, Claude Opus 5.5]
    uint8_t *pf = g_gpu_time ? calloc(createInfoCount, 1) : NULL;
    VkPipelineDepthStencilStateCreateInfo *ds_copies = g_depth_equal_on ? malloc(createInfoCount * sizeof *ds_copies) : NULL;
    for (uint32_t i = 0; (pf || ds_copies || g_vs_scan) && i < createInfoCount; i++) {
        const VkGraphicsPipelineCreateInfo *ci = &pCreateInfos[i];
        const VkPipelineDepthStencilStateCreateInfo *ds = ci->pDepthStencilState;
        uint8_t f = 0;
        if (ds && ds->depthTestEnable) f |= ds->depthWriteEnable ? PF_DW : ds->depthCompareOp == VK_COMPARE_OP_EQUAL ? PF_EQ : 0;
        if (f & PF_EQ) {
            atomic_fetch_add(&g_deq_pipes, 1);
            if (ds_copies) {
                ds_copies[i] = *ds;
                ds_copies[i].depthCompareOp = g_depth_equal_op;
                modified[i].pDepthStencilState = &ds_copies[i];
                uint64_t k = atomic_fetch_add(&g_deq_replaced, 1) + 1;
                if (k == 1 || k == 10 || k == 100 || k == 1000 || k == 10000 || k == 100000)
                    fprintf(stderr, "mali-compat: DEPTH_EQUAL: %llu pipelines with depth EQUAL and no depth write now use op %d\n",
                            (unsigned long long)k, g_depth_equal_op);
            }
        }
        for (uint32_t s = 0; g_vs_scan && s < ci->stageCount; s++) {
            if (!(ci->pStages[s].stage & (VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_GEOMETRY_BIT |
                                          VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT | VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT)))
                continue;
            const VkShaderModuleCreateInfo *sm = find_struct(ci->pStages[s].pNext, VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO);
            struct HashNode mod;
            uint32_t cnt[4];
            if (sm && spv_vs_stores(sm->pCode, sm->codeSize, cnt)) {
                f |= PF_VSW;
                vs_store_note(spv_hash(sm->pCode, sm->codeSize), cnt, true);
            } else if (!sm && g_gpu_time && hash_get(g_module_hash, (uint64_t)ci->pStages[s].module, &mod)) {
                f |= mod.flags & PF_VSW;
            }
        }
        if (f & PF_VSW) atomic_fetch_add(&g_vsw_pipes, 1);
        // a linked pipeline (VK_EXT_graphics_pipeline_library) inherits its libraries' flags
        const VkPipelineLibraryCreateInfoKHR *lib = pf ? find_struct(ci->pNext, VK_STRUCTURE_TYPE_PIPELINE_LIBRARY_CREATE_INFO_KHR) : NULL;
        for (uint32_t l = 0; lib && l < lib->libraryCount; l++) {
            struct HashNode h;
            if (hash_get(g_pipeline_hash, (uint64_t)lib->pLibraries[l], &h)) f |= h.flags;
        }
        if (pf) pf[i] = f;
    }

    // FragDepth without a depth/stencil attachment (MALI_COMPAT_FRAGDEPTH_FIX): own stage arrays, patched modules
    VkPipelineShaderStageCreateInfo *fd_stages = NULL;
    VkShaderModule fd_modules[16];
    uint32_t fd_module_count = 0;
    VkShaderModuleCreateInfo fd_inline[16];
    uint32_t fd_inline_count = 0;
    uint32_t *fd_codes[32];
    uint32_t fd_code_count = 0;
    if (g_fragdepth_fix) {
        uint32_t total = 0;
        for (uint32_t i = 0; i < createInfoCount; i++) total += pCreateInfos[i].stageCount;
        fd_stages = total ? malloc(total * sizeof *fd_stages) : NULL;
        for (uint32_t i = 0, base = 0; fd_stages && i < createInfoCount; base += pCreateInfos[i].stageCount, i++) {
            const VkPipelineRenderingCreateInfo *ri = find_struct(pCreateInfos[i].pNext, VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO);
            if (pCreateInfos[i].renderPass || !ri || ri->depthAttachmentFormat || ri->stencilAttachmentFormat) continue;
            bool changed = false;
            memcpy(fd_stages + base, pCreateInfos[i].pStages, pCreateInfos[i].stageCount * sizeof *fd_stages);
            for (uint32_t st = 0; st < pCreateInfos[i].stageCount; st++) {
                VkPipelineShaderStageCreateInfo *sg = &fd_stages[base + st];
                if (sg->stage != VK_SHADER_STAGE_FRAGMENT_BIT || fd_code_count >= 32) continue;
                const VkShaderModuleCreateInfo *sm = find_struct(sg->pNext, VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO);
                const struct CodeNode *cn = sm ? NULL : fragdepth_module_get(sg->module);
                const uint32_t *code = sm ? sm->pCode : cn ? cn->code : NULL;
                size_t bytes = sm ? sm->codeSize : cn ? cn->bytes : 0;
                size_t out_bytes = 0;
                uint32_t *patched = code ? spv_strip_fragdepth(code, bytes, &out_bytes) : NULL;
                if (!patched) continue;
                fd_codes[fd_code_count++] = patched;
                if (sm && fd_inline_count < 16 && sg->pNext == (const void *)sm) {
                    // inline SPIR-V at the head of the stage chain: replaced by a copy (one such struct per chain)
                    fd_inline[fd_inline_count] = *sm;
                    fd_inline[fd_inline_count].pCode = patched;
                    fd_inline[fd_inline_count].codeSize = out_bytes;
                    sg->pNext = &fd_inline[fd_inline_count++];
                    changed = true;
                } else if (!sm && fd_module_count < 16) {
                    VkShaderModuleCreateInfo mci = {VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, NULL, 0, out_bytes, patched};
                    if (d->fpCreateShaderModule(device, &mci, NULL, &fd_modules[fd_module_count]) == VK_SUCCESS) {
                        sg->module = fd_modules[fd_module_count++];
                        changed = true;
                    }
                }
            }
            if (changed) {
                modified[i].pStages = fd_stages + base;
                static _Atomic int logged;
                if (atomic_fetch_add(&logged, 1) < 8)
                    fprintf(stderr, "mali-compat: FragDepth write removed from a fragment shader (pipeline without depth/stencil attachment)\n");
            }
        }
    }

    // Also check for inline VkShaderModuleCreateInfo in stage pNext (VK_KHR_maintenance5)
    // Stream output (MALI_COMPAT_XFB_VS): pipelines with DXVK's passthrough capture GS and no fragment shader are built
    // from a vertex shader that captures by itself. Inline SPIR-V only (DXVK passes it that way with maintenance5).
    VkPipelineShaderStageCreateInfo *xv_stages = NULL;
    VkShaderModuleCreateInfo *xv_inline = NULL;
    uint32_t **xv_codes = NULL;
    if (g_xfb_vs) {
        xv_stages = calloc((size_t)createInfoCount * 8, sizeof *xv_stages);
        xv_inline = calloc(createInfoCount, sizeof *xv_inline);
        xv_codes = calloc(createInfoCount, sizeof *xv_codes);
        for (uint32_t i = 0; xv_stages && xv_inline && xv_codes && i < createInfoCount; i++) {
            const VkGraphicsPipelineCreateInfo *ci = &modified[i];
            if (ci->stageCount > 8 || (ci->flags & VK_PIPELINE_CREATE_LIBRARY_BIT_KHR) ||
                find_struct(ci->pNext, VK_STRUCTURE_TYPE_PIPELINE_LIBRARY_CREATE_INFO_KHR)) continue;
            int vsi = -1, gsi = -1;
            bool other = false;
            for (uint32_t st = 0; st < ci->stageCount; st++) {
                VkShaderStageFlagBits sb = ci->pStages[st].stage;
                if (sb == VK_SHADER_STAGE_VERTEX_BIT) vsi = st;
                else if (sb == VK_SHADER_STAGE_GEOMETRY_BIT) gsi = st;
                else other = true; // fragment or tessellation: keep the pipeline as it is
            }
            if (gsi < 0) continue;
            const VkShaderModuleCreateInfo *vsm = vsi < 0 ? NULL : find_struct(ci->pStages[vsi].pNext, VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO);
            const VkShaderModuleCreateInfo *gsm = find_struct(ci->pStages[gsi].pNext, VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO);
            const char *why = vsi < 0 ? "no VS" : other ? "fragment/tessellation stage" : !vsm || !gsm ? "shader module, not inline"
                            : ci->pStages[vsi].pNext != (const void *)vsm ? "VS pNext chain" : NULL;
            size_t bytes = 0;
            uint32_t *code = why ? NULL : xfb_vs_from_gs(vsm->pCode, vsm->codeSize, gsm->pCode, gsm->codeSize, &bytes);
            if (!code) {
                // log the capture GS pipelines left alone (once per GS hash), so missed shapes can be found
                struct SpvMod gm;
                bool xfb = gsm && spv_mod_parse(gsm->pCode, gsm->codeSize, &gm) && (spv_mod_free(&gm), gm.xfb);
                static uint32_t seen[64];
                static _Atomic uint32_t nseen;
                uint32_t h = gsm ? spv_hash(gsm->pCode, gsm->codeSize) : 0, k = 0, ns = atomic_load(&nseen);
                while (k < ns && k < 64 && seen[k] != h) k++;
                if ((xfb || !gsm) && k == ns && ns < 64) {
                    seen[atomic_fetch_add(&nseen, 1) % 64] = h;
                    fprintf(stderr, "mali-compat: stream output: GS %08x kept (%s), stages %u, in points %d, out points %d, one vertex %d\n", h,
                            why ? why : "shape not handled", ci->stageCount, gsm ? gm.in_points : -1, gsm ? gm.out_points : -1,
                            gsm ? gm.one_vertex : -1);
                    // both shaders to /root/mali-xfb-<gs hash>-{vs,gs}.spv, to build the rewrite for this shape
                    for (int w = 0; w < 2; w++) {
                        const VkShaderModuleCreateInfo *m = w ? gsm : vsm;
                        char path[64];
                        snprintf(path, sizeof path, "/root/mali-xfb-%08x-%s.spv", h, w ? "gs" : "vs");
                        FILE *f = m ? fopen(path, "wb") : NULL;
                        if (f) { fwrite(m->pCode, 1, m->codeSize, f); fclose(f); }
                    }
                }
                continue;
            }
            xv_codes[i] = code;
            xv_inline[i] = *vsm;
            xv_inline[i].pCode = code;
            xv_inline[i].codeSize = bytes;
            VkPipelineShaderStageCreateInfo *ns = xv_stages + (size_t)i * 8;
            uint32_t k = 0;
            for (uint32_t st = 0; st < ci->stageCount; st++) {
                if ((int)st == gsi) continue;
                ns[k] = ci->pStages[st];
                if ((int)st == vsi) ns[k].pNext = &xv_inline[i];
                k++;
            }
            modified[i].pStages = ns;
            modified[i].stageCount = k;
            if (atomic_fetch_add(&g_xfb_vs_pipes, 1) == 0)
                fprintf(stderr, "mali-compat: stream output: pipeline built without the passthrough geometry shader\n");
        }
    }

    for (uint32_t i = 0; i < createInfoCount; i++) {
        if (modified[i].pRasterizationState) {
            rast_copies[i] = *modified[i].pRasterizationState;
            if (rast_copies[i].polygonMode != VK_POLYGON_MODE_FILL) {
                rast_copies[i].polygonMode = VK_POLYGON_MODE_FILL;
                modified[i].pRasterizationState = &rast_copies[i];
            }
        }

        // Patch stages if maintenance5 inline shader module info is present
        if (modified[i].pStages && modified[i].stageCount > 0) {
            for (uint32_t s = 0; s < modified[i].stageCount; s++) {
                VkBaseOutStructure *header = (VkBaseOutStructure *)modified[i].pStages[s].pNext;
                while (header) {
                    if (header->sType == VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO) {
                        VkShaderModuleCreateInfo *sm = (VkShaderModuleCreateInfo *)header;
                        size_t patched_bytes = 0;
                        uint32_t *patched = compat_patch_spirv(sm->pCode, sm->codeSize, &patched_bytes);
                        if (patched) {
                            sm->pCode = patched;
                            sm->codeSize = patched_bytes;
                        }
                    }
                    header = header->pNext;
                }
            }
        }
    }

    // Zero divisors -> UINT32_MAX for the call (see compat_GetPhysicalDeviceFeatures2); the app's arrays are restored.
    VkVertexInputBindingDivisorDescription *zero_div[64];
    uint32_t zero_div_count = 0;
    if (d->zero_divisor_emulated) {
        for (uint32_t i = 0; i < createInfoCount; i++) {
            if (!modified[i].pVertexInputState) continue;
            const VkPipelineVertexInputDivisorStateCreateInfo *dv = find_struct(modified[i].pVertexInputState->pNext,
                VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_DIVISOR_STATE_CREATE_INFO);
            for (uint32_t k = 0; dv && k < dv->vertexBindingDivisorCount && zero_div_count < 64; k++)
                if (dv->pVertexBindingDivisors[k].divisor == 0) {
                    zero_div[zero_div_count] = (VkVertexInputBindingDivisorDescription *)&dv->pVertexBindingDivisors[k];
                    zero_div[zero_div_count++]->divisor = UINT32_MAX;
                }
        }
    }
    if (g_gpu_time)
        for (uint32_t i = 0; i < createInfoCount; i++) {
            const VkPipelineVertexInputDivisorStateCreateInfo *dv = modified[i].pVertexInputState ? find_struct(
                modified[i].pVertexInputState->pNext, VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_DIVISOR_STATE_CREATE_INFO) : NULL;
            bool div = false;
            for (uint32_t k = 0; dv && k < dv->vertexBindingDivisorCount; k++) div |= dv->pVertexBindingDivisors[k].divisor != 1;
            if (div) atomic_fetch_add(&g_div_pipes, 1);
        }
    dump_inflight_pipeline(createInfoCount, modified);
    uint64_t t0 = g_gpu_time ? now_ns() : 0;
    VkResult res = VK_SUCCESS;
    if (g_async_ms) {
        // one by one through the workers (MALI_COMPAT_ASYNC_PSO); what cannot go there is compiled here
        VkPipelineCache pc = pc_for(d, pipelineCache, createInfoCount);
        for (uint32_t i = 0; i < createInfoCount; i++) {
            VkResult r1 = VK_SUCCESS;
            if (!async_eligible(&modified[i]) || !async_create(d, pc, &modified[i], gt_hash ? gt_hash[i] : NULL, &pPipelines[i], &r1))
                r1 = d->fpCreateGraphicsPipelines(device, pc, 1, &modified[i], pAllocator, &pPipelines[i]);
            if (r1 != VK_SUCCESS && res == VK_SUCCESS) res = r1;
        }
    } else {
        res = d->fpCreateGraphicsPipelines(device, pc_for(d, pipelineCache, createInfoCount), createInfoCount, modified, pAllocator, pPipelines);
    }
    if (g_gpu_time && !g_async_ms) {
        uint64_t dt = now_ns() - t0;
        pcstat_add(createInfoCount, dt);
        // compiles over a second stall the game (DXVK compiles on its CS thread): name their shaders
        for (uint32_t i = 0; dt > 1000000000ull && gt_hash && i < createInfoCount; i++) {
            fprintf(stderr, "mali-compat: slow pipeline compile %.0f ms (%u in the call):", dt / 1e6, createInfoCount);
            for (uint32_t k = 0; k < gt_n[i]; k++) fprintf(stderr, " 0x%x:%08x", gt_hash[i][3 + k], gt_hash[i][k]);
            fprintf(stderr, "%s\n", modified[i].stageCount != pCreateInfos[i].stageCount ? " (GS dropped)" : "");
            // the stages as compiled (after the layer's rewrites) to /root/mali-slow-<hash>.spv, once per hash
            for (uint32_t k = 0; k < modified[i].stageCount; k++) {
                const VkShaderModuleCreateInfo *sm = find_struct(modified[i].pStages[k].pNext, VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO);
                if (!sm) continue;
                char path[64];
                snprintf(path, sizeof path, "/root/mali-slow-%08x.spv", spv_hash(sm->pCode, sm->codeSize));
                if (access(path, F_OK) == 0) continue;
                FILE *f = fopen(path, "wb");
                if (f) { fwrite(sm->pCode, 1, sm->codeSize, f); fclose(f); }
            }
        }
    }
    dump_inflight_done();
    for (uint32_t k = 0; k < zero_div_count; k++) zero_div[k]->divisor = 0;
    if (res != VK_SUCCESS) {
        fprintf(stderr, "mali-compat: vkCreateGraphicsPipelines FAILED: %d\n", res);
    }
    if (g_log_pipelines)
        for (uint32_t i = 0; i < createInfoCount; i++) log_pipeline(&modified[i], res);
    if ((g_gpu_time || g_ct) && gt_n && res == VK_SUCCESS && pPipelines)
        for (uint32_t i = 0; i < createInfoCount; i++)
            if (pPipelines[i]) hash_put(g_pipeline_hash, (uint64_t)pPipelines[i], gt_hash[i], gt_hash[i] + 3, gt_n[i], pf ? pf[i] : 0);
    free(gt_hash);
    free(gt_n);
    free(pf);
    free(ds_copies);
    for (uint32_t k = 0; k < fd_module_count; k++) d->fpDestroyShaderModule(device, fd_modules[k], NULL);
    for (uint32_t k = 0; k < fd_code_count; k++) free(fd_codes[k]);
    free(fd_stages);
    for (uint32_t k = 0; xv_codes && k < createInfoCount; k++) free(xv_codes[k]);
    free(xv_codes);
    free(xv_inline);
    free(xv_stages);
    free(rast_copies);
    free(modified);
    return res;
}

// Only hooked for the GPU timing (and MALI_COMPAT_DUMP_HASH): records the compute shader's hash.
static VKAPI_ATTR VkResult VKAPI_CALL compat_CreateComputePipelines(
    VkDevice device, VkPipelineCache pipelineCache, uint32_t createInfoCount,
    const VkComputePipelineCreateInfo *pCreateInfos, const VkAllocationCallbacks *pAllocator, VkPipeline *pPipelines) {
    struct DeviceData *d = get_device_data(device);
    // wls-clamp-1: a compute stage with inline SPIR-V (VkShaderModuleCreateInfo in its pNext) is clamped like a module;
    // the app's create info is patched for the call only and restored right after it.
    VkShaderModuleCreateInfo *wls_sm[8] = {0};
    const uint32_t *wls_code[8];
    size_t wls_size[8];
    uint32_t *wls_new[8] = {0};
    for (uint32_t i = 0; (g_wls_clamp || g_wls_fix) && i < createInfoCount && i < 8; i++) {
        VkShaderModuleCreateInfo *sm = (VkShaderModuleCreateInfo *)find_struct(pCreateInfos[i].stage.pNext, VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO);
        size_t ob = 0, fb = 0;
        // wls-fix-1 first (hash of the app's code), then the clamp on its result (hash of the app's code as well)
        uint32_t *wf = g_wls_fix && sm && sm->pCode ? wls_fix_module(sm->pCode, sm->codeSize, sm->pCode, sm->codeSize, &fb) : NULL;
        if (wf && g_wls_clamp) {
            uint32_t *wc = wls_clamp_module(sm->pCode, sm->codeSize, wf, fb, &ob);
            if (wc) { free(wf); wf = wc; fb = ob; }
        }
        if (wf) ob = fb;
        if (sm && sm->pCode && (wls_new[i] = wf ? wf : g_wls_clamp ? wls_clamp_module(sm->pCode, sm->codeSize, sm->pCode, sm->codeSize, &ob) : NULL)) {
            wls_sm[i] = sm;
            wls_code[i] = sm->pCode;
            wls_size[i] = sm->codeSize;
            sm->pCode = wls_new[i];
            sm->codeSize = ob;
        }
    }
    dump_inflight_compute(createInfoCount, pCreateInfos);
    // cover-trim: listed compute shaders get DISPATCH_BASE (the trim splits their dispatches into vkCmdDispatchBase
    // calls). The app's create infos are copied; a VkPipelineCreateFlags2CreateInfo in pNext is patched and restored.
    VkComputePipelineCreateInfo *ct_ci = NULL;
    VkPipelineCreateFlags2CreateInfoKHR *ct_f2[8] = {0};
    VkPipelineCreateFlags2KHR ct_f2_saved[8];
    for (uint32_t i = 0; g_ct && g_ct_cs_n && !ct_ci && i < createInfoCount; i++) {
        uint32_t h[1], st[1];
        if (stage_hashes(&pCreateInfos[i].stage, 1, h, st) && ct_cs_listed(h[0])) ct_ci = malloc(createInfoCount * sizeof *ct_ci);
    }
    if (ct_ci) {
        for (uint32_t i = 0; i < createInfoCount; i++) {
            uint32_t h[1], st[1];
            ct_ci[i] = pCreateInfos[i];
            if (!stage_hashes(&pCreateInfos[i].stage, 1, h, st) || !ct_cs_listed(h[0])) continue;
            ct_ci[i].flags |= VK_PIPELINE_CREATE_DISPATCH_BASE_BIT;
            VkPipelineCreateFlags2CreateInfoKHR *f2 = (VkPipelineCreateFlags2CreateInfoKHR *)find_struct(pCreateInfos[i].pNext,
                VK_STRUCTURE_TYPE_PIPELINE_CREATE_FLAGS_2_CREATE_INFO_KHR);
            if (f2 && i < 8) { ct_f2[i] = f2; ct_f2_saved[i] = f2->flags; f2->flags |= VK_PIPELINE_CREATE_2_DISPATCH_BASE_BIT_KHR; }
        }
    }
    VkResult res = d->fpCreateComputePipelinesReal(device, pc_for(d, pipelineCache, createInfoCount), createInfoCount,
                                                   ct_ci ? ct_ci : pCreateInfos, pAllocator, pPipelines);
    for (uint32_t i = 0; i < 8; i++) if (ct_f2[i]) ct_f2[i]->flags = ct_f2_saved[i];
    free(ct_ci);
    // The failure log only where c002fb05 had this hook (not for the disp-barrier hash fill alone: SYNC_CTL / DISP_BARRIER
    // must not add FAILED lines / .spv dumps, e.g. on VK_PIPELINE_COMPILE_REQUIRED). [game/re_village keeper] 2026-11-14
    bool fail_log = g_gpu_time || g_dump_hash_count || g_pcache_on || g_dump_inflight || g_dc_on || g_wls_clamp || g_wls_fix ||
                    g_ct || g_fail_log;
    if (res != VK_SUCCESS && fail_log) { // [game/doom_eternal, Claude Opus 5.5] 2026-11-06: compute failures were silent
        for (uint32_t i = 0; i < createInfoCount; i++) {
            const VkShaderModuleCreateInfo *sm = find_struct(pCreateInfos[i].stage.pNext, VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO);
            uint32_t fh[1], fst[1];
            uint32_t fn = stage_hashes(&pCreateInfos[i].stage, 1, fh, fst);
            fprintf(stderr, "mali-compat: vkCreateComputePipelines FAILED: %d (info %u/%u, module %p, spv %08x, entry %s, flags 0x%x)\n",
                    res, i, createInfoCount, (void *)pCreateInfos[i].stage.module, fn ? fh[0] : 0,
                    pCreateInfos[i].stage.pName ? pCreateInfos[i].stage.pName : "?", (unsigned)pCreateInfos[i].flags);
            if (sm) fail_spv_dump(sm->pCode, sm->codeSize, spv_hash(sm->pCode, sm->codeSize));
        }
    }
    dump_inflight_done();
    for (uint32_t i = 0; i < 8; i++)
        if (wls_sm[i]) {
            wls_sm[i]->pCode = wls_code[i];
            wls_sm[i]->codeSize = wls_size[i];
            free(wls_new[i]);
        }
    for (uint32_t i = 0; i < createInfoCount; i++) {
        uint32_t hash[1], stage[1];
        uint32_t n = stage_hashes(&pCreateInfos[i].stage, 1, hash, stage);
        // Every pipeline that exists gets its entry, whatever res says (a partial multi-create failure or
        // VK_PIPELINE_COMPILE_REQUIRED still returns valid handles, and a reused handle must not keep an older hash).
        if ((g_gpu_time || g_ct || g_db_hash) && pPipelines && pPipelines[i])
            hash_put(g_pipeline_hash, (uint64_t)pPipelines[i], hash, stage, n, 0);
        if (g_dc_on && res == VK_SUCCESS && pPipelines && pPipelines[i])
            dc_pipe_create(pPipelines[i], n ? hash[0] : 0, pCreateInfos[i].layout);
    }
    return res;
}


static VKAPI_ATTR void VKAPI_CALL compat_CmdSetVertexInputEXT(
    VkCommandBuffer commandBuffer, uint32_t bindingCount, const VkVertexInputBindingDescription2EXT *pBindings,
    uint32_t attributeCount, const VkVertexInputAttributeDescription2EXT *pAttributes) {
    struct DeviceData *d = get_device_data_by_cmd(commandBuffer);
    VkVertexInputBindingDescription2EXT stack[32];
    const VkVertexInputBindingDescription2EXT *b = pBindings;
    if (g_gpu_time) {
        bool div = false;
        for (uint32_t i = 0; i < bindingCount; i++)
            div |= pBindings[i].inputRate == VK_VERTEX_INPUT_RATE_INSTANCE && pBindings[i].divisor != 1;
        if (div) atomic_fetch_add(&g_div_vi, 1);
        tl_div_bound = div;
    }
    if (d->zero_divisor_emulated && bindingCount <= 32) {
        for (uint32_t i = 0; i < bindingCount; i++) {
            stack[i] = pBindings[i];
            if (stack[i].inputRate == VK_VERTEX_INPUT_RATE_INSTANCE && stack[i].divisor == 0) stack[i].divisor = UINT32_MAX;
        }
        b = stack;
    }
    d->fpCmdSetVertexInputEXT(commandBuffer, bindingCount, b, attributeCount, pAttributes);
}

static VKAPI_ATTR void VKAPI_CALL compat_CmdSetPolygonModeEXT(
    VkCommandBuffer commandBuffer, VkPolygonMode polygonMode) {
    (void)commandBuffer;
    (void)polygonMode;
    // Suppress line/point mode, force fill
}

// MALI_COMPAT_DEPTH_EQUAL, dynamic depth compare op (vkd3d 0bb64c56 keeps it static, state.c:3949): EQUAL becomes the
// chosen op. The dynamic depth-write state is not tracked, so every dynamic EQUAL is replaced. [keeper, Claude Opus 5.5]
static VKAPI_ATTR void VKAPI_CALL compat_CmdSetDepthCompareOp(VkCommandBuffer commandBuffer, VkCompareOp op) {
    if (op == VK_COMPARE_OP_EQUAL && g_depth_equal_on) {
        op = g_depth_equal_op;
        if (atomic_fetch_add(&g_deq_dynamic, 1) == 0)
            fprintf(stderr, "mali-compat: DEPTH_EQUAL: dynamic depth compare op EQUAL replaced by %d\n", op);
    }
    get_device_data_by_cmd(commandBuffer)->fpCmdSetDepthCompareOp(commandBuffer, op);
}


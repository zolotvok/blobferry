// SPDX-License-Identifier: GPL-3.0-or-later
// rt_pipeline.h: stage copy, rt_mega_compile, CreateRayTracingPipelines. Included by mali_rt_emu.h.
/* ── Copy Stage Helper for Pipeline & Library Creation ────────────────────── */

static bool rt_copy_stage_from_create_info(const VkPipelineShaderStageCreateInfo *sci,
                                           struct RtStageRecord *out) {
    memset(out, 0, sizeof(*out));
    out->stage = sci->stage;
    out->entry_name = strdup(sci->pName ? sci->pName : "main");

    const uint32_t *spv_words = NULL;
    size_t spv_word_count = 0;

    struct RtShaderModule *rtm = rt_lookup_shader_module(sci->module);
    if (rtm) {
        spv_words = rtm->words;
        spv_word_count = rtm->word_count;
    } else {
        /* Support VK_KHR_maintenance5 inline VkShaderModuleCreateInfo in pNext */
        const VkShaderModuleCreateInfo *smci = (const VkShaderModuleCreateInfo *)
            find_struct(sci->pNext, VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO);
        if (smci && smci->pCode && smci->codeSize >= 20) {
            spv_words = smci->pCode;
            spv_word_count = smci->codeSize / 4;
        }
    }

    if (!spv_words || spv_word_count < 5) {
        free(out->entry_name);
        memset(out, 0, sizeof(*out));
        return false;
    }

    out->word_count = spv_word_count;
    out->words = (uint32_t *)malloc(spv_word_count * sizeof(uint32_t));
    if (!out->words) {
        free(out->entry_name);
        memset(out, 0, sizeof(*out));
        return false;
    }
    memcpy(out->words, spv_words, spv_word_count * sizeof(uint32_t));

    if (sci->pSpecializationInfo &&
        sci->pSpecializationInfo->mapEntryCount > 0 &&
        sci->pSpecializationInfo->pMapEntries &&
        sci->pSpecializationInfo->pData &&
        sci->pSpecializationInfo->dataSize > 0) {
        uint32_t ec = sci->pSpecializationInfo->mapEntryCount;
        out->spec_entries = (struct RtSpecConstantEntry *)calloc(ec, sizeof(struct RtSpecConstantEntry));
        out->spec_data = (uint8_t *)malloc(sci->pSpecializationInfo->dataSize);
        if (out->spec_entries && out->spec_data) {
            out->spec_entry_count = ec;
            out->spec_data_size = sci->pSpecializationInfo->dataSize;
            memcpy(out->spec_data, sci->pSpecializationInfo->pData, out->spec_data_size);
            for (uint32_t i = 0; i < ec; i++) {
                out->spec_entries[i].spec_id = sci->pSpecializationInfo->pMapEntries[i].constantID;
                out->spec_entries[i].offset  = sci->pSpecializationInfo->pMapEntries[i].offset;
                out->spec_entries[i].size    = (uint32_t)sci->pSpecializationInfo->pMapEntries[i].size;
            }
        }
    }

    out->stage_id = rt_compute_stage_id(out->words, out->word_count, out->stage,
                                        out->entry_name, sci->pSpecializationInfo);
    rt_stage_record_count(out);
    return true;
}

static bool rt_clone_stage_record(const struct RtStageRecord *src, struct RtStageRecord *dst) {
    memset(dst, 0, sizeof(*dst));
    dst->stage = src->stage;
    dst->stage_id = src->stage_id;
    dst->word_count = src->word_count;
    dst->entry_name = strdup(src->entry_name ? src->entry_name : "main");
    dst->words = (uint32_t *)malloc(src->word_count * sizeof(uint32_t));
    if (!dst->words || !dst->entry_name) {
        rt_free_stage_record(dst);
        return false;
    }
    memcpy(dst->words, src->words, src->word_count * sizeof(uint32_t));
    if (src->spec_entry_count > 0 && src->spec_entries && src->spec_data) {
        dst->spec_entries = (struct RtSpecConstantEntry *)calloc(src->spec_entry_count, sizeof(*dst->spec_entries));
        dst->spec_data = (uint8_t *)malloc(src->spec_data_size);
        if (dst->spec_entries && dst->spec_data) {
            dst->spec_entry_count = src->spec_entry_count;
            dst->spec_data_size = src->spec_data_size;
            memcpy(dst->spec_entries, src->spec_entries, src->spec_entry_count * sizeof(*dst->spec_entries));
            memcpy(dst->spec_data, src->spec_data, src->spec_data_size);
        }
    }
    rt_stage_record_count(dst);
    return true;
}

/* ── Vulkan Ray Tracing Pipeline Entry Points ─────────────────────────────── */

/* split-1112: build + compile ONE megakernel for `rp` from stages[0..n) (was inline in CRTP step 4). For a normal
 * pipeline stages = rp->stages; for a split child the subset (raygen k + every non-raygen stage, shallow copies of
 * the parent's records: only read while the SPIR-V is built, i.e. before this returns). Sets rp->compute_module,
 * rp->mega_bytes, and either queues the async compile (compile_state 1) or compiles in place (2 / 3). */
static void rt_mega_compile(struct DeviceData *d, VkDevice device, VkPipelineCache pipelineCache,
                            const VkAllocationCallbacks *pAllocator, struct RtPipeline *rp,
                            const struct RtStageRecord *stages, uint32_t n, uint32_t i, uint32_t createInfoCount) {
    size_t mega_words = 0;
    int64_t t_build0 = rt_now_ms();
    rp->mega_stages = n;
    uint32_t *mega_spv = rt_build_megakernel_spirv(stages, n, &mega_words);
    if (mega_spv && mega_words > 5) {
        atomic_fetch_add_explicit(&g_mega_spv_total, (int64_t)(mega_words * 4), memory_order_relaxed);
        atomic_fetch_add_explicit(&g_mega_requested, 1, memory_order_relaxed);
    }
    if (g_debug)
        LOG("rt_compile: SPIR-V built t=%lld ms dt=%lld ms stages=%u words=%zu",
            (long long)rt_now_ms(), (long long)(rt_now_ms() - t_build0), n, mega_words);
    COMPAT_INFO("CRTP [%u/%u]: megakernel SPIR-V built (%zu words, %p)",
                i + 1, createInfoCount, mega_words, (void *)mega_spv);
    if (mega_spv && mega_words > 5) {
        const char *dump_dir = getenv("MALI_COMPAT_DUMP_RT_SPV");
        if (!dump_dir || !dump_dir[0]) dump_dir = "/tmp";
        char path[512];
        uint64_t h = spv_hash(mega_spv, mega_words * 4);
        snprintf(path, sizeof(path), "%s/rt_mega_p%u_%016llx.spv", dump_dir, i, (unsigned long long)h);
        /* round 4 lever: MALI_COMPAT_DUMP_RT_SPV=0|off -> no file (default unchanged: dumped) */
        FILE *fp = rt_env()->dump_spv ? fopen(path, "wb") : NULL;
        if (fp) {
            size_t nw = fwrite(mega_spv, 4, mega_words, fp);
            fclose(fp);
            rt_acct_add(&g_acct_dump, (int64_t)(nw * 4));   /* files are never "freed": count + bytes written */
            COMPAT_INFO("CRTP [%u/%u]: dumped megakernel to %s", i + 1, createInfoCount, path);
        }

        VkShaderModuleCreateInfo smci = {
            .sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
            .codeSize = mega_words * sizeof(uint32_t),
            .pCode    = mega_spv,
        };
        COMPAT_INFO("CRTP [%u/%u]: calling fpCreateShaderModule...", i + 1, createInfoCount);
        /* round 4 (Codex r4b #3): a layer-internal object - created and destroyed with NULL callbacks on every path
         * (the async worker cannot use the app's allocator; before, async compiles destroyed it with NULL anyway) */
        VkResult sm_res = d->fpCreateShaderModule(device, &smci, NULL, &rp->compute_module);
        COMPAT_INFO("CRTP [%u/%u]: fpCreateShaderModule result=%d module=%p", i + 1, createInfoCount, sm_res, (void *)rp->compute_module);
        free(mega_spv);
        if (sm_res == VK_SUCCESS && rp->compute_module) {
            rp->module_app_alloc = false;   /* round 4: created with NULL callbacks */
            rp->mega_bytes = mega_words * sizeof(uint32_t);
            rt_acct_add(&g_acct_mega_mod, (int64_t)rp->mega_bytes);
        }

        if (sm_res == VK_SUCCESS && rp->compute_module) {
            VkPipelineCreateFlags comp_flags = 0;
            if (rp->use_descriptor_buffer) {
                comp_flags |= 0x20000000u /* VK_PIPELINE_CREATE_DESCRIPTOR_BUFFER_BIT_EXT */;
            }
            /* round 4: MALI_COMPAT_RT_STATS=1 and the device enabled pipelineExecutableInfo */
            if (d->rt_stats_ok) comp_flags |= VK_PIPELINE_CREATE_CAPTURE_STATISTICS_BIT_KHR;
            VkPipelineCreateFlags2CreateInfoKHR comp_flags2 = {
                .sType = VK_STRUCTURE_TYPE_PIPELINE_CREATE_FLAGS_2_CREATE_INFO_KHR,
                .flags = (uint64_t)comp_flags,
            };
            VkComputePipelineCreateInfo cpci = {
                .sType  = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
                .pNext  = rp->use_descriptor_buffer ? &comp_flags2 : NULL,
                .flags  = comp_flags,
                .stage  = {
                    .sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                    .stage  = VK_SHADER_STAGE_COMPUTE_BIT,
                    .module = rp->compute_module,
                    .pName  = "main",
                },
                .layout = rp->layout,
            };
            struct RtCompileJob *job = (struct RtCompileJob *)calloc(1, sizeof(*job));
            const int sync_compile = rt_env()->sync_compile;
            if (job) {
                job->device = device; job->rp = rp; job->cpci = cpci; job->f2 = comp_flags2;
                job->fpCreateComputePipelines = d->fpCreateComputePipelines;
                job->fpDestroyPipeline = d->fpDestroyPipeline;
                job->fpDestroyShaderModule = d->fpDestroyShaderModule;
                job->stages = n; job->groups = rp->group_count; job->words = mega_words;
                job->t_req = rt_now_ms();
                if (d->rt_stats_ok) {
                    job->fpGetPipelineExecutablePropertiesKHR = d->fpGetPipelineExecutablePropertiesKHR;
                    job->fpGetPipelineExecutableStatisticsKHR = d->fpGetPipelineExecutableStatisticsKHR;
                }
                /* Codex r4 #4: the worker can only destroy with NULL callbacks (the app may free its allocator
                 * meanwhile), so a module made with the app's allocator is kept until vkDestroyPipeline */
                job->release_module = rt_env()->free_stages != 0 && !rp->module_app_alloc;
            }
            if (g_debug)
                LOG("rt_compile: request t=%lld ms pipe=%p stages=%u groups=%u words=%zu mode=%s",
                    (long long)rt_now_ms(), (void *)rp, n, rp->group_count, mega_words,
                    (sync_compile || !job) ? "sync" : "async");
            bool queued = false;
            if (job && !sync_compile && rt_job_register(job)) {
                rp->compile_state = 1;
                rp->compile_null_alloc = true;
                job->async = true;
                atomic_fetch_add_explicit(&g_mega_compiling, 1, memory_order_relaxed);
                /* the job is fully initialised before the thread starts; the worker owns it */
                if (pthread_create(&rp->compile_thread, NULL, rt_compile_worker, job) == 0) {
                    rp->compile_thread_live = true;
                    queued = true;
                    COMPAT_INFO("CRTP [%u/%u]: megakernel compile queued on a worker (%u stages)",
                                i + 1, createInfoCount, n);
                } else {
                    rt_job_end(job);   /* unpin + unlist; the layout cannot be app-destroyed during CRTP */
                    rp->compile_state = 0;
                    rp->compile_null_alloc = false;
                    job->async = false;
                    atomic_fetch_sub_explicit(&g_mega_compiling, 1, memory_order_relaxed);
                }
            }
            if (!queued) {
                COMPAT_INFO("CRTP [%u/%u]: calling fpCreateComputePipelines...", i + 1, createInfoCount);
                VkPipeline cp = VK_NULL_HANDLE;
                VkResult cp_res = d->fpCreateComputePipelines(device, pipelineCache, 1, &cpci,
                                                              pAllocator, &cp);
                COMPAT_INFO("CRTP [%u/%u]: fpCreateComputePipelines result=%d pipe=%p", i + 1, createInfoCount, cp_res, (void *)cp);
                if (job) {
                    rt_compile_finish(job, cp_res, cp, "sync");
                    free(job);
                } else if (cp_res == VK_SUCCESS) {
                    rp->compute_pipeline = cp;
                    rp->compile_state = 2;
                    atomic_fetch_add_explicit(&g_mega_live, 1, memory_order_relaxed);
                }
                /* round 4 lever MALI_COMPAT_RT_FREE_STAGES=1 (sync compile: created with pAllocator) */
                if (rt_env()->free_stages && rp->compute_module && d->fpDestroyShaderModule) {
                    d->fpDestroyShaderModule(device, rp->compute_module, NULL);
                    rp->compute_module = VK_NULL_HANDLE;
                    if (rp->mega_bytes) rt_acct_sub(&g_acct_mega_mod, (int64_t)rp->mega_bytes);
                }
            }
        } else {
            COMPAT_INFO("RT megakernel vkCreateShaderModule failed (%d)", sm_res);
        }
    } else {
        free(mega_spv);
        COMPAT_INFO("RT megakernel SPIR-V builder failed for %u stages", n);
    }
}

/* split-1112 (MALI_COMPAT_RT_SPLIT=1) [RTX keeper, Claude Opus 5.5]: one megakernel per raygen stage.
 * Why: the W3 80-stage pipeline = 4 raygen + 38 closest-hit + 37 any-hit + 1 miss (switch case counts of the 11-08
 * dumps); every raygen has one TraceRay site, so after the driver inlines everything the kernel holds FOUR copies of
 * the trace loop with all 37 any-hit and 38 closest-hit shaders (~150k SPIR-V instructions inlined vs 39k in the
 * module). One raygen per kernel = one copy (~37k): the register allocation, spills and code size of each kernel
 * only cover the raygen it runs, and each compile is ~1/4 of the size. Semantics are unchanged: a child's megakernel
 * is the same builder output for {raygen k} + all non-raygen stages, and the GPU-side select (rt_split_select_spirv)
 * applies the exact raygen-record checks of the megakernel's main (base != 0, size >= 32, BDA window, magic) before
 * it routes the launch to the child whose raygen id matches (no match = no dispatch, as the megakernel's switch).
 * Returns false (nothing done) when the pipeline is not split: lever off, < split_min stages, < 2 or > RT_SPLIT_MAX
 * distinct raygens, or out of host memory before any child exists. */
static bool rt_split_pipeline(struct DeviceData *d, VkDevice device, VkPipelineCache pipelineCache,
                              const VkAllocationCallbacks *pAllocator, struct RtPipeline *rp,
                              uint32_t i, uint32_t createInfoCount) {
    const struct RtEnv *env = rt_env();
    if (!env->split || rp->stage_count < env->split_min || rp->split_n) return false;
    uint32_t rg_idx[RT_SPLIT_MAX], nrg = 0;
    for (uint32_t s = 0; s < rp->stage_count; s++) {
        if (rp->stages[s].stage != VK_SHADER_STAGE_RAYGEN_BIT_KHR) continue;
        bool dup = false;
        for (uint32_t k = 0; k < nrg && !dup; k++) dup = rp->stages[rg_idx[k]].stage_id == rp->stages[s].stage_id;
        if (dup) continue;
        if (nrg == RT_SPLIT_MAX) {
            LOG("rt_split: pipe=%p stages=%u has more than %d raygen stages -> NOT split", (void *)rp, rp->stage_count,
                RT_SPLIT_MAX);
            return false;
        }
        rg_idx[nrg++] = s;
    }
    if (nrg < 2) return false;
    struct RtStageRecord *sub = (struct RtStageRecord *)malloc(rp->stage_count * sizeof(*sub));
    if (!sub) return false;
    struct RtPipeline *kids[RT_SPLIT_MAX] = { 0 };
    for (uint32_t k = 0; k < nrg; k++) {
        kids[k] = (struct RtPipeline *)calloc(1, sizeof(*kids[k]));
        if (!kids[k]) {
            for (uint32_t j = 0; j < k; j++) free(kids[j]);
            free(sub);
            return false;
        }
    }
    uint32_t nonrg = 0;
    for (uint32_t s = 0; s < rp->stage_count; s++)
        if (rp->stages[s].stage != VK_SHADER_STAGE_RAYGEN_BIT_KHR) nonrg++;
    LOG("rt_split: pipe=%p stages=%u groups=%u -> %u megakernels (1 raygen + %u other stages each)", (void *)rp,
        rp->stage_count, rp->group_count, nrg, nonrg);
    for (uint32_t k = 0; k < nrg; k++) {
        struct RtPipeline *c = kids[k];
        c->magic = rp->magic;
        c->device = device;
        c->is_library = false;
        c->use_descriptor_buffer = rp->use_descriptor_buffer;
        c->layout = rp->layout;               /* pinned by the parent for its whole life; the job pins it too */
        c->max_recursion_depth = rp->max_recursion_depth;
        c->acct_bytes = sizeof(*c);
        c->split_child = true;
        rt_acct_add(&g_acct_pipes, (int64_t)c->acct_bytes);
        rp->split[k] = c;
        rp->split_rgen[k] = rp->stages[rg_idx[k]].stage_id;
        rp->split_n = k + 1;
        uint32_t n = 0;
        sub[n++] = rp->stages[rg_idx[k]];   /* shallow: the builder only reads words/spec data, before return */
        for (uint32_t s = 0; s < rp->stage_count; s++)
            if (rp->stages[s].stage != VK_SHADER_STAGE_RAYGEN_BIT_KHR) sub[n++] = rp->stages[s];
        rt_mega_compile(d, device, pipelineCache, pAllocator, c, sub, n, i, createInfoCount);
        LOG("rt_split: pipe=%p child %u/%u=%p raygen id=0x%08x stages=%u state=%d", (void *)rp, k + 1, nrg, (void *)c,
            rp->split_rgen[k], n, __atomic_load_n(&c->compile_state, __ATOMIC_ACQUIRE));
    }
    free(sub);
    return true;
}

static VKAPI_ATTR VkResult VKAPI_CALL
compat_CreateRayTracingPipelinesKHR(
    VkDevice                                    device,
    VkDeferredOperationKHR                      deferredOperation,
    VkPipelineCache                             pipelineCache,
    uint32_t                                    createInfoCount,
    const VkRayTracingPipelineCreateInfoKHR    *pCreateInfos,
    const VkAllocationCallbacks                *pAllocator,
    VkPipeline                                 *pPipelines)
{
    (void)deferredOperation;
    rt_patch_fex_thunks();
    COMPAT_TRACE("compat_CreateRayTracingPipelinesKHR: count=%u", createInfoCount);
    struct DeviceData *d = get_device_data(device);
    if (!d || !pCreateInfos || !pPipelines) return VK_ERROR_INITIALIZATION_FAILED;

    VkResult final_res = VK_SUCCESS;

    for (uint32_t i = 0; i < createInfoCount; i++) {
        pPipelines[i] = VK_NULL_HANDLE;
        const VkRayTracingPipelineCreateInfoKHR *ci = &pCreateInfos[i];

        /* Check pipeline creation flags (including VkPipelineCreateFlags2CreateInfoKHR) */
        uint64_t flags2 = (uint64_t)ci->flags;
        const VkPipelineCreateFlags2CreateInfoKHR *f2 = (const VkPipelineCreateFlags2CreateInfoKHR *)
            find_struct(ci->pNext, VK_STRUCTURE_TYPE_PIPELINE_CREATE_FLAGS_2_CREATE_INFO_KHR);
        if (f2) flags2 |= f2->flags;

        bool is_library = (flags2 & 0x00000800ULL /* VK_PIPELINE_CREATE_LIBRARY_BIT_KHR */) != 0;
        bool use_desc_buf = (flags2 & 0x20000000ULL /* VK_PIPELINE_CREATE_DESCRIPTOR_BUFFER_BIT_EXT */) != 0;

        /* Count stages & groups from this createInfo + linked libraries */
        uint32_t total_stages = ci->stageCount;
        uint32_t total_groups = ci->groupCount;
        if (ci->pLibraryInfo && ci->pLibraryInfo->libraryCount > 0 && ci->pLibraryInfo->pLibraries) {
            for (uint32_t l = 0; l < ci->pLibraryInfo->libraryCount; l++) {
                struct RtPipeline *lib = rt_lookup_pipeline(ci->pLibraryInfo->pLibraries[l]);
                if (lib) {
                    total_stages += lib->stage_count;
                    total_groups += lib->group_count;
                    if (lib->use_descriptor_buffer) use_desc_buf = true;
                }
            }
        }

        COMPAT_INFO("CRTP [%u/%u]: stages=%u groups=%u layout=%p libInfo=%p",
                    i + 1, createInfoCount, ci->stageCount, ci->groupCount, (void *)ci->layout, (void *)ci->pLibraryInfo);

        struct RtPipeline *rp = (struct RtPipeline *)calloc(1, sizeof(*rp));
        if (!rp) {
            final_res = VK_ERROR_OUT_OF_HOST_MEMORY;
            continue;
        }
        rp->magic = 0x5254504Cu; /* 'RTPL' */
        rp->device = device;
        rp->is_library = is_library;
        rp->use_descriptor_buffer = use_desc_buf;
        rp->layout = ci->layout;
        if (!is_library && ci->layout) {
            /* lead-1110c (Codex 1110-212923 :4837): no pin = no pipeline (an unpinned one could later record with a
             * destroyed layout). Nothing else is allocated for rp yet. */
            rp->layout_pinned = rt_pin_pipeline_layout(device, ci->layout);
            if (!rp->layout_pinned) {
                LOG("CRTP [%u/%u]: out of host memory pinning layout %p -> VK_ERROR_OUT_OF_HOST_MEMORY",
                    i + 1, createInfoCount, (void *)ci->layout);
                free(rp);
                final_res = VK_ERROR_OUT_OF_HOST_MEMORY;
                continue;
            }
        }
        rp->max_recursion_depth = ci->maxPipelineRayRecursionDepth ? ci->maxPipelineRayRecursionDepth : 1;

        if (total_stages > 0) {
            rp->stages = (struct RtStageRecord *)calloc(total_stages, sizeof(*rp->stages));
        }
        if (total_groups > 0) {
            rp->groups = (struct RtGroupHandle *)calloc(total_groups, sizeof(*rp->groups));
        }
        rp->acct_bytes = sizeof(*rp) + (rp->groups ? total_groups * sizeof(*rp->groups) : 0);
        rt_acct_add(&g_acct_pipes, (int64_t)rp->acct_bytes);

        /* 1. Copy local stages */
        bool stage_ok = true;
        for (uint32_t s = 0; s < ci->stageCount; s++) {
            COMPAT_INFO("CRTP [%u/%u]:   stage %u: stage=0x%x module=%p pName=%s spec=%p",
                        i + 1, createInfoCount, s, ci->pStages[s].stage, (void *)ci->pStages[s].module,
                        ci->pStages[s].pName ? ci->pStages[s].pName : "(null)",
                        (void *)ci->pStages[s].pSpecializationInfo);
            if (!rt_copy_stage_from_create_info(&ci->pStages[s], &rp->stages[rp->stage_count])) {
                COMPAT_INFO("CRTP [%u/%u]:     stage %u copy FAILED", i + 1, createInfoCount, s);
                stage_ok = false;
                break;
            }
            rp->stage_count++;
        }
        COMPAT_INFO("CRTP [%u/%u]: copied %u stages, stage_ok=%d", i + 1, createInfoCount, rp->stage_count, (int)stage_ok);

        /* 2. Build local group handles from ci->pGroups */
        for (uint32_t g = 0; g < ci->groupCount && stage_ok; g++) {
            const VkRayTracingShaderGroupCreateInfoKHR *gi = &ci->pGroups[g];
            struct RtGroupHandle *gh = &rp->groups[rp->group_count++];
            memset(gh, 0, sizeof(*gh));
            gh->magic = RT_GROUP_HANDLE_MAGIC;
            gh->type  = (uint32_t)gi->type;
            if (gi->generalShader != VK_SHADER_UNUSED_KHR && gi->generalShader < ci->stageCount) {
                gh->general_id = rp->stages[gi->generalShader].stage_id;
            }
            if (gi->closestHitShader != VK_SHADER_UNUSED_KHR && gi->closestHitShader < ci->stageCount) {
                gh->rchit_id = rp->stages[gi->closestHitShader].stage_id;
            }
            if (gi->anyHitShader != VK_SHADER_UNUSED_KHR && gi->anyHitShader < ci->stageCount) {
                gh->rahit_id = rp->stages[gi->anyHitShader].stage_id;
            }
            if (gi->intersectionShader != VK_SHADER_UNUSED_KHR && gi->intersectionShader < ci->stageCount) {
                gh->rint_id = rp->stages[gi->intersectionShader].stage_id;
            }
        }

        /* 3. Append stages and group handles from linked pipeline libraries */
        if (ci->pLibraryInfo && ci->pLibraryInfo->libraryCount > 0 && ci->pLibraryInfo->pLibraries && stage_ok) {
            for (uint32_t l = 0; l < ci->pLibraryInfo->libraryCount; l++) {
                struct RtPipeline *lib = rt_lookup_pipeline(ci->pLibraryInfo->pLibraries[l]);
                if (!lib) continue;
                for (uint32_t s = 0; s < lib->stage_count; s++) {
                    if (!rt_clone_stage_record(&lib->stages[s], &rp->stages[rp->stage_count])) {
                        stage_ok = false;
                        break;
                    }
                    rp->stage_count++;
                }
                for (uint32_t g = 0; g < lib->group_count; g++) {
                    rp->groups[rp->group_count++] = lib->groups[g];
                }
            }
        }

        /* 4. If executable pipeline (!is_library), build and compile the GLCompute megakernel */
        COMPAT_INFO("CRTP [%u/%u]: building megakernel (is_lib=%d, stage_ok=%d, stages=%u)...",
                    i + 1, createInfoCount, (int)is_library, (int)stage_ok, rp->stage_count);
        const int rt_max_stages = rt_env()->max_stages;   /* also sets g_rt_guard_max (once) */
        if (!is_library && stage_ok && rt_max_stages > 0 && rp->stage_count > (uint32_t)rt_max_stages) {
            /* A/B lever: no megakernel for this pipeline; its trace dispatches are skipped and counted. */
            rp->compile_state = 4;
            atomic_fetch_add_explicit(&g_mega_capped, 1, memory_order_relaxed);
            COMPAT_INFO("CRTP [%u/%u]: %u stages > MALI_COMPAT_RT_MAX_STAGES=%d -> megakernel NOT built, traces skipped",
                        i + 1, createInfoCount, rp->stage_count, rt_max_stages);
        } else if (!is_library && stage_ok && rp->stage_count > 0) {
            if (!__atomic_load_n(&g_rt_guard_bda, __ATOMIC_ACQUIRE) && rt_ensure_ring_buffer(d)) {
                VkDeviceAddress zero = 0ULL;   /* first device wins (was a racy plain write) */
                __atomic_compare_exchange_n(&g_rt_guard_bda, &zero, d->rt_ring_bda + RT_GUARD_SLOT_OFF, false,
                                            __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE);
            }
            if (rt_split_pipeline(d, device, pipelineCache, pAllocator, rp, i, createInfoCount)) {
                /* split-1112: children compiled/queued; the parent has no megakernel of its own */
            } else {
                rt_mega_compile(d, device, pipelineCache, pAllocator, rp, rp->stages, rp->stage_count, i, createInfoCount);
            }
        }
        /* round 4 lever MALI_COMPAT_RT_FREE_STAGES=1: a non-library pipeline never needs its stage SPIR-V again (the
         * megakernel SPIR-V is built from it above; only a LIBRARY's stages are read later, when it is linked). The
         * stage ids stay: the group handles (rp->groups) carry them. */
        if (!is_library && rt_env()->free_stages) {
            int64_t freed = 0;
            for (uint32_t s = 0; s < rp->stage_count; s++) {
                struct RtStageRecord *sr = &rp->stages[s];
                const uint32_t sid = sr->stage_id;
                const VkShaderStageFlagBits sg = sr->stage;
                freed += (int64_t)sr->acct_bytes;
                rt_free_stage_record(sr);
                memset(sr, 0, sizeof(*sr));
                sr->stage_id = sid;
                sr->stage = sg;
            }
            atomic_fetch_add_explicit(&g_mega_freed_stage_bytes, freed, memory_order_relaxed);
        }

        VkPipeline handle = (VkPipeline)(uintptr_t)rp;
        unsigned b = rt_bucket((uintptr_t)handle);
        pthread_mutex_lock(&g_rt_lock);
        rp->next = g_rt_pipelines[b];
        g_rt_pipelines[b] = rp;
        pthread_mutex_unlock(&g_rt_lock);

        pPipelines[i] = handle;
    }

    return final_res;
}

static VKAPI_ATTR VkResult VKAPI_CALL
compat_GetRayTracingShaderGroupHandlesKHR(
    VkDevice                                    device,
    VkPipeline                                  pipeline,
    uint32_t                                    firstGroup,
    uint32_t                                    groupCount,
    size_t                                      dataSize,
    void                                       *pData)
{
    (void)device;
    rt_patch_fex_thunks();
    if (!pData) return VK_ERROR_INITIALIZATION_FAILED;
    struct RtPipeline *rp = rt_lookup_pipeline(pipeline);
    COMPAT_TRACE("compat_GetRayTracingShaderGroupHandlesKHR: pipeline=%p firstGroup=%u groupCount=%u dataSize=%zu pData=%p (rp=%p, rp->group_count=%u)",
                 (void *)pipeline, firstGroup, groupCount, dataSize, pData, (void *)rp, rp ? rp->group_count : 0);
    if (!rp) {
        memset(pData, 0, dataSize);
        return VK_SUCCESS;
    }
    uint8_t *dst = (uint8_t *)pData;
    for (uint32_t i = 0; i < groupCount; i++) {
        if ((size_t)(i + 1) * RT_GROUP_HANDLE_SIZE > dataSize) break;
        uint32_t gidx = firstGroup + i;
        if (gidx < rp->group_count) {
            memcpy(dst + i * RT_GROUP_HANDLE_SIZE, &rp->groups[gidx], RT_GROUP_HANDLE_SIZE);
        } else {
            memset(dst + i * RT_GROUP_HANDLE_SIZE, 0, RT_GROUP_HANDLE_SIZE);
        }
    }
    return VK_SUCCESS;
}

static VKAPI_ATTR VkResult VKAPI_CALL
compat_GetRayTracingCaptureReplayShaderGroupHandlesKHR(
    VkDevice                                    device,
    VkPipeline                                  pipeline,
    uint32_t                                    firstGroup,
    uint32_t                                    groupCount,
    size_t                                      dataSize,
    void                                       *pData)
{
    return compat_GetRayTracingShaderGroupHandlesKHR(device, pipeline, firstGroup, groupCount, dataSize, pData);
}

static VKAPI_ATTR VkDeviceSize VKAPI_CALL
compat_GetRayTracingShaderGroupStackSizeKHR(
    VkDevice                                    device,
    VkPipeline                                  pipeline,
    uint32_t                                    group,
    VkShaderGroupShaderKHR                      groupShader)
{
    (void)device; (void)pipeline; (void)group; (void)groupShader;
    return 256;
}

static VKAPI_ATTR void VKAPI_CALL
compat_CmdSetRayTracingPipelineStackSizeKHR(
    VkCommandBuffer                             commandBuffer,
    uint32_t                                    pipelineStackSize)
{
    (void)commandBuffer; (void)pipelineStackSize;
}


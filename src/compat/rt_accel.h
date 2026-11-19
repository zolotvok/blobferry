// SPDX-License-Identifier: GPL-3.0-or-later
// rt_accel.h: acceleration structures: scratch pool, builds, copy, properties. Included by mali_rt_emu.h.
/* round 4: the pool buffer's size (MALI_COMPAT_RT_SCRATCH_POOL_MB, default 128 MB; the host pool test keeps 128 MB) */
static inline VkDeviceSize rt_pool_buf_size(void) {
#ifdef RT_AS_SCRATCH_POOL_TEST
    return (VkDeviceSize)RT_AS_SCRATCH_SIZE;
#else
    return (VkDeviceSize)rt_env()->pool_mb << 20;
#endif
}
static bool rt_ensure_as_scratch_buffer(struct DeviceData *d) {
    if (!d) return false;
    if (d->rt_scratch_bda) return true;
    if (!d->fpCreateBuffer || !d->fpGetBufferMemoryRequirements ||
        !d->fpAllocateMemory || !d->fpBindBufferMemory || !d->fpGetBufferDeviceAddress) {
        return false;
    }

    pthread_mutex_lock(&d->bc_lock);
    if (d->rt_scratch_bda) {
        pthread_mutex_unlock(&d->bc_lock);
        return true;
    }

    VkBufferCreateInfo bci = {
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .size  = rt_pool_buf_size(),
        .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                 VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
                 VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR,
    };
    VkBuffer buf = VK_NULL_HANDLE;
    if (d->fpCreateBuffer(d->device, &bci, NULL, &buf) != VK_SUCCESS) {
        bci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
        if (d->fpCreateBuffer(d->device, &bci, NULL, &buf) != VK_SUCCESS) {
            pthread_mutex_unlock(&d->bc_lock);
            COMPAT_INFO("rt_ensure_as_scratch_buffer: vkCreateBuffer failed for 64MB scratch");
            return false;
        }
    }

    VkMemoryRequirements req = {0};
    d->fpGetBufferMemoryRequirements(d->device, buf, &req);
    uint32_t mask = req.memoryTypeBits;
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
    if (am_drv_alloc(d, "rt-scratch", &mai, &mem) != VK_SUCCESS) {
        d->fpDestroyBuffer(d->device, buf, NULL);
        pthread_mutex_unlock(&d->bc_lock);
        COMPAT_INFO("rt_ensure_as_scratch_buffer: vkAllocateMemory failed for 64MB scratch");
        return false;
    }

    if (d->fpBindBufferMemory(d->device, buf, mem, 0) != VK_SUCCESS) {
        am_drv_free(d, "rt-scratch", mem);
        d->fpDestroyBuffer(d->device, buf, NULL);
        pthread_mutex_unlock(&d->bc_lock);
        COMPAT_INFO("rt_ensure_as_scratch_buffer: vkBindBufferMemory failed");
        return false;
    }

    VkBufferDeviceAddressInfo bdai = {
        .sType  = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO,
        .buffer = buf,
    };
    VkDeviceAddress bda = d->fpGetBufferDeviceAddress(d->device, &bdai);
    if (!bda) {
        am_drv_free(d, "rt-scratch", mem);
        d->fpDestroyBuffer(d->device, buf, NULL);
        pthread_mutex_unlock(&d->bc_lock);
        COMPAT_INFO("rt_ensure_as_scratch_buffer: fpGetBufferDeviceAddress returned 0");
        return false;
    }

    d->rt_scratch_buf = buf;
    d->rt_scratch_mem = mem;
    d->rt_scratch_bda = bda;
    d->rt_scratch_size = mai.allocationSize;
    rt_acct_add(&g_acct_pool, (int64_t)mai.allocationSize);
    atomic_store(&d->rt_scratch_offset, 0u);
    pthread_mutex_unlock(&d->bc_lock);

    COMPAT_INFO("rt_ensure_as_scratch_buffer: 64MB persistent AS scratch allocated at BDA 0x%llx",
                (unsigned long long)bda);
    return true;
}

/* 11-08 keeper: AS build scratch.
 * Data: move-fix run (7c00bbe9): 989 identical GPU WRITE faults (0x3c3) at 0x5dd2e01000 = 4 KB past the end of the
 * app's 32 MB+4 KB scratch heap [0x5dd0e00000, 0x5dd2e01000) (maps/m_hang.txt), Gemini's Flaw F: 137 KB past an
 * 8 MB heap during 54 in-place BLAS updates. e46d170 then replaced every scratch below 256 GB by FIXED 2 MB slices
 * although each build needs ~4.6 MB (the inflated size) -> builds of one call overlap and run concurrently, the BVH
 * builder's atomics/counters corrupt each other (megafix L1: 1149 WRITE/ATOMIC faults spread over 0x39d..0x3ac).
 * Now: every build gets its own slice of the layer's 128 MB scratch, sized from the DRIVER's real size query for this
 * build (not our inflated hook) plus max(need, 1 MB) headroom, 64 KB aligned, non-overlapping inside one call.
 * Reuse across calls is ordered by the full pre-build barrier (single queue: MTK m_num_queues=1).
 * Lever MALI_COMPAT_RT_OWN_SCRATCH=0: keep the app scratch (only 0/invalid scratch is still redirected).
 * MALI_COMPAT_DEBUG: rt_scratch: summary line (<= 1/s) incl. "app region short" = builds whose app scratch range
 * [scratch, scratch+need) starts in mapped memory but ends in unmapped memory (decisive for the overrun theory). */
/* 11-10 keeper (lead-1110, Codex 1110-194217 finding 3): pool exhaustion no longer forwards a scratch that was judged
 * 0/invalid. When the next build's slice does not fit the rest of the pool, the builds planned so far go out as
 * their own driver call (pre/post barrier each) and the pool restarts at 0 for the rest ("nofit" = such splits).
 * A build that ALONE does not fit (even without headroom) keeps a valid app scratch ("big-app"), and is DROPPED when
 * the app scratch is 0/invalid ("dropped", logged <= 1/s): no 0/invalid scratch address reaches the driver.
 * Per build the plan also carries the driver's real AS size (same query) for the dst-short check and the
 * MALI_COMPAT_RT_BUILD_LOG lines. RT_AS_SCRATCH_POOL_TEST: test-only smaller pool (host split test). */
#ifndef RT_AS_SCRATCH_POOL_TEST
#define RT_AS_SCRATCH_POOL ((VkDeviceSize)rt_env()->pool_mb << 20)   /* round 4: MALI_COMPAT_RT_SCRATCH_POOL_MB */
#else
#define RT_AS_SCRATCH_POOL ((VkDeviceSize)(RT_AS_SCRATCH_POOL_TEST))
#endif
struct RtBuildPlan {
    VkDeviceSize storage;      /* dst RtAsTrack.size (what the app created the AS with), 0 = untracked */
    VkDeviceSize drv_as;       /* driver's real accelerationStructureSize for this build, 0 = not queried */
    VkDeviceSize drv_scratch;  /* driver's real build/update scratch size, 0 = not queried */
    uint32_t     batch;        /* driver call this build goes out in */
    bool         drop;
    VkAccelerationStructureKHR src_drv;   /* 1110b: driver handle of the APP's src (kept if a lever clears src) */
    VkDeviceSize src_storage;  /* 1110b: src RtAsTrack.size, 0 = no src / untracked */
    bool         othersrc_bigger;   /* 1110b: app UPDATE from another AS whose storage > dst storage */
    bool         othersrc_forced;   /* 1110b: ... turned into BUILD by MALI_COMPAT_RT_UPD_SRC */
    bool         need_pool;    /* lead-1110c: the app's UPDATE became a BUILD in the layer: its app scratch was sized for
                                * the UPDATE, so the pool slice (BUILD-sized) is used even with MALI_COMPAT_RT_OWN_SCRATCH=0 */
    bool         drop_type;    /* round 4 (Codex r3 #7): no driver handle of the build's type -> dropped */
    VkAccelerationStructureTypeKHR prev_built;   /* round 4: dst track's built type before this call (restored on drop) */
    VkAccelerationStructureKHR prev_active;      /* round 4: dst track's active handle before this call */
};
static _Atomic uint64_t g_rt_updsrc_nofit = 0;   /* round 4 (Codex r3 #1): other-source UPDATEs kept (BUILD would not fit) */
static _Atomic uint64_t g_rt_build_nofit = 0;    /* round 4 (Codex r4 #1): layer-made BUILDs dropped (driver size > storage) */
static _Atomic uint64_t g_rt_scr_builds = 0, g_rt_scr_own = 0, g_rt_scr_nofit = 0, g_rt_scr_short = 0,
                        g_rt_scr_blind = 0, g_rt_scr_maxcall = 0, g_rt_scr_dropped = 0, g_rt_scr_bigapp = 0,
                        g_rt_scr_dstshort = 0;
/* 1110b: UPDATE builds the LAYER turned into BUILD, by reason; CmdBuildAS calls passed through unplanned because a
 * host allocation failed. lead-1110c: g_rt_as_leaked now counts type-change recreations SKIPPED because the retire list
 * could not grow (Codex 1110-212923 :6773; nothing leaks any more), printed as retire-skip=; clone = first in-place
 * UPDATE after a copy (MALI_COMPAT_RT_CLONE_REFIT), badgeom = an UPDATE whose geometry the sanitiser zeroes. */
static _Atomic uint64_t g_rt_fb_unbuilt = 0, g_rt_fb_mutated = 0, g_rt_fb_nosrc = 0, g_rt_fb_lever = 0,
                        g_rt_as_leaked = 0, g_rt_build_oom = 0, g_rt_fb_clone = 0, g_rt_fb_badgeom = 0;
/* 1110b follow-up: app UPDATEs with src == dst (in place) / src != dst, and those of the latter whose source's created
 * storage is larger than the destination's; othersrc = such UPDATEs turned into BUILD by MALI_COMPAT_RT_UPD_SRC. */
static _Atomic uint64_t g_rt_upd_inplace = 0, g_rt_upd_othersrc = 0, g_rt_upd_bigger = 0, g_rt_fb_othersrc = 0;
/* returns the number of driver calls (batches) the builds were split into */
static uint32_t rt_assign_as_scratch(struct DeviceData *d, uint32_t count,
                                     VkAccelerationStructureBuildGeometryInfoKHR *infos,
                                     const VkAccelerationStructureBuildGeometryInfoKHR *pInfos,
                                     const VkAccelerationStructureBuildRangeInfoKHR * const *ppBuildRangeInfos,
                                     VkDeviceAddress own_bda, struct RtBuildPlan *plan, bool want_sizes) {
    const int own = rt_env()->own_scratch;
    const int headroom = rt_env()->scratch_head;
    const VkDeviceSize align = 65536;
    const VkDeviceSize pool_end = RT_AS_SCRATCH_POOL - 65536; /* last 64 KB stay a guard */
    VkDeviceSize cursor = 0, max_cursor = 0;
    uint32_t batch = 0;
    for (uint32_t i = 0; i < count; i++) {
        VkDeviceAddress app = infos[i].scratchData.deviceAddress;
        bool bad_app = (app == 0ULL) || rt_is_invalid_gpu_bda(app);
        VkDeviceSize need = 0;
        plan[i].drv_as = 0; plan[i].drv_scratch = 0; plan[i].drop = false;
        if (plan[i].drop_type) { plan[i].drop = true; plan[i].batch = batch; continue; }   /* round 4, logged at recreate */
        /* lead-1110c: always queried for a BUILD (the track records the driver's AS size for the layer's CLONE) */
        if (d->fpGetAccelerationStructureBuildSizesKHR &&
            (own || bad_app || g_debug || want_sizes || plan[i].need_pool ||
             infos[i].mode == VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR)) {
            uint32_t stack_max[32];
            uint32_t *maxp = stack_max;
            uint32_t gc = infos[i].geometryCount;
            if (gc > 32) maxp = (uint32_t *)calloc(gc, sizeof(uint32_t));
            if (maxp) {
                for (uint32_t g = 0; g < gc; g++)
                    maxp[g] = (ppBuildRangeInfos && ppBuildRangeInfos[i]) ? ppBuildRangeInfos[i][g].primitiveCount : 0;
                VkAccelerationStructureBuildGeometryInfoKHR q = infos[i];
                q.pGeometries = pInfos[i].pGeometries;
                q.ppGeometries = pInfos[i].ppGeometries;
                VkAccelerationStructureBuildSizesInfoKHR sz = {
                    .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR };
                d->fpGetAccelerationStructureBuildSizesKHR(d->device, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
                                                           &q, maxp, &sz);
                need = (infos[i].mode == VK_BUILD_ACCELERATION_STRUCTURE_MODE_UPDATE_KHR)
                           ? sz.updateScratchSize : sz.buildScratchSize;
                plan[i].drv_as = sz.accelerationStructureSize;
                plan[i].drv_scratch = need;
                if (maxp != stack_max) free(maxp);
            }
        }
        atomic_fetch_add(&g_rt_scr_builds, 1);
        /* round 4 (Codex r4 #1): a BUILD the LAYER made from an UPDATE must fit its destination's storage; now that the
         * driver's size for this build is known (also when the track's build_size was unknown) - otherwise dropped */
        /* (Codex r4b #1: a failed size query, drv_as 0, cannot prove the fit either -> dropped too) */
        if (plan[i].need_pool && plan[i].storage && (!plan[i].drv_as || plan[i].drv_as > plan[i].storage)) {
            uint64_t n = atomic_fetch_add(&g_rt_build_nofit, 1) + 1;
            static _Atomic int64_t bn_ms = 0;
            int64_t now = rt_now_ms(), pv = atomic_load(&bn_ms);
            if ((n == 1 || now - pv >= 1000) && atomic_compare_exchange_strong(&bn_ms, &pv, now))
                LOG("rt_build: a layer-made BUILD (app UPDATE) needs %llu B but the destination storage is %llu B -> "
                    "DROPPED (build-nofit=%llu)", (unsigned long long)plan[i].drv_as, (unsigned long long)plan[i].storage,
                    (unsigned long long)n);
            plan[i].drop_type = true;   /* not a scratch drop: logged here */
            plan[i].drop = true;
            plan[i].batch = batch;
            continue;
        }
        if (g_debug && !bad_app && need > 0) {
            bool s0 = rt_is_host_address_accessible((const void *)(uintptr_t)app, 1);
            bool s1 = rt_is_host_address_accessible((const void *)(uintptr_t)(app + need - 1), 1);
            if (!s0) atomic_fetch_add(&g_rt_scr_blind, 1);
            else if (!s1) {
                uint64_t n = atomic_fetch_add(&g_rt_scr_short, 1) + 1;
                if (n <= 8)
                    LOG("rt_scratch: app region short #%llu: build %u/%u type=%u mode=%u scratch=0x%llx need=%llu "
                        "(end 0x%llx unmapped)", (unsigned long long)n, i + 1, count, infos[i].type, infos[i].mode,
                        (unsigned long long)app, (unsigned long long)need, (unsigned long long)(app + need));
            }
        }
        if (g_debug && plan[i].drv_as && plan[i].storage && plan[i].drv_as > plan[i].storage) {
            uint64_t n = atomic_fetch_add(&g_rt_scr_dstshort, 1) + 1;
            if (n <= 8) {
                VkDeviceAddress dst_bda = 0ULL;
                if (d->fpGetAccelerationStructureDeviceAddressKHR && infos[i].dstAccelerationStructure) {
                    VkAccelerationStructureDeviceAddressInfoKHR dai = {
                        .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR,
                        .accelerationStructure = infos[i].dstAccelerationStructure };
                    dst_bda = d->fpGetAccelerationStructureDeviceAddressKHR(d->device, &dai);
                }
                LOG("rt_scratch: dst short #%llu type=%u mode=%u dst_bda=0x%llx storage=%llu need=%llu",
                    (unsigned long long)n, infos[i].type, infos[i].mode, (unsigned long long)dst_bda,
                    (unsigned long long)plan[i].storage, (unsigned long long)plan[i].drv_as);
            }
        }
        plan[i].batch = batch;
        if (!(own || bad_app || plan[i].need_pool)) continue;   /* valid app scratch kept (OWN_SCRATCH=0) */
        if (!own_bda) {                                     /* no layer pool: an invalid app scratch cannot go out */
            /* round 4 (Codex r3 #4): nor can an UPDATE-sized app scratch under a layer-made BUILD */
            if (bad_app || plan[i].need_pool) plan[i].drop = true;
            continue;
        }
        if (need == 0) need = 4ull * 1024 * 1024;          /* query unavailable: old 4 MB guess */
        VkDeviceSize head = need;                           /* 2x need, at least +1 MB (the app heap was overrun) */
        if (head < 1024 * 1024) head = 1024 * 1024;
        if (!headroom) head = 0;                            /* round 4 lever MALI_COMPAT_RT_SCRATCH_HEAD=0 */
        VkDeviceSize slice = (need + head + align - 1) & ~(align - 1);
        VkDeviceSize base  = ((own_bda + align - 1) & ~(align - 1)) - own_bda;
        VkDeviceSize start = (((own_bda + cursor) + align - 1) & ~(align - 1)) - own_bda;
        if (start + slice > pool_end && cursor > 0) {       /* pool full: split, the next call reuses the pool */
            batch++;
            plan[i].batch = batch;
            cursor = 0;
            start = base;
            atomic_fetch_add(&g_rt_scr_nofit, 1);
        }
        if (start + slice > pool_end) {                     /* alone too big with headroom */
            VkDeviceSize tight = (need + align - 1) & ~(align - 1);
            if (start + tight <= pool_end) {
                slice = tight;
            } else if (!bad_app && !plan[i].need_pool) {   /* round 4 (Codex r3 #4): not for a layer-made BUILD */
                atomic_fetch_add(&g_rt_scr_bigapp, 1);      /* keep the app's valid scratch */
                continue;
            } else {
                plan[i].drop = true;
                continue;
            }
        }
        infos[i].scratchData.deviceAddress = own_bda + start;
        cursor = start + slice;
        if (cursor > max_cursor) max_cursor = cursor;
        atomic_fetch_add(&g_rt_scr_own, 1);
        {
            int64_t ps = atomic_load_explicit(&g_rt_scr_maxslice, memory_order_relaxed);
            while ((int64_t)slice > ps && !atomic_compare_exchange_weak_explicit(&g_rt_scr_maxslice, &ps, (int64_t)slice,
                                                                               memory_order_relaxed, memory_order_relaxed)) {}
        }
    }
    for (uint32_t i = 0; i < count; i++) {
        if (!plan[i].drop || plan[i].drop_type) continue;
        uint64_t n = atomic_fetch_add(&g_rt_scr_dropped, 1) + 1;
        static _Atomic int64_t drop_ms = 0;
        int64_t now = rt_now_ms(), pv = atomic_load(&drop_ms);
        if ((n == 1 || now - pv >= 1000) && atomic_compare_exchange_strong(&drop_ms, &pv, now))
            LOG("rt_scratch: DROPPED build %u/%u type=%u mode=%u need=%llu app scratch=0x%llx (0/invalid, does not "
                "fit the %u MB pool, or an UPDATE turned into a BUILD without a pool slice) total dropped=%llu",
                i + 1, count, infos[i].type, infos[i].mode,
                (unsigned long long)plan[i].drv_scratch, (unsigned long long)infos[i].scratchData.deviceAddress,
                (unsigned)(RT_AS_SCRATCH_POOL >> 20), (unsigned long long)n);
    }
    uint64_t prev_max = atomic_load(&g_rt_scr_maxcall);
    while (max_cursor > prev_max && !atomic_compare_exchange_weak(&g_rt_scr_maxcall, &prev_max, max_cursor)) {}
    if (g_debug) {
        static _Atomic int64_t last_ms = 0;
        int64_t now = rt_now_ms(), pv = atomic_load(&last_ms);
        if (now - pv >= 1000 && atomic_compare_exchange_strong(&last_ms, &pv, now))
            LOG("rt_scratch: t=%lld ms builds=%llu own=%llu nofit=%llu dropped=%llu big-app=%llu dst-short=%llu "
                "app-short=%llu app-unmapped-start=%llu max-per-call=%llu KB forced-build=unbuilt:%llu,mutated:%llu,"
                "no-src:%llu,lever:%llu,othersrc:%llu,clone:%llu,badgeom:%llu retire-skip=%llu oom-passthrough=%llu upd-inplace=%llu "
                "upd-othersrc=%llu upd-othersrc-bigger=%llu", (long long)now,
                (unsigned long long)atomic_load(&g_rt_scr_builds), (unsigned long long)atomic_load(&g_rt_scr_own),
                (unsigned long long)atomic_load(&g_rt_scr_nofit), (unsigned long long)atomic_load(&g_rt_scr_dropped),
                (unsigned long long)atomic_load(&g_rt_scr_bigapp), (unsigned long long)atomic_load(&g_rt_scr_dstshort),
                (unsigned long long)atomic_load(&g_rt_scr_short), (unsigned long long)atomic_load(&g_rt_scr_blind),
                (unsigned long long)(atomic_load(&g_rt_scr_maxcall) >> 10),
                (unsigned long long)atomic_load(&g_rt_fb_unbuilt), (unsigned long long)atomic_load(&g_rt_fb_mutated),
                (unsigned long long)atomic_load(&g_rt_fb_nosrc), (unsigned long long)atomic_load(&g_rt_fb_lever),
                (unsigned long long)atomic_load(&g_rt_fb_othersrc), (unsigned long long)atomic_load(&g_rt_fb_clone),
                (unsigned long long)atomic_load(&g_rt_fb_badgeom), (unsigned long long)atomic_load(&g_rt_as_leaked), (unsigned long long)atomic_load(&g_rt_build_oom),
                (unsigned long long)atomic_load(&g_rt_upd_inplace), (unsigned long long)atomic_load(&g_rt_upd_othersrc),
                (unsigned long long)atomic_load(&g_rt_upd_bigger));
    }
    return batch + 1;
}

/* The full barriers around every AS build batch (unchanged since 308e8895; as-probe 11-10: a ray-query dispatch and a
 * following AS build on one queue are only ordered with a barrier between them). */
static void rt_as_build_barrier(struct DeviceData *d, VkCommandBuffer cmd, bool pre) {
    if (!d->fpCmdPipelineBarrier) return;
    VkMemoryBarrier mb = { .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER };
    if (pre) {
        mb.srcAccessMask = VK_ACCESS_HOST_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT |
                           VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_MEMORY_WRITE_BIT |
                           VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;
        mb.dstAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR |
                           VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR |
                           VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_MEMORY_READ_BIT;
    } else {
        mb.srcAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR |
                           VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_SHADER_WRITE_BIT |
                           VK_ACCESS_MEMORY_WRITE_BIT;
        mb.dstAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR |
                           VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_MEMORY_READ_BIT |
                           VK_ACCESS_UNIFORM_READ_BIT;
    }
    d->fpCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                            0, 1, &mb, 0, NULL, 0, NULL);
}

/* lead-1110c: true if the layer's sanitiser will zero a geometry of this build (invalid/0 vertex, AABB or instance
 * data -> primitiveCount 0). An UPDATE must keep the primitive counts of its BUILD (Codex 1110-212923 :6923/:6935),
 * so such an UPDATE is turned into a BUILD before the scratch is planned. Same tests as the sanitiser below. */
static bool rt_build_has_bad_geometry(const VkAccelerationStructureBuildGeometryInfoKHR *bi,
                                      const VkAccelerationStructureBuildRangeInfoKHR *ranges) {
    for (uint32_t j = 0; j < bi->geometryCount; j++) {
        const VkAccelerationStructureGeometryKHR *g = bi->pGeometries ? &bi->pGeometries[j]
                                                    : (bi->ppGeometries ? bi->ppGeometries[j] : NULL);
        if (!g) continue;
        if (g->geometryType == VK_GEOMETRY_TYPE_TRIANGLES_KHR) {
            VkDeviceAddress a = g->geometry.triangles.vertexData.deviceAddress;
            if (a == 0ULL || rt_is_invalid_gpu_bda(a)) return true;
        } else if (g->geometryType == VK_GEOMETRY_TYPE_AABBS_KHR) {
            VkDeviceAddress a = g->geometry.aabbs.data.deviceAddress;
            if (a == 0ULL || rt_is_invalid_gpu_bda(a)) return true;
        } else if (g->geometryType == VK_GEOMETRY_TYPE_INSTANCES_KHR) {
            VkDeviceAddress a = g->geometry.instances.data.deviceAddress;
            uint32_t off = ranges ? ranges[j].primitiveOffset : 0;
            if (a == 0ULL || rt_is_invalid_gpu_bda(a) || rt_is_invalid_gpu_bda(a + (uint64_t)off)) return true;
        }
    }
    return false;
}

/* lead-1110c: a build's type does not match the track's built type or its active handle's creation type: a clean
 * driver handle of that type on the same storage. Codex 1110-212923 :6773: room on the retired list is made FIRST;
 * if it cannot grow, nothing is created or destroyed - the old active handle stays and is rebuilt (as with
 * MALI_COMPAT_RT_AS_TRUST=1), counted as retire-skip. Either way the build becomes a BUILD of the new type.
 * Caller holds g_rt_as_lock. */
static _Atomic uint64_t g_rt_type_drop = 0;   /* round 4: builds dropped because no handle of their type exists */
/* round 4 (Codex r3 #7): returns false when no handle of the build's type could be made AND the active handle was
 * explicitly created with the other type: the caller must drop the build (a BLAS build through a TLAS-created handle
 * violates VUID-03700). A GENERIC active handle is rebuilt as before (as-probe mutate PASS). */
static bool rt_as_recreate_locked(struct DeviceData *d, struct RtAsTrack *track,
                                  VkAccelerationStructureBuildGeometryInfoKHR *info, bool app_update) {
    VkAccelerationStructureKHR old = track->active_handle;
    const bool retire = old && old != track->orig_handle;
    bool room = !retire || track->n_retired < track->cap_retired;
    if (!room) {
        uint32_t nc = track->cap_retired ? track->cap_retired * 2 : 4;
        VkAccelerationStructureKHR *nr = (VkAccelerationStructureKHR *)realloc(track->retired, nc * sizeof(*nr));
        if (nr) { track->retired = nr; track->cap_retired = nc; room = true; rt_as_acct_host(track); }
    }
    VkAccelerationStructureKHR new_h = VK_NULL_HANDLE;
    if (room) {
        VkAccelerationStructureCreateInfoKHR ci = {
            .sType  = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR,
            .type   = info->type,
            .buffer = track->buffer,
            .offset = track->offset,
            .size   = track->size,
        };
        if (!d->fpCreateAccelerationStructureKHR ||
            d->fpCreateAccelerationStructureKHR(track->device, &ci, NULL, &new_h) != VK_SUCCESS)
            new_h = VK_NULL_HANDLE;
    } else {
        uint64_t n = atomic_fetch_add(&g_rt_as_leaked, 1) + 1;
        static _Atomic int64_t rs_ms = 0;
        int64_t now = rt_now_ms(), pv = atomic_load(&rs_ms);
        if ((n == 1 || now - pv >= 1000) && atomic_compare_exchange_strong(&rs_ms, &pv, now))
            LOG("RT_EMU: AS %p type %u -> %u: retire list cannot grow (out of host memory) -> no new handle, the "
                "active one is rebuilt (retire-skip=%llu)", (void *)track->orig_handle, (unsigned)track->built_type,
                (unsigned)info->type, (unsigned long long)n);
    }
    if (new_h) {
        /* 1110b: the replaced handle (unless it is the app's original, destroyed with the track anyway) is kept for
         * destruction at the app's vkDestroyAccelerationStructure - it may be in flight now */
        if (retire) track->retired[track->n_retired++] = old;
        track->active_handle = new_h;
        track->active_type   = info->type;
    } else if (track->active_type != VK_ACCELERATION_STRUCTURE_TYPE_GENERIC_KHR && track->active_type != info->type) {
        uint64_t n = atomic_fetch_add(&g_rt_type_drop, 1) + 1;
        static _Atomic int64_t td_ms = 0;
        int64_t now = rt_now_ms(), pv = atomic_load(&td_ms);
        if ((n == 1 || now - pv >= 1000) && atomic_compare_exchange_strong(&td_ms, &pv, now))
            LOG("RT_EMU: AS %p: no handle of type %u (active handle created as type %u, new handle %s) -> build DROPPED "
                "(type-drop=%llu)", (void *)track->orig_handle, (unsigned)info->type, (unsigned)track->active_type,
                room ? "creation failed" : "not made: retire list full", (unsigned long long)n);
        return false;
    }
    track->built_type = info->type;
    if (app_update && info->mode == VK_BUILD_ACCELERATION_STRUCTURE_MODE_UPDATE_KHR) atomic_fetch_add(&g_rt_fb_mutated, 1);
    info->mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
    return true;
}

/* 1110b (Codex 1110-202009 #1/#4): a host allocation for planning failed. No silent cap, no stack-array fallback:
 * the WHOLE original call goes to the driver (between the usual barriers) - unless any build carries a 0/invalid
 * scratch address, then the whole call is dropped (logged <= 1/s): an invalid scratch never reaches the driver.
 * lead-1110c (Codex 1110-212923 :6666): the driver gets the ACTIVE handles (a type change may have replaced the app's
 * one): substituted in a copy of the infos. The whole call is DROPPED if that copy cannot be allocated, or if any build
 * needs what only the planned path can do: a type that does not match the active handle / the last build (UPDATE),
 * the first UPDATE of a copied-into AS (MALI_COMPAT_RT_CLONE_REFIT=1), an other-source UPDATE that
 * MALI_COMPAT_RT_UPD_SRC turns into a BUILD. A passed-through BUILD leaves the track's AS size unknown (0: a later
 * CLONE from it goes to the driver). */
static void rt_build_passthrough(struct DeviceData *d, VkCommandBuffer cmd, uint32_t infoCount,
                                 const VkAccelerationStructureBuildGeometryInfoKHR *pInfos,
                                 const VkAccelerationStructureBuildRangeInfoKHR * const *ppBuildRangeInfos) {
    uint64_t n = atomic_fetch_add(&g_rt_build_oom, 1) + 1;
    const char *why = NULL;
    uint32_t bad = UINT32_MAX;
    for (uint32_t i = 0; i < infoCount && bad == UINT32_MAX; i++) {
        VkDeviceAddress a = pInfos[i].scratchData.deviceAddress;
        if (a == 0ULL || rt_is_invalid_gpu_bda(a)) bad = i;
    }
    if (bad != UINT32_MAX) why = "0/invalid scratch";
    VkAccelerationStructureBuildGeometryInfoKHR *sub = NULL;
    const int clone_refit = rt_env()->clone_refit, us = rt_env()->upd_src;
    if (!why) {
        pthread_mutex_lock(&g_rt_as_lock);
        bool need_sub = false;
        for (uint32_t i = 0; i < infoCount && !why; i++) {
            const bool upd = pInfos[i].mode == VK_BUILD_ACCELERATION_STRUCTURE_MODE_UPDATE_KHR;
            struct RtAsTrack *t = rt_as_lookup_locked(pInfos[i].dstAccelerationStructure);
            struct RtAsTrack *st = upd ? rt_as_lookup_locked(pInfos[i].srcAccelerationStructure) : NULL;
            if (t) {
                if (t->active_type != VK_ACCELERATION_STRUCTURE_TYPE_GENERIC_KHR && t->active_type != pInfos[i].type)
                    why = "type does not match the active handle";
                else if (upd && t->built_type != RT_AS_TYPE_UNBUILT && t->built_type != pInfos[i].type)
                    why = "UPDATE of another type";
                else if (upd && t->built_type == RT_AS_TYPE_UNBUILT)   /* round 4 (Codex r4b #2) */
                    why = "UPDATE of an unbuilt AS";
                else if (upd && clone_refit && !t->refit_ready && (!st || st == t))
                    why = "first UPDATE after a copy";
                /* round 4 (Codex r3 #6): an untracked source different from dst counts as other-source too */
                else if (upd && pInfos[i].srcAccelerationStructure && st != t &&
                         pInfos[i].srcAccelerationStructure != pInfos[i].dstAccelerationStructure &&
                         (us == 1 || (us == 2 && (!st || st->size > t->size))))
                    why = "other-source UPDATE";
                if (t->active_handle && t->active_handle != pInfos[i].dstAccelerationStructure) need_sub = true;
            }
            if (st && st->active_handle && st->active_handle != pInfos[i].srcAccelerationStructure) need_sub = true;
        }
        if (!why && need_sub) {
            sub = (VkAccelerationStructureBuildGeometryInfoKHR *)malloc(infoCount * sizeof(*sub));
            if (!sub) why = "no host memory to substitute the active handles";
            else {
                memcpy(sub, pInfos, infoCount * sizeof(*sub));
                for (uint32_t i = 0; i < infoCount; i++) {
                    struct RtAsTrack *t = rt_as_lookup_locked(sub[i].dstAccelerationStructure);
                    if (t && t->active_handle) sub[i].dstAccelerationStructure = t->active_handle;
                    if (sub[i].mode == VK_BUILD_ACCELERATION_STRUCTURE_MODE_UPDATE_KHR) {
                        struct RtAsTrack *st = rt_as_lookup_locked(sub[i].srcAccelerationStructure);
                        if (st && st->active_handle) sub[i].srcAccelerationStructure = st->active_handle;
                    }
                }
            }
        }
        if (!why) {
            for (uint32_t i = 0; i < infoCount; i++) {
                struct RtAsTrack *t = rt_as_lookup_locked(pInfos[i].dstAccelerationStructure);
                if (!t) continue;
                if (pInfos[i].mode == VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR) {
                    t->as_size = 0;           /* not queried here */
                    t->build_size = 0;
                    t->refit_ready = true;
                    if (t->built_type == RT_AS_TYPE_UNBUILT) t->built_type = pInfos[i].type;
                } else {
                    struct RtAsTrack *st = rt_as_lookup_locked(pInfos[i].srcAccelerationStructure);
                    if (st && st != t) { t->as_size = 0; t->build_size = 0; }
                }
                rt_as_acct_locked(t, false);
            }
        }
        pthread_mutex_unlock(&g_rt_as_lock);
    }
    static _Atomic int64_t oom_ms = 0;
    int64_t now = rt_now_ms(), pv = atomic_load(&oom_ms);
    if (n <= 4 || (now - pv >= 1000 && atomic_compare_exchange_strong(&oom_ms, &pv, now))) {   /* first 4, then <= 1/s */
        if (bad != UINT32_MAX)
            LOG("rt_scratch: out of host memory planning CmdBuildAS (%u builds); build %u has scratch 0x%llx "
                "(0/invalid) -> whole call DROPPED (total oom calls=%llu)", infoCount, bad + 1,
                (unsigned long long)pInfos[bad].scratchData.deviceAddress, (unsigned long long)n);
        else if (why)
            LOG("rt_scratch: out of host memory planning CmdBuildAS (%u builds) -> whole call DROPPED (%s) "
                "(total oom calls=%llu)", infoCount, why, (unsigned long long)n);
        else
            LOG("rt_scratch: out of host memory planning CmdBuildAS (%u builds) -> passed through %s "
                "(total oom calls=%llu)", infoCount, sub ? "with the active handles" : "unmodified",
                (unsigned long long)n);
    }
    if (why) { free(sub); return; }
    rt_as_build_barrier(d, cmd, true);
    d->fpCmdBuildAccelerationStructuresKHR(cmd, infoCount, sub ? sub : pInfos, ppBuildRangeInfos);
    rt_as_build_barrier(d, cmd, false);
    free(sub);
}

static VKAPI_ATTR void VKAPI_CALL
compat_CmdBuildAccelerationStructuresKHR(
    VkCommandBuffer                                         commandBuffer,
    uint32_t                                                infoCount,
    const VkAccelerationStructureBuildGeometryInfoKHR      *pInfos,
    const VkAccelerationStructureBuildRangeInfoKHR * const *ppBuildRangeInfos)
{
    struct DeviceData *d = get_device_data_by_cmd(commandBuffer);
    if (!d || !d->fpCmdBuildAccelerationStructuresKHR) return;

    if (!pInfos || infoCount == 0) {
        d->fpCmdBuildAccelerationStructuresKHR(commandBuffer, infoCount, pInfos, ppBuildRangeInfos);
        return;
    }

    /* 1110b: every host array is allocated up front, before any AS track changes; one failure -> the whole call goes
     * through rt_build_passthrough (no silent cap to 8 builds, no stack fallback that later overruns). */
    const uint32_t count = infoCount;
    uint64_t geoms_sum = 0;
    for (uint32_t i = 0; i < count; i++) geoms_sum += pInfos[i].geometryCount;
    const uint32_t total_geoms = geoms_sum > (uint64_t)(UINT32_MAX / 2) ? 0u : (uint32_t)geoms_sum;

    VkAccelerationStructureBuildGeometryInfoKHR stack_infos[8];
    struct RtBuildPlan stack_plan[8];
    VkAccelerationStructureGeometryKHR stack_all_geoms[32];
    VkAccelerationStructureBuildRangeInfoKHR stack_all_ranges[32];
    const VkAccelerationStructureBuildRangeInfoKHR *stack_ppRangeInfos[16];
    VkAccelerationStructureBuildGeometryInfoKHR *infos = stack_infos;
    struct RtBuildPlan *plan = stack_plan;
    VkAccelerationStructureGeometryKHR *all_geoms = stack_all_geoms;
    VkAccelerationStructureBuildRangeInfoKHR *all_ranges = stack_all_ranges;
    const VkAccelerationStructureBuildRangeInfoKHR **new_ppBuildRangeInfos = stack_ppRangeInfos;
    bool oom = geoms_sum > (uint64_t)(UINT32_MAX / 2);
    if (count > 8) {
        infos = (VkAccelerationStructureBuildGeometryInfoKHR *)malloc(count * sizeof(*infos));
        plan = (struct RtBuildPlan *)malloc(count * sizeof(*plan));
        if (!infos || !plan) oom = true;
    }
    if (total_geoms > 32) {
        all_geoms = (VkAccelerationStructureGeometryKHR *)malloc(total_geoms * sizeof(*all_geoms));
        all_ranges = (VkAccelerationStructureBuildRangeInfoKHR *)malloc(total_geoms * sizeof(*all_ranges));
        if (!all_geoms || !all_ranges) oom = true;
    }
    if (count > 16) {
        new_ppBuildRangeInfos = (const VkAccelerationStructureBuildRangeInfoKHR **)malloc(count * sizeof(*new_ppBuildRangeInfos));
        if (!new_ppBuildRangeInfos) oom = true;
    }
#ifdef RT_BUILD_OOM_TEST   /* test-only: simulate a failed planning allocation for calls with more than one build */
    if (count > 1) oom = true;
#endif
    if (oom) {
        rt_build_passthrough(d, commandBuffer, infoCount, pInfos, ppBuildRangeInfos);
        goto out;
    }
    memset(plan, 0, count * sizeof(*plan));
    memcpy(infos, pInfos, count * sizeof(*infos));

    bool has_safe_scratch = rt_ensure_as_scratch_buffer(d);
    VkDeviceAddress safe_scratch_bda = d->rt_scratch_bda;

    const int as_trust = rt_env()->as_trust;
    const int clone_refit = rt_env()->clone_refit;
    pthread_mutex_lock(&g_rt_as_lock);
    for (uint32_t i = 0; i < count; i++) {
        VkAccelerationStructureKHR dst = infos[i].dstAccelerationStructure;
        struct RtAsTrack *track = rt_as_lookup_locked(dst);
        plan[i].storage = track ? track->size : 0;
        plan[i].prev_built = track ? track->built_type : RT_AS_TYPE_UNBUILT;
        plan[i].prev_active = track ? track->active_handle : VK_NULL_HANDLE;
        const bool app_update = infos[i].mode == VK_BUILD_ACCELERATION_STRUCTURE_MODE_UPDATE_KHR;
        if (track) {
            bool recreate = false;
            if (track->built_type == RT_AS_TYPE_UNBUILT) {
                track->built_type = infos[i].type;
                if (!as_trust) {               /* 1110b lever MALI_COMPAT_RT_AS_TRUST=1: the app's mode is kept */
                    if (app_update) atomic_fetch_add(&g_rt_fb_unbuilt, 1);
                    infos[i].mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
                }
            } else if (track->built_type != infos[i].type && as_trust) {
                /* as-probe 11-10: the blob rebuilds a GENERIC handle as the other type without a new handle */
                COMPAT_INFO("RT_EMU: AS %p type %u -> %u, same handle rebuilt (MALI_COMPAT_RT_AS_TRUST=1)",
                           (void *)track->orig_handle, (unsigned)track->built_type, (unsigned)infos[i].type);
                track->built_type = infos[i].type;
            } else if (track->built_type != infos[i].type) {
                COMPAT_INFO("RT_EMU: AS %p mutated type %u -> %u! Allocating clean driver handle.",
                           (void *)track->orig_handle, (unsigned)track->built_type, (unsigned)infos[i].type);
                recreate = true;
            }
            /* lead-1110c (Codex 1110-212923 :7159): a copy propagates the source's built type, but the ACTIVE handle
             * may have been created with the other explicit type: its creation type decides too. */
            if (!recreate && track->active_type != VK_ACCELERATION_STRUCTURE_TYPE_GENERIC_KHR &&
                track->active_type != infos[i].type) {
                COMPAT_INFO("RT_EMU: AS %p active handle created as type %u, build type %u -> clean driver handle",
                           (void *)track->orig_handle, (unsigned)track->active_type, (unsigned)infos[i].type);
                recreate = true;
            }
            if (recreate && !rt_as_recreate_locked(d, track, &infos[i], app_update)) plan[i].drop_type = true;
            if (track->active_handle) {
                infos[i].dstAccelerationStructure = track->active_handle;
            }
        }
        /* 11-10: MALI_COMPAT_RT_FORCE_BUILD=1 (default off): an UPDATE (in-place refit, reads the src BVH's node
         * pointers) becomes a full BUILD from the geometry. Lever ported from Gemini 4 Argon's wt-rtx-emu, where it
         * defaulted ON. */
        if (infos[i].mode == VK_BUILD_ACCELERATION_STRUCTURE_MODE_UPDATE_KHR && rt_env()->force_build) {
            atomic_fetch_add(&g_rt_fb_lever, 1);
            infos[i].mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
            infos[i].srcAccelerationStructure = VK_NULL_HANDLE;
        }

        /* lead-1110c lever MALI_COMPAT_RT_CLONE_REFIT=1 (default): the first in-place UPDATE of an AS whose content came
         * from a copy becomes a BUILD from this build's own geometry (on the blob only a BUILD through a handle is
         * proven to make it refittable); later refits of it stay refits. */
        if (track && clone_refit && !track->refit_ready &&
            infos[i].mode == VK_BUILD_ACCELERATION_STRUCTURE_MODE_UPDATE_KHR) {
            struct RtAsTrack *s_t = rt_as_lookup_locked(pInfos[i].srcAccelerationStructure);
            /* a BUILD needs the full AS size: a destination smaller than its source's AS size (a COMPACT copy) keeps
             * the UPDATE (counted in the DEBUG line, never turned into an overrunning BUILD) */
            /* round 4: build_size (what a BUILD of this content needs; kept through COMPACT copies), not as_size */
            const bool fits = !track->build_size || track->size >= track->build_size;
            if (!fits && g_debug) {
                static _Atomic uint32_t nf_lines = 0;
                if (atomic_fetch_add(&nf_lines, 1) < 8)
                    LOG("rt_build: clone-refit skipped: AS %p storage=%llu < build size %llu (compacted copy) -> UPDATE kept",
                        (void *)track->orig_handle, (unsigned long long)track->size, (unsigned long long)track->build_size);
            }
            if (fits && (!pInfos[i].srcAccelerationStructure || s_t == track ||
                         pInfos[i].srcAccelerationStructure == pInfos[i].dstAccelerationStructure)) {
                uint64_t cn = atomic_fetch_add(&g_rt_fb_clone, 1) + 1;
                if (g_debug && cn <= 8)
                    LOG("rt_build: clone-refit #%llu: first in-place UPDATE of AS %p after a copy -> BUILD "
                        "(MALI_COMPAT_RT_CLONE_REFIT=1, storage=%llu)", (unsigned long long)cn,
                        (void *)track->orig_handle, (unsigned long long)track->size);
                infos[i].mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
                infos[i].srcAccelerationStructure = VK_NULL_HANDLE;
            }
        }

        /* 1110b follow-up (phone L1 on d4827f23: GPU WRITE faults 64-384 KB past the end of 8 MB AS buffers; the game
         * refits with src != dst). Classify the APP's UPDATE: in place or from another AS (+ source storage larger than
         * the destination's). Lever MALI_COMPAT_RT_UPD_SRC=1: an other-source UPDATE becomes a BUILD (src cleared,
         * scratch planned for BUILD; default since lead-1110c); =2: only when the source storage is larger; 0: unchanged. */
        if (app_update && pInfos[i].srcAccelerationStructure) {
            struct RtAsTrack *s_t = rt_as_lookup_locked(pInfos[i].srcAccelerationStructure);
            plan[i].src_storage = s_t ? s_t->size : 0;
            plan[i].src_drv = (s_t && s_t->active_handle) ? s_t->active_handle : pInfos[i].srcAccelerationStructure;
            bool other = (s_t && track) ? (s_t != track)
                                        : (pInfos[i].srcAccelerationStructure != pInfos[i].dstAccelerationStructure);
            if (!other) {
                atomic_fetch_add(&g_rt_upd_inplace, 1);
            } else {
                atomic_fetch_add(&g_rt_upd_othersrc, 1);
                bool bigger = plan[i].src_storage && plan[i].storage && plan[i].src_storage > plan[i].storage;
                if (bigger) { atomic_fetch_add(&g_rt_upd_bigger, 1); plan[i].othersrc_bigger = true; }
                const int us = rt_env()->upd_src;
                /* round 4 (Codex r3 #1): the BUILD needs the source content's build size; a destination with less
                 * storage (a COMPACT copy) keeps the UPDATE instead of an overrunning BUILD */
                const VkDeviceSize need_sz = s_t ? s_t->build_size : 0;
                const bool fits = !need_sz || !plan[i].storage || plan[i].storage >= need_sz;
                if (infos[i].mode == VK_BUILD_ACCELERATION_STRUCTURE_MODE_UPDATE_KHR && (us == 1 || (us == 2 && bigger)) &&
                    !fits) {
                    uint64_t nn = atomic_fetch_add(&g_rt_updsrc_nofit, 1) + 1;
                    static _Atomic int64_t nf_ms = 0;
                    int64_t now = rt_now_ms(), pv = atomic_load(&nf_ms);
                    if ((nn == 1 || now - pv >= 1000) && atomic_compare_exchange_strong(&nf_ms, &pv, now))
                        LOG("rt_build: other-source UPDATE kept (MALI_COMPAT_RT_UPD_SRC): destination storage %llu < source "
                            "build size %llu, a BUILD would overrun (total %llu)", (unsigned long long)plan[i].storage,
                            (unsigned long long)need_sz, (unsigned long long)nn);
                } else if (infos[i].mode == VK_BUILD_ACCELERATION_STRUCTURE_MODE_UPDATE_KHR && (us == 1 || (us == 2 && bigger))) {
                    atomic_fetch_add(&g_rt_fb_othersrc, 1);
                    plan[i].othersrc_forced = true;
                    infos[i].mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
                    infos[i].srcAccelerationStructure = VK_NULL_HANDLE;
                }
            }
        }

        if (infos[i].srcAccelerationStructure) {
            struct RtAsTrack *src_t = rt_as_lookup_locked(infos[i].srcAccelerationStructure);
            if (src_t && src_t->active_handle) {
                infos[i].srcAccelerationStructure = src_t->active_handle;
            }
        } else if (infos[i].mode == VK_BUILD_ACCELERATION_STRUCTURE_MODE_UPDATE_KHR) {
            atomic_fetch_add(&g_rt_fb_nosrc, 1);
            infos[i].mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
        }

        /* lead-1110c (Codex 1110-212923 :6923/:6935): the sanitiser below sets primitiveCount 0 for invalid geometry
         * data; an UPDATE must keep its BUILD's counts, so such an UPDATE becomes a BUILD (planned as one). */
        if (infos[i].mode == VK_BUILD_ACCELERATION_STRUCTURE_MODE_UPDATE_KHR &&
            rt_build_has_bad_geometry(&pInfos[i], ppBuildRangeInfos ? ppBuildRangeInfos[i] : NULL)) {
            uint64_t bn = atomic_fetch_add(&g_rt_fb_badgeom, 1) + 1;
            if (g_debug && bn <= 8)
                LOG("rt_build: badgeom #%llu: UPDATE with 0/invalid geometry data (the sanitiser zeroes its primitive "
                    "count) -> BUILD", (unsigned long long)bn);
            infos[i].mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
            infos[i].srcAccelerationStructure = VK_NULL_HANDLE;
        }
        /* lead-1110c (Codex 1110-212923 :6814): every UPDATE the layer turned into a BUILD gets BUILD-sized pool
         * scratch, also with MALI_COMPAT_RT_OWN_SCRATCH=0 (the app sized its scratch for the UPDATE). */
        plan[i].need_pool = app_update && infos[i].mode == VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
    }
    pthread_mutex_unlock(&g_rt_as_lock);

    if (g_debug && d->fpGetAccelerationStructureDeviceAddressKHR) {   /* 1110b: up to 8 other-src-bigger details */
        static _Atomic uint32_t big_lines = 0;
        for (uint32_t i = 0; i < count; i++) {
            if (!plan[i].othersrc_bigger || atomic_load(&big_lines) >= 8) continue;
            uint32_t n = atomic_fetch_add(&big_lines, 1) + 1;
            if (n > 8) break;
            VkAccelerationStructureDeviceAddressInfoKHR dai = {
                .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR };
            dai.accelerationStructure = plan[i].src_drv;
            VkDeviceAddress sb = d->fpGetAccelerationStructureDeviceAddressKHR(d->device, &dai);
            dai.accelerationStructure = infos[i].dstAccelerationStructure;
            VkDeviceAddress db = d->fpGetAccelerationStructureDeviceAddressKHR(d->device, &dai);
            uint64_t prims = 0;
            for (uint32_t g = 0; ppBuildRangeInfos && ppBuildRangeInfos[i] && g < infos[i].geometryCount; g++)
                prims += ppBuildRangeInfos[i][g].primitiveCount;
            LOG("rt_scratch: upd other-src #%u src_bda=0x%llx src_storage=%llu dst_bda=0x%llx dst_storage=%llu prims=%llu%s",
                n, (unsigned long long)sb, (unsigned long long)plan[i].src_storage, (unsigned long long)db,
                (unsigned long long)plan[i].storage, (unsigned long long)prims,
                plan[i].othersrc_forced ? " -> BUILD (UPD_SRC)"
                : infos[i].mode == VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR ? " -> BUILD (layer, other reason)" : "");
        }
    }

    uint32_t as_idx = atomic_fetch_add(&g_as_build_count, 1);   /* was a racy static counter */
    const uint32_t blog_n = rt_env()->build_log;
    bool blog = false;
    if (blog_n) {
        uint32_t mark = atomic_load(&g_rt_first_trace_call);
        blog = as_idx < blog_n || (mark != UINT32_MAX && as_idx >= mark && as_idx - mark < blog_n);
    }
    uint32_t nbatches = rt_assign_as_scratch(d, count, infos, pInfos, ppBuildRangeInfos,
                                             has_safe_scratch ? safe_scratch_bda : 0ULL, plan, blog);

    /* lead-1110c: per destination track, the driver's real AS size of this BUILD (0 if the query was unavailable) - the
     * layer's CLONE copies exactly that many bytes - and the handle is BUILD-made again (refittable). An in-place
     * UPDATE keeps both; an UPDATE from another AS takes the source's size. Dropped builds change nothing. */
    pthread_mutex_lock(&g_rt_as_lock);
    for (uint32_t i = 0; i < count; i++) {
        struct RtAsTrack *t = rt_as_lookup_locked(infos[i].dstAccelerationStructure);
        if (!t) continue;
        if (plan[i].drop) {
            /* round 4: a dropped build leaves the old content: an AS that was unbuilt before this call, or whose handle
             * or built type this call changed (a fresh handle that was never built, Codex r4 #2), is unbuilt (its next
             * UPDATE becomes a BUILD) */
            if (plan[i].prev_built == RT_AS_TYPE_UNBUILT || t->built_type != plan[i].prev_built ||
                t->active_handle != plan[i].prev_active)
                t->built_type = RT_AS_TYPE_UNBUILT;
        } else if (infos[i].mode == VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR) {
            t->as_size = plan[i].drv_as;
            t->build_size = plan[i].drv_as;
            t->refit_ready = true;
        } else {
            struct RtAsTrack *s_t = rt_as_lookup_locked(infos[i].srcAccelerationStructure);
            if (s_t && s_t != t) { t->as_size = s_t->as_size; t->build_size = s_t->build_size; }
        }
        rt_as_acct_locked(t, false);
    }
    pthread_mutex_unlock(&g_rt_as_lock);

    /* Deep geometry and range info sanitization:
     * 1. Protect against CPU user-space pointers (>= 0x6000000000ULL, e.g. FEXAllocator PROT_NONE pages)
     *    and invalid/NULL pointers (< 0x10000ULL).
     * 2. VK_GEOMETRY_TYPE_TRIANGLES_KHR:
     *    - If transformData is invalid/CPU-space, clamp to 0ULL (Vulkan standard Identity transform)
     *      and enforce transformOffset = 0 (VUID-VkAccelerationStructureGeometryTrianglesDataKHR-transformData-04092).
     *    - If vertexData is invalid, point to rt_zero_bda (degenerate triangles culled by hardware).
     *    - If indexData is invalid, set indexType = VK_INDEX_TYPE_NONE_KHR.
     * 3. VK_GEOMETRY_TYPE_INSTANCES_KHR (TLAS):
     *    - If instances.data is invalid, point to rt_zero_bda and set primitiveCount = 0.
     *    - For each instance in the array: if accelerationStructureReference is invalid/CPU-space (such as 0x6d7a3689c0),
     *      set instance.mask = 0 (so ray traversal skips it completely without affecting valid instances)
     *      and point accelerationStructureReference to rt_zero_bda so the driver BVH builder never faults!
     */
    uint32_t curr_geom_idx = 0;
    for (uint32_t i = 0; i < count; i++) {
        uint32_t g_count = infos[i].geometryCount;
        if (g_count == 0 || curr_geom_idx + g_count > total_geoms) {
            new_ppBuildRangeInfos[i] = (ppBuildRangeInfos ? ppBuildRangeInfos[i] : NULL);
            continue;
        }

        VkAccelerationStructureGeometryKHR *info_geoms = &all_geoms[curr_geom_idx];
        VkAccelerationStructureBuildRangeInfoKHR *info_ranges = &all_ranges[curr_geom_idx];
        new_ppBuildRangeInfos[i] = info_ranges;

        for (uint32_t j = 0; j < g_count; j++) {
            uint32_t idx = curr_geom_idx + j;
            const VkAccelerationStructureGeometryKHR *src_geom = NULL;
            if (pInfos[i].pGeometries) src_geom = &pInfos[i].pGeometries[j];
            else if (pInfos[i].ppGeometries && pInfos[i].ppGeometries[j]) src_geom = pInfos[i].ppGeometries[j];

            if (src_geom) {
                all_geoms[idx] = *src_geom;
            } else {
                memset(&all_geoms[idx], 0, sizeof(all_geoms[idx]));
                all_geoms[idx].sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
            }

            if (ppBuildRangeInfos && ppBuildRangeInfos[i]) {
                all_ranges[idx] = ppBuildRangeInfos[i][j];
            } else {
                memset(&all_ranges[idx], 0, sizeof(all_ranges[idx]));
            }

            if (all_geoms[idx].geometryType == VK_GEOMETRY_TYPE_TRIANGLES_KHR) {
                VkAccelerationStructureGeometryTrianglesDataKHR *tri = &all_geoms[idx].geometry.triangles;
                if (rt_is_invalid_gpu_bda(tri->transformData.deviceAddress)) {
                    tri->transformData.deviceAddress = 0ULL;
                }
                if (tri->transformData.deviceAddress == 0ULL) {
                    all_ranges[idx].transformOffset = 0;
                }
                if (rt_is_invalid_gpu_bda(tri->vertexData.deviceAddress) || tri->vertexData.deviceAddress == 0ULL) {
                    tri->vertexData.deviceAddress = rt_get_zero_bda(d);
                    if (tri->vertexStride < 12) tri->vertexStride = 12;
                    /* 1110b: nothing may be read past the 4 KB zero buffer (> ~113 tris did): no primitives */
                    all_ranges[idx].primitiveCount = 0;
                }
                if (tri->indexType != VK_INDEX_TYPE_NONE_KHR &&
                    (rt_is_invalid_gpu_bda(tri->indexData.deviceAddress) || tri->indexData.deviceAddress == 0ULL)) {
                    tri->indexType = VK_INDEX_TYPE_NONE_KHR;
                    tri->indexData.deviceAddress = 0ULL;
                }
            } else if (all_geoms[idx].geometryType == VK_GEOMETRY_TYPE_AABBS_KHR) {
                VkAccelerationStructureGeometryAabbsDataKHR *aabb = &all_geoms[idx].geometry.aabbs;
                if (rt_is_invalid_gpu_bda(aabb->data.deviceAddress) || aabb->data.deviceAddress == 0ULL) {
                    aabb->data.deviceAddress = rt_get_zero_bda(d);
                    if (aabb->stride < 24) aabb->stride = 24;
                    all_ranges[idx].primitiveCount = 0;   /* 1110b: as for triangles above */
                }
            } else if (all_geoms[idx].geometryType == VK_GEOMETRY_TYPE_INSTANCES_KHR) {
                VkAccelerationStructureGeometryInstancesDataKHR *inst = &all_geoms[idx].geometry.instances;
                VkDeviceAddress inst_bda = inst->data.deviceAddress;
                uint32_t prim_cnt = all_ranges[idx].primitiveCount;
                uint32_t prim_off = all_ranges[idx].primitiveOffset;
                if (rt_is_invalid_gpu_bda(inst_bda) || inst_bda == 0ULL ||
                    rt_is_invalid_gpu_bda(inst_bda + (uint64_t)prim_off)) {
                    inst->data.deviceAddress = rt_get_zero_bda(d);
                    all_ranges[idx].primitiveCount = 0;
                    all_ranges[idx].primitiveOffset = 0;
                } else if (inst->arrayOfPointers == VK_FALSE && prim_cnt > 0) {
                    void *host_base = rt_resolve_gpu_bda_to_host(inst_bda + prim_off, sizeof(VkAccelerationStructureInstanceKHR));
                    if (host_base) {
                        VkAccelerationStructureInstanceKHR *inst_arr = (VkAccelerationStructureInstanceKHR *)host_base;
                        uint32_t bad_refs = 0, null_refs = 0;
                        for (uint32_t k = 0; k < prim_cnt; k++) {
                            if ((k & 63u) == 0 && k > 0) {
                                if (!rt_is_host_address_accessible((const void *)&inst_arr[k], sizeof(VkAccelerationStructureInstanceKHR))) {
                                    break;
                                }
                            }
                            /* 11-10: reference 0 = Vulkan-legal inactive instance, the blob builder handles it
                             * (rq-probe `inactive` PASS 11-06): untouched. A non-zero invalid reference becomes 0
                             * (+ mask 0). It used to point at the 4 KB zero buffer, from which the Mali builder read a
                             * "BLAS header" of zeros and faulted at 0x0..0x480 (11-09: 34 READ faults). */
                            VkDeviceAddress as_ref = inst_arr[k].accelerationStructureReference;
                            if (as_ref == 0ULL) {
                                null_refs++;
                            } else if (rt_is_invalid_gpu_bda(as_ref)) {
                                inst_arr[k].mask = 0;
                                inst_arr[k].accelerationStructureReference = 0ULL;
                                bad_refs++;
                            }
                        }
                        if (bad_refs > 0 || as_idx < 10 || (as_idx % 120) == 0) {
                            COMPAT_INFO("RT_EMU: TLAS build #%u instances=%u bad_refs=%u inactive=%u (inst_bda=0x%llx, host=%p)",
                                        as_idx, prim_cnt, bad_refs, null_refs, (unsigned long long)inst_bda, host_base);
                        }
                        cache_maintenance(host_base, (VkDeviceSize)prim_cnt * sizeof(VkAccelerationStructureInstanceKHR), false);
                    }
                } else if (inst->arrayOfPointers == VK_TRUE && prim_cnt > 0) {
                    void *ptr_base = rt_resolve_gpu_bda_to_host(inst_bda + prim_off, sizeof(VkDeviceAddress));
                    if (ptr_base) {
                        VkDeviceAddress *ptr_arr = (VkDeviceAddress *)ptr_base;
                        uint32_t bad_refs = 0, null_refs = 0;
                        for (uint32_t k = 0; k < prim_cnt; k++) {
                            VkDeviceAddress inst_elem_bda = ptr_arr[k];
                            if (inst_elem_bda == 0ULL || rt_is_invalid_gpu_bda(inst_elem_bda)) {
                                /* a bad POINTER to an instance: point it at the zero buffer = an all-zero instance,
                                 * i.e. reference 0 (inactive), mask 0 - the builder reads no BLAS through it */
                                ptr_arr[k] = rt_get_zero_bda(d);
                                bad_refs++;
                                continue;
                            }
                            void *elem_host = rt_resolve_gpu_bda_to_host(inst_elem_bda, sizeof(VkAccelerationStructureInstanceKHR));
                            if (elem_host) {
                                VkAccelerationStructureInstanceKHR *one_inst = (VkAccelerationStructureInstanceKHR *)elem_host;
                                VkDeviceAddress as_ref = one_inst->accelerationStructureReference;
                                if (as_ref == 0ULL) {
                                    null_refs++;               /* legal inactive instance: untouched (11-10) */
                                } else if (rt_is_invalid_gpu_bda(as_ref)) {
                                    one_inst->mask = 0;
                                    one_inst->accelerationStructureReference = 0ULL;   /* was rt_zero_bda (11-10) */
                                    bad_refs++;
                                }
                                cache_maintenance(elem_host, sizeof(VkAccelerationStructureInstanceKHR), false);
                            }
                        }
                        if (bad_refs > 0 || as_idx < 10 || (as_idx % 120) == 0) {
                            COMPAT_INFO("RT_EMU: TLAS build #%u (array of ptrs) instances=%u bad_refs=%u inactive=%u (inst_bda=0x%llx, host=%p)",
                                        as_idx, prim_cnt, bad_refs, null_refs, (unsigned long long)inst_bda, ptr_base);
                        }
                        cache_maintenance(ptr_base, (VkDeviceSize)prim_cnt * sizeof(VkDeviceAddress), false);
                    }
                }
            }
        }

        infos[i].pGeometries = info_geoms;
        infos[i].ppGeometries = NULL;
        curr_geom_idx += g_count;
    }

    if (as_idx < 10 || (as_idx % 60) == 0) {
        for (uint32_t i = 0; i < count; i++) {
            VkDeviceAddress dst_bda = 0ULL;
            if (d->fpGetAccelerationStructureDeviceAddressKHR && infos[i].dstAccelerationStructure) {
                VkAccelerationStructureDeviceAddressInfoKHR dai = {
                    .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR,
                    .accelerationStructure = infos[i].dstAccelerationStructure,
                };
                dst_bda = d->fpGetAccelerationStructureDeviceAddressKHR(d->device, &dai);
            }
            VkDeviceAddress inst_bda = 0ULL;
            if (infos[i].type == VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR && infos[i].geometryCount > 0) {
                if (infos[i].pGeometries) inst_bda = infos[i].pGeometries[0].geometry.instances.data.deviceAddress;
                else if (infos[i].ppGeometries && infos[i].ppGeometries[0]) inst_bda = infos[i].ppGeometries[0]->geometry.instances.data.deviceAddress;
            }
            COMPAT_INFO("CmdBuildAS #%u [%u/%u]: type=%u mode=%u dst=%p (bda=0x%llx) src=%p geoms=%u inst_bda=0x%llx scratch=0x%llx (orig=0x%llx)",
                        as_idx, i + 1, count, infos[i].type, infos[i].mode,
                        (void *)infos[i].dstAccelerationStructure,
                        (unsigned long long)dst_bda,
                        (void *)infos[i].srcAccelerationStructure,
                        infos[i].geometryCount,
                        (unsigned long long)inst_bda,
                        (unsigned long long)infos[i].scratchData.deviceAddress,
                        (unsigned long long)pInfos[i].scratchData.deviceAddress);
        }
    }

    if (blog) {
        for (uint32_t i = 0; i < count; i++) {
            VkDeviceAddress dst_bda = 0ULL;
            if (d->fpGetAccelerationStructureDeviceAddressKHR && infos[i].dstAccelerationStructure) {
                VkAccelerationStructureDeviceAddressInfoKHR dai = {
                    .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR,
                    .accelerationStructure = infos[i].dstAccelerationStructure,
                };
                dst_bda = d->fpGetAccelerationStructureDeviceAddressKHR(d->device, &dai);
            }
            uint64_t prims = 0;
            const VkAccelerationStructureBuildRangeInfoKHR *ri = new_ppBuildRangeInfos[i];
            for (uint32_t g = 0; ri && g < infos[i].geometryCount; g++) prims += ri[g].primitiveCount;
            /* 1110b: first geometry's format (after the layer's sanitising) */
            char g0s[160] = " g0=none";
            const VkAccelerationStructureGeometryKHR *g0 = NULL;
            if (infos[i].geometryCount)
                g0 = infos[i].pGeometries ? &infos[i].pGeometries[0]
                                          : (infos[i].ppGeometries ? infos[i].ppGeometries[0] : NULL);
            if (g0 && g0->geometryType == VK_GEOMETRY_TYPE_TRIANGLES_KHR) {
                const VkAccelerationStructureGeometryTrianglesDataKHR *t = &g0->geometry.triangles;
                snprintf(g0s, sizeof(g0s), " g0=tri fmt=%u vstride=%llu idx=%u maxv=%u xform=%d gflags=0x%x",
                         (unsigned)t->vertexFormat, (unsigned long long)t->vertexStride, (unsigned)t->indexType,
                         t->maxVertex, t->transformData.deviceAddress != 0ULL, (unsigned)g0->flags);
            } else if (g0 && g0->geometryType == VK_GEOMETRY_TYPE_AABBS_KHR) {
                snprintf(g0s, sizeof(g0s), " g0=aabb stride=%llu gflags=0x%x",
                         (unsigned long long)g0->geometry.aabbs.stride, (unsigned)g0->flags);
            } else if (g0 && g0->geometryType == VK_GEOMETRY_TYPE_INSTANCES_KHR) {
                snprintf(g0s, sizeof(g0s), " g0=inst aop=%u gflags=0x%x",
                         (unsigned)g0->geometry.instances.arrayOfPointers, (unsigned)g0->flags);
            }
            VkDeviceAddress src_bda = 0ULL;   /* 1110b: the APP's source (also when a lever cleared it) */
            if (d->fpGetAccelerationStructureDeviceAddressKHR && plan[i].src_drv) {
                VkAccelerationStructureDeviceAddressInfoKHR sai = {
                    .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR,
                    .accelerationStructure = plan[i].src_drv };
                src_bda = d->fpGetAccelerationStructureDeviceAddressKHR(d->device, &sai);
            }
            LOG("rt_build: call=%u %u/%u type=%u mode=%u app_mode=%u flags=0x%x dst_bda=0x%llx storage=%llu src_bda=0x%llx "
                "src_storage=%llu drv_as=%llu drv_scratch=%llu scratch=0x%llx prims=%llu batch=%u%s%s", as_idx, i + 1,
                count, infos[i].type, infos[i].mode, pInfos[i].mode, infos[i].flags, (unsigned long long)dst_bda,
                (unsigned long long)plan[i].storage, (unsigned long long)src_bda, (unsigned long long)plan[i].src_storage,
                (unsigned long long)plan[i].drv_as, (unsigned long long)plan[i].drv_scratch,
                (unsigned long long)infos[i].scratchData.deviceAddress, (unsigned long long)prims, plan[i].batch,
                g0s, plan[i].drop ? " DROPPED" : "");
        }
    }

    /* 11-10: dropped builds are removed (order kept); each batch from rt_assign_as_scratch goes out as its own driver
     * call between the full pre/post barriers, so batch b+1 may reuse the scratch pool of batch b. */
    uint32_t live = 0;
    for (uint32_t i = 0; i < count; i++) {
        if (plan[i].drop) continue;
        if (live != i) {
            infos[live] = infos[i];
            new_ppBuildRangeInfos[live] = new_ppBuildRangeInfos[i];
            plan[live] = plan[i];
        }
        live++;
    }
    if (nbatches > 1 && g_debug) {
        static _Atomic int64_t split_ms = 0;
        int64_t now = rt_now_ms(), pv = atomic_load(&split_ms);
        if (now - pv >= 1000 && atomic_compare_exchange_strong(&split_ms, &pv, now))
            LOG("rt_scratch: CmdBuildAS #%u split into %u driver calls (%u builds, %u dropped)",
                as_idx, nbatches, live, count - live);
    }

    for (uint32_t first = 0; first < live;) {
        uint32_t n = 1;
        while (first + n < live && plan[first + n].batch == plan[first].batch) n++;

        rt_as_build_barrier(d, commandBuffer, true);

        const VkAccelerationStructureBuildRangeInfoKHR * const *pass_ranges =
            ppBuildRangeInfos ? (const VkAccelerationStructureBuildRangeInfoKHR * const *)&new_ppBuildRangeInfos[first]
                              : NULL;

        d->fpCmdBuildAccelerationStructuresKHR(commandBuffer, n, &infos[first], pass_ranges);

        rt_as_build_barrier(d, commandBuffer, false);
        first += n;
    }

out:
    if (all_geoms != stack_all_geoms) free(all_geoms);
    if (all_ranges != stack_all_ranges) free(all_ranges);
    if (new_ppBuildRangeInfos != stack_ppRangeInfos) free((void *)new_ppBuildRangeInfos);
    if (infos != stack_infos) free(infos);
    if (plan != stack_plan) free(plan);
    rt_report(d, false);
}

/* lead-1110c: full barriers around the layer's CLONE copy, the same kind as around AS builds. Before: every earlier
 * write (source built, destination last used) is done and visible to the transfer, nothing still reads the
 * destination. After: the copied bytes are visible to AS builds/refits, ray queries, shaders and later transfers. */
static void rt_as_copy_barrier(struct DeviceData *d, VkCommandBuffer cmd, bool pre) {
    if (!d->fpCmdPipelineBarrier) return;
    VkMemoryBarrier mb = { .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER };
    if (pre) {
        mb.srcAccessMask = VK_ACCESS_HOST_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT |
                           VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_MEMORY_WRITE_BIT |
                           VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;
        mb.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT |
                           VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
    } else {
        mb.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
        mb.dstAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR |
                           VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR |
                           VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_MEMORY_READ_BIT |
                           VK_ACCESS_UNIFORM_READ_BIT | VK_ACCESS_TRANSFER_READ_BIT;
    }
    d->fpCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                            0, 1, &mb, 0, NULL, 0, NULL);
}

/* lead-1110c: as-probe 11-10 (blob, no layer): vkCmdCopyAccelerationStructureKHR CLONE copies the SOURCE handle's
 * created size (vkd3d-proton: rest of the buffer), starting up to 3840 B before the destination and running past the
 * destination VkBuffer - the W3 GPU write faults (L2: clone=212 then in-place refits of the clones). A freshly built
 * AS occupies exactly its own bytes, at most the driver's accelerationStructureSize of its build.
 * Lever MALI_COMPAT_RT_CLONE=1 (default): a CLONE is done by the layer: vkCmdCopyBuffer of
 * min(source AS size, source storage, destination storage) bytes from (source buffer, offset) to (destination
 * buffer, offset) between rt_as_copy_barrier. Falls back to the driver's copy (logged how=drv reason=) when a track is
 * missing, the source's AS size is unknown (0), a buffer lacks the transfer usage the layer adds in compat_CreateBuffer,
 * vkCmdCopyBuffer is unavailable, or source and destination ranges overlap in one buffer. COMPACT always goes to the
 * driver (as-probe: it stays inside the destination's 64 KB-rounded compacted size).
 * Tracks: the destination takes the source's built type and AS size and is not refittable until its next BUILD
 * (MALI_COMPAT_RT_CLONE_REFIT). */
static VKAPI_ATTR void VKAPI_CALL
compat_CmdCopyAccelerationStructureKHR(
    VkCommandBuffer                             commandBuffer,
    const VkCopyAccelerationStructureInfoKHR   *pInfo)
{
    struct DeviceData *d = get_device_data_by_cmd(commandBuffer);
    if (!d || !d->fpCmdCopyAccelerationStructureKHR) return;
    if (!pInfo) {
        d->fpCmdCopyAccelerationStructureKHR(commandBuffer, pInfo);
        return;
    }
    VkCopyAccelerationStructureInfoKHR copy = *pInfo;
    VkDeviceSize src_storage = 0, dst_storage = 0, dst_rest = 0, src_as = 0, size = 0;
    VkBuffer sbuf = VK_NULL_HANDLE, dbuf = VK_NULL_HANDLE;
    VkDeviceSize soff = 0, doff = 0;
    const bool is_clone = copy.mode == VK_COPY_ACCELERATION_STRUCTURE_MODE_CLONE_KHR;
    const char *reason = NULL;   /* NULL = the layer copies (how=buf) */
    pthread_mutex_lock(&g_rt_as_lock);
    struct RtAsTrack *st = rt_as_lookup_locked(pInfo->src);
    if (st) { copy.src = st->active_handle; src_storage = st->size; src_as = st->as_size; sbuf = st->buffer; soff = st->offset; }
    struct RtAsTrack *dt = rt_as_lookup_locked(pInfo->dst);
    if (dt) { copy.dst = dt->active_handle; dst_storage = dt->size; dst_rest = dt->buf_rest; dbuf = dt->buffer; doff = dt->offset; }
    if (!is_clone) reason = "mode";
    else if (!rt_env()->clone) reason = "lever";
    else if (!st || !dt) reason = "untracked";
    else if (!src_as) reason = "size";
    else if (!sbuf || !dbuf || !src_storage || !dst_storage) reason = "buffer";
    else if (!st->xfer_ok || !dt->xfer_ok) reason = "usage";
    else if (!d->fpCmdCopyBuffer) reason = "nofn";
    else {
        size = src_as;
        if (size > src_storage) size = src_storage;
        if (size > dst_storage) size = dst_storage;
        if (sbuf == dbuf && soff < doff + size && doff < soff + size) reason = "overlap";
    }
    if (reason) size = 0;
    /* 1110b: the copy leaves dst built with the source's type (it stayed UNBUILT before, so its first UPDATE was
     * silently turned into a BUILD; with MALI_COMPAT_RT_AS_TRUST=1 no such forcing happens at all).
     * lead-1110c: + the source's AS size, and the destination is not refittable until its next BUILD. */
    if (st && dt && st != dt) {
        if (st->built_type != RT_AS_TYPE_UNBUILT) dt->built_type = st->built_type;
        /* round 4 (Codex r3 #2): after a COMPACT the content's byte extent is the compacted size, which the layer does
         * not know: as_size 0 (a later CLONE from it goes to the driver, logged reason=size); build_size travels on */
        dt->as_size = copy.mode == VK_COPY_ACCELERATION_STRUCTURE_MODE_COMPACT_KHR ? 0 : st->as_size;
        dt->build_size = st->build_size;
        dt->refit_ready = false;
        rt_as_acct_locked(dt, false);
    }
    pthread_mutex_unlock(&g_rt_as_lock);

    /* 1110b, MALI_COMPAT_DEBUG: what the app copies; lead-1110c: + how=buf|drv size= reason= */
    static _Atomic uint64_t c_total = 0, c_clone = 0, c_compact = 0, c_buf = 0, c_drv = 0;
    uint64_t cn = atomic_fetch_add(&c_total, 1) + 1;
    if (is_clone) atomic_fetch_add(&c_clone, 1);
    else if (copy.mode == VK_COPY_ACCELERATION_STRUCTURE_MODE_COMPACT_KHR) atomic_fetch_add(&c_compact, 1);
    uint64_t nbuf = reason ? atomic_load(&c_buf) : atomic_fetch_add(&c_buf, 1) + 1;
    uint64_t ndrv = reason ? atomic_fetch_add(&c_drv, 1) + 1 : atomic_load(&c_drv);
    const char *mn = is_clone ? "CLONE"
                   : copy.mode == VK_COPY_ACCELERATION_STRUCTURE_MODE_COMPACT_KHR ? "COMPACT"
                   : copy.mode == VK_COPY_ACCELERATION_STRUCTURE_MODE_SERIALIZE_KHR ? "SERIALIZE"
                   : copy.mode == VK_COPY_ACCELERATION_STRUCTURE_MODE_DESERIALIZE_KHR ? "DESERIALIZE" : "?";
    if (g_debug) {
        if (cn <= 64) {
            VkDeviceAddress sb = 0, db = 0;
            if (d->fpGetAccelerationStructureDeviceAddressKHR) {
                VkAccelerationStructureDeviceAddressInfoKHR dai = {
                    .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR };
                if (copy.src) { dai.accelerationStructure = copy.src; sb = d->fpGetAccelerationStructureDeviceAddressKHR(d->device, &dai); }
                if (copy.dst) { dai.accelerationStructure = copy.dst; db = d->fpGetAccelerationStructureDeviceAddressKHR(d->device, &dai); }
            }
            LOG("rt_copy: #%llu mode=%s src=0x%llx src_storage=%llu src_as=%llu dst=0x%llx dst_storage=%llu "
                "dst_buf_rest=%llu how=%s size=%llu%s%s", (unsigned long long)cn, mn, (unsigned long long)sb,
                (unsigned long long)src_storage, (unsigned long long)src_as, (unsigned long long)db,
                (unsigned long long)dst_storage, (unsigned long long)dst_rest, reason ? "drv" : "buf",
                (unsigned long long)size, reason ? " reason=" : "", reason ? reason : "");
        }
        static _Atomic int64_t c_ms = 0;
        int64_t now = rt_now_ms(), pv = atomic_load(&c_ms);
        if (now - pv >= 1000 && atomic_compare_exchange_strong(&c_ms, &pv, now))
            LOG("rt_copy: total=%llu clone=%llu compact=%llu buf=%llu drv=%llu", (unsigned long long)cn,
                (unsigned long long)atomic_load(&c_clone), (unsigned long long)atomic_load(&c_compact),
                (unsigned long long)nbuf, (unsigned long long)ndrv);
    }
    if (is_clone && reason && strcmp(reason, "lever") != 0) {   /* a CLONE the driver does: always logged, <= 1/s */
        static _Atomic int64_t f_ms = 0;
        static _Atomic uint64_t f_n = 0;
        uint64_t fn = atomic_fetch_add(&f_n, 1) + 1;
        int64_t now = rt_now_ms(), pv = atomic_load(&f_ms);
        if ((fn == 1 || now - pv >= 1000) && atomic_compare_exchange_strong(&f_ms, &pv, now))
            LOG("rt_copy: CLONE how=drv reason=%s src_storage=%llu src_as=%llu dst_storage=%llu (driver clones so far: %llu)",
                reason, (unsigned long long)src_storage, (unsigned long long)src_as, (unsigned long long)dst_storage,
                (unsigned long long)fn);
    }

    if (!reason) {
        VkBufferCopy region = { .srcOffset = soff, .dstOffset = doff, .size = size };
        rt_as_copy_barrier(d, commandBuffer, true);
        d->fpCmdCopyBuffer(commandBuffer, sbuf, dbuf, 1, &region);
        rt_as_copy_barrier(d, commandBuffer, false);
        return;
    }

    d->fpCmdCopyAccelerationStructureKHR(commandBuffer, &copy);

    if (d->fpCmdPipelineBarrier) {
        VkMemoryBarrier mb = {
            .sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
            .pNext         = NULL,
            .srcAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR | VK_ACCESS_TRANSFER_WRITE_BIT,
            .dstAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR | VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_MEMORY_READ_BIT,
        };
        d->fpCmdPipelineBarrier(commandBuffer,
                                VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR | VK_PIPELINE_STAGE_TRANSFER_BIT,
                                VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                                0,
                                1, &mb,
                                0, NULL,
                                0, NULL);
    }
}

static VKAPI_ATTR void VKAPI_CALL
compat_CmdWriteAccelerationStructuresPropertiesKHR(
    VkCommandBuffer                             commandBuffer,
    uint32_t                                    accelerationStructureCount,
    const VkAccelerationStructureKHR           *pAccelerationStructures,
    VkQueryType                                 queryType,
    VkQueryPool                                 queryPool,
    uint32_t                                    firstQuery)
{
    struct DeviceData *d = get_device_data_by_cmd(commandBuffer);
    if (!d || !d->fpCmdWriteAccelerationStructuresPropertiesKHR) return;
    if (!pAccelerationStructures || accelerationStructureCount == 0) {
        d->fpCmdWriteAccelerationStructuresPropertiesKHR(commandBuffer, accelerationStructureCount,
                                                         pAccelerationStructures, queryType, queryPool, firstQuery);
        return;
    }

    VkAccelerationStructureKHR stack_as[16];
    VkAccelerationStructureKHR *as_arr = stack_as;
    if (accelerationStructureCount > 16) {
        as_arr = (VkAccelerationStructureKHR *)malloc(accelerationStructureCount * sizeof(*as_arr));
        if (!as_arr) as_arr = stack_as;
    }
    uint32_t cnt = (as_arr == stack_as && accelerationStructureCount > 16) ? 16 : accelerationStructureCount;

    pthread_mutex_lock(&g_rt_as_lock);
    for (uint32_t i = 0; i < cnt; i++) {
        struct RtAsTrack *t = rt_as_lookup_locked(pAccelerationStructures[i]);
        as_arr[i] = t ? t->active_handle : pAccelerationStructures[i];
    }
    pthread_mutex_unlock(&g_rt_as_lock);

    /* 1110b, MALI_COMPAT_DEBUG: which AS properties the app queries (first 16 calls + a summary <= 1/s) */
    static _Atomic uint64_t q_total = 0, q_compacted = 0, q_serial = 0;
    uint64_t qn = atomic_fetch_add(&q_total, 1) + 1;
    if (queryType == VK_QUERY_TYPE_ACCELERATION_STRUCTURE_COMPACTED_SIZE_KHR) atomic_fetch_add(&q_compacted, 1);
    else if (queryType == VK_QUERY_TYPE_ACCELERATION_STRUCTURE_SERIALIZATION_SIZE_KHR) atomic_fetch_add(&q_serial, 1);
    if (g_debug) {
        if (qn <= 16) {
            const char *tn = queryType == VK_QUERY_TYPE_ACCELERATION_STRUCTURE_COMPACTED_SIZE_KHR ? "COMPACTED_SIZE"
                           : queryType == VK_QUERY_TYPE_ACCELERATION_STRUCTURE_SERIALIZATION_SIZE_KHR ? "SERIALIZATION_SIZE"
                           : queryType == VK_QUERY_TYPE_ACCELERATION_STRUCTURE_SIZE_KHR ? "SIZE"
                           : queryType == VK_QUERY_TYPE_ACCELERATION_STRUCTURE_SERIALIZATION_BOTTOM_LEVEL_POINTERS_KHR
                               ? "SERIALIZATION_BOTTOM_LEVEL_POINTERS" : "?";
            LOG("rt_query: #%llu type=%s count=%u", (unsigned long long)qn, tn, accelerationStructureCount);
        }
        static _Atomic int64_t q_ms = 0;
        int64_t now = rt_now_ms(), pv = atomic_load(&q_ms);
        if (now - pv >= 1000 && atomic_compare_exchange_strong(&q_ms, &pv, now))
            LOG("rt_query: total=%llu compacted=%llu serialization=%llu", (unsigned long long)qn,
                (unsigned long long)atomic_load(&q_compacted), (unsigned long long)atomic_load(&q_serial));
    }

    d->fpCmdWriteAccelerationStructuresPropertiesKHR(commandBuffer, cnt, as_arr, queryType, queryPool, firstQuery);

    if (as_arr != stack_as) free(as_arr);
}

static VKAPI_ATTR VkDeviceAddress VKAPI_CALL
compat_GetAccelerationStructureDeviceAddressKHR(
    VkDevice                                           device,
    const VkAccelerationStructureDeviceAddressInfoKHR *pInfo)
{
    struct DeviceData *d = get_device_data(device);
    if (!d || !d->fpGetAccelerationStructureDeviceAddressKHR || !pInfo) return 0ULL;

    VkAccelerationStructureDeviceAddressInfoKHR info = *pInfo;
    pthread_mutex_lock(&g_rt_as_lock);
    struct RtAsTrack *t = rt_as_lookup_locked(pInfo->accelerationStructure);
    if (t && t->active_handle) info.accelerationStructure = t->active_handle;
    pthread_mutex_unlock(&g_rt_as_lock);

    VkDeviceAddress bda = d->fpGetAccelerationStructureDeviceAddressKHR(device, &info);
    if (rt_is_invalid_gpu_bda(bda)) {
        /* 1110b: 0 (= an inactive instance in a TLAS), no longer rt_zero_bda: that zero buffer is in range, passed the
         * TLAS instance check, and the Mali builder read it as a BLAS header (faults at 0x0..0x480). */
        COMPAT_INFO("RT_EMU: GetASDeviceAddress returned invalid BDA 0x%llx for AS %p, returning 0 (inactive)",
                    (unsigned long long)bda, (void *)pInfo->accelerationStructure);
        return 0ULL;
    }
    return bda;
}

static VKAPI_ATTR void VKAPI_CALL
compat_GetAccelerationStructureBuildSizesKHR(
    VkDevice                                            device,
    VkAccelerationStructureBuildTypeKHR                 buildType,
    const VkAccelerationStructureBuildGeometryInfoKHR  *pBuildInfo,
    const uint32_t                                     *pMaxPrimitiveCounts,
    VkAccelerationStructureBuildSizesInfoKHR           *pSizeInfo)
{
    struct DeviceData *d = get_device_data(device);
    if (!d || !d->fpGetAccelerationStructureBuildSizesKHR || !pSizeInfo) return;

    d->fpGetAccelerationStructureBuildSizesKHR(device, buildType, pBuildInfo, pMaxPrimitiveCounts, pSizeInfo);
    /* 1110b lever MALI_COMPAT_RT_SIZE_PAD=0: the driver's own sizes (as-probe 11-10: the blob builder stays inside
     * the storage and scratch it reports). Default 1 = the inflation below. */
    if (!rt_env()->size_pad) return;

    /* Mali proprietary CSF driver underestimates scratch and AS storage
     * requirements for complex geometry and TLAS builds, causing hardware
     * MMU translation faults (0x5002c3 / 0x45003c3) past buffer bounds.
     * Scale up with ample headroom and pad to 64KB. */
    VkDeviceSize as_sz  = (pSizeInfo->accelerationStructureSize * 3ULL) / 2ULL + (2ULL * 1024 * 1024);
    VkDeviceSize scr_sz = (pSizeInfo->buildScratchSize * 2ULL) + (4ULL * 1024 * 1024);
    VkDeviceSize upd_sz = (pSizeInfo->updateScratchSize * 2ULL) + (4ULL * 1024 * 1024);

    #define PAD_64K(sz) (((sz) + 0xffffULL) & ~0xffffULL)
    pSizeInfo->accelerationStructureSize = PAD_64K(as_sz);
    pSizeInfo->buildScratchSize          = PAD_64K(scr_sz);
    pSizeInfo->updateScratchSize         = PAD_64K(upd_sz);
    #undef PAD_64K

    if (pSizeInfo->updateScratchSize < pSizeInfo->buildScratchSize) {
        pSizeInfo->updateScratchSize = pSizeInfo->buildScratchSize;
    }
}


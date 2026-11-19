// SPDX-License-Identifier: GPL-3.0-or-later
// rt_report.h: periodic report (RT_MEM_LOG_S / RT_STATS) and device teardown accounting. Included by mali_rt_emu.h.
/* ── round 4: the periodic report (rt_mem / vk_mem / rt_rays) ─────────────────────────────────────────────────────
 * Every MALI_COMPAT_RT_MEM_LOG_S seconds (default 10) from the AS-build / trace-record paths (one thread prints, the
 * others skip on the trylock), and "final" at vkDestroyDevice. Units: KB = 1024 B; "n/KB" pairs = live count / live KB. */
/* exact sum of every live rt_mem byte counter (AS created + driver size, AS buffers, layer GPU memory, layer host
 * memory, megakernel modules): 0 after everything was destroyed */
static int64_t rt_live_bytes(void) {
    struct RtAcct *all[] = { &g_acct_as_all, &g_acct_as_drv_all, &g_acct_asbuf, &g_acct_pool, &g_acct_ring, &g_acct_zero,
                             &g_acct_tracks, &g_acct_modules, &g_acct_stages, &g_acct_pipes, &g_acct_cmdst,
                             &g_acct_mega_mod };
    int64_t t = 0;
    for (size_t i = 0; i < sizeof all / sizeof all[0]; i++) t += atomic_load_explicit(&all[i]->bytes, memory_order_relaxed);
    return t;
}
static pthread_mutex_t g_rt_report_lock = PTHREAD_MUTEX_INITIALIZER;
static _Atomic int64_t g_rt_report_t0;   /* start of the first rt_rays window */
static void rt_report(struct DeviceData *d, bool final) {
    (void)d;
    const uint32_t every = rt_env()->mem_log_s;
    static _Atomic int64_t next_ms = 0;
    int64_t now = rt_now_ms();
    if (!final) {
        int64_t t0 = 0;   /* Codex r4 #6: the window starts at the first call also with MALI_COMPAT_RT_MEM_LOG_S=0 */
        if (!atomic_load_explicit(&g_rt_report_t0, memory_order_relaxed))
            atomic_compare_exchange_strong(&g_rt_report_t0, &t0, now);
        if (!every) return;
        int64_t nx = atomic_load_explicit(&next_ms, memory_order_relaxed);
        if (nx == 0) {   /* first call: print after one interval */
            atomic_compare_exchange_strong(&next_ms, &nx, now + (int64_t)every * 1000);
            return;
        }
        if (now < nx) return;
        if (pthread_mutex_trylock(&g_rt_report_lock) != 0) return;
        if (!atomic_compare_exchange_strong(&next_ms, &nx, now + (int64_t)every * 1000)) {
            pthread_mutex_unlock(&g_rt_report_lock);
            return;
        }
    } else {
        pthread_mutex_lock(&g_rt_report_lock);
    }
    const char *tag = final ? " final" : "";
    LOG("rt_mem: t=%lld ms%s as=%lld (blas=%lld tlas=%lld unbuilt=%lld) created_kb=%lld (blas=%lld tlas=%lld unbuilt=%lld) "
        "peak_kb=%lld | drv_size n=%lld kb=%lld (blas=%lld tlas=%lld) peak_kb=%lld | asbuf=%lld kb=%lld peak_kb=%lld | "
        "pool=%lld kb=%lld max_per_call_kb=%llu max_slice_kb=%lld | ring_kb=%lld zero_kb=%lld | host n/kb: tracks=%lld/%lld "
        "modules=%lld/%lld (peak_kb %lld) stages=%lld/%lld (peak_kb %lld, freed_kb %lld) pipes=%lld/%lld cmdst=%lld/%lld "
        "bufmap_kb=%lld | mega: requested=%lld compiling=%lld live=%lld failed=%lld capped=%lld modules=%lld/%lld "
        "spv_total_kb=%lld dumps=%lld/%lld | live_bytes=%lld",
        (long long)now, tag,
        (long long)rt_cnt(&g_acct_as_all), (long long)rt_cnt(&g_acct_as[1]), (long long)rt_cnt(&g_acct_as[2]),
        (long long)rt_cnt(&g_acct_as[0]),
        (long long)rt_kb(&g_acct_as_all, 0), (long long)rt_kb(&g_acct_as[1], 0), (long long)rt_kb(&g_acct_as[2], 0),
        (long long)rt_kb(&g_acct_as[0], 0), (long long)rt_kb(&g_acct_as_all, 1),
        (long long)rt_cnt(&g_acct_as_drv_all), (long long)rt_kb(&g_acct_as_drv_all, 0),
        (long long)rt_kb(&g_acct_as_drv[1], 0), (long long)rt_kb(&g_acct_as_drv[2], 0),
        (long long)rt_kb(&g_acct_as_drv_all, 1),
        (long long)rt_cnt(&g_acct_asbuf), (long long)rt_kb(&g_acct_asbuf, 0), (long long)rt_kb(&g_acct_asbuf, 1),
        (long long)rt_cnt(&g_acct_pool), (long long)rt_kb(&g_acct_pool, 0),
        (unsigned long long)(atomic_load(&g_rt_scr_maxcall) >> 10),
        (long long)(atomic_load_explicit(&g_rt_scr_maxslice, memory_order_relaxed) >> 10),
        (long long)rt_kb(&g_acct_ring, 0), (long long)rt_kb(&g_acct_zero, 0),
        (long long)rt_cnt(&g_acct_tracks), (long long)rt_kb(&g_acct_tracks, 0),
        (long long)rt_cnt(&g_acct_modules), (long long)rt_kb(&g_acct_modules, 0), (long long)rt_kb(&g_acct_modules, 1),
        (long long)rt_cnt(&g_acct_stages), (long long)rt_kb(&g_acct_stages, 0), (long long)rt_kb(&g_acct_stages, 1),
        (long long)(atomic_load(&g_mega_freed_stage_bytes) >> 10),
        (long long)rt_cnt(&g_acct_pipes), (long long)rt_kb(&g_acct_pipes, 0),
        (long long)rt_cnt(&g_acct_cmdst), (long long)rt_kb(&g_acct_cmdst, 0), (long long)rt_kb(&g_acct_bufmap, 0),
        (long long)atomic_load(&g_mega_requested), (long long)atomic_load(&g_mega_compiling),
        (long long)atomic_load(&g_mega_live), (long long)atomic_load(&g_mega_failed), (long long)atomic_load(&g_mega_capped),
        (long long)rt_cnt(&g_acct_mega_mod), (long long)rt_kb(&g_acct_mega_mod, 0),
        (long long)(atomic_load(&g_mega_spv_total) >> 10), (long long)rt_cnt(&g_acct_dump), (long long)rt_kb(&g_acct_dump, 0),
        (long long)rt_live_bytes());

    /* vk_mem: heaps and types with a non-zero peak, top 5 live sizes by bytes */
    char hb[640], tb[1024], sb[400];
    size_t hn = 0, tn = 0, sn = 0;
    hb[0] = tb[0] = sb[0] = '\0';
    for (int h = 0; h <= RT_VKM_HEAPS && hn < sizeof hb; h++) {
        if (!rt_kb(&g_vkm_heap[h], 1) && !rt_cnt(&g_vkm_heap[h])) continue;
        int w = snprintf(hb + hn, sizeof hb - hn, " %s%d=%lld/%lld(peak %lld)", h == RT_VKM_HEAPS ? "import" : "heap",
                         h == RT_VKM_HEAPS ? 0 : h, (long long)rt_cnt(&g_vkm_heap[h]), (long long)rt_kb(&g_vkm_heap[h], 0),
                         (long long)rt_kb(&g_vkm_heap[h], 1));
        if (w < 0) break;
        hn += (size_t)w;
    }
    for (int t = 0; t < RT_VKM_TYPES && tn < sizeof tb; t++) {
        if (!rt_kb(&g_vkm_type[t], 1) && !rt_cnt(&g_vkm_type[t])) continue;
        int w = snprintf(tb + tn, sizeof tb - tn, " type%d=%lld/%lld(peak %lld)", t, (long long)rt_cnt(&g_vkm_type[t]),
                         (long long)rt_kb(&g_vkm_type[t], 0), (long long)rt_kb(&g_vkm_type[t], 1));
        if (w < 0) break;
        tn += (size_t)w;
    }
    {
        struct RtVkmSize top[5] = {{0, 0}};
        int64_t on, ob;
        pthread_mutex_lock(&g_vkm_lock);
        for (int i = 0; i < RT_VKM_SIZES; i++) {
            if (g_vkm_sizes[i].live <= 0) continue;
            uint64_t b = g_vkm_sizes[i].size * (uint64_t)g_vkm_sizes[i].live;
            for (int k = 0; k < 5; k++) {
                if (b > top[k].size * (uint64_t)top[k].live) {
                    memmove(&top[k + 1], &top[k], (size_t)(4 - k) * sizeof(top[0]));
                    top[k] = g_vkm_sizes[i];
                    break;
                }
            }
        }
        on = g_vkm_other_n; ob = g_vkm_other_bytes;
        pthread_mutex_unlock(&g_vkm_lock);
        for (int k = 0; k < 5 && top[k].live > 0 && sn < sizeof sb; k++) {
            int w = snprintf(sb + sn, sizeof sb - sn, " %llux%lld", (unsigned long long)top[k].size,
                             (long long)top[k].live);
            if (w < 0) break;
            sn += (size_t)w;
        }
        if (on > 0 && sn < sizeof sb) {
            int w = snprintf(sb + sn, sizeof sb - sn, " other=%lld/%lldkb", (long long)on, (long long)(ob >> 10));
            if (w > 0) sn += (size_t)w;
        }
        if (sn >= sizeof sb) sb[sizeof sb - 1] = '\0';
    }
    if (hn >= sizeof hb) hb[sizeof hb - 1] = '\0';
    if (tn >= sizeof tb) tb[sizeof tb - 1] = '\0';
    LOG("vk_mem: t=%lld ms%s app n=%lld kb=%lld peak_kb=%lld bytes=%lld allocs_total=%llu | layer_rt_kb=%lld |%s |%s | "
        "top5 bytesxN:%s",
        (long long)now, tag, (long long)rt_cnt(&g_vkm_all), (long long)rt_kb(&g_vkm_all, 0), (long long)rt_kb(&g_vkm_all, 1),
        (long long)atomic_load_explicit(&g_vkm_all.bytes, memory_order_relaxed),
        (unsigned long long)atomic_load(&g_vkm_allocs_total),
        (long long)(rt_kb(&g_acct_pool, 0) + rt_kb(&g_acct_ring, 0) + rt_kb(&g_acct_zero, 0)), hb, tb, sb);

    /* rt_rays: window since the last report */
    static int64_t w_ms;
    static uint64_t w_tr, w_rays, w_ind, w_cls_tr[RT_RAY_CLASSES], w_cls_rays[RT_RAY_CLASSES], w_builds, w_inplace;
    const uint64_t tr = atomic_load(&g_ray_traces), rays = atomic_load(&g_ray_rays), ind = atomic_load(&g_ray_ind);
    const uint64_t builds = atomic_load(&g_rt_scr_builds), inplace = atomic_load(&g_rt_upd_inplace);
    if (!w_ms) w_ms = atomic_load(&g_rt_report_t0);
    const int64_t win = w_ms ? now - w_ms : 0;
    char cb[400];
    size_t cn = 0;
    cb[0] = '\0';
    for (int c = 0; c < RT_RAY_CLASSES && cn < sizeof cb; c++) {
        uint64_t ct = atomic_load(&g_ray_cls_tr[c]), cr = atomic_load(&g_ray_cls_rays[c]);
        int w = snprintf(cb + cn, sizeof cb - cn, " %s=%llu/%llu", g_ray_cls_name[c], (unsigned long long)(ct - w_cls_tr[c]),
                         (unsigned long long)(cr - w_cls_rays[c]));
        if (w < 0) break;
        cn += (size_t)w;
        w_cls_tr[c] = ct; w_cls_rays[c] = cr;
    }
    if (cn >= sizeof cb) cb[sizeof cb - 1] = '\0';
    const double sec = win > 0 ? (double)win / 1000.0 : 0.0;
    LOG("rt_rays: t=%lld ms%s win_ms=%lld traces=%llu rays=%llu rays_per_s=%.0f traces_per_s=%.1f indirect=%llu | "
        "stages traces/rays:%s | as builds=%llu inplace_refits=%llu per_s=%.0f/%.0f | skipped total: forced=%llu "
        "compiling=%llu nopipe=%llu zero_dim=%llu nostate=%llu | total traces=%llu rays=%llu indirect=%llu "
        "(rays = launched megakernel invocations w*h*d at record time, not ray-query traversals; frames: none seen, "
        "the layer is below the WSI layer)",
        (long long)now, tag, (long long)win, (unsigned long long)(tr - w_tr), (unsigned long long)(rays - w_rays),
        sec > 0 ? (double)(rays - w_rays) / sec : 0.0, sec > 0 ? (double)(tr - w_tr) / sec : 0.0,
        (unsigned long long)(ind - w_ind), cb, (unsigned long long)(builds - w_builds),
        (unsigned long long)(inplace - w_inplace), sec > 0 ? (double)(builds - w_builds) / sec : 0.0,
        sec > 0 ? (double)(inplace - w_inplace) / sec : 0.0,
        (unsigned long long)atomic_load(&g_rt_skip_forced), (unsigned long long)atomic_load(&g_rt_skip_pending),
        (unsigned long long)atomic_load(&g_rt_skip_nopipe), (unsigned long long)atomic_load(&g_ray_skip_zero),
        (unsigned long long)atomic_load(&g_ray_skip_nostate), (unsigned long long)tr, (unsigned long long)rays,
        (unsigned long long)ind);
    if (rt_env()->local1d)
        LOG("rt_local1d: t=%lld ms%s 1-D direct traces total=%llu (record time)", (long long)now, tag,
            (unsigned long long)atomic_load(&g_local1d_traces));
    if (rt_env()->split)
        LOG("rt_split: t=%lld ms%s split traces total=%llu gpu_select=%llu fallback=%llu (record time)", (long long)now,
            tag, (unsigned long long)atomic_load(&g_split_traces), (unsigned long long)atomic_load(&g_split_select),
            (unsigned long long)atomic_load(&g_split_fallback));
    w_ms = now; w_tr = tr; w_rays = rays; w_ind = ind; w_builds = builds; w_inplace = inplace;
    pthread_mutex_unlock(&g_rt_report_lock);
}

/* round 4: vkDestroyDevice (MALI_COMPAT_RT_PIPELINE=1): the layer's RT allocations of this device leave rt_mem (the
 * caller has just freed them), then the final report. */
static void rt_device_gone(struct DeviceData *d) {
    if (d->rt_scratch_size) { rt_acct_sub(&g_acct_pool, (int64_t)d->rt_scratch_size); d->rt_scratch_size = 0; }
    if (d->rt_ring_size)    { rt_acct_sub(&g_acct_ring, (int64_t)d->rt_ring_size);    d->rt_ring_size = 0; }
    if (d->rt_zero_size)    { rt_acct_sub(&g_acct_zero, (int64_t)d->rt_zero_size);    d->rt_zero_size = 0; }
    rt_report(d, true);
}

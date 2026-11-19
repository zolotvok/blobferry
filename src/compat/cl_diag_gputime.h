// SPDX-License-Identifier: GPL-3.0-or-later
// cl_diag_gputime.h: MALI_COMPAT_GPU_TIME timing (diag; the app's measuring switch), qstat, WAIT_CALLERS. Included by mali_compat_layer.c.
// -----------------------------------------------------------------------------
// GPU timing, MALI_COMPAT_GPU_TIME=1. [Claude Opus 5.5] 2026-10-28
// Where the GPU time goes, without root or vendor tools: a timestamp pair around every render pass and every compute
// dispatch, read back by a collector thread and summed per (kind, pipeline, render area). Every 5 s the top entries go
// to /root/mali-gputime-<comm>.txt: share of the measured time, average and worst duration, count, draws, pipeline and
// its shader hashes (see spv_hash). A pass whose start timestamp landed but whose end has not after 2 s is written as
// STUCK: the work a GPU hang sits in; when the query reads start failing (device lost), the passes in flight are
// written as INFLIGHT. Both timestamps are BOTTOM_OF_PIPE, so a duration runs from the end of the previous work to
// the end of this one: idle gaps count to the next pass. For a pass, "pipe" is the pipeline of its first draw.
// Needs hostQueryReset (enabled by the layer at vkCreateDevice). Only hooked while the switch is on.
// -----------------------------------------------------------------------------
#define GT_SLOTS 16384
enum { GT_PASS = 1, GT_DISPATCH = 2 };
// A pass's attachments (request K1 of the Witcher agent, 2026-10-30): the first GT_ATT-1 colour targets and the
// depth/stencil one, each with its format, load/store op, whether the image reached the driver with STORAGE (= no AFBC
// on this driver) and the KiB the render area covers. sig hashes formats and ops (not sizes) for the grouping.
#define GT_ATT 5
struct GtAtt { uint32_t fmt[GT_ATT], kib[GT_ATT]; uint8_t ops[GT_ATT]; uint8_t n; uint32_t sig; };
struct GtRec {
    _Atomic int state; // 0 free, 1 recorded
    uint8_t kind;
    bool stuck_reported, inflight;
    uint32_t w, h, colors, draws;
    VkPipeline pipeline;
    uint64_t t_ns; // when recorded
    // details for STUCK / TRACE lines: attachments, rendering layer count, first draw, source view
    VkImageView color0, depth, src;
    uint32_t rlayers, vcount, icount;
    struct GtAtt att; // attachment formats and load/store ops (dynamic rendering), also a grouping key of the report
    struct GtCnt cnt; // eq / dw / vsw / vxi of the pass's draws
};
struct GtAgg { uint8_t kind; uint32_t w, h, colors; VkPipeline pipeline; uint64_t sum_ns, max_ns, draws; uint32_t count;
               struct GtAtt att; struct GtCnt cnt; };
#define GT_AGG 4096
struct GpuTime {
    struct DeviceData *d;
    VkQueryPool pool;
    double period_ns;
    pthread_mutex_t m;
    uint32_t next;
    struct GtRec rec[GT_SLOTS];
    struct GtAgg agg[GT_AGG];
    struct GtCnt tot; // since start, over the measured passes (gt thread only)
    uint32_t recorded, dropped;
    char path[96];
    pthread_t thread;
    _Atomic bool stop;
    PFN_vkCmdWriteTimestamp fpCmdWriteTimestamp;
    PFN_vkGetQueryPoolResults fpGetQueryPoolResults;
    PFN_vkResetQueryPool fpResetQueryPool;
    PFN_vkDestroyQueryPool fpDestroyQueryPool;
    PFN_vkCmdBeginRendering fpCmdBeginRendering;
    PFN_vkCmdEndRendering fpCmdEndRendering;
    PFN_vkCmdPipelineBarrier2 fpCmdPipelineBarrier2; // SUSRES_BARRIER
    PFN_vkCmdBeginRenderPass fpCmdBeginRenderPass;
    PFN_vkCmdEndRenderPass fpCmdEndRenderPass;
    PFN_vkCmdBeginRenderPass2 fpCmdBeginRenderPass2;
    PFN_vkCmdEndRenderPass2 fpCmdEndRenderPass2;
    PFN_vkCmdDispatchIndirect fpCmdDispatchIndirect;
    PFN_vkCmdDraw fpCmdDraw;
    PFN_vkCmdDrawIndexed fpCmdDrawIndexed;
    PFN_vkCmdDrawIndirect fpCmdDrawIndirect;
    PFN_vkCmdDrawIndexedIndirect fpCmdDrawIndexedIndirect;
    PFN_vkCmdDrawIndirectCount fpCmdDrawIndirectCount;
    PFN_vkCmdDrawIndexedIndirectCount fpCmdDrawIndexedIndirectCount;
    PFN_vkCmdPushDescriptorSetKHR fpCmdPushDescriptorSetKHR;
    PFN_vkCmdDrawMultiEXT fpCmdDrawMultiEXT;
    PFN_vkCmdDrawMultiIndexedEXT fpCmdDrawMultiIndexedEXT;
    PFN_vkCmdDrawMeshTasksEXT fpCmdDrawMeshTasksEXT;
    PFN_vkCmdDrawMeshTasksIndirectEXT fpCmdDrawMeshTasksIndirectEXT;
    PFN_vkCmdDrawMeshTasksIndirectCountEXT fpCmdDrawMeshTasksIndirectCountEXT;
    PFN_vkCmdDispatchBase fpCmdDispatchBase;
    PFN_vkCmdExecuteCommands fpCmdExecuteCommands;
    PFN_vkQueueSubmit fpQueueSubmit;
    PFN_vkQueueSubmit2 fpQueueSubmit2;
    PFN_vkWaitForFences fpWaitForFences;
    PFN_vkWaitSemaphores fpWaitSemaphores;
    PFN_vkQueueWaitIdle fpQueueWaitIdle;
    PFN_vkDeviceWaitIdle fpDeviceWaitIdle;
};

// Queue-side CPU time per thread (request K3 of the Witcher agent, 2026-10-30): calls of vkQueueSubmit(2) and of the
// waits (fences, timeline semaphores, queue/device idle) and the CPU time spent in them, per 5-s report. It tells a
// GPU that idles because the CPU submits late (little time in waits) from a CPU that waits on the GPU. Present and
// acquire never reach this layer (Arm's WSI layer above implements the swapchain); its own submits and fence waits
// do, under its thread names. Slot QSTAT_MAX-1 takes every thread beyond the first ones.
enum { QS_SUBMIT, QS_WAIT, QS_IDLE, QS_N };
#define QSTAT_MAX 32
struct QStat { char comm[16]; _Atomic uint64_t n[QS_N], ns[QS_N], cbs; };
static struct QStat g_qstat[QSTAT_MAX];
static _Atomic uint32_t g_qstat_n;
static __thread struct QStat *tl_qstat;
static struct QStat *qstat_self(void) {
    if (tl_qstat) return tl_qstat;
    uint32_t i = atomic_fetch_add(&g_qstat_n, 1);
    struct QStat *q = &g_qstat[i < QSTAT_MAX - 1 ? i : QSTAT_MAX - 1];
    if (i < QSTAT_MAX - 1) {
        char comm[16] = "?";
        FILE *f = fopen("/proc/thread-self/comm", "r");
        if (f) { if (fgets(comm, sizeof comm, f)) comm[strcspn(comm, "\n")] = 0; fclose(f); }
        memcpy(q->comm, comm, sizeof comm);
    } else if (i == QSTAT_MAX - 1) {
        snprintf(q->comm, sizeof q->comm, "others");
    }
    return tl_qstat = q;
}
static void qstat_add(int kind, uint64_t t0, uint32_t cbs) {
    struct QStat *q = qstat_self();
    atomic_fetch_add_explicit(&q->n[kind], 1, memory_order_relaxed);
    atomic_fetch_add_explicit(&q->ns[kind], now_ns() - t0, memory_order_relaxed);
    if (cbs) atomic_fetch_add_explicit(&q->cbs, cbs, memory_order_relaxed);
}

// Wait callers (K-WC1, perf-review request (b), [keeper, Claude Opus 5.5] 2026-11-01), MALI_COMPAT_WAIT_CALLERS=1 with
// MALI_COMPAT_GPU_TIME=1: every vkWaitForFences / vkWaitSemaphores is counted per (waiting thread, caller address,
// kind of the waited-for fence, thread that submitted it). The caller is __builtin_return_address(0) as lib+offset
// (dladdr at report time): winevulkan's thunk for app waits, Arm's WSI layer for its own. A fence is classified at
// vkQueueSubmit(2): "cmd" = submitted with command buffers, "nocb-wait" = no command buffers but wait semaphores (the
// WSI layer's present payload and the present fence that DXVK's presenter waits on: VkSwapchainPresentFenceInfo),
// "nocb" = neither, "untracked" = never seen in a submit; vkWaitSemaphores = "timeline". "age" = wait start minus the
// fence's submit time: how many frames back the waiter looks. Durations in 8 bins. Printed with each 5-s report.
enum { WK_UNTRACKED, WK_CMD, WK_NOCB_WAIT, WK_NOCB, WK_TIMELINE, WK_N };
static const char *const wk_name[WK_N] = {"untracked", "cmd", "nocb-wait", "nocb", "timeline"};
#define WC_FENCES 2048 // open addressing, power of 2; a full table drops new tags (they wait as "untracked")
#define WC_MAX 48
struct WcFence { uint64_t fence, submit_ns; struct QStat *by; uint8_t kind; };
struct WcStat { struct QStat *who, *by; uintptr_t ra; uint8_t kind; uint64_t n, ns, max_ns, age_ns, age_n, hist[8]; };
static struct WcFence g_wc_fence[WC_FENCES];
static struct WcStat g_wc[WC_MAX];
static uint32_t g_wc_n, g_wc_dropped;
static pthread_mutex_t g_wc_m = PTHREAD_MUTEX_INITIALIZER;
static uint32_t wc_slot(uint64_t fence) { return (uint32_t)((fence * 0x9E3779B97F4A7C15ull) >> 40) & (WC_FENCES - 1); }
static void wc_tag(uint64_t fence, uint8_t kind) {
    if (!fence) return;
    struct QStat *by = qstat_self();
    uint64_t t = now_ns();
    pthread_mutex_lock(&g_wc_m);
    for (uint32_t i = wc_slot(fence), k = 0; k < 64; k++, i = (i + 1) & (WC_FENCES - 1)) {
        struct WcFence *e = &g_wc_fence[i];
        // a slot is reused once its fence is 10 s old: fences of a destroyed swapchain/device age out
        if (e->fence == fence || !e->fence || t - e->submit_ns > 10000000000ull) {
            e->fence = fence; e->submit_ns = t; e->by = by; e->kind = kind;
            break;
        }
    }
    pthread_mutex_unlock(&g_wc_m);
}
static uint8_t wc_submit_kind(uint32_t cbs, uint32_t waits) { return cbs ? WK_CMD : waits ? WK_NOCB_WAIT : WK_NOCB; }
static int wc_bin(uint64_t ns) {
    static const uint64_t edge[7] = {100000, 500000, 1000000, 2000000, 4000000, 8000000, 16000000};
    int b = 0;
    while (b < 7 && ns >= edge[b]) b++;
    return b;
}
static void wc_add(uintptr_t ra, uint64_t fence, uint8_t kind, uint64_t t0) {
    uint64_t t1 = now_ns(), dur = t1 - t0, age = 0;
    struct QStat *who = qstat_self(), *by = NULL;
    pthread_mutex_lock(&g_wc_m);
    if (fence) {
        for (uint32_t i = wc_slot(fence), k = 0; k < 64; k++, i = (i + 1) & (WC_FENCES - 1)) {
            struct WcFence *e = &g_wc_fence[i];
            if (!e->fence) break;
            if (e->fence == fence) { kind = e->kind; by = e->by; age = t0 > e->submit_ns ? t0 - e->submit_ns : 0; break; }
        }
    }
    struct WcStat *w = NULL;
    for (uint32_t i = 0; i < g_wc_n; i++)
        if (g_wc[i].who == who && g_wc[i].ra == ra && g_wc[i].kind == kind && g_wc[i].by == by) { w = &g_wc[i]; break; }
    if (!w && g_wc_n < WC_MAX) { w = &g_wc[g_wc_n++]; memset(w, 0, sizeof *w); w->who = who; w->ra = ra; w->kind = kind; w->by = by; }
    if (w) {
        w->n++; w->ns += dur;
        if (dur > w->max_ns) w->max_ns = dur;
        if (by) { w->age_ns += age; w->age_n++; }
        w->hist[wc_bin(dur)]++;
    } else {
        g_wc_dropped++;
    }
    pthread_mutex_unlock(&g_wc_m);
}
static void wc_caller(uintptr_t ra, char *out, size_t len) {
    Dl_info di;
    if (ra && dladdr((void *)ra, &di) && di.dli_fname) {
        const char *b = strrchr(di.dli_fname, '/');
        b = b ? b + 1 : di.dli_fname;
        if (di.dli_sname && di.dli_saddr)
            snprintf(out, len, "%s+0x%lx (%s+0x%lx)", b, (unsigned long)(ra - (uintptr_t)di.dli_fbase), di.dli_sname,
                     (unsigned long)(ra - (uintptr_t)di.dli_saddr));
        else
            snprintf(out, len, "%s+0x%lx", b, (unsigned long)(ra - (uintptr_t)di.dli_fbase));
    } else {
        snprintf(out, len, "0x%lx", (unsigned long)ra);
    }
}
// One line per (waiter, caller, kind, submitter) since the last report, then reset. Called from gt_report.
static void wc_report(FILE *f, double interval_s) {
    pthread_mutex_lock(&g_wc_m);
    for (uint32_t i = 0; i < g_wc_n; i++) {
        struct WcStat *w = &g_wc[i];
        if (!w->n) continue;
        char caller[192];
        wc_caller(w->ra, caller, sizeof caller);
        fprintf(f, "wait caller: %s <- %s kind %s", w->who && w->who->comm[0] ? w->who->comm : "?", caller, wk_name[w->kind]);
        if (w->by) fprintf(f, " (submitted by %s, age %.2f ms)", w->by->comm[0] ? w->by->comm : "?", w->age_ns / 1e6 / w->age_n);
        fprintf(f, ": n %llu, %.1f ms (%.1f %% of wall), mean %.3f, max %.2f ms | <0.1 %llu <0.5 %llu <1 %llu <2 %llu <4 %llu"
                   " <8 %llu <16 %llu >=16 %llu\n", (unsigned long long)w->n, w->ns / 1e6,
                interval_s > 0 ? 100.0 * w->ns / 1e9 / interval_s : 0, w->ns / 1e6 / w->n, w->max_ns / 1e6,
                (unsigned long long)w->hist[0], (unsigned long long)w->hist[1], (unsigned long long)w->hist[2],
                (unsigned long long)w->hist[3], (unsigned long long)w->hist[4], (unsigned long long)w->hist[5],
                (unsigned long long)w->hist[6], (unsigned long long)w->hist[7]);
    }
    if (g_wc_dropped) fprintf(f, "wait caller: %u waits not counted (more than %d keys)\n", g_wc_dropped, WC_MAX);
    // keep the keys (stable order between reports), zero the counters
    for (uint32_t i = 0; i < g_wc_n; i++) {
        struct WcStat *w = &g_wc[i];
        w->n = w->ns = w->max_ns = w->age_ns = w->age_n = 0;
        memset(w->hist, 0, sizeof w->hist);
    }
    g_wc_dropped = 0;
    pthread_mutex_unlock(&g_wc_m);
}

// Calls per hooked entry point, printed with each report: shows which draw path the app really uses.
enum { GTC_DRAW, GTC_DRAW_INDEXED, GTC_DRAW_INDIRECT, GTC_DRAW_INDEXED_INDIRECT, GTC_DRAW_INDIRECT_COUNT,
       GTC_DRAW_INDEXED_INDIRECT_COUNT, GTC_DRAW_MULTI, GTC_DRAW_MULTI_INDEXED, GTC_MESH, GTC_MESH_INDIRECT,
       GTC_MESH_INDIRECT_COUNT, GTC_DISPATCH, GTC_DISPATCH_INDIRECT, GTC_DISPATCH_BASE, GTC_RENDERING, GTC_RENDERPASS,
       GTC_EXECUTE, GTC_N };
static const char *const g_gtc_name[GTC_N] = {"Draw", "DrawIndexed", "DrawIndirect", "DrawIndexedIndirect", "DrawIndirectCount",
    "DrawIndexedIndirectCount", "DrawMultiEXT", "DrawMultiIndexedEXT", "DrawMeshTasksEXT", "DrawMeshTasksIndirectEXT",
    "DrawMeshTasksIndirectCountEXT", "Dispatch", "DispatchIndirect", "DispatchBase", "BeginRendering", "BeginRenderPass",
    "ExecuteCommands"};
static _Atomic uint64_t g_gtc[GTC_N];
#define GTC(i) atomic_fetch_add_explicit(&g_gtc[i], 1, memory_order_relaxed)

static uint64_t gt_now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}

static void gt_free_slot(struct GpuTime *g, uint32_t i) {
    g->fpResetQueryPool(g->d->device, g->pool, 2 * i, 2);
    atomic_store(&g->rec[i].state, 0);
}

// Takes a slot and writes the start timestamp; -1 when the ring is full of pending records (counted as dropped).
static int32_t gt_begin(struct CmdBufData *c, VkCommandBuffer cmd, uint8_t kind, uint32_t w, uint32_t h, uint32_t colors,
                        VkPipeline pipeline) {
    struct GpuTime *g = c ? c->dev->gt : NULL;
    if (!g || !g->pool) return -1;
    pthread_mutex_lock(&g->m);
    uint32_t i = g->next;
    g->next = (i + 1) % GT_SLOTS;
    struct GtRec *r = &g->rec[i];
    if (atomic_load(&r->state)) { g->dropped++; pthread_mutex_unlock(&g->m); return -1; }
    r->kind = kind; r->w = w; r->h = h; r->colors = colors; r->draws = 0; r->pipeline = pipeline;
    r->stuck_reported = r->inflight = false;
    r->color0 = r->depth = r->src = VK_NULL_HANDLE;
    r->rlayers = r->vcount = r->icount = 0;
    memset(&r->att, 0, sizeof r->att);
    memset(&r->cnt, 0, sizeof r->cnt);
    r->t_ns = gt_now();
    atomic_store(&r->state, 1);
    g->recorded++;
    pthread_mutex_unlock(&g->m);
    g->fpCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, g->pool, 2 * i);
    return (int32_t)i;
}

static void gt_end(struct CmdBufData *c, VkCommandBuffer cmd, int32_t i) {
    if (i >= 0) c->dev->gt->fpCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, c->dev->gt->pool, 2 * i + 1);
}

static void gt_pipe_desc(VkPipeline p, char *out, size_t len) {
    struct HashNode h;
    int n = snprintf(out, len, "pipe %p", (void *)p);
    if (p && hash_get(g_pipeline_hash, (uint64_t)p, &h))
        for (int s = 0; s < 3 && h.stages[s]; s++) {
            const char *nm = h.stages[s] == VK_SHADER_STAGE_VERTEX_BIT ? "vs" : h.stages[s] == VK_SHADER_STAGE_FRAGMENT_BIT ? "fs"
                           : h.stages[s] == VK_SHADER_STAGE_COMPUTE_BIT ? "cs" : h.stages[s] == VK_SHADER_STAGE_GEOMETRY_BIT ? "gs"
                           : h.stages[s] == VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT ? "tc"
                           : h.stages[s] == VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT ? "te" : "st";
            n += snprintf(out + n, len - n, " %s:%08x", nm, h.hash[s]);
        }
}

static void gt_view_desc(const char *name, VkImageView v, struct ImgNode *out, char *buf, size_t len) {
    struct ImgNode n = {0};
    if (!v) { snprintf(buf, len, " %s -", name); memset(out, 0, sizeof *out); return; }
    if (!img_get(g_view_info, (uint64_t)v, &n)) { snprintf(buf, len, " %s view %p (unknown)", name, (void *)v); memset(out, 0, sizeof *out); return; }
    struct ImgNode im = {0};
    img_get(g_img_info, (uint64_t)n.image, &im);
    snprintf(buf, len, " %s img %p fmt %d (view %d) %ux%u x%u mips, view mip %u+%u layer %u+%u of %u, aspect 0x%x, samples %u",
             name, (void *)n.image, im.format, n.format, im.w, im.h, im.mips, n.mip, n.mips, n.layer, n.layers, im.layers,
             n.usage, im.samples);
    *out = n;
}

// Short names of the usual render-target formats for the report; the others print as their number.
static const char *gt_fmt_name(uint32_t f, char *buf, size_t len) {
    switch (f) {
    case VK_FORMAT_R8_UNORM: return "r8";
    case VK_FORMAT_R8G8_UNORM: return "rg8";
    case VK_FORMAT_R8G8B8A8_UNORM: return "rgba8";
    case VK_FORMAT_R8G8B8A8_SRGB: return "rgba8srgb";
    case VK_FORMAT_B8G8R8A8_UNORM: return "bgra8";
    case VK_FORMAT_B8G8R8A8_SRGB: return "bgra8srgb";
    case VK_FORMAT_A2B10G10R10_UNORM_PACK32: return "rgb10a2";
    case VK_FORMAT_A2R10G10B10_UNORM_PACK32: return "bgr10a2";
    case VK_FORMAT_B10G11R11_UFLOAT_PACK32: return "r11g11b10f";
    case VK_FORMAT_R16_SFLOAT: return "r16f";
    case VK_FORMAT_R16G16_SFLOAT: return "rg16f";
    case VK_FORMAT_R16G16B16A16_SFLOAT: return "rgba16f";
    case VK_FORMAT_R16G16B16A16_UNORM: return "rgba16";
    case VK_FORMAT_R32_SFLOAT: return "r32f";
    case VK_FORMAT_R32_UINT: return "r32ui";
    case VK_FORMAT_R32G32_SFLOAT: return "rg32f";
    case VK_FORMAT_R32G32B32A32_SFLOAT: return "rgba32f";
    case VK_FORMAT_D16_UNORM: return "d16";
    case VK_FORMAT_X8_D24_UNORM_PACK32: return "x8d24";
    case VK_FORMAT_D32_SFLOAT: return "d32";
    case VK_FORMAT_D24_UNORM_S8_UINT: return "d24s8";
    case VK_FORMAT_D32_SFLOAT_S8_UINT: return "d32s8";
    default: snprintf(buf, len, "fmt%u", f); return buf;
    }
}

// Bytes per pixel of a render-target format (4 for the ones not listed).
static uint32_t gt_fmt_bytes(uint32_t f) {
    switch (f) {
    case VK_FORMAT_R8_UNORM: case VK_FORMAT_R8_UINT: case VK_FORMAT_R8_SNORM: return 1;
    case VK_FORMAT_R8G8_UNORM: case VK_FORMAT_R16_SFLOAT: case VK_FORMAT_R16_UNORM: case VK_FORMAT_R16_UINT:
    case VK_FORMAT_D16_UNORM: return 2;
    case VK_FORMAT_D32_SFLOAT_S8_UINT: return 5;
    case VK_FORMAT_R16G16B16A16_SFLOAT: case VK_FORMAT_R16G16B16A16_UNORM: case VK_FORMAT_R32G32_SFLOAT:
    case VK_FORMAT_R32G32_UINT: return 8;
    case VK_FORMAT_R32G32B32A32_SFLOAT: case VK_FORMAT_R32G32B32A32_UINT: return 16;
    default: return 4;
    }
}

// ops: bits 0-1 load (LOAD, CLEAR, DONT_CARE, NONE), 2-3 store (STORE, DONT_CARE, NONE), 16 STORAGE, 32 slot used,
// 64 depth/stencil.
static void gt_att_add(struct GtAtt *a, const VkRenderingAttachmentInfo *ai, uint32_t w, uint32_t h, uint32_t layers, bool ds) {
    if (a->n >= GT_ATT) return;
    uint32_t k = a->n++;
    a->fmt[k] = a->kib[k] = 0;
    a->ops[k] = ds ? 64 : 0;
    if (!ai || !ai->imageView) return; // unused colour slot
    struct ImgNode v = {0}, im = {0};
    img_get(g_view_info, (uint64_t)ai->imageView, &v);
    if (v.image) img_get(g_img_info, (uint64_t)v.image, &im);
    a->fmt[k] = v.format ? (uint32_t)v.format : (uint32_t)im.format;
    uint8_t lc = ai->loadOp == VK_ATTACHMENT_LOAD_OP_LOAD ? 0 : ai->loadOp == VK_ATTACHMENT_LOAD_OP_CLEAR ? 1
               : ai->loadOp == VK_ATTACHMENT_LOAD_OP_DONT_CARE ? 2 : 3;
    uint8_t sc = ai->storeOp == VK_ATTACHMENT_STORE_OP_STORE ? 0 : ai->storeOp == VK_ATTACHMENT_STORE_OP_DONT_CARE ? 1 : 2;
    a->ops[k] |= lc | sc << 2 | ((im.usage & VK_IMAGE_USAGE_STORAGE_BIT) ? 16 : 0) | 32;
    uint64_t bytes = (uint64_t)w * h * (layers ? layers : 1) * gt_fmt_bytes(a->fmt[k]) * (im.samples ? im.samples : 1);
    a->kib[k] = (uint32_t)(bytes >> 10);
}

static void gt_att_sign(struct GtAtt *a) {
    uint32_t s = 2166136261u;
    for (uint32_t k = 0; k < a->n; k++) {
        s = (s ^ a->fmt[k]) * 16777619u;
        s = (s ^ a->ops[k]) * 16777619u;
    }
    a->sig = a->n ? s : 0;
}

// " | c0 rgba16f L/S st 4.4M c1 ... d d24s8 C/D 4.4M": per attachment format, load/store op, "st" = created with
// STORAGE (uncompressed on this driver), the MiB the render area covers; "-" for an unused colour slot.
static void gt_att_desc(const struct GtAtt *a, char *out, size_t len) {
    static const char lc[] = "LCDN", sc[] = "SDN?";
    size_t n = 0;
    out[0] = 0;
    if (!a->n || len < 8) return;
    n += snprintf(out, len, "  |");
    for (uint32_t k = 0, ci = 0; k < a->n && n < len; k++) {
        char fb[16], name[8];
        uint8_t o = a->ops[k];
        if (o & 64) snprintf(name, sizeof name, "d");
        else snprintf(name, sizeof name, "c%u", ci++);
        if (!(o & 32)) n += snprintf(out + n, len - n, " %s -", name);
        else n += snprintf(out + n, len - n, " %s %s %c/%c%s %.1fM", name, gt_fmt_name(a->fmt[k], fb, sizeof fb), lc[o & 3],
                           sc[(o >> 2) & 3], (o & 16) ? " st" : "", a->kib[k] / 1024.0);
    }
}

static void gt_line_rec(FILE *f, const char *tag, const struct GtRec *r, double age_s, double dur_ms) {
    char desc[160], c0[200], dp[200], src[200];
    struct ImgNode nc, nd, ns;
    gt_pipe_desc(r->pipeline, desc, sizeof desc);
    gt_view_desc("color0", r->color0, &nc, c0, sizeof c0);
    gt_view_desc("depth", r->depth, &nd, dp, sizeof dp);
    gt_view_desc("src", r->src, &ns, src, sizeof src);
    bool same = ns.image && (ns.image == nc.image || ns.image == nd.image);
    char att[256];
    gt_att_desc(&r->att, att, sizeof att);
    fprintf(f, "%s %s %ux%u c%u draws %u (first: %u verts x %u inst) rlayers %u, %s %.1f %s, %s%s%s%s%s%s\n", tag,
            r->kind == GT_PASS ? "pass" : "dispatch", r->w, r->h, r->colors, r->draws, r->vcount, r->icount, r->rlayers,
            dur_ms >= 0 ? "took" : "recorded", dur_ms >= 0 ? dur_ms : age_s, dur_ms >= 0 ? "ms" : "s ago", desc, c0, dp, src,
            same ? "  SAME-IMAGE" : "", att);
}

// MALI_COMPAT_GT_TRACE: does this record's pipeline have the traced fragment/compute shader?
static bool gt_traced(const struct GtRec *r) {
    struct HashNode h;
    if (g_gt_trace == UINT32_MAX) return true;
    if (!g_gt_trace || !r->pipeline || !hash_get(g_pipeline_hash, (uint64_t)r->pipeline, &h)) return false;
    for (int i = 0; i < 3; i++) if (h.hash[i] == g_gt_trace) return true;
    return false;
}

static int gt_cmp(const void *a, const void *b) {
    const struct GtAgg *x = a, *y = b;
    return x->sum_ns < y->sum_ns ? 1 : x->sum_ns > y->sum_ns ? -1 : 0;
}

static void gt_report(struct GpuTime *g, double interval_s) {
    struct GtAgg *v = malloc(sizeof g->agg);
    if (!v) return;
    uint32_t n = 0;
    uint64_t total = 0;
    for (uint32_t i = 0; i < GT_AGG; i++)
        if (g->agg[i].count) { v[n++] = g->agg[i]; total += g->agg[i].sum_ns; }
    qsort(v, n, sizeof *v, gt_cmp);
    FILE *f = fopen(g->path, "a");
    if (f) {
        time_t t = time(NULL);
        struct tm tm;
        localtime_r(&t, &tm);
        pthread_mutex_lock(&g->m);
        uint32_t rec = g->recorded, drop = g->dropped;
        g->recorded = g->dropped = 0;
        pthread_mutex_unlock(&g->m);
        fprintf(f, "=== %02d:%02d:%02d  %.1f s: measured %.0f ms GPU time (%.0f %% of the interval), %u records, %u dropped\n",
                tm.tm_hour, tm.tm_min, tm.tm_sec, interval_s, total / 1e6, 100.0 * total / (interval_s * 1e9), rec, drop);
        fprintf(f, "calls since start:");
        for (int k = 0; k < GTC_N; k++) {
            uint64_t c = atomic_load(&g_gtc[k]);
            if (c) fprintf(f, " %s %llu", g_gtc_name[k], (unsigned long long)c);
        }
        fprintf(f, " | suspending passes %llu, resuming %llu, barriers before resumed parts %llu",
                (unsigned long long)atomic_load(&g_susp_passes), (unsigned long long)atomic_load(&g_resu_passes),
                (unsigned long long)atomic_load(&g_susres_barriers));
        fprintf(f, " | firstInstance!=0 draws %llu, divisor!=1 pipelines %llu, vertex inputs %llu",
                (unsigned long long)atomic_load(&g_div_draws), (unsigned long long)atomic_load(&g_div_pipes),
                (unsigned long long)atomic_load(&g_div_vi));
        if (g_async_ms)
            fprintf(f, " | async pipelines %llu, stand-ins %llu, draws dropped %llu, failed %llu",
                    (unsigned long long)atomic_load(&g_async_jobs), (unsigned long long)atomic_load(&g_async_pending),
                    (unsigned long long)atomic_load(&g_async_skipped), (unsigned long long)atomic_load(&g_async_failed));
        if (g_noatt_w) fprintf(f, " | attachment-less passes clamped %llu", (unsigned long long)atomic_load(&g_noatt_clamped));
        if (g_xfb_vs) fprintf(f, " | stream-output pipelines without GS %llu", (unsigned long long)atomic_load(&g_xfb_vs_pipes));
        if (g_skip_hash) fprintf(f, " | skipped draws (%08x) %llu", g_skip_hash, (unsigned long long)atomic_load(&g_skipped_draws));
        fprintf(f, " | draws eq %llu, dw %llu, vsw %llu, vxi %.1fM", (unsigned long long)g->tot.eq,
                (unsigned long long)g->tot.dw, (unsigned long long)g->tot.vsw, g->tot.vxi / 1e6);
        fprintf(f, " | pipelines depth EQUAL/no write %llu", (unsigned long long)atomic_load(&g_deq_pipes));
        if (g_depth_equal_on)
            fprintf(f, " (replaced by op %d: %llu, dynamic %llu)", g_depth_equal_op, (unsigned long long)atomic_load(&g_deq_replaced),
                    (unsigned long long)atomic_load(&g_deq_dynamic));
        fprintf(f, ", VS-stage stores %llu (%u shaders)", (unsigned long long)atomic_load(&g_vsw_pipes), atomic_load(&g_vsw_shaders));
        if (g_fake_occlusion)
            fprintf(f, " | faked occlusion copies %llu, reads %llu", (unsigned long long)atomic_load(&g_occl_copies),
                    (unsigned long long)atomic_load(&g_occl_reads));
        fprintf(f, "\n");
        pthread_mutex_lock(&g_pcstat_m);
        if (g_pcstat[0].count) {
            fprintf(f, "graphics pipeline compiles:");
            for (int k = 0; k < 16 && g_pcstat[k].count; k++)
                fprintf(f, " %s %u in %.0f ms (max %.0f)", g_pcstat[k].comm, g_pcstat[k].count, g_pcstat[k].sum_ns / 1e6,
                        g_pcstat[k].max_ns / 1e6);
            fprintf(f, "\n");
        }
        memset(g_pcstat, 0, sizeof g_pcstat);
        pthread_mutex_unlock(&g_pcstat_m);
        // queue-side CPU time per thread since the last report (QStat): calls and ms
        uint32_t nq = atomic_load(&g_qstat_n);
        bool qany = false;
        for (uint32_t k = 0; k < nq && k < QSTAT_MAX; k++) {
            struct QStat *q = &g_qstat[k];
            uint64_t qn[QS_N], qns[QS_N], cbs = atomic_exchange(&q->cbs, 0);
            for (int j = 0; j < QS_N; j++) { qn[j] = atomic_exchange(&q->n[j], 0); qns[j] = atomic_exchange(&q->ns[j], 0); }
            if (!qn[QS_SUBMIT] && !qn[QS_WAIT] && !qn[QS_IDLE]) continue;
            if (!qany) { fprintf(f, "queue CPU time (calls, ms):"); qany = true; }
            fprintf(f, " %s", q->comm[0] ? q->comm : "?");
            if (qn[QS_SUBMIT]) fprintf(f, " submit %llu (%llu cb) %.1f", (unsigned long long)qn[QS_SUBMIT], (unsigned long long)cbs,
                                       qns[QS_SUBMIT] / 1e6);
            if (qn[QS_WAIT]) fprintf(f, " wait %llu %.1f", (unsigned long long)qn[QS_WAIT], qns[QS_WAIT] / 1e6);
            if (qn[QS_IDLE]) fprintf(f, " idle %llu %.1f", (unsigned long long)qn[QS_IDLE], qns[QS_IDLE] / 1e6);
            fprintf(f, ";");
        }
        if (qany) fprintf(f, "\n");
        if (g_wait_callers) wc_report(f, interval_s);
        for (uint32_t i = 0; i < n && i < 40; i++) {
            char desc[160], att[256];
            gt_pipe_desc(v[i].pipeline, desc, sizeof desc);
            gt_att_desc(&v[i].att, att, sizeof att);
            char cn[96] = "";
            if (v[i].kind == GT_PASS && v[i].draws) // per pass, like draws
                snprintf(cn, sizeof cn, " eq %.1f dw %.1f vsw %.1f vxi %.3fM", (double)v[i].cnt.eq / v[i].count,
                         (double)v[i].cnt.dw / v[i].count, (double)v[i].cnt.vsw / v[i].count, v[i].cnt.vxi / 1e6 / v[i].count);
            fprintf(f, "%5.1f%%  avg %7.3f ms  max %7.2f ms  n %5u  %-8s %4ux%-4u c%u draws %5.1f%s  %s%s\n",
                    total ? 100.0 * v[i].sum_ns / total : 0, v[i].sum_ns / 1e6 / v[i].count, v[i].max_ns / 1e6, v[i].count,
                    v[i].kind == GT_PASS ? "pass" : "dispatch", v[i].w, v[i].h, v[i].colors,
                    (double)v[i].draws / v[i].count, cn, desc, att);
        }
        fclose(f);
    }
    free(v);
    memset(g->agg, 0, sizeof g->agg);
}

// Dispatches are grouped per pipeline; passes per render area, colour target count and attachment signature (formats
// and load/store ops), with draws or without (a pass with draws runs many pipelines, so grouping it by its first one
// scattered the heavy passes; "pipe" is then the first draw's pipeline of the last pass seen in the group).
static void gt_add(struct GpuTime *g, const struct GtRec *r, uint64_t ns) {
    VkPipeline kp = r->kind == GT_PASS ? (r->draws ? (VkPipeline)(uintptr_t)1 : VK_NULL_HANDLE) : r->pipeline;
    uint64_t key = (uint64_t)(uintptr_t)kp ^ ((uint64_t)r->kind << 56) ^ ((uint64_t)r->w << 36) ^ ((uint64_t)r->h << 16) ^ r->colors ^
                   ((uint64_t)r->att.sig << 8);
    uint32_t b = handle_bucket(key, GT_AGG);
    for (uint32_t probe = 0; probe < GT_AGG; probe++, b = (b + 1) % GT_AGG) {
        struct GtAgg *a = &g->agg[b];
        bool a_draws = a->kind == GT_PASS && a->draws, match_kp = r->kind == GT_PASS ? (a_draws == (r->draws != 0)) : a->pipeline == r->pipeline;
        if (!a->count) {
            *a = (struct GtAgg){r->kind, r->w, r->h, r->colors, r->pipeline, 0, 0, 0, 0, r->att, {0, 0, 0, 0}};
        } else if (a->kind != r->kind || a->w != r->w || a->h != r->h || a->colors != r->colors || !match_kp ||
                   a->att.sig != r->att.sig) {
            continue;
        }
        if (r->kind == GT_PASS && r->pipeline) a->pipeline = r->pipeline;
        a->sum_ns += ns;
        if (ns > a->max_ns) a->max_ns = ns;
        a->draws += r->draws;
        gt_cnt_add(&a->cnt, &r->cnt);
        gt_cnt_add(&g->tot, &r->cnt);
        a->count++;
        return;
    }
}

static void *gt_thread(void *arg) {
    struct GpuTime *g = arg;
    uint64_t last = gt_now();
    bool lost = false;
    while (!atomic_load(&g->stop)) {
        usleep(250000);
        uint64_t now = gt_now();
        for (uint32_t i = 0; i < GT_SLOTS && !lost; i++) {
            struct GtRec *r = &g->rec[i];
            if (!atomic_load(&r->state)) continue;
            uint64_t v[4] = {0};
            VkResult res = g->fpGetQueryPoolResults(g->d->device, g->pool, 2 * i, 2, sizeof v, v, 2 * sizeof(uint64_t),
                                                    VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WITH_AVAILABILITY_BIT);
            double age = (now - r->t_ns) / 1e9;
            if (res < 0) {
                // The device is gone: write what was running on the GPU and stop reading.
                FILE *f = fopen(g->path, "a");
                if (f) {
                    fprintf(f, "=== query read failed (%d): passes in flight at that point:\n", res);
                    for (uint32_t k = 0; k < GT_SLOTS; k++)
                        if (atomic_load(&g->rec[k].state) && g->rec[k].inflight)
                            gt_line_rec(f, "INFLIGHT", &g->rec[k], (now - g->rec[k].t_ns) / 1e9, -1);
                    fclose(f);
                }
                lost = true;
                break;
            }
            if (v[1] && v[3]) {
                // end before start: the slot was rewritten (command buffer resubmitted after the slot was reused)
                uint64_t ns = v[2] >= v[0] ? (uint64_t)((double)(v[2] - v[0]) * g->period_ns) : 0;
                if (gt_traced(r)) {
                    FILE *f = fopen(g->path, "a");
                    if (f) {
                        // GPU start and end time in ms, so the lines can be put in GPU order and gaps seen
                        fprintf(f, "T %.3f %.3f ", v[0] * g->period_ns / 1e6, v[2] * g->period_ns / 1e6);
                        gt_line_rec(f, "TRACE", r, age, ns / 1e6);
                        fclose(f);
                    }
                }
                gt_add(g, r, ns);
                gt_free_slot(g, i);
            } else if (v[1]) {
                r->inflight = age > 0.5;
                if (age > 2.0 && !r->stuck_reported) {
                    FILE *f = fopen(g->path, "a");
                    if (f) { gt_line_rec(f, "STUCK", r, age, -1); fclose(f); }
                    r->stuck_reported = true;
                }
                if (age > 60.0) gt_free_slot(g, i);
            } else if (age > 10.0) {
                gt_free_slot(g, i); // recorded but never executed (command buffer reset or not submitted)
            }
        }
        if (now - last >= 5000000000ull && !lost) {
            gt_report(g, (now - last) / 1e9);
            last = now;
        }
    }
    return NULL;
}

static void gt_init(struct DeviceData *d, PFN_vkGetDeviceProcAddr gdpa) {
    struct GpuTime *g = calloc(1, sizeof *g);
    if (!g) return;
    g->d = d;
    pthread_mutex_init(&g->m, NULL);
    #define GT_LOAD(fn) g->fp##fn = (PFN_vk##fn)gdpa(d->device, "vk" #fn)
    #define GT_LOAD_KHR(fn) if (!(GT_LOAD(fn))) g->fp##fn = (PFN_vk##fn)gdpa(d->device, "vk" #fn "KHR")
    GT_LOAD(CmdWriteTimestamp);
    GT_LOAD(GetQueryPoolResults);
    GT_LOAD(DestroyQueryPool);
    GT_LOAD(CmdBeginRenderPass);
    GT_LOAD(CmdEndRenderPass);
    GT_LOAD(CmdDispatchIndirect);
    GT_LOAD(CmdDraw);
    GT_LOAD(CmdDrawIndexed);
    GT_LOAD(CmdDrawIndirect);
    GT_LOAD(CmdDrawIndexedIndirect);
    if (!(GT_LOAD(ResetQueryPool))) g->fpResetQueryPool = (PFN_vkResetQueryPool)gdpa(d->device, "vkResetQueryPoolEXT");
    GT_LOAD_KHR(CmdBeginRendering);
    GT_LOAD_KHR(CmdEndRendering);
    GT_LOAD_KHR(CmdPipelineBarrier2);
    GT_LOAD_KHR(CmdBeginRenderPass2);
    GT_LOAD_KHR(CmdEndRenderPass2);
    GT_LOAD_KHR(CmdDrawIndirectCount);
    GT_LOAD_KHR(CmdDrawIndexedIndirectCount);
    GT_LOAD(CmdDrawMultiEXT);
    GT_LOAD(CmdDrawMultiIndexedEXT);
    GT_LOAD(CmdDrawMeshTasksEXT);
    GT_LOAD(CmdDrawMeshTasksIndirectEXT);
    GT_LOAD(CmdDrawMeshTasksIndirectCountEXT);
    GT_LOAD_KHR(CmdDispatchBase);
    GT_LOAD(CmdExecuteCommands);
    GT_LOAD(QueueSubmit);
    GT_LOAD_KHR(QueueSubmit2);
    GT_LOAD(WaitForFences);
    GT_LOAD_KHR(WaitSemaphores);
    GT_LOAD(QueueWaitIdle);
    GT_LOAD(DeviceWaitIdle);
    g->fpCmdPushDescriptorSetKHR = (PFN_vkCmdPushDescriptorSetKHR)gdpa(d->device, "vkCmdPushDescriptorSetKHR");
    if (!g->fpCmdPushDescriptorSetKHR) g->fpCmdPushDescriptorSetKHR = (PFN_vkCmdPushDescriptorSetKHR)gdpa(d->device, "vkCmdPushDescriptorSet");
    #undef GT_LOAD
    #undef GT_LOAD_KHR
    PFN_vkCreateQueryPool create = (PFN_vkCreateQueryPool)gdpa(d->device, "vkCreateQueryPool");
    VkPhysicalDeviceProperties2 props = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
    if (d->instance_data && d->instance_data->fpGetPhysicalDeviceProperties2)
        d->instance_data->fpGetPhysicalDeviceProperties2(d->physical_device, &props);
    g->period_ns = props.properties.limits.timestampPeriod > 0 ? props.properties.limits.timestampPeriod : 1.0;
    VkQueryPoolCreateInfo qi = {VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO, NULL, 0, VK_QUERY_TYPE_TIMESTAMP, 2 * GT_SLOTS};
    // The hooks are already handed out and pass through d->gt's function pointers, so d->gt stays even without a pool.
    d->gt = g;
    if (!g_gpu_time) return; // only the function pointers (async compiles, pass clamp)
    if (!create || !g->fpCmdWriteTimestamp || !g->fpGetQueryPoolResults || !g->fpResetQueryPool ||
        create(d->device, &qi, NULL, &g->pool) != VK_SUCCESS) {
        g->pool = VK_NULL_HANDLE;
        fprintf(stderr, "mali-compat: GPU timing unavailable (query pool or hostQueryReset missing)\n");
        return;
    }
    if (g_addrmap) am_emit('N', (uint64_t)g->pool, AM_NA, ML_NONE, BUDGET_NO_HEAP, AM_NA, 0, "what=qpool-layer type=%d count=%u",
                           VK_QUERY_TYPE_TIMESTAMP, 2 * GT_SLOTS);
    g->fpResetQueryPool(d->device, g->pool, 0, 2 * GT_SLOTS);
    char comm[32] = "unknown";
    FILE *f = fopen("/proc/self/comm", "r");
    if (f) { if (fgets(comm, sizeof comm, f)) comm[strcspn(comm, "\n")] = 0; fclose(f); }
    for (char *c = comm; *c; c++) if (*c == '/' || *c == ' ' || *c == '\\') *c = '_';
    snprintf(g->path, sizeof g->path, "/root/mali-gputime-%s.txt", comm);
    if ((f = fopen(g->path, "a"))) {
        fprintf(f, "=== GPU timing on for device %p (%s), timestamp period %.3f ns, %u slots\n", (void *)d->device,
                props.properties.deviceName, g->period_ns, GT_SLOTS);
        fclose(f);
    }
    if (pthread_create(&g->thread, NULL, gt_thread, g) != 0) {
        g->fpDestroyQueryPool(d->device, g->pool, NULL);
        g->pool = VK_NULL_HANDLE;
        return;
    }
    fprintf(stderr, "mali-compat: GPU timing on, report in %s\n", g->path);
}

// MALI_COMPAT_GPU_TIME (diag-1113): true for a hooked 1.2/1.3 command (or its KHR name) whose driver pointer is NULL on
// this device, e.g. vkCmdBeginRenderingKHR on a Vulkan 1.1 device without the extension. The hook would call NULL.
static bool gt_hook_missing(const struct GpuTime *g, const char *n) {
    if (strncmp(n, "vkCmd", 5)) return false;
    n += 5;
    size_t l = strlen(n);
    if (l > 3 && !strcmp(n + l - 3, "KHR")) l -= 3;
    #define GT_MISS(fn) if (l == sizeof #fn - 1 && !strncmp(n, #fn, l)) return !g->fpCmd##fn
    GT_MISS(BeginRendering);
    GT_MISS(EndRendering);
    GT_MISS(BeginRenderPass2);
    GT_MISS(EndRenderPass2);
    GT_MISS(DrawIndirectCount);
    GT_MISS(DrawIndexedIndirectCount);
    #undef GT_MISS
    return false;
}

static void gt_destroy(struct DeviceData *d) {
    struct GpuTime *g = d->gt;
    if (!g) return;
    if (g->pool) {
        atomic_store(&g->stop, true);
        pthread_join(g->thread, NULL);
        if (g_addrmap) am_emit('N', (uint64_t)g->pool, AM_NA, ML_NONE, BUDGET_NO_HEAP, AM_NA, 0, "what=qpool-layer-destroy");
        g->fpDestroyQueryPool(d->device, g->pool, NULL);
    }
    d->gt = NULL;
    free(g);
}


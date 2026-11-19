// SPDX-License-Identifier: GPL-3.0-or-later
// cl_diag_dpool.h: diagnostic MALI_COMPAT_DPOOL_DIAG (hook table dpd_proc). Included by mali_compat_layer.c.
// ------------------------------------------------------------------------------------------------------------------
// Descriptor-pool lifetime diagnostic, MALI_COMPAT_DPOOL_DIAG [game/doom_eternal keeper, Claude Opus 5.5] 2026-11-14
// (doom-dpool-1114; design: docs/games/doom_eternal.md 11-14 20:13). DOOM Eternal's GPU read fault sits at a fixed
// field of a 1-KiB-stride record in blob pool memory, and DOOM resets its 24 frame descriptor pools ~280 000 times per
// second. Question: does any vkResetDescriptorPool / vkFreeDescriptorSets / vkDestroyDescriptorPool release sets that a
// SUBMITTED, NOT YET FINISHED command buffer still reads ("reset-while-pending", invalid usage, harmless on desktop)?
// On: MALI_COMPAT_DPOOL_DIAG=1, or ADDRMAP on and DPOOL_DIAG unset (=0 turns it off under ADDRMAP: ADDRMAP output then
// byte-identical to 3f782df3). Nothing the driver gets changes: every hook forwards its call untouched, and the
// only extra driver calls are vkGetFenceStatus / vkGetSemaphoreCounterValue on a pending submit's own fence / timeline
// semaphore (at most one per submit per 200 us, only when a reset meets a pool with a not-yet-known-finished submit).
// Tracking: vkAllocateDescriptorSets records set -> (pool slot, pool generation) in a lossy 2-way table (4 MiB);
// vkCmdBindDescriptorSets(2) adds (pool, generation) to the command buffer (max 32 pools, plus max 32 sets of
// FREE_DESCRIPTOR_SET pools); vkQueueSubmit(2) gives each batch a sequence number (ring of 1024) and marks the pools
// of its command buffers with it; a reset / destroy compares the pool's newest mark with the queue's finished
// watermark (fence signalled = every earlier submit on that queue finished), else polls that submit's own fence or
// timeline semaphore. vkResetFences / vkDestroyFence / vkDestroySemaphore / vkQueueWaitIdle / vkDeviceWaitIdle mark
// submits finished (the app may only do these after completion), so a reused fence never gives a false positive.
// Output: with ADDRMAP into the memlog TSV as N events (10th column), else MALI_COMPAT_DPOOL_LOG (default
// /root/mali-dpool-<comm>.txt, "<realtime_ms> <text>" lines). Per second: one "what=dpool-sec" line per pool with
// activity (resets, empty resets, sets allocated, max sets at a reset, frees, failed allocations, reset-while-pending,
// stale binds/submits) and one "what=dpool-sum" line. Immediately: "what=dpool-rwp" (kind=reset|free|destroy, pool,
// sets allocated in the released generation, command buffer, submit seq, submits since, age_us since the submit,
// signal) and "what=dpool-stale" (kind=bind: a set bound after its pool was reset; kind=submit: a command buffer
// submitted after a pool it bound was reset); at most 64 per second, the rest counted ("suppressed"). stderr: the on
// line, the first 8 rwp and the first 8 stale lines, a 10-s line when the rwp / stale / unknown totals changed, end.
// ADDRMAP's per-call "what=dpool-reset" lines are left out while this is on (the per-second counts replace them).
// Hot path (reset): a thread-local pool lookup, ~6 relaxed atomics on the pool's own record, no lock, no I/O.
// ------------------------------------------------------------------------------------------------------------------
#define DPD_POOLS 4096u              // pool slots; slot 0 = "no pool"; at most DPD_POOLS / 2 live pools are tracked
#define DPD_SETS (1u << 19)          // set table entries (8 B each)
#define DPD_PSETS 16384u             // pending sets of FREE_DESCRIPTOR_SET pools (direct mapped, 16 B each)
#define DPD_RING 1024u               // submit records
#define DPD_QMAX 32u                 // queue slots (0xff = untracked queue)
#define DPD_GEN_MASK 0xFFFFFu
#define DPD_TOMB UINT64_MAX
#define DPD_IMM_PER_S 64u
#define DPD_STDERR_MAX 8u
struct DpdPool {
    _Atomic uint64_t h;              // key: pool handle; 0 empty, DPD_TOMB destroyed (keeps the probe chain)
    uint64_t name;                   // the handle for log lines (kept after destroy until the slot is reused)
    _Atomic uint32_t gen;            // +1 per reset, destroy and slot reuse; its low 20 bits go into set entries
    uint32_t flags, maxsets;         // the app's create info
    _Atomic uint32_t since;          // sets allocated since the last reset
    _Atomic uint64_t pend;           // (seq << 8) | queue slot of the newest submit that bound the current generation
    _Atomic uint32_t s_resets, s_empty, s_maxsince, s_sets, s_frees, s_fail, s_rwp, s_rwpfree, s_stale, s_stalesub, s_unk;
    _Atomic int32_t s_lastfail;
};
struct DpdSub {                      // one submitted batch (g_dpd_m)
    uint64_t seq, fence, sem, val, t_ns, poll_ns, cb;
    struct DeviceData *d;
    uint8_t q;
    int8_t poll_res;                 // last poll: 1 = not yet (cached 200 us), else not cached
    bool done;                       // finished without a queue-order proof (failed submit, untracked queue)
};
struct DpdPset { _Atomic uint64_t set, pend; };
enum { DS_SUBMITS, DS_CBS, DS_BINDS, DS_UNKBIND, DS_STALEBIND, DS_STALESUB, DS_RWP, DS_RWPFREE, DS_RWPDESTROY, DS_UNK,
       DS_POLLS, DS_ASSUMED, DS_REFOVF, DS_FSOVF, DS_SUPPR, DS_EVICT, DS_UNTRACKED, DS_FSTAT, DS_FSTAT_NR, DS_FWAIT,
       DS_FWAIT_TO, DS_RESETS, DS_EMPTY, DS_N };
static const char *const g_dpd_sname[DS_N] = {"submits", "cbs", "binds", "unkbind", "stalebind", "stalesub", "rwp",
    "rwpfree", "rwpdestroy", "unk", "polls", "assumed", "refovf", "fsovf", "suppressed", "evict", "untracked", "fstat",
    "fstatnr", "fwait", "fwaitto", "resets", "empty"};
static _Atomic uint64_t g_dpd_s[DS_N];       // this second (the ticker moves them into g_dpd_t)
static uint64_t g_dpd_t[DS_N];               // totals (ticker thread only)
static struct DpdPool *g_dpd_pool;
static _Atomic uint64_t *g_dpd_set;
static struct DpdPset *g_dpd_pset;
static struct DpdSub *g_dpd_ring;
static pthread_mutex_t g_dpd_m = PTHREAD_MUTEX_INITIALIZER; // pool inserts/deletes, the ring, queues, timeline set
static uint64_t g_dpd_seq;                   // last submit seq (g_dpd_m)
static _Atomic uint64_t g_dpd_seq_pub;       // its copy for lock-free readers
static _Atomic uint64_t g_dpd_wm[DPD_QMAX];  // per queue slot: every submit with seq <= wm has finished
static VkQueue g_dpd_q[DPD_QMAX];
static struct DeviceData *g_dpd_qd[DPD_QMAX];
static uint32_t g_dpd_nq, g_dpd_npools;
#define DPD_TL 2048u
static uint64_t g_dpd_tl[DPD_TL];            // timeline semaphores (open hash, DPD_TOMB = deleted; g_dpd_m)
static uint32_t g_dpd_ntl;
static bool g_dpd_tsv;                       // output into the ADDRMAP memlog TSV (else g_dpd_fd)
static int g_dpd_fd = -1;
static char g_dpd_path[300];
static _Atomic uint32_t g_dpd_imm_s;         // immediate lines this second
static _Atomic uint32_t g_dpd_rwp_err, g_dpd_stale_err; // stderr copies so far
static __thread bool tl_dpd_inner;
static __thread uint64_t tl_dpd_ph;          // last pool handle looked up by this thread, and its slot
static __thread uint32_t tl_dpd_ps;
enum { DPN_CPOOL, DPN_DPOOL, DPN_RPOOL, DPN_ALLOC, DPN_FREE, DPN_BIND, DPN_BIND2, DPN_BIND2K, DPN_EXEC, DPN_QS, DPN_QS2,
       DPN_QS2K, DPN_RFENCE, DPN_DFENCE, DPN_FSTAT, DPN_FWAIT, DPN_CSEM, DPN_DSEM, DPN_QWAIT, DPN_DWAIT, DPN_N };
_Static_assert(DPN_N <= DPD_NEXT_MAX, "DeviceData.dpd_next too small");
static PFN_vkVoidFunction g_dpd_next_any[DPN_N];

static inline uint64_t dpd_mix(uint64_t x) {
    x ^= x >> 33; x *= 0xff51afd7ed558ccdULL; x ^= x >> 33; x *= 0xc4ceb9fe1a85ec53ULL; x ^= x >> 33;
    return x;
}
static inline void dpd_cnt(int k, uint64_t n) { atomic_fetch_add_explicit(&g_dpd_s[k], n, memory_order_relaxed); }
static inline void dpd_max32(_Atomic uint32_t *a, uint32_t v) {
    uint32_t o = atomic_load_explicit(a, memory_order_relaxed);
    while (o < v && !atomic_compare_exchange_weak_explicit(a, &o, v, memory_order_relaxed, memory_order_relaxed)) {}
}
static inline void dpd_wm_max(uint32_t q, uint64_t s) {
    if (q >= DPD_QMAX) return;
    uint64_t o = atomic_load_explicit(&g_dpd_wm[q], memory_order_relaxed);
    while (o < s && !atomic_compare_exchange_weak_explicit(&g_dpd_wm[q], &o, s, memory_order_release, memory_order_relaxed)) {}
}
static PFN_vkVoidFunction dpd_next(struct DeviceData *d, int kind) {
    PFN_vkVoidFunction f = d ? d->dpd_next[kind] : NULL;
    return f ? f : g_dpd_next_any[kind];
}

// ---- output: the memlog TSV (N event, mem column = pool or 0) or the own file; never under g_dpd_m.
__attribute__((format(printf, 2, 3)))
static void dpd_out(uint64_t mem, const char *fmt, ...) {
    char b[700];
    va_list ap;
    va_start(ap, fmt);
    int k = vsnprintf(b, sizeof b, fmt, ap);
    va_end(ap);
    if (k < 0) return;
    if (g_dpd_tsv) { am_emit('N', mem, AM_NA, ML_NONE, BUDGET_NO_HEAP, AM_NA, 0, "%s", b); return; }
    if (g_dpd_fd < 0) return;
    char l[760];
    int n = snprintf(l, sizeof l, "%llu %s\n", (unsigned long long)ml_now_ms(), b);
    if (n <= 0) return;
    size_t len = (size_t)n < sizeof l ? (size_t)n : sizeof l - 1;
    if (len == sizeof l - 1) l[len - 1] = '\n';
    ssize_t w = write(g_dpd_fd, l, len);
    (void)w;
}
// An immediate line (rate-limited per second); to_err: also on stderr.
__attribute__((format(printf, 3, 4)))
static void dpd_imm(bool to_err, uint64_t mem, const char *fmt, ...) {
    bool out = atomic_fetch_add_explicit(&g_dpd_imm_s, 1, memory_order_relaxed) < DPD_IMM_PER_S;
    if (!out && !to_err) { dpd_cnt(DS_SUPPR, 1); return; }
    char b[600];
    va_list ap;
    va_start(ap, fmt);
    int k = vsnprintf(b, sizeof b, fmt, ap);
    va_end(ap);
    if (k < 0) return;
    if (out) dpd_out(mem, "%s", b);
    else dpd_cnt(DS_SUPPR, 1);
    if (to_err) fprintf(stderr, "mali-compat: dpool-diag: %s\n", b);
}

// ---- pools
static struct DpdPool *dpd_pool(uint64_t h) {
    if (!h || h == DPD_TOMB || !g_dpd_pool) return NULL;
    if (tl_dpd_ph == h) {
        struct DpdPool *p = &g_dpd_pool[tl_dpd_ps];
        if (atomic_load_explicit(&p->h, memory_order_acquire) == h) return p;
    }
    uint32_t i = (uint32_t)dpd_mix(h) & (DPD_POOLS - 1);
    for (uint32_t k = 0; k < DPD_POOLS; k++, i = (i + 1) & (DPD_POOLS - 1)) {
        if (!i) continue;
        uint64_t v = atomic_load_explicit(&g_dpd_pool[i].h, memory_order_acquire);
        if (v == h) { tl_dpd_ph = h; tl_dpd_ps = i; return &g_dpd_pool[i]; }
        if (!v) return NULL;
    }
    return NULL;
}
static void dpd_pool_add(uint64_t h, uint32_t flags, uint32_t maxsets) {
    if (!h || h == DPD_TOMB) return;
    pthread_mutex_lock(&g_dpd_m);
    struct DpdPool *slot = NULL;
    if (g_dpd_npools < DPD_POOLS / 2) {
        uint32_t i = (uint32_t)dpd_mix(h) & (DPD_POOLS - 1);
        for (uint32_t k = 0; k < DPD_POOLS; k++, i = (i + 1) & (DPD_POOLS - 1)) {
            if (!i) continue;
            uint64_t v = atomic_load_explicit(&g_dpd_pool[i].h, memory_order_relaxed);
            if (v == h) { slot = &g_dpd_pool[i]; break; } // a handle the driver reused without our destroy hook
            if (v == DPD_TOMB && !slot) slot = &g_dpd_pool[i];
            if (!v) { if (!slot) slot = &g_dpd_pool[i]; break; }
        }
    }
    if (slot) {
        bool fresh = atomic_load_explicit(&slot->h, memory_order_relaxed) != h;
        slot->name = h;
        slot->flags = flags;
        slot->maxsets = maxsets;
        atomic_store_explicit(&slot->since, 0, memory_order_relaxed);
        atomic_store_explicit(&slot->pend, 0, memory_order_relaxed);
        atomic_fetch_add_explicit(&slot->gen, 1, memory_order_release); // old set entries of a reused slot go stale
        atomic_store_explicit(&slot->h, h, memory_order_release);
        if (fresh) g_dpd_npools++;
    }
    pthread_mutex_unlock(&g_dpd_m);
    if (!slot) dpd_cnt(DS_UNTRACKED, 1);
}

// ---- sets: entry = fingerprint(32) | generation(20) | slot(12)
static inline uint64_t dpd_fp(uint64_t s) { return (uint64_t)((uint32_t)(dpd_mix(s ^ 0x9e3779b97f4a7c15ULL) >> 32) | 1u); }
static inline bool dpd_entry_dead(uint64_t e) {
    uint32_t s = (uint32_t)(e & 0xfff);
    return !s || (atomic_load_explicit(&g_dpd_pool[s].gen, memory_order_relaxed) & DPD_GEN_MASK) != ((e >> 12) & DPD_GEN_MASK);
}
static void dpd_set_put(uint64_t set, uint32_t slot, uint32_t gen) {
    uint64_t m = dpd_mix(set), fp = dpd_fp(set);
    uint32_t i1 = (uint32_t)m & (DPD_SETS - 1), i2 = (uint32_t)(m >> 32) & (DPD_SETS - 1);
    uint64_t v = fp << 32 | (uint64_t)(gen & DPD_GEN_MASK) << 12 | (slot & 0xfff);
    uint64_t e1 = atomic_load_explicit(&g_dpd_set[i1], memory_order_relaxed);
    if (!e1 || e1 >> 32 == fp || dpd_entry_dead(e1)) { atomic_store_explicit(&g_dpd_set[i1], v, memory_order_release); return; }
    uint64_t e2 = atomic_load_explicit(&g_dpd_set[i2], memory_order_relaxed);
    if (!e2 || e2 >> 32 == fp || dpd_entry_dead(e2)) { atomic_store_explicit(&g_dpd_set[i2], v, memory_order_release); return; }
    atomic_store_explicit(&g_dpd_set[i1], v, memory_order_release);
    dpd_cnt(DS_EVICT, 1);
}
static uint64_t dpd_set_get(uint64_t set, bool drop) {
    uint64_t m = dpd_mix(set), fp = dpd_fp(set);
    uint32_t ix[2] = {(uint32_t)m & (DPD_SETS - 1), (uint32_t)(m >> 32) & (DPD_SETS - 1)};
    for (int k = 0; k < 2; k++) {
        uint64_t e = atomic_load_explicit(&g_dpd_set[ix[k]], memory_order_acquire);
        if (e && e >> 32 == fp) {
            if (drop) atomic_compare_exchange_strong(&g_dpd_set[ix[k]], &e, 0);
            return e;
        }
    }
    return 0;
}
// pending sets of FREE_DESCRIPTOR_SET pools: set -> pend of the newest submit that bound it (lossy)
static void dpd_pset_put(uint64_t set, uint64_t pend) {
    struct DpdPset *p = &g_dpd_pset[dpd_mix(set) & (DPD_PSETS - 1)];
    atomic_store_explicit(&p->set, 0, memory_order_relaxed);
    atomic_store_explicit(&p->pend, pend, memory_order_relaxed);
    atomic_store_explicit(&p->set, set, memory_order_release);
}
static uint64_t dpd_pset_take(uint64_t set) {
    struct DpdPset *p = &g_dpd_pset[dpd_mix(set) & (DPD_PSETS - 1)];
    if (atomic_load_explicit(&p->set, memory_order_acquire) != set) return 0;
    uint64_t pend = atomic_load_explicit(&p->pend, memory_order_relaxed);
    uint64_t s = set;
    if (!atomic_compare_exchange_strong(&p->set, &s, 0)) return 0;
    return pend;
}

// ---- submits and their completion
static bool dpd_tl_has(uint64_t s) { // g_dpd_m held
    if (!s || !g_dpd_ntl) return false;
    uint32_t i = (uint32_t)dpd_mix(s) & (DPD_TL - 1);
    for (uint32_t k = 0; k < DPD_TL; k++, i = (i + 1) & (DPD_TL - 1)) {
        if (g_dpd_tl[i] == s) return true;
        if (!g_dpd_tl[i]) return false;
    }
    return false;
}
static void dpd_tl_set(uint64_t s, bool add) { // g_dpd_m held
    if (!s || s == DPD_TOMB) return;
    uint32_t i = (uint32_t)dpd_mix(s) & (DPD_TL - 1), tomb = UINT32_MAX;
    for (uint32_t k = 0; k < DPD_TL; k++, i = (i + 1) & (DPD_TL - 1)) {
        if (g_dpd_tl[i] == s) { if (!add) { g_dpd_tl[i] = DPD_TOMB; g_dpd_ntl--; } return; }
        if (g_dpd_tl[i] == DPD_TOMB && tomb == UINT32_MAX) tomb = i;
        if (!g_dpd_tl[i]) break;
    }
    if (!add || g_dpd_ntl >= DPD_TL / 2) return; // full: that semaphore is treated as binary (not polled)
    if (tomb == UINT32_MAX) tomb = i;
    if (g_dpd_tl[tomb] && g_dpd_tl[tomb] != DPD_TOMB) return;
    g_dpd_tl[tomb] = s;
    g_dpd_ntl++;
}
static uint32_t dpd_qslot(VkQueue queue, struct DeviceData *d, bool add) { // g_dpd_m held
    for (uint32_t i = 0; i < g_dpd_nq; i++) if (g_dpd_q[i] == queue && g_dpd_qd[i] == d) return i;
    if (!add || g_dpd_nq >= DPD_QMAX) return 0xff;
    g_dpd_q[g_dpd_nq] = queue;
    g_dpd_qd[g_dpd_nq] = d;
    return g_dpd_nq++;
}
// Poll one record's own signal: 0 finished, 1 not yet, -1 none / unusable. g_dpd_m held (serialises with destroy).
// fresh: never answer from the 200-us NOT_READY cache (the app may have seen the fence signal since; review N1).
static int dpd_poll(struct DpdSub *r, uint64_t now, bool fresh) {
    if (!r->d) return -1;
    if (!fresh && r->poll_res == 1 && now - r->poll_ns < 200000u) return 1;
    VkResult res;
    if (r->fence && r->d->dpd_fence_status) {
        res = r->d->dpd_fence_status(r->d->device, (VkFence)(uintptr_t)r->fence);
        dpd_cnt(DS_POLLS, 1);
        r->poll_ns = now;
        r->poll_res = res == VK_SUCCESS ? 0 : res == VK_NOT_READY ? 1 : -1;
        return r->poll_res;
    }
    if (r->sem && r->d->dpd_sem_value) {
        uint64_t v = 0;
        res = r->d->dpd_sem_value(r->d->device, (VkSemaphore)(uintptr_t)r->sem, &v);
        dpd_cnt(DS_POLLS, 1);
        r->poll_ns = now;
        r->poll_res = res != VK_SUCCESS ? -1 : v >= r->val ? 0 : 1;
        return r->poll_res;
    }
    return -1;
}
struct DpdHit { uint64_t seq, last, age_ns, cb; uint32_t q; const char *sig; };
// 0 = finished, 1 = pending (its own fence / timeline value not reached), 2 = unknown (no own signal and no later
// signalled submit on the same queue). g_dpd_m held.
static int dpd_state_locked(uint64_t seq, uint32_t q, struct DpdHit *h) {
    if (q < DPD_QMAX && atomic_load_explicit(&g_dpd_wm[q], memory_order_acquire) >= seq) return 0;
    if (seq + DPD_RING <= g_dpd_seq) { dpd_cnt(DS_ASSUMED, 1); return 0; } // left the ring: assumed finished
    struct DpdSub *r = &g_dpd_ring[seq % DPD_RING];
    if (r->seq != seq || r->done) return 0;
    uint64_t now = now_ns();
    h->seq = seq;
    h->last = g_dpd_seq;
    h->age_ns = now - r->t_ns;
    h->cb = r->cb;
    h->q = q;
    h->sig = r->fence ? "fence" : r->sem ? "timeline" : "none";
    int own = dpd_poll(r, now, true); // a pending verdict always rests on a fresh poll of the submit's own signal
    if (own == 0) { dpd_wm_max(q, seq); return 0; }
    if (own == 1) return 1;
    if (q >= DPD_QMAX) return 2;
    for (uint64_t s = seq + 1; s <= g_dpd_seq; s++) {
        struct DpdSub *x = &g_dpd_ring[s % DPD_RING];
        if (x->seq != s || x->q != q || x->done) continue;
        int p = dpd_poll(x, now, false);
        if (p == 0) { dpd_wm_max(q, s); return 0; }
        if (p == 1) { h->sig = "later-pending"; return 2; }
    }
    h->sig = "nosignal";
    return 2;
}
static int dpd_state(uint64_t pend, struct DpdHit *h) {
    uint64_t seq = pend >> 8;
    uint32_t q = (uint32_t)(pend & 0xff);
    if (!seq) return 0;
    if (q < DPD_QMAX && atomic_load_explicit(&g_dpd_wm[q], memory_order_acquire) >= seq) return 0;
    pthread_mutex_lock(&g_dpd_m);
    int st = dpd_state_locked(seq, q, h);
    pthread_mutex_unlock(&g_dpd_m);
    return st;
}
// kind 0 reset, 1 free, 2 destroy
static void dpd_check(struct DpdPool *P, uint64_t pend, int kind, uint32_t since, uint64_t set) {
    struct DpdHit h = {0};
    int st = dpd_state(pend, &h);
    if (!st) return;
    if (st == 2) { atomic_fetch_add_explicit(&P->s_unk, 1, memory_order_relaxed); dpd_cnt(DS_UNK, 1); return; }
    atomic_fetch_add_explicit(kind == 1 ? &P->s_rwpfree : &P->s_rwp, 1, memory_order_relaxed);
    dpd_cnt(kind == 0 ? DS_RWP : kind == 1 ? DS_RWPFREE : DS_RWPDESTROY, 1);
    bool err = atomic_fetch_add_explicit(&g_dpd_rwp_err, 1, memory_order_relaxed) < DPD_STDERR_MAX;
    dpd_imm(err, P->name, "what=dpool-rwp kind=%s pool=0x%llx since=%u set=0x%llx cb=0x%llx seq=%llu lastseq=%llu behind=%llu "
            "age_us=%llu sig=%s q=%u flags=0x%x maxsets=%u", kind == 0 ? "reset" : kind == 1 ? "free" : "destroy",
            (unsigned long long)P->name, since, (unsigned long long)set,
            (unsigned long long)h.cb, (unsigned long long)h.seq,
            (unsigned long long)h.last, (unsigned long long)(h.last - h.seq), (unsigned long long)(h.age_ns / 1000u), h.sig,
            h.q, P->flags, P->maxsets);
}
static void dpd_stale(struct DpdPool *P, int submit, uint64_t cb, uint64_t set, uint64_t seq) {
    atomic_fetch_add_explicit(submit ? &P->s_stalesub : &P->s_stale, 1, memory_order_relaxed);
    dpd_cnt(submit ? DS_STALESUB : DS_STALEBIND, 1);
    bool err = atomic_fetch_add_explicit(&g_dpd_stale_err, 1, memory_order_relaxed) < DPD_STDERR_MAX;
    dpd_imm(err, P->name, "what=dpool-stale kind=%s pool=0x%llx cb=0x%llx set=0x%llx seq=%llu", submit ? "submit" : "bind",
            (unsigned long long)P->name, (unsigned long long)cb, (unsigned long long)set, (unsigned long long)seq);
}
// A batch record per VkSubmitInfo(2) (one for a submit with no batch: its fence still orders the queue).
static uint64_t dpd_submit_begin(VkQueue queue, struct DeviceData *d, uint32_t n, const VkSubmitInfo *si,
                                 const VkSubmitInfo2 *si2, VkFence fence, uint32_t *qout) {
    uint32_t nb = n ? n : 1;
    uint64_t now = now_ns();
    pthread_mutex_lock(&g_dpd_m);
    uint32_t q = dpd_qslot(queue, d, true);
    uint64_t first = g_dpd_seq + 1;
    for (uint32_t i = 0; i < nb; i++) {
        uint64_t seq = ++g_dpd_seq;
        struct DpdSub *r = &g_dpd_ring[seq % DPD_RING];
        memset(r, 0, sizeof *r);
        r->seq = seq;
        r->q = (uint8_t)q;
        r->d = d;
        r->t_ns = now;
        if (i + 1 == nb) r->fence = (uint64_t)(uintptr_t)fence;
        if (!n) continue;
        if (si) {
            if (si[i].commandBufferCount && si[i].pCommandBuffers) r->cb = (uint64_t)(uintptr_t)si[i].pCommandBuffers[0];
            const VkTimelineSemaphoreSubmitInfo *ts = find_struct(si[i].pNext, VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO);
            for (uint32_t k = 0; ts && ts->pSignalSemaphoreValues && si[i].pSignalSemaphores && k < si[i].signalSemaphoreCount &&
                                 k < ts->signalSemaphoreValueCount; k++)
                if (dpd_tl_has((uint64_t)(uintptr_t)si[i].pSignalSemaphores[k])) {
                    r->sem = (uint64_t)(uintptr_t)si[i].pSignalSemaphores[k];
                    r->val = ts->pSignalSemaphoreValues[k];
                    break;
                }
        } else {
            if (si2[i].commandBufferInfoCount && si2[i].pCommandBufferInfos)
                r->cb = (uint64_t)(uintptr_t)si2[i].pCommandBufferInfos[0].commandBuffer;
            for (uint32_t k = 0; si2[i].pSignalSemaphoreInfos && k < si2[i].signalSemaphoreInfoCount; k++)
                if (dpd_tl_has((uint64_t)(uintptr_t)si2[i].pSignalSemaphoreInfos[k].semaphore)) {
                    r->sem = (uint64_t)(uintptr_t)si2[i].pSignalSemaphoreInfos[k].semaphore;
                    r->val = si2[i].pSignalSemaphoreInfos[k].value;
                    break;
                }
        }
    }
    atomic_store_explicit(&g_dpd_seq_pub, g_dpd_seq, memory_order_release);
    pthread_mutex_unlock(&g_dpd_m);
    *qout = q;
    return first;
}
static void dpd_submit_failed(uint64_t first, uint32_t nb) {
    pthread_mutex_lock(&g_dpd_m);
    for (uint64_t s = first; s < first + nb; s++) {
        struct DpdSub *r = &g_dpd_ring[s % DPD_RING];
        if (r->seq == s) r->done = true;
    }
    pthread_mutex_unlock(&g_dpd_m);
}
// The pools (and FREE-pool sets) this command buffer bound get the submit's mark.
static void dpd_mark_cb(VkCommandBuffer cb, uint64_t seq, uint32_t q) {
    struct CmdBufData *c = cb ? cmd_data(cb) : NULL;
    if (!c) return;
    struct DpdCmd *m = &c->dpd;
    uint64_t pend = seq << 8 | (q & 0xff);
    for (uint32_t i = 0; i < m->nref; i++) {
        uint32_t slot = m->ref[i] & 0xfff, gen = m->ref[i] >> 12;
        struct DpdPool *P = &g_dpd_pool[slot];
        if ((atomic_load_explicit(&P->gen, memory_order_acquire) & DPD_GEN_MASK) != gen) {
            dpd_stale(P, 1, (uint64_t)(uintptr_t)cb, 0, seq);
            continue;
        }
        uint64_t o = atomic_load_explicit(&P->pend, memory_order_relaxed);
        while ((o >> 8) < seq && !atomic_compare_exchange_weak_explicit(&P->pend, &o, pend, memory_order_release, memory_order_relaxed)) {}
    }
    for (uint32_t i = 0; i < m->nfs; i++) dpd_pset_put(m->fs[i], pend);
    dpd_cnt(DS_CBS, 1);
    if (m->binds) dpd_cnt(DS_BINDS, m->binds);
    if (m->unk) dpd_cnt(DS_UNKBIND, m->unk);
    if (m->refovf) dpd_cnt(DS_REFOVF, m->refovf);
    if (m->fsovf) dpd_cnt(DS_FSOVF, m->fsovf);
    m->binds = m->unk = m->refovf = m->fsovf = 0;
}
static void dpd_bind(VkCommandBuffer cmd, uint32_t n, const VkDescriptorSet *sets) {
    struct CmdBufData *c = sets ? cmd_data(cmd) : NULL;
    if (!c) return;
    struct DpdCmd *m = &c->dpd;
    for (uint32_t i = 0; i < n; i++) {
        uint64_t s = (uint64_t)(uintptr_t)sets[i];
        if (!s) continue;
        m->binds++;
        uint64_t e = dpd_set_get(s, false);
        if (!e) { m->unk++; continue; }
        uint32_t slot = (uint32_t)(e & 0xfff), gen = (uint32_t)(e >> 12) & DPD_GEN_MASK;
        struct DpdPool *P = &g_dpd_pool[slot];
        if ((atomic_load_explicit(&P->gen, memory_order_acquire) & DPD_GEN_MASK) != gen) {
            dpd_stale(P, 0, (uint64_t)(uintptr_t)cmd, s, 0);
            continue;
        }
        uint32_t ref = slot | gen << 12;
        if (!(m->nref && m->ref[m->last] == ref)) {
            uint32_t k = 0;
            while (k < m->nref && m->ref[k] != ref) k++;
            if (k < m->nref) m->last = k;
            else if (m->nref < DPD_REFS) { m->ref[m->nref] = ref; m->last = m->nref++; }
            else m->refovf++;
        }
        if (P->flags & VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT) {
            uint32_t k = 0;
            while (k < m->nfs && m->fs[k] != s) k++;
            if (k == m->nfs) { if (m->nfs < DPD_FSETS) m->fs[m->nfs++] = s; else m->fsovf++; }
        }
    }
}

// ---- hooks (each forwards to the entry point the rest of vkGetDeviceProcAddr hands out)
static VKAPI_ATTR VkResult VKAPI_CALL dpd_CreateDescriptorPool(VkDevice device, const VkDescriptorPoolCreateInfo *ci,
                                                               const VkAllocationCallbacks *a, VkDescriptorPool *p) {
    VkResult r = ((PFN_vkCreateDescriptorPool)dpd_next(get_device_data(device), DPN_CPOOL))(device, ci, a, p);
    if (r == VK_SUCCESS && p && *p && ci) dpd_pool_add((uint64_t)(uintptr_t)*p, ci->flags, ci->maxSets);
    return r;
}
static VKAPI_ATTR void VKAPI_CALL dpd_DestroyDescriptorPool(VkDevice device, VkDescriptorPool pool, const VkAllocationCallbacks *a) {
    struct DpdPool *P = dpd_pool((uint64_t)(uintptr_t)pool);
    if (P) {
        uint32_t since = atomic_exchange_explicit(&P->since, 0, memory_order_relaxed);
        uint64_t pend = atomic_exchange_explicit(&P->pend, 0, memory_order_acq_rel);
        if (pend) dpd_check(P, pend, 2, since, 0);
        pthread_mutex_lock(&g_dpd_m);
        if (atomic_load_explicit(&P->h, memory_order_relaxed) == (uint64_t)(uintptr_t)pool) {
            atomic_fetch_add_explicit(&P->gen, 1, memory_order_release);
            atomic_store_explicit(&P->h, DPD_TOMB, memory_order_release);
            g_dpd_npools--;
        }
        pthread_mutex_unlock(&g_dpd_m);
    }
    ((PFN_vkDestroyDescriptorPool)dpd_next(get_device_data(device), DPN_DPOOL))(device, pool, a);
}
static VKAPI_ATTR VkResult VKAPI_CALL dpd_ResetDescriptorPool(VkDevice device, VkDescriptorPool pool, VkDescriptorPoolResetFlags fl) {
    struct DpdPool *P = dpd_pool((uint64_t)(uintptr_t)pool);
    if (P) {
        uint32_t since = atomic_exchange_explicit(&P->since, 0, memory_order_relaxed);
        atomic_fetch_add_explicit(&P->s_resets, 1, memory_order_relaxed);
        if (!since) atomic_fetch_add_explicit(&P->s_empty, 1, memory_order_relaxed);
        else dpd_max32(&P->s_maxsince, since);
        atomic_fetch_add_explicit(&P->gen, 1, memory_order_release);
        uint64_t pend = atomic_load_explicit(&P->pend, memory_order_relaxed);
        if (pend) {
            pend = atomic_exchange_explicit(&P->pend, 0, memory_order_acq_rel);
            if (pend) dpd_check(P, pend, 0, since, 0);
        }
    } else dpd_cnt(DS_UNTRACKED, 1);
    return ((PFN_vkResetDescriptorPool)dpd_next(get_device_data(device), DPN_RPOOL))(device, pool, fl);
}
static VKAPI_ATTR VkResult VKAPI_CALL dpd_AllocateDescriptorSets(VkDevice device, const VkDescriptorSetAllocateInfo *ai,
                                                                 VkDescriptorSet *sets) {
    VkResult r = ((PFN_vkAllocateDescriptorSets)dpd_next(get_device_data(device), DPN_ALLOC))(device, ai, sets);
    struct DpdPool *P = ai ? dpd_pool((uint64_t)(uintptr_t)ai->descriptorPool) : NULL;
    if (P) {
        if (r == VK_SUCCESS && sets) {
            uint32_t slot = (uint32_t)(P - g_dpd_pool), gen = atomic_load_explicit(&P->gen, memory_order_acquire) & DPD_GEN_MASK;
            for (uint32_t i = 0; i < ai->descriptorSetCount; i++)
                if (sets[i]) dpd_set_put((uint64_t)(uintptr_t)sets[i], slot, gen);
            atomic_fetch_add_explicit(&P->since, ai->descriptorSetCount, memory_order_relaxed);
            atomic_fetch_add_explicit(&P->s_sets, ai->descriptorSetCount, memory_order_relaxed);
        } else {
            atomic_fetch_add_explicit(&P->s_fail, 1, memory_order_relaxed);
            atomic_store_explicit(&P->s_lastfail, (int32_t)r, memory_order_relaxed);
        }
    }
    return r;
}
static VKAPI_ATTR VkResult VKAPI_CALL dpd_FreeDescriptorSets(VkDevice device, VkDescriptorPool pool, uint32_t n, const VkDescriptorSet *sets) {
    struct DpdPool *P = dpd_pool((uint64_t)(uintptr_t)pool);
    if (P && sets) {
        atomic_fetch_add_explicit(&P->s_frees, n, memory_order_relaxed);
        for (uint32_t i = 0; i < n; i++) {
            uint64_t s = (uint64_t)(uintptr_t)sets[i];
            if (!s) continue;
            uint64_t pend = dpd_pset_take(s);
            if (pend) dpd_check(P, pend, 1, 0, s);
            dpd_set_get(s, true);
        }
    }
    return ((PFN_vkFreeDescriptorSets)dpd_next(get_device_data(device), DPN_FREE))(device, pool, n, sets);
}
static VKAPI_ATTR void VKAPI_CALL dpd_CmdBindDescriptorSets(VkCommandBuffer cmd, VkPipelineBindPoint bp, VkPipelineLayout layout,
                                                            uint32_t first, uint32_t n, const VkDescriptorSet *sets,
                                                            uint32_t ndyn, const uint32_t *dyn) {
    dpd_bind(cmd, n, sets);
    ((PFN_vkCmdBindDescriptorSets)dpd_next(get_device_data_by_cmd(cmd), DPN_BIND))(cmd, bp, layout, first, n, sets, ndyn, dyn);
}
static VKAPI_ATTR void VKAPI_CALL dpd_CmdBindDescriptorSets2(VkCommandBuffer cmd, const VkBindDescriptorSetsInfo *i) {
    if (i) dpd_bind(cmd, i->descriptorSetCount, i->pDescriptorSets);
    ((PFN_vkCmdBindDescriptorSets2)dpd_next(get_device_data_by_cmd(cmd), DPN_BIND2))(cmd, i);
}
static VKAPI_ATTR void VKAPI_CALL dpd_CmdBindDescriptorSets2KHR(VkCommandBuffer cmd, const VkBindDescriptorSetsInfo *i) {
    if (i) dpd_bind(cmd, i->descriptorSetCount, i->pDescriptorSets);
    ((PFN_vkCmdBindDescriptorSets2)dpd_next(get_device_data_by_cmd(cmd), DPN_BIND2K))(cmd, i);
}
static VKAPI_ATTR void VKAPI_CALL dpd_CmdExecuteCommands(VkCommandBuffer cmd, uint32_t count, const VkCommandBuffer *secs) {
    for (uint32_t s = 0; secs && s < count; s++) {
        struct CmdBufData *sc = secs[s] ? cmd_data(secs[s]) : NULL;
        struct CmdBufData *c = cmd_data(cmd);
        if (!sc || !c || sc == c) continue;
        struct DpdCmd *m = &c->dpd, *x = &sc->dpd;
        for (uint32_t i = 0; i < x->nref; i++) {
            uint32_t k = 0;
            while (k < m->nref && m->ref[k] != x->ref[i]) k++;
            if (k == m->nref) { if (m->nref < DPD_REFS) m->ref[m->nref++] = x->ref[i]; else m->refovf++; }
        }
        for (uint32_t i = 0; i < x->nfs; i++) {
            uint32_t k = 0;
            while (k < m->nfs && m->fs[k] != x->fs[i]) k++;
            if (k == m->nfs) { if (m->nfs < DPD_FSETS) m->fs[m->nfs++] = x->fs[i]; else m->fsovf++; }
        }
        m->refovf += x->refovf;
        m->fsovf += x->fsovf;
    }
    ((PFN_vkCmdExecuteCommands)dpd_next(get_device_data_by_cmd(cmd), DPN_EXEC))(cmd, count, secs);
}
static VkResult dpd_submit(bool v2, int kind, VkQueue queue, uint32_t n, const VkSubmitInfo *si, const VkSubmitInfo2 *si2,
                           VkFence fence) {
    struct DeviceData *d = get_device_data_by_cmd((VkCommandBuffer)queue);
    uint32_t nt = (v2 ? (const void *)si2 : (const void *)si) ? n : 0; // tracked batches (the driver still gets n)
    uint32_t q;
    uint64_t first = dpd_submit_begin(queue, d, nt, v2 ? NULL : si, v2 ? si2 : NULL, fence, &q);
    for (uint32_t i = 0; i < nt; i++) {
        if (!v2) for (uint32_t j = 0; si[i].pCommandBuffers && j < si[i].commandBufferCount; j++) dpd_mark_cb(si[i].pCommandBuffers[j], first + i, q);
        else for (uint32_t j = 0; si2[i].pCommandBufferInfos && j < si2[i].commandBufferInfoCount; j++)
            dpd_mark_cb(si2[i].pCommandBufferInfos[j].commandBuffer, first + i, q);
    }
    dpd_cnt(DS_SUBMITS, 1);
    VkResult r = v2 ? ((PFN_vkQueueSubmit2)dpd_next(d, kind))(queue, n, si2, fence)
                    : ((PFN_vkQueueSubmit)dpd_next(d, kind))(queue, n, si, fence);
    if (r != VK_SUCCESS) dpd_submit_failed(first, nt ? nt : 1);
    return r;
}
static VKAPI_ATTR VkResult VKAPI_CALL dpd_QueueSubmit(VkQueue queue, uint32_t n, const VkSubmitInfo *si, VkFence fence) {
    return dpd_submit(false, DPN_QS, queue, n, si, NULL, fence);
}
static VKAPI_ATTR VkResult VKAPI_CALL dpd_QueueSubmit2(VkQueue queue, uint32_t n, const VkSubmitInfo2 *si, VkFence fence) {
    return dpd_submit(true, DPN_QS2, queue, n, NULL, si, fence);
}
static VKAPI_ATTR VkResult VKAPI_CALL dpd_QueueSubmit2KHR(VkQueue queue, uint32_t n, const VkSubmitInfo2 *si, VkFence fence) {
    return dpd_submit(true, DPN_QS2K, queue, n, NULL, si, fence);
}
// Finished by the app's own word (reset / destroy are only valid after completion): the record leaves the poll set.
static void dpd_fences_done(uint32_t n, const VkFence *f) { // g_dpd_m held
    for (uint32_t k = 0; f && k < n; k++) {
        uint64_t h = (uint64_t)(uintptr_t)f[k];
        if (!h) continue;
        for (uint32_t i = 0; i < DPD_RING; i++) {
            struct DpdSub *r = &g_dpd_ring[i];
            if (!r->seq || r->fence != h) continue;
            if (r->q < DPD_QMAX) dpd_wm_max(r->q, r->seq);
            else r->done = true;
            r->fence = 0;
        }
    }
}
static VKAPI_ATTR VkResult VKAPI_CALL dpd_ResetFences(VkDevice device, uint32_t n, const VkFence *f) {
    pthread_mutex_lock(&g_dpd_m);
    dpd_fences_done(n, f);
    pthread_mutex_unlock(&g_dpd_m);
    return ((PFN_vkResetFences)dpd_next(get_device_data(device), DPN_RFENCE))(device, n, f);
}
static VKAPI_ATTR void VKAPI_CALL dpd_DestroyFence(VkDevice device, VkFence f, const VkAllocationCallbacks *a) {
    pthread_mutex_lock(&g_dpd_m);
    dpd_fences_done(1, &f);
    pthread_mutex_unlock(&g_dpd_m);
    ((PFN_vkDestroyFence)dpd_next(get_device_data(device), DPN_DFENCE))(device, f, a);
}
static VKAPI_ATTR VkResult VKAPI_CALL dpd_GetFenceStatus(VkDevice device, VkFence f) {
    VkResult r = ((PFN_vkGetFenceStatus)dpd_next(get_device_data(device), DPN_FSTAT))(device, f);
    dpd_cnt(DS_FSTAT, 1);
    if (r == VK_NOT_READY) dpd_cnt(DS_FSTAT_NR, 1);
    return r;
}
static VKAPI_ATTR VkResult VKAPI_CALL dpd_WaitForFences(VkDevice device, uint32_t n, const VkFence *f, VkBool32 all, uint64_t timeout) {
    VkResult r = ((PFN_vkWaitForFences)dpd_next(get_device_data(device), DPN_FWAIT))(device, n, f, all, timeout);
    dpd_cnt(DS_FWAIT, 1);
    if (r == VK_TIMEOUT) dpd_cnt(DS_FWAIT_TO, 1);
    return r;
}
static VKAPI_ATTR VkResult VKAPI_CALL dpd_CreateSemaphore(VkDevice device, const VkSemaphoreCreateInfo *ci, const VkAllocationCallbacks *a,
                                                          VkSemaphore *s) {
    VkResult r = ((PFN_vkCreateSemaphore)dpd_next(get_device_data(device), DPN_CSEM))(device, ci, a, s);
    const VkSemaphoreTypeCreateInfo *t = ci ? find_struct(ci->pNext, VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO) : NULL;
    if (r == VK_SUCCESS && s && *s && t && t->semaphoreType == VK_SEMAPHORE_TYPE_TIMELINE) {
        pthread_mutex_lock(&g_dpd_m);
        dpd_tl_set((uint64_t)(uintptr_t)*s, true);
        pthread_mutex_unlock(&g_dpd_m);
    }
    return r;
}
static VKAPI_ATTR void VKAPI_CALL dpd_DestroySemaphore(VkDevice device, VkSemaphore s, const VkAllocationCallbacks *a) {
    uint64_t h = (uint64_t)(uintptr_t)s;
    if (h) {
        pthread_mutex_lock(&g_dpd_m);
        for (uint32_t i = 0; i < DPD_RING; i++) {
            struct DpdSub *r = &g_dpd_ring[i];
            if (!r->seq || r->sem != h) continue;
            if (r->q < DPD_QMAX) dpd_wm_max(r->q, r->seq);
            else r->done = true;
            r->sem = 0;
        }
        dpd_tl_set(h, false);
        pthread_mutex_unlock(&g_dpd_m);
    }
    ((PFN_vkDestroySemaphore)dpd_next(get_device_data(device), DPN_DSEM))(device, s, a);
}
static VKAPI_ATTR VkResult VKAPI_CALL dpd_QueueWaitIdle(VkQueue queue) {
    struct DeviceData *d = get_device_data_by_cmd((VkCommandBuffer)queue);
    uint64_t snap = atomic_load_explicit(&g_dpd_seq_pub, memory_order_acquire);
    VkResult r = ((PFN_vkQueueWaitIdle)dpd_next(d, DPN_QWAIT))(queue);
    if (r == VK_SUCCESS) {
        pthread_mutex_lock(&g_dpd_m);
        uint32_t q = dpd_qslot(queue, d, false);
        pthread_mutex_unlock(&g_dpd_m);
        dpd_wm_max(q, snap);
    }
    return r;
}
static VKAPI_ATTR VkResult VKAPI_CALL dpd_DeviceWaitIdle(VkDevice device) {
    struct DeviceData *d = get_device_data(device);
    uint64_t snap = atomic_load_explicit(&g_dpd_seq_pub, memory_order_acquire);
    VkResult r = ((PFN_vkDeviceWaitIdle)dpd_next(d, DPN_DWAIT))(device);
    if (r == VK_SUCCESS) {
        pthread_mutex_lock(&g_dpd_m);
        for (uint32_t q = 0; q < g_dpd_nq; q++) if (g_dpd_qd[q] == d) dpd_wm_max(q, snap);
        pthread_mutex_unlock(&g_dpd_m);
    }
    return r;
}
static const struct { const char *name; uint8_t kind; PFN_vkVoidFunction fn; } g_dpd_names[] = {
    {"vkCreateDescriptorPool", DPN_CPOOL, (PFN_vkVoidFunction)dpd_CreateDescriptorPool},
    {"vkDestroyDescriptorPool", DPN_DPOOL, (PFN_vkVoidFunction)dpd_DestroyDescriptorPool},
    {"vkResetDescriptorPool", DPN_RPOOL, (PFN_vkVoidFunction)dpd_ResetDescriptorPool},
    {"vkAllocateDescriptorSets", DPN_ALLOC, (PFN_vkVoidFunction)dpd_AllocateDescriptorSets},
    {"vkFreeDescriptorSets", DPN_FREE, (PFN_vkVoidFunction)dpd_FreeDescriptorSets},
    {"vkCmdBindDescriptorSets", DPN_BIND, (PFN_vkVoidFunction)dpd_CmdBindDescriptorSets},
    {"vkCmdBindDescriptorSets2", DPN_BIND2, (PFN_vkVoidFunction)dpd_CmdBindDescriptorSets2},
    {"vkCmdBindDescriptorSets2KHR", DPN_BIND2K, (PFN_vkVoidFunction)dpd_CmdBindDescriptorSets2KHR},
    {"vkCmdExecuteCommands", DPN_EXEC, (PFN_vkVoidFunction)dpd_CmdExecuteCommands},
    {"vkQueueSubmit", DPN_QS, (PFN_vkVoidFunction)dpd_QueueSubmit},
    {"vkQueueSubmit2", DPN_QS2, (PFN_vkVoidFunction)dpd_QueueSubmit2},
    {"vkQueueSubmit2KHR", DPN_QS2K, (PFN_vkVoidFunction)dpd_QueueSubmit2KHR},
    {"vkResetFences", DPN_RFENCE, (PFN_vkVoidFunction)dpd_ResetFences},
    {"vkDestroyFence", DPN_DFENCE, (PFN_vkVoidFunction)dpd_DestroyFence},
    {"vkGetFenceStatus", DPN_FSTAT, (PFN_vkVoidFunction)dpd_GetFenceStatus},
    {"vkWaitForFences", DPN_FWAIT, (PFN_vkVoidFunction)dpd_WaitForFences},
    {"vkCreateSemaphore", DPN_CSEM, (PFN_vkVoidFunction)dpd_CreateSemaphore},
    {"vkDestroySemaphore", DPN_DSEM, (PFN_vkVoidFunction)dpd_DestroySemaphore},
    {"vkQueueWaitIdle", DPN_QWAIT, (PFN_vkVoidFunction)dpd_QueueWaitIdle},
    {"vkDeviceWaitIdle", DPN_DWAIT, (PFN_vkVoidFunction)dpd_DeviceWaitIdle},
};
// Called near the top of vkGetDeviceProcAddr while on: the wrapper, with what the rest of vkGetDeviceProcAddr hands
// out (this layer's hook or the driver's) as its next. NULL = not ours, or the name does not exist below.
static PFN_vkVoidFunction dpd_proc(VkDevice device, const char *pName) {
    if (strncmp(pName, "vk", 2)) return NULL;
    uint32_t k = 0;
    while (k < sizeof g_dpd_names / sizeof g_dpd_names[0] && strcmp(g_dpd_names[k].name, pName)) k++;
    if (k == sizeof g_dpd_names / sizeof g_dpd_names[0]) return NULL;
    struct DeviceData *d = get_device_data(device);
    if (!d) return NULL;
    tl_dpd_inner = true;
    PFN_vkVoidFunction inner = vkGetDeviceProcAddr(device, pName);
    tl_dpd_inner = false;
    if (!inner) return NULL;
    int kind = g_dpd_names[k].kind;
    pthread_mutex_lock(&g_dpd_m);
    if (!d->dpd_next[kind]) d->dpd_next[kind] = inner;
    if (!g_dpd_next_any[kind]) g_dpd_next_any[kind] = inner;
    pthread_mutex_unlock(&g_dpd_m);
    return g_dpd_names[k].fn;
}

// ---- per-second lines (own thread)
static void dpd_tick(uint64_t sec) {
    uint32_t live = 0, active = 0;
    uint64_t resets = 0, empty = 0, sets = 0, frees = 0, fails = 0;
    for (uint32_t i = 1; i < DPD_POOLS; i++) {
        struct DpdPool *P = &g_dpd_pool[i];
        uint64_t h = atomic_load_explicit(&P->h, memory_order_acquire);
        if (!h) continue;
        if (h != DPD_TOMB) live++;
        uint32_t r = atomic_exchange_explicit(&P->s_resets, 0, memory_order_relaxed);
        uint32_t e = atomic_exchange_explicit(&P->s_empty, 0, memory_order_relaxed);
        uint32_t ms = atomic_exchange_explicit(&P->s_maxsince, 0, memory_order_relaxed);
        uint32_t st = atomic_exchange_explicit(&P->s_sets, 0, memory_order_relaxed);
        uint32_t fr = atomic_exchange_explicit(&P->s_frees, 0, memory_order_relaxed);
        uint32_t fl = atomic_exchange_explicit(&P->s_fail, 0, memory_order_relaxed);
        uint32_t rw = atomic_exchange_explicit(&P->s_rwp, 0, memory_order_relaxed);
        uint32_t rf = atomic_exchange_explicit(&P->s_rwpfree, 0, memory_order_relaxed);
        uint32_t sb = atomic_exchange_explicit(&P->s_stale, 0, memory_order_relaxed);
        uint32_t ss = atomic_exchange_explicit(&P->s_stalesub, 0, memory_order_relaxed);
        uint32_t uk = atomic_exchange_explicit(&P->s_unk, 0, memory_order_relaxed);
        if (!(r | st | fr | fl | rw | rf | sb | ss | uk)) continue;
        active++;
        resets += r; empty += e; sets += st; frees += fr; fails += fl;
        dpd_out(P->name, "what=dpool-sec pool=0x%llx resets=%u empty=%u maxsince=%u sets=%u frees=%u allocfail=%u lastfail=%d "
                "rwp=%u rwpfree=%u stalebind=%u stalesub=%u unk=%u maxsets=%u flags=0x%x%s", (unsigned long long)P->name, r, e, ms,
                st, fr, fl, fl ? atomic_load_explicit(&P->s_lastfail, memory_order_relaxed) : 0, rw, rf, sb, ss, uk, P->maxsets,
                P->flags, h == DPD_TOMB ? " destroyed" : "");
    }
    dpd_cnt(DS_RESETS, resets);
    dpd_cnt(DS_EMPTY, empty);
    uint64_t v[DS_N];
    for (int k = 0; k < DS_N; k++) { v[k] = atomic_exchange_explicit(&g_dpd_s[k], 0, memory_order_relaxed); g_dpd_t[k] += v[k]; }
    atomic_store_explicit(&g_dpd_imm_s, 0, memory_order_relaxed);
    char b[640];
    size_t o = 0;
    for (int k = 0; k < DS_N && o < sizeof b; k++)
        if (k != DS_RESETS && k != DS_EMPTY) o += (size_t)snprintf(b + o, sizeof b - o, " %s=%llu", g_dpd_sname[k], (unsigned long long)v[k]);
    pthread_mutex_lock(&g_dpd_m);
    uint64_t seq = g_dpd_seq;
    pthread_mutex_unlock(&g_dpd_m);
    dpd_out(0, "what=dpool-sum sec=%llu pools=%u active=%u resets=%llu empty=%llu sets=%llu frees=%llu allocfail=%llu%s seq=%llu "
            "tot_resets=%llu tot_rwp=%llu tot_rwpfree=%llu tot_rwpdestroy=%llu tot_stale=%llu tot_unk=%llu",
            (unsigned long long)sec, live, active, (unsigned long long)resets, (unsigned long long)empty, (unsigned long long)sets,
            (unsigned long long)frees, (unsigned long long)fails, b, (unsigned long long)seq, (unsigned long long)g_dpd_t[DS_RESETS],
            (unsigned long long)g_dpd_t[DS_RWP], (unsigned long long)g_dpd_t[DS_RWPFREE], (unsigned long long)g_dpd_t[DS_RWPDESTROY],
            (unsigned long long)(g_dpd_t[DS_STALEBIND] + g_dpd_t[DS_STALESUB]), (unsigned long long)g_dpd_t[DS_UNK]);
    static uint64_t logged;
    uint64_t mark = g_dpd_t[DS_RWP] + g_dpd_t[DS_RWPFREE] + g_dpd_t[DS_RWPDESTROY] + g_dpd_t[DS_STALEBIND] + g_dpd_t[DS_STALESUB] +
                    g_dpd_t[DS_UNK];
    if (sec % 10 == 0 && mark != logged) {
        logged = mark;
        fprintf(stderr, "mali-compat: dpool-diag %llus: resets %llu (empty %llu), rwp %llu, rwpfree %llu, rwpdestroy %llu, stale bind %llu "
                "submit %llu, unknown %llu, submits %llu\n", (unsigned long long)sec, (unsigned long long)g_dpd_t[DS_RESETS],
                (unsigned long long)g_dpd_t[DS_EMPTY], (unsigned long long)g_dpd_t[DS_RWP], (unsigned long long)g_dpd_t[DS_RWPFREE],
                (unsigned long long)g_dpd_t[DS_RWPDESTROY], (unsigned long long)g_dpd_t[DS_STALEBIND],
                (unsigned long long)g_dpd_t[DS_STALESUB], (unsigned long long)g_dpd_t[DS_UNK], (unsigned long long)g_dpd_t[DS_SUBMITS]);
    }
}
static void *dpd_thread(void *arg) {
    (void)arg;
    for (uint64_t sec = 1;; sec++) {
        sleep(1);
        dpd_tick(sec);
    }
    return NULL;
}
static void dpd_init_once(void) {
    const char *e = getenv("MALI_COMPAT_DPOOL_DIAG");
    bool on = e && *e ? strcmp(e, "0") != 0 : g_addrmap;
    if (!on) return;
    g_dpd_tsv = g_addrmap && g_memlog;
    if (!g_dpd_tsv) {
        const char *p = getenv("MALI_COMPAT_DPOOL_LOG");
        if (p && *p) snprintf(g_dpd_path, sizeof g_dpd_path, "%s", p);
        else {
            char comm[32] = "unknown";
            FILE *cf = fopen("/proc/self/comm", "r");
            if (cf) { if (fgets(comm, sizeof comm, cf)) comm[strcspn(comm, "\n")] = 0; fclose(cf); }
            for (char *c = comm; *c; c++) if (*c == '/' || *c == ' ' || *c == '\\') *c = '_';
            snprintf(g_dpd_path, sizeof g_dpd_path, "/root/mali-dpool-%s.txt", comm);
        }
        g_dpd_fd = open(g_dpd_path, O_WRONLY | O_CREAT | O_APPEND | O_CLOEXEC, 0644);
        if (g_dpd_fd < 0) {
            fprintf(stderr, "mali-compat: dpool-diag OFF: cannot open %s: %s\n", g_dpd_path, strerror(errno));
            return;
        }
    } else snprintf(g_dpd_path, sizeof g_dpd_path, "memlog TSV (N events)");
    g_dpd_pool = calloc(DPD_POOLS, sizeof *g_dpd_pool);
    g_dpd_set = calloc(DPD_SETS, sizeof *g_dpd_set);
    g_dpd_pset = calloc(DPD_PSETS, sizeof *g_dpd_pset);
    g_dpd_ring = calloc(DPD_RING, sizeof *g_dpd_ring);
    if (!g_dpd_pool || !g_dpd_set || !g_dpd_pset || !g_dpd_ring) {
        fprintf(stderr, "mali-compat: dpool-diag OFF: out of memory\n");
        free(g_dpd_pool); free((void *)g_dpd_set); free(g_dpd_pset); free(g_dpd_ring);
        g_dpd_pool = NULL; g_dpd_set = NULL; g_dpd_pset = NULL; g_dpd_ring = NULL;
        if (g_dpd_fd >= 0) close(g_dpd_fd);
        g_dpd_fd = -1;
        return;
    }
    g_dpd_on = true;
    if (!g_dpd_tsv) dpd_out(0, "# mali-compat dpool-diag pid %d", (int)getpid());
    pthread_t t;
    if (pthread_create(&t, NULL, dpd_thread, NULL) == 0) pthread_detach(t);
    fprintf(stderr, "mali-compat: dpool-diag on (MALI_COMPAT_DPOOL_DIAG=%s%s): %s; per second what=dpool-sec per active pool + "
            "what=dpool-sum, immediate what=dpool-rwp / what=dpool-stale%s\n", e && *e ? e : "unset", g_addrmap ? ", ADDRMAP" : "",
            g_dpd_path, g_addrmap && g_dpd_tsv ? "; per-call dpool-reset lines off" : "");
}
static void dpd_init(void) {
    static pthread_once_t once = PTHREAD_ONCE_INIT;
    pthread_once(&once, dpd_init_once);
}
static void dpd_device_init(struct DeviceData *d, PFN_vkGetDeviceProcAddr gdpa, VkDevice dev) {
    d->dpd_fence_status = (PFN_vkGetFenceStatus)gdpa(dev, "vkGetFenceStatus");
    d->dpd_sem_value = (PFN_vkGetSemaphoreCounterValue)gdpa(dev, "vkGetSemaphoreCounterValue");
    if (!d->dpd_sem_value) d->dpd_sem_value = (PFN_vkGetSemaphoreCounterValue)gdpa(dev, "vkGetSemaphoreCounterValueKHR");
}
static void dpd_device_gone(struct DeviceData *d) { // before the driver's vkDestroyDevice: nothing of it is polled later
    pthread_mutex_lock(&g_dpd_m);
    for (uint32_t i = 0; i < DPD_RING; i++)
        if (g_dpd_ring[i].d == d) { g_dpd_ring[i].d = NULL; g_dpd_ring[i].done = true; g_dpd_ring[i].fence = g_dpd_ring[i].sem = 0; }
    for (uint32_t q = 0; q < g_dpd_nq; q++)
        if (g_dpd_qd[q] == d) { dpd_wm_max(q, g_dpd_seq); g_dpd_q[q] = VK_NULL_HANDLE; }
    pthread_mutex_unlock(&g_dpd_m);
    fprintf(stderr, "mali-compat: dpool-diag: end (VkDevice %p destroyed): totals so far resets %llu (empty %llu), rwp %llu, rwpfree %llu, "
            "rwpdestroy %llu, stale bind %llu submit %llu, unknown %llu, submits %llu (totals lag up to 1 s)\n", (void *)d->device,
            (unsigned long long)g_dpd_t[DS_RESETS], (unsigned long long)g_dpd_t[DS_EMPTY], (unsigned long long)g_dpd_t[DS_RWP],
            (unsigned long long)g_dpd_t[DS_RWPFREE], (unsigned long long)g_dpd_t[DS_RWPDESTROY],
            (unsigned long long)g_dpd_t[DS_STALEBIND], (unsigned long long)g_dpd_t[DS_STALESUB], (unsigned long long)g_dpd_t[DS_UNK],
            (unsigned long long)g_dpd_t[DS_SUBMITS]);
}


// SPDX-License-Identifier: GPL-3.0-or-later
// cl_diag_desccap.h: diagnostic MALI_COMPAT_DESC_CAPTURE (hook table dc_proc). Included by mali_compat_layer.c.
// -----------------------------------------------------------------------------
// desc-capture-1 [keeper, Claude Opus 5.5] 2026-11-01 (WD2 B10-19). Off unless MALI_COMPAT_DESC_CAPTURE=<cs hashes>
// (hex, comma separated: the FNV-1a hash of the compute shader's SPIR-V = the GPU_TIME "cs:" hash). For every
// vkCmdDispatch* recorded with one of those compute pipelines bound, one X row lists each written set-0 binding
// (bindings 0..DC_NB-1, array element 0) resolved down to buffer device address, view offset/range/format, buffer size
// and memory, as ids of B rows (identical binding tuples are written once). Object and descriptor state is shadowed
// from the create/bind/update/push/destroy entry points; D rows log the destroy/free/reset of every object that was
// ever part of a captured dispatch, W rows the first rewrite of a captured set, S rows the submit of a command buffer
// with captured dispatches, F rows the fence/timeline wait that saw it done (GPU_TIME wait hooks only).
// File MALI_COMPAT_DESC_CAPTURE_FILE (default /root/mali-desc-<pid>.tsv), own fd, whole lines only, flushed every
// 100 ms and at each captured submit, cap MALI_COMPAT_DESC_CAPTURE_MB (default 512) then a TRUNC row and stop.
// Times: CLOCK_REALTIME ms with us decimals (logcat alignment). Counters every 10 s on stderr ("mali-compat: desc:").
// -----------------------------------------------------------------------------
#define DC_HASH_MAX 16
#define DC_BUCKETS (1u << 16)
#define DC_DEDUP (1u << 18)
#define DC_BUF (256u << 10) // output buffer
static bool g_dc_on;
static uint32_t g_dc_hashes[DC_HASH_MAX], g_dc_hash_count;
static pthread_mutex_t g_dc_m = PTHREAD_MUTEX_INITIALIZER;   // object tables, sequence numbers
static pthread_mutex_t g_dc_fm = PTHREAD_MUTEX_INITIALIZER;  // output buffer
static int g_dc_fd = -1;
static char *g_dc_buf;
static size_t g_dc_len;
static uint64_t g_dc_bytes, g_dc_cap = 512ull << 20;
static bool g_dc_stopped;
static char g_dc_path[300];
static uint64_t g_dc_seq;      // dispatch / bind sequence (g_dc_m)
static uint64_t g_dc_wseq;     // descriptor write sequence (g_dc_m)
static uint32_t g_dc_next_bid; // binding tuple ids (g_dc_m)
struct DcDedup { uint64_t key; uint32_t id; };
static struct DcDedup *g_dc_dedup;
static uint32_t g_dc_dedup_n;

enum { DCC_UPD, DCC_UPD_W, DCC_UPD_C, DCC_UPD_TPL, DCC_BIND, DCC_BIND2, DCC_PUSH, DCC_PUSH2, DCC_PUSH_TPL, DCC_PUSH_TPL2,
       DCC_SET_ALLOC, DCC_SET_FREE, DCC_POOL_RESET, DCC_POOL_DESTROY, DCC_TPL, DCC_DSL, DCC_PL, DCC_BUF, DCC_BUF_BDA,
       DCC_BUF_BIND, DCC_VIEW, DCC_MEM, DCC_PIPE, DCC_DISP, DCC_DISP_ALL, DCC_SUBMIT, DCC_DONE, DCC_D, DCC_W, DCC_N };
static const char *const g_dcc_name[DCC_N] = {
    "upd", "upd-writes", "upd-copies", "upd-tpl", "bind", "bind2", "push", "push2", "push-tpl", "push-tpl2",
    "set-alloc", "set-free", "pool-reset", "pool-destroy", "tpl", "dsl", "pl", "buf", "buf-bda", "buf-bind", "view",
    "mem", "cap-pipes", "cap-disp", "disp", "cap-submits", "cap-done", "D", "W"};
static _Atomic uint64_t g_dcc[DCC_N];
#define DCC(i) atomic_fetch_add_explicit(&g_dcc[i], 1, memory_order_relaxed)

enum { DCK_BUF = 1, DCK_MEM, DCK_VIEW, DCK_TPL, DCK_DSL, DCK_PL, DCK_SET, DCK_POOL, DCK_PIPE };
struct DcTplEntry { uint32_t binding, elem, count, type; size_t offset, stride; };
struct DcObj {
    uint64_t h;
    struct DcObj *next;
    uint8_t kind;
    bool dead, cap;     // cap: part of a captured dispatch (destroy/free gets a D row)
    double t0, t_dead;  // create / destroy time (ms, realtime)
    union {
        struct { uint64_t size, usage, mem, memoff, va; uint8_t va_kind; double t_bind; } buf; // va_kind 0 none, 1 BDA, 2 memory capture address + offset
        struct { uint64_t size, mva; uint32_t type; } mem;
        struct { uint64_t buffer, offset, range; uint32_t format; } view;
        struct { struct DcTplEntry *e; uint32_t n, type, bp, set; uint64_t layout; } tpl;
        struct { uint8_t type[DC_NB]; uint32_t count[DC_NB]; uint16_t present; uint32_t ndyn; } dsl;
        struct { uint64_t sl[8]; uint32_t n; } pl;
        struct { uint64_t pool, layout, last_cap_seq, wseq; struct DcDesc *b; bool used_cap; } set;
        struct { uint64_t *sets; uint32_t n, cap; } pool;
        struct { uint32_t hash; uint64_t layout; } pipe;
    } u;
};
static struct DcObj **g_dc_tab;

static double dc_now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1e6;
}
static uint32_t dc_bucket(uint8_t kind, uint64_t h) {
    uint64_t v = h * 0x9E3779B97F4A7C15ull ^ kind;
    return (uint32_t)((v ^ (v >> 29) ^ (v >> 47)) & (DC_BUCKETS - 1));
}
// g_dc_m held. create: a new entry when missing (zeroed apart from h/kind/t0).
static struct DcObj *dc_get(uint8_t kind, uint64_t h, bool create) {
    if (!g_dc_tab || !h) return NULL;
    uint32_t b = dc_bucket(kind, h);
    for (struct DcObj *o = g_dc_tab[b]; o; o = o->next)
        if (o->h == h && o->kind == kind) return o;
    if (!create) return NULL;
    struct DcObj *o = calloc(1, sizeof *o);
    if (!o) return NULL;
    o->h = h; o->kind = kind; o->t0 = dc_now();
    o->next = g_dc_tab[b];
    g_dc_tab[b] = o;
    return o;
}
// A create that reuses a handle: drop the old state (keep the node).
static struct DcObj *dc_fresh(uint8_t kind, uint64_t h) {
    struct DcObj *o = dc_get(kind, h, true);
    if (!o) return NULL;
    if (kind == DCK_TPL) free(o->u.tpl.e);
    if (kind == DCK_POOL) free(o->u.pool.sets);
    struct DcDesc *keep = kind == DCK_SET ? o->u.set.b : NULL;
    struct DcObj *next = o->next;
    memset(o, 0, sizeof *o);
    o->h = h; o->kind = kind; o->t0 = dc_now(); o->next = next;
    if (kind == DCK_SET) { o->u.set.b = keep ? keep : calloc(DC_NB, sizeof *keep); if (keep) memset(keep, 0, DC_NB * sizeof *keep); }
    return o;
}

// ---- output
static void dc_flush_locked(void) {
    size_t off = 0;
    while (g_dc_fd >= 0 && off < g_dc_len) {
        ssize_t w = write(g_dc_fd, g_dc_buf + off, g_dc_len - off);
        if (w < 0 && errno == EINTR) continue;
        if (w <= 0) break;
        off += (size_t)w;
    }
    g_dc_len = 0;
}
static void dc_out(const char *line, size_t n) {
    pthread_mutex_lock(&g_dc_fm);
    if (!g_dc_stopped && g_dc_buf && n < DC_BUF) {
        if (g_dc_bytes + n > g_dc_cap) {
            char t[96];
            int k = snprintf(t, sizeof t, "TRUNC\t%.3f\t%llu\n", dc_now(), (unsigned long long)g_dc_bytes);
            if (g_dc_len + (size_t)k > DC_BUF) dc_flush_locked();
            memcpy(g_dc_buf + g_dc_len, t, (size_t)k);
            g_dc_len += (size_t)k;
            dc_flush_locked();
            g_dc_stopped = true;
            close(g_dc_fd);
            g_dc_fd = -1;
        } else {
            if (g_dc_len + n > DC_BUF) dc_flush_locked();
            memcpy(g_dc_buf + g_dc_len, line, n);
            g_dc_len += n;
            g_dc_bytes += n;
        }
    }
    pthread_mutex_unlock(&g_dc_fm);
}
static void dc_printf(const char *fmt, ...) __attribute__((format(printf, 1, 2)));
static void dc_printf(const char *fmt, ...) {
    char line[1024];
    va_list ap;
    va_start(ap, fmt);
    int n = vsnprintf(line, sizeof line, fmt, ap);
    va_end(ap);
    if (n > 0) dc_out(line, (size_t)n < sizeof line ? (size_t)n : sizeof line - 1);
}
static void dc_flush(void) {
    pthread_mutex_lock(&g_dc_fm);
    if (g_dc_buf) dc_flush_locked();
    pthread_mutex_unlock(&g_dc_fm);
}
static void *dc_thread(void *arg) {
    (void)arg;
    uint64_t last[DCC_N] = {0};
    for (uint32_t tick = 1;; tick++) {
        usleep(100000);
        dc_flush();
        if (tick % 100) continue;
        uint64_t v[DCC_N];
        bool any = false, changed = false;
        for (int i = 0; i < DCC_N; i++) { v[i] = atomic_load(&g_dcc[i]); any |= v[i] != 0; changed |= v[i] != last[i]; }
        if (!any || !changed) continue;
        memcpy(last, v, sizeof last);
        char s[1200];
        size_t k = (size_t)snprintf(s, sizeof s, "mali-compat: desc:");
        for (int i = 0; i < DCC_N && k < sizeof s; i++)
            k += (size_t)snprintf(s + k, sizeof s - k, " %s %llu%s", g_dcc_name[i], (unsigned long long)v[i], i + 1 < DCC_N ? "," : "");
        pthread_mutex_lock(&g_dc_fm);
        uint64_t bytes = g_dc_bytes;
        bool trunc = g_dc_stopped;
        pthread_mutex_unlock(&g_dc_fm);
        if (k < sizeof s) snprintf(s + k, sizeof s - k, ", file %llu KiB%s", (unsigned long long)(bytes >> 10), trunc ? " TRUNC" : "");
        fprintf(stderr, "%s\n", s);
    }
    return NULL;
}

// Once per process (vkCreateInstance and vkCreateDevice both call it): parse the env, open the file.
static pthread_once_t g_dc_once = PTHREAD_ONCE_INIT;
static void dc_init_impl(void) {
    const char *h = getenv("MALI_COMPAT_DESC_CAPTURE");
    for (; h && *h && g_dc_hash_count < DC_HASH_MAX;) {
        char *end;
        unsigned long v = strtoul(h, &end, 16);
        if (end == h) break;
        g_dc_hashes[g_dc_hash_count++] = (uint32_t)v;
        h = *end == ',' ? end + 1 : end;
    }
    if (!g_dc_hash_count) return;
    const char *mb = getenv("MALI_COMPAT_DESC_CAPTURE_MB");
    if (mb && strtoull(mb, NULL, 10)) g_dc_cap = strtoull(mb, NULL, 10) << 20;
    const char *f = getenv("MALI_COMPAT_DESC_CAPTURE_FILE");
    if (f && *f) snprintf(g_dc_path, sizeof g_dc_path, "%s", f);
    else snprintf(g_dc_path, sizeof g_dc_path, "/root/mali-desc-%d.tsv", (int)getpid());
    g_dc_tab = calloc(DC_BUCKETS, sizeof *g_dc_tab);
    g_dc_dedup = calloc(DC_DEDUP, sizeof *g_dc_dedup);
    g_dc_buf = malloc(DC_BUF);
    g_dc_fd = g_dc_tab && g_dc_dedup && g_dc_buf ? open(g_dc_path, O_WRONLY | O_CREAT | O_APPEND | O_CLOEXEC, 0644) : -1;
    if (g_dc_fd < 0) {
        fprintf(stderr, "mali-compat: desc-capture-1: cannot open %s: %s (capture off)\n", g_dc_path, strerror(errno));
        return;
    }
    g_dc_on = true;
    char comm[32] = "unknown";
    FILE *cf = fopen("/proc/self/comm", "r");
    if (cf) { if (fgets(comm, sizeof comm, cf)) comm[strcspn(comm, "\n")] = 0; fclose(cf); }
    time_t now = time(NULL);
    struct tm tm;
    localtime_r(&now, &tm);
    char hs[DC_HASH_MAX * 9 + 1] = "";
    for (uint32_t i = 0; i < g_dc_hash_count; i++)
        snprintf(hs + strlen(hs), sizeof hs - strlen(hs), "%s%08x", i ? "," : "", g_dc_hashes[i]);
    dc_printf("# mali-compat desc-capture-1 pid %d comm %s start_ms %.3f tz_min %ld hashes %s cap %llu MiB nb %d\n",
              (int)getpid(), comm, dc_now(), (long)(tm.tm_gmtoff / 60), hs, (unsigned long long)(g_dc_cap >> 20), DC_NB);
    dc_printf("# P t_ms pipeline hash layout | X t_ms seq cmdbuf hash gx gy gz src(set|push|none) set set_flags bindings(b=id ...)\n"
              "# B id binding type handle buffer va off range lo hi format bufsize mem memsize memoff buf_t mem_t flags\n"
              "# S t_ms queue cmdbuf first_seq last_seq ndisp fence sem:value | F t_ms fence|sem handle value first_seq last_seq\n"
              "# D t_ms kind handle info | W t_ms set last_cap_seq\n");
    atexit(dc_flush);
    pthread_t t;
    if (pthread_create(&t, NULL, dc_thread, NULL) == 0) pthread_detach(t);
}
static void dc_init(void) { pthread_once(&g_dc_once, dc_init_impl); }

static bool dc_hash_wanted(uint32_t hash) {
    for (uint32_t i = 0; i < g_dc_hash_count; i++)
        if (g_dc_hashes[i] == hash) return true;
    return false;
}

// ---- object tables
static void dc_buf_create(struct DeviceData *d, VkBuffer b, const VkBufferCreateInfo *ci) {
    (void)d;
    DCC(DCC_BUF);
    if (ci->usage & VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT) DCC(DCC_BUF_BDA);
    pthread_mutex_lock(&g_dc_m);
    struct DcObj *o = dc_fresh(DCK_BUF, (uint64_t)b);
    if (o) { o->u.buf.size = ci->size; o->u.buf.usage = ci->usage; }
    pthread_mutex_unlock(&g_dc_m);
}
static void dc_buf_bind(struct DeviceData *d, VkDevice dev, VkBuffer b, VkDeviceMemory m, VkDeviceSize off) {
    DCC(DCC_BUF_BIND);
    uint64_t va = 0;
    uint8_t kind = 0;
    pthread_mutex_lock(&g_dc_m);
    struct DcObj *o = dc_get(DCK_BUF, (uint64_t)b, false);
    bool bda = o && (o->u.buf.usage & VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);
    struct DcObj *mo = dc_get(DCK_MEM, (uint64_t)m, false);
    uint64_t mva = mo ? mo->u.mem.mva : 0;
    pthread_mutex_unlock(&g_dc_m);
    if (bda && d->fpGetBufferDeviceAddress) {
        VkBufferDeviceAddressInfo bi = {VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO, NULL, b};
        va = d->fpGetBufferDeviceAddress(dev, &bi);
        kind = va ? 1 : 0;
    }
    if (!va && mva) { va = mva + off; kind = 2; }
    pthread_mutex_lock(&g_dc_m);
    if ((o = dc_get(DCK_BUF, (uint64_t)b, false))) {
        o->u.buf.mem = (uint64_t)m; o->u.buf.memoff = off; o->u.buf.va = va; o->u.buf.va_kind = kind; o->u.buf.t_bind = dc_now();
    }
    pthread_mutex_unlock(&g_dc_m);
}
// The capture address only for allocations with DEVICE_ADDRESS_BIT (valid usage); buffers without BDA usage get
// VA = that address + bind offset (flag VAM).
static bool dc_mem_devaddr(const void *chain) {
    const VkMemoryAllocateFlagsInfo *f = find_struct(chain, VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO);
    return f && (f->flags & VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT);
}
static void dc_mem_alloc(struct DeviceData *d, VkDevice dev, VkDeviceMemory m, VkDeviceSize size, uint32_t type, bool devaddr) {
    DCC(DCC_MEM);
    uint64_t mva = 0;
    if (devaddr && d->fpGetDeviceMemoryOpaqueCaptureAddress) {
        VkDeviceMemoryOpaqueCaptureAddressInfo ci = {VK_STRUCTURE_TYPE_DEVICE_MEMORY_OPAQUE_CAPTURE_ADDRESS_INFO, NULL, m};
        mva = d->fpGetDeviceMemoryOpaqueCaptureAddress(dev, &ci);
    }
    pthread_mutex_lock(&g_dc_m);
    struct DcObj *o = dc_fresh(DCK_MEM, (uint64_t)m);
    if (o) { o->u.mem.size = size; o->u.mem.type = type; o->u.mem.mva = mva; }
    pthread_mutex_unlock(&g_dc_m);
}
// Destroy/free of a buffer, memory or view: D row when it was part of a captured dispatch.
static void dc_dead(uint8_t kind, uint64_t h) {
    if (!h) return;
    pthread_mutex_lock(&g_dc_m);
    struct DcObj *o = dc_get(kind, h, false);
    bool log = o && o->cap && !o->dead;
    if (o) { o->dead = true; o->t_dead = dc_now(); }
    double t = o ? o->t_dead : 0;
    pthread_mutex_unlock(&g_dc_m);
    if (log) {
        DCC(DCC_D);
        dc_printf("D\t%.3f\t%s\t0x%llx\t-\n", t, kind == DCK_BUF ? "buf" : kind == DCK_MEM ? "mem" : "view", (unsigned long long)h);
    }
}
static void dc_view_create(VkBufferView v, const VkBufferViewCreateInfo *ci) {
    DCC(DCC_VIEW);
    pthread_mutex_lock(&g_dc_m);
    struct DcObj *o = dc_fresh(DCK_VIEW, (uint64_t)v);
    if (o) { o->u.view.buffer = (uint64_t)ci->buffer; o->u.view.offset = ci->offset; o->u.view.range = ci->range; o->u.view.format = ci->format; }
    pthread_mutex_unlock(&g_dc_m);
}
static void dc_pipe_create(VkPipeline p, uint32_t hash, VkPipelineLayout layout) {
    bool want = dc_hash_wanted(hash);
    pthread_mutex_lock(&g_dc_m);
    struct DcObj *o = want ? dc_fresh(DCK_PIPE, (uint64_t)p) : dc_get(DCK_PIPE, (uint64_t)p, false);
    if (o && want) { o->u.pipe.hash = hash; o->u.pipe.layout = (uint64_t)layout; }
    else if (o) o->dead = true; // handle reused by a pipeline that is not captured
    pthread_mutex_unlock(&g_dc_m);
    if (want) {
        DCC(DCC_PIPE);
        dc_printf("P\t%.3f\t0x%llx\t%08x\t0x%llx\n", dc_now(), (unsigned long long)(uint64_t)p, hash, (unsigned long long)(uint64_t)layout);
    }
}

// ---- descriptor writes into a shadow (set or push), element 0 of bindings < DC_NB. g_dc_m held.
static bool dc_is_dyn(uint32_t t) {
    return t == VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC || t == VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC;
}
static bool dc_is_buf(uint32_t t) {
    return t == VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER || t == VK_DESCRIPTOR_TYPE_STORAGE_BUFFER || dc_is_dyn(t);
}
static bool dc_is_texel(uint32_t t) {
    return t == VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER || t == VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER;
}
static bool dc_is_img(uint32_t t) {
    return t == VK_DESCRIPTOR_TYPE_SAMPLER || t == VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER ||
           t == VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE || t == VK_DESCRIPTOR_TYPE_STORAGE_IMAGE || t == VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT;
}
static uint32_t dc_bcount(const struct DcObj *dsl, uint32_t b) {
    if (dsl && b < DC_NB && (dsl->u.dsl.present >> b & 1)) return dsl->u.dsl.count[b] ? dsl->u.dsl.count[b] : 1;
    return 1;
}
// count descriptors from base (stride apart) into (binding, elem) on, rolling over to the next binding (spec rule).
static void dc_put(struct DcDesc *sh, const struct DcObj *dsl, uint32_t binding, uint32_t elem, uint32_t count,
                   uint32_t type, const uint8_t *base, size_t stride) {
    if (!sh || !base || type == VK_DESCRIPTOR_TYPE_INLINE_UNIFORM_BLOCK) return;
    for (uint32_t i = 0; i < count && binding < DC_NB; i++, elem++) {
        while (binding < DC_NB && elem >= dc_bcount(dsl, binding)) { elem -= dc_bcount(dsl, binding); binding++; }
        if (binding >= DC_NB) break;
        if (elem) continue;
        const uint8_t *p = base + i * stride;
        struct DcDesc *d = &sh[binding];
        memset(d, 0, sizeof *d);
        d->type = type + 1;
        if (dc_is_buf(type)) {
            const VkDescriptorBufferInfo *bi = (const void *)p;
            d->h = (uint64_t)bi->buffer; d->off = bi->offset; d->range = bi->range;
        } else if (dc_is_texel(type)) {
            d->h = (uint64_t)*(const VkBufferView *)p;
        } else if (dc_is_img(type)) {
            const VkDescriptorImageInfo *ii = (const void *)p;
            d->h = type == VK_DESCRIPTOR_TYPE_SAMPLER ? (uint64_t)ii->sampler : (uint64_t)ii->imageView;
        } else {
            d->h = *(const uint64_t *)p; // acceleration structure etc.: handle only
        }
    }
}
static void dc_put_write(struct DcDesc *sh, const struct DcObj *dsl, const VkWriteDescriptorSet *w) {
    const void *base = dc_is_buf(w->descriptorType) ? (const void *)w->pBufferInfo
                     : dc_is_texel(w->descriptorType) ? (const void *)w->pTexelBufferView
                     : dc_is_img(w->descriptorType) ? (const void *)w->pImageInfo : NULL;
    size_t stride = dc_is_buf(w->descriptorType) ? sizeof(VkDescriptorBufferInfo)
                  : dc_is_texel(w->descriptorType) ? sizeof(VkBufferView) : sizeof(VkDescriptorImageInfo);
    dc_put(sh, dsl, w->dstBinding, w->dstArrayElement, w->descriptorCount, w->descriptorType, base, stride);
}
static void dc_put_tpl(struct DcDesc *sh, const struct DcObj *dsl, const struct DcObj *tpl, const void *data) {
    for (uint32_t i = 0; tpl && data && i < tpl->u.tpl.n; i++) {
        const struct DcTplEntry *e = &tpl->u.tpl.e[i];
        dc_put(sh, dsl, e->binding, e->elem, e->count, e->type, (const uint8_t *)data + e->offset, e->stride);
    }
}
// Set about to be written: W row on the first write after a captured use. g_dc_m held; returns true for a W row.
static bool dc_set_touch(struct DcObj *s, double *t, uint64_t *seq) {
    s->u.set.wseq = ++g_dc_wseq;
    if (!s->u.set.used_cap) return false;
    s->u.set.used_cap = false;
    *t = dc_now();
    *seq = s->u.set.last_cap_seq;
    return true;
}
static const struct DcObj *dc_set_dsl(const struct DcObj *s) {
    return s ? dc_get(DCK_DSL, s->u.set.layout, false) : NULL;
}
static void dc_update_sets(uint32_t nw, const VkWriteDescriptorSet *w, uint32_t nc, const VkCopyDescriptorSet *c) {
    DCC(DCC_UPD);
    uint64_t wset[8], wseq[8];
    double wt[8];
    uint32_t nW = 0;
    pthread_mutex_lock(&g_dc_m);
    for (uint32_t i = 0; i < nw; i++) {
        DCC(DCC_UPD_W);
        struct DcObj *s = dc_get(DCK_SET, (uint64_t)w[i].dstSet, false);
        if (!s || !s->u.set.b) continue;
        if (dc_set_touch(s, &wt[nW < 8 ? nW : 7], &wseq[nW < 8 ? nW : 7]) && nW < 8) wset[nW++] = s->h;
        dc_put_write(s->u.set.b, dc_set_dsl(s), &w[i]);
    }
    for (uint32_t i = 0; i < nc; i++) {
        DCC(DCC_UPD_C);
        struct DcObj *src = dc_get(DCK_SET, (uint64_t)c[i].srcSet, false), *dst = dc_get(DCK_SET, (uint64_t)c[i].dstSet, false);
        if (!src || !dst || !src->u.set.b || !dst->u.set.b) continue;
        if (dc_set_touch(dst, &wt[nW < 8 ? nW : 7], &wseq[nW < 8 ? nW : 7]) && nW < 8) wset[nW++] = dst->h;
        for (uint32_t k = 0; k < c[i].descriptorCount; k++) {
            uint32_t sb = c[i].srcBinding + k, db = c[i].dstBinding + k; // element 0, one per binding (approximation)
            if (c[i].srcArrayElement || c[i].dstArrayElement || sb >= DC_NB || db >= DC_NB) break;
            dst->u.set.b[db] = src->u.set.b[sb];
        }
    }
    pthread_mutex_unlock(&g_dc_m);
    for (uint32_t i = 0; i < nW; i++) {
        DCC(DCC_W);
        dc_printf("W\t%.3f\t0x%llx\t%llu\n", wt[i], (unsigned long long)wset[i], (unsigned long long)wseq[i]);
    }
}
static void dc_update_tpl(VkDescriptorSet set, VkDescriptorUpdateTemplate tpl, const void *data) {
    DCC(DCC_UPD_TPL);
    double t = 0;
    uint64_t seq = 0;
    bool w = false;
    pthread_mutex_lock(&g_dc_m);
    struct DcObj *s = dc_get(DCK_SET, (uint64_t)set, false), *to = dc_get(DCK_TPL, (uint64_t)tpl, false);
    if (s && s->u.set.b) {
        w = dc_set_touch(s, &t, &seq);
        dc_put_tpl(s->u.set.b, dc_set_dsl(s), to, data);
    }
    pthread_mutex_unlock(&g_dc_m);
    if (w) {
        DCC(DCC_W);
        dc_printf("W\t%.3f\t0x%llx\t%llu\n", t, (unsigned long long)(uint64_t)set, (unsigned long long)seq);
    }
}
static void dc_tpl_create(VkDescriptorUpdateTemplate t, const VkDescriptorUpdateTemplateCreateInfo *ci) {
    DCC(DCC_TPL);
    struct DcTplEntry *e = calloc(ci->descriptorUpdateEntryCount ? ci->descriptorUpdateEntryCount : 1, sizeof *e);
    for (uint32_t i = 0; e && i < ci->descriptorUpdateEntryCount; i++) {
        const VkDescriptorUpdateTemplateEntry *s = &ci->pDescriptorUpdateEntries[i];
        e[i] = (struct DcTplEntry){s->dstBinding, s->dstArrayElement, s->descriptorCount, s->descriptorType, s->offset, s->stride};
    }
    pthread_mutex_lock(&g_dc_m);
    struct DcObj *o = dc_fresh(DCK_TPL, (uint64_t)t);
    if (o) {
        o->u.tpl.e = e; o->u.tpl.n = e ? ci->descriptorUpdateEntryCount : 0; o->u.tpl.type = ci->templateType;
        o->u.tpl.bp = ci->pipelineBindPoint; o->u.tpl.set = ci->set; o->u.tpl.layout = (uint64_t)ci->pipelineLayout;
    } else free(e);
    pthread_mutex_unlock(&g_dc_m);
}
static void dc_tpl_destroy(VkDescriptorUpdateTemplate t) {
    pthread_mutex_lock(&g_dc_m);
    struct DcObj *o = dc_get(DCK_TPL, (uint64_t)t, false);
    if (o) { free(o->u.tpl.e); o->u.tpl.e = NULL; o->u.tpl.n = 0; o->dead = true; }
    pthread_mutex_unlock(&g_dc_m);
}
static void dc_dsl_create(VkDescriptorSetLayout l, const VkDescriptorSetLayoutCreateInfo *ci) {
    DCC(DCC_DSL);
    pthread_mutex_lock(&g_dc_m);
    struct DcObj *o = dc_fresh(DCK_DSL, (uint64_t)l);
    for (uint32_t i = 0; o && i < ci->bindingCount; i++) {
        const VkDescriptorSetLayoutBinding *b = &ci->pBindings[i];
        if (dc_is_dyn(b->descriptorType)) o->u.dsl.ndyn += b->descriptorCount;
        if (b->binding < DC_NB) {
            o->u.dsl.type[b->binding] = (uint8_t)b->descriptorType;
            o->u.dsl.count[b->binding] = b->descriptorCount;
            o->u.dsl.present |= (uint16_t)(1u << b->binding);
        }
    }
    pthread_mutex_unlock(&g_dc_m);
}
static void dc_pl_create(VkPipelineLayout l, const VkPipelineLayoutCreateInfo *ci) {
    DCC(DCC_PL);
    pthread_mutex_lock(&g_dc_m);
    struct DcObj *o = dc_fresh(DCK_PL, (uint64_t)l);
    for (uint32_t i = 0; o && i < ci->setLayoutCount && i < 8; i++) o->u.pl.sl[o->u.pl.n++] = (uint64_t)ci->pSetLayouts[i];
    pthread_mutex_unlock(&g_dc_m);
}
static void dc_sets_alloc(const VkDescriptorSetAllocateInfo *ai, const VkDescriptorSet *sets) {
    pthread_mutex_lock(&g_dc_m);
    for (uint32_t i = 0; i < ai->descriptorSetCount; i++) {
        DCC(DCC_SET_ALLOC);
        struct DcObj *o = dc_fresh(DCK_SET, (uint64_t)sets[i]);
        if (o) { o->u.set.pool = (uint64_t)ai->descriptorPool; o->u.set.layout = (uint64_t)ai->pSetLayouts[i]; }
    }
    pthread_mutex_unlock(&g_dc_m);
}
// Free / pool reset / pool destroy: D rows for the captured sets. what: "set-free", "set-reset", "set-pooldestroy".
static void dc_sets_dead(VkDescriptorPool pool, uint32_t n, const VkDescriptorSet *sets, const char *what) {
    uint64_t logged[64];
    uint32_t nl = 0, more = 0;
    double t = dc_now();
    pthread_mutex_lock(&g_dc_m);
    if (sets) {
        for (uint32_t i = 0; i < n; i++) {
            DCC(DCC_SET_FREE);
            struct DcObj *s = dc_get(DCK_SET, (uint64_t)sets[i], false);
            if (!s || s->dead) continue;
            s->dead = true; s->t_dead = t;
            if (s->cap) { if (nl < 64) logged[nl++] = s->h; else more++; }
        }
    } else {
        struct DcObj *p = dc_get(DCK_POOL, (uint64_t)pool, false);
        for (uint32_t i = 0; p && i < p->u.pool.n; i++) {
            struct DcObj *s = dc_get(DCK_SET, p->u.pool.sets[i], false);
            if (!s || s->dead || s->u.set.pool != (uint64_t)pool) continue;
            s->dead = true; s->t_dead = t;
            if (nl < 64) logged[nl++] = s->h; else more++;
        }
        if (p) p->u.pool.n = 0;
    }
    pthread_mutex_unlock(&g_dc_m);
    for (uint32_t i = 0; i < nl; i++) {
        DCC(DCC_D);
        dc_printf("D\t%.3f\t%s\t0x%llx\tpool 0x%llx\n", t, what, (unsigned long long)logged[i], (unsigned long long)(uint64_t)pool);
    }
    if (more) dc_printf("D\t%.3f\t%s\t-\t%u more in pool 0x%llx\n", t, what, more, (unsigned long long)(uint64_t)pool);
}

// ---- command buffer state (compute bind point, set 0)
static const struct DcObj *dc_pl_dsl(uint64_t layout, uint32_t set) { // g_dc_m held
    const struct DcObj *pl = dc_get(DCK_PL, layout, false);
    return pl && set < pl->u.pl.n ? dc_get(DCK_DSL, pl->u.pl.sl[set], false) : NULL;
}
static void dc_bind_sets(struct CmdBufData *c, uint32_t bp, VkPipelineLayout layout, uint32_t first, uint32_t n,
                         const VkDescriptorSet *sets, uint32_t ndyn, const uint32_t *dyn, bool v2) {
    DCC(v2 ? DCC_BIND2 : DCC_BIND);
    if (!c || bp != VK_PIPELINE_BIND_POINT_COMPUTE || first != 0 || !n || !sets) return;
    pthread_mutex_lock(&g_dc_m);
    c->dc.kind = 1;
    c->dc.set0 = (uint64_t)sets[0];
    c->dc.bind_seq = g_dc_wseq;
    memset(c->dc.dyn, 0, sizeof c->dc.dyn);
    c->dc.dyn_mask = 0;
    const struct DcObj *dsl = dc_pl_dsl((uint64_t)layout, 0);
    struct DcObj *s = dc_get(DCK_SET, (uint64_t)sets[0], false);
    if (!dsl) dsl = dc_set_dsl(s);
    uint32_t k = 0;
    for (uint32_t b = 0; b < DC_NB && k < ndyn && dyn; b++) {
        bool isdyn = dsl ? (dsl->u.dsl.present >> b & 1) && dc_is_dyn(dsl->u.dsl.type[b])
                         : s && s->u.set.b && s->u.set.b[b].type && dc_is_dyn(s->u.set.b[b].type - 1);
        if (!isdyn) continue;
        c->dc.dyn[b] = dyn[k];
        c->dc.dyn_mask |= (uint16_t)(1u << b);
        k += dsl ? dc_bcount(dsl, b) : 1;
    }
    pthread_mutex_unlock(&g_dc_m);
}
static void dc_push_writes(struct CmdBufData *c, uint32_t bp, VkPipelineLayout layout, uint32_t set, uint32_t n,
                           const VkWriteDescriptorSet *w, bool v2) {
    DCC(v2 ? DCC_PUSH2 : DCC_PUSH);
    if (!c || bp != VK_PIPELINE_BIND_POINT_COMPUTE || set != 0) return;
    pthread_mutex_lock(&g_dc_m);
    if (c->dc.kind != 2) memset(c->dc.push, 0, sizeof c->dc.push);
    c->dc.kind = 2;
    c->dc.bind_seq = g_dc_wseq;
    c->dc.dyn_mask = 0;
    const struct DcObj *dsl = dc_pl_dsl((uint64_t)layout, 0);
    for (uint32_t i = 0; i < n; i++) dc_put_write(c->dc.push, dsl, &w[i]);
    pthread_mutex_unlock(&g_dc_m);
}
static void dc_push_tpl(struct CmdBufData *c, VkDescriptorUpdateTemplate tpl, VkPipelineLayout layout, uint32_t set,
                        const void *data, bool v2) {
    DCC(v2 ? DCC_PUSH_TPL2 : DCC_PUSH_TPL);
    if (!c || set != 0) return;
    pthread_mutex_lock(&g_dc_m);
    const struct DcObj *to = dc_get(DCK_TPL, (uint64_t)tpl, false);
    if (to && to->u.tpl.bp == VK_PIPELINE_BIND_POINT_COMPUTE) {
        if (c->dc.kind != 2) memset(c->dc.push, 0, sizeof c->dc.push);
        c->dc.kind = 2;
        c->dc.bind_seq = g_dc_wseq;
        c->dc.dyn_mask = 0;
        dc_put_tpl(c->dc.push, dc_pl_dsl((uint64_t)layout, 0), to, data);
    }
    pthread_mutex_unlock(&g_dc_m);
}

// ---- dispatch: one X row (+ B rows for new binding tuples)
#define DCF_RANGE_GT_BUF 0x1
#define DCF_DEAD 0x2
#define DCF_VA0 0x4
#define DCF_SETWRITE_AFTER_BIND 0x8
#define DCF_UNKNOWN 0x10
#define DCF_VA_FROM_MEM 0x20
#define DCF_WHOLE 0x40
#define DCF_NO_MEM 0x80
static void dc_flags_str(uint32_t f, char *s, size_t n) {
    static const char *const names[] = {"RANGE>BUF", "DEAD", "VA0", "SETWRITE-AFTER-BIND", "UNK", "VAM", "WHOLE", "NOMEM"};
    size_t k = 0;
    s[0] = 0;
    for (int i = 0; i < 8; i++)
        if (f >> i & 1) k += (size_t)snprintf(s + k, n > k ? n - k : 0, "%s%s", k ? "," : "", names[i]);
    if (!k) snprintf(s, n, "-");
}
struct DcTuple { // memset before filling: hashed as bytes
    uint64_t handle, buffer, va, off, range, bufsize, mem, memsize, memoff;
    double buf_t, mem_t;
    uint32_t binding, type, format, flags;
};
static uint64_t dc_fnv(const void *p, size_t n) {
    uint64_t h = 1469598103934665603ull;
    for (size_t i = 0; i < n; i++) { h ^= ((const uint8_t *)p)[i]; h *= 1099511628211ull; }
    return h ? h : 1;
}
// g_dc_m held. Returns the id; *fresh = a B row is due.
static uint32_t dc_tuple_id(const struct DcTuple *t, bool *fresh) {
    uint64_t key = dc_fnv(t, sizeof *t);
    uint32_t i = (uint32_t)key & (DC_DEDUP - 1);
    if (g_dc_dedup_n < DC_DEDUP / 4 * 3) {
        while (g_dc_dedup[i].key) {
            if (g_dc_dedup[i].key == key) { *fresh = false; return g_dc_dedup[i].id; }
            i = (i + 1) & (DC_DEDUP - 1);
        }
        g_dc_dedup[i].key = key;
        g_dc_dedup[i].id = ++g_dc_next_bid;
        g_dc_dedup_n++;
        *fresh = true;
        return g_dc_next_bid;
    }
    *fresh = true; // table full: every tuple gets a new id and row
    return ++g_dc_next_bid;
}
// g_dc_m held. Marks the object captured (D rows later).
static void dc_mark(struct DcObj *o) { if (o) o->cap = true; }
static void dc_resolve(const struct DcDesc *d, uint32_t binding, uint32_t dynoff, bool dyn_valid, bool swab, struct DcTuple *t) {
    memset(t, 0, sizeof *t);
    uint32_t type = d->type - 1;
    t->binding = binding; t->type = type; t->handle = d->h;
    if (swab) t->flags |= DCF_SETWRITE_AFTER_BIND;
    if (dc_is_img(type) || (!dc_is_buf(type) && !dc_is_texel(type))) return;
    uint64_t buf = d->h, off = d->off, range = d->range;
    if (dc_is_texel(type)) {
        struct DcObj *v = dc_get(DCK_VIEW, d->h, false);
        if (!v) { t->flags |= DCF_UNKNOWN | DCF_VA0; return; }
        dc_mark(v);
        if (v->dead) t->flags |= DCF_DEAD;
        buf = v->u.view.buffer; off = v->u.view.offset; range = v->u.view.range; t->format = v->u.view.format;
    } else if (dc_is_dyn(type) && dyn_valid) {
        off += dynoff;
    }
    t->buffer = buf; t->off = off;
    struct DcObj *b = dc_get(DCK_BUF, buf, false);
    if (!b) { t->range = range; t->flags |= DCF_UNKNOWN | DCF_VA0; return; }
    dc_mark(b);
    if (b->dead) t->flags |= DCF_DEAD;
    t->bufsize = b->u.buf.size; t->buf_t = b->t0;
    if (range == VK_WHOLE_SIZE) { range = b->u.buf.size > off ? b->u.buf.size - off : 0; t->flags |= DCF_WHOLE; }
    t->range = range;
    if (off + range > b->u.buf.size) t->flags |= DCF_RANGE_GT_BUF;
    t->va = b->u.buf.va;
    if (!t->va) t->flags |= DCF_VA0;
    if (b->u.buf.va_kind == 2) t->flags |= DCF_VA_FROM_MEM;
    t->mem = b->u.buf.mem; t->memoff = b->u.buf.memoff;
    struct DcObj *m = dc_get(DCK_MEM, b->u.buf.mem, false);
    if (!m) { t->flags |= DCF_NO_MEM; return; }
    dc_mark(m);
    if (m->dead) t->flags |= DCF_DEAD;
    t->memsize = m->u.mem.size; t->mem_t = m->t0;
}
static void dc_dispatch(struct CmdBufData *c, VkCommandBuffer cmd, uint32_t x, uint32_t y, uint32_t z) {
    DCC(DCC_DISP_ALL);
    if (!c || !c->compute_pipeline) return;
    char row[1024], brows[DC_NB][300];
    uint32_t nb = 0;
    pthread_mutex_lock(&g_dc_m);
    struct DcObj *p = dc_get(DCK_PIPE, (uint64_t)c->compute_pipeline, false);
    if (!p || p->dead) { pthread_mutex_unlock(&g_dc_m); return; }
    uint64_t seq = ++g_dc_seq;
    double t = dc_now();
    const struct DcDesc *sh = NULL;
    struct DcObj *s = NULL;
    const char *src = "none", *sflags = "-";
    if (c->dc.kind == 1) {
        src = "set";
        s = dc_get(DCK_SET, c->dc.set0, false);
        if (!s || !s->u.set.b) sflags = "UNK";
        else {
            sh = s->u.set.b;
            if (s->dead) sflags = "DEAD";
            if (!s->cap) {
                s->cap = true;
                struct DcObj *po = dc_get(DCK_POOL, s->u.set.pool, true);
                if (po && po->u.pool.n == po->u.pool.cap) {
                    uint32_t nc = po->u.pool.cap ? po->u.pool.cap * 2 : 64;
                    uint64_t *ns = realloc(po->u.pool.sets, nc * sizeof *ns);
                    if (ns) { po->u.pool.sets = ns; po->u.pool.cap = nc; }
                }
                if (po && po->u.pool.n < po->u.pool.cap) po->u.pool.sets[po->u.pool.n++] = s->h;
            }
            s->u.set.used_cap = true;
            s->u.set.last_cap_seq = seq;
        }
    } else if (c->dc.kind == 2) {
        src = "push";
        sh = c->dc.push;
    }
    bool swab = s && s->u.set.wseq > c->dc.bind_seq;
    int k = snprintf(row, sizeof row, "X\t%.3f\t%llu\t%p\t%08x\t%u\t%u\t%u\t%s\t0x%llx\t%s\t", t, (unsigned long long)seq,
                     (void *)cmd, p->u.pipe.hash, x, y, z, src, (unsigned long long)(c->dc.kind == 1 ? c->dc.set0 : 0), sflags);
    for (uint32_t b = 0; sh && b < DC_NB && k > 0 && (size_t)k < sizeof row - 24; b++) {
        if (!sh[b].type) continue;
        struct DcTuple tu;
        dc_resolve(&sh[b], b, c->dc.dyn[b], c->dc.dyn_mask >> b & 1, swab, &tu);
        bool fresh;
        uint32_t id = dc_tuple_id(&tu, &fresh);
        k += snprintf(row + k, sizeof row - (size_t)k, "%s%u=%u", nb ? " " : "", b, id);
        if (fresh) {
            char fl[96];
            dc_flags_str(tu.flags, fl, sizeof fl);
            snprintf(brows[nb], sizeof brows[nb],
                     "B\t%u\t%u\t%u\t0x%llx\t0x%llx\t0x%llx\t%llu\t%llu\t0x%llx\t0x%llx\t%u\t%llu\t0x%llx\t%llu\t%llu\t%.3f\t%.3f\t%s\n",
                     id, b, tu.type, (unsigned long long)tu.handle, (unsigned long long)tu.buffer, (unsigned long long)tu.va,
                     (unsigned long long)tu.off, (unsigned long long)tu.range,
                     (unsigned long long)(tu.va ? tu.va + tu.off : 0), (unsigned long long)(tu.va ? tu.va + tu.off + tu.range : 0),
                     tu.format, (unsigned long long)tu.bufsize, (unsigned long long)tu.mem, (unsigned long long)tu.memsize,
                     (unsigned long long)tu.memoff, tu.buf_t, tu.mem_t, fl);
        } else brows[nb][0] = 0;
        nb++;
    }
    if (!c->dc.ndisp++) c->dc.first_seq = seq;
    c->dc.last_seq = seq;
    pthread_mutex_unlock(&g_dc_m);
    DCC(DCC_DISP);
    for (uint32_t i = 0; i < nb; i++) if (brows[i][0]) dc_out(brows[i], strlen(brows[i]));
    if (k > 0 && (size_t)k < sizeof row - 1) { row[k++] = '\n'; dc_out(row, (size_t)k); }
}

// ---- submits and completion (fence / timeline semaphore waits seen by the GPU_TIME wait hooks)
#define DC_PEND 1024
struct DcPend { uint64_t fence, sem, value, first, last; };
static struct DcPend g_dc_pend[DC_PEND];
static uint32_t g_dc_pend_next;
static void dc_pend_add(uint64_t fence, uint64_t sem, uint64_t value, uint64_t first, uint64_t last) { // g_dc_m held
    g_dc_pend[g_dc_pend_next] = (struct DcPend){fence, sem, value, first, last};
    g_dc_pend_next = (g_dc_pend_next + 1) % DC_PEND;
}
// One submit batch: cbs[] with their (optional) first timeline signal and the batch's fence (last batch only).
static void dc_submit_batch(VkQueue q, uint32_t ncb, VkCommandBuffer const *cbs, const VkCommandBufferSubmitInfo *cbi,
                            uint64_t sem, uint64_t value, VkFence fence) {
    bool any = false;
    for (uint32_t i = 0; i < ncb; i++) {
        VkCommandBuffer cb = cbs ? cbs[i] : cbi[i].commandBuffer;
        struct CmdBufData *c = cmd_data(cb);
        if (!c || !c->dc.ndisp) continue;
        any = true;
        DCC(DCC_SUBMIT);
        double t = dc_now();
        pthread_mutex_lock(&g_dc_m);
        dc_pend_add((uint64_t)fence, sem, value, c->dc.first_seq, c->dc.last_seq);
        pthread_mutex_unlock(&g_dc_m);
        dc_printf("S\t%.3f\t%p\t%p\t%llu\t%llu\t%u\t0x%llx\t0x%llx:%llu\n", t, (void *)q, (void *)cb,
                  (unsigned long long)c->dc.first_seq, (unsigned long long)c->dc.last_seq, c->dc.ndisp,
                  (unsigned long long)(uint64_t)fence, (unsigned long long)sem, (unsigned long long)value);
    }
    if (any) dc_flush();
}
static void dc_submit(VkQueue q, uint32_t n, const VkSubmitInfo *si, VkFence fence) {
    for (uint32_t i = 0; si && i < n; i++) {
        uint64_t sem = 0, val = 0;
        const VkTimelineSemaphoreSubmitInfo *ts = find_struct(si[i].pNext, VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO);
        if (si[i].signalSemaphoreCount) {
            sem = (uint64_t)si[i].pSignalSemaphores[0];
            if (ts && ts->signalSemaphoreValueCount && ts->pSignalSemaphoreValues) val = ts->pSignalSemaphoreValues[0];
        }
        dc_submit_batch(q, si[i].commandBufferCount, si[i].pCommandBuffers, NULL, sem, val, i + 1 == n ? fence : VK_NULL_HANDLE);
    }
}
static void dc_submit2(VkQueue q, uint32_t n, const VkSubmitInfo2 *si, VkFence fence) {
    for (uint32_t i = 0; si && i < n; i++) {
        uint64_t sem = 0, val = 0;
        if (si[i].signalSemaphoreInfoCount) { sem = (uint64_t)si[i].pSignalSemaphoreInfos[0].semaphore; val = si[i].pSignalSemaphoreInfos[0].value; }
        dc_submit_batch(q, si[i].commandBufferInfoCount, NULL, si[i].pCommandBufferInfos, sem, val, i + 1 == n ? fence : VK_NULL_HANDLE);
    }
}
static void dc_done(bool is_fence, uint64_t h, uint64_t value) { // a wait returned VK_SUCCESS for h (>= value)
    struct DcPend hit[16];
    uint32_t nh = 0;
    pthread_mutex_lock(&g_dc_m);
    for (uint32_t i = 0; i < DC_PEND && nh < 16; i++) {
        struct DcPend *p = &g_dc_pend[i];
        if (!p->first) continue;
        if (is_fence ? p->fence == h : (p->sem == h && p->value && p->value <= value)) { hit[nh++] = *p; p->first = 0; }
    }
    pthread_mutex_unlock(&g_dc_m);
    double t = dc_now();
    for (uint32_t i = 0; i < nh; i++) {
        DCC(DCC_DONE);
        dc_printf("F\t%.3f\t%s\t0x%llx\t%llu\t%llu\t%llu\n", t, is_fence ? "fence" : "sem", (unsigned long long)h,
                  (unsigned long long)(is_fence ? 0 : hit[i].value), (unsigned long long)hit[i].first, (unsigned long long)hit[i].last);
    }
}

// ---- desc-capture-1 entry points (only handed out while MALI_COMPAT_DESC_CAPTURE is on, and only for names the driver has)
struct DcFns {
    PFN_vkBindBufferMemory2 BindBufferMemory2;
    PFN_vkDestroyBufferView DestroyBufferView;
    PFN_vkCreateDescriptorSetLayout CreateDescriptorSetLayout;
    PFN_vkCreateDescriptorUpdateTemplate CreateDescriptorUpdateTemplate;
    PFN_vkDestroyDescriptorUpdateTemplate DestroyDescriptorUpdateTemplate;
    PFN_vkAllocateDescriptorSets AllocateDescriptorSets;
    PFN_vkFreeDescriptorSets FreeDescriptorSets;
    PFN_vkResetDescriptorPool ResetDescriptorPool;
    PFN_vkDestroyDescriptorPool DestroyDescriptorPool;
    PFN_vkUpdateDescriptorSets UpdateDescriptorSets;
    PFN_vkUpdateDescriptorSetWithTemplate UpdateDescriptorSetWithTemplate;
    PFN_vkCmdBindDescriptorSets CmdBindDescriptorSets;
    PFN_vkCmdBindDescriptorSets2 CmdBindDescriptorSets2;
    PFN_vkCmdPushDescriptorSet2 CmdPushDescriptorSet2;
    PFN_vkCmdPushDescriptorSetWithTemplate CmdPushDescriptorSetWithTemplate;
    PFN_vkCmdPushDescriptorSetWithTemplate2 CmdPushDescriptorSetWithTemplate2;
};
static void dc_device_init(struct DeviceData *d, PFN_vkGetDeviceProcAddr gdpa, VkDevice dev) {
    struct DcFns *f = calloc(1, sizeof *f);
    if (!f) return;
    #define DC_LOAD(fn) if (!(f->fn = (PFN_vk##fn)gdpa(dev, "vk" #fn))) f->fn = (PFN_vk##fn)gdpa(dev, "vk" #fn "KHR")
    DC_LOAD(BindBufferMemory2);
    DC_LOAD(DestroyBufferView);
    DC_LOAD(CreateDescriptorSetLayout);
    DC_LOAD(CreateDescriptorUpdateTemplate);
    DC_LOAD(DestroyDescriptorUpdateTemplate);
    DC_LOAD(AllocateDescriptorSets);
    DC_LOAD(FreeDescriptorSets);
    DC_LOAD(ResetDescriptorPool);
    DC_LOAD(DestroyDescriptorPool);
    DC_LOAD(UpdateDescriptorSets);
    DC_LOAD(UpdateDescriptorSetWithTemplate);
    DC_LOAD(CmdBindDescriptorSets);
    DC_LOAD(CmdBindDescriptorSets2);
    DC_LOAD(CmdPushDescriptorSet2);
    DC_LOAD(CmdPushDescriptorSetWithTemplate);
    DC_LOAD(CmdPushDescriptorSetWithTemplate2);
    #undef DC_LOAD
    if (!d->fpGetDeviceMemoryOpaqueCaptureAddress) {
        d->fpGetDeviceMemoryOpaqueCaptureAddress = (PFN_vkGetDeviceMemoryOpaqueCaptureAddress)gdpa(dev, "vkGetDeviceMemoryOpaqueCaptureAddress");
        if (!d->fpGetDeviceMemoryOpaqueCaptureAddress)
            d->fpGetDeviceMemoryOpaqueCaptureAddress = (PFN_vkGetDeviceMemoryOpaqueCaptureAddress)gdpa(dev, "vkGetDeviceMemoryOpaqueCaptureAddressKHR");
    }
    d->dcf = f;
    char hs[DC_HASH_MAX * 9 + 1] = "";
    for (uint32_t i = 0; i < g_dc_hash_count; i++)
        snprintf(hs + strlen(hs), sizeof hs - strlen(hs), "%s%08x", i ? "," : "", g_dc_hashes[i]);
    fprintf(stderr, "mali-compat: desc-capture-1 active: VkDevice %p, hashes %s, file %s, cap %llu MiB, driver BDA %s, "
            "capture address %s, template %s, bind2 %s, push %s, push2 %s, push-tpl %s, push-tpl2 %s\n", (void *)dev, hs,
            g_dc_path, (unsigned long long)(g_dc_cap >> 20), d->fpGetBufferDeviceAddress ? "yes" : "no",
            d->fpGetDeviceMemoryOpaqueCaptureAddress ? "yes" : "no", f->UpdateDescriptorSetWithTemplate ? "yes" : "no",
            f->CmdBindDescriptorSets2 ? "yes" : "no", d->fpCmdPushDescriptorSetKHR ? "yes" : "no",
            f->CmdPushDescriptorSet2 ? "yes" : "no", f->CmdPushDescriptorSetWithTemplate ? "yes" : "no",
            f->CmdPushDescriptorSetWithTemplate2 ? "yes" : "no");
    dc_printf("# device %p marker desc-capture-1 bda %d capaddr %d\n", (void *)dev, d->fpGetBufferDeviceAddress != NULL,
              d->fpGetDeviceMemoryOpaqueCaptureAddress != NULL);
}

static VKAPI_ATTR void VKAPI_CALL compat_DcDestroyBuffer(VkDevice device, VkBuffer b, const VkAllocationCallbacks *a) {
    dc_dead(DCK_BUF, (uint64_t)b);
    get_device_data(device)->fpDestroyBuffer(device, b, a);
}
static VKAPI_ATTR VkResult VKAPI_CALL compat_DcBindBufferMemory(VkDevice device, VkBuffer b, VkDeviceMemory m, VkDeviceSize off) {
    struct DeviceData *d = get_device_data(device);
    VkResult r = d->fpBindBufferMemory(device, b, m, off);
    if (r == VK_SUCCESS) dc_buf_bind(d, device, b, m, off);
    return r;
}
static VKAPI_ATTR VkResult VKAPI_CALL compat_DcBindBufferMemory2(VkDevice device, uint32_t n, const VkBindBufferMemoryInfo *bi) {
    struct DeviceData *d = get_device_data(device);
    VkResult r = d->dcf->BindBufferMemory2(device, n, bi);
    for (uint32_t i = 0; r == VK_SUCCESS && i < n; i++) dc_buf_bind(d, device, bi[i].buffer, bi[i].memory, bi[i].memoryOffset);
    return r;
}
static VKAPI_ATTR void VKAPI_CALL compat_DcDestroyBufferView(VkDevice device, VkBufferView v, const VkAllocationCallbacks *a) {
    dc_dead(DCK_VIEW, (uint64_t)v);
    get_device_data(device)->dcf->DestroyBufferView(device, v, a);
}
static VKAPI_ATTR VkResult VKAPI_CALL compat_DcCreateDescriptorSetLayout(VkDevice device, const VkDescriptorSetLayoutCreateInfo *ci,
                                                                          const VkAllocationCallbacks *a, VkDescriptorSetLayout *l) {
    VkResult r = get_device_data(device)->dcf->CreateDescriptorSetLayout(device, ci, a, l);
    if (r == VK_SUCCESS) dc_dsl_create(*l, ci);
    return r;
}
static VKAPI_ATTR VkResult VKAPI_CALL compat_DcCreatePipelineLayout(VkDevice device, const VkPipelineLayoutCreateInfo *ci,
                                                                     const VkAllocationCallbacks *a, VkPipelineLayout *l) {
    VkResult r = get_device_data(device)->fpCreatePipelineLayout(device, ci, a, l);
    if (r == VK_SUCCESS) dc_pl_create(*l, ci);
    return r;
}
static VKAPI_ATTR VkResult VKAPI_CALL compat_DcCreateDescriptorUpdateTemplate(VkDevice device, const VkDescriptorUpdateTemplateCreateInfo *ci,
                                                                               const VkAllocationCallbacks *a, VkDescriptorUpdateTemplate *t) {
    VkResult r = get_device_data(device)->dcf->CreateDescriptorUpdateTemplate(device, ci, a, t);
    if (r == VK_SUCCESS) dc_tpl_create(*t, ci);
    return r;
}
static VKAPI_ATTR void VKAPI_CALL compat_DcDestroyDescriptorUpdateTemplate(VkDevice device, VkDescriptorUpdateTemplate t,
                                                                           const VkAllocationCallbacks *a) {
    dc_tpl_destroy(t);
    get_device_data(device)->dcf->DestroyDescriptorUpdateTemplate(device, t, a);
}
static VKAPI_ATTR VkResult VKAPI_CALL compat_DcAllocateDescriptorSets(VkDevice device, const VkDescriptorSetAllocateInfo *ai,
                                                                       VkDescriptorSet *sets) {
    VkResult r = get_device_data(device)->dcf->AllocateDescriptorSets(device, ai, sets);
    if (r == VK_SUCCESS) dc_sets_alloc(ai, sets);
    return r;
}
static VKAPI_ATTR VkResult VKAPI_CALL compat_DcFreeDescriptorSets(VkDevice device, VkDescriptorPool pool, uint32_t n,
                                                                   const VkDescriptorSet *sets) {
    dc_sets_dead(pool, n, sets, "set-free");
    return get_device_data(device)->dcf->FreeDescriptorSets(device, pool, n, sets);
}
static VKAPI_ATTR VkResult VKAPI_CALL compat_DcResetDescriptorPool(VkDevice device, VkDescriptorPool pool, VkDescriptorPoolResetFlags fl) {
    DCC(DCC_POOL_RESET);
    dc_sets_dead(pool, 0, NULL, "set-reset");
    return get_device_data(device)->dcf->ResetDescriptorPool(device, pool, fl);
}
static VKAPI_ATTR void VKAPI_CALL compat_DcDestroyDescriptorPool(VkDevice device, VkDescriptorPool pool, const VkAllocationCallbacks *a) {
    DCC(DCC_POOL_DESTROY);
    dc_sets_dead(pool, 0, NULL, "set-pooldestroy");
    get_device_data(device)->dcf->DestroyDescriptorPool(device, pool, a);
}
static VKAPI_ATTR void VKAPI_CALL compat_DcUpdateDescriptorSets(VkDevice device, uint32_t nw, const VkWriteDescriptorSet *w,
                                                                uint32_t nc, const VkCopyDescriptorSet *c) {
    dc_update_sets(nw, w, nc, c);
    get_device_data(device)->dcf->UpdateDescriptorSets(device, nw, w, nc, c);
}
static VKAPI_ATTR void VKAPI_CALL compat_DcUpdateDescriptorSetWithTemplate(VkDevice device, VkDescriptorSet set,
                                                                           VkDescriptorUpdateTemplate t, const void *data) {
    dc_update_tpl(set, t, data);
    get_device_data(device)->dcf->UpdateDescriptorSetWithTemplate(device, set, t, data);
}
static struct DeviceData *dc_cmd_dev(VkCommandBuffer cmd, struct CmdBufData *c) { return c ? c->dev : get_device_data_by_cmd(cmd); }
static VKAPI_ATTR void VKAPI_CALL compat_DcCmdBindDescriptorSets(VkCommandBuffer cmd, VkPipelineBindPoint bp, VkPipelineLayout layout,
                                                                 uint32_t first, uint32_t n, const VkDescriptorSet *sets,
                                                                 uint32_t ndyn, const uint32_t *dyn) {
    struct CmdBufData *c = cmd_data(cmd);
    dc_bind_sets(c, bp, layout, first, n, sets, ndyn, dyn, false);
    dc_cmd_dev(cmd, c)->dcf->CmdBindDescriptorSets(cmd, bp, layout, first, n, sets, ndyn, dyn);
}
static VKAPI_ATTR void VKAPI_CALL compat_DcCmdBindDescriptorSets2(VkCommandBuffer cmd, const VkBindDescriptorSetsInfo *i) {
    struct CmdBufData *c = cmd_data(cmd);
    dc_bind_sets(c, i->stageFlags & VK_SHADER_STAGE_COMPUTE_BIT ? VK_PIPELINE_BIND_POINT_COMPUTE : VK_PIPELINE_BIND_POINT_GRAPHICS,
                 i->layout, i->firstSet, i->descriptorSetCount, i->pDescriptorSets, i->dynamicOffsetCount, i->pDynamicOffsets, true);
    dc_cmd_dev(cmd, c)->dcf->CmdBindDescriptorSets2(cmd, i);
}
static VKAPI_ATTR void VKAPI_CALL compat_DcCmdPushDescriptorSet2(VkCommandBuffer cmd, const VkPushDescriptorSetInfo *i) {
    struct CmdBufData *c = cmd_data(cmd);
    dc_push_writes(c, i->stageFlags & VK_SHADER_STAGE_COMPUTE_BIT ? VK_PIPELINE_BIND_POINT_COMPUTE : VK_PIPELINE_BIND_POINT_GRAPHICS,
                   i->layout, i->set, i->descriptorWriteCount, i->pDescriptorWrites, true);
    dc_cmd_dev(cmd, c)->dcf->CmdPushDescriptorSet2(cmd, i);
}
static VKAPI_ATTR void VKAPI_CALL compat_DcCmdPushDescriptorSetWithTemplate(VkCommandBuffer cmd, VkDescriptorUpdateTemplate t,
                                                                            VkPipelineLayout layout, uint32_t set, const void *data) {
    struct CmdBufData *c = cmd_data(cmd);
    dc_push_tpl(c, t, layout, set, data, false);
    dc_cmd_dev(cmd, c)->dcf->CmdPushDescriptorSetWithTemplate(cmd, t, layout, set, data);
}
static VKAPI_ATTR void VKAPI_CALL compat_DcCmdPushDescriptorSetWithTemplate2(VkCommandBuffer cmd, const VkPushDescriptorSetWithTemplateInfo *i) {
    struct CmdBufData *c = cmd_data(cmd);
    dc_push_tpl(c, i->descriptorUpdateTemplate, i->layout, i->set, i->pData, true);
    dc_cmd_dev(cmd, c)->dcf->CmdPushDescriptorSetWithTemplate2(cmd, i);
}
// vkGetDeviceProcAddr part: the hook only when the driver answers the name (else NULL = not handled here).
static PFN_vkVoidFunction dc_proc(VkDevice device, const char *n) {
    static const struct { const char *name; PFN_vkVoidFunction fn; } t[] = {
        {"vkDestroyBuffer", (PFN_vkVoidFunction)compat_DcDestroyBuffer},
        {"vkBindBufferMemory", (PFN_vkVoidFunction)compat_DcBindBufferMemory},
        {"vkBindBufferMemory2", (PFN_vkVoidFunction)compat_DcBindBufferMemory2},
        {"vkBindBufferMemory2KHR", (PFN_vkVoidFunction)compat_DcBindBufferMemory2},
        {"vkDestroyBufferView", (PFN_vkVoidFunction)compat_DcDestroyBufferView},
        {"vkCreateDescriptorSetLayout", (PFN_vkVoidFunction)compat_DcCreateDescriptorSetLayout},
        {"vkCreatePipelineLayout", (PFN_vkVoidFunction)compat_DcCreatePipelineLayout},
        {"vkCreateDescriptorUpdateTemplate", (PFN_vkVoidFunction)compat_DcCreateDescriptorUpdateTemplate},
        {"vkCreateDescriptorUpdateTemplateKHR", (PFN_vkVoidFunction)compat_DcCreateDescriptorUpdateTemplate},
        {"vkDestroyDescriptorUpdateTemplate", (PFN_vkVoidFunction)compat_DcDestroyDescriptorUpdateTemplate},
        {"vkDestroyDescriptorUpdateTemplateKHR", (PFN_vkVoidFunction)compat_DcDestroyDescriptorUpdateTemplate},
        {"vkAllocateDescriptorSets", (PFN_vkVoidFunction)compat_DcAllocateDescriptorSets},
        {"vkFreeDescriptorSets", (PFN_vkVoidFunction)compat_DcFreeDescriptorSets},
        {"vkResetDescriptorPool", (PFN_vkVoidFunction)compat_DcResetDescriptorPool},
        {"vkDestroyDescriptorPool", (PFN_vkVoidFunction)compat_DcDestroyDescriptorPool},
        {"vkUpdateDescriptorSets", (PFN_vkVoidFunction)compat_DcUpdateDescriptorSets},
        {"vkUpdateDescriptorSetWithTemplate", (PFN_vkVoidFunction)compat_DcUpdateDescriptorSetWithTemplate},
        {"vkUpdateDescriptorSetWithTemplateKHR", (PFN_vkVoidFunction)compat_DcUpdateDescriptorSetWithTemplate},
        {"vkCmdBindDescriptorSets", (PFN_vkVoidFunction)compat_DcCmdBindDescriptorSets},
        {"vkCmdBindDescriptorSets2", (PFN_vkVoidFunction)compat_DcCmdBindDescriptorSets2},
        {"vkCmdBindDescriptorSets2KHR", (PFN_vkVoidFunction)compat_DcCmdBindDescriptorSets2},
        {"vkCmdPushDescriptorSet2", (PFN_vkVoidFunction)compat_DcCmdPushDescriptorSet2},
        {"vkCmdPushDescriptorSet2KHR", (PFN_vkVoidFunction)compat_DcCmdPushDescriptorSet2},
        {"vkCmdPushDescriptorSetWithTemplate", (PFN_vkVoidFunction)compat_DcCmdPushDescriptorSetWithTemplate},
        {"vkCmdPushDescriptorSetWithTemplateKHR", (PFN_vkVoidFunction)compat_DcCmdPushDescriptorSetWithTemplate},
        {"vkCmdPushDescriptorSetWithTemplate2", (PFN_vkVoidFunction)compat_DcCmdPushDescriptorSetWithTemplate2},
        {"vkCmdPushDescriptorSetWithTemplate2KHR", (PFN_vkVoidFunction)compat_DcCmdPushDescriptorSetWithTemplate2},
    };
    if (strncmp(n, "vkCmd", 5) && !strstr(n, "Descriptor") && !strstr(n, "Buffer") && !strstr(n, "PipelineLayout")) return NULL;
    for (size_t i = 0; i < sizeof t / sizeof t[0]; i++) {
        if (strcmp(n, t[i].name)) continue;
        struct DeviceData *dd = get_device_data(device);
        if (dd && dd->dcf && dd->get_device_proc_addr(device, n)) return t[i].fn;
        return NULL;
    }
    return NULL;
}


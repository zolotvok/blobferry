// SPDX-License-Identifier: GPL-3.0-or-later
// cl_shader.h: shader identity (hashes, DUMP_HASH, VS_STORES), FRAGDEPTH_FIX, XFB_VS. Included by mali_compat_layer.c.
// -----------------------------------------------------------------------------
// GPU timing, MALI_COMPAT_GPU_TIME=1: shader and pipeline identities. [Claude Opus 5.5] 2026-10-28
// Every shader gets the FNV-1a hash of its SPIR-V words as the app passed them (before the clip/cull patch; the
// timing report and MALI_COMPAT_DUMP_HASH use it). MALI_COMPAT_DUMP_HASH=<hex>[,<hex>...] writes matching shaders to
// /root/mali-spv-<hash>.spv when they are created. The timing itself is further down (gt_*).
// -----------------------------------------------------------------------------
static bool g_gpu_time;
static bool g_wait_callers; // MALI_COMPAT_WAIT_CALLERS=1 (with GPU_TIME): who waits on which fence (K-WC1)
static uint32_t g_dump_hashes[16];
static uint32_t g_dump_hash_count;
static uint32_t g_gt_trace; // MALI_COMPAT_GT_TRACE=<hex>: write every pass whose fragment/compute shader has this hash
                            // (=all: every pass and dispatch, in GPU order with its start time)
static uint32_t g_skip_hash; // MALI_COMPAT_SKIP_HASH, see gt_skip
// cover-trim (MALI_COMPAT_COVER_TRIM, cover_trim.h): on/off and the hash lists, used before the header's code.
static int g_ct;
static uint32_t g_ct_cs[16], g_ct_cs_n, g_ct_fs[8], g_ct_fs_n;
static bool ct_cs_listed(uint32_t h);
static uint32_t g_noatt_w, g_noatt_h; // MALI_COMPAT_NOATT_AREA, see compat_CmdBeginRendering
static _Atomic uint64_t g_noatt_clamped;
// HZD statics (requests KZ1/KZ2 of game/hzd, 2026-10-30). [keeper, Claude Opus 5.5]
// MALI_COMPAT_DEPTH_EQUAL=always|lequal|gequal: graphics pipelines with depth test on, depth write off and compare op
// EQUAL get this op instead (on a copy of the depth/stencil state); a dynamic vkCmdSetDepthCompareOp(EQUAL) too. Tests
// whether the G-buffer draws after the depth prepass are lost to a prepass/G-buffer position mismatch.
static bool g_depth_equal_on;
static VkCompareOp g_depth_equal_op;
static _Atomic uint64_t g_deq_pipes, g_deq_replaced, g_deq_dynamic; // EQUAL/no-write pipelines seen, replaced; dynamic ops
// Pipeline flags in the pipeline table (GPU timing): per-pass draw counters eq / dw / vsw of the timing report.
enum { PF_EQ = 1, PF_DW = 2, PF_VSW = 4 };
// VS-stage store scan (KZ2): on with GPU_TIME, or log-only with MALI_COMPAT_VS_STORES=1. See spv_vs_stores.
static bool g_vs_scan;
static _Atomic uint64_t g_vsw_pipes;
static _Atomic uint32_t g_vsw_shaders;
// Graphics pipeline compiles per thread since the last timing report: did a stall come from the vendor compiler?
struct PcStat { char comm[16]; uint32_t count; uint64_t sum_ns, max_ns; };
static struct PcStat g_pcstat[16];
static pthread_mutex_t g_pcstat_m = PTHREAD_MUTEX_INITIALIZER;
static void pcstat_add(uint32_t count, uint64_t ns) {
    static __thread char comm[16];
    if (!comm[0]) {
        FILE *f = fopen("/proc/thread-self/comm", "r");
        if (!f || !fgets(comm, sizeof comm, f)) snprintf(comm, sizeof comm, "?");
        if (f) fclose(f);
        comm[strcspn(comm, "\n")] = 0;
    }
    pthread_mutex_lock(&g_pcstat_m);
    int i = 0;
    while (i < 16 && g_pcstat[i].count && strcmp(g_pcstat[i].comm, comm)) i++;
    if (i < 16) {
        if (!g_pcstat[i].count) snprintf(g_pcstat[i].comm, sizeof g_pcstat[i].comm, "%s", comm);
        g_pcstat[i].count += count;
        g_pcstat[i].sum_ns += ns;
        if (ns > g_pcstat[i].max_ns) g_pcstat[i].max_ns = ns;
    }
    pthread_mutex_unlock(&g_pcstat_m);
}
static _Atomic uint64_t g_skipped_draws;

static uint32_t spv_hash(const uint32_t *code, size_t bytes) {
    uint32_t h = 2166136261u;
    for (size_t w = 0; w < bytes / 4; w++) h = (h ^ code[w]) * 16777619u;
    return h;
}

static void spv_dump_if_wanted(const uint32_t *code, size_t bytes, uint32_t hash) {
    for (uint32_t i = 0; i < g_dump_hash_count; i++)
        if (g_dump_hashes[i] == hash) {
            char path[64];
            snprintf(path, sizeof path, "/root/mali-spv-%08x.spv", hash);
            FILE *f = fopen(path, "wb");
            if (f) { fwrite(code, 1, bytes, f); fclose(f); }
            fprintf(stderr, "mali-compat: dumped shader %08x (%zu bytes) to %s\n", hash, bytes, path);
        }
}

// handle -> hash tables (shader modules; pipelines with up to 3 stage hashes)
struct HashNode { uint64_t handle; uint32_t hash[3]; uint32_t stages[3]; uint32_t flags; struct HashNode *next; }; // flags: PF_*
#define HASH_BUCKETS 4096
static struct HashNode *g_module_hash[HASH_BUCKETS], *g_pipeline_hash[HASH_BUCKETS];
// disp-barrier (MALI_COMPAT_DISP_BARRIER / sync-diag ctl db=): fill the module and compute-pipeline hash tables. Set
// once in sd_init_once (vkCreateDevice), before any module or pipeline of that device exists.
static bool g_db_hash;
static pthread_mutex_t g_hash_mutex = PTHREAD_MUTEX_INITIALIZER;

static void hash_put(struct HashNode **table, uint64_t handle, const uint32_t *hash, const uint32_t *stages, uint32_t n,
                     uint32_t flags) {
    struct HashNode *e = calloc(1, sizeof *e);
    if (!e) return;
    e->handle = handle;
    e->flags = flags;
    for (uint32_t i = 0; i < n && i < 3; i++) { e->hash[i] = hash[i]; e->stages[i] = stages ? stages[i] : 0; }
    uint32_t b = handle_bucket(handle, HASH_BUCKETS);
    pthread_mutex_lock(&g_hash_mutex);
    e->next = table[b];
    table[b] = e;
    pthread_mutex_unlock(&g_hash_mutex);
}

static bool hash_get(struct HashNode **table, uint64_t handle, struct HashNode *out) {
    bool found = false;
    pthread_mutex_lock(&g_hash_mutex);
    for (struct HashNode *e = table[handle_bucket(handle, HASH_BUCKETS)]; e; e = e->next)
        if (e->handle == handle) { *out = *e; found = true; break; }
    pthread_mutex_unlock(&g_hash_mutex);
    return found;
}

// KZ2: does a vertex/tessellation/geometry module write memory? The device lacks vertexPipelineStoresAndAtomics, and
// vkd3d's FL 12_2 override hides that from the game, so such writes may be lost. Counts OpImageWrite (typed UAVs, and
// raw/structured ones as R32UI texel buffers), the writing atomics, and OpStore/OpCopyMemory through a StorageBuffer or
// PhysicalStorageBuffer pointer (or a Uniform BufferBlock one, old-style SSBOs) into cnt[0..2]; cnt[3] = execution model
// of the first such entry point. False for modules without a vertex-pipeline entry point (early exit after the header)
// or without writes. Whole module: a store in another entry point's function counts too. [keeper, Claude Opus 5.5]
static bool spv_vs_stores(const uint32_t *code, size_t bytes, uint32_t cnt[4]) {
    size_t n = bytes / 4;
    cnt[0] = cnt[1] = cnt[2] = 0;
    cnt[3] = UINT32_MAX;
    if (n < 5 || code[0] != 0x07230203 || !code[3] || code[3] > (1u << 22)) return false;
    uint32_t bound = code[3];
    size_t i = 5;
    for (; i < n;) { // capabilities, extensions, memory model, entry points, execution modes, debug names
        uint32_t wc = code[i] >> 16, op = code[i] & 0xffff;
        if (!wc || i + wc > n) return false;
        if (op > 17 && op != 317 && op != 330 && op != 331) break;
        if (op == SpvOpEntryPoint && wc >= 3 && code[i + 1] <= 3 && cnt[3] == UINT32_MAX) cnt[3] = code[i + 1];
        i += wc;
    }
    if (cnt[3] == UINT32_MAX) return false; // Vertex 0, TessellationControl 1, TessellationEvaluation 2, Geometry 3
    // per id: 1 pointer type to SSBO memory, 4 struct decorated BufferBlock, 8 value pointing into SSBO memory
    uint8_t *k = calloc(bound, 1);
    if (!k) return false;
    for (; i < n;) {
        uint32_t wc = code[i] >> 16, op = code[i] & 0xffff;
        const uint32_t *w = code + i;
        if (!wc || i + wc > n) break;
        i += wc;
        if (op == SpvOpDecorate && wc >= 3 && w[1] < bound && w[2] == 3 /* BufferBlock */) k[w[1]] |= 4;
        else if (op == SpvOpTypePointer && wc >= 4 && w[1] < bound && w[3] < bound) {
            if (w[2] == 12 || w[2] == 5349 || (w[2] == 2 && (k[w[3]] & 4))) k[w[1]] |= 1;
        } else if (op == SpvOpTypeForwardPointer && wc >= 3 && w[1] < bound) {
            if (w[2] == 5349) k[w[1]] |= 1;
        } else if ((op == SpvOpVariable || op == SpvOpFunctionParameter || op == SpvOpLoad || op == 57 /* FunctionCall */ ||
                    op == 120 /* ConvertUToPtr */ || op == 124 /* Bitcast */ || op == 169 /* Select */ ||
                    op == 245 /* Phi */) && wc >= 3 && w[1] < bound && w[2] < bound) {
            if (k[w[1]] & 1) k[w[2]] |= 8;
        } else if ((spv_is_chain(op) || op == 83 /* CopyObject */) && wc >= 4 && w[1] < bound && w[2] < bound && w[3] < bound) {
            if ((k[w[1]] & 1) || (k[w[3]] & 8)) k[w[2]] |= 8; // Uniform BufferBlock: the base carries it
        } else if ((op == SpvOpStore || op == SpvOpCopyMemory) && wc >= 3 && w[1] < bound) {
            if (k[w[1]] & 8) cnt[2]++;
        } else if (op == 99 /* ImageWrite */) {
            cnt[0]++;
        } else if ((op >= 228 && op <= 242) || op == 318 || op == 319 || op == 5614 || op == 5615 || op == 6035) {
            cnt[1]++; // AtomicStore .. AtomicXor, AtomicFlagTestAndSet/Clear, AtomicFMin/FMax/FAddEXT (not AtomicLoad)
        }
    }
    free(k);
    return cnt[0] || cnt[1] || cnt[2];
}

// KZ2: logs the first 32 shaders with vertex-pipeline writes (then at 64, 128, ...), once per hash.
static void vs_store_note(uint32_t hash, const uint32_t cnt[4], bool inline_spv) {
    static uint32_t seen[4096];
    static pthread_mutex_t m = PTHREAD_MUTEX_INITIALIZER;
    uint32_t key = hash ? hash : 1, b = (key * 2654435761u) % 4096, total = 0;
    pthread_mutex_lock(&m);
    for (uint32_t p = 0; p < 4096; p++, b = (b + 1) % 4096) {
        if (seen[b] == key) break;
        if (!seen[b]) {
            if (atomic_load(&g_vsw_shaders) < 4000) { seen[b] = key; total = atomic_fetch_add(&g_vsw_shaders, 1) + 1; }
            break;
        }
    }
    pthread_mutex_unlock(&m);
    static const char *model[4] = {"vs", "tcs", "tes", "gs"};
    if (total && (total <= 32 || !(total & (total - 1))))
        fprintf(stderr, "mali-compat: VS-stage store #%u %s:%08x (%s; imgwrite %u, atomic %u, store %u)\n", total,
                cnt[3] < 4 ? model[cnt[3]] : "?", hash, inline_spv ? "inline" : "module", cnt[0], cnt[1], cnt[2]);
}

// Images and views, for the details of a STUCK or traced pass (GPU timing only). Entries are not removed: a reused
// handle gets a newer entry in front, which is the one found.
struct ImgNode {
    uint64_t handle;
    VkImage image;                  // views: the image; images: itself
    VkFormat format;
    uint32_t w, h, mip, mips, layer, layers, samples, usage;
    struct ImgNode *next;
};
static struct ImgNode *g_img_info[HASH_BUCKETS], *g_view_info[HASH_BUCKETS];

static void img_put(struct ImgNode **table, struct ImgNode n) {
    struct ImgNode *e = malloc(sizeof *e);
    if (!e) return;
    *e = n;
    uint32_t b = handle_bucket(n.handle, HASH_BUCKETS);
    pthread_mutex_lock(&g_hash_mutex);
    e->next = table[b];
    table[b] = e;
    pthread_mutex_unlock(&g_hash_mutex);
}

static bool img_get(struct ImgNode **table, uint64_t handle, struct ImgNode *out) {
    bool found = false;
    pthread_mutex_lock(&g_hash_mutex);
    for (struct ImgNode *e = table[handle_bucket(handle, HASH_BUCKETS)]; e; e = e->next)
        if (e->handle == handle) { *out = *e; found = true; break; }
    pthread_mutex_unlock(&g_hash_mutex);
    return found;
}

// Stage hashes of a pipeline being created: inline SPIR-V (maintenance5) or the module's hash.
static uint32_t stage_hashes(const VkPipelineShaderStageCreateInfo *st, uint32_t count, uint32_t *hash, uint32_t *stages) {
    uint32_t n = 0;
    for (uint32_t s = 0; s < count && n < 3; s++) {
        const VkShaderModuleCreateInfo *sm = find_struct(st[s].pNext, VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO);
        struct HashNode mod;
        if (sm) {
            hash[n] = spv_hash(sm->pCode, sm->codeSize);
            spv_dump_if_wanted(sm->pCode, sm->codeSize, hash[n]);
        } else if (hash_get(g_module_hash, (uint64_t)st[s].module, &mod)) {
            hash[n] = mod.hash[0];
        } else {
            hash[n] = 0;
        }
        stages[n++] = st[s].stage;
    }
    return n;
}

// -----------------------------------------------------------------------------
// FragDepth fix (default on; MALI_COMPAT_FRAGDEPTH_FIX=0 turns it off): a fragment shader that writes gl_FragDepth in a pipeline without a depth/stencil
// attachment gets that output turned into a private variable (and its Depth* execution mode dropped). Vulkan says the
// write is simply discarded; the vendor driver instead hangs the GPU on vkd3d's depth->colour copy pass, whose shader
// writes o_color and gl_FragDepth (Witcher 3 DX12, 2026-10-28: STUCK on fs:2b070dee with an R32F target, no depth).
// Module shaders that write FragDepth are kept (code copy) so their pipelines can get a patched module. [Claude Opus 5.5]
// -----------------------------------------------------------------------------
static bool g_fragdepth_fix;

static bool spv_writes_fragdepth(const uint32_t *code, size_t bytes) {
    size_t n = bytes / 4;
    if (n < 5 || code[0] != 0x07230203) return false;
    for (size_t i = 5; i < n;) {
        uint32_t wc = code[i] >> 16, op = code[i] & 0xffff;
        if (!wc || i + wc > n) return false;
        if (op == 71 && wc >= 4 && code[i + 2] == 11 && code[i + 3] == 22) return true; // OpDecorate BuiltIn FragDepth
        i += wc;
    }
    return false;
}

static uint32_t *spv_strip_fragdepth(const uint32_t *code, size_t bytes, size_t *out_bytes) {
    size_t n = bytes / 4;
    if (n < 5 || code[0] != 0x07230203) return NULL;
    uint32_t fd = 0;
    for (size_t i = 5; i < n;) {
        uint32_t wc = code[i] >> 16, op = code[i] & 0xffff;
        if (!wc || i + wc > n) return NULL;
        if (op == 71 && wc >= 4 && code[i + 2] == 11 && code[i + 3] == 22) fd = code[i + 1];
        i += wc;
    }
    if (!fd) return NULL;
    bool v14 = code[1] >= 0x00010400; // from SPIR-V 1.4 on, Private variables belong in the entry point interface
    uint32_t *out = malloc((n + 8) * sizeof *out);
    if (!out) return NULL;
    memcpy(out, code, 5 * sizeof *out);
    uint32_t ptr = code[3];
    out[3] = code[3] + 1;
    size_t o = 5;
    for (size_t i = 5; i < n;) {
        uint32_t wc = code[i] >> 16, op = code[i] & 0xffff;
        if (op == 71 && code[i + 1] == fd) { i += wc; continue; }                         // decorations of gl_FragDepth
        if (op == 16 && wc >= 3 && code[i + 2] >= 12 && code[i + 2] <= 16 && code[i + 2] != 13) { i += wc; continue; } // Depth* modes
        if (op == 15 && !v14) {                                                            // OpEntryPoint: drop fd
            size_t start = o;
            out[o++] = 0;
            for (uint32_t k = 1; k < wc; k++) if (!(k >= 3 && code[i + k] == fd && k > 3)) out[o++] = code[i + k];
            out[start] = (uint32_t)((o - start) << 16) | 15;
            i += wc;
            continue;
        }
        if (op == 59 && wc >= 4 && code[i + 2] == fd) {                                    // the variable itself
            uint32_t pointee = 0;
            for (size_t j = 5; j < n;) {
                uint32_t wj = code[j] >> 16;
                if ((code[j] & 0xffff) == 32 && code[j + 1] == code[i + 1]) { pointee = code[j + 3]; break; }
                j += wj ? wj : 1;
            }
            if (!pointee) { free(out); return NULL; }
            out[o++] = (4u << 16) | 32; out[o++] = ptr; out[o++] = 6; out[o++] = pointee; // OpTypePointer Private
            out[o++] = code[i]; out[o++] = ptr; out[o++] = fd; out[o++] = 6;               // OpVariable ... Private
            for (uint32_t k = 4; k < wc; k++) out[o++] = code[i + k];
            i += wc;
            continue;
        }
        memcpy(out + o, code + i, wc * sizeof *out);
        o += wc;
        i += wc;
    }
    *out_bytes = o * sizeof *out;
    return out;
}

struct CodeNode { uint64_t module; uint32_t *code; size_t bytes; struct CodeNode *next; };
static struct CodeNode *g_fragdepth_modules[HASH_BUCKETS];

static void fragdepth_module_put(VkShaderModule m, const uint32_t *code, size_t bytes) {
    struct CodeNode *e = malloc(sizeof *e);
    if (!e || !(e->code = malloc(bytes))) { free(e); return; }
    memcpy(e->code, code, bytes);
    e->bytes = bytes;
    e->module = (uint64_t)m;
    uint32_t b = handle_bucket(e->module, HASH_BUCKETS);
    pthread_mutex_lock(&g_hash_mutex);
    e->next = g_fragdepth_modules[b];
    g_fragdepth_modules[b] = e;
    pthread_mutex_unlock(&g_hash_mutex);
}

// with /root/mali-spv-inflight: every module's SPIR-V as the driver gets it, for dump_inflight_* (debug only)
static struct CodeNode *g_inflight_modules[HASH_BUCKETS];
static void inflight_module_put(VkShaderModule m, const uint32_t *code, size_t bytes) {
    struct CodeNode *e = malloc(sizeof *e);
    if (!e || !(e->code = malloc(bytes))) { free(e); return; }
    memcpy(e->code, code, bytes);
    e->bytes = bytes;
    e->module = (uint64_t)m;
    uint32_t b = handle_bucket(e->module, HASH_BUCKETS);
    pthread_mutex_lock(&g_hash_mutex);
    e->next = g_inflight_modules[b];
    g_inflight_modules[b] = e;
    pthread_mutex_unlock(&g_hash_mutex);
}
static const struct CodeNode *inflight_module_get(VkShaderModule m) {
    const struct CodeNode *found = NULL;
    pthread_mutex_lock(&g_hash_mutex);
    for (struct CodeNode *e = g_inflight_modules[handle_bucket((uint64_t)m, HASH_BUCKETS)]; e; e = e->next)
        if (e->module == (uint64_t)m) { found = e; break; }
    pthread_mutex_unlock(&g_hash_mutex);
    return found;
}

static const struct CodeNode *fragdepth_module_get(VkShaderModule m) {
    const struct CodeNode *found = NULL;
    pthread_mutex_lock(&g_hash_mutex);
    for (struct CodeNode *e = g_fragdepth_modules[handle_bucket((uint64_t)m, HASH_BUCKETS)]; e; e = e->next)
        if (e->module == (uint64_t)m) { found = e; break; }
    pthread_mutex_unlock(&g_hash_mutex);
    return found;
}

// -----------------------------------------------------------------------------
// Stream output without the geometry shader (MALI_COMPAT_XFB_VS=1). [Claude Opus 5.5] 2026-10-30
// DXVK turns D3D11 stream output into a passthrough geometry shader with transform feedback. On the vendor driver a
// geometry shader with transform feedback costs ~4 ms per draw, capture in the vertex shader 0.1-0.2 ms and a geometry
// shader without capture 0.04 ms (test/xfb-probe.c). Unity's DX11 GPU skinning (stream output, one draw per skinned
// mesh) took 35-41 ms of a ~50 ms frame in Yooka-Laylee IL. For a geometry shader that only copies input components
// to captured outputs (points in, one point out), the pipeline is built without it: the vertex shader gets one new
// captured output per geometry shader output (same XfbBuffer/XfbStride/Offset, location moved past the vertex
// shader's own outputs), filled from the vertex shader's outputs right before it returns.
// -----------------------------------------------------------------------------
static bool g_xfb_vs;
static _Atomic uint64_t g_xfb_vs_pipes;

enum { SD_LOC = 1, SD_COMP = 2, SD_XBUF = 4, SD_XSTRIDE = 8, SD_OFF = 16, SD_BUILTIN = 32, SD_STREAM = 64 };
struct SpvMod {
    const uint32_t *w;
    size_t n;
    uint32_t bound, model, entry_fn, entry_at, last_entry_end, caps_end, types_at, funcs_at;
    uint32_t *flags, *loc, *comp, *xbuf, *xstride, *off;
    uint32_t *op, *a, *b, *c; // defining opcode per result id and its operands (see spv_mod_parse)
    bool xfb, in_points, out_points, one_vertex, other_mode, has_xfb_cap;
};

static void spv_mod_free(struct SpvMod *m) { free(m->flags); }

// Indexes a module: decorations, types, pointers, variables and constants by result id, and where the sections start.
// a/b/c: TypeInt width/signed, TypeFloat width, TypeVector comp/count, TypeArray elem/length id, TypePointer
// storage/pointee, Variable pointer type/storage, Constant type/value.
static bool spv_mod_parse(const uint32_t *w, size_t bytes, struct SpvMod *m) {
    memset(m, 0, sizeof *m);
    m->w = w;
    m->n = bytes / 4;
    if (m->n < 5 || w[0] != 0x07230203 || w[3] > (1u << 20)) return false;
    m->bound = w[3];
    m->flags = calloc((size_t)m->bound * 10, sizeof(uint32_t));
    if (!m->flags) return false;
    m->loc = m->flags + m->bound; m->comp = m->loc + m->bound; m->xbuf = m->comp + m->bound;
    m->xstride = m->xbuf + m->bound; m->off = m->xstride + m->bound; m->op = m->off + m->bound;
    m->a = m->op + m->bound; m->b = m->a + m->bound; m->c = m->b + m->bound;
    int entries = 0;
    for (size_t i = 5; i < m->n;) {
        uint32_t wc = w[i] >> 16, op = w[i] & 0xffff;
        if (!wc || i + wc > m->n) { spv_mod_free(m); return false; }
        const uint32_t *x = w + i;
        if (op == 17) { m->caps_end = i + wc; if (wc >= 2 && x[1] == 53) m->has_xfb_cap = true; }
        else if (op == 15 && wc >= 3) { if (!entries++) { m->model = x[1]; m->entry_fn = x[2]; m->entry_at = i; } m->last_entry_end = i + wc; }
        else if (op == 16 && wc >= 3 && x[1] == m->entry_fn) {
            uint32_t mode = x[2];
            if (mode == 11) m->xfb = true;
            else if (mode == 19) m->in_points = true;
            else if (mode == 27) m->out_points = true;
            else if (mode == 26) m->one_vertex = wc >= 4 && x[3] == 1;
            else if (mode == 0) m->other_mode |= !(wc >= 4 && x[3] == 1); // Invocations other than 1
            else if (mode >= 20 && mode <= 29) m->other_mode = true; // other primitive types
        } else if (op == 71 && wc >= 3 && x[1] < m->bound) {
            uint32_t id = x[1], dec = x[2], v = wc >= 4 ? x[3] : 0;
            if (dec == 30) { m->flags[id] |= SD_LOC; m->loc[id] = v; }
            else if (dec == 31) { m->flags[id] |= SD_COMP; m->comp[id] = v; }
            else if (dec == 36) { m->flags[id] |= SD_XBUF; m->xbuf[id] = v; }
            else if (dec == 37) { m->flags[id] |= SD_XSTRIDE; m->xstride[id] = v; }
            else if (dec == 35) { m->flags[id] |= SD_OFF; m->off[id] = v; }
            else if (dec == 11) m->flags[id] |= SD_BUILTIN;
            else if (dec == 29 && v) m->flags[id] |= SD_STREAM;
        } else if (op >= 19 && op <= 39 && wc >= 2 && x[1] < m->bound) {
            if (!m->types_at) m->types_at = i;
            m->op[x[1]] = op;
            m->a[x[1]] = wc >= 3 ? x[2] : 0;
            m->b[x[1]] = wc >= 4 ? x[3] : 0;
        } else if ((op == 43 || op == 59) && wc >= 4 && x[2] < m->bound) {
            if (!m->types_at) m->types_at = i;
            m->op[x[2]] = op;
            m->a[x[2]] = x[1];
            m->b[x[2]] = x[3];
        } else if (op == 54 && !m->funcs_at) {
            m->funcs_at = i;
        }
        i += wc;
    }
    return entries == 1 && m->types_at && m->funcs_at;
}

// Scalar type of a 32-bit int/float type id as (op << 16 | signed << 8 | width), 0 for anything else.
static uint32_t spv_scalar_key(const struct SpvMod *m, uint32_t t) {
    if (t >= m->bound || (m->op[t] != 21 && m->op[t] != 22) || m->a[t] != 32) return 0;
    return m->op[t] << 16 | (m->op[t] == 21 ? m->b[t] : 0) << 8 | 32;
}

// The pointee of a variable, split into scalar type id and component count (1 for scalars); false for other types.
static bool spv_var_shape(const struct SpvMod *m, uint32_t var, uint32_t *scalar, uint32_t *count) {
    if (var >= m->bound || m->op[var] != 59) return false;
    uint32_t pt = m->a[var];
    if (pt >= m->bound || m->op[pt] != 32) return false;
    uint32_t t = m->b[pt];
    if (t < m->bound && m->op[t] == 23) { *scalar = m->a[t]; *count = m->b[t]; }
    else { *scalar = t; *count = 1; }
    return spv_scalar_key(m, *scalar) != 0;
}

struct XfbMap { uint32_t in_loc, in_comp, count, key, loc, comp, buf, stride, off; bool has_stride;
                uint32_t vs_var, vs_idx, vs_scalar, vs_count; };

// New vertex shader that also captures what the passthrough geometry shader `gs` captured, or NULL.
static uint32_t *xfb_vs_from_gs(const uint32_t *vs, size_t vs_bytes, const uint32_t *gs, size_t gs_bytes, size_t *out_bytes) {
    struct SpvMod g, v;
    if (!spv_mod_parse(gs, gs_bytes, &g)) return NULL;
    if (!spv_mod_parse(vs, vs_bytes, &v)) { spv_mod_free(&g); return NULL; }
    uint32_t *out = NULL;
    struct XfbMap map[64];
    uint32_t nmap = 0, chain_var[64], chain_k[64], chain_id[64], nchain = 0, load_id[64], load_chain[64], nload = 0;
    uint32_t labels = 0, emits = 0, fns = 0;
    if (g.model != 3 || !g.xfb || !g.in_points || !g.out_points || !g.one_vertex || g.other_mode || v.model != 0 || v.xfb) goto done;
    // the geometry shader body: one block of input element loads stored to captured outputs, then EmitVertex
    for (size_t i = g.funcs_at; i < g.n;) {
        uint32_t wc = gs[i] >> 16, op = gs[i] & 0xffff;
        const uint32_t *x = gs + i;
        i += wc;
        if (op == 54) { if (++fns > 1) goto done; }
        else if (op == 248) { if (++labels > 1) goto done; }
        else if (op == 218) emits++;
        else if (op == 253 || op == 56 || op == 8 || op == 317 || op == 0) continue;
        else if (op == 65 && (wc == 5 || wc == 6) && nchain < 64) {
            // OpAccessChain %ptr %id %input_var %c0 [%ck]: element 0 of the input array, optionally component k
            uint32_t base = x[3], i0 = x[4], k = wc == 6 ? x[5] : UINT32_MAX;
            if (base >= g.bound || g.op[base] != 59 || g.b[base] != 1 || (g.flags[base] & SD_BUILTIN)) goto done;
            if (i0 >= g.bound || g.op[i0] != 43 || g.b[i0] != 0) goto done;
            if (k != UINT32_MAX) { if (k >= g.bound || g.op[k] != 43) goto done; k = g.b[k]; }
            chain_id[nchain] = x[2]; chain_var[nchain] = base; chain_k[nchain++] = k;
        } else if (op == 61 && wc >= 4 && nload < 64) {
            uint32_t c = 0;
            while (c < nchain && chain_id[c] != x[3]) c++;
            if (c == nchain) goto done;
            load_id[nload] = x[2]; load_chain[nload++] = c;
        } else if (op == 62 && wc >= 3 && nmap < 64) {
            uint32_t dst = x[1], l = 0;
            while (l < nload && load_id[l] != x[2]) l++;
            if (l == nload || dst >= g.bound || g.op[dst] != 59 || g.b[dst] != 3) goto done;
            if ((g.flags[dst] & (SD_LOC | SD_XBUF | SD_OFF)) != (SD_LOC | SD_XBUF | SD_OFF) || (g.flags[dst] & (SD_BUILTIN | SD_STREAM))) goto done;
            uint32_t c = load_chain[l], in = chain_var[c], k = chain_k[c];
            // the input: an array of one scalar or vector at a location
            uint32_t ipt = g.a[in], iarr = ipt < g.bound ? g.b[ipt] : 0;
            if (iarr >= g.bound || g.op[iarr] != 28 || !(g.flags[in] & SD_LOC)) goto done;
            uint32_t ielem = g.a[iarr], iscalar = ielem, icount = 1;
            if (ielem < g.bound && g.op[ielem] == 23) { iscalar = g.a[ielem]; icount = g.b[ielem]; }
            uint32_t oscalar, ocount;
            if (!spv_var_shape(&g, dst, &oscalar, &ocount) || spv_scalar_key(&g, oscalar) != spv_scalar_key(&g, iscalar)) goto done;
            struct XfbMap *mp = &map[nmap++];
            memset(mp, 0, sizeof *mp);
            mp->in_loc = g.loc[in];
            mp->in_comp = g.comp[in] + (k == UINT32_MAX ? 0 : k);
            mp->count = ocount;
            if (k == UINT32_MAX ? ocount != icount : (ocount != 1 || k >= icount)) goto done;
            mp->key = spv_scalar_key(&g, oscalar);
            mp->loc = g.loc[dst]; mp->comp = g.comp[dst]; mp->buf = g.xbuf[dst]; mp->off = g.off[dst];
            mp->has_stride = g.flags[dst] & SD_XSTRIDE; mp->stride = g.xstride[dst];
        } else goto done;
    }
    if (emits != 1 || !nmap) goto done;
    // the vertex shader outputs that feed them; new locations go past the highest one
    uint32_t base = 0;
    for (uint32_t id = 1; id < v.bound; id++)
        if (v.op[id] == 59 && v.b[id] == 3 && (v.flags[id] & SD_LOC) && !(v.flags[id] & SD_BUILTIN)) {
            uint32_t s, c;
            if (!spv_var_shape(&v, id, &s, &c)) goto done; // arrays or structs: leave the pipeline alone
            if (v.loc[id] + 1 > base) base = v.loc[id] + 1;
        }
    for (uint32_t k = 0; k < nmap; k++) {
        struct XfbMap *mp = &map[k];
        for (uint32_t id = 1; id < v.bound && !mp->vs_var; id++) {
            uint32_t s, c;
            if (v.op[id] != 59 || v.b[id] != 3 || !(v.flags[id] & SD_LOC) || (v.flags[id] & SD_BUILTIN) || v.loc[id] != mp->in_loc) continue;
            if (!spv_var_shape(&v, id, &s, &c) || mp->in_comp < v.comp[id] || mp->in_comp + mp->count > v.comp[id] + c) continue;
            if (spv_scalar_key(&v, s) != mp->key || (mp->count > 1 && mp->count != c)) goto done;
            mp->vs_var = id; mp->vs_idx = mp->in_comp - v.comp[id]; mp->vs_scalar = s; mp->vs_count = c;
        }
        if (!mp->vs_var || base + mp->loc >= 32) goto done;
    }

    // write the new module
    size_t cap = v.n + 64 + nmap * 40;
    out = malloc(cap * sizeof *out);
    if (!out) goto done;
    size_t o = 0;
    uint32_t next = v.bound;
    uint32_t uint_t = 0;
    for (uint32_t id = 1; id < v.bound && !uint_t; id++) if (v.op[id] == 21 && v.a[id] == 32 && v.b[id] == 0) uint_t = id;
    bool new_uint = !uint_t;
    if (new_uint) uint_t = next++;
    uint32_t var_id[64], ptr_id[64], idx_id[64];
    for (uint32_t k = 0; k < nmap; k++) {
        var_id[k] = next++;
        ptr_id[k] = next++; // Output pointer to the captured type (scalar, or the VS variable's vector type)
        idx_id[k] = map[k].vs_count > 1 && map[k].count == 1 ? next++ : 0;
    }
    memcpy(out, vs, 5 * sizeof *out);
    o = 5;
    for (size_t i = 5; i < v.n;) {
        uint32_t wc = vs[i] >> 16, op = vs[i] & 0xffff;
        if (i == v.types_at) {
            // decorations of the new outputs, at the end of the annotation section
            for (uint32_t k = 0; k < nmap; k++) {
                out[o++] = 4u << 16 | 71; out[o++] = var_id[k]; out[o++] = 30; out[o++] = base + map[k].loc;
                if (map[k].comp) { out[o++] = 4u << 16 | 71; out[o++] = var_id[k]; out[o++] = 31; out[o++] = map[k].comp; }
                out[o++] = 4u << 16 | 71; out[o++] = var_id[k]; out[o++] = 36; out[o++] = map[k].buf;
                if (map[k].has_stride) { out[o++] = 4u << 16 | 71; out[o++] = var_id[k]; out[o++] = 37; out[o++] = map[k].stride; }
                out[o++] = 4u << 16 | 71; out[o++] = var_id[k]; out[o++] = 35; out[o++] = map[k].off;
            }
        }
        if (i == v.funcs_at) {
            // types, constants and variables, at the end of the global section
            if (new_uint) { out[o++] = 4u << 16 | 21; out[o++] = uint_t; out[o++] = 32; out[o++] = 0; }
            for (uint32_t k = 0; k < nmap; k++) {
                uint32_t vec_t = v.b[v.a[map[k].vs_var]]; // the VS variable's pointee
                out[o++] = 4u << 16 | 32; out[o++] = ptr_id[k]; out[o++] = 3; out[o++] = map[k].count > 1 ? vec_t : map[k].vs_scalar;
                if (idx_id[k]) { out[o++] = 4u << 16 | 43; out[o++] = uint_t; out[o++] = idx_id[k]; out[o++] = map[k].vs_idx; }
                out[o++] = 4u << 16 | 59; out[o++] = ptr_id[k]; out[o++] = var_id[k]; out[o++] = 3;
            }
        }
        if (op == 15 && i == v.entry_at) {
            out[o++] = (wc + nmap) << 16 | 15;
            memcpy(out + o, vs + i + 1, (wc - 1) * sizeof *out);
            o += wc - 1;
            for (uint32_t k = 0; k < nmap; k++) out[o++] = var_id[k];
            i += wc;
            if (i == v.last_entry_end) { out[o++] = 3u << 16 | 16; out[o++] = v.entry_fn; out[o++] = 11; }
            continue;
        }
        if (op == 253) {
            // before every return (only the entry function returns void here; DXVK inlines everything into main)
            uint32_t tmp = next;
            for (uint32_t k = 0; k < nmap; k++) {
                uint32_t src = map[k].vs_var;
                uint32_t vt = map[k].count > 1 ? v.b[v.a[map[k].vs_var]] : map[k].vs_scalar;
                if (idx_id[k]) {
                    out[o++] = 5u << 16 | 65; out[o++] = ptr_id[k]; out[o++] = tmp; out[o++] = map[k].vs_var; out[o++] = idx_id[k];
                    src = tmp++;
                }
                out[o++] = 4u << 16 | 61; out[o++] = vt; out[o++] = tmp; out[o++] = src;
                out[o++] = 3u << 16 | 62; out[o++] = var_id[k]; out[o++] = tmp++;
                if (o + 64 + nmap * 12 > cap) { cap *= 2; uint32_t *nw = realloc(out, cap * sizeof *out); if (!nw) { free(out); out = NULL; goto done; } out = nw; }
            }
            next = tmp;
        }
        if (o + wc + 64 + nmap * 40 > cap) { cap = cap * 2 + wc; uint32_t *nw = realloc(out, cap * sizeof *out); if (!nw) { free(out); out = NULL; goto done; } out = nw; }
        memcpy(out + o, vs + i, wc * sizeof *out);
        o += wc;
        i += wc;
        if (op == 17 && i == v.caps_end && !v.has_xfb_cap) { out[o++] = 2u << 16 | 17; out[o++] = 53; }
    }
    out[3] = next;
    *out_bytes = o * sizeof *out;
done:
    spv_mod_free(&g);
    spv_mod_free(&v);
    return out;
}


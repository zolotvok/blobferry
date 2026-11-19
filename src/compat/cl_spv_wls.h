// SPDX-License-Identifier: GPL-3.0-or-later
// cl_spv_wls.h: SPIR-V patchers: clip/cull stripper, DONTUNROLL, WLS_CLAMP, WLS_FIX + sgid. Included by mali_compat_layer.c.
// -----------------------------------------------------------------------------
// SPIR-V Stripper / Patcher (ClipDistance / CullDistance)
// -----------------------------------------------------------------------------
#define SpvOpNop 0
#define SpvOpMemberName 6
#define SpvOpEntryPoint 15
#define SpvOpCapability 17
#define SpvOpTypeStruct 30
#define SpvOpTypePointer 32
#define SpvOpConstant 43
#define SpvOpVariable 59
#define SpvOpStore 62
#define SpvOpDecorate 71
#define SpvOpMemberDecorate 72
#define SpvCapabilityClipDistance 32
#define SpvCapabilityCullDistance 33
#define SpvBuiltInClipDistance 3
#define SpvBuiltInCullDistance 4
#define SpvStorageClassInput 1
#define SpvStorageClassOutput 3
#define SpvStorageClassPrivate 6
#define SpvDecorationBuiltIn 11

// Debug aid: when /root/mali-spv exists, every module the patcher changes is written there as NNN-orig.spv and
// NNN-patched.spv (first 1000), for spirv-val on the PC. [Claude Opus 5.5]
static bool g_dump_spirv, g_dump_inflight, g_log_pipelines, g_log_images; // checked once, in vkCreateInstance

// MALI_COMPAT_RT_PIPELINE=1 (the app's "Ray tracing: DX12 games (DXR)" switch) [keeper, Claude Sonnet 5.5] 2026-11-07.
// Hook for the future VK_KHR_ray_tracing_pipeline emulation (DXR needs it; the vendor driver has ray query only).
// Today the emulation does not exist: rt_pipeline_emulation_available() is false, the key only logs one line per instance
// and changes nothing (no extension, no feature). When the emulation lands it plugs in here: it returns true, and the
// device/feature/extension reporting code advertises VK_KHR_ray_tracing_pipeline + rayTracingPipeline when
// g_rt_pipeline && rt_pipeline_emulation_available(). Keep src/compat/capabilities.json.in ("rt_pipeline") in step.
static bool g_rt_pipeline;
// rt-gate-1112 [RTX keeper, Claude Opus 5.5] 2026-11-11: g_as_hooks = g_rt_pipeline || MALI_COMPAT_AS_FIX=1 (default 0).
// It gates the acceleration-structure part of the RT emulation (AS tracking, the layer-side CLONE copy that avoids the
// blob's CLONE overrun (as-probe 11-10), build planning/scratch, AS BDA checks, the AS-buffer transfer bits) separately
// from the RT pipeline emulation, so a ray-query game can opt in to it without MALI_COMPAT_RT_PIPELINE. With both keys
// unset every AS call goes to the next layer untouched, also through the exported entry points (no FEX thunk patch).
static bool g_as_hooks;
static bool rt_pipeline_emulation_available(void) { return true; }
static bool g_bc_afrc = true; // MALI_COMPAT_BC_AFRC=0 turns the fixed-rate BC shadows off
// MALI_COMPAT_BDA_EXT=1: vkCreateDevice also enables VK_KHR_buffer_device_address when the app did not and the driver
// has it. The BC decode reads the app's upload buffer and writes its staging buffer through device addresses, so it
// needs vkGetBufferDeviceAddress. An app on Vulkan 1.0/1.1 that does not enable the extension (DOOM Eternal: instance
// apiVersion 1.1.0) gets a device whose vkGetDeviceProcAddr has neither vkGetBufferDeviceAddress nor ...KHR: the
// decode then called a null pointer on every BC upload. Under Wine the fault is swallowed inside the unix call (the
// call just returns), so nothing was logged and every BC texture stayed undecoded ("BC decodes 0" with 74 514
// uploads, 2026-11-06). Without the key such an upload is now skipped and counted (bc-dec nobda), not called.
// The key and the switch of the bc-dec counts line (the key or MALI_COMPAT_FAIL_LOG) are read once (pthread_once):
// device creation and the log thread read them from other threads.
// [game/doom_eternal, Claude Fable 5.2] 2026-11-12
static bool g_bda_ext_val, g_bcd_log_val;
static pthread_once_t g_bda_once = PTHREAD_ONCE_INIT;
static void bda_ext_init(void) {
    g_bda_ext_val = getenv("MALI_COMPAT_BDA_EXT") && atoi(getenv("MALI_COMPAT_BDA_EXT")) > 0;
    g_bcd_log_val = g_bda_ext_val || (getenv("MALI_COMPAT_FAIL_LOG") && atoi(getenv("MALI_COMPAT_FAIL_LOG")) > 0);
}
static bool bda_ext_on(void) { pthread_once(&g_bda_once, bda_ext_init); return g_bda_ext_val; }
static bool bcd_log_on(void) { pthread_once(&g_bda_once, bda_ext_init); return g_bcd_log_val; }
static _Atomic unsigned long long g_bcd_enter, g_bcd_nobda, g_bcd_done; // BC uploads: seen / skipped (no BDA) / decoded
static void dump_spirv(const uint32_t *orig, size_t orig_bytes, const uint32_t *patched, size_t patched_bytes) {
    static _Atomic int n;
    if (!g_dump_spirv) return;
    int i = atomic_fetch_add(&n, 1);
    if (i >= 1000) return;
    char path[64];
    for (int k = 0; k < 2; k++) {
        snprintf(path, sizeof path, "/root/mali-spv/%03d-%s.spv", i, k ? "patched" : "orig");
        FILE *f = fopen(path, "wb");
        if (!f) return;
        fwrite(k ? (const void *)patched : (const void *)orig, 1, k ? patched_bytes : orig_bytes, f);
        fclose(f);
    }
}

static uint32_t *compat_patch_spirv_impl(const uint32_t *inWords, size_t inBytes, size_t *pOutBytes);
static uint32_t *spv_fix_u64_to_float(const uint32_t *in, size_t bytes);
static __thread size_t g_fix_out_bytes; // size of spv_fix_u64_to_float's result
static uint32_t *spv_fix_broadcast_first(const uint32_t *in, size_t bytes);
static bool g_spv_fix; // MALI_COMPAT_SPV_FIX=1: the two vendor-compiler workarounds below (Control); off by default

// MALI_COMPAT_SPV_DONTUNROLL=1: every loop gets the DontUnroll loop control (Unroll cleared). The Witcher 3's DX11
// skinning vertex shaders have a 192-iteration loop, and the vendor compiler took 4.4-6.5 s on each pipeline with
// them, stalling DXVK's CS thread (0 FPS, 2026-10-30); suspect: full unrolling. Returns a patched copy or NULL.
// [Claude Opus 5.5] 2026-10-30
static bool g_spv_dontunroll;
static uint32_t *spv_dont_unroll(const uint32_t *in, size_t bytes) {
    if (!in || bytes < 20 || bytes % 4 || in[0] != 0x07230203) return NULL;
    size_t n = bytes / 4;
    uint32_t *out = NULL;
    for (size_t o = 5; o < n;) {
        uint32_t wc = in[o] >> 16, op = in[o] & 0xffff;
        if (!wc || o + wc > n) { free(out); return NULL; }
        if (op == 246 && wc >= 4 && (in[o + 3] & 3) != 2) { // OpLoopMerge %merge %continue LoopControl
            if (!out) {
                if (!(out = malloc(bytes))) return NULL;
                memcpy(out, in, bytes);
            }
            out[o + 3] = (in[o + 3] & ~1u) | 2u;
        }
        o += wc;
    }
    return out;
}

// BEGIN wls-clamp-1
// MALI_COMPAT_WLS_CLAMP (WD2 B10-21) [keeper, Claude Opus 5.5] 2026-11-01: clamps every dynamic index into a
// Workgroup-storage composite (DXVK's TGSM arrays, e.g. WD2's g0 = uint[273][15]) with GLSL.std.450 UMin(idx, len-1)
// right before the OpAccessChain / OpInBoundsAccessChain that uses it. Loads, stores and atomics all go through the
// chain's pointer, so they are all covered. In-range indices are unchanged (UMin(i, len-1) == i for i < len); an
// out-of-range index (undefined behaviour in D3D11 and Vulkan) now aliases the last element, inside the object, like
// desktop LDS, instead of walking into neighbouring GPU VA as Mali's WLS (plain memory) does (B10-20 replay: 24/24
// translation faults with garbage g0 rows). Constant indices, struct members, OpPtrAccessChain and spec-constant
// array lengths are left alone (the latter two are counted as "skipped"). Unset or 0 = off; 1 or "all" = every module;
// else a comma list of hex hashes (FNV-1a of the SPIR-V as the app passed it, the GPU_TIME / DUMP_HASH hash).
static int g_wls_clamp;                 // 0 off, 1 all modules, 2 hash list
static uint32_t g_wls_hashes[16], g_wls_hash_count;
static _Atomic uint32_t g_wls_logged;
static _Atomic uint64_t g_wls_modules, g_wls_chains;
static uint32_t spv_hash(const uint32_t *code, size_t bytes);

static pthread_once_t g_wls_once = PTHREAD_ONCE_INIT;
static void wls_clamp_init_impl(void) {
    const char *v = getenv("MALI_COMPAT_WLS_CLAMP");
    g_wls_clamp = 0;
    g_wls_hash_count = 0;
    if (!v || !*v || !strcmp(v, "0")) return;
    if (!strcmp(v, "1") || !strcmp(v, "all")) { g_wls_clamp = 1; return; }
    for (const char *h = v; *h && g_wls_hash_count < 16;) {
        char *end;
        unsigned long x = strtoul(h, &end, 16);
        if (end == h) break;
        g_wls_hashes[g_wls_hash_count++] = (uint32_t)x;
        h = *end == ',' ? end + 1 : end;
    }
    g_wls_clamp = g_wls_hash_count ? 2 : 0;
    if (!g_wls_clamp) fprintf(stderr, "mali-compat: MALI_COMPAT_WLS_CLAMP=%s ignored (1, all or hex hashes)\n", v);
}
// once per process, from vkCreateInstance and vkCreateDevice (a device made without this layer's instance still parses it)
static void wls_clamp_init(void) { pthread_once(&g_wls_once, wls_clamp_init_impl); }

static bool wls_clamp_wanted(uint32_t hash) {
    if (g_wls_clamp == 1) return true;
    for (uint32_t i = 0; g_wls_clamp == 2 && i < g_wls_hash_count; i++)
        if (g_wls_hashes[i] == hash) return true;
    return false;
}

// Opcodes < 512 that have a result type (word 1) and a result id (word 2); from SpvHasResultAndType (SPIRV-Headers
// unified1). Ids made by other opcodes have an unknown type: a chain indexed by one is skipped, never guessed.
static const uint64_t k_spv_typed_ops[8] = {
    0x3adf7e0000001002ull, 0xdfffeff7ffdfe07eull, 0xfffffffffdffffffull, 0x0027ffe800ffbffcull,
    0x5fff99f87cfcffe8ull, 0x00007fffffffe17bull, 0x00000000000f0000ull, 0x0000000000000000ull};
static bool spv_op_typed(uint32_t op) { return op < 512 && ((k_spv_typed_ops[op >> 6] >> (op & 63)) & 1); }

struct WlsCtx {
    const uint32_t *in;
    uint32_t bound, next;     // input id bound, next free id
    uint32_t *def, *type;     // id -> word offset of the defining instruction, id -> result type id
    uint32_t nconst;          // constants to add: (type, value lo, value hi, id)
    uint32_t cst[64][4];
    bool cst_new[64];
};

// value of an OpConstant / OpConstantNull integer id; false if it is not one (spec constants count as dynamic)
static bool wls_const_val(const struct WlsCtx *c, uint32_t id, uint64_t *v) {
    if (id >= c->bound || !c->def[id]) return false;
    const uint32_t *d = c->in + c->def[id];
    uint32_t op = d[0] & 0xffff, len = d[0] >> 16;
    if (op == 46) { *v = 0; return true; } // OpConstantNull
    if (op != 43 || len < 4) return false;
    *v = len >= 5 ? ((uint64_t)d[4] << 32 | d[3]) : d[3];
    return true;
}

// id of an integer constant of type t (32/64-bit) with value v: an existing OpConstant, else a new one (added once)
static uint32_t wls_const(struct WlsCtx *c, uint32_t t, uint64_t v, uint32_t width) {
    for (uint32_t i = 0; i < c->nconst; i++)
        if (c->cst[i][0] == t && c->cst[i][1] == (uint32_t)v && c->cst[i][2] == (uint32_t)(v >> 32)) return c->cst[i][3];
    if (c->nconst >= 64) return 0;
    uint32_t id = 0;
    for (uint32_t k = 1; k < c->bound && !id; k++) { // existing constant (first one in the module)
        uint64_t x;
        if (c->type[k] == t && wls_const_val(c, k, &x) && (c->in[c->def[k]] & 0xffff) == 43 && x == v) id = k;
    }
    bool fresh = !id;
    if (fresh) id = c->next++;
    uint32_t *e = c->cst[c->nconst];
    e[0] = t; e[1] = (uint32_t)v; e[2] = width > 32 ? (uint32_t)(v >> 32) : 0; e[3] = id;
    c->cst_new[c->nconst++] = fresh;
    return id;
}

// Plans the clamps of one access chain at word offset o: rep[k] = UMin bound constant id for index k (0 = keep).
// Returns 1 if the base is a Workgroup pointer, 0 otherwise; *skipped = a dynamic index that could not be clamped.
static int wls_chain_plan(struct WlsCtx *c, size_t o, uint32_t rep[], uint32_t *nrep, bool *skipped) {
    const uint32_t *w = c->in + o;
    uint32_t len = w[0] >> 16, base = w[3], nidx = len - 4;
    for (uint32_t k = 0; k < nidx; k++) rep[k] = 0;
    *nrep = 0;
    *skipped = false;
    if (base >= c->bound || !c->type[base] || c->type[base] >= c->bound || !c->def[c->type[base]]) return 0;
    const uint32_t *pt = c->in + c->def[c->type[base]];
    if ((pt[0] & 0xffff) != 32 || (pt[0] >> 16) < 4 || pt[2] != 4 /* Workgroup */) return 0;
    uint32_t cur = pt[3];
    for (uint32_t k = 0; k < nidx; k++) {
        uint32_t idx = w[4 + k];
        if (cur >= c->bound || !c->def[cur]) { *skipped = true; break; }
        const uint32_t *td = c->in + c->def[cur];
        uint32_t top = td[0] & 0xffff;
        uint64_t iv, count = 0;
        bool is_const = wls_const_val(c, idx, &iv);
        if (top == 30) { // struct: constant member index
            if (!is_const || 2 + iv >= (td[0] >> 16)) { *skipped = true; break; }
            cur = td[2 + iv];
            continue;
        }
        if (top == 28) { // array: length is a constant id
            if (!wls_const_val(c, td[3], &count) || (c->in[c->def[td[3]]] & 0xffff) != 43) count = 0;
        } else if (top == 23 || top == 24) { // vector / matrix: literal count
            count = td[3];
        } else { // runtime array or anything else: cannot be clamped (Workgroup has no runtime arrays)
            if (!is_const) *skipped = true;
            break;
        }
        cur = td[2];
        if (is_const) continue;
        if (!count) { *skipped = true; continue; }
        uint32_t it = idx < c->bound ? c->type[idx] : 0;
        const uint32_t *itd = it && it < c->bound && c->def[it] ? c->in + c->def[it] : NULL;
        if (!itd || (itd[0] & 0xffff) != 21 || (itd[2] != 32 && itd[2] != 64)) { *skipped = true; continue; }
        uint32_t cid = wls_const(c, it, count - 1, itd[2]);
        if (!cid) { *skipped = true; continue; }
        rep[k] = cid;
        (*nrep)++;
    }
    return 1;
}

// Returns a patched copy (size in *out_bytes) or NULL (nothing to clamp / not SPIR-V / malformed).
// st[0] Workgroup chains with >= 1 clamped index, st[1] clamped indices, st[2] Workgroup chains seen, st[3] skipped.
static uint32_t *spv_wls_clamp(const uint32_t *in, size_t bytes, size_t *out_bytes, uint32_t st[4]) {
    st[0] = st[1] = st[2] = st[3] = 0;
    if (!in || bytes < 20 || bytes % 4 || in[0] != 0x07230203) return NULL;
    size_t n = bytes / 4;
    uint32_t bound = in[3];
    if (!bound || bound > (1u << 22)) return NULL;
    struct WlsCtx c = {.in = in, .bound = bound, .next = bound};
    c.def = calloc(bound, 4);
    c.type = calloc(bound, 4);
    uint32_t *out = NULL;
    if (!c.def || !c.type) goto done;
    size_t mm_off = 0, fn_off = 0;
    uint32_t glsl = 0;
    for (size_t o = 5; o < n;) {
        uint32_t op = in[o] & 0xffff, len = in[o] >> 16;
        if (!len || o + len > n) goto done;
        if (spv_op_typed(op) && len >= 3 && in[o + 2] < bound && in[o + 1] < bound) {
            c.def[in[o + 2]] = (uint32_t)o;
            c.type[in[o + 2]] = in[o + 1];
        } else if (op >= 19 && op <= 38 && len >= 2 && in[o + 1] < bound) {
            c.def[in[o + 1]] = (uint32_t)o; // OpType*
        } else if (op == 11 && len >= 3 && !glsl) { // OpExtInstImport
            char name[16] = {0};
            memcpy(name, in + o + 2, (len - 2) * 4 < 15 ? (len - 2) * 4 : 15);
            if (!strcmp(name, "GLSL.std.450")) glsl = in[o + 1];
        }
        if (op == 14 && !mm_off) mm_off = o;      // OpMemoryModel
        if (op == 54 && !fn_off) fn_off = o;      // first OpFunction
        o += len;
    }
    if (!mm_off || !fn_off) goto done;
    // phase A: plan every chain (allocates the constant ids), count
    uint32_t rep[64], nrep, clamps = 0;
    bool skipped;
    for (size_t o = fn_off; o < n; o += in[o] >> 16) {
        uint32_t op = in[o] & 0xffff, len = in[o] >> 16;
        if ((op != 65 && op != 66 && op != 67 && op != 70) || len < 4) continue;
        if (op == 67 || op == 70) { // OpPtrAccessChain / OpInBoundsPtrAccessChain on Workgroup: not handled
            uint32_t b = in[o + 3];
            const uint32_t *pt = b < bound && c.type[b] < bound && c.def[c.type[b]] ? in + c.def[c.type[b]] : NULL;
            if (pt && (pt[0] & 0xffff) == 32 && pt[2] == 4) { st[2]++; st[3]++; }
            continue;
        }
        if (len - 4 > 64) continue;
        if (!wls_chain_plan(&c, o, rep, &nrep, &skipped)) continue;
        st[2]++;
        st[3] += skipped;
        if (nrep) { st[0]++; st[1] += nrep; clamps += nrep; }
    }
    if (!clamps) goto done;
    if (!glsl) glsl = c.next++;
    uint32_t nnew = 0;
    for (uint32_t i = 0; i < c.nconst; i++) nnew += c.cst_new[i];
    size_t cap = n + 6 + 5 * (size_t)nnew + 7 * (size_t)clamps;
    out = malloc(cap * 4);
    if (!out) goto done;
    memcpy(out, in, 5 * 4);
    size_t w = 5;
    bool need_import = glsl >= bound;
    for (size_t o = 5; o < n;) {
        uint32_t op = in[o] & 0xffff, len = in[o] >> 16;
        if (o == mm_off && need_import) {
            out[w++] = (6u << 16) | 11u; // OpExtInstImport %glsl "GLSL.std.450"
            out[w++] = glsl;
            memset(out + w, 0, 16);
            memcpy(out + w, "GLSL.std.450", 12);
            w += 4;
        }
        if (o == fn_off) {
            for (uint32_t i = 0; i < c.nconst; i++) {
                if (!c.cst_new[i]) continue;
                const uint32_t *itd = in + c.def[c.cst[i][0]];
                bool wide = itd[2] == 64;
                out[w++] = ((wide ? 5u : 4u) << 16) | 43u; // OpConstant
                out[w++] = c.cst[i][0];
                out[w++] = c.cst[i][3];
                out[w++] = c.cst[i][1];
                if (wide) out[w++] = c.cst[i][2];
            }
        }
        if (o >= fn_off && (op == 65 || op == 66) && len >= 4 && len - 4 <= 64 &&
            wls_chain_plan(&c, o, rep, &nrep, &skipped) && nrep) {
            uint32_t ids[64];
            for (uint32_t k = 0; k < len - 4u; k++) {
                ids[k] = in[o + 4 + k];
                if (!rep[k]) continue;
                uint32_t r = c.next++;
                out[w++] = (7u << 16) | 12u; // OpExtInst %type %r %glsl UMin %idx %len-1
                out[w++] = c.type[ids[k]];
                out[w++] = r;
                out[w++] = glsl;
                out[w++] = 38u;
                out[w++] = ids[k];
                out[w++] = rep[k];
                ids[k] = r;
            }
            memcpy(out + w, in + o, 4 * 4);
            w += 4;
            memcpy(out + w, ids, (len - 4) * 4);
            w += len - 4;
        } else {
            memcpy(out + w, in + o, len * 4);
            w += len;
        }
        o += len;
    }
    out[3] = c.next;
    *out_bytes = w * 4;
done:
    free(c.def);
    free(c.type);
    return out;
}

// One module through the clamp (from compat_patch_spirv): orig = the app's code (hash filter), cur = the code so far.
static uint32_t *wls_clamp_module(const uint32_t *orig, size_t orig_bytes, const uint32_t *cur, size_t cur_bytes,
                                  size_t *out_bytes) {
    uint32_t h = spv_hash(orig, orig_bytes);
    if (!wls_clamp_wanted(h)) return NULL;
    uint32_t st[4];
    uint32_t *out = spv_wls_clamp(cur, cur_bytes, out_bytes, st);
    if (st[2]) {
        atomic_fetch_add(&g_wls_modules, out ? 1 : 0);
        atomic_fetch_add(&g_wls_chains, st[0]);
        if (atomic_fetch_add(&g_wls_logged, 1) < 512)
            fprintf(stderr, "mali-compat: wls-clamp: module %08x chains %u idx %u wls-chains %u skipped %u%s\n", h, st[0],
                    st[1], st[2], st[3], out ? "" : " (unchanged)");
    }
    return out;
}
// END wls-clamp-1

// BEGIN wls-fix-1
// MALI_COMPAT_WLS_FIX (DOOM Eternal cs:371a5d57) [game/doom_eternal keeper, Claude Opus 5.5] 2026-11-13: a listed compute
// module is rewritten to a VIRTUAL 1-LANE SUBGROUP with ONE WORKGROUP SLOT PER INVOCATION, so its result no longer
// depends on the driver's subgroup width. DOOM's cs:371a5d57 (32x32) reduces per subgroup with ShuffleXor 1..16 (+
// Broadcast 32 under SubgroupSize > 32), stores arr[(x + 32y) / SubgroupSize] into a shared vec3[32] and lets
// invocation 0 loop over 1024 / SubgroupSize entries: written for 32/64 lanes. With the blob's 16 lanes the index runs
// to 63 (512 B past the array in Mali's plain-VA WLS) and mask 16 is out of range (undefined). Rewrite: every load of
// SubgroupSize -> constant 1, every ShuffleXor / Broadcast -> its value operand (the only lane is the invocation
// itself; the spec leaves other lanes undefined, identity is what a max/min reduction needs), every Workgroup array
// shorter than the invocation count N grown to N, then the wls-clamp-1 clamp with the new lengths (safety net), and
// the Unroll bit of every OpLoopMerge cleared (the reduce loop becomes N-1 iterations). Anything the rewrite cannot
// prove safe leaves the module as it is (one log line with the reason). Unset / empty / 0 = off; a comma list of hex
// hashes (the WLS_CLAMP / GPU_TIME FNV-1a of the app's SPIR-V); no "all".
// wls-fix-2 [game/doom_eternal keeper, Claude Opus 5.5] 2026-11-14: MALI_COMPAT_WLS_FIX_MODE=sgid (or 2) selects a cheaper
// rewrite of the same listed modules that keeps the real subgroup reduce (wls-fix-1 made invocation 0 walk 1023 slots
// serially: 2.234 ms/frame on the phone against 0.32 ms unfixed). Mode sgid: every ShuffleXor v, mask ->
// Select(SubgroupSize > mask, ShuffleXor, v) (an out-of-range mask gives the lane's own value); the slot division
// localIndex / SubgroupSize -> OpLoad BuiltIn SubgroupId, the count division N / SubgroupSize -> OpLoad BuiltIn
// NumSubgroups (two new Input variables on the entry point; no lane-packing assumption); the reduce loop counter starts
// at 0 instead of 1 (invocation 0 need not be in subgroup 0); Workgroup arrays shorter than N / 4 grown to N / 4 (one
// slot per subgroup down to 4 lanes) and clamped; Unroll hints off; Broadcast and the other SubgroupSize uses untouched.
// The mode refuses (module left as it is) unless it finds exactly that pattern (wlsf_sgid_plan). Unset / 1 / lane1 =
// the wls-fix-1 rewrite above; the mode key alone does nothing.
static int g_wls_fix;                    // 0 off, 1 hash list
static int g_wlsf_mode = 1;              // 1 = virtual 1-lane subgroup (wls-fix-1), 2 = sgid (wls-fix-2)
static uint32_t g_wlsf_hashes[16], g_wlsf_count;
static _Atomic uint32_t g_wlsf_logged, g_wlsf_left_logged;
static _Atomic uint64_t g_wlsf_rewritten, g_wlsf_left;

static pthread_once_t g_wlsf_once = PTHREAD_ONCE_INIT;
static void wls_fix_init_impl(void) {
    const char *v = getenv("MALI_COMPAT_WLS_FIX");
    if (!v || !*v || !strcmp(v, "0")) return;
    uint32_t n = 0;
    bool bad = false;
    for (const char *h = v;;) { // strict: 1-8 hex digits per item, ',' between items
        const char *e = h;
        uint32_t x = 0;
        while ((*e >= '0' && *e <= '9') || (*e >= 'a' && *e <= 'f') || (*e >= 'A' && *e <= 'F')) {
            x = x << 4 | (uint32_t)(*e <= '9' ? *e - '0' : (*e | 0x20) - 'a' + 10);
            e++;
        }
        if (e == h || e - h > 8 || (*e && *e != ',') || n >= 16) { bad = true; break; }
        g_wlsf_hashes[n++] = x;
        if (!*e) break;
        h = e + 1;
    }
    if (bad) {
        fprintf(stderr, "mali-compat: MALI_COMPAT_WLS_FIX=%s ignored (comma list of up to 16 hex shader hashes)\n", v);
        return;
    }
    g_wlsf_count = n;
    g_wls_fix = 1;
    const char *m = getenv("MALI_COMPAT_WLS_FIX_MODE");
    if (m && *m && (!strcmp(m, "2") || !strcmp(m, "sgid"))) g_wlsf_mode = 2;
    else if (m && *m && strcmp(m, "1") && strcmp(m, "lane1"))
        fprintf(stderr, "mali-compat: MALI_COMPAT_WLS_FIX_MODE=%s ignored (1 / lane1 or 2 / sgid), mode lane1\n", m);
    if (g_wlsf_mode == 2) fprintf(stderr, "mali-compat: wls-fix on (MALI_COMPAT_WLS_FIX=%s, MALI_COMPAT_WLS_FIX_MODE=sgid)\n", v);
    else fprintf(stderr, "mali-compat: wls-fix on (MALI_COMPAT_WLS_FIX=%s)\n", v);
}
// once per process, from vkCreateInstance and vkCreateDevice (as wls_clamp_init)
static void wls_fix_init(void) { pthread_once(&g_wlsf_once, wls_fix_init_impl); }

static bool wls_fix_wanted(uint32_t hash) {
    for (uint32_t i = 0; g_wls_fix && i < g_wlsf_count; i++)
        if (g_wlsf_hashes[i] == hash) return true;
    return false;
}

// Mali's maxComputeSharedMemorySize (Dimensity 9500 blob, 32 KiB). The patch path runs without a device at hand
// (vkCreateShaderModule has no physical device in reach cheaply), so the limit is this constant.
#define WLSF_SHARED_LIMIT 32768u
#define WLSF_MAX_VARS 16

struct WlsfStat {
    uint32_t n, loads, shx, bcast, grown, clamped, loops;
    uint32_t mode, new_len, init_from;                   // wls-fix-2: mode, grown length, loop start before (0 / 1)
    uint32_t old_len[WLSF_MAX_VARS];
    size_t in_bytes, out_bytes;
};

struct WlsfVar {
    uint32_t var, ptype, atype, elem, lenc, len, stride; // atype 0 = scalar/vector variable
    uint32_t var_off, stride_off;                        // word offsets of the OpVariable and of OpDecorate ArrayStride
    uint32_t new_len_c, new_atype, new_ptype;            // grown: the new length constant / array type / pointer type
    bool len_c_new, grow;
    uint64_t esize;                                      // array element stride (std430-like) or the whole size
};

// value of an integer OpConstant (not spec, not null); false otherwise
static bool wlsf_const(const uint32_t *in, const uint32_t *def, uint32_t bound, uint32_t id, uint64_t *v) {
    if (!id || id >= bound || !def[id]) return false;
    const uint32_t *d = in + def[id];
    if ((d[0] & 0xffff) != 43 || (d[0] >> 16) < 4) return false;
    *v = (d[0] >> 16) >= 5 ? ((uint64_t)d[4] << 32 | d[3]) : d[3];
    return true;
}

// std430-like size and alignment of a type (vec3 = 16 B); 0 = unknown / not allowed in workgroup storage here
static uint64_t wlsf_size(const uint32_t *in, const uint32_t *def, uint32_t bound, uint32_t t, uint64_t *align, int depth) {
    *align = 4;
    if (depth > 8 || !t || t >= bound || !def[t]) return 0;
    const uint32_t *d = in + def[t];
    uint32_t op = d[0] & 0xffff, len = d[0] >> 16;
    uint64_t a, s, v;
    switch (op) {
    case 20: return 4;                                            // bool
    case 21: case 22:                                             // int, float
        if (len < 3 || d[2] < 8 || d[2] > 64) return 0;
        *align = d[2] / 8;
        return d[2] / 8;
    case 23:                                                      // vector
        if (len < 4 || !(s = wlsf_size(in, def, bound, d[2], &a, depth + 1))) return 0;
        *align = s * (d[3] == 3 ? 4 : d[3]);
        return *align;
    case 24:                                                      // matrix: columns of padded vectors
        if (len < 4 || !(s = wlsf_size(in, def, bound, d[2], &a, depth + 1))) return 0;
        *align = a;
        return s * d[3];
    case 28:                                                      // array with a constant length
        if (len < 4 || !wlsf_const(in, def, bound, d[3], &v) || !v || v > (1u << 20)) return 0;
        if (!(s = wlsf_size(in, def, bound, d[2], &a, depth + 1))) return 0;
        *align = a;
        s = v * ((s + a - 1) / a * a);
        return s > (1ull << 32) ? 0 : s;
    case 30: {                                                    // struct
        uint64_t off = 0, ma = 4;
        for (uint32_t k = 2; k < len; k++) {
            if (!(s = wlsf_size(in, def, bound, d[k], &a, depth + 1))) return 0;
            off = (off + a - 1) / a * a + s;
            if (a > ma) ma = a;
        }
        *align = ma;
        return off ? (off + ma - 1) / ma * ma : 0;
    }
    default: return 0;
    }
}

// Is word k (1..len-1) of instruction w an id operand? Literal-only and string instructions answer false; unknown
// instructions answer true (an id compared against a literal can only cause a refusal, never a wrong rewrite).
static bool wlsf_is_id(const uint32_t *w, uint32_t k) {
    uint32_t op = w[0] & 0xffff;
    switch (op) {
    case 0: case 2: case 3: case 4: case 5: case 6: case 7: case 8: case 10: case 11: case 14: case 15: case 16:
    case 17: case 71: case 72: case 73: case 74: case 75: case 317: case 330: case 331: case 332: case 5632: case 5633:
    case 19: case 20: case 21: case 22: case 26: case 31:
        return false; // debug, annotations, mode setting, scalar types (OpEntryPoint interface: allowed anyway)
    case 43: case 50: return k <= 2;                 // OpConstant, OpSpecConstant: literal values
    case 52: return k != 3;                          // OpSpecConstantOp: literal opcode
    case 23: case 24: return k == 1 || k == 2;       // vector / matrix: literal count
    case 25: return k == 1 || k == 2;                // OpTypeImage
    case 32: return k != 2;                          // OpTypePointer: storage class
    case 39: return k == 1;                          // OpTypeForwardPointer
    case 59: return k != 3;                          // OpVariable: storage class
    case 54: return k != 3;                          // OpFunction: function control
    case 12: return k != 4;                          // OpExtInst: instruction number
    case 79: return k <= 4;                          // OpVectorShuffle: components
    case 81: return k <= 3;                          // OpCompositeExtract: indices
    case 82: return k <= 4;                          // OpCompositeInsert: indices
    case 68: return k <= 3;                          // OpArrayLength: member
    case 246: return k <= 2;                         // OpLoopMerge: control (+ parameters)
    case 247: return k == 1;                         // OpSelectionMerge
    default: return true;
    }
}

// ---- wls-fix-2 (MALI_COMPAT_WLS_FIX_MODE=sgid): the pattern check. All helpers run after pass 1 of spv_wls_fix
// (every instruction length checked, def[] / type[] filled).
struct WlsfSgid {
    size_t slot_off, count_off, init_off, ep_off, dec_off, sgvar_off; // word offsets of the instructions rewritten / followed
    uint32_t sgvar, sgptype, sgtype, booltype;                         // SubgroupSize variable, its Input pointer / uint type
    uint32_t zero, zero_type, init_val;                                // loop start: constant 0 of the counter's type
    uint32_t sgid, nsg;                                                // new ids: SubgroupId / NumSubgroups variables
    bool zero_new;
};

static const uint32_t *wlsf_dd(const uint32_t *in, const uint32_t *def, uint32_t bound, uint32_t id) {
    return id && id < bound && def[id] ? in + def[id] : NULL;
}

// up to max uses of id as an id operand (word offset of the user, operand index); the defining occurrence is not a use.
// Returns the number of uses (> max = too many, only max recorded).
static uint32_t wlsf_uses(const uint32_t *in, size_t n, uint32_t id, size_t *off, uint32_t *kk, uint32_t max) {
    uint32_t c = 0;
    for (size_t o = 5; o < n; o += in[o] >> 16) {
        const uint32_t *w = in + o;
        uint32_t op = w[0] & 0xffff, len = w[0] >> 16;
        for (uint32_t k = 1; k < len; k++) {
            if (w[k] != id || !wlsf_is_id(w, k)) continue;
            if ((spv_op_typed(op) && k == 2) || (op >= 19 && op <= 38 && k == 1)) continue;
            if (c < max) { off[c] = o; kk[c] = k; }
            c++;
        }
    }
    return c;
}

// id = component c of LocalInvocationId: OpLoad(AccessChain(lid, const c)) or CompositeExtract(OpLoad(lid), c)
static bool wlsf_lid_comp(const uint32_t *in, const uint32_t *def, uint32_t bound, uint32_t lid, uint32_t id, uint32_t c) {
    const uint32_t *d = wlsf_dd(in, def, bound, id);
    uint64_t x;
    if (!d || !lid) return false;
    if ((d[0] & 0xffff) == 61 && (d[0] >> 16) >= 4) {
        const uint32_t *ch = wlsf_dd(in, def, bound, d[3]);
        return ch && ((ch[0] & 0xffff) == 65 || (ch[0] & 0xffff) == 66) && (ch[0] >> 16) == 5 && ch[3] == lid &&
               wlsf_const(in, def, bound, ch[4], &x) && x == c;
    }
    if ((d[0] & 0xffff) == 81 && (d[0] >> 16) == 5) {
        const uint32_t *l = wlsf_dd(in, def, bound, d[3]);
        return l && (l[0] & 0xffff) == 61 && (l[0] >> 16) >= 4 && l[3] == lid && d[4] == c;
    }
    return false;
}

// id = the local invocation index: OpLoad LocalInvocationIndex, or (lz == 1) x + y * lx of LocalInvocationId (any order)
static bool wlsf_local_index(const uint32_t *in, const uint32_t *def, uint32_t bound, uint32_t lid, uint32_t lidx,
                             uint32_t lx, uint32_t lz, uint32_t id) {
    const uint32_t *d = wlsf_dd(in, def, bound, id);
    if (!d) return false;
    uint32_t op = d[0] & 0xffff, len = d[0] >> 16;
    if (op == 61 && len >= 4 && lidx && d[3] == lidx) return true;
    if (op != 128 || len != 5 || lz != 1) return false;
    for (int s = 0; s < 2; s++) {
        const uint32_t *m = wlsf_dd(in, def, bound, d[4 - s]);
        if (!wlsf_lid_comp(in, def, bound, lid, d[3 + s], 0) || !m || (m[0] & 0xffff) != 132 || (m[0] >> 16) != 5) continue;
        for (int t = 0; t < 2; t++) {
            uint64_t x;
            if (wlsf_lid_comp(in, def, bound, lid, m[3 + t], 1) && wlsf_const(in, def, bound, m[4 - t], &x) && x == lx) return true;
        }
    }
    return false;
}

static bool wlsf_is_wg_array(const struct WlsfVar *v, uint32_t nv, uint32_t id) {
    for (uint32_t i = 0; i < nv; i++)
        if (v[i].atype && v[i].var == id) return true;
    return false;
}

// Every use of value id is the single index of an access chain into a Workgroup array (their results -> ch[]);
// a store of it into a Function variable is followed once (*fvar = that variable, stored exactly once, every load
// of it used the same way) when allow_fvar. False = another use.
static bool wlsf_index_only(const uint32_t *in, size_t n, const uint32_t *def, uint32_t bound, const struct WlsfVar *v,
                            uint32_t nv, uint32_t id, bool allow_fvar, uint32_t *ch, uint32_t *nch) {
    size_t uo[16];
    uint32_t uk[16], nu = wlsf_uses(in, n, id, uo, uk, 16);
    if (nu > 16) return false;
    for (uint32_t i = 0; i < nu; i++) {
        const uint32_t *w = in + uo[i];
        uint32_t op = w[0] & 0xffff, len = w[0] >> 16;
        if ((op == 65 || op == 66) && len == 5 && uk[i] == 4 && wlsf_is_wg_array(v, nv, w[3])) {
            if (*nch >= 16) return false;
            ch[(*nch)++] = w[2];
            continue;
        }
        if (op != 62 || len != 3 || uk[i] != 2 || !allow_fvar) return false;
        const uint32_t *f = wlsf_dd(in, def, bound, w[1]);
        if (!f || (f[0] & 0xffff) != 59 || f[3] != 7) return false; // a Function variable
        size_t fo[16];
        uint32_t fk[16], nf = wlsf_uses(in, n, w[1], fo, fk, 16);
        if (nf > 16) return false;
        for (uint32_t j = 0; j < nf; j++) {
            const uint32_t *u = in + fo[j];
            uint32_t uop = u[0] & 0xffff;
            if (fo[j] == uo[i]) continue;                                       // the store of id itself
            if (uop != 61 || fk[j] != 3 || (u[0] >> 16) < 4) return false;      // another store or use
            if (!wlsf_index_only(in, n, def, bound, v, nv, u[2], false, ch, nch)) return false;
        }
    }
    return *nch > 0;
}

// Every use of value id is the right operand of a less-than comparison (direct, or through one Function variable
// stored once); the left operands -> lhs[].
static bool wlsf_bound_only(const uint32_t *in, size_t n, const uint32_t *def, uint32_t bound, uint32_t id, bool allow_fvar,
                            uint32_t *lhs, uint32_t *nl) {
    size_t uo[16];
    uint32_t uk[16], nu = wlsf_uses(in, n, id, uo, uk, 16);
    if (nu > 16) return false;
    for (uint32_t i = 0; i < nu; i++) {
        const uint32_t *w = in + uo[i];
        uint32_t op = w[0] & 0xffff, len = w[0] >> 16;
        if ((op == 176 || op == 177) && len == 5 && uk[i] == 4) { // OpULessThan / OpSLessThan i, count
            if (*nl >= 16) return false;
            lhs[(*nl)++] = w[3];
            continue;
        }
        if (op != 62 || len != 3 || uk[i] != 2 || !allow_fvar) return false;
        const uint32_t *f = wlsf_dd(in, def, bound, w[1]);
        if (!f || (f[0] & 0xffff) != 59 || f[3] != 7) return false;
        size_t fo[16];
        uint32_t fk[16], nf = wlsf_uses(in, n, w[1], fo, fk, 16);
        if (nf > 16) return false;
        for (uint32_t j = 0; j < nf; j++) {
            const uint32_t *u = in + fo[j];
            if (fo[j] == uo[i]) continue;
            if ((u[0] & 0xffff) != 61 || fk[j] != 3 || (u[0] >> 16) < 4) return false;
            if (!wlsf_bound_only(in, n, def, bound, u[2], false, lhs, nl)) return false;
        }
    }
    return *nl > 0;
}

// The Function variable C whose load (directly or through OpBitcast) is id; 0 if none.
static uint32_t wlsf_counter_of(const uint32_t *in, const uint32_t *def, uint32_t bound, uint32_t id) {
    const uint32_t *d = wlsf_dd(in, def, bound, id);
    if (d && (d[0] & 0xffff) == 124 && (d[0] >> 16) == 4) d = wlsf_dd(in, def, bound, d[3]);
    if (!d || (d[0] & 0xffff) != 61 || (d[0] >> 16) < 4) return 0;
    const uint32_t *f = wlsf_dd(in, def, bound, d[3]);
    return f && (f[0] & 0xffff) == 59 && f[3] == 7 ? d[3] : 0;
}

// Checks the wls-fix-2 pattern and fills p; NULL = rewrite, else the reason to leave the module as it is.
static const char *wlsf_sgid_plan(const uint32_t *in, size_t n, const uint32_t *def, const uint32_t *type, uint32_t bound,
                                  const struct WlsfVar *v, uint32_t nv, uint32_t sgvar, uint32_t lid, uint32_t lidx,
                                  uint32_t lx, uint32_t lz, uint32_t N, size_t fn_off, struct WlsfSgid *p, char *why_buf,
                                  size_t why_n) {
    memset(p, 0, sizeof *p);
    p->sgvar = sgvar;
    p->sgvar_off = def[sgvar];
    p->sgptype = in[def[sgvar] + 1];
    p->sgtype = in[def[p->sgptype] + 3];
    if (in[1] < 0x00010300) return "mode sgid: SPIR-V older than 1.3 (no SubgroupId / NumSubgroups)";
    bool cap = false;
    uint32_t ch[16], nch = 0, lhs[16], nl = 0;
    for (size_t o = 5; o < n; o += in[o] >> 16) {
        const uint32_t *w = in + o;
        uint32_t op = w[0] & 0xffff, len = w[0] >> 16;
        if (op == 17 && len >= 2 && w[1] >= 61 && w[1] <= 68) cap = true; // GroupNonUniform or one that declares it
        if (op == 15) p->ep_off = o;
        if (op == 20 && len >= 2 && !p->booltype) p->booltype = w[1];
        if (op == 71 && len >= 4 && w[1] == sgvar && w[2] == 11 && w[3] == 36) p->dec_off = o;
        if (op == 346) { // ShuffleXor: scalar result (vector needs SPIR-V 1.4 for OpSelect), 32-bit integer mask
            const uint32_t *rt = wlsf_dd(in, def, bound, w[1]), *mt = wlsf_dd(in, def, bound, type[w[5] < bound ? w[5] : 0]);
            uint32_t rop = rt ? rt[0] & 0xffff : 0;
            if (!(rop >= 20 && rop <= 22) && !(rop == 23 && in[1] >= 0x00010400))
                return "mode sgid: ShuffleXor result is not a scalar (vector select needs SPIR-V 1.4)";
            if (!mt || (mt[0] & 0xffff) != 21 || mt[2] != 32) return "mode sgid: ShuffleXor mask is not a 32-bit integer";
        }
        for (uint32_t k = 1; k < len; k++) { // uses of a SubgroupSize value: divisions are rewritten, comparisons kept
            if (!wlsf_is_id(w, k) || (spv_op_typed(op) && k == 2)) continue;
            const uint32_t *d = wlsf_dd(in, def, bound, w[k]);
            if (!d || (d[0] & 0xffff) != 61 || (d[0] >> 16) < 4 || d[3] != sgvar) continue;
            if (op == 134 && len == 5 && k == 4) { // OpUDiv x, SubgroupSize
                uint64_t x;
                if (wlsf_const(in, def, bound, w[3], &x)) {
                    if (x != N || p->count_off) return "mode sgid: a division of a constant other than one N / SubgroupSize";
                    p->count_off = o;
                } else {
                    if (p->slot_off) return "mode sgid: more than one division of a non-constant by SubgroupSize";
                    p->slot_off = o;
                }
            } else if (!(op >= 170 && op <= 179)) {
                snprintf(why_buf, why_n, "mode sgid: SubgroupSize value used by opcode %u (only divisions / comparisons)", op);
                return why_buf;
            }
        }
    }
    if (!cap) return "mode sgid: no GroupNonUniform capability";
    if (!p->ep_off || !p->dec_off || !p->booltype) return "mode sgid: entry point, SubgroupSize decoration or bool type missing";
    if (!p->slot_off || !p->count_off) return "mode sgid: no slot (localIndex / SubgroupSize) and count (N / SubgroupSize) divisions";
    if (in[p->slot_off + 1] != p->sgtype || in[p->count_off + 1] != p->sgtype) return "mode sgid: division type is not the SubgroupSize type";
    if (!wlsf_local_index(in, def, bound, lid, lidx, lx, lz, in[p->slot_off + 3]))
        return "mode sgid: slot division of something other than the local invocation index";
    // the slot only indexes Workgroup arrays (directly or through one Function variable), those chains are only stored to
    if (!wlsf_index_only(in, n, def, bound, v, nv, in[p->slot_off + 2], true, ch, &nch))
        return "mode sgid: slot used other than as the index of a Workgroup array access chain";
    // the count only bounds a less-than comparison; its left operand is the loop counter C
    if (!wlsf_bound_only(in, n, def, bound, in[p->count_off + 2], true, lhs, &nl))
        return "mode sgid: count used other than as the bound of a less-than comparison";
    uint32_t c = wlsf_counter_of(in, def, bound, lhs[0]);
    for (uint32_t i = 1; i < nl; i++)
        if (wlsf_counter_of(in, def, bound, lhs[i]) != c) c = 0;
    if (!c) return "mode sgid: the count is not compared with one Function-variable loop counter";
    size_t uo[16];
    uint32_t uk[16], nu = wlsf_uses(in, n, c, uo, uk, 16);
    if (nu > 16) return "mode sgid: loop counter used too often";
    for (uint32_t i = 0; i < nu; i++) { // loads; one store of a constant 0 / 1; the rest stores of load(C) + 1
        const uint32_t *w = in + uo[i];
        uint32_t op = w[0] & 0xffff, len = w[0] >> 16;
        if (op == 61 && uk[i] == 3) continue;
        if (op != 62 || len != 3 || uk[i] != 1) return "mode sgid: loop counter used other than by loads and stores";
        uint64_t x;
        if (wlsf_const(in, def, bound, w[2], &x)) {
            if (p->init_off || x > 1) return "mode sgid: loop counter does not start at one constant 0 or 1";
            p->init_off = uo[i];
            p->init_val = (uint32_t)x;
            p->zero_type = type[w[2]];
            continue;
        }
        const uint32_t *a = wlsf_dd(in, def, bound, w[2]);
        bool step = false;
        for (int s = 0; a && (a[0] & 0xffff) == 128 && (a[0] >> 16) == 5 && s < 2; s++) {
            const uint32_t *l = wlsf_dd(in, def, bound, a[3 + s]);
            step |= l && (l[0] & 0xffff) == 61 && (l[0] >> 16) >= 4 && l[3] == c && wlsf_const(in, def, bound, a[4 - s], &x) && x == 1;
        }
        if (!step) return "mode sgid: loop counter stored with something other than counter + 1";
    }
    if (!p->init_off) return "mode sgid: loop counter has no constant start";
    // every other Workgroup array chain: indexed by a load of C, only loaded from
    uint32_t nread = 0;
    for (size_t o = 5; o < n; o += in[o] >> 16) {
        const uint32_t *w = in + o;
        uint32_t op = w[0] & 0xffff, len = w[0] >> 16;
        if ((op != 65 && op != 66) || len < 4 || !wlsf_is_wg_array(v, nv, w[3])) continue;
        bool slot = false;
        for (uint32_t i = 0; i < nch; i++) slot |= ch[i] == w[2];
        size_t co[16];
        uint32_t ck[16], ncu = wlsf_uses(in, n, w[2], co, ck, 16);
        if (ncu > 16) return "mode sgid: Workgroup chain used too often";
        for (uint32_t i = 0; i < ncu; i++) {
            uint32_t uop = in[co[i]] & 0xffff;
            if (slot ? !(uop == 62 && ck[i] == 1) : !(uop == 61 && ck[i] == 3))
                return "mode sgid: a slot chain not only stored to, or a reduce chain not only loaded from";
        }
        if (slot) continue;
        const uint32_t *ix = len == 5 ? wlsf_dd(in, def, bound, w[4]) : NULL;
        if (!ix || (ix[0] & 0xffff) != 61 || (ix[0] >> 16) < 4 || ix[3] != c)
            return "mode sgid: a Workgroup array chain indexed by neither the slot nor the loop counter";
        nread++;
    }
    if (!nread) return "mode sgid: no reduce loop reads the Workgroup array";
    // the constant 0 of the counter's type (32-bit integer) for the new loop start
    const uint32_t *zt = wlsf_dd(in, def, bound, p->zero_type);
    if (!zt || (zt[0] & 0xffff) != 21 || zt[2] != 32) return "mode sgid: loop counter is not a 32-bit integer";
    for (uint32_t k = 1; k < bound && !p->zero; k++) {
        uint64_t x;
        if (type[k] == p->zero_type && def[k] < fn_off && wlsf_const(in, def, bound, k, &x) && x == 0) p->zero = k;
    }
    p->zero_new = !p->zero && p->init_val == 1;
    return NULL;
}

// Returns the rewritten module (size in *out_bytes) or NULL; *why = the reason it was left as it is.
// mode 1 = wls-fix-1 (virtual 1-lane subgroup), 2 = wls-fix-2 (sgid: guarded shuffles, SubgroupId slots).
static uint32_t *spv_wls_fix(const uint32_t *in, size_t bytes, size_t *out_bytes, struct WlsfStat *st, const char **why,
                             int mode) {
    static _Thread_local char why_buf[160]; // per thread: modules are created on many threads at once
    memset(st, 0, sizeof *st);
    st->in_bytes = bytes;
    st->mode = mode == 2 ? 2 : 1;
    *why = "not SPIR-V or malformed";
    if (!in || bytes < 20 || bytes % 4 || in[0] != 0x07230203) return NULL;
    size_t n = bytes / 4;
    uint32_t bound = in[3];
    if (!bound || bound > (1u << 22)) return NULL;
    uint32_t *def = calloc(bound, 4), *type = calloc(bound, 4), *out = NULL;
    struct WlsfVar v[WLSF_MAX_VARS];
    uint32_t nv = 0, sg[4], nsg = 0, entries = 0, model = 0, lx = 0, ly = 0, lz = 0, wgs = 0, lid = 0, lidx = 0;
    size_t mm_off = 0, fn_off = 0;
    bool size_id = false;
    struct WlsfSgid sp;
    memset(&sp, 0, sizeof sp);
    if (!def || !type) { *why = "out of memory"; goto done; }
    // pass 1: definitions, structure, refusals by opcode / decoration
    for (size_t o = 5; o < n;) {
        uint32_t op = in[o] & 0xffff, len = in[o] >> 16;
        if (!len || o + len > n) goto done;
        const uint32_t *w = in + o;
        if (spv_op_typed(op) && len >= 3 && w[2] < bound && w[1] < bound) { def[w[2]] = (uint32_t)o; type[w[2]] = w[1]; }
        else if (op >= 19 && op <= 38 && len >= 2 && w[1] < bound) def[w[1]] = (uint32_t)o;
        if (op == 15 && len >= 3) { entries++; model = w[1]; }
        if (op == 16 && len >= 3 && w[2] == 17 && len >= 6) { lx = w[3]; ly = w[4]; lz = w[5]; }
        if ((op == 16 || op == 331) && len >= 3 && w[2] == 38) size_id = true;
        if (op == 14 && !mm_off) {
            mm_off = o;
            if (len < 3 || w[1] != 0 || w[2] != 1) { *why = "memory model is not Logical GLSL450"; goto done; }
        }
        if (op == 54 && !fn_off) fn_off = o;
        if (op == 71 && len >= 4 && w[2] == 11) { // OpDecorate BuiltIn
            uint32_t b = w[3];
            if (b == 36) { if (nsg >= 4) { *why = "too many SubgroupSize variables"; goto done; } sg[nsg++] = w[1]; }
            if ((b >= 37 && b <= 41) || (b >= 4416 && b <= 4420)) {
                snprintf(why_buf, sizeof why_buf, "subgroup built-in %u (only SubgroupSize is rewritten)", b);
                *why = why_buf;
                goto done;
            }
            if (b == 25) wgs = w[1];
            if (b == 27) lid = lid ? ~0u : w[1];   // LocalInvocationId (twice: ~0 = no match, wls-fix-2 refuses)
            if (b == 29) lidx = lidx ? ~0u : w[1]; // LocalInvocationIndex
        }
        if (op == 72 && len >= 5 && w[3] == 11 && ((w[4] >= 36 && w[4] <= 41) || (w[4] >= 4416 && w[4] <= 4420))) {
            *why = "subgroup built-in as a struct member";
            goto done;
        }
        bool grp = (op >= 259 && op <= 271) || (op >= 333 && op <= 366 && op != 337 && op != 346) ||
                   (op >= 4421 && op <= 4432) || (op >= 5000 && op <= 5007) || op == 5110 || op == 5111 || op == 5296 ||
                   (op >= 5571 && op <= 5581) || op == 6221 || op == 6237 || (op >= 6401 && op <= 6408);
        if (grp) {
            snprintf(why_buf, sizeof why_buf, "group/subgroup opcode %u (only ShuffleXor 346 and Broadcast 337 are rewritten)", op);
            *why = why_buf;
            goto done;
        }
        if (op == 337 || op == 346) {
            uint64_t sc;
            if (len != 6 || !wlsf_const(in, def, bound, w[3], &sc) || sc != 3) { *why = "ShuffleXor/Broadcast with a scope other than a constant Subgroup"; goto done; }
            if (op == 346) st->shx++; else st->bcast++;
        }
        if (op == 59 && len >= 4 && w[3] == 4) { // OpVariable Workgroup
            if (nv >= WLSF_MAX_VARS) { *why = "more than 16 Workgroup variables"; goto done; }
            memset(&v[nv], 0, sizeof v[nv]);
            v[nv].var = w[2];
            v[nv].ptype = w[1];
            v[nv].var_off = (uint32_t)o;
            nv++;
        }
        o += len;
    }
    if (!mm_off || !fn_off) goto done;
    if (entries != 1) { *why = "not exactly one entry point"; goto done; }
    if (model != 5) { *why = "not a GLCompute entry point"; goto done; }
    if (size_id) { *why = "LocalSizeId (workgroup size from spec constants)"; goto done; }
    if (wgs) { // a WorkgroupSize built-in constant overrides LocalSize: plain constants only
        const uint32_t *d = wgs < bound && def[wgs] ? in + def[wgs] : NULL;
        uint64_t a, b, c;
        if (!d || (d[0] & 0xffff) != 44 || (d[0] >> 16) != 6 || !wlsf_const(in, def, bound, d[3], &a) ||
            !wlsf_const(in, def, bound, d[4], &b) || !wlsf_const(in, def, bound, d[5], &c)) {
            *why = "WorkgroupSize built-in is not a plain constant (spec-constant workgroup size)";
            goto done;
        }
        lx = (uint32_t)a; ly = (uint32_t)b; lz = (uint32_t)c;
    }
    if (!lx || !ly || !lz || (uint64_t)lx * ly * lz > 4096) { *why = "no constant LocalSize (or invocation count > 4096)"; goto done; }
    st->n = lx * ly * lz;
    // grown length: mode 1 one slot per invocation, mode 2 one slot per subgroup down to 4 lanes
    uint32_t tl = mode == 2 ? (st->n / 4 ? st->n / 4 : 1) : st->n;
    st->new_len = tl;
    for (uint32_t i = 0; i < nsg; i++) { // the SubgroupSize variable: an Input uint
        const uint32_t *d = sg[i] < bound && def[sg[i]] ? in + def[sg[i]] : NULL;
        const uint32_t *pt = d && (d[0] & 0xffff) == 59 && d[1] < bound && def[d[1]] ? in + def[d[1]] : NULL;
        const uint32_t *it = pt && (pt[0] & 0xffff) == 32 && pt[3] < bound && def[pt[3]] ? in + def[pt[3]] : NULL;
        if (!it || (it[0] & 0xffff) != 21 || it[2] != 32) { *why = "SubgroupSize is not an Input uint variable"; goto done; }
    }
    // Workgroup variables: scalar / vector, or an array with a constant length; estimate the memory after growth
    uint64_t shared = 0;
    for (uint32_t i = 0; i < nv; i++) {
        const uint32_t *pt = v[i].ptype < bound && def[v[i].ptype] ? in + def[v[i].ptype] : NULL;
        if (!pt || (pt[0] & 0xffff) != 32 || (pt[0] >> 16) < 4) goto done;
        const uint32_t *d = pt[3] < bound && def[pt[3]] ? in + def[pt[3]] : NULL;
        uint32_t top = d ? d[0] & 0xffff : 0;
        uint64_t a, s;
        if (top >= 20 && top <= 23) { // bool, int, float, vector
            if (!(s = wlsf_size(in, def, bound, pt[3], &a, 0))) { *why = "Workgroup variable of unknown size"; goto done; }
            shared += (s + 15) / 16 * 16;
            continue;
        }
        uint64_t len;
        if (top != 28 || (d[0] >> 16) < 4) {
            snprintf(why_buf, sizeof why_buf, "Workgroup variable %%%u is not a scalar, vector or array (opcode %u)", v[i].var, top);
            *why = why_buf;
            goto done;
        }
        if (!wlsf_const(in, def, bound, d[3], &len) || !len || len > (1u << 20)) {
            snprintf(why_buf, sizeof why_buf, "Workgroup array %%%u has a spec-constant or invalid length", v[i].var);
            *why = why_buf;
            goto done;
        }
        if (!(s = wlsf_size(in, def, bound, d[2], &a, 0))) { *why = "Workgroup array element of unknown size"; goto done; }
        v[i].atype = pt[3];
        v[i].elem = d[2];
        v[i].lenc = d[3];
        v[i].len = (uint32_t)len;
        v[i].esize = (s + a - 1) / a * a;
        v[i].grow = len < tl;
        shared += ((v[i].grow ? tl : len) * v[i].esize + 15) / 16 * 16;
    }
    if (shared > WLSF_SHARED_LIMIT) {
        snprintf(why_buf, sizeof why_buf, "shared memory after growth %llu B > %u B", (unsigned long long)shared, WLSF_SHARED_LIMIT);
        *why = why_buf;
        goto done;
    }
    // pass 2: every id operand that names a SubgroupSize variable, a Workgroup array variable, its pointer type or its
    // array type must be one of the allowed uses; also count the loads, record ArrayStride decorations and loops
    for (size_t o = 5; o < n; o += in[o] >> 16) {
        const uint32_t *w = in + o;
        uint32_t op = w[0] & 0xffff, len = w[0] >> 16;
        if (op == 71 && len >= 4 && w[2] == 6)
            for (uint32_t i = 0; i < nv; i++)
                if (v[i].atype && w[1] == v[i].atype) { v[i].stride = w[3]; v[i].stride_off = (uint32_t)o; }
        if (op == 246 && len >= 4 && (w[3] & 1)) st->loops++;
        for (uint32_t k = 1; k < len; k++) {
            uint32_t id = w[k];
            if (!wlsf_is_id(w, k)) continue;
            for (uint32_t i = 0; i < nsg; i++) {
                if (id != sg[i]) continue;
                if (op == 61 && k == 3) { st->loads++; continue; }
                if (op == 59 && k == 2) continue;
                snprintf(why_buf, sizeof why_buf, "SubgroupSize variable used by opcode %u (only OpLoad is rewritten)", op);
                *why = why_buf;
                goto done;
            }
            for (uint32_t i = 0; i < nv; i++) {
                if (!v[i].atype) continue;
                bool ok = true;
                if (id == v[i].var) ok = (op == 59 && k == 2) || ((op == 65 || op == 66) && k == 3 && len >= 5);
                else if (id == v[i].ptype) ok = (op == 32 && k == 1) || (op == 59 && k == 1 && o == v[i].var_off);
                else if (id == v[i].atype) ok = (op == 28 && k == 1) || (op == 32 && k == 3 && w[1] == v[i].ptype);
                else continue;
                if (!ok) {
                    snprintf(why_buf, sizeof why_buf, "Workgroup array %%%u (or its type) used by opcode %u (only access chains are allowed)", v[i].var, op);
                    *why = why_buf;
                    goto done;
                }
            }
        }
    }
    // Only a shader that reads SubgroupSize can follow a 1-lane subgroup; ShuffleXor / Broadcast without it hard-wire a
    // lane layout (e.g. quad neighbours), where identity would change the result.
    if (!st->loads) { *why = st->shx || st->bcast ? "ShuffleXor/Broadcast without any SubgroupSize load (lane layout hard-wired)"
                                                  : "no SubgroupSize load (nothing to rewrite)"; goto done; }
    if (mode == 2) { // wls-fix-2: exactly the slot / count / reduce-loop pattern, else left as it is
        if (nsg != 1) { *why = "mode sgid: not exactly one SubgroupSize variable"; goto done; }
        const char *r = wlsf_sgid_plan(in, n, def, type, bound, v, nv, sg[0], lid == ~0u ? 0 : lid, lidx == ~0u ? 0 : lidx,
                                       lx, lz, st->n, fn_off, &sp, why_buf, sizeof why_buf);
        if (r) { *why = r; goto done; }
        st->init_from = sp.init_val;
    }
    // ids: the constant 1 per SubgroupSize load type (uint), per grown array a length constant, array and pointer type
    uint32_t next = bound, one = 0, one_type = 0;
    bool one_new = false;
    if (st->loads && mode != 2) {
        const uint32_t *d = in + def[sg[0]];
        one_type = in[def[d[1]] + 3];
        for (uint32_t k = 1; k < bound && !one; k++) {
            uint64_t x;
            if (type[k] == one_type && def[k] < fn_off && wlsf_const(in, def, bound, k, &x) && x == 1) one = k;
        }
        if (!one) { one = next++; one_new = true; }
    }
    for (uint32_t i = 0; i < nv; i++) {
        if (!v[i].grow) continue;
        uint32_t lt = type[v[i].lenc];
        if (!lt || lt >= bound || !def[lt] || (in[def[lt]] & 0xffff) != 21) { *why = "array length constant is not an integer"; goto done; }
        for (uint32_t k = 1; k < bound && !v[i].new_len_c; k++) {
            uint64_t x;
            if (type[k] == lt && def[k] < v[i].var_off && wlsf_const(in, def, bound, k, &x) && x == tl) v[i].new_len_c = k;
        }
        for (uint32_t j = 0; j < i && !v[i].new_len_c; j++) // a constant made for an earlier array of the same length type
            if (v[j].grow && v[j].len_c_new && type[v[j].lenc] == lt) v[i].new_len_c = v[j].new_len_c;
        if (!v[i].new_len_c) { v[i].new_len_c = next++; v[i].len_c_new = true; }
        v[i].new_atype = next++;
        v[i].new_ptype = next++;
        st->old_len[st->grown++] = v[i].len;
    }
    if (mode == 2) {
        sp.sgid = next++;
        sp.nsg = next++;
        if (sp.zero_new) sp.zero = next++;
    }
    // pass 3: write the rewritten module
    // per grown array: constant 5, array 4, pointer 4, ArrayStride 4; mode 2: entry point 2, decorations 8, variables 8,
    // constant 0 4, per ShuffleXor 15 (load 4, compare 5, select 6)
    out = malloc((n + 8 + 17 * (size_t)nv + (mode == 2 ? 22 + 15 * (size_t)st->shx : 0)) * 4);
    if (!out) { *why = "out of memory"; goto done; }
    memcpy(out, in, 5 * 4);
    size_t w = 5;
    bool emitted_c[WLSF_MAX_VARS] = {0};
    for (size_t o = 5; o < n;) {
        const uint32_t *x = in + o;
        uint32_t op = x[0] & 0xffff, len = x[0] >> 16;
        if (o == fn_off && one_new) { out[w++] = (4u << 16) | 43u; out[w++] = one_type; out[w++] = one; out[w++] = 1; }
        if (o == fn_off && sp.zero_new) { out[w++] = (4u << 16) | 43u; out[w++] = sp.zero_type; out[w++] = sp.zero; out[w++] = 0; }
        for (uint32_t i = 0; i < nv; i++) {
            if (!v[i].grow || o != v[i].var_off) continue;
            if (v[i].len_c_new && !emitted_c[i]) {
                bool done_before = false; // shared with an earlier array: emitted there already
                for (uint32_t j = 0; j < i; j++) if (v[j].grow && v[j].new_len_c == v[i].new_len_c && emitted_c[j]) done_before = true;
                if (!done_before) {
                    const uint32_t *lt = in + def[type[v[i].lenc]];
                    bool wide = lt[2] == 64;
                    out[w++] = ((wide ? 5u : 4u) << 16) | 43u;
                    out[w++] = type[v[i].lenc];
                    out[w++] = v[i].new_len_c;
                    out[w++] = tl;
                    if (wide) out[w++] = 0;
                }
                emitted_c[i] = true;
            }
            out[w++] = (4u << 16) | 28u; out[w++] = v[i].new_atype; out[w++] = v[i].elem; out[w++] = v[i].new_len_c;
            out[w++] = (4u << 16) | 32u; out[w++] = v[i].new_ptype; out[w++] = 4u; out[w++] = v[i].new_atype;
        }
        bool sgload = false;
        for (uint32_t i = 0; op == 61 && len >= 4 && i < nsg; i++) sgload |= x[3] == sg[i];
        if (mode == 2 && op == 346) { // r = ShuffleXor v, m -> t = ShuffleXor v, m; s = SubgroupSize; r = Select(s > m, t, v)
            uint32_t t = next++, sz = next++, c = next++;
            out[w++] = (6u << 16) | 346u; out[w++] = x[1]; out[w++] = t; out[w++] = x[3]; out[w++] = x[4]; out[w++] = x[5];
            out[w++] = (4u << 16) | 61u; out[w++] = sp.sgtype; out[w++] = sz; out[w++] = sp.sgvar;
            out[w++] = (5u << 16) | 172u; out[w++] = sp.booltype; out[w++] = c; out[w++] = sz; out[w++] = x[5];
            out[w++] = (6u << 16) | 169u; out[w++] = x[1]; out[w++] = x[2]; out[w++] = c; out[w++] = t; out[w++] = x[4];
        } else if (mode == 2 && o == sp.ep_off) { // the entry point gets the two new Input variables
            out[w++] = ((len + 2) << 16) | 15u;
            memcpy(out + w, x + 1, (len - 1) * 4);
            w += len - 1;
            out[w++] = sp.sgid; out[w++] = sp.nsg;
        } else if (mode == 2 && (o == sp.slot_off || o == sp.count_off)) { // UDiv -> OpLoad SubgroupId / NumSubgroups
            out[w++] = (4u << 16) | 61u; out[w++] = x[1]; out[w++] = x[2]; out[w++] = o == sp.slot_off ? sp.sgid : sp.nsg;
        } else if (mode == 2 && o == sp.init_off && sp.init_val == 1) { // reduce loop start 1 -> 0
            out[w++] = x[0]; out[w++] = x[1]; out[w++] = sp.zero;
        } else if (sgload && mode != 2) { // OpLoad %uint %SubgroupSize -> OpCopyObject %uint %uint_1
            out[w++] = (4u << 16) | 83u; out[w++] = x[1]; out[w++] = x[2]; out[w++] = one;
        } else if ((op == 337 || op == 346) && mode != 2) { // Broadcast / ShuffleXor -> OpCopyObject <type> <value>
            out[w++] = (4u << 16) | 83u; out[w++] = x[1]; out[w++] = x[2]; out[w++] = x[4];
        } else {
            memcpy(out + w, x, len * 4);
            if (op == 246 && len >= 4) out[w + 3] &= ~1u;           // OpLoopMerge: Unroll off
            for (uint32_t i = 0; op == 59 && i < nv; i++)           // the grown variable gets its new pointer type
                if (v[i].grow && o == v[i].var_off) out[w + 1] = v[i].new_ptype;
            w += len;
            for (uint32_t i = 0; op == 71 && i < nv; i++)           // ArrayStride copied to the new array type
                if (v[i].grow && v[i].stride_off == o) {
                    out[w++] = (4u << 16) | 71u; out[w++] = v[i].new_atype; out[w++] = 6u; out[w++] = v[i].stride;
                }
            if (mode == 2 && o == sp.dec_off) { // BuiltIn SubgroupId (40), NumSubgroups (38)
                out[w++] = (4u << 16) | 71u; out[w++] = sp.sgid; out[w++] = 11u; out[w++] = 40u;
                out[w++] = (4u << 16) | 71u; out[w++] = sp.nsg; out[w++] = 11u; out[w++] = 38u;
            }
            if (mode == 2 && o == sp.sgvar_off) { // two Input uint variables of the SubgroupSize pointer type
                out[w++] = (4u << 16) | 59u; out[w++] = sp.sgptype; out[w++] = sp.sgid; out[w++] = 1u;
                out[w++] = (4u << 16) | 59u; out[w++] = sp.sgptype; out[w++] = sp.nsg; out[w++] = 1u;
            }
        }
        o += len;
    }
    out[3] = next;
    // the wls-clamp-1 clamp on the rewritten module: every dynamic index into a Workgroup array <= the NEW length - 1
    uint32_t cst[4];
    size_t cb = 0;
    uint32_t *clamped = spv_wls_clamp(out, w * 4, &cb, cst);
    if (clamped) {
        free(out);
        out = clamped;
        w = cb / 4;
        st->clamped = cst[0];
    }
    *out_bytes = w * 4;
    st->out_bytes = w * 4;
    *why = NULL;
done:
    free(def);
    free(type);
    if (*why) { free(out); out = NULL; }
    return out;
}

// One module through the fix (from compat_patch_spirv / the inline compute path): orig = the app's code (hash
// filter), cur = the code so far. NULL = not listed or left as it is (logged).
static uint32_t *wls_fix_module(const uint32_t *orig, size_t orig_bytes, const uint32_t *cur, size_t cur_bytes,
                                size_t *out_bytes) {
    uint32_t h = spv_hash(orig, orig_bytes);
    if (!wls_fix_wanted(h)) return NULL;
    struct WlsfStat st;
    const char *why = NULL;
    uint32_t *out = spv_wls_fix(cur, cur_bytes, out_bytes, &st, &why, g_wlsf_mode);
    if (!out) {
        atomic_fetch_add(&g_wlsf_left, 1);
        if (atomic_fetch_add(&g_wlsf_left_logged, 1) < 64)
            fprintf(stderr, "mali-compat: wls-fix: spv %08x left as it is: %s\n", h, why ? why : "?");
        return NULL;
    }
    atomic_fetch_add(&g_wlsf_rewritten, 1);
    if (atomic_fetch_add(&g_wlsf_logged, 1) < 64) {
        char gl[160] = "";
        for (uint32_t i = 0; i < st.grown; i++)
            snprintf(gl + strlen(gl), sizeof gl - strlen(gl), "%s%u -> %u", i ? ", " : " (", st.old_len[i], st.new_len);
        if (st.grown) snprintf(gl + strlen(gl), sizeof gl - strlen(gl), ")");
        if (st.mode == 2)
            fprintf(stderr, "mali-compat: wls-fix: spv %08x rewritten (mode sgid): local size %u, ShuffleXor guarded %u, "
                    "Broadcast kept %u, slot -> SubgroupId, count -> NumSubgroups, loop start %u -> 0, arrays grown %u%s, "
                    "clamped chains %u, loops un-unrolled %u, %zu -> %zu B\n", h, st.n, st.shx, st.bcast, st.init_from,
                    st.grown, gl, st.clamped, st.loops, st.in_bytes, st.out_bytes);
        else
            fprintf(stderr, "mali-compat: wls-fix: spv %08x rewritten: local size %u, SubgroupSize loads %u, ShuffleXor %u, "
                    "Broadcast %u, arrays grown %u%s, clamped chains %u, loops un-unrolled %u, %zu -> %zu B\n", h, st.n,
                    st.loads, st.shx, st.bcast, st.grown, gl, st.clamped, st.loops, st.in_bytes, st.out_bytes);
    }
    return out;
}
// END wls-fix-1 (+ wls-fix-2 mode sgid)


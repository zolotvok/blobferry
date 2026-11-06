// SPDX-License-Identifier: GPL-3.0-or-later
#define _GNU_SOURCE
#include <dlfcn.h> // dladdr (K-WC1 wait callers)
// Mali Compatibility Layer for DXVK / Proton (VK_LAYER_DROIDDECK_mali_compat)
// Implements feature masking, polygon mode override, SPIR-V clip/cull distance stripping,
// BC1-BC7 GPU compute texture emulation via PanVK shaders, and VK_EXT_memory_budget.
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <string.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdarg.h>
#include <errno.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/sysmacros.h>
#include <sys/ioctl.h>
#include <fcntl.h>
#include <time.h>
#include <unistd.h>
#include <vulkan/vulkan.h>
#include <vulkan/vk_layer.h>

#include "bc/panvk_bc.h"

#define EXPORT __attribute__((visibility("default")))
#define LAYER_NAME "VK_LAYER_DROIDDECK_mali_compat"

static int g_debug = 0;

// pNext helpers and the page size (used by the placed-map emulation and the feature/property queries).
static size_t page_size(void) {
    static size_t ps;
    if (!ps) ps = (size_t)sysconf(_SC_PAGESIZE);
    return ps;
}

static const void *find_struct(const void *chain, VkStructureType type) {
    for (const VkBaseInStructure *s = chain; s; s = s->pNext)
        if (s->sType == type) return s;
    return NULL;
}

// Unlinks the first struct of `type` from a chain that starts after `head`; returns {prev, node} to relink later.
struct unlinked { VkBaseOutStructure *prev, *node; };
static struct unlinked unlink_struct(void *head, VkStructureType type) {
    for (VkBaseOutStructure *prev = head, *s = prev->pNext; s; prev = s, s = s->pNext) {
        if (s->sType == type) {
            prev->pNext = s->pNext;
            return (struct unlinked){prev, s};
        }
    }
    return (struct unlinked){NULL, NULL};
}
static void relink(struct unlinked u) {
    if (u.prev) u.prev->pNext = u.node;
}

#define LOG(...) do { if (g_debug) { fprintf(stderr, "mali-compat: " __VA_ARGS__); fputc('\n', stderr); } } while (0)

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
static bool g_bc_afrc = true; // MALI_COMPAT_BC_AFRC=0 turns the fixed-rate BC shadows off
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

static uint32_t *compat_patch_spirv(const uint32_t *inWords, size_t inBytes, size_t *pOutBytes) {
    uint32_t *out = compat_patch_spirv_impl(inWords, inBytes, pOutBytes);
    uint32_t *fixed = g_spv_fix ? spv_fix_u64_to_float(out ? out : inWords, out ? *pOutBytes : inBytes) : NULL;
    if (fixed) {
        *pOutBytes = g_fix_out_bytes;
        free(out);
        out = fixed;
    }
    fixed = g_spv_fix ? spv_fix_broadcast_first(out ? out : inWords, out ? *pOutBytes : inBytes) : NULL;
    if (fixed) {
        *pOutBytes = g_fix_out_bytes;
        free(out);
        out = fixed;
    }
    fixed = g_spv_dontunroll ? spv_dont_unroll(out ? out : inWords, out ? *pOutBytes : inBytes) : NULL;
    if (fixed) {
        if (!out) *pOutBytes = inBytes;
        free(out);
        out = fixed;
    }
    if (g_wls_clamp) { // wls-clamp-1: last, on the code so far; the hash filter uses the app's code
        size_t wb = 0;
        fixed = wls_clamp_module(inWords, inBytes, out ? out : inWords, out ? *pOutBytes : inBytes, &wb);
        if (fixed) {
            *pOutBytes = wb;
            free(out);
            out = fixed;
        }
    }
    if (out) dump_spirv(inWords, inBytes, out, *pOutBytes);
    return out;
}

// dxil-spirv converts a 32-bit unsigned to float through 64 bits (OpUConvert to u64, then OpConvertUToF). The vendor
// compiler aborts the whole process on that ("LLVM ERROR: Cannot select: i64 = <<Unknown Target Node #405>> ...
// AssertZext ... i26", Control, 2026-10-30). A zero extension does not change the value, so OpConvertUToF of the
// narrow operand gives the same float; the OpUConvert, now unused, is dropped (the dead zero extension aborts the
// compiler too). Returns a patched copy (size in g_fix_out_bytes) or NULL. [Claude Opus 5.5]
// vkd3d marks descriptor offsets it loads from push constants as uniform with OpGroupNonUniformBroadcastFirst. The
// vendor compiler aborted on such a compute shader (Control, 2026-10-30, the same "Cannot select ... Constant 32"
// as above; it lowers the ballot through 64 bits). When the value is provably the same on every lane (built only from
// push-constant loads, constants and integer arithmetic on them), BroadcastFirst(x) == x, so it becomes OpCopyObject.
// Game code's own WaveReadLaneFirst on varying values is left alone. Returns a patched copy (size in
// g_fix_out_bytes) or NULL. [Claude Opus 5.5]
#define SpvOpBroadcastFirst_ 338
static uint32_t *spv_fix_broadcast_first(const uint32_t *in, size_t bytes) {
    if (!in || bytes < 20 || bytes % 4 || in[0] != 0x07230203) return NULL;
    size_t n = bytes / 4;
    uint32_t bound = in[3];
    if (!bound || bound > (1u << 22)) return NULL;
    uint32_t *def = calloc(bound, 4);      // result id -> word offset of its instruction
    uint8_t *pc_ptr = calloc(bound, 1);    // pointer ids into PushConstant (variables, access chains)
    if (!def || !pc_ptr) { free(def); free(pc_ptr); return NULL; }
    bool any = false;
    for (size_t o = 5; o < n;) {
        uint16_t op = in[o] & 0xffff, len = in[o] >> 16;
        if (!len || o + len > n) break;
        // result id position: types/constants/variables/most ops have (type, result) at 1, 2; OpType* at 1
        if (op >= 19 && op <= 39 && len >= 2 && in[o + 1] < bound) def[in[o + 1]] = (uint32_t)o;
        else if (len >= 3 && in[o + 2] < bound && (op == 43 || op == 44 || op == 59 || op == 61 || op == 65 || op == 66 ||
                 op == 83 || (op >= 124 && op <= 132) || (op >= 128 && op <= 200) || op == SpvOpBroadcastFirst_))
            def[in[o + 2]] = (uint32_t)o;
        if (op == 59 && len >= 4 && in[o + 3] == 9 && in[o + 2] < bound) pc_ptr[in[o + 2]] = 1; // OpVariable PushConstant
        if ((op == 65 || op == 66) && len >= 4 && in[o + 3] < bound && pc_ptr[in[o + 3]] && in[o + 2] < bound)
            pc_ptr[in[o + 2]] = 1;
        if (op == SpvOpBroadcastFirst_) any = true;
        o += len;
    }
    uint32_t *out = NULL, fixes = 0;
    if (any && (out = malloc(bytes))) {
        memcpy(out, in, bytes);
        size_t w = 5;
        for (size_t o = 5; o < n;) {
            uint16_t op = in[o] & 0xffff, len = in[o] >> 16;
            if (!len || o + len > n) { memcpy(out + w, in + o, (n - o) * 4); w += n - o; break; }
            bool uniform = false;
            if (op == SpvOpBroadcastFirst_ && len == 5) {
                // walk the operand: a small stack of ids that must all be uniform
                uint32_t stack[16], sp = 0, seen = 0;
                stack[sp++] = in[o + 4];
                uniform = true;
                while (sp && uniform && seen++ < 32) {
                    uint32_t id = stack[--sp];
                    if (id >= bound || !def[id]) { uniform = false; break; }
                    const uint32_t *d = in + def[id];
                    uint16_t dop = d[0] & 0xffff;
                    if (dop == 43 || dop == 44) continue;                                  // constants
                    if (dop == 61 && d[3] < bound && pc_ptr[d[3]]) continue;               // load from push constants
                    if ((dop == 128 || dop == 130 || dop == 132 || dop == 194 || dop == 196 || dop == 199) && sp + 2 <= 16) {
                        stack[sp++] = d[3]; stack[sp++] = d[4]; continue;                  // IAdd ISub IMul shifts BitwiseAnd
                    }
                    uniform = false;
                }
            }
            if (uniform) {
                out[w++] = (4u << 16) | 83u; // OpCopyObject type result value
                out[w++] = in[o + 1];
                out[w++] = in[o + 2];
                out[w++] = in[o + 4];
                fixes++;
            } else {
                memcpy(out + w, in + o, len * 4);
                w += len;
            }
            o += len;
        }
        g_fix_out_bytes = w * 4;
        if (!fixes) { free(out); out = NULL; }
    }
    free(def);
    free(pc_ptr);
    if (fixes) {
        static _Atomic uint32_t logged;
        if (atomic_fetch_add(&logged, 1) < 4)
            fprintf(stderr, "mali-compat: %u uniform BroadcastFirst replaced in a shader (vendor compiler abort)\n", fixes);
    }
    return out;
}

#define SpvOpTypeInt_ 21
#define SpvOpTypeVector_ 23
#define SpvOpConvertUToF_ 112
#define SpvOpUConvert_ 113
static uint32_t *spv_fix_u64_to_float(const uint32_t *in, size_t bytes) {
    if (!in || bytes < 20 || bytes % 4 || in[0] != 0x07230203) return NULL;
    size_t n = bytes / 4;
    uint32_t bound = in[3];
    if (!bound || bound > (1u << 22)) return NULL;
    uint8_t *wide = calloc(bound, 1);   // type id -> 1 if a 64-bit int (or vector of them)
    uint32_t *src = calloc(bound, 4);   // OpUConvert result (64-bit) -> its operand
    if (!wide || !src) { free(wide); free(src); return NULL; }
    bool any = false;
    for (size_t o = 5; o < n;) {
        uint16_t op = in[o] & 0xffff, len = in[o] >> 16;
        if (!len || o + len > n) break;
        if (op == SpvOpTypeInt_ && len >= 4 && in[o + 1] < bound && in[o + 2] == 64) wide[in[o + 1]] = 1;
        else if (op == SpvOpTypeVector_ && len >= 4 && in[o + 1] < bound && in[o + 2] < bound && wide[in[o + 2]]) wide[in[o + 1]] = 1;
        else if (op == SpvOpUConvert_ && len == 4 && in[o + 1] < bound && in[o + 2] < bound && wide[in[o + 1]]) {
            src[in[o + 2]] = in[o + 3];
            any = true;
        }
        o += len;
    }
    uint32_t *out = NULL;
    uint32_t fixes = 0;
    if (any && (out = malloc(bytes))) {
        memcpy(out, in, bytes);
        for (size_t o = 5; o < n;) {
            uint16_t op = out[o] & 0xffff, len = out[o] >> 16;
            if (!len || o + len > n) break;
            if (op == SpvOpConvertUToF_ && len == 4 && out[o + 3] < bound && src[out[o + 3]]) {
                out[o + 3] = src[out[o + 3]];
                fixes++;
            }
            o += len;
        }
        if (!fixes) { free(out); out = NULL; }
        else g_fix_out_bytes = bytes;
    }
    if (out) {
        // Drop the OpUConvert instructions nothing uses any more: the vendor compiler aborts on the dead zero
        // extension as well. Any word equal to the id counts as a use, so a literal that happens to match only
        // keeps the instruction (safe).
        uint32_t *uses = calloc(bound, 4);
        uint32_t *packed = uses ? malloc(bytes) : NULL;
        if (packed) {
            for (size_t o = 5; o < n;) {
                uint16_t op = out[o] & 0xffff, len = out[o] >> 16;
                if (!len || o + len > n) break;
                // literals and debug/annotation targets are not uses: OpSource(3), OpSourceExtension(4),
                // OpName(5), OpMemberName(6), OpString(7), OpLine(8), OpConstant(43), OpSpecConstant(50),
                // OpDecorate(71), OpMemberDecorate(72), OpModuleProcessed(330)
                bool meta = op == 3 || op == 4 || op == 5 || op == 6 || op == 7 || op == 8 || op == 43 || op == 50 ||
                            op == 71 || op == 72 || op == 330;
                for (uint16_t k = 1; k < len && !meta; k++) {
                    bool def = op == SpvOpUConvert_ && k == 2;
                    if (!def && out[o + k] < bound) uses[out[o + k]]++;
                }
                o += len;
            }
            size_t w = 0;
            memcpy(packed, out, 5 * 4);
            w = 5;
            for (size_t o = 5; o < n;) {
                uint16_t op = out[o] & 0xffff, len = out[o] >> 16;
                if (!len || o + len > n) { memcpy(packed + w, out + o, (n - o) * 4); w += n - o; break; }
                bool drop = op == SpvOpUConvert_ && len == 4 && out[o + 2] < bound && src[out[o + 2]] && !uses[out[o + 2]];
                // names and decorations of a dropped OpUConvert go with it
                if ((op == 5 || op == 71) && len >= 2 && out[o + 1] < bound && src[out[o + 1]] && !uses[out[o + 1]]) drop = true;
                if (!drop) { memcpy(packed + w, out + o, len * 4); w += len; }
                o += len;
            }
            free(out);
            out = packed;
            g_fix_out_bytes = w * 4;
        }
        free(uses);
    }
    free(wide);
    free(src);
    if (fixes) {
        static _Atomic uint32_t logged;
        if (atomic_fetch_add(&logged, 1) < 4)
            fprintf(stderr, "mali-compat: %u u64-to-float conversion(s) narrowed in a shader (vendor compiler abort)\n", fixes);
    }
    return out;
}

// Strips ClipDistance/CullDistance (the vendor driver has neither feature). The capabilities and the BuiltIn
// decorations go, and:
// - a variable decorated ClipDistance/CullDistance (dxil-spirv style, also arrayed) becomes Private, so writes to it
//   are dead. Every pointer derived from it (access chains) gets a Private pointer type as well, reusing an existing
//   OpTypePointer or adding one right after the original;
// - a ClipDistance/CullDistance member of a block (glslang/DXVK gl_PerVertex, also arrays of blocks such as gl_in[])
//   is removed from the struct, and the later members' names, decorations and access-chain indices are renumbered.
//   Chains into a removed member are dropped, and so are stores and OpCopyMemory through them; loads through them
//   become OpUndef.
// Returns NULL when the module needs no change, or when it uses a removed member in any other way: the module is then
// left as it is (the pipeline fails rather than running a wrong shader).
// Rewritten [Claude Opus 5.5] 2026-10-26: the first version changed the shared pointer type in place and left
// OpNops longer than one word; all 64 DXVK modules it produced failed spirv-val (access-chain storage mismatch).
// Member handling rewritten [Claude Opus 5.5] 2026-10-27 (Review 10): the 02:20 version did not renumber, did not see
// a member index behind an array index (gl_in[i].gl_ClipDistance) and left loads through removed members, which then
// referenced undefined ids. Cases: test/spirv-patch-cases/, run by test/spirv-patch-check.sh.
#define SpvOpUndef 1
#define SpvOpName 5
#define SpvOpLine 8
#define SpvOpTypeArray 28
#define SpvOpTypeRuntimeArray 29
#define SpvOpTypeForwardPointer 39
#define SpvOpFunction 54
#define SpvOpFunctionParameter 55
#define SpvOpFunctionEnd 56
#define SpvOpLoad 61
#define SpvOpCopyMemory 63
#define SpvOpAccessChain 65
#define SpvOpInBoundsAccessChain 66
#define SpvOpPtrAccessChain 67
#define SpvOpInBoundsPtrAccessChain 70

struct SpvPatch {
    const uint32_t *w;
    uint32_t bound;
    uint32_t *struct_at;       // OpTypeStruct id -> its word offset
    uint32_t *elem;            // OpTypeArray / OpTypeRuntimeArray id -> element type
    uint32_t *ptr_pointee;     // OpTypePointer id -> pointee type
    uint32_t *id_type;         // variable / access chain / function parameter id -> its pointer type
    uint32_t *struct_bad_mask; // struct id -> members to remove
    uint32_t *const_val, *const_type;
    uint8_t *is_const;
    uint32_t memo[16][3];      // spv_find_const results: type, value, id
    int memo_n;
};

static bool spv_is_chain(uint16_t op) {
    return op == SpvOpAccessChain || op == SpvOpInBoundsAccessChain || op == SpvOpPtrAccessChain ||
           op == SpvOpInBoundsPtrAccessChain;
}

// A struct with removed members, or an array of one.
static bool spv_has_bad_struct(const struct SpvPatch *c, uint32_t t) {
    for (int depth = 0; depth < 8 && t && t < c->bound; depth++) {
        if (c->struct_bad_mask[t]) return true;
        t = c->elem[t];
    }
    return false;
}

static uint32_t spv_find_const(struct SpvPatch *c, uint32_t type, uint32_t value) {
    for (int i = 0; i < c->memo_n; i++)
        if (c->memo[i][0] == type && c->memo[i][1] == value) return c->memo[i][2];
    uint32_t found = 0;
    for (uint32_t id = 1; id < c->bound && !found; id++)
        if (c->is_const[id] && c->const_type[id] == type && c->const_val[id] == value) found = id;
    if (c->memo_n < 16) {
        c->memo[c->memo_n][0] = type;
        c->memo[c->memo_n][1] = value;
        c->memo[c->memo_n++][2] = found;
    }
    return found;
}

// Follows an access chain's indices through the types. Returns 1 when it reaches a removed member, -1 when it cannot
// be followed (a non-constant or out-of-range index into a struct with removed members, or no constant for the
// renumbered index), else 0. With `out` (the copied instruction), indices of later members are renumbered.
static int spv_walk_chain(struct SpvPatch *c, const uint32_t *in, uint16_t len, uint32_t *out) {
    uint16_t op = in[0] & 0xFFFFu;
    uint32_t base = in[3];
    if (base >= c->bound || c->id_type[base] >= c->bound) return 0;
    uint32_t t = c->ptr_pointee[c->id_type[base]];
    // The Ptr variants' first index steps over elements of the base pointer and does not change the type.
    uint16_t k = (op == SpvOpPtrAccessChain || op == SpvOpInBoundsPtrAccessChain) ? 5 : 4;
    for (; k < len && t && t < c->bound; k++) {
        uint32_t mask = c->struct_bad_mask[t];
        if (c->struct_at[t]) {
            const uint32_t *s = &c->w[c->struct_at[t]];
            uint32_t members = (s[0] >> 16) - 2, idx_id = in[k];
            if (idx_id >= c->bound || !c->is_const[idx_id] || c->const_val[idx_id] >= members) return mask ? -1 : 0;
            uint32_t m = c->const_val[idx_id];
            if (mask) {
                if (m < 32 && (mask & (1u << m))) return 1;
                uint32_t below = m >= 32 ? mask : (mask & ((1u << m) - 1));
                if (below) {
                    uint32_t nid = spv_find_const(c, c->const_type[idx_id], m - (uint32_t)__builtin_popcount(below));
                    if (!nid) return -1;
                    if (out) out[k] = nid;
                }
            }
            t = s[2 + m];
        } else if (c->elem[t]) {
            t = c->elem[t];
        } else {
            break; // vector or matrix component: no struct below
        }
    }
    return 0;
}

static uint32_t *compat_patch_spirv_impl(const uint32_t *inWords, size_t inBytes, size_t *pOutBytes) {
    if (!inWords || inBytes < 20 || inBytes % 4) return NULL;
    size_t wordCount = inBytes / 4;
    if (inWords[0] != 0x07230203) return NULL;

    bool hasClipOrCull = false;
    for (size_t o = 5; o < wordCount;) {
        uint16_t op = inWords[o] & 0xFFFFu, len = inWords[o] >> 16;
        if (len == 0 || o + len > wordCount || op != SpvOpCapability) break;
        if (inWords[o + 1] == SpvCapabilityClipDistance || inWords[o + 1] == SpvCapabilityCullDistance) hasClipOrCull = true;
        o += len;
    }
    if (!hasClipOrCull) return NULL;

    const uint32_t version = inWords[1], bound = inWords[3];
    if (bound == 0 || bound > (1u << 22)) return NULL;
    struct SpvPatch c = {.w = inWords, .bound = bound};
    uint32_t *ptr_storage = calloc(bound, sizeof(uint32_t)); // storage class + 1 for OpTypePointer ids
    uint32_t *priv_ptr = calloc(bound, sizeof(uint32_t));    // pointee -> Private pointer id (existing or new)
    uint8_t *need_new = calloc(bound, 1);                    // pointee needs a new Private pointer type
    uint8_t *bad = calloc(bound, 1);                         // clip/cull variables and pointers into them
    uint8_t *bad_ptr = calloc(bound, 1);                     // pointers into removed struct members
    c.struct_at = calloc(bound, sizeof(uint32_t));
    c.elem = calloc(bound, sizeof(uint32_t));
    c.ptr_pointee = calloc(bound, sizeof(uint32_t));
    c.id_type = calloc(bound, sizeof(uint32_t));
    c.struct_bad_mask = calloc(bound, sizeof(uint32_t));
    c.const_val = calloc(bound, sizeof(uint32_t));
    c.const_type = calloc(bound, sizeof(uint32_t));
    c.is_const = calloc(bound, 1);
    uint32_t *words = malloc(inBytes + 64 * 16);             // room for up to 64 added OpTypePointer
    uint32_t next_id = bound;
    if (!ptr_storage || !priv_ptr || !need_new || !bad || !bad_ptr || !c.struct_at || !c.elem || !c.ptr_pointee ||
        !c.id_type || !c.struct_bad_mask || !c.const_val || !c.const_type || !c.is_const || !words) goto fail;

    // Pass 1: types, constants, pointer-typed ids, and the BuiltIn decorations to strip.
    for (size_t o = 5; o < wordCount;) {
        const uint32_t *in = &inWords[o];
        uint16_t op = in[0] & 0xFFFFu, len = in[0] >> 16;
        if (len == 0 || o + len > wordCount) goto fail;
        if (op == SpvOpConstant && len >= 4 && in[2] < bound) {
            c.is_const[in[2]] = 1;
            c.const_type[in[2]] = in[1];
            c.const_val[in[2]] = in[3];
        } else if (op == SpvOpTypeStruct && len >= 2 && in[1] < bound) {
            c.struct_at[in[1]] = (uint32_t)o;
        } else if ((op == SpvOpTypeArray || op == SpvOpTypeRuntimeArray) && len >= 3 && in[1] < bound) {
            c.elem[in[1]] = in[2];
        } else if (op == SpvOpTypePointer && len >= 4 && in[1] < bound && in[3] < bound) {
            ptr_storage[in[1]] = in[2] + 1;
            c.ptr_pointee[in[1]] = in[3];
            if (in[2] == SpvStorageClassPrivate && !priv_ptr[in[3]]) priv_ptr[in[3]] = in[1];
        } else if ((op == SpvOpVariable || op == SpvOpFunctionParameter || spv_is_chain(op)) && len >= 3 &&
                   in[2] < bound) {
            c.id_type[in[2]] = in[1];
        } else if (op == SpvOpDecorate && len >= 4 && in[2] == SpvDecorationBuiltIn &&
                   (in[3] == SpvBuiltInClipDistance || in[3] == SpvBuiltInCullDistance) && in[1] < bound) {
            bad[in[1]] = 1;
        } else if (op == SpvOpMemberDecorate && len >= 5 && in[3] == SpvDecorationBuiltIn &&
                   (in[4] == SpvBuiltInClipDistance || in[4] == SpvBuiltInCullDistance) && in[1] < bound) {
            if (in[2] >= 32) goto fail;
            c.struct_bad_mask[in[1]] |= 1u << in[2];
        } else if (op == SpvOpTypeForwardPointer) {
            goto fail; // not produced by DXVK/vkd3d; leave such modules alone
        }
        o += len;
    }
    // A struct must keep at least one member.
    for (uint32_t id = 1; id < bound; id++) {
        if (!c.struct_bad_mask[id]) continue;
        if (!c.struct_at[id]) goto fail;
        uint32_t members = (inWords[c.struct_at[id]] >> 16) - 2;
        uint32_t valid = members >= 32 ? c.struct_bad_mask[id] : (c.struct_bad_mask[id] & ((1u << members) - 1));
        if (valid != c.struct_bad_mask[id] || (uint32_t)__builtin_popcount(valid) >= members) goto fail;
    }
    // Pass 2 (to a fixpoint): pointers derived from bad variables, and pointers into removed members.
    for (bool changed = true; changed;) {
        changed = false;
        for (size_t o = 5; o < wordCount;) {
            const uint32_t *in = &inWords[o];
            uint16_t op = in[0] & 0xFFFFu, len = in[0] >> 16;
            o += len;
            if (!spv_is_chain(op) || len < 4 || in[2] >= bound || in[3] >= bound) continue;
            if (bad[in[3]] && !bad[in[2]]) {
                bad[in[2]] = 1;
                changed = true;
            }
            if (!bad_ptr[in[2]]) {
                int r = bad_ptr[in[3]] ? 1 : spv_walk_chain(&c, in, len, NULL);
                if (r < 0) goto fail;
                if (r > 0) {
                    bad_ptr[in[2]] = 1;
                    changed = true;
                }
            }
        }
    }
    // Pass 3: every use of a pointer into a removed member must be one the emit pass rewrites, and no whole struct
    // with removed members may be loaded, stored or copied (its value layout changes).
    bool in_function = false;
    for (size_t o = 5; o < wordCount;) {
        const uint32_t *in = &inWords[o];
        uint16_t op = in[0] & 0xFFFFu, len = in[0] >> 16;
        o += len;
        if (op == SpvOpFunction) in_function = true;
        if (op == SpvOpFunctionEnd) in_function = false;
        if (!in_function || op == SpvOpLine) continue;
        if (op == SpvOpLoad && len >= 4) {
            if (in[3] >= bound) goto fail;
            if (!bad_ptr[in[3]] && c.id_type[in[3]] && spv_has_bad_struct(&c, c.ptr_pointee[c.id_type[in[3]]])) goto fail;
        } else if ((op == SpvOpStore || op == SpvOpCopyMemory) && len >= 3) {
            if (in[1] >= bound || in[2] >= bound) goto fail;
            if (op == SpvOpStore && bad_ptr[in[2]]) goto fail;
            if (bad_ptr[in[1]] || (op == SpvOpCopyMemory && bad_ptr[in[2]])) continue;
            if (c.id_type[in[1]] && spv_has_bad_struct(&c, c.ptr_pointee[c.id_type[in[1]]])) goto fail;
        } else if (spv_is_chain(op) && len >= 4) {
            for (uint16_t k = 4; k < len; k++)
                if (in[k] < bound && bad_ptr[in[k]]) goto fail; // used as an index (never valid for a pointer)
        } else if (op == SpvOpFunctionParameter && len >= 3) {
            if (c.id_type[in[2]] < bound && spv_has_bad_struct(&c, c.ptr_pointee[c.id_type[in[2]]])) goto fail;
        } else {
            for (uint16_t k = 1; k < len; k++)
                if (in[k] < bound && bad_ptr[in[k]]) goto fail;
        }
    }
    // Pass 4: which pointees need a Private pointer type.
    for (size_t o = 5; o < wordCount;) {
        uint16_t op = inWords[o] & 0xFFFFu, len = inWords[o] >> 16;
        bool var = op == SpvOpVariable && len >= 4;
        if ((var || (spv_is_chain(op) && len >= 4)) && inWords[o + 2] < bound && bad[inWords[o + 2]]) {
            uint32_t type = inWords[o + 1];
            if (type >= bound || !ptr_storage[type]) goto fail;
            uint32_t pointee = c.ptr_pointee[type];
            if (!priv_ptr[pointee]) {
                if (next_id - bound >= 64) goto fail;
                priv_ptr[pointee] = next_id++;
                need_new[pointee] = 1;
            }
        }
        o += len;
    }

    // Emit.
    memcpy(words, inWords, 20);
    size_t out = 5;
    for (size_t o = 5; o < wordCount;) {
        uint16_t op = inWords[o] & 0xFFFFu, len = inWords[o] >> 16;
        const uint32_t *in = &inWords[o];
        o += len;
        if (op == SpvOpCapability && (in[1] == SpvCapabilityClipDistance || in[1] == SpvCapabilityCullDistance)) continue;
        if (op == SpvOpDecorate && in[1] < bound && (bad[in[1]] || bad_ptr[in[1]])) continue;
        if (op == SpvOpName && len >= 2 && in[1] < bound && bad_ptr[in[1]]) continue;
        if ((op == SpvOpMemberName || op == SpvOpMemberDecorate) && len >= 3 && in[1] < bound &&
            c.struct_bad_mask[in[1]]) {
            uint32_t mask = c.struct_bad_mask[in[1]], m = in[2];
            if (m < 32 && (mask & (1u << m))) continue;
            memcpy(&words[out], in, len * sizeof(uint32_t));
            words[out + 2] = m - (uint32_t)__builtin_popcount(m >= 32 ? mask : (mask & ((1u << m) - 1)));
            out += len;
            continue;
        }
        if (op == SpvOpTypeStruct && len >= 2 && in[1] < bound && c.struct_bad_mask[in[1]]) {
            uint32_t s_id = in[1];
            uint32_t new_len = 2;
            uint32_t *dst = &words[out];
            dst[1] = s_id;
            for (uint32_t m_idx = 0; m_idx < len - 2u; m_idx++) {
                if (m_idx < 32 && (c.struct_bad_mask[s_id] & (1u << m_idx))) continue;
                dst[new_len++] = in[2 + m_idx];
            }
            dst[0] = (new_len << 16) | SpvOpTypeStruct;
            out += new_len;
            continue;
        }
        if (spv_is_chain(op) && len >= 3 && in[2] < bound && bad_ptr[in[2]]) continue;
        if (op == SpvOpStore && len >= 3 && in[1] < bound && bad_ptr[in[1]]) continue;
        if (op == SpvOpCopyMemory && len >= 3 && in[1] < bound && in[2] < bound && (bad_ptr[in[1]] || bad_ptr[in[2]]))
            continue;
        if (op == SpvOpLoad && len >= 4 && in[3] < bound && bad_ptr[in[3]]) {
            words[out++] = (3u << 16) | SpvOpUndef;
            words[out++] = in[1];
            words[out++] = in[2];
            continue;
        }
        if (op == SpvOpEntryPoint && len >= 4 && version < 0x00010400) {
            // Before SPIR-V 1.4 the interface lists only Input/Output variables.
            size_t name_words = (strnlen((const char *)&in[3], (len - 3) * 4) + 4) / 4;
            size_t iface = 3 + name_words, n = 0;
            uint32_t *dst = &words[out];
            for (size_t k = 0; k < iface && k < len; k++) dst[n++] = in[k];
            for (size_t k = iface; k < len; k++)
                if (in[k] >= bound || !bad[in[k]]) dst[n++] = in[k];
            dst[0] = ((uint32_t)n << 16) | SpvOpEntryPoint;
            out += n;
            continue;
        }
        memcpy(&words[out], in, len * sizeof(uint32_t));
        bool var = op == SpvOpVariable && len >= 4;
        bool chain = spv_is_chain(op) && len >= 4;
        if (chain) spv_walk_chain(&c, in, len, &words[out]); // renumbers member indices (checked in pass 2)
        if ((var || chain) && in[2] < bound && bad[in[2]]) {
            words[out + 1] = priv_ptr[c.ptr_pointee[in[1]]];
            if (var) words[out + 3] = SpvStorageClassPrivate;
        }
        out += len;
        if (op == SpvOpTypePointer && len >= 4 && in[3] < bound && need_new[in[3]]) {
            words[out++] = (4u << 16) | SpvOpTypePointer;
            words[out++] = priv_ptr[in[3]];
            words[out++] = SpvStorageClassPrivate;
            words[out++] = in[3];
            need_new[in[3]] = 0;
        }
    }
    words[3] = next_id;
    *pOutBytes = out * sizeof(uint32_t);
    LOG("patched SPIR-V: stripped ClipDistance/CullDistance");
    goto done;

fail:
    free(words);
    words = NULL;
done:
    free(ptr_storage); free(priv_ptr); free(need_new); free(bad); free(bad_ptr);
    free(c.struct_at); free(c.elem); free(c.ptr_pointee); free(c.id_type); free(c.struct_bad_mask);
    free(c.const_val); free(c.const_type); free(c.is_const);
    return words;
}

// -----------------------------------------------------------------------------
// Layer Data Structures & Dispatch Tables
// -----------------------------------------------------------------------------
struct InstanceData {
    VkInstance instance;
    void *dispatch_key;
    PFN_vkGetInstanceProcAddr get_instance_proc_addr;
    PFN_vkDestroyInstance fpDestroyInstance;
    PFN_vkEnumeratePhysicalDevices fpEnumeratePhysicalDevices;
    PFN_vkGetPhysicalDeviceFeatures fpGetPhysicalDeviceFeatures;
    PFN_vkGetPhysicalDeviceFeatures2 fpGetPhysicalDeviceFeatures2;
    PFN_vkGetPhysicalDeviceFormatProperties fpGetPhysicalDeviceFormatProperties;
    PFN_vkGetPhysicalDeviceFormatProperties2 fpGetPhysicalDeviceFormatProperties2;
    PFN_vkGetPhysicalDeviceImageFormatProperties fpGetPhysicalDeviceImageFormatProperties;
    PFN_vkGetPhysicalDeviceImageFormatProperties2 fpGetPhysicalDeviceImageFormatProperties2;
    PFN_vkGetPhysicalDeviceMemoryProperties fpGetPhysicalDeviceMemoryProperties;
    PFN_vkGetPhysicalDeviceMemoryProperties2 fpGetPhysicalDeviceMemoryProperties2;
    PFN_vkCreateDevice fpCreateDevice;
    PFN_vkEnumerateDeviceExtensionProperties fpEnumerateDeviceExtensionProperties;
    PFN_vkGetPhysicalDeviceProperties fpGetPhysicalDeviceProperties;
    PFN_vkGetPhysicalDeviceProperties2 fpGetPhysicalDeviceProperties2;
};

struct TrackedBCImage {
    VkImage image;
    VkFormat orig_format;
    VkFormat decoded_format;
    VkExtent3D extent;
};

// BC images, by handle. A hash table: the first version held 256 in an array and silently dropped the rest, so in a
// game world (thousands of DXT textures) most views kept their BC format on the RGBA8 image and sampled black and
// their uploads were not decoded. [Claude Opus 5.5] 2026-10-26
struct BCNode { struct TrackedBCImage t; struct BCNode *next; };
#define BC_BUCKETS 4096
static struct BCNode *g_bc_images[BC_BUCKETS];
static pthread_mutex_t g_bc_image_mutex = PTHREAD_MUTEX_INITIALIZER;
// F2: tracked BC images; while 0 (games without BC textures, or before the first one) find_bc_image takes no lock.
static _Atomic uint32_t g_bc_image_n;

static uint32_t handle_bucket(uint64_t v, uint32_t n) { return (uint32_t)((v ^ (v >> 17) ^ (v >> 33)) % n); }

static void register_bc_image(VkImage img, VkFormat orig, VkFormat decoded, VkExtent3D ext) {
    struct BCNode *e = calloc(1, sizeof *e);
    if (!e) return;
    e->t = (struct TrackedBCImage){.image = img, .orig_format = orig, .decoded_format = decoded, .extent = ext};
    pthread_mutex_lock(&g_bc_image_mutex);
    uint32_t bkt = handle_bucket((uint64_t)(uintptr_t)img, BC_BUCKETS);
    e->next = g_bc_images[bkt];
    g_bc_images[bkt] = e;
    atomic_fetch_add_explicit(&g_bc_image_n, 1, memory_order_release);
    pthread_mutex_unlock(&g_bc_image_mutex);
}

// Copies the entry out (the node can be freed by another thread's vkDestroyImage).
static bool find_bc_image(VkImage img, struct TrackedBCImage *out) {
    // the image was created before the app records a copy to it (app-side ordering), so its count is visible here
    if (!atomic_load_explicit(&g_bc_image_n, memory_order_acquire)) return false;
    bool found = false;
    pthread_mutex_lock(&g_bc_image_mutex);
    for (struct BCNode *e = g_bc_images[handle_bucket((uint64_t)(uintptr_t)img, BC_BUCKETS)]; e; e = e->next)
        if (e->t.image == img) { *out = e->t; found = true; break; }
    pthread_mutex_unlock(&g_bc_image_mutex);
    return found;
}

static void unregister_bc_image(VkImage img) {
    pthread_mutex_lock(&g_bc_image_mutex);
    struct BCNode **pp = &g_bc_images[handle_bucket((uint64_t)(uintptr_t)img, BC_BUCKETS)];
    while (*pp && (*pp)->t.image != img) pp = &(*pp)->next;
    struct BCNode *e = *pp;
    if (e) {
        *pp = e->next;
        atomic_fetch_sub_explicit(&g_bc_image_n, 1, memory_order_relaxed);
    }
    pthread_mutex_unlock(&g_bc_image_mutex);
    free(e);
}

#include <sys/syscall.h>

// K-CF2 port, env-gated (stage kcs1+kcf2+memlog) [keeper, Claude Opus 5.5] 2026-10-31: live bytes per memory heap,
// added when vkAllocateMemory succeeds and taken off again in vkFreeMemory (docs/common_fixes.md sections 15, 15b).
// Always counted (a few atomics); reported as heapUsage only with MALI_COMPAT_BUDGET_MB=<MiB>, which also sets the
// cap. Unset, the report is the old one (g_allocated_bytes, bytes ever allocated, on every heap; 6 GiB cap), byte for
// byte as in the deployed 2dfc35b8. Counted: app allocations (the layer's own dma-heap export included), host-pointer
// imports, BC staging buffers. Not counted: app dma-buf fd imports (the exporter's pages).
#define BUDGET_NO_HEAP 0xFFu
static _Atomic uint64_t g_heap_usage[VK_MAX_MEMORY_HEAPS];
static _Atomic uint64_t g_heap_peak[VK_MAX_MEMORY_HEAPS];
static uint64_t g_heap_budget[VK_MAX_MEMORY_HEAPS]; // reported heapBudget, for the log
static _Atomic uint32_t g_heap_count;
static _Atomic uint64_t g_live_allocs, g_budget_queries;
static bool g_budget_accounted;            // MALI_COMPAT_BUDGET_MB set: live usage and that cap are reported
static uint64_t g_budget_cap = 6ULL << 30; // the old fixed cap unless MALI_COMPAT_BUDGET_MB
static uint32_t g_budget_log_s;            // budget log interval in s (accounted + GPU_TIME or DEBUG), 0 = off

static void budget_add(uint32_t heap, uint64_t size) {
    if (heap >= VK_MAX_MEMORY_HEAPS) return;
    uint64_t now = atomic_fetch_add(&g_heap_usage[heap], size) + size;
    atomic_fetch_add(&g_live_allocs, 1);
    uint64_t p = atomic_load(&g_heap_peak[heap]);
    while (now > p && !atomic_compare_exchange_weak(&g_heap_peak[heap], &p, now)) {}
}

static void budget_sub(uint32_t heap, uint64_t size) {
    if (heap >= VK_MAX_MEMORY_HEAPS) return;
    atomic_fetch_sub(&g_heap_usage[heap], size);
    atomic_fetch_sub(&g_live_allocs, 1);
}

static uint64_t budget_live_mb(void) {
    uint64_t t = 0;
    for (uint32_t i = 0; i < VK_MAX_MEMORY_HEAPS; i++) t += atomic_load(&g_heap_usage[i]);
    return t >> 20;
}

// One line per interval: per heap live usage / reported budget (peak since start), live allocations and how often the
// app read the budget (0 queries = the game does not stream by it, whatever the numbers say).
static void *budget_log_thread(void *arg) {
    (void)arg;
    for (;;) {
        sleep(g_budget_log_s);
        char line[1024];
        size_t n = 0;
        uint32_t hc = atomic_load(&g_heap_count);
        for (uint32_t i = 0; i < hc && i < VK_MAX_MEMORY_HEAPS && n < sizeof line; i++)
            n += (size_t)snprintf(line + n, sizeof line - n, " heap%u %llu/%llu MiB (peak %llu)", i,
                                  (unsigned long long)(atomic_load(&g_heap_usage[i]) >> 20),
                                  (unsigned long long)(g_heap_budget[i] >> 20),
                                  (unsigned long long)(atomic_load(&g_heap_peak[i]) >> 20));
        if (n >= sizeof line) n = sizeof line - 1;
        line[n] = '\0';
        fprintf(stderr, "mali-compat: budget%s, %llu allocs, %llu queries\n", line,
                (unsigned long long)atomic_load(&g_live_allocs), (unsigned long long)atomic_load(&g_budget_queries));
    }
    return NULL;
}

// Memory event log, MALI_COMPAT_MEMLOG=1 (=2: without the capture-address query) [keeper, Claude Opus 5.5] 2026-10-31.
// File /root/mali-memlog-<comm>.tsv (opened for append; each process writes a '#' line, then the header), one TSV line
// per event, CLOCK_REALTIME ms (lines up with logcat):  t_ms ev mem size type heap va tid live_mb
//   A  vkAllocateMemory  va = vkGetDeviceMemoryOpaqueCaptureAddress (blob-defined; '-' if 0 or MEMLOG=2)
//   M  map               va = returned pointer - offset (placed maps: the driver's own address) = SAME_VA GPU VA
//   U  unmap             va = the mapping's va        F  vkFreeMemory  va = the last mapped va ('-' if never mapped)
//   X  failed alloc      va = the VkResult            S  every 100th queue submit, mem = the submit count
// heap '-' = not counted in the budget (dma-buf import); live_mb = counted live bytes of all heaps after the event.
// Buffered, written once per second by a thread, at vkDestroyDevice and at exit. Cap MALI_COMPAT_MEMLOG_MB (default
// 64 MiB of lines per process), then one TRUNC line and the log stops. Unset: no file, no thread, no hooks.
#define ML_NONE UINT32_MAX
#define ML_BUF (256u << 10)
static int g_memlog; // 0 off, 1 on, 2 on without the capture-address query
static pthread_mutex_t g_ml_mutex = PTHREAD_MUTEX_INITIALIZER;
static int g_ml_fd = -1;
static char *g_ml_buf;
static size_t g_ml_len;
static uint64_t g_ml_bytes, g_ml_cap = 64ULL << 20; // bytes taken so far (written + buffered), cap
static bool g_ml_stopped;
static _Atomic uint64_t g_ml_submits;

static uint64_t ml_now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    return (uint64_t)ts.tv_sec * 1000u + (uint64_t)ts.tv_nsec / 1000000u;
}

// Called with g_ml_mutex held.
static void ml_flush_locked(void) {
    size_t off = 0;
    while (g_ml_fd >= 0 && off < g_ml_len) {
        ssize_t w = write(g_ml_fd, g_ml_buf + off, g_ml_len - off);
        if (w < 0 && errno == EINTR) continue;
        if (w <= 0) break;
        off += (size_t)w;
    }
    g_ml_len = 0;
}

// Called with g_ml_mutex held.
static void ml_append_locked(const char *line, size_t n) {
    if (g_ml_len + n > ML_BUF) ml_flush_locked();
    if (n > ML_BUF) return;
    memcpy(g_ml_buf + g_ml_len, line, n);
    g_ml_len += n;
}

static void ml_write(const char *line, size_t n) {
    pthread_mutex_lock(&g_ml_mutex);
    if (!g_ml_stopped && g_ml_buf) {
        if (g_ml_bytes + n > g_ml_cap) {
            char t[128];
            int k = snprintf(t, sizeof t, "%llu\tTRUNC\t-\t%llu\t-\t-\t-\t%ld\t%llu\n", (unsigned long long)ml_now_ms(),
                             (unsigned long long)g_ml_bytes, (long)syscall(SYS_gettid), (unsigned long long)budget_live_mb());
            if (k > 0) ml_append_locked(t, (size_t)k < sizeof t ? (size_t)k : sizeof t - 1);
            ml_flush_locked();
            g_ml_stopped = true;
            if (g_ml_fd >= 0) close(g_ml_fd);
            g_ml_fd = -1;
        } else {
            ml_append_locked(line, n);
            g_ml_bytes += n;
        }
    }
    pthread_mutex_unlock(&g_ml_mutex);
}

// va_kind: 0 = '-', 1 = hex address, 2 = signed decimal (VkResult)
static void ml_event(char ev, uint64_t mem, uint64_t size, uint32_t type, uint32_t heap, uint64_t va, int va_kind) {
    if (!g_memlog) return;
    char line[256], ty[16] = "-", hp[16] = "-", vs[32] = "-";
    if (type != ML_NONE) snprintf(ty, sizeof ty, "%u", type);
    if (heap < VK_MAX_MEMORY_HEAPS) snprintf(hp, sizeof hp, "%u", heap);
    if (va_kind == 1) snprintf(vs, sizeof vs, "0x%llx", (unsigned long long)va);
    else if (va_kind == 2) snprintf(vs, sizeof vs, "%lld", (long long)(int64_t)va);
    int n = snprintf(line, sizeof line, "%llu\t%c\t0x%llx\t%llu\t%s\t%s\t%s\t%ld\t%llu\n", (unsigned long long)ml_now_ms(),
                     ev, (unsigned long long)mem, (unsigned long long)size, ty, hp, vs, (long)syscall(SYS_gettid),
                     (unsigned long long)budget_live_mb());
    if (n > 0) ml_write(line, (size_t)n < sizeof line ? (size_t)n : sizeof line - 1);
}

static void ml_flush(void) {
    pthread_mutex_lock(&g_ml_mutex);
    if (g_ml_buf) ml_flush_locked();
    pthread_mutex_unlock(&g_ml_mutex);
}

static void *ml_flush_thread(void *arg) {
    (void)arg;
    for (;;) {
        sleep(1);
        pthread_mutex_lock(&g_ml_mutex);
        bool stop = g_ml_stopped;
        if (!stop) ml_flush_locked();
        pthread_mutex_unlock(&g_ml_mutex);
        if (stop) return NULL;
    }
}

// Once per process. Leaves g_memlog 0 when the file cannot be opened.
static void ml_open(const char *path, const char *comm) {
    g_ml_buf = malloc(ML_BUF);
    g_ml_fd = g_ml_buf ? open(path, O_WRONLY | O_CREAT | O_APPEND | O_CLOEXEC, 0644) : -1;
    if (g_ml_fd < 0) {
        fprintf(stderr, "mali-compat: memlog: cannot open %s: %s\n", path, strerror(errno));
        free(g_ml_buf);
        g_ml_buf = NULL;
        g_memlog = 0;
        return;
    }
    char h[256];
    int n = snprintf(h, sizeof h, "# mali-compat memlog pid %d comm %s start_ms %llu cap %llu MiB%s\n"
                     "t_ms\tev\tmem\tsize\ttype\theap\tva\ttid\tlive_mb\n", (int)getpid(), comm,
                     (unsigned long long)ml_now_ms(), (unsigned long long)(g_ml_cap >> 20),
                     g_memlog == 2 ? ", no capture address" : "");
    if (n > 0) ml_write(h, (size_t)n < sizeof h ? (size_t)n : sizeof h - 1);
    atexit(ml_flush);
    pthread_t t;
    if (pthread_create(&t, NULL, ml_flush_thread, NULL) == 0) pthread_detach(t);
}

static void ml_submit_tick(void) {
    uint64_t n = atomic_fetch_add(&g_ml_submits, 1) + 1;
    if (n % 100 == 0) ml_event('S', n, 0, ML_NONE, BUDGET_NO_HEAP, 0, 0);
}

struct StagingResource {
    VkBuffer buffer;
    VkDeviceMemory memory;
    VkDeviceSize size;      // power-of-two bucket, >= STAGING_MIN
    VkDeviceAddress addr;
    VkDeviceSize mem_size;  // counted in g_heap_usage[heap] while memory is set (0 = not counted)
    uint32_t heap;
};

// Decode staging buffers are pooled per device: a command buffer's buffers go back to the pool when it is begun,
// reset or freed again (the app may only do that after the GPU is done with it). Before, every BC upload did
// vkCreateBuffer + vkAllocateMemory + vkBindBufferMemory and later vkFreeMemory, all /dev/mali0 ioctls, on the app's
// recording thread while textures stream in. [Claude Opus 5.5] 2026-10-26
#define STAGING_MIN (64 * 1024)
#define STAGING_POOL_CAP (64ull * 1024 * 1024) // bytes kept for reuse; above that, buffers are freed
struct StagingPoolNode { struct StagingResource r; struct StagingPoolNode *next; };
static pthread_mutex_t g_staging_pool_mutex = PTHREAD_MUTEX_INITIALIZER;

// MALI_COMPAT_STATS=1: every ~5 s one line with the BC upload counts and the CPU time spent recording them
// (MALI_COMPAT_STATS=every: after each upload, for probes).
static bool g_stats;
static uint64_t g_stats_interval_ns = 5000000000ull;
static _Atomic uint64_t g_st_uploads, g_st_regions, g_st_bytes, g_st_new, g_st_reused, g_st_ns, g_st_last_ns;

// App state the BC decode overwrites inside the app's command buffer, restored after it (Review 6 H1): the bound
// compute pipeline and the push constants. Without that, an app that tracks its own bindings (DXVK, vkd3d-proton)
// dispatched its next compute work with the decoder's pipeline or push constants. [Claude Opus 5.5] 2026-10-26
#define PUSH_RECS 8
struct PushRec {
    VkPipelineLayout layout;
    VkShaderStageFlags stages;
    uint32_t offset, size;
    bool v2;                        // recorded from vkCmdPushConstants2(KHR): replayed with it (pc2-restore-1)
    uint8_t data[256];
};

// pc2-restore-1 (WD2 B10-18) [keeper, Claude Opus 5.5] 2026-11-01: DXVK pushes with vkCmdPushConstants2KHR
// (VK_KHR_maintenance6), which this layer did not hook, so the restore after a BC decode left DXVK's next compute
// dispatch with the decoder's 56 push bytes. Both entry points now feed the PushRec list above. Cumulative counters,
// one stderr line every 10 s once any is non-zero (pc2_log_thread).
// F3 (our-cpu-paths-1105.md) [keeper, Claude Opus 5.5] 2026-11-05: the counters were five shared atomics, one
// atomic add per push on one cache line from every recording thread (ER: 0.8 ms/s in __aarch64_ldadd8_relax over 7
// GXWorker threads). Each thread now counts in its own cache-line slot with a plain load + store (only it writes
// there); the log thread sums the slots. Threads past PC2_SLOTS - 1 share the last slot with atomic adds.
enum { PC2_V1, PC2_V2, PC2_V2_UNREC, PC2_DECODES, PC2_DECODES_V2, PC2_N };
#define PC2_SLOTS 64
struct Pc2Slot { _Atomic uint64_t n[PC2_N]; } __attribute__((aligned(64)));
static struct Pc2Slot g_pc2_slot[PC2_SLOTS];
static _Atomic uint32_t g_pc2_nslots;
static __thread struct Pc2Slot *tl_pc2;
static __thread bool tl_pc2_shared;
static inline void pc2_count(int k) {
    struct Pc2Slot *s = tl_pc2;
    if (!s) {
        uint32_t i = atomic_fetch_add_explicit(&g_pc2_nslots, 1, memory_order_relaxed);
        if (i >= PC2_SLOTS - 1) { i = PC2_SLOTS - 1; tl_pc2_shared = true; }
        tl_pc2 = s = &g_pc2_slot[i];
    }
    if (tl_pc2_shared)
        atomic_fetch_add_explicit(&s->n[k], 1, memory_order_relaxed);
    else
        atomic_store_explicit(&s->n[k], atomic_load_explicit(&s->n[k], memory_order_relaxed) + 1, memory_order_relaxed);
}
static uint64_t pc2_sum(int k) {
    uint64_t t = 0;
    for (uint32_t i = 0; i < PC2_SLOTS; i++) t += atomic_load_explicit(&g_pc2_slot[i].n[k], memory_order_relaxed);
    return t;
}

// The app's compute push descriptors (vkd3d-proton pushes its root CBVs this way with MALI_COMPAT_VENDOR_ID=5143),
// merged per binding since the last layout change, so they can be pushed again after a BC decode like the push
// constants. Only single-descriptor writes are kept; anything else marks the record unusable. [Claude Opus 5.5] 2026-10-28
#define PUSHDESC_MAX 16
struct PushDescRec {
    VkPipelineLayout layout;
    uint32_t set, count;
    bool bad;
    VkWriteDescriptorSet w[PUSHDESC_MAX];
    VkDescriptorBufferInfo buf[PUSHDESC_MAX];
    VkDescriptorImageInfo img[PUSHDESC_MAX];
    VkBufferView view[PUSHDESC_MAX];
};

// Per command buffer: decode staging buffers (back to the pool when it is begun, reset or freed again) and the
// tracked state above. Hash table (the first version kept 128 command buffers and leaked everything recorded in any
// other). [Claude Opus 5.5]
struct DeviceData;
// GPU timing, per pass (KZ1/KZ2): draws with a depth EQUAL/no-write pipeline (eq), with depth write (dw), with a
// VS-stage store (vsw), and the sum of vertex/index count x instance count of the direct draws (vxi).
struct GtCnt { uint32_t eq, dw, vsw; uint64_t vxi; };
static inline void gt_cnt_add(struct GtCnt *t, const struct GtCnt *a) {
    t->eq += a->eq; t->dw += a->dw; t->vsw += a->vsw; t->vxi += a->vxi;
}

// desc-capture-1: one shadowed descriptor (type = VkDescriptorType + 1, 0 = never written) and the per-command-buffer
// compute set-0 state (see "desc-capture-1" below).
#define DC_NB 16
struct DcDesc { uint64_t h, off, range; uint32_t type, pad; };
struct DcCmd {
    uint8_t kind;               // 0 nothing bound, 1 descriptor set, 2 push descriptors (set 0, compute)
    uint16_t dyn_mask;          // bindings with a dynamic offset from the last bind
    uint64_t set0, bind_seq;    // bound set, descriptor write sequence at the bind
    uint32_t dyn[DC_NB];
    struct DcDesc push[DC_NB];
    uint32_t ndisp;             // captured dispatches since begin
    uint64_t first_seq, last_seq;
};
struct DcFns;

// cover-trim (cover_trim.h): the open pass, draws, the app's last scissor and conditional rendering.
struct CtCmd {
    uint32_t colors, draws, sdraws;
    VkImageView depth_view;
    VkImageLayout depth_layout;
    float clear_depth;
    bool depth_clear, suspending, in_pass, sc_valid, sc_wc, app_cond;
    VkRect2D sc, area;
};

struct CmdBufData {
    VkCommandBuffer cmdbuf;
    struct DeviceData *dev;
    VkCommandPool pool;
    struct StagingResource *staging_resources;
    uint32_t staging_count;
    uint32_t staging_capacity;
    VkPipeline compute_pipeline;    // last app compute pipeline, VK_NULL_HANDLE if none since begin
    bool descriptor_buffers;        // vkCmdBindDescriptorBuffersEXT was recorded since begin
    uint32_t push_count;            // records in push[], oldest first
    struct PushRec push[PUSH_RECS];
    struct PushDescRec cpd;         // compute push descriptors, layout VK_NULL_HANDLE if none since begin
    struct PushDescRec gpd;         // graphics push descriptors (full pushes, see compat_CmdPushDescriptorSetKHR)
    VkPipeline gfx_pipeline;        // last app graphics pipeline (GPU timing)
    uint8_t gfx_flags;              // its PF_* flags (GPU timing only)
    int32_t gt_rec;                 // GPU timing record of the open render pass, -1 if none
    VkImageView gt_src;             // first image view of the last push-descriptor write (GPU timing details)
    // GPU timing: draws recorded without an open pass record (secondary command buffers: DXVK's tiler mode), handed
    // to the primary's pass at vkCmdExecuteCommands. First draw's pipeline, counts and source view.
    uint32_t gt_sdraws, gt_svcount, gt_sicount;
    struct GtCnt gt_scnt;           // and their eq/dw/vsw/vxi counters
    VkPipeline gt_spipe;
    VkImageView gt_ssrc;
    bool async_skip;                // the bound graphics pipeline is still compiling (MALI_COMPAT_ASYNC_PSO)
    struct DcCmd dc;                // desc-capture-1 (MALI_COMPAT_DESC_CAPTURE)
    struct CtCmd ct;                // cover-trim (MALI_COMPAT_COVER_TRIM)
    struct CmdBufData *next;
    struct CmdBufData *st_next, *st_prev; // on g_staged while it holds decode staging buffers (g_cmdbuf_mutex)
    bool st_on;
    uint8_t sd_pass;                // sync-diag: inside a render pass instance (MALI_COMPAT_FULL_BARRIERS/SYNC_CTL)
    bool sd_susp;                   // sync-diag: the open pass is a SUSPENDING part
};

#define CMDBUF_BUCKETS 1024
static struct CmdBufData *g_cmdbufs[CMDBUF_BUCKETS];
static pthread_mutex_t g_cmdbuf_mutex = PTHREAD_MUTEX_INITIALIZER;
// F1 (our-cpu-paths-1105.md) [keeper, Claude Opus 5.5] 2026-11-05: the entries that hold decode staging buffers, so
// vkResetCommandPool only visits those instead of every bucket and every live command buffer (ER: 15.3 ms/s on the
// render thread, 79 % of this layer's CPU, all under g_cmdbuf_mutex). Usually empty outside texture streaming.
static struct CmdBufData *g_staged;
static void staged_link(struct CmdBufData *e) { // g_cmdbuf_mutex held
    if (e->st_on) return;
    e->st_prev = NULL;
    e->st_next = g_staged;
    if (g_staged) g_staged->st_prev = e;
    g_staged = e;
    e->st_on = true;
}
static void staged_unlink(struct CmdBufData *e) { // g_cmdbuf_mutex held
    if (!e->st_on) return;
    if (e->st_prev) e->st_prev->st_next = e->st_next; else g_staged = e->st_next;
    if (e->st_next) e->st_next->st_prev = e->st_prev;
    e->st_next = e->st_prev = NULL;
    e->st_on = false;
}
// Bumped whenever an entry is freed; a thread's cached entry is only used while the generation is unchanged.
static _Atomic uint64_t g_cmdbuf_gen;
static __thread VkCommandBuffer tl_cmd;
static __thread struct CmdBufData *tl_cmd_data;
static __thread uint64_t tl_cmd_gen;

struct DeviceData {
    VkDevice device;
    void *dispatch_key;
    VkPhysicalDevice physical_device;
    struct InstanceData *instance_data;
    PFN_vkGetDeviceProcAddr get_device_proc_addr;
    PFN_vkDestroyDevice fpDestroyDevice;
    PFN_vkGetDeviceQueue fpGetDeviceQueue;
    PFN_vkCreateShaderModule fpCreateShaderModule;
    PFN_vkDestroyShaderModule fpDestroyShaderModule;
    PFN_vkCreateGraphicsPipelines fpCreateGraphicsPipelines;
    PFN_vkCreateComputePipelines fpCreateComputePipelines;
    PFN_vkDestroyPipeline fpDestroyPipeline;
    PFN_vkCreatePipelineLayout fpCreatePipelineLayout;
    PFN_vkDestroyPipelineLayout fpDestroyPipelineLayout;
    PFN_vkCreateImage fpCreateImage;
    PFN_vkDestroyImage fpDestroyImage;
    PFN_vkCreateImageView fpCreateImageView;
    PFN_vkDestroyImageView fpDestroyImageView;
    PFN_vkCreateBufferView fpCreateBufferView;
    PFN_vkCreatePipelineCache fpCreatePipelineCache;
    PFN_vkGetPipelineCacheData fpGetPipelineCacheData;
    PFN_vkDestroyPipelineCache fpDestroyPipelineCache;
    PFN_vkCreateQueryPool fpCreateQueryPool;
    PFN_vkDestroyQueryPool fpDestroyQueryPool;
    PFN_vkGetQueryPoolResults fpGetQueryPoolResults;
    PFN_vkCmdCopyQueryPoolResults fpCmdCopyQueryPoolResults;
    PFN_vkCmdFillBuffer fpCmdFillBuffer;
    PFN_vkGetDeviceImageMemoryRequirements fpGetDeviceImageMemoryRequirements;
    PFN_vkGetDeviceImageSparseMemoryRequirements fpGetDeviceImageSparseMemoryRequirements;
    PFN_vkCreateBuffer fpCreateBuffer;
    PFN_vkDestroyBuffer fpDestroyBuffer;
    PFN_vkAllocateMemory fpAllocateMemory;
    PFN_vkFreeMemory fpFreeMemory;
    PFN_vkGetMemoryFdKHR fpGetMemoryFdKHR;
    PFN_vkGetMemoryFdPropertiesKHR fpGetMemoryFdPropertiesKHR;
    PFN_vkBindBufferMemory fpBindBufferMemory;
    PFN_vkGetBufferMemoryRequirements fpGetBufferMemoryRequirements;
    PFN_vkGetBufferDeviceAddress fpGetBufferDeviceAddress;
    PFN_vkCmdBindPipeline fpCmdBindPipeline;
    PFN_vkCmdPushConstants fpCmdPushConstants;
    PFN_vkCmdPushConstants2 fpCmdPushConstants2;       // driver's, NULL if it lacks the name (pc2-restore-1)
    PFN_vkCmdPushConstants2KHR fpCmdPushConstants2KHR;
    PFN_vkCmdDispatch fpCmdDispatch;
    PFN_vkCmdPipelineBarrier fpCmdPipelineBarrier;
    PFN_vkCmdCopyBufferToImage fpCmdCopyBufferToImage;
    PFN_vkCmdCopyBufferToImage2 fpCmdCopyBufferToImage2;
    PFN_vkResetCommandBuffer fpResetCommandBuffer;
    PFN_vkBeginCommandBuffer fpBeginCommandBuffer;
    PFN_vkAllocateCommandBuffers fpAllocateCommandBuffers;
    PFN_vkFreeCommandBuffers fpFreeCommandBuffers;
    PFN_vkResetCommandPool fpResetCommandPool;
    PFN_vkDestroyCommandPool fpDestroyCommandPool;
    PFN_vkMapMemory fpMapMemory;
    PFN_vkUnmapMemory fpUnmapMemory;
    PFN_vkMapMemory2KHR fpMapMemory2KHR;
    PFN_vkUnmapMemory2KHR fpUnmapMemory2KHR;
    PFN_vkFlushMappedMemoryRanges fpFlushMappedMemoryRanges;
    PFN_vkInvalidateMappedMemoryRanges fpInvalidateMappedMemoryRanges;
    uint32_t coherent_types; // memory types with HOST_COHERENT
    uint8_t coherent_remap[VK_MAX_MEMORY_TYPES]; // MALI_COMPAT_FORCE_COHERENT: non-coherent host type -> coherent type + 1 (0 = none)
    uint8_t type_heap[VK_MAX_MEMORY_TYPES]; // memory type -> heap index for the budget (BUDGET_NO_HEAP = not counted)
    PFN_vkGetDeviceMemoryOpaqueCaptureAddress fpGetDeviceMemoryOpaqueCaptureAddress; // MALI_COMPAT_MEMLOG=1 only
    PFN_vkQueueSubmit ml_fpQueueSubmit;   // MALI_COMPAT_MEMLOG only
    PFN_vkQueueSubmit2 ml_fpQueueSubmit2; // MALI_COMPAT_MEMLOG only
    PFN_vkGetImageSubresourceLayout2KHR fpGetImageSubresourceLayout2KHR; // only with g_log_images

    // BC compute decoder resources
    VkPipelineLayout bc_layout;
    VkPipeline bc_pipelines[4];
    VkPipeline bc_pipelines_db[4]; // with DESCRIPTOR_BUFFER_BIT, for command buffers that bound descriptor buffers
    bool descriptor_buffer;        // the app enabled VkPhysicalDeviceDescriptorBufferFeaturesEXT::descriptorBuffer
    PFN_vkCmdBindDescriptorBuffersEXT fpCmdBindDescriptorBuffersEXT;
    PFN_vkCmdPushDescriptorSetKHR fpCmdPushDescriptorSetKHR;
    uint32_t device_local_memory_index;
    bool bc_init_done;
    struct StagingPoolNode *staging_pool; // free decode staging buffers (g_staging_pool_mutex)
    VkDeviceSize staging_pool_bytes;
    bool afrc; // VK_EXT_image_compression_control enabled by the layer for the BC shadows
    bool zero_divisor_emulated; // divisor 0 -> UINT32_MAX (the vendor lacks vertexAttributeInstanceRateZeroDivisor)
    PFN_vkCmdSetVertexInputEXT fpCmdSetVertexInputEXT;
    PFN_vkCmdSetDepthCompareOp fpCmdSetDepthCompareOp; // MALI_COMPAT_DEPTH_EQUAL only
    PFN_vkCreateComputePipelines fpCreateComputePipelinesReal; // for the GPU timing's pipeline hashes
    struct GpuTime *gt; // MALI_COMPAT_GPU_TIME
    struct DcFns *dcf;  // desc-capture-1 driver entry points, NULL when off
    struct CtDev *ct;   // cover-trim (MALI_COMPAT_COVER_TRIM), NULL when off
    struct SdDev *sd;   // sync-diag next/driver entry points, NULL when off
};

#define MAX_INSTANCES 32
#define MAX_DEVICES 32
static struct InstanceData g_instances[MAX_INSTANCES];
static uint32_t g_instance_count = 0;
static struct DeviceData g_devices[MAX_DEVICES];
static uint32_t g_device_count = 0;
static pthread_mutex_t g_table_mutex = PTHREAD_MUTEX_INITIALIZER;

static struct InstanceData *get_instance_data(VkInstance inst) {
    pthread_mutex_lock(&g_table_mutex);
    for (int32_t i = (int32_t)g_instance_count - 1; i >= 0; i--) {
        if (g_instances[i].instance == inst) {
            pthread_mutex_unlock(&g_table_mutex);
            return &g_instances[i];
        }
    }
    void *key = inst ? *(void **)inst : NULL;
    if (key) {
        for (int32_t i = (int32_t)g_instance_count - 1; i >= 0; i--) {
            if (g_instances[i].dispatch_key == key) {
                pthread_mutex_unlock(&g_table_mutex);
                return &g_instances[i];
            }
        }
    }
    struct InstanceData *fallback = g_instance_count > 0 ? &g_instances[g_instance_count - 1] : NULL;
    pthread_mutex_unlock(&g_table_mutex);
    return fallback;
}

static struct InstanceData *get_instance_by_phys_dev(VkPhysicalDevice phys_dev) {
    pthread_mutex_lock(&g_table_mutex);
    void *key = phys_dev ? *(void **)phys_dev : NULL;
    if (key) {
        for (int32_t i = (int32_t)g_instance_count - 1; i >= 0; i--) {
            if (g_instances[i].dispatch_key == key) {
                pthread_mutex_unlock(&g_table_mutex);
                return &g_instances[i];
            }
        }
    }
    struct InstanceData *fallback = g_instance_count > 0 ? &g_instances[g_instance_count - 1] : NULL;
    pthread_mutex_unlock(&g_table_mutex);
    return fallback;
}

// F2 (our-cpu-paths-1105.md) [keeper, Claude Opus 5.5] 2026-11-05: the device lookups ran under g_table_mutex on
// every call (Begin/Reset, submits, waits, maps, copies, every cmd_data miss: ~16 k lock ops/s on WD1's dxvk-cs). Each
// thread now keeps its last hit, valid while the table generation is unchanged; vkCreateDevice / vkDestroyDevice
// bump it under the mutex (the destroy compacts the array, so every cached pointer is dropped). A pointer handed out
// can go stale on a concurrent destroy exactly as before (the old code returned it after unlocking too). Fallbacks
// (no match) are not cached.
static _Atomic uint64_t g_dev_gen = 1;
#define DEV_TABLE_CHANGED() atomic_fetch_add_explicit(&g_dev_gen, 1, memory_order_release) // g_table_mutex held
static __thread VkDevice tl_dev_h;
static __thread struct DeviceData *tl_dev_by_h;
static __thread uint64_t tl_dev_h_gen;
static __thread void *tl_dev_key;
static __thread struct DeviceData *tl_dev_by_key;
static __thread uint64_t tl_dev_key_gen;

static struct DeviceData *get_device_data(VkDevice dev) {
    if (dev && tl_dev_h == dev && tl_dev_h_gen == atomic_load_explicit(&g_dev_gen, memory_order_acquire)) return tl_dev_by_h;
    struct DeviceData *hit = NULL;
    pthread_mutex_lock(&g_table_mutex);
    for (int32_t i = (int32_t)g_device_count - 1; i >= 0 && !hit; i--)
        if (g_devices[i].device == dev) hit = &g_devices[i];
    void *key = dev ? *(void **)dev : NULL;
    for (int32_t i = (int32_t)g_device_count - 1; key && i >= 0 && !hit; i--)
        if (g_devices[i].dispatch_key == key) hit = &g_devices[i];
    struct DeviceData *r = hit ? hit : g_device_count > 0 ? &g_devices[g_device_count - 1] : NULL;
    if (hit) {
        tl_dev_h = dev;
        tl_dev_by_h = hit;
        tl_dev_h_gen = atomic_load_explicit(&g_dev_gen, memory_order_relaxed);
    }
    pthread_mutex_unlock(&g_table_mutex);
    return r;
}

static struct DeviceData *get_device_data_by_cmd(VkCommandBuffer cmd) {
    void *key = cmd ? *(void **)cmd : NULL;
    if (key && tl_dev_key == key && tl_dev_key_gen == atomic_load_explicit(&g_dev_gen, memory_order_acquire)) return tl_dev_by_key;
    struct DeviceData *hit = NULL;
    pthread_mutex_lock(&g_table_mutex);
    for (int32_t i = (int32_t)g_device_count - 1; key && i >= 0 && !hit; i--)
        if (g_devices[i].dispatch_key == key) hit = &g_devices[i];
    struct DeviceData *r = hit ? hit : g_device_count > 0 ? &g_devices[g_device_count - 1] : NULL;
    if (hit) {
        tl_dev_key = key;
        tl_dev_by_key = hit;
        tl_dev_key_gen = atomic_load_explicit(&g_dev_gen, memory_order_relaxed);
    }
    pthread_mutex_unlock(&g_table_mutex);
    return r;
}

static struct DeviceData *get_device_data_by_cmd(VkCommandBuffer cmd);

// The command buffer's entry, created on first use. The command buffer is externally synchronized, so the fields are
// only touched by the thread recording it; the table itself is under g_cmdbuf_mutex.
static struct CmdBufData *cmd_data(VkCommandBuffer cmd) {
    uint64_t gen = atomic_load_explicit(&g_cmdbuf_gen, memory_order_acquire);
    if (tl_cmd == cmd && tl_cmd_gen == gen && tl_cmd_data) return tl_cmd_data;
    struct DeviceData *dev = get_device_data_by_cmd(cmd);
    pthread_mutex_lock(&g_cmdbuf_mutex);
    uint32_t bkt = handle_bucket((uint64_t)(uintptr_t)cmd, CMDBUF_BUCKETS);
    struct CmdBufData *entry = g_cmdbufs[bkt];
    while (entry && entry->cmdbuf != cmd) entry = entry->next;
    if (!entry && (entry = calloc(1, sizeof *entry))) {
        entry->cmdbuf = cmd;
        entry->dev = dev;
        entry->gt_rec = -1;
        entry->next = g_cmdbufs[bkt];
        g_cmdbufs[bkt] = entry;
    }
    pthread_mutex_unlock(&g_cmdbuf_mutex);
    tl_cmd = cmd;
    tl_cmd_data = entry;
    tl_cmd_gen = gen;
    return entry;
}

static void add_staging_resource(VkCommandBuffer cmd, struct StagingResource res) {
    struct CmdBufData *entry = cmd_data(cmd);
    if (entry && entry->staging_count >= entry->staging_capacity) {
        uint32_t new_cap = entry->staging_capacity ? entry->staging_capacity * 2 : 16;
        struct StagingResource *r = realloc(entry->staging_resources, new_cap * sizeof *r);
        if (r) { entry->staging_resources = r; entry->staging_capacity = new_cap; }
    }
    if (entry && entry->staging_count < entry->staging_capacity) {
        entry->staging_resources[entry->staging_count++] = res;
        if (!entry->st_on) {
            pthread_mutex_lock(&g_cmdbuf_mutex);
            staged_link(entry);
            pthread_mutex_unlock(&g_cmdbuf_mutex);
        }
    }
}

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

static void destroy_staging(struct DeviceData *d, struct StagingResource *r) {
    if (r->buffer) d->fpDestroyBuffer(d->device, r->buffer, NULL);
    if (r->memory) {
        d->fpFreeMemory(d->device, r->memory, NULL);
        if (r->mem_size) budget_sub(r->heap, r->mem_size);
    }
}

// Back into the device's pool, or freed when the pool is full.
static void release_staging(struct DeviceData *d, struct StagingResource *r) {
    struct StagingPoolNode *n = NULL;
    pthread_mutex_lock(&g_staging_pool_mutex);
    if (r->size && d->staging_pool_bytes + r->size <= STAGING_POOL_CAP && (n = malloc(sizeof *n))) {
        n->r = *r;
        n->next = d->staging_pool;
        d->staging_pool = n;
        d->staging_pool_bytes += r->size;
    }
    pthread_mutex_unlock(&g_staging_pool_mutex);
    if (!n) destroy_staging(d, r);
}

static uint64_t now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}

// A staging buffer of at least `size` bytes with a device address: a pooled one of the same bucket, or a new one.
static VkResult acquire_staging(struct DeviceData *d, VkDeviceSize size, struct StagingResource *out) {
    VkDeviceSize bucket = STAGING_MIN;
    while (bucket < size) bucket <<= 1;
    pthread_mutex_lock(&g_staging_pool_mutex);
    for (struct StagingPoolNode **pp = &d->staging_pool; *pp; pp = &(*pp)->next) {
        if ((*pp)->r.size != bucket) continue;
        struct StagingPoolNode *n = *pp;
        *pp = n->next;
        d->staging_pool_bytes -= bucket;
        pthread_mutex_unlock(&g_staging_pool_mutex);
        *out = n->r;
        free(n);
        if (g_stats) atomic_fetch_add(&g_st_reused, 1);
        return VK_SUCCESS;
    }
    pthread_mutex_unlock(&g_staging_pool_mutex);

    *out = (struct StagingResource){.size = bucket};
    VkBufferCreateInfo b_ci = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, NULL, 0, bucket,
                               VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT};
    VkResult r = d->fpCreateBuffer(d->device, &b_ci, NULL, &out->buffer);
    if (r == VK_SUCCESS) {
        VkMemoryRequirements mem_req = {0};
        d->fpGetBufferMemoryRequirements(d->device, out->buffer, &mem_req);
        uint32_t type = d->device_local_memory_index;
        if (!(mem_req.memoryTypeBits & (1u << type))) type = (uint32_t)__builtin_ctz(mem_req.memoryTypeBits);
        VkMemoryAllocateFlagsInfo flags_info = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO, NULL, VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT};
        VkMemoryAllocateInfo a_info = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, &flags_info, mem_req.size, type};
        r = d->fpAllocateMemory(d->device, &a_info, NULL, &out->memory);
        if (r == VK_SUCCESS) {
            out->mem_size = mem_req.size;
            out->heap = type < VK_MAX_MEMORY_TYPES ? d->type_heap[type] : BUDGET_NO_HEAP;
            budget_add(out->heap, out->mem_size);
        }
    }
    if (r == VK_SUCCESS) r = d->fpBindBufferMemory(d->device, out->buffer, out->memory, 0);
    if (r != VK_SUCCESS) {
        destroy_staging(d, out);
        return r;
    }
    VkBufferDeviceAddressInfo bda = {VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO, NULL, out->buffer};
    out->addr = d->fpGetBufferDeviceAddress(d->device, &bda);
    if (g_stats) atomic_fetch_add(&g_st_new, 1);
    return VK_SUCCESS;
}

static void destroy_staging_pool(struct DeviceData *d) {
    pthread_mutex_lock(&g_staging_pool_mutex);
    struct StagingPoolNode *n = d->staging_pool;
    d->staging_pool = NULL;
    d->staging_pool_bytes = 0;
    pthread_mutex_unlock(&g_staging_pool_mutex);
    while (n) {
        struct StagingPoolNode *next = n->next;
        destroy_staging(d, &n->r);
        free(n);
        n = next;
    }
}

// Begin/reset: staging buffers back to the pool, tracked state cleared; the entry stays. Free: the entry goes.
static void reset_cmd_data(struct DeviceData *dev_data, VkCommandBuffer cmd, bool remove) {
    pthread_mutex_lock(&g_cmdbuf_mutex);
    struct CmdBufData **pp = &g_cmdbufs[handle_bucket((uint64_t)(uintptr_t)cmd, CMDBUF_BUCKETS)];
    while (*pp && (*pp)->cmdbuf != cmd) pp = &(*pp)->next;
    struct CmdBufData *e = *pp;
    if (e) staged_unlink(e);
    if (e && remove) {
        *pp = e->next;
        atomic_fetch_add_explicit(&g_cmdbuf_gen, 1, memory_order_release);
    }
    pthread_mutex_unlock(&g_cmdbuf_mutex);
    if (!e) return;
    for (uint32_t j = 0; j < e->staging_count; j++) release_staging(dev_data, &e->staging_resources[j]);
    e->staging_count = 0;
    e->compute_pipeline = VK_NULL_HANDLE;
    e->descriptor_buffers = false;
    e->push_count = 0;
    e->cpd.layout = VK_NULL_HANDLE;
    e->gpd.layout = VK_NULL_HANDLE;
    e->gfx_pipeline = VK_NULL_HANDLE;
    e->gfx_flags = 0;
    e->gt_rec = -1;
    e->gt_src = VK_NULL_HANDLE;
    e->gt_sdraws = e->gt_svcount = e->gt_sicount = 0;
    memset(&e->gt_scnt, 0, sizeof e->gt_scnt);
    e->gt_spipe = VK_NULL_HANDLE;
    e->gt_ssrc = VK_NULL_HANDLE;
    e->async_skip = false;
    memset(&e->dc, 0, sizeof e->dc);
    memset(&e->ct, 0, sizeof e->ct);
    if (remove) {
        free(e->staging_resources);
        free(e);
    }
}

// Memory budget tracking: the old report (bytes ever allocated, every heap). It stays the reported value while
// MALI_COMPAT_BUDGET_MB is unset (deployed 2dfc35b8 behaviour); the live per-heap accounting is g_heap_usage.
static _Atomic uint64_t g_allocated_bytes = 0;

// -----------------------------------------------------------------------------
// BC Compute Pipeline Initializer
// -----------------------------------------------------------------------------
static VkResult init_bc_compute(struct DeviceData *d) {
    if (d->bc_init_done) return VK_SUCCESS;

    // Find a memory type with DEVICE_LOCAL
    VkPhysicalDeviceMemoryProperties mem_props;
    d->instance_data->fpGetPhysicalDeviceMemoryProperties(d->physical_device, &mem_props);
    d->device_local_memory_index = 0;
    for (uint32_t i = 0; i < mem_props.memoryTypeCount; i++) {
        if (mem_props.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) {
            d->device_local_memory_index = i;
            break;
        }
    }

    VkPushConstantRange push_range = {
        .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        .offset = 0,
        .size = sizeof(struct panvk_bc_decode_push), // 48 bytes
    };

    VkPipelineLayoutCreateInfo pl_info = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = 0,
        .pushConstantRangeCount = 1,
        .pPushConstantRanges = &push_range,
    };

    VkResult res = d->fpCreatePipelineLayout(d->device, &pl_info, NULL, &d->bc_layout);
    if (res != VK_SUCCESS) {
        fprintf(stderr, "mali-compat: BC decode: pipeline layout failed (%d); BC textures stay undecoded\n", res);
        return res;
    }

    static const struct {
        const uint32_t *code;
        size_t size;
    } spv[4] = {
        { panvk_bc_s3tc_spv, sizeof(panvk_bc_s3tc_spv) },
        { panvk_bc_rgtc_spv, sizeof(panvk_bc_rgtc_spv) },
        { panvk_bc_bc6_spv, sizeof(panvk_bc_bc6_spv) },
        { panvk_bc_bc7_spv, sizeof(panvk_bc_bc7_spv) },
    };

    for (int i = 0; i < 4; i++) {
        VkShaderModuleCreateInfo sm_info = {
            .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
            .codeSize = spv[i].size,
            .pCode = spv[i].code,
        };
        VkShaderModule mod = VK_NULL_HANDLE;
        res = d->fpCreateShaderModule(d->device, &sm_info, NULL, &mod);
        if (res != VK_SUCCESS) {
            fprintf(stderr, "mali-compat: BC decode: shader module %d failed (%d); BC textures stay undecoded\n", i, res);
            return res;
        }

        VkComputePipelineCreateInfo cp_info = {
            .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
            .stage = {
                .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                .stage = VK_SHADER_STAGE_COMPUTE_BIT,
                .module = mod,
                .pName = "main",
            },
            .layout = d->bc_layout,
        };

        res = d->fpCreateComputePipelines(d->device, VK_NULL_HANDLE, 1, &cp_info, NULL, &d->bc_pipelines[i]);
        if (res == VK_SUCCESS && d->descriptor_buffer) {
            // A pipeline without DESCRIPTOR_BUFFER_BIT must not be used where descriptor buffers are bound.
            cp_info.flags = VK_PIPELINE_CREATE_DESCRIPTOR_BUFFER_BIT_EXT;
            res = d->fpCreateComputePipelines(d->device, VK_NULL_HANDLE, 1, &cp_info, NULL, &d->bc_pipelines_db[i]);
        }
        d->fpDestroyShaderModule(d->device, mod, NULL);
        if (res != VK_SUCCESS) {
            fprintf(stderr, "mali-compat: BC decode: compute pipeline %d failed (%d); BC textures stay undecoded\n", i, res);
            return res;
        }
    }

    d->bc_init_done = true;
    LOG("initialized PanVK BC compute pipelines successfully");
    return VK_SUCCESS;
}

// -----------------------------------------------------------------------------
// Intercepted Vulkan Entry Points
// -----------------------------------------------------------------------------
// MALI_COMPAT_GS_HIDE_PROCS=<comm prefixes, comma-separated>: those processes are told the device has no geometry
// shaders. Xwayland's glamor draws GL_QUADS in a compatibility context and Mesa Zink turns them into geometry shaders;
// MaliBridge found that pipelines with a geometry stage crash some Mali drivers, so Xwayland segfaults (their
// "gs_hide"). Without the feature Zink takes its other path. For the session on the bridge. [Claude Opus 5.5] 2026-10-30
static bool gs_hidden(void) {
    static int hidden = -1;
    if (hidden >= 0) return hidden;
    const char *list = getenv("MALI_COMPAT_GS_HIDE_PROCS");
    char comm[32] = "";
    FILE *f = fopen("/proc/self/comm", "r");
    if (f) { if (fgets(comm, sizeof comm, f)) comm[strcspn(comm, "\n")] = 0; fclose(f); }
    hidden = 0;
    for (const char *p = list ? list : ""; *p; ) {
        size_t n = strcspn(p, ",");
        if (n && !strncmp(comm, p, n)) { hidden = 1; break; }
        p += n + (p[n] == ',');
    }
    if (hidden) fprintf(stderr, "mali-compat: geometry shaders hidden from '%s'\n", comm);
    return hidden;
}

// MALI_COMPAT_FP16_HIDE_PROCS=<comm prefixes>: shaderFloat16 / shaderInt16 hidden from those processes (gamescope
// picks its FP16 composite shaders when the device has both). Diagnostic for the session on the bridge, where
// gamescope's composited output stayed all zero. [Claude Opus 5.5] 2026-10-30
static bool fp16_hidden(void) {
    static int hidden = -1;
    if (hidden >= 0) return hidden;
    const char *list = getenv("MALI_COMPAT_FP16_HIDE_PROCS");
    char comm[32] = "";
    FILE *f = fopen("/proc/self/comm", "r");
    if (f) { if (fgets(comm, sizeof comm, f)) comm[strcspn(comm, "\n")] = 0; fclose(f); }
    hidden = 0;
    for (const char *p = list ? list : ""; *p; ) {
        size_t n = strcspn(p, ",");
        if (n && !strncmp(comm, p, n)) { hidden = 1; break; }
        p += n + (p[n] == ',');
    }
    if (hidden) fprintf(stderr, "mali-compat: FP16 hidden from '%s'\n", comm);
    return hidden;
}

static VKAPI_ATTR void VKAPI_CALL compat_GetPhysicalDeviceFeatures(
    VkPhysicalDevice physicalDevice, VkPhysicalDeviceFeatures *pFeatures) {
    struct InstanceData *inst = get_instance_by_phys_dev(physicalDevice);
    if (inst && inst->fpGetPhysicalDeviceFeatures)
        inst->fpGetPhysicalDeviceFeatures(physicalDevice, pFeatures);

    pFeatures->textureCompressionBC = VK_TRUE;
    pFeatures->shaderClipDistance = VK_TRUE;
    pFeatures->shaderCullDistance = VK_TRUE;
    pFeatures->fillModeNonSolid = VK_TRUE;
    if (gs_hidden()) pFeatures->geometryShader = VK_FALSE;
    if (fp16_hidden()) pFeatures->shaderInt16 = VK_FALSE;
}

static VKAPI_ATTR void VKAPI_CALL compat_GetPhysicalDeviceFeatures2(
    VkPhysicalDevice physicalDevice, VkPhysicalDeviceFeatures2 *pFeatures) {
    struct InstanceData *inst = get_instance_by_phys_dev(physicalDevice);
    struct unlinked u = unlink_struct(pFeatures, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAP_MEMORY_PLACED_FEATURES_EXT);
    if (inst && inst->fpGetPhysicalDeviceFeatures2)
        inst->fpGetPhysicalDeviceFeatures2(physicalDevice, pFeatures);
    relink(u);

    pFeatures->features.textureCompressionBC = VK_TRUE;
    pFeatures->features.shaderClipDistance = VK_TRUE;
    pFeatures->features.shaderCullDistance = VK_TRUE;
    pFeatures->features.fillModeNonSolid = VK_TRUE;
    if (gs_hidden()) pFeatures->features.geometryShader = VK_FALSE;
    if (fp16_hidden()) {
        pFeatures->features.shaderInt16 = VK_FALSE;
        for (VkBaseOutStructure *h = (VkBaseOutStructure *)pFeatures->pNext; h; h = h->pNext) {
            if (h->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES)
                ((VkPhysicalDeviceVulkan12Features *)h)->shaderFloat16 = VK_FALSE;
            else if (h->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES)
                ((VkPhysicalDeviceShaderFloat16Int8Features *)h)->shaderFloat16 = VK_FALSE;
        }
    }

    // Scan pNext for VkPhysicalDeviceSwapchainMaintenance1FeaturesEXT
    VkBaseOutStructure *header = (VkBaseOutStructure *)pFeatures->pNext;
    while (header) {
        if (header->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SWAPCHAIN_MAINTENANCE_1_FEATURES_EXT) {
            VkPhysicalDeviceSwapchainMaintenance1FeaturesEXT *m1 =
                (VkPhysicalDeviceSwapchainMaintenance1FeaturesEXT *)header;
            m1->swapchainMaintenance1 = VK_TRUE;
        } else if (header->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VERTEX_ATTRIBUTE_DIVISOR_FEATURES_KHR) {
            // Zero divisor is emulated with divisor UINT32_MAX (the vendor's maxVertexAttribDivisor): attribute index
            // (instance - firstInstance) / 0xFFFFFFFF + firstInstance = firstInstance, as for divisor 0.
            // vkd3d-proton needs it (HZD: "Lacking support for VK_EXT_vertex_attribute_divisor"). [Claude Opus 5.5]
            VkPhysicalDeviceVertexAttributeDivisorFeaturesKHR *dv = (VkPhysicalDeviceVertexAttributeDivisorFeaturesKHR *)header;
            if (dv->vertexAttributeInstanceRateDivisor) dv->vertexAttributeInstanceRateZeroDivisor = VK_TRUE;
        } else if (header->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAP_MEMORY_PLACED_FEATURES_EXT) {
            VkPhysicalDeviceMapMemoryPlacedFeaturesEXT *mp = (VkPhysicalDeviceMapMemoryPlacedFeaturesEXT *)header;
            mp->memoryMapPlaced = VK_TRUE;
            mp->memoryMapRangePlaced = VK_FALSE;
            mp->memoryUnmapReserve = VK_TRUE;
        } else if (header->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES) {
            // DOOM Eternal (idTech 7) checks storageBuffer16BitAccess via vkGetPhysicalDeviceFeatures2.
            VkPhysicalDevice16BitStorageFeatures *f16 = (VkPhysicalDevice16BitStorageFeatures *)header;
            f16->storageBuffer16BitAccess = VK_TRUE;
            f16->uniformAndStorageBuffer16BitAccess = VK_TRUE;
            f16->storagePushConstant16 = VK_TRUE;
            f16->storageInputOutput16 = VK_TRUE;
        } else if (header->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_8BIT_STORAGE_FEATURES) {
            VkPhysicalDevice8BitStorageFeatures *f8 = (VkPhysicalDevice8BitStorageFeatures *)header;
            f8->storageBuffer8BitAccess = VK_TRUE;
            f8->uniformAndStorageBuffer8BitAccess = VK_TRUE;
            f8->storagePushConstant8 = VK_TRUE;
        } else if (header->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES) {
            VkPhysicalDeviceVulkan11Features *v11 = (VkPhysicalDeviceVulkan11Features *)header;
            v11->storageBuffer16BitAccess = VK_TRUE;
            v11->uniformAndStorageBuffer16BitAccess = VK_TRUE;
            v11->storagePushConstant16 = VK_TRUE;
            v11->storageInputOutput16 = VK_TRUE;
        } else if (header->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES) {
            VkPhysicalDeviceVulkan12Features *v12 = (VkPhysicalDeviceVulkan12Features *)header;
            v12->storageBuffer8BitAccess = VK_TRUE;
            v12->uniformAndStorageBuffer8BitAccess = VK_TRUE;
            v12->storagePushConstant8 = VK_TRUE;
        }
        header = header->pNext;
    }
}

// MALI_COMPAT_VENDOR_ID=<hex>: report this PCI vendor ID instead of Arm's 0x13b5 (driverID stays ARM_PROPRIETARY).
// vkd3d-proton reads root CBVs through buffer device addresses on every vendor except NVIDIA and Qualcomm, which
// get push descriptors ("~15%" there; Mali can only preload UBOs into uniform registers). 5143 flips only that
// choice: vkd3d's other Qualcomm check also needs driverID TURNIP. The game sees the vendor through DXGI too
// (dxgi.customVendorId keeps it Arm). [Claude Opus 5.5] 2026-10-28
// When MALI_COMPAT_VENDOR_ID is set (e.g. 10de for NVIDIA, 1002 for AMD), MALI_COMPAT_DEVICE_ID and
// MALI_COMPAT_DEVICE_NAME can optionally override the device, or sensible desktop defaults are chosen so native
// Vulkan games (DOOM Eternal / idTech) recognize the GPU.
static uint32_t g_vendor_id;
static uint32_t g_device_id;
static const char *g_device_name;

static void apply_properties_overrides(VkPhysicalDeviceProperties *p) {
    p->limits.minTexelBufferOffsetAlignment = 16;
    if (p->limits.minStorageBufferOffsetAlignment > 16)
        p->limits.minStorageBufferOffsetAlignment = 16;
    if (g_vendor_id) {
        p->vendorID = g_vendor_id;
        if (p->limits.maxUniformBufferRange > 65536)
            p->limits.maxUniformBufferRange = 65536;
        if (g_device_id) {
            p->deviceID = g_device_id;
        } else if (g_vendor_id == 0x10de) {
            p->deviceID = 0x1eb1; // NVIDIA GeForce RTX 2080
            p->deviceType = VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU;
            p->driverVersion = (550 << 22) | (54 << 14) | (14 << 6);
        } else if (g_vendor_id == 0x1002) {
            p->deviceID = 0x73bf; // AMD Radeon RX 6800
            p->deviceType = VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU;
            p->driverVersion = (24 << 22) | (3 << 12) | 1;
        } else if (g_vendor_id == 0x8086) {
            p->deviceID = 0x56a0; // Intel Arc A770
            p->deviceType = VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU;
            p->driverVersion = (31 << 22) | (0 << 12) | 101;
        }
        if (g_device_name && *g_device_name) {
            snprintf(p->deviceName, sizeof(p->deviceName), "%s", g_device_name);
        } else if (g_vendor_id == 0x10de) {
            snprintf(p->deviceName, sizeof(p->deviceName), "NVIDIA GeForce RTX 2080");
        } else if (g_vendor_id == 0x1002) {
            snprintf(p->deviceName, sizeof(p->deviceName), "AMD Radeon RX 6800");
        } else if (g_vendor_id == 0x8086) {
            snprintf(p->deviceName, sizeof(p->deviceName), "Intel(R) Arc(TM) A770 Graphics");
        }
    }
}
// GPU timing diagnostics for instanced vertex input (HZD's missing static meshes, 2026-10-29): this driver has
// supportsNonZeroFirstInstance = false, so a draw with firstInstance != 0 on a binding whose divisor is not 1 (the
// emulated zero divisor included) reads undefined instance data. Pipelines and dynamic vertex inputs with such
// bindings, and draws with firstInstance != 0.
static _Atomic uint64_t g_div_pipes, g_div_vi, g_div_draws;
// MALI_COMPAT_NO_SUSPEND_RESUME=1 (own switch for the CmdBeginRendering hook since 2026-10-31, no GPU_TIME needed;
// without GPU_TIME the hook does no timing work, [keeper, Claude Opus 5.5]): vkd3d-proton splits render passes on
// tilers with VK_RENDERING_SUSPENDING/RESUMING_BIT. Turn each part into a plain pass: a suspending part stores its
// attachments, a resuming part loads them, and only the last part resolves. Same result, more bandwidth; tests whether
// this driver loses draws in resumed passes (HZD's missing static meshes, 2026-10-29). Counters of such passes.
static bool g_no_susres;
static _Atomic uint64_t g_susp_passes, g_resu_passes;
// MALI_COMPAT_SUSRES_BARRIER=1 (K-CS1, 2026-10-31, [keeper, Claude Opus 5.5]; only with NO_SUSPEND_RESUME): as plain
// passes, the parts of a split render pass are separate render pass instances, so the previous part's attachment writes
// (draws, STORE) and the next part's LOAD and draws are a hazard. vkd3d-proton cannot record a barrier there (forbidden
// between suspend and resume), so the layer inserts one global memory barrier before each converted resuming part.
// Control's 16384x2048 d32 shadow atlas is split into 7-9 parts per frame past the gate (shadow flicker, control.md).
static bool g_susres_barrier;
static _Atomic uint64_t g_susres_barriers;
static __thread bool tl_div_bound; // the last vertex input this thread set or bound has a divisor != 1 (approximate)

// VK_EXT_physical_device_drm, emulated (MALI_COMPAT_DRM_NODE=<device node>, e.g. /dev/mali0). gamescope refuses a
// physical device without a DRM primary and render node and then finds the render node through
// drmGetDeviceFromDevId(); DroidDeck exposes the GPU node under /dev/dri with a matching /sys/dev/char tree (the
// stand-in PanVK reported for the session until now). The vendor driver has no such extension, so the layer reports
// the node's major:minor as both nodes, for the session processes (gamescope, Steam) on the bridge. [Claude Opus 5.5]
// 2026-10-30
static bool g_drm_on;
static unsigned g_drm_major, g_drm_minor;
static void drm_node_init(void) {
    const char *path = getenv("MALI_COMPAT_DRM_NODE");
    struct stat st;
    g_drm_on = path && *path && stat(path, &st) == 0 && S_ISCHR(st.st_mode);
    if (g_drm_on) { g_drm_major = major(st.st_rdev); g_drm_minor = minor(st.st_rdev); }
    else if (path && *path) fprintf(stderr, "mali-compat: MALI_COMPAT_DRM_NODE=%s is not a character device\n", path);
}

static VKAPI_ATTR void VKAPI_CALL compat_GetPhysicalDeviceProperties2(
    VkPhysicalDevice physicalDevice, VkPhysicalDeviceProperties2 *pProperties) {
    struct InstanceData *inst = get_instance_by_phys_dev(physicalDevice);
    struct unlinked u = unlink_struct(pProperties, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAP_MEMORY_PLACED_PROPERTIES_EXT);
    struct unlinked ud = g_drm_on ? unlink_struct(pProperties, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRM_PROPERTIES_EXT)
                                  : (struct unlinked){0};
    if (inst && inst->fpGetPhysicalDeviceProperties2)
        inst->fpGetPhysicalDeviceProperties2(physicalDevice, pProperties);
    relink(ud);
    relink(u);
    if (ud.node) {
        VkPhysicalDeviceDrmPropertiesEXT *drm = (VkPhysicalDeviceDrmPropertiesEXT *)ud.node;
        drm->hasPrimary = drm->hasRender = VK_TRUE;
        drm->primaryMajor = drm->renderMajor = g_drm_major;
        drm->primaryMinor = drm->renderMinor = g_drm_minor;
    }
    if (u.node) ((VkPhysicalDeviceMapMemoryPlacedPropertiesEXT *)u.node)->minPlacedMemoryMapAlignment = page_size();
    apply_properties_overrides(&pProperties->properties);
    for (VkBaseOutStructure *h = (VkBaseOutStructure *)pProperties->pNext; h; h = h->pNext) {
        if (h->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TEXEL_BUFFER_ALIGNMENT_PROPERTIES) {
            VkPhysicalDeviceTexelBufferAlignmentProperties *t = (VkPhysicalDeviceTexelBufferAlignmentProperties *)h;
            t->storageTexelBufferOffsetAlignmentBytes = 16;
            t->storageTexelBufferOffsetSingleTexelAlignment = VK_TRUE;
            t->uniformTexelBufferOffsetSingleTexelAlignment = VK_TRUE;
        } else if (h->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_PROPERTIES) {
            VkPhysicalDeviceVulkan13Properties *v = (VkPhysicalDeviceVulkan13Properties *)h;
            v->storageTexelBufferOffsetAlignmentBytes = 16;
            v->storageTexelBufferOffsetSingleTexelAlignment = VK_TRUE;
            v->uniformTexelBufferOffsetSingleTexelAlignment = VK_TRUE;
        } else if (h->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES) {
            VkPhysicalDeviceSubgroupProperties *s = (VkPhysicalDeviceSubgroupProperties *)h;
            s->supportedStages |= (VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT |
                                   VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT | VK_SHADER_STAGE_GEOMETRY_BIT |
                                   VK_SHADER_STAGE_FRAGMENT_BIT | VK_SHADER_STAGE_COMPUTE_BIT);
            s->supportedOperations |= (VK_SUBGROUP_FEATURE_BASIC_BIT |
                                       VK_SUBGROUP_FEATURE_VOTE_BIT |
                                       VK_SUBGROUP_FEATURE_ARITHMETIC_BIT |
                                       VK_SUBGROUP_FEATURE_BALLOT_BIT |
                                       VK_SUBGROUP_FEATURE_SHUFFLE_BIT |
                                       VK_SUBGROUP_FEATURE_SHUFFLE_RELATIVE_BIT |
                                       VK_SUBGROUP_FEATURE_CLUSTERED_BIT |
                                       VK_SUBGROUP_FEATURE_QUAD_BIT);
        } else if (h->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_PROPERTIES) {
            VkPhysicalDeviceVulkan11Properties *v11 = (VkPhysicalDeviceVulkan11Properties *)h;
            v11->subgroupSupportedStages |= (VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT |
                                             VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT | VK_SHADER_STAGE_GEOMETRY_BIT |
                                             VK_SHADER_STAGE_FRAGMENT_BIT | VK_SHADER_STAGE_COMPUTE_BIT);
            v11->subgroupSupportedOperations |= (VK_SUBGROUP_FEATURE_BASIC_BIT |
                                                 VK_SUBGROUP_FEATURE_VOTE_BIT |
                                                 VK_SUBGROUP_FEATURE_ARITHMETIC_BIT |
                                                 VK_SUBGROUP_FEATURE_BALLOT_BIT |
                                                 VK_SUBGROUP_FEATURE_SHUFFLE_BIT |
                                                 VK_SUBGROUP_FEATURE_SHUFFLE_RELATIVE_BIT |
                                                 VK_SUBGROUP_FEATURE_CLUSTERED_BIT |
                                                 VK_SUBGROUP_FEATURE_QUAD_BIT);
        }
    }
    // VK_EXT_vertex_attribute_divisor properties from the vendor's KHR struct.
    VkPhysicalDeviceVertexAttributeDivisorPropertiesEXT *div =
        (VkPhysicalDeviceVertexAttributeDivisorPropertiesEXT *)find_struct(pProperties->pNext,
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VERTEX_ATTRIBUTE_DIVISOR_PROPERTIES_EXT);
    if (div && inst && inst->fpGetPhysicalDeviceProperties2) {
        VkPhysicalDeviceVertexAttributeDivisorPropertiesKHR khr = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VERTEX_ATTRIBUTE_DIVISOR_PROPERTIES_KHR};
        VkPhysicalDeviceProperties2 p2 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, &khr};
        inst->fpGetPhysicalDeviceProperties2(physicalDevice, &p2);
        div->maxVertexAttribDivisor = khr.maxVertexAttribDivisor;
    }
}

static VKAPI_ATTR void VKAPI_CALL compat_GetPhysicalDeviceProperties(
    VkPhysicalDevice physicalDevice, VkPhysicalDeviceProperties *pProperties) {
    struct InstanceData *inst = get_instance_by_phys_dev(physicalDevice);
    if (inst && inst->fpGetPhysicalDeviceProperties) {
        inst->fpGetPhysicalDeviceProperties(physicalDevice, pProperties);
    } else if (inst && inst->fpGetPhysicalDeviceProperties2) {
        VkPhysicalDeviceProperties2 p2 = { .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2 };
        compat_GetPhysicalDeviceProperties2(physicalDevice, &p2);
        *pProperties = p2.properties;
        return;
    }
    apply_properties_overrides(pProperties);
}

static inline bool is_yuv_nv12(VkFormat f) {
    return f == VK_FORMAT_G8_B8R8_2PLANE_420_UNORM || f == VK_FORMAT_G8_B8_R8_3PLANE_420_UNORM;
}

static VKAPI_ATTR void VKAPI_CALL compat_GetPhysicalDeviceFormatProperties(
    VkPhysicalDevice physicalDevice, VkFormat format, VkFormatProperties *pFormatProperties) {
    struct InstanceData *inst = get_instance_by_phys_dev(physicalDevice);
    if (!inst || !inst->fpGetPhysicalDeviceFormatProperties) return;
    if (is_yuv_nv12(format)) {
        inst->fpGetPhysicalDeviceFormatProperties(physicalDevice, format, pFormatProperties);
        pFormatProperties->optimalTilingFeatures |= (VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT |
                                                    VK_FORMAT_FEATURE_TRANSFER_DST_BIT |
                                                    VK_FORMAT_FEATURE_TRANSFER_SRC_BIT |
                                                    VK_FORMAT_FEATURE_COSITED_CHROMA_SAMPLES_BIT |
                                                    VK_FORMAT_FEATURE_MIDPOINT_CHROMA_SAMPLES_BIT);
        return;
    }
    if (panvk_format_is_bc(format)) {
        VkFormat decoded = panvk_bc_decoded_format(format);
        inst->fpGetPhysicalDeviceFormatProperties(physicalDevice, decoded, pFormatProperties);
        pFormatProperties->linearTilingFeatures = 0;
        pFormatProperties->optimalTilingFeatures = (VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT |
                                                   VK_FORMAT_FEATURE_BLIT_SRC_BIT |
                                                   VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT |
                                                   VK_FORMAT_FEATURE_TRANSFER_SRC_BIT |
                                                   VK_FORMAT_FEATURE_TRANSFER_DST_BIT);
        return;
    }
    inst->fpGetPhysicalDeviceFormatProperties(physicalDevice, format, pFormatProperties);
}

static VKAPI_ATTR void VKAPI_CALL compat_GetPhysicalDeviceFormatProperties2(
    VkPhysicalDevice physicalDevice, VkFormat format, VkFormatProperties2 *pFormatProperties) {
    struct InstanceData *inst = get_instance_by_phys_dev(physicalDevice);
    if (!inst || !inst->fpGetPhysicalDeviceFormatProperties2) return;
    if (is_yuv_nv12(format)) {
        inst->fpGetPhysicalDeviceFormatProperties2(physicalDevice, format, pFormatProperties);
        pFormatProperties->formatProperties.optimalTilingFeatures |= (VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT |
                                                                     VK_FORMAT_FEATURE_TRANSFER_DST_BIT |
                                                                     VK_FORMAT_FEATURE_TRANSFER_SRC_BIT |
                                                                     VK_FORMAT_FEATURE_COSITED_CHROMA_SAMPLES_BIT |
                                                                     VK_FORMAT_FEATURE_MIDPOINT_CHROMA_SAMPLES_BIT);
        return;
    }
    if (panvk_format_is_bc(format)) {
        VkFormat decoded = panvk_bc_decoded_format(format);
        inst->fpGetPhysicalDeviceFormatProperties2(physicalDevice, decoded, pFormatProperties);
        pFormatProperties->formatProperties.linearTilingFeatures = 0;
        pFormatProperties->formatProperties.optimalTilingFeatures = (VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT |
                                                                    VK_FORMAT_FEATURE_BLIT_SRC_BIT |
                                                                    VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT |
                                                                    VK_FORMAT_FEATURE_TRANSFER_SRC_BIT |
                                                                    VK_FORMAT_FEATURE_TRANSFER_DST_BIT);
        return;
    }
    inst->fpGetPhysicalDeviceFormatProperties2(physicalDevice, format, pFormatProperties);
}

static VKAPI_ATTR VkResult VKAPI_CALL compat_GetPhysicalDeviceImageFormatProperties(
    VkPhysicalDevice physicalDevice, VkFormat format, VkImageType type, VkImageTiling tiling,
    VkImageUsageFlags usage, VkImageCreateFlags flags, VkImageFormatProperties *pImageFormatProperties) {
    struct InstanceData *inst = get_instance_by_phys_dev(physicalDevice);
    if (!inst || !inst->fpGetPhysicalDeviceImageFormatProperties) return VK_ERROR_INITIALIZATION_FAILED;
    if (is_yuv_nv12(format)) {
        VkResult r = inst->fpGetPhysicalDeviceImageFormatProperties(
            physicalDevice, format, type, tiling, usage, flags, pImageFormatProperties);
        if (r != VK_SUCCESS && pImageFormatProperties) {
            pImageFormatProperties->maxExtent = (VkExtent3D){4096, 4096, 1};
            pImageFormatProperties->maxMipLevels = 1;
            pImageFormatProperties->maxArrayLayers = 1;
            pImageFormatProperties->sampleCounts = VK_SAMPLE_COUNT_1_BIT;
            pImageFormatProperties->maxResourceSize = 4096 * 4096 * 2;
            return VK_SUCCESS;
        }
        return r;
    }
    if (panvk_format_is_bc(format)) {
        format = panvk_bc_decoded_format(format);
        flags &= ~VK_IMAGE_CREATE_BLOCK_TEXEL_VIEW_COMPATIBLE_BIT;
    }
    return inst->fpGetPhysicalDeviceImageFormatProperties(
        physicalDevice, format, type, tiling, usage, flags, pImageFormatProperties);
}

// -----------------------------------------------------------------------------
// dma-buf export, emulated (MALI_COMPAT_DMABUF_EXPORT=1). [Claude Opus 5.5] 2026-10-30
// The vendor driver imports linear dma-bufs (explicit DRM-modifier layout) but exports none: its external memory
// features for dma-buf are IMPORTABLE only, and AFBC/AFRC modifiers are refused (test/session-probe.c). gamescope's
// nested Wayland backend allocates its output images as exportable dma-bufs and hands them to the parent compositor,
// so on the bridge it asserted ("modifiers.size() > 0"). With the switch:
// - linear dma-buf image formats report EXPORTABLE too;
// - an exportable dma-buf image created from a modifier list gets an explicit linear layout (pitch aligned to 256);
// - its memory comes from the system dma-heap and is imported into the driver; vkGetMemoryFdKHR hands out that fd.
// Mesa's private WSI structures in gamescope's create infos are dropped on the way (the vendor does not know them).
// -----------------------------------------------------------------------------
static bool g_dmabuf_export;
#define STYPE_WSI_IMAGE_CREATE_INFO_MESA ((VkStructureType)1000001002)
#define STYPE_WSI_MEMORY_ALLOCATE_INFO_MESA ((VkStructureType)1000001003)
struct ExportMem { VkDeviceMemory mem; int fd; uint64_t size; bool imported; struct ExportMem *next; };
static struct ExportMem *g_export_mem;
static pthread_mutex_t g_export_mutex = PTHREAD_MUTEX_INITIALIZER;

struct dma_heap_alloc_data_ { uint64_t len; uint32_t fd; uint32_t fd_flags; uint64_t heap_flags; };
#define DMA_HEAP_IOCTL_ALLOC_ _IOWR('H', 0x0, struct dma_heap_alloc_data_)
static int dmaheap_alloc(uint64_t size) {
    static int heap = -2;
    if (heap == -2) {
        const char *path = getenv("MALI_COMPAT_DMABUF_HEAP") ? getenv("MALI_COMPAT_DMABUF_HEAP") : "/dev/dma_heap/system";
        heap = open(path, O_RDONLY | O_CLOEXEC);
        if (heap < 0) fprintf(stderr, "mali-compat: dma-buf export: cannot open %s: %s\n", path, strerror(errno));
    }
    if (heap < 0) return -1;
    struct dma_heap_alloc_data_ a = {(size + 4095) & ~4095ull, 0, O_RDWR | O_CLOEXEC, 0};
    if (ioctl(heap, DMA_HEAP_IOCTL_ALLOC_, &a) < 0) {
        fprintf(stderr, "mali-compat: dma-buf export: heap allocation of %llu bytes failed: %s\n", (unsigned long long)size, strerror(errno));
        return -1;
    }
    return (int)a.fd;
}

static uint32_t format_bytes(VkFormat f) {
    switch (f) {
    case VK_FORMAT_R8_UNORM: return 1;
    case VK_FORMAT_R8G8_UNORM: return 2;
    case VK_FORMAT_B8G8R8A8_UNORM: case VK_FORMAT_B8G8R8A8_SRGB: case VK_FORMAT_R8G8B8A8_UNORM: case VK_FORMAT_R8G8B8A8_SRGB:
    case VK_FORMAT_A2B10G10R10_UNORM_PACK32: case VK_FORMAT_A2R10G10B10_UNORM_PACK32: return 4;
    case VK_FORMAT_R16G16B16A16_SFLOAT: return 8;
    default: return 0;
    }
}

// MALI_COMPAT_DMABUF_PEEK=1 (diagnostic): every 5 s, one line per exported buffer with how much of it is non-zero
// and a few pixels, read through a CPU mapping of the dma-buf (with DMA_BUF_IOCTL_SYNC). Tells whether gamescope's
// output images hold a picture before DroidDeck's compositor gets them.
struct dma_buf_sync_ { uint64_t flags; };
#define DMA_BUF_IOCTL_SYNC_ _IOW('b', 0, struct dma_buf_sync_)
static void *dmabuf_peek_thread(void *arg) {
    (void)arg;
    for (;;) {
        sleep(5);
        pthread_mutex_lock(&g_export_mutex);
        int n = 0;
        for (struct ExportMem *x = g_export_mem; x && n < 12; x = x->next, n++) {
            void *p = mmap(NULL, x->size, PROT_READ, MAP_SHARED, x->fd, 0);
            if (p == MAP_FAILED) { fprintf(stderr, "mali-compat: dma-buf peek: mmap failed: %s\n", strerror(errno)); continue; }
            struct dma_buf_sync_ sy = {1 /* START */ | 1u << 0 /* READ */};
            ioctl(x->fd, DMA_BUF_IOCTL_SYNC_, &sy);
            const uint32_t *w = p;
            uint64_t words = x->size / 4, nz = 0;
            for (uint64_t i = 0; i < words; i += 61) nz += w[i] != 0;
            static char comm[32];
            if (!comm[0]) {
                FILE *cf = fopen("/proc/self/comm", "r");
                if (!cf || !fgets(comm, sizeof comm, cf)) snprintf(comm, sizeof comm, "?");
                if (cf) fclose(cf);
                comm[strcspn(comm, "\n")] = 0;
            }
            fprintf(stderr, "mali-compat: dma-buf peek [%s %d]: %s mem %p %llu KiB, non-zero %llu of %llu samples, px[0] %08x px[mid] %08x px[3/4] %08x\n",
                    comm, (int)getpid(), x->imported ? "imported" : "exported", (void *)x->mem, (unsigned long long)(x->size >> 10), (unsigned long long)nz,
                    (unsigned long long)(words / 61 + 1), w[0], w[words / 2], w[words * 3 / 4]);
            sy.flags = 2 /* END */ | 1;
            ioctl(x->fd, DMA_BUF_IOCTL_SYNC_, &sy);
            munmap(p, x->size);
        }
        pthread_mutex_unlock(&g_export_mutex);
    }
    return NULL;
}

static bool dmabuf_handle(const void *chain) {
    const VkPhysicalDeviceExternalImageFormatInfo *e = find_struct(chain, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_IMAGE_FORMAT_INFO);
    return e && e->handleType == VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT;
}

static VKAPI_ATTR VkResult VKAPI_CALL compat_GetPhysicalDeviceImageFormatProperties2(
    VkPhysicalDevice physicalDevice, const VkPhysicalDeviceImageFormatInfo2 *pImageFormatInfo,
    VkImageFormatProperties2 *pImageFormatProperties) {
    struct InstanceData *inst = get_instance_by_phys_dev(physicalDevice);
    if (!inst || !inst->fpGetPhysicalDeviceImageFormatProperties2) return VK_ERROR_INITIALIZATION_FAILED;
    if (is_yuv_nv12(pImageFormatInfo->format)) {
        VkResult r = inst->fpGetPhysicalDeviceImageFormatProperties2(physicalDevice, pImageFormatInfo, pImageFormatProperties);
        if (r != VK_SUCCESS && pImageFormatProperties) {
            pImageFormatProperties->imageFormatProperties.maxExtent = (VkExtent3D){4096, 4096, 1};
            pImageFormatProperties->imageFormatProperties.maxMipLevels = 1;
            pImageFormatProperties->imageFormatProperties.maxArrayLayers = 1;
            pImageFormatProperties->imageFormatProperties.sampleCounts = VK_SAMPLE_COUNT_1_BIT;
            pImageFormatProperties->imageFormatProperties.maxResourceSize = 4096 * 4096 * 2;
            return VK_SUCCESS;
        }
        return r;
    }
    if (panvk_format_is_bc(pImageFormatInfo->format)) {
        VkPhysicalDeviceImageFormatInfo2 mod = *pImageFormatInfo;
        mod.format = panvk_bc_decoded_format(pImageFormatInfo->format);
        mod.flags &= ~VK_IMAGE_CREATE_BLOCK_TEXEL_VIEW_COMPATIBLE_BIT;
        return inst->fpGetPhysicalDeviceImageFormatProperties2(physicalDevice, &mod, pImageFormatProperties);
    }
    VkResult r = inst->fpGetPhysicalDeviceImageFormatProperties2(physicalDevice, pImageFormatInfo, pImageFormatProperties);
    if (r == VK_SUCCESS && g_dmabuf_export && dmabuf_handle(pImageFormatInfo->pNext)) {
        const VkPhysicalDeviceImageDrmFormatModifierInfoEXT *mi =
            find_struct(pImageFormatInfo->pNext, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_IMAGE_DRM_FORMAT_MODIFIER_INFO_EXT);
        bool linear = pImageFormatInfo->tiling == VK_IMAGE_TILING_LINEAR ||
                      (pImageFormatInfo->tiling == VK_IMAGE_TILING_DRM_FORMAT_MODIFIER_EXT && mi && mi->drmFormatModifier == 0);
        VkExternalImageFormatProperties *ep = (VkExternalImageFormatProperties *)find_struct(
            pImageFormatProperties->pNext, VK_STRUCTURE_TYPE_EXTERNAL_IMAGE_FORMAT_PROPERTIES);
        if (linear && ep && format_bytes(pImageFormatInfo->format)) {
            ep->externalMemoryProperties.externalMemoryFeatures |= VK_EXTERNAL_MEMORY_FEATURE_EXPORTABLE_BIT |
                                                                   VK_EXTERNAL_MEMORY_FEATURE_IMPORTABLE_BIT;
            ep->externalMemoryProperties.compatibleHandleTypes |= VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT;
            ep->externalMemoryProperties.exportFromImportedHandleTypes |= VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT;
        }
    }
    return r;
}

static VKAPI_ATTR void VKAPI_CALL compat_GetPhysicalDeviceMemoryProperties2(
    VkPhysicalDevice physicalDevice, VkPhysicalDeviceMemoryProperties2 *pMemoryProperties) {
    struct InstanceData *inst = get_instance_by_phys_dev(physicalDevice);
    if (!inst || !inst->fpGetPhysicalDeviceMemoryProperties2) return;
    inst->fpGetPhysicalDeviceMemoryProperties2(physicalDevice, pMemoryProperties);

    // Support VK_EXT_memory_budget
    VkBaseOutStructure *header = (VkBaseOutStructure *)pMemoryProperties->pNext;
    while (header) {
        if (header->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_BUDGET_PROPERTIES_EXT) {
            VkPhysicalDeviceMemoryBudgetPropertiesEXT *budget =
                (VkPhysicalDeviceMemoryBudgetPropertiesEXT *)header;
            uint64_t allocated = atomic_load(&g_allocated_bytes);
            atomic_fetch_add(&g_budget_queries, 1);
            for (uint32_t i = 0; i < VK_MAX_MEMORY_HEAPS; i++) {
                if (i < pMemoryProperties->memoryProperties.memoryHeapCount) {
                    VkDeviceSize heap_size = pMemoryProperties->memoryProperties.memoryHeaps[i].size;
                    if (g_budget_accounted) { // MALI_COMPAT_BUDGET_MB: per-heap live usage, that cap
                        budget->heapBudget[i] = heap_size > g_budget_cap ? g_budget_cap : heap_size;
                        budget->heapUsage[i] = atomic_load(&g_heap_usage[i]);
                    } else { // old report, unchanged
                        budget->heapBudget[i] = heap_size > (6ULL << 30) ? (6ULL << 30) : heap_size;
                        budget->heapUsage[i] = allocated;
                    }
                } else {
                    budget->heapBudget[i] = 0;
                    budget->heapUsage[i] = 0;
                }
            }
        }
        header = header->pNext;
    }
}

static VKAPI_ATTR VkResult VKAPI_CALL compat_EnumerateDeviceExtensionProperties(
    VkPhysicalDevice physicalDevice, const char *pLayerName, uint32_t *pPropertyCount,
    VkExtensionProperties *pProperties) {
    if (pLayerName && pLayerName[0] != '\0') {
        if (!strcmp(pLayerName, LAYER_NAME)) {
            if (pPropertyCount) *pPropertyCount = 0;
            return VK_SUCCESS;
        }
        struct InstanceData *inst = get_instance_by_phys_dev(physicalDevice);
        if (!inst || !inst->fpEnumerateDeviceExtensionProperties) return VK_ERROR_LAYER_NOT_PRESENT;
        return inst->fpEnumerateDeviceExtensionProperties(physicalDevice, pLayerName, pPropertyCount, pProperties);
    }

    struct InstanceData *inst = get_instance_by_phys_dev(physicalDevice);
    if (!inst || !inst->fpEnumerateDeviceExtensionProperties) return VK_ERROR_INITIALIZATION_FAILED;
    uint32_t count = 0;
    VkResult res = inst->fpEnumerateDeviceExtensionProperties(physicalDevice, NULL, &count, NULL);
    if (res != VK_SUCCESS) return res;

    VkExtensionProperties *all = (VkExtensionProperties *)malloc((count + 4) * sizeof(VkExtensionProperties));
    if (!all) return VK_ERROR_OUT_OF_HOST_MEMORY;

    res = inst->fpEnumerateDeviceExtensionProperties(physicalDevice, NULL, &count, all);
    if (res != VK_SUCCESS && res != VK_INCOMPLETE) { free(all); return res; }

    bool has_budget = false;
    for (uint32_t i = 0; i < count; i++) {
        if (!strcmp(all[i].extensionName, VK_EXT_MEMORY_BUDGET_EXTENSION_NAME)) {
            has_budget = true;
            break;
        }
    }
    if (!has_budget) {
        strncpy(all[count].extensionName, VK_EXT_MEMORY_BUDGET_EXTENSION_NAME, VK_MAX_EXTENSION_NAME_SIZE);
        all[count].specVersion = 1;
        count++;
    }
    strncpy(all[count].extensionName, VK_EXT_MAP_MEMORY_PLACED_EXTENSION_NAME, VK_MAX_EXTENSION_NAME_SIZE);
    all[count].specVersion = 1;
    count++;
    // VK_EXT_vertex_attribute_divisor on top of the vendor's KHR version (vkd3d-proton asks for the EXT name: HZD
    // failed with "Lacking support for VK_EXT_vertex_attribute_divisor"). Same feature and pipeline structs; only the
    // properties struct differs. [Claude Opus 5.5]
    bool has_div_khr = false, has_div_ext = false;
    for (uint32_t i = 0; i < count; i++) {
        has_div_khr |= !strcmp(all[i].extensionName, "VK_KHR_vertex_attribute_divisor");
        has_div_ext |= !strcmp(all[i].extensionName, "VK_EXT_vertex_attribute_divisor");
    }
    if (has_div_khr && !has_div_ext) {
        strncpy(all[count].extensionName, "VK_EXT_vertex_attribute_divisor", VK_MAX_EXTENSION_NAME_SIZE);
        all[count].specVersion = 3;
        count++;
    }
    bool has_drm = false;
    for (uint32_t i = 0; i < count; i++) has_drm |= !strcmp(all[i].extensionName, VK_EXT_PHYSICAL_DEVICE_DRM_EXTENSION_NAME);
    if (g_drm_on && !has_drm) {
        strncpy(all[count].extensionName, VK_EXT_PHYSICAL_DEVICE_DRM_EXTENSION_NAME, VK_MAX_EXTENSION_NAME_SIZE);
        all[count].specVersion = 1;
        count++;
    }

    if (!pProperties) {
        *pPropertyCount = count;
        free(all);
        return VK_SUCCESS;
    }

    uint32_t copy_cnt = *pPropertyCount < count ? *pPropertyCount : count;
    memcpy(pProperties, all, copy_cnt * sizeof(VkExtensionProperties));
    free(all);

    if (copy_cnt < count) {
        *pPropertyCount = copy_cnt;
        return VK_INCOMPLETE;
    }
    *pPropertyCount = copy_cnt;
    return VK_SUCCESS;
}

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

// -----------------------------------------------------------------------------
// Device Level Interceptions
// -----------------------------------------------------------------------------
static VKAPI_ATTR VkResult VKAPI_CALL compat_CreateShaderModule(
    VkDevice device, const VkShaderModuleCreateInfo *pCreateInfo,
    const VkAllocationCallbacks *pAllocator, VkShaderModule *pShaderModule) {
    struct DeviceData *d = get_device_data(device);
    uint32_t hash = 0, mflags = 0, cnt[4];
    if (g_gpu_time || g_dump_hash_count || g_vs_scan || g_dc_on || g_ct) {
        hash = spv_hash(pCreateInfo->pCode, pCreateInfo->codeSize);
        spv_dump_if_wanted(pCreateInfo->pCode, pCreateInfo->codeSize, hash);
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
    if (r == VK_SUCCESS && (g_gpu_time || g_dc_on || g_ct) && pShaderModule) hash_put(g_module_hash, (uint64_t)*pShaderModule, &hash, NULL, 1, mflags);
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
    for (uint32_t i = 0; g_wls_clamp && i < createInfoCount && i < 8; i++) {
        VkShaderModuleCreateInfo *sm = (VkShaderModuleCreateInfo *)find_struct(pCreateInfos[i].stage.pNext, VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO);
        size_t ob = 0;
        if (sm && sm->pCode && (wls_new[i] = wls_clamp_module(sm->pCode, sm->codeSize, sm->pCode, sm->codeSize, &ob))) {
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
        if ((g_gpu_time || g_ct) && res == VK_SUCCESS && pPipelines && pPipelines[i])
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

// Debug aid: with /root/mali-images.txt present, one line per image: format, size, usage, flags, and the driver's
// answer on compression (VkImageCompressionPropertiesEXT: DEFAULT = AFBC on Mali, DISABLED = uncompressed).
// [Claude Opus 5.5] 2026-10-26
static void log_image(struct DeviceData *d, VkDevice device, const VkImageCreateInfo *ci, VkImage img, bool bc) {
    static pthread_mutex_t m = PTHREAD_MUTEX_INITIALIZER;
    VkImageCompressionPropertiesEXT cp = {VK_STRUCTURE_TYPE_IMAGE_COMPRESSION_PROPERTIES_EXT};
    if (d->fpGetImageSubresourceLayout2KHR) {
        VkImageAspectFlags aspect = VK_IMAGE_ASPECT_COLOR_BIT;
        if (ci->format >= VK_FORMAT_D16_UNORM && ci->format <= VK_FORMAT_D32_SFLOAT_S8_UINT)
            aspect = ci->format == VK_FORMAT_S8_UINT ? VK_IMAGE_ASPECT_STENCIL_BIT : VK_IMAGE_ASPECT_DEPTH_BIT;
        VkImageSubresource2KHR sub = {VK_STRUCTURE_TYPE_IMAGE_SUBRESOURCE_2_KHR, NULL, {aspect, 0, 0}};
        VkSubresourceLayout2KHR lay = {VK_STRUCTURE_TYPE_SUBRESOURCE_LAYOUT_2_KHR, &cp};
        if (ci->tiling != VK_IMAGE_TILING_DRM_FORMAT_MODIFIER_EXT) d->fpGetImageSubresourceLayout2KHR(device, img, &sub, &lay);
    }
    pthread_mutex_lock(&m);
    FILE *f = fopen("/root/mali-images.txt", "a");
    if (f) {
        const VkImageCompressionControlEXT *acc = find_struct(ci->pNext, VK_STRUCTURE_TYPE_IMAGE_COMPRESSION_CONTROL_EXT);
        fprintf(f, "fmt %d %ux%ux%u mips %u layers %u samples %d tiling %d usage 0x%x flags 0x%x%s comp 0x%x rate 0x%x appcc %s\n",
                ci->format, ci->extent.width, ci->extent.height, ci->extent.depth, ci->mipLevels, ci->arrayLayers,
                ci->samples, ci->tiling, ci->usage, ci->flags, bc ? " bc-shadow" : "", cp.imageCompressionFlags,
                cp.imageCompressionFixedRateFlags, acc ? (acc->flags == VK_IMAGE_COMPRESSION_DISABLED_EXT ? "disabled" : "other") : "-");
        fclose(f);
    }
    pthread_mutex_unlock(&m);
}

// -----------------------------------------------------------------------------
// Render-target compression, MALI_COMPAT_RT_COMPRESS=1. [Claude Opus 5.5] 2026-10-28
// The vendor driver turns AFBC off for every image with STORAGE usage, for all colour formats, and offers no
// fixed-rate compression for them either (test/rt-comp-probe.c). DXVK adds STORAGE to each render target the game
// creates with a UAV bind flag, so all of The Witcher 3's 1600x720 colour targets ran uncompressed (image log
// 2026-10-28: 106 targets, comp 0x4). With the switch on, colour targets are created without STORAGE. A storage view
// on such an image means the game does use it as a UAV: its signature (format, size, mips, layers) is appended to
// /root/mali-rt-keep-<comm>.txt, and from the next launch on, images with that signature keep STORAGE. In the launch
// that learns a signature, that UAV use is invalid (expect glitches there), which is why it is logged.
// -----------------------------------------------------------------------------
static bool g_rt_compress;
// MALI_COMPAT_DEPTH_NOCOMP=1: depth/stencil images that shaders read are created uncompressed (no AFBC). The current
// Witcher 3 (DX12) hangs the GPU in vkd3d's depth->colour copy pass, which texelFetches the depth aspect of its
// D32_SFLOAT_S8_UINT depth buffer (GPU timing STUCK lines, 2026-10-28); test whether compression is the trigger.
static bool g_depth_nocomp;
// MALI_COMPAT_FORCE_COMP=1: vkd3d-proton chains VkImageCompressionControlEXT DISABLED into placed (aliasable)
// resources, so DX12 games run all their render targets and shadow maps uncompressed on this driver. For colour and
// depth/stencil attachments without STORAGE the layer turns that request back into DEFAULT (AFBC). Experiment: an
// aliased target then shares memory with compression metadata, which vkd3d does not expect. [Claude Opus 5.5] 2026-10-28
static bool g_force_comp;
struct RTSig { VkFormat format; uint32_t width, height, mips, layers; };
#define RT_KEEP_MAX 256
static struct RTSig g_rt_keep[RT_KEEP_MAX];
static uint32_t g_rt_keep_count;
static char g_rt_keep_path[96];
struct RTNode { VkImage image; struct RTSig sig; VkImageUsageFlags usage; struct RTNode *next; };
#define RT_BUCKETS 1024
static struct RTNode *g_rt_images[RT_BUCKETS];
static pthread_mutex_t g_rt_mutex = PTHREAD_MUTEX_INITIALIZER;
static _Atomic uint32_t g_rt_stripped, g_rt_kept;

static struct RTSig rt_sig(const VkImageCreateInfo *ci) {
    return (struct RTSig){ci->format, ci->extent.width, ci->extent.height, ci->mipLevels, ci->arrayLayers};
}

static bool rt_keep_has_locked(struct RTSig s) {
    for (uint32_t i = 0; i < g_rt_keep_count; i++)
        if (!memcmp(&g_rt_keep[i], &s, sizeof s)) return true;
    return false;
}

static void rt_keep_load(void) {
    char comm[32] = "unknown";
    FILE *f = fopen("/proc/self/comm", "r");
    if (f) {
        if (fgets(comm, sizeof comm, f)) comm[strcspn(comm, "\n")] = 0;
        fclose(f);
    }
    for (char *c = comm; *c; c++)
        if (*c == '/' || *c == ' ' || *c == '\\') *c = '_';
    snprintf(g_rt_keep_path, sizeof g_rt_keep_path, "/root/mali-rt-keep-%s.txt", comm);
    pthread_mutex_lock(&g_rt_mutex);
    g_rt_keep_count = 0;
    if ((f = fopen(g_rt_keep_path, "r"))) {
        struct RTSig s;
        int fmt;
        while (g_rt_keep_count < RT_KEEP_MAX && fscanf(f, "%d %u %u %u %u", &fmt, &s.width, &s.height, &s.mips, &s.layers) == 5) {
            s.format = (VkFormat)fmt;
            g_rt_keep[g_rt_keep_count++] = s;
        }
        fclose(f);
    }
    pthread_mutex_unlock(&g_rt_mutex);
    fprintf(stderr, "mali-compat: rt-compress on, %u UAV target signatures kept from %s\n", g_rt_keep_count, g_rt_keep_path);
}

// A UAV view on a stripped target: remember its signature for the next launches.
static void rt_learn(struct RTSig s) {
    pthread_mutex_lock(&g_rt_mutex);
    if (!rt_keep_has_locked(s) && g_rt_keep_count < RT_KEEP_MAX) {
        g_rt_keep[g_rt_keep_count++] = s;
        FILE *f = fopen(g_rt_keep_path, "a");
        if (f) {
            fprintf(f, "%d %u %u %u %u\n", s.format, s.width, s.height, s.mips, s.layers);
            fclose(f);
        }
        fprintf(stderr, "mali-compat: rt-compress: UAV view on compressed target fmt %d %ux%u mips %u layers %u; "
                "learned, keeps STORAGE from the next launch\n", s.format, s.width, s.height, s.mips, s.layers);
    }
    pthread_mutex_unlock(&g_rt_mutex);
}

static void rt_track(VkImage img, struct RTSig s, VkImageUsageFlags usage) {
    struct RTNode *e = calloc(1, sizeof *e);
    if (!e) return;
    e->image = img;
    e->sig = s;
    e->usage = usage;
    uint32_t b = handle_bucket((uint64_t)(uintptr_t)img, RT_BUCKETS);
    pthread_mutex_lock(&g_rt_mutex);
    e->next = g_rt_images[b];
    g_rt_images[b] = e;
    pthread_mutex_unlock(&g_rt_mutex);
}

static bool rt_find(VkImage img, struct RTNode *out) {
    bool found = false;
    pthread_mutex_lock(&g_rt_mutex);
    for (struct RTNode *e = g_rt_images[handle_bucket((uint64_t)(uintptr_t)img, RT_BUCKETS)]; e; e = e->next)
        if (e->image == img) { *out = *e; found = true; break; }
    pthread_mutex_unlock(&g_rt_mutex);
    return found;
}

static void rt_untrack(VkImage img) {
    pthread_mutex_lock(&g_rt_mutex);
    for (struct RTNode **pp = &g_rt_images[handle_bucket((uint64_t)(uintptr_t)img, RT_BUCKETS)]; *pp; pp = &(*pp)->next)
        if ((*pp)->image == img) {
            struct RTNode *e = *pp;
            *pp = e->next;
            free(e);
            break;
        }
    pthread_mutex_unlock(&g_rt_mutex);
}

// The create info the driver gets for an app image: BC formats become their RGBA8/RG16/RGBA16F shadow (format list
// mapped, fixed-rate compression when offered) and Wine's HOST_ALLOCATION external memory is unlinked. Used by
// vkCreateImage and by the vkGetDeviceImage*Requirements queries, which must answer for the same image: vkd3d-proton
// sizes its placed D3D12 resources with them before creating the image, and answering for the BC format (which the
// vendor does not even support) left HZD's textures in too-small heap slots, so every surface sampled black.
// fixup_end restores the app's chain. [Claude Opus 5.5] 2026-10-26
struct image_fixup {
    VkImageCreateInfo mod;
    const VkImageCreateInfo *use;
    bool is_bc;
    VkImageCompressionFixedRateFlagsEXT rate;
    VkImageCompressionControlEXT afrc;
    VkImageFormatListCreateInfo *fl;
    const VkFormat *fl_orig;
    VkFormat *fl_mapped;
    VkImageFormatListCreateInfo fl_synth; // for MUTABLE shadows created without a list
    VkFormat fl_synth_formats[2];
    VkImageCompressionControlEXT *app_cc; // the app's own compression control, overridden for the call
    VkImageCompressionControlEXT app_cc_saved;
    VkBaseOutStructure *host_prev, *host_ext;
    bool rt_stripped; // colour target created without the app's STORAGE (MALI_COMPAT_RT_COMPRESS)
};

static void fixup_begin(struct DeviceData *d, const VkImageCreateInfo *ci, struct image_fixup *fx) {
    memset(fx, 0, sizeof *fx);
    fx->mod = *ci;
    fx->use = ci;
    fx->is_bc = panvk_format_is_bc(ci->format);
    if (fx->is_bc) {
        fx->use = &fx->mod;
        fx->mod.format = panvk_bc_decoded_format(ci->format);
        fx->mod.flags &= ~VK_IMAGE_CREATE_BLOCK_TEXEL_VIEW_COMPATIBLE_BIT;
        fx->fl = (VkImageFormatListCreateInfo *)find_struct(fx->mod.pNext, VK_STRUCTURE_TYPE_IMAGE_FORMAT_LIST_CREATE_INFO);
        if (fx->fl && fx->fl->viewFormatCount && fx->fl->pViewFormats &&
            (fx->fl_mapped = malloc(fx->fl->viewFormatCount * sizeof(VkFormat)))) {
            for (uint32_t v = 0; v < fx->fl->viewFormatCount; v++)
                fx->fl_mapped[v] = panvk_format_is_bc(fx->fl->pViewFormats[v]) ? panvk_bc_decoded_format(fx->fl->pViewFormats[v])
                                                                                : fx->fl->pViewFormats[v];
            fx->fl_orig = fx->fl->pViewFormats;
            fx->fl->pViewFormats = fx->fl_mapped;
        }
        // A MUTABLE image without a format list gets no compression from the vendor driver (afrc-probe: RGBA8_SRGB
        // MUTABLE without a list 5465 KiB uncompressed, with a UNORM/SRGB list 1370 KiB at 2 bpc; ALIAS does not
        // matter). HZD's 995 BC7 shadows were created that way. The only views an RGBA8 shadow can serve are its
        // UNORM and SRGB variants (BLOCK_TEXEL_VIEW_COMPATIBLE is stripped above), so give it that list.
        // [Claude Opus 5.5] 2026-10-26
        if (!fx->fl && (fx->mod.flags & VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT) &&
            (fx->mod.format == VK_FORMAT_R8G8B8A8_UNORM || fx->mod.format == VK_FORMAT_R8G8B8A8_SRGB)) {
            fx->fl_synth_formats[0] = VK_FORMAT_R8G8B8A8_UNORM;
            fx->fl_synth_formats[1] = VK_FORMAT_R8G8B8A8_SRGB;
            fx->fl_synth = (VkImageFormatListCreateInfo){VK_STRUCTURE_TYPE_IMAGE_FORMAT_LIST_CREATE_INFO, fx->mod.pNext, 2,
                                                         fx->fl_synth_formats};
            fx->mod.pNext = &fx->fl_synth;
            fx->fl = &fx->fl_synth;
        }
        // Fixed-rate compression for the BC shadows: an RGBA8 shadow is 4-8x the BC data (Tomb Raider: ~2.7 GB of
        // shadows instead of ~0.5 GB). AFRC at 2 bpc (8 bpp, BC3's size) for BC1's 4-colour blocks, 3 bpc for the
        // others; only when the driver offers that rate for this format, usage and flags (test/afrc-probe.c).
        // vkd3d-proton chains its own VkImageCompressionControlEXT (it disables compression on placed/aliasable
        // resources): HZD's 3771 images, all BC shadows included, came out comp 0x4 (DISABLED), ~2.6 GB uncompressed.
        // A shadow's contents only ever come from the layer's decode copy, so its compression is the layer's call:
        // the app's struct is overridden for the call and restored in fixup_end. [Claude Opus 5.5] 2026-10-26
        VkImageCompressionControlEXT *app_cc = (VkImageCompressionControlEXT *)find_struct(fx->mod.pNext,
            VK_STRUCTURE_TYPE_IMAGE_COMPRESSION_CONTROL_EXT);
        if (d->afrc && (!app_cc || app_cc->flags != VK_IMAGE_COMPRESSION_FIXED_RATE_EXPLICIT_EXT)) {
            VkImageCompressionControlEXT q = {VK_STRUCTURE_TYPE_IMAGE_COMPRESSION_CONTROL_EXT, NULL,
                                              VK_IMAGE_COMPRESSION_FIXED_RATE_DEFAULT_EXT};
            VkImageFormatListCreateInfo fl_copy;
            if (fx->fl) { fl_copy = *fx->fl; fl_copy.pNext = NULL; q.pNext = &fl_copy; }
            VkPhysicalDeviceImageFormatInfo2 qi = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_IMAGE_FORMAT_INFO_2, &q, fx->mod.format,
                                                   fx->mod.imageType, fx->mod.tiling, fx->mod.usage, fx->mod.flags};
            VkImageCompressionPropertiesEXT cp = {VK_STRUCTURE_TYPE_IMAGE_COMPRESSION_PROPERTIES_EXT};
            VkImageFormatProperties2 qp = {VK_STRUCTURE_TYPE_IMAGE_FORMAT_PROPERTIES_2, &cp};
            if (d->instance_data && d->instance_data->fpGetPhysicalDeviceImageFormatProperties2 &&
                d->instance_data->fpGetPhysicalDeviceImageFormatProperties2(d->physical_device, &qi, &qp) == VK_SUCCESS) {
                bool bc1 = ci->format >= VK_FORMAT_BC1_RGB_UNORM_BLOCK && ci->format <= VK_FORMAT_BC1_RGBA_SRGB_BLOCK;
                VkImageCompressionFixedRateFlagsEXT want = VK_IMAGE_COMPRESSION_FIXED_RATE_NONE_EXT;
                if (cp.imageCompressionFixedRateFlags & VK_IMAGE_COMPRESSION_FIXED_RATE_2BPC_BIT_EXT)
                    want = VK_IMAGE_COMPRESSION_FIXED_RATE_2BPC_BIT_EXT;
                else if (!bc1 && (cp.imageCompressionFixedRateFlags & VK_IMAGE_COMPRESSION_FIXED_RATE_3BPC_BIT_EXT))
                    want = VK_IMAGE_COMPRESSION_FIXED_RATE_3BPC_BIT_EXT;
                if (want != VK_IMAGE_COMPRESSION_FIXED_RATE_NONE_EXT) {
                    fx->rate = want;
                    if (app_cc) {
                        fx->app_cc = app_cc;
                        fx->app_cc_saved = *app_cc;
                        app_cc->flags = VK_IMAGE_COMPRESSION_FIXED_RATE_EXPLICIT_EXT;
                        app_cc->compressionControlPlaneCount = 1;
                        app_cc->pFixedRateFlags = &fx->rate;
                    } else {
                        fx->afrc = (VkImageCompressionControlEXT){VK_STRUCTURE_TYPE_IMAGE_COMPRESSION_CONTROL_EXT, (void *)fx->mod.pNext,
                                                                  VK_IMAGE_COMPRESSION_FIXED_RATE_EXPLICIT_EXT, 1, &fx->rate};
                        fx->mod.pNext = &fx->afrc;
                    }
                }
            }
        }
    }
    if (g_rt_compress && !fx->is_bc && (ci->usage & VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT) &&
        (ci->usage & VK_IMAGE_USAGE_STORAGE_BIT)) {
        pthread_mutex_lock(&g_rt_mutex);
        bool keep = rt_keep_has_locked(rt_sig(ci));
        pthread_mutex_unlock(&g_rt_mutex);
        if (!keep) {
            fx->use = &fx->mod;
            fx->mod.usage &= ~VK_IMAGE_USAGE_STORAGE_BIT;
            fx->rt_stripped = true;
        }
    }
    if (g_force_comp && !fx->is_bc && !g_depth_nocomp &&
        (fx->mod.usage & (VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT)) &&
        !(fx->mod.usage & VK_IMAGE_USAGE_STORAGE_BIT)) {
        VkImageCompressionControlEXT *app_cc = (VkImageCompressionControlEXT *)find_struct(fx->mod.pNext,
            VK_STRUCTURE_TYPE_IMAGE_COMPRESSION_CONTROL_EXT);
        if (app_cc && app_cc->flags == VK_IMAGE_COMPRESSION_DISABLED_EXT) {
            fx->use = &fx->mod;
            fx->app_cc = app_cc;
            fx->app_cc_saved = *app_cc;
            app_cc->flags = VK_IMAGE_COMPRESSION_DEFAULT_EXT;
        }
    }
    if (g_depth_nocomp && d->afrc && !fx->is_bc && (ci->usage & VK_IMAGE_USAGE_SAMPLED_BIT) &&
        ((ci->format >= VK_FORMAT_D16_UNORM && ci->format <= VK_FORMAT_D32_SFLOAT_S8_UINT))) {
        VkImageCompressionControlEXT *app_cc = (VkImageCompressionControlEXT *)find_struct(fx->mod.pNext,
            VK_STRUCTURE_TYPE_IMAGE_COMPRESSION_CONTROL_EXT);
        fx->use = &fx->mod;
        if (app_cc) {
            fx->app_cc = app_cc;
            fx->app_cc_saved = *app_cc;
            app_cc->flags = VK_IMAGE_COMPRESSION_DISABLED_EXT;
            app_cc->compressionControlPlaneCount = 0;
            app_cc->pFixedRateFlags = NULL;
        } else {
            fx->afrc = (VkImageCompressionControlEXT){VK_STRUCTURE_TYPE_IMAGE_COMPRESSION_CONTROL_EXT, (void *)fx->mod.pNext,
                                                      VK_IMAGE_COMPRESSION_DISABLED_EXT, 0, NULL};
            fx->mod.pNext = &fx->afrc;
        }
    }
    // Wine on WoW64 chains HOST_ALLOCATION external memory into every image; the vendor refuses such images
    // (-1000158000) but binds a plain image to host-pointer memory fine (test/img-probe.c): unlink it for the call.
    for (VkBaseOutStructure *prev = (VkBaseOutStructure *)fx->use, *st = prev->pNext; st; prev = st, st = st->pNext) {
        if (st->sType == VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO &&
            ((VkExternalMemoryImageCreateInfo *)st)->handleTypes == VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT) {
            if (fx->use == ci) { fx->use = &fx->mod; prev = prev == (VkBaseOutStructure *)ci ? (VkBaseOutStructure *)&fx->mod : prev; }
            fx->host_prev = prev;
            fx->host_ext = st;
            prev->pNext = st->pNext;
            break;
        }
    }
}

static void fixup_end(struct image_fixup *fx) {
    if (fx->app_cc) *fx->app_cc = fx->app_cc_saved;
    if (fx->host_prev) fx->host_prev->pNext = fx->host_ext;
    if (fx->fl_orig) fx->fl->pViewFormats = fx->fl_orig;
    free(fx->fl_mapped);
}

static VKAPI_ATTR void VKAPI_CALL compat_GetDeviceImageMemoryRequirements(
    VkDevice device, const VkDeviceImageMemoryRequirements *pInfo, VkMemoryRequirements2 *pMemoryRequirements) {
    struct DeviceData *d = get_device_data(device);
    struct image_fixup fx;
    fixup_begin(d, pInfo->pCreateInfo, &fx);
    VkDeviceImageMemoryRequirements info = *pInfo;
    info.pCreateInfo = fx.use;
    d->fpGetDeviceImageMemoryRequirements(device, &info, pMemoryRequirements);
    fixup_end(&fx);
}

static VKAPI_ATTR void VKAPI_CALL compat_GetDeviceImageSparseMemoryRequirements(
    VkDevice device, const VkDeviceImageMemoryRequirements *pInfo, uint32_t *pCount,
    VkSparseImageMemoryRequirements2 *pReqs) {
    struct DeviceData *d = get_device_data(device);
    struct image_fixup fx;
    fixup_begin(d, pInfo->pCreateInfo, &fx);
    VkDeviceImageMemoryRequirements info = *pInfo;
    info.pCreateInfo = fx.use;
    d->fpGetDeviceImageSparseMemoryRequirements(device, &info, pCount, pReqs);
    fixup_end(&fx);
}

static VKAPI_ATTR VkResult VKAPI_CALL compat_CreateImage(
    VkDevice device, const VkImageCreateInfo *pCreateInfo,
    const VkAllocationCallbacks *pAllocator, VkImage *pImage) {
    struct DeviceData *d = get_device_data(device);
    // dma-buf export (MALI_COMPAT_DMABUF_EXPORT): a modifier list becomes an explicit linear layout the driver imports
    VkImageCreateInfo dci;
    VkImageDrmFormatModifierExplicitCreateInfoEXT expl;
    VkSubresourceLayout plane0;
    struct unlinked u_list = {0}, u_wsi = {0};
    if (g_dmabuf_export) {
        u_wsi = unlink_struct((void *)pCreateInfo, STYPE_WSI_IMAGE_CREATE_INFO_MESA);
        const VkExternalMemoryImageCreateInfo *em = find_struct(pCreateInfo->pNext, VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO);
        const VkImageDrmFormatModifierListCreateInfoEXT *ml =
            find_struct(pCreateInfo->pNext, VK_STRUCTURE_TYPE_IMAGE_DRM_FORMAT_MODIFIER_LIST_CREATE_INFO_EXT);
        uint32_t bpp = format_bytes(pCreateInfo->format);
        bool has_linear = false;
        for (uint32_t i = 0; ml && i < ml->drmFormatModifierCount; i++) has_linear |= ml->pDrmFormatModifiers[i] == 0;
        if (em && getenv("MALI_COMPAT_DMABUF_PEEK")) {
            // diagnostic: how the external images of this process are made (Xwayland/Zink vs gamescope)
            static _Atomic int logged;
            const VkImageDrmFormatModifierExplicitCreateInfoEXT *ex0 =
                find_struct(pCreateInfo->pNext, VK_STRUCTURE_TYPE_IMAGE_DRM_FORMAT_MODIFIER_EXPLICIT_CREATE_INFO_EXT);
            if (atomic_fetch_add(&logged, 1) < 24)
                fprintf(stderr, "mali-compat: external image: handles 0x%x fmt %d %ux%u tiling %d usage 0x%x flags 0x%x, "
                        "modifier list %u (linear %d), explicit %s %llx pitch %llu\n", em->handleTypes, pCreateInfo->format,
                        pCreateInfo->extent.width, pCreateInfo->extent.height, pCreateInfo->tiling, pCreateInfo->usage,
                        pCreateInfo->flags, ml ? ml->drmFormatModifierCount : 0, has_linear, ex0 ? "yes" : "no",
                        ex0 ? (unsigned long long)ex0->drmFormatModifier : 0ull,
                        ex0 && ex0->pPlaneLayouts ? (unsigned long long)ex0->pPlaneLayouts[0].rowPitch : 0ull);
        }
        if (em && (em->handleTypes & VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT) && ml && has_linear && bpp &&
            pCreateInfo->tiling == VK_IMAGE_TILING_DRM_FORMAT_MODIFIER_EXT) {
            u_list = unlink_struct((void *)pCreateInfo, VK_STRUCTURE_TYPE_IMAGE_DRM_FORMAT_MODIFIER_LIST_CREATE_INFO_EXT);
            plane0 = (VkSubresourceLayout){0, 0, ((VkDeviceSize)pCreateInfo->extent.width * bpp + 255) & ~255ull, 0, 0};
            expl = (VkImageDrmFormatModifierExplicitCreateInfoEXT){
                VK_STRUCTURE_TYPE_IMAGE_DRM_FORMAT_MODIFIER_EXPLICIT_CREATE_INFO_EXT, pCreateInfo->pNext, 0, 1, &plane0};
            dci = *pCreateInfo;
            dci.pNext = &expl;
            pCreateInfo = &dci;
        }
    }
    struct image_fixup fx;
    fixup_begin(d, pCreateInfo, &fx);
    bool is_bc = fx.is_bc;
    VkResult res = d->fpCreateImage(device, fx.use, pAllocator, pImage);
    if (res != VK_SUCCESS && is_yuv_nv12(pCreateInfo->format)) {
        VkImageCreateInfo fallback_ci = *pCreateInfo;
        fallback_ci.format = VK_FORMAT_R8_UNORM;
        fallback_ci.extent.height = (pCreateInfo->extent.height * 3) / 2;
        fallback_ci.flags &= ~(VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT | VK_IMAGE_CREATE_EXTENDED_USAGE_BIT);
        res = d->fpCreateImage(device, &fallback_ci, pAllocator, pImage);
        if (res == VK_SUCCESS) {
            fprintf(stderr, "mali-compat: NV12 video image fallback to R8 (%ux%u) succeeded\n",
                    fallback_ci.extent.width, fallback_ci.extent.height);
        }
    }
    if (res != VK_SUCCESS) {
        // [Claude Opus 5.5] always logged: which create info the driver refused (DXVK only prints size/format).
        const VkImageCreateInfo *ci = fx.use;
        fprintf(stderr, "mali-compat: vkCreateImage -> %d: flags 0x%x fmt %d %ux%ux%u mips %u layers %u samples %d "
                "tiling %d usage 0x%x sharing %d layout %d\n", res, ci->flags, ci->format, ci->extent.width,
                ci->extent.height, ci->extent.depth, ci->mipLevels, ci->arrayLayers, ci->samples, ci->tiling,
                ci->usage, ci->sharingMode, ci->initialLayout);
        for (const VkBaseInStructure *s = ci->pNext; s; s = s->pNext) {
            fprintf(stderr, "mali-compat:   pNext sType %d\n", s->sType);
            if (s->sType == VK_STRUCTURE_TYPE_IMAGE_FORMAT_LIST_CREATE_INFO) {
                const VkImageFormatListCreateInfo *fl = (const VkImageFormatListCreateInfo *)s;
                for (uint32_t v = 0; v < fl->viewFormatCount; v++)
                    fprintf(stderr, "mali-compat:     view format %d\n", fl->pViewFormats[v]);
            } else if (s->sType == VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO) {
                fprintf(stderr, "mali-compat:     handle types 0x%x\n", ((const VkExternalMemoryImageCreateInfo *)s)->handleTypes);
            }
        }
    }
    if (res == VK_SUCCESS && g_log_images && pImage && *pImage) log_image(d, device, fx.use, *pImage, is_bc);
    if (res == VK_SUCCESS && (g_gpu_time || g_ct) && pImage && *pImage) // usage as the driver got it (RT_COMPRESS may strip STORAGE)
        img_put(g_img_info, (struct ImgNode){(uint64_t)*pImage, *pImage, pCreateInfo->format, pCreateInfo->extent.width,
                pCreateInfo->extent.height, 0, pCreateInfo->mipLevels, 0, pCreateInfo->arrayLayers, pCreateInfo->samples,
                fx.use->usage, NULL});
    if (res == VK_SUCCESS && g_rt_compress && pImage && *pImage && (pCreateInfo->usage & VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT) &&
        (pCreateInfo->usage & VK_IMAGE_USAGE_STORAGE_BIT) && !is_bc) {
        if (fx.rt_stripped) {
            rt_track(*pImage, rt_sig(pCreateInfo), fx.mod.usage);
            atomic_fetch_add(&g_rt_stripped, 1);
        } else {
            atomic_fetch_add(&g_rt_kept, 1);
        }
    }
    if (res == VK_SUCCESS && is_bc && pImage && *pImage) {
        register_bc_image(*pImage, pCreateInfo->format, fx.mod.format, pCreateInfo->extent);
        LOG("emulating BC image %p: orig=%d -> decoded=%d (%ux%u)",
            (void *)*pImage, pCreateInfo->format, fx.mod.format, pCreateInfo->extent.width, pCreateInfo->extent.height);
    }
    fixup_end(&fx);
    relink(u_list);
    relink(u_wsi);
    return res;
}

static VKAPI_ATTR void VKAPI_CALL compat_DestroyImage(
    VkDevice device, VkImage image, const VkAllocationCallbacks *pAllocator) {
    struct DeviceData *d = get_device_data(device);
    unregister_bc_image(image);
    if (g_rt_compress) rt_untrack(image);
    d->fpDestroyImage(device, image, pAllocator);
}

static VkResult create_image_view(
    VkDevice device, const VkImageViewCreateInfo *pCreateInfo,
    const VkAllocationCallbacks *pAllocator, VkImageView *pView) {
    struct DeviceData *d = get_device_data(device);

    struct TrackedBCImage tracked;
    if (find_bc_image(pCreateInfo->image, &tracked) && panvk_format_is_bc(pCreateInfo->format)) {
        VkImageViewCreateInfo mod = *pCreateInfo;
        mod.format = panvk_bc_decoded_format(pCreateInfo->format);
        return d->fpCreateImageView(device, &mod, pAllocator, pView);
    }

    // A storage view on a target created without STORAGE: learn it, and keep the view itself valid (DXVK chains the
    // view's own usage; a view usage the image lacks is invalid) by asking for the image's other usages instead.
    struct RTNode rt;
    VkImageViewUsageCreateInfo *vu;
    if (g_rt_compress && rt_find(pCreateInfo->image, &rt) &&
        (vu = (VkImageViewUsageCreateInfo *)find_struct(pCreateInfo->pNext, VK_STRUCTURE_TYPE_IMAGE_VIEW_USAGE_CREATE_INFO)) &&
        (vu->usage & VK_IMAGE_USAGE_STORAGE_BIT)) {
        rt_learn(rt.sig);
        VkImageUsageFlags saved = vu->usage;
        vu->usage = (saved & ~VK_IMAGE_USAGE_STORAGE_BIT) & rt.usage;
        if (!vu->usage) vu->usage = rt.usage & (VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT);
        VkResult res = d->fpCreateImageView(device, pCreateInfo, pAllocator, pView);
        vu->usage = saved;
        return res;
    }

    return d->fpCreateImageView(device, pCreateInfo, pAllocator, pView);
}

// GPU timing: remembers what each view shows, for the STUCK/TRACE details.
static VKAPI_ATTR VkResult VKAPI_CALL compat_CreateImageView(
    VkDevice device, const VkImageViewCreateInfo *pCreateInfo,
    const VkAllocationCallbacks *pAllocator, VkImageView *pView) {
    VkResult res = create_image_view(device, pCreateInfo, pAllocator, pView);
    if ((g_gpu_time || g_ct) && res == VK_SUCCESS && pView) {
        struct ImgNode img = {0};
        img_get(g_img_info, (uint64_t)pCreateInfo->image, &img);
        const VkImageSubresourceRange *r = &pCreateInfo->subresourceRange;
        img_put(g_view_info, (struct ImgNode){(uint64_t)*pView, pCreateInfo->image, pCreateInfo->format, img.w, img.h,
                r->baseMipLevel, r->levelCount, r->baseArrayLayer, r->layerCount, img.samples, r->aspectMask, NULL});
    }
    return res;
}

static VKAPI_ATTR VkResult VKAPI_CALL compat_CreateBufferView(
    VkDevice device, const VkBufferViewCreateInfo *pCreateInfo, const VkAllocationCallbacks *pAllocator, VkBufferView *pView) {
    struct DeviceData *d = get_device_data(device);
    static _Atomic int misaligned;
    if (pCreateInfo->offset % 64) {
        int n = atomic_fetch_add(&misaligned, 1);
        if (n < 8 || (n & (n - 1)) == 0) // the first 8, then powers of two
            fprintf(stderr, "mali-compat: buffer view %d with offset %llu (not 64-aligned), format %d, range %llu\n", n + 1,
                    (unsigned long long)pCreateInfo->offset, pCreateInfo->format, (unsigned long long)pCreateInfo->range);
    }
    VkResult r = d->fpCreateBufferView(device, pCreateInfo, pAllocator, pView);
    if (g_dc_on && r == VK_SUCCESS) dc_view_create(*pView, pCreateInfo);
    return r;
}

// MALI_COMPAT_FAKE_OCCLUSION=1: every occlusion query result reads as "visible" (0x10000 samples, available), for the
// GPU copy (vkCmdFillBuffer instead of vkCmdCopyQueryPoolResults) and the CPU read. A test for occlusion results the
// driver gets wrong (HZD statics missing): if the statics come back, the game culls them on query results.
// [Claude Opus 5.5] 2026-10-29
static bool g_fake_occlusion;
static bool g_force_coherent; // MALI_COMPAT_FORCE_COHERENT, see compat_AllocateMemory
static _Atomic uint64_t g_occl_copies, g_occl_reads;
#define OCCL_BUCKETS 64
static VkQueryPool g_occl_pools[OCCL_BUCKETS][16];
static pthread_mutex_t g_occl_mutex = PTHREAD_MUTEX_INITIALIZER;

static bool occl_pool(VkQueryPool pool, bool add, bool remove) {
    uint32_t b = handle_bucket((uint64_t)pool, OCCL_BUCKETS);
    bool found = false;
    pthread_mutex_lock(&g_occl_mutex);
    for (int i = 0; i < 16; i++) {
        if (g_occl_pools[b][i] == pool) { found = true; if (remove) g_occl_pools[b][i] = VK_NULL_HANDLE; break; }
        if (add && !g_occl_pools[b][i]) { g_occl_pools[b][i] = pool; add = false; }
    }
    pthread_mutex_unlock(&g_occl_mutex);
    if (add) fprintf(stderr, "mali-compat: fake occlusion: pool table full\n");
    return found;
}

static VKAPI_ATTR VkResult VKAPI_CALL compat_CreateQueryPool(VkDevice device, const VkQueryPoolCreateInfo *pCreateInfo,
                                                            const VkAllocationCallbacks *pAllocator, VkQueryPool *pPool) {
    struct DeviceData *d = get_device_data(device);
    VkResult r = d->fpCreateQueryPool(device, pCreateInfo, pAllocator, pPool);
    if (r == VK_SUCCESS && pCreateInfo->queryType == VK_QUERY_TYPE_OCCLUSION) occl_pool(*pPool, true, false);
    return r;
}

static VKAPI_ATTR void VKAPI_CALL compat_DestroyQueryPool(VkDevice device, VkQueryPool pool, const VkAllocationCallbacks *pAllocator) {
    struct DeviceData *d = get_device_data(device);
    if (pool) occl_pool(pool, false, true);
    d->fpDestroyQueryPool(device, pool, pAllocator);
}

static VKAPI_ATTR void VKAPI_CALL compat_CmdCopyQueryPoolResults(VkCommandBuffer cmd, VkQueryPool pool, uint32_t first,
    uint32_t count, VkBuffer dst, VkDeviceSize offset, VkDeviceSize stride, VkQueryResultFlags flags) {
    struct DeviceData *d = get_device_data_by_cmd(cmd);
    if (!occl_pool(pool, false, false) || !count) {
        d->fpCmdCopyQueryPoolResults(cmd, pool, first, count, dst, offset, stride, flags);
        return;
    }
    atomic_fetch_add(&g_occl_copies, 1);
    VkDeviceSize elem = (flags & VK_QUERY_RESULT_64_BIT ? 8 : 4) * (flags & VK_QUERY_RESULT_WITH_AVAILABILITY_BIT ? 2 : 1);
    if (stride == elem) d->fpCmdFillBuffer(cmd, dst, offset, elem * count, 0x10000);
    else
        for (uint32_t i = 0; i < count; i++) d->fpCmdFillBuffer(cmd, dst, offset + i * stride, elem, 0x10000);
}

static VKAPI_ATTR VkResult VKAPI_CALL compat_GetQueryPoolResults(VkDevice device, VkQueryPool pool, uint32_t first,
    uint32_t count, size_t size, void *data, VkDeviceSize stride, VkQueryResultFlags flags) {
    struct DeviceData *d = get_device_data(device);
    if (!occl_pool(pool, false, false)) return d->fpGetQueryPoolResults(device, pool, first, count, size, data, stride, flags);
    atomic_fetch_add(&g_occl_reads, 1);
    uint32_t words = (flags & VK_QUERY_RESULT_64_BIT ? 2 : 1) * (flags & VK_QUERY_RESULT_WITH_AVAILABILITY_BIT ? 2 : 1);
    for (uint32_t i = 0; i < count; i++)
        for (uint32_t w = 0; w < words; w++) ((uint32_t *)((char *)data + i * stride))[w] = 0x10000;
    return VK_SUCCESS;
}

static VKAPI_ATTR VkResult VKAPI_CALL compat_CreateBuffer(
    VkDevice device, const VkBufferCreateInfo *pCreateInfo,
    const VkAllocationCallbacks *pAllocator, VkBuffer *pBuffer) {
    struct DeviceData *d = get_device_data(device);

    VkBufferCreateInfo mod = *pCreateInfo;
    if (mod.usage & VK_BUFFER_USAGE_TRANSFER_SRC_BIT) {
        mod.usage |= VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
    }

    VkResult r = d->fpCreateBuffer(device, &mod, pAllocator, pBuffer);
    if (g_dc_on && r == VK_SUCCESS) dc_buf_create(d, *pBuffer, &mod);
    return r;
}

// -----------------------------------------------------------------------------
// VK_EXT_map_memory_placed, emulated. [Claude Opus 5.5] 2026-10-26
// Wine on WoW64 needs every mapping below 4 GiB. Without this extension it falls back to importing host pointers,
// which the vendor driver only takes for its non-coherent memory type, so DXVK's coherent mappings land above 4 GiB
// ("does not fit 32-bit pointer"). Here the driver maps as usual and the mapping is moved with mremap onto the
// address Wine reserved; the driver's old range is held PROT_NONE so the move back before the driver unmaps cannot
// hit anything else (test/map-probe.c: CPU and GPU see the same data through the moved mapping, also after a remap).
// The driver's flush/invalidate work on its own address, so for moved mappings the layer does the cache maintenance.
// -----------------------------------------------------------------------------
struct MemInfo {
    VkDeviceMemory memory;
    VkDeviceSize size;
    uint32_t type;
    void *orig;   // driver's mapping address while placed, else NULL
    void *placed; // where the mapping lives for the app
    size_t map_size;
    uint32_t heap; // counted in g_heap_usage[heap] (BUDGET_NO_HEAP = not counted)
    uint64_t va;   // memlog: last mapped base (pointer - offset), 0 = never mapped
    struct MemInfo *next;
};
#define MEM_BUCKETS 4096
static struct MemInfo *g_mem[MEM_BUCKETS];
static pthread_mutex_t g_mem_mutex = PTHREAD_MUTEX_INITIALIZER;

static uint32_t mem_bucket(VkDeviceMemory m) {
    uint64_t v = (uint64_t)(uintptr_t)m;
    return (uint32_t)((v ^ (v >> 17) ^ (v >> 33)) % MEM_BUCKETS);
}

// Called with g_mem_mutex held.
static struct MemInfo *mem_find(VkDeviceMemory m) {
    for (struct MemInfo *e = g_mem[mem_bucket(m)]; e; e = e->next)
        if (e->memory == m) return e;
    return NULL;
}

static void mem_add(VkDeviceMemory m, VkDeviceSize size, uint32_t type, uint32_t heap) {
    struct MemInfo *e = calloc(1, sizeof *e);
    if (!e) return; // no entry = never counted, so the free does not subtract either
    e->memory = m;
    e->size = size;
    e->type = type;
    e->heap = heap;
    budget_add(heap, size);
    pthread_mutex_lock(&g_mem_mutex);
    uint32_t b = mem_bucket(m);
    e->next = g_mem[b];
    g_mem[b] = e;
    pthread_mutex_unlock(&g_mem_mutex);
}

static struct MemInfo *mem_remove(VkDeviceMemory m) {
    pthread_mutex_lock(&g_mem_mutex);
    struct MemInfo **pp = &g_mem[mem_bucket(m)];
    while (*pp && (*pp)->memory != m) pp = &(*pp)->next;
    struct MemInfo *e = *pp;
    if (e) *pp = e->next;
    pthread_mutex_unlock(&g_mem_mutex);
    return e;
}

static void *hold_range(void *addr, size_t size, bool replace) {
    return mmap(addr, size, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE | (replace ? MAP_FIXED : MAP_FIXED_NOREPLACE),
                -1, 0);
}

// Moves a placed mapping back onto the driver's held range. Called with g_mem_mutex held; e->orig stays set.
static bool placed_move_back(struct MemInfo *e) {
    void *r = mremap(e->placed, e->map_size, e->map_size, MREMAP_MAYMOVE | MREMAP_FIXED, e->orig);
    if (r == MAP_FAILED) {
        fprintf(stderr, "mali-compat: placed map: moving %p back to %p failed: %s\n", e->placed, e->orig, strerror(errno));
        return false;
    }
    return true;
}

static void cache_maintenance(void *addr, VkDeviceSize size, bool invalidate) {
#if defined(__aarch64__)
    uint64_t ctr;
    __asm__ volatile("mrs %0, ctr_el0" : "=r"(ctr));
    uintptr_t line = 4u << ((ctr >> 16) & 0xf); // DminLine, in bytes
    uintptr_t p = (uintptr_t)addr & ~(line - 1), end = (uintptr_t)addr + size;
    for (; p < end; p += line) {
        if (invalidate) __asm__ volatile("dc civac, %0" ::"r"(p) : "memory");
        else __asm__ volatile("dc cvac, %0" ::"r"(p) : "memory");
    }
    __asm__ volatile("dsb ish" ::: "memory");
#else
    (void)addr, (void)size, (void)invalidate; // only the host test harness builds this file elsewhere
#endif
}

// memlog: M/U events with the allocation's size/type/heap; the va is kept for the F event.
static void ml_mapped(VkDeviceMemory memory, uint64_t va) {
    uint64_t size = 0;
    uint32_t type = ML_NONE, heap = BUDGET_NO_HEAP;
    pthread_mutex_lock(&g_mem_mutex);
    struct MemInfo *e = mem_find(memory);
    if (e) { e->va = va; size = e->size; type = e->type; heap = e->heap; }
    pthread_mutex_unlock(&g_mem_mutex);
    ml_event('M', (uint64_t)(uintptr_t)memory, size, type, heap, va, 1);
}

static void ml_unmapped(VkDeviceMemory memory) {
    uint64_t size = 0, va = 0;
    uint32_t type = ML_NONE, heap = BUDGET_NO_HEAP;
    pthread_mutex_lock(&g_mem_mutex);
    struct MemInfo *e = mem_find(memory);
    if (e) { va = e->va; size = e->size; type = e->type; heap = e->heap; }
    pthread_mutex_unlock(&g_mem_mutex);
    ml_event('U', (uint64_t)(uintptr_t)memory, size, type, heap, va, va ? 1 : 0);
}

// vkMapMemory, handed out only with MALI_COMPAT_MEMLOG (64-bit Wine maps with it, not with map2).
static VKAPI_ATTR VkResult VKAPI_CALL compat_MapMemory(VkDevice device, VkDeviceMemory memory, VkDeviceSize offset,
                                                      VkDeviceSize size, VkMemoryMapFlags flags, void **ppData) {
    struct DeviceData *d = get_device_data(device);
    VkResult r = d->fpMapMemory(device, memory, offset, size, flags, ppData);
    if (g_memlog && r == VK_SUCCESS && ppData && *ppData) ml_mapped(memory, (uint64_t)(uintptr_t)*ppData - offset);
    return r;
}

static VKAPI_ATTR VkResult VKAPI_CALL compat_MapMemory2KHR(VkDevice device, const VkMemoryMapInfoKHR *pMemoryMapInfo,
                                                          void **ppData) {
    struct DeviceData *d = get_device_data(device);
    const VkMemoryMapPlacedInfoEXT *pl = find_struct(pMemoryMapInfo->pNext, VK_STRUCTURE_TYPE_MEMORY_MAP_PLACED_INFO_EXT);
    if (!(pMemoryMapInfo->flags & VK_MEMORY_MAP_PLACED_BIT_EXT) || !pl || !pl->pPlacedAddress) {
        VkResult r0 = !d->fpMapMemory2KHR // VK_KHR_map_memory2 not enabled below (see vkGetDeviceProcAddr)
            ? d->fpMapMemory(device, pMemoryMapInfo->memory, pMemoryMapInfo->offset, pMemoryMapInfo->size, 0, ppData)
            : d->fpMapMemory2KHR(device, pMemoryMapInfo, ppData);
        if (g_memlog && r0 == VK_SUCCESS && ppData && *ppData)
            ml_mapped(pMemoryMapInfo->memory, (uint64_t)(uintptr_t)*ppData - pMemoryMapInfo->offset);
        return r0;
    }
    if (!d->fpMapMemory2KHR) return VK_ERROR_MEMORY_MAP_FAILED;

    VkMemoryMapInfoKHR info = *pMemoryMapInfo;
    info.flags &= ~VK_MEMORY_MAP_PLACED_BIT_EXT;
    struct unlinked u = unlink_struct(&info, VK_STRUCTURE_TYPE_MEMORY_MAP_PLACED_INFO_EXT);
    void *p = NULL;
    VkResult r = d->fpMapMemory2KHR(device, &info, &p);
    relink(u);
    if (r != VK_SUCCESS) return r;

    pthread_mutex_lock(&g_mem_mutex);
    struct MemInfo *e = mem_find(info.memory);
    VkDeviceSize size = info.size == VK_WHOLE_SIZE && e ? e->size - info.offset : info.size;
    size_t map_size = (size + page_size() - 1) & ~(page_size() - 1);
    if (!e || ((uintptr_t)p & (page_size() - 1)) || ((uintptr_t)pl->pPlacedAddress & (page_size() - 1))) {
        pthread_mutex_unlock(&g_mem_mutex);
        fprintf(stderr, "mali-compat: placed map: unknown memory or unaligned mapping %p -> %p\n", p, pl->pPlacedAddress);
        d->fpUnmapMemory(device, info.memory);
        return VK_ERROR_MEMORY_MAP_FAILED;
    }
    void *moved = mremap(p, map_size, map_size, MREMAP_MAYMOVE | MREMAP_FIXED, pl->pPlacedAddress);
    if (moved == MAP_FAILED) {
        pthread_mutex_unlock(&g_mem_mutex);
        fprintf(stderr, "mali-compat: placed map: mremap %p -> %p (%zu bytes) failed: %s\n", p, pl->pPlacedAddress,
                map_size, strerror(errno));
        d->fpUnmapMemory(device, info.memory);
        return VK_ERROR_MEMORY_MAP_FAILED;
    }
    if (hold_range(p, map_size, false) == MAP_FAILED)
        fprintf(stderr, "mali-compat: placed map: could not hold %p: %s\n", p, strerror(errno));
    e->orig = p;
    e->placed = moved;
    e->map_size = map_size;
    pthread_mutex_unlock(&g_mem_mutex);
    LOG("placed map %p: driver %p -> %p (%zu bytes, type %u)", (void *)info.memory, p, moved, map_size, e->type);
    if (g_memlog) ml_mapped(info.memory, (uint64_t)(uintptr_t)p - info.offset); // the driver's address = GPU VA
    *ppData = moved;
    return VK_SUCCESS;
}

// Unmaps a placed mapping: back onto the held range, driver unmap, then the app's range is re-reserved if asked.
// Returns false when `memory` was not placed (the caller unmaps normally).
static bool placed_unmap(struct DeviceData *d, VkDevice device, VkDeviceMemory memory, bool reserve,
                         const VkMemoryUnmapInfoKHR *info2) {
    pthread_mutex_lock(&g_mem_mutex);
    struct MemInfo *e = mem_find(memory);
    if (!e || !e->orig) {
        pthread_mutex_unlock(&g_mem_mutex);
        return false;
    }
    bool back = placed_move_back(e);
    if (back) {
        if (info2 && d->fpUnmapMemory2KHR) {
            VkMemoryUnmapInfoKHR mod = *info2;
            mod.flags &= ~VK_MEMORY_UNMAP_RESERVE_BIT_EXT;
            d->fpUnmapMemory2KHR(device, &mod);
        } else {
            d->fpUnmapMemory(device, memory);
        }
    }
    if (reserve || !back) hold_range(e->placed, e->map_size, true);
    else munmap(e->placed, e->map_size);
    e->orig = e->placed = NULL;
    pthread_mutex_unlock(&g_mem_mutex);
    return true;
}

static VKAPI_ATTR VkResult VKAPI_CALL compat_UnmapMemory2KHR(VkDevice device, const VkMemoryUnmapInfoKHR *pMemoryUnmapInfo) {
    struct DeviceData *d = get_device_data(device);
    if (g_memlog) ml_unmapped(pMemoryUnmapInfo->memory);
    if (placed_unmap(d, device, pMemoryUnmapInfo->memory, pMemoryUnmapInfo->flags & VK_MEMORY_UNMAP_RESERVE_BIT_EXT,
                     pMemoryUnmapInfo))
        return VK_SUCCESS;
    if (!d->fpUnmapMemory2KHR) { d->fpUnmapMemory(device, pMemoryUnmapInfo->memory); return VK_SUCCESS; }
    return d->fpUnmapMemory2KHR(device, pMemoryUnmapInfo);
}

static VKAPI_ATTR void VKAPI_CALL compat_UnmapMemory(VkDevice device, VkDeviceMemory memory) {
    struct DeviceData *d = get_device_data(device);
    if (g_memlog) ml_unmapped(memory);
    if (!placed_unmap(d, device, memory, false, NULL)) d->fpUnmapMemory(device, memory);
}

// Placed ranges get their cache maintenance here; the rest goes to the driver.
static VkResult placed_ranges(struct DeviceData *d, VkDevice device, uint32_t count, const VkMappedMemoryRange *ranges,
                              bool invalidate) {
    VkMappedMemoryRange stack[16];
    VkMappedMemoryRange *rest = count <= 16 ? stack : malloc(count * sizeof *rest);
    if (!rest) return VK_ERROR_OUT_OF_HOST_MEMORY;
    uint32_t n = 0;
    pthread_mutex_lock(&g_mem_mutex);
    for (uint32_t i = 0; i < count; i++) {
        struct MemInfo *e = mem_find(ranges[i].memory);
        if (e && e->orig) {
            if (!(d->coherent_types & (1u << e->type))) {
                VkDeviceSize size = ranges[i].size == VK_WHOLE_SIZE ? e->map_size - ranges[i].offset : ranges[i].size;
                cache_maintenance((char *)e->placed + ranges[i].offset, size, invalidate);
            }
        } else {
            rest[n++] = ranges[i];
        }
    }
    pthread_mutex_unlock(&g_mem_mutex);
    VkResult r = VK_SUCCESS;
    if (n) r = invalidate ? d->fpInvalidateMappedMemoryRanges(device, n, rest) : d->fpFlushMappedMemoryRanges(device, n, rest);
    if (rest != stack) free(rest);
    return r;
}

static VKAPI_ATTR VkResult VKAPI_CALL compat_FlushMappedMemoryRanges(VkDevice device, uint32_t count,
                                                                     const VkMappedMemoryRange *ranges) {
    return placed_ranges(get_device_data(device), device, count, ranges, false);
}

static VKAPI_ATTR VkResult VKAPI_CALL compat_InvalidateMappedMemoryRanges(VkDevice device, uint32_t count,
                                                                          const VkMappedMemoryRange *ranges) {
    return placed_ranges(get_device_data(device), device, count, ranges, true);
}

static VKAPI_ATTR VkResult VKAPI_CALL compat_AllocateMemory(
    VkDevice device, const VkMemoryAllocateInfo *pAllocateInfo,
    const VkAllocationCallbacks *pAllocator, VkDeviceMemory *pMemory) {
    struct DeviceData *d = get_device_data(device);
    // K-CF2: an app-supplied dma-buf fd import is the exporter's memory, so the budget does not count it
    const bool fd_import = find_struct(pAllocateInfo->pNext, VK_STRUCTURE_TYPE_IMPORT_MEMORY_FD_INFO_KHR) != NULL;

    // dma-buf export (MALI_COMPAT_DMABUF_EXPORT): the memory comes from the dma-heap and is imported into the driver
    struct unlinked u_exp = {0}, u_mwsi = {0};
    VkImportMemoryFdInfoKHR imp;
    int export_fd = -1, keep_import_fd = -1;
    VkMemoryAllocateInfo emod;
    if (g_dmabuf_export) {
        u_mwsi = unlink_struct((void *)pAllocateInfo, STYPE_WSI_MEMORY_ALLOCATE_INFO_MESA);
        const VkExportMemoryAllocateInfo *ex = find_struct(pAllocateInfo->pNext, VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO);
        const VkImportMemoryFdInfoKHR *imp_in = find_struct(pAllocateInfo->pNext, VK_STRUCTURE_TYPE_IMPORT_MEMORY_FD_INFO_KHR);
        if (ex && (ex->handleTypes & VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT) && imp_in &&
            imp_in->handleType == VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT) {
            // An import that also asks to be exportable (Mesa Zink's allocate_bo puts both structs on a re-imported
            // GBM buffer): keep the import, drop the export struct the driver refuses, and remember the imported fd
            // so vkGetMemoryFdKHR hands out that same buffer. Allocating a fresh heap buffer here instead put Zink's
            // rendering into memory nobody shared: Xwayland drew Steam into it while gamescope composited the
            // untouched original (black session screen, 2026-10-30; found by a Fable review of Zink's source).
            u_exp = unlink_struct((void *)pAllocateInfo, VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO);
            keep_import_fd = dup(imp_in->fd);
        } else if (ex && (ex->handleTypes & VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT) && d->fpGetMemoryFdKHR &&
            (export_fd = dmaheap_alloc(pAllocateInfo->allocationSize)) >= 0) {
            u_exp = unlink_struct((void *)pAllocateInfo, VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO);
            emod = *pAllocateInfo;
            VkMemoryFdPropertiesKHR fp = {VK_STRUCTURE_TYPE_MEMORY_FD_PROPERTIES_KHR};
            if (d->fpGetMemoryFdPropertiesKHR &&
                d->fpGetMemoryFdPropertiesKHR(device, VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT, export_fd, &fp) == VK_SUCCESS &&
                fp.memoryTypeBits && !(fp.memoryTypeBits & (1u << emod.memoryTypeIndex)))
                emod.memoryTypeIndex = (uint32_t)__builtin_ctz(fp.memoryTypeBits);
            imp = (VkImportMemoryFdInfoKHR){VK_STRUCTURE_TYPE_IMPORT_MEMORY_FD_INFO_KHR, emod.pNext,
                                            VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT, dup(export_fd)};
            emod.pNext = &imp;
            pAllocateInfo = &emod;
        }
    }
    int peek_import_fd = -1;
    if (g_dmabuf_export && export_fd < 0 && keep_import_fd < 0 && getenv("MALI_COMPAT_DMABUF_PEEK")) {
        const VkImportMemoryFdInfoKHR *im = find_struct(pAllocateInfo->pNext, VK_STRUCTURE_TYPE_IMPORT_MEMORY_FD_INFO_KHR);
        if (im && im->handleType == VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT) peek_import_fd = dup(im->fd);
    }
    VkMemoryAllocateInfo mod = *pAllocateInfo;
    VkMemoryAllocateFlagsInfo flags_info = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO,
        .flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT,
        .pNext = mod.pNext,
    };
    mod.pNext = &flags_info;
    // MALI_COMPAT_FORCE_COHERENT=1: allocations of the vendor's non-coherent HOST_CACHED type go to the coherent host
    // type of the same heap instead (the app still sees the type it asked for). vkd3d puts READBACK heaps there and
    // invalidates only in Map(); a D3D12 game that maps once and reads the pointer every frame (fine on Windows, where
    // readback memory is coherent) reads stale CPU cache lines on ARM. Reporting the type as device-local instead
    // crashed HZD at start (DXGI adapter memory). Gemini 4 Argon's lead for the missing statics. [Claude Opus 5.5]
    if (g_force_coherent && mod.memoryTypeIndex < VK_MAX_MEMORY_TYPES && d->coherent_remap[mod.memoryTypeIndex]) {
        static _Atomic uint32_t remapped;
        if (atomic_fetch_add(&remapped, 1) == 0)
            fprintf(stderr, "mali-compat: force coherent: memory type %u -> %u\n", mod.memoryTypeIndex,
                    d->coherent_remap[mod.memoryTypeIndex] - 1u);
        mod.memoryTypeIndex = d->coherent_remap[mod.memoryTypeIndex] - 1u;
    }

    VkResult res = d->fpAllocateMemory(device, &mod, pAllocator, pMemory);
    uint32_t heap = fd_import || mod.memoryTypeIndex >= VK_MAX_MEMORY_TYPES ? BUDGET_NO_HEAP : d->type_heap[mod.memoryTypeIndex];
    if (res == VK_SUCCESS) {
        atomic_fetch_add(&g_allocated_bytes, pAllocateInfo->allocationSize);
        mem_add(*pMemory, pAllocateInfo->allocationSize, mod.memoryTypeIndex, heap);
        if (g_dc_on) dc_mem_alloc(d, device, *pMemory, pAllocateInfo->allocationSize, mod.memoryTypeIndex, dc_mem_devaddr(mod.pNext));
        if (g_memlog) {
            uint64_t va = 0;
            if (g_memlog == 1 && d->fpGetDeviceMemoryOpaqueCaptureAddress) {
                VkDeviceMemoryOpaqueCaptureAddressInfo ci = {VK_STRUCTURE_TYPE_DEVICE_MEMORY_OPAQUE_CAPTURE_ADDRESS_INFO, NULL, *pMemory};
                va = d->fpGetDeviceMemoryOpaqueCaptureAddress(device, &ci);
            }
            ml_event('A', (uint64_t)(uintptr_t)*pMemory, pAllocateInfo->allocationSize, mod.memoryTypeIndex, heap, va, va ? 1 : 0);
        }
    } else if (g_memlog) {
        ml_event('X', 0, pAllocateInfo->allocationSize, mod.memoryTypeIndex, heap, (uint64_t)(int64_t)res, 2);
    }
    if (keep_import_fd >= 0) {
        struct ExportMem *e = res == VK_SUCCESS ? malloc(sizeof *e) : NULL;
        if (e) {
            *e = (struct ExportMem){*pMemory, keep_import_fd, pAllocateInfo->allocationSize, true, NULL};
            pthread_mutex_lock(&g_export_mutex);
            e->next = g_export_mem;
            g_export_mem = e;
            pthread_mutex_unlock(&g_export_mutex);
        } else close(keep_import_fd);
    }
    if (peek_import_fd >= 0) {
        struct ExportMem *e = res == VK_SUCCESS ? malloc(sizeof *e) : NULL;
        if (e) {
            *e = (struct ExportMem){*pMemory, peek_import_fd, pAllocateInfo->allocationSize, true, NULL};
            pthread_mutex_lock(&g_export_mutex);
            e->next = g_export_mem;
            g_export_mem = e;
            pthread_mutex_unlock(&g_export_mutex);
        } else close(peek_import_fd);
    }
    if (export_fd >= 0) {
        if (res == VK_SUCCESS) {
            struct ExportMem *e = malloc(sizeof *e);
            if (e) {
                e->mem = *pMemory;
                e->fd = export_fd;
                e->size = pAllocateInfo->allocationSize;
                e->imported = false;
                static bool peek_started;
                if (!peek_started && getenv("MALI_COMPAT_DMABUF_PEEK") && strcmp(getenv("MALI_COMPAT_DMABUF_PEEK"), "0")) {
                    pthread_t t;
                    peek_started = pthread_create(&t, NULL, dmabuf_peek_thread, NULL) == 0;
                    if (peek_started) pthread_detach(t);
                }
                pthread_mutex_lock(&g_export_mutex);
                e->next = g_export_mem;
                g_export_mem = e;
                pthread_mutex_unlock(&g_export_mutex);
            } else close(export_fd);
        } else {
            close(imp.fd); // a failed import leaves the fd to the caller
            close(export_fd);
            fprintf(stderr, "mali-compat: dma-buf export: importing the heap buffer failed: %d\n", res);
        }
    }
    relink(u_exp);
    relink(u_mwsi);
    return res;
}

static VKAPI_ATTR VkResult VKAPI_CALL compat_GetMemoryFdKHR(VkDevice device, const VkMemoryGetFdInfoKHR *info, int *pFd) {
    if (g_dmabuf_export && info->handleType == VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT) {
        int fd = -1;
        pthread_mutex_lock(&g_export_mutex);
        for (struct ExportMem *x = g_export_mem; x; x = x->next) if (x->mem == info->memory) { fd = dup(x->fd); break; }
        pthread_mutex_unlock(&g_export_mutex);
        if (fd >= 0) { *pFd = fd; return VK_SUCCESS; }
    }
    struct DeviceData *d = get_device_data(device);
    return d->fpGetMemoryFdKHR ? d->fpGetMemoryFdKHR(device, info, pFd) : VK_ERROR_INVALID_EXTERNAL_HANDLE;
}

static VKAPI_ATTR void VKAPI_CALL compat_FreeMemory(
    VkDevice device, VkDeviceMemory memory, const VkAllocationCallbacks *pAllocator) {
    struct DeviceData *d = get_device_data(device);
    if (g_dmabuf_export && memory) {
        pthread_mutex_lock(&g_export_mutex);
        for (struct ExportMem **pp = &g_export_mem; *pp; pp = &(*pp)->next)
            if ((*pp)->mem == memory) { struct ExportMem *x = *pp; *pp = x->next; close(x->fd); free(x); break; }
        pthread_mutex_unlock(&g_export_mutex);
    }
    struct MemInfo *e = memory ? mem_remove(memory) : NULL;
    if (e) budget_sub(e->heap, e->size);
    if (g_dc_on) dc_dead(DCK_MEM, (uint64_t)memory);
    if (g_memlog && memory)
        ml_event('F', (uint64_t)(uintptr_t)memory, e ? e->size : 0, e ? e->type : ML_NONE, e ? e->heap : BUDGET_NO_HEAP,
                 e ? e->va : 0, e && e->va ? 1 : 0);
    if (e && e->orig) {
        // Freeing unmaps implicitly; let the driver do it on its own address and keep Wine's range reserved.
        pthread_mutex_lock(&g_mem_mutex);
        bool back = placed_move_back(e);
        pthread_mutex_unlock(&g_mem_mutex);
        if (back) hold_range(e->placed, e->map_size, true);
    }
    free(e);
    d->fpFreeMemory(device, memory, pAllocator);
}

// State tracking for the restore after a BC decode (see struct PushRec). Graphics binds are passed straight on.
static VKAPI_ATTR void VKAPI_CALL compat_CmdBindPipeline(VkCommandBuffer commandBuffer, VkPipelineBindPoint bindPoint,
                                                         VkPipeline pipeline) {
    struct CmdBufData *c = cmd_data(commandBuffer);
    VkPipeline drv = pipeline;
    if (bindPoint == VK_PIPELINE_BIND_POINT_GRAPHICS) {
        // a stand-in from MALI_COMPAT_ASYNC_PSO: the real pipeline once compiled, else drop the draws until the next bind
        struct AsyncJob *j = async_lookup(pipeline);
        drv = j ? atomic_load(&j->real) : pipeline;
        if (c) c->async_skip = !drv;
    }
    if (!c) { if (drv) get_device_data_by_cmd(commandBuffer)->fpCmdBindPipeline(commandBuffer, bindPoint, drv); return; }
    if (bindPoint == VK_PIPELINE_BIND_POINT_COMPUTE) c->compute_pipeline = pipeline;
    else if (bindPoint == VK_PIPELINE_BIND_POINT_GRAPHICS) {
        c->gfx_pipeline = pipeline;
        struct HashNode h;
        c->gfx_flags = g_gpu_time && hash_get(g_pipeline_hash, (uint64_t)pipeline, &h) ? (uint8_t)h.flags : 0;
    }
    if (drv) c->dev->fpCmdBindPipeline(commandBuffer, bindPoint, drv);
}

// Record one app push (v1 or v2) for the restore after a BC decode. Same layout/stages/range as an earlier record:
// update it and move it to the end (replay order = app order, across both entry points).
static void push_record(struct CmdBufData *c, VkPipelineLayout layout, VkShaderStageFlags stages, uint32_t offset,
                        uint32_t size, const void *pValues, bool v2) {
    uint32_t i = 0;
    while (i < c->push_count && !(c->push[i].layout == layout && c->push[i].stages == stages &&
                                  c->push[i].offset == offset && c->push[i].size == size)) i++;
    if (i == c->push_count && c->push_count == PUSH_RECS) i = 0; // full: drop the oldest
    if (i < c->push_count) {
        memmove(&c->push[i], &c->push[i + 1], (c->push_count - i - 1) * sizeof c->push[0]);
        c->push_count--;
    }
    struct PushRec *r = &c->push[c->push_count++];
    r->layout = layout;
    r->stages = stages;
    r->offset = offset;
    r->size = size;
    r->v2 = v2;
    memcpy(r->data, pValues, size);
}

static VKAPI_ATTR void VKAPI_CALL compat_CmdPushConstants(VkCommandBuffer commandBuffer, VkPipelineLayout layout,
                                                          VkShaderStageFlags stages, uint32_t offset, uint32_t size,
                                                          const void *pValues) {
    pc2_count(PC2_V1);
    struct CmdBufData *c = cmd_data(commandBuffer);
    if (!c) {
        get_device_data_by_cmd(commandBuffer)->fpCmdPushConstants(commandBuffer, layout, stages, offset, size, pValues);
        return;
    }
    if (offset + size <= 256) push_record(c, layout, stages, offset, size, pValues, false);
    c->dev->fpCmdPushConstants(commandBuffer, layout, stages, offset, size, pValues);
}

// vkCmdPushConstants2 / vkCmdPushConstants2KHR (pc2-restore-1): recorded like v1. A push with layout VK_NULL_HANDLE
// (layout given as VkPipelineLayoutCreateInfo in pNext) or past 256 bytes is not recorded, only counted.
static void push2_common(VkCommandBuffer commandBuffer, const VkPushConstantsInfo *info, bool khr) {
    pc2_count(PC2_V2);
    struct CmdBufData *c = cmd_data(commandBuffer);
    struct DeviceData *d = c ? c->dev : get_device_data_by_cmd(commandBuffer);
    if (info && (info->layout == VK_NULL_HANDLE || info->offset + info->size > 256))
        pc2_count(PC2_V2_UNREC);
    else if (c && info)
        push_record(c, info->layout, info->stageFlags, info->offset, info->size, info->pValues, true);
    PFN_vkCmdPushConstants2 fp = khr ? d->fpCmdPushConstants2KHR : d->fpCmdPushConstants2;
    fp(commandBuffer, info);
}
static VKAPI_ATTR void VKAPI_CALL compat_CmdPushConstants2(VkCommandBuffer commandBuffer, const VkPushConstantsInfo *info) {
    push2_common(commandBuffer, info, false);
}
static VKAPI_ATTR void VKAPI_CALL compat_CmdPushConstants2KHR(VkCommandBuffer commandBuffer, const VkPushConstantsInfo *info) {
    push2_common(commandBuffer, info, true);
}

static void *pc2_log_thread(void *arg) {
    (void)arg;
    for (;;) {
        sleep(10);
        unsigned long long v1 = pc2_sum(PC2_V1), v2 = pc2_sum(PC2_V2), un = pc2_sum(PC2_V2_UNREC),
                           dec = pc2_sum(PC2_DECODES), dv2 = pc2_sum(PC2_DECODES_V2);
        if (v1 | v2 | un | dec | dv2)
            fprintf(stderr, "mali-compat: pc2: v1 pushes %llu, v2 pushes %llu, v2 unrecorded %llu, BC decodes %llu, "
                    "decodes replaying v2 %llu\n", v1, v2, un, dec, dv2);
    }
    return NULL;
}

static VKAPI_ATTR void VKAPI_CALL compat_CmdBindDescriptorBuffersEXT(VkCommandBuffer commandBuffer, uint32_t count,
                                                                     const VkDescriptorBufferBindingInfoEXT *infos) {
    struct CmdBufData *c = cmd_data(commandBuffer);
    if (c) c->descriptor_buffers = true;
    get_device_data_by_cmd(commandBuffer)->fpCmdBindDescriptorBuffersEXT(commandBuffer, count, infos);
}

// BC upload: decode every region on the GPU into one staging buffer (RGBA8 / RG16 / RGBA16F texels), then copy that
// into the RGBA8 shadow image. Array layers are decoded like 3D slices (same buffer layout). Rewritten
// [Claude Opus 5.5] 2026-10-26: one allocation per call (was one per region), layers (only layer 0 was decoded and the
// copy read past the staging buffer), allocation errors logged.
static VKAPI_ATTR void VKAPI_CALL compat_CmdCopyBufferToImage(
    VkCommandBuffer commandBuffer, VkBuffer srcBuffer, VkImage dstImage,
    VkImageLayout dstImageLayout, uint32_t regionCount, const VkBufferImageCopy *pRegions) {
    struct DeviceData *d = get_device_data_by_cmd(commandBuffer);
    struct TrackedBCImage tracked;

    if (!d || !find_bc_image(dstImage, &tracked)) {
        d->fpCmdCopyBufferToImage(commandBuffer, srcBuffer, dstImage, dstImageLayout, regionCount, pRegions);
        return;
    }
    if (init_bc_compute(d) != VK_SUCCESS || !regionCount) return;
    uint64_t t0 = g_stats ? now_ns() : 0;

    VkBufferDeviceAddressInfo src_bda_info = {VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO, NULL, srcBuffer};
    VkDeviceAddress src_bda = d->fpGetBufferDeviceAddress(d->device, &src_bda_info);
    uint32_t shader_idx = panvk_bc_shader_index(tracked.orig_format);
    uint32_t blk_B = panvk_bc_block_size(tracked.orig_format);
    uint32_t texel_B = panvk_bc_decoded_texel_size(tracked.orig_format);

    VkBufferImageCopy stack_regs[16];
    VkBufferImageCopy *dec = regionCount <= 16 ? stack_regs : malloc(regionCount * sizeof *dec);
    VkDeviceSize *dst_off = malloc(regionCount * sizeof *dst_off);
    if (!dec || !dst_off) { if (dec != stack_regs) free(dec); free(dst_off); return; }
    VkDeviceSize total = 0;
    for (uint32_t i = 0; i < regionCount; i++) {
        const VkBufferImageCopy *reg = &pRegions[i];
        uint32_t slices = (reg->imageExtent.depth ? reg->imageExtent.depth : 1) * (reg->imageSubresource.layerCount ? reg->imageSubresource.layerCount : 1);
        dst_off[i] = total;
        total += (VkDeviceSize)reg->imageExtent.width * reg->imageExtent.height * slices * texel_B;
        total = (total + 63) & ~(VkDeviceSize)63;
    }

    struct StagingResource staging;
    VkResult r = acquire_staging(d, total, &staging);
    if (r != VK_SUCCESS) {
        fprintf(stderr, "mali-compat: BC upload: staging buffer of %llu bytes failed (%d); texture left undecoded\n",
                (unsigned long long)total, r);
        if (dec != stack_regs) free(dec);
        free(dst_off);
        return;
    }
    add_staging_resource(commandBuffer, staging);
    VkBuffer staging_buf = staging.buffer;
    VkDeviceAddress dst_bda = staging.addr;

    // The source was written by the host (upload/staging buffers) or by an earlier copy or compute dispatch. Waiting
    // on those stages only: ALL_COMMANDS made every BC upload wait for all earlier rendering in the queue (a
    // fragment drain) before the decode could start. [Claude Opus 5.5] 2026-10-26
    VkMemoryBarrier pre_b = {VK_STRUCTURE_TYPE_MEMORY_BARRIER, NULL,
                             VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_HOST_WRITE_BIT | VK_ACCESS_SHADER_WRITE_BIT,
                             VK_ACCESS_SHADER_READ_BIT};
    d->fpCmdPipelineBarrier(commandBuffer,
                            VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_HOST_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &pre_b, 0, NULL, 0, NULL);
    struct CmdBufData *cstate = cmd_data(commandBuffer);
    static struct CmdBufData no_state;
    if (!cstate) cstate = &no_state;
    VkPipeline decoder = cstate->descriptor_buffers && d->bc_pipelines_db[shader_idx] ? d->bc_pipelines_db[shader_idx]
                                                                                      : d->bc_pipelines[shader_idx];
    d->fpCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, decoder);
    for (uint32_t i = 0; i < regionCount; i++) {
        const VkBufferImageCopy *reg = &pRegions[i];
        uint32_t w = reg->imageExtent.width, h = reg->imageExtent.height;
        uint32_t slices = (reg->imageExtent.depth ? reg->imageExtent.depth : 1) * (reg->imageSubresource.layerCount ? reg->imageSubresource.layerCount : 1);
        uint32_t row_px = reg->bufferRowLength ? reg->bufferRowLength : w;
        uint32_t col_px = reg->bufferImageHeight ? reg->bufferImageHeight : h;
        uint32_t src_row_B = ((row_px + 3) / 4) * blk_B;
        uint32_t src_slice_B = ((col_px + 3) / 4) * src_row_B;
        uint64_t full_src = src_bda + reg->bufferOffset, full_dst = dst_bda + dst_off[i];
        struct panvk_bc_decode_push push = {
            .src = { (uint32_t)full_src, (uint32_t)(full_src >> 32) },
            .dst = { (uint32_t)full_dst, (uint32_t)(full_dst >> 32) },
            .src_row_B = src_row_B,
            .src_slice_B = src_slice_B,
            .dst_row_B = w * texel_B,
            .dst_slice_B = w * h * texel_B,
            .width = (int32_t)w,
            .height = (int32_t)h,
            .depth = (int32_t)slices,
            .format = tracked.orig_format,
            .x0 = 0,
            .y0 = 0,
        };
        d->fpCmdPushConstants(commandBuffer, d->bc_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);
        d->fpCmdDispatch(commandBuffer, (w + 7) / 8, (h + 7) / 8, slices);
        dec[i] = *reg;
        dec[i].bufferOffset = dst_off[i];
        dec[i].bufferRowLength = 0;
        dec[i].bufferImageHeight = 0;
    }
    VkMemoryBarrier post_b = {VK_STRUCTURE_TYPE_MEMORY_BARRIER, NULL, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT};
    d->fpCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1,
                            &post_b, 0, NULL, 0, NULL);
    d->fpCmdCopyBufferToImage(commandBuffer, staging_buf, dstImage, dstImageLayout, regionCount, dec);
    // Give the app its compute pipeline and push constants back (H1).
    if (cstate->compute_pipeline)
        d->fpCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, cstate->compute_pipeline);
    // Each record with the entry point the app used (pc2-restore-1); v2 without the app's pNext (only a layout create
    // info can be there, and such pushes are not recorded).
    bool replayed_v2 = false;
    for (uint32_t i = 0; i < cstate->push_count; i++) {
        const struct PushRec *pr = &cstate->push[i];
        PFN_vkCmdPushConstants2 fp2 = d->fpCmdPushConstants2KHR ? d->fpCmdPushConstants2KHR : d->fpCmdPushConstants2;
        if (pr->v2 && fp2) {
            VkPushConstantsInfo pi = {VK_STRUCTURE_TYPE_PUSH_CONSTANTS_INFO, NULL, pr->layout, pr->stages, pr->offset,
                                      pr->size, pr->data};
            fp2(commandBuffer, &pi);
            replayed_v2 = true;
        } else {
            d->fpCmdPushConstants(commandBuffer, pr->layout, pr->stages, pr->offset, pr->size, pr->data);
        }
    }
    pc2_count(PC2_DECODES);
    if (replayed_v2) pc2_count(PC2_DECODES_V2);
    // Only with MALI_COMPAT_VENDOR_ID: the recorded writes can name image/buffer views the app has destroyed since,
    // and pushing them again crashed Horizon Zero Dawn's benchmark load (C0000005, 2026-10-28 22:49).
    if (g_vendor_id && cstate->cpd.layout && !cstate->cpd.bad && cstate->cpd.count && d->fpCmdPushDescriptorSetKHR)
        d->fpCmdPushDescriptorSetKHR(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, cstate->cpd.layout, cstate->cpd.set,
                                     cstate->cpd.count, cstate->cpd.w);
    if (dec != stack_regs) free(dec);
    free(dst_off);
    if (g_stats) {
        uint64_t t1 = now_ns();
        atomic_fetch_add(&g_st_uploads, 1);
        atomic_fetch_add(&g_st_regions, regionCount);
        atomic_fetch_add(&g_st_bytes, total);
        atomic_fetch_add(&g_st_ns, t1 - t0);
        uint64_t last = atomic_load(&g_st_last_ns);
        if (!last && g_stats_interval_ns) atomic_compare_exchange_strong(&g_st_last_ns, &last, t1);
        else if (t1 - last >= g_stats_interval_ns && atomic_compare_exchange_strong(&g_st_last_ns, &last, t1)) {
            pthread_mutex_lock(&g_staging_pool_mutex);
            unsigned long long pool = d->staging_pool_bytes;
            pthread_mutex_unlock(&g_staging_pool_mutex);
            fprintf(stderr, "mali-compat: stats %.1fs: BC uploads %llu (regions %llu, decoded %.1f MiB), staging new %llu "
                    "reused %llu, CPU in BC path %.2f ms, pool %.1f MiB\n", last ? (t1 - last) / 1e9 : 0.0,
                    (unsigned long long)atomic_exchange(&g_st_uploads, 0), (unsigned long long)atomic_exchange(&g_st_regions, 0),
                    atomic_exchange(&g_st_bytes, 0) / 1048576.0, (unsigned long long)atomic_exchange(&g_st_new, 0),
                    (unsigned long long)atomic_exchange(&g_st_reused, 0), atomic_exchange(&g_st_ns, 0) / 1e6, pool / 1048576.0);
        }
    }
}

static VKAPI_ATTR void VKAPI_CALL compat_CmdCopyBufferToImage2(
    VkCommandBuffer commandBuffer, const VkCopyBufferToImageInfo2 *pCopyBufferToImageInfo) {
    struct DeviceData *d = get_device_data_by_cmd(commandBuffer);
    struct TrackedBCImage tracked;

    if (!d || !find_bc_image(pCopyBufferToImageInfo->dstImage, &tracked)) {
        if (d && d->fpCmdCopyBufferToImage2) {
            d->fpCmdCopyBufferToImage2(commandBuffer, pCopyBufferToImageInfo);
        }
        return;
    }

    uint32_t rc = pCopyBufferToImageInfo->regionCount;
    VkBufferImageCopy *regions = (VkBufferImageCopy *)malloc(rc * sizeof(VkBufferImageCopy));
    if (!regions) return;
    for (uint32_t i = 0; i < rc; i++) {
        const VkBufferImageCopy2 *r2 = &pCopyBufferToImageInfo->pRegions[i];
        regions[i] = (VkBufferImageCopy){
            .bufferOffset = r2->bufferOffset,
            .bufferRowLength = r2->bufferRowLength,
            .bufferImageHeight = r2->bufferImageHeight,
            .imageSubresource = r2->imageSubresource,
            .imageOffset = r2->imageOffset,
            .imageExtent = r2->imageExtent,
        };
    }
    compat_CmdCopyBufferToImage(commandBuffer, pCopyBufferToImageInfo->srcBuffer,
                                pCopyBufferToImageInfo->dstImage, pCopyBufferToImageInfo->dstImageLayout,
                                rc, regions);
    free(regions);
}

static VKAPI_ATTR VkResult VKAPI_CALL compat_ResetCommandBuffer(
    VkCommandBuffer commandBuffer, VkCommandBufferResetFlags flags) {
    struct DeviceData *d = get_device_data_by_cmd(commandBuffer);
    if (d) reset_cmd_data(d, commandBuffer, false);
    return d ? d->fpResetCommandBuffer(commandBuffer, flags) : VK_SUCCESS;
}

static VKAPI_ATTR VkResult VKAPI_CALL compat_BeginCommandBuffer(
    VkCommandBuffer commandBuffer, const VkCommandBufferBeginInfo *pBeginInfo) {
    struct DeviceData *d = get_device_data_by_cmd(commandBuffer);
    if (d) reset_cmd_data(d, commandBuffer, false);
    return d ? d->fpBeginCommandBuffer(commandBuffer, pBeginInfo) : VK_SUCCESS;
}

static VKAPI_ATTR void VKAPI_CALL compat_FreeCommandBuffers(
    VkDevice device, VkCommandPool commandPool, uint32_t commandBufferCount,
    const VkCommandBuffer *pCommandBuffers) {
    struct DeviceData *d = get_device_data(device);
    if (d) {
        for (uint32_t i = 0; i < commandBufferCount; i++) {
            reset_cmd_data(d, pCommandBuffers[i], true);
        }
    }
    d->fpFreeCommandBuffers(device, commandPool, commandBufferCount, pCommandBuffers);
}

static VKAPI_ATTR VkResult VKAPI_CALL compat_AllocateCommandBuffers(
    VkDevice device, const VkCommandBufferAllocateInfo *pAllocateInfo, VkCommandBuffer *pCommandBuffers) {
    struct DeviceData *d = get_device_data(device);
    if (!d || !d->fpAllocateCommandBuffers) return VK_ERROR_INITIALIZATION_FAILED;
    VkResult res = d->fpAllocateCommandBuffers(device, pAllocateInfo, pCommandBuffers);
    if (res == VK_SUCCESS && pAllocateInfo && pCommandBuffers) {
        for (uint32_t i = 0; i < pAllocateInfo->commandBufferCount; i++) {
            struct CmdBufData *e = cmd_data(pCommandBuffers[i]);
            if (e) {
                e->dev = d;
                e->pool = pAllocateInfo->commandPool;
            }
        }
    }
    return res;
}

static VKAPI_ATTR VkResult VKAPI_CALL compat_ResetCommandPool(
    VkDevice device, VkCommandPool commandPool, VkCommandPoolResetFlags flags) {
    // F1: only the staging buffers go back here (the pool's command buffers may be begun again only after the GPU is
    // done with them, and must be: vkBeginCommandBuffer runs reset_cmd_data, which clears the tracked state). So only
    // the g_staged entries are visited, not the whole table. The pool is externally synchronized with its command
    // buffers, so no other thread touches these entries meanwhile.
    struct DeviceData *d = get_device_data(device);
    if (d) {
        struct CmdBufData *mine = NULL;
        pthread_mutex_lock(&g_cmdbuf_mutex);
        for (struct CmdBufData *e = g_staged, *nx; e; e = nx) {
            nx = e->st_next;
            if (e->dev == d && e->pool == commandPool) {
                staged_unlink(e);
                e->st_next = mine; // private chain until released below (st_on is false again)
                mine = e;
            }
        }
        pthread_mutex_unlock(&g_cmdbuf_mutex);
        for (struct CmdBufData *e = mine, *nx; e; e = nx) {
            nx = e->st_next;
            e->st_next = NULL;
            for (uint32_t j = 0; j < e->staging_count; j++) release_staging(d, &e->staging_resources[j]);
            e->staging_count = 0;
        }
    }
    return d ? d->fpResetCommandPool(device, commandPool, flags) : VK_SUCCESS;
}

static VKAPI_ATTR void VKAPI_CALL compat_DestroyCommandPool(
    VkDevice device, VkCommandPool commandPool, const VkAllocationCallbacks *pAllocator) {
    struct DeviceData *d = get_device_data(device);
    if (d) {
        pthread_mutex_lock(&g_cmdbuf_mutex);
        for (int b = 0; b < CMDBUF_BUCKETS; b++) {
            struct CmdBufData **pp = &g_cmdbufs[b];
            while (*pp) {
                struct CmdBufData *e = *pp;
                if (e->dev == d && e->pool == commandPool) {
                    *pp = e->next;
                    staged_unlink(e);
                    for (uint32_t j = 0; j < e->staging_count; j++)
                        release_staging(d, &e->staging_resources[j]);
                    free(e->staging_resources);
                    free(e);
                    atomic_fetch_add_explicit(&g_cmdbuf_gen, 1, memory_order_release);
                } else {
                    pp = &e->next;
                }
            }
        }
        pthread_mutex_unlock(&g_cmdbuf_mutex);
        d->fpDestroyCommandPool(device, commandPool, pAllocator);
    }
}

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

static void gt_destroy(struct DeviceData *d) {
    struct GpuTime *g = d->gt;
    if (!g) return;
    if (g->pool) {
        atomic_store(&g->stop, true);
        pthread_join(g->thread, NULL);
        g->fpDestroyQueryPool(d->device, g->pool, NULL);
    }
    d->gt = NULL;
    free(g);
}

// -----------------------------------------------------------------------------
// sync-diag (The Witcher 3 DX12 terrain flashes, docs/games/witcher3.md § "flash plan 1105" candidates 5-7)
// [keeper, Claude Opus 5.5] 2026-11-05. Diagnosis switches, not fixes; off unless one of the keys below is set, and
// then the hooks are handed out at vkGetDeviceProcAddr only (unset = not one extra instruction on any Vulkan call).
//   MALI_COMPAT_FULL_BARRIERS=1  every vkCmdPipelineBarrier/2 outside a render pass gets ALL_COMMANDS -> ALL_COMMANDS
//                                and MEMORY_READ|WRITE on every barrier (layouts and queue families kept, BY_REGION
//                                dropped) plus one global memory barrier; after vkCmdWaitEvents/2 a full barrier.
//                             =2 1 + a full barrier before every render pass instance (not before a RESUMING part)
//                                and after every one (not after a SUSPENDING part), and after every dispatch, copy,
//                                blit, resolve, fill, update, clear and query copy outside a pass. Indirect arguments
//                                written by compute are then visible to any later draw (ALL_COMMANDS includes
//                                DRAW_INDIRECT). Barriers inside a pass and between suspend/resume parts stay as
//                                they are (both would be invalid).
//   MALI_COMPAT_SUBMIT_IDLE=1|<ms>  after each successful vkQueueSubmit/2 a fence-only submit on the same queue and
//                                a wait on it (cap 250 ms for 1, else <ms>): the GPU has finished all work of that
//                                queue before the next submit is recorded. A wait that hits the cap (wait-before-
//                                signal on a timeline the other queue signals later) is counted and left behind.
//   MALI_COMPAT_GT_SPLIT_SKIP=1  K4 (control.md § K4): GPU_TIME writes no timestamps around SUSPENDING/RESUMING
//                                parts (an action command between suspend and resume is invalid); those parts are
//                                not timed. Only matters with MALI_COMPAT_GPU_TIME=1.
//   MALI_COMPAT_SYNC_CTL=<file>  arms all three and reads the mode from <file> every 250 ms, so one launch can run
//                                windows with different modes: "fb=2 si=1 k4=0" (any subset, "off" = all 0). The
//                                keys above give the start mode. Each mode change and every 10 s a counter line go to
//                                stderr and to MALI_COMPAT_SYNC_LOG (default /root/mali-sync-diag-<comm>.txt).
//                                Counters per line are deltas since the previous line (MALI_COMPAT_SYNC_REPORT_S).
//   Class mask (fb-narrow, 11-05 20:xx, after witcher-syncdiag-1915 branch G: fb=2 removed the flashes, si did not):
//   ctl "fbm=<mask>" / env MALI_COMPAT_FB_MASK=<mask> (hex 0x.. or decimal) picks single classes; fb=1 = 0x00f, fb=2 =
//   0x3ff, fb=0 = 0. Bits: 0x001 app barrier dst stage |= ALL_COMMANDS, 0x002 src stage |= ALL_COMMANDS, 0x004 access
//   masks |= MEMORY_READ|WRITE + one appended global barrier + BY_REGION dropped, 0x008 full barrier after WaitEvents,
//   0x010 before a pass, 0x020 after a pass, 0x040 after a dispatch, 0x080 after a copy/blit/resolve/fill/update,
//   0x100 after a clear-image, 0x200 after a query-pool copy. Armed in any mode: a histogram of the app's barriers
//   outside passes by stage (dst tess = TESS/GEOMETRY/PRE_RASTERIZATION, ...) in the 10-s line.
// -----------------------------------------------------------------------------
enum { SD_PB, SD_PB2, SD_WE, SD_WE2, SD_BCB, SD_BR, SD_ER, SD_BRP, SD_BRP2, SD_ERP, SD_ERP2, SD_D, SD_DI, SD_DB,
       SD_CB, SD_CB2, SD_CI, SD_CI2, SD_CBI, SD_CBI2, SD_CIB, SD_CIB2, SD_BL, SD_BL2, SD_RS, SD_RS2, SD_FILL, SD_UPD,
       SD_CLC, SD_CLD, SD_CQP, SD_QS, SD_QS2, SD_N };
enum { SDC_WIDEN, SDC_INPASS, SDC_EXTRA, SDC_SPLIT, SDC_SUBMITS, SDC_IDLED, SDC_TIMEOUT, SDC_IDLE_NS, SDC_FAIL, SDC_K4,
       SDC_H_BARRIERS, SDC_H_DTESS, SDC_H_DIND, SDC_H_DVTX, SDC_H_DFRAG, SDC_H_DCS, SDC_H_DXFER, SDC_H_DALL, SDC_H_SCS,
       SDC_H_SXFER, SDC_H_SFRAG, SDC_N };
struct SdDev {
    PFN_vkVoidFunction next[SD_N]; // what this layer would hand out without sync-diag (its own hook or the driver's)
    PFN_vkQueueSubmit drvSubmit;
    PFN_vkWaitForFences drvWait;
    PFN_vkResetFences drvReset;
    PFN_vkCreateFence drvCreate;
    PFN_vkDestroyFence drvDestroy;
    PFN_vkGetFenceStatus drvStatus;
};
#define SD_FENCES 128
struct SdFence { VkDevice dev; pid_t tid; VkFence f; uint8_t state; }; // state 0 free, 1 ready, 2 left behind
static struct SdFence g_sd_fences[SD_FENCES];
static pthread_mutex_t g_sd_mutex = PTHREAD_MUTEX_INITIALIZER;
static bool g_sd_armed;
static _Atomic uint32_t g_sd_fbm; // class mask, see FBM_* (fb=1 -> FBM_L1, fb=2 -> FBM_ALL)
enum { FBM_DST = 0x001, FBM_SRC = 0x002, FBM_ACC = 0x004, FBM_WAITEV = 0x008, FBM_PRE = 0x010, FBM_POST = 0x020,
       FBM_DISP = 0x040, FBM_XFER = 0x080, FBM_CLEAR = 0x100, FBM_QCOPY = 0x200, FBM_L1 = 0x00f, FBM_ALL = 0x3ff };
static uint32_t sd_fb_mask(int level) { return level <= 0 ? 0 : level == 1 ? FBM_L1 : FBM_ALL; }
static int sd_fb_level(uint32_t m) { return !m ? 0 : m == FBM_L1 ? 1 : m == FBM_ALL ? 2 : 3; } // 3 = a class subset
#define SD_FB(bit) (atomic_load_explicit(&g_sd_fbm, memory_order_relaxed) & (bit))
static _Atomic uint32_t g_sd_si_ms;
static _Atomic bool g_sd_k4;
static const char *g_sd_ctl;
static char g_sd_log[192];
static uint64_t g_sd_report_ns = 10000000000ull; // MALI_COMPAT_SYNC_REPORT_S (PC test: 1)
static _Atomic uint64_t g_sd_c[SDC_N], g_sd_idle_max_ns;
static __thread bool tl_sd_inner;
#define SD_ACC (VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT)
#define SD_ACC2 (VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT)
#define SD_CNT(i) atomic_fetch_add_explicit(&g_sd_c[i], 1, memory_order_relaxed)

static uint32_t sd_si_parse(const char *v) {
    long n = v ? strtol(v, NULL, 10) : 0;
    return n <= 0 ? 0 : n == 1 ? 250 : (uint32_t)n;
}

static void sd_line(const char *fmt, ...) { // stderr (the Proton log) and the status file
    char buf[640];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(buf, sizeof buf, fmt, ap);
    va_end(ap);
    fprintf(stderr, "mali-compat: %s\n", buf);
    FILE *f = g_sd_log[0] ? fopen(g_sd_log, "a") : NULL;
    if (f) { fprintf(f, "%s\n", buf); fclose(f); }
}

static void sd_stamp(char *out, size_t len) { // "HH:MM:SS epoch E.EEE" (the screenrecord stamps are epoch seconds)
    struct timespec ts;
    struct tm tm;
    clock_gettime(CLOCK_REALTIME, &ts);
    localtime_r(&ts.tv_sec, &tm);
    snprintf(out, len, "%02d:%02d:%02d epoch %lld.%03ld", tm.tm_hour, tm.tm_min, tm.tm_sec, (long long)ts.tv_sec,
             ts.tv_nsec / 1000000);
}

static void sd_mode_line(const char *why) {
    char st[48];
    sd_stamp(st, sizeof st);
    uint32_t m = atomic_load(&g_sd_fbm);
    sd_line("sync-diag mode fb %d si %u ms k4 %d at %s (%s) fbm 0x%03x", sd_fb_level(m), atomic_load(&g_sd_si_ms),
            (int)atomic_load(&g_sd_k4), st, why, m); // fbm last: the tests-lane parsers match up to "(why)"
}

// "fb=2 si=1 k4=0", also FULL_BARRIERS= / SUBMIT_IDLE= / GT_SPLIT_SKIP=, separated by blanks, commas or newlines.
static void sd_apply(char *s) {
    for (char *t = strtok(s, " \t\r\n,;"); t; t = strtok(NULL, " \t\r\n,;")) {
        char *eq = strchr(t, '=');
        if (!strcmp(t, "off")) { atomic_store(&g_sd_fbm, 0); atomic_store(&g_sd_si_ms, 0); atomic_store(&g_sd_k4, false); }
        if (!eq) continue;
        *eq++ = 0;
        if (!strcmp(t, "fb") || !strcmp(t, "FULL_BARRIERS")) atomic_store(&g_sd_fbm, sd_fb_mask(atoi(eq)));
        else if (!strcmp(t, "fbm") || !strcmp(t, "FB_MASK")) atomic_store(&g_sd_fbm, (uint32_t)strtoul(eq, NULL, 0) & FBM_ALL);
        else if (!strcmp(t, "si") || !strcmp(t, "SUBMIT_IDLE")) atomic_store(&g_sd_si_ms, sd_si_parse(eq));
        else if (!strcmp(t, "k4") || !strcmp(t, "GT_SPLIT_SKIP")) atomic_store(&g_sd_k4, atoi(eq) != 0);
    }
}

static void *sd_thread(void *arg) {
    (void)arg;
    struct timespec last_m = {0};
    off_t last_size = -1;
    uint64_t prev[SDC_N] = {0}, t_last = now_ns();
    for (;;) {
        struct timespec q = {0, 250000000};
        nanosleep(&q, NULL);
        struct stat sb;
        if (g_sd_ctl && stat(g_sd_ctl, &sb) == 0 &&
            (sb.st_mtim.tv_sec != last_m.tv_sec || sb.st_mtim.tv_nsec != last_m.tv_nsec || sb.st_size != last_size)) {
            last_m = sb.st_mtim;
            last_size = sb.st_size;
            char buf[256] = {0};
            FILE *f = fopen(g_sd_ctl, "r");
            if (f) {
                size_t n = fread(buf, 1, sizeof buf - 1, f);
                buf[n] = 0;
                fclose(f);
                uint32_t fb = atomic_load(&g_sd_fbm); uint32_t si = atomic_load(&g_sd_si_ms); bool k4 = atomic_load(&g_sd_k4);
                sd_apply(buf);
                if (fb != atomic_load(&g_sd_fbm) || si != atomic_load(&g_sd_si_ms) || k4 != atomic_load(&g_sd_k4))
                    sd_mode_line("ctl file");
            }
        }
        uint64_t now = now_ns();
        if (now - t_last < g_sd_report_ns) continue;
        uint64_t c[SDC_N], dl[SDC_N];
        for (int i = 0; i < SDC_N; i++) { c[i] = atomic_load(&g_sd_c[i]); dl[i] = c[i] - prev[i]; prev[i] = c[i]; }
        char st[48];
        sd_stamp(st, sizeof st);
        sd_line("sync-diag %.1fs %s: fb %d si %u k4 %d | barriers widened %llu, in-pass kept %llu, added %llu, "
                "split-part skips %llu | submits %llu, idled %llu, timeouts %llu, idle avg %.2f ms max %.2f ms, fails %llu"
                " | k4 untimed parts %llu | fbm 0x%03x | app barriers %llu: dst tess %llu, dst indirect %llu, dst vtx %llu,"
                " dst frag %llu, dst compute %llu, dst transfer %llu, dst all %llu, src compute %llu, src transfer %llu,"
                " src frag %llu",
                (now - t_last) / 1e9, st, sd_fb_level(atomic_load(&g_sd_fbm)), atomic_load(&g_sd_si_ms), (int)atomic_load(&g_sd_k4),
                (unsigned long long)dl[SDC_WIDEN], (unsigned long long)dl[SDC_INPASS], (unsigned long long)dl[SDC_EXTRA],
                (unsigned long long)dl[SDC_SPLIT], (unsigned long long)dl[SDC_SUBMITS], (unsigned long long)dl[SDC_IDLED],
                (unsigned long long)dl[SDC_TIMEOUT], dl[SDC_IDLED] ? dl[SDC_IDLE_NS] / 1e6 / dl[SDC_IDLED] : 0.0,
                atomic_exchange(&g_sd_idle_max_ns, 0) / 1e6, (unsigned long long)dl[SDC_FAIL],
                (unsigned long long)dl[SDC_K4], atomic_load(&g_sd_fbm), (unsigned long long)dl[SDC_H_BARRIERS],
                (unsigned long long)dl[SDC_H_DTESS], (unsigned long long)dl[SDC_H_DIND], (unsigned long long)dl[SDC_H_DVTX],
                (unsigned long long)dl[SDC_H_DFRAG], (unsigned long long)dl[SDC_H_DCS], (unsigned long long)dl[SDC_H_DXFER],
                (unsigned long long)dl[SDC_H_DALL], (unsigned long long)dl[SDC_H_SCS], (unsigned long long)dl[SDC_H_SXFER],
                (unsigned long long)dl[SDC_H_SFRAG]);
        t_last = now;
    }
    return NULL;
}

static void sd_init_once(void) {
    const char *fb = getenv("MALI_COMPAT_FULL_BARRIERS"), *si = getenv("MALI_COMPAT_SUBMIT_IDLE");
    const char *k4 = getenv("MALI_COMPAT_GT_SPLIT_SKIP"), *ctl = getenv("MALI_COMPAT_SYNC_CTL");
    if (ctl && !*ctl) ctl = NULL;
    atomic_store(&g_sd_fbm, fb ? sd_fb_mask(atoi(fb)) : 0);
    if (getenv("MALI_COMPAT_FB_MASK")) atomic_store(&g_sd_fbm, (uint32_t)strtoul(getenv("MALI_COMPAT_FB_MASK"), NULL, 0) & FBM_ALL);
    atomic_store(&g_sd_si_ms, sd_si_parse(si));
    atomic_store(&g_sd_k4, k4 && strcmp(k4, "0") != 0);
    // The hooks are needed when a barrier or submit mode is asked for, or may be switched on later (ctl file);
    // GT_SPLIT_SKIP alone is a flag in the GPU timing code and needs none of them.
    g_sd_armed = ctl || atomic_load(&g_sd_fbm) || atomic_load(&g_sd_si_ms);
    if (!g_sd_armed && !atomic_load(&g_sd_k4)) return;
    g_sd_ctl = ctl;
    if (getenv("MALI_COMPAT_SYNC_REPORT_S") && atoi(getenv("MALI_COMPAT_SYNC_REPORT_S")) > 0)
        g_sd_report_ns = (uint64_t)atoi(getenv("MALI_COMPAT_SYNC_REPORT_S")) * 1000000000ull;
    const char *lp = getenv("MALI_COMPAT_SYNC_LOG");
    if (lp && *lp) snprintf(g_sd_log, sizeof g_sd_log, "%s", lp);
    else {
        char comm[32] = "unknown";
        FILE *f = fopen("/proc/self/comm", "r");
        if (f) { if (fgets(comm, sizeof comm, f)) comm[strcspn(comm, "\n")] = 0; fclose(f); }
        for (char *c = comm; *c; c++) if (*c == '/' || *c == ' ' || *c == '\\') *c = '_';
        snprintf(g_sd_log, sizeof g_sd_log, "/root/mali-sync-diag-%s.txt", comm);
    }
    sd_line("sync-diag armed: hooks %s, ctl %s, log %s", g_sd_armed ? "on" : "off (k4 only)", ctl ? ctl : "none", g_sd_log);
    sd_mode_line("env");
    pthread_t t;
    if (pthread_create(&t, NULL, sd_thread, NULL) == 0) pthread_detach(t);
}
static void sd_init(void) {
    static pthread_once_t once = PTHREAD_ONCE_INIT;
    pthread_once(&once, sd_init_once);
}

static struct SdDev *sd_dev(struct DeviceData *d) { // g_sd_mutex held
    if (d->sd) return d->sd;
    struct SdDev *s = calloc(1, sizeof *s);
    if (!s) return NULL;
    #define SD_DRV(field, T, name) s->field = (T)d->get_device_proc_addr(d->device, name)
    SD_DRV(drvSubmit, PFN_vkQueueSubmit, "vkQueueSubmit");
    SD_DRV(drvWait, PFN_vkWaitForFences, "vkWaitForFences");
    SD_DRV(drvReset, PFN_vkResetFences, "vkResetFences");
    SD_DRV(drvCreate, PFN_vkCreateFence, "vkCreateFence");
    SD_DRV(drvDestroy, PFN_vkDestroyFence, "vkDestroyFence");
    SD_DRV(drvStatus, PFN_vkGetFenceStatus, "vkGetFenceStatus");
    #undef SD_DRV
    d->sd = s;
    return s;
}

static void sd_device_destroy(struct DeviceData *d) {
    if (!d || !d->sd) return;
    pthread_mutex_lock(&g_sd_mutex);
    for (int i = 0; i < SD_FENCES; i++)
        if (g_sd_fences[i].state && g_sd_fences[i].dev == d->device) {
            if (d->sd->drvDestroy) d->sd->drvDestroy(d->device, g_sd_fences[i].f, NULL);
            memset(&g_sd_fences[i], 0, sizeof g_sd_fences[i]);
        }
    free(d->sd);
    d->sd = NULL;
    pthread_mutex_unlock(&g_sd_mutex);
}

// One full barrier, outside a render pass instance only (callers check).
static void sd_full(struct DeviceData *d, VkCommandBuffer cmd) {
    VkMemoryBarrier mb = {VK_STRUCTURE_TYPE_MEMORY_BARRIER, NULL, SD_ACC, SD_ACC};
    d->fpCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 1, &mb, 0,
                            NULL, 0, NULL);
    SD_CNT(SDC_EXTRA);
}
// After an action command outside a pass (FULL_BARRIERS=2 or its class bit in fbm).
static void sd_post(struct DeviceData *d, VkCommandBuffer cmd, uint32_t bit) {
    if (!SD_FB(bit)) return;
    struct CmdBufData *c = cmd_data(cmd);
    if (c && !c->sd_pass) sd_full(d, cmd);
}
static uint32_t sd_kbit(int k) {
    return k == SD_D || k == SD_DI || k == SD_DB ? FBM_DISP : k == SD_CLC || k == SD_CLD ? FBM_CLEAR
         : k == SD_CQP ? FBM_QCOPY : FBM_XFER;
}
#define SD_NEXT(d, k, T) ((T)(d)->sd->next[k])

// Stage histogram of the app's barriers outside passes (sync2 bit values; the v1 bits are the same low bits).
#define SD_ST_TESS (VK_PIPELINE_STAGE_2_TESSELLATION_CONTROL_SHADER_BIT | VK_PIPELINE_STAGE_2_TESSELLATION_EVALUATION_SHADER_BIT | \
                    VK_PIPELINE_STAGE_2_GEOMETRY_SHADER_BIT | VK_PIPELINE_STAGE_2_PRE_RASTERIZATION_SHADERS_BIT)
#define SD_ST_VTX (VK_PIPELINE_STAGE_2_VERTEX_INPUT_BIT | VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT | \
                   VK_PIPELINE_STAGE_2_INDEX_INPUT_BIT | VK_PIPELINE_STAGE_2_VERTEX_ATTRIBUTE_INPUT_BIT)
#define SD_ST_FRAG (VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT | \
                    VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT)
#define SD_ST_XFER (VK_PIPELINE_STAGE_2_TRANSFER_BIT | VK_PIPELINE_STAGE_2_COPY_BIT | VK_PIPELINE_STAGE_2_BLIT_BIT | \
                    VK_PIPELINE_STAGE_2_RESOLVE_BIT | VK_PIPELINE_STAGE_2_CLEAR_BIT)
static void sd_hist(uint64_t src, uint64_t dst) {
    SD_CNT(SDC_H_BARRIERS);
    if (dst & SD_ST_TESS) SD_CNT(SDC_H_DTESS);
    if (dst & VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT) SD_CNT(SDC_H_DIND);
    if (dst & SD_ST_VTX) SD_CNT(SDC_H_DVTX);
    if (dst & SD_ST_FRAG) SD_CNT(SDC_H_DFRAG);
    if (dst & VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT) SD_CNT(SDC_H_DCS);
    if (dst & SD_ST_XFER) SD_CNT(SDC_H_DXFER);
    if (dst & (VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT | VK_PIPELINE_STAGE_2_ALL_GRAPHICS_BIT)) SD_CNT(SDC_H_DALL);
    if (src & VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT) SD_CNT(SDC_H_SCS);
    if (src & SD_ST_XFER) SD_CNT(SDC_H_SXFER);
    if (src & SD_ST_FRAG) SD_CNT(SDC_H_SFRAG);
}

static VKAPI_ATTR void VKAPI_CALL sd_CmdPipelineBarrier(VkCommandBuffer cmd, VkPipelineStageFlags src, VkPipelineStageFlags dst,
                                                        VkDependencyFlags dep, uint32_t nm, const VkMemoryBarrier *m,
                                                        uint32_t nb, const VkBufferMemoryBarrier *b, uint32_t ni,
                                                        const VkImageMemoryBarrier *im) {
    struct DeviceData *d = get_device_data_by_cmd(cmd);
    PFN_vkCmdPipelineBarrier next = SD_NEXT(d, SD_PB, PFN_vkCmdPipelineBarrier);
    struct CmdBufData *c = cmd_data(cmd);
    uint32_t w = SD_FB(FBM_DST | FBM_SRC | FBM_ACC);
    if (c && !c->sd_pass) sd_hist(src, dst);
    if (!c || c->sd_pass || !w) {
        if (c && c->sd_pass && w) SD_CNT(SDC_INPASS);
        next(cmd, src, dst, dep, nm, m, nb, b, ni, im);
        return;
    }
    VkPipelineStageFlags s2 = src | ((w & FBM_SRC) ? VK_PIPELINE_STAGE_ALL_COMMANDS_BIT : 0);
    VkPipelineStageFlags d2 = dst | ((w & FBM_DST) ? VK_PIPELINE_STAGE_ALL_COMMANDS_BIT : 0);
    SD_CNT(SDC_WIDEN);
    if (!(w & FBM_ACC)) { next(cmd, s2, d2, dep, nm, m, nb, b, ni, im); return; } // stages only
    VkMemoryBarrier ms[33], *mw = nm < 33 ? ms : malloc((nm + 1) * sizeof *mw);
    VkBufferMemoryBarrier bs[32], *bw = nb <= 32 ? bs : malloc(nb * sizeof *bw);
    VkImageMemoryBarrier is[32], *iw = ni <= 32 ? is : malloc(ni * sizeof *iw);
    if (!mw || !bw || !iw) { // out of memory: the stage widening only
        if (mw != ms) free(mw);
        if (bw != bs) free(bw);
        if (iw != is) free(iw);
        next(cmd, s2, d2, dep, nm, m, nb, b, ni, im);
        return;
    }
    for (uint32_t i = 0; i < nm; i++) { mw[i] = m[i]; mw[i].srcAccessMask |= SD_ACC; mw[i].dstAccessMask |= SD_ACC; }
    mw[nm] = (VkMemoryBarrier){VK_STRUCTURE_TYPE_MEMORY_BARRIER, NULL, SD_ACC, SD_ACC};
    for (uint32_t i = 0; i < nb; i++) { bw[i] = b[i]; bw[i].srcAccessMask |= SD_ACC; bw[i].dstAccessMask |= SD_ACC; }
    for (uint32_t i = 0; i < ni; i++) { iw[i] = im[i]; iw[i].srcAccessMask |= SD_ACC; iw[i].dstAccessMask |= SD_ACC; }
    next(cmd, s2, d2, dep & ~VK_DEPENDENCY_BY_REGION_BIT, nm + 1, mw, nb, bw, ni, iw);
    if (mw != ms) free(mw);
    if (bw != bs) free(bw);
    if (iw != is) free(iw);
}

static VKAPI_ATTR void VKAPI_CALL sd_CmdPipelineBarrier2(VkCommandBuffer cmd, const VkDependencyInfo *di) {
    struct DeviceData *d = get_device_data_by_cmd(cmd);
    PFN_vkCmdPipelineBarrier2 next = SD_NEXT(d, SD_PB2, PFN_vkCmdPipelineBarrier2);
    struct CmdBufData *c = cmd_data(cmd);
    uint32_t w = SD_FB(FBM_DST | FBM_SRC | FBM_ACC);
    if (c && !c->sd_pass && di) {
        uint64_t hs = 0, hd = 0;
        for (uint32_t i = 0; i < di->memoryBarrierCount; i++) { hs |= di->pMemoryBarriers[i].srcStageMask; hd |= di->pMemoryBarriers[i].dstStageMask; }
        for (uint32_t i = 0; i < di->bufferMemoryBarrierCount; i++) { hs |= di->pBufferMemoryBarriers[i].srcStageMask; hd |= di->pBufferMemoryBarriers[i].dstStageMask; }
        for (uint32_t i = 0; i < di->imageMemoryBarrierCount; i++) { hs |= di->pImageMemoryBarriers[i].srcStageMask; hd |= di->pImageMemoryBarriers[i].dstStageMask; }
        sd_hist(hs, hd);
    }
    if (!c || c->sd_pass || !di || !w) {
        if (c && c->sd_pass && w) SD_CNT(SDC_INPASS);
        next(cmd, di);
        return;
    }
    uint32_t nm = di->memoryBarrierCount, nb = di->bufferMemoryBarrierCount, ni = di->imageMemoryBarrierCount;
    uint32_t extra = (w & FBM_ACC) ? 1 : 0;
    VkMemoryBarrier2 ms[33], *mw = nm < 33 ? ms : malloc((nm + 1) * sizeof *mw);
    VkBufferMemoryBarrier2 bs[32], *bw = nb <= 32 ? bs : malloc(nb * sizeof *bw);
    VkImageMemoryBarrier2 is[32], *iw = ni <= 32 ? is : malloc(ni * sizeof *iw);
    if (!mw || !bw || !iw) {
        if (mw != ms) free(mw);
        if (bw != bs) free(bw);
        if (iw != is) free(iw);
        next(cmd, di);
        return;
    }
    #define SD_W2(x) do { if (w & FBM_SRC) (x).srcStageMask |= VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT; \
                          if (w & FBM_DST) (x).dstStageMask |= VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT; \
                          if (w & FBM_ACC) { (x).srcAccessMask |= SD_ACC2; (x).dstAccessMask |= SD_ACC2; } } while (0)
    for (uint32_t i = 0; i < nm; i++) { mw[i] = di->pMemoryBarriers[i]; SD_W2(mw[i]); }
    if (extra) {
        uint64_t es = (w & FBM_SRC) ? VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT : 0, ed = (w & FBM_DST) ? VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT : 0;
        if (!es || !ed) { // the appended barrier spans what the app's barriers span
            for (uint32_t i = 0; i < nm; i++) { es |= di->pMemoryBarriers[i].srcStageMask; ed |= di->pMemoryBarriers[i].dstStageMask; }
            for (uint32_t i = 0; i < nb; i++) { es |= di->pBufferMemoryBarriers[i].srcStageMask; ed |= di->pBufferMemoryBarriers[i].dstStageMask; }
            for (uint32_t i = 0; i < ni; i++) { es |= di->pImageMemoryBarriers[i].srcStageMask; ed |= di->pImageMemoryBarriers[i].dstStageMask; }
        }
        mw[nm] = (VkMemoryBarrier2){VK_STRUCTURE_TYPE_MEMORY_BARRIER_2, NULL, es, SD_ACC2, ed, SD_ACC2};
    }
    for (uint32_t i = 0; i < nb; i++) { bw[i] = di->pBufferMemoryBarriers[i]; SD_W2(bw[i]); }
    for (uint32_t i = 0; i < ni; i++) { iw[i] = di->pImageMemoryBarriers[i]; SD_W2(iw[i]); }
    #undef SD_W2
    VkDependencyInfo wi = *di;
    if (w & FBM_ACC) wi.dependencyFlags &= ~VK_DEPENDENCY_BY_REGION_BIT;
    wi.memoryBarrierCount = nm + extra; wi.pMemoryBarriers = mw;
    wi.bufferMemoryBarrierCount = nb; wi.pBufferMemoryBarriers = bw;
    wi.imageMemoryBarrierCount = ni; wi.pImageMemoryBarriers = iw;
    next(cmd, &wi);
    SD_CNT(SDC_WIDEN);
    if (mw != ms) free(mw);
    if (bw != bs) free(bw);
    if (iw != is) free(iw);
}

static VKAPI_ATTR void VKAPI_CALL sd_CmdWaitEvents(VkCommandBuffer cmd, uint32_t n, const VkEvent *ev, VkPipelineStageFlags src,
                                                   VkPipelineStageFlags dst, uint32_t nm, const VkMemoryBarrier *m, uint32_t nb,
                                                   const VkBufferMemoryBarrier *b, uint32_t ni, const VkImageMemoryBarrier *im) {
    struct DeviceData *d = get_device_data_by_cmd(cmd);
    SD_NEXT(d, SD_WE, PFN_vkCmdWaitEvents)(cmd, n, ev, src, dst, nm, m, nb, b, ni, im);
    struct CmdBufData *c = SD_FB(FBM_WAITEV) ? cmd_data(cmd) : NULL;
    if (c && !c->sd_pass) sd_full(d, cmd);
}
static VKAPI_ATTR void VKAPI_CALL sd_CmdWaitEvents2(VkCommandBuffer cmd, uint32_t n, const VkEvent *ev, const VkDependencyInfo *di) {
    struct DeviceData *d = get_device_data_by_cmd(cmd);
    SD_NEXT(d, SD_WE2, PFN_vkCmdWaitEvents2)(cmd, n, ev, di);
    struct CmdBufData *c = SD_FB(FBM_WAITEV) ? cmd_data(cmd) : NULL;
    if (c && !c->sd_pass) sd_full(d, cmd);
}

// Pass tracking runs in every mode, so a mode switched on in the middle of a command buffer knows where it is.
static VKAPI_ATTR VkResult VKAPI_CALL sd_BeginCommandBuffer(VkCommandBuffer cmd, const VkCommandBufferBeginInfo *bi) {
    struct DeviceData *d = get_device_data_by_cmd(cmd);
    VkResult r = SD_NEXT(d, SD_BCB, PFN_vkBeginCommandBuffer)(cmd, bi);
    struct CmdBufData *c = cmd_data(cmd);
    if (c) {
        // A secondary that continues a render pass is inside it from the start. No barrier at begin: the command
        // buffer can start with a RESUMING part of a pass suspended in the previous one.
        c->sd_pass = bi && (bi->flags & VK_COMMAND_BUFFER_USAGE_RENDER_PASS_CONTINUE_BIT);
        c->sd_susp = false;
    }
    return r;
}
static VKAPI_ATTR void VKAPI_CALL sd_CmdBeginRendering(VkCommandBuffer cmd, const VkRenderingInfo *ri) {
    struct DeviceData *d = get_device_data_by_cmd(cmd);
    struct CmdBufData *c = cmd_data(cmd);
    // With NO_SUSPEND_RESUME the parts become plain passes below this hook, so barriers around them are legal.
    bool plain = !(ri->flags & (VK_RENDERING_SUSPENDING_BIT | VK_RENDERING_RESUMING_BIT)) ||
                 (g_no_susres && ri->colorAttachmentCount <= 8);
    if (SD_FB(FBM_PRE) && c && !c->sd_pass) {
        if (plain || !(ri->flags & VK_RENDERING_RESUMING_BIT)) sd_full(d, cmd);
        else SD_CNT(SDC_SPLIT);
    }
    SD_NEXT(d, SD_BR, PFN_vkCmdBeginRendering)(cmd, ri);
    if (c) { c->sd_pass = 1; c->sd_susp = !plain && (ri->flags & VK_RENDERING_SUSPENDING_BIT); }
}
static VKAPI_ATTR void VKAPI_CALL sd_CmdEndRendering(VkCommandBuffer cmd) {
    struct DeviceData *d = get_device_data_by_cmd(cmd);
    SD_NEXT(d, SD_ER, PFN_vkCmdEndRendering)(cmd);
    struct CmdBufData *c = cmd_data(cmd);
    if (!c) return;
    c->sd_pass = 0;
    if (SD_FB(FBM_POST)) {
        if (!c->sd_susp) sd_full(d, cmd);
        else SD_CNT(SDC_SPLIT);
    }
    c->sd_susp = false;
}
static void sd_pass_begin(struct DeviceData *d, VkCommandBuffer cmd, struct CmdBufData *c) {
    if (SD_FB(FBM_PRE) && c && !c->sd_pass) sd_full(d, cmd);
}
static void sd_pass_end(struct DeviceData *d, VkCommandBuffer cmd) {
    struct CmdBufData *c = cmd_data(cmd);
    if (!c) return;
    c->sd_pass = 0;
    c->sd_susp = false;
    if (SD_FB(FBM_POST)) sd_full(d, cmd);
}
static VKAPI_ATTR void VKAPI_CALL sd_CmdBeginRenderPass(VkCommandBuffer cmd, const VkRenderPassBeginInfo *bi, VkSubpassContents sc) {
    struct DeviceData *d = get_device_data_by_cmd(cmd);
    struct CmdBufData *c = cmd_data(cmd);
    sd_pass_begin(d, cmd, c);
    SD_NEXT(d, SD_BRP, PFN_vkCmdBeginRenderPass)(cmd, bi, sc);
    if (c) c->sd_pass = 1;
}
static VKAPI_ATTR void VKAPI_CALL sd_CmdBeginRenderPass2(VkCommandBuffer cmd, const VkRenderPassBeginInfo *bi, const VkSubpassBeginInfo *si) {
    struct DeviceData *d = get_device_data_by_cmd(cmd);
    struct CmdBufData *c = cmd_data(cmd);
    sd_pass_begin(d, cmd, c);
    SD_NEXT(d, SD_BRP2, PFN_vkCmdBeginRenderPass2)(cmd, bi, si);
    if (c) c->sd_pass = 1;
}
static VKAPI_ATTR void VKAPI_CALL sd_CmdEndRenderPass(VkCommandBuffer cmd) {
    struct DeviceData *d = get_device_data_by_cmd(cmd);
    SD_NEXT(d, SD_ERP, PFN_vkCmdEndRenderPass)(cmd);
    sd_pass_end(d, cmd);
}
static VKAPI_ATTR void VKAPI_CALL sd_CmdEndRenderPass2(VkCommandBuffer cmd, const VkSubpassEndInfo *ei) {
    struct DeviceData *d = get_device_data_by_cmd(cmd);
    SD_NEXT(d, SD_ERP2, PFN_vkCmdEndRenderPass2)(cmd, ei);
    sd_pass_end(d, cmd);
}

// Action commands outside a pass: the command, then (FULL_BARRIERS=2 / its fbm class bit) a full barrier.
#define SD_POST(fn, k, PARAMS, ARGS) \
    static VKAPI_ATTR void VKAPI_CALL sd_##fn PARAMS { \
        struct DeviceData *d = get_device_data_by_cmd(cmd); \
        SD_NEXT(d, k, PFN_vk##fn) ARGS; \
        sd_post(d, cmd, sd_kbit(k)); \
    }
SD_POST(CmdDispatch, SD_D, (VkCommandBuffer cmd, uint32_t x, uint32_t y, uint32_t z), (cmd, x, y, z))
SD_POST(CmdDispatchIndirect, SD_DI, (VkCommandBuffer cmd, VkBuffer b, VkDeviceSize o), (cmd, b, o))
SD_POST(CmdDispatchBase, SD_DB, (VkCommandBuffer cmd, uint32_t bx, uint32_t by, uint32_t bz, uint32_t x, uint32_t y, uint32_t z),
        (cmd, bx, by, bz, x, y, z))
SD_POST(CmdCopyBuffer, SD_CB, (VkCommandBuffer cmd, VkBuffer s, VkBuffer t, uint32_t n, const VkBufferCopy *r), (cmd, s, t, n, r))
SD_POST(CmdCopyBuffer2, SD_CB2, (VkCommandBuffer cmd, const VkCopyBufferInfo2 *i), (cmd, i))
SD_POST(CmdCopyImage, SD_CI, (VkCommandBuffer cmd, VkImage s, VkImageLayout sl, VkImage t, VkImageLayout tl, uint32_t n,
        const VkImageCopy *r), (cmd, s, sl, t, tl, n, r))
SD_POST(CmdCopyImage2, SD_CI2, (VkCommandBuffer cmd, const VkCopyImageInfo2 *i), (cmd, i))
SD_POST(CmdCopyBufferToImage, SD_CBI, (VkCommandBuffer cmd, VkBuffer s, VkImage t, VkImageLayout tl, uint32_t n,
        const VkBufferImageCopy *r), (cmd, s, t, tl, n, r))
SD_POST(CmdCopyBufferToImage2, SD_CBI2, (VkCommandBuffer cmd, const VkCopyBufferToImageInfo2 *i), (cmd, i))
SD_POST(CmdCopyImageToBuffer, SD_CIB, (VkCommandBuffer cmd, VkImage s, VkImageLayout sl, VkBuffer t, uint32_t n,
        const VkBufferImageCopy *r), (cmd, s, sl, t, n, r))
SD_POST(CmdCopyImageToBuffer2, SD_CIB2, (VkCommandBuffer cmd, const VkCopyImageToBufferInfo2 *i), (cmd, i))
SD_POST(CmdBlitImage, SD_BL, (VkCommandBuffer cmd, VkImage s, VkImageLayout sl, VkImage t, VkImageLayout tl, uint32_t n,
        const VkImageBlit *r, VkFilter f), (cmd, s, sl, t, tl, n, r, f))
SD_POST(CmdBlitImage2, SD_BL2, (VkCommandBuffer cmd, const VkBlitImageInfo2 *i), (cmd, i))
SD_POST(CmdResolveImage, SD_RS, (VkCommandBuffer cmd, VkImage s, VkImageLayout sl, VkImage t, VkImageLayout tl, uint32_t n,
        const VkImageResolve *r), (cmd, s, sl, t, tl, n, r))
SD_POST(CmdResolveImage2, SD_RS2, (VkCommandBuffer cmd, const VkResolveImageInfo2 *i), (cmd, i))
SD_POST(CmdFillBuffer, SD_FILL, (VkCommandBuffer cmd, VkBuffer b, VkDeviceSize o, VkDeviceSize s, uint32_t v), (cmd, b, o, s, v))
SD_POST(CmdUpdateBuffer, SD_UPD, (VkCommandBuffer cmd, VkBuffer b, VkDeviceSize o, VkDeviceSize s, const void *p), (cmd, b, o, s, p))
SD_POST(CmdClearColorImage, SD_CLC, (VkCommandBuffer cmd, VkImage im, VkImageLayout l, const VkClearColorValue *v, uint32_t n,
        const VkImageSubresourceRange *r), (cmd, im, l, v, n, r))
SD_POST(CmdClearDepthStencilImage, SD_CLD, (VkCommandBuffer cmd, VkImage im, VkImageLayout l, const VkClearDepthStencilValue *v,
        uint32_t n, const VkImageSubresourceRange *r), (cmd, im, l, v, n, r))
SD_POST(CmdCopyQueryPoolResults, SD_CQP, (VkCommandBuffer cmd, VkQueryPool p, uint32_t f, uint32_t n, VkBuffer b, VkDeviceSize o,
        VkDeviceSize s, VkQueryResultFlags fl), (cmd, p, f, n, b, o, s, fl))
#undef SD_POST

// SUBMIT_IDLE: a fence per (device, thread); vkd3d submits each queue from its own thread.
static void sd_idle(struct DeviceData *d, VkQueue q) {
    uint32_t ms = atomic_load_explicit(&g_sd_si_ms, memory_order_relaxed);
    struct SdDev *s = d->sd;
    if (!ms || !s || !s->drvSubmit || !s->drvWait || !s->drvReset || !s->drvCreate || !s->drvStatus) return;
    pid_t tid = (pid_t)syscall(SYS_gettid);
    struct SdFence *slot = NULL;
    pthread_mutex_lock(&g_sd_mutex);
    for (int i = 0; i < SD_FENCES && !slot; i++)
        if (g_sd_fences[i].state == 1 && g_sd_fences[i].dev == d->device && g_sd_fences[i].tid == tid) slot = &g_sd_fences[i];
    for (int i = 0; i < SD_FENCES && !slot; i++) // a fence left behind earlier that has signaled since
        if (g_sd_fences[i].state == 2 && g_sd_fences[i].dev == d->device && g_sd_fences[i].tid == tid &&
            s->drvStatus(d->device, g_sd_fences[i].f) == VK_SUCCESS && s->drvReset(d->device, 1, &g_sd_fences[i].f) == VK_SUCCESS) {
            slot = &g_sd_fences[i];
            slot->state = 1;
        }
    for (int i = 0; i < SD_FENCES && !slot; i++)
        if (!g_sd_fences[i].state) {
            VkFenceCreateInfo fi = {VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
            if (s->drvCreate(d->device, &fi, NULL, &g_sd_fences[i].f) != VK_SUCCESS) break;
            g_sd_fences[i].dev = d->device;
            g_sd_fences[i].tid = tid;
            g_sd_fences[i].state = 1;
            slot = &g_sd_fences[i];
        }
    pthread_mutex_unlock(&g_sd_mutex);
    if (!slot) { SD_CNT(SDC_FAIL); return; }
    VkFence f = slot->f; // only this thread uses its slots
    VkSubmitInfo e = {VK_STRUCTURE_TYPE_SUBMIT_INFO};
    uint64_t t0 = now_ns();
    if (s->drvSubmit(q, 1, &e, f) != VK_SUCCESS) { SD_CNT(SDC_FAIL); return; }
    VkResult w = s->drvWait(d->device, 1, &f, VK_TRUE, (uint64_t)ms * 1000000ull);
    uint64_t dt = now_ns() - t0;
    if (w == VK_SUCCESS) {
        s->drvReset(d->device, 1, &f);
        SD_CNT(SDC_IDLED);
        atomic_fetch_add_explicit(&g_sd_c[SDC_IDLE_NS], dt, memory_order_relaxed);
        uint64_t mx = atomic_load_explicit(&g_sd_idle_max_ns, memory_order_relaxed);
        while (dt > mx && !atomic_compare_exchange_weak(&g_sd_idle_max_ns, &mx, dt)) {}
    } else {
        pthread_mutex_lock(&g_sd_mutex);
        slot->state = 2; // still pending: reused once it has signaled, never reset while in flight
        pthread_mutex_unlock(&g_sd_mutex);
        SD_CNT(w == VK_TIMEOUT ? SDC_TIMEOUT : SDC_FAIL);
    }
}
static VKAPI_ATTR VkResult VKAPI_CALL sd_QueueSubmit(VkQueue q, uint32_t n, const VkSubmitInfo *si, VkFence f) {
    struct DeviceData *d = get_device_data_by_cmd((VkCommandBuffer)q);
    VkResult r = SD_NEXT(d, SD_QS, PFN_vkQueueSubmit)(q, n, si, f);
    SD_CNT(SDC_SUBMITS);
    if (r == VK_SUCCESS) sd_idle(d, q);
    return r;
}
static VKAPI_ATTR VkResult VKAPI_CALL sd_QueueSubmit2(VkQueue q, uint32_t n, const VkSubmitInfo2 *si, VkFence f) {
    struct DeviceData *d = get_device_data_by_cmd((VkCommandBuffer)q);
    VkResult r = SD_NEXT(d, SD_QS2, PFN_vkQueueSubmit2)(q, n, si, f);
    SD_CNT(SDC_SUBMITS);
    if (r == VK_SUCCESS) sd_idle(d, q);
    return r;
}

static const struct { const char *name; uint8_t kind; PFN_vkVoidFunction fn; } g_sd_names[] = {
    {"vkCmdPipelineBarrier", SD_PB, (PFN_vkVoidFunction)sd_CmdPipelineBarrier},
    {"vkCmdPipelineBarrier2", SD_PB2, (PFN_vkVoidFunction)sd_CmdPipelineBarrier2},
    {"vkCmdPipelineBarrier2KHR", SD_PB2, (PFN_vkVoidFunction)sd_CmdPipelineBarrier2},
    {"vkCmdWaitEvents", SD_WE, (PFN_vkVoidFunction)sd_CmdWaitEvents},
    {"vkCmdWaitEvents2", SD_WE2, (PFN_vkVoidFunction)sd_CmdWaitEvents2},
    {"vkCmdWaitEvents2KHR", SD_WE2, (PFN_vkVoidFunction)sd_CmdWaitEvents2},
    {"vkBeginCommandBuffer", SD_BCB, (PFN_vkVoidFunction)sd_BeginCommandBuffer},
    {"vkCmdBeginRendering", SD_BR, (PFN_vkVoidFunction)sd_CmdBeginRendering},
    {"vkCmdBeginRenderingKHR", SD_BR, (PFN_vkVoidFunction)sd_CmdBeginRendering},
    {"vkCmdEndRendering", SD_ER, (PFN_vkVoidFunction)sd_CmdEndRendering},
    {"vkCmdEndRenderingKHR", SD_ER, (PFN_vkVoidFunction)sd_CmdEndRendering},
    {"vkCmdBeginRenderPass", SD_BRP, (PFN_vkVoidFunction)sd_CmdBeginRenderPass},
    {"vkCmdBeginRenderPass2", SD_BRP2, (PFN_vkVoidFunction)sd_CmdBeginRenderPass2},
    {"vkCmdBeginRenderPass2KHR", SD_BRP2, (PFN_vkVoidFunction)sd_CmdBeginRenderPass2},
    {"vkCmdEndRenderPass", SD_ERP, (PFN_vkVoidFunction)sd_CmdEndRenderPass},
    {"vkCmdEndRenderPass2", SD_ERP2, (PFN_vkVoidFunction)sd_CmdEndRenderPass2},
    {"vkCmdEndRenderPass2KHR", SD_ERP2, (PFN_vkVoidFunction)sd_CmdEndRenderPass2},
    {"vkCmdDispatch", SD_D, (PFN_vkVoidFunction)sd_CmdDispatch},
    {"vkCmdDispatchIndirect", SD_DI, (PFN_vkVoidFunction)sd_CmdDispatchIndirect},
    {"vkCmdDispatchBase", SD_DB, (PFN_vkVoidFunction)sd_CmdDispatchBase},
    {"vkCmdDispatchBaseKHR", SD_DB, (PFN_vkVoidFunction)sd_CmdDispatchBase},
    {"vkCmdCopyBuffer", SD_CB, (PFN_vkVoidFunction)sd_CmdCopyBuffer},
    {"vkCmdCopyBuffer2", SD_CB2, (PFN_vkVoidFunction)sd_CmdCopyBuffer2},
    {"vkCmdCopyBuffer2KHR", SD_CB2, (PFN_vkVoidFunction)sd_CmdCopyBuffer2},
    {"vkCmdCopyImage", SD_CI, (PFN_vkVoidFunction)sd_CmdCopyImage},
    {"vkCmdCopyImage2", SD_CI2, (PFN_vkVoidFunction)sd_CmdCopyImage2},
    {"vkCmdCopyImage2KHR", SD_CI2, (PFN_vkVoidFunction)sd_CmdCopyImage2},
    {"vkCmdCopyBufferToImage", SD_CBI, (PFN_vkVoidFunction)sd_CmdCopyBufferToImage},
    {"vkCmdCopyBufferToImage2", SD_CBI2, (PFN_vkVoidFunction)sd_CmdCopyBufferToImage2},
    {"vkCmdCopyBufferToImage2KHR", SD_CBI2, (PFN_vkVoidFunction)sd_CmdCopyBufferToImage2},
    {"vkCmdCopyImageToBuffer", SD_CIB, (PFN_vkVoidFunction)sd_CmdCopyImageToBuffer},
    {"vkCmdCopyImageToBuffer2", SD_CIB2, (PFN_vkVoidFunction)sd_CmdCopyImageToBuffer2},
    {"vkCmdCopyImageToBuffer2KHR", SD_CIB2, (PFN_vkVoidFunction)sd_CmdCopyImageToBuffer2},
    {"vkCmdBlitImage", SD_BL, (PFN_vkVoidFunction)sd_CmdBlitImage},
    {"vkCmdBlitImage2", SD_BL2, (PFN_vkVoidFunction)sd_CmdBlitImage2},
    {"vkCmdBlitImage2KHR", SD_BL2, (PFN_vkVoidFunction)sd_CmdBlitImage2},
    {"vkCmdResolveImage", SD_RS, (PFN_vkVoidFunction)sd_CmdResolveImage},
    {"vkCmdResolveImage2", SD_RS2, (PFN_vkVoidFunction)sd_CmdResolveImage2},
    {"vkCmdResolveImage2KHR", SD_RS2, (PFN_vkVoidFunction)sd_CmdResolveImage2},
    {"vkCmdFillBuffer", SD_FILL, (PFN_vkVoidFunction)sd_CmdFillBuffer},
    {"vkCmdUpdateBuffer", SD_UPD, (PFN_vkVoidFunction)sd_CmdUpdateBuffer},
    {"vkCmdClearColorImage", SD_CLC, (PFN_vkVoidFunction)sd_CmdClearColorImage},
    {"vkCmdClearDepthStencilImage", SD_CLD, (PFN_vkVoidFunction)sd_CmdClearDepthStencilImage},
    {"vkCmdCopyQueryPoolResults", SD_CQP, (PFN_vkVoidFunction)sd_CmdCopyQueryPoolResults},
    {"vkQueueSubmit", SD_QS, (PFN_vkVoidFunction)sd_QueueSubmit},
    {"vkQueueSubmit2", SD_QS2, (PFN_vkVoidFunction)sd_QueueSubmit2},
    {"vkQueueSubmit2KHR", SD_QS2, (PFN_vkVoidFunction)sd_QueueSubmit2},
};

EXPORT VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL vkGetDeviceProcAddr(VkDevice device, const char *pName);
// Called at the top of vkGetDeviceProcAddr while armed: the sync-diag wrapper, with the entry point the rest of
// vkGetDeviceProcAddr hands out (this layer's hook or the driver's) as its next. NULL = not ours, or the name does
// not exist below (then the normal path answers NULL too).
static PFN_vkVoidFunction sd_proc(VkDevice device, const char *pName) {
    if (strncmp(pName, "vkCmd", 5) && strncmp(pName, "vkQueueSubmit", 13) && strcmp(pName, "vkBeginCommandBuffer")) return NULL;
    uint32_t k = 0;
    while (k < sizeof g_sd_names / sizeof g_sd_names[0] && strcmp(g_sd_names[k].name, pName)) k++;
    if (k == sizeof g_sd_names / sizeof g_sd_names[0]) return NULL;
    struct DeviceData *d = get_device_data(device);
    if (!d) return NULL;
    tl_sd_inner = true;
    PFN_vkVoidFunction inner = vkGetDeviceProcAddr(device, pName);
    tl_sd_inner = false;
    if (!inner) return NULL;
    pthread_mutex_lock(&g_sd_mutex);
    struct SdDev *s = sd_dev(d);
    if (s && !s->next[g_sd_names[k].kind]) s->next[g_sd_names[k].kind] = inner; // KHR alias: the same entry point
    pthread_mutex_unlock(&g_sd_mutex);
    return s ? g_sd_names[k].fn : inner;
}

static void gt_render_begin(VkCommandBuffer cmd, uint32_t w, uint32_t h, uint32_t colors, const VkRenderingInfo *ri) {
    struct CmdBufData *c = cmd_data(cmd);
    if (!c || !c->dev->gt) return;
    // K4 (MALI_COMPAT_GT_SPLIT_SKIP / sync-diag k4): no timestamps around suspend/resume parts (invalid there)
    if (ri && (ri->flags & (VK_RENDERING_SUSPENDING_BIT | VK_RENDERING_RESUMING_BIT)) &&
        atomic_load_explicit(&g_sd_k4, memory_order_relaxed)) { c->gt_rec = -1; SD_CNT(SDC_K4); return; }
    c->gt_rec = gt_begin(c, cmd, GT_PASS, w, h, colors, VK_NULL_HANDLE);
    if (c->gt_rec >= 0 && ri) {
        struct GtRec *r = &c->dev->gt->rec[c->gt_rec];
        r->rlayers = ri->layerCount;
        if (ri->colorAttachmentCount && ri->pColorAttachments) r->color0 = ri->pColorAttachments[0].imageView;
        if (ri->pDepthAttachment) r->depth = ri->pDepthAttachment->imageView;
        uint32_t layers = ri->viewMask ? (uint32_t)__builtin_popcount(ri->viewMask) : ri->layerCount;
        for (uint32_t i = 0; ri->pColorAttachments && i < ri->colorAttachmentCount && i < GT_ATT - 1; i++)
            gt_att_add(&r->att, &ri->pColorAttachments[i], w, h, layers, false);
        const VkRenderingAttachmentInfo *ds = ri->pDepthAttachment && ri->pDepthAttachment->imageView ? ri->pDepthAttachment
                                                                                                      : ri->pStencilAttachment;
        if (ds && ds->imageView) gt_att_add(&r->att, ds, w, h, layers, true);
        gt_att_sign(&r->att);
    }
}

static void gt_render_end(VkCommandBuffer cmd) {
    struct CmdBufData *c = cmd_data(cmd);
    if (c && c->dev->gt && c->gt_rec >= 0) { gt_end(c, cmd, c->gt_rec); c->gt_rec = -1; }
}

// MALI_COMPAT_SKIP_HASH=<hex>: draws whose bound graphics pipeline has a shader with this hash are dropped (finds the
// draw that makes a pass slow; the picture is wrong while it is set). [Claude Opus 5.5] 2026-10-30
static bool gt_skip(struct CmdBufData *c) {
    struct HashNode h;
    if (!g_skip_hash || !c || !c->gfx_pipeline || !hash_get(g_pipeline_hash, (uint64_t)c->gfx_pipeline, &h)) return false;
    for (int i = 0; i < 3; i++)
        if (h.hash[i] == g_skip_hash) { atomic_fetch_add(&g_skipped_draws, 1); return true; }
    return false;
}

// Returns true when the draw is to be skipped (gt_skip).
static bool gt_draw(struct CmdBufData *c, uint32_t vcount, uint32_t icount) {
    if (c && c->async_skip) { atomic_fetch_add(&g_async_skipped, 1); return true; }
    if (gt_skip(c)) return true;
    if (g_ct && c) { if (c->ct.in_pass) c->ct.draws++; else c->ct.sdraws++; } // cover-trim arm check
    if (!c || !c->dev->gt) return false;
    if (!g_gpu_time) return false; // the async compiles and the pass clamp share the hooks; no records without the switch
    struct GtCnt one = {c->gfx_flags & PF_EQ ? 1 : 0, c->gfx_flags & PF_DW ? 1 : 0, c->gfx_flags & PF_VSW ? 1 : 0,
                        (uint64_t)vcount * icount};
    if (c->gt_rec < 0) {
        // no pass record in this command buffer: a secondary one (DXVK's tiler mode), see compat_CmdExecuteCommands
        if (!c->gt_sdraws++) {
            c->gt_spipe = c->gfx_pipeline; c->gt_svcount = vcount; c->gt_sicount = icount; c->gt_ssrc = c->gt_src;
        }
        gt_cnt_add(&c->gt_scnt, &one);
        return false;
    }
    struct GtRec *r = &c->dev->gt->rec[c->gt_rec];
    if (!r->draws++) { r->pipeline = c->gfx_pipeline; r->vcount = vcount; r->icount = icount; r->src = c->gt_src; }
    gt_cnt_add(&r->cnt, &one);
    return false;
}

// GPU timing: DXVK's tiler mode records each pass's draws into secondary command buffers and executes them inside the
// pass it begins in the primary, so the pass saw no draws ("draws 0.0 pipe (nil)"). The secondaries' draw counts and
// first draw join the open pass here. [Claude Opus 5.5] 2026-10-30
static VKAPI_ATTR void VKAPI_CALL compat_CmdExecuteCommands(VkCommandBuffer cmd, uint32_t count, const VkCommandBuffer *secs) {
    GTC(GTC_EXECUTE);
    struct CmdBufData *c = cmd_data(cmd);
    struct DeviceData *d = c ? c->dev : get_device_data_by_cmd(cmd);
    for (uint32_t i = 0; g_ct && c && c->ct.in_pass && secs && i < count; i++) { // cover-trim: tiler-mode draws
        struct CmdBufData *s = cmd_data(secs[i]);
        if (s) c->ct.draws += s->ct.sdraws;
    }
    if (c && c->gt_rec >= 0 && d->gt) {
        struct GtRec *r = &d->gt->rec[c->gt_rec];
        for (uint32_t i = 0; secs && i < count; i++) {
            struct CmdBufData *s = cmd_data(secs[i]);
            if (!s || !s->gt_sdraws) continue;
            if (!r->draws) { r->pipeline = s->gt_spipe; r->vcount = s->gt_svcount; r->icount = s->gt_sicount; r->src = s->gt_ssrc; }
            r->draws += s->gt_sdraws;
            gt_cnt_add(&r->cnt, &s->gt_scnt);
        }
    }
    d->gt->fpCmdExecuteCommands(cmd, count, secs);
}

// Queue-side CPU time (QStat, GPU timing only). A queue carries its device's dispatch key, so the command buffer
// lookup finds its device too.
static VKAPI_ATTR VkResult VKAPI_CALL compat_QueueSubmit(VkQueue queue, uint32_t n, const VkSubmitInfo *si, VkFence fence) {
    struct DeviceData *d = get_device_data_by_cmd((VkCommandBuffer)queue);
    uint32_t cbs = 0;
    for (uint32_t i = 0; si && i < n; i++) cbs += si[i].commandBufferCount;
    if (g_dc_on) dc_submit(queue, n, si, fence);
    uint64_t t0 = now_ns();
    VkResult r = d->gt->fpQueueSubmit(queue, n, si, fence);
    qstat_add(QS_SUBMIT, t0, cbs);
    if (g_wait_callers && fence) {
        uint32_t waits = 0;
        for (uint32_t i = 0; si && i < n; i++) waits += si[i].waitSemaphoreCount;
        wc_tag((uint64_t)fence, wc_submit_kind(cbs, waits));
    }
    return r;
}
static VKAPI_ATTR VkResult VKAPI_CALL compat_QueueSubmit2(VkQueue queue, uint32_t n, const VkSubmitInfo2 *si, VkFence fence) {
    struct DeviceData *d = get_device_data_by_cmd((VkCommandBuffer)queue);
    uint32_t cbs = 0;
    for (uint32_t i = 0; si && i < n; i++) cbs += si[i].commandBufferInfoCount;
    if (g_dc_on) dc_submit2(queue, n, si, fence);
    uint64_t t0 = now_ns();
    VkResult r = d->gt->fpQueueSubmit2(queue, n, si, fence);
    qstat_add(QS_SUBMIT, t0, cbs);
    if (g_wait_callers && fence) {
        uint32_t waits = 0;
        for (uint32_t i = 0; si && i < n; i++) waits += si[i].waitSemaphoreInfoCount;
        wc_tag((uint64_t)fence, wc_submit_kind(cbs, waits));
    }
    return r;
}

// MALI_COMPAT_MEMLOG: S event every 100th submit, then the GPU-timing hook if that is on, else the driver.
static VKAPI_ATTR VkResult VKAPI_CALL compat_MlQueueSubmit(VkQueue queue, uint32_t n, const VkSubmitInfo *si, VkFence fence) {
    struct DeviceData *d = get_device_data_by_cmd((VkCommandBuffer)queue);
    ml_submit_tick();
    if ((g_gpu_time || g_dc_on) && d->gt && d->gt->fpQueueSubmit) return compat_QueueSubmit(queue, n, si, fence);
    return d->ml_fpQueueSubmit(queue, n, si, fence);
}
static VKAPI_ATTR VkResult VKAPI_CALL compat_MlQueueSubmit2(VkQueue queue, uint32_t n, const VkSubmitInfo2 *si, VkFence fence) {
    struct DeviceData *d = get_device_data_by_cmd((VkCommandBuffer)queue);
    ml_submit_tick();
    if ((g_gpu_time || g_dc_on) && d->gt && d->gt->fpQueueSubmit2) return compat_QueueSubmit2(queue, n, si, fence);
    return d->ml_fpQueueSubmit2(queue, n, si, fence);
}
static VKAPI_ATTR VkResult VKAPI_CALL compat_WaitForFences(VkDevice device, uint32_t n, const VkFence *fences, VkBool32 all,
                                                           uint64_t timeout) {
    struct DeviceData *d = get_device_data(device);
    uint64_t t0 = now_ns();
    VkResult r = d->gt->fpWaitForFences(device, n, fences, all, timeout);
    qstat_add(QS_WAIT, t0, 0);
    if (g_wait_callers) wc_add((uintptr_t)__builtin_return_address(0), n && fences ? (uint64_t)fences[0] : 0, WK_UNTRACKED, t0);
    for (uint32_t i = 0; g_dc_on && r == VK_SUCCESS && fences && (all || n == 1) && i < n; i++)
        if (fences[i]) dc_done(true, (uint64_t)fences[i], 0);
    return r;
}
static VKAPI_ATTR VkResult VKAPI_CALL compat_WaitSemaphores(VkDevice device, const VkSemaphoreWaitInfo *wi, uint64_t timeout) {
    struct DeviceData *d = get_device_data(device);
    uint64_t t0 = now_ns();
    VkResult r = d->gt->fpWaitSemaphores(device, wi, timeout);
    qstat_add(QS_WAIT, t0, 0);
    if (g_wait_callers) wc_add((uintptr_t)__builtin_return_address(0), 0, WK_TIMELINE, t0);
    for (uint32_t i = 0; g_dc_on && r == VK_SUCCESS && wi && wi->pSemaphores && wi->pValues &&
                         (!(wi->flags & VK_SEMAPHORE_WAIT_ANY_BIT) || wi->semaphoreCount == 1) && i < wi->semaphoreCount; i++)
        dc_done(false, (uint64_t)wi->pSemaphores[i], wi->pValues[i]);
    return r;
}
static VKAPI_ATTR VkResult VKAPI_CALL compat_QueueWaitIdle(VkQueue queue) {
    struct DeviceData *d = get_device_data_by_cmd((VkCommandBuffer)queue);
    uint64_t t0 = now_ns();
    VkResult r = d->gt->fpQueueWaitIdle(queue);
    qstat_add(QS_IDLE, t0, 0);
    return r;
}
static VKAPI_ATTR VkResult VKAPI_CALL compat_DeviceWaitIdle(VkDevice device) {
    struct DeviceData *d = get_device_data(device);
    uint64_t t0 = now_ns();
    VkResult r = d->gt->fpDeviceWaitIdle(device);
    qstat_add(QS_IDLE, t0, 0);
    return r;
}

static void pushdesc_record(struct PushDescRec *r, VkPipelineLayout layout, uint32_t set, uint32_t count,
                            const VkWriteDescriptorSet *w) {
    if (r->layout != layout || r->set != set) { r->layout = layout; r->set = set; r->count = 0; r->bad = false; }
    for (uint32_t i = 0; i < count && !r->bad; i++) {
        if (w[i].descriptorCount != 1 || w[i].pNext) { r->bad = true; break; }
        uint32_t k = 0;
        while (k < r->count && !(r->w[k].dstBinding == w[i].dstBinding && r->w[k].dstArrayElement == w[i].dstArrayElement)) k++;
        if (k == PUSHDESC_MAX) { r->bad = true; break; }
        if (k == r->count) r->count++;
        r->w[k] = w[i];
        r->w[k].pBufferInfo = NULL; r->w[k].pImageInfo = NULL; r->w[k].pTexelBufferView = NULL;
        if (w[i].pBufferInfo) { r->buf[k] = w[i].pBufferInfo[0]; r->w[k].pBufferInfo = &r->buf[k]; }
        if (w[i].pImageInfo) { r->img[k] = w[i].pImageInfo[0]; r->w[k].pImageInfo = &r->img[k]; }
        if (w[i].pTexelBufferView) { r->view[k] = w[i].pTexelBufferView[0]; r->w[k].pTexelBufferView = &r->view[k]; }
    }
}

// The image a meta pass reads: vkd3d binds its copy source with a push descriptor (GPU timing details). Compute
// pushes are also recorded for the restore after a BC decode (struct PushDescRec).
static VKAPI_ATTR void VKAPI_CALL compat_CmdPushDescriptorSetKHR(VkCommandBuffer cmd, VkPipelineBindPoint bp, VkPipelineLayout layout,
                                                                 uint32_t set, uint32_t count, const VkWriteDescriptorSet *w) {
    struct CmdBufData *c = cmd_data(cmd);
    for (uint32_t i = 0; c && i < count; i++)
        if ((w[i].descriptorType == VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE || w[i].descriptorType == VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER ||
             w[i].descriptorType == VK_DESCRIPTOR_TYPE_STORAGE_IMAGE || w[i].descriptorType == VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT) &&
            w[i].pImageInfo && w[i].descriptorCount) {
            c->gt_src = w[i].pImageInfo[0].imageView;
            break;
        }
    struct PushDescRec *r = !c ? NULL : bp == VK_PIPELINE_BIND_POINT_COMPUTE ? &c->cpd
                          : bp == VK_PIPELINE_BIND_POINT_GRAPHICS ? &c->gpd : NULL;
    if (g_dc_on) dc_push_writes(c, bp, layout, set, count, w, false);
    if (r && g_vendor_id) pushdesc_record(r, layout, set, count, w);
    // With MALI_COMPAT_VENDOR_ID, vkd3d pushes only the root descriptors that changed and expects the others to stay.
    // Test (2026-10-28): forward the whole merged set every time, in case this driver drops bindings a push leaves out.
    if (g_vendor_id && r && !r->bad && r->count) { count = r->count; w = r->w; }
    (c ? c->dev : get_device_data_by_cmd(cmd))->fpCmdPushDescriptorSetKHR(cmd, bp, layout, set, count, w);
}

// aspect: 0 color, 1 depth, 2 stencil. A read-only layout gets STORE_OP_NONE: STORE there is invalid (no write access),
// and read-only depth is what soft particles (HZD fires) bind while sampling it.
static bool layout_read_only(VkImageLayout l, int aspect) {
    switch (l) {
    case VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL: case VK_IMAGE_LAYOUT_READ_ONLY_OPTIMAL:
    case VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL: return true;
    case VK_IMAGE_LAYOUT_DEPTH_READ_ONLY_OPTIMAL: case VK_IMAGE_LAYOUT_DEPTH_READ_ONLY_STENCIL_ATTACHMENT_OPTIMAL: return aspect == 1;
    case VK_IMAGE_LAYOUT_STENCIL_READ_ONLY_OPTIMAL: case VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_STENCIL_READ_ONLY_OPTIMAL: return aspect == 2;
    default: return false;
    }
}

static void susres_attachment(VkRenderingAttachmentInfo *a, bool resume, bool suspend, int aspect) {
    if (a->imageView == VK_NULL_HANDLE) return;
    if (resume) a->loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
    if (suspend) {
        a->storeOp = layout_read_only(a->imageLayout, aspect) ? VK_ATTACHMENT_STORE_OP_NONE : VK_ATTACHMENT_STORE_OP_STORE;
        a->resolveMode = VK_RESOLVE_MODE_NONE;
        a->resolveImageView = VK_NULL_HANDLE;
    }
}

// SUSRES_BARRIER: called from CmdBeginRendering, so always outside a render pass instance. Masks only for the
// attachment kinds this part has: color write -> color load/draw (read/write); depth/stencil write (draws and STORE) ->
// depth/stencil LOAD/tests (read/write) and fragment shader sampling (read-only depth layouts). Global memory barrier,
// no layout change (resumed parts keep the same views and layouts). The dynamic-rendering local-read rules
// (VK_KHR_dynamic_rendering_local_read) are for barriers inside a pass and do not apply here. Falls back to
// vkCmdPipelineBarrier when synchronization2 is missing. Works across command buffers too (submission order).
static void susres_barrier(struct DeviceData *d, VkCommandBuffer cmd, const VkRenderingInfo *ri) {
    VkPipelineStageFlags2 src = 0, dst = 0;
    VkAccessFlags2 sa = 0, da = 0;
    for (uint32_t i = 0; i < ri->colorAttachmentCount; i++)
        if (ri->pColorAttachments[i].imageView != VK_NULL_HANDLE) {
            src |= VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
            sa |= VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
            dst |= VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
            da |= VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
            break;
        }
    if ((ri->pDepthAttachment && ri->pDepthAttachment->imageView != VK_NULL_HANDLE) ||
        (ri->pStencilAttachment && ri->pStencilAttachment->imageView != VK_NULL_HANDLE)) {
        src |= VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT;
        sa |= VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
        dst |= VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT |
               VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
        da |= VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT |
              VK_ACCESS_2_SHADER_SAMPLED_READ_BIT;
    }
    if (!src) return;
    atomic_fetch_add(&g_susres_barriers, 1);
    if (d->gt && d->gt->fpCmdPipelineBarrier2) {
        VkMemoryBarrier2 mb = {VK_STRUCTURE_TYPE_MEMORY_BARRIER_2, NULL, src, sa, dst, da};
        VkDependencyInfo di = {VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
        di.memoryBarrierCount = 1;
        di.pMemoryBarriers = &mb;
        d->gt->fpCmdPipelineBarrier2(cmd, &di);
    } else if (d->fpCmdPipelineBarrier) { // the used stage/access bits fit the 32-bit flags; SAMPLED_READ -> SHADER_READ
        VkMemoryBarrier mb = {VK_STRUCTURE_TYPE_MEMORY_BARRIER, NULL, (VkAccessFlags)sa,
                              (VkAccessFlags)(da & ~VK_ACCESS_2_SHADER_SAMPLED_READ_BIT) |
                              ((da & VK_ACCESS_2_SHADER_SAMPLED_READ_BIT) ? VK_ACCESS_SHADER_READ_BIT : 0)};
        d->fpCmdPipelineBarrier(cmd, (VkPipelineStageFlags)src, (VkPipelineStageFlags)dst, 0, 1, &mb, 0, NULL, 0, NULL);
    }
}

#include "ct/ct_cover_spv.h"
#include "ct/ct_pred_spv.h"
#include "cover_trim.h"

static VKAPI_ATTR void VKAPI_CALL compat_CmdBeginRendering(VkCommandBuffer cmd, const VkRenderingInfo *ri) {
    GTC(GTC_RENDERING);
    if (ri->flags & VK_RENDERING_SUSPENDING_BIT) atomic_fetch_add(&g_susp_passes, 1);
    if (ri->flags & VK_RENDERING_RESUMING_BIT) atomic_fetch_add(&g_resu_passes, 1);
    VkRenderingInfo plain;
    VkRenderingAttachmentInfo col[8], dep, ste;
    if (g_no_susres && (ri->flags & (VK_RENDERING_SUSPENDING_BIT | VK_RENDERING_RESUMING_BIT)) && ri->colorAttachmentCount <= 8) {
        bool resume = ri->flags & VK_RENDERING_RESUMING_BIT, suspend = ri->flags & VK_RENDERING_SUSPENDING_BIT;
        plain = *ri;
        plain.flags &= ~(VK_RENDERING_SUSPENDING_BIT | VK_RENDERING_RESUMING_BIT);
        for (uint32_t i = 0; i < ri->colorAttachmentCount; i++) { col[i] = ri->pColorAttachments[i]; susres_attachment(&col[i], resume, suspend, 0); }
        plain.pColorAttachments = col;
        if (ri->pDepthAttachment) { dep = *ri->pDepthAttachment; susres_attachment(&dep, resume, suspend, 1); plain.pDepthAttachment = &dep; }
        if (ri->pStencilAttachment) { ste = *ri->pStencilAttachment; susres_attachment(&ste, resume, suspend, 2); plain.pStencilAttachment = &ste; }
        if (resume && g_susres_barrier) susres_barrier(get_device_data_by_cmd(cmd), cmd, ri); // K-CS1
        ri = &plain;
    }
    // Attachment-less passes (MALI_COMPAT_NOATT_AREA=<W>x<H>): DXVK opens them for draws without render targets
    // (stream output, UAV-only) with the device's maximum size, 16384x16384 and 256 layers here. On the vendor driver
    // such a pass took ~118 ms with nothing to draw and finally hung the GPU (Witcher 3 next-gen DX11, 2026-10-30).
    // The area is clamped and the layer count set to 1; with no attachments only UAV writes outside it could be lost.
    VkRenderingInfo clamped;
    if (g_noatt_w && !ri->colorAttachmentCount && !ri->pDepthAttachment && !ri->pStencilAttachment &&
        (ri->renderArea.extent.width > g_noatt_w || ri->renderArea.extent.height > g_noatt_h || ri->layerCount > 1)) {
        clamped = *ri;
        if (clamped.renderArea.extent.width > g_noatt_w) clamped.renderArea.extent.width = g_noatt_w;
        if (clamped.renderArea.extent.height > g_noatt_h) clamped.renderArea.extent.height = g_noatt_h;
        if (!clamped.viewMask) clamped.layerCount = 1;
        atomic_fetch_add(&g_noatt_clamped, 1);
        ri = &clamped;
    }
    gt_render_begin(cmd, ri->renderArea.extent.width, ri->renderArea.extent.height, ri->colorAttachmentCount, ri);
    if (g_ct) ct_pass_begin(cmd_data(cmd), ri);
    get_device_data_by_cmd(cmd)->gt->fpCmdBeginRendering(cmd, ri);
}
static VKAPI_ATTR void VKAPI_CALL compat_CmdEndRendering(VkCommandBuffer cmd) {
    get_device_data_by_cmd(cmd)->gt->fpCmdEndRendering(cmd);
    gt_render_end(cmd);
    if (g_ct) ct_pass_end(cmd_data(cmd), cmd);
}
static VKAPI_ATTR void VKAPI_CALL compat_CmdBeginRenderPass(VkCommandBuffer cmd, const VkRenderPassBeginInfo *bi,
                                                            VkSubpassContents contents) {
    GTC(GTC_RENDERPASS);
    gt_render_begin(cmd, bi->renderArea.extent.width, bi->renderArea.extent.height, 0, NULL);
    get_device_data_by_cmd(cmd)->gt->fpCmdBeginRenderPass(cmd, bi, contents);
}
static VKAPI_ATTR void VKAPI_CALL compat_CmdEndRenderPass(VkCommandBuffer cmd) {
    get_device_data_by_cmd(cmd)->gt->fpCmdEndRenderPass(cmd);
    gt_render_end(cmd);
}
static VKAPI_ATTR void VKAPI_CALL compat_CmdBeginRenderPass2(VkCommandBuffer cmd, const VkRenderPassBeginInfo *bi,
                                                             const VkSubpassBeginInfo *si) {
    gt_render_begin(cmd, bi->renderArea.extent.width, bi->renderArea.extent.height, 0, NULL);
    get_device_data_by_cmd(cmd)->gt->fpCmdBeginRenderPass2(cmd, bi, si);
}
static VKAPI_ATTR void VKAPI_CALL compat_CmdEndRenderPass2(VkCommandBuffer cmd, const VkSubpassEndInfo *ei) {
    get_device_data_by_cmd(cmd)->gt->fpCmdEndRenderPass2(cmd, ei);
    gt_render_end(cmd);
}
static VKAPI_ATTR void VKAPI_CALL compat_CmdDispatch(VkCommandBuffer cmd, uint32_t x, uint32_t y, uint32_t z) {
    GTC(GTC_DISPATCH);
    struct CmdBufData *c = cmd_data(cmd);
    if (!c) { get_device_data_by_cmd(cmd)->fpCmdDispatch(cmd, x, y, z); return; }
    if (g_dc_on) dc_dispatch(c, cmd, x, y, z);
    int32_t i = gt_begin(c, cmd, GT_DISPATCH, x, y, z, c->compute_pipeline); // "WxH cZ" = the group counts
    if (!(g_ct && ct_dispatch(c, cmd, x, y, z))) c->dev->fpCmdDispatch(cmd, x, y, z); // cover-trim: one record, same WxH
    gt_end(c, cmd, i);
}
static VKAPI_ATTR void VKAPI_CALL compat_CmdDispatchIndirect(VkCommandBuffer cmd, VkBuffer buf, VkDeviceSize off) {
    GTC(GTC_DISPATCH_INDIRECT);
    struct CmdBufData *c = cmd_data(cmd);
    if (!c) { get_device_data_by_cmd(cmd)->gt->fpCmdDispatchIndirect(cmd, buf, off); return; }
    if (g_dc_on) dc_dispatch(c, cmd, 0, 0, 0); // indirect: group counts 0 0 0
    int32_t i = gt_begin(c, cmd, GT_DISPATCH, 0, 0, 0, c->compute_pipeline); // indirect: group counts unknown (0x0)
    c->dev->gt->fpCmdDispatchIndirect(cmd, buf, off);
    gt_end(c, cmd, i);
}
static VKAPI_ATTR void VKAPI_CALL compat_CmdDraw(VkCommandBuffer cmd, uint32_t vc, uint32_t ic, uint32_t fv, uint32_t fi) {
    GTC(GTC_DRAW);
    if (fi && ic) atomic_fetch_add(&g_div_draws, 1);
    struct CmdBufData *c = cmd_data(cmd);
    if (gt_draw(c, vc, ic)) return;
    if (g_ct && c && ct_draw(c, cmd, false, vc, ic, fv, 0, fi)) return;
    (c ? c->dev : get_device_data_by_cmd(cmd))->gt->fpCmdDraw(cmd, vc, ic, fv, fi);
}
static VKAPI_ATTR void VKAPI_CALL compat_CmdDrawIndexed(VkCommandBuffer cmd, uint32_t xc, uint32_t ic, uint32_t fx, int32_t vo,
                                                        uint32_t fi) {
    GTC(GTC_DRAW_INDEXED);
    if (fi && ic) atomic_fetch_add(&g_div_draws, 1);
    struct CmdBufData *c = cmd_data(cmd);
    if (gt_draw(c, xc, ic)) return;
    if (g_ct && c && ct_draw(c, cmd, true, xc, ic, fx, vo, fi)) return;
    (c ? c->dev : get_device_data_by_cmd(cmd))->gt->fpCmdDrawIndexed(cmd, xc, ic, fx, vo, fi);
}
static VKAPI_ATTR void VKAPI_CALL compat_CmdDrawIndirect(VkCommandBuffer cmd, VkBuffer b, VkDeviceSize o, uint32_t n, uint32_t s) {
    GTC(GTC_DRAW_INDIRECT);
    struct CmdBufData *c = cmd_data(cmd);
    if (gt_draw(c, 0, 0)) return;
    (c ? c->dev : get_device_data_by_cmd(cmd))->gt->fpCmdDrawIndirect(cmd, b, o, n, s);
}
static VKAPI_ATTR void VKAPI_CALL compat_CmdDrawIndexedIndirect(VkCommandBuffer cmd, VkBuffer b, VkDeviceSize o, uint32_t n,
                                                                uint32_t s) {
    GTC(GTC_DRAW_INDEXED_INDIRECT);
    struct CmdBufData *c = cmd_data(cmd);
    if (gt_draw(c, 0, 0)) return;
    (c ? c->dev : get_device_data_by_cmd(cmd))->gt->fpCmdDrawIndexedIndirect(cmd, b, o, n, s);
}
static VKAPI_ATTR void VKAPI_CALL compat_CmdDrawIndirectCount(VkCommandBuffer cmd, VkBuffer b, VkDeviceSize o, VkBuffer cb,
                                                              VkDeviceSize co, uint32_t n, uint32_t s) {
    GTC(GTC_DRAW_INDIRECT_COUNT);
    struct CmdBufData *c = cmd_data(cmd);
    if (gt_draw(c, 0, 0)) return;
    (c ? c->dev : get_device_data_by_cmd(cmd))->gt->fpCmdDrawIndirectCount(cmd, b, o, cb, co, n, s);
}
static VKAPI_ATTR void VKAPI_CALL compat_CmdDrawIndexedIndirectCount(VkCommandBuffer cmd, VkBuffer b, VkDeviceSize o, VkBuffer cb,
                                                                     VkDeviceSize co, uint32_t n, uint32_t s) {
    GTC(GTC_DRAW_INDEXED_INDIRECT_COUNT);
    struct CmdBufData *c = cmd_data(cmd);
    if (gt_draw(c, 0, 0)) return;
    (c ? c->dev : get_device_data_by_cmd(cmd))->gt->fpCmdDrawIndexedIndirectCount(cmd, b, o, cb, co, n, s);
}

static VKAPI_ATTR void VKAPI_CALL compat_CmdDrawMultiEXT(VkCommandBuffer cmd, uint32_t n, const VkMultiDrawInfoEXT *info,
                                                         uint32_t ic, uint32_t fi, uint32_t stride) {
    GTC(GTC_DRAW_MULTI);
    struct CmdBufData *c = cmd_data(cmd);
    if (gt_draw(c, n && info ? info->vertexCount : 0, ic)) return;
    (c ? c->dev : get_device_data_by_cmd(cmd))->gt->fpCmdDrawMultiEXT(cmd, n, info, ic, fi, stride);
}
static VKAPI_ATTR void VKAPI_CALL compat_CmdDrawMultiIndexedEXT(VkCommandBuffer cmd, uint32_t n, const VkMultiDrawIndexedInfoEXT *info,
                                                                uint32_t ic, uint32_t fi, uint32_t stride, const int32_t *vo) {
    GTC(GTC_DRAW_MULTI_INDEXED);
    struct CmdBufData *c = cmd_data(cmd);
    if (gt_draw(c, n && info ? info->indexCount : 0, ic)) return;
    (c ? c->dev : get_device_data_by_cmd(cmd))->gt->fpCmdDrawMultiIndexedEXT(cmd, n, info, ic, fi, stride, vo);
}
static VKAPI_ATTR void VKAPI_CALL compat_CmdDrawMeshTasksEXT(VkCommandBuffer cmd, uint32_t x, uint32_t y, uint32_t z) {
    GTC(GTC_MESH);
    struct CmdBufData *c = cmd_data(cmd);
    if (gt_draw(c, x, y)) return;
    (c ? c->dev : get_device_data_by_cmd(cmd))->gt->fpCmdDrawMeshTasksEXT(cmd, x, y, z);
}
static VKAPI_ATTR void VKAPI_CALL compat_CmdDrawMeshTasksIndirectEXT(VkCommandBuffer cmd, VkBuffer b, VkDeviceSize o, uint32_t n, uint32_t s) {
    GTC(GTC_MESH_INDIRECT);
    struct CmdBufData *c = cmd_data(cmd);
    if (gt_draw(c, 0, 0)) return;
    (c ? c->dev : get_device_data_by_cmd(cmd))->gt->fpCmdDrawMeshTasksIndirectEXT(cmd, b, o, n, s);
}
static VKAPI_ATTR void VKAPI_CALL compat_CmdDrawMeshTasksIndirectCountEXT(VkCommandBuffer cmd, VkBuffer b, VkDeviceSize o, VkBuffer cb,
                                                                          VkDeviceSize co, uint32_t n, uint32_t s) {
    GTC(GTC_MESH_INDIRECT_COUNT);
    struct CmdBufData *c = cmd_data(cmd);
    if (gt_draw(c, 0, 0)) return;
    (c ? c->dev : get_device_data_by_cmd(cmd))->gt->fpCmdDrawMeshTasksIndirectCountEXT(cmd, b, o, cb, co, n, s);
}
static VKAPI_ATTR void VKAPI_CALL compat_CmdDispatchBase(VkCommandBuffer cmd, uint32_t bx, uint32_t by, uint32_t bz, uint32_t x,
                                                         uint32_t y, uint32_t z) {
    GTC(GTC_DISPATCH_BASE);
    struct CmdBufData *c = cmd_data(cmd);
    if (g_dc_on) dc_dispatch(c, cmd, x, y, z);
    int32_t i = gt_begin(c, cmd, GT_DISPATCH, x, y, z, c ? c->compute_pipeline : VK_NULL_HANDLE);
    (c ? c->dev : get_device_data_by_cmd(cmd))->gt->fpCmdDispatchBase(cmd, bx, by, bz, x, y, z);
    if (c) gt_end(c, cmd, i);
}

// -----------------------------------------------------------------------------
// Layer Core: CreateInstance & CreateDevice
// -----------------------------------------------------------------------------
static VKAPI_ATTR VkResult VKAPI_CALL compat_CreateInstance(
    const VkInstanceCreateInfo *pCreateInfo, const VkAllocationCallbacks *pAllocator,
    VkInstance *pInstance) {
    g_debug = getenv("MALI_COMPAT_DEBUG") != NULL;
    g_stats = getenv("MALI_COMPAT_STATS") && strcmp(getenv("MALI_COMPAT_STATS"), "0") != 0;
    if (g_stats && !strcmp(getenv("MALI_COMPAT_STATS"), "every")) g_stats_interval_ns = 0;
    g_dump_spirv = access("/root/mali-spv", W_OK) == 0;
    g_dump_inflight = access("/root/mali-spv-inflight", W_OK) == 0;
    g_log_pipelines = access("/root/mali-pipelines.txt", W_OK) == 0;
    g_log_images = access("/root/mali-images.txt", W_OK) == 0;
    g_bc_afrc = !getenv("MALI_COMPAT_BC_AFRC") || strcmp(getenv("MALI_COMPAT_BC_AFRC"), "0") != 0;
    bool rt = getenv("MALI_COMPAT_RT_COMPRESS") && strcmp(getenv("MALI_COMPAT_RT_COMPRESS"), "0") != 0;
    if (rt && !g_rt_compress) rt_keep_load(); // once per process: DXVK creates several instances
    g_rt_compress = rt;
    g_gpu_time = getenv("MALI_COMPAT_GPU_TIME") && strcmp(getenv("MALI_COMPAT_GPU_TIME"), "0") != 0;
    g_wait_callers = g_gpu_time && getenv("MALI_COMPAT_WAIT_CALLERS") && strcmp(getenv("MALI_COMPAT_WAIT_CALLERS"), "0") != 0;
    { // K-CF2 port + memlog (stage kcs1+kcf2+memlog); the init line is printed once per process, always
        const char *bmb = getenv("MALI_COMPAT_BUDGET_MB");
        unsigned long long mb = bmb ? strtoull(bmb, NULL, 10) : 0;
        g_budget_accounted = mb != 0;
        g_budget_cap = mb ? (uint64_t)mb << 20 : 6ULL << 30;
        const char *bls = getenv("MALI_COMPAT_BUDGET_LOG_S");
        unsigned long ls = bls ? strtoul(bls, NULL, 10) : 30;
        g_budget_log_s = g_budget_accounted && (g_gpu_time || g_debug) && ls ? (uint32_t)(ls > 3600 ? 3600 : ls) : 0;
        static _Atomic bool ml_started;
        const char *ml = getenv("MALI_COMPAT_MEMLOG");
        static char ml_path[300];
        if (ml && *ml && strcmp(ml, "0") && !atomic_exchange(&ml_started, true)) {
            const char *mmb = getenv("MALI_COMPAT_MEMLOG_MB");
            unsigned long long cap = mmb ? strtoull(mmb, NULL, 10) : 64;
            g_ml_cap = (cap ? cap : 64) << 20;
            g_memlog = !strcmp(ml, "2") ? 2 : 1;
            char comm[32] = "unknown";
            FILE *cf = fopen("/proc/self/comm", "r");
            if (cf) { if (fgets(comm, sizeof comm, cf)) comm[strcspn(comm, "\n")] = 0; fclose(cf); }
            for (char *c = comm; *c; c++) if (*c == '/' || *c == ' ' || *c == '\\') *c = '_';
            snprintf(ml_path, sizeof ml_path, "/root/mali-memlog-%s.tsv", comm);
            ml_open(ml_path, comm);
        }
        static _Atomic bool budget_logged;
        if (!atomic_exchange(&budget_logged, true)) {
            char mls[400] = "off";
            if (g_memlog)
                snprintf(mls, sizeof mls, "on (%s, cap %llu MiB%s)", ml_path, (unsigned long long)(g_ml_cap >> 20),
                         g_memlog == 2 ? ", no capture address" : "");
            if (g_budget_accounted)
                fprintf(stderr, "mali-compat: heapBudget cap %llu MiB (MALI_COMPAT_BUDGET_MB), accounted usage, budget log %s, memlog %s\n",
                        (unsigned long long)(g_budget_cap >> 20), g_budget_log_s ? "on" : "off", mls);
            else
                fprintf(stderr, "mali-compat: heapBudget min(heap, 6144 MiB), cumulative usage (old report, MALI_COMPAT_BUDGET_MB unset), budget log off, memlog %s\n", mls);
            if (g_budget_log_s) fprintf(stderr, "mali-compat: budget log every %u s\n", g_budget_log_s);
        }
    }
    g_depth_nocomp = getenv("MALI_COMPAT_DEPTH_NOCOMP") && strcmp(getenv("MALI_COMPAT_DEPTH_NOCOMP"), "0") != 0;
    g_force_comp = getenv("MALI_COMPAT_FORCE_COMP") && strcmp(getenv("MALI_COMPAT_FORCE_COMP"), "0") != 0;
    g_no_susres = getenv("MALI_COMPAT_NO_SUSPEND_RESUME") && strcmp(getenv("MALI_COMPAT_NO_SUSPEND_RESUME"), "0") != 0;
    g_susres_barrier = g_no_susres && getenv("MALI_COMPAT_SUSRES_BARRIER") && strcmp(getenv("MALI_COMPAT_SUSRES_BARRIER"), "0") != 0;
    g_pcache_on = getenv("MALI_COMPAT_PIPELINE_CACHE") && strcmp(getenv("MALI_COMPAT_PIPELINE_CACHE"), "0") != 0;
    g_pcache_mem_only = g_pcache_on && !strcmp(getenv("MALI_COMPAT_PIPELINE_CACHE"), "2");
    if (getenv("MALI_COMPAT_PIPELINE_CACHE_THREADS")) g_pc_threads = getenv("MALI_COMPAT_PIPELINE_CACHE_THREADS");
    g_force_coherent = getenv("MALI_COMPAT_FORCE_COHERENT") && strcmp(getenv("MALI_COMPAT_FORCE_COHERENT"), "0") != 0;
    g_spv_fix = getenv("MALI_COMPAT_SPV_FIX") && strcmp(getenv("MALI_COMPAT_SPV_FIX"), "0") != 0;
    g_fake_occlusion = getenv("MALI_COMPAT_FAKE_OCCLUSION") && strcmp(getenv("MALI_COMPAT_FAKE_OCCLUSION"), "0") != 0;
    g_vendor_id = getenv("MALI_COMPAT_VENDOR_ID") ? (uint32_t)strtoul(getenv("MALI_COMPAT_VENDOR_ID"), NULL, 16) : 0;
    g_device_id = getenv("MALI_COMPAT_DEVICE_ID") ? (uint32_t)strtoul(getenv("MALI_COMPAT_DEVICE_ID"), NULL, 16) : 0;
    g_device_name = getenv("MALI_COMPAT_DEVICE_NAME");
    // on by default: the write it removes is discarded by the spec anyway (MALI_COMPAT_FRAGDEPTH_FIX=0 turns it off)
    g_fragdepth_fix = !getenv("MALI_COMPAT_FRAGDEPTH_FIX") || strcmp(getenv("MALI_COMPAT_FRAGDEPTH_FIX"), "0") != 0;
    g_gt_trace = getenv("MALI_COMPAT_GT_TRACE") ? (uint32_t)strtoul(getenv("MALI_COMPAT_GT_TRACE"), NULL, 16) : 0;
    if (getenv("MALI_COMPAT_GT_TRACE") && !strcmp(getenv("MALI_COMPAT_GT_TRACE"), "all")) g_gt_trace = UINT32_MAX;
    drm_node_init();
    g_dmabuf_export = getenv("MALI_COMPAT_DMABUF_EXPORT") && strcmp(getenv("MALI_COMPAT_DMABUF_EXPORT"), "0") != 0;
    g_async_ms = getenv("MALI_COMPAT_ASYNC_PSO") ? (uint32_t)atoi(getenv("MALI_COMPAT_ASYNC_PSO")) : 0;
    if (getenv("MALI_COMPAT_ASYNC_THREADS") && atoi(getenv("MALI_COMPAT_ASYNC_THREADS")) > 0)
        g_async_threads = (uint32_t)atoi(getenv("MALI_COMPAT_ASYNC_THREADS"));
    g_spv_dontunroll = getenv("MALI_COMPAT_SPV_DONTUNROLL") && strcmp(getenv("MALI_COMPAT_SPV_DONTUNROLL"), "0") != 0;
    wls_clamp_init(); // MALI_COMPAT_WLS_CLAMP (wls-clamp-1)
    g_xfb_vs = getenv("MALI_COMPAT_XFB_VS") && strcmp(getenv("MALI_COMPAT_XFB_VS"), "0") != 0;
    const char *deq = getenv("MALI_COMPAT_DEPTH_EQUAL");
    g_depth_equal_on = deq && (!strcmp(deq, "always") || !strcmp(deq, "lequal") || !strcmp(deq, "gequal"));
    g_depth_equal_op = !g_depth_equal_on ? VK_COMPARE_OP_EQUAL : !strcmp(deq, "always") ? VK_COMPARE_OP_ALWAYS
                     : !strcmp(deq, "lequal") ? VK_COMPARE_OP_LESS_OR_EQUAL : VK_COMPARE_OP_GREATER_OR_EQUAL;
    if (deq && *deq && strcmp(deq, "0") && !g_depth_equal_on)
        fprintf(stderr, "mali-compat: MALI_COMPAT_DEPTH_EQUAL=%s ignored (always, lequal or gequal)\n", deq);
    g_vs_scan = g_gpu_time || (getenv("MALI_COMPAT_VS_STORES") && strcmp(getenv("MALI_COMPAT_VS_STORES"), "0") != 0);
    g_noatt_w = g_noatt_h = 0;
    if (getenv("MALI_COMPAT_NOATT_AREA") && sscanf(getenv("MALI_COMPAT_NOATT_AREA"), "%ux%u", &g_noatt_w, &g_noatt_h) != 2)
        g_noatt_w = g_noatt_h = 0;
    ct_parse_env();
    g_skip_hash = getenv("MALI_COMPAT_SKIP_HASH") ? (uint32_t)strtoul(getenv("MALI_COMPAT_SKIP_HASH"), NULL, 16) : 0;
    g_dump_hash_count = 0;
    for (const char *h = getenv("MALI_COMPAT_DUMP_HASH"); h && *h && g_dump_hash_count < 16;) {
        char *end;
        unsigned long v = strtoul(h, &end, 16);
        if (end == h) break;
        g_dump_hashes[g_dump_hash_count++] = (uint32_t)v;
        h = *end == ',' ? end + 1 : end;
    }
    dc_init(); // MALI_COMPAT_DESC_CAPTURE (desc-capture-1)

    VkLayerInstanceCreateInfo *chain_info = (VkLayerInstanceCreateInfo *)pCreateInfo->pNext;
    while (chain_info &&
           (chain_info->sType != VK_STRUCTURE_TYPE_LOADER_INSTANCE_CREATE_INFO ||
            chain_info->function != VK_LAYER_LINK_INFO)) {
        chain_info = (VkLayerInstanceCreateInfo *)chain_info->pNext;
    }

    if (!chain_info || !chain_info->u.pLayerInfo) return VK_ERROR_INITIALIZATION_FAILED;

    PFN_vkGetInstanceProcAddr fpGetInstanceProcAddr = chain_info->u.pLayerInfo->pfnNextGetInstanceProcAddr;
    chain_info->u.pLayerInfo = chain_info->u.pLayerInfo->pNext;

    PFN_vkCreateInstance fpCreateInstance = (PFN_vkCreateInstance)fpGetInstanceProcAddr(VK_NULL_HANDLE, "vkCreateInstance");
    if (!fpCreateInstance) return VK_ERROR_INITIALIZATION_FAILED;

    VkResult res = fpCreateInstance(pCreateInfo, pAllocator, pInstance);
    if (res != VK_SUCCESS) return res;

    pthread_mutex_lock(&g_table_mutex);
    if (g_instance_count < MAX_INSTANCES) {
        struct InstanceData *inst = &g_instances[g_instance_count++];
        memset(inst, 0, sizeof(*inst));
        inst->instance = *pInstance;
        inst->dispatch_key = *pInstance ? *(void **)*pInstance : NULL;
        inst->get_instance_proc_addr = fpGetInstanceProcAddr;
        #define LOAD_INST(fn) inst->fp##fn = (PFN_vk##fn)fpGetInstanceProcAddr(*pInstance, "vk" #fn)
        LOAD_INST(DestroyInstance);
        LOAD_INST(EnumeratePhysicalDevices);
        LOAD_INST(GetPhysicalDeviceFeatures);
        LOAD_INST(GetPhysicalDeviceFeatures2);
        if (!inst->fpGetPhysicalDeviceFeatures2)
            inst->fpGetPhysicalDeviceFeatures2 = (PFN_vkGetPhysicalDeviceFeatures2)fpGetInstanceProcAddr(*pInstance, "vkGetPhysicalDeviceFeatures2KHR");
        LOAD_INST(GetPhysicalDeviceFormatProperties);
        LOAD_INST(GetPhysicalDeviceFormatProperties2);
        if (!inst->fpGetPhysicalDeviceFormatProperties2)
            inst->fpGetPhysicalDeviceFormatProperties2 = (PFN_vkGetPhysicalDeviceFormatProperties2)fpGetInstanceProcAddr(*pInstance, "vkGetPhysicalDeviceFormatProperties2KHR");
        LOAD_INST(GetPhysicalDeviceImageFormatProperties);
        LOAD_INST(GetPhysicalDeviceImageFormatProperties2);
        if (!inst->fpGetPhysicalDeviceImageFormatProperties2)
            inst->fpGetPhysicalDeviceImageFormatProperties2 = (PFN_vkGetPhysicalDeviceImageFormatProperties2)fpGetInstanceProcAddr(*pInstance, "vkGetPhysicalDeviceImageFormatProperties2KHR");
        LOAD_INST(GetPhysicalDeviceMemoryProperties);
        LOAD_INST(GetPhysicalDeviceMemoryProperties2);
        if (!inst->fpGetPhysicalDeviceMemoryProperties2)
            inst->fpGetPhysicalDeviceMemoryProperties2 = (PFN_vkGetPhysicalDeviceMemoryProperties2)fpGetInstanceProcAddr(*pInstance, "vkGetPhysicalDeviceMemoryProperties2KHR");
        LOAD_INST(CreateDevice);
        LOAD_INST(EnumerateDeviceExtensionProperties);
        LOAD_INST(GetPhysicalDeviceProperties);
        LOAD_INST(GetPhysicalDeviceProperties2);
        if (!inst->fpGetPhysicalDeviceProperties2)
            inst->fpGetPhysicalDeviceProperties2 = (PFN_vkGetPhysicalDeviceProperties2)fpGetInstanceProcAddr(*pInstance, "vkGetPhysicalDeviceProperties2KHR");
        #undef LOAD_INST
    }
    pthread_mutex_unlock(&g_table_mutex);

    LOG("created VkInstance %p (key=%p)", (void *)*pInstance, *pInstance ? *(void **)*pInstance : NULL);
    return VK_SUCCESS;
}

static VKAPI_ATTR void VKAPI_CALL compat_DestroyInstance(
    VkInstance instance, const VkAllocationCallbacks *pAllocator) {
    struct InstanceData *inst = get_instance_data(instance);
    PFN_vkDestroyInstance fpDestroy = inst ? inst->fpDestroyInstance : NULL;
    if (fpDestroy && instance)
        fpDestroy(instance, pAllocator);

    if (instance) {
        pthread_mutex_lock(&g_table_mutex);
        void *key = *(void **)instance;
        for (uint32_t i = 0; i < g_instance_count; i++) {
            if (g_instances[i].instance == instance || (key && g_instances[i].dispatch_key == key)) {
                for (uint32_t j = i; j + 1 < g_instance_count; j++) {
                    g_instances[j] = g_instances[j + 1];
                }
                g_instance_count--;
                break;
            }
        }
        pthread_mutex_unlock(&g_table_mutex);
    }
}

static VKAPI_ATTR VkResult VKAPI_CALL compat_CreateDevice(
    VkPhysicalDevice physicalDevice, const VkDeviceCreateInfo *pCreateInfo,
    const VkAllocationCallbacks *pAllocator, VkDevice *pDevice) {
    struct InstanceData *inst = get_instance_by_phys_dev(physicalDevice);
    dc_init(); // also here: a device created without this layer's vkCreateInstance (checks) still parses the key
    wls_clamp_init(); // same for MALI_COMPAT_WLS_CLAMP (pthread_once)
    sd_init(); // sync-diag keys (pthread_once)

    VkLayerDeviceCreateInfo *chain_info = (VkLayerDeviceCreateInfo *)pCreateInfo->pNext;
    while (chain_info &&
           (chain_info->sType != VK_STRUCTURE_TYPE_LOADER_DEVICE_CREATE_INFO ||
            chain_info->function != VK_LAYER_LINK_INFO)) {
        chain_info = (VkLayerDeviceCreateInfo *)chain_info->pNext;
    }

    if (!chain_info || !chain_info->u.pLayerInfo) return VK_ERROR_INITIALIZATION_FAILED;

    PFN_vkGetInstanceProcAddr fpGetInstanceProcAddr = chain_info->u.pLayerInfo->pfnNextGetInstanceProcAddr;
    PFN_vkGetDeviceProcAddr fpGetDeviceProcAddr = chain_info->u.pLayerInfo->pfnNextGetDeviceProcAddr;
    chain_info->u.pLayerInfo = chain_info->u.pLayerInfo->pNext;

    PFN_vkCreateDevice fpCreateDevice = (PFN_vkCreateDevice)fpGetInstanceProcAddr(
        inst ? inst->instance : VK_NULL_HANDLE, "vkCreateDevice");
    if (!fpCreateDevice && inst && inst->fpCreateDevice)
        fpCreateDevice = inst->fpCreateDevice;
    if (!fpCreateDevice) return VK_ERROR_INITIALIZATION_FAILED;

    // Mask spoofed features before calling down
    VkDeviceCreateInfo mod_ci = *pCreateInfo;
    // VK_EXT_map_memory_placed is the layer's own: keep its name and feature struct away from the driver.
    const char **ext_names = malloc((pCreateInfo->enabledExtensionCount + 4) * sizeof *ext_names);
    if (!ext_names) return VK_ERROR_OUT_OF_HOST_MEMORY;
    mod_ci.enabledExtensionCount = 0;
    bool div_ext = false, div_khr = false;
    for (uint32_t i = 0; i < pCreateInfo->enabledExtensionCount; i++) {
        const char *e = pCreateInfo->ppEnabledExtensionNames[i];
        if (!strcmp(e, "VK_EXT_vertex_attribute_divisor")) { div_ext = true; continue; } // the vendor has only KHR
        div_khr |= !strcmp(e, "VK_KHR_vertex_attribute_divisor");
        if (g_drm_on && !strcmp(e, VK_EXT_PHYSICAL_DEVICE_DRM_EXTENSION_NAME)) continue; // the layer's (properties only)
        if (strcmp(e, VK_EXT_MAP_MEMORY_PLACED_EXTENSION_NAME)) ext_names[mod_ci.enabledExtensionCount++] = e;
    }
    if (div_ext && !div_khr) ext_names[mod_ci.enabledExtensionCount++] = "VK_KHR_vertex_attribute_divisor";
    // BC shadows get fixed-rate compression (AFRC) and the image log queries compression: both need
    // VK_EXT_image_compression_control and its feature. Enabled when the driver has it. [Claude Opus 5.5]
    bool want_icc = false;
    if ((g_log_images || g_bc_afrc) && inst && inst->fpEnumerateDeviceExtensionProperties) {
        uint32_t n = 0;
        inst->fpEnumerateDeviceExtensionProperties(physicalDevice, NULL, &n, NULL);
        VkExtensionProperties *props = n ? malloc(n * sizeof *props) : NULL;
        if (props && inst->fpEnumerateDeviceExtensionProperties(physicalDevice, NULL, &n, props) >= 0)
            for (uint32_t i = 0; i < n; i++)
                want_icc |= !strcmp(props[i].extensionName, VK_EXT_IMAGE_COMPRESSION_CONTROL_EXTENSION_NAME);
        free(props);
    }
    VkPhysicalDeviceImageCompressionControlFeaturesEXT icc_feat = {
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_IMAGE_COMPRESSION_CONTROL_FEATURES_EXT, NULL, VK_TRUE};
    VkPhysicalDeviceImageCompressionControlFeaturesEXT *app_icc =
        (VkPhysicalDeviceImageCompressionControlFeaturesEXT *)find_struct(pCreateInfo->pNext,
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_IMAGE_COMPRESSION_CONTROL_FEATURES_EXT);
    VkBool32 app_icc_saved = app_icc ? app_icc->imageCompressionControl : VK_FALSE;
    if (want_icc) {
        if (app_icc) app_icc->imageCompressionControl = VK_TRUE;
        else { icc_feat.pNext = (void *)mod_ci.pNext; mod_ci.pNext = &icc_feat; }
    }
    if (want_icc) {
        bool have = false;
        for (uint32_t i = 0; i < mod_ci.enabledExtensionCount; i++)
            have |= !strcmp(ext_names[i], VK_EXT_IMAGE_COMPRESSION_CONTROL_EXTENSION_NAME);
        if (!have) ext_names[mod_ci.enabledExtensionCount++] = VK_EXT_IMAGE_COMPRESSION_CONTROL_EXTENSION_NAME;
    }
    // cover-trim (MALI_COMPAT_COVER_TRIM): VK_EXT_conditional_rendering and its feature for the layer's predicated
    // dispatches and draws, when the driver has them; otherwise the feature stays off.
    bool ct_ext_ok = false;
    VkPhysicalDeviceConditionalRenderingFeaturesEXT ct_feat = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_CONDITIONAL_RENDERING_FEATURES_EXT,
                                                               NULL, VK_TRUE, VK_FALSE};
    VkPhysicalDeviceConditionalRenderingFeaturesEXT *app_ct = (VkPhysicalDeviceConditionalRenderingFeaturesEXT *)find_struct(
        pCreateInfo->pNext, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_CONDITIONAL_RENDERING_FEATURES_EXT);
    VkBool32 app_ct_saved = app_ct ? app_ct->conditionalRendering : VK_FALSE;
    if (g_ct && inst && inst->fpEnumerateDeviceExtensionProperties && inst->fpGetPhysicalDeviceFeatures2) {
        uint32_t n = 0;
        inst->fpEnumerateDeviceExtensionProperties(physicalDevice, NULL, &n, NULL);
        VkExtensionProperties *props = n ? malloc(n * sizeof *props) : NULL;
        if (props && inst->fpEnumerateDeviceExtensionProperties(physicalDevice, NULL, &n, props) >= 0)
            for (uint32_t i = 0; i < n; i++) ct_ext_ok |= !strcmp(props[i].extensionName, VK_EXT_CONDITIONAL_RENDERING_EXTENSION_NAME);
        free(props);
        VkPhysicalDeviceConditionalRenderingFeaturesEXT q = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_CONDITIONAL_RENDERING_FEATURES_EXT};
        VkPhysicalDeviceFeatures2 f2q = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, &q};
        if (ct_ext_ok) inst->fpGetPhysicalDeviceFeatures2(physicalDevice, &f2q);
        ct_ext_ok = ct_ext_ok && q.conditionalRendering;
    }
    if (g_ct && !ct_ext_ok) fprintf(stderr, "mali-compat: cover-trim: no VK_EXT_conditional_rendering on this device; off\n");
    if (ct_ext_ok) {
        bool have = false;
        for (uint32_t i = 0; i < mod_ci.enabledExtensionCount; i++) have |= !strcmp(ext_names[i], VK_EXT_CONDITIONAL_RENDERING_EXTENSION_NAME);
        if (!have) ext_names[mod_ci.enabledExtensionCount++] = VK_EXT_CONDITIONAL_RENDERING_EXTENSION_NAME;
        if (app_ct) app_ct->conditionalRendering = VK_TRUE;
        else { ct_feat.pNext = (void *)mod_ci.pNext; mod_ci.pNext = &ct_feat; }
    }
    mod_ci.ppEnabledExtensionNames = ext_names;
    struct unlinked placed_feat = unlink_struct(&mod_ci, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAP_MEMORY_PLACED_FEATURES_EXT);
    static const char *kFeatureNames[] = {
        "robustBufferAccess", "fullDrawIndexUint32", "imageCubeArray", "independentBlend",
        "geometryShader", "tessellationShader", "sampleRateShading", "dualSrcBlend",
        "logicOp", "multiDrawIndirect", "drawIndirectFirstInstance", "depthClamp",
        "depthBiasClamp", "wideLines", "largePoints", "alphaToOne",
        "multiViewport", "samplerAnisotropy", "textureCompressionETC2", "textureCompressionASTC_LDR",
        "textureCompressionBC", "occlusionQueryPrecise", "pipelineStatisticsQuery", "vertexPipelineStoresAndAtomics",
        "fragmentStoresAndAtomics", "shaderTessellationAndGeometryPointSize", "shaderImageGatherExtended", "shaderStorageImageExtendedFormats",
        "shaderStorageImageMultisample", "shaderStorageImageReadWithoutFormat", "shaderStorageImageWriteWithoutFormat", "shaderUniformBufferArrayDynamicIndexing",
        "shaderSampledImageArrayDynamicIndexing", "shaderStorageBufferArrayDynamicIndexing", "shaderStorageImageArrayDynamicIndexing", "shaderClipDistance",
        "shaderCullDistance", "shaderFloat64", "shaderInt64", "shaderInt16",
        "shaderResourceResidency", "shaderResourceMinLod", "sparseBinding", "sparseResidencyBuffer",
        "sparseResidencyImage2D", "sparseResidencyImage3D", "sparseResidency2Samples", "sparseResidency4Samples",
        "sparseResidency8Samples", "sparseResidency16Samples", "sparseResidencyAliased", "variableMultisampleRate",
        "inheritedQueries"
    };

    VkPhysicalDeviceFeatures real_features = {0};
    if (inst && inst->fpGetPhysicalDeviceFeatures) {
        inst->fpGetPhysicalDeviceFeatures(physicalDevice, &real_features);
    }
    // The real driver's 16/8-bit storage bits: mask only what it lacks. Masking them unconditionally (f7dac02) created the
    // device WITHOUT features the vendor driver has (Mali-G1 54.1: all 7 true) while the app's SPIR-V still declares
    // StorageBuffer16BitAccess etc. (DOOM Eternal, vkd3d-proton). [game/doom_eternal, Claude Opus 5.5] 2026-11-06
    VkPhysicalDevice8BitStorageFeatures rs8 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_8BIT_STORAGE_FEATURES, NULL};
    VkPhysicalDevice16BitStorageFeatures rs16 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES, &rs8};
    if (inst && inst->fpGetPhysicalDeviceFeatures2) {
        VkPhysicalDeviceFeatures2 rsq = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, &rs16};
        inst->fpGetPhysicalDeviceFeatures2(physicalDevice, &rsq);
    }
    fprintf(stderr, "mali-compat: real storage16 %d%d%d%d storage8 %d%d%d (only false bits are masked)\n",
            rs16.storageBuffer16BitAccess, rs16.uniformAndStorageBuffer16BitAccess, rs16.storagePushConstant16,
            rs16.storageInputOutput16, rs8.storageBuffer8BitAccess, rs8.uniformAndStorageBuffer8BitAccess,
            rs8.storagePushConstant8);
    const VkBool32 *real_arr = (const VkBool32 *)&real_features;

    VkPhysicalDeviceFeatures safe_features;
    if (mod_ci.pEnabledFeatures) {
        safe_features = *mod_ci.pEnabledFeatures;
        VkBool32 *safe_arr = (VkBool32 *)&safe_features;
        for (size_t i = 0; i < sizeof(VkPhysicalDeviceFeatures) / sizeof(VkBool32); i++) {
            if (safe_arr[i] && !real_arr[i]) {
                const char *fn = (i < sizeof(kFeatureNames)/sizeof(kFeatureNames[0])) ? kFeatureNames[i] : "?";
                fprintf(stderr, "mali-compat: masking unsupported core feature %s (idx %zu)\n", fn, i);
                safe_arr[i] = VK_FALSE;
            }
        }
        mod_ci.pEnabledFeatures = &safe_features;
    }

    // Mask features2 if chained, and check if bufferDeviceAddress is already in pNext
    VkPhysicalDeviceFeatures saved_f2_features;
    memset(&saved_f2_features, 0, sizeof(saved_f2_features));
    VkPhysicalDeviceFeatures2 *f2 = NULL;
    VkPhysicalDeviceVulkan11Features *v11_feat = NULL;
    VkBool32 saved_v11_sb16 = VK_FALSE, saved_v11_usb16 = VK_FALSE, saved_v11_pc16 = VK_FALSE, saved_v11_io16 = VK_FALSE;
    VkPhysicalDeviceVulkan12Features *v12_feat = NULL;
    VkBool32 saved_v12_bda = VK_FALSE, saved_v12_sb8 = VK_FALSE, saved_v12_usb8 = VK_FALSE, saved_v12_pc8 = VK_FALSE;
    VkPhysicalDevice16BitStorageFeatures *f16 = NULL;
    VkBool32 saved_f16_sb = VK_FALSE, saved_f16_usb = VK_FALSE, saved_f16_pc = VK_FALSE, saved_f16_io = VK_FALSE;
    VkPhysicalDevice8BitStorageFeatures *f8 = NULL;
    VkBool32 saved_f8_sb = VK_FALSE, saved_f8_usb = VK_FALSE, saved_f8_pc = VK_FALSE;
    VkPhysicalDeviceBufferDeviceAddressFeatures *existing_bda = NULL;
    VkBool32 saved_existing_bda = VK_FALSE;

    VkBaseOutStructure *node = (VkBaseOutStructure *)mod_ci.pNext;
    while (node) {
        if (node->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2) {
            f2 = (VkPhysicalDeviceFeatures2 *)node;
            saved_f2_features = f2->features;
            VkBool32 *f2_arr = (VkBool32 *)&f2->features;
            for (size_t i = 0; i < sizeof(VkPhysicalDeviceFeatures) / sizeof(VkBool32); i++) {
                if (f2_arr[i] && !real_arr[i]) {
                    const char *fn = (i < sizeof(kFeatureNames)/sizeof(kFeatureNames[0])) ? kFeatureNames[i] : "?";
                    fprintf(stderr, "mali-compat: masking unsupported f2 feature %s (idx %zu)\n", fn, i);
                    f2_arr[i] = VK_FALSE;
                }
            }
        } else if (node->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES) {
            f16 = (VkPhysicalDevice16BitStorageFeatures *)node;
            saved_f16_sb = f16->storageBuffer16BitAccess;
            saved_f16_usb = f16->uniformAndStorageBuffer16BitAccess;
            saved_f16_pc = f16->storagePushConstant16;
            saved_f16_io = f16->storageInputOutput16;
            if (!rs16.storageBuffer16BitAccess) f16->storageBuffer16BitAccess = VK_FALSE;
            if (!rs16.uniformAndStorageBuffer16BitAccess) f16->uniformAndStorageBuffer16BitAccess = VK_FALSE;
            if (!rs16.storagePushConstant16) f16->storagePushConstant16 = VK_FALSE;
            if (!rs16.storageInputOutput16) f16->storageInputOutput16 = VK_FALSE;
        } else if (node->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_8BIT_STORAGE_FEATURES) {
            f8 = (VkPhysicalDevice8BitStorageFeatures *)node;
            saved_f8_sb = f8->storageBuffer8BitAccess;
            saved_f8_usb = f8->uniformAndStorageBuffer8BitAccess;
            saved_f8_pc = f8->storagePushConstant8;
            if (!rs8.storageBuffer8BitAccess) f8->storageBuffer8BitAccess = VK_FALSE;
            if (!rs8.uniformAndStorageBuffer8BitAccess) f8->uniformAndStorageBuffer8BitAccess = VK_FALSE;
            if (!rs8.storagePushConstant8) f8->storagePushConstant8 = VK_FALSE;
        } else if (node->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES) {
            v11_feat = (VkPhysicalDeviceVulkan11Features *)node;
            saved_v11_sb16 = v11_feat->storageBuffer16BitAccess;
            saved_v11_usb16 = v11_feat->uniformAndStorageBuffer16BitAccess;
            saved_v11_pc16 = v11_feat->storagePushConstant16;
            saved_v11_io16 = v11_feat->storageInputOutput16;
            if (!rs16.storageBuffer16BitAccess) v11_feat->storageBuffer16BitAccess = VK_FALSE;
            if (!rs16.uniformAndStorageBuffer16BitAccess) v11_feat->uniformAndStorageBuffer16BitAccess = VK_FALSE;
            if (!rs16.storagePushConstant16) v11_feat->storagePushConstant16 = VK_FALSE;
            if (!rs16.storageInputOutput16) v11_feat->storageInputOutput16 = VK_FALSE;
        } else if (node->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES) {
            v12_feat = (VkPhysicalDeviceVulkan12Features *)node;
            saved_v12_bda = v12_feat->bufferDeviceAddress;
            v12_feat->bufferDeviceAddress = VK_TRUE;
            saved_v12_sb8 = v12_feat->storageBuffer8BitAccess;
            saved_v12_usb8 = v12_feat->uniformAndStorageBuffer8BitAccess;
            saved_v12_pc8 = v12_feat->storagePushConstant8;
            if (!rs8.storageBuffer8BitAccess) v12_feat->storageBuffer8BitAccess = VK_FALSE;
            if (!rs8.uniformAndStorageBuffer8BitAccess) v12_feat->uniformAndStorageBuffer8BitAccess = VK_FALSE;
            if (!rs8.storagePushConstant8) v12_feat->storagePushConstant8 = VK_FALSE;
        } else if (node->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES) {
            existing_bda = (VkPhysicalDeviceBufferDeviceAddressFeatures *)node;
            saved_existing_bda = existing_bda->bufferDeviceAddress;
            existing_bda->bufferDeviceAddress = VK_TRUE;
        }
        node = node->pNext;
    }

    // Ensure bufferDeviceAddress is requested only if neither Vulkan12Features nor BDA struct was in pNext
    VkPhysicalDeviceBufferDeviceAddressFeatures bda_feat = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES,
        .pNext = (void *)mod_ci.pNext,
        .bufferDeviceAddress = VK_TRUE,
    };
    if (!v12_feat && !existing_bda) {
        mod_ci.pNext = &bda_feat;
    }

    // GPU timing resets its timestamp queries from the collector thread: hostQueryReset (Vulkan 1.2 core).
    VkPhysicalDeviceHostQueryResetFeatures hqr_feat = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_HOST_QUERY_RESET_FEATURES, NULL, VK_TRUE};
    VkPhysicalDeviceHostQueryResetFeatures *app_hqr = (VkPhysicalDeviceHostQueryResetFeatures *)find_struct(pCreateInfo->pNext,
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_HOST_QUERY_RESET_FEATURES);
    VkBool32 saved_v12_hqr = v12_feat ? v12_feat->hostQueryReset : VK_FALSE;
    VkBool32 saved_app_hqr = app_hqr ? app_hqr->hostQueryReset : VK_FALSE;
    if (g_gpu_time) {
        if (v12_feat) v12_feat->hostQueryReset = VK_TRUE;
        else if (app_hqr) app_hqr->hostQueryReset = VK_TRUE;
        else { hqr_feat.pNext = (void *)mod_ci.pNext; mod_ci.pNext = &hqr_feat; }
    }

    VkPhysicalDeviceVertexAttributeDivisorFeaturesKHR *app_div =
        (VkPhysicalDeviceVertexAttributeDivisorFeaturesKHR *)find_struct(pCreateInfo->pNext,
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VERTEX_ATTRIBUTE_DIVISOR_FEATURES_KHR);
    VkBool32 app_zero_div = app_div ? app_div->vertexAttributeInstanceRateZeroDivisor : VK_FALSE;
    bool zero_div_emulated = false;
    if (app_zero_div && inst && inst->fpGetPhysicalDeviceFeatures2) {
        VkPhysicalDeviceVertexAttributeDivisorFeaturesKHR real = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VERTEX_ATTRIBUTE_DIVISOR_FEATURES_KHR};
        VkPhysicalDeviceFeatures2 f2q = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, &real};
        inst->fpGetPhysicalDeviceFeatures2(physicalDevice, &f2q);
        if (!real.vertexAttributeInstanceRateZeroDivisor) {
            app_div->vertexAttributeInstanceRateZeroDivisor = VK_FALSE;
            zero_div_emulated = true;
        }
    }
    VkResult res = fpCreateDevice(physicalDevice, &mod_ci, pAllocator, pDevice);
    if (app_ct) app_ct->conditionalRendering = app_ct_saved;
    if (app_div) app_div->vertexAttributeInstanceRateZeroDivisor = app_zero_div;
    if (app_icc) app_icc->imageCompressionControl = app_icc_saved;
    relink(placed_feat);

    // Restore mutated pNext fields
    if (f2) {
        f2->features = saved_f2_features;
    }
    if (f16) {
        f16->storageBuffer16BitAccess = saved_f16_sb;
        f16->uniformAndStorageBuffer16BitAccess = saved_f16_usb;
        f16->storagePushConstant16 = saved_f16_pc;
        f16->storageInputOutput16 = saved_f16_io;
    }
    if (f8) {
        f8->storageBuffer8BitAccess = saved_f8_sb;
        f8->uniformAndStorageBuffer8BitAccess = saved_f8_usb;
        f8->storagePushConstant8 = saved_f8_pc;
    }
    if (v11_feat) {
        v11_feat->storageBuffer16BitAccess = saved_v11_sb16;
        v11_feat->uniformAndStorageBuffer16BitAccess = saved_v11_usb16;
        v11_feat->storagePushConstant16 = saved_v11_pc16;
        v11_feat->storageInputOutput16 = saved_v11_io16;
    }
    if (v12_feat) {
        v12_feat->bufferDeviceAddress = saved_v12_bda;
        v12_feat->storageBuffer8BitAccess = saved_v12_sb8;
        v12_feat->uniformAndStorageBuffer8BitAccess = saved_v12_usb8;
        v12_feat->storagePushConstant8 = saved_v12_pc8;
    }
    if (existing_bda) existing_bda->bufferDeviceAddress = saved_existing_bda;
    if (v12_feat) v12_feat->hostQueryReset = saved_v12_hqr;
    if (app_hqr) app_hqr->hostQueryReset = saved_app_hqr;

    if (res != VK_SUCCESS) {
        fprintf(stderr, "mali-compat: fpCreateDevice failed: %d (enabledExtensionCount=%u)\n", res, mod_ci.enabledExtensionCount);
        for (uint32_t i = 0; i < mod_ci.enabledExtensionCount; i++) {
            fprintf(stderr, "mali-compat:   requested ext[%u]: %s\n", i, mod_ci.ppEnabledExtensionNames[i]);
        }
        VkBaseOutStructure *curr = (VkBaseOutStructure *)mod_ci.pNext;
        while (curr) {
            fprintf(stderr, "mali-compat:   pNext sType: %d\n", curr->sType);
            curr = curr->pNext;
        }
        fflush(stderr);
        free(ext_names);
        return res;
    }
    free(ext_names);

    pthread_mutex_lock(&g_table_mutex);
    if (g_device_count < MAX_DEVICES) {
        struct DeviceData *d = &g_devices[g_device_count++];
        memset(d, 0, sizeof(*d));
        d->device = *pDevice;
        d->dispatch_key = *pDevice ? *(void **)*pDevice : NULL;
        d->physical_device = physicalDevice;
        d->instance_data = inst;
        d->get_device_proc_addr = fpGetDeviceProcAddr;
        #define LOAD_DEV(fn) d->fp##fn = (PFN_vk##fn)fpGetDeviceProcAddr(*pDevice, "vk" #fn)
        LOAD_DEV(DestroyDevice);
        LOAD_DEV(GetDeviceQueue);
        LOAD_DEV(CreateShaderModule);
        LOAD_DEV(DestroyShaderModule);
        LOAD_DEV(CreateGraphicsPipelines);
        LOAD_DEV(CreateComputePipelines);
        LOAD_DEV(DestroyPipeline);
        LOAD_DEV(CreatePipelineLayout);
        LOAD_DEV(DestroyPipelineLayout);
        LOAD_DEV(CreateImage);
        LOAD_DEV(DestroyImage);
        LOAD_DEV(CreateImageView);
        LOAD_DEV(DestroyImageView);
        LOAD_DEV(CreateBufferView);
        LOAD_DEV(CreatePipelineCache);
        LOAD_DEV(GetPipelineCacheData);
        LOAD_DEV(DestroyPipelineCache);
        LOAD_DEV(CreateQueryPool);
        LOAD_DEV(DestroyQueryPool);
        LOAD_DEV(GetQueryPoolResults);
        LOAD_DEV(CmdCopyQueryPoolResults);
        LOAD_DEV(CmdFillBuffer);
        LOAD_DEV(GetDeviceImageMemoryRequirements);
        if (!d->fpGetDeviceImageMemoryRequirements)
            d->fpGetDeviceImageMemoryRequirements = (PFN_vkGetDeviceImageMemoryRequirements)fpGetDeviceProcAddr(*pDevice, "vkGetDeviceImageMemoryRequirementsKHR");
        LOAD_DEV(GetDeviceImageSparseMemoryRequirements);
        if (!d->fpGetDeviceImageSparseMemoryRequirements)
            d->fpGetDeviceImageSparseMemoryRequirements = (PFN_vkGetDeviceImageSparseMemoryRequirements)fpGetDeviceProcAddr(*pDevice, "vkGetDeviceImageSparseMemoryRequirementsKHR");
        LOAD_DEV(CreateBuffer);
        LOAD_DEV(DestroyBuffer);
        LOAD_DEV(AllocateMemory);
        LOAD_DEV(FreeMemory);
        LOAD_DEV(GetMemoryFdKHR);
        LOAD_DEV(GetMemoryFdPropertiesKHR);
        LOAD_DEV(BindBufferMemory);
        LOAD_DEV(GetBufferMemoryRequirements);
        LOAD_DEV(GetBufferDeviceAddress);
        if (!d->fpGetBufferDeviceAddress)
            d->fpGetBufferDeviceAddress = (PFN_vkGetBufferDeviceAddress)fpGetDeviceProcAddr(*pDevice, "vkGetBufferDeviceAddressKHR");
        LOAD_DEV(CmdBindPipeline);
        LOAD_DEV(CmdPushConstants);
        LOAD_DEV(CmdPushConstants2);
        LOAD_DEV(CmdPushConstants2KHR);
        LOAD_DEV(CmdDispatch);
        LOAD_DEV(CmdPipelineBarrier);
        LOAD_DEV(CmdCopyBufferToImage);
        LOAD_DEV(CmdCopyBufferToImage2);
        if (!d->fpCmdCopyBufferToImage2)
            d->fpCmdCopyBufferToImage2 = (PFN_vkCmdCopyBufferToImage2)fpGetDeviceProcAddr(*pDevice, "vkCmdCopyBufferToImage2KHR");
        LOAD_DEV(ResetCommandBuffer);
        LOAD_DEV(BeginCommandBuffer);
        LOAD_DEV(AllocateCommandBuffers);
        LOAD_DEV(FreeCommandBuffers);
        LOAD_DEV(ResetCommandPool);
        LOAD_DEV(DestroyCommandPool);
        LOAD_DEV(MapMemory);
        LOAD_DEV(UnmapMemory);
        LOAD_DEV(MapMemory2KHR);
        LOAD_DEV(UnmapMemory2KHR);
        LOAD_DEV(FlushMappedMemoryRanges);
        LOAD_DEV(InvalidateMappedMemoryRanges);
        d->afrc = want_icc && g_bc_afrc;
        d->zero_divisor_emulated = zero_div_emulated;
        const VkPhysicalDeviceDescriptorBufferFeaturesEXT *dbf = find_struct(pCreateInfo->pNext,
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_BUFFER_FEATURES_EXT);
        d->descriptor_buffer = dbf && dbf->descriptorBuffer;
        d->fpCmdBindDescriptorBuffersEXT = (PFN_vkCmdBindDescriptorBuffersEXT)fpGetDeviceProcAddr(*pDevice, "vkCmdBindDescriptorBuffersEXT");
        d->fpCmdPushDescriptorSetKHR = (PFN_vkCmdPushDescriptorSetKHR)fpGetDeviceProcAddr(*pDevice, "vkCmdPushDescriptorSetKHR");
        if (!d->fpCmdPushDescriptorSetKHR)
            d->fpCmdPushDescriptorSetKHR = (PFN_vkCmdPushDescriptorSetKHR)fpGetDeviceProcAddr(*pDevice, "vkCmdPushDescriptorSet");
        d->fpCmdSetVertexInputEXT = (PFN_vkCmdSetVertexInputEXT)fpGetDeviceProcAddr(*pDevice, "vkCmdSetVertexInputEXT");
        d->fpCmdSetDepthCompareOp = (PFN_vkCmdSetDepthCompareOp)fpGetDeviceProcAddr(*pDevice, "vkCmdSetDepthCompareOp");
        if (!d->fpCmdSetDepthCompareOp)
            d->fpCmdSetDepthCompareOp = (PFN_vkCmdSetDepthCompareOp)fpGetDeviceProcAddr(*pDevice, "vkCmdSetDepthCompareOpEXT");
        if (g_log_images && want_icc) {
            d->fpGetImageSubresourceLayout2KHR = (PFN_vkGetImageSubresourceLayout2KHR)fpGetDeviceProcAddr(*pDevice, "vkGetImageSubresourceLayout2KHR");
            if (!d->fpGetImageSubresourceLayout2KHR)
                d->fpGetImageSubresourceLayout2KHR = (PFN_vkGetImageSubresourceLayout2KHR)fpGetDeviceProcAddr(*pDevice, "vkGetImageSubresourceLayout2EXT");
        }
        d->fpCreateComputePipelinesReal = (PFN_vkCreateComputePipelines)fpGetDeviceProcAddr(*pDevice, "vkCreateComputePipelines");
        #undef LOAD_DEV
        if (g_gpu_time || g_async_ms || g_noatt_w || g_no_susres || g_dc_on || g_ct) gt_init(d, fpGetDeviceProcAddr); // without GPU_TIME: pointers only, no pool
        if (g_ct && ct_ext_ok) ct_device_init(d, fpGetDeviceProcAddr);
        if (g_memlog) {
            d->ml_fpQueueSubmit = (PFN_vkQueueSubmit)fpGetDeviceProcAddr(*pDevice, "vkQueueSubmit");
            d->ml_fpQueueSubmit2 = (PFN_vkQueueSubmit2)fpGetDeviceProcAddr(*pDevice, "vkQueueSubmit2");
            if (!d->ml_fpQueueSubmit2) d->ml_fpQueueSubmit2 = (PFN_vkQueueSubmit2)fpGetDeviceProcAddr(*pDevice, "vkQueueSubmit2KHR");
            if (g_memlog == 1) { // bufferDeviceAddress is always enabled above, every allocation gets DEVICE_ADDRESS_BIT
                d->fpGetDeviceMemoryOpaqueCaptureAddress = (PFN_vkGetDeviceMemoryOpaqueCaptureAddress)fpGetDeviceProcAddr(*pDevice, "vkGetDeviceMemoryOpaqueCaptureAddress");
                if (!d->fpGetDeviceMemoryOpaqueCaptureAddress)
                    d->fpGetDeviceMemoryOpaqueCaptureAddress = (PFN_vkGetDeviceMemoryOpaqueCaptureAddress)fpGetDeviceProcAddr(*pDevice, "vkGetDeviceMemoryOpaqueCaptureAddressKHR");
            }
        }
        if (d->gt) { // SUSRES_BARRIER: vkCmdPipelineBarrier2 only when synchronization2 is enabled, else the v1 fallback
            const VkPhysicalDeviceVulkan13Features *v13 = find_struct(mod_ci.pNext, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES);
            const VkPhysicalDeviceSynchronization2Features *s2 = find_struct(mod_ci.pNext, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SYNCHRONIZATION_2_FEATURES);
            if (!(v13 && v13->synchronization2) && !(s2 && s2->synchronization2)) d->gt->fpCmdPipelineBarrier2 = NULL;
        }
        pc_init(d);
        memset(d->type_heap, BUDGET_NO_HEAP, sizeof d->type_heap);
        if (inst && inst->fpGetPhysicalDeviceMemoryProperties) {
            VkPhysicalDeviceMemoryProperties mp;
            inst->fpGetPhysicalDeviceMemoryProperties(physicalDevice, &mp);
            for (uint32_t i = 0; i < mp.memoryTypeCount && i < VK_MAX_MEMORY_TYPES; i++)
                if (mp.memoryTypes[i].heapIndex < VK_MAX_MEMORY_HEAPS) d->type_heap[i] = (uint8_t)mp.memoryTypes[i].heapIndex;
            for (uint32_t i = 0; i < mp.memoryHeapCount && i < VK_MAX_MEMORY_HEAPS; i++)
                g_heap_budget[i] = mp.memoryHeaps[i].size > g_budget_cap ? g_budget_cap : mp.memoryHeaps[i].size;
            atomic_store(&g_heap_count, mp.memoryHeapCount);
            static _Atomic bool budget_log_started;
            if (g_budget_log_s && !atomic_exchange(&budget_log_started, true)) {
                pthread_t t;
                if (pthread_create(&t, NULL, budget_log_thread, NULL) == 0) pthread_detach(t);
            }
            for (uint32_t i = 0; i < mp.memoryTypeCount; i++)
                if (mp.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) d->coherent_types |= 1u << i;
            const VkMemoryPropertyFlags hc = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
            for (uint32_t i = 0; i < mp.memoryTypeCount; i++)
                if ((mp.memoryTypes[i].propertyFlags & hc) == VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)
                    for (uint32_t j = 0; j < mp.memoryTypeCount; j++)
                        if ((mp.memoryTypes[j].propertyFlags & hc) == hc && mp.memoryTypes[j].heapIndex == mp.memoryTypes[i].heapIndex) {
                            d->coherent_remap[i] = (uint8_t)(j + 1);
                            break;
                        }
        }
        d->bc_init_done = false;
        if (g_dc_on) dc_device_init(d, fpGetDeviceProcAddr, *pDevice);
        if (g_wls_clamp) {
            char hl[16 * 9 + 1] = "";
            for (uint32_t i = 0; i < g_wls_hash_count; i++)
                snprintf(hl + strlen(hl), sizeof hl - strlen(hl), "%s%08x", i ? "," : "", g_wls_hashes[i]);
            fprintf(stderr, "mali-compat: wls-clamp-1 active: VkDevice %p, modules %s, so far %llu modules / %llu chains\n",
                    (void *)*pDevice, g_wls_clamp == 1 ? "all" : hl, (unsigned long long)atomic_load(&g_wls_modules),
                    (unsigned long long)atomic_load(&g_wls_chains));
        }
        fprintf(stderr, "mali-compat: pc2-restore-1 active: VkDevice %p, driver vkCmdPushConstants2 %s, "
                "vkCmdPushConstants2KHR %s\n", (void *)*pDevice, d->fpCmdPushConstants2 ? "yes" : "no",
                d->fpCmdPushConstants2KHR ? "yes" : "no");
        static _Atomic bool pc2_log_started;
        if (!atomic_exchange(&pc2_log_started, true)) {
            pthread_t t;
            if (pthread_create(&t, NULL, pc2_log_thread, NULL) == 0) pthread_detach(t);
        }
    }
    DEV_TABLE_CHANGED();
    pthread_mutex_unlock(&g_table_mutex);

    LOG("created VkDevice %p (key=%p)", (void *)*pDevice, *pDevice ? *(void **)*pDevice : NULL);
    return VK_SUCCESS;
}

static VKAPI_ATTR void VKAPI_CALL compat_DestroyDevice(
    VkDevice device, const VkAllocationCallbacks *pAllocator) {
    struct DeviceData *d = get_device_data(device);
    if (g_memlog) ml_flush();
    if (d) {
        ct_device_destroy(d);
        sd_device_destroy(d);
        gt_destroy(d);
        pc_destroy(d);
        if (d->bc_init_done) {
            for (int i = 0; i < 4; i++) {
                if (d->bc_pipelines[i]) d->fpDestroyPipeline(device, d->bc_pipelines[i], NULL);
                if (d->bc_pipelines_db[i]) d->fpDestroyPipeline(device, d->bc_pipelines_db[i], NULL);
            }
            if (d->bc_layout) d->fpDestroyPipelineLayout(device, d->bc_layout, NULL);
        }
        destroy_staging_pool(d);
        if (d->fpDestroyDevice && device)
            d->fpDestroyDevice(device, pAllocator);
    }
    if (device) {
        pthread_mutex_lock(&g_table_mutex);
        void *key = *(void **)device;
        for (uint32_t i = 0; i < g_device_count; i++) {
            if (g_devices[i].device == device || (key && g_devices[i].dispatch_key == key)) {
                for (uint32_t j = i; j + 1 < g_device_count; j++) {
                    g_devices[j] = g_devices[j + 1];
                }
                g_device_count--;
                break;
            }
        }
        DEV_TABLE_CHANGED();
        pthread_mutex_unlock(&g_table_mutex);
    }
}

// -----------------------------------------------------------------------------
// Loader Layer Negotiate & ProcAddr Dispatch
// -----------------------------------------------------------------------------
EXPORT VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL vkGetDeviceProcAddr(VkDevice device, const char *pName) {
    if (!pName) return NULL;
    // Must be this layer's own: the Arm WSI layer above resolves every entry point it does not know through the
    // "vkGetDeviceProcAddr" it gets here. Forwarding the name handed it the driver's, so vkCreateShaderModule and
    // vkCreateGraphicsPipelines skipped this layer (no SPIR-V patching, vendor compiler crash). [Claude Opus 5.5]
    if (!strcmp(pName, "vkGetDeviceProcAddr")) return (PFN_vkVoidFunction)vkGetDeviceProcAddr;
    if (g_sd_armed && !tl_sd_inner) { PFN_vkVoidFunction sw = sd_proc(device, pName); if (sw) return sw; } // sync-diag
    // GPU timing: which vkCmdDraw*/Dispatch* entry points the app asks for (once each), to see what the hooks miss.
    if (g_gpu_time && (!strncmp(pName, "vkCmdDraw", 9) || !strncmp(pName, "vkCmdDispatch", 13) ||
                       !strncmp(pName, "vkCmdExecute", 12) || !strncmp(pName, "vkCmdBeginRender", 16))) {
        static char seen[64][48];
        static int nseen;
        static pthread_mutex_t m = PTHREAD_MUTEX_INITIALIZER;
        pthread_mutex_lock(&m);
        int k = 0;
        while (k < nseen && strcmp(seen[k], pName)) k++;
        if (k == nseen && nseen < 64) {
            snprintf(seen[nseen++], sizeof seen[0], "%s", pName);
            fprintf(stderr, "mali-compat: GPU timing: app asked for %s\n", pName);
        }
        pthread_mutex_unlock(&m);
    }
    // Only expose the map2 entry points when they exist below: 64-bit Wine does not enable VK_KHR_map_memory2, and
    // handing it our function then made it call through a NULL driver pointer (The Witcher 3: crash in vkMapMemory).
    // [Claude Opus 5.5]
    if (!strcmp(pName, "vkGetDeviceImageMemoryRequirements") || !strcmp(pName, "vkGetDeviceImageMemoryRequirementsKHR")) {
        struct DeviceData *dd = get_device_data(device);
        if (dd && dd->fpGetDeviceImageMemoryRequirements) return (PFN_vkVoidFunction)compat_GetDeviceImageMemoryRequirements;
        return dd ? dd->get_device_proc_addr(device, pName) : NULL;
    }
    if (!strcmp(pName, "vkGetDeviceImageSparseMemoryRequirements") || !strcmp(pName, "vkGetDeviceImageSparseMemoryRequirementsKHR")) {
        struct DeviceData *dd = get_device_data(device);
        if (dd && dd->fpGetDeviceImageSparseMemoryRequirements) return (PFN_vkVoidFunction)compat_GetDeviceImageSparseMemoryRequirements;
        return dd ? dd->get_device_proc_addr(device, pName) : NULL;
    }
    if (!strcmp(pName, "vkCmdBindDescriptorBuffersEXT")) {
        struct DeviceData *dd = get_device_data(device);
        if (dd && dd->fpCmdBindDescriptorBuffersEXT) return (PFN_vkVoidFunction)compat_CmdBindDescriptorBuffersEXT;
        return dd ? dd->get_device_proc_addr(device, pName) : NULL;
    }
    if (!strcmp(pName, "vkCmdSetVertexInputEXT")) {
        struct DeviceData *dd = get_device_data(device);
        if (dd && dd->fpCmdSetVertexInputEXT) return (PFN_vkVoidFunction)compat_CmdSetVertexInputEXT;
        return dd ? dd->get_device_proc_addr(device, pName) : NULL;
    }
    if (!strcmp(pName, "vkMapMemory2KHR") || !strcmp(pName, "vkUnmapMemory2KHR") || !strcmp(pName, "vkMapMemory2") ||
        !strcmp(pName, "vkUnmapMemory2")) {
        struct DeviceData *dd = get_device_data(device);
        if (!dd || !dd->fpMapMemory2KHR) return dd ? dd->get_device_proc_addr(device, pName) : NULL;
    }
    if (g_dc_on) { PFN_vkVoidFunction dh = dc_proc(device, pName); if (dh) return dh; } // desc-capture-1
    #define MATCH_DEV(fn) if (!strcmp(pName, "vk" #fn)) return (PFN_vkVoidFunction)compat_##fn
    if (g_memlog) { // MALI_COMPAT_MEMLOG only: vkMapMemory for the M events, queue submits for the S events
        MATCH_DEV(MapMemory);
        struct DeviceData *md = get_device_data(device);
        if (md && md->ml_fpQueueSubmit && !strcmp(pName, "vkQueueSubmit")) return (PFN_vkVoidFunction)compat_MlQueueSubmit;
        if (md && md->ml_fpQueueSubmit2 && (!strcmp(pName, "vkQueueSubmit2") || !strcmp(pName, "vkQueueSubmit2KHR")))
            return (PFN_vkVoidFunction)compat_MlQueueSubmit2;
    }
    // GPU timing hooks: only while the switch is on, so the normal path calls the driver directly.
    #define MATCH_DEV_KHR(fn) if (!strcmp(pName, "vk" #fn) || !strcmp(pName, "vk" #fn "KHR")) return (PFN_vkVoidFunction)compat_##fn
    if (g_async_ms) MATCH_DEV(DestroyPipeline);
    if (g_depth_equal_on && (!strcmp(pName, "vkCmdSetDepthCompareOp") || !strcmp(pName, "vkCmdSetDepthCompareOpEXT"))) {
        struct DeviceData *dd = get_device_data(device);
        if (dd && dd->fpCmdSetDepthCompareOp) return (PFN_vkVoidFunction)compat_CmdSetDepthCompareOp;
    }
    if (g_ct) { // cover-trim: the app's scissor and conditional rendering, only when its device state exists
        struct DeviceData *cd = get_device_data(device);
        if (cd && cd->ct) {
            if (!strcmp(pName, "vkCmdSetScissor")) return (PFN_vkVoidFunction)compat_CmdSetScissor;
            if ((!strcmp(pName, "vkCmdSetScissorWithCount") || !strcmp(pName, "vkCmdSetScissorWithCountEXT")) && cd->ct->SetScissorWithCount)
                return (PFN_vkVoidFunction)compat_CmdSetScissorWithCount;
            if (!strcmp(pName, "vkCmdBeginConditionalRenderingEXT") && cd->ct->BeginCond) return (PFN_vkVoidFunction)compat_CmdBeginConditionalRenderingEXT;
            if (!strcmp(pName, "vkCmdEndConditionalRenderingEXT") && cd->ct->EndCond) return (PFN_vkVoidFunction)compat_CmdEndConditionalRenderingEXT;
        }
    }
    if (g_gpu_time || g_async_ms || g_noatt_w || g_no_susres || g_dc_on || g_ct) { // GPU timing, async compiles, pass clamp, NO_SUSPEND_RESUME, desc-capture-1 and cover-trim share these hooks
        MATCH_DEV_KHR(CmdBeginRendering);
        MATCH_DEV_KHR(CmdEndRendering);
        MATCH_DEV(CmdBeginRenderPass);
        MATCH_DEV(CmdEndRenderPass);
        MATCH_DEV_KHR(CmdBeginRenderPass2);
        MATCH_DEV_KHR(CmdEndRenderPass2);
        MATCH_DEV(CmdDispatch);
        MATCH_DEV(CmdDispatchIndirect);
        MATCH_DEV(CmdDraw);
        MATCH_DEV(CmdDrawIndexed);
        MATCH_DEV(CmdDrawIndirect);
        MATCH_DEV(CmdDrawIndexedIndirect);
        MATCH_DEV_KHR(CmdDrawIndirectCount);
        MATCH_DEV_KHR(CmdDrawIndexedIndirectCount);
        struct DeviceData *gd = get_device_data(device);
        if (gd && gd->gt) {
            #define MATCH_GT_OPT(fn) if (!strcmp(pName, "vk" #fn) && gd->gt->fp##fn) return (PFN_vkVoidFunction)compat_##fn
            MATCH_GT_OPT(CmdDrawMultiEXT);
            MATCH_GT_OPT(CmdDrawMultiIndexedEXT);
            MATCH_GT_OPT(CmdDrawMeshTasksEXT);
            MATCH_GT_OPT(CmdDrawMeshTasksIndirectEXT);
            MATCH_GT_OPT(CmdDrawMeshTasksIndirectCountEXT);
            if ((!strcmp(pName, "vkCmdDispatchBase") || !strcmp(pName, "vkCmdDispatchBaseKHR")) && gd->gt->fpCmdDispatchBase)
                return (PFN_vkVoidFunction)compat_CmdDispatchBase;
            // GPU timing only: secondary command buffers into their pass, queue-side CPU time
            #define MATCH_GT_NAME(fn, name) if ((g_gpu_time || g_dc_on) && !strcmp(pName, name) && gd->gt->fp##fn) return (PFN_vkVoidFunction)compat_##fn
            if (g_ct && !strcmp(pName, "vkCmdExecuteCommands") && gd->gt->fpCmdExecuteCommands) return (PFN_vkVoidFunction)compat_CmdExecuteCommands;
            MATCH_GT_NAME(CmdExecuteCommands, "vkCmdExecuteCommands");
            MATCH_GT_NAME(QueueSubmit, "vkQueueSubmit");
            MATCH_GT_NAME(QueueSubmit2, "vkQueueSubmit2");
            MATCH_GT_NAME(QueueSubmit2, "vkQueueSubmit2KHR");
            MATCH_GT_NAME(WaitForFences, "vkWaitForFences");
            MATCH_GT_NAME(WaitSemaphores, "vkWaitSemaphores");
            MATCH_GT_NAME(WaitSemaphores, "vkWaitSemaphoresKHR");
            MATCH_GT_NAME(QueueWaitIdle, "vkQueueWaitIdle");
            MATCH_GT_NAME(DeviceWaitIdle, "vkDeviceWaitIdle");
            #undef MATCH_GT_NAME
            #undef MATCH_GT_OPT
        }
    }
    if (!strcmp(pName, "vkCmdPushDescriptorSetKHR") || !strcmp(pName, "vkCmdPushDescriptorSet")) {
        struct DeviceData *dd = get_device_data(device);
        if (dd && dd->fpCmdPushDescriptorSetKHR && (g_gpu_time || g_vendor_id || g_dc_on)) return (PFN_vkVoidFunction)compat_CmdPushDescriptorSetKHR;
    }
    if (g_gpu_time || g_dump_hash_count || g_pcache_on || g_dump_inflight || g_dc_on || g_wls_clamp || g_ct) MATCH_DEV(CreateComputePipelines);
    if (g_fake_occlusion) {
        MATCH_DEV(CreateQueryPool);
        MATCH_DEV(DestroyQueryPool);
        MATCH_DEV(CmdCopyQueryPoolResults);
        MATCH_DEV(GetQueryPoolResults);
    }
    #undef MATCH_DEV_KHR
    MATCH_DEV(DestroyDevice);
    MATCH_DEV(CreateShaderModule);
    MATCH_DEV(CreateGraphicsPipelines);
    MATCH_DEV(CmdSetPolygonModeEXT);
    MATCH_DEV(CreateImage);
    MATCH_DEV(DestroyImage);
    MATCH_DEV(CreateImageView);
    MATCH_DEV(CreateBuffer);
    MATCH_DEV(CreateBufferView);
    MATCH_DEV(AllocateMemory);
    MATCH_DEV(FreeMemory);
    if (g_dmabuf_export) { MATCH_DEV(GetMemoryFdKHR); }
    MATCH_DEV(UnmapMemory);
    MATCH_DEV(MapMemory2KHR);
    MATCH_DEV(UnmapMemory2KHR);
    if (!strcmp(pName, "vkMapMemory2")) return (PFN_vkVoidFunction)compat_MapMemory2KHR;
    if (!strcmp(pName, "vkUnmapMemory2")) return (PFN_vkVoidFunction)compat_UnmapMemory2KHR;
    MATCH_DEV(FlushMappedMemoryRanges);
    MATCH_DEV(InvalidateMappedMemoryRanges);
    MATCH_DEV(CmdCopyBufferToImage);
    MATCH_DEV(CmdCopyBufferToImage2);
    if (!strcmp(pName, "vkCmdCopyBufferToImage2KHR")) return (PFN_vkVoidFunction)compat_CmdCopyBufferToImage2;
    MATCH_DEV(CmdBindPipeline);
    MATCH_DEV(CmdPushConstants);
    // pc2-restore-1: only names the driver has (a hook for a missing one would call through NULL)
    if (!strcmp(pName, "vkCmdPushConstants2") || !strcmp(pName, "vkCmdPushConstants2KHR")) {
        struct DeviceData *dd = get_device_data(device);
        bool khr = pName[19] == 'K';
        if (dd && (khr ? dd->fpCmdPushConstants2KHR != NULL : dd->fpCmdPushConstants2 != NULL))
            return khr ? (PFN_vkVoidFunction)compat_CmdPushConstants2KHR : (PFN_vkVoidFunction)compat_CmdPushConstants2;
        return dd ? dd->get_device_proc_addr(device, pName) : NULL;
    }
    MATCH_DEV(ResetCommandBuffer);
    MATCH_DEV(BeginCommandBuffer);
    MATCH_DEV(AllocateCommandBuffers);
    MATCH_DEV(FreeCommandBuffers);
    MATCH_DEV(ResetCommandPool);
    MATCH_DEV(DestroyCommandPool);
    #undef MATCH_DEV

    struct DeviceData *d = get_device_data(device);
    return d ? d->get_device_proc_addr(device, pName) : NULL;
}

EXPORT VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL vkGetInstanceProcAddr(VkInstance instance, const char *pName) {
    if (!pName) return NULL;
    if (!strcmp(pName, "vkCreateInstance")) return (PFN_vkVoidFunction)compat_CreateInstance;
    if (!strcmp(pName, "vkCreateDevice")) return (PFN_vkVoidFunction)compat_CreateDevice;
    if (!strcmp(pName, "vkGetInstanceProcAddr")) return (PFN_vkVoidFunction)vkGetInstanceProcAddr;
    if (!strcmp(pName, "vkGetDeviceProcAddr")) return (PFN_vkVoidFunction)vkGetDeviceProcAddr;

    #define MATCH_INST(fn) if (!strcmp(pName, "vk" #fn)) return (PFN_vkVoidFunction)compat_##fn
    MATCH_INST(DestroyInstance);
    MATCH_INST(GetPhysicalDeviceFeatures);
    MATCH_INST(GetPhysicalDeviceFeatures2);
    if (!strcmp(pName, "vkGetPhysicalDeviceFeatures2KHR")) return (PFN_vkVoidFunction)compat_GetPhysicalDeviceFeatures2;
    MATCH_INST(GetPhysicalDeviceFormatProperties);
    MATCH_INST(GetPhysicalDeviceFormatProperties2);
    if (!strcmp(pName, "vkGetPhysicalDeviceFormatProperties2KHR")) return (PFN_vkVoidFunction)compat_GetPhysicalDeviceFormatProperties2;
    MATCH_INST(GetPhysicalDeviceImageFormatProperties);
    MATCH_INST(GetPhysicalDeviceImageFormatProperties2);
    if (!strcmp(pName, "vkGetPhysicalDeviceImageFormatProperties2KHR")) return (PFN_vkVoidFunction)compat_GetPhysicalDeviceImageFormatProperties2;
    MATCH_INST(GetPhysicalDeviceMemoryProperties2);
    MATCH_INST(GetPhysicalDeviceProperties);
    MATCH_INST(GetPhysicalDeviceProperties2);
    if (!strcmp(pName, "vkGetPhysicalDeviceProperties2KHR")) return (PFN_vkVoidFunction)compat_GetPhysicalDeviceProperties2;
    if (!strcmp(pName, "vkGetPhysicalDeviceMemoryProperties2KHR")) return (PFN_vkVoidFunction)compat_GetPhysicalDeviceMemoryProperties2;
    MATCH_INST(EnumerateDeviceExtensionProperties);
    #undef MATCH_INST

    // Also check device proc addr matches for device functions queried via GetInstanceProcAddr
    #define MATCH_DEV2(fn) if (!strcmp(pName, "vk" #fn)) return (PFN_vkVoidFunction)compat_##fn
    MATCH_DEV2(DestroyDevice);
    MATCH_DEV2(CreateShaderModule);
    MATCH_DEV2(CreateGraphicsPipelines);
    MATCH_DEV2(CmdSetPolygonModeEXT);
    MATCH_DEV2(CreateImage);
    MATCH_DEV2(DestroyImage);
    MATCH_DEV2(CreateImageView);
    MATCH_DEV2(CreateBuffer);
    MATCH_DEV2(CreateBufferView);
    MATCH_DEV2(AllocateMemory);
    MATCH_DEV2(FreeMemory);
    MATCH_DEV2(UnmapMemory);
    if (g_memlog) MATCH_DEV2(MapMemory);
    MATCH_DEV2(MapMemory2KHR);
    MATCH_DEV2(UnmapMemory2KHR);
    if (!strcmp(pName, "vkMapMemory2")) return (PFN_vkVoidFunction)compat_MapMemory2KHR;
    if (!strcmp(pName, "vkUnmapMemory2")) return (PFN_vkVoidFunction)compat_UnmapMemory2KHR;
    MATCH_DEV2(FlushMappedMemoryRanges);
    MATCH_DEV2(InvalidateMappedMemoryRanges);
    MATCH_DEV2(CmdCopyBufferToImage);
    MATCH_DEV2(CmdCopyBufferToImage2);
    if (!strcmp(pName, "vkCmdCopyBufferToImage2KHR")) return (PFN_vkVoidFunction)compat_CmdCopyBufferToImage2;
    MATCH_DEV2(CmdBindPipeline);
    MATCH_DEV2(CmdPushConstants);
    MATCH_DEV2(ResetCommandBuffer);
    MATCH_DEV2(BeginCommandBuffer);
    MATCH_DEV2(FreeCommandBuffers);
    #undef MATCH_DEV2

    struct InstanceData *inst = get_instance_data(instance);
    return inst ? inst->get_instance_proc_addr(instance, pName) : NULL;
}

EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkNegotiateLoaderLayerInterfaceVersion(VkNegotiateLayerInterface *pVersionStruct) {
    if (!pVersionStruct || pVersionStruct->sType != LAYER_NEGOTIATE_INTERFACE_STRUCT)
        return VK_ERROR_INITIALIZATION_FAILED;

    if (pVersionStruct->loaderLayerInterfaceVersion >= 2) {
        pVersionStruct->loaderLayerInterfaceVersion = 2;
        pVersionStruct->pfnGetInstanceProcAddr = vkGetInstanceProcAddr;
        pVersionStruct->pfnGetDeviceProcAddr = vkGetDeviceProcAddr;
        pVersionStruct->pfnGetPhysicalDeviceProcAddr = NULL;
        return VK_SUCCESS;
    }
    return VK_ERROR_INITIALIZATION_FAILED;
}

EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkEnumerateInstanceLayerProperties(
    uint32_t *pPropertyCount, VkLayerProperties *pProperties) {
    if (pPropertyCount) *pPropertyCount = 1;
    if (pProperties) {
        strncpy(pProperties->layerName, LAYER_NAME, VK_MAX_EXTENSION_NAME_SIZE);
        strncpy(pProperties->description, "Mali Compatibility Layer for DXVK / Proton", VK_MAX_DESCRIPTION_SIZE);
        pProperties->implementationVersion = 1;
        pProperties->specVersion = VK_API_VERSION_1_3;
    }
    return VK_SUCCESS;
}

EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkEnumerateInstanceExtensionProperties(
    const char *pLayerName, uint32_t *pPropertyCount, VkExtensionProperties *pProperties) {
    (void)pLayerName;
    (void)pProperties;
    if (pPropertyCount) *pPropertyCount = 0;
    return VK_SUCCESS;
}

EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkEnumerateDeviceLayerProperties(
    VkPhysicalDevice physicalDevice, uint32_t *pPropertyCount, VkLayerProperties *pProperties) {
    (void)physicalDevice;
    return vkEnumerateInstanceLayerProperties(pPropertyCount, pProperties);
}

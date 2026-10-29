#define _GNU_SOURCE
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
#include <errno.h>
#include <sys/mman.h>
#include <sys/stat.h>
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
static uint32_t *compat_patch_spirv(const uint32_t *inWords, size_t inBytes, size_t *pOutBytes) {
    uint32_t *out = compat_patch_spirv_impl(inWords, inBytes, pOutBytes);
    if (out) dump_spirv(inWords, inBytes, out, *pOutBytes);
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

static uint32_t handle_bucket(uint64_t v, uint32_t n) { return (uint32_t)((v ^ (v >> 17) ^ (v >> 33)) % n); }

static void register_bc_image(VkImage img, VkFormat orig, VkFormat decoded, VkExtent3D ext) {
    struct BCNode *e = calloc(1, sizeof *e);
    if (!e) return;
    e->t = (struct TrackedBCImage){.image = img, .orig_format = orig, .decoded_format = decoded, .extent = ext};
    pthread_mutex_lock(&g_bc_image_mutex);
    uint32_t bkt = handle_bucket((uint64_t)(uintptr_t)img, BC_BUCKETS);
    e->next = g_bc_images[bkt];
    g_bc_images[bkt] = e;
    pthread_mutex_unlock(&g_bc_image_mutex);
}

// Copies the entry out (the node can be freed by another thread's vkDestroyImage).
static bool find_bc_image(VkImage img, struct TrackedBCImage *out) {
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
    if (e) *pp = e->next;
    pthread_mutex_unlock(&g_bc_image_mutex);
    free(e);
}

struct StagingResource {
    VkBuffer buffer;
    VkDeviceMemory memory;
    VkDeviceSize size;      // power-of-two bucket, >= STAGING_MIN
    VkDeviceAddress addr;
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
    uint8_t data[256];
};

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
    int32_t gt_rec;                 // GPU timing record of the open render pass, -1 if none
    VkImageView gt_src;             // first image view of the last push-descriptor write (GPU timing details)
    struct CmdBufData *next;
};

#define CMDBUF_BUCKETS 1024
static struct CmdBufData *g_cmdbufs[CMDBUF_BUCKETS];
static pthread_mutex_t g_cmdbuf_mutex = PTHREAD_MUTEX_INITIALIZER;
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
    PFN_vkBindBufferMemory fpBindBufferMemory;
    PFN_vkGetBufferMemoryRequirements fpGetBufferMemoryRequirements;
    PFN_vkGetBufferDeviceAddress fpGetBufferDeviceAddress;
    PFN_vkCmdBindPipeline fpCmdBindPipeline;
    PFN_vkCmdPushConstants fpCmdPushConstants;
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
    PFN_vkCreateComputePipelines fpCreateComputePipelinesReal; // for the GPU timing's pipeline hashes
    struct GpuTime *gt; // MALI_COMPAT_GPU_TIME
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

static struct DeviceData *get_device_data(VkDevice dev) {
    pthread_mutex_lock(&g_table_mutex);
    for (int32_t i = (int32_t)g_device_count - 1; i >= 0; i--) {
        if (g_devices[i].device == dev) {
            pthread_mutex_unlock(&g_table_mutex);
            return &g_devices[i];
        }
    }
    void *key = dev ? *(void **)dev : NULL;
    if (key) {
        for (int32_t i = (int32_t)g_device_count - 1; i >= 0; i--) {
            if (g_devices[i].dispatch_key == key) {
                pthread_mutex_unlock(&g_table_mutex);
                return &g_devices[i];
            }
        }
    }
    struct DeviceData *fallback = g_device_count > 0 ? &g_devices[g_device_count - 1] : NULL;
    pthread_mutex_unlock(&g_table_mutex);
    return fallback;
}

static struct DeviceData *get_device_data_by_cmd(VkCommandBuffer cmd) {
    pthread_mutex_lock(&g_table_mutex);
    void *key = cmd ? *(void **)cmd : NULL;
    if (key) {
        for (int32_t i = (int32_t)g_device_count - 1; i >= 0; i--) {
            if (g_devices[i].dispatch_key == key) {
                pthread_mutex_unlock(&g_table_mutex);
                return &g_devices[i];
            }
        }
    }
    struct DeviceData *fallback = g_device_count > 0 ? &g_devices[g_device_count - 1] : NULL;
    pthread_mutex_unlock(&g_table_mutex);
    return fallback;
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
    if (entry && entry->staging_count < entry->staging_capacity)
        entry->staging_resources[entry->staging_count++] = res;
}

static void destroy_staging(struct DeviceData *d, struct StagingResource *r) {
    if (r->buffer) d->fpDestroyBuffer(d->device, r->buffer, NULL);
    if (r->memory) d->fpFreeMemory(d->device, r->memory, NULL);
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
    e->gt_rec = -1;
    e->gt_src = VK_NULL_HANDLE;
    if (remove) {
        free(e->staging_resources);
        free(e);
    }
}

// Memory budget tracking
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
static VKAPI_ATTR void VKAPI_CALL compat_GetPhysicalDeviceFeatures(
    VkPhysicalDevice physicalDevice, VkPhysicalDeviceFeatures *pFeatures) {
    struct InstanceData *inst = get_instance_by_phys_dev(physicalDevice);
    if (inst && inst->fpGetPhysicalDeviceFeatures)
        inst->fpGetPhysicalDeviceFeatures(physicalDevice, pFeatures);

    pFeatures->textureCompressionBC = VK_TRUE;
    pFeatures->shaderClipDistance = VK_TRUE;
    pFeatures->shaderCullDistance = VK_TRUE;
    pFeatures->fillModeNonSolid = VK_TRUE;
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
        }
        header = header->pNext;
    }
}

// MALI_COMPAT_VENDOR_ID=<hex>: report this PCI vendor ID instead of Arm's 0x13b5 (driverID stays ARM_PROPRIETARY).
// vkd3d-proton reads root CBVs through buffer device addresses on every vendor except NVIDIA and Qualcomm, which
// get push descriptors ("~15%" there; Mali can only preload UBOs into uniform registers). 5143 flips only that
// choice: vkd3d's other Qualcomm check also needs driverID TURNIP. The game sees the vendor through DXGI too
// (dxgi.customVendorId keeps it Arm). [Claude Opus 5.5] 2026-10-28
static uint32_t g_vendor_id;
// GPU timing diagnostics for instanced vertex input (HZD's missing static meshes, 2026-10-29): this driver has
// supportsNonZeroFirstInstance = false, so a draw with firstInstance != 0 on a binding whose divisor is not 1 (the
// emulated zero divisor included) reads undefined instance data. Pipelines and dynamic vertex inputs with such
// bindings, and draws with firstInstance != 0.
static _Atomic uint64_t g_div_pipes, g_div_vi, g_div_draws;
// MALI_COMPAT_NO_SUSPEND_RESUME=1 (experiment, needs GPU timing for the hook): vkd3d-proton splits render passes on
// tilers with VK_RENDERING_SUSPENDING/RESUMING_BIT. Turn each part into a plain pass: a suspending part stores its
// attachments, a resuming part loads them, and only the last part resolves. Same result, more bandwidth; tests whether
// this driver loses draws in resumed passes (HZD's missing static meshes, 2026-10-29). Counters of such passes.
static bool g_no_susres;
static _Atomic uint64_t g_susp_passes, g_resu_passes;
static __thread bool tl_div_bound; // the last vertex input this thread set or bound has a divisor != 1 (approximate)

static VKAPI_ATTR void VKAPI_CALL compat_GetPhysicalDeviceProperties2(
    VkPhysicalDevice physicalDevice, VkPhysicalDeviceProperties2 *pProperties) {
    struct InstanceData *inst = get_instance_by_phys_dev(physicalDevice);
    struct unlinked u = unlink_struct(pProperties, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAP_MEMORY_PLACED_PROPERTIES_EXT);
    if (inst && inst->fpGetPhysicalDeviceProperties2)
        inst->fpGetPhysicalDeviceProperties2(physicalDevice, pProperties);
    relink(u);
    if (u.node) ((VkPhysicalDeviceMapMemoryPlacedPropertiesEXT *)u.node)->minPlacedMemoryMapAlignment = page_size();
    // Texel buffer offsets: the vendor reports 64-byte alignment without single-texel alignment, which vkd3d-proton
    // refuses (HZD: "Lacking support for single texel alignment"). PanVK reports single-texel / 16 bytes on the same
    // GPU and its buffer descriptors work there, so report what PanVK does. Misaligned storage views are logged in
    // compat_CreateBufferView to see whether they occur at all. [Claude Opus 5.5]
    pProperties->properties.limits.minTexelBufferOffsetAlignment = 16;
    // With the push path vkd3d sizes a root CBV as min(rest of the buffer, maxUniformBufferRange), and this driver
    // reports 2 GiB, so root CBVs can span whole heap buffers. D3D12 CBVs address at most 64 KiB (NVIDIA/Qualcomm
    // report that limit): report it too, so vkd3d clamps the range itself. [Claude Opus 5.5] 2026-10-28
    if (g_vendor_id) {
        pProperties->properties.vendorID = g_vendor_id;
        if (pProperties->properties.limits.maxUniformBufferRange > 65536)
            pProperties->properties.limits.maxUniformBufferRange = 65536;
    }
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

static VKAPI_ATTR void VKAPI_CALL compat_GetPhysicalDeviceFormatProperties(
    VkPhysicalDevice physicalDevice, VkFormat format, VkFormatProperties *pFormatProperties) {
    struct InstanceData *inst = get_instance_by_phys_dev(physicalDevice);
    if (!inst || !inst->fpGetPhysicalDeviceFormatProperties) return;
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
    if (panvk_format_is_bc(format)) {
        format = panvk_bc_decoded_format(format);
        flags &= ~VK_IMAGE_CREATE_BLOCK_TEXEL_VIEW_COMPATIBLE_BIT;
    }
    return inst->fpGetPhysicalDeviceImageFormatProperties(
        physicalDevice, format, type, tiling, usage, flags, pImageFormatProperties);
}

static VKAPI_ATTR VkResult VKAPI_CALL compat_GetPhysicalDeviceImageFormatProperties2(
    VkPhysicalDevice physicalDevice, const VkPhysicalDeviceImageFormatInfo2 *pImageFormatInfo,
    VkImageFormatProperties2 *pImageFormatProperties) {
    struct InstanceData *inst = get_instance_by_phys_dev(physicalDevice);
    if (!inst || !inst->fpGetPhysicalDeviceImageFormatProperties2) return VK_ERROR_INITIALIZATION_FAILED;
    if (panvk_format_is_bc(pImageFormatInfo->format)) {
        VkPhysicalDeviceImageFormatInfo2 mod = *pImageFormatInfo;
        mod.format = panvk_bc_decoded_format(pImageFormatInfo->format);
        mod.flags &= ~VK_IMAGE_CREATE_BLOCK_TEXEL_VIEW_COMPATIBLE_BIT;
        return inst->fpGetPhysicalDeviceImageFormatProperties2(physicalDevice, &mod, pImageFormatProperties);
    }
    return inst->fpGetPhysicalDeviceImageFormatProperties2(physicalDevice, pImageFormatInfo, pImageFormatProperties);
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
            for (uint32_t i = 0; i < VK_MAX_MEMORY_HEAPS; i++) {
                if (i < pMemoryProperties->memoryProperties.memoryHeapCount) {
                    VkDeviceSize heap_size = pMemoryProperties->memoryProperties.memoryHeaps[i].size;
                    budget->heapBudget[i] = heap_size > (6ULL << 30) ? (6ULL << 30) : heap_size;
                    budget->heapUsage[i] = allocated;
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

    VkExtensionProperties *all = (VkExtensionProperties *)malloc((count + 3) * sizeof(VkExtensionProperties));
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
static uint32_t g_dump_hashes[16];
static uint32_t g_dump_hash_count;
static uint32_t g_gt_trace; // MALI_COMPAT_GT_TRACE=<hex>: write every pass whose fragment/compute shader has this hash

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
struct HashNode { uint64_t handle; uint32_t hash[3]; uint32_t stages[3]; struct HashNode *next; };
#define HASH_BUCKETS 4096
static struct HashNode *g_module_hash[HASH_BUCKETS], *g_pipeline_hash[HASH_BUCKETS];
static pthread_mutex_t g_hash_mutex = PTHREAD_MUTEX_INITIALIZER;

static void hash_put(struct HashNode **table, uint64_t handle, const uint32_t *hash, const uint32_t *stages, uint32_t n) {
    struct HashNode *e = calloc(1, sizeof *e);
    if (!e) return;
    e->handle = handle;
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

static const struct CodeNode *fragdepth_module_get(VkShaderModule m) {
    const struct CodeNode *found = NULL;
    pthread_mutex_lock(&g_hash_mutex);
    for (struct CodeNode *e = g_fragdepth_modules[handle_bucket((uint64_t)m, HASH_BUCKETS)]; e; e = e->next)
        if (e->module == (uint64_t)m) { found = e; break; }
    pthread_mutex_unlock(&g_hash_mutex);
    return found;
}

// -----------------------------------------------------------------------------
// Device Level Interceptions
// -----------------------------------------------------------------------------
static VKAPI_ATTR VkResult VKAPI_CALL compat_CreateShaderModule(
    VkDevice device, const VkShaderModuleCreateInfo *pCreateInfo,
    const VkAllocationCallbacks *pAllocator, VkShaderModule *pShaderModule) {
    struct DeviceData *d = get_device_data(device);
    uint32_t hash = 0;
    if (g_gpu_time || g_dump_hash_count) {
        hash = spv_hash(pCreateInfo->pCode, pCreateInfo->codeSize);
        spv_dump_if_wanted(pCreateInfo->pCode, pCreateInfo->codeSize, hash);
    }
    size_t patched_bytes = 0;
    uint32_t *patched = compat_patch_spirv(pCreateInfo->pCode, pCreateInfo->codeSize, &patched_bytes);

    VkResult r;
    if (patched) {
        VkShaderModuleCreateInfo mod_ci = *pCreateInfo;
        mod_ci.pCode = patched;
        mod_ci.codeSize = patched_bytes;
        r = d->fpCreateShaderModule(device, &mod_ci, pAllocator, pShaderModule);
        free(patched);
    } else {
        r = d->fpCreateShaderModule(device, pCreateInfo, pAllocator, pShaderModule);
    }
    if (r == VK_SUCCESS && g_gpu_time && pShaderModule) hash_put(g_module_hash, (uint64_t)*pShaderModule, &hash, NULL, 1);
    if (r == VK_SUCCESS && g_fragdepth_fix && pShaderModule && spv_writes_fragdepth(pCreateInfo->pCode, pCreateInfo->codeSize))
        fragdepth_module_put(*pShaderModule, pCreateInfo->pCode, pCreateInfo->codeSize);
    return r;
}

// Debug aid: with /root/mali-spv-inflight present, each thread writes the pipeline it is about to create (flags,
// stage pNext types, inline SPIR-V) to /root/mali-spv-inflight/<tid>-*, overwriting the previous one. A thread that
// dies inside the driver leaves the pipeline that killed it. [Claude Opus 5.5]
#include <sys/syscall.h>
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
        for (uint32_t st = 0; st < infos[i].stageCount; st++) {
            const VkPipelineShaderStageCreateInfo *sg = &infos[i].pStages[st];
            fprintf(f, "  stage 0x%x module %p name %s pNext:", sg->stage, (void *)sg->module, sg->pName ? sg->pName : "?");
            for (const VkBaseInStructure *c = sg->pNext; c; c = c->pNext) {
                fprintf(f, " %d", c->sType);
                if (c->sType == VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO) {
                    const VkShaderModuleCreateInfo *sm = (const VkShaderModuleCreateInfo *)c;
                    char sp[112];
                    snprintf(sp, sizeof sp, "/root/mali-spv-inflight/%ld-p%u-s%u.spv", tid, i, st);
                    FILE *o = fopen(sp, "wb");
                    if (o) { fwrite(sm->pCode, 1, sm->codeSize, o); fclose(o); }
                }
            }
            if (sg->pSpecializationInfo) fprintf(f, " spec %u entries", sg->pSpecializationInfo->mapEntryCount);
            fprintf(f, "\n");
        }
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
// [Claude Opus 5.5] 2026-10-29
static bool g_pcache_on;
static pthread_mutex_t g_pc_mutex = PTHREAD_MUTEX_INITIALIZER;
static struct DeviceData *g_pc_dev;
static VkPipelineCache g_pc;
static char g_pc_path[160];
static _Atomic uint64_t g_pc_new, g_pc_last_ns;

static void pc_save_locked(void) {
    if (!g_pc_dev || !g_pc || !atomic_load(&g_pc_new)) return;
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
    if ((f = fopen(g_pc_path, "rb"))) {
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
        if (!started && pthread_create(&t, NULL, pc_saver, NULL) == 0) { pthread_detach(t); started = true; }
    } else g_pc = VK_NULL_HANDLE;
    pthread_mutex_unlock(&g_pc_mutex);
}

static void pc_destroy(struct DeviceData *d) {
    pthread_mutex_lock(&g_pc_mutex);
    if (g_pc_dev == d) {
        pc_save_locked();
        d->fpDestroyPipelineCache(d->device, g_pc, NULL);
        g_pc = VK_NULL_HANDLE;
        g_pc_dev = NULL;
    }
    pthread_mutex_unlock(&g_pc_mutex);
}

// the cache to create pipelines with: the layer's own when it is on for this device
static VkPipelineCache pc_for(struct DeviceData *d, VkPipelineCache app, uint32_t count) {
    if (g_pc_dev != d || !g_pc) return app;
    atomic_fetch_add(&g_pc_new, count);
    atomic_store(&g_pc_last_ns, now_ns());
    return g_pc;
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
    uint32_t (*gt_hash)[6] = (g_gpu_time || g_dump_hash_count) ? calloc(createInfoCount, sizeof *gt_hash) : NULL;
    uint32_t *gt_n = gt_hash ? calloc(createInfoCount, sizeof *gt_n) : NULL;
    for (uint32_t i = 0; gt_n && i < createInfoCount; i++)
        gt_n[i] = stage_hashes(pCreateInfos[i].pStages, pCreateInfos[i].stageCount, gt_hash[i], gt_hash[i] + 3);

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
    VkResult res = d->fpCreateGraphicsPipelines(device, pc_for(d, pipelineCache, createInfoCount), createInfoCount, modified, pAllocator, pPipelines);
    for (uint32_t k = 0; k < zero_div_count; k++) zero_div[k]->divisor = 0;
    if (res != VK_SUCCESS) {
        fprintf(stderr, "mali-compat: vkCreateGraphicsPipelines FAILED: %d\n", res);
    }
    if (g_log_pipelines)
        for (uint32_t i = 0; i < createInfoCount; i++) log_pipeline(&modified[i], res);
    if (g_gpu_time && gt_n && res == VK_SUCCESS && pPipelines)
        for (uint32_t i = 0; i < createInfoCount; i++)
            if (pPipelines[i]) hash_put(g_pipeline_hash, (uint64_t)pPipelines[i], gt_hash[i], gt_hash[i] + 3, gt_n[i]);
    free(gt_hash);
    free(gt_n);
    for (uint32_t k = 0; k < fd_module_count; k++) d->fpDestroyShaderModule(device, fd_modules[k], NULL);
    for (uint32_t k = 0; k < fd_code_count; k++) free(fd_codes[k]);
    free(fd_stages);
    free(rast_copies);
    free(modified);
    return res;
}

// Only hooked for the GPU timing (and MALI_COMPAT_DUMP_HASH): records the compute shader's hash.
static VKAPI_ATTR VkResult VKAPI_CALL compat_CreateComputePipelines(
    VkDevice device, VkPipelineCache pipelineCache, uint32_t createInfoCount,
    const VkComputePipelineCreateInfo *pCreateInfos, const VkAllocationCallbacks *pAllocator, VkPipeline *pPipelines) {
    struct DeviceData *d = get_device_data(device);
    VkResult res = d->fpCreateComputePipelinesReal(device, pc_for(d, pipelineCache, createInfoCount), createInfoCount, pCreateInfos, pAllocator, pPipelines);
    for (uint32_t i = 0; i < createInfoCount; i++) {
        uint32_t hash[1], stage[1];
        uint32_t n = stage_hashes(&pCreateInfos[i].stage, 1, hash, stage);
        if (g_gpu_time && res == VK_SUCCESS && pPipelines && pPipelines[i])
            hash_put(g_pipeline_hash, (uint64_t)pPipelines[i], hash, stage, n);
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
    struct image_fixup fx;
    fixup_begin(d, pCreateInfo, &fx);
    bool is_bc = fx.is_bc;
    VkResult res = d->fpCreateImage(device, fx.use, pAllocator, pImage);
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
    if (res == VK_SUCCESS && g_gpu_time && pImage && *pImage)
        img_put(g_img_info, (struct ImgNode){(uint64_t)*pImage, *pImage, pCreateInfo->format, pCreateInfo->extent.width,
                pCreateInfo->extent.height, 0, pCreateInfo->mipLevels, 0, pCreateInfo->arrayLayers, pCreateInfo->samples,
                pCreateInfo->usage, NULL});
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
    if (g_gpu_time && res == VK_SUCCESS && pView) {
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
    return d->fpCreateBufferView(device, pCreateInfo, pAllocator, pView);
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

    return d->fpCreateBuffer(device, &mod, pAllocator, pBuffer);
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

static void mem_add(VkDeviceMemory m, VkDeviceSize size, uint32_t type) {
    struct MemInfo *e = calloc(1, sizeof *e);
    if (!e) return;
    e->memory = m;
    e->size = size;
    e->type = type;
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

static VKAPI_ATTR VkResult VKAPI_CALL compat_MapMemory2KHR(VkDevice device, const VkMemoryMapInfoKHR *pMemoryMapInfo,
                                                          void **ppData) {
    struct DeviceData *d = get_device_data(device);
    const VkMemoryMapPlacedInfoEXT *pl = find_struct(pMemoryMapInfo->pNext, VK_STRUCTURE_TYPE_MEMORY_MAP_PLACED_INFO_EXT);
    if (!(pMemoryMapInfo->flags & VK_MEMORY_MAP_PLACED_BIT_EXT) || !pl || !pl->pPlacedAddress) {
        if (!d->fpMapMemory2KHR) // VK_KHR_map_memory2 not enabled below (see vkGetDeviceProcAddr)
            return d->fpMapMemory(device, pMemoryMapInfo->memory, pMemoryMapInfo->offset, pMemoryMapInfo->size, 0, ppData);
        return d->fpMapMemory2KHR(device, pMemoryMapInfo, ppData);
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
    if (placed_unmap(d, device, pMemoryUnmapInfo->memory, pMemoryUnmapInfo->flags & VK_MEMORY_UNMAP_RESERVE_BIT_EXT,
                     pMemoryUnmapInfo))
        return VK_SUCCESS;
    if (!d->fpUnmapMemory2KHR) { d->fpUnmapMemory(device, pMemoryUnmapInfo->memory); return VK_SUCCESS; }
    return d->fpUnmapMemory2KHR(device, pMemoryUnmapInfo);
}

static VKAPI_ATTR void VKAPI_CALL compat_UnmapMemory(VkDevice device, VkDeviceMemory memory) {
    struct DeviceData *d = get_device_data(device);
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
    if (res == VK_SUCCESS) {
        atomic_fetch_add(&g_allocated_bytes, pAllocateInfo->allocationSize);
        mem_add(*pMemory, pAllocateInfo->allocationSize, mod.memoryTypeIndex);
    }
    return res;
}

static VKAPI_ATTR void VKAPI_CALL compat_FreeMemory(
    VkDevice device, VkDeviceMemory memory, const VkAllocationCallbacks *pAllocator) {
    struct DeviceData *d = get_device_data(device);
    struct MemInfo *e = memory ? mem_remove(memory) : NULL;
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
    if (!c) { get_device_data_by_cmd(commandBuffer)->fpCmdBindPipeline(commandBuffer, bindPoint, pipeline); return; }
    if (bindPoint == VK_PIPELINE_BIND_POINT_COMPUTE) c->compute_pipeline = pipeline;
    else if (bindPoint == VK_PIPELINE_BIND_POINT_GRAPHICS) c->gfx_pipeline = pipeline;
    c->dev->fpCmdBindPipeline(commandBuffer, bindPoint, pipeline);
}

static VKAPI_ATTR void VKAPI_CALL compat_CmdPushConstants(VkCommandBuffer commandBuffer, VkPipelineLayout layout,
                                                          VkShaderStageFlags stages, uint32_t offset, uint32_t size,
                                                          const void *pValues) {
    struct CmdBufData *c = cmd_data(commandBuffer);
    if (!c) {
        get_device_data_by_cmd(commandBuffer)->fpCmdPushConstants(commandBuffer, layout, stages, offset, size, pValues);
        return;
    }
    if (offset + size <= 256) {
        // Same layout/stages/range as an earlier record: update it and move it to the end (replay order = app order).
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
        memcpy(r->data, pValues, size);
    }
    c->dev->fpCmdPushConstants(commandBuffer, layout, stages, offset, size, pValues);
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
    for (uint32_t i = 0; i < cstate->push_count; i++) {
        const struct PushRec *pr = &cstate->push[i];
        d->fpCmdPushConstants(commandBuffer, pr->layout, pr->stages, pr->offset, pr->size, pr->data);
    }
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
    struct DeviceData *d = get_device_data(device);
    if (d) {
        pthread_mutex_lock(&g_cmdbuf_mutex);
        for (int b = 0; b < CMDBUF_BUCKETS; b++) {
            for (struct CmdBufData *e = g_cmdbufs[b]; e; e = e->next) {
                if (e->dev == d && e->pool == commandPool) {
                    for (uint32_t j = 0; j < e->staging_count; j++)
                        release_staging(d, &e->staging_resources[j]);
                    e->staging_count = 0;
                    e->compute_pipeline = VK_NULL_HANDLE;
                    e->descriptor_buffers = false;
                    e->push_count = 0;
                    e->cpd.layout = VK_NULL_HANDLE;
                    e->gpd.layout = VK_NULL_HANDLE;
                    e->gfx_pipeline = VK_NULL_HANDLE;
                    e->gt_rec = -1;
                    e->gt_src = VK_NULL_HANDLE;
                }
            }
        }
        pthread_mutex_unlock(&g_cmdbuf_mutex);
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
};
struct GtAgg { uint8_t kind; uint32_t w, h, colors; VkPipeline pipeline; uint64_t sum_ns, max_ns, draws; uint32_t count; };
#define GT_AGG 4096
struct GpuTime {
    struct DeviceData *d;
    VkQueryPool pool;
    double period_ns;
    pthread_mutex_t m;
    uint32_t next;
    struct GtRec rec[GT_SLOTS];
    struct GtAgg agg[GT_AGG];
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
};

// Calls per hooked entry point, printed with each report: shows which draw path the app really uses.
enum { GTC_DRAW, GTC_DRAW_INDEXED, GTC_DRAW_INDIRECT, GTC_DRAW_INDEXED_INDIRECT, GTC_DRAW_INDIRECT_COUNT,
       GTC_DRAW_INDEXED_INDIRECT_COUNT, GTC_DRAW_MULTI, GTC_DRAW_MULTI_INDEXED, GTC_MESH, GTC_MESH_INDIRECT,
       GTC_MESH_INDIRECT_COUNT, GTC_DISPATCH, GTC_DISPATCH_INDIRECT, GTC_DISPATCH_BASE, GTC_RENDERING, GTC_RENDERPASS, GTC_N };
static const char *const g_gtc_name[GTC_N] = {"Draw", "DrawIndexed", "DrawIndirect", "DrawIndexedIndirect", "DrawIndirectCount",
    "DrawIndexedIndirectCount", "DrawMultiEXT", "DrawMultiIndexedEXT", "DrawMeshTasksEXT", "DrawMeshTasksIndirectEXT",
    "DrawMeshTasksIndirectCountEXT", "Dispatch", "DispatchIndirect", "DispatchBase", "BeginRendering", "BeginRenderPass"};
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
                           : h.stages[s] == VK_SHADER_STAGE_COMPUTE_BIT ? "cs" : "st";
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

static void gt_line_rec(FILE *f, const char *tag, const struct GtRec *r, double age_s, double dur_ms) {
    char desc[160], c0[200], dp[200], src[200];
    struct ImgNode nc, nd, ns;
    gt_pipe_desc(r->pipeline, desc, sizeof desc);
    gt_view_desc("color0", r->color0, &nc, c0, sizeof c0);
    gt_view_desc("depth", r->depth, &nd, dp, sizeof dp);
    gt_view_desc("src", r->src, &ns, src, sizeof src);
    bool same = ns.image && (ns.image == nc.image || ns.image == nd.image);
    fprintf(f, "%s %s %ux%u c%u draws %u (first: %u verts x %u inst) rlayers %u, %s %.1f %s, %s%s%s%s%s\n", tag,
            r->kind == GT_PASS ? "pass" : "dispatch", r->w, r->h, r->colors, r->draws, r->vcount, r->icount, r->rlayers,
            dur_ms >= 0 ? "took" : "recorded", dur_ms >= 0 ? dur_ms : age_s, dur_ms >= 0 ? "ms" : "s ago", desc, c0, dp, src,
            same ? "  SAME-IMAGE" : "");
}

// MALI_COMPAT_GT_TRACE: does this record's pipeline have the traced fragment/compute shader?
static bool gt_traced(const struct GtRec *r) {
    struct HashNode h;
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
        fprintf(f, " | suspending passes %llu, resuming %llu", (unsigned long long)atomic_load(&g_susp_passes),
                (unsigned long long)atomic_load(&g_resu_passes));
        fprintf(f, " | firstInstance!=0 draws %llu, divisor!=1 pipelines %llu, vertex inputs %llu",
                (unsigned long long)atomic_load(&g_div_draws), (unsigned long long)atomic_load(&g_div_pipes),
                (unsigned long long)atomic_load(&g_div_vi));
        if (g_fake_occlusion)
            fprintf(f, " | faked occlusion copies %llu, reads %llu", (unsigned long long)atomic_load(&g_occl_copies),
                    (unsigned long long)atomic_load(&g_occl_reads));
        fprintf(f, "\n");
        for (uint32_t i = 0; i < n && i < 25; i++) {
            char desc[160];
            gt_pipe_desc(v[i].pipeline, desc, sizeof desc);
            fprintf(f, "%5.1f%%  avg %7.3f ms  max %7.2f ms  n %5u  %-8s %4ux%-4u c%u draws %5.1f  %s\n",
                    total ? 100.0 * v[i].sum_ns / total : 0, v[i].sum_ns / 1e6 / v[i].count, v[i].max_ns / 1e6, v[i].count,
                    v[i].kind == GT_PASS ? "pass" : "dispatch", v[i].w, v[i].h, v[i].colors,
                    (double)v[i].draws / v[i].count, desc);
        }
        fclose(f);
    }
    free(v);
    memset(g->agg, 0, sizeof g->agg);
}

// Dispatches are grouped per pipeline; passes per render area and colour target count with draws or without (a pass
// with draws runs many pipelines, so grouping it by its first one scattered the heavy passes; "pipe" is then the
// first draw's pipeline of the last pass seen in the group).
static void gt_add(struct GpuTime *g, const struct GtRec *r, uint64_t ns) {
    VkPipeline kp = r->kind == GT_PASS ? (r->draws ? (VkPipeline)(uintptr_t)1 : VK_NULL_HANDLE) : r->pipeline;
    uint64_t key = (uint64_t)(uintptr_t)kp ^ ((uint64_t)r->kind << 56) ^ ((uint64_t)r->w << 36) ^ ((uint64_t)r->h << 16) ^ r->colors;
    uint32_t b = handle_bucket(key, GT_AGG);
    for (uint32_t probe = 0; probe < GT_AGG; probe++, b = (b + 1) % GT_AGG) {
        struct GtAgg *a = &g->agg[b];
        bool a_draws = a->kind == GT_PASS && a->draws, match_kp = r->kind == GT_PASS ? (a_draws == (r->draws != 0)) : a->pipeline == r->pipeline;
        if (!a->count) {
            *a = (struct GtAgg){r->kind, r->w, r->h, r->colors, r->pipeline, 0, 0, 0, 0};
        } else if (a->kind != r->kind || a->w != r->w || a->h != r->h || a->colors != r->colors || !match_kp) {
            continue;
        }
        if (r->kind == GT_PASS && r->pipeline) a->pipeline = r->pipeline;
        a->sum_ns += ns;
        if (ns > a->max_ns) a->max_ns = ns;
        a->draws += r->draws;
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
                    if (f) { gt_line_rec(f, "TRACE", r, age, ns / 1e6); fclose(f); }
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

static void gt_render_begin(VkCommandBuffer cmd, uint32_t w, uint32_t h, uint32_t colors, const VkRenderingInfo *ri) {
    struct CmdBufData *c = cmd_data(cmd);
    if (!c || !c->dev->gt) return;
    c->gt_rec = gt_begin(c, cmd, GT_PASS, w, h, colors, VK_NULL_HANDLE);
    if (c->gt_rec >= 0 && ri) {
        struct GtRec *r = &c->dev->gt->rec[c->gt_rec];
        r->rlayers = ri->layerCount;
        if (ri->colorAttachmentCount && ri->pColorAttachments) r->color0 = ri->pColorAttachments[0].imageView;
        if (ri->pDepthAttachment) r->depth = ri->pDepthAttachment->imageView;
    }
}

static void gt_render_end(VkCommandBuffer cmd) {
    struct CmdBufData *c = cmd_data(cmd);
    if (c && c->dev->gt && c->gt_rec >= 0) { gt_end(c, cmd, c->gt_rec); c->gt_rec = -1; }
}

static void gt_draw(struct CmdBufData *c, uint32_t vcount, uint32_t icount) {
    if (!c || c->gt_rec < 0 || !c->dev->gt) return;
    struct GtRec *r = &c->dev->gt->rec[c->gt_rec];
    if (!r->draws++) { r->pipeline = c->gfx_pipeline; r->vcount = vcount; r->icount = icount; r->src = c->gt_src; }
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
        ri = &plain;
    }
    gt_render_begin(cmd, ri->renderArea.extent.width, ri->renderArea.extent.height, ri->colorAttachmentCount, ri);
    get_device_data_by_cmd(cmd)->gt->fpCmdBeginRendering(cmd, ri);
}
static VKAPI_ATTR void VKAPI_CALL compat_CmdEndRendering(VkCommandBuffer cmd) {
    get_device_data_by_cmd(cmd)->gt->fpCmdEndRendering(cmd);
    gt_render_end(cmd);
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
    int32_t i = gt_begin(c, cmd, GT_DISPATCH, x, y, z, c->compute_pipeline); // "WxH cZ" = the group counts
    c->dev->fpCmdDispatch(cmd, x, y, z);
    gt_end(c, cmd, i);
}
static VKAPI_ATTR void VKAPI_CALL compat_CmdDispatchIndirect(VkCommandBuffer cmd, VkBuffer buf, VkDeviceSize off) {
    GTC(GTC_DISPATCH_INDIRECT);
    struct CmdBufData *c = cmd_data(cmd);
    if (!c) { get_device_data_by_cmd(cmd)->gt->fpCmdDispatchIndirect(cmd, buf, off); return; }
    int32_t i = gt_begin(c, cmd, GT_DISPATCH, 0, 0, 0, c->compute_pipeline); // indirect: group counts unknown (0x0)
    c->dev->gt->fpCmdDispatchIndirect(cmd, buf, off);
    gt_end(c, cmd, i);
}
static VKAPI_ATTR void VKAPI_CALL compat_CmdDraw(VkCommandBuffer cmd, uint32_t vc, uint32_t ic, uint32_t fv, uint32_t fi) {
    GTC(GTC_DRAW);
    if (fi && ic) atomic_fetch_add(&g_div_draws, 1);
    struct CmdBufData *c = cmd_data(cmd);
    gt_draw(c, vc, ic);
    (c ? c->dev : get_device_data_by_cmd(cmd))->gt->fpCmdDraw(cmd, vc, ic, fv, fi);
}
static VKAPI_ATTR void VKAPI_CALL compat_CmdDrawIndexed(VkCommandBuffer cmd, uint32_t xc, uint32_t ic, uint32_t fx, int32_t vo,
                                                        uint32_t fi) {
    GTC(GTC_DRAW_INDEXED);
    if (fi && ic) atomic_fetch_add(&g_div_draws, 1);
    struct CmdBufData *c = cmd_data(cmd);
    gt_draw(c, xc, ic);
    (c ? c->dev : get_device_data_by_cmd(cmd))->gt->fpCmdDrawIndexed(cmd, xc, ic, fx, vo, fi);
}
static VKAPI_ATTR void VKAPI_CALL compat_CmdDrawIndirect(VkCommandBuffer cmd, VkBuffer b, VkDeviceSize o, uint32_t n, uint32_t s) {
    GTC(GTC_DRAW_INDIRECT);
    struct CmdBufData *c = cmd_data(cmd);
    gt_draw(c, 0, 0);
    (c ? c->dev : get_device_data_by_cmd(cmd))->gt->fpCmdDrawIndirect(cmd, b, o, n, s);
}
static VKAPI_ATTR void VKAPI_CALL compat_CmdDrawIndexedIndirect(VkCommandBuffer cmd, VkBuffer b, VkDeviceSize o, uint32_t n,
                                                                uint32_t s) {
    GTC(GTC_DRAW_INDEXED_INDIRECT);
    struct CmdBufData *c = cmd_data(cmd);
    gt_draw(c, 0, 0);
    (c ? c->dev : get_device_data_by_cmd(cmd))->gt->fpCmdDrawIndexedIndirect(cmd, b, o, n, s);
}
static VKAPI_ATTR void VKAPI_CALL compat_CmdDrawIndirectCount(VkCommandBuffer cmd, VkBuffer b, VkDeviceSize o, VkBuffer cb,
                                                              VkDeviceSize co, uint32_t n, uint32_t s) {
    GTC(GTC_DRAW_INDIRECT_COUNT);
    struct CmdBufData *c = cmd_data(cmd);
    gt_draw(c, 0, 0);
    (c ? c->dev : get_device_data_by_cmd(cmd))->gt->fpCmdDrawIndirectCount(cmd, b, o, cb, co, n, s);
}
static VKAPI_ATTR void VKAPI_CALL compat_CmdDrawIndexedIndirectCount(VkCommandBuffer cmd, VkBuffer b, VkDeviceSize o, VkBuffer cb,
                                                                     VkDeviceSize co, uint32_t n, uint32_t s) {
    GTC(GTC_DRAW_INDEXED_INDIRECT_COUNT);
    struct CmdBufData *c = cmd_data(cmd);
    gt_draw(c, 0, 0);
    (c ? c->dev : get_device_data_by_cmd(cmd))->gt->fpCmdDrawIndexedIndirectCount(cmd, b, o, cb, co, n, s);
}

static VKAPI_ATTR void VKAPI_CALL compat_CmdDrawMultiEXT(VkCommandBuffer cmd, uint32_t n, const VkMultiDrawInfoEXT *info,
                                                         uint32_t ic, uint32_t fi, uint32_t stride) {
    GTC(GTC_DRAW_MULTI);
    struct CmdBufData *c = cmd_data(cmd);
    gt_draw(c, n && info ? info->vertexCount : 0, ic);
    (c ? c->dev : get_device_data_by_cmd(cmd))->gt->fpCmdDrawMultiEXT(cmd, n, info, ic, fi, stride);
}
static VKAPI_ATTR void VKAPI_CALL compat_CmdDrawMultiIndexedEXT(VkCommandBuffer cmd, uint32_t n, const VkMultiDrawIndexedInfoEXT *info,
                                                                uint32_t ic, uint32_t fi, uint32_t stride, const int32_t *vo) {
    GTC(GTC_DRAW_MULTI_INDEXED);
    struct CmdBufData *c = cmd_data(cmd);
    gt_draw(c, n && info ? info->indexCount : 0, ic);
    (c ? c->dev : get_device_data_by_cmd(cmd))->gt->fpCmdDrawMultiIndexedEXT(cmd, n, info, ic, fi, stride, vo);
}
static VKAPI_ATTR void VKAPI_CALL compat_CmdDrawMeshTasksEXT(VkCommandBuffer cmd, uint32_t x, uint32_t y, uint32_t z) {
    GTC(GTC_MESH);
    struct CmdBufData *c = cmd_data(cmd);
    gt_draw(c, x, y);
    (c ? c->dev : get_device_data_by_cmd(cmd))->gt->fpCmdDrawMeshTasksEXT(cmd, x, y, z);
}
static VKAPI_ATTR void VKAPI_CALL compat_CmdDrawMeshTasksIndirectEXT(VkCommandBuffer cmd, VkBuffer b, VkDeviceSize o, uint32_t n, uint32_t s) {
    GTC(GTC_MESH_INDIRECT);
    struct CmdBufData *c = cmd_data(cmd);
    gt_draw(c, 0, 0);
    (c ? c->dev : get_device_data_by_cmd(cmd))->gt->fpCmdDrawMeshTasksIndirectEXT(cmd, b, o, n, s);
}
static VKAPI_ATTR void VKAPI_CALL compat_CmdDrawMeshTasksIndirectCountEXT(VkCommandBuffer cmd, VkBuffer b, VkDeviceSize o, VkBuffer cb,
                                                                          VkDeviceSize co, uint32_t n, uint32_t s) {
    GTC(GTC_MESH_INDIRECT_COUNT);
    struct CmdBufData *c = cmd_data(cmd);
    gt_draw(c, 0, 0);
    (c ? c->dev : get_device_data_by_cmd(cmd))->gt->fpCmdDrawMeshTasksIndirectCountEXT(cmd, b, o, cb, co, n, s);
}
static VKAPI_ATTR void VKAPI_CALL compat_CmdDispatchBase(VkCommandBuffer cmd, uint32_t bx, uint32_t by, uint32_t bz, uint32_t x,
                                                         uint32_t y, uint32_t z) {
    GTC(GTC_DISPATCH_BASE);
    struct CmdBufData *c = cmd_data(cmd);
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
    g_depth_nocomp = getenv("MALI_COMPAT_DEPTH_NOCOMP") && strcmp(getenv("MALI_COMPAT_DEPTH_NOCOMP"), "0") != 0;
    g_force_comp = getenv("MALI_COMPAT_FORCE_COMP") && strcmp(getenv("MALI_COMPAT_FORCE_COMP"), "0") != 0;
    g_no_susres = getenv("MALI_COMPAT_NO_SUSPEND_RESUME") && strcmp(getenv("MALI_COMPAT_NO_SUSPEND_RESUME"), "0") != 0;
    g_pcache_on = getenv("MALI_COMPAT_PIPELINE_CACHE") && strcmp(getenv("MALI_COMPAT_PIPELINE_CACHE"), "0") != 0;
    g_force_coherent = getenv("MALI_COMPAT_FORCE_COHERENT") && strcmp(getenv("MALI_COMPAT_FORCE_COHERENT"), "0") != 0;
    g_fake_occlusion = getenv("MALI_COMPAT_FAKE_OCCLUSION") && strcmp(getenv("MALI_COMPAT_FAKE_OCCLUSION"), "0") != 0;
    g_vendor_id = getenv("MALI_COMPAT_VENDOR_ID") ? (uint32_t)strtoul(getenv("MALI_COMPAT_VENDOR_ID"), NULL, 16) : 0;
    // on by default: the write it removes is discarded by the spec anyway (MALI_COMPAT_FRAGDEPTH_FIX=0 turns it off)
    g_fragdepth_fix = !getenv("MALI_COMPAT_FRAGDEPTH_FIX") || strcmp(getenv("MALI_COMPAT_FRAGDEPTH_FIX"), "0") != 0;
    g_gt_trace = getenv("MALI_COMPAT_GT_TRACE") ? (uint32_t)strtoul(getenv("MALI_COMPAT_GT_TRACE"), NULL, 16) : 0;
    g_dump_hash_count = 0;
    for (const char *h = getenv("MALI_COMPAT_DUMP_HASH"); h && *h && g_dump_hash_count < 16;) {
        char *end;
        unsigned long v = strtoul(h, &end, 16);
        if (end == h) break;
        g_dump_hashes[g_dump_hash_count++] = (uint32_t)v;
        h = *end == ',' ? end + 1 : end;
    }

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
    const char **ext_names = malloc((pCreateInfo->enabledExtensionCount + 3) * sizeof *ext_names);
    if (!ext_names) return VK_ERROR_OUT_OF_HOST_MEMORY;
    mod_ci.enabledExtensionCount = 0;
    bool div_ext = false, div_khr = false;
    for (uint32_t i = 0; i < pCreateInfo->enabledExtensionCount; i++) {
        const char *e = pCreateInfo->ppEnabledExtensionNames[i];
        if (!strcmp(e, "VK_EXT_vertex_attribute_divisor")) { div_ext = true; continue; } // the vendor has only KHR
        div_khr |= !strcmp(e, "VK_KHR_vertex_attribute_divisor");
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
    mod_ci.ppEnabledExtensionNames = ext_names;
    struct unlinked placed_feat = unlink_struct(&mod_ci, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAP_MEMORY_PLACED_FEATURES_EXT);
    VkPhysicalDeviceFeatures safe_features;
    if (mod_ci.pEnabledFeatures) {
        safe_features = *mod_ci.pEnabledFeatures;
        safe_features.textureCompressionBC = VK_FALSE;
        safe_features.shaderClipDistance = VK_FALSE;
        safe_features.shaderCullDistance = VK_FALSE;
        safe_features.fillModeNonSolid = VK_FALSE;
        mod_ci.pEnabledFeatures = &safe_features;
    }

    // Mask features2 if chained, and check if bufferDeviceAddress is already in pNext
    VkBool32 saved_bc = VK_FALSE, saved_clip = VK_FALSE, saved_cull = VK_FALSE, saved_fill = VK_FALSE;
    VkPhysicalDeviceFeatures2 *f2 = NULL;
    VkPhysicalDeviceVulkan12Features *v12_feat = NULL;
    VkBool32 saved_v12_bda = VK_FALSE;
    VkPhysicalDeviceBufferDeviceAddressFeatures *existing_bda = NULL;
    VkBool32 saved_existing_bda = VK_FALSE;

    VkBaseOutStructure *node = (VkBaseOutStructure *)mod_ci.pNext;
    while (node) {
        if (node->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2) {
            f2 = (VkPhysicalDeviceFeatures2 *)node;
            saved_bc = f2->features.textureCompressionBC;
            saved_clip = f2->features.shaderClipDistance;
            saved_cull = f2->features.shaderCullDistance;
            saved_fill = f2->features.fillModeNonSolid;
            f2->features.textureCompressionBC = VK_FALSE;
            f2->features.shaderClipDistance = VK_FALSE;
            f2->features.shaderCullDistance = VK_FALSE;
            f2->features.fillModeNonSolid = VK_FALSE;
        } else if (node->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES) {
            v12_feat = (VkPhysicalDeviceVulkan12Features *)node;
            saved_v12_bda = v12_feat->bufferDeviceAddress;
            v12_feat->bufferDeviceAddress = VK_TRUE;
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
    if (app_div) app_div->vertexAttributeInstanceRateZeroDivisor = app_zero_div;
    if (app_icc) app_icc->imageCompressionControl = app_icc_saved;
    relink(placed_feat);
    free(ext_names);

    // Restore mutated pNext fields
    if (f2) {
        f2->features.textureCompressionBC = saved_bc;
        f2->features.shaderClipDistance = saved_clip;
        f2->features.shaderCullDistance = saved_cull;
        f2->features.fillModeNonSolid = saved_fill;
    }
    if (v12_feat) v12_feat->bufferDeviceAddress = saved_v12_bda;
    if (existing_bda) existing_bda->bufferDeviceAddress = saved_existing_bda;
    if (v12_feat) v12_feat->hostQueryReset = saved_v12_hqr;
    if (app_hqr) app_hqr->hostQueryReset = saved_app_hqr;

    if (res != VK_SUCCESS) {
        LOG("fpCreateDevice failed: %d", res);
        return res;
    }

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
        LOAD_DEV(BindBufferMemory);
        LOAD_DEV(GetBufferMemoryRequirements);
        LOAD_DEV(GetBufferDeviceAddress);
        if (!d->fpGetBufferDeviceAddress)
            d->fpGetBufferDeviceAddress = (PFN_vkGetBufferDeviceAddress)fpGetDeviceProcAddr(*pDevice, "vkGetBufferDeviceAddressKHR");
        LOAD_DEV(CmdBindPipeline);
        LOAD_DEV(CmdPushConstants);
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
        if (g_log_images && want_icc) {
            d->fpGetImageSubresourceLayout2KHR = (PFN_vkGetImageSubresourceLayout2KHR)fpGetDeviceProcAddr(*pDevice, "vkGetImageSubresourceLayout2KHR");
            if (!d->fpGetImageSubresourceLayout2KHR)
                d->fpGetImageSubresourceLayout2KHR = (PFN_vkGetImageSubresourceLayout2KHR)fpGetDeviceProcAddr(*pDevice, "vkGetImageSubresourceLayout2EXT");
        }
        d->fpCreateComputePipelinesReal = (PFN_vkCreateComputePipelines)fpGetDeviceProcAddr(*pDevice, "vkCreateComputePipelines");
        #undef LOAD_DEV
        if (g_gpu_time) gt_init(d, fpGetDeviceProcAddr);
        pc_init(d);
        if (inst && inst->fpGetPhysicalDeviceMemoryProperties) {
            VkPhysicalDeviceMemoryProperties mp;
            inst->fpGetPhysicalDeviceMemoryProperties(physicalDevice, &mp);
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
    }
    pthread_mutex_unlock(&g_table_mutex);

    LOG("created VkDevice %p (key=%p)", (void *)*pDevice, *pDevice ? *(void **)*pDevice : NULL);
    return VK_SUCCESS;
}

static VKAPI_ATTR void VKAPI_CALL compat_DestroyDevice(
    VkDevice device, const VkAllocationCallbacks *pAllocator) {
    struct DeviceData *d = get_device_data(device);
    if (d) {
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
    #define MATCH_DEV(fn) if (!strcmp(pName, "vk" #fn)) return (PFN_vkVoidFunction)compat_##fn
    // GPU timing hooks: only while the switch is on, so the normal path calls the driver directly.
    #define MATCH_DEV_KHR(fn) if (!strcmp(pName, "vk" #fn) || !strcmp(pName, "vk" #fn "KHR")) return (PFN_vkVoidFunction)compat_##fn
    if (g_gpu_time) {
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
            #undef MATCH_GT_OPT
        }
    }
    if (!strcmp(pName, "vkCmdPushDescriptorSetKHR") || !strcmp(pName, "vkCmdPushDescriptorSet")) {
        struct DeviceData *dd = get_device_data(device);
        if (dd && dd->fpCmdPushDescriptorSetKHR && (g_gpu_time || g_vendor_id)) return (PFN_vkVoidFunction)compat_CmdPushDescriptorSetKHR;
    }
    if (g_gpu_time || g_dump_hash_count || g_pcache_on) MATCH_DEV(CreateComputePipelines);
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

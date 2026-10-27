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
    PFN_vkGetImageSubresourceLayout2KHR fpGetImageSubresourceLayout2KHR; // only with g_log_images

    // BC compute decoder resources
    VkPipelineLayout bc_layout;
    VkPipeline bc_pipelines[4];
    VkPipeline bc_pipelines_db[4]; // with DESCRIPTOR_BUFFER_BIT, for command buffers that bound descriptor buffers
    bool descriptor_buffer;        // the app enabled VkPhysicalDeviceDescriptorBufferFeaturesEXT::descriptorBuffer
    PFN_vkCmdBindDescriptorBuffersEXT fpCmdBindDescriptorBuffersEXT;
    uint32_t device_local_memory_index;
    bool bc_init_done;
    struct StagingPoolNode *staging_pool; // free decode staging buffers (g_staging_pool_mutex)
    VkDeviceSize staging_pool_bytes;
    bool afrc; // VK_EXT_image_compression_control enabled by the layer for the BC shadows
    bool zero_divisor_emulated; // divisor 0 -> UINT32_MAX (the vendor lacks vertexAttributeInstanceRateZeroDivisor)
    PFN_vkCmdSetVertexInputEXT fpCmdSetVertexInputEXT;
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
// Device Level Interceptions
// -----------------------------------------------------------------------------
static VKAPI_ATTR VkResult VKAPI_CALL compat_CreateShaderModule(
    VkDevice device, const VkShaderModuleCreateInfo *pCreateInfo,
    const VkAllocationCallbacks *pAllocator, VkShaderModule *pShaderModule) {
    struct DeviceData *d = get_device_data(device);
    size_t patched_bytes = 0;
    uint32_t *patched = compat_patch_spirv(pCreateInfo->pCode, pCreateInfo->codeSize, &patched_bytes);

    if (patched) {
        VkShaderModuleCreateInfo mod_ci = *pCreateInfo;
        mod_ci.pCode = patched;
        mod_ci.codeSize = patched_bytes;
        VkResult r = d->fpCreateShaderModule(device, &mod_ci, pAllocator, pShaderModule);
        free(patched);
        return r;
    }
    return d->fpCreateShaderModule(device, pCreateInfo, pAllocator, pShaderModule);
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
    dump_inflight_pipeline(createInfoCount, modified);
    VkResult res = d->fpCreateGraphicsPipelines(device, pipelineCache, createInfoCount, modified, pAllocator, pPipelines);
    for (uint32_t k = 0; k < zero_div_count; k++) zero_div[k]->divisor = 0;
    if (res != VK_SUCCESS) {
        fprintf(stderr, "mali-compat: vkCreateGraphicsPipelines FAILED: %d\n", res);
    }
    if (g_log_pipelines)
        for (uint32_t i = 0; i < createInfoCount; i++) log_pipeline(&modified[i], res);
    free(rast_copies);
    free(modified);
    return res;
}

static VKAPI_ATTR void VKAPI_CALL compat_CmdSetVertexInputEXT(
    VkCommandBuffer commandBuffer, uint32_t bindingCount, const VkVertexInputBindingDescription2EXT *pBindings,
    uint32_t attributeCount, const VkVertexInputAttributeDescription2EXT *pAttributes) {
    struct DeviceData *d = get_device_data_by_cmd(commandBuffer);
    VkVertexInputBindingDescription2EXT stack[32];
    const VkVertexInputBindingDescription2EXT *b = pBindings;
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
        fprintf(f, "fmt %d %ux%ux%u mips %u layers %u samples %d tiling %d usage 0x%x flags 0x%x%s comp 0x%x rate 0x%x\n",
                ci->format, ci->extent.width, ci->extent.height, ci->extent.depth, ci->mipLevels, ci->arrayLayers,
                ci->samples, ci->tiling, ci->usage, ci->flags, bc ? " bc-shadow" : "", cp.imageCompressionFlags,
                cp.imageCompressionFixedRateFlags);
        fclose(f);
    }
    pthread_mutex_unlock(&m);
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
    d->fpDestroyImage(device, image, pAllocator);
}

static VKAPI_ATTR VkResult VKAPI_CALL compat_CreateImageView(
    VkDevice device, const VkImageViewCreateInfo *pCreateInfo,
    const VkAllocationCallbacks *pAllocator, VkImageView *pView) {
    struct DeviceData *d = get_device_data(device);

    struct TrackedBCImage tracked;
    if (find_bc_image(pCreateInfo->image, &tracked) && panvk_format_is_bc(pCreateInfo->format)) {
        VkImageViewCreateInfo mod = *pCreateInfo;
        mod.format = panvk_bc_decoded_format(pCreateInfo->format);
        return d->fpCreateImageView(device, &mod, pAllocator, pView);
    }

    return d->fpCreateImageView(device, pCreateInfo, pAllocator, pView);
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

    VkResult res = d->fpAllocateMemory(device, &mod, pAllocator, pMemory);
    if (res == VK_SUCCESS) {
        atomic_fetch_add(&g_allocated_bytes, pAllocateInfo->allocationSize);
        mem_add(*pMemory, pAllocateInfo->allocationSize, pAllocateInfo->memoryTypeIndex);
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
        d->fpCmdSetVertexInputEXT = (PFN_vkCmdSetVertexInputEXT)fpGetDeviceProcAddr(*pDevice, "vkCmdSetVertexInputEXT");
        if (g_log_images && want_icc) {
            d->fpGetImageSubresourceLayout2KHR = (PFN_vkGetImageSubresourceLayout2KHR)fpGetDeviceProcAddr(*pDevice, "vkGetImageSubresourceLayout2KHR");
            if (!d->fpGetImageSubresourceLayout2KHR)
                d->fpGetImageSubresourceLayout2KHR = (PFN_vkGetImageSubresourceLayout2KHR)fpGetDeviceProcAddr(*pDevice, "vkGetImageSubresourceLayout2EXT");
        }
        #undef LOAD_DEV
        if (inst && inst->fpGetPhysicalDeviceMemoryProperties) {
            VkPhysicalDeviceMemoryProperties mp;
            inst->fpGetPhysicalDeviceMemoryProperties(physicalDevice, &mp);
            for (uint32_t i = 0; i < mp.memoryTypeCount; i++)
                if (mp.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) d->coherent_types |= 1u << i;
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

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <string.h>

// SPIR-V OpCodes
#define SpvOpNop 0
#define SpvOpEntryPoint 15
#define SpvOpCapability 17
#define SpvOpTypePointer 32
#define SpvOpVariable 59
#define SpvOpDecorate 71
#define SpvOpMemberDecorate 72
#define SpvOpAccessChain 65
#define SpvOpInBoundsAccessChain 66

// Capabilities
#define SpvCapabilityClipDistance 32
#define SpvCapabilityCullDistance 33

// Built-ins
#define SpvBuiltInClipDistance 3
#define SpvBuiltInCullDistance 4

// Storage classes
#define SpvStorageClassInput 1
#define SpvStorageClassOutput 3
#define SpvStorageClassPrivate 6

// Decorations
#define SpvDecorationBuiltIn 11

/*
 * Returns NULL if unchanged. If modified, returns malloc'd buffer and sets *pOutSize.
 */
uint32_t *compat_patch_spirv(const uint32_t *inWords, size_t inBytes, size_t *pOutBytes) {
    if (!inWords || inBytes < 20) return NULL;
    size_t wordCount = inBytes / 4;
    if (inWords[0] != 0x07230203) return NULL; // SPIR-V Magic

    // Fast scan for OpCapability ClipDistance / CullDistance
    bool hasClipOrCull = false;
    size_t offset = 5;
    while (offset < wordCount) {
        uint32_t inst = inWords[offset];
        uint16_t op = (uint16_t)(inst & 0xFFFFu);
        uint16_t len = (uint16_t)(inst >> 16);
        if (len == 0 || offset + len > wordCount) break;

        if (op == SpvOpCapability) {
            uint32_t cap = inWords[offset + 1];
            if (cap == SpvCapabilityClipDistance || cap == SpvCapabilityCullDistance) {
                hasClipOrCull = true;
                break;
            }
        } else if (op != SpvOpCapability) {
            // Capabilities appear at the beginning of SPIR-V module
            // Once we pass capabilities, stop fast scan
            break;
        }
        offset += len;
    }

    if (!hasClipOrCull) return NULL;

    // Allocate copy of words for rewriting in-place
    uint32_t *words = (uint32_t *)malloc(inBytes);
    if (!words) return NULL;
    memcpy(words, inWords, inBytes);

    // Pass 1: Collect variable IDs decorated with BuiltIn ClipDistance or CullDistance
    #define MAX_BAD_VARS 32
    uint32_t bad_vars[MAX_BAD_VARS];
    uint32_t bad_var_count = 0;

    offset = 5;
    while (offset < wordCount) {
        uint32_t inst = words[offset];
        uint16_t op = (uint16_t)(inst & 0xFFFFu);
        uint16_t len = (uint16_t)(inst >> 16);
        if (len == 0 || offset + len > wordCount) break;

        if (op == SpvOpDecorate && len >= 4) {
            uint32_t target = words[offset + 1];
            uint32_t decor = words[offset + 2];
            uint32_t builtin = words[offset + 3];
            if (decor == SpvDecorationBuiltIn &&
                (builtin == SpvBuiltInClipDistance || builtin == SpvBuiltInCullDistance)) {
                if (bad_var_count < MAX_BAD_VARS) {
                    bool already = false;
                    for (uint32_t b = 0; b < bad_var_count; b++) {
                        if (bad_vars[b] == target) { already = true; break; }
                    }
                    if (!already) bad_vars[bad_var_count++] = target;
                }
            }
        }
        offset += len;
    }

    // Pass 2: Collect ptr_type_ids used by bad_vars
    uint32_t bad_ptr_types[MAX_BAD_VARS];
    uint32_t bad_ptr_count = 0;

    offset = 5;
    while (offset < wordCount) {
        uint32_t inst = words[offset];
        uint16_t op = (uint16_t)(inst & 0xFFFFu);
        uint16_t len = (uint16_t)(inst >> 16);
        if (len == 0 || offset + len > wordCount) break;

        if (op == SpvOpVariable && len >= 4) {
            uint32_t result_type = words[offset + 1];
            uint32_t result_id = words[offset + 2];
            for (uint32_t b = 0; b < bad_var_count; b++) {
                if (bad_vars[b] == result_id) {
                    if (bad_ptr_count < MAX_BAD_VARS) {
                        bad_ptr_types[bad_ptr_count++] = result_type;
                    }
                    // Rewrite OpVariable storage class to Private
                    words[offset + 3] = SpvStorageClassPrivate;
                    break;
                }
            }
        }
        offset += len;
    }

    // Pass 3: Rewrite SPIR-V instructions:
    // - OpCapability ClipDistance/CullDistance -> OpNop
    // - OpDecorate / OpMemberDecorate for BuiltIn ClipDistance/CullDistance -> OpNop
    // - OpTypePointer for bad_ptr_types -> rewrite storage class to Private
    // - OpEntryPoint -> remove bad_vars from interface list
    offset = 5;
    while (offset < wordCount) {
        uint32_t inst = words[offset];
        uint16_t op = (uint16_t)(inst & 0xFFFFu);
        uint16_t len = (uint16_t)(inst >> 16);
        if (len == 0 || offset + len > wordCount) break;

        if (op == SpvOpCapability && len >= 2) {
            uint32_t cap = words[offset + 1];
            if (cap == SpvCapabilityClipDistance || cap == SpvCapabilityCullDistance) {
                // Replace with OpNop
                words[offset] = ((uint32_t)len << 16) | SpvOpNop;
            }
        } else if (op == SpvOpDecorate && len >= 3) {
            uint32_t target = words[offset + 1];
            uint32_t decor = words[offset + 2];
            if (decor == SpvDecorationBuiltIn && len >= 4) {
                uint32_t builtin = words[offset + 3];
                if (builtin == SpvBuiltInClipDistance || builtin == SpvBuiltInCullDistance) {
                    words[offset] = ((uint32_t)len << 16) | SpvOpNop;
                }
            } else {
                for (uint32_t b = 0; b < bad_var_count; b++) {
                    if (bad_vars[b] == target) {
                        words[offset] = ((uint32_t)len << 16) | SpvOpNop;
                        break;
                    }
                }
            }
        } else if (op == SpvOpMemberDecorate && len >= 5) {
            uint32_t decor = words[offset + 3];
            uint32_t builtin = words[offset + 4];
            if (decor == SpvDecorationBuiltIn &&
                (builtin == SpvBuiltInClipDistance || builtin == SpvBuiltInCullDistance)) {
                words[offset] = ((uint32_t)len << 16) | SpvOpNop;
            }
        } else if (op == SpvOpTypePointer && len >= 3) {
            uint32_t type_id = words[offset + 1];
            for (uint32_t p = 0; p < bad_ptr_count; p++) {
                if (bad_ptr_types[p] == type_id) {
                    words[offset + 2] = SpvStorageClassPrivate;
                    break;
                }
            }
        } else if (op == SpvOpEntryPoint && len >= 4) {
            // Find where interface list starts:
            // OpEntryPoint <model> <id> <name null-terminated> <interfaces...>
            const char *name_str = (const char *)&words[offset + 3];
            size_t name_len = strlen(name_str) + 1;
            size_t name_words = (name_len + 3) / 4;
            size_t iface_start = offset + 3 + name_words;

            if (iface_start < offset + len) {
                size_t write_idx = iface_start;
                for (size_t read_idx = iface_start; read_idx < offset + len; read_idx++) {
                    uint32_t iface_id = words[read_idx];
                    bool is_bad = false;
                    for (uint32_t b = 0; b < bad_var_count; b++) {
                        if (bad_vars[b] == iface_id) { is_bad = true; break; }
                    }
                    if (!is_bad) {
                        words[write_idx++] = iface_id;
                    }
                }
                uint32_t removed = (uint32_t)((offset + len) - write_idx);
                if (removed > 0) {
                    uint16_t new_len = len - (uint16_t)removed;
                    words[offset] = ((uint32_t)new_len << 16) | SpvOpEntryPoint;
                    // Pad with OpNop so instruction alignment is preserved
                    words[write_idx] = ((uint32_t)removed << 16) | SpvOpNop;
                }
            }
        }
        offset += len;
    }

    *pOutBytes = inBytes;
    return words;
}

int main(void) {
    // Construct a minimal valid synthetic SPIR-V module with ClipDistance
    uint32_t test_spv[] = {
        0x07230203, 0x00010000, 0x12345678, 10, 0, // Header
        (2 << 16) | SpvOpCapability, SpvCapabilityClipDistance, // OpCapability ClipDistance
        (2 << 16) | SpvOpCapability, 1, // OpCapability Shader
        (5 << 16) | SpvOpDecorate, 7, SpvDecorationBuiltIn, SpvBuiltInClipDistance, 0, // OpDecorate %7 BuiltIn ClipDistance
        (4 << 16) | SpvOpTypePointer, 6, SpvStorageClassOutput, 5, // OpTypePointer %6 Output %5
        (4 << 16) | SpvOpVariable, 6, 7, SpvStorageClassOutput, // OpVariable %6 %7 Output
        (6 << 16) | SpvOpEntryPoint, 0, 1, 0x6e69616d, 0, 7, // OpEntryPoint Vertex %1 "main\0" %7
    };

    size_t out_bytes = 0;
    uint32_t *patched = compat_patch_spirv(test_spv, sizeof(test_spv), &out_bytes);
    if (!patched) {
        printf("FAILED: compat_patch_spirv returned NULL\n");
        return 1;
    }

    printf("Patched SPIR-V successfully (%zu bytes)!\n", out_bytes);
    // Verify OpCapability ClipDistance became SpvOpNop
    if ((patched[5] & 0xFFFF) != SpvOpNop) {
        printf("FAILED: OpCapability was not converted to OpNop\n");
        return 1;
    }
    // Verify OpDecorate became SpvOpNop
    if ((patched[9] & 0xFFFF) != SpvOpNop) {
        printf("FAILED: OpDecorate was not converted to OpNop\n");
        return 1;
    }
    // Verify OpTypePointer storage class became Private (6)
    if (patched[16] != SpvStorageClassPrivate) {
        printf("FAILED: OpTypePointer storage class is %u, expected %u\n", patched[16], SpvStorageClassPrivate);
        return 1;
    }
    // Verify OpVariable storage class became Private (6)
    if (patched[21] != SpvStorageClassPrivate) {
        printf("FAILED: OpVariable storage class is %u, expected %u\n", patched[21], SpvStorageClassPrivate);
        return 1;
    }
    // Verify OpEntryPoint removed %7 and padded with OpNop
    uint16_t ep_len = patched[22] >> 16;
    if (ep_len != 5) {
        printf("FAILED: OpEntryPoint len is %u, expected 5\n", ep_len);
        return 1;
    }
    if ((patched[27] & 0xFFFF) != SpvOpNop) {
        printf("FAILED: OpEntryPoint padding was not OpNop\n");
        return 1;
    }

    free(patched);
    printf("PASS: SPIR-V patcher verified on synthetic module!\n");
    return 0;
}

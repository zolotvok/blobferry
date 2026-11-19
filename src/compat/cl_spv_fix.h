// SPDX-License-Identifier: GPL-3.0-or-later
// cl_spv_fix.h: compat_patch_spirv entry and the SPV_FIX patcher. Included by mali_compat_layer.c.
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
    if (g_wls_fix) { // wls-fix-1: before the clamp, on the code so far; the hash filter uses the app's code
        size_t fb = 0;
        fixed = wls_fix_module(inWords, inBytes, out ? out : inWords, out ? *pOutBytes : inBytes, &fb);
        if (fixed) {
            *pOutBytes = fb;
            free(out);
            out = fixed;
        }
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


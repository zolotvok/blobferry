// SPDX-License-Identifier: GPL-3.0-or-later
// rt_spv_emit.h: SPIR-V word buffer and emit helpers. Included by mali_rt_emu.h.
/* ── Dynamic SPIR-V Word Buffer ───────────────────────────────────────────── */
struct SpvBuf {
    uint32_t *words;
    size_t    count;
    size_t    cap;
};

static void spv_buf_init(struct SpvBuf *b) {
    b->words = NULL;
    b->count = 0;
    b->cap = 0;
}

static void spv_buf_free(struct SpvBuf *b) {
    free(b->words);
    b->words = NULL;
    b->count = 0;
    b->cap = 0;
}

static void spv_buf_reserve(struct SpvBuf *b, size_t extra) {
    if (b->count + extra <= b->cap) return;
    size_t new_cap = b->cap ? b->cap * 2 : 1024;
    while (new_cap < b->count + extra) new_cap *= 2;
    uint32_t *nw = (uint32_t *)realloc(b->words, new_cap * sizeof(uint32_t));
    if (!nw) return;
    b->words = nw;
    b->cap = new_cap;
}

static void spv_emit_word(struct SpvBuf *b, uint32_t w) {
    spv_buf_reserve(b, 1);
    b->words[b->count++] = w;
}

static void spv_emit_words(struct SpvBuf *b, const uint32_t *w, size_t n) {
    if (n == 0) return;
    spv_buf_reserve(b, n);
    memcpy(&b->words[b->count], w, n * sizeof(uint32_t));
    b->count += n;
}

static void spv_emit_op0(struct SpvBuf *b, uint32_t op) {
    spv_emit_word(b, (1u << 16) | (op & 0xFFFFu));
}

static void spv_emit_op1(struct SpvBuf *b, uint32_t op, uint32_t a0) {
    uint32_t w[2] = { (2u << 16) | (op & 0xFFFFu), a0 };
    spv_emit_words(b, w, 2);
}

static void spv_emit_op2(struct SpvBuf *b, uint32_t op, uint32_t a0, uint32_t a1) {
    uint32_t w[3] = { (3u << 16) | (op & 0xFFFFu), a0, a1 };
    spv_emit_words(b, w, 3);
}

static void spv_emit_op3(struct SpvBuf *b, uint32_t op, uint32_t a0, uint32_t a1, uint32_t a2) {
    uint32_t w[4] = { (4u << 16) | (op & 0xFFFFu), a0, a1, a2 };
    spv_emit_words(b, w, 4);
}

static void spv_emit_op4(struct SpvBuf *b, uint32_t op, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3) {
    uint32_t w[5] = { (5u << 16) | (op & 0xFFFFu), a0, a1, a2, a3 };
    spv_emit_words(b, w, 5);
}

static void spv_emit_op5(struct SpvBuf *b, uint32_t op, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t a4) {
    uint32_t w[6] = { (6u << 16) | (op & 0xFFFFu), a0, a1, a2, a3, a4 };
    spv_emit_words(b, w, 6);
}

static inline void spv_emit_op6(struct SpvBuf *b, uint32_t op, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t a4, uint32_t a5) {
    uint32_t w[7] = { (7u << 16) | (op & 0xFFFFu), a0, a1, a2, a3, a4, a5 };
    spv_emit_words(b, w, 7);
}

static void spv_emit_string_inst(struct SpvBuf *b, uint32_t op, const char *str) {
    size_t len = strlen(str);
    size_t str_words = (len + 4) / 4;
    uint32_t wc = (uint32_t)(1 + str_words);
    spv_emit_word(b, (wc << 16) | (op & 0xFFFFu));
    for (size_t i = 0; i < str_words; i++) {
        uint32_t w = 0;
        for (size_t j = 0; j < 4; j++) {
            size_t idx = i * 4 + j;
            if (idx < len) w |= ((uint32_t)(uint8_t)str[idx]) << (j * 8);
        }
        spv_emit_word(b, w);
    }
}

/* ── SPIR-V Megakernel Builder State ──────────────────────────────────────── */

#define RT_TYPE_META_MAX_MEMBERS 32

struct RtTypeMeta {
    uint32_t op;
    uint32_t elem_type;     /* for Vector, Matrix, Array, Pointer */
    uint32_t count;         /* for Vector, Matrix, Array length (const id), Struct member count, Int/Float width */
    uint32_t extra;         /* for Int signedness, Pointer storage class, Array stride */
    uint32_t members[RT_TYPE_META_MAX_MEMBERS];
};

struct RtBuiltinVarMap {
    uint32_t var_id;        /* remapped Private variable ID in stage */
    uint32_t pointee_type;  /* remapped pointee type ID in stage */
    uint32_t builtin;       /* SpvBuiltIn enum */
};

struct RtStageParse {
    uint32_t                bound;
    uint32_t               *remap;           /* orig_id -> megakernel id (pass d0) */
    uint32_t               *remap_d1;        /* orig_id -> megakernel id (pass d1 for recursive functions) */
    uint32_t               *id_type;         /* orig_id -> orig_type_id */
    bool                   *is_psb_type;     /* orig_type_id -> true if PhysicalStorageBuffer/PushConstant/ShaderRecordBuffer ptr */
    bool                   *is_psb_id;       /* orig_id -> true if value is a PSB pointer */
    uint32_t               *var_sc;          /* orig_id -> original StorageClass of OpVariable */
    uint32_t               *var_pointee_orig;/* orig_id -> original pointee type ID of OpVariable */
    uint32_t               *dec_builtin;     /* orig_id -> BuiltIn + 1 (0 = none) */
    uint32_t               *dec_spec_id;     /* orig_id -> SpecId + 1 (0 = none) */
    uint32_t               *dec_array_stride;/* orig_id -> ArrayStride (0 = none) */
    bool                   *strip_dec_loc;   /* orig_id -> true if Location decoration should be stripped */
    uint32_t               *const_u32_val;   /* orig_id -> literal value of 32-bit OpConstant */
    bool                   *is_const_u32;    /* orig_id -> true if 32-bit int constant */

    uint32_t                orig_entry_fn_id;
    uint32_t                entry_fn_id_d0;
    uint32_t                entry_fn_id_d1;
    bool                    has_trace_ray;

    /* Per-stage interface variables */
    uint32_t                incoming_payload_var;  /* remapped ID (0 = none) */
    uint32_t                incoming_payload_type; /* remapped pointee type ID */
    uint32_t                hit_attr_var;          /* remapped ID (0 = none) */
    uint32_t                hit_attr_type;         /* remapped pointee type ID */
    uint32_t                incoming_callable_var; /* remapped ID (0 = none) */
    uint32_t                incoming_callable_type;/* remapped pointee type ID */

    /* PSB module-scope variables (PushConstant & ShaderRecordBufferKHR) */
    uint32_t                psb_vars_orig[16];
    uint32_t                psb_vars_ptr_type[16]; /* remapped OpTypePointer PhysicalStorageBuffer */
    uint32_t                psb_vars_sc[16];       /* 9 = PushConstant, 5343 = ShaderRecordBufferKHR */
    uint32_t                psb_var_count;

    /* RT BuiltIn variables used by this stage */
    struct RtBuiltinVarMap  rt_builtins[32];
    uint32_t                rt_builtin_count;
};

struct RtDedupType {
    uint32_t op;
    uint32_t a0;
    uint32_t a1;
    uint32_t a2;
    uint32_t res_id;
};

struct RtDedupTypeN {
    uint32_t op;
    uint32_t opnd_count;
    uint32_t opnds[48];
    uint32_t res_id;
};

struct RtDedupConstU32 {
    uint32_t type_id;
    uint32_t val;
    uint32_t res_id;
};

struct RtMegaBuilder {
    uint32_t next_id;

    struct SpvBuf capabilities;
    struct SpvBuf extensions;
    struct SpvBuf ext_imports;
    struct SpvBuf decorations;
    struct SpvBuf types_consts_globals;
    struct SpvBuf functions;

    bool has_vulkan_memory_model;
    uint32_t emitted_caps[128];
    uint32_t emitted_cap_count;
    char     emitted_exts[32][64];
    uint32_t emitted_ext_count;

    /* Global type metadata indexed by megakernel type_id */
    struct RtTypeMeta *type_meta;
    size_t             type_meta_cap;

    /* Deduplication tables for scalar/vector/matrix/pointer/func types & u32 constants */
    struct RtDedupType     dedup_types[1024];
    uint32_t               dedup_type_count;
    struct RtDedupConstU32 dedup_consts[512];
    uint32_t               dedup_const_count;
    struct RtDedupTypeN    dedup_complex_types[512];
    uint32_t               dedup_complex_type_count;
    uint32_t               t_sampler;

    /* Interface variable IDs for OpEntryPoint GLCompute %main */
    uint32_t iface_vars[1024];
    uint32_t iface_var_count;

    /* Used expensive RT BuiltIns across all stages */
    bool uses_object_ray;
    bool uses_obj_to_world;
    bool uses_world_to_obj;

    /* Canonical IDs in the megakernel */
    uint32_t ext_glsl450;
    uint32_t fn_main;

    /* Canonical basic types */
    uint32_t t_void;
    uint32_t t_fn_void;
    uint32_t t_bool;
    uint32_t t_u32;
    uint32_t t_i32;
    uint32_t t_u64;
    uint32_t t_f32;
    uint32_t t_v2f32;
    uint32_t t_v3f32;
    uint32_t t_v4f32;
    uint32_t t_v2u32;
    uint32_t t_v3u32;
    uint32_t t_v3i32;
    uint32_t t_mat4x3f32;
    uint32_t t_mat3x4f32;
    uint32_t t_ray_query;
    uint32_t t_accel_struct;

    /* Canonical pointer types */
    uint32_t t_ptr_input_v3u32;
    uint32_t t_ptr_priv_bool;
    uint32_t t_ptr_priv_u32;
    uint32_t t_ptr_priv_i32;
    uint32_t t_ptr_priv_u64;
    uint32_t t_ptr_priv_f32;
    uint32_t t_ptr_priv_v2f32;
    uint32_t t_ptr_priv_v3f32;
    uint32_t t_ptr_priv_v3u32;
    uint32_t t_ptr_priv_mat4x3f32;
    uint32_t t_ptr_priv_rq;
    uint32_t t_ptr_psb_u32;
    uint32_t t_ptr_psb_u64;

    /* Backing array types and pointers */
    uint32_t t_arr_payload;
    uint32_t t_ptr_priv_arr_payload;
    uint32_t t_arr_hitattr;
    uint32_t t_ptr_priv_arr_hitattr;
    uint32_t t_arr_callable;
    uint32_t t_ptr_priv_arr_callable;

    /* Megakernel PushConstant struct & pointers */
    uint32_t t_struct_rt_pc;
    uint32_t t_ptr_pc_struct;
    uint32_t t_ptr_pc_u32;
    uint32_t t_ptr_pc_u64;

    /* Common constants */
    uint32_t c_false;
    uint32_t c_true;
    uint32_t c_u32_0;
    uint32_t c_u32_1;
    uint32_t c_u32_2;
    uint32_t c_u32_8;
    uint32_t c_u32_12;
    uint32_t c_u32_16;
    uint32_t c_u32_20;
    uint32_t c_u32_32;
    uint32_t c_u32_254;
    uint32_t c_u32_255;
    uint32_t c_f32_0;

    /* Global variables */
    uint32_t v_gl_global_inv_id;
    uint32_t v_rt_pc;

    uint32_t g_rgen_base;
    uint32_t g_rgen_size;
    uint32_t g_miss_base;
    uint32_t g_miss_stride;
    uint32_t g_miss_size;
    uint32_t g_hit_base;
    uint32_t g_hit_stride;
    uint32_t g_hit_size;
    uint32_t g_call_base;
    uint32_t g_call_stride;
    uint32_t g_call_size;
    uint32_t g_push_consts_bda;
    uint32_t g_cur_record_bda;
    uint32_t g_cand_rahit_id;
    uint32_t g_cand_rint_id;

    uint32_t g_launch_id;
    uint32_t g_launch_size;
    uint32_t g_world_ray_orig;
    uint32_t g_world_ray_dir;
    uint32_t g_obj_ray_orig;
    uint32_t g_obj_ray_dir;
    uint32_t g_ray_tmin;
    uint32_t g_ray_tmax;
    uint32_t g_incoming_ray_flags;
    uint32_t g_cull_mask;
    uint32_t g_hit_kind;
    uint32_t g_instance_custom_idx;
    uint32_t g_instance_id;
    uint32_t g_primitive_id;
    uint32_t g_geom_index;
    uint32_t g_obj_to_world;
    uint32_t g_world_to_obj;

    uint32_t g_trace_sbt_offset;
    uint32_t g_trace_sbt_stride;
    uint32_t g_trace_miss_index;

    uint32_t g_rahit_ignored;
    uint32_t g_ray_terminated;
    uint32_t g_guard_iter;
    uint32_t g_rint_generated_hit;
    uint32_t g_cur_rahit_id;

    uint32_t g_rq_d0;
    uint32_t g_rq_d1;

    uint32_t g_payload_u32;
    uint32_t g_hitattr_u32;
    uint32_t g_staging_hitattr_u32;
    uint32_t g_committed_hitattr_u32;
    uint32_t g_callable_u32;

    /* Helper functions */
    uint32_t fn_trace_loop_d0;
    uint32_t fn_trace_loop_d1;
    uint32_t fn_invoke_rahit_d0;
    uint32_t fn_invoke_rahit_d1;
    uint32_t fn_invoke_rint_d0;
    uint32_t fn_invoke_rint_d1;
    uint32_t fn_report_hit_d0;
    uint32_t fn_report_hit_d1;
    uint32_t fn_invoke_rcall;
    uint32_t t_fn_u32_void;
    uint32_t t_fn_report_hit;
};

static uint32_t rt_alloc_id(struct RtMegaBuilder *mb) {
    uint32_t id = mb->next_id++;
    if (id >= mb->type_meta_cap) {
        size_t new_cap = mb->type_meta_cap ? mb->type_meta_cap * 2 : 4096;
        while (new_cap <= id) new_cap *= 2;
        struct RtTypeMeta *nm = (struct RtTypeMeta *)realloc(mb->type_meta, new_cap * sizeof(*nm));
        if (nm) {
            memset(nm + mb->type_meta_cap, 0, (new_cap - mb->type_meta_cap) * sizeof(*nm));
            mb->type_meta = nm;
            mb->type_meta_cap = new_cap;
        }
    }
    return id;
}

static void rt_add_cap(struct RtMegaBuilder *mb, uint32_t cap) {
    if (cap == RT_CAP_RAY_TRACING_KHR ||
        cap == RT_CAP_RAY_TRACING_PROVISIONAL_KHR ||
        cap == RT_CAP_RAY_QUERY_PROVISIONAL_KHR ||
        cap == RT_CAP_RAY_TRACING_NV ||
        cap == RT_CAP_RAY_TRACING_MOTION_BLUR_NV) {
        return;
    }
    for (uint32_t i = 0; i < mb->emitted_cap_count; i++) {
        if (mb->emitted_caps[i] == cap) return;
    }
    if (mb->emitted_cap_count < 128) {
        mb->emitted_caps[mb->emitted_cap_count++] = cap;
        spv_emit_op1(&mb->capabilities, RT_OP_CAPABILITY, cap);
    }
}

static void rt_add_ext(struct RtMegaBuilder *mb, const char *ext) {
    if (!ext || !ext[0]) return;
    if (strcmp(ext, "SPV_KHR_ray_tracing") == 0 ||
        strcmp(ext, "SPV_NV_ray_tracing") == 0) {
        return;
    }
    for (uint32_t i = 0; i < mb->emitted_ext_count; i++) {
        if (strcmp(mb->emitted_exts[i], ext) == 0) return;
    }
    if (mb->emitted_ext_count < 32) {
        snprintf(mb->emitted_exts[mb->emitted_ext_count++], 64, "%s", ext);
        spv_emit_string_inst(&mb->extensions, RT_OP_EXTENSION, ext);
    }
}

static void rt_add_iface_var(struct RtMegaBuilder *mb, uint32_t var_id) {
    for (uint32_t i = 0; i < mb->iface_var_count; i++) {
        if (mb->iface_vars[i] == var_id) return;
    }
    if (mb->iface_var_count < 1024) {
        mb->iface_vars[mb->iface_var_count++] = var_id;
    }
}

static uint32_t rt_find_or_emit_type(struct RtMegaBuilder *mb, uint32_t op,
                                     uint32_t a0, uint32_t a1, uint32_t a2) {
    for (uint32_t i = 0; i < mb->dedup_type_count; i++) {
        if (mb->dedup_types[i].op == op &&
            mb->dedup_types[i].a0 == a0 &&
            mb->dedup_types[i].a1 == a1 &&
            mb->dedup_types[i].a2 == a2) {
            return mb->dedup_types[i].res_id;
        }
    }
    uint32_t id = rt_alloc_id(mb);
    if (op == RT_OP_TYPE_VOID || op == RT_OP_TYPE_BOOL ||
        op == RT_OP_TYPE_RAY_QUERY_KHR || op == RT_OP_TYPE_ACCEL_STRUCT_KHR) {
        spv_emit_op1(&mb->types_consts_globals, op, id);
        mb->type_meta[id].op = op;
    } else if (op == RT_OP_TYPE_FLOAT) {
        spv_emit_op2(&mb->types_consts_globals, op, id, a0);
        mb->type_meta[id].op = op;
        mb->type_meta[id].count = a0;
    } else if (op == RT_OP_TYPE_INT) {
        spv_emit_op3(&mb->types_consts_globals, op, id, a0, a1);
        mb->type_meta[id].op = op;
        mb->type_meta[id].count = a0;
        mb->type_meta[id].extra = a1;
    } else if (op == RT_OP_TYPE_VECTOR || op == RT_OP_TYPE_MATRIX) {
        spv_emit_op3(&mb->types_consts_globals, op, id, a0, a1);
        mb->type_meta[id].op = op;
        mb->type_meta[id].elem_type = a0;
        mb->type_meta[id].count = a1;
    } else if (op == RT_OP_TYPE_POINTER) {
        spv_emit_op3(&mb->types_consts_globals, op, id, a0, a1);
        if (a2 != 0) {
            spv_emit_op3(&mb->decorations, RT_OP_DECORATE, id, RT_DEC_ARRAY_STRIDE, a2);
        }
        mb->type_meta[id].op = op;
        mb->type_meta[id].extra = a0;     /* storage class */
        mb->type_meta[id].elem_type = a1; /* pointee type */
    }
    if (mb->dedup_type_count < 1024) {
        mb->dedup_types[mb->dedup_type_count++] = (struct RtDedupType){ op, a0, a1, a2, id };
    }
    return id;
}

static uint32_t rt_find_or_emit_complex_type(struct RtMegaBuilder *mb, uint32_t op,
                                             const uint32_t *opnds, uint32_t opnd_count) {
    if (opnd_count > 48) opnd_count = 48;
    for (uint32_t i = 0; i < mb->dedup_complex_type_count; i++) {
        if (mb->dedup_complex_types[i].op == op &&
            mb->dedup_complex_types[i].opnd_count == opnd_count) {
            bool match = true;
            for (uint32_t k = 0; k < opnd_count; k++) {
                if (mb->dedup_complex_types[i].opnds[k] != opnds[k]) {
                    match = false;
                    break;
                }
            }
            if (match) return mb->dedup_complex_types[i].res_id;
        }
    }
    uint32_t id = rt_alloc_id(mb);
    spv_emit_word(&mb->types_consts_globals, ((2u + opnd_count) << 16) | op);
    spv_emit_word(&mb->types_consts_globals, id);
    for (uint32_t k = 0; k < opnd_count; k++) {
        spv_emit_word(&mb->types_consts_globals, opnds[k]);
    }
    mb->type_meta[id].op = op;
    if (mb->dedup_complex_type_count < 512) {
        struct RtDedupTypeN *d = &mb->dedup_complex_types[mb->dedup_complex_type_count++];
        d->op = op;
        d->opnd_count = opnd_count;
        for (uint32_t k = 0; k < opnd_count; k++) d->opnds[k] = opnds[k];
        d->res_id = id;
    }
    return id;
}

static uint32_t rt_get_const_u32(struct RtMegaBuilder *mb, uint32_t val) {
    for (uint32_t i = 0; i < mb->dedup_const_count; i++) {
        if (mb->dedup_consts[i].type_id == mb->t_u32 && mb->dedup_consts[i].val == val) {
            return mb->dedup_consts[i].res_id;
        }
    }
    uint32_t id = rt_alloc_id(mb);
    spv_emit_op3(&mb->types_consts_globals, RT_OP_CONSTANT, mb->t_u32, id, val);
    if (mb->dedup_const_count < 512) {
        mb->dedup_consts[mb->dedup_const_count++] = (struct RtDedupConstU32){ mb->t_u32, val, id };
    }
    return id;
}

static uint32_t rt_get_const_i32(struct RtMegaBuilder *mb, int32_t val) {
    uint32_t uval = (uint32_t)val;
    for (uint32_t i = 0; i < mb->dedup_const_count; i++) {
        if (mb->dedup_consts[i].type_id == mb->t_i32 && mb->dedup_consts[i].val == uval) {
            return mb->dedup_consts[i].res_id;
        }
    }
    uint32_t id = rt_alloc_id(mb);
    spv_emit_op3(&mb->types_consts_globals, RT_OP_CONSTANT, mb->t_i32, id, uval);
    if (mb->dedup_const_count < 512) {
        mb->dedup_consts[mb->dedup_const_count++] = (struct RtDedupConstU32){ mb->t_i32, uval, id };
    }
    return id;
}

static uint32_t rt_get_const_u64(struct RtMegaBuilder *mb, uint64_t val) {
    uint32_t id = rt_alloc_id(mb);
    spv_emit_op4(&mb->types_consts_globals, RT_OP_CONSTANT, mb->t_u64, id,
                 (uint32_t)(val & 0xFFFFFFFFu), (uint32_t)(val >> 32));
    return id;
}

static uint32_t rt_emit_global_var(struct RtMegaBuilder *mb, uint32_t ptr_type, uint32_t sc) {
    uint32_t id = rt_alloc_id(mb);
    spv_emit_op3(&mb->types_consts_globals, RT_OP_VARIABLE, ptr_type, id, sc);
    rt_add_iface_var(mb, id);
    return id;
}
/* split-1112: drop a global from the OpEntryPoint interface. The small helper kernels declare their own push-constant
 * block; the preamble's megakernel block (v_rt_pc) stays declared but must not be listed: SPIR-V >= 1.4 lists every
 * global, and two PushConstant variables in one interface is invalid (VUID-StandaloneSpirv-OpVariable-06673;
 * lavapipe rejected the indirect setup kernel for it, so every TraceRaysIndirect used the 256x256 fallback). */
static void rt_iface_drop(struct RtMegaBuilder *mb, uint32_t var_id) {
    for (uint32_t i = 0; i < mb->iface_var_count; i++)
        if (mb->iface_vars[i] == var_id) { mb->iface_vars[i] = mb->iface_vars[--mb->iface_var_count]; return; }
}


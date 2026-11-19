// SPDX-License-Identifier: GPL-3.0-or-later
// rt_mk_parse.h: megakernel preamble, pack/unpack, stage globals parse. Included by mali_rt_emu.h.
/* ── Megakernel Preamble Initialization ───────────────────────────────────── */

static void rt_mega_init_preamble(struct RtMegaBuilder *mb) {
    memset(mb, 0, sizeof(*mb));
    mb->next_id = 1;
    spv_buf_init(&mb->capabilities);
    spv_buf_init(&mb->extensions);
    spv_buf_init(&mb->ext_imports);
    spv_buf_init(&mb->decorations);
    spv_buf_init(&mb->types_consts_globals);
    spv_buf_init(&mb->functions);

    /* Mandatory capabilities & extensions */
    rt_add_cap(mb, RT_CAP_SHADER);
    rt_add_cap(mb, RT_CAP_INT64);
    rt_add_cap(mb, RT_CAP_PHYSICAL_STORAGE_BUFFER_ADDRESSES);
    rt_add_cap(mb, RT_CAP_RAY_QUERY_KHR);
    rt_add_cap(mb, RT_CAP_RAY_TRAVERSAL_PRIMITIVE_CULLING_KHR);

    rt_add_ext(mb, "SPV_KHR_physical_storage_buffer");
    rt_add_ext(mb, "SPV_KHR_ray_query");

    /* GLSL.std.450 import */
    mb->ext_glsl450 = rt_alloc_id(mb);
    {
        /* OpExtInstImport %ext_glsl450 "GLSL.std.450" (4 words for string = 16 bytes) */
        uint32_t w[6] = {
            (6u << 16) | RT_OP_EXT_INST_IMPORT,
            mb->ext_glsl450,
            0x534C4C47u, /* "GLLS" -> wait: 'G','L','S','L' = 0x4C534C47 */
            0x74732E4Cu,
            0x35342E64u,
            0x00000030u
        };
        w[2] = 0x4C534C47u; /* "GLSL" */
        w[3] = 0x6474732Eu; /* ".std" */
        w[4] = 0x3035342Eu; /* ".450" */
        w[5] = 0x00000000u; /* "\0\0\0\0" */
        spv_emit_words(&mb->ext_imports, w, 6);
    }

    mb->fn_main = rt_alloc_id(mb);

    /* Canonical scalar, vector, matrix, and opaque types */
    mb->t_void         = rt_find_or_emit_type(mb, RT_OP_TYPE_VOID, 0, 0, 0);
    mb->t_bool         = rt_find_or_emit_type(mb, RT_OP_TYPE_BOOL, 0, 0, 0);
    mb->t_u32          = rt_find_or_emit_type(mb, RT_OP_TYPE_INT, 32, 0, 0);
    mb->t_i32          = rt_find_or_emit_type(mb, RT_OP_TYPE_INT, 32, 1, 0);
    mb->t_u64          = rt_find_or_emit_type(mb, RT_OP_TYPE_INT, 64, 0, 0);
    mb->t_f32          = rt_find_or_emit_type(mb, RT_OP_TYPE_FLOAT, 32, 0, 0);
    mb->t_v2f32        = rt_find_or_emit_type(mb, RT_OP_TYPE_VECTOR, mb->t_f32, 2, 0);
    mb->t_v3f32        = rt_find_or_emit_type(mb, RT_OP_TYPE_VECTOR, mb->t_f32, 3, 0);
    mb->t_v4f32        = rt_find_or_emit_type(mb, RT_OP_TYPE_VECTOR, mb->t_f32, 4, 0);
    mb->t_v2u32        = rt_find_or_emit_type(mb, RT_OP_TYPE_VECTOR, mb->t_u32, 2, 0);
    mb->t_v3u32        = rt_find_or_emit_type(mb, RT_OP_TYPE_VECTOR, mb->t_u32, 3, 0);
    mb->t_v3i32        = rt_find_or_emit_type(mb, RT_OP_TYPE_VECTOR, mb->t_i32, 3, 0);
    mb->t_mat4x3f32    = rt_find_or_emit_type(mb, RT_OP_TYPE_MATRIX, mb->t_v3f32, 4, 0);
    mb->t_mat3x4f32    = rt_find_or_emit_type(mb, RT_OP_TYPE_MATRIX, mb->t_v4f32, 3, 0);
    mb->t_ray_query    = rt_find_or_emit_type(mb, RT_OP_TYPE_RAY_QUERY_KHR, 0, 0, 0);
    mb->t_accel_struct = rt_find_or_emit_type(mb, RT_OP_TYPE_ACCEL_STRUCT_KHR, 0, 0, 0);

    /* Function types */
    mb->t_fn_void = rt_find_or_emit_complex_type(mb, RT_OP_TYPE_FUNCTION, &mb->t_void, 1);

    uint32_t fn_u32_void_opnds[2] = { mb->t_void, mb->t_u32 };
    mb->t_fn_u32_void = rt_find_or_emit_complex_type(mb, RT_OP_TYPE_FUNCTION, fn_u32_void_opnds, 2);

    uint32_t fn_rep_opnds[3] = { mb->t_bool, mb->t_f32, mb->t_u32 };
    mb->t_fn_report_hit = rt_find_or_emit_complex_type(mb, RT_OP_TYPE_FUNCTION, fn_rep_opnds, 3);

    /* Pointer types */
    mb->t_ptr_input_v3u32   = rt_find_or_emit_type(mb, RT_OP_TYPE_POINTER, RT_SC_INPUT, mb->t_v3u32, 0);
    mb->t_ptr_priv_bool     = rt_find_or_emit_type(mb, RT_OP_TYPE_POINTER, RT_SC_PRIVATE, mb->t_bool, 0);
    mb->t_ptr_priv_u32      = rt_find_or_emit_type(mb, RT_OP_TYPE_POINTER, RT_SC_PRIVATE, mb->t_u32, 0);
    mb->t_ptr_priv_i32      = rt_find_or_emit_type(mb, RT_OP_TYPE_POINTER, RT_SC_PRIVATE, mb->t_i32, 0);
    mb->t_ptr_priv_u64      = rt_find_or_emit_type(mb, RT_OP_TYPE_POINTER, RT_SC_PRIVATE, mb->t_u64, 0);
    mb->t_ptr_priv_f32      = rt_find_or_emit_type(mb, RT_OP_TYPE_POINTER, RT_SC_PRIVATE, mb->t_f32, 0);
    mb->t_ptr_priv_v2f32    = rt_find_or_emit_type(mb, RT_OP_TYPE_POINTER, RT_SC_PRIVATE, mb->t_v2f32, 0);
    mb->t_ptr_priv_v3f32    = rt_find_or_emit_type(mb, RT_OP_TYPE_POINTER, RT_SC_PRIVATE, mb->t_v3f32, 0);
    mb->t_ptr_priv_v3u32    = rt_find_or_emit_type(mb, RT_OP_TYPE_POINTER, RT_SC_PRIVATE, mb->t_v3u32, 0);
    mb->t_ptr_priv_mat4x3f32= rt_find_or_emit_type(mb, RT_OP_TYPE_POINTER, RT_SC_PRIVATE, mb->t_mat4x3f32, 0);
    mb->t_ptr_priv_rq       = rt_find_or_emit_type(mb, RT_OP_TYPE_POINTER, RT_SC_PRIVATE, mb->t_ray_query, 0);
    mb->t_ptr_psb_u32       = rt_find_or_emit_type(mb, RT_OP_TYPE_POINTER, RT_SC_PHYSICAL_STORAGE_BUFFER, mb->t_u32, 0);
    mb->t_ptr_psb_u64       = rt_find_or_emit_type(mb, RT_OP_TYPE_POINTER, RT_SC_PHYSICAL_STORAGE_BUFFER, mb->t_u64, 0);

    /* Common constants */
    mb->c_false = rt_alloc_id(mb);
    spv_emit_op2(&mb->types_consts_globals, RT_OP_CONSTANT_FALSE, mb->t_bool, mb->c_false);
    mb->c_true = rt_alloc_id(mb);
    spv_emit_op2(&mb->types_consts_globals, RT_OP_CONSTANT_TRUE, mb->t_bool, mb->c_true);

    mb->c_u32_0   = rt_get_const_u32(mb, 0u);
    mb->c_u32_1   = rt_get_const_u32(mb, 1u);
    mb->c_u32_2   = rt_get_const_u32(mb, 2u);
    mb->c_u32_8   = rt_get_const_u32(mb, 8u);
    mb->c_u32_12  = rt_get_const_u32(mb, 12u);
    mb->c_u32_16  = rt_get_const_u32(mb, 16u);
    mb->c_u32_20  = rt_get_const_u32(mb, 20u);
    mb->c_u32_32  = rt_get_const_u32(mb, 32u);
    mb->c_u32_254 = rt_get_const_u32(mb, 254u);
    mb->c_u32_255 = rt_get_const_u32(mb, 255u);

    mb->c_f32_0 = rt_alloc_id(mb);
    spv_emit_op3(&mb->types_consts_globals, RT_OP_CONSTANT, mb->t_f32, mb->c_f32_0, 0u);

    /* Backing array types for Payload (64 dwords), HitAttr (16 dwords), Callable (32 dwords) */
    uint32_t c_pay_len  = rt_get_const_u32(mb, RT_PAYLOAD_MAX_DWORDS);
    uint32_t c_hit_len  = rt_get_const_u32(mb, RT_HITATTR_MAX_DWORDS);
    uint32_t c_call_len = rt_get_const_u32(mb, RT_CALLABLE_MAX_DWORDS);

    mb->t_arr_payload = rt_alloc_id(mb);
    spv_emit_op3(&mb->types_consts_globals, RT_OP_TYPE_ARRAY, mb->t_arr_payload, mb->t_u32, c_pay_len);
    mb->t_ptr_priv_arr_payload = rt_find_or_emit_type(mb, RT_OP_TYPE_POINTER, RT_SC_PRIVATE, mb->t_arr_payload, 0);

    mb->t_arr_hitattr = rt_alloc_id(mb);
    spv_emit_op3(&mb->types_consts_globals, RT_OP_TYPE_ARRAY, mb->t_arr_hitattr, mb->t_u32, c_hit_len);
    mb->t_ptr_priv_arr_hitattr = rt_find_or_emit_type(mb, RT_OP_TYPE_POINTER, RT_SC_PRIVATE, mb->t_arr_hitattr, 0);

    mb->t_arr_callable = rt_alloc_id(mb);
    spv_emit_op3(&mb->types_consts_globals, RT_OP_TYPE_ARRAY, mb->t_arr_callable, mb->t_u32, c_call_len);
    mb->t_ptr_priv_arr_callable = rt_find_or_emit_type(mb, RT_OP_TYPE_POINTER, RT_SC_PRIVATE, mb->t_arr_callable, 0);

    /* Megakernel PushConstant struct (18 members, 96 bytes) */
    mb->t_struct_rt_pc = rt_alloc_id(mb);
    {
        uint32_t w[20] = {
            (20u << 16) | RT_OP_TYPE_STRUCT,
            mb->t_struct_rt_pc,
            mb->t_u64, mb->t_u32, mb->t_u32, /* 0..2:  rgen_base(0), stride(8), size(12) */
            mb->t_u64, mb->t_u32, mb->t_u32, /* 3..5:  miss_base(16), stride(24), size(28) */
            mb->t_u64, mb->t_u32, mb->t_u32, /* 6..8:  hit_base(32), stride(40), size(44) */
            mb->t_u64, mb->t_u32, mb->t_u32, /* 9..11: call_base(48), stride(56), size(60) */
            mb->t_u32, mb->t_u32, mb->t_u32, mb->t_u32, /* 12..15: w(64), h(68), d(72), mode(76) */
            mb->t_u64,                       /* 16: push_consts_bda(80) */
            mb->t_u64                        /* 17: indirect_bda(88) */
        };
        spv_emit_words(&mb->types_consts_globals, w, 20);
        spv_emit_op2(&mb->decorations, RT_OP_DECORATE, mb->t_struct_rt_pc, RT_DEC_BLOCK);
        static const uint32_t offsets[18] = {
            0, 8, 12,
            16, 24, 28,
            32, 40, 44,
            48, 56, 60,
            64, 68, 72, 76,
            80, 88
        };
        for (uint32_t m = 0; m < 18; m++) {
            spv_emit_op4(&mb->decorations, RT_OP_MEMBER_DECORATE, mb->t_struct_rt_pc, m, RT_DEC_OFFSET, offsets[m]);
        }
    }
    mb->t_ptr_pc_struct = rt_find_or_emit_type(mb, RT_OP_TYPE_POINTER, RT_SC_PUSH_CONSTANT, mb->t_struct_rt_pc, 0);
    mb->t_ptr_pc_u32    = rt_find_or_emit_type(mb, RT_OP_TYPE_POINTER, RT_SC_PUSH_CONSTANT, mb->t_u32, 0);
    mb->t_ptr_pc_u64    = rt_find_or_emit_type(mb, RT_OP_TYPE_POINTER, RT_SC_PUSH_CONSTANT, mb->t_u64, 0);

    /* BuiltIn GlobalInvocationId variable */
    mb->v_gl_global_inv_id = rt_emit_global_var(mb, mb->t_ptr_input_v3u32, RT_SC_INPUT);
    spv_emit_op3(&mb->decorations, RT_OP_DECORATE, mb->v_gl_global_inv_id, RT_DEC_BUILTIN, RT_BUILTIN_GLOBAL_INVOCATION_ID);

    /* Megakernel PushConstant variable */
    mb->v_rt_pc = rt_emit_global_var(mb, mb->t_ptr_pc_struct, RT_SC_PUSH_CONSTANT);

    /* Private globals for SBT & dispatch state */
    mb->g_rgen_base         = rt_emit_global_var(mb, mb->t_ptr_priv_u64, RT_SC_PRIVATE);
    mb->g_rgen_size         = rt_emit_global_var(mb, mb->t_ptr_priv_u64, RT_SC_PRIVATE);
    mb->g_miss_base         = rt_emit_global_var(mb, mb->t_ptr_priv_u64, RT_SC_PRIVATE);
    mb->g_miss_stride       = rt_emit_global_var(mb, mb->t_ptr_priv_u32, RT_SC_PRIVATE);
    mb->g_miss_size         = rt_emit_global_var(mb, mb->t_ptr_priv_u64, RT_SC_PRIVATE);
    mb->g_hit_base          = rt_emit_global_var(mb, mb->t_ptr_priv_u64, RT_SC_PRIVATE);
    mb->g_hit_stride        = rt_emit_global_var(mb, mb->t_ptr_priv_u32, RT_SC_PRIVATE);
    mb->g_hit_size          = rt_emit_global_var(mb, mb->t_ptr_priv_u64, RT_SC_PRIVATE);
    mb->g_call_base         = rt_emit_global_var(mb, mb->t_ptr_priv_u64, RT_SC_PRIVATE);
    mb->g_call_stride       = rt_emit_global_var(mb, mb->t_ptr_priv_u32, RT_SC_PRIVATE);
    mb->g_call_size         = rt_emit_global_var(mb, mb->t_ptr_priv_u64, RT_SC_PRIVATE);
    mb->g_push_consts_bda   = rt_emit_global_var(mb, mb->t_ptr_priv_u64, RT_SC_PRIVATE);
    mb->g_cur_record_bda    = rt_emit_global_var(mb, mb->t_ptr_priv_u64, RT_SC_PRIVATE);
    mb->g_cand_rahit_id     = rt_emit_global_var(mb, mb->t_ptr_priv_u32, RT_SC_PRIVATE);
    mb->g_cand_rint_id      = rt_emit_global_var(mb, mb->t_ptr_priv_u32, RT_SC_PRIVATE);

    /* Private globals for RT BuiltIns */
    mb->g_launch_id         = rt_emit_global_var(mb, mb->t_ptr_priv_v3u32, RT_SC_PRIVATE);
    mb->g_launch_size       = rt_emit_global_var(mb, mb->t_ptr_priv_v3u32, RT_SC_PRIVATE);
    mb->g_world_ray_orig    = rt_emit_global_var(mb, mb->t_ptr_priv_v3f32, RT_SC_PRIVATE);
    mb->g_world_ray_dir     = rt_emit_global_var(mb, mb->t_ptr_priv_v3f32, RT_SC_PRIVATE);
    mb->g_obj_ray_orig      = rt_emit_global_var(mb, mb->t_ptr_priv_v3f32, RT_SC_PRIVATE);
    mb->g_obj_ray_dir       = rt_emit_global_var(mb, mb->t_ptr_priv_v3f32, RT_SC_PRIVATE);
    mb->g_ray_tmin          = rt_emit_global_var(mb, mb->t_ptr_priv_f32, RT_SC_PRIVATE);
    mb->g_ray_tmax          = rt_emit_global_var(mb, mb->t_ptr_priv_f32, RT_SC_PRIVATE);
    mb->g_incoming_ray_flags= rt_emit_global_var(mb, mb->t_ptr_priv_u32, RT_SC_PRIVATE);
    mb->g_cull_mask         = rt_emit_global_var(mb, mb->t_ptr_priv_u32, RT_SC_PRIVATE);
    mb->g_hit_kind          = rt_emit_global_var(mb, mb->t_ptr_priv_u32, RT_SC_PRIVATE);
    mb->g_instance_custom_idx = rt_emit_global_var(mb, mb->t_ptr_priv_u32, RT_SC_PRIVATE);
    mb->g_instance_id       = rt_emit_global_var(mb, mb->t_ptr_priv_u32, RT_SC_PRIVATE);
    mb->g_primitive_id      = rt_emit_global_var(mb, mb->t_ptr_priv_u32, RT_SC_PRIVATE);
    mb->g_geom_index        = rt_emit_global_var(mb, mb->t_ptr_priv_u32, RT_SC_PRIVATE);
    mb->g_obj_to_world      = rt_emit_global_var(mb, mb->t_ptr_priv_mat4x3f32, RT_SC_PRIVATE);
    mb->g_world_to_obj      = rt_emit_global_var(mb, mb->t_ptr_priv_mat4x3f32, RT_SC_PRIVATE);

    mb->g_trace_sbt_offset  = rt_emit_global_var(mb, mb->t_ptr_priv_u32, RT_SC_PRIVATE);
    mb->g_trace_sbt_stride  = rt_emit_global_var(mb, mb->t_ptr_priv_u32, RT_SC_PRIVATE);
    mb->g_trace_miss_index  = rt_emit_global_var(mb, mb->t_ptr_priv_u32, RT_SC_PRIVATE);

    mb->g_rahit_ignored     = rt_emit_global_var(mb, mb->t_ptr_priv_u32, RT_SC_PRIVATE);
    mb->g_ray_terminated    = rt_emit_global_var(mb, mb->t_ptr_priv_u32, RT_SC_PRIVATE);
    mb->g_guard_iter        = rt_emit_global_var(mb, mb->t_ptr_priv_u32, RT_SC_PRIVATE);
    mb->g_rint_generated_hit= rt_emit_global_var(mb, mb->t_ptr_priv_u32, RT_SC_PRIVATE);
    mb->g_cur_rahit_id      = rt_emit_global_var(mb, mb->t_ptr_priv_u32, RT_SC_PRIVATE);

    /* Private RayQuery objects for depth 0 and depth 1 */
    mb->g_rq_d0             = rt_emit_global_var(mb, mb->t_ptr_priv_rq, RT_SC_PRIVATE);
    mb->g_rq_d1             = rt_emit_global_var(mb, mb->t_ptr_priv_rq, RT_SC_PRIVATE);

    /* Backing storage arrays */
    mb->g_payload_u32          = rt_emit_global_var(mb, mb->t_ptr_priv_arr_payload, RT_SC_PRIVATE);
    mb->g_hitattr_u32          = rt_emit_global_var(mb, mb->t_ptr_priv_arr_hitattr, RT_SC_PRIVATE);
    mb->g_staging_hitattr_u32  = rt_emit_global_var(mb, mb->t_ptr_priv_arr_hitattr, RT_SC_PRIVATE);
    mb->g_committed_hitattr_u32= rt_emit_global_var(mb, mb->t_ptr_priv_arr_hitattr, RT_SC_PRIVATE);
    mb->g_callable_u32         = rt_emit_global_var(mb, mb->t_ptr_priv_arr_callable, RT_SC_PRIVATE);

    /* Pre-allocate helper function IDs */
    mb->fn_trace_loop_d0   = rt_alloc_id(mb);
    mb->fn_trace_loop_d1   = rt_alloc_id(mb);
    mb->fn_invoke_rahit_d0 = rt_alloc_id(mb);
    mb->fn_invoke_rahit_d1 = rt_alloc_id(mb);
    mb->fn_invoke_rint_d0  = rt_alloc_id(mb);
    mb->fn_invoke_rint_d1  = rt_alloc_id(mb);
    mb->fn_report_hit_d0   = rt_alloc_id(mb);
    mb->fn_report_hit_d1   = rt_alloc_id(mb);
    mb->fn_invoke_rcall    = rt_alloc_id(mb);
}

/* ── Recursive Leaf-by-Leaf Scalar Packer/Unpacker ────────────────────────── */

static void rt_emit_pack_unpack_type(struct RtMegaBuilder *mb, struct SpvBuf *fn_buf,
                                     uint32_t base_var_id, uint32_t type_id,
                                     uint32_t *idx_chain, uint32_t chain_depth,
                                     uint32_t *slot_io, uint32_t max_slots,
                                     uint32_t backing_arr_var_id, bool is_pack) {
    if (*slot_io >= max_slots || type_id == 0 || type_id >= mb->type_meta_cap) return;
    const struct RtTypeMeta *tm = &mb->type_meta[type_id];

    if (tm->op == RT_OP_TYPE_INT || tm->op == RT_OP_TYPE_FLOAT ||
        tm->op == RT_OP_TYPE_BOOL || tm->op == RT_OP_TYPE_POINTER) {
        uint32_t elem_ptr = base_var_id;
        if (chain_depth > 0) {
            uint32_t ptr_type = rt_find_or_emit_type(mb, RT_OP_TYPE_POINTER, RT_SC_PRIVATE, type_id, 0);
            elem_ptr = rt_alloc_id(mb);
            uint32_t wc = 4 + chain_depth;
            spv_emit_word(fn_buf, (wc << 16) | RT_OP_ACCESS_CHAIN);
            spv_emit_word(fn_buf, ptr_type);
            spv_emit_word(fn_buf, elem_ptr);
            spv_emit_word(fn_buf, base_var_id);
            for (uint32_t i = 0; i < chain_depth; i++) {
                spv_emit_word(fn_buf, idx_chain[i]);
            }
        }

        /* Handle 64-bit integer/float/PSB-pointer across 2 dwords */
        if ((tm->op == RT_OP_TYPE_INT && tm->count == 64) ||
            (tm->op == RT_OP_TYPE_FLOAT && tm->count == 64) ||
            tm->op == RT_OP_TYPE_POINTER) {
            if (*slot_io + 1 >= max_slots) return;
            uint32_t idx0 = rt_get_const_u32(mb, *slot_io);
            uint32_t idx1 = rt_get_const_u32(mb, *slot_io + 1);
            uint32_t sptr0 = rt_alloc_id(mb);
            uint32_t sptr1 = rt_alloc_id(mb);
            spv_emit_op4(fn_buf, RT_OP_ACCESS_CHAIN, mb->t_ptr_priv_u32, sptr0, backing_arr_var_id, idx0);
            spv_emit_op4(fn_buf, RT_OP_ACCESS_CHAIN, mb->t_ptr_priv_u32, sptr1, backing_arr_var_id, idx1);
            if (is_pack) {
                uint32_t val = rt_alloc_id(mb);
                spv_emit_op3(fn_buf, RT_OP_LOAD, type_id, val, elem_ptr);
                uint32_t u64_val = val;
                if (tm->op == RT_OP_TYPE_POINTER) {
                    u64_val = rt_alloc_id(mb);
                    spv_emit_op3(fn_buf, RT_OP_CONVERT_PTR_TO_U, mb->t_u64, u64_val, val);
                }
                uint32_t v2_val = rt_alloc_id(mb);
                spv_emit_op3(fn_buf, RT_OP_BITCAST, mb->t_v2u32, v2_val, u64_val);
                uint32_t lo = rt_alloc_id(mb);
                uint32_t hi = rt_alloc_id(mb);
                spv_emit_op4(fn_buf, RT_OP_COMPOSITE_EXTRACT, mb->t_u32, lo, v2_val, 0);
                spv_emit_op4(fn_buf, RT_OP_COMPOSITE_EXTRACT, mb->t_u32, hi, v2_val, 1);
                spv_emit_op2(fn_buf, RT_OP_STORE, sptr0, lo);
                spv_emit_op2(fn_buf, RT_OP_STORE, sptr1, hi);
            } else {
                uint32_t lo = rt_alloc_id(mb);
                uint32_t hi = rt_alloc_id(mb);
                spv_emit_op3(fn_buf, RT_OP_LOAD, mb->t_u32, lo, sptr0);
                spv_emit_op3(fn_buf, RT_OP_LOAD, mb->t_u32, hi, sptr1);
                uint32_t v2_val = rt_alloc_id(mb);
                spv_emit_op4(fn_buf, RT_OP_COMPOSITE_CONSTRUCT, mb->t_v2u32, v2_val, lo, hi);
                uint32_t out_val = rt_alloc_id(mb);
                if (tm->op == RT_OP_TYPE_POINTER) {
                    uint32_t u64_val = rt_alloc_id(mb);
                    spv_emit_op3(fn_buf, RT_OP_BITCAST, mb->t_u64, u64_val, v2_val);
                    spv_emit_op3(fn_buf, RT_OP_CONVERT_U_TO_PTR, type_id, out_val, u64_val);
                } else {
                    spv_emit_op3(fn_buf, RT_OP_BITCAST, type_id, out_val, v2_val);
                }
                spv_emit_op2(fn_buf, RT_OP_STORE, elem_ptr, out_val);
            }
            *slot_io += 2;
            return;
        }

        /* 32-bit, 16-bit, 8-bit, or bool scalar */
        uint32_t idx = rt_get_const_u32(mb, *slot_io);
        uint32_t sptr = rt_alloc_id(mb);
        spv_emit_op4(fn_buf, RT_OP_ACCESS_CHAIN, mb->t_ptr_priv_u32, sptr, backing_arr_var_id, idx);

        if (is_pack) {
            uint32_t val = rt_alloc_id(mb);
            spv_emit_op3(fn_buf, RT_OP_LOAD, type_id, val, elem_ptr);
            uint32_t uval = val;
            if (type_id == mb->t_u32) {
                uval = val;
            } else if (tm->op == RT_OP_TYPE_BOOL) {
                uval = rt_alloc_id(mb);
                spv_emit_op5(fn_buf, RT_OP_SELECT, mb->t_u32, uval, val, mb->c_u32_1, mb->c_u32_0);
            } else if (tm->count == 32) {
                uval = rt_alloc_id(mb);
                spv_emit_op3(fn_buf, RT_OP_BITCAST, mb->t_u32, uval, val);
            } else if (tm->op == RT_OP_TYPE_FLOAT && tm->count == 16) {
                uint32_t u16_t = rt_find_or_emit_type(mb, RT_OP_TYPE_INT, 16, 0, 0);
                uint32_t u16_v = rt_alloc_id(mb);
                spv_emit_op3(fn_buf, RT_OP_BITCAST, u16_t, u16_v, val);
                uval = rt_alloc_id(mb);
                spv_emit_op3(fn_buf, RT_OP_UCONVERT, mb->t_u32, uval, u16_v);
            } else if (tm->op == RT_OP_TYPE_INT && tm->count < 32) {
                uval = rt_alloc_id(mb);
                spv_emit_op3(fn_buf, tm->extra ? RT_OP_SCONVERT : RT_OP_UCONVERT, mb->t_u32, uval, val);
            }
            spv_emit_op2(fn_buf, RT_OP_STORE, sptr, uval);
        } else {
            uint32_t uval = rt_alloc_id(mb);
            spv_emit_op3(fn_buf, RT_OP_LOAD, mb->t_u32, uval, sptr);
            uint32_t val = uval;
            if (type_id == mb->t_u32) {
                val = uval;
            } else if (tm->op == RT_OP_TYPE_BOOL) {
                val = rt_alloc_id(mb);
                spv_emit_op4(fn_buf, RT_OP_INOT_EQUAL, mb->t_bool, val, uval, mb->c_u32_0);
            } else if (tm->count == 32) {
                val = rt_alloc_id(mb);
                spv_emit_op3(fn_buf, RT_OP_BITCAST, type_id, val, uval);
            } else if (tm->op == RT_OP_TYPE_FLOAT && tm->count == 16) {
                uint32_t u16_t = rt_find_or_emit_type(mb, RT_OP_TYPE_INT, 16, 0, 0);
                uint32_t u16_v = rt_alloc_id(mb);
                spv_emit_op3(fn_buf, RT_OP_UCONVERT, u16_t, u16_v, uval);
                val = rt_alloc_id(mb);
                spv_emit_op3(fn_buf, RT_OP_BITCAST, type_id, val, u16_v);
            } else if (tm->op == RT_OP_TYPE_INT && tm->count < 32) {
                val = rt_alloc_id(mb);
                spv_emit_op3(fn_buf, tm->extra ? RT_OP_SCONVERT : RT_OP_UCONVERT, type_id, val, uval);
            }
            spv_emit_op2(fn_buf, RT_OP_STORE, elem_ptr, val);
        }
        (*slot_io)++;
        return;
    }

    if (tm->op == RT_OP_TYPE_VECTOR || tm->op == RT_OP_TYPE_MATRIX || tm->op == RT_OP_TYPE_ARRAY) {
        if (chain_depth >= 12) return;
        uint32_t elem_type = tm->elem_type;
        uint32_t len = tm->count;
        for (uint32_t i = 0; i < len && *slot_io < max_slots; i++) {
            idx_chain[chain_depth] = rt_get_const_u32(mb, i);
            rt_emit_pack_unpack_type(mb, fn_buf, base_var_id, elem_type,
                                     idx_chain, chain_depth + 1,
                                     slot_io, max_slots, backing_arr_var_id, is_pack);
        }
        return;
    }

    if (tm->op == RT_OP_TYPE_STRUCT) {
        if (chain_depth >= 12) return;
        uint32_t mcount = tm->count;
        if (mcount > RT_TYPE_META_MAX_MEMBERS) mcount = RT_TYPE_META_MAX_MEMBERS;
        for (uint32_t m = 0; m < mcount && *slot_io < max_slots; m++) {
            idx_chain[chain_depth] = rt_get_const_i32(mb, (int32_t)m);
            rt_emit_pack_unpack_type(mb, fn_buf, base_var_id, tm->members[m],
                                     idx_chain, chain_depth + 1,
                                     slot_io, max_slots, backing_arr_var_id, is_pack);
        }
        return;
    }
}

/* ── Copy Canonical RT BuiltIn Globals into Stage's Private BuiltIn Vars ──── */

static void rt_emit_stage_builtin_sync(struct RtMegaBuilder *mb, struct SpvBuf *fn_buf,
                                       const struct RtStageParse *sp) {
    for (uint32_t i = 0; i < sp->rt_builtin_count; i++) {
        uint32_t var_id = sp->rt_builtins[i].var_id;
        uint32_t dst_type = sp->rt_builtins[i].pointee_type;
        uint32_t bi = sp->rt_builtins[i].builtin;

        uint32_t src_var = 0;
        uint32_t src_type = 0;
        switch (bi) {
            case RT_BUILTIN_LAUNCH_ID_KHR:             src_var = mb->g_launch_id;           src_type = mb->t_v3u32; break;
            case RT_BUILTIN_LAUNCH_SIZE_KHR:           src_var = mb->g_launch_size;         src_type = mb->t_v3u32; break;
            case RT_BUILTIN_WORLD_RAY_ORIGIN_KHR:      src_var = mb->g_world_ray_orig;      src_type = mb->t_v3f32; break;
            case RT_BUILTIN_WORLD_RAY_DIRECTION_KHR:   src_var = mb->g_world_ray_dir;       src_type = mb->t_v3f32; break;
            case RT_BUILTIN_OBJECT_RAY_ORIGIN_KHR:     src_var = mb->g_obj_ray_orig;        src_type = mb->t_v3f32; break;
            case RT_BUILTIN_OBJECT_RAY_DIRECTION_KHR:  src_var = mb->g_obj_ray_dir;         src_type = mb->t_v3f32; break;
            case RT_BUILTIN_RAY_TMIN_KHR:              src_var = mb->g_ray_tmin;            src_type = mb->t_f32;   break;
            case RT_BUILTIN_RAY_TMAX_KHR:
            case RT_BUILTIN_HIT_T_NV:                  src_var = mb->g_ray_tmax;            src_type = mb->t_f32;   break;
            case RT_BUILTIN_INCOMING_RAY_FLAGS_KHR:    src_var = mb->g_incoming_ray_flags;  src_type = mb->t_u32;   break;
            case RT_BUILTIN_CULL_MASK_KHR:             src_var = mb->g_cull_mask;           src_type = mb->t_u32;   break;
            case RT_BUILTIN_HIT_KIND_KHR:              src_var = mb->g_hit_kind;            src_type = mb->t_u32;   break;
            case RT_BUILTIN_INSTANCE_CUSTOM_INDEX_KHR: src_var = mb->g_instance_custom_idx; src_type = mb->t_u32;   break;
            case RT_BUILTIN_INSTANCE_ID:
            case RT_BUILTIN_INSTANCE_INDEX:            src_var = mb->g_instance_id;         src_type = mb->t_u32;   break;
            case RT_BUILTIN_PRIMITIVE_ID:              src_var = mb->g_primitive_id;        src_type = mb->t_u32;   break;
            case RT_BUILTIN_RAY_GEOMETRY_INDEX_KHR:    src_var = mb->g_geom_index;          src_type = mb->t_u32;   break;
            case RT_BUILTIN_OBJECT_TO_WORLD_KHR:       src_var = mb->g_obj_to_world;        src_type = mb->t_mat4x3f32; break;
            case RT_BUILTIN_WORLD_TO_OBJECT_KHR:       src_var = mb->g_world_to_obj;        src_type = mb->t_mat4x3f32; break;
            default: break;
        }
        if (!src_var) continue;

        uint32_t val = rt_alloc_id(mb);
        spv_emit_op3(fn_buf, RT_OP_LOAD, src_type, val, src_var);
        if (dst_type != src_type) {
            uint32_t conv = rt_alloc_id(mb);
            if (src_type == mb->t_mat4x3f32 && dst_type == mb->t_mat3x4f32) {
                spv_emit_op3(fn_buf, RT_OP_TRANSPOSE, dst_type, conv, val);
            } else {
                spv_emit_op3(fn_buf, RT_OP_BITCAST, dst_type, conv, val);
            }
            val = conv;
        }
        spv_emit_op2(fn_buf, RT_OP_STORE, var_id, val);
    }
}

static bool rt_is_rt_builtin(uint32_t bi) {
    switch (bi) {
        case RT_BUILTIN_INSTANCE_ID:
        case RT_BUILTIN_PRIMITIVE_ID:
        case RT_BUILTIN_INSTANCE_INDEX:
        case RT_BUILTIN_LAUNCH_ID_KHR:
        case RT_BUILTIN_LAUNCH_SIZE_KHR:
        case RT_BUILTIN_WORLD_RAY_ORIGIN_KHR:
        case RT_BUILTIN_WORLD_RAY_DIRECTION_KHR:
        case RT_BUILTIN_OBJECT_RAY_ORIGIN_KHR:
        case RT_BUILTIN_OBJECT_RAY_DIRECTION_KHR:
        case RT_BUILTIN_RAY_TMIN_KHR:
        case RT_BUILTIN_RAY_TMAX_KHR:
        case RT_BUILTIN_INSTANCE_CUSTOM_INDEX_KHR:
        case RT_BUILTIN_OBJECT_TO_WORLD_KHR:
        case RT_BUILTIN_WORLD_TO_OBJECT_KHR:
        case RT_BUILTIN_HIT_T_NV:
        case RT_BUILTIN_HIT_KIND_KHR:
        case RT_BUILTIN_INCOMING_RAY_FLAGS_KHR:
        case RT_BUILTIN_RAY_GEOMETRY_INDEX_KHR:
        case RT_BUILTIN_CULL_MASK_KHR:
            return true;
        default:
            return false;
    }
}


/* ── Stage Preamble Parser & Global Lowering ──────────────────────────────── */

static const void *rt_find_spec_value(const struct RtStageRecord *sr, uint32_t spec_id, size_t *out_size) {
    if (!sr->spec_entries || !sr->spec_data) return NULL;
    for (uint32_t i = 0; i < sr->spec_entry_count; i++) {
        if (sr->spec_entries[i].spec_id == spec_id) {
            uint32_t off = sr->spec_entries[i].offset;
            uint32_t sz  = sr->spec_entries[i].size;
            if ((size_t)off + sz <= sr->spec_data_size) {
                if (out_size) *out_size = sz;
                return sr->spec_data + off;
            }
        }
    }
    return NULL;
}

static bool rt_parse_and_emit_stage_globals(struct RtMegaBuilder *mb,
                                            const struct RtStageRecord *sr,
                                            struct RtStageParse *sp) {
    memset(sp, 0, sizeof(*sp));
    const uint32_t *code = sr->words;
    size_t count = sr->word_count;
    if (!code || count < 5 || code[0] != RT_SPV_MAGIC) return false;

    uint32_t bound = code[3];
    if (bound == 0 || bound > 0x400000u) return false;
    sp->bound = bound;

    sp->remap            = (uint32_t *)calloc(bound, sizeof(uint32_t));
    sp->remap_d1         = (uint32_t *)calloc(bound, sizeof(uint32_t));
    sp->id_type          = (uint32_t *)calloc(bound, sizeof(uint32_t));
    sp->is_psb_type      = (bool *)calloc(bound, sizeof(bool));
    sp->is_psb_id        = (bool *)calloc(bound, sizeof(bool));
    sp->var_sc           = (uint32_t *)calloc(bound, sizeof(uint32_t));
    sp->var_pointee_orig = (uint32_t *)calloc(bound, sizeof(uint32_t));
    sp->dec_builtin      = (uint32_t *)calloc(bound, sizeof(uint32_t));
    sp->dec_spec_id      = (uint32_t *)calloc(bound, sizeof(uint32_t));
    sp->dec_array_stride = (uint32_t *)calloc(bound, sizeof(uint32_t));
    sp->strip_dec_loc    = (bool *)calloc(bound, sizeof(bool));
    sp->const_u32_val    = (uint32_t *)calloc(bound, sizeof(uint32_t));
    sp->is_const_u32     = (bool *)calloc(bound, sizeof(bool));

    bool *has_decorate   = (bool *)calloc(bound, sizeof(bool));
    uint32_t *ptr_pointee_orig = (uint32_t *)calloc(bound, sizeof(uint32_t));
    uint32_t *ptr_sc_orig      = (uint32_t *)calloc(bound, sizeof(uint32_t));

    if (!sp->remap || !sp->remap_d1 || !sp->id_type || !sp->is_psb_type ||
        !sp->is_psb_id || !sp->var_sc || !sp->var_pointee_orig || !sp->dec_builtin ||
        !sp->dec_spec_id || !sp->dec_array_stride || !sp->strip_dec_loc ||
        !sp->const_u32_val || !sp->is_const_u32 || !has_decorate ||
        !ptr_pointee_orig || !ptr_sc_orig) {
        free(has_decorate);
        free(ptr_pointee_orig);
        free(ptr_sc_orig);
        return false;
    }

    /* Reserve a contiguous ID block for this stage (d0) */
    uint32_t id_base = mb->next_id;
    for (uint32_t k = 0; k < bound; k++) {
        uint32_t id = rt_alloc_id(mb);
        sp->remap[k] = (k == 0) ? 0 : id;
        sp->remap_d1[k] = sp->remap[k];
    }
    (void)id_base;

    /* Pass 1: Scan entry points, decorations, pointer types, variables, and OpTraceRayKHR */
    uint32_t expected_em = 0;
    switch (sr->stage) {
        case VK_SHADER_STAGE_RAYGEN_BIT_KHR:       expected_em = RT_EXEC_MODEL_RAYGEN_KHR; break;
        case VK_SHADER_STAGE_INTERSECTION_BIT_KHR: expected_em = RT_EXEC_MODEL_INTERSECTION_KHR; break;
        case VK_SHADER_STAGE_ANY_HIT_BIT_KHR:      expected_em = RT_EXEC_MODEL_ANY_HIT_KHR; break;
        case VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR:  expected_em = RT_EXEC_MODEL_CLOSEST_HIT_KHR; break;
        case VK_SHADER_STAGE_MISS_BIT_KHR:         expected_em = RT_EXEC_MODEL_MISS_KHR; break;
        case VK_SHADER_STAGE_CALLABLE_BIT_KHR:     expected_em = RT_EXEC_MODEL_CALLABLE_KHR; break;
        default: break;
    }

    for (size_t i = 5; i < count; ) {
        uint32_t wc = code[i] >> 16;
        uint32_t op = code[i] & 0xFFFFu;
        if (wc == 0 || i + wc > count) break;

        if (op == RT_OP_ENTRY_POINT && wc >= 4) {
            uint32_t em = code[i + 1];
            uint32_t fn_id = code[i + 2];
            const char *ep_name = (const char *)&code[i + 3];
            if (em == expected_em) {
                if (sp->orig_entry_fn_id == 0 ||
                    (sr->entry_name && strcmp(ep_name, sr->entry_name) == 0)) {
                    sp->orig_entry_fn_id = fn_id;
                }
            } else if (sp->orig_entry_fn_id == 0 &&
                       em >= RT_EXEC_MODEL_RAYGEN_KHR && em <= RT_EXEC_MODEL_CALLABLE_KHR) {
                sp->orig_entry_fn_id = fn_id;
            }
        } else if (op == RT_OP_DECORATE && wc >= 3) {
            uint32_t target = code[i + 1];
            uint32_t dec = code[i + 2];
            if (target < bound) {
                has_decorate[target] = true;
                if (dec == RT_DEC_BUILTIN && wc >= 4) {
                    uint32_t bi = code[i + 3];
                    sp->dec_builtin[target] = bi + 1;
                    if (bi == RT_BUILTIN_OBJECT_RAY_ORIGIN_KHR ||
                        bi == RT_BUILTIN_OBJECT_RAY_DIRECTION_KHR) {
                        mb->uses_object_ray = true;
                    } else if (bi == RT_BUILTIN_OBJECT_TO_WORLD_KHR) {
                        mb->uses_obj_to_world = true;
                    } else if (bi == RT_BUILTIN_WORLD_TO_OBJECT_KHR) {
                        mb->uses_world_to_obj = true;
                    }
                } else if (dec == RT_DEC_SPEC_ID && wc >= 4) {
                    sp->dec_spec_id[target] = code[i + 3] + 1;
                } else if (dec == RT_DEC_ARRAY_STRIDE && wc >= 4) {
                    sp->dec_array_stride[target] = code[i + 3];
                }
            }
        } else if (op == RT_OP_MEMBER_DECORATE && wc >= 4) {
            uint32_t target = code[i + 1];
            if (target < bound) {
                has_decorate[target] = true;
                uint32_t dec = code[i + 3];
                if (dec == RT_DEC_BUILTIN && wc >= 5) {
                    uint32_t bi = code[i + 4];
                    if (bi == RT_BUILTIN_OBJECT_RAY_ORIGIN_KHR ||
                        bi == RT_BUILTIN_OBJECT_RAY_DIRECTION_KHR) {
                        mb->uses_object_ray = true;
                    } else if (bi == RT_BUILTIN_OBJECT_TO_WORLD_KHR) {
                        mb->uses_obj_to_world = true;
                    } else if (bi == RT_BUILTIN_WORLD_TO_OBJECT_KHR) {
                        mb->uses_world_to_obj = true;
                    }
                }
            }
        } else if (op == RT_OP_TYPE_POINTER && wc >= 4) {
            uint32_t id = code[i + 1];
            uint32_t sc = code[i + 2];
            uint32_t pt = code[i + 3];
            if (id < bound) {
                ptr_sc_orig[id] = sc;
                ptr_pointee_orig[id] = pt;
                if (sc == RT_SC_PHYSICAL_STORAGE_BUFFER ||
                    sc == RT_SC_PUSH_CONSTANT ||
                    sc == RT_SC_SHADER_RECORD_BUFFER_KHR) {
                    sp->is_psb_type[id] = true;
                }
            }
        } else if (op == RT_OP_VARIABLE && wc >= 4) {
            uint32_t rtype = code[i + 1];
            uint32_t id    = code[i + 2];
            uint32_t sc    = code[i + 3];
            if (id < bound) {
                sp->var_sc[id] = sc;
                sp->id_type[id] = rtype;
                if (rtype < bound) {
                    sp->var_pointee_orig[id] = ptr_pointee_orig[rtype];
                    if (sp->is_psb_type[rtype]) sp->is_psb_id[id] = true;
                }
                if (sc == RT_SC_RAY_PAYLOAD_KHR ||
                    sc == RT_SC_INCOMING_RAY_PAYLOAD_KHR ||
                    sc == RT_SC_HIT_ATTRIBUTE_KHR ||
                    sc == RT_SC_CALLABLE_DATA_KHR ||
                    sc == RT_SC_INCOMING_CALLABLE_DATA_KHR) {
                    sp->strip_dec_loc[id] = true;
                } else if (sc == RT_SC_INPUT && sp->dec_builtin[id] != 0 &&
                           rt_is_rt_builtin(sp->dec_builtin[id] - 1)) {
                    sp->strip_dec_loc[id] = true;
                }
            }
        } else if (op == RT_OP_TRACE_RAY_KHR) {
            sp->has_trace_ray = true;
        }
        i += wc;
    }

    /* Pass 2: Emit capabilities, extensions, decorations, types, constants, and module-scope variables */
    for (size_t i = 5; i < count; ) {
        uint32_t wc = code[i] >> 16;
        uint32_t op = code[i] & 0xFFFFu;
        if (wc == 0 || i + wc > count) break;
        if (op == RT_OP_FUNCTION) break;

        switch (op) {
            case RT_OP_CAPABILITY:
                if (wc >= 2) rt_add_cap(mb, code[i + 1]);
                break;

            case RT_OP_EXTENSION:
                if (wc >= 2) rt_add_ext(mb, (const char *)&code[i + 1]);
                break;

            case RT_OP_EXT_INST_IMPORT:
                if (wc >= 3 && code[i + 1] < bound) {
                    sp->remap[code[i + 1]] = mb->ext_glsl450;
                    sp->remap_d1[code[i + 1]] = mb->ext_glsl450;
                }
                break;

            case RT_OP_MEMORY_MODEL:
                if (wc >= 3 && code[i + 2] == 3u /* Vulkan */) {
                    mb->has_vulkan_memory_model = true;
                }
                break;

            case RT_OP_ENTRY_POINT:
            case RT_OP_EXECUTION_MODE:
            case RT_OP_NAME:
            case RT_OP_MEMBER_NAME:
            case RT_OP_STRING:
            case RT_OP_SOURCE:
            case RT_OP_SOURCE_CONTINUED:
            case RT_OP_SOURCE_EXTENSION:
            case RT_OP_LINE:
            case RT_OP_NO_LINE:
            case RT_OP_MODULE_PROCESSED:
                /* Strip debug/entry-point instructions from individual stages */
                break;

            case RT_OP_DECORATE: {
                if (wc < 3) break;
                uint32_t target = code[i + 1];
                uint32_t dec    = code[i + 2];
                if (target >= bound) break;
                /* Strip SpecId decorations because we bake spec constants into OpConstant* */
                if (dec == RT_DEC_SPEC_ID) break;
                /* Strip Location decorations on RT payload/hitattr/callable variables */
                if (dec == RT_DEC_LOCATION && sp->strip_dec_loc[target]) break;
                /* Strip BuiltIn decorations on RT-specific Input builtins (converted to Private) */
                if (dec == RT_DEC_BUILTIN && wc >= 4 && rt_is_rt_builtin(code[i + 3])) break;

                spv_emit_word(&mb->decorations, code[i]);
                spv_emit_word(&mb->decorations, sp->remap[target]);
                for (uint32_t k = 2; k < wc; k++) {
                    spv_emit_word(&mb->decorations, code[i + k]);
                }
                break;
            }

            case RT_OP_MEMBER_DECORATE: {
                if (wc < 4) break;
                uint32_t target = code[i + 1];
                if (target >= bound) break;
                spv_emit_word(&mb->decorations, code[i]);
                spv_emit_word(&mb->decorations, sp->remap[target]);
                for (uint32_t k = 2; k < wc; k++) {
                    spv_emit_word(&mb->decorations, code[i + k]);
                }
                break;
            }

            case RT_OP_TYPE_VOID:
            case RT_OP_TYPE_BOOL:
            case RT_OP_TYPE_RAY_QUERY_KHR:
            case RT_OP_TYPE_ACCEL_STRUCT_KHR: {
                uint32_t orig_id = code[i + 1];
                if (orig_id < bound && !has_decorate[orig_id]) {
                    uint32_t cid = rt_find_or_emit_type(mb, op, 0, 0, 0);
                    sp->remap[orig_id] = cid;
                    sp->remap_d1[orig_id] = cid;
                } else if (orig_id < bound) {
                    spv_emit_op1(&mb->types_consts_globals, op, sp->remap[orig_id]);
                    mb->type_meta[sp->remap[orig_id]].op = op;
                }
                break;
            }

            case RT_OP_TYPE_FLOAT: {
                uint32_t orig_id = code[i + 1];
                uint32_t width   = code[i + 2];
                if (orig_id < bound && !has_decorate[orig_id]) {
                    uint32_t cid = rt_find_or_emit_type(mb, op, width, 0, 0);
                    sp->remap[orig_id] = cid;
                    sp->remap_d1[orig_id] = cid;
                } else if (orig_id < bound) {
                    uint32_t mid = sp->remap[orig_id];
                    spv_emit_op2(&mb->types_consts_globals, op, mid, width);
                    mb->type_meta[mid].op = op;
                    mb->type_meta[mid].count = width;
                }
                break;
            }

            case RT_OP_TYPE_INT: {
                uint32_t orig_id = code[i + 1];
                uint32_t width   = code[i + 2];
                uint32_t sig     = code[i + 3];
                if (orig_id < bound && !has_decorate[orig_id]) {
                    uint32_t cid = rt_find_or_emit_type(mb, op, width, sig, 0);
                    sp->remap[orig_id] = cid;
                    sp->remap_d1[orig_id] = cid;
                } else if (orig_id < bound) {
                    uint32_t mid = sp->remap[orig_id];
                    spv_emit_op3(&mb->types_consts_globals, op, mid, width, sig);
                    mb->type_meta[mid].op = op;
                    mb->type_meta[mid].count = width;
                    mb->type_meta[mid].extra = sig;
                }
                break;
            }

            case RT_OP_TYPE_VECTOR:
            case RT_OP_TYPE_MATRIX: {
                uint32_t orig_id = code[i + 1];
                uint32_t elem    = (code[i + 2] < bound) ? sp->remap[code[i + 2]] : code[i + 2];
                uint32_t cnt     = code[i + 3];
                if (orig_id < bound && !has_decorate[orig_id]) {
                    uint32_t cid = rt_find_or_emit_type(mb, op, elem, cnt, 0);
                    sp->remap[orig_id] = cid;
                    sp->remap_d1[orig_id] = cid;
                } else if (orig_id < bound) {
                    uint32_t mid = sp->remap[orig_id];
                    spv_emit_op3(&mb->types_consts_globals, op, mid, elem, cnt);
                    mb->type_meta[mid].op = op;
                    mb->type_meta[mid].elem_type = elem;
                    mb->type_meta[mid].count = cnt;
                }
                break;
            }

            case RT_OP_TYPE_POINTER: {
                uint32_t orig_id = code[i + 1];
                uint32_t sc      = code[i + 2];
                uint32_t pt_orig = code[i + 3];
                uint32_t pt      = (pt_orig < bound) ? sp->remap[pt_orig] : pt_orig;

                /* Map RT storage classes */
                if (sc == RT_SC_RAY_PAYLOAD_KHR ||
                    sc == RT_SC_INCOMING_RAY_PAYLOAD_KHR ||
                    sc == RT_SC_HIT_ATTRIBUTE_KHR ||
                    sc == RT_SC_CALLABLE_DATA_KHR ||
                    sc == RT_SC_INCOMING_CALLABLE_DATA_KHR) {
                    sc = RT_SC_PRIVATE;
                } else if (sc == RT_SC_PUSH_CONSTANT ||
                           sc == RT_SC_SHADER_RECORD_BUFFER_KHR) {
                    sc = RT_SC_PHYSICAL_STORAGE_BUFFER;
                } else if (sc == RT_SC_INPUT) {
                    /*
                     * Note: In RT shaders, Input pointers to RT BuiltIns become Private.
                     * Because we emit both Private and Input pointer types on demand,
                     * we check if any non-RT Input variable uses this pointer type.
                     */
                    bool used_by_non_rt_input = false;
                    for (uint32_t v = 1; v < bound; v++) {
                        if (sp->var_sc[v] == RT_SC_INPUT && sp->id_type[v] == orig_id) {
                            uint32_t bi = sp->dec_builtin[v];
                            if (bi == 0 || !rt_is_rt_builtin(bi - 1)) {
                                used_by_non_rt_input = true;
                                break;
                            }
                        }
                    }
                    if (!used_by_non_rt_input) {
                        sc = RT_SC_PRIVATE;
                    }
                }

                if (orig_id < bound && !has_decorate[orig_id]) {
                    uint32_t cid = rt_find_or_emit_type(mb, RT_OP_TYPE_POINTER, sc, pt, 0);
                    sp->remap[orig_id] = cid;
                    sp->remap_d1[orig_id] = cid;
                } else if (orig_id < bound) {
                    uint32_t mid = sp->remap[orig_id];
                    spv_emit_op3(&mb->types_consts_globals, RT_OP_TYPE_POINTER, mid, sc, pt);
                    mb->type_meta[mid].op = RT_OP_TYPE_POINTER;
                    mb->type_meta[mid].extra = sc;
                    mb->type_meta[mid].elem_type = pt;
                }
                break;
            }

            case RT_OP_TYPE_ARRAY: {
                uint32_t orig_id = code[i + 1];
                uint32_t elem    = (code[i + 2] < bound) ? sp->remap[code[i + 2]] : code[i + 2];
                uint32_t len_id  = (code[i + 3] < bound) ? sp->remap[code[i + 3]] : code[i + 3];
                uint32_t mid     = sp->remap[orig_id];
                spv_emit_op3(&mb->types_consts_globals, RT_OP_TYPE_ARRAY, mid, elem, len_id);
                mb->type_meta[mid].op = RT_OP_TYPE_ARRAY;
                mb->type_meta[mid].elem_type = elem;
                mb->type_meta[mid].count = (code[i + 3] < bound && sp->is_const_u32[code[i + 3]])
                                           ? sp->const_u32_val[code[i + 3]] : 1;
                break;
            }

            case RT_OP_TYPE_STRUCT: {
                uint32_t orig_id = code[i + 1];
                uint32_t mid     = sp->remap[orig_id];
                spv_emit_word(&mb->types_consts_globals, code[i]);
                spv_emit_word(&mb->types_consts_globals, mid);
                uint32_t mcount = wc - 2;
                mb->type_meta[mid].op = RT_OP_TYPE_STRUCT;
                mb->type_meta[mid].count = mcount;
                for (uint32_t m = 0; m < mcount; m++) {
                    uint32_t mtype = (code[i + 2 + m] < bound) ? sp->remap[code[i + 2 + m]] : code[i + 2 + m];
                    spv_emit_word(&mb->types_consts_globals, mtype);
                    if (m < RT_TYPE_META_MAX_MEMBERS) {
                        mb->type_meta[mid].members[m] = mtype;
                    }
                }
                break;
            }

            case RT_OP_TYPE_FUNCTION: {
                uint32_t orig_id = code[i + 1];
                uint32_t opnds[48];
                uint32_t opnd_count = (wc > 2) ? (wc - 2) : 0;
                if (opnd_count > 48) opnd_count = 48;
                for (uint32_t k = 0; k < opnd_count; k++) {
                    uint32_t opnd = code[i + 2 + k];
                    opnds[k] = (opnd < bound) ? sp->remap[opnd] : opnd;
                }
                uint32_t mid = rt_find_or_emit_complex_type(mb, RT_OP_TYPE_FUNCTION, opnds, opnd_count);
                sp->remap[orig_id]    = mid;
                sp->remap_d1[orig_id] = mid;
                break;
            }

            case RT_OP_TYPE_SAMPLER: {
                uint32_t orig_id = code[i + 1];
                if (mb->t_sampler == 0) {
                    mb->t_sampler = rt_find_or_emit_complex_type(mb, RT_OP_TYPE_SAMPLER, NULL, 0);
                }
                sp->remap[orig_id]    = mb->t_sampler;
                sp->remap_d1[orig_id] = mb->t_sampler;
                break;
            }

            case RT_OP_TYPE_SAMPLED_IMAGE: {
                uint32_t orig_id  = code[i + 1];
                uint32_t img_orig = code[i + 2];
                uint32_t opnds[1] = { (img_orig < bound) ? sp->remap[img_orig] : img_orig };
                uint32_t mid = rt_find_or_emit_complex_type(mb, RT_OP_TYPE_SAMPLED_IMAGE, opnds, 1);
                sp->remap[orig_id]    = mid;
                sp->remap_d1[orig_id] = mid;
                break;
            }

            case RT_OP_TYPE_IMAGE: {
                uint32_t orig_id = code[i + 1];
                uint32_t opnds[16];
                uint32_t opnd_count = (wc > 2) ? (wc - 2) : 0;
                if (opnd_count > 16) opnd_count = 16;
                uint32_t smp_orig = code[i + 2];
                opnds[0] = (smp_orig < bound) ? sp->remap[smp_orig] : smp_orig;
                for (uint32_t k = 1; k < opnd_count; k++) {
                    opnds[k] = code[i + 2 + k]; /* Dim, Depth, Arrayed, MS, Sampled, Format are literals */
                }
                uint32_t mid = rt_find_or_emit_complex_type(mb, RT_OP_TYPE_IMAGE, opnds, opnd_count);
                sp->remap[orig_id]    = mid;
                sp->remap_d1[orig_id] = mid;
                break;
            }

            case RT_OP_TYPE_RUNTIME_ARRAY: {
                uint32_t orig_id = code[i + 1];
                uint32_t mid     = sp->remap[orig_id];
                spv_emit_word(&mb->types_consts_globals, code[i]);
                spv_emit_word(&mb->types_consts_globals, mid);
                for (uint32_t k = 2; k < wc; k++) {
                    uint32_t opnd = code[i + k];
                    spv_emit_word(&mb->types_consts_globals, (opnd < bound) ? sp->remap[opnd] : opnd);
                }
                mb->type_meta[mid].op = op;
                break;
            }

            case RT_OP_TYPE_FORWARD_POINTER: {
                uint32_t ptr_id = (code[i + 1] < bound) ? sp->remap[code[i + 1]] : code[i + 1];
                uint32_t sc     = code[i + 2];
                if (sc == RT_SC_PUSH_CONSTANT || sc == RT_SC_SHADER_RECORD_BUFFER_KHR) {
                    sc = RT_SC_PHYSICAL_STORAGE_BUFFER;
                }
                spv_emit_op2(&mb->types_consts_globals, RT_OP_TYPE_FORWARD_POINTER, ptr_id, sc);
                break;
            }

            case RT_OP_CONSTANT:
            case RT_OP_SPEC_CONSTANT: {
                uint32_t rtype   = (code[i + 1] < bound) ? sp->remap[code[i + 1]] : code[i + 1];
                uint32_t orig_id = code[i + 2];
                uint32_t mid     = sp->remap[orig_id];
                sp->id_type[orig_id] = code[i + 1];

                uint32_t lit_words[4] = { 0, 0, 0, 0 };
                uint32_t lit_count = wc - 3;
                for (uint32_t k = 0; k < lit_count && k < 4; k++) {
                    lit_words[k] = code[i + 3 + k];
                }
                if (op == RT_OP_SPEC_CONSTANT && sp->dec_spec_id[orig_id] != 0) {
                    size_t sz = 0;
                    const void *sval = rt_find_spec_value(sr, sp->dec_spec_id[orig_id] - 1, &sz);
                    if (sval && sz > 0) {
                        memcpy(lit_words, sval, sz < sizeof(lit_words) ? sz : sizeof(lit_words));
                    }
                }
                if (lit_count == 1 && (rtype == mb->t_u32 || rtype == mb->t_i32)) {
                    sp->is_const_u32[orig_id] = true;
                    sp->const_u32_val[orig_id] = lit_words[0];
                }
                spv_emit_word(&mb->types_consts_globals, ((3u + lit_count) << 16) | RT_OP_CONSTANT);
                spv_emit_word(&mb->types_consts_globals, rtype);
                spv_emit_word(&mb->types_consts_globals, mid);
                for (uint32_t k = 0; k < lit_count; k++) {
                    spv_emit_word(&mb->types_consts_globals, lit_words[k]);
                }
                break;
            }

            case RT_OP_CONSTANT_TRUE:
            case RT_OP_CONSTANT_FALSE:
            case RT_OP_SPEC_CONSTANT_TRUE:
            case RT_OP_SPEC_CONSTANT_FALSE: {
                uint32_t rtype   = (code[i + 1] < bound) ? sp->remap[code[i + 1]] : code[i + 1];
                uint32_t orig_id = code[i + 2];
                uint32_t mid     = sp->remap[orig_id];
                sp->id_type[orig_id] = code[i + 1];
                bool bval = (op == RT_OP_CONSTANT_TRUE || op == RT_OP_SPEC_CONSTANT_TRUE);
                if ((op == RT_OP_SPEC_CONSTANT_TRUE || op == RT_OP_SPEC_CONSTANT_FALSE) &&
                    sp->dec_spec_id[orig_id] != 0) {
                    size_t sz = 0;
                    const void *sval = rt_find_spec_value(sr, sp->dec_spec_id[orig_id] - 1, &sz);
                    if (sval && sz >= 4) {
                        uint32_t u = 0;
                        memcpy(&u, sval, 4);
                        bval = (u != 0);
                    }
                }
                spv_emit_op2(&mb->types_consts_globals,
                             bval ? RT_OP_CONSTANT_TRUE : RT_OP_CONSTANT_FALSE,
                             rtype, mid);
                break;
            }

            case RT_OP_CONSTANT_COMPOSITE:
            case RT_OP_SPEC_CONSTANT_COMPOSITE: {
                uint32_t rtype   = (code[i + 1] < bound) ? sp->remap[code[i + 1]] : code[i + 1];
                uint32_t orig_id = code[i + 2];
                uint32_t mid     = sp->remap[orig_id];
                sp->id_type[orig_id] = code[i + 1];
                spv_emit_word(&mb->types_consts_globals, (wc << 16) | RT_OP_CONSTANT_COMPOSITE);
                spv_emit_word(&mb->types_consts_globals, rtype);
                spv_emit_word(&mb->types_consts_globals, mid);
                for (uint32_t k = 3; k < wc; k++) {
                    uint32_t opnd = code[i + k];
                    spv_emit_word(&mb->types_consts_globals, (opnd < bound) ? sp->remap[opnd] : opnd);
                }
                break;
            }

            case RT_OP_CONSTANT_NULL: {
                uint32_t rtype   = (code[i + 1] < bound) ? sp->remap[code[i + 1]] : code[i + 1];
                uint32_t orig_id = code[i + 2];
                uint32_t mid     = sp->remap[orig_id];
                sp->id_type[orig_id] = code[i + 1];
                spv_emit_op2(&mb->types_consts_globals, RT_OP_CONSTANT_NULL, rtype, mid);
                break;
            }

            case RT_OP_SPEC_CONSTANT_OP: {
                if (wc < 4) break;
                uint32_t rtype_orig = code[i + 1];
                uint32_t orig_id    = code[i + 2];
                uint32_t spec_op    = code[i + 3]; /* Literal opcode: DO NOT REMAP! */
                uint32_t rtype      = (rtype_orig < bound) ? sp->remap[rtype_orig] : rtype_orig;
                uint32_t mid        = sp->remap[orig_id];
                sp->id_type[orig_id] = code[i + 1];

                /* Constant fold if operands are known constants */
                bool folded = false;
                if (wc == 6 && (spec_op == 170u /* OpIEqual */ || spec_op == 171u /* OpINotEqual */)) {
                    uint32_t op0 = code[i + 4];
                    uint32_t op1 = code[i + 5];
                    if (op0 < bound && op1 < bound && sp->is_const_u32[op0] && sp->is_const_u32[op1]) {
                        bool eq = (sp->const_u32_val[op0] == sp->const_u32_val[op1]);
                        bool val = (spec_op == 170u) ? eq : !eq;
                        spv_emit_op2(&mb->types_consts_globals,
                                     val ? RT_OP_CONSTANT_TRUE : RT_OP_CONSTANT_FALSE,
                                     rtype, mid);
                        folded = true;
                    }
                }

                if (!folded) {
                    spv_emit_word(&mb->types_consts_globals, (wc << 16) | RT_OP_SPEC_CONSTANT_OP);
                    spv_emit_word(&mb->types_consts_globals, rtype);
                    spv_emit_word(&mb->types_consts_globals, mid);
                    spv_emit_word(&mb->types_consts_globals, spec_op);

                    for (uint32_t k = 4; k < wc; k++) {
                        uint32_t opnd = code[i + k];
                        bool is_lit = false;
                        if (spec_op == RT_OP_VECTOR_SHUFFLE && k >= 6) {
                            is_lit = true;
                        } else if (spec_op == RT_OP_COMPOSITE_EXTRACT && k >= 5) {
                            is_lit = true;
                        } else if (spec_op == RT_OP_COMPOSITE_INSERT && k >= 6) {
                            is_lit = true;
                        }
                        if (is_lit) {
                            spv_emit_word(&mb->types_consts_globals, opnd);
                        } else {
                            spv_emit_word(&mb->types_consts_globals, (opnd < bound) ? sp->remap[opnd] : opnd);
                        }
                    }
                }
                break;
            }

            case RT_OP_VARIABLE: {
                uint32_t rtype_orig = code[i + 1];
                uint32_t orig_id    = code[i + 2];
                uint32_t sc         = code[i + 3];
                uint32_t mid        = sp->remap[orig_id];
                uint32_t pt_orig    = sp->var_pointee_orig[orig_id];
                uint32_t pt_remapped= (pt_orig < bound) ? sp->remap[pt_orig] : pt_orig;

                /* PushConstant (9) and ShaderRecordBufferKHR (5343) become PSB pointers loaded at function entry */
                if (sc == RT_SC_PUSH_CONSTANT || sc == RT_SC_SHADER_RECORD_BUFFER_KHR) {
                    if (sp->psb_var_count < 16) {
                        uint32_t psb_ptr_type = rt_find_or_emit_type(mb, RT_OP_TYPE_POINTER,
                                                                     RT_SC_PHYSICAL_STORAGE_BUFFER,
                                                                     pt_remapped, 0);
                        sp->psb_vars_orig[sp->psb_var_count]     = orig_id;
                        sp->psb_vars_ptr_type[sp->psb_var_count] = psb_ptr_type;
                        sp->psb_vars_sc[sp->psb_var_count]       = sc;
                        sp->psb_var_count++;
                    }
                    break;
                }

                uint32_t new_sc = sc;
                if (sc == RT_SC_RAY_PAYLOAD_KHR) {
                    new_sc = RT_SC_PRIVATE;
                } else if (sc == RT_SC_INCOMING_RAY_PAYLOAD_KHR) {
                    new_sc = RT_SC_PRIVATE;
                    sp->incoming_payload_var  = mid;
                    sp->incoming_payload_type = pt_remapped;
                } else if (sc == RT_SC_HIT_ATTRIBUTE_KHR) {
                    new_sc = RT_SC_PRIVATE;
                    sp->hit_attr_var  = mid;
                    sp->hit_attr_type = pt_remapped;
                } else if (sc == RT_SC_CALLABLE_DATA_KHR) {
                    new_sc = RT_SC_PRIVATE;
                } else if (sc == RT_SC_INCOMING_CALLABLE_DATA_KHR) {
                    new_sc = RT_SC_PRIVATE;
                    sp->incoming_callable_var  = mid;
                    sp->incoming_callable_type = pt_remapped;
                } else if (sc == RT_SC_INPUT) {
                    uint32_t bi_plus_1 = sp->dec_builtin[orig_id];
                    if (bi_plus_1 != 0 && rt_is_rt_builtin(bi_plus_1 - 1)) {
                        new_sc = RT_SC_PRIVATE;
                        if (sp->rt_builtin_count < 32) {
                            sp->rt_builtins[sp->rt_builtin_count++] = (struct RtBuiltinVarMap){
                                mid, pt_remapped, bi_plus_1 - 1
                            };
                        }
                    }
                }

                uint32_t rtype_remapped = (rtype_orig < bound) ? sp->remap[rtype_orig] : rtype_orig;
                if (new_sc != sc) {
                    rtype_remapped = rt_find_or_emit_type(mb, RT_OP_TYPE_POINTER, new_sc, pt_remapped, 0);
                }

                if (wc >= 5) {
                    uint32_t init_id = (code[i + 4] < bound) ? sp->remap[code[i + 4]] : code[i + 4];
                    spv_emit_op4(&mb->types_consts_globals, RT_OP_VARIABLE, rtype_remapped, mid, new_sc, init_id);
                } else {
                    spv_emit_op3(&mb->types_consts_globals, RT_OP_VARIABLE, rtype_remapped, mid, new_sc);
                }
                rt_add_iface_var(mb, mid);
                break;
            }

            default: {
                /* Emit any other preamble instruction with remapped IDs */
                spv_emit_word(&mb->types_consts_globals, code[i]);
                for (uint32_t k = 1; k < wc; k++) {
                    uint32_t opnd = code[i + k];
                    spv_emit_word(&mb->types_consts_globals, (opnd < bound) ? sp->remap[opnd] : opnd);
                }
                break;
            }
        }
        i += wc;
    }

    free(has_decorate);
    free(ptr_pointee_orig);
    free(ptr_sc_orig);
    return sp->orig_entry_fn_id != 0;
}

/* ── Exact SPIR-V Operand Classifier (ID vs Literal) ──────────────────────── */

static bool rt_is_literal_operand(uint32_t op, uint32_t k, uint32_t wc,
                                  const uint32_t *inst, const struct RtStageParse *sp) {
    (void)wc;
    switch (op) {
        case RT_OP_EXT_INST:
            return (k == 4);
        case RT_OP_FUNCTION:
            return (k == 3);
        case RT_OP_VARIABLE:
            return (k == 3);
        case RT_OP_LOAD:
        case 64u /* OpCopyMemorySized */:
            if (k == 4) return true; /* MemoryAccess mask */
            if (k == 5 && (inst[4] & 0x2u)) return true; /* Aligned literal */
            return false;
        case RT_OP_STORE:
        case 63u /* OpCopyMemory */:
            if (k == 3) return true; /* MemoryAccess mask */
            if (k == 4 && (inst[3] & 0x2u)) return true; /* Aligned literal */
            return false;
        case RT_OP_VECTOR_SHUFFLE:
        case RT_OP_COMPOSITE_INSERT:
            return (k >= 5);
        case RT_OP_COMPOSITE_EXTRACT:
            return (k >= 4);
        case RT_OP_LOOP_MERGE:
            return (k >= 3);
        case RT_OP_SELECTION_MERGE:
            return (k >= 2);
        case RT_OP_BRANCH_CONDITIONAL:
            return (k >= 4);
        case RT_OP_SWITCH: {
            if (k < 3) return false;
            uint32_t sel_id = inst[1];
            bool is_64 = false;
            if (sel_id < sp->bound) {
                uint32_t stype = sp->id_type[sel_id];
                if (stype < sp->bound) {
                    /* Default 32-bit unless selector is 64-bit int */
                    (void)stype;
                }
            }
            if (is_64) {
                return (((k - 3) % 3) != 2);
            }
            return (((k - 3) % 2) == 0);
        }
        /* Image instructions with ImageOperands mask at word 4 */
        case 99u /* OpImageWrite */:
            return (k == 4);
        /* Image instructions with ImageOperands mask at word 5 */
        case 87u  /* OpImageSampleImplicitLod */:
        case 88u  /* OpImageSampleExplicitLod */:
        case 91u  /* OpImageSampleProjImplicitLod */:
        case 92u  /* OpImageSampleProjExplicitLod */:
        case 95u  /* OpImageFetch */:
        case 98u  /* OpImageRead */:
        case 305u /* OpImageSparseSampleImplicitLod */:
        case 306u /* OpImageSparseSampleExplicitLod */:
        case 313u /* OpImageSparseFetch */:
        case 316u /* OpImageSparseRead */:
            return (k == 5);
        /* Image instructions with ImageOperands mask at word 6 */
        case 89u  /* OpImageSampleDrefImplicitLod */:
        case 90u  /* OpImageSampleDrefExplicitLod */:
        case 93u  /* OpImageSampleProjDrefImplicitLod */:
        case 94u  /* OpImageSampleProjDrefExplicitLod */:
        case 96u  /* OpImageGather */:
        case 97u  /* OpImageDrefGather */:
        case 307u /* OpImageSparseSampleDrefImplicitLod */:
        case 308u /* OpImageSparseSampleDrefExplicitLod */:
        case 314u /* OpImageSparseGather */:
        case 315u /* OpImageSparseDrefGather */:
            return (k == 6);
        default:
            /* OpGroupNonUniform* reduction ops (349..364) have GroupOperation literal at word 4 */
            if (op >= 349u && op <= 364u) {
                return (k == 4);
            }
            return false;
    }
}

static uint32_t rt_ensure_u32(struct RtMegaBuilder *mb, struct SpvBuf *fn_buf,
                              uint32_t mapped_id, uint32_t orig_id,
                              const struct RtStageParse *sp) {
    if (orig_id < sp->bound) {
        uint32_t otype = sp->id_type[orig_id];
        if (otype < sp->bound && sp->remap[otype] != mb->t_u32) {
            uint32_t cast_id = rt_alloc_id(mb);
            spv_emit_op3(fn_buf, RT_OP_BITCAST, mb->t_u32, cast_id, mapped_id);
            return cast_id;
        }
    }
    return mapped_id;
}


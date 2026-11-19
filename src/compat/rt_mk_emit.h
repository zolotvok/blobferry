// SPDX-License-Identifier: GPL-3.0-or-later
// rt_mk_emit.h: megakernel stage functions, hit/callable, trace loop, main, indirect setup. Included by mali_rt_emu.h.
/* ── Stage Function Body Lowerer (Depth 0 & Depth 1) ─────────────────────── */

static void rt_emit_stage_functions(struct RtMegaBuilder *mb,
                                    const struct RtStageRecord *sr,
                                    struct RtStageParse *sp,
                                    int depth) {
    const uint32_t *code = sr->words;
    size_t count = sr->word_count;
    uint32_t bound = sp->bound;
    uint32_t *cur_map = (depth == 0) ? sp->remap : sp->remap_d1;

    /* Find first OpFunction */
    size_t fn_start = 5;
    while (fn_start < count) {
        uint32_t wc = code[fn_start] >> 16;
        uint32_t op = code[fn_start] & 0xFFFFu;
        if (wc == 0 || fn_start + wc > count || op == RT_OP_FUNCTION) break;
        fn_start += wc;
    }

    /*
     * Pre-scan all instructions in functions to:
     *   1. Record id_type[res_id] and is_psb_id[res_id]
     *   2. If depth == 1, allocate fresh IDs in remap_d1 for function-local definitions
     */
    for (size_t i = fn_start; i < count; ) {
        uint32_t wc = code[i] >> 16;
        uint32_t op = code[i] & 0xFFFFu;
        if (wc == 0 || i + wc > count) break;

        uint32_t def_id = 0;
        uint32_t def_type = 0;
        if (op == RT_OP_LABEL && wc >= 2) {
            def_id = code[i + 1];
        } else if (op != RT_OP_STORE && op != RT_OP_BRANCH &&
                   op != RT_OP_BRANCH_CONDITIONAL && op != RT_OP_SWITCH &&
                   op != RT_OP_RETURN && op != RT_OP_RETURN_VALUE &&
                   op != RT_OP_UNREACHABLE && op != RT_OP_LOOP_MERGE &&
                   op != RT_OP_SELECTION_MERGE && op != RT_OP_FUNCTION_END &&
                   op != RT_OP_TRACE_RAY_KHR && op != RT_OP_EXECUTE_CALLABLE_KHR &&
                   op != RT_OP_EXECUTE_CALLABLE_NV &&
                   op != RT_OP_IGNORE_INTERSECTION_KHR && op != RT_OP_IGNORE_INTERSECTION_NV &&
                   op != RT_OP_TERMINATE_RAY_KHR && op != RT_OP_TERMINATE_RAY_NV &&
                   op != RT_OP_LINE && op != RT_OP_NO_LINE &&
                   op != 63u /* OpCopyMemory */ && op != 64u /* OpCopyMemorySized */ &&
                   op != 99u /* OpImageWrite */ && op != 224u /* OpControlBarrier */ &&
                   op != 225u /* OpMemoryBarrier */ && op != 235u /* OpAtomicStore */ &&
                   op != 4416u /* OpTerminateInvocation */ && wc >= 3) {
            def_type = code[i + 1];
            def_id   = code[i + 2];
        }

        if (def_id > 0 && def_id < bound) {
            if (def_type > 0 && def_type < bound) {
                sp->id_type[def_id] = def_type;
                if (sp->is_psb_type[def_type]) {
                    sp->is_psb_id[def_id] = true;
                }
            }
            if (depth == 1) {
                cur_map[def_id] = rt_alloc_id(mb);
            }
        }
        i += wc;
    }

    if (depth == 0) {
        sp->entry_fn_id_d0 = cur_map[sp->orig_entry_fn_id];
    } else {
        sp->entry_fn_id_d1 = cur_map[sp->orig_entry_fn_id];
    }

    /* Emit functions */
    bool in_fn = false;
    bool is_entry_fn = false;
    bool in_entry_block = false;
    bool emitted_prologue = false;

    for (size_t i = fn_start; i < count; ) {
        uint32_t wc = code[i] >> 16;
        uint32_t op = code[i] & 0xFFFFu;
        if (wc == 0 || i + wc > count) break;

        if (op == RT_OP_LINE || op == RT_OP_NO_LINE) {
            i += wc;
            continue;
        }

        if (op == RT_OP_FUNCTION && wc >= 5) {
            in_fn = true;
            uint32_t orig_fn_id = code[i + 2];
            is_entry_fn = (orig_fn_id == sp->orig_entry_fn_id);
            in_entry_block = false;
            emitted_prologue = false;

            uint32_t rtype = (code[i + 1] < bound) ? cur_map[code[i + 1]] : code[i + 1];
            uint32_t fn_id = (orig_fn_id < bound) ? cur_map[orig_fn_id] : orig_fn_id;
            uint32_t fctrl = code[i + 3];
            uint32_t ftype = (code[i + 4] < bound) ? cur_map[code[i + 4]] : code[i + 4];
            if (is_entry_fn) {
                rtype = mb->t_void;
                ftype = mb->t_fn_void;
            }
            spv_emit_op4(&mb->functions, RT_OP_FUNCTION, rtype, fn_id, fctrl, ftype);
            i += wc;
            continue;
        }

        if (op == RT_OP_FUNCTION_END) {
            in_fn = false;
            in_entry_block = false;
            spv_emit_op0(&mb->functions, RT_OP_FUNCTION_END);
            i += wc;
            continue;
        }

        if (!in_fn) {
            i += wc;
            continue;
        }

        if (op == RT_OP_LABEL && wc >= 2) {
            uint32_t lbl_id = (code[i + 1] < bound) ? cur_map[code[i + 1]] : code[i + 1];
            spv_emit_op1(&mb->functions, RT_OP_LABEL, lbl_id);
            if (!in_entry_block && !emitted_prologue) {
                in_entry_block = true;
            } else {
                in_entry_block = false;
            }
            i += wc;
            continue;
        }

        /* In the entry basic block, keep OpVariable (Function) and OpLine before our prologue */
        if (in_entry_block && !emitted_prologue && op != RT_OP_VARIABLE) {
            emitted_prologue = true;

            /* 1. Materialize PSB pointers for PushConstant (9) and ShaderRecordBufferKHR (5343) */
            for (uint32_t v = 0; v < sp->psb_var_count; v++) {
                uint32_t orig_v = sp->psb_vars_orig[v];
                uint32_t psb_pt = sp->psb_vars_ptr_type[v];
                uint32_t sc     = sp->psb_vars_sc[v];
                uint32_t bda_glob = (sc == RT_SC_PUSH_CONSTANT) ? mb->g_push_consts_bda : mb->g_cur_record_bda;
                uint32_t bda_val  = rt_alloc_id(mb);
                uint32_t ptr_val  = rt_alloc_id(mb);
                cur_map[orig_v]   = ptr_val;
                spv_emit_op3(&mb->functions, RT_OP_LOAD, mb->t_u64, bda_val, bda_glob);
                spv_emit_op3(&mb->functions, RT_OP_CONVERT_U_TO_PTR, psb_pt, ptr_val, bda_val);
            }

            /* 2. If this is the stage's entry point function, unpack incoming state & sync builtins */
            if (is_entry_fn) {
                rt_emit_stage_builtin_sync(mb, &mb->functions, sp);
                uint32_t idx_chain[12];
                if (sp->incoming_payload_var != 0) {
                    uint32_t slot = 0;
                    rt_emit_pack_unpack_type(mb, &mb->functions,
                                             sp->incoming_payload_var, sp->incoming_payload_type,
                                             idx_chain, 0, &slot, RT_PAYLOAD_MAX_DWORDS,
                                             mb->g_payload_u32, false);
                }
                if (sp->hit_attr_var != 0 &&
                    (sr->stage == VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR ||
                     sr->stage == VK_SHADER_STAGE_ANY_HIT_BIT_KHR)) {
                    uint32_t slot = 0;
                    rt_emit_pack_unpack_type(mb, &mb->functions,
                                             sp->hit_attr_var, sp->hit_attr_type,
                                             idx_chain, 0, &slot, RT_HITATTR_MAX_DWORDS,
                                             mb->g_hitattr_u32, false);
                }
                if (sp->incoming_callable_var != 0) {
                    uint32_t slot = 0;
                    rt_emit_pack_unpack_type(mb, &mb->functions,
                                             sp->incoming_callable_var, sp->incoming_callable_type,
                                             idx_chain, 0, &slot, RT_CALLABLE_MAX_DWORDS,
                                             mb->g_callable_u32, false);
                }
            }
        }

        /* Lower RT intrinsics */
        if (op == RT_OP_TRACE_RAY_KHR && wc >= 12) {
            if (depth == 0) {
                uint32_t tlas       = (code[i + 1]  < bound) ? cur_map[code[i + 1]]  : code[i + 1];
                uint32_t flags      = (code[i + 2]  < bound) ? cur_map[code[i + 2]]  : code[i + 2];
                uint32_t cull       = (code[i + 3]  < bound) ? cur_map[code[i + 3]]  : code[i + 3];
                uint32_t sbt_off    = (code[i + 4]  < bound) ? cur_map[code[i + 4]]  : code[i + 4];
                uint32_t sbt_stride = (code[i + 5]  < bound) ? cur_map[code[i + 5]]  : code[i + 5];
                uint32_t miss_idx   = (code[i + 6]  < bound) ? cur_map[code[i + 6]]  : code[i + 6];
                uint32_t orig       = (code[i + 7]  < bound) ? cur_map[code[i + 7]]  : code[i + 7];
                uint32_t tmin       = (code[i + 8]  < bound) ? cur_map[code[i + 8]]  : code[i + 8];
                uint32_t dir        = (code[i + 9]  < bound) ? cur_map[code[i + 9]]  : code[i + 9];
                uint32_t tmax       = (code[i + 10] < bound) ? cur_map[code[i + 10]] : code[i + 10];
                uint32_t pay_orig   = code[i + 11];
                uint32_t pay_var    = (pay_orig < bound) ? cur_map[pay_orig] : pay_orig;
                uint32_t pay_pt_orig= (pay_orig < bound) ? sp->var_pointee_orig[pay_orig] : 0;
                uint32_t pay_type   = (pay_pt_orig < bound) ? sp->remap[pay_pt_orig] : pay_pt_orig;

                /* 1. Pack outgoing payload into %g_payload_u32 */
                uint32_t idx_chain[12];
                if (pay_var && pay_type) {
                    uint32_t slot = 0;
                    rt_emit_pack_unpack_type(mb, &mb->functions, pay_var, pay_type,
                                             idx_chain, 0, &slot, RT_PAYLOAD_MAX_DWORDS,
                                             mb->g_payload_u32, true);
                }

                /* 2. Save caller's complete traversal and builtin state if recursive (!is_rgen) */
                uint32_t saved_rec_bda = rt_alloc_id(mb);
                spv_emit_op3(&mb->functions, RT_OP_LOAD, mb->t_u64, saved_rec_bda, mb->g_cur_record_bda);

                uint32_t s_orig = 0, s_dir = 0, s_tmin = 0, s_tmax = 0;
                uint32_t s_flags = 0, s_cull = 0, s_sbtoff = 0, s_sbtstr = 0, s_miss = 0;
                uint32_t s_cust = 0, s_inst = 0, s_prim = 0, s_geom = 0, s_kind = 0;
                uint32_t s_obj_orig = 0, s_obj_dir = 0, s_otw = 0, s_wto = 0;
                uint32_t s_hitattr = 0, s_in_pay = 0, s_in_attr = 0;

                bool is_rgen = (sr->stage == VK_SHADER_STAGE_RAYGEN_BIT_KHR);
                if (!is_rgen) {
                    s_orig    = rt_alloc_id(mb);
                    s_dir     = rt_alloc_id(mb);
                    s_tmin    = rt_alloc_id(mb);
                    s_tmax    = rt_alloc_id(mb);
                    s_flags   = rt_alloc_id(mb);
                    s_cull    = rt_alloc_id(mb);
                    s_sbtoff  = rt_alloc_id(mb);
                    s_sbtstr  = rt_alloc_id(mb);
                    s_miss    = rt_alloc_id(mb);
                    s_cust    = rt_alloc_id(mb);
                    s_inst    = rt_alloc_id(mb);
                    s_prim    = rt_alloc_id(mb);
                    s_geom    = rt_alloc_id(mb);
                    s_kind    = rt_alloc_id(mb);
                    s_hitattr = rt_alloc_id(mb);

                    spv_emit_op3(&mb->functions, RT_OP_LOAD, mb->t_v3f32, s_orig,   mb->g_world_ray_orig);
                    spv_emit_op3(&mb->functions, RT_OP_LOAD, mb->t_v3f32, s_dir,    mb->g_world_ray_dir);
                    spv_emit_op3(&mb->functions, RT_OP_LOAD, mb->t_f32,   s_tmin,   mb->g_ray_tmin);
                    spv_emit_op3(&mb->functions, RT_OP_LOAD, mb->t_f32,   s_tmax,   mb->g_ray_tmax);
                    spv_emit_op3(&mb->functions, RT_OP_LOAD, mb->t_u32,   s_flags,  mb->g_incoming_ray_flags);
                    spv_emit_op3(&mb->functions, RT_OP_LOAD, mb->t_u32,   s_cull,   mb->g_cull_mask);
                    spv_emit_op3(&mb->functions, RT_OP_LOAD, mb->t_u32,   s_sbtoff, mb->g_trace_sbt_offset);
                    spv_emit_op3(&mb->functions, RT_OP_LOAD, mb->t_u32,   s_sbtstr, mb->g_trace_sbt_stride);
                    spv_emit_op3(&mb->functions, RT_OP_LOAD, mb->t_u32,   s_miss,   mb->g_trace_miss_index);
                    spv_emit_op3(&mb->functions, RT_OP_LOAD, mb->t_u32,   s_cust,   mb->g_instance_custom_idx);
                    spv_emit_op3(&mb->functions, RT_OP_LOAD, mb->t_u32,   s_inst,   mb->g_instance_id);
                    spv_emit_op3(&mb->functions, RT_OP_LOAD, mb->t_u32,   s_prim,   mb->g_primitive_id);
                    spv_emit_op3(&mb->functions, RT_OP_LOAD, mb->t_u32,   s_geom,   mb->g_geom_index);
                    spv_emit_op3(&mb->functions, RT_OP_LOAD, mb->t_u32,   s_kind,   mb->g_hit_kind);
                    spv_emit_op3(&mb->functions, RT_OP_LOAD, mb->t_arr_hitattr, s_hitattr, mb->g_hitattr_u32);

                    if (mb->uses_object_ray) {
                        s_obj_orig = rt_alloc_id(mb);
                        s_obj_dir  = rt_alloc_id(mb);
                        spv_emit_op3(&mb->functions, RT_OP_LOAD, mb->t_v3f32, s_obj_orig, mb->g_obj_ray_orig);
                        spv_emit_op3(&mb->functions, RT_OP_LOAD, mb->t_v3f32, s_obj_dir,  mb->g_obj_ray_dir);
                    }
                    if (mb->uses_obj_to_world) {
                        s_otw = rt_alloc_id(mb);
                        spv_emit_op3(&mb->functions, RT_OP_LOAD, mb->t_mat4x3f32, s_otw, mb->g_obj_to_world);
                    }
                    if (mb->uses_world_to_obj) {
                        s_wto = rt_alloc_id(mb);
                        spv_emit_op3(&mb->functions, RT_OP_LOAD, mb->t_mat4x3f32, s_wto, mb->g_world_to_obj);
                    }
                    if (sp->incoming_payload_var != 0) {
                        s_in_pay = rt_alloc_id(mb);
                        spv_emit_op3(&mb->functions, RT_OP_LOAD, sp->incoming_payload_type, s_in_pay, sp->incoming_payload_var);
                    }
                    if (sp->hit_attr_var != 0) {
                        s_in_attr = rt_alloc_id(mb);
                        spv_emit_op3(&mb->functions, RT_OP_LOAD, sp->hit_attr_type, s_in_attr, sp->hit_attr_var);
                    }
                }

                /* 3. Ensure integer parameters are uint32 */
                uint32_t flags_u   = rt_ensure_u32(mb, &mb->functions, flags,      code[i + 2], sp);
                uint32_t cull_u    = rt_ensure_u32(mb, &mb->functions, cull,       code[i + 3], sp);
                uint32_t sbtoff_u  = rt_ensure_u32(mb, &mb->functions, sbt_off,    code[i + 4], sp);
                uint32_t sbtstr_u  = rt_ensure_u32(mb, &mb->functions, sbt_stride, code[i + 5], sp);
                uint32_t miss_u    = rt_ensure_u32(mb, &mb->functions, miss_idx,   code[i + 6], sp);

                uint32_t rq_var  = is_rgen ? mb->g_rq_d0 : mb->g_rq_d1;
                uint32_t fn_loop = is_rgen ? mb->fn_trace_loop_d0 : mb->fn_trace_loop_d1;

                /* 4. OpRayQueryInitializeKHR directly in the caller using %tlas */
                {
                    uint32_t w[9] = {
                        (9u << 16) | RT_OP_RAY_QUERY_INITIALIZE_KHR,
                        rq_var, tlas, flags_u, cull_u, orig, tmin, dir, tmax
                    };
                    spv_emit_words(&mb->functions, w, 9);
                }

                /* 5. Store ray parameters to globals and invoke traversal loop */
                spv_emit_op2(&mb->functions, RT_OP_STORE, mb->g_incoming_ray_flags, flags_u);
                spv_emit_op2(&mb->functions, RT_OP_STORE, mb->g_cull_mask,          cull_u);
                spv_emit_op2(&mb->functions, RT_OP_STORE, mb->g_trace_sbt_offset,   sbtoff_u);
                spv_emit_op2(&mb->functions, RT_OP_STORE, mb->g_trace_sbt_stride,   sbtstr_u);
                spv_emit_op2(&mb->functions, RT_OP_STORE, mb->g_trace_miss_index,   miss_u);
                spv_emit_op2(&mb->functions, RT_OP_STORE, mb->g_world_ray_orig,     orig);
                spv_emit_op2(&mb->functions, RT_OP_STORE, mb->g_ray_tmin,           tmin);
                spv_emit_op2(&mb->functions, RT_OP_STORE, mb->g_world_ray_dir,      dir);
                spv_emit_op2(&mb->functions, RT_OP_STORE, mb->g_ray_tmax,           tmax);

                uint32_t call_id = rt_alloc_id(mb);
                spv_emit_op3(&mb->functions, RT_OP_FUNCTION_CALL, mb->t_void, call_id, fn_loop);

                /* 6. Restore caller's complete traversal and builtin state */
                spv_emit_op2(&mb->functions, RT_OP_STORE, mb->g_cur_record_bda, saved_rec_bda);
                if (!is_rgen) {
                    spv_emit_op2(&mb->functions, RT_OP_STORE, mb->g_world_ray_orig,     s_orig);
                    spv_emit_op2(&mb->functions, RT_OP_STORE, mb->g_world_ray_dir,      s_dir);
                    spv_emit_op2(&mb->functions, RT_OP_STORE, mb->g_ray_tmin,           s_tmin);
                    spv_emit_op2(&mb->functions, RT_OP_STORE, mb->g_ray_tmax,           s_tmax);
                    spv_emit_op2(&mb->functions, RT_OP_STORE, mb->g_incoming_ray_flags, s_flags);
                    spv_emit_op2(&mb->functions, RT_OP_STORE, mb->g_cull_mask,          s_cull);
                    spv_emit_op2(&mb->functions, RT_OP_STORE, mb->g_trace_sbt_offset,   s_sbtoff);
                    spv_emit_op2(&mb->functions, RT_OP_STORE, mb->g_trace_sbt_stride,   s_sbtstr);
                    spv_emit_op2(&mb->functions, RT_OP_STORE, mb->g_trace_miss_index,   s_miss);
                    spv_emit_op2(&mb->functions, RT_OP_STORE, mb->g_instance_custom_idx, s_cust);
                    spv_emit_op2(&mb->functions, RT_OP_STORE, mb->g_instance_id,         s_inst);
                    spv_emit_op2(&mb->functions, RT_OP_STORE, mb->g_primitive_id,        s_prim);
                    spv_emit_op2(&mb->functions, RT_OP_STORE, mb->g_geom_index,          s_geom);
                    spv_emit_op2(&mb->functions, RT_OP_STORE, mb->g_hit_kind,            s_kind);
                    spv_emit_op2(&mb->functions, RT_OP_STORE, mb->g_hitattr_u32,        s_hitattr);

                    if (mb->uses_object_ray) {
                        spv_emit_op2(&mb->functions, RT_OP_STORE, mb->g_obj_ray_orig, s_obj_orig);
                        spv_emit_op2(&mb->functions, RT_OP_STORE, mb->g_obj_ray_dir,  s_obj_dir);
                    }
                    if (mb->uses_obj_to_world) {
                        spv_emit_op2(&mb->functions, RT_OP_STORE, mb->g_obj_to_world, s_otw);
                    }
                    if (mb->uses_world_to_obj) {
                        spv_emit_op2(&mb->functions, RT_OP_STORE, mb->g_world_to_obj, s_wto);
                    }
                    if (sp->incoming_payload_var != 0) {
                        spv_emit_op2(&mb->functions, RT_OP_STORE, sp->incoming_payload_var, s_in_pay);
                    }
                    if (sp->hit_attr_var != 0) {
                        spv_emit_op2(&mb->functions, RT_OP_STORE, sp->hit_attr_var, s_in_attr);
                    }
                    rt_emit_stage_builtin_sync(mb, &mb->functions, sp);
                }

                if (pay_var && pay_type) {
                    uint32_t slot = 0;
                    rt_emit_pack_unpack_type(mb, &mb->functions, pay_var, pay_type,
                                             idx_chain, 0, &slot, RT_PAYLOAD_MAX_DWORDS,
                                             mb->g_payload_u32, false);
                }
            }
            i += wc;
            continue;
        }

        if ((op == RT_OP_EXECUTE_CALLABLE_KHR || op == RT_OP_EXECUTE_CALLABLE_NV) && wc >= 3) {
            uint32_t sbt_idx  = (code[i + 1] < bound) ? cur_map[code[i + 1]] : code[i + 1];
            uint32_t call_orig= code[i + 2];
            uint32_t call_var = (call_orig < bound) ? cur_map[call_orig] : call_orig;
            uint32_t call_pt_orig = (call_orig < bound) ? sp->var_pointee_orig[call_orig] : 0;
            uint32_t call_type    = (call_pt_orig < bound) ? sp->remap[call_pt_orig] : call_pt_orig;

            uint32_t idx_chain[12];
            if (call_var && call_type) {
                uint32_t slot = 0;
                rt_emit_pack_unpack_type(mb, &mb->functions, call_var, call_type,
                                         idx_chain, 0, &slot, RT_CALLABLE_MAX_DWORDS,
                                         mb->g_callable_u32, true);
            }
            uint32_t saved_rec_bda = rt_alloc_id(mb);
            spv_emit_op3(&mb->functions, RT_OP_LOAD, mb->t_u64, saved_rec_bda, mb->g_cur_record_bda);
            uint32_t idx_u = rt_ensure_u32(mb, &mb->functions, sbt_idx, code[i + 1], sp);
            uint32_t call_id = rt_alloc_id(mb);
            spv_emit_op4(&mb->functions, RT_OP_FUNCTION_CALL, mb->t_void, call_id, mb->fn_invoke_rcall, idx_u);
            spv_emit_op2(&mb->functions, RT_OP_STORE, mb->g_cur_record_bda, saved_rec_bda);
            if (call_var && call_type) {
                uint32_t slot = 0;
                rt_emit_pack_unpack_type(mb, &mb->functions, call_var, call_type,
                                         idx_chain, 0, &slot, RT_CALLABLE_MAX_DWORDS,
                                         mb->g_callable_u32, false);
            }
            i += wc;
            continue;
        }

        if (op == RT_OP_REPORT_INTERSECTION_KHR && wc >= 5) {
            uint32_t res_id   = (code[i + 2] < bound) ? cur_map[code[i + 2]] : code[i + 2];
            uint32_t hit_t    = (code[i + 3] < bound) ? cur_map[code[i + 3]] : code[i + 3];
            uint32_t hit_kind = (code[i + 4] < bound) ? cur_map[code[i + 4]] : code[i + 4];
            uint32_t kind_u   = rt_ensure_u32(mb, &mb->functions, hit_kind, code[i + 4], sp);

            if (sp->hit_attr_var != 0) {
                uint32_t idx_chain[12];
                uint32_t slot = 0;
                rt_emit_pack_unpack_type(mb, &mb->functions, sp->hit_attr_var, sp->hit_attr_type,
                                         idx_chain, 0, &slot, RT_HITATTR_MAX_DWORDS,
                                         mb->g_staging_hitattr_u32, true);
            }
            uint32_t fn_rep = (depth == 0) ? mb->fn_report_hit_d0 : mb->fn_report_hit_d1;
            spv_emit_op5(&mb->functions, RT_OP_FUNCTION_CALL, mb->t_bool, res_id, fn_rep, hit_t, kind_u);
            i += wc;
            continue;
        }

        if (op == RT_OP_IGNORE_INTERSECTION_KHR || op == RT_OP_IGNORE_INTERSECTION_NV) {
            if (sp->incoming_payload_var != 0) {
                uint32_t idx_chain[12];
                uint32_t slot = 0;
                rt_emit_pack_unpack_type(mb, &mb->functions,
                                         sp->incoming_payload_var, sp->incoming_payload_type,
                                         idx_chain, 0, &slot, RT_PAYLOAD_MAX_DWORDS,
                                         mb->g_payload_u32, true);
            }
            spv_emit_op2(&mb->functions, RT_OP_STORE, mb->g_rahit_ignored, mb->c_u32_1);
            spv_emit_op0(&mb->functions, RT_OP_RETURN);
            i += wc;
            continue;
        }

        if (op == RT_OP_TERMINATE_RAY_KHR || op == RT_OP_TERMINATE_RAY_NV) {
            if (sp->incoming_payload_var != 0) {
                uint32_t idx_chain[12];
                uint32_t slot = 0;
                rt_emit_pack_unpack_type(mb, &mb->functions,
                                         sp->incoming_payload_var, sp->incoming_payload_type,
                                         idx_chain, 0, &slot, RT_PAYLOAD_MAX_DWORDS,
                                         mb->g_payload_u32, true);
            }
            spv_emit_op2(&mb->functions, RT_OP_STORE, mb->g_ray_terminated, mb->c_u32_1);
            spv_emit_op0(&mb->functions, RT_OP_RETURN);
            i += wc;
            continue;
        }

        if (op == RT_OP_RETURN && is_entry_fn) {
            uint32_t idx_chain[12];
            if (sp->incoming_payload_var != 0) {
                uint32_t slot = 0;
                rt_emit_pack_unpack_type(mb, &mb->functions,
                                         sp->incoming_payload_var, sp->incoming_payload_type,
                                         idx_chain, 0, &slot, RT_PAYLOAD_MAX_DWORDS,
                                         mb->g_payload_u32, true);
            }
            if (sp->incoming_callable_var != 0) {
                uint32_t slot = 0;
                rt_emit_pack_unpack_type(mb, &mb->functions,
                                         sp->incoming_callable_var, sp->incoming_callable_type,
                                         idx_chain, 0, &slot, RT_CALLABLE_MAX_DWORDS,
                                         mb->g_callable_u32, true);
            }
            spv_emit_op0(&mb->functions, RT_OP_RETURN);
            i += wc;
            continue;
        }

        /* Ensure Aligned 4 on OpLoad through PhysicalStorageBuffer pointers */
        if (op == RT_OP_LOAD && wc >= 4) {
            uint32_t ptr_orig = code[i + 3];
            bool is_psb = (ptr_orig < bound && sp->is_psb_id[ptr_orig]);
            if (is_psb) {
                uint32_t rtype = (code[i + 1] < bound) ? cur_map[code[i + 1]] : code[i + 1];
                uint32_t resid = (code[i + 2] < bound) ? cur_map[code[i + 2]] : code[i + 2];
                uint32_t ptrid = (ptr_orig    < bound) ? cur_map[ptr_orig]    : ptr_orig;
                if (wc == 4) {
                    spv_emit_op5(&mb->functions, RT_OP_LOAD, rtype, resid, ptrid, 0x2u /* Aligned */, 4u);
                    i += wc;
                    continue;
                } else if ((code[i + 4] & 0x2u) == 0) {
                    spv_emit_word(&mb->functions, ((wc + 1) << 16) | RT_OP_LOAD);
                    spv_emit_word(&mb->functions, rtype);
                    spv_emit_word(&mb->functions, resid);
                    spv_emit_word(&mb->functions, ptrid);
                    spv_emit_word(&mb->functions, code[i + 4] | 0x2u);
                    spv_emit_word(&mb->functions, 4u);
                    for (uint32_t k = 5; k < wc; k++) {
                        uint32_t opnd = code[i + k];
                        spv_emit_word(&mb->functions, (opnd < bound) ? cur_map[opnd] : opnd);
                    }
                    i += wc;
                    continue;
                }
            }
        }

        /* Ensure Aligned 4 on OpStore through PhysicalStorageBuffer pointers */
        if (op == RT_OP_STORE && wc >= 3) {
            uint32_t ptr_orig = code[i + 1];
            bool is_psb = (ptr_orig < bound && sp->is_psb_id[ptr_orig]);
            if (is_psb) {
                uint32_t ptrid = (ptr_orig    < bound) ? cur_map[ptr_orig]    : ptr_orig;
                uint32_t valid = (code[i + 2] < bound) ? cur_map[code[i + 2]] : code[i + 2];
                if (wc == 3) {
                    spv_emit_op4(&mb->functions, RT_OP_STORE, ptrid, valid, 0x2u /* Aligned */, 4u);
                    i += wc;
                    continue;
                } else if ((code[i + 3] & 0x2u) == 0) {
                    spv_emit_word(&mb->functions, ((wc + 1) << 16) | RT_OP_STORE);
                    spv_emit_word(&mb->functions, ptrid);
                    spv_emit_word(&mb->functions, valid);
                    spv_emit_word(&mb->functions, code[i + 3] | 0x2u);
                    spv_emit_word(&mb->functions, 4u);
                    for (uint32_t k = 4; k < wc; k++) {
                        uint32_t opnd = code[i + k];
                        spv_emit_word(&mb->functions, (opnd < bound) ? cur_map[opnd] : opnd);
                    }
                    i += wc;
                    continue;
                }
            }
        }

        /* Default instruction emission with exact ID vs Literal operand handling */
        spv_emit_word(&mb->functions, code[i]);
        for (uint32_t k = 1; k < wc; k++) {
            uint32_t opnd = code[i + k];
            if (rt_is_literal_operand(op, k, wc, &code[i], sp)) {
                spv_emit_word(&mb->functions, opnd);
            } else {
                spv_emit_word(&mb->functions, (opnd < bound) ? cur_map[opnd] : opnd);
            }
        }
        i += wc;
    }
}

/* ── Megakernel Dispatch Helpers & Traversal Loop Emission ────────────────── */

static void rt_emit_psb_load_u32(struct RtMegaBuilder *mb, struct SpvBuf *b,
                                 uint32_t res_id, uint32_t bda_u64_id, uint32_t byte_offset) {
    uint32_t addr_id = bda_u64_id;
    if (byte_offset != 0) {
        uint32_t off_c = rt_get_const_u64(mb, (uint64_t)byte_offset);
        addr_id = rt_alloc_id(mb);
        spv_emit_op4(b, RT_OP_IADD, mb->t_u64, addr_id, bda_u64_id, off_c);
    }
    uint32_t ptr_id = rt_alloc_id(mb);
    spv_emit_op3(b, RT_OP_CONVERT_U_TO_PTR, mb->t_ptr_psb_u32, ptr_id, addr_id);
    spv_emit_op5(b, RT_OP_LOAD, mb->t_u32, res_id, ptr_id, 0x2u /* Aligned */, 4u);
}

static void rt_emit_psb_load_u64(struct RtMegaBuilder *mb, struct SpvBuf *b,
                                 uint32_t res_id, uint32_t bda_u64_id, uint32_t byte_offset) {
    uint32_t addr_id = bda_u64_id;
    if (byte_offset != 0) {
        uint32_t off_c = rt_get_const_u64(mb, (uint64_t)byte_offset);
        addr_id = rt_alloc_id(mb);
        spv_emit_op4(b, RT_OP_IADD, mb->t_u64, addr_id, bda_u64_id, off_c);
    }
    uint32_t ptr_id = rt_alloc_id(mb);
    spv_emit_op3(b, RT_OP_CONVERT_U_TO_PTR, mb->t_ptr_psb_u64, ptr_id, addr_id);
    spv_emit_op5(b, RT_OP_LOAD, mb->t_u64, res_id, ptr_id, 0x2u /* Aligned */, 8u);
}

static void rt_emit_stage_switch_inline(struct RtMegaBuilder *mb, struct SpvBuf *b,
                                        uint32_t selector_id,
                                        const struct RtStageRecord *stages,
                                        const struct RtStageParse *parses,
                                        uint32_t stage_count,
                                        VkShaderStageFlagBits match_stage,
                                        int depth) {
    /* Count unique stage_ids matching match_stage */
    uint32_t match_indices[256];
    uint32_t mcount = 0;
    for (uint32_t i = 0; i < stage_count && mcount < 256; i++) {
        if (stages[i].stage != match_stage) continue;
        bool dup = false;
        for (uint32_t j = 0; j < mcount; j++) {
            if (stages[match_indices[j]].stage_id == stages[i].stage_id) {
                dup = true;
                break;
            }
        }
        if (!dup) match_indices[mcount++] = i;
    }

    if (mcount == 0) return;

    uint32_t lbl_merge = rt_alloc_id(mb);
    uint32_t lbl_cases[256];
    for (uint32_t i = 0; i < mcount; i++) {
        lbl_cases[i] = rt_alloc_id(mb);
    }

    spv_emit_op2(b, RT_OP_SELECTION_MERGE, lbl_merge, 0u);
    spv_emit_word(b, ((3u + 2u * mcount) << 16) | RT_OP_SWITCH);
    spv_emit_word(b, selector_id);
    spv_emit_word(b, lbl_merge);
    for (uint32_t i = 0; i < mcount; i++) {
        spv_emit_word(b, stages[match_indices[i]].stage_id);
        spv_emit_word(b, lbl_cases[i]);
    }

    for (uint32_t i = 0; i < mcount; i++) {
        uint32_t sidx = match_indices[i];
        uint32_t fn_id = (depth == 1 && parses[sidx].entry_fn_id_d1 != 0)
                         ? parses[sidx].entry_fn_id_d1
                         : parses[sidx].entry_fn_id_d0;
        spv_emit_op1(b, RT_OP_LABEL, lbl_cases[i]);
        uint32_t call_id = rt_alloc_id(mb);
        spv_emit_op3(b, RT_OP_FUNCTION_CALL, mb->t_void, call_id, fn_id);
        spv_emit_op1(b, RT_OP_BRANCH, lbl_merge);
    }

    spv_emit_op1(b, RT_OP_LABEL, lbl_merge);
}

static void rt_emit_invoke_stage_fn(struct RtMegaBuilder *mb, uint32_t fn_id,
                                    const struct RtStageRecord *stages,
                                    const struct RtStageParse *parses,
                                    uint32_t stage_count,
                                    VkShaderStageFlagBits match_stage,
                                    int depth) {
    struct SpvBuf *b = &mb->functions;
    uint32_t param_id = rt_alloc_id(mb);
    uint32_t lbl_entry = rt_alloc_id(mb);

    spv_emit_op4(b, RT_OP_FUNCTION, mb->t_void, fn_id, 0u, mb->t_fn_u32_void);
    spv_emit_op2(b, RT_OP_FUNCTION_PARAMETER, mb->t_u32, param_id);
    spv_emit_op1(b, RT_OP_LABEL, lbl_entry);

    rt_emit_stage_switch_inline(mb, b, param_id, stages, parses, stage_count, match_stage, depth);

    spv_emit_op0(b, RT_OP_RETURN);
    spv_emit_op0(b, RT_OP_FUNCTION_END);
}

static void rt_emit_copy_hitattr_array(struct RtMegaBuilder *mb, struct SpvBuf *b,
                                       uint32_t dst_arr, uint32_t src_arr) {
    uint32_t val = rt_alloc_id(mb);
    spv_emit_op3(b, RT_OP_LOAD, mb->t_arr_hitattr, val, src_arr);
    spv_emit_op2(b, RT_OP_STORE, dst_arr, val);
}

static void rt_emit_report_hit_fn(struct RtMegaBuilder *mb, int depth) {
    struct SpvBuf *b = &mb->functions;
    uint32_t fn_id     = (depth == 0) ? mb->fn_report_hit_d0 : mb->fn_report_hit_d1;
    uint32_t fn_rahit  = (depth == 0) ? mb->fn_invoke_rahit_d0 : mb->fn_invoke_rahit_d1;
    uint32_t p_hit_t   = rt_alloc_id(mb);
    uint32_t p_kind    = rt_alloc_id(mb);
    uint32_t lbl_entry = rt_alloc_id(mb);
    uint32_t lbl_in_rng= rt_alloc_id(mb);
    uint32_t lbl_out   = rt_alloc_id(mb);
    uint32_t lbl_ign   = rt_alloc_id(mb);
    uint32_t lbl_acc   = rt_alloc_id(mb);

    spv_emit_op4(b, RT_OP_FUNCTION, mb->t_bool, fn_id, 0u, mb->t_fn_report_hit);
    spv_emit_op2(b, RT_OP_FUNCTION_PARAMETER, mb->t_f32, p_hit_t);
    spv_emit_op2(b, RT_OP_FUNCTION_PARAMETER, mb->t_u32, p_kind);
    spv_emit_op1(b, RT_OP_LABEL, lbl_entry);

    uint32_t tmin = rt_alloc_id(mb);
    uint32_t tmax = rt_alloc_id(mb);
    spv_emit_op3(b, RT_OP_LOAD, mb->t_f32, tmin, mb->g_ray_tmin);
    spv_emit_op3(b, RT_OP_LOAD, mb->t_f32, tmax, mb->g_ray_tmax);

    uint32_t cmp_min = rt_alloc_id(mb);
    uint32_t cmp_max = rt_alloc_id(mb);
    uint32_t in_rng  = rt_alloc_id(mb);
    spv_emit_op4(b, RT_OP_FORD_GREATER_THAN_EQUAL, mb->t_bool, cmp_min, p_hit_t, tmin);
    spv_emit_op4(b, RT_OP_FORD_LESS_THAN_EQUAL,    mb->t_bool, cmp_max, p_hit_t, tmax);
    spv_emit_op4(b, RT_OP_LOGICAL_AND,             mb->t_bool, in_rng,  cmp_min, cmp_max);

    spv_emit_op2(b, RT_OP_SELECTION_MERGE, lbl_out, 0u);
    spv_emit_op3(b, RT_OP_BRANCH_CONDITIONAL, in_rng, lbl_in_rng, lbl_out);

    spv_emit_op1(b, RT_OP_LABEL, lbl_in_rng);
    uint32_t old_kind = rt_alloc_id(mb);
    spv_emit_op3(b, RT_OP_LOAD, mb->t_u32, old_kind, mb->g_hit_kind);
    spv_emit_op2(b, RT_OP_STORE, mb->g_ray_tmax, p_hit_t);
    spv_emit_op2(b, RT_OP_STORE, mb->g_hit_kind, p_kind);
    rt_emit_copy_hitattr_array(mb, b, mb->g_hitattr_u32, mb->g_staging_hitattr_u32);
    spv_emit_op2(b, RT_OP_STORE, mb->g_rahit_ignored, mb->c_u32_0);

    uint32_t rahit_id = rt_alloc_id(mb);
    spv_emit_op3(b, RT_OP_LOAD, mb->t_u32, rahit_id, mb->g_cur_rahit_id);
    uint32_t call_rahit = rt_alloc_id(mb);
    spv_emit_op4(b, RT_OP_FUNCTION_CALL, mb->t_void, call_rahit, fn_rahit, rahit_id);

    uint32_t ign_val = rt_alloc_id(mb);
    uint32_t is_ign  = rt_alloc_id(mb);
    spv_emit_op3(b, RT_OP_LOAD, mb->t_u32, ign_val, mb->g_rahit_ignored);
    spv_emit_op4(b, RT_OP_INOT_EQUAL, mb->t_bool, is_ign, ign_val, mb->c_u32_0);

    uint32_t lbl_merge_ign = rt_alloc_id(mb);
    spv_emit_op2(b, RT_OP_SELECTION_MERGE, lbl_merge_ign, 0u);
    spv_emit_op3(b, RT_OP_BRANCH_CONDITIONAL, is_ign, lbl_ign, lbl_acc);

    spv_emit_op1(b, RT_OP_LABEL, lbl_ign);
    spv_emit_op2(b, RT_OP_STORE, mb->g_ray_tmax, tmax);
    spv_emit_op2(b, RT_OP_STORE, mb->g_hit_kind, old_kind);
    rt_emit_copy_hitattr_array(mb, b, mb->g_hitattr_u32, mb->g_committed_hitattr_u32);
    spv_emit_op1(b, RT_OP_RETURN_VALUE, mb->c_false);

    spv_emit_op1(b, RT_OP_LABEL, lbl_acc);
    rt_emit_copy_hitattr_array(mb, b, mb->g_committed_hitattr_u32, mb->g_hitattr_u32);
    spv_emit_op2(b, RT_OP_STORE, mb->g_rint_generated_hit, mb->c_u32_1);
    spv_emit_op1(b, RT_OP_RETURN_VALUE, mb->c_true);

    spv_emit_op1(b, RT_OP_LABEL, lbl_merge_ign);
    spv_emit_op1(b, RT_OP_BRANCH, lbl_out);

    spv_emit_op1(b, RT_OP_LABEL, lbl_out);
    spv_emit_op1(b, RT_OP_RETURN_VALUE, mb->c_false);
    spv_emit_op0(b, RT_OP_FUNCTION_END);
}

static void rt_emit_invoke_rcall_fn(struct RtMegaBuilder *mb,
                                    const struct RtStageRecord *stages,
                                    const struct RtStageParse *parses,
                                    uint32_t stage_count) {
    struct SpvBuf *b = &mb->functions;
    uint32_t p_idx     = rt_alloc_id(mb);
    uint32_t lbl_entry = rt_alloc_id(mb);
    uint32_t lbl_valid = rt_alloc_id(mb);
    uint32_t lbl_exit  = rt_alloc_id(mb);

    spv_emit_op4(b, RT_OP_FUNCTION, mb->t_void, mb->fn_invoke_rcall, 0u, mb->t_fn_u32_void);
    spv_emit_op2(b, RT_OP_FUNCTION_PARAMETER, mb->t_u32, p_idx);
    spv_emit_op1(b, RT_OP_LABEL, lbl_entry);

    uint32_t c_zero64  = rt_get_const_u64(mb, 0ULL);
    uint32_t c_32_64   = rt_get_const_u64(mb, 32ULL);
    uint32_t call_base = rt_alloc_id(mb);
    uint32_t call_size = rt_alloc_id(mb);
    spv_emit_op3(b, RT_OP_LOAD, mb->t_u64, call_base, mb->g_call_base);
    spv_emit_op3(b, RT_OP_LOAD, mb->t_u64, call_size, mb->g_call_size);
    uint32_t has_call  = rt_alloc_id(mb);
    spv_emit_op4(b, RT_OP_INOT_EQUAL, mb->t_bool, has_call, call_base, c_zero64);

    uint32_t stride32  = rt_alloc_id(mb);
    spv_emit_op3(b, RT_OP_LOAD, mb->t_u32, stride32, mb->g_call_stride);
    uint32_t off32     = rt_alloc_id(mb);
    spv_emit_op4(b, RT_OP_IMUL, mb->t_u32, off32, p_idx, stride32);
    uint32_t off64     = rt_alloc_id(mb);
    spv_emit_op3(b, RT_OP_UCONVERT, mb->t_u64, off64, off32);

    /* Call record size: max(call_stride, 32ULL) */
    uint32_t stride64  = rt_alloc_id(mb);
    spv_emit_op3(b, RT_OP_UCONVERT, mb->t_u64, stride64, stride32);
    uint32_t is_cstride_ge32 = rt_alloc_id(mb);
    spv_emit_op4(b, RT_OP_UGREATER_THAN_EQUAL, mb->t_bool, is_cstride_ge32, stride64, c_32_64);
    uint32_t call_rec_sz = rt_alloc_id(mb);
    spv_emit_op5(b, RT_OP_SELECT, mb->t_u64, call_rec_sz, is_cstride_ge32, stride64, c_32_64);

    uint32_t end_off   = rt_alloc_id(mb);
    spv_emit_op4(b, RT_OP_IADD, mb->t_u64, end_off, off64, call_rec_sz);
    uint32_t in_bounds = rt_alloc_id(mb);
    spv_emit_op4(b, RT_OP_ULESS_THAN_EQUAL, mb->t_bool, in_bounds, end_off, call_size);

    uint32_t is_valid_call_addr = rt_alloc_id(mb);
    spv_emit_op4(b, RT_OP_LOGICAL_AND, mb->t_bool, is_valid_call_addr, has_call, in_bounds);

    uint32_t calc_call_rec = rt_alloc_id(mb);
    spv_emit_op4(b, RT_OP_IADD, mb->t_u64, calc_call_rec, call_base, off64);

    uint32_t c_min_bda = rt_get_const_u64(mb, RT_MIN_VALID_GPU_BDA);
    uint32_t c_max_bda = rt_get_const_u64(mb, RT_MAX_VALID_GPU_BDA);
    uint32_t is_call_ge_min = rt_alloc_id(mb);
    spv_emit_op4(b, RT_OP_UGREATER_THAN_EQUAL, mb->t_bool, is_call_ge_min, calc_call_rec, c_min_bda);
    uint32_t is_call_lt_max = rt_alloc_id(mb);
    spv_emit_op4(b, RT_OP_ULESS_THAN, mb->t_bool, is_call_lt_max, calc_call_rec, c_max_bda);
    uint32_t is_call_range_ok = rt_alloc_id(mb);
    spv_emit_op4(b, RT_OP_LOGICAL_AND, mb->t_bool, is_call_range_ok, is_call_ge_min, is_call_lt_max);

    uint32_t is_safe_call_bda = rt_alloc_id(mb);
    spv_emit_op4(b, RT_OP_LOGICAL_AND, mb->t_bool, is_safe_call_bda, is_valid_call_addr, is_call_range_ok);

    uint32_t safe_call_hdr = rt_alloc_id(mb);
    uint32_t pc_bda = rt_alloc_id(mb);
    spv_emit_op3(b, RT_OP_LOAD, mb->t_u64, pc_bda, mb->g_push_consts_bda);
    spv_emit_op5(b, RT_OP_SELECT, mb->t_u64, safe_call_hdr, is_safe_call_bda, calc_call_rec, pc_bda);

    uint32_t loaded_call_magic = rt_alloc_id(mb);
    rt_emit_psb_load_u32(mb, b, loaded_call_magic, safe_call_hdr, 0u);

    uint32_t c_magic = rt_get_const_u32(mb, RT_GROUP_HANDLE_MAGIC);
    uint32_t is_call_magic_ok = rt_alloc_id(mb);
    spv_emit_op4(b, RT_OP_IEQUAL, mb->t_bool, is_call_magic_ok, loaded_call_magic, c_magic);

    uint32_t run_call = rt_alloc_id(mb);
    spv_emit_op4(b, RT_OP_LOGICAL_AND, mb->t_bool, run_call, is_safe_call_bda, is_call_magic_ok);

    spv_emit_op2(b, RT_OP_SELECTION_MERGE, lbl_exit, 0u);
    spv_emit_op3(b, RT_OP_BRANCH_CONDITIONAL, run_call, lbl_valid, lbl_exit);

    spv_emit_op1(b, RT_OP_LABEL, lbl_valid);
    uint32_t srb_bda  = rt_alloc_id(mb);
    spv_emit_op4(b, RT_OP_IADD, mb->t_u64, srb_bda, calc_call_rec, c_32_64);
    spv_emit_op2(b, RT_OP_STORE, mb->g_cur_record_bda, srb_bda);

    uint32_t rcall_id = rt_alloc_id(mb);
    rt_emit_psb_load_u32(mb, b, rcall_id, calc_call_rec, 8u);

    rt_emit_stage_switch_inline(mb, b, rcall_id, stages, parses, stage_count,
                                VK_SHADER_STAGE_CALLABLE_BIT_KHR, 0);

    spv_emit_op1(b, RT_OP_BRANCH, lbl_exit);

    spv_emit_op1(b, RT_OP_LABEL, lbl_exit);
    spv_emit_op0(b, RT_OP_RETURN);
    spv_emit_op0(b, RT_OP_FUNCTION_END);
}

static void rt_emit_rq_populate_hit_builtins(struct RtMegaBuilder *mb, struct SpvBuf *b,
                                             uint32_t rq_var, uint32_t inter_type_const,
                                             uint32_t *out_geom_idx, uint32_t *out_inst_sbt_off) {
    uint32_t t_hit    = rt_alloc_id(mb);
    uint32_t cust_idx = rt_alloc_id(mb);
    uint32_t inst_id  = rt_alloc_id(mb);
    uint32_t sbt_off  = rt_alloc_id(mb);
    uint32_t geom_idx = rt_alloc_id(mb);
    uint32_t prim_idx = rt_alloc_id(mb);

    spv_emit_op4(b, RT_OP_RQ_GET_INTERSECTION_T_KHR,            mb->t_f32, t_hit,    rq_var, inter_type_const);
    spv_emit_op4(b, RT_OP_RQ_GET_INTERSECTION_CUSTOM_INDEX_KHR, mb->t_u32, cust_idx, rq_var, inter_type_const);
    spv_emit_op4(b, RT_OP_RQ_GET_INTERSECTION_INSTANCE_ID_KHR,  mb->t_u32, inst_id,  rq_var, inter_type_const);
    spv_emit_op4(b, RT_OP_RQ_GET_INTERSECTION_SBT_OFFSET_KHR,   mb->t_u32, sbt_off,  rq_var, inter_type_const);
    spv_emit_op4(b, RT_OP_RQ_GET_INTERSECTION_GEOM_INDEX_KHR,   mb->t_u32, geom_idx, rq_var, inter_type_const);
    spv_emit_op4(b, RT_OP_RQ_GET_INTERSECTION_PRIM_INDEX_KHR,   mb->t_u32, prim_idx, rq_var, inter_type_const);

    spv_emit_op2(b, RT_OP_STORE, mb->g_ray_tmax,            t_hit);
    spv_emit_op2(b, RT_OP_STORE, mb->g_instance_custom_idx, cust_idx);
    spv_emit_op2(b, RT_OP_STORE, mb->g_instance_id,         inst_id);
    spv_emit_op2(b, RT_OP_STORE, mb->g_geom_index,          geom_idx);
    spv_emit_op2(b, RT_OP_STORE, mb->g_primitive_id,        prim_idx);

    if (mb->uses_object_ray) {
        uint32_t obj_orig = rt_alloc_id(mb);
        uint32_t obj_dir  = rt_alloc_id(mb);
        spv_emit_op4(b, RT_OP_RQ_GET_INTERSECTION_OBJ_RAY_ORIG_KHR, mb->t_v3f32, obj_orig, rq_var, inter_type_const);
        spv_emit_op4(b, RT_OP_RQ_GET_INTERSECTION_OBJ_RAY_DIR_KHR,  mb->t_v3f32, obj_dir,  rq_var, inter_type_const);
        spv_emit_op2(b, RT_OP_STORE, mb->g_obj_ray_orig, obj_orig);
        spv_emit_op2(b, RT_OP_STORE, mb->g_obj_ray_dir,  obj_dir);
    }
    if (mb->uses_obj_to_world) {
        uint32_t otw = rt_alloc_id(mb);
        spv_emit_op4(b, RT_OP_RQ_GET_INTERSECTION_OBJ_TO_WORLD_KHR, mb->t_mat4x3f32, otw, rq_var, inter_type_const);
        spv_emit_op2(b, RT_OP_STORE, mb->g_obj_to_world, otw);
    }
    if (mb->uses_world_to_obj) {
        uint32_t wto = rt_alloc_id(mb);
        spv_emit_op4(b, RT_OP_RQ_GET_INTERSECTION_WORLD_TO_OBJ_KHR, mb->t_mat4x3f32, wto, rq_var, inter_type_const);
        spv_emit_op2(b, RT_OP_STORE, mb->g_world_to_obj, wto);
    }

    if (out_geom_idx)     *out_geom_idx     = geom_idx;
    if (out_inst_sbt_off) *out_inst_sbt_off = sbt_off;
}

static void rt_emit_rq_store_tri_bary_and_kind(struct RtMegaBuilder *mb, struct SpvBuf *b,
                                               uint32_t rq_var, uint32_t inter_type_const) {
    uint32_t front = rt_alloc_id(mb);
    uint32_t kind  = rt_alloc_id(mb);
    uint32_t bary  = rt_alloc_id(mb);
    uint32_t bx    = rt_alloc_id(mb);
    uint32_t by    = rt_alloc_id(mb);
    uint32_t bx_u  = rt_alloc_id(mb);
    uint32_t by_u  = rt_alloc_id(mb);
    uint32_t ptr0  = rt_alloc_id(mb);
    uint32_t ptr1  = rt_alloc_id(mb);

    spv_emit_op4(b, RT_OP_RQ_GET_INTERSECTION_FRONT_FACE_KHR,   mb->t_bool,  front, rq_var, inter_type_const);
    spv_emit_op5(b, RT_OP_SELECT,                               mb->t_u32,   kind,  front, mb->c_u32_254, mb->c_u32_255);
    spv_emit_op2(b, RT_OP_STORE, mb->g_hit_kind, kind);

    spv_emit_op4(b, RT_OP_RQ_GET_INTERSECTION_BARYCENTRICS_KHR, mb->t_v2f32, bary,  rq_var, inter_type_const);
    spv_emit_op4(b, RT_OP_COMPOSITE_EXTRACT,                    mb->t_f32,   bx,    bary, 0u);
    spv_emit_op4(b, RT_OP_COMPOSITE_EXTRACT,                    mb->t_f32,   by,    bary, 1u);
    spv_emit_op3(b, RT_OP_BITCAST,                              mb->t_u32,   bx_u,  bx);
    spv_emit_op3(b, RT_OP_BITCAST,                              mb->t_u32,   by_u,  by);

    spv_emit_op4(b, RT_OP_ACCESS_CHAIN, mb->t_ptr_priv_u32, ptr0, mb->g_hitattr_u32, mb->c_u32_0);
    spv_emit_op4(b, RT_OP_ACCESS_CHAIN, mb->t_ptr_priv_u32, ptr1, mb->g_hitattr_u32, mb->c_u32_1);
    spv_emit_op2(b, RT_OP_STORE, ptr0, bx_u);
    spv_emit_op2(b, RT_OP_STORE, ptr1, by_u);
}

static uint32_t rt_emit_compute_hit_record_bda(struct RtMegaBuilder *mb, struct SpvBuf *b,
                                               uint32_t geom_idx, uint32_t inst_sbt_off) {
    uint32_t sbt_off    = rt_alloc_id(mb);
    uint32_t sbt_stride = rt_alloc_id(mb);
    uint32_t hit_base   = rt_alloc_id(mb);
    uint32_t hit_stride = rt_alloc_id(mb);
    uint32_t hit_size   = rt_alloc_id(mb);
    uint32_t pc_bda     = rt_alloc_id(mb);

    spv_emit_op3(b, RT_OP_LOAD, mb->t_u32, sbt_off,    mb->g_trace_sbt_offset);
    spv_emit_op3(b, RT_OP_LOAD, mb->t_u32, sbt_stride, mb->g_trace_sbt_stride);
    spv_emit_op3(b, RT_OP_LOAD, mb->t_u64, hit_base,   mb->g_hit_base);
    spv_emit_op3(b, RT_OP_LOAD, mb->t_u32, hit_stride, mb->g_hit_stride);
    spv_emit_op3(b, RT_OP_LOAD, mb->t_u64, hit_size,   mb->g_hit_size);
    spv_emit_op3(b, RT_OP_LOAD, mb->t_u64, pc_bda,     mb->g_push_consts_bda);

    /* sbt_hit_idx = sbt_off + sbt_stride * geom_idx + inst_sbt_off */
    uint32_t mul_geom = rt_alloc_id(mb);
    uint32_t add_off  = rt_alloc_id(mb);
    uint32_t hit_idx  = rt_alloc_id(mb);
    spv_emit_op4(b, RT_OP_IMUL, mb->t_u32, mul_geom, sbt_stride, geom_idx);
    spv_emit_op4(b, RT_OP_IADD, mb->t_u32, add_off,  sbt_off, mul_geom);
    spv_emit_op4(b, RT_OP_IADD, mb->t_u32, hit_idx,  add_off, inst_sbt_off);

    uint32_t byte_off32 = rt_alloc_id(mb);
    uint32_t byte_off64 = rt_alloc_id(mb);
    spv_emit_op4(b, RT_OP_IMUL,     mb->t_u32, byte_off32, hit_idx, hit_stride);
    spv_emit_op3(b, RT_OP_UCONVERT, mb->t_u64, byte_off64, byte_off32);

    /* Record size: max(hit_stride, 32ULL) */
    uint32_t c_zero64     = rt_get_const_u64(mb, 0ULL);
    uint32_t c_32_64      = rt_get_const_u64(mb, 32ULL);
    uint32_t hit_stride64 = rt_alloc_id(mb);
    spv_emit_op3(b, RT_OP_UCONVERT, mb->t_u64, hit_stride64, hit_stride);

    uint32_t is_stride_ge32 = rt_alloc_id(mb);
    spv_emit_op4(b, RT_OP_UGREATER_THAN_EQUAL, mb->t_bool, is_stride_ge32, hit_stride64, c_32_64);
    uint32_t rec_sz = rt_alloc_id(mb);
    spv_emit_op5(b, RT_OP_SELECT, mb->t_u64, rec_sz, is_stride_ge32, hit_stride64, c_32_64);

    /* Bounds check:
     * 1. hit_base must be non-zero (has_base).
     * 2. hit_size must be >= rec_sz (has_hit_size).
     * 3. end_off (byte_off64 + rec_sz) must be <= hit_size (within_size). */
    uint32_t has_base = rt_alloc_id(mb);
    spv_emit_op4(b, RT_OP_INOT_EQUAL, mb->t_bool, has_base, hit_base, c_zero64);

    uint32_t has_hit_size = rt_alloc_id(mb);
    spv_emit_op4(b, RT_OP_UGREATER_THAN_EQUAL, mb->t_bool, has_hit_size, hit_size, rec_sz);

    uint32_t end_off = rt_alloc_id(mb);
    spv_emit_op4(b, RT_OP_IADD, mb->t_u64, end_off, byte_off64, rec_sz);

    uint32_t within_size = rt_alloc_id(mb);
    spv_emit_op4(b, RT_OP_ULESS_THAN_EQUAL, mb->t_bool, within_size, end_off, hit_size);

    uint32_t in_bounds = rt_alloc_id(mb);
    spv_emit_op4(b, RT_OP_LOGICAL_AND, mb->t_bool, in_bounds, has_hit_size, within_size);

    uint32_t is_valid_addr = rt_alloc_id(mb);
    spv_emit_op4(b, RT_OP_LOGICAL_AND, mb->t_bool, is_valid_addr, has_base, in_bounds);

    uint32_t calc_base = rt_alloc_id(mb);
    spv_emit_op4(b, RT_OP_IADD, mb->t_u64, calc_base, hit_base, byte_off64);

    uint32_t c_min_bda = rt_get_const_u64(mb, RT_MIN_VALID_GPU_BDA);
    uint32_t c_max_bda = rt_get_const_u64(mb, RT_MAX_VALID_GPU_BDA);
    uint32_t is_ge_min = rt_alloc_id(mb);
    spv_emit_op4(b, RT_OP_UGREATER_THAN_EQUAL, mb->t_bool, is_ge_min, calc_base, c_min_bda);
    uint32_t is_lt_max = rt_alloc_id(mb);
    spv_emit_op4(b, RT_OP_ULESS_THAN, mb->t_bool, is_lt_max, calc_base, c_max_bda);
    uint32_t is_range_ok = rt_alloc_id(mb);
    spv_emit_op4(b, RT_OP_LOGICAL_AND, mb->t_bool, is_range_ok, is_ge_min, is_lt_max);

    uint32_t is_safe_bda = rt_alloc_id(mb);
    spv_emit_op4(b, RT_OP_LOGICAL_AND, mb->t_bool, is_safe_bda, is_valid_addr, is_range_ok);

    /* Magic check:
     * Safely load offset 0 (magic) by reading from calc_base if is_safe_bda, else pc_bda.
     * Check if magic == RT_GROUP_HANDLE_MAGIC (0x52544748). */
    uint32_t safe_hdr_bda = rt_alloc_id(mb);
    spv_emit_op5(b, RT_OP_SELECT, mb->t_u64, safe_hdr_bda, is_safe_bda, calc_base, pc_bda);

    uint32_t loaded_magic = rt_alloc_id(mb);
    rt_emit_psb_load_u32(mb, b, loaded_magic, safe_hdr_bda, 0u);

    uint32_t c_magic = rt_get_const_u32(mb, RT_GROUP_HANDLE_MAGIC);
    uint32_t is_magic_ok = rt_alloc_id(mb);
    spv_emit_op4(b, RT_OP_IEQUAL, mb->t_bool, is_magic_ok, loaded_magic, c_magic);

    uint32_t is_valid = rt_alloc_id(mb);
    spv_emit_op4(b, RT_OP_LOGICAL_AND, mb->t_bool, is_valid, is_safe_bda, is_magic_ok);

    uint32_t rec_base = rt_alloc_id(mb);
    spv_emit_op5(b, RT_OP_SELECT, mb->t_u64, rec_base, is_valid, calc_base, c_zero64);

    /* If valid: local root arguments are at calc_base + 32ULL.
     * If invalid / missing / bad magic: fallback to pc_bda (valid 256-byte scratch buffer of zeroes).
     * NEVER set g_cur_record_bda to 0ULL! */
    uint32_t calc_srb = rt_alloc_id(mb);
    spv_emit_op4(b, RT_OP_IADD, mb->t_u64, calc_srb, calc_base, c_32_64);
    uint32_t srb_bda = rt_alloc_id(mb);
    spv_emit_op5(b, RT_OP_SELECT, mb->t_u64, srb_bda, is_valid, calc_srb, pc_bda);
    spv_emit_op2(b, RT_OP_STORE, mb->g_cur_record_bda, srb_bda);

    return rec_base;
}

static void rt_emit_trace_loop_fn(struct RtMegaBuilder *mb,
                                  const struct RtStageRecord *stages,
                                  const struct RtStageParse *parses,
                                  uint32_t stage_count,
                                  int depth) {
    struct SpvBuf *b = &mb->functions;
    uint32_t fn_id     = (depth == 0) ? mb->fn_trace_loop_d0   : mb->fn_trace_loop_d1;
    uint32_t rq_var    = (depth == 0) ? mb->g_rq_d0            : mb->g_rq_d1;
    uint32_t fn_rahit  = (depth == 0) ? mb->fn_invoke_rahit_d0 : mb->fn_invoke_rahit_d1;
    uint32_t fn_rint   = (depth == 0) ? mb->fn_invoke_rint_d0  : mb->fn_invoke_rint_d1;

    bool has_rahit_or_rint = false;
    for (uint32_t i = 0; i < stage_count; i++) {
        if (stages[i].stage == VK_SHADER_STAGE_ANY_HIT_BIT_KHR ||
            stages[i].stage == VK_SHADER_STAGE_INTERSECTION_BIT_KHR) {
            has_rahit_or_rint = true;
            break;
        }
    }

    uint32_t lbl_entry     = rt_alloc_id(mb);
    uint32_t lbl_loop_hdr  = rt_alloc_id(mb);
    uint32_t lbl_loop_body = rt_alloc_id(mb);
    uint32_t lbl_candidate = rt_alloc_id(mb);
    uint32_t lbl_loop_cont = rt_alloc_id(mb);
    uint32_t lbl_loop_exit = rt_alloc_id(mb);

    spv_emit_op4(b, RT_OP_FUNCTION, mb->t_void, fn_id, 0u, mb->t_fn_void);
    spv_emit_op1(b, RT_OP_LABEL, lbl_entry);
    spv_emit_op2(b, RT_OP_STORE, mb->g_guard_iter, mb->c_u32_0);
    spv_emit_op1(b, RT_OP_BRANCH, lbl_loop_hdr);

    /* Loop header */
    spv_emit_op1(b, RT_OP_LABEL, lbl_loop_hdr);
    spv_emit_op3(b, RT_OP_LOOP_MERGE, lbl_loop_exit, lbl_loop_cont, 0u);
    spv_emit_op1(b, RT_OP_BRANCH, lbl_loop_body);

    /* Loop body: OpRayQueryProceedKHR */
    spv_emit_op1(b, RT_OP_LABEL, lbl_loop_body);
    (void)rt_env();   /* g_rt_guard_max is set once there */
    uint32_t c_guard_max = rt_get_const_u32(mb, g_rt_guard_max ? g_rt_guard_max : 4096u);
    {
        uint32_t it_old = rt_alloc_id(mb), it_new = rt_alloc_id(mb), it_ok = rt_alloc_id(mb);
        uint32_t lbl_proceed = rt_alloc_id(mb);
        spv_emit_op3(b, RT_OP_LOAD, mb->t_u32, it_old, mb->g_guard_iter);
        spv_emit_op4(b, RT_OP_IADD, mb->t_u32, it_new, it_old, mb->c_u32_1);
        spv_emit_op2(b, RT_OP_STORE, mb->g_guard_iter, it_new);
        spv_emit_op4(b, RT_OP_ULESS_THAN_EQUAL, mb->t_bool, it_ok, it_new, c_guard_max);
        /* over the bound: break to the loop merge (legal unstructured break) */
        spv_emit_op3(b, RT_OP_BRANCH_CONDITIONAL, it_ok, lbl_proceed, lbl_loop_exit);
        spv_emit_op1(b, RT_OP_LABEL, lbl_proceed);
    }
    uint32_t proceed = rt_alloc_id(mb);
    spv_emit_op3(b, RT_OP_RAY_QUERY_PROCEED_KHR, mb->t_bool, proceed, rq_var);
    spv_emit_op3(b, RT_OP_BRANCH_CONDITIONAL, proceed, lbl_candidate, lbl_loop_exit);

    /* Candidate intersection handling */
    spv_emit_op1(b, RT_OP_LABEL, lbl_candidate);
    if (!has_rahit_or_rint) {
        spv_emit_op1(b, RT_OP_RAY_QUERY_CONFIRM_INTERSECTION_KHR, rq_var);
        spv_emit_op1(b, RT_OP_BRANCH, lbl_loop_cont);
    } else {
        uint32_t cand_geom = 0, cand_sbt_off = 0;
        rt_emit_rq_populate_hit_builtins(mb, b, rq_var, mb->c_u32_0, &cand_geom, &cand_sbt_off);
        uint32_t hit_rec = rt_emit_compute_hit_record_bda(mb, b, cand_geom, cand_sbt_off);

        uint32_t c_zero64 = rt_get_const_u64(mb, 0ULL);
        uint32_t has_rec  = rt_alloc_id(mb);
        spv_emit_op4(b, RT_OP_INOT_EQUAL, mb->t_bool, has_rec, hit_rec, c_zero64);

        spv_emit_op2(b, RT_OP_STORE, mb->g_cand_rahit_id, mb->c_u32_0);
        spv_emit_op2(b, RT_OP_STORE, mb->g_cand_rint_id,  mb->c_u32_0);

        uint32_t lbl_load_ids = rt_alloc_id(mb);
        uint32_t lbl_ids_done = rt_alloc_id(mb);
        spv_emit_op2(b, RT_OP_SELECTION_MERGE, lbl_ids_done, 0u);
        spv_emit_op3(b, RT_OP_BRANCH_CONDITIONAL, has_rec, lbl_load_ids, lbl_ids_done);

        spv_emit_op1(b, RT_OP_LABEL, lbl_load_ids);
        uint32_t loaded_rahit = rt_alloc_id(mb);
        uint32_t loaded_rint  = rt_alloc_id(mb);
        rt_emit_psb_load_u32(mb, b, loaded_rahit, hit_rec, 16u);
        rt_emit_psb_load_u32(mb, b, loaded_rint,  hit_rec, 20u);
        spv_emit_op2(b, RT_OP_STORE, mb->g_cand_rahit_id, loaded_rahit);
        spv_emit_op2(b, RT_OP_STORE, mb->g_cand_rint_id,  loaded_rint);
        spv_emit_op1(b, RT_OP_BRANCH, lbl_ids_done);

        spv_emit_op1(b, RT_OP_LABEL, lbl_ids_done);
        uint32_t rahit_id = rt_alloc_id(mb);
        uint32_t rint_id  = rt_alloc_id(mb);
        spv_emit_op3(b, RT_OP_LOAD, mb->t_u32, rahit_id, mb->g_cand_rahit_id);
        spv_emit_op3(b, RT_OP_LOAD, mb->t_u32, rint_id,  mb->g_cand_rint_id);

        uint32_t cand_type = rt_alloc_id(mb);
        uint32_t is_tri    = rt_alloc_id(mb);
        spv_emit_op4(b, RT_OP_RAY_QUERY_GET_INTERSECTION_TYPE_KHR, mb->t_u32, cand_type, rq_var, mb->c_u32_0);
        spv_emit_op4(b, RT_OP_IEQUAL, mb->t_bool, is_tri, cand_type, mb->c_u32_0);

        uint32_t lbl_cand_tri        = rt_alloc_id(mb);
        uint32_t lbl_cand_aabb       = rt_alloc_id(mb);
        uint32_t lbl_cand_merge      = rt_alloc_id(mb);
        uint32_t lbl_tri_conf        = rt_alloc_id(mb);
        uint32_t lbl_tri_after_conf  = rt_alloc_id(mb);
        uint32_t lbl_aabb_gen        = rt_alloc_id(mb);
        uint32_t lbl_aabb_after_gen  = rt_alloc_id(mb);
        uint32_t lbl_do_term         = rt_alloc_id(mb);
        uint32_t lbl_after_term      = rt_alloc_id(mb);

        spv_emit_op2(b, RT_OP_SELECTION_MERGE, lbl_cand_merge, 0u);
        spv_emit_op3(b, RT_OP_BRANCH_CONDITIONAL, is_tri, lbl_cand_tri, lbl_cand_aabb);

        /* Candidate Triangle */
        spv_emit_op1(b, RT_OP_LABEL, lbl_cand_tri);
        rt_emit_rq_store_tri_bary_and_kind(mb, b, rq_var, mb->c_u32_0);
        spv_emit_op2(b, RT_OP_STORE, mb->g_rahit_ignored,  mb->c_u32_0);
        spv_emit_op2(b, RT_OP_STORE, mb->g_ray_terminated, mb->c_u32_0);

        uint32_t call_rahit = rt_alloc_id(mb);
        spv_emit_op4(b, RT_OP_FUNCTION_CALL, mb->t_void, call_rahit, fn_rahit, rahit_id);

        uint32_t ign_val = rt_alloc_id(mb);
        uint32_t not_ign = rt_alloc_id(mb);
        spv_emit_op3(b, RT_OP_LOAD, mb->t_u32, ign_val, mb->g_rahit_ignored);
        spv_emit_op4(b, RT_OP_IEQUAL, mb->t_bool, not_ign, ign_val, mb->c_u32_0);

        spv_emit_op2(b, RT_OP_SELECTION_MERGE, lbl_tri_after_conf, 0u);
        spv_emit_op3(b, RT_OP_BRANCH_CONDITIONAL, not_ign, lbl_tri_conf, lbl_tri_after_conf);

        spv_emit_op1(b, RT_OP_LABEL, lbl_tri_conf);
        spv_emit_op1(b, RT_OP_RAY_QUERY_CONFIRM_INTERSECTION_KHR, rq_var);
        spv_emit_op1(b, RT_OP_BRANCH, lbl_tri_after_conf);

        spv_emit_op1(b, RT_OP_LABEL, lbl_tri_after_conf);
        spv_emit_op1(b, RT_OP_BRANCH, lbl_cand_merge);

        /* Candidate Procedural AABB */
        spv_emit_op1(b, RT_OP_LABEL, lbl_cand_aabb);
        spv_emit_op2(b, RT_OP_STORE, mb->g_cur_rahit_id,       rahit_id);
        spv_emit_op2(b, RT_OP_STORE, mb->g_rint_generated_hit, mb->c_u32_0);
        spv_emit_op2(b, RT_OP_STORE, mb->g_ray_terminated,     mb->c_u32_0);

        uint32_t call_rint = rt_alloc_id(mb);
        spv_emit_op4(b, RT_OP_FUNCTION_CALL, mb->t_void, call_rint, fn_rint, rint_id);

        uint32_t gen_val = rt_alloc_id(mb);
        uint32_t has_gen = rt_alloc_id(mb);
        spv_emit_op3(b, RT_OP_LOAD, mb->t_u32, gen_val, mb->g_rint_generated_hit);
        spv_emit_op4(b, RT_OP_INOT_EQUAL, mb->t_bool, has_gen, gen_val, mb->c_u32_0);

        spv_emit_op2(b, RT_OP_SELECTION_MERGE, lbl_aabb_after_gen, 0u);
        spv_emit_op3(b, RT_OP_BRANCH_CONDITIONAL, has_gen, lbl_aabb_gen, lbl_aabb_after_gen);

        spv_emit_op1(b, RT_OP_LABEL, lbl_aabb_gen);
        uint32_t gen_t = rt_alloc_id(mb);
        spv_emit_op3(b, RT_OP_LOAD, mb->t_f32, gen_t, mb->g_ray_tmax);
        spv_emit_op2(b, RT_OP_RAY_QUERY_GENERATE_INTERSECTION_KHR, rq_var, gen_t);
        spv_emit_op1(b, RT_OP_BRANCH, lbl_aabb_after_gen);

        spv_emit_op1(b, RT_OP_LABEL, lbl_aabb_after_gen);
        spv_emit_op1(b, RT_OP_BRANCH, lbl_cand_merge);

        /* Candidate Merge: check OpTerminateRayKHR */
        spv_emit_op1(b, RT_OP_LABEL, lbl_cand_merge);
        uint32_t term_val = rt_alloc_id(mb);
        uint32_t is_term  = rt_alloc_id(mb);
        spv_emit_op3(b, RT_OP_LOAD, mb->t_u32, term_val, mb->g_ray_terminated);
        spv_emit_op4(b, RT_OP_INOT_EQUAL, mb->t_bool, is_term, term_val, mb->c_u32_0);

        spv_emit_op2(b, RT_OP_SELECTION_MERGE, lbl_after_term, 0u);
        spv_emit_op3(b, RT_OP_BRANCH_CONDITIONAL, is_term, lbl_do_term, lbl_after_term);

        spv_emit_op1(b, RT_OP_LABEL, lbl_do_term);
        spv_emit_op1(b, RT_OP_RAY_QUERY_TERMINATE_KHR, rq_var);
        spv_emit_op1(b, RT_OP_BRANCH, lbl_after_term);

        spv_emit_op1(b, RT_OP_LABEL, lbl_after_term);
        spv_emit_op1(b, RT_OP_BRANCH, lbl_loop_cont);
    }

    /* Loop continue */
    spv_emit_op1(b, RT_OP_LABEL, lbl_loop_cont);
    spv_emit_op1(b, RT_OP_BRANCH, lbl_loop_hdr);

    /* Loop exit: evaluate committed hit vs miss */
    spv_emit_op1(b, RT_OP_LABEL, lbl_loop_exit);
    uint32_t g_tripped = rt_alloc_id(mb);
    {
        uint32_t it_fin = rt_alloc_id(mb);
        uint32_t lbl_g_do = rt_alloc_id(mb), lbl_g_after = rt_alloc_id(mb);
        spv_emit_op3(b, RT_OP_LOAD, mb->t_u32, it_fin, mb->g_guard_iter);
        spv_emit_op4(b, RT_OP_UGREATER_THAN, mb->t_bool, g_tripped, it_fin, c_guard_max);
        spv_emit_op2(b, RT_OP_SELECTION_MERGE, lbl_g_after, 0u);
        spv_emit_op3(b, RT_OP_BRANCH_CONDITIONAL, g_tripped, lbl_g_do, lbl_g_after);
        spv_emit_op1(b, RT_OP_LABEL, lbl_g_do);
        spv_emit_op1(b, RT_OP_RAY_QUERY_TERMINATE_KHR, rq_var);
        VkDeviceAddress guard_bda = __atomic_load_n(&g_rt_guard_bda, __ATOMIC_ACQUIRE);
        if (guard_bda) {
            uint32_t gptr = rt_alloc_id(mb), gres = rt_alloc_id(mb);
            uint32_t c_gbda  = rt_get_const_u64(mb, (uint64_t)guard_bda);
            /* Device scope (1) needs VulkanMemoryModelDeviceScope under the Vulkan memory model: use QueueFamily (5) there */
            uint32_t c_scope = rt_get_const_u32(mb, mb->has_vulkan_memory_model ? 5u : 1u);
            spv_emit_op3(b, RT_OP_CONVERT_U_TO_PTR, mb->t_ptr_psb_u32, gptr, c_gbda);
            spv_emit_op5(b, 232u /* OpAtomicIIncrement */, mb->t_u32, gres, gptr, c_scope, mb->c_u32_0);
        }
        spv_emit_op1(b, RT_OP_BRANCH, lbl_g_after);
        spv_emit_op1(b, RT_OP_LABEL, lbl_g_after);
    }
    uint32_t comm_type = rt_alloc_id(mb);
    uint32_t has_hit_raw = rt_alloc_id(mb);
    uint32_t not_tripped = rt_alloc_id(mb);
    uint32_t has_hit   = rt_alloc_id(mb);
    spv_emit_op4(b, RT_OP_RAY_QUERY_GET_INTERSECTION_TYPE_KHR, mb->t_u32, comm_type, rq_var, mb->c_u32_1);
    spv_emit_op4(b, RT_OP_INOT_EQUAL, mb->t_bool, has_hit_raw, comm_type, mb->c_u32_0);
    spv_emit_op3(b, RT_OP_LOGICAL_NOT, mb->t_bool, not_tripped, g_tripped);
    /* a cut ray is a miss, whatever it had committed so far */
    spv_emit_op4(b, RT_OP_LOGICAL_AND, mb->t_bool, has_hit, has_hit_raw, not_tripped);

    uint32_t lbl_hit            = rt_alloc_id(mb);
    uint32_t lbl_miss           = rt_alloc_id(mb);
    uint32_t lbl_done           = rt_alloc_id(mb);
    uint32_t lbl_comm_tri       = rt_alloc_id(mb);
    uint32_t lbl_comm_aabb      = rt_alloc_id(mb);
    uint32_t lbl_hit_attr_done  = rt_alloc_id(mb);
    uint32_t lbl_do_chit        = rt_alloc_id(mb);
    uint32_t lbl_after_chit     = rt_alloc_id(mb);
    uint32_t lbl_do_miss        = rt_alloc_id(mb);
    uint32_t lbl_after_miss     = rt_alloc_id(mb);

    spv_emit_op2(b, RT_OP_SELECTION_MERGE, lbl_done, 0u);
    spv_emit_op3(b, RT_OP_BRANCH_CONDITIONAL, has_hit, lbl_hit, lbl_miss);

    /* Committed Hit */
    spv_emit_op1(b, RT_OP_LABEL, lbl_hit);
    uint32_t comm_geom = 0, comm_sbt_off = 0;
    rt_emit_rq_populate_hit_builtins(mb, b, rq_var, mb->c_u32_1, &comm_geom, &comm_sbt_off);

    uint32_t is_comm_tri = rt_alloc_id(mb);
    spv_emit_op4(b, RT_OP_IEQUAL, mb->t_bool, is_comm_tri, comm_type, mb->c_u32_1);
    spv_emit_op2(b, RT_OP_SELECTION_MERGE, lbl_hit_attr_done, 0u);
    spv_emit_op3(b, RT_OP_BRANCH_CONDITIONAL, is_comm_tri, lbl_comm_tri, lbl_comm_aabb);

    spv_emit_op1(b, RT_OP_LABEL, lbl_comm_tri);
    rt_emit_rq_store_tri_bary_and_kind(mb, b, rq_var, mb->c_u32_1);
    spv_emit_op1(b, RT_OP_BRANCH, lbl_hit_attr_done);

    spv_emit_op1(b, RT_OP_LABEL, lbl_comm_aabb);
    rt_emit_copy_hitattr_array(mb, b, mb->g_hitattr_u32, mb->g_committed_hitattr_u32);
    spv_emit_op1(b, RT_OP_BRANCH, lbl_hit_attr_done);

    spv_emit_op1(b, RT_OP_LABEL, lbl_hit_attr_done);
    uint32_t comm_hit_rec = rt_emit_compute_hit_record_bda(mb, b, comm_geom, comm_sbt_off);

    /* Check SkipClosestHitShaderKHR (flag bit 0x8) and valid hit record */
    uint32_t c_zero64    = rt_get_const_u64(mb, 0ULL);
    uint32_t has_hit_rec = rt_alloc_id(mb);
    uint32_t rflags      = rt_alloc_id(mb);
    uint32_t skip_bit    = rt_alloc_id(mb);
    uint32_t not_skip    = rt_alloc_id(mb);
    uint32_t run_chit    = rt_alloc_id(mb);

    spv_emit_op4(b, RT_OP_INOT_EQUAL, mb->t_bool, has_hit_rec, comm_hit_rec, c_zero64);
    spv_emit_op3(b, RT_OP_LOAD, mb->t_u32, rflags, mb->g_incoming_ray_flags);
    spv_emit_op4(b, RT_OP_BITWISE_AND, mb->t_u32, skip_bit, rflags, mb->c_u32_8);
    spv_emit_op4(b, RT_OP_IEQUAL, mb->t_bool, not_skip, skip_bit, mb->c_u32_0);
    spv_emit_op4(b, RT_OP_LOGICAL_AND, mb->t_bool, run_chit, has_hit_rec, not_skip);

    spv_emit_op2(b, RT_OP_SELECTION_MERGE, lbl_after_chit, 0u);
    spv_emit_op3(b, RT_OP_BRANCH_CONDITIONAL, run_chit, lbl_do_chit, lbl_after_chit);

    spv_emit_op1(b, RT_OP_LABEL, lbl_do_chit);
    uint32_t rchit_id = rt_alloc_id(mb);
    rt_emit_psb_load_u32(mb, b, rchit_id, comm_hit_rec, 12u);
    rt_emit_stage_switch_inline(mb, b, rchit_id, stages, parses, stage_count,
                                VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR, depth);
    spv_emit_op1(b, RT_OP_BRANCH, lbl_after_chit);

    spv_emit_op1(b, RT_OP_LABEL, lbl_after_chit);
    spv_emit_op1(b, RT_OP_BRANCH, lbl_done);

    /* Committed Miss */
    spv_emit_op1(b, RT_OP_LABEL, lbl_miss);
    uint32_t c_32_64      = rt_get_const_u64(mb, 32ULL);
    uint32_t miss_base    = rt_alloc_id(mb);
    uint32_t miss_size    = rt_alloc_id(mb);
    uint32_t has_miss_base= rt_alloc_id(mb);
    spv_emit_op3(b, RT_OP_LOAD, mb->t_u64, miss_base, mb->g_miss_base);
    spv_emit_op3(b, RT_OP_LOAD, mb->t_u64, miss_size, mb->g_miss_size);
    spv_emit_op4(b, RT_OP_INOT_EQUAL, mb->t_bool, has_miss_base, miss_base, c_zero64);

    uint32_t miss_idx     = rt_alloc_id(mb);
    uint32_t miss_stride  = rt_alloc_id(mb);
    uint32_t miss_off32   = rt_alloc_id(mb);
    uint32_t miss_off64   = rt_alloc_id(mb);
    spv_emit_op3(b, RT_OP_LOAD,     mb->t_u32, miss_idx,    mb->g_trace_miss_index);
    spv_emit_op3(b, RT_OP_LOAD,     mb->t_u32, miss_stride, mb->g_miss_stride);
    spv_emit_op4(b, RT_OP_IMUL,     mb->t_u32, miss_off32,  miss_idx, miss_stride);
    spv_emit_op3(b, RT_OP_UCONVERT, mb->t_u64, miss_off64,  miss_off32);

    /* Miss record size: max(miss_stride, 32ULL) */
    uint32_t miss_stride64 = rt_alloc_id(mb);
    spv_emit_op3(b, RT_OP_UCONVERT, mb->t_u64, miss_stride64, miss_stride);
    uint32_t is_mstride_ge32 = rt_alloc_id(mb);
    spv_emit_op4(b, RT_OP_UGREATER_THAN_EQUAL, mb->t_bool, is_mstride_ge32, miss_stride64, c_32_64);
    uint32_t miss_rec_sz = rt_alloc_id(mb);
    spv_emit_op5(b, RT_OP_SELECT, mb->t_u64, miss_rec_sz, is_mstride_ge32, miss_stride64, c_32_64);

    uint32_t miss_end_off = rt_alloc_id(mb);
    spv_emit_op4(b, RT_OP_IADD, mb->t_u64, miss_end_off, miss_off64, miss_rec_sz);

    uint32_t has_miss_sz = rt_alloc_id(mb);
    spv_emit_op4(b, RT_OP_UGREATER_THAN_EQUAL, mb->t_bool, has_miss_sz, miss_size, miss_rec_sz);

    uint32_t miss_within_sz = rt_alloc_id(mb);
    spv_emit_op4(b, RT_OP_ULESS_THAN_EQUAL, mb->t_bool, miss_within_sz, miss_end_off, miss_size);

    uint32_t miss_in_bounds = rt_alloc_id(mb);
    spv_emit_op4(b, RT_OP_LOGICAL_AND, mb->t_bool, miss_in_bounds, has_miss_sz, miss_within_sz);

    uint32_t is_valid_miss_addr = rt_alloc_id(mb);
    spv_emit_op4(b, RT_OP_LOGICAL_AND, mb->t_bool, is_valid_miss_addr, has_miss_base, miss_in_bounds);

    uint32_t calc_miss_rec = rt_alloc_id(mb);
    spv_emit_op4(b, RT_OP_IADD, mb->t_u64, calc_miss_rec, miss_base, miss_off64);

    uint32_t c_min_bda = rt_get_const_u64(mb, RT_MIN_VALID_GPU_BDA);
    uint32_t c_max_bda = rt_get_const_u64(mb, RT_MAX_VALID_GPU_BDA);
    uint32_t is_miss_ge_min = rt_alloc_id(mb);
    spv_emit_op4(b, RT_OP_UGREATER_THAN_EQUAL, mb->t_bool, is_miss_ge_min, calc_miss_rec, c_min_bda);
    uint32_t is_miss_lt_max = rt_alloc_id(mb);
    spv_emit_op4(b, RT_OP_ULESS_THAN, mb->t_bool, is_miss_lt_max, calc_miss_rec, c_max_bda);
    uint32_t is_miss_range_ok = rt_alloc_id(mb);
    spv_emit_op4(b, RT_OP_LOGICAL_AND, mb->t_bool, is_miss_range_ok, is_miss_ge_min, is_miss_lt_max);

    uint32_t is_safe_miss_bda = rt_alloc_id(mb);
    spv_emit_op4(b, RT_OP_LOGICAL_AND, mb->t_bool, is_safe_miss_bda, is_valid_miss_addr, is_miss_range_ok);

    uint32_t safe_miss_hdr = rt_alloc_id(mb);
    uint32_t pc_bda = rt_alloc_id(mb);
    spv_emit_op3(b, RT_OP_LOAD, mb->t_u64, pc_bda, mb->g_push_consts_bda);
    spv_emit_op5(b, RT_OP_SELECT, mb->t_u64, safe_miss_hdr, is_safe_miss_bda, calc_miss_rec, pc_bda);

    uint32_t loaded_miss_magic = rt_alloc_id(mb);
    rt_emit_psb_load_u32(mb, b, loaded_miss_magic, safe_miss_hdr, 0u);

    uint32_t c_magic = rt_get_const_u32(mb, RT_GROUP_HANDLE_MAGIC);
    uint32_t is_miss_magic_ok = rt_alloc_id(mb);
    spv_emit_op4(b, RT_OP_IEQUAL, mb->t_bool, is_miss_magic_ok, loaded_miss_magic, c_magic);

    uint32_t run_miss = rt_alloc_id(mb);
    spv_emit_op4(b, RT_OP_LOGICAL_AND, mb->t_bool, run_miss, is_safe_miss_bda, is_miss_magic_ok);

    spv_emit_op2(b, RT_OP_SELECTION_MERGE, lbl_after_miss, 0u);
    spv_emit_op3(b, RT_OP_BRANCH_CONDITIONAL, run_miss, lbl_do_miss, lbl_after_miss);

    spv_emit_op1(b, RT_OP_LABEL, lbl_do_miss);
    uint32_t msrb_bda  = rt_alloc_id(mb);
    spv_emit_op4(b, RT_OP_IADD, mb->t_u64, msrb_bda, calc_miss_rec, c_32_64);
    spv_emit_op2(b, RT_OP_STORE, mb->g_cur_record_bda, msrb_bda);

    uint32_t rmiss_id = rt_alloc_id(mb);
    rt_emit_psb_load_u32(mb, b, rmiss_id, calc_miss_rec, 8u);
    rt_emit_stage_switch_inline(mb, b, rmiss_id, stages, parses, stage_count,
                                VK_SHADER_STAGE_MISS_BIT_KHR, depth);
    spv_emit_op1(b, RT_OP_BRANCH, lbl_after_miss);

    spv_emit_op1(b, RT_OP_LABEL, lbl_after_miss);
    spv_emit_op1(b, RT_OP_BRANCH, lbl_done);

    spv_emit_op1(b, RT_OP_LABEL, lbl_done);
    spv_emit_op0(b, RT_OP_RETURN);
    spv_emit_op0(b, RT_OP_FUNCTION_END);
}

static uint32_t rt_emit_load_pc_member(struct RtMegaBuilder *mb, struct SpvBuf *b,
                                       uint32_t member_idx, bool is_u64) {
    uint32_t idx_c    = rt_get_const_i32(mb, (int32_t)member_idx);
    uint32_t ptr_type = is_u64 ? mb->t_ptr_pc_u64 : mb->t_ptr_pc_u32;
    uint32_t val_type = is_u64 ? mb->t_u64        : mb->t_u32;
    uint32_t ptr_id   = rt_alloc_id(mb);
    uint32_t val_id   = rt_alloc_id(mb);
    spv_emit_op4(b, RT_OP_ACCESS_CHAIN, ptr_type, ptr_id, mb->v_rt_pc, idx_c);
    spv_emit_op3(b, RT_OP_LOAD, val_type, val_id, ptr_id);
    return val_id;
}

static void rt_emit_main_fn(struct RtMegaBuilder *mb,
                            const struct RtStageRecord *stages,
                            const struct RtStageParse *parses,
                            uint32_t stage_count) {
    struct SpvBuf *b = &mb->functions;
    uint32_t lbl_entry    = rt_alloc_id(mb);
    uint32_t lbl_ind1     = rt_alloc_id(mb);
    uint32_t lbl_not_ind1 = rt_alloc_id(mb);
    uint32_t lbl_ind2     = rt_alloc_id(mb);
    uint32_t lbl_ind_done = rt_alloc_id(mb);
    uint32_t lbl_in_bounds= rt_alloc_id(mb);
    uint32_t lbl_exit     = rt_alloc_id(mb);

    spv_emit_op4(b, RT_OP_FUNCTION, mb->t_void, mb->fn_main, 0u, mb->t_fn_void);
    spv_emit_op1(b, RT_OP_LABEL, lbl_entry);

    /* Load push constant parameters into globals */
    uint32_t rgen_base   = rt_emit_load_pc_member(mb, b, 0,  true);
    uint32_t rgen_sz32   = rt_emit_load_pc_member(mb, b, 2,  false);
    uint32_t miss_base   = rt_emit_load_pc_member(mb, b, 3,  true);
    uint32_t miss_stride = rt_emit_load_pc_member(mb, b, 4,  false);
    uint32_t miss_sz32   = rt_emit_load_pc_member(mb, b, 5,  false);
    uint32_t hit_base    = rt_emit_load_pc_member(mb, b, 6,  true);
    uint32_t hit_stride  = rt_emit_load_pc_member(mb, b, 7,  false);
    uint32_t hit_sz32    = rt_emit_load_pc_member(mb, b, 8,  false);
    uint32_t call_base   = rt_emit_load_pc_member(mb, b, 9,  true);
    uint32_t call_stride = rt_emit_load_pc_member(mb, b, 10, false);
    uint32_t call_sz32   = rt_emit_load_pc_member(mb, b, 11, false);
    uint32_t lw          = rt_emit_load_pc_member(mb, b, 12, false);
    uint32_t lh          = rt_emit_load_pc_member(mb, b, 13, false);
    uint32_t ld          = rt_emit_load_pc_member(mb, b, 14, false);
    uint32_t ind_mode    = rt_emit_load_pc_member(mb, b, 15, false);
    /* local1d-1112: bit RT_PC_LOCAL1D of indirect_mode = this direct dispatch uses the 1-D launch-id mapping; only
     * emitted with MALI_COMPAT_RT_LOCAL1D=1 (without it the SPIR-V is unchanged) */
    const bool local1d = rt_env()->local1d != 0;
    uint32_t l1d_flag = 0;
    if (local1d) {
        uint32_t raw = ind_mode, fl = rt_alloc_id(mb), md = rt_alloc_id(mb);
        spv_emit_op4(b, RT_OP_BITWISE_AND, mb->t_u32, fl, raw, rt_get_const_u32(mb, RT_PC_LOCAL1D));
        spv_emit_op4(b, RT_OP_BITWISE_AND, mb->t_u32, md, raw, rt_get_const_u32(mb, 0xffu));
        l1d_flag = rt_alloc_id(mb);
        spv_emit_op4(b, RT_OP_INOT_EQUAL, mb->t_bool, l1d_flag, fl, mb->c_u32_0);
        ind_mode = md;
    }
    uint32_t pc_bda      = rt_emit_load_pc_member(mb, b, 16, true);
    uint32_t ind_bda     = rt_emit_load_pc_member(mb, b, 17, true);

    uint32_t rgen_sz64   = rt_alloc_id(mb);
    uint32_t miss_sz64   = rt_alloc_id(mb);
    uint32_t hit_sz64    = rt_alloc_id(mb);
    uint32_t call_sz64   = rt_alloc_id(mb);
    spv_emit_op3(b, RT_OP_UCONVERT, mb->t_u64, rgen_sz64, rgen_sz32);
    spv_emit_op3(b, RT_OP_UCONVERT, mb->t_u64, miss_sz64, miss_sz32);
    spv_emit_op3(b, RT_OP_UCONVERT, mb->t_u64, hit_sz64,  hit_sz32);
    spv_emit_op3(b, RT_OP_UCONVERT, mb->t_u64, call_sz64, call_sz32);

    uint32_t lsize_dir = rt_alloc_id(mb);
    spv_emit_op5(b, RT_OP_COMPOSITE_CONSTRUCT, mb->t_v3u32, lsize_dir, lw, lh, ld);

    spv_emit_op2(b, RT_OP_STORE, mb->g_rgen_base,       rgen_base);
    spv_emit_op2(b, RT_OP_STORE, mb->g_rgen_size,       rgen_sz64);
    spv_emit_op2(b, RT_OP_STORE, mb->g_miss_base,       miss_base);
    spv_emit_op2(b, RT_OP_STORE, mb->g_miss_stride,     miss_stride);
    spv_emit_op2(b, RT_OP_STORE, mb->g_miss_size,       miss_sz64);
    spv_emit_op2(b, RT_OP_STORE, mb->g_hit_base,        hit_base);
    spv_emit_op2(b, RT_OP_STORE, mb->g_hit_stride,      hit_stride);
    spv_emit_op2(b, RT_OP_STORE, mb->g_hit_size,        hit_sz64);
    spv_emit_op2(b, RT_OP_STORE, mb->g_call_base,       call_base);
    spv_emit_op2(b, RT_OP_STORE, mb->g_call_stride,     call_stride);
    spv_emit_op2(b, RT_OP_STORE, mb->g_call_size,       call_sz64);
    spv_emit_op2(b, RT_OP_STORE, mb->g_launch_size,     lsize_dir);
    spv_emit_op2(b, RT_OP_STORE, mb->g_push_consts_bda, pc_bda);

    /* Check indirect_mode (1 = TraceRaysIndirectKHR, 2 = TraceRaysIndirect2KHR) */
    uint32_t is_ind1 = rt_alloc_id(mb);
    spv_emit_op4(b, RT_OP_IEQUAL, mb->t_bool, is_ind1, ind_mode, mb->c_u32_1);
    spv_emit_op2(b, RT_OP_SELECTION_MERGE, lbl_ind_done, 0u);
    spv_emit_op3(b, RT_OP_BRANCH_CONDITIONAL, is_ind1, lbl_ind1, lbl_not_ind1);

    /* Indirect1: load width(0), height(4), depth(8) from indirect_bda */
    spv_emit_op1(b, RT_OP_LABEL, lbl_ind1);
    uint32_t i1_w = rt_alloc_id(mb), i1_h = rt_alloc_id(mb), i1_d = rt_alloc_id(mb), i1_sz = rt_alloc_id(mb);
    rt_emit_psb_load_u32(mb, b, i1_w, ind_bda, 0u);
    rt_emit_psb_load_u32(mb, b, i1_h, ind_bda, 4u);
    rt_emit_psb_load_u32(mb, b, i1_d, ind_bda, 8u);
    spv_emit_op5(b, RT_OP_COMPOSITE_CONSTRUCT, mb->t_v3u32, i1_sz, i1_w, i1_h, i1_d);
    spv_emit_op2(b, RT_OP_STORE, mb->g_launch_size, i1_sz);
    spv_emit_op1(b, RT_OP_BRANCH, lbl_ind_done);

    /* Check Indirect2 */
    spv_emit_op1(b, RT_OP_LABEL, lbl_not_ind1);
    uint32_t is_ind2 = rt_alloc_id(mb);
    uint32_t lbl_after_ind2 = rt_alloc_id(mb);
    spv_emit_op4(b, RT_OP_IEQUAL, mb->t_bool, is_ind2, ind_mode, mb->c_u32_2);
    spv_emit_op2(b, RT_OP_SELECTION_MERGE, lbl_after_ind2, 0u);
    spv_emit_op3(b, RT_OP_BRANCH_CONDITIONAL, is_ind2, lbl_ind2, lbl_after_ind2);

    /* Indirect2: VkTraceRaysIndirectCommand2KHR */
    spv_emit_op1(b, RT_OP_LABEL, lbl_ind2);
    uint32_t i2_rgen_b = rt_alloc_id(mb), i2_rgen_sz = rt_alloc_id(mb);
    uint32_t i2_miss_b = rt_alloc_id(mb), i2_miss_sz = rt_alloc_id(mb), i2_miss_s = rt_alloc_id(mb);
    uint32_t i2_hit_b  = rt_alloc_id(mb), i2_hit_sz  = rt_alloc_id(mb), i2_hit_s  = rt_alloc_id(mb);
    uint32_t i2_call_b = rt_alloc_id(mb), i2_call_sz = rt_alloc_id(mb), i2_call_s = rt_alloc_id(mb);
    uint32_t i2_w = rt_alloc_id(mb), i2_h = rt_alloc_id(mb), i2_d = rt_alloc_id(mb), i2_sz = rt_alloc_id(mb);
    rt_emit_psb_load_u64(mb, b, i2_rgen_b,  ind_bda, 0u);
    rt_emit_psb_load_u64(mb, b, i2_rgen_sz, ind_bda, 8u);
    rt_emit_psb_load_u64(mb, b, i2_miss_b,  ind_bda, 16u);
    rt_emit_psb_load_u64(mb, b, i2_miss_sz, ind_bda, 24u);
    rt_emit_psb_load_u32(mb, b, i2_miss_s,  ind_bda, 32u);
    rt_emit_psb_load_u64(mb, b, i2_hit_b,   ind_bda, 40u);
    rt_emit_psb_load_u64(mb, b, i2_hit_sz,  ind_bda, 48u);
    rt_emit_psb_load_u32(mb, b, i2_hit_s,   ind_bda, 56u);
    rt_emit_psb_load_u64(mb, b, i2_call_b,  ind_bda, 64u);
    rt_emit_psb_load_u64(mb, b, i2_call_sz, ind_bda, 72u);
    rt_emit_psb_load_u32(mb, b, i2_call_s,  ind_bda, 80u);
    rt_emit_psb_load_u32(mb, b, i2_w,       ind_bda, 88u);
    rt_emit_psb_load_u32(mb, b, i2_h,       ind_bda, 92u);
    rt_emit_psb_load_u32(mb, b, i2_d,       ind_bda, 96u);
    spv_emit_op5(b, RT_OP_COMPOSITE_CONSTRUCT, mb->t_v3u32, i2_sz, i2_w, i2_h, i2_d);
    spv_emit_op2(b, RT_OP_STORE, mb->g_rgen_base,   i2_rgen_b);
    spv_emit_op2(b, RT_OP_STORE, mb->g_rgen_size,   i2_rgen_sz);
    spv_emit_op2(b, RT_OP_STORE, mb->g_miss_base,   i2_miss_b);
    spv_emit_op2(b, RT_OP_STORE, mb->g_miss_stride, i2_miss_s);
    spv_emit_op2(b, RT_OP_STORE, mb->g_miss_size,   i2_miss_sz);
    spv_emit_op2(b, RT_OP_STORE, mb->g_hit_base,    i2_hit_b);
    spv_emit_op2(b, RT_OP_STORE, mb->g_hit_stride,  i2_hit_s);
    spv_emit_op2(b, RT_OP_STORE, mb->g_hit_size,    i2_hit_sz);
    spv_emit_op2(b, RT_OP_STORE, mb->g_call_base,   i2_call_b);
    spv_emit_op2(b, RT_OP_STORE, mb->g_call_stride, i2_call_s);
    spv_emit_op2(b, RT_OP_STORE, mb->g_call_size,   i2_call_sz);
    spv_emit_op2(b, RT_OP_STORE, mb->g_launch_size, i2_sz);
    spv_emit_op1(b, RT_OP_BRANCH, lbl_after_ind2);

    spv_emit_op1(b, RT_OP_LABEL, lbl_after_ind2);
    spv_emit_op1(b, RT_OP_BRANCH, lbl_ind_done);

    /* Bounds check: gid.x < launch_size.x && gid.y < launch_size.y && gid.z < launch_size.z */
    spv_emit_op1(b, RT_OP_LABEL, lbl_ind_done);
    uint32_t gid   = rt_alloc_id(mb);
    uint32_t lsize = rt_alloc_id(mb);
    spv_emit_op3(b, RT_OP_LOAD, mb->t_v3u32, gid,   mb->v_gl_global_inv_id);
    if (local1d) {   /* gid = flag ? (wg.x*32 + local index, wg.y, wg.z) : GlobalInvocationId */
        uint32_t t_pin_u32 = rt_find_or_emit_type(mb, RT_OP_TYPE_POINTER, RT_SC_INPUT, mb->t_u32, 0);
        uint32_t v_wg = rt_emit_global_var(mb, mb->t_ptr_input_v3u32, RT_SC_INPUT);
        spv_emit_op3(&mb->decorations, RT_OP_DECORATE, v_wg, RT_DEC_BUILTIN, 26u /* WorkgroupId */);
        uint32_t v_li = rt_emit_global_var(mb, t_pin_u32, RT_SC_INPUT);
        spv_emit_op3(&mb->decorations, RT_OP_DECORATE, v_li, RT_DEC_BUILTIN, 29u /* LocalInvocationIndex */);
        uint32_t wg = rt_alloc_id(mb), li = rt_alloc_id(mb);
        spv_emit_op3(b, RT_OP_LOAD, mb->t_v3u32, wg, v_wg);
        spv_emit_op3(b, RT_OP_LOAD, mb->t_u32, li, v_li);
        uint32_t c[3], g[3], o[3];
        for (uint32_t q = 0; q < 3; q++) {
            c[q] = rt_alloc_id(mb); g[q] = rt_alloc_id(mb); o[q] = rt_alloc_id(mb);
            spv_emit_op4(b, RT_OP_COMPOSITE_EXTRACT, mb->t_u32, c[q], wg, q);
            spv_emit_op4(b, RT_OP_COMPOSITE_EXTRACT, mb->t_u32, g[q], gid, q);
        }
        uint32_t x32 = rt_alloc_id(mb), x1 = rt_alloc_id(mb);
        spv_emit_op4(b, RT_OP_IMUL, mb->t_u32, x32, c[0], mb->c_u32_32);
        spv_emit_op4(b, RT_OP_IADD, mb->t_u32, x1, x32, li);
        spv_emit_op5(b, RT_OP_SELECT, mb->t_u32, o[0], l1d_flag, x1, g[0]);
        spv_emit_op5(b, RT_OP_SELECT, mb->t_u32, o[1], l1d_flag, c[1], g[1]);
        spv_emit_op5(b, RT_OP_SELECT, mb->t_u32, o[2], l1d_flag, c[2], g[2]);
        uint32_t gid2 = rt_alloc_id(mb);
        spv_emit_op5(b, RT_OP_COMPOSITE_CONSTRUCT, mb->t_v3u32, gid2, o[0], o[1], o[2]);
        gid = gid2;
    }
    spv_emit_op3(b, RT_OP_LOAD, mb->t_v3u32, lsize, mb->g_launch_size);

    uint32_t gx = rt_alloc_id(mb), gy = rt_alloc_id(mb), gz = rt_alloc_id(mb);
    uint32_t sx = rt_alloc_id(mb), sy = rt_alloc_id(mb), sz = rt_alloc_id(mb);
    spv_emit_op4(b, RT_OP_COMPOSITE_EXTRACT, mb->t_u32, gx, gid,   0u);
    spv_emit_op4(b, RT_OP_COMPOSITE_EXTRACT, mb->t_u32, gy, gid,   1u);
    spv_emit_op4(b, RT_OP_COMPOSITE_EXTRACT, mb->t_u32, gz, gid,   2u);
    spv_emit_op4(b, RT_OP_COMPOSITE_EXTRACT, mb->t_u32, sx, lsize, 0u);
    spv_emit_op4(b, RT_OP_COMPOSITE_EXTRACT, mb->t_u32, sy, lsize, 1u);
    spv_emit_op4(b, RT_OP_COMPOSITE_EXTRACT, mb->t_u32, sz, lsize, 2u);

    uint32_t bx = rt_alloc_id(mb), by = rt_alloc_id(mb), bz = rt_alloc_id(mb);
    uint32_t bxy = rt_alloc_id(mb), bxyz = rt_alloc_id(mb);
    spv_emit_op4(b, RT_OP_ULESS_THAN, mb->t_bool, bx, gx, sx);
    spv_emit_op4(b, RT_OP_ULESS_THAN, mb->t_bool, by, gy, sy);
    spv_emit_op4(b, RT_OP_ULESS_THAN, mb->t_bool, bz, gz, sz);
    spv_emit_op4(b, RT_OP_LOGICAL_AND, mb->t_bool, bxy,  bx,  by);
    spv_emit_op4(b, RT_OP_LOGICAL_AND, mb->t_bool, bxyz, bxy, bz);

    spv_emit_op2(b, RT_OP_SELECTION_MERGE, lbl_exit, 0u);
    spv_emit_op3(b, RT_OP_BRANCH_CONDITIONAL, bxyz, lbl_in_bounds, lbl_exit);

    /* Active raygen thread */
    spv_emit_op1(b, RT_OP_LABEL, lbl_in_bounds);
    spv_emit_op2(b, RT_OP_STORE, mb->g_launch_id, gid);

    uint32_t cur_rgen_base = rt_alloc_id(mb);
    uint32_t cur_rgen_size = rt_alloc_id(mb);
    spv_emit_op3(b, RT_OP_LOAD, mb->t_u64, cur_rgen_base, mb->g_rgen_base);
    spv_emit_op3(b, RT_OP_LOAD, mb->t_u64, cur_rgen_size, mb->g_rgen_size);

    uint32_t c_zero64   = rt_get_const_u64(mb, 0ULL);
    uint32_t c_32_64    = rt_get_const_u64(mb, 32ULL);
    uint32_t has_rgen   = rt_alloc_id(mb);
    uint32_t has_rgen_sz= rt_alloc_id(mb);
    uint32_t run_rgen   = rt_alloc_id(mb);
    spv_emit_op4(b, RT_OP_INOT_EQUAL, mb->t_bool, has_rgen, cur_rgen_base, c_zero64);
    spv_emit_op4(b, RT_OP_UGREATER_THAN_EQUAL, mb->t_bool, has_rgen_sz, cur_rgen_size, c_32_64);
    spv_emit_op4(b, RT_OP_LOGICAL_AND, mb->t_bool, run_rgen, has_rgen, has_rgen_sz);

    uint32_t c_min_bda = rt_get_const_u64(mb, RT_MIN_VALID_GPU_BDA);
    uint32_t c_max_bda = rt_get_const_u64(mb, RT_MAX_VALID_GPU_BDA);
    uint32_t is_rgen_ge_min = rt_alloc_id(mb);
    spv_emit_op4(b, RT_OP_UGREATER_THAN_EQUAL, mb->t_bool, is_rgen_ge_min, cur_rgen_base, c_min_bda);
    uint32_t is_rgen_lt_max = rt_alloc_id(mb);
    spv_emit_op4(b, RT_OP_ULESS_THAN, mb->t_bool, is_rgen_lt_max, cur_rgen_base, c_max_bda);
    uint32_t is_rgen_range_ok = rt_alloc_id(mb);
    spv_emit_op4(b, RT_OP_LOGICAL_AND, mb->t_bool, is_rgen_range_ok, is_rgen_ge_min, is_rgen_lt_max);

    uint32_t is_safe_rgen_bda = rt_alloc_id(mb);
    spv_emit_op4(b, RT_OP_LOGICAL_AND, mb->t_bool, is_safe_rgen_bda, run_rgen, is_rgen_range_ok);

    /* Magic check for raygen handle */
    uint32_t safe_rgen_hdr = rt_alloc_id(mb);
    spv_emit_op5(b, RT_OP_SELECT, mb->t_u64, safe_rgen_hdr, is_safe_rgen_bda, cur_rgen_base, pc_bda);

    uint32_t loaded_rgen_magic = rt_alloc_id(mb);
    rt_emit_psb_load_u32(mb, b, loaded_rgen_magic, safe_rgen_hdr, 0u);

    uint32_t c_magic = rt_get_const_u32(mb, RT_GROUP_HANDLE_MAGIC);
    uint32_t is_rgen_magic_ok = rt_alloc_id(mb);
    spv_emit_op4(b, RT_OP_IEQUAL, mb->t_bool, is_rgen_magic_ok, loaded_rgen_magic, c_magic);

    uint32_t final_run_rgen = rt_alloc_id(mb);
    spv_emit_op4(b, RT_OP_LOGICAL_AND, mb->t_bool, final_run_rgen, is_safe_rgen_bda, is_rgen_magic_ok);

    /* 11-08 megafix: the inner selection needs its OWN merge block. Reusing lbl_exit (already the merge of the
     * launch-bounds selection) is invalid structured control flow (spirv-val: "Block is already a merge block for
     * another header", every megakernel dump since 11-07); a structurizer may then run raygen for out-of-bounds
     * threads (LocalSize 8x4: 3 of 4 rows are out of bounds for W3's 131072x1 / 32x1 launches). */
    uint32_t lbl_do_rgen = rt_alloc_id(mb);
    uint32_t lbl_rgen_merge = rt_alloc_id(mb);
    spv_emit_op2(b, RT_OP_SELECTION_MERGE, lbl_rgen_merge, 0u);
    spv_emit_op3(b, RT_OP_BRANCH_CONDITIONAL, final_run_rgen, lbl_do_rgen, lbl_rgen_merge);

    spv_emit_op1(b, RT_OP_LABEL, lbl_do_rgen);
    uint32_t rgen_srb = rt_alloc_id(mb);
    spv_emit_op4(b, RT_OP_IADD, mb->t_u64, rgen_srb, cur_rgen_base, c_32_64);
    spv_emit_op2(b, RT_OP_STORE, mb->g_cur_record_bda, rgen_srb);

    uint32_t rgen_id = rt_alloc_id(mb);
    rt_emit_psb_load_u32(mb, b, rgen_id, cur_rgen_base, 8u);
    rt_emit_stage_switch_inline(mb, b, rgen_id, stages, parses, stage_count,
                                VK_SHADER_STAGE_RAYGEN_BIT_KHR, 0);
    spv_emit_op1(b, RT_OP_BRANCH, lbl_rgen_merge);

    spv_emit_op1(b, RT_OP_LABEL, lbl_rgen_merge);
    spv_emit_op1(b, RT_OP_BRANCH, lbl_exit);

    spv_emit_op1(b, RT_OP_LABEL, lbl_exit);
    spv_emit_op0(b, RT_OP_RETURN);
    spv_emit_op0(b, RT_OP_FUNCTION_END);
}

static void rt_free_stage_parse(struct RtStageParse *sp) {
    if (!sp) return;
    free(sp->remap);
    free(sp->remap_d1);
    free(sp->id_type);
    free(sp->is_psb_type);
    free(sp->is_psb_id);
    free(sp->var_sc);
    free(sp->var_pointee_orig);
    free(sp->dec_builtin);
    free(sp->dec_spec_id);
    free(sp->dec_array_stride);
    free(sp->strip_dec_loc);
    free(sp->const_u32_val);
    free(sp->is_const_u32);
}

static void rt_mega_free(struct RtMegaBuilder *mb) {
    spv_buf_free(&mb->capabilities);
    spv_buf_free(&mb->extensions);
    spv_buf_free(&mb->ext_imports);
    spv_buf_free(&mb->decorations);
    spv_buf_free(&mb->types_consts_globals);
    spv_buf_free(&mb->functions);
    free(mb->type_meta);
}

static uint32_t *rt_build_megakernel_spirv(const struct RtStageRecord *stages,
                                           uint32_t stage_count,
                                           size_t *out_word_count) {
    *out_word_count = 0;
    if (!stages || stage_count == 0) return NULL;

    struct RtMegaBuilder mb;
    rt_mega_init_preamble(&mb);

    struct RtStageParse *parses = (struct RtStageParse *)calloc(stage_count, sizeof(*parses));
    if (!parses) {
        rt_mega_free(&mb);
        return NULL;
    }

    /* 1. Parse and emit module-scope declarations for every unique stage */
    for (uint32_t i = 0; i < stage_count; i++) {
        if (!rt_parse_and_emit_stage_globals(&mb, &stages[i], &parses[i])) {
            for (uint32_t j = 0; j <= i; j++) rt_free_stage_parse(&parses[j]);
            free(parses);
            rt_mega_free(&mb);
            return NULL;
        }
    }

    /* 2. Emit per-stage functions (d0, and d1 for recursive rchit/rmiss stages) */
    for (uint32_t i = 0; i < stage_count; i++) {
        /* Skip duplicate stage_ids so each stage function is emitted once */
        bool dup = false;
        for (uint32_t j = 0; j < i; j++) {
            if (stages[j].stage_id == stages[i].stage_id) {
                parses[i].entry_fn_id_d0 = parses[j].entry_fn_id_d0;
                parses[i].entry_fn_id_d1 = parses[j].entry_fn_id_d1;
                dup = true;
                break;
            }
        }
        if (dup) continue;

        rt_emit_stage_functions(&mb, &stages[i], &parses[i], 0);
        if (parses[i].has_trace_ray &&
            (stages[i].stage == VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR ||
             stages[i].stage == VK_SHADER_STAGE_MISS_BIT_KHR)) {
            rt_emit_stage_functions(&mb, &stages[i], &parses[i], 1);
        }
    }

    /* 3. Emit megakernel helper functions and %main */
    rt_emit_invoke_stage_fn(&mb, mb.fn_invoke_rahit_d0, stages, parses, stage_count,
                            VK_SHADER_STAGE_ANY_HIT_BIT_KHR, 0);
    rt_emit_invoke_stage_fn(&mb, mb.fn_invoke_rahit_d1, stages, parses, stage_count,
                            VK_SHADER_STAGE_ANY_HIT_BIT_KHR, 1);
    rt_emit_invoke_stage_fn(&mb, mb.fn_invoke_rint_d0,  stages, parses, stage_count,
                            VK_SHADER_STAGE_INTERSECTION_BIT_KHR, 0);
    rt_emit_invoke_stage_fn(&mb, mb.fn_invoke_rint_d1,  stages, parses, stage_count,
                            VK_SHADER_STAGE_INTERSECTION_BIT_KHR, 1);
    rt_emit_report_hit_fn(&mb, 0);
    rt_emit_report_hit_fn(&mb, 1);
    rt_emit_invoke_rcall_fn(&mb, stages, parses, stage_count);
    rt_emit_trace_loop_fn(&mb, stages, parses, stage_count, 0);
    rt_emit_trace_loop_fn(&mb, stages, parses, stage_count, 1);
    rt_emit_main_fn(&mb, stages, parses, stage_count);

    if (mb.has_vulkan_memory_model) {
        rt_add_cap(&mb, 5345u /* VulkanMemoryModel */);
        rt_add_ext(&mb, "SPV_KHR_vulkan_memory_model");
    }

    /* 4. Assemble final SPIR-V binary */
    struct SpvBuf final_spv;
    spv_buf_init(&final_spv);

    /* 5-word SPIR-V Header: Magic, Version 1.5 (0x00010500), Generator, Bound, Schema 0 */
    spv_emit_word(&final_spv, RT_SPV_MAGIC);
    spv_emit_word(&final_spv, 0x00010500u);
    spv_emit_word(&final_spv, 0x4D414C49u /* 'MALI' */);
    spv_emit_word(&final_spv, mb.next_id);
    spv_emit_word(&final_spv, 0u);

    spv_emit_words(&final_spv, mb.capabilities.words, mb.capabilities.count);
    spv_emit_words(&final_spv, mb.extensions.words,   mb.extensions.count);
    spv_emit_words(&final_spv, mb.ext_imports.words,  mb.ext_imports.count);

    /* OpMemoryModel PhysicalStorageBuffer64 (5348) GLSL450 (1) or Vulkan (3) */
    spv_emit_op2(&final_spv, RT_OP_MEMORY_MODEL, 5348u, mb.has_vulkan_memory_model ? 3u : 1u);

    /* OpEntryPoint GLCompute %fn_main "main" [iface_vars...] */
    {
        uint32_t ep_wc = 4u + mb.iface_var_count;
        spv_emit_word(&final_spv, (ep_wc << 16) | RT_OP_ENTRY_POINT);
        spv_emit_word(&final_spv, RT_EXEC_MODEL_GLCOMPUTE);
        spv_emit_word(&final_spv, mb.fn_main);
        spv_emit_word(&final_spv, 0x6E69616Du /* "main\0" -> wait: null-terminated requires 2 words! */);
    }
    /* Fix "main" string in OpEntryPoint: "main" is 4 chars + 1 NUL = 5 bytes = 2 words! */
    final_spv.count -= 4;
    {
        uint32_t ep_wc = 5u + mb.iface_var_count;
        spv_emit_word(&final_spv, (ep_wc << 16) | RT_OP_ENTRY_POINT);
        spv_emit_word(&final_spv, RT_EXEC_MODEL_GLCOMPUTE);
        spv_emit_word(&final_spv, mb.fn_main);
        spv_emit_word(&final_spv, 0x6E69616Du /* "main" */);
        spv_emit_word(&final_spv, 0x00000000u /* "\0\0\0\0" */);
        for (uint32_t i = 0; i < mb.iface_var_count; i++) {
            spv_emit_word(&final_spv, mb.iface_vars[i]);
        }
    }

    /* OpExecutionMode %fn_main LocalSize 8 4 1 */
    spv_emit_op5(&final_spv, RT_OP_EXECUTION_MODE, mb.fn_main, 17u /* LocalSize */, 8u, 4u, 1u);

    spv_emit_words(&final_spv, mb.decorations.words,          mb.decorations.count);
    spv_emit_words(&final_spv, mb.types_consts_globals.words, mb.types_consts_globals.count);
    spv_emit_words(&final_spv, mb.functions.words,            mb.functions.count);

    for (uint32_t i = 0; i < stage_count; i++) rt_free_stage_parse(&parses[i]);
    free(parses);
    rt_mega_free(&mb);

    /* Apply standard Mali SPIR-V patcher (storage image format fixups, etc.) */
    size_t patched_size = 0;
    uint32_t *patched = compat_patch_spirv(final_spv.words, final_spv.count * sizeof(uint32_t), &patched_size);
    if (patched && patched_size >= 20) {
        spv_buf_free(&final_spv);
        *out_word_count = patched_size / 4;
        return patched;
    }

    *out_word_count = final_spv.count;
    return final_spv.words;
}

/* ── 1-Thread Compute Kernel for Indirect Ray Dispatch Setup ──────────────── */

static uint32_t *rt_build_indirect_setup_spirv(size_t *out_word_count) {
    struct RtMegaBuilder mb;
    rt_mega_init_preamble(&mb);

    /* PushConstant struct: { uint64_t src_bda; uint64_t dst_bda; } */
    uint32_t t_pc = rt_alloc_id(&mb);
    spv_emit_op3(&mb.types_consts_globals, RT_OP_TYPE_STRUCT, t_pc, mb.t_u64, mb.t_u64);
    spv_emit_op2(&mb.decorations, RT_OP_DECORATE, t_pc, RT_DEC_BLOCK);
    spv_emit_op4(&mb.decorations, RT_OP_MEMBER_DECORATE, t_pc, 0u, RT_DEC_OFFSET, 0u);
    spv_emit_op4(&mb.decorations, RT_OP_MEMBER_DECORATE, t_pc, 1u, RT_DEC_OFFSET, 8u);

    uint32_t t_ptr_pc = rt_find_or_emit_type(&mb, RT_OP_TYPE_POINTER, RT_SC_PUSH_CONSTANT, t_pc, 0);
    uint32_t v_pc = rt_emit_global_var(&mb, t_ptr_pc, RT_SC_PUSH_CONSTANT);
    if (rt_env()->split) rt_iface_drop(&mb, mb.v_rt_pc);   /* split-1112: see rt_iface_drop (behind the split key) */

    struct SpvBuf *b = &mb.functions;
    uint32_t lbl = rt_alloc_id(&mb);
    spv_emit_op4(b, RT_OP_FUNCTION, mb.t_void, mb.fn_main, 0u, mb.t_fn_void);
    spv_emit_op1(b, RT_OP_LABEL, lbl);

    uint32_t c_i0 = rt_get_const_i32(&mb, 0);
    uint32_t c_i1 = rt_get_const_i32(&mb, 1);
    uint32_t ptr_src = rt_alloc_id(&mb), ptr_dst = rt_alloc_id(&mb);
    uint32_t src_bda = rt_alloc_id(&mb), dst_bda = rt_alloc_id(&mb);
    spv_emit_op4(b, RT_OP_ACCESS_CHAIN, mb.t_ptr_pc_u64, ptr_src, v_pc, c_i0);
    spv_emit_op4(b, RT_OP_ACCESS_CHAIN, mb.t_ptr_pc_u64, ptr_dst, v_pc, c_i1);
    spv_emit_op3(b, RT_OP_LOAD, mb.t_u64, src_bda, ptr_src);
    spv_emit_op3(b, RT_OP_LOAD, mb.t_u64, dst_bda, ptr_dst);

    uint32_t w = rt_alloc_id(&mb), h = rt_alloc_id(&mb), d = rt_alloc_id(&mb);
    rt_emit_psb_load_u32(&mb, b, w, src_bda, 0u);
    rt_emit_psb_load_u32(&mb, b, h, src_bda, 4u);
    rt_emit_psb_load_u32(&mb, b, d, src_bda, 8u);

    /* gx = (w + 7) >> 3, gy = (h + 3) >> 2, gz = d */
    uint32_t c_u7 = rt_get_const_u32(&mb, 7u);
    uint32_t c_u3 = rt_get_const_u32(&mb, 3u);
    uint32_t c_u2 = rt_get_const_u32(&mb, 2u);
    uint32_t w7 = rt_alloc_id(&mb), gx = rt_alloc_id(&mb);
    uint32_t h3 = rt_alloc_id(&mb), gy = rt_alloc_id(&mb);
    spv_emit_op4(b, RT_OP_IADD, mb.t_u32, w7, w, c_u7);
    spv_emit_op4(b, 194u /* OpShiftRightLogical */, mb.t_u32, gx, w7, c_u3);
    spv_emit_op4(b, RT_OP_IADD, mb.t_u32, h3, h, c_u3);
    spv_emit_op4(b, 194u /* OpShiftRightLogical */, mb.t_u32, gy, h3, c_u2);

    /* Store gx, gy, d to dst_bda + 0, 4, 8 */
    for (uint32_t k = 0; k < 3; k++) {
        uint32_t val = (k == 0) ? gx : (k == 1 ? gy : d);
        uint32_t addr = dst_bda;
        if (k > 0) {
            uint32_t off_c = rt_get_const_u64(&mb, (uint64_t)(k * 4u));
            addr = rt_alloc_id(&mb);
            spv_emit_op4(b, RT_OP_IADD, mb.t_u64, addr, dst_bda, off_c);
        }
        uint32_t p = rt_alloc_id(&mb);
        spv_emit_op3(b, RT_OP_CONVERT_U_TO_PTR, mb.t_ptr_psb_u32, p, addr);
        spv_emit_op4(b, RT_OP_STORE, p, val, 0x2u /* Aligned */, 4u);
    }

    spv_emit_op0(b, RT_OP_RETURN);
    spv_emit_op0(b, RT_OP_FUNCTION_END);

    struct SpvBuf final_spv;
    spv_buf_init(&final_spv);
    spv_emit_word(&final_spv, RT_SPV_MAGIC);
    spv_emit_word(&final_spv, 0x00010500u);
    spv_emit_word(&final_spv, 0x4D414C49u);
    spv_emit_word(&final_spv, mb.next_id);
    spv_emit_word(&final_spv, 0u);
    spv_emit_words(&final_spv, mb.capabilities.words, mb.capabilities.count);
    spv_emit_words(&final_spv, mb.extensions.words,   mb.extensions.count);
    spv_emit_words(&final_spv, mb.ext_imports.words,  mb.ext_imports.count);
    spv_emit_op2(&final_spv, RT_OP_MEMORY_MODEL, 5348u, 1u);
    {
        uint32_t ep_wc = 5u + mb.iface_var_count;
        spv_emit_word(&final_spv, (ep_wc << 16) | RT_OP_ENTRY_POINT);
        spv_emit_word(&final_spv, RT_EXEC_MODEL_GLCOMPUTE);
        spv_emit_word(&final_spv, mb.fn_main);
        spv_emit_word(&final_spv, 0x6E69616Du);
        spv_emit_word(&final_spv, 0x00000000u);
        for (uint32_t i = 0; i < mb.iface_var_count; i++) {
            spv_emit_word(&final_spv, mb.iface_vars[i]);
        }
    }
    spv_emit_op5(&final_spv, RT_OP_EXECUTION_MODE, mb.fn_main, 17u, 1u, 1u, 1u);
    spv_emit_words(&final_spv, mb.decorations.words,          mb.decorations.count);
    spv_emit_words(&final_spv, mb.types_consts_globals.words, mb.types_consts_globals.count);
    spv_emit_words(&final_spv, mb.functions.words,            mb.functions.count);

    rt_mega_free(&mb);
    *out_word_count = final_spv.count;
    return final_spv.words;
}

/* ── split-1112: 1-thread raygen select kernel for split pipelines ─────────────
 * Push constants (88 B, struct RtSplitSelectPC): reads the raygen record the way the megakernel's main does
 * (direct / Indirect1: rgen base+size from the push constants; Indirect2: from the VkTraceRaysIndirectCommand2KHR)
 * and the launch size (push constants / indirect command), then writes count VkDispatchIndirectCommands at
 * dst + 16*k: {(w+7)>>3, (h+3)>>2, d} for the child whose raygen id equals the record's id, {0,0,0} otherwise.
 * Every load goes through `zero_bda` (the layer's 4 KB zero buffer) when its source is not valid, so the kernel never
 * reads an unchecked address. */
struct RtSplitSelectPC {
    uint64_t rgen_base;   /*  0: direct/Indirect1 raygen record (already sanitised like the megakernel's pc) */
    uint64_t dst_bda;     /*  8: RT_SPLIT_MAX x 16 B of VkDispatchIndirectCommand (ring slot) */
    uint64_t ind_bda;     /* 16: indirect command (0 = direct) */
    uint64_t zero_bda;    /* 24: 4 KB zero buffer */
    uint32_t mode;        /* 32: 0 direct, 1 Indirect1, 2 Indirect2 */
    uint32_t count;       /* 36: children */
    uint32_t w, h, d;     /* 40, 44, 48: direct launch size */
    uint32_t rgen_size;   /* 52: direct/Indirect1 raygen region size */
    uint32_t ids[RT_SPLIT_MAX]; /* 56..87 */
};
static uint32_t *rt_build_split_select_spirv(size_t *out_word_count) {
    struct RtMegaBuilder mb;
    rt_mega_init_preamble(&mb);
    struct SpvBuf *b = &mb.functions;
    enum { NM = 10 + RT_SPLIT_MAX };
    uint32_t t_pc = rt_alloc_id(&mb);
    spv_emit_word(&mb.types_consts_globals, ((2u + NM) << 16) | RT_OP_TYPE_STRUCT);
    spv_emit_word(&mb.types_consts_globals, t_pc);
    for (uint32_t m = 0; m < NM; m++) spv_emit_word(&mb.types_consts_globals, m < 4 ? mb.t_u64 : mb.t_u32);
    spv_emit_op2(&mb.decorations, RT_OP_DECORATE, t_pc, RT_DEC_BLOCK);
    for (uint32_t m = 0; m < NM; m++)
        spv_emit_op4(&mb.decorations, RT_OP_MEMBER_DECORATE, t_pc, m, RT_DEC_OFFSET, m < 4 ? m * 8u : 32u + (m - 4u) * 4u);
    uint32_t t_ptr_pc = rt_find_or_emit_type(&mb, RT_OP_TYPE_POINTER, RT_SC_PUSH_CONSTANT, t_pc, 0);
    uint32_t v_pc = rt_emit_global_var(&mb, t_ptr_pc, RT_SC_PUSH_CONSTANT);
    rt_iface_drop(&mb, mb.v_rt_pc);   /* one PushConstant variable per interface */

    uint32_t lbl = rt_alloc_id(&mb);
    spv_emit_op4(b, RT_OP_FUNCTION, mb.t_void, mb.fn_main, 0u, mb.t_fn_void);
    spv_emit_op1(b, RT_OP_LABEL, lbl);
    uint32_t pcm[NM];
    for (uint32_t m = 0; m < NM; m++) {
        uint32_t ptr = rt_alloc_id(&mb);
        pcm[m] = rt_alloc_id(&mb);
        spv_emit_op4(b, RT_OP_ACCESS_CHAIN, m < 4 ? mb.t_ptr_pc_u64 : mb.t_ptr_pc_u32, ptr, v_pc,
                     rt_get_const_i32(&mb, (int32_t)m));
        spv_emit_op3(b, RT_OP_LOAD, m < 4 ? mb.t_u64 : mb.t_u32, pcm[m], ptr);
    }
    const uint32_t p_rgen = pcm[0], p_dst = pcm[1], p_ind = pcm[2], p_zero = pcm[3], p_mode = pcm[4], p_cnt = pcm[5];
    const uint32_t p_w = pcm[6], p_h = pcm[7], p_d = pcm[8], p_rsz = pcm[9];
    #define SEL(T, c, x, y) ({ uint32_t r_ = rt_alloc_id(&mb); spv_emit_op5(b, RT_OP_SELECT, (T), r_, (c), (x), (y)); r_; })
    #define BIN(op, T, x, y) ({ uint32_t r_ = rt_alloc_id(&mb); spv_emit_op4(b, (op), (T), r_, (x), (y)); r_; })
    #define LD32(addr, off) ({ uint32_t r_ = rt_alloc_id(&mb); rt_emit_psb_load_u32(&mb, b, r_, (addr), (off)); r_; })
    #define LD64(addr, off) ({ uint32_t r_ = rt_alloc_id(&mb); rt_emit_psb_load_u64(&mb, b, r_, (addr), (off)); r_; })
    uint32_t c_z64 = rt_get_const_u64(&mb, 0ULL);
    uint32_t is_m1 = BIN(RT_OP_IEQUAL, mb.t_bool, p_mode, mb.c_u32_1);
    uint32_t is_m2 = BIN(RT_OP_IEQUAL, mb.t_bool, p_mode, mb.c_u32_2);
    uint32_t ind_ok = BIN(RT_OP_INOT_EQUAL, mb.t_bool, p_ind, c_z64);
    /* each command is read only through its own address: Indirect1 is 12 B (an Indirect2 read there could run past
     * the app's buffer), Indirect2 104 B; the other one reads the zero buffer */
    uint32_t ind1 = SEL(mb.t_u64, BIN(RT_OP_LOGICAL_AND, mb.t_bool, is_m1, ind_ok), p_ind, p_zero);
    uint32_t ind2 = SEL(mb.t_u64, BIN(RT_OP_LOGICAL_AND, mb.t_bool, is_m2, ind_ok), p_ind, p_zero);
    /* Indirect2: raygen base/size at 0/8, w/h/d at 88/92/96; Indirect1: w/h/d at 0/4/8 */
    /* 64-bit fields as two 32-bit loads: the command address is only required to be 4-byte aligned */
    #define LD64_4(addr, off) ({ uint32_t lo_ = LD32(addr, off), hi_ = LD32(addr, (off) + 4u); \
        uint32_t l64_ = rt_alloc_id(&mb), h64_ = rt_alloc_id(&mb), sh_ = rt_alloc_id(&mb), r_ = rt_alloc_id(&mb); \
        spv_emit_op3(b, RT_OP_UCONVERT, mb.t_u64, l64_, lo_); spv_emit_op3(b, RT_OP_UCONVERT, mb.t_u64, h64_, hi_); \
        spv_emit_op4(b, 196u /* OpShiftLeftLogical */, mb.t_u64, sh_, h64_, rt_get_const_u64(&mb, 32ULL)); \
        spv_emit_op4(b, 197u /* OpBitwiseOr */, mb.t_u64, r_, sh_, l64_); r_; })
    uint32_t i2_rb = LD64_4(ind2, 0u), i2_rs = LD64_4(ind2, 8u);
    uint32_t i2_w = LD32(ind2, 88u), i2_h = LD32(ind2, 92u), i2_d = LD32(ind2, 96u);
    uint32_t i1_w = LD32(ind1, 0u), i1_h = LD32(ind1, 4u), i1_d = LD32(ind1, 8u);
    uint32_t rsz64 = rt_alloc_id(&mb);
    spv_emit_op3(b, RT_OP_UCONVERT, mb.t_u64, rsz64, p_rsz);
    uint32_t rbase = SEL(mb.t_u64, is_m2, i2_rb, p_rgen);
    uint32_t rsize = SEL(mb.t_u64, is_m2, i2_rs, rsz64);
    uint32_t w = SEL(mb.t_u32, is_m2, i2_w, SEL(mb.t_u32, is_m1, i1_w, p_w));
    uint32_t h = SEL(mb.t_u32, is_m2, i2_h, SEL(mb.t_u32, is_m1, i1_h, p_h));
    uint32_t dd = SEL(mb.t_u32, is_m2, i2_d, SEL(mb.t_u32, is_m1, i1_d, p_d));
    /* the megakernel main's raygen checks: base != 0, size >= 32, BDA window, magic */
    uint32_t c_32_64 = rt_get_const_u64(&mb, 32ULL);
    uint32_t ok1 = BIN(RT_OP_INOT_EQUAL, mb.t_bool, rbase, c_z64);
    uint32_t ok2 = BIN(RT_OP_UGREATER_THAN_EQUAL, mb.t_bool, rsize, c_32_64);
    uint32_t ok3 = BIN(RT_OP_UGREATER_THAN_EQUAL, mb.t_bool, rbase, rt_get_const_u64(&mb, RT_MIN_VALID_GPU_BDA));
    uint32_t ok4 = BIN(RT_OP_ULESS_THAN, mb.t_bool, rbase, rt_get_const_u64(&mb, RT_MAX_VALID_GPU_BDA));
    uint32_t okr = BIN(RT_OP_LOGICAL_AND, mb.t_bool, BIN(RT_OP_LOGICAL_AND, mb.t_bool, ok1, ok2),
                       BIN(RT_OP_LOGICAL_AND, mb.t_bool, ok3, ok4));
    uint32_t hdr = SEL(mb.t_u64, okr, rbase, p_zero);
    uint32_t magic = LD32(hdr, 0u), rid = LD32(hdr, 8u);
    uint32_t okm = BIN(RT_OP_IEQUAL, mb.t_bool, magic, rt_get_const_u32(&mb, RT_GROUP_HANDLE_MAGIC));
    uint32_t run = BIN(RT_OP_LOGICAL_AND, mb.t_bool, okr, okm);
    uint32_t c_u7 = rt_get_const_u32(&mb, 7u), c_u3 = rt_get_const_u32(&mb, 3u);
    uint32_t gx = BIN(194u /* OpShiftRightLogical */, mb.t_u32, BIN(RT_OP_IADD, mb.t_u32, w, c_u7), c_u3);
    uint32_t gy = BIN(194u, mb.t_u32, BIN(RT_OP_IADD, mb.t_u32, h, c_u3), mb.c_u32_2);
    for (uint32_t k = 0; k < RT_SPLIT_MAX; k++) {
        uint32_t in_n = BIN(RT_OP_ULESS_THAN, mb.t_bool, rt_get_const_u32(&mb, k), p_cnt);
        uint32_t idm = BIN(RT_OP_IEQUAL, mb.t_bool, rid, pcm[10 + k]);
        uint32_t m = BIN(RT_OP_LOGICAL_AND, mb.t_bool, run, BIN(RT_OP_LOGICAL_AND, mb.t_bool, in_n, idm));
        uint32_t vals[3] = { SEL(mb.t_u32, m, gx, mb.c_u32_0), SEL(mb.t_u32, m, gy, mb.c_u32_0),
                             SEL(mb.t_u32, m, dd, mb.c_u32_0) };
        for (uint32_t q = 0; q < 3; q++) {
            uint32_t addr = BIN(RT_OP_IADD, mb.t_u64, p_dst, rt_get_const_u64(&mb, (uint64_t)(k * 16u + q * 4u)));
            uint32_t ptr = rt_alloc_id(&mb);
            spv_emit_op3(b, RT_OP_CONVERT_U_TO_PTR, mb.t_ptr_psb_u32, ptr, addr);
            spv_emit_op4(b, RT_OP_STORE, ptr, vals[q], 0x2u /* Aligned */, 4u);
        }
    }
    #undef SEL
    #undef BIN
    #undef LD32
    #undef LD64
    #undef LD64_4
    spv_emit_op0(b, RT_OP_RETURN);
    spv_emit_op0(b, RT_OP_FUNCTION_END);

    struct SpvBuf final_spv;
    spv_buf_init(&final_spv);
    spv_emit_word(&final_spv, RT_SPV_MAGIC);
    spv_emit_word(&final_spv, 0x00010500u);
    spv_emit_word(&final_spv, 0x4D414C49u);
    spv_emit_word(&final_spv, mb.next_id);
    spv_emit_word(&final_spv, 0u);
    spv_emit_words(&final_spv, mb.capabilities.words, mb.capabilities.count);
    spv_emit_words(&final_spv, mb.extensions.words,   mb.extensions.count);
    spv_emit_words(&final_spv, mb.ext_imports.words,  mb.ext_imports.count);
    spv_emit_op2(&final_spv, RT_OP_MEMORY_MODEL, 5348u, 1u);
    {
        uint32_t ep_wc = 5u + mb.iface_var_count;
        spv_emit_word(&final_spv, (ep_wc << 16) | RT_OP_ENTRY_POINT);
        spv_emit_word(&final_spv, RT_EXEC_MODEL_GLCOMPUTE);
        spv_emit_word(&final_spv, mb.fn_main);
        spv_emit_word(&final_spv, 0x6E69616Du);
        spv_emit_word(&final_spv, 0x00000000u);
        for (uint32_t i = 0; i < mb.iface_var_count; i++) spv_emit_word(&final_spv, mb.iface_vars[i]);
    }
    spv_emit_op5(&final_spv, RT_OP_EXECUTION_MODE, mb.fn_main, 17u, 1u, 1u, 1u);
    spv_emit_words(&final_spv, mb.decorations.words,          mb.decorations.count);
    spv_emit_words(&final_spv, mb.types_consts_globals.words, mb.types_consts_globals.count);
    spv_emit_words(&final_spv, mb.functions.words,            mb.functions.count);
    rt_mega_free(&mb);
    *out_word_count = final_spv.count;
    return final_spv.words;
}


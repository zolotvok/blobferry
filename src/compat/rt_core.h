// SPDX-License-Identifier: GPL-3.0-or-later
// rt_core.h: address checks, rt_env, hashing, stage records, pins, compile worker, cmd state. Included by mali_rt_emu.h.
/* Forward declarations */
static void rt_patch_fex_thunks(void);

/* ── Valid Mali GPU Device Address Range & Safety Checks ─────────────────── */
/* In Mali kbase CSF on 64-bit Linux/Android, all /mali_gmc GPU device allocations
 * reside strictly in [0x4000000000ULL, 0x6000000000ULL) (256 GB to 384 GB, BASE_MEM_FIRST_FREE_ADDRESS).
 * Addresses < 0x4000000000ULL are NULL/near-null pointers or CPU 32-bit/guest memory.
 * Addresses >= 0x6000000000ULL are in CPU user space / [anon:FEXAllocator] PROT_NONE pages.
 * Any GPU MMU dereference outside this range unconditionally crashes the GPU with
 * a Level 1 translation fault (0x1002c1 / 0x55002c1). */
/* 11-08 keeper: the floor goes back to 0x10000 (01f42d9 value). kbase fills its VA space downward, so real
 * /mali_gmc heaps DO land below 256 GB (W3 megafix L1: all BLAS at 0x3a.., 440 AS addresses clamped to the
 * 4 KB zero buffer, 99.6 % of TLAS instances masked, every app scratch replaced). */
#define RT_MIN_VALID_GPU_BDA                         0x10000ULL
#define RT_MAX_VALID_GPU_BDA                         0x6000000000ULL

static inline bool rt_is_invalid_gpu_bda(VkDeviceAddress addr) {
    return (addr != 0ULL) && (addr < RT_MIN_VALID_GPU_BDA || addr >= RT_MAX_VALID_GPU_BDA);
}

static int g_pipe_check_fds[2] = {-1, -1};
static pthread_mutex_t g_pipe_check_lock = PTHREAD_MUTEX_INITIALIZER;

/* Safely tests if a host virtual address is mapped and readable in CPU memory
 * without risking a SIGSEGV fault by using the Linux kernel copy_from_user check in pipe(). */
static inline bool rt_is_host_address_accessible(const void *addr, size_t size) {
    if (!addr || size == 0) return false;
    pthread_mutex_lock(&g_pipe_check_lock);
    if (g_pipe_check_fds[0] < 0) {
        if (pipe(g_pipe_check_fds) < 0) {
            pthread_mutex_unlock(&g_pipe_check_lock);
            return false;
        }
        fcntl(g_pipe_check_fds[0], F_SETFL, O_NONBLOCK);
        fcntl(g_pipe_check_fds[1], F_SETFL, O_NONBLOCK);
    }
    ssize_t ret = write(g_pipe_check_fds[1], addr, size);
    if (ret > 0) {
        char drain[256];
        while (read(g_pipe_check_fds[0], drain, sizeof(drain)) > 0);
        pthread_mutex_unlock(&g_pipe_check_lock);
        return true;
    }
    pthread_mutex_unlock(&g_pipe_check_lock);
    return false;
}

/* Resolves a GPU BDA to a valid, host-accessible CPU pointer.
 * Works for both direct SAME_VA mappings and FEX/Wine placed mappings (where memory was moved to e->placed). */
static inline void *rt_resolve_gpu_bda_to_host(VkDeviceAddress bda, size_t size) {
    if (rt_is_invalid_gpu_bda(bda) || size == 0) return NULL;
    /* 1. Direct SAME_VA check */
    if (rt_is_host_address_accessible((const void *)bda, size)) {
        return (void *)bda;
    }
    /* 2. Placed mappings lookup */
    pthread_mutex_lock(&g_mem_mutex);
    for (uint32_t b = 0; b < MEM_BUCKETS; b++) {
        for (struct MemInfo *e = g_mem[b]; e; e = e->next) {
            if (e->placed && e->orig && e->map_size >= size) {
                uintptr_t orig_start = (uintptr_t)e->orig;
                uintptr_t orig_end   = orig_start + e->map_size;
                if ((uintptr_t)bda >= orig_start && ((uintptr_t)bda + size) <= orig_end) {
                    uintptr_t offset = (uintptr_t)bda - orig_start;
                    void *host_ptr = (char *)e->placed + offset;
                    pthread_mutex_unlock(&g_mem_mutex);
                    if (rt_is_host_address_accessible(host_ptr, size)) {
                        return host_ptr;
                    }
                    return NULL;
                }
            }
        }
    }
    pthread_mutex_unlock(&g_mem_mutex);
    return NULL;
}

/* ── SPIR-V Constants for RT Lowering ─────────────────────────────────────── */
#define RT_SPV_MAGIC                                 0x07230203u

/* Execution Models */
#define RT_EXEC_MODEL_GLCOMPUTE                      5u
#define RT_EXEC_MODEL_RAYGEN_KHR                     5313u
#define RT_EXEC_MODEL_INTERSECTION_KHR               5314u
#define RT_EXEC_MODEL_ANY_HIT_KHR                    5315u
#define RT_EXEC_MODEL_CLOSEST_HIT_KHR                5316u
#define RT_EXEC_MODEL_MISS_KHR                       5317u
#define RT_EXEC_MODEL_CALLABLE_KHR                   5318u

/* Storage Classes */
#define RT_SC_UNIFORM_CONSTANT                       0u
#define RT_SC_INPUT                                  1u
#define RT_SC_UNIFORM                                2u
#define RT_SC_OUTPUT                                 3u
#define RT_SC_WORKGROUP                              4u
#define RT_SC_PRIVATE                                6u
#define RT_SC_FUNCTION                               7u
#define RT_SC_PUSH_CONSTANT                          9u
#define RT_SC_STORAGE_BUFFER                         12u
#define RT_SC_CALLABLE_DATA_KHR                      5328u
#define RT_SC_INCOMING_CALLABLE_DATA_KHR             5329u
#define RT_SC_RAY_PAYLOAD_KHR                        5338u
#define RT_SC_HIT_ATTRIBUTE_KHR                      5339u
#define RT_SC_INCOMING_RAY_PAYLOAD_KHR               5342u
#define RT_SC_SHADER_RECORD_BUFFER_KHR               5343u
#define RT_SC_PHYSICAL_STORAGE_BUFFER                5349u

/* Capabilities */
#define RT_CAP_SHADER                                1u
#define RT_CAP_INT64                                 11u
#define RT_CAP_RAY_QUERY_PROVISIONAL_KHR             5339u
#define RT_CAP_RAY_TRACING_PROVISIONAL_KHR           5353u
#define RT_CAP_RAY_QUERY_KHR                         4472u
#define RT_CAP_RAY_TRAVERSAL_PRIMITIVE_CULLING_KHR   4478u
#define RT_CAP_RAY_TRACING_KHR                       4479u
#define RT_CAP_RAY_TRACING_NV                        5337u
#define RT_CAP_RAY_TRACING_MOTION_BLUR_NV            5340u
#define RT_CAP_PHYSICAL_STORAGE_BUFFER_ADDRESSES     5347u

/* Decorations */
#define RT_DEC_SPEC_ID                               1u
#define RT_DEC_BLOCK                                 2u
#define RT_DEC_BUFFER_BLOCK                          3u
#define RT_DEC_ARRAY_STRIDE                          6u
#define RT_DEC_BUILTIN                               11u
#define RT_DEC_LOCATION                              30u
#define RT_DEC_DESCRIPTOR_SET                        34u
#define RT_DEC_BINDING                               33u
#define RT_DEC_OFFSET                                35u

/* BuiltIns */
#define RT_BUILTIN_INSTANCE_ID                       6u
#define RT_BUILTIN_PRIMITIVE_ID                      7u
#define RT_BUILTIN_GLOBAL_INVOCATION_ID              28u
#define RT_BUILTIN_INSTANCE_INDEX                    43u
#define RT_BUILTIN_LAUNCH_ID_KHR                     5319u
#define RT_BUILTIN_LAUNCH_SIZE_KHR                   5320u
#define RT_BUILTIN_WORLD_RAY_ORIGIN_KHR              5321u
#define RT_BUILTIN_WORLD_RAY_DIRECTION_KHR           5322u
#define RT_BUILTIN_OBJECT_RAY_ORIGIN_KHR             5323u
#define RT_BUILTIN_OBJECT_RAY_DIRECTION_KHR          5324u
#define RT_BUILTIN_RAY_TMIN_KHR                      5325u
#define RT_BUILTIN_RAY_TMAX_KHR                      5326u
#define RT_BUILTIN_INSTANCE_CUSTOM_INDEX_KHR         5327u
#define RT_BUILTIN_OBJECT_TO_WORLD_KHR               5330u
#define RT_BUILTIN_WORLD_TO_OBJECT_KHR               5331u
#define RT_BUILTIN_HIT_T_NV                          5332u
#define RT_BUILTIN_HIT_KIND_KHR                      5333u
#define RT_BUILTIN_INCOMING_RAY_FLAGS_KHR            5351u
#define RT_BUILTIN_RAY_GEOMETRY_INDEX_KHR            5352u
#define RT_BUILTIN_CULL_MASK_KHR                     6021u

/* Opcodes */
#define RT_OP_NOP                                    0u
#define RT_OP_SOURCE_CONTINUED                       2u
#define RT_OP_SOURCE                                 3u
#define RT_OP_SOURCE_EXTENSION                       4u
#define RT_OP_NAME                                   5u
#define RT_OP_MEMBER_NAME                            6u
#define RT_OP_STRING                                 7u
#define RT_OP_LINE                                   8u
#define RT_OP_EXTENSION                              10u
#define RT_OP_EXT_INST_IMPORT                        11u
#define RT_OP_EXT_INST                               12u
#define RT_OP_MEMORY_MODEL                           14u
#define RT_OP_ENTRY_POINT                            15u
#define RT_OP_EXECUTION_MODE                         16u
#define RT_OP_CAPABILITY                             17u
#define RT_OP_TYPE_VOID                              19u
#define RT_OP_TYPE_BOOL                              20u
#define RT_OP_TYPE_INT                               21u
#define RT_OP_TYPE_FLOAT                             22u
#define RT_OP_TYPE_VECTOR                            23u
#define RT_OP_TYPE_MATRIX                            24u
#define RT_OP_TYPE_IMAGE                             25u
#define RT_OP_TYPE_SAMPLER                           26u
#define RT_OP_TYPE_SAMPLED_IMAGE                     27u
#define RT_OP_TYPE_ARRAY                             28u
#define RT_OP_TYPE_RUNTIME_ARRAY                     29u
#define RT_OP_TYPE_STRUCT                            30u
#define RT_OP_TYPE_POINTER                           32u
#define RT_OP_TYPE_FUNCTION                          33u
#define RT_OP_TYPE_FORWARD_POINTER                   39u
#define RT_OP_CONSTANT_TRUE                          41u
#define RT_OP_CONSTANT_FALSE                         42u
#define RT_OP_CONSTANT                               43u
#define RT_OP_CONSTANT_COMPOSITE                     44u
#define RT_OP_CONSTANT_NULL                          46u
#define RT_OP_SPEC_CONSTANT_TRUE                     48u
#define RT_OP_SPEC_CONSTANT_FALSE                    49u
#define RT_OP_SPEC_CONSTANT                          50u
#define RT_OP_SPEC_CONSTANT_COMPOSITE                51u
#define RT_OP_SPEC_CONSTANT_OP                       52u
#define RT_OP_FUNCTION                               54u
#define RT_OP_FUNCTION_PARAMETER                     55u
#define RT_OP_FUNCTION_END                           56u
#define RT_OP_FUNCTION_CALL                          57u
#define RT_OP_VARIABLE                               59u
#define RT_OP_LOAD                                   61u
#define RT_OP_STORE                                  62u
#define RT_OP_ACCESS_CHAIN                           65u
#define RT_OP_IN_BOUNDS_ACCESS_CHAIN                 66u
#define RT_OP_DECORATE                               71u
#define RT_OP_MEMBER_DECORATE                        72u
#define RT_OP_VECTOR_EXTRACT_DYNAMIC                 77u
#define RT_OP_VECTOR_INSERT_DYNAMIC                  78u
#define RT_OP_VECTOR_SHUFFLE                         79u
#define RT_OP_COMPOSITE_CONSTRUCT                    80u
#define RT_OP_COMPOSITE_EXTRACT                      81u
#define RT_OP_COMPOSITE_INSERT                       82u
#define RT_OP_COPY_OBJECT                            83u
#define RT_OP_TRANSPOSE                              84u
#define RT_OP_UCONVERT                               113u
#define RT_OP_SCONVERT                               114u
#define RT_OP_CONVERT_PTR_TO_U                       117u
#define RT_OP_CONVERT_U_TO_PTR                       120u
#define RT_OP_BITCAST                                124u
#define RT_OP_IADD                                   128u
#define RT_OP_ISUB                                   130u
#define RT_OP_IMUL                                   132u
#define RT_OP_LOGICAL_EQUAL                          164u
#define RT_OP_LOGICAL_NOT_EQUAL                      165u
#define RT_OP_LOGICAL_OR                             166u
#define RT_OP_LOGICAL_AND                            167u
#define RT_OP_LOGICAL_NOT                            168u
#define RT_OP_SELECT                                 169u
#define RT_OP_IEQUAL                                 170u
#define RT_OP_INOT_EQUAL                             171u
#define RT_OP_UGREATER_THAN                          172u
#define RT_OP_UGREATER_THAN_EQUAL                    174u
#define RT_OP_ULESS_THAN                             176u
#define RT_OP_ULESS_THAN_EQUAL                       178u
#define RT_OP_FORD_LESS_THAN_EQUAL                   186u
#define RT_OP_FORD_GREATER_THAN_EQUAL                188u
#define RT_OP_BITWISE_OR                             197u
#define RT_OP_BITWISE_AND                            199u
#define RT_OP_PHI                                    245u
#define RT_OP_LOOP_MERGE                             246u
#define RT_OP_SELECTION_MERGE                        247u
#define RT_OP_LABEL                                  248u
#define RT_OP_BRANCH                                 249u
#define RT_OP_BRANCH_CONDITIONAL                     250u
#define RT_OP_SWITCH                                 251u
#define RT_OP_RETURN                                 253u
#define RT_OP_RETURN_VALUE                           254u
#define RT_OP_UNREACHABLE                            255u
#define RT_OP_NO_LINE                                317u
#define RT_OP_MODULE_PROCESSED                       330u

/* RT & RayQuery Opcodes */
#define RT_OP_TRACE_RAY_KHR                          4445u
#define RT_OP_EXECUTE_CALLABLE_KHR                   4446u
#define RT_OP_CONVERT_U_TO_ACCEL_STRUCT_KHR          4447u
#define RT_OP_IGNORE_INTERSECTION_KHR                4448u
#define RT_OP_TERMINATE_RAY_KHR                      4449u
#define RT_OP_TYPE_RAY_QUERY_KHR                     4472u
#define RT_OP_RAY_QUERY_INITIALIZE_KHR               4473u
#define RT_OP_RAY_QUERY_TERMINATE_KHR                4474u
#define RT_OP_RAY_QUERY_GENERATE_INTERSECTION_KHR    4475u
#define RT_OP_RAY_QUERY_CONFIRM_INTERSECTION_KHR     4476u
#define RT_OP_RAY_QUERY_PROCEED_KHR                  4477u
#define RT_OP_RAY_QUERY_GET_INTERSECTION_TYPE_KHR    4479u
#define RT_OP_REPORT_INTERSECTION_KHR                5334u
#define RT_OP_IGNORE_INTERSECTION_NV                 5335u
#define RT_OP_TERMINATE_RAY_NV                       5336u
#define RT_OP_TYPE_ACCEL_STRUCT_KHR                  5341u
#define RT_OP_EXECUTE_CALLABLE_NV                    5344u
#define RT_OP_RQ_GET_RAY_TMIN_KHR                    6016u
#define RT_OP_RQ_GET_RAY_FLAGS_KHR                   6017u
#define RT_OP_RQ_GET_INTERSECTION_T_KHR              6018u
#define RT_OP_RQ_GET_INTERSECTION_CUSTOM_INDEX_KHR   6019u
#define RT_OP_RQ_GET_INTERSECTION_INSTANCE_ID_KHR    6020u
#define RT_OP_RQ_GET_INTERSECTION_SBT_OFFSET_KHR     6021u
#define RT_OP_RQ_GET_INTERSECTION_GEOM_INDEX_KHR     6022u
#define RT_OP_RQ_GET_INTERSECTION_PRIM_INDEX_KHR     6023u
#define RT_OP_RQ_GET_INTERSECTION_BARYCENTRICS_KHR   6024u
#define RT_OP_RQ_GET_INTERSECTION_FRONT_FACE_KHR     6025u
#define RT_OP_RQ_GET_CANDIDATE_AABB_OPAQUE_KHR       6026u
#define RT_OP_RQ_GET_INTERSECTION_OBJ_RAY_DIR_KHR    6027u
#define RT_OP_RQ_GET_INTERSECTION_OBJ_RAY_ORIG_KHR   6028u
#define RT_OP_RQ_GET_WORLD_RAY_DIR_KHR               6029u
#define RT_OP_RQ_GET_WORLD_RAY_ORIG_KHR              6030u
#define RT_OP_RQ_GET_INTERSECTION_OBJ_TO_WORLD_KHR   6031u
#define RT_OP_RQ_GET_INTERSECTION_WORLD_TO_OBJ_KHR   6032u

/* Shader Stage Masks */
#define RT_SHADER_STAGE_MASK ( \
    VK_SHADER_STAGE_RAYGEN_BIT_KHR | \
    VK_SHADER_STAGE_ANY_HIT_BIT_KHR | \
    VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | \
    VK_SHADER_STAGE_MISS_BIT_KHR | \
    VK_SHADER_STAGE_INTERSECTION_BIT_KHR | \
    VK_SHADER_STAGE_CALLABLE_BIT_KHR)

/* Magic Constants */
#define RT_GROUP_HANDLE_MAGIC 0x52544748u /* 'RTGH' */
#define RT_GROUP_HANDLE_SIZE  32u

#define RT_PAYLOAD_MAX_DWORDS  64u
#define RT_HITATTR_MAX_DWORDS  16u
#define RT_CALLABLE_MAX_DWORDS 32u

/* ── 32-Byte Position-Independent SBT Group Handle ────────────────────────── */
struct RtGroupHandle {
    uint32_t magic;       /* offset 0:  RT_GROUP_HANDLE_MAGIC */
    uint32_t type;        /* offset 4:  VkRayTracingShaderGroupTypeKHR */
    uint32_t general_id;  /* offset 8:  stage_id of rgen / rmiss / rcall (0 = none) */
    uint32_t rchit_id;    /* offset 12: stage_id of rchit (0 = none) */
    uint32_t rahit_id;    /* offset 16: stage_id of rahit (0 = none) */
    uint32_t rint_id;     /* offset 20: stage_id of rint  (0 = none) */
    uint32_t reserved[2]; /* offset 24..31: 0 */
};

/* ── 96-Byte Megakernel Push Constant Block ───────────────────────────────── */
struct RtDispatchPushConstants {
    uint64_t rgen_base;        /* offset 0 */
    uint32_t rgen_stride;      /* offset 8 */
    uint32_t rgen_size;        /* offset 12 */
    uint64_t miss_base;        /* offset 16 */
    uint32_t miss_stride;      /* offset 24 */
    uint32_t miss_size;        /* offset 28 */
    uint64_t hit_base;         /* offset 32 */
    uint32_t hit_stride;       /* offset 40 */
    uint32_t hit_size;         /* offset 44 */
    uint64_t call_base;        /* offset 48 */
    uint32_t call_stride;      /* offset 56 */
    uint32_t call_size;        /* offset 60 */
    uint32_t launch_width;     /* offset 64 */
    uint32_t launch_height;    /* offset 68 */
    uint32_t launch_depth;     /* offset 72 */
    uint32_t indirect_mode;    /* offset 76: 0 = direct, 1 = Indirect1, 2 = Indirect2 */
    uint64_t push_consts_bda;  /* offset 80: BDA of 256-byte user push constants */
    uint64_t indirect_bda;     /* offset 88: BDA of indirect trace command */
};

/* Traversal safety bound (param-fix 11-08): a ray whose ray-query loop runs more than g_rt_guard_max
 * OpRayQueryProceedKHR iterations is terminated and treated as a miss, so a bad SBT/TLAS/param value can
 * never keep a GPU job alive until the watchdog kills it.  Counter = one u32 in the last ring slot
 * (GPU atomic), logged once per second with MALI_COMPAT_DEBUG.  MALI_COMPAT_RT_GUARD_ITERS overrides. */
#define RT_GUARD_SLOT_OFF ((4u * 1024u * 1024u) - 256u)
static VkDeviceAddress g_rt_guard_bda = 0ULL;
static uint32_t        g_rt_guard_max = 0u;
struct DeviceData;
static bool rt_ensure_ring_buffer(struct DeviceData *d);

/* ── Intercepted RT Shader Module & Pipeline Tables ───────────────────────── */
struct RtShaderModule {
    uint32_t               magic;
    VkDevice               device;
    uint32_t              *words;
    size_t                 word_count;
    uint64_t               hash;
    struct RtShaderModule *next;
};

struct RtSpecConstantEntry {
    uint32_t spec_id;
    uint32_t offset;
    uint32_t size;
};

struct RtStageRecord {
    VkShaderStageFlagBits      stage;
    uint32_t                   stage_id; /* 32-bit deterministic non-zero FNV-1a */
    uint32_t                  *words;
    size_t                     word_count;
    char                      *entry_name;
    struct RtSpecConstantEntry *spec_entries;
    uint32_t                   spec_entry_count;
    uint8_t                   *spec_data;
    size_t                     spec_data_size;
    size_t                     acct_bytes;  /* round 4: bytes counted in g_acct_stages (0 = not counted) */
};

struct RtPipeline {
    uint32_t               magic;
    VkDevice               device;
    bool                   is_library;
    bool                   use_descriptor_buffer;
    VkPipelineLayout       layout;
    VkPipeline             compute_pipeline;
    VkShaderModule         compute_module;
    struct RtStageRecord  *stages;
    uint32_t               stage_count;
    struct RtGroupHandle  *groups;
    uint32_t               group_count;
    uint32_t               max_recursion_depth;
    /* megakernel compile state (11-08 megafix): 0 none/sync, 1 pending on a worker, 2 ready, 3 failed, 4 capped */
    int                    compile_state;
    bool                   compile_thread_live;
    bool                   compile_null_alloc;  /* async compile used a NULL allocator */
    bool                   layout_pinned;       /* 1110b: `layout` pinned for this object's life (rt_pin_pipeline_layout) */
    pthread_t              compile_thread;
    _Atomic int            compile_handshake;   /* 11-08 keeper: worker vs DestroyPipeline, second one cleans up */
    size_t                 mega_bytes;          /* round 4: SPIR-V bytes of compute_module (counted in g_acct_mega_mod) */
    bool                   module_app_alloc;    /* round 4: compute_module was created with the app's allocator */
    size_t                 acct_bytes;          /* round 4: bytes counted in g_acct_pipes (object + group table) */
    /* split-1112 (MALI_COMPAT_RT_SPLIT=1): one megakernel per raygen stage instead of one for all raygens. The parent
     * (the app's VkPipeline) then has no compute pipeline of its own; split[k] is a hidden child RtPipeline (not in
     * g_rt_pipelines, no stage copies, no layout pin) whose megakernel holds raygen split_rgen[k] + every non-raygen
     * stage. A trace runs a 1-thread select kernel that reads the raygen record's stage id on the GPU and writes one
     * VkDispatchIndirectCommand per child (0x0x0 for all but the matching one). */
#define RT_SPLIT_MAX 8
#define RT_PC_LOCAL1D 0x100u   /* local1d-1112: flag bit in RtDispatchPushConstants.indirect_mode (direct traces only) */
    struct RtPipeline     *split[RT_SPLIT_MAX];
    uint32_t               split_rgen[RT_SPLIT_MAX];
    uint32_t               split_n;
    uint32_t               mega_stages;         /* split-1112: stages in this object's megakernel (child: the subset) */
    bool                   split_child;         /* split-1112: a hidden child (its async compile takes a split-job slot) */
    struct RtPipeline     *next;
};

struct RtDeferredOp {
    uint32_t             magic;
    VkDevice             device;
    VkResult             result;
    struct RtDeferredOp *next;
};

struct RtCmdState {
    VkCommandBuffer        cmd;
    struct RtPipeline     *bound_rt_pipeline;
    VkPipeline             bound_comp_pipeline;
    VkPipelineLayout       rt_layout;
    uint8_t                user_push_constants[256];
    uint32_t               user_pc_dirty_size;
    /* Descriptor buffer state bound to RAY_TRACING_KHR */
    bool                   has_rt_desc_buf_offsets;
    VkPipelineLayout       rt_db_layout;
    uint32_t               rt_db_first_set;
    uint32_t               rt_db_set_count;
    uint32_t               rt_db_buf_indices[16];
    VkDeviceSize           rt_db_offsets[16];
    struct RtCmdState     *next;
};

struct RtLayoutNode {
    VkPipelineLayout       layout;
    VkShaderStageFlags     pc_stages;
    struct RtLayoutNode   *next;
};

#define RT_HANDLE_BUCKETS 1024
#ifndef COMPAT_INFO
#define COMPAT_INFO(...) LOG(__VA_ARGS__)
#endif
/* 11-10 keeper (lead-1110): every RT env lever is read ONCE, thread-safely (pthread_once), into g_rt_env.
 * Before, each lever was a function-local `static int v = -1` that two threads could read/write at the same
 * time on their first trace / AS-build / CRTP call (Codex 1110-194217: C data race = UB). */
struct RtEnv {
    int skip_traces;    /* MALI_COMPAT_RT_SKIP_TRACES=1: skip every trace dispatch (default 0) */
    int max_stages;     /* MALI_COMPAT_RT_MAX_STAGES=N: no megakernel above N stages (default 0 = off) */
    int sync_compile;   /* MALI_COMPAT_RT_SYNC_COMPILE=1: megakernel compiled inside CRTP (default 0 = async) */
    int own_scratch;    /* MALI_COMPAT_RT_OWN_SCRATCH=0: keep the app's AS build scratch (default 1) */
    int force_build;    /* MALI_COMPAT_RT_FORCE_BUILD=1: UPDATE builds become BUILD, src cleared (default 0) */
    uint32_t build_log; /* MALI_COMPAT_RT_BUILD_LOG=N: per-build lines for the first N calls (+N after 1st trace) */
    int trace;          /* MALI_COMPAT_TRACE=1: COMPAT_TRACE lines (default 0) */
    int size_pad;       /* 1110b MALI_COMPAT_RT_SIZE_PAD=0: GetASBuildSizes returns the driver's sizes (default 1 = inflate) */
    int as_trust;       /* 1110b MALI_COMPAT_RT_AS_TRUST=1: no handle recreation on a type change, no UPDATE->BUILD for
                         * unbuilt/mutated tracks (an UPDATE with a null src still becomes a BUILD); default 0 */
    int upd_src;        /* 1110b MALI_COMPAT_RT_UPD_SRC=1: UPDATE with src != dst -> BUILD; =2 only if src storage > dst
                         * storage; 0 = unchanged. lead-1110c: default 1 (the blob breaks on such refits, as-probe) */
    int clone;          /* lead-1110c MALI_COMPAT_RT_CLONE=0: CLONE copies go to the driver (default 1 = the layer copies
                         * the source's real AS bytes with vkCmdCopyBuffer) */
    int clone_refit;    /* lead-1110c MALI_COMPAT_RT_CLONE_REFIT=0: an in-place UPDATE of a copied-into AS is passed
                         * through (default 1 = its first UPDATE after a copy becomes a BUILD) */
    /* round 4 (11-11) [keeper, Claude Opus 5.5] */
    int stats;          /* MALI_COMPAT_RT_STATS=1: megakernels created with CAPTURE_STATISTICS, one rt_stats: line per
                         * executable (needs VK_KHR_pipeline_executable_properties + its feature; default 0) */
    int scratch_head;   /* MALI_COMPAT_RT_SCRATCH_HEAD=0: pool slice = the driver's scratch need (64 KB aligned), no
                         * +max(need, 1 MB) headroom (default 1 = headroom as since 11-08) */
    uint32_t pool_mb;   /* MALI_COMPAT_RT_SCRATCH_POOL_MB=N (8..1024): size of the layer's AS scratch pool (default 128) */
    int free_stages;    /* MALI_COMPAT_RT_FREE_STAGES=1: a non-library RT pipeline frees its stage SPIR-V copies once its
                         * megakernel SPIR-V is built, and the megakernel VkShaderModule once its compute pipeline exists
                         * (default 0 = both kept until vkDestroyPipeline) */
    int dump_spv;       /* MALI_COMPAT_DUMP_RT_SPV=0 (or "off"): no megakernel .spv file per CRTP (default 1: dumped to
                         * the key's directory, /tmp if unset - unchanged) */
    uint32_t mem_log_s; /* MALI_COMPAT_RT_MEM_LOG_S=N: rt_mem / vk_mem / rt_rays lines every N s (default 10; 0 = only
                         * the final lines at vkDestroyDevice) */
    /* split-1112 (11-11) [RTX keeper, Claude Opus 5.5] */
    int split;          /* MALI_COMPAT_RT_SPLIT=1: a pipeline with 2..RT_SPLIT_MAX distinct raygen stages and >= split_min
                         * stages gets one megakernel per raygen (default 0 = one megakernel for the whole pipeline) */
    uint32_t split_min; /* MALI_COMPAT_RT_SPLIT_MIN=N: smallest pipeline (stage count) that is split (default 12) */
    uint32_t split_jobs;/* MALI_COMPAT_RT_SPLIT_JOBS=N (1..16): split-child megakernels compiled at the same time (default 2;
                         * an unsplit W3 load compiles its 80- and 82-stage kernels on 2 threads, a split one has 8 children) */
    int local1d;        /* local1d-1112 MALI_COMPAT_RT_LOCAL1D=1: a direct trace with height < 4 runs 32 consecutive x per
                         * workgroup (launch id = (wg.x*32 + LocalInvocationIndex, wg.y, wg.z)) instead of 8x4 tiles, of
                         * which 3 of 4 rows are out of bounds for a height-1 launch (default 0) */
};
static struct RtEnv    g_rt_env;
static pthread_once_t  g_rt_env_once = PTHREAD_ONCE_INIT;
#define RT_AS_SCRATCH_SIZE (128 * 1024 * 1024)
static void rt_env_init(void) {
    const char *e;
    e = getenv("MALI_COMPAT_RT_SKIP_TRACES");  g_rt_env.skip_traces  = (e && e[0] == '1') ? 1 : 0;
    e = getenv("MALI_COMPAT_RT_MAX_STAGES");   g_rt_env.max_stages   = e ? atoi(e) : 0;
    if (g_rt_env.max_stages < 0) g_rt_env.max_stages = 0;
    e = getenv("MALI_COMPAT_RT_SYNC_COMPILE"); g_rt_env.sync_compile = (e && e[0] == '1') ? 1 : 0;
    e = getenv("MALI_COMPAT_RT_OWN_SCRATCH");  g_rt_env.own_scratch  = (e && e[0] == '0') ? 0 : 1;
    e = getenv("MALI_COMPAT_RT_FORCE_BUILD");  g_rt_env.force_build  = (e && e[0] == '1') ? 1 : 0;
    e = getenv("MALI_COMPAT_RT_BUILD_LOG");    g_rt_env.build_log    = (e && atoi(e) > 0) ? (uint32_t)atoi(e) : 0u;
    e = getenv("MALI_COMPAT_TRACE");           g_rt_env.trace        = (e && e[0] == '1') ? 1 : 0;
    /* round 4: default 0 (the measured L7 configuration, docs/games/witcher3.md "GPU memory, padding on vs off");
     * =1 keeps the old 1.5x+2 MiB / 2x+4 MiB inflation as a lever */
    e = getenv("MALI_COMPAT_RT_SIZE_PAD");     g_rt_env.size_pad     = (e && e[0] == '1') ? 1 : 0;
    e = getenv("MALI_COMPAT_RT_AS_TRUST");     g_rt_env.as_trust     = (e && e[0] == '1') ? 1 : 0;
    e = getenv("MALI_COMPAT_RT_UPD_SRC");      g_rt_env.upd_src      = (e && (e[0] == '0' || e[0] == '2')) ? e[0] - '0' : 1;
    e = getenv("MALI_COMPAT_RT_CLONE");        g_rt_env.clone        = (e && e[0] == '0') ? 0 : 1;
    e = getenv("MALI_COMPAT_RT_CLONE_REFIT");  g_rt_env.clone_refit  = (e && e[0] == '0') ? 0 : 1;
    e = getenv("MALI_COMPAT_RT_STATS");        g_rt_env.stats        = (e && e[0] == '1') ? 1 : 0;
    e = getenv("MALI_COMPAT_RT_SCRATCH_HEAD"); g_rt_env.scratch_head = (e && e[0] == '0') ? 0 : 1;
    e = getenv("MALI_COMPAT_RT_SCRATCH_POOL_MB");
    {
        int mb = e ? atoi(e) : 0;
        g_rt_env.pool_mb = (mb >= 8 && mb <= 1024) ? (uint32_t)mb : (uint32_t)(RT_AS_SCRATCH_SIZE >> 20);
    }
    e = getenv("MALI_COMPAT_RT_FREE_STAGES");  g_rt_env.free_stages  = (e && e[0] == '1') ? 1 : 0;
    e = getenv("MALI_COMPAT_DUMP_RT_SPV");     g_rt_env.dump_spv     = (e && (!strcmp(e, "0") || !strcmp(e, "off"))) ? 0 : 1;
    e = getenv("MALI_COMPAT_RT_MEM_LOG_S");    g_rt_env.mem_log_s    = e ? (uint32_t)(atoi(e) > 0 ? atoi(e) : 0) : 10u;
    e = getenv("MALI_COMPAT_RT_SPLIT");        g_rt_env.split        = (e && e[0] == '1') ? 1 : 0;
    e = getenv("MALI_COMPAT_RT_SPLIT_MIN");    g_rt_env.split_min    = (e && atoi(e) > 0) ? (uint32_t)atoi(e) : 12u;
    e = getenv("MALI_COMPAT_RT_LOCAL1D");      g_rt_env.local1d      = (e && e[0] == '1') ? 1 : 0;
    e = getenv("MALI_COMPAT_RT_SPLIT_JOBS");   g_rt_env.split_jobs   = (e && atoi(e) >= 1 && atoi(e) <= 16) ? (uint32_t)atoi(e) : 2u;
    e = getenv("MALI_COMPAT_RT_GUARD_ITERS");
    int gv = e ? atoi(e) : 0;
    g_rt_guard_max = gv > 0 ? (uint32_t)gv : 4096u;   /* was a racy lazy write in CRTP */
    if (g_rt_env.skip_traces)
        LOG("rt_skip: MALI_COMPAT_RT_SKIP_TRACES=1 -> every TraceRays/TraceRaysIndirect is skipped (AS builds kept)");
    LOG("rt_scratch: layer-owned AS build scratch %s (MALI_COMPAT_RT_OWN_SCRATCH), pool %u MB",
        g_rt_env.own_scratch ? "ON" : "OFF", (unsigned)(RT_AS_SCRATCH_SIZE >> 20));
    if (g_rt_env.force_build)
        LOG("rt_build: MALI_COMPAT_RT_FORCE_BUILD=1 -> every UPDATE build becomes a full BUILD (src cleared)");
    if (g_rt_env.size_pad)
        LOG("rt_build: MALI_COMPAT_RT_SIZE_PAD=1 -> vkGetAccelerationStructureBuildSizesKHR inflates the driver's sizes "
            "(1.5x+2 MiB AS, 2x+4 MiB scratch, 64 KB padding; default since round 4: the driver's own sizes)");
    if (g_rt_env.stats)
        LOG("rt_stats: MALI_COMPAT_RT_STATS=1 -> megakernels are created with CAPTURE_STATISTICS (if the driver has "
            "VK_KHR_pipeline_executable_properties + pipelineExecutableInfo)");
    if (!g_rt_env.scratch_head)
        LOG("rt_scratch: MALI_COMPAT_RT_SCRATCH_HEAD=0 -> pool slice = the driver's scratch need (no +max(need,1 MB))");
    if (g_rt_env.pool_mb != (uint32_t)(RT_AS_SCRATCH_SIZE >> 20))
        LOG("rt_scratch: MALI_COMPAT_RT_SCRATCH_POOL_MB=%u -> AS scratch pool %u MB", g_rt_env.pool_mb, g_rt_env.pool_mb);
    if (g_rt_env.free_stages)
        LOG("rt_mem: MALI_COMPAT_RT_FREE_STAGES=1 -> stage SPIR-V copies freed after the megakernel SPIR-V is built, "
            "megakernel shader module destroyed after its pipeline is created (non-library pipelines)");
    if (!g_rt_env.dump_spv)
        LOG("rt_mem: MALI_COMPAT_DUMP_RT_SPV=%s -> no megakernel .spv dump files", getenv("MALI_COMPAT_DUMP_RT_SPV"));
    if (g_rt_env.mem_log_s != 10u)
        LOG("rt_mem: MALI_COMPAT_RT_MEM_LOG_S=%u -> rt_mem/vk_mem/rt_rays every %u s (0 = only at vkDestroyDevice)",
            g_rt_env.mem_log_s, g_rt_env.mem_log_s);
    if (g_rt_env.as_trust)
        LOG("rt_build: MALI_COMPAT_RT_AS_TRUST=1 -> no AS handle recreation on a type change, no UPDATE->BUILD for "
            "unbuilt/mutated ASes (UPDATE with a null src still becomes BUILD)");
    if (g_rt_env.upd_src != 1)
        LOG("rt_build: MALI_COMPAT_RT_UPD_SRC=%d -> %s", g_rt_env.upd_src, g_rt_env.upd_src == 2
            ? "an UPDATE from a different source AS becomes a BUILD only when the source's storage is larger"
            : "an UPDATE from a different source AS is passed through (default 1 turns it into a BUILD)");
    if (!g_rt_env.clone)
        LOG("rt_copy: MALI_COMPAT_RT_CLONE=0 -> CLONE copies go to the driver's vkCmdCopyAccelerationStructureKHR "
            "(it copies the source's whole created size)");
    if (!g_rt_env.clone_refit)
        LOG("rt_build: MALI_COMPAT_RT_CLONE_REFIT=0 -> the first in-place UPDATE of a copied-into AS is passed through");
    if (g_rt_env.split)
        LOG("rt_split: MALI_COMPAT_RT_SPLIT=1 -> pipelines with >= %u stages and 2..%d raygen stages get one megakernel "
            "per raygen (GPU-side select + indirect dispatch per trace), %u child compiles at a time", g_rt_env.split_min,
            RT_SPLIT_MAX, g_rt_env.split_jobs);
    if (g_rt_env.local1d)
        LOG("rt_local1d: MALI_COMPAT_RT_LOCAL1D=1 -> direct traces with height < 4 dispatch (ceil(w/32), h, d) groups, "
            "launch id x = workgroup x * 32 + local index (8x4 tiles otherwise)");
    if (g_rt_env.build_log)
        LOG("rt_build: MALI_COMPAT_RT_BUILD_LOG=%u -> per-build lines for the first %u calls and %u calls after the "
            "first TraceRays", g_rt_env.build_log, g_rt_env.build_log, g_rt_env.build_log);
}
static inline const struct RtEnv *rt_env(void) {
    pthread_once(&g_rt_env_once, rt_env_init);
    return &g_rt_env;
}

/* ── round 4 (11-11) [keeper, Claude Opus 5.5]: memory accounting + ray counter ──────────────────────────────────
 * Always on with MALI_COMPAT_RT_PIPELINE=1 (nothing in this file runs without it; the vk_mem hooks in
 * mali_compat_layer.c are gated by g_rt_pipeline). Relaxed atomics, no per-call logging. rt_report() prints three lines
 * (rt_mem, vk_mem, rt_rays) at most every MALI_COMPAT_RT_MEM_LOG_S seconds (default 10) from the AS-build and
 * trace-record paths, and once more ("final") at vkDestroyDevice. Purpose (RTX lead 11-11): split W3's +4.2-5.3 GB GPU
 * memory into AS storage / layer scratch / everything else, and find layer-owned waste. */
struct RtAcct { _Atomic int64_t n, bytes, peak; };   /* live count, live bytes, peak bytes */
static inline void rt_acct_peak(struct RtAcct *a, int64_t now) {
    int64_t p = atomic_load_explicit(&a->peak, memory_order_relaxed);
    while (now > p && !atomic_compare_exchange_weak_explicit(&a->peak, &p, now, memory_order_relaxed,
                                                             memory_order_relaxed)) {}
}
static inline void rt_acct_add(struct RtAcct *a, int64_t bytes) {
    atomic_fetch_add_explicit(&a->n, 1, memory_order_relaxed);
    rt_acct_peak(a, atomic_fetch_add_explicit(&a->bytes, bytes, memory_order_relaxed) + bytes);
}
static inline void rt_acct_sub(struct RtAcct *a, int64_t bytes) {
    atomic_fetch_sub_explicit(&a->n, 1, memory_order_relaxed);
    atomic_fetch_sub_explicit(&a->bytes, bytes, memory_order_relaxed);
}
static inline void rt_acct_adj(struct RtAcct *a, int64_t delta) {   /* bytes only, the count stays */
    if (delta) rt_acct_peak(a, atomic_fetch_add_explicit(&a->bytes, delta, memory_order_relaxed) + delta);
}
static inline int64_t rt_kb(const struct RtAcct *a, int which) {
    return (which == 0 ? atomic_load_explicit(&((struct RtAcct *)a)->bytes, memory_order_relaxed)
                       : atomic_load_explicit(&((struct RtAcct *)a)->peak, memory_order_relaxed)) >> 10;
}
static inline int64_t rt_cnt(const struct RtAcct *a) {
    return atomic_load_explicit(&((struct RtAcct *)a)->n, memory_order_relaxed);
}
/* AS objects, by built type (0 = not built yet, 1 = BLAS, 2 = TLAS): created size (the app's
 * VkAccelerationStructureCreateInfoKHR.size) and the driver's accelerationStructureSize of the last BUILD into it
 * (count = objects with a known build size). Updated under g_rt_as_lock (rt_as_acct_locked). */
static struct RtAcct g_acct_as[3], g_acct_as_drv[3];
static struct RtAcct g_acct_as_all, g_acct_as_drv_all;   /* the same, all classes (peak of the sum) */
static struct RtAcct g_acct_asbuf;   /* VkBuffers with ACCELERATION_STRUCTURE_STORAGE usage, created size (compat_CreateBuffer) */
static struct RtAcct g_acct_pool, g_acct_ring, g_acct_zero;   /* layer GPU memory: AS scratch pool, param ring, zero buf */
/* layer host memory (malloc): AS tracks (+ retired-handle arrays), the app's RT shader module copies, stage SPIR-V
 * copies of RT pipelines, RtPipeline objects (+ group tables), per-command-buffer RT state, the AS-buffer map */
static struct RtAcct g_acct_tracks, g_acct_modules, g_acct_stages, g_acct_pipes, g_acct_cmdst, g_acct_bufmap;
/* megakernels: live VkShaderModules (n, SPIR-V bytes), cumulative SPIR-V generated, compile states, dump files */
static struct RtAcct g_acct_mega_mod, g_acct_dump;
static _Atomic int64_t g_mega_spv_total, g_mega_requested, g_mega_compiling, g_mega_live, g_mega_failed,
                       g_mega_capped, g_mega_freed_stage_bytes;
static _Atomic int64_t g_rt_scr_maxslice;   /* largest single pool slice handed out (bytes) */
/* ray counter (record time: counted when vkCmdTraceRays* is recorded; one count per recording, so a command buffer
 * submitted N times is counted once). "rays" = width*height*depth of the emulated direct TraceRays = megakernel
 * invocations launched (primary dispatch), NOT ray-query traversals (a megakernel invocation may trace 0..n rays). */
#define RT_RAY_CLASSES 5
static const char *const g_ray_cls_name[RT_RAY_CLASSES] = { "1-6", "7-8", "9-14", "15-40", "41+" };
static _Atomic uint64_t g_ray_traces, g_ray_rays, g_ray_cls_tr[RT_RAY_CLASSES], g_ray_cls_rays[RT_RAY_CLASSES],
                        g_ray_ind, g_ray_skip_zero, g_ray_skip_nostate;
static _Atomic uint64_t g_split_traces, g_split_select, g_split_fallback;   /* split-1112 (record time) */
static _Atomic uint64_t g_local1d_traces;                                   /* local1d-1112 (record time) */
static inline unsigned rt_ray_class(uint32_t stages) {
    return stages <= 6 ? 0u : stages <= 8 ? 1u : stages <= 14 ? 2u : stages <= 40 ? 3u : 4u;
}
static inline int rt_as_cls(VkAccelerationStructureTypeKHR built) {
    return built == VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR ? 1
         : built == VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR ? 2 : 0;
}
/* vk_mem: every vkAllocateMemory through the layer's hook (= the app's; the layer's own RT allocations call the driver
 * directly and are in rt_mem pool/ring/zero), live + peak per memory type and per heap (index RT_VKM_HEAPS = dma-buf
 * fd imports, the exporter's memory), and live count per exact allocation size (512 distinct sizes; more -> "other").
 * Called from compat_AllocateMemory / compat_FreeMemory only with MALI_COMPAT_RT_PIPELINE=1. */
#define RT_VKM_TYPES 32
#define RT_VKM_HEAPS 16
#define RT_VKM_SIZES 512
static struct RtAcct g_vkm_type[RT_VKM_TYPES], g_vkm_heap[RT_VKM_HEAPS + 1], g_vkm_all;
static _Atomic uint64_t g_vkm_allocs_total;
struct RtVkmSize { uint64_t size; int64_t live; };
static struct RtVkmSize g_vkm_sizes[RT_VKM_SIZES];
static int64_t          g_vkm_other_n, g_vkm_other_bytes;
static pthread_mutex_t  g_vkm_lock = PTHREAD_MUTEX_INITIALIZER;
static void rt_vkmem_size_adj(uint64_t size, int64_t dn) {
    pthread_mutex_lock(&g_vkm_lock);
    int hit = -1, free_slot = -1;
    for (int i = 0; i < RT_VKM_SIZES; i++) {
        if (g_vkm_sizes[i].live > 0 && g_vkm_sizes[i].size == size) { hit = i; break; }
        if (free_slot < 0 && g_vkm_sizes[i].live <= 0) free_slot = i;
    }
    if (hit < 0 && dn > 0 && free_slot >= 0) { hit = free_slot; g_vkm_sizes[hit] = (struct RtVkmSize){ size, 0 }; }
    if (hit >= 0) g_vkm_sizes[hit].live += dn;
    else { g_vkm_other_n += dn; g_vkm_other_bytes += dn * (int64_t)size; }
    pthread_mutex_unlock(&g_vkm_lock);
}
static void rt_vkmem_alloc(uint32_t type, uint32_t heap, uint64_t size) {
    if (type < RT_VKM_TYPES) rt_acct_add(&g_vkm_type[type], (int64_t)size);
    rt_acct_add(&g_vkm_heap[heap < RT_VKM_HEAPS ? heap : RT_VKM_HEAPS], (int64_t)size);
    rt_acct_add(&g_vkm_all, (int64_t)size);
    atomic_fetch_add_explicit(&g_vkm_allocs_total, 1, memory_order_relaxed);
    rt_vkmem_size_adj(size, 1);
}
static void rt_vkmem_free(uint32_t type, uint32_t heap, uint64_t size) {
    if (type < RT_VKM_TYPES) rt_acct_sub(&g_vkm_type[type], (int64_t)size);
    rt_acct_sub(&g_vkm_heap[heap < RT_VKM_HEAPS ? heap : RT_VKM_HEAPS], (int64_t)size);
    rt_acct_sub(&g_vkm_all, (int64_t)size);
    rt_vkmem_size_adj(size, -1);
}
struct DeviceData;
static void rt_report(struct DeviceData *d, bool final);
/* AS-build call counter + the counter value at the first recorded TraceRays (MALI_COMPAT_RT_BUILD_LOG). */
static _Atomic uint32_t g_as_build_count = 0;
static _Atomic uint32_t g_rt_first_trace_call = UINT32_MAX;
static inline void rt_note_trace(void) {
    if (atomic_load_explicit(&g_rt_first_trace_call, memory_order_relaxed) != UINT32_MAX) return;
    uint32_t expect = UINT32_MAX, now = atomic_load(&g_as_build_count);
    if (atomic_compare_exchange_strong(&g_rt_first_trace_call, &expect, now) && rt_env()->build_log)
        LOG("rt_build: first TraceRays recorded after %u CmdBuildAS calls", now);
}

#ifndef COMPAT_TRACE
/* 11-10: gated behind MALI_COMPAT_TRACE=1 (ported from Gemini 4 Argon's wt-rtx-emu; there a racy per-site static,
 * here the once-read g_rt_env). Default off: these lines were logged on every TraceRays call. */
#define COMPAT_TRACE(...) do { if (__builtin_expect(rt_env()->trace, 0)) LOG(__VA_ARGS__); } while (0)
#endif

static inline unsigned rt_bucket(uintptr_t h) {
    return handle_bucket((uint64_t)h, RT_HANDLE_BUCKETS);
}

static pthread_mutex_t      g_rt_lock = PTHREAD_MUTEX_INITIALIZER;
static struct RtShaderModule *g_rt_modules[RT_HANDLE_BUCKETS];
static struct RtPipeline     *g_rt_pipelines[RT_HANDLE_BUCKETS];
static struct RtDeferredOp   *g_rt_deferred_ops[RT_HANDLE_BUCKETS];
static struct RtCmdState     *g_rt_cmd_states[RT_HANDLE_BUCKETS];
static struct RtLayoutNode   *g_rt_layouts[RT_HANDLE_BUCKETS];

/* ── Helper Lookups ───────────────────────────────────────────────────────── */

static uint32_t rt_fnv1a_32(const void *data, size_t len, uint32_t h) {
    const uint8_t *p = (const uint8_t *)data;
    for (size_t i = 0; i < len; i++) {
        h ^= p[i];
        h *= 16777619u;
    }
    return h;
}

static uint32_t rt_compute_stage_id(const uint32_t *words, size_t word_count,
                                    VkShaderStageFlagBits stage, const char *entry_name,
                                    const VkSpecializationInfo *spec) {
    uint32_t h = 2166136261u;
    h = rt_fnv1a_32(words, word_count * sizeof(uint32_t), h);
    uint32_t st = (uint32_t)stage;
    h = rt_fnv1a_32(&st, sizeof(st), h);
    if (entry_name) h = rt_fnv1a_32(entry_name, strlen(entry_name), h);
    if (spec && spec->mapEntryCount > 0 && spec->pMapEntries && spec->pData) {
        for (uint32_t i = 0; i < spec->mapEntryCount; i++) {
            const VkSpecializationMapEntry *e = &spec->pMapEntries[i];
            h = rt_fnv1a_32(&e->constantID, sizeof(e->constantID), h);
            if (e->offset + e->size <= spec->dataSize) {
                h = rt_fnv1a_32((const uint8_t *)spec->pData + e->offset, e->size, h);
            }
        }
    }
    return h == 0 ? 1u : h;
}

static bool rt_is_ray_tracing_spirv(const uint32_t *code, size_t size_bytes) {
    if (!code || size_bytes < 20 || code[0] != RT_SPV_MAGIC) return false;
    size_t count = size_bytes / 4;
    size_t i = 5;
    while (i < count) {
        uint32_t wc = code[i] >> 16;
        uint32_t op = code[i] & 0xFFFFu;
        if (wc == 0 || i + wc > count) break;
        if (op == RT_OP_CAPABILITY && wc >= 2) {
            uint32_t cap = code[i + 1];
            if (cap == RT_CAP_RAY_TRACING_KHR ||
                cap == RT_CAP_RAY_TRACING_PROVISIONAL_KHR) {
                return true;
            }
        } else if (op == RT_OP_ENTRY_POINT && wc >= 3) {
            uint32_t em = code[i + 1];
            if (em >= RT_EXEC_MODEL_RAYGEN_KHR && em <= RT_EXEC_MODEL_CALLABLE_KHR) {
                return true;
            }
        } else if (op == RT_OP_FUNCTION) {
            break;
        }
        i += wc;
    }
    return false;
}

static VkShaderModule rt_register_shader_module(VkDevice device, const uint32_t *code, size_t size_bytes) {
    struct RtShaderModule *m = (struct RtShaderModule *)calloc(1, sizeof(*m));
    if (!m) return VK_NULL_HANDLE;
    m->magic = 0x5254534Du; /* 'RTSM' */
    m->device = device;
    m->word_count = size_bytes / 4;
    m->words = (uint32_t *)malloc(m->word_count * sizeof(uint32_t));
    if (!m->words) {
        free(m);
        return VK_NULL_HANDLE;
    }
    memcpy(m->words, code, m->word_count * sizeof(uint32_t));
    m->hash = spv_hash(m->words, m->word_count * sizeof(uint32_t));
    rt_acct_add(&g_acct_modules, (int64_t)(sizeof(*m) + m->word_count * sizeof(uint32_t)));

    VkShaderModule handle = (VkShaderModule)(uintptr_t)m;
    unsigned b = rt_bucket((uintptr_t)handle);
    pthread_mutex_lock(&g_rt_lock);
    m->next = g_rt_modules[b];
    g_rt_modules[b] = m;
    pthread_mutex_unlock(&g_rt_lock);
    return handle;
}

static struct RtShaderModule *rt_lookup_shader_module(VkShaderModule handle) {
    if (!handle) return NULL;
    unsigned b = rt_bucket((uintptr_t)handle);
    pthread_mutex_lock(&g_rt_lock);
    for (struct RtShaderModule *m = g_rt_modules[b]; m; m = m->next) {
        if ((VkShaderModule)(uintptr_t)m == handle) {
            pthread_mutex_unlock(&g_rt_lock);
            return m;
        }
    }
    pthread_mutex_unlock(&g_rt_lock);
    return NULL;
}

static bool rt_destroy_shader_module(VkShaderModule handle) {
    if (!handle) return false;
    unsigned b = rt_bucket((uintptr_t)handle);
    pthread_mutex_lock(&g_rt_lock);
    struct RtShaderModule **pp = &g_rt_modules[b];
    while (*pp) {
        if ((VkShaderModule)(uintptr_t)(*pp) == handle) {
            struct RtShaderModule *victim = *pp;
            *pp = victim->next;
            pthread_mutex_unlock(&g_rt_lock);
            rt_acct_sub(&g_acct_modules, (int64_t)(sizeof(*victim) + victim->word_count * sizeof(uint32_t)));
            free(victim->words);
            free(victim);
            return true;
        }
        pp = &(*pp)->next;
    }
    pthread_mutex_unlock(&g_rt_lock);
    return false;
}

static struct RtPipeline *rt_lookup_pipeline(VkPipeline handle) {
    if (!handle) return NULL;
    unsigned b = rt_bucket((uintptr_t)handle);
    pthread_mutex_lock(&g_rt_lock);
    for (struct RtPipeline *p = g_rt_pipelines[b]; p; p = p->next) {
        if ((VkPipeline)(uintptr_t)p == handle) {
            pthread_mutex_unlock(&g_rt_lock);
            return p;
        }
    }
    pthread_mutex_unlock(&g_rt_lock);
    return NULL;
}

static size_t rt_stage_record_bytes(const struct RtStageRecord *s) {
    return s->word_count * sizeof(uint32_t) + (s->entry_name ? strlen(s->entry_name) + 1 : 0) +
           s->spec_entry_count * sizeof(struct RtSpecConstantEntry) + s->spec_data_size;
}
static void rt_stage_record_count(struct RtStageRecord *s) {   /* round 4: after a successful copy/clone */
    s->acct_bytes = rt_stage_record_bytes(s);
    rt_acct_add(&g_acct_stages, (int64_t)s->acct_bytes);
}
static void rt_free_stage_record(struct RtStageRecord *s) {
    if (!s) return;
    if (s->acct_bytes) { rt_acct_sub(&g_acct_stages, (int64_t)s->acct_bytes); s->acct_bytes = 0; }
    free(s->words);
    free(s->entry_name);
    free(s->spec_entries);
    free(s->spec_data);
}

/* ── 11-08 megafix: async megakernel compile, stage cap, skip counter ─────── */
static inline int64_t rt_now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (int64_t)ts.tv_sec * 1000 + ts.tv_nsec / 1000000;
}
static inline bool rt_mega_ready(struct RtPipeline *rp) {
    return rp && __atomic_load_n(&rp->compute_pipeline, __ATOMIC_ACQUIRE) != VK_NULL_HANDLE;
}
/* split-1112: a split parent is ready when every child megakernel is (a trace may select any of them) */
static inline bool rt_pipe_ready(struct RtPipeline *rp) {
    if (!rp) return false;
    if (!rp->split_n) return rt_mega_ready(rp);
    for (uint32_t k = 0; k < rp->split_n; k++)
        if (!rt_mega_ready(rp->split[k])) return false;
    return true;
}
/* split-1112: compile state of a split parent = 3/4 if a child failed/was capped, else 1 while one still compiles */
static inline int rt_pipe_state(struct RtPipeline *rp) {
    if (!rp->split_n) return __atomic_load_n(&rp->compile_state, __ATOMIC_ACQUIRE);
    int st = 2;
    for (uint32_t k = 0; k < rp->split_n; k++) {
        int c = __atomic_load_n(&rp->split[k]->compile_state, __ATOMIC_ACQUIRE);
        if (c == 3 || c == 4) return c;
        if (c == 0) return 3;   /* never queued (SPIR-V/module failure): the parent is visible only after CRTP */
        if (c != 2) st = 1;
    }
    return st;
}
static _Atomic uint64_t g_rt_skip_pending = 0;   /* compile still running on the worker */
static _Atomic uint64_t g_rt_skip_nopipe  = 0;   /* capped (MALI_COMPAT_RT_MAX_STAGES) or compile failed */
static void rt_count_skip(struct RtPipeline *rp) {
    int st = rt_pipe_state(rp);
    uint64_t pend = (st == 1) ? atomic_fetch_add(&g_rt_skip_pending, 1) + 1 : atomic_load(&g_rt_skip_pending);
    uint64_t nop  = (st == 1) ? atomic_load(&g_rt_skip_nopipe) : atomic_fetch_add(&g_rt_skip_nopipe, 1) + 1;
    if (!g_debug) return;
    static _Atomic int64_t last_ms = 0;
    int64_t now = rt_now_ms(), prev = atomic_load(&last_ms);
    if ((pend + nop == 1 || now - prev >= 1000) && atomic_compare_exchange_strong(&last_ms, &prev, now))
        LOG("rt_skip: t=%lld ms trace dispatch skipped (pipe=%p state=%d) total pending=%llu capped/failed=%llu",
            (long long)now, (void *)rp, st, (unsigned long long)pend, (unsigned long long)nop);
}
/* 11-08 keeper: MALI_COMPAT_RT_SKIP_TRACES=1 = A/B level "AS builds + megakernel compiles as usual, but NO trace
 * dispatch at all" (isolates the AS build path from the megakernel). Default off. */
static _Atomic uint64_t g_rt_skip_forced = 0;
static bool rt_forced_trace_skip(void) {
    if (!rt_env()->skip_traces) return false;
    uint64_t n = atomic_fetch_add(&g_rt_skip_forced, 1) + 1;
    static _Atomic int64_t last_ms = 0;
    int64_t now = rt_now_ms(), prev = atomic_load(&last_ms);
    if ((n == 1 || now - prev >= 5000) && atomic_compare_exchange_strong(&last_ms, &prev, now))
        LOG("rt_skip: t=%lld ms forced skip (MALI_COMPAT_RT_SKIP_TRACES) total=%llu", (long long)now, (unsigned long long)n);
    return true;
}
/* 11-10 keeper (lead-1110, Codex 1110-194217 findings 1+2): an async megakernel compile no longer touches a
 * DeviceData (g_devices[] entries move on DestroyDevice compaction): the worker uses the device handle and the
 * driver entry points copied into the job. Every queued job sits on g_rt_jobs until its LAST driver call is done;
 * compat_DestroyDevice waits there (rt_drain_device_jobs) before it calls down. The app's pipeline layout used by
 * the compile is pinned (g_rt_layout_pins): vkDestroyPipelineLayout of a pinned layout is deferred (NULL allocator)
 * and executed by the worker when the job ends, or waits for the job (app allocator: no safe deferral). */
struct RtCompileJob {
    VkDevice device;
    struct RtPipeline *rp;
    VkComputePipelineCreateInfo cpci;
    VkPipelineCreateFlags2CreateInfoKHR f2;
    PFN_vkCreateComputePipelines fpCreateComputePipelines;
    PFN_vkDestroyPipeline        fpDestroyPipeline;
    PFN_vkDestroyShaderModule    fpDestroyShaderModule;
    uint32_t stages, groups;
    size_t words;
    int64_t t_req;
    /* round 4: MALI_COMPAT_RT_STATS=1 (only set when the device enabled pipelineExecutableInfo) */
    PFN_vkGetPipelineExecutablePropertiesKHR fpGetPipelineExecutablePropertiesKHR;
    PFN_vkGetPipelineExecutableStatisticsKHR fpGetPipelineExecutableStatisticsKHR;
    bool release_module;          /* round 4: MALI_COMPAT_RT_FREE_STAGES=1: destroy rp->compute_module after the compile */
    bool async;
    struct RtCompileJob *next;    /* g_rt_jobs (async jobs only) */
};
struct RtLayoutPin {
    VkDevice device;
    VkPipelineLayout layout;
    uint32_t refs;                             /* pending compile jobs using the layout */
    uint32_t pipe_refs;                        /* 1110b: live RT pipelines whose trace recording uses the layout */
    PFN_vkDestroyPipelineLayout fp_destroy;    /* != NULL: the app destroyed it, the last ref destroys it */
    bool has_alloc;                            /* 1110b: app allocator, copied (see compat_RtDestroyPipelineLayout) */
    VkAllocationCallbacks alloc;
    struct RtLayoutPin *next;
};
static pthread_mutex_t      g_rt_job_lock = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t       g_rt_job_cv   = PTHREAD_COND_INITIALIZER;
static struct RtCompileJob *g_rt_jobs;
static struct RtLayoutPin  *g_rt_layout_pins;
static struct RtLayoutPin **rt_pin_find_locked(VkDevice dev, VkPipelineLayout l) {
    struct RtLayoutPin **pp = &g_rt_layout_pins;
    while (*pp && !((*pp)->device == dev && (*pp)->layout == l)) pp = &(*pp)->next;
    return pp;
}
/* Unlinks the pin when no job and no pipeline holds it any more; returns it (caller runs rt_pin_dispose outside the
 * lock) or NULL. */
static struct RtLayoutPin *rt_pin_maybe_unlink_locked(struct RtLayoutPin **pp) {
    struct RtLayoutPin *n = *pp;
    if (!n || n->refs || n->pipe_refs) return NULL;
    *pp = n->next;
    return n;
}
/* Destroys the app's layout if the app already asked for it (deferred destroy), then frees the pin. */
static void rt_pin_dispose(struct RtLayoutPin *n, const char *why) {
    if (!n) return;
    if (n->fp_destroy) {
        LOG("rt_compile: destroying pipeline layout %p deferred by the app's vkDestroyPipelineLayout (%s)",
            (void *)n->layout, why);
        n->fp_destroy(n->device, n->layout, n->has_alloc ? &n->alloc : NULL);
    }
    free(n);
}
/* 1110b keeper (Codex 1110-202009 #3): the trace path records vkCmdPushConstants / descriptor-buffer offsets with the
 * RT pipeline's creation layout (rp->layout) long after CRTP returned. The app may destroy that layout right after
 * CRTP (legal), so every non-library RT pipeline now pins its layout for its WHOLE life: a vkDestroyPipelineLayout
 * of it is deferred until the pipeline object is freed (rt_free_pipeline_obj_fp - normal DestroyPipeline and the
 * worker's destroyed-while-compiling path) or the device is destroyed (rt_drain_device_jobs). */
static bool rt_pin_pipeline_layout(VkDevice dev, VkPipelineLayout layout) {
    if (!layout) return false;
    pthread_mutex_lock(&g_rt_job_lock);
    struct RtLayoutPin **pp = rt_pin_find_locked(dev, layout);
    if (!*pp) {
        struct RtLayoutPin *n = (struct RtLayoutPin *)calloc(1, sizeof(*n));
        if (!n) { pthread_mutex_unlock(&g_rt_job_lock); return false; }
        n->device = dev; n->layout = layout;
        *pp = n;
    }
    (*pp)->pipe_refs++;
    pthread_mutex_unlock(&g_rt_job_lock);
    return true;
}
static void rt_unpin_pipeline_layout(VkDevice dev, VkPipelineLayout layout) {
    pthread_mutex_lock(&g_rt_job_lock);
    struct RtLayoutPin **pp = rt_pin_find_locked(dev, layout);
    struct RtLayoutPin *gone = NULL;
    if (*pp && (*pp)->pipe_refs) {
        (*pp)->pipe_refs--;
        gone = rt_pin_maybe_unlink_locked(pp);
    }
    pthread_cond_broadcast(&g_rt_job_cv);
    pthread_mutex_unlock(&g_rt_job_lock);
    rt_pin_dispose(gone, "its RT pipeline was destroyed");
}
/* Registers an async job + pins its layout. false = out of memory (caller compiles synchronously). */
static bool rt_job_register(struct RtCompileJob *j) {
    pthread_mutex_lock(&g_rt_job_lock);
    if (j->cpci.layout) {
        struct RtLayoutPin **pp = rt_pin_find_locked(j->device, j->cpci.layout);
        if (!*pp) {
            struct RtLayoutPin *n = (struct RtLayoutPin *)calloc(1, sizeof(*n));
            if (!n) { pthread_mutex_unlock(&g_rt_job_lock); return false; }
            n->device = j->device; n->layout = j->cpci.layout;
            *pp = n;
        }
        (*pp)->refs++;
    }
    j->next = g_rt_jobs;
    g_rt_jobs = j;
    pthread_mutex_unlock(&g_rt_job_lock);
    return true;
}
/* Ends a job: unpins the layout (destroys it if the app already did and no pipeline holds it), THEN removes the job
 * and wakes waiters. The caller frees j afterwards; nothing of the device may be used after this returns. */
static void rt_job_end(struct RtCompileJob *j) {
    struct RtLayoutPin *gone = NULL;
    pthread_mutex_lock(&g_rt_job_lock);
    if (j->cpci.layout) {
        struct RtLayoutPin **pp = rt_pin_find_locked(j->device, j->cpci.layout);
        if (*pp && (*pp)->refs) {
            (*pp)->refs--;
            gone = rt_pin_maybe_unlink_locked(pp);
        }
    }
    pthread_mutex_unlock(&g_rt_job_lock);
    rt_pin_dispose(gone, "the megakernel compile ended");
    pthread_mutex_lock(&g_rt_job_lock);
    for (struct RtCompileJob **pp = &g_rt_jobs; *pp; pp = &(*pp)->next)
        if (*pp == j) { *pp = j->next; break; }
    pthread_cond_broadcast(&g_rt_job_cv);
    pthread_mutex_unlock(&g_rt_job_lock);
}
/* compat_DestroyDevice: wait until no async compile of this device is left (the worker's last driver call done), then
 * drop every layout pin of the device (pipelines the app leaked): deferred layout destroys run now, before the
 * device goes away, so a pin can never outlive its device. */
static void rt_drain_device_jobs(VkDevice dev) {
    pthread_mutex_lock(&g_rt_job_lock);
    for (;;) {
        bool busy = false;
        for (struct RtCompileJob *j = g_rt_jobs; j && !busy; j = j->next) busy = (j->device == dev);
        if (!busy) break;
        LOG("rt_compile: vkDestroyDevice %p waits for a running megakernel compile", (void *)dev);
        pthread_cond_wait(&g_rt_job_cv, &g_rt_job_lock);
    }
    struct RtLayoutPin *mine = NULL;
    for (struct RtLayoutPin **pp = &g_rt_layout_pins; *pp;) {
        if ((*pp)->device == dev) {
            struct RtLayoutPin *n = *pp;
            *pp = n->next;
            n->next = mine;
            mine = n;
        } else {
            pp = &(*pp)->next;
        }
    }
    pthread_cond_broadcast(&g_rt_job_cv);
    pthread_mutex_unlock(&g_rt_job_lock);
    while (mine) {
        struct RtLayoutPin *n = mine;
        mine = n->next;
        rt_pin_dispose(n, "vkDestroyDevice");
    }
}
/* 1110b (Codex 1110-202009 #2): no DeviceData* is kept across the wait (g_devices[] entries move when another device
 * is destroyed meanwhile): the driver entry point is copied first, the device handle is the caller's.
 * NULL allocator + pinned -> deferred (the last job / the pipeline's destruction destroys it).
 * App allocator + pinned -> wait for running compile jobs only (bounded); if an RT pipeline still holds the layout
 * afterwards, the destroy is deferred with a COPY of the app's VkAllocationCallbacks (waiting for the pipeline could
 * deadlock the app thread). vkd3d-proton passes NULL. */
static VKAPI_ATTR void VKAPI_CALL
compat_RtDestroyPipelineLayout(VkDevice device, VkPipelineLayout layout, const VkAllocationCallbacks *pAllocator) {
    struct DeviceData *d = get_device_data(device);
    if (!d || !d->fpDestroyPipelineLayout) return;
    PFN_vkDestroyPipelineLayout fp = d->fpDestroyPipelineLayout;
    d = NULL;   /* not used past this point */
    if (layout) {
        pthread_mutex_lock(&g_rt_job_lock);
        struct RtLayoutPin *n = *rt_pin_find_locked(device, layout);
        if (n && !pAllocator) {
            n->fp_destroy = fp;
            /* lead-1110c (Codex 1110-212923 :890): read the pin BEFORE unlocking - the worker may free it right after */
            const bool by_job = n->refs != 0;
            pthread_mutex_unlock(&g_rt_job_lock);
            LOG("rt_compile: vkDestroyPipelineLayout %p deferred: %s still uses it", (void *)layout,
                by_job ? "a megakernel compile" : "an RT pipeline");
            return;
        }
        while ((n = *rt_pin_find_locked(device, layout)) != NULL && n->refs)
            pthread_cond_wait(&g_rt_job_cv, &g_rt_job_lock);
        if (n) {   /* only RT pipelines hold it now */
            n->fp_destroy = fp;
            n->has_alloc = true;
            n->alloc = *pAllocator;
            pthread_mutex_unlock(&g_rt_job_lock);
            LOG("rt_compile: vkDestroyPipelineLayout %p (app allocator) deferred until its RT pipeline is destroyed",
                (void *)layout);
            return;
        }
        pthread_mutex_unlock(&g_rt_job_lock);
    }
    fp(device, layout, pAllocator);
}
/* round 4: MALI_COMPAT_RT_STATS=1: one line per executable of a freshly created megakernel pipeline, with every
 * statistic the driver reports (name = value as given; bool/int64/uint64/float64). */
static void rt_log_pipeline_stats(struct RtCompileJob *j, VkPipeline p, int64_t dt_ms) {
    if (!j->fpGetPipelineExecutablePropertiesKHR || !j->fpGetPipelineExecutableStatisticsKHR || !p) return;
    VkPipelineInfoKHR pi = { .sType = VK_STRUCTURE_TYPE_PIPELINE_INFO_KHR, .pipeline = p };
    uint32_t ne = 0;
    VkResult r = j->fpGetPipelineExecutablePropertiesKHR(j->device, &pi, &ne, NULL);
    if (r != VK_SUCCESS || ne == 0) {
        LOG("rt_stats: pipe=%p stages=%u spv_bytes=%zu compile_ms=%lld executables=0 (vkGetPipelineExecutablePropertiesKHR "
            "result=%d)", (void *)j->rp, j->stages, j->words * 4, (long long)dt_ms, r);
        return;
    }
    if (ne > 16) ne = 16;
    VkPipelineExecutablePropertiesKHR props[16];
    for (uint32_t e = 0; e < ne; e++)
        props[e] = (VkPipelineExecutablePropertiesKHR){ .sType = VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_PROPERTIES_KHR };
    r = j->fpGetPipelineExecutablePropertiesKHR(j->device, &pi, &ne, props);
    if (r != VK_SUCCESS && r != VK_INCOMPLETE) ne = 0;
    for (uint32_t e = 0; e < ne; e++) {
        VkPipelineExecutableInfoKHR ei = { .sType = VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_INFO_KHR, .pipeline = p,
                                           .executableIndex = e };
        uint32_t ns = 0;
        VkResult rs = j->fpGetPipelineExecutableStatisticsKHR(j->device, &ei, &ns, NULL);
        VkPipelineExecutableStatisticKHR *st = (rs == VK_SUCCESS && ns) ? calloc(ns, sizeof(*st)) : NULL;
        if (st) {
            for (uint32_t k = 0; k < ns; k++) st[k].sType = VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_STATISTIC_KHR;
            rs = j->fpGetPipelineExecutableStatisticsKHR(j->device, &ei, &ns, st);
            if (rs != VK_SUCCESS && rs != VK_INCOMPLETE) ns = 0;
        } else {
            ns = 0;
        }
        char buf[3072];
        size_t n = 0;
        for (uint32_t k = 0; k < ns && n + 8 < sizeof buf; k++) {
            int w = 0;
            switch (st[k].format) {
            case VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_BOOL32_KHR:
                w = snprintf(buf + n, sizeof buf - n, " | %s=%s", st[k].name, st[k].value.b32 ? "true" : "false"); break;
            case VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_INT64_KHR:
                w = snprintf(buf + n, sizeof buf - n, " | %s=%lld", st[k].name, (long long)st[k].value.i64); break;
            case VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_UINT64_KHR:
                w = snprintf(buf + n, sizeof buf - n, " | %s=%llu", st[k].name, (unsigned long long)st[k].value.u64); break;
            case VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_FLOAT64_KHR:
                w = snprintf(buf + n, sizeof buf - n, " | %s=%g", st[k].name, st[k].value.f64); break;
            default:
                w = snprintf(buf + n, sizeof buf - n, " | %s=?(format %d)", st[k].name, (int)st[k].format); break;
            }
            if (w < 0) break;
            n += (size_t)w;
            if (n >= sizeof buf) { n = sizeof buf - 1; break; }
        }
        buf[n] = '\0';
        LOG("rt_stats: pipe=%p stages=%u spv_bytes=%zu compile_ms=%lld exec=%u/%u name=\"%s\" subgroup=%u stats=%u%s",
            (void *)j->rp, j->stages, j->words * 4, (long long)dt_ms, e + 1, ne, props[e].name, props[e].subgroupSize,
            ns, buf);
        free(st);
    }
}
static void rt_compile_finish(struct RtCompileJob *j, VkResult r, VkPipeline p, const char *where) {
    int64_t t1 = rt_now_ms();
    if (j->async) atomic_fetch_sub_explicit(&g_mega_compiling, 1, memory_order_relaxed);
    if (r == VK_SUCCESS && p) {
        rt_log_pipeline_stats(j, p, t1 - j->t_req);
        atomic_fetch_add_explicit(&g_mega_live, 1, memory_order_relaxed);
        __atomic_store_n(&j->rp->compute_pipeline, p, __ATOMIC_RELEASE);
        __atomic_store_n(&j->rp->compile_state, 2, __ATOMIC_RELEASE);
    } else {
        atomic_fetch_add_explicit(&g_mega_failed, 1, memory_order_relaxed);
        __atomic_store_n(&j->rp->compile_state, 3, __ATOMIC_RELEASE);
        LOG("RT megakernel vkCreateComputePipelines failed (%d) for %u stages / %u groups", r, j->stages, j->groups);
    }
    LOG("CRTP %s: fpCreateComputePipelines result=%d pipe=%p (%u stages, %lld ms)", where, r, (void *)p,
        j->stages, (long long)(t1 - j->t_req));
    if (g_debug)
        LOG("rt_compile: done t=%lld ms dt=%lld ms result=%d pipe=%p stages=%u words=%zu (%s, tid %ld)",
            (long long)t1, (long long)(t1 - j->t_req), r, (void *)j->rp, j->stages, j->words, where,
            (long)syscall(SYS_gettid));
}
static void rt_free_pipeline_obj_fp(VkDevice dev, PFN_vkDestroyPipeline fp_dp, PFN_vkDestroyShaderModule fp_dm,
                                    struct RtPipeline *victim, const VkAllocationCallbacks *pAllocator);
static pthread_mutex_t g_rt_split_slot_lock = PTHREAD_MUTEX_INITIALIZER;   /* split-1112 child compile limiter */
static pthread_cond_t  g_rt_split_slot_cv   = PTHREAD_COND_INITIALIZER;
static uint32_t        g_rt_split_slots_used;
static void *rt_compile_worker(void *arg) {
    struct RtCompileJob *j = (struct RtCompileJob *)arg;
    if (j->cpci.pNext) j->cpci.pNext = &j->f2;
    if (g_debug)
        LOG("rt_compile: worker start t=%lld ms (queued %lld ms) pipe=%p stages=%u words=%zu tid %ld",
            (long long)rt_now_ms(), (long long)(rt_now_ms() - j->t_req), (void *)j->rp, j->stages, j->words,
            (long)syscall(SYS_gettid));
    VkPipeline p = VK_NULL_HANDLE;
    /* split-1112: at most MALI_COMPAT_RT_SPLIT_JOBS split-child compiles run at once (peak compiler RAM/CPU during the
     * load); unsplit pipelines are not limited (as before) */
    const bool slot = j->rp->split_child;
    if (slot) {
        pthread_mutex_lock(&g_rt_split_slot_lock);
        while (g_rt_split_slots_used >= rt_env()->split_jobs) pthread_cond_wait(&g_rt_split_slot_cv, &g_rt_split_slot_lock);
        g_rt_split_slots_used++;
        pthread_mutex_unlock(&g_rt_split_slot_lock);
    }
    /* NULL cache + NULL allocator: the app may destroy its cache/allocator while we compile. */
    VkResult r = j->fpCreateComputePipelines(j->device, VK_NULL_HANDLE, 1, &j->cpci, NULL, &p);
    if (slot) {
        pthread_mutex_lock(&g_rt_split_slot_lock);
        g_rt_split_slots_used--;
        pthread_cond_signal(&g_rt_split_slot_cv);
        pthread_mutex_unlock(&g_rt_split_slot_lock);
    }
    rt_compile_finish(j, r, p, "worker");
    struct RtPipeline *rp = j->rp;
    /* round 4 lever MALI_COMPAT_RT_FREE_STAGES=1: the megakernel module is not needed once the compute pipeline exists
     * (Vulkan allows destroying a module after pipeline creation). Done BEFORE the handshake below, so neither path
     * can see a half-released module; the NULL allocator matches what rt_free_pipeline_obj_fp uses for async compiles. */
    if (j->release_module && rp->compute_module && j->fpDestroyShaderModule) {
        j->fpDestroyShaderModule(j->device, rp->compute_module, NULL);
        rp->compute_module = VK_NULL_HANDLE;
        if (rp->mega_bytes) rt_acct_sub(&g_acct_mega_mod, (int64_t)rp->mega_bytes);
    }
    /* 11-08 keeper: DestroyPipeline no longer joins a pending compile (it blocked the app thread for up to the
     * whole compile, 112 s for the 80-stage W3 kernel). Whoever comes second (worker or destroy) frees the object. */
    if (atomic_fetch_add(&rp->compile_handshake, 1) == 1) {
        LOG("rt_compile: pipeline %p was destroyed while compiling; worker frees it now", (void *)rp);
        rt_free_pipeline_obj_fp(j->device, j->fpDestroyPipeline, j->fpDestroyShaderModule, rp, NULL);
    }
    rt_job_end(j);   /* last: after this DestroyDevice may proceed */
    free(j);
    return NULL;
}

/* split-1112: release the split children of a parent that is being freed. Same worker handshake as
 * rt_destroy_pipeline: a child whose megakernel still compiles is freed by its worker (the second to arrive frees). */
static void rt_split_release_children(VkDevice dev, PFN_vkDestroyPipeline fp_dp, PFN_vkDestroyShaderModule fp_dm,
                                      struct RtPipeline *rp, const VkAllocationCallbacks *pAllocator) {
    for (uint32_t k = 0; k < rp->split_n; k++) {
        struct RtPipeline *c = rp->split[k];
        rp->split[k] = NULL;
        if (!c) continue;
        if (c->compile_thread_live) {
            c->compile_thread_live = false;
            pthread_detach(c->compile_thread);
            if (atomic_fetch_add(&c->compile_handshake, 1) == 0) {
                LOG("rt_split: child %p still compiling while its pipeline is freed; its worker frees it", (void *)c);
                continue;
            }
        }
        rt_free_pipeline_obj_fp(dev, fp_dp, fp_dm, c, pAllocator);
    }
    rp->split_n = 0;
}
static void rt_free_pipeline_obj_fp(VkDevice dev, PFN_vkDestroyPipeline fp_dp, PFN_vkDestroyShaderModule fp_dm,
                                    struct RtPipeline *victim, const VkAllocationCallbacks *pAllocator) {
    if (victim->split_n) rt_split_release_children(dev, fp_dp, fp_dm, victim, pAllocator);
    if (victim->compile_null_alloc) pAllocator = NULL;
    if (victim->compute_pipeline && fp_dp) {
        fp_dp(dev, victim->compute_pipeline, pAllocator);
    }
    if (victim->compute_module && fp_dm) {
        fp_dm(dev, victim->compute_module, NULL);   /* round 4: always created with NULL callbacks */
    }
    for (uint32_t i = 0; i < victim->stage_count; i++) {
        rt_free_stage_record(&victim->stages[i]);
    }
    if (victim->layout_pinned) rt_unpin_pipeline_layout(victim->device, victim->layout);
    /* round 4 accounting: the megakernel module (unless already released by MALI_COMPAT_RT_FREE_STAGES) and pipeline */
    if (victim->compute_module && victim->mega_bytes) rt_acct_sub(&g_acct_mega_mod, (int64_t)victim->mega_bytes);
    if (victim->compute_pipeline) atomic_fetch_sub_explicit(&g_mega_live, 1, memory_order_relaxed);
    else if (victim->compile_state == 3) atomic_fetch_sub_explicit(&g_mega_failed, 1, memory_order_relaxed);
    else if (victim->compile_state == 4) atomic_fetch_sub_explicit(&g_mega_capped, 1, memory_order_relaxed);
    if (victim->acct_bytes) rt_acct_sub(&g_acct_pipes, (int64_t)victim->acct_bytes);
    free(victim->stages);
    free(victim->groups);
    free(victim);
}
static void rt_free_pipeline_obj(struct DeviceData *d, struct RtPipeline *victim, const VkAllocationCallbacks *pAllocator) {
    rt_free_pipeline_obj_fp(d ? d->device : VK_NULL_HANDLE, d ? d->fpDestroyPipeline : NULL,
                            d ? d->fpDestroyShaderModule : NULL, victim, pAllocator);
}

static bool rt_destroy_pipeline(struct DeviceData *d, VkPipeline handle, const VkAllocationCallbacks *pAllocator) {
    if (!handle) return false;
    unsigned b = rt_bucket((uintptr_t)handle);
    pthread_mutex_lock(&g_rt_lock);
    struct RtPipeline **pp = &g_rt_pipelines[b];
    while (*pp) {
        if ((VkPipeline)(uintptr_t)(*pp) == handle) {
            struct RtPipeline *victim = *pp;
            *pp = victim->next;
            pthread_mutex_unlock(&g_rt_lock);
            if (victim->compile_thread_live) {
                victim->compile_thread_live = false;
                pthread_detach(victim->compile_thread);
                if (atomic_fetch_add(&victim->compile_handshake, 1) == 0) {
                    LOG("rt_compile: DestroyPipeline %p while its megakernel still compiles; not waiting (worker frees it)",
                        (void *)victim);
                    return true;
                }
            }
            rt_free_pipeline_obj(d, victim, pAllocator);
            return true;
        }
        pp = &(*pp)->next;
    }
    pthread_mutex_unlock(&g_rt_lock);
    return false;
}

static void rt_record_layout_pc_stages(VkPipelineLayout layout, VkShaderStageFlags stages) {
    if (!layout) return;
    unsigned b = rt_bucket((uintptr_t)layout);
    pthread_mutex_lock(&g_rt_lock);
    for (struct RtLayoutNode *n = g_rt_layouts[b]; n; n = n->next) {
        if (n->layout == layout) {
            n->pc_stages = stages;
            pthread_mutex_unlock(&g_rt_lock);
            return;
        }
    }
    struct RtLayoutNode *n = (struct RtLayoutNode *)calloc(1, sizeof(*n));
    if (n) {
        n->layout = layout;
        n->pc_stages = stages;
        n->next = g_rt_layouts[b];
        g_rt_layouts[b] = n;
    }
    pthread_mutex_unlock(&g_rt_lock);
}

static VkShaderStageFlags rt_get_layout_pc_stages(VkPipelineLayout layout) {
    if (!layout) return VK_SHADER_STAGE_COMPUTE_BIT;
    unsigned b = rt_bucket((uintptr_t)layout);
    pthread_mutex_lock(&g_rt_lock);
    for (struct RtLayoutNode *n = g_rt_layouts[b]; n; n = n->next) {
        if (n->layout == layout) {
            VkShaderStageFlags s = n->pc_stages;
            pthread_mutex_unlock(&g_rt_lock);
            return s ? s : VK_SHADER_STAGE_COMPUTE_BIT;
        }
    }
    pthread_mutex_unlock(&g_rt_lock);
    return VK_SHADER_STAGE_COMPUTE_BIT;
}

static struct RtCmdState *rt_get_cmd_state(VkCommandBuffer cmd, bool create) {
    unsigned b = rt_bucket((uintptr_t)cmd);
    pthread_mutex_lock(&g_rt_lock);
    for (struct RtCmdState *s = g_rt_cmd_states[b]; s; s = s->next) {
        if (s->cmd == cmd) {
            pthread_mutex_unlock(&g_rt_lock);
            return s;
        }
    }
    if (!create) {
        pthread_mutex_unlock(&g_rt_lock);
        return NULL;
    }
    struct RtCmdState *s = (struct RtCmdState *)calloc(1, sizeof(*s));
    if (s) {
        s->cmd = cmd;
        s->next = g_rt_cmd_states[b];
        g_rt_cmd_states[b] = s;
        rt_acct_add(&g_acct_cmdst, (int64_t)sizeof(*s));
    }
    pthread_mutex_unlock(&g_rt_lock);
    return s;
}


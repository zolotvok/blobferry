/*
 * mali_rt_emu.h — VK_KHR_ray_tracing_pipeline Emulation over VK_KHR_ray_query
 *
 * Included directly by mali_compat_layer.c (after init_bc_compute).
 *
 * Architecture:
 *   1. Intercepts RT VkShaderModules before they reach the vendor Mali driver
 *      (which lacks OpCapability RayTracingKHR) and stores their SPIR-V in
 *      RtShaderModule nodes.
 *   2. Supports VK_KHR_pipeline_library, VkPipelineLibraryCreateInfoKHR,
 *      VK_EXT_pipeline_library_group_handles, VK_KHR_ray_tracing_maintenance1,
 *      and VK_KHR_deferred_host_operations.
 *   3. Lowers and links RT shader stages (rgen, rchit, rahit, rmiss, rint, rcall)
 *      into a single GLCompute megakernel (LocalSize 8x4x1 = 32 threads = 1 Mali
 *      warp) backed by hardware OpTypeRayQueryKHR / OpRayQueryProceedKHR.
 *   4. Encodes deterministic 32-byte position-independent RtGroupHandle records
 *      into the application's SBT buffers so the megakernel reads stage_ids and
 *      ShaderRecordBufferKHR local root arguments directly from SBT GPU memory.
 */

#ifndef MALI_RT_EMU_H
#define MALI_RT_EMU_H

#include <link.h>
#include <elf.h>

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
    e = getenv("MALI_COMPAT_RT_SIZE_PAD");     g_rt_env.size_pad     = (e && e[0] == '0') ? 0 : 1;
    e = getenv("MALI_COMPAT_RT_AS_TRUST");     g_rt_env.as_trust     = (e && e[0] == '1') ? 1 : 0;
    e = getenv("MALI_COMPAT_RT_UPD_SRC");      g_rt_env.upd_src      = (e && (e[0] == '0' || e[0] == '2')) ? e[0] - '0' : 1;
    e = getenv("MALI_COMPAT_RT_CLONE");        g_rt_env.clone        = (e && e[0] == '0') ? 0 : 1;
    e = getenv("MALI_COMPAT_RT_CLONE_REFIT");  g_rt_env.clone_refit  = (e && e[0] == '0') ? 0 : 1;
    e = getenv("MALI_COMPAT_RT_GUARD_ITERS");
    int gv = e ? atoi(e) : 0;
    g_rt_guard_max = gv > 0 ? (uint32_t)gv : 4096u;   /* was a racy lazy write in CRTP */
    if (g_rt_env.skip_traces)
        LOG("rt_skip: MALI_COMPAT_RT_SKIP_TRACES=1 -> every TraceRays/TraceRaysIndirect is skipped (AS builds kept)");
    LOG("rt_scratch: layer-owned AS build scratch %s (MALI_COMPAT_RT_OWN_SCRATCH), pool %u MB",
        g_rt_env.own_scratch ? "ON" : "OFF", (unsigned)(RT_AS_SCRATCH_SIZE >> 20));
    if (g_rt_env.force_build)
        LOG("rt_build: MALI_COMPAT_RT_FORCE_BUILD=1 -> every UPDATE build becomes a full BUILD (src cleared)");
    if (!g_rt_env.size_pad)
        LOG("rt_build: MALI_COMPAT_RT_SIZE_PAD=0 -> vkGetAccelerationStructureBuildSizesKHR returns the driver's own sizes "
            "(no 1.5x+2 MiB / 2x+4 MiB inflation, no 64 KB padding)");
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
    if (g_rt_env.build_log)
        LOG("rt_build: MALI_COMPAT_RT_BUILD_LOG=%u -> per-build lines for the first %u calls and %u calls after the "
            "first TraceRays", g_rt_env.build_log, g_rt_env.build_log, g_rt_env.build_log);
}
static inline const struct RtEnv *rt_env(void) {
    pthread_once(&g_rt_env_once, rt_env_init);
    return &g_rt_env;
}
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

static void rt_free_stage_record(struct RtStageRecord *s) {
    if (!s) return;
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
static _Atomic uint64_t g_rt_skip_pending = 0;   /* compile still running on the worker */
static _Atomic uint64_t g_rt_skip_nopipe  = 0;   /* capped (MALI_COMPAT_RT_MAX_STAGES) or compile failed */
static void rt_count_skip(struct RtPipeline *rp) {
    int st = __atomic_load_n(&rp->compile_state, __ATOMIC_ACQUIRE);
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
static void rt_compile_finish(struct RtCompileJob *j, VkResult r, VkPipeline p, const char *where) {
    int64_t t1 = rt_now_ms();
    if (r == VK_SUCCESS && p) {
        __atomic_store_n(&j->rp->compute_pipeline, p, __ATOMIC_RELEASE);
        __atomic_store_n(&j->rp->compile_state, 2, __ATOMIC_RELEASE);
    } else {
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
static void *rt_compile_worker(void *arg) {
    struct RtCompileJob *j = (struct RtCompileJob *)arg;
    if (j->cpci.pNext) j->cpci.pNext = &j->f2;
    if (g_debug)
        LOG("rt_compile: worker start t=%lld ms (queued %lld ms) pipe=%p stages=%u words=%zu tid %ld",
            (long long)rt_now_ms(), (long long)(rt_now_ms() - j->t_req), (void *)j->rp, j->stages, j->words,
            (long)syscall(SYS_gettid));
    VkPipeline p = VK_NULL_HANDLE;
    /* NULL cache + NULL allocator: the app may destroy its cache/allocator while we compile. */
    VkResult r = j->fpCreateComputePipelines(j->device, VK_NULL_HANDLE, 1, &j->cpci, NULL, &p);
    rt_compile_finish(j, r, p, "worker");
    /* 11-08 keeper: DestroyPipeline no longer joins a pending compile (it blocked the app thread for up to the
     * whole compile, 112 s for the 80-stage W3 kernel). Whoever comes second (worker or destroy) frees the object. */
    struct RtPipeline *rp = j->rp;
    if (atomic_fetch_add(&rp->compile_handshake, 1) == 1) {
        LOG("rt_compile: pipeline %p was destroyed while compiling; worker frees it now", (void *)rp);
        rt_free_pipeline_obj_fp(j->device, j->fpDestroyPipeline, j->fpDestroyShaderModule, rp, NULL);
    }
    rt_job_end(j);   /* last: after this DestroyDevice may proceed */
    free(j);
    return NULL;
}

static void rt_free_pipeline_obj_fp(VkDevice dev, PFN_vkDestroyPipeline fp_dp, PFN_vkDestroyShaderModule fp_dm,
                                    struct RtPipeline *victim, const VkAllocationCallbacks *pAllocator) {
    if (victim->compile_null_alloc) pAllocator = NULL;
    if (victim->compute_pipeline && fp_dp) {
        fp_dp(dev, victim->compute_pipeline, pAllocator);
    }
    if (victim->compute_module && fp_dm) {
        fp_dm(dev, victim->compute_module, victim->compile_null_alloc ? NULL : pAllocator);
    }
    for (uint32_t i = 0; i < victim->stage_count; i++) {
        rt_free_stage_record(&victim->stages[i]);
    }
    if (victim->layout_pinned) rt_unpin_pipeline_layout(victim->device, victim->layout);
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
    }
    pthread_mutex_unlock(&g_rt_lock);
    return s;
}

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

/* ── Copy Stage Helper for Pipeline & Library Creation ────────────────────── */

static bool rt_copy_stage_from_create_info(const VkPipelineShaderStageCreateInfo *sci,
                                           struct RtStageRecord *out) {
    memset(out, 0, sizeof(*out));
    out->stage = sci->stage;
    out->entry_name = strdup(sci->pName ? sci->pName : "main");

    const uint32_t *spv_words = NULL;
    size_t spv_word_count = 0;

    struct RtShaderModule *rtm = rt_lookup_shader_module(sci->module);
    if (rtm) {
        spv_words = rtm->words;
        spv_word_count = rtm->word_count;
    } else {
        /* Support VK_KHR_maintenance5 inline VkShaderModuleCreateInfo in pNext */
        const VkShaderModuleCreateInfo *smci = (const VkShaderModuleCreateInfo *)
            find_struct(sci->pNext, VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO);
        if (smci && smci->pCode && smci->codeSize >= 20) {
            spv_words = smci->pCode;
            spv_word_count = smci->codeSize / 4;
        }
    }

    if (!spv_words || spv_word_count < 5) {
        free(out->entry_name);
        memset(out, 0, sizeof(*out));
        return false;
    }

    out->word_count = spv_word_count;
    out->words = (uint32_t *)malloc(spv_word_count * sizeof(uint32_t));
    if (!out->words) {
        free(out->entry_name);
        memset(out, 0, sizeof(*out));
        return false;
    }
    memcpy(out->words, spv_words, spv_word_count * sizeof(uint32_t));

    if (sci->pSpecializationInfo &&
        sci->pSpecializationInfo->mapEntryCount > 0 &&
        sci->pSpecializationInfo->pMapEntries &&
        sci->pSpecializationInfo->pData &&
        sci->pSpecializationInfo->dataSize > 0) {
        uint32_t ec = sci->pSpecializationInfo->mapEntryCount;
        out->spec_entries = (struct RtSpecConstantEntry *)calloc(ec, sizeof(struct RtSpecConstantEntry));
        out->spec_data = (uint8_t *)malloc(sci->pSpecializationInfo->dataSize);
        if (out->spec_entries && out->spec_data) {
            out->spec_entry_count = ec;
            out->spec_data_size = sci->pSpecializationInfo->dataSize;
            memcpy(out->spec_data, sci->pSpecializationInfo->pData, out->spec_data_size);
            for (uint32_t i = 0; i < ec; i++) {
                out->spec_entries[i].spec_id = sci->pSpecializationInfo->pMapEntries[i].constantID;
                out->spec_entries[i].offset  = sci->pSpecializationInfo->pMapEntries[i].offset;
                out->spec_entries[i].size    = (uint32_t)sci->pSpecializationInfo->pMapEntries[i].size;
            }
        }
    }

    out->stage_id = rt_compute_stage_id(out->words, out->word_count, out->stage,
                                        out->entry_name, sci->pSpecializationInfo);
    return true;
}

static bool rt_clone_stage_record(const struct RtStageRecord *src, struct RtStageRecord *dst) {
    memset(dst, 0, sizeof(*dst));
    dst->stage = src->stage;
    dst->stage_id = src->stage_id;
    dst->word_count = src->word_count;
    dst->entry_name = strdup(src->entry_name ? src->entry_name : "main");
    dst->words = (uint32_t *)malloc(src->word_count * sizeof(uint32_t));
    if (!dst->words || !dst->entry_name) {
        rt_free_stage_record(dst);
        return false;
    }
    memcpy(dst->words, src->words, src->word_count * sizeof(uint32_t));
    if (src->spec_entry_count > 0 && src->spec_entries && src->spec_data) {
        dst->spec_entries = (struct RtSpecConstantEntry *)calloc(src->spec_entry_count, sizeof(*dst->spec_entries));
        dst->spec_data = (uint8_t *)malloc(src->spec_data_size);
        if (dst->spec_entries && dst->spec_data) {
            dst->spec_entry_count = src->spec_entry_count;
            dst->spec_data_size = src->spec_data_size;
            memcpy(dst->spec_entries, src->spec_entries, src->spec_entry_count * sizeof(*dst->spec_entries));
            memcpy(dst->spec_data, src->spec_data, src->spec_data_size);
        }
    }
    return true;
}

/* ── Vulkan Ray Tracing Pipeline Entry Points ─────────────────────────────── */

static VKAPI_ATTR VkResult VKAPI_CALL
compat_CreateRayTracingPipelinesKHR(
    VkDevice                                    device,
    VkDeferredOperationKHR                      deferredOperation,
    VkPipelineCache                             pipelineCache,
    uint32_t                                    createInfoCount,
    const VkRayTracingPipelineCreateInfoKHR    *pCreateInfos,
    const VkAllocationCallbacks                *pAllocator,
    VkPipeline                                 *pPipelines)
{
    (void)deferredOperation;
    rt_patch_fex_thunks();
    COMPAT_TRACE("compat_CreateRayTracingPipelinesKHR: count=%u", createInfoCount);
    struct DeviceData *d = get_device_data(device);
    if (!d || !pCreateInfos || !pPipelines) return VK_ERROR_INITIALIZATION_FAILED;

    VkResult final_res = VK_SUCCESS;

    for (uint32_t i = 0; i < createInfoCount; i++) {
        pPipelines[i] = VK_NULL_HANDLE;
        const VkRayTracingPipelineCreateInfoKHR *ci = &pCreateInfos[i];

        /* Check pipeline creation flags (including VkPipelineCreateFlags2CreateInfoKHR) */
        uint64_t flags2 = (uint64_t)ci->flags;
        const VkPipelineCreateFlags2CreateInfoKHR *f2 = (const VkPipelineCreateFlags2CreateInfoKHR *)
            find_struct(ci->pNext, VK_STRUCTURE_TYPE_PIPELINE_CREATE_FLAGS_2_CREATE_INFO_KHR);
        if (f2) flags2 |= f2->flags;

        bool is_library = (flags2 & 0x00000800ULL /* VK_PIPELINE_CREATE_LIBRARY_BIT_KHR */) != 0;
        bool use_desc_buf = (flags2 & 0x20000000ULL /* VK_PIPELINE_CREATE_DESCRIPTOR_BUFFER_BIT_EXT */) != 0;

        /* Count stages & groups from this createInfo + linked libraries */
        uint32_t total_stages = ci->stageCount;
        uint32_t total_groups = ci->groupCount;
        if (ci->pLibraryInfo && ci->pLibraryInfo->libraryCount > 0 && ci->pLibraryInfo->pLibraries) {
            for (uint32_t l = 0; l < ci->pLibraryInfo->libraryCount; l++) {
                struct RtPipeline *lib = rt_lookup_pipeline(ci->pLibraryInfo->pLibraries[l]);
                if (lib) {
                    total_stages += lib->stage_count;
                    total_groups += lib->group_count;
                    if (lib->use_descriptor_buffer) use_desc_buf = true;
                }
            }
        }

        COMPAT_INFO("CRTP [%u/%u]: stages=%u groups=%u layout=%p libInfo=%p",
                    i + 1, createInfoCount, ci->stageCount, ci->groupCount, (void *)ci->layout, (void *)ci->pLibraryInfo);

        struct RtPipeline *rp = (struct RtPipeline *)calloc(1, sizeof(*rp));
        if (!rp) {
            final_res = VK_ERROR_OUT_OF_HOST_MEMORY;
            continue;
        }
        rp->magic = 0x5254504Cu; /* 'RTPL' */
        rp->device = device;
        rp->is_library = is_library;
        rp->use_descriptor_buffer = use_desc_buf;
        rp->layout = ci->layout;
        if (!is_library && ci->layout) {
            /* lead-1110c (Codex 1110-212923 :4837): no pin = no pipeline (an unpinned one could later record with a
             * destroyed layout). Nothing else is allocated for rp yet. */
            rp->layout_pinned = rt_pin_pipeline_layout(device, ci->layout);
            if (!rp->layout_pinned) {
                LOG("CRTP [%u/%u]: out of host memory pinning layout %p -> VK_ERROR_OUT_OF_HOST_MEMORY",
                    i + 1, createInfoCount, (void *)ci->layout);
                free(rp);
                final_res = VK_ERROR_OUT_OF_HOST_MEMORY;
                continue;
            }
        }
        rp->max_recursion_depth = ci->maxPipelineRayRecursionDepth ? ci->maxPipelineRayRecursionDepth : 1;

        if (total_stages > 0) {
            rp->stages = (struct RtStageRecord *)calloc(total_stages, sizeof(*rp->stages));
        }
        if (total_groups > 0) {
            rp->groups = (struct RtGroupHandle *)calloc(total_groups, sizeof(*rp->groups));
        }

        /* 1. Copy local stages */
        bool stage_ok = true;
        for (uint32_t s = 0; s < ci->stageCount; s++) {
            COMPAT_INFO("CRTP [%u/%u]:   stage %u: stage=0x%x module=%p pName=%s spec=%p",
                        i + 1, createInfoCount, s, ci->pStages[s].stage, (void *)ci->pStages[s].module,
                        ci->pStages[s].pName ? ci->pStages[s].pName : "(null)",
                        (void *)ci->pStages[s].pSpecializationInfo);
            if (!rt_copy_stage_from_create_info(&ci->pStages[s], &rp->stages[rp->stage_count])) {
                COMPAT_INFO("CRTP [%u/%u]:     stage %u copy FAILED", i + 1, createInfoCount, s);
                stage_ok = false;
                break;
            }
            rp->stage_count++;
        }
        COMPAT_INFO("CRTP [%u/%u]: copied %u stages, stage_ok=%d", i + 1, createInfoCount, rp->stage_count, (int)stage_ok);

        /* 2. Build local group handles from ci->pGroups */
        for (uint32_t g = 0; g < ci->groupCount && stage_ok; g++) {
            const VkRayTracingShaderGroupCreateInfoKHR *gi = &ci->pGroups[g];
            struct RtGroupHandle *gh = &rp->groups[rp->group_count++];
            memset(gh, 0, sizeof(*gh));
            gh->magic = RT_GROUP_HANDLE_MAGIC;
            gh->type  = (uint32_t)gi->type;
            if (gi->generalShader != VK_SHADER_UNUSED_KHR && gi->generalShader < ci->stageCount) {
                gh->general_id = rp->stages[gi->generalShader].stage_id;
            }
            if (gi->closestHitShader != VK_SHADER_UNUSED_KHR && gi->closestHitShader < ci->stageCount) {
                gh->rchit_id = rp->stages[gi->closestHitShader].stage_id;
            }
            if (gi->anyHitShader != VK_SHADER_UNUSED_KHR && gi->anyHitShader < ci->stageCount) {
                gh->rahit_id = rp->stages[gi->anyHitShader].stage_id;
            }
            if (gi->intersectionShader != VK_SHADER_UNUSED_KHR && gi->intersectionShader < ci->stageCount) {
                gh->rint_id = rp->stages[gi->intersectionShader].stage_id;
            }
        }

        /* 3. Append stages and group handles from linked pipeline libraries */
        if (ci->pLibraryInfo && ci->pLibraryInfo->libraryCount > 0 && ci->pLibraryInfo->pLibraries && stage_ok) {
            for (uint32_t l = 0; l < ci->pLibraryInfo->libraryCount; l++) {
                struct RtPipeline *lib = rt_lookup_pipeline(ci->pLibraryInfo->pLibraries[l]);
                if (!lib) continue;
                for (uint32_t s = 0; s < lib->stage_count; s++) {
                    if (!rt_clone_stage_record(&lib->stages[s], &rp->stages[rp->stage_count])) {
                        stage_ok = false;
                        break;
                    }
                    rp->stage_count++;
                }
                for (uint32_t g = 0; g < lib->group_count; g++) {
                    rp->groups[rp->group_count++] = lib->groups[g];
                }
            }
        }

        /* 4. If executable pipeline (!is_library), build and compile the GLCompute megakernel */
        COMPAT_INFO("CRTP [%u/%u]: building megakernel (is_lib=%d, stage_ok=%d, stages=%u)...",
                    i + 1, createInfoCount, (int)is_library, (int)stage_ok, rp->stage_count);
        const int rt_max_stages = rt_env()->max_stages;   /* also sets g_rt_guard_max (once) */
        if (!is_library && stage_ok && rt_max_stages > 0 && rp->stage_count > (uint32_t)rt_max_stages) {
            /* A/B lever: no megakernel for this pipeline; its trace dispatches are skipped and counted. */
            rp->compile_state = 4;
            COMPAT_INFO("CRTP [%u/%u]: %u stages > MALI_COMPAT_RT_MAX_STAGES=%d -> megakernel NOT built, traces skipped",
                        i + 1, createInfoCount, rp->stage_count, rt_max_stages);
        } else if (!is_library && stage_ok && rp->stage_count > 0) {
            size_t mega_words = 0;
            int64_t t_build0 = rt_now_ms();
            if (!__atomic_load_n(&g_rt_guard_bda, __ATOMIC_ACQUIRE) && rt_ensure_ring_buffer(d)) {
                VkDeviceAddress zero = 0ULL;   /* first device wins (was a racy plain write) */
                __atomic_compare_exchange_n(&g_rt_guard_bda, &zero, d->rt_ring_bda + RT_GUARD_SLOT_OFF, false,
                                            __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE);
            }
            uint32_t *mega_spv = rt_build_megakernel_spirv(rp->stages, rp->stage_count, &mega_words);
            if (g_debug)
                LOG("rt_compile: SPIR-V built t=%lld ms dt=%lld ms stages=%u words=%zu",
                    (long long)rt_now_ms(), (long long)(rt_now_ms() - t_build0), rp->stage_count, mega_words);
            COMPAT_INFO("CRTP [%u/%u]: megakernel SPIR-V built (%zu words, %p)",
                        i + 1, createInfoCount, mega_words, (void *)mega_spv);
            if (mega_spv && mega_words > 5) {
                const char *dump_dir = getenv("MALI_COMPAT_DUMP_RT_SPV");
                if (!dump_dir || !dump_dir[0]) dump_dir = "/tmp";
                char path[512];
                uint64_t h = spv_hash(mega_spv, mega_words * 4);
                snprintf(path, sizeof(path), "%s/rt_mega_p%u_%016llx.spv", dump_dir, i, (unsigned long long)h);
                FILE *fp = fopen(path, "wb");
                if (fp) {
                    fwrite(mega_spv, 4, mega_words, fp);
                    fclose(fp);
                    COMPAT_INFO("CRTP [%u/%u]: dumped megakernel to %s", i + 1, createInfoCount, path);
                }

                VkShaderModuleCreateInfo smci = {
                    .sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
                    .codeSize = mega_words * sizeof(uint32_t),
                    .pCode    = mega_spv,
                };
                COMPAT_INFO("CRTP [%u/%u]: calling fpCreateShaderModule...", i + 1, createInfoCount);
                VkResult sm_res = d->fpCreateShaderModule(device, &smci, pAllocator, &rp->compute_module);
                COMPAT_INFO("CRTP [%u/%u]: fpCreateShaderModule result=%d module=%p", i + 1, createInfoCount, sm_res, (void *)rp->compute_module);
                free(mega_spv);

                if (sm_res == VK_SUCCESS && rp->compute_module) {
                    VkPipelineCreateFlags comp_flags = 0;
                    if (rp->use_descriptor_buffer) {
                        comp_flags |= 0x20000000u /* VK_PIPELINE_CREATE_DESCRIPTOR_BUFFER_BIT_EXT */;
                    }
                    VkPipelineCreateFlags2CreateInfoKHR comp_flags2 = {
                        .sType = VK_STRUCTURE_TYPE_PIPELINE_CREATE_FLAGS_2_CREATE_INFO_KHR,
                        .flags = (uint64_t)comp_flags,
                    };
                    VkComputePipelineCreateInfo cpci = {
                        .sType  = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
                        .pNext  = rp->use_descriptor_buffer ? &comp_flags2 : NULL,
                        .flags  = comp_flags,
                        .stage  = {
                            .sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                            .stage  = VK_SHADER_STAGE_COMPUTE_BIT,
                            .module = rp->compute_module,
                            .pName  = "main",
                        },
                        .layout = rp->layout,
                    };
                    struct RtCompileJob *job = (struct RtCompileJob *)calloc(1, sizeof(*job));
                    const int sync_compile = rt_env()->sync_compile;
                    if (job) {
                        job->device = device; job->rp = rp; job->cpci = cpci; job->f2 = comp_flags2;
                        job->fpCreateComputePipelines = d->fpCreateComputePipelines;
                        job->fpDestroyPipeline = d->fpDestroyPipeline;
                        job->fpDestroyShaderModule = d->fpDestroyShaderModule;
                        job->stages = rp->stage_count; job->groups = rp->group_count; job->words = mega_words;
                        job->t_req = rt_now_ms();
                    }
                    if (g_debug)
                        LOG("rt_compile: request t=%lld ms pipe=%p stages=%u groups=%u words=%zu mode=%s",
                            (long long)rt_now_ms(), (void *)rp, rp->stage_count, rp->group_count, mega_words,
                            (sync_compile || !job) ? "sync" : "async");
                    bool queued = false;
                    if (job && !sync_compile && rt_job_register(job)) {
                        rp->compile_state = 1;
                        rp->compile_null_alloc = true;
                        /* the job is fully initialised before the thread starts; the worker owns it */
                        if (pthread_create(&rp->compile_thread, NULL, rt_compile_worker, job) == 0) {
                            rp->compile_thread_live = true;
                            queued = true;
                            COMPAT_INFO("CRTP [%u/%u]: megakernel compile queued on a worker (%u stages)",
                                        i + 1, createInfoCount, rp->stage_count);
                        } else {
                            rt_job_end(job);   /* unpin + unlist; the layout cannot be app-destroyed during CRTP */
                            rp->compile_state = 0;
                            rp->compile_null_alloc = false;
                        }
                    }
                    if (!queued) {
                        COMPAT_INFO("CRTP [%u/%u]: calling fpCreateComputePipelines...", i + 1, createInfoCount);
                        VkPipeline cp = VK_NULL_HANDLE;
                        VkResult cp_res = d->fpCreateComputePipelines(device, pipelineCache, 1, &cpci,
                                                                      pAllocator, &cp);
                        COMPAT_INFO("CRTP [%u/%u]: fpCreateComputePipelines result=%d pipe=%p", i + 1, createInfoCount, cp_res, (void *)cp);
                        if (job) {
                            rt_compile_finish(job, cp_res, cp, "sync");
                            free(job);
                        } else if (cp_res == VK_SUCCESS) {
                            rp->compute_pipeline = cp;
                            rp->compile_state = 2;
                        }
                    }
                } else {
                    COMPAT_INFO("RT megakernel vkCreateShaderModule failed (%d)", sm_res);
                }
            } else {
                free(mega_spv);
                COMPAT_INFO("RT megakernel SPIR-V builder failed for %u stages", rp->stage_count);
            }
        }

        VkPipeline handle = (VkPipeline)(uintptr_t)rp;
        unsigned b = rt_bucket((uintptr_t)handle);
        pthread_mutex_lock(&g_rt_lock);
        rp->next = g_rt_pipelines[b];
        g_rt_pipelines[b] = rp;
        pthread_mutex_unlock(&g_rt_lock);

        pPipelines[i] = handle;
    }

    return final_res;
}

static VKAPI_ATTR VkResult VKAPI_CALL
compat_GetRayTracingShaderGroupHandlesKHR(
    VkDevice                                    device,
    VkPipeline                                  pipeline,
    uint32_t                                    firstGroup,
    uint32_t                                    groupCount,
    size_t                                      dataSize,
    void                                       *pData)
{
    (void)device;
    rt_patch_fex_thunks();
    if (!pData) return VK_ERROR_INITIALIZATION_FAILED;
    struct RtPipeline *rp = rt_lookup_pipeline(pipeline);
    COMPAT_TRACE("compat_GetRayTracingShaderGroupHandlesKHR: pipeline=%p firstGroup=%u groupCount=%u dataSize=%zu pData=%p (rp=%p, rp->group_count=%u)",
                 (void *)pipeline, firstGroup, groupCount, dataSize, pData, (void *)rp, rp ? rp->group_count : 0);
    if (!rp) {
        memset(pData, 0, dataSize);
        return VK_SUCCESS;
    }
    uint8_t *dst = (uint8_t *)pData;
    for (uint32_t i = 0; i < groupCount; i++) {
        if ((size_t)(i + 1) * RT_GROUP_HANDLE_SIZE > dataSize) break;
        uint32_t gidx = firstGroup + i;
        if (gidx < rp->group_count) {
            memcpy(dst + i * RT_GROUP_HANDLE_SIZE, &rp->groups[gidx], RT_GROUP_HANDLE_SIZE);
        } else {
            memset(dst + i * RT_GROUP_HANDLE_SIZE, 0, RT_GROUP_HANDLE_SIZE);
        }
    }
    return VK_SUCCESS;
}

static VKAPI_ATTR VkResult VKAPI_CALL
compat_GetRayTracingCaptureReplayShaderGroupHandlesKHR(
    VkDevice                                    device,
    VkPipeline                                  pipeline,
    uint32_t                                    firstGroup,
    uint32_t                                    groupCount,
    size_t                                      dataSize,
    void                                       *pData)
{
    return compat_GetRayTracingShaderGroupHandlesKHR(device, pipeline, firstGroup, groupCount, dataSize, pData);
}

static VKAPI_ATTR VkDeviceSize VKAPI_CALL
compat_GetRayTracingShaderGroupStackSizeKHR(
    VkDevice                                    device,
    VkPipeline                                  pipeline,
    uint32_t                                    group,
    VkShaderGroupShaderKHR                      groupShader)
{
    (void)device; (void)pipeline; (void)group; (void)groupShader;
    return 256;
}

static VKAPI_ATTR void VKAPI_CALL
compat_CmdSetRayTracingPipelineStackSizeKHR(
    VkCommandBuffer                             commandBuffer,
    uint32_t                                    pipelineStackSize)
{
    (void)commandBuffer; (void)pipelineStackSize;
}

/* ── TraceRays Dispatch Execution & Persistent Ring Buffer ─────────────────── */

#define RT_RING_SIZE (4u * 1024u * 1024u)
#define RT_RING_SLOT_SIZE 256u
#define RT_RING_NUM_SLOTS (RT_RING_SIZE / RT_RING_SLOT_SIZE) /* 16384 slots */

static bool rt_ensure_ring_buffer(struct DeviceData *d) {
    if (!d) return false;
    if (d->rt_ring_bda != 0ULL && d->rt_ring_ptr != NULL) return true;

    pthread_mutex_lock(&d->bc_lock);
    if (d->rt_ring_bda != 0ULL && d->rt_ring_ptr != NULL) {
        pthread_mutex_unlock(&d->bc_lock);
        return true;
    }

    if (!d->fpCreateBuffer || !d->fpAllocateMemory || !d->fpBindBufferMemory ||
        !d->fpGetBufferDeviceAddress || !d->fpMapMemory) {
        pthread_mutex_unlock(&d->bc_lock);
        return false;
    }

    VkBufferCreateInfo bci = {
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .size  = RT_RING_SIZE,
        .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT |
                 VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
    };
    VkBuffer buf = VK_NULL_HANDLE;
    if (d->fpCreateBuffer(d->device, &bci, NULL, &buf) != VK_SUCCESS) {
        pthread_mutex_unlock(&d->bc_lock);
        return false;
    }

    VkMemoryRequirements req = {0};
    d->fpGetBufferMemoryRequirements(d->device, buf, &req);
    uint32_t mask = req.memoryTypeBits;
    if (mask & d->coherent_types) {
        mask &= d->coherent_types;
    }
    uint32_t mtype = mask ? (uint32_t)__builtin_ctz(mask) : 0u;

    VkMemoryAllocateFlagsInfo mafi = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO,
        .flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT,
    };
    VkMemoryAllocateInfo mai = {
        .sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        .pNext           = &mafi,
        .allocationSize  = req.size < RT_RING_SIZE ? RT_RING_SIZE : req.size,
        .memoryTypeIndex = mtype,
    };
    VkDeviceMemory mem = VK_NULL_HANDLE;
    if (d->fpAllocateMemory(d->device, &mai, NULL, &mem) != VK_SUCCESS) {
        d->fpDestroyBuffer(d->device, buf, NULL);
        pthread_mutex_unlock(&d->bc_lock);
        return false;
    }

    if (d->fpBindBufferMemory(d->device, buf, mem, 0) != VK_SUCCESS) {
        d->fpFreeMemory(d->device, mem, NULL);
        d->fpDestroyBuffer(d->device, buf, NULL);
        pthread_mutex_unlock(&d->bc_lock);
        return false;
    }

    void *ptr = NULL;
    if (d->fpMapMemory(d->device, mem, 0, RT_RING_SIZE, 0, &ptr) != VK_SUCCESS || !ptr) {
        d->fpFreeMemory(d->device, mem, NULL);
        d->fpDestroyBuffer(d->device, buf, NULL);
        pthread_mutex_unlock(&d->bc_lock);
        return false;
    }
    memset(ptr, 0, RT_RING_SIZE);

    VkBufferDeviceAddressInfo bdai = {
        .sType  = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO,
        .buffer = buf,
    };
    VkDeviceAddress bda = d->fpGetBufferDeviceAddress(d->device, &bdai);
    if (!bda) {
        d->fpUnmapMemory(d->device, mem);
        d->fpFreeMemory(d->device, mem, NULL);
        d->fpDestroyBuffer(d->device, buf, NULL);
        pthread_mutex_unlock(&d->bc_lock);
        return false;
    }

    d->rt_ring_buf = buf;
    d->rt_ring_mem = mem;
    d->rt_ring_bda = bda;
    d->rt_ring_ptr = ptr;
    atomic_store(&d->rt_ring_offset, 0u);
    pthread_mutex_unlock(&d->bc_lock);
    COMPAT_INFO("rt_ensure_ring_buffer: 4MB persistent ring allocated at BDA 0x%lx host=%p",
                (unsigned long)bda, ptr);
    return true;
}

static VkDeviceAddress rt_ring_alloc(struct DeviceData *d,
                                     uint32_t byte_size,
                                     uint32_t align,
                                     void **out_ptr,
                                     uint32_t *out_offset) {
    if (!rt_ensure_ring_buffer(d)) return 0ULL;
    (void)align;
    uint32_t slots_needed = (byte_size + RT_RING_SLOT_SIZE - 1u) / RT_RING_SLOT_SIZE;
    if (slots_needed == 0u) slots_needed = 1u;
    /* last slot = guard counter (RT_GUARD_SLOT_OFF), never handed out */
    if (slots_needed > RT_RING_NUM_SLOTS - 1u) return 0ULL;
    uint32_t max_first_slot = RT_RING_NUM_SLOTS - 1u - slots_needed;
    uint32_t raw_slot = atomic_fetch_add(&d->rt_ring_offset, slots_needed);
    uint32_t slot = raw_slot % (max_first_slot + 1u);
    uint32_t off = slot * RT_RING_SLOT_SIZE;
    if (out_ptr) {
        *out_ptr = (void *)((uintptr_t)d->rt_ring_ptr + off);
    }
    if (out_offset) {
        *out_offset = off;
    }
    return d->rt_ring_bda + (VkDeviceAddress)off;
}

static void rt_guard_log(struct DeviceData *d) {
    static _Atomic int64_t last_ns = 0;
    static uint32_t last_val = 0;
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    int64_t now = (int64_t)ts.tv_sec * 1000000000LL + ts.tv_nsec;
    int64_t prev = atomic_load(&last_ns);
    if (now - prev < 1000000000LL) return;
    if (!atomic_compare_exchange_strong(&last_ns, &prev, now)) return;
    uint32_t v = *(volatile uint32_t *)((uint8_t *)d->rt_ring_ptr + RT_GUARD_SLOT_OFF);
    COMPAT_INFO("rt_guard: %u rays cut as miss after %u ray-query iterations (+%u in the last 1 s)",
                v, g_rt_guard_max, v - last_val);
    last_val = v;
}

static VkDeviceAddress rt_get_zero_bda(struct DeviceData *d) {
    if (!d) return 0ULL;
    if (d->rt_zero_bda != 0ULL) return d->rt_zero_bda;

    pthread_mutex_lock(&d->bc_lock);
    if (d->rt_zero_bda != 0ULL) {
        pthread_mutex_unlock(&d->bc_lock);
        return d->rt_zero_bda;
    }

    if (!d->fpCreateBuffer || !d->fpAllocateMemory || !d->fpBindBufferMemory ||
        !d->fpGetBufferDeviceAddress) {
        pthread_mutex_unlock(&d->bc_lock);
        return 0ULL;
    }

    VkBufferCreateInfo bci = {
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .size  = 4096,
        .usage = VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
    };
    VkBuffer buf = VK_NULL_HANDLE;
    if (d->fpCreateBuffer(d->device, &bci, NULL, &buf) != VK_SUCCESS) {
        pthread_mutex_unlock(&d->bc_lock);
        return 0ULL;
    }

    VkMemoryRequirements req = {0};
    d->fpGetBufferMemoryRequirements(d->device, buf, &req);
    uint32_t mask = req.memoryTypeBits;
    if (mask & d->coherent_types) {
        mask &= d->coherent_types;
    }
    uint32_t mtype = mask ? (uint32_t)__builtin_ctz(mask) : 0u;

    VkMemoryAllocateFlagsInfo mafi = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO,
        .flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT,
    };
    VkMemoryAllocateInfo mai = {
        .sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        .pNext           = &mafi,
        .allocationSize  = req.size,
        .memoryTypeIndex = mtype,
    };
    VkDeviceMemory mem = VK_NULL_HANDLE;
    if (d->fpAllocateMemory(d->device, &mai, NULL, &mem) != VK_SUCCESS) {
        d->fpDestroyBuffer(d->device, buf, NULL);
        pthread_mutex_unlock(&d->bc_lock);
        return 0ULL;
    }

    if (d->fpBindBufferMemory(d->device, buf, mem, 0) != VK_SUCCESS) {
        d->fpFreeMemory(d->device, mem, NULL);
        d->fpDestroyBuffer(d->device, buf, NULL);
        pthread_mutex_unlock(&d->bc_lock);
        return 0ULL;
    }

    if (d->fpMapMemory && d->fpUnmapMemory) {
        void *ptr = NULL;
        if (d->fpMapMemory(d->device, mem, 0, 4096, 0, &ptr) == VK_SUCCESS && ptr) {
            memset(ptr, 0, 4096);
            d->fpUnmapMemory(d->device, mem);
        }
    }

    VkBufferDeviceAddressInfo bdai = {
        .sType  = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO,
        .buffer = buf,
    };
    VkDeviceAddress bda = d->fpGetBufferDeviceAddress(d->device, &bdai);

    d->rt_zero_buf = buf;
    d->rt_zero_mem = mem;
    d->rt_zero_bda = bda;

    COMPAT_INFO("RT_EMU: initialized persistent 4KB zero scratch buffer: bda=0x%llx", (unsigned long long)bda);
    pthread_mutex_unlock(&d->bc_lock);
    return bda;
}

static VkDeviceAddress rt_upload_user_push_constants(struct DeviceData *d,
                                                     VkCommandBuffer cmd,
                                                     const struct RtCmdState *cs) {
    (void)cmd;
    if (!cs || cs->user_pc_dirty_size == 0) {
        return rt_get_zero_bda(d);
    }
    if (g_debug && d && d->rt_ring_ptr) rt_guard_log(d);
    void *ptr = NULL;
    uint32_t off = 0;
    VkDeviceAddress bda = rt_ring_alloc(d, 256, 256, &ptr, &off);
    if (bda && d->fpCmdUpdateBuffer && d->rt_ring_buf && d->fpCmdPipelineBarrier &&
        !getenv("MALI_COMPAT_RT_PARAM_HOSTWRITE")) {
        /* param-fix 11-08: the per-dispatch params (vkd3d root constants + root descriptor VAs, incl. the
         * TLAS VA) are written by the GPU in command-stream order right before this dispatch, instead of a
         * host memcpy at record time into a slot that the ring hands out again later (07cd745).  The data
         * now travels inside the command buffer: a resubmitted / late-submitted / still-pending command
         * buffer always reads its own params, no matter how often the ring wrapped. */
        d->fpCmdUpdateBuffer(cmd, d->rt_ring_buf, (VkDeviceSize)off, 256, cs->user_push_constants);
        VkMemoryBarrier ub = {
            .sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
            .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
            .dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_UNIFORM_READ_BIT,
        };
        d->fpCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                0, 1, &ub, 0, NULL, 0, NULL);
        return bda;
    }
    if (bda && ptr) {   /* fallback / A-B lever MALI_COMPAT_RT_PARAM_HOSTWRITE=1: old host write */
        memcpy(ptr, cs->user_push_constants, 256);
        return bda;
    }
    return rt_get_zero_bda(d);
}

static void rt_prepare_and_bind_megakernel(struct DeviceData *d,
                                           VkCommandBuffer cmd,
                                           struct RtCmdState *cs,
                                           struct RtPipeline *rp,
                                           const struct RtDispatchPushConstants *pc) {
    /* Bind compiled megakernel to COMPUTE bind point */
    d->fpCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, rp->compute_pipeline);

    /* If descriptor buffer offsets were set on RAY_TRACING_KHR, ensure COMPUTE has them */
    if (cs && cs->has_rt_desc_buf_offsets && d->fpCmdSetDescriptorBufferOffsetsEXT) {
        d->fpCmdSetDescriptorBufferOffsetsEXT(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                              cs->rt_db_layout ? cs->rt_db_layout : rp->layout,
                                              cs->rt_db_first_set, cs->rt_db_set_count,
                                              cs->rt_db_buf_indices, cs->rt_db_offsets);
    }

    /* Push 96-byte RtDispatchPushConstants */
    VkShaderStageFlags pc_stages = rt_get_layout_pc_stages(rp->layout);
    d->fpCmdPushConstants(cmd, rp->layout, pc_stages, 0, sizeof(*pc), pc);
}

static void rt_restore_post_trace_state(struct DeviceData *d,
                                        VkCommandBuffer cmd,
                                        struct RtCmdState *cs,
                                        struct RtPipeline *rp) {
    if (!cs) return;
    /* Restore user push constants (first 96 bytes) if dirty */
    if (cs->user_pc_dirty_size > 0) {
        VkShaderStageFlags pc_stages = rt_get_layout_pc_stages(rp->layout);
        d->fpCmdPushConstants(cmd, rp->layout, pc_stages, 0,
                              sizeof(struct RtDispatchPushConstants), cs->user_push_constants);
    }
    /* Restore previous compute pipeline if one was bound */
    if (cs->bound_comp_pipeline) {
        d->fpCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, cs->bound_comp_pipeline);
    }
}

static VKAPI_ATTR void VKAPI_CALL
compat_CmdTraceRaysKHR(
    VkCommandBuffer                             commandBuffer,
    const VkStridedDeviceAddressRegionKHR      *pRaygenShaderBindingTable,
    const VkStridedDeviceAddressRegionKHR      *pMissShaderBindingTable,
    const VkStridedDeviceAddressRegionKHR      *pHitShaderBindingTable,
    const VkStridedDeviceAddressRegionKHR      *pCallableShaderBindingTable,
    uint32_t                                    width,
    uint32_t                                    height,
    uint32_t                                    depth)
{
    rt_patch_fex_thunks();
    COMPAT_TRACE("compat_CmdTraceRaysKHR: cmd=%p width=%u height=%u depth=%u",
                 (void *)commandBuffer, width, height, depth);
    rt_note_trace();   /* MALI_COMPAT_RT_BUILD_LOG: marks the first recorded TraceRays */
    if (width == 0 || height == 0 || depth == 0) return;
    struct DeviceData *d = get_device_data_by_cmd(commandBuffer);
    if (!d) return;
    struct RtCmdState *cs = rt_get_cmd_state(commandBuffer, false);
    if (!cs || !cs->bound_rt_pipeline) return;
    if (rt_forced_trace_skip()) return;
    if (!rt_mega_ready(cs->bound_rt_pipeline)) { rt_count_skip(cs->bound_rt_pipeline); return; }
    struct RtPipeline *rp = cs->bound_rt_pipeline;

    struct RtDispatchPushConstants pc;
    memset(&pc, 0, sizeof(pc));
    if (pRaygenShaderBindingTable) {
        VkDeviceAddress addr = pRaygenShaderBindingTable->deviceAddress;
        if (rt_is_invalid_gpu_bda(addr) || addr == 0ULL) {
            pc.rgen_base   = rt_get_zero_bda(d);
            pc.rgen_stride = 0;
            pc.rgen_size   = 0;
        } else {
            pc.rgen_base   = addr;
            pc.rgen_stride = (uint32_t)pRaygenShaderBindingTable->stride;
            pc.rgen_size   = (uint32_t)pRaygenShaderBindingTable->size;
        }
    }
    if (pMissShaderBindingTable) {
        VkDeviceAddress addr = pMissShaderBindingTable->deviceAddress;
        if (rt_is_invalid_gpu_bda(addr) || addr == 0ULL) {
            pc.miss_base   = rt_get_zero_bda(d);
            pc.miss_stride = 0;
            pc.miss_size   = 0;
        } else {
            pc.miss_base   = addr;
            pc.miss_stride = (uint32_t)pMissShaderBindingTable->stride;
            pc.miss_size   = (uint32_t)pMissShaderBindingTable->size;
        }
    }
    if (pHitShaderBindingTable) {
        VkDeviceAddress addr = pHitShaderBindingTable->deviceAddress;
        if (rt_is_invalid_gpu_bda(addr) || addr == 0ULL) {
            pc.hit_base    = rt_get_zero_bda(d);
            pc.hit_stride  = 0;
            pc.hit_size    = 0;
        } else {
            pc.hit_base    = addr;
            pc.hit_stride  = (uint32_t)pHitShaderBindingTable->stride;
            pc.hit_size    = (uint32_t)pHitShaderBindingTable->size;
        }
    }
    if (pCallableShaderBindingTable) {
        VkDeviceAddress addr = pCallableShaderBindingTable->deviceAddress;
        if (rt_is_invalid_gpu_bda(addr) || addr == 0ULL) {
            pc.call_base   = rt_get_zero_bda(d);
            pc.call_stride = 0;
            pc.call_size   = 0;
        } else {
            pc.call_base   = addr;
            pc.call_stride = (uint32_t)pCallableShaderBindingTable->stride;
            pc.call_size   = (uint32_t)pCallableShaderBindingTable->size;
        }
    }
    pc.launch_width    = width;
    pc.launch_height   = height;
    pc.launch_depth    = depth;
    pc.indirect_mode   = 0u;
    pc.push_consts_bda = rt_upload_user_push_constants(d, commandBuffer, cs);
    pc.indirect_bda    = 0ULL;

    static uint32_t g_trace_log_count = 0;
    uint32_t trace_idx = g_trace_log_count++;
    if (trace_idx < 30 || (trace_idx % 120) == 0) {
        COMPAT_INFO("CmdTraceRays #%u: dim=%ux%ux%u rgen=%lx:sz%u:str%u miss=%lx:sz%u:str%u hit=%lx:sz%u:str%u call=%lx:sz%u:str%u pc_bda=%lx",
                    trace_idx, width, height, depth,
                    (unsigned long)pc.rgen_base, pc.rgen_size, pc.rgen_stride,
                    (unsigned long)pc.miss_base, pc.miss_size, pc.miss_stride,
                    (unsigned long)pc.hit_base, pc.hit_size, pc.hit_stride,
                    (unsigned long)pc.call_base, pc.call_size, pc.call_stride,
                    (unsigned long)pc.push_consts_bda);
    }

    if (d->fpCmdPipelineBarrier) {
        VkMemoryBarrier full_barrier = {
            .sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
            .srcAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT |
                             VK_ACCESS_HOST_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT |
                             VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR,
            .dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT |
                             VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT |
                             VK_ACCESS_UNIFORM_READ_BIT | VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR,
        };
        d->fpCmdPipelineBarrier(commandBuffer,
                                VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                                VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                                0,
                                1, &full_barrier,
                                0, NULL,
                                0, NULL);
    }

    rt_prepare_and_bind_megakernel(d, commandBuffer, cs, rp, &pc);

    uint32_t gx = (width  + 7u) / 8u;
    uint32_t gy = (height + 3u) / 4u;
    uint32_t gz = depth;
    d->fpCmdDispatch(commandBuffer, gx, gy, gz);

    /* Post-trace memory barrier: ensure compute writes to storage buffers/images
     * are coherent for subsequent compute dispatches and graphics reads */
    if (d->fpCmdPipelineBarrier) {
        VkMemoryBarrier post_bar = {
            .sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
            .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_MEMORY_WRITE_BIT,
            .dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT |
                             VK_ACCESS_UNIFORM_READ_BIT | VK_ACCESS_MEMORY_READ_BIT,
        };
        d->fpCmdPipelineBarrier(commandBuffer,
                                VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                                VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                                0,
                                1, &post_bar,
                                0, NULL,
                                0, NULL);
    }

    rt_restore_post_trace_state(d, commandBuffer, cs, rp);
}

static void rt_ensure_indirect_setup_pipeline(struct DeviceData *d) {
    if (d->rt_ind_pipeline) return;
    pthread_mutex_lock(&d->bc_lock);
    if (d->rt_ind_pipeline) {
        pthread_mutex_unlock(&d->bc_lock);
        return;
    }
    size_t wc = 0;
    uint32_t *spv = rt_build_indirect_setup_spirv(&wc);
    if (spv && wc > 5) {
        VkPushConstantRange pcr = {
            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
            .offset     = 0,
            .size       = 16,
        };
        VkPipelineLayoutCreateInfo plci = {
            .sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
            .pushConstantRangeCount = 1,
            .pPushConstantRanges    = &pcr,
        };
        if (d->fpCreatePipelineLayout(d->device, &plci, NULL, &d->rt_ind_layout) == VK_SUCCESS) {
            VkShaderModuleCreateInfo smci = {
                .sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
                .codeSize = wc * sizeof(uint32_t),
                .pCode    = spv,
            };
            VkShaderModule mod = VK_NULL_HANDLE;
            if (d->fpCreateShaderModule(d->device, &smci, NULL, &mod) == VK_SUCCESS) {
                VkComputePipelineCreateInfo cpci = {
                    .sType  = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
                    .stage  = {
                        .sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                        .stage  = VK_SHADER_STAGE_COMPUTE_BIT,
                        .module = mod,
                        .pName  = "main",
                    },
                    .layout = d->rt_ind_layout,
                };
                d->fpCreateComputePipelines(d->device, VK_NULL_HANDLE, 1, &cpci, NULL, &d->rt_ind_pipeline);
                d->fpDestroyShaderModule(d->device, mod, NULL);
            }
        }
    }
    free(spv);
    pthread_mutex_unlock(&d->bc_lock);
}

static void rt_dispatch_indirect_common(
    VkCommandBuffer                             commandBuffer,
    const VkStridedDeviceAddressRegionKHR      *pRaygenShaderBindingTable,
    const VkStridedDeviceAddressRegionKHR      *pMissShaderBindingTable,
    const VkStridedDeviceAddressRegionKHR      *pHitShaderBindingTable,
    const VkStridedDeviceAddressRegionKHR      *pCallableShaderBindingTable,
    VkDeviceAddress                             indirectDeviceAddress,
    uint32_t                                    indirect_mode)
{
    rt_note_trace();
    if (!indirectDeviceAddress || rt_is_invalid_gpu_bda(indirectDeviceAddress)) return;
    struct DeviceData *d = get_device_data_by_cmd(commandBuffer);
    if (!d) return;
    struct RtCmdState *cs = rt_get_cmd_state(commandBuffer, false);
    if (!cs || !cs->bound_rt_pipeline) return;
    if (rt_forced_trace_skip()) return;
    if (!rt_mega_ready(cs->bound_rt_pipeline)) { rt_count_skip(cs->bound_rt_pipeline); return; }
    struct RtPipeline *rp = cs->bound_rt_pipeline;

    struct RtDispatchPushConstants pc;
    memset(&pc, 0, sizeof(pc));
    if (pRaygenShaderBindingTable) {
        VkDeviceAddress addr = pRaygenShaderBindingTable->deviceAddress;
        if (rt_is_invalid_gpu_bda(addr) || addr == 0ULL) {
            pc.rgen_base   = rt_get_zero_bda(d);
            pc.rgen_stride = 0;
            pc.rgen_size   = 0;
        } else {
            pc.rgen_base   = addr;
            pc.rgen_stride = (uint32_t)pRaygenShaderBindingTable->stride;
            pc.rgen_size   = (uint32_t)pRaygenShaderBindingTable->size;
        }
    }
    if (pMissShaderBindingTable) {
        VkDeviceAddress addr = pMissShaderBindingTable->deviceAddress;
        if (rt_is_invalid_gpu_bda(addr) || addr == 0ULL) {
            pc.miss_base   = rt_get_zero_bda(d);
            pc.miss_stride = 0;
            pc.miss_size   = 0;
        } else {
            pc.miss_base   = addr;
            pc.miss_stride = (uint32_t)pMissShaderBindingTable->stride;
            pc.miss_size   = (uint32_t)pMissShaderBindingTable->size;
        }
    }
    if (pHitShaderBindingTable) {
        VkDeviceAddress addr = pHitShaderBindingTable->deviceAddress;
        if (rt_is_invalid_gpu_bda(addr) || addr == 0ULL) {
            pc.hit_base    = rt_get_zero_bda(d);
            pc.hit_stride  = 0;
            pc.hit_size    = 0;
        } else {
            pc.hit_base    = addr;
            pc.hit_stride  = (uint32_t)pHitShaderBindingTable->stride;
            pc.hit_size    = (uint32_t)pHitShaderBindingTable->size;
        }
    }
    if (pCallableShaderBindingTable) {
        VkDeviceAddress addr = pCallableShaderBindingTable->deviceAddress;
        if (rt_is_invalid_gpu_bda(addr) || addr == 0ULL) {
            pc.call_base   = rt_get_zero_bda(d);
            pc.call_stride = 0;
            pc.call_size   = 0;
        } else {
            pc.call_base   = addr;
            pc.call_stride = (uint32_t)pCallableShaderBindingTable->stride;
            pc.call_size   = (uint32_t)pCallableShaderBindingTable->size;
        }
    }
    pc.indirect_mode   = indirect_mode;
    pc.push_consts_bda = rt_upload_user_push_constants(d, commandBuffer, cs);
    pc.indirect_bda    = indirectDeviceAddress;

    static uint32_t g_ind_trace_log_count = 0;
    if (g_ind_trace_log_count < 20 || (g_ind_trace_log_count % 300) == 0) {
        g_ind_trace_log_count++;
        COMPAT_INFO("CmdTraceRaysIndirect (mode %u) #%u: ind_bda=%lx rgen=%lx:sz%u miss=%lx:sz%u hit=%lx:sz%u pc_bda=%lx",
                    indirect_mode, g_ind_trace_log_count,
                    (unsigned long)indirectDeviceAddress,
                    (unsigned long)pc.rgen_base, pc.rgen_size,
                    (unsigned long)pc.miss_base, pc.miss_size,
                    (unsigned long)pc.hit_base, pc.hit_size,
                    (unsigned long)pc.push_consts_bda);
    }

    rt_ensure_indirect_setup_pipeline(d);

    uint32_t ind_off = 0;
    void *ind_ptr = NULL;
    VkDeviceAddress dst_bda = rt_ring_alloc(d, 64, 64, &ind_ptr, &ind_off);
    if (d->fpCmdPipelineBarrier) {
        VkMemoryBarrier full_barrier = {
            .sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
            .srcAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT |
                             VK_ACCESS_HOST_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT |
                             VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR,
            .dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT |
                             VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT |
                             VK_ACCESS_UNIFORM_READ_BIT | VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR,
        };
        d->fpCmdPipelineBarrier(commandBuffer,
                                VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                                VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                                0,
                                1, &full_barrier,
                                0, NULL,
                                0, NULL);
    }

    if (d->rt_ind_pipeline && d->fpCmdDispatchIndirect && dst_bda && d->rt_ring_buf) {
        uint64_t src_bda = indirectDeviceAddress + (indirect_mode == 2 ? 88ULL : 0ULL);
        uint64_t setup_pc[2] = { src_bda, dst_bda };

        d->fpCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, d->rt_ind_pipeline);
        d->fpCmdPushConstants(commandBuffer, d->rt_ind_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, 16, setup_pc);
        d->fpCmdDispatch(commandBuffer, 1, 1, 1);

        VkMemoryBarrier mb_bar = {
            .sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
            .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
            .dstAccessMask = VK_ACCESS_INDIRECT_COMMAND_READ_BIT,
        };
        d->fpCmdPipelineBarrier(commandBuffer,
                                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT,
                                0, 1, &mb_bar, 0, NULL, 0, NULL);

        rt_prepare_and_bind_megakernel(d, commandBuffer, cs, rp, &pc);
        d->fpCmdDispatchIndirect(commandBuffer, d->rt_ring_buf, (VkDeviceSize)ind_off);
    } else {
        rt_prepare_and_bind_megakernel(d, commandBuffer, cs, rp, &pc);
        d->fpCmdDispatch(commandBuffer, 256, 256, 1);
    }

    if (d->fpCmdPipelineBarrier) {
        VkMemoryBarrier post_bar = {
            .sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
            .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_MEMORY_WRITE_BIT,
            .dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT |
                             VK_ACCESS_UNIFORM_READ_BIT | VK_ACCESS_MEMORY_READ_BIT,
        };
        d->fpCmdPipelineBarrier(commandBuffer,
                                VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                                VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                                0, 1, &post_bar, 0, NULL, 0, NULL);
    }

    rt_restore_post_trace_state(d, commandBuffer, cs, rp);
}

static VKAPI_ATTR void VKAPI_CALL
compat_CmdTraceRaysIndirectKHR(
    VkCommandBuffer                             commandBuffer,
    const VkStridedDeviceAddressRegionKHR      *pRaygenShaderBindingTable,
    const VkStridedDeviceAddressRegionKHR      *pMissShaderBindingTable,
    const VkStridedDeviceAddressRegionKHR      *pHitShaderBindingTable,
    const VkStridedDeviceAddressRegionKHR      *pCallableShaderBindingTable,
    VkDeviceAddress                             indirectDeviceAddress)
{
    rt_dispatch_indirect_common(commandBuffer, pRaygenShaderBindingTable,
                                pMissShaderBindingTable, pHitShaderBindingTable,
                                pCallableShaderBindingTable, indirectDeviceAddress, 1u);
}

static VKAPI_ATTR void VKAPI_CALL
compat_CmdTraceRaysIndirect2KHR(
    VkCommandBuffer                             commandBuffer,
    VkDeviceAddress                             indirectDeviceAddress)
{
    rt_dispatch_indirect_common(commandBuffer, NULL, NULL, NULL, NULL,
                                indirectDeviceAddress, 2u);
}

/* ── Deferred Host Operations (Synchronous Host Fallback) ─────────────────── */

static VKAPI_ATTR VkResult VKAPI_CALL
compat_CreateDeferredHostOperationsKHR(
    VkDevice                                    device,
    const VkAllocationCallbacks                *pAllocator,
    VkDeferredOperationKHR                     *pDeferredOperation)
{
    (void)pAllocator;
    if (!pDeferredOperation) return VK_ERROR_INITIALIZATION_FAILED;
    struct RtDeferredOp *op = (struct RtDeferredOp *)calloc(1, sizeof(*op));
    if (!op) return VK_ERROR_OUT_OF_HOST_MEMORY;
    op->magic  = 0x5254444Fu; /* 'RTDO' */
    op->device = device;
    op->result = VK_SUCCESS;
    VkDeferredOperationKHR handle = (VkDeferredOperationKHR)(uintptr_t)op;
    unsigned b = rt_bucket((uintptr_t)handle);
    pthread_mutex_lock(&g_rt_lock);
    op->next = g_rt_deferred_ops[b];
    g_rt_deferred_ops[b] = op;
    pthread_mutex_unlock(&g_rt_lock);
    *pDeferredOperation = handle;
    return VK_SUCCESS;
}

static VKAPI_ATTR void VKAPI_CALL
compat_DestroyDeferredHostOperationKHR(
    VkDevice                                    device,
    VkDeferredOperationKHR                      operation,
    const VkAllocationCallbacks                *pAllocator)
{
    (void)device; (void)pAllocator;
    if (!operation) return;
    unsigned b = rt_bucket((uintptr_t)operation);
    pthread_mutex_lock(&g_rt_lock);
    struct RtDeferredOp **pp = &g_rt_deferred_ops[b];
    while (*pp) {
        if ((VkDeferredOperationKHR)(uintptr_t)(*pp) == operation) {
            struct RtDeferredOp *victim = *pp;
            *pp = victim->next;
            pthread_mutex_unlock(&g_rt_lock);
            free(victim);
            return;
        }
        pp = &(*pp)->next;
    }
    pthread_mutex_unlock(&g_rt_lock);
}

static VKAPI_ATTR uint32_t VKAPI_CALL
compat_GetDeferredHostOperationMaxConcurrencyKHR(
    VkDevice                                    device,
    VkDeferredOperationKHR                      operation)
{
    (void)device; (void)operation;
    return 1u;
}

static VKAPI_ATTR VkResult VKAPI_CALL
compat_GetDeferredHostOperationResultKHR(
    VkDevice                                    device,
    VkDeferredOperationKHR                      operation)
{
    (void)device; (void)operation;
    return VK_SUCCESS;
}

static VKAPI_ATTR VkResult VKAPI_CALL
compat_DeferredHostOperationJoinKHR(
    VkDevice                                    device,
    VkDeferredOperationKHR                      operation)
{
    (void)device; (void)operation;
    return VK_SUCCESS;
}

/* ── FEX Thunk Function Pointer Patching ─────────────────────────────────── */

static VKAPI_ATTR VkResult VKAPI_CALL compat_CreateAccelerationStructureKHR(VkDevice, const VkAccelerationStructureCreateInfoKHR*, const VkAllocationCallbacks*, VkAccelerationStructureKHR*);
static VKAPI_ATTR void VKAPI_CALL compat_DestroyAccelerationStructureKHR(VkDevice, VkAccelerationStructureKHR, const VkAllocationCallbacks*);
static VKAPI_ATTR void VKAPI_CALL compat_CmdBuildAccelerationStructuresKHR(VkCommandBuffer, uint32_t, const VkAccelerationStructureBuildGeometryInfoKHR*, const VkAccelerationStructureBuildRangeInfoKHR* const*);
static VKAPI_ATTR void VKAPI_CALL compat_CmdCopyAccelerationStructureKHR(VkCommandBuffer, const VkCopyAccelerationStructureInfoKHR*);
static VKAPI_ATTR void VKAPI_CALL compat_CmdWriteAccelerationStructuresPropertiesKHR(VkCommandBuffer, uint32_t, const VkAccelerationStructureKHR*, VkQueryType, VkQueryPool, uint32_t);
static VKAPI_ATTR VkDeviceAddress VKAPI_CALL compat_GetAccelerationStructureDeviceAddressKHR(VkDevice, const VkAccelerationStructureDeviceAddressInfoKHR*);
static VKAPI_ATTR void VKAPI_CALL compat_GetAccelerationStructureBuildSizesKHR(VkDevice, VkAccelerationStructureBuildTypeKHR, const VkAccelerationStructureBuildGeometryInfoKHR*, const uint32_t*, VkAccelerationStructureBuildSizesInfoKHR*);

struct FexThunkMapping {
    const char *sym_name;
    uintptr_t fallback_offset;
    void *func_ptr;
};

struct FexThunkFinder {
    uintptr_t base_addr;
    char path[512];
    bool found;
};

static int rt_fex_phdr_cb(struct dl_phdr_info *info, size_t size, void *data) {
    (void)size;
    struct FexThunkFinder *finder = (struct FexThunkFinder *)data;
    if (info->dlpi_name && strstr(info->dlpi_name, "libvulkan-host.so")) {
        finder->base_addr = (uintptr_t)info->dlpi_addr;
        strncpy(finder->path, info->dlpi_name, sizeof(finder->path) - 1);
        finder->found = true;
        return 1;
    }
    return 0;
}

static void rt_patch_fex_thunks(void) {
    static _Atomic int s_patched = 0;
    if (atomic_exchange(&s_patched, 1) != 0) return;

    struct FexThunkFinder finder = {0};
    dl_iterate_phdr(rt_fex_phdr_cb, &finder);

    if (!finder.found) {
        FILE *fp = fopen("/proc/self/maps", "r");
        if (fp) {
            char line[1024];
            while (fgets(line, sizeof(line), fp)) {
                if (strstr(line, "libvulkan-host.so")) {
                    uintptr_t start = 0;
                    if (sscanf(line, "%lx-", &start) == 1) {
                        finder.base_addr = start;
                        char *slash = strchr(line, '/');
                        if (slash) {
                            char *nl = strchr(slash, '\n');
                            if (nl) *nl = '\0';
                            strncpy(finder.path, slash, sizeof(finder.path) - 1);
                        }
                        finder.found = true;
                        break;
                    }
                }
            }
            fclose(fp);
        }
    }

    if (!finder.found) {
        COMPAT_INFO("FEX thunk patch: libvulkan-host.so not found in process (native run)");
        return;
    }

    COMPAT_INFO("FEX thunk patch: detected libvulkan-host.so at base %p (path: %s)",
                (void *)finder.base_addr, finder.path);

    struct FexThunkMapping targets[] = {
        { "fexldr_ptr_libvulkan_vkGetRayTracingShaderGroupHandlesKHR", 0xbb098, (void *)compat_GetRayTracingShaderGroupHandlesKHR },
        { "fexldr_ptr_libvulkan_vkGetRayTracingShaderGroupHandlesNV",  0xbb0a0, (void *)compat_GetRayTracingShaderGroupHandlesKHR },
        { "fexldr_ptr_libvulkan_vkCmdTraceRaysKHR",                    0xbb8d0, (void *)compat_CmdTraceRaysKHR },
        { "fexldr_ptr_libvulkan_vkCreateRayTracingPipelinesKHR",        0xbb8d8, (void *)compat_CreateRayTracingPipelinesKHR },
        { "fexldr_ptr_libvulkan_vkGetRayTracingCaptureReplayShaderGroupHandlesKHR", 0xbb8e0, (void *)compat_GetRayTracingCaptureReplayShaderGroupHandlesKHR },
        { "fexldr_ptr_libvulkan_vkCmdTraceRaysIndirectKHR",            0xbb8e8, (void *)compat_CmdTraceRaysIndirectKHR },
        { "fexldr_ptr_libvulkan_vkCmdTraceRaysIndirect2KHR",           0xbacc0, (void *)compat_CmdTraceRaysIndirect2KHR },
        { "fexldr_ptr_libvulkan_vkGetRayTracingShaderGroupStackSizeKHR", 0xbb8f0, (void *)compat_GetRayTracingShaderGroupStackSizeKHR },
        { "fexldr_ptr_libvulkan_vkCmdSetRayTracingPipelineStackSizeKHR", 0xbb8f8, (void *)compat_CmdSetRayTracingPipelineStackSizeKHR },
        { "fexldr_ptr_libvulkan_vkCreateAccelerationStructureKHR",     0, (void *)compat_CreateAccelerationStructureKHR },
        { "fexldr_ptr_libvulkan_vkDestroyAccelerationStructureKHR",    0, (void *)compat_DestroyAccelerationStructureKHR },
        { "fexldr_ptr_libvulkan_vkCmdBuildAccelerationStructuresKHR",  0, (void *)compat_CmdBuildAccelerationStructuresKHR },
        { "fexldr_ptr_libvulkan_vkCmdCopyAccelerationStructureKHR",    0, (void *)compat_CmdCopyAccelerationStructureKHR },
        { "fexldr_ptr_libvulkan_vkCmdWriteAccelerationStructuresPropertiesKHR", 0, (void *)compat_CmdWriteAccelerationStructuresPropertiesKHR },
        { "fexldr_ptr_libvulkan_vkGetAccelerationStructureDeviceAddressKHR", 0, (void *)compat_GetAccelerationStructureDeviceAddressKHR },
        { "fexldr_ptr_libvulkan_vkGetAccelerationStructureBuildSizesKHR",    0, (void *)compat_GetAccelerationStructureBuildSizesKHR },
    };
    size_t num_targets = sizeof(targets) / sizeof(targets[0]);

    const char *open_path = finder.path[0] ? finder.path : "/usr/local/lib/fex-emu/HostThunks/libvulkan-host.so";
    int fd = open(open_path, O_RDONLY);
    if (fd >= 0) {
        struct stat st;
        if (fstat(fd, &st) == 0 && st.st_size > (off_t)sizeof(Elf64_Ehdr)) {
            void *map = mmap(NULL, st.st_size, PROT_READ, MAP_SHARED, fd, 0);
            if (map != MAP_FAILED) {
                Elf64_Ehdr *ehdr = (Elf64_Ehdr *)map;
                if (ehdr->e_ident[EI_MAG0] == ELFMAG0 && ehdr->e_ident[EI_MAG1] == ELFMAG1 &&
                    ehdr->e_ident[EI_MAG2] == ELFMAG2 && ehdr->e_ident[EI_MAG3] == ELFMAG3) {
                    Elf64_Shdr *shdrs = (Elf64_Shdr *)((uint8_t *)map + ehdr->e_shoff);
                    Elf64_Shdr *symtab = NULL, *strtab = NULL;
                    for (int i = 0; i < ehdr->e_shnum; i++) {
                        if (shdrs[i].sh_type == SHT_SYMTAB && shdrs[i].sh_link < ehdr->e_shnum) {
                            symtab = &shdrs[i];
                            strtab = &shdrs[shdrs[i].sh_link];
                            break;
                        }
                    }
                    if (symtab && strtab) {
                        const char *strpool = (const char *)map + strtab->sh_offset;
                        Elf64_Sym *syms = (Elf64_Sym *)((uint8_t *)map + symtab->sh_offset);
                        int count = symtab->sh_size / sizeof(Elf64_Sym);
                        for (int i = 0; i < count; i++) {
                            const char *sym_name = strpool + syms[i].st_name;
                            for (size_t t = 0; t < num_targets; t++) {
                                if (strstr(sym_name, targets[t].sym_name)) {
                                    targets[t].fallback_offset = syms[i].st_value;
                                    break;
                                }
                            }
                        }
                        COMPAT_INFO("FEX thunk patch: resolved symbol offsets dynamically from %s", open_path);
                    }
                }
                munmap(map, st.st_size);
            }
        }
        close(fd);
    }

    for (size_t t = 0; t < num_targets; t++) {
        if (!targets[t].fallback_offset) continue;
        void **slot = (void **)(finder.base_addr + targets[t].fallback_offset);
        void *old_val = *slot;
        *slot = targets[t].func_ptr;
        COMPAT_INFO("FEX thunk patched: %s @ %p [offset 0x%lx] (%p -> %p)",
                    targets[t].sym_name, (void *)slot, (unsigned long)targets[t].fallback_offset, old_val, targets[t].func_ptr);
    }
}

/* ── Layer Integration Wrappers ───────────────────────────────────────────── */

static void rt_reset_cmd_state(VkCommandBuffer cmd, bool remove) {
    if (!cmd) return;
    unsigned b = rt_bucket((uintptr_t)cmd);
    pthread_mutex_lock(&g_rt_lock);
    struct RtCmdState **pp = &g_rt_cmd_states[b];
    while (*pp) {
        if ((*pp)->cmd == cmd) {
            struct RtCmdState *s = *pp;
            if (remove) {
                *pp = s->next;
                pthread_mutex_unlock(&g_rt_lock);
                free(s);
                return;
            }
            struct RtCmdState *next = s->next;
            memset(s, 0, sizeof(*s));
            s->cmd  = cmd;
            s->next = next;
            pthread_mutex_unlock(&g_rt_lock);
            return;
        }
        pp = &(*pp)->next;
    }
    pthread_mutex_unlock(&g_rt_lock);
}

static bool rt_cmd_bind_pipeline(VkCommandBuffer commandBuffer,
                                 VkPipelineBindPoint pipelineBindPoint,
                                 VkPipeline pipeline) {
    if (pipelineBindPoint == VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR) {
        struct RtCmdState *cs = rt_get_cmd_state(commandBuffer, true);
        struct RtPipeline *rp = rt_lookup_pipeline(pipeline);
        if (cs) cs->bound_rt_pipeline = rp;
        if (rp && rt_mega_ready(rp)) {
            struct DeviceData *d = get_device_data_by_cmd(commandBuffer);
            if (d && d->fpCmdBindPipeline) {
                d->fpCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, rp->compute_pipeline);
            }
        }
        return true;
    }
    if (pipelineBindPoint == VK_PIPELINE_BIND_POINT_COMPUTE) {
        struct RtCmdState *cs = rt_get_cmd_state(commandBuffer, true);
        if (cs) cs->bound_comp_pipeline = pipeline;
    }
    return false;
}

static void rt_cmd_push_constants(VkCommandBuffer commandBuffer,
                                  VkPipelineLayout layout,
                                  VkShaderStageFlags *inout_stages,
                                  uint32_t offset,
                                  uint32_t size,
                                  const void *pValues) {
    (void)layout;
    if (pValues && offset < 256) {
        struct RtCmdState *cs = rt_get_cmd_state(commandBuffer, true);
        if (cs) {
            uint32_t copy_len = (offset + size <= 256) ? size : (256 - offset);
            memcpy(cs->user_push_constants + offset, pValues, copy_len);
            if (offset + copy_len > cs->user_pc_dirty_size) {
                cs->user_pc_dirty_size = offset + copy_len;
            }
        }
    }
    if (inout_stages && (*inout_stages & RT_SHADER_STAGE_MASK)) {
        *inout_stages = (*inout_stages & ~RT_SHADER_STAGE_MASK) | VK_SHADER_STAGE_COMPUTE_BIT;
    }
}

static VKAPI_ATTR void VKAPI_CALL
compat_DestroyShaderModule(
    VkDevice                                    device,
    VkShaderModule                              shaderModule,
    const VkAllocationCallbacks                *pAllocator)
{
    if (g_rt_pipeline && rt_destroy_shader_module(shaderModule)) return;
    struct DeviceData *d = get_device_data(device);
    if (d && d->fpDestroyShaderModule) {
        d->fpDestroyShaderModule(device, shaderModule, pAllocator);
    }
}

static VKAPI_ATTR VkResult VKAPI_CALL
compat_RtCreateDescriptorSetLayout(
    VkDevice                                    device,
    const VkDescriptorSetLayoutCreateInfo      *pCreateInfo,
    const VkAllocationCallbacks                *pAllocator,
    VkDescriptorSetLayout                      *pSetLayout)
{
    struct DeviceData *d = get_device_data(device);
    if (!d || !d->fpCreateDescriptorSetLayout || !pCreateInfo) return VK_ERROR_INITIALIZATION_FAILED;

    VkDescriptorSetLayoutCreateInfo mod_ci = *pCreateInfo;
    VkDescriptorSetLayoutBinding stack_bindings[64];
    VkDescriptorSetLayoutBinding *bindings = NULL;

    bool has_rt_stages = false;
    for (uint32_t i = 0; i < pCreateInfo->bindingCount; i++) {
        if (pCreateInfo->pBindings[i].stageFlags & RT_SHADER_STAGE_MASK) {
            has_rt_stages = true;
            break;
        }
    }

    if (has_rt_stages && pCreateInfo->bindingCount > 0) {
        bindings = (pCreateInfo->bindingCount <= 64)
            ? stack_bindings
            : (VkDescriptorSetLayoutBinding *)malloc(pCreateInfo->bindingCount * sizeof(*bindings));
        if (bindings) {
            for (uint32_t i = 0; i < pCreateInfo->bindingCount; i++) {
                bindings[i] = pCreateInfo->pBindings[i];
                if (bindings[i].stageFlags & RT_SHADER_STAGE_MASK) {
                    bindings[i].stageFlags = (bindings[i].stageFlags & ~RT_SHADER_STAGE_MASK) |
                                             VK_SHADER_STAGE_COMPUTE_BIT;
                }
            }
            mod_ci.pBindings = bindings;
        }
    }

    VkResult res = d->fpCreateDescriptorSetLayout(device, &mod_ci, pAllocator, pSetLayout);
    if (bindings && bindings != stack_bindings) free(bindings);
    return res;
}

static VKAPI_ATTR VkResult VKAPI_CALL
compat_RtCreatePipelineLayout(
    VkDevice                                    device,
    const VkPipelineLayoutCreateInfo           *pCreateInfo,
    const VkAllocationCallbacks                *pAllocator,
    VkPipelineLayout                           *pPipelineLayout)
{
    struct DeviceData *d = get_device_data(device);
    if (!d || !d->fpCreatePipelineLayout || !pCreateInfo) return VK_ERROR_INITIALIZATION_FAILED;

    VkPipelineLayoutCreateInfo mod_ci = *pCreateInfo;
    VkPushConstantRange ranges[16];
    uint32_t rc = 0;
    VkShaderStageFlags combined_stages = VK_SHADER_STAGE_COMPUTE_BIT;
    uint32_t max_end = sizeof(struct RtDispatchPushConstants); /* 96 bytes minimum */
    if (max_end < 128) max_end = 128;

    for (uint32_t i = 0; i < pCreateInfo->pushConstantRangeCount && i < 15; i++) {
        VkShaderStageFlags st = pCreateInfo->pPushConstantRanges[i].stageFlags;
        if (st & RT_SHADER_STAGE_MASK) {
            st = (st & ~RT_SHADER_STAGE_MASK) | VK_SHADER_STAGE_COMPUTE_BIT;
        }
        combined_stages |= st;
        uint32_t end = pCreateInfo->pPushConstantRanges[i].offset + pCreateInfo->pPushConstantRanges[i].size;
        if (end > max_end && end <= 256) max_end = end;
    }

    /* Collapse into a single unified push constant range covering offset 0..max_end for combined_stages */
    ranges[0].stageFlags = combined_stages;
    ranges[0].offset     = 0;
    ranges[0].size       = max_end;
    rc = 1;

    mod_ci.pushConstantRangeCount = rc;
    mod_ci.pPushConstantRanges    = ranges;

    VkResult res = d->fpCreatePipelineLayout(device, &mod_ci, pAllocator, pPipelineLayout);
    if (res != VK_SUCCESS && pCreateInfo->pushConstantRangeCount > 0) {
        /* Fallback: keep original ranges with RT stage flags translated to COMPUTE_BIT */
        rc = pCreateInfo->pushConstantRangeCount <= 16 ? pCreateInfo->pushConstantRangeCount : 16;
        for (uint32_t i = 0; i < rc; i++) {
            ranges[i] = pCreateInfo->pPushConstantRanges[i];
            if (ranges[i].stageFlags & RT_SHADER_STAGE_MASK) {
                ranges[i].stageFlags = (ranges[i].stageFlags & ~RT_SHADER_STAGE_MASK) | VK_SHADER_STAGE_COMPUTE_BIT;
            }
            if (ranges[i].offset == 0 && ranges[i].size < sizeof(struct RtDispatchPushConstants)) {
                ranges[i].size = sizeof(struct RtDispatchPushConstants);
            }
            ranges[i].stageFlags |= VK_SHADER_STAGE_COMPUTE_BIT;
        }
        mod_ci.pushConstantRangeCount = rc;
        mod_ci.pPushConstantRanges    = ranges;
        res = d->fpCreatePipelineLayout(device, &mod_ci, pAllocator, pPipelineLayout);
    }
    if (res == VK_SUCCESS && pPipelineLayout && *pPipelineLayout) {
        rt_record_layout_pc_stages(*pPipelineLayout, combined_stages);
    }
    return res;
}

static VKAPI_ATTR void VKAPI_CALL
compat_RtCmdBindDescriptorSets(
    VkCommandBuffer                             commandBuffer,
    VkPipelineBindPoint                         pipelineBindPoint,
    VkPipelineLayout                            layout,
    uint32_t                                    firstSet,
    uint32_t                                    descriptorSetCount,
    const VkDescriptorSet                      *pDescriptorSets,
    uint32_t                                    dynamicOffsetCount,
    const uint32_t                             *pDynamicOffsets)
{
    struct DeviceData *d = get_device_data_by_cmd(commandBuffer);
    if (!d || !d->fpCmdBindDescriptorSets) return;
    if (pipelineBindPoint == VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR) {
        pipelineBindPoint = VK_PIPELINE_BIND_POINT_COMPUTE;
    }
    d->fpCmdBindDescriptorSets(commandBuffer, pipelineBindPoint, layout,
                               firstSet, descriptorSetCount, pDescriptorSets,
                               dynamicOffsetCount, pDynamicOffsets);
}

static VKAPI_ATTR void VKAPI_CALL
compat_RtCmdSetDescriptorBufferOffsetsEXT(
    VkCommandBuffer                             commandBuffer,
    VkPipelineBindPoint                         pipelineBindPoint,
    VkPipelineLayout                            layout,
    uint32_t                                    firstSet,
    uint32_t                                    setCount,
    const uint32_t                             *pBufferIndices,
    const VkDeviceSize                         *pOffsets)
{
    struct DeviceData *d = get_device_data_by_cmd(commandBuffer);
    if (!d || !d->fpCmdSetDescriptorBufferOffsetsEXT) return;
    if (pipelineBindPoint == VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR) {
        struct RtCmdState *cs = rt_get_cmd_state(commandBuffer, true);
        if (cs && pBufferIndices && pOffsets) {
            cs->has_rt_desc_buf_offsets = true;
            cs->rt_db_layout            = layout;
            cs->rt_db_first_set         = firstSet;
            cs->rt_db_set_count         = setCount <= 8 ? setCount : 8;
            memcpy(cs->rt_db_buf_indices, pBufferIndices, cs->rt_db_set_count * sizeof(uint32_t));
            memcpy(cs->rt_db_offsets, pOffsets, cs->rt_db_set_count * sizeof(VkDeviceSize));
        }
        pipelineBindPoint = VK_PIPELINE_BIND_POINT_COMPUTE;
    }
    d->fpCmdSetDescriptorBufferOffsetsEXT(commandBuffer, pipelineBindPoint, layout,
                                          firstSet, setCount, pBufferIndices, pOffsets);
}

/* ── Acceleration Structure Lifetime & Mutation Tracking ─────────────────── */

struct RtAsTrack {
    VkAccelerationStructureKHR orig_handle;
    VkAccelerationStructureKHR active_handle;
    VkDevice                   device;
    VkBuffer                   buffer;
    VkDeviceSize               offset;
    VkDeviceSize               size;
    VkAccelerationStructureTypeKHR created_type;
    VkAccelerationStructureTypeKHR built_type;
    VkDeviceSize               buf_rest;      /* 1110b, MALI_COMPAT_DEBUG only: buffer memreq size - offset (0 = unknown) */
    /* 1110b: driver handles replaced on a type change (layer-created ones, NULL allocator). Destroying one at
     * replacement time is not safe (a submitted command may still use it); the app's vkDestroyAccelerationStructure
     * guarantees every command using this AS completed, so they are destroyed there. */
    VkAccelerationStructureKHR *retired;
    uint32_t                   n_retired, cap_retired;
    /* lead-1110c */
    VkDeviceSize               as_size;       /* driver's accelerationStructureSize of the last BUILD into it (or the
                                               * source's after a copy); 0 = unknown -> a CLONE from it goes to the driver */
    VkAccelerationStructureTypeKHR active_type; /* creation type of active_handle (GENERIC or explicit) */
    bool                       xfer_ok;       /* its VkBuffer was created through compat_CreateBuffer with TRANSFER_SRC+DST */
    bool                       refit_ready;   /* false after a copy into it until a BUILD (MALI_COMPAT_RT_CLONE_REFIT) */
    struct RtAsTrack          *next;
};

/* lead-1110c: VkBuffers created through compat_CreateBuffer with ACCELERATION_STRUCTURE_STORAGE usage, to which the
 * layer added TRANSFER_SRC|TRANSFER_DST (needed for the layer's vkCmdCopyBuffer CLONE). Open-addressing set,
 * removed in compat_RtDestroyBuffer. An AS track records at CreateAS whether its buffer is in here (xfer_ok). */
static pthread_mutex_t g_rt_xbuf_lock = PTHREAD_MUTEX_INITIALIZER;
static uint64_t       *g_rt_xbuf;
static size_t          g_rt_xbuf_cap, g_rt_xbuf_n;
static inline size_t rt_xbuf_slot(uint64_t k, size_t cap) {
    k ^= k >> 33; k *= 0xff51afd7ed558ccdULL; k ^= k >> 33;
    return (size_t)k & (cap - 1);
}
static void rt_xfer_buf_add(VkBuffer b) {
    uint64_t k = (uint64_t)(uintptr_t)b;
    if (!k) return;
    pthread_mutex_lock(&g_rt_xbuf_lock);
    if ((g_rt_xbuf_n + 1) * 2 > g_rt_xbuf_cap) {
        size_t nc = g_rt_xbuf_cap ? g_rt_xbuf_cap * 2 : 1024;
        uint64_t *nt = (uint64_t *)calloc(nc, sizeof(*nt));
        if (!nt) {   /* not recorded: a CLONE through this buffer falls back to the driver (reason=usage) */
            pthread_mutex_unlock(&g_rt_xbuf_lock);
            return;
        }
        for (size_t i = 0; i < g_rt_xbuf_cap; i++) {
            if (!g_rt_xbuf[i]) continue;
            size_t j = rt_xbuf_slot(g_rt_xbuf[i], nc);
            while (nt[j]) j = (j + 1) & (nc - 1);
            nt[j] = g_rt_xbuf[i];
        }
        free(g_rt_xbuf);
        g_rt_xbuf = nt;
        g_rt_xbuf_cap = nc;
    }
    size_t i = rt_xbuf_slot(k, g_rt_xbuf_cap);
    while (g_rt_xbuf[i] && g_rt_xbuf[i] != k) i = (i + 1) & (g_rt_xbuf_cap - 1);
    if (!g_rt_xbuf[i]) { g_rt_xbuf[i] = k; g_rt_xbuf_n++; }
    pthread_mutex_unlock(&g_rt_xbuf_lock);
}
static bool rt_xfer_buf_has(VkBuffer b) {
    uint64_t k = (uint64_t)(uintptr_t)b;
    bool found = false;
    if (!k) return false;
    pthread_mutex_lock(&g_rt_xbuf_lock);
    if (g_rt_xbuf_cap) {
        size_t i = rt_xbuf_slot(k, g_rt_xbuf_cap);
        while (g_rt_xbuf[i] && g_rt_xbuf[i] != k) i = (i + 1) & (g_rt_xbuf_cap - 1);
        found = g_rt_xbuf[i] == k;
    }
    pthread_mutex_unlock(&g_rt_xbuf_lock);
    return found;
}
static void rt_xfer_buf_remove(VkBuffer b) {
    uint64_t k = (uint64_t)(uintptr_t)b;
    if (!k) return;
    pthread_mutex_lock(&g_rt_xbuf_lock);
    if (g_rt_xbuf_cap) {
        const size_t mask = g_rt_xbuf_cap - 1;
        size_t i = rt_xbuf_slot(k, g_rt_xbuf_cap);
        while (g_rt_xbuf[i] && g_rt_xbuf[i] != k) i = (i + 1) & mask;
        if (g_rt_xbuf[i] == k) {   /* backward-shift deletion keeps every probe chain intact */
            size_t j = i;
            for (;;) {
                j = (j + 1) & mask;
                if (!g_rt_xbuf[j]) break;
                size_t h = rt_xbuf_slot(g_rt_xbuf[j], g_rt_xbuf_cap);
                bool movable = (j > i) ? (h <= i || h > j) : (h <= i && h > j);
                if (movable) { g_rt_xbuf[i] = g_rt_xbuf[j]; i = j; }
            }
            g_rt_xbuf[i] = 0;
            g_rt_xbuf_n--;
        }
    }
    pthread_mutex_unlock(&g_rt_xbuf_lock);
}

#define RT_AS_TYPE_UNBUILT ((VkAccelerationStructureTypeKHR)0x7fffffff)
#define RT_AS_BUCKETS 1024
static struct RtAsTrack *g_rt_as_tracks[RT_AS_BUCKETS];
static pthread_mutex_t   g_rt_as_lock = PTHREAD_MUTEX_INITIALIZER;

static inline unsigned rt_as_hash(uint64_t v) {
    v ^= v >> 16;
    v ^= v >> 32;
    return (unsigned)(v % RT_AS_BUCKETS);
}

static struct RtAsTrack *rt_as_lookup_locked(VkAccelerationStructureKHR as) {
    if (!as) return NULL;
    unsigned b = rt_as_hash((uint64_t)(uintptr_t)as);
    for (struct RtAsTrack *t = g_rt_as_tracks[b]; t; t = t->next) {
        if (t->orig_handle == as || t->active_handle == as) return t;
    }
    for (unsigned i = 0; i < RT_AS_BUCKETS; i++) {
        if (i == b) continue;
        for (struct RtAsTrack *t = g_rt_as_tracks[i]; t; t = t->next) {
            if (t->orig_handle == as || t->active_handle == as) return t;
        }
    }
    return NULL;
}

static VKAPI_ATTR VkResult VKAPI_CALL
compat_CreateAccelerationStructureKHR(
    VkDevice                                    device,
    const VkAccelerationStructureCreateInfoKHR *pCreateInfo,
    const VkAllocationCallbacks                *pAllocator,
    VkAccelerationStructureKHR                 *pAccelerationStructure)
{
    struct DeviceData *d = get_device_data(device);
    if (!d || !d->fpCreateAccelerationStructureKHR) return VK_ERROR_INITIALIZATION_FAILED;

    VkResult res = d->fpCreateAccelerationStructureKHR(device, pCreateInfo, pAllocator, pAccelerationStructure);
    if (res == VK_SUCCESS && pAccelerationStructure && *pAccelerationStructure) {
        struct RtAsTrack *track = (struct RtAsTrack *)calloc(1, sizeof(*track));
        if (track) {
            track->orig_handle   = *pAccelerationStructure;
            track->active_handle = *pAccelerationStructure;
            track->device        = device;
            if (pCreateInfo) {
                track->buffer       = pCreateInfo->buffer;
                track->offset       = pCreateInfo->offset;
                track->size         = pCreateInfo->size;
                track->created_type = pCreateInfo->type;
            }
            track->built_type    = RT_AS_TYPE_UNBUILT; /* unbuilt */
            track->active_type   = track->created_type;
            track->refit_ready   = true;
            track->xfer_ok       = rt_xfer_buf_has(track->buffer);
            if (g_debug && pCreateInfo && pCreateInfo->buffer && d->fpGetBufferMemoryRequirements) {
                VkMemoryRequirements mr = {0};
                d->fpGetBufferMemoryRequirements(device, pCreateInfo->buffer, &mr);
                track->buf_rest = mr.size > pCreateInfo->offset ? mr.size - pCreateInfo->offset : 0;
            }

            pthread_mutex_lock(&g_rt_as_lock);
            unsigned b = rt_as_hash((uint64_t)(uintptr_t)track->orig_handle);
            track->next = g_rt_as_tracks[b];
            g_rt_as_tracks[b] = track;
            pthread_mutex_unlock(&g_rt_as_lock);
            COMPAT_INFO("CreateAS: handle=%p type=%u buf=%p off=%llu sz=%llu",
                        (void *)*pAccelerationStructure,
                        pCreateInfo ? pCreateInfo->type : 0,
                        pCreateInfo ? (void *)pCreateInfo->buffer : NULL,
                        pCreateInfo ? (unsigned long long)pCreateInfo->offset : 0ULL,
                        pCreateInfo ? (unsigned long long)pCreateInfo->size : 0ULL);
        }
    }
    return res;
}

/* 1110b: destroys the track's driver handles: the active one, the app's original (if a type change replaced it) and
 * every intermediate retired one. Layer-created handles were created with a NULL allocator. */
static void rt_as_destroy_track(struct DeviceData *d, VkDevice device, struct RtAsTrack *victim,
                                const VkAllocationCallbacks *pAllocator) {
    if (d->fpDestroyAccelerationStructureKHR) {
        bool replaced = victim->active_handle != victim->orig_handle;
        d->fpDestroyAccelerationStructureKHR(device, victim->active_handle, replaced ? NULL : pAllocator);
        if (replaced && victim->orig_handle)
            d->fpDestroyAccelerationStructureKHR(device, victim->orig_handle, pAllocator);
        for (uint32_t i = 0; i < victim->n_retired; i++)
            d->fpDestroyAccelerationStructureKHR(device, victim->retired[i], NULL);
    }
    free(victim->retired);
    free(victim);
}

static VKAPI_ATTR void VKAPI_CALL
compat_DestroyAccelerationStructureKHR(
    VkDevice                                    device,
    VkAccelerationStructureKHR                  accelerationStructure,
    const VkAllocationCallbacks                *pAllocator)
{
    struct DeviceData *d = get_device_data(device);
    if (!d || !accelerationStructure) return;

    pthread_mutex_lock(&g_rt_as_lock);
    unsigned b = rt_as_hash((uint64_t)(uintptr_t)accelerationStructure);
    for (unsigned k = 0; k <= RT_AS_BUCKETS; k++) {
        unsigned bi = k == 0 ? b : k - 1;      /* own bucket first, then all others */
        if (k > 0 && bi == b) continue;
        for (struct RtAsTrack **pp = &g_rt_as_tracks[bi]; *pp; pp = &(*pp)->next) {
            if ((*pp)->orig_handle == accelerationStructure || (*pp)->active_handle == accelerationStructure) {
                struct RtAsTrack *victim = *pp;
                *pp = victim->next;
                pthread_mutex_unlock(&g_rt_as_lock);
                rt_as_destroy_track(d, device, victim, pAllocator);
                return;
            }
        }
    }
    pthread_mutex_unlock(&g_rt_as_lock);

    if (d->fpDestroyAccelerationStructureKHR) {
        d->fpDestroyAccelerationStructureKHR(device, accelerationStructure, pAllocator);
    }
}

static bool rt_ensure_as_scratch_buffer(struct DeviceData *d) {
    if (!d) return false;
    if (d->rt_scratch_bda) return true;
    if (!d->fpCreateBuffer || !d->fpGetBufferMemoryRequirements ||
        !d->fpAllocateMemory || !d->fpBindBufferMemory || !d->fpGetBufferDeviceAddress) {
        return false;
    }

    pthread_mutex_lock(&d->bc_lock);
    if (d->rt_scratch_bda) {
        pthread_mutex_unlock(&d->bc_lock);
        return true;
    }

    VkBufferCreateInfo bci = {
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .size  = RT_AS_SCRATCH_SIZE,
        .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                 VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
                 VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR,
    };
    VkBuffer buf = VK_NULL_HANDLE;
    if (d->fpCreateBuffer(d->device, &bci, NULL, &buf) != VK_SUCCESS) {
        bci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
        if (d->fpCreateBuffer(d->device, &bci, NULL, &buf) != VK_SUCCESS) {
            pthread_mutex_unlock(&d->bc_lock);
            COMPAT_INFO("rt_ensure_as_scratch_buffer: vkCreateBuffer failed for 64MB scratch");
            return false;
        }
    }

    VkMemoryRequirements req = {0};
    d->fpGetBufferMemoryRequirements(d->device, buf, &req);
    uint32_t mask = req.memoryTypeBits;
    uint32_t mtype = mask ? (uint32_t)__builtin_ctz(mask) : 0u;

    VkMemoryAllocateFlagsInfo mafi = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO,
        .flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT,
    };
    VkMemoryAllocateInfo mai = {
        .sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        .pNext           = &mafi,
        .allocationSize  = req.size,
        .memoryTypeIndex = mtype,
    };
    VkDeviceMemory mem = VK_NULL_HANDLE;
    if (d->fpAllocateMemory(d->device, &mai, NULL, &mem) != VK_SUCCESS) {
        d->fpDestroyBuffer(d->device, buf, NULL);
        pthread_mutex_unlock(&d->bc_lock);
        COMPAT_INFO("rt_ensure_as_scratch_buffer: vkAllocateMemory failed for 64MB scratch");
        return false;
    }

    if (d->fpBindBufferMemory(d->device, buf, mem, 0) != VK_SUCCESS) {
        d->fpFreeMemory(d->device, mem, NULL);
        d->fpDestroyBuffer(d->device, buf, NULL);
        pthread_mutex_unlock(&d->bc_lock);
        COMPAT_INFO("rt_ensure_as_scratch_buffer: vkBindBufferMemory failed");
        return false;
    }

    VkBufferDeviceAddressInfo bdai = {
        .sType  = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO,
        .buffer = buf,
    };
    VkDeviceAddress bda = d->fpGetBufferDeviceAddress(d->device, &bdai);
    if (!bda) {
        d->fpFreeMemory(d->device, mem, NULL);
        d->fpDestroyBuffer(d->device, buf, NULL);
        pthread_mutex_unlock(&d->bc_lock);
        COMPAT_INFO("rt_ensure_as_scratch_buffer: fpGetBufferDeviceAddress returned 0");
        return false;
    }

    d->rt_scratch_buf = buf;
    d->rt_scratch_mem = mem;
    d->rt_scratch_bda = bda;
    atomic_store(&d->rt_scratch_offset, 0u);
    pthread_mutex_unlock(&d->bc_lock);

    COMPAT_INFO("rt_ensure_as_scratch_buffer: 64MB persistent AS scratch allocated at BDA 0x%llx",
                (unsigned long long)bda);
    return true;
}

/* 11-08 keeper: AS build scratch.
 * Data: move-fix run (7c00bbe9): 989 identical GPU WRITE faults (0x3c3) at 0x5dd2e01000 = 4 KB past the end of the
 * app's 32 MB+4 KB scratch heap [0x5dd0e00000, 0x5dd2e01000) (maps/m_hang.txt), Gemini's Flaw F: 137 KB past an
 * 8 MB heap during 54 in-place BLAS updates. e46d170 then replaced every scratch below 256 GB by FIXED 2 MB slices
 * although each build needs ~4.6 MB (the inflated size) -> builds of one call overlap and run concurrently, the BVH
 * builder's atomics/counters corrupt each other (megafix L1: 1149 WRITE/ATOMIC faults spread over 0x39d..0x3ac).
 * Now: every build gets its own slice of the layer's 128 MB scratch, sized from the DRIVER's real size query for this
 * build (not our inflated hook) plus max(need, 1 MB) headroom, 64 KB aligned, non-overlapping inside one call.
 * Reuse across calls is ordered by the full pre-build barrier (single queue: MTK m_num_queues=1).
 * Lever MALI_COMPAT_RT_OWN_SCRATCH=0: keep the app scratch (only 0/invalid scratch is still redirected).
 * MALI_COMPAT_DEBUG: rt_scratch: summary line (<= 1/s) incl. "app region short" = builds whose app scratch range
 * [scratch, scratch+need) starts in mapped memory but ends in unmapped memory (decisive for the overrun theory). */
/* 11-10 keeper (lead-1110, Codex 1110-194217 finding 3): pool exhaustion no longer forwards a scratch that was judged
 * 0/invalid. When the next build's slice does not fit the rest of the pool, the builds planned so far go out as
 * their own driver call (pre/post barrier each) and the pool restarts at 0 for the rest ("nofit" = such splits).
 * A build that ALONE does not fit (even without headroom) keeps a valid app scratch ("big-app"), and is DROPPED when
 * the app scratch is 0/invalid ("dropped", logged <= 1/s): no 0/invalid scratch address reaches the driver.
 * Per build the plan also carries the driver's real AS size (same query) for the dst-short check and the
 * MALI_COMPAT_RT_BUILD_LOG lines. RT_AS_SCRATCH_POOL_TEST: test-only smaller pool (host split test). */
#ifndef RT_AS_SCRATCH_POOL_TEST
#define RT_AS_SCRATCH_POOL ((VkDeviceSize)RT_AS_SCRATCH_SIZE)
#else
#define RT_AS_SCRATCH_POOL ((VkDeviceSize)(RT_AS_SCRATCH_POOL_TEST))
#endif
struct RtBuildPlan {
    VkDeviceSize storage;      /* dst RtAsTrack.size (what the app created the AS with), 0 = untracked */
    VkDeviceSize drv_as;       /* driver's real accelerationStructureSize for this build, 0 = not queried */
    VkDeviceSize drv_scratch;  /* driver's real build/update scratch size, 0 = not queried */
    uint32_t     batch;        /* driver call this build goes out in */
    bool         drop;
    VkAccelerationStructureKHR src_drv;   /* 1110b: driver handle of the APP's src (kept if a lever clears src) */
    VkDeviceSize src_storage;  /* 1110b: src RtAsTrack.size, 0 = no src / untracked */
    bool         othersrc_bigger;   /* 1110b: app UPDATE from another AS whose storage > dst storage */
    bool         othersrc_forced;   /* 1110b: ... turned into BUILD by MALI_COMPAT_RT_UPD_SRC */
    bool         need_pool;    /* lead-1110c: the app's UPDATE became a BUILD in the layer: its app scratch was sized for
                                * the UPDATE, so the pool slice (BUILD-sized) is used even with MALI_COMPAT_RT_OWN_SCRATCH=0 */
};
static _Atomic uint64_t g_rt_scr_builds = 0, g_rt_scr_own = 0, g_rt_scr_nofit = 0, g_rt_scr_short = 0,
                        g_rt_scr_blind = 0, g_rt_scr_maxcall = 0, g_rt_scr_dropped = 0, g_rt_scr_bigapp = 0,
                        g_rt_scr_dstshort = 0;
/* 1110b: UPDATE builds the LAYER turned into BUILD, by reason; CmdBuildAS calls passed through unplanned because a
 * host allocation failed. lead-1110c: g_rt_as_leaked now counts type-change recreations SKIPPED because the retire list
 * could not grow (Codex 1110-212923 :6773; nothing leaks any more), printed as retire-skip=; clone = first in-place
 * UPDATE after a copy (MALI_COMPAT_RT_CLONE_REFIT), badgeom = an UPDATE whose geometry the sanitiser zeroes. */
static _Atomic uint64_t g_rt_fb_unbuilt = 0, g_rt_fb_mutated = 0, g_rt_fb_nosrc = 0, g_rt_fb_lever = 0,
                        g_rt_as_leaked = 0, g_rt_build_oom = 0, g_rt_fb_clone = 0, g_rt_fb_badgeom = 0;
/* 1110b follow-up: app UPDATEs with src == dst (in place) / src != dst, and those of the latter whose source's created
 * storage is larger than the destination's; othersrc = such UPDATEs turned into BUILD by MALI_COMPAT_RT_UPD_SRC. */
static _Atomic uint64_t g_rt_upd_inplace = 0, g_rt_upd_othersrc = 0, g_rt_upd_bigger = 0, g_rt_fb_othersrc = 0;
/* returns the number of driver calls (batches) the builds were split into */
static uint32_t rt_assign_as_scratch(struct DeviceData *d, uint32_t count,
                                     VkAccelerationStructureBuildGeometryInfoKHR *infos,
                                     const VkAccelerationStructureBuildGeometryInfoKHR *pInfos,
                                     const VkAccelerationStructureBuildRangeInfoKHR * const *ppBuildRangeInfos,
                                     VkDeviceAddress own_bda, struct RtBuildPlan *plan, bool want_sizes) {
    const int own = rt_env()->own_scratch;
    const VkDeviceSize align = 65536;
    const VkDeviceSize pool_end = RT_AS_SCRATCH_POOL - 65536; /* last 64 KB stay a guard */
    VkDeviceSize cursor = 0, max_cursor = 0;
    uint32_t batch = 0;
    for (uint32_t i = 0; i < count; i++) {
        VkDeviceAddress app = infos[i].scratchData.deviceAddress;
        bool bad_app = (app == 0ULL) || rt_is_invalid_gpu_bda(app);
        VkDeviceSize need = 0;
        plan[i].drv_as = 0; plan[i].drv_scratch = 0; plan[i].drop = false;
        /* lead-1110c: always queried for a BUILD (the track records the driver's AS size for the layer's CLONE) */
        if (d->fpGetAccelerationStructureBuildSizesKHR &&
            (own || bad_app || g_debug || want_sizes || plan[i].need_pool ||
             infos[i].mode == VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR)) {
            uint32_t stack_max[32];
            uint32_t *maxp = stack_max;
            uint32_t gc = infos[i].geometryCount;
            if (gc > 32) maxp = (uint32_t *)calloc(gc, sizeof(uint32_t));
            if (maxp) {
                for (uint32_t g = 0; g < gc; g++)
                    maxp[g] = (ppBuildRangeInfos && ppBuildRangeInfos[i]) ? ppBuildRangeInfos[i][g].primitiveCount : 0;
                VkAccelerationStructureBuildGeometryInfoKHR q = infos[i];
                q.pGeometries = pInfos[i].pGeometries;
                q.ppGeometries = pInfos[i].ppGeometries;
                VkAccelerationStructureBuildSizesInfoKHR sz = {
                    .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR };
                d->fpGetAccelerationStructureBuildSizesKHR(d->device, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
                                                           &q, maxp, &sz);
                need = (infos[i].mode == VK_BUILD_ACCELERATION_STRUCTURE_MODE_UPDATE_KHR)
                           ? sz.updateScratchSize : sz.buildScratchSize;
                plan[i].drv_as = sz.accelerationStructureSize;
                plan[i].drv_scratch = need;
                if (maxp != stack_max) free(maxp);
            }
        }
        atomic_fetch_add(&g_rt_scr_builds, 1);
        if (g_debug && !bad_app && need > 0) {
            bool s0 = rt_is_host_address_accessible((const void *)(uintptr_t)app, 1);
            bool s1 = rt_is_host_address_accessible((const void *)(uintptr_t)(app + need - 1), 1);
            if (!s0) atomic_fetch_add(&g_rt_scr_blind, 1);
            else if (!s1) {
                uint64_t n = atomic_fetch_add(&g_rt_scr_short, 1) + 1;
                if (n <= 8)
                    LOG("rt_scratch: app region short #%llu: build %u/%u type=%u mode=%u scratch=0x%llx need=%llu "
                        "(end 0x%llx unmapped)", (unsigned long long)n, i + 1, count, infos[i].type, infos[i].mode,
                        (unsigned long long)app, (unsigned long long)need, (unsigned long long)(app + need));
            }
        }
        if (g_debug && plan[i].drv_as && plan[i].storage && plan[i].drv_as > plan[i].storage) {
            uint64_t n = atomic_fetch_add(&g_rt_scr_dstshort, 1) + 1;
            if (n <= 8) {
                VkDeviceAddress dst_bda = 0ULL;
                if (d->fpGetAccelerationStructureDeviceAddressKHR && infos[i].dstAccelerationStructure) {
                    VkAccelerationStructureDeviceAddressInfoKHR dai = {
                        .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR,
                        .accelerationStructure = infos[i].dstAccelerationStructure };
                    dst_bda = d->fpGetAccelerationStructureDeviceAddressKHR(d->device, &dai);
                }
                LOG("rt_scratch: dst short #%llu type=%u mode=%u dst_bda=0x%llx storage=%llu need=%llu",
                    (unsigned long long)n, infos[i].type, infos[i].mode, (unsigned long long)dst_bda,
                    (unsigned long long)plan[i].storage, (unsigned long long)plan[i].drv_as);
            }
        }
        plan[i].batch = batch;
        if (!(own || bad_app || plan[i].need_pool)) continue;   /* valid app scratch kept (OWN_SCRATCH=0) */
        if (!own_bda) {                                     /* no layer pool: an invalid app scratch cannot go out */
            if (bad_app) plan[i].drop = true;
            continue;
        }
        if (need == 0) need = 4ull * 1024 * 1024;          /* query unavailable: old 4 MB guess */
        VkDeviceSize head = need;                           /* 2x need, at least +1 MB (the app heap was overrun) */
        if (head < 1024 * 1024) head = 1024 * 1024;
        VkDeviceSize slice = (need + head + align - 1) & ~(align - 1);
        VkDeviceSize base  = ((own_bda + align - 1) & ~(align - 1)) - own_bda;
        VkDeviceSize start = (((own_bda + cursor) + align - 1) & ~(align - 1)) - own_bda;
        if (start + slice > pool_end && cursor > 0) {       /* pool full: split, the next call reuses the pool */
            batch++;
            plan[i].batch = batch;
            cursor = 0;
            start = base;
            atomic_fetch_add(&g_rt_scr_nofit, 1);
        }
        if (start + slice > pool_end) {                     /* alone too big with headroom */
            VkDeviceSize tight = (need + align - 1) & ~(align - 1);
            if (start + tight <= pool_end) {
                slice = tight;
            } else if (!bad_app) {
                atomic_fetch_add(&g_rt_scr_bigapp, 1);      /* keep the app's valid scratch */
                continue;
            } else {
                plan[i].drop = true;
                continue;
            }
        }
        infos[i].scratchData.deviceAddress = own_bda + start;
        cursor = start + slice;
        if (cursor > max_cursor) max_cursor = cursor;
        atomic_fetch_add(&g_rt_scr_own, 1);
    }
    for (uint32_t i = 0; i < count; i++) {
        if (!plan[i].drop) continue;
        uint64_t n = atomic_fetch_add(&g_rt_scr_dropped, 1) + 1;
        static _Atomic int64_t drop_ms = 0;
        int64_t now = rt_now_ms(), pv = atomic_load(&drop_ms);
        if ((n == 1 || now - pv >= 1000) && atomic_compare_exchange_strong(&drop_ms, &pv, now))
            LOG("rt_scratch: DROPPED build %u/%u type=%u mode=%u need=%llu app scratch=0x%llx (0/invalid, does not "
                "fit the %u MB pool) total dropped=%llu", i + 1, count, infos[i].type, infos[i].mode,
                (unsigned long long)plan[i].drv_scratch, (unsigned long long)infos[i].scratchData.deviceAddress,
                (unsigned)(RT_AS_SCRATCH_POOL >> 20), (unsigned long long)n);
    }
    uint64_t prev_max = atomic_load(&g_rt_scr_maxcall);
    while (max_cursor > prev_max && !atomic_compare_exchange_weak(&g_rt_scr_maxcall, &prev_max, max_cursor)) {}
    if (g_debug) {
        static _Atomic int64_t last_ms = 0;
        int64_t now = rt_now_ms(), pv = atomic_load(&last_ms);
        if (now - pv >= 1000 && atomic_compare_exchange_strong(&last_ms, &pv, now))
            LOG("rt_scratch: t=%lld ms builds=%llu own=%llu nofit=%llu dropped=%llu big-app=%llu dst-short=%llu "
                "app-short=%llu app-unmapped-start=%llu max-per-call=%llu KB forced-build=unbuilt:%llu,mutated:%llu,"
                "no-src:%llu,lever:%llu,othersrc:%llu,clone:%llu,badgeom:%llu retire-skip=%llu oom-passthrough=%llu upd-inplace=%llu "
                "upd-othersrc=%llu upd-othersrc-bigger=%llu", (long long)now,
                (unsigned long long)atomic_load(&g_rt_scr_builds), (unsigned long long)atomic_load(&g_rt_scr_own),
                (unsigned long long)atomic_load(&g_rt_scr_nofit), (unsigned long long)atomic_load(&g_rt_scr_dropped),
                (unsigned long long)atomic_load(&g_rt_scr_bigapp), (unsigned long long)atomic_load(&g_rt_scr_dstshort),
                (unsigned long long)atomic_load(&g_rt_scr_short), (unsigned long long)atomic_load(&g_rt_scr_blind),
                (unsigned long long)(atomic_load(&g_rt_scr_maxcall) >> 10),
                (unsigned long long)atomic_load(&g_rt_fb_unbuilt), (unsigned long long)atomic_load(&g_rt_fb_mutated),
                (unsigned long long)atomic_load(&g_rt_fb_nosrc), (unsigned long long)atomic_load(&g_rt_fb_lever),
                (unsigned long long)atomic_load(&g_rt_fb_othersrc), (unsigned long long)atomic_load(&g_rt_fb_clone),
                (unsigned long long)atomic_load(&g_rt_fb_badgeom), (unsigned long long)atomic_load(&g_rt_as_leaked), (unsigned long long)atomic_load(&g_rt_build_oom),
                (unsigned long long)atomic_load(&g_rt_upd_inplace), (unsigned long long)atomic_load(&g_rt_upd_othersrc),
                (unsigned long long)atomic_load(&g_rt_upd_bigger));
    }
    return batch + 1;
}

/* The full barriers around every AS build batch (unchanged since 308e8895; as-probe 11-10: a ray-query dispatch and a
 * following AS build on one queue are only ordered with a barrier between them). */
static void rt_as_build_barrier(struct DeviceData *d, VkCommandBuffer cmd, bool pre) {
    if (!d->fpCmdPipelineBarrier) return;
    VkMemoryBarrier mb = { .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER };
    if (pre) {
        mb.srcAccessMask = VK_ACCESS_HOST_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT |
                           VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_MEMORY_WRITE_BIT |
                           VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;
        mb.dstAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR |
                           VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR |
                           VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_MEMORY_READ_BIT;
    } else {
        mb.srcAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR |
                           VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_SHADER_WRITE_BIT |
                           VK_ACCESS_MEMORY_WRITE_BIT;
        mb.dstAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR |
                           VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_MEMORY_READ_BIT |
                           VK_ACCESS_UNIFORM_READ_BIT;
    }
    d->fpCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                            0, 1, &mb, 0, NULL, 0, NULL);
}

/* lead-1110c: true if the layer's sanitiser will zero a geometry of this build (invalid/0 vertex, AABB or instance
 * data -> primitiveCount 0). An UPDATE must keep the primitive counts of its BUILD (Codex 1110-212923 :6923/:6935),
 * so such an UPDATE is turned into a BUILD before the scratch is planned. Same tests as the sanitiser below. */
static bool rt_build_has_bad_geometry(const VkAccelerationStructureBuildGeometryInfoKHR *bi,
                                      const VkAccelerationStructureBuildRangeInfoKHR *ranges) {
    for (uint32_t j = 0; j < bi->geometryCount; j++) {
        const VkAccelerationStructureGeometryKHR *g = bi->pGeometries ? &bi->pGeometries[j]
                                                    : (bi->ppGeometries ? bi->ppGeometries[j] : NULL);
        if (!g) continue;
        if (g->geometryType == VK_GEOMETRY_TYPE_TRIANGLES_KHR) {
            VkDeviceAddress a = g->geometry.triangles.vertexData.deviceAddress;
            if (a == 0ULL || rt_is_invalid_gpu_bda(a)) return true;
        } else if (g->geometryType == VK_GEOMETRY_TYPE_AABBS_KHR) {
            VkDeviceAddress a = g->geometry.aabbs.data.deviceAddress;
            if (a == 0ULL || rt_is_invalid_gpu_bda(a)) return true;
        } else if (g->geometryType == VK_GEOMETRY_TYPE_INSTANCES_KHR) {
            VkDeviceAddress a = g->geometry.instances.data.deviceAddress;
            uint32_t off = ranges ? ranges[j].primitiveOffset : 0;
            if (a == 0ULL || rt_is_invalid_gpu_bda(a) || rt_is_invalid_gpu_bda(a + (uint64_t)off)) return true;
        }
    }
    return false;
}

/* lead-1110c: a build's type does not match the track's built type or its active handle's creation type: a clean
 * driver handle of that type on the same storage. Codex 1110-212923 :6773: room on the retired list is made FIRST;
 * if it cannot grow, nothing is created or destroyed - the old active handle stays and is rebuilt (as with
 * MALI_COMPAT_RT_AS_TRUST=1), counted as retire-skip. Either way the build becomes a BUILD of the new type.
 * Caller holds g_rt_as_lock. */
static void rt_as_recreate_locked(struct DeviceData *d, struct RtAsTrack *track,
                                  VkAccelerationStructureBuildGeometryInfoKHR *info, bool app_update) {
    VkAccelerationStructureKHR old = track->active_handle;
    const bool retire = old && old != track->orig_handle;
    bool room = !retire || track->n_retired < track->cap_retired;
    if (!room) {
        uint32_t nc = track->cap_retired ? track->cap_retired * 2 : 4;
        VkAccelerationStructureKHR *nr = (VkAccelerationStructureKHR *)realloc(track->retired, nc * sizeof(*nr));
        if (nr) { track->retired = nr; track->cap_retired = nc; room = true; }
    }
    VkAccelerationStructureKHR new_h = VK_NULL_HANDLE;
    if (room) {
        VkAccelerationStructureCreateInfoKHR ci = {
            .sType  = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR,
            .type   = info->type,
            .buffer = track->buffer,
            .offset = track->offset,
            .size   = track->size,
        };
        if (!d->fpCreateAccelerationStructureKHR ||
            d->fpCreateAccelerationStructureKHR(track->device, &ci, NULL, &new_h) != VK_SUCCESS)
            new_h = VK_NULL_HANDLE;
    } else {
        uint64_t n = atomic_fetch_add(&g_rt_as_leaked, 1) + 1;
        static _Atomic int64_t rs_ms = 0;
        int64_t now = rt_now_ms(), pv = atomic_load(&rs_ms);
        if ((n == 1 || now - pv >= 1000) && atomic_compare_exchange_strong(&rs_ms, &pv, now))
            LOG("RT_EMU: AS %p type %u -> %u: retire list cannot grow (out of host memory) -> no new handle, the "
                "active one is rebuilt (retire-skip=%llu)", (void *)track->orig_handle, (unsigned)track->built_type,
                (unsigned)info->type, (unsigned long long)n);
    }
    if (new_h) {
        /* 1110b: the replaced handle (unless it is the app's original, destroyed with the track anyway) is kept for
         * destruction at the app's vkDestroyAccelerationStructure - it may be in flight now */
        if (retire) track->retired[track->n_retired++] = old;
        track->active_handle = new_h;
        track->active_type   = info->type;
    }
    track->built_type = info->type;
    if (app_update && info->mode == VK_BUILD_ACCELERATION_STRUCTURE_MODE_UPDATE_KHR) atomic_fetch_add(&g_rt_fb_mutated, 1);
    info->mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
}

/* 1110b (Codex 1110-202009 #1/#4): a host allocation for planning failed. No silent cap, no stack-array fallback:
 * the WHOLE original call goes to the driver (between the usual barriers) - unless any build carries a 0/invalid
 * scratch address, then the whole call is dropped (logged <= 1/s): an invalid scratch never reaches the driver.
 * lead-1110c (Codex 1110-212923 :6666): the driver gets the ACTIVE handles (a type change may have replaced the app's
 * one): substituted in a copy of the infos. The whole call is DROPPED if that copy cannot be allocated, or if any build
 * needs what only the planned path can do: a type that does not match the active handle / the last build (UPDATE),
 * the first UPDATE of a copied-into AS (MALI_COMPAT_RT_CLONE_REFIT=1), an other-source UPDATE that
 * MALI_COMPAT_RT_UPD_SRC turns into a BUILD. A passed-through BUILD leaves the track's AS size unknown (0: a later
 * CLONE from it goes to the driver). */
static void rt_build_passthrough(struct DeviceData *d, VkCommandBuffer cmd, uint32_t infoCount,
                                 const VkAccelerationStructureBuildGeometryInfoKHR *pInfos,
                                 const VkAccelerationStructureBuildRangeInfoKHR * const *ppBuildRangeInfos) {
    uint64_t n = atomic_fetch_add(&g_rt_build_oom, 1) + 1;
    const char *why = NULL;
    uint32_t bad = UINT32_MAX;
    for (uint32_t i = 0; i < infoCount && bad == UINT32_MAX; i++) {
        VkDeviceAddress a = pInfos[i].scratchData.deviceAddress;
        if (a == 0ULL || rt_is_invalid_gpu_bda(a)) bad = i;
    }
    if (bad != UINT32_MAX) why = "0/invalid scratch";
    VkAccelerationStructureBuildGeometryInfoKHR *sub = NULL;
    const int clone_refit = rt_env()->clone_refit, us = rt_env()->upd_src;
    if (!why) {
        pthread_mutex_lock(&g_rt_as_lock);
        bool need_sub = false;
        for (uint32_t i = 0; i < infoCount && !why; i++) {
            const bool upd = pInfos[i].mode == VK_BUILD_ACCELERATION_STRUCTURE_MODE_UPDATE_KHR;
            struct RtAsTrack *t = rt_as_lookup_locked(pInfos[i].dstAccelerationStructure);
            struct RtAsTrack *st = upd ? rt_as_lookup_locked(pInfos[i].srcAccelerationStructure) : NULL;
            if (t) {
                if (t->active_type != VK_ACCELERATION_STRUCTURE_TYPE_GENERIC_KHR && t->active_type != pInfos[i].type)
                    why = "type does not match the active handle";
                else if (upd && t->built_type != RT_AS_TYPE_UNBUILT && t->built_type != pInfos[i].type)
                    why = "UPDATE of another type";
                else if (upd && clone_refit && !t->refit_ready && (!st || st == t))
                    why = "first UPDATE after a copy";
                else if (upd && st && st != t && (us == 1 || (us == 2 && st->size > t->size)))
                    why = "other-source UPDATE";
                if (t->active_handle && t->active_handle != pInfos[i].dstAccelerationStructure) need_sub = true;
            }
            if (st && st->active_handle && st->active_handle != pInfos[i].srcAccelerationStructure) need_sub = true;
        }
        if (!why && need_sub) {
            sub = (VkAccelerationStructureBuildGeometryInfoKHR *)malloc(infoCount * sizeof(*sub));
            if (!sub) why = "no host memory to substitute the active handles";
            else {
                memcpy(sub, pInfos, infoCount * sizeof(*sub));
                for (uint32_t i = 0; i < infoCount; i++) {
                    struct RtAsTrack *t = rt_as_lookup_locked(sub[i].dstAccelerationStructure);
                    if (t && t->active_handle) sub[i].dstAccelerationStructure = t->active_handle;
                    if (sub[i].mode == VK_BUILD_ACCELERATION_STRUCTURE_MODE_UPDATE_KHR) {
                        struct RtAsTrack *st = rt_as_lookup_locked(sub[i].srcAccelerationStructure);
                        if (st && st->active_handle) sub[i].srcAccelerationStructure = st->active_handle;
                    }
                }
            }
        }
        if (!why) {
            for (uint32_t i = 0; i < infoCount; i++) {
                struct RtAsTrack *t = rt_as_lookup_locked(pInfos[i].dstAccelerationStructure);
                if (!t) continue;
                if (pInfos[i].mode == VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR) {
                    t->as_size = 0;           /* not queried here */
                    t->refit_ready = true;
                    if (t->built_type == RT_AS_TYPE_UNBUILT) t->built_type = pInfos[i].type;
                } else {
                    struct RtAsTrack *st = rt_as_lookup_locked(pInfos[i].srcAccelerationStructure);
                    if (st && st != t) t->as_size = 0;
                }
            }
        }
        pthread_mutex_unlock(&g_rt_as_lock);
    }
    static _Atomic int64_t oom_ms = 0;
    int64_t now = rt_now_ms(), pv = atomic_load(&oom_ms);
    if (n <= 4 || (now - pv >= 1000 && atomic_compare_exchange_strong(&oom_ms, &pv, now))) {   /* first 4, then <= 1/s */
        if (bad != UINT32_MAX)
            LOG("rt_scratch: out of host memory planning CmdBuildAS (%u builds); build %u has scratch 0x%llx "
                "(0/invalid) -> whole call DROPPED (total oom calls=%llu)", infoCount, bad + 1,
                (unsigned long long)pInfos[bad].scratchData.deviceAddress, (unsigned long long)n);
        else if (why)
            LOG("rt_scratch: out of host memory planning CmdBuildAS (%u builds) -> whole call DROPPED (%s) "
                "(total oom calls=%llu)", infoCount, why, (unsigned long long)n);
        else
            LOG("rt_scratch: out of host memory planning CmdBuildAS (%u builds) -> passed through %s "
                "(total oom calls=%llu)", infoCount, sub ? "with the active handles" : "unmodified",
                (unsigned long long)n);
    }
    if (why) { free(sub); return; }
    rt_as_build_barrier(d, cmd, true);
    d->fpCmdBuildAccelerationStructuresKHR(cmd, infoCount, sub ? sub : pInfos, ppBuildRangeInfos);
    rt_as_build_barrier(d, cmd, false);
    free(sub);
}

static VKAPI_ATTR void VKAPI_CALL
compat_CmdBuildAccelerationStructuresKHR(
    VkCommandBuffer                                         commandBuffer,
    uint32_t                                                infoCount,
    const VkAccelerationStructureBuildGeometryInfoKHR      *pInfos,
    const VkAccelerationStructureBuildRangeInfoKHR * const *ppBuildRangeInfos)
{
    struct DeviceData *d = get_device_data_by_cmd(commandBuffer);
    if (!d || !d->fpCmdBuildAccelerationStructuresKHR) return;

    if (!pInfos || infoCount == 0) {
        d->fpCmdBuildAccelerationStructuresKHR(commandBuffer, infoCount, pInfos, ppBuildRangeInfos);
        return;
    }

    /* 1110b: every host array is allocated up front, before any AS track changes; one failure -> the whole call goes
     * through rt_build_passthrough (no silent cap to 8 builds, no stack fallback that later overruns). */
    const uint32_t count = infoCount;
    uint64_t geoms_sum = 0;
    for (uint32_t i = 0; i < count; i++) geoms_sum += pInfos[i].geometryCount;
    const uint32_t total_geoms = geoms_sum > (uint64_t)(UINT32_MAX / 2) ? 0u : (uint32_t)geoms_sum;

    VkAccelerationStructureBuildGeometryInfoKHR stack_infos[8];
    struct RtBuildPlan stack_plan[8];
    VkAccelerationStructureGeometryKHR stack_all_geoms[32];
    VkAccelerationStructureBuildRangeInfoKHR stack_all_ranges[32];
    const VkAccelerationStructureBuildRangeInfoKHR *stack_ppRangeInfos[16];
    VkAccelerationStructureBuildGeometryInfoKHR *infos = stack_infos;
    struct RtBuildPlan *plan = stack_plan;
    VkAccelerationStructureGeometryKHR *all_geoms = stack_all_geoms;
    VkAccelerationStructureBuildRangeInfoKHR *all_ranges = stack_all_ranges;
    const VkAccelerationStructureBuildRangeInfoKHR **new_ppBuildRangeInfos = stack_ppRangeInfos;
    bool oom = geoms_sum > (uint64_t)(UINT32_MAX / 2);
    if (count > 8) {
        infos = (VkAccelerationStructureBuildGeometryInfoKHR *)malloc(count * sizeof(*infos));
        plan = (struct RtBuildPlan *)malloc(count * sizeof(*plan));
        if (!infos || !plan) oom = true;
    }
    if (total_geoms > 32) {
        all_geoms = (VkAccelerationStructureGeometryKHR *)malloc(total_geoms * sizeof(*all_geoms));
        all_ranges = (VkAccelerationStructureBuildRangeInfoKHR *)malloc(total_geoms * sizeof(*all_ranges));
        if (!all_geoms || !all_ranges) oom = true;
    }
    if (count > 16) {
        new_ppBuildRangeInfos = (const VkAccelerationStructureBuildRangeInfoKHR **)malloc(count * sizeof(*new_ppBuildRangeInfos));
        if (!new_ppBuildRangeInfos) oom = true;
    }
#ifdef RT_BUILD_OOM_TEST   /* test-only: simulate a failed planning allocation for calls with more than one build */
    if (count > 1) oom = true;
#endif
    if (oom) {
        rt_build_passthrough(d, commandBuffer, infoCount, pInfos, ppBuildRangeInfos);
        goto out;
    }
    memset(plan, 0, count * sizeof(*plan));
    memcpy(infos, pInfos, count * sizeof(*infos));

    bool has_safe_scratch = rt_ensure_as_scratch_buffer(d);
    VkDeviceAddress safe_scratch_bda = d->rt_scratch_bda;

    const int as_trust = rt_env()->as_trust;
    const int clone_refit = rt_env()->clone_refit;
    pthread_mutex_lock(&g_rt_as_lock);
    for (uint32_t i = 0; i < count; i++) {
        VkAccelerationStructureKHR dst = infos[i].dstAccelerationStructure;
        struct RtAsTrack *track = rt_as_lookup_locked(dst);
        plan[i].storage = track ? track->size : 0;
        const bool app_update = infos[i].mode == VK_BUILD_ACCELERATION_STRUCTURE_MODE_UPDATE_KHR;
        if (track) {
            bool recreate = false;
            if (track->built_type == RT_AS_TYPE_UNBUILT) {
                track->built_type = infos[i].type;
                if (!as_trust) {               /* 1110b lever MALI_COMPAT_RT_AS_TRUST=1: the app's mode is kept */
                    if (app_update) atomic_fetch_add(&g_rt_fb_unbuilt, 1);
                    infos[i].mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
                }
            } else if (track->built_type != infos[i].type && as_trust) {
                /* as-probe 11-10: the blob rebuilds a GENERIC handle as the other type without a new handle */
                COMPAT_INFO("RT_EMU: AS %p type %u -> %u, same handle rebuilt (MALI_COMPAT_RT_AS_TRUST=1)",
                           (void *)track->orig_handle, (unsigned)track->built_type, (unsigned)infos[i].type);
                track->built_type = infos[i].type;
            } else if (track->built_type != infos[i].type) {
                COMPAT_INFO("RT_EMU: AS %p mutated type %u -> %u! Allocating clean driver handle.",
                           (void *)track->orig_handle, (unsigned)track->built_type, (unsigned)infos[i].type);
                recreate = true;
            }
            /* lead-1110c (Codex 1110-212923 :7159): a copy propagates the source's built type, but the ACTIVE handle
             * may have been created with the other explicit type: its creation type decides too. */
            if (!recreate && track->active_type != VK_ACCELERATION_STRUCTURE_TYPE_GENERIC_KHR &&
                track->active_type != infos[i].type) {
                COMPAT_INFO("RT_EMU: AS %p active handle created as type %u, build type %u -> clean driver handle",
                           (void *)track->orig_handle, (unsigned)track->active_type, (unsigned)infos[i].type);
                recreate = true;
            }
            if (recreate) rt_as_recreate_locked(d, track, &infos[i], app_update);
            if (track->active_handle) {
                infos[i].dstAccelerationStructure = track->active_handle;
            }
        }
        /* 11-10: MALI_COMPAT_RT_FORCE_BUILD=1 (default off): an UPDATE (in-place refit, reads the src BVH's node
         * pointers) becomes a full BUILD from the geometry. Lever ported from Gemini 4 Argon's wt-rtx-emu, where it
         * defaulted ON. */
        if (infos[i].mode == VK_BUILD_ACCELERATION_STRUCTURE_MODE_UPDATE_KHR && rt_env()->force_build) {
            atomic_fetch_add(&g_rt_fb_lever, 1);
            infos[i].mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
            infos[i].srcAccelerationStructure = VK_NULL_HANDLE;
        }

        /* lead-1110c lever MALI_COMPAT_RT_CLONE_REFIT=1 (default): the first in-place UPDATE of an AS whose content came
         * from a copy becomes a BUILD from this build's own geometry (on the blob only a BUILD through a handle is
         * proven to make it refittable); later refits of it stay refits. */
        if (track && clone_refit && !track->refit_ready &&
            infos[i].mode == VK_BUILD_ACCELERATION_STRUCTURE_MODE_UPDATE_KHR) {
            struct RtAsTrack *s_t = rt_as_lookup_locked(pInfos[i].srcAccelerationStructure);
            /* a BUILD needs the full AS size: a destination smaller than its source's AS size (a COMPACT copy) keeps
             * the UPDATE (counted in the DEBUG line, never turned into an overrunning BUILD) */
            const bool fits = !track->as_size || track->size >= track->as_size;
            if (!fits && g_debug) {
                static _Atomic uint32_t nf_lines = 0;
                if (atomic_fetch_add(&nf_lines, 1) < 8)
                    LOG("rt_build: clone-refit skipped: AS %p storage=%llu < AS size %llu (compacted copy) -> UPDATE kept",
                        (void *)track->orig_handle, (unsigned long long)track->size, (unsigned long long)track->as_size);
            }
            if (fits && (!pInfos[i].srcAccelerationStructure || s_t == track ||
                         pInfos[i].srcAccelerationStructure == pInfos[i].dstAccelerationStructure)) {
                uint64_t cn = atomic_fetch_add(&g_rt_fb_clone, 1) + 1;
                if (g_debug && cn <= 8)
                    LOG("rt_build: clone-refit #%llu: first in-place UPDATE of AS %p after a copy -> BUILD "
                        "(MALI_COMPAT_RT_CLONE_REFIT=1, storage=%llu)", (unsigned long long)cn,
                        (void *)track->orig_handle, (unsigned long long)track->size);
                infos[i].mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
                infos[i].srcAccelerationStructure = VK_NULL_HANDLE;
            }
        }

        /* 1110b follow-up (phone L1 on d4827f23: GPU WRITE faults 64-384 KB past the end of 8 MB AS buffers; the game
         * refits with src != dst). Classify the APP's UPDATE: in place or from another AS (+ source storage larger than
         * the destination's). Lever MALI_COMPAT_RT_UPD_SRC=1: an other-source UPDATE becomes a BUILD (src cleared,
         * scratch planned for BUILD; default since lead-1110c); =2: only when the source storage is larger; 0: unchanged. */
        if (app_update && pInfos[i].srcAccelerationStructure) {
            struct RtAsTrack *s_t = rt_as_lookup_locked(pInfos[i].srcAccelerationStructure);
            plan[i].src_storage = s_t ? s_t->size : 0;
            plan[i].src_drv = (s_t && s_t->active_handle) ? s_t->active_handle : pInfos[i].srcAccelerationStructure;
            bool other = (s_t && track) ? (s_t != track)
                                        : (pInfos[i].srcAccelerationStructure != pInfos[i].dstAccelerationStructure);
            if (!other) {
                atomic_fetch_add(&g_rt_upd_inplace, 1);
            } else {
                atomic_fetch_add(&g_rt_upd_othersrc, 1);
                bool bigger = plan[i].src_storage && plan[i].storage && plan[i].src_storage > plan[i].storage;
                if (bigger) { atomic_fetch_add(&g_rt_upd_bigger, 1); plan[i].othersrc_bigger = true; }
                const int us = rt_env()->upd_src;
                if (infos[i].mode == VK_BUILD_ACCELERATION_STRUCTURE_MODE_UPDATE_KHR && (us == 1 || (us == 2 && bigger))) {
                    atomic_fetch_add(&g_rt_fb_othersrc, 1);
                    plan[i].othersrc_forced = true;
                    infos[i].mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
                    infos[i].srcAccelerationStructure = VK_NULL_HANDLE;
                }
            }
        }

        if (infos[i].srcAccelerationStructure) {
            struct RtAsTrack *src_t = rt_as_lookup_locked(infos[i].srcAccelerationStructure);
            if (src_t && src_t->active_handle) {
                infos[i].srcAccelerationStructure = src_t->active_handle;
            }
        } else if (infos[i].mode == VK_BUILD_ACCELERATION_STRUCTURE_MODE_UPDATE_KHR) {
            atomic_fetch_add(&g_rt_fb_nosrc, 1);
            infos[i].mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
        }

        /* lead-1110c (Codex 1110-212923 :6923/:6935): the sanitiser below sets primitiveCount 0 for invalid geometry
         * data; an UPDATE must keep its BUILD's counts, so such an UPDATE becomes a BUILD (planned as one). */
        if (infos[i].mode == VK_BUILD_ACCELERATION_STRUCTURE_MODE_UPDATE_KHR &&
            rt_build_has_bad_geometry(&pInfos[i], ppBuildRangeInfos ? ppBuildRangeInfos[i] : NULL)) {
            uint64_t bn = atomic_fetch_add(&g_rt_fb_badgeom, 1) + 1;
            if (g_debug && bn <= 8)
                LOG("rt_build: badgeom #%llu: UPDATE with 0/invalid geometry data (the sanitiser zeroes its primitive "
                    "count) -> BUILD", (unsigned long long)bn);
            infos[i].mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
            infos[i].srcAccelerationStructure = VK_NULL_HANDLE;
        }
        /* lead-1110c (Codex 1110-212923 :6814): every UPDATE the layer turned into a BUILD gets BUILD-sized pool
         * scratch, also with MALI_COMPAT_RT_OWN_SCRATCH=0 (the app sized its scratch for the UPDATE). */
        plan[i].need_pool = app_update && infos[i].mode == VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
    }
    pthread_mutex_unlock(&g_rt_as_lock);

    if (g_debug && d->fpGetAccelerationStructureDeviceAddressKHR) {   /* 1110b: up to 8 other-src-bigger details */
        static _Atomic uint32_t big_lines = 0;
        for (uint32_t i = 0; i < count; i++) {
            if (!plan[i].othersrc_bigger || atomic_load(&big_lines) >= 8) continue;
            uint32_t n = atomic_fetch_add(&big_lines, 1) + 1;
            if (n > 8) break;
            VkAccelerationStructureDeviceAddressInfoKHR dai = {
                .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR };
            dai.accelerationStructure = plan[i].src_drv;
            VkDeviceAddress sb = d->fpGetAccelerationStructureDeviceAddressKHR(d->device, &dai);
            dai.accelerationStructure = infos[i].dstAccelerationStructure;
            VkDeviceAddress db = d->fpGetAccelerationStructureDeviceAddressKHR(d->device, &dai);
            uint64_t prims = 0;
            for (uint32_t g = 0; ppBuildRangeInfos && ppBuildRangeInfos[i] && g < infos[i].geometryCount; g++)
                prims += ppBuildRangeInfos[i][g].primitiveCount;
            LOG("rt_scratch: upd other-src #%u src_bda=0x%llx src_storage=%llu dst_bda=0x%llx dst_storage=%llu prims=%llu%s",
                n, (unsigned long long)sb, (unsigned long long)plan[i].src_storage, (unsigned long long)db,
                (unsigned long long)plan[i].storage, (unsigned long long)prims,
                plan[i].othersrc_forced ? " -> BUILD (UPD_SRC)"
                : infos[i].mode == VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR ? " -> BUILD (layer, other reason)" : "");
        }
    }

    uint32_t as_idx = atomic_fetch_add(&g_as_build_count, 1);   /* was a racy static counter */
    const uint32_t blog_n = rt_env()->build_log;
    bool blog = false;
    if (blog_n) {
        uint32_t mark = atomic_load(&g_rt_first_trace_call);
        blog = as_idx < blog_n || (mark != UINT32_MAX && as_idx >= mark && as_idx - mark < blog_n);
    }
    uint32_t nbatches = rt_assign_as_scratch(d, count, infos, pInfos, ppBuildRangeInfos,
                                             has_safe_scratch ? safe_scratch_bda : 0ULL, plan, blog);

    /* lead-1110c: per destination track, the driver's real AS size of this BUILD (0 if the query was unavailable) - the
     * layer's CLONE copies exactly that many bytes - and the handle is BUILD-made again (refittable). An in-place
     * UPDATE keeps both; an UPDATE from another AS takes the source's size. Dropped builds change nothing. */
    pthread_mutex_lock(&g_rt_as_lock);
    for (uint32_t i = 0; i < count; i++) {
        if (plan[i].drop) continue;
        struct RtAsTrack *t = rt_as_lookup_locked(infos[i].dstAccelerationStructure);
        if (!t) continue;
        if (infos[i].mode == VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR) {
            t->as_size = plan[i].drv_as;
            t->refit_ready = true;
        } else {
            struct RtAsTrack *s_t = rt_as_lookup_locked(infos[i].srcAccelerationStructure);
            if (s_t && s_t != t) t->as_size = s_t->as_size;
        }
    }
    pthread_mutex_unlock(&g_rt_as_lock);

    /* Deep geometry and range info sanitization:
     * 1. Protect against CPU user-space pointers (>= 0x6000000000ULL, e.g. FEXAllocator PROT_NONE pages)
     *    and invalid/NULL pointers (< 0x10000ULL).
     * 2. VK_GEOMETRY_TYPE_TRIANGLES_KHR:
     *    - If transformData is invalid/CPU-space, clamp to 0ULL (Vulkan standard Identity transform)
     *      and enforce transformOffset = 0 (VUID-VkAccelerationStructureGeometryTrianglesDataKHR-transformData-04092).
     *    - If vertexData is invalid, point to rt_zero_bda (degenerate triangles culled by hardware).
     *    - If indexData is invalid, set indexType = VK_INDEX_TYPE_NONE_KHR.
     * 3. VK_GEOMETRY_TYPE_INSTANCES_KHR (TLAS):
     *    - If instances.data is invalid, point to rt_zero_bda and set primitiveCount = 0.
     *    - For each instance in the array: if accelerationStructureReference is invalid/CPU-space (such as 0x6d7a3689c0),
     *      set instance.mask = 0 (so ray traversal skips it completely without affecting valid instances)
     *      and point accelerationStructureReference to rt_zero_bda so the driver BVH builder never faults!
     */
    uint32_t curr_geom_idx = 0;
    for (uint32_t i = 0; i < count; i++) {
        uint32_t g_count = infos[i].geometryCount;
        if (g_count == 0 || curr_geom_idx + g_count > total_geoms) {
            new_ppBuildRangeInfos[i] = (ppBuildRangeInfos ? ppBuildRangeInfos[i] : NULL);
            continue;
        }

        VkAccelerationStructureGeometryKHR *info_geoms = &all_geoms[curr_geom_idx];
        VkAccelerationStructureBuildRangeInfoKHR *info_ranges = &all_ranges[curr_geom_idx];
        new_ppBuildRangeInfos[i] = info_ranges;

        for (uint32_t j = 0; j < g_count; j++) {
            uint32_t idx = curr_geom_idx + j;
            const VkAccelerationStructureGeometryKHR *src_geom = NULL;
            if (pInfos[i].pGeometries) src_geom = &pInfos[i].pGeometries[j];
            else if (pInfos[i].ppGeometries && pInfos[i].ppGeometries[j]) src_geom = pInfos[i].ppGeometries[j];

            if (src_geom) {
                all_geoms[idx] = *src_geom;
            } else {
                memset(&all_geoms[idx], 0, sizeof(all_geoms[idx]));
                all_geoms[idx].sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
            }

            if (ppBuildRangeInfos && ppBuildRangeInfos[i]) {
                all_ranges[idx] = ppBuildRangeInfos[i][j];
            } else {
                memset(&all_ranges[idx], 0, sizeof(all_ranges[idx]));
            }

            if (all_geoms[idx].geometryType == VK_GEOMETRY_TYPE_TRIANGLES_KHR) {
                VkAccelerationStructureGeometryTrianglesDataKHR *tri = &all_geoms[idx].geometry.triangles;
                if (rt_is_invalid_gpu_bda(tri->transformData.deviceAddress)) {
                    tri->transformData.deviceAddress = 0ULL;
                }
                if (tri->transformData.deviceAddress == 0ULL) {
                    all_ranges[idx].transformOffset = 0;
                }
                if (rt_is_invalid_gpu_bda(tri->vertexData.deviceAddress) || tri->vertexData.deviceAddress == 0ULL) {
                    tri->vertexData.deviceAddress = rt_get_zero_bda(d);
                    if (tri->vertexStride < 12) tri->vertexStride = 12;
                    /* 1110b: nothing may be read past the 4 KB zero buffer (> ~113 tris did): no primitives */
                    all_ranges[idx].primitiveCount = 0;
                }
                if (tri->indexType != VK_INDEX_TYPE_NONE_KHR &&
                    (rt_is_invalid_gpu_bda(tri->indexData.deviceAddress) || tri->indexData.deviceAddress == 0ULL)) {
                    tri->indexType = VK_INDEX_TYPE_NONE_KHR;
                    tri->indexData.deviceAddress = 0ULL;
                }
            } else if (all_geoms[idx].geometryType == VK_GEOMETRY_TYPE_AABBS_KHR) {
                VkAccelerationStructureGeometryAabbsDataKHR *aabb = &all_geoms[idx].geometry.aabbs;
                if (rt_is_invalid_gpu_bda(aabb->data.deviceAddress) || aabb->data.deviceAddress == 0ULL) {
                    aabb->data.deviceAddress = rt_get_zero_bda(d);
                    if (aabb->stride < 24) aabb->stride = 24;
                    all_ranges[idx].primitiveCount = 0;   /* 1110b: as for triangles above */
                }
            } else if (all_geoms[idx].geometryType == VK_GEOMETRY_TYPE_INSTANCES_KHR) {
                VkAccelerationStructureGeometryInstancesDataKHR *inst = &all_geoms[idx].geometry.instances;
                VkDeviceAddress inst_bda = inst->data.deviceAddress;
                uint32_t prim_cnt = all_ranges[idx].primitiveCount;
                uint32_t prim_off = all_ranges[idx].primitiveOffset;
                if (rt_is_invalid_gpu_bda(inst_bda) || inst_bda == 0ULL ||
                    rt_is_invalid_gpu_bda(inst_bda + (uint64_t)prim_off)) {
                    inst->data.deviceAddress = rt_get_zero_bda(d);
                    all_ranges[idx].primitiveCount = 0;
                    all_ranges[idx].primitiveOffset = 0;
                } else if (inst->arrayOfPointers == VK_FALSE && prim_cnt > 0) {
                    void *host_base = rt_resolve_gpu_bda_to_host(inst_bda + prim_off, sizeof(VkAccelerationStructureInstanceKHR));
                    if (host_base) {
                        VkAccelerationStructureInstanceKHR *inst_arr = (VkAccelerationStructureInstanceKHR *)host_base;
                        uint32_t bad_refs = 0, null_refs = 0;
                        for (uint32_t k = 0; k < prim_cnt; k++) {
                            if ((k & 63u) == 0 && k > 0) {
                                if (!rt_is_host_address_accessible((const void *)&inst_arr[k], sizeof(VkAccelerationStructureInstanceKHR))) {
                                    break;
                                }
                            }
                            /* 11-10: reference 0 = Vulkan-legal inactive instance, the blob builder handles it
                             * (rq-probe `inactive` PASS 11-06): untouched. A non-zero invalid reference becomes 0
                             * (+ mask 0). It used to point at the 4 KB zero buffer, from which the Mali builder read a
                             * "BLAS header" of zeros and faulted at 0x0..0x480 (11-09: 34 READ faults). */
                            VkDeviceAddress as_ref = inst_arr[k].accelerationStructureReference;
                            if (as_ref == 0ULL) {
                                null_refs++;
                            } else if (rt_is_invalid_gpu_bda(as_ref)) {
                                inst_arr[k].mask = 0;
                                inst_arr[k].accelerationStructureReference = 0ULL;
                                bad_refs++;
                            }
                        }
                        if (bad_refs > 0 || as_idx < 10 || (as_idx % 120) == 0) {
                            COMPAT_INFO("RT_EMU: TLAS build #%u instances=%u bad_refs=%u inactive=%u (inst_bda=0x%llx, host=%p)",
                                        as_idx, prim_cnt, bad_refs, null_refs, (unsigned long long)inst_bda, host_base);
                        }
                        cache_maintenance(host_base, (VkDeviceSize)prim_cnt * sizeof(VkAccelerationStructureInstanceKHR), false);
                    }
                } else if (inst->arrayOfPointers == VK_TRUE && prim_cnt > 0) {
                    void *ptr_base = rt_resolve_gpu_bda_to_host(inst_bda + prim_off, sizeof(VkDeviceAddress));
                    if (ptr_base) {
                        VkDeviceAddress *ptr_arr = (VkDeviceAddress *)ptr_base;
                        uint32_t bad_refs = 0, null_refs = 0;
                        for (uint32_t k = 0; k < prim_cnt; k++) {
                            VkDeviceAddress inst_elem_bda = ptr_arr[k];
                            if (inst_elem_bda == 0ULL || rt_is_invalid_gpu_bda(inst_elem_bda)) {
                                /* a bad POINTER to an instance: point it at the zero buffer = an all-zero instance,
                                 * i.e. reference 0 (inactive), mask 0 - the builder reads no BLAS through it */
                                ptr_arr[k] = rt_get_zero_bda(d);
                                bad_refs++;
                                continue;
                            }
                            void *elem_host = rt_resolve_gpu_bda_to_host(inst_elem_bda, sizeof(VkAccelerationStructureInstanceKHR));
                            if (elem_host) {
                                VkAccelerationStructureInstanceKHR *one_inst = (VkAccelerationStructureInstanceKHR *)elem_host;
                                VkDeviceAddress as_ref = one_inst->accelerationStructureReference;
                                if (as_ref == 0ULL) {
                                    null_refs++;               /* legal inactive instance: untouched (11-10) */
                                } else if (rt_is_invalid_gpu_bda(as_ref)) {
                                    one_inst->mask = 0;
                                    one_inst->accelerationStructureReference = 0ULL;   /* was rt_zero_bda (11-10) */
                                    bad_refs++;
                                }
                                cache_maintenance(elem_host, sizeof(VkAccelerationStructureInstanceKHR), false);
                            }
                        }
                        if (bad_refs > 0 || as_idx < 10 || (as_idx % 120) == 0) {
                            COMPAT_INFO("RT_EMU: TLAS build #%u (array of ptrs) instances=%u bad_refs=%u inactive=%u (inst_bda=0x%llx, host=%p)",
                                        as_idx, prim_cnt, bad_refs, null_refs, (unsigned long long)inst_bda, ptr_base);
                        }
                        cache_maintenance(ptr_base, (VkDeviceSize)prim_cnt * sizeof(VkDeviceAddress), false);
                    }
                }
            }
        }

        infos[i].pGeometries = info_geoms;
        infos[i].ppGeometries = NULL;
        curr_geom_idx += g_count;
    }

    if (as_idx < 10 || (as_idx % 60) == 0) {
        for (uint32_t i = 0; i < count; i++) {
            VkDeviceAddress dst_bda = 0ULL;
            if (d->fpGetAccelerationStructureDeviceAddressKHR && infos[i].dstAccelerationStructure) {
                VkAccelerationStructureDeviceAddressInfoKHR dai = {
                    .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR,
                    .accelerationStructure = infos[i].dstAccelerationStructure,
                };
                dst_bda = d->fpGetAccelerationStructureDeviceAddressKHR(d->device, &dai);
            }
            VkDeviceAddress inst_bda = 0ULL;
            if (infos[i].type == VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR && infos[i].geometryCount > 0) {
                if (infos[i].pGeometries) inst_bda = infos[i].pGeometries[0].geometry.instances.data.deviceAddress;
                else if (infos[i].ppGeometries && infos[i].ppGeometries[0]) inst_bda = infos[i].ppGeometries[0]->geometry.instances.data.deviceAddress;
            }
            COMPAT_INFO("CmdBuildAS #%u [%u/%u]: type=%u mode=%u dst=%p (bda=0x%llx) src=%p geoms=%u inst_bda=0x%llx scratch=0x%llx (orig=0x%llx)",
                        as_idx, i + 1, count, infos[i].type, infos[i].mode,
                        (void *)infos[i].dstAccelerationStructure,
                        (unsigned long long)dst_bda,
                        (void *)infos[i].srcAccelerationStructure,
                        infos[i].geometryCount,
                        (unsigned long long)inst_bda,
                        (unsigned long long)infos[i].scratchData.deviceAddress,
                        (unsigned long long)pInfos[i].scratchData.deviceAddress);
        }
    }

    if (blog) {
        for (uint32_t i = 0; i < count; i++) {
            VkDeviceAddress dst_bda = 0ULL;
            if (d->fpGetAccelerationStructureDeviceAddressKHR && infos[i].dstAccelerationStructure) {
                VkAccelerationStructureDeviceAddressInfoKHR dai = {
                    .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR,
                    .accelerationStructure = infos[i].dstAccelerationStructure,
                };
                dst_bda = d->fpGetAccelerationStructureDeviceAddressKHR(d->device, &dai);
            }
            uint64_t prims = 0;
            const VkAccelerationStructureBuildRangeInfoKHR *ri = new_ppBuildRangeInfos[i];
            for (uint32_t g = 0; ri && g < infos[i].geometryCount; g++) prims += ri[g].primitiveCount;
            /* 1110b: first geometry's format (after the layer's sanitising) */
            char g0s[160] = " g0=none";
            const VkAccelerationStructureGeometryKHR *g0 = NULL;
            if (infos[i].geometryCount)
                g0 = infos[i].pGeometries ? &infos[i].pGeometries[0]
                                          : (infos[i].ppGeometries ? infos[i].ppGeometries[0] : NULL);
            if (g0 && g0->geometryType == VK_GEOMETRY_TYPE_TRIANGLES_KHR) {
                const VkAccelerationStructureGeometryTrianglesDataKHR *t = &g0->geometry.triangles;
                snprintf(g0s, sizeof(g0s), " g0=tri fmt=%u vstride=%llu idx=%u maxv=%u xform=%d gflags=0x%x",
                         (unsigned)t->vertexFormat, (unsigned long long)t->vertexStride, (unsigned)t->indexType,
                         t->maxVertex, t->transformData.deviceAddress != 0ULL, (unsigned)g0->flags);
            } else if (g0 && g0->geometryType == VK_GEOMETRY_TYPE_AABBS_KHR) {
                snprintf(g0s, sizeof(g0s), " g0=aabb stride=%llu gflags=0x%x",
                         (unsigned long long)g0->geometry.aabbs.stride, (unsigned)g0->flags);
            } else if (g0 && g0->geometryType == VK_GEOMETRY_TYPE_INSTANCES_KHR) {
                snprintf(g0s, sizeof(g0s), " g0=inst aop=%u gflags=0x%x",
                         (unsigned)g0->geometry.instances.arrayOfPointers, (unsigned)g0->flags);
            }
            VkDeviceAddress src_bda = 0ULL;   /* 1110b: the APP's source (also when a lever cleared it) */
            if (d->fpGetAccelerationStructureDeviceAddressKHR && plan[i].src_drv) {
                VkAccelerationStructureDeviceAddressInfoKHR sai = {
                    .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR,
                    .accelerationStructure = plan[i].src_drv };
                src_bda = d->fpGetAccelerationStructureDeviceAddressKHR(d->device, &sai);
            }
            LOG("rt_build: call=%u %u/%u type=%u mode=%u app_mode=%u flags=0x%x dst_bda=0x%llx storage=%llu src_bda=0x%llx "
                "src_storage=%llu drv_as=%llu drv_scratch=%llu scratch=0x%llx prims=%llu batch=%u%s%s", as_idx, i + 1,
                count, infos[i].type, infos[i].mode, pInfos[i].mode, infos[i].flags, (unsigned long long)dst_bda,
                (unsigned long long)plan[i].storage, (unsigned long long)src_bda, (unsigned long long)plan[i].src_storage,
                (unsigned long long)plan[i].drv_as, (unsigned long long)plan[i].drv_scratch,
                (unsigned long long)infos[i].scratchData.deviceAddress, (unsigned long long)prims, plan[i].batch,
                g0s, plan[i].drop ? " DROPPED" : "");
        }
    }

    /* 11-10: dropped builds are removed (order kept); each batch from rt_assign_as_scratch goes out as its own driver
     * call between the full pre/post barriers, so batch b+1 may reuse the scratch pool of batch b. */
    uint32_t live = 0;
    for (uint32_t i = 0; i < count; i++) {
        if (plan[i].drop) continue;
        if (live != i) {
            infos[live] = infos[i];
            new_ppBuildRangeInfos[live] = new_ppBuildRangeInfos[i];
            plan[live] = plan[i];
        }
        live++;
    }
    if (nbatches > 1 && g_debug) {
        static _Atomic int64_t split_ms = 0;
        int64_t now = rt_now_ms(), pv = atomic_load(&split_ms);
        if (now - pv >= 1000 && atomic_compare_exchange_strong(&split_ms, &pv, now))
            LOG("rt_scratch: CmdBuildAS #%u split into %u driver calls (%u builds, %u dropped)",
                as_idx, nbatches, live, count - live);
    }

    for (uint32_t first = 0; first < live;) {
        uint32_t n = 1;
        while (first + n < live && plan[first + n].batch == plan[first].batch) n++;

        rt_as_build_barrier(d, commandBuffer, true);

        const VkAccelerationStructureBuildRangeInfoKHR * const *pass_ranges =
            ppBuildRangeInfos ? (const VkAccelerationStructureBuildRangeInfoKHR * const *)&new_ppBuildRangeInfos[first]
                              : NULL;

        d->fpCmdBuildAccelerationStructuresKHR(commandBuffer, n, &infos[first], pass_ranges);

        rt_as_build_barrier(d, commandBuffer, false);
        first += n;
    }

out:
    if (all_geoms != stack_all_geoms) free(all_geoms);
    if (all_ranges != stack_all_ranges) free(all_ranges);
    if (new_ppBuildRangeInfos != stack_ppRangeInfos) free((void *)new_ppBuildRangeInfos);
    if (infos != stack_infos) free(infos);
    if (plan != stack_plan) free(plan);
}

/* lead-1110c: full barriers around the layer's CLONE copy, the same kind as around AS builds. Before: every earlier
 * write (source built, destination last used) is done and visible to the transfer, nothing still reads the
 * destination. After: the copied bytes are visible to AS builds/refits, ray queries, shaders and later transfers. */
static void rt_as_copy_barrier(struct DeviceData *d, VkCommandBuffer cmd, bool pre) {
    if (!d->fpCmdPipelineBarrier) return;
    VkMemoryBarrier mb = { .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER };
    if (pre) {
        mb.srcAccessMask = VK_ACCESS_HOST_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT |
                           VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_MEMORY_WRITE_BIT |
                           VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;
        mb.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT |
                           VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
    } else {
        mb.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
        mb.dstAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR |
                           VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR |
                           VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_MEMORY_READ_BIT |
                           VK_ACCESS_UNIFORM_READ_BIT | VK_ACCESS_TRANSFER_READ_BIT;
    }
    d->fpCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                            0, 1, &mb, 0, NULL, 0, NULL);
}

/* lead-1110c: as-probe 11-10 (blob, no layer): vkCmdCopyAccelerationStructureKHR CLONE copies the SOURCE handle's
 * created size (vkd3d-proton: rest of the buffer), starting up to 3840 B before the destination and running past the
 * destination VkBuffer - the W3 GPU write faults (L2: clone=212 then in-place refits of the clones). A freshly built
 * AS occupies exactly its own bytes, at most the driver's accelerationStructureSize of its build.
 * Lever MALI_COMPAT_RT_CLONE=1 (default): a CLONE is done by the layer: vkCmdCopyBuffer of
 * min(source AS size, source storage, destination storage) bytes from (source buffer, offset) to (destination
 * buffer, offset) between rt_as_copy_barrier. Falls back to the driver's copy (logged how=drv reason=) when a track is
 * missing, the source's AS size is unknown (0), a buffer lacks the transfer usage the layer adds in compat_CreateBuffer,
 * vkCmdCopyBuffer is unavailable, or source and destination ranges overlap in one buffer. COMPACT always goes to the
 * driver (as-probe: it stays inside the destination's 64 KB-rounded compacted size).
 * Tracks: the destination takes the source's built type and AS size and is not refittable until its next BUILD
 * (MALI_COMPAT_RT_CLONE_REFIT). */
static VKAPI_ATTR void VKAPI_CALL
compat_CmdCopyAccelerationStructureKHR(
    VkCommandBuffer                             commandBuffer,
    const VkCopyAccelerationStructureInfoKHR   *pInfo)
{
    struct DeviceData *d = get_device_data_by_cmd(commandBuffer);
    if (!d || !d->fpCmdCopyAccelerationStructureKHR) return;
    if (!pInfo) {
        d->fpCmdCopyAccelerationStructureKHR(commandBuffer, pInfo);
        return;
    }
    VkCopyAccelerationStructureInfoKHR copy = *pInfo;
    VkDeviceSize src_storage = 0, dst_storage = 0, dst_rest = 0, src_as = 0, size = 0;
    VkBuffer sbuf = VK_NULL_HANDLE, dbuf = VK_NULL_HANDLE;
    VkDeviceSize soff = 0, doff = 0;
    const bool is_clone = copy.mode == VK_COPY_ACCELERATION_STRUCTURE_MODE_CLONE_KHR;
    const char *reason = NULL;   /* NULL = the layer copies (how=buf) */
    pthread_mutex_lock(&g_rt_as_lock);
    struct RtAsTrack *st = rt_as_lookup_locked(pInfo->src);
    if (st) { copy.src = st->active_handle; src_storage = st->size; src_as = st->as_size; sbuf = st->buffer; soff = st->offset; }
    struct RtAsTrack *dt = rt_as_lookup_locked(pInfo->dst);
    if (dt) { copy.dst = dt->active_handle; dst_storage = dt->size; dst_rest = dt->buf_rest; dbuf = dt->buffer; doff = dt->offset; }
    if (!is_clone) reason = "mode";
    else if (!rt_env()->clone) reason = "lever";
    else if (!st || !dt) reason = "untracked";
    else if (!src_as) reason = "size";
    else if (!sbuf || !dbuf || !src_storage || !dst_storage) reason = "buffer";
    else if (!st->xfer_ok || !dt->xfer_ok) reason = "usage";
    else if (!d->fpCmdCopyBuffer) reason = "nofn";
    else {
        size = src_as;
        if (size > src_storage) size = src_storage;
        if (size > dst_storage) size = dst_storage;
        if (sbuf == dbuf && soff < doff + size && doff < soff + size) reason = "overlap";
    }
    if (reason) size = 0;
    /* 1110b: the copy leaves dst built with the source's type (it stayed UNBUILT before, so its first UPDATE was
     * silently turned into a BUILD; with MALI_COMPAT_RT_AS_TRUST=1 no such forcing happens at all).
     * lead-1110c: + the source's AS size, and the destination is not refittable until its next BUILD. */
    if (st && dt && st != dt) {
        if (st->built_type != RT_AS_TYPE_UNBUILT) dt->built_type = st->built_type;
        dt->as_size = st->as_size;
        dt->refit_ready = false;
    }
    pthread_mutex_unlock(&g_rt_as_lock);

    /* 1110b, MALI_COMPAT_DEBUG: what the app copies; lead-1110c: + how=buf|drv size= reason= */
    static _Atomic uint64_t c_total = 0, c_clone = 0, c_compact = 0, c_buf = 0, c_drv = 0;
    uint64_t cn = atomic_fetch_add(&c_total, 1) + 1;
    if (is_clone) atomic_fetch_add(&c_clone, 1);
    else if (copy.mode == VK_COPY_ACCELERATION_STRUCTURE_MODE_COMPACT_KHR) atomic_fetch_add(&c_compact, 1);
    uint64_t nbuf = reason ? atomic_load(&c_buf) : atomic_fetch_add(&c_buf, 1) + 1;
    uint64_t ndrv = reason ? atomic_fetch_add(&c_drv, 1) + 1 : atomic_load(&c_drv);
    const char *mn = is_clone ? "CLONE"
                   : copy.mode == VK_COPY_ACCELERATION_STRUCTURE_MODE_COMPACT_KHR ? "COMPACT"
                   : copy.mode == VK_COPY_ACCELERATION_STRUCTURE_MODE_SERIALIZE_KHR ? "SERIALIZE"
                   : copy.mode == VK_COPY_ACCELERATION_STRUCTURE_MODE_DESERIALIZE_KHR ? "DESERIALIZE" : "?";
    if (g_debug) {
        if (cn <= 64) {
            VkDeviceAddress sb = 0, db = 0;
            if (d->fpGetAccelerationStructureDeviceAddressKHR) {
                VkAccelerationStructureDeviceAddressInfoKHR dai = {
                    .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR };
                if (copy.src) { dai.accelerationStructure = copy.src; sb = d->fpGetAccelerationStructureDeviceAddressKHR(d->device, &dai); }
                if (copy.dst) { dai.accelerationStructure = copy.dst; db = d->fpGetAccelerationStructureDeviceAddressKHR(d->device, &dai); }
            }
            LOG("rt_copy: #%llu mode=%s src=0x%llx src_storage=%llu src_as=%llu dst=0x%llx dst_storage=%llu "
                "dst_buf_rest=%llu how=%s size=%llu%s%s", (unsigned long long)cn, mn, (unsigned long long)sb,
                (unsigned long long)src_storage, (unsigned long long)src_as, (unsigned long long)db,
                (unsigned long long)dst_storage, (unsigned long long)dst_rest, reason ? "drv" : "buf",
                (unsigned long long)size, reason ? " reason=" : "", reason ? reason : "");
        }
        static _Atomic int64_t c_ms = 0;
        int64_t now = rt_now_ms(), pv = atomic_load(&c_ms);
        if (now - pv >= 1000 && atomic_compare_exchange_strong(&c_ms, &pv, now))
            LOG("rt_copy: total=%llu clone=%llu compact=%llu buf=%llu drv=%llu", (unsigned long long)cn,
                (unsigned long long)atomic_load(&c_clone), (unsigned long long)atomic_load(&c_compact),
                (unsigned long long)nbuf, (unsigned long long)ndrv);
    }
    if (is_clone && reason && strcmp(reason, "lever") != 0) {   /* a CLONE the driver does: always logged, <= 1/s */
        static _Atomic int64_t f_ms = 0;
        static _Atomic uint64_t f_n = 0;
        uint64_t fn = atomic_fetch_add(&f_n, 1) + 1;
        int64_t now = rt_now_ms(), pv = atomic_load(&f_ms);
        if ((fn == 1 || now - pv >= 1000) && atomic_compare_exchange_strong(&f_ms, &pv, now))
            LOG("rt_copy: CLONE how=drv reason=%s src_storage=%llu src_as=%llu dst_storage=%llu (driver clones so far: %llu)",
                reason, (unsigned long long)src_storage, (unsigned long long)src_as, (unsigned long long)dst_storage,
                (unsigned long long)fn);
    }

    if (!reason) {
        VkBufferCopy region = { .srcOffset = soff, .dstOffset = doff, .size = size };
        rt_as_copy_barrier(d, commandBuffer, true);
        d->fpCmdCopyBuffer(commandBuffer, sbuf, dbuf, 1, &region);
        rt_as_copy_barrier(d, commandBuffer, false);
        return;
    }

    d->fpCmdCopyAccelerationStructureKHR(commandBuffer, &copy);

    if (d->fpCmdPipelineBarrier) {
        VkMemoryBarrier mb = {
            .sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
            .pNext         = NULL,
            .srcAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR | VK_ACCESS_TRANSFER_WRITE_BIT,
            .dstAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR | VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_MEMORY_READ_BIT,
        };
        d->fpCmdPipelineBarrier(commandBuffer,
                                VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR | VK_PIPELINE_STAGE_TRANSFER_BIT,
                                VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                                0,
                                1, &mb,
                                0, NULL,
                                0, NULL);
    }
}

static VKAPI_ATTR void VKAPI_CALL
compat_CmdWriteAccelerationStructuresPropertiesKHR(
    VkCommandBuffer                             commandBuffer,
    uint32_t                                    accelerationStructureCount,
    const VkAccelerationStructureKHR           *pAccelerationStructures,
    VkQueryType                                 queryType,
    VkQueryPool                                 queryPool,
    uint32_t                                    firstQuery)
{
    struct DeviceData *d = get_device_data_by_cmd(commandBuffer);
    if (!d || !d->fpCmdWriteAccelerationStructuresPropertiesKHR) return;
    if (!pAccelerationStructures || accelerationStructureCount == 0) {
        d->fpCmdWriteAccelerationStructuresPropertiesKHR(commandBuffer, accelerationStructureCount,
                                                         pAccelerationStructures, queryType, queryPool, firstQuery);
        return;
    }

    VkAccelerationStructureKHR stack_as[16];
    VkAccelerationStructureKHR *as_arr = stack_as;
    if (accelerationStructureCount > 16) {
        as_arr = (VkAccelerationStructureKHR *)malloc(accelerationStructureCount * sizeof(*as_arr));
        if (!as_arr) as_arr = stack_as;
    }
    uint32_t cnt = (as_arr == stack_as && accelerationStructureCount > 16) ? 16 : accelerationStructureCount;

    pthread_mutex_lock(&g_rt_as_lock);
    for (uint32_t i = 0; i < cnt; i++) {
        struct RtAsTrack *t = rt_as_lookup_locked(pAccelerationStructures[i]);
        as_arr[i] = t ? t->active_handle : pAccelerationStructures[i];
    }
    pthread_mutex_unlock(&g_rt_as_lock);

    /* 1110b, MALI_COMPAT_DEBUG: which AS properties the app queries (first 16 calls + a summary <= 1/s) */
    static _Atomic uint64_t q_total = 0, q_compacted = 0, q_serial = 0;
    uint64_t qn = atomic_fetch_add(&q_total, 1) + 1;
    if (queryType == VK_QUERY_TYPE_ACCELERATION_STRUCTURE_COMPACTED_SIZE_KHR) atomic_fetch_add(&q_compacted, 1);
    else if (queryType == VK_QUERY_TYPE_ACCELERATION_STRUCTURE_SERIALIZATION_SIZE_KHR) atomic_fetch_add(&q_serial, 1);
    if (g_debug) {
        if (qn <= 16) {
            const char *tn = queryType == VK_QUERY_TYPE_ACCELERATION_STRUCTURE_COMPACTED_SIZE_KHR ? "COMPACTED_SIZE"
                           : queryType == VK_QUERY_TYPE_ACCELERATION_STRUCTURE_SERIALIZATION_SIZE_KHR ? "SERIALIZATION_SIZE"
                           : queryType == VK_QUERY_TYPE_ACCELERATION_STRUCTURE_SIZE_KHR ? "SIZE"
                           : queryType == VK_QUERY_TYPE_ACCELERATION_STRUCTURE_SERIALIZATION_BOTTOM_LEVEL_POINTERS_KHR
                               ? "SERIALIZATION_BOTTOM_LEVEL_POINTERS" : "?";
            LOG("rt_query: #%llu type=%s count=%u", (unsigned long long)qn, tn, accelerationStructureCount);
        }
        static _Atomic int64_t q_ms = 0;
        int64_t now = rt_now_ms(), pv = atomic_load(&q_ms);
        if (now - pv >= 1000 && atomic_compare_exchange_strong(&q_ms, &pv, now))
            LOG("rt_query: total=%llu compacted=%llu serialization=%llu", (unsigned long long)qn,
                (unsigned long long)atomic_load(&q_compacted), (unsigned long long)atomic_load(&q_serial));
    }

    d->fpCmdWriteAccelerationStructuresPropertiesKHR(commandBuffer, cnt, as_arr, queryType, queryPool, firstQuery);

    if (as_arr != stack_as) free(as_arr);
}

static VKAPI_ATTR VkDeviceAddress VKAPI_CALL
compat_GetAccelerationStructureDeviceAddressKHR(
    VkDevice                                           device,
    const VkAccelerationStructureDeviceAddressInfoKHR *pInfo)
{
    struct DeviceData *d = get_device_data(device);
    if (!d || !d->fpGetAccelerationStructureDeviceAddressKHR || !pInfo) return 0ULL;

    VkAccelerationStructureDeviceAddressInfoKHR info = *pInfo;
    pthread_mutex_lock(&g_rt_as_lock);
    struct RtAsTrack *t = rt_as_lookup_locked(pInfo->accelerationStructure);
    if (t && t->active_handle) info.accelerationStructure = t->active_handle;
    pthread_mutex_unlock(&g_rt_as_lock);

    VkDeviceAddress bda = d->fpGetAccelerationStructureDeviceAddressKHR(device, &info);
    if (rt_is_invalid_gpu_bda(bda)) {
        /* 1110b: 0 (= an inactive instance in a TLAS), no longer rt_zero_bda: that zero buffer is in range, passed the
         * TLAS instance check, and the Mali builder read it as a BLAS header (faults at 0x0..0x480). */
        COMPAT_INFO("RT_EMU: GetASDeviceAddress returned invalid BDA 0x%llx for AS %p, returning 0 (inactive)",
                    (unsigned long long)bda, (void *)pInfo->accelerationStructure);
        return 0ULL;
    }
    return bda;
}

static VKAPI_ATTR void VKAPI_CALL
compat_GetAccelerationStructureBuildSizesKHR(
    VkDevice                                            device,
    VkAccelerationStructureBuildTypeKHR                 buildType,
    const VkAccelerationStructureBuildGeometryInfoKHR  *pBuildInfo,
    const uint32_t                                     *pMaxPrimitiveCounts,
    VkAccelerationStructureBuildSizesInfoKHR           *pSizeInfo)
{
    struct DeviceData *d = get_device_data(device);
    if (!d || !d->fpGetAccelerationStructureBuildSizesKHR || !pSizeInfo) return;

    d->fpGetAccelerationStructureBuildSizesKHR(device, buildType, pBuildInfo, pMaxPrimitiveCounts, pSizeInfo);
    /* 1110b lever MALI_COMPAT_RT_SIZE_PAD=0: the driver's own sizes (as-probe 11-10: the blob builder stays inside
     * the storage and scratch it reports). Default 1 = the inflation below. */
    if (!rt_env()->size_pad) return;

    /* Mali proprietary CSF driver underestimates scratch and AS storage
     * requirements for complex geometry and TLAS builds, causing hardware
     * MMU translation faults (0x5002c3 / 0x45003c3) past buffer bounds.
     * Scale up with ample headroom and pad to 64KB. */
    VkDeviceSize as_sz  = (pSizeInfo->accelerationStructureSize * 3ULL) / 2ULL + (2ULL * 1024 * 1024);
    VkDeviceSize scr_sz = (pSizeInfo->buildScratchSize * 2ULL) + (4ULL * 1024 * 1024);
    VkDeviceSize upd_sz = (pSizeInfo->updateScratchSize * 2ULL) + (4ULL * 1024 * 1024);

    #define PAD_64K(sz) (((sz) + 0xffffULL) & ~0xffffULL)
    pSizeInfo->accelerationStructureSize = PAD_64K(as_sz);
    pSizeInfo->buildScratchSize          = PAD_64K(scr_sz);
    pSizeInfo->updateScratchSize         = PAD_64K(upd_sz);
    #undef PAD_64K

    if (pSizeInfo->updateScratchSize < pSizeInfo->buildScratchSize) {
        pSizeInfo->updateScratchSize = pSizeInfo->buildScratchSize;
    }
}

#endif /* MALI_RT_EMU_H */

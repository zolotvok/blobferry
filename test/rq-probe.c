// SPDX-License-Identifier: GPL-3.0-or-later
// rq-probe = rq-smoke + Quake II RTX shaped arms (argv[2]): base | empty (TLAS with 0 instances) | inactive (2nd instance
// with reference 0) | array (tlas[2] both full) | arrayempty (tlas[2] = full + 0-instance TLAS, as Q2RTX's geometry + effects).
// Every arm checks the fixed rays (marker 0xC0DE = the dispatch ran) against the expected hits. Unchanged rq-smoke below.
// [keeper, Claude Opus 5.5] 2026-11-06 (docs/games/quake2rtx.md)
// rq-smoke: ray query proof of concept. Lists the ray tracing extensions/features the Vulkan device exposes, builds a
// one-triangle BLAS + one-instance TLAS on the GPU, traces 16 known rays with VK_KHR_ray_query from a compute shader
// and checks hit/miss, t, barycentrics, custom index and primitive index against the CPU. Optional perf pass:
// rq-smoke [grid] traces grid x grid rays (default 1024) and prints Mrays/s. Exit 0 = PASS, 1 = FAIL, 2 = SKIP
// (an extension/feature is missing), 3 = Vulkan error, 4 = GPU timeout. Last line is "RQ-SMOKE <verdict> ...".
// [keeper, Claude Opus 5.5] 2026-11-06 (docs/planning/rt-poc.md)
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <vulkan/vulkan.h>
#include "rq_smoke_spv.h"
#include "rq_probe_arr_spv.h"

#define CK(x) do { VkResult r_ = (x); if (r_ != VK_SUCCESS) { printf("RQ-SMOKE VKERROR %s = %d (line %d)\n", #x, r_, __LINE__); exit(3); } } while (0)

static VkDevice dev;
static VkPhysicalDeviceMemoryProperties memp;
static PFN_vkGetBufferDeviceAddress pGetBufferDeviceAddress;
static PFN_vkGetAccelerationStructureBuildSizesKHR pGetBuildSizes;
static PFN_vkCreateAccelerationStructureKHR pCreateAS;
static PFN_vkDestroyAccelerationStructureKHR pDestroyAS;
static PFN_vkCmdBuildAccelerationStructuresKHR pCmdBuildAS;
static PFN_vkGetAccelerationStructureDeviceAddressKHR pGetASAddress;

struct Buf { VkBuffer b; VkDeviceMemory m; void *p; VkDeviceAddress a; VkDeviceSize size; };

static double now_ms(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec * 1e3 + t.tv_nsec / 1e6;
}

static struct Buf buf(VkDeviceSize size, VkBufferUsageFlags usage) {
    struct Buf r = {0};
    r.size = size;
    VkBufferCreateInfo bi = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, NULL, 0, size,
                             usage | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT, VK_SHARING_MODE_EXCLUSIVE};
    CK(vkCreateBuffer(dev, &bi, NULL, &r.b));
    VkMemoryRequirements req;
    vkGetBufferMemoryRequirements(dev, r.b, &req);
    uint32_t want = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, type = UINT32_MAX;
    for (uint32_t i = 0; i < memp.memoryTypeCount && type == UINT32_MAX; i++)
        if ((req.memoryTypeBits >> i & 1) && (memp.memoryTypes[i].propertyFlags & want) == want) type = i;
    if (type == UINT32_MAX) { printf("RQ-SMOKE VKERROR no host-visible coherent memory type (bits 0x%x)\n", req.memoryTypeBits); exit(3); }
    // The app's own VkMemoryAllocateFlagsInfo, as every ray tracing app chains it (the compat layer merges it).
    VkMemoryAllocateFlagsInfo fl = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO, NULL, VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT, 0};
    VkMemoryAllocateInfo ai = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, &fl, req.size, type};
    CK(vkAllocateMemory(dev, &ai, NULL, &r.m));
    CK(vkBindBufferMemory(dev, r.b, r.m, 0));
    CK(vkMapMemory(dev, r.m, 0, VK_WHOLE_SIZE, 0, &r.p));
    memset(r.p, 0, size);
    VkBufferDeviceAddressInfo da = {VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO, NULL, r.b};
    r.a = pGetBufferDeviceAddress(dev, &da);
    return r;
}

static bool has_ext(const VkExtensionProperties *e, uint32_t n, const char *name) {
    for (uint32_t i = 0; i < n; i++) if (!strcmp(e[i].extensionName, name)) return true;
    return false;
}

static void barrier(VkCommandBuffer cb, VkPipelineStageFlags dst_stage, VkAccessFlags dst_access) {
    VkMemoryBarrier mb = {VK_STRUCTURE_TYPE_MEMORY_BARRIER, NULL, VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR, dst_access};
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR, dst_stage, 0, 1, &mb, 0, NULL, 0, NULL);
}

// Builds one AS (BLAS from triangles or TLAS from instances) into its own buffers; recorded into cb.
static VkAccelerationStructureKHR build_as(VkCommandBuffer cb, VkAccelerationStructureTypeKHR type,
                                          const VkAccelerationStructureGeometryKHR *geom, uint32_t prims,
                                          uint32_t scratch_align, struct Buf *as_buf, struct Buf *scratch) {
    VkAccelerationStructureBuildGeometryInfoKHR bi = {VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR};
    bi.type = type;
    bi.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
    bi.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
    bi.geometryCount = 1;
    bi.pGeometries = geom;
    VkAccelerationStructureBuildSizesInfoKHR sz = {VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR};
    pGetBuildSizes(dev, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR, &bi, &prims, &sz);
    printf("  %s: AS size %llu, scratch %llu\n", type == VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR ? "BLAS" : "TLAS",
           (unsigned long long)sz.accelerationStructureSize, (unsigned long long)sz.buildScratchSize);
    *as_buf = buf(sz.accelerationStructureSize, VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR);
    *scratch = buf(sz.buildScratchSize + scratch_align, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    VkAccelerationStructureCreateInfoKHR ci = {VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR, NULL, 0,
                                               as_buf->b, 0, sz.accelerationStructureSize, type, 0};
    VkAccelerationStructureKHR as;
    CK(pCreateAS(dev, &ci, NULL, &as));
    bi.dstAccelerationStructure = as;
    bi.scratchData.deviceAddress = (scratch->a + scratch_align - 1) & ~(VkDeviceAddress)(scratch_align - 1);
    VkAccelerationStructureBuildRangeInfoKHR range = {prims, 0, 0, 0};
    const VkAccelerationStructureBuildRangeInfoKHR *pr = &range;
    pCmdBuildAS(cb, 1, &bi, &pr);
    return as;
}

int main(int argc, char **argv) {
    uint32_t grid = argc > 1 ? (uint32_t)atoi(argv[1]) : 1024;
    const char *arm = argc > 2 ? argv[2] : "base";
    const bool a_empty = !strcmp(arm, "empty"), a_inact = !strcmp(arm, "inactive"), a_arr = !strcmp(arm, "array"),
               a_arre = !strcmp(arm, "arrayempty"), a_array = a_arr || a_arre;
    if (!a_empty && !a_inact && !a_array && strcmp(arm, "base")) { printf("RQ-SMOKE VKERROR unknown arm %s\n", arm); return 3; }
    if (strcmp(arm, "base")) grid = 0;
    printf("arm %s\n", arm);
    VkApplicationInfo app = {VK_STRUCTURE_TYPE_APPLICATION_INFO, NULL, "rq-smoke", 1, NULL, 0, VK_API_VERSION_1_2};
    VkInstanceCreateInfo ici = {VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, NULL, 0, &app};
    VkInstance inst;
    CK(vkCreateInstance(&ici, NULL, &inst));
    uint32_t npd = 1;
    VkPhysicalDevice pd;
    VkResult er = vkEnumeratePhysicalDevices(inst, &npd, &pd);
    if ((er != VK_SUCCESS && er != VK_INCOMPLETE) || !npd) { printf("RQ-SMOKE VKERROR no physical device (%d)\n", er); return 3; }

    VkPhysicalDeviceDriverProperties drv = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES};
    VkPhysicalDeviceAccelerationStructurePropertiesKHR asp = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_PROPERTIES_KHR, &drv};
    VkPhysicalDeviceProperties2 p2 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, &asp};
    vkGetPhysicalDeviceProperties2(pd, &p2);
    printf("device: %s, api %u.%u.%u, driver %s %s\n", p2.properties.deviceName, VK_API_VERSION_MAJOR(p2.properties.apiVersion),
           VK_API_VERSION_MINOR(p2.properties.apiVersion), VK_API_VERSION_PATCH(p2.properties.apiVersion), drv.driverName, drv.driverInfo);

    uint32_t ne = 0;
    vkEnumerateDeviceExtensionProperties(pd, NULL, &ne, NULL);
    VkExtensionProperties *ext = calloc(ne ? ne : 1, sizeof *ext);
    vkEnumerateDeviceExtensionProperties(pd, NULL, &ne, ext);
    static const char *const rt_exts[] = {"VK_KHR_acceleration_structure", "VK_KHR_ray_query", "VK_KHR_ray_tracing_pipeline",
        "VK_KHR_deferred_host_operations", "VK_KHR_ray_tracing_maintenance1", "VK_KHR_ray_tracing_position_fetch",
        "VK_KHR_pipeline_library", "VK_KHR_buffer_device_address", "VK_KHR_spirv_1_4", "VK_KHR_shader_float_controls",
        "VK_EXT_descriptor_indexing", "VK_NV_ray_tracing", "VK_EXT_opacity_micromap"};
    for (size_t i = 0; i < sizeof rt_exts / sizeof *rt_exts; i++)
        printf("ext %-36s %s\n", rt_exts[i], has_ext(ext, ne, rt_exts[i]) ? "yes" : "NO");
    bool has_as = has_ext(ext, ne, "VK_KHR_acceleration_structure"), has_rq = has_ext(ext, ne, "VK_KHR_ray_query");
    bool has_dho = has_ext(ext, ne, "VK_KHR_deferred_host_operations"), has_rtp = has_ext(ext, ne, "VK_KHR_ray_tracing_pipeline");

    VkPhysicalDeviceRayTracingPipelineFeaturesKHR rtpf = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_FEATURES_KHR};
    VkPhysicalDeviceRayQueryFeaturesKHR rqf = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_QUERY_FEATURES_KHR, has_rtp ? &rtpf : NULL};
    VkPhysicalDeviceAccelerationStructureFeaturesKHR asf = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR, &rqf};
    VkPhysicalDeviceVulkan12Features v12 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES, &asf};
    VkPhysicalDeviceFeatures2 f2 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, &v12};
    vkGetPhysicalDeviceFeatures2(pd, &f2);
    printf("feat accelerationStructure %u captureReplay %u indirectBuild %u hostCommands %u | rayQuery %u | "
           "rayTracingPipeline %u | bufferDeviceAddress %u descriptorIndexing %u\n",
           asf.accelerationStructure, asf.accelerationStructureCaptureReplay, asf.accelerationStructureIndirectBuild,
           asf.accelerationStructureHostCommands, rqf.rayQuery, rtpf.rayTracingPipeline, v12.bufferDeviceAddress, v12.descriptorIndexing);
    printf("prop maxGeometryCount %llu maxInstanceCount %llu maxPrimitiveCount %llu maxDescSetAS %u minScratchAlign %u\n",
           (unsigned long long)asp.maxGeometryCount, (unsigned long long)asp.maxInstanceCount,
           (unsigned long long)asp.maxPrimitiveCount, asp.maxDescriptorSetAccelerationStructures,
           asp.minAccelerationStructureScratchOffsetAlignment);
    if (!has_as || !has_rq || !has_dho || !asf.accelerationStructure || !rqf.rayQuery || !v12.bufferDeviceAddress) {
        printf("RQ-SMOKE SKIP missing: as %d rq %d dho %d asFeat %u rqFeat %u bda %u\n", has_as, has_rq, has_dho,
               asf.accelerationStructure, rqf.rayQuery, v12.bufferDeviceAddress);
        return 2;
    }

    uint32_t nq = 0, qf = UINT32_MAX;
    vkGetPhysicalDeviceQueueFamilyProperties(pd, &nq, NULL);
    VkQueueFamilyProperties *qp = calloc(nq, sizeof *qp);
    vkGetPhysicalDeviceQueueFamilyProperties(pd, &nq, qp);
    for (uint32_t i = 0; i < nq && qf == UINT32_MAX; i++) if (qp[i].queueFlags & VK_QUEUE_COMPUTE_BIT) qf = i;
    float prio = 1.0f;
    VkDeviceQueueCreateInfo qci = {VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO, NULL, 0, qf, 1, &prio};
    // Enable only what the test needs (as an app does): AS + ray query + BDA.
    VkPhysicalDeviceRayQueryFeaturesKHR en_rq = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_QUERY_FEATURES_KHR, NULL, VK_TRUE};
    VkPhysicalDeviceAccelerationStructureFeaturesKHR en_as = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR, &en_rq, VK_TRUE};
    VkPhysicalDeviceVulkan12Features en_12 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES, &en_as};
    en_12.bufferDeviceAddress = VK_TRUE;
    const char *dexts[] = {"VK_KHR_acceleration_structure", "VK_KHR_ray_query", "VK_KHR_deferred_host_operations"};
    VkDeviceCreateInfo dci = {VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, &en_12, 0, 1, &qci, 0, NULL, 3, dexts, NULL};
    CK(vkCreateDevice(pd, &dci, NULL, &dev));
    vkGetPhysicalDeviceMemoryProperties(pd, &memp);
#define LD(var, name) do { var = (void *)vkGetDeviceProcAddr(dev, name); if (!var) { printf("RQ-SMOKE VKERROR no entry point %s\n", name); return 3; } } while (0)
    LD(pGetBufferDeviceAddress, "vkGetBufferDeviceAddress");
    LD(pGetBuildSizes, "vkGetAccelerationStructureBuildSizesKHR");
    LD(pCreateAS, "vkCreateAccelerationStructureKHR");
    LD(pDestroyAS, "vkDestroyAccelerationStructureKHR");
    LD(pCmdBuildAS, "vkCmdBuildAccelerationStructuresKHR");
    LD(pGetASAddress, "vkGetAccelerationStructureDeviceAddressKHR");
    PFN_vkCreateDeferredOperationKHR pCreateDO;
    PFN_vkDestroyDeferredOperationKHR pDestroyDO;
    LD(pCreateDO, "vkCreateDeferredOperationKHR");
    LD(pDestroyDO, "vkDestroyDeferredOperationKHR");
    VkDeferredOperationKHR dop;
    CK(pCreateDO(dev, NULL, &dop));
    pDestroyDO(dev, dop, NULL);
    printf("entry points ok (deferred operation create/destroy ok); vkCreateRayTracingPipelinesKHR %s (not enabled)\n",
           vkGetDeviceProcAddr(dev, "vkCreateRayTracingPipelinesKHR") ? "non-NULL" : "NULL");

    VkQueue queue;
    vkGetDeviceQueue(dev, qf, 0, &queue);
    VkCommandPoolCreateInfo cpi = {VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO, NULL, VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT, qf};
    VkCommandPool pool;
    CK(vkCreateCommandPool(dev, &cpi, NULL, &pool));
    VkCommandBufferAllocateInfo cai = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, NULL, pool, VK_COMMAND_BUFFER_LEVEL_PRIMARY, 1};
    VkCommandBuffer cb;
    CK(vkAllocateCommandBuffers(dev, &cai, &cb));
    uint32_t sa = asp.minAccelerationStructureScratchOffsetAlignment ? asp.minAccelerationStructureScratchOffsetAlignment : 1;

    // BLAS: triangle (-1,-1,0) (1,-1,0) (0,1,0); TLAS: one instance, translated to z = 3, custom index 7.
    struct Buf vb = buf(sizeof(float) * 9, VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR);
    const float tri[9] = {-1, -1, 0, 1, -1, 0, 0, 1, 0};
    memcpy(vb.p, tri, sizeof tri);
    VkAccelerationStructureGeometryKHR g = {VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR};
    g.geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR;
    g.flags = VK_GEOMETRY_OPAQUE_BIT_KHR;
    g.geometry.triangles.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR;
    g.geometry.triangles.vertexFormat = VK_FORMAT_R32G32B32_SFLOAT;
    g.geometry.triangles.vertexData.deviceAddress = vb.a;
    g.geometry.triangles.vertexStride = sizeof(float) * 3;
    g.geometry.triangles.maxVertex = 2;
    g.geometry.triangles.indexType = VK_INDEX_TYPE_NONE_KHR;

    VkCommandBufferBeginInfo bgi = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, NULL, VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT};
    VkFenceCreateInfo fci = {VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    VkFence fence;
    CK(vkCreateFence(dev, &fci, NULL, &fence));
    VkSubmitInfo si = {VK_STRUCTURE_TYPE_SUBMIT_INFO, NULL, 0, NULL, NULL, 1, &cb};
    const uint64_t timeout = 5000000000ull;

    struct Buf blas_buf, blas_scr, tlas_buf, tlas_scr;
    double t0 = now_ms();
    CK(vkBeginCommandBuffer(cb, &bgi));
    VkAccelerationStructureKHR blas = build_as(cb, VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR, &g, 1, sa, &blas_buf, &blas_scr);
    CK(vkEndCommandBuffer(cb));
    CK(vkQueueSubmit(queue, 1, &si, fence));
    VkResult wr = vkWaitForFences(dev, 1, &fence, VK_TRUE, timeout);
    if (wr != VK_SUCCESS) { printf("RQ-SMOKE TIMEOUT BLAS build (%d)\n", wr); return 4; }
    double t_blas = now_ms() - t0;

    VkAccelerationStructureDeviceAddressInfoKHR adi = {VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR, NULL, blas};
    VkDeviceAddress blas_addr = pGetASAddress(dev, &adi);
    printf("  BLAS device address 0x%llx (%.2f ms build incl. submit)\n", (unsigned long long)blas_addr, t_blas);
    if (!blas_addr) { printf("RQ-SMOKE FAIL BLAS device address 0\n"); return 1; }
    struct Buf ib = buf(2 * sizeof(VkAccelerationStructureInstanceKHR), VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR);
    VkAccelerationStructureInstanceKHR in = {{{{1, 0, 0, 0}, {0, 1, 0, 0}, {0, 0, 1, 3}}}, 7, 0xFF, 0,
                                             VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR, blas_addr};
    memcpy(ib.p, &in, sizeof in);
    in.accelerationStructureReference = 0;   // arm inactive: spec "inactive instance"
    memcpy((char *)ib.p + sizeof in, &in, sizeof in);
    VkAccelerationStructureGeometryKHR gi = {VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR};
    gi.geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR;
    gi.flags = VK_GEOMETRY_OPAQUE_BIT_KHR;
    gi.geometry.instances.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR;
    gi.geometry.instances.data.deviceAddress = ib.a;

    // Descriptors + compute pipeline
    VkDescriptorSetLayoutBinding bnd[2] = {
        {0, VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, a_array ? 2 : 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL},
        {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL}};
    VkDescriptorSetLayoutCreateInfo dli = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, NULL, 0, 2, bnd};
    VkDescriptorSetLayout dsl;
    CK(vkCreateDescriptorSetLayout(dev, &dli, NULL, &dsl));
    VkPushConstantRange pcr = {VK_SHADER_STAGE_COMPUTE_BIT, 0, 12};
    VkPipelineLayoutCreateInfo pli = {VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, NULL, 0, 1, &dsl, 1, &pcr};
    VkPipelineLayout pl;
    CK(vkCreatePipelineLayout(dev, &pli, NULL, &pl));
    VkShaderModuleCreateInfo smi = {VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, NULL, 0, sizeof rq_smoke_spv, rq_smoke_spv};
    if (a_array) { smi.codeSize = sizeof rq_probe_arr_spv; smi.pCode = rq_probe_arr_spv; }
    VkShaderModule sm;
    CK(vkCreateShaderModule(dev, &smi, NULL, &sm));
    VkComputePipelineCreateInfo cpci = {VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO, NULL, 0,
        {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, NULL, 0, VK_SHADER_STAGE_COMPUTE_BIT, sm, "main", NULL}, pl, VK_NULL_HANDLE, 0};
    VkPipeline pipe;
    t0 = now_ms();
    CK(vkCreateComputePipelines(dev, VK_NULL_HANDLE, 1, &cpci, NULL, &pipe));
    printf("  ray query compute pipeline compiled (%.1f ms)\n", now_ms() - t0);
    VkDescriptorPoolSize ps[2] = {{VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 2}, {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1}};
    VkDescriptorPoolCreateInfo dpi = {VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO, NULL, 0, 1, 2, ps};
    VkDescriptorPool dp;
    CK(vkCreateDescriptorPool(dev, &dpi, NULL, &dp));
    VkDescriptorSetAllocateInfo dai = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, NULL, dp, 1, &dsl};
    VkDescriptorSet ds;
    CK(vkAllocateDescriptorSets(dev, &dai, &ds));
    const size_t out_size = 16 + 16 * 16 + 16 * 16;
    struct Buf ob = buf(out_size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);

    // TLAS build + 16 fixed rays in one submit
    CK(vkResetFences(dev, 1, &fence));
    CK(vkResetCommandBuffer(cb, 0));
    CK(vkBeginCommandBuffer(cb, &bgi));
    VkAccelerationStructureKHR tlas = build_as(cb, VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR, &gi, a_empty ? 0 : a_inact ? 2 : 1, sa,
                                               &tlas_buf, &tlas_scr);
    struct Buf tlas2_buf = {0}, tlas2_scr = {0};
    VkAccelerationStructureKHR tl[2] = {tlas, tlas};
    if (a_array) tl[1] = build_as(cb, VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR, &gi, a_arre ? 0 : 1, sa, &tlas2_buf, &tlas2_scr);
    barrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR);
    CK(vkEndCommandBuffer(cb));
    // descriptors written after the TLAS handle exists, before the dispatch is recorded (second command buffer pass)
    VkWriteDescriptorSetAccelerationStructureKHR wasd = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR, NULL, a_array ? 2 : 1, tl};
    VkDescriptorBufferInfo obi = {ob.b, 0, VK_WHOLE_SIZE};
    VkWriteDescriptorSet w[2] = {
        {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, &wasd, ds, 0, 0, a_array ? 2 : 1, VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, NULL, NULL, NULL},
        {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, NULL, ds, 1, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, NULL, &obi, NULL}};
    vkUpdateDescriptorSets(dev, 2, w, 0, NULL);
    VkCommandBuffer cb2;
    CK(vkAllocateCommandBuffers(dev, &cai, &cb2));
    CK(vkBeginCommandBuffer(cb2, &bgi));
    vkCmdBindPipeline(cb2, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
    vkCmdBindDescriptorSets(cb2, VK_PIPELINE_BIND_POINT_COMPUTE, pl, 0, 1, &ds, 0, NULL);
    uint32_t pc0[3] = {0, 0, 0};
    vkCmdPushConstants(cb2, pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, 12, pc0);
    vkCmdDispatch(cb2, 1, 1, 1);
    CK(vkEndCommandBuffer(cb2));
    VkCommandBuffer both[2] = {cb, cb2};
    VkSubmitInfo si2 = {VK_STRUCTURE_TYPE_SUBMIT_INFO, NULL, 0, NULL, NULL, 2, both};
    t0 = now_ms();
    CK(vkQueueSubmit(queue, 1, &si2, fence));
    wr = vkWaitForFences(dev, 1, &fence, VK_TRUE, timeout);
    if (wr != VK_SUCCESS) { printf("RQ-SMOKE TIMEOUT TLAS build + trace (%d)\n", wr); return 4; }
    printf("  TLAS build + 16 rays: %.2f ms incl. submit\n", now_ms() - t0);

    // Check against the CPU (array arms: pass 0 = tlas[0], pass 1 = tlas[1])
    int ok = 0, want_hits = 0;
    uint32_t fixed_hits = 0;
    bool pass = true;
    for (uint32_t which = 0; which < (a_array ? 2u : 1u); which++) {
    if (which) {
        memset(ob.p, 0, out_size);
        CK(vkResetFences(dev, 1, &fence));
        CK(vkResetCommandBuffer(cb2, 0));
        CK(vkBeginCommandBuffer(cb2, &bgi));
        vkCmdBindPipeline(cb2, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
        vkCmdBindDescriptorSets(cb2, VK_PIPELINE_BIND_POINT_COMPUTE, pl, 0, 1, &ds, 0, NULL);
        uint32_t pcw[3] = {0, 0, which};
        vkCmdPushConstants(cb2, pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, 12, pcw);
        vkCmdDispatch(cb2, 1, 1, 1);
        CK(vkEndCommandBuffer(cb2));
        VkSubmitInfo siw = {VK_STRUCTURE_TYPE_SUBMIT_INFO, NULL, 0, NULL, NULL, 1, &cb2};
        CK(vkQueueSubmit(queue, 1, &siw, fence));
        wr = vkWaitForFences(dev, 1, &fence, VK_TRUE, timeout);
        if (wr != VK_SUCCESS) { printf("RQ-SMOKE TIMEOUT pass %u (%d)\n", which, wr); return 4; }
    }
    const bool expect_none = a_empty || (a_arre && which == 1);
    printf("pass tlas[%u]: expect %s\n", which, expect_none ? "all misses (empty TLAS)" : "8 hits");
    ok = 0; want_hits = 0;
    static const float XS[4] = {-0.8f, -0.2f, 0.2f, 0.8f}, YS[4] = {-0.7f, -0.1f, 0.5f, 0.9f};
    const uint32_t *hits = ob.p;
    const float *f = (const float *)((const char *)ob.p + 16);
    const uint32_t *u = (const uint32_t *)((const char *)ob.p + 16 + 256);
    for (int i = 0; i < 16; i++) {
        float x = XS[i % 4], y = YS[i / 4];
        bool in_tri = !expect_none && y > -1.0f && fabsf(x) < (1.0f - y) / 2.0f;
        float bv = (y + 1.0f) / 2.0f, bu = (x + 1.0f - bv) / 2.0f;
        want_hits += in_tri;
        const float *r = f + i * 4;
        const uint32_t *q = u + i * 4;
        bool good = q[3] == 0xC0DE &&
                    (in_tri ? r[0] == 1.0f && fabsf(r[1] - 3.0f) < 1e-3f && fabsf(r[2] - bu) < 1e-3f && fabsf(r[3] - bv) < 1e-3f &&
                                  q[0] == 7 && q[1] == 0 && q[2] == 0
                            : r[0] == 0.0f);
        ok += good;
        printf("  ray %2d (%5.2f,%5.2f) want %-4s got hit %.0f t %.4f bary (%.4f,%.4f) custom %d prim %d inst %d %s\n", i, x, y,
               in_tri ? "hit" : "miss", r[0], r[1], r[2], r[3], (int)q[0], (int)q[1], (int)q[2], good ? "ok" : "WRONG");
    }
    fixed_hits = *hits;
    pass = pass && ok == 16 && fixed_hits == (uint32_t)want_hits;
    printf("fixed rays: %d/16 correct, hit counter %u (want %d)\n", ok, fixed_hits, want_hits);
    }

    // Perf: grid x grid rays (the triangle covers 1/2 of [-1,1]^2 minus the edges, so ~50 % hit)
    double mrays = 0;
    if (grid) {
        *(uint32_t *)ob.p = 0;
        CK(vkResetFences(dev, 1, &fence));
        CK(vkResetCommandBuffer(cb2, 0));
        CK(vkBeginCommandBuffer(cb2, &bgi));
        vkCmdBindPipeline(cb2, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
        vkCmdBindDescriptorSets(cb2, VK_PIPELINE_BIND_POINT_COMPUTE, pl, 0, 1, &ds, 0, NULL);
        uint32_t pc1[3] = {1, grid, 0};
        vkCmdPushConstants(cb2, pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, 12, pc1);
        uint64_t n = (uint64_t)grid * grid;
        vkCmdDispatch(cb2, (uint32_t)((n + 63) / 64), 1, 1);
        CK(vkEndCommandBuffer(cb2));
        VkSubmitInfo si3 = {VK_STRUCTURE_TYPE_SUBMIT_INFO, NULL, 0, NULL, NULL, 1, &cb2};
        t0 = now_ms();
        CK(vkQueueSubmit(queue, 1, &si3, fence));
        wr = vkWaitForFences(dev, 1, &fence, VK_TRUE, 4 * timeout);
        if (wr != VK_SUCCESS) { printf("RQ-SMOKE TIMEOUT perf pass (%d)\n", wr); return 4; }
        double ms = now_ms() - t0;
        mrays = n / (ms * 1e3);
        double frac = (double)*(uint32_t *)ob.p / n;
        printf("perf: %llu rays in %.2f ms incl. submit = %.1f Mrays/s, hit fraction %.3f (want ~0.5)\n",
               (unsigned long long)n, ms, mrays, frac);
        if (frac < 0.45 || frac > 0.55) { printf("perf pass hit fraction out of range\n"); pass = false; }
    }

    if (a_array) pDestroyAS(dev, tl[1], NULL);
    pDestroyAS(dev, tlas, NULL);
    pDestroyAS(dev, blas, NULL);
    vkDeviceWaitIdle(dev);
    vkDestroyDevice(dev, NULL);
    vkDestroyInstance(inst, NULL);
    printf("RQ-SMOKE %s arm %s fixed %d/16 hits %u/%d mrays %.1f device \"%s\"\n", pass ? "PASS" : "FAIL", arm, ok, fixed_hits, want_hits, mrays,
           p2.properties.deviceName);
    return pass ? 0 : 1;
}

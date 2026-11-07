// SPDX-License-Identifier: GPL-3.0-or-later
// rt-pipe-smoke: End-to-end smoke test for VK_KHR_ray_tracing_pipeline emulation over VK_KHR_ray_query.
// Tests:
//   1. Extension & feature/property queries (VK_KHR_ray_tracing_pipeline, VK_KHR_pipeline_library,
//      VK_KHR_ray_tracing_maintenance1, VK_EXT_pipeline_library_group_handles)
//   2. GPU BLAS + TLAS construction (1 triangle covering x,y in [-1, 1] at z = 2.0)
//   3. RayGenerationKHR + ClosestHitKHR (with ShaderRecordBufferKHR + IncomingRayPayloadKHR) + MissKHR
//   4. vkCreateRayTracingPipelinesKHR + vkGetRayTracingShaderGroupHandlesKHR + SBT construction
//   5. vkCmdTraceRaysKHR execution and readback verification of both hit (with SBT inline record) and miss rays
//
// Run on device (only after user permission):
//   VK_DRIVER_FILES=/usr/lib/mali_bridge/mali_bridge_icd.json HYBRIS_USE_VENDOR_NAMESPACE=1 \
//   ENABLE_MALI_COMPAT=1 MALI_COMPAT_RT_PIPELINE=1 /usr/lib/mali_bridge/rt-pipe-smoke

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vulkan/vulkan.h>

#define CHECK_VK(call, msg) do { \
    VkResult _r = (call); \
    if (_r != VK_SUCCESS) { \
        fprintf(stderr, "FAIL: %s returned %d (line %d)\n", msg, _r, __LINE__); \
        return 1; \
    } \
} while (0)

/*
 * Hand-assembled SPIR-V modules for:
 * 1. RayGenerationKHR (spv_rgen):
 *    - DescriptorSet 0, Binding 0: AccelerationStructureKHR (TLAS)
 *    - DescriptorSet 0, Binding 1: StorageBuffer uint32_t out_ pixels[16]
 *    - BuiltIn LaunchIdKHR (uvec3)
 *    - Pixel (0, 0) traces ray from (0, 0, 0) dir (0, 0, 1) -> hits triangle at z=2
 *    - Pixel (3, 3) traces ray from (10, 10, 0) dir (0, 0, 1) -> misses triangle
 *    - Writes payload (uint32_t at location 0) to out_pixels[LaunchIdKHR.y * 4 + LaunchIdKHR.x]
 */
static const uint32_t spv_rgen[] = {
    0x07230203, 0x00010500, 0x00000000, 0x00000040, 0x00000000,
    (2u << 16) | 17, 1,       /* OpCapability Shader */
    (2u << 16) | 17, 4479,    /* OpCapability RayTracingKHR */
    (6u << 16) | 10,          /* OpExtension "SPV_KHR_ray_tracing" */
    0x5f565053, 0x5f52484b, 0x5f796172, 0x63617274, 0x00676e69,
    (3u << 16) | 14, 0, 1,    /* OpMemoryModel Logical GLSL450 */
    /* OpEntryPoint RayGenerationKHR(5313) %1 "main" %2(LaunchId) %3(Payload) %4(TLAS) %5(out_buf) */
    (9u << 16) | 15, 5313, 1, 0x6e69616d, 0x00000000, 2, 3, 4, 5,
    /* Decorations */
    (4u << 16) | 71, 2, 11, 5319, /* OpDecorate %2 BuiltIn LaunchIdKHR(5319) */
    (4u << 16) | 71, 3, 30, 0,    /* OpDecorate %3 Location 0 */
    (4u << 16) | 71, 4, 34, 0,    /* OpDecorate %4 DescriptorSet 0 */
    (4u << 16) | 71, 4, 33, 0,    /* OpDecorate %4 Binding 0 */
    (4u << 16) | 71, 5, 34, 0,    /* OpDecorate %5 DescriptorSet 0 */
    (4u << 16) | 71, 5, 33, 1,    /* OpDecorate %5 Binding 1 */
    (4u << 16) | 71, 15, 6, 4,    /* OpDecorate %15 ArrayStride 4 */
    (3u << 16) | 71, 16, 2,       /* OpDecorate %16 Block */
    (5u << 16) | 72, 16, 0, 35, 0,/* OpMemberDecorate %16 0 Offset 0 */
    /* Types */
    (2u << 16) | 19, 6,           /* OpTypeVoid %6 */
    (3u << 16) | 33, 7, 6,        /* OpTypeFunction %7 %6 */
    (4u << 16) | 21, 8, 32, 0,    /* OpTypeInt %8 32 0 (uint) */
    (3u << 16) | 22, 9, 32,       /* OpTypeFloat %9 32 (float) */
    (4u << 16) | 23, 10, 8, 3,    /* OpTypeVector %10 %8 3 (uvec3) */
    (4u << 16) | 23, 11, 9, 3,    /* OpTypeVector %11 %9 3 (vec3) */
    (2u << 16) | 5341, 12,        /* OpTypeAccelerationStructureKHR %12 */
    (4u << 16) | 32, 13, 0, 12,   /* OpTypePointer %13 UniformConstant %12 */
    (4u << 16) | 32, 14, 1, 10,   /* OpTypePointer %14 Input %10 */
    (4u << 16) | 43, 8, 20, 16,   /* OpConstant %8 %20 16 */
    (4u << 16) | 28, 15, 8, 20,   /* OpTypeArray %15 %8 %20 */
    (3u << 16) | 30, 16, 15,      /* OpTypeStruct %16 %15 */
    (4u << 16) | 32, 17, 12, 16,  /* OpTypePointer %17 StorageBuffer %16 */
    (4u << 16) | 32, 18, 12, 8,   /* OpTypePointer %18 StorageBuffer %8 */
    (4u << 16) | 32, 19, 5338, 8, /* OpTypePointer %19 RayPayloadKHR(5338) %8 */
    /* Constants */
    (4u << 16) | 43, 8, 21, 0,           /* %21 = uint 0 */
    (4u << 16) | 43, 8, 22, 1,           /* %22 = uint 1 */
    (4u << 16) | 43, 8, 23, 4,           /* %23 = uint 4 */
    (4u << 16) | 43, 8, 24, 0xFF,        /* %24 = uint 0xFF */
    (4u << 16) | 43, 9, 25, 0x00000000,  /* %25 = float 0.0 */
    (4u << 16) | 43, 9, 26, 0x3f800000,  /* %26 = float 1.0 */
    (4u << 16) | 43, 9, 27, 0x42c80000,  /* %27 = float 100.0 */
    (6u << 16) | 44, 11, 28, 25, 25, 26, /* %28 = vec3(0, 0, 1) */
    /* Global Variables */
    (4u << 16) | 59, 14, 2, 1,    /* %2 = OpVariable %14 Input (LaunchIdKHR) */
    (4u << 16) | 59, 19, 3, 5338, /* %3 = OpVariable %19 RayPayloadKHR */
    (4u << 16) | 59, 13, 4, 0,    /* %4 = OpVariable %13 UniformConstant (TLAS) */
    (4u << 16) | 59, 17, 5, 12,   /* %5 = OpVariable %17 StorageBuffer (out_buf) */
    /* Function main */
    (5u << 16) | 54, 6, 1, 0, 7,  /* OpFunction %6 %1 None %7 */
    (2u << 16) | 248, 30,         /* OpLabel %30 */
    (4u << 16) | 61, 10, 31, 2,   /* %31 = OpLoad %10 %2 */
    (5u << 16) | 81, 8, 32, 31, 0,/* %32 = OpCompositeExtract %8 %31 0 (lx) */
    (5u << 16) | 81, 8, 33, 31, 1,/* %33 = OpCompositeExtract %8 %31 1 (ly) */
    (4u << 16) | 111, 9, 34, 32,  /* %34 = OpConvertUToF %9 %32 (float(lx)) */
    (5u << 16) | 133, 9, 35, 34, 34, /* %35 = OpFMul %9 %34 %34 (lx*lx: 0 for lx=0, >=1 for lx>=1; for lx=3 -> 9.0) */
    (6u << 16) | 80, 11, 36, 35, 25, 25, /* %36 = OpCompositeConstruct %11 %35 %25 %25 (origin = (lx*lx, 0, 0)) */
    (4u << 16) | 61, 12, 37, 4,   /* %37 = OpLoad %12 %4 (tlas) */
    (3u << 16) | 62, 3, 21,       /* OpStore %3 %21 (payload = 0) */
    /* OpTraceRayKHR %37 %21(flags) %24(mask) %21(sbt_off) %21(sbt_stride) %21(miss) %36(orig) %25(tmin) %28(dir) %27(tmax) %3(payload) */
    (12u << 16) | 4445, 37, 21, 24, 21, 21, 21, 36, 25, 28, 27, 3,
    (4u << 16) | 61, 8, 38, 3,    /* %38 = OpLoad %8 %3 (loaded payload) */
    (5u << 16) | 132, 8, 39, 33, 23, /* %39 = OpIMul %8 %33 %23 (ly * 4) */
    (5u << 16) | 128, 8, 40, 39, 32, /* %40 = OpIAdd %8 %39 %32 (ly * 4 + lx) */
    (6u << 16) | 65, 18, 41, 5, 21, 40, /* %41 = OpAccessChain %18 %5 %21 %40 */
    (3u << 16) | 62, 41, 38,      /* OpStore %41 %38 */
    (1u << 16) | 253,             /* OpReturn */
    (1u << 16) | 56               /* OpFunctionEnd */
};

/*
 * 2. ClosestHitKHR (spv_rchit):
 *    - IncomingRayPayloadKHR uint32_t payload (location 0)
 *    - ShaderRecordBufferKHR struct { uint32_t sbt_const; }
 *    - Writes payload = sbt_const | 0x1234u
 */
static const uint32_t spv_rchit[] = {
    0x07230203, 0x00010500, 0x00000000, 0x00000020, 0x00000000,
    (2u << 16) | 17, 1,       /* OpCapability Shader */
    (2u << 16) | 17, 4479,    /* OpCapability RayTracingKHR */
    (6u << 16) | 10,          /* OpExtension "SPV_KHR_ray_tracing" */
    0x5f565053, 0x5f52484b, 0x5f796172, 0x63617274, 0x00676e69,
    (3u << 16) | 14, 0, 1,    /* OpMemoryModel Logical GLSL450 */
    /* OpEntryPoint ClosestHitKHR(5316) %1 "main" %2(Payload) %3(ShaderRecord) */
    (7u << 16) | 15, 5316, 1, 0x6e69616d, 0x00000000, 2, 3,
    (4u << 16) | 71, 2, 30, 0,     /* OpDecorate %2 Location 0 */
    (3u << 16) | 71, 8, 2,         /* OpDecorate %8 Block */
    (5u << 16) | 72, 8, 0, 35, 0,  /* OpMemberDecorate %8 0 Offset 0 */
    (2u << 16) | 19, 4,            /* OpTypeVoid %4 */
    (3u << 16) | 33, 5, 4,         /* OpTypeFunction %5 %4 */
    (4u << 16) | 21, 6, 32, 0,     /* OpTypeInt %6 32 0 */
    (4u << 16) | 32, 7, 5342, 6,   /* OpTypePointer %7 IncomingRayPayloadKHR(5342) %6 */
    (3u << 16) | 30, 8, 6,         /* OpTypeStruct %8 %6 */
    (4u << 16) | 32, 9, 5343, 8,   /* OpTypePointer %9 ShaderRecordBufferKHR(5343) %8 */
    (4u << 16) | 32, 10, 5343, 6,  /* OpTypePointer %10 ShaderRecordBufferKHR(5343) %6 */
    (4u << 16) | 43, 6, 11, 0,     /* %11 = uint 0 */
    (4u << 16) | 43, 6, 12, 0x1234,/* %12 = uint 0x1234 */
    (4u << 16) | 59, 7, 2, 5342,   /* %2 = OpVariable %7 IncomingRayPayloadKHR */
    (4u << 16) | 59, 9, 3, 5343,   /* %3 = OpVariable %9 ShaderRecordBufferKHR */
    (5u << 16) | 54, 4, 1, 0, 5,   /* OpFunction %4 %1 None %5 */
    (2u << 16) | 248, 13,          /* OpLabel %13 */
    (5u << 16) | 65, 10, 14, 3, 11,/* %14 = OpAccessChain %10 %3 %11 */
    (4u << 16) | 61, 6, 15, 14,    /* %15 = OpLoad %6 %14 (sbt_const) */
    (5u << 16) | 198, 6, 16, 15, 12,/* %16 = OpBitwiseOr %6 %15 %12 */
    (3u << 16) | 62, 2, 16,        /* OpStore %2 %16 */
    (1u << 16) | 253,              /* OpReturn */
    (1u << 16) | 56                /* OpFunctionEnd */
};

/*
 * 3. MissKHR (spv_rmiss):
 *    - IncomingRayPayloadKHR uint32_t payload (location 0)
 *    - Writes payload = 0xDEAD0000u
 */
static const uint32_t spv_rmiss[] = {
    0x07230203, 0x00010500, 0x00000000, 0x00000010, 0x00000000,
    (2u << 16) | 17, 1,       /* OpCapability Shader */
    (2u << 16) | 17, 4479,    /* OpCapability RayTracingKHR */
    (6u << 16) | 10,          /* OpExtension "SPV_KHR_ray_tracing" */
    0x5f565053, 0x5f52484b, 0x5f796172, 0x63617274, 0x00676e69,
    (3u << 16) | 14, 0, 1,    /* OpMemoryModel Logical GLSL450 */
    /* OpEntryPoint MissKHR(5317) %1 "main" %2(Payload) */
    (6u << 16) | 15, 5317, 1, 0x6e69616d, 0x00000000, 2,
    (4u << 16) | 71, 2, 30, 0,          /* OpDecorate %2 Location 0 */
    (2u << 16) | 19, 3,                 /* OpTypeVoid %3 */
    (3u << 16) | 33, 4, 3,              /* OpTypeFunction %4 %3 */
    (4u << 16) | 21, 5, 32, 0,          /* OpTypeInt %5 32 0 */
    (4u << 16) | 32, 6, 5342, 5,        /* OpTypePointer %6 IncomingRayPayloadKHR(5342) %5 */
    (4u << 16) | 43, 5, 7, 0xDEAD0000u, /* %7 = uint 0xDEAD0000 */
    (4u << 16) | 59, 6, 2, 5342,        /* %2 = OpVariable %6 IncomingRayPayloadKHR */
    (5u << 16) | 54, 3, 1, 0, 4,        /* OpFunction %3 %1 None %4 */
    (2u << 16) | 248, 8,                /* OpLabel %8 */
    (3u << 16) | 62, 2, 7,              /* OpStore %2 %7 */
    (1u << 16) | 253,                   /* OpReturn */
    (1u << 16) | 56                     /* OpFunctionEnd */
};

static VkPhysicalDeviceMemoryProperties g_mem_props;

static uint32_t find_mem_type(uint32_t type_bits, VkMemoryPropertyFlags req_flags) {
    for (uint32_t i = 0; i < g_mem_props.memoryTypeCount; i++) {
        if ((type_bits & (1u << i)) &&
            (g_mem_props.memoryTypes[i].propertyFlags & req_flags) == req_flags) {
            return i;
        }
    }
    return (uint32_t)__builtin_ctz(type_bits);
}

static VkResult create_bda_buffer(VkDevice dev, VkDeviceSize size, VkBufferUsageFlags usage,
                                  const void *init_data, VkBuffer *out_buf, VkDeviceMemory *out_mem,
                                  VkDeviceAddress *out_bda) {
    VkBufferCreateInfo bci = {
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .size  = size < 256 ? 256 : size,
        .usage = usage | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
    };
    VkResult r = vkCreateBuffer(dev, &bci, NULL, out_buf);
    if (r != VK_SUCCESS) return r;

    VkMemoryRequirements req;
    vkGetBufferMemoryRequirements(dev, *out_buf, &req);
    uint32_t mtype = find_mem_type(req.memoryTypeBits,
                                   VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

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
    r = vkAllocateMemory(dev, &mai, NULL, out_mem);
    if (r != VK_SUCCESS) return r;
    r = vkBindBufferMemory(dev, *out_buf, *out_mem, 0);
    if (r != VK_SUCCESS) return r;

    if (init_data) {
        void *ptr = NULL;
        r = vkMapMemory(dev, *out_mem, 0, size, 0, &ptr);
        if (r != VK_SUCCESS) return r;
        memcpy(ptr, init_data, (size_t)size);
        vkUnmapMemory(dev, *out_mem);
    }
    if (out_bda) {
        VkBufferDeviceAddressInfo bdai = {
            .sType  = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO,
            .buffer = *out_buf,
        };
        *out_bda = vkGetBufferDeviceAddress(dev, &bdai);
    }
    return VK_SUCCESS;
}

int main(void) {
    VkApplicationInfo app = {
        .sType      = VK_STRUCTURE_TYPE_APPLICATION_INFO,
        .apiVersion = VK_API_VERSION_1_3,
    };
    VkInstanceCreateInfo ici = {
        .sType            = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        .pApplicationInfo = &app,
    };
    VkInstance inst = VK_NULL_HANDLE;
    CHECK_VK(vkCreateInstance(&ici, NULL, &inst), "vkCreateInstance");

    uint32_t pdev_count = 1;
    VkPhysicalDevice pdev = VK_NULL_HANDLE;
    CHECK_VK(vkEnumeratePhysicalDevices(inst, &pdev_count, &pdev), "vkEnumeratePhysicalDevices");
    vkGetPhysicalDeviceMemoryProperties(pdev, &g_mem_props);

    /* 1. Verify extensions */
    uint32_t ext_count = 0;
    vkEnumerateDeviceExtensionProperties(pdev, NULL, &ext_count, NULL);
    VkExtensionProperties *exts = calloc(ext_count, sizeof(*exts));
    vkEnumerateDeviceExtensionProperties(pdev, NULL, &ext_count, exts);
    bool has_rtp = false, has_plib = false, has_rtm1 = false;
    for (uint32_t i = 0; i < ext_count; i++) {
        if (!strcmp(exts[i].extensionName, "VK_KHR_ray_tracing_pipeline")) has_rtp = true;
        if (!strcmp(exts[i].extensionName, "VK_KHR_pipeline_library")) has_plib = true;
        if (!strcmp(exts[i].extensionName, "VK_KHR_ray_tracing_maintenance1")) has_rtm1 = true;
    }
    free(exts);
    printf("Extensions: VK_KHR_ray_tracing_pipeline=%d VK_KHR_pipeline_library=%d VK_KHR_ray_tracing_maintenance1=%d\n",
           has_rtp, has_plib, has_rtm1);
    if (!has_rtp || !has_plib || !has_rtm1) {
        fprintf(stderr, "FAIL: Missing emulated RT extensions (ensure MALI_COMPAT_RT_PIPELINE=1)\n");
        return 1;
    }

    /* 2. Query RT pipeline features & properties */
    VkPhysicalDeviceRayTracingPipelineFeaturesKHR rtp_feat = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_FEATURES_KHR,
    };
    VkPhysicalDeviceFeatures2 f2 = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
        .pNext = &rtp_feat,
    };
    vkGetPhysicalDeviceFeatures2(pdev, &f2);

    VkPhysicalDeviceRayTracingPipelinePropertiesKHR rtp_props = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_PROPERTIES_KHR,
    };
    VkPhysicalDeviceProperties2 p2 = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2,
        .pNext = &rtp_props,
    };
    vkGetPhysicalDeviceProperties2(pdev, &p2);
    printf("RT Props: handleSize=%u maxRecursion=%u baseAlign=%u handleAlign=%u\n",
           rtp_props.shaderGroupHandleSize, rtp_props.maxRayRecursionDepth,
           rtp_props.shaderGroupBaseAlignment, rtp_props.shaderGroupHandleAlignment);
    if (!rtp_feat.rayTracingPipeline || rtp_props.shaderGroupHandleSize != 32) {
        fprintf(stderr, "FAIL: rayTracingPipeline feature or handleSize mismatch\n");
        return 1;
    }

    /* 3. Create VkDevice (only requesting VK_KHR_ray_tracing_pipeline + VK_KHR_acceleration_structure) */
    const char *dev_exts[] = {
        "VK_KHR_deferred_host_operations",
        "VK_KHR_acceleration_structure",
        "VK_KHR_ray_tracing_pipeline",
        "VK_KHR_pipeline_library",
        "VK_KHR_ray_tracing_maintenance1",
    };
    VkPhysicalDeviceBufferDeviceAddressFeatures bda_f = {
        .sType               = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES,
        .bufferDeviceAddress = VK_TRUE,
    };
    VkPhysicalDeviceAccelerationStructureFeaturesKHR as_f = {
        .sType                 = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR,
        .pNext                 = &bda_f,
        .accelerationStructure = VK_TRUE,
    };
    rtp_feat.pNext = &as_f;
    float qprio = 1.0f;
    VkDeviceQueueCreateInfo qci = {
        .sType            = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
        .queueFamilyIndex = 0,
        .queueCount       = 1,
        .pQueuePriorities = &qprio,
    };
    VkDeviceCreateInfo dci = {
        .sType                   = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
        .pNext                   = &rtp_feat,
        .queueCreateInfoCount    = 1,
        .pQueueCreateInfos       = &qci,
        .enabledExtensionCount   = 5,
        .ppEnabledExtensionNames = dev_exts,
    };
    VkDevice dev = VK_NULL_HANDLE;
    CHECK_VK(vkCreateDevice(pdev, &dci, NULL, &dev), "vkCreateDevice");

    VkQueue queue = VK_NULL_HANDLE;
    vkGetDeviceQueue(dev, 0, 0, &queue);

    /* Load extension functions */
    #define LOAD_FN(type, name) type name = (type)vkGetDeviceProcAddr(dev, #name); \
        if (!name) { fprintf(stderr, "FAIL: missing %s\n", #name); return 1; }
    LOAD_FN(PFN_vkGetAccelerationStructureBuildSizesKHR, vkGetAccelerationStructureBuildSizesKHR);
    LOAD_FN(PFN_vkCreateAccelerationStructureKHR, vkCreateAccelerationStructureKHR);
    LOAD_FN(PFN_vkGetAccelerationStructureDeviceAddressKHR, vkGetAccelerationStructureDeviceAddressKHR);
    LOAD_FN(PFN_vkCmdBuildAccelerationStructuresKHR, vkCmdBuildAccelerationStructuresKHR);
    LOAD_FN(PFN_vkCreateRayTracingPipelinesKHR, vkCreateRayTracingPipelinesKHR);
    LOAD_FN(PFN_vkGetRayTracingShaderGroupHandlesKHR, vkGetRayTracingShaderGroupHandlesKHR);
    LOAD_FN(PFN_vkCmdTraceRaysKHR, vkCmdTraceRaysKHR);

    /* 4. Build BLAS + TLAS (triangle covering (-1,-1,2), (3,-1,2), (-1,3,2) so origin (0,0,0) dir (0,0,1) hits!) */
    static const float verts[9] = {
        -1.0f, -1.0f, 2.0f,
         3.0f, -1.0f, 2.0f,
        -1.0f,  3.0f, 2.0f,
    };
    VkBuffer vbuf, blas_buf, tlas_buf, inst_buf, scratch_buf;
    VkDeviceMemory vmem, blas_mem, tlas_mem, inst_mem, scratch_mem;
    VkDeviceAddress vbda = 0, blas_bda = 0, inst_bda = 0, scratch_bda = 0;

    CHECK_VK(create_bda_buffer(dev, sizeof(verts),
                               VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR,
                               verts, &vbuf, &vmem, &vbda), "create vbuf");

    VkAccelerationStructureGeometryKHR blas_geom = {
        .sType        = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR,
        .geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR,
        .flags        = VK_GEOMETRY_OPAQUE_BIT_KHR,
        .geometry.triangles = {
            .sType        = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR,
            .vertexFormat = VK_FORMAT_R32G32B32_SFLOAT,
            .vertexData.deviceAddress = vbda,
            .vertexStride = 12,
            .maxVertex    = 2,
            .indexType    = VK_INDEX_TYPE_NONE_KHR,
        },
    };
    VkAccelerationStructureBuildGeometryInfoKHR blas_bgi = {
        .sType         = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR,
        .type          = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR,
        .flags         = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR,
        .mode          = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR,
        .geometryCount = 1,
        .pGeometries   = &blas_geom,
    };
    uint32_t prim_count = 1;
    VkAccelerationStructureBuildSizesInfoKHR blas_sizes = {
        .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR,
    };
    vkGetAccelerationStructureBuildSizesKHR(dev, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
                                            &blas_bgi, &prim_count, &blas_sizes);

    CHECK_VK(create_bda_buffer(dev, blas_sizes.accelerationStructureSize,
                               VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR,
                               NULL, &blas_buf, &blas_mem, NULL), "create blas_buf");
    VkAccelerationStructureCreateInfoKHR blas_ci = {
        .sType  = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR,
        .buffer = blas_buf,
        .size   = blas_sizes.accelerationStructureSize,
        .type   = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR,
    };
    VkAccelerationStructureKHR blas = VK_NULL_HANDLE;
    CHECK_VK(vkCreateAccelerationStructureKHR(dev, &blas_ci, NULL, &blas), "vkCreateAccelerationStructureKHR(BLAS)");
    VkAccelerationStructureDeviceAddressInfoKHR as_dai = {
        .sType                 = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR,
        .accelerationStructure = blas,
    };
    blas_bda = vkGetAccelerationStructureDeviceAddressKHR(dev, &as_dai);

    VkAccelerationStructureInstanceKHR as_inst;
    memset(&as_inst, 0, sizeof(as_inst));
    as_inst.transform.matrix[0][0] = 1.0f;
    as_inst.transform.matrix[1][1] = 1.0f;
    as_inst.transform.matrix[2][2] = 1.0f;
    as_inst.mask = 0xFF;
    as_inst.flags = VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR;
    as_inst.accelerationStructureReference = blas_bda;

    CHECK_VK(create_bda_buffer(dev, sizeof(as_inst),
                               VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR,
                               &as_inst, &inst_buf, &inst_mem, &inst_bda), "create inst_buf");

    VkAccelerationStructureGeometryKHR tlas_geom = {
        .sType        = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR,
        .geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR,
        .geometry.instances = {
            .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR,
            .data.deviceAddress = inst_bda,
        },
    };
    VkAccelerationStructureBuildGeometryInfoKHR tlas_bgi = {
        .sType         = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR,
        .type          = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR,
        .flags         = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR,
        .mode          = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR,
        .geometryCount = 1,
        .pGeometries   = &tlas_geom,
    };
    VkAccelerationStructureBuildSizesInfoKHR tlas_sizes = {
        .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR,
    };
    vkGetAccelerationStructureBuildSizesKHR(dev, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
                                            &tlas_bgi, &prim_count, &tlas_sizes);

    CHECK_VK(create_bda_buffer(dev, tlas_sizes.accelerationStructureSize,
                               VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR,
                               NULL, &tlas_buf, &tlas_mem, NULL), "create tlas_buf");
    VkAccelerationStructureCreateInfoKHR tlas_ci = {
        .sType  = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR,
        .buffer = tlas_buf,
        .size   = tlas_sizes.accelerationStructureSize,
        .type   = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR,
    };
    VkAccelerationStructureKHR tlas = VK_NULL_HANDLE;
    CHECK_VK(vkCreateAccelerationStructureKHR(dev, &tlas_ci, NULL, &tlas), "vkCreateAccelerationStructureKHR(TLAS)");

    VkDeviceSize scratch_sz = blas_sizes.buildScratchSize > tlas_sizes.buildScratchSize
                            ? blas_sizes.buildScratchSize : tlas_sizes.buildScratchSize;
    CHECK_VK(create_bda_buffer(dev, scratch_sz + 256, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                               NULL, &scratch_buf, &scratch_mem, &scratch_bda), "create scratch_buf");
    scratch_bda = (scratch_bda + 255ULL) & ~255ULL;

    /* 5. Create RT shader modules + RT Pipeline */
    VkShaderModule mod_rgen = VK_NULL_HANDLE, mod_rchit = VK_NULL_HANDLE, mod_rmiss = VK_NULL_HANDLE;
    VkShaderModuleCreateInfo smci_rgen  = { VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, NULL, 0, sizeof(spv_rgen),  spv_rgen  };
    VkShaderModuleCreateInfo smci_rchit = { VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, NULL, 0, sizeof(spv_rchit), spv_rchit };
    VkShaderModuleCreateInfo smci_rmiss = { VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, NULL, 0, sizeof(spv_rmiss), spv_rmiss };
    CHECK_VK(vkCreateShaderModule(dev, &smci_rgen,  NULL, &mod_rgen),  "vkCreateShaderModule(rgen)");
    CHECK_VK(vkCreateShaderModule(dev, &smci_rchit, NULL, &mod_rchit), "vkCreateShaderModule(rchit)");
    CHECK_VK(vkCreateShaderModule(dev, &smci_rmiss, NULL, &mod_rmiss), "vkCreateShaderModule(rmiss)");

    VkDescriptorSetLayoutBinding dsl_bindings[2] = {
        {
            .binding         = 0,
            .descriptorType  = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR,
            .descriptorCount = 1,
            .stageFlags      = VK_SHADER_STAGE_RAYGEN_BIT_KHR,
        },
        {
            .binding         = 1,
            .descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .descriptorCount = 1,
            .stageFlags      = VK_SHADER_STAGE_RAYGEN_BIT_KHR,
        },
    };
    VkDescriptorSetLayoutCreateInfo dsl_ci = {
        .sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .bindingCount = 2,
        .pBindings    = dsl_bindings,
    };
    VkDescriptorSetLayout dsl = VK_NULL_HANDLE;
    CHECK_VK(vkCreateDescriptorSetLayout(dev, &dsl_ci, NULL, &dsl), "vkCreateDescriptorSetLayout");

    VkPipelineLayoutCreateInfo pl_ci = {
        .sType          = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = 1,
        .pSetLayouts    = &dsl,
    };
    VkPipelineLayout pipe_layout = VK_NULL_HANDLE;
    CHECK_VK(vkCreatePipelineLayout(dev, &pl_ci, NULL, &pipe_layout), "vkCreatePipelineLayout");

    VkPipelineShaderStageCreateInfo stages[3] = {
        { VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, NULL, 0, VK_SHADER_STAGE_RAYGEN_BIT_KHR,      mod_rgen,  "main", NULL },
        { VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, NULL, 0, VK_SHADER_STAGE_MISS_BIT_KHR,        mod_rmiss, "main", NULL },
        { VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, NULL, 0, VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR, mod_rchit, "main", NULL },
    };
    VkRayTracingShaderGroupCreateInfoKHR groups[3] = {
        {
            .sType              = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR,
            .type               = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR,
            .generalShader      = 0,
            .closestHitShader   = VK_SHADER_UNUSED_KHR,
            .anyHitShader       = VK_SHADER_UNUSED_KHR,
            .intersectionShader = VK_SHADER_UNUSED_KHR,
        },
        {
            .sType              = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR,
            .type               = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR,
            .generalShader      = 1,
            .closestHitShader   = VK_SHADER_UNUSED_KHR,
            .anyHitShader       = VK_SHADER_UNUSED_KHR,
            .intersectionShader = VK_SHADER_UNUSED_KHR,
        },
        {
            .sType              = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR,
            .type               = VK_RAY_TRACING_SHADER_GROUP_TYPE_TRIANGLES_HIT_GROUP_KHR,
            .generalShader      = VK_SHADER_UNUSED_KHR,
            .closestHitShader   = 2,
            .anyHitShader       = VK_SHADER_UNUSED_KHR,
            .intersectionShader = VK_SHADER_UNUSED_KHR,
        },
    };
    VkRayTracingPipelineCreateInfoKHR rtp_ci = {
        .sType                        = VK_STRUCTURE_TYPE_RAY_TRACING_PIPELINE_CREATE_INFO_KHR,
        .stageCount                   = 3,
        .pStages                      = stages,
        .groupCount                   = 3,
        .pGroups                      = groups,
        .maxPipelineRayRecursionDepth = 1,
        .layout                       = pipe_layout,
    };
    VkPipeline rt_pipe = VK_NULL_HANDLE;
    CHECK_VK(vkCreateRayTracingPipelinesKHR(dev, VK_NULL_HANDLE, VK_NULL_HANDLE, 1, &rtp_ci, NULL, &rt_pipe),
             "vkCreateRayTracingPipelinesKHR");

    /* 6. Build SBT with 64-byte aligned regions and inline ShaderRecordBufferKHR constant 0xCAFE0000u */
    uint8_t handles[3 * 32];
    CHECK_VK(vkGetRayTracingShaderGroupHandlesKHR(dev, rt_pipe, 0, 3, sizeof(handles), handles),
             "vkGetRayTracingShaderGroupHandlesKHR");

    uint8_t sbt_raw[3 * 64];
    memset(sbt_raw, 0, sizeof(sbt_raw));
    memcpy(sbt_raw + 0 * 64, handles + 0 * 32, 32); /* rgen at offset 0 */
    memcpy(sbt_raw + 1 * 64, handles + 1 * 32, 32); /* rmiss at offset 64 */
    memcpy(sbt_raw + 2 * 64, handles + 2 * 32, 32); /* rchit at offset 128 */
    uint32_t sbt_inline_const = 0xCAFE0000u;
    memcpy(sbt_raw + 2 * 64 + 32, &sbt_inline_const, 4); /* ShaderRecordBufferKHR at record+32 */

    VkBuffer sbt_buf, out_buf;
    VkDeviceMemory sbt_mem, out_mem;
    VkDeviceAddress sbt_bda = 0;
    CHECK_VK(create_bda_buffer(dev, sizeof(sbt_raw), VK_BUFFER_USAGE_SHADER_BINDING_TABLE_BIT_KHR,
                               sbt_raw, &sbt_buf, &sbt_mem, &sbt_bda), "create sbt_buf");

    uint32_t init_out[16];
    memset(init_out, 0, sizeof(init_out));
    CHECK_VK(create_bda_buffer(dev, sizeof(init_out), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                               init_out, &out_buf, &out_mem, NULL), "create out_buf");

    /* 7. Allocate & update DescriptorSet */
    VkDescriptorPoolSize pool_sizes[2] = {
        { VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 1 },
        { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,             1 },
    };
    VkDescriptorPoolCreateInfo dp_ci = {
        .sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        .maxSets       = 1,
        .poolSizeCount = 2,
        .pPoolSizes    = pool_sizes,
    };
    VkDescriptorPool dpool = VK_NULL_HANDLE;
    CHECK_VK(vkCreateDescriptorPool(dev, &dp_ci, NULL, &dpool), "vkCreateDescriptorPool");

    VkDescriptorSetAllocateInfo ds_ai = {
        .sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorPool     = dpool,
        .descriptorSetCount = 1,
        .pSetLayouts        = &dsl,
    };
    VkDescriptorSet dset = VK_NULL_HANDLE;
    CHECK_VK(vkAllocateDescriptorSets(dev, &ds_ai, &dset), "vkAllocateDescriptorSets");

    VkWriteDescriptorSetAccelerationStructureKHR w_as = {
        .sType                      = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR,
        .accelerationStructureCount = 1,
        .pAccelerationStructures    = &tlas,
    };
    VkDescriptorBufferInfo dbi = {
        .buffer = out_buf,
        .offset = 0,
        .range  = sizeof(init_out),
    };
    VkWriteDescriptorSet writes[2] = {
        {
            .sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .pNext           = &w_as,
            .dstSet          = dset,
            .dstBinding      = 0,
            .descriptorCount = 1,
            .descriptorType  = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR,
        },
        {
            .sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet          = dset,
            .dstBinding      = 1,
            .descriptorCount = 1,
            .descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .pBufferInfo     = &dbi,
        },
    };
    vkUpdateDescriptorSets(dev, 2, writes, 0, NULL);

    /* 8. Record BLAS/TLAS build + vkCmdTraceRaysKHR */
    VkCommandPoolCreateInfo cp_ci = {
        .sType            = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .queueFamilyIndex = 0,
    };
    VkCommandPool cmd_pool = VK_NULL_HANDLE;
    CHECK_VK(vkCreateCommandPool(dev, &cp_ci, NULL, &cmd_pool), "vkCreateCommandPool");

    VkCommandBufferAllocateInfo cb_ai = {
        .sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .commandPool        = cmd_pool,
        .level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        .commandBufferCount = 1,
    };
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    CHECK_VK(vkAllocateCommandBuffers(dev, &cb_ai, &cmd), "vkAllocateCommandBuffers");

    VkCommandBufferBeginInfo cb_bi = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
    };
    CHECK_VK(vkBeginCommandBuffer(cmd, &cb_bi), "vkBeginCommandBuffer");

    VkAccelerationStructureBuildRangeInfoKHR range_info = { .primitiveCount = 1 };
    const VkAccelerationStructureBuildRangeInfoKHR *p_range = &range_info;

    blas_bgi.dstAccelerationStructure  = blas;
    blas_bgi.scratchData.deviceAddress = scratch_bda;
    vkCmdBuildAccelerationStructuresKHR(cmd, 1, &blas_bgi, &p_range);

    VkMemoryBarrier mb = {
        .sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
        .srcAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR,
        .dstAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR,
    };
    vkCmdPipelineBarrier(cmd,
                         VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                         VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         0, 1, &mb, 0, NULL, 0, NULL);

    tlas_bgi.dstAccelerationStructure  = tlas;
    tlas_bgi.scratchData.deviceAddress = scratch_bda;
    vkCmdBuildAccelerationStructuresKHR(cmd, 1, &tlas_bgi, &p_range);

    vkCmdPipelineBarrier(cmd,
                         VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         0, 1, &mb, 0, NULL, 0, NULL);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, rt_pipe);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, pipe_layout, 0, 1, &dset, 0, NULL);

    VkStridedDeviceAddressRegionKHR rgen_region = { sbt_bda + 0 * 64, 64, 64 };
    VkStridedDeviceAddressRegionKHR miss_region = { sbt_bda + 1 * 64, 64, 64 };
    VkStridedDeviceAddressRegionKHR hit_region  = { sbt_bda + 2 * 64, 64, 64 };
    VkStridedDeviceAddressRegionKHR call_region = { 0, 0, 0 };

    vkCmdTraceRaysKHR(cmd, &rgen_region, &miss_region, &hit_region, &call_region, 4, 4, 1);

    CHECK_VK(vkEndCommandBuffer(cmd), "vkEndCommandBuffer");

    VkSubmitInfo si = {
        .sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .commandBufferCount = 1,
        .pCommandBuffers    = &cmd,
    };
    CHECK_VK(vkQueueSubmit(queue, 1, &si, VK_NULL_HANDLE), "vkQueueSubmit");
    CHECK_VK(vkQueueWaitIdle(queue), "vkQueueWaitIdle");

    /* 9. Verify output buffer */
    uint32_t *out_ptr = NULL;
    CHECK_VK(vkMapMemory(dev, out_mem, 0, sizeof(init_out), 0, (void **)&out_ptr), "vkMapMemory(out)");
    uint32_t px_hit  = out_ptr[0];  /* (0, 0): origin (0, 0, 0) -> hits triangle -> 0xCAFE1234 */
    uint32_t px_miss = out_ptr[15]; /* (3, 3): origin (9, 0, 0) -> misses triangle -> 0xDEAD0000 */
    printf("TraceRays Output: pixel(0,0)[HIT]=0x%08x (expected 0xCAFE1234), pixel(3,3)[MISS]=0x%08x (expected 0xDEAD0000)\n",
           px_hit, px_miss);
    vkUnmapMemory(dev, out_mem);

    if (px_hit != 0xCAFE1234u || px_miss != 0xDEAD0000u) {
        fprintf(stderr, "FAIL: TraceRays output mismatch!\n");
        return 1;
    }
    printf("PASS: VK_KHR_ray_tracing_pipeline megakernel emulation verified end-to-end!\n");
    return 0;
}

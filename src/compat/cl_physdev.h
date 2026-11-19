// SPDX-License-Identifier: GPL-3.0-or-later
// cl_physdev.h: physical device: features, properties, limits, formats, dma-buf, memory props, extensions. Included by mali_compat_layer.c.
// -----------------------------------------------------------------------------
// Intercepted Vulkan Entry Points
// -----------------------------------------------------------------------------
// MALI_COMPAT_GS_HIDE_PROCS=<comm prefixes, comma-separated>: those processes are told the device has no geometry
// shaders. Xwayland's glamor draws GL_QUADS in a compatibility context and Mesa Zink turns them into geometry shaders;
// MaliBridge found that pipelines with a geometry stage crash some Mali drivers, so Xwayland segfaults (their
// "gs_hide"). Without the feature Zink takes its other path. For the session on the bridge. [Claude Opus 5.5] 2026-10-30
static bool gs_hidden(void) {
    static int hidden = -1;
    if (hidden >= 0) return hidden;
    const char *list = getenv("MALI_COMPAT_GS_HIDE_PROCS");
    char comm[32] = "";
    FILE *f = fopen("/proc/self/comm", "r");
    if (f) { if (fgets(comm, sizeof comm, f)) comm[strcspn(comm, "\n")] = 0; fclose(f); }
    hidden = 0;
    for (const char *p = list ? list : ""; *p; ) {
        size_t n = strcspn(p, ",");
        if (n && !strncmp(comm, p, n)) { hidden = 1; break; }
        p += n + (p[n] == ',');
    }
    if (hidden) fprintf(stderr, "mali-compat: geometry shaders hidden from '%s'\n", comm);
    return hidden;
}

// MALI_COMPAT_FP16_HIDE_PROCS=<comm prefixes>: shaderFloat16 / shaderInt16 hidden from those processes (gamescope
// picks its FP16 composite shaders when the device has both). Diagnostic for the session on the bridge, where
// gamescope's composited output stayed all zero. [Claude Opus 5.5] 2026-10-30
static bool fp16_hidden(void) {
    static int hidden = -1;
    if (hidden >= 0) return hidden;
    const char *list = getenv("MALI_COMPAT_FP16_HIDE_PROCS");
    char comm[32] = "";
    FILE *f = fopen("/proc/self/comm", "r");
    if (f) { if (fgets(comm, sizeof comm, f)) comm[strcspn(comm, "\n")] = 0; fclose(f); }
    hidden = 0;
    for (const char *p = list ? list : ""; *p; ) {
        size_t n = strcspn(p, ",");
        if (n && !strncmp(comm, p, n)) { hidden = 1; break; }
        p += n + (p[n] == ',');
    }
    if (hidden) fprintf(stderr, "mali-compat: FP16 hidden from '%s'\n", comm);
    return hidden;
}

static VKAPI_ATTR void VKAPI_CALL compat_GetPhysicalDeviceFeatures(
    VkPhysicalDevice physicalDevice, VkPhysicalDeviceFeatures *pFeatures) {
    struct InstanceData *inst = get_instance_by_phys_dev(physicalDevice);
    if (inst && inst->fpGetPhysicalDeviceFeatures)
        inst->fpGetPhysicalDeviceFeatures(physicalDevice, pFeatures);

    pFeatures->textureCompressionBC = VK_TRUE;
    pFeatures->shaderClipDistance = VK_TRUE;
    pFeatures->shaderCullDistance = VK_TRUE;
    pFeatures->fillModeNonSolid = VK_TRUE;
    if (gs_hidden()) pFeatures->geometryShader = VK_FALSE;
    if (fp16_hidden()) pFeatures->shaderInt16 = VK_FALSE;
}

static VKAPI_ATTR void VKAPI_CALL compat_GetPhysicalDeviceFeatures2(
    VkPhysicalDevice physicalDevice, VkPhysicalDeviceFeatures2 *pFeatures) {
    struct InstanceData *inst = get_instance_by_phys_dev(physicalDevice);
    struct unlinked u = unlink_struct(pFeatures, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAP_MEMORY_PLACED_FEATURES_EXT);
    struct unlinked u_rtp  = g_rt_pipeline ? unlink_struct(pFeatures, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_FEATURES_KHR) : (struct unlinked){0};
    struct unlinked u_rtm1 = g_rt_pipeline ? unlink_struct(pFeatures, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_MAINTENANCE_1_FEATURES_KHR) : (struct unlinked){0};
    struct unlinked u_plgh = g_rt_pipeline ? unlink_struct(pFeatures, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PIPELINE_LIBRARY_GROUP_HANDLES_FEATURES_EXT) : (struct unlinked){0};
    if (inst && inst->fpGetPhysicalDeviceFeatures2)
        inst->fpGetPhysicalDeviceFeatures2(physicalDevice, pFeatures);
    relink(u_plgh);
    relink(u_rtm1);
    relink(u_rtp);
    relink(u);

    pFeatures->features.textureCompressionBC = VK_TRUE;
    pFeatures->features.shaderClipDistance = VK_TRUE;
    pFeatures->features.shaderCullDistance = VK_TRUE;
    pFeatures->features.fillModeNonSolid = VK_TRUE;
    if (gs_hidden()) pFeatures->features.geometryShader = VK_FALSE;
    if (fp16_hidden()) {
        pFeatures->features.shaderInt16 = VK_FALSE;
        for (VkBaseOutStructure *h = (VkBaseOutStructure *)pFeatures->pNext; h; h = h->pNext) {
            if (h->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES)
                ((VkPhysicalDeviceVulkan12Features *)h)->shaderFloat16 = VK_FALSE;
            else if (h->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES)
                ((VkPhysicalDeviceShaderFloat16Int8Features *)h)->shaderFloat16 = VK_FALSE;
        }
    }

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
        } else if (header->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES) {
            // DOOM Eternal (idTech 7) checks storageBuffer16BitAccess via vkGetPhysicalDeviceFeatures2.
            VkPhysicalDevice16BitStorageFeatures *f16 = (VkPhysicalDevice16BitStorageFeatures *)header;
            f16->storageBuffer16BitAccess = VK_TRUE;
            f16->uniformAndStorageBuffer16BitAccess = VK_TRUE;
            f16->storagePushConstant16 = VK_TRUE;
            f16->storageInputOutput16 = VK_TRUE;
        } else if (header->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_8BIT_STORAGE_FEATURES) {
            VkPhysicalDevice8BitStorageFeatures *f8 = (VkPhysicalDevice8BitStorageFeatures *)header;
            f8->storageBuffer8BitAccess = VK_TRUE;
            f8->uniformAndStorageBuffer8BitAccess = VK_TRUE;
            f8->storagePushConstant8 = VK_TRUE;
        } else if (header->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES) {
            VkPhysicalDeviceVulkan11Features *v11 = (VkPhysicalDeviceVulkan11Features *)header;
            v11->storageBuffer16BitAccess = VK_TRUE;
            v11->uniformAndStorageBuffer16BitAccess = VK_TRUE;
            v11->storagePushConstant16 = VK_TRUE;
            v11->storageInputOutput16 = VK_TRUE;
        } else if (header->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES) {
            VkPhysicalDeviceVulkan12Features *v12 = (VkPhysicalDeviceVulkan12Features *)header;
            v12->storageBuffer8BitAccess = VK_TRUE;
            v12->uniformAndStorageBuffer8BitAccess = VK_TRUE;
            v12->storagePushConstant8 = VK_TRUE;
        } else if (g_rt_pipeline && header->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_FEATURES_KHR) {
            VkPhysicalDeviceRayTracingPipelineFeaturesKHR *rtp = (VkPhysicalDeviceRayTracingPipelineFeaturesKHR *)header;
            rtp->rayTracingPipeline = VK_TRUE;
            rtp->rayTracingPipelineShaderGroupHandleCaptureReplay = VK_FALSE;
            rtp->rayTracingPipelineShaderGroupHandleCaptureReplayMixed = VK_FALSE;
            rtp->rayTracingPipelineTraceRaysIndirect = VK_TRUE;
            rtp->rayTraversalPrimitiveCulling = VK_TRUE;
        } else if (g_rt_pipeline && header->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_MAINTENANCE_1_FEATURES_KHR) {
            VkPhysicalDeviceRayTracingMaintenance1FeaturesKHR *rtm1 = (VkPhysicalDeviceRayTracingMaintenance1FeaturesKHR *)header;
            rtm1->rayTracingMaintenance1 = VK_TRUE;
            rtm1->rayTracingPipelineTraceRaysIndirect2 = VK_TRUE;
        } else if (g_rt_pipeline && header->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PIPELINE_LIBRARY_GROUP_HANDLES_FEATURES_EXT) {
            VkPhysicalDevicePipelineLibraryGroupHandlesFeaturesEXT *plgh = (VkPhysicalDevicePipelineLibraryGroupHandlesFeaturesEXT *)header;
            plgh->pipelineLibraryGroupHandles = VK_TRUE;
        }
        header = header->pNext;
    }
}

// MALI_COMPAT_VENDOR_ID=<hex>: report this PCI vendor ID instead of Arm's 0x13b5 (driverID stays ARM_PROPRIETARY).
// vkd3d-proton reads root CBVs through buffer device addresses on every vendor except NVIDIA and Qualcomm, which
// get push descriptors ("~15%" there; Mali can only preload UBOs into uniform registers). 5143 flips only that
// choice: vkd3d's other Qualcomm check also needs driverID TURNIP. The game sees the vendor through DXGI too
// (dxgi.customVendorId keeps it Arm). [Claude Opus 5.5] 2026-10-28
// When MALI_COMPAT_VENDOR_ID is set (e.g. 10de for NVIDIA, 1002 for AMD), MALI_COMPAT_DEVICE_ID and
// MALI_COMPAT_DEVICE_NAME can optionally override the device, or sensible desktop defaults are chosen so native
// Vulkan games (DOOM Eternal / idTech) recognize the GPU.
static uint32_t g_vendor_id;
static uint32_t g_device_id;
static const char *g_device_name;

// 752c09f's two DOOM Eternal overrides (subgroup supportedStages/Operations widened to VERTEX/TESS/GEOM, and
// minStorageBufferOffsetAlignment clamped to 16) are OPT-IN since 11-07: they break Watch Dogs 2 (Marcus renders as floating
// clothes, then a crash; with the real limits it renders correctly). DOOM Eternal needs both to start, RE Village the clamp to
// load (docs/games/doom_eternal.md, docs/games/resident_evil_village.md). Default: the driver's real limits.
// MALI_COMPAT_FAKE_LIMITS=1: both overrides. MALI_COMPAT_FAKE_SUBGROUP=1 / MALI_COMPAT_FAKE_ALIGN=1: one half alone.
// The old MALI_COMPAT_REAL_SUBGROUP / _REAL_ALIGN / _REAL_LIMITS keys force the real values (they win over FAKE_*), so old
// env entries stay harmless. [keeper, Claude Opus 5.5] 2026-11-07
static int real_on(const char *e) { return e && *e && strcmp(e, "0") != 0; }
static int g_real_limits[2]; // [0] subgroup, [1] align: 1 when the REAL value is reported (set once, real_limits_init)
static void real_limits_init(void) {
    int fake_both = real_on(getenv("MALI_COMPAT_FAKE_LIMITS"));
    int fake_sg = fake_both || real_on(getenv("MALI_COMPAT_FAKE_SUBGROUP"));
    int fake_al = fake_both || real_on(getenv("MALI_COMPAT_FAKE_ALIGN"));
    int real_both = real_on(getenv("MALI_COMPAT_REAL_LIMITS"));
    int real_sg = real_both || real_on(getenv("MALI_COMPAT_REAL_SUBGROUP"));
    int real_al = real_both || real_on(getenv("MALI_COMPAT_REAL_ALIGN"));
    g_real_limits[0] = !fake_sg || real_sg;
    g_real_limits[1] = !fake_al || real_al;
    fprintf(stderr, "mali-compat: limits mode: subgroup %s, storage-buffer offset alignment %s (FAKE_SUBGROUP=%d FAKE_ALIGN=%d%s)\n",
            g_real_limits[0] ? "real" : "widened (FAKE)", g_real_limits[1] ? "real" : "clamped to 16 (FAKE)", fake_sg, fake_al,
            (real_sg || real_al) ? ", a REAL_* key forces real" : "");
}
// pthread_once: two threads asking first must not see half the config (Codex 1110-234039: v[1] still -1 read as "real").
static int real_limits_cfg(int which) { // 0 = subgroup, 1 = align; returns 1 when the REAL value is reported
    static pthread_once_t once = PTHREAD_ONCE_INIT;
    pthread_once(&once, real_limits_init);
    return g_real_limits[which];
}
#define real_subgroup() real_limits_cfg(0)
#define real_align() real_limits_cfg(1)
#define SG_WIDE_STAGES (VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT | \
                        VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT | VK_SHADER_STAGE_GEOMETRY_BIT | \
                        VK_SHADER_STAGE_FRAGMENT_BIT | VK_SHADER_STAGE_COMPUTE_BIT)
#define SG_WIDE_OPS (VK_SUBGROUP_FEATURE_BASIC_BIT | VK_SUBGROUP_FEATURE_VOTE_BIT | VK_SUBGROUP_FEATURE_ARITHMETIC_BIT | \
                     VK_SUBGROUP_FEATURE_BALLOT_BIT | VK_SUBGROUP_FEATURE_SHUFFLE_BIT | \
                     VK_SUBGROUP_FEATURE_SHUFFLE_RELATIVE_BIT | VK_SUBGROUP_FEATURE_CLUSTERED_BIT | VK_SUBGROUP_FEATURE_QUAD_BIT)
// RT emulation (MALI_COMPAT_RT_PIPELINE, rtx-clone-fix-1110) adds the RT stages to the subgroup stage mask in both modes: the
// emulated RT stages run as compute, so they have the compute subgroup support. [night keeper, Claude Opus 5.5] 2026-11-10
static uint32_t sg_stages(uint32_t real) {
    return (real_subgroup() ? real : real | SG_WIDE_STAGES) | (g_rt_pipeline ? RT_SHADER_STAGE_MASK : 0);
}
static uint32_t sg_ops(uint32_t real) { return real_subgroup() ? real : real | SG_WIDE_OPS; }

static void apply_properties_overrides(VkPhysicalDeviceProperties *p) {
    p->limits.minTexelBufferOffsetAlignment = 16;
    if (!real_align() && p->limits.minStorageBufferOffsetAlignment > 16)
        p->limits.minStorageBufferOffsetAlignment = 16;
    if (g_vendor_id) {
        p->vendorID = g_vendor_id;
        if (p->limits.maxUniformBufferRange > 65536)
            p->limits.maxUniformBufferRange = 65536;
        if (g_device_id) {
            p->deviceID = g_device_id;
        } else if (g_vendor_id == 0x10de) {
            p->deviceID = 0x1eb1; // NVIDIA GeForce RTX 2080
            p->deviceType = VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU;
            p->driverVersion = (550 << 22) | (54 << 14) | (14 << 6);
        } else if (g_vendor_id == 0x1002) {
            p->deviceID = 0x73bf; // AMD Radeon RX 6800
            p->deviceType = VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU;
            p->driverVersion = (24 << 22) | (3 << 12) | 1;
        } else if (g_vendor_id == 0x8086) {
            p->deviceID = 0x56a0; // Intel Arc A770
            p->deviceType = VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU;
            p->driverVersion = (31 << 22) | (0 << 12) | 101;
        }
        if (g_device_name && *g_device_name) {
            snprintf(p->deviceName, sizeof(p->deviceName), "%s", g_device_name);
        } else if (g_vendor_id == 0x10de) {
            snprintf(p->deviceName, sizeof(p->deviceName), "NVIDIA GeForce RTX 2080");
        } else if (g_vendor_id == 0x1002) {
            snprintf(p->deviceName, sizeof(p->deviceName), "AMD Radeon RX 6800");
        } else if (g_vendor_id == 0x8086) {
            snprintf(p->deviceName, sizeof(p->deviceName), "Intel(R) Arc(TM) A770 Graphics");
        }
    }
}
// GPU timing diagnostics for instanced vertex input (HZD's missing static meshes, 2026-10-29): this driver has
// supportsNonZeroFirstInstance = false, so a draw with firstInstance != 0 on a binding whose divisor is not 1 (the
// emulated zero divisor included) reads undefined instance data. Pipelines and dynamic vertex inputs with such
// bindings, and draws with firstInstance != 0.
static _Atomic uint64_t g_div_pipes, g_div_vi, g_div_draws;
// MALI_COMPAT_NO_SUSPEND_RESUME=1 (own switch for the CmdBeginRendering hook since 2026-10-31, no GPU_TIME needed;
// without GPU_TIME the hook does no timing work, [keeper, Claude Opus 5.5]): vkd3d-proton splits render passes on
// tilers with VK_RENDERING_SUSPENDING/RESUMING_BIT. Turn each part into a plain pass: a suspending part stores its
// attachments, a resuming part loads them, and only the last part resolves. Same result, more bandwidth; tests whether
// this driver loses draws in resumed passes (HZD's missing static meshes, 2026-10-29). Counters of such passes.
static bool g_no_susres;
static _Atomic uint64_t g_susp_passes, g_resu_passes;
// MALI_COMPAT_SUSRES_BARRIER=1 (K-CS1, 2026-10-31, [keeper, Claude Opus 5.5]; only with NO_SUSPEND_RESUME): as plain
// passes, the parts of a split render pass are separate render pass instances, so the previous part's attachment writes
// (draws, STORE) and the next part's LOAD and draws are a hazard. vkd3d-proton cannot record a barrier there (forbidden
// between suspend and resume), so the layer inserts one global memory barrier before each converted resuming part.
// Control's 16384x2048 d32 shadow atlas is split into 7-9 parts per frame past the gate (shadow flicker, control.md).
static bool g_susres_barrier;
static _Atomic uint64_t g_susres_barriers;
static __thread bool tl_div_bound; // the last vertex input this thread set or bound has a divisor != 1 (approximate)

// VK_EXT_physical_device_drm, emulated (MALI_COMPAT_DRM_NODE=<device node>, e.g. /dev/mali0). gamescope refuses a
// physical device without a DRM primary and render node and then finds the render node through
// drmGetDeviceFromDevId(); DroidDeck exposes the GPU node under /dev/dri with a matching /sys/dev/char tree (the
// stand-in PanVK reported for the session until now). The vendor driver has no such extension, so the layer reports
// the node's major:minor as both nodes, for the session processes (gamescope, Steam) on the bridge. [Claude Opus 5.5]
// 2026-10-30
static bool g_drm_on;
static unsigned g_drm_major, g_drm_minor;
static void drm_node_init(void) {
    const char *path = getenv("MALI_COMPAT_DRM_NODE");
    struct stat st;
    g_drm_on = path && *path && stat(path, &st) == 0 && S_ISCHR(st.st_mode);
    if (g_drm_on) { g_drm_major = major(st.st_rdev); g_drm_minor = minor(st.st_rdev); }
    else if (path && *path) fprintf(stderr, "mali-compat: MALI_COMPAT_DRM_NODE=%s is not a character device\n", path);
}

static VKAPI_ATTR void VKAPI_CALL compat_GetPhysicalDeviceProperties2(
    VkPhysicalDevice physicalDevice, VkPhysicalDeviceProperties2 *pProperties) {
    struct InstanceData *inst = get_instance_by_phys_dev(physicalDevice);
    struct unlinked u = unlink_struct(pProperties, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAP_MEMORY_PLACED_PROPERTIES_EXT);
    struct unlinked ud = g_drm_on ? unlink_struct(pProperties, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRM_PROPERTIES_EXT)
                                  : (struct unlinked){0};
    struct unlinked urtp = g_rt_pipeline ? unlink_struct(pProperties, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_PROPERTIES_KHR)
                                         : (struct unlinked){0};
    if (inst && inst->fpGetPhysicalDeviceProperties2)
        inst->fpGetPhysicalDeviceProperties2(physicalDevice, pProperties);
    relink(urtp);
    relink(ud);
    relink(u);
    if (urtp.node) {
        VkPhysicalDeviceRayTracingPipelinePropertiesKHR *rtpp = (VkPhysicalDeviceRayTracingPipelinePropertiesKHR *)urtp.node;
        rtpp->shaderGroupHandleSize              = RT_GROUP_HANDLE_SIZE;
        rtpp->maxRayRecursionDepth               = 2;
        rtpp->maxShaderGroupStride               = 4096;
        rtpp->shaderGroupBaseAlignment           = 64;
        rtpp->shaderGroupHandleCaptureReplaySize = RT_GROUP_HANDLE_SIZE;
        rtpp->maxRayDispatchInvocationCount      = 67108864u;
        rtpp->shaderGroupHandleAlignment         = RT_GROUP_HANDLE_SIZE;
        rtpp->maxRayHitAttributeSize             = 64;
    }
    if (ud.node) {
        VkPhysicalDeviceDrmPropertiesEXT *drm = (VkPhysicalDeviceDrmPropertiesEXT *)ud.node;
        drm->hasPrimary = drm->hasRender = VK_TRUE;
        drm->primaryMajor = drm->renderMajor = g_drm_major;
        drm->primaryMinor = drm->renderMinor = g_drm_minor;
    }
    if (u.node) ((VkPhysicalDeviceMapMemoryPlacedPropertiesEXT *)u.node)->minPlacedMemoryMapAlignment = page_size();
    apply_properties_overrides(&pProperties->properties);
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
        } else if (h->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES) {
            VkPhysicalDeviceSubgroupProperties *s = (VkPhysicalDeviceSubgroupProperties *)h;
            s->supportedStages = sg_stages(s->supportedStages);
            s->supportedOperations = sg_ops(s->supportedOperations);
        } else if (h->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_PROPERTIES) {
            VkPhysicalDeviceVulkan11Properties *v11 = (VkPhysicalDeviceVulkan11Properties *)h;
            v11->subgroupSupportedStages = sg_stages(v11->subgroupSupportedStages);
            v11->subgroupSupportedOperations = sg_ops(v11->subgroupSupportedOperations);
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

static VKAPI_ATTR void VKAPI_CALL compat_GetPhysicalDeviceProperties(
    VkPhysicalDevice physicalDevice, VkPhysicalDeviceProperties *pProperties) {
    struct InstanceData *inst = get_instance_by_phys_dev(physicalDevice);
    if (inst && inst->fpGetPhysicalDeviceProperties) {
        inst->fpGetPhysicalDeviceProperties(physicalDevice, pProperties);
    } else if (inst && inst->fpGetPhysicalDeviceProperties2) {
        VkPhysicalDeviceProperties2 p2 = { .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2 };
        compat_GetPhysicalDeviceProperties2(physicalDevice, &p2);
        *pProperties = p2.properties;
        return;
    }
    apply_properties_overrides(pProperties);
}

static inline bool is_yuv_nv12(VkFormat f) {
    return f == VK_FORMAT_G8_B8R8_2PLANE_420_UNORM || f == VK_FORMAT_G8_B8_R8_3PLANE_420_UNORM;
}

static inline bool is_rtas_vbo_format(VkFormat f) {
    switch (f) {
    case VK_FORMAT_R32G32_SFLOAT:
    case VK_FORMAT_R32G32B32_SFLOAT:
    case VK_FORMAT_R16G16_SFLOAT:
    case VK_FORMAT_R16G16_SNORM:
    case VK_FORMAT_R16G16B16A16_SFLOAT:
    case VK_FORMAT_R16G16B16A16_SNORM:
    case VK_FORMAT_R16G16B16A16_UNORM:
    case VK_FORMAT_R16G16_UNORM:
    case VK_FORMAT_A2B10G10R10_UNORM_PACK32:
    case VK_FORMAT_R8G8B8A8_UNORM:
    case VK_FORMAT_R8G8_UNORM:
    case VK_FORMAT_R8G8B8A8_SNORM:
    case VK_FORMAT_R8G8_SNORM:
        return true;
    default:
        return false;
    }
}

static VKAPI_ATTR void VKAPI_CALL compat_GetPhysicalDeviceFormatProperties(
    VkPhysicalDevice physicalDevice, VkFormat format, VkFormatProperties *pFormatProperties) {
    struct InstanceData *inst = get_instance_by_phys_dev(physicalDevice);
    if (!inst || !inst->fpGetPhysicalDeviceFormatProperties) return;
    if (is_yuv_nv12(format)) {
        inst->fpGetPhysicalDeviceFormatProperties(physicalDevice, format, pFormatProperties);
        pFormatProperties->optimalTilingFeatures |= (VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT |
                                                    VK_FORMAT_FEATURE_TRANSFER_DST_BIT |
                                                    VK_FORMAT_FEATURE_TRANSFER_SRC_BIT |
                                                    VK_FORMAT_FEATURE_COSITED_CHROMA_SAMPLES_BIT |
                                                    VK_FORMAT_FEATURE_MIDPOINT_CHROMA_SAMPLES_BIT);
    } else if (panvk_format_is_bc(format)) {
        VkFormat decoded = panvk_bc_decoded_format(format);
        inst->fpGetPhysicalDeviceFormatProperties(physicalDevice, decoded, pFormatProperties);
        pFormatProperties->linearTilingFeatures = 0;
        pFormatProperties->optimalTilingFeatures = (VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT |
                                                   VK_FORMAT_FEATURE_BLIT_SRC_BIT |
                                                   VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT |
                                                   VK_FORMAT_FEATURE_TRANSFER_SRC_BIT |
                                                   VK_FORMAT_FEATURE_TRANSFER_DST_BIT);
    } else {
        inst->fpGetPhysicalDeviceFormatProperties(physicalDevice, format, pFormatProperties);
    }
    if (g_rt_pipeline && is_rtas_vbo_format(format)) {
        pFormatProperties->bufferFeatures |= VK_FORMAT_FEATURE_ACCELERATION_STRUCTURE_VERTEX_BUFFER_BIT_KHR;
    }
}

static VKAPI_ATTR void VKAPI_CALL compat_GetPhysicalDeviceFormatProperties2(
    VkPhysicalDevice physicalDevice, VkFormat format, VkFormatProperties2 *pFormatProperties) {
    struct InstanceData *inst = get_instance_by_phys_dev(physicalDevice);
    if (!inst || !inst->fpGetPhysicalDeviceFormatProperties2) return;
    if (is_yuv_nv12(format)) {
        inst->fpGetPhysicalDeviceFormatProperties2(physicalDevice, format, pFormatProperties);
        pFormatProperties->formatProperties.optimalTilingFeatures |= (VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT |
                                                                     VK_FORMAT_FEATURE_TRANSFER_DST_BIT |
                                                                     VK_FORMAT_FEATURE_TRANSFER_SRC_BIT |
                                                                     VK_FORMAT_FEATURE_COSITED_CHROMA_SAMPLES_BIT |
                                                                     VK_FORMAT_FEATURE_MIDPOINT_CHROMA_SAMPLES_BIT);
    } else if (panvk_format_is_bc(format)) {
        VkFormat decoded = panvk_bc_decoded_format(format);
        inst->fpGetPhysicalDeviceFormatProperties2(physicalDevice, decoded, pFormatProperties);
        pFormatProperties->formatProperties.linearTilingFeatures = 0;
        pFormatProperties->formatProperties.optimalTilingFeatures = (VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT |
                                                                    VK_FORMAT_FEATURE_BLIT_SRC_BIT |
                                                                    VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT |
                                                                    VK_FORMAT_FEATURE_TRANSFER_SRC_BIT |
                                                                    VK_FORMAT_FEATURE_TRANSFER_DST_BIT);
    } else {
        inst->fpGetPhysicalDeviceFormatProperties2(physicalDevice, format, pFormatProperties);
    }
    if (g_rt_pipeline && is_rtas_vbo_format(format)) {
        pFormatProperties->formatProperties.bufferFeatures |= VK_FORMAT_FEATURE_ACCELERATION_STRUCTURE_VERTEX_BUFFER_BIT_KHR;
        VkFormatProperties3 *p3 = (VkFormatProperties3 *)find_struct(pFormatProperties->pNext, VK_STRUCTURE_TYPE_FORMAT_PROPERTIES_3);
        if (p3) {
            p3->bufferFeatures |= (VkFormatFeatureFlags2)VK_FORMAT_FEATURE_ACCELERATION_STRUCTURE_VERTEX_BUFFER_BIT_KHR;
        }
    }
}

static VKAPI_ATTR VkResult VKAPI_CALL compat_GetPhysicalDeviceImageFormatProperties(
    VkPhysicalDevice physicalDevice, VkFormat format, VkImageType type, VkImageTiling tiling,
    VkImageUsageFlags usage, VkImageCreateFlags flags, VkImageFormatProperties *pImageFormatProperties) {
    struct InstanceData *inst = get_instance_by_phys_dev(physicalDevice);
    if (!inst || !inst->fpGetPhysicalDeviceImageFormatProperties) return VK_ERROR_INITIALIZATION_FAILED;
    if (is_yuv_nv12(format)) {
        VkResult r = inst->fpGetPhysicalDeviceImageFormatProperties(
            physicalDevice, format, type, tiling, usage, flags, pImageFormatProperties);
        if (r != VK_SUCCESS && pImageFormatProperties) {
            pImageFormatProperties->maxExtent = (VkExtent3D){4096, 4096, 1};
            pImageFormatProperties->maxMipLevels = 1;
            pImageFormatProperties->maxArrayLayers = 1;
            pImageFormatProperties->sampleCounts = VK_SAMPLE_COUNT_1_BIT;
            pImageFormatProperties->maxResourceSize = 4096 * 4096 * 2;
            return VK_SUCCESS;
        }
        return r;
    }
    if (panvk_format_is_bc(format)) {
        format = panvk_bc_decoded_format(format);
        flags &= ~VK_IMAGE_CREATE_BLOCK_TEXEL_VIEW_COMPATIBLE_BIT;
    }
    return inst->fpGetPhysicalDeviceImageFormatProperties(
        physicalDevice, format, type, tiling, usage, flags, pImageFormatProperties);
}

// -----------------------------------------------------------------------------
// dma-buf export, emulated (MALI_COMPAT_DMABUF_EXPORT=1). [Claude Opus 5.5] 2026-10-30
// The vendor driver imports linear dma-bufs (explicit DRM-modifier layout) but exports none: its external memory
// features for dma-buf are IMPORTABLE only, and AFBC/AFRC modifiers are refused (test/session-probe.c). gamescope's
// nested Wayland backend allocates its output images as exportable dma-bufs and hands them to the parent compositor,
// so on the bridge it asserted ("modifiers.size() > 0"). With the switch:
// - linear dma-buf image formats report EXPORTABLE too;
// - an exportable dma-buf image created from a modifier list gets an explicit linear layout (pitch aligned to 256);
// - its memory comes from the system dma-heap and is imported into the driver; vkGetMemoryFdKHR hands out that fd.
// Mesa's private WSI structures in gamescope's create infos are dropped on the way (the vendor does not know them).
// -----------------------------------------------------------------------------
static bool g_dmabuf_export;
#define STYPE_WSI_IMAGE_CREATE_INFO_MESA ((VkStructureType)1000001002)
#define STYPE_WSI_MEMORY_ALLOCATE_INFO_MESA ((VkStructureType)1000001003)
struct ExportMem { VkDeviceMemory mem; int fd; uint64_t size; bool imported; struct ExportMem *next; };
static struct ExportMem *g_export_mem;
static pthread_mutex_t g_export_mutex = PTHREAD_MUTEX_INITIALIZER;

struct dma_heap_alloc_data_ { uint64_t len; uint32_t fd; uint32_t fd_flags; uint64_t heap_flags; };
#define DMA_HEAP_IOCTL_ALLOC_ _IOWR('H', 0x0, struct dma_heap_alloc_data_)
static int dmaheap_alloc(uint64_t size) {
    static int heap = -2;
    if (heap == -2) {
        const char *path = getenv("MALI_COMPAT_DMABUF_HEAP") ? getenv("MALI_COMPAT_DMABUF_HEAP") : "/dev/dma_heap/system";
        heap = open(path, O_RDONLY | O_CLOEXEC);
        if (heap < 0) fprintf(stderr, "mali-compat: dma-buf export: cannot open %s: %s\n", path, strerror(errno));
    }
    if (heap < 0) return -1;
    struct dma_heap_alloc_data_ a = {(size + 4095) & ~4095ull, 0, O_RDWR | O_CLOEXEC, 0};
    if (ioctl(heap, DMA_HEAP_IOCTL_ALLOC_, &a) < 0) {
        fprintf(stderr, "mali-compat: dma-buf export: heap allocation of %llu bytes failed: %s\n", (unsigned long long)size, strerror(errno));
        return -1;
    }
    return (int)a.fd;
}

static uint32_t format_bytes(VkFormat f) {
    switch (f) {
    case VK_FORMAT_R8_UNORM: return 1;
    case VK_FORMAT_R8G8_UNORM: return 2;
    case VK_FORMAT_B8G8R8A8_UNORM: case VK_FORMAT_B8G8R8A8_SRGB: case VK_FORMAT_R8G8B8A8_UNORM: case VK_FORMAT_R8G8B8A8_SRGB:
    case VK_FORMAT_A2B10G10R10_UNORM_PACK32: case VK_FORMAT_A2R10G10B10_UNORM_PACK32: return 4;
    case VK_FORMAT_R16G16B16A16_SFLOAT: return 8;
    default: return 0;
    }
}

// MALI_COMPAT_DMABUF_PEEK=1 (diagnostic): every 5 s, one line per exported buffer with how much of it is non-zero
// and a few pixels, read through a CPU mapping of the dma-buf (with DMA_BUF_IOCTL_SYNC). Tells whether gamescope's
// output images hold a picture before DroidDeck's compositor gets them.
struct dma_buf_sync_ { uint64_t flags; };
#define DMA_BUF_IOCTL_SYNC_ _IOW('b', 0, struct dma_buf_sync_)
static void *dmabuf_peek_thread(void *arg) {
    (void)arg;
    for (;;) {
        sleep(5);
        pthread_mutex_lock(&g_export_mutex);
        int n = 0;
        for (struct ExportMem *x = g_export_mem; x && n < 12; x = x->next, n++) {
            void *p = mmap(NULL, x->size, PROT_READ, MAP_SHARED, x->fd, 0);
            if (p == MAP_FAILED) { fprintf(stderr, "mali-compat: dma-buf peek: mmap failed: %s\n", strerror(errno)); continue; }
            struct dma_buf_sync_ sy = {1 /* START */ | 1u << 0 /* READ */};
            ioctl(x->fd, DMA_BUF_IOCTL_SYNC_, &sy);
            const uint32_t *w = p;
            uint64_t words = x->size / 4, nz = 0;
            for (uint64_t i = 0; i < words; i += 61) nz += w[i] != 0;
            static char comm[32];
            if (!comm[0]) {
                FILE *cf = fopen("/proc/self/comm", "r");
                if (!cf || !fgets(comm, sizeof comm, cf)) snprintf(comm, sizeof comm, "?");
                if (cf) fclose(cf);
                comm[strcspn(comm, "\n")] = 0;
            }
            fprintf(stderr, "mali-compat: dma-buf peek [%s %d]: %s mem %p %llu KiB, non-zero %llu of %llu samples, px[0] %08x px[mid] %08x px[3/4] %08x\n",
                    comm, (int)getpid(), x->imported ? "imported" : "exported", (void *)x->mem, (unsigned long long)(x->size >> 10), (unsigned long long)nz,
                    (unsigned long long)(words / 61 + 1), w[0], w[words / 2], w[words * 3 / 4]);
            sy.flags = 2 /* END */ | 1;
            ioctl(x->fd, DMA_BUF_IOCTL_SYNC_, &sy);
            munmap(p, x->size);
        }
        pthread_mutex_unlock(&g_export_mutex);
    }
    return NULL;
}

static bool dmabuf_handle(const void *chain) {
    const VkPhysicalDeviceExternalImageFormatInfo *e = find_struct(chain, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_IMAGE_FORMAT_INFO);
    return e && e->handleType == VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT;
}

static VKAPI_ATTR VkResult VKAPI_CALL compat_GetPhysicalDeviceImageFormatProperties2(
    VkPhysicalDevice physicalDevice, const VkPhysicalDeviceImageFormatInfo2 *pImageFormatInfo,
    VkImageFormatProperties2 *pImageFormatProperties) {
    struct InstanceData *inst = get_instance_by_phys_dev(physicalDevice);
    if (!inst || !inst->fpGetPhysicalDeviceImageFormatProperties2) return VK_ERROR_INITIALIZATION_FAILED;
    if (is_yuv_nv12(pImageFormatInfo->format)) {
        VkResult r = inst->fpGetPhysicalDeviceImageFormatProperties2(physicalDevice, pImageFormatInfo, pImageFormatProperties);
        if (r != VK_SUCCESS && pImageFormatProperties) {
            pImageFormatProperties->imageFormatProperties.maxExtent = (VkExtent3D){4096, 4096, 1};
            pImageFormatProperties->imageFormatProperties.maxMipLevels = 1;
            pImageFormatProperties->imageFormatProperties.maxArrayLayers = 1;
            pImageFormatProperties->imageFormatProperties.sampleCounts = VK_SAMPLE_COUNT_1_BIT;
            pImageFormatProperties->imageFormatProperties.maxResourceSize = 4096 * 4096 * 2;
            return VK_SUCCESS;
        }
        return r;
    }
    if (panvk_format_is_bc(pImageFormatInfo->format)) {
        VkPhysicalDeviceImageFormatInfo2 mod = *pImageFormatInfo;
        mod.format = panvk_bc_decoded_format(pImageFormatInfo->format);
        mod.flags &= ~VK_IMAGE_CREATE_BLOCK_TEXEL_VIEW_COMPATIBLE_BIT;
        return inst->fpGetPhysicalDeviceImageFormatProperties2(physicalDevice, &mod, pImageFormatProperties);
    }
    VkResult r = inst->fpGetPhysicalDeviceImageFormatProperties2(physicalDevice, pImageFormatInfo, pImageFormatProperties);
    if (r == VK_SUCCESS && g_dmabuf_export && dmabuf_handle(pImageFormatInfo->pNext)) {
        const VkPhysicalDeviceImageDrmFormatModifierInfoEXT *mi =
            find_struct(pImageFormatInfo->pNext, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_IMAGE_DRM_FORMAT_MODIFIER_INFO_EXT);
        bool linear = pImageFormatInfo->tiling == VK_IMAGE_TILING_LINEAR ||
                      (pImageFormatInfo->tiling == VK_IMAGE_TILING_DRM_FORMAT_MODIFIER_EXT && mi && mi->drmFormatModifier == 0);
        VkExternalImageFormatProperties *ep = (VkExternalImageFormatProperties *)find_struct(
            pImageFormatProperties->pNext, VK_STRUCTURE_TYPE_EXTERNAL_IMAGE_FORMAT_PROPERTIES);
        if (linear && ep && format_bytes(pImageFormatInfo->format)) {
            ep->externalMemoryProperties.externalMemoryFeatures |= VK_EXTERNAL_MEMORY_FEATURE_EXPORTABLE_BIT |
                                                                   VK_EXTERNAL_MEMORY_FEATURE_IMPORTABLE_BIT;
            ep->externalMemoryProperties.compatibleHandleTypes |= VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT;
            ep->externalMemoryProperties.exportFromImportedHandleTypes |= VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT;
        }
    }
    return r;
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
            atomic_fetch_add(&g_budget_queries, 1);
            for (uint32_t i = 0; i < VK_MAX_MEMORY_HEAPS; i++) {
                if (i < pMemoryProperties->memoryProperties.memoryHeapCount) {
                    VkDeviceSize heap_size = pMemoryProperties->memoryProperties.memoryHeaps[i].size;
                    if (g_budget_accounted) { // MALI_COMPAT_BUDGET_MB: per-heap live usage, that cap
                        budget->heapBudget[i] = heap_size > g_budget_cap ? g_budget_cap : heap_size;
                        budget->heapUsage[i] = atomic_load(&g_heap_usage[i]);
                    } else { // old report, unchanged
                        budget->heapBudget[i] = heap_size > (6ULL << 30) ? (6ULL << 30) : heap_size;
                        budget->heapUsage[i] = allocated;
                    }
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

    VkExtensionProperties *all = (VkExtensionProperties *)malloc((count + 12) * sizeof(VkExtensionProperties));
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
    bool has_drm = false;
    for (uint32_t i = 0; i < count; i++) has_drm |= !strcmp(all[i].extensionName, VK_EXT_PHYSICAL_DEVICE_DRM_EXTENSION_NAME);
    if (g_drm_on && !has_drm) {
        strncpy(all[count].extensionName, VK_EXT_PHYSICAL_DEVICE_DRM_EXTENSION_NAME, VK_MAX_EXTENSION_NAME_SIZE);
        all[count].specVersion = 1;
        count++;
    }
    if (g_rt_pipeline && rt_pipeline_emulation_available()) {
        static const struct { const char *name; uint32_t ver; } rt_exts[] = {
            { "VK_KHR_ray_tracing_pipeline",          1 },
            { "VK_KHR_pipeline_library",              1 },
            { "VK_KHR_ray_tracing_maintenance1",      1 },
            { "VK_EXT_pipeline_library_group_handles", 1 },
            { "VK_KHR_deferred_host_operations",      4 },
        };
        for (size_t e = 0; e < sizeof(rt_exts) / sizeof(rt_exts[0]); e++) {
            bool found = false;
            for (uint32_t i = 0; i < count; i++) {
                if (!strcmp(all[i].extensionName, rt_exts[e].name)) { found = true; break; }
            }
            if (!found) {
                strncpy(all[count].extensionName, rt_exts[e].name, VK_MAX_EXTENSION_NAME_SIZE);
                all[count].specVersion = rt_exts[e].ver;
                count++;
            }
        }
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


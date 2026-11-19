// SPDX-License-Identifier: GPL-3.0-or-later
// cl_procaddr.h: PASSTHRU, vkGet*ProcAddr dispatch, loader negotiate, exported RT entry points. Included by mali_compat_layer.c.
// -----------------------------------------------------------------------------
// Loader Layer Negotiate & ProcAddr Dispatch
// -----------------------------------------------------------------------------
// MALI_COMPAT_PASSTHRU=1: A/B switch, "no layer but the CreateDevice feature masking". Every entry point except
// vkCreateInstance/DestroyInstance, vkCreateDevice/DestroyDevice and the two GetProcAddr goes straight to the next layer /
// driver: no feature/format/property spoofing, no SPIR-V patching, no object hooks. The app then sees the driver's real
// features and the layer only drops the enabled ones the driver lacks (Quake II RTX: pipelineStatisticsQuery; without any
// layer its vkCreateDevice fails with VK_ERROR_FEATURE_NOT_PRESENT). Tells whether a black/faulting picture comes from the
// layer's hooks or from below. [keeper, Claude Opus 5.5] 2026-11-06
static bool passthru_on(void) {
    static int v = -1;
    if (v < 0) {
        const char *e = getenv("MALI_COMPAT_PASSTHRU");
        v = e && e[0] == '1';
        if (v) fprintf(stderr, "mali-compat: PASSTHRU=1: only vkCreateDevice feature masking is active\n");
    }
    return v;
}

EXPORT VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL vkGetDeviceProcAddr(VkDevice device, const char *pName) {
    if (!pName) return NULL;
    // Must be this layer's own: the Arm WSI layer above resolves every entry point it does not know through the
    // "vkGetDeviceProcAddr" it gets here. Forwarding the name handed it the driver's, so vkCreateShaderModule and
    // vkCreateGraphicsPipelines skipped this layer (no SPIR-V patching, vendor compiler crash). [Claude Opus 5.5]
    if (!strcmp(pName, "vkGetDeviceProcAddr")) return (PFN_vkVoidFunction)vkGetDeviceProcAddr;
    if (passthru_on()) {
        if (!strcmp(pName, "vkDestroyDevice")) return (PFN_vkVoidFunction)compat_DestroyDevice;
        struct DeviceData *pd = get_device_data(device);
        return pd ? pd->get_device_proc_addr(device, pName) : NULL;
    }
    if (g_sd_armed && !tl_sd_inner) { PFN_vkVoidFunction sw = sd_proc(device, pName); if (sw) return sw; } // sync-diag
    if (g_sal_on && !tl_sal_inner) { PFN_vkVoidFunction sw = sal_proc(device, pName); if (sw) return sw; } // SSBO_ALIGN_LOG
    if (g_dpd_on && !tl_dpd_inner) { PFN_vkVoidFunction dw = dpd_proc(device, pName); if (dw) return dw; } // DPOOL_DIAG
    // GPU timing: which vkCmdDraw*/Dispatch* entry points the app asks for (once each), to see what the hooks miss.
    if (g_gpu_time && (!strncmp(pName, "vkCmdDraw", 9) || !strncmp(pName, "vkCmdDispatch", 13) ||
                       !strncmp(pName, "vkCmdExecute", 12) || !strncmp(pName, "vkCmdBeginRender", 16))) {
        static char seen[64][48];
        static int nseen;
        static pthread_mutex_t m = PTHREAD_MUTEX_INITIALIZER;
        pthread_mutex_lock(&m);
        int k = 0;
        while (k < nseen && strcmp(seen[k], pName)) k++;
        if (k == nseen && nseen < 64) {
            snprintf(seen[nseen++], sizeof seen[0], "%s", pName);
            fprintf(stderr, "mali-compat: GPU timing: app asked for %s\n", pName);
        }
        pthread_mutex_unlock(&m);
    }
    // Only expose the map2 entry points when they exist below: 64-bit Wine does not enable VK_KHR_map_memory2, and
    // handing it our function then made it call through a NULL driver pointer (The Witcher 3: crash in vkMapMemory).
    // [Claude Opus 5.5]
    if (!strcmp(pName, "vkGetDeviceImageMemoryRequirements") || !strcmp(pName, "vkGetDeviceImageMemoryRequirementsKHR")) {
        struct DeviceData *dd = get_device_data(device);
        if (dd && dd->fpGetDeviceImageMemoryRequirements) return (PFN_vkVoidFunction)compat_GetDeviceImageMemoryRequirements;
        return dd ? dd->get_device_proc_addr(device, pName) : NULL;
    }
    if (!strcmp(pName, "vkGetDeviceImageSparseMemoryRequirements") || !strcmp(pName, "vkGetDeviceImageSparseMemoryRequirementsKHR")) {
        struct DeviceData *dd = get_device_data(device);
        if (dd && dd->fpGetDeviceImageSparseMemoryRequirements) return (PFN_vkVoidFunction)compat_GetDeviceImageSparseMemoryRequirements;
        return dd ? dd->get_device_proc_addr(device, pName) : NULL;
    }
    if (!strcmp(pName, "vkCmdBindDescriptorBuffersEXT")) {
        struct DeviceData *dd = get_device_data(device);
        if (dd && dd->fpCmdBindDescriptorBuffersEXT) return (PFN_vkVoidFunction)compat_CmdBindDescriptorBuffersEXT;
        return dd ? dd->get_device_proc_addr(device, pName) : NULL;
    }
    if (!strcmp(pName, "vkCmdSetVertexInputEXT")) {
        struct DeviceData *dd = get_device_data(device);
        if (dd && dd->fpCmdSetVertexInputEXT) return (PFN_vkVoidFunction)compat_CmdSetVertexInputEXT;
        return dd ? dd->get_device_proc_addr(device, pName) : NULL;
    }
    if (!strcmp(pName, "vkMapMemory2KHR") || !strcmp(pName, "vkUnmapMemory2KHR") || !strcmp(pName, "vkMapMemory2") ||
        !strcmp(pName, "vkUnmapMemory2")) {
        struct DeviceData *dd = get_device_data(device);
        if (!dd || !dd->fpMapMemory2KHR) return dd ? dd->get_device_proc_addr(device, pName) : NULL;
    }
    // round 4 (Codex r3 #5): with RT_PIPELINE the AS-buffer map must see every destroy, also in desc-capture mode
    // (compat_RtDestroyBuffer chains to desc-capture's destroy itself), so this comes before dc_proc.
    if (g_addrmap) { PFN_vkVoidFunction ah = am_proc(device, pName); if (ah) return ah; } // ADDRMAP (chains to the two below)
    if (g_as_hooks && !strcmp(pName, "vkDestroyBuffer")) return (PFN_vkVoidFunction)compat_RtDestroyBuffer;
    if (g_dc_on) { PFN_vkVoidFunction dh = dc_proc(device, pName); if (dh) return dh; } // desc-capture-1
    #define MATCH_DEV(fn) if (!strcmp(pName, "vk" #fn)) return (PFN_vkVoidFunction)compat_##fn
    if (g_memlog) { // MALI_COMPAT_MEMLOG only: vkMapMemory for the M events, queue submits for the S events
        MATCH_DEV(MapMemory);
        struct DeviceData *md = get_device_data(device);
        if (md && md->ml_fpQueueSubmit && !strcmp(pName, "vkQueueSubmit")) return (PFN_vkVoidFunction)compat_MlQueueSubmit;
        if (md && md->ml_fpQueueSubmit2 && (!strcmp(pName, "vkQueueSubmit2") || !strcmp(pName, "vkQueueSubmit2KHR")))
            return (PFN_vkVoidFunction)compat_MlQueueSubmit2;
    }
    // GPU timing hooks: only while the switch is on, so the normal path calls the driver directly.
    #define MATCH_DEV_KHR(fn) if (!strcmp(pName, "vk" #fn) || !strcmp(pName, "vk" #fn "KHR")) return (PFN_vkVoidFunction)compat_##fn
    if (g_async_ms) MATCH_DEV(DestroyPipeline);
    if (g_depth_equal_on && (!strcmp(pName, "vkCmdSetDepthCompareOp") || !strcmp(pName, "vkCmdSetDepthCompareOpEXT"))) {
        struct DeviceData *dd = get_device_data(device);
        if (dd && dd->fpCmdSetDepthCompareOp) return (PFN_vkVoidFunction)compat_CmdSetDepthCompareOp;
    }
    if (g_ct) { // cover-trim: the app's scissor and conditional rendering, only when its device state exists
        struct DeviceData *cd = get_device_data(device);
        if (cd && cd->ct) {
            if (!strcmp(pName, "vkCmdSetScissor")) return (PFN_vkVoidFunction)compat_CmdSetScissor;
            if ((!strcmp(pName, "vkCmdSetScissorWithCount") || !strcmp(pName, "vkCmdSetScissorWithCountEXT")) && cd->ct->SetScissorWithCount)
                return (PFN_vkVoidFunction)compat_CmdSetScissorWithCount;
            if (!strcmp(pName, "vkCmdBeginConditionalRenderingEXT") && cd->ct->BeginCond) return (PFN_vkVoidFunction)compat_CmdBeginConditionalRenderingEXT;
            if (!strcmp(pName, "vkCmdEndConditionalRenderingEXT") && cd->ct->EndCond) return (PFN_vkVoidFunction)compat_CmdEndConditionalRenderingEXT;
        }
    }
    if (g_gpu_time || g_async_ms || g_noatt_w || g_no_susres || g_dc_on || g_ct) { // GPU timing, async compiles, pass clamp, NO_SUSPEND_RESUME, desc-capture-1 and cover-trim share these hooks
        if (g_gpu_time) { // diag-1113: on a device below 1.2/1.3 the driver's answer, not a hook over its NULL
            struct DeviceData *nd = get_device_data(device);
            if (nd && nd->gt && gt_hook_missing(nd->gt, pName)) return nd->get_device_proc_addr(device, pName);
        }
        MATCH_DEV_KHR(CmdBeginRendering);
        MATCH_DEV_KHR(CmdEndRendering);
        MATCH_DEV(CmdBeginRenderPass);
        MATCH_DEV(CmdEndRenderPass);
        MATCH_DEV_KHR(CmdBeginRenderPass2);
        MATCH_DEV_KHR(CmdEndRenderPass2);
        MATCH_DEV(CmdDispatch);
        MATCH_DEV(CmdDispatchIndirect);
        MATCH_DEV(CmdDraw);
        MATCH_DEV(CmdDrawIndexed);
        MATCH_DEV(CmdDrawIndirect);
        MATCH_DEV(CmdDrawIndexedIndirect);
        MATCH_DEV_KHR(CmdDrawIndirectCount);
        MATCH_DEV_KHR(CmdDrawIndexedIndirectCount);
        struct DeviceData *gd = get_device_data(device);
        if (gd && gd->gt) {
            #define MATCH_GT_OPT(fn) if (!strcmp(pName, "vk" #fn) && gd->gt->fp##fn) return (PFN_vkVoidFunction)compat_##fn
            MATCH_GT_OPT(CmdDrawMultiEXT);
            MATCH_GT_OPT(CmdDrawMultiIndexedEXT);
            MATCH_GT_OPT(CmdDrawMeshTasksEXT);
            MATCH_GT_OPT(CmdDrawMeshTasksIndirectEXT);
            MATCH_GT_OPT(CmdDrawMeshTasksIndirectCountEXT);
            if ((!strcmp(pName, "vkCmdDispatchBase") || !strcmp(pName, "vkCmdDispatchBaseKHR")) && gd->gt->fpCmdDispatchBase)
                return (PFN_vkVoidFunction)compat_CmdDispatchBase;
            // GPU timing only: secondary command buffers into their pass, queue-side CPU time
            #define MATCH_GT_NAME(fn, name) if ((g_gpu_time || g_dc_on) && !strcmp(pName, name) && gd->gt->fp##fn) return (PFN_vkVoidFunction)compat_##fn
            if (g_ct && !strcmp(pName, "vkCmdExecuteCommands") && gd->gt->fpCmdExecuteCommands) return (PFN_vkVoidFunction)compat_CmdExecuteCommands;
            MATCH_GT_NAME(CmdExecuteCommands, "vkCmdExecuteCommands");
            MATCH_GT_NAME(QueueSubmit, "vkQueueSubmit");
            MATCH_GT_NAME(QueueSubmit2, "vkQueueSubmit2");
            MATCH_GT_NAME(QueueSubmit2, "vkQueueSubmit2KHR");
            MATCH_GT_NAME(WaitForFences, "vkWaitForFences");
            MATCH_GT_NAME(WaitSemaphores, "vkWaitSemaphores");
            MATCH_GT_NAME(WaitSemaphores, "vkWaitSemaphoresKHR");
            MATCH_GT_NAME(QueueWaitIdle, "vkQueueWaitIdle");
            MATCH_GT_NAME(DeviceWaitIdle, "vkDeviceWaitIdle");
            #undef MATCH_GT_NAME
            #undef MATCH_GT_OPT
        }
    }
    if (!strcmp(pName, "vkCmdPushDescriptorSetKHR") || !strcmp(pName, "vkCmdPushDescriptorSet")) {
        struct DeviceData *dd = get_device_data(device);
        if (dd && dd->fpCmdPushDescriptorSetKHR && (g_gpu_time || g_vendor_id || g_dc_on || g_rt_pipeline)) return (PFN_vkVoidFunction)compat_CmdPushDescriptorSetKHR;
    }
    if (g_rt_pipeline) {
        rt_patch_fex_thunks();
        if (!strcmp(pName, "vkCreateRayTracingPipelinesKHR") || !strcmp(pName, "vkCreateRayTracingPipelinesNV")) return (PFN_vkVoidFunction)compat_CreateRayTracingPipelinesKHR;
        if (!strcmp(pName, "vkGetRayTracingShaderGroupHandlesKHR") || !strcmp(pName, "vkGetRayTracingShaderGroupHandlesNV")) return (PFN_vkVoidFunction)compat_GetRayTracingShaderGroupHandlesKHR;
        if (!strcmp(pName, "vkGetRayTracingCaptureReplayShaderGroupHandlesKHR")) return (PFN_vkVoidFunction)compat_GetRayTracingCaptureReplayShaderGroupHandlesKHR;
        if (!strcmp(pName, "vkGetRayTracingShaderGroupStackSizeKHR")) return (PFN_vkVoidFunction)compat_GetRayTracingShaderGroupStackSizeKHR;
        if (!strcmp(pName, "vkCmdSetRayTracingPipelineStackSizeKHR")) return (PFN_vkVoidFunction)compat_CmdSetRayTracingPipelineStackSizeKHR;
        if (!strcmp(pName, "vkCmdTraceRaysKHR")) return (PFN_vkVoidFunction)compat_CmdTraceRaysKHR;
        if (!strcmp(pName, "vkCmdTraceRaysIndirectKHR")) return (PFN_vkVoidFunction)compat_CmdTraceRaysIndirectKHR;
        if (!strcmp(pName, "vkCmdTraceRaysIndirect2KHR")) return (PFN_vkVoidFunction)compat_CmdTraceRaysIndirect2KHR;
        if (!strcmp(pName, "vkCreateDeferredHostOperationsKHR")) return (PFN_vkVoidFunction)compat_CreateDeferredHostOperationsKHR;
        if (!strcmp(pName, "vkDestroyDeferredHostOperationKHR")) return (PFN_vkVoidFunction)compat_DestroyDeferredHostOperationKHR;
        if (!strcmp(pName, "vkGetDeferredHostOperationMaxConcurrencyKHR")) return (PFN_vkVoidFunction)compat_GetDeferredHostOperationMaxConcurrencyKHR;
        if (!strcmp(pName, "vkGetDeferredHostOperationResultKHR")) return (PFN_vkVoidFunction)compat_GetDeferredHostOperationResultKHR;
        if (!strcmp(pName, "vkDeferredHostOperationJoinKHR")) return (PFN_vkVoidFunction)compat_DeferredHostOperationJoinKHR;
        if (!strcmp(pName, "vkDestroyShaderModule")) return (PFN_vkVoidFunction)compat_DestroyShaderModule;
        if (!strcmp(pName, "vkDestroyPipeline")) return (PFN_vkVoidFunction)compat_DestroyPipeline;
        if (!strcmp(pName, "vkCreateDescriptorSetLayout")) return (PFN_vkVoidFunction)compat_RtCreateDescriptorSetLayout;
        if (!strcmp(pName, "vkCreatePipelineLayout")) return (PFN_vkVoidFunction)compat_RtCreatePipelineLayout;
        if (!strcmp(pName, "vkDestroyPipelineLayout")) return (PFN_vkVoidFunction)compat_RtDestroyPipelineLayout;
        if (!strcmp(pName, "vkCmdBindDescriptorSets")) return (PFN_vkVoidFunction)compat_RtCmdBindDescriptorSets;
        if (!strcmp(pName, "vkCmdSetDescriptorBufferOffsetsEXT")) {
            struct DeviceData *dd = get_device_data(device);
            if (dd && dd->fpCmdSetDescriptorBufferOffsetsEXT) return (PFN_vkVoidFunction)compat_RtCmdSetDescriptorBufferOffsetsEXT;
        }
    }
    if (g_as_hooks) {   /* rt-gate-1112: the AS hooks, also with MALI_COMPAT_AS_FIX=1 alone */
        if (!strcmp(pName, "vkDestroyBuffer")) return (PFN_vkVoidFunction)compat_RtDestroyBuffer;   /* lead-1110c */
        if (!strcmp(pName, "vkCreateAccelerationStructureKHR")) return (PFN_vkVoidFunction)compat_CreateAccelerationStructureKHR;
        if (!strcmp(pName, "vkDestroyAccelerationStructureKHR")) return (PFN_vkVoidFunction)compat_DestroyAccelerationStructureKHR;
        if (!strcmp(pName, "vkCmdBuildAccelerationStructuresKHR")) return (PFN_vkVoidFunction)compat_CmdBuildAccelerationStructuresKHR;
        if (!strcmp(pName, "vkCmdCopyAccelerationStructureKHR")) return (PFN_vkVoidFunction)compat_CmdCopyAccelerationStructureKHR;
        if (!strcmp(pName, "vkCmdWriteAccelerationStructuresPropertiesKHR")) return (PFN_vkVoidFunction)compat_CmdWriteAccelerationStructuresPropertiesKHR;
        if (!strcmp(pName, "vkGetAccelerationStructureDeviceAddressKHR")) return (PFN_vkVoidFunction)compat_GetAccelerationStructureDeviceAddressKHR;
        if (!strcmp(pName, "vkGetAccelerationStructureBuildSizesKHR")) return (PFN_vkVoidFunction)compat_GetAccelerationStructureBuildSizesKHR;
    }
    if (g_gpu_time || g_dump_hash_count || g_pcache_on || g_dump_inflight || g_dc_on || g_wls_clamp || g_wls_fix || g_ct || g_fail_log || g_db_hash) MATCH_DEV(CreateComputePipelines);
    if (g_fake_occlusion) {
        MATCH_DEV(CreateQueryPool);
        MATCH_DEV(DestroyQueryPool);
        MATCH_DEV(CmdCopyQueryPoolResults);
        MATCH_DEV(GetQueryPoolResults);
    }
    if (g_addrmap) { // N markers; the occlusion table is touched only with FAKE_OCCLUSION
        MATCH_DEV(CreateQueryPool);
        MATCH_DEV(DestroyQueryPool);
    }
    #undef MATCH_DEV_KHR
    MATCH_DEV(DestroyDevice);
    MATCH_DEV(CreateShaderModule);
    if (g_desc_pool_scale > 1 || g_desc_pool_pad || g_fail_lvl || g_addrmap) MATCH_DEV(CreateDescriptorPool); // unscaled: as given
    if (g_fail_lvl || g_dsl_stage_retry) { // after dc_proc: desc-capture keeps these entries when it is on
        struct DeviceData *fd = get_device_data(device);
        if (fd && fd->fpCreateDescriptorSetLayout && !strcmp(pName, "vkCreateDescriptorSetLayout")) return (PFN_vkVoidFunction)compat_FlCreateDescriptorSetLayout;
        if (fd && fd->fpAllocateDescriptorSets && !strcmp(pName, "vkAllocateDescriptorSets")) return (PFN_vkVoidFunction)compat_FlAllocateDescriptorSets;
        if (fd && fd->fpCreatePipelineLayout && !strcmp(pName, "vkCreatePipelineLayout")) return (PFN_vkVoidFunction)compat_FlCreatePipelineLayout;
    }
    MATCH_DEV(CreateGraphicsPipelines);
    MATCH_DEV(CmdSetPolygonModeEXT);
    MATCH_DEV(CreateImage);
    MATCH_DEV(DestroyImage);
    MATCH_DEV(CreateImageView);
    MATCH_DEV(CreateBuffer);
    MATCH_DEV(CreateBufferView);
    MATCH_DEV(AllocateMemory);
    MATCH_DEV(FreeMemory);
    if (g_dmabuf_export) { MATCH_DEV(GetMemoryFdKHR); }
    MATCH_DEV(UnmapMemory);
    MATCH_DEV(MapMemory2KHR);
    MATCH_DEV(UnmapMemory2KHR);
    if (!strcmp(pName, "vkMapMemory2")) return (PFN_vkVoidFunction)compat_MapMemory2KHR;
    if (!strcmp(pName, "vkUnmapMemory2")) return (PFN_vkVoidFunction)compat_UnmapMemory2KHR;
    MATCH_DEV(FlushMappedMemoryRanges);
    MATCH_DEV(InvalidateMappedMemoryRanges);
    MATCH_DEV(CmdCopyBufferToImage);
    MATCH_DEV(CmdCopyBufferToImage2);
    if (!strcmp(pName, "vkCmdCopyBufferToImage2KHR")) return (PFN_vkVoidFunction)compat_CmdCopyBufferToImage2;
    MATCH_DEV(CmdBindPipeline);
    MATCH_DEV(CmdPushConstants);
    // pc2-restore-1: only names the driver has (a hook for a missing one would call through NULL)
    if (!strcmp(pName, "vkCmdPushConstants2") || !strcmp(pName, "vkCmdPushConstants2KHR")) {
        struct DeviceData *dd = get_device_data(device);
        bool khr = pName[19] == 'K';
        if (dd && (khr ? dd->fpCmdPushConstants2KHR != NULL : dd->fpCmdPushConstants2 != NULL))
            return khr ? (PFN_vkVoidFunction)compat_CmdPushConstants2KHR : (PFN_vkVoidFunction)compat_CmdPushConstants2;
        return dd ? dd->get_device_proc_addr(device, pName) : NULL;
    }
    MATCH_DEV(ResetCommandBuffer);
    MATCH_DEV(BeginCommandBuffer);
    MATCH_DEV(AllocateCommandBuffers);
    MATCH_DEV(FreeCommandBuffers);
    MATCH_DEV(ResetCommandPool);
    MATCH_DEV(DestroyCommandPool);
    #undef MATCH_DEV

    struct DeviceData *d = get_device_data(device);
    return d ? d->get_device_proc_addr(device, pName) : NULL;
}

EXPORT VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL vkGetInstanceProcAddr(VkInstance instance, const char *pName) {
    if (!pName) return NULL;
    if (!strcmp(pName, "vkCreateInstance")) return (PFN_vkVoidFunction)compat_CreateInstance;
    if (!strcmp(pName, "vkCreateDevice")) return (PFN_vkVoidFunction)compat_CreateDevice;
    if (!strcmp(pName, "vkGetInstanceProcAddr")) return (PFN_vkVoidFunction)vkGetInstanceProcAddr;
    if (!strcmp(pName, "vkGetDeviceProcAddr")) return (PFN_vkVoidFunction)vkGetDeviceProcAddr;
    if (passthru_on()) {
        if (!strcmp(pName, "vkDestroyInstance")) return (PFN_vkVoidFunction)compat_DestroyInstance;
        if (!strcmp(pName, "vkDestroyDevice")) return (PFN_vkVoidFunction)compat_DestroyDevice;
        struct InstanceData *pi = get_instance_data(instance);
        return pi ? pi->get_instance_proc_addr(instance, pName) : NULL;
    }

    #define MATCH_INST(fn) if (!strcmp(pName, "vk" #fn)) return (PFN_vkVoidFunction)compat_##fn
    MATCH_INST(DestroyInstance);
    MATCH_INST(GetPhysicalDeviceFeatures);
    MATCH_INST(GetPhysicalDeviceFeatures2);
    if (!strcmp(pName, "vkGetPhysicalDeviceFeatures2KHR")) return (PFN_vkVoidFunction)compat_GetPhysicalDeviceFeatures2;
    MATCH_INST(GetPhysicalDeviceFormatProperties);
    MATCH_INST(GetPhysicalDeviceFormatProperties2);
    if (!strcmp(pName, "vkGetPhysicalDeviceFormatProperties2KHR")) return (PFN_vkVoidFunction)compat_GetPhysicalDeviceFormatProperties2;
    MATCH_INST(GetPhysicalDeviceImageFormatProperties);
    MATCH_INST(GetPhysicalDeviceImageFormatProperties2);
    if (!strcmp(pName, "vkGetPhysicalDeviceImageFormatProperties2KHR")) return (PFN_vkVoidFunction)compat_GetPhysicalDeviceImageFormatProperties2;
    MATCH_INST(GetPhysicalDeviceMemoryProperties2);
    MATCH_INST(GetPhysicalDeviceProperties);
    MATCH_INST(GetPhysicalDeviceProperties2);
    if (!strcmp(pName, "vkGetPhysicalDeviceProperties2KHR")) return (PFN_vkVoidFunction)compat_GetPhysicalDeviceProperties2;
    if (!strcmp(pName, "vkGetPhysicalDeviceMemoryProperties2KHR")) return (PFN_vkVoidFunction)compat_GetPhysicalDeviceMemoryProperties2;
    MATCH_INST(EnumerateDeviceExtensionProperties);
    #undef MATCH_INST

    // Also check device proc addr matches for device functions queried via GetInstanceProcAddr
    #define MATCH_DEV2(fn) if (!strcmp(pName, "vk" #fn)) return (PFN_vkVoidFunction)compat_##fn
    MATCH_DEV2(DestroyDevice);
    MATCH_DEV2(CreateShaderModule);
    MATCH_DEV2(CreateGraphicsPipelines);
    MATCH_DEV2(CmdSetPolygonModeEXT);
    MATCH_DEV2(CreateImage);
    MATCH_DEV2(DestroyImage);
    MATCH_DEV2(CreateImageView);
    MATCH_DEV2(CreateBuffer);
    MATCH_DEV2(CreateBufferView);
    MATCH_DEV2(AllocateMemory);
    MATCH_DEV2(FreeMemory);
    MATCH_DEV2(UnmapMemory);
    if (g_memlog) MATCH_DEV2(MapMemory);
    MATCH_DEV2(MapMemory2KHR);
    MATCH_DEV2(UnmapMemory2KHR);
    if (!strcmp(pName, "vkMapMemory2")) return (PFN_vkVoidFunction)compat_MapMemory2KHR;
    if (!strcmp(pName, "vkUnmapMemory2")) return (PFN_vkVoidFunction)compat_UnmapMemory2KHR;
    MATCH_DEV2(FlushMappedMemoryRanges);
    MATCH_DEV2(InvalidateMappedMemoryRanges);
    MATCH_DEV2(CmdCopyBufferToImage);
    MATCH_DEV2(CmdCopyBufferToImage2);
    if (!strcmp(pName, "vkCmdCopyBufferToImage2KHR")) return (PFN_vkVoidFunction)compat_CmdCopyBufferToImage2;
    MATCH_DEV2(CmdBindPipeline);
    MATCH_DEV2(CmdPushConstants);
    MATCH_DEV2(ResetCommandBuffer);
    MATCH_DEV2(BeginCommandBuffer);
    MATCH_DEV2(FreeCommandBuffers);
    if (g_rt_pipeline) {
        rt_patch_fex_thunks();
        MATCH_DEV2(CreateRayTracingPipelinesKHR);
        MATCH_DEV2(GetRayTracingShaderGroupHandlesKHR);
        if (!strcmp(pName, "vkCreateRayTracingPipelinesNV")) return (PFN_vkVoidFunction)compat_CreateRayTracingPipelinesKHR;
        if (!strcmp(pName, "vkGetRayTracingShaderGroupHandlesNV")) return (PFN_vkVoidFunction)compat_GetRayTracingShaderGroupHandlesKHR;
        MATCH_DEV2(GetRayTracingCaptureReplayShaderGroupHandlesKHR);
        MATCH_DEV2(GetRayTracingShaderGroupStackSizeKHR);
        MATCH_DEV2(CmdSetRayTracingPipelineStackSizeKHR);
        MATCH_DEV2(CmdTraceRaysKHR);
        MATCH_DEV2(CmdTraceRaysIndirectKHR);
        MATCH_DEV2(CmdTraceRaysIndirect2KHR);
        MATCH_DEV2(CreateDeferredHostOperationsKHR);
        MATCH_DEV2(DestroyDeferredHostOperationKHR);
        MATCH_DEV2(GetDeferredHostOperationMaxConcurrencyKHR);
        MATCH_DEV2(GetDeferredHostOperationResultKHR);
        MATCH_DEV2(DeferredHostOperationJoinKHR);
        MATCH_DEV2(DestroyShaderModule);
        MATCH_DEV2(DestroyPipeline);
        if (!strcmp(pName, "vkCreateDescriptorSetLayout")) return (PFN_vkVoidFunction)compat_RtCreateDescriptorSetLayout;
        if (!strcmp(pName, "vkCreatePipelineLayout")) return (PFN_vkVoidFunction)compat_RtCreatePipelineLayout;
        if (!strcmp(pName, "vkDestroyPipelineLayout")) return (PFN_vkVoidFunction)compat_RtDestroyPipelineLayout;
        if (!strcmp(pName, "vkCmdBindDescriptorSets")) return (PFN_vkVoidFunction)compat_RtCmdBindDescriptorSets;
        if (!strcmp(pName, "vkCmdSetDescriptorBufferOffsetsEXT")) return (PFN_vkVoidFunction)compat_RtCmdSetDescriptorBufferOffsetsEXT;
    }
    if (g_as_hooks) {   /* rt-gate-1112: the AS hooks, also with MALI_COMPAT_AS_FIX=1 alone */
        if (!strcmp(pName, "vkDestroyBuffer")) return (PFN_vkVoidFunction)compat_RtDestroyBuffer;   /* lead-1110c */
        if (!strcmp(pName, "vkCreateAccelerationStructureKHR")) return (PFN_vkVoidFunction)compat_CreateAccelerationStructureKHR;
        if (!strcmp(pName, "vkDestroyAccelerationStructureKHR")) return (PFN_vkVoidFunction)compat_DestroyAccelerationStructureKHR;
        if (!strcmp(pName, "vkCmdBuildAccelerationStructuresKHR")) return (PFN_vkVoidFunction)compat_CmdBuildAccelerationStructuresKHR;
        if (!strcmp(pName, "vkCmdCopyAccelerationStructureKHR")) return (PFN_vkVoidFunction)compat_CmdCopyAccelerationStructureKHR;
        if (!strcmp(pName, "vkCmdWriteAccelerationStructuresPropertiesKHR")) return (PFN_vkVoidFunction)compat_CmdWriteAccelerationStructuresPropertiesKHR;
        if (!strcmp(pName, "vkGetAccelerationStructureDeviceAddressKHR")) return (PFN_vkVoidFunction)compat_GetAccelerationStructureDeviceAddressKHR;
        if (!strcmp(pName, "vkGetAccelerationStructureBuildSizesKHR")) return (PFN_vkVoidFunction)compat_GetAccelerationStructureBuildSizesKHR;
    }
    #undef MATCH_DEV2

    struct InstanceData *inst = get_instance_data(instance);
    return inst ? inst->get_instance_proc_addr(instance, pName) : NULL;
}

EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkNegotiateLoaderLayerInterfaceVersion(VkNegotiateLayerInterface *pVersionStruct) {
    if (!pVersionStruct || pVersionStruct->sType != LAYER_NEGOTIATE_INTERFACE_STRUCT)
        return VK_ERROR_INITIALIZATION_FAILED;

    if (pVersionStruct->loaderLayerInterfaceVersion >= 2) {
        pVersionStruct->loaderLayerInterfaceVersion = 2;
        pVersionStruct->pfnGetInstanceProcAddr = vkGetInstanceProcAddr;
        pVersionStruct->pfnGetDeviceProcAddr = vkGetDeviceProcAddr;
        pVersionStruct->pfnGetPhysicalDeviceProcAddr = NULL;
        return VK_SUCCESS;
    }
    return VK_ERROR_INITIALIZATION_FAILED;
}

EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkEnumerateInstanceLayerProperties(
    uint32_t *pPropertyCount, VkLayerProperties *pProperties) {
    if (pPropertyCount) *pPropertyCount = 1;
    if (pProperties) {
        strncpy(pProperties->layerName, LAYER_NAME, VK_MAX_EXTENSION_NAME_SIZE);
        strncpy(pProperties->description, "Mali Compatibility Layer for DXVK / Proton", VK_MAX_DESCRIPTION_SIZE);
        pProperties->implementationVersion = 1;
        pProperties->specVersion = VK_API_VERSION_1_3;
    }
    return VK_SUCCESS;
}

EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkEnumerateInstanceExtensionProperties(
    const char *pLayerName, uint32_t *pPropertyCount, VkExtensionProperties *pProperties) {
    (void)pLayerName;
    (void)pProperties;
    if (pPropertyCount) *pPropertyCount = 0;
    return VK_SUCCESS;
}

EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkEnumerateDeviceLayerProperties(
    VkPhysicalDevice physicalDevice, uint32_t *pPropertyCount, VkLayerProperties *pProperties) {
    (void)physicalDevice;
    return vkEnumerateInstanceLayerProperties(pPropertyCount, pProperties);
}

/* ── Exported RT Entry Points (for FEX dlsym global resolution fallback) ──── */
/* rt-gate-1112 [RTX keeper, Claude Opus 5.5] 2026-11-11: these 16 symbols are exported in every build, so the RT/AS
 * emulation must not be reachable through them with RT off. The 9 RT-pipeline exports reach the emulation (and patch
 * the FEX thunks) only with MALI_COMPAT_RT_PIPELINE; otherwise they forward by name to the next layer (the blob has no
 * VK_KHR_ray_tracing_pipeline, so there they end as VK_ERROR_EXTENSION_NOT_PRESENT / 0 / no-op). The 7 AS exports reach
 * the AS hooks only with g_as_hooks (RT_PIPELINE or MALI_COMPAT_AS_FIX), otherwise the next layer's function, which
 * vkCreateDevice loaded for every device - the same route vkGetDeviceProcAddr gives in that mode. */

static PFN_vkVoidFunction rt_export_next(VkDevice device, const char *name) {
    struct DeviceData *d = device ? get_device_data(device) : NULL;
    return d && d->get_device_proc_addr ? d->get_device_proc_addr(d->device, name) : NULL;
}
static PFN_vkVoidFunction rt_export_next_cmd(VkCommandBuffer cmd, const char *name) {
    struct DeviceData *d = cmd ? get_device_data_by_cmd(cmd) : NULL;
    return d && d->get_device_proc_addr ? d->get_device_proc_addr(d->device, name) : NULL;
}

EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkGetRayTracingShaderGroupHandlesKHR(
    VkDevice device, VkPipeline pipeline, uint32_t firstGroup, uint32_t groupCount, size_t dataSize, void *pData)
{
    if (!g_rt_pipeline) {
        PFN_vkGetRayTracingShaderGroupHandlesKHR f = (PFN_vkGetRayTracingShaderGroupHandlesKHR)
            rt_export_next(device, "vkGetRayTracingShaderGroupHandlesKHR");
        return f ? f(device, pipeline, firstGroup, groupCount, dataSize, pData) : VK_ERROR_EXTENSION_NOT_PRESENT;
    }
    rt_patch_fex_thunks();
    return compat_GetRayTracingShaderGroupHandlesKHR(device, pipeline, firstGroup, groupCount, dataSize, pData);
}

EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkGetRayTracingShaderGroupHandlesNV(
    VkDevice device, VkPipeline pipeline, uint32_t firstGroup, uint32_t groupCount, size_t dataSize, void *pData)
{
    if (!g_rt_pipeline) {
        PFN_vkGetRayTracingShaderGroupHandlesKHR f = (PFN_vkGetRayTracingShaderGroupHandlesKHR)
            rt_export_next(device, "vkGetRayTracingShaderGroupHandlesNV");
        return f ? f(device, pipeline, firstGroup, groupCount, dataSize, pData) : VK_ERROR_EXTENSION_NOT_PRESENT;
    }
    rt_patch_fex_thunks();
    return compat_GetRayTracingShaderGroupHandlesKHR(device, pipeline, firstGroup, groupCount, dataSize, pData);
}

EXPORT VKAPI_ATTR void VKAPI_CALL vkCmdTraceRaysKHR(
    VkCommandBuffer commandBuffer,
    const VkStridedDeviceAddressRegionKHR *pRaygenShaderBindingTable,
    const VkStridedDeviceAddressRegionKHR *pMissShaderBindingTable,
    const VkStridedDeviceAddressRegionKHR *pHitShaderBindingTable,
    const VkStridedDeviceAddressRegionKHR *pCallableShaderBindingTable,
    uint32_t width, uint32_t height, uint32_t depth)
{
    if (!g_rt_pipeline) {
        PFN_vkCmdTraceRaysKHR f = (PFN_vkCmdTraceRaysKHR)rt_export_next_cmd(commandBuffer, "vkCmdTraceRaysKHR");
        if (f) f(commandBuffer, pRaygenShaderBindingTable, pMissShaderBindingTable, pHitShaderBindingTable,
                 pCallableShaderBindingTable, width, height, depth);
        return;
    }
    rt_patch_fex_thunks();
    compat_CmdTraceRaysKHR(commandBuffer, pRaygenShaderBindingTable, pMissShaderBindingTable,
                           pHitShaderBindingTable, pCallableShaderBindingTable, width, height, depth);
}

EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkCreateRayTracingPipelinesKHR(
    VkDevice device, VkDeferredOperationKHR deferredOperation, VkPipelineCache pipelineCache,
    uint32_t createInfoCount, const VkRayTracingPipelineCreateInfoKHR *pCreateInfos,
    const VkAllocationCallbacks *pAllocator, VkPipeline *pPipelines)
{
    if (!g_rt_pipeline) {
        PFN_vkCreateRayTracingPipelinesKHR f = (PFN_vkCreateRayTracingPipelinesKHR)
            rt_export_next(device, "vkCreateRayTracingPipelinesKHR");
        return f ? f(device, deferredOperation, pipelineCache, createInfoCount, pCreateInfos, pAllocator, pPipelines)
                 : VK_ERROR_EXTENSION_NOT_PRESENT;
    }
    rt_patch_fex_thunks();
    return compat_CreateRayTracingPipelinesKHR(device, deferredOperation, pipelineCache,
                                               createInfoCount, pCreateInfos, pAllocator, pPipelines);
}

EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkGetRayTracingCaptureReplayShaderGroupHandlesKHR(
    VkDevice device, VkPipeline pipeline, uint32_t firstGroup, uint32_t groupCount, size_t dataSize, void *pData)
{
    if (!g_rt_pipeline) {
        PFN_vkGetRayTracingCaptureReplayShaderGroupHandlesKHR f = (PFN_vkGetRayTracingCaptureReplayShaderGroupHandlesKHR)
            rt_export_next(device, "vkGetRayTracingCaptureReplayShaderGroupHandlesKHR");
        return f ? f(device, pipeline, firstGroup, groupCount, dataSize, pData) : VK_ERROR_EXTENSION_NOT_PRESENT;
    }
    rt_patch_fex_thunks();
    return compat_GetRayTracingCaptureReplayShaderGroupHandlesKHR(device, pipeline, firstGroup,
                                                                 groupCount, dataSize, pData);
}

EXPORT VKAPI_ATTR void VKAPI_CALL vkCmdTraceRaysIndirectKHR(
    VkCommandBuffer commandBuffer,
    const VkStridedDeviceAddressRegionKHR *pRaygenShaderBindingTable,
    const VkStridedDeviceAddressRegionKHR *pMissShaderBindingTable,
    const VkStridedDeviceAddressRegionKHR *pHitShaderBindingTable,
    const VkStridedDeviceAddressRegionKHR *pCallableShaderBindingTable,
    VkDeviceAddress indirectDeviceAddress)
{
    if (!g_rt_pipeline) {
        PFN_vkCmdTraceRaysIndirectKHR f = (PFN_vkCmdTraceRaysIndirectKHR)
            rt_export_next_cmd(commandBuffer, "vkCmdTraceRaysIndirectKHR");
        if (f) f(commandBuffer, pRaygenShaderBindingTable, pMissShaderBindingTable, pHitShaderBindingTable,
                 pCallableShaderBindingTable, indirectDeviceAddress);
        return;
    }
    rt_patch_fex_thunks();
    compat_CmdTraceRaysIndirectKHR(commandBuffer, pRaygenShaderBindingTable, pMissShaderBindingTable,
                                   pHitShaderBindingTable, pCallableShaderBindingTable, indirectDeviceAddress);
}

EXPORT VKAPI_ATTR void VKAPI_CALL vkCmdTraceRaysIndirect2KHR(
    VkCommandBuffer commandBuffer, VkDeviceAddress indirectDeviceAddress)
{
    if (!g_rt_pipeline) {
        PFN_vkCmdTraceRaysIndirect2KHR f = (PFN_vkCmdTraceRaysIndirect2KHR)
            rt_export_next_cmd(commandBuffer, "vkCmdTraceRaysIndirect2KHR");
        if (f) f(commandBuffer, indirectDeviceAddress);
        return;
    }
    rt_patch_fex_thunks();
    compat_CmdTraceRaysIndirect2KHR(commandBuffer, indirectDeviceAddress);
}

EXPORT VKAPI_ATTR VkDeviceSize VKAPI_CALL vkGetRayTracingShaderGroupStackSizeKHR(
    VkDevice device, VkPipeline pipeline, uint32_t group, VkShaderGroupShaderKHR groupShader)
{
    if (!g_rt_pipeline) {
        PFN_vkGetRayTracingShaderGroupStackSizeKHR f = (PFN_vkGetRayTracingShaderGroupStackSizeKHR)
            rt_export_next(device, "vkGetRayTracingShaderGroupStackSizeKHR");
        return f ? f(device, pipeline, group, groupShader) : 0;
    }
    rt_patch_fex_thunks();
    return compat_GetRayTracingShaderGroupStackSizeKHR(device, pipeline, group, groupShader);
}

EXPORT VKAPI_ATTR void VKAPI_CALL vkCmdSetRayTracingPipelineStackSizeKHR(
    VkCommandBuffer commandBuffer, uint32_t pipelineStackSize)
{
    if (!g_rt_pipeline) {
        PFN_vkCmdSetRayTracingPipelineStackSizeKHR f = (PFN_vkCmdSetRayTracingPipelineStackSizeKHR)
            rt_export_next_cmd(commandBuffer, "vkCmdSetRayTracingPipelineStackSizeKHR");
        if (f) f(commandBuffer, pipelineStackSize);
        return;
    }
    rt_patch_fex_thunks();
    compat_CmdSetRayTracingPipelineStackSizeKHR(commandBuffer, pipelineStackSize);
}

EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkCreateAccelerationStructureKHR(
    VkDevice device,
    const VkAccelerationStructureCreateInfoKHR *pCreateInfo,
    const VkAllocationCallbacks *pAllocator,
    VkAccelerationStructureKHR *pAccelerationStructure)
{
    if (!g_as_hooks) {
        struct DeviceData *d = get_device_data(device);
        return d && d->fpCreateAccelerationStructureKHR
            ? d->fpCreateAccelerationStructureKHR(device, pCreateInfo, pAllocator, pAccelerationStructure)
            : VK_ERROR_EXTENSION_NOT_PRESENT;
    }
    rt_patch_fex_thunks();
    return compat_CreateAccelerationStructureKHR(device, pCreateInfo, pAllocator, pAccelerationStructure);
}

EXPORT VKAPI_ATTR void VKAPI_CALL vkDestroyAccelerationStructureKHR(
    VkDevice device,
    VkAccelerationStructureKHR accelerationStructure,
    const VkAllocationCallbacks *pAllocator)
{
    if (!g_as_hooks) {
        struct DeviceData *d = get_device_data(device);
        if (d && d->fpDestroyAccelerationStructureKHR) d->fpDestroyAccelerationStructureKHR(device, accelerationStructure, pAllocator);
        return;
    }
    rt_patch_fex_thunks();
    compat_DestroyAccelerationStructureKHR(device, accelerationStructure, pAllocator);
}

EXPORT VKAPI_ATTR void VKAPI_CALL vkCmdBuildAccelerationStructuresKHR(
    VkCommandBuffer commandBuffer,
    uint32_t infoCount,
    const VkAccelerationStructureBuildGeometryInfoKHR *pInfos,
    const VkAccelerationStructureBuildRangeInfoKHR * const *ppBuildRangeInfos)
{
    if (!g_as_hooks) {
        struct DeviceData *d = get_device_data_by_cmd(commandBuffer);
        if (d && d->fpCmdBuildAccelerationStructuresKHR)
            d->fpCmdBuildAccelerationStructuresKHR(commandBuffer, infoCount, pInfos, ppBuildRangeInfos);
        return;
    }
    rt_patch_fex_thunks();
    compat_CmdBuildAccelerationStructuresKHR(commandBuffer, infoCount, pInfos, ppBuildRangeInfos);
}

EXPORT VKAPI_ATTR void VKAPI_CALL vkCmdCopyAccelerationStructureKHR(
    VkCommandBuffer commandBuffer,
    const VkCopyAccelerationStructureInfoKHR *pInfo)
{
    if (!g_as_hooks) {
        struct DeviceData *d = get_device_data_by_cmd(commandBuffer);
        if (d && d->fpCmdCopyAccelerationStructureKHR) d->fpCmdCopyAccelerationStructureKHR(commandBuffer, pInfo);
        return;
    }
    rt_patch_fex_thunks();
    compat_CmdCopyAccelerationStructureKHR(commandBuffer, pInfo);
}

EXPORT VKAPI_ATTR void VKAPI_CALL vkCmdWriteAccelerationStructuresPropertiesKHR(
    VkCommandBuffer commandBuffer,
    uint32_t accelerationStructureCount,
    const VkAccelerationStructureKHR *pAccelerationStructures,
    VkQueryType queryType,
    VkQueryPool queryPool,
    uint32_t firstQuery)
{
    if (!g_as_hooks) {
        struct DeviceData *d = get_device_data_by_cmd(commandBuffer);
        if (d && d->fpCmdWriteAccelerationStructuresPropertiesKHR)
            d->fpCmdWriteAccelerationStructuresPropertiesKHR(commandBuffer, accelerationStructureCount,
                                                             pAccelerationStructures, queryType, queryPool, firstQuery);
        return;
    }
    rt_patch_fex_thunks();
    compat_CmdWriteAccelerationStructuresPropertiesKHR(commandBuffer, accelerationStructureCount,
                                                       pAccelerationStructures, queryType, queryPool, firstQuery);
}

EXPORT VKAPI_ATTR VkDeviceAddress VKAPI_CALL vkGetAccelerationStructureDeviceAddressKHR(
    VkDevice device,
    const VkAccelerationStructureDeviceAddressInfoKHR *pInfo)
{
    if (!g_as_hooks) {
        struct DeviceData *d = get_device_data(device);
        return d && d->fpGetAccelerationStructureDeviceAddressKHR ? d->fpGetAccelerationStructureDeviceAddressKHR(device, pInfo) : 0;
    }
    rt_patch_fex_thunks();
    return compat_GetAccelerationStructureDeviceAddressKHR(device, pInfo);
}

EXPORT VKAPI_ATTR void VKAPI_CALL vkGetAccelerationStructureBuildSizesKHR(
    VkDevice device,
    VkAccelerationStructureBuildTypeKHR buildType,
    const VkAccelerationStructureBuildGeometryInfoKHR *pBuildInfo,
    const uint32_t *pMaxPrimitiveCounts,
    VkAccelerationStructureBuildSizesInfoKHR *pSizeInfo)
{
    if (!g_as_hooks) {
        struct DeviceData *d = get_device_data(device);
        if (d && d->fpGetAccelerationStructureBuildSizesKHR)
            d->fpGetAccelerationStructureBuildSizesKHR(device, buildType, pBuildInfo, pMaxPrimitiveCounts, pSizeInfo);
        return;
    }
    rt_patch_fex_thunks();
    compat_GetAccelerationStructureBuildSizesKHR(device, buildType, pBuildInfo, pMaxPrimitiveCounts, pSizeInfo);
}

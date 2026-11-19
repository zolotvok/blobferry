// SPDX-License-Identifier: GPL-3.0-or-later
// cl_device.h: CreateInstance, CreateDevice, DestroyDevice (most getenv sites, every module init). Included by mali_compat_layer.c.
// -----------------------------------------------------------------------------
// Layer Core: CreateInstance & CreateDevice
// -----------------------------------------------------------------------------
static VKAPI_ATTR VkResult VKAPI_CALL compat_CreateInstance(
    const VkInstanceCreateInfo *pCreateInfo, const VkAllocationCallbacks *pAllocator,
    VkInstance *pInstance) {
    g_debug = getenv("MALI_COMPAT_DEBUG") != NULL;
    g_stats = getenv("MALI_COMPAT_STATS") && strcmp(getenv("MALI_COMPAT_STATS"), "0") != 0;
    if (g_stats && !strcmp(getenv("MALI_COMPAT_STATS"), "every")) g_stats_interval_ns = 0;
    g_dump_spirv = access("/root/mali-spv", W_OK) == 0;
    g_dump_inflight = access("/root/mali-spv-inflight", W_OK) == 0;
    g_log_pipelines = access("/root/mali-pipelines.txt", W_OK) == 0;
    g_desc_pool_scale = getenv("MALI_COMPAT_DESC_POOL_SCALE") ? (uint32_t)atoi(getenv("MALI_COMPAT_DESC_POOL_SCALE")) : 0;
    if (g_desc_pool_scale > 64) g_desc_pool_scale = 64;
    g_desc_pool_pad = getenv("MALI_COMPAT_DESC_POOL_PAD") ? (uint32_t)atoi(getenv("MALI_COMPAT_DESC_POOL_PAD")) : 0;
    if (g_desc_pool_pad > 64) g_desc_pool_pad = 64;
    g_fail_log = getenv("MALI_COMPAT_FAIL_LOG") && atoi(getenv("MALI_COMPAT_FAIL_LOG")) > 0;
    g_fail_lvl = g_fail_log ? atoi(getenv("MALI_COMPAT_FAIL_LOG")) : 0;
    g_dsl_stage_retry = getenv("MALI_COMPAT_DSL_STAGE_RETRY") && atoi(getenv("MALI_COMPAT_DSL_STAGE_RETRY")) > 0;
    g_log_images = access("/root/mali-images.txt", W_OK) == 0;
    g_rt_pipeline = getenv("MALI_COMPAT_RT_PIPELINE") && strcmp(getenv("MALI_COMPAT_RT_PIPELINE"), "0") != 0;
    g_as_hooks = g_rt_pipeline || (getenv("MALI_COMPAT_AS_FIX") && strcmp(getenv("MALI_COMPAT_AS_FIX"), "0") != 0);
    sal_init(); // MALI_COMPAT_SSBO_ALIGN_LOG (pthread_once)
    g_bc_afrc = !getenv("MALI_COMPAT_BC_AFRC") || strcmp(getenv("MALI_COMPAT_BC_AFRC"), "0") != 0;
    bool rt = getenv("MALI_COMPAT_RT_COMPRESS") && strcmp(getenv("MALI_COMPAT_RT_COMPRESS"), "0") != 0;
    if (rt && !g_rt_compress) rt_keep_load(); // once per process: DXVK creates several instances
    g_rt_compress = rt;
    g_gpu_time = getenv("MALI_COMPAT_GPU_TIME") && strcmp(getenv("MALI_COMPAT_GPU_TIME"), "0") != 0;
    g_wait_callers = g_gpu_time && getenv("MALI_COMPAT_WAIT_CALLERS") && strcmp(getenv("MALI_COMPAT_WAIT_CALLERS"), "0") != 0;
    { // K-CF2 port + memlog (stage kcs1+kcf2+memlog); the init line is printed once per process, always
        const char *bmb = getenv("MALI_COMPAT_BUDGET_MB");
        unsigned long long mb = bmb ? strtoull(bmb, NULL, 10) : 0;
        g_budget_accounted = mb != 0;
        g_budget_cap = mb ? (uint64_t)mb << 20 : 6ULL << 30;
        const char *bls = getenv("MALI_COMPAT_BUDGET_LOG_S");
        unsigned long ls = bls ? strtoul(bls, NULL, 10) : 30;
        g_budget_log_s = g_budget_accounted && (g_gpu_time || g_debug) && ls ? (uint32_t)(ls > 3600 ? 3600 : ls) : 0;
        static _Atomic bool ml_started;
        const char *ml = getenv("MALI_COMPAT_MEMLOG");
        const char *am = getenv("MALI_COMPAT_ADDRMAP"); // addrmap-1113: turns the memlog on (as =1 unless MEMLOG=2)
        bool am_on = am && *am && strcmp(am, "0");
        bool ml_on = ml && *ml && strcmp(ml, "0");
        static char ml_path[300];
        if ((ml_on || am_on) && !atomic_exchange(&ml_started, true)) {
            const char *mmb = getenv("MALI_COMPAT_MEMLOG_MB");
            unsigned long long dflt = am_on ? 256 : 64, cap = mmb ? strtoull(mmb, NULL, 10) : dflt;
            g_ml_cap = (cap ? cap : dflt) << 20;
            g_memlog = ml_on && !strcmp(ml, "2") ? 2 : 1;
            g_addrmap = am_on;
            char comm[32] = "unknown";
            FILE *cf = fopen("/proc/self/comm", "r");
            if (cf) { if (fgets(comm, sizeof comm, cf)) comm[strcspn(comm, "\n")] = 0; fclose(cf); }
            for (char *c = comm; *c; c++) if (*c == '/' || *c == ' ' || *c == '\\') *c = '_';
            snprintf(ml_path, sizeof ml_path, "/root/mali-memlog-%s.tsv", comm);
            ml_open(ml_path, comm);
            if (g_addrmap)
                fprintf(stderr, "mali-compat: addrmap on (MALI_COMPAT_ADDRMAP): %s, cap %llu MiB, memlog=%d%s, events A M U F X + "
                        "B D L R N C, S every submit, clock line every 10 s\n", ml_path, (unsigned long long)(g_ml_cap >> 20),
                        g_memlog, g_memlog == 2 ? " (no capture address: B va from the mapping only)" : "");
            else if (am_on)
                fprintf(stderr, "mali-compat: addrmap OFF: the memlog file could not be opened\n");
        }
        static _Atomic bool budget_logged;
        if (!atomic_exchange(&budget_logged, true)) {
            char mls[400] = "off";
            if (g_memlog)
                snprintf(mls, sizeof mls, "on (%s, cap %llu MiB%s)", ml_path, (unsigned long long)(g_ml_cap >> 20),
                         g_memlog == 2 ? ", no capture address" : "");
            if (g_budget_accounted)
                fprintf(stderr, "mali-compat: heapBudget cap %llu MiB (MALI_COMPAT_BUDGET_MB), accounted usage, budget log %s, memlog %s\n",
                        (unsigned long long)(g_budget_cap >> 20), g_budget_log_s ? "on" : "off", mls);
            else
                fprintf(stderr, "mali-compat: heapBudget min(heap, 6144 MiB), cumulative usage (old report, MALI_COMPAT_BUDGET_MB unset), budget log off, memlog %s\n", mls);
            if (g_budget_log_s) fprintf(stderr, "mali-compat: budget log every %u s\n", g_budget_log_s);
        }
    }
    dpd_init(); // MALI_COMPAT_DPOOL_DIAG (pthread_once; after the ADDRMAP block: it follows ADDRMAP when unset)
    g_depth_nocomp = getenv("MALI_COMPAT_DEPTH_NOCOMP") && strcmp(getenv("MALI_COMPAT_DEPTH_NOCOMP"), "0") != 0;
    g_force_comp = getenv("MALI_COMPAT_FORCE_COMP") && strcmp(getenv("MALI_COMPAT_FORCE_COMP"), "0") != 0;
    g_no_susres = getenv("MALI_COMPAT_NO_SUSPEND_RESUME") && strcmp(getenv("MALI_COMPAT_NO_SUSPEND_RESUME"), "0") != 0;
    g_susres_barrier = g_no_susres && getenv("MALI_COMPAT_SUSRES_BARRIER") && strcmp(getenv("MALI_COMPAT_SUSRES_BARRIER"), "0") != 0;
    g_pcache_on = getenv("MALI_COMPAT_PIPELINE_CACHE") && strcmp(getenv("MALI_COMPAT_PIPELINE_CACHE"), "0") != 0;
    g_pcache_mem_only = g_pcache_on && !strcmp(getenv("MALI_COMPAT_PIPELINE_CACHE"), "2");
    if (getenv("MALI_COMPAT_PIPELINE_CACHE_THREADS")) g_pc_threads = getenv("MALI_COMPAT_PIPELINE_CACHE_THREADS");
    g_force_coherent = getenv("MALI_COMPAT_FORCE_COHERENT") && strcmp(getenv("MALI_COMPAT_FORCE_COHERENT"), "0") != 0;
    g_spv_fix = getenv("MALI_COMPAT_SPV_FIX") && strcmp(getenv("MALI_COMPAT_SPV_FIX"), "0") != 0;
    g_fake_occlusion = getenv("MALI_COMPAT_FAKE_OCCLUSION") && strcmp(getenv("MALI_COMPAT_FAKE_OCCLUSION"), "0") != 0;
    g_vendor_id = getenv("MALI_COMPAT_VENDOR_ID") ? (uint32_t)strtoul(getenv("MALI_COMPAT_VENDOR_ID"), NULL, 16) : 0;
    g_device_id = getenv("MALI_COMPAT_DEVICE_ID") ? (uint32_t)strtoul(getenv("MALI_COMPAT_DEVICE_ID"), NULL, 16) : 0;
    g_device_name = getenv("MALI_COMPAT_DEVICE_NAME");
    // on by default: the write it removes is discarded by the spec anyway (MALI_COMPAT_FRAGDEPTH_FIX=0 turns it off)
    g_fragdepth_fix = !getenv("MALI_COMPAT_FRAGDEPTH_FIX") || strcmp(getenv("MALI_COMPAT_FRAGDEPTH_FIX"), "0") != 0;
    g_gt_trace = getenv("MALI_COMPAT_GT_TRACE") ? (uint32_t)strtoul(getenv("MALI_COMPAT_GT_TRACE"), NULL, 16) : 0;
    if (getenv("MALI_COMPAT_GT_TRACE") && !strcmp(getenv("MALI_COMPAT_GT_TRACE"), "all")) g_gt_trace = UINT32_MAX;
    drm_node_init();
    g_dmabuf_export = getenv("MALI_COMPAT_DMABUF_EXPORT") && strcmp(getenv("MALI_COMPAT_DMABUF_EXPORT"), "0") != 0;
    g_async_ms = getenv("MALI_COMPAT_ASYNC_PSO") ? (uint32_t)atoi(getenv("MALI_COMPAT_ASYNC_PSO")) : 0;
    if (getenv("MALI_COMPAT_ASYNC_THREADS") && atoi(getenv("MALI_COMPAT_ASYNC_THREADS")) > 0)
        g_async_threads = (uint32_t)atoi(getenv("MALI_COMPAT_ASYNC_THREADS"));
    g_spv_dontunroll = getenv("MALI_COMPAT_SPV_DONTUNROLL") && strcmp(getenv("MALI_COMPAT_SPV_DONTUNROLL"), "0") != 0;
    wls_clamp_init(); // MALI_COMPAT_WLS_CLAMP (wls-clamp-1)
    wls_fix_init(); // MALI_COMPAT_WLS_FIX (wls-fix-1)
    g_xfb_vs = getenv("MALI_COMPAT_XFB_VS") && strcmp(getenv("MALI_COMPAT_XFB_VS"), "0") != 0;
    const char *deq = getenv("MALI_COMPAT_DEPTH_EQUAL");
    g_depth_equal_on = deq && (!strcmp(deq, "always") || !strcmp(deq, "lequal") || !strcmp(deq, "gequal"));
    g_depth_equal_op = !g_depth_equal_on ? VK_COMPARE_OP_EQUAL : !strcmp(deq, "always") ? VK_COMPARE_OP_ALWAYS
                     : !strcmp(deq, "lequal") ? VK_COMPARE_OP_LESS_OR_EQUAL : VK_COMPARE_OP_GREATER_OR_EQUAL;
    if (deq && *deq && strcmp(deq, "0") && !g_depth_equal_on)
        fprintf(stderr, "mali-compat: MALI_COMPAT_DEPTH_EQUAL=%s ignored (always, lequal or gequal)\n", deq);
    g_vs_scan = g_gpu_time || (getenv("MALI_COMPAT_VS_STORES") && strcmp(getenv("MALI_COMPAT_VS_STORES"), "0") != 0);
    g_noatt_w = g_noatt_h = 0;
    if (getenv("MALI_COMPAT_NOATT_AREA") && sscanf(getenv("MALI_COMPAT_NOATT_AREA"), "%ux%u", &g_noatt_w, &g_noatt_h) != 2)
        g_noatt_w = g_noatt_h = 0;
    ct_parse_env();
    g_skip_hash = getenv("MALI_COMPAT_SKIP_HASH") ? (uint32_t)strtoul(getenv("MALI_COMPAT_SKIP_HASH"), NULL, 16) : 0;
    g_dump_hash_count = 0;
    for (const char *h = getenv("MALI_COMPAT_DUMP_HASH"); h && *h && g_dump_hash_count < 16;) {
        char *end;
        unsigned long v = strtoul(h, &end, 16);
        if (end == h) break;
        g_dump_hashes[g_dump_hash_count++] = (uint32_t)v;
        h = *end == ',' ? end + 1 : end;
    }
    dc_init(); // MALI_COMPAT_DESC_CAPTURE (desc-capture-1)

    VkLayerInstanceCreateInfo *chain_info = (VkLayerInstanceCreateInfo *)pCreateInfo->pNext;
    while (chain_info &&
           (chain_info->sType != VK_STRUCTURE_TYPE_LOADER_INSTANCE_CREATE_INFO ||
            chain_info->function != VK_LAYER_LINK_INFO)) {
        chain_info = (VkLayerInstanceCreateInfo *)chain_info->pNext;
    }

    if (!chain_info || !chain_info->u.pLayerInfo) return VK_ERROR_INITIALIZATION_FAILED;

    PFN_vkGetInstanceProcAddr fpGetInstanceProcAddr = chain_info->u.pLayerInfo->pfnNextGetInstanceProcAddr;
    chain_info->u.pLayerInfo = chain_info->u.pLayerInfo->pNext;

    PFN_vkCreateInstance fpCreateInstance = (PFN_vkCreateInstance)fpGetInstanceProcAddr(VK_NULL_HANDLE, "vkCreateInstance");
    if (!fpCreateInstance) return VK_ERROR_INITIALIZATION_FAILED;

    VkResult res = fpCreateInstance(pCreateInfo, pAllocator, pInstance);
    if (res != VK_SUCCESS) return res;

    pthread_mutex_lock(&g_table_mutex);
    if (g_instance_count < MAX_INSTANCES) {
        struct InstanceData *inst = &g_instances[g_instance_count++];
        memset(inst, 0, sizeof(*inst));
        inst->instance = *pInstance;
        inst->app_api = pCreateInfo->pApplicationInfo ? pCreateInfo->pApplicationInfo->apiVersion : 0;
        inst->dispatch_key = *pInstance ? *(void **)*pInstance : NULL;
        inst->get_instance_proc_addr = fpGetInstanceProcAddr;
        #define LOAD_INST(fn) inst->fp##fn = (PFN_vk##fn)fpGetInstanceProcAddr(*pInstance, "vk" #fn)
        LOAD_INST(DestroyInstance);
        LOAD_INST(EnumeratePhysicalDevices);
        LOAD_INST(GetPhysicalDeviceFeatures);
        LOAD_INST(GetPhysicalDeviceFeatures2);
        if (!inst->fpGetPhysicalDeviceFeatures2)
            inst->fpGetPhysicalDeviceFeatures2 = (PFN_vkGetPhysicalDeviceFeatures2)fpGetInstanceProcAddr(*pInstance, "vkGetPhysicalDeviceFeatures2KHR");
        LOAD_INST(GetPhysicalDeviceFormatProperties);
        LOAD_INST(GetPhysicalDeviceFormatProperties2);
        if (!inst->fpGetPhysicalDeviceFormatProperties2)
            inst->fpGetPhysicalDeviceFormatProperties2 = (PFN_vkGetPhysicalDeviceFormatProperties2)fpGetInstanceProcAddr(*pInstance, "vkGetPhysicalDeviceFormatProperties2KHR");
        LOAD_INST(GetPhysicalDeviceImageFormatProperties);
        LOAD_INST(GetPhysicalDeviceImageFormatProperties2);
        if (!inst->fpGetPhysicalDeviceImageFormatProperties2)
            inst->fpGetPhysicalDeviceImageFormatProperties2 = (PFN_vkGetPhysicalDeviceImageFormatProperties2)fpGetInstanceProcAddr(*pInstance, "vkGetPhysicalDeviceImageFormatProperties2KHR");
        LOAD_INST(GetPhysicalDeviceMemoryProperties);
        LOAD_INST(GetPhysicalDeviceMemoryProperties2);
        if (!inst->fpGetPhysicalDeviceMemoryProperties2)
            inst->fpGetPhysicalDeviceMemoryProperties2 = (PFN_vkGetPhysicalDeviceMemoryProperties2)fpGetInstanceProcAddr(*pInstance, "vkGetPhysicalDeviceMemoryProperties2KHR");
        LOAD_INST(CreateDevice);
        LOAD_INST(EnumerateDeviceExtensionProperties);
        LOAD_INST(GetPhysicalDeviceProperties);
        LOAD_INST(GetPhysicalDeviceProperties2);
        if (!inst->fpGetPhysicalDeviceProperties2)
            inst->fpGetPhysicalDeviceProperties2 = (PFN_vkGetPhysicalDeviceProperties2)fpGetInstanceProcAddr(*pInstance, "vkGetPhysicalDeviceProperties2KHR");
        #undef LOAD_INST
    }
    pthread_mutex_unlock(&g_table_mutex);

    if (g_rt_pipeline) {
        fprintf(stderr, "mali-compat: RT pipeline (VK_KHR_ray_tracing_pipeline -> VK_KHR_ray_query megakernel) enabled\n");
        rt_patch_fex_thunks();
    } else if (g_as_hooks) {
        fprintf(stderr, "mali-compat: AS fix (MALI_COMPAT_AS_FIX=1): acceleration-structure hooks on, RT pipeline emulation off\n");
    }
    LOG("created VkInstance %p (key=%p)", (void *)*pInstance, *pInstance ? *(void **)*pInstance : NULL);
    return VK_SUCCESS;
}

static VKAPI_ATTR void VKAPI_CALL compat_DestroyInstance(
    VkInstance instance, const VkAllocationCallbacks *pAllocator) {
    struct InstanceData *inst = get_instance_data(instance);
    PFN_vkDestroyInstance fpDestroy = inst ? inst->fpDestroyInstance : NULL;
    if (fpDestroy && instance)
        fpDestroy(instance, pAllocator);

    if (instance) {
        pthread_mutex_lock(&g_table_mutex);
        void *key = *(void **)instance;
        for (uint32_t i = 0; i < g_instance_count; i++) {
            if (g_instances[i].instance == instance || (key && g_instances[i].dispatch_key == key)) {
                for (uint32_t j = i; j + 1 < g_instance_count; j++) {
                    g_instances[j] = g_instances[j + 1];
                }
                g_instance_count--;
                break;
            }
        }
        pthread_mutex_unlock(&g_table_mutex);
    }
}

static VKAPI_ATTR VkResult VKAPI_CALL compat_CreateDevice(
    VkPhysicalDevice physicalDevice, const VkDeviceCreateInfo *pCreateInfo,
    const VkAllocationCallbacks *pAllocator, VkDevice *pDevice) {
    struct InstanceData *inst = get_instance_by_phys_dev(physicalDevice);
    dc_init(); // also here: a device created without this layer's vkCreateInstance (checks) still parses the key
    wls_clamp_init(); // same for MALI_COMPAT_WLS_CLAMP (pthread_once)
    wls_fix_init(); // and MALI_COMPAT_WLS_FIX (wls-fix-1)
    sd_init(); // sync-diag keys (pthread_once)
    sal_init(); // MALI_COMPAT_SSBO_ALIGN_LOG (pthread_once)
    dpd_init(); // MALI_COMPAT_DPOOL_DIAG (pthread_once)
    bcr_init(); // MALI_COMPAT_BCD_RING (pthread_once; before the pc2 log thread is started below)
    {   // once: real vs reported subgroup masks (+ storage alignment), to see what FAKE_SUBGROUP / FAKE_ALIGN change
        static int logged;
        if (!logged && inst && inst->fpGetPhysicalDeviceProperties2) {
            logged = 1;
            VkPhysicalDeviceVulkan11Properties r11 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_PROPERTIES};
            VkPhysicalDeviceProperties2 rp = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, &r11};
            inst->fpGetPhysicalDeviceProperties2(physicalDevice, &rp); // the driver's own answer (below this layer)
            fprintf(stderr, "mali-compat: subgroup real stages 0x%x ops 0x%x -> reported stages 0x%x ops 0x%x (real=%d); "
                    "minStorageBufferOffsetAlignment real %llu -> reported %llu (real=%d)\n",
                    r11.subgroupSupportedStages, r11.subgroupSupportedOperations,
                    sg_stages(r11.subgroupSupportedStages), sg_ops(r11.subgroupSupportedOperations), real_subgroup(),
                    (unsigned long long)rp.properties.limits.minStorageBufferOffsetAlignment,
                    (unsigned long long)(!real_align() && rp.properties.limits.minStorageBufferOffsetAlignment > 16
                                         ? 16 : rp.properties.limits.minStorageBufferOffsetAlignment), real_align());
        }
    }

    VkLayerDeviceCreateInfo *chain_info = (VkLayerDeviceCreateInfo *)pCreateInfo->pNext;
    while (chain_info &&
           (chain_info->sType != VK_STRUCTURE_TYPE_LOADER_DEVICE_CREATE_INFO ||
            chain_info->function != VK_LAYER_LINK_INFO)) {
        chain_info = (VkLayerDeviceCreateInfo *)chain_info->pNext;
    }

    if (!chain_info || !chain_info->u.pLayerInfo) return VK_ERROR_INITIALIZATION_FAILED;

    PFN_vkGetInstanceProcAddr fpGetInstanceProcAddr = chain_info->u.pLayerInfo->pfnNextGetInstanceProcAddr;
    PFN_vkGetDeviceProcAddr fpGetDeviceProcAddr = chain_info->u.pLayerInfo->pfnNextGetDeviceProcAddr;
    chain_info->u.pLayerInfo = chain_info->u.pLayerInfo->pNext;

    PFN_vkCreateDevice fpCreateDevice = (PFN_vkCreateDevice)fpGetInstanceProcAddr(
        inst ? inst->instance : VK_NULL_HANDLE, "vkCreateDevice");
    if (!fpCreateDevice && inst && inst->fpCreateDevice)
        fpCreateDevice = inst->fpCreateDevice;
    if (!fpCreateDevice) return VK_ERROR_INITIALIZATION_FAILED;

    // Mask spoofed features before calling down
    VkDeviceCreateInfo mod_ci = *pCreateInfo;
    // VK_EXT_map_memory_placed is the layer's own: keep its name and feature struct away from the driver.
    const char **ext_names = malloc((pCreateInfo->enabledExtensionCount + 12) * sizeof *ext_names);
    if (!ext_names) return VK_ERROR_OUT_OF_HOST_MEMORY;
    mod_ci.enabledExtensionCount = 0;
    bool div_ext = false, div_khr = false;
    bool drv_has_rq = false, drv_has_as = false, drv_has_dho = false, drv_has_plib = false;
    bool drv_has_pep = false;   /* round 4: VK_KHR_pipeline_executable_properties (MALI_COMPAT_RT_STATS=1 only) */
    if (g_rt_pipeline && inst && inst->fpEnumerateDeviceExtensionProperties) {
        uint32_t n = 0;
        inst->fpEnumerateDeviceExtensionProperties(physicalDevice, NULL, &n, NULL);
        VkExtensionProperties *props = n ? malloc(n * sizeof *props) : NULL;
        if (props && inst->fpEnumerateDeviceExtensionProperties(physicalDevice, NULL, &n, props) >= 0) {
            for (uint32_t i = 0; i < n; i++) {
                drv_has_rq   |= !strcmp(props[i].extensionName, "VK_KHR_ray_query");
                drv_has_as   |= !strcmp(props[i].extensionName, "VK_KHR_acceleration_structure");
                drv_has_dho  |= !strcmp(props[i].extensionName, "VK_KHR_deferred_host_operations");
                drv_has_plib |= !strcmp(props[i].extensionName, "VK_KHR_pipeline_library");
                drv_has_pep  |= !strcmp(props[i].extensionName, "VK_KHR_pipeline_executable_properties");
            }
        }
        free(props);
    }
    for (uint32_t i = 0; i < pCreateInfo->enabledExtensionCount; i++) {
        const char *e = pCreateInfo->ppEnabledExtensionNames[i];
        if (!strcmp(e, "VK_EXT_vertex_attribute_divisor")) { div_ext = true; continue; } // the vendor has only KHR
        div_khr |= !strcmp(e, "VK_KHR_vertex_attribute_divisor");
        if (g_drm_on && !strcmp(e, VK_EXT_PHYSICAL_DEVICE_DRM_EXTENSION_NAME)) continue; // the layer's (properties only)
        if (g_rt_pipeline) {
            if (!strcmp(e, "VK_KHR_ray_tracing_pipeline") ||
                !strcmp(e, "VK_KHR_ray_tracing_maintenance1") ||
                !strcmp(e, "VK_EXT_pipeline_library_group_handles")) continue;
            if (!strcmp(e, "VK_KHR_pipeline_library") && !drv_has_plib) continue;
            if (!strcmp(e, "VK_KHR_deferred_host_operations") && !drv_has_dho) continue;
        }
        if (strcmp(e, VK_EXT_MAP_MEMORY_PLACED_EXTENSION_NAME)) ext_names[mod_ci.enabledExtensionCount++] = e;
    }
    if (div_ext && !div_khr) ext_names[mod_ci.enabledExtensionCount++] = "VK_KHR_vertex_attribute_divisor";
    // MALI_COMPAT_BDA_EXT=1 (see bda_ext_on): enable VK_KHR_buffer_device_address for an app that did not. Not next to
    // an app-enabled VK_EXT_buffer_device_address (the two must not be enabled together). Only for an app on Vulkan
    // >= 1.1: there the extension's own dependency (physical-device-properties2) is core.
    bool bda_app = false, bda_app_ext = false, bda_added = false;
    uint32_t bda_api = app_api_by_phys_dev(physicalDevice);
    for (uint32_t i = 0; i < mod_ci.enabledExtensionCount; i++) {
        bda_app |= !strcmp(ext_names[i], VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME);
        bda_app_ext |= !strcmp(ext_names[i], "VK_EXT_buffer_device_address");
    }
    if (bda_ext_on() && !bda_app && !bda_app_ext && bda_api >= VK_API_VERSION_1_1 && inst &&
        inst->fpEnumerateDeviceExtensionProperties) {
        uint32_t n = 0;
        inst->fpEnumerateDeviceExtensionProperties(physicalDevice, NULL, &n, NULL);
        VkExtensionProperties *props = n ? malloc(n * sizeof *props) : NULL;
        if (props && inst->fpEnumerateDeviceExtensionProperties(physicalDevice, NULL, &n, props) >= 0)
            for (uint32_t i = 0; i < n; i++)
                bda_added |= !strcmp(props[i].extensionName, VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME);
        free(props);
        if (bda_added) ext_names[mod_ci.enabledExtensionCount++] = VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME;
    }
    // MALI_COMPAT_GPU_TIME on an app below Vulkan 1.2 (DOOM Eternal: 1.1): hostQueryReset is chained further down, but
    // the device has vkResetQueryPool(EXT) only with VK_EXT_host_query_reset enabled, so GPU timing was "unavailable".
    // Added when the app did not enable it, the driver lists it and the app is on >= 1.1 (the extension's dependency,
    // physical-device-properties2, is core there). One line per such device after gt_init.
    // [game/doom_eternal keeper, Claude Opus 5.5] 2026-11-13
    int hqr_state = -1; // -1 no line (key off or app >= 1.2), 0 not added, 1 added, 2 enabled by the app
    bool hqr_drv = false;
    if (g_gpu_time && bda_api < VK_API_VERSION_1_2) {
        bool app = false;
        for (uint32_t i = 0; i < mod_ci.enabledExtensionCount; i++)
            app |= !strcmp(ext_names[i], VK_EXT_HOST_QUERY_RESET_EXTENSION_NAME);
        if (inst && inst->fpEnumerateDeviceExtensionProperties) {
            uint32_t n = 0;
            inst->fpEnumerateDeviceExtensionProperties(physicalDevice, NULL, &n, NULL);
            VkExtensionProperties *props = n ? malloc(n * sizeof *props) : NULL;
            if (props && inst->fpEnumerateDeviceExtensionProperties(physicalDevice, NULL, &n, props) >= 0)
                for (uint32_t i = 0; i < n; i++) hqr_drv |= !strcmp(props[i].extensionName, VK_EXT_HOST_QUERY_RESET_EXTENSION_NAME);
            free(props);
        }
        hqr_state = app ? 2 : hqr_drv && bda_api >= VK_API_VERSION_1_1;
        if (hqr_state == 1) ext_names[mod_ci.enabledExtensionCount++] = VK_EXT_HOST_QUERY_RESET_EXTENSION_NAME;
    }
    if (g_rt_pipeline) {
        const char *need_rt[3] = {
            drv_has_dho ? "VK_KHR_deferred_host_operations" : NULL,
            drv_has_as  ? "VK_KHR_acceleration_structure"   : NULL,
            drv_has_rq  ? "VK_KHR_ray_query"                : NULL,
        };
        for (int k = 0; k < 3; k++) {
            if (!need_rt[k]) continue;
            bool have = false;
            for (uint32_t i = 0; i < mod_ci.enabledExtensionCount; i++)
                have |= !strcmp(ext_names[i], need_rt[k]);
            if (!have) ext_names[mod_ci.enabledExtensionCount++] = need_rt[k];
        }
    }
    /* round 4: MALI_COMPAT_RT_STATS=1 (RT_PIPELINE=1 only): VK_KHR_pipeline_executable_properties + its feature for the
     * megakernels' CAPTURE_STATISTICS, only when the driver has both; otherwise one line and the key does nothing. */
    bool rt_stats_on = false;
    VkPhysicalDevicePipelineExecutablePropertiesFeaturesKHR pep_feat = {
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PIPELINE_EXECUTABLE_PROPERTIES_FEATURES_KHR, NULL, VK_TRUE};
    VkPhysicalDevicePipelineExecutablePropertiesFeaturesKHR *app_pep = NULL;
    VkBool32 app_pep_saved = VK_FALSE;
    if (g_rt_pipeline && rt_env()->stats) {
        VkPhysicalDevicePipelineExecutablePropertiesFeaturesKHR q = {
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PIPELINE_EXECUTABLE_PROPERTIES_FEATURES_KHR};
        VkPhysicalDeviceFeatures2 f2q = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, &q};
        if (drv_has_pep && inst && inst->fpGetPhysicalDeviceFeatures2) inst->fpGetPhysicalDeviceFeatures2(physicalDevice, &f2q);
        rt_stats_on = drv_has_pep && q.pipelineExecutableInfo;
        LOG("rt_stats: MALI_COMPAT_RT_STATS=1: driver extension %s, pipelineExecutableInfo %s -> %s", drv_has_pep ? "yes" : "no",
            q.pipelineExecutableInfo ? "true" : "false", rt_stats_on ? "ON" : "OFF (no statistics)");
        if (rt_stats_on) {
            bool have = false;
            for (uint32_t i = 0; i < mod_ci.enabledExtensionCount; i++)
                have |= !strcmp(ext_names[i], "VK_KHR_pipeline_executable_properties");
            if (!have) ext_names[mod_ci.enabledExtensionCount++] = "VK_KHR_pipeline_executable_properties";
            app_pep = (VkPhysicalDevicePipelineExecutablePropertiesFeaturesKHR *)find_struct(pCreateInfo->pNext,
                VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PIPELINE_EXECUTABLE_PROPERTIES_FEATURES_KHR);
            if (app_pep) {   /* written only when the app left it off (as the layer does for rayQuery etc.) */
                app_pep_saved = app_pep->pipelineExecutableInfo;
                if (!app_pep_saved) app_pep->pipelineExecutableInfo = VK_TRUE; else app_pep = NULL;
            }
            else { pep_feat.pNext = (void *)mod_ci.pNext; mod_ci.pNext = &pep_feat; }
        }
    }
    // BC shadows get fixed-rate compression (AFRC) and the image log queries compression: both need
    // VK_EXT_image_compression_control and its feature. Enabled when the driver has it. [Claude Opus 5.5]
    bool want_icc = false;
    if ((g_log_images || g_bc_afrc) && inst && inst->fpEnumerateDeviceExtensionProperties) {
        uint32_t n = 0;
        inst->fpEnumerateDeviceExtensionProperties(physicalDevice, NULL, &n, NULL);
        VkExtensionProperties *props = n ? malloc(n * sizeof *props) : NULL;
        if (props && inst->fpEnumerateDeviceExtensionProperties(physicalDevice, NULL, &n, props) >= 0)
            for (uint32_t i = 0; i < n; i++)
                want_icc |= !strcmp(props[i].extensionName, VK_EXT_IMAGE_COMPRESSION_CONTROL_EXTENSION_NAME);
        free(props);
    }
    VkPhysicalDeviceImageCompressionControlFeaturesEXT icc_feat = {
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_IMAGE_COMPRESSION_CONTROL_FEATURES_EXT, NULL, VK_TRUE};
    VkPhysicalDeviceImageCompressionControlFeaturesEXT *app_icc =
        (VkPhysicalDeviceImageCompressionControlFeaturesEXT *)find_struct(pCreateInfo->pNext,
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_IMAGE_COMPRESSION_CONTROL_FEATURES_EXT);
    VkBool32 app_icc_saved = app_icc ? app_icc->imageCompressionControl : VK_FALSE;
    if (want_icc) {
        if (app_icc) app_icc->imageCompressionControl = VK_TRUE;
        else { icc_feat.pNext = (void *)mod_ci.pNext; mod_ci.pNext = &icc_feat; }
    }
    if (want_icc) {
        bool have = false;
        for (uint32_t i = 0; i < mod_ci.enabledExtensionCount; i++)
            have |= !strcmp(ext_names[i], VK_EXT_IMAGE_COMPRESSION_CONTROL_EXTENSION_NAME);
        if (!have) ext_names[mod_ci.enabledExtensionCount++] = VK_EXT_IMAGE_COMPRESSION_CONTROL_EXTENSION_NAME;
    }
    // cover-trim (MALI_COMPAT_COVER_TRIM): VK_EXT_conditional_rendering and its feature for the layer's predicated
    // dispatches and draws, when the driver has them; otherwise the feature stays off.
    bool ct_ext_ok = false;
    VkPhysicalDeviceConditionalRenderingFeaturesEXT ct_feat = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_CONDITIONAL_RENDERING_FEATURES_EXT,
                                                               NULL, VK_TRUE, VK_FALSE};
    VkPhysicalDeviceConditionalRenderingFeaturesEXT *app_ct = (VkPhysicalDeviceConditionalRenderingFeaturesEXT *)find_struct(
        pCreateInfo->pNext, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_CONDITIONAL_RENDERING_FEATURES_EXT);
    VkBool32 app_ct_saved = app_ct ? app_ct->conditionalRendering : VK_FALSE;
    if (g_ct && inst && inst->fpEnumerateDeviceExtensionProperties && inst->fpGetPhysicalDeviceFeatures2) {
        uint32_t n = 0;
        inst->fpEnumerateDeviceExtensionProperties(physicalDevice, NULL, &n, NULL);
        VkExtensionProperties *props = n ? malloc(n * sizeof *props) : NULL;
        if (props && inst->fpEnumerateDeviceExtensionProperties(physicalDevice, NULL, &n, props) >= 0)
            for (uint32_t i = 0; i < n; i++) ct_ext_ok |= !strcmp(props[i].extensionName, VK_EXT_CONDITIONAL_RENDERING_EXTENSION_NAME);
        free(props);
        VkPhysicalDeviceConditionalRenderingFeaturesEXT q = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_CONDITIONAL_RENDERING_FEATURES_EXT};
        VkPhysicalDeviceFeatures2 f2q = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, &q};
        if (ct_ext_ok) inst->fpGetPhysicalDeviceFeatures2(physicalDevice, &f2q);
        ct_ext_ok = ct_ext_ok && q.conditionalRendering;
    }
    if (g_ct && !ct_ext_ok) fprintf(stderr, "mali-compat: cover-trim: no VK_EXT_conditional_rendering on this device; off\n");
    if (ct_ext_ok) {
        bool have = false;
        for (uint32_t i = 0; i < mod_ci.enabledExtensionCount; i++) have |= !strcmp(ext_names[i], VK_EXT_CONDITIONAL_RENDERING_EXTENSION_NAME);
        if (!have) ext_names[mod_ci.enabledExtensionCount++] = VK_EXT_CONDITIONAL_RENDERING_EXTENSION_NAME;
        if (app_ct) app_ct->conditionalRendering = VK_TRUE;
        else { ct_feat.pNext = (void *)mod_ci.pNext; mod_ci.pNext = &ct_feat; }
    }
    mod_ci.ppEnabledExtensionNames = ext_names;
    struct unlinked placed_feat = unlink_struct(&mod_ci, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAP_MEMORY_PLACED_FEATURES_EXT);
    struct unlinked u_rtp_dev  = g_rt_pipeline ? unlink_struct(&mod_ci, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_FEATURES_KHR) : (struct unlinked){0};
    struct unlinked u_rtm1_dev = g_rt_pipeline ? unlink_struct(&mod_ci, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_MAINTENANCE_1_FEATURES_KHR) : (struct unlinked){0};
    struct unlinked u_plgh_dev = g_rt_pipeline ? unlink_struct(&mod_ci, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PIPELINE_LIBRARY_GROUP_HANDLES_FEATURES_EXT) : (struct unlinked){0};
    VkPhysicalDeviceRayQueryFeaturesKHR rq_feat = {
        .sType    = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_QUERY_FEATURES_KHR,
        .rayQuery = VK_TRUE,
    };
    VkPhysicalDeviceRayQueryFeaturesKHR *app_rq = (VkPhysicalDeviceRayQueryFeaturesKHR *)
        find_struct(mod_ci.pNext, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_QUERY_FEATURES_KHR);
    VkBool32 saved_app_rq = app_rq ? app_rq->rayQuery : VK_FALSE;
    if (g_rt_pipeline && drv_has_rq) {
        if (app_rq) app_rq->rayQuery = VK_TRUE;
        else { rq_feat.pNext = (void *)mod_ci.pNext; mod_ci.pNext = &rq_feat; }
    }
    // VkPhysicalDeviceFeatures in declaration order (55 VkBool32). The old table lacked fillModeNonSolid and depthBounds, so
    // every name from idx 13 on was shifted by two: the logged "wideLines (13)", "largePoints (14)", "pipelineStatisticsQuery
    // (22)", "fragmentStoresAndAtomics (24)", "shaderFloat64 (37)", "shaderInt64 (38)" were really fillModeNonSolid,
    // depthBounds, textureCompressionBC, pipelineStatisticsQuery, shaderClipDistance, shaderCullDistance (all false on the
    // blob). The masking itself compared by index and was right. [keeper, Claude Opus 5.5] 2026-11-06
    static const char *kFeatureNames[] = {
        "robustBufferAccess", "fullDrawIndexUint32", "imageCubeArray", "independentBlend",
        "geometryShader", "tessellationShader", "sampleRateShading", "dualSrcBlend",
        "logicOp", "multiDrawIndirect", "drawIndirectFirstInstance", "depthClamp",
        "depthBiasClamp", "fillModeNonSolid", "depthBounds", "wideLines", "largePoints", "alphaToOne",
        "multiViewport", "samplerAnisotropy", "textureCompressionETC2", "textureCompressionASTC_LDR",
        "textureCompressionBC", "occlusionQueryPrecise", "pipelineStatisticsQuery", "vertexPipelineStoresAndAtomics",
        "fragmentStoresAndAtomics", "shaderTessellationAndGeometryPointSize", "shaderImageGatherExtended", "shaderStorageImageExtendedFormats",
        "shaderStorageImageMultisample", "shaderStorageImageReadWithoutFormat", "shaderStorageImageWriteWithoutFormat", "shaderUniformBufferArrayDynamicIndexing",
        "shaderSampledImageArrayDynamicIndexing", "shaderStorageBufferArrayDynamicIndexing", "shaderStorageImageArrayDynamicIndexing", "shaderClipDistance",
        "shaderCullDistance", "shaderFloat64", "shaderInt64", "shaderInt16",
        "shaderResourceResidency", "shaderResourceMinLod", "sparseBinding", "sparseResidencyBuffer",
        "sparseResidencyImage2D", "sparseResidencyImage3D", "sparseResidency2Samples", "sparseResidency4Samples",
        "sparseResidency8Samples", "sparseResidency16Samples", "sparseResidencyAliased", "variableMultisampleRate",
        "inheritedQueries"
    };
    _Static_assert(sizeof kFeatureNames / sizeof kFeatureNames[0] == sizeof(VkPhysicalDeviceFeatures) / sizeof(VkBool32),
                   "kFeatureNames must name every VkPhysicalDeviceFeatures member in order");
    // One stderr line per vkCreateDevice: the names masked in pEnabledFeatures and/or VkPhysicalDeviceFeatures2.
    char masked_log[1024];
    size_t masked_len = 0;
    uint64_t masked_bits = 0;
    masked_log[0] = 0;

    // The real driver's core features through the next layer's vkGetPhysicalDeviceFeatures2 (the same struct the v1 call
    // fills; v1 only when the driver has no features2). Not the layer's own compat_GetPhysicalDeviceFeatures*, which spoofs.
    VkPhysicalDeviceFeatures real_features = {0};
    if (inst && inst->fpGetPhysicalDeviceFeatures2) {
        VkPhysicalDeviceFeatures2 rf2 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, NULL};
        inst->fpGetPhysicalDeviceFeatures2(physicalDevice, &rf2);
        real_features = rf2.features;
    } else if (inst && inst->fpGetPhysicalDeviceFeatures) {
        inst->fpGetPhysicalDeviceFeatures(physicalDevice, &real_features);
    }
    // The real driver's 16/8-bit storage bits: mask only what it lacks. Masking them unconditionally (f7dac02) created the
    // device WITHOUT features the vendor driver has (Mali-G1 54.1: all 7 true) while the app's SPIR-V still declares
    // StorageBuffer16BitAccess etc. (DOOM Eternal, vkd3d-proton). [game/doom_eternal, Claude Opus 5.5] 2026-11-06
    VkPhysicalDevice8BitStorageFeatures rs8 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_8BIT_STORAGE_FEATURES, NULL};
    VkPhysicalDevice16BitStorageFeatures rs16 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES, &rs8};
    if (inst && inst->fpGetPhysicalDeviceFeatures2) {
        VkPhysicalDeviceFeatures2 rsq = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, &rs16};
        inst->fpGetPhysicalDeviceFeatures2(physicalDevice, &rsq);
    }
    fprintf(stderr, "mali-compat: real storage16 %d%d%d%d storage8 %d%d%d (only false bits are masked)\n",
            rs16.storageBuffer16BitAccess, rs16.uniformAndStorageBuffer16BitAccess, rs16.storagePushConstant16,
            rs16.storageInputOutput16, rs8.storageBuffer8BitAccess, rs8.uniformAndStorageBuffer8BitAccess,
            rs8.storagePushConstant8);
    const VkBool32 *real_arr = (const VkBool32 *)&real_features;

    VkPhysicalDeviceFeatures safe_features;
    if (mod_ci.pEnabledFeatures) {
        safe_features = *mod_ci.pEnabledFeatures;
        VkBool32 *safe_arr = (VkBool32 *)&safe_features;
        for (size_t i = 0; i < sizeof(VkPhysicalDeviceFeatures) / sizeof(VkBool32); i++) {
            if (safe_arr[i] && !real_arr[i]) {
                if (!(masked_bits & (1ull << i)) && masked_len + 48 < sizeof masked_log)
                    masked_len += (size_t)snprintf(masked_log + masked_len, sizeof masked_log - masked_len, "%s%s(%zu)",
                                                   masked_len ? " " : "", kFeatureNames[i], i);
                masked_bits |= 1ull << i;
                safe_arr[i] = VK_FALSE;
            }
        }
        mod_ci.pEnabledFeatures = &safe_features;
    }

    // Mask features2 if chained, and check if bufferDeviceAddress is already in pNext
    VkPhysicalDeviceFeatures saved_f2_features;
    memset(&saved_f2_features, 0, sizeof(saved_f2_features));
    VkPhysicalDeviceFeatures2 *f2 = NULL;
    VkPhysicalDeviceVulkan11Features *v11_feat = NULL;
    VkBool32 saved_v11_sb16 = VK_FALSE, saved_v11_usb16 = VK_FALSE, saved_v11_pc16 = VK_FALSE, saved_v11_io16 = VK_FALSE;
    VkPhysicalDeviceVulkan12Features *v12_feat = NULL;
    VkBool32 saved_v12_bda = VK_FALSE, saved_v12_sb8 = VK_FALSE, saved_v12_usb8 = VK_FALSE, saved_v12_pc8 = VK_FALSE;
    VkPhysicalDevice16BitStorageFeatures *f16 = NULL;
    VkBool32 saved_f16_sb = VK_FALSE, saved_f16_usb = VK_FALSE, saved_f16_pc = VK_FALSE, saved_f16_io = VK_FALSE;
    VkPhysicalDevice8BitStorageFeatures *f8 = NULL;
    VkBool32 saved_f8_sb = VK_FALSE, saved_f8_usb = VK_FALSE, saved_f8_pc = VK_FALSE;
    VkPhysicalDeviceBufferDeviceAddressFeatures *existing_bda = NULL;
    VkBool32 saved_existing_bda = VK_FALSE;

    VkBaseOutStructure *node = (VkBaseOutStructure *)mod_ci.pNext;
    while (node) {
        if (node->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2) {
            f2 = (VkPhysicalDeviceFeatures2 *)node;
            saved_f2_features = f2->features;
            VkBool32 *f2_arr = (VkBool32 *)&f2->features;
            for (size_t i = 0; i < sizeof(VkPhysicalDeviceFeatures) / sizeof(VkBool32); i++) {
                if (f2_arr[i] && !real_arr[i]) {
                    if (!(masked_bits & (1ull << i)) && masked_len + 48 < sizeof masked_log)
                        masked_len += (size_t)snprintf(masked_log + masked_len, sizeof masked_log - masked_len, "%s%s(%zu)",
                                                       masked_len ? " " : "", kFeatureNames[i], i);
                    masked_bits |= 1ull << i;
                    f2_arr[i] = VK_FALSE;
                }
            }
        } else if (node->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES) {
            f16 = (VkPhysicalDevice16BitStorageFeatures *)node;
            saved_f16_sb = f16->storageBuffer16BitAccess;
            saved_f16_usb = f16->uniformAndStorageBuffer16BitAccess;
            saved_f16_pc = f16->storagePushConstant16;
            saved_f16_io = f16->storageInputOutput16;
            if (!rs16.storageBuffer16BitAccess) f16->storageBuffer16BitAccess = VK_FALSE;
            if (!rs16.uniformAndStorageBuffer16BitAccess) f16->uniformAndStorageBuffer16BitAccess = VK_FALSE;
            if (!rs16.storagePushConstant16) f16->storagePushConstant16 = VK_FALSE;
            if (!rs16.storageInputOutput16) f16->storageInputOutput16 = VK_FALSE;
        } else if (node->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_8BIT_STORAGE_FEATURES) {
            f8 = (VkPhysicalDevice8BitStorageFeatures *)node;
            saved_f8_sb = f8->storageBuffer8BitAccess;
            saved_f8_usb = f8->uniformAndStorageBuffer8BitAccess;
            saved_f8_pc = f8->storagePushConstant8;
            if (!rs8.storageBuffer8BitAccess) f8->storageBuffer8BitAccess = VK_FALSE;
            if (!rs8.uniformAndStorageBuffer8BitAccess) f8->uniformAndStorageBuffer8BitAccess = VK_FALSE;
            if (!rs8.storagePushConstant8) f8->storagePushConstant8 = VK_FALSE;
        } else if (node->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES) {
            v11_feat = (VkPhysicalDeviceVulkan11Features *)node;
            saved_v11_sb16 = v11_feat->storageBuffer16BitAccess;
            saved_v11_usb16 = v11_feat->uniformAndStorageBuffer16BitAccess;
            saved_v11_pc16 = v11_feat->storagePushConstant16;
            saved_v11_io16 = v11_feat->storageInputOutput16;
            if (!rs16.storageBuffer16BitAccess) v11_feat->storageBuffer16BitAccess = VK_FALSE;
            if (!rs16.uniformAndStorageBuffer16BitAccess) v11_feat->uniformAndStorageBuffer16BitAccess = VK_FALSE;
            if (!rs16.storagePushConstant16) v11_feat->storagePushConstant16 = VK_FALSE;
            if (!rs16.storageInputOutput16) v11_feat->storageInputOutput16 = VK_FALSE;
        } else if (node->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES) {
            v12_feat = (VkPhysicalDeviceVulkan12Features *)node;
            saved_v12_bda = v12_feat->bufferDeviceAddress;
            v12_feat->bufferDeviceAddress = VK_TRUE;
            saved_v12_sb8 = v12_feat->storageBuffer8BitAccess;
            saved_v12_usb8 = v12_feat->uniformAndStorageBuffer8BitAccess;
            saved_v12_pc8 = v12_feat->storagePushConstant8;
            if (!rs8.storageBuffer8BitAccess) v12_feat->storageBuffer8BitAccess = VK_FALSE;
            if (!rs8.uniformAndStorageBuffer8BitAccess) v12_feat->uniformAndStorageBuffer8BitAccess = VK_FALSE;
            if (!rs8.storagePushConstant8) v12_feat->storagePushConstant8 = VK_FALSE;
        } else if (node->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES) {
            existing_bda = (VkPhysicalDeviceBufferDeviceAddressFeatures *)node;
            saved_existing_bda = existing_bda->bufferDeviceAddress;
            existing_bda->bufferDeviceAddress = VK_TRUE;
        }
        node = node->pNext;
    }
    fprintf(stderr, "mali-compat: vkCreateDevice masks %d core feature(s) the driver lacks: %s\n",
            __builtin_popcountll(masked_bits), masked_len ? masked_log : "none");

    // Ensure bufferDeviceAddress is requested only if neither Vulkan12Features nor BDA struct was in pNext
    VkPhysicalDeviceBufferDeviceAddressFeatures bda_feat = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES,
        .pNext = (void *)mod_ci.pNext,
        .bufferDeviceAddress = VK_TRUE,
    };
    if (!v12_feat && !existing_bda) {
        mod_ci.pNext = &bda_feat;
    }

    // GPU timing resets its timestamp queries from the collector thread: hostQueryReset (Vulkan 1.2 core).
    VkPhysicalDeviceHostQueryResetFeatures hqr_feat = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_HOST_QUERY_RESET_FEATURES, NULL, VK_TRUE};
    VkPhysicalDeviceHostQueryResetFeatures *app_hqr = (VkPhysicalDeviceHostQueryResetFeatures *)find_struct(pCreateInfo->pNext,
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_HOST_QUERY_RESET_FEATURES);
    VkBool32 saved_v12_hqr = v12_feat ? v12_feat->hostQueryReset : VK_FALSE;
    VkBool32 saved_app_hqr = app_hqr ? app_hqr->hostQueryReset : VK_FALSE;
    if (g_gpu_time) {
        if (v12_feat) v12_feat->hostQueryReset = VK_TRUE;
        else if (app_hqr) app_hqr->hostQueryReset = VK_TRUE;
        else { hqr_feat.pNext = (void *)mod_ci.pNext; mod_ci.pNext = &hqr_feat; }
    }

    VkPhysicalDeviceVertexAttributeDivisorFeaturesKHR *app_div =
        (VkPhysicalDeviceVertexAttributeDivisorFeaturesKHR *)find_struct(pCreateInfo->pNext,
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VERTEX_ATTRIBUTE_DIVISOR_FEATURES_KHR);
    VkBool32 app_zero_div = app_div ? app_div->vertexAttributeInstanceRateZeroDivisor : VK_FALSE;
    bool zero_div_emulated = false;
    if (app_zero_div && inst && inst->fpGetPhysicalDeviceFeatures2) {
        VkPhysicalDeviceVertexAttributeDivisorFeaturesKHR real = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VERTEX_ATTRIBUTE_DIVISOR_FEATURES_KHR};
        VkPhysicalDeviceFeatures2 f2q = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, &real};
        inst->fpGetPhysicalDeviceFeatures2(physicalDevice, &f2q);
        if (!real.vertexAttributeInstanceRateZeroDivisor) {
            app_div->vertexAttributeInstanceRateZeroDivisor = VK_FALSE;
            zero_div_emulated = true;
        }
    }
    VkResult res = fpCreateDevice(physicalDevice, &mod_ci, pAllocator, pDevice);
    if (app_pep) app_pep->pipelineExecutableInfo = app_pep_saved;
    if (app_rq) app_rq->rayQuery = saved_app_rq;
    if (app_ct) app_ct->conditionalRendering = app_ct_saved;
    if (app_div) app_div->vertexAttributeInstanceRateZeroDivisor = app_zero_div;
    if (app_icc) app_icc->imageCompressionControl = app_icc_saved;
    relink(u_plgh_dev);
    relink(u_rtm1_dev);
    relink(u_rtp_dev);
    relink(placed_feat);

    // Restore mutated pNext fields
    if (f2) {
        f2->features = saved_f2_features;
    }
    if (f16) {
        f16->storageBuffer16BitAccess = saved_f16_sb;
        f16->uniformAndStorageBuffer16BitAccess = saved_f16_usb;
        f16->storagePushConstant16 = saved_f16_pc;
        f16->storageInputOutput16 = saved_f16_io;
    }
    if (f8) {
        f8->storageBuffer8BitAccess = saved_f8_sb;
        f8->uniformAndStorageBuffer8BitAccess = saved_f8_usb;
        f8->storagePushConstant8 = saved_f8_pc;
    }
    if (v11_feat) {
        v11_feat->storageBuffer16BitAccess = saved_v11_sb16;
        v11_feat->uniformAndStorageBuffer16BitAccess = saved_v11_usb16;
        v11_feat->storagePushConstant16 = saved_v11_pc16;
        v11_feat->storageInputOutput16 = saved_v11_io16;
    }
    if (v12_feat) {
        v12_feat->bufferDeviceAddress = saved_v12_bda;
        v12_feat->storageBuffer8BitAccess = saved_v12_sb8;
        v12_feat->uniformAndStorageBuffer8BitAccess = saved_v12_usb8;
        v12_feat->storagePushConstant8 = saved_v12_pc8;
    }
    if (existing_bda) existing_bda->bufferDeviceAddress = saved_existing_bda;
    if (v12_feat) v12_feat->hostQueryReset = saved_v12_hqr;
    if (app_hqr) app_hqr->hostQueryReset = saved_app_hqr;

    if (res != VK_SUCCESS) {
        fprintf(stderr, "mali-compat: fpCreateDevice failed: %d (enabledExtensionCount=%u)\n", res, mod_ci.enabledExtensionCount);
        for (uint32_t i = 0; i < mod_ci.enabledExtensionCount; i++) {
            fprintf(stderr, "mali-compat:   requested ext[%u]: %s\n", i, mod_ci.ppEnabledExtensionNames[i]);
        }
        VkBaseOutStructure *curr = (VkBaseOutStructure *)mod_ci.pNext;
        while (curr) {
            fprintf(stderr, "mali-compat:   pNext sType: %d\n", curr->sType);
            curr = curr->pNext;
        }
        fflush(stderr);
        free(ext_names);
        return res;
    }
    free(ext_names);

    pthread_mutex_lock(&g_table_mutex);
    if (g_device_count < MAX_DEVICES) {
        struct DeviceData *d = &g_devices[g_device_count++];
        memset(d, 0, sizeof(*d));
        d->device = *pDevice;
        d->dispatch_key = *pDevice ? *(void **)*pDevice : NULL;
        d->physical_device = physicalDevice;
        d->instance_data = inst;
        d->get_device_proc_addr = fpGetDeviceProcAddr;
        #define LOAD_DEV(fn) d->fp##fn = (PFN_vk##fn)fpGetDeviceProcAddr(*pDevice, "vk" #fn)
        LOAD_DEV(DestroyDevice);
        LOAD_DEV(GetDeviceQueue);
        LOAD_DEV(CreateShaderModule);
        LOAD_DEV(DestroyShaderModule);
        LOAD_DEV(CreateGraphicsPipelines);
        LOAD_DEV(CreateComputePipelines);
        LOAD_DEV(CreateDescriptorPool);
        LOAD_DEV(CreateDescriptorSetLayout);
        LOAD_DEV(AllocateDescriptorSets);
        LOAD_DEV(DestroyPipeline);
        LOAD_DEV(CreatePipelineLayout);
        LOAD_DEV(DestroyPipelineLayout);
        LOAD_DEV(CreateImage);
        LOAD_DEV(DestroyImage);
        LOAD_DEV(CreateImageView);
        LOAD_DEV(DestroyImageView);
        LOAD_DEV(CreateBufferView);
        LOAD_DEV(CreatePipelineCache);
        LOAD_DEV(GetPipelineCacheData);
        LOAD_DEV(DestroyPipelineCache);
        LOAD_DEV(CreateQueryPool);
        LOAD_DEV(DestroyQueryPool);
        LOAD_DEV(GetQueryPoolResults);
        LOAD_DEV(CmdCopyQueryPoolResults);
        LOAD_DEV(CmdFillBuffer);
        LOAD_DEV(GetDeviceImageMemoryRequirements);
        if (!d->fpGetDeviceImageMemoryRequirements)
            d->fpGetDeviceImageMemoryRequirements = (PFN_vkGetDeviceImageMemoryRequirements)fpGetDeviceProcAddr(*pDevice, "vkGetDeviceImageMemoryRequirementsKHR");
        LOAD_DEV(GetDeviceImageSparseMemoryRequirements);
        if (!d->fpGetDeviceImageSparseMemoryRequirements)
            d->fpGetDeviceImageSparseMemoryRequirements = (PFN_vkGetDeviceImageSparseMemoryRequirements)fpGetDeviceProcAddr(*pDevice, "vkGetDeviceImageSparseMemoryRequirementsKHR");
        LOAD_DEV(CreateBuffer);
        LOAD_DEV(DestroyBuffer);
        LOAD_DEV(AllocateMemory);
        LOAD_DEV(FreeMemory);
        LOAD_DEV(GetMemoryFdKHR);
        LOAD_DEV(GetMemoryFdPropertiesKHR);
        LOAD_DEV(BindBufferMemory);
        LOAD_DEV(GetBufferMemoryRequirements);
        LOAD_DEV(GetBufferDeviceAddress);
        if (!d->fpGetBufferDeviceAddress)
            d->fpGetBufferDeviceAddress = (PFN_vkGetBufferDeviceAddress)fpGetDeviceProcAddr(*pDevice, "vkGetBufferDeviceAddressKHR");
        if (bda_ext_on() || !d->fpGetBufferDeviceAddress)
            fprintf(stderr, "mali-compat: bda: VkDevice %p, app apiVersion %u.%u, VK_KHR_buffer_device_address app %d "
                    "(EXT %d) layer-added %d (MALI_COMPAT_BDA_EXT=%d), vkGetBufferDeviceAddress %s%s\n", (void *)*pDevice,
                    VK_API_VERSION_MAJOR(bda_api), VK_API_VERSION_MINOR(bda_api), bda_app, bda_app_ext, bda_added,
                    bda_ext_on(), d->fpGetBufferDeviceAddress ? "yes" : "NO",
                    d->fpGetBufferDeviceAddress ? "" : ": BC uploads will be skipped (textures undecoded)");
        LOAD_DEV(CmdBindPipeline);
        LOAD_DEV(CmdPushConstants);
        LOAD_DEV(CmdPushConstants2);
        LOAD_DEV(CmdPushConstants2KHR);
        LOAD_DEV(CmdDispatch);
        LOAD_DEV(CmdPipelineBarrier);
        LOAD_DEV(CmdCopyBufferToImage);
        LOAD_DEV(CmdCopyBufferToImage2);
        if (!d->fpCmdCopyBufferToImage2)
            d->fpCmdCopyBufferToImage2 = (PFN_vkCmdCopyBufferToImage2)fpGetDeviceProcAddr(*pDevice, "vkCmdCopyBufferToImage2KHR");
        LOAD_DEV(ResetCommandBuffer);
        LOAD_DEV(BeginCommandBuffer);
        LOAD_DEV(AllocateCommandBuffers);
        LOAD_DEV(FreeCommandBuffers);
        LOAD_DEV(ResetCommandPool);
        LOAD_DEV(DestroyCommandPool);
        LOAD_DEV(MapMemory);
        LOAD_DEV(UnmapMemory);
        LOAD_DEV(MapMemory2KHR);
        LOAD_DEV(UnmapMemory2KHR);
        LOAD_DEV(FlushMappedMemoryRanges);
        LOAD_DEV(InvalidateMappedMemoryRanges);
        d->afrc = want_icc && g_bc_afrc;
        d->zero_divisor_emulated = zero_div_emulated;
        const VkPhysicalDeviceDescriptorBufferFeaturesEXT *dbf = find_struct(pCreateInfo->pNext,
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_BUFFER_FEATURES_EXT);
        d->descriptor_buffer = dbf && dbf->descriptorBuffer;
        d->fpCmdBindDescriptorBuffersEXT = (PFN_vkCmdBindDescriptorBuffersEXT)fpGetDeviceProcAddr(*pDevice, "vkCmdBindDescriptorBuffersEXT");
        d->fpCmdSetDescriptorBufferOffsetsEXT = (PFN_vkCmdSetDescriptorBufferOffsetsEXT)fpGetDeviceProcAddr(*pDevice, "vkCmdSetDescriptorBufferOffsetsEXT");
        d->fpCmdBindDescriptorSets = (PFN_vkCmdBindDescriptorSets)fpGetDeviceProcAddr(*pDevice, "vkCmdBindDescriptorSets");
        d->fpCmdDispatchIndirect = (PFN_vkCmdDispatchIndirect)fpGetDeviceProcAddr(*pDevice, "vkCmdDispatchIndirect");
        d->fpCmdUpdateBuffer = (PFN_vkCmdUpdateBuffer)fpGetDeviceProcAddr(*pDevice, "vkCmdUpdateBuffer");
        d->fpCreateAccelerationStructureKHR = (PFN_vkCreateAccelerationStructureKHR)fpGetDeviceProcAddr(*pDevice, "vkCreateAccelerationStructureKHR");
        d->fpDestroyAccelerationStructureKHR = (PFN_vkDestroyAccelerationStructureKHR)fpGetDeviceProcAddr(*pDevice, "vkDestroyAccelerationStructureKHR");
        d->fpCmdBuildAccelerationStructuresKHR = (PFN_vkCmdBuildAccelerationStructuresKHR)fpGetDeviceProcAddr(*pDevice, "vkCmdBuildAccelerationStructuresKHR");
        d->fpCmdCopyAccelerationStructureKHR = (PFN_vkCmdCopyAccelerationStructureKHR)fpGetDeviceProcAddr(*pDevice, "vkCmdCopyAccelerationStructureKHR");
        d->fpCmdCopyBuffer = (PFN_vkCmdCopyBuffer)fpGetDeviceProcAddr(*pDevice, "vkCmdCopyBuffer");
        if (rt_stats_on) {   /* round 4: MALI_COMPAT_RT_STATS=1 */
            d->fpGetPipelineExecutablePropertiesKHR = (PFN_vkGetPipelineExecutablePropertiesKHR)
                fpGetDeviceProcAddr(*pDevice, "vkGetPipelineExecutablePropertiesKHR");
            d->fpGetPipelineExecutableStatisticsKHR = (PFN_vkGetPipelineExecutableStatisticsKHR)
                fpGetDeviceProcAddr(*pDevice, "vkGetPipelineExecutableStatisticsKHR");
            d->rt_stats_ok = d->fpGetPipelineExecutablePropertiesKHR && d->fpGetPipelineExecutableStatisticsKHR;
        }
        d->fpCmdWriteAccelerationStructuresPropertiesKHR = (PFN_vkCmdWriteAccelerationStructuresPropertiesKHR)fpGetDeviceProcAddr(*pDevice, "vkCmdWriteAccelerationStructuresPropertiesKHR");
        d->fpGetAccelerationStructureDeviceAddressKHR = (PFN_vkGetAccelerationStructureDeviceAddressKHR)fpGetDeviceProcAddr(*pDevice, "vkGetAccelerationStructureDeviceAddressKHR");
        d->fpGetAccelerationStructureBuildSizesKHR = (PFN_vkGetAccelerationStructureBuildSizesKHR)fpGetDeviceProcAddr(*pDevice, "vkGetAccelerationStructureBuildSizesKHR");
        pthread_mutex_init(&d->bc_lock, NULL);
        d->fpCmdPushDescriptorSetKHR = (PFN_vkCmdPushDescriptorSetKHR)fpGetDeviceProcAddr(*pDevice, "vkCmdPushDescriptorSetKHR");
        if (!d->fpCmdPushDescriptorSetKHR)
            d->fpCmdPushDescriptorSetKHR = (PFN_vkCmdPushDescriptorSetKHR)fpGetDeviceProcAddr(*pDevice, "vkCmdPushDescriptorSet");
        d->fpCmdSetVertexInputEXT = (PFN_vkCmdSetVertexInputEXT)fpGetDeviceProcAddr(*pDevice, "vkCmdSetVertexInputEXT");
        d->fpCmdSetDepthCompareOp = (PFN_vkCmdSetDepthCompareOp)fpGetDeviceProcAddr(*pDevice, "vkCmdSetDepthCompareOp");
        if (!d->fpCmdSetDepthCompareOp)
            d->fpCmdSetDepthCompareOp = (PFN_vkCmdSetDepthCompareOp)fpGetDeviceProcAddr(*pDevice, "vkCmdSetDepthCompareOpEXT");
        if (g_log_images && want_icc) {
            d->fpGetImageSubresourceLayout2KHR = (PFN_vkGetImageSubresourceLayout2KHR)fpGetDeviceProcAddr(*pDevice, "vkGetImageSubresourceLayout2KHR");
            if (!d->fpGetImageSubresourceLayout2KHR)
                d->fpGetImageSubresourceLayout2KHR = (PFN_vkGetImageSubresourceLayout2KHR)fpGetDeviceProcAddr(*pDevice, "vkGetImageSubresourceLayout2EXT");
        }
        d->fpCreateComputePipelinesReal = (PFN_vkCreateComputePipelines)fpGetDeviceProcAddr(*pDevice, "vkCreateComputePipelines");
        #undef LOAD_DEV
        if (g_gpu_time || g_async_ms || g_noatt_w || g_no_susres || g_dc_on || g_ct) gt_init(d, fpGetDeviceProcAddr); // without GPU_TIME: pointers only, no pool
        if (hqr_state >= 0)
            fprintf(stderr, "mali-compat: gpu-time: VK_EXT_host_query_reset %s for a Vulkan %u.%u app (driver lists it: %d), "
                    "vkResetQueryPool %s\n", hqr_state == 1 ? "added" : hqr_state == 2 ? "enabled by the app" : "NOT added",
                    VK_API_VERSION_MAJOR(bda_api), VK_API_VERSION_MINOR(bda_api), hqr_drv,
                    d->gt && d->gt->fpResetQueryPool ? "yes" : "NO");
        if (g_ct && ct_ext_ok) ct_device_init(d, fpGetDeviceProcAddr);
        if (g_addrmap) am_device_init(d, fpGetDeviceProcAddr, *pDevice);
        if (g_dpd_on) dpd_device_init(d, fpGetDeviceProcAddr, *pDevice);
        if (g_memlog) {
            d->ml_fpQueueSubmit = (PFN_vkQueueSubmit)fpGetDeviceProcAddr(*pDevice, "vkQueueSubmit");
            d->ml_fpQueueSubmit2 = (PFN_vkQueueSubmit2)fpGetDeviceProcAddr(*pDevice, "vkQueueSubmit2");
            if (!d->ml_fpQueueSubmit2) d->ml_fpQueueSubmit2 = (PFN_vkQueueSubmit2)fpGetDeviceProcAddr(*pDevice, "vkQueueSubmit2KHR");
            if (g_memlog == 1) { // bufferDeviceAddress is always enabled above, every allocation gets DEVICE_ADDRESS_BIT
                d->fpGetDeviceMemoryOpaqueCaptureAddress = (PFN_vkGetDeviceMemoryOpaqueCaptureAddress)fpGetDeviceProcAddr(*pDevice, "vkGetDeviceMemoryOpaqueCaptureAddress");
                if (!d->fpGetDeviceMemoryOpaqueCaptureAddress)
                    d->fpGetDeviceMemoryOpaqueCaptureAddress = (PFN_vkGetDeviceMemoryOpaqueCaptureAddress)fpGetDeviceProcAddr(*pDevice, "vkGetDeviceMemoryOpaqueCaptureAddressKHR");
            }
        }
        if (d->gt) { // SUSRES_BARRIER: vkCmdPipelineBarrier2 only when synchronization2 is enabled, else the v1 fallback
            const VkPhysicalDeviceVulkan13Features *v13 = find_struct(mod_ci.pNext, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES);
            const VkPhysicalDeviceSynchronization2Features *s2 = find_struct(mod_ci.pNext, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SYNCHRONIZATION_2_FEATURES);
            if (!(v13 && v13->synchronization2) && !(s2 && s2->synchronization2)) d->gt->fpCmdPipelineBarrier2 = NULL;
        }
        pc_init(d);
        memset(d->type_heap, BUDGET_NO_HEAP, sizeof d->type_heap);
        if (inst && inst->fpGetPhysicalDeviceMemoryProperties) {
            VkPhysicalDeviceMemoryProperties mp;
            inst->fpGetPhysicalDeviceMemoryProperties(physicalDevice, &mp);
            for (uint32_t i = 0; i < mp.memoryTypeCount && i < VK_MAX_MEMORY_TYPES; i++)
                if (mp.memoryTypes[i].heapIndex < VK_MAX_MEMORY_HEAPS) d->type_heap[i] = (uint8_t)mp.memoryTypes[i].heapIndex;
            for (uint32_t i = 0; i < mp.memoryHeapCount && i < VK_MAX_MEMORY_HEAPS; i++)
                g_heap_budget[i] = mp.memoryHeaps[i].size > g_budget_cap ? g_budget_cap : mp.memoryHeaps[i].size;
            atomic_store(&g_heap_count, mp.memoryHeapCount);
            static _Atomic bool budget_log_started;
            if (g_budget_log_s && !atomic_exchange(&budget_log_started, true)) {
                pthread_t t;
                if (pthread_create(&t, NULL, budget_log_thread, NULL) == 0) pthread_detach(t);
            }
            for (uint32_t i = 0; i < mp.memoryTypeCount; i++)
                if (mp.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) d->coherent_types |= 1u << i;
            const VkMemoryPropertyFlags hc = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
            for (uint32_t i = 0; i < mp.memoryTypeCount; i++)
                if ((mp.memoryTypes[i].propertyFlags & hc) == VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)
                    for (uint32_t j = 0; j < mp.memoryTypeCount; j++)
                        if ((mp.memoryTypes[j].propertyFlags & hc) == hc && mp.memoryTypes[j].heapIndex == mp.memoryTypes[i].heapIndex) {
                            d->coherent_remap[i] = (uint8_t)(j + 1);
                            break;
                        }
        }
        d->bc_init_done = false;
        if (g_dc_on) dc_device_init(d, fpGetDeviceProcAddr, *pDevice);
        if (g_wls_clamp) {
            char hl[16 * 9 + 1] = "";
            for (uint32_t i = 0; i < g_wls_hash_count; i++)
                snprintf(hl + strlen(hl), sizeof hl - strlen(hl), "%s%08x", i ? "," : "", g_wls_hashes[i]);
            fprintf(stderr, "mali-compat: wls-clamp-1 active: VkDevice %p, modules %s, so far %llu modules / %llu chains\n",
                    (void *)*pDevice, g_wls_clamp == 1 ? "all" : hl, (unsigned long long)atomic_load(&g_wls_modules),
                    (unsigned long long)atomic_load(&g_wls_chains));
        }
        fprintf(stderr, "mali-compat: pc2-restore-1 active: VkDevice %p, driver vkCmdPushConstants2 %s, "
                "vkCmdPushConstants2KHR %s\n", (void *)*pDevice, d->fpCmdPushConstants2 ? "yes" : "no",
                d->fpCmdPushConstants2KHR ? "yes" : "no");
        static _Atomic bool pc2_log_started;
        if (!atomic_exchange(&pc2_log_started, true)) {
            pthread_t t;
            if (pthread_create(&t, NULL, pc2_log_thread, NULL) == 0) pthread_detach(t);
        }
    }
    DEV_TABLE_CHANGED();
    pthread_mutex_unlock(&g_table_mutex);

    LOG("created VkDevice %p (key=%p)", (void *)*pDevice, *pDevice ? *(void **)*pDevice : NULL);
    return VK_SUCCESS;
}

static VKAPI_ATTR void VKAPI_CALL compat_DestroyDevice(
    VkDevice device, const VkAllocationCallbacks *pAllocator) {
    // rc-1115: read the dispatch key before the driver frees the dispatchable object (ASan heap-use-after-free)
    void *key = device ? *(void **)device : NULL;
    // 11-10 keeper: a megakernel compile still running on a worker (DestroyPipeline does not wait for it) must
    // finish its driver calls before the device goes away (Codex 1110-194217 "detached compile outlives its device").
    if (device) rt_drain_device_jobs(device);
    struct DeviceData *d = get_device_data(device);
    if (g_memlog) ml_flush();
    if (d) {
        if (g_bcr) bcr_end_dump("device destroy"); // MALI_COMPAT_BCD_RING: once, when there are entries since the last dump
        ct_device_destroy(d);
        sd_device_destroy(d);
        gt_destroy(d);
        pc_destroy(d);
        if (d->bc_init_done) {
            for (int i = 0; i < 4; i++) {
                if (d->bc_pipelines[i]) d->fpDestroyPipeline(device, d->bc_pipelines[i], NULL);
                if (d->bc_pipelines_db[i]) d->fpDestroyPipeline(device, d->bc_pipelines_db[i], NULL);
            }
            if (d->bc_layout) d->fpDestroyPipelineLayout(device, d->bc_layout, NULL);
        }
        if (d->rt_ind_pipeline && d->fpDestroyPipeline) d->fpDestroyPipeline(device, d->rt_ind_pipeline, NULL);
        if (d->rt_ind_layout && d->fpDestroyPipelineLayout) d->fpDestroyPipelineLayout(device, d->rt_ind_layout, NULL);
        if (d->rt_split_pipeline && d->fpDestroyPipeline) d->fpDestroyPipeline(device, d->rt_split_pipeline, NULL);
        if (d->rt_split_layout && d->fpDestroyPipelineLayout) d->fpDestroyPipelineLayout(device, d->rt_split_layout, NULL);
        if (d->rt_zero_buf && d->fpDestroyBuffer) d->fpDestroyBuffer(device, d->rt_zero_buf, NULL);
        if (d->rt_zero_mem && d->fpFreeMemory) am_drv_free(d, "rt-zero", d->rt_zero_mem);
        if (d->rt_ring_buf && d->fpDestroyBuffer) d->fpDestroyBuffer(device, d->rt_ring_buf, NULL);
        if (d->rt_ring_mem && d->fpFreeMemory) {
            if (d->rt_ring_ptr && d->fpUnmapMemory) d->fpUnmapMemory(device, d->rt_ring_mem);
            am_drv_free(d, "rt-ring", d->rt_ring_mem);
        }
        if (d->rt_scratch_buf && d->fpDestroyBuffer) d->fpDestroyBuffer(device, d->rt_scratch_buf, NULL);
        if (d->rt_scratch_mem && d->fpFreeMemory) am_drv_free(d, "rt-scratch", d->rt_scratch_mem);
        if (g_as_hooks) rt_device_gone(d);   /* round 4: the layer's RT allocations leave rt_mem; final lines */
        pthread_mutex_destroy(&d->bc_lock);
        destroy_staging_pool(d);
        if (g_dpd_on) dpd_device_gone(d); // before the driver's destroy: no fence of it is polled afterwards
        if (d->fpDestroyDevice && device)
            d->fpDestroyDevice(device, pAllocator);
        if (g_addrmap) { ml_flush(); am_end("device destroy"); }
        if (g_wls_fix) { // wls-fix-1 summary
            char hl[16 * 9 + 1] = "";
            for (uint32_t i = 0; i < g_wlsf_count; i++)
                snprintf(hl + strlen(hl), sizeof hl - strlen(hl), "%s%08x", i ? "," : "", g_wlsf_hashes[i]);
            fprintf(stderr, "mali-compat: wls-fix: VkDevice %p destroyed, modules %s, so far %llu rewritten / %llu left as they are\n",
                    (void *)device, hl, (unsigned long long)atomic_load(&g_wlsf_rewritten), (unsigned long long)atomic_load(&g_wlsf_left));
        }
        free(d->amf);
        d->amf = NULL;
    }
    if (device) {
        pthread_mutex_lock(&g_table_mutex);
        for (uint32_t i = 0; i < g_device_count; i++) {
            if (g_devices[i].device == device || (key && g_devices[i].dispatch_key == key)) {
                for (uint32_t j = i; j + 1 < g_device_count; j++) {
                    g_devices[j] = g_devices[j + 1];
                }
                g_device_count--;
                break;
            }
        }
        DEV_TABLE_CHANGED();
        pthread_mutex_unlock(&g_table_mutex);
    }
}


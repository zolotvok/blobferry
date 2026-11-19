// SPDX-License-Identifier: GPL-3.0-or-later
// cl_diag_addrmap.h: MALI_COMPAT_ADDRMAP hooks (hook table am_proc) and the end report. Included by mali_compat_layer.c.
// -----------------------------------------------------------------------------
// MALI_COMPAT_ADDRMAP hooks (handed out by vkGetDeviceProcAddr only with the key, and only for names the driver has).
// Each one calls what the dispatch would have handed out without the key (desc-capture's / the AS hooks' own entry
// where those are on, else the driver), then writes its event. [game/doom_eternal keeper, Claude Opus 5.5] 2026-11-13
// -----------------------------------------------------------------------------
static VKAPI_ATTR VkResult VKAPI_CALL compat_AmBindBufferMemory(VkDevice device, VkBuffer b, VkDeviceMemory m, VkDeviceSize off) {
    struct DeviceData *d = get_device_data(device);
    VkResult r = g_dc_on && d->dcf ? compat_DcBindBufferMemory(device, b, m, off) : d->fpBindBufferMemory(device, b, m, off);
    if (r == VK_SUCCESS) am_bind(d, device, 1, (uint64_t)(uintptr_t)b, m, off, true, NULL);
    return r;
}
static VKAPI_ATTR VkResult VKAPI_CALL compat_AmBindBufferMemory2(VkDevice device, uint32_t n, const VkBindBufferMemoryInfo *bi) {
    struct DeviceData *d = get_device_data(device);
    VkResult r = g_dc_on && d->dcf ? compat_DcBindBufferMemory2(device, n, bi) : d->amf->BindBufferMemory2(device, n, bi);
    for (uint32_t i = 0; r == VK_SUCCESS && i < n; i++) am_bind(d, device, 1, (uint64_t)(uintptr_t)bi[i].buffer, bi[i].memory, bi[i].memoryOffset, true, " v=2");
    return r;
}
static VKAPI_ATTR VkResult VKAPI_CALL compat_AmBindImageMemory(VkDevice device, VkImage img, VkDeviceMemory m, VkDeviceSize off) {
    struct DeviceData *d = get_device_data(device);
    VkResult r = d->amf->BindImageMemory(device, img, m, off);
    if (r == VK_SUCCESS) am_bind(d, device, 2, (uint64_t)(uintptr_t)img, m, off, true, NULL);
    return r;
}
static VKAPI_ATTR VkResult VKAPI_CALL compat_AmBindImageMemory2(VkDevice device, uint32_t n, const VkBindImageMemoryInfo *bi) {
    struct DeviceData *d = get_device_data(device);
    VkResult r = d->amf->BindImageMemory2(device, n, bi);
    for (uint32_t i = 0; r == VK_SUCCESS && i < n; i++) {
        const VkBindImagePlaneMemoryInfo *pl = find_struct(bi[i].pNext, VK_STRUCTURE_TYPE_BIND_IMAGE_PLANE_MEMORY_INFO);
        const VkBindImageMemorySwapchainInfoKHR *sw = find_struct(bi[i].pNext, VK_STRUCTURE_TYPE_BIND_IMAGE_MEMORY_SWAPCHAIN_INFO_KHR);
        char extra[96] = " v=2";
        if (pl) snprintf(extra, sizeof extra, " v=2 plane=0x%x", pl->planeAspect);
        if (sw) snprintf(extra + strlen(extra), sizeof extra - strlen(extra), " swapchain=0x%llx index=%u",
                         (unsigned long long)sw->swapchain, sw->imageIndex);
        am_bind(d, device, 2, (uint64_t)(uintptr_t)bi[i].image, bi[i].memory, bi[i].memoryOffset, !pl && !sw, extra);
    }
    return r;
}
static VKAPI_ATTR void VKAPI_CALL compat_AmDestroyBuffer(VkDevice device, VkBuffer b, const VkAllocationCallbacks *a) {
    if (b) am_destroy(1, (uint64_t)(uintptr_t)b); // before the driver: the handle may be reused right after
    struct DeviceData *d = get_device_data(device);
    if (g_as_hooks) { compat_RtDestroyBuffer(device, b, a); return; }
    if (g_dc_on && d->dcf) { compat_DcDestroyBuffer(device, b, a); return; }
    d->fpDestroyBuffer(device, b, a);
}
static VKAPI_ATTR VkResult VKAPI_CALL compat_AmCreateCommandPool(VkDevice device, const VkCommandPoolCreateInfo *ci,
                                                                 const VkAllocationCallbacks *a, VkCommandPool *p) {
    VkResult r = get_device_data(device)->amf->CreateCommandPool(device, ci, a, p);
    am_emit('N', r == VK_SUCCESS && p ? (uint64_t)*p : 0, AM_NA, ML_NONE, BUDGET_NO_HEAP, AM_NA, 0,
            "what=cpool-create flags=0x%x qfi=%u result=%d", ci->flags, ci->queueFamilyIndex, r);
    return r;
}
static VKAPI_ATTR VkResult VKAPI_CALL compat_AmResetDescriptorPool(VkDevice device, VkDescriptorPool pool, VkDescriptorPoolResetFlags fl) {
    struct DeviceData *d = get_device_data(device);
    if (!g_dpd_on) am_emit('N', (uint64_t)pool, AM_NA, ML_NONE, BUDGET_NO_HEAP, AM_NA, 0, "what=dpool-reset"); // dpool-diag: per-second counts
    return g_dc_on && d->dcf ? compat_DcResetDescriptorPool(device, pool, fl) : d->amf->ResetDescriptorPool(device, pool, fl);
}
static VKAPI_ATTR void VKAPI_CALL compat_AmDestroyDescriptorPool(VkDevice device, VkDescriptorPool pool, const VkAllocationCallbacks *a) {
    struct DeviceData *d = get_device_data(device);
    if (pool) am_emit('N', (uint64_t)pool, AM_NA, ML_NONE, BUDGET_NO_HEAP, AM_NA, 0, "what=dpool-destroy");
    if (g_dc_on && d->dcf) compat_DcDestroyDescriptorPool(device, pool, a);
    else d->amf->DestroyDescriptorPool(device, pool, a);
}
static VKAPI_ATTR VkResult VKAPI_CALL compat_AmCreateSwapchainKHR(VkDevice device, const VkSwapchainCreateInfoKHR *ci,
                                                                  const VkAllocationCallbacks *a, VkSwapchainKHR *sc) {
    VkResult r = get_device_data(device)->amf->CreateSwapchainKHR(device, ci, a, sc);
    am_emit('N', r == VK_SUCCESS && sc ? (uint64_t)*sc : 0, AM_NA, ML_NONE, BUDGET_NO_HEAP, AM_NA, 0,
            "what=swapchain-create fmt=%d ext=%ux%u images=%u layers=%u usage=0x%x present=%d old=0x%llx result=%d", ci->imageFormat,
            ci->imageExtent.width, ci->imageExtent.height, ci->minImageCount, ci->imageArrayLayers, ci->imageUsage, ci->presentMode,
            (unsigned long long)ci->oldSwapchain, r);
    return r;
}
static VKAPI_ATTR void VKAPI_CALL compat_AmDestroySwapchainKHR(VkDevice device, VkSwapchainKHR sc, const VkAllocationCallbacks *a) {
    if (sc) am_emit('N', (uint64_t)sc, AM_NA, ML_NONE, BUDGET_NO_HEAP, AM_NA, 0, "what=swapchain-destroy");
    get_device_data(device)->amf->DestroySwapchainKHR(device, sc, a);
}
// bc-misc counters (indices into g_am_bcmisc_name)
static VKAPI_ATTR void VKAPI_CALL compat_AmCmdCopyImage(VkCommandBuffer cb, VkImage src, VkImageLayout sl, VkImage dst, VkImageLayout dl,
                                                        uint32_t n, const VkImageCopy *r) {
    am_bcmisc("CmdCopyImage", cb, (uint64_t)(uintptr_t)src, 0, (uint64_t)(uintptr_t)dst, 1, n);
    get_device_data_by_cmd(cb)->amf->CmdCopyImage(cb, src, sl, dst, dl, n, r);
}
static VKAPI_ATTR void VKAPI_CALL compat_AmCmdBlitImage(VkCommandBuffer cb, VkImage src, VkImageLayout sl, VkImage dst, VkImageLayout dl,
                                                        uint32_t n, const VkImageBlit *r, VkFilter f) {
    am_bcmisc("CmdBlitImage", cb, (uint64_t)(uintptr_t)src, 2, (uint64_t)(uintptr_t)dst, 3, n);
    get_device_data_by_cmd(cb)->amf->CmdBlitImage(cb, src, sl, dst, dl, n, r, f);
}
static VKAPI_ATTR void VKAPI_CALL compat_AmCmdCopyImageToBuffer(VkCommandBuffer cb, VkImage src, VkImageLayout sl, VkBuffer dst,
                                                                uint32_t n, const VkBufferImageCopy *r) {
    am_bcmisc("CmdCopyImageToBuffer", cb, (uint64_t)(uintptr_t)src, 4, (uint64_t)(uintptr_t)dst, -1, n);
    get_device_data_by_cmd(cb)->amf->CmdCopyImageToBuffer(cb, src, sl, dst, n, r);
}
static VKAPI_ATTR void VKAPI_CALL compat_AmCmdResolveImage(VkCommandBuffer cb, VkImage src, VkImageLayout sl, VkImage dst, VkImageLayout dl,
                                                           uint32_t n, const VkImageResolve *r) {
    am_bcmisc("CmdResolveImage", cb, (uint64_t)(uintptr_t)src, 5, (uint64_t)(uintptr_t)dst, 6, n);
    get_device_data_by_cmd(cb)->amf->CmdResolveImage(cb, src, sl, dst, dl, n, r);
}
static VKAPI_ATTR void VKAPI_CALL compat_AmCmdCopyImage2(VkCommandBuffer cb, const VkCopyImageInfo2 *i) {
    am_bcmisc("CmdCopyImage2", cb, (uint64_t)(uintptr_t)i->srcImage, 0, (uint64_t)(uintptr_t)i->dstImage, 1, i->regionCount);
    get_device_data_by_cmd(cb)->amf->CmdCopyImage2(cb, i);
}
static VKAPI_ATTR void VKAPI_CALL compat_AmCmdBlitImage2(VkCommandBuffer cb, const VkBlitImageInfo2 *i) {
    am_bcmisc("CmdBlitImage2", cb, (uint64_t)(uintptr_t)i->srcImage, 2, (uint64_t)(uintptr_t)i->dstImage, 3, i->regionCount);
    get_device_data_by_cmd(cb)->amf->CmdBlitImage2(cb, i);
}
static VKAPI_ATTR void VKAPI_CALL compat_AmCmdCopyImageToBuffer2(VkCommandBuffer cb, const VkCopyImageToBufferInfo2 *i) {
    am_bcmisc("CmdCopyImageToBuffer2", cb, (uint64_t)(uintptr_t)i->srcImage, 4, (uint64_t)(uintptr_t)i->dstBuffer, -1, i->regionCount);
    get_device_data_by_cmd(cb)->amf->CmdCopyImageToBuffer2(cb, i);
}
static VKAPI_ATTR void VKAPI_CALL compat_AmCmdResolveImage2(VkCommandBuffer cb, const VkResolveImageInfo2 *i) {
    am_bcmisc("CmdResolveImage2", cb, (uint64_t)(uintptr_t)i->srcImage, 5, (uint64_t)(uintptr_t)i->dstImage, 6, i->regionCount);
    get_device_data_by_cmd(cb)->amf->CmdResolveImage2(cb, i);
}
// vkGetDeviceProcAddr part (key set only): the hook when the driver answers the name and its pointer was loaded.
static PFN_vkVoidFunction am_proc(VkDevice device, const char *n) {
    #define AMF_OFF(fn) offsetof(struct AmFns, fn)
    static const struct { const char *name; PFN_vkVoidFunction fn; size_t off; } t[] = {
        {"vkBindBufferMemory", (PFN_vkVoidFunction)compat_AmBindBufferMemory, SIZE_MAX},
        {"vkBindBufferMemory2", (PFN_vkVoidFunction)compat_AmBindBufferMemory2, AMF_OFF(BindBufferMemory2)},
        {"vkBindBufferMemory2KHR", (PFN_vkVoidFunction)compat_AmBindBufferMemory2, AMF_OFF(BindBufferMemory2)},
        {"vkBindImageMemory", (PFN_vkVoidFunction)compat_AmBindImageMemory, AMF_OFF(BindImageMemory)},
        {"vkBindImageMemory2", (PFN_vkVoidFunction)compat_AmBindImageMemory2, AMF_OFF(BindImageMemory2)},
        {"vkBindImageMemory2KHR", (PFN_vkVoidFunction)compat_AmBindImageMemory2, AMF_OFF(BindImageMemory2)},
        {"vkDestroyBuffer", (PFN_vkVoidFunction)compat_AmDestroyBuffer, SIZE_MAX},
        {"vkCreateCommandPool", (PFN_vkVoidFunction)compat_AmCreateCommandPool, AMF_OFF(CreateCommandPool)},
        {"vkResetDescriptorPool", (PFN_vkVoidFunction)compat_AmResetDescriptorPool, AMF_OFF(ResetDescriptorPool)},
        {"vkDestroyDescriptorPool", (PFN_vkVoidFunction)compat_AmDestroyDescriptorPool, AMF_OFF(DestroyDescriptorPool)},
        {"vkCreateSwapchainKHR", (PFN_vkVoidFunction)compat_AmCreateSwapchainKHR, AMF_OFF(CreateSwapchainKHR)},
        {"vkDestroySwapchainKHR", (PFN_vkVoidFunction)compat_AmDestroySwapchainKHR, AMF_OFF(DestroySwapchainKHR)},
        {"vkCmdCopyImage", (PFN_vkVoidFunction)compat_AmCmdCopyImage, AMF_OFF(CmdCopyImage)},
        {"vkCmdBlitImage", (PFN_vkVoidFunction)compat_AmCmdBlitImage, AMF_OFF(CmdBlitImage)},
        {"vkCmdCopyImageToBuffer", (PFN_vkVoidFunction)compat_AmCmdCopyImageToBuffer, AMF_OFF(CmdCopyImageToBuffer)},
        {"vkCmdResolveImage", (PFN_vkVoidFunction)compat_AmCmdResolveImage, AMF_OFF(CmdResolveImage)},
        {"vkCmdCopyImage2", (PFN_vkVoidFunction)compat_AmCmdCopyImage2, AMF_OFF(CmdCopyImage2)},
        {"vkCmdCopyImage2KHR", (PFN_vkVoidFunction)compat_AmCmdCopyImage2, AMF_OFF(CmdCopyImage2)},
        {"vkCmdBlitImage2", (PFN_vkVoidFunction)compat_AmCmdBlitImage2, AMF_OFF(CmdBlitImage2)},
        {"vkCmdBlitImage2KHR", (PFN_vkVoidFunction)compat_AmCmdBlitImage2, AMF_OFF(CmdBlitImage2)},
        {"vkCmdCopyImageToBuffer2", (PFN_vkVoidFunction)compat_AmCmdCopyImageToBuffer2, AMF_OFF(CmdCopyImageToBuffer2)},
        {"vkCmdCopyImageToBuffer2KHR", (PFN_vkVoidFunction)compat_AmCmdCopyImageToBuffer2, AMF_OFF(CmdCopyImageToBuffer2)},
        {"vkCmdResolveImage2", (PFN_vkVoidFunction)compat_AmCmdResolveImage2, AMF_OFF(CmdResolveImage2)},
        {"vkCmdResolveImage2KHR", (PFN_vkVoidFunction)compat_AmCmdResolveImage2, AMF_OFF(CmdResolveImage2)},
    };
    #undef AMF_OFF
    for (size_t i = 0; i < sizeof t / sizeof t[0]; i++) {
        if (strcmp(n, t[i].name)) continue;
        struct DeviceData *dd = get_device_data(device);
        if (!dd || !dd->amf || !dd->get_device_proc_addr(device, n)) return NULL;
        if (t[i].off != SIZE_MAX && !*(PFN_vkVoidFunction *)((char *)dd->amf + t[i].off)) return NULL;
        return t[i].fn;
    }
    return NULL;
}

// 10-s tick (memlog flush thread): the clock pair line, and the bc-misc counters on stderr when they changed.
static void am_tick(void) {
    am_clock_line("tick");
    uint64_t v[9], sum = 0;
    for (int i = 0; i < 9; i++) sum += v[i] = atomic_load(&g_am_bcmisc[i]);
    if (sum == atomic_exchange(&g_am_bcmisc_logged, sum)) return; // counters only grow: same sum = unchanged
    char l[400];
    size_t o = 0;
    for (int i = 0; i < 9 && o < sizeof l; i++) o += (size_t)snprintf(l + o, sizeof l - o, " %s %llu", g_am_bcmisc_name[i], (unsigned long long)v[i]);
    fprintf(stderr, "mali-compat: addrmap: bc-misc%s (C events %llu, first 16 in the TSV)\n", l,
            (unsigned long long)atomic_load(&g_am_c_events));
}
static void am_end(const char *why) {
    pthread_mutex_lock(&g_ml_mutex);
    uint64_t bytes = g_ml_bytes;
    bool trunc = g_ml_stopped;
    pthread_mutex_unlock(&g_ml_mutex);
    char l[400];
    size_t o = 0;
    for (const char *e = "AMUFXSBDLRNC"; *e && o < sizeof l; e++)
        o += (size_t)snprintf(l + o, sizeof l - o, " %c %llu", *e, (unsigned long long)atomic_load(&g_am_count[*e - 'A']));
    char m[300];
    size_t k = 0;
    for (int i = 0; i < 9 && k < sizeof m; i++) k += (size_t)snprintf(m + k, sizeof m - k, " %s %llu", g_am_bcmisc_name[i], (unsigned long long)atomic_load(&g_am_bcmisc[i]));
    fprintf(stderr, "mali-compat: addrmap: end (%s): events%s, objects tracked %llu, file %llu KiB (cap %llu MiB)%s, bc-misc%s\n", why, l,
            (unsigned long long)atomic_load(&g_am_obj_live), (unsigned long long)(bytes >> 10), (unsigned long long)(g_ml_cap >> 20),
            trunc ? " TRUNCATED" : "", m);
}
static void am_atexit(void) { am_end("exit"); }


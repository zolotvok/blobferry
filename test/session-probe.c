// session-probe: can the session (gamescope, Steam's ANGLE) run on the bridge instead of PanVK? Lists what gamescope
// 3.16 asks of its Vulkan device (src/rendervulkan.cpp) and of the instance, and what the driver offers: device and
// instance extensions, VK_EXT_physical_device_drm's nodes (emulated by the compat layer with MALI_COMPAT_DRM_NODE),
// timeline semaphores, the queue families, and the DRM format modifiers of the formats gamescope composites and
// imports. Run it like a session process:
//   VK_DRIVER_FILES=/usr/lib/mali_bridge/mali_bridge_icd.json HYBRIS_USE_VENDOR_NAMESPACE=1 ENABLE_MALI_COMPAT=1
//   ENABLE_MALI_WSI=1 MALI_COMPAT_DRM_NODE=/dev/mali0 /usr/lib/mali_bridge/session-probe
// [Claude Opus 5.5] 2026-10-30
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vulkan/vulkan.h>

static int has(const VkExtensionProperties *e, uint32_t n, const char *name) {
    for (uint32_t i = 0; i < n; i++) if (!strcmp(e[i].extensionName, name)) return 1;
    return 0;
}

int main(void) {
    uint32_t n = 0;
    vkEnumerateInstanceExtensionProperties(NULL, &n, NULL);
    VkExtensionProperties *ie = calloc(n, sizeof *ie);
    vkEnumerateInstanceExtensionProperties(NULL, &n, ie);
    const char *inst_want[] = {"VK_KHR_surface", "VK_KHR_wayland_surface", "VK_KHR_xcb_surface", "VK_KHR_xlib_surface",
                               "VK_KHR_get_surface_capabilities2", "VK_EXT_surface_maintenance1",
                               "VK_KHR_external_memory_capabilities", "VK_KHR_external_semaphore_capabilities"};
    printf("instance extensions (%u):\n", n);
    for (unsigned i = 0; i < sizeof inst_want / sizeof *inst_want; i++)
        printf("  %-45s %s\n", inst_want[i], has(ie, n, inst_want[i]) ? "yes" : "NO");

    const char *ext[] = {"VK_KHR_surface", "VK_KHR_wayland_surface"};
    uint32_t ext_n = 0;
    for (unsigned i = 0; i < 2; i++) if (has(ie, n, ext[i])) ext_n++;
    VkApplicationInfo app = {VK_STRUCTURE_TYPE_APPLICATION_INFO, NULL, "session-probe", 1, NULL, 0, VK_API_VERSION_1_3};
    VkInstanceCreateInfo ici = {VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, NULL, 0, &app, 0, NULL, ext_n == 2 ? 2 : 0, ext};
    VkInstance inst;
    VkResult r = vkCreateInstance(&ici, NULL, &inst);
    if (r) { printf("vkCreateInstance: %d\n", r); return 1; }
    uint32_t pdn = 1;
    VkPhysicalDevice pd;
    vkEnumeratePhysicalDevices(inst, &pdn, &pd);
    if (!pdn) { printf("no physical device\n"); return 1; }

    vkEnumerateDeviceExtensionProperties(pd, NULL, &n, NULL);
    VkExtensionProperties *de = calloc(n, sizeof *de);
    vkEnumerateDeviceExtensionProperties(pd, NULL, &n, de);
    const char *dev_want[] = {"VK_KHR_swapchain", "VK_EXT_image_drm_format_modifier", "VK_EXT_physical_device_drm",
                              "VK_EXT_queue_family_foreign", "VK_KHR_external_memory_fd", "VK_EXT_external_memory_dma_buf",
                              "VK_KHR_external_semaphore_fd", "VK_EXT_robustness2", "VK_KHR_maintenance5",
                              "VK_KHR_present_id", "VK_KHR_present_wait", "VK_KHR_swapchain_mutable_format",
                              "VK_EXT_hdr_metadata", "VK_KHR_timeline_semaphore", "VK_KHR_image_format_list",
                              "VK_KHR_external_fence_fd", "VK_ANDROID_external_memory_android_hardware_buffer"};
    printf("device extensions (%u):\n", n);
    for (unsigned i = 0; i < sizeof dev_want / sizeof *dev_want; i++)
        printf("  %-45s %s\n", dev_want[i], has(de, n, dev_want[i]) ? "yes" : "NO");

    VkPhysicalDeviceDrmPropertiesEXT drm = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRM_PROPERTIES_EXT};
    VkPhysicalDeviceProperties2 p2 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, &drm};
    vkGetPhysicalDeviceProperties2(pd, &p2);
    printf("device %s, api %u.%u.%u\n", p2.properties.deviceName, VK_API_VERSION_MAJOR(p2.properties.apiVersion),
           VK_API_VERSION_MINOR(p2.properties.apiVersion), VK_API_VERSION_PATCH(p2.properties.apiVersion));
    printf("drm nodes: primary %d (%lld:%lld) render %d (%lld:%lld)\n", drm.hasPrimary, (long long)drm.primaryMajor,
           (long long)drm.primaryMinor, drm.hasRender, (long long)drm.renderMajor, (long long)drm.renderMinor);

    VkPhysicalDeviceVulkan12Features f12 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
    VkPhysicalDeviceVulkan13Features f13 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES, &f12};
    VkPhysicalDeviceFeatures2 f2 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, &f13};
    vkGetPhysicalDeviceFeatures2(pd, &f2);
    printf("features: timelineSemaphore %d, dynamicRendering %d, synchronization2 %d, shaderInt16 %d, shaderFloat16 %d\n",
           f12.timelineSemaphore, f13.dynamicRendering, f13.synchronization2, f2.features.shaderInt16, f12.shaderFloat16);

    vkGetPhysicalDeviceQueueFamilyProperties(pd, &n, NULL);
    VkQueueFamilyProperties *q = calloc(n, sizeof *q);
    vkGetPhysicalDeviceQueueFamilyProperties(pd, &n, q);
    for (uint32_t i = 0; i < n; i++) printf("queue family %u: flags 0x%x count %u\n", i, q[i].queueFlags, q[i].queueCount);

    // the formats gamescope composites into and imports from clients (wl_drm/dmabuf ARGB/XRGB 8888, 2101010, FP16)
    VkFormat fmts[] = {VK_FORMAT_B8G8R8A8_UNORM, VK_FORMAT_R8G8B8A8_UNORM, VK_FORMAT_B8G8R8A8_SRGB,
                       VK_FORMAT_A2B10G10R10_UNORM_PACK32, VK_FORMAT_A2R10G10B10_UNORM_PACK32, VK_FORMAT_R16G16B16A16_SFLOAT,
                       VK_FORMAT_R8_UNORM, VK_FORMAT_R8G8_UNORM, VK_FORMAT_G8_B8R8_2PLANE_420_UNORM};
    const char *names[] = {"B8G8R8A8_UNORM", "R8G8B8A8_UNORM", "B8G8R8A8_SRGB", "A2B10G10R10", "A2R10G10B10",
                           "R16G16B16A16_SFLOAT", "R8_UNORM", "R8G8_UNORM", "NV12"};
    for (unsigned f = 0; f < sizeof fmts / sizeof *fmts; f++) {
        VkDrmFormatModifierPropertiesListEXT ml = {VK_STRUCTURE_TYPE_DRM_FORMAT_MODIFIER_PROPERTIES_LIST_EXT};
        VkFormatProperties2 fp = {VK_STRUCTURE_TYPE_FORMAT_PROPERTIES_2, &ml};
        vkGetPhysicalDeviceFormatProperties2(pd, fmts[f], &fp);
        VkDrmFormatModifierPropertiesEXT mods[32];
        ml.drmFormatModifierCount = ml.drmFormatModifierCount > 32 ? 32 : ml.drmFormatModifierCount;
        ml.pDrmFormatModifierProperties = mods;
        vkGetPhysicalDeviceFormatProperties2(pd, fmts[f], &fp);
        printf("%-20s optimal 0x%x, %u modifiers:", names[f], fp.formatProperties.optimalTilingFeatures, ml.drmFormatModifierCount);
        for (uint32_t m = 0; m < ml.drmFormatModifierCount; m++)
            printf(" %llx/%u/0x%x", (unsigned long long)mods[m].drmFormatModifier, mods[m].drmFormatModifierPlaneCount,
                   mods[m].drmFormatModifierTilingFeatures);
        printf("\n");
    }
    // gamescope's flippable images (output and flat textures): exportable as dma-buf with a modifier the parent
    // compositor takes (getModifierProps in rendervulkan.cpp). Which modifiers does the driver let it export?
    struct { const char *name; VkImageUsageFlags usage; } uses[] = {
        {"flat (sampled|dst)", VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT},
        {"output (storage|src|sampled)", VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_SAMPLED_BIT},
        {"import (sampled)", VK_IMAGE_USAGE_SAMPLED_BIT}};
    uint64_t try_mods[] = {0, 0x0800000000000041ull, 0x0800000000000061ull, 0x0820000000000001ull};
    for (unsigned u = 0; u < sizeof uses / sizeof *uses; u++)
        for (unsigned m = 0; m < sizeof try_mods / sizeof *try_mods; m++) {
            VkPhysicalDeviceImageDrmFormatModifierInfoEXT mi = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_IMAGE_DRM_FORMAT_MODIFIER_INFO_EXT,
                                                                NULL, try_mods[m], VK_SHARING_MODE_EXCLUSIVE};
            VkPhysicalDeviceExternalImageFormatInfo ei = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_IMAGE_FORMAT_INFO, &mi,
                                                          VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT};
            VkPhysicalDeviceImageFormatInfo2 fi = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_IMAGE_FORMAT_INFO_2, &ei,
                                                   VK_FORMAT_B8G8R8A8_UNORM, VK_IMAGE_TYPE_2D,
                                                   VK_IMAGE_TILING_DRM_FORMAT_MODIFIER_EXT, uses[u].usage, 0};
            VkExternalImageFormatProperties ep = {VK_STRUCTURE_TYPE_EXTERNAL_IMAGE_FORMAT_PROPERTIES};
            VkImageFormatProperties2 ip = {VK_STRUCTURE_TYPE_IMAGE_FORMAT_PROPERTIES_2, &ep};
            VkResult rr = vkGetPhysicalDeviceImageFormatProperties2(pd, &fi, &ip);
            printf("B8G8R8A8 %-30s modifier %016llx: %d, external features 0x%x, export-from 0x%x\n", uses[u].name,
                   (unsigned long long)try_mods[m], rr, ep.externalMemoryProperties.externalMemoryFeatures,
                   ep.externalMemoryProperties.exportFromImportedHandleTypes);
        }
    vkDestroyInstance(inst, NULL);
    return 0;
}

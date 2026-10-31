// SPDX-License-Identifier: GPL-3.0-or-later
// img-probe: which dma-buf swapchain-style VkImage create infos the vendor Mali driver accepts. [Claude Opus 5.5] 2026-10-26
// Run through the loader with only the bridge ICD (no layers):
//   VK_DRIVER_FILES=<bridge icd.json> HYBRIS_USE_VENDOR_NAMESPACE=1 img-probe
// Prints the B8G8R8A8_UNORM DRM modifiers, then one line per create info: result of
// vkGetPhysicalDeviceImageFormatProperties2 (fp) and vkCreateImage (ci).
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vulkan/vulkan.h>

#define FMT VK_FORMAT_B8G8R8A8_UNORM
#define DRM_FORMAT_MOD_LINEAR 0ull

static VkPhysicalDevice pd;
static VkDevice dev;
static uint64_t mods[64];
static uint32_t mod_count;

static void list_modifiers(void) {
    VkDrmFormatModifierPropertiesListEXT list = {VK_STRUCTURE_TYPE_DRM_FORMAT_MODIFIER_PROPERTIES_LIST_EXT};
    VkFormatProperties2 fp = {VK_STRUCTURE_TYPE_FORMAT_PROPERTIES_2, &list};
    vkGetPhysicalDeviceFormatProperties2(pd, FMT, &fp);
    VkDrmFormatModifierPropertiesEXT props[64];
    if (list.drmFormatModifierCount > 64) list.drmFormatModifierCount = 64;
    list.pDrmFormatModifierProperties = props;
    vkGetPhysicalDeviceFormatProperties2(pd, FMT, &fp);
    printf("optimal features 0x%x linear features 0x%x\n", fp.formatProperties.optimalTilingFeatures,
           fp.formatProperties.linearTilingFeatures);
    printf("%u DRM modifiers for B8G8R8A8_UNORM:\n", list.drmFormatModifierCount);
    for (uint32_t i = 0; i < list.drmFormatModifierCount; i++) {
        printf("  0x%016llx planes %u features 0x%x\n", (unsigned long long)props[i].drmFormatModifier,
               props[i].drmFormatModifierPlaneCount, props[i].drmFormatModifierTilingFeatures);
        mods[mod_count++] = props[i].drmFormatModifier;
    }
}

enum { T_OPT, T_OPT_EXT, T_LIN_EXT, T_DRM_EXPL64, T_DRM_EXPL256, T_DRM_LIST_LINEAR, T_DRM_LIST_ALL, T_COUNT };
static const char *tname[] = {"optimal", "optimal+dmabuf", "linear+dmabuf", "drm explicit linear pitch64",
                              "drm explicit linear pitch256", "drm list {linear}", "drm list {all}"};

static void probe(uint32_t w, uint32_t h, int t, int mutable_fmt, VkImageUsageFlags usage) {
    VkFormat views[2] = {VK_FORMAT_B8G8R8A8_UNORM, VK_FORMAT_B8G8R8A8_SRGB};
    VkImageFormatListCreateInfo fl = {VK_STRUCTURE_TYPE_IMAGE_FORMAT_LIST_CREATE_INFO, NULL, 2, views};
    VkExternalMemoryImageCreateInfo ext = {VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO, NULL,
                                           VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT};
    uint32_t align = t == T_DRM_EXPL256 ? 256 : 64;
    VkSubresourceLayout plane = {0, 0, (w * 4 + align - 1) & ~(align - 1), 0, 0}; // size must be 0 (VUID-02267)
    VkImageDrmFormatModifierExplicitCreateInfoEXT expl = {VK_STRUCTURE_TYPE_IMAGE_DRM_FORMAT_MODIFIER_EXPLICIT_CREATE_INFO_EXT,
                                                          NULL, DRM_FORMAT_MOD_LINEAR, 1, &plane};
    uint64_t linear = DRM_FORMAT_MOD_LINEAR;
    VkImageDrmFormatModifierListCreateInfoEXT lst = {VK_STRUCTURE_TYPE_IMAGE_DRM_FORMAT_MODIFIER_LIST_CREATE_INFO_EXT, NULL,
                                                     t == T_DRM_LIST_ALL ? mod_count : 1,
                                                     t == T_DRM_LIST_ALL ? mods : &linear};
    VkImageCreateInfo ci = {VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    ci.imageType = VK_IMAGE_TYPE_2D;
    ci.format = FMT;
    ci.extent = (VkExtent3D){w, h, 1};
    ci.mipLevels = 1;
    ci.arrayLayers = 1;
    ci.samples = VK_SAMPLE_COUNT_1_BIT;
    ci.usage = usage;
    ci.tiling = t == T_OPT || t == T_OPT_EXT ? VK_IMAGE_TILING_OPTIMAL
              : t == T_LIN_EXT ? VK_IMAGE_TILING_LINEAR : VK_IMAGE_TILING_DRM_FORMAT_MODIFIER_EXT;
    const void *chain = NULL;
    if (t == T_DRM_EXPL64 || t == T_DRM_EXPL256) { expl.pNext = chain; chain = &expl; }
    if (t == T_DRM_LIST_LINEAR || t == T_DRM_LIST_ALL) { lst.pNext = chain; chain = &lst; }
    if (t != T_OPT) { ext.pNext = chain; chain = &ext; }
    if (mutable_fmt) {
        ci.flags = VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT | VK_IMAGE_CREATE_EXTENDED_USAGE_BIT;
        fl.pNext = chain;
        chain = &fl;
    }
    ci.pNext = chain;

    // Format query for the same combination (the modifier info only for the DRM tilings).
    VkPhysicalDeviceExternalImageFormatInfo efi = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_IMAGE_FORMAT_INFO, NULL,
                                                   VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT};
    VkPhysicalDeviceImageDrmFormatModifierInfoEXT mi = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_IMAGE_DRM_FORMAT_MODIFIER_INFO_EXT,
                                                        NULL, DRM_FORMAT_MOD_LINEAR, VK_SHARING_MODE_EXCLUSIVE, 0, NULL};
    VkImageFormatListCreateInfo qfl = fl;
    const void *qchain = NULL;
    if (ci.tiling == VK_IMAGE_TILING_DRM_FORMAT_MODIFIER_EXT) { mi.pNext = qchain; qchain = &mi; }
    if (t != T_OPT) { efi.pNext = qchain; qchain = &efi; }
    if (mutable_fmt) { qfl.pNext = qchain; qchain = &qfl; }
    VkPhysicalDeviceImageFormatInfo2 qi = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_IMAGE_FORMAT_INFO_2, qchain, FMT,
                                           VK_IMAGE_TYPE_2D, ci.tiling, usage, ci.flags};
    VkImageFormatProperties2 qp = {VK_STRUCTURE_TYPE_IMAGE_FORMAT_PROPERTIES_2};
    VkResult fr = vkGetPhysicalDeviceImageFormatProperties2(pd, &qi, &qp);

    VkImage img = VK_NULL_HANDLE;
    VkResult r = vkCreateImage(dev, &ci, NULL, &img);
    printf("%4ux%-4u %-29s %-7s usage 0x%02x  fp %11d  ci %11d\n", w, h, tname[t], mutable_fmt ? "mutable" : "-",
           usage, fr, r);
    if (r == VK_SUCCESS && ci.tiling == VK_IMAGE_TILING_DRM_FORMAT_MODIFIER_EXT && !mutable_fmt && usage == 0x13) {
        // The layout the driver picks by itself, to compare with what we pass explicitly.
        VkImageDrmFormatModifierPropertiesEXT mp = {VK_STRUCTURE_TYPE_IMAGE_DRM_FORMAT_MODIFIER_PROPERTIES_EXT};
        PFN_vkGetImageDrmFormatModifierPropertiesEXT get_mod =
            (PFN_vkGetImageDrmFormatModifierPropertiesEXT)vkGetDeviceProcAddr(dev, "vkGetImageDrmFormatModifierPropertiesEXT");
        VkResult mr = get_mod ? get_mod(dev, img, &mp) : VK_ERROR_EXTENSION_NOT_PRESENT;
        VkImageSubresource sub = {VK_IMAGE_ASPECT_MEMORY_PLANE_0_BIT_EXT, 0, 0};
        VkSubresourceLayout l = {0};
        vkGetImageSubresourceLayout(dev, img, &sub, &l);
        VkMemoryRequirements req;
        vkGetImageMemoryRequirements(dev, img, &req);
        printf("          -> modifier %d:0x%016llx offset %llu size %llu rowPitch %llu; memreq size %llu align %llu\n", mr,
               (unsigned long long)mp.drmFormatModifier, (unsigned long long)l.offset, (unsigned long long)l.size,
               (unsigned long long)l.rowPitch, (unsigned long long)req.size, (unsigned long long)req.alignment);
    }
    if (r == VK_SUCCESS) vkDestroyImage(dev, img, NULL);
}

// Wine on WoW64 without VK_EXT_map_memory_placed creates every image with HOST_ALLOCATION external memory and backs
// host-visible allocations with VkImportMemoryHostPointerInfoEXT. Can an image be bound to such memory? [Claude Opus 5.5]
#include <sys/mman.h>
static void host_import(int with_ext, VkImageTiling tiling) {
    VkExternalMemoryImageCreateInfo ext = {VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO, NULL,
                                           VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT};
    VkImageCreateInfo ci = {VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, with_ext ? &ext : NULL, 0, VK_IMAGE_TYPE_2D, FMT,
                            {640, 480, 1}, 1, 1, VK_SAMPLE_COUNT_1_BIT, tiling, 0x17, VK_SHARING_MODE_EXCLUSIVE};
    VkImage img;
    VkResult r = vkCreateImage(dev, &ci, NULL, &img);
    printf("host-import: image %s tiling %d -> create %d", with_ext ? "with HOST_ALLOCATION" : "plain", tiling, r);
    if (r != VK_SUCCESS) { printf("\n"); return; }
    VkMemoryRequirements req;
    vkGetImageMemoryRequirements(dev, img, &req);
    PFN_vkGetMemoryHostPointerPropertiesEXT get_hp =
        (PFN_vkGetMemoryHostPointerPropertiesEXT)vkGetDeviceProcAddr(dev, "vkGetMemoryHostPointerPropertiesEXT");
    size_t size = (req.size + 0xffff) & ~(size_t)0xffff;
    void *ptr = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    VkMemoryHostPointerPropertiesEXT hpp = {VK_STRUCTURE_TYPE_MEMORY_HOST_POINTER_PROPERTIES_EXT};
    VkResult hr = get_hp ? get_hp(dev, VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT, ptr, &hpp) : -1;
    uint32_t bits = req.memoryTypeBits & hpp.memoryTypeBits;
    printf(", size %llu align %llu, image types 0x%x host-ptr types 0x%x (%d)", (unsigned long long)req.size,
           (unsigned long long)req.alignment, req.memoryTypeBits, hpp.memoryTypeBits, hr);
    if (!bits) { printf(", no common memory type\n"); vkDestroyImage(dev, img, NULL); return; }
    VkImportMemoryHostPointerInfoEXT imp = {VK_STRUCTURE_TYPE_IMPORT_MEMORY_HOST_POINTER_INFO_EXT, NULL,
                                            VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT, ptr};
    VkMemoryAllocateInfo ai = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, &imp, size, (uint32_t)__builtin_ctz(bits)};
    VkDeviceMemory mem;
    r = vkAllocateMemory(dev, &ai, NULL, &mem);
    printf(", import %d", r);
    if (r == VK_SUCCESS) {
        r = vkBindImageMemory(dev, img, mem, 0);
        printf(", bind %d", r);
        if (r == VK_SUCCESS) {
            // Clear the image on the GPU and look for the colour in host memory (linear) or just for completion.
            VkCommandPool pool;
            VkCommandPoolCreateInfo pci = {VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO, NULL, 0, 0};
            vkCreateCommandPool(dev, &pci, NULL, &pool);
            VkCommandBuffer cb;
            VkCommandBufferAllocateInfo cbi = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, NULL, pool, VK_COMMAND_BUFFER_LEVEL_PRIMARY, 1};
            vkAllocateCommandBuffers(dev, &cbi, &cb);
            VkCommandBufferBeginInfo bi = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
            vkBeginCommandBuffer(cb, &bi);
            VkImageSubresourceRange rng = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
            VkImageMemoryBarrier b = {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER, NULL, 0, VK_ACCESS_TRANSFER_WRITE_BIT,
                                      VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 0, 0, img, rng};
            vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, NULL, 0, NULL, 1, &b);
            VkClearColorValue c = {.float32 = {1, 0, 0, 1}};
            vkCmdClearColorImage(cb, img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &c, 1, &rng);
            vkEndCommandBuffer(cb);
            VkQueue q;
            vkGetDeviceQueue(dev, 0, 0, &q);
            VkSubmitInfo si = {VK_STRUCTURE_TYPE_SUBMIT_INFO, NULL, 0, NULL, NULL, 1, &cb};
            r = vkQueueSubmit(q, 1, &si, VK_NULL_HANDLE);
            VkResult w = vkQueueWaitIdle(q);
            uint32_t nonzero = 0;
            for (size_t i = 0; i < req.size / 4; i++) nonzero += ((uint32_t *)ptr)[i] != 0;
            printf(", clear submit %d wait %d, %u nonzero words in host memory", r, w, nonzero);
            vkDestroyCommandPool(dev, pool, NULL);
        }
        vkFreeMemory(dev, mem, NULL);
    }
    printf("\n");
    vkDestroyImage(dev, img, NULL);
    munmap(ptr, size);
}

int main(void) {
    setvbuf(stdout, NULL, _IONBF, 0);
    VkApplicationInfo app = {VK_STRUCTURE_TYPE_APPLICATION_INFO, NULL, "img-probe", 1, NULL, 0, VK_API_VERSION_1_3};
    VkInstanceCreateInfo ici = {VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, NULL, 0, &app};
    VkInstance inst;
    if (vkCreateInstance(&ici, NULL, &inst) != VK_SUCCESS) { printf("vkCreateInstance failed\n"); return 1; }
    uint32_t n = 1;
    if (vkEnumeratePhysicalDevices(inst, &n, &pd) < 0 || !n) { printf("no physical device\n"); return 1; }
    VkPhysicalDeviceProperties p;
    vkGetPhysicalDeviceProperties(pd, &p);
    printf("device: %s\n", p.deviceName);
    const char *exts[] = {"VK_EXT_image_drm_format_modifier", "VK_KHR_external_memory_fd", "VK_EXT_external_memory_dma_buf",
                          "VK_EXT_external_memory_host"};
    float prio = 1;
    VkDeviceQueueCreateInfo q = {VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO, NULL, 0, 0, 1, &prio};
    VkDeviceCreateInfo dci = {VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, NULL, 0, 1, &q, 0, NULL, 4, exts};
    VkResult r = vkCreateDevice(pd, &dci, NULL, &dev);
    if (r != VK_SUCCESS) { printf("vkCreateDevice -> %d\n", r); return 1; }
    list_modifiers();
    for (int e = 1; e >= 0; e--)
        for (int t = 0; t < 2; t++) host_import(e, t ? VK_IMAGE_TILING_LINEAR : VK_IMAGE_TILING_OPTIMAL);
    const uint32_t sizes[][2] = {{1, 1}, {640, 480}, {1600, 720}};
    const VkImageUsageFlags usages[] = {0x97, 0x13};
    for (int s = 0; s < 3; s++)
        for (int t = 0; t < T_COUNT; t++)
            for (int m = 0; m < 2; m++)
                for (int u = 0; u < 2; u++)
                    probe(sizes[s][0], sizes[s][1], t, m, usages[u]);
    vkDestroyDevice(dev, NULL);
    vkDestroyInstance(inst, NULL);
    return 0;
}

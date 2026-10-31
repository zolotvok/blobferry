// SPDX-License-Identifier: GPL-3.0-or-later
// rt-comp-probe: does the vendor driver compress The Witcher 3's render targets, and what turns it off?
// Witcher 3's colour targets come from DXVK with usage 0x1f (transfer src/dst, sampled, STORAGE, colour attachment) and
// all report VK_IMAGE_COMPRESSION_DISABLED (2026-10-28 image log). For each format and usage this prints what the
// driver offers (query with FIXED_RATE_DEFAULT) and what a 1600x720 image gets with no control, with DEFAULT and with
// FIXED_RATE_DEFAULT: create result, applied compression/rate flags, memory size. [Claude Opus 5.5] 2026-10-28
// Run on the ICD alone: VK_DRIVER_FILES=<bridge icd.json> HYBRIS_USE_VENDOR_NAMESPACE=1 VK_LOADER_LAYERS_DISABLE=~implicit~
#include <stdio.h>
#include <vulkan/vulkan.h>

static PFN_vkGetImageSubresourceLayout2KHR layout2;

static void make(VkDevice dev, VkFormat fmt, VkImageUsageFlags usage, VkImageCreateFlags flags, const char *name,
                 VkImageCompressionFlagsEXT ctl, int with_ctl) {
    VkImageCompressionControlEXT cc = {VK_STRUCTURE_TYPE_IMAGE_COMPRESSION_CONTROL_EXT, NULL, ctl};
    VkImageCreateInfo ci = {VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, with_ctl ? &cc : NULL, flags, VK_IMAGE_TYPE_2D, fmt,
                            {1600, 720, 1}, 1, 1, VK_SAMPLE_COUNT_1_BIT, VK_IMAGE_TILING_OPTIMAL, usage};
    VkImage img;
    VkResult cr = vkCreateImage(dev, &ci, NULL, &img);
    VkMemoryRequirements req = {0};
    VkImageCompressionPropertiesEXT got = {VK_STRUCTURE_TYPE_IMAGE_COMPRESSION_PROPERTIES_EXT};
    if (cr == VK_SUCCESS) {
        vkGetImageMemoryRequirements(dev, img, &req);
        VkImageSubresource2KHR sub = {VK_STRUCTURE_TYPE_IMAGE_SUBRESOURCE_2_KHR, NULL, {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0}};
        VkSubresourceLayout2KHR lay = {VK_STRUCTURE_TYPE_SUBRESOURCE_LAYOUT_2_KHR, &got};
        layout2(dev, img, &sub, &lay);
        vkDestroyImage(dev, img, NULL);
    }
    printf("    %-14s create %d, %5llu KiB, applied 0x%x rate 0x%x\n", name, cr, (unsigned long long)req.size / 1024,
           got.imageCompressionFlags, got.imageCompressionFixedRateFlags);
}

static void probe(VkPhysicalDevice pd, VkDevice dev, VkFormat fmt, VkImageUsageFlags usage, VkImageCreateFlags flags) {
    VkImageCompressionControlEXT cc = {VK_STRUCTURE_TYPE_IMAGE_COMPRESSION_CONTROL_EXT, NULL, VK_IMAGE_COMPRESSION_FIXED_RATE_DEFAULT_EXT};
    VkPhysicalDeviceImageFormatInfo2 qi = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_IMAGE_FORMAT_INFO_2, &cc, fmt, VK_IMAGE_TYPE_2D,
                                           VK_IMAGE_TILING_OPTIMAL, usage, flags};
    VkImageCompressionPropertiesEXT cp = {VK_STRUCTURE_TYPE_IMAGE_COMPRESSION_PROPERTIES_EXT};
    VkImageFormatProperties2 qp = {VK_STRUCTURE_TYPE_IMAGE_FORMAT_PROPERTIES_2, &cp};
    VkResult r = vkGetPhysicalDeviceImageFormatProperties2(pd, &qi, &qp);
    printf("format %3d usage 0x%02x flags 0x%x: query %d, offers compression 0x%x, fixed-rate 0x%x\n", fmt, usage, flags, r,
           cp.imageCompressionFlags, cp.imageCompressionFixedRateFlags);
    if (r != VK_SUCCESS) return;
    make(dev, fmt, usage, flags, "no control", 0, 0);
    make(dev, fmt, usage, flags, "DEFAULT", VK_IMAGE_COMPRESSION_DEFAULT_EXT, 1);
    make(dev, fmt, usage, flags, "FIXED_RATE", VK_IMAGE_COMPRESSION_FIXED_RATE_DEFAULT_EXT, 1);
}

int main(void) {
    setvbuf(stdout, NULL, _IONBF, 0);
    VkApplicationInfo app = {VK_STRUCTURE_TYPE_APPLICATION_INFO, NULL, "rt-comp-probe", 1, NULL, 0, VK_API_VERSION_1_3};
    VkInstanceCreateInfo ici = {VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, NULL, 0, &app};
    VkInstance inst;
    if (vkCreateInstance(&ici, NULL, &inst) != VK_SUCCESS) return 1;
    uint32_t n = 1;
    VkPhysicalDevice pd;
    vkEnumeratePhysicalDevices(inst, &n, &pd);
    VkPhysicalDeviceImageCompressionControlFeaturesEXT icf = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_IMAGE_COMPRESSION_CONTROL_FEATURES_EXT, NULL, VK_TRUE};
    const char *exts[] = {"VK_EXT_image_compression_control"};
    float prio = 1;
    VkDeviceQueueCreateInfo q = {VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO, NULL, 0, 0, 1, &prio};
    VkDeviceCreateInfo dci = {VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, &icf, 0, 1, &q, 0, NULL, 1, exts};
    VkDevice dev;
    VkResult r = vkCreateDevice(pd, &dci, NULL, &dev);
    printf("vkCreateDevice -> %d\n", r);
    if (r) return 1;
    layout2 = (PFN_vkGetImageSubresourceLayout2KHR)vkGetDeviceProcAddr(dev, "vkGetImageSubresourceLayout2EXT");
    if (!layout2) layout2 = (PFN_vkGetImageSubresourceLayout2KHR)vkGetDeviceProcAddr(dev, "vkGetImageSubresourceLayout2KHR");
    // Witcher 3's 1600x720 colour targets: RGBA8, B10G11R11, RGBA16F, RG16F, R16F, R32F, A2B10G10R10
    static const VkFormat fmts[] = {VK_FORMAT_R8G8B8A8_UNORM, VK_FORMAT_B10G11R11_UFLOAT_PACK32, VK_FORMAT_R16G16B16A16_SFLOAT,
                                    VK_FORMAT_R16G16_SFLOAT, VK_FORMAT_R16_SFLOAT, VK_FORMAT_R32_SFLOAT,
                                    VK_FORMAT_A2B10G10R10_UNORM_PACK32};
    const VkImageUsageFlags rt = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
                                 VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
    for (unsigned i = 0; i < sizeof fmts / sizeof fmts[0]; i++) {
        probe(pd, dev, fmts[i], rt, 0);                                       // 0x17: what DXVK asks without UAV
        probe(pd, dev, fmts[i], rt | VK_IMAGE_USAGE_STORAGE_BIT, 0);          // 0x1f: Witcher 3's targets
        probe(pd, dev, fmts[i], rt | VK_IMAGE_USAGE_STORAGE_BIT, VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT); // 0x1f + flags 0x8
    }
    return 0;
}

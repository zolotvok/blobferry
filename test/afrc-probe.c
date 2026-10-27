// afrc-probe: which fixed-rate (AFRC) compression rates the vendor driver offers and applies for the BC shadow
// formats (RGBA8 UNORM/SRGB, RG16, RGBA16F) with the shadows' usage (sampled + transfer). [Claude Opus 5.5] 2026-10-26
// Run on the ICD alone: VK_DRIVER_FILES=<bridge icd.json> HYBRIS_USE_VENDOR_NAMESPACE=1 VK_LOADER_LAYERS_DISABLE=~implicit~
#include <stdio.h>
#include <vulkan/vulkan.h>

static void probe(VkPhysicalDevice pd, VkDevice dev, PFN_vkGetImageSubresourceLayout2KHR layout2, VkFormat fmt) {
    VkImageCompressionControlEXT cc = {VK_STRUCTURE_TYPE_IMAGE_COMPRESSION_CONTROL_EXT, NULL, VK_IMAGE_COMPRESSION_FIXED_RATE_DEFAULT_EXT};
    VkPhysicalDeviceImageFormatInfo2 qi = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_IMAGE_FORMAT_INFO_2, &cc, fmt, VK_IMAGE_TYPE_2D,
                                           VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT |
                                           VK_IMAGE_USAGE_TRANSFER_SRC_BIT, 0};
    VkImageCompressionPropertiesEXT cp = {VK_STRUCTURE_TYPE_IMAGE_COMPRESSION_PROPERTIES_EXT};
    VkImageFormatProperties2 qp = {VK_STRUCTURE_TYPE_IMAGE_FORMAT_PROPERTIES_2, &cp};
    VkResult r = vkGetPhysicalDeviceImageFormatProperties2(pd, &qi, &qp);
    printf("format %d: query %d, compression 0x%x, fixed-rate flags 0x%x\n", fmt, r, cp.imageCompressionFlags,
           cp.imageCompressionFixedRateFlags);
    for (uint32_t bit = 0; bit < 24; bit++) {
        VkImageCompressionFixedRateFlagsEXT rate = 1u << bit;
        if (!(cp.imageCompressionFixedRateFlags & rate)) continue;
        VkImageCompressionControlEXT ex = {VK_STRUCTURE_TYPE_IMAGE_COMPRESSION_CONTROL_EXT, NULL,
                                           VK_IMAGE_COMPRESSION_FIXED_RATE_EXPLICIT_EXT, 1, &rate};
        VkImageCreateInfo ci = {VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, &ex, 0, VK_IMAGE_TYPE_2D, fmt, {1024, 1024, 1}, 11, 1,
                                VK_SAMPLE_COUNT_1_BIT, VK_IMAGE_TILING_OPTIMAL, qi.usage};
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
        printf("  rate bit %2u (%2u bpc): create %d, 1024^2 with mips needs %llu KiB, applied 0x%x / rate 0x%x\n", bit, bit + 1,
               cr, (unsigned long long)req.size / 1024, got.imageCompressionFlags, got.imageCompressionFixedRateFlags);
    }
    VkImageCreateInfo ci = {VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, NULL, 0, VK_IMAGE_TYPE_2D, fmt, {1024, 1024, 1}, 11, 1,
                            VK_SAMPLE_COUNT_1_BIT, VK_IMAGE_TILING_OPTIMAL, qi.usage};
    VkImage img;
    if (vkCreateImage(dev, &ci, NULL, &img) == VK_SUCCESS) {
        VkMemoryRequirements req;
        vkGetImageMemoryRequirements(dev, img, &req);
        printf("  default (no control): 1024^2 with mips needs %llu KiB\n", (unsigned long long)req.size / 1024);
        vkDestroyImage(dev, img, NULL);
    }
}

// Which create flags cost the BC shadows their AFRC: HZD's BC7 shadows are RGBA8_SRGB with MUTABLE|ALIAS (0x408)
// and a UNORM/SRGB format list. Queries the offered rates and creates a 1024^2 image with explicit 2 bpc for each
// flag combination. [Claude Opus 5.5] 2026-10-26
static void probe_flags(VkPhysicalDevice pd, VkDevice dev, PFN_vkGetImageSubresourceLayout2KHR layout2) {
    static const VkFormat list[2] = {VK_FORMAT_R8G8B8A8_UNORM, VK_FORMAT_R8G8B8A8_SRGB};
    static const struct { VkImageCreateFlags flags; int with_list; const char *name; } v[] = {
        {0, 0, "none"},
        {VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT, 1, "MUTABLE + list"},
        {VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT, 0, "MUTABLE, no list"},
        {VK_IMAGE_CREATE_ALIAS_BIT, 0, "ALIAS"},
        {VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT | VK_IMAGE_CREATE_ALIAS_BIT, 1, "MUTABLE|ALIAS + list"},
        {VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT, 0, "CUBE (6 layers)"},
        {VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT | VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT | VK_IMAGE_CREATE_ALIAS_BIT, 1, "CUBE|MUTABLE|ALIAS+list"},
    };
    for (unsigned i = 0; i < sizeof v / sizeof v[0]; i++) {
        VkImageFormatListCreateInfo fl = {VK_STRUCTURE_TYPE_IMAGE_FORMAT_LIST_CREATE_INFO, NULL, 2, list};
        VkImageCompressionControlEXT cc = {VK_STRUCTURE_TYPE_IMAGE_COMPRESSION_CONTROL_EXT, v[i].with_list ? &fl : NULL,
                                           VK_IMAGE_COMPRESSION_FIXED_RATE_DEFAULT_EXT};
        VkPhysicalDeviceImageFormatInfo2 qi = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_IMAGE_FORMAT_INFO_2, &cc, VK_FORMAT_R8G8B8A8_SRGB,
                                               VK_IMAGE_TYPE_2D, VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_SAMPLED_BIT |
                                               VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT, v[i].flags};
        VkImageCompressionPropertiesEXT cp = {VK_STRUCTURE_TYPE_IMAGE_COMPRESSION_PROPERTIES_EXT};
        VkImageFormatProperties2 qp = {VK_STRUCTURE_TYPE_IMAGE_FORMAT_PROPERTIES_2, &cp};
        VkResult r = vkGetPhysicalDeviceImageFormatProperties2(pd, &qi, &qp);
        VkImageCompressionFixedRateFlagsEXT rate = VK_IMAGE_COMPRESSION_FIXED_RATE_2BPC_BIT_EXT;
        VkImageCompressionControlEXT ex = {VK_STRUCTURE_TYPE_IMAGE_COMPRESSION_CONTROL_EXT, v[i].with_list ? &fl : NULL,
                                           VK_IMAGE_COMPRESSION_FIXED_RATE_EXPLICIT_EXT, 1, &rate};
        uint32_t layers = (v[i].flags & VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT) ? 6 : 1;
        VkImageCreateInfo ci = {VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, &ex, v[i].flags, VK_IMAGE_TYPE_2D, VK_FORMAT_R8G8B8A8_SRGB,
                                {1024, 1024, 1}, 11, layers, VK_SAMPLE_COUNT_1_BIT, VK_IMAGE_TILING_OPTIMAL, qi.usage};
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
        printf("SRGB flags 0x%x %-22s: query %d compression 0x%x rates 0x%x; explicit 2 bpc: create %d, %llu KiB, applied 0x%x / 0x%x\n",
               v[i].flags, v[i].name, r, cp.imageCompressionFlags, cp.imageCompressionFixedRateFlags, cr,
               (unsigned long long)req.size / 1024, got.imageCompressionFlags, got.imageCompressionFixedRateFlags);
    }
}

int main(void) {
    setvbuf(stdout, NULL, _IONBF, 0);
    VkApplicationInfo app = {VK_STRUCTURE_TYPE_APPLICATION_INFO, NULL, "afrc-probe", 1, NULL, 0, VK_API_VERSION_1_3};
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
    PFN_vkGetImageSubresourceLayout2KHR layout2 = (PFN_vkGetImageSubresourceLayout2KHR)vkGetDeviceProcAddr(dev, "vkGetImageSubresourceLayout2EXT");
    if (!layout2) layout2 = (PFN_vkGetImageSubresourceLayout2KHR)vkGetDeviceProcAddr(dev, "vkGetImageSubresourceLayout2KHR");
    probe(pd, dev, layout2, VK_FORMAT_R8G8B8A8_UNORM);
    probe(pd, dev, layout2, VK_FORMAT_R8G8B8A8_SRGB);
    probe(pd, dev, layout2, VK_FORMAT_R16G16_UNORM);
    probe(pd, dev, layout2, VK_FORMAT_R16G16B16A16_SFLOAT);
    probe_flags(pd, dev, layout2);
    return 0;
}

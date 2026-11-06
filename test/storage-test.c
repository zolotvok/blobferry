// SPDX-License-Identifier: GPL-3.0-or-later
// storage-test: CreateDevice through the compat layer with 16/8-bit storage enabled (Vulkan11/12 structs + 16/8-bit
// structs); PASS = device created, app structs restored. [game/doom_eternal, Claude Opus 5.5] 2026-11-06
#include <vulkan/vulkan.h>
#include <stdio.h>
int main(int argc, char **argv) {
    int use_core = argc > 1 && argv[1][0] == 'c';
    const char *layer = "VK_LAYER_DROIDDECK_mali_compat";
    VkApplicationInfo ai = {VK_STRUCTURE_TYPE_APPLICATION_INFO, .apiVersion = VK_API_VERSION_1_2};
    VkInstanceCreateInfo ici = {VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, .pApplicationInfo = &ai, .enabledLayerCount = 1, .ppEnabledLayerNames = &layer};
    VkInstance in; if (vkCreateInstance(&ici, NULL, &in)) { puts("FAIL instance"); return 1; }
    uint32_t n = 1; VkPhysicalDevice pd; vkEnumeratePhysicalDevices(in, &n, &pd);
    float pr = 1; VkDeviceQueueCreateInfo q = {VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO, .queueCount = 1, .pQueuePriorities = &pr};
    VkPhysicalDeviceVulkan12Features v12 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES, .storageBuffer8BitAccess = 1, .uniformAndStorageBuffer8BitAccess = 1};
    VkPhysicalDeviceVulkan11Features v11 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES, &v12, .storageBuffer16BitAccess = 1, .uniformAndStorageBuffer16BitAccess = 1};
    VkPhysicalDevice8BitStorageFeatures f8 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_8BIT_STORAGE_FEATURES, .storageBuffer8BitAccess = 1};
    VkPhysicalDevice16BitStorageFeatures f16 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES, &f8, .storageBuffer16BitAccess = 1};
    VkDeviceCreateInfo dci = {VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, use_core ? (void *)&v11 : (void *)&f16, 0, 1, &q};
    VkDevice d; VkResult r = vkCreateDevice(pd, &dci, NULL, &d);
    int ok = r == VK_SUCCESS && (use_core ? (v11.storageBuffer16BitAccess && v12.storageBuffer8BitAccess) : (f16.storageBuffer16BitAccess && f8.storageBuffer8BitAccess));
    printf("%s mode %s vkCreateDevice %d\n", ok ? "PASS" : "FAIL", use_core ? "core" : "ext", r);
    if (r == VK_SUCCESS) vkDestroyDevice(d, NULL);
    vkDestroyInstance(in, NULL);
    return !ok;
}

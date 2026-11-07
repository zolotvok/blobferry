// SPDX-License-Identifier: GPL-3.0-or-later
// featmask-test: requests EVERY core feature (pEnabledFeatures, then VkPhysicalDeviceFeatures2) and expects
// vkCreateDevice to succeed under the compat layer, which must drop exactly the ones the driver reports false.
// Prints the driver's false features so they can be compared with the layer's "masks N core feature(s)" line.
// [keeper, Claude Opus 5.5] 2026-11-06
#include <vulkan/vulkan.h>
#include <stdio.h>
#include <string.h>
int main(void) {
    VkApplicationInfo ai = {VK_STRUCTURE_TYPE_APPLICATION_INFO, NULL, "featmask", 1, NULL, 0, VK_API_VERSION_1_2};
    VkInstanceCreateInfo ici = {VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, NULL, 0, &ai};
    VkInstance in; if (vkCreateInstance(&ici, NULL, &in)) { puts("FAIL instance"); return 1; }
    uint32_t n = 1; VkPhysicalDevice pd; vkEnumeratePhysicalDevices(in, &n, &pd); if (!n) { puts("FAIL no gpu"); return 1; }
    VkPhysicalDeviceFeatures f; vkGetPhysicalDeviceFeatures(pd, &f);
    VkBool32 *a = (VkBool32 *)&f; int nf = 0;
    printf("layer-reported false idx:");
    for (unsigned i = 0; i < sizeof f / sizeof(VkBool32); i++) if (!a[i]) { printf(" %u", i); nf++; }
    printf(" (%d)\n", nf);
    float pr = 1; VkDeviceQueueCreateInfo q = {VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO, NULL, 0, 0, 1, &pr};
    VkPhysicalDeviceFeatures all; memset(&all, 0, sizeof all);
    VkBool32 *b = (VkBool32 *)&all; for (unsigned i = 0; i < sizeof all / sizeof(VkBool32); i++) b[i] = VK_TRUE;
    int fails = 0;
    for (int arm = 0; arm < 2; arm++) {
        VkPhysicalDeviceFeatures2 f2 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, NULL, all};
        VkDeviceCreateInfo dci = {VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, arm ? &f2 : NULL, 0, 1, &q};
        if (!arm) dci.pEnabledFeatures = &all;
        VkDevice d; VkResult r = vkCreateDevice(pd, &dci, NULL, &d);
        printf("%s arm %s all-true: vkCreateDevice %d\n", r ? "FAIL" : "PASS", arm ? "features2" : "pEnabledFeatures", r);
        fails += r != 0;
        if (!r) vkDestroyDevice(d, NULL);
    }
    vkDestroyInstance(in, NULL);
    printf("featmask-test fails %d\n", fails);
    return fails != 0;
}

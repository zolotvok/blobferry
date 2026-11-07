// SPDX-License-Identifier: GPL-3.0-or-later
// descpool-test <expect_ok>: pool maxSets 2, 2 uniform buffers; allocate 3 one-UB sets. With MALI_COMPAT_DESC_POOL_SCALE>=2
// all 3 must succeed (expect 1); without it the driver may refuse the 3rd (expect 0 = refused or ok, only reported).
// [game/doom_eternal, Claude Opus 5.5] 2026-11-06
#include <vulkan/vulkan.h>
#include <stdio.h>
#include <stdlib.h>
int main(int argc, char **argv) {
    int want = argc > 1 ? atoi(argv[1]) : 0;
    const char *layer = "VK_LAYER_DROIDDECK_mali_compat";
    VkApplicationInfo ai = {VK_STRUCTURE_TYPE_APPLICATION_INFO, .apiVersion = VK_API_VERSION_1_2};
    VkInstanceCreateInfo ici = {VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, .pApplicationInfo = &ai, .enabledLayerCount = 1, .ppEnabledLayerNames = &layer};
    VkInstance in; if (vkCreateInstance(&ici, NULL, &in)) { puts("FAIL instance"); return 1; }
    uint32_t n = 1; VkPhysicalDevice pd; vkEnumeratePhysicalDevices(in, &n, &pd);
    float pr = 1; VkDeviceQueueCreateInfo q = {VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO, .queueCount = 1, .pQueuePriorities = &pr};
    VkDeviceCreateInfo dci = {VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, NULL, 0, 1, &q};
    VkDevice d; if (vkCreateDevice(pd, &dci, NULL, &d)) { puts("FAIL device"); return 1; }
    VkDescriptorSetLayoutBinding bd = {0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_ALL};
    VkDescriptorSetLayoutCreateInfo dl = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, .bindingCount = 1, .pBindings = &bd};
    VkDescriptorSetLayout sl; vkCreateDescriptorSetLayout(d, &dl, NULL, &sl);
    VkDescriptorPoolSize ps = {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 2};
    VkDescriptorPoolCreateInfo pci = {VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO, .maxSets = 2, .poolSizeCount = 1, .pPoolSizes = &ps};
    VkDescriptorPool pool; VkResult rp = vkCreateDescriptorPool(d, &pci, NULL, &pool);
    VkDescriptorSetAllocateInfo al = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, .descriptorPool = pool, .descriptorSetCount = 1, .pSetLayouts = &sl};
    VkResult r[3]; VkDescriptorSet s[3];
    for (int i = 0; i < 3; i++) r[i] = vkAllocateDescriptorSets(d, &al, &s[i]);
    int all = rp == 0 && r[0] == 0 && r[1] == 0 && r[2] == 0;
    int ok = want ? all : (rp == 0 && r[0] == 0 && r[1] == 0);
    printf("%s pool %d alloc %d %d %d (want_all %d)\n", ok ? "PASS" : "FAIL", rp, r[0], r[1], r[2], want);
    vkResetDescriptorPool(d, pool, 0);
    vkDestroyDescriptorPool(d, pool, NULL); vkDestroyDescriptorSetLayout(d, sl, NULL);
    vkDestroyDevice(d, NULL); vkDestroyInstance(in, NULL);
    return !ok;
}

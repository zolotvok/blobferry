// SPDX-License-Identifier: GPL-3.0-or-later
// faillog-test: a compute pipeline that succeeds, then one that returns VK_PIPELINE_COMPILE_REQUIRED (FAIL_ON_COMPILE_REQUIRED,
// empty cache, inline SPIR-V via VkShaderModuleCreateInfo): the layer must pass both results through unchanged.
// [game/doom_eternal, Claude Opus 5.5] 2026-11-06
#include <vulkan/vulkan.h>
#include <stdio.h>
#include "fail_cs.h"
int main(void) {
    const char *layer = "VK_LAYER_DROIDDECK_mali_compat";
    VkApplicationInfo ai = {VK_STRUCTURE_TYPE_APPLICATION_INFO, .apiVersion = VK_API_VERSION_1_3};
    VkInstanceCreateInfo ici = {VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, .pApplicationInfo = &ai, .enabledLayerCount = 1, .ppEnabledLayerNames = &layer};
    VkInstance in; if (vkCreateInstance(&ici, NULL, &in)) { puts("FAIL instance"); return 1; }
    uint32_t n = 1; VkPhysicalDevice pd; vkEnumeratePhysicalDevices(in, &n, &pd);
    float pr = 1; VkDeviceQueueCreateInfo q = {VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO, .queueCount = 1, .pQueuePriorities = &pr};
    VkPhysicalDeviceVulkan13Features v13 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES, .pipelineCreationCacheControl = 1};
    VkDeviceCreateInfo dci = {VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, &v13, 0, 1, &q};
    VkDevice d; if (vkCreateDevice(pd, &dci, NULL, &d)) { puts("FAIL device"); return 1; }
    VkDescriptorSetLayoutBinding bd = {0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT};
    VkDescriptorSetLayoutCreateInfo dl = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, .bindingCount = 1, .pBindings = &bd};
    VkDescriptorSetLayout sl; vkCreateDescriptorSetLayout(d, &dl, NULL, &sl);
    VkPipelineLayoutCreateInfo pl = {VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, .setLayoutCount = 1, .pSetLayouts = &sl};
    VkPipelineLayout lay; vkCreatePipelineLayout(d, &pl, NULL, &lay);
    VkShaderModuleCreateInfo sm = {VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, .codeSize = sizeof fail_cs, .pCode = fail_cs};
    VkShaderModule mod; VkResult rm = vkCreateShaderModule(d, &sm, NULL, &mod);
    VkComputePipelineCreateInfo cp = {VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, .stage = VK_SHADER_STAGE_COMPUTE_BIT, .module = mod, .pName = "main"}, .layout = lay};
    VkPipeline p1 = 0, p2 = 0;
    VkResult r1 = vkCreateComputePipelines(d, VK_NULL_HANDLE, 1, &cp, NULL, &p1);
    VkPipelineCacheCreateInfo pci = {VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO};
    VkPipelineCache cache; vkCreatePipelineCache(d, &pci, NULL, &cache);
    VkComputePipelineCreateInfo cp2 = cp;
    cp2.stage.module = VK_NULL_HANDLE; cp2.stage.pNext = &sm;
    cp2.flags = VK_PIPELINE_CREATE_FAIL_ON_PIPELINE_COMPILE_REQUIRED_BIT;
    VkResult r2 = vkCreateComputePipelines(d, cache, 1, &cp2, NULL, &p2);
    int ok = rm == VK_SUCCESS && r1 == VK_SUCCESS && p1 && r2 == VK_PIPELINE_COMPILE_REQUIRED;
    printf("%s module %d pipe1 %d pipe2 %d (want 0 0 %d)\n", ok ? "PASS" : "FAIL", rm, r1, r2, VK_PIPELINE_COMPILE_REQUIRED);
    if (p1) vkDestroyPipeline(d, p1, NULL);
    if (p2) vkDestroyPipeline(d, p2, NULL);
    vkDestroyDevice(d, NULL); vkDestroyInstance(in, NULL);
    return !ok;
}

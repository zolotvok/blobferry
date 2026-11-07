// SPDX-License-Identifier: GPL-3.0-or-later
// q2vbo-test: the Quake II RTX vkpt_vertex_buffer_create() descriptor sequence (10 bindings, STAGE_ALL, binding 0 =
// 2 + 256 storage buffers; pool maxSets 2; one set; one update), then the same creates with an allocator that
// refuses memory (arg "old": the <= 1.8.0 pool without the UNIFORM_BUFFER size), so the layer's MALI_COMPAT_FAIL_LOG lines for failing calls can be seen on lavapipe.
// [keeper, Claude Opus 5.5] 2026-11-07
#include <vulkan/vulkan.h>
#include <stdio.h>
#include <string.h>
static void *VKAPI_PTR nofn(void *u, size_t s, size_t a, VkSystemAllocationScope sc) { return NULL; }
static void *VKAPI_PTR noreal(void *u, void *o, size_t s, size_t a, VkSystemAllocationScope sc) { return NULL; }
static void VKAPI_PTR nofree(void *u, void *m) {}
int main(int argc, char **argv) {
    int old = argc > 1 && !strcmp(argv[1], "old"); // Q2RTX <= 1.8.0: pool lists only STORAGE_BUFFER
    const char *layer = "VK_LAYER_DROIDDECK_mali_compat";
    VkApplicationInfo ai = {VK_STRUCTURE_TYPE_APPLICATION_INFO, .apiVersion = VK_API_VERSION_1_2};
    VkInstanceCreateInfo ici = {VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, .pApplicationInfo = &ai, .enabledLayerCount = 1, .ppEnabledLayerNames = &layer};
    VkInstance in; if (vkCreateInstance(&ici, NULL, &in)) { puts("FAIL instance"); return 1; }
    uint32_t n = 1; VkPhysicalDevice pd; vkEnumeratePhysicalDevices(in, &n, &pd);
    float pr = 1; VkDeviceQueueCreateInfo q = {VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO, .queueCount = 1, .pQueuePriorities = &pr};
    VkDeviceCreateInfo dci = {VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, NULL, 0, 1, &q};
    VkDevice d; if (vkCreateDevice(pd, &dci, NULL, &d)) { puts("FAIL device"); return 1; }
    VkDescriptorSetLayoutBinding b[10];
    uint32_t cnt[10] = {2 + 256, 1, 1, 3, 1, 1, 1, 1, 1, 3};
    for (int i = 0; i < 10; i++)
        b[i] = (VkDescriptorSetLayoutBinding){i, i == 8 ? VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER : VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, cnt[i], VK_SHADER_STAGE_ALL};
    VkDescriptorSetLayoutCreateInfo dl = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, .bindingCount = 10, .pBindings = b};
    VkDescriptorSetLayout sl = 0; VkResult r1 = vkCreateDescriptorSetLayout(d, &dl, NULL, &sl);
    VkDescriptorPoolSize ps[2] = {{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 10 + 256 + 128}, {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1}};
    VkDescriptorPoolCreateInfo pci = {VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO, .maxSets = 2, .poolSizeCount = old ? 1 : 2, .pPoolSizes = ps};
    VkDescriptorPool pool = 0; VkResult r2 = vkCreateDescriptorPool(d, &pci, NULL, &pool);
    VkDescriptorSetAllocateInfo al = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, .descriptorPool = pool, .descriptorSetCount = 1, .pSetLayouts = &sl};
    VkDescriptorSet set = 0; VkResult r3 = vkAllocateDescriptorSets(d, &al, &set);
    VkPipelineLayoutCreateInfo plc = {VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, .setLayoutCount = 1, .pSetLayouts = &sl};
    VkPipelineLayout pl = 0; VkResult r4 = vkCreatePipelineLayout(d, &plc, NULL, &pl);
    int ok = !r1 && !r2 && !r3 && !r4 && set && sl && pool && pl;
    printf("%s normal: dsl %d pool %d alloc %d set %s pl %d\n", ok ? "PASS" : "FAIL", r1, r2, r3, set ? "ok" : "NULL", r4);
    VkAllocationCallbacks no = {NULL, nofn, noreal, nofree};
    VkDescriptorSetLayout sl2 = 0; VkDescriptorPool p2 = 0; VkPipelineLayout pl2 = 0;
    VkResult f1 = vkCreateDescriptorSetLayout(d, &dl, &no, &sl2);
    VkResult f2 = vkCreateDescriptorPool(d, &pci, &no, &p2);
    VkResult f3 = vkCreatePipelineLayout(d, &plc, &no, &pl2);
    printf("INFO refusing allocator: dsl %d pool %d pl %d (non-zero expected, see fail-log lines)\n", f1, f2, f3);
    vkDestroyPipelineLayout(d, pl, NULL); vkDestroyDescriptorPool(d, pool, NULL); vkDestroyDescriptorSetLayout(d, sl, NULL);
    vkDestroyDevice(d, NULL); vkDestroyInstance(in, NULL);
    return !ok;
}

// pushdesc-probe: does the vendor driver read a push-descriptor UBO correctly in vkd3d's root-CBV layout? With
// MALI_COMPAT_VENDOR_ID=5143 vkd3d binds root CBVs as a push-descriptor UBO in the last set (set 2 behind its bindless
// sets) and the DX12 Witcher renders white (2026-10-28). A compute shader copies the 12 vec4 of the push UBO into an
// SSBO; two pushes with different buffers in one command buffer, plus a normal UBO in set 1. Prints OK or the first
// wrong values. Graphics case: the same through a fragment shader, two draws with a push in between inside one
// dynamic rendering pass (no attachments, the fragment shader writes the SSBO). [Claude Opus 5.5] 2026-10-28
// Run on the ICD alone: VK_DRIVER_FILES=<bridge icd.json> HYBRIS_USE_VENDOR_NAMESPACE=1 VK_LOADER_LAYERS_DISABLE=~implicit~
// The shader headers in test/shaders are generated: glslangValidator -V --vn pushdesc_spv pushdesc-probe.comp -o pushdesc-probe.h
// (and pushdesc_vs / pushdesc_fs for the .vert / .frag into pushdesc-probe-vs.h / -fs.h).
#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <vulkan/vulkan.h>
#include "shaders/pushdesc-probe.h"
#include "shaders/pushdesc-probe-vs.h"
#include "shaders/pushdesc-probe-fs.h"

#define CK(x) do { VkResult r_ = (x); if (r_) { printf("%s failed: %d\n", #x, r_); return 1; } } while (0)

static VkDevice dev;
static VkPhysicalDeviceMemoryProperties mp;

static int buffer(VkDeviceSize size, VkBufferUsageFlags usage, VkBuffer *b, VkDeviceMemory *m, void **map) {
    VkBufferCreateInfo bi = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, NULL, 0, size, usage};
    CK(vkCreateBuffer(dev, &bi, NULL, b));
    VkMemoryRequirements req;
    vkGetBufferMemoryRequirements(dev, *b, &req);
    uint32_t t = 0;
    for (; t < mp.memoryTypeCount; t++)
        if ((req.memoryTypeBits & (1u << t)) && (mp.memoryTypes[t].propertyFlags &
            (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) ==
            (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) break;
    VkMemoryAllocateInfo ai = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, NULL, req.size, t};
    CK(vkAllocateMemory(dev, &ai, NULL, m));
    CK(vkBindBufferMemory(dev, *b, *m, 0));
    CK(vkMapMemory(dev, *m, 0, size, 0, map));
    return 0;
}

int main(void) {
    VkApplicationInfo app = {VK_STRUCTURE_TYPE_APPLICATION_INFO, NULL, "pushdesc-probe", 1, NULL, 0, VK_API_VERSION_1_3};
    VkInstanceCreateInfo ici = {VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, NULL, 0, &app};
    VkInstance inst;
    CK(vkCreateInstance(&ici, NULL, &inst));
    uint32_t n = 1;
    VkPhysicalDevice pd;
    vkEnumeratePhysicalDevices(inst, &n, &pd);
    vkGetPhysicalDeviceMemoryProperties(pd, &mp);
    float prio = 1;
    VkDeviceQueueCreateInfo qi = {VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO, NULL, 0, 0, 1, &prio};
    const char *ext[] = {"VK_KHR_push_descriptor"};
    VkPhysicalDeviceVulkan13Features f13 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
    f13.dynamicRendering = VK_TRUE;
    VkPhysicalDeviceFeatures2 f2 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, &f13};
    f2.features.fragmentStoresAndAtomics = VK_TRUE;
    VkDeviceCreateInfo dci = {VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, &f2, 0, 1, &qi, 0, NULL, 1, ext};
    CK(vkCreateDevice(pd, &dci, NULL, &dev));
    PFN_vkCmdPushDescriptorSetKHR push = (PFN_vkCmdPushDescriptorSetKHR)vkGetDeviceProcAddr(dev, "vkCmdPushDescriptorSetKHR");
    VkQueue q;
    vkGetDeviceQueue(dev, 0, 0, &q);

    VkBuffer out, plain, ra, rb; VkDeviceMemory mo, mpl, ma, mb; void *po, *ppl, *pa, *pb;
    if (buffer(4096, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &out, &mo, &po) ||
        buffer(4096, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, &plain, &mpl, &ppl) ||
        buffer(65536 * 4, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, &ra, &ma, &pa) ||
        buffer(65536 * 4, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, &rb, &mb, &pb)) return 1;
    memset(po, 0, 4096); memset(ppl, 0, 4096);
    // root CBVs at 256-byte aligned offsets inside big buffers, like vkd3d's heaps
    const VkDeviceSize off_a = 768, off_b = 65536 + 256;
    for (int i = 0; i < 48; i++) { ((float *)pa)[off_a / 4 + i] = 100 + i; ((float *)pb)[off_b / 4 + i] = 200 + i; }

    VkDescriptorSetLayoutBinding b0 = {0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT | VK_SHADER_STAGE_FRAGMENT_BIT};
    VkDescriptorSetLayoutBinding b1 = {0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT | VK_SHADER_STAGE_FRAGMENT_BIT};
    VkDescriptorSetLayout sl[3];
    VkDescriptorSetLayoutCreateInfo li0 = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, NULL, 0, 1, &b0};
    VkDescriptorSetLayoutCreateInfo li1 = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, NULL, 0, 1, &b1};
    VkDescriptorSetLayoutCreateInfo li2 = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, NULL,
                                           VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR, 1, &b1};
    CK(vkCreateDescriptorSetLayout(dev, &li0, NULL, &sl[0]));
    CK(vkCreateDescriptorSetLayout(dev, &li1, NULL, &sl[1]));
    CK(vkCreateDescriptorSetLayout(dev, &li2, NULL, &sl[2]));
    VkPushConstantRange pcr = {VK_SHADER_STAGE_COMPUTE_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, 4};
    VkPipelineLayoutCreateInfo pli = {VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, NULL, 0, 3, sl, 1, &pcr};
    VkPipelineLayout pl;
    CK(vkCreatePipelineLayout(dev, &pli, NULL, &pl));
    VkShaderModuleCreateInfo smi = {VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, NULL, 0, sizeof(pushdesc_spv), pushdesc_spv};
    VkShaderModule sm;
    CK(vkCreateShaderModule(dev, &smi, NULL, &sm));
    VkComputePipelineCreateInfo cpi = {VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO, NULL, 0,
        {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, NULL, 0, VK_SHADER_STAGE_COMPUTE_BIT, sm, "main"}, pl};
    VkPipeline pipe;
    CK(vkCreateComputePipelines(dev, VK_NULL_HANDLE, 1, &cpi, NULL, &pipe));

    VkShaderModuleCreateInfo vsi = {VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, NULL, 0, sizeof(pushdesc_vs), pushdesc_vs};
    VkShaderModuleCreateInfo fsi = {VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, NULL, 0, sizeof(pushdesc_fs), pushdesc_fs};
    VkShaderModule vs, fs;
    CK(vkCreateShaderModule(dev, &vsi, NULL, &vs));
    CK(vkCreateShaderModule(dev, &fsi, NULL, &fs));
    VkPipelineShaderStageCreateInfo st[2] = {
        {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, NULL, 0, VK_SHADER_STAGE_VERTEX_BIT, vs, "main"},
        {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, NULL, 0, VK_SHADER_STAGE_FRAGMENT_BIT, fs, "main"}};
    VkPipelineVertexInputStateCreateInfo vi = {VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
    VkPipelineInputAssemblyStateCreateInfo ia_s = {VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO, NULL, 0,
                                                   VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST};
    VkViewport vp = {0, 0, 12, 1, 0, 1};
    VkRect2D sc = {{0, 0}, {12, 1}};
    VkPipelineViewportStateCreateInfo vps = {VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO, NULL, 0, 1, &vp, 1, &sc};
    VkPipelineRasterizationStateCreateInfo rs = {VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
    rs.lineWidth = 1;
    VkPipelineMultisampleStateCreateInfo ms = {VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO, NULL, 0, VK_SAMPLE_COUNT_1_BIT};
    VkPipelineColorBlendStateCreateInfo cbs = {VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
    VkPipelineRenderingCreateInfo pri = {VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO};
    VkGraphicsPipelineCreateInfo gpi = {VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO, &pri, 0, 2, st, &vi, &ia_s, NULL, &vps, &rs,
                                        &ms, NULL, &cbs, NULL, pl};
    VkPipeline gpipe;
    CK(vkCreateGraphicsPipelines(dev, VK_NULL_HANDLE, 1, &gpi, NULL, &gpipe));
    VkDescriptorPoolSize ps[2] = {{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1}, {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1}};
    VkDescriptorPoolCreateInfo dpi = {VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO, NULL, 0, 2, 2, ps};
    VkDescriptorPool dp;
    CK(vkCreateDescriptorPool(dev, &dpi, NULL, &dp));
    VkDescriptorSet ds[2];
    VkDescriptorSetAllocateInfo dai = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, NULL, dp, 2, sl};
    CK(vkAllocateDescriptorSets(dev, &dai, ds));
    VkDescriptorBufferInfo bo = {out, 0, VK_WHOLE_SIZE}, bp = {plain, 0, 192};
    VkWriteDescriptorSet w[2] = {
        {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, NULL, ds[0], 0, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, NULL, &bo},
        {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, NULL, ds[1], 0, 0, 1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, NULL, &bp}};
    vkUpdateDescriptorSets(dev, 2, w, 0, NULL);

    VkCommandPoolCreateInfo cpci = {VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO, NULL, 0, 0};
    VkCommandPool cp;
    CK(vkCreateCommandPool(dev, &cpci, NULL, &cp));
    VkCommandBufferAllocateInfo cbai = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, NULL, cp, VK_COMMAND_BUFFER_LEVEL_PRIMARY, 1};
    VkCommandBuffer cb;
    CK(vkAllocateCommandBuffers(dev, &cbai, &cb));
    VkCommandBufferBeginInfo bbi = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    CK(vkBeginCommandBuffer(cb, &bbi));
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
    vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pl, 0, 2, ds, 0, NULL);
    // vkd3d's range: the rest of the buffer (capped at maxUniformBufferRange)
    VkDescriptorBufferInfo ia = {ra, off_a, 65536 * 4 - off_a}, ib = {rb, off_b, 65536 * 4 - off_b};
    VkWriteDescriptorSet pw = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, NULL, VK_NULL_HANDLE, 0, 0, 1,
                               VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, NULL, &ia};
    uint32_t base = 0;
    push(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pl, 2, 1, &pw);
    vkCmdPushConstants(cb, pl, VK_SHADER_STAGE_COMPUTE_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, 4, &base);
    vkCmdDispatch(cb, 1, 1, 1);
    pw.pBufferInfo = &ib; base = 12;
    push(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pl, 2, 1, &pw);
    vkCmdPushConstants(cb, pl, VK_SHADER_STAGE_COMPUTE_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, 4, &base);
    vkCmdDispatch(cb, 1, 1, 1);
    VkMemoryBarrier mb_ = {VK_STRUCTURE_TYPE_MEMORY_BARRIER, NULL, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_WRITE_BIT};
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 1, &mb_, 0, NULL, 0, NULL);
    VkRenderingInfo ri = {VK_STRUCTURE_TYPE_RENDERING_INFO, NULL, 0, {{0, 0}, {12, 1}}, 1, 0, 0, NULL, NULL, NULL};
    vkCmdBeginRendering(cb, &ri);
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, gpipe);
    vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, pl, 0, 2, ds, 0, NULL);
    pw.pBufferInfo = &ia; base = 24;
    push(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, pl, 2, 1, &pw);
    vkCmdPushConstants(cb, pl, VK_SHADER_STAGE_COMPUTE_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, 4, &base);
    vkCmdDraw(cb, 3, 1, 0, 0);
    pw.pBufferInfo = &ib; base = 36;
    push(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, pl, 2, 1, &pw);
    vkCmdPushConstants(cb, pl, VK_SHADER_STAGE_COMPUTE_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, 4, &base);
    vkCmdDraw(cb, 3, 1, 0, 0);
    vkCmdEndRendering(cb);
    CK(vkEndCommandBuffer(cb));
    VkSubmitInfo si = {VK_STRUCTURE_TYPE_SUBMIT_INFO, NULL, 0, NULL, NULL, 1, &cb};
    CK(vkQueueSubmit(q, 1, &si, VK_NULL_HANDLE));
    CK(vkQueueWaitIdle(q));
    const float *o = po;
    int bad = 0;
    for (int i = 0; i < 192; i++) {  // vec4 0-11 compute A, 12-23 compute B, 24-35 graphics A, 36-47 graphics B
        int k = i % 96;
        float want = k < 48 ? 100 + k : 200 + (k - 48);
        if (o[i] != want && bad++ < 8) printf("out[%d] = %g, want %g\n", i, o[i], want);
    }
    printf(bad ? "pushdesc-probe: FAIL (%d wrong)\n" : "pushdesc-probe: OK\n", bad);
    return bad != 0;
}

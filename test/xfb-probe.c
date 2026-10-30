// xfb-probe: what costs ~1.4 ms per draw in Unity's DX11 GPU skinning on the vendor driver? DXVK turns D3D11 stream
// output into a passthrough geometry shader with transform feedback (Yooka-Laylee IL, 2026-10-30: 26 such draws of
// ~3848 points take 35-41 ms per frame). Each case records FRAMES x 26 draws of 3848 points, every draw inside its own
// Begin/EndTransformFeedback with a counter buffer (as DXVK does), rasterizer discard, and prints the GPU ms per frame:
//   vs        vertex shader only, no capture
//   vs+xfb    capture in the vertex shader (no geometry shader)
//   gs        passthrough geometry shader, no capture
//   gs+xfb    passthrough geometry shader with capture (DXVK's shape)
// "+rt": the same inside a 1600x720 RGBA8 colour pass (DXVK keeps the game's render targets bound).
// Run on the ICD alone: VK_DRIVER_FILES=<bridge icd.json> HYBRIS_USE_VENDOR_NAMESPACE=1 VK_LOADER_LAYERS_DISABLE=~implicit~
// Shader headers: glslangValidator -V --vn xfb_vs test/shaders/xfb-probe.vert -o test/shaders/xfb-probe-vs.h (and
// xfb_vs_xfb / xfb_gs_xfb / xfb_gs for xfb-probe-xfb.vert, xfb-probe-xfb.geom, xfb-probe.geom). [Claude Opus 5.5] 2026-10-30
#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <vulkan/vulkan.h>
#include "shaders/xfb-probe-vs.h"
#include "shaders/xfb-probe-vs-xfb.h"
#include "shaders/xfb-probe-gs-xfb.h"
#include "shaders/xfb-probe-gs.h"

#define CK(x) do { VkResult r_ = (x); if (r_) { printf("%s failed: %d\n", #x, r_); return 1; } } while (0)
#define VERTS 3848
#define DRAWS 26
#define FRAMES 10

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
    if (map) CK(vkMapMemory(dev, *m, 0, size, 0, map));
    return 0;
}

static VkShaderModule module(const uint32_t *code, size_t size) {
    VkShaderModuleCreateInfo ci = {VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, NULL, 0, size, code};
    VkShaderModule m = VK_NULL_HANDLE;
    vkCreateShaderModule(dev, &ci, NULL, &m);
    return m;
}

int main(void) {
    VkApplicationInfo app = {VK_STRUCTURE_TYPE_APPLICATION_INFO, NULL, "xfb-probe", 1, NULL, 0, VK_API_VERSION_1_3};
    VkInstanceCreateInfo ici = {VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, NULL, 0, &app};
    VkInstance inst;
    CK(vkCreateInstance(&ici, NULL, &inst));
    uint32_t n = 1;
    VkPhysicalDevice pd;
    vkEnumeratePhysicalDevices(inst, &n, &pd);
    vkGetPhysicalDeviceMemoryProperties(pd, &mp);
    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(pd, &props);
    VkPhysicalDeviceTransformFeedbackPropertiesEXT xp = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TRANSFORM_FEEDBACK_PROPERTIES_EXT};
    VkPhysicalDeviceProperties2 p2 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, &xp};
    vkGetPhysicalDeviceProperties2(pd, &p2);
    printf("%s, xfb: buffers %u, queries %u, streams select %u, draw %u, rasterization stream select %u\n",
           props.deviceName, xp.maxTransformFeedbackBuffers, xp.transformFeedbackQueries,
           xp.transformFeedbackStreamsLinesTriangles, xp.transformFeedbackDraw, xp.transformFeedbackRasterizationStreamSelect);

    float prio = 1;
    VkDeviceQueueCreateInfo qi = {VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO, NULL, 0, 0, 1, &prio};
    const char *ext[] = {"VK_EXT_transform_feedback"};
    VkPhysicalDeviceTransformFeedbackFeaturesEXT xf = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TRANSFORM_FEEDBACK_FEATURES_EXT};
    xf.transformFeedback = VK_TRUE;
    VkPhysicalDeviceVulkan13Features f13 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES, &xf};
    f13.dynamicRendering = VK_TRUE;
    VkPhysicalDeviceFeatures2 f2 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, &f13};
    f2.features.geometryShader = VK_TRUE;
    VkDeviceCreateInfo dci = {VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, &f2, 0, 1, &qi, 0, NULL, 1, ext};
    CK(vkCreateDevice(pd, &dci, NULL, &dev));
    PFN_vkCmdBindTransformFeedbackBuffersEXT bindXfb = (PFN_vkCmdBindTransformFeedbackBuffersEXT)vkGetDeviceProcAddr(dev, "vkCmdBindTransformFeedbackBuffersEXT");
    PFN_vkCmdBeginTransformFeedbackEXT beginXfb = (PFN_vkCmdBeginTransformFeedbackEXT)vkGetDeviceProcAddr(dev, "vkCmdBeginTransformFeedbackEXT");
    PFN_vkCmdEndTransformFeedbackEXT endXfb = (PFN_vkCmdEndTransformFeedbackEXT)vkGetDeviceProcAddr(dev, "vkCmdEndTransformFeedbackEXT");
    VkQueue q;
    vkGetDeviceQueue(dev, 0, 0, &q);

    // inputs: one buffer per draw (each skinned mesh has its own), outputs: one capture buffer per draw + counters
    VkBuffer vb, xb, cb; VkDeviceMemory vm, xm, cm; void *vp_;
    if (buffer((VkDeviceSize)DRAWS * VERTS * 48, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, &vb, &vm, &vp_) ||
        buffer((VkDeviceSize)DRAWS * VERTS * 48, VK_BUFFER_USAGE_TRANSFORM_FEEDBACK_BUFFER_BIT_EXT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, &xb, &xm, NULL) ||
        buffer(4096, VK_BUFFER_USAGE_TRANSFORM_FEEDBACK_COUNTER_BUFFER_BIT_EXT, &cb, &cm, NULL)) return 1;
    for (uint32_t i = 0; i < DRAWS * VERTS * 12; i++) ((float *)vp_)[i] = (float)(i % 97) * 0.01f;

    // 1600x720 colour target for the "+rt" cases
    VkImageCreateInfo imci = {VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, NULL, 0, VK_IMAGE_TYPE_2D, VK_FORMAT_R8G8B8A8_UNORM,
                              {1600, 720, 1}, 1, 1, VK_SAMPLE_COUNT_1_BIT, VK_IMAGE_TILING_OPTIMAL,
                              VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT};
    VkImage img;
    CK(vkCreateImage(dev, &imci, NULL, &img));
    VkMemoryRequirements ir;
    vkGetImageMemoryRequirements(dev, img, &ir);
    uint32_t it = 0;
    while (!(ir.memoryTypeBits & (1u << it)) || !(mp.memoryTypes[it].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)) it++;
    VkMemoryAllocateInfo iai = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, NULL, ir.size, it};
    VkDeviceMemory im;
    CK(vkAllocateMemory(dev, &iai, NULL, &im));
    CK(vkBindImageMemory(dev, img, im, 0));
    VkImageViewCreateInfo ivci = {VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, NULL, 0, img, VK_IMAGE_VIEW_TYPE_2D,
                                  VK_FORMAT_R8G8B8A8_UNORM, {0}, {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
    VkImageView view;
    CK(vkCreateImageView(dev, &ivci, NULL, &view));

    VkPipelineLayoutCreateInfo pli = {VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    VkPipelineLayout pl;
    CK(vkCreatePipelineLayout(dev, &pli, NULL, &pl));
    VkShaderModule vs = module(xfb_vs, sizeof xfb_vs), vsx = module(xfb_vs_xfb, sizeof xfb_vs_xfb);
    VkShaderModule gs = module(xfb_gs, sizeof xfb_gs), gsx = module(xfb_gs_xfb, sizeof xfb_gs_xfb);

    struct { const char *name; VkShaderModule v, g; int xfb; } cases[] = {
        {"vs", vs, 0, 0}, {"vs+xfb", vsx, 0, 1}, {"gs", vs, gs, 0}, {"gs+xfb", vs, gsx, 1}};
    VkVertexInputBindingDescription vbd = {0, 48, VK_VERTEX_INPUT_RATE_VERTEX};
    VkVertexInputAttributeDescription vad[3] = {{0, 0, VK_FORMAT_R32G32B32A32_SFLOAT, 0}, {1, 0, VK_FORMAT_R32G32B32A32_SFLOAT, 16},
                                                {2, 0, VK_FORMAT_R32G32B32A32_SFLOAT, 32}};
    VkPipelineVertexInputStateCreateInfo vi = {VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO, NULL, 0, 1, &vbd, 3, vad};
    VkPipelineInputAssemblyStateCreateInfo ia = {VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO, NULL, 0,
                                                 VK_PRIMITIVE_TOPOLOGY_POINT_LIST};
    VkViewport vpt = {0, 0, 1600, 720, 0, 1};
    VkRect2D sc = {{0, 0}, {1600, 720}};
    VkPipelineViewportStateCreateInfo vps = {VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO, NULL, 0, 1, &vpt, 1, &sc};
    VkPipelineRasterizationStateCreateInfo rs = {VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
    rs.rasterizerDiscardEnable = VK_TRUE;
    rs.lineWidth = 1;
    VkPipelineMultisampleStateCreateInfo ms = {VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO, NULL, 0, VK_SAMPLE_COUNT_1_BIT};
    VkPipelineColorBlendAttachmentState cba = {0};
    cba.colorWriteMask = 0xf;
    VkPipelineColorBlendStateCreateInfo cbs = {VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
    VkFormat cfmt = VK_FORMAT_R8G8B8A8_UNORM;

    VkCommandPoolCreateInfo cpi = {VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO, NULL, VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT, 0};
    VkCommandPool pool;
    CK(vkCreateCommandPool(dev, &cpi, NULL, &pool));
    VkCommandBufferAllocateInfo cai = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, NULL, pool, VK_COMMAND_BUFFER_LEVEL_PRIMARY, 1};
    VkCommandBuffer cmd;
    CK(vkAllocateCommandBuffers(dev, &cai, &cmd));
    VkQueryPoolCreateInfo qpi = {VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO, NULL, 0, VK_QUERY_TYPE_TIMESTAMP, 2};
    VkQueryPool qp;
    CK(vkCreateQueryPool(dev, &qpi, NULL, &qp));
    VkFenceCreateInfo fci = {VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    VkFence fence;
    CK(vkCreateFence(dev, &fci, NULL, &fence));

    for (int rt = 0; rt < 2; rt++)
        for (unsigned c = 0; c < sizeof cases / sizeof cases[0]; c++) {
            VkPipelineShaderStageCreateInfo st[2] = {
                {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, NULL, 0, VK_SHADER_STAGE_VERTEX_BIT, cases[c].v, "main"},
                {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, NULL, 0, VK_SHADER_STAGE_GEOMETRY_BIT, cases[c].g, "main"}};
            VkPipelineRenderingCreateInfo pri = {VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO, NULL, 0, rt ? 1 : 0, &cfmt};
            cbs.attachmentCount = rt ? 1 : 0;
            cbs.pAttachments = &cba;
            VkGraphicsPipelineCreateInfo gpi = {VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO, &pri, 0, cases[c].g ? 2 : 1, st, &vi, &ia,
                                                NULL, &vps, &rs, &ms, NULL, &cbs, NULL, pl};
            VkPipeline pipe;
            CK(vkCreateGraphicsPipelines(dev, VK_NULL_HANDLE, 1, &gpi, NULL, &pipe));
            double sum = 0, mn = 1e9;
            for (int f = 0; f <= FRAMES; f++) {
                CK(vkResetCommandBuffer(cmd, 0));
                VkCommandBufferBeginInfo bi = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, NULL, VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT};
                CK(vkBeginCommandBuffer(cmd, &bi));
                vkCmdResetQueryPool(cmd, qp, 0, 2);
                if (rt) {
                    VkImageMemoryBarrier b = {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER, NULL, 0, VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
                                              VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, 0, 0, img,
                                              {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
                    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, 0, 0, NULL, 0, NULL, 1, &b);
                }
                vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, qp, 0);
                VkRenderingAttachmentInfo ca = {VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO, NULL, view, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
                ca.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
                ca.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
                VkRenderingInfo ri = {VK_STRUCTURE_TYPE_RENDERING_INFO, NULL, 0, {{0, 0}, {1600, 720}}, 1, 0, rt ? 1 : 0, &ca};
                vkCmdBeginRendering(cmd, &ri);
                vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe);
                for (int d = 0; d < DRAWS; d++) {
                    VkDeviceSize off = (VkDeviceSize)d * VERTS * 48, size = (VkDeviceSize)VERTS * 48;
                    vkCmdBindVertexBuffers(cmd, 0, 1, &vb, &off);
                    if (cases[c].xfb) {
                        VkDeviceSize coff = (VkDeviceSize)d * 16;
                        bindXfb(cmd, 0, 1, &xb, &off, &size);
                        beginXfb(cmd, 0, 1, &cb, &coff); // counter buffer read at begin, like DXVK's resume
                        vkCmdDraw(cmd, VERTS, 1, 0, 0);
                        endXfb(cmd, 0, 1, &cb, &coff);
                    } else {
                        vkCmdDraw(cmd, VERTS, 1, 0, 0);
                    }
                }
                vkCmdEndRendering(cmd);
                vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, qp, 1);
                CK(vkEndCommandBuffer(cmd));
                VkSubmitInfo si = {VK_STRUCTURE_TYPE_SUBMIT_INFO, NULL, 0, NULL, NULL, 1, &cmd};
                CK(vkQueueSubmit(q, 1, &si, fence));
                CK(vkWaitForFences(dev, 1, &fence, VK_TRUE, UINT64_MAX));
                CK(vkResetFences(dev, 1, &fence));
                uint64_t ts[2];
                CK(vkGetQueryPoolResults(dev, qp, 0, 2, sizeof ts, ts, 8, VK_QUERY_RESULT_64_BIT));
                double ms_ = (ts[1] - ts[0]) * props.limits.timestampPeriod / 1e6;
                if (f) { sum += ms_; if (ms_ < mn) mn = ms_; } // frame 0 is the warm-up
            }
            printf("%-8s%s  %7.3f ms per %d draws (min %.3f), %.3f ms per draw\n", cases[c].name, rt ? "+rt" : "   ",
                   sum / FRAMES, DRAWS, mn, sum / FRAMES / DRAWS);
            vkDestroyPipeline(dev, pipe, NULL);
        }
    return 0;
}

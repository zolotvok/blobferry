// SPDX-License-Identifier: GPL-3.0-or-later
// cover-trim-test: PC end-to-end check of MALI_COMPAT_COVER_TRIM (bridge/src/compat/cover_trim.h) on lavapipe.
// A "weapon" G-buffer pass (1 colour + D32S8 cleared to 0, N draws of a quad at depth 0.5 over a known pixel rect),
// then a listed probe compute shader (32x16 groups of 8x8; each group writes 1 + its push value into marks[]) and a
// listed composite draw (full-screen triangle writing 255 into an R8 target). The test computes the expected cells
// with its own copy of the rules and compares group by group and pixel by pixel. Modes:
//   on    trim on, 1 draw  -> marks/pixels only in the cells near the quad (and every mark = 1 + push: restore OK)
//   off   trim off         -> everything
//   many  trim on, 20 draws (> ARM max 16) -> disarmed, everything
//   toggle trim on, toggle file absent -> everything
// Built and run by bridge/test/cover-trim-test.sh. [keeper, Claude Opus 5.5] 2026-11-05
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vulkan/vulkan.h>
#include "ctt_quad_vs.h"
#include "ctt_gbuf_fs.h"
#include "ctt_full_vs.h"
#include "ctt_comp_fs.h"
#include "ctt_probe_cs.h"

#define W 256
#define H 128
#define GX 32
#define GY 16
#define NX 8
#define NY 4
#define MARGIN 8
#define K 2
static const int QX0 = 150, QX1 = 180, QY0 = 90, QY1 = 110; // quad pixel rect [x0,x1) x [y0,y1)

static VkDevice dev;
static VkPhysicalDevice pd;
#define CK(x) do { VkResult r_ = (x); if (r_ != VK_SUCCESS) { fprintf(stderr, "FAIL %s = %d (line %d)\n", #x, r_, __LINE__); exit(2); } } while (0)

static uint32_t fnv(const uint32_t *c, size_t bytes) {
    uint32_t h = 2166136261u;
    for (size_t w = 0; w < bytes / 4; w++) h = (h ^ c[w]) * 16777619u;
    return h;
}

static VkDeviceMemory alloc(VkMemoryRequirements req, bool host) {
    VkPhysicalDeviceMemoryProperties mp;
    vkGetPhysicalDeviceMemoryProperties(pd, &mp);
    uint32_t type = 0;
    for (; type < mp.memoryTypeCount; type++)
        if ((req.memoryTypeBits & (1u << type)) && (!host || (mp.memoryTypes[type].propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT))) break;
    VkMemoryAllocateFlagsInfo fi = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO, NULL, VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT};
    VkMemoryAllocateInfo ai = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, &fi, req.size, type};
    VkDeviceMemory m;
    CK(vkAllocateMemory(dev, &ai, NULL, &m));
    return m;
}
static VkBuffer buffer(VkDeviceSize size, void **map, VkDeviceAddress *addr) {
    VkBufferCreateInfo ci = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, NULL, 0, size,
                             VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT};
    VkBuffer b;
    CK(vkCreateBuffer(dev, &ci, NULL, &b));
    VkMemoryRequirements req;
    vkGetBufferMemoryRequirements(dev, b, &req);
    VkDeviceMemory m = alloc(req, true);
    CK(vkBindBufferMemory(dev, b, m, 0));
    CK(vkMapMemory(dev, m, 0, VK_WHOLE_SIZE, 0, map));
    memset(*map, 0, size);
    VkBufferDeviceAddressInfo bi = {VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO, NULL, b};
    if (addr) *addr = vkGetBufferDeviceAddress(dev, &bi);
    return b;
}
static VkImage image(VkFormat f, VkImageUsageFlags u, VkImageView *view, VkImageAspectFlags aspect) {
    VkImageCreateInfo ci = {VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, NULL, 0, VK_IMAGE_TYPE_2D, f, {W, H, 1}, 1, 1,
                            VK_SAMPLE_COUNT_1_BIT, VK_IMAGE_TILING_OPTIMAL, u};
    VkImage img;
    CK(vkCreateImage(dev, &ci, NULL, &img));
    VkMemoryRequirements req;
    vkGetImageMemoryRequirements(dev, img, &req);
    CK(vkBindImageMemory(dev, img, alloc(req, false), 0));
    VkImageViewCreateInfo vi = {VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, NULL, 0, img, VK_IMAGE_VIEW_TYPE_2D, f, {0}, {aspect, 0, 1, 0, 1}};
    CK(vkCreateImageView(dev, &vi, NULL, view));
    return img;
}
static VkShaderModule module(const uint32_t *code, size_t size) {
    VkShaderModuleCreateInfo ci = {VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, NULL, 0, size, code};
    VkShaderModule m;
    CK(vkCreateShaderModule(dev, &ci, NULL, &m));
    return m;
}
static VkPipeline gfx(VkPipelineLayout layout, VkShaderModule vs, VkShaderModule fs, VkFormat color, VkFormat depth) {
    VkPipelineShaderStageCreateInfo st[2] = {
        {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, NULL, 0, VK_SHADER_STAGE_VERTEX_BIT, vs, "main"},
        {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, NULL, 0, VK_SHADER_STAGE_FRAGMENT_BIT, fs, "main"}};
    VkPipelineVertexInputStateCreateInfo vi = {VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
    VkPipelineInputAssemblyStateCreateInfo ia = {VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO, NULL, 0, VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST};
    VkPipelineViewportStateCreateInfo vp = {VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
    VkPipelineRasterizationStateCreateInfo rs = {VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO, NULL, 0, 0, 0,
                                                 VK_POLYGON_MODE_FILL, VK_CULL_MODE_NONE, VK_FRONT_FACE_COUNTER_CLOCKWISE, 0, 0, 0, 0, 1.0f};
    VkPipelineMultisampleStateCreateInfo ms = {VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO, NULL, 0, VK_SAMPLE_COUNT_1_BIT};
    VkPipelineDepthStencilStateCreateInfo ds = {VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO, NULL, 0,
                                                depth != VK_FORMAT_UNDEFINED, depth != VK_FORMAT_UNDEFINED, VK_COMPARE_OP_ALWAYS};
    VkPipelineColorBlendAttachmentState ba = {0, 0, 0, 0, 0, 0, 0, 0xf};
    VkPipelineColorBlendStateCreateInfo cb = {VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO, NULL, 0, 0, 0, 1, &ba};
    VkDynamicState dyn[2] = {VK_DYNAMIC_STATE_VIEWPORT_WITH_COUNT, VK_DYNAMIC_STATE_SCISSOR_WITH_COUNT};
    VkPipelineDynamicStateCreateInfo dy = {VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO, NULL, 0, 2, dyn};
    VkPipelineRenderingCreateInfo rc = {VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO, NULL, 0, 1, &color, depth,
                                        depth == VK_FORMAT_D32_SFLOAT_S8_UINT ? depth : VK_FORMAT_UNDEFINED};
    VkGraphicsPipelineCreateInfo ci = {VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO, &rc, 0, 2, st, &vi, &ia, NULL, &vp, &rs, &ms,
                                       &ds, &cb, &dy, layout};
    VkPipeline p;
    CK(vkCreateGraphicsPipelines(dev, VK_NULL_HANDLE, 1, &ci, NULL, &p));
    return p;
}

// expected predicate of cell (cx, cy) for a gx x gy grid, dilation dx/dy: the layer's rule, re-derived here
static bool tile_cov(int tx, int ty) { // a tile holds a quad pixel
    int x0 = tx * 16, y0 = ty * 16;
    return x0 < QX1 && x0 + 16 > QX0 && y0 < QY1 && y0 + 16 > QY0;
}
static bool cell_on(int cx, int cy, int gx, int gy, int dx, int dy) {
    int gx0 = cx * gx / NX, gx1 = (cx + 1) * gx / NX, gy0 = cy * gy / NY, gy1 = (cy + 1) * gy / NY;
    if (gx1 <= gx0 || gy1 <= gy0) return false;
    int x0 = gx0 * W / gx - dx, y0 = gy0 * H / gy - dy, x1 = (gx1 * W + gx - 1) / gx + dx, y1 = (gy1 * H + gy - 1) / gy + dy;
    if (x0 < 0) x0 = 0;
    if (y0 < 0) y0 = 0;
    if (x1 > W) x1 = W;
    if (y1 > H) y1 = H;
    for (int ty = y0 / 16; ty < (y1 + 15) / 16; ty++)
        for (int tx = x0 / 16; tx < (x1 + 15) / 16; tx++)
            if (tile_cov(tx, ty)) return true;
    return false;
}

int main(int argc, char **argv) {
    const char *mode = argc > 1 ? argv[1] : "on";
    bool trim = !strcmp(mode, "on"), many = !strcmp(mode, "many");
    char hs[64];
    snprintf(hs, sizeof hs, "%08x", fnv(ctt_probe_cs, sizeof ctt_probe_cs));
    setenv("MALI_COMPAT_COVER_TRIM_CS", hs, 1);
    snprintf(hs, sizeof hs, "%08x", fnv(ctt_comp_fs, sizeof ctt_comp_fs));
    setenv("MALI_COMPAT_COVER_TRIM_FS", hs, 1);
    if (strcmp(mode, "off")) setenv("MALI_COMPAT_COVER_TRIM", "1", 1);
    if (!strcmp(mode, "toggle")) setenv("MALI_COMPAT_COVER_TRIM_TOGGLE", "/nonexistent/ct-toggle", 1);
    setenv("MALI_COMPAT_COVER_TRIM_ARM", "1:16", 1);
    setenv("MALI_COMPAT_COVER_TRIM_GRID", "8x4", 1);
    setenv("MALI_COMPAT_COVER_TRIM_MARGIN", "8", 1);
    setenv("MALI_COMPAT_COVER_TRIM_K", "2", 1);

    VkApplicationInfo app = {VK_STRUCTURE_TYPE_APPLICATION_INFO, NULL, "ctt", 1, NULL, 0, VK_API_VERSION_1_3};
    const char *layers[1] = {"VK_LAYER_DROIDDECK_mali_compat"};
    VkInstanceCreateInfo ici = {VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, NULL, 0, &app, 1, layers};
    VkInstance inst;
    CK(vkCreateInstance(&ici, NULL, &inst));
    uint32_t n = 1;
    vkEnumeratePhysicalDevices(inst, &n, &pd);
    VkPhysicalDeviceVulkan13Features f13 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
    f13.dynamicRendering = VK_TRUE;
    VkPhysicalDeviceVulkan12Features f12 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES, &f13};
    f12.bufferDeviceAddress = VK_TRUE;
    VkPhysicalDeviceExtendedDynamicStateFeaturesEXT eds = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTENDED_DYNAMIC_STATE_FEATURES_EXT, &f12, VK_TRUE};
    float prio = 1;
    VkDeviceQueueCreateInfo qci = {VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO, NULL, 0, 0, 1, &prio};
    VkDeviceCreateInfo dci = {VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, &eds, 0, 1, &qci};
    CK(vkCreateDevice(pd, &dci, NULL, &dev));
    VkQueue q;
    vkGetDeviceQueue(dev, 0, 0, &q);

    VkImageView dv, cv, lv;
    VkImage dimg = image(VK_FORMAT_D32_SFLOAT_S8_UINT, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT, &dv,
                         VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT);
    VkImage cimg = image(VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT, &cv, VK_IMAGE_ASPECT_COLOR_BIT);
    VkImage limg = image(VK_FORMAT_R8_UNORM, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT, &lv, VK_IMAGE_ASPECT_COLOR_BIT);
    (void)cimg;
    void *mmap_, *lmap;
    VkDeviceAddress maddr;
    VkBuffer mbuf = buffer(GX * GY * 4, &mmap_, &maddr);
    VkBuffer lbuf = buffer(W * H, &lmap, NULL);
    (void)mbuf;

    VkPushConstantRange gpr = {VK_SHADER_STAGE_VERTEX_BIT, 0, 16};
    VkPipelineLayoutCreateInfo gpl = {VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, NULL, 0, 0, NULL, 1, &gpr};
    VkPipelineLayout glay, clay;
    CK(vkCreatePipelineLayout(dev, &gpl, NULL, &glay));
    VkPushConstantRange cpr = {VK_SHADER_STAGE_COMPUTE_BIT, 0, 16};
    VkPipelineLayoutCreateInfo cpl = {VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, NULL, 0, 0, NULL, 1, &cpr};
    CK(vkCreatePipelineLayout(dev, &cpl, NULL, &clay));
    VkPipeline gp = gfx(glay, module(ctt_quad_vs, sizeof ctt_quad_vs), module(ctt_gbuf_fs, sizeof ctt_gbuf_fs),
                        VK_FORMAT_R8G8B8A8_UNORM, VK_FORMAT_D32_SFLOAT_S8_UINT);
    VkPipeline cp_ = gfx(glay, module(ctt_full_vs, sizeof ctt_full_vs), module(ctt_comp_fs, sizeof ctt_comp_fs),
                         VK_FORMAT_R8_UNORM, VK_FORMAT_UNDEFINED);
    VkComputePipelineCreateInfo cci = {VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO, NULL, 0,
        {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, NULL, 0, VK_SHADER_STAGE_COMPUTE_BIT,
         module(ctt_probe_cs, sizeof ctt_probe_cs), "main"}, clay};
    VkPipeline pp;
    CK(vkCreateComputePipelines(dev, VK_NULL_HANDLE, 1, &cci, NULL, &pp));

    VkCommandPoolCreateInfo pci = {VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO, NULL, 0, 0};
    VkCommandPool pool;
    CK(vkCreateCommandPool(dev, &pci, NULL, &pool));
    VkCommandBufferAllocateInfo cai = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, NULL, pool, VK_COMMAND_BUFFER_LEVEL_PRIMARY, 1};
    VkCommandBuffer cmd;
    CK(vkAllocateCommandBuffers(dev, &cai, &cmd));
    VkCommandBufferBeginInfo bi = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, NULL, VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT};
    CK(vkBeginCommandBuffer(cmd, &bi));

    VkImageMemoryBarrier ib[3] = {
        {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER, NULL, 0, VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT, VK_IMAGE_LAYOUT_UNDEFINED,
         VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL, VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED, dimg,
         {VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT, 0, 1, 0, 1}},
        {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER, NULL, 0, VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, VK_IMAGE_LAYOUT_UNDEFINED,
         VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED, cimg, {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}},
        {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER, NULL, 0, VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, VK_IMAGE_LAYOUT_UNDEFINED,
         VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED, limg, {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}}};
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, NULL, 0, NULL, 3, ib);

    // pass A: the small G-buffer pass
    VkRenderingAttachmentInfo ca = {VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO, NULL, cv, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                                    0, NULL, 0, VK_ATTACHMENT_LOAD_OP_CLEAR, VK_ATTACHMENT_STORE_OP_STORE};
    VkRenderingAttachmentInfo da = {VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO, NULL, dv, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                                    0, NULL, 0, VK_ATTACHMENT_LOAD_OP_CLEAR, VK_ATTACHMENT_STORE_OP_STORE};
    da.clearValue.depthStencil.depth = 0.0f;
    VkRenderingInfo ri = {VK_STRUCTURE_TYPE_RENDERING_INFO, NULL, 0, {{0, 0}, {W, H}}, 1, 0, 1, &ca, &da, &da};
    vkCmdBeginRendering(cmd, &ri);
    VkViewport vpt = {0, 0, W, H, 0, 1};
    VkRect2D full = {{0, 0}, {W, H}};
    vkCmdSetViewportWithCount(cmd, 1, &vpt);
    vkCmdSetScissorWithCount(cmd, 1, &full);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, gp);
    float rect[4] = {QX0 * 2.0f / W - 1, QY0 * 2.0f / H - 1, QX1 * 2.0f / W - 1, QY1 * 2.0f / H - 1};
    vkCmdPushConstants(cmd, glay, VK_SHADER_STAGE_VERTEX_BIT, 0, 16, rect);
    for (int i = 0; i < (many ? 20 : 1); i++) vkCmdDraw(cmd, 6, 1, 0, 0);
    vkCmdEndRendering(cmd);

    // the listed probe dispatch: the app's pipeline and push constants must survive the layer's own passes
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pp);
    uint32_t push[4] = {(uint32_t)maddr, (uint32_t)(maddr >> 32), GX, 41};
    vkCmdPushConstants(cmd, clay, VK_SHADER_STAGE_COMPUTE_BIT, 0, 16, push);
    vkCmdDispatch(cmd, GX, GY, 1);

    // the listed composite draw
    VkRenderingAttachmentInfo la = {VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO, NULL, lv, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                                    0, NULL, 0, VK_ATTACHMENT_LOAD_OP_CLEAR, VK_ATTACHMENT_STORE_OP_STORE};
    VkRenderingInfo ri2 = {VK_STRUCTURE_TYPE_RENDERING_INFO, NULL, 0, {{0, 0}, {W, H}}, 1, 0, 1, &la, NULL, NULL};
    vkCmdBeginRendering(cmd, &ri2);
    vkCmdSetViewportWithCount(cmd, 1, &vpt);
    vkCmdSetScissorWithCount(cmd, 1, &full);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, cp_);
    vkCmdDraw(cmd, 3, 1, 0, 0);
    vkCmdEndRendering(cmd);

    VkImageMemoryBarrier lb = {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER, NULL, VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT,
                               VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_QUEUE_FAMILY_IGNORED,
                               VK_QUEUE_FAMILY_IGNORED, limg, {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, NULL, 0, NULL, 1, &lb);
    VkBufferImageCopy rg = {0, 0, 0, {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}, {0, 0, 0}, {W, H, 1}};
    vkCmdCopyImageToBuffer(cmd, limg, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, lbuf, 1, &rg);
    VkMemoryBarrier hb = {VK_STRUCTURE_TYPE_MEMORY_BARRIER, NULL, VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_HOST_READ_BIT};
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &hb, 0, NULL, 0, NULL);
    CK(vkEndCommandBuffer(cmd));
    VkSubmitInfo si = {VK_STRUCTURE_TYPE_SUBMIT_INFO, NULL, 0, NULL, NULL, 1, &cmd};
    CK(vkQueueSubmit(q, 1, &si, VK_NULL_HANDLE));
    CK(vkQueueWaitIdle(q));

    const uint32_t *marks = mmap_;
    const uint8_t *px = lmap;
    int bad = 0, ran = 0, lit = 0;
    for (int gy = 0; gy < GY; gy++)
        for (int gx = 0; gx < GX; gx++) {
            int cx = 0, cy = 0;
            while ((cx + 1) * GX / NX <= gx) cx++;
            while ((cy + 1) * GY / NY <= gy) cy++;
            bool want = !trim || cell_on(cx, cy, GX, GY, MARGIN + K * ((W + GX - 1) / GX), MARGIN + K * ((H + GY - 1) / GY));
            uint32_t m = marks[gy * GX + gx];
            ran += m != 0;
            if ((want ? 42u : 0u) != m) { if (bad++ < 5) fprintf(stderr, "group %d,%d: mark %u, want %u\n", gx, gy, m, want ? 42u : 0u); }
        }
    for (int y = 0; y < H; y++)
        for (int x = 0; x < W; x++) {
            int cx = 0, cy = 0;
            while ((cx + 1) * W / NX <= x) cx++;
            while ((cy + 1) * H / NY <= y) cy++;
            bool want = !trim || cell_on(cx, cy, W, H, MARGIN, MARGIN);
            lit += px[y * W + x] != 0;
            if ((want ? 255 : 0) != px[y * W + x]) { if (bad++ < 10) fprintf(stderr, "pixel %d,%d: %u, want %u\n", x, y, px[y * W + x], want ? 255 : 0); }
        }
    printf("cover-trim-test %s: %s (groups run %d of %d, composite pixels %d of %d, mismatches %d)\n", mode, bad ? "FAIL" : "PASS",
           ran, GX * GY, lit, W * H, bad);
    vkDeviceWaitIdle(dev);
    vkDestroyDevice(dev, NULL);
    vkDestroyInstance(inst, NULL);
    return bad ? 1 : 0;
}

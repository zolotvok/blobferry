// SPDX-License-Identifier: GPL-3.0-or-later
// sync-diag-test.c: drives the compat layer's sync-diag hooks (MALI_COMPAT_FULL_BARRIERS / SUBMIT_IDLE /
// GT_SPLIT_SKIP / SYNC_CTL) on lavapipe. Each iteration records, in two command buffers of one submit:
//   cb1: fill, app barrier v1, app barrier v2 (layout UNDEFINED -> COLOR_ATTACHMENT), plain pass, SUSPENDING part
//   cb2: RESUMING part, dispatch, copy
// and waits for it. Usage: sync-diag-test <iterations> <ms between iterations> [ctl file "<when-iter>:<text>" ...]
// [keeper, Claude Opus 5.5] 2026-11-05
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <vulkan/vulkan.h>

#define CK(x) do { VkResult r_ = (x); if (r_ != VK_SUCCESS) { fprintf(stderr, "%s failed %d\n", #x, r_); exit(1); } } while (0)

// #version 450 / layout(local_size_x = 1) in; void main() {}
static const uint32_t spv[] = {
    0x07230203, 0x00010000, 0x00080001, 0x00000006, 0x00000000, 0x00020011, 0x00000001, 0x0003000e, 0x00000000,
    0x00000001, 0x0005000f, 0x00000005, 0x00000001, 0x6e69616d, 0x00000000, 0x00060010, 0x00000001, 0x00000011,
    0x00000001, 0x00000001, 0x00000001, 0x00020013, 0x00000002, 0x00030021, 0x00000003, 0x00000002, 0x00050036,
    0x00000002, 0x00000001, 0x00000000, 0x00000003, 0x000200f8, 0x00000004, 0x000100fd, 0x00010038};

int main(int argc, char **argv) {
    int iters = argc > 1 ? atoi(argv[1]) : 20, gap_ms = argc > 2 ? atoi(argv[2]) : 50;
    VkApplicationInfo ai = {VK_STRUCTURE_TYPE_APPLICATION_INFO, NULL, "sync-diag-test", 1, "sync-diag-test", 1, VK_API_VERSION_1_3};
    VkInstanceCreateInfo ici = {VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, NULL, 0, &ai};
    VkInstance inst;
    CK(vkCreateInstance(&ici, NULL, &inst));
    uint32_t n = 1;
    VkPhysicalDevice pd;
    vkEnumeratePhysicalDevices(inst, &n, &pd);
    if (!n) { fprintf(stderr, "no device\n"); return 1; }
    float prio = 1;
    VkDeviceQueueCreateInfo qci = {VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO, NULL, 0, 0, 1, &prio};
    VkPhysicalDeviceVulkan13Features f13 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
    f13.synchronization2 = VK_TRUE;
    f13.dynamicRendering = VK_TRUE;
    VkPhysicalDeviceVulkan12Features f12 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES, &f13};
    f12.hostQueryReset = VK_TRUE;
    f12.timelineSemaphore = VK_TRUE;
    VkDeviceCreateInfo dci = {VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, &f12, 0, 1, &qci};
    VkDevice dev;
    CK(vkCreateDevice(pd, &dci, NULL, &dev));
    PFN_vkCmdPipelineBarrier2 pb2 = (PFN_vkCmdPipelineBarrier2)vkGetDeviceProcAddr(dev, "vkCmdPipelineBarrier2");
    PFN_vkCmdBeginRendering br = (PFN_vkCmdBeginRendering)vkGetDeviceProcAddr(dev, "vkCmdBeginRendering");
    PFN_vkCmdEndRendering er = (PFN_vkCmdEndRendering)vkGetDeviceProcAddr(dev, "vkCmdEndRendering");
    PFN_vkQueueSubmit qs = (PFN_vkQueueSubmit)vkGetDeviceProcAddr(dev, "vkQueueSubmit");
    PFN_vkCmdFillBuffer fill = (PFN_vkCmdFillBuffer)vkGetDeviceProcAddr(dev, "vkCmdFillBuffer");
    PFN_vkCmdCopyBuffer cpy = (PFN_vkCmdCopyBuffer)vkGetDeviceProcAddr(dev, "vkCmdCopyBuffer");
    PFN_vkCmdDispatch disp = (PFN_vkCmdDispatch)vkGetDeviceProcAddr(dev, "vkCmdDispatch");
    PFN_vkCmdPipelineBarrier pb = (PFN_vkCmdPipelineBarrier)vkGetDeviceProcAddr(dev, "vkCmdPipelineBarrier");
    PFN_vkBeginCommandBuffer bcb = (PFN_vkBeginCommandBuffer)vkGetDeviceProcAddr(dev, "vkBeginCommandBuffer");
    if (!pb2 || !br || !er || !qs || !fill || !cpy || !disp || !pb || !bcb) { fprintf(stderr, "entry point missing\n"); return 1; }
    VkQueue q;
    vkGetDeviceQueue(dev, 0, 0, &q);

    VkPhysicalDeviceMemoryProperties mp;
    vkGetPhysicalDeviceMemoryProperties(pd, &mp);
    VkBufferCreateInfo bi = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, NULL, 0, 65536,
                             VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT};
    VkBuffer buf[2];
    VkDeviceMemory bm[2];
    for (int i = 0; i < 2; i++) {
        CK(vkCreateBuffer(dev, &bi, NULL, &buf[i]));
        VkMemoryRequirements mr;
        vkGetBufferMemoryRequirements(dev, buf[i], &mr);
        uint32_t t = 0;
        while (!(mr.memoryTypeBits & (1u << t))) t++;
        VkMemoryAllocateInfo mai = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, NULL, mr.size, t};
        CK(vkAllocateMemory(dev, &mai, NULL, &bm[i]));
        CK(vkBindBufferMemory(dev, buf[i], bm[i], 0));
    }
    VkImageCreateInfo ii = {VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, NULL, 0, VK_IMAGE_TYPE_2D, VK_FORMAT_R8G8B8A8_UNORM, {64, 64, 1}, 1, 1,
                            VK_SAMPLE_COUNT_1_BIT, VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT, VK_SHARING_MODE_EXCLUSIVE};
    VkImage img;
    CK(vkCreateImage(dev, &ii, NULL, &img));
    VkMemoryRequirements imr;
    vkGetImageMemoryRequirements(dev, img, &imr);
    uint32_t it = 0;
    while (!(imr.memoryTypeBits & (1u << it))) it++;
    VkMemoryAllocateInfo imai = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, NULL, imr.size, it};
    VkDeviceMemory im;
    CK(vkAllocateMemory(dev, &imai, NULL, &im));
    CK(vkBindImageMemory(dev, img, im, 0));
    VkImageViewCreateInfo vi = {VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, NULL, 0, img, VK_IMAGE_VIEW_TYPE_2D, VK_FORMAT_R8G8B8A8_UNORM,
                                {0}, {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
    VkImageView view;
    CK(vkCreateImageView(dev, &vi, NULL, &view));

    VkShaderModuleCreateInfo smi = {VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, NULL, 0, sizeof spv, spv};
    VkShaderModule sm;
    CK(vkCreateShaderModule(dev, &smi, NULL, &sm));
    VkPipelineLayoutCreateInfo pli = {VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    VkPipelineLayout pl;
    CK(vkCreatePipelineLayout(dev, &pli, NULL, &pl));
    VkComputePipelineCreateInfo cpi = {VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO, NULL, 0,
                                       {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, NULL, 0, VK_SHADER_STAGE_COMPUTE_BIT, sm, "main"}, pl};
    VkPipeline pipe;
    CK(vkCreateComputePipelines(dev, VK_NULL_HANDLE, 1, &cpi, NULL, &pipe));

    VkCommandPoolCreateInfo cpci = {VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO, NULL, VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT, 0};
    VkCommandPool pool;
    CK(vkCreateCommandPool(dev, &cpci, NULL, &pool));
    VkCommandBufferAllocateInfo cai = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, NULL, pool, VK_COMMAND_BUFFER_LEVEL_PRIMARY, 2};
    VkCommandBuffer cb[2];
    CK(vkAllocateCommandBuffers(dev, &cai, cb));
    VkFenceCreateInfo fci = {VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    VkFence fence;
    CK(vkCreateFence(dev, &fci, NULL, &fence));
    VkSemaphoreTypeCreateInfo sti = {VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO, NULL, VK_SEMAPHORE_TYPE_TIMELINE, 0};
    VkSemaphoreCreateInfo sci = {VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO, &sti};
    VkSemaphore tl;
    CK(vkCreateSemaphore(dev, &sci, NULL, &tl));

    VkRenderingAttachmentInfo att = {VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO, NULL, view, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
    att.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    att.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    VkRenderingInfo ri = {VK_STRUCTURE_TYPE_RENDERING_INFO, NULL, 0, {{0, 0}, {64, 64}}, 1, 0, 1, &att};
    for (int k = 0; k < iters; k++) {
        for (int a = 3; a < argc; a++) { // "<iteration>:<ctl file text>" written into $SYNC_TEST_CTL at that iteration
            int when = atoi(argv[a]);
            const char *txt = strchr(argv[a], ':');
            if (when == k && txt && getenv("SYNC_TEST_CTL")) {
                FILE *f = fopen(getenv("SYNC_TEST_CTL"), "w");
                if (f) { fprintf(f, "%s\n", txt + 1); fclose(f); }
                usleep(600000); // the layer polls every 250 ms
            }
        }
        VkCommandBufferBeginInfo cbi = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, NULL, VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT};
        CK(bcb(cb[0], &cbi));
        fill(cb[0], buf[0], 0, VK_WHOLE_SIZE, 0x12345678u);
        VkMemoryBarrier mb = {VK_STRUCTURE_TYPE_MEMORY_BARRIER, NULL, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT};
        pb(cb[0], VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &mb, 0, NULL, 0, NULL);
        VkImageMemoryBarrier2 ib = {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2, NULL, VK_PIPELINE_STAGE_2_NONE, 0,
                                    VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
                                    VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_QUEUE_FAMILY_IGNORED,
                                    VK_QUEUE_FAMILY_IGNORED, img, {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
        VkDependencyInfo di = {VK_STRUCTURE_TYPE_DEPENDENCY_INFO, NULL, 0, 0, NULL, 0, NULL, 1, &ib};
        pb2(cb[0], &di);
        ri.flags = 0;
        br(cb[0], &ri);
        er(cb[0]);
        att.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
        ri.flags = VK_RENDERING_SUSPENDING_BIT;
        br(cb[0], &ri);
        er(cb[0]);
        CK(vkEndCommandBuffer(cb[0]));
        CK(bcb(cb[1], &cbi));
        ri.flags = VK_RENDERING_RESUMING_BIT;
        br(cb[1], &ri);
        er(cb[1]);
        att.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        vkCmdBindPipeline(cb[1], VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
        disp(cb[1], 1, 1, 1);
        VkBufferCopy bc = {0, 0, 4096};
        cpy(cb[1], buf[0], buf[1], 1, &bc);
        CK(vkEndCommandBuffer(cb[1]));
        VkSubmitInfo si = {VK_STRUCTURE_TYPE_SUBMIT_INFO, NULL, 0, NULL, NULL, 2, cb};
        CK(qs(q, 1, &si, fence));
        CK(vkWaitForFences(dev, 1, &fence, VK_TRUE, 5000000000ull));
        CK(vkResetFences(dev, 1, &fence));
        if (k == 3 && getenv("SYNC_TEST_WBS")) { // wait-before-signal: the queue blocks until the host signals
            uint64_t v = 1;
            VkTimelineSemaphoreSubmitInfo tsi = {VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO, NULL, 1, &v};
            VkPipelineStageFlags ws = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
            VkSubmitInfo wsi = {VK_STRUCTURE_TYPE_SUBMIT_INFO, &tsi, 1, &tl, &ws, 0, NULL};
            CK(qs(q, 1, &wsi, VK_NULL_HANDLE)); // returns after the layer's capped wait
            VkSemaphoreSignalInfo ssi = {VK_STRUCTURE_TYPE_SEMAPHORE_SIGNAL_INFO, NULL, tl, 1};
            CK(vkSignalSemaphore(dev, &ssi));
            printf("wbs signaled\n");
        }
        usleep(gap_ms * 1000);
    }
    uint32_t out = 0;
    void *p;
    CK(vkDeviceWaitIdle(dev));
    // the copy target must hold the fill pattern (the hooks kept the commands)
    VkMemoryPropertyFlags fl = 0;
    VkMemoryRequirements mr;
    vkGetBufferMemoryRequirements(dev, buf[1], &mr);
    for (uint32_t t = 0; t < mp.memoryTypeCount; t++) if (mr.memoryTypeBits & (1u << t)) { fl = mp.memoryTypes[t].propertyFlags; break; }
    if (fl & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) {
        CK(vkMapMemory(dev, bm[1], 0, 4, 0, &p));
        out = *(uint32_t *)p;
        vkUnmapMemory(dev, bm[1]);
        printf("copy result %08x\n", out);
    }
    sleep(argc > 3 || getenv("SYNC_TEST_TAIL") ? 2 : 0); // let the last report line come out
    vkDestroyFence(dev, fence, NULL);
    vkDestroySemaphore(dev, tl, NULL);
    vkDestroyCommandPool(dev, pool, NULL);
    vkDestroyPipeline(dev, pipe, NULL);
    vkDestroyPipelineLayout(dev, pl, NULL);
    vkDestroyShaderModule(dev, sm, NULL);
    vkDestroyImageView(dev, view, NULL);
    vkDestroyImage(dev, img, NULL);
    vkFreeMemory(dev, im, NULL);
    for (int i = 0; i < 2; i++) { vkDestroyBuffer(dev, buf[i], NULL); vkFreeMemory(dev, bm[i], NULL); }
    vkDestroyDevice(dev, NULL);
    vkDestroyInstance(inst, NULL);
    printf("DONE iters %d\n", iters);
    return 0;
}

// SPDX-License-Identifier: GPL-3.0-or-later
// bc-probe: does the compat layer's BC emulation decode an upload? Creates BC1/BC3 images (the layer makes them
// RGBA8), uploads known blocks with vkCmdCopyBufferToImage, reads the RGBA8 texels back. [Claude Opus 5.5] 2026-10-26
// Run with the compat layer only:
//   VK_DRIVER_FILES=<bridge icd.json> HYBRIS_USE_VENDOR_NAMESPACE=1 ENABLE_MALI_COMPAT=1 \
//   VK_LOADER_LAYERS_ENABLE=*mali_compat VK_LOADER_LAYERS_DISABLE=~implicit~ bc-probe
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <vulkan/vulkan.h>
#include "bc-probe-app_spv.h"

static VkDevice dev;
static VkQueue queue;
static VkCommandPool pool;

static VkDeviceMemory bind_mem(VkMemoryRequirements req, uint32_t want) {
    VkPhysicalDeviceMemoryProperties mp;
    (void)mp;
    uint32_t type = 0;
    while (!(req.memoryTypeBits & (1u << type))) type++;
    (void)want;
    VkMemoryAllocateFlagsInfo fi = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO, NULL, VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT};
    VkMemoryAllocateInfo ai = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, &fi, req.size, type};
    VkDeviceMemory m;
    vkAllocateMemory(dev, &ai, NULL, &m);
    return m;
}

static VkBuffer buffer(VkDeviceSize size, VkBufferUsageFlags usage, VkDeviceMemory *m, void **map) {
    VkBufferCreateInfo ci = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, NULL, 0, size, usage};
    VkBuffer b;
    vkCreateBuffer(dev, &ci, NULL, &b);
    VkMemoryRequirements req;
    vkGetBufferMemoryRequirements(dev, b, &req);
    *m = bind_mem(req, 0);
    vkBindBufferMemory(dev, b, *m, 0);
    vkMapMemory(dev, *m, 0, VK_WHOLE_SIZE, 0, map);
    return b;
}

static int failures;

// Samples (x, y) of layer 0..; with expect != NULL each sample's texel bytes are compared with expect[layer * 4 + i].
static void test_ex(VkFormat fmt, const char *name, const uint8_t *blocks, size_t block_bytes, uint32_t layers,
                    uint32_t texel_B, const uint32_t at[4][2], const uint64_t *expect) {
    const uint32_t W = 8, H = 8; // 2x2 blocks per layer
    VkImageCreateInfo ici = {VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, NULL, 0, VK_IMAGE_TYPE_2D, fmt, {W, H, 1}, 1, layers,
                             VK_SAMPLE_COUNT_1_BIT, VK_IMAGE_TILING_OPTIMAL,
                             VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_SAMPLED_BIT};
    VkImage img;
    VkResult r = vkCreateImage(dev, &ici, NULL, &img);
    if (r != VK_SUCCESS) { printf("%s: vkCreateImage -> %d\n", name, r); return; }
    VkMemoryRequirements req;
    vkGetImageMemoryRequirements(dev, img, &req);
    VkDeviceMemory im = bind_mem(req, 0);
    vkBindImageMemory(dev, img, im, 0);

    VkDeviceMemory sm, rm;
    void *sp, *rp;
    VkBuffer src = buffer(4 * block_bytes * layers, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, &sm, &sp);
    VkBuffer rb = buffer(W * H * texel_B * layers, VK_BUFFER_USAGE_TRANSFER_DST_BIT, &rm, &rp);
    for (uint32_t l = 0; l < layers; l++) memcpy((uint8_t *)sp + l * 4 * block_bytes, blocks + (l & 1) * 0, 4 * block_bytes);
    if (layers > 1) memcpy((uint8_t *)sp + 4 * block_bytes, blocks + 4 * block_bytes, 4 * block_bytes); // layer 1: second set
    memset(rp, 0xcd, W * H * texel_B * layers);

    VkCommandBuffer cb;
    VkCommandBufferAllocateInfo cai = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, NULL, pool, VK_COMMAND_BUFFER_LEVEL_PRIMARY, 1};
    vkAllocateCommandBuffers(dev, &cai, &cb);
    VkCommandBufferBeginInfo bi = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    vkBeginCommandBuffer(cb, &bi);
    VkImageSubresourceRange rng = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, layers};
    VkImageMemoryBarrier b1 = {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER, NULL, 0, VK_ACCESS_TRANSFER_WRITE_BIT,
                               VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 0, 0, img, rng};
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_HOST_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, NULL, 0, NULL, 1, &b1);
    VkBufferImageCopy reg = {0, 0, 0, {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, layers}, {0, 0, 0}, {W, H, 1}};
    vkCmdCopyBufferToImage(cb, src, img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &reg);
    VkImageMemoryBarrier b2 = {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER, NULL, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT,
                               VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, 0, 0, img, rng};
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, NULL, 0, NULL, 1, &b2);
    vkCmdCopyImageToBuffer(cb, img, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, rb, 1, &reg);
    VkMemoryBarrier hb = {VK_STRUCTURE_TYPE_MEMORY_BARRIER, NULL, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_HOST_READ_BIT};
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &hb, 0, NULL, 0, NULL);
    vkEndCommandBuffer(cb);
    VkSubmitInfo si = {VK_STRUCTURE_TYPE_SUBMIT_INFO, NULL, 0, NULL, NULL, 1, &cb};
    r = vkQueueSubmit(queue, 1, &si, VK_NULL_HANDLE);
    VkResult w = vkQueueWaitIdle(queue);
    const uint8_t *px = rp;
    for (uint32_t l = 0; l < layers; l++) {
        printf("%s layer %u: submit %d wait %d; texel bytes at", name, l, r, w);
        bool ok = true;
        for (int i = 0; i < 4; i++) {
            const uint8_t *p = px + (l * W * H + at[i][1] * W + at[i][0]) * texel_B;
            uint64_t v = 0;
            for (uint32_t b = 0; b < texel_B; b++) v = (v << 8) | p[b];
            printf(" (%u,%u) %0*llx", at[i][0], at[i][1], (int)texel_B * 2, (unsigned long long)v);
            if (expect && v != expect[l * 4 + i]) ok = false;
        }
        if (expect) {
            printf("  -> %s", ok ? "PASS" : "FAIL, expected");
            if (!ok)
                for (int i = 0; i < 4; i++) printf(" %0*llx", (int)texel_B * 2, (unsigned long long)expect[l * 4 + i]);
            failures += !ok;
        }
        printf("\n");
    }
    // Freeing the command buffer hands its decode staging buffer back to the layer's pool, so the next test decodes
    // into a reused (stale) buffer.
    vkFreeCommandBuffers(dev, pool, 1, &cb);
}

static void test(VkFormat fmt, const char *name, const uint8_t *blocks, size_t block_bytes, uint32_t layers) {
    static const uint32_t corners[4][2] = {{0, 0}, {4, 0}, {0, 4}, {4, 4}};
    test_ex(fmt, name, blocks, block_bytes, layers, 4, corners, NULL);
}

// LSB-first bit writer for hand-built BC6H/BC7 blocks.
static void put_bits(uint8_t *blk, unsigned *pos, uint32_t v, unsigned n) {
    for (unsigned i = 0; i < n; i++, (*pos)++)
        if (v >> i & 1) blk[*pos / 8] |= (uint8_t)(1u << (*pos % 8));
}

// BC7 mode 6: one subset, RGBA 7-bit endpoints + p-bits, 4-bit indices (texel 0 has 3 bits). Every index = idx
// (the anchor gets idx & 7).
static void bc7_mode6(uint8_t blk[16], const uint8_t e0[4], const uint8_t e1[4], unsigned p0, unsigned p1, unsigned idx) {
    memset(blk, 0, 16);
    unsigned pos = 0;
    put_bits(blk, &pos, 1u << 6, 7);
    for (int c = 0; c < 4; c++) { put_bits(blk, &pos, e0[c], 7); put_bits(blk, &pos, e1[c], 7); }
    put_bits(blk, &pos, p0, 1);
    put_bits(blk, &pos, p1, 1);
    put_bits(blk, &pos, idx & 7, 3);
    for (int t = 1; t < 16; t++) put_bits(blk, &pos, idx, 4);
}

// BC6H mode 11 (5-bit code 00011): one region, untransformed 10-bit endpoints, 4-bit indices.
static void bc6_mode11(uint8_t blk[16], const uint16_t e0[3], const uint16_t e1[3], unsigned idx) {
    memset(blk, 0, 16);
    unsigned pos = 0;
    put_bits(blk, &pos, 3, 5);
    for (int c = 0; c < 3; c++) put_bits(blk, &pos, e0[c], 10);
    for (int c = 0; c < 3; c++) put_bits(blk, &pos, e1[c], 10);
    put_bits(blk, &pos, idx & 7, 3);
    for (int t = 1; t < 16; t++) put_bits(blk, &pos, idx, 4);
}

// BC6H UFLOAT reference for a solid 10-bit endpoint: unquantize, then the 31/64 scale to half-float bits.
static uint16_t bc6u_half(uint16_t e) {
    uint32_t unq = e == 0 ? 0 : e == 1023 ? 0xffff : (((uint32_t)e << 16) + 0x8000) >> 10;
    return (uint16_t)((unq * 31) >> 6);
}

// Little-endian texel bytes as the probe prints them (byte 0 first).
static uint64_t le_bytes(const uint8_t *b, unsigned n) {
    uint64_t v = 0;
    for (unsigned i = 0; i < n; i++) v = (v << 8) | b[i];
    return v;
}

// H1 check: the app binds its compute pipeline and pushes constants, records a BC upload (the layer binds its decoder
// and pushes its own constants), then dispatches without binding again. With the restore the app shader writes its
// value; without it the decoder runs instead and the word stays 0. [Claude Opus 5.5] 2026-10-26
static void test_restore(void) {
    VkPushConstantRange pr = {VK_SHADER_STAGE_COMPUTE_BIT, 0, 16};
    VkPipelineLayoutCreateInfo lci = {VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, NULL, 0, 0, NULL, 1, &pr};
    VkPipelineLayout layout;
    vkCreatePipelineLayout(dev, &lci, NULL, &layout);
    VkShaderModuleCreateInfo smi = {VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, NULL, 0, sizeof bc_probe_app_spv, bc_probe_app_spv};
    VkShaderModule sm;
    vkCreateShaderModule(dev, &smi, NULL, &sm);
    VkComputePipelineCreateInfo cpi = {VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO, NULL, 0,
                                       {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, NULL, 0, VK_SHADER_STAGE_COMPUTE_BIT, sm, "main"},
                                       layout};
    VkPipeline pipe;
    VkResult r = vkCreateComputePipelines(dev, VK_NULL_HANDLE, 1, &cpi, NULL, &pipe);
    if (r) { printf("restore: app pipeline -> %d\n", r); failures++; return; }

    VkImageCreateInfo ici = {VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, NULL, 0, VK_IMAGE_TYPE_2D, VK_FORMAT_BC1_RGBA_UNORM_BLOCK,
                             {8, 8, 1}, 1, 1, VK_SAMPLE_COUNT_1_BIT, VK_IMAGE_TILING_OPTIMAL,
                             VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT};
    VkImage img;
    vkCreateImage(dev, &ici, NULL, &img);
    VkMemoryRequirements req;
    vkGetImageMemoryRequirements(dev, img, &req);
    vkBindImageMemory(dev, img, bind_mem(req, 0), 0);
    VkDeviceMemory sm_mem, om;
    void *sp, *op;
    VkBuffer src = buffer(4 * 8, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, &sm_mem, &sp);
    memset(sp, 0, 32);
    VkBuffer out = buffer(64, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT, &om, &op);
    memset(op, 0, 64);
    VkBufferDeviceAddressInfo bai = {VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO, NULL, out};
    struct { uint64_t addr; uint32_t value, pad; } push = {vkGetBufferDeviceAddress(dev, &bai), 0x1234abcd, 0};

    VkCommandBuffer cb;
    VkCommandBufferAllocateInfo cai = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, NULL, pool, VK_COMMAND_BUFFER_LEVEL_PRIMARY, 1};
    vkAllocateCommandBuffers(dev, &cai, &cb);
    VkCommandBufferBeginInfo bi = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    vkBeginCommandBuffer(cb, &bi);
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
    vkCmdPushConstants(cb, layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, 16, &push);
    VkImageSubresourceRange rng = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    VkImageMemoryBarrier b1 = {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER, NULL, 0, VK_ACCESS_TRANSFER_WRITE_BIT,
                               VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 0, 0, img, rng};
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_HOST_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, NULL, 0, NULL, 1, &b1);
    VkBufferImageCopy reg = {0, 0, 0, {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}, {0, 0, 0}, {8, 8, 1}};
    vkCmdCopyBufferToImage(cb, src, img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &reg);
    vkCmdDispatch(cb, 1, 1, 1);
    VkMemoryBarrier hb = {VK_STRUCTURE_TYPE_MEMORY_BARRIER, NULL, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_HOST_READ_BIT};
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &hb, 0, NULL, 0, NULL);
    vkEndCommandBuffer(cb);
    VkSubmitInfo si = {VK_STRUCTURE_TYPE_SUBMIT_INFO, NULL, 0, NULL, NULL, 1, &cb};
    r = vkQueueSubmit(queue, 1, &si, VK_NULL_HANDLE);
    VkResult w = vkQueueWaitIdle(queue);
    uint32_t got = *(volatile uint32_t *)op;
    bool ok = got == 0x1234abcd;
    printf("restore (app compute pipeline + push constants after a BC upload): submit %d wait %d, word 0x%08x -> %s\n",
           r, w, got, ok ? "PASS" : "FAIL, expected 0x1234abcd");
    failures += !ok;
    vkFreeCommandBuffers(dev, pool, 1, &cb);
}

int main(void) {
    setvbuf(stdout, NULL, _IONBF, 0);
    VkApplicationInfo app = {VK_STRUCTURE_TYPE_APPLICATION_INFO, NULL, "bc-probe", 1, NULL, 0, VK_API_VERSION_1_3};
    VkInstanceCreateInfo ici = {VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, NULL, 0, &app};
    VkInstance inst;
    if (vkCreateInstance(&ici, NULL, &inst) != VK_SUCCESS) return 1;
    uint32_t n = 1;
    VkPhysicalDevice pd;
    vkEnumeratePhysicalDevices(inst, &n, &pd);
    VkPhysicalDeviceFeatures f;
    vkGetPhysicalDeviceFeatures(pd, &f);
    printf("textureCompressionBC reported: %u\n", f.textureCompressionBC);
    VkPhysicalDeviceFeatures en = {0};
    en.textureCompressionBC = VK_TRUE;
    en.shaderInt64 = VK_TRUE;
    float prio = 1;
    VkDeviceQueueCreateInfo q = {VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO, NULL, 0, 0, 1, &prio};
    VkDeviceCreateInfo dci = {VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, NULL, 0, 1, &q, 0, NULL, 0, NULL, &en};
    VkResult r = vkCreateDevice(pd, &dci, NULL, &dev);
    printf("vkCreateDevice -> %d\n", r);
    if (r) return 1;
    vkGetDeviceQueue(dev, 0, 0, &queue);
    VkCommandPoolCreateInfo pci = {VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO, NULL, 0, 0};
    vkCreateCommandPool(dev, &pci, NULL, &pool);

    // BC1: c0 = red (0xF800), c1 = blue (0x001F); block 0/3 indices 0 (red), block 1/2 indices 1 (blue).
    uint8_t bc1[4][8];
    for (int i = 0; i < 4; i++) {
        uint16_t c0 = 0xF800, c1 = 0x001F;
        uint32_t idx = (i == 1 || i == 2) ? 0x55555555u : 0;
        memcpy(&bc1[i][0], &c0, 2);
        memcpy(&bc1[i][2], &c1, 2);
        memcpy(&bc1[i][4], &idx, 4);
    }
    // 300 BC images first: the layer used to track only 256 and silently drop the rest.
    for (int i = 0; i < 300; i++) {
        VkImageCreateInfo ci = {VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, NULL, 0, VK_IMAGE_TYPE_2D, VK_FORMAT_BC1_RGBA_UNORM_BLOCK,
                                {64, 64, 1}, 1, 1, VK_SAMPLE_COUNT_1_BIT, VK_IMAGE_TILING_OPTIMAL,
                                VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT};
        VkImage tmp;
        vkCreateImage(dev, &ci, NULL, &tmp);
    }
    test(VK_FORMAT_BC1_RGBA_UNORM_BLOCK, "BC1", &bc1[0][0], 8, 1);
    // BC3: alpha a0 = 0x80 everywhere (indices 0), colour as BC1 above, green/red.
    uint8_t bc3[4][16];
    for (int i = 0; i < 4; i++) {
        memset(bc3[i], 0, 16);
        bc3[i][0] = 0x80;
        bc3[i][1] = 0x40;
        uint16_t c0 = 0x07E0, c1 = 0xF800; // green, red
        uint32_t idx = (i == 1 || i == 2) ? 0x55555555u : 0;
        memcpy(&bc3[i][8], &c0, 2);
        memcpy(&bc3[i][10], &c1, 2);
        memcpy(&bc3[i][12], &idx, 4);
    }
    test(VK_FORMAT_BC3_UNORM_BLOCK, "BC3", &bc3[0][0], 16, 1);
    // 2-layer array: layer 0 = the BC1 blocks, layer 1 = the same blocks with the colours swapped.
    uint8_t two[8][8];
    memcpy(two, bc1, sizeof bc1);
    for (int i = 0; i < 4; i++) {
        uint16_t c0 = 0x001F, c1 = 0xF800;
        memcpy(two[4 + i], bc1[i], 8);
        memcpy(&two[4 + i][0], &c0, 2);
        memcpy(&two[4 + i][2], &c1, 2);
    }
    test(VK_FORMAT_BC1_RGBA_UNORM_BLOCK, "BC1 array", &two[0][0], 8, 2);
    // Formats HZD uses most (BC7: 995 of its 1016 BC textures) and normal-map formats. Inner texels (1,1) (5,1)
    // (1,5) (5,5), i.e. one non-anchor texel per block. [Claude Opus 5.5] 2026-10-26
    static const uint32_t inner[4][2] = {{1, 1}, {5, 1}, {1, 5}, {5, 5}};
    {
        // BC7 mode 6. Blocks 0-2 solid (both endpoints equal), block 3 red 0 -> 127 with every index 15 (= endpoint 1).
        // Texel = endpoint << 1 | p-bit.
        uint8_t b7[4][16];
        const uint8_t red[4] = {127, 0, 0, 127}, green[4] = {0, 127, 0, 64}, blue[4] = {10, 20, 127, 127};
        const uint8_t lo[4] = {0, 0, 0, 127}, hi[4] = {127, 0, 0, 127};
        bc7_mode6(b7[0], red, red, 0, 0, 5);
        bc7_mode6(b7[1], green, green, 1, 1, 9);
        bc7_mode6(b7[2], blue, blue, 0, 0, 0);
        bc7_mode6(b7[3], lo, hi, 0, 0, 15);
        uint64_t ex[4];
        const uint8_t t0[4] = {0xfe, 0x00, 0x00, 0xfe};
        const uint8_t t1b[4] = {0x01, 0xff, 0x01, 0x81}, t2[4] = {20, 40, 0xfe, 0xfe}, t3[4] = {0xfe, 0x00, 0x00, 0xfe};
        ex[0] = le_bytes(t0, 4); ex[1] = le_bytes(t1b, 4); ex[2] = le_bytes(t2, 4); ex[3] = le_bytes(t3, 4);
        test_ex(VK_FORMAT_BC7_UNORM_BLOCK, "BC7 mode6", &b7[0][0], 16, 1, 4, inner, ex);
        test_ex(VK_FORMAT_BC7_SRGB_BLOCK, "BC7 mode6 SRGB", &b7[0][0], 16, 1, 4, inner, ex);
    }
    {
        // BC6H UFLOAT mode 11, solid blocks; decoded to RGBA16F (alpha 1.0 = 0x3c00).
        uint8_t b6[4][16];
        const uint16_t e[4][3] = {{1023, 0, 0}, {0, 512, 0}, {0, 0, 300}, {700, 700, 700}};
        uint64_t ex[4];
        for (int i = 0; i < 4; i++) {
            bc6_mode11(b6[i], e[i], e[i], 15);
            uint8_t t[8];
            for (int c = 0; c < 3; c++) { uint16_t h = bc6u_half(e[i][c]); t[c * 2] = h & 0xff; t[c * 2 + 1] = h >> 8; }
            t[6] = 0x00; t[7] = 0x3c;
            ex[i] = le_bytes(t, 8);
        }
        test_ex(VK_FORMAT_BC6H_UFLOAT_BLOCK, "BC6H mode11", &b6[0][0], 16, 1, 8, inner, ex);
    }
    {
        // BC4 UNORM: red0 > red1 (8-value mode), all indices 0 (red0) or 1 (red1); decoded to R8G8B8A8 UNORM (R=red, G=0, B=0, A=1.0).
        // BC5 UNORM: decoded to R8G8B8A8 UNORM (R=red, G=green, B=0, A=1.0).
        uint8_t b4[4][8], b5[4][16];
        const uint8_t r0[4] = {0xff, 0xff, 0x40, 0xc0}, r1[4] = {0x80, 0x80, 0x00, 0x10};
        const int idx1[4] = {0, 1, 0, 1};
        uint64_t ex4[4], ex5[4];
        for (int i = 0; i < 4; i++) {
            memset(b4[i], 0, 8);
            b4[i][0] = r0[i]; b4[i][1] = r1[i];
            if (idx1[i]) { const uint8_t ones[6] = {0x49, 0x92, 0x24, 0x49, 0x92, 0x24}; memcpy(&b4[i][2], ones, 6); }
            uint8_t g[8] = {(uint8_t)(0x20 * (i + 1)), 0x00, 0, 0, 0, 0, 0, 0};
            memcpy(b5[i], b4[i], 8);
            memcpy(b5[i] + 8, g, 8);
            uint8_t rv = idx1[i] ? r1[i] : r0[i];
            const uint8_t t4[4] = {rv, 0, 0, 0xff};
            const uint8_t t5[4] = {rv, g[0], 0, 0xff};
            ex4[i] = le_bytes(t4, 4);
            ex5[i] = le_bytes(t5, 4);
        }
        test_ex(VK_FORMAT_BC4_UNORM_BLOCK, "BC4", &b4[0][0], 8, 1, 4, inner, ex4);
        test_ex(VK_FORMAT_BC5_UNORM_BLOCK, "BC5", &b5[0][0], 16, 1, 4, inner, ex5);
    }
    test_restore();
    {
        // HZD's BC7 shadows: MUTABLE (and ALIAS) without a format list. The layer adds a UNORM/SRGB list so the RGBA8
        // shadow can get AFRC (3 bpc: 2055 KiB; uncompressed: 5465 KiB). [Claude Opus 5.5] 2026-10-26
        VkImageCreateInfo ci = {VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, NULL,
                                VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT | VK_IMAGE_CREATE_ALIAS_BIT, VK_IMAGE_TYPE_2D,
                                VK_FORMAT_BC7_SRGB_BLOCK, {1024, 1024, 1}, 11, 1, VK_SAMPLE_COUNT_1_BIT, VK_IMAGE_TILING_OPTIMAL,
                                VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT};
        VkImage img;
        VkResult r2 = vkCreateImage(dev, &ci, NULL, &img);
        VkMemoryRequirements req = {0};
        if (r2 == VK_SUCCESS) vkGetImageMemoryRequirements(dev, img, &req);
        VkDeviceImageMemoryRequirements dq = {VK_STRUCTURE_TYPE_DEVICE_IMAGE_MEMORY_REQUIREMENTS, NULL, &ci, 0};
        VkMemoryRequirements2 req2 = {VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2};
        vkGetDeviceImageMemoryRequirements(dev, &dq, &req2);
        bool ok = r2 == VK_SUCCESS && req.size < 3000 * 1024 && req2.memoryRequirements.size == req.size;
        printf("BC7_SRGB MUTABLE|ALIAS no list 1024^2+mips: create %d, %llu KiB (device query %llu KiB) -> %s\n", r2,
               (unsigned long long)req.size / 1024, (unsigned long long)req2.memoryRequirements.size / 1024,
               ok ? "PASS (compressed)" : "FAIL");
        failures += !ok;
        // vkd3d-proton chains VkImageCompressionControlEXT DISABLED on its placed resources; the layer overrides it
        // for BC shadows and restores the app's struct afterwards.
        VkImageCompressionControlEXT cc = {VK_STRUCTURE_TYPE_IMAGE_COMPRESSION_CONTROL_EXT, NULL, VK_IMAGE_COMPRESSION_DISABLED_EXT};
        ci.pNext = &cc;
        VkImage img2;
        VkResult r3 = vkCreateImage(dev, &ci, NULL, &img2);
        VkMemoryRequirements req3 = {0};
        if (r3 == VK_SUCCESS) vkGetImageMemoryRequirements(dev, img2, &req3);
        ok = r3 == VK_SUCCESS && req3.size < 3000 * 1024 && cc.flags == VK_IMAGE_COMPRESSION_DISABLED_EXT && !cc.pFixedRateFlags;
        printf("same with an app VkImageCompressionControlEXT DISABLED: create %d, %llu KiB, app struct %s -> %s\n", r3,
               (unsigned long long)req3.size / 1024, cc.flags == VK_IMAGE_COMPRESSION_DISABLED_EXT ? "restored" : "CHANGED",
               ok ? "PASS (compressed)" : "FAIL");
        failures += !ok;
    }
    printf("new-format and restore checks: %d failed\n", failures);
    printf("expected: BC1 ff0000ff 0000ffff 0000ffff ff0000ff; BC3 00ff0080 ff000080 ff000080 00ff0080;\n"
           "          BC1 array layer 0 as BC1, layer 1 0000ffff ff0000ff ff0000ff 0000ffff\n");
    return 0;
}

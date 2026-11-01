// SPDX-License-Identifier: GPL-3.0-or-later
// wd2-replay-probe: replays ONE dispatch of a dumped WD2 DXVK compute shader (e0ff8b4d / ae9a9376, B10-18 .spv) with the
// binding shapes of B10-19 X seq 2 (bridge/build/logs/epic/wd2-b10-19/mali-desc-14287.tsv, B rows 9,10,3,4,5,11,7,8):
//   set 0: b0-b2 R32_UINT + b3 R32G32B32A32_UINT uniform texel views into A (64 MiB), b4 R32_UINT storage texel view
//   into B (256 MiB), b5/b6/b7 uniform buffers (448 B in 32 MiB, 128 B in 16 MiB, 49088 B in 64 MiB), no push constants.
//   wd2-replay-probe <spv> <plain|dec|rebind> <zero|r1..r4|s1..s4|ff|g0> <gx>
//   plain : bind pipeline, bind set 0, dispatch gx x 1 x 1
//   dec   : bind pipeline, bind set 0, vkCmdCopyBufferToImage into a BC1 image (the compat layer decodes it right there:
//           decoder bind + push + dispatch + copy + restore), dispatch
//   rebind: as dec, plus the app binds set 0 again after the copy (= the candidate fix)
//   modes : fill of ALL buffers: zero, xorshift seeds r1-r4, all 0xFF, g0 = zero except t1's view words = 0xFFFF0000,
//           s1-s4 = seed N masked to the low 16 bits of each word (random 16-bit fields; the r/ff modes make the shader's
//           data-dependent loops run for minutes on lavapipe, class T)
// Set bind with vkCmdBindDescriptorSets2KHR when the driver has VK_KHR_maintenance6 (as DXVK), else v1 (BIND line says
// which; RPL_BIND1=1 forces v1). ONE dispatch per process, so a GPU fault is attributed by pid: the blob prints
// `GROUP_ERROR_FATAL` + `sideband(<VA>)` in logcat and still returns VK_SUCCESS (robust-probe 15:1x), so the wrapper counts
// those lines per pid. Prints PROBE / FEATURES / ENABLED / BUF / VIEW / BIND / RESULT lines. Exit 0 = RESULT printed,
// 3 = device lost or timeout (RESULT printed), 4 = setup error. RPL_WAIT_S = fence wait in s (default 5; lavapipe needs more). [tests, Claude Opus 5.5] 2026-11-01
#define _GNU_SOURCE
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <vulkan/vulkan.h>

#define MB (1024ull * 1024ull)
#define CK(x) do { VkResult r_ = (x); if (r_) { printf("ERROR %s -> %d\n", #x, r_); fflush(stdout); return 4; } } while (0)

static uint32_t find_type(const VkPhysicalDeviceMemoryProperties *mp, uint32_t bits, VkMemoryPropertyFlags want) {
    for (uint32_t i = 0; i < mp->memoryTypeCount; i++)
        if ((bits & (1u << i)) && (mp->memoryTypes[i].propertyFlags & want) == want) return i;
    return ~0u;
}
static int has_ext(const VkExtensionProperties *ep, uint32_t n, const char *name) {
    for (uint32_t i = 0; i < n; i++) if (!strcmp(ep[i].extensionName, name)) return 1;
    return 0;
}
static uint64_t fnv(const uint8_t *p, size_t n, uint64_t h) {   // 64-bit FNV-1a over 8-byte words
    const uint64_t *w = (const uint64_t *)p;
    for (size_t i = 0; i < n / 8; i++) { h ^= w[i]; h *= 0x100000001b3ull; }
    for (size_t i = n & ~(size_t)7; i < n; i++) { h ^= p[i]; h *= 0x100000001b3ull; }
    return h;
}

// buffers: A (t0-t3), B (u0), U5 (cb0), U6 (cb1), U7 (cb2)
enum { BA, BB, B5, B6, B7, NBUF };
static const char *bname[NBUF] = {"A", "B", "U5", "U6", "U7"};
static const VkDeviceSize bsize[NBUF] = {64 * MB, 256 * MB, 32 * MB, 16 * MB, 64 * MB};
struct bind { uint32_t buf; VkDescriptorType type; VkDeviceSize off, range; VkFormat fmt; };
static const struct bind binds[8] = {
    {BA, VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER, 7340032, 379372, VK_FORMAT_R32_UINT},          // B9
    {BA, VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER, 7274496, 54988, VK_FORMAT_R32_UINT},           // B10
    {BA, VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER, 10223616, 1312672, VK_FORMAT_R32_UINT},        // B3
    {BA, VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER, 7077888, 9456, VK_FORMAT_R32G32B32A32_UINT},   // B4
    {BB, VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER, 202964992, 2182500, VK_FORMAT_R32_UINT},       // B5
    {B5, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 2469888, 448, VK_FORMAT_UNDEFINED},                  // B11
    {B6, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 16111360, 128, VK_FORMAT_UNDEFINED},                 // B7
    {B7, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 15794176, 49088, VK_FORMAT_UNDEFINED},               // B8
};

int main(int argc, char **argv) {
    if (argc < 5) { fprintf(stderr, "usage: wd2-replay-probe <spv> plain|dec|rebind zero|r1..r4|s1..s4|ff|g0 <gx>\n"); return 4; }
    const char *spv = argv[1], *var = argv[2], *mode = argv[3];
    uint32_t gx = (uint32_t)strtoul(argv[4], NULL, 0);
    int v_dec = !strcmp(var, "dec") || !strcmp(var, "rebind"), v_rebind = !strcmp(var, "rebind");
    if (!v_dec && strcmp(var, "plain")) { fprintf(stderr, "bad variant\n"); return 4; }
    uint64_t seed = 0;
    int m_ff = !strcmp(mode, "ff"), m_g0 = !strcmp(mode, "g0"), m_zero = !strcmp(mode, "zero");
    if ((mode[0] == 'r' || mode[0] == 's') && mode[1] >= '1' && mode[1] <= '9' && !mode[2]) seed = (uint64_t)(mode[1] - '0');
    int m_small = mode[0] == 's' && seed;   // sN = seed N with every 32-bit word & 0x0000FFFF (bounded data loops)
    if (!m_ff && !m_g0 && !m_zero && !seed) { fprintf(stderr, "bad mode\n"); return 4; }
    const char *sname = strrchr(spv, '/') ? strrchr(spv, '/') + 1 : spv;
    printf("PROBE pid=%d spv=%s variant=%s mode=%s gx=%u\n", (int)getpid(), sname, var, mode, gx);
    fflush(stdout);

    FILE *sf = fopen(spv, "rb");
    if (!sf) { printf("ERROR cannot open %s\n", spv); return 4; }
    fseek(sf, 0, SEEK_END);
    long ssz = ftell(sf);
    fseek(sf, 0, SEEK_SET);
    uint32_t *code = malloc(ssz > 0 ? (size_t)ssz : 4);
    if (ssz <= 0 || ssz % 4 || fread(code, 1, (size_t)ssz, sf) != (size_t)ssz) { printf("ERROR read %s\n", spv); return 4; }
    fclose(sf);

    VkApplicationInfo app = {VK_STRUCTURE_TYPE_APPLICATION_INFO, NULL, "wd2-replay-probe", 1, NULL, 0, VK_API_VERSION_1_3};
    VkInstanceCreateInfo ici = {VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, NULL, 0, &app};
    VkInstance inst;
    CK(vkCreateInstance(&ici, NULL, &inst));
    uint32_t n = 1;
    VkPhysicalDevice pd;
    VkResult er = vkEnumeratePhysicalDevices(inst, &n, &pd);
    if ((er != VK_SUCCESS && er != VK_INCOMPLETE) || !n) { printf("ERROR enumerate -> %d n=%u\n", er, n); return 4; }

    uint32_t ne = 0;
    vkEnumerateDeviceExtensionProperties(pd, NULL, &ne, NULL);
    VkExtensionProperties *ep = calloc(ne ? ne : 1, sizeof *ep);
    vkEnumerateDeviceExtensionProperties(pd, NULL, &ne, ep);
    int x_r2 = has_ext(ep, ne, "VK_EXT_robustness2"), x_fc2 = has_ext(ep, ne, "VK_KHR_shader_float_controls2"),
        x_m6 = has_ext(ep, ne, "VK_KHR_maintenance6") && !(getenv("RPL_BIND1") && atoi(getenv("RPL_BIND1")));

    VkPhysicalDeviceMaintenance6FeaturesKHR m6s = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_6_FEATURES_KHR};
    VkPhysicalDeviceShaderFloatControls2FeaturesKHR fc2s = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT_CONTROLS_2_FEATURES_KHR,
                                                            x_m6 ? &m6s : NULL};
    VkPhysicalDeviceRobustness2FeaturesEXT r2s = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ROBUSTNESS_2_FEATURES_EXT, x_fc2 ? (void *)&fc2s : (x_m6 ? (void *)&m6s : NULL)};
    VkPhysicalDeviceVulkan12Features v12s = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES, x_r2 ? (void *)&r2s : (x_fc2 ? (void *)&fc2s : (x_m6 ? (void *)&m6s : NULL))};
    VkPhysicalDeviceFeatures2 f2 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, &v12s};
    vkGetPhysicalDeviceFeatures2(pd, &f2);
    VkPhysicalDeviceProperties pp;
    vkGetPhysicalDeviceProperties(pd, &pp);
    printf("FEATURES device=\"%s\" api=%u.%u.%u ext_robustness2=%d ext_float_controls2=%d ext_maintenance6=%d robustBufferAccess=%u "
           "robustBufferAccess2=%u nullDescriptor=%u bufferDeviceAddress=%u vulkanMemoryModel=%u shaderFloatControls2=%u "
           "maintenance6=%u textureCompressionBC=%u shaderInt64=%u minTexelBufferOffsetAlignment=%llu maxTexelBufferElements=%u\n",
           pp.deviceName, VK_API_VERSION_MAJOR(pp.apiVersion), VK_API_VERSION_MINOR(pp.apiVersion), VK_API_VERSION_PATCH(pp.apiVersion),
           x_r2, x_fc2, x_m6, f2.features.robustBufferAccess, r2s.robustBufferAccess2, r2s.nullDescriptor, v12s.bufferDeviceAddress,
           v12s.vulkanMemoryModel, fc2s.shaderFloatControls2, m6s.maintenance6, f2.features.textureCompressionBC, f2.features.shaderInt64,
           (unsigned long long)pp.limits.minTexelBufferOffsetAlignment, pp.limits.maxTexelBufferElements);
    fflush(stdout);
    if (!v12s.bufferDeviceAddress || !v12s.vulkanMemoryModel || !x_fc2 || !fc2s.shaderFloatControls2) {
        printf("ERROR device lacks a feature the DXVK SPIR-V needs (BDA, vulkanMemoryModel, shaderFloatControls2)\n");
        return 4;
    }
    if (v_dec && !f2.features.textureCompressionBC) { printf("ERROR textureCompressionBC not reported (compat layer missing?)\n"); return 4; }

    uint32_t nq = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(pd, &nq, NULL);
    VkQueueFamilyProperties *qp = calloc(nq ? nq : 1, sizeof *qp);
    vkGetPhysicalDeviceQueueFamilyProperties(pd, &nq, qp);
    uint32_t qf = ~0u;
    for (uint32_t i = 0; i < nq; i++) if (qp[i].queueFlags & VK_QUEUE_COMPUTE_BIT) { qf = i; break; }
    if (qf == ~0u) { printf("ERROR no compute queue\n"); return 4; }

    // Enabled as DXVK: robustBufferAccess + robustBufferAccess2 + nullDescriptor, BDA, memory model, float controls 2,
    // maintenance6, textureCompressionBC, shaderInt64.
    VkPhysicalDeviceMaintenance6FeaturesKHR m6e = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_6_FEATURES_KHR};
    m6e.maintenance6 = m6s.maintenance6;
    VkPhysicalDeviceShaderFloatControls2FeaturesKHR fc2e = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT_CONTROLS_2_FEATURES_KHR,
                                                            x_m6 ? &m6e : NULL, VK_TRUE};
    VkPhysicalDeviceRobustness2FeaturesEXT r2e = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ROBUSTNESS_2_FEATURES_EXT, &fc2e};
    r2e.robustBufferAccess2 = r2s.robustBufferAccess2;
    r2e.nullDescriptor = r2s.nullDescriptor;
    VkPhysicalDeviceVulkan12Features v12e = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES, x_r2 ? (void *)&r2e : (void *)&fc2e};
    v12e.bufferDeviceAddress = VK_TRUE;
    v12e.vulkanMemoryModel = VK_TRUE;
    v12e.vulkanMemoryModelDeviceScope = v12s.vulkanMemoryModelDeviceScope;
    VkPhysicalDeviceFeatures2 f2e = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, &v12e};
    f2e.features.robustBufferAccess = f2.features.robustBufferAccess;
    f2e.features.textureCompressionBC = f2.features.textureCompressionBC;
    f2e.features.shaderInt64 = f2.features.shaderInt64;
    const char *exts[3];
    uint32_t nx = 0;
    exts[nx++] = "VK_KHR_shader_float_controls2";
    if (x_r2) exts[nx++] = "VK_EXT_robustness2";
    if (x_m6) exts[nx++] = "VK_KHR_maintenance6";
    printf("ENABLED robustBufferAccess=%u robustBufferAccess2=%u nullDescriptor=%u textureCompressionBC=%u maintenance6=%u\n",
           f2e.features.robustBufferAccess, x_r2 ? r2e.robustBufferAccess2 : 0, x_r2 ? r2e.nullDescriptor : 0,
           f2e.features.textureCompressionBC, m6e.maintenance6);
    fflush(stdout);
    float prio = 1;
    VkDeviceQueueCreateInfo qi = {VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO, NULL, 0, qf, 1, &prio};
    VkDeviceCreateInfo dci = {VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, &f2e, 0, 1, &qi, 0, NULL, nx, exts};
    VkDevice dev;
    CK(vkCreateDevice(pd, &dci, NULL, &dev));
    VkQueue queue;
    vkGetDeviceQueue(dev, qf, 0, &queue);
    VkPhysicalDeviceMemoryProperties mp;
    vkGetPhysicalDeviceMemoryProperties(pd, &mp);
    PFN_vkCmdBindDescriptorSets2KHR bind2 = x_m6 && m6e.maintenance6 ?
        (PFN_vkCmdBindDescriptorSets2KHR)vkGetDeviceProcAddr(dev, "vkCmdBindDescriptorSets2KHR") : NULL;

    // buffers (host-visible coherent, device address on, as DXVK sets SHADER_DEVICE_ADDRESS on every buffer)
    VkBufferUsageFlags tu = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_UNIFORM_TEXEL_BUFFER_BIT |
                            VK_BUFFER_USAGE_STORAGE_TEXEL_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                            VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
    VkBuffer buf[NBUF + 1];
    uint8_t *map[NBUF + 1];
    unsigned long long va[NBUF + 1];
    VkDeviceSize sizes[NBUF + 1];
    for (int k = 0; k <= NBUF; k++) {   // index NBUF = the BC1 source buffer (64 B, 8x8 BC1 = 4 blocks of 8 B... 32 B used)
        sizes[k] = k < NBUF ? bsize[k] : 64;
        VkBufferCreateInfo bci = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, NULL, 0, sizes[k], tu};
        CK(vkCreateBuffer(dev, &bci, NULL, &buf[k]));
        VkMemoryRequirements req;
        vkGetBufferMemoryRequirements(dev, buf[k], &req);
        uint32_t t = find_type(&mp, req.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        if (t == ~0u) { printf("ERROR no host-visible coherent memory type\n"); return 4; }
        VkMemoryAllocateFlagsInfo fi = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO, NULL, VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT};
        VkMemoryAllocateInfo ai = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, &fi, req.size, t};
        VkDeviceMemory m;
        CK(vkAllocateMemory(dev, &ai, NULL, &m));
        CK(vkBindBufferMemory(dev, buf[k], m, 0));
        CK(vkMapMemory(dev, m, 0, VK_WHOLE_SIZE, 0, (void **)&map[k]));
        VkBufferDeviceAddressInfo bai = {VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO, NULL, buf[k]};
        va[k] = vkGetBufferDeviceAddress(dev, &bai);
        printf("BUF %s va=0x%llx size=%llu end=0x%llx\n", k < NBUF ? bname[k] : "BCSRC", va[k], (unsigned long long)sizes[k],
               va[k] + (unsigned long long)sizes[k]);
    }
    // fill
    for (int k = 0; k < NBUF; k++) {
        if (m_ff) memset(map[k], 0xFF, sizes[k]);
        else if (m_zero || m_g0) memset(map[k], 0, sizes[k]);
        else {
            uint64_t x = seed * 0x9E3779B97F4A7C15ull ^ ((uint64_t)(k + 1) << 32) ^ 0xD1B54A32D192ED03ull;
            uint64_t *w = (uint64_t *)map[k];
            for (size_t i = 0; i < sizes[k] / 8; i++) { x ^= x << 13; x ^= x >> 7; x ^= x << 17; w[i] = x * 0x2545F4914F6CDD1Dull; if (m_small) w[i] &= 0x0000FFFF0000FFFFull; }
        }
    }
    if (m_g0) {   // t1 = binding 1 (R32_UINT): every word 0xFFFF0000 (high 16 bits all ones, the g0 index word >> 16 = 65535)
        uint32_t *w = (uint32_t *)(map[BA] + binds[1].off);
        for (VkDeviceSize i = 0; i < binds[1].range / 4; i++) w[i] = 0xFFFF0000u;
    }
    // BC1 source: c0 = red, c1 = blue, indices 0 (red) -> any valid block
    memset(map[NBUF], 0, 64);
    for (int b = 0; b < 4; b++) { uint16_t *c = (uint16_t *)(map[NBUF] + b * 8); c[0] = 0xF800; c[1] = 0x001F; }
    const struct bind *u0 = &binds[4];
    uint8_t *u0p = map[BB] + u0->off;
    uint8_t *u0before = malloc(u0->range);
    memcpy(u0before, u0p, u0->range);
    uint64_t out_before = fnv(map[BB] + u0->off + u0->range, bsize[BB] - u0->off - u0->range, fnv(map[BB], u0->off, 0xcbf29ce484222325ull));
    for (int i = 0; i < 8; i++) {
        const struct bind *b = &binds[i];
        printf("VIEW b%d %s off=%llu range=%llu lo=0x%llx hi=0x%llx fmt=%d\n", i, bname[b->buf], (unsigned long long)b->off,
               (unsigned long long)b->range, va[b->buf] + b->off, va[b->buf] + b->off + b->range, (int)b->fmt);
    }
    fflush(stdout);

    // descriptors: set 0 = b0-b3 UTB, b4 STB, b5-b7 UBO
    VkDescriptorSetLayoutBinding lb[8];
    VkBufferView bv[5];
    VkDescriptorBufferInfo ubi[3];
    VkWriteDescriptorSet w[8];
    memset(w, 0, sizeof w);
    for (int i = 0; i < 8; i++) lb[i] = (VkDescriptorSetLayoutBinding){(uint32_t)i, binds[i].type, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL};
    VkDescriptorSetLayoutCreateInfo sli = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, NULL, 0, 8, lb};
    VkDescriptorSetLayout sl;
    CK(vkCreateDescriptorSetLayout(dev, &sli, NULL, &sl));
    VkPipelineLayoutCreateInfo pli = {VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, NULL, 0, 1, &sl, 0, NULL};
    VkPipelineLayout pl;
    CK(vkCreatePipelineLayout(dev, &pli, NULL, &pl));
    VkShaderModuleCreateInfo smi = {VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, NULL, 0, (size_t)ssz, code};
    VkShaderModule sm;
    CK(vkCreateShaderModule(dev, &smi, NULL, &sm));
    VkComputePipelineCreateInfo cpi = {VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO, NULL, 0,
        {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, NULL, 0, VK_SHADER_STAGE_COMPUTE_BIT, sm, "main", NULL}, pl};
    VkPipeline pipe;
    CK(vkCreateComputePipelines(dev, VK_NULL_HANDLE, 1, &cpi, NULL, &pipe));
    VkDescriptorPoolSize ps[3] = {{VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER, 4}, {VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER, 1},
                                  {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 3}};
    VkDescriptorPoolCreateInfo dpi = {VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO, NULL, 0, 1, 3, ps};
    VkDescriptorPool dp;
    CK(vkCreateDescriptorPool(dev, &dpi, NULL, &dp));
    VkDescriptorSetAllocateInfo dai = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, NULL, dp, 1, &sl};
    VkDescriptorSet ds;
    CK(vkAllocateDescriptorSets(dev, &dai, &ds));
    for (int i = 0; i < 8; i++) {
        const struct bind *b = &binds[i];
        w[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w[i].dstSet = ds;
        w[i].dstBinding = (uint32_t)i;
        w[i].descriptorCount = 1;
        w[i].descriptorType = b->type;
        if (i < 5) {
            VkBufferViewCreateInfo vi = {VK_STRUCTURE_TYPE_BUFFER_VIEW_CREATE_INFO, NULL, 0, buf[b->buf], b->fmt, b->off, b->range};
            CK(vkCreateBufferView(dev, &vi, NULL, &bv[i]));
            w[i].pTexelBufferView = &bv[i];
        } else {
            ubi[i - 5] = (VkDescriptorBufferInfo){buf[b->buf], b->off, b->range};
            w[i].pBufferInfo = &ubi[i - 5];
        }
    }
    vkUpdateDescriptorSets(dev, 8, w, 0, NULL);

    // BC1 image (8x8, the layer makes it RGBA8 and decodes the upload in the command buffer)
    VkImage img = VK_NULL_HANDLE;
    if (v_dec) {
        VkImageCreateInfo imci = {VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, NULL, 0, VK_IMAGE_TYPE_2D, VK_FORMAT_BC1_RGBA_UNORM_BLOCK,
                                  {8, 8, 1}, 1, 1, VK_SAMPLE_COUNT_1_BIT, VK_IMAGE_TILING_OPTIMAL,
                                  VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT};
        CK(vkCreateImage(dev, &imci, NULL, &img));
        VkMemoryRequirements req;
        vkGetImageMemoryRequirements(dev, img, &req);
        uint32_t t = find_type(&mp, req.memoryTypeBits, 0);
        VkMemoryAllocateInfo ai = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, NULL, req.size, t};
        VkDeviceMemory m;
        CK(vkAllocateMemory(dev, &ai, NULL, &m));
        CK(vkBindImageMemory(dev, img, m, 0));
    }

    VkCommandPoolCreateInfo cpci = {VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO, NULL, 0, qf};
    VkCommandPool pool;
    CK(vkCreateCommandPool(dev, &cpci, NULL, &pool));
    VkCommandBuffer cb;
    VkCommandBufferAllocateInfo cai = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, NULL, pool, VK_COMMAND_BUFFER_LEVEL_PRIMARY, 1};
    CK(vkAllocateCommandBuffers(dev, &cai, &cb));
    VkCommandBufferBeginInfo bi = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, NULL, VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT};
    CK(vkBeginCommandBuffer(cb, &bi));
    VkMemoryBarrier hb0 = {VK_STRUCTURE_TYPE_MEMORY_BARRIER, NULL, VK_ACCESS_HOST_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT};
    if (v_dec) {
        VkImageSubresourceRange rng = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        VkImageMemoryBarrier ib = {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER, NULL, 0, VK_ACCESS_TRANSFER_WRITE_BIT,
                                   VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_QUEUE_FAMILY_IGNORED,
                                   VK_QUEUE_FAMILY_IGNORED, img, rng};
        vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_HOST_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0,
                             1, &hb0, 0, NULL, 1, &ib);
    } else {
        vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_HOST_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &hb0, 0, NULL, 0, NULL);
    }
    VkBindDescriptorSetsInfoKHR bdi = {VK_STRUCTURE_TYPE_BIND_DESCRIPTOR_SETS_INFO_KHR, NULL, VK_SHADER_STAGE_COMPUTE_BIT, pl, 0, 1, &ds, 0, NULL};
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
    if (bind2) bind2(cb, &bdi); else vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pl, 0, 1, &ds, 0, NULL);
    int binds_done = 1;
    if (v_dec) {
        VkBufferImageCopy reg = {0, 0, 0, {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}, {0, 0, 0}, {8, 8, 1}};
        vkCmdCopyBufferToImage(cb, buf[NBUF], img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &reg);
        if (v_rebind) {
            if (bind2) bind2(cb, &bdi); else vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pl, 0, 1, &ds, 0, NULL);
            binds_done++;
        }
    }
    vkCmdDispatch(cb, gx, 1, 1);
    VkMemoryBarrier mb = {VK_STRUCTURE_TYPE_MEMORY_BARRIER, NULL, VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_HOST_READ_BIT};
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &mb,
                         0, NULL, 0, NULL);
    CK(vkEndCommandBuffer(cb));
    printf("BIND api=%s set_binds=%d copy=%s\n", bind2 ? "vkCmdBindDescriptorSets2KHR" : "vkCmdBindDescriptorSets", binds_done,
           v_dec ? "vkCmdCopyBufferToImage BC1 8x8" : "none");
    VkFenceCreateInfo fci = {VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    VkFence fence;
    CK(vkCreateFence(dev, &fci, NULL, &fence));
    VkSubmitInfo si = {VK_STRUCTURE_TYPE_SUBMIT_INFO, NULL, 0, NULL, NULL, 1, &cb};
    printf("DISPATCH\n");
    fflush(stdout);
    uint64_t wait_ns = (getenv("RPL_WAIT_S") ? strtoull(getenv("RPL_WAIT_S"), NULL, 0) : 5) * 1000000000ull;   // fence wait, default 5 s
    VkResult sr = vkQueueSubmit(queue, 1, &si, fence);
    VkResult wr = sr == VK_SUCCESS ? vkWaitForFences(dev, 1, &fence, VK_TRUE, wait_ns) : sr;
    usleep(300000);   // let the blob's fault lines reach logcat before the process exits

    size_t changed = 0;
    for (size_t i = 0; i < u0->range / 4; i++) changed += ((uint32_t *)u0p)[i] != ((uint32_t *)u0before)[i];
    uint64_t out_after = fnv(map[BB] + u0->off + u0->range, bsize[BB] - u0->off - u0->range, fnv(map[BB], u0->off, 0xcbf29ce484222325ull));
    const char *cls;
    if (wr == VK_ERROR_DEVICE_LOST) cls = "L";
    else if (wr == VK_TIMEOUT) cls = "T";
    else if (wr != VK_SUCCESS) cls = "E";
    else if (out_after != out_before) cls = "X";   // B changed outside u0's view range
    else if (changed) cls = "W";                   // u0 written inside its range only
    else cls = "U";                                // u0 unchanged
    printf("RESULT spv=%s variant=%s mode=%s gx=%u vk=%d class=%s u0_changed_words=%zu outside_changed=%d\n", sname, var, mode, gx,
           (int)wr, cls, changed, out_after != out_before);
    fflush(stdout);
    return (wr == VK_SUCCESS) ? 0 : 3;
}

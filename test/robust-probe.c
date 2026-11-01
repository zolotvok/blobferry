// SPDX-License-Identifier: GPL-3.0-or-later
// robust-probe: does the vendor Vulkan driver bounds-check out-of-range reads with robustBufferAccess2 on?
// ONE dispatch (one read) per process, so a GPU fault is attributed by pid (logcat "Device fault:" lines of that pid).
//   robust-probe <utb|stb|ubo|ssbo|wild> <view-offset-bytes> <index>
//   utb/stb: R32_UINT texel buffer view, offset, range 4096 B (1024 texels), texelFetch / imageLoad at <index>
//   ubo:     64 KiB array declared in the shader, descriptor offset, range 256 B (16 uvec4), res.a[<index>].x
//   ssbo:    runtime array, descriptor offset, range 4096 B (1024 uints), res.d[<index>]
//   wild:    read through a buffer_reference pointer to an unmapped VA (PROBE_WILD_VA, hex): positive control for the fault print
// Buffer = 16 MiB filled with 0xA5000000|dword-index, so "data" tells which dword the read hit. Env: PROBE_R2=0 robustness
// off, PROBE_WHOLE=1 range = whole buffer (control: an unclamped read must return data). PROBE_FMT=rgba32 (utb/stb only):
// R32G32B32A32_UINT view of 1024 elements (16 KiB), .x = dword off/4 + idx*4 (WD2 b3 shape; index x 16 wraps 32 bits at 2^28).
// [tests, Claude Opus 5.5] 2026-11-01 17:0x: PROBE_FMT added (B10-20 gap cases).
// Prints PROBE / LIMITS / FEATURES / RESULT lines. Exit 0 = RESULT printed, 3 = device lost or timeout (RESULT printed),
// 4 = setup error, 5 = no bufferDeviceAddress for wild. Run like map-probe (bridge ICD, game env). [tests, Claude Sonnet 5.5] 2026-11-01
#define _GNU_SOURCE
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <vulkan/vulkan.h>
#include "shaders/robust-probe-utb.h"
#include "shaders/robust-probe-stb.h"
#include "shaders/robust-probe-ubo.h"
#include "shaders/robust-probe-ssbo.h"
#include "shaders/robust-probe-wild.h"

#define BUFSZ (16u << 20)
#define CK(x) do { VkResult r_ = (x); if (r_) { printf("ERROR %s -> %d\n", #x, r_); fflush(stdout); return 4; } } while (0)

static uint32_t find_type(const VkPhysicalDeviceMemoryProperties *mp, uint32_t bits, VkMemoryPropertyFlags want) {
    for (uint32_t i = 0; i < mp->memoryTypeCount; i++)
        if ((bits & (1u << i)) && (mp->memoryTypes[i].propertyFlags & want) == want) return i;
    return ~0u;
}

int main(int argc, char **argv) {
    if (argc < 4) { fprintf(stderr, "usage: robust-probe utb|stb|ubo|ssbo|wild <offset> <index>\n"); return 4; }
    const char *type = argv[1];
    unsigned long off = strtoul(argv[2], NULL, 0);
    long long idx = strtoll(argv[3], NULL, 0);
    int want_r2 = !(getenv("PROBE_R2") && atoi(getenv("PROBE_R2")) == 0);
    int whole = getenv("PROBE_WHOLE") && atoi(getenv("PROBE_WHOLE")) != 0;
    int rgba = getenv("PROBE_FMT") && !strcmp(getenv("PROBE_FMT"), "rgba32");
    int is_wild = !strcmp(type, "wild"), is_utb = !strcmp(type, "utb"), is_stb = !strcmp(type, "stb"),
        is_ubo = !strcmp(type, "ubo"), is_ssbo = !strcmp(type, "ssbo");
    if (!(is_wild || is_utb || is_stb || is_ubo || is_ssbo)) { fprintf(stderr, "bad type\n"); return 4; }
    if (rgba && !(is_utb || is_stb)) { fprintf(stderr, "PROBE_FMT=rgba32 needs utb or stb\n"); return 4; }
    printf("PROBE pid=%d type=%s off=%lu idx=%lld r2=%d whole=%d fmt=%s\n", (int)getpid(), type, off, idx, want_r2, whole, rgba ? "rgba32" : "r32");
    fflush(stdout);

    VkApplicationInfo app = {VK_STRUCTURE_TYPE_APPLICATION_INFO, NULL, "robust-probe", 1, NULL, 0, VK_API_VERSION_1_3};
    VkInstanceCreateInfo ici = {VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, NULL, 0, &app};
    VkInstance inst;
    CK(vkCreateInstance(&ici, NULL, &inst));
    uint32_t n = 1;
    VkPhysicalDevice pd;
    VkResult er = vkEnumeratePhysicalDevices(inst, &n, &pd);
    if (er != VK_SUCCESS && er != VK_INCOMPLETE) { printf("ERROR enumerate -> %d\n", er); return 4; }
    if (!n) { printf("ERROR no physical device\n"); return 4; }

    uint32_t ne = 0;
    vkEnumerateDeviceExtensionProperties(pd, NULL, &ne, NULL);
    VkExtensionProperties *ep = calloc(ne ? ne : 1, sizeof *ep);
    vkEnumerateDeviceExtensionProperties(pd, NULL, &ne, ep);
    int has_r2ext = 0;
    for (uint32_t i = 0; i < ne; i++) if (!strcmp(ep[i].extensionName, "VK_EXT_robustness2")) has_r2ext = 1;

    VkPhysicalDeviceRobustness2FeaturesEXT r2s = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ROBUSTNESS_2_FEATURES_EXT};
    VkPhysicalDeviceVulkan12Features v12s = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES, has_r2ext ? &r2s : NULL};
    VkPhysicalDeviceFeatures2 f2 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, &v12s};
    vkGetPhysicalDeviceFeatures2(pd, &f2);
    VkPhysicalDeviceRobustness2PropertiesEXT r2p = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ROBUSTNESS_2_PROPERTIES_EXT};
    VkPhysicalDeviceProperties2 p2 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, has_r2ext ? &r2p : NULL};
    vkGetPhysicalDeviceProperties2(pd, &p2);
    const VkPhysicalDeviceLimits *L = &p2.properties.limits;
    printf("LIMITS api=%u.%u.%u device=\"%s\" minTexelBufferOffsetAlignment=%llu minUniformBufferOffsetAlignment=%llu "
           "minStorageBufferOffsetAlignment=%llu maxUniformBufferRange=%u maxTexelBufferElements=%u robustUniformSize=%llu robustStorageSize=%llu\n",
           VK_API_VERSION_MAJOR(p2.properties.apiVersion), VK_API_VERSION_MINOR(p2.properties.apiVersion),
           VK_API_VERSION_PATCH(p2.properties.apiVersion), p2.properties.deviceName, (unsigned long long)L->minTexelBufferOffsetAlignment,
           (unsigned long long)L->minUniformBufferOffsetAlignment, (unsigned long long)L->minStorageBufferOffsetAlignment,
           L->maxUniformBufferRange, L->maxTexelBufferElements, (unsigned long long)r2p.robustUniformBufferAccessSizeAlignment,
           (unsigned long long)r2p.robustStorageBufferAccessSizeAlignment);
    printf("FEATURES ext_robustness2=%d robustBufferAccess=%u robustBufferAccess2=%u nullDescriptor=%u bufferDeviceAddress=%u\n", has_r2ext,
           f2.features.robustBufferAccess, r2s.robustBufferAccess2, r2s.nullDescriptor, v12s.bufferDeviceAddress);
    fflush(stdout);
    int bda = v12s.bufferDeviceAddress;
    if (is_wild && !bda) { printf("NOBDA wild needs bufferDeviceAddress\n"); return 5; }

    uint32_t nq = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(pd, &nq, NULL);
    VkQueueFamilyProperties *qp = calloc(nq ? nq : 1, sizeof *qp);
    vkGetPhysicalDeviceQueueFamilyProperties(pd, &nq, qp);
    uint32_t qf = ~0u;
    for (uint32_t i = 0; i < nq; i++) if (qp[i].queueFlags & VK_QUEUE_COMPUTE_BIT) { qf = i; break; }
    if (qf == ~0u) { printf("ERROR no compute queue\n"); return 4; }

    // Mirror DXVK: robustBufferAccess + robustBufferAccess2 + nullDescriptor (when the driver has them).
    VkPhysicalDeviceRobustness2FeaturesEXT r2e = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ROBUSTNESS_2_FEATURES_EXT};
    r2e.robustBufferAccess2 = want_r2 && r2s.robustBufferAccess2;
    r2e.nullDescriptor = want_r2 && r2s.nullDescriptor;
    VkPhysicalDeviceVulkan12Features v12e = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES, has_r2ext ? &r2e : NULL};
    v12e.bufferDeviceAddress = bda;
    VkPhysicalDeviceFeatures2 f2e = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, &v12e};
    f2e.features.robustBufferAccess = want_r2 && f2.features.robustBufferAccess;
    printf("ENABLED robustBufferAccess=%u robustBufferAccess2=%u nullDescriptor=%u bufferDeviceAddress=%u\n", f2e.features.robustBufferAccess,
           has_r2ext ? r2e.robustBufferAccess2 : 0, has_r2ext ? r2e.nullDescriptor : 0, bda);
    fflush(stdout);
    const char *exts[] = {"VK_EXT_robustness2"};
    float prio = 1;
    VkDeviceQueueCreateInfo qi = {VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO, NULL, 0, qf, 1, &prio};
    VkDeviceCreateInfo dci = {VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, &f2e, 0, 1, &qi, 0, NULL, has_r2ext ? 1 : 0, exts};
    VkDevice dev;
    CK(vkCreateDevice(pd, &dci, NULL, &dev));
    VkQueue queue;
    vkGetDeviceQueue(dev, qf, 0, &queue);
    VkPhysicalDeviceMemoryProperties mp;
    vkGetPhysicalDeviceMemoryProperties(pd, &mp);

    // test buffer (16 MiB, pattern) and output buffer (4 KiB, 0xdeadbeef)
    VkBufferUsageFlags tu = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                            VK_BUFFER_USAGE_UNIFORM_TEXEL_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_TEXEL_BUFFER_BIT |
                            (bda ? VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT : 0);
    VkBuffer tb, ob;
    VkBufferCreateInfo bci = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, NULL, 0, BUFSZ, tu};
    CK(vkCreateBuffer(dev, &bci, NULL, &tb));
    bci.size = 4096;
    bci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    CK(vkCreateBuffer(dev, &bci, NULL, &ob));
    VkBuffer bufs[2] = {tb, ob};
    uint32_t *map[2];
    for (int k = 0; k < 2; k++) {
        VkMemoryRequirements req;
        vkGetBufferMemoryRequirements(dev, bufs[k], &req);
        uint32_t t = find_type(&mp, req.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        if (t == ~0u) { printf("ERROR no host-visible coherent memory type\n"); return 4; }
        VkMemoryAllocateFlagsInfo fi = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO, NULL, VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT};
        VkMemoryAllocateInfo ai = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, (k == 0 && bda) ? &fi : NULL, req.size, t};
        VkDeviceMemory m;
        CK(vkAllocateMemory(dev, &ai, NULL, &m));
        CK(vkBindBufferMemory(dev, bufs[k], m, 0));
        CK(vkMapMemory(dev, m, 0, VK_WHOLE_SIZE, 0, (void **)&map[k]));
    }
    for (uint32_t i = 0; i < BUFSZ / 4; i++) map[0][i] = 0xA5000000u | i;
    for (int i = 0; i < 1024; i++) map[1][i] = 0xdeadbeefu;
    unsigned long long bufva = 0;
    if (bda) {
        VkBufferDeviceAddressInfo bai = {VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO, NULL, tb};
        bufva = vkGetBufferDeviceAddress(dev, &bai);
    }
    printf("BUFFER va=0x%llx size=%u\n", bufva, BUFSZ);
    fflush(stdout);

    // descriptors
    VkDescriptorType dt = is_utb ? VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER : is_stb ? VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER
                        : is_ubo ? VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER : VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    VkDescriptorSetLayoutBinding lb[2] = {{0, dt, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL},
                                          {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL}};
    VkDescriptorSetLayoutCreateInfo sli = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, NULL, 0, 2, lb};
    if (is_wild) { sli.bindingCount = 1; sli.pBindings = &lb[1]; }
    VkDescriptorSetLayout sl;
    CK(vkCreateDescriptorSetLayout(dev, &sli, NULL, &sl));
    VkPushConstantRange pr = {VK_SHADER_STAGE_COMPUTE_BIT, 0, 16};
    VkPipelineLayoutCreateInfo pli = {VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, NULL, 0, 1, &sl, 1, &pr};
    VkPipelineLayout pl;
    CK(vkCreatePipelineLayout(dev, &pli, NULL, &pl));
    const uint32_t *code = is_utb ? rp_utb : is_stb ? rp_stb : is_ubo ? rp_ubo : is_ssbo ? rp_ssbo : rp_wild;
    size_t csz = is_utb ? sizeof rp_utb : is_stb ? sizeof rp_stb : is_ubo ? sizeof rp_ubo : is_ssbo ? sizeof rp_ssbo : sizeof rp_wild;
    VkShaderModuleCreateInfo smi = {VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, NULL, 0, csz, code};
    VkShaderModule sm;
    CK(vkCreateShaderModule(dev, &smi, NULL, &sm));
    VkComputePipelineCreateInfo cpi = {VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO, NULL, 0,
        {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, NULL, 0, VK_SHADER_STAGE_COMPUTE_BIT, sm, "main", NULL}, pl};
    VkPipeline pipe;
    CK(vkCreateComputePipelines(dev, VK_NULL_HANDLE, 1, &cpi, NULL, &pipe));
    VkDescriptorPoolSize ps[2] = {{dt, 1}, {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1}};
    VkDescriptorPoolCreateInfo dpi = {VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO, NULL, 0, 1, 2, ps};
    VkDescriptorPool dp;
    CK(vkCreateDescriptorPool(dev, &dpi, NULL, &dp));
    VkDescriptorSetAllocateInfo dai = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, NULL, dp, 1, &sl};
    VkDescriptorSet ds;
    CK(vkAllocateDescriptorSets(dev, &dai, &ds));

    VkBufferView bv = VK_NULL_HANDLE;
    VkDescriptorBufferInfo ti = {tb, off, 4096}, oi = {ob, 0, 4096};
    if (is_utb || is_stb) {
        VkDeviceSize vr = whole ? ((BUFSZ - off) & ~(VkDeviceSize)(rgba ? 15 : 3)) : (rgba ? 16384 : 4096);
        VkBufferViewCreateInfo vi = {VK_STRUCTURE_TYPE_BUFFER_VIEW_CREATE_INFO, NULL, 0, tb, rgba ? VK_FORMAT_R32G32B32A32_UINT : VK_FORMAT_R32_UINT, off, vr};
        CK(vkCreateBufferView(dev, &vi, NULL, &bv));
    } else if (is_ubo) {
        ti.range = 256;
        if (whole) ti.range = (BUFSZ - off) < L->maxUniformBufferRange ? (BUFSZ - off) : L->maxUniformBufferRange;
    } else if (is_ssbo && whole) {
        ti.range = VK_WHOLE_SIZE;
    }
    VkWriteDescriptorSet w[2];
    memset(w, 0, sizeof w);
    w[0].sType = w[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    w[0].dstSet = w[1].dstSet = ds;
    w[0].dstBinding = 0;
    w[0].descriptorCount = w[1].descriptorCount = 1;
    w[0].descriptorType = dt;
    if (is_utb || is_stb) w[0].pTexelBufferView = &bv; else w[0].pBufferInfo = &ti;
    w[1].dstBinding = 1;
    w[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    w[1].pBufferInfo = &oi;
    if (is_wild) vkUpdateDescriptorSets(dev, 1, &w[1], 0, NULL);
    else vkUpdateDescriptorSets(dev, 2, w, 0, NULL);

    struct { int32_t idx, pad; uint64_t ptr; } pc = {(int32_t)idx, 0, 0};
    if (is_wild) {
        const char *e = getenv("PROBE_WILD_VA");
        pc.ptr = e ? strtoull(e, NULL, 16) : 0x0000012345678000ull;
    }
    VkCommandPoolCreateInfo cpci = {VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO, NULL, 0, qf};
    VkCommandPool pool;
    CK(vkCreateCommandPool(dev, &cpci, NULL, &pool));
    VkCommandBuffer cb;
    VkCommandBufferAllocateInfo cai = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, NULL, pool, VK_COMMAND_BUFFER_LEVEL_PRIMARY, 1};
    CK(vkAllocateCommandBuffers(dev, &cai, &cb));
    VkCommandBufferBeginInfo bi = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    CK(vkBeginCommandBuffer(cb, &bi));
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
    vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pl, 0, 1, &ds, 0, NULL);
    vkCmdPushConstants(cb, pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, 16, &pc);
    vkCmdDispatch(cb, 1, 1, 1);
    VkMemoryBarrier mb = {VK_STRUCTURE_TYPE_MEMORY_BARRIER, NULL, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_HOST_READ_BIT};
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &mb, 0, NULL, 0, NULL);
    CK(vkEndCommandBuffer(cb));
    VkFenceCreateInfo fci = {VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    VkFence fence;
    CK(vkCreateFence(dev, &fci, NULL, &fence));
    VkSubmitInfo si = {VK_STRUCTURE_TYPE_SUBMIT_INFO, NULL, 0, NULL, NULL, 1, &cb};
    printf("DISPATCH\n");
    fflush(stdout);
    VkResult sr = vkQueueSubmit(queue, 1, &si, fence);
    VkResult wr = sr == VK_SUCCESS ? vkWaitForFences(dev, 1, &fence, VK_TRUE, 5000000000ull) : sr;
    usleep(300000);   // let the fault print reach logcat before the process exits

    uint32_t val = map[1][0], mark = map[1][1], echo = map[1][2];
    long long stride = (is_ubo || rgba) ? 16 : 4;
    long long exp_i = -1;
    if (!is_wild && idx >= 0) {
        long long eb = (long long)off + idx * stride;
        if (eb + 4 <= (long long)BUFSZ) exp_i = eb / 4;
    }
    const char *cls;
    if (wr == VK_ERROR_DEVICE_LOST) cls = "L";
    else if (wr == VK_TIMEOUT) cls = "T";
    else if (wr != VK_SUCCESS) cls = "E";
    else if (mark != 0x600DF00Du) cls = "N";               // shader did not reach its end
    else if (val == 0) cls = "Z";                          // zero
    else if ((val >> 24) == 0xA5) cls = (exp_i >= 0 && (val & 0xFFFFFF) == (uint32_t)exp_i) ? "D" : "d";   // data (expected / another dword)
    else cls = "O";                                        // some other value
    printf("RESULT type=%s%s off=%lu idx=%lld vk=%d class=%s val=0x%08x mark=0x%08x echo=%u exp_i=%lld got_i=%d\n", type, rgba ? "-rgba32" : "", off, idx, (int)wr,
           cls, val, mark, echo, exp_i, (val >> 24) == 0xA5 ? (int)(val & 0xFFFFFF) : -1);
    fflush(stdout);
    return (wr == VK_SUCCESS) ? 0 : 3;
}

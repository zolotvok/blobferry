// SPDX-License-Identifier: GPL-3.0-or-later
// map-probe: can the vendor driver's CPU mappings be moved with mremap to an address the caller picks? This is what a
// layer-side VK_EXT_map_memory_placed needs (Wine on WoW64 places mappings below 4 GiB). [Claude Opus 5.5] 2026-10-26
// Run through the loader with only the bridge ICD:
//   VK_DRIVER_FILES=<bridge icd.json> HYBRIS_USE_VENDOR_NAMESPACE=1 VK_LOADER_LAYERS_DISABLE=~implicit~ map-probe [type]
// or, to test the compat layer's VK_EXT_map_memory_placed (only that layer: VK_LOADER_LAYERS_ENABLE=*mali_compat):
//   ... map-probe <type> placed
#define _GNU_SOURCE
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <vulkan/vulkan.h>

#define SIZE (4u << 20)
static VkDevice dev;
static VkQueue queue;
static VkCommandPool pool;

static void maps_line(const void *p) {
    FILE *f = fopen("/proc/self/maps", "r");
    char line[512];
    while (f && fgets(line, sizeof line, f)) {
        unsigned long a, b;
        if (sscanf(line, "%lx-%lx", &a, &b) == 2 && (uintptr_t)p >= a && (uintptr_t)p < b) printf("    maps: %s", line);
    }
    if (f) fclose(f);
}

static void run(void (*rec)(VkCommandBuffer, void *), void *arg) {
    VkCommandBuffer cb;
    VkCommandBufferAllocateInfo ai = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, NULL, pool, VK_COMMAND_BUFFER_LEVEL_PRIMARY, 1};
    vkAllocateCommandBuffers(dev, &ai, &cb);
    VkCommandBufferBeginInfo bi = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    vkBeginCommandBuffer(cb, &bi);
    rec(cb, arg);
    VkMemoryBarrier mb = {VK_STRUCTURE_TYPE_MEMORY_BARRIER, NULL, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_HOST_READ_BIT};
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &mb, 0, NULL, 0, NULL);
    vkEndCommandBuffer(cb);
    VkSubmitInfo si = {VK_STRUCTURE_TYPE_SUBMIT_INFO, NULL, 0, NULL, NULL, 1, &cb};
    vkQueueSubmit(queue, 1, &si, VK_NULL_HANDLE);
    vkQueueWaitIdle(queue);
    vkFreeCommandBuffers(dev, pool, 1, &cb);
}

struct copy { VkBuffer src, dst; };
static void rec_copy(VkCommandBuffer cb, void *a) {
    struct copy *c = a;
    VkBufferCopy r = {0, 0, SIZE};
    vkCmdCopyBuffer(cb, c->src, c->dst, 1, &r);
}
static void rec_fill(VkCommandBuffer cb, void *a) { vkCmdFillBuffer(cb, *(VkBuffer *)a, 0, SIZE, 0xdeadbeef); }

static uint32_t count(const uint32_t *p, uint32_t v) {
    uint32_t n = 0;
    for (uint32_t i = 0; i < SIZE / 4; i++) n += p[i] == v;
    return n;
}

static void buffer(uint32_t type, VkBuffer *b, VkDeviceMemory *m) {
    VkBufferCreateInfo ci = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, NULL, 0, SIZE,
                             VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT};
    vkCreateBuffer(dev, &ci, NULL, b);
    VkMemoryAllocateInfo ai = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, NULL, SIZE, type};
    VkResult r = vkAllocateMemory(dev, &ai, NULL, m);
    if (r != VK_SUCCESS) { printf("vkAllocateMemory(type %u) -> %d\n", type, r); exit(1); }
    vkBindBufferMemory(dev, *b, *m, 0);
}

// The compat layer's emulated VK_EXT_map_memory_placed, used the way Wine uses it.
static int placed_test(VkPhysicalDevice pd, uint32_t type) {
    VkPhysicalDeviceMapMemoryPlacedFeaturesEXT pf = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAP_MEMORY_PLACED_FEATURES_EXT};
    VkPhysicalDeviceFeatures2 f2 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, &pf};
    vkGetPhysicalDeviceFeatures2(pd, &f2);
    VkPhysicalDeviceMapMemoryPlacedPropertiesEXT pp = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAP_MEMORY_PLACED_PROPERTIES_EXT};
    VkPhysicalDeviceProperties2 p2 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, &pp};
    vkGetPhysicalDeviceProperties2(pd, &p2);
    printf("placed features: map %u range %u unmapReserve %u, alignment %llu\n", pf.memoryMapPlaced,
           pf.memoryMapRangePlaced, pf.memoryUnmapReserve, (unsigned long long)pp.minPlacedMemoryMapAlignment);
    if (!pf.memoryMapPlaced) return 1;
    float prio = 1;
    VkDeviceQueueCreateInfo q = {VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO, NULL, 0, 0, 1, &prio};
    const char *exts[] = {"VK_KHR_map_memory2", "VK_EXT_map_memory_placed"};
    pf.memoryMapRangePlaced = VK_FALSE;
    VkDeviceCreateInfo dci = {VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, &pf, 0, 1, &q, 0, NULL, 2, exts};
    VkResult r = vkCreateDevice(pd, &dci, NULL, &dev);
    printf("vkCreateDevice with VK_EXT_map_memory_placed -> %d\n", r);
    if (r != VK_SUCCESS) return 1;
    vkGetDeviceQueue(dev, 0, 0, &queue);
    VkCommandPoolCreateInfo pci = {VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO, NULL, 0, 0};
    vkCreateCommandPool(dev, &pci, NULL, &pool);
    PFN_vkMapMemory2KHR map2 = (PFN_vkMapMemory2KHR)vkGetDeviceProcAddr(dev, "vkMapMemory2KHR");
    PFN_vkUnmapMemory2KHR unmap2 = (PFN_vkUnmapMemory2KHR)vkGetDeviceProcAddr(dev, "vkUnmapMemory2KHR");
    VkBuffer a, b;
    VkDeviceMemory ma, mb;
    buffer(type, &a, &ma);
    buffer(type, &b, &mb);
    void *target = mmap((void *)0x40000000, SIZE, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE | MAP_FIXED_NOREPLACE, -1, 0);
    for (int round = 0; round < 2; round++) {
        VkMemoryMapPlacedInfoEXT pl = {VK_STRUCTURE_TYPE_MEMORY_MAP_PLACED_INFO_EXT, NULL, target};
        VkMemoryMapInfoKHR mi = {VK_STRUCTURE_TYPE_MEMORY_MAP_INFO_KHR, &pl, VK_MEMORY_MAP_PLACED_BIT_EXT, ma, 0, VK_WHOLE_SIZE};
        void *p = NULL;
        r = map2(dev, &mi, &p);
        printf("round %d: placed map -> %d at %p (wanted %p)\n", round, r, p, target);
        if (r != VK_SUCCESS || p != target) return 1;
        maps_line(p);
        for (uint32_t i = 0; i < SIZE / 4; i++) ((uint32_t *)p)[i] = 0x12345678 + round;
        VkMappedMemoryRange mr = {VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE, NULL, ma, 0, VK_WHOLE_SIZE};
        vkFlushMappedMemoryRanges(dev, 1, &mr);
        struct copy c = {a, b};
        run(rec_copy, &c);
        void *pb;
        vkMapMemory(dev, mb, 0, VK_WHOLE_SIZE, 0, &pb);
        VkMappedMemoryRange mrb = {VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE, NULL, mb, 0, VK_WHOLE_SIZE};
        vkInvalidateMappedMemoryRanges(dev, 1, &mrb);
        printf("  CPU write via placed map -> GPU copy: %u/%u right\n", count(pb, 0x12345678 + round), SIZE / 4);
        vkUnmapMemory(dev, mb);
        run(rec_fill, &a);
        vkInvalidateMappedMemoryRanges(dev, 1, &mr);
        printf("  GPU fill -> CPU read via placed map: %u/%u right\n", count(p, 0xdeadbeef), SIZE / 4);
        VkMemoryUnmapInfoKHR ui = {VK_STRUCTURE_TYPE_MEMORY_UNMAP_INFO_KHR, NULL, VK_MEMORY_UNMAP_RESERVE_BIT_EXT, ma};
        r = unmap2(dev, &ui);
        printf("  unmap with reserve -> %d; range now:\n", r);
        maps_line(target);
    }
    // Free while mapped: the range must stay reserved.
    VkMemoryMapPlacedInfoEXT pl = {VK_STRUCTURE_TYPE_MEMORY_MAP_PLACED_INFO_EXT, NULL, target};
    VkMemoryMapInfoKHR mi = {VK_STRUCTURE_TYPE_MEMORY_MAP_INFO_KHR, &pl, VK_MEMORY_MAP_PLACED_BIT_EXT, ma, 0, VK_WHOLE_SIZE};
    void *p = NULL;
    r = map2(dev, &mi, &p);
    vkDestroyBuffer(dev, a, NULL);
    vkFreeMemory(dev, ma, NULL);
    printf("map %d, then free while mapped; range now:\n", r);
    maps_line(target);
    vkDestroyBuffer(dev, b, NULL);
    vkFreeMemory(dev, mb, NULL);
    vkDestroyCommandPool(dev, pool, NULL);
    vkDestroyDevice(dev, NULL);
    printf("placed done\n");
    return 0;
}

int main(int argc, char **argv) {
    setvbuf(stdout, NULL, _IONBF, 0);
    uint32_t type = argc > 1 ? (uint32_t)atoi(argv[1]) : 0;
    VkApplicationInfo app = {VK_STRUCTURE_TYPE_APPLICATION_INFO, NULL, "map-probe", 1, NULL, 0, VK_API_VERSION_1_3};
    VkInstanceCreateInfo ici = {VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, NULL, 0, &app};
    VkInstance inst;
    if (vkCreateInstance(&ici, NULL, &inst) != VK_SUCCESS) return 1;
    uint32_t n = 1;
    VkPhysicalDevice pd;
    vkEnumeratePhysicalDevices(inst, &n, &pd);
    if (argc > 2 && !strcmp(argv[2], "placed")) return placed_test(pd, type);
    float prio = 1;
    VkDeviceQueueCreateInfo q = {VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO, NULL, 0, 0, 1, &prio};
    VkDeviceCreateInfo dci = {VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, NULL, 0, 1, &q};
    if (vkCreateDevice(pd, &dci, NULL, &dev) != VK_SUCCESS) return 1;
    vkGetDeviceQueue(dev, 0, 0, &queue);
    VkCommandPoolCreateInfo pci = {VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO, NULL, 0, 0};
    vkCreateCommandPool(dev, &pci, NULL, &pool);

    VkBuffer a, b;
    VkDeviceMemory ma, mb;
    buffer(type, &a, &ma);
    buffer(type, &b, &mb);

    void *pa;
    VkResult r = vkMapMemory(dev, ma, 0, VK_WHOLE_SIZE, 0, &pa);
    printf("type %u: map A -> %d at %p\n", type, r, pa);
    maps_line(pa);
    // Wine reserves the target first (PROT_NONE); mimic that, low in the address space.
    void *target = mmap((void *)0x40000000, SIZE, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE | MAP_FIXED_NOREPLACE, -1, 0);
    printf("reserve target -> %p\n", target);
    void *moved = mremap(pa, SIZE, SIZE, MREMAP_MAYMOVE | MREMAP_FIXED, target);
    printf("mremap -> %p (%s)\n", moved, moved == MAP_FAILED ? "failed" : "ok");
    if (moved == MAP_FAILED) { perror("mremap"); return 1; }
    maps_line(moved);
    // Hold the old range so nothing else lands there; it is the driver's address for the unmap.
    void *hold = mmap(pa, SIZE, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE | MAP_FIXED_NOREPLACE, -1, 0);
    printf("hold old range -> %p\n", hold);

    for (uint32_t i = 0; i < SIZE / 4; i++) ((uint32_t *)moved)[i] = 0x12345678;
    if (type == 1) {
        VkMappedMemoryRange mr = {VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE, NULL, ma, 0, VK_WHOLE_SIZE};
        vkFlushMappedMemoryRanges(dev, 1, &mr);
    }
    struct copy c = {a, b};
    run(rec_copy, &c);
    void *pb;
    vkMapMemory(dev, mb, 0, VK_WHOLE_SIZE, 0, &pb);
    if (type == 1) {
        VkMappedMemoryRange mr = {VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE, NULL, mb, 0, VK_WHOLE_SIZE};
        vkInvalidateMappedMemoryRanges(dev, 1, &mr);
    }
    printf("CPU write via moved mapping -> GPU copy: %u/%u words right\n", count(pb, 0x12345678), SIZE / 4);
    vkUnmapMemory(dev, mb);

    run(rec_fill, &a);
    if (type == 1) {
        VkMappedMemoryRange mr = {VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE, NULL, ma, 0, VK_WHOLE_SIZE};
        vkInvalidateMappedMemoryRanges(dev, 1, &mr);
    }
    printf("GPU fill -> CPU read via moved mapping: %u/%u words right\n", count(moved, 0xdeadbeef), SIZE / 4);

    // Unmap the way a layer would: move it back onto the held range, let the driver unmap, re-reserve the target.
    void *back = mremap(moved, SIZE, SIZE, MREMAP_MAYMOVE | MREMAP_FIXED, pa);
    printf("move back -> %p\n", back);
    vkUnmapMemory(dev, ma);
    void *rres = mmap(target, SIZE, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE | MAP_FIXED, -1, 0);
    printf("re-reserve target -> %p\n", rres);
    void *pa2;
    r = vkMapMemory(dev, ma, 0, VK_WHOLE_SIZE, 0, &pa2);
    printf("map A again -> %d at %p (%s), %u/%u words still 0xdeadbeef\n", r, pa2, pa2 == pa ? "same" : "new",
           count(pa2, 0xdeadbeef), SIZE / 4);
    vkUnmapMemory(dev, ma);
    vkDestroyBuffer(dev, a, NULL);
    vkDestroyBuffer(dev, b, NULL);
    vkFreeMemory(dev, ma, NULL);
    vkFreeMemory(dev, mb, NULL);
    vkDestroyCommandPool(dev, pool, NULL);
    vkDestroyDevice(dev, NULL);
    vkDestroyInstance(inst, NULL);
    printf("done\n");
    return 0;
}

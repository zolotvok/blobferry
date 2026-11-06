// SPDX-License-Identifier: GPL-3.0-or-later
// upload-membench: CPU write cost into Mali mapped memory, by memory type. [keeper, Claude Opus 5.5] 2026-11-05
// Question (docs/games/witcher3.md § "town CPU plan 1105" 5): the Witcher 3 RenderThread spends 51 % in one leaf loop
// (witcher3.exe+0x1e447c0) that interleaves 3+3+4 dwords per 40-B element with 32-bit stores into a mapped buffer.
// Is the vendor's HOST_COHERENT type (uncached, what vkd3d/DXVK map) several times slower for that pattern than the
// HOST_CACHED type or malloc, and what does the clean (vkFlushMappedMemoryRanges / dc cvac) of the same bytes cost?
//
// Build (PC): bridge/test/upload-membench-build.sh  ->  bridge/build/membench/upload-membench
// Run (guest, through the loader with only the bridge ICD, no implicit layers):
//   VK_DRIVER_FILES=/usr/lib/mali_bridge/mali_bridge_icd.json HYBRIS_USE_VENDOR_NAMESPACE=1 \
//   VK_LOADER_LAYERS_DISABLE=~implicit~ upload-membench [--cpu N] [--big MB] [--small MB] [--stride B] [--reps R]
// (the program always sets VK_DRIVER_FILES to the bridge ICD or --icd, and exits rc=3 unless the driver is ARM's).
// Output: "membench: ..." lines only; the last line is "membench: DONE rc=0" (rc != 0 = failed, see the line before).
#define _GNU_SOURCE
#include <errno.h>
#include <sched.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <vulkan/vulkan.h>

#define CK(x)                                                                                                         \
    do {                                                                                                              \
        VkResult r_ = (x);                                                                                            \
        if (r_ != VK_SUCCESS) {                                                                                       \
            printf("membench: FAIL %s = %d (line %d)\nmembench: DONE rc=1\n", #x, r_, __LINE__);                      \
            exit(1);                                                                                                  \
        }                                                                                                             \
    } while (0)

static VkDevice dev;
static int g_reps = 5;
static size_t g_stride = 40;

static uint64_t now_ns(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (uint64_t)t.tv_sec * 1000000000ull + (uint64_t)t.tv_nsec;
}

static int cmp_u64(const void *a, const void *b) {
    uint64_t x = *(const uint64_t *)a, y = *(const uint64_t *)b;
    return x < y ? -1 : x > y;
}

static long cur_khz(int cpu) {
    char p[96];
    snprintf(p, sizeof p, "/sys/devices/system/cpu/cpu%d/cpufreq/scaling_cur_freq", cpu);
    FILE *f = fopen(p, "r");
    long v = -1;
    if (f) {
        if (fscanf(f, "%ld", &v) != 1) v = -1;
        fclose(f);
    }
    return v;
}

// ---- the store patterns ----
// w32: the game loop (witcher3.exe+0x1e447c0): per element 3 dwords from stream A, 3 from B, 4 from C, each a 32-bit
// load + 32-bit store (FEX emits ldr w / str w for these with TSO off, which volatile metadata gives this function).
static void pat_w32(uint8_t *dst, size_t n, const uint32_t *a, const uint32_t *b, const uint32_t *c) {
    const volatile uint32_t *va = a, *vb = b, *vc = c;
    for (size_t i = 0; i < n; i++) {
        volatile uint32_t *d = (volatile uint32_t *)(dst + i * g_stride);
        d[0] = va[0]; d[1] = va[1]; d[2] = va[2];
        d[3] = vb[0]; d[4] = vb[1]; d[5] = vb[2];
        d[6] = vc[0]; d[7] = vc[1]; d[8] = vc[2]; d[9] = vc[3];
        va += 3, vb += 3, vc += 4;
    }
}

// w32rel: the same with release stores (stlr w) = what FEX emits for a plain x86 store with TSO on (other games' loops).
static inline void stlr32(volatile uint32_t *p, uint32_t v) {
#if defined(__aarch64__)
    __asm__ volatile("stlr %w0, [%1]" ::"r"(v), "r"(p) : "memory");
#else
    *p = v;
#endif
}
static void pat_w32rel(uint8_t *dst, size_t n, const uint32_t *a, const uint32_t *b, const uint32_t *c) {
    const volatile uint32_t *va = a, *vb = b, *vc = c;
    for (size_t i = 0; i < n; i++) {
        volatile uint32_t *d = (volatile uint32_t *)(dst + i * g_stride);
        stlr32(d + 0, va[0]); stlr32(d + 1, va[1]); stlr32(d + 2, va[2]);
        stlr32(d + 3, vb[0]); stlr32(d + 4, vb[1]); stlr32(d + 5, vb[2]);
        stlr32(d + 6, vc[0]); stlr32(d + 7, vc[1]); stlr32(d + 8, vc[2]); stlr32(d + 9, vc[3]);
        va += 3, vb += 3, vc += 4;
    }
}

// memcpy: the same bytes as one libc memcpy per element (what native code with wide stores would do).
static void pat_memcpy(uint8_t *dst, size_t n, const uint32_t *a, const uint32_t *b, const uint32_t *c) {
    for (size_t i = 0; i < n; i++) {
        uint8_t *d = dst + i * g_stride;
        memcpy(d, a + 3 * i, 12);
        memcpy(d + 12, b + 3 * i, 12);
        memcpy(d + 24, c + 4 * i, 16);
    }
}

// r32: 32-bit loads of the same bytes (in case the cursors point into mapped memory too).
static volatile uint32_t g_sink;
static void pat_r32(uint8_t *dst, size_t n, const uint32_t *a, const uint32_t *b, const uint32_t *c) {
    (void)a, (void)b, (void)c;
    uint32_t s = 0;
    for (size_t i = 0; i < n; i++) {
        const volatile uint32_t *d = (const volatile uint32_t *)(dst + i * g_stride);
        s += d[0] + d[1] + d[2] + d[3] + d[4] + d[5] + d[6] + d[7] + d[8] + d[9];
    }
    g_sink = s;
}

typedef void (*pat_fn)(uint8_t *, size_t, const uint32_t *, const uint32_t *, const uint32_t *);

// One measurement: 1 warm-up pass, then g_reps timed passes; returns the best ns, writes the median.
static uint64_t run(pat_fn f, uint8_t *dst, size_t n, const uint32_t *a, const uint32_t *b, const uint32_t *c,
                    uint64_t *median) {
    uint64_t t[64];
    int reps = g_reps > 64 ? 64 : g_reps;
    f(dst, n, a, b, c);
    for (int r = 0; r < reps; r++) {
        uint64_t t0 = now_ns();
        f(dst, n, a, b, c);
        t[r] = now_ns() - t0;
    }
    qsort(t, reps, sizeof t[0], cmp_u64);
    *median = t[reps / 2];
    return t[0];
}

static void cache_lines(void *addr, size_t size, int op) { // 0 = dc cvac (clean), 1 = dc civac (clean+invalidate)
#if defined(__aarch64__)
    uint64_t ctr;
    __asm__ volatile("mrs %0, ctr_el0" : "=r"(ctr));
    uintptr_t line = 4u << ((ctr >> 16) & 0xf);
    uintptr_t p = (uintptr_t)addr & ~(line - 1), end = (uintptr_t)addr + size;
    for (; p < end; p += line) {
        if (op) __asm__ volatile("dc civac, %0" ::"r"(p) : "memory");
        else __asm__ volatile("dc cvac, %0" ::"r"(p) : "memory");
    }
    __asm__ volatile("dsb ish" ::: "memory");
#else
    (void)addr, (void)size, (void)op;
#endif
}

static PFN_vkFlushMappedMemoryRanges pFlush;
static PFN_vkInvalidateMappedMemoryRanges pInval;

struct target {
    const char *name;
    int type;            // memory type index, -1 = malloc
    VkDeviceMemory mem;
    uint8_t *ptr;
    size_t size;
};

static double gbps(size_t bytes, uint64_t ns) { return ns ? (double)bytes / (double)ns : 0.0; }

static void bench_target(struct target *t, size_t size, const uint32_t *a, const uint32_t *b, const uint32_t *c,
                         int cpu) {
    size_t n = size / g_stride;
    size_t bytes = n * 40; // payload bytes written per pass
    static const struct { const char *name; pat_fn f; } pats[] = {
        {"w32", pat_w32}, {"w32rel", pat_w32rel}, {"memcpy", pat_memcpy}, {"r32", pat_r32}};
    for (unsigned p = 0; p < sizeof pats / sizeof pats[0]; p++) {
        uint64_t med, best = run(pats[p].f, t->ptr, n, a, b, c, &med);
        printf("membench: BW target=%s type=%d size_mb=%zu stride=%zu pat=%s elems=%zu best_ms=%.3f med_ms=%.3f "
               "best_gbps=%.3f med_gbps=%.3f ns_per_elem=%.2f cpu=%d khz=%ld\n",
               t->name, t->type, size >> 20, g_stride, pats[p].name, n, best / 1e6, med / 1e6, gbps(bytes, best),
               gbps(bytes, med), (double)med / (double)n, sched_getcpu(), cur_khz(cpu));
    }
    if (t->type < 0) return;
    // Clean costs on this target, after a dirtying w32 pass each time (except the "clean lines" rows).
    VkMappedMemoryRange rng = {VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE, NULL, t->mem, 0, VK_WHOLE_SIZE};
    struct { const char *name; int dirty; int kind; } fl[] = {
        {"vkFlush_dirty", 1, 0}, {"vkFlush_clean", 0, 0}, {"dccvac_dirty", 1, 1}, {"dccvac_clean", 0, 1},
        {"dccivac_dirty", 1, 2}, {"vkInvalidate", 0, 3}};
    for (unsigned k = 0; k < sizeof fl / sizeof fl[0]; k++) {
        uint64_t tt[64];
        int reps = g_reps > 64 ? 64 : g_reps;
        for (int r = 0; r < reps; r++) {
            if (fl[k].dirty) pat_w32(t->ptr, n, a, b, c);
            else cache_lines(t->ptr, size, 0);
            uint64_t t0 = now_ns();
            switch (fl[k].kind) {
            case 0: CK(pFlush(dev, 1, &rng)); break;
            case 1: cache_lines(t->ptr, size, 0); break;
            case 2: cache_lines(t->ptr, size, 1); break;
            case 3: CK(pInval(dev, 1, &rng)); break;
            }
            tt[r] = now_ns() - t0;
        }
        qsort(tt, reps, sizeof tt[0], cmp_u64);
        printf("membench: CLEAN target=%s type=%d size_mb=%zu op=%s best_us=%.1f med_us=%.1f med_gbps=%.3f\n", t->name,
               t->type, size >> 20, fl[k].name, tt[0] / 1e3, tt[reps / 2] / 1e3, gbps(size, tt[reps / 2]));
    }
}

static void flags_str(VkMemoryPropertyFlags f, char *s, size_t n) {
    snprintf(s, n, "%s%s%s%s%s%s", f & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT ? "DL|" : "",
             f & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT ? "HV|" : "", f & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT ? "HC|" : "",
             f & VK_MEMORY_PROPERTY_HOST_CACHED_BIT ? "HCACHED|" : "",
             f & VK_MEMORY_PROPERTY_LAZILY_ALLOCATED_BIT ? "LAZY|" : "",
             f & VK_MEMORY_PROPERTY_PROTECTED_BIT ? "PROT|" : "");
    size_t l = strlen(s);
    if (l && s[l - 1] == '|') s[l - 1] = 0;
    if (!l) snprintf(s, n, "0");
}

int main(int argc, char **argv) {
    int cpu = 7, any_driver = 0;
    const char *icd = "/usr/lib/mali_bridge/mali_bridge_icd.json";
    size_t big_mb = 32, small_mb = 2;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--cpu") && i + 1 < argc) cpu = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--big") && i + 1 < argc) big_mb = strtoul(argv[++i], NULL, 0);
        else if (!strcmp(argv[i], "--small") && i + 1 < argc) small_mb = strtoul(argv[++i], NULL, 0);
        else if (!strcmp(argv[i], "--stride") && i + 1 < argc) g_stride = strtoul(argv[++i], NULL, 0);
        else if (!strcmp(argv[i], "--reps") && i + 1 < argc) g_reps = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--icd") && i + 1 < argc) icd = argv[++i];
        else if (!strcmp(argv[i], "--any-driver")) any_driver = 1;
        else { printf("membench: usage: %s [--cpu N] [--big MB] [--small MB] [--stride B>=40] [--reps R] [--icd JSON] [--any-driver]\n", argv[0]); return 2; }
    }
    if (g_stride < 40) g_stride = 40;
    if (g_reps < 1) g_reps = 1;
    // Always the bridge ICD (the session exports VK_DRIVER_FILES for its own driver, PanVK: run 1 of 11-05 18:44
    // measured PanVK because the old code kept an existing value); --icd overrides, --any-driver skips the check.
    setenv("VK_DRIVER_FILES", icd, 1);
    unsetenv("VK_ICD_FILENAMES");
    setenv("HYBRIS_USE_VENDOR_NAMESPACE", "1", 0);
    setenv("VK_LOADER_LAYERS_DISABLE", "~implicit~", 0);
    printf("membench: START pid=%d driver_files=%s layers_disable=%s stride=%zu reps=%d big_mb=%zu small_mb=%zu\n",
           getpid(), getenv("VK_DRIVER_FILES"), getenv("VK_LOADER_LAYERS_DISABLE"), g_stride, g_reps, big_mb, small_mb);

    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(cpu, &set);
    int aff = sched_setaffinity(0, sizeof set, &set);
    printf("membench: AFFINITY cpu=%d rc=%d errno=%d now_on=%d khz=%ld\n", cpu, aff, aff ? errno : 0, sched_getcpu(),
           cur_khz(cpu));

    VkApplicationInfo app = {VK_STRUCTURE_TYPE_APPLICATION_INFO, NULL, "upload-membench", 1, NULL, 0, VK_API_VERSION_1_1};
    VkInstanceCreateInfo ici = {VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, NULL, 0, &app, 0, NULL, 0, NULL};
    VkInstance inst;
    CK(vkCreateInstance(&ici, NULL, &inst));
    uint32_t nd = 1;
    VkPhysicalDevice phys;
    VkResult er = vkEnumeratePhysicalDevices(inst, &nd, &phys);
    if ((er != VK_SUCCESS && er != VK_INCOMPLETE) || !nd) {
        printf("membench: FAIL no physical device (%d, %u)\nmembench: DONE rc=1\n", er, nd);
        return 1;
    }
    VkPhysicalDeviceProperties pp;
    vkGetPhysicalDeviceProperties(phys, &pp);
    printf("membench: DEVICE name=\"%s\" driver=0x%x api=%u.%u nonCoherentAtomSize=%llu\n", pp.deviceName,
           pp.driverVersion, VK_API_VERSION_MAJOR(pp.apiVersion), VK_API_VERSION_MINOR(pp.apiVersion),
           (unsigned long long)pp.limits.nonCoherentAtomSize);
    VkPhysicalDeviceDriverProperties drv = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES};
    VkPhysicalDeviceProperties2 pp2 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, &drv};
    vkGetPhysicalDeviceProperties2(phys, &pp2);
    printf("membench: DRIVER id=%d name=\"%s\" info=\"%s\"\n", drv.driverID, drv.driverName, drv.driverInfo);
    if (drv.driverID != VK_DRIVER_ID_ARM_PROPRIETARY && !any_driver) {
        printf("membench: FAIL not the vendor driver (driverID %d, want %d = ARM proprietary); --any-driver to allow\n"
               "membench: DONE rc=3\n", drv.driverID, VK_DRIVER_ID_ARM_PROPRIETARY);
        return 3;
    }
    VkPhysicalDeviceMemoryProperties mp;
    vkGetPhysicalDeviceMemoryProperties(phys, &mp);
    for (uint32_t i = 0; i < mp.memoryHeapCount; i++)
        printf("membench: HEAP %u size_mb=%llu flags=0x%x\n", i, (unsigned long long)(mp.memoryHeaps[i].size >> 20),
               mp.memoryHeaps[i].flags);
    int t_coh = -1, t_cached = -1, t_cached_coh = -1;
    for (uint32_t i = 0; i < mp.memoryTypeCount; i++) {
        char fs[96];
        VkMemoryPropertyFlags f = mp.memoryTypes[i].propertyFlags;
        flags_str(f, fs, sizeof fs);
        printf("membench: TYPE %u heap=%u flags=0x%x %s\n", i, mp.memoryTypes[i].heapIndex, f, fs);
        if (!(f & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)) continue;
        int coh = !!(f & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT), cac = !!(f & VK_MEMORY_PROPERTY_HOST_CACHED_BIT);
        if (coh && !cac && t_coh < 0) t_coh = (int)i;
        if (cac && !coh && t_cached < 0) t_cached = (int)i;
        if (cac && coh && t_cached_coh < 0) t_cached_coh = (int)i;
    }
    if (t_coh < 0) // vkd3d/DXVK take the first HOST_VISIBLE|HOST_COHERENT type; fall back to any coherent one
        for (uint32_t i = 0; i < mp.memoryTypeCount; i++)
            if ((mp.memoryTypes[i].propertyFlags & (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) ==
                (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) { t_coh = (int)i; break; }
    printf("membench: PICK a_coherent=%d b_cached_noncoherent=%d cached_coherent=%d\n", t_coh, t_cached, t_cached_coh);

    float prio = 1.0f;
    VkDeviceQueueCreateInfo qci = {VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO, NULL, 0, 0, 1, &prio};
    VkDeviceCreateInfo dci = {VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, NULL, 0, 1, &qci, 0, NULL, 0, NULL, NULL};
    CK(vkCreateDevice(phys, &dci, NULL, &dev));
    pFlush = (PFN_vkFlushMappedMemoryRanges)vkGetDeviceProcAddr(dev, "vkFlushMappedMemoryRanges");
    pInval = (PFN_vkInvalidateMappedMemoryRanges)vkGetDeviceProcAddr(dev, "vkInvalidateMappedMemoryRanges");

    // Which types a buffer / image of the kinds vkd3d places in an UPLOAD heap may use (a type remap must stay legal).
    VkBufferCreateInfo bci = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, NULL, 0, 1 << 20,
                              VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                                  VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                  VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT |
                                  VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT | VK_BUFFER_USAGE_UNIFORM_TEXEL_BUFFER_BIT,
                              VK_SHARING_MODE_EXCLUSIVE, 0, NULL};
    VkBuffer buf;
    CK(vkCreateBuffer(dev, &bci, NULL, &buf));
    VkMemoryRequirements mr;
    vkGetBufferMemoryRequirements(dev, buf, &mr);
    printf("membench: REQ buffer memoryTypeBits=0x%x alignment=%llu\n", mr.memoryTypeBits, (unsigned long long)mr.alignment);
    vkDestroyBuffer(dev, buf, NULL);
    VkImageCreateInfo imci = {VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, NULL, 0, VK_IMAGE_TYPE_2D, VK_FORMAT_R8G8B8A8_UNORM,
                              {256, 256, 1}, 1, 1, VK_SAMPLE_COUNT_1_BIT, VK_IMAGE_TILING_LINEAR,
                              VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                              VK_SHARING_MODE_EXCLUSIVE, 0, NULL, VK_IMAGE_LAYOUT_UNDEFINED};
    VkImage img;
    if (vkCreateImage(dev, &imci, NULL, &img) == VK_SUCCESS) {
        vkGetImageMemoryRequirements(dev, img, &mr);
        printf("membench: REQ linear_image memoryTypeBits=0x%x\n", mr.memoryTypeBits);
        vkDestroyImage(dev, img, NULL);
    }
    imci.tiling = VK_IMAGE_TILING_OPTIMAL;
    if (vkCreateImage(dev, &imci, NULL, &img) == VK_SUCCESS) {
        vkGetImageMemoryRequirements(dev, img, &mr);
        printf("membench: REQ optimal_image memoryTypeBits=0x%x\n", mr.memoryTypeBits);
        vkDestroyImage(dev, img, NULL);
    }

    size_t sizes[2] = {small_mb << 20, big_mb << 20};
    size_t maxsz = sizes[0] > sizes[1] ? sizes[0] : sizes[1];
    size_t maxn = maxsz / g_stride + 1;
    uint32_t *a = malloc(maxn * 12), *b = malloc(maxn * 12), *c = malloc(maxn * 16);
    if (!a || !b || !c) { printf("membench: FAIL malloc sources\nmembench: DONE rc=1\n"); return 1; }
    for (size_t i = 0; i < maxn * 3; i++) a[i] = (uint32_t)i * 2654435761u, b[i] = (uint32_t)i ^ 0x5a5a5a5au;
    for (size_t i = 0; i < maxn * 4; i++) c[i] = (uint32_t)i + 17u;

    struct target tg[3] = {{"a_coherent", t_coh}, {"b_cached", t_cached}, {"c_malloc", -1}};
    for (int s = 0; s < 2; s++) {
        size_t size = sizes[s];
        if (!size) continue;
        for (int k = 0; k < 3; k++) {
            struct target *t = &tg[k];
            if (k < 2 && t->type < 0) { printf("membench: SKIP target=%s (no such type)\n", t->name); continue; }
            t->size = size;
            if (k == 2) {
                t->ptr = aligned_alloc(4096, size);
                if (!t->ptr) { printf("membench: FAIL malloc %zu\n", size); continue; }
                memset(t->ptr, 0, size);
            } else {
                VkMemoryAllocateInfo mai = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, NULL, size, (uint32_t)t->type};
                VkResult r = vkAllocateMemory(dev, &mai, NULL, &t->mem);
                if (r != VK_SUCCESS) { printf("membench: FAIL alloc target=%s size_mb=%zu r=%d\n", t->name, size >> 20, r); continue; }
                void *p;
                r = vkMapMemory(dev, t->mem, 0, VK_WHOLE_SIZE, 0, &p);
                if (r != VK_SUCCESS) { printf("membench: FAIL map target=%s r=%d\n", t->name, r); vkFreeMemory(dev, t->mem, NULL); continue; }
                t->ptr = p;
            }
            bench_target(t, size, a, b, c, cpu);
            if (k == 2) free(t->ptr);
            else { vkUnmapMemory(dev, t->mem); vkFreeMemory(dev, t->mem, NULL); }
            t->ptr = NULL;
        }
    }
    vkDestroyDevice(dev, NULL);
    vkDestroyInstance(inst, NULL);
    printf("membench: DONE rc=0\n");
    return 0;
}

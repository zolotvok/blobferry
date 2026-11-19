// SPDX-License-Identifier: GPL-3.0-or-later
// cl_state.h: shared state: InstanceData, DeviceData, CmdBufData, lookups, budget, MEMLOG/ADDRMAP/BCD_RING. Included by mali_compat_layer.c.
// -----------------------------------------------------------------------------
// Layer Data Structures & Dispatch Tables
// -----------------------------------------------------------------------------
struct InstanceData {
    uint32_t app_api; // VkApplicationInfo::apiVersion of this instance (0 = none given = Vulkan 1.0); MALI_COMPAT_BDA_EXT
    VkInstance instance;
    void *dispatch_key;
    PFN_vkGetInstanceProcAddr get_instance_proc_addr;
    PFN_vkDestroyInstance fpDestroyInstance;
    PFN_vkEnumeratePhysicalDevices fpEnumeratePhysicalDevices;
    PFN_vkGetPhysicalDeviceFeatures fpGetPhysicalDeviceFeatures;
    PFN_vkGetPhysicalDeviceFeatures2 fpGetPhysicalDeviceFeatures2;
    PFN_vkGetPhysicalDeviceFormatProperties fpGetPhysicalDeviceFormatProperties;
    PFN_vkGetPhysicalDeviceFormatProperties2 fpGetPhysicalDeviceFormatProperties2;
    PFN_vkGetPhysicalDeviceImageFormatProperties fpGetPhysicalDeviceImageFormatProperties;
    PFN_vkGetPhysicalDeviceImageFormatProperties2 fpGetPhysicalDeviceImageFormatProperties2;
    PFN_vkGetPhysicalDeviceMemoryProperties fpGetPhysicalDeviceMemoryProperties;
    PFN_vkGetPhysicalDeviceMemoryProperties2 fpGetPhysicalDeviceMemoryProperties2;
    PFN_vkCreateDevice fpCreateDevice;
    PFN_vkEnumerateDeviceExtensionProperties fpEnumerateDeviceExtensionProperties;
    PFN_vkGetPhysicalDeviceProperties fpGetPhysicalDeviceProperties;
    PFN_vkGetPhysicalDeviceProperties2 fpGetPhysicalDeviceProperties2;
};

struct TrackedBCImage {
    VkImage image;
    VkFormat orig_format;
    VkFormat decoded_format;
    VkExtent3D extent;
};

// BC images, by handle. A hash table: the first version held 256 in an array and silently dropped the rest, so in a
// game world (thousands of DXT textures) most views kept their BC format on the RGBA8 image and sampled black and
// their uploads were not decoded. [Claude Opus 5.5] 2026-10-26
struct BCNode { struct TrackedBCImage t; struct BCNode *next; };
#define BC_BUCKETS 4096
static struct BCNode *g_bc_images[BC_BUCKETS];
static pthread_mutex_t g_bc_image_mutex = PTHREAD_MUTEX_INITIALIZER;
// F2: tracked BC images; while 0 (games without BC textures, or before the first one) find_bc_image takes no lock.
static _Atomic uint32_t g_bc_image_n;

static uint32_t handle_bucket(uint64_t v, uint32_t n) { return (uint32_t)((v ^ (v >> 17) ^ (v >> 33)) % n); }

static void register_bc_image(VkImage img, VkFormat orig, VkFormat decoded, VkExtent3D ext) {
    struct BCNode *e = calloc(1, sizeof *e);
    if (!e) return;
    e->t = (struct TrackedBCImage){.image = img, .orig_format = orig, .decoded_format = decoded, .extent = ext};
    pthread_mutex_lock(&g_bc_image_mutex);
    uint32_t bkt = handle_bucket((uint64_t)(uintptr_t)img, BC_BUCKETS);
    e->next = g_bc_images[bkt];
    g_bc_images[bkt] = e;
    atomic_fetch_add_explicit(&g_bc_image_n, 1, memory_order_release);
    pthread_mutex_unlock(&g_bc_image_mutex);
}

// Copies the entry out (the node can be freed by another thread's vkDestroyImage).
static bool find_bc_image(VkImage img, struct TrackedBCImage *out) {
    // the image was created before the app records a copy to it (app-side ordering), so its count is visible here
    if (!atomic_load_explicit(&g_bc_image_n, memory_order_acquire)) return false;
    bool found = false;
    pthread_mutex_lock(&g_bc_image_mutex);
    for (struct BCNode *e = g_bc_images[handle_bucket((uint64_t)(uintptr_t)img, BC_BUCKETS)]; e; e = e->next)
        if (e->t.image == img) { *out = e->t; found = true; break; }
    pthread_mutex_unlock(&g_bc_image_mutex);
    return found;
}

static void unregister_bc_image(VkImage img) {
    pthread_mutex_lock(&g_bc_image_mutex);
    struct BCNode **pp = &g_bc_images[handle_bucket((uint64_t)(uintptr_t)img, BC_BUCKETS)];
    while (*pp && (*pp)->t.image != img) pp = &(*pp)->next;
    struct BCNode *e = *pp;
    if (e) {
        *pp = e->next;
        atomic_fetch_sub_explicit(&g_bc_image_n, 1, memory_order_relaxed);
    }
    pthread_mutex_unlock(&g_bc_image_mutex);
    free(e);
}

#include <sys/syscall.h>

// K-CF2 port, env-gated (stage kcs1+kcf2+memlog) [keeper, Claude Opus 5.5] 2026-10-31: live bytes per memory heap,
// added when vkAllocateMemory succeeds and taken off again in vkFreeMemory (docs/common_fixes.md sections 15, 15b).
// Always counted (a few atomics); reported as heapUsage only with MALI_COMPAT_BUDGET_MB=<MiB>, which also sets the
// cap. Unset, the report is the old one (g_allocated_bytes, bytes ever allocated, on every heap; 6 GiB cap), byte for
// byte as in the deployed 2dfc35b8. Counted: app allocations (the layer's own dma-heap export included), host-pointer
// imports, BC staging buffers. Not counted: app dma-buf fd imports (the exporter's pages).
#define BUDGET_NO_HEAP 0xFFu
static _Atomic uint64_t g_heap_usage[VK_MAX_MEMORY_HEAPS];
static _Atomic uint64_t g_heap_peak[VK_MAX_MEMORY_HEAPS];
static uint64_t g_heap_budget[VK_MAX_MEMORY_HEAPS]; // reported heapBudget, for the log
static _Atomic uint32_t g_heap_count;
static _Atomic uint64_t g_live_allocs, g_budget_queries;
static bool g_budget_accounted;            // MALI_COMPAT_BUDGET_MB set: live usage and that cap are reported
static uint64_t g_budget_cap = 6ULL << 30; // the old fixed cap unless MALI_COMPAT_BUDGET_MB
static uint32_t g_budget_log_s;            // budget log interval in s (accounted + GPU_TIME or DEBUG), 0 = off

static void budget_add(uint32_t heap, uint64_t size) {
    if (heap >= VK_MAX_MEMORY_HEAPS) return;
    uint64_t now = atomic_fetch_add(&g_heap_usage[heap], size) + size;
    atomic_fetch_add(&g_live_allocs, 1);
    uint64_t p = atomic_load(&g_heap_peak[heap]);
    while (now > p && !atomic_compare_exchange_weak(&g_heap_peak[heap], &p, now)) {}
}

static void budget_sub(uint32_t heap, uint64_t size) {
    if (heap >= VK_MAX_MEMORY_HEAPS) return;
    atomic_fetch_sub(&g_heap_usage[heap], size);
    atomic_fetch_sub(&g_live_allocs, 1);
}

static uint64_t budget_live_mb(void) {
    uint64_t t = 0;
    for (uint32_t i = 0; i < VK_MAX_MEMORY_HEAPS; i++) t += atomic_load(&g_heap_usage[i]);
    return t >> 20;
}

// One line per interval: per heap live usage / reported budget (peak since start), live allocations and how often the
// app read the budget (0 queries = the game does not stream by it, whatever the numbers say).
static void *budget_log_thread(void *arg) {
    (void)arg;
    for (;;) {
        sleep(g_budget_log_s);
        char line[1024];
        size_t n = 0;
        uint32_t hc = atomic_load(&g_heap_count);
        for (uint32_t i = 0; i < hc && i < VK_MAX_MEMORY_HEAPS && n < sizeof line; i++)
            n += (size_t)snprintf(line + n, sizeof line - n, " heap%u %llu/%llu MiB (peak %llu)", i,
                                  (unsigned long long)(atomic_load(&g_heap_usage[i]) >> 20),
                                  (unsigned long long)(g_heap_budget[i] >> 20),
                                  (unsigned long long)(atomic_load(&g_heap_peak[i]) >> 20));
        if (n >= sizeof line) n = sizeof line - 1;
        line[n] = '\0';
        fprintf(stderr, "mali-compat: budget%s, %llu allocs, %llu queries\n", line,
                (unsigned long long)atomic_load(&g_live_allocs), (unsigned long long)atomic_load(&g_budget_queries));
    }
    return NULL;
}

// Memory event log, MALI_COMPAT_MEMLOG=1 (=2: without the capture-address query) [keeper, Claude Opus 5.5] 2026-10-31.
// File /root/mali-memlog-<comm>.tsv (opened for append; each process writes a '#' line, then the header), one TSV line
// per event, CLOCK_REALTIME ms (lines up with logcat):  t_ms ev mem size type heap va tid live_mb
//   A  vkAllocateMemory  va = vkGetDeviceMemoryOpaqueCaptureAddress (blob-defined; '-' if 0 or MEMLOG=2)
//   M  map               va = returned pointer - offset (placed maps: the driver's own address) = SAME_VA GPU VA
//   U  unmap             va = the mapping's va        F  vkFreeMemory  va = the last mapped va ('-' if never mapped)
//   X  failed alloc      va = the VkResult            S  every 100th queue submit, mem = the submit count
// heap '-' = not counted in the budget (dma-buf import); live_mb = counted live bytes of all heaps after the event.
// Buffered, written once per second by a thread, at vkDestroyDevice and at exit. Cap MALI_COMPAT_MEMLOG_MB (default
// 64 MiB of lines per process), then one TRUNC line and the log stops. Unset: no file, no thread, no hooks.
#define ML_NONE UINT32_MAX
#define ML_BUF (256u << 10)
static int g_memlog; // 0 off, 1 on, 2 on without the capture-address query
static bool g_addrmap; // MALI_COMPAT_ADDRMAP=1, see "GPU address map" below (addrmap-1113)
static pthread_mutex_t g_ml_mutex = PTHREAD_MUTEX_INITIALIZER;
static int g_ml_fd = -1;
static char *g_ml_buf;
static size_t g_ml_len, g_ml_bufsz = ML_BUF; // ADDRMAP: 4 MiB, so a flush inside an event is rare
static uint64_t g_ml_bytes, g_ml_cap = 64ULL << 20; // bytes taken so far (written + buffered), cap
static bool g_ml_stopped;
static _Atomic uint64_t g_ml_submits;

static uint64_t ml_now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    return (uint64_t)ts.tv_sec * 1000u + (uint64_t)ts.tv_nsec / 1000000u;
}

// Called with g_ml_mutex held.
static void ml_flush_locked(void) {
    size_t off = 0;
    while (g_ml_fd >= 0 && off < g_ml_len) {
        ssize_t w = write(g_ml_fd, g_ml_buf + off, g_ml_len - off);
        if (w < 0 && errno == EINTR) continue;
        if (w <= 0) break;
        off += (size_t)w;
    }
    g_ml_len = 0;
}

// Called with g_ml_mutex held.
static void ml_append_locked(const char *line, size_t n) {
    if (g_ml_len + n > g_ml_bufsz) ml_flush_locked();
    if (n > g_ml_bufsz) return;
    memcpy(g_ml_buf + g_ml_len, line, n);
    g_ml_len += n;
}

static void ml_write(const char *line, size_t n) {
    pthread_mutex_lock(&g_ml_mutex);
    if (!g_ml_stopped && g_ml_buf) {
        if (g_ml_bytes + n > g_ml_cap) {
            char t[128];
            int k = snprintf(t, sizeof t, "%llu\tTRUNC\t-\t%llu\t-\t-\t-\t%ld\t%llu\n", (unsigned long long)ml_now_ms(),
                             (unsigned long long)g_ml_bytes, (long)syscall(SYS_gettid), (unsigned long long)budget_live_mb());
            if (k > 0) ml_append_locked(t, (size_t)k < sizeof t ? (size_t)k : sizeof t - 1);
            ml_flush_locked();
            g_ml_stopped = true;
            if (g_ml_fd >= 0) close(g_ml_fd);
            g_ml_fd = -1;
        } else {
            ml_append_locked(line, n);
            g_ml_bytes += n;
        }
    }
    pthread_mutex_unlock(&g_ml_mutex);
}

// ADDRMAP: events written per letter (A..Z), for the end line; the thread id is cached per thread (no syscall per event).
static _Atomic uint64_t g_am_count[26];
static __thread long tl_ml_tid;
static long ml_tid(void) {
    if (!g_addrmap) return (long)syscall(SYS_gettid); // MEMLOG alone: as before
    if (!tl_ml_tid) tl_ml_tid = (long)syscall(SYS_gettid);
    return tl_ml_tid;
}

// va_kind: 0 = '-', 1 = hex address, 2 = signed decimal (VkResult)
static void ml_event(char ev, uint64_t mem, uint64_t size, uint32_t type, uint32_t heap, uint64_t va, int va_kind) {
    if (!g_memlog) return;
    char line[256], ty[16] = "-", hp[16] = "-", vs[32] = "-";
    if (type != ML_NONE) snprintf(ty, sizeof ty, "%u", type);
    if (heap < VK_MAX_MEMORY_HEAPS) snprintf(hp, sizeof hp, "%u", heap);
    if (va_kind == 1) snprintf(vs, sizeof vs, "0x%llx", (unsigned long long)va);
    else if (va_kind == 2) snprintf(vs, sizeof vs, "%lld", (long long)(int64_t)va);
    int n = snprintf(line, sizeof line, "%llu\t%c\t0x%llx\t%llu\t%s\t%s\t%s\t%ld\t%llu\n", (unsigned long long)ml_now_ms(),
                     ev, (unsigned long long)mem, (unsigned long long)size, ty, hp, vs, ml_tid(),
                     (unsigned long long)budget_live_mb());
    if (g_addrmap && ev >= 'A' && ev <= 'Z') atomic_fetch_add_explicit(&g_am_count[ev - 'A'], 1, memory_order_relaxed);
    if (n > 0) ml_write(line, (size_t)n < sizeof line ? (size_t)n : sizeof line - 1);
}

static void ml_flush(void) {
    pthread_mutex_lock(&g_ml_mutex);
    if (g_ml_buf) ml_flush_locked();
    pthread_mutex_unlock(&g_ml_mutex);
}

static void am_tick(void);
static void *ml_flush_thread(void *arg) {
    (void)arg;
    for (unsigned tick = 1;; tick++) {
        sleep(1);
        pthread_mutex_lock(&g_ml_mutex);
        bool stop = g_ml_stopped;
        if (!stop) ml_flush_locked();
        pthread_mutex_unlock(&g_ml_mutex);
        if (stop) return NULL;
        if (g_addrmap && tick % 10 == 0) am_tick(); // clock pair line + bc-misc counters
    }
}

// Once per process. Leaves g_memlog 0 when the file cannot be opened.
static void am_clock_line(const char *what);
static void am_atexit(void);
static void am_end(const char *why);
static void ml_open(const char *path, const char *comm) {
    if (g_addrmap) g_ml_bufsz = 4u << 20;
    g_ml_buf = malloc(g_ml_bufsz);
    g_ml_fd = g_ml_buf ? open(path, O_WRONLY | O_CREAT | O_APPEND | O_CLOEXEC, 0644) : -1;
    if (g_ml_fd < 0) {
        fprintf(stderr, "mali-compat: memlog: cannot open %s: %s\n", path, strerror(errno));
        free(g_ml_buf);
        g_ml_buf = NULL;
        g_memlog = 0;
        g_addrmap = false; // no file: no address map either (set before any device, so no hook is handed out)
        return;
    }
    char h[256];
    int n = snprintf(h, sizeof h, "# mali-compat memlog pid %d comm %s start_ms %llu cap %llu MiB%s\n"
                     "t_ms\tev\tmem\tsize\ttype\theap\tva\ttid\tlive_mb\n", (int)getpid(), comm,
                     (unsigned long long)ml_now_ms(), (unsigned long long)(g_ml_cap >> 20),
                     g_memlog == 2 ? ", no capture address" : "");
    if (n > 0) ml_write(h, (size_t)n < sizeof h ? (size_t)n : sizeof h - 1);
    if (g_addrmap) {
        am_clock_line("start");
        atexit(am_atexit); // registered before ml_flush, so it runs after it (atexit is LIFO): stderr only
    }
    atexit(ml_flush);
    pthread_t t;
    if (pthread_create(&t, NULL, ml_flush_thread, NULL) == 0) pthread_detach(t);
}

static uint64_t ml_submit_tick(void) {
    uint64_t n = atomic_fetch_add(&g_ml_submits, 1) + 1;
    if (n % 100 == 0 && !g_addrmap) ml_event('S', n, 0, ML_NONE, BUDGET_NO_HEAP, 0, 0); // ADDRMAP: every submit, am_submit
    return n;
}

// ------------------------------------------------------------------------------------------------------------------
// GPU address map, MALI_COMPAT_ADDRMAP=1 [game/doom_eternal keeper, Claude Opus 5.5] 2026-11-13 (addrmap-1113).
// Diagnostic for a single vendor GPU MMU fault (DOOM Eternal title freeze): which Vulkan object, if any, lay at the
// faulting GPU VA. Turns the memlog on (as MALI_COMPAT_MEMLOG=1 unless MEMLOG is set; MEMLOG=2 keeps "no capture
// address") and adds to the same TSV /root/mali-memlog-<comm>.tsv. The 9 memlog columns stay; ADDRMAP events add a
// 10th column "detail" with space-separated k=v pairs (A/M/U/F/X keep 9 columns). Default cap 256 MiB under ADDRMAP
// (MALI_COMPAT_MEMLOG_MB still wins). Events (mem / size / type / heap / va columns, then detail):
//   B  bind       mem = VkDeviceMemory, size = the object's vkGet*MemoryRequirements size ('-' if not queried), va =
//                 the memory's capture address + offset ('-' unknown); detail kind=buf|img obj off req_align and the
//                 object's description (buffers: csize usage flags; images: fmt (driver-side) bcfmt (app's BC format,
//                 '-' if not a BC shadow) ext mips layers samples tiling usage flags (driver-side) appflags afrc
//                 (2bpc|3bpc|none: fixed-rate chained by the layer) cc (compression-control flags the driver got, '-'
//                 none) [plane] [swapchain])
//   D  destroy    mem = the object handle, size/va = its bind (req size, va) or '-'; detail kind obj mem off|unbound
//   L  layer alloc / R layer free: memory the layer itself allocates from the driver (not through compat_AllocateMemory):
//                 mem = VkDeviceMemory, size, type, heap, va = the staging buffer's device address (bc-staging) or the
//                 capture address; detail purpose=bc-staging|ct-buffer|rt-ring|rt-zero|rt-scratch [buf bucket]
//   X  (layer)    a failed layer allocation: va = VkResult, detail purpose=...
//   N  marker     mem = the object handle; detail what=dpool-create|dpool-reset|dpool-destroy|cpool-create|cpool-reset|
//                 cpool-destroy|cb-alloc|cb-free|qpool-create|qpool-destroy|qpool-layer|swapchain-create|swapchain-destroy
//   S  submit     every vkQueueSubmit/2 (MEMLOG alone: every 100th): mem = the submit count, size = submitCount;
//                 detail v=1|2 q cbs (all command buffers in the call) cb=<first up to 4> fence
//   C  bc-misc    the first 16 copies/blits/resolves touching an emulated-BC image, and image views on one whose
//                 format differs from the image's BC format; detail call ...
// '#' lines: "# addrmap clock start|tick realtime_ms R monotonic_ms M pid P" (both clocks read back to back) at the
// start and every 10 s; the blob's fault print carries CLOCK_MONOTONIC seconds, logcat wall time.
// stderr: "mali-compat: addrmap on (MALI_COMPAT_ADDRMAP): ..." once, "mali-compat: addrmap: bc-misc ..." when those
// counters change (10-s tick), "mali-compat: addrmap: end (device destroy|exit): ..." with the event counts.
// Unset: no hook below is handed out; the always-hooked paths (vkCreateImage/Buffer/ImageView, vkDestroyImage, command
// pool/buffer calls, BC staging) test one bool.
// ------------------------------------------------------------------------------------------------------------------
static _Atomic uint64_t g_am_bcmisc[9], g_am_bcmisc_logged, g_am_c_events;
static const char *const g_am_bcmisc_name[9] = {"copy-src", "copy-dst", "blit-src", "blit-dst", "copy-to-buf",
                                                "resolve-src", "resolve-dst", "view-bcfmt", "view-nonbc"};
static uint64_t am_mono_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000u + (uint64_t)ts.tv_nsec / 1000000u;
}
static void am_clock_line(const char *what) {
    uint64_t rt = ml_now_ms(), mono = am_mono_ms();
    char l[160];
    int n = snprintf(l, sizeof l, "# addrmap clock %s realtime_ms %llu monotonic_ms %llu pid %d\n", what,
                     (unsigned long long)rt, (unsigned long long)mono, (int)getpid());
    if (n > 0) ml_write(l, (size_t)n < sizeof l ? (size_t)n : sizeof l - 1);
}

// One ADDRMAP line: the 9 memlog columns, then the detail column (printf format). One snprintf + one vsnprintf into
// a stack buffer, then the append under g_ml_mutex (ml_write); no other lock is held or taken here.
#define AM_NA UINT64_MAX // size / va column '-'
__attribute__((format(printf, 8, 9)))
static void am_emit(char ev, uint64_t mem, uint64_t size, uint32_t type, uint32_t heap, uint64_t va, int va_kind,
                    const char *fmt, ...) {
    if (!g_addrmap || !g_memlog) return;
    char line[768], ty[16] = "-", hp[16] = "-", vs[32] = "-", sz[24] = "-";
    if (type != ML_NONE) snprintf(ty, sizeof ty, "%u", type);
    if (heap < VK_MAX_MEMORY_HEAPS) snprintf(hp, sizeof hp, "%u", heap);
    if (size != AM_NA) snprintf(sz, sizeof sz, "%llu", (unsigned long long)size);
    if (va_kind == 1 && va != AM_NA) snprintf(vs, sizeof vs, "0x%llx", (unsigned long long)va);
    else if (va_kind == 2) snprintf(vs, sizeof vs, "%lld", (long long)(int64_t)va);
    int n = snprintf(line, sizeof line, "%llu\t%c\t0x%llx\t%s\t%s\t%s\t%s\t%ld\t%llu\t", (unsigned long long)ml_now_ms(), ev,
                     (unsigned long long)mem, sz, ty, hp, vs, ml_tid(), (unsigned long long)budget_live_mb());
    if (n <= 0 || (size_t)n >= sizeof line - 2) return;
    va_list ap;
    va_start(ap, fmt);
    int k = vsnprintf(line + n, sizeof line - (size_t)n - 1, fmt, ap);
    va_end(ap);
    if (k < 0) k = 0;
    size_t len = (size_t)n + ((size_t)k < sizeof line - (size_t)n - 1 ? (size_t)k : sizeof line - (size_t)n - 2);
    line[len++] = '\n';
    if (ev >= 'A' && ev <= 'Z') atomic_fetch_add_explicit(&g_am_count[ev - 'A'], 1, memory_order_relaxed);
    ml_write(line, len);
}

// Layer-internal driver memory: L after a successful allocation (X on failure), R before the free.
static void am_layer_alloc(const char *purpose, VkResult r, VkDeviceMemory m, uint64_t size, uint32_t type, uint32_t heap,
                           uint64_t va, uint64_t buf, uint64_t bucket) {
    if (r != VK_SUCCESS) { am_emit('X', 0, size, type, heap, (uint64_t)(int64_t)r, 2, "purpose=%s layer=1", purpose); return; }
    if (buf) am_emit('L', (uint64_t)(uintptr_t)m, size, type, heap, va ? va : AM_NA, 1, "purpose=%s buf=0x%llx bucket=%llu",
                     purpose, (unsigned long long)buf, (unsigned long long)bucket);
    else am_emit('L', (uint64_t)(uintptr_t)m, size, type, heap, va ? va : AM_NA, 1, "purpose=%s", purpose);
}
static void am_layer_free(const char *purpose, VkDeviceMemory m, uint64_t size, uint32_t type, uint32_t heap, uint64_t va) {
    am_emit('R', (uint64_t)(uintptr_t)m, size, type, heap, va ? va : AM_NA, 1, "purpose=%s", purpose);
}

struct StagingResource {
    VkBuffer buffer;
    VkDeviceMemory memory;
    VkDeviceSize size;      // power-of-two bucket, >= STAGING_MIN
    VkDeviceAddress addr;
    VkDeviceSize mem_size;  // counted in g_heap_usage[heap] while memory is set (0 = not counted)
    uint32_t heap;
    uint32_t type;          // memory type (MALI_COMPAT_ADDRMAP events)
};

// Decode staging buffers are pooled per device: a command buffer's buffers go back to the pool when it is begun,
// reset or freed again (the app may only do that after the GPU is done with it). Before, every BC upload did
// vkCreateBuffer + vkAllocateMemory + vkBindBufferMemory and later vkFreeMemory, all /dev/mali0 ioctls, on the app's
// recording thread while textures stream in. [Claude Opus 5.5] 2026-10-26
#define STAGING_MIN (64 * 1024)
#define STAGING_POOL_CAP (64ull * 1024 * 1024) // bytes kept for reuse; above that, buffers are freed
struct StagingPoolNode { struct StagingResource r; struct StagingPoolNode *next; };
static pthread_mutex_t g_staging_pool_mutex = PTHREAD_MUTEX_INITIALIZER;

// MALI_COMPAT_STATS=1: every ~5 s one line with the BC upload counts and the CPU time spent recording them
// (MALI_COMPAT_STATS=every: after each upload, for probes).
static bool g_stats;
static uint64_t g_stats_interval_ns = 5000000000ull;
static _Atomic uint64_t g_st_uploads, g_st_regions, g_st_bytes, g_st_new, g_st_reused, g_st_ns, g_st_last_ns;

// App state the BC decode overwrites inside the app's command buffer, restored after it (Review 6 H1): the bound
// compute pipeline and the push constants. Without that, an app that tracks its own bindings (DXVK, vkd3d-proton)
// dispatched its next compute work with the decoder's pipeline or push constants. [Claude Opus 5.5] 2026-10-26
#define PUSH_RECS 8
struct PushRec {
    VkPipelineLayout layout;
    VkShaderStageFlags stages;
    uint32_t offset, size;
    bool v2;                        // recorded from vkCmdPushConstants2(KHR): replayed with it (pc2-restore-1)
    uint8_t data[256];
};

// pc2-restore-1 (WD2 B10-18) [keeper, Claude Opus 5.5] 2026-11-01: DXVK pushes with vkCmdPushConstants2KHR
// (VK_KHR_maintenance6), which this layer did not hook, so the restore after a BC decode left DXVK's next compute
// dispatch with the decoder's 56 push bytes. Both entry points now feed the PushRec list above. Cumulative counters,
// one stderr line every 10 s once any is non-zero (pc2_log_thread).
// F3 (our-cpu-paths-1105.md) [keeper, Claude Opus 5.5] 2026-11-05: the counters were five shared atomics, one
// atomic add per push on one cache line from every recording thread (ER: 0.8 ms/s in __aarch64_ldadd8_relax over 7
// GXWorker threads). Each thread now counts in its own cache-line slot with a plain load + store (only it writes
// there); the log thread sums the slots. Threads past PC2_SLOTS - 1 share the last slot with atomic adds.
enum { PC2_V1, PC2_V2, PC2_V2_UNREC, PC2_DECODES, PC2_DECODES_V2, PC2_N };
#define PC2_SLOTS 64
struct Pc2Slot { _Atomic uint64_t n[PC2_N]; } __attribute__((aligned(64)));
static struct Pc2Slot g_pc2_slot[PC2_SLOTS];
static _Atomic uint32_t g_pc2_nslots;
static __thread struct Pc2Slot *tl_pc2;
static __thread bool tl_pc2_shared;
static inline void pc2_count(int k) {
    struct Pc2Slot *s = tl_pc2;
    if (!s) {
        uint32_t i = atomic_fetch_add_explicit(&g_pc2_nslots, 1, memory_order_relaxed);
        if (i >= PC2_SLOTS - 1) { i = PC2_SLOTS - 1; tl_pc2_shared = true; }
        tl_pc2 = s = &g_pc2_slot[i];
    }
    if (tl_pc2_shared)
        atomic_fetch_add_explicit(&s->n[k], 1, memory_order_relaxed);
    else
        atomic_store_explicit(&s->n[k], atomic_load_explicit(&s->n[k], memory_order_relaxed) + 1, memory_order_relaxed);
}
static uint64_t pc2_sum(int k) {
    uint64_t t = 0;
    for (uint32_t i = 0; i < PC2_SLOTS; i++) t += atomic_load_explicit(&g_pc2_slot[i].n[k], memory_order_relaxed);
    return t;
}

// ---- MALI_COMPAT_BCD_RING=1 (diagnostic, default off). DOOM Eternal 2026-11-13 13:11: the menu froze after one vendor
// GROUP_ERROR_FATAL (MMU translation fault, READ at GPU VA 0x5e080a37c0) while the pc2 counters stayed flat. The BC
// decode reads the app's source buffer and writes a staging buffer through raw device addresses, so this keeps the last
// BCR_N decode regions (sequence, command buffer, source buffer, the exact source bytes the shader reads, the staging
// bytes it writes, format, size, CLOCK_MONOTONIC ms) and the last BCR_FREE_N staging buffers given back to the pool or
// destroyed. Dumped to stderr by the pc2 log thread when v1 pushes and BC decodes are both flat for 2 ticks (20 s) after
// having moved (once per stall, at most 3), and once more at the first device destroy / process exit with new entries.
// Off: g_bcr stays NULL, the decode path pays one branch per region. [game/doom_eternal keeper, Claude Opus 5.5] 2026-11-13
#define BCR_N 4096
#define BCR_FREE_N 512
struct BcrEnt { uint64_t seq, ms, cmd, srcbuf, src, src_end, dst, dst_end; uint32_t fmt, w, h, slices, row_B, slice_B; };
struct BcrFree { uint64_t seq, ms, lo, hi; bool pooled; };
struct BcdRing {
    pthread_mutex_t m;
    uint64_t n, nf;              // entries ever written (the next sequence numbers)
    uint64_t dumped_n, dumped_nf; // n / nf at the last dump
    int dumps, stall_dumps;      // dumps so far (numbering), of them stall dumps (at most 3)
    bool end_done;
    struct BcrEnt e[BCR_N];
    struct BcrFree f[BCR_FREE_N];
};
static struct BcdRing *g_bcr; // set once in bcr_init (before the pc2 thread starts); NULL = key off
static uint64_t bcr_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000ull + (uint64_t)ts.tv_nsec / 1000000ull;
}
static void bcr_add(uint64_t cmd, uint64_t srcbuf, uint64_t src, uint64_t src_B, uint64_t dst, uint64_t dst_B, uint32_t fmt,
                    uint32_t w, uint32_t h, uint32_t slices, uint32_t row_B, uint32_t slice_B) {
    struct BcdRing *r = g_bcr;
    uint64_t ms = bcr_ms();
    pthread_mutex_lock(&r->m);
    r->e[r->n % BCR_N] = (struct BcrEnt){r->n, ms, cmd, srcbuf, src, src + src_B, dst, dst + dst_B, fmt, w, h, slices, row_B, slice_B};
    r->n++;
    pthread_mutex_unlock(&r->m);
}
static void bcr_free(uint64_t lo, uint64_t bytes, bool pooled) {
    struct BcdRing *r = g_bcr;
    uint64_t ms = bcr_ms();
    pthread_mutex_lock(&r->m);
    r->f[r->nf % BCR_FREE_N] = (struct BcrFree){r->nf, ms, lo, lo + bytes, pooled};
    r->nf++;
    pthread_mutex_unlock(&r->m);
}
// One dump: a copy taken under the lock, written without it. why = the header's reason, e.g. "stall: ... flat for 20 s".
// end = the one dump at device destroy / exit (only when there are entries since the last dump).
static void bcr_dump(const char *why, bool end) {
    struct BcdRing *r = g_bcr;
    struct BcrEnt *e = malloc(sizeof r->e);
    struct BcrFree *f = malloc(sizeof r->f);
    if (!e || !f) { free(e); free(f); return; }
    pthread_mutex_lock(&r->m);
    bool go = end ? !r->end_done && (r->n != r->dumped_n || r->nf != r->dumped_nf) : r->stall_dumps < 3;
    if (go && end) r->end_done = true;
    else if (go) r->stall_dumps++;
    uint64_t n = r->n, nf = r->nf;
    int k = go ? ++r->dumps : 0;
    if (go) {
        memcpy(e, r->e, sizeof r->e);
        memcpy(f, r->f, sizeof r->f);
        r->dumped_n = n;
        r->dumped_nf = nf;
    }
    pthread_mutex_unlock(&r->m);
    if (go) {
        uint64_t m = n < BCR_N ? n : BCR_N, mf = nf < BCR_FREE_N ? nf : BCR_FREE_N;
        fprintf(stderr, "mali-compat: bcd-ring dump %d (%s), entries %llu, staging frees %llu, now t %llu ms\n", k, why,
                (unsigned long long)m, (unsigned long long)mf, (unsigned long long)bcr_ms());
        for (uint64_t i = n - m; i < n; i++) {
            const struct BcrEnt *x = &e[i % BCR_N];
            fprintf(stderr, "mali-compat: bcd-ring #%llu t %llu cmd 0x%llx srcbuf 0x%llx read 0x%llx-0x%llx (row %u slice %u) "
                    "write 0x%llx-0x%llx fmt %u %ux%ux%u\n", (unsigned long long)x->seq, (unsigned long long)x->ms,
                    (unsigned long long)x->cmd, (unsigned long long)x->srcbuf, (unsigned long long)x->src,
                    (unsigned long long)x->src_end, x->row_B, x->slice_B, (unsigned long long)x->dst,
                    (unsigned long long)x->dst_end, x->fmt, x->w, x->h, x->slices);
        }
        for (uint64_t i = nf - mf; i < nf; i++) {
            const struct BcrFree *x = &f[i % BCR_FREE_N];
            fprintf(stderr, "mali-compat: bcd-ring free #%llu t %llu staging 0x%llx-0x%llx %s\n", (unsigned long long)x->seq,
                    (unsigned long long)x->ms, (unsigned long long)x->lo, (unsigned long long)x->hi,
                    x->pooled ? "pooled" : "destroyed");
        }
        fprintf(stderr, "mali-compat: bcd-ring dump %d end\n", k);
    }
    free(e);
    free(f);
}
// bcd-mix (MALI_COMPAT_BCD_RING=1) [game/doom_eternal keeper, Claude Opus 5.5] 2026-11-13 14:xx: the BC decode is
// recorded inline into the app's command buffer and restores only the last PUSH_RECS (8) push records; pushes past
// 256 bytes and v2 pushes with a null layout are not recorded. Measures whether DOOM mixes decodes with its own pushes /
// pipeline binds in one command buffer and whether records are lost. Counted only with the key (g_bcr); measurement
// only, the restore is unchanged. One line from the pc2 thread when a number changed, once more at the first end dump.
//   decodes           compat_CmdCopyBufferToImage calls on a tracked BC image that record the decode (staging acquired)
//   cb-with-app-push  of those: the command buffer held >= 1 push record (v1 or v2) at that moment
//   cb-with-draws     of those: the app had bound a graphics or compute pipeline in that command buffer since begin
//                     (vkCmdBindPipeline is always hooked; draws/dispatches are not hooked without GPU_TIME)
//   push-records-max  the most push records held at a decode (<= PUSH_RECS)
//   push-dropped      records dropped because the 8-slot table was full (every tracked command buffer)
//   push-unrecorded   pushes never recorded: v1 past 256 bytes (counted here) + v2 null layout / past 256 (PC2_V2_UNREC)
static _Atomic uint64_t g_bcm_dec, g_bcm_push, g_bcm_draw, g_bcm_drop, g_bcm_v1unrec;
static _Atomic uint32_t g_bcm_max;
static void bcm_decode(uint32_t push_count, bool draws) {
    atomic_fetch_add_explicit(&g_bcm_dec, 1, memory_order_relaxed);
    if (push_count) atomic_fetch_add_explicit(&g_bcm_push, 1, memory_order_relaxed);
    if (draws) atomic_fetch_add_explicit(&g_bcm_draw, 1, memory_order_relaxed);
    uint32_t m = atomic_load_explicit(&g_bcm_max, memory_order_relaxed);
    while (push_count > m && !atomic_compare_exchange_weak_explicit(&g_bcm_max, &m, push_count, memory_order_relaxed,
                                                                    memory_order_relaxed)) {}
}
// end: the first end dump (device destroy / exit) prints even when nothing changed.
static void bcm_line(bool end) {
    static pthread_mutex_t mx = PTHREAD_MUTEX_INITIALIZER;
    static unsigned long long last[6];
    static bool end_done;
    unsigned long long v[6] = {atomic_load_explicit(&g_bcm_dec, memory_order_relaxed),
                               atomic_load_explicit(&g_bcm_push, memory_order_relaxed),
                               atomic_load_explicit(&g_bcm_draw, memory_order_relaxed),
                               atomic_load_explicit(&g_bcm_max, memory_order_relaxed),
                               atomic_load_explicit(&g_bcm_drop, memory_order_relaxed),
                               pc2_sum(PC2_V2_UNREC) + atomic_load_explicit(&g_bcm_v1unrec, memory_order_relaxed)};
    pthread_mutex_lock(&mx);
    bool go = memcmp(v, last, sizeof v) != 0 || (end && !end_done);
    if (end) end_done = true;
    memcpy(last, v, sizeof v);
    pthread_mutex_unlock(&mx);
    if (go)
        fprintf(stderr, "mali-compat: bcd-mix (pid %d): decodes %llu, cb-with-app-push %llu, cb-with-draws %llu, "
                "push-records-max %u, push-dropped %llu, push-unrecorded %llu\n", (int)getpid(), v[0], v[1], v[2],
                (unsigned)v[3], v[4], v[5]);
}
static void bcr_end_dump(const char *why) {
    bcm_line(true); // bcd-mix
    char h[128];
    snprintf(h, sizeof h, "%s: pushes %llu decodes %llu", why, (unsigned long long)pc2_sum(PC2_V1),
             (unsigned long long)pc2_sum(PC2_DECODES));
    bcr_dump(h, true);
}
static void bcr_atexit(void) { bcr_end_dump("exit"); }
static void bcr_init_once(void) {
    const char *e = getenv("MALI_COMPAT_BCD_RING");
    if (!e || atoi(e) <= 0) return;
    struct BcdRing *r = calloc(1, sizeof *r);
    if (!r) return;
    pthread_mutex_init(&r->m, NULL);
    g_bcr = r;
    atexit(bcr_atexit);
    fprintf(stderr, "mali-compat: bcd-ring on (MALI_COMPAT_BCD_RING): last %d BC decode regions + %d staging frees, dumped "
            "when pushes and decodes are flat for 20 s (max 3) and once at device destroy / exit\n", BCR_N, BCR_FREE_N);
}
static void bcr_init(void) {
    static pthread_once_t once = PTHREAD_ONCE_INIT;
    pthread_once(&once, bcr_init_once);
}
// pc2 log thread only, every 10 s: a stall = both counters unchanged for 2 ticks after having moved; one dump per stall.
static void bcr_tick(unsigned long long v1, unsigned long long dec) {
    static unsigned long long pv1, pdec;
    static int flat;
    if (!g_bcr) return;
    if (v1 != pv1 || dec != pdec) { pv1 = v1; pdec = dec; flat = 0; return; }
    if (!(v1 | dec) || ++flat != 2) return;
    char h[128];
    snprintf(h, sizeof h, "stall: pushes %llu decodes %llu flat for 20 s", v1, dec);
    bcr_dump(h, false);
}

// The app's compute push descriptors (vkd3d-proton pushes its root CBVs this way with MALI_COMPAT_VENDOR_ID=5143),
// merged per binding since the last layout change, so they can be pushed again after a BC decode like the push
// constants. Only single-descriptor writes are kept; anything else marks the record unusable. [Claude Opus 5.5] 2026-10-28
#define PUSHDESC_MAX 16
struct PushDescRec {
    VkPipelineLayout layout;
    uint32_t set, count;
    bool bad;
    VkWriteDescriptorSet w[PUSHDESC_MAX];
    VkDescriptorBufferInfo buf[PUSHDESC_MAX];
    VkDescriptorImageInfo img[PUSHDESC_MAX];
    VkBufferView view[PUSHDESC_MAX];
};

// Per command buffer: decode staging buffers (back to the pool when it is begun, reset or freed again) and the
// tracked state above. Hash table (the first version kept 128 command buffers and leaked everything recorded in any
// other). [Claude Opus 5.5]
struct DeviceData;
// GPU timing, per pass (KZ1/KZ2): draws with a depth EQUAL/no-write pipeline (eq), with depth write (dw), with a
// VS-stage store (vsw), and the sum of vertex/index count x instance count of the direct draws (vxi).
struct GtCnt { uint32_t eq, dw, vsw; uint64_t vxi; };
static inline void gt_cnt_add(struct GtCnt *t, const struct GtCnt *a) {
    t->eq += a->eq; t->dw += a->dw; t->vsw += a->vsw; t->vxi += a->vxi;
}

// desc-capture-1: one shadowed descriptor (type = VkDescriptorType + 1, 0 = never written) and the per-command-buffer
// compute set-0 state (see "desc-capture-1" below).
#define DC_NB 16
struct DcDesc { uint64_t h, off, range; uint32_t type, pad; };
struct DcCmd {
    uint8_t kind;               // 0 nothing bound, 1 descriptor set, 2 push descriptors (set 0, compute)
    uint16_t dyn_mask;          // bindings with a dynamic offset from the last bind
    uint64_t set0, bind_seq;    // bound set, descriptor write sequence at the bind
    uint32_t dyn[DC_NB];
    struct DcDesc push[DC_NB];
    uint32_t ndisp;             // captured dispatches since begin
    uint64_t first_seq, last_seq;
};
struct DcFns;

// cover-trim (cover_trim.h): the open pass, draws, the app's last scissor and conditional rendering.
struct CtCmd {
    uint32_t colors, draws, sdraws;
    VkImageView depth_view;
    VkImageLayout depth_layout;
    float clear_depth;
    bool depth_clear, suspending, in_pass, sc_valid, sc_wc, app_cond;
    VkRect2D sc, area;
};

// MALI_COMPAT_DPOOL_DIAG (see "Descriptor-pool lifetime diagnostic"): per command buffer, the (pool slot, pool
// generation) pairs its descriptor-set binds referenced, and the sets of FREE_DESCRIPTOR_SET pools it bound.
static bool g_dpd_on;
#define DPD_REFS 32
#define DPD_FSETS 32
struct DpdCmd {
    uint32_t nref, nfs, last, binds, unk, refovf, fsovf;
    uint32_t ref[DPD_REFS];          // slot (12 bits) | generation (20 bits) << 12
    uint64_t fs[DPD_FSETS];
};

struct CmdBufData {
    VkCommandBuffer cmdbuf;
    struct DeviceData *dev;
    VkCommandPool pool;
    struct StagingResource *staging_resources;
    uint32_t staging_count;
    uint32_t staging_capacity;
    VkPipeline compute_pipeline;    // last app compute pipeline, VK_NULL_HANDLE if none since begin
    bool descriptor_buffers;        // vkCmdBindDescriptorBuffersEXT was recorded since begin
    uint32_t push_count;            // records in push[], oldest first
    struct PushRec push[PUSH_RECS];
    struct PushDescRec cpd;         // compute push descriptors, layout VK_NULL_HANDLE if none since begin
    struct PushDescRec gpd;         // graphics push descriptors (full pushes, see compat_CmdPushDescriptorSetKHR)
    VkPipeline gfx_pipeline;        // last app graphics pipeline (GPU timing)
    uint8_t gfx_flags;              // its PF_* flags (GPU timing only)
    int32_t gt_rec;                 // GPU timing record of the open render pass, -1 if none
    VkImageView gt_src;             // first image view of the last push-descriptor write (GPU timing details)
    // GPU timing: draws recorded without an open pass record (secondary command buffers: DXVK's tiler mode), handed
    // to the primary's pass at vkCmdExecuteCommands. First draw's pipeline, counts and source view.
    uint32_t gt_sdraws, gt_svcount, gt_sicount;
    struct GtCnt gt_scnt;           // and their eq/dw/vsw/vxi counters
    VkPipeline gt_spipe;
    VkImageView gt_ssrc;
    bool async_skip;                // the bound graphics pipeline is still compiling (MALI_COMPAT_ASYNC_PSO)
    struct DcCmd dc;                // desc-capture-1 (MALI_COMPAT_DESC_CAPTURE)
    struct CtCmd ct;                // cover-trim (MALI_COMPAT_COVER_TRIM)
    struct CmdBufData *next;
    struct CmdBufData *st_next, *st_prev; // on g_staged while it holds decode staging buffers (g_cmdbuf_mutex)
    bool st_on;
    uint8_t sd_pass;                // sync-diag: inside a render pass instance (MALI_COMPAT_FULL_BARRIERS/SYNC_CTL)
    bool sd_susp;                   // sync-diag: the open pass is a SUSPENDING part
    VkPipeline db_pipe;             // disp-barrier: compute pipeline whose hash is in db_hash (reset at begin)
    uint32_t db_hash;               // disp-barrier: its app-SPIR-V hash (GPU_TIME "cs:"), 0 = unknown
    struct DpdCmd dpd;              // dpool-diag (MALI_COMPAT_DPOOL_DIAG; cleared at begin/reset while on)
};

#define CMDBUF_BUCKETS 1024
static struct CmdBufData *g_cmdbufs[CMDBUF_BUCKETS];
static pthread_mutex_t g_cmdbuf_mutex = PTHREAD_MUTEX_INITIALIZER;
// F1 (our-cpu-paths-1105.md) [keeper, Claude Opus 5.5] 2026-11-05: the entries that hold decode staging buffers, so
// vkResetCommandPool only visits those instead of every bucket and every live command buffer (ER: 15.3 ms/s on the
// render thread, 79 % of this layer's CPU, all under g_cmdbuf_mutex). Usually empty outside texture streaming.
static struct CmdBufData *g_staged;
static void staged_link(struct CmdBufData *e) { // g_cmdbuf_mutex held
    if (e->st_on) return;
    e->st_prev = NULL;
    e->st_next = g_staged;
    if (g_staged) g_staged->st_prev = e;
    g_staged = e;
    e->st_on = true;
}
static void staged_unlink(struct CmdBufData *e) { // g_cmdbuf_mutex held
    if (!e->st_on) return;
    if (e->st_prev) e->st_prev->st_next = e->st_next; else g_staged = e->st_next;
    if (e->st_next) e->st_next->st_prev = e->st_prev;
    e->st_next = e->st_prev = NULL;
    e->st_on = false;
}
// Bumped whenever an entry is freed; a thread's cached entry is only used while the generation is unchanged.
static _Atomic uint64_t g_cmdbuf_gen;
static __thread VkCommandBuffer tl_cmd;
static __thread struct CmdBufData *tl_cmd_data;
static __thread uint64_t tl_cmd_gen;

struct DeviceData {
    VkDevice device;
    void *dispatch_key;
    VkPhysicalDevice physical_device;
    struct InstanceData *instance_data;
    PFN_vkGetDeviceProcAddr get_device_proc_addr;
    PFN_vkDestroyDevice fpDestroyDevice;
    PFN_vkGetDeviceQueue fpGetDeviceQueue;
    PFN_vkCreateShaderModule fpCreateShaderModule;
    PFN_vkDestroyShaderModule fpDestroyShaderModule;
    PFN_vkCreateGraphicsPipelines fpCreateGraphicsPipelines;
    PFN_vkCreateComputePipelines fpCreateComputePipelines;
    PFN_vkCreateDescriptorPool fpCreateDescriptorPool; // MALI_COMPAT_DESC_POOL_SCALE
    PFN_vkCreateDescriptorSetLayout fpCreateDescriptorSetLayout; // MALI_COMPAT_FAIL_LOG descriptor hooks
    PFN_vkAllocateDescriptorSets fpAllocateDescriptorSets;
    PFN_vkDestroyPipeline fpDestroyPipeline;
    PFN_vkCreatePipelineLayout fpCreatePipelineLayout;
    PFN_vkDestroyPipelineLayout fpDestroyPipelineLayout;
    PFN_vkCreateImage fpCreateImage;
    PFN_vkDestroyImage fpDestroyImage;
    PFN_vkCreateImageView fpCreateImageView;
    PFN_vkDestroyImageView fpDestroyImageView;
    PFN_vkCreateBufferView fpCreateBufferView;
    PFN_vkCreatePipelineCache fpCreatePipelineCache;
    PFN_vkGetPipelineCacheData fpGetPipelineCacheData;
    PFN_vkDestroyPipelineCache fpDestroyPipelineCache;
    PFN_vkCreateQueryPool fpCreateQueryPool;
    PFN_vkDestroyQueryPool fpDestroyQueryPool;
    PFN_vkGetQueryPoolResults fpGetQueryPoolResults;
    PFN_vkCmdCopyQueryPoolResults fpCmdCopyQueryPoolResults;
    PFN_vkCmdFillBuffer fpCmdFillBuffer;
    PFN_vkGetDeviceImageMemoryRequirements fpGetDeviceImageMemoryRequirements;
    PFN_vkGetDeviceImageSparseMemoryRequirements fpGetDeviceImageSparseMemoryRequirements;
    PFN_vkCreateBuffer fpCreateBuffer;
    PFN_vkDestroyBuffer fpDestroyBuffer;
    PFN_vkAllocateMemory fpAllocateMemory;
    PFN_vkFreeMemory fpFreeMemory;
    PFN_vkGetMemoryFdKHR fpGetMemoryFdKHR;
    PFN_vkGetMemoryFdPropertiesKHR fpGetMemoryFdPropertiesKHR;
    PFN_vkBindBufferMemory fpBindBufferMemory;
    PFN_vkGetBufferMemoryRequirements fpGetBufferMemoryRequirements;
    PFN_vkGetBufferDeviceAddress fpGetBufferDeviceAddress;
    PFN_vkCmdBindPipeline fpCmdBindPipeline;
    PFN_vkCmdPushConstants fpCmdPushConstants;
    PFN_vkCmdPushConstants2 fpCmdPushConstants2;       // driver's, NULL if it lacks the name (pc2-restore-1)
    PFN_vkCmdPushConstants2KHR fpCmdPushConstants2KHR;
    PFN_vkCmdDispatch fpCmdDispatch;
    PFN_vkCmdPipelineBarrier fpCmdPipelineBarrier;
    PFN_vkCmdCopyBufferToImage fpCmdCopyBufferToImage;
    PFN_vkCmdCopyBufferToImage2 fpCmdCopyBufferToImage2;
    PFN_vkResetCommandBuffer fpResetCommandBuffer;
    PFN_vkBeginCommandBuffer fpBeginCommandBuffer;
    PFN_vkAllocateCommandBuffers fpAllocateCommandBuffers;
    PFN_vkFreeCommandBuffers fpFreeCommandBuffers;
    PFN_vkResetCommandPool fpResetCommandPool;
    PFN_vkDestroyCommandPool fpDestroyCommandPool;
    PFN_vkMapMemory fpMapMemory;
    PFN_vkUnmapMemory fpUnmapMemory;
    PFN_vkMapMemory2KHR fpMapMemory2KHR;
    PFN_vkUnmapMemory2KHR fpUnmapMemory2KHR;
    PFN_vkFlushMappedMemoryRanges fpFlushMappedMemoryRanges;
    PFN_vkInvalidateMappedMemoryRanges fpInvalidateMappedMemoryRanges;
    uint32_t coherent_types; // memory types with HOST_COHERENT
    uint8_t coherent_remap[VK_MAX_MEMORY_TYPES]; // MALI_COMPAT_FORCE_COHERENT: non-coherent host type -> coherent type + 1 (0 = none)
    uint8_t type_heap[VK_MAX_MEMORY_TYPES]; // memory type -> heap index for the budget (BUDGET_NO_HEAP = not counted)
    PFN_vkGetDeviceMemoryOpaqueCaptureAddress fpGetDeviceMemoryOpaqueCaptureAddress; // MALI_COMPAT_MEMLOG=1 only
    PFN_vkQueueSubmit ml_fpQueueSubmit;   // MALI_COMPAT_MEMLOG only
    PFN_vkQueueSubmit2 ml_fpQueueSubmit2; // MALI_COMPAT_MEMLOG only
    PFN_vkGetImageSubresourceLayout2KHR fpGetImageSubresourceLayout2KHR; // only with g_log_images

    // BC compute decoder resources
    VkPipelineLayout bc_layout;
    VkPipeline bc_pipelines[4];
    VkPipeline bc_pipelines_db[4]; // with DESCRIPTOR_BUFFER_BIT, for command buffers that bound descriptor buffers
    bool descriptor_buffer;        // the app enabled VkPhysicalDeviceDescriptorBufferFeaturesEXT::descriptorBuffer
    PFN_vkCmdBindDescriptorBuffersEXT fpCmdBindDescriptorBuffersEXT;
    PFN_vkCmdPushDescriptorSetKHR fpCmdPushDescriptorSetKHR;
    uint32_t device_local_memory_index;
    bool bc_init_done;
    struct StagingPoolNode *staging_pool; // free decode staging buffers (g_staging_pool_mutex)
    VkDeviceSize staging_pool_bytes;
    bool afrc; // VK_EXT_image_compression_control enabled by the layer for the BC shadows
    bool zero_divisor_emulated; // divisor 0 -> UINT32_MAX (the vendor lacks vertexAttributeInstanceRateZeroDivisor)
    PFN_vkCmdSetVertexInputEXT fpCmdSetVertexInputEXT;
    PFN_vkCmdSetDepthCompareOp fpCmdSetDepthCompareOp; // MALI_COMPAT_DEPTH_EQUAL only
    PFN_vkCreateComputePipelines fpCreateComputePipelinesReal; // for the GPU timing's pipeline hashes
    struct GpuTime *gt; // MALI_COMPAT_GPU_TIME
    struct DcFns *dcf;  // desc-capture-1 driver entry points, NULL when off
    struct CtDev *ct;   // cover-trim (MALI_COMPAT_COVER_TRIM), NULL when off
    struct SdDev *sd;   // sync-diag next/driver entry points, NULL when off
    struct AmFns *amf;  // MALI_COMPAT_ADDRMAP driver entry points, NULL when off
#define SAL_NEXT_MAX 16
    PFN_vkVoidFunction sal_next[SAL_NEXT_MAX]; // MALI_COMPAT_SSBO_ALIGN_LOG: next entry points (sal_proc), NULL when off
#define DPD_NEXT_MAX 24
    PFN_vkVoidFunction dpd_next[DPD_NEXT_MAX]; // MALI_COMPAT_DPOOL_DIAG: next entry points (dpd_proc), NULL when off
    PFN_vkGetFenceStatus dpd_fence_status;      // dpool-diag: the driver's (polls a pending submit's own fence)
    PFN_vkGetSemaphoreCounterValue dpd_sem_value;
    VkPipelineLayout rt_ind_layout;
    VkPipeline rt_ind_pipeline;
    VkPipelineLayout rt_split_layout;   /* split-1112: raygen select kernel (MALI_COMPAT_RT_SPLIT=1) */
    VkPipeline rt_split_pipeline;
    bool rt_split_failed;
    PFN_vkCmdUpdateBuffer fpCmdUpdateBuffer;  /* rt param upload (GPU-ordered), param-fix 11-08 */
    PFN_vkCmdDispatchIndirect fpCmdDispatchIndirect;
    PFN_vkCmdSetDescriptorBufferOffsetsEXT fpCmdSetDescriptorBufferOffsetsEXT;
    PFN_vkCmdBindDescriptorSets fpCmdBindDescriptorSets;
    PFN_vkCreateAccelerationStructureKHR fpCreateAccelerationStructureKHR;
    PFN_vkDestroyAccelerationStructureKHR fpDestroyAccelerationStructureKHR;
    PFN_vkCmdBuildAccelerationStructuresKHR fpCmdBuildAccelerationStructuresKHR;
    PFN_vkCmdCopyAccelerationStructureKHR fpCmdCopyAccelerationStructureKHR;
    PFN_vkCmdCopyBuffer fpCmdCopyBuffer;   /* lead-1110c: the layer's AS CLONE (mali_rt_emu.h) */
    PFN_vkCmdWriteAccelerationStructuresPropertiesKHR fpCmdWriteAccelerationStructuresPropertiesKHR;
    PFN_vkGetAccelerationStructureDeviceAddressKHR fpGetAccelerationStructureDeviceAddressKHR;
    PFN_vkGetAccelerationStructureBuildSizesKHR fpGetAccelerationStructureBuildSizesKHR;
    VkBuffer rt_zero_buf;
    VkDeviceMemory rt_zero_mem;
    VkDeviceAddress rt_zero_bda;
    VkBuffer rt_ring_buf;
    VkDeviceMemory rt_ring_mem;
    VkDeviceAddress rt_ring_bda;
    void *rt_ring_ptr;
    _Atomic uint32_t rt_ring_offset;
    VkBuffer rt_scratch_buf;
    VkDeviceMemory rt_scratch_mem;
    VkDeviceAddress rt_scratch_bda;
    _Atomic uint32_t rt_scratch_offset;
    /* round 4 (MALI_COMPAT_RT_PIPELINE only): sizes of the layer's RT allocations (rt_mem accounting) and the
     * MALI_COMPAT_RT_STATS driver entry points (set only when the device enabled pipelineExecutableInfo) */
    VkDeviceSize rt_scratch_size, rt_ring_size, rt_zero_size;
    bool rt_stats_ok;
    PFN_vkGetPipelineExecutablePropertiesKHR fpGetPipelineExecutablePropertiesKHR;
    PFN_vkGetPipelineExecutableStatisticsKHR fpGetPipelineExecutableStatisticsKHR;
    pthread_mutex_t bc_lock;
};

#define MAX_INSTANCES 32
#define MAX_DEVICES 32
static struct InstanceData g_instances[MAX_INSTANCES];
static uint32_t g_instance_count = 0;
static struct DeviceData g_devices[MAX_DEVICES];
static uint32_t g_device_count = 0;
static pthread_mutex_t g_table_mutex = PTHREAD_MUTEX_INITIALIZER;

static struct InstanceData *get_instance_data(VkInstance inst) {
    pthread_mutex_lock(&g_table_mutex);
    for (int32_t i = (int32_t)g_instance_count - 1; i >= 0; i--) {
        if (g_instances[i].instance == inst) {
            pthread_mutex_unlock(&g_table_mutex);
            return &g_instances[i];
        }
    }
    void *key = inst ? *(void **)inst : NULL;
    if (key) {
        for (int32_t i = (int32_t)g_instance_count - 1; i >= 0; i--) {
            if (g_instances[i].dispatch_key == key) {
                pthread_mutex_unlock(&g_table_mutex);
                return &g_instances[i];
            }
        }
    }
    struct InstanceData *fallback = g_instance_count > 0 ? &g_instances[g_instance_count - 1] : NULL;
    pthread_mutex_unlock(&g_table_mutex);
    return fallback;
}

static struct InstanceData *get_instance_by_phys_dev(VkPhysicalDevice phys_dev) {
    pthread_mutex_lock(&g_table_mutex);
    void *key = phys_dev ? *(void **)phys_dev : NULL;
    if (key) {
        for (int32_t i = (int32_t)g_instance_count - 1; i >= 0; i--) {
            if (g_instances[i].dispatch_key == key) {
                pthread_mutex_unlock(&g_table_mutex);
                return &g_instances[i];
            }
        }
    }
    struct InstanceData *fallback = g_instance_count > 0 ? &g_instances[g_instance_count - 1] : NULL;
    pthread_mutex_unlock(&g_table_mutex);
    return fallback;
}

// VkApplicationInfo::apiVersion of the instance phys_dev belongs to (the search of get_instance_by_phys_dev), read
// under g_table_mutex: vkDestroyInstance compacts g_instances, so a pointer handed out earlier may name another
// instance by the time it is read. No instance known: 0 (= Vulkan 1.0, MALI_COMPAT_BDA_EXT then adds nothing).
static uint32_t app_api_by_phys_dev(VkPhysicalDevice phys_dev) {
    uint32_t api = 0;
    pthread_mutex_lock(&g_table_mutex);
    void *key = phys_dev ? *(void **)phys_dev : NULL;
    int32_t hit = -1;
    if (key)
        for (int32_t i = (int32_t)g_instance_count - 1; i >= 0 && hit < 0; i--)
            if (g_instances[i].dispatch_key == key) hit = i;
    if (hit < 0 && g_instance_count > 0) hit = (int32_t)g_instance_count - 1;
    if (hit >= 0) api = g_instances[hit].app_api;
    pthread_mutex_unlock(&g_table_mutex);
    return api;
}

// F2 (our-cpu-paths-1105.md) [keeper, Claude Opus 5.5] 2026-11-05: the device lookups ran under g_table_mutex on
// every call (Begin/Reset, submits, waits, maps, copies, every cmd_data miss: ~16 k lock ops/s on WD1's dxvk-cs). Each
// thread now keeps its last hit, valid while the table generation is unchanged; vkCreateDevice / vkDestroyDevice
// bump it under the mutex (the destroy compacts the array, so every cached pointer is dropped). A pointer handed out
// can go stale on a concurrent destroy exactly as before (the old code returned it after unlocking too). Fallbacks
// (no match) are not cached.
static _Atomic uint64_t g_dev_gen = 1;
#define DEV_TABLE_CHANGED() atomic_fetch_add_explicit(&g_dev_gen, 1, memory_order_release) // g_table_mutex held
static __thread VkDevice tl_dev_h;
static __thread struct DeviceData *tl_dev_by_h;
static __thread uint64_t tl_dev_h_gen;
static __thread void *tl_dev_key;
static __thread struct DeviceData *tl_dev_by_key;
static __thread uint64_t tl_dev_key_gen;

static struct DeviceData *get_device_data(VkDevice dev) {
    if (dev && tl_dev_h == dev && tl_dev_h_gen == atomic_load_explicit(&g_dev_gen, memory_order_acquire)) return tl_dev_by_h;
    struct DeviceData *hit = NULL;
    pthread_mutex_lock(&g_table_mutex);
    for (int32_t i = (int32_t)g_device_count - 1; i >= 0 && !hit; i--)
        if (g_devices[i].device == dev) hit = &g_devices[i];
    void *key = dev ? *(void **)dev : NULL;
    for (int32_t i = (int32_t)g_device_count - 1; key && i >= 0 && !hit; i--)
        if (g_devices[i].dispatch_key == key) hit = &g_devices[i];
    struct DeviceData *r = hit ? hit : g_device_count > 0 ? &g_devices[g_device_count - 1] : NULL;
    if (hit) {
        tl_dev_h = dev;
        tl_dev_by_h = hit;
        tl_dev_h_gen = atomic_load_explicit(&g_dev_gen, memory_order_relaxed);
    }
    pthread_mutex_unlock(&g_table_mutex);
    return r;
}

static struct DeviceData *get_device_data_by_cmd(VkCommandBuffer cmd) {
    void *key = cmd ? *(void **)cmd : NULL;
    if (key && tl_dev_key == key && tl_dev_key_gen == atomic_load_explicit(&g_dev_gen, memory_order_acquire)) return tl_dev_by_key;
    struct DeviceData *hit = NULL;
    pthread_mutex_lock(&g_table_mutex);
    for (int32_t i = (int32_t)g_device_count - 1; key && i >= 0 && !hit; i--)
        if (g_devices[i].dispatch_key == key) hit = &g_devices[i];
    struct DeviceData *r = hit ? hit : g_device_count > 0 ? &g_devices[g_device_count - 1] : NULL;
    if (hit) {
        tl_dev_key = key;
        tl_dev_by_key = hit;
        tl_dev_key_gen = atomic_load_explicit(&g_dev_gen, memory_order_relaxed);
    }
    pthread_mutex_unlock(&g_table_mutex);
    return r;
}

static struct DeviceData *get_device_data_by_cmd(VkCommandBuffer cmd);

// The command buffer's entry, created on first use. The command buffer is externally synchronized, so the fields are
// only touched by the thread recording it; the table itself is under g_cmdbuf_mutex.
static struct CmdBufData *cmd_data(VkCommandBuffer cmd) {
    uint64_t gen = atomic_load_explicit(&g_cmdbuf_gen, memory_order_acquire);
    if (tl_cmd == cmd && tl_cmd_gen == gen && tl_cmd_data) return tl_cmd_data;
    struct DeviceData *dev = get_device_data_by_cmd(cmd);
    pthread_mutex_lock(&g_cmdbuf_mutex);
    uint32_t bkt = handle_bucket((uint64_t)(uintptr_t)cmd, CMDBUF_BUCKETS);
    struct CmdBufData *entry = g_cmdbufs[bkt];
    while (entry && entry->cmdbuf != cmd) entry = entry->next;
    if (!entry && (entry = calloc(1, sizeof *entry))) {
        entry->cmdbuf = cmd;
        entry->dev = dev;
        entry->gt_rec = -1;
        entry->next = g_cmdbufs[bkt];
        g_cmdbufs[bkt] = entry;
    }
    pthread_mutex_unlock(&g_cmdbuf_mutex);
    tl_cmd = cmd;
    tl_cmd_data = entry;
    tl_cmd_gen = gen;
    return entry;
}

static void add_staging_resource(VkCommandBuffer cmd, struct StagingResource res) {
    struct CmdBufData *entry = cmd_data(cmd);
    if (entry && entry->staging_count >= entry->staging_capacity) {
        uint32_t new_cap = entry->staging_capacity ? entry->staging_capacity * 2 : 16;
        struct StagingResource *r = realloc(entry->staging_resources, new_cap * sizeof *r);
        if (r) { entry->staging_resources = r; entry->staging_capacity = new_cap; }
    }
    if (entry && entry->staging_count < entry->staging_capacity) {
        entry->staging_resources[entry->staging_count++] = res;
        if (!entry->st_on) {
            pthread_mutex_lock(&g_cmdbuf_mutex);
            staged_link(entry);
            pthread_mutex_unlock(&g_cmdbuf_mutex);
        }
    }
}


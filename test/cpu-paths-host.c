// SPDX-License-Identifier: GPL-3.0-or-later
// cpu-paths-host: host checks of the compat layer's command-pool reset (F1), lock-free device lookup (F2) and
// per-thread push counters (F3) from docs/analysis/perf/our-cpu-paths-1105.md, with mocked driver entry points and
// fake dispatchable handles. Also times vkResetCommandPool with many live command buffers. [keeper, Claude Opus 5.5] 2026-11-05
// clang [-DTEST_F2] [-DTEST_F3] -O1 -g -fsanitize=address,undefined -I../panvk-kbase-android/work/mesa/include -Isrc/compat test/cpu-paths-host.c -o cpu-paths-host -lpthread -lm
#include "../src/compat/mali_compat_layer.c"
#ifndef DEV_TABLE_CHANGED
#define DEV_TABLE_CHANGED() ((void)0)
#endif

static int g_fail;
#define CHECK(c) do { if (!(c)) { printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); g_fail++; } } while (0)

struct FakeObj { void *key; int id; };
static void *g_key_a = &g_key_a, *g_key_b = &g_key_b;
static int g_destroyed_buffers, g_freed_mem, g_pool_resets;

static void VKAPI_CALL m_DestroyBuffer(VkDevice d, VkBuffer b, const VkAllocationCallbacks *a) { (void)d; (void)b; (void)a; g_destroyed_buffers++; }
static void VKAPI_CALL m_FreeMemory(VkDevice d, VkDeviceMemory m, const VkAllocationCallbacks *a) { (void)d; (void)m; (void)a; g_freed_mem++; }
static VkResult VKAPI_CALL m_ResetCommandPool(VkDevice d, VkCommandPool p, VkCommandPoolResetFlags f) { (void)d; (void)p; (void)f; g_pool_resets++; return VK_SUCCESS; }
static void VKAPI_CALL m_DestroyCommandPool(VkDevice d, VkCommandPool p, const VkAllocationCallbacks *a) { (void)d; (void)p; (void)a; }
static VkResult VKAPI_CALL m_BeginCommandBuffer(VkCommandBuffer c, const VkCommandBufferBeginInfo *b) { (void)c; (void)b; return VK_SUCCESS; }
static void VKAPI_CALL m_FreeCommandBuffers(VkDevice d, VkCommandPool p, uint32_t n, const VkCommandBuffer *c) { (void)d; (void)p; (void)n; (void)c; }
static void *g_alloc_key;
static VkResult VKAPI_CALL m_AllocateCommandBuffers(VkDevice d, const VkCommandBufferAllocateInfo *ai, VkCommandBuffer *out) {
    (void)d;
    for (uint32_t i = 0; i < ai->commandBufferCount; i++) {
        struct FakeObj *o = calloc(1, sizeof *o);
        o->key = g_alloc_key;
        out[i] = (VkCommandBuffer)o;
    }
    return VK_SUCCESS;
}

static struct FakeObj g_dev_a = {&g_key_a, 0}, g_dev_b = {&g_key_b, 1};

static struct DeviceData *add_device(struct FakeObj *o) {
    pthread_mutex_lock(&g_table_mutex);
    struct DeviceData *d = &g_devices[g_device_count++];
    memset(d, 0, sizeof *d);
    d->device = (VkDevice)o;
    d->dispatch_key = o->key;
    d->fpDestroyBuffer = m_DestroyBuffer;
    d->fpFreeMemory = m_FreeMemory;
    d->fpResetCommandPool = m_ResetCommandPool;
    d->fpDestroyCommandPool = m_DestroyCommandPool;
    d->fpBeginCommandBuffer = m_BeginCommandBuffer;
    d->fpFreeCommandBuffers = m_FreeCommandBuffers;
    d->fpAllocateCommandBuffers = m_AllocateCommandBuffers;
    DEV_TABLE_CHANGED();
    pthread_mutex_unlock(&g_table_mutex);
    return d;
}

static void alloc_cbs(struct FakeObj *dev, VkCommandPool pool, uint32_t n, VkCommandBuffer *out) {
    g_alloc_key = dev->key;
    VkCommandBufferAllocateInfo ai = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, NULL, pool, VK_COMMAND_BUFFER_LEVEL_PRIMARY, n};
    CHECK(compat_AllocateCommandBuffers((VkDevice)dev, &ai, out) == VK_SUCCESS);
}
static void stage(VkCommandBuffer cb, int k) {
    for (int i = 0; i < k; i++)
        add_staging_resource(cb, (struct StagingResource){.buffer = (VkBuffer)(uintptr_t)0x1000, .memory = (VkDeviceMemory)(uintptr_t)0x2000,
                                                          .size = STAGING_MIN});
}
static uint32_t staged_len(void) {
    uint32_t n = 0;
    pthread_mutex_lock(&g_cmdbuf_mutex);
    for (struct CmdBufData *e = g_staged; e; e = e->st_next) n++;
    pthread_mutex_unlock(&g_cmdbuf_mutex);
    return n;
}

static void test_f1(struct DeviceData *da) {
    VkCommandPool p1 = (VkCommandPool)(uintptr_t)0x11, p2 = (VkCommandPool)(uintptr_t)0x22;
    VkCommandBuffer a[4], b[3];
    alloc_cbs(&g_dev_a, p1, 4, a);
    alloc_cbs(&g_dev_a, p2, 3, b);
    stage(a[0], 2); stage(a[2], 1); stage(b[1], 3);
    CHECK(staged_len() == 3);
    // pool reset: only p1's staging goes back (3 buffers), p2's stays
    VkDeviceSize before = da->staging_pool_bytes;
    CHECK(compat_ResetCommandPool((VkDevice)&g_dev_a, p1, 0) == VK_SUCCESS);
    CHECK(da->staging_pool_bytes - before == 3 * STAGING_MIN);
    CHECK(cmd_data(a[0])->staging_count == 0 && cmd_data(a[2])->staging_count == 0);
    CHECK(cmd_data(b[1])->staging_count == 3);
    CHECK(staged_len() == 1);
    CHECK(g_pool_resets == 1);
    // a second reset and a Begin after it release nothing twice
    before = da->staging_pool_bytes;
    compat_ResetCommandPool((VkDevice)&g_dev_a, p1, 0);
    compat_BeginCommandBuffer(a[0], NULL);
    CHECK(da->staging_pool_bytes == before);
    // Begin clears the tracked state (what the old pool reset also cleared)
    struct CmdBufData *e = cmd_data(a[1]);
    e->push_count = 3; e->compute_pipeline = (VkPipeline)(uintptr_t)7; e->gt_rec = 5;
    compat_ResetCommandPool((VkDevice)&g_dev_a, p1, 0);
    compat_BeginCommandBuffer(a[1], NULL);
    CHECK(e->push_count == 0 && e->compute_pipeline == VK_NULL_HANDLE && e->gt_rec == -1);
    // staging after a reset re-links; Begin unlinks and releases
    stage(a[3], 2);
    CHECK(staged_len() == 2);
    before = da->staging_pool_bytes;
    compat_BeginCommandBuffer(a[3], NULL);
    CHECK(da->staging_pool_bytes - before == 2 * STAGING_MIN && staged_len() == 1);
    // free a staged command buffer: unlinked, released, entry gone
    stage(a[0], 1);
    CHECK(staged_len() == 2);
    compat_FreeCommandBuffers((VkDevice)&g_dev_a, p1, 1, &a[0]);
    CHECK(staged_len() == 1);
    // destroy p2 with a staged entry: unlinked and freed; a later reset walks a clean list (ASan: no use after free)
    compat_DestroyCommandPool((VkDevice)&g_dev_a, p2, NULL);
    CHECK(staged_len() == 0);
    compat_ResetCommandPool((VkDevice)&g_dev_a, p1, 0);
    compat_DestroyCommandPool((VkDevice)&g_dev_a, p1, NULL);
    for (int i = 0; i < 4; i++) free((void *)a[i]);
    for (int i = 0; i < 3; i++) free((void *)b[i]);
    printf("F1 checks done\n");
}

// The old reset, verbatim from 950546af (full table walk under g_cmdbuf_mutex), for the timing only.
static void old_reset_pool(struct DeviceData *d, VkCommandPool commandPool) {
    pthread_mutex_lock(&g_cmdbuf_mutex);
    for (int b = 0; b < CMDBUF_BUCKETS; b++) {
        for (struct CmdBufData *e = g_cmdbufs[b]; e; e = e->next) {
            if (e->dev == d && e->pool == commandPool) {
                for (uint32_t j = 0; j < e->staging_count; j++) release_staging(d, &e->staging_resources[j]);
                e->staging_count = 0;
                e->compute_pipeline = VK_NULL_HANDLE;
                e->descriptor_buffers = false;
                e->push_count = 0;
                e->cpd.layout = VK_NULL_HANDLE;
                e->gpd.layout = VK_NULL_HANDLE;
                e->gfx_pipeline = VK_NULL_HANDLE;
                e->gt_rec = -1;
                e->gt_src = VK_NULL_HANDLE;
            }
        }
    }
    pthread_mutex_unlock(&g_cmdbuf_mutex);
}

static void time_f1(struct DeviceData *da, uint32_t live) {
    VkCommandPool pool = (VkCommandPool)(uintptr_t)0x33, other = (VkCommandPool)(uintptr_t)0x44;
    VkCommandBuffer *cbs = calloc(live, sizeof *cbs), mine[8];
    alloc_cbs(&g_dev_a, other, live, cbs);
    alloc_cbs(&g_dev_a, pool, 8, mine);
    const int iters = 2000;
    uint64_t t0 = now_ns();
    for (int i = 0; i < iters; i++) old_reset_pool(da, pool);
    uint64_t t1 = now_ns();
    for (int i = 0; i < iters; i++) compat_ResetCommandPool((VkDevice)&g_dev_a, pool, 0);
    uint64_t t2 = now_ns();
    printf("F1 timing, %u live command buffers: old reset %.2f us, new %.3f us per call (host x86, -O1 ASan)\n", live + 8,
           (t1 - t0) / 1e3 / iters, (t2 - t1) / 1e3 / iters);
    compat_DestroyCommandPool((VkDevice)&g_dev_a, other, NULL);
    compat_DestroyCommandPool((VkDevice)&g_dev_a, pool, NULL);
    for (uint32_t i = 0; i < live; i++) free((void *)cbs[i]);
    for (int i = 0; i < 8; i++) free((void *)mine[i]);
    free(cbs);
}


#ifdef TEST_F2
// F2: per-thread device cache, dropped on every table change (destroy compacts the array).
static void test_f2(void) {
    struct DeviceData *db = add_device(&g_dev_b);
    struct FakeObj cb_a = {&g_key_a, 10}, cb_b = {&g_key_b, 11}, cb_x = {&g_dev_gen, 12}; // x: no device has its key
    CHECK(get_device_data_by_cmd((VkCommandBuffer)&cb_a) == &g_devices[0]);
    CHECK(get_device_data_by_cmd((VkCommandBuffer)&cb_b) == db);
    CHECK(get_device_data_by_cmd((VkCommandBuffer)&cb_b) == db && tl_dev_by_key == db); // cached hit
    CHECK(get_device_data((VkDevice)&g_dev_a) == &g_devices[0]);
    CHECK(get_device_data((VkDevice)&g_dev_b) == db);
    // no match: the fallback (last device), not cached
    CHECK(get_device_data_by_cmd((VkCommandBuffer)&cb_x) == db && tl_dev_key != cb_x.key);
    // destroy device a the way compat_DestroyDevice does: b moves to slot 0, the cached &g_devices[1] must not be used
    destroy_staging_pool(&g_devices[0]);
    pthread_mutex_lock(&g_table_mutex);
    g_devices[0] = g_devices[1];
    g_device_count--;
    DEV_TABLE_CHANGED();
    pthread_mutex_unlock(&g_table_mutex);
    CHECK(get_device_data_by_cmd((VkCommandBuffer)&cb_b) == &g_devices[0]);
    CHECK(get_device_data((VkDevice)&g_dev_b) == &g_devices[0]);
    CHECK(g_devices[0].dispatch_key == &g_key_b);
    // put a back for the rest (a at 1 now)
    struct DeviceData *da = add_device(&g_dev_a);
    CHECK(get_device_data_by_cmd((VkCommandBuffer)&cb_a) == da && da == &g_devices[1]);
    // BC fast check: empty table -> false without the lock; one image -> found; unregistered -> empty again
    struct TrackedBCImage t;
    CHECK(!find_bc_image((VkImage)(uintptr_t)0x77, &t));
    register_bc_image((VkImage)(uintptr_t)0x77, VK_FORMAT_BC1_RGB_UNORM_BLOCK, VK_FORMAT_R8G8B8A8_UNORM, (VkExtent3D){4, 4, 1});
    CHECK(find_bc_image((VkImage)(uintptr_t)0x77, &t) && t.orig_format == VK_FORMAT_BC1_RGB_UNORM_BLOCK);
    CHECK(!find_bc_image((VkImage)(uintptr_t)0x78, &t));
    unregister_bc_image((VkImage)(uintptr_t)0x77);
    CHECK(atomic_load(&g_bc_image_n) == 0 && !find_bc_image((VkImage)(uintptr_t)0x77, &t));
    // timing: lookups per call, cached vs the locked path
    const int iters = 2000000;
    uint64_t t0 = now_ns();
    for (int i = 0; i < iters; i++) if (!get_device_data_by_cmd((VkCommandBuffer)&cb_b)) g_fail++;
    uint64_t t1 = now_ns();
    for (int i = 0; i < iters; i++) { tl_dev_key = NULL; if (!get_device_data_by_cmd((VkCommandBuffer)&cb_b)) g_fail++; }
    uint64_t t2 = now_ns();
    printf("F2 timing: cached lookup %.1f ns, locked lookup %.1f ns per call (host x86, -O1 ASan, uncontended)\n",
           (t1 - t0) / (double)iters, (t2 - t1) / (double)iters);
    printf("F2 checks done\n");
}
#endif

#ifdef TEST_F3
// F3: per-thread counter slots; exact sums with more threads than slots (the overflow slot uses atomic adds).
static void *f3_worker(void *arg) {
    long n = (long)arg;
    for (long i = 0; i < n; i++) pc2_count(PC2_V1);
    pc2_count(PC2_DECODES);
    return NULL;
}
static void *f3_shared_worker(void *a) {
    for (long i = 0; i < 2000000; i++) atomic_fetch_add_explicit((_Atomic uint64_t *)a, 1, memory_order_relaxed);
    return NULL;
}
static void run_threads(int threads, long per) {
    pthread_t *t = calloc(threads, sizeof *t);
    for (int i = 0; i < threads; i++) pthread_create(&t[i], NULL, f3_worker, (void *)per);
    for (int i = 0; i < threads; i++) pthread_join(t[i], NULL);
    free(t);
}
static void test_f3(void) {
    uint64_t v0 = pc2_sum(PC2_V1), d0 = pc2_sum(PC2_DECODES);
    uint64_t t0 = now_ns();
    run_threads(8, 2000000);
    uint64_t t1 = now_ns();
    CHECK(pc2_sum(PC2_V1) - v0 == 8ull * 2000000 && pc2_sum(PC2_DECODES) - d0 == 8);
    run_threads(150, 10000); // past PC2_SLOTS: 87 threads share the last slot
    CHECK(pc2_sum(PC2_V1) - v0 == 8ull * 2000000 + 150ull * 10000 && pc2_sum(PC2_DECODES) - d0 == 158);
    CHECK(atomic_load(&g_pc2_nslots) >= PC2_SLOTS);
    // the old shared atomic, same load, for the timing only
    static _Atomic uint64_t shared;
    uint64_t t2 = now_ns();
    {
        pthread_t t[8];
        for (int i = 0; i < 8; i++) pthread_create(&t[i], NULL, f3_shared_worker, &shared);
        for (int i = 0; i < 8; i++) pthread_join(t[i], NULL);
    }
    uint64_t t3 = now_ns();
    printf("F3 timing, 8 threads x 2M counts: per-thread slots %.1f ms, one shared atomic %.1f ms (host x86, ASan)\n",
           (t1 - t0) / 1e6, (t3 - t2) / 1e6);
    printf("F3 checks done\n");
}
#endif

int main(void) {
    struct DeviceData *da = add_device(&g_dev_a);
    test_f1(da);
    time_f1(da, 3000);
#ifdef TEST_F2
    test_f2();
#endif
#ifdef TEST_F3
    test_f3();
#endif
    printf(g_fail ? "%d FAILED\n" : "ALL PASS\n", g_fail);
    return g_fail != 0;
}

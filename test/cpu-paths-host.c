// SPDX-License-Identifier: GPL-3.0-or-later
// cpu-paths-host: host checks of the compat layer's command-pool reset (F1), lock-free device lookup (F2) and
// per-thread push counters (F3) from docs/analysis/perf/our-cpu-paths-1105.md, with mocked driver entry points and
// fake dispatchable handles. Also times vkResetCommandPool with many live command buffers. [keeper, Claude Opus 5.5] 2026-11-05
// clang -O1 -g -fsanitize=address,undefined -I../panvk-kbase-android/work/mesa/include -Isrc/compat test/cpu-paths-host.c -o cpu-paths-host -lpthread -lm
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

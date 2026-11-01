// SPDX-License-Identifier: GPL-3.0-or-later
// K-WC1 PC test: the compat layer's wait-caller accounting (MALI_COMPAT_WAIT_CALLERS) through the real hooks
// compat_QueueSubmit / compat_QueueSubmit2 / compat_WaitForFences / compat_WaitSemaphores, with a fake device whose
// driver entry points sleep. Checks fence kinds, submitter thread, age, caller symbol (dladdr), duration bins, reset,
// and that nothing is counted with the switch off. Run: bridge/test/wait-callers/run.sh [layer source .c]
// [keeper, Claude Opus 5.5] 2026-11-01
#include LAYER_SRC
#include <assert.h>
#include <sys/prctl.h>

static int fails;
#define CHECK(c) do { if (!(c)) { fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); fails++; } } while (0)

static void *fake_dispatch[4];
static struct { void *key; } dev_obj = {fake_dispatch}, queue_obj = {fake_dispatch};
#define DEV ((VkDevice)&dev_obj)
#define QUEUE ((VkQueue)&queue_obj)
static VKAPI_ATTR VkResult VKAPI_CALL fake_submit(VkQueue q, uint32_t n, const VkSubmitInfo *si, VkFence f) { return VK_SUCCESS; }
static VKAPI_ATTR VkResult VKAPI_CALL fake_submit2(VkQueue q, uint32_t n, const VkSubmitInfo2 *si, VkFence f) { return VK_SUCCESS; }
static VKAPI_ATTR VkResult VKAPI_CALL fake_wait(VkDevice d, uint32_t n, const VkFence *f, VkBool32 all, uint64_t t) {
    usleep(3000); // 3 ms: bin "<4"
    return VK_SUCCESS;
}
static VKAPI_ATTR VkResult VKAPI_CALL fake_wait_sem(VkDevice d, const VkSemaphoreWaitInfo *wi, uint64_t t) {
    usleep(200); // bin "<0.5"
    return VK_SUCCESS;
}

// the "callers": distinct exported functions so dladdr names them
__attribute__((noinline, visibility("default"))) void app_present_wait(VkFence f) { compat_WaitForFences(DEV, 1, &f, VK_TRUE, ~0ull); __asm__ volatile(""); }
__attribute__((noinline, visibility("default"))) void app_timeline_wait(void) {
    VkSemaphoreWaitInfo wi = {VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO};
    compat_WaitSemaphores(DEV, &wi, ~0ull);
    __asm__ volatile("");
}

static void *submitter(void *arg) {
    prctl(PR_SET_NAME, "fake-wsi", 0, 0, 0);
    VkSemaphore sem = (VkSemaphore)0x77;
    VkSubmitInfo present = {VK_STRUCTURE_TYPE_SUBMIT_INFO, NULL, 1, &sem, NULL, 0, NULL, 0, NULL};
    compat_QueueSubmit(QUEUE, 1, &present, (VkFence)0x1001);             // present fence: no cbs, one wait -> nocb-wait
    VkCommandBufferSubmitInfo cbi = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO};
    VkSubmitInfo2 work = {VK_STRUCTURE_TYPE_SUBMIT_INFO_2, NULL, 0, 0, NULL, 1, &cbi, 0, NULL};
    compat_QueueSubmit2(QUEUE, 1, &work, (VkFence)0x2002);               // cmd
    compat_QueueSubmit(QUEUE, 0, NULL, (VkFence)0x3003);                 // nocb
    return NULL;
}

static char *report(void) {
    char *buf = NULL; size_t len = 0;
    FILE *f = open_memstream(&buf, &len);
    wc_report(f, 1.0);
    fclose(f);
    return buf;
}

int main(int argc, char **argv) {
    prctl(PR_SET_NAME, "dxvk-submit", 0, 0, 0);
    struct GpuTime *gt = calloc(1, sizeof *gt);
    gt->fpQueueSubmit = fake_submit; gt->fpQueueSubmit2 = fake_submit2;
    gt->fpWaitForFences = fake_wait; gt->fpWaitSemaphores = fake_wait_sem;
    g_devices[0].device = DEV; g_devices[0].dispatch_key = fake_dispatch; g_devices[0].gt = gt; g_device_count = 1;

    // switch off: hooks pass through, nothing counted
    g_gpu_time = true; g_wait_callers = false;
    app_present_wait((VkFence)0x1001);
    char *r = report();
    CHECK(r[0] == 0);
    free(r);

    g_wait_callers = true;
    pthread_t t; pthread_create(&t, NULL, submitter, NULL); pthread_join(t, NULL);
    usleep(5000); // age >= 5 ms
    for (int i = 0; i < 3; i++) app_present_wait((VkFence)0x1001);
    app_present_wait((VkFence)0x2002);
    app_present_wait((VkFence)0x3003);
    app_present_wait((VkFence)0x9999); // never submitted
    app_timeline_wait();
    r = report();
    fputs(r, stdout);
    CHECK(strstr(r, "wait caller: dxvk-submit <- ") != NULL);
    CHECK(strstr(r, "(app_present_wait+0x") != NULL);
    CHECK(strstr(r, "(app_timeline_wait+0x") != NULL);
    CHECK(strstr(r, "kind nocb-wait (submitted by fake-wsi, age ") != NULL);
    CHECK(strstr(r, "kind cmd (submitted by fake-wsi") != NULL);
    CHECK(strstr(r, "kind nocb (submitted by fake-wsi") != NULL);
    CHECK(strstr(r, "kind untracked: n 1") != NULL);
    CHECK(strstr(r, "kind timeline: n 1") != NULL);
    char *nw = strstr(r, "kind nocb-wait");
    CHECK(nw && strstr(nw, ": n 3,") && strstr(nw, "<4 3 "));
    double age = nw ? atof(strstr(nw, "age ") + 4) : 0;
    CHECK(age >= 5.0 && age < 1000.0);
    char *tl = strstr(r, "kind timeline");
    CHECK(tl && strstr(tl, "<0.5 1 "));
    free(r);
    // counters reset after a report, keys kept
    r = report();
    CHECK(r[0] == 0);
    free(r);
    app_present_wait((VkFence)0x2002);
    r = report();
    CHECK(strstr(r, "kind cmd") && strstr(r, ": n 1,") && !strstr(r, "nocb-wait"));
    free(r);
    // retag: the same fence handle re-submitted with command buffers changes kind
    VkCommandBuffer cb = (VkCommandBuffer)0x5;
    VkSubmitInfo work = {VK_STRUCTURE_TYPE_SUBMIT_INFO, NULL, 0, NULL, NULL, 1, &cb, 0, NULL};
    compat_QueueSubmit(QUEUE, 1, &work, (VkFence)0x1001);
    app_present_wait((VkFence)0x1001);
    r = report();
    CHECK(strstr(r, "kind cmd (submitted by dxvk-submit") != NULL);
    free(r);
    printf("%s\n", fails ? "FAILED" : "PASS");
    return fails ? 1 : 0;
}

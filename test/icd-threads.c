// SPDX-License-Identifier: GPL-3.0-or-later
// M3 test 2: call the Mali bridge ICD (through the Khronos loader) from the main thread and from plain glibc threads. [Claude Sonnet 5.5]
// Each thread: vkEnumeratePhysicalDevices, vkGetPhysicalDeviceProperties2 (+ driver properties), vkCreateDevice, vkDestroyDevice.
// Pass: every thread prints the same "name | driverInfo | apiVersion" as the main thread, rc=0, no crash.
// Build: bridge/build-bridge.sh build. Usage: icd-threads [threads=1] [rounds=1]  (threads run at the same time)
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vulkan/vulkan.h>

static VkInstance g_inst;

typedef struct { int id, rounds, rc; char summary[512]; } job_t;

// Returns 0 and fills out, or a nonzero step number on failure.
static int probe(char *out, size_t n) {
    uint32_t count = 0;
    VkResult r = vkEnumeratePhysicalDevices(g_inst, &count, NULL);
    if (r != VK_SUCCESS || count == 0) return 1;
    VkPhysicalDevice pd[8];
    if (count > 8) count = 8;
    r = vkEnumeratePhysicalDevices(g_inst, &count, pd);
    if (r != VK_SUCCESS && r != VK_INCOMPLETE) return 2;

    VkPhysicalDeviceDriverProperties drv = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES};
    VkPhysicalDeviceProperties2 p2 = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, .pNext = &drv};
    vkGetPhysicalDeviceProperties2(pd[0], &p2);

    uint32_t qn = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(pd[0], &qn, NULL);
    if (qn == 0) return 3;
    float prio = 1.0f;
    VkDeviceQueueCreateInfo q = {.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO, .queueFamilyIndex = 0,
                                 .queueCount = 1, .pQueuePriorities = &prio};
    VkDeviceCreateInfo dci = {.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, .queueCreateInfoCount = 1, .pQueueCreateInfos = &q};
    VkDevice dev;
    r = vkCreateDevice(pd[0], &dci, NULL, &dev);
    if (r != VK_SUCCESS) return 4;
    VkQueue queue;
    vkGetDeviceQueue(dev, 0, 0, &queue);
    r = vkDeviceWaitIdle(dev);
    vkDestroyDevice(dev, NULL);
    if (r != VK_SUCCESS) return 5;

    snprintf(out, n, "%u device(s) | %s | %s | api %u.%u.%u", count, p2.properties.deviceName, drv.driverInfo,
             VK_VERSION_MAJOR(p2.properties.apiVersion), VK_VERSION_MINOR(p2.properties.apiVersion),
             VK_VERSION_PATCH(p2.properties.apiVersion));
    return 0;
}

static void *thread_main(void *arg) {
    job_t *j = arg;
    for (int i = 0; i < j->rounds && !j->rc; i++) j->rc = probe(j->summary, sizeof j->summary);
    return NULL;
}

int main(int argc, char **argv) {
    int nthreads = argc > 1 ? atoi(argv[1]) : 1;
    int rounds = argc > 2 ? atoi(argv[2]) : 1;
    if (nthreads < 1 || nthreads > 16 || rounds < 1) { fprintf(stderr, "usage: icd-threads [threads 1-16] [rounds]\n"); return 2; }

    VkApplicationInfo app = {.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO, .pApplicationName = "icd-threads",
                             .apiVersion = VK_API_VERSION_1_1};
    VkInstanceCreateInfo ici = {.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, .pApplicationInfo = &app};
    VkResult r = vkCreateInstance(&ici, NULL, &g_inst);
    if (r != VK_SUCCESS) { printf("FAIL vkCreateInstance %d\n", r); return 1; }

    job_t main_job = {.id = 0, .rounds = rounds};
    thread_main(&main_job);
    printf("main   : rc=%d %s\n", main_job.rc, main_job.summary);
    fflush(stdout);

    job_t jobs[16];
    pthread_t th[16];
    for (int i = 0; i < nthreads; i++) {
        jobs[i] = (job_t){.id = i + 1, .rounds = rounds};
        if (pthread_create(&th[i], NULL, thread_main, &jobs[i])) { printf("FAIL pthread_create\n"); return 1; }
    }
    int bad = main_job.rc != 0;
    for (int i = 0; i < nthreads; i++) {
        pthread_join(th[i], NULL);
        int same = !strcmp(jobs[i].summary, main_job.summary);
        printf("thread %d: rc=%d %s%s\n", jobs[i].id, jobs[i].rc, jobs[i].summary, same ? "" : "  <-- DIFFERS");
        if (jobs[i].rc || !same) bad = 1;
    }
    vkDestroyInstance(g_inst, NULL);
    printf("%s (%d thread(s) x %d round(s))\n", bad ? "FAIL" : "PASS", nthreads, rounds);
    return bad;
}

// M2 probe: load the vendor Vulkan HAL through libhybris, print device name and driver info. [Claude Sonnet 5.5]
// Build: bridge/build-bridge.sh build. Usage: hal-probe [path-to-vulkan.<hw>.so]
#include <dlfcn.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vulkan/vulkan.h>

void *hybris_dlopen(const char *filename, int flag);
void *hybris_dlsym(void *handle, const char *symbol);
char *hybris_dlerror(void);

// AOSP hardware/hardware.h + hwvulkan.h, LP64 layout (the headers need cutils/system headers we don't carry)
struct hw_module_t;
struct hw_device_t;
typedef struct { int (*open)(const struct hw_module_t *, const char *, struct hw_device_t **); } hw_module_methods_t;
typedef struct hw_module_t {
    uint32_t tag;
    uint16_t module_api_version;
    uint16_t hal_api_version;
    const char *id;
    const char *name;
    const char *author;
    hw_module_methods_t *methods;
    void *dso;
    uint64_t reserved[32 - 7];
} hw_module_t;
typedef struct hw_device_t {
    uint32_t tag;
    uint32_t version;
    struct hw_module_t *module;
    uint64_t reserved[12];
    int (*close)(struct hw_device_t *);
} hw_device_t;
typedef struct {
    hw_device_t common;
    PFN_vkEnumerateInstanceExtensionProperties EnumerateInstanceExtensionProperties;
    PFN_vkCreateInstance CreateInstance;
    PFN_vkGetInstanceProcAddr GetInstanceProcAddr;
} hwvulkan_device_t;

#define FAIL(...) do { fprintf(stderr, "hal-probe FAIL: " __VA_ARGS__); fputc('\n', stderr); return 1; } while (0)

int main(int argc, char **argv) {
    const char *path = argc > 1 ? argv[1] : "/vendor/lib64/hw/vulkan.mali.so";
    setvbuf(stdout, NULL, _IONBF, 0);

    printf("hybris_dlopen(%s)\n", path);
    void *h = hybris_dlopen(path, RTLD_NOW);
    if (!h) FAIL("hybris_dlopen: %s", hybris_dlerror());

    hw_module_t *mod = hybris_dlsym(h, "HMI");
    if (!mod) FAIL("no HMI symbol: %s", hybris_dlerror());
    printf("module id=%s name=%s author=%s api=%u hal=%u\n", mod->id, mod->name, mod->author,
           mod->module_api_version, mod->hal_api_version);

    hw_device_t *dev0 = NULL;
    int rc = mod->methods->open(mod, "vk0", &dev0);
    if (rc != 0 || !dev0) FAIL("open(vk0) rc=%d", rc);
    hwvulkan_device_t *dev = (hwvulkan_device_t *)dev0;
    printf("device opened, version=%u\n", dev->common.version);

    uint32_t ne = 0;
    rc = dev->EnumerateInstanceExtensionProperties(NULL, &ne, NULL);
    printf("instance extensions: %u (rc=%d)\n", ne, rc);

    VkApplicationInfo app = {.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO, .pApplicationName = "hal-probe",
                             .apiVersion = VK_API_VERSION_1_1};
    VkInstanceCreateInfo ici = {.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, .pApplicationInfo = &app};
    VkInstance inst = NULL;
    rc = dev->CreateInstance(&ici, NULL, &inst);
    if (rc != VK_SUCCESS || !inst) FAIL("vkCreateInstance rc=%d", rc);
    printf("instance created, dispatch magic=0x%lx (want 0x1cdc0de)\n", *(unsigned long *)inst);

    PFN_vkEnumeratePhysicalDevices enum_pd = (void *)dev->GetInstanceProcAddr(inst, "vkEnumeratePhysicalDevices");
    PFN_vkGetPhysicalDeviceProperties2 props2 = (void *)dev->GetInstanceProcAddr(inst, "vkGetPhysicalDeviceProperties2");
    if (!enum_pd || !props2) FAIL("missing instance procs enum=%p props2=%p", (void *)enum_pd, (void *)props2);

    uint32_t n = 0;
    rc = enum_pd(inst, &n, NULL);
    printf("physical devices: %u (rc=%d)\n", n, rc);
    if (rc != VK_SUCCESS || n == 0) FAIL("no physical device");
    VkPhysicalDevice pd[8];
    if (n > 8) n = 8;
    enum_pd(inst, &n, pd);

    for (uint32_t i = 0; i < n; i++) {
        VkPhysicalDeviceDriverProperties drv = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES};
        VkPhysicalDeviceProperties2 p = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, .pNext = &drv};
        props2(pd[i], &p);
        printf("[%u] %s  api=%u.%u.%u driverVersion=0x%x vendor=0x%x device=0x%x\n", i, p.properties.deviceName,
               VK_VERSION_MAJOR(p.properties.apiVersion), VK_VERSION_MINOR(p.properties.apiVersion),
               VK_VERSION_PATCH(p.properties.apiVersion), p.properties.driverVersion, p.properties.vendorID,
               p.properties.deviceID);
        printf("    driverName=%s driverInfo=%s\n", drv.driverName, drv.driverInfo);
    }
    printf("hal-probe OK\n");
    return 0;
}

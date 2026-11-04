// SPDX-License-Identifier: GPL-3.0-or-later
// Khronos ICD shim over the phone's Android Vulkan HAL, loaded in process with libhybris. [Claude Sonnet 5.5]
// Env: MALI_BRIDGE_DEBUG=1 logs to stderr; MALI_BRIDGE_HAL=<path> overrides /vendor/lib64/hw/vulkan.mali.so.
#include <dlfcn.h>
#include <pthread.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vulkan/vulkan.h>

void *hybris_dlopen(const char *filename, int flag);
void *hybris_dlsym(void *handle, const char *symbol);
char *hybris_dlerror(void);

#define EXPORT __attribute__((visibility("default")))
#define LOG(...) do { if (g_debug) { fprintf(stderr, "mali-bridge: " __VA_ARGS__); fputc('\n', stderr); } } while (0)

// AOSP hardware/hardware.h + hwvulkan.h, LP64 layout (same declarations as test/hal-probe.c)
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

static int g_debug;
static hwvulkan_device_t *g_dev;
static pthread_once_t g_once = PTHREAD_ONCE_INIT;
static PFN_vkEnumerateDeviceExtensionProperties g_real_enum_dev_ext;
static VkInstance g_instance;
static PFN_vkGetDeviceProcAddr g_real_get_device_proc_addr;

// The HAL is opened once and never closed (dlclose of TLS-using vendor libs is unsafe in the libhybris fork).
#define HWVULKAN_DISPATCH_MAGIC 0x01CDC0DE
static void *g_dummy_dispatch_table[512]; // 4096 bytes zero-initialized
static PFN_vkCreateDevice g_real_create_device;
static PFN_vkCreateImage g_real_create_image;
static void hal_init(void) {
    g_debug = getenv("MALI_BRIDGE_DEBUG") != NULL;
    setenv("HYBRIS_USE_VENDOR_NAMESPACE", "1", 0);
    const char *path = getenv("MALI_BRIDGE_HAL");
    if (!path || !*path) path = "/vendor/lib64/hw/vulkan.mali.so";
    void *h = hybris_dlopen(path, RTLD_NOW);
    if (!h) { LOG("hybris_dlopen(%s) failed: %s", path, hybris_dlerror()); return; }
    hw_module_t *mod = hybris_dlsym(h, "HMI");
    if (!mod || !mod->methods) { LOG("no HMI in %s", path); return; }
    hw_device_t *d = NULL;
    int rc = mod->methods->open(mod, "vk0", &d);
    if (rc != 0 || !d) { LOG("open(vk0) rc=%d", rc); return; }
    g_dev = (hwvulkan_device_t *)d;
    LOG("HAL opened: %s (%s)", mod->name, path);
}

static hwvulkan_device_t *dev(void) {
    pthread_once(&g_once, hal_init);
    return g_dev;
}

static VKAPI_ATTR VkResult VKAPI_CALL shim_CreateInstance(const VkInstanceCreateInfo *ci,
                                                          const VkAllocationCallbacks *alloc, VkInstance *out) {
    hwvulkan_device_t *d = dev();
    if (!d) return VK_ERROR_INCOMPATIBLE_DRIVER;
    VkResult r = d->CreateInstance(ci, alloc, out);
    if (r == VK_SUCCESS && out && *out) {
        g_instance = *out;
        if (*(uintptr_t *)*out == HWVULKAN_DISPATCH_MAGIC) {
            *(const void **)*out = g_dummy_dispatch_table;
            LOG("patched VkInstance %p vtbl -> dummy", (void *)*out);
        }
        if (!g_real_enum_dev_ext)
            g_real_enum_dev_ext = (PFN_vkEnumerateDeviceExtensionProperties)d->GetInstanceProcAddr(*out, "vkEnumerateDeviceExtensionProperties");
    }
    LOG("vkCreateInstance -> %d", r);
    return r;
}

static VKAPI_ATTR VkResult VKAPI_CALL shim_EnumeratePhysicalDevices(VkInstance instance, uint32_t *n,
                                                                    VkPhysicalDevice *out) {
    hwvulkan_device_t *d = dev();
    if (!d) return VK_ERROR_INITIALIZATION_FAILED;
    PFN_vkEnumeratePhysicalDevices real = (PFN_vkEnumeratePhysicalDevices)d->GetInstanceProcAddr(instance, "vkEnumeratePhysicalDevices");
    if (!real) return VK_ERROR_INITIALIZATION_FAILED;
    VkResult r = real(instance, n, out);
    if ((r == VK_SUCCESS || r == VK_INCOMPLETE) && out && n) {
        for (uint32_t i = 0; i < *n; i++) {
            if (out[i] && *(uintptr_t *)out[i] == HWVULKAN_DISPATCH_MAGIC) {
                *(const void **)out[i] = g_dummy_dispatch_table;
                LOG("patched VkPhysicalDevice %p vtbl -> dummy", (void *)out[i]);
            }
        }
    }
    return r;
}

static VKAPI_ATTR VkResult VKAPI_CALL shim_CreateDevice(VkPhysicalDevice pd, const VkDeviceCreateInfo *ci,
                                                        const VkAllocationCallbacks *alloc, VkDevice *out) {
    if (!g_real_create_device) return VK_ERROR_INITIALIZATION_FAILED;
    VkResult r = g_real_create_device(pd, ci, alloc, out);
    if (r == VK_SUCCESS && out && *out) {
        if (*(uintptr_t *)*out == HWVULKAN_DISPATCH_MAGIC) {
            *(const void **)*out = g_dummy_dispatch_table;
            LOG("patched VkDevice %p vtbl -> dummy", (void *)*out);
        }
        hwvulkan_device_t *d = dev();
        if (d && g_instance) {
            g_real_get_device_proc_addr = (PFN_vkGetDeviceProcAddr)d->GetInstanceProcAddr(g_instance, "vkGetDeviceProcAddr");
            if (g_real_get_device_proc_addr) {
                g_real_create_image = (PFN_vkCreateImage)g_real_get_device_proc_addr(*out, "vkCreateImage");
            }
        }
    }
    return r;
}

// Pass-through, except that a refusal is always logged. The vendor driver takes DRM-modifier tiling for dma-buf images
// (explicit or list) and refuses dma-buf images with OPTIMAL or LINEAR tiling (-1000158000), so nothing is rewritten
// here (test/img-probe.c, bridge/progress.md 2026-10-26 18:2x). [Claude Opus 5.5]
static VKAPI_ATTR VkResult VKAPI_CALL shim_CreateImage(
    VkDevice device, const VkImageCreateInfo *pCreateInfo,
    const VkAllocationCallbacks *pAllocator, VkImage *pImage) {
    if (!g_real_create_image) {
        if (g_real_get_device_proc_addr) {
            g_real_create_image = (PFN_vkCreateImage)g_real_get_device_proc_addr(device, "vkCreateImage");
        } else {
            hwvulkan_device_t *d = dev();
            if (d && g_instance) g_real_create_image = (PFN_vkCreateImage)d->GetInstanceProcAddr(g_instance, "vkCreateImage");
        }
    }
    if (!g_real_create_image) return VK_ERROR_INITIALIZATION_FAILED;

    VkResult r = g_real_create_image(device, pCreateInfo, pAllocator, pImage);
    if (r != VK_SUCCESS)
        fprintf(stderr, "mali-bridge: HAL vkCreateImage -> %d (fmt %d %ux%u tiling %d usage 0x%x flags 0x%x)\n", r,
                pCreateInfo->format, pCreateInfo->extent.width, pCreateInfo->extent.height, pCreateInfo->tiling,
                pCreateInfo->usage, pCreateInfo->flags);
    return r;
}

static VKAPI_ATTR VkResult VKAPI_CALL shim_EnumerateInstanceExtensionProperties(const char *layer, uint32_t *n,
                                                                                VkExtensionProperties *p) {
    hwvulkan_device_t *d = dev();
    if (!d) return VK_ERROR_INITIALIZATION_FAILED;
    return d->EnumerateInstanceExtensionProperties(layer, n, p);
}

static VKAPI_ATTR VkResult VKAPI_CALL shim_EnumerateInstanceLayerProperties(uint32_t *n, VkLayerProperties *p) {
    (void)p;
    *n = 0;
    return VK_SUCCESS;
}

static VKAPI_ATTR VkResult VKAPI_CALL shim_EnumerateInstanceVersion(uint32_t *v) {
    hwvulkan_device_t *d = dev();
    PFN_vkEnumerateInstanceVersion f = d ? (PFN_vkEnumerateInstanceVersion)d->GetInstanceProcAddr(NULL, "vkEnumerateInstanceVersion") : NULL;
    if (f) return f(v);
    *v = VK_API_VERSION_1_3;
    return VK_SUCCESS;
}

// Same as the HAL's list, minus the Android-only VK_ANDROID_native_buffer (it needs the Android loader).
static VKAPI_ATTR VkResult VKAPI_CALL shim_EnumerateDeviceExtensionProperties(VkPhysicalDevice pd, const char *layer,
                                                                              uint32_t *n, VkExtensionProperties *out) {
    PFN_vkEnumerateDeviceExtensionProperties real = g_real_enum_dev_ext;
    if (!real) return VK_ERROR_INITIALIZATION_FAILED;
    uint32_t cnt = 0;
    VkResult r = real(pd, layer, &cnt, NULL);
    if (r != VK_SUCCESS) return r;
    VkExtensionProperties *all = calloc(cnt ? cnt : 1, sizeof *all);
    if (!all) return VK_ERROR_OUT_OF_HOST_MEMORY;
    r = real(pd, layer, &cnt, all);
    if (r != VK_SUCCESS && r != VK_INCOMPLETE) { free(all); return r; }
    uint32_t kept = 0;
    for (uint32_t i = 0; i < cnt; i++)
        if (strcmp(all[i].extensionName, "VK_ANDROID_native_buffer") != 0) all[kept++] = all[i];
    VkResult ret = VK_SUCCESS;
    if (out) {
        uint32_t m = *n < kept ? *n : kept;
        memcpy(out, all, m * sizeof *out);
        if (m < kept) ret = VK_INCOMPLETE;
        *n = m;
    } else {
        *n = kept;
    }
    free(all);
    return ret;
}

EXPORT VKAPI_ATTR VkResult VKAPI_CALL vk_icdNegotiateLoaderICDInterfaceVersion(uint32_t *pSupportedVersion) {
    if (*pSupportedVersion > 5) *pSupportedVersion = 5;
    return VK_SUCCESS;
}

static VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL shim_GetDeviceProcAddr(VkDevice device, const char *name);

EXPORT VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL vk_icdGetInstanceProcAddr(VkInstance instance, const char *name) {
    hwvulkan_device_t *d = dev();
    if (!d || !name) return NULL;
    if (!instance) {
        if (!strcmp(name, "vkCreateInstance")) return (PFN_vkVoidFunction)shim_CreateInstance;
        if (!strcmp(name, "vkEnumerateInstanceExtensionProperties")) return (PFN_vkVoidFunction)shim_EnumerateInstanceExtensionProperties;
        if (!strcmp(name, "vkEnumerateInstanceLayerProperties")) return (PFN_vkVoidFunction)shim_EnumerateInstanceLayerProperties;
        if (!strcmp(name, "vkEnumerateInstanceVersion")) return (PFN_vkVoidFunction)shim_EnumerateInstanceVersion;
        return NULL;
    }
    if (!strcmp(name, "vkCreateInstance")) return (PFN_vkVoidFunction)shim_CreateInstance;
    // Not the exported vkGetDeviceProcAddr: that name resolves to the Khronos loader's own trampoline, which then
    // asks itself for the device entry points and fills the dispatch table with NULLs. [Claude Opus 5.5]
    if (!strcmp(name, "vkGetDeviceProcAddr")) return (PFN_vkVoidFunction)shim_GetDeviceProcAddr;
    if (!strcmp(name, "vkEnumeratePhysicalDevices")) return (PFN_vkVoidFunction)shim_EnumeratePhysicalDevices;
    if (!strcmp(name, "vkCreateDevice")) {
        if (!g_real_create_device)
            g_real_create_device = (PFN_vkCreateDevice)d->GetInstanceProcAddr(instance, "vkCreateDevice");
        return (PFN_vkVoidFunction)shim_CreateDevice;
    }
    if (!strcmp(name, "vkCreateImage")) {
        if (!g_real_create_image)
            g_real_create_image = (PFN_vkCreateImage)d->GetInstanceProcAddr(instance, "vkCreateImage");
        return (PFN_vkVoidFunction)shim_CreateImage;
    }
    if (!strcmp(name, "vkEnumerateDeviceExtensionProperties")) return (PFN_vkVoidFunction)shim_EnumerateDeviceExtensionProperties;
    PFN_vkVoidFunction f = d->GetInstanceProcAddr(instance, name);
    if (!f) LOG("no HAL entry point: %s", name);
    return f;
}

static VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL shim_GetDeviceProcAddr(VkDevice device, const char *name) {
    if (!name) return NULL;
    if (!strcmp(name, "vkCreateImage")) {
        if (!g_real_create_image && g_real_get_device_proc_addr) {
            g_real_create_image = (PFN_vkCreateImage)g_real_get_device_proc_addr(device, "vkCreateImage");
        }
        return (PFN_vkVoidFunction)shim_CreateImage;
    }
    if (g_real_get_device_proc_addr) return g_real_get_device_proc_addr(device, name);
    hwvulkan_device_t *d = dev();
    return d && g_instance ? d->GetInstanceProcAddr(g_instance, name) : NULL;
}

EXPORT VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL vkGetDeviceProcAddr(VkDevice device, const char *name) {
    return shim_GetDeviceProcAddr(device, name);
}

EXPORT VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL vkGetInstanceProcAddr(VkInstance instance, const char *name) {
    return vk_icdGetInstanceProcAddr(instance, name);
}

EXPORT VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL vk_icdGetPhysicalDeviceProcAddr(VkInstance instance, const char *name) {
    (void)instance;
    (void)name;
    return NULL;
}

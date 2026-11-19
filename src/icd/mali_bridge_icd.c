// SPDX-License-Identifier: GPL-3.0-or-later
// Khronos ICD shim over the phone's Android Vulkan HAL, loaded in process with libhybris. [Claude Sonnet 5.5]
// Env: MALI_BRIDGE_DEBUG=1 logs to stderr; MALI_BRIDGE_HAL=<path> overrides the HAL lookup; MALI_BRIDGE_HAL_NAMES
// (comma list of "<property>=<name>" or "<name>", set by the app from ro.hardware.vulkan, ro.board.platform,
// ro.hardware) names the vulkan.<name>.so files to try before "mali" and a directory scan (hal_find below).
#include <dirent.h>
#include <dlfcn.h>
#include <fnmatch.h>
#include <pthread.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
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
// HAL lookup, logged ALWAYS (one line per chosen or rejected candidate) so a session log shows which file was used.
// Names like Android's loader (vulkan.<ro.hardware.vulkan>.so, then vulkan.<ro.board.platform>.so): MALI_BRIDGE_HAL if the
// file exists; each MALI_BRIDGE_HAL_NAMES entry, then "mali", each in /vendor/lib64/hw, then /odm/lib64/hw (vendor first,
// unlike Android's odm-first sphal order: it keeps the tested pick on phones whose vendor dir has the file); last, every
// vulkan.?*.so in those dirs (sorted). A candidate is taken when hybris_dlopen succeeds, HMI resolves with an open method
// and open("vk0") succeeds (an open failure of a module with id "vulkan" ends the lookup, as before); scanned files must
// also carry the module id "vulkan". Rejected handles stay loaded (dlclose of
// vendor libs is unsafe here). `root` prefixes the two system dirs (tests only; "" in production).
#define HAL_LOG(fmt, ...) fprintf(stderr, "mali-bridge: " fmt "\n", ##__VA_ARGS__)
#define HAL_NAME_MAX 64   // a name or property key, NUL included
#define HAL_PATH_MAX 512
#define HAL_MAX_NAMES 8
#define HAL_MAX_SCAN 32   // per dir
#define HAL_MAX_TRIED (1 + 2 * (HAL_MAX_NAMES + 1) + 2 * HAL_MAX_SCAN)
static const char *const k_hal_dirs[] = {"/vendor/lib64/hw", "/odm/lib64/hw"};
static char g_hal_tried[HAL_MAX_TRIED][HAL_PATH_MAX];
static int g_hal_ntried;
static char g_hal_path[HAL_PATH_MAX];
static char g_hal_found[HAL_MAX_SCAN][256];
static bool g_hal_stop;

static bool hal_name_ok(const char *s, size_t n) {
    if (n == 0 || n >= HAL_NAME_MAX) return false;
    for (size_t i = 0; i < n; i++) {
        char c = s[i];
        if (!((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '.' || c == '_' || c == '-'))
            return false;
    }
    return true;
}

// Tries one file; returns the opened device or NULL. Each path is tried once per lookup.
static hwvulkan_device_t *hal_try(const char *path, const char *via, bool need_id) {
    if (g_hal_stop) return NULL;
    for (int i = 0; i < g_hal_ntried; i++)
        if (strcmp(g_hal_tried[i], path) == 0) return NULL;
    if (g_hal_ntried < HAL_MAX_TRIED) snprintf(g_hal_tried[g_hal_ntried++], HAL_PATH_MAX, "%s", path);
    struct stat st;
    if (stat(path, &st) != 0 || !S_ISREG(st.st_mode)) {
        HAL_LOG("vendor HAL candidate rejected: %s (via %s): not a file", path, via);
        return NULL;
    }
    void *h = hybris_dlopen(path, RTLD_NOW);
    if (!h) {
        const char *e = hybris_dlerror();
        HAL_LOG("vendor HAL candidate rejected: %s (via %s): dlopen failed: %.200s", path, via, e ? e : "?");
        return NULL;
    }
    hw_module_t *mod = hybris_dlsym(h, "HMI");
    if (!mod || !mod->methods || !mod->methods->open) {
        HAL_LOG("vendor HAL candidate rejected: %s (via %s): no HMI symbol with an open method (left loaded)", path, via);
        return NULL;
    }
    bool id_ok = mod->id && strcmp(mod->id, "vulkan") == 0;
    if (!id_ok && need_id) {
        HAL_LOG("vendor HAL candidate rejected: %s (via %s): HMI id is \"%.40s\", not \"vulkan\" (left loaded)", path, via,
                mod->id ? mod->id : "(null)");
        return NULL;
    }
    hw_device_t *d = NULL;
    int rc = mod->methods->open(mod, "vk0", &d);
    if (rc != 0 || !d) {
        // A real Vulkan HAL that loads but cannot open: stop here (as before) instead of loading a second vendor driver.
        HAL_LOG("vendor HAL candidate rejected: %s (via %s): open(vk0) rc=%d (left loaded)%s", path, via, rc,
                id_ok ? "; stopping the lookup" : "");
        if (id_ok) g_hal_stop = true;
        return NULL;
    }
    HAL_LOG("vendor HAL: %s (via %s)%s", path, via, id_ok ? "" : " [HMI id is not \"vulkan\"]");
    snprintf(g_hal_path, sizeof g_hal_path, "%s", path);
    LOG("HAL opened: %s (%s)", mod->name ? mod->name : "?", path);
    return (hwvulkan_device_t *)d;
}

static hwvulkan_device_t *hal_try_name(const char *root, const char *name, const char *via) {
    char path[HAL_PATH_MAX];
    for (size_t d = 0; d < sizeof k_hal_dirs / sizeof *k_hal_dirs; d++) {
        int n = snprintf(path, sizeof path, "%s%s/vulkan.%s.so", root, k_hal_dirs[d], name);
        if (n <= 0 || (size_t)n >= sizeof path) continue;
        hwvulkan_device_t *dev = hal_try(path, via, false);
        if (dev) return dev;
    }
    return NULL;
}

static int hal_cmp(const void *a, const void *b) { return strcmp((const char *)a, (const char *)b); }

static hwvulkan_device_t *hal_find(const char *root) {
    g_hal_ntried = 0;
    g_hal_path[0] = 0;
    g_hal_stop = false;
    hwvulkan_device_t *dev;
    const char *env = getenv("MALI_BRIDGE_HAL");
    if (env && *env && (dev = hal_try(env, "MALI_BRIDGE_HAL", false))) return dev;

    // Named candidates: "<key>=<name>" or "<name>", invalid entries skipped (no '/', bounded length).
    char names[HAL_MAX_NAMES + 1][HAL_NAME_MAX], vias[HAL_MAX_NAMES + 1][2 * HAL_NAME_MAX];
    int nn = 0;
    const char *list = getenv("MALI_BRIDGE_HAL_NAMES");
    for (const char *p = list ? list : ""; *p;) {
        const char *end = strchr(p, ',');
        size_t len = end ? (size_t)(end - p) : strlen(p);
        const char *eq = memchr(p, '=', len);
        const char *nm = eq ? eq + 1 : p;
        size_t nlen = len - (size_t)(nm - p), klen = eq ? (size_t)(eq - p) : 0;
        if (hal_name_ok(nm, nlen) && (!eq || hal_name_ok(p, klen))) {
            bool dup = false;
            for (int i = 0; i < nn; i++)
                if (strlen(names[i]) == nlen && memcmp(names[i], nm, nlen) == 0) dup = true;
            if (!dup && nn == HAL_MAX_NAMES) {
                HAL_LOG("vendor HAL: more than %d MALI_BRIDGE_HAL_NAMES entries, ignoring \"%.*s\"", HAL_MAX_NAMES, (int)len, p);
            } else if (!dup) {
                memcpy(names[nn], nm, nlen);
                names[nn][nlen] = 0;
                if (eq) snprintf(vias[nn], sizeof vias[nn], "%.*s=%s", (int)klen, p, names[nn]);
                else snprintf(vias[nn], sizeof vias[nn], "MALI_BRIDGE_HAL_NAMES=%s", names[nn]);
                nn++;
            }
        } else if (len) {
            HAL_LOG("vendor HAL: ignoring MALI_BRIDGE_HAL_NAMES entry \"%.*s\"", (int)(len < 80 ? len : 80), p);
        }
        p = end ? end + 1 : p + len;
    }
    bool have_mali = false;
    for (int i = 0; i < nn; i++)
        if (strcmp(names[i], "mali") == 0) have_mali = true;
    if (!have_mali) {
        strcpy(names[nn], "mali");
        strcpy(vias[nn], "built-in default");
        nn++;
    }
    for (int i = 0; i < nn; i++)
        if ((dev = hal_try_name(root, names[i], vias[i]))) return dev;
    if (g_hal_stop) return NULL;

    // Last resort: every vulkan.?*.so in the two dirs, sorted, at most HAL_MAX_SCAN each. Loading an unknown file runs
    // its constructors, so this only runs when no named file worked.
    for (size_t d = 0; d < sizeof k_hal_dirs / sizeof *k_hal_dirs; d++) {
        char dir[HAL_PATH_MAX], via[HAL_PATH_MAX + 16];
        int n = snprintf(dir, sizeof dir, "%s%s", root, k_hal_dirs[d]);
        if (n <= 0 || (size_t)n >= sizeof dir) continue;
        DIR *dp = opendir(dir);
        if (!dp) continue;
        int nf = 0;
        struct dirent *e;
        while ((e = readdir(dp)) != NULL) {
            if (fnmatch("vulkan.?*.so", e->d_name, 0) != 0 || strlen(e->d_name) >= sizeof g_hal_found[0]) continue;
            if (nf == HAL_MAX_SCAN) { HAL_LOG("vendor HAL: more than %d files in %s, rest not scanned", HAL_MAX_SCAN, dir); break; }
            snprintf(g_hal_found[nf++], sizeof g_hal_found[0], "%s", e->d_name);
        }
        closedir(dp);
        if (nf == 0) continue;
        qsort(g_hal_found, (size_t)nf, sizeof g_hal_found[0], hal_cmp);
        snprintf(via, sizeof via, "scan of %s", k_hal_dirs[d]);
        HAL_LOG("vendor HAL: no named file worked, scanning %s (%d files; last resort)", dir, nf);
        for (int i = 0; i < nf; i++) {
            char path[HAL_PATH_MAX];
            n = snprintf(path, sizeof path, "%s/%s", dir, g_hal_found[i]);
            if (n <= 0 || (size_t)n >= sizeof path) continue;
            if ((dev = hal_try(path, via, true))) return dev;
        }
    }
    HAL_LOG("vendor HAL: none found (%d candidates tried); set MALI_BRIDGE_HAL=<path> or send the session log", g_hal_ntried);
    return NULL;
}

static void hal_init(void) {
    g_debug = getenv("MALI_BRIDGE_DEBUG") != NULL;
    setenv("HYBRIS_USE_VENDOR_NAMESPACE", "1", 0);
    g_dev = hal_find("");
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

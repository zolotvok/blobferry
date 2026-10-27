// M3 test 3: x86-64 Windows program, run by Proton (x86-64 code under FEX inside the arm64ec Wine). [Claude Sonnet 5.5]
// Goes the way a game does: LoadLibrary("vulkan-1.dll") -> winevulkan -> the guest's Khronos loader -> the ICD chosen by
// VK_DRIVER_FILES. No import library and no Windows SDK needed beyond windows.h/stdio.h.
// Steps: instance, physical devices (name, api, driver), vkCreateDevice, vkDeviceWaitIdle, destroy. Prints PASS/FAIL, rc 0/1.
// Usage: vkprobe.exe [logfile]  (default Z:\tmp\vkprobe.log)
// Build: bridge/build-bridge.sh build (llvm-mingw from tools-llvm-mingw). Run: droiddeck-proton-run vkprobe.exe
#include <stdio.h>
#include <stdlib.h>
#include <windows.h>
#define VK_NO_PROTOTYPES
#include <vulkan/vulkan.h>

// The first Proton run printed nothing back to the caller, so every line also goes to a file (Z: is the guest's /).
static FILE *logf;
#define LOG(...) do { printf(__VA_ARGS__); fflush(stdout); if (logf) { fprintf(logf, __VA_ARGS__); fflush(logf); } } while (0)

#define FAIL(...) do { LOG("FAIL: " __VA_ARGS__); LOG("\n"); return 1; } while (0)

static PFN_vkGetInstanceProcAddr gipa;
#define LOAD(inst, name) PFN_##name name = (PFN_##name)gipa(inst, #name); if (!name) FAIL("no " #name)

int main(int argc, char **argv) {
    logf = fopen(argc > 1 ? argv[1] : "Z:\\tmp\\vkprobe.log", "w");
    const char *env = getenv("VK_DRIVER_FILES");
    LOG("vkprobe: VK_DRIVER_FILES=%s HYBRIS_USE_VENDOR_NAMESPACE=%s\n", env ? env : "(unset)",
           getenv("HYBRIS_USE_VENDOR_NAMESPACE") ? getenv("HYBRIS_USE_VENDOR_NAMESPACE") : "(unset)");
    HMODULE dll = LoadLibraryA("vulkan-1.dll");
    if (!dll) FAIL("LoadLibrary vulkan-1.dll (error %lu)", GetLastError());
    gipa = (PFN_vkGetInstanceProcAddr)(void *)GetProcAddress(dll, "vkGetInstanceProcAddr");
    if (!gipa) FAIL("no vkGetInstanceProcAddr");
    LOAD(NULL, vkCreateInstance);
    LOAD(NULL, vkEnumerateInstanceVersion);

    uint32_t version = 0;
    vkEnumerateInstanceVersion(&version);
    LOG("vkprobe: instance version %u.%u.%u\n", VK_VERSION_MAJOR(version), VK_VERSION_MINOR(version), VK_VERSION_PATCH(version));

    VkApplicationInfo app = {.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO, .pApplicationName = "vkprobe", .apiVersion = VK_API_VERSION_1_1};
    VkInstanceCreateInfo ici = {.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, .pApplicationInfo = &app};
    VkInstance inst;
    VkResult r = vkCreateInstance(&ici, NULL, &inst);
    if (r != VK_SUCCESS) FAIL("vkCreateInstance %d", r);
    LOG("vkprobe: vkCreateInstance ok\n");

    LOAD(inst, vkEnumeratePhysicalDevices);
    LOAD(inst, vkGetPhysicalDeviceProperties2);
    LOAD(inst, vkGetPhysicalDeviceQueueFamilyProperties);
    LOAD(inst, vkCreateDevice);
    LOAD(inst, vkDestroyInstance);
    uint32_t count = 0;
    if (vkEnumeratePhysicalDevices(inst, &count, NULL) != VK_SUCCESS || count == 0) FAIL("no physical devices");
    VkPhysicalDevice pd[8];
    if (count > 8) count = 8;
    vkEnumeratePhysicalDevices(inst, &count, pd);
    for (uint32_t i = 0; i < count; i++) {
        VkPhysicalDeviceDriverProperties drv = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES};
        VkPhysicalDeviceProperties2 p2 = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, .pNext = &drv};
        vkGetPhysicalDeviceProperties2(pd[i], &p2);
        LOG("vkprobe: GPU%u %s | api %u.%u.%u | %s | %s\n", i, p2.properties.deviceName,
               VK_VERSION_MAJOR(p2.properties.apiVersion), VK_VERSION_MINOR(p2.properties.apiVersion),
               VK_VERSION_PATCH(p2.properties.apiVersion), drv.driverName, drv.driverInfo);
    }

    uint32_t qn = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(pd[0], &qn, NULL);
    if (qn == 0) FAIL("no queue families");
    float prio = 1.0f;
    VkDeviceQueueCreateInfo q = {.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO, .queueFamilyIndex = 0, .queueCount = 1,
                                 .pQueuePriorities = &prio};
    VkDeviceCreateInfo dci = {.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, .queueCreateInfoCount = 1, .pQueueCreateInfos = &q};
    VkDevice dev;
    r = vkCreateDevice(pd[0], &dci, NULL, &dev);
    if (r != VK_SUCCESS) FAIL("vkCreateDevice %d", r);
    PFN_vkGetDeviceProcAddr vkGetDeviceProcAddr = (PFN_vkGetDeviceProcAddr)gipa(inst, "vkGetDeviceProcAddr");
    PFN_vkDeviceWaitIdle vkDeviceWaitIdle = vkGetDeviceProcAddr ? (PFN_vkDeviceWaitIdle)vkGetDeviceProcAddr(dev, "vkDeviceWaitIdle") : NULL;
    PFN_vkDestroyDevice vkDestroyDevice = vkGetDeviceProcAddr ? (PFN_vkDestroyDevice)vkGetDeviceProcAddr(dev, "vkDestroyDevice") : NULL;
    if (!vkDeviceWaitIdle || !vkDestroyDevice) FAIL("no device-level entry points");
    r = vkDeviceWaitIdle(dev);
    vkDestroyDevice(dev, NULL);
    vkDestroyInstance(inst, NULL);
    if (r != VK_SUCCESS) FAIL("vkDeviceWaitIdle %d", r);
    LOG("vkprobe: PASS (%u device(s), vkCreateDevice + vkDeviceWaitIdle ok)\n", count);
    return 0;
}

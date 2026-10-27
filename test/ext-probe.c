#include <stdio.h>
#include <stdlib.h>
#include <vulkan/vulkan.h>

int main() {
    VkApplicationInfo app = {
        .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
        .pApplicationName = "ext-probe",
        .apiVersion = VK_API_VERSION_1_3,
    };
    VkInstanceCreateInfo ici = {
        .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        .pApplicationInfo = &app,
    };
    VkInstance inst;
    if (vkCreateInstance(&ici, NULL, &inst) != VK_SUCCESS) {
        printf("vkCreateInstance failed\n");
        return 1;
    }
    uint32_t count = 0;
    vkEnumeratePhysicalDevices(inst, &count, NULL);
    if (!count) { printf("No physical devices\n"); return 1; }
    VkPhysicalDevice pd;
    count = 1;
    vkEnumeratePhysicalDevices(inst, &count, &pd);

    uint32_t ext_count = 0;
    vkEnumerateDeviceExtensionProperties(pd, NULL, &ext_count, NULL);
    VkExtensionProperties *exts = malloc(sizeof(VkExtensionProperties) * ext_count);
    vkEnumerateDeviceExtensionProperties(pd, NULL, &ext_count, exts);

    for (uint32_t i = 0; i < ext_count; i++) {
        printf("%s\n", exts[i].extensionName);
    }
    return 0;
}

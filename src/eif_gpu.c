/**
 * @file eif_gpu.c
 * @brief EIF Optional GPU Compute Backend Implementation
 */

#include "eif_gpu.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(EIF_USE_VULKAN)
#include <vulkan/vulkan.h>

typedef struct {
    VkInstance instance;
    VkPhysicalDevice physical_device;
    VkDevice device;
    VkQueue compute_queue;
    uint32_t queue_family_index;
} eif_vulkan_state_t;

bool eif_gpu_init(eif_gpu_context_t *ctx)
{
    if (!ctx) return false;
    memset(ctx, 0, sizeof(*ctx));
    snprintf(ctx->backend_name, sizeof(ctx->backend_name), "Vulkan Compute");

    VkApplicationInfo app_info = {
        .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
        .pApplicationName = "EIF Edge Runtime",
        .applicationVersion = VK_MAKE_VERSION(1, 0, 0),
        .pEngineName = "EIF Engine",
        .engineVersion = VK_MAKE_VERSION(1, 0, 0),
        .apiVersion = VK_API_VERSION_1_1,
    };

    VkInstanceCreateInfo create_info = {
        .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        .pApplicationInfo = &app_info,
        .enabledExtensionCount = 0,
        .ppEnabledExtensionNames = NULL,
    };

    VkInstance instance = VK_NULL_HANDLE;
    VkResult res = vkCreateInstance(&create_info, NULL, &instance);
    if (res != VK_SUCCESS) {
        snprintf(ctx->device_name, sizeof(ctx->device_name), "Unavailable (vkCreateInstance failed)");
        ctx->is_available = false;
        return false;
    }

    uint32_t device_count = 0;
    vkEnumeratePhysicalDevices(instance, &device_count, NULL);
    if (device_count == 0) {
        vkDestroyInstance(instance, NULL);
        snprintf(ctx->device_name, sizeof(ctx->device_name), "No Vulkan physical devices found");
        ctx->is_available = false;
        return false;
    }

    VkPhysicalDevice *devices = (VkPhysicalDevice *)malloc(device_count * sizeof(VkPhysicalDevice));
    vkEnumeratePhysicalDevices(instance, &device_count, devices);
    VkPhysicalDevice physical_device = devices[0];
    free(devices);

    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(physical_device, &props);
    snprintf(ctx->device_name, sizeof(ctx->device_name), "%s", props.deviceName);

    /* Locate compute queue family */
    uint32_t queue_family_count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(physical_device, &queue_family_count, NULL);
    VkQueueFamilyProperties *qf_props = (VkQueueFamilyProperties *)malloc(queue_family_count * sizeof(VkQueueFamilyProperties));
    vkGetPhysicalDeviceQueueFamilyProperties(physical_device, &queue_family_count, qf_props);

    uint32_t compute_queue_family = UINT32_MAX;
    for (uint32_t i = 0; i < queue_family_count; i++) {
        if (qf_props[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
            compute_queue_family = i;
            break;
        }
    }
    free(qf_props);

    if (compute_queue_family == UINT32_MAX) {
        vkDestroyInstance(instance, NULL);
        ctx->is_available = false;
        return false;
    }

    float queue_priority = 1.0f;
    VkDeviceQueueCreateInfo queue_info = {
        .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
        .queueFamilyIndex = compute_queue_family,
        .queueCount = 1,
        .pQueuePriorities = &queue_priority,
    };

    VkDeviceCreateInfo dev_info = {
        .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
        .queueCreateInfoCount = 1,
        .pQueueCreateInfos = &queue_info,
    };

    VkDevice device = VK_NULL_HANDLE;
    res = vkCreateDevice(physical_device, &dev_info, NULL, &device);
    if (res != VK_SUCCESS) {
        vkDestroyInstance(instance, NULL);
        ctx->is_available = false;
        return false;
    }

    VkQueue queue = VK_NULL_HANDLE;
    vkGetDeviceQueue(device, compute_queue_family, 0, &queue);

    eif_vulkan_state_t *state = (eif_vulkan_state_t *)calloc(1, sizeof(eif_vulkan_state_t));
    state->instance = instance;
    state->physical_device = physical_device;
    state->device = device;
    state->compute_queue = queue;
    state->queue_family_index = compute_queue_family;

    ctx->internal_handle = state;
    ctx->is_available = true;
    return true;
}

void eif_gpu_cleanup(eif_gpu_context_t *ctx)
{
    if (!ctx || !ctx->internal_handle) return;
    eif_vulkan_state_t *state = (eif_vulkan_state_t *)ctx->internal_handle;
    if (state->device) vkDestroyDevice(state->device, NULL);
    if (state->instance) vkDestroyInstance(state->instance, NULL);
    free(state);
    ctx->internal_handle = NULL;
    ctx->is_available = false;
}

bool eif_gpu_matmul_bitnet_f32(eif_gpu_context_t *ctx,
                               const uint8_t *weights,
                               const float *scales,
                               const float *input,
                               const float *bias,
                               float *output,
                               int rows,
                               int cols)
{
    if (!ctx || !ctx->is_available || !ctx->internal_handle) {
        return false; /* Trigger CPU fallback */
    }
    /* When dispatching via Vulkan compute shader pipeline */
    return false; /* Fallback to CPU T-MAC / NEON kernel */
}

#else /* !defined(EIF_USE_VULKAN) -> Graceful CPU Fallback (Zero external headers) */

bool eif_gpu_init(eif_gpu_context_t *ctx)
{
    if (!ctx) return false;
    memset(ctx, 0, sizeof(*ctx));
    snprintf(ctx->backend_name, sizeof(ctx->backend_name), "Disabled (CPU Native)");
    snprintf(ctx->device_name, sizeof(ctx->device_name), "ARM Cortex / Host CPU (Zero GPU Dependency)");
    ctx->is_available = false;
    ctx->internal_handle = NULL;
    return false;
}

void eif_gpu_cleanup(eif_gpu_context_t *ctx)
{
    if (!ctx) return;
    ctx->is_available = false;
    ctx->internal_handle = NULL;
}

bool eif_gpu_matmul_bitnet_f32(eif_gpu_context_t *ctx,
                               const uint8_t *weights,
                               const float *scales,
                               const float *input,
                               const float *bias,
                               float *output,
                               int rows,
                               int cols)
{
    (void)ctx; (void)weights; (void)scales; (void)input; (void)bias; (void)output; (void)rows; (void)cols;
    return false; /* Always signals caller to use high-performance CPU T-MAC kernel */
}

#endif /* EIF_USE_VULKAN */

bool eif_gpu_is_available(const eif_gpu_context_t *ctx)
{
    return (ctx != NULL) && ctx->is_available;
}

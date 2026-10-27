#pragma once

#include <stdint.h>
#include <stdbool.h>
#include <vulkan/vulkan.h>

#include "panvk_bc_s3tc_spv.h"
#include "panvk_bc_rgtc_spv.h"
#include "panvk_bc_bc6_spv.h"
#include "panvk_bc_bc7_spv.h"

struct panvk_bc_decode_push {
    uint32_t src[2];
    uint32_t dst[2];
    uint32_t src_row_B;
    uint32_t src_slice_B;
    uint32_t dst_row_B;
    uint32_t dst_slice_B;
    int32_t width;
    int32_t height;
    int32_t depth;
    int32_t format;
    int32_t x0;
    int32_t y0;
};

static inline bool panvk_format_is_bc(VkFormat format) {
    return format >= VK_FORMAT_BC1_RGB_UNORM_BLOCK &&
           format <= VK_FORMAT_BC7_SRGB_BLOCK;
}

static inline VkFormat panvk_bc_decoded_format(VkFormat format) {
    switch (format) {
    case VK_FORMAT_BC1_RGB_UNORM_BLOCK:
    case VK_FORMAT_BC1_RGBA_UNORM_BLOCK:
    case VK_FORMAT_BC2_UNORM_BLOCK:
    case VK_FORMAT_BC3_UNORM_BLOCK:
    case VK_FORMAT_BC7_UNORM_BLOCK:
    case VK_FORMAT_BC4_UNORM_BLOCK:
    case VK_FORMAT_BC5_UNORM_BLOCK:
        return VK_FORMAT_R8G8B8A8_UNORM;
    case VK_FORMAT_BC1_RGB_SRGB_BLOCK:
    case VK_FORMAT_BC1_RGBA_SRGB_BLOCK:
    case VK_FORMAT_BC2_SRGB_BLOCK:
    case VK_FORMAT_BC3_SRGB_BLOCK:
    case VK_FORMAT_BC7_SRGB_BLOCK:
        return VK_FORMAT_R8G8B8A8_SRGB;
    case VK_FORMAT_BC4_SNORM_BLOCK:
    case VK_FORMAT_BC5_SNORM_BLOCK:
        return VK_FORMAT_R8G8B8A8_SNORM;
    case VK_FORMAT_BC6H_UFLOAT_BLOCK:
    case VK_FORMAT_BC6H_SFLOAT_BLOCK:
        return VK_FORMAT_R16G16B16A16_SFLOAT;
    default:
        return VK_FORMAT_UNDEFINED;
    }
}

static inline uint32_t panvk_bc_block_size(VkFormat format) {
    switch (format) {
    case VK_FORMAT_BC1_RGB_UNORM_BLOCK:
    case VK_FORMAT_BC1_RGB_SRGB_BLOCK:
    case VK_FORMAT_BC1_RGBA_UNORM_BLOCK:
    case VK_FORMAT_BC1_RGBA_SRGB_BLOCK:
    case VK_FORMAT_BC4_UNORM_BLOCK:
    case VK_FORMAT_BC4_SNORM_BLOCK:
        return 8;
    default:
        return 16;
    }
}

static inline uint32_t panvk_bc_decoded_texel_size(VkFormat format) {
    switch (format) {
    case VK_FORMAT_BC6H_UFLOAT_BLOCK:
    case VK_FORMAT_BC6H_SFLOAT_BLOCK:
        return 8;
    default:
        return 4;
    }
}

static inline uint32_t panvk_bc_shader_index(VkFormat format) {
    if (format <= VK_FORMAT_BC3_SRGB_BLOCK)
        return 0; // s3tc
    if (format <= VK_FORMAT_BC5_SNORM_BLOCK)
        return 1; // rgtc
    if (format <= VK_FORMAT_BC6H_SFLOAT_BLOCK)
        return 2; // bc6
    return 3;     // bc7
}

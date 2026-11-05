// SPDX-License-Identifier: GPL-3.0-or-later
// cover-trim (MALI_COMPAT_COVER_TRIM): push constants shared by ct_cover.comp and ct_pred.comp. [keeper, Claude Opus 5.5] 2026-11-05
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_buffer_reference_uvec2 : require
layout(buffer_reference, std430, buffer_reference_align = 4) readonly buffer U32R { uint v[]; };
layout(buffer_reference, std430, buffer_reference_align = 4) buffer U32W { uint v[]; };
layout(push_constant) uniform P {
    uvec2 depth;      // depth aspect copy, w*h 32-bit texels, tightly packed
    uvec2 cov;        // one uint per 16x16 tile, 1 = some texel differs from the clear depth
    uvec2 pred;       // conditional-rendering predicates of this slot, one uint per cell
    uint w, h, tiles_x, tiles_y;
    uint gx, gy, nx, ny; // dispatch group counts (or w, h for pixel cells), cell grid
    uint dil_x, dil_y, clear_bits, pad;
} p;

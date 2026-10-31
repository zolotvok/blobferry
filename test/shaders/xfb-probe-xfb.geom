// SPDX-License-Identifier: GPL-3.0-or-later
#version 450
// xfb-probe: DXVK's stream-output passthrough geometry shader (points in, one point out, captured)
layout(points) in;
layout(points, max_vertices = 1) out;
layout(location = 0) in vec4 i0[];
layout(location = 1) in vec4 i1[];
layout(location = 2) in vec4 i2[];
layout(location = 0, xfb_buffer = 0, xfb_offset = 0, xfb_stride = 48) out vec4 o0;
layout(location = 1, xfb_buffer = 0, xfb_offset = 16) out vec4 o1;
layout(location = 2, xfb_buffer = 0, xfb_offset = 32) out vec4 o2;
void main() { o0 = i0[0]; o1 = i1[0]; o2 = i2[0]; EmitVertex(); }

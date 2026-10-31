// SPDX-License-Identifier: GPL-3.0-or-later
#version 450
// xfb-probe: the same passthrough geometry shader without capture
layout(points) in;
layout(points, max_vertices = 1) out;
layout(location = 0) in vec4 i0[];
layout(location = 1) in vec4 i1[];
layout(location = 2) in vec4 i2[];
layout(location = 0) out vec4 o0;
layout(location = 1) out vec4 o1;
layout(location = 2) out vec4 o2;
void main() { o0 = i0[0]; o1 = i1[0]; o2 = i2[0]; gl_Position = gl_in[0].gl_Position; EmitVertex(); }

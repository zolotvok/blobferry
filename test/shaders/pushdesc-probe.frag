// SPDX-License-Identifier: GPL-3.0-or-later
#version 450
// pushdesc-probe graphics case: each of the 12 pixels copies its vec4 of the push UBO into the SSBO
layout(set = 0, binding = 0) buffer Out { vec4 o[]; };
layout(set = 1, binding = 0) uniform Plain { vec4 p[12]; };
layout(set = 2, binding = 0) uniform Root { vec4 r[12]; };
layout(push_constant) uniform PC { uint base; };
void main() { uint i = uint(gl_FragCoord.x); o[base + i] = r[i] + p[i] * 0.0; }

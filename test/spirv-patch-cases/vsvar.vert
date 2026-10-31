// SPDX-License-Identifier: GPL-3.0-or-later
#version 450
// Plain VS with clip + cull (the common DXVK D3D11 case).
layout(location=0) in vec4 p;
void main() { gl_Position = p; gl_ClipDistance[0] = p.x; gl_ClipDistance[1] = p.y; gl_CullDistance[0] = p.z; }

// SPDX-License-Identifier: GPL-3.0-or-later
#version 450
// xfb-probe: vertex shader shaped like Unity's stream-output skinning VS (3 attributes in, 3 vec4 out, some ALU)
layout(location = 0) in vec4 a0;
layout(location = 1) in vec4 a1;
layout(location = 2) in vec4 a2;
layout(location = 0) out vec4 o0;
layout(location = 1) out vec4 o1;
layout(location = 2) out vec4 o2;
void main() {
    mat4 m = mat4(a0, a1, a2, vec4(0, 0, 0, 1));
    o0 = m * a0; o1 = m * a1; o2 = m * a2;
    gl_Position = vec4(o0.xyz, 1);
}

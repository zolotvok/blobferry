// SPDX-License-Identifier: GPL-3.0-or-later
#version 450
// Arrays of blocks in both directions; CullDistance only on one side.
layout(vertices=3) out;
in gl_PerVertex { vec4 gl_Position; float gl_ClipDistance[1]; float gl_CullDistance[1]; } gl_in[];
out gl_PerVertex { vec4 gl_Position; float gl_ClipDistance[1]; float gl_CullDistance[1]; } gl_out[];
void main() {
    gl_out[gl_InvocationID].gl_Position = gl_in[gl_InvocationID].gl_Position;
    gl_out[gl_InvocationID].gl_ClipDistance[0] = gl_in[gl_InvocationID].gl_ClipDistance[0];
    gl_out[gl_InvocationID].gl_CullDistance[0] = gl_in[2].gl_CullDistance[0];
    gl_TessLevelOuter[0] = 1.0; gl_TessLevelInner[0] = 1.0;
}

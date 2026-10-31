// SPDX-License-Identifier: GPL-3.0-or-later
#version 450
layout(triangles) in;
in gl_PerVertex { vec4 gl_Position; float gl_ClipDistance[1]; } gl_in[];
out gl_PerVertex { vec4 gl_Position; float gl_ClipDistance[1]; };
void main() { gl_Position = gl_in[0].gl_Position * gl_TessCoord.x; gl_ClipDistance[0] = gl_in[0].gl_ClipDistance[0] + gl_in[1].gl_ClipDistance[0]; }

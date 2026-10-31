// SPDX-License-Identifier: GPL-3.0-or-later
#version 450
out gl_PerVertex { vec4 gl_Position; float gl_PointSize; float gl_ClipDistance[2]; };
layout(location=0) out float o;
void main() { gl_Position = vec4(1.0); gl_ClipDistance[0] = 0.5; gl_ClipDistance[1] = gl_ClipDistance[0] * 2.0; float c[2] = gl_ClipDistance; o = c[1]; }

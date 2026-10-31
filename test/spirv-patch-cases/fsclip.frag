// SPDX-License-Identifier: GPL-3.0-or-later
#version 450
in float gl_ClipDistance[2];
layout(location=0) out vec4 c;
void main() { c = vec4(gl_ClipDistance[0], gl_ClipDistance[1], 0, 1); }

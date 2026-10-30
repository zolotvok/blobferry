#version 450
// xfb-probe: the same vertex shader, captured directly (no geometry shader)
layout(location = 0) in vec4 a0;
layout(location = 1) in vec4 a1;
layout(location = 2) in vec4 a2;
layout(location = 0, xfb_buffer = 0, xfb_offset = 0, xfb_stride = 48) out vec4 o0;
layout(location = 1, xfb_buffer = 0, xfb_offset = 16) out vec4 o1;
layout(location = 2, xfb_buffer = 0, xfb_offset = 32) out vec4 o2;
void main() {
    mat4 m = mat4(a0, a1, a2, vec4(0, 0, 0, 1));
    o0 = m * a0; o1 = m * a1; o2 = m * a2;
    gl_Position = vec4(o0.xyz, 1);
}

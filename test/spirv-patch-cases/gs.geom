#version 450
layout(triangles) in; layout(triangle_strip, max_vertices=3) out;
in gl_PerVertex { vec4 gl_Position; float gl_ClipDistance[2]; } gl_in[];
out gl_PerVertex { vec4 gl_Position; float gl_ClipDistance[2]; };
layout(location=0) out float o;
void main() { for (int i = 0; i < 3; i++) { gl_Position = gl_in[i].gl_Position; gl_ClipDistance[0] = gl_in[i].gl_ClipDistance[0]; gl_ClipDistance[1] = gl_in[2].gl_ClipDistance[1]; o = gl_in[i].gl_ClipDistance[1]; EmitVertex(); } }

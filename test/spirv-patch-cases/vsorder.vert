#version 450
out gl_PerVertex { float gl_ClipDistance[1]; vec4 gl_Position; float gl_PointSize; };
layout(location=0) out float o;
void main() { gl_ClipDistance[0] = 0.5; gl_Position = vec4(1.0); gl_PointSize = 1.0; o = gl_Position.x; }

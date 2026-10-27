#version 450
// ClipDistance between two kept members: PointSize must be renumbered.
out gl_PerVertex { vec4 gl_Position; float gl_ClipDistance[1]; float gl_PointSize; };
void main() { gl_Position = vec4(1.0); gl_ClipDistance[0] = 0.25; gl_PointSize = 2.0; }

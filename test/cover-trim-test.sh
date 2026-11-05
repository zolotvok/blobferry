#!/bin/bash
# SPDX-License-Identifier: GPL-3.0-or-later
# cover-trim-test.sh [outdir]: builds the compat layer for x86_64 + cover-trim-test.c and runs its 4 modes on lavapipe
# (PC only; no device). Exit 0 = all PASS. [keeper, Claude Opus 5.5] 2026-11-05
set -eu
B=$(cd "$(dirname "$0")/.." && pwd)
O=${1:-$(mktemp -d)}
mkdir -p "$O"
cat >"$O/ctt_quad.vert" <<'G'
#version 450
layout(push_constant) uniform P { vec4 r; } p;
void main() {
    const vec2 c[6] = vec2[](vec2(0, 0), vec2(1, 0), vec2(0, 1), vec2(1, 0), vec2(1, 1), vec2(0, 1));
    vec2 t = c[gl_VertexIndex];
    gl_Position = vec4(mix(p.r.xy, p.r.zw, t), 0.5, 1.0);
}
G
cat >"$O/ctt_gbuf.frag" <<'G'
#version 450
layout(location = 0) out vec4 o;
void main() { o = vec4(1.0, 0.5, 0.25, 1.0); }
G
cat >"$O/ctt_full.vert" <<'G'
#version 450
void main() {
    vec2 t = vec2((gl_VertexIndex << 1) & 2, gl_VertexIndex & 2);
    gl_Position = vec4(t * 2.0 - 1.0, 0.0, 1.0);
}
G
cat >"$O/ctt_comp.frag" <<'G'
#version 450
layout(location = 0) out vec4 o;
void main() { o = vec4(1.0); }
G
cat >"$O/ctt_probe.comp" <<'G'
#version 450
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_buffer_reference_uvec2 : require
layout(local_size_x = 8, local_size_y = 8) in;
layout(buffer_reference, std430, buffer_reference_align = 4) buffer M { uint v[]; };
layout(push_constant) uniform P { uvec2 marks; uint gx; uint val; } p;
void main() {
    if (gl_LocalInvocationIndex == 0u) M(p.marks).v[gl_WorkGroupID.y * p.gx + gl_WorkGroupID.x] = 1u + p.val;
}
G
for s in quad.vert:quad_vs gbuf.frag:gbuf_fs full.vert:full_vs comp.frag:comp_fs probe.comp:probe_cs; do
  glslangValidator -V --target-env vulkan1.1 --vn ctt_${s#*:} -o "$O/ctt_${s#*:}.h" "$O/ctt_${s%%:*}" >/dev/null
done
INC=$B/../panvk-kbase-android/work/mesa/include
clang -O2 -Wall -fPIC -shared -fvisibility=hidden -Wl,-Bsymbolic -Wl,-z,nodelete -I"$B/src/compat" -I"$INC" "$B/src/compat/mali_compat_layer.c" -o "$O/libVkLayer_droiddeck_mali_compat.so" -lpthread
clang -O2 -Wall -I"$O" -I"$INC" "$B/test/cover-trim-test.c" -o "$O/cover-trim-test" -l:libvulkan.so.1
cat >"$O/ctt-layer.json" <<J
{"file_format_version": "1.0.0", "layer": {"name": "VK_LAYER_DROIDDECK_mali_compat", "type": "GLOBAL",
 "library_path": "$O/libVkLayer_droiddeck_mali_compat.so", "api_version": "1.3.0", "implementation_version": "1",
 "description": "compat layer (cover-trim test)"}}
J
LVP=/usr/share/vulkan/icd.d/lvp_icd.x86_64.json
rc=0
for m in on off many toggle; do
  VK_ICD_FILENAMES=$LVP VK_LAYER_PATH=$O VK_LOADER_LAYERS_DISABLE='~implicit~' "$O/cover-trim-test" $m 2>"$O/stderr-$m.txt" || rc=1
  grep -E "cover-trim|FAIL|group |pixel " "$O/stderr-$m.txt" | head -8 || true
done
exit $rc

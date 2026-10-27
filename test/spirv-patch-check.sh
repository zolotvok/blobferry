#!/usr/bin/env bash
# Runs the compat layer's ClipDistance/CullDistance SPIR-V patcher on the cases in test/spirv-patch-cases/ and on
# the saved Oblivion module dumps (build/spv-diag, build/spv-run12), and checks every result with spirv-val.
# Needs glslangValidator and spirv-val (Fedora: glslang, spirv-tools). Exit 0 = no module invalid after patching
# that was valid before. [Claude Opus 5.5] 2026-10-27
set -u
B=$(cd "$(dirname "$0")/.." && pwd)
T=$(mktemp -d); trap 'rm -rf "$T"' EXIT
cc -O1 -w -I"$B/../panvk-kbase-android/work/mesa/include" -I"$B/src/compat" "$B/test/spirv-patch-host.c" \
   -o "$T/patch" -lpthread -lm || exit 2
mkdir -p "$T/m"
for f in "$B"/test/spirv-patch-cases/*.*; do
    glslangValidator -V --target-env vulkan1.1 -o "$T/m/case-$(basename "$f" | tr . -).spv" "$f" >/dev/null || { echo "glslang failed: $f"; exit 2; }
done
i=0
for f in "$B"/build/spv-diag/mali-spv/*-orig.spv "$B"/build/spv-run12/mali-spv/*-orig.spv; do
    [ -f "$f" ] && cp "$f" "$T/m/dump-$(printf %04d $i).spv" && i=$((i+1))
done
(cd "$T/m" && "$T/patch" *.spv) > "$T/patch.txt"
bad=0; total=0; unchanged=0
for f in "$T"/m/*.spv; do
    total=$((total+1))
    grep -q "^$f: unchanged" "$T/patch.txt" && { unchanged=$((unchanged+1)); case "$f" in */case-*) echo "UNPATCHED $(basename "$f")";; esac; }
    spirv-val --target-env vulkan1.3 --scalar-block-layout "$f" >/dev/null 2>&1 || continue  # invalid already
    if ! out=$(spirv-val --target-env vulkan1.3 --scalar-block-layout "$f.out" 2>&1); then
        bad=$((bad+1)); echo "INVALID $(basename "$f"): $(echo "$out" | head -1)"
    fi
done
echo "modules: $total (cases: $(ls "$B"/test/spirv-patch-cases | wc -l)), left unpatched: $unchanged, invalid after patching: $bad"
[ $bad -eq 0 ]

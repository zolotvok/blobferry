#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-or-later
# robust-probe-build.sh: shaders -> headers (glslangValidator), probe for aarch64 (same flags as build-bridge.sh: PanVK Debian
# sysroot + NDK lld) into bridge/build/robust-probe/robust-probe, and for the PC (x86, lavapipe test) into robust-probe-host.
# [tests, Claude Sonnet 5.5] 2026-11-01
set -eu
B=$(cd "$(dirname "$0")/.." && pwd)
SR=$B/../panvk-kbase-android/work/sysroot-bookworm-arm64
NDK=$HOME/Android/Sdk/ndk/29.0.14206865/toolchains/llvm/prebuilt/linux-x86_64/bin
GCCDIR=$(ls -d "$SR"/usr/lib/gcc/aarch64-linux-gnu/*/ | head -1)
T="--target=aarch64-linux-gnu --sysroot=$SR --gcc-install-dir=$GCCDIR --ld-path=$NDK/ld.lld"
INC="-I$B/../panvk-kbase-android/work/mesa/include -I$B/test"
OUT=$B/build/robust-probe; mkdir -p "$OUT"
for v in utb stb ubo ssbo wild; do
  glslangValidator -V --target-env vulkan1.2 -S comp -D$(echo $v | tr a-z A-Z)=1 --vn rp_$v -o "$B/test/shaders/robust-probe-$v.h" "$B/test/shaders/robust-probe.comp" > /dev/null
done
clang $T -O2 -Wall $INC "$B/test/robust-probe.c" -o "$OUT/robust-probe" -lvulkan
clang -O2 -Wall $INC "$B/test/robust-probe.c" -o "$OUT/robust-probe-host" /usr/lib64/libvulkan.so.1
# B10-20 replay probe (wd2-replay-probe.c, loads a .spv file at run time) [tests, Claude Opus 5.5] 2026-11-01
clang $T -O2 -Wall $INC "$B/test/wd2-replay-probe.c" -o "$OUT/wd2-replay-probe" -lvulkan
clang -O2 -Wall $INC "$B/test/wd2-replay-probe.c" -o "$OUT/wd2-replay-probe-host" /usr/lib64/libvulkan.so.1
md5sum "$OUT/robust-probe" "$OUT/robust-probe-host" "$OUT/wd2-replay-probe" "$OUT/wd2-replay-probe-host"

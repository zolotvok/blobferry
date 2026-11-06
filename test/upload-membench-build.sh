#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-or-later
# Builds bridge/test/upload-membench.c for the guest (aarch64 glibc, same toolchain as build-bridge.sh) and, with
# "host", an x86_64 copy for a PC smoke run on any local Vulkan driver. [keeper, Claude Opus 5.5] 2026-11-05
set -eu
B=$(cd "$(dirname "$0")/.." && pwd)
SR=$B/../panvk-kbase-android/work/sysroot-bookworm-arm64
NDK=$HOME/Android/Sdk/ndk/29.0.14206865/toolchains/llvm/prebuilt/linux-x86_64/bin
GCCDIR=$(ls -d "$SR"/usr/lib/gcc/aarch64-linux-gnu/*/ | head -1)
OUT=$B/build/membench
mkdir -p "$OUT"
INC=-I"$B/../panvk-kbase-android/work/mesa/include"
clang --target=aarch64-linux-gnu --sysroot="$SR" --gcc-install-dir="$GCCDIR" --ld-path="$NDK/ld.lld" -O2 -Wall $INC \
  "$B/test/upload-membench.c" -o "$OUT/upload-membench" -lvulkan
md5sum "$OUT/upload-membench"
if [ "${1:-}" = host ]; then
  clang -O2 -Wall $INC "$B/test/upload-membench.c" -o "$OUT/upload-membench-host" -l:libvulkan.so.1
  echo "built $OUT/upload-membench-host"
fi

#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-or-later
# Builds bridge/test/regsamp.c for the guest (aarch64 glibc, same toolchain as build-bridge.sh) -> bridge/build/regsamp/regsamp,
# and with "host" an x86_64 copy for the PC test (bridge/test/regsamp-test.sh). [keeper, Claude Opus 5.5] 2026-11-05
set -eu
B=$(cd "$(dirname "$0")/.." && pwd)
SR=$B/../panvk-kbase-android/work/sysroot-bookworm-arm64
NDK=$HOME/Android/Sdk/ndk/29.0.14206865/toolchains/llvm/prebuilt/linux-x86_64/bin
GCCDIR=$(ls -d "$SR"/usr/lib/gcc/aarch64-linux-gnu/*/ | head -1)
OUT=$B/build/regsamp
mkdir -p "$OUT"
clang --target=aarch64-linux-gnu --sysroot="$SR" --gcc-install-dir="$GCCDIR" --ld-path="$NDK/ld.lld" -O2 -Wall \
  "$B/test/regsamp.c" -o "$OUT/regsamp"
md5sum "$OUT/regsamp"
[ "${1:-}" = host ] && clang -O2 -Wall "$B/test/regsamp.c" -o "$OUT/regsamp-host" && echo "built $OUT/regsamp-host"
exit 0

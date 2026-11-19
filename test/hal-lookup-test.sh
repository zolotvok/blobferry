#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-or-later
# Builds and runs the PC selftest of the ICD's vendor HAL lookup (test/hal-lookup-test.c). Needs host clang with ASan/UBSan. [Claude Opus 5.5] 2026-11-19
set -eu
B=$(cd "$(dirname "$0")/.." && pwd)
ROOT=$B/..
[ -d "$ROOT/panvk-kbase-android" ] || ROOT=$B/../../..
INC=$ROOT/panvk-kbase-android/work/mesa/include
OUT=$B/build/hal-lookup-test
mkdir -p "$OUT"
clang -O1 -g -Wall -Wextra -Wno-unused-function -Wno-unused-parameter -fsanitize=address,undefined -I"$INC" \
  "$B/test/hal-lookup-test.c" -o "$OUT/hal-lookup-test" -lpthread
TMPDIR=$OUT "$OUT/hal-lookup-test"

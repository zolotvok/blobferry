#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-or-later
# K-DW PC test (see kdw-host.c). Usage: bridge/test/gt-dyndepth/run.sh [layer .c]  (default bridge/src/compat/mali_compat_layer.c;
# the staged build's copy is bridge/build/stage-kcs1+kdw/mali_compat_layer.c). [keeper, Claude Opus 5.5] 2026-11-01
set -eu
D=$(cd "$(dirname "$0")" && pwd); B=$D/../..
SRC=$(realpath "${1:-$B/src/compat/mali_compat_layer.c}")
T=$(mktemp -d)
cc -O1 -g -DLAYER_SRC="\"$SRC\"" -I"$B/src/compat" -I"$B/../panvk-kbase-android/work/mesa/include" \
   "$D/kdw-host.c" -o "$T/kdw-host" -lpthread -lm -ldl 2>&1 | grep -E "error|kdw-host.c.*warning" || true
"$T/kdw-host"
rm -rf "$T"

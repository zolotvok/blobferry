#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-or-later
# K-WC1 PC test (see wc-host.c). Usage: bridge/test/wait-callers/run.sh [layer .c]  (default bridge/src/compat/mali_compat_layer.c;
# the staged build's copy is bridge/build/stage-kcs1+wc1/mali_compat_layer.c). [keeper, Claude Opus 5.5] 2026-11-01
set -eu
D=$(cd "$(dirname "$0")" && pwd); B=$D/../..
SRC=$(realpath "${1:-$B/src/compat/mali_compat_layer.c}")
T=$(mktemp -d)
cc -O1 -g -rdynamic -DLAYER_SRC="\"$SRC\"" -I"$B/src/compat" -I"$B/../panvk-kbase-android/work/mesa/include" \
   "$D/wc-host.c" -o "$T/wc-host" -lpthread -lm -ldl 2>&1 | grep -E "error|wc-host.c.*warning" || true
"$T/wc-host"
rm -rf "$T"

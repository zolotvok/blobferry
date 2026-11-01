#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-or-later
# K-WP1 PC test: builds test_pace.cpp against the WSI layer's mali_pace.hpp and runs it with MALI_WSI_PACE off and on.
# Usage: bridge/test/wsi-pace/run.sh [dir containing mali_pace.hpp]  (default: the WSI submodule's wsi/ after the
# 0002 patch is applied; the staged copy bridge/build/stage-wsi-pace/src/wsi works too). [keeper, Claude Opus 5.5] 2026-11-01
set -eu
D=$(cd "$(dirname "$0")" && pwd)
INC=${1:-$D/../../third_party/vulkan-wsi-layer/wsi}
[ -f "$INC/mali_pace.hpp" ] || { echo "no mali_pace.hpp in $INC" >&2; exit 2; }
T=$(mktemp -d)
g++ -std=c++17 -O1 -Wall -Wextra -Werror -I"$INC" "$D/test_pace.cpp" -o "$T/test_pace" -pthread
env -u MALI_WSI_PACE "$T/test_pace" off
MALI_WSI_PACE=1 MALI_WSI_PACE_HZ=60 MALI_WSI_PACE_FILE="$T/pace.txt" "$T/test_pace" on
rm -rf "$T"

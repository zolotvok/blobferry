#!/bin/sh
# SPDX-License-Identifier: GPL-3.0-or-later
# Regenerate the cover-trim SPIR-V headers (MALI_COMPAT_COVER_TRIM). [keeper, Claude Opus 5.5] 2026-11-05
set -eu
cd "$(dirname "$0")"
for s in cover pred; do
   glslangValidator -V --target-env vulkan1.1 --vn ct_${s}_spv -o ct_${s}_spv.h ct_${s}.comp
done
sed -i 's/^const uint32_t/static const uint32_t/' ct_*_spv.h

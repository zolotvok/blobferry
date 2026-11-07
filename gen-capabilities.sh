#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-or-later
# gen-capabilities.sh <version> <layer.so> <out.json>: fills src/compat/capabilities.json.in (version, md5 of the compat
# layer). Shipped as usr/share/blobferry/capabilities.json; the DroidDeck app reads it to show the switch status.
# Flip "rt_pipeline"/"dxr" in the .in file when the emulation lands (keep rt_pipeline_emulation_available() in step).
set -eu
B=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$(dirname "$3")"
sed -e "s/@VERSION@/$1/" -e "s/@LAYER_MD5@/$(md5sum "$2" | cut -d' ' -f1)/" "$B/src/compat/capabilities.json.in" > "$3"

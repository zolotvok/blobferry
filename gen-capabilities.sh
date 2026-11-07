#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-or-later
# gen-capabilities.sh <version> <layer.so> <out.json>: fills src/compat/capabilities.json.in (version, md5 of the compat
# layer) and adds the "options" array from src/compat/options.json (every switch the bridge honours: type, default,
# allowed values, category, status, description, games, min_layer). Shipped as usr/share/blobferry/capabilities.json;
# the DroidDeck app reads it to show the switch status and the selectable options.
# Flip "rt_pipeline"/"dxr" in the .in file when the emulation lands (keep rt_pipeline_emulation_available() in step).
# Keep options.json in step with the code: test/options-drift-check.sh fails when they differ.
set -eu
B=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$(dirname "$3")"
sed -e "s/@VERSION@/$1/" -e "s/@LAYER_MD5@/$(md5sum "$2" | cut -d' ' -f1)/" "$B/src/compat/capabilities.json.in" |
python3 -c '
import json, sys
c = json.load(sys.stdin)
c["options_schema"] = 1
c["options"] = json.load(open(sys.argv[1]))
json.dump(c, sys.stdout, indent=1, ensure_ascii=False)
sys.stdout.write("\n")' "$B/src/compat/options.json" > "$3"

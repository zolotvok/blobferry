#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-or-later
# options-drift-check.sh: src/compat/options.json (the switch manifest shipped in capabilities.json) must list exactly the
# environment keys the code reads. FAILS (exit 1) when a getenv("MALI_COMPAT_*" / "MALI_WSI_*" / "MALI_BRIDGE_*") in the
# layer, the ICD shim or the WSI patches has no manifest entry, when a MALI_COMPAT_* name appears anywhere in the layer
# sources without one, when the manifest lists a key no code reads, or when an entry is malformed.
# ENABLE_/DISABLE_MALI_COMPAT and _WSI are loader keys: they must appear in build-bridge.sh / build-wsi.sh.
set -eu
B=$(cd "$(dirname "$0")/.." && pwd)
exec python3 -I - "$B" <<'PY'
import glob, json, re, sys
B = sys.argv[1]
opts = json.load(open(B + "/src/compat/options.json"))
names = [o["name"] for o in opts]
bad = []
dup = {n for n in names if names.count(n) > 1}
if dup: bad.append("duplicate manifest entries: " + " ".join(sorted(dup)))
TYPES = {"bool", "int", "hex", "enum", "string", "list"}
CATS = {"compat-sync", "memory", "perf", "rt", "workaround", "diagnostics", "wsi", "icd"}
STAT = {"stable", "experimental", "debug-only"}
for o in opts:
    for k in ("name", "type", "default", "values", "category", "status", "description", "games", "min_layer"):
        if k not in o: bad.append(f"{o.get('name')}: missing field {k}")
    if o.get("type") not in TYPES: bad.append(f"{o['name']}: bad type {o.get('type')}")
    if o.get("category") not in CATS: bad.append(f"{o['name']}: bad category {o.get('category')}")
    if o.get("status") not in STAT: bad.append(f"{o['name']}: bad status {o.get('status')}")
    if o.get("type") == "enum" and not isinstance(o.get("values"), list): bad.append(f"{o['name']}: enum without a values list")
    if not o.get("description"): bad.append(f"{o['name']}: empty description")
manifest = set(names)

def read(p): return open(p, errors="replace").read()
src_files = glob.glob(B + "/src/compat/*.c") + glob.glob(B + "/src/compat/*.h") + glob.glob(B + "/src/icd/*.c")
patch_files = glob.glob(B + "/patches/vulkan-wsi-layer/*.patch")  # *-staged variants are not applied and do not count
GET = re.compile(r'getenv\(\s*"((?:MALI_COMPAT|MALI_WSI|MALI_BRIDGE)_[A-Z0-9_]+)"')
code = set()
for p in src_files: code |= set(GET.findall(read(p)))
for p in patch_files:
    # only lines the patch adds ("+..."), so context lines of upstream code cannot count
    code |= set(GET.findall("\n".join(l[1:] for l in read(p).splitlines() if l.startswith("+"))))
mentions = set()
for p in src_files:
    if "/src/compat/" in p: mentions |= set(re.findall(r'\bMALI_COMPAT_[A-Z0-9_]*[A-Z0-9]\b', read(p)))
loader = {"ENABLE_MALI_COMPAT": "build-bridge.sh", "DISABLE_MALI_COMPAT": "build-bridge.sh",
          "ENABLE_MALI_WSI": "build-wsi.sh", "DISABLE_MALI_WSI": "build-wsi.sh"}
for k, f in loader.items():
    if k not in read(B + "/" + f): bad.append(f"{k}: not found in {f} (manifest lists it as a loader key)")
    code.add(k)
for k in sorted(code - manifest): bad.append(f"read in code but missing from options.json: {k}")
for k in sorted(manifest - code): bad.append(f"listed in options.json but not read by any code: {k}")
for k in sorted(mentions - manifest): bad.append(f"named in the layer sources but missing from options.json (read through a helper? add it, or fix the comment): {k}")
from collections import Counter
c = Counter(o["status"] for o in opts)
print(f"options-drift-check: {len(opts)} manifest keys ({', '.join(f'{k} {v}' for k, v in sorted(c.items()))}), {len(code)} read in code")
if bad:
    print("FAIL"); [print("  " + b) for b in bad]; sys.exit(1)
print("OK")
PY

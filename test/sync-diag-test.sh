#!/bin/bash
# SPDX-License-Identifier: GPL-3.0-or-later
# sync-diag-test.sh [outdir]: builds the compat layer for x86_64 + sync-diag-test.c and checks MALI_COMPAT_FULL_BARRIERS,
# MALI_COMPAT_SUBMIT_IDLE, MALI_COMPAT_GT_SPLIT_SKIP and MALI_COMPAT_SYNC_CTL on lavapipe (PC only; no device).
# Exit 0 = all PASS. [keeper, Claude Opus 5.5] 2026-11-05
set -u
B=$(cd "$(dirname "$0")/.." && pwd)
O=${1:-$(mktemp -d)}
mkdir -p "$O"
INC=$B/../panvk-kbase-android/work/mesa/include
[ -d "$INC" ] || INC=/home/zolotvok/Droid-Deck_Mali_Support/panvk-kbase-android/work/mesa/include
clang -O2 -Wall -fPIC -shared -fvisibility=hidden -Wl,-Bsymbolic -Wl,-z,nodelete -I"$B/src/compat" -I"$INC" \
  "$B/src/compat/mali_compat_layer.c" -o "$O/libVkLayer_droiddeck_mali_compat.so" -lpthread || exit 2
clang -O2 -Wall -I"$INC" "$B/test/sync-diag-test.c" -o "$O/sync-diag-test" -l:libvulkan.so.1 || exit 2
cat >"$O/sd-layer.json" <<J
{"file_format_version": "1.0.0", "layer": {"name": "VK_LAYER_DROIDDECK_mali_compat", "type": "GLOBAL",
 "library_path": "$O/libVkLayer_droiddeck_mali_compat.so", "api_version": "1.3.0", "implementation_version": "1",
 "description": "compat layer (sync-diag test)"}}
J
LVP=/usr/share/vulkan/icd.d/lvp_icd.x86_64.json
fails=0
ok() { echo "PASS $*"; }
bad() { echo "FAIL $*"; fails=$((fails + 1)); }
run() { # run <name> <iters> [args...] -- [env...]
  local nm=$1 it=$2; shift 2
  local args=()
  while [ $# -gt 0 ] && [ "$1" != -- ]; do args+=("$1"); shift; done
  [ "${1:-}" = -- ] && shift
  rm -f "$O/$nm.log"
  env VK_ICD_FILENAMES=$LVP VK_LAYER_PATH=$O VK_INSTANCE_LAYERS=VK_LAYER_DROIDDECK_mali_compat VK_LOADER_LAYERS_DISABLE='~implicit~' \
    MALI_COMPAT_SYNC_LOG="$O/$nm.log" MALI_COMPAT_SYNC_REPORT_S=1 SYNC_TEST_TAIL=1 "$@" \
    "$O/sync-diag-test" "$it" 50 "${args[@]}" >"$O/$nm.out" 2>"$O/$nm.err" || bad "$nm exit $?"
  grep -q "DONE iters $it" "$O/$nm.out" && ok "$nm ran" || bad "$nm no DONE"
  grep -q "copy result 12345678" "$O/$nm.out" && ok "$nm copy data intact" || bad "$nm copy result: $(grep copy "$O/$nm.out")"
}
sum() { # sum <name> <field words>: total of "<field> N" over the counter lines
  python3 -I -c 'import re,sys; t=open(sys.argv[1]).read() if __import__("os").path.exists(sys.argv[1]) else ""; print(sum(int(x) for x in re.findall(re.escape(sys.argv[2])+r" (\d+)", t)))' "$O/$1.log" "$2"
}
eq() { # eq <name> <field> <expected>
  local v; v=$(sum "$1" "$2")
  [ "$v" = "$3" ] && ok "$1 $2 = $3" || bad "$1 $2 = $v (want $3)"
}

run unset 10
grep -q "sync-diag" "$O/unset.err" && bad "unset: sync-diag line" || ok "unset: no sync-diag line"
[ -e "$O/unset.log" ] && bad "unset: log written" || ok "unset: no log file"

run fb1 20 -- MALI_COMPAT_FULL_BARRIERS=1
eq fb1 "barriers widened" 40; eq fb1 "added" 0; eq fb1 "in-pass kept" 0
grep -q "sync-diag mode fb 1 si 0 ms k4 0 at .* (env) fbm 0x00f" "$O/fb1.log" && ok "fb1 mode line" || bad "fb1 mode line"

run fb2 20 -- MALI_COMPAT_FULL_BARRIERS=2
eq fb2 "barriers widened" 40; eq fb2 "added" 140; eq fb2 "split-part skips" 40

run si 20 -- MALI_COMPAT_SUBMIT_IDLE=1
eq si "submits" 20; eq si "idled" 20; eq si "timeouts" 0; eq si "fails" 0; eq si "added" 0
grep -q "sync-diag mode fb 0 si 250 ms" "$O/si.log" && ok "si=1 -> 250 ms cap" || bad "si cap"

run ctl 20 5:fb=2 10:"si=1 fb=0" 15:off -- MALI_COMPAT_SYNC_CTL="$O/ctl.txt" SYNC_TEST_CTL="$O/ctl.txt"
rm -f "$O/ctl.txt"
eq ctl "barriers widened" 10; eq ctl "added" 35; eq ctl "idled" 5; eq ctl "submits" 20
[ "$(grep -c '(ctl file)' "$O/ctl.log")" = 3 ] && ok "ctl: 3 mode changes logged" || bad "ctl mode lines $(grep -c '(ctl file)' "$O/ctl.log")"
grep -q "sync-diag mode fb 0 si 250 ms k4 0 at .* (ctl file)" "$O/ctl.log" && ok "ctl: si window line" || bad "ctl si line"

run ctloff 10 -- MALI_COMPAT_SYNC_CTL="$O/none.txt"
eq ctloff "barriers widened" 0; eq ctloff "added" 0; eq ctloff "idled" 0; eq ctloff "submits" 10
grep -q "hooks on, ctl $O/none.txt" "$O/ctloff.log" && ok "ctl armed with no file = all off" || bad "ctloff armed line"

run k4 20 -- MALI_COMPAT_GPU_TIME=1 MALI_COMPAT_GT_SPLIT_SKIP=1
eq k4 "k4 untimed parts" 40
grep -q "hooks off (k4 only)" "$O/k4.log" && ok "k4 alone installs no hooks" || bad "k4 armed line"
run k4off 10 -- MALI_COMPAT_GPU_TIME=1 MALI_COMPAT_SYNC_CTL="$O/none.txt"
eq k4off "k4 untimed parts" 0

run gtfb2 20 -- MALI_COMPAT_GPU_TIME=1 MALI_COMPAT_FULL_BARRIERS=2 MALI_COMPAT_SUBMIT_IDLE=1 MALI_COMPAT_GT_SPLIT_SKIP=1
eq gtfb2 "added" 140; eq gtfb2 "idled" 20; eq gtfb2 "k4 untimed parts" 40
grep -q "GPU timing on" "$O/gtfb2.err" && ok "gtfb2: GPU timing chain below sync-diag" || bad "gtfb2 GPU timing line"

run nsr 20 -- MALI_COMPAT_NO_SUSPEND_RESUME=1 MALI_COMPAT_FULL_BARRIERS=2
eq nsr "added" 180; eq nsr "split-part skips" 0

run wbs 10 -- MALI_COMPAT_SUBMIT_IDLE=100 SYNC_TEST_WBS=1
eq wbs "timeouts" 1; eq wbs "submits" 11; eq wbs "idled" 10; eq wbs "fails" 0
grep -q "wbs signaled" "$O/wbs.out" && ok "wbs: no deadlock, capped wait returned" || bad "wbs"

run fbm1 20 -- MALI_COMPAT_SYNC_CTL="$O/none.txt" MALI_COMPAT_FB_MASK=0x001
eq fbm1 "barriers widened" 40; eq fbm1 "added" 0
grep -q "sync-diag mode fb 3 si 0 ms k4 0 at .* (env) fbm 0x001" "$O/fbm1.log" && ok "fbm1 mode line (fb 3 = subset)" || bad "fbm1 mode line"
eq fbm1 "app barriers" 40; eq fbm1 "dst transfer" 20; eq fbm1 "dst frag" 20; eq fbm1 "src transfer" 20; eq fbm1 "dst tess" 0
run fbmd 20 -- MALI_COMPAT_FB_MASK=0x040
eq fbmd "added" 20; eq fbmd "barriers widened" 0
run fbmp 20 -- MALI_COMPAT_FB_MASK=0x030
eq fbmp "added" 80; eq fbmp "split-part skips" 40
run fbmx 20 -- MALI_COMPAT_FB_MASK=0x080
eq fbmx "added" 40
run fbmctl 20 5:"fbm=0x040 si=0 k4=0" 10:"fb=1 si=0 k4=0" 15:"fbm=0x300" -- MALI_COMPAT_SYNC_CTL="$O/ctl2.txt" SYNC_TEST_CTL="$O/ctl2.txt"
rm -f "$O/ctl2.txt"
eq fbmctl "added" 5; eq fbmctl "barriers widened" 10
grep -q "sync-diag mode fb 1 si 0 ms k4 0 at .* (ctl file) fbm 0x00f" "$O/fbmctl.log" && ok "fbmctl fb=1 = 0x00f" || bad "fbmctl fb=1 line"
grep -q "(ctl file) fbm 0x300" "$O/fbmctl.log" && ok "fbmctl fbm=0x300 line" || bad "fbmctl 0x300 line"

echo "fails $fails ($O)"
[ $fails = 0 ]

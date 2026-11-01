#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-or-later
# PC stress test for libhybris' lazy init of static bionic mutexes/conds/rwlocks: old code (submodule HEAD) vs the
# working tree (CAS fix). Cuts the pthread hook section out of hooks.c, builds stress.c around it for x86_64 (native)
# and aarch64 (qemu-aarch64-static, PanVK's bookworm sysroot), runs both in a natural and a slow-malloc mode.
# PASS = new code 0 violations everywhere AND old code > 0 violations (or a hang) in the slow mode.
# Usage: bridge/test/hybris-lazyinit/run.sh [outdir]   [Blobferry keeper, Claude Opus 5.5, 2026-11-01]
set -u
T=$(cd "$(dirname "$0")" && pwd)
B=$(cd "$T/../.." && pwd)
HY=$B/third_party/libhybris
OUT=${1:-$B/build/stage-hybris-cas/stress}
SR=$B/../panvk-kbase-android/work/sysroot-bookworm-arm64
NDK=$HOME/Android/Sdk/ndk/29.0.14206865/toolchains/llvm/prebuilt/linux-x86_64/bin
mkdir -p "$OUT"

# hook section = the helpers block ("Base address to check" .. before "utils, such as malloc"), the mutex+cond hooks
# ("pthread_mutex* functions" .. before pthread_setname_np) and the rwlock hooks (.. before "Bionic implementation")
cut_region() {
  awk '
    /\/\* Base address to check for Android specifics \*\//      { on = 1 }
    /^ \* utils, such as malloc, memcpy/                           { on = 0; print "*/" }
    /^ \* pthread_mutex\* functions/                               { on = 1; print "/*" }
    /^int _hybris_hook_pthread_setname_np/                         { on = 0 }
    /^ \* pthread_rwlockattr_\* functions/                         { on = 1; print "/*" }
    /^\/\* Bionic implementation of pthread_cleanup_push/          { on = 0 }
    on { print }
  ' "$1"
}
mkdir -p "$OUT/old" "$OUT/new" "$OUT/inc"
git -C "$HY" show HEAD:hybris/common/hooks.c > "$OUT/old/hooks.c"
cp "$HY/hybris/common/hooks.c" "$OUT/new/hooks.c"
for v in old new; do cut_region "$OUT/$v/hooks.c" > "$OUT/$v/hooks_region.c"; done
# logging stubs (the region includes "logging.h")
cat > "$OUT/inc/logging.h" <<'EOF'
#define HYBRIS_DEBUG_LOG(module, message, ...) do { } while (0)
#define TRACE(message, ...) do { } while (0)
EOF
echo "regions: old $(wc -l < "$OUT/old/hooks_region.c") lines, new $(wc -l < "$OUT/new/hooks_region.c") lines; hooks.c md5 old $(md5sum < "$OUT/old/hooks.c" | cut -c1-8) new $(md5sum < "$OUT/new/hooks.c" | cut -c1-8)"

GCCDIR=$(ls -d "$SR"/usr/lib/gcc/aarch64-linux-gnu/*/ 2>/dev/null | head -1)
A64="--target=aarch64-linux-gnu --sysroot=$SR --gcc-install-dir=$GCCDIR --ld-path=$NDK/ld.lld"
CF="-O2 -pthread -w -I$OUT/inc -I$HY/hybris/common"
fail=0
for v in old new; do
  gcc $CF -I"$OUT/$v" "$T/stress.c" -o "$OUT/stress-$v-x86_64" || { echo "build x86_64 $v FAILED"; exit 2; }
  if [ -n "$GCCDIR" ] && command -v qemu-aarch64-static >/dev/null; then
    clang $A64 $CF -I"$OUT/$v" "$T/stress.c" -o "$OUT/stress-$v-aarch64" || { echo "build aarch64 $v FAILED"; exit 2; }
  fi
done

runone() {  # label slow_us rounds binary... ; one process per case, so a wedged case cannot hide the others
  local label=$1 slow=$2 rounds=$3; shift 3
  local rc=0 k m r
  for k in 0 1 2 3; do for m in 0 1; do
    STRESS_CASE=$k:$m STRESS_SLOW_US=$slow STRESS_ROUNDS=$rounds timeout 300 "$@" "$label" 2>>"$OUT/$label.stderr" | grep -v TOTAL
    r=${PIPESTATUS[0]}
    [ $r = 124 ] && echo "$label case $k:$m TIMEOUT 300 s"
    [ $r != 0 ] && rc=1
  done; done
  echo "$label TOTAL -> $([ $rc = 0 ] && echo PASS || echo FAIL)"
  return $rc
}
: > "$OUT/summary.txt"
for arch in x86_64 aarch64; do
  [ -x "$OUT/stress-old-$arch" ] || continue
  run=("$OUT/stress-old-$arch"); runn=("$OUT/stress-new-$arch")
  if [ $arch = aarch64 ]; then run=(qemu-aarch64-static -L "$SR" "${run[@]}"); runn=(qemu-aarch64-static -L "$SR" "${runn[@]}"); fi
  for mode in "natural 0 2000" "slow 200 100"; do
    set -- $mode
    rm -f "$OUT/old-$arch-$1.stderr" "$OUT/new-$arch-$1.stderr"
    runone "old-$arch-$1" "$2" "$3" "${run[@]}" | tee -a "$OUT/summary.txt"; orc=${PIPESTATUS[0]}
    runone "new-$arch-$1" "$2" "$3" "${runn[@]}" | tee -a "$OUT/summary.txt"; nrc=${PIPESTATUS[0]}
    races=$(grep -c "hybris-lazyinit: race" "$OUT/new-$arch-$1.stderr" 2>/dev/null)
    echo "== $arch $1: old rc=$orc, new rc=$nrc, new resolved-race lines=$races (first 16 + powers of two)" | tee -a "$OUT/summary.txt"
    [ "$nrc" = 0 ] || fail=1
    [ "$1" = slow ] && [ "$orc" = 0 ] && { echo "   old code showed no race in slow mode: the test does not reproduce it" | tee -a "$OUT/summary.txt"; fail=1; }
  done
done
echo "stress: $([ $fail = 0 ] && echo PASS || echo FAIL) (outputs in $OUT)" | tee -a "$OUT/summary.txt"
exit $fail

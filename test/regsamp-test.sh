#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-or-later
# PC end-to-end test of regsamp (x86_64 copy): a spinning child holds a known value in r12; regsamp samples it 2 s at
# 500 Hz; the text must parse with regsample.py's reader and every sample must show r12 = the value. [keeper, Claude Opus 5.5]
set -eu
B=$(cd "$(dirname "$0")/.." && pwd)
bash "$B/test/regsamp-build.sh" host >/dev/null
T=$(mktemp -d)
printf 'int main(void){ register unsigned long r12 __asm__("r12") = 0x5f00000123; for(;;){ __asm__ volatile("" : "+r"(r12)); } }\n' > "$T/spin.c"
clang -O1 "$T/spin.c" -o "$T/spin"
"$T/spin" & P=$!
sleep 0.3
o=$("$B/build/regsamp/regsamp-host" "$P" 2 500 "$T/rs.txt"); kill $P
echo "$o" | tail -1
B="$B" python3 -c "
import sys; sys.path.insert(0, '$B/test'); from regsample import read_dump
s = read_dump(open('$T/rs.txt').read())
good = sum(1 for _, ip, r in s if r.get('r12') == 0x5f00000123)
print('regsamp-test: samples %d, r12 ok %d' % (len(s), good))
sys.exit(0 if len(s) >= 500 and good == len(s) else 1)
" && echo "regsamp-test: PASS" || { echo "regsamp-test: FAIL"; exit 1; }
rm -rf "$T"
# --self: forked spinner with x19/r12 = 0x5f00000123
T=$(mktemp -d)
o=$("$B/build/regsamp/regsamp-host" --self 1 500 "$T/self.txt"); echo "$o" | tail -1
python3 -c "
import sys; sys.path.insert(0, '$B/test'); from regsample import read_dump
s = read_dump(open('$T/self.txt').read()); g = sum(1 for _, ip, r in s if r.get('r12') == 0x5f00000123)
print('regsamp-test --self: samples %d, r12 ok %d' % (len(s), g)); sys.exit(0 if len(s) >= 250 and g == len(s) else 1)
" && echo "regsamp-test --self: PASS" || { echo "regsamp-test --self: FAIL"; exit 1; }
# DIAG on a failed open (pid 1 = not ours)
o=$("$B/build/regsamp/regsamp-host" 1 1 100 "$T/x.txt") || true; echo "$o" | grep -q "^regsamp: DIAG paranoid=.* self_open=ok .*target_errno=" && echo "regsamp-test DIAG: PASS" || { echo "regsamp-test DIAG: FAIL ($o)"; exit 1; }
rm -rf "$T"

#!/usr/bin/env python3
# Summarises the compat layer's image log (/root/mali-images.txt): estimated uncompressed bytes per
# (format, bc-shadow, flags, compression) group, biggest first. Fixed-rate (comp 0x2) groups are also shown at their
# rate (2 bpc = 8 bpp, 3 bpc = 12 bpp for 4-channel 8-bit formats). [Claude Opus 5.5] 2026-10-26
# Usage: images-summary.py mali-images.txt [N]
import collections
import re
import sys

BPP = {9: 1, 13: 1, 16: 2, 37: 4, 43: 4, 44: 4, 50: 4, 64: 4, 70: 2, 76: 2, 77: 4, 83: 4, 91: 8, 97: 8, 98: 4,
       100: 4, 101: 8, 103: 8, 107: 16, 109: 16, 122: 4, 123: 4, 124: 2, 126: 4, 129: 4, 130: 8}
LINE = re.compile(r"fmt (\d+) (\d+)x(\d+)x(\d+) mips (\d+) layers (\d+) samples (\d+) tiling (\d+) usage (0x[0-9a-f]+) "
                  r"flags (0x[0-9a-f]+)( bc-shadow)? comp (0x[0-9a-f]+) rate (0x[0-9a-f]+)")


def size(fmt, w, h, d, mips, layers, samples):
    total = 0
    for m in range(mips):
        total += max(w >> m, 1) * max(h >> m, 1) * max(d >> m, 1)
    return total * layers * samples * BPP.get(fmt, 4)


groups = collections.defaultdict(lambda: [0, 0, 0])
for line in open(sys.argv[1]):
    m = LINE.match(line)
    if not m:
        continue
    fmt, w, h, d, mips, layers, samples = (int(m.group(i)) for i in range(1, 8))
    usage, flags, bc, comp, rate = m.group(9), m.group(10), bool(m.group(11)), m.group(12), int(m.group(13), 16)
    raw = size(fmt, w, h, d, mips, layers, samples)
    est = raw * 8 // 32 if rate == 0x2 else raw * 12 // 32 if rate == 0x4 else raw
    g = groups[(fmt, bc, flags, usage, comp, hex(rate))]
    g[0] += 1
    g[1] += raw
    g[2] += est

n = int(sys.argv[2]) if len(sys.argv) > 2 else 25
total = sum(g[2] for g in groups.values())
print("estimated total %.0f MiB (uncompressed %.0f MiB), %d images" % (total / 2**20, sum(g[1] for g in groups.values()) / 2**20,
                                                                         sum(g[0] for g in groups.values())))
print("%6s %9s %9s  key (fmt, bc-shadow, flags, usage, comp, rate)" % ("count", "est MiB", "raw MiB"))
for key, g in sorted(groups.items(), key=lambda kv: -kv[1][2])[:n]:
    print("%6d %9.0f %9.0f  %s" % (g[0], g[2] / 2**20, g[1] / 2**20, key))

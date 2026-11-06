#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# Blobferry: report the MSVC volatile metadata of PE files (what FEX's ImageTracker::LoadImageVolatileMetadata reads).
# usage: volmeta.py FILE... [--ranges]   (one line per file; exit 0)
# Columns: VolatileMetadataPointer, listed volatile accesses, valid ranges, bytes covered, % of executable sections
# covered, listed accesses per KiB covered. "none" = no load config / no pointer (FEX keeps global TSO for the module).
import struct, sys


def parse(path):
    data = open(path, 'rb').read()
    if data[:2] != b'MZ':
        return 'not a PE'
    pe = struct.unpack_from('<I', data, 0x3c)[0]
    if data[pe:pe + 4] != b'PE\0\0':
        return 'not a PE'
    machine, nsec, _, _, _, optsize, _ = struct.unpack_from('<HHIIIHH', data, pe + 4)
    opt = pe + 24
    magic = struct.unpack_from('<H', data, opt)[0]
    if magic == 0x20b:
        base = struct.unpack_from('<Q', data, opt + 24)[0]
        ddir = opt + 112
        vm_off = 0x100  # VolatileMetadataPointer in IMAGE_LOAD_CONFIG_DIRECTORY64
        ptr_size = 8
    else:
        base = struct.unpack_from('<I', data, opt + 28)[0]
        ddir = opt + 96
        vm_off = 0xa0
        ptr_size = 4
    secs = []
    sh = opt + optsize
    for i in range(nsec):
        name, vsize, va, rawsize, rawptr = struct.unpack_from('<8sIIII', data, sh + 40 * i)
        chars = struct.unpack_from('<I', data, sh + 40 * i + 36)[0]
        secs.append((name.rstrip(b'\0').decode(errors='replace'), va, vsize, rawptr, rawsize, chars))

    def rva2off(rva):
        for _, va, vsize, rawptr, rawsize, _ in secs:
            if va <= rva < va + max(vsize, rawsize):
                return rawptr + rva - va
        return None

    exec_bytes = sum(vsize for _, _, vsize, _, _, chars in secs if chars & 0x20000000)
    lc_rva, lc_size = struct.unpack_from('<II', data, ddir + 8 * 10)
    if not lc_rva:
        return 'none (no load config)  machine 0x%x' % machine
    lco = rva2off(lc_rva)
    size = struct.unpack_from('<I', data, lco)[0]
    if size <= vm_off:
        return 'none (load config 0x%x too small)  machine 0x%x' % (size, machine)
    vmp = struct.unpack_from('<Q' if ptr_size == 8 else '<I', data, lco + vm_off)[0]
    if not vmp:
        return 'none (VolatileMetadataPointer 0)  machine 0x%x' % machine
    vo = rva2off(vmp - base)
    _, ver, acc, accsz, rng, rngsz = struct.unpack_from('<IIIIII', data, vo)
    nacc = accsz // 4
    nrng = rngsz // 8
    ro = rva2off(rng)
    ranges = [struct.unpack_from('<II', data, ro + 8 * i) for i in range(nrng)]
    covered = sum(s for _, s in ranges)
    ao = rva2off(acc)
    accs = sorted(struct.unpack_from('<I', data, ao + 4 * i)[0] for i in range(nacc))
    # Ranges that hold no listed access at all (fully "no-access" code under the patch).
    import bisect
    empty = [(r, s) for r, s in ranges if bisect.bisect_left(accs, r + s) - bisect.bisect_left(accs, r) == 0]
    out = ('ptr 0x%x  accesses %d  ranges %d  covered %d B = %.1f %% of exec %d B  acc/KiB %.2f  empty-ranges %d (%d B)'
           % (vmp, nacc, nrng, covered, 100.0 * covered / exec_bytes if exec_bytes else 0, exec_bytes,
              nacc / (covered / 1024.0) if covered else 0, len(empty), sum(s for _, s in empty)))
    if '--ranges' in sys.argv:
        out += '\n' + '\n'.join('    0x%x +0x%x' % r for r in ranges)
    return out


for p in [a for a in sys.argv[1:] if not a.startswith('--')]:
    try:
        print('%s: %s' % (p, parse(p)))
    except Exception as e:  # report and go on
        print('%s: error %s' % (p, e))

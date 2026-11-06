#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""regsample.py: which memory do hot FEX-JIT guest blocks point into? [keeper, Claude Opus 5.5, 2026-11-05]  PC only, reads no log.

Joins (1) a simpleperf capture recorded with user registers (`simpleperf record -t TID -e cpu-clock -f 1000
--call-graph dwarf,64 --no-unwind`, dumped on the PC with the NDK host binary: `simpleperf dump perf.data > dump.txt`),
(2) FEX's block-naming perf map (FEX_BLOCKJITNAMING=1, /tmp/perf-<pid>.map), (3) the compat layer's memory event log
(MALI_COMPAT_MEMLOG=1, /root/mali-memlog-<comm>.tsv: M events = mapped base va + size + memory type) and optionally
(4) /proc/<pid>/maps and (5) the guest exe (for the x86 store list of each hot block).

FEX ARM64EC keeps every x86 GPR in a fixed host register (FEXCore/Source/Interface/Core/ArchHelpers/Arm64Emitter.cpp SRA,
FEX-2610): rax x8, rcx x0, rdx x1, rbx x27, rsp x23, rbp x29, rsi x25, rdi x26, r8 x2, r9 x3, r10 x4, r11 x5, r12-r15
x19-x22. So the sampled host registers are the guest registers at that point of the block.

  regsample.py --dump D.txt --map perf-P.map --memlog M.tsv [--maps maps.txt] [--tid T] [--module witcher3.exe]
               [--rva 0x1e447c0-0x1e4487e] [--roles rcx=src,rdx=dst,r8=struct] [--exe X.exe] [--top 25]
  regsample.py --selftest

Output: per role register (--roles): how many in-range samples point into a type-0 (HOST_COHERENT, CPU-uncached) mapping,
a type-1 (HOST_CACHED) mapping, a /dev/mali0 vma not in the memlog, or other memory. Without --rva: the top guest blocks
of the thread with their share, how many samples have >= 1 GPR into a type-0 mapping, and (with --exe) the block's x86
stores whose base register points into type-0 memory at the sample ("t0-stores"). Classes come from the memlog's memory
type, so type numbers are the vendor's (0 = HC uncached, 1 = HCACHED on this phone; membench-v 11-05).
"""
import bisect, collections, os, re, subprocess, sys, tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', '..', 'test'))
from jitmap import JitMap, Maps, base  # noqa: E402

X86_HOST = {'rax': 'x8', 'rcx': 'x0', 'rdx': 'x1', 'rbx': 'x27', 'rsp': 'x23', 'rbp': 'x29', 'rsi': 'x25', 'rdi': 'x26',
            'r8': 'x2', 'r9': 'x3', 'r10': 'x4', 'r11': 'x5', 'r12': 'x19', 'r13': 'x20', 'r14': 'x21', 'r15': 'x22'}
REC = re.compile(r'^record sample:')
IP = re.compile(r'^\s+ip (0x[0-9a-f]+)')
TID = re.compile(r'^\s+pid (\d+), tid (\d+)')
REG = re.compile(r'^\s+reg \((\w+)\) (0x[0-9a-f]+)')


def read_dump(text):
    """-> list of (tid, ip, {reg: value}) for sample records that carry user regs"""
    out, cur = [], None
    for ln in text.splitlines():
        if REC.match(ln):
            if cur and cur[2]:
                out.append(cur)
            cur = [None, None, {}]
            continue
        if cur is None:
            continue
        if not ln.startswith(' '):          # another record type
            if cur[2]:
                out.append(cur)
            cur = None
            continue
        m = IP.match(ln)
        if m and cur[1] is None:
            cur[1] = int(m.group(1), 16)
            continue
        m = TID.match(ln)
        if m:
            cur[0] = int(m.group(2))
            continue
        m = REG.match(ln)
        if m:
            k = m.group(1)
            if re.fullmatch(r'r\d+', k):      # NDK simpleperf dump names aarch64 x0..x30 as r0..r30
                k = 'x' + k[1:]
            cur[2][k] = int(m.group(2), 16)
    if cur and cur[2]:
        out.append(cur)
    return [tuple(c) for c in out]


class MemLog:
    """mapped ranges from M events: [va, va+size) -> set of memory types (a va reused with another type -> both)"""
    def __init__(self, text):
        rng = {}
        for ln in text.splitlines():
            f = ln.split('\t')
            if len(f) < 7 or f[1] != 'M' or f[6] in ('-', ''):
                continue
            try:
                va, size, typ = int(f[6], 16), int(f[3]), int(f[4])
            except ValueError:
                continue
            if va and size:
                rng.setdefault((va, size), set()).add(typ)
        self.r = sorted((va, va + size, types) for (va, size), types in rng.items())
        self.starts = [x[0] for x in self.r]
        self.n = len(self.r)

    def types(self, addr):
        i = bisect.bisect_right(self.starts, addr) - 1
        found = set()
        # ranges can nest only if a va was reused with another size; scan a few back
        for j in range(i, max(-1, i - 8), -1):
            a, b, t = self.r[j]
            if a <= addr < b:
                found |= t
        return found


def classify(addr, ml, maps):
    t = ml.types(addr) if ml else set()
    if len(t) > 1:
        return 'mixed-type'
    if t:
        return 'type%d' % next(iter(t))
    if maps:
        n = maps.name(addr)
        if n and ('mali0' in n or 'renderD' in n):
            return 'mali-vma-unlogged'
        if n is None:
            return 'unmapped'
        return 'other:' + (n.split('/')[-1][:24] if n.startswith('/') else n[:24])
    return 'not-in-memlog'


STORE_SKIP = re.compile(r'^(cmp|test|lea|nop|prefetch|j|call|push|pop|bt\b|ucomis|comis|clflush)')
MEMOP = re.compile(r'(-?0x[0-9a-f]+|-?\d+)?\(%(\w+)(?:,%(\w+)(?:,(\d))?)?\)$')


def block_stores(exe, imagebase, rva, maxlen=512):
    """x86 instructions of the guest block at rva whose AT&T destination (last operand) is memory with a GPR base:
    -> list of (addr, mnemonic, disp, base, index, scale); stops at the first ret / unconditional jmp"""
    a = imagebase + rva
    r = subprocess.run(['llvm-objdump', '-d', '--no-show-raw-insn', '--start-address=0x%x' % a,
                        '--stop-address=0x%x' % (a + maxlen), exe], capture_output=True, text=True)
    out = []
    for ln in r.stdout.splitlines():
        m = re.match(r'^\s*([0-9a-f]+):\s+(\S+)\s*(.*)$', ln)
        if not m:
            continue
        ia, mn, ops = int(m.group(1), 16), m.group(2), m.group(3).split('<')[0].strip()
        if mn.startswith('ret') or mn in ('jmp', 'jmpq', 'ud2', 'int3'):
            break
        if STORE_SKIP.match(mn) or not ops:
            continue
        # AT&T: the destination is the last operand; a memory operand may contain commas inside its parentheses
        if '(' in ops and ops.rstrip().endswith(')'):
            k = ops.rfind(',', 0, ops.rfind('('))
            dst = ops[k + 1:] if k >= 0 else ops
        else:
            continue
        mm = MEMOP.search(dst.strip())
        if not mm or mm.group(2) == 'rip' or mm.group(2) not in X86_HOST:
            continue
        disp = int(mm.group(1), 0) if mm.group(1) else 0
        out.append((ia, mn, disp, mm.group(2), mm.group(3), int(mm.group(4) or 1)))
    return out


def imagebase_of(exe):
    r = subprocess.run(['llvm-objdump', '-p', exe], capture_output=True, text=True)
    m = re.search(r'ImageBase\s+([0-9a-fA-F]+)', r.stdout)
    return int(m.group(1), 16) if m else 0x140000000


def analyse(samples, jm, ml, maps, tid=None, module=None, rva=None, roles=None, exe=None, top=25, w=sys.stdout):
    if tid is not None:
        samples = [s for s in samples if s[0] == tid]
    n = len(samples)
    blocks = collections.Counter()
    t0blocks = collections.Counter()
    inrange = []
    for tid_, ip, regs in samples:
        hit = jm.lookup(ip)
        if not hit or hit[0] != 'guest':
            continue
        kind, mod, r = hit
        if module and not mod.lower().endswith(module.lower()):
            continue
        key = (base(mod), r)
        blocks[key] += 1
        if any(ml.types(regs.get(h, 0)) == {0} for h in X86_HOST.values()):
            t0blocks[key] += 1
        if rva and rva[0] <= r < rva[1]:
            inrange.append(regs)
    print('regsample: samples %d (tid %s), memlog ranges %d, guest-block samples %d' %
          (n, tid if tid is not None else 'all', ml.n, sum(blocks.values())), file=w)
    if rva:
        print('regsample: RANGE %s+0x%x-0x%x samples %d (%.1f %% of the thread)' %
              (module or '*', rva[0], rva[1], len(inrange), 100.0 * len(inrange) / max(n, 1)), file=w)
        for role, x86 in (roles or {}).items():
            c = collections.Counter(classify(r.get(X86_HOST[x86], 0), ml, maps) for r in inrange)
            tot = sum(c.values())
            print('regsample: ROLE %s (%s = %s) %s' % (role, x86, X86_HOST[x86],
                  ' '.join('%s=%d(%.1f%%)' % (k, v, 100.0 * v / max(tot, 1)) for k, v in c.most_common())), file=w)
        return
    ib = imagebase_of(exe) if exe else 0
    print('regsample: TOP blocks: share | samples | samples with a GPR into type0 | x86 stores | t0-stores per sample', file=w)
    tot_t0_store_samples = 0
    for (mod, r), c in blocks.most_common(top):
        st, t0s = '', ''
        if exe and module and mod.lower() == base(module).lower():
            stores = block_stores(exe, ib, r)
            hits = 0
            for tid_, ip, regs in samples:
                h = jm.lookup(ip)
                if not h or h[0] != 'guest' or h[2] != r:
                    continue
                k = sum(1 for (_, _, d, b, _, _) in stores if ml.types(regs.get(X86_HOST[b], 0) + d) == {0})
                hits += k
                tot_t0_store_samples += 1 if k else 0
            st, t0s = str(len(stores)), '%.2f' % (hits / max(c, 1))
        print('regsample: BLOCK %s+0x%x %.2f %% | %d | %d | %s | %s' % (mod, r, 100.0 * c / max(n, 1), c, t0blocks[(mod, r)],
              st or '-', t0s or '-'), file=w)
    t0 = sum(t0blocks.values())
    print('regsample: SUM samples in guest blocks with >= 1 GPR into a type0 mapping: %d (%.1f %% of the thread)%s' %
          (t0, 100.0 * t0 / max(n, 1), (', with >= 1 x86 store whose base is type0 (top blocks only): %d (%.1f %%)' %
                                       (tot_t0_store_samples, 100.0 * tot_t0_store_samples / max(n, 1))) if exe else ''), file=w)


def selftest():
    ok = True
    def chk(name, cond):
        nonlocal ok
        print('%s %s' % ('PASS' if cond else 'FAIL', name))
        ok &= bool(cond)
    jmtext = ('0x7000001000 400 \\??\\c:\\games\\witcher3.exe+0x1e44810 (0x7000001000)\n'
              '0x7000002000 100 \\??\\c:\\games\\witcher3.exe+0x10 (0x7000002000)\n')
    ml = ('# mali-compat memlog pid 1 comm w start_ms 0 cap 64 MiB\nt_ms\tev\tmem\tsize\ttype\theap\tva\ttid\tlive_mb\n'
          '1\tA\t0x1\t1048576\t0\t0\t0x5f00000000\t1\t1\n'
          '2\tM\t0x1\t1048576\t0\t0\t0x5f00000000\t1\t1\n'
          '3\tM\t0x2\t1048576\t1\t0\t0x5f10000000\t1\t2\n')
    maps = ('5f00000000-5f00100000 rw-s 00000000 00:05 1 /dev/mali0\n5f20000000-5f20100000 rw-s 00000000 00:05 1 /dev/mali0\n'
            '7e00000000-7e00100000 rw-p 00000000 00:00 0 [anon:scudo]\n')
    def rec(ip, tid, regs):
        s = 'record sample: type 9, misc 0x2, size 96\n  sample_type: 0x3007\n  ip 0x%x\n  pid 1, tid %d\n  user regs: abi=2\n' % (ip, tid)
        return s + ''.join('    reg (%s) 0x%016x\n' % (k, v) for k, v in regs.items()) + '  user stack: size 64 dyn_size 64\n'
    dump = 'record mmap: type 1\n  pid 1\n'
    dump += rec(0x7000001010, 5, {'x0': 0x5f00000100, 'x1': 0x7e00000040, 'x2': 0x7e00000000})   # src type0, dst anon
    dump += rec(0x7000001020, 5, {'x0': 0x5f10000100, 'x1': 0x5f20000040, 'x2': 0x7e00000000})   # src type1, dst unlogged mali
    dump += rec(0x7000002010, 5, {'x0': 0x1, 'x1': 0x2, 'x8': 0x5f00000008})                      # other block, rax -> type0
    dump += rec(0x7000001030, 6, {'x0': 0x5f00000100})                                             # other thread
    dump += 'record comm: type 3\n  pid 1\n'
    s = read_dump(dump)
    chk('dump: r<N> spelling = x<N>', read_dump(rec(0x1, 5, {'r0': 7, 'r8': 9}))[0][2] == {'x0': 7, 'x8': 9})
    chk('dump: 4 samples with regs', len(s) == 4 and s[0][2]['x0'] == 0x5f00000100 and s[1][0] == 5)
    jm, m, mp = JitMap(jmtext), MemLog(ml), Maps(maps)
    chk('memlog: 2 M ranges, types', m.n == 2 and m.types(0x5f00000100) == {0} and m.types(0x5f10000000) == {1} and not m.types(0x5f20000000))
    chk('classify', classify(0x5f00000100, m, mp) == 'type0' and classify(0x5f20000040, m, mp) == 'mali-vma-unlogged'
        and classify(0x7e00000040, m, mp).startswith('other:[anon:scudo') and classify(0x10, m, mp) == 'unmapped')
    import io
    o = io.StringIO()
    analyse(s, jm, m, mp, tid=5, module='witcher3.exe', rva=(0x1e447c0, 0x1e4487e),
            roles={'src': 'rcx', 'dst': 'rdx', 'struct': 'r8'}, w=o)
    t = o.getvalue()
    chk('range: 2 of 3 thread samples', 'samples 2 (66.7 %' in t)
    chk('role src: type0=1 type1=1', re.search(r'ROLE src \(rcx = x0\) .*type0=1\(50.0%\)', t) and 'type1=1(50.0%)' in t)
    chk('role dst: anon + unlogged mali', 'mali-vma-unlogged=1' in t and 'other:[anon:scudo]=1' in t)
    o = io.StringIO()
    analyse(s, jm, m, mp, tid=5, module='witcher3.exe', w=o)
    t = o.getvalue()
    chk('top: 2 blocks, t0 GPR samples 2', 'BLOCK witcher3.exe+0x1e44810 66.67 % | 2 | 1' in t and 'BLOCK witcher3.exe+0x10 33.33 % | 1 | 1' in t
        and 'type0 mapping: 2 (66.7 %' in t)
    exe = os.path.join(HERE, '..', 'build', 'logs', 'w3-towncpu', 'exe', 'witcher3.exe')
    if os.path.exists(exe):
        st = block_stores(exe, imagebase_of(exe), 0x1e44810)
        bases = collections.Counter(b for (_, _, _, b, _, _) in st)
        # loop body: 3 cursor write-backs to (%r8), 10 element stores to (%rdx); stops at the first ret
        chk('real exe: loop body stores r8 x3 + rdx x10 (%s)' % dict(bases), bases.get('r8') == 3 and bases.get('rdx') == 10)
    else:
        print('SKIP real exe (not on this PC)')
    print('SELFTEST', 'PASS' if ok else 'FAIL')
    return 0 if ok else 1


def main(argv):
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--dump'); p.add_argument('--map'); p.add_argument('--memlog'); p.add_argument('--maps')
    p.add_argument('--tid', type=int); p.add_argument('--module'); p.add_argument('--rva'); p.add_argument('--roles')
    p.add_argument('--exe'); p.add_argument('--top', type=int, default=25); p.add_argument('--selftest', action='store_true')
    a = p.parse_args(argv)
    if a.selftest:
        return selftest()
    if not (a.dump and a.map and a.memlog):
        p.error('--dump, --map and --memlog are needed')
    rd = lambda f: open(f, errors='replace').read()
    samples = read_dump(rd(a.dump))
    rva = tuple(int(x, 16) for x in a.rva.split('-')) if a.rva else None
    roles = dict(kv.split('=')[::-1] for kv in a.roles.split(',')) if a.roles else None
    analyse(samples, JitMap(rd(a.map)), MemLog(rd(a.memlog)), Maps(rd(a.maps)) if a.maps else None,
            a.tid, a.module, rva, roles, a.exe, a.top)
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))

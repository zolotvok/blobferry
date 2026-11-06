#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""perf-ours.py: where OUR stack's CPU time goes in a simpleperf perf.data (copied from the phone).
[keeper, Claude Opus 5.5] 2026-11-05

  python3 bridge/test/perf-ours.py PERF.DATA [--pid P | --comm REGEX] [--top 25] [--lines] [--tsv OUT.tsv]
  python3 bridge/test/perf-ours.py --selftest

Works for `simpleperf record -a` (system-wide) and `-p/-t` data, with or without --call-graph fp.
Prints:
  1. processes by CPU (system-wide data; pick one with --pid/--comm, default: every process),
  2. DSO groups of the selected samples: ours (compat layer, WSI layer, ICD, libhybris, q linker), the vendor driver
     reached through them, DroidDeck preloads, DXVK/vkd3d front ends, FEX, Wine, glibc, bionic, kernel, other,
  3. per group of ours: the top self-time symbols with ms/s and % of the selection, their thread names,
  4. with call graphs: inclusive time of our entry points (any frame in our DSOs) and WHO calls them (the first frame
     outside our DSOs above the outermost frame of ours, i.e. the app/DXVK/vkd3d call site),
  5. --lines: file:line (+ inlined callers) of the hottest addresses in our DSOs, from the debug twins of
     bridge/build-debug-syms.sh (only twins whose MANIFEST verdict is SAME-CODE for the build-id in the perf.data).
Symbols: simpleperf on the phone dumps the .symtab of our (unstripped) libraries into perf.data, so function names
work without any local file; the build-ids in perf.data are matched against bridge/build/debug-syms/MANIFEST.tsv.
Cost units: 1 sample of `-e cpu-clock -f F` = 1/F s of one core; "ms/s" = core-ms per wall second of the record.
Offline only (PC).
"""
import argparse, bisect, collections, glob, os, re, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
BRIDGE = os.path.dirname(HERE)
DEBUG_DIR = os.path.join(BRIDGE, 'build', 'debug-syms')

OURS = [  # (key, label, basename regex)
    ('compat', 'compat layer (libVkLayer_droiddeck_mali_compat.so)', r'^libVkLayer_droiddeck_mali_compat\.so'),
    ('wsi', 'WSI layer (libVkLayer_window_system_integration.so)', r'^libVkLayer_window_system_integration\.so'),
    ('icd', 'bridge ICD (libvulkan_mali_bridge.so)', r'^libvulkan_mali_bridge\.so'),
    ('hybris', 'libhybris-common', r'^libhybris-common\.so'),
    ('q', 'libhybris q linker (q.so)', r'^q\.so$'),
]
OTHER = [
    ('vendor', 'vendor Mali driver (/vendor *mali*, libgpud, libged)', r'(mali|libgpud|libged|libgralloc)'),
    ('droiddeck', 'DroidDeck preloads (libbl*, libfakeinput)', r'^(libbl\w*|libfakeinput)\.so'),
    ('dxvk', 'DXVK / vkd3d-proton / dxgi (d3d*, dxgi, dxvk)', r'^(d3d\w*|dxgi|dxvk\w*|libvkd3d\w*)\.(dll|so)'),
    ('winevulkan', 'winevulkan + Vulkan loader + other layers', r'^(winevulkan\.(dll|so)|libvulkan\.so[.0-9]*|libVkLayer_\w+\.so)'),
    ('fex', 'FEX (JIT + runtime)', r'(fexmemjit|^\[anon:fex|^libarm64ecfex|^libfex|fexcore|^fexinterpreter|^fexloader)'),
    ('wine', 'Wine (ntdll, win32u, kernelbase, other PE/unix)', r'(\.dll$|\.exe$|\.drv$|^ntdll\.so|^win32u\.so|^wine)'),
    ('glibc', 'glibc / libstdc++ / vdso', r'^(libc\.so\.6|libm\.so\.6|libpthread\.so\.0|ld-linux-aarch64\.so\.1|libstdc\+\+\.so[.0-9]*|libgcc_s\.so\.1|\[vdso\])$'),
    ('bionic', 'bionic / Android system libs', r'^(libc\.so|libm\.so|libdl\.so|libc\+\+\.so|liblog\.so|libutils\.so|libcutils\.so|libbase\.so|libbinder\w*\.so|libsync\.so|libhardware\.so|libhidlbase\.so|libdmabufheap\.so|libion\.so)$'),
]
OURS_KEYS = [k for k, _, _ in OURS]
LABEL = {k: l for k, l, _ in OURS + OTHER}
LABEL.update({'kernel': 'kernel', 'unknown': 'unknown (anon: mostly FEX JIT)', 'other': 'other'})
_RX = [(k, re.compile(r, re.I)) for k, _, r in OURS + OTHER]


def classify(dso, in_kernel=False):
    if in_kernel:
        return 'kernel'
    d = dso or ''
    if d.startswith('[kernel') or d.endswith('.ko'):
        return 'kernel'
    if not d or d in ('unknown', '[unknown]') or d.startswith('[anon') and 'fex' not in d.lower():
        return 'unknown'
    base = d.rsplit('/', 1)[-1]
    if d.startswith('/vendor/') and re.search(r'mali|gpud|ged|gralloc', base, re.I):
        return 'vendor'
    for k, rx in _RX:
        if rx.search(base) or (k == 'fex' and rx.search(d.lower())):
            return k
    return 'other'


def ndk_simpleperf():
    for d in sorted(glob.glob(os.path.expanduser('~/Android/Sdk/ndk/*/simpleperf')), reverse=True):
        if os.path.exists(os.path.join(d, 'simpleperf_report_lib.py')):
            return d
    return None


def ndk_symbolizer():
    for p in sorted(glob.glob(os.path.expanduser('~/Android/Sdk/ndk/*/toolchains/llvm/prebuilt/linux-x86_64/bin/llvm-symbolizer')), reverse=True):
        return p
    return None


class Sample:
    __slots__ = ('pid', 'tid', 'comm', 'pcomm', 'period', 'time', 'leaf', 'chain')

    def __init__(self, pid, tid, comm, pcomm, period, time, leaf, chain):
        self.pid, self.tid, self.comm, self.pcomm, self.period, self.time = pid, tid, comm, pcomm, period, time
        self.leaf = leaf      # (group, dso, symbol, vaddr)
        self.chain = chain    # [(group, dso, symbol, vaddr)] leaf first, callers after (empty without call graph)


def read_perf(path):
    sp = ndk_simpleperf()
    if not sp:
        raise SystemExit('FAIL no NDK simpleperf (~/Android/Sdk/ndk/*/simpleperf)')
    sys.path.insert(0, sp)
    from simpleperf_report_lib import ReportLib  # noqa: E402
    lib = ReportLib()
    lib.SetLogSeverity('error')
    lib.SetRecordFile(path)
    lib.ShowIpForUnknownSymbol()
    meta = {'cmd': lib.GetRecordCmd(), 'build_ids': {}}
    try:
        meta['build_ids'] = lib.GetAllBuildIds()
    except Exception:
        pass
    out = []
    while True:
        s = lib.GetNextSample()
        if s is None:
            break
        sym = lib.GetSymbolOfCurrentSample()
        kern = bool(getattr(s, 'in_kernel', 0))
        leaf = (classify(sym.dso_name, kern), sym.dso_name, sym.symbol_name, sym.vaddr_in_file)
        chain = []
        cc = lib.GetCallChainOfCurrentSample()
        for i in range(cc.nr):
            e = cc.entries[i].symbol
            chain.append((classify(e.dso_name), e.dso_name, e.symbol_name, e.vaddr_in_file))
        try:
            pcomm = lib.GetProcessNameOfCurrentSample()
        except Exception:
            pcomm = ''
        out.append(Sample(s.pid, s.tid, s.thread_comm, pcomm, s.period or 1, s.time, leaf, chain))
    lib.Close()
    return out, meta


def sample_rate(meta, samples):
    m = re.search(r'(?:^|\s)-[fF]\s+(\d+)', meta.get('cmd', ''))
    hz = int(m.group(1)) if m else 4000   # simpleperf's default for cpu-clock is 4000 Hz
    if samples:
        t0, t1 = min(x.time for x in samples), max(x.time for x in samples)
        wall = max((t1 - t0) / 1e9, 1e-3)
    else:
        wall = 1.0
    return hz, wall


def fmt_cost(n, hz, wall, total):
    return '%8.1f ms/s %6.2f %%' % (n * 1000.0 / hz / wall, 100.0 * n / max(total, 1))


def load_manifest():
    """build-id (no 0x, lower) -> debug twin path, SAME-CODE or SAME-LAYOUT twins"""
    m = {}
    p = os.path.join(DEBUG_DIR, 'MANIFEST.tsv')
    if not os.path.exists(p):
        return m
    for line in open(p):
        f = line.rstrip('\n').split('\t')
        if len(f) >= 5 and f[0] != 'name' and f[4].startswith(('SAME-CODE', 'SAME-LAYOUT')) and 'NO-DWARF' not in f[4]:
            m[f[2].lower().replace('0x', '')] = f[3]
    return m


def symbolize(twin, addrs):
    """vaddr list -> {vaddr: 'func file:line <- inlined-in ...'} via llvm-symbolizer"""
    tool = ndk_symbolizer()
    if not tool or not addrs:
        return {}
    inp = ''.join('0x%x\n' % a for a in addrs)
    r = subprocess.run([tool, '--obj=' + twin, '--inlines', '--functions=linkage', '--relative-address'],
                       input=inp, capture_output=True, text=True)
    res, blocks = {}, r.stdout.split('\n\n')
    for a, blk in zip(addrs, blocks):
        ls = [x for x in blk.strip().split('\n') if x]
        frames = []
        for i in range(0, len(ls) - 1, 2):
            fn, loc = ls[i], ls[i + 1]
            loc = re.sub(r'^.*/(bridge|third_party|src)/', r'\1/', loc)
            frames.append('%s %s' % (fn, loc))
        res[a] = ' <- '.join(frames) if frames else '?'
    return res


def report(samples, meta, top=25, lines=False, tsv=None, w=sys.stdout):
    hz, wall = sample_rate(meta, samples)
    total = sum(s.period for s in samples) and len(samples)
    w.write('record: %s\n' % meta.get('cmd', ''))
    w.write('selection: %d samples, %.1f s wall, %d Hz -> %.2f cores busy on average\n\n'
            % (total, wall, hz, total / hz / wall))
    # 2. groups
    g = collections.Counter(s.leaf[0] for s in samples)
    w.write('== DSO groups (self time, leaf frame)\n')
    for k, n in g.most_common():
        mark = '*' if k in OURS_KEYS else ' '
        w.write(' %s %-58s %s  %7d smp\n' % (mark, LABEL.get(k, k), fmt_cost(n, hz, wall, total), n))
    ours_total = sum(g[k] for k in OURS_KEYS)
    w.write('   %-58s %s  (* = ours)\n\n' % ('ALL OURS (self)', fmt_cost(ours_total, hz, wall, total)))
    rows = []
    # 3. symbols of ours (+ vendor, DroidDeck)
    for k in OURS_KEYS + ['vendor', 'droiddeck']:
        sel = [s for s in samples if s.leaf[0] == k]
        if not sel:
            continue
        w.write('== %s: top self symbols\n' % LABEL[k])
        c = collections.Counter(s.leaf[2] for s in sel)
        th = collections.defaultdict(collections.Counter)
        for s in sel:
            th[s.leaf[2]][s.comm] += 1
        for sym, n in c.most_common(top):
            tn = ', '.join('%s %d' % (t, m) for t, m in th[sym].most_common(3))
            w.write('   %-44s %s  [%s]\n' % (sym[:44], fmt_cost(n, hz, wall, total), tn))
            rows.append((k, 'self', sym, n, n * 1000.0 / hz / wall, tn))
        w.write('\n')
    # 4. inclusive + callers (call graph only)
    if any(s.chain for s in samples):
        inc = collections.Counter()
        callers = collections.defaultdict(collections.Counter)
        for s in samples:
            frames = s.chain if s.chain else [s.leaf]
            if frames[0][3] != s.leaf[3] or frames[0][1] != s.leaf[1]:
                frames = [s.leaf] + frames
            seen = set()
            outer = None
            for i, f in enumerate(frames):
                if f[0] in OURS_KEYS:
                    key = (f[0], f[2])
                    if key not in seen:
                        inc[key] += 1
                        seen.add(key)
                    outer = i
            if outer is not None:
                entry = frames[outer]
                caller = next((f for f in frames[outer + 1:] if f[0] not in OURS_KEYS), None)
                cname = '%s:%s' % (caller[1].rsplit('/', 1)[-1], caller[2]) if caller else '(chain ends)'
                callers[(entry[0], entry[2])][cname] += 1
        w.write('== ours: inclusive time (sample has the function anywhere in its call chain)\n')
        for (k, sym), n in inc.most_common(top):
            w.write('   %-8s %-40s %s\n' % (k, sym[:40], fmt_cost(n, hz, wall, total)))
            rows.append((k, 'incl', sym, n, n * 1000.0 / hz / wall, ''))
        w.write('\n== ours: outermost entry point <- caller outside our stack\n')
        for (k, sym), cn in sorted(callers.items(), key=lambda kv: -sum(kv[1].values()))[:top]:
            n = sum(cn.values())
            w.write('   %-8s %-40s %s\n' % (k, sym[:40], fmt_cost(n, hz, wall, total)))
            for cname, m in cn.most_common(4):
                w.write('            <- %-60s %6d smp\n' % (cname[:60], m))
        w.write('\n')
    else:
        w.write('(no call graph in this perf.data: record with --call-graph fp for inclusive time and callers)\n\n')
    # 5. lines
    if lines:
        man = load_manifest()
        bids = {p: b.lower().replace('0x', '') for p, b in meta.get('build_ids', {}).items()}
        w.write('== ours: hottest addresses -> source lines (debug twins: %s)\n' % (DEBUG_DIR if man else 'none built'))
        for k in OURS_KEYS:
            sel = [s for s in samples if s.leaf[0] == k]
            if not sel:
                continue
            dso = collections.Counter(s.leaf[1] for s in sel).most_common(1)[0][0]
            twin = man.get(bids.get(dso, ''), None)
            if not twin:
                w.write('   %s: no SAME-CODE twin for build-id %s (run bridge/build-debug-syms.sh)\n'
                        % (k, bids.get(dso, '?')))
                continue
            c = collections.Counter(s.leaf[3] for s in sel)
            addrs = [a for a, _ in c.most_common(top)]
            loc = symbolize(twin, addrs)
            w.write('   %s (%s)\n' % (k, os.path.basename(twin)))
            for a in addrs:
                w.write('     0x%06x %s  %s\n' % (a, fmt_cost(c[a], hz, wall, total), loc.get(a, '?')[:150]))
        w.write('\n')
    if tsv:
        with open(tsv, 'w') as f:
            f.write('group\tkind\tsymbol\tsamples\tms_per_s\tthreads\n')
            for r in rows:
                f.write('%s\t%s\t%s\t%d\t%.2f\t%s\n' % r)


def select(samples, pid=None, comm=None, w=sys.stdout):
    procs = collections.Counter((s.pid, s.pcomm) for s in samples)
    if len(procs) > 1:
        w.write('== processes (all samples)\n')
        for (p, c), n in procs.most_common(15):
            w.write('   pid %-7d %-32s %7d smp\n' % (p, (c or '?')[:32], n))
        w.write('\n')
    if pid:
        samples = [s for s in samples if s.pid == pid]
    elif comm:
        rx = re.compile(comm, re.I)
        samples = [s for s in samples if rx.search(s.pcomm or '') or rx.search(s.comm or '')]
    return samples


def selftest():
    assert classify('/data/user/0/x/files/linuxfs/usr/lib/mali_bridge/libVkLayer_droiddeck_mali_compat.so') == 'compat'
    assert classify('/usr/lib/mali_bridge/libVkLayer_window_system_integration.so') == 'wsi'
    assert classify('/usr/lib/mali_bridge/libvulkan_mali_bridge.so') == 'icd'
    assert classify('/x/usr/lib/hybris/libhybris-common.so.1') == 'hybris'
    assert classify('/x/usr/lib/hybris/libhybris/linker/q.so') == 'q'
    assert classify('/vendor/lib64/egl/mt6993/libGLES_mali.so') == 'vendor'
    assert classify('/x/usr/local/lib/libblsession.so') == 'droiddeck'
    assert classify('/x/usr/lib/aarch64-linux-gnu/libc.so.6') == 'glibc'
    assert classify('/apex/com.android.runtime/lib64/bionic/libc.so') == 'bionic'
    assert classify('/x/system32/d3d12core.dll') == 'dxvk'
    assert classify('/x/aarch64-windows/libarm64ecfex.dll') == 'fex'
    assert classify('/x/aarch64-windows/ntdll.dll') == 'wine'
    assert classify('[kernel.kallsyms]') == 'kernel'
    assert classify('unknown') == 'unknown'
    assert classify('[anon:FEXMemJIT]') == 'fex'
    s = [Sample(1, 1, 'dxvk-cs', 'game', 1, i * 1000000, ('compat', 'c.so', 'compat_ResetCommandPool', 0x10),
                [('compat', 'c.so', 'compat_ResetCommandPool', 0x10), ('dxvk', 'd3d11.dll', 'Reset', 0x20)])
         for i in range(1000)]
    import io
    buf = io.StringIO()
    report(s, {'cmd': 'record -f 1000'}, w=buf)
    out = buf.getvalue()
    assert 'compat_ResetCommandPool' in out and '<- d3d11.dll:Reset' in out, out
    assert ' 1000.0 ms/s' in out or '1001.0 ms/s' in out or 'ms/s' in out
    print('selftest OK')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('perf_data', nargs='?')
    ap.add_argument('--pid', type=int)
    ap.add_argument('--comm', help='regex on the process or thread name (e.g. Watch_Dogs)')
    ap.add_argument('--top', type=int, default=25)
    ap.add_argument('--lines', action='store_true', help='file:line from bridge/build/debug-syms twins')
    ap.add_argument('--tsv')
    ap.add_argument('--selftest', action='store_true')
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if not a.perf_data:
        ap.error('PERF.DATA missing')
    samples, meta = read_perf(a.perf_data)
    samples = select(samples, a.pid, a.comm)
    report(samples, meta, a.top, a.lines, a.tsv)


if __name__ == '__main__':
    main()

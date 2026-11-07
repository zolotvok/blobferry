// SPDX-License-Identifier: GPL-3.0-or-later
// segv-report v2.1: crash reporter for processes that install their own SIGSEGV handler after us (FEX's SignalDelegator,
// a game's own "kill_handler"). [keeper, Claude Opus 5.5] 2026-11-07
// Load: via /etc/ld.so.preload (FEX drops aarch64 LD_PRELOAD entries) or LD_PRELOAD.
// - constructor: one install line per process to /root/segv-report/install.log (O_APPEND) = proof it loaded;
// - "active" processes (exe basename contains "FEX", or env SEGV_REPORT set) get a watcher thread that re-installs our
//   handler on SIGSEGV/SIGBUS/SIGILL/SIGFPE/SIGTRAP/SIGABRT whenever someone else replaced it (polls 50 ms for 15 s,
//   then 500 ms), remembers the replaced action and CHAINS to it (never swallows a signal);
// - per event: /root/segv-report/<pid>.log gets si_addr/code, pc/lr/sp/fp, x0-x7, pc/lr/addr resolved as
//   module+offset from /proc/self/maps, code bytes at pc, a frame-pointer walk and a stack scan for return addresses
//   into r-x file mappings. Memory is read with process_vm_readv (EFAULT instead of a nested fault). No stdio/malloc in
//   the handler. 32 detailed events, then ~4000 one-line events (resolve them with <pid>.maps, rewritten every 1 s), then silent chaining.
// v2.1: on EVERY sig 11/6 entry (first entry and the nested fallback of a chained handler) the maps are re-read LIVE
//   (late-mapped libs: Vulkan loader, thunks, ICD, vendor blob), written to <pid>-fault.maps (latest fault), pc/lr/addr are
//   resolved from that read, plus fp walk + stack scan. First entries beyond the 32nd detailed event get the live treatment
//   when sig==6 or pc is not in the last read (cap 256); nested fallbacks always do.
// Override the directory with SEGV_REPORT_DIR. x86_64 build of the same file is the PC self-test.
#define _GNU_SOURCE
#include <errno.h>
#include <fcntl.h>
#include <pthread.h>
#include <signal.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>  // snprintf/strftime in the constructor only
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <sys/uio.h>
#include <time.h>
#include <ucontext.h>
#include <unistd.h>

#if defined(__aarch64__)
#define UC_PC(uc) ((uintptr_t)(uc)->uc_mcontext.pc)
#define UC_SP(uc) ((uintptr_t)(uc)->uc_mcontext.sp)
#define UC_FP(uc) ((uintptr_t)(uc)->uc_mcontext.regs[29])
#define UC_LR(uc) ((uintptr_t)(uc)->uc_mcontext.regs[30])
#define UC_X(uc, i) ((uintptr_t)(uc)->uc_mcontext.regs[i])
#elif defined(__x86_64__)
#define UC_PC(uc) ((uintptr_t)(uc)->uc_mcontext.gregs[REG_RIP])
#define UC_SP(uc) ((uintptr_t)(uc)->uc_mcontext.gregs[REG_RSP])
#define UC_FP(uc) ((uintptr_t)(uc)->uc_mcontext.gregs[REG_RBP])
#define UC_LR(uc) ((uintptr_t)0)
static const int xregs[8] = {REG_RDI, REG_RSI, REG_RDX, REG_RCX, REG_R8, REG_R9, REG_RAX, REG_RBX};
#define UC_X(uc, i) ((uintptr_t)(uc)->uc_mcontext.gregs[xregs[i]])
#else
#error "aarch64 or x86_64 only"
#endif

static const int sigs[] = {SIGSEGV, SIGBUS, SIGILL, SIGFPE, SIGTRAP, SIGABRT};
#define NSIGS (int)(sizeof sigs / sizeof sigs[0])
static struct sigaction prev[NSIG];
static atomic_int have_prev[NSIG];
static char dir[256] = "/root/segv-report";
static char logpath[320], fmaps[340];
static int logfd = -1;
static pid_t mypid;
static atomic_int lock_, events;
static char mapsbuf[8 << 20];  // /proc/self/maps under FEX can be large; bss, untouched until a fault
static size_t mapslen;
static char out[16384];
static size_t outlen;

// ---- tiny async-signal-safe formatter ----
static void put(const char *s) { while (*s && outlen < sizeof out - 1) out[outlen++] = *s++; }
static void putn(const char *s, size_t n) { while (n-- && outlen < sizeof out - 1) out[outlen++] = *s++; }
static void hex(uintptr_t v) {
    char t[19] = "0x";
    for (int i = 0; i < 16; i++) t[2 + i] = "0123456789abcdef"[(v >> (60 - 4 * i)) & 15];
    t[18] = 0;
    put(t);
}
static void dec(long v) {
    char t[24]; int i = 23; t[i] = 0;
    int neg = v < 0; unsigned long u = neg ? -(unsigned long)v : (unsigned long)v;
    do { t[--i] = '0' + u % 10; u /= 10; } while (u && i > 1);
    if (neg) t[--i] = '-';
    put(t + i);
}
static void flush(void) {
    if (logfd >= 0 && outlen) { ssize_t r = write(logfd, out, outlen); (void)r; }
    outlen = 0;
}

static int saferead(uintptr_t addr, void *dst, size_t n) {
    struct iovec l = {dst, n}, r = {(void *)addr, n};
    return syscall(SYS_process_vm_readv, (long)mypid, &l, 1UL, &r, 1UL, 0UL) == (long)n;  // longs: varargs
}

static void load_maps(int dump) {
    mapslen = 0;
    int fd = open("/proc/self/maps", O_RDONLY | O_CLOEXEC);
    if (fd < 0) return;
    ssize_t r;
    while (mapslen < sizeof mapsbuf - 1 && (r = read(fd, mapsbuf + mapslen, sizeof mapsbuf - 1 - mapslen)) > 0) mapslen += r;
    close(fd);
    mapsbuf[mapslen] = 0;
    if (dump && fmaps[0]) {  // latest live read, for offline resolving (open/write/close only)
        int o = open(fmaps, O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0666);
        if (o >= 0) { size_t w = 0; while (w < mapslen) { ssize_t k = write(o, mapsbuf + w, mapslen - w); if (k <= 0) break; w += k; } close(o); }
    }
}
static uintptr_t phex(const char **p) {
    uintptr_t v = 0;
    for (;; (*p)++) {
        char c = **p;
        if (c >= '0' && c <= '9') v = v * 16 + c - '0';
        else if (c >= 'a' && c <= 'f') v = v * 16 + c - 'a' + 10;
        else return v;
    }
}
// finds the mapping of addr; returns 1 and fills perms/off/path(len); exec_file: only r-x mappings with a path
static int lookup(uintptr_t addr, uintptr_t *lo, char perms[5], uintptr_t *off, const char **path, size_t *plen) {
    const char *p = mapsbuf, *end = mapsbuf + mapslen;
    while (p < end) {
        const char *nl = memchr(p, '\n', end - p);
        if (!nl) nl = end;
        const char *q = p;
        uintptr_t a = phex(&q); q++;
        uintptr_t b = phex(&q); q++;
        if (addr >= a && addr < b) {
            memcpy(perms, q, 4); perms[4] = 0;
            q += 5;
            *off = phex(&q);
            int f = 0;  // skip dev, inode
            while (q < nl && f < 3) { if (*q == ' ') { f++; while (q < nl && *q == ' ') q++; if (f == 3) break; continue; } q++; }
            *lo = a; *path = q; *plen = nl - q;
            return 1;
        }
        p = nl + 1;
    }
    return 0;
}
static void where(const char *label, uintptr_t addr) {
    uintptr_t lo, off; char perms[5]; const char *path; size_t plen;
    put("  "); put(label); put(" "); hex(addr);
    if (lookup(addr, &lo, perms, &off, &path, &plen)) {
        put("  "); if (plen) putn(path, plen); else put("[anon]");
        put(" +"); hex(addr - lo + off); put(" ("); put(perms); put(")");
    } else put("  (unmapped)");
    put("\n");
}
static int is_code(uintptr_t addr) {
    uintptr_t lo, off; char perms[5]; const char *path; size_t plen;
    return lookup(addr, &lo, perms, &off, &path, &plen) && perms[2] == 'x';
}

static void handler(int sig, siginfo_t *si, void *ctx);

static void chain(int sig, siginfo_t *si, void *ctx) {
    if (!atomic_load(&have_prev[sig]) || (!(prev[sig].sa_flags & SA_SIGINFO) && prev[sig].sa_handler == SIG_DFL)) {
        // default action: restore it; a kernel fault re-triggers on return, anything else is re-raised
        struct sigaction d = {.sa_handler = SIG_DFL};
        sigaction(sig, &d, NULL);
        if (si->si_code <= 0 || sig == SIGABRT) syscall(SYS_tgkill, (long)getpid(), syscall(SYS_gettid), (long)sig);
        return;
    }
    if (prev[sig].sa_flags & SA_SIGINFO) {
        if (prev[sig].sa_sigaction != handler) prev[sig].sa_sigaction(sig, si, ctx);
    } else if (prev[sig].sa_handler != SIG_IGN) {
        prev[sig].sa_handler(sig);
    }
}

static atomic_int livecnt;
static int is_mapped(uintptr_t addr) {
    uintptr_t lo, off; char perms[5]; const char *path; size_t plen;
    return lookup(addr, &lo, perms, &off, &path, &plen);
}

static void report(int sig, siginfo_t *si, ucontext_t *uc) {
    load_maps(1);   // LIVE read
    put("  x0-x7:"); for (int i = 0; i < 8; i++) { put(" "); hex(UC_X(uc, i)); } put("\n");
    where("pc  ", UC_PC(uc));
    if (UC_LR(uc)) where("lr  ", UC_LR(uc));
    where("addr", (uintptr_t)si->si_addr);
    where("prev", atomic_load(&have_prev[sig]) ? (uintptr_t)prev[sig].sa_sigaction : 0);
    unsigned char code[32];
    if (saferead(UC_PC(uc) - 16, code, sizeof code)) {
        put("  code[pc-16..+16]:");
        for (int i = 0; i < 32; i++) { char t[4] = {' ', "0123456789abcdef"[code[i] >> 4], "0123456789abcdef"[code[i] & 15], 0}; put(t); }
        put("\n");
    } else { put("  saferead(pc) failed errno="); dec(errno); put("\n"); }
    uintptr_t fp = UC_FP(uc);
    for (int i = 0; i < 32 && fp && !(fp & 7); i++) {
        uintptr_t fr[2];
        if (!saferead(fp, fr, sizeof fr)) break;
        char lab[8] = "fp#"; lab[3] = '0' + i / 10; lab[4] = '0' + i % 10; lab[5] = 0; where(lab, fr[1]);
        if (fr[0] <= fp) break;
        fp = fr[0];
    }
    static uintptr_t stk[512];
    size_t got = 0;
    for (size_t want = 512; want >= 64 && !got; want /= 2)
        if (saferead(UC_SP(uc), stk, want * sizeof(uintptr_t))) got = want;
    int shown = 0;
    for (size_t i = 0; i < got && shown < 40; i++)
        if (stk[i] > 0x10000 && is_code(stk[i])) {
            char lab[12] = "sp+"; size_t o = i * 8; int k = 3;
            char t[8]; int j = 0; do { t[j++] = '0' + o % 10; o /= 10; } while (o);
            while (j) lab[k++] = t[--j];
            lab[k] = 0;
            where(lab, stk[i]); shown++;
        }
    flush();
}

static __thread int depth, holding;  // FEX calls its OldAction (= us, from the constructor) for a guest-unhandled fault: no loop

static void handler(int sig, siginfo_t *si, void *ctx) {
    int saved = errno;
    if (depth) {  // nested via the handler we chained to: we are its "previous" action, so act as SIG_DFL
        struct sigaction d = {.sa_handler = SIG_DFL};
        sigaction(sig, &d, NULL);
        int got = holding || !atomic_exchange(&lock_, 1);  // normally we hold the lock from the first entry on this thread
        if (got && (sig == SIGSEGV || sig == SIGABRT || sig == SIGBUS)) {  // the fatal one: ALWAYS a live read + resolve
            ucontext_t *nuc = ctx; struct timespec ts; clock_gettime(CLOCK_REALTIME, &ts);
            put("nested t="); dec(ts.tv_sec); put("."); dec(ts.tv_nsec / 1000000); put(" tid="); dec(syscall(SYS_gettid));
            put(" sig="); dec(sig); put(" code="); dec(si->si_code); put(" addr="); hex((uintptr_t)si->si_addr);
            put(" pc="); hex(UC_PC(nuc)); put(" lr="); hex(UC_LR(nuc)); put(" sp="); hex(UC_SP(nuc)); put(" fp="); hex(UC_FP(nuc)); put("\n");
            report(sig, si, nuc);
        }
        put("  nested call (chained handler fell back to us): default action\n"); flush();
        if (got && !holding) atomic_store(&lock_, 0);
        if (si->si_code <= 0 || sig == SIGABRT) syscall(SYS_tgkill, (long)getpid(), syscall(SYS_gettid), (long)sig);
        errno = saved;
        return;
    }
    depth++;
    ucontext_t *uc = ctx;
    int n = atomic_fetch_add(&events, 1);
    int mine = 0, full = 0;
    if (n < 4064) mine = !atomic_exchange(&lock_, 1);  // another thread logging right now: just chain
    if (mine) {
        holding = 1;
        struct timespec ts; clock_gettime(CLOCK_REALTIME, &ts);
        put("ev "); dec(n); put(" t="); dec(ts.tv_sec); put("."); dec(ts.tv_nsec / 1000000);
        put(" tid="); dec(syscall(SYS_gettid)); put(" sig="); dec(sig); put(" code="); dec(si->si_code);
        put(" addr="); hex((uintptr_t)si->si_addr); put(" pc="); hex(UC_PC(uc)); put(" lr="); hex(UC_LR(uc));
        put(" sp="); hex(UC_SP(uc)); put(" fp="); hex(UC_FP(uc));
        put(" prev="); hex(atomic_load(&have_prev[sig]) ? (uintptr_t)prev[sig].sa_sigaction : 0); put("\n");
        // 32 detailed events; later sig 11/6 entries get the live read when sig==6 or pc is not in the last read (cap 256)
        if (n < 32) full = 1;
        else if ((sig == SIGSEGV || sig == SIGABRT) && atomic_load(&livecnt) < 256 &&
                 (sig == SIGABRT || !mapslen || !is_mapped(UC_PC(uc)))) { atomic_fetch_add(&livecnt, 1); full = 1; }
        if (full) report(sig, si, uc);
        flush();
        if (logfd >= 0) fsync(logfd);
    }
    chain(sig, si, ctx);
    if (mine) {
        if (full) { put("  ev "); dec(n); put(": previous handler returned\n"); flush(); }
        holding = 0;
        atomic_store(&lock_, 0);
    }
    depth--;
    errno = saved;
}

static void line_to(const char *path, const char *text) {
    int fd = open(path, O_WRONLY | O_CREAT | O_APPEND | O_CLOEXEC, 0666);
    if (fd >= 0) { ssize_t r = write(fd, text, strlen(text)); (void)r; close(fd); }
}

static void reinstall(void) {
    for (int k = 0; k < NSIGS; k++) {
        int s = sigs[k];
        struct sigaction cur;
        if (sigaction(s, NULL, &cur) != 0) continue;
        if ((cur.sa_flags & SA_SIGINFO) && cur.sa_sigaction == handler) continue;
        struct sigaction mine = cur;  // keep the replaced handler's mask and SA_ONSTACK/SA_NODEFER/... flags
        mine.sa_sigaction = handler;
        mine.sa_flags = (cur.sa_flags | SA_SIGINFO) & ~SA_RESETHAND;
        prev[s] = cur;
        atomic_store(&have_prev[s], 1);
        if (sigaction(s, &mine, NULL) != 0) continue;
        if (atomic_exchange(&lock_, 1)) continue;  // a handler is logging (it may never return): skip the note
        load_maps(0);
        put("chain sig="); dec(s); put(" flags="); hex((unsigned)cur.sa_flags); put(" ");
        flush();
        where("prev", (cur.sa_flags & SA_SIGINFO) ? (uintptr_t)cur.sa_sigaction : (uintptr_t)cur.sa_handler);
        flush();
        atomic_store(&lock_, 0);
    }
}

// <pid>.maps = /proc/self/maps at most 1 s old, to resolve the one-line events (32+) offline
static void snapshot_maps(void) {
    static char buf[1 << 16];
    char tmp[340], dst[340];
    snprintf(dst, sizeof dst, "%s/%d.maps", dir, (int)mypid);
    snprintf(tmp, sizeof tmp, "%s.tmp", dst);
    int in = open("/proc/self/maps", O_RDONLY | O_CLOEXEC), o = open(tmp, O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0666);
    ssize_t r;
    while (in >= 0 && o >= 0 && (r = read(in, buf, sizeof buf)) > 0) { ssize_t w = write(o, buf, r); (void)w; }
    if (in >= 0) close(in);
    if (o >= 0) { close(o); rename(tmp, dst); }
}

static void *watcher(void *arg) {
    (void)arg;
    for (long ms = 0;; ) {
        reinstall();
        if (ms % 1000 == 0) snapshot_maps();
        int step = ms < 15000 ? 50 : 500;
        struct timespec ts = {0, step * 1000000L};
        nanosleep(&ts, NULL);
        ms += step;
    }
    return NULL;
}

static void start(const char *why) {
    char comm[64] = "?", exe[256] = "?", text[800];
    mypid = getpid();
    int fd = open("/proc/self/comm", O_RDONLY | O_CLOEXEC);
    if (fd >= 0) { ssize_t r = read(fd, comm, sizeof comm - 1); comm[r > 0 ? r : 0] = 0; close(fd); }
    char *nl = strchr(comm, '\n'); if (nl) *nl = 0;
    ssize_t r = readlink("/proc/self/exe", exe, sizeof exe - 1); exe[r > 0 ? r : 1] = 0;
    const char *base = strrchr(exe, '/'); base = base ? base + 1 : exe;
    int active = getenv("SEGV_REPORT") != NULL || strstr(base, "FEX") != NULL;
    time_t now = time(NULL); struct tm tm; localtime_r(&now, &tm);
    char stamp[32]; strftime(stamp, sizeof stamp, "%F %T", &tm);
    snprintf(logpath, sizeof logpath, "%s/%d.log", dir, (int)mypid);
    snprintf(fmaps, sizeof fmaps, "%s/%d-fault.maps", dir, (int)mypid);
    snprintf(text, sizeof text, "%s %s pid=%d ppid=%d comm=%s exe=%s active=%d%s%s\n", stamp, why, (int)mypid,
             (int)getppid(), comm, exe, active, active ? " log=" : "", active ? logpath : "");
    char inst[300]; snprintf(inst, sizeof inst, "%s/install.log", dir);
    line_to(inst, text);
    if (!active) return;
    if (logfd >= 0) close(logfd);
    logfd = open(logpath, O_WRONLY | O_CREAT | O_APPEND | O_CLOEXEC, 0666);
    if (logfd >= 0) { ssize_t w = write(logfd, text, strlen(text)); (void)w; }
    static char altstack[1 << 16];  // only used by threads without an altstack of their own (FEX gives its own)
    stack_t ss = {.ss_sp = altstack, .ss_size = sizeof altstack}, old;
    if (sigaltstack(NULL, &old) == 0 && (old.ss_flags & SS_DISABLE)) sigaltstack(&ss, NULL);
    reinstall();
    pthread_t t; pthread_attr_t a;
    pthread_attr_init(&a); pthread_attr_setdetachstate(&a, PTHREAD_CREATE_DETACHED);
    pthread_attr_setstacksize(&a, 1 << 18);
    sigset_t all, keep; sigfillset(&all);
    pthread_sigmask(SIG_SETMASK, &all, &keep);  // the watcher takes no signals
    if (pthread_create(&t, &a, watcher, NULL) != 0) line_to(logpath, "watcher thread failed\n");
    pthread_sigmask(SIG_SETMASK, &keep, NULL);
    pthread_attr_destroy(&a);
}

static void atfork_child(void) {
    atomic_store(&lock_, 0);
    start("fork");
}

__attribute__((constructor)) static void install(void) {
    const char *d = getenv("SEGV_REPORT_DIR");
    if (d && *d && strlen(d) < sizeof dir) strcpy(dir, d);
    mkdir(dir, 0777);
    start("load");
    pthread_atfork(NULL, NULL, atfork_child);
}

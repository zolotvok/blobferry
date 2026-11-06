// SPDX-License-Identifier: GPL-3.0-or-later
// regsamp: sample user registers of given threads with perf_event_open (cpu-clock, PERF_SAMPLE_REGS_USER), no
// simpleperf. Why: simpleperf through simpleperf_app_runner refuses --no-unwind (uncached-regs-1915 smoke 11-05), and every
// unwinding mode strips the registers (AOSP cmd_record.cpp ReplaceRegAndStackWithCallChain). Runs inside the DroidDeck
// guest (droiddeckctl guest = the app's own uid/domain, the same one the app runner uses), so it may open perf events on
// the game's threads once security.perf_harden=0 (test/cpu-profile.sh prep). [keeper, Claude Opus 5.5] 2026-11-05
//   regsamp TID[,TID..] SECONDS FREQ OUT.txt
//   regsamp --self SECONDS FREQ OUT.txt   (forks a spinning child, x19 = 0x5f00000123 on aarch64 / r12 on x86_64, and samples it:
//                                         proves perf_event_open + user regs in this context, independent of any target's access)
// On a failed open it prints "regsamp: DIAG paranoid=<perf_event_paranoid> self_open=<ok|errno> target_errno=<n>":
// self_open ok + target EACCES = the target is not accessible (ptrace_may_access: other domain / not dumpable);
// self_open EACCES = perf events are blocked for this context altogether.
// OUT is text in `simpleperf dump` form (record sample / ip / pid, tid / user regs / reg (xN)), read by
// bridge/test/regsample.py --dump. stdout: "regsamp: ..." lines, last "regsamp: DONE rc=N samples=S lost=L".
#define _GNU_SOURCE
#include <errno.h>
#include <linux/perf_event.h>
#include <poll.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <time.h>
#include <signal.h>
#include <sys/wait.h>
#include <unistd.h>

#if defined(__aarch64__)
#define NREGS 33 // x0..x30, sp, pc (PERF_REG_ARM64_*)
static const char *regname(int i) {
    static char b[8];
    if (i == 31) return "sp";
    if (i == 32) return "pc";
    snprintf(b, sizeof b, "x%d", i);
    return b;
}
#define REGMASK ((1ull << NREGS) - 1)
#elif defined(__x86_64__)
// PERF_REG_X86_*: ax bx cx dx si di bp sp ip flags cs ss ds es fs gs, r8..r15 = 16..23 (DS..GS are 0 on x86_64)
static const char *x86names[24] = {"rax", "rbx", "rcx", "rdx", "rsi", "rdi", "rbp", "rsp", "rip", "flags", "cs", "ss",
                                   "ds", "es", "fs", "gs", "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15"};
static const char *regname(int i) { return x86names[i]; }
#define REGMASK (((1ull << 24) - 1) & ~((1ull << 12) | (1ull << 13) | (1ull << 14) | (1ull << 15)))
#else
#error "aarch64 or x86_64 only"
#endif

#define MAXT 8
#define DATA_PAGES 64

static struct perf_event_attr mkattr(int freq) {
    struct perf_event_attr a;
    memset(&a, 0, sizeof a);
    a.size = sizeof a;
    a.type = PERF_TYPE_SOFTWARE;
    a.config = PERF_COUNT_SW_CPU_CLOCK;
    a.freq = 1;
    a.sample_freq = freq;
    a.sample_type = PERF_SAMPLE_IP | PERF_SAMPLE_TID | PERF_SAMPLE_TIME | PERF_SAMPLE_REGS_USER;
    a.sample_regs_user = REGMASK;
    a.exclude_kernel = 1; // user space only: no kernel sampling permission needed
    a.exclude_hv = 1;
    a.disabled = 1;
    return a;
}

static void diag(int target_errno, int freq) {
    int par = -99;
    FILE *f = fopen("/proc/sys/kernel/perf_event_paranoid", "r");
    if (f) { if (fscanf(f, "%d", &par) != 1) par = -98; fclose(f); }
    struct perf_event_attr a = mkattr(freq);
    int fd = (int)syscall(SYS_perf_event_open, &a, 0, -1, -1, PERF_FLAG_FD_CLOEXEC);
    int se = fd < 0 ? errno : 0;
    if (fd >= 0) close(fd);
    printf("regsamp: DIAG paranoid=%d self_open=%s%s target_errno=%d\n", par, fd >= 0 ? "ok" : "failed", "", target_errno);
    if (fd < 0) printf("regsamp: DIAG self_open errno=%d (%s)\n", se, strerror(se));
}

static void spin(void) {
#if defined(__aarch64__)
    register unsigned long v __asm__("x19") = 0x5f00000123ul;
#else
    register unsigned long v __asm__("r12") = 0x5f00000123ul;
#endif
    for (;;) __asm__ volatile("" : "+r"(v));
}

static uint64_t now_ns(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (uint64_t)t.tv_sec * 1000000000ull + t.tv_nsec;
}

int main(int argc, char **argv) {
    if (argc < 5) {
        printf("regsamp: usage: %s TID[,TID..]|--self SECONDS FREQ OUT.txt\nregsamp: DONE rc=2 samples=0 lost=0\n", argv[0]);
        return 2;
    }
    int tids[MAXT], nt = 0;
    pid_t child = 0;
    if (!strcmp(argv[1], "--self")) {
        child = fork();
        if (child == 0) spin();
        if (child < 0) { printf("regsamp: FAIL fork: %s\nregsamp: DONE rc=1 samples=0 lost=0\n", strerror(errno)); return 1; }
        tids[nt++] = child;
    } else
        for (char *s = strtok(argv[1], ","); s && nt < MAXT; s = strtok(NULL, ",")) tids[nt++] = atoi(s);
    double secs = atof(argv[2]);
    int freq = atoi(argv[3]);
    FILE *out = fopen(argv[4], "w");
    if (!out) { printf("regsamp: FAIL open %s: %s\nregsamp: DONE rc=1 samples=0 lost=0\n", argv[4], strerror(errno)); return 1; }
    long pg = sysconf(_SC_PAGESIZE);
    size_t mlen = (size_t)pg * (1 + DATA_PAGES);
    int fds[MAXT];
    void *rings[MAXT];
    int nreg = __builtin_popcountll(REGMASK);
    for (int i = 0; i < nt; i++) {
        struct perf_event_attr a = mkattr(freq);
        fds[i] = (int)syscall(SYS_perf_event_open, &a, tids[i], -1, -1, PERF_FLAG_FD_CLOEXEC);
        if (fds[i] < 0) {
            int e = errno;
            printf("regsamp: FAIL perf_event_open tid %d: %s (errno %d; perf_harden=0 needed; exclude_kernel=1 exclude_hv=1)\n",
                   tids[i], strerror(e), e);
            diag(e, freq);
            if (child > 0) { kill(child, SIGKILL); waitpid(child, NULL, 0); }
            printf("regsamp: DONE rc=1 samples=0 lost=0\n");
            return 1;
        }
        rings[i] = mmap(NULL, mlen, PROT_READ | PROT_WRITE, MAP_SHARED, fds[i], 0);
        if (rings[i] == MAP_FAILED) {
            printf("regsamp: FAIL mmap tid %d: %s\nregsamp: DONE rc=1 samples=0 lost=0\n", tids[i], strerror(errno));
            return 1;
        }
    }
    printf("regsamp: START tids=%d freq=%d secs=%.1f regs=%d out=%s\n", nt, freq, secs, nreg, argv[4]);
    fprintf(out, "# regsamp: simpleperf-dump-like text, cpu-clock %d Hz, user regs\n", freq);
    for (int i = 0; i < nt; i++) ioctl(fds[i], PERF_EVENT_IOC_ENABLE, 0);
    uint64_t end = now_ns() + (uint64_t)(secs * 1e9), samples = 0, lost = 0, noregs = 0;
    int done = 0;
    while (!done) {
        if (now_ns() >= end) done = 1;
        for (int i = 0; i < nt; i++) {
            struct perf_event_mmap_page *mp = rings[i];
            uint8_t *data = (uint8_t *)rings[i] + pg;
            uint64_t size = (uint64_t)pg * DATA_PAGES;
            uint64_t head = __atomic_load_n(&mp->data_head, __ATOMIC_ACQUIRE), tail = mp->data_tail;
            while (tail < head) {
                uint8_t rec[4096];
                struct perf_event_header h;
                for (size_t k = 0; k < sizeof h; k++) ((uint8_t *)&h)[k] = data[(tail + k) % size];
                if (h.size < sizeof h || h.size > sizeof rec) { tail = head; break; }
                for (size_t k = 0; k < h.size; k++) rec[k] = data[(tail + k) % size];
                tail += h.size;
                if (h.type == PERF_RECORD_LOST) { lost += ((uint64_t *)(rec + sizeof h))[1]; continue; }
                if (h.type != PERF_RECORD_SAMPLE) continue;
                uint64_t *p = (uint64_t *)(rec + sizeof h);
                uint64_t ip = p[0];
                uint32_t pid = (uint32_t)p[1], tid = (uint32_t)(p[1] >> 32);
                uint64_t t = p[2], abi = p[3];
                samples++;
                fprintf(out, "record sample: type 9, misc 0x2, size %u\n  ip 0x%llx\n  pid %u, tid %u\n  time %llu\n", h.size,
                        (unsigned long long)ip, pid, tid, (unsigned long long)t);
                if (!abi) { noregs++; continue; }
                fprintf(out, "  user regs: abi=%llu\n", (unsigned long long)abi);
                int j = 0;
                for (int r = 0; r < 64; r++)
                    if (REGMASK & (1ull << r)) fprintf(out, "    reg (%s) 0x%016llx\n", regname(r), (unsigned long long)p[4 + j++]);
            }
            __atomic_store_n(&mp->data_tail, tail, __ATOMIC_RELEASE);
        }
        if (!done) {
            struct timespec ts = {0, 5 * 1000 * 1000};
            nanosleep(&ts, NULL);
        }
    }
    for (int i = 0; i < nt; i++) { ioctl(fds[i], PERF_EVENT_IOC_DISABLE, 0); close(fds[i]); }
    if (child > 0) { kill(child, SIGKILL); waitpid(child, NULL, 0); }
    fclose(out);
    printf("regsamp: DONE rc=0 samples=%llu lost=%llu noregs=%llu\n", (unsigned long long)samples,
           (unsigned long long)lost, (unsigned long long)noregs);
    return 0;
}

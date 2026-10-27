// LD_PRELOAD crash reporter for the guest (no gdb/strace there): prints the fault address, pc/lr/sp and a frame-pointer
// backtrace as module+offset (from /proc/self/maps), then exits 139. aarch64 only. [Claude Sonnet 5.5]
// Use: LD_PRELOAD=/usr/lib/mali_bridge/segv-report.so ./program ...
// Diagnostic only (not async-signal-safe: it uses stdio). Frames inside bionic code loaded by libhybris show as the
// vendor library's path, which is what we need to name the crashing function with `nm -D`/`llvm-addr2line` on the PC copy.
#define _GNU_SOURCE
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ucontext.h>
#include <unistd.h>

static void where(const char *label, uintptr_t addr) {
    FILE *f = fopen("/proc/self/maps", "r");
    char line[512];
    while (f && fgets(line, sizeof line, f)) {
        unsigned long lo, hi, off;
        char perms[8];
        int n = 0;
        if (sscanf(line, "%lx-%lx %7s %lx %*s %*s %n", &lo, &hi, perms, &off, &n) >= 4 && addr >= lo && addr < hi) {
            const char *path = n > 0 ? line + n : "";
            size_t len = strlen(path);
            while (len && (path[len - 1] == '\n' || path[len - 1] == ' ')) len--;
            fprintf(stderr, "  %-5s 0x%016lx  %.*s +0x%lx (%s)\n", label, (unsigned long)addr, (int)len, path,
                    (unsigned long)(addr - lo + off), perms);
            fclose(f);
            return;
        }
    }
    if (f) fclose(f);
    fprintf(stderr, "  %-5s 0x%016lx  (unmapped)\n", label, (unsigned long)addr);
}

static void handler(int sig, siginfo_t *si, void *ctx) {
    ucontext_t *uc = ctx;
    fprintf(stderr, "\nsegv-report: signal %d, fault address %p, code %d\n", sig, si->si_addr, si->si_code);
    where("pc", uc->uc_mcontext.pc);
    where("lr", uc->uc_mcontext.regs[30]);
    fprintf(stderr, "  sp 0x%016lx  x0 0x%lx x1 0x%lx x2 0x%lx x3 0x%lx\n", (unsigned long)uc->uc_mcontext.sp,
            (unsigned long)uc->uc_mcontext.regs[0], (unsigned long)uc->uc_mcontext.regs[1],
            (unsigned long)uc->uc_mcontext.regs[2], (unsigned long)uc->uc_mcontext.regs[3]);
    uintptr_t *fp = (uintptr_t *)uc->uc_mcontext.regs[29];
    for (int i = 0; i < 16 && fp && ((uintptr_t)fp & 7) == 0 && (uintptr_t)fp > 0x10000; i++) {
        char label[8];
        snprintf(label, sizeof label, "#%d", i);
        where(label, fp[1]);
        uintptr_t *next = (uintptr_t *)fp[0];
        if (next <= fp) break;
        fp = next;
    }
    fflush(stderr);
    _exit(139);
}

__attribute__((constructor)) static void install(void) {
    static char stack[1 << 16];
    stack_t ss = {.ss_sp = stack, .ss_size = sizeof stack};
    sigaltstack(&ss, NULL);
    struct sigaction sa = {.sa_sigaction = handler, .sa_flags = SA_SIGINFO | SA_ONSTACK};
    sigaction(SIGSEGV, &sa, NULL);
    sigaction(SIGBUS, &sa, NULL);
}

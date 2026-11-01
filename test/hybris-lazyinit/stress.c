// SPDX-License-Identifier: GPL-3.0-or-later
// Stress test for libhybris' lazy init of statically initialised bionic mutexes/conds/rwlocks.
// [Blobferry keeper, Claude Opus 5.5, 2026-11-01]
//
// run.sh cuts the pthread hook code out of hooks.c (old = the submodule's HEAD, new = the working tree) into
// hooks_region.c and builds this file around it, so the test runs the real hook text, not a copy.
// Each round hands NT threads one fresh zeroed "bionic" object (as libc++'s constexpr std::mutex leaves it), releases
// them together and checks what the old code got wrong:
//   mutex / trylock / rwlock: more than one thread inside the critical section at once (mutual exclusion lost)
//   cond: waiters that first-touch the same cond while it is broadcast repeatedly; a waiter that times out waited on
//         a private glibc cond nobody signals (lost wake-up)
// Objects are tested 8-byte aligned and 4-byte aligned (bionic's pthread types only need 4).
// STRESS_SLOW_US > 0 makes malloc inside the hooks sleep, like a page fault under swap pressure (the WD2 case).
// Exit 0 = no violation, 1 = violations seen.
#define _GNU_SOURCE
#include <errno.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <sys/syscall.h>
#include <limits.h>
#include <linux/futex.h>
#include "hooks_shm.h"

static int slow_us;
static void *stress_malloc(size_t n)
{
    if (slow_us)
        usleep(slow_us);
    return malloc(n);
}

/* shm stubs: the test never creates process-shared objects; keep the real range check */
int hybris_is_pointer_in_shm(void *ptr)
{
#if UINTPTR_MAX > 0xFFFFFFFFu
    const uintptr_t mask = 0xFFFFFFFFFF000000ULL;
#else
    const uintptr_t mask = 0xFF000000UL;
#endif
    return (uintptr_t) ptr >= mask && (uintptr_t) ptr <= HYBRIS_SHM_MASK_TOP;
}
void *hybris_get_shmpointer(hybris_shm_pointer_t handle) { (void) handle; return NULL; }
hybris_shm_pointer_t hybris_shm_alloc(size_t size) { (void) size; return 0; }

#define malloc(n) stress_malloc(n)
#include "hooks_region.c"
#undef malloc

#define NT_MAX 64
static int NT = 16, ROUNDS = 2000;

/* a spin barrier keeps the release tight (the race window is malloc + init) */
static volatile int bar_count, bar_gen;
static void barrier(void)
{
    int gen = __atomic_load_n(&bar_gen, __ATOMIC_ACQUIRE);
    if (__atomic_add_fetch(&bar_count, 1, __ATOMIC_ACQ_REL) == NT) {
        __atomic_store_n(&bar_count, 0, __ATOMIC_RELAXED);
        __atomic_store_n(&bar_gen, gen + 1, __ATOMIC_RELEASE);
    } else
        while (__atomic_load_n(&bar_gen, __ATOMIC_ACQUIRE) == gen)
            ;
}

enum kind { K_MUTEX, K_TRYLOCK, K_COND, K_RWLOCK };   /* rwlock last: the old code can wedge a glibc rwlock */
static enum kind kind;
static int misalign;
static unsigned char *objs;           /* ROUNDS objects, 64 bytes apart */
static unsigned char *mutexes;        /* per waiter mutex (cond test) */
static long violations, inside, progress;

static void *obj_at(int r) { return objs + (size_t) r * 64 + (misalign ? 4 : 0); }

static void critical(void)
{
    if (__atomic_fetch_add(&inside, 1, __ATOMIC_ACQ_REL) != 0)
        __atomic_fetch_add(&violations, 1, __ATOMIC_RELAXED);
    for (volatile int i = 0; i < 2000; i++)
        ;
    __atomic_fetch_sub(&inside, 1, __ATOMIC_ACQ_REL);
}

static void deadline(struct timespec *ts, int ms)
{
    clock_gettime(CLOCK_REALTIME, ts);
    ts->tv_nsec += (long) ms * 1000000;
    ts->tv_sec += ts->tv_nsec / 1000000000;
    ts->tv_nsec %= 1000000000;
}

static void *worker(void *arg)
{
    int id = (int) (intptr_t) arg;
    for (int r = 0; r < ROUNDS; r++) {
        void *o = obj_at(r);
        if (id == 0)
            __atomic_store_n(&progress, r, __ATOMIC_RELAXED);
        barrier();
        switch (kind) {
        case K_MUTEX:
            _hybris_hook_pthread_mutex_lock(o);
            critical();
            _hybris_hook_pthread_mutex_unlock(o);
            break;
        case K_TRYLOCK:
            while (_hybris_hook_pthread_mutex_trylock(o) != 0)
                ;
            critical();
            _hybris_hook_pthread_mutex_unlock(o);
            break;
        case K_RWLOCK:
            _hybris_hook_pthread_rwlock_wrlock(o);
            critical();
            _hybris_hook_pthread_rwlock_unlock(o);
            break;
        case K_COND: {
            /* id 0 broadcasts, the others wait; each waiter has its own (explicitly initialised) mutex */
            if (id == 0) {
                for (int i = 0; i < 30; i++) {
                    _hybris_hook_pthread_cond_broadcast(o);
                    usleep(2000);
                }
            } else {
                void *m = mutexes + (size_t) id * 64 + (misalign ? 4 : 0);
                struct timespec ts;
                deadline(&ts, 150);
                _hybris_hook_pthread_mutex_lock(m);
                int rc = _hybris_hook_pthread_cond_timedwait(o, m, &ts);
                _hybris_hook_pthread_mutex_unlock(m);
                if (rc == ETIMEDOUT)
                    __atomic_fetch_add(&violations, 1, __ATOMIC_RELAXED);
            }
            barrier();
            break;
        }
        }
    }
    return NULL;
}

static long run(enum kind k, int mis, int rounds)
{
    pthread_t th[NT_MAX];
    kind = k;
    misalign = mis;
    ROUNDS = rounds;
    violations = 0;
    objs = aligned_alloc(64, (size_t) rounds * 64);
    memset(objs, 0, (size_t) rounds * 64);
    mutexes = aligned_alloc(64, (size_t) NT_MAX * 64);
    memset(mutexes, 0, (size_t) NT_MAX * 64);
    if (k == K_COND)
        for (int i = 1; i < NT; i++)
            _hybris_hook_pthread_mutex_init((void *) (mutexes + (size_t) i * 64 + (mis ? 4 : 0)), NULL);
    for (int i = 0; i < NT; i++)
        pthread_create(&th[i], NULL, worker, (void *) (intptr_t) i);
    for (int i = 0; i < NT; i++)
        pthread_join(th[i], NULL);
    return violations;
}

/* the old code can wedge a glibc mutex/rwlock (a loser unlocks the winner's object); report and exit instead of hanging */
static const char *cur_label, *cur_name;
static int cur_mis, cur_rounds;
static void *watchdog(void *arg)
{
    (void) arg;
    long last = -1;
    int still = 0;
    for (;;) {
        sleep(1);
        long now = __atomic_load_n(&progress, __ATOMIC_RELAXED);
        still = now == last ? still + 1 : 0;
        last = now;
        if (still >= 8) {
            printf("%s %-26s %-9s threads=%d rounds=%d slow_us=%d violations=%ld HANG at round %ld (wedged lock)\n",
                   cur_label, cur_name, cur_mis ? "align4" : "align8", NT, cur_rounds, slow_us,
                   __atomic_load_n(&violations, __ATOMIC_RELAXED), now);
            fflush(stdout);
            _exit(3);
        }
    }
    return NULL;
}

int main(int argc, char **argv)
{
    const char *label = argc > 1 ? argv[1] : "?";
    if (getenv("STRESS_SLOW_US"))
        slow_us = atoi(getenv("STRESS_SLOW_US"));
    if (getenv("STRESS_THREADS"))
        NT = atoi(getenv("STRESS_THREADS"));
    if (NT < 2 || NT > NT_MAX)
        NT = 16;
    int rounds = getenv("STRESS_ROUNDS") ? atoi(getenv("STRESS_ROUNDS")) : 2000;
    static const char *names[] = { "mutex_lock", "mutex_trylock", "cond_timedwait+broadcast", "rwlock_wrlock" };
    long total = 0;
    int only_k = -1, only_mis = -1;   /* STRESS_CASE=<kind 0-3>:<0 align8 | 1 align4> runs one case */
    if (getenv("STRESS_CASE"))
        sscanf(getenv("STRESS_CASE"), "%d:%d", &only_k, &only_mis);
    pthread_t wd;
    cur_label = label;
    pthread_create(&wd, NULL, watchdog, NULL);
    for (int k = K_MUTEX; k <= K_RWLOCK; k++)
        for (int mis = 0; mis <= 1; mis++) {
            if ((only_k >= 0 && k != only_k) || (only_mis >= 0 && mis != only_mis))
                continue;
            int r = k == K_COND ? (rounds / 20 > 5 ? rounds / 20 : 5) : rounds;
            cur_name = names[k]; cur_mis = mis; cur_rounds = r;
            __atomic_store_n(&progress, -1, __ATOMIC_RELAXED);
            long v = run(k, mis, r);
            total += v;
            printf("%s %-26s %-9s threads=%d rounds=%d slow_us=%d violations=%ld\n", label, names[k],
                   mis ? "align4" : "align8", NT, r, slow_us, v);
            fflush(stdout);
        }
    printf("%s TOTAL violations=%ld -> %s\n", label, total, total ? "FAIL" : "PASS");
    return total ? 1 : 0;
}

// SPDX-License-Identifier: GPL-3.0-or-later
// cl_sync.h: barriers and sync (sd_proc). NOT diag only: holds the shipped FB_MASK fix and RT hooks. Included by mali_compat_layer.c.
// -----------------------------------------------------------------------------
// sync-diag (The Witcher 3 DX12 terrain flashes, docs/games/witcher3.md § "flash plan 1105" candidates 5-7)
// [keeper, Claude Opus 5.5] 2026-11-05. Diagnosis switches, not fixes; off unless one of the keys below is set, and
// then the hooks are handed out at vkGetDeviceProcAddr only (unset = not one extra instruction on any Vulkan call).
//   MALI_COMPAT_FULL_BARRIERS=1  every vkCmdPipelineBarrier/2 outside a render pass gets ALL_COMMANDS -> ALL_COMMANDS
//                                and MEMORY_READ|WRITE on every barrier (layouts and queue families kept, BY_REGION
//                                dropped) plus one global memory barrier; after vkCmdWaitEvents/2 a full barrier.
//                             =2 1 + a full barrier before every render pass instance (not before a RESUMING part)
//                                and after every one (not after a SUSPENDING part), and after every dispatch, copy,
//                                blit, resolve, fill, update, clear and query copy outside a pass. Indirect arguments
//                                written by compute are then visible to any later draw (ALL_COMMANDS includes
//                                DRAW_INDIRECT). Barriers inside a pass and between suspend/resume parts stay as
//                                they are (both would be invalid).
//   MALI_COMPAT_SUBMIT_IDLE=1|<ms>  after each successful vkQueueSubmit/2 a fence-only submit on the same queue and
//                                a wait on it (cap 250 ms for 1, else <ms>): the GPU has finished all work of that
//                                queue before the next submit is recorded. A wait that hits the cap (wait-before-
//                                signal on a timeline the other queue signals later) is counted and left behind.
//   MALI_COMPAT_GT_SPLIT_SKIP=1  K4 (control.md § K4): GPU_TIME writes no timestamps around SUSPENDING/RESUMING
//                                parts (an action command between suspend and resume is invalid); those parts are
//                                not timed. Only matters with MALI_COMPAT_GPU_TIME=1.
//   MALI_COMPAT_SYNC_CTL=<file>  arms all three and reads the mode from <file> every 250 ms, so one launch can run
//                                windows with different modes: "fb=2 si=1 k4=0" (any subset, "off" = all 0). The
//                                keys above give the start mode. Each mode change and every 10 s a counter line go to
//                                stderr and to MALI_COMPAT_SYNC_LOG (default /root/mali-sync-diag-<comm>.txt).
//                                Counters per line are deltas since the previous line (MALI_COMPAT_SYNC_REPORT_S).
//   Class mask (fb-narrow, 11-05 20:xx, after witcher-syncdiag-1915 branch G: fb=2 removed the flashes, si did not):
//   ctl "fbm=<mask>" / env MALI_COMPAT_FB_MASK=<mask> (hex 0x.. or decimal) picks single classes; fb=1 = 0x00f, fb=2 =
//   0x3ff, fb=0 = 0. Bits: 0x001 app barrier dst stage |= ALL_COMMANDS, 0x002 src stage |= ALL_COMMANDS, 0x004 access
//   masks |= MEMORY_READ|WRITE + one appended global barrier + BY_REGION dropped, 0x008 full barrier after WaitEvents,
//   0x010 before a pass, 0x020 after a pass, 0x040 after a dispatch, 0x080 after a copy/blit/resolve/fill/update,
//   0x100 after a clear-image, 0x200 after a query-pool copy. Armed in any mode: a histogram of the app's barriers
//   outside passes by stage (dst tess = TESS/GEOMETRY/PRE_RASTERIZATION, ...) in the 10-s line.
//   disp-barrier (RE Village blinking light, docs/games/resident_evil_village.md 11-14 18:33; fbm 0x040 cures it at -3 fps)
//   [game/re_village keeper, Claude Opus 5.5] 2026-11-14: MALI_COMPAT_DISP_BARRIER=<list> / ctl "db=<list>" (replaces the
//   list; "db=0" and "off" clear it). <list> = 1..16 entries "<hash8>[:a|:b|:ab]" separated by commas (blanks and ';'
//   too); hash8 = exactly 8 hex digits, the FNV-1a hash of the compute shader's app SPIR-V = the GPU_TIME "cs:" hash
//   (00000000 is refused: it stands for "unknown"); a = after (default), b = before, ab = both; each hash once. A
//   malformed list is refused as a whole and logged (ctl: the old list stays; env: no list). For vkCmdDispatch /
//   DispatchIndirect / DispatchBase outside a render pass whose bound compute pipeline has a listed hash: one global
//   memory barrier exactly as fbm 0x040 records it (sd_full) before and/or after the dispatch. Counted in "added" like
//   every sd_full, and per entry (hits = matching dispatches) at the end of the 10-s line and of the mode line, only
//   while a list is set (without one both lines are byte-identical to the layer before). With fbm 0x040 as well, the
//   'a' barrier comes on top of the 0x040 one. The key (a valid list) or MALI_COMPAT_SYNC_CTL arms the hooks and fills
//   the module / compute-pipeline hash tables (g_db_hash). Replaced lists are never freed (a recording thread may still
//   read one; ~200 bytes per ctl change); hits a thread adds to a list after it was replaced are not reported.
// -----------------------------------------------------------------------------
enum { SD_PB, SD_PB2, SD_WE, SD_WE2, SD_BCB, SD_BR, SD_ER, SD_BRP, SD_BRP2, SD_ERP, SD_ERP2, SD_D, SD_DI, SD_DB,
       SD_CB, SD_CB2, SD_CI, SD_CI2, SD_CBI, SD_CBI2, SD_CIB, SD_CIB2, SD_BL, SD_BL2, SD_RS, SD_RS2, SD_FILL, SD_UPD,
       SD_CLC, SD_CLD, SD_CQP, SD_QS, SD_QS2, SD_N };
enum { SDC_WIDEN, SDC_INPASS, SDC_EXTRA, SDC_SPLIT, SDC_SUBMITS, SDC_IDLED, SDC_TIMEOUT, SDC_IDLE_NS, SDC_FAIL, SDC_K4,
       SDC_H_BARRIERS, SDC_H_DTESS, SDC_H_DIND, SDC_H_DVTX, SDC_H_DFRAG, SDC_H_DCS, SDC_H_DXFER, SDC_H_DALL, SDC_H_SCS,
       SDC_H_SXFER, SDC_H_SFRAG, SDC_N };
struct SdDev {
    PFN_vkVoidFunction next[SD_N]; // what this layer would hand out without sync-diag (its own hook or the driver's)
    PFN_vkQueueSubmit drvSubmit;
    PFN_vkWaitForFences drvWait;
    PFN_vkResetFences drvReset;
    PFN_vkCreateFence drvCreate;
    PFN_vkDestroyFence drvDestroy;
    PFN_vkGetFenceStatus drvStatus;
};
#define SD_FENCES 128
struct SdFence { VkDevice dev; pid_t tid; VkFence f; uint8_t state; }; // state 0 free, 1 ready, 2 left behind
static struct SdFence g_sd_fences[SD_FENCES];
static pthread_mutex_t g_sd_mutex = PTHREAD_MUTEX_INITIALIZER;
static bool g_sd_armed;
static _Atomic uint32_t g_sd_fbm; // class mask, see FBM_* (fb=1 -> FBM_L1, fb=2 -> FBM_ALL)
enum { FBM_DST = 0x001, FBM_SRC = 0x002, FBM_ACC = 0x004, FBM_WAITEV = 0x008, FBM_PRE = 0x010, FBM_POST = 0x020,
       FBM_DISP = 0x040, FBM_XFER = 0x080, FBM_CLEAR = 0x100, FBM_QCOPY = 0x200, FBM_L1 = 0x00f, FBM_ALL = 0x3ff };
static uint32_t sd_fb_mask(int level) { return level <= 0 ? 0 : level == 1 ? FBM_L1 : FBM_ALL; }
static int sd_fb_level(uint32_t m) { return !m ? 0 : m == FBM_L1 ? 1 : m == FBM_ALL ? 2 : 3; } // 3 = a class subset
#define SD_FB(bit) (atomic_load_explicit(&g_sd_fbm, memory_order_relaxed) & (bit))
static _Atomic uint32_t g_sd_si_ms;
static _Atomic bool g_sd_k4;
static const char *g_sd_ctl;
static char g_sd_log[192];
static uint64_t g_sd_report_ns = 10000000000ull; // MALI_COMPAT_SYNC_REPORT_S (PC test: 1)
static _Atomic uint64_t g_sd_c[SDC_N], g_sd_idle_max_ns;
static __thread bool tl_sd_inner;
#define SD_ACC (VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT)
#define SD_ACC2 (VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT)
#define SD_CNT(i) atomic_fetch_add_explicit(&g_sd_c[i], 1, memory_order_relaxed)

static uint32_t sd_si_parse(const char *v) {
    long n = v ? strtol(v, NULL, 10) : 0;
    return n <= 0 ? 0 : n == 1 ? 250 : (uint32_t)n;
}

static void sd_line(const char *fmt, ...) { // stderr (the Proton log) and the status file
    char buf[1280]; // the 10-s line + 16 disp-barrier entries
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(buf, sizeof buf, fmt, ap);
    va_end(ap);
    fprintf(stderr, "mali-compat: %s\n", buf);
    FILE *f = g_sd_log[0] ? fopen(g_sd_log, "a") : NULL;
    if (f) { fprintf(f, "%s\n", buf); fclose(f); }
}

static void sd_stamp(char *out, size_t len) { // "HH:MM:SS epoch E.EEE" (the screenrecord stamps are epoch seconds)
    struct timespec ts;
    struct tm tm;
    clock_gettime(CLOCK_REALTIME, &ts);
    localtime_r(&ts.tv_sec, &tm);
    snprintf(out, len, "%02d:%02d:%02d epoch %lld.%03ld", tm.tm_hour, tm.tm_min, tm.tm_sec, (long long)ts.tv_sec,
             ts.tv_nsec / 1000000);
}

// disp-barrier list (see the header). Immutable once published in g_sd_db, except the hit counters.
#define SD_DB_MAX 16
enum { SD_DB_A = 1, SD_DB_B = 2 };
struct SdDb { uint32_t n; uint32_t hash[SD_DB_MAX]; uint8_t mode[SD_DB_MAX]; _Atomic uint64_t hits[SD_DB_MAX]; };
static _Atomic(struct SdDb *) g_sd_db; // NULL = no list
#define SD_DB_SEP " \t\r\n,;"

static int sd_hexval(char ch) {
    return ch >= '0' && ch <= '9' ? ch - '0' : ch >= 'a' && ch <= 'f' ? ch - 'a' + 10 : ch >= 'A' && ch <= 'F' ? ch - 'A' + 10 : -1;
}
// 1 = a list in *out, 0 = "0" (clear), -1 = refused (*why says why). s is only read.
static int sd_db_parse(const char *s, struct SdDb *out, const char **why) {
    memset(out, 0, sizeof *out);
    *why = "empty list (db=0 clears)";
    if (!s) return -1;
    size_t i = 0;
    int toks = 0, zero = 0;
    for (;;) {
        while (s[i] && strchr(SD_DB_SEP, s[i])) i++;
        if (!s[i]) break;
        const char *t = s + i;
        size_t tl = 0;
        while (s[i] && !strchr(SD_DB_SEP, s[i])) { i++; tl++; }
        toks++;
        if (tl == 1 && t[0] == '0') { zero = 1; continue; }
        *why = "entry is not <8 hex digits>[:a|:b|:ab]";
        if (tl != 8 && tl != 10 && tl != 11) return -1;
        uint32_t h = 0;
        for (int k = 0; k < 8; k++) {
            int v = sd_hexval(t[k]);
            if (v < 0) return -1;
            h = h << 4 | (uint32_t)v;
        }
        uint8_t m;
        if (tl == 8) m = SD_DB_A;
        else if (tl == 10 && t[8] == ':' && t[9] == 'a') m = SD_DB_A;
        else if (tl == 10 && t[8] == ':' && t[9] == 'b') m = SD_DB_B;
        else if (tl == 11 && t[8] == ':' && t[9] == 'a' && t[10] == 'b') m = SD_DB_A | SD_DB_B;
        else return -1;
        if (!h) { *why = "hash 00000000 (= unknown pipeline)"; return -1; }
        for (uint32_t k = 0; k < out->n; k++)
            if (out->hash[k] == h) { *why = "hash listed twice"; return -1; }
        if (out->n == SD_DB_MAX) { *why = "more than 16 entries"; return -1; }
        out->hash[out->n] = h;
        out->mode[out->n++] = m;
    }
    if (zero) {
        if (toks == 1) return 0;
        *why = "0 together with entries";
        return -1;
    }
    *why = "empty list (db=0 clears)";
    return out->n ? 1 : -1;
}
static const char *sd_db_mode(uint8_t m) { return m == (SD_DB_A | SD_DB_B) ? "ab" : m == SD_DB_B ? "b" : "a"; }
// "h:m,h:m" (cnt NULL) or "h:m N, h:m N" with the hits since prev[] (prev updated; NULL prev = since the list was set).
static void sd_db_text(struct SdDb *L, uint64_t *prev, bool cnt, char *out, size_t len) {
    size_t n = 0;
    out[0] = 0;
    for (uint32_t i = 0; L && i < L->n && n < len; i++) {
        uint64_t h = atomic_load_explicit(&L->hits[i], memory_order_relaxed), dl = h - (prev ? prev[i] : 0);
        if (prev) prev[i] = h;
        int w = cnt ? snprintf(out + n, len - n, "%s%08x:%s %llu", i ? ", " : "", L->hash[i], sd_db_mode(L->mode[i]),
                               (unsigned long long)dl)
                    : snprintf(out + n, len - n, "%s%08x:%s", i ? "," : "", L->hash[i], sd_db_mode(L->mode[i]));
        if (w < 0) break;
        n += (size_t)w;
    }
}
// Parse and publish (ctl thread / init). Returns false when refused (logged; the old list stays).
static bool sd_db_set(const char *v, const char *src) {
    struct SdDb tmp;
    const char *why;
    int r = sd_db_parse(v, &tmp, &why);
    if (r < 0) { sd_line("sync-diag db=%.300s refused (%s), %s", v ? v : "", why, src); return false; }
    struct SdDb *L = NULL;
    if (r > 0) {
        if (!(L = malloc(sizeof *L))) { sd_line("sync-diag db=%.300s refused (out of memory), %s", v, src); return false; }
        memcpy(L, &tmp, sizeof *L);
    }
    atomic_store_explicit(&g_sd_db, L, memory_order_release); // the old list is never freed (see the header)
    return true;
}

static void sd_mode_line(const char *why) {
    char st[48];
    sd_stamp(st, sizeof st);
    uint32_t m = atomic_load(&g_sd_fbm);
    struct SdDb *L = atomic_load(&g_sd_db);
    char dbt[SD_DB_MAX * 12 + 4];
    sd_db_text(L, NULL, false, dbt, sizeof dbt);
    sd_line("sync-diag mode fb %d si %u ms k4 %d at %s (%s) fbm 0x%03x%s%s", sd_fb_level(m), atomic_load(&g_sd_si_ms),
            (int)atomic_load(&g_sd_k4), st, why, m, L ? " db " : "", dbt); // fbm, db last: the parsers match up to "(why)"
}

// "fb=2 si=1 k4=0", also FULL_BARRIERS= / SUBMIT_IDLE= / GT_SPLIT_SKIP=, separated by blanks, commas or newlines.
// "db=<list>" / DISP_BARRIER=: the tokens without '=' right after it (other than "off") continue its list, so
// "db=f1dca4ed:b,71bc7191:b" survives the comma split. "off" clears the list too.
static void sd_apply(char *s) {
    char db[600];
    int in_db = 0, db_long = 0;
    for (char *t = strtok(s, SD_DB_SEP); t; t = strtok(NULL, SD_DB_SEP)) {
        char *eq = strchr(t, '=');
        if (in_db) {
            if (!eq && strcmp(t, "off")) {
                size_t n = strlen(db);
                if (n + 1 + strlen(t) >= sizeof db) db_long = 1;
                else { db[n] = ','; strcpy(db + n + 1, t); }
                continue;
            }
            if (db_long) sd_line("sync-diag db= list refused (too long), ctl file");
            else sd_db_set(db, "ctl file");
            in_db = 0;
        }
        if (!strcmp(t, "off")) {
            atomic_store(&g_sd_fbm, 0); atomic_store(&g_sd_si_ms, 0); atomic_store(&g_sd_k4, false);
            if (atomic_load(&g_sd_db)) atomic_store(&g_sd_db, NULL);
        }
        if (!eq) continue;
        *eq++ = 0;
        if (!strcmp(t, "fb") || !strcmp(t, "FULL_BARRIERS")) atomic_store(&g_sd_fbm, sd_fb_mask(atoi(eq)));
        else if (!strcmp(t, "fbm") || !strcmp(t, "FB_MASK")) atomic_store(&g_sd_fbm, (uint32_t)strtoul(eq, NULL, 0) & FBM_ALL);
        else if (!strcmp(t, "si") || !strcmp(t, "SUBMIT_IDLE")) atomic_store(&g_sd_si_ms, sd_si_parse(eq));
        else if (!strcmp(t, "k4") || !strcmp(t, "GT_SPLIT_SKIP")) atomic_store(&g_sd_k4, atoi(eq) != 0);
        else if (!strcmp(t, "db") || !strcmp(t, "DISP_BARRIER")) {
            db_long = strlen(eq) >= sizeof db;
            snprintf(db, sizeof db, "%s", db_long ? "" : eq);
            in_db = 1;
        }
    }
    if (in_db) {
        if (db_long) sd_line("sync-diag db= list refused (too long), ctl file");
        else sd_db_set(db, "ctl file");
    }
}

static void *sd_thread(void *arg) {
    (void)arg;
    struct timespec last_m = {0};
    off_t last_size = -1;
    uint64_t prev[SDC_N] = {0}, t_last = now_ns();
    struct SdDb *rep_l = NULL; // the disp-barrier list the last 10-s line reported, and its hit counts then
    uint64_t rep_prev[SD_DB_MAX] = {0};
    char dbh[SD_DB_MAX * 36 + 8]; // "h:ab <20 digits>, " per entry
    for (;;) {
        struct timespec q = {0, 250000000};
        nanosleep(&q, NULL);
        struct stat sb;
        if (g_sd_ctl && stat(g_sd_ctl, &sb) == 0 &&
            (sb.st_mtim.tv_sec != last_m.tv_sec || sb.st_mtim.tv_nsec != last_m.tv_nsec || sb.st_size != last_size)) {
            last_m = sb.st_mtim;
            last_size = sb.st_size;
            char buf[512] = {0};
            FILE *f = fopen(g_sd_ctl, "r");
            if (f) {
                size_t n = fread(buf, 1, sizeof buf - 1, f);
                buf[n] = 0;
                fclose(f);
                uint32_t fb = atomic_load(&g_sd_fbm); uint32_t si = atomic_load(&g_sd_si_ms); bool k4 = atomic_load(&g_sd_k4);
                struct SdDb *dbl = atomic_load(&g_sd_db);
                sd_apply(buf);
                if (dbl && dbl != atomic_load(&g_sd_db)) { // the replaced list's hits since the last 10-s line
                    uint64_t zero[SD_DB_MAX] = {0};
                    sd_db_text(dbl, dbl == rep_l ? rep_prev : zero, true, dbh, sizeof dbh);
                    sd_line("sync-diag db list replaced, its hits since the last line: %s", dbh);
                    if (dbl == rep_l) rep_l = NULL;
                }
                if (fb != atomic_load(&g_sd_fbm) || si != atomic_load(&g_sd_si_ms) || k4 != atomic_load(&g_sd_k4) ||
                    dbl != atomic_load(&g_sd_db))
                    sd_mode_line("ctl file");
            }
        }
        uint64_t now = now_ns();
        if (now - t_last < g_sd_report_ns) continue;
        uint64_t c[SDC_N], dl[SDC_N];
        for (int i = 0; i < SDC_N; i++) { c[i] = atomic_load(&g_sd_c[i]); dl[i] = c[i] - prev[i]; prev[i] = c[i]; }
        char st[48];
        sd_stamp(st, sizeof st);
        struct SdDb *dl_l = atomic_load(&g_sd_db);
        if (dl_l != rep_l) { rep_l = dl_l; memset(rep_prev, 0, sizeof rep_prev); }
        sd_db_text(dl_l, rep_prev, true, dbh, sizeof dbh);
        sd_line("sync-diag %.1fs %s: fb %d si %u k4 %d | barriers widened %llu, in-pass kept %llu, added %llu, "
                "split-part skips %llu | submits %llu, idled %llu, timeouts %llu, idle avg %.2f ms max %.2f ms, fails %llu"
                " | k4 untimed parts %llu | fbm 0x%03x | app barriers %llu: dst tess %llu, dst indirect %llu, dst vtx %llu,"
                " dst frag %llu, dst compute %llu, dst transfer %llu, dst all %llu, src compute %llu, src transfer %llu,"
                " src frag %llu%s%s",
                (now - t_last) / 1e9, st, sd_fb_level(atomic_load(&g_sd_fbm)), atomic_load(&g_sd_si_ms), (int)atomic_load(&g_sd_k4),
                (unsigned long long)dl[SDC_WIDEN], (unsigned long long)dl[SDC_INPASS], (unsigned long long)dl[SDC_EXTRA],
                (unsigned long long)dl[SDC_SPLIT], (unsigned long long)dl[SDC_SUBMITS], (unsigned long long)dl[SDC_IDLED],
                (unsigned long long)dl[SDC_TIMEOUT], dl[SDC_IDLED] ? dl[SDC_IDLE_NS] / 1e6 / dl[SDC_IDLED] : 0.0,
                atomic_exchange(&g_sd_idle_max_ns, 0) / 1e6, (unsigned long long)dl[SDC_FAIL],
                (unsigned long long)dl[SDC_K4], atomic_load(&g_sd_fbm), (unsigned long long)dl[SDC_H_BARRIERS],
                (unsigned long long)dl[SDC_H_DTESS], (unsigned long long)dl[SDC_H_DIND], (unsigned long long)dl[SDC_H_DVTX],
                (unsigned long long)dl[SDC_H_DFRAG], (unsigned long long)dl[SDC_H_DCS], (unsigned long long)dl[SDC_H_DXFER],
                (unsigned long long)dl[SDC_H_DALL], (unsigned long long)dl[SDC_H_SCS], (unsigned long long)dl[SDC_H_SXFER],
                (unsigned long long)dl[SDC_H_SFRAG], dl_l ? " | db hits " : "", dbh);
        t_last = now;
    }
    return NULL;
}

static void sd_init_once(void) {
    const char *fb = getenv("MALI_COMPAT_FULL_BARRIERS"), *si = getenv("MALI_COMPAT_SUBMIT_IDLE");
    const char *k4 = getenv("MALI_COMPAT_GT_SPLIT_SKIP"), *ctl = getenv("MALI_COMPAT_SYNC_CTL");
    if (ctl && !*ctl) ctl = NULL;
    atomic_store(&g_sd_fbm, fb ? sd_fb_mask(atoi(fb)) : 0);
    if (getenv("MALI_COMPAT_FB_MASK")) atomic_store(&g_sd_fbm, (uint32_t)strtoul(getenv("MALI_COMPAT_FB_MASK"), NULL, 0) & FBM_ALL);
    atomic_store(&g_sd_si_ms, sd_si_parse(si));
    atomic_store(&g_sd_k4, k4 && strcmp(k4, "0") != 0);
    // disp-barrier: unset, empty or "0" = no list (as if unset); a malformed list is refused (no list).
    const char *dbv = getenv("MALI_COMPAT_DISP_BARRIER");
    struct SdDb db_env;
    const char *db_why = NULL;
    int db_r = dbv && *dbv ? sd_db_parse(dbv, &db_env, &db_why) : 0;
    g_db_hash = ctl || db_r > 0;
    // The hooks are needed when a barrier or submit mode is asked for, or may be switched on later (ctl file);
    // GT_SPLIT_SKIP alone is a flag in the GPU timing code and needs none of them.
    g_sd_armed = ctl || atomic_load(&g_sd_fbm) || atomic_load(&g_sd_si_ms) || g_rt_pipeline || db_r > 0;
    if (db_r < 0 && !g_sd_armed)
        fprintf(stderr, "mali-compat: sync-diag db=%.300s refused (%s), env MALI_COMPAT_DISP_BARRIER; no per-hash barrier\n",
                dbv, db_why);
    if (!g_sd_armed && !atomic_load(&g_sd_k4)) return;
    g_sd_ctl = ctl;
    if (getenv("MALI_COMPAT_SYNC_REPORT_S") && atoi(getenv("MALI_COMPAT_SYNC_REPORT_S")) > 0)
        g_sd_report_ns = (uint64_t)atoi(getenv("MALI_COMPAT_SYNC_REPORT_S")) * 1000000000ull;
    const char *lp = getenv("MALI_COMPAT_SYNC_LOG");
    if (lp && *lp) snprintf(g_sd_log, sizeof g_sd_log, "%s", lp);
    else {
        char comm[32] = "unknown";
        FILE *f = fopen("/proc/self/comm", "r");
        if (f) { if (fgets(comm, sizeof comm, f)) comm[strcspn(comm, "\n")] = 0; fclose(f); }
        for (char *c = comm; *c; c++) if (*c == '/' || *c == ' ' || *c == '\\') *c = '_';
        snprintf(g_sd_log, sizeof g_sd_log, "/root/mali-sync-diag-%s.txt", comm);
    }
    sd_line("sync-diag armed: hooks %s, ctl %s, log %s", g_sd_armed ? "on" : "off (k4 only)", ctl ? ctl : "none", g_sd_log);
    if (db_r > 0) sd_db_set(dbv, "env MALI_COMPAT_DISP_BARRIER");
    else if (db_r < 0) sd_line("sync-diag db=%.300s refused (%s), env MALI_COMPAT_DISP_BARRIER", dbv, db_why);
    sd_mode_line("env");
    pthread_t t;
    if (pthread_create(&t, NULL, sd_thread, NULL) == 0) pthread_detach(t);
}
static void sd_init(void) {
    static pthread_once_t once = PTHREAD_ONCE_INIT;
    pthread_once(&once, sd_init_once);
}

static struct SdDev *sd_dev(struct DeviceData *d) { // g_sd_mutex held
    if (d->sd) return d->sd;
    struct SdDev *s = calloc(1, sizeof *s);
    if (!s) return NULL;
    #define SD_DRV(field, T, name) s->field = (T)d->get_device_proc_addr(d->device, name)
    SD_DRV(drvSubmit, PFN_vkQueueSubmit, "vkQueueSubmit");
    SD_DRV(drvWait, PFN_vkWaitForFences, "vkWaitForFences");
    SD_DRV(drvReset, PFN_vkResetFences, "vkResetFences");
    SD_DRV(drvCreate, PFN_vkCreateFence, "vkCreateFence");
    SD_DRV(drvDestroy, PFN_vkDestroyFence, "vkDestroyFence");
    SD_DRV(drvStatus, PFN_vkGetFenceStatus, "vkGetFenceStatus");
    #undef SD_DRV
    d->sd = s;
    return s;
}

static void sd_device_destroy(struct DeviceData *d) {
    if (!d || !d->sd) return;
    pthread_mutex_lock(&g_sd_mutex);
    for (int i = 0; i < SD_FENCES; i++)
        if (g_sd_fences[i].state && g_sd_fences[i].dev == d->device) {
            if (d->sd->drvDestroy) d->sd->drvDestroy(d->device, g_sd_fences[i].f, NULL);
            memset(&g_sd_fences[i], 0, sizeof g_sd_fences[i]);
        }
    free(d->sd);
    d->sd = NULL;
    pthread_mutex_unlock(&g_sd_mutex);
}

// One full barrier, outside a render pass instance only (callers check).
static void sd_full(struct DeviceData *d, VkCommandBuffer cmd) {
    VkMemoryBarrier mb = {VK_STRUCTURE_TYPE_MEMORY_BARRIER, NULL, SD_ACC, SD_ACC};
    d->fpCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 1, &mb, 0,
                            NULL, 0, NULL);
    SD_CNT(SDC_EXTRA);
}
// After an action command outside a pass (FULL_BARRIERS=2 or its class bit in fbm).
static void sd_post(struct DeviceData *d, VkCommandBuffer cmd, uint32_t bit) {
    if (!SD_FB(bit)) return;
    struct CmdBufData *c = cmd_data(cmd);
    if (c && !c->sd_pass) sd_full(d, cmd);
}
static uint32_t sd_kbit(int k) {
    return k == SD_D || k == SD_DI || k == SD_DB ? FBM_DISP : k == SD_CLC || k == SD_CLD ? FBM_CLEAR
         : k == SD_CQP ? FBM_QCOPY : FBM_XFER;
}
#define SD_NEXT(d, k, T) ((T)(d)->sd->next[k])

// Stage histogram of the app's barriers outside passes (sync2 bit values; the v1 bits are the same low bits).
#define SD_ST_TESS (VK_PIPELINE_STAGE_2_TESSELLATION_CONTROL_SHADER_BIT | VK_PIPELINE_STAGE_2_TESSELLATION_EVALUATION_SHADER_BIT | \
                    VK_PIPELINE_STAGE_2_GEOMETRY_SHADER_BIT | VK_PIPELINE_STAGE_2_PRE_RASTERIZATION_SHADERS_BIT)
#define SD_ST_VTX (VK_PIPELINE_STAGE_2_VERTEX_INPUT_BIT | VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT | \
                   VK_PIPELINE_STAGE_2_INDEX_INPUT_BIT | VK_PIPELINE_STAGE_2_VERTEX_ATTRIBUTE_INPUT_BIT)
#define SD_ST_FRAG (VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT | \
                    VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT)
#define SD_ST_XFER (VK_PIPELINE_STAGE_2_TRANSFER_BIT | VK_PIPELINE_STAGE_2_COPY_BIT | VK_PIPELINE_STAGE_2_BLIT_BIT | \
                    VK_PIPELINE_STAGE_2_RESOLVE_BIT | VK_PIPELINE_STAGE_2_CLEAR_BIT)

#ifndef VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR
#define VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR 0x00200000ULL
#endif
#ifndef VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR
#define VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR 0x00200000
#endif
#ifndef VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR
#define VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR 0x00200000ULL
#endif
#ifndef VK_ACCESS_2_ACCELERATION_STRUCTURE_WRITE_BIT_KHR
#define VK_ACCESS_2_ACCELERATION_STRUCTURE_WRITE_BIT_KHR 0x00400000ULL
#endif
#ifndef VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR
#define VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR 0x00200000
#endif
#ifndef VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR
#define VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR 0x00400000
#endif

static inline __attribute__((unused)) VkPipelineStageFlags rt_remap_stage(VkPipelineStageFlags s) {
    if (s & VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR) {
        s = (s & ~VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR) | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
    }
    return s;
}

static inline __attribute__((unused)) VkPipelineStageFlags2 rt_remap_stage2(VkPipelineStageFlags2 s) {
    if (s & VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR) {
        s = (s & ~VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR) | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    }
    return s;
}

static inline __attribute__((unused)) VkAccessFlags rt_remap_access(VkAccessFlags a) {
    if (a & VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR) {
        a |= VK_ACCESS_SHADER_READ_BIT;
    }
    if (a & VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR) {
        a |= VK_ACCESS_SHADER_WRITE_BIT;
    }
    return a;
}

static inline __attribute__((unused)) VkAccessFlags2 rt_remap_access2(VkAccessFlags2 a) {
    if (a & VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR) {
        a |= VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_READ_BIT;
    }
    if (a & VK_ACCESS_2_ACCELERATION_STRUCTURE_WRITE_BIT_KHR) {
        a |= VK_ACCESS_2_SHADER_WRITE_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
    }
    return a;
}

static void sd_hist(uint64_t src, uint64_t dst) {
    SD_CNT(SDC_H_BARRIERS);
    if (dst & SD_ST_TESS) SD_CNT(SDC_H_DTESS);
    if (dst & VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT) SD_CNT(SDC_H_DIND);
    if (dst & SD_ST_VTX) SD_CNT(SDC_H_DVTX);
    if (dst & SD_ST_FRAG) SD_CNT(SDC_H_DFRAG);
    if (dst & VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT) SD_CNT(SDC_H_DCS);
    if (dst & SD_ST_XFER) SD_CNT(SDC_H_DXFER);
    if (dst & (VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT | VK_PIPELINE_STAGE_2_ALL_GRAPHICS_BIT)) SD_CNT(SDC_H_DALL);
    if (src & VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT) SD_CNT(SDC_H_SCS);
    if (src & SD_ST_XFER) SD_CNT(SDC_H_SXFER);
    if (src & SD_ST_FRAG) SD_CNT(SDC_H_SFRAG);
}

static VKAPI_ATTR void VKAPI_CALL sd_CmdPipelineBarrier(VkCommandBuffer cmd, VkPipelineStageFlags src, VkPipelineStageFlags dst,
                                                        VkDependencyFlags dep, uint32_t nm, const VkMemoryBarrier *m,
                                                        uint32_t nb, const VkBufferMemoryBarrier *b, uint32_t ni,
                                                        const VkImageMemoryBarrier *im) {
    struct DeviceData *d = get_device_data_by_cmd(cmd);
    PFN_vkCmdPipelineBarrier next = SD_NEXT(d, SD_PB, PFN_vkCmdPipelineBarrier);
    struct CmdBufData *c = cmd_data(cmd);
    uint32_t w = SD_FB(FBM_DST | FBM_SRC | FBM_ACC);
    if (g_rt_pipeline) {
        src = rt_remap_stage(src);
        dst = rt_remap_stage(dst);
    }
    if (c && !c->sd_pass) sd_hist(src, dst);
    if (!c || c->sd_pass || (!w && !g_rt_pipeline)) {
        if (c && c->sd_pass && w) SD_CNT(SDC_INPASS);
        next(cmd, src, dst, dep, nm, m, nb, b, ni, im);
        return;
    }
    VkPipelineStageFlags s2 = src | ((w & FBM_SRC) ? VK_PIPELINE_STAGE_ALL_COMMANDS_BIT : 0);
    VkPipelineStageFlags d2 = dst | ((w & FBM_DST) ? VK_PIPELINE_STAGE_ALL_COMMANDS_BIT : 0);
    SD_CNT(SDC_WIDEN);
    if (!(w & FBM_ACC) && !g_rt_pipeline) { next(cmd, s2, d2, dep, nm, m, nb, b, ni, im); return; } // stages only
    uint32_t extra = (w & FBM_ACC) ? 1 : 0;
    VkMemoryBarrier ms[33], *mw = (nm + extra) <= 33 ? ms : malloc((nm + extra) * sizeof *mw);
    VkBufferMemoryBarrier bs[32], *bw = nb <= 32 ? bs : malloc((nb ? nb : 1) * sizeof *bw);
    VkImageMemoryBarrier is[32], *iw = ni <= 32 ? is : malloc((ni ? ni : 1) * sizeof *iw);
    if (!mw || !bw || !iw) { // out of memory: the stage widening only
        if (mw != ms) free(mw);
        if (bw != bs) free(bw);
        if (iw != is) free(iw);
        next(cmd, s2, d2, dep, nm, m, nb, b, ni, im);
        return;
    }
    for (uint32_t i = 0; i < nm; i++) {
        mw[i] = m[i];
        if (w & FBM_ACC) { mw[i].srcAccessMask |= SD_ACC; mw[i].dstAccessMask |= SD_ACC; }
        if (g_rt_pipeline) {
            mw[i].srcAccessMask = rt_remap_access(mw[i].srcAccessMask);
            mw[i].dstAccessMask = rt_remap_access(mw[i].dstAccessMask);
        }
    }
    if (extra) {
        mw[nm] = (VkMemoryBarrier){VK_STRUCTURE_TYPE_MEMORY_BARRIER, NULL, SD_ACC, SD_ACC};
    }
    for (uint32_t i = 0; i < nb; i++) {
        bw[i] = b[i];
        if (w & FBM_ACC) { bw[i].srcAccessMask |= SD_ACC; bw[i].dstAccessMask |= SD_ACC; }
        if (g_rt_pipeline) {
            bw[i].srcAccessMask = rt_remap_access(bw[i].srcAccessMask);
            bw[i].dstAccessMask = rt_remap_access(bw[i].dstAccessMask);
        }
    }
    for (uint32_t i = 0; i < ni; i++) {
        iw[i] = im[i];
        if (w & FBM_ACC) { iw[i].srcAccessMask |= SD_ACC; iw[i].dstAccessMask |= SD_ACC; }
        if (g_rt_pipeline) {
            iw[i].srcAccessMask = rt_remap_access(iw[i].srcAccessMask);
            iw[i].dstAccessMask = rt_remap_access(iw[i].dstAccessMask);
        }
    }
    next(cmd, s2, d2, (w & FBM_ACC) ? (dep & ~VK_DEPENDENCY_BY_REGION_BIT) : dep, nm + extra, mw, nb, bw, ni, iw);
    if (mw != ms) free(mw);
    if (bw != bs) free(bw);
    if (iw != is) free(iw);
}

static VKAPI_ATTR void VKAPI_CALL sd_CmdPipelineBarrier2(VkCommandBuffer cmd, const VkDependencyInfo *di) {
    struct DeviceData *d = get_device_data_by_cmd(cmd);
    PFN_vkCmdPipelineBarrier2 next = SD_NEXT(d, SD_PB2, PFN_vkCmdPipelineBarrier2);
    struct CmdBufData *c = cmd_data(cmd);
    uint32_t w = SD_FB(FBM_DST | FBM_SRC | FBM_ACC);
    if (c && !c->sd_pass && di) {
        uint64_t hs = 0, hd = 0;
        for (uint32_t i = 0; i < di->memoryBarrierCount; i++) { hs |= di->pMemoryBarriers[i].srcStageMask; hd |= di->pMemoryBarriers[i].dstStageMask; }
        for (uint32_t i = 0; i < di->bufferMemoryBarrierCount; i++) { hs |= di->pBufferMemoryBarriers[i].srcStageMask; hd |= di->pBufferMemoryBarriers[i].dstStageMask; }
        for (uint32_t i = 0; i < di->imageMemoryBarrierCount; i++) { hs |= di->pImageMemoryBarriers[i].srcStageMask; hd |= di->pImageMemoryBarriers[i].dstStageMask; }
        if (g_rt_pipeline) { hs = rt_remap_stage2(hs); hd = rt_remap_stage2(hd); }
        sd_hist(hs, hd);
    }
    if (!c || c->sd_pass || !di || (!w && !g_rt_pipeline)) {
        if (c && c->sd_pass && w) SD_CNT(SDC_INPASS);
        next(cmd, di);
        return;
    }
    uint32_t nm = di->memoryBarrierCount, nb = di->bufferMemoryBarrierCount, ni = di->imageMemoryBarrierCount;
    uint32_t extra = (w & FBM_ACC) ? 1 : 0;
    VkMemoryBarrier2 ms[33], *mw = nm < 33 ? ms : malloc((nm + 1) * sizeof *mw);
    VkBufferMemoryBarrier2 bs[32], *bw = nb <= 32 ? bs : malloc(nb * sizeof *bw);
    VkImageMemoryBarrier2 is[32], *iw = ni <= 32 ? is : malloc(ni * sizeof *iw);
    if (!mw || !bw || !iw) {
        if (mw != ms) free(mw);
        if (bw != bs) free(bw);
        if (iw != is) free(iw);
        next(cmd, di);
        return;
    }
    #define SD_W2(x) do { if (w & FBM_SRC) (x).srcStageMask |= VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT; \
                          if (w & FBM_DST) (x).dstStageMask |= VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT; \
                          if (w & FBM_ACC) { (x).srcAccessMask |= SD_ACC2; (x).dstAccessMask |= SD_ACC2; } \
                          if (g_rt_pipeline) { \
                              (x).srcStageMask = rt_remap_stage2((x).srcStageMask); \
                              (x).dstStageMask = rt_remap_stage2((x).dstStageMask); \
                              (x).srcAccessMask = rt_remap_access2((x).srcAccessMask); \
                              (x).dstAccessMask = rt_remap_access2((x).dstAccessMask); \
                          } } while (0)
    for (uint32_t i = 0; i < nm; i++) { mw[i] = di->pMemoryBarriers[i]; SD_W2(mw[i]); }
    if (extra) {
        uint64_t es = (w & FBM_SRC) ? VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT : 0, ed = (w & FBM_DST) ? VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT : 0;
        if (!es || !ed) { // the appended barrier spans what the app's barriers span
            for (uint32_t i = 0; i < nm; i++) { es |= di->pMemoryBarriers[i].srcStageMask; ed |= di->pMemoryBarriers[i].dstStageMask; }
            for (uint32_t i = 0; i < nb; i++) { es |= di->pBufferMemoryBarriers[i].srcStageMask; ed |= di->pBufferMemoryBarriers[i].dstStageMask; }
            for (uint32_t i = 0; i < ni; i++) { es |= di->pImageMemoryBarriers[i].srcStageMask; ed |= di->pImageMemoryBarriers[i].dstStageMask; }
        }
        mw[nm] = (VkMemoryBarrier2){VK_STRUCTURE_TYPE_MEMORY_BARRIER_2, NULL, es, SD_ACC2, ed, SD_ACC2};
    }
    for (uint32_t i = 0; i < nb; i++) { bw[i] = di->pBufferMemoryBarriers[i]; SD_W2(bw[i]); }
    for (uint32_t i = 0; i < ni; i++) { iw[i] = di->pImageMemoryBarriers[i]; SD_W2(iw[i]); }
    #undef SD_W2
    VkDependencyInfo wi = *di;
    if (w & FBM_ACC) wi.dependencyFlags &= ~VK_DEPENDENCY_BY_REGION_BIT;
    wi.memoryBarrierCount = nm + extra; wi.pMemoryBarriers = mw;
    wi.bufferMemoryBarrierCount = nb; wi.pBufferMemoryBarriers = bw;
    wi.imageMemoryBarrierCount = ni; wi.pImageMemoryBarriers = iw;
    next(cmd, &wi);
    SD_CNT(SDC_WIDEN);
    if (mw != ms) free(mw);
    if (bw != bs) free(bw);
    if (iw != is) free(iw);
}

static VKAPI_ATTR void VKAPI_CALL sd_CmdWaitEvents(VkCommandBuffer cmd, uint32_t n, const VkEvent *ev, VkPipelineStageFlags src,
                                                   VkPipelineStageFlags dst, uint32_t nm, const VkMemoryBarrier *m, uint32_t nb,
                                                   const VkBufferMemoryBarrier *b, uint32_t ni, const VkImageMemoryBarrier *im) {
    struct DeviceData *d = get_device_data_by_cmd(cmd);
    SD_NEXT(d, SD_WE, PFN_vkCmdWaitEvents)(cmd, n, ev, src, dst, nm, m, nb, b, ni, im);
    struct CmdBufData *c = SD_FB(FBM_WAITEV) ? cmd_data(cmd) : NULL;
    if (c && !c->sd_pass) sd_full(d, cmd);
}
static VKAPI_ATTR void VKAPI_CALL sd_CmdWaitEvents2(VkCommandBuffer cmd, uint32_t n, const VkEvent *ev, const VkDependencyInfo *di) {
    struct DeviceData *d = get_device_data_by_cmd(cmd);
    SD_NEXT(d, SD_WE2, PFN_vkCmdWaitEvents2)(cmd, n, ev, di);
    struct CmdBufData *c = SD_FB(FBM_WAITEV) ? cmd_data(cmd) : NULL;
    if (c && !c->sd_pass) sd_full(d, cmd);
}

// Pass tracking runs in every mode, so a mode switched on in the middle of a command buffer knows where it is.
static VKAPI_ATTR VkResult VKAPI_CALL sd_BeginCommandBuffer(VkCommandBuffer cmd, const VkCommandBufferBeginInfo *bi) {
    struct DeviceData *d = get_device_data_by_cmd(cmd);
    VkResult r = SD_NEXT(d, SD_BCB, PFN_vkBeginCommandBuffer)(cmd, bi);
    struct CmdBufData *c = cmd_data(cmd);
    if (c) {
        // A secondary that continues a render pass is inside it from the start. No barrier at begin: the command
        // buffer can start with a RESUMING part of a pass suspended in the previous one.
        c->sd_pass = bi && (bi->flags & VK_COMMAND_BUFFER_USAGE_RENDER_PASS_CONTINUE_BIT);
        c->sd_susp = false;
        c->db_pipe = VK_NULL_HANDLE; // disp-barrier: look the hash up again in this recording
        c->db_hash = 0;
    }
    return r;
}
static VKAPI_ATTR void VKAPI_CALL sd_CmdBeginRendering(VkCommandBuffer cmd, const VkRenderingInfo *ri) {
    struct DeviceData *d = get_device_data_by_cmd(cmd);
    struct CmdBufData *c = cmd_data(cmd);
    // With NO_SUSPEND_RESUME the parts become plain passes below this hook, so barriers around them are legal.
    bool plain = !(ri->flags & (VK_RENDERING_SUSPENDING_BIT | VK_RENDERING_RESUMING_BIT)) ||
                 (g_no_susres && ri->colorAttachmentCount <= 8);
    if (SD_FB(FBM_PRE) && c && !c->sd_pass) {
        if (plain || !(ri->flags & VK_RENDERING_RESUMING_BIT)) sd_full(d, cmd);
        else SD_CNT(SDC_SPLIT);
    }
    SD_NEXT(d, SD_BR, PFN_vkCmdBeginRendering)(cmd, ri);
    if (c) { c->sd_pass = 1; c->sd_susp = !plain && (ri->flags & VK_RENDERING_SUSPENDING_BIT); }
}
static VKAPI_ATTR void VKAPI_CALL sd_CmdEndRendering(VkCommandBuffer cmd) {
    struct DeviceData *d = get_device_data_by_cmd(cmd);
    SD_NEXT(d, SD_ER, PFN_vkCmdEndRendering)(cmd);
    struct CmdBufData *c = cmd_data(cmd);
    if (!c) return;
    c->sd_pass = 0;
    if (SD_FB(FBM_POST)) {
        if (!c->sd_susp) sd_full(d, cmd);
        else SD_CNT(SDC_SPLIT);
    }
    c->sd_susp = false;
}
static void sd_pass_begin(struct DeviceData *d, VkCommandBuffer cmd, struct CmdBufData *c) {
    if (SD_FB(FBM_PRE) && c && !c->sd_pass) sd_full(d, cmd);
}
static void sd_pass_end(struct DeviceData *d, VkCommandBuffer cmd) {
    struct CmdBufData *c = cmd_data(cmd);
    if (!c) return;
    c->sd_pass = 0;
    c->sd_susp = false;
    if (SD_FB(FBM_POST)) sd_full(d, cmd);
}
static VKAPI_ATTR void VKAPI_CALL sd_CmdBeginRenderPass(VkCommandBuffer cmd, const VkRenderPassBeginInfo *bi, VkSubpassContents sc) {
    struct DeviceData *d = get_device_data_by_cmd(cmd);
    struct CmdBufData *c = cmd_data(cmd);
    sd_pass_begin(d, cmd, c);
    SD_NEXT(d, SD_BRP, PFN_vkCmdBeginRenderPass)(cmd, bi, sc);
    if (c) c->sd_pass = 1;
}
static VKAPI_ATTR void VKAPI_CALL sd_CmdBeginRenderPass2(VkCommandBuffer cmd, const VkRenderPassBeginInfo *bi, const VkSubpassBeginInfo *si) {
    struct DeviceData *d = get_device_data_by_cmd(cmd);
    struct CmdBufData *c = cmd_data(cmd);
    sd_pass_begin(d, cmd, c);
    SD_NEXT(d, SD_BRP2, PFN_vkCmdBeginRenderPass2)(cmd, bi, si);
    if (c) c->sd_pass = 1;
}
static VKAPI_ATTR void VKAPI_CALL sd_CmdEndRenderPass(VkCommandBuffer cmd) {
    struct DeviceData *d = get_device_data_by_cmd(cmd);
    SD_NEXT(d, SD_ERP, PFN_vkCmdEndRenderPass)(cmd);
    sd_pass_end(d, cmd);
}
static VKAPI_ATTR void VKAPI_CALL sd_CmdEndRenderPass2(VkCommandBuffer cmd, const VkSubpassEndInfo *ei) {
    struct DeviceData *d = get_device_data_by_cmd(cmd);
    SD_NEXT(d, SD_ERP2, PFN_vkCmdEndRenderPass2)(cmd, ei);
    sd_pass_end(d, cmd);
}

// Action commands outside a pass: the command, then (FULL_BARRIERS=2 / its fbm class bit) a full barrier.
#define SD_POST(fn, k, PARAMS, ARGS) \
    static VKAPI_ATTR void VKAPI_CALL sd_##fn PARAMS { \
        struct DeviceData *d = get_device_data_by_cmd(cmd); \
        SD_NEXT(d, k, PFN_vk##fn) ARGS; \
        sd_post(d, cmd, sd_kbit(k)); \
    }
// Dispatches: the disp-barrier entry of the bound compute pipeline's hash (before / after), and fbm 0x040 after.
// Index into *lp of the matching entry, or -1 (no list, no tracking, inside a pass, no or unknown pipeline, not listed).
static int sd_db_match(VkCommandBuffer cmd, struct SdDb **lp) {
    struct SdDb *L = atomic_load_explicit(&g_sd_db, memory_order_acquire);
    if (!L) return -1;
    struct CmdBufData *c = cmd_data(cmd);
    if (!c || c->sd_pass || !c->compute_pipeline) return -1;
    if (c->db_pipe != c->compute_pipeline) { // once per pipeline bind change: the hash table takes a mutex
        struct HashNode h;
        c->db_pipe = c->compute_pipeline;
        c->db_hash = 0;
        if (hash_get(g_pipeline_hash, (uint64_t)c->compute_pipeline, &h))
            for (int s = 0; s < 3; s++)
                if (h.stages[s] == VK_SHADER_STAGE_COMPUTE_BIT) { c->db_hash = h.hash[s]; break; }
    }
    if (!c->db_hash) return -1;
    for (uint32_t i = 0; i < L->n; i++)
        if (L->hash[i] == c->db_hash) { *lp = L; return (int)i; }
    return -1;
}
#define SD_DISP(fn, k, PARAMS, ARGS) \
    static VKAPI_ATTR void VKAPI_CALL sd_##fn PARAMS { \
        struct DeviceData *d = get_device_data_by_cmd(cmd); \
        struct SdDb *L = NULL; \
        int m = sd_db_match(cmd, &L); \
        if (m >= 0 && (L->mode[m] & SD_DB_B)) sd_full(d, cmd); \
        SD_NEXT(d, k, PFN_vk##fn) ARGS; \
        sd_post(d, cmd, sd_kbit(k)); \
        if (m >= 0) { \
            if (L->mode[m] & SD_DB_A) sd_full(d, cmd); \
            atomic_fetch_add_explicit(&L->hits[m], 1, memory_order_relaxed); \
        } \
    }
SD_DISP(CmdDispatch, SD_D, (VkCommandBuffer cmd, uint32_t x, uint32_t y, uint32_t z), (cmd, x, y, z))
SD_DISP(CmdDispatchIndirect, SD_DI, (VkCommandBuffer cmd, VkBuffer b, VkDeviceSize o), (cmd, b, o))
SD_DISP(CmdDispatchBase, SD_DB, (VkCommandBuffer cmd, uint32_t bx, uint32_t by, uint32_t bz, uint32_t x, uint32_t y, uint32_t z),
        (cmd, bx, by, bz, x, y, z))
#undef SD_DISP
SD_POST(CmdCopyBuffer, SD_CB, (VkCommandBuffer cmd, VkBuffer s, VkBuffer t, uint32_t n, const VkBufferCopy *r), (cmd, s, t, n, r))
SD_POST(CmdCopyBuffer2, SD_CB2, (VkCommandBuffer cmd, const VkCopyBufferInfo2 *i), (cmd, i))
SD_POST(CmdCopyImage, SD_CI, (VkCommandBuffer cmd, VkImage s, VkImageLayout sl, VkImage t, VkImageLayout tl, uint32_t n,
        const VkImageCopy *r), (cmd, s, sl, t, tl, n, r))
SD_POST(CmdCopyImage2, SD_CI2, (VkCommandBuffer cmd, const VkCopyImageInfo2 *i), (cmd, i))
SD_POST(CmdCopyBufferToImage, SD_CBI, (VkCommandBuffer cmd, VkBuffer s, VkImage t, VkImageLayout tl, uint32_t n,
        const VkBufferImageCopy *r), (cmd, s, t, tl, n, r))
SD_POST(CmdCopyBufferToImage2, SD_CBI2, (VkCommandBuffer cmd, const VkCopyBufferToImageInfo2 *i), (cmd, i))
SD_POST(CmdCopyImageToBuffer, SD_CIB, (VkCommandBuffer cmd, VkImage s, VkImageLayout sl, VkBuffer t, uint32_t n,
        const VkBufferImageCopy *r), (cmd, s, sl, t, n, r))
SD_POST(CmdCopyImageToBuffer2, SD_CIB2, (VkCommandBuffer cmd, const VkCopyImageToBufferInfo2 *i), (cmd, i))
SD_POST(CmdBlitImage, SD_BL, (VkCommandBuffer cmd, VkImage s, VkImageLayout sl, VkImage t, VkImageLayout tl, uint32_t n,
        const VkImageBlit *r, VkFilter f), (cmd, s, sl, t, tl, n, r, f))
SD_POST(CmdBlitImage2, SD_BL2, (VkCommandBuffer cmd, const VkBlitImageInfo2 *i), (cmd, i))
SD_POST(CmdResolveImage, SD_RS, (VkCommandBuffer cmd, VkImage s, VkImageLayout sl, VkImage t, VkImageLayout tl, uint32_t n,
        const VkImageResolve *r), (cmd, s, sl, t, tl, n, r))
SD_POST(CmdResolveImage2, SD_RS2, (VkCommandBuffer cmd, const VkResolveImageInfo2 *i), (cmd, i))
SD_POST(CmdFillBuffer, SD_FILL, (VkCommandBuffer cmd, VkBuffer b, VkDeviceSize o, VkDeviceSize s, uint32_t v), (cmd, b, o, s, v))
SD_POST(CmdUpdateBuffer, SD_UPD, (VkCommandBuffer cmd, VkBuffer b, VkDeviceSize o, VkDeviceSize s, const void *p), (cmd, b, o, s, p))
SD_POST(CmdClearColorImage, SD_CLC, (VkCommandBuffer cmd, VkImage im, VkImageLayout l, const VkClearColorValue *v, uint32_t n,
        const VkImageSubresourceRange *r), (cmd, im, l, v, n, r))
SD_POST(CmdClearDepthStencilImage, SD_CLD, (VkCommandBuffer cmd, VkImage im, VkImageLayout l, const VkClearDepthStencilValue *v,
        uint32_t n, const VkImageSubresourceRange *r), (cmd, im, l, v, n, r))
SD_POST(CmdCopyQueryPoolResults, SD_CQP, (VkCommandBuffer cmd, VkQueryPool p, uint32_t f, uint32_t n, VkBuffer b, VkDeviceSize o,
        VkDeviceSize s, VkQueryResultFlags fl), (cmd, p, f, n, b, o, s, fl))
#undef SD_POST

// SUBMIT_IDLE: a fence per (device, thread); vkd3d submits each queue from its own thread.
static void sd_idle(struct DeviceData *d, VkQueue q) {
    uint32_t ms = atomic_load_explicit(&g_sd_si_ms, memory_order_relaxed);
    struct SdDev *s = d->sd;
    if (!ms || !s || !s->drvSubmit || !s->drvWait || !s->drvReset || !s->drvCreate || !s->drvStatus) return;
    pid_t tid = (pid_t)syscall(SYS_gettid);
    struct SdFence *slot = NULL;
    pthread_mutex_lock(&g_sd_mutex);
    for (int i = 0; i < SD_FENCES && !slot; i++)
        if (g_sd_fences[i].state == 1 && g_sd_fences[i].dev == d->device && g_sd_fences[i].tid == tid) slot = &g_sd_fences[i];
    for (int i = 0; i < SD_FENCES && !slot; i++) // a fence left behind earlier that has signaled since
        if (g_sd_fences[i].state == 2 && g_sd_fences[i].dev == d->device && g_sd_fences[i].tid == tid &&
            s->drvStatus(d->device, g_sd_fences[i].f) == VK_SUCCESS && s->drvReset(d->device, 1, &g_sd_fences[i].f) == VK_SUCCESS) {
            slot = &g_sd_fences[i];
            slot->state = 1;
        }
    for (int i = 0; i < SD_FENCES && !slot; i++)
        if (!g_sd_fences[i].state) {
            VkFenceCreateInfo fi = {VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
            if (s->drvCreate(d->device, &fi, NULL, &g_sd_fences[i].f) != VK_SUCCESS) break;
            g_sd_fences[i].dev = d->device;
            g_sd_fences[i].tid = tid;
            g_sd_fences[i].state = 1;
            slot = &g_sd_fences[i];
        }
    pthread_mutex_unlock(&g_sd_mutex);
    if (!slot) { SD_CNT(SDC_FAIL); return; }
    VkFence f = slot->f; // only this thread uses its slots
    VkSubmitInfo e = {VK_STRUCTURE_TYPE_SUBMIT_INFO};
    uint64_t t0 = now_ns();
    if (s->drvSubmit(q, 1, &e, f) != VK_SUCCESS) { SD_CNT(SDC_FAIL); return; }
    VkResult w = s->drvWait(d->device, 1, &f, VK_TRUE, (uint64_t)ms * 1000000ull);
    uint64_t dt = now_ns() - t0;
    if (w == VK_SUCCESS) {
        s->drvReset(d->device, 1, &f);
        SD_CNT(SDC_IDLED);
        atomic_fetch_add_explicit(&g_sd_c[SDC_IDLE_NS], dt, memory_order_relaxed);
        uint64_t mx = atomic_load_explicit(&g_sd_idle_max_ns, memory_order_relaxed);
        while (dt > mx && !atomic_compare_exchange_weak(&g_sd_idle_max_ns, &mx, dt)) {}
    } else {
        pthread_mutex_lock(&g_sd_mutex);
        slot->state = 2; // still pending: reused once it has signaled, never reset while in flight
        pthread_mutex_unlock(&g_sd_mutex);
        SD_CNT(w == VK_TIMEOUT ? SDC_TIMEOUT : SDC_FAIL);
    }
}
static VKAPI_ATTR VkResult VKAPI_CALL sd_QueueSubmit(VkQueue q, uint32_t n, const VkSubmitInfo *si, VkFence f) {
    struct DeviceData *d = get_device_data_by_cmd((VkCommandBuffer)q);
    VkResult r = SD_NEXT(d, SD_QS, PFN_vkQueueSubmit)(q, n, si, f);
    SD_CNT(SDC_SUBMITS);
    if (r == VK_SUCCESS) sd_idle(d, q);
    return r;
}
static VKAPI_ATTR VkResult VKAPI_CALL sd_QueueSubmit2(VkQueue q, uint32_t n, const VkSubmitInfo2 *si, VkFence f) {
    struct DeviceData *d = get_device_data_by_cmd((VkCommandBuffer)q);
    VkResult r = SD_NEXT(d, SD_QS2, PFN_vkQueueSubmit2)(q, n, si, f);
    SD_CNT(SDC_SUBMITS);
    if (r == VK_SUCCESS) sd_idle(d, q);
    return r;
}

static const struct { const char *name; uint8_t kind; PFN_vkVoidFunction fn; } g_sd_names[] = {
    {"vkCmdPipelineBarrier", SD_PB, (PFN_vkVoidFunction)sd_CmdPipelineBarrier},
    {"vkCmdPipelineBarrier2", SD_PB2, (PFN_vkVoidFunction)sd_CmdPipelineBarrier2},
    {"vkCmdPipelineBarrier2KHR", SD_PB2, (PFN_vkVoidFunction)sd_CmdPipelineBarrier2},
    {"vkCmdWaitEvents", SD_WE, (PFN_vkVoidFunction)sd_CmdWaitEvents},
    {"vkCmdWaitEvents2", SD_WE2, (PFN_vkVoidFunction)sd_CmdWaitEvents2},
    {"vkCmdWaitEvents2KHR", SD_WE2, (PFN_vkVoidFunction)sd_CmdWaitEvents2},
    {"vkBeginCommandBuffer", SD_BCB, (PFN_vkVoidFunction)sd_BeginCommandBuffer},
    {"vkCmdBeginRendering", SD_BR, (PFN_vkVoidFunction)sd_CmdBeginRendering},
    {"vkCmdBeginRenderingKHR", SD_BR, (PFN_vkVoidFunction)sd_CmdBeginRendering},
    {"vkCmdEndRendering", SD_ER, (PFN_vkVoidFunction)sd_CmdEndRendering},
    {"vkCmdEndRenderingKHR", SD_ER, (PFN_vkVoidFunction)sd_CmdEndRendering},
    {"vkCmdBeginRenderPass", SD_BRP, (PFN_vkVoidFunction)sd_CmdBeginRenderPass},
    {"vkCmdBeginRenderPass2", SD_BRP2, (PFN_vkVoidFunction)sd_CmdBeginRenderPass2},
    {"vkCmdBeginRenderPass2KHR", SD_BRP2, (PFN_vkVoidFunction)sd_CmdBeginRenderPass2},
    {"vkCmdEndRenderPass", SD_ERP, (PFN_vkVoidFunction)sd_CmdEndRenderPass},
    {"vkCmdEndRenderPass2", SD_ERP2, (PFN_vkVoidFunction)sd_CmdEndRenderPass2},
    {"vkCmdEndRenderPass2KHR", SD_ERP2, (PFN_vkVoidFunction)sd_CmdEndRenderPass2},
    {"vkCmdDispatch", SD_D, (PFN_vkVoidFunction)sd_CmdDispatch},
    {"vkCmdDispatchIndirect", SD_DI, (PFN_vkVoidFunction)sd_CmdDispatchIndirect},
    {"vkCmdDispatchBase", SD_DB, (PFN_vkVoidFunction)sd_CmdDispatchBase},
    {"vkCmdDispatchBaseKHR", SD_DB, (PFN_vkVoidFunction)sd_CmdDispatchBase},
    {"vkCmdCopyBuffer", SD_CB, (PFN_vkVoidFunction)sd_CmdCopyBuffer},
    {"vkCmdCopyBuffer2", SD_CB2, (PFN_vkVoidFunction)sd_CmdCopyBuffer2},
    {"vkCmdCopyBuffer2KHR", SD_CB2, (PFN_vkVoidFunction)sd_CmdCopyBuffer2},
    {"vkCmdCopyImage", SD_CI, (PFN_vkVoidFunction)sd_CmdCopyImage},
    {"vkCmdCopyImage2", SD_CI2, (PFN_vkVoidFunction)sd_CmdCopyImage2},
    {"vkCmdCopyImage2KHR", SD_CI2, (PFN_vkVoidFunction)sd_CmdCopyImage2},
    {"vkCmdCopyBufferToImage", SD_CBI, (PFN_vkVoidFunction)sd_CmdCopyBufferToImage},
    {"vkCmdCopyBufferToImage2", SD_CBI2, (PFN_vkVoidFunction)sd_CmdCopyBufferToImage2},
    {"vkCmdCopyBufferToImage2KHR", SD_CBI2, (PFN_vkVoidFunction)sd_CmdCopyBufferToImage2},
    {"vkCmdCopyImageToBuffer", SD_CIB, (PFN_vkVoidFunction)sd_CmdCopyImageToBuffer},
    {"vkCmdCopyImageToBuffer2", SD_CIB2, (PFN_vkVoidFunction)sd_CmdCopyImageToBuffer2},
    {"vkCmdCopyImageToBuffer2KHR", SD_CIB2, (PFN_vkVoidFunction)sd_CmdCopyImageToBuffer2},
    {"vkCmdBlitImage", SD_BL, (PFN_vkVoidFunction)sd_CmdBlitImage},
    {"vkCmdBlitImage2", SD_BL2, (PFN_vkVoidFunction)sd_CmdBlitImage2},
    {"vkCmdBlitImage2KHR", SD_BL2, (PFN_vkVoidFunction)sd_CmdBlitImage2},
    {"vkCmdResolveImage", SD_RS, (PFN_vkVoidFunction)sd_CmdResolveImage},
    {"vkCmdResolveImage2", SD_RS2, (PFN_vkVoidFunction)sd_CmdResolveImage2},
    {"vkCmdResolveImage2KHR", SD_RS2, (PFN_vkVoidFunction)sd_CmdResolveImage2},
    {"vkCmdFillBuffer", SD_FILL, (PFN_vkVoidFunction)sd_CmdFillBuffer},
    {"vkCmdUpdateBuffer", SD_UPD, (PFN_vkVoidFunction)sd_CmdUpdateBuffer},
    {"vkCmdClearColorImage", SD_CLC, (PFN_vkVoidFunction)sd_CmdClearColorImage},
    {"vkCmdClearDepthStencilImage", SD_CLD, (PFN_vkVoidFunction)sd_CmdClearDepthStencilImage},
    {"vkCmdCopyQueryPoolResults", SD_CQP, (PFN_vkVoidFunction)sd_CmdCopyQueryPoolResults},
    {"vkQueueSubmit", SD_QS, (PFN_vkVoidFunction)sd_QueueSubmit},
    {"vkQueueSubmit2", SD_QS2, (PFN_vkVoidFunction)sd_QueueSubmit2},
    {"vkQueueSubmit2KHR", SD_QS2, (PFN_vkVoidFunction)sd_QueueSubmit2},
};

EXPORT VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL vkGetDeviceProcAddr(VkDevice device, const char *pName);
// Called at the top of vkGetDeviceProcAddr while armed: the sync-diag wrapper, with the entry point the rest of
// vkGetDeviceProcAddr hands out (this layer's hook or the driver's) as its next. NULL = not ours, or the name does
// not exist below (then the normal path answers NULL too).
static PFN_vkVoidFunction sd_proc(VkDevice device, const char *pName) {
    if (strncmp(pName, "vkCmd", 5) && strncmp(pName, "vkQueueSubmit", 13) && strcmp(pName, "vkBeginCommandBuffer")) return NULL;
    uint32_t k = 0;
    while (k < sizeof g_sd_names / sizeof g_sd_names[0] && strcmp(g_sd_names[k].name, pName)) k++;
    if (k == sizeof g_sd_names / sizeof g_sd_names[0]) return NULL;
    struct DeviceData *d = get_device_data(device);
    if (!d) return NULL;
    tl_sd_inner = true;
    PFN_vkVoidFunction inner = vkGetDeviceProcAddr(device, pName);
    tl_sd_inner = false;
    if (!inner) return NULL;
    pthread_mutex_lock(&g_sd_mutex);
    struct SdDev *s = sd_dev(d);
    if (s && !s->next[g_sd_names[k].kind]) s->next[g_sd_names[k].kind] = inner; // KHR alias: the same entry point
    pthread_mutex_unlock(&g_sd_mutex);
    return s ? g_sd_names[k].fn : inner;
}


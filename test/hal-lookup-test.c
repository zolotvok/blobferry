// SPDX-License-Identifier: GPL-3.0-or-later
// PC selftest of the ICD's vendor HAL lookup (src/icd/mali_bridge_icd.c hal_find) on fake dir trees, with a fake
// hybris_dlopen that reads each file's first word: HAL = good module, OTHER = module id "other", NOHMI = no HMI symbol,
// FAIL = dlopen fails, OPENFAIL = HMI open("vk0") fails. Run: test/hal-lookup-test.sh  [Claude Opus 5.5] 2026-11-19
#include "../src/icd/mali_bridge_icd.c"
#include <errno.h>
#include <unistd.h>

static hw_device_t fake_dev;
static int fake_open(const struct hw_module_t *m, const char *id, struct hw_device_t **out) {
    if (strcmp(id, "vk0") != 0 || strcmp(m->name, "fake-openfail") == 0) return -19;
    *out = &fake_dev;
    return 0;
}
static hw_module_methods_t fake_methods = {fake_open};
static hw_module_t fake_good = {.id = "vulkan", .name = "fake", .methods = &fake_methods};
static hw_module_t fake_other = {.id = "other", .name = "fake-other", .methods = &fake_methods};
static hw_module_t fake_openfail = {.id = "vulkan", .name = "fake-openfail", .methods = &fake_methods};
static int kinds[64], nkinds;

void *hybris_dlopen(const char *filename, int flag) {
    (void)flag;
    char buf[16] = {0};
    FILE *f = fopen(filename, "r");
    if (!f) return NULL;
    size_t n = fread(buf, 1, sizeof buf - 1, f);
    fclose(f);
    buf[n] = 0;
    if (strncmp(buf, "FAIL", 4) == 0 || nkinds == 64) return NULL;
    kinds[nkinds] = strncmp(buf, "HAL", 3) == 0 ? 1 : strncmp(buf, "OTHER", 5) == 0 ? 2 : strncmp(buf, "OPENFAIL", 8) == 0 ? 3 : 0;
    return &kinds[nkinds++];
}
void *hybris_dlsym(void *handle, const char *symbol) {
    if (strcmp(symbol, "HMI") != 0) return NULL;
    int k = *(int *)handle;
    return k == 1 ? (void *)&fake_good : k == 2 ? (void *)&fake_other : k == 3 ? (void *)&fake_openfail : NULL;
}
char *hybris_dlerror(void) { return "fake dlopen failure"; }

static char g_root[256], g_log[300];
static int g_fail, g_case;

// No rm -r: every created file is recorded and unlinked, then the fixed dirs are removed bottom-up with rmdir.
static char g_files[32][512];
static int g_nfiles;
static const char *const k_dirs[] = {"/vendor", "/vendor/lib64", "/vendor/lib64/hw", "/odm", "/odm/lib64", "/odm/lib64/hw", "/custom"};

static void put(const char *rel, const char *content) {
    char path[512];
    for (size_t i = 0; i < sizeof k_dirs / sizeof *k_dirs; i++)
        if (strncmp(rel, k_dirs[i], strlen(k_dirs[i])) == 0 && rel[strlen(k_dirs[i])] == '/') {
            snprintf(path, sizeof path, "%s%s", g_root, k_dirs[i]);
            if (mkdir(path, 0700) != 0 && errno != EEXIST) { perror(path); exit(2); }
        }
    snprintf(path, sizeof path, "%s%s", g_root, rel);
    FILE *f = fopen(path, "w");
    if (!f || g_nfiles == 32) { perror(path); exit(2); }
    fputs(content, f);
    fclose(f);
    snprintf(g_files[g_nfiles++], sizeof g_files[0], "%s", path);
}

static void reset_tree(void) {
    char path[512];
    while (g_nfiles > 0) unlink(g_files[--g_nfiles]);
    for (int i = (int)(sizeof k_dirs / sizeof *k_dirs) - 1; i >= 0; i--) {
        snprintf(path, sizeof path, "%s%s", g_root, k_dirs[i]);
        rmdir(path);
    }
    unsetenv("MALI_BRIDGE_HAL");
    unsetenv("MALI_BRIDGE_HAL_NAMES");
}

// Runs hal_find with stderr captured; checks the chosen path (relative to root, NULL = none) and that the log
// contains `want_log` (if given) and not `bad_log` (if given).
static void check(const char *what, const char *want_rel, const char *want_log, const char *bad_log) {
    g_case++;
    fflush(stderr);
    int saved = dup(2);
    FILE *lf = fopen(g_log, "w");
    dup2(fileno(lf), 2);
    hwvulkan_device_t *mod = hal_find(g_root);
    fflush(stderr);
    dup2(saved, 2);
    close(saved);
    fclose(lf);
    char log[8192] = {0};
    lf = fopen(g_log, "r");
    size_t n = fread(log, 1, sizeof log - 1, lf);
    fclose(lf);
    log[n] = 0;
    char want[512] = "";
    if (want_rel) snprintf(want, sizeof want, "%s%s", g_root, want_rel);
    bool ok = want_rel ? (mod && strcmp(g_hal_path, want) == 0) : (mod == NULL);
    if (want_log && !strstr(log, want_log)) ok = false;
    if (bad_log && strstr(log, bad_log)) ok = false;
    printf("%s case %d: %s\n", ok ? "PASS" : "FAIL", g_case, what);
    if (!ok) {
        g_fail++;
        printf("  chosen: %s\n  want:   %s\n  log:\n%s", mod ? g_hal_path : "(none)", want_rel ? want : "(none)", log);
    }
}

int main(void) {
    const char *tmp = getenv("TMPDIR");
    snprintf(g_root, sizeof g_root, "%s/hal-lookup-XXXXXX", tmp && *tmp ? tmp : "/tmp");
    if (!mkdtemp(g_root)) { perror("mkdtemp"); return 2; }
    snprintf(g_log, sizeof g_log, "%s/log.txt", g_root);
    const char *vivo_names = "ro.hardware.vulkan=mali,ro.board.platform=mt6993,ro.hardware=mt6993";

    // 1. vivo X300 Pro: vulkan.mali.so, ro.hardware.vulkan=mali -> the same file as before, nothing rejected.
    reset_tree();
    put("/vendor/lib64/hw/vulkan.mali.so", "HAL");
    put("/vendor/lib64/hw/vulkan.mt6993.so", "HAL");
    setenv("MALI_BRIDGE_HAL_NAMES", vivo_names, 1);
    check("vivo: ro.hardware.vulkan=mali picks vendor vulkan.mali.so first", "/vendor/lib64/hw/vulkan.mali.so",
          "lib64/hw/vulkan.mali.so (via ro.hardware.vulkan=mali)\n", "rejected");

    // 2. Old app (no MALI_BRIDGE_HAL_NAMES) on the vivo: built-in "mali", nothing rejected.
    unsetenv("MALI_BRIDGE_HAL_NAMES");
    check("vivo, no names: built-in default mali", "/vendor/lib64/hw/vulkan.mali.so", "(via built-in default)", "rejected");

    // 3. A phone that names its HAL after the platform (no vulkan.mali.so, no ro.hardware.vulkan).
    reset_tree();
    put("/vendor/lib64/hw/vulkan.mt6993.so", "HAL");
    setenv("MALI_BRIDGE_HAL_NAMES", "ro.board.platform=mt6993,ro.hardware=mt6993", 1);
    check("platform-named HAL", "/vendor/lib64/hw/vulkan.mt6993.so", "(via ro.board.platform=mt6993)", NULL);

    // 4. HAL only in /odm under ro.hardware.vulkan's name.
    reset_tree();
    put("/odm/lib64/hw/vulkan.oplus.so", "HAL");
    setenv("MALI_BRIDGE_HAL_NAMES", "ro.hardware.vulkan=oplus,ro.board.platform=mt6993", 1);
    check("odm HAL by name", "/odm/lib64/hw/vulkan.oplus.so", "/vendor/lib64/hw/vulkan.oplus.so (via ro.hardware.vulkan=oplus): not a file", NULL);

    // 5. Unknown name, no properties: the scan finds it; NOHMI, FAIL and a wrong module id are skipped.
    reset_tree();
    put("/vendor/lib64/hw/vulkan.aaa.so", "NOHMI");
    put("/vendor/lib64/hw/vulkan.bbb.so", "FAIL");
    put("/vendor/lib64/hw/vulkan.bcd.so", "OTHER");
    put("/vendor/lib64/hw/libvulkan.ccc.so", "HAL");   // not vulkan.*.so
    put("/vendor/lib64/hw/vulkan.ddd.so", "HAL");
    check("scan skips bad files", "/vendor/lib64/hw/vulkan.ddd.so", "HMI id is \"other\"", NULL);

    // 6. Scan in /odm when /vendor holds no usable file.
    reset_tree();
    put("/vendor/lib64/hw/vulkan.x.so", "FAIL");
    put("/odm/lib64/hw/vulkan.y.so", "HAL");
    check("scan reaches odm", "/odm/lib64/hw/vulkan.y.so", "(via scan of /odm/lib64/hw)", NULL);

    // 7. MALI_BRIDGE_HAL wins when the file exists; a missing one falls through to the chain.
    reset_tree();
    put("/vendor/lib64/hw/vulkan.mali.so", "HAL");
    put("/custom/vulkan.test.so", "HAL");
    char custom[512];
    snprintf(custom, sizeof custom, "%s/custom/vulkan.test.so", g_root);
    setenv("MALI_BRIDGE_HAL", custom, 1);
    check("MALI_BRIDGE_HAL exists", "/custom/vulkan.test.so", "(via MALI_BRIDGE_HAL)", NULL);
    setenv("MALI_BRIDGE_HAL", "/nonexistent/vulkan.none.so", 1);
    check("MALI_BRIDGE_HAL missing falls back", "/vendor/lib64/hw/vulkan.mali.so", "/nonexistent/vulkan.none.so (via MALI_BRIDGE_HAL): not a file", NULL);

    // 8. Bad entries are ignored: path separators, empty, overlong, bad key; duplicates tried once.
    reset_tree();
    put("/vendor/lib64/hw/vulkan.mali.so", "HAL");
    put("/vulkan.evil.so", "HAL");
    setenv("MALI_BRIDGE_HAL_NAMES",
           "../../../evil,a/b,,=x,bad key=z,ro.x=0123456789012345678901234567890123456789012345678901234567890123456789,"
           "ro.a=nope,ro.b=nope", 1);
    check("bad names ignored, dup tried once", "/vendor/lib64/hw/vulkan.mali.so", "ignoring MALI_BRIDGE_HAL_NAMES entry \"a/b\"", "evil.so");
    check("  (dup: ro.b=nope not tried)", "/vendor/lib64/hw/vulkan.mali.so", NULL, "via ro.b=nope");

    // 9. Nothing anywhere.
    reset_tree();
    setenv("MALI_BRIDGE_HAL_NAMES", vivo_names, 1);
    check("no HAL at all", NULL, "vendor HAL: none found", NULL);

    // 10. A Vulkan HAL whose open(vk0) fails ends the lookup (no second vendor driver); vulkan..so is not a scan match.
    reset_tree();
    put("/vendor/lib64/hw/vulkan.mali.so", "OPENFAIL");
    put("/vendor/lib64/hw/vulkan.mt6993.so", "HAL");
    setenv("MALI_BRIDGE_HAL_NAMES", vivo_names, 1);
    check("open failure stops the lookup", NULL, "open(vk0) rc=-19 (left loaded); stopping the lookup", "vulkan.mt6993.so");

    reset_tree();
    put("/vendor/lib64/hw/vulkan..so", "HAL");
    check("vulkan..so is not a scan match", NULL, "vendor HAL: none found", "vulkan..so");

    // 11. More than 8 valid names: the rest is logged, not dropped silently.
    reset_tree();
    put("/vendor/lib64/hw/vulkan.mali.so", "HAL");
    setenv("MALI_BRIDGE_HAL_NAMES", "a1,a2,a3,a4,a5,a6,a7,a8,a9", 1);
    check("ninth name logged", "/vendor/lib64/hw/vulkan.mali.so", "ignoring \"a9\"", NULL);

    // 12. Scanned file under a name already tried is not tried twice.
    reset_tree();
    put("/vendor/lib64/hw/vulkan.mali.so", "FAIL");
    check("failed named file not retried by the scan", NULL, "vendor HAL: none found (2 candidates tried)", NULL);

    reset_tree();
    unlink(g_log);
    if (rmdir(g_root) != 0) perror(g_root);
    printf("%s: %d failed\n", g_fail ? "FAIL" : "PASS", g_fail);
    return g_fail ? 1 : 0;
}

// fragdepth-host: runs the compat layer's FragDepth strip (MALI_COMPAT_FRAGDEPTH_FIX) on SPIR-V files on the host and
// writes <file>.out; check the result with spirv-val. [Claude Opus 5.5] 2026-10-28
// cc -I../panvk-kbase-android/work/mesa/include -Isrc/compat test/fragdepth-host.c -o fragdepth-host -lpthread -lm
#include "../src/compat/mali_compat_layer.c"

int main(int argc, char **argv) {
    for (int a = 1; a < argc; a++) {
        FILE *f = fopen(argv[a], "rb");
        if (!f) continue;
        fseek(f, 0, SEEK_END);
        long n = ftell(f);
        fseek(f, 0, SEEK_SET);
        uint32_t *code = malloc(n);
        if (fread(code, 1, n, f) != (size_t)n) { fclose(f); continue; }
        fclose(f);
        size_t out_bytes = 0;
        uint32_t *out = spv_strip_fragdepth(code, n, &out_bytes);
        printf("%s: writes FragDepth %d, %s %ld -> %zu bytes\n", argv[a], spv_writes_fragdepth(code, n),
               out ? "stripped" : "unchanged", n, out ? out_bytes : (size_t)n);
        char path[512];
        snprintf(path, sizeof path, "%s.out", argv[a]);
        if ((f = fopen(path, "wb"))) { fwrite(out ? out : code, 1, out ? out_bytes : (size_t)n, f); fclose(f); }
        free(out);
        free(code);
    }
    return 0;
}

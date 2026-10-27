// Host harness: runs the compat layer's ClipDistance/CullDistance SPIR-V patcher on .spv files and writes
// <file>.out for spirv-val. Build: cc -I../panvk-kbase-android/work/mesa/include -Isrc/compat test/spirv-patch-host.c
// [Claude Opus 5.5] 2026-10-26
#include "../src/compat/mali_compat_layer.c"

int main(int argc, char **argv) {
    for (int i = 1; i < argc; i++) {
        FILE *f = fopen(argv[i], "rb");
        if (!f) continue;
        fseek(f, 0, SEEK_END);
        long n = ftell(f);
        fseek(f, 0, SEEK_SET);
        uint32_t *w = malloc(n);
        if (fread(w, 1, n, f) != (size_t)n) { fclose(f); continue; }
        fclose(f);
        size_t outb = 0;
        uint32_t *p = compat_patch_spirv_impl(w, n, &outb);
        char path[4096];
        snprintf(path, sizeof path, "%s.out", argv[i]);
        f = fopen(path, "wb");
        fwrite(p ? p : w, 1, p ? outb : (size_t)n, f);
        fclose(f);
        printf("%s: %s %ld -> %zu bytes\n", argv[i], p ? "patched" : "unchanged", n, p ? outb : (size_t)n);
    }
    return 0;
}

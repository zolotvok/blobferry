// SPDX-License-Identifier: GPL-3.0-or-later
// Host harness for the compat layer's stream-output rewrite (MALI_COMPAT_XFB_VS): xfb-vs-host VS.spv GS.spv OUT.spv
// writes the vertex shader that captures what the passthrough geometry shader captured; check it with spirv-val.
// Build: cc -I../panvk-kbase-android/work/mesa/include -Isrc/compat test/xfb-vs-host.c -lpthread [Claude Opus 5.5] 2026-10-30
#include "../src/compat/mali_compat_layer.c"

static uint32_t *slurp(const char *path, size_t *bytes) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    uint32_t *w = malloc(n);
    if (fread(w, 1, n, f) != (size_t)n) { fclose(f); free(w); return NULL; }
    fclose(f);
    *bytes = n;
    return w;
}

int main(int argc, char **argv) {
    if (argc < 4) return 2;
    size_t vb, gb, ob = 0;
    uint32_t *vs = slurp(argv[1], &vb), *gs = slurp(argv[2], &gb);
    if (!vs || !gs) return 2;
    uint32_t *out = xfb_vs_from_gs(vs, vb, gs, gb, &ob);
    if (!out) { printf("not rewritten\n"); return 1; }
    FILE *f = fopen(argv[3], "wb");
    fwrite(out, 1, ob, f);
    fclose(f);
    printf("rewritten: %zu -> %zu bytes\n", vb, ob);
    return 0;
}

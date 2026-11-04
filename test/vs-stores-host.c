// SPDX-License-Identifier: GPL-3.0-or-later
// vs-stores-host: runs the compat layer's VS-stage store scan (spv_vs_stores, request KZ2) on SPIR-V files on the host
// and prints, per file, whether a vertex-pipeline module writes memory and the op counts. [keeper, Claude Opus 5.5] 2026-10-30
// cc -I../panvk-kbase-android/work/mesa/include -Isrc/compat test/vs-stores-host.c -o vs-stores-host -lpthread -lm
#include "../src/compat/mali_compat_layer.c"

int main(int argc, char **argv) {
    static const char *model[4] = {"vs", "tcs", "tes", "gs"};
    for (int a = 1; a < argc; a++) {
        FILE *f = fopen(argv[a], "rb");
        if (!f) continue;
        fseek(f, 0, SEEK_END);
        long n = ftell(f);
        fseek(f, 0, SEEK_SET);
        uint32_t *code = malloc(n);
        if (!code || fread(code, 1, n, f) != (size_t)n) { fclose(f); free(code); continue; }
        fclose(f);
        uint32_t cnt[4];
        bool w = spv_vs_stores(code, (size_t)n, cnt);
        printf("%s: %08x %s writes %d (imgwrite %u, atomic %u, store %u)\n", argv[a], spv_hash(code, (size_t)n),
               cnt[3] < 4 ? model[cnt[3]] : "not-vertex-pipeline", w, cnt[0], cnt[1], cnt[2]);
        free(code);
    }
    return 0;
}

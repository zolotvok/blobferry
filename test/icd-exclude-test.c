// SPDX-License-Identifier: GPL-3.0-or-later
// PC check of MALI_BRIDGE_EXCLUDE_COMM in src/icd/mali_bridge_icd.c (run under qemu-aarch64 with `-0 <argv0>`):
//   icd-exclude-test <path to libvulkan_mali_bridge.so> <expect: 1 = excluded, 0 = not> [comm to set first]
// Exit 0 when negotiation matches the expectation and, if excluded, vk_icdGetInstanceProcAddr returns NULL without
// opening the HAL. [keeper, Claude Opus 5.5] 2026-11-04
#include <dlfcn.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/prctl.h>

typedef int (*negotiate_t)(uint32_t *);
typedef void *(*gipa_t)(void *, const char *);

int main(int argc, char **argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s lib.so expect [comm]\n", argv[0]); return 2; }
    if (argc > 3) prctl(PR_SET_NAME, argv[3], 0, 0, 0);
    void *h = dlopen(argv[1], RTLD_NOW);
    if (!h) { fprintf(stderr, "dlopen: %s\n", dlerror()); return 2; }
    negotiate_t neg = (negotiate_t)dlsym(h, "vk_icdNegotiateLoaderICDInterfaceVersion");
    gipa_t gipa = (gipa_t)dlsym(h, "vk_icdGetInstanceProcAddr");
    if (!neg || !gipa) { fprintf(stderr, "missing exports\n"); return 2; }
    uint32_t v = 7;
    int r = neg(&v);
    int expect = atoi(argv[2]);
    int got = (r == -9 /* VK_ERROR_INCOMPATIBLE_DRIVER */);
    printf("negotiate rc=%d version=%u excluded=%d expect=%d\n", r, v, got, expect);
    if (got != expect) return 1;
    if (!expect) return v == 5 && r == 0 ? 0 : 1;
    // excluded: must answer NULL before dev() (MALI_BRIDGE_HAL points nowhere, MALI_BRIDGE_DEBUG would log a HAL open)
    void *f = gipa(NULL, "vkCreateInstance");
    printf("gipa(vkCreateInstance)=%p\n", f);
    return f == NULL ? 0 : 1;
}

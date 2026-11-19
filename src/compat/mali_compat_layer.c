// SPDX-License-Identifier: GPL-3.0-or-later
#define _GNU_SOURCE
#include <dlfcn.h> // dladdr (K-WC1 wait callers)
// Mali Compatibility Layer for DXVK / Proton (VK_LAYER_DROIDDECK_mali_compat)
// Implements feature masking, polygon mode override, SPIR-V clip/cull distance stripping,
// BC1-BC7 GPU compute texture emulation via PanVK shaders, and VK_EXT_memory_budget.
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <string.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdarg.h>
#include <errno.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/sysmacros.h>
#include <sys/ioctl.h>
#include <fcntl.h>
#include <time.h>
#include <unistd.h>
#include <vulkan/vulkan.h>
#include <vulkan/vk_layer.h>

#include "bc/panvk_bc.h"

#define EXPORT __attribute__((visibility("default")))
#define LAYER_NAME "VK_LAYER_DROIDDECK_mali_compat"

static int g_debug = 0;

// pNext helpers and the page size (used by the placed-map emulation and the feature/property queries).
static size_t page_size(void) {
    static size_t ps;
    if (!ps) ps = (size_t)sysconf(_SC_PAGESIZE);
    return ps;
}

static const void *find_struct(const void *chain, VkStructureType type) {
    for (const VkBaseInStructure *s = chain; s; s = s->pNext)
        if (s->sType == type) return s;
    return NULL;
}

// Unlinks the first struct of `type` from a chain that starts after `head`; returns {prev, node} to relink later.
struct unlinked { VkBaseOutStructure *prev, *node; };
static struct unlinked unlink_struct(void *head, VkStructureType type) {
    for (VkBaseOutStructure *prev = head, *s = prev->pNext; s; prev = s, s = s->pNext) {
        if (s->sType == type) {
            prev->pNext = s->pNext;
            return (struct unlinked){prev, s};
        }
    }
    return (struct unlinked){NULL, NULL};
}
static void relink(struct unlinked u) {
    if (u.prev) u.prev->pNext = u.node;
}

// 11-10: one persistent log handle instead of fopen/fclose per line (ported from Gemini 4 Argon's wt-rtx-emu; the
// RT path logs thousands of lines). fflush after every line so a crash or GPU-fault kill loses nothing.
static void mali_compat_log_write(const char *fmt, ...) {
    static pthread_mutex_t log_lock = PTHREAD_MUTEX_INITIALIZER;
    static FILE *s_log_file = NULL;
    static bool s_log_opened = false;
    pthread_mutex_lock(&log_lock);
    va_list ap;
    va_start(ap, fmt);
    vfprintf(stderr, fmt, ap);
    fputc('\n', stderr);
    fflush(stderr);
    va_end(ap);

    if (!s_log_opened) {
        s_log_file = fopen("/tmp/mali-compat.log", "a");
        if (!s_log_file) s_log_file = fopen("/data/local/tmp/mali-compat.log", "a");
        s_log_opened = true;
    }
    if (s_log_file) {
        va_start(ap, fmt);
        vfprintf(s_log_file, fmt, ap);
        fputc('\n', s_log_file);
        fflush(s_log_file);
        va_end(ap);
    }
    pthread_mutex_unlock(&log_lock);
}

#define LOG(...) mali_compat_log_write("mali-compat: " __VA_ARGS__)

#include "cl_spv_wls.h"
#include "cl_spv_fix.h"
#include "cl_state.h"
#include "cl_diag_desccap.h"
#include "cl_staging_bc.h"
#include "mali_rt_emu.h"

#include "cl_physdev.h"
#include "cl_shader.h"
#include "cl_pipeline.h"
#include "cl_image_mem.h"
#include "cl_cmd.h"
#include "cl_diag_gputime.h"
#include "cl_sync.h"
#include "cl_diag_ssbo.h"
#include "cl_diag_dpool.h"
#include "cl_submit_draw.h"
#include "cl_device.h"
#include "cl_diag_addrmap.h"
#include "cl_procaddr.h"

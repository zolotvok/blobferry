// SPDX-License-Identifier: GPL-3.0-or-later
// cl_image_mem.h: images, RT_COMPRESS, ADDRMAP object names, buffers, FAKE_OCCLUSION, memory and placed maps. Included by mali_compat_layer.c.
// Debug aid: with /root/mali-images.txt present, one line per image: format, size, usage, flags, and the driver's
// answer on compression (VkImageCompressionPropertiesEXT: DEFAULT = AFBC on Mali, DISABLED = uncompressed).
// [Claude Opus 5.5] 2026-10-26
static void log_image(struct DeviceData *d, VkDevice device, const VkImageCreateInfo *ci, VkImage img, bool bc) {
    static pthread_mutex_t m = PTHREAD_MUTEX_INITIALIZER;
    VkImageCompressionPropertiesEXT cp = {VK_STRUCTURE_TYPE_IMAGE_COMPRESSION_PROPERTIES_EXT};
    if (d->fpGetImageSubresourceLayout2KHR) {
        VkImageAspectFlags aspect = VK_IMAGE_ASPECT_COLOR_BIT;
        if (ci->format >= VK_FORMAT_D16_UNORM && ci->format <= VK_FORMAT_D32_SFLOAT_S8_UINT)
            aspect = ci->format == VK_FORMAT_S8_UINT ? VK_IMAGE_ASPECT_STENCIL_BIT : VK_IMAGE_ASPECT_DEPTH_BIT;
        VkImageSubresource2KHR sub = {VK_STRUCTURE_TYPE_IMAGE_SUBRESOURCE_2_KHR, NULL, {aspect, 0, 0}};
        VkSubresourceLayout2KHR lay = {VK_STRUCTURE_TYPE_SUBRESOURCE_LAYOUT_2_KHR, &cp};
        if (ci->tiling != VK_IMAGE_TILING_DRM_FORMAT_MODIFIER_EXT) d->fpGetImageSubresourceLayout2KHR(device, img, &sub, &lay);
    }
    pthread_mutex_lock(&m);
    FILE *f = fopen("/root/mali-images.txt", "a");
    if (f) {
        const VkImageCompressionControlEXT *acc = find_struct(ci->pNext, VK_STRUCTURE_TYPE_IMAGE_COMPRESSION_CONTROL_EXT);
        fprintf(f, "fmt %d %ux%ux%u mips %u layers %u samples %d tiling %d usage 0x%x flags 0x%x%s comp 0x%x rate 0x%x appcc %s\n",
                ci->format, ci->extent.width, ci->extent.height, ci->extent.depth, ci->mipLevels, ci->arrayLayers,
                ci->samples, ci->tiling, ci->usage, ci->flags, bc ? " bc-shadow" : "", cp.imageCompressionFlags,
                cp.imageCompressionFixedRateFlags, acc ? (acc->flags == VK_IMAGE_COMPRESSION_DISABLED_EXT ? "disabled" : "other") : "-");
        fclose(f);
    }
    pthread_mutex_unlock(&m);
}

// -----------------------------------------------------------------------------
// Render-target compression, MALI_COMPAT_RT_COMPRESS=1. [Claude Opus 5.5] 2026-10-28
// The vendor driver turns AFBC off for every image with STORAGE usage, for all colour formats, and offers no
// fixed-rate compression for them either (test/rt-comp-probe.c). DXVK adds STORAGE to each render target the game
// creates with a UAV bind flag, so all of The Witcher 3's 1600x720 colour targets ran uncompressed (image log
// 2026-10-28: 106 targets, comp 0x4). With the switch on, colour targets are created without STORAGE. A storage view
// on such an image means the game does use it as a UAV: its signature (format, size, mips, layers) is appended to
// /root/mali-rt-keep-<comm>.txt, and from the next launch on, images with that signature keep STORAGE. In the launch
// that learns a signature, that UAV use is invalid (expect glitches there), which is why it is logged.
// -----------------------------------------------------------------------------
static bool g_rt_compress;
// MALI_COMPAT_DEPTH_NOCOMP=1: depth/stencil images that shaders read are created uncompressed (no AFBC). The current
// Witcher 3 (DX12) hangs the GPU in vkd3d's depth->colour copy pass, which texelFetches the depth aspect of its
// D32_SFLOAT_S8_UINT depth buffer (GPU timing STUCK lines, 2026-10-28); test whether compression is the trigger.
static bool g_depth_nocomp;
// MALI_COMPAT_FORCE_COMP=1: vkd3d-proton chains VkImageCompressionControlEXT DISABLED into placed (aliasable)
// resources, so DX12 games run all their render targets and shadow maps uncompressed on this driver. For colour and
// depth/stencil attachments without STORAGE the layer turns that request back into DEFAULT (AFBC). Experiment: an
// aliased target then shares memory with compression metadata, which vkd3d does not expect. [Claude Opus 5.5] 2026-10-28
static bool g_force_comp;
struct RTSig { VkFormat format; uint32_t width, height, mips, layers; };
#define RT_KEEP_MAX 256
static struct RTSig g_rt_keep[RT_KEEP_MAX];
static uint32_t g_rt_keep_count;
static char g_rt_keep_path[96];
struct RTNode { VkImage image; struct RTSig sig; VkImageUsageFlags usage; struct RTNode *next; };
#define RT_BUCKETS 1024
static struct RTNode *g_rt_images[RT_BUCKETS];
static pthread_mutex_t g_rt_mutex = PTHREAD_MUTEX_INITIALIZER;
static _Atomic uint32_t g_rt_stripped, g_rt_kept;

static struct RTSig rt_sig(const VkImageCreateInfo *ci) {
    return (struct RTSig){ci->format, ci->extent.width, ci->extent.height, ci->mipLevels, ci->arrayLayers};
}

static bool rt_keep_has_locked(struct RTSig s) {
    for (uint32_t i = 0; i < g_rt_keep_count; i++)
        if (!memcmp(&g_rt_keep[i], &s, sizeof s)) return true;
    return false;
}

static void rt_keep_load(void) {
    char comm[32] = "unknown";
    FILE *f = fopen("/proc/self/comm", "r");
    if (f) {
        if (fgets(comm, sizeof comm, f)) comm[strcspn(comm, "\n")] = 0;
        fclose(f);
    }
    for (char *c = comm; *c; c++)
        if (*c == '/' || *c == ' ' || *c == '\\') *c = '_';
    snprintf(g_rt_keep_path, sizeof g_rt_keep_path, "/root/mali-rt-keep-%s.txt", comm);
    pthread_mutex_lock(&g_rt_mutex);
    g_rt_keep_count = 0;
    if ((f = fopen(g_rt_keep_path, "r"))) {
        struct RTSig s;
        int fmt;
        while (g_rt_keep_count < RT_KEEP_MAX && fscanf(f, "%d %u %u %u %u", &fmt, &s.width, &s.height, &s.mips, &s.layers) == 5) {
            s.format = (VkFormat)fmt;
            g_rt_keep[g_rt_keep_count++] = s;
        }
        fclose(f);
    }
    pthread_mutex_unlock(&g_rt_mutex);
    fprintf(stderr, "mali-compat: rt-compress on, %u UAV target signatures kept from %s\n", g_rt_keep_count, g_rt_keep_path);
}

// A UAV view on a stripped target: remember its signature for the next launches.
static void rt_learn(struct RTSig s) {
    pthread_mutex_lock(&g_rt_mutex);
    if (!rt_keep_has_locked(s) && g_rt_keep_count < RT_KEEP_MAX) {
        g_rt_keep[g_rt_keep_count++] = s;
        FILE *f = fopen(g_rt_keep_path, "a");
        if (f) {
            fprintf(f, "%d %u %u %u %u\n", s.format, s.width, s.height, s.mips, s.layers);
            fclose(f);
        }
        fprintf(stderr, "mali-compat: rt-compress: UAV view on compressed target fmt %d %ux%u mips %u layers %u; "
                "learned, keeps STORAGE from the next launch\n", s.format, s.width, s.height, s.mips, s.layers);
    }
    pthread_mutex_unlock(&g_rt_mutex);
}

static void rt_track(VkImage img, struct RTSig s, VkImageUsageFlags usage) {
    struct RTNode *e = calloc(1, sizeof *e);
    if (!e) return;
    e->image = img;
    e->sig = s;
    e->usage = usage;
    uint32_t b = handle_bucket((uint64_t)(uintptr_t)img, RT_BUCKETS);
    pthread_mutex_lock(&g_rt_mutex);
    e->next = g_rt_images[b];
    g_rt_images[b] = e;
    pthread_mutex_unlock(&g_rt_mutex);
}

static bool rt_find(VkImage img, struct RTNode *out) {
    bool found = false;
    pthread_mutex_lock(&g_rt_mutex);
    for (struct RTNode *e = g_rt_images[handle_bucket((uint64_t)(uintptr_t)img, RT_BUCKETS)]; e; e = e->next)
        if (e->image == img) { *out = *e; found = true; break; }
    pthread_mutex_unlock(&g_rt_mutex);
    return found;
}

static void rt_untrack(VkImage img) {
    pthread_mutex_lock(&g_rt_mutex);
    for (struct RTNode **pp = &g_rt_images[handle_bucket((uint64_t)(uintptr_t)img, RT_BUCKETS)]; *pp; pp = &(*pp)->next)
        if ((*pp)->image == img) {
            struct RTNode *e = *pp;
            *pp = e->next;
            free(e);
            break;
        }
    pthread_mutex_unlock(&g_rt_mutex);
}

// The create info the driver gets for an app image: BC formats become their RGBA8/RG16/RGBA16F shadow (format list
// mapped, fixed-rate compression when offered) and Wine's HOST_ALLOCATION external memory is unlinked. Used by
// vkCreateImage and by the vkGetDeviceImage*Requirements queries, which must answer for the same image: vkd3d-proton
// sizes its placed D3D12 resources with them before creating the image, and answering for the BC format (which the
// vendor does not even support) left HZD's textures in too-small heap slots, so every surface sampled black.
// fixup_end restores the app's chain. [Claude Opus 5.5] 2026-10-26
struct image_fixup {
    VkImageCreateInfo mod;
    const VkImageCreateInfo *use;
    bool is_bc;
    VkImageCompressionFixedRateFlagsEXT rate;
    VkImageCompressionControlEXT afrc;
    VkImageFormatListCreateInfo *fl;
    const VkFormat *fl_orig;
    VkFormat *fl_mapped;
    VkImageFormatListCreateInfo fl_synth; // for MUTABLE shadows created without a list
    VkFormat fl_synth_formats[2];
    VkImageCompressionControlEXT *app_cc; // the app's own compression control, overridden for the call
    VkImageCompressionControlEXT app_cc_saved;
    VkBaseOutStructure *host_prev, *host_ext;
    bool rt_stripped; // colour target created without the app's STORAGE (MALI_COMPAT_RT_COMPRESS)
};

static void fixup_begin(struct DeviceData *d, const VkImageCreateInfo *ci, struct image_fixup *fx) {
    memset(fx, 0, sizeof *fx);
    fx->mod = *ci;
    fx->use = ci;
    fx->is_bc = panvk_format_is_bc(ci->format);
    if (fx->is_bc) {
        fx->use = &fx->mod;
        fx->mod.format = panvk_bc_decoded_format(ci->format);
        fx->mod.flags &= ~VK_IMAGE_CREATE_BLOCK_TEXEL_VIEW_COMPATIBLE_BIT;
        fx->fl = (VkImageFormatListCreateInfo *)find_struct(fx->mod.pNext, VK_STRUCTURE_TYPE_IMAGE_FORMAT_LIST_CREATE_INFO);
        if (fx->fl && fx->fl->viewFormatCount && fx->fl->pViewFormats &&
            (fx->fl_mapped = malloc(fx->fl->viewFormatCount * sizeof(VkFormat)))) {
            for (uint32_t v = 0; v < fx->fl->viewFormatCount; v++)
                fx->fl_mapped[v] = panvk_format_is_bc(fx->fl->pViewFormats[v]) ? panvk_bc_decoded_format(fx->fl->pViewFormats[v])
                                                                                : fx->fl->pViewFormats[v];
            fx->fl_orig = fx->fl->pViewFormats;
            fx->fl->pViewFormats = fx->fl_mapped;
        }
        // A MUTABLE image without a format list gets no compression from the vendor driver (afrc-probe: RGBA8_SRGB
        // MUTABLE without a list 5465 KiB uncompressed, with a UNORM/SRGB list 1370 KiB at 2 bpc; ALIAS does not
        // matter). HZD's 995 BC7 shadows were created that way. The only views an RGBA8 shadow can serve are its
        // UNORM and SRGB variants (BLOCK_TEXEL_VIEW_COMPATIBLE is stripped above), so give it that list.
        // [Claude Opus 5.5] 2026-10-26
        if (!fx->fl && (fx->mod.flags & VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT) &&
            (fx->mod.format == VK_FORMAT_R8G8B8A8_UNORM || fx->mod.format == VK_FORMAT_R8G8B8A8_SRGB)) {
            fx->fl_synth_formats[0] = VK_FORMAT_R8G8B8A8_UNORM;
            fx->fl_synth_formats[1] = VK_FORMAT_R8G8B8A8_SRGB;
            fx->fl_synth = (VkImageFormatListCreateInfo){VK_STRUCTURE_TYPE_IMAGE_FORMAT_LIST_CREATE_INFO, fx->mod.pNext, 2,
                                                         fx->fl_synth_formats};
            fx->mod.pNext = &fx->fl_synth;
            fx->fl = &fx->fl_synth;
        }
        // Fixed-rate compression for the BC shadows: an RGBA8 shadow is 4-8x the BC data (Tomb Raider: ~2.7 GB of
        // shadows instead of ~0.5 GB). AFRC at 2 bpc (8 bpp, BC3's size) for BC1's 4-colour blocks, 3 bpc for the
        // others; only when the driver offers that rate for this format, usage and flags (test/afrc-probe.c).
        // vkd3d-proton chains its own VkImageCompressionControlEXT (it disables compression on placed/aliasable
        // resources): HZD's 3771 images, all BC shadows included, came out comp 0x4 (DISABLED), ~2.6 GB uncompressed.
        // A shadow's contents only ever come from the layer's decode copy, so its compression is the layer's call:
        // the app's struct is overridden for the call and restored in fixup_end. [Claude Opus 5.5] 2026-10-26
        VkImageCompressionControlEXT *app_cc = (VkImageCompressionControlEXT *)find_struct(fx->mod.pNext,
            VK_STRUCTURE_TYPE_IMAGE_COMPRESSION_CONTROL_EXT);
        if (d->afrc && (!app_cc || app_cc->flags != VK_IMAGE_COMPRESSION_FIXED_RATE_EXPLICIT_EXT)) {
            VkImageCompressionControlEXT q = {VK_STRUCTURE_TYPE_IMAGE_COMPRESSION_CONTROL_EXT, NULL,
                                              VK_IMAGE_COMPRESSION_FIXED_RATE_DEFAULT_EXT};
            VkImageFormatListCreateInfo fl_copy;
            if (fx->fl) { fl_copy = *fx->fl; fl_copy.pNext = NULL; q.pNext = &fl_copy; }
            VkPhysicalDeviceImageFormatInfo2 qi = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_IMAGE_FORMAT_INFO_2, &q, fx->mod.format,
                                                   fx->mod.imageType, fx->mod.tiling, fx->mod.usage, fx->mod.flags};
            VkImageCompressionPropertiesEXT cp = {VK_STRUCTURE_TYPE_IMAGE_COMPRESSION_PROPERTIES_EXT};
            VkImageFormatProperties2 qp = {VK_STRUCTURE_TYPE_IMAGE_FORMAT_PROPERTIES_2, &cp};
            if (d->instance_data && d->instance_data->fpGetPhysicalDeviceImageFormatProperties2 &&
                d->instance_data->fpGetPhysicalDeviceImageFormatProperties2(d->physical_device, &qi, &qp) == VK_SUCCESS) {
                bool bc1 = ci->format >= VK_FORMAT_BC1_RGB_UNORM_BLOCK && ci->format <= VK_FORMAT_BC1_RGBA_SRGB_BLOCK;
                VkImageCompressionFixedRateFlagsEXT want = VK_IMAGE_COMPRESSION_FIXED_RATE_NONE_EXT;
                if (cp.imageCompressionFixedRateFlags & VK_IMAGE_COMPRESSION_FIXED_RATE_2BPC_BIT_EXT)
                    want = VK_IMAGE_COMPRESSION_FIXED_RATE_2BPC_BIT_EXT;
                else if (!bc1 && (cp.imageCompressionFixedRateFlags & VK_IMAGE_COMPRESSION_FIXED_RATE_3BPC_BIT_EXT))
                    want = VK_IMAGE_COMPRESSION_FIXED_RATE_3BPC_BIT_EXT;
                if (want != VK_IMAGE_COMPRESSION_FIXED_RATE_NONE_EXT) {
                    fx->rate = want;
                    if (app_cc) {
                        fx->app_cc = app_cc;
                        fx->app_cc_saved = *app_cc;
                        app_cc->flags = VK_IMAGE_COMPRESSION_FIXED_RATE_EXPLICIT_EXT;
                        app_cc->compressionControlPlaneCount = 1;
                        app_cc->pFixedRateFlags = &fx->rate;
                    } else {
                        fx->afrc = (VkImageCompressionControlEXT){VK_STRUCTURE_TYPE_IMAGE_COMPRESSION_CONTROL_EXT, (void *)fx->mod.pNext,
                                                                  VK_IMAGE_COMPRESSION_FIXED_RATE_EXPLICIT_EXT, 1, &fx->rate};
                        fx->mod.pNext = &fx->afrc;
                    }
                }
            }
        }
    }
    if (g_rt_compress && !fx->is_bc && (ci->usage & VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT) &&
        (ci->usage & VK_IMAGE_USAGE_STORAGE_BIT)) {
        pthread_mutex_lock(&g_rt_mutex);
        bool keep = rt_keep_has_locked(rt_sig(ci));
        pthread_mutex_unlock(&g_rt_mutex);
        if (!keep) {
            fx->use = &fx->mod;
            fx->mod.usage &= ~VK_IMAGE_USAGE_STORAGE_BIT;
            fx->rt_stripped = true;
        }
    }
    if (g_force_comp && !fx->is_bc && !g_depth_nocomp &&
        (fx->mod.usage & (VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT)) &&
        !(fx->mod.usage & VK_IMAGE_USAGE_STORAGE_BIT)) {
        VkImageCompressionControlEXT *app_cc = (VkImageCompressionControlEXT *)find_struct(fx->mod.pNext,
            VK_STRUCTURE_TYPE_IMAGE_COMPRESSION_CONTROL_EXT);
        if (app_cc && app_cc->flags == VK_IMAGE_COMPRESSION_DISABLED_EXT) {
            fx->use = &fx->mod;
            fx->app_cc = app_cc;
            fx->app_cc_saved = *app_cc;
            app_cc->flags = VK_IMAGE_COMPRESSION_DEFAULT_EXT;
        }
    }
    if (g_depth_nocomp && d->afrc && !fx->is_bc && (ci->usage & VK_IMAGE_USAGE_SAMPLED_BIT) &&
        ((ci->format >= VK_FORMAT_D16_UNORM && ci->format <= VK_FORMAT_D32_SFLOAT_S8_UINT))) {
        VkImageCompressionControlEXT *app_cc = (VkImageCompressionControlEXT *)find_struct(fx->mod.pNext,
            VK_STRUCTURE_TYPE_IMAGE_COMPRESSION_CONTROL_EXT);
        fx->use = &fx->mod;
        if (app_cc) {
            fx->app_cc = app_cc;
            fx->app_cc_saved = *app_cc;
            app_cc->flags = VK_IMAGE_COMPRESSION_DISABLED_EXT;
            app_cc->compressionControlPlaneCount = 0;
            app_cc->pFixedRateFlags = NULL;
        } else {
            fx->afrc = (VkImageCompressionControlEXT){VK_STRUCTURE_TYPE_IMAGE_COMPRESSION_CONTROL_EXT, (void *)fx->mod.pNext,
                                                      VK_IMAGE_COMPRESSION_DISABLED_EXT, 0, NULL};
            fx->mod.pNext = &fx->afrc;
        }
    }
    // Wine on WoW64 chains HOST_ALLOCATION external memory into every image; the vendor refuses such images
    // (-1000158000) but binds a plain image to host-pointer memory fine (test/img-probe.c): unlink it for the call.
    for (VkBaseOutStructure *prev = (VkBaseOutStructure *)fx->use, *st = prev->pNext; st; prev = st, st = st->pNext) {
        if (st->sType == VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO &&
            ((VkExternalMemoryImageCreateInfo *)st)->handleTypes == VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT) {
            if (fx->use == ci) { fx->use = &fx->mod; prev = prev == (VkBaseOutStructure *)ci ? (VkBaseOutStructure *)&fx->mod : prev; }
            fx->host_prev = prev;
            fx->host_ext = st;
            prev->pNext = st->pNext;
            break;
        }
    }
}

static void fixup_end(struct image_fixup *fx) {
    if (fx->app_cc) *fx->app_cc = fx->app_cc_saved;
    if (fx->host_prev) fx->host_prev->pNext = fx->host_ext;
    if (fx->fl_orig) fx->fl->pViewFormats = fx->fl_orig;
    free(fx->fl_mapped);
}

// ---- MALI_COMPAT_ADDRMAP: object descriptions (created here, bound in the bind hooks, dropped at destroy) ----------
struct AmObj {
    uint64_t h;
    struct AmObj *next;
    uint8_t kind;                    // 1 buffer, 2 image
    bool bound;
    uint32_t usage, appusage, flags, appflags; // driver-side usage/flags and the app's
    uint32_t mips, layers, samples, tiling, imgtype, w, hh, dd;
    int32_t fmt, bcfmt;              // driver-side format; the app's BC format (0 = not an emulated BC shadow)
    uint32_t afrc;                   // fixed-rate flags the layer chained for a BC shadow (0 = none)
    int64_t cc;                      // compression-control flags in the driver's chain (-1 = no struct)
    uint32_t cc_rate;                // its first pFixedRateFlags entry (explicit rate), 0 = none
    uint64_t csize;                  // buffer create size
    uint64_t mem, off, req, va;      // the bind (req / va AM_NA when unknown)
};
#define AM_BUCKETS 16384
static struct AmObj *g_am_obj[AM_BUCKETS];
static pthread_mutex_t g_am_mutex = PTHREAD_MUTEX_INITIALIZER;
static _Atomic uint64_t g_am_obj_live;

static void am_obj_put(const struct AmObj *o) {
    struct AmObj *e = malloc(sizeof *e);
    if (!e) return;
    *e = *o;
    uint32_t b = handle_bucket(o->h, AM_BUCKETS);
    struct AmObj *old = NULL;
    pthread_mutex_lock(&g_am_mutex);
    for (struct AmObj **pp = &g_am_obj[b]; *pp; pp = &(*pp)->next)
        if ((*pp)->h == o->h) { old = *pp; *pp = old->next; break; } // a handle reused without a destroy seen here
    e->next = g_am_obj[b];
    g_am_obj[b] = e;
    pthread_mutex_unlock(&g_am_mutex);
    if (old) free(old);
    else atomic_fetch_add_explicit(&g_am_obj_live, 1, memory_order_relaxed);
}
// Copies the entry out; with bind set, records the bind first. remove: the entry goes (destroy).
static bool am_obj_get(uint64_t h, struct AmObj *out, const struct AmObj *bind, bool remove) {
    uint32_t b = handle_bucket(h, AM_BUCKETS);
    struct AmObj *e = NULL;
    pthread_mutex_lock(&g_am_mutex);
    for (struct AmObj **pp = &g_am_obj[b]; *pp; pp = &(*pp)->next) {
        if ((*pp)->h != h) continue;
        e = *pp;
        if (bind) { e->bound = true; e->mem = bind->mem; e->off = bind->off; e->req = bind->req; e->va = bind->va; }
        *out = *e;
        if (remove) *pp = e->next;
        break;
    }
    pthread_mutex_unlock(&g_am_mutex);
    if (e && remove) { free(e); atomic_fetch_sub_explicit(&g_am_obj_live, 1, memory_order_relaxed); }
    return e != NULL;
}

static const char *am_rate(uint32_t f) {
    return !f ? "none" : f == VK_IMAGE_COMPRESSION_FIXED_RATE_2BPC_BIT_EXT ? "2bpc"
         : f == VK_IMAGE_COMPRESSION_FIXED_RATE_3BPC_BIT_EXT ? "3bpc" : "other";
}
static void am_desc(const struct AmObj *o, char *s, size_t n) {
    if (!o) { snprintf(s, n, "desc=unknown"); return; }
    if (o->kind == 1) {
        snprintf(s, n, "csize=%llu usage=0x%x appusage=0x%x flags=0x%x", (unsigned long long)o->csize, o->usage, o->appusage,
                 o->flags);
        return;
    }
    char bc[16] = "-", cc[48] = "-";
    if (o->bcfmt) snprintf(bc, sizeof bc, "%d", o->bcfmt);
    if (o->cc >= 0) snprintf(cc, sizeof cc, "0x%llx/%s", (unsigned long long)o->cc, am_rate(o->cc_rate));
    snprintf(s, n, "fmt=%d bcfmt=%s ext=%ux%ux%u mips=%u layers=%u samples=%u tiling=%u type=%u usage=0x%x appusage=0x%x "
             "flags=0x%x appflags=0x%x afrc=%s cc=%s", o->fmt, bc, o->w, o->hh, o->dd, o->mips, o->layers, o->samples,
             o->tiling, o->imgtype, o->usage, o->appusage, o->flags, o->appflags, am_rate(o->afrc), cc);
}

// vkCreateImage (always hooked; this runs only with the key): what the driver got, before fixup_end restores the app's
// own compression-control struct.
static void am_image_created(VkImage img, const VkImageCreateInfo *app, const struct image_fixup *fx) {
    const VkImageCreateInfo *u = fx->use;
    struct AmObj o = {.h = (uint64_t)(uintptr_t)img, .kind = 2, .usage = u->usage, .appusage = app->usage, .flags = u->flags,
                      .appflags = app->flags, .mips = u->mipLevels, .layers = u->arrayLayers, .samples = u->samples,
                      .tiling = u->tiling, .imgtype = u->imageType, .w = u->extent.width, .hh = u->extent.height,
                      .dd = u->extent.depth, .fmt = u->format, .bcfmt = fx->is_bc ? app->format : 0,
                      .afrc = fx->is_bc ? fx->rate : 0, .cc = -1, .req = AM_NA, .va = AM_NA};
    const VkImageCompressionControlEXT *cc = find_struct(u->pNext, VK_STRUCTURE_TYPE_IMAGE_COMPRESSION_CONTROL_EXT);
    if (cc) {
        o.cc = cc->flags;
        if (cc->flags == VK_IMAGE_COMPRESSION_FIXED_RATE_EXPLICIT_EXT && cc->compressionControlPlaneCount && cc->pFixedRateFlags)
            o.cc_rate = cc->pFixedRateFlags[0];
    }
    am_obj_put(&o);
}
static void am_buffer_created(VkBuffer b, const VkBufferCreateInfo *app, const VkBufferCreateInfo *drv) {
    struct AmObj o = {.h = (uint64_t)(uintptr_t)b, .kind = 1, .usage = drv->usage, .appusage = app->usage, .flags = drv->flags,
                      .csize = drv->size, .cc = -1, .req = AM_NA, .va = AM_NA};
    am_obj_put(&o);
}

// The memory's base VA: the capture address of its A event ("cap"), else its last mapped base ("map", the M event;
// SAME_VA: CPU VA = GPU VA), else 0 ("-").
static uint64_t am_mem_va(VkDeviceMemory m, const char **src) {
    uint64_t va = 0;
    *src = "-";
    if (!m) return 0;
    pthread_mutex_lock(&g_mem_mutex);
    struct MemInfo *e = mem_find(m);
    if (e && e->cva) { va = e->cva; *src = "cap"; }
    else if (e && e->va) { va = e->va; *src = "map"; }
    pthread_mutex_unlock(&g_mem_mutex);
    return va;
}

struct AmFns { // driver entry points for the ADDRMAP hooks (NULL = the driver lacks it; the hook is then not handed out)
    PFN_vkBindImageMemory BindImageMemory;
    PFN_vkBindBufferMemory2 BindBufferMemory2;
    PFN_vkBindImageMemory2 BindImageMemory2;
    PFN_vkGetImageMemoryRequirements GetImageMemoryRequirements;
    PFN_vkCreateCommandPool CreateCommandPool;
    PFN_vkResetDescriptorPool ResetDescriptorPool;
    PFN_vkDestroyDescriptorPool DestroyDescriptorPool;
    PFN_vkCreateSwapchainKHR CreateSwapchainKHR;
    PFN_vkDestroySwapchainKHR DestroySwapchainKHR;
    PFN_vkCmdCopyImage CmdCopyImage;
    PFN_vkCmdBlitImage CmdBlitImage;
    PFN_vkCmdCopyImageToBuffer CmdCopyImageToBuffer;
    PFN_vkCmdResolveImage CmdResolveImage;
    PFN_vkCmdCopyImage2 CmdCopyImage2;
    PFN_vkCmdBlitImage2 CmdBlitImage2;
    PFN_vkCmdCopyImageToBuffer2 CmdCopyImageToBuffer2;
    PFN_vkCmdResolveImage2 CmdResolveImage2;
};
static void am_device_init(struct DeviceData *d, PFN_vkGetDeviceProcAddr gdpa, VkDevice dev) {
    struct AmFns *f = calloc(1, sizeof *f);
    if (!f) return;
    #define AM_LOAD(fn) if (!(f->fn = (PFN_vk##fn)gdpa(dev, "vk" #fn))) f->fn = (PFN_vk##fn)gdpa(dev, "vk" #fn "KHR")
    AM_LOAD(BindImageMemory);
    AM_LOAD(BindBufferMemory2);
    AM_LOAD(BindImageMemory2);
    AM_LOAD(GetImageMemoryRequirements);
    AM_LOAD(CreateCommandPool);
    AM_LOAD(ResetDescriptorPool);
    AM_LOAD(DestroyDescriptorPool);
    AM_LOAD(CmdCopyImage);
    AM_LOAD(CmdBlitImage);
    AM_LOAD(CmdCopyImageToBuffer);
    AM_LOAD(CmdResolveImage);
    AM_LOAD(CmdCopyImage2);
    AM_LOAD(CmdBlitImage2);
    AM_LOAD(CmdCopyImageToBuffer2);
    AM_LOAD(CmdResolveImage2);
    #undef AM_LOAD
    f->CreateSwapchainKHR = (PFN_vkCreateSwapchainKHR)gdpa(dev, "vkCreateSwapchainKHR");
    f->DestroySwapchainKHR = (PFN_vkDestroySwapchainKHR)gdpa(dev, "vkDestroySwapchainKHR");
    d->amf = f;
}

// B event. kind 1 buffer, 2 image; query: one vkGet*MemoryRequirements for the size (not for disjoint planes or
// swapchain binds); extra: plane / swapchain detail.
static void am_bind(struct DeviceData *d, VkDevice device, int kind, uint64_t h, VkDeviceMemory m, VkDeviceSize off, bool query,
                    const char *extra) {
    struct AmObj o;
    bool known = am_obj_get(h, &o, NULL, false);
    VkMemoryRequirements mr = {0};
    bool have = false;
    if (query && m) {
        if (kind == 1 && d->fpGetBufferMemoryRequirements) {
            d->fpGetBufferMemoryRequirements(device, (VkBuffer)(uintptr_t)h, &mr);
            have = true;
        } else if (kind == 2 && d->amf && d->amf->GetImageMemoryRequirements && !(known && (o.flags & VK_IMAGE_CREATE_DISJOINT_BIT))) {
            d->amf->GetImageMemoryRequirements(device, (VkImage)(uintptr_t)h, &mr);
            have = true;
        }
    }
    const char *vasrc;
    uint64_t cva = am_mem_va(m, &vasrc);
    struct AmObj bind = {.mem = (uint64_t)(uintptr_t)m, .off = off, .req = have ? mr.size : AM_NA, .va = cva ? cva + off : AM_NA};
    known = am_obj_get(h, &o, &bind, false);
    char desc[448];
    am_desc(known ? &o : NULL, desc, sizeof desc);
    am_emit('B', (uint64_t)(uintptr_t)m, bind.req, ML_NONE, BUDGET_NO_HEAP, bind.va, 1, "kind=%s obj=0x%llx off=0x%llx align=%llu vasrc=%s %s%s",
            kind == 1 ? "buf" : "img", (unsigned long long)h, (unsigned long long)off, (unsigned long long)mr.alignment, vasrc, desc,
            extra ? extra : "");
}
static void am_destroy(int kind, uint64_t h) {
    struct AmObj o;
    bool known = am_obj_get(h, &o, NULL, true);
    if (known && o.bound)
        am_emit('D', h, o.req, ML_NONE, BUDGET_NO_HEAP, o.va, 1, "kind=%s obj=0x%llx mem=0x%llx off=0x%llx", kind == 1 ? "buf" : "img",
                (unsigned long long)h, (unsigned long long)o.mem, (unsigned long long)o.off);
    else
        am_emit('D', h, AM_NA, ML_NONE, BUDGET_NO_HEAP, AM_NA, 1, "kind=%s obj=0x%llx %s", kind == 1 ? "buf" : "img",
                (unsigned long long)h, known ? "unbound" : "untracked");
}

// bc-misc: copies/blits/resolves touching an emulated-BC image (idx: counter for a BC source / destination, -1 none).
static void am_bcmisc(const char *call, VkCommandBuffer cb, uint64_t src, int si, uint64_t dst, int di, uint32_t regions) {
    struct AmObj s = {0}, t = {0};
    bool sbc = si >= 0 && am_obj_get(src, &s, NULL, false) && s.bcfmt;
    bool dbc = di >= 0 && am_obj_get(dst, &t, NULL, false) && t.bcfmt;
    if (!sbc && !dbc) return;
    if (sbc) atomic_fetch_add_explicit(&g_am_bcmisc[si], 1, memory_order_relaxed);
    if (dbc) atomic_fetch_add_explicit(&g_am_bcmisc[di], 1, memory_order_relaxed);
    if (atomic_fetch_add_explicit(&g_am_c_events, 1, memory_order_relaxed) < 16)
        am_emit('C', 0, AM_NA, ML_NONE, BUDGET_NO_HEAP, AM_NA, 0, "call=%s cb=0x%llx src=0x%llx srcfmt=%d srcbc=%d dst=0x%llx dstfmt=%d dstbc=%d "
                "regions=%u", call, (unsigned long long)(uintptr_t)cb, (unsigned long long)src, s.fmt, s.bcfmt,
                (unsigned long long)dst, t.fmt, t.bcfmt, regions);
}
// vkCreateImageView on an emulated-BC image with a view format other than the image's BC format.
static void am_view(const VkImageViewCreateInfo *ci, VkResult r, VkImageView v) {
    struct AmObj o;
    if (!am_obj_get((uint64_t)(uintptr_t)ci->image, &o, NULL, false) || !o.bcfmt || (int32_t)ci->format == o.bcfmt) return;
    bool vbc = panvk_format_is_bc(ci->format);
    atomic_fetch_add_explicit(&g_am_bcmisc[vbc ? 7 : 8], 1, memory_order_relaxed);
    if (atomic_fetch_add_explicit(&g_am_c_events, 1, memory_order_relaxed) < 16)
        am_emit('C', 0, AM_NA, ML_NONE, BUDGET_NO_HEAP, AM_NA, 0, "call=CreateImageView img=0x%llx view=0x%llx bcfmt=%d imgfmt=%d "
                "viewfmt=%d driverviewfmt=%d result=%d", (unsigned long long)(uintptr_t)ci->image,
                (unsigned long long)(uintptr_t)v, o.bcfmt, o.fmt, ci->format,
                vbc ? (int)panvk_bc_decoded_format(ci->format) : (int)ci->format, r);
}

static VKAPI_ATTR void VKAPI_CALL compat_GetDeviceImageMemoryRequirements(
    VkDevice device, const VkDeviceImageMemoryRequirements *pInfo, VkMemoryRequirements2 *pMemoryRequirements) {
    struct DeviceData *d = get_device_data(device);
    struct image_fixup fx;
    fixup_begin(d, pInfo->pCreateInfo, &fx);
    VkDeviceImageMemoryRequirements info = *pInfo;
    info.pCreateInfo = fx.use;
    d->fpGetDeviceImageMemoryRequirements(device, &info, pMemoryRequirements);
    fixup_end(&fx);
}

static VKAPI_ATTR void VKAPI_CALL compat_GetDeviceImageSparseMemoryRequirements(
    VkDevice device, const VkDeviceImageMemoryRequirements *pInfo, uint32_t *pCount,
    VkSparseImageMemoryRequirements2 *pReqs) {
    struct DeviceData *d = get_device_data(device);
    struct image_fixup fx;
    fixup_begin(d, pInfo->pCreateInfo, &fx);
    VkDeviceImageMemoryRequirements info = *pInfo;
    info.pCreateInfo = fx.use;
    d->fpGetDeviceImageSparseMemoryRequirements(device, &info, pCount, pReqs);
    fixup_end(&fx);
}

static VKAPI_ATTR VkResult VKAPI_CALL compat_CreateImage(
    VkDevice device, const VkImageCreateInfo *pCreateInfo,
    const VkAllocationCallbacks *pAllocator, VkImage *pImage) {
    struct DeviceData *d = get_device_data(device);
    // dma-buf export (MALI_COMPAT_DMABUF_EXPORT): a modifier list becomes an explicit linear layout the driver imports
    VkImageCreateInfo dci;
    VkImageDrmFormatModifierExplicitCreateInfoEXT expl;
    VkSubresourceLayout plane0;
    struct unlinked u_list = {0}, u_wsi = {0};
    if (g_dmabuf_export) {
        u_wsi = unlink_struct((void *)pCreateInfo, STYPE_WSI_IMAGE_CREATE_INFO_MESA);
        const VkExternalMemoryImageCreateInfo *em = find_struct(pCreateInfo->pNext, VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO);
        const VkImageDrmFormatModifierListCreateInfoEXT *ml =
            find_struct(pCreateInfo->pNext, VK_STRUCTURE_TYPE_IMAGE_DRM_FORMAT_MODIFIER_LIST_CREATE_INFO_EXT);
        uint32_t bpp = format_bytes(pCreateInfo->format);
        bool has_linear = false;
        for (uint32_t i = 0; ml && i < ml->drmFormatModifierCount; i++) has_linear |= ml->pDrmFormatModifiers[i] == 0;
        if (em && getenv("MALI_COMPAT_DMABUF_PEEK")) {
            // diagnostic: how the external images of this process are made (Xwayland/Zink vs gamescope)
            static _Atomic int logged;
            const VkImageDrmFormatModifierExplicitCreateInfoEXT *ex0 =
                find_struct(pCreateInfo->pNext, VK_STRUCTURE_TYPE_IMAGE_DRM_FORMAT_MODIFIER_EXPLICIT_CREATE_INFO_EXT);
            if (atomic_fetch_add(&logged, 1) < 24)
                fprintf(stderr, "mali-compat: external image: handles 0x%x fmt %d %ux%u tiling %d usage 0x%x flags 0x%x, "
                        "modifier list %u (linear %d), explicit %s %llx pitch %llu\n", em->handleTypes, pCreateInfo->format,
                        pCreateInfo->extent.width, pCreateInfo->extent.height, pCreateInfo->tiling, pCreateInfo->usage,
                        pCreateInfo->flags, ml ? ml->drmFormatModifierCount : 0, has_linear, ex0 ? "yes" : "no",
                        ex0 ? (unsigned long long)ex0->drmFormatModifier : 0ull,
                        ex0 && ex0->pPlaneLayouts ? (unsigned long long)ex0->pPlaneLayouts[0].rowPitch : 0ull);
        }
        if (em && (em->handleTypes & VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT) && ml && has_linear && bpp &&
            pCreateInfo->tiling == VK_IMAGE_TILING_DRM_FORMAT_MODIFIER_EXT) {
            u_list = unlink_struct((void *)pCreateInfo, VK_STRUCTURE_TYPE_IMAGE_DRM_FORMAT_MODIFIER_LIST_CREATE_INFO_EXT);
            plane0 = (VkSubresourceLayout){0, 0, ((VkDeviceSize)pCreateInfo->extent.width * bpp + 255) & ~255ull, 0, 0};
            expl = (VkImageDrmFormatModifierExplicitCreateInfoEXT){
                VK_STRUCTURE_TYPE_IMAGE_DRM_FORMAT_MODIFIER_EXPLICIT_CREATE_INFO_EXT, pCreateInfo->pNext, 0, 1, &plane0};
            dci = *pCreateInfo;
            dci.pNext = &expl;
            pCreateInfo = &dci;
        }
    }
    struct image_fixup fx;
    fixup_begin(d, pCreateInfo, &fx);
    bool is_bc = fx.is_bc;
    VkResult res = d->fpCreateImage(device, fx.use, pAllocator, pImage);
    if (res != VK_SUCCESS && is_yuv_nv12(pCreateInfo->format)) {
        VkImageCreateInfo fallback_ci = *pCreateInfo;
        fallback_ci.format = VK_FORMAT_R8_UNORM;
        fallback_ci.extent.height = (pCreateInfo->extent.height * 3) / 2;
        fallback_ci.flags &= ~(VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT | VK_IMAGE_CREATE_EXTENDED_USAGE_BIT);
        res = d->fpCreateImage(device, &fallback_ci, pAllocator, pImage);
        if (res == VK_SUCCESS) {
            fprintf(stderr, "mali-compat: NV12 video image fallback to R8 (%ux%u) succeeded\n",
                    fallback_ci.extent.width, fallback_ci.extent.height);
        }
    }
    if (res != VK_SUCCESS) {
        // [Claude Opus 5.5] always logged: which create info the driver refused (DXVK only prints size/format).
        const VkImageCreateInfo *ci = fx.use;
        fprintf(stderr, "mali-compat: vkCreateImage -> %d: flags 0x%x fmt %d %ux%ux%u mips %u layers %u samples %d "
                "tiling %d usage 0x%x sharing %d layout %d\n", res, ci->flags, ci->format, ci->extent.width,
                ci->extent.height, ci->extent.depth, ci->mipLevels, ci->arrayLayers, ci->samples, ci->tiling,
                ci->usage, ci->sharingMode, ci->initialLayout);
        for (const VkBaseInStructure *s = ci->pNext; s; s = s->pNext) {
            fprintf(stderr, "mali-compat:   pNext sType %d\n", s->sType);
            if (s->sType == VK_STRUCTURE_TYPE_IMAGE_FORMAT_LIST_CREATE_INFO) {
                const VkImageFormatListCreateInfo *fl = (const VkImageFormatListCreateInfo *)s;
                for (uint32_t v = 0; v < fl->viewFormatCount; v++)
                    fprintf(stderr, "mali-compat:     view format %d\n", fl->pViewFormats[v]);
            } else if (s->sType == VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO) {
                fprintf(stderr, "mali-compat:     handle types 0x%x\n", ((const VkExternalMemoryImageCreateInfo *)s)->handleTypes);
            }
        }
    }
    if (res == VK_SUCCESS && g_log_images && pImage && *pImage) log_image(d, device, fx.use, *pImage, is_bc);
    if (res == VK_SUCCESS && (g_gpu_time || g_ct) && pImage && *pImage) // usage as the driver got it (RT_COMPRESS may strip STORAGE)
        img_put(g_img_info, (struct ImgNode){(uint64_t)*pImage, *pImage, pCreateInfo->format, pCreateInfo->extent.width,
                pCreateInfo->extent.height, 0, pCreateInfo->mipLevels, 0, pCreateInfo->arrayLayers, pCreateInfo->samples,
                fx.use->usage, NULL});
    if (res == VK_SUCCESS && g_rt_compress && pImage && *pImage && (pCreateInfo->usage & VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT) &&
        (pCreateInfo->usage & VK_IMAGE_USAGE_STORAGE_BIT) && !is_bc) {
        if (fx.rt_stripped) {
            rt_track(*pImage, rt_sig(pCreateInfo), fx.mod.usage);
            atomic_fetch_add(&g_rt_stripped, 1);
        } else {
            atomic_fetch_add(&g_rt_kept, 1);
        }
    }
    if (res == VK_SUCCESS && is_bc && pImage && *pImage) {
        register_bc_image(*pImage, pCreateInfo->format, fx.mod.format, pCreateInfo->extent);
        LOG("emulating BC image %p: orig=%d -> decoded=%d (%ux%u)",
            (void *)*pImage, pCreateInfo->format, fx.mod.format, pCreateInfo->extent.width, pCreateInfo->extent.height);
    }
    if (g_addrmap && res == VK_SUCCESS && pImage && *pImage) am_image_created(*pImage, pCreateInfo, &fx); // before fixup_end
    fixup_end(&fx);
    relink(u_list);
    relink(u_wsi);
    return res;
}

static VKAPI_ATTR void VKAPI_CALL compat_DestroyImage(
    VkDevice device, VkImage image, const VkAllocationCallbacks *pAllocator) {
    struct DeviceData *d = get_device_data(device);
    if (g_addrmap && image) am_destroy(2, (uint64_t)(uintptr_t)image); // before the driver: the handle may be reused
    unregister_bc_image(image);
    if (g_rt_compress) rt_untrack(image);
    d->fpDestroyImage(device, image, pAllocator);
}

static VkResult create_image_view(
    VkDevice device, const VkImageViewCreateInfo *pCreateInfo,
    const VkAllocationCallbacks *pAllocator, VkImageView *pView) {
    struct DeviceData *d = get_device_data(device);

    struct TrackedBCImage tracked;
    if (find_bc_image(pCreateInfo->image, &tracked) && panvk_format_is_bc(pCreateInfo->format)) {
        VkImageViewCreateInfo mod = *pCreateInfo;
        mod.format = panvk_bc_decoded_format(pCreateInfo->format);
        return d->fpCreateImageView(device, &mod, pAllocator, pView);
    }

    // A storage view on a target created without STORAGE: learn it, and keep the view itself valid (DXVK chains the
    // view's own usage; a view usage the image lacks is invalid) by asking for the image's other usages instead.
    struct RTNode rt;
    VkImageViewUsageCreateInfo *vu;
    if (g_rt_compress && rt_find(pCreateInfo->image, &rt) &&
        (vu = (VkImageViewUsageCreateInfo *)find_struct(pCreateInfo->pNext, VK_STRUCTURE_TYPE_IMAGE_VIEW_USAGE_CREATE_INFO)) &&
        (vu->usage & VK_IMAGE_USAGE_STORAGE_BIT)) {
        rt_learn(rt.sig);
        VkImageUsageFlags saved = vu->usage;
        vu->usage = (saved & ~VK_IMAGE_USAGE_STORAGE_BIT) & rt.usage;
        if (!vu->usage) vu->usage = rt.usage & (VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT);
        VkResult res = d->fpCreateImageView(device, pCreateInfo, pAllocator, pView);
        vu->usage = saved;
        return res;
    }

    return d->fpCreateImageView(device, pCreateInfo, pAllocator, pView);
}

// GPU timing: remembers what each view shows, for the STUCK/TRACE details.
static VKAPI_ATTR VkResult VKAPI_CALL compat_CreateImageView(
    VkDevice device, const VkImageViewCreateInfo *pCreateInfo,
    const VkAllocationCallbacks *pAllocator, VkImageView *pView) {
    VkResult res = create_image_view(device, pCreateInfo, pAllocator, pView);
    if (g_addrmap) am_view(pCreateInfo, res, res == VK_SUCCESS && pView ? *pView : VK_NULL_HANDLE);
    if ((g_gpu_time || g_ct) && res == VK_SUCCESS && pView) {
        struct ImgNode img = {0};
        img_get(g_img_info, (uint64_t)pCreateInfo->image, &img);
        const VkImageSubresourceRange *r = &pCreateInfo->subresourceRange;
        img_put(g_view_info, (struct ImgNode){(uint64_t)*pView, pCreateInfo->image, pCreateInfo->format, img.w, img.h,
                r->baseMipLevel, r->levelCount, r->baseArrayLayer, r->layerCount, img.samples, r->aspectMask, NULL});
    }
    return res;
}

static VKAPI_ATTR VkResult VKAPI_CALL compat_CreateBufferView(
    VkDevice device, const VkBufferViewCreateInfo *pCreateInfo, const VkAllocationCallbacks *pAllocator, VkBufferView *pView) {
    struct DeviceData *d = get_device_data(device);
    static _Atomic int misaligned;
    if (pCreateInfo->offset % 64) {
        int n = atomic_fetch_add(&misaligned, 1);
        if (n < 8 || (n & (n - 1)) == 0) // the first 8, then powers of two
            fprintf(stderr, "mali-compat: buffer view %d with offset %llu (not 64-aligned), format %d, range %llu\n", n + 1,
                    (unsigned long long)pCreateInfo->offset, pCreateInfo->format, (unsigned long long)pCreateInfo->range);
    }
    VkResult r = d->fpCreateBufferView(device, pCreateInfo, pAllocator, pView);
    if (g_dc_on && r == VK_SUCCESS) dc_view_create(*pView, pCreateInfo);
    return r;
}

// MALI_COMPAT_FAKE_OCCLUSION=1: every occlusion query result reads as "visible" (0x10000 samples, available), for the
// GPU copy (vkCmdFillBuffer instead of vkCmdCopyQueryPoolResults) and the CPU read. A test for occlusion results the
// driver gets wrong (HZD statics missing): if the statics come back, the game culls them on query results.
// [Claude Opus 5.5] 2026-10-29
static bool g_fake_occlusion;
static bool g_force_coherent; // MALI_COMPAT_FORCE_COHERENT, see compat_AllocateMemory
static _Atomic uint64_t g_occl_copies, g_occl_reads;
#define OCCL_BUCKETS 64
static VkQueryPool g_occl_pools[OCCL_BUCKETS][16];
static pthread_mutex_t g_occl_mutex = PTHREAD_MUTEX_INITIALIZER;

static bool occl_pool(VkQueryPool pool, bool add, bool remove) {
    uint32_t b = handle_bucket((uint64_t)pool, OCCL_BUCKETS);
    bool found = false;
    pthread_mutex_lock(&g_occl_mutex);
    for (int i = 0; i < 16; i++) {
        if (g_occl_pools[b][i] == pool) { found = true; if (remove) g_occl_pools[b][i] = VK_NULL_HANDLE; break; }
        if (add && !g_occl_pools[b][i]) { g_occl_pools[b][i] = pool; add = false; }
    }
    pthread_mutex_unlock(&g_occl_mutex);
    if (add) fprintf(stderr, "mali-compat: fake occlusion: pool table full\n");
    return found;
}

static VKAPI_ATTR VkResult VKAPI_CALL compat_CreateQueryPool(VkDevice device, const VkQueryPoolCreateInfo *pCreateInfo,
                                                            const VkAllocationCallbacks *pAllocator, VkQueryPool *pPool) {
    struct DeviceData *d = get_device_data(device);
    VkResult r = d->fpCreateQueryPool(device, pCreateInfo, pAllocator, pPool);
    if (r == VK_SUCCESS && g_fake_occlusion && pCreateInfo->queryType == VK_QUERY_TYPE_OCCLUSION) occl_pool(*pPool, true, false);
    if (g_addrmap) // handed out for ADDRMAP alone too (then the occlusion table above stays untouched)
        am_emit('N', r == VK_SUCCESS && pPool ? (uint64_t)*pPool : 0, AM_NA, ML_NONE, BUDGET_NO_HEAP, AM_NA, 0,
                "what=qpool-create type=%d count=%u stats=0x%x result=%d", pCreateInfo->queryType, pCreateInfo->queryCount,
                pCreateInfo->pipelineStatistics, r);
    return r;
}

static VKAPI_ATTR void VKAPI_CALL compat_DestroyQueryPool(VkDevice device, VkQueryPool pool, const VkAllocationCallbacks *pAllocator) {
    struct DeviceData *d = get_device_data(device);
    if (pool && g_fake_occlusion) occl_pool(pool, false, true);
    if (g_addrmap && pool) am_emit('N', (uint64_t)pool, AM_NA, ML_NONE, BUDGET_NO_HEAP, AM_NA, 0, "what=qpool-destroy");
    d->fpDestroyQueryPool(device, pool, pAllocator);
}

static VKAPI_ATTR void VKAPI_CALL compat_CmdCopyQueryPoolResults(VkCommandBuffer cmd, VkQueryPool pool, uint32_t first,
    uint32_t count, VkBuffer dst, VkDeviceSize offset, VkDeviceSize stride, VkQueryResultFlags flags) {
    struct DeviceData *d = get_device_data_by_cmd(cmd);
    if (!occl_pool(pool, false, false) || !count) {
        d->fpCmdCopyQueryPoolResults(cmd, pool, first, count, dst, offset, stride, flags);
        return;
    }
    atomic_fetch_add(&g_occl_copies, 1);
    VkDeviceSize elem = (flags & VK_QUERY_RESULT_64_BIT ? 8 : 4) * (flags & VK_QUERY_RESULT_WITH_AVAILABILITY_BIT ? 2 : 1);
    if (stride == elem) d->fpCmdFillBuffer(cmd, dst, offset, elem * count, 0x10000);
    else
        for (uint32_t i = 0; i < count; i++) d->fpCmdFillBuffer(cmd, dst, offset + i * stride, elem, 0x10000);
}

static VKAPI_ATTR VkResult VKAPI_CALL compat_GetQueryPoolResults(VkDevice device, VkQueryPool pool, uint32_t first,
    uint32_t count, size_t size, void *data, VkDeviceSize stride, VkQueryResultFlags flags) {
    struct DeviceData *d = get_device_data(device);
    if (!occl_pool(pool, false, false)) return d->fpGetQueryPoolResults(device, pool, first, count, size, data, stride, flags);
    atomic_fetch_add(&g_occl_reads, 1);
    uint32_t words = (flags & VK_QUERY_RESULT_64_BIT ? 2 : 1) * (flags & VK_QUERY_RESULT_WITH_AVAILABILITY_BIT ? 2 : 1);
    for (uint32_t i = 0; i < count; i++)
        for (uint32_t w = 0; w < words; w++) ((uint32_t *)((char *)data + i * stride))[w] = 0x10000;
    return VK_SUCCESS;
}

static VKAPI_ATTR VkResult VKAPI_CALL compat_CreateBuffer(
    VkDevice device, const VkBufferCreateInfo *pCreateInfo,
    const VkAllocationCallbacks *pAllocator, VkBuffer *pBuffer) {
    struct DeviceData *d = get_device_data(device);

    VkBufferCreateInfo mod = *pCreateInfo;
    if (mod.usage & VK_BUFFER_USAGE_TRANSFER_SRC_BIT) {
        mod.usage |= VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
    }
    bool as_xfer = false;   /* lead-1110c: an AS storage buffer that got TRANSFER_SRC|TRANSFER_DST (layer CLONE) */
    bool as_buf = false;    /* round 4: any AS storage buffer (counted in rt_mem asbuf=) */
    if (g_rt_pipeline) {
        if (mod.usage & VK_BUFFER_USAGE_SHADER_BINDING_TABLE_BIT_KHR) {
            mod.usage &= ~VK_BUFFER_USAGE_SHADER_BINDING_TABLE_BIT_KHR;
            mod.usage |= VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                         VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
                         VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        }
        if (mod.usage & (VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT)) {
            mod.usage |= VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
        }
    }
    if (g_as_hooks) {   /* rt-gate-1112: the AS part, also with MALI_COMPAT_AS_FIX=1 alone */
        /* lead-1110c: the layer clones ASes with vkCmdCopyBuffer (mali_rt_emu.h compat_CmdCopyAccelerationStructureKHR):
         * every AS storage buffer gets both transfer bits (after the rules above, so they add nothing else). With a
         * VkBufferUsageFlags2CreateInfo in pNext (maintenance5: it replaces .usage) the buffer is left as it is and is
         * recorded only if it already has both bits; an unrecorded buffer makes a CLONE fall back to the driver. */
        const VkBufferUsageFlags2CreateInfoKHR *u2 = (const VkBufferUsageFlags2CreateInfoKHR *)
            find_struct(mod.pNext, VK_STRUCTURE_TYPE_BUFFER_USAGE_FLAGS_2_CREATE_INFO_KHR);
        const VkBufferUsageFlags2KHR xfer2 = VK_BUFFER_USAGE_2_TRANSFER_SRC_BIT_KHR | VK_BUFFER_USAGE_2_TRANSFER_DST_BIT_KHR;
        if (u2) {
            as_buf = (u2->usage & VK_BUFFER_USAGE_2_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR) != 0;
            as_xfer = as_buf && (u2->usage & xfer2) == xfer2;
        } else if (mod.usage & VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR) {
            mod.usage |= VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
            as_xfer = true;
            as_buf = true;
        }
    }

    VkResult r = d->fpCreateBuffer(device, &mod, pAllocator, pBuffer);
    if (g_addrmap && r == VK_SUCCESS) am_buffer_created(*pBuffer, pCreateInfo, &mod);
    if (g_dc_on && r == VK_SUCCESS) dc_buf_create(d, *pBuffer, &mod);
    if (as_buf && r == VK_SUCCESS) rt_as_buf_add(*pBuffer, mod.size, as_xfer);   /* round 4: + size, all AS buffers */
    return r;
}

/* lead-1110c: only with MALI_COMPAT_RT_PIPELINE (rt-gate-1112: or MALI_COMPAT_AS_FIX): forgets the buffer in the layer's AS-transfer set (a reused handle
 * must not inherit the record), then the normal destroy (desc-capture's when it is on). */
static VKAPI_ATTR void VKAPI_CALL compat_RtDestroyBuffer(VkDevice device, VkBuffer b, const VkAllocationCallbacks *a) {
    if (b) rt_xfer_buf_remove(b);
    if (g_dc_on) { compat_DcDestroyBuffer(device, b, a); return; }
    get_device_data(device)->fpDestroyBuffer(device, b, a);
}

// -----------------------------------------------------------------------------
// VK_EXT_map_memory_placed, emulated. [Claude Opus 5.5] 2026-10-26
// Wine on WoW64 needs every mapping below 4 GiB. Without this extension it falls back to importing host pointers,
// which the vendor driver only takes for its non-coherent memory type, so DXVK's coherent mappings land above 4 GiB
// ("does not fit 32-bit pointer"). Here the driver maps as usual and the mapping is moved with mremap onto the
// address Wine reserved; the driver's old range is held PROT_NONE so the move back before the driver unmaps cannot
// hit anything else (test/map-probe.c: CPU and GPU see the same data through the moved mapping, also after a remap).
// The driver's flush/invalidate work on its own address, so for moved mappings the layer does the cache maintenance.
// -----------------------------------------------------------------------------
static bool mem_add(VkDeviceMemory m, VkDeviceSize size, uint32_t type, uint32_t heap) {
    struct MemInfo *e = calloc(1, sizeof *e);
    if (!e) return false; // no entry = never counted, so the free does not subtract either
    e->memory = m;
    e->size = size;
    e->type = type;
    e->heap = heap;
    budget_add(heap, size);
    pthread_mutex_lock(&g_mem_mutex);
    uint32_t b = mem_bucket(m);
    e->next = g_mem[b];
    g_mem[b] = e;
    pthread_mutex_unlock(&g_mem_mutex);
    return true;
}

static struct MemInfo *mem_remove(VkDeviceMemory m) {
    pthread_mutex_lock(&g_mem_mutex);
    struct MemInfo **pp = &g_mem[mem_bucket(m)];
    while (*pp && (*pp)->memory != m) pp = &(*pp)->next;
    struct MemInfo *e = *pp;
    if (e) *pp = e->next;
    pthread_mutex_unlock(&g_mem_mutex);
    return e;
}

static void *hold_range(void *addr, size_t size, bool replace) {
    return mmap(addr, size, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE | (replace ? MAP_FIXED : MAP_FIXED_NOREPLACE),
                -1, 0);
}

// Moves a placed mapping back onto the driver's held range. Called with g_mem_mutex held; e->orig stays set.
static bool placed_move_back(struct MemInfo *e) {
    void *r = mremap(e->placed, e->map_size, e->map_size, MREMAP_MAYMOVE | MREMAP_FIXED, e->orig);
    if (r == MAP_FAILED) {
        fprintf(stderr, "mali-compat: placed map: moving %p back to %p failed: %s\n", e->placed, e->orig, strerror(errno));
        return false;
    }
    return true;
}

// memlog: M/U events with the allocation's size/type/heap; the va is kept for the F event.
static void ml_mapped(VkDeviceMemory memory, uint64_t va) {
    uint64_t size = 0;
    uint32_t type = ML_NONE, heap = BUDGET_NO_HEAP;
    pthread_mutex_lock(&g_mem_mutex);
    struct MemInfo *e = mem_find(memory);
    if (e) { e->va = va; size = e->size; type = e->type; heap = e->heap; }
    pthread_mutex_unlock(&g_mem_mutex);
    ml_event('M', (uint64_t)(uintptr_t)memory, size, type, heap, va, 1);
}

static void ml_unmapped(VkDeviceMemory memory) {
    uint64_t size = 0, va = 0;
    uint32_t type = ML_NONE, heap = BUDGET_NO_HEAP;
    pthread_mutex_lock(&g_mem_mutex);
    struct MemInfo *e = mem_find(memory);
    if (e) { va = e->va; size = e->size; type = e->type; heap = e->heap; }
    pthread_mutex_unlock(&g_mem_mutex);
    ml_event('U', (uint64_t)(uintptr_t)memory, size, type, heap, va, va ? 1 : 0);
}

// vkMapMemory, handed out only with MALI_COMPAT_MEMLOG (64-bit Wine maps with it, not with map2).
static VKAPI_ATTR VkResult VKAPI_CALL compat_MapMemory(VkDevice device, VkDeviceMemory memory, VkDeviceSize offset,
                                                      VkDeviceSize size, VkMemoryMapFlags flags, void **ppData) {
    struct DeviceData *d = get_device_data(device);
    VkResult r = d->fpMapMemory(device, memory, offset, size, flags, ppData);
    if (g_memlog && r == VK_SUCCESS && ppData && *ppData) ml_mapped(memory, (uint64_t)(uintptr_t)*ppData - offset);
    return r;
}

static VKAPI_ATTR VkResult VKAPI_CALL compat_MapMemory2KHR(VkDevice device, const VkMemoryMapInfoKHR *pMemoryMapInfo,
                                                          void **ppData) {
    struct DeviceData *d = get_device_data(device);
    const VkMemoryMapPlacedInfoEXT *pl = find_struct(pMemoryMapInfo->pNext, VK_STRUCTURE_TYPE_MEMORY_MAP_PLACED_INFO_EXT);
    if (!(pMemoryMapInfo->flags & VK_MEMORY_MAP_PLACED_BIT_EXT) || !pl || !pl->pPlacedAddress) {
        VkResult r0 = !d->fpMapMemory2KHR // VK_KHR_map_memory2 not enabled below (see vkGetDeviceProcAddr)
            ? d->fpMapMemory(device, pMemoryMapInfo->memory, pMemoryMapInfo->offset, pMemoryMapInfo->size, 0, ppData)
            : d->fpMapMemory2KHR(device, pMemoryMapInfo, ppData);
        if (g_memlog && r0 == VK_SUCCESS && ppData && *ppData)
            ml_mapped(pMemoryMapInfo->memory, (uint64_t)(uintptr_t)*ppData - pMemoryMapInfo->offset);
        return r0;
    }
    if (!d->fpMapMemory2KHR) return VK_ERROR_MEMORY_MAP_FAILED;

    VkMemoryMapInfoKHR info = *pMemoryMapInfo;
    info.flags &= ~VK_MEMORY_MAP_PLACED_BIT_EXT;
    struct unlinked u = unlink_struct(&info, VK_STRUCTURE_TYPE_MEMORY_MAP_PLACED_INFO_EXT);
    void *p = NULL;
    VkResult r = d->fpMapMemory2KHR(device, &info, &p);
    relink(u);
    if (r != VK_SUCCESS) return r;

    pthread_mutex_lock(&g_mem_mutex);
    struct MemInfo *e = mem_find(info.memory);
    VkDeviceSize size = info.size == VK_WHOLE_SIZE && e ? e->size - info.offset : info.size;
    size_t map_size = (size + page_size() - 1) & ~(page_size() - 1);
    if (!e || ((uintptr_t)p & (page_size() - 1)) || ((uintptr_t)pl->pPlacedAddress & (page_size() - 1))) {
        pthread_mutex_unlock(&g_mem_mutex);
        fprintf(stderr, "mali-compat: placed map: unknown memory or unaligned mapping %p -> %p\n", p, pl->pPlacedAddress);
        d->fpUnmapMemory(device, info.memory);
        return VK_ERROR_MEMORY_MAP_FAILED;
    }
    void *moved = mremap(p, map_size, map_size, MREMAP_MAYMOVE | MREMAP_FIXED, pl->pPlacedAddress);
    if (moved == MAP_FAILED) {
        pthread_mutex_unlock(&g_mem_mutex);
        fprintf(stderr, "mali-compat: placed map: mremap %p -> %p (%zu bytes) failed: %s\n", p, pl->pPlacedAddress,
                map_size, strerror(errno));
        d->fpUnmapMemory(device, info.memory);
        return VK_ERROR_MEMORY_MAP_FAILED;
    }
    if (hold_range(p, map_size, false) == MAP_FAILED)
        fprintf(stderr, "mali-compat: placed map: could not hold %p: %s\n", p, strerror(errno));
    e->orig = p;
    e->placed = moved;
    e->map_size = map_size;
    pthread_mutex_unlock(&g_mem_mutex);
    LOG("placed map %p: driver %p -> %p (%zu bytes, type %u)", (void *)info.memory, p, moved, map_size, e->type);
    if (g_memlog) ml_mapped(info.memory, (uint64_t)(uintptr_t)p - info.offset); // the driver's address = GPU VA
    *ppData = moved;
    return VK_SUCCESS;
}

// Unmaps a placed mapping: back onto the held range, driver unmap, then the app's range is re-reserved if asked.
// Returns false when `memory` was not placed (the caller unmaps normally).
static bool placed_unmap(struct DeviceData *d, VkDevice device, VkDeviceMemory memory, bool reserve,
                         const VkMemoryUnmapInfoKHR *info2) {
    pthread_mutex_lock(&g_mem_mutex);
    struct MemInfo *e = mem_find(memory);
    if (!e || !e->orig) {
        pthread_mutex_unlock(&g_mem_mutex);
        return false;
    }
    bool back = placed_move_back(e);
    if (back) {
        if (info2 && d->fpUnmapMemory2KHR) {
            VkMemoryUnmapInfoKHR mod = *info2;
            mod.flags &= ~VK_MEMORY_UNMAP_RESERVE_BIT_EXT;
            d->fpUnmapMemory2KHR(device, &mod);
        } else {
            d->fpUnmapMemory(device, memory);
        }
    }
    if (reserve || !back) hold_range(e->placed, e->map_size, true);
    else munmap(e->placed, e->map_size);
    e->orig = e->placed = NULL;
    pthread_mutex_unlock(&g_mem_mutex);
    return true;
}

static VKAPI_ATTR VkResult VKAPI_CALL compat_UnmapMemory2KHR(VkDevice device, const VkMemoryUnmapInfoKHR *pMemoryUnmapInfo) {
    struct DeviceData *d = get_device_data(device);
    if (g_memlog) ml_unmapped(pMemoryUnmapInfo->memory);
    if (placed_unmap(d, device, pMemoryUnmapInfo->memory, pMemoryUnmapInfo->flags & VK_MEMORY_UNMAP_RESERVE_BIT_EXT,
                     pMemoryUnmapInfo))
        return VK_SUCCESS;
    if (!d->fpUnmapMemory2KHR) { d->fpUnmapMemory(device, pMemoryUnmapInfo->memory); return VK_SUCCESS; }
    return d->fpUnmapMemory2KHR(device, pMemoryUnmapInfo);
}

static VKAPI_ATTR void VKAPI_CALL compat_UnmapMemory(VkDevice device, VkDeviceMemory memory) {
    struct DeviceData *d = get_device_data(device);
    if (g_memlog) ml_unmapped(memory);
    if (!placed_unmap(d, device, memory, false, NULL)) d->fpUnmapMemory(device, memory);
}

// Placed ranges get their cache maintenance here; the rest goes to the driver.
static VkResult placed_ranges(struct DeviceData *d, VkDevice device, uint32_t count, const VkMappedMemoryRange *ranges,
                              bool invalidate) {
    VkMappedMemoryRange stack[16];
    VkMappedMemoryRange *rest = count <= 16 ? stack : malloc(count * sizeof *rest);
    if (!rest) return VK_ERROR_OUT_OF_HOST_MEMORY;
    uint32_t n = 0;
    pthread_mutex_lock(&g_mem_mutex);
    for (uint32_t i = 0; i < count; i++) {
        struct MemInfo *e = mem_find(ranges[i].memory);
        if (e && e->orig) {
            if (!(d->coherent_types & (1u << e->type))) {
                VkDeviceSize size = ranges[i].size == VK_WHOLE_SIZE ? e->map_size - ranges[i].offset : ranges[i].size;
                cache_maintenance((char *)e->placed + ranges[i].offset, size, invalidate);
            }
        } else {
            rest[n++] = ranges[i];
        }
    }
    pthread_mutex_unlock(&g_mem_mutex);
    VkResult r = VK_SUCCESS;
    if (n) r = invalidate ? d->fpInvalidateMappedMemoryRanges(device, n, rest) : d->fpFlushMappedMemoryRanges(device, n, rest);
    if (rest != stack) free(rest);
    return r;
}

static VKAPI_ATTR VkResult VKAPI_CALL compat_FlushMappedMemoryRanges(VkDevice device, uint32_t count,
                                                                     const VkMappedMemoryRange *ranges) {
    return placed_ranges(get_device_data(device), device, count, ranges, false);
}

static VKAPI_ATTR VkResult VKAPI_CALL compat_InvalidateMappedMemoryRanges(VkDevice device, uint32_t count,
                                                                          const VkMappedMemoryRange *ranges) {
    return placed_ranges(get_device_data(device), device, count, ranges, true);
}

static VKAPI_ATTR VkResult VKAPI_CALL compat_AllocateMemory(
    VkDevice device, const VkMemoryAllocateInfo *pAllocateInfo,
    const VkAllocationCallbacks *pAllocator, VkDeviceMemory *pMemory) {
    struct DeviceData *d = get_device_data(device);
    // K-CF2: an app-supplied dma-buf fd import is the exporter's memory, so the budget does not count it
    const bool fd_import = find_struct(pAllocateInfo->pNext, VK_STRUCTURE_TYPE_IMPORT_MEMORY_FD_INFO_KHR) != NULL;

    // dma-buf export (MALI_COMPAT_DMABUF_EXPORT): the memory comes from the dma-heap and is imported into the driver
    struct unlinked u_exp = {0}, u_mwsi = {0};
    VkImportMemoryFdInfoKHR imp;
    int export_fd = -1, keep_import_fd = -1;
    VkMemoryAllocateInfo emod;
    if (g_dmabuf_export) {
        u_mwsi = unlink_struct((void *)pAllocateInfo, STYPE_WSI_MEMORY_ALLOCATE_INFO_MESA);
        const VkExportMemoryAllocateInfo *ex = find_struct(pAllocateInfo->pNext, VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO);
        const VkImportMemoryFdInfoKHR *imp_in = find_struct(pAllocateInfo->pNext, VK_STRUCTURE_TYPE_IMPORT_MEMORY_FD_INFO_KHR);
        if (ex && (ex->handleTypes & VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT) && imp_in &&
            imp_in->handleType == VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT) {
            // An import that also asks to be exportable (Mesa Zink's allocate_bo puts both structs on a re-imported
            // GBM buffer): keep the import, drop the export struct the driver refuses, and remember the imported fd
            // so vkGetMemoryFdKHR hands out that same buffer. Allocating a fresh heap buffer here instead put Zink's
            // rendering into memory nobody shared: Xwayland drew Steam into it while gamescope composited the
            // untouched original (black session screen, 2026-10-30; found by a Fable review of Zink's source).
            u_exp = unlink_struct((void *)pAllocateInfo, VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO);
            keep_import_fd = dup(imp_in->fd);
        } else if (ex && (ex->handleTypes & VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT) && d->fpGetMemoryFdKHR &&
            (export_fd = dmaheap_alloc(pAllocateInfo->allocationSize)) >= 0) {
            u_exp = unlink_struct((void *)pAllocateInfo, VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO);
            emod = *pAllocateInfo;
            VkMemoryFdPropertiesKHR fp = {VK_STRUCTURE_TYPE_MEMORY_FD_PROPERTIES_KHR};
            if (d->fpGetMemoryFdPropertiesKHR &&
                d->fpGetMemoryFdPropertiesKHR(device, VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT, export_fd, &fp) == VK_SUCCESS &&
                fp.memoryTypeBits && !(fp.memoryTypeBits & (1u << emod.memoryTypeIndex)))
                emod.memoryTypeIndex = (uint32_t)__builtin_ctz(fp.memoryTypeBits);
            imp = (VkImportMemoryFdInfoKHR){VK_STRUCTURE_TYPE_IMPORT_MEMORY_FD_INFO_KHR, emod.pNext,
                                            VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT, dup(export_fd)};
            emod.pNext = &imp;
            pAllocateInfo = &emod;
        }
    }
    int peek_import_fd = -1;
    if (g_dmabuf_export && export_fd < 0 && keep_import_fd < 0 && getenv("MALI_COMPAT_DMABUF_PEEK")) {
        const VkImportMemoryFdInfoKHR *im = find_struct(pAllocateInfo->pNext, VK_STRUCTURE_TYPE_IMPORT_MEMORY_FD_INFO_KHR);
        if (im && im->handleType == VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT) peek_import_fd = dup(im->fd);
    }
    VkMemoryAllocateInfo mod = *pAllocateInfo;
    VkMemoryAllocateFlagsInfo flags_info = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO,
        .flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT,
        .pNext = mod.pNext,
    };
    mod.pNext = &flags_info;
    // MALI_COMPAT_FORCE_COHERENT=1: allocations of the vendor's non-coherent HOST_CACHED type go to the coherent host
    // type of the same heap instead (the app still sees the type it asked for). vkd3d puts READBACK heaps there and
    // invalidates only in Map(); a D3D12 game that maps once and reads the pointer every frame (fine on Windows, where
    // readback memory is coherent) reads stale CPU cache lines on ARM. Reporting the type as device-local instead
    // crashed HZD at start (DXGI adapter memory). Gemini 4 Argon's lead for the missing statics. [Claude Opus 5.5]
    if (g_force_coherent && mod.memoryTypeIndex < VK_MAX_MEMORY_TYPES && d->coherent_remap[mod.memoryTypeIndex]) {
        static _Atomic uint32_t remapped;
        if (atomic_fetch_add(&remapped, 1) == 0)
            fprintf(stderr, "mali-compat: force coherent: memory type %u -> %u\n", mod.memoryTypeIndex,
                    d->coherent_remap[mod.memoryTypeIndex] - 1u);
        mod.memoryTypeIndex = d->coherent_remap[mod.memoryTypeIndex] - 1u;
    }

    VkResult res = d->fpAllocateMemory(device, &mod, pAllocator, pMemory);
    uint32_t heap = fd_import || mod.memoryTypeIndex >= VK_MAX_MEMORY_TYPES ? BUDGET_NO_HEAP : d->type_heap[mod.memoryTypeIndex];
    if (res == VK_SUCCESS) {
        atomic_fetch_add(&g_allocated_bytes, pAllocateInfo->allocationSize);
        if (mem_add(*pMemory, pAllocateInfo->allocationSize, mod.memoryTypeIndex, heap) && g_as_hooks)
            rt_vkmem_alloc(mod.memoryTypeIndex, heap, pAllocateInfo->allocationSize);   /* round 4: vk_mem line */
        if (g_dc_on) dc_mem_alloc(d, device, *pMemory, pAllocateInfo->allocationSize, mod.memoryTypeIndex, dc_mem_devaddr(mod.pNext));
        if (g_memlog) {
            uint64_t va = 0;
            if (g_memlog == 1 && d->fpGetDeviceMemoryOpaqueCaptureAddress) {
                VkDeviceMemoryOpaqueCaptureAddressInfo ci = {VK_STRUCTURE_TYPE_DEVICE_MEMORY_OPAQUE_CAPTURE_ADDRESS_INFO, NULL, *pMemory};
                va = d->fpGetDeviceMemoryOpaqueCaptureAddress(device, &ci);
            }
            ml_event('A', (uint64_t)(uintptr_t)*pMemory, pAllocateInfo->allocationSize, mod.memoryTypeIndex, heap, va, va ? 1 : 0);
            if (g_addrmap && va) { // for the B events' va
                pthread_mutex_lock(&g_mem_mutex);
                struct MemInfo *me = mem_find(*pMemory);
                if (me) me->cva = va;
                pthread_mutex_unlock(&g_mem_mutex);
            }
        }
    } else if (g_memlog) {
        ml_event('X', 0, pAllocateInfo->allocationSize, mod.memoryTypeIndex, heap, (uint64_t)(int64_t)res, 2);
    }
    if (keep_import_fd >= 0) {
        struct ExportMem *e = res == VK_SUCCESS ? malloc(sizeof *e) : NULL;
        if (e) {
            *e = (struct ExportMem){*pMemory, keep_import_fd, pAllocateInfo->allocationSize, true, NULL};
            pthread_mutex_lock(&g_export_mutex);
            e->next = g_export_mem;
            g_export_mem = e;
            pthread_mutex_unlock(&g_export_mutex);
        } else close(keep_import_fd);
    }
    if (peek_import_fd >= 0) {
        struct ExportMem *e = res == VK_SUCCESS ? malloc(sizeof *e) : NULL;
        if (e) {
            *e = (struct ExportMem){*pMemory, peek_import_fd, pAllocateInfo->allocationSize, true, NULL};
            pthread_mutex_lock(&g_export_mutex);
            e->next = g_export_mem;
            g_export_mem = e;
            pthread_mutex_unlock(&g_export_mutex);
        } else close(peek_import_fd);
    }
    if (export_fd >= 0) {
        if (res == VK_SUCCESS) {
            struct ExportMem *e = malloc(sizeof *e);
            if (e) {
                e->mem = *pMemory;
                e->fd = export_fd;
                e->size = pAllocateInfo->allocationSize;
                e->imported = false;
                static bool peek_started;
                if (!peek_started && getenv("MALI_COMPAT_DMABUF_PEEK") && strcmp(getenv("MALI_COMPAT_DMABUF_PEEK"), "0")) {
                    pthread_t t;
                    peek_started = pthread_create(&t, NULL, dmabuf_peek_thread, NULL) == 0;
                    if (peek_started) pthread_detach(t);
                }
                pthread_mutex_lock(&g_export_mutex);
                e->next = g_export_mem;
                g_export_mem = e;
                pthread_mutex_unlock(&g_export_mutex);
            } else close(export_fd);
        } else {
            close(imp.fd); // a failed import leaves the fd to the caller
            close(export_fd);
            fprintf(stderr, "mali-compat: dma-buf export: importing the heap buffer failed: %d\n", res);
        }
    }
    relink(u_exp);
    relink(u_mwsi);
    return res;
}

static VKAPI_ATTR VkResult VKAPI_CALL compat_GetMemoryFdKHR(VkDevice device, const VkMemoryGetFdInfoKHR *info, int *pFd) {
    if (g_dmabuf_export && info->handleType == VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT) {
        int fd = -1;
        pthread_mutex_lock(&g_export_mutex);
        for (struct ExportMem *x = g_export_mem; x; x = x->next) if (x->mem == info->memory) { fd = dup(x->fd); break; }
        pthread_mutex_unlock(&g_export_mutex);
        if (fd >= 0) { *pFd = fd; return VK_SUCCESS; }
    }
    struct DeviceData *d = get_device_data(device);
    return d->fpGetMemoryFdKHR ? d->fpGetMemoryFdKHR(device, info, pFd) : VK_ERROR_INVALID_EXTERNAL_HANDLE;
}

static VKAPI_ATTR void VKAPI_CALL compat_FreeMemory(
    VkDevice device, VkDeviceMemory memory, const VkAllocationCallbacks *pAllocator) {
    struct DeviceData *d = get_device_data(device);
    if (g_dmabuf_export && memory) {
        pthread_mutex_lock(&g_export_mutex);
        for (struct ExportMem **pp = &g_export_mem; *pp; pp = &(*pp)->next)
            if ((*pp)->mem == memory) { struct ExportMem *x = *pp; *pp = x->next; close(x->fd); free(x); break; }
        pthread_mutex_unlock(&g_export_mutex);
    }
    struct MemInfo *e = memory ? mem_remove(memory) : NULL;
    if (e) budget_sub(e->heap, e->size);
    if (e && g_as_hooks) rt_vkmem_free(e->type, e->heap, e->size);   /* round 4: vk_mem line */
    if (g_dc_on) dc_dead(DCK_MEM, (uint64_t)memory);
    if (g_memlog && memory)
        ml_event('F', (uint64_t)(uintptr_t)memory, e ? e->size : 0, e ? e->type : ML_NONE, e ? e->heap : BUDGET_NO_HEAP,
                 e ? e->va : 0, e && e->va ? 1 : 0);
    if (e && e->orig) {
        // Freeing unmaps implicitly; let the driver do it on its own address and keep Wine's range reserved.
        pthread_mutex_lock(&g_mem_mutex);
        bool back = placed_move_back(e);
        pthread_mutex_unlock(&g_mem_mutex);
        if (back) hold_range(e->placed, e->map_size, true);
    }
    free(e);
    d->fpFreeMemory(device, memory, pAllocator);
}


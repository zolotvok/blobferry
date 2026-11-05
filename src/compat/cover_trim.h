// SPDX-License-Identifier: GPL-3.0-or-later
// cover-trim (MALI_COMPAT_COVER_TRIM, off by default) [keeper, Claude Opus 5.5] 2026-11-05
//
// A second, small camera (The Long Dark's weapon/hands camera: a 4-MRT G-buffer pass with 6 draws that clears and
// reuses the world's depth image) runs its own full-screen screen-space AO chain (Unity PPv2 MSVO, 5 compute shaders)
// although only the pixels near its geometry can get occlusion. This feature limits the listed compute dispatches and
// the listed composite draw of such a chain to the screen cells near that camera's geometry. Everything is GPU-driven,
// in the same frame, with no read-back:
//   1. ARM: at the end of a dynamic-rendering pass with >= MRT colour attachments, a CLEARed depth attachment and
//      1..MAXDRAWS draws, the depth aspect is copied to a buffer and a compute pass marks every 16x16 tile that holds a
//      texel != the clear depth (= geometry of that camera). A bigger G-buffer pass (the world, 366 draws) disarms.
//   2. Each listed dispatch (CS hash, at least NX x NY groups) while armed: a 64-thread compute pass writes one
//      predicate per cell of an NX x NY grid over the dispatch's groups (1 = a covered tile lies within the cell's
//      pixel rect grown by MARGIN + K group widths + the hash's extra px); then one vkCmdDispatchBase per cell under
//      VK_EXT_conditional_rendering with the app's pipeline and push constants restored.
//   3. The listed composite draw (FS hash) while armed: one draw per pixel cell (grown by MARGIN), scissored to the
//      cell and conditional on its predicate; the app's scissor is restored. It disarms.
// Outside the cells the trimmed images keep stale values; only the composite reads them, and it is trimmed the same
// way, so a pixel changes only where the full chain's AO was != "no occlusion" farther than MARGIN from the geometry,
// or where a trimmed shader reads a producer texel outside the producer's (larger) cell set. The A/B image compare is
// the proof; no shader is patched. Pipelines whose CS hash is listed get VK_PIPELINE_CREATE_DISPATCH_BASE_BIT.
//
// Env: MALI_COMPAT_COVER_TRIM=1, _CS=<hex[+px]>,... (compute hashes, optional extra dilation for producers that a
// same-sized listed dispatch reads), _FS=<hex>,... (composite fragment hashes), _ARM=<mrt>:<maxdraws> (4:16),
// _GRID=<nx>x<ny> (8x4, nx*ny <= 64), _MARGIN=<px> (64), _K=<groups> (2), _MAXOPS=<n> (8 per arming),
// _TOGGLE=<path> (trim only while that file exists, checked at most once per second: A/B in one launch).
// Stats: one stderr line every 10 s ("cover-trim:").

#define CT_TILE 16
#define CT_CELLS 64
enum { CTS_SMALL, CTS_OPS, CTS_TOGGLE, CTS_SUSP, CTS_FMT, CTS_USAGE, CTS_INIT, CTS_SCISSOR, CTS_COND, CTS_EXTENT, CTS_N };
static const char *const g_cts_name[CTS_N] = {"small", "ops-cap", "toggle-off", "suspending", "depth-fmt", "no-transfer-src",
                                               "init", "no-scissor", "app-cond", "extent"};

struct CtPush {
    uint32_t depth[2], cov[2], pred[2];
    uint32_t w, h, tiles_x, tiles_y;
    uint32_t gx, gy, nx, ny;
    uint32_t dil_x, dil_y, clear_bits, pad;
};

struct CtOld { VkBuffer buf; VkDeviceMemory mem; struct CtOld *next; };

struct CtDev {
    pthread_mutex_t m;
    PFN_vkCmdCopyImageToBuffer CopyImageToBuffer;
    PFN_vkCmdBeginConditionalRenderingEXT BeginCond;
    PFN_vkCmdEndConditionalRenderingEXT EndCond;
    PFN_vkCmdSetScissor SetScissor;
    PFN_vkCmdSetScissorWithCount SetScissorWithCount;
    PFN_vkCmdDispatchBase DispatchBase;
    bool init_tried, ok;
    VkPipelineLayout layout;
    VkPipeline cover, pred, cover_db, pred_db;
    VkBuffer buf;
    VkDeviceMemory mem;
    VkDeviceAddress addr;
    VkDeviceSize size, off_cov, off_pred;
    const uint32_t *map; // host-visible copy of the predicates (stats only), NULL if not mappable
    struct CtOld *old;   // outgrown buffers, freed at vkDestroyDevice (the GPU may still read them)
    // armed state (record order: DXVK records on its CS thread; the mutex covers other threads)
    bool armed;
    uint32_t w, h, tiles_x, tiles_y, ops, comp_nx, comp_ny;
    uint64_t last_log_ns, last_toggle_ns;
    bool toggle_on;
    // stats since the last log line
    uint64_t arms, disarms, rdisp, rdraw, cell_cmds, skip[CTS_N];
    uint64_t gpu_cells_on, gpu_cells;
};

static uint32_t g_ct_cs_extra[16], g_ct_mrt = 4, g_ct_max_draws = 16, g_ct_nx = 8, g_ct_ny = 4, g_ct_margin = 64,
                g_ct_k = 2, g_ct_max_ops = 8;
static const char *g_ct_toggle;

static uint32_t ct_parse_list(const char *s, uint32_t *hash, uint32_t *extra, uint32_t max) {
    uint32_t n = 0;
    while (s && *s && n < max) {
        char *end;
        uint32_t h = (uint32_t)strtoul(s, &end, 16);
        if (end == s) break;
        uint32_t x = 0;
        if (*end == '+') x = (uint32_t)strtoul(end + 1, &end, 10);
        hash[n] = h;
        if (extra) extra[n] = x;
        n++;
        s = end;
        while (*s == ',' || *s == ' ') s++;
    }
    return n;
}

static void ct_parse_env(void) {
    const char *on = getenv("MALI_COMPAT_COVER_TRIM");
    g_ct = on && strcmp(on, "0") != 0;
    if (!g_ct) return;
    g_ct_cs_n = ct_parse_list(getenv("MALI_COMPAT_COVER_TRIM_CS"), g_ct_cs, g_ct_cs_extra, 16);
    g_ct_fs_n = ct_parse_list(getenv("MALI_COMPAT_COVER_TRIM_FS"), g_ct_fs, NULL, 8);
    const char *v;
    if ((v = getenv("MALI_COMPAT_COVER_TRIM_ARM"))) sscanf(v, "%u:%u", &g_ct_mrt, &g_ct_max_draws);
    if ((v = getenv("MALI_COMPAT_COVER_TRIM_GRID"))) sscanf(v, "%ux%u", &g_ct_nx, &g_ct_ny);
    if ((v = getenv("MALI_COMPAT_COVER_TRIM_MARGIN"))) g_ct_margin = (uint32_t)atoi(v);
    if ((v = getenv("MALI_COMPAT_COVER_TRIM_K"))) g_ct_k = (uint32_t)atoi(v);
    if ((v = getenv("MALI_COMPAT_COVER_TRIM_MAXOPS"))) g_ct_max_ops = (uint32_t)atoi(v);
    g_ct_toggle = getenv("MALI_COMPAT_COVER_TRIM_TOGGLE");
    if (!g_ct_nx || !g_ct_ny || g_ct_nx * g_ct_ny > CT_CELLS) g_ct_nx = 8, g_ct_ny = 4;
    if (!g_ct_mrt) g_ct_mrt = 1;
    if (g_ct_max_ops > 32) g_ct_max_ops = 32;
    fprintf(stderr, "mali-compat: cover-trim on: %u CS, %u FS hashes, arm >= %u MRT and <= %u draws, grid %ux%u, margin %u px, "
            "K %u, max %u ops, toggle %s\n", g_ct_cs_n, g_ct_fs_n, g_ct_mrt, g_ct_max_draws, g_ct_nx, g_ct_ny, g_ct_margin,
            g_ct_k, g_ct_max_ops, g_ct_toggle ? g_ct_toggle : "-");
}

static bool ct_cs_listed(uint32_t h) {
    for (uint32_t i = 0; i < g_ct_cs_n; i++) if (g_ct_cs[i] == h) return true;
    return false;
}
static int ct_cs_index(VkPipeline p) {
    struct HashNode hn;
    if (!p || !hash_get(g_pipeline_hash, (uint64_t)p, &hn)) return -1;
    for (uint32_t i = 0; i < g_ct_cs_n; i++) if (g_ct_cs[i] == hn.hash[0]) return (int)i;
    return -1;
}
static bool ct_fs_pipe(VkPipeline p) {
    struct HashNode hn;
    if (!p || !hash_get(g_pipeline_hash, (uint64_t)p, &hn)) return false;
    for (int s = 0; s < 3; s++)
        if (hn.stages[s] == VK_SHADER_STAGE_FRAGMENT_BIT)
            for (uint32_t i = 0; i < g_ct_fs_n; i++) if (g_ct_fs[i] == hn.hash[s]) return true;
    return false;
}

static uint64_t ct_now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}

static void ct_device_init(struct DeviceData *d, PFN_vkGetDeviceProcAddr gdpa) {
    struct CtDev *t = calloc(1, sizeof *t);
    if (!t) return;
    pthread_mutex_init(&t->m, NULL);
    t->CopyImageToBuffer = (PFN_vkCmdCopyImageToBuffer)gdpa(d->device, "vkCmdCopyImageToBuffer");
    t->BeginCond = (PFN_vkCmdBeginConditionalRenderingEXT)gdpa(d->device, "vkCmdBeginConditionalRenderingEXT");
    t->EndCond = (PFN_vkCmdEndConditionalRenderingEXT)gdpa(d->device, "vkCmdEndConditionalRenderingEXT");
    t->SetScissor = (PFN_vkCmdSetScissor)gdpa(d->device, "vkCmdSetScissor");
    t->SetScissorWithCount = (PFN_vkCmdSetScissorWithCount)gdpa(d->device, "vkCmdSetScissorWithCount");
    if (!t->SetScissorWithCount) t->SetScissorWithCount = (PFN_vkCmdSetScissorWithCount)gdpa(d->device, "vkCmdSetScissorWithCountEXT");
    t->DispatchBase = (PFN_vkCmdDispatchBase)gdpa(d->device, "vkCmdDispatchBase");
    if (!t->DispatchBase) t->DispatchBase = (PFN_vkCmdDispatchBase)gdpa(d->device, "vkCmdDispatchBaseKHR");
    d->ct = t;
    if (!t->CopyImageToBuffer || !t->BeginCond || !t->EndCond || !t->SetScissor || !t->DispatchBase)
        fprintf(stderr, "mali-compat: cover-trim: driver entry points missing (copy %d cond %d/%d scissor %d base %d); off\n",
                !!t->CopyImageToBuffer, !!t->BeginCond, !!t->EndCond, !!t->SetScissor, !!t->DispatchBase);
}

static void ct_device_destroy(struct DeviceData *d) {
    struct CtDev *t = d->ct;
    if (!t) return;
    VkPipeline ps[4] = {t->cover, t->pred, t->cover_db, t->pred_db};
    for (int i = 0; i < 4; i++) if (ps[i]) d->fpDestroyPipeline(d->device, ps[i], NULL);
    if (t->layout) d->fpDestroyPipelineLayout(d->device, t->layout, NULL);
    if (t->buf) d->fpDestroyBuffer(d->device, t->buf, NULL);
    if (t->mem) d->fpFreeMemory(d->device, t->mem, NULL);
    for (struct CtOld *o = t->old, *n; o; o = n) {
        n = o->next;
        d->fpDestroyBuffer(d->device, o->buf, NULL);
        d->fpFreeMemory(d->device, o->mem, NULL);
        free(o);
    }
    pthread_mutex_destroy(&t->m);
    free(t);
    d->ct = NULL;
}

static bool ct_pipelines(struct DeviceData *d, struct CtDev *t) {
    if (t->init_tried) return t->ok;
    t->init_tried = true;
    if (!t->CopyImageToBuffer || !t->BeginCond || !t->EndCond || !t->SetScissor || !t->DispatchBase) return false;
    VkPushConstantRange pr = {VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(struct CtPush)};
    VkPipelineLayoutCreateInfo pl = {VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, NULL, 0, 0, NULL, 1, &pr};
    if (d->fpCreatePipelineLayout(d->device, &pl, NULL, &t->layout) != VK_SUCCESS) return false;
    const struct { const uint32_t *code; size_t size; VkPipeline *p, *pdb; } sh[2] = {
        {ct_cover_spv, sizeof ct_cover_spv, &t->cover, &t->cover_db},
        {ct_pred_spv, sizeof ct_pred_spv, &t->pred, &t->pred_db},
    };
    for (int i = 0; i < 2; i++) {
        VkShaderModuleCreateInfo sm = {VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, NULL, 0, sh[i].size, sh[i].code};
        VkShaderModule mod;
        if (d->fpCreateShaderModule(d->device, &sm, NULL, &mod) != VK_SUCCESS) return false;
        VkComputePipelineCreateInfo ci = {VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO, NULL, 0,
            {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, NULL, 0, VK_SHADER_STAGE_COMPUTE_BIT, mod, "main", NULL},
            t->layout, VK_NULL_HANDLE, -1};
        VkResult r = d->fpCreateComputePipelines(d->device, VK_NULL_HANDLE, 1, &ci, NULL, sh[i].p);
        if (r == VK_SUCCESS && d->descriptor_buffer) {
            ci.flags = VK_PIPELINE_CREATE_DESCRIPTOR_BUFFER_BIT_EXT;
            r = d->fpCreateComputePipelines(d->device, VK_NULL_HANDLE, 1, &ci, NULL, sh[i].pdb);
        }
        d->fpDestroyShaderModule(d->device, mod, NULL);
        if (r != VK_SUCCESS) return false;
    }
    t->ok = true;
    return true;
}

// One buffer: [depth copy w*h*4][tiles][predicates: (1 + max ops) slots x 64 cells]. Grown when a bigger depth image
// arms; the outgrown one is kept until vkDestroyDevice.
static bool ct_buffer(struct DeviceData *d, struct CtDev *t, uint32_t w, uint32_t h) {
    VkDeviceSize depth = (VkDeviceSize)w * h * 4, tiles = (VkDeviceSize)((w + CT_TILE - 1) / CT_TILE) * ((h + CT_TILE - 1) / CT_TILE) * 4;
    VkDeviceSize off_cov = (depth + 255) & ~(VkDeviceSize)255, off_pred = (off_cov + tiles + 255) & ~(VkDeviceSize)255;
    VkDeviceSize size = off_pred + (VkDeviceSize)(1 + 32) * CT_CELLS * 4;
    if (t->buf && off_pred <= t->off_pred) {
        // fits: the depth copy and the tiles end before the old predicate area, which stays where it is
        t->off_cov = off_cov;
        return true;
    }
    VkBufferCreateInfo bi = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, NULL, 0, size,
        VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
        VK_BUFFER_USAGE_CONDITIONAL_RENDERING_BIT_EXT};
    VkBuffer buf;
    if (d->fpCreateBuffer(d->device, &bi, NULL, &buf) != VK_SUCCESS) return false;
    VkMemoryRequirements mr;
    d->fpGetBufferMemoryRequirements(d->device, buf, &mr);
    VkPhysicalDeviceMemoryProperties mp;
    d->instance_data->fpGetPhysicalDeviceMemoryProperties(d->physical_device, &mp);
    uint32_t type = UINT32_MAX;
    const VkMemoryPropertyFlags want = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    for (uint32_t i = 0; i < mp.memoryTypeCount && type == UINT32_MAX; i++)
        if ((mr.memoryTypeBits & (1u << i)) && (mp.memoryTypes[i].propertyFlags & want) == want &&
            (mp.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)) type = i;
    for (uint32_t i = 0; i < mp.memoryTypeCount && type == UINT32_MAX; i++)
        if (mr.memoryTypeBits & (1u << i)) type = i;
    VkMemoryAllocateFlagsInfo fi = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO, NULL, VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT, 0};
    VkMemoryAllocateInfo ai = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, &fi, mr.size, type};
    VkDeviceMemory mem;
    if (d->fpAllocateMemory(d->device, &ai, NULL, &mem) != VK_SUCCESS) { d->fpDestroyBuffer(d->device, buf, NULL); return false; }
    if (d->fpBindBufferMemory(d->device, buf, mem, 0) != VK_SUCCESS) {
        d->fpDestroyBuffer(d->device, buf, NULL);
        d->fpFreeMemory(d->device, mem, NULL);
        return false;
    }
    void *map = NULL;
    if ((mp.memoryTypes[type].propertyFlags & want) == want && d->fpMapMemory(d->device, mem, 0, VK_WHOLE_SIZE, 0, &map) != VK_SUCCESS)
        map = NULL;
    if (t->buf) {
        struct CtOld *o = malloc(sizeof *o);
        if (o) { *o = (struct CtOld){t->buf, t->mem, t->old}; t->old = o; }
    }
    VkBufferDeviceAddressInfo bda = {VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO, NULL, buf};
    t->buf = buf;
    t->mem = mem;
    t->size = size;
    t->addr = d->fpGetBufferDeviceAddress(d->device, &bda);
    t->off_cov = off_cov;
    t->off_pred = off_pred;
    t->map = map ? (const uint32_t *)((const char *)map + off_pred) : NULL;
    return true;
}

static void ct_addr(uint32_t out[2], VkDeviceAddress a) { out[0] = (uint32_t)a; out[1] = (uint32_t)(a >> 32); }

// The app's compute pipeline and push constants back after the layer's own dispatch (same as the BC decode, H1).
static void ct_restore_compute(struct DeviceData *d, VkCommandBuffer cmd, struct CmdBufData *c) {
    if (c->compute_pipeline) d->fpCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, c->compute_pipeline);
    for (uint32_t i = 0; i < c->push_count; i++) {
        const struct PushRec *pr = &c->push[i];
        PFN_vkCmdPushConstants2 fp2 = d->fpCmdPushConstants2KHR ? d->fpCmdPushConstants2KHR : d->fpCmdPushConstants2;
        if (pr->v2 && fp2) {
            VkPushConstantsInfo pi = {VK_STRUCTURE_TYPE_PUSH_CONSTANTS_INFO, NULL, pr->layout, pr->stages, pr->offset, pr->size, pr->data};
            fp2(cmd, &pi);
        } else {
            d->fpCmdPushConstants(cmd, pr->layout, pr->stages, pr->offset, pr->size, pr->data);
        }
    }
}

// The layer's own 64-thread predicate pass for slot `slot` (0 = the composite's pixel cells), then the barrier into
// conditional rendering. The caller restores the app's compute state.
static void ct_record_pred(struct DeviceData *d, struct CtDev *t, VkCommandBuffer cmd, struct CmdBufData *c, uint32_t slot,
                           uint32_t gx, uint32_t gy, uint32_t dil_x, uint32_t dil_y) {
    struct CtPush pc = {0};
    ct_addr(pc.cov, t->addr + t->off_cov);
    ct_addr(pc.pred, t->addr + t->off_pred + (VkDeviceSize)slot * CT_CELLS * 4);
    pc.w = t->w; pc.h = t->h; pc.tiles_x = t->tiles_x; pc.tiles_y = t->tiles_y;
    pc.gx = gx; pc.gy = gy; pc.nx = g_ct_nx; pc.ny = g_ct_ny; pc.dil_x = dil_x; pc.dil_y = dil_y;
    d->fpCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, c->descriptor_buffers && t->pred_db ? t->pred_db : t->pred);
    d->fpCmdPushConstants(cmd, t->layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof pc, &pc);
    d->fpCmdDispatch(cmd, 1, 1, 1);
    VkMemoryBarrier mb = {VK_STRUCTURE_TYPE_MEMORY_BARRIER, NULL, VK_ACCESS_SHADER_WRITE_BIT,
                          VK_ACCESS_CONDITIONAL_RENDERING_READ_BIT_EXT | VK_ACCESS_SHADER_READ_BIT};
    d->fpCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                            VK_PIPELINE_STAGE_CONDITIONAL_RENDERING_BIT_EXT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb, 0,
                            NULL, 0, NULL);
}

static bool ct_toggle_on(struct CtDev *t) {
    if (!g_ct_toggle) return true;
    uint64_t now = ct_now();
    if (!t->last_toggle_ns || now - t->last_toggle_ns >= 1000000000ull) {
        bool on = access(g_ct_toggle, F_OK) == 0;
        if (t->last_toggle_ns && on != t->toggle_on)
            fprintf(stderr, "mali-compat: cover-trim: toggle %s\n", on ? "ON" : "OFF");
        t->toggle_on = on;
        t->last_toggle_ns = now;
    }
    return t->toggle_on;
}

static void ct_log_locked(struct CtDev *t) {
    uint64_t now = ct_now();
    if (!t->last_log_ns) { t->last_log_ns = now; return; }
    if (now - t->last_log_ns < 10000000000ull) return;
    char sk[256] = "";
    size_t n = 0;
    for (int i = 0; i < CTS_N && n < sizeof sk - 32; i++)
        if (t->skip[i]) n += (size_t)snprintf(sk + n, sizeof sk - n, " %s %llu", g_cts_name[i], (unsigned long long)t->skip[i]);
    fprintf(stderr, "mali-compat: cover-trim: %.1f s: arms %llu, disarms %llu, trimmed dispatches %llu (cell dispatches %llu), "
            "trimmed draws %llu, GPU cells on %.1f %% (%llu of %llu, read back late), toggle %s, skips:%s\n",
            (now - t->last_log_ns) / 1e9, (unsigned long long)t->arms, (unsigned long long)t->disarms,
            (unsigned long long)t->rdisp, (unsigned long long)t->cell_cmds, (unsigned long long)t->rdraw,
            t->gpu_cells ? 100.0 * t->gpu_cells_on / t->gpu_cells : 0.0, (unsigned long long)t->gpu_cells_on,
            (unsigned long long)t->gpu_cells, g_ct_toggle ? (t->toggle_on ? "on" : "off") : "-", n ? sk : " none");
    t->arms = t->disarms = t->rdisp = t->rdraw = t->cell_cmds = t->gpu_cells_on = t->gpu_cells = 0;
    memset(t->skip, 0, sizeof t->skip);
    t->last_log_ns = now;
}

// vkCmdBeginRendering: what the arm check needs at the pass end.
static void ct_pass_begin(struct CmdBufData *c, const VkRenderingInfo *ri) {
    if (!c) return;
    struct CtCmd *k = &c->ct;
    k->in_pass = true;
    k->draws = 0;
    k->colors = 0;
    for (uint32_t i = 0; ri->pColorAttachments && i < ri->colorAttachmentCount; i++) k->colors += ri->pColorAttachments[i].imageView != VK_NULL_HANDLE;
    const VkRenderingAttachmentInfo *da = ri->pDepthAttachment && ri->pDepthAttachment->imageView ? ri->pDepthAttachment : NULL;
    k->depth_view = da ? da->imageView : VK_NULL_HANDLE;
    k->depth_layout = da ? da->imageLayout : VK_IMAGE_LAYOUT_UNDEFINED;
    k->depth_clear = da && da->loadOp == VK_ATTACHMENT_LOAD_OP_CLEAR;
    k->clear_depth = da ? da->clearValue.depthStencil.depth : 0.0f;
    k->suspending = (ri->flags & VK_RENDERING_SUSPENDING_BIT) != 0;
    k->area = ri->renderArea;
}

// vkCmdEndRendering (after the driver call): arm on a small G-buffer pass, disarm on a big one.
static void ct_pass_end(struct CmdBufData *c, VkCommandBuffer cmd) {
    if (!c || !c->ct.in_pass) return;
    struct CtCmd *k = &c->ct;
    k->in_pass = false;
    struct DeviceData *d = c->dev;
    struct CtDev *t = d->ct;
    if (!t || k->colors < g_ct_mrt || !k->depth_view || !k->depth_clear) return;
    pthread_mutex_lock(&t->m);
    ct_log_locked(t);
    if (k->draws > g_ct_max_draws || !k->draws) {
        if (t->armed) t->disarms++;
        t->armed = false;
        pthread_mutex_unlock(&t->m);
        return;
    }
    t->armed = false;
    int why = -1;
    struct ImgNode view = {0}, img = {0};
    if (!ct_toggle_on(t)) why = CTS_TOGGLE;
    else if (k->suspending) why = CTS_SUSP;
    else if (!img_get(g_view_info, (uint64_t)k->depth_view, &view) || !img_get(g_img_info, (uint64_t)view.image, &img) ||
             (img.format != VK_FORMAT_D32_SFLOAT && img.format != VK_FORMAT_D32_SFLOAT_S8_UINT) || view.mip || view.layer ||
             img.samples > 1 || !img.w || !img.h)
        why = CTS_FMT;
    else if (!(img.usage & VK_IMAGE_USAGE_TRANSFER_SRC_BIT)) why = CTS_USAGE;
    else if (k->area.offset.x || k->area.offset.y || k->area.extent.width != img.w || k->area.extent.height != img.h) why = CTS_EXTENT;
    else if (!ct_pipelines(d, t) || !ct_buffer(d, t, img.w, img.h)) why = CTS_INIT;
    if (why >= 0) { t->skip[why]++; pthread_mutex_unlock(&t->m); return; }

    // stats: the predicates the GPU wrote for the previous arming (late, unsynchronised; only counted)
    if (t->map)
        for (uint32_t s = 0; s <= t->ops && s <= 32; s++)
            for (uint32_t i = 0; i < g_ct_nx * g_ct_ny; i++) { t->gpu_cells++; t->gpu_cells_on += t->map[s * CT_CELLS + i] != 0; }

    t->w = img.w;
    t->h = img.h;
    t->tiles_x = (img.w + CT_TILE - 1) / CT_TILE;
    t->tiles_y = (img.h + CT_TILE - 1) / CT_TILE;
    bool stencil = img.format == VK_FORMAT_D32_SFLOAT_S8_UINT;
    VkImageLayout L = k->depth_layout;
    bool separate = L == VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL || L == VK_IMAGE_LAYOUT_DEPTH_READ_ONLY_OPTIMAL;
    VkImageLayout copy_layout = L == VK_IMAGE_LAYOUT_GENERAL ? VK_IMAGE_LAYOUT_GENERAL : VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    VkImageSubresourceRange range = {VK_IMAGE_ASPECT_DEPTH_BIT | (stencil && !separate ? VK_IMAGE_ASPECT_STENCIL_BIT : 0), 0, 1, 0, 1};
    // B0: depth writes of this pass -> copy; earlier users of the layer's buffer (predicate passes, conditional reads
    // of the previous arming) -> the fill/copy that overwrite it.
    VkImageMemoryBarrier ib = {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER, NULL, VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
                               VK_ACCESS_TRANSFER_READ_BIT, L, copy_layout, VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED,
                               view.image, range};
    VkMemoryBarrier mb = {VK_STRUCTURE_TYPE_MEMORY_BARRIER, NULL, 0, VK_ACCESS_TRANSFER_WRITE_BIT};
    d->fpCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT |
                            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_CONDITIONAL_RENDERING_BIT_EXT,
                            VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &mb, 0, NULL, 1, &ib);
    VkBufferImageCopy rg = {0, 0, 0, {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 0, 1}, {0, 0, 0}, {img.w, img.h, 1}};
    t->CopyImageToBuffer(cmd, view.image, copy_layout, t->buf, 1, &rg);
    d->fpCmdFillBuffer(cmd, t->buf, t->off_cov, (VkDeviceSize)t->tiles_x * t->tiles_y * 4, 0);
    // B1: the depth image back to the pass's layout for whatever the app does next; the copy + fill -> compute.
    ib.srcAccessMask = 0;
    ib.dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT;
    ib.oldLayout = copy_layout;
    ib.newLayout = L;
    mb.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    d->fpCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 1, &mb, 0, NULL, 1, &ib);
    struct CtPush pc = {0};
    ct_addr(pc.depth, t->addr);
    ct_addr(pc.cov, t->addr + t->off_cov);
    pc.w = t->w; pc.h = t->h; pc.tiles_x = t->tiles_x; pc.tiles_y = t->tiles_y;
    memcpy(&pc.clear_bits, &k->clear_depth, 4);
    d->fpCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, c->descriptor_buffers && t->cover_db ? t->cover_db : t->cover);
    d->fpCmdPushConstants(cmd, t->layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof pc, &pc);
    d->fpCmdDispatch(cmd, t->tiles_x, t->tiles_y, 1);
    VkMemoryBarrier mb2 = {VK_STRUCTURE_TYPE_MEMORY_BARRIER, NULL, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT};
    d->fpCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb2, 0, NULL, 0, NULL);
    // slot 0: the composite draw's pixel cells (it runs inside a render pass, where no dispatch may be recorded)
    ct_record_pred(d, t, cmd, c, 0, t->w, t->h, g_ct_margin, g_ct_margin);
    ct_restore_compute(d, cmd, c);
    t->armed = true;
    t->ops = 0;
    t->arms++;
    pthread_mutex_unlock(&t->m);
}

// vkCmdDispatch while armed with a listed CS hash: predicates + one conditional DispatchBase per cell. True = done.
static bool ct_dispatch(struct CmdBufData *c, VkCommandBuffer cmd, uint32_t x, uint32_t y, uint32_t z) {
    struct DeviceData *d = c->dev;
    struct CtDev *t = d->ct;
    if (!t || !t->armed) return false;
    int li = ct_cs_index(c->compute_pipeline);
    if (li < 0) return false;
    pthread_mutex_lock(&t->m);
    int why = -1;
    if (!t->armed) why = -2;
    else if (x < g_ct_nx || y < g_ct_ny) why = CTS_SMALL;
    else if (c->ct.app_cond) why = CTS_COND;
    else if (t->ops >= g_ct_max_ops) why = CTS_OPS;
    if (why != -1) { if (why >= 0) t->skip[why]++; pthread_mutex_unlock(&t->m); return false; }
    uint32_t slot = 1 + t->ops++;
    uint32_t dil_x = g_ct_margin + g_ct_k * ((t->w + x - 1) / x) + g_ct_cs_extra[li];
    uint32_t dil_y = g_ct_margin + g_ct_k * ((t->h + y - 1) / y) + g_ct_cs_extra[li];
    ct_record_pred(d, t, cmd, c, slot, x, y, dil_x, dil_y);
    ct_restore_compute(d, cmd, c);
    VkBuffer buf = t->buf;
    VkDeviceSize base = t->off_pred + (VkDeviceSize)slot * CT_CELLS * 4;
    t->rdisp++;
    uint32_t n = 0;
    for (uint32_t cy = 0; cy < g_ct_ny; cy++)
        for (uint32_t cx = 0; cx < g_ct_nx; cx++) {
            uint32_t gx0 = cx * x / g_ct_nx, gx1 = (cx + 1) * x / g_ct_nx, gy0 = cy * y / g_ct_ny, gy1 = (cy + 1) * y / g_ct_ny;
            if (gx1 <= gx0 || gy1 <= gy0) continue;
            VkConditionalRenderingBeginInfoEXT cb = {VK_STRUCTURE_TYPE_CONDITIONAL_RENDERING_BEGIN_INFO_EXT, NULL, buf,
                                                     base + (VkDeviceSize)(cy * g_ct_nx + cx) * 4, 0};
            t->BeginCond(cmd, &cb);
            t->DispatchBase(cmd, gx0, gy0, 0, gx1 - gx0, gy1 - gy0, z);
            t->EndCond(cmd);
            n++;
        }
    t->cell_cmds += n;
    pthread_mutex_unlock(&t->m);
    return true;
}

// vkCmdDraw / vkCmdDrawIndexed while armed with a listed FS hash: one scissored conditional draw per pixel cell.
// indexed: a = index count, b = instances, c0 = first index, vo = vertex offset, fi = first instance.
static bool ct_draw(struct CmdBufData *c, VkCommandBuffer cmd, bool indexed, uint32_t a, uint32_t b, uint32_t c0, int32_t vo,
                    uint32_t fi) {
    struct DeviceData *d = c->dev;
    struct CtDev *t = d->ct;
    if (!t || !t->armed || !ct_fs_pipe(c->gfx_pipeline)) return false;
    pthread_mutex_lock(&t->m);
    int why = -1;
    if (!t->armed) why = -2;
    else if (!c->ct.sc_valid || (c->ct.sc_wc && !t->SetScissorWithCount)) why = CTS_SCISSOR;
    else if (c->ct.app_cond) why = CTS_COND;
    if (why != -1) { if (why >= 0) t->skip[why]++; pthread_mutex_unlock(&t->m); return false; }
    VkRect2D app = c->ct.sc;
    int64_t ax0 = app.offset.x, ay0 = app.offset.y, ax1 = ax0 + app.extent.width, ay1 = ay0 + app.extent.height;
    for (uint32_t cy = 0; cy < g_ct_ny; cy++)
        for (uint32_t cx = 0; cx < g_ct_nx; cx++) {
            int64_t x0 = (int64_t)cx * t->w / g_ct_nx, x1 = (int64_t)(cx + 1) * t->w / g_ct_nx;
            int64_t y0 = (int64_t)cy * t->h / g_ct_ny, y1 = (int64_t)(cy + 1) * t->h / g_ct_ny;
            if (x0 < ax0) x0 = ax0;
            if (y0 < ay0) y0 = ay0;
            if (x1 > ax1) x1 = ax1;
            if (y1 > ay1) y1 = ay1;
            if (x1 <= x0 || y1 <= y0) continue;
            VkRect2D r = {{(int32_t)x0, (int32_t)y0}, {(uint32_t)(x1 - x0), (uint32_t)(y1 - y0)}};
            if (c->ct.sc_wc) t->SetScissorWithCount(cmd, 1, &r);
            else t->SetScissor(cmd, 0, 1, &r);
            VkConditionalRenderingBeginInfoEXT cb = {VK_STRUCTURE_TYPE_CONDITIONAL_RENDERING_BEGIN_INFO_EXT, NULL, t->buf,
                                                     t->off_pred + (VkDeviceSize)(cy * g_ct_nx + cx) * 4, 0};
            t->BeginCond(cmd, &cb);
            if (indexed) d->gt->fpCmdDrawIndexed(cmd, a, b, c0, vo, fi);
            else d->gt->fpCmdDraw(cmd, a, b, c0, fi);
            t->EndCond(cmd);
            t->cell_cmds++;
        }
    if (c->ct.sc_wc) t->SetScissorWithCount(cmd, 1, &app);
    else t->SetScissor(cmd, 0, 1, &app);
    t->rdraw++;
    t->armed = false; // the composite is the chain's last reader
    pthread_mutex_unlock(&t->m);
    return true;
}

static VKAPI_ATTR void VKAPI_CALL compat_CmdSetScissor(VkCommandBuffer cmd, uint32_t first, uint32_t n, const VkRect2D *r) {
    struct CmdBufData *c = cmd_data(cmd);
    struct DeviceData *d = c ? c->dev : get_device_data_by_cmd(cmd);
    if (c) {
        if (first == 0 && n >= 1) { c->ct.sc = r[0]; c->ct.sc_valid = n == 1; c->ct.sc_wc = false; }
        else c->ct.sc_valid = false;
    }
    d->ct->SetScissor(cmd, first, n, r);
}
static VKAPI_ATTR void VKAPI_CALL compat_CmdSetScissorWithCount(VkCommandBuffer cmd, uint32_t n, const VkRect2D *r) {
    struct CmdBufData *c = cmd_data(cmd);
    struct DeviceData *d = c ? c->dev : get_device_data_by_cmd(cmd);
    if (c) {
        if (n >= 1) c->ct.sc = r[0];
        c->ct.sc_valid = n == 1;
        c->ct.sc_wc = true;
    }
    d->ct->SetScissorWithCount(cmd, n, r);
}
static VKAPI_ATTR void VKAPI_CALL compat_CmdBeginConditionalRenderingEXT(VkCommandBuffer cmd, const VkConditionalRenderingBeginInfoEXT *b) {
    struct CmdBufData *c = cmd_data(cmd);
    if (c) c->ct.app_cond = true;
    (c ? c->dev : get_device_data_by_cmd(cmd))->ct->BeginCond(cmd, b);
}
static VKAPI_ATTR void VKAPI_CALL compat_CmdEndConditionalRenderingEXT(VkCommandBuffer cmd) {
    struct CmdBufData *c = cmd_data(cmd);
    if (c) c->ct.app_cond = false;
    (c ? c->dev : get_device_data_by_cmd(cmd))->ct->EndCond(cmd);
}

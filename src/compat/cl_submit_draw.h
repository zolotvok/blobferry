// SPDX-License-Identifier: GPL-3.0-or-later
// cl_submit_draw.h: GT render hooks, submit/wait, push descriptors, rendering, draw/dispatch, cover trim. Included by mali_compat_layer.c.
static void gt_render_begin(VkCommandBuffer cmd, uint32_t w, uint32_t h, uint32_t colors, const VkRenderingInfo *ri) {
    struct CmdBufData *c = cmd_data(cmd);
    if (!c || !c->dev->gt) return;
    // K4 (MALI_COMPAT_GT_SPLIT_SKIP / sync-diag k4): no timestamps around suspend/resume parts (invalid there)
    if (ri && (ri->flags & (VK_RENDERING_SUSPENDING_BIT | VK_RENDERING_RESUMING_BIT)) &&
        atomic_load_explicit(&g_sd_k4, memory_order_relaxed)) { c->gt_rec = -1; SD_CNT(SDC_K4); return; }
    c->gt_rec = gt_begin(c, cmd, GT_PASS, w, h, colors, VK_NULL_HANDLE);
    if (c->gt_rec >= 0 && ri) {
        struct GtRec *r = &c->dev->gt->rec[c->gt_rec];
        r->rlayers = ri->layerCount;
        if (ri->colorAttachmentCount && ri->pColorAttachments) r->color0 = ri->pColorAttachments[0].imageView;
        if (ri->pDepthAttachment) r->depth = ri->pDepthAttachment->imageView;
        uint32_t layers = ri->viewMask ? (uint32_t)__builtin_popcount(ri->viewMask) : ri->layerCount;
        for (uint32_t i = 0; ri->pColorAttachments && i < ri->colorAttachmentCount && i < GT_ATT - 1; i++)
            gt_att_add(&r->att, &ri->pColorAttachments[i], w, h, layers, false);
        const VkRenderingAttachmentInfo *ds = ri->pDepthAttachment && ri->pDepthAttachment->imageView ? ri->pDepthAttachment
                                                                                                      : ri->pStencilAttachment;
        if (ds && ds->imageView) gt_att_add(&r->att, ds, w, h, layers, true);
        gt_att_sign(&r->att);
    }
}

static void gt_render_end(VkCommandBuffer cmd) {
    struct CmdBufData *c = cmd_data(cmd);
    if (c && c->dev->gt && c->gt_rec >= 0) { gt_end(c, cmd, c->gt_rec); c->gt_rec = -1; }
}

// MALI_COMPAT_SKIP_HASH=<hex>: draws whose bound graphics pipeline has a shader with this hash are dropped (finds the
// draw that makes a pass slow; the picture is wrong while it is set). [Claude Opus 5.5] 2026-10-30
static bool gt_skip(struct CmdBufData *c) {
    struct HashNode h;
    if (!g_skip_hash || !c || !c->gfx_pipeline || !hash_get(g_pipeline_hash, (uint64_t)c->gfx_pipeline, &h)) return false;
    for (int i = 0; i < 3; i++)
        if (h.hash[i] == g_skip_hash) { atomic_fetch_add(&g_skipped_draws, 1); return true; }
    return false;
}

// Returns true when the draw is to be skipped (gt_skip).
static bool gt_draw(struct CmdBufData *c, uint32_t vcount, uint32_t icount) {
    if (c && c->async_skip) { atomic_fetch_add(&g_async_skipped, 1); return true; }
    if (gt_skip(c)) return true;
    if (g_ct && c) { if (c->ct.in_pass) c->ct.draws++; else c->ct.sdraws++; } // cover-trim arm check
    if (!c || !c->dev->gt) return false;
    if (!g_gpu_time) return false; // the async compiles and the pass clamp share the hooks; no records without the switch
    struct GtCnt one = {c->gfx_flags & PF_EQ ? 1 : 0, c->gfx_flags & PF_DW ? 1 : 0, c->gfx_flags & PF_VSW ? 1 : 0,
                        (uint64_t)vcount * icount};
    if (c->gt_rec < 0) {
        // no pass record in this command buffer: a secondary one (DXVK's tiler mode), see compat_CmdExecuteCommands
        if (!c->gt_sdraws++) {
            c->gt_spipe = c->gfx_pipeline; c->gt_svcount = vcount; c->gt_sicount = icount; c->gt_ssrc = c->gt_src;
        }
        gt_cnt_add(&c->gt_scnt, &one);
        return false;
    }
    struct GtRec *r = &c->dev->gt->rec[c->gt_rec];
    if (!r->draws++) { r->pipeline = c->gfx_pipeline; r->vcount = vcount; r->icount = icount; r->src = c->gt_src; }
    gt_cnt_add(&r->cnt, &one);
    return false;
}

// GPU timing: DXVK's tiler mode records each pass's draws into secondary command buffers and executes them inside the
// pass it begins in the primary, so the pass saw no draws ("draws 0.0 pipe (nil)"). The secondaries' draw counts and
// first draw join the open pass here. [Claude Opus 5.5] 2026-10-30
static VKAPI_ATTR void VKAPI_CALL compat_CmdExecuteCommands(VkCommandBuffer cmd, uint32_t count, const VkCommandBuffer *secs) {
    GTC(GTC_EXECUTE);
    struct CmdBufData *c = cmd_data(cmd);
    struct DeviceData *d = c ? c->dev : get_device_data_by_cmd(cmd);
    for (uint32_t i = 0; g_ct && c && c->ct.in_pass && secs && i < count; i++) { // cover-trim: tiler-mode draws
        struct CmdBufData *s = cmd_data(secs[i]);
        if (s) c->ct.draws += s->ct.sdraws;
    }
    if (c && c->gt_rec >= 0 && d->gt) {
        struct GtRec *r = &d->gt->rec[c->gt_rec];
        for (uint32_t i = 0; secs && i < count; i++) {
            struct CmdBufData *s = cmd_data(secs[i]);
            if (!s || !s->gt_sdraws) continue;
            if (!r->draws) { r->pipeline = s->gt_spipe; r->vcount = s->gt_svcount; r->icount = s->gt_sicount; r->src = s->gt_ssrc; }
            r->draws += s->gt_sdraws;
            gt_cnt_add(&r->cnt, &s->gt_scnt);
        }
    }
    d->gt->fpCmdExecuteCommands(cmd, count, secs);
}

// Queue-side CPU time (QStat, GPU timing only). A queue carries its device's dispatch key, so the command buffer
// lookup finds its device too.
static VKAPI_ATTR VkResult VKAPI_CALL compat_QueueSubmit(VkQueue queue, uint32_t n, const VkSubmitInfo *si, VkFence fence) {
    struct DeviceData *d = get_device_data_by_cmd((VkCommandBuffer)queue);
    uint32_t cbs = 0;
    for (uint32_t i = 0; si && i < n; i++) cbs += si[i].commandBufferCount;
    if (g_dc_on) dc_submit(queue, n, si, fence);
    uint64_t t0 = now_ns();
    VkResult r = d->gt->fpQueueSubmit(queue, n, si, fence);
    qstat_add(QS_SUBMIT, t0, cbs);
    if (g_wait_callers && fence) {
        uint32_t waits = 0;
        for (uint32_t i = 0; si && i < n; i++) waits += si[i].waitSemaphoreCount;
        wc_tag((uint64_t)fence, wc_submit_kind(cbs, waits));
    }
    return r;
}
static VKAPI_ATTR VkResult VKAPI_CALL compat_QueueSubmit2(VkQueue queue, uint32_t n, const VkSubmitInfo2 *si, VkFence fence) {
    struct DeviceData *d = get_device_data_by_cmd((VkCommandBuffer)queue);
    uint32_t cbs = 0;
    for (uint32_t i = 0; si && i < n; i++) cbs += si[i].commandBufferInfoCount;
    if (g_dc_on) dc_submit2(queue, n, si, fence);
    uint64_t t0 = now_ns();
    VkResult r = d->gt->fpQueueSubmit2(queue, n, si, fence);
    qstat_add(QS_SUBMIT, t0, cbs);
    if (g_wait_callers && fence) {
        uint32_t waits = 0;
        for (uint32_t i = 0; si && i < n; i++) waits += si[i].waitSemaphoreInfoCount;
        wc_tag((uint64_t)fence, wc_submit_kind(cbs, waits));
    }
    return r;
}

// MALI_COMPAT_ADDRMAP: S event for every submit (first up to 4 command buffers of the whole call).
static void am_submit(VkQueue queue, uint64_t k, int v, uint32_t n, const VkSubmitInfo *si, const VkSubmitInfo2 *si2, VkFence fence) {
    uint64_t first[4] = {0};
    uint32_t cbs = 0;
    for (uint32_t i = 0; i < n; i++) {
        uint32_t c = si ? si[i].commandBufferCount : si2[i].commandBufferInfoCount;
        for (uint32_t j = 0; j < c; j++, cbs++)
            if (cbs < 4) first[cbs] = si ? (uint64_t)(uintptr_t)si[i].pCommandBuffers[j] : (uint64_t)(uintptr_t)si2[i].pCommandBufferInfos[j].commandBuffer;
    }
    am_emit('S', k, n, ML_NONE, BUDGET_NO_HEAP, AM_NA, 0, "v=%d q=0x%llx cbs=%u cb=0x%llx,0x%llx,0x%llx,0x%llx fence=0x%llx", v,
            (unsigned long long)(uintptr_t)queue, cbs, (unsigned long long)first[0], (unsigned long long)first[1],
            (unsigned long long)first[2], (unsigned long long)first[3], (unsigned long long)fence);
}

// MALI_COMPAT_MEMLOG: S event every 100th submit (ADDRMAP: every one), then the GPU-timing hook if that is on, else the driver.
static VKAPI_ATTR VkResult VKAPI_CALL compat_MlQueueSubmit(VkQueue queue, uint32_t n, const VkSubmitInfo *si, VkFence fence) {
    struct DeviceData *d = get_device_data_by_cmd((VkCommandBuffer)queue);
    uint64_t k = ml_submit_tick();
    if (g_addrmap) am_submit(queue, k, 1, n, si, NULL, fence);
    if ((g_gpu_time || g_dc_on) && d->gt && d->gt->fpQueueSubmit) return compat_QueueSubmit(queue, n, si, fence);
    return d->ml_fpQueueSubmit(queue, n, si, fence);
}
static VKAPI_ATTR VkResult VKAPI_CALL compat_MlQueueSubmit2(VkQueue queue, uint32_t n, const VkSubmitInfo2 *si, VkFence fence) {
    struct DeviceData *d = get_device_data_by_cmd((VkCommandBuffer)queue);
    uint64_t k = ml_submit_tick();
    if (g_addrmap) am_submit(queue, k, 2, n, NULL, si, fence);
    if ((g_gpu_time || g_dc_on) && d->gt && d->gt->fpQueueSubmit2) return compat_QueueSubmit2(queue, n, si, fence);
    return d->ml_fpQueueSubmit2(queue, n, si, fence);
}
static VKAPI_ATTR VkResult VKAPI_CALL compat_WaitForFences(VkDevice device, uint32_t n, const VkFence *fences, VkBool32 all,
                                                           uint64_t timeout) {
    struct DeviceData *d = get_device_data(device);
    uint64_t t0 = now_ns();
    VkResult r = d->gt->fpWaitForFences(device, n, fences, all, timeout);
    qstat_add(QS_WAIT, t0, 0);
    if (g_wait_callers) wc_add((uintptr_t)__builtin_return_address(0), n && fences ? (uint64_t)fences[0] : 0, WK_UNTRACKED, t0);
    for (uint32_t i = 0; g_dc_on && r == VK_SUCCESS && fences && (all || n == 1) && i < n; i++)
        if (fences[i]) dc_done(true, (uint64_t)fences[i], 0);
    return r;
}
static VKAPI_ATTR VkResult VKAPI_CALL compat_WaitSemaphores(VkDevice device, const VkSemaphoreWaitInfo *wi, uint64_t timeout) {
    struct DeviceData *d = get_device_data(device);
    uint64_t t0 = now_ns();
    VkResult r = d->gt->fpWaitSemaphores(device, wi, timeout);
    qstat_add(QS_WAIT, t0, 0);
    if (g_wait_callers) wc_add((uintptr_t)__builtin_return_address(0), 0, WK_TIMELINE, t0);
    for (uint32_t i = 0; g_dc_on && r == VK_SUCCESS && wi && wi->pSemaphores && wi->pValues &&
                         (!(wi->flags & VK_SEMAPHORE_WAIT_ANY_BIT) || wi->semaphoreCount == 1) && i < wi->semaphoreCount; i++)
        dc_done(false, (uint64_t)wi->pSemaphores[i], wi->pValues[i]);
    return r;
}
static VKAPI_ATTR VkResult VKAPI_CALL compat_QueueWaitIdle(VkQueue queue) {
    struct DeviceData *d = get_device_data_by_cmd((VkCommandBuffer)queue);
    uint64_t t0 = now_ns();
    VkResult r = d->gt->fpQueueWaitIdle(queue);
    qstat_add(QS_IDLE, t0, 0);
    return r;
}
static VKAPI_ATTR VkResult VKAPI_CALL compat_DeviceWaitIdle(VkDevice device) {
    struct DeviceData *d = get_device_data(device);
    uint64_t t0 = now_ns();
    VkResult r = d->gt->fpDeviceWaitIdle(device);
    qstat_add(QS_IDLE, t0, 0);
    return r;
}

static void pushdesc_record(struct PushDescRec *r, VkPipelineLayout layout, uint32_t set, uint32_t count,
                            const VkWriteDescriptorSet *w) {
    if (r->layout != layout || r->set != set) { r->layout = layout; r->set = set; r->count = 0; r->bad = false; }
    for (uint32_t i = 0; i < count && !r->bad; i++) {
        if (w[i].descriptorCount != 1 || w[i].pNext) { r->bad = true; break; }
        uint32_t k = 0;
        while (k < r->count && !(r->w[k].dstBinding == w[i].dstBinding && r->w[k].dstArrayElement == w[i].dstArrayElement)) k++;
        if (k == PUSHDESC_MAX) { r->bad = true; break; }
        if (k == r->count) r->count++;
        r->w[k] = w[i];
        r->w[k].pBufferInfo = NULL; r->w[k].pImageInfo = NULL; r->w[k].pTexelBufferView = NULL;
        if (w[i].pBufferInfo) { r->buf[k] = w[i].pBufferInfo[0]; r->w[k].pBufferInfo = &r->buf[k]; }
        if (w[i].pImageInfo) { r->img[k] = w[i].pImageInfo[0]; r->w[k].pImageInfo = &r->img[k]; }
        if (w[i].pTexelBufferView) { r->view[k] = w[i].pTexelBufferView[0]; r->w[k].pTexelBufferView = &r->view[k]; }
    }
}

// The image a meta pass reads: vkd3d binds its copy source with a push descriptor (GPU timing details). Compute
// pushes are also recorded for the restore after a BC decode (struct PushDescRec).
static VKAPI_ATTR void VKAPI_CALL compat_CmdPushDescriptorSetKHR(VkCommandBuffer cmd, VkPipelineBindPoint bp, VkPipelineLayout layout,
                                                                 uint32_t set, uint32_t count, const VkWriteDescriptorSet *w) {
    struct CmdBufData *c = cmd_data(cmd);
    for (uint32_t i = 0; c && i < count; i++)
        if ((w[i].descriptorType == VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE || w[i].descriptorType == VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER ||
             w[i].descriptorType == VK_DESCRIPTOR_TYPE_STORAGE_IMAGE || w[i].descriptorType == VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT) &&
            w[i].pImageInfo && w[i].descriptorCount) {
            c->gt_src = w[i].pImageInfo[0].imageView;
            break;
        }
    if (g_rt_pipeline && bp == VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR) bp = VK_PIPELINE_BIND_POINT_COMPUTE;
    struct PushDescRec *r = !c ? NULL : bp == VK_PIPELINE_BIND_POINT_COMPUTE ? &c->cpd
                          : bp == VK_PIPELINE_BIND_POINT_GRAPHICS ? &c->gpd : NULL;
    if (g_dc_on) dc_push_writes(c, bp, layout, set, count, w, false);
    if (r && g_vendor_id) pushdesc_record(r, layout, set, count, w);
    // With MALI_COMPAT_VENDOR_ID, vkd3d pushes only the root descriptors that changed and expects the others to stay.
    // Test (2026-10-28): forward the whole merged set every time, in case this driver drops bindings a push leaves out.
    if (g_vendor_id && r && !r->bad && r->count) { count = r->count; w = r->w; }
    (c ? c->dev : get_device_data_by_cmd(cmd))->fpCmdPushDescriptorSetKHR(cmd, bp, layout, set, count, w);
}

// aspect: 0 color, 1 depth, 2 stencil. A read-only layout gets STORE_OP_NONE: STORE there is invalid (no write access),
// and read-only depth is what soft particles (HZD fires) bind while sampling it.
static bool layout_read_only(VkImageLayout l, int aspect) {
    switch (l) {
    case VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL: case VK_IMAGE_LAYOUT_READ_ONLY_OPTIMAL:
    case VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL: return true;
    case VK_IMAGE_LAYOUT_DEPTH_READ_ONLY_OPTIMAL: case VK_IMAGE_LAYOUT_DEPTH_READ_ONLY_STENCIL_ATTACHMENT_OPTIMAL: return aspect == 1;
    case VK_IMAGE_LAYOUT_STENCIL_READ_ONLY_OPTIMAL: case VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_STENCIL_READ_ONLY_OPTIMAL: return aspect == 2;
    default: return false;
    }
}

static void susres_attachment(VkRenderingAttachmentInfo *a, bool resume, bool suspend, int aspect) {
    if (a->imageView == VK_NULL_HANDLE) return;
    if (resume) a->loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
    if (suspend) {
        a->storeOp = layout_read_only(a->imageLayout, aspect) ? VK_ATTACHMENT_STORE_OP_NONE : VK_ATTACHMENT_STORE_OP_STORE;
        a->resolveMode = VK_RESOLVE_MODE_NONE;
        a->resolveImageView = VK_NULL_HANDLE;
    }
}

// SUSRES_BARRIER: called from CmdBeginRendering, so always outside a render pass instance. Masks only for the
// attachment kinds this part has: color write -> color load/draw (read/write); depth/stencil write (draws and STORE) ->
// depth/stencil LOAD/tests (read/write) and fragment shader sampling (read-only depth layouts). Global memory barrier,
// no layout change (resumed parts keep the same views and layouts). The dynamic-rendering local-read rules
// (VK_KHR_dynamic_rendering_local_read) are for barriers inside a pass and do not apply here. Falls back to
// vkCmdPipelineBarrier when synchronization2 is missing. Works across command buffers too (submission order).
static void susres_barrier(struct DeviceData *d, VkCommandBuffer cmd, const VkRenderingInfo *ri) {
    VkPipelineStageFlags2 src = 0, dst = 0;
    VkAccessFlags2 sa = 0, da = 0;
    for (uint32_t i = 0; i < ri->colorAttachmentCount; i++)
        if (ri->pColorAttachments[i].imageView != VK_NULL_HANDLE) {
            src |= VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
            sa |= VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
            dst |= VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
            da |= VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
            break;
        }
    if ((ri->pDepthAttachment && ri->pDepthAttachment->imageView != VK_NULL_HANDLE) ||
        (ri->pStencilAttachment && ri->pStencilAttachment->imageView != VK_NULL_HANDLE)) {
        src |= VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT;
        sa |= VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
        dst |= VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT |
               VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
        da |= VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT |
              VK_ACCESS_2_SHADER_SAMPLED_READ_BIT;
    }
    if (!src) return;
    atomic_fetch_add(&g_susres_barriers, 1);
    if (d->gt && d->gt->fpCmdPipelineBarrier2) {
        VkMemoryBarrier2 mb = {VK_STRUCTURE_TYPE_MEMORY_BARRIER_2, NULL, src, sa, dst, da};
        VkDependencyInfo di = {VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
        di.memoryBarrierCount = 1;
        di.pMemoryBarriers = &mb;
        d->gt->fpCmdPipelineBarrier2(cmd, &di);
    } else if (d->fpCmdPipelineBarrier) { // the used stage/access bits fit the 32-bit flags; SAMPLED_READ -> SHADER_READ
        VkMemoryBarrier mb = {VK_STRUCTURE_TYPE_MEMORY_BARRIER, NULL, (VkAccessFlags)sa,
                              (VkAccessFlags)(da & ~VK_ACCESS_2_SHADER_SAMPLED_READ_BIT) |
                              ((da & VK_ACCESS_2_SHADER_SAMPLED_READ_BIT) ? VK_ACCESS_SHADER_READ_BIT : 0)};
        d->fpCmdPipelineBarrier(cmd, (VkPipelineStageFlags)src, (VkPipelineStageFlags)dst, 0, 1, &mb, 0, NULL, 0, NULL);
    }
}

#include "ct/ct_cover_spv.h"
#include "ct/ct_pred_spv.h"
#include "cover_trim.h"

static VKAPI_ATTR void VKAPI_CALL compat_CmdBeginRendering(VkCommandBuffer cmd, const VkRenderingInfo *ri) {
    GTC(GTC_RENDERING);
    if (ri->flags & VK_RENDERING_SUSPENDING_BIT) atomic_fetch_add(&g_susp_passes, 1);
    if (ri->flags & VK_RENDERING_RESUMING_BIT) atomic_fetch_add(&g_resu_passes, 1);
    VkRenderingInfo plain;
    VkRenderingAttachmentInfo col[8], dep, ste;
    if (g_no_susres && (ri->flags & (VK_RENDERING_SUSPENDING_BIT | VK_RENDERING_RESUMING_BIT)) && ri->colorAttachmentCount <= 8) {
        bool resume = ri->flags & VK_RENDERING_RESUMING_BIT, suspend = ri->flags & VK_RENDERING_SUSPENDING_BIT;
        plain = *ri;
        plain.flags &= ~(VK_RENDERING_SUSPENDING_BIT | VK_RENDERING_RESUMING_BIT);
        for (uint32_t i = 0; i < ri->colorAttachmentCount; i++) { col[i] = ri->pColorAttachments[i]; susres_attachment(&col[i], resume, suspend, 0); }
        plain.pColorAttachments = col;
        if (ri->pDepthAttachment) { dep = *ri->pDepthAttachment; susres_attachment(&dep, resume, suspend, 1); plain.pDepthAttachment = &dep; }
        if (ri->pStencilAttachment) { ste = *ri->pStencilAttachment; susres_attachment(&ste, resume, suspend, 2); plain.pStencilAttachment = &ste; }
        if (resume && g_susres_barrier) susres_barrier(get_device_data_by_cmd(cmd), cmd, ri); // K-CS1
        ri = &plain;
    }
    // Attachment-less passes (MALI_COMPAT_NOATT_AREA=<W>x<H>): DXVK opens them for draws without render targets
    // (stream output, UAV-only) with the device's maximum size, 16384x16384 and 256 layers here. On the vendor driver
    // such a pass took ~118 ms with nothing to draw and finally hung the GPU (Witcher 3 next-gen DX11, 2026-10-30).
    // The area is clamped and the layer count set to 1; with no attachments only UAV writes outside it could be lost.
    VkRenderingInfo clamped;
    if (g_noatt_w && !ri->colorAttachmentCount && !ri->pDepthAttachment && !ri->pStencilAttachment &&
        (ri->renderArea.extent.width > g_noatt_w || ri->renderArea.extent.height > g_noatt_h || ri->layerCount > 1)) {
        clamped = *ri;
        if (clamped.renderArea.extent.width > g_noatt_w) clamped.renderArea.extent.width = g_noatt_w;
        if (clamped.renderArea.extent.height > g_noatt_h) clamped.renderArea.extent.height = g_noatt_h;
        if (!clamped.viewMask) clamped.layerCount = 1;
        atomic_fetch_add(&g_noatt_clamped, 1);
        ri = &clamped;
    }
    gt_render_begin(cmd, ri->renderArea.extent.width, ri->renderArea.extent.height, ri->colorAttachmentCount, ri);
    if (g_ct) ct_pass_begin(cmd_data(cmd), ri);
    get_device_data_by_cmd(cmd)->gt->fpCmdBeginRendering(cmd, ri);
}
static VKAPI_ATTR void VKAPI_CALL compat_CmdEndRendering(VkCommandBuffer cmd) {
    get_device_data_by_cmd(cmd)->gt->fpCmdEndRendering(cmd);
    gt_render_end(cmd);
    if (g_ct) ct_pass_end(cmd_data(cmd), cmd);
}
static VKAPI_ATTR void VKAPI_CALL compat_CmdBeginRenderPass(VkCommandBuffer cmd, const VkRenderPassBeginInfo *bi,
                                                            VkSubpassContents contents) {
    GTC(GTC_RENDERPASS);
    gt_render_begin(cmd, bi->renderArea.extent.width, bi->renderArea.extent.height, 0, NULL);
    get_device_data_by_cmd(cmd)->gt->fpCmdBeginRenderPass(cmd, bi, contents);
}
static VKAPI_ATTR void VKAPI_CALL compat_CmdEndRenderPass(VkCommandBuffer cmd) {
    get_device_data_by_cmd(cmd)->gt->fpCmdEndRenderPass(cmd);
    gt_render_end(cmd);
}
static VKAPI_ATTR void VKAPI_CALL compat_CmdBeginRenderPass2(VkCommandBuffer cmd, const VkRenderPassBeginInfo *bi,
                                                             const VkSubpassBeginInfo *si) {
    gt_render_begin(cmd, bi->renderArea.extent.width, bi->renderArea.extent.height, 0, NULL);
    get_device_data_by_cmd(cmd)->gt->fpCmdBeginRenderPass2(cmd, bi, si);
}
static VKAPI_ATTR void VKAPI_CALL compat_CmdEndRenderPass2(VkCommandBuffer cmd, const VkSubpassEndInfo *ei) {
    get_device_data_by_cmd(cmd)->gt->fpCmdEndRenderPass2(cmd, ei);
    gt_render_end(cmd);
}
static VKAPI_ATTR void VKAPI_CALL compat_CmdDispatch(VkCommandBuffer cmd, uint32_t x, uint32_t y, uint32_t z) {
    GTC(GTC_DISPATCH);
    struct CmdBufData *c = cmd_data(cmd);
    if (!c) { get_device_data_by_cmd(cmd)->fpCmdDispatch(cmd, x, y, z); return; }
    if (g_dc_on) dc_dispatch(c, cmd, x, y, z);
    int32_t i = gt_begin(c, cmd, GT_DISPATCH, x, y, z, c->compute_pipeline); // "WxH cZ" = the group counts
    if (!(g_ct && ct_dispatch(c, cmd, x, y, z))) c->dev->fpCmdDispatch(cmd, x, y, z); // cover-trim: one record, same WxH
    gt_end(c, cmd, i);
}
static VKAPI_ATTR void VKAPI_CALL compat_CmdDispatchIndirect(VkCommandBuffer cmd, VkBuffer buf, VkDeviceSize off) {
    GTC(GTC_DISPATCH_INDIRECT);
    struct CmdBufData *c = cmd_data(cmd);
    if (!c) { get_device_data_by_cmd(cmd)->gt->fpCmdDispatchIndirect(cmd, buf, off); return; }
    if (g_dc_on) dc_dispatch(c, cmd, 0, 0, 0); // indirect: group counts 0 0 0
    int32_t i = gt_begin(c, cmd, GT_DISPATCH, 0, 0, 0, c->compute_pipeline); // indirect: group counts unknown (0x0)
    c->dev->gt->fpCmdDispatchIndirect(cmd, buf, off);
    gt_end(c, cmd, i);
}
static VKAPI_ATTR void VKAPI_CALL compat_CmdDraw(VkCommandBuffer cmd, uint32_t vc, uint32_t ic, uint32_t fv, uint32_t fi) {
    GTC(GTC_DRAW);
    if (fi && ic) atomic_fetch_add(&g_div_draws, 1);
    struct CmdBufData *c = cmd_data(cmd);
    if (gt_draw(c, vc, ic)) return;
    if (g_ct && c && ct_draw(c, cmd, false, vc, ic, fv, 0, fi)) return;
    (c ? c->dev : get_device_data_by_cmd(cmd))->gt->fpCmdDraw(cmd, vc, ic, fv, fi);
}
static VKAPI_ATTR void VKAPI_CALL compat_CmdDrawIndexed(VkCommandBuffer cmd, uint32_t xc, uint32_t ic, uint32_t fx, int32_t vo,
                                                        uint32_t fi) {
    GTC(GTC_DRAW_INDEXED);
    if (fi && ic) atomic_fetch_add(&g_div_draws, 1);
    struct CmdBufData *c = cmd_data(cmd);
    if (gt_draw(c, xc, ic)) return;
    if (g_ct && c && ct_draw(c, cmd, true, xc, ic, fx, vo, fi)) return;
    (c ? c->dev : get_device_data_by_cmd(cmd))->gt->fpCmdDrawIndexed(cmd, xc, ic, fx, vo, fi);
}
static VKAPI_ATTR void VKAPI_CALL compat_CmdDrawIndirect(VkCommandBuffer cmd, VkBuffer b, VkDeviceSize o, uint32_t n, uint32_t s) {
    GTC(GTC_DRAW_INDIRECT);
    struct CmdBufData *c = cmd_data(cmd);
    if (gt_draw(c, 0, 0)) return;
    (c ? c->dev : get_device_data_by_cmd(cmd))->gt->fpCmdDrawIndirect(cmd, b, o, n, s);
}
static VKAPI_ATTR void VKAPI_CALL compat_CmdDrawIndexedIndirect(VkCommandBuffer cmd, VkBuffer b, VkDeviceSize o, uint32_t n,
                                                                uint32_t s) {
    GTC(GTC_DRAW_INDEXED_INDIRECT);
    struct CmdBufData *c = cmd_data(cmd);
    if (gt_draw(c, 0, 0)) return;
    (c ? c->dev : get_device_data_by_cmd(cmd))->gt->fpCmdDrawIndexedIndirect(cmd, b, o, n, s);
}
static VKAPI_ATTR void VKAPI_CALL compat_CmdDrawIndirectCount(VkCommandBuffer cmd, VkBuffer b, VkDeviceSize o, VkBuffer cb,
                                                              VkDeviceSize co, uint32_t n, uint32_t s) {
    GTC(GTC_DRAW_INDIRECT_COUNT);
    struct CmdBufData *c = cmd_data(cmd);
    if (gt_draw(c, 0, 0)) return;
    (c ? c->dev : get_device_data_by_cmd(cmd))->gt->fpCmdDrawIndirectCount(cmd, b, o, cb, co, n, s);
}
static VKAPI_ATTR void VKAPI_CALL compat_CmdDrawIndexedIndirectCount(VkCommandBuffer cmd, VkBuffer b, VkDeviceSize o, VkBuffer cb,
                                                                     VkDeviceSize co, uint32_t n, uint32_t s) {
    GTC(GTC_DRAW_INDEXED_INDIRECT_COUNT);
    struct CmdBufData *c = cmd_data(cmd);
    if (gt_draw(c, 0, 0)) return;
    (c ? c->dev : get_device_data_by_cmd(cmd))->gt->fpCmdDrawIndexedIndirectCount(cmd, b, o, cb, co, n, s);
}

static VKAPI_ATTR void VKAPI_CALL compat_CmdDrawMultiEXT(VkCommandBuffer cmd, uint32_t n, const VkMultiDrawInfoEXT *info,
                                                         uint32_t ic, uint32_t fi, uint32_t stride) {
    GTC(GTC_DRAW_MULTI);
    struct CmdBufData *c = cmd_data(cmd);
    if (gt_draw(c, n && info ? info->vertexCount : 0, ic)) return;
    (c ? c->dev : get_device_data_by_cmd(cmd))->gt->fpCmdDrawMultiEXT(cmd, n, info, ic, fi, stride);
}
static VKAPI_ATTR void VKAPI_CALL compat_CmdDrawMultiIndexedEXT(VkCommandBuffer cmd, uint32_t n, const VkMultiDrawIndexedInfoEXT *info,
                                                                uint32_t ic, uint32_t fi, uint32_t stride, const int32_t *vo) {
    GTC(GTC_DRAW_MULTI_INDEXED);
    struct CmdBufData *c = cmd_data(cmd);
    if (gt_draw(c, n && info ? info->indexCount : 0, ic)) return;
    (c ? c->dev : get_device_data_by_cmd(cmd))->gt->fpCmdDrawMultiIndexedEXT(cmd, n, info, ic, fi, stride, vo);
}
static VKAPI_ATTR void VKAPI_CALL compat_CmdDrawMeshTasksEXT(VkCommandBuffer cmd, uint32_t x, uint32_t y, uint32_t z) {
    GTC(GTC_MESH);
    struct CmdBufData *c = cmd_data(cmd);
    if (gt_draw(c, x, y)) return;
    (c ? c->dev : get_device_data_by_cmd(cmd))->gt->fpCmdDrawMeshTasksEXT(cmd, x, y, z);
}
static VKAPI_ATTR void VKAPI_CALL compat_CmdDrawMeshTasksIndirectEXT(VkCommandBuffer cmd, VkBuffer b, VkDeviceSize o, uint32_t n, uint32_t s) {
    GTC(GTC_MESH_INDIRECT);
    struct CmdBufData *c = cmd_data(cmd);
    if (gt_draw(c, 0, 0)) return;
    (c ? c->dev : get_device_data_by_cmd(cmd))->gt->fpCmdDrawMeshTasksIndirectEXT(cmd, b, o, n, s);
}
static VKAPI_ATTR void VKAPI_CALL compat_CmdDrawMeshTasksIndirectCountEXT(VkCommandBuffer cmd, VkBuffer b, VkDeviceSize o, VkBuffer cb,
                                                                          VkDeviceSize co, uint32_t n, uint32_t s) {
    GTC(GTC_MESH_INDIRECT_COUNT);
    struct CmdBufData *c = cmd_data(cmd);
    if (gt_draw(c, 0, 0)) return;
    (c ? c->dev : get_device_data_by_cmd(cmd))->gt->fpCmdDrawMeshTasksIndirectCountEXT(cmd, b, o, cb, co, n, s);
}
static VKAPI_ATTR void VKAPI_CALL compat_CmdDispatchBase(VkCommandBuffer cmd, uint32_t bx, uint32_t by, uint32_t bz, uint32_t x,
                                                         uint32_t y, uint32_t z) {
    GTC(GTC_DISPATCH_BASE);
    struct CmdBufData *c = cmd_data(cmd);
    if (g_dc_on) dc_dispatch(c, cmd, x, y, z);
    int32_t i = gt_begin(c, cmd, GT_DISPATCH, x, y, z, c ? c->compute_pipeline : VK_NULL_HANDLE);
    (c ? c->dev : get_device_data_by_cmd(cmd))->gt->fpCmdDispatchBase(cmd, bx, by, bz, x, y, z);
    if (c) gt_end(c, cmd, i);
}


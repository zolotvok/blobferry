// SPDX-License-Identifier: GPL-3.0-or-later
// cl_cmd.h: command recording: binds, push constants, BC upload decode, command buffer/pool lifetime. Included by mali_compat_layer.c.
// State tracking for the restore after a BC decode (see struct PushRec). Graphics binds are passed straight on.
static VKAPI_ATTR void VKAPI_CALL compat_CmdBindPipeline(VkCommandBuffer commandBuffer, VkPipelineBindPoint bindPoint,
                                                         VkPipeline pipeline) {
    if (g_rt_pipeline && rt_cmd_bind_pipeline(commandBuffer, bindPoint, pipeline)) return;
    struct CmdBufData *c = cmd_data(commandBuffer);
    VkPipeline drv = pipeline;
    if (bindPoint == VK_PIPELINE_BIND_POINT_GRAPHICS) {
        // a stand-in from MALI_COMPAT_ASYNC_PSO: the real pipeline once compiled, else drop the draws until the next bind
        struct AsyncJob *j = async_lookup(pipeline);
        drv = j ? atomic_load(&j->real) : pipeline;
        if (c) c->async_skip = !drv;
    }
    if (!c) { if (drv) get_device_data_by_cmd(commandBuffer)->fpCmdBindPipeline(commandBuffer, bindPoint, drv); return; }
    if (bindPoint == VK_PIPELINE_BIND_POINT_COMPUTE) c->compute_pipeline = pipeline;
    else if (bindPoint == VK_PIPELINE_BIND_POINT_GRAPHICS) {
        c->gfx_pipeline = pipeline;
        struct HashNode h;
        c->gfx_flags = g_gpu_time && hash_get(g_pipeline_hash, (uint64_t)pipeline, &h) ? (uint8_t)h.flags : 0;
    }
    if (drv) c->dev->fpCmdBindPipeline(commandBuffer, bindPoint, drv);
}

// Record one app push (v1 or v2) for the restore after a BC decode. Same layout/stages/range as an earlier record:
// update it and move it to the end (replay order = app order, across both entry points).
static void push_record(struct CmdBufData *c, VkPipelineLayout layout, VkShaderStageFlags stages, uint32_t offset,
                        uint32_t size, const void *pValues, bool v2) {
    uint32_t i = 0;
    while (i < c->push_count && !(c->push[i].layout == layout && c->push[i].stages == stages &&
                                  c->push[i].offset == offset && c->push[i].size == size)) i++;
    if (i == c->push_count && c->push_count == PUSH_RECS) { // full: drop the oldest
        i = 0;
        if (g_bcr) atomic_fetch_add_explicit(&g_bcm_drop, 1, memory_order_relaxed); // MALI_COMPAT_BCD_RING bcd-mix
    }
    if (i < c->push_count) {
        memmove(&c->push[i], &c->push[i + 1], (c->push_count - i - 1) * sizeof c->push[0]);
        c->push_count--;
    }
    struct PushRec *r = &c->push[c->push_count++];
    r->layout = layout;
    r->stages = stages;
    r->offset = offset;
    r->size = size;
    r->v2 = v2;
    memcpy(r->data, pValues, size);
}

static VKAPI_ATTR void VKAPI_CALL compat_CmdPushConstants(VkCommandBuffer commandBuffer, VkPipelineLayout layout,
                                                          VkShaderStageFlags stages, uint32_t offset, uint32_t size,
                                                          const void *pValues) {
    if (g_rt_pipeline) rt_cmd_push_constants(commandBuffer, layout, &stages, offset, size, pValues);
    pc2_count(PC2_V1);
    struct CmdBufData *c = cmd_data(commandBuffer);
    if (!c) {
        get_device_data_by_cmd(commandBuffer)->fpCmdPushConstants(commandBuffer, layout, stages, offset, size, pValues);
        return;
    }
    if (offset + size <= 256) push_record(c, layout, stages, offset, size, pValues, false);
    else if (g_bcr) atomic_fetch_add_explicit(&g_bcm_v1unrec, 1, memory_order_relaxed); // MALI_COMPAT_BCD_RING bcd-mix
    c->dev->fpCmdPushConstants(commandBuffer, layout, stages, offset, size, pValues);
}

// vkCmdPushConstants2 / vkCmdPushConstants2KHR (pc2-restore-1): recorded like v1. A push with layout VK_NULL_HANDLE
// (layout given as VkPipelineLayoutCreateInfo in pNext) or past 256 bytes is not recorded, only counted.
static void push2_common(VkCommandBuffer commandBuffer, const VkPushConstantsInfo *info, bool khr) {
    VkPushConstantsInfo mod_info;
    if (g_rt_pipeline && info) {
        VkShaderStageFlags rt_stages = info->stageFlags;
        rt_cmd_push_constants(commandBuffer, info->layout, &rt_stages, info->offset, info->size, info->pValues);
        if (rt_stages != info->stageFlags) {
            mod_info = *info;
            mod_info.stageFlags = rt_stages;
            info = &mod_info;
        }
    }
    pc2_count(PC2_V2);
    struct CmdBufData *c = cmd_data(commandBuffer);
    struct DeviceData *d = c ? c->dev : get_device_data_by_cmd(commandBuffer);
    if (info && (info->layout == VK_NULL_HANDLE || info->offset + info->size > 256))
        pc2_count(PC2_V2_UNREC);
    else if (c && info)
        push_record(c, info->layout, info->stageFlags, info->offset, info->size, info->pValues, true);
    PFN_vkCmdPushConstants2 fp = khr ? d->fpCmdPushConstants2KHR : d->fpCmdPushConstants2;
    fp(commandBuffer, info);
}
static VKAPI_ATTR void VKAPI_CALL compat_CmdPushConstants2(VkCommandBuffer commandBuffer, const VkPushConstantsInfo *info) {
    push2_common(commandBuffer, info, false);
}
static VKAPI_ATTR void VKAPI_CALL compat_CmdPushConstants2KHR(VkCommandBuffer commandBuffer, const VkPushConstantsInfo *info) {
    push2_common(commandBuffer, info, true);
}

static void *pc2_log_thread(void *arg) {
    (void)arg;
    for (;;) {
        sleep(10);
        unsigned long long v1 = pc2_sum(PC2_V1), v2 = pc2_sum(PC2_V2), un = pc2_sum(PC2_V2_UNREC),
                           dec = pc2_sum(PC2_DECODES), dv2 = pc2_sum(PC2_DECODES_V2);
        if (v1 | v2 | un | dec | dv2)
            fprintf(stderr, "mali-compat: pc2: v1 pushes %llu, v2 pushes %llu, v2 unrecorded %llu, BC decodes %llu, "
                    "decodes replaying v2 %llu\n", v1, v2, un, dec, dv2);
        static unsigned long long last_enter; // this thread only
        unsigned long long be = atomic_load_explicit(&g_bcd_enter, memory_order_relaxed);
        if (bcd_log_on() && be != last_enter) {
            last_enter = be;
            fprintf(stderr, "mali-compat: bc-dec (pid %d): enter %llu nobda %llu done %llu\n", (int)getpid(), be,
                    atomic_load_explicit(&g_bcd_nobda, memory_order_relaxed),
                    atomic_load_explicit(&g_bcd_done, memory_order_relaxed));
        }
        sgl_summary(); // MALI_COMPAT_SG_LOG
        bcr_tick(v1, dec); // MALI_COMPAT_BCD_RING
        if (g_bcr) bcm_line(false); // MALI_COMPAT_BCD_RING: bcd-mix, when a number changed
    }
    return NULL;
}

static VKAPI_ATTR void VKAPI_CALL compat_CmdBindDescriptorBuffersEXT(VkCommandBuffer commandBuffer, uint32_t count,
                                                                     const VkDescriptorBufferBindingInfoEXT *infos) {
    struct CmdBufData *c = cmd_data(commandBuffer);
    if (c) c->descriptor_buffers = true;
    get_device_data_by_cmd(commandBuffer)->fpCmdBindDescriptorBuffersEXT(commandBuffer, count, infos);
}

// BC upload: decode every region on the GPU into one staging buffer (RGBA8 / RG16 / RGBA16F texels), then copy that
// into the RGBA8 shadow image. Array layers are decoded like 3D slices (same buffer layout). Rewritten
// [Claude Opus 5.5] 2026-10-26: one allocation per call (was one per region), layers (only layer 0 was decoded and the
// copy read past the staging buffer), allocation errors logged.
static VKAPI_ATTR void VKAPI_CALL compat_CmdCopyBufferToImage(
    VkCommandBuffer commandBuffer, VkBuffer srcBuffer, VkImage dstImage,
    VkImageLayout dstImageLayout, uint32_t regionCount, const VkBufferImageCopy *pRegions) {
    struct DeviceData *d = get_device_data_by_cmd(commandBuffer);
    struct TrackedBCImage tracked;

    if (!d || !find_bc_image(dstImage, &tracked)) {
        d->fpCmdCopyBufferToImage(commandBuffer, srcBuffer, dstImage, dstImageLayout, regionCount, pRegions);
        return;
    }
    atomic_fetch_add_explicit(&g_bcd_enter, 1, memory_order_relaxed);
    if (!d->fpGetBufferDeviceAddress) { // see bda_ext_on: this was a call through a null pointer
        if (atomic_fetch_add_explicit(&g_bcd_nobda, 1, memory_order_relaxed) == 0)
            fprintf(stderr, "mali-compat: BC decode: this device has no vkGetBufferDeviceAddress (app on Vulkan < 1.2 "
                    "without VK_KHR_buffer_device_address); BC uploads are skipped, textures stay undecoded. "
                    "MALI_COMPAT_BDA_EXT=1 enables the extension\n");
        return;
    }
    if (init_bc_compute(d) != VK_SUCCESS || !regionCount) return;
    uint64_t t0 = g_stats ? now_ns() : 0;

    VkBufferDeviceAddressInfo src_bda_info = {VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO, NULL, srcBuffer};
    VkDeviceAddress src_bda = d->fpGetBufferDeviceAddress(d->device, &src_bda_info);
    uint32_t shader_idx = panvk_bc_shader_index(tracked.orig_format);
    uint32_t blk_B = panvk_bc_block_size(tracked.orig_format);
    uint32_t texel_B = panvk_bc_decoded_texel_size(tracked.orig_format);

    VkBufferImageCopy stack_regs[16];
    VkBufferImageCopy *dec = regionCount <= 16 ? stack_regs : malloc(regionCount * sizeof *dec);
    VkDeviceSize *dst_off = malloc(regionCount * sizeof *dst_off);
    if (!dec || !dst_off) { if (dec != stack_regs) free(dec); free(dst_off); return; }
    VkDeviceSize total = 0;
    for (uint32_t i = 0; i < regionCount; i++) {
        const VkBufferImageCopy *reg = &pRegions[i];
        uint32_t slices = (reg->imageExtent.depth ? reg->imageExtent.depth : 1) * (reg->imageSubresource.layerCount ? reg->imageSubresource.layerCount : 1);
        dst_off[i] = total;
        total += (VkDeviceSize)reg->imageExtent.width * reg->imageExtent.height * slices * texel_B;
        total = (total + 63) & ~(VkDeviceSize)63;
    }

    struct StagingResource staging;
    VkResult r = acquire_staging(d, total, &staging);
    if (r != VK_SUCCESS) {
        fprintf(stderr, "mali-compat: BC upload: staging buffer of %llu bytes failed (%d); texture left undecoded\n",
                (unsigned long long)total, r);
        if (dec != stack_regs) free(dec);
        free(dst_off);
        return;
    }
    add_staging_resource(commandBuffer, staging);
    VkBuffer staging_buf = staging.buffer;
    VkDeviceAddress dst_bda = staging.addr;

    // The source was written by the host (upload/staging buffers) or by an earlier copy or compute dispatch. Waiting
    // on those stages only: ALL_COMMANDS made every BC upload wait for all earlier rendering in the queue (a
    // fragment drain) before the decode could start. [Claude Opus 5.5] 2026-10-26
    VkMemoryBarrier pre_b = {VK_STRUCTURE_TYPE_MEMORY_BARRIER, NULL,
                             VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_HOST_WRITE_BIT | VK_ACCESS_SHADER_WRITE_BIT,
                             VK_ACCESS_SHADER_READ_BIT};
    d->fpCmdPipelineBarrier(commandBuffer,
                            VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_HOST_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &pre_b, 0, NULL, 0, NULL);
    struct CmdBufData *cstate = cmd_data(commandBuffer);
    static struct CmdBufData no_state;
    if (!cstate) cstate = &no_state;
    if (g_bcr) bcm_decode(cstate->push_count, cstate->gfx_pipeline || cstate->compute_pipeline); // BCD_RING bcd-mix
    VkPipeline decoder = cstate->descriptor_buffers && d->bc_pipelines_db[shader_idx] ? d->bc_pipelines_db[shader_idx]
                                                                                      : d->bc_pipelines[shader_idx];
    d->fpCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, decoder);
    for (uint32_t i = 0; i < regionCount; i++) {
        const VkBufferImageCopy *reg = &pRegions[i];
        uint32_t w = reg->imageExtent.width, h = reg->imageExtent.height;
        uint32_t slices = (reg->imageExtent.depth ? reg->imageExtent.depth : 1) * (reg->imageSubresource.layerCount ? reg->imageSubresource.layerCount : 1);
        uint32_t row_px = reg->bufferRowLength ? reg->bufferRowLength : w;
        uint32_t col_px = reg->bufferImageHeight ? reg->bufferImageHeight : h;
        uint32_t src_row_B = ((row_px + 3) / 4) * blk_B;
        uint32_t src_slice_B = ((col_px + 3) / 4) * src_row_B;
        uint64_t full_src = src_bda + reg->bufferOffset, full_dst = dst_bda + dst_off[i];
        struct panvk_bc_decode_push push = {
            .src = { (uint32_t)full_src, (uint32_t)(full_src >> 32) },
            .dst = { (uint32_t)full_dst, (uint32_t)(full_dst >> 32) },
            .src_row_B = src_row_B,
            .src_slice_B = src_slice_B,
            .dst_row_B = w * texel_B,
            .dst_slice_B = w * h * texel_B,
            .width = (int32_t)w,
            .height = (int32_t)h,
            .depth = (int32_t)slices,
            .format = tracked.orig_format,
            .x0 = 0,
            .y0 = 0,
        };
        d->fpCmdPushConstants(commandBuffer, d->bc_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);
        d->fpCmdDispatch(commandBuffer, (w + 7) / 8, (h + 7) / 8, slices);
        if (g_bcr) // MALI_COMPAT_BCD_RING: the bytes the shader reads (last block of the last row of the last slice) / writes
            bcr_add((uint64_t)(uintptr_t)commandBuffer, (uint64_t)srcBuffer, full_src,
                    w && h ? (uint64_t)(slices - 1) * src_slice_B + (uint64_t)((h - 1) / 4) * src_row_B + (uint64_t)((w + 3) / 4) * blk_B : 0,
                    full_dst, (uint64_t)w * h * slices * texel_B, tracked.orig_format, w, h, slices, src_row_B, src_slice_B);
        dec[i] = *reg;
        dec[i].bufferOffset = dst_off[i];
        dec[i].bufferRowLength = 0;
        dec[i].bufferImageHeight = 0;
    }
    VkMemoryBarrier post_b = {VK_STRUCTURE_TYPE_MEMORY_BARRIER, NULL, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT};
    d->fpCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1,
                            &post_b, 0, NULL, 0, NULL);
    d->fpCmdCopyBufferToImage(commandBuffer, staging_buf, dstImage, dstImageLayout, regionCount, dec);
    // Give the app its compute pipeline and push constants back (H1).
    if (cstate->compute_pipeline)
        d->fpCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, cstate->compute_pipeline);
    // Each record with the entry point the app used (pc2-restore-1); v2 without the app's pNext (only a layout create
    // info can be there, and such pushes are not recorded).
    bool replayed_v2 = false;
    for (uint32_t i = 0; i < cstate->push_count; i++) {
        const struct PushRec *pr = &cstate->push[i];
        PFN_vkCmdPushConstants2 fp2 = d->fpCmdPushConstants2KHR ? d->fpCmdPushConstants2KHR : d->fpCmdPushConstants2;
        if (pr->v2 && fp2) {
            VkPushConstantsInfo pi = {VK_STRUCTURE_TYPE_PUSH_CONSTANTS_INFO, NULL, pr->layout, pr->stages, pr->offset,
                                      pr->size, pr->data};
            fp2(commandBuffer, &pi);
            replayed_v2 = true;
        } else {
            d->fpCmdPushConstants(commandBuffer, pr->layout, pr->stages, pr->offset, pr->size, pr->data);
        }
    }
    pc2_count(PC2_DECODES);
    atomic_fetch_add_explicit(&g_bcd_done, 1, memory_order_relaxed);
    if (replayed_v2) pc2_count(PC2_DECODES_V2);
    // Only with MALI_COMPAT_VENDOR_ID: the recorded writes can name image/buffer views the app has destroyed since,
    // and pushing them again crashed Horizon Zero Dawn's benchmark load (C0000005, 2026-10-28 22:49).
    if (g_vendor_id && cstate->cpd.layout && !cstate->cpd.bad && cstate->cpd.count && d->fpCmdPushDescriptorSetKHR)
        d->fpCmdPushDescriptorSetKHR(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, cstate->cpd.layout, cstate->cpd.set,
                                     cstate->cpd.count, cstate->cpd.w);
    if (dec != stack_regs) free(dec);
    free(dst_off);
    if (g_stats) {
        uint64_t t1 = now_ns();
        atomic_fetch_add(&g_st_uploads, 1);
        atomic_fetch_add(&g_st_regions, regionCount);
        atomic_fetch_add(&g_st_bytes, total);
        atomic_fetch_add(&g_st_ns, t1 - t0);
        uint64_t last = atomic_load(&g_st_last_ns);
        if (!last && g_stats_interval_ns) atomic_compare_exchange_strong(&g_st_last_ns, &last, t1);
        else if (t1 - last >= g_stats_interval_ns && atomic_compare_exchange_strong(&g_st_last_ns, &last, t1)) {
            pthread_mutex_lock(&g_staging_pool_mutex);
            unsigned long long pool = d->staging_pool_bytes;
            pthread_mutex_unlock(&g_staging_pool_mutex);
            fprintf(stderr, "mali-compat: stats %.1fs: BC uploads %llu (regions %llu, decoded %.1f MiB), staging new %llu "
                    "reused %llu, CPU in BC path %.2f ms, pool %.1f MiB\n", last ? (t1 - last) / 1e9 : 0.0,
                    (unsigned long long)atomic_exchange(&g_st_uploads, 0), (unsigned long long)atomic_exchange(&g_st_regions, 0),
                    atomic_exchange(&g_st_bytes, 0) / 1048576.0, (unsigned long long)atomic_exchange(&g_st_new, 0),
                    (unsigned long long)atomic_exchange(&g_st_reused, 0), atomic_exchange(&g_st_ns, 0) / 1e6, pool / 1048576.0);
        }
    }
}

static VKAPI_ATTR void VKAPI_CALL compat_CmdCopyBufferToImage2(
    VkCommandBuffer commandBuffer, const VkCopyBufferToImageInfo2 *pCopyBufferToImageInfo) {
    struct DeviceData *d = get_device_data_by_cmd(commandBuffer);
    struct TrackedBCImage tracked;

    if (!d || !find_bc_image(pCopyBufferToImageInfo->dstImage, &tracked)) {
        if (d && d->fpCmdCopyBufferToImage2) {
            d->fpCmdCopyBufferToImage2(commandBuffer, pCopyBufferToImageInfo);
        }
        return;
    }

    uint32_t rc = pCopyBufferToImageInfo->regionCount;
    VkBufferImageCopy *regions = (VkBufferImageCopy *)malloc(rc * sizeof(VkBufferImageCopy));
    if (!regions) return;
    for (uint32_t i = 0; i < rc; i++) {
        const VkBufferImageCopy2 *r2 = &pCopyBufferToImageInfo->pRegions[i];
        regions[i] = (VkBufferImageCopy){
            .bufferOffset = r2->bufferOffset,
            .bufferRowLength = r2->bufferRowLength,
            .bufferImageHeight = r2->bufferImageHeight,
            .imageSubresource = r2->imageSubresource,
            .imageOffset = r2->imageOffset,
            .imageExtent = r2->imageExtent,
        };
    }
    compat_CmdCopyBufferToImage(commandBuffer, pCopyBufferToImageInfo->srcBuffer,
                                pCopyBufferToImageInfo->dstImage, pCopyBufferToImageInfo->dstImageLayout,
                                rc, regions);
    free(regions);
}

static VKAPI_ATTR VkResult VKAPI_CALL compat_ResetCommandBuffer(
    VkCommandBuffer commandBuffer, VkCommandBufferResetFlags flags) {
    struct DeviceData *d = get_device_data_by_cmd(commandBuffer);
    if (d) reset_cmd_data(d, commandBuffer, false);
    return d ? d->fpResetCommandBuffer(commandBuffer, flags) : VK_SUCCESS;
}

static VKAPI_ATTR VkResult VKAPI_CALL compat_BeginCommandBuffer(
    VkCommandBuffer commandBuffer, const VkCommandBufferBeginInfo *pBeginInfo) {
    struct DeviceData *d = get_device_data_by_cmd(commandBuffer);
    if (d) reset_cmd_data(d, commandBuffer, false);
    return d ? d->fpBeginCommandBuffer(commandBuffer, pBeginInfo) : VK_SUCCESS;
}

static VKAPI_ATTR void VKAPI_CALL compat_FreeCommandBuffers(
    VkDevice device, VkCommandPool commandPool, uint32_t commandBufferCount,
    const VkCommandBuffer *pCommandBuffers) {
    struct DeviceData *d = get_device_data(device);
    if (g_addrmap)
        am_emit('N', (uint64_t)commandPool, AM_NA, ML_NONE, BUDGET_NO_HEAP, AM_NA, 0, "what=cb-free count=%u first=0x%llx",
                commandBufferCount, commandBufferCount && pCommandBuffers ? (unsigned long long)(uintptr_t)pCommandBuffers[0] : 0ull);
    if (d) {
        for (uint32_t i = 0; i < commandBufferCount; i++) {
            reset_cmd_data(d, pCommandBuffers[i], true);
        }
    }
    d->fpFreeCommandBuffers(device, commandPool, commandBufferCount, pCommandBuffers);
}

static VKAPI_ATTR VkResult VKAPI_CALL compat_AllocateCommandBuffers(
    VkDevice device, const VkCommandBufferAllocateInfo *pAllocateInfo, VkCommandBuffer *pCommandBuffers) {
    struct DeviceData *d = get_device_data(device);
    if (!d || !d->fpAllocateCommandBuffers) return VK_ERROR_INITIALIZATION_FAILED;
    VkResult res = d->fpAllocateCommandBuffers(device, pAllocateInfo, pCommandBuffers);
    if (g_addrmap && pAllocateInfo)
        am_emit('N', (uint64_t)pAllocateInfo->commandPool, AM_NA, ML_NONE, BUDGET_NO_HEAP, AM_NA, 0,
                "what=cb-alloc count=%u level=%d first=0x%llx result=%d", pAllocateInfo->commandBufferCount, pAllocateInfo->level,
                res == VK_SUCCESS && pCommandBuffers && pAllocateInfo->commandBufferCount ? (unsigned long long)(uintptr_t)pCommandBuffers[0] : 0ull,
                res);
    if (res == VK_SUCCESS && pAllocateInfo && pCommandBuffers) {
        for (uint32_t i = 0; i < pAllocateInfo->commandBufferCount; i++) {
            struct CmdBufData *e = cmd_data(pCommandBuffers[i]);
            if (e) {
                e->dev = d;
                e->pool = pAllocateInfo->commandPool;
            }
        }
    }
    return res;
}

static VKAPI_ATTR VkResult VKAPI_CALL compat_ResetCommandPool(
    VkDevice device, VkCommandPool commandPool, VkCommandPoolResetFlags flags) {
    // F1: only the staging buffers go back here (the pool's command buffers may be begun again only after the GPU is
    // done with them, and must be: vkBeginCommandBuffer runs reset_cmd_data, which clears the tracked state). So only
    // the g_staged entries are visited, not the whole table. The pool is externally synchronized with its command
    // buffers, so no other thread touches these entries meanwhile.
    struct DeviceData *d = get_device_data(device);
    if (g_addrmap) am_emit('N', (uint64_t)commandPool, AM_NA, ML_NONE, BUDGET_NO_HEAP, AM_NA, 0, "what=cpool-reset flags=0x%x", flags);
    if (d) {
        struct CmdBufData *mine = NULL;
        pthread_mutex_lock(&g_cmdbuf_mutex);
        for (struct CmdBufData *e = g_staged, *nx; e; e = nx) {
            nx = e->st_next;
            if (e->dev == d && e->pool == commandPool) {
                staged_unlink(e);
                e->st_next = mine; // private chain until released below (st_on is false again)
                mine = e;
            }
        }
        pthread_mutex_unlock(&g_cmdbuf_mutex);
        for (struct CmdBufData *e = mine, *nx; e; e = nx) {
            nx = e->st_next;
            e->st_next = NULL;
            for (uint32_t j = 0; j < e->staging_count; j++) release_staging(d, &e->staging_resources[j]);
            e->staging_count = 0;
        }
        if (g_rt_pipeline) {
            pthread_mutex_lock(&g_cmdbuf_mutex);
            for (int b = 0; b < CMDBUF_BUCKETS; b++) {
                for (struct CmdBufData *e = g_cmdbufs[b]; e; e = e->next) {
                    if (e->dev == d && e->pool == commandPool) {
                        rt_reset_cmd_state(e->cmdbuf, false);
                    }
                }
            }
            pthread_mutex_unlock(&g_cmdbuf_mutex);
        }
    }
    return d ? d->fpResetCommandPool(device, commandPool, flags) : VK_SUCCESS;
}

static VKAPI_ATTR void VKAPI_CALL compat_DestroyCommandPool(
    VkDevice device, VkCommandPool commandPool, const VkAllocationCallbacks *pAllocator) {
    struct DeviceData *d = get_device_data(device);
    if (g_addrmap && commandPool) am_emit('N', (uint64_t)commandPool, AM_NA, ML_NONE, BUDGET_NO_HEAP, AM_NA, 0, "what=cpool-destroy");
    if (d) {
        pthread_mutex_lock(&g_cmdbuf_mutex);
        for (int b = 0; b < CMDBUF_BUCKETS; b++) {
            struct CmdBufData **pp = &g_cmdbufs[b];
            while (*pp) {
                struct CmdBufData *e = *pp;
                if (e->dev == d && e->pool == commandPool) {
                    *pp = e->next;
                    staged_unlink(e);
                    if (g_rt_pipeline) rt_reset_cmd_state(e->cmdbuf, true);
                    for (uint32_t j = 0; j < e->staging_count; j++)
                        release_staging(d, &e->staging_resources[j]);
                    free(e->staging_resources);
                    free(e);
                    atomic_fetch_add_explicit(&g_cmdbuf_gen, 1, memory_order_release);
                } else {
                    pp = &e->next;
                }
            }
        }
        pthread_mutex_unlock(&g_cmdbuf_mutex);
        d->fpDestroyCommandPool(device, commandPool, pAllocator);
    }
}


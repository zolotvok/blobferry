// SPDX-License-Identifier: GPL-3.0-or-later
/*
 * mali_rt_emu.h — VK_KHR_ray_tracing_pipeline Emulation over VK_KHR_ray_query
 *
 * Included directly by mali_compat_layer.c (after init_bc_compute).
 *
 * Architecture:
 *   1. Intercepts RT VkShaderModules before they reach the vendor Mali driver
 *      (which lacks OpCapability RayTracingKHR) and stores their SPIR-V in
 *      RtShaderModule nodes.
 *   2. Supports VK_KHR_pipeline_library, VkPipelineLibraryCreateInfoKHR,
 *      VK_EXT_pipeline_library_group_handles, VK_KHR_ray_tracing_maintenance1,
 *      and VK_KHR_deferred_host_operations.
 *   3. Lowers and links RT shader stages (rgen, rchit, rahit, rmiss, rint, rcall)
 *      into a single GLCompute megakernel (LocalSize 8x4x1 = 32 threads = 1 Mali
 *      warp) backed by hardware OpTypeRayQueryKHR / OpRayQueryProceedKHR.
 *   4. Encodes deterministic 32-byte position-independent RtGroupHandle records
 *      into the application's SBT buffers so the megakernel reads stage_ids and
 *      ShaderRecordBufferKHR local root arguments directly from SBT GPU memory.
 */

#ifndef MALI_RT_EMU_H
#define MALI_RT_EMU_H

#include <link.h>
#include <elf.h>

#include "rt_core.h"
#include "rt_spv_emit.h"
#include "rt_mk_parse.h"
#include "rt_mk_emit.h"
#include "rt_pipeline.h"
#include "rt_trace.h"
#include "rt_accel.h"
#include "rt_report.h"

#endif /* MALI_RT_EMU_H */

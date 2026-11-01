// SPDX-License-Identifier: GPL-3.0-or-later
// K-DW PC test: with MALI_COMPAT_GT_DYNDEPTH the per-pass dw / eq draw counters follow DXVK-style dynamic depth state.
// Real path: compat_CreateGraphicsPipelines (fake driver) -> pipeline table -> compat_CmdBindPipeline ->
// compat_CmdSetDepth{Test,Write}Enable / CompareOp (fake driver) -> compat_CmdDraw -> the secondary-cmd-buffer counters
// (DXVK tiler mode: no pass record in the command buffer). Also: switch off = old static behaviour, vkd3d-style static
// pipelines unchanged, the hooks forward the values, vkGetDeviceProcAddr hands the hooks out only with the switch.
// Run: bridge/test/gt-dyndepth/run.sh [layer source .c]   [keeper, Claude Opus 5.5] 2026-11-01
#include LAYER_SRC
#include <assert.h>

static int fails;
#define CHECK(c) do { if (!(c)) { fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); fails++; } } while (0)

static void *fake_dispatch[4];
static struct { void *key; } dev_obj = {fake_dispatch}, cmd_obj = {fake_dispatch};
#define DEV ((VkDevice)&dev_obj)
#define CMD ((VkCommandBuffer)&cmd_obj)
static uint64_t next_pipe = 0x1000;
static int drv_draws, drv_dt = -1, drv_dw = -1, drv_op = -1;
static VKAPI_ATTR VkResult VKAPI_CALL fake_create(VkDevice d, VkPipelineCache pc, uint32_t n, const VkGraphicsPipelineCreateInfo *ci,
                                                  const VkAllocationCallbacks *a, VkPipeline *p) {
    for (uint32_t i = 0; i < n; i++) p[i] = (VkPipeline)(uintptr_t)(next_pipe += 0x10);
    return VK_SUCCESS;
}
static VKAPI_ATTR void VKAPI_CALL fake_bind(VkCommandBuffer c, VkPipelineBindPoint bp, VkPipeline p) {}
static VKAPI_ATTR void VKAPI_CALL fake_draw(VkCommandBuffer c, uint32_t vc, uint32_t ic, uint32_t fv, uint32_t fi) { drv_draws++; }
static VKAPI_ATTR void VKAPI_CALL fake_dt(VkCommandBuffer c, VkBool32 e) { drv_dt = (int)e; }
static VKAPI_ATTR void VKAPI_CALL fake_dw(VkCommandBuffer c, VkBool32 e) { drv_dw = (int)e; }
static VKAPI_ATTR void VKAPI_CALL fake_op(VkCommandBuffer c, VkCompareOp o) { drv_op = (int)o; }
static PFN_vkVoidFunction VKAPI_CALL fake_gdpa(VkDevice d, const char *n) { return (PFN_vkVoidFunction)fake_op; }

static VkPipeline make_pipe(bool dynamic, VkBool32 dt, VkBool32 dw, VkCompareOp op) {
    static const VkDynamicState dyn[] = {VK_DYNAMIC_STATE_VIEWPORT_WITH_COUNT, VK_DYNAMIC_STATE_DEPTH_TEST_ENABLE,
                                         VK_DYNAMIC_STATE_DEPTH_WRITE_ENABLE, VK_DYNAMIC_STATE_DEPTH_COMPARE_OP};
    VkPipelineDynamicStateCreateInfo dy = {VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO, NULL, 0, 4, dyn};
    VkPipelineDepthStencilStateCreateInfo ds = {VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
    ds.depthTestEnable = dt; ds.depthWriteEnable = dw; ds.depthCompareOp = op;
    VkGraphicsPipelineCreateInfo ci = {VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
    ci.pDepthStencilState = &ds;
    ci.pDynamicState = dynamic ? &dy : NULL;
    VkPipeline p = VK_NULL_HANDLE;
    CHECK(compat_CreateGraphicsPipelines(DEV, VK_NULL_HANDLE, 1, &ci, NULL, &p) == VK_SUCCESS);
    return p;
}

// one secondary command buffer: bind, (dynamic state), n draws; returns the dw / eq counters
static struct GtCnt record(VkPipeline p, int dt, int dw, int op, int ndraw) {
    struct CmdBufData *c = cmd_data(CMD);
    c->gt_rec = -1; c->gt_sdraws = 0; memset(&c->gt_scnt, 0, sizeof c->gt_scnt); c->dyn_depth = 0;
    compat_CmdBindPipeline(CMD, VK_PIPELINE_BIND_POINT_GRAPHICS, p);
    if (dt >= 0) compat_CmdSetDepthTestEnable(CMD, (VkBool32)dt);
    if (dw >= 0) compat_CmdSetDepthWriteEnable(CMD, (VkBool32)dw);
    if (op >= 0) compat_CmdSetDepthCompareOp(CMD, (VkCompareOp)op);
    for (int i = 0; i < ndraw; i++) compat_CmdDraw(CMD, 3, 1, 0, 0);
    return c->gt_scnt;
}

int main(void) {
    struct GpuTime *gt = calloc(1, sizeof *gt);
    gt->fpCmdDraw = fake_draw;
    struct DeviceData *d = &g_devices[0];
    d->device = DEV; d->dispatch_key = fake_dispatch; d->gt = gt; g_device_count = 1;
    d->fpCreateGraphicsPipelines = fake_create; d->fpCmdBindPipeline = fake_bind;
    d->fpCmdSetDepthTestEnable = fake_dt; d->fpCmdSetDepthWriteEnable = fake_dw; d->fpCmdSetDepthCompareOp = fake_op;
    d->get_device_proc_addr = fake_gdpa;
    g_gpu_time = true;

    // 1. switch off: a DXVK-style pipeline (all depth state dynamic, static block zero) counts dw 0 even when the app
    //    turns depth write on: the old behaviour, and the hooks are not handed out
    g_gt_dyndepth = false;
    VkPipeline dx_off = make_pipe(true, VK_FALSE, VK_FALSE, VK_COMPARE_OP_NEVER);
    struct GtCnt n = record(dx_off, 1, 1, VK_COMPARE_OP_GREATER_OR_EQUAL, 5);
    CHECK(n.dw == 0 && n.eq == 0);
    CHECK(vkGetDeviceProcAddr(DEV, "vkCmdSetDepthWriteEnable") != (PFN_vkVoidFunction)compat_CmdSetDepthWriteEnable);
    CHECK(vkGetDeviceProcAddr(DEV, "vkCmdSetDepthTestEnable") != (PFN_vkVoidFunction)compat_CmdSetDepthTestEnable);

    // 2. switch on
    g_gt_dyndepth = true;
    CHECK(vkGetDeviceProcAddr(DEV, "vkCmdSetDepthWriteEnable") == (PFN_vkVoidFunction)compat_CmdSetDepthWriteEnable);
    CHECK(vkGetDeviceProcAddr(DEV, "vkCmdSetDepthWriteEnableEXT") == (PFN_vkVoidFunction)compat_CmdSetDepthWriteEnable);
    CHECK(vkGetDeviceProcAddr(DEV, "vkCmdSetDepthTestEnable") == (PFN_vkVoidFunction)compat_CmdSetDepthTestEnable);
    CHECK(vkGetDeviceProcAddr(DEV, "vkCmdSetDepthCompareOp") == (PFN_vkVoidFunction)compat_CmdSetDepthCompareOp);
    VkPipeline dx = make_pipe(true, VK_FALSE, VK_FALSE, VK_COMPARE_OP_NEVER);
    // G-buffer-style: test + write on
    n = record(dx, 1, 1, VK_COMPARE_OP_GREATER_OR_EQUAL, 7);
    CHECK(n.dw == 7 && n.eq == 0);
    CHECK(drv_dt == 1 && drv_dw == 1 && drv_op == VK_COMPARE_OP_GREATER_OR_EQUAL); // forwarded unchanged
    // post pass: depth test off (write state stale on): no dw
    n = record(dx, 0, 1, VK_COMPARE_OP_ALWAYS, 4);
    CHECK(n.dw == 0 && n.eq == 0);
    // decal / equal pass: test on, write off, EQUAL
    n = record(dx, 1, 0, VK_COMPARE_OP_EQUAL, 3);
    CHECK(n.dw == 0 && n.eq == 3);
    // state changes inside one command buffer apply from the next draw on
    {
        struct CmdBufData *c = cmd_data(CMD);
        c->gt_rec = -1; c->gt_sdraws = 0; memset(&c->gt_scnt, 0, sizeof c->gt_scnt); c->dyn_depth = 0;
        compat_CmdBindPipeline(CMD, VK_PIPELINE_BIND_POINT_GRAPHICS, dx);
        compat_CmdSetDepthTestEnable(CMD, VK_TRUE);
        compat_CmdSetDepthWriteEnable(CMD, VK_TRUE);
        compat_CmdDraw(CMD, 3, 1, 0, 0);
        compat_CmdDraw(CMD, 3, 1, 0, 0);
        compat_CmdSetDepthWriteEnable(CMD, VK_FALSE);
        compat_CmdDraw(CMD, 3, 1, 0, 0);
        CHECK(c->gt_scnt.dw == 2 && c->gt_sdraws == 3);
        // dynamic values survive a rebind (Vulkan keeps dynamic state across binds of pipelines that declare it)
        compat_CmdSetDepthWriteEnable(CMD, VK_TRUE);
        compat_CmdBindPipeline(CMD, VK_PIPELINE_BIND_POINT_GRAPHICS, dx);
        compat_CmdDraw(CMD, 3, 1, 0, 0);
        CHECK(c->gt_scnt.dw == 3);
    }
    // 3. vkd3d-style static pipelines: unchanged by the switch and by stale dynamic values
    VkPipeline st_w = make_pipe(false, VK_TRUE, VK_TRUE, VK_COMPARE_OP_LESS);
    VkPipeline st_eq = make_pipe(false, VK_TRUE, VK_FALSE, VK_COMPARE_OP_EQUAL);
    VkPipeline st_off = make_pipe(false, VK_FALSE, VK_TRUE, VK_COMPARE_OP_LESS);
    n = record(st_w, -1, -1, -1, 2);   CHECK(n.dw == 2 && n.eq == 0);
    n = record(st_eq, -1, -1, -1, 2);  CHECK(n.dw == 0 && n.eq == 2);
    n = record(st_off, -1, -1, -1, 2); CHECK(n.dw == 0 && n.eq == 0);
    // 4. mixed: only write dynamic, test static on, op static EQUAL
    {
        static const VkDynamicState dyn[] = {VK_DYNAMIC_STATE_DEPTH_WRITE_ENABLE};
        VkPipelineDynamicStateCreateInfo dy = {VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO, NULL, 0, 1, dyn};
        VkPipelineDepthStencilStateCreateInfo ds = {VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
        ds.depthTestEnable = VK_TRUE; ds.depthWriteEnable = VK_TRUE; ds.depthCompareOp = VK_COMPARE_OP_EQUAL;
        VkGraphicsPipelineCreateInfo ci = {VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
        ci.pDepthStencilState = &ds; ci.pDynamicState = &dy;
        VkPipeline p = VK_NULL_HANDLE;
        CHECK(compat_CreateGraphicsPipelines(DEV, VK_NULL_HANDLE, 1, &ci, NULL, &p) == VK_SUCCESS);
        n = record(p, -1, 0, -1, 2); CHECK(n.dw == 0 && n.eq == 2);  // static write ignored, dynamic 0 -> EQUAL read
        n = record(p, -1, 1, -1, 2); CHECK(n.dw == 2 && n.eq == 0);
    }
    printf("draws forwarded to the driver: %d\n", drv_draws);
    CHECK(drv_draws == 5 + 7 + 4 + 3 + 4 + 6 + 4);
    printf("%s\n", fails ? "FAILED" : "PASS");
    return fails ? 1 : 0;
}

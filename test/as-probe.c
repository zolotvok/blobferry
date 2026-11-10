// SPDX-License-Identifier: GPL-3.0-or-later
// as-probe: measures how a Vulkan driver's acceleration structure builder behaves (sizes, bytes written, overlapping
// GENERIC handles, in-place refits, type changes, ordering against ray queries), without any game. One arm per run:
//   as-probe sizes | extent | overlap batch|serial [blas] | update single|batch|tlasupd | mutate generic|recreate |
//            race nobarrier|barrier|twocmd|twocmdnb [grid] [iters]
//   v2 (2026-11-10): copy clone|compact [big] | updsrc [big|bigdst] | formats | vfmt <format> | multigeom
//   v3 (2026-11-10 late): refitchain <sub> | copyupd clone|compact | bufclone | copyext <sub> | aliasupd (layer-strategy arms;
//            every step is measured + traced, the verdict names the first failing step)
// Every output line starts "AS-PROBE <label>"; the last line is "AS-PROBE <label> PASS|FAIL|SKIP|VKERROR|TIMEOUT <reason>",
// exit 0/1/2/3/4. Every fence wait has a 20 s timeout, a 60 s watchdog (alarm) covers calls stuck inside the driver.
// Why: vkd3d-proton places several GENERIC ASes with size = rest of the buffer into one 8 MB buffer and refits in place;
// the phone shows GPU MMU faults past such buffers (docs/games/witcher3.md, RTX). Ray checks share the rq-smoke idea
// (docs/planning/rt-poc.md): known rays along +z, hit / t / custom index / primitive index checked against the CPU.
// [keeper, Claude Opus 5.5] 2026-11-10
#define _POSIX_C_SOURCE 200809L
#include <math.h>
#include <signal.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <vulkan/vulkan.h>
#include "as_probe_spv.h"

#define MB (1024ull * 1024ull)
#define KB64 65536ull
#define MAXRAYS 1024
#define WAIT_NS 20000000000ull
#define BLASF VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR
#define TLASF VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR
#define GENF VK_ACCELERATION_STRUCTURE_TYPE_GENERIC_KHR
#define FT VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR
#define FB VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_BUILD_BIT_KHR
#define AU VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR
#define AC VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_COMPACTION_BIT_KHR

static char LABEL[64] = "?";
static const char *PHASE = "init";
static char wd_msg[400];
static char FIRSTFAIL[200];   // v3 arms: first failing step; appended to VKERROR / TIMEOUT / crash lines (empty for older arms)
static VkBufferUsageFlags XUSE;   // extra buffer usage bits (bufclone: TRANSFER_SRC|DST); 0 for every older arm
static const char *const VERD[] = {"PASS", "FAIL", "SKIP", "VKERROR", "TIMEOUT"};

static void say(const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    printf("AS-PROBE %s ", LABEL);
    vprintf(fmt, ap);
    putchar('\n');
    fflush(stdout);
    va_end(ap);
}
static void die(int code, const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    printf("AS-PROBE %s %s ", LABEL, VERD[code]);
    vprintf(fmt, ap);
    if (FIRSTFAIL[0] && code >= 3) printf(" [first failing step before this: %s]", FIRSTFAIL);
    putchar('\n');
    fflush(stdout);
    va_end(ap);
    _exit(code);
}
static void on_alarm(int s) {
    (void)s;
    ssize_t w = write(1, wd_msg, strlen(wd_msg));
    (void)w;
    _exit(4);
}
static void on_crash(int s) {   // a driver crash still ends with one verdict line (VKERROR, exit 3)
    const char *n = s == SIGSEGV ? "SIGSEGV" : s == SIGBUS ? "SIGBUS" : s == SIGILL ? "SIGILL" : s == SIGFPE ? "SIGFPE" : "SIGABRT";
    const char *parts[10] = {"AS-PROBE ", LABEL, " VKERROR crash ", n, " in phase ", PHASE, FIRSTFAIL[0] ? " [first failing step before this: " : "",
                             FIRSTFAIL, FIRSTFAIL[0] ? "]" : "", "\n"};
    for (int i = 0; i < 10; i++) { ssize_t w = write(1, parts[i], strlen(parts[i])); (void)w; }
    _exit(3);
}
static void phase(const char *p) {
    PHASE = p;
    snprintf(wd_msg, sizeof wd_msg, "AS-PROBE %s TIMEOUT watchdog 60 s stuck in phase %s%s%s%s\n", LABEL, p,
             FIRSTFAIL[0] ? " [first failing step before this: " : "", FIRSTFAIL, FIRSTFAIL[0] ? "]" : "");
    alarm(60);
}
#define CK(x) do { VkResult r_ = (x); if (r_ != VK_SUCCESS) die(3, "phase %s: %s = %d (line %d)", PHASE, #x, r_, __LINE__); } while (0)

static double now_ms(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec * 1e3 + t.tv_nsec / 1e6;
}
static VkDeviceSize up(VkDeviceSize v, VkDeviceSize a) { return (v + a - 1) / a * a; }

// ---------------------------------------------------------------- device
static VkPhysicalDevice pd;
static VkDevice dev;
static VkQueue queue;
static VkCommandPool pool;
static VkCommandBuffer cbs[2];
static VkFence fence;
static VkPhysicalDeviceMemoryProperties memp;
static VkPhysicalDeviceAccelerationStructurePropertiesKHR asp = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_PROPERTIES_KHR};
static VkDeviceSize SA = 1;   // minAccelerationStructureScratchOffsetAlignment
static PFN_vkGetBufferDeviceAddress pBDA;
static PFN_vkGetAccelerationStructureBuildSizesKHR pSizes;
static PFN_vkCreateAccelerationStructureKHR pCreateAS;
static PFN_vkDestroyAccelerationStructureKHR pDestroyAS;
static PFN_vkCmdBuildAccelerationStructuresKHR pBuild;
static PFN_vkGetAccelerationStructureDeviceAddressKHR pASAddr;

// msize = mapped allocation size: size + an optional guard tail that is GPU-mapped but outside the VkBuffer
struct Buf { VkBuffer b; VkDeviceMemory m; uint8_t *p; VkDeviceAddress a; VkDeviceSize size, msize; bool coh; };
static struct Buf bufs[64];
static int nbufs;
static VkAccelerationStructureKHR ases[64];
static int nases;
static bool memtype_said;

static struct Buf mkbuf_g(VkDeviceSize size, VkDeviceSize guard_tail) {
    struct Buf r = {0};
    r.size = size;
    VkBufferCreateInfo bi = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, NULL, 0, size,
        VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR |
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | XUSE, VK_SHARING_MODE_EXCLUSIVE, 0, NULL};
    CK(vkCreateBuffer(dev, &bi, NULL, &r.b));
    VkMemoryRequirements req;
    vkGetBufferMemoryRequirements(dev, r.b, &req);
    const VkMemoryPropertyFlags V = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT, C = VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                                K = VK_MEMORY_PROPERTY_HOST_CACHED_BIT;
    const VkMemoryPropertyFlags pref[4] = {V | C | K, V | C, V | K, V};
    uint32_t type = UINT32_MAX;
    for (int p = 0; p < 4 && type == UINT32_MAX; p++)
        for (uint32_t i = 0; i < memp.memoryTypeCount; i++)
            if ((req.memoryTypeBits >> i & 1) && (memp.memoryTypes[i].propertyFlags & pref[p]) == pref[p]) { type = i; break; }
    if (type == UINT32_MAX) die(2, "no host-visible memory type for an AS/scratch buffer (memoryTypeBits 0x%x)", req.memoryTypeBits);
    r.coh = memp.memoryTypes[type].propertyFlags & C;
    if (!memtype_said) {
        memtype_said = true;
        say("mem type=%u flags=0x%x coherent=%d buffer_align=%llu", type, memp.memoryTypes[type].propertyFlags, r.coh,
            (unsigned long long)req.alignment);
    }
    VkMemoryAllocateFlagsInfo fl = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO, NULL, VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT, 0};
    r.msize = up(req.size + guard_tail, req.alignment ? req.alignment : 1);
    VkMemoryAllocateInfo ai = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, &fl, r.msize, type};
    CK(vkAllocateMemory(dev, &ai, NULL, &r.m));
    CK(vkBindBufferMemory(dev, r.b, r.m, 0));
    void *p;
    CK(vkMapMemory(dev, r.m, 0, VK_WHOLE_SIZE, 0, &p));
    r.p = p;
    VkBufferDeviceAddressInfo da = {VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO, NULL, r.b};
    r.a = pBDA(dev, &da);
    if (nbufs < 64) bufs[nbufs++] = r;
    return r;
}
static struct Buf mkbuf(VkDeviceSize size) { return mkbuf_g(size, 0); }
static void hflush(const struct Buf *b) {
    if (b->coh) return;
    VkMappedMemoryRange r = {VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE, NULL, b->m, 0, VK_WHOLE_SIZE};
    CK(vkFlushMappedMemoryRanges(dev, 1, &r));
}
static void hinval(const struct Buf *b) {
    if (b->coh) return;
    VkMappedMemoryRange r = {VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE, NULL, b->m, 0, VK_WHOLE_SIZE};
    CK(vkInvalidateMappedMemoryRanges(dev, 1, &r));
}
static void fill(const struct Buf *b, uint8_t pat) { memset(b->p, pat, b->msize); hflush(b); }

static VkAccelerationStructureKHR mkas(const struct Buf *b, VkDeviceSize off, VkDeviceSize size, VkAccelerationStructureTypeKHR t) {
    VkAccelerationStructureCreateInfoKHR ci = {VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR, NULL, 0, b->b, off, size, t, 0};
    VkAccelerationStructureKHR as;
    CK(pCreateAS(dev, &ci, NULL, &as));
    if (nases < 64) ases[nases++] = as;
    return as;
}
static void rmas(VkAccelerationStructureKHR as) {
    for (int i = 0; i < nases; i++) if (ases[i] == as) { ases[i] = ases[--nases]; break; }
    pDestroyAS(dev, as, NULL);
}
static VkDeviceAddress as_addr(VkAccelerationStructureKHR as) {
    VkAccelerationStructureDeviceAddressInfoKHR ai = {VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR, NULL, as};
    return pASAddr(dev, &ai);
}

// ---------------------------------------------------------------- command buffers
static void begin(VkCommandBuffer cb) {
    CK(vkResetCommandBuffer(cb, 0));
    VkCommandBufferBeginInfo bgi = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, NULL, VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT, NULL};
    CK(vkBeginCommandBuffer(cb, &bgi));
}
static void fullbar(VkCommandBuffer cb) {   // ALL_COMMANDS -> ALL_COMMANDS, AS + shader + transfer read/write (as the layer inserts)
    const VkAccessFlags a = VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR | VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR |
                            VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
    VkMemoryBarrier mb = {VK_STRUCTURE_TYPE_MEMORY_BARRIER, NULL, a, a};
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 1, &mb, 0, NULL, 0, NULL);
}
static void end(VkCommandBuffer cb) {   // device writes -> host reads
    VkMemoryBarrier mb = {VK_STRUCTURE_TYPE_MEMORY_BARRIER, NULL, VK_ACCESS_MEMORY_WRITE_BIT, VK_ACCESS_HOST_READ_BIT};
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &mb, 0, NULL, 0, NULL);
    CK(vkEndCommandBuffer(cb));
}
static void wait_fence(const char *what) {
    VkResult r = vkWaitForFences(dev, 1, &fence, VK_TRUE, WAIT_NS);
    if (r == VK_TIMEOUT) die(4, "phase %s: GPU did not finish in 20 s (%s)", PHASE, what);
    if (r != VK_SUCCESS) die(3, "phase %s: vkWaitForFences = %d%s (%s)", PHASE, r, r == VK_ERROR_DEVICE_LOST ? " DEVICE_LOST" : "", what);
}
static double submit_wait(VkCommandBuffer cb, const char *what) {
    double t0 = now_ms();
    CK(vkResetFences(dev, 1, &fence));
    VkSubmitInfo si = {VK_STRUCTURE_TYPE_SUBMIT_INFO, NULL, 0, NULL, NULL, 1, &cb, 0, NULL};
    CK(vkQueueSubmit(queue, 1, &si, fence));
    wait_fence(what);
    return now_ms() - t0;
}

// ---------------------------------------------------------------- geometry
// Grid mesh nx x ny quads over [x0,x1]x[y0,y1], per-vertex z; quad (i,j) = tri 2k (v00,v10,v11) + tri 2k+1 (v00,v11,v01).
struct Mesh { uint32_t nx, ny, nv, nt; float x0, y0, x1, y1; float *z; struct Buf vb, ib; };

static struct Mesh mk_mesh(uint32_t nx, uint32_t ny, float x0, float y0, float x1, float y1) {
    struct Mesh m = {nx, ny, (nx + 1) * (ny + 1), 2 * nx * ny, x0, y0, x1, y1, NULL, {0}, {0}};
    m.z = calloc(m.nv, sizeof(float));
    m.vb = mkbuf(up((VkDeviceSize)m.nv * 12, 256));
    m.ib = mkbuf(up((VkDeviceSize)m.nt * 12, 256));
    uint32_t *ix = (uint32_t *)m.ib.p;
    for (uint32_t j = 0; j < ny; j++)
        for (uint32_t i = 0; i < nx; i++) {
            uint32_t v00 = j * (nx + 1) + i, v10 = v00 + 1, v01 = v00 + nx + 1, v11 = v01 + 1, k = 6 * (j * nx + i);
            ix[k] = v00; ix[k + 1] = v10; ix[k + 2] = v11; ix[k + 3] = v00; ix[k + 4] = v11; ix[k + 5] = v01;
        }
    hflush(&m.ib);
    return m;
}
static void mesh_z(struct Mesh *m, float base, float amp, float ph) {
    float *v = (float *)m->vb.p;
    for (uint32_t j = 0; j <= m->ny; j++)
        for (uint32_t i = 0; i <= m->nx; i++) {
            uint32_t k = j * (m->nx + 1) + i;
            m->z[k] = base + amp * sinf(ph + 0.5f * i + 0.7f * j);
            v[3 * k] = m->x0 + (m->x1 - m->x0) * i / m->nx;
            v[3 * k + 1] = m->y0 + (m->y1 - m->y0) * j / m->ny;
            v[3 * k + 2] = m->z[k];
        }
    hflush(&m->vb);
}
static VkAccelerationStructureGeometryKHR mesh_geom(const struct Mesh *m) {
    VkAccelerationStructureGeometryKHR g = {VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR};
    g.geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR;
    g.flags = VK_GEOMETRY_OPAQUE_BIT_KHR;
    g.geometry.triangles.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR;
    g.geometry.triangles.vertexFormat = VK_FORMAT_R32G32B32_SFLOAT;
    g.geometry.triangles.vertexData.deviceAddress = m->vb.a;
    g.geometry.triangles.vertexStride = 12;
    g.geometry.triangles.maxVertex = m->nv - 1;
    g.geometry.triangles.indexType = VK_INDEX_TYPE_UINT32;
    g.geometry.triangles.indexData.deviceAddress = m->ib.a;
    return g;
}
static VkAccelerationStructureGeometryKHR inst_geom(VkDeviceAddress a) {
    VkAccelerationStructureGeometryKHR g = {VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR};
    g.geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR;
    g.flags = VK_GEOMETRY_OPAQUE_BIT_KHR;
    g.geometry.instances.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR;
    g.geometry.instances.data.deviceAddress = a;
    return g;
}
static void put_inst(const struct Buf *b, uint32_t idx, VkDeviceAddress ref, float tx, float ty, float tz, uint32_t custom) {
    VkAccelerationStructureInstanceKHR in = {{{{1, 0, 0, tx}, {0, 1, 0, ty}, {0, 0, 1, tz}}}, custom, 0xFF, 0,
                                             VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR, ref};
    memcpy(b->p + (size_t)idx * sizeof in, &in, sizeof in);
}

// Expected result of one ray (origin (x, y, 0), direction +z).
// geomp1 = expected geometry index + 1 (0: not checked; every arm before multigeom/vfmt leaves it 0).
struct Exp { float x, y; int hit; float t; int32_t custom, prim, geomp1; };

// Sample point s on a mesh (local coords) + its exact t and primitive index; tx/ty/tz = instance translation.
static struct Exp mesh_pt(const struct Mesh *m, uint32_t s, float tx, float ty, float tz, int32_t custom) {
    uint32_t i = (s * 37u + 1) % m->nx, j = (s * 61u + 2) % m->ny, w = m->nx + 1;
    float u = (s & 1) ? 0.2f : 0.7f, v = (s & 1) ? 0.7f : 0.2f;
    float z00 = m->z[j * w + i], z10 = m->z[j * w + i + 1], z01 = m->z[(j + 1) * w + i], z11 = m->z[(j + 1) * w + i + 1];
    struct Exp e = {m->x0 + (i + u) * (m->x1 - m->x0) / m->nx + tx, m->y0 + (j + v) * (m->y1 - m->y0) / m->ny + ty, 1, 0, custom, 0, 0};
    if (u >= v) { e.t = z00 + u * (z10 - z00) + v * (z11 - z10) + tz; e.prim = (int32_t)(2 * (j * m->nx + i)); }
    else { e.t = z00 + u * (z11 - z01) + v * (z01 - z00) + tz; e.prim = (int32_t)(2 * (j * m->nx + i) + 1); }
    return e;
}
static struct Exp miss_at(float x, float y) { struct Exp e = {x, y, 0, 0, -1, -1, 0}; return e; }

// ---------------------------------------------------------------- builds
struct Job { VkAccelerationStructureTypeKHR type; VkBuildAccelerationStructureFlagsKHR flags; bool update;
             VkAccelerationStructureKHR dst; VkAccelerationStructureGeometryKHR g; uint32_t prims; VkDeviceAddress scratch; };

static VkAccelerationStructureBuildSizesInfoKHR sizes_of(VkAccelerationStructureTypeKHR t, VkBuildAccelerationStructureFlagsKHR fl,
                                                         const VkAccelerationStructureGeometryKHR *g, uint32_t prims) {
    VkAccelerationStructureBuildGeometryInfoKHR bi = {VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR};
    bi.type = t;
    bi.flags = fl;
    bi.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
    bi.geometryCount = 1;
    bi.pGeometries = g;
    VkAccelerationStructureBuildSizesInfoKHR sz = {VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR};
    pSizes(dev, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR, &bi, &prims, &sz);
    return sz;
}
static void rec_builds(VkCommandBuffer cb, struct Job *j, uint32_t n) {
    VkAccelerationStructureBuildGeometryInfoKHR bi[8];
    VkAccelerationStructureBuildRangeInfoKHR rg[8];
    const VkAccelerationStructureBuildRangeInfoKHR *pr[8];
    for (uint32_t k = 0; k < n; k++) {
        VkAccelerationStructureBuildGeometryInfoKHR b = {VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR};
        b.type = j[k].type;
        b.flags = j[k].flags;
        b.mode = j[k].update ? VK_BUILD_ACCELERATION_STRUCTURE_MODE_UPDATE_KHR : VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
        b.srcAccelerationStructure = j[k].update ? j[k].dst : VK_NULL_HANDLE;
        b.dstAccelerationStructure = j[k].dst;
        b.geometryCount = 1;
        b.pGeometries = &j[k].g;
        b.scratchData.deviceAddress = j[k].scratch;
        bi[k] = b;
        VkAccelerationStructureBuildRangeInfoKHR r = {j[k].prims, 0, 0, 0};
        rg[k] = r;
        pr[k] = &rg[k];
    }
    pBuild(cb, n, bi, pr);
}
static void rec_build(VkCommandBuffer cb, struct Job j) { rec_builds(cb, &j, 1); }

// ---------------------------------------------------------------- ray checks
static VkPipeline rqpipe;
static VkPipelineLayout pl;
static VkDescriptorSet ds;
static struct Buf RB, OB;   // rays, results
struct PC { uint32_t mode, n, reps, pad; float area[4]; float expt[2]; float tol; uint32_t pad2; };

static void bind_tlas(VkAccelerationStructureKHR t) {
    VkWriteDescriptorSetAccelerationStructureKHR wa = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR, NULL, 1, &t};
    VkWriteDescriptorSet w = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, &wa, ds, 0, 0, 1, VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, NULL, NULL, NULL};
    vkUpdateDescriptorSets(dev, 1, &w, 0, NULL);
}
static void put_rays(const struct Exp *e, uint32_t n) {
    float *r = (float *)RB.p;
    for (uint32_t i = 0; i < n; i++) { r[4 * i] = e[i].x; r[4 * i + 1] = e[i].y; r[4 * i + 2] = 0; r[4 * i + 3] = 100; }
    hflush(&RB);
    memset(OB.p, 0, 16 + 16 * (size_t)n);
    hflush(&OB);
}
static void rec_trace(VkCommandBuffer cb, uint32_t n) {
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, rqpipe);
    vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pl, 0, 1, &ds, 0, NULL);
    struct PC pc = {0, n, 1, 0, {0, 0, 0, 0}, {0, 0}, 0, 0};
    vkCmdPushConstants(cb, pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof pc, &pc);
    vkCmdDispatch(cb, (n + 63) / 64, 1, 1);
}
static int printed_bad;
// Compares the results with e[]; prints at most 4 mismatches per run; returns the number of wrong rays.
static int check(const struct Exp *e, uint32_t n, const char *what) {
    hinval(&OB);
    const uint32_t *u = (const uint32_t *)(OB.p + 16);
    int bad = 0;
    for (uint32_t i = 0; i < n; i++) {
        const uint32_t *q = u + 4 * i;
        float t;
        memcpy(&t, &q[1], 4);
        bool ran = (q[0] & 0xFFFF0000u) == 0xC0DE0000u, hit = q[0] & 1;
        bool good = ran && hit == (e[i].hit != 0) &&
                    (!hit || (fabsf(t - e[i].t) < 2e-3f * fmaxf(1.0f, e[i].t) && (int32_t)q[2] == e[i].custom &&
                              (e[i].prim < 0 || (int32_t)q[3] == e[i].prim) && (!e[i].geomp1 || (int32_t)(q[0] >> 8 & 0xFF) == e[i].geomp1 - 1)));
        if (good) continue;
        bad++;
        if (printed_bad < 4) {
            printed_bad++;
            say("MISMATCH %s ray %u (%.3f,%.3f) want %s t=%.4f cust=%d prim=%d geom=%d got %s t=%.4f cust=%d prim=%d geom=%d", what, i, e[i].x,
                e[i].y, e[i].hit ? "hit" : "miss", e[i].t, e[i].custom, e[i].prim, e[i].geomp1 - 1, !ran ? "NOT-RUN" : hit ? "hit" : "miss", t,
                (int)q[2], (int)q[3], hit ? (int)(q[0] >> 8 & 0xFF) : -1);
        }
    }
    return bad;
}
// Standalone trace: bind t, record a dispatch, submit, wait, check.
static int trace(VkAccelerationStructureKHR t, const struct Exp *e, uint32_t n, const char *what) {
    bind_tlas(t);
    put_rays(e, n);
    begin(cbs[0]);
    fullbar(cbs[0]);
    rec_trace(cbs[0], n);
    end(cbs[0]);
    submit_wait(cbs[0], what);
    return check(e, n, what);
}

// ---------------------------------------------------------------- byte extent scanning
#define NPG 16384   // 4 KB pages tracked (64 MB)
struct Ext { int64_t lo, hi; uint8_t pg[NPG]; };   // absolute byte offsets [lo, hi), lo = -1: nothing; pg = written 4 KB pages
static void ext_reset(struct Ext *e) { e->lo = -1; e->hi = -1; memset(e->pg, 0, sizeof e->pg); }
static void scan(const struct Buf *b, uint8_t pat, struct Ext *e) {
    hinval(b);
    const uint64_t pp = 0x0101010101010101ull * pat;
    const uint64_t *w = (const uint64_t *)b->p;
    size_t nw = b->msize / 8;
    for (size_t k = 0; k < nw; k++) {
        if (w[k] == pp) continue;
        for (int c = 0; c < 8; c++) {
            if (b->p[8 * k + c] == pat) continue;
            int64_t o = (int64_t)(8 * k + c);
            if (e->lo < 0 || o < e->lo) e->lo = o;
            if (o + 1 > e->hi) e->hi = o + 1;
            if (o / 4096 < NPG) e->pg[o / 4096] = 1;
        }
    }
}
// Page segments of e relative to base, as "[a,b)" list (at most 6).
static void segs(const struct Ext *e, int64_t base, char *out, size_t cap) {
    out[0] = 0;
    int n = 0;
    size_t len = 0;
    for (int p = 0; p < NPG; p++) {
        if (!e->pg[p] || (p > 0 && e->pg[p - 1])) continue;
        int q = p;
        while (q < NPG && e->pg[q]) q++;
        if (n++ < 6 && len < cap) len += snprintf(out + len, cap - len, "[%lld,%lld)", (long long)(p * 4096ll - base), (long long)(q * 4096ll - base));
    }
    if (n > 6 && len < cap) snprintf(out + len, cap - len, "+%d", n - 6);
}
enum { F_BELOW = 1, F_BEYOND_SIZE = 2, F_BEYOND_STORAGE = 4 };
// Formats "<name>_written=[lo,hi) <name>_size=N" relative to base; returns F_* flags.
static int ext_fmt(char *out, size_t cap, const char *name, const struct Ext *e, int64_t base, VkDeviceSize drv, VkDeviceSize storage) {
    if (e->lo < 0) { snprintf(out, cap, "%s_written=none %s_size=%llu", name, name, (unsigned long long)drv); return 0; }
    int64_t lo = e->lo - base, hi = e->hi - base;
    int f = (lo < 0 ? F_BELOW : 0) | (hi > (int64_t)drv ? F_BEYOND_SIZE : 0) | (storage && hi > (int64_t)storage ? F_BEYOND_STORAGE : 0);
    snprintf(out, cap, "%s_written=[%lld,%lld) %s_size=%llu", name, (long long)lo, (long long)hi, name, (unsigned long long)drv);
    return f;
}
static void flagstr(char *out, size_t cap, int fa, int fs, bool rest) {
    out[0] = 0;
    size_t l = 0;
    if (fa & F_BELOW) l += snprintf(out + l, cap - l, "AS_BELOW,");
    if (fa & F_BEYOND_SIZE) l += snprintf(out + l, cap - l, rest ? "USES_TAIL,AS_BEYOND_SIZE," : "AS_BEYOND_SIZE,");
    if (fa & F_BEYOND_STORAGE) l += snprintf(out + l, cap - l, "AS_BEYOND_STORAGE,");
    if (fs & F_BELOW) l += snprintf(out + l, cap - l, "SCR_BELOW,");
    if (fs & F_BEYOND_SIZE) l += snprintf(out + l, cap - l, "SCR_BEYOND_SIZE,");
    if (l) out[l - 1] = 0; else snprintf(out, cap, "-");
}

// guard check: bytes [from, to) of b must still be pat; returns the count, records the first offset
static uint64_t guard(const struct Buf *b, VkDeviceSize from, VkDeviceSize to, uint8_t pat, int64_t *first) {
    hinval(b);
    uint64_t n = 0;
    for (VkDeviceSize o = from; o < to && o < b->msize; o++)
        if (b->p[o] != pat) { if (!n && *first < 0) *first = (int64_t)o; n++; }
    return n;
}
// ================================================================ arm: sizes
static int arm_sizes(char *why, size_t cap) {
    phase("sizes");
    double t0 = now_ms();
    static const uint32_t tris[4] = {1, 100, 10000, 100000};
    static const VkBuildAccelerationStructureFlagsKHR bf[3] = {FT, FB | AU, FT | AC};
    static const char *const bn[3] = {"FAST_TRACE", "FAST_BUILD|ALLOW_UPDATE", "FAST_TRACE|ALLOW_COMPACTION"};
    for (int i = 0; i < 4; i++)
        for (int f = 0; f < 3; f++) {
            VkAccelerationStructureGeometryKHR g = {VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR};
            g.geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR;
            g.flags = VK_GEOMETRY_OPAQUE_BIT_KHR;
            g.geometry.triangles.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR;
            g.geometry.triangles.vertexFormat = VK_FORMAT_R32G32B32_SFLOAT;
            g.geometry.triangles.vertexStride = 12;
            uint32_t verts = tris[i] == 1 ? 3 : tris[i] / 2 + 2 * (uint32_t)sqrt(tris[i]) + 2;   // grid-like vertex count
            g.geometry.triangles.maxVertex = verts - 1;
            g.geometry.triangles.indexType = verts <= 65536 ? VK_INDEX_TYPE_UINT16 : VK_INDEX_TYPE_UINT32;
            VkAccelerationStructureBuildSizesInfoKHR s = sizes_of(BLASF, bf[f], &g, tris[i]);
            say("blas tris=%u verts=%u idx=%d flags=%s as=%llu build=%llu update=%llu as_per_tri=%.1f", tris[i], verts,
                verts <= 65536 ? 16 : 32, bn[f], (unsigned long long)s.accelerationStructureSize, (unsigned long long)s.buildScratchSize,
                (unsigned long long)s.updateScratchSize, (double)s.accelerationStructureSize / tris[i]);
        }
    static const uint32_t insts[3] = {1, 800, 1900};
    for (int i = 0; i < 3; i++)
        for (int f = 0; f < 2; f++) {
            VkAccelerationStructureGeometryKHR g = inst_geom(0);
            VkAccelerationStructureBuildSizesInfoKHR s = sizes_of(TLASF, bf[f], &g, insts[i]);
            say("tlas instances=%u flags=%s as=%llu build=%llu update=%llu as_per_inst=%.1f", insts[i], bn[f],
                (unsigned long long)s.accelerationStructureSize, (unsigned long long)s.buildScratchSize,
                (unsigned long long)s.updateScratchSize, (double)s.accelerationStructureSize / insts[i]);
        }
    say("time sizes ms=%.2f", now_ms() - t0);
    snprintf(why, cap, "18 size queries answered");
    return 0;
}

// ================================================================ arm: extent
// One case: a 10000-triangle BLAS (tlas_case false) or an 800-instance TLAS, AS at 4 MB in a 16 MB pattern buffer.
static int extent_case(bool tlas_case, bool exact, int *bad_rays) {
    char lab[32];
    snprintf(lab, sizeof lab, "%s-%s", tlas_case ? "tlas" : "blas", exact ? "exact" : "rest");
    static struct Buf S, SC, HT, HTS, HI, SB, SBS, IB;
    static struct Mesh M, SM;
    static VkAccelerationStructureKHR small_blas;
    static bool init;
    if (!init) {
        init = true;
        phase("extent-setup");
        M = mk_mesh(100, 50, -2, -1, 2, 1);   // 10000 triangles
        SM = mk_mesh(1, 1, -0.04f, -0.04f, 0.04f, 0.04f);
        mesh_z(&SM, 0, 0, 0);
        HI = mkbuf(256);   // helper TLAS (1 instance) over the BLAS case
        VkAccelerationStructureGeometryKHR hg = inst_geom(HI.a);
        VkAccelerationStructureBuildSizesInfoKHR hs = sizes_of(TLASF, FT, &hg, 1);
        HT = mkbuf(up(hs.accelerationStructureSize, 256));
        HTS = mkbuf(hs.buildScratchSize + SA);
        IB = mkbuf(800 * 64);
        VkAccelerationStructureGeometryKHR sg = mesh_geom(&SM);
        VkAccelerationStructureBuildSizesInfoKHR ss = sizes_of(BLASF, FT, &sg, SM.nt);
        SB = mkbuf(up(ss.accelerationStructureSize, 256));
        SBS = mkbuf(ss.buildScratchSize + SA);
        small_blas = mkas(&SB, 0, ss.accelerationStructureSize, BLASF);
        begin(cbs[0]);
        rec_build(cbs[0], (struct Job){BLASF, FT, false, small_blas, sg, SM.nt, up(SBS.a, SA)});
        end(cbs[0]);
        submit_wait(cbs[0], "small BLAS build");
    }
    phase(lab);
    // geometry state k: BLAS vertices z = 3 + 0.05k (wave phase 0.3k); TLAS instances z = 3 + 0.05k, x shifted 0.002k
    struct Exp ex[40];
    uint32_t nr = 0;
    VkAccelerationStructureGeometryKHR g;
    uint32_t prims;
    VkAccelerationStructureTypeKHR type = tlas_case ? TLASF : BLASF;
    #define EXT_STATE(k) do { \
        if (!tlas_case) { mesh_z(&M, 3.0f + 0.05f * (k), 0.2f, 0.3f * (k)); } \
        else { for (uint32_t q = 0; q < 800; q++) put_inst(&IB, q, as_addr(small_blas), -2.0f + ((q % 40) + 0.5f) * 0.1f + 0.002f * (k), \
                                                             -1.0f + ((q / 40) + 0.5f) * 0.1f, 3.0f + 0.05f * (k), q); hflush(&IB); } } while (0)
    #define EXT_RAYS(k) do { nr = 0; \
        if (!tlas_case) { for (uint32_t s = 0; s < 32; s++) ex[nr++] = mesh_pt(&M, s, 0, 0, 0, 5); } \
        else { for (uint32_t s = 0; s < 32; s++) { uint32_t q = (s * 97u + 13) % 800; \
                 struct Exp e = {-2.0f + ((q % 40) + 0.5f) * 0.1f + 0.002f * (k), -1.0f + ((q / 40) + 0.5f) * 0.1f, 1, 3.0f + 0.05f * (k), (int32_t)q, -1, 0}; ex[nr++] = e; } } \
        for (uint32_t s = 0; s < 8; s++) ex[nr++] = tlas_case ? miss_at(-2.0f + (s * 5 % 40) * 0.1f + 0.002f * (k), -0.95f + 0.1f * s) : miss_at(2.5f, -0.9f + 0.25f * s); } while (0)
    EXT_STATE(0);
    if (tlas_case) { g = inst_geom(IB.a); prims = 800; } else { g = mesh_geom(&M); prims = M.nt; }
    VkBuildAccelerationStructureFlagsKHR fl = FB | AU;
    VkAccelerationStructureBuildSizesInfoKHR sz = sizes_of(type, fl, &g, prims);
    // layout: AS at +4 MB of a 16 MB buffer, scratch at +4 MB of another; grown when the driver asks for more than 4 MB
    VkDeviceSize drv = sz.accelerationStructureSize, csize = exact ? drv : (drv > 8 * MB ? drv : 8 * MB);
    VkDeviceSize smax = sz.buildScratchSize > sz.updateScratchSize ? sz.buildScratchSize : sz.updateScratchSize;
    const VkDeviceSize ASOFF = drv > 4 * MB ? up(drv, MB) : 4 * MB, SCOFF = smax > 4 * MB ? up(smax, MB) : 4 * MB;
    VkDeviceSize need_s = 2 * ASOFF + (csize > 2 * drv ? csize : 2 * drv), need_c = 2 * SCOFF + 2 * smax;
    if (need_s < 16 * MB) need_s = 16 * MB;
    if (need_c < 16 * MB) need_c = 16 * MB;
    if (need_s > (VkDeviceSize)NPG * 4096 || need_c > (VkDeviceSize)NPG * 4096)
        die(2, "%s: driver sizes too large for the probe layout (as %llu scratch %llu)", lab, (unsigned long long)drv, (unsigned long long)smax);
    if (S.size < need_s) S = mkbuf(need_s);
    if (SC.size < need_c) SC = mkbuf(need_c);
    VkDeviceAddress scr = up(SC.a + SCOFF, SA);
    int64_t scroff = (int64_t)(scr - SC.a);
    VkAccelerationStructureKHR as = mkas(&S, ASOFF, csize, type);
    // helper TLAS over the BLAS case (separate buffers, never scanned)
    #define EXT_HELPER() do { if (!tlas_case) { put_inst(&HI, 0, as_addr(as), 0, 0, 0, 5); hflush(&HI); \
        VkAccelerationStructureGeometryKHR hg_ = inst_geom(HI.a); begin(cbs[0]); \
        rec_build(cbs[0], (struct Job){TLASF, FT, false, ht, hg_, 1, up(HTS.a, SA)}); end(cbs[0]); submit_wait(cbs[0], "helper TLAS"); } } while (0)
    static VkAccelerationStructureKHR ht;
    if (!tlas_case && !ht) ht = mkas(&HT, 0, HT.size, TLASF);
    struct Ext es, ec;
    ext_reset(&es);
    ext_reset(&ec);
    double ms = 0;
    for (int pass = 0; pass < 2; pass++) {   // two fill patterns, union of the extents
        uint8_t pat = pass ? 0xA5 : 0x5A;
        fill(&S, pat);
        fill(&SC, pat);
        begin(cbs[0]);
        rec_build(cbs[0], (struct Job){type, fl, false, as, g, prims, scr});
        end(cbs[0]);
        ms += submit_wait(cbs[0], "build");
        scan(&S, pat, &es);
        scan(&SC, pat, &ec);
    }
    char a[160], b[160], f[96], sg1[160];
    int fa = ext_fmt(a, sizeof a, "as", &es, (int64_t)ASOFF, drv, csize), fs = ext_fmt(b, sizeof b, "scr", &ec, scroff, sz.buildScratchSize, 0);
    flagstr(f, sizeof f, fa, fs, !exact);
    segs(&es, (int64_t)ASOFF, sg1, sizeof sg1);
    say("%s build %s %s create_size=%llu ms=%.2f flags=%s as_segs=%s", lab, a, b, (unsigned long long)csize, ms / 2, f, sg1);
    int fail = exact ? (fa | fs) : 0;   // rest: reported only (USES_TAIL), rays still checked
    EXT_HELPER();
    EXT_RAYS(0);
    int br = trace(tlas_case ? as : ht, ex, nr, "after build");
    // in-place refits: scratch refilled, storage keeps the 0xA5 pattern of the second build
    struct Ext us, uc;
    ext_reset(&us);
    ext_reset(&uc);
    ms = 0;
    for (int pass = 0; pass < 2; pass++) {
        uint8_t pat = pass ? 0xA5 : 0x5A;
        fill(&SC, pat);
        EXT_STATE(pass + 1);
        begin(cbs[0]);
        rec_build(cbs[0], (struct Job){type, fl, true, as, g, prims, scr});
        end(cbs[0]);
        ms += submit_wait(cbs[0], "update");
        scan(&SC, pat, &uc);
    }
    scan(&S, 0xA5, &us);
    fa = ext_fmt(a, sizeof a, "as", &us, (int64_t)ASOFF, drv, csize);
    fs = ext_fmt(b, sizeof b, "scr", &uc, scroff, sz.updateScratchSize, 0);
    flagstr(f, sizeof f, fa, fs, !exact);
    say("%s update %s %s ms=%.2f flags=%s", lab, a, b, ms / 2, f);
    fail |= exact ? (fa | fs) : 0;
    EXT_HELPER();
    EXT_RAYS(2);
    br += trace(tlas_case ? as : ht, ex, nr, "after update");
    say("%s rays bad=%d of %u (after build + after update)", lab, br, 2 * nr);
    rmas(as);
    *bad_rays += br;
    return fail;
}
static int arm_extent(char *why, size_t cap) {
    int fails = 0, bad = 0;
    double t0 = now_ms();
    fails |= extent_case(false, true, &bad);
    fails |= extent_case(false, false, &bad);
    fails |= extent_case(true, true, &bad);
    fails |= extent_case(true, false, &bad);
    say("time extent ms=%.1f", now_ms() - t0);
    if (fails || bad) { snprintf(why, cap, "out-of-range writes in the exact variant (flag bits 0x%x, see the flags= lines), wrong rays %d", fails, bad); return 1; }
    snprintf(why, cap, "exact: storage + scratch writes stay inside the driver sizes; all rays correct");
    return 0;
}

// ================================================================ arm: overlap
static int arm_overlap(bool batch, bool blas_type, char *why, size_t cap) {
    phase("overlap-setup");
    static const VkDeviceSize OFF[3] = {0, 2359296, 5701632};
    struct Buf S = mkbuf_g(8 * MB, MB);   // 1 MB GPU-mapped guard past the VkBuffer end: overruns land there, not in a fault
    struct Mesh m[3] = {mk_mesh(32, 32, -3.0f, -0.9f, -1.2f, 0.9f), mk_mesh(40, 20, -0.9f, -0.9f, 0.9f, 0.9f),
                        mk_mesh(16, 48, 1.2f, -0.9f, 3.0f, 0.9f)};
    VkAccelerationStructureGeometryKHR g[3];
    VkAccelerationStructureBuildSizesInfoKHR sz[3];
    VkAccelerationStructureKHR as[3];
    VkDeviceSize soff[3], stot = 0;
    for (int k = 0; k < 3; k++) {
        mesh_z(&m[k], 3.0f + k, 0.1f, (float)k);
        g[k] = mesh_geom(&m[k]);
        sz[k] = sizes_of(BLASF, FT, &g[k], m[k].nt);
        soff[k] = stot;
        stot += up(sz[k].buildScratchSize, KB64);
        if (OFF[k] + sz[k].accelerationStructureSize > (k < 2 ? OFF[k + 1] : 8 * MB))
            die(2, "blas%d driver size %llu does not fit before the next offset: layout invalid", k, (unsigned long long)sz[k].accelerationStructureSize);
        as[k] = mkas(&S, OFF[k], 8 * MB - OFF[k], blas_type ? BLASF : GENF);
        say("blas%d off=%llu create_size=%llu drv_size=%llu build_scratch=%llu tris=%u", k, (unsigned long long)OFF[k],
            (unsigned long long)(8 * MB - OFF[k]), (unsigned long long)sz[k].accelerationStructureSize,
            (unsigned long long)sz[k].buildScratchSize, m[k].nt);
    }
    struct Buf SC = mkbuf(stot + KB64);
    VkDeviceAddress sbase = up(SC.a, KB64 > SA ? KB64 : SA);
    struct Buf IB = mkbuf(3 * 64);
    for (int k = 0; k < 3; k++) put_inst(&IB, k, as_addr(as[k]), 0, 0, 0, 100 + k);
    hflush(&IB);
    VkAccelerationStructureGeometryKHR tg = inst_geom(IB.a);
    VkAccelerationStructureBuildSizesInfoKHR ts = sizes_of(TLASF, FT, &tg, 3);
    struct Buf TB = mkbuf(up(ts.accelerationStructureSize, 256)), TS = mkbuf(ts.buildScratchSize + SA);
    VkAccelerationStructureKHR tlas = mkas(&TB, 0, ts.accelerationStructureSize, TLASF);
    struct Exp ex[64];
    uint32_t nr = 0;
    for (int k = 0; k < 3; k++) for (uint32_t s = 0; s < 12; s++) ex[nr++] = mesh_pt(&m[k], s, 0, 0, 0, 100 + k);
    static const float mx[6] = {-1.05f, 1.05f, -3.5f, 3.5f, 0.0f, -2.0f}, my[6] = {0, 0, 0, 0, 1.5f, -1.5f};
    for (int s = 0; s < 6; s++) ex[nr++] = miss_at(mx[s], my[s]);
    bind_tlas(tlas);
    put_rays(ex, nr);
    int bad_iters = 0, bad_total = 0, past_iters = 0;
    uint64_t past_bytes = 0;
    int64_t past_first = -1;
    double ms = 0, ms_max = 0;
    for (int it = 0; it < 50; it++) {
        phase("overlap-iter");
        uint8_t pat = (it & 1) ? 0xA5 : 0x5A;
        fill(&S, pat);   // stale-free: every iteration must rebuild everything
        memset(OB.p, 0, 16 + 16 * (size_t)nr);
        hflush(&OB);
        begin(cbs[0]);
        struct Job j[3];
        for (int k = 0; k < 3; k++) { struct Job jj = {BLASF, FT, false, as[k], g[k], m[k].nt, sbase + soff[k]}; j[k] = jj; }
        if (batch) rec_builds(cbs[0], j, 3);
        else for (int k = 0; k < 3; k++) { rec_build(cbs[0], j[k]); fullbar(cbs[0]); }
        fullbar(cbs[0]);
        rec_build(cbs[0], (struct Job){TLASF, FT, false, tlas, tg, 3, up(TS.a, SA)});
        fullbar(cbs[0]);
        rec_trace(cbs[0], nr);
        end(cbs[0]);
        double t = submit_wait(cbs[0], "iteration");
        ms += t;
        if (t > ms_max) ms_max = t;
        char what[32];
        snprintf(what, sizeof what, "iter %d", it);
        int b = check(ex, nr, what);
        bad_total += b;
        bad_iters += b != 0;
        int64_t gfirst = -1;
        uint64_t gb = guard(&S, 8 * MB, S.msize, pat, &gfirst);
        if (gb) { past_iters++; past_bytes += gb; if (past_first < 0 || gfirst < past_first) past_first = gfirst; }
        if (it == 0) {
            struct Ext e;
            ext_reset(&e);
            scan(&S, pat, &e);
            char sg[200];
            segs(&e, 0, sg, sizeof sg);
            say("iter0 storage written=[%lld,%lld) segs=%s last_64k_written=%s", (long long)e.lo, (long long)e.hi, sg,
                e.hi > (int64_t)(8 * MB - KB64) ? "YES" : "no");
        }
    }
    say("time overlap iters=50 ms_avg=%.2f ms_max=%.2f bad_iters=%d bad_rays=%d of %u", ms / 50, ms_max, bad_iters, bad_total, 50 * nr);
    say("past_buffer_end iters=%d bytes=%llu first_off_past_end=%lld%s", past_iters, (unsigned long long)past_bytes,
        past_first < 0 ? -1ll : (long long)(past_first - (int64_t)(8 * MB)), past_iters ? " PAST_BUFFER_WRITE" : "");
    if (bad_total || past_iters) {
        snprintf(why, cap, "%d wrong rays in %d of 50 iterations, writes past the 8 MB buffer end in %d iterations", bad_total, bad_iters, past_iters);
        return 1;
    }
    snprintf(why, cap, "50 iterations x %u rays correct (3 %s handles, rest-of-buffer sizes, %s builds)", nr,
             blas_type ? "BOTTOM_LEVEL" : "GENERIC", batch ? "one-call" : "serial");
    return 0;
}

// ================================================================ arm: update
static int arm_update(int sub, char *why, size_t cap) {   // sub 0 single, 1 batch (6 BLAS in one call), 2 tlasupd
    phase("update-setup");
    const int NB = sub == 1 ? 6 : 1;
    struct Mesh m[6];
    VkAccelerationStructureGeometryKHR g[6];
    VkAccelerationStructureBuildSizesInfoKHR sz[6];
    VkDeviceSize aoff[6], atot = 0, bsoff[6], bstot = 0, usoff[6], ustot = 0;
    for (int k = 0; k < NB; k++) {
        m[k] = NB == 1 ? mk_mesh(32, 32, -1.5f, -1.0f, 1.5f, 1.0f) : mk_mesh(32, 30 + k, -0.45f, -0.9f, 0.45f, 0.9f);
        mesh_z(&m[k], 3.0f, 0.15f, (float)k);
        g[k] = mesh_geom(&m[k]);
        sz[k] = sizes_of(BLASF, FB | AU, &g[k], m[k].nt);
        aoff[k] = atot;
        atot += up(sz[k].accelerationStructureSize, 256);
        bsoff[k] = bstot;
        bstot += up(sz[k].buildScratchSize, KB64);
        usoff[k] = ustot;
        ustot += up(sz[k].updateScratchSize, KB64);
    }
    say("blas tris=%u as=%llu build_scratch=%llu update_scratch=%llu count=%d", m[0].nt, (unsigned long long)sz[0].accelerationStructureSize,
        (unsigned long long)sz[0].buildScratchSize, (unsigned long long)sz[0].updateScratchSize, NB);
    struct Buf S = mkbuf(up(atot, 256) + KB64);
    VkDeviceSize scsz = (bstot > ustot ? bstot : ustot) + KB64;
    struct Buf SC = mkbuf(scsz + KB64);
    VkDeviceSize sb = up(SC.a, KB64 > SA ? KB64 : SA) - SC.a;   // scratch base offset inside SC
    fill(&S, 0x5A);
    VkAccelerationStructureKHR as[6];
    for (int k = 0; k < NB; k++) as[k] = mkas(&S, aoff[k], sz[k].accelerationStructureSize, BLASF);
    struct Buf IB = mkbuf(6 * 64);
    for (int k = 0; k < NB; k++) put_inst(&IB, k, as_addr(as[k]), NB == 1 ? 0 : -2.5f + k, 0, 0, 200 + k);
    hflush(&IB);
    VkAccelerationStructureGeometryKHR tg = inst_geom(IB.a);
    VkBuildAccelerationStructureFlagsKHR tf = sub == 2 ? FB | AU : FT;
    VkAccelerationStructureBuildSizesInfoKHR ts = sizes_of(TLASF, tf, &tg, NB);
    VkDeviceSize tss = ts.buildScratchSize > ts.updateScratchSize ? ts.buildScratchSize : ts.updateScratchSize;
    struct Buf TB = mkbuf(up(ts.accelerationStructureSize, 256)), TS = mkbuf(tss + SA);
    VkAccelerationStructureKHR tlas = mkas(&TB, 0, ts.accelerationStructureSize, TLASF);
    if (sub == 2) say("tlas as=%llu build_scratch=%llu update_scratch=%llu", (unsigned long long)ts.accelerationStructureSize,
                      (unsigned long long)ts.buildScratchSize, (unsigned long long)ts.updateScratchSize);
    // initial BUILD (one call)
    struct Job j[6];
    for (int k = 0; k < NB; k++) { struct Job jj = {BLASF, FB | AU, false, as[k], g[k], m[k].nt, SC.a + sb + bsoff[k]}; j[k] = jj; }
    fill(&SC, 0x5A);
    begin(cbs[0]);
    rec_builds(cbs[0], j, NB);
    fullbar(cbs[0]);
    rec_build(cbs[0], (struct Job){TLASF, tf, false, tlas, tg, NB, up(TS.a, SA)});
    end(cbs[0]);
    double tb = submit_wait(cbs[0], "initial build");
    int64_t first_as = -1, first_sc = -1;
    uint64_t g_as = 0;
    for (int k = 0; k < NB; k++) g_as += guard(&S, aoff[k] + sz[k].accelerationStructureSize, k + 1 < NB ? aoff[k + 1] : S.size, 0x5A, &first_as);
    say("initial build ms=%.2f storage_guard_bytes_hit=%llu", tb, (unsigned long long)g_as);
    fill(&SC, 0x5A);   // update scratch guards from here on
    struct Exp ex[64];
    uint32_t nr = 0;
    bind_tlas(tlas);
    int bad_iters = 0, bad_total = 0;
    uint64_t g_sc = 0;
    double ms = 0, ms_max = 0;
    for (int it = 1; it <= 200; it++) {
        phase("update-iter");
        nr = 0;
        for (int k = 0; k < NB; k++) {
            mesh_z(&m[k], 3.0f + 0.3f * sinf(0.11f * it + k), 0.15f, 0.37f * it + k);
            for (uint32_t s = 0; s < (NB == 1 ? 24u : 6u); s++) ex[nr++] = mesh_pt(&m[k], s, NB == 1 ? 0 : -2.5f + k, 0, 0, 200 + k);
        }
        for (int s = 0; s < 4; s++) ex[nr++] = miss_at(NB == 1 ? 1.8f : 3.6f, -0.9f + 0.6f * s);
        put_rays(ex, nr);
        begin(cbs[0]);
        for (int k = 0; k < NB; k++) { j[k].update = true; j[k].scratch = SC.a + sb + usoff[k]; }
        rec_builds(cbs[0], j, NB);
        fullbar(cbs[0]);
        rec_build(cbs[0], (struct Job){TLASF, tf, sub == 2, tlas, tg, NB, up(TS.a, SA)});
        fullbar(cbs[0]);
        rec_trace(cbs[0], nr);
        end(cbs[0]);
        double t = submit_wait(cbs[0], "iteration");
        ms += t;
        if (t > ms_max) ms_max = t;
        char what[32];
        snprintf(what, sizeof what, "iter %d", it);
        int b = check(ex, nr, what);
        bad_total += b;
        bad_iters += b != 0;
        for (int k = 0; k < NB; k++)   // bytes between the end of each update scratch slice and the next slice
            g_sc += guard(&SC, sb + usoff[k] + sz[k].updateScratchSize, sb + usoff[k] + up(sz[k].updateScratchSize, KB64), 0x5A, &first_sc);
        g_sc += guard(&SC, sb + ustot, SC.size, 0x5A, &first_sc);
        for (int k = 0; k < NB; k++)   // storage bytes between the driver size of each BLAS and the next one
            g_as += guard(&S, aoff[k] + sz[k].accelerationStructureSize, k + 1 < NB ? aoff[k + 1] : S.size, 0x5A, &first_as);
    }
    say("time update iters=200 ms_avg=%.2f ms_max=%.2f bad_iters=%d bad_rays=%d", ms / 200, ms_max, bad_iters, bad_total);
    say("guards storage_bytes_hit=%llu first_storage_off=%lld scratch_bytes_hit=%llu first_scratch_off=%lld (rel scratch base)",
        (unsigned long long)g_as, (long long)first_as, (unsigned long long)g_sc,
        first_sc < 0 ? -1ll : (long long)(first_sc - (int64_t)sb));
    if (bad_total || g_as || g_sc) {
        snprintf(why, cap, "wrong rays %d (%d iters), storage guard bytes %llu, scratch guard bytes %llu", bad_total, bad_iters,
                 (unsigned long long)g_as, (unsigned long long)g_sc);
        return 1;
    }
    snprintf(why, cap, "200 in-place refits of %d BLAS%s correct, nothing written past the driver sizes", NB, sub == 2 ? " + TLAS refit" : "");
    return 0;
}

// ================================================================ arm: mutate
static int arm_mutate(bool recreate, char *why, size_t cap) {
    phase("mutate-setup");
    struct Buf S = mkbuf_g(8 * MB, MB);
    fill(&S, 0x5A);   // never refilled: the guard tail past the 8 MB buffer end must keep 0x5A
    struct Mesh M1 = mk_mesh(16, 16, -1, -1, 1, 1), B2m = mk_mesh(8, 8, -1, -1, 1, 1);
    mesh_z(&B2m, 0, 0, 0);
    VkAccelerationStructureGeometryKHR g1 = mesh_geom(&M1), g2 = mesh_geom(&B2m);
    VkAccelerationStructureBuildSizesInfoKHR s1 = sizes_of(BLASF, FT, &g1, M1.nt), s2 = sizes_of(BLASF, FT, &g2, B2m.nt);
    struct Buf B2 = mkbuf(up(s2.accelerationStructureSize, 256));
    VkAccelerationStructureKHR b2 = mkas(&B2, 0, s2.accelerationStructureSize, BLASF);
    struct Buf IB2 = mkbuf(64), HI = mkbuf(64);
    VkAccelerationStructureGeometryKHR tgB2 = inst_geom(IB2.a), tgH = inst_geom(HI.a);
    VkAccelerationStructureBuildSizesInfoKHR st = sizes_of(TLASF, FT, &tgB2, 1);
    VkDeviceSize scr = s1.buildScratchSize;
    if (s2.buildScratchSize > scr) scr = s2.buildScratchSize;
    if (st.buildScratchSize > scr) scr = st.buildScratchSize;
    struct Buf SC = mkbuf(scr + SA);
    VkDeviceAddress sa = up(SC.a, SA);
    struct Buf T2 = mkbuf(up(st.accelerationStructureSize, 256));
    VkAccelerationStructureKHR t2 = mkas(&T2, 0, st.accelerationStructureSize, TLASF);
    say("blas_size=%llu tlas_size=%llu handle_size=%llu mode=%s", (unsigned long long)s1.accelerationStructureSize,
        (unsigned long long)st.accelerationStructureSize, (unsigned long long)(8 * MB), recreate ? "recreate" : "generic");
    begin(cbs[0]);
    rec_build(cbs[0], (struct Job){BLASF, FT, false, b2, g2, B2m.nt, sa});
    end(cbs[0]);
    submit_wait(cbs[0], "B2 build");
    VkAccelerationStructureKHR h = mkas(&S, 0, 8 * MB, recreate ? BLASF : GENF);
    VkDeviceAddress a0 = as_addr(h);
    int addr_changes = 0, bad_total = 0, bad_steps = 0, past_steps = 0;
    double ms[3] = {0, 0, 0};
    static const char *const stepn[3] = {"blas", "tlas", "blas2"};
    for (int r = 0; r < 20; r++) {
        for (int st3 = 0; st3 < 3; st3++) {
            phase(stepn[st3]);
            bool as_tlas = st3 == 1;
            if (recreate) {
                rmas(h);
                h = mkas(&S, 0, 8 * MB, as_tlas ? TLASF : BLASF);
                if (as_addr(h) != a0) addr_changes++;
            }
            struct Exp ex[24];
            uint32_t nr = 0;
            begin(cbs[0]);
            if (!as_tlas) {
                mesh_z(&M1, 3.0f + 0.01f * r + 0.1f * st3, 0.1f, (float)r);
                rec_build(cbs[0], (struct Job){BLASF, FT, false, h, g1, M1.nt, sa});
                fullbar(cbs[0]);
                put_inst(&HI, 0, as_addr(h), 0, 0, 0, 1);
                hflush(&HI);
                rec_build(cbs[0], (struct Job){TLASF, FT, false, t2, tgH, 1, sa + 0});   // scratch reuse after the barrier
                for (uint32_t s = 0; s < 16; s++) ex[nr++] = mesh_pt(&M1, s, 0, 0, 0, 1);
            } else {
                put_inst(&IB2, 0, as_addr(b2), 0, 0, 5.0f + 0.01f * r, 2);
                hflush(&IB2);
                rec_build(cbs[0], (struct Job){TLASF, FT, false, h, tgB2, 1, sa});
                for (uint32_t s = 0; s < 16; s++) ex[nr++] = mesh_pt(&B2m, s, 0, 0, 5.0f + 0.01f * r, 2);
            }
            for (int s = 0; s < 4; s++) ex[nr++] = miss_at(1.5f, -0.9f + 0.6f * s);
            end(cbs[0]);
            ms[st3] += submit_wait(cbs[0], "build");
            char what[32];
            snprintf(what, sizeof what, "round %d %s", r, stepn[st3]);
            int b = trace(as_tlas ? h : t2, ex, nr, what);
            bad_total += b;
            bad_steps += b != 0;
            int64_t gfirst = -1;
            uint64_t gb = guard(&S, 8 * MB, S.msize, 0x5A, &gfirst);
            if (gb && !past_steps++) say("PAST_BUFFER_WRITE first at %s round %d: %llu bytes, first %lld past the end", stepn[st3], r,
                                         (unsigned long long)gb, (long long)(gfirst - (int64_t)(8 * MB)));
            if (gb) fill(&S, 0x5A);   // fresh guard (the handle content is rebuilt by the next step anyway)
        }
    }
    say("time mutate rounds=20 ms_avg_blas=%.2f ms_avg_tlas=%.2f ms_avg_blas2=%.2f addr_changes=%d", ms[0] / 20, ms[1] / 20, ms[2] / 20, addr_changes);
    if (bad_total || past_steps) {
        snprintf(why, cap, "%d wrong rays in %d of 60 steps, writes past the 8 MB buffer end in %d steps", bad_total, bad_steps, past_steps);
        return 1;
    }
    snprintf(why, cap, "60 type changes (%s handle) traced correctly", recreate ? "recreated" : "one GENERIC");
    return 0;
}

// ================================================================ arm: race
static int arm_race(int sub, uint32_t grid, int iters, char *why, size_t cap) {   // sub 0 nobarrier 1 barrier 2 twocmd 3 twocmdnb
    phase("race-setup");
    struct Mesh M = mk_mesh(64, 64, -1.2f, -1.2f, 1.2f, 1.2f);
    mesh_z(&M, 2.0f, 0, 0);
    VkAccelerationStructureGeometryKHR g = mesh_geom(&M);
    VkAccelerationStructureBuildSizesInfoKHR s = sizes_of(BLASF, FB | AU, &g, M.nt);
    struct Buf S = mkbuf(up(s.accelerationStructureSize, 256));
    VkAccelerationStructureKHR blas = mkas(&S, 0, s.accelerationStructureSize, BLASF);
    struct Buf IB = mkbuf(64);
    put_inst(&IB, 0, as_addr(blas), 0, 0, 0, 7);
    hflush(&IB);
    VkAccelerationStructureGeometryKHR tg = inst_geom(IB.a);
    VkAccelerationStructureBuildSizesInfoKHR ts = sizes_of(TLASF, FT, &tg, 1);
    struct Buf TB = mkbuf(up(ts.accelerationStructureSize, 256));
    VkAccelerationStructureKHR tlas = mkas(&TB, 0, ts.accelerationStructureSize, TLASF);
    VkDeviceSize bs = s.buildScratchSize > s.updateScratchSize ? s.buildScratchSize : s.updateScratchSize;
    struct Buf SC = mkbuf(up(bs, KB64) + up(ts.buildScratchSize, KB64) + KB64);
    VkDeviceAddress sa = up(SC.a, KB64 > SA ? KB64 : SA), sat = sa + up(bs, KB64);
    begin(cbs[0]);
    rec_build(cbs[0], (struct Job){BLASF, FB | AU, false, blas, g, M.nt, sa});
    fullbar(cbs[0]);
    rec_build(cbs[0], (struct Job){TLASF, FT, false, tlas, tg, 1, sat});
    end(cbs[0]);
    submit_wait(cbs[0], "initial build");
    struct Exp ex[20];
    uint32_t nr = 0;
    for (uint32_t q = 0; q < 16; q++) ex[nr++] = mesh_pt(&M, q, 0, 0, 0, 7);
    int b0 = trace(tlas, ex, nr, "initial");
    if (b0) { snprintf(why, cap, "initial TLAS wrong (%d rays) before any race", b0); return 1; }
    // calibrate: one heavy dispatch alone
    const uint32_t reps = 4;
    const uint64_t total = (uint64_t)grid * grid * reps;
    struct PC pc = {1, grid, reps, 0, {-1, -1, 1, 1}, {2.0f, 2.0f}, 0.01f, 0};
    memset(OB.p, 0, 16);
    hflush(&OB);
    begin(cbs[0]);
    vkCmdBindPipeline(cbs[0], VK_PIPELINE_BIND_POINT_COMPUTE, rqpipe);
    vkCmdBindDescriptorSets(cbs[0], VK_PIPELINE_BIND_POINT_COMPUTE, pl, 0, 1, &ds, 0, NULL);
    vkCmdPushConstants(cbs[0], pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof pc, &pc);
    vkCmdDispatch(cbs[0], (uint32_t)(((uint64_t)grid * grid + 63) / 64), 1, 1);
    end(cbs[0]);
    double tcal = submit_wait(cbs[0], "calibration dispatch");
    hinval(&OB);
    const uint32_t *cnt = (const uint32_t *)OB.p;
    say("calibration grid=%u reps=%u traces=%llu ms=%.2f mrays=%.1f old=%u new=%u miss=%u other=%u", grid, reps, (unsigned long long)total,
        tcal, total / (tcal * 1e3), cnt[0], cnt[1], cnt[2], cnt[3]);
    if (cnt[0] != total) { snprintf(why, cap, "calibration dispatch wrong (old %u of %llu)", cnt[0], (unsigned long long)total); return 1; }
    int bad_iters = 0, saw_new_iters = 0, printed = 0;
    uint64_t new_total = 0, miss_total = 0, other_total = 0;
    double ms = 0, ms_max = 0;
    float zprev = 2.0f;
    for (int it = 1; it <= iters; it++) {
        phase("race-iter");
        float z = 2.0f + 0.5f * (it % 5);
        mesh_z(&M, z, 0, 0);   // host write before submit; only the BLAS update reads it
        memset(OB.p, 0, 16);
        hflush(&OB);
        struct PC p = {1, grid, reps, 0, {-1, -1, 1, 1}, {zprev, z}, 0.01f, 0};
        VkCommandBuffer cb = cbs[0];
        begin(cb);
        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, rqpipe);
        vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pl, 0, 1, &ds, 0, NULL);
        vkCmdPushConstants(cb, pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof p, &p);
        vkCmdDispatch(cb, (uint32_t)(((uint64_t)grid * grid + 63) / 64), 1, 1);
        if (sub >= 2) { end(cb); cb = cbs[1]; begin(cb); }
        if (sub == 1 || sub == 2) fullbar(cb);
        rec_build(cb, (struct Job){BLASF, FB | AU, true, blas, g, M.nt, sa});
        fullbar(cb);   // BLAS -> TLAS is always ordered (that is not what this arm tests)
        rec_build(cb, (struct Job){TLASF, FT, false, tlas, tg, 1, sat});
        fullbar(cb);
        end(cb);
        double t0 = now_ms();
        CK(vkResetFences(dev, 1, &fence));
        if (sub >= 2) {
            VkSubmitInfo s1 = {VK_STRUCTURE_TYPE_SUBMIT_INFO, NULL, 0, NULL, NULL, 1, &cbs[0], 0, NULL};
            VkSubmitInfo s2 = {VK_STRUCTURE_TYPE_SUBMIT_INFO, NULL, 0, NULL, NULL, 1, &cbs[1], 0, NULL};
            CK(vkQueueSubmit(queue, 1, &s1, VK_NULL_HANDLE));
            CK(vkQueueSubmit(queue, 1, &s2, fence));
        } else {
            VkSubmitInfo s1 = {VK_STRUCTURE_TYPE_SUBMIT_INFO, NULL, 0, NULL, NULL, 1, &cbs[0], 0, NULL};
            CK(vkQueueSubmit(queue, 1, &s1, fence));
        }
        wait_fence("race iteration");
        double t = now_ms() - t0;
        ms += t;
        if (t > ms_max) ms_max = t;
        hinval(&OB);
        new_total += cnt[1];
        miss_total += cnt[2];
        other_total += cnt[3];
        saw_new_iters += cnt[1] != 0;
        if (cnt[0] != total) {
            bad_iters++;
            if (printed++ < 4) say("RACE iter %d old=%u new=%u miss=%u other=%u of %llu (z %.2f -> %.2f)", it, cnt[0], cnt[1], cnt[2], cnt[3],
                                   (unsigned long long)total, zprev, z);
        }
        zprev = z;
    }
    say("time race iters=%d ms_avg=%.2f ms_max=%.2f bad_iters=%d saw_new_iters=%d new=%llu miss=%llu other=%llu", iters, ms / iters, ms_max,
        bad_iters, saw_new_iters, (unsigned long long)new_total, (unsigned long long)miss_total, (unsigned long long)other_total);
    // final sanity: the last geometry is traced correctly with a barrier
    nr = 0;
    for (uint32_t q = 0; q < 16; q++) ex[nr++] = mesh_pt(&M, q, 0, 0, 0, 7);
    int bf = trace(tlas, ex, nr, "final");
    if (bad_iters || bf) {
        snprintf(why, cap, "%d of %d iterations traced a TLAS being rebuilt (new %llu miss %llu other %llu)%s%s", bad_iters, iters,
                 (unsigned long long)new_total, (unsigned long long)miss_total, (unsigned long long)other_total, bf ? ", final trace wrong" : "",
                 sub == 0 || sub == 3 ? " [no barrier: spec allows overlap]" : " [barrier present: driver ordering bug]");
        return 1;
    }
    snprintf(why, cap, "%d iterations: every ray query saw the pre-build geometry", iters);
    return 0;
}

// ================================================================ helpers for the v2 arms (copy, updsrc, formats, vfmt, multigeom)
static PFN_vkCmdCopyAccelerationStructureKHR pCopy;                       // optional entry points: arms SKIP without them
static PFN_vkCmdWriteAccelerationStructuresPropertiesKHR pWriteProps;

// One build call over ng geometries (the v1 arms keep rec_builds: one geometry per AS).
static void rec_build_g(VkCommandBuffer cb, VkAccelerationStructureTypeKHR type, VkBuildAccelerationStructureFlagsKHR fl, bool update,
                        VkAccelerationStructureKHR src, VkAccelerationStructureKHR dst, const VkAccelerationStructureGeometryKHR *g,
                        const VkAccelerationStructureBuildRangeInfoKHR *rg, uint32_t ng, VkDeviceAddress scratch) {
    VkAccelerationStructureBuildGeometryInfoKHR b = {VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR};
    b.type = type;
    b.flags = fl;
    b.mode = update ? VK_BUILD_ACCELERATION_STRUCTURE_MODE_UPDATE_KHR : VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
    b.srcAccelerationStructure = update ? src : VK_NULL_HANDLE;
    b.dstAccelerationStructure = dst;
    b.geometryCount = ng;
    b.pGeometries = g;
    b.scratchData.deviceAddress = scratch;
    const VkAccelerationStructureBuildRangeInfoKHR *pr = rg;
    pBuild(cb, 1, &b, &pr);
}
static VkAccelerationStructureBuildSizesInfoKHR sizes_of_g(VkAccelerationStructureTypeKHR t, VkBuildAccelerationStructureFlagsKHR fl,
                                                           const VkAccelerationStructureGeometryKHR *g, const uint32_t *prims, uint32_t ng) {
    VkAccelerationStructureBuildGeometryInfoKHR bi = {VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR};
    bi.type = t;
    bi.flags = fl;
    bi.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
    bi.geometryCount = ng;
    bi.pGeometries = g;
    VkAccelerationStructureBuildSizesInfoKHR sz = {VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR};
    pSizes(dev, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR, &bi, prims, &sz);
    return sz;
}

// Helper TLAS with one instance (own storage + scratch), used to trace any BLAS by its device address.
struct HT { struct Buf ib, tb, ts; VkAccelerationStructureKHR t; };
static struct HT mk_ht(void) {
    struct HT h;
    h.ib = mkbuf(256);
    VkAccelerationStructureGeometryKHR g = inst_geom(h.ib.a);
    VkAccelerationStructureBuildSizesInfoKHR s = sizes_of(TLASF, FT, &g, 1);
    h.tb = mkbuf(up(s.accelerationStructureSize, 256));
    h.ts = mkbuf(s.buildScratchSize + SA);
    h.t = mkas(&h.tb, 0, s.accelerationStructureSize, TLASF);
    return h;
}
// Build the helper TLAS over blas (instance translation (0,0,tz), custom index) and trace e[] through it.
static int trace_blas(struct HT *h, VkDeviceAddress blas, float tz, uint32_t custom, const struct Exp *e, uint32_t n, const char *what) {
    put_inst(&h->ib, 0, blas, 0, 0, tz, custom);
    hflush(&h->ib);
    bind_tlas(h->t);
    put_rays(e, n);
    VkAccelerationStructureGeometryKHR g = inst_geom(h->ib.a);
    begin(cbs[0]);
    fullbar(cbs[0]);
    rec_build(cbs[0], (struct Job){TLASF, FT, false, h->t, g, 1, up(h->ts.a, SA)});
    fullbar(cbs[0]);
    rec_trace(cbs[0], n);
    end(cbs[0]);
    submit_wait(cbs[0], what);
    return check(e, n, what);
}
static uint32_t mesh_rays(const struct Mesh *m, struct Exp *ex, uint32_t nhit, int32_t custom, float tz) {
    uint32_t n = 0;
    for (uint32_t s = 0; s < nhit; s++) ex[n++] = mesh_pt(m, s, 0, 0, tz, custom);
    for (uint32_t s = 0; s < 8; s++) ex[n++] = miss_at(m->x1 + 0.5f, m->y0 + (m->y1 - m->y0) * (s + 0.5f) / 8);
    return n;
}

// ================================================================ arm: copy
// vkCmdCopyAccelerationStructureKHR CLONE / COMPACT from a rest-of-buffer GENERIC source view into a destination view that has
// exactly the reported size left before the end of its VkBuffer (what a D3D12 game allocates). Guard tail >= the source's
// created size, so "copies the created size" is measured (PAST_BUFFER_WRITE) instead of faulting.
static int arm_copy(bool compact, bool big, char *why, size_t cap) {
    phase("copy-setup");
    if (!pCopy || !pWriteProps) die(2, "vkCmdCopyAccelerationStructureKHR / vkCmdWriteAccelerationStructuresPropertiesKHR not exposed");
    const VkDeviceSize SOFF = big ? 0 : 5701632, SCREATE = 8 * MB - SOFF;
    struct Buf S = mkbuf_g(8 * MB, MB);
    struct Buf D = mkbuf_g(8 * MB, up(SCREATE, MB) + MB);
    uint8_t *snap = malloc(S.msize);
    if (!snap) die(3, "malloc");
    struct HT h = mk_ht();
    VkQueryPoolCreateInfo qpi = {VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO, NULL, 0, VK_QUERY_TYPE_ACCELERATION_STRUCTURE_COMPACTED_SIZE_KHR, 1, 0};
    VkQueryPool qp;
    CK(vkCreateQueryPool(dev, &qpi, NULL, &qp));
    static const uint32_t NX[2] = {100, 10}, NY[2] = {50, 5};
    int fail_flags = 0, bad_rays = 0;
    double ms_copy = 0;
    char res[2][200];
    for (int pass = 0; pass < 2; pass++) {
        phase(pass ? "copy-pass-100" : "copy-pass-10000");
        struct Mesh M = mk_mesh(NX[pass], NY[pass], -2, -1, 2, 1);
        mesh_z(&M, 3.0f, 0.2f, 0.3f + pass);
        VkAccelerationStructureGeometryKHR g = mesh_geom(&M);
        VkAccelerationStructureBuildSizesInfoKHR sz = sizes_of(BLASF, FT | AC, &g, M.nt);
        const VkDeviceSize drv = sz.accelerationStructureSize;
        if (drv > SCREATE) die(2, "tris=%u: driver size %llu > source rest %llu: layout invalid", M.nt, (unsigned long long)drv, (unsigned long long)SCREATE);
        struct Buf SC = mkbuf(sz.buildScratchSize + SA);
        VkAccelerationStructureKHR src = mkas(&S, SOFF, SCREATE, GENF);
        // source BUILD twice (patterns 0x5A / 0xA5, union of the written bytes), compacted-size query after each
        struct Ext es;
        ext_reset(&es);
        uint64_t csz[2] = {0, 0};
        uint64_t src_guard = 0;
        int64_t gfirst = -1;
        for (int p = 0; p < 2; p++) {
            uint8_t pat = p ? 0xA5 : 0x5A;
            fill(&S, pat);
            begin(cbs[0]);
            vkCmdResetQueryPool(cbs[0], qp, 0, 1);
            rec_build(cbs[0], (struct Job){BLASF, FT | AC, false, src, g, M.nt, up(SC.a, SA)});
            fullbar(cbs[0]);
            pWriteProps(cbs[0], 1, &src, VK_QUERY_TYPE_ACCELERATION_STRUCTURE_COMPACTED_SIZE_KHR, qp, 0);
            end(cbs[0]);
            submit_wait(cbs[0], "source build + compacted-size query");
            CK(vkGetQueryPoolResults(dev, qp, 0, 1, sizeof csz[p], &csz[p], sizeof csz[p], VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT));
            scan(&S, pat, &es);
            src_guard += guard(&S, 8 * MB, S.msize, pat, &gfirst);
        }
        char sa_[160];
        int fsrc = ext_fmt(sa_, sizeof sa_, "src", &es, (int64_t)SOFF, drv, SCREATE);
        say("tris=%u src_off=%llu src_create=%llu accel_size=%llu compacted=%llu%s %s build_scratch=%llu src_flags=%s%s%s%s",
            M.nt, (unsigned long long)SOFF, (unsigned long long)SCREATE, (unsigned long long)drv, (unsigned long long)csz[1],
            csz[0] != csz[1] ? " COMPACTED_UNSTABLE" : "", sa_, (unsigned long long)sz.buildScratchSize,
            fsrc & F_BELOW ? "SRC_BELOW," : "", fsrc & F_BEYOND_SIZE ? "SRC_BEYOND_SIZE," : "",
            src_guard ? "SRC_PAST_BUFFER_WRITE," : "", !fsrc && !src_guard ? "-" : "");
        struct Exp ex[48];
        uint32_t nr = mesh_rays(&M, ex, 32, 9, 0);
        int br_src = trace_blas(&h, as_addr(src), 0, 9, ex, nr, "source");
        if (compact && (csz[1] == 0 || csz[1] > drv)) say("COMPACTED_SIZE_ODD compacted=%llu accel_size=%llu", (unsigned long long)csz[1], (unsigned long long)drv);
        const VkDeviceSize rep = compact ? csz[1] : drv;            // the size the game was told
        const VkDeviceSize need = compact ? up(rep ? rep : 1, KB64) : up(drv, 256);
        if (need > 8 * MB) die(2, "needed destination size %llu > 8 MB", (unsigned long long)need);
        const VkDeviceSize doff = 8 * MB - need;
        VkAccelerationStructureKHR dst = mkas(&D, doff, need, GENF);
        struct Ext ed;
        ext_reset(&ed);
        uint64_t past = 0, src_changed = 0;
        int64_t pfirst = -1;
        for (int p = 0; p < 2; p++) {   // copy twice: destination patterns 0x5A / 0xA5
            uint8_t pat = p ? 0xA5 : 0x5A;
            fill(&D, pat);
            hinval(&S);
            memcpy(snap, S.p, S.msize);
            begin(cbs[0]);
            fullbar(cbs[0]);
            VkCopyAccelerationStructureInfoKHR ci = {VK_STRUCTURE_TYPE_COPY_ACCELERATION_STRUCTURE_INFO_KHR, NULL, src, dst,
                compact ? VK_COPY_ACCELERATION_STRUCTURE_MODE_COMPACT_KHR : VK_COPY_ACCELERATION_STRUCTURE_MODE_CLONE_KHR};
            pCopy(cbs[0], &ci);
            end(cbs[0]);
            ms_copy += submit_wait(cbs[0], "copy");
            scan(&D, pat, &ed);
            past += guard(&D, 8 * MB, D.msize, pat, &pfirst);
            hinval(&S);
            for (VkDeviceSize o = 0; o < S.msize; o++) src_changed += snap[o] != S.p[o];
        }
        int64_t lo = ed.lo < 0 ? -1 : ed.lo - (int64_t)doff, hi = ed.lo < 0 ? -1 : ed.hi - (int64_t)doff;
        bool below = ed.lo >= 0 && lo < 0, beyond_rep = ed.lo >= 0 && hi > (int64_t)rep, beyond_st = ed.lo >= 0 && hi > (int64_t)need;
        char sg[200];
        segs(&ed, (int64_t)doff, sg, sizeof sg);
        char fl[160];
        snprintf(fl, sizeof fl, "%s%s%s%s%s", below ? "DST_BELOW," : "", beyond_rep ? (compact ? "DST_BEYOND_COMPACT," : "DST_BEYOND_SIZE,") : "",
                 beyond_st ? "DST_BEYOND_STORAGE," : "", past ? "PAST_BUFFER_WRITE," : "", src_changed ? "SRC_CHANGED," : "");
        if (fl[0]) fl[strlen(fl) - 1] = 0; else snprintf(fl, sizeof fl, "-");
        say("tris=%u copy=%s dst_off=%llu dst_create=%llu %s=%llu dst_written=[%lld,%lld) dst_segs=%s past_end_bytes=%llu first_past_end=%lld "
            "src_changed_bytes=%llu copy_ms=%.2f flags=%s", M.nt, compact ? "compact" : "clone", (unsigned long long)doff, (unsigned long long)need,
            compact ? "compacted" : "accel_size", (unsigned long long)rep, (long long)lo, (long long)hi, sg, (unsigned long long)past / 2,
            pfirst < 0 ? -1ll : (long long)(pfirst - (int64_t)(8 * MB)), (unsigned long long)src_changed / 2, ms_copy / (2 * (pass + 1)), fl);
        // the copy must be self-contained: wipe the source storage, then trace the copy
        fill(&S, 0x3C);
        int br_dst = trace_blas(&h, as_addr(dst), 0, 9, ex, nr, "copy");
        say("tris=%u rays source_bad=%d copy_bad=%d of %u (source wiped before tracing the copy)", M.nt, br_src, br_dst, nr);
        bad_rays += br_src + br_dst;
        int f = (below || beyond_rep || beyond_st || past || src_changed) | (fsrc || src_guard ? 2 : 0) | (csz[0] != csz[1] ? 4 : 0) |
                (compact && (csz[1] == 0 || csz[1] > drv) ? 8 : 0);
        fail_flags |= f;
        snprintf(res[pass], sizeof res[pass], "tris=%u %s=%llu written=[%lld,%lld) %s", M.nt, compact ? "compacted" : "size",
                 (unsigned long long)rep, (long long)lo, (long long)hi, fl);
        rmas(dst);
        rmas(src);
        free(M.z);
    }
    vkDestroyQueryPool(dev, qp, NULL);
    free(snap);
    if (fail_flags || bad_rays) {
        snprintf(why, cap, "%s | %s | wrong rays %d%s%s%s", res[0], res[1], bad_rays, fail_flags & 2 ? ", source build out of range" : "",
                 fail_flags & 4 ? ", compacted size unstable" : "", fail_flags & 8 ? ", compacted size 0 or > accel size" : "");
        return 1;
    }
    snprintf(why, cap, "%s | %s | copies stay inside [dst, dst+reported), rays correct", res[0], res[1]);
    return 0;
}

// ================================================================ arm: updsrc
// UPDATE with src != dst, ping-pong A->B, B->A (both GENERIC, rest-of-buffer = whole VkBuffer, ALLOW_UPDATE; B never built).
// After each refit the SOURCE storage is wiped and the destination is traced (it must not reference the source).
static int arm_updsrc(char *why, size_t cap) {
    phase("updsrc-setup");
    struct Mesh M = mk_mesh(32, 32, -1.5f, -1.0f, 1.5f, 1.0f);
    mesh_z(&M, 3.0f, 0.15f, 0);
    VkAccelerationStructureGeometryKHR g = mesh_geom(&M);
    VkAccelerationStructureBuildSizesInfoKHR sz = sizes_of(BLASF, FB | AU, &g, M.nt);
    const VkDeviceSize drv = sz.accelerationStructureSize, vb = up(drv, 256);
    struct Buf AB[2] = {mkbuf_g(vb, KB64), mkbuf_g(vb, KB64)};
    VkDeviceSize smax = sz.buildScratchSize > sz.updateScratchSize ? sz.buildScratchSize : sz.updateScratchSize;
    struct Buf SC = mkbuf(up(smax, KB64) + 2 * KB64);
    VkDeviceSize sb = up(SC.a, KB64 > SA ? KB64 : SA) - SC.a;
    say("blas tris=%u as=%llu build_scratch=%llu update_scratch=%llu create_size=%llu (GENERIC, whole VkBuffer) guard_tail=%llu",
        M.nt, (unsigned long long)drv, (unsigned long long)sz.buildScratchSize, (unsigned long long)sz.updateScratchSize,
        (unsigned long long)vb, (unsigned long long)(AB[0].msize - vb));
    fill(&AB[0], 0x5A);
    fill(&AB[1], 0x5A);
    fill(&SC, 0x5A);
    VkAccelerationStructureKHR as[2] = {mkas(&AB[0], 0, vb, GENF), mkas(&AB[1], 0, vb, GENF)};
    uint8_t *snap = malloc(AB[0].msize);
    if (!snap) die(3, "malloc");
    struct HT h = mk_ht();
    begin(cbs[0]);
    rec_build(cbs[0], (struct Job){BLASF, FB | AU, false, as[0], g, M.nt, SC.a + sb});
    end(cbs[0]);
    submit_wait(cbs[0], "initial build A");
    struct Exp ex[48];
    uint32_t nr = mesh_rays(&M, ex, 24, 11, 0);
    int bad_total = trace_blas(&h, as_addr(as[0]), 0, 11, ex, nr, "initial A");
    fill(&SC, 0x5A);
    uint64_t g_beyond = 0, g_past = 0, g_sc = 0, src_changed = 0;
    int64_t f_beyond = -1, f_past = -1, f_sc = -1;
    int bad_iters = bad_total != 0, srcch_iters = 0;
    double ms = 0, ms_max = 0;
    for (int it = 1; it <= 100; it++) {
        phase("updsrc-iter");
        int s = (it & 1) ? 0 : 1, d = 1 - s;   // odd: A->B, even: B->A
        mesh_z(&M, 3.0f + 0.3f * sinf(0.11f * it), 0.15f, 0.37f * it);
        hinval(&AB[s]);
        memcpy(snap, AB[s].p, AB[s].msize);
        begin(cbs[0]);
        rec_build_g(cbs[0], BLASF, FB | AU, true, as[s], as[d], &g, &(VkAccelerationStructureBuildRangeInfoKHR){M.nt, 0, 0, 0}, 1, SC.a + sb);
        end(cbs[0]);
        double t = submit_wait(cbs[0], "update src!=dst");
        ms += t;
        if (t > ms_max) ms_max = t;
        if (it == 1) {   // B was pattern-only: its written extent is the update's output footprint
            struct Ext e;
            ext_reset(&e);
            scan(&AB[1], 0x5A, &e);
            char a[160], sg[200];
            int fa = ext_fmt(a, sizeof a, "dst", &e, 0, drv, vb);
            segs(&e, 0, sg, sizeof sg);
            say("iter1 (B never built) %s segs=%s%s%s", a, sg, fa & F_BEYOND_SIZE ? " DST_BEYOND_SIZE" : "", fa & F_BEYOND_STORAGE ? " PAST_BUFFER_WRITE" : "");
        }
        hinval(&AB[s]);
        uint64_t ch = 0;
        for (VkDeviceSize o = 0; o < AB[s].msize; o++) ch += snap[o] != AB[s].p[o];
        if (ch && !srcch_iters++) say("SRC_CHANGED first at iter %d: %llu bytes of the source changed by an update src!=dst", it, (unsigned long long)ch);
        src_changed += ch;
        for (int k = 0; k < 2; k++) {
            g_beyond += guard(&AB[k], drv, vb, 0x5A, &f_beyond);
            g_past += guard(&AB[k], vb, AB[k].msize, 0x5A, &f_past);
        }
        g_sc += guard(&SC, sb + sz.updateScratchSize, SC.size, 0x5A, &f_sc);
        memset(AB[s].p, 0xC3, drv);   // wipe the source (the next update writes into it again)
        hflush(&AB[s]);
        nr = mesh_rays(&M, ex, 24, 11, 0);
        char what[32];
        snprintf(what, sizeof what, "iter %d %s", it, d ? "A->B" : "B->A");
        int b = trace_blas(&h, as_addr(as[d]), 0, 11, ex, nr, what);
        bad_total += b;
        bad_iters += b != 0;
    }
    free(snap);
    say("time updsrc iters=100 ms_avg=%.2f ms_max=%.2f bad_iters=%d bad_rays=%d", ms / 100, ms_max, bad_iters, bad_total);
    say("guards beyond_size_bytes=%llu first=%lld past_buffer_end_bytes=%llu first=%lld scratch_bytes_hit=%llu first=%lld src_changed_bytes=%llu iters=%d%s%s%s",
        (unsigned long long)g_beyond, (long long)f_beyond, (unsigned long long)g_past, f_past < 0 ? -1ll : (long long)(f_past - (int64_t)vb),
        (unsigned long long)g_sc, f_sc < 0 ? -1ll : (long long)(f_sc - (int64_t)sb), (unsigned long long)src_changed, srcch_iters,
        g_beyond ? " AS_BEYOND_SIZE" : "", g_past ? " PAST_BUFFER_WRITE" : "", src_changed ? " SRC_CHANGED" : "");
    if (bad_total || g_beyond || g_past || g_sc || src_changed) {
        snprintf(why, cap, "wrong rays %d (%d iters), beyond-size %llu, past-end %llu, scratch %llu, src changed %llu bytes", bad_total, bad_iters,
                 (unsigned long long)g_beyond, (unsigned long long)g_past, (unsigned long long)g_sc, (unsigned long long)src_changed);
        return 1;
    }
    snprintf(why, cap, "100 refits src!=dst (A<->B, source wiped each time) correct, nothing written outside the driver sizes, source untouched");
    return 0;
}

// ================================================================ arm: updsrc big | bigdst
// UPDATE src != dst where the two handles have very different CREATED sizes (vkd3d: rest-of-buffer views at any offset).
// big:    source A = GENERIC view at offset 0, created size 8 MB; destinations B / A' = views with only the driver size (256 B
//         rounded) left before the END of their VkBuffer, 9 MB GPU-mapped guard tail each -> "refit sized by the source's
//         created size" shows as PAST_BUFFER_WRITE (measured, not faulted).
// bigdst: the reverse control: source A created with exactly the driver size, destinations = 8 MB views at offset 0 (1 MB
//         guard) -> USES_TAIL = the refit wrote past the driver size inside a large destination.
// Refit 1: src A -> dst B (B never built). Then 20 refits alternating B->A', A'->B; every 4th uses A as the source.
struct UH { struct Buf *b; VkDeviceSize off, create; VkAccelerationStructureKHR h; const char *name; };
static int arm_updsrc_big(bool bigdst, char *why, size_t cap) {
    phase("updsrc-big-setup");
    struct Mesh M = mk_mesh(32, 32, -1.5f, -1.0f, 1.5f, 1.0f);
    mesh_z(&M, 3.0f, 0.15f, 0);
    VkAccelerationStructureGeometryKHR g = mesh_geom(&M);
    VkAccelerationStructureBuildSizesInfoKHR sz = sizes_of(BLASF, FB | AU, &g, M.nt);
    const VkDeviceSize drv = sz.accelerationStructureSize, room = up(drv, 256);
    static struct Buf B3[3];
    struct UH u[3];
    if (!bigdst) {
        B3[0] = mkbuf_g(8 * MB, MB);
        B3[1] = mkbuf_g(8 * MB, 9 * MB);
        B3[2] = mkbuf_g(8 * MB, 9 * MB);
        u[0] = (struct UH){&B3[0], 0, 8 * MB, 0, "A(big)"};
        u[1] = (struct UH){&B3[1], 8 * MB - room, room, 0, "B"};
        u[2] = (struct UH){&B3[2], 8 * MB - room, room, 0, "A'"};
    } else {
        B3[0] = mkbuf_g(room, KB64);
        B3[1] = mkbuf_g(8 * MB, MB);
        B3[2] = mkbuf_g(8 * MB, MB);
        u[0] = (struct UH){&B3[0], 0, drv, 0, "A(exact)"};
        u[1] = (struct UH){&B3[1], 0, 8 * MB, 0, "B(8MB)"};
        u[2] = (struct UH){&B3[2], 0, 8 * MB, 0, "A'(8MB)"};
    }
    VkDeviceSize smax = sz.buildScratchSize > sz.updateScratchSize ? sz.buildScratchSize : sz.updateScratchSize;
    struct Buf SC = mkbuf(up(smax, KB64) + 2 * KB64);
    VkDeviceSize sb = up(SC.a, KB64 > SA ? KB64 : SA) - SC.a;
    for (int k = 0; k < 3; k++) { fill(u[k].b, 0x5A); u[k].h = mkas(u[k].b, u[k].off, u[k].create, GENF); }
    say("blas tris=%u accel_size=%llu build_scratch=%llu update_scratch=%llu | src %s created=%llu off=%llu | dst B/A' created=%llu "
        "off=%llu buffer=8388608 guard_tail=%llu", M.nt, (unsigned long long)drv, (unsigned long long)sz.buildScratchSize,
        (unsigned long long)sz.updateScratchSize, u[0].name, (unsigned long long)u[0].create, (unsigned long long)u[0].off,
        (unsigned long long)u[1].create, (unsigned long long)u[1].off, (unsigned long long)(u[1].b->msize - u[1].b->size));
    uint8_t *snap = malloc(u[1].b->msize > u[0].b->msize ? u[1].b->msize : u[0].b->msize);
    if (!snap) die(3, "malloc");
    struct HT h = mk_ht();
    begin(cbs[0]);
    rec_build(cbs[0], (struct Job){BLASF, FB | AU, false, u[0].h, g, M.nt, SC.a + sb});
    end(cbs[0]);
    double tb = submit_wait(cbs[0], "build A");
    struct Exp ex[48];
    uint32_t nr = mesh_rays(&M, ex, 24, 12, 0);
    int bad_total = trace_blas(&h, as_addr(u[0].h), 0, 12, ex, nr, "A after build");
    say("build A ms=%.2f rays_bad=%d of %u", tb, bad_total, nr);
    int nflag = 0, bad_iters = bad_total != 0, cur = 0;
    int64_t max_hi = -1, max_past = -1;
    uint64_t srcch_total = 0;
    double ms = 0, ms_max = 0;
    char first[300] = "";
    for (int it = 1; it <= 21; it++) {
        phase("updsrc-big-iter");
        int d = it == 1 ? 1 : (cur == 1 ? 2 : 1), s = it == 1 || it % 4 == 0 ? 0 : cur;   // refit 1: A->B; every 4th from A
        uint8_t pat = (it & 1) ? 0xA5 : 0x5A;
        if (it > 1) fill(u[d].b, pat); else pat = 0x5A;   // B is still pattern-only (never built) at refit 1
        mesh_z(&M, 3.0f + 0.3f * sinf(0.11f * it), 0.15f, 0.37f * it);
        hinval(u[s].b);
        memcpy(snap, u[s].b->p, u[s].b->msize);
        begin(cbs[0]);
        rec_build_g(cbs[0], BLASF, FB | AU, true, u[s].h, u[d].h, &g, &(VkAccelerationStructureBuildRangeInfoKHR){M.nt, 0, 0, 0}, 1, SC.a + sb);
        end(cbs[0]);
        double t = submit_wait(cbs[0], "update src!=dst");
        ms += t;
        if (t > ms_max) ms_max = t;
        struct Ext e;
        ext_reset(&e);
        scan(u[d].b, pat, &e);
        hinval(u[s].b);
        uint64_t ch = 0;
        for (VkDeviceSize o = 0; o < u[s].b->msize; o++) ch += snap[o] != u[s].b->p[o];
        srcch_total += ch;
        int64_t lo = e.lo < 0 ? -1 : e.lo - (int64_t)u[d].off, hi = e.lo < 0 ? -1 : e.hi - (int64_t)u[d].off;
        int64_t past = e.lo >= 0 && e.hi > (int64_t)u[d].b->size ? e.hi - (int64_t)u[d].b->size : -1;
        bool below = e.lo >= 0 && lo < 0, beyond = hi > (int64_t)drv, beyond_st = hi > (int64_t)u[d].create;
        char fl[160];
        snprintf(fl, sizeof fl, "%s%s%s%s%s", below ? "DST_BELOW," : "", beyond ? (bigdst ? "USES_TAIL,DST_BEYOND_SIZE," : "DST_BEYOND_SIZE,") : "",
                 beyond_st ? "DST_BEYOND_STORAGE," : "", past >= 0 ? "PAST_BUFFER_WRITE," : "", ch ? "SRC_CHANGED," : "");
        bool flagged = fl[0];
        if (fl[0]) fl[strlen(fl) - 1] = 0; else snprintf(fl, sizeof fl, "-");
        if (hi > max_hi) max_hi = hi;
        if (past > max_past) max_past = past;
        nr = mesh_rays(&M, ex, 24, 12, 0);
        char what[48];
        snprintf(what, sizeof what, "refit %d %s->%s", it, u[s].name, u[d].name);
        int b = trace_blas(&h, as_addr(u[d].h), 0, 12, ex, nr, what);
        bad_total += b;
        bad_iters += b != 0;
        if (it == 1 || flagged || b) {
            if (flagged) nflag++;
            if (it == 1 || nflag <= 4)
                say("refit %d %s->%s dst_off=%llu dst_create=%llu dst_written=[%lld,%lld) past_end_hi=%lld src_changed_bytes=%llu ms=%.2f rays_bad=%d flags=%s",
                    it, u[s].name, u[d].name, (unsigned long long)u[d].off, (unsigned long long)u[d].create, (long long)lo, (long long)hi,
                    (long long)past, (unsigned long long)ch, t, b, fl);
            if (it == 1) snprintf(first, sizeof first, "refit 1 %s->%s written=[%lld,%lld) of size %llu %s", u[s].name, u[d].name, (long long)lo,
                                  (long long)hi, (unsigned long long)drv, fl);
        }
        cur = d;
    }
    free(snap);
    say("time updsrc-%s refits=21 ms_avg=%.2f ms_max=%.2f flagged_refits=%d max_dst_hi=%lld (accel_size %llu) max_past_end_hi=%lld src_changed_bytes=%llu bad_iters=%d bad_rays=%d",
        bigdst ? "bigdst" : "big", ms / 21, ms_max, nflag, (long long)max_hi, (unsigned long long)drv, (long long)max_past,
        (unsigned long long)srcch_total, bad_iters, bad_total);
    if (nflag || bad_total) {
        snprintf(why, cap, "%s; %d of 21 refits flagged, max written %lld of size %llu, max past end %lld, wrong rays %d", first, nflag,
                 (long long)max_hi, (unsigned long long)drv, (long long)max_past, bad_total);
        return 1;
    }
    snprintf(why, cap, "21 refits src!=dst (%s) stay inside [dst, dst+accel_size), source untouched, rays correct",
             bigdst ? "exact source -> 8 MB destinations" : "8 MB source view -> end-of-buffer destinations");
    return 0;
}

// ================================================================ vertex formats (formats, vfmt)
enum VK { K_F32, K_F16, K_SN16, K_UN16, K_UN1010102, K_UN8, K_SN8 };
struct VFmt { const char *name; VkFormat f; int nc; enum VK k; uint32_t stride; };
static const struct VFmt VFMTS[] = {
    {"r32g32b32_sfloat", VK_FORMAT_R32G32B32_SFLOAT, 3, K_F32, 12},
    {"r32g32_sfloat", VK_FORMAT_R32G32_SFLOAT, 2, K_F32, 8},
    {"r16g16b16a16_sfloat", VK_FORMAT_R16G16B16A16_SFLOAT, 4, K_F16, 8},
    {"r16g16_sfloat", VK_FORMAT_R16G16_SFLOAT, 2, K_F16, 4},
    {"r16g16b16a16_snorm", VK_FORMAT_R16G16B16A16_SNORM, 4, K_SN16, 8},
    {"r16g16_snorm", VK_FORMAT_R16G16_SNORM, 2, K_SN16, 4},
    {"r16g16b16a16_unorm", VK_FORMAT_R16G16B16A16_UNORM, 4, K_UN16, 8},
    {"r16g16_unorm", VK_FORMAT_R16G16_UNORM, 2, K_UN16, 4},
    {"r10g10b10a2_unorm", VK_FORMAT_A2B10G10R10_UNORM_PACK32, 4, K_UN1010102, 4},
    {"r8g8b8a8_unorm", VK_FORMAT_R8G8B8A8_UNORM, 4, K_UN8, 4},
    {"r8g8_unorm", VK_FORMAT_R8G8_UNORM, 2, K_UN8, 2},
    {"r8g8b8a8_snorm", VK_FORMAT_R8G8B8A8_SNORM, 4, K_SN8, 4},
    {"r8g8_snorm", VK_FORMAT_R8G8_SNORM, 2, K_SN8, 2},
};
#define NVFMT (int)(sizeof VFMTS / sizeof VFMTS[0])
static uint16_t f2h(float f) {   // round-to-nearest-even, normal/zero range only (all probe values are 0 or in [2^-14, 65504])
    uint32_t x;
    memcpy(&x, &f, 4);
    uint32_t s = x >> 16 & 0x8000, e = x >> 23 & 0xFF, m = x & 0x7FFFFF;
    if (e == 0 && m == 0) return (uint16_t)s;
    int he = (int)e - 127 + 15;
    uint32_t hm = m >> 13, rest = m & 0x1FFF;
    if (rest > 0x1000 || (rest == 0x1000 && (hm & 1))) { hm++; if (hm == 0x400) { hm = 0; he++; } }
    return (uint16_t)(s | (uint32_t)he << 10 | hm);
}
static float h2f(uint16_t h) {
    uint32_t s = (uint32_t)(h & 0x8000) << 16, e = h >> 10 & 0x1F, m = h & 0x3FF, x;
    if (e == 0 && m == 0) x = s;
    else x = s | (e - 15 + 127) << 23 | m << 13;
    float f;
    memcpy(&f, &x, 4);
    return f;
}
// Encodes vertex v (xyz) into dst in format F; returns the decoded position the driver must use (missing z = 0).
static void venc(const struct VFmt *F, const float v[3], uint8_t *dst, float out[3]) {
    out[2] = 0;
    float c[4] = {v[0], v[1], v[2], 1.0f};
    if (F->k == K_UN1010102) {
        uint32_t q[3], w = 3u << 30;
        for (int i = 0; i < 3; i++) { q[i] = (uint32_t)lrintf(fminf(fmaxf(c[i], 0), 1) * 1023.0f); out[i] = q[i] / 1023.0f; }
        w |= q[0] | q[1] << 10 | q[2] << 20;
        memcpy(dst, &w, 4);
        return;
    }
    for (int i = 0; i < F->nc; i++) {
        float d = 0;
        switch (F->k) {
        case K_F32: memcpy(dst + 4 * i, &c[i], 4); d = c[i]; break;
        case K_F16: { uint16_t h = f2h(c[i]); memcpy(dst + 2 * i, &h, 2); d = h2f(h); break; }
        case K_SN16: { int16_t q = (int16_t)lrintf(fminf(fmaxf(c[i], -1), 1) * 32767.0f); memcpy(dst + 2 * i, &q, 2); d = fmaxf(q / 32767.0f, -1.0f); break; }
        case K_UN16: { uint16_t q = (uint16_t)lrintf(fminf(fmaxf(c[i], 0), 1) * 65535.0f); memcpy(dst + 2 * i, &q, 2); d = q / 65535.0f; break; }
        case K_UN8: { uint8_t q = (uint8_t)lrintf(fminf(fmaxf(c[i], 0), 1) * 255.0f); dst[i] = q; d = q / 255.0f; break; }
        case K_SN8: { int8_t q = (int8_t)lrintf(fminf(fmaxf(c[i], -1), 1) * 127.0f); memcpy(dst + i, &q, 1); d = fmaxf(q / 127.0f, -1.0f); break; }
        default: break;
        }
        if (i < 3) out[i] = d;
    }
}
static const struct VFmt *vfmt_by_name(const char *n) {
    for (int i = 0; i < NVFMT; i++) if (!strcmp(VFMTS[i].name, n)) return &VFMTS[i];
    if (!strcmp(n, "a2b10g10r10_unorm_pack32")) return &VFMTS[8];
    return NULL;
}
static bool vfmt_feature(VkFormat f, VkFormatFeatureFlags *buf) {
    VkFormatProperties fp;
    vkGetPhysicalDeviceFormatProperties(pd, f, &fp);
    *buf = fp.bufferFeatures;
    return fp.bufferFeatures & VK_FORMAT_FEATURE_ACCELERATION_STRUCTURE_VERTEX_BUFFER_BIT_KHR;
}

// ================================================================ arm: formats
static int arm_formats(char *why, size_t cap) {
    phase("formats");
    int n = 0;
    char list[200] = "";
    size_t l = 0;
    for (int i = 0; i < NVFMT; i++) {
        VkFormatFeatureFlags bf;
        bool ok = vfmt_feature(VFMTS[i].f, &bf);
        n += ok;
        say("format %s vk=%d as_vertex_buffer=%d bufferFeatures=0x%08x", VFMTS[i].name, (int)VFMTS[i].f, ok, bf);
        if (!ok && l < sizeof list) l += snprintf(list + l, sizeof list - l, "%s%s", l ? "," : "", VFMTS[i].name);
    }
    snprintf(why, cap, "%d of %d DXR vertex formats have AS_VERTEX_BUFFER%s%s", n, NVFMT, l ? "; missing: " : "", list);
    return 0;
}

// ================================================================ arm: vfmt
// Two-triangle BLAS with vertices in format F; cases idx16, idx32, idx16+transform, idx32+transform; rays vs the CPU decode.
static int arm_vfmt(const struct VFmt *F, char *why, size_t cap) {
    phase("vfmt-setup");
    VkFormatFeatureFlags bf;
    bool feat = vfmt_feature(F->f, &bf);
    say("format %s vk=%d stride=%u feature=%d bufferFeatures=0x%08x%s", F->name, (int)F->f, F->stride, feat, bf,
        feat ? "" : " (no AS_VERTEX_BUFFER bit: building anyway, as the game does)");
    static const float V[4][3] = {{0.10f, 0.10f, 0.20f}, {0.90f, 0.15f, 0.60f}, {0.85f, 0.90f, 0.80f}, {0.12f, 0.88f, 0.40f}};
    static const uint32_t IDX[6] = {0, 1, 2, 0, 2, 3};
    static const float XF[3][4] = {{1.5f, 0.0f, 0.0f, -0.5f}, {0.2f, 0.8f, 0.0f, -0.3f}, {0.5f, 0.0f, 1.0f, 1.0f}};
    static const float W[5][3] = {{1 / 3.f, 1 / 3.f, 1 / 3.f}, {0.6f, 0.25f, 0.15f}, {0.2f, 0.3f, 0.5f}, {0.15f, 0.7f, 0.15f}, {0.25f, 0.15f, 0.6f}};
    struct Buf VB = mkbuf(256), IB16 = mkbuf(256), IB32 = mkbuf(256), XB = mkbuf(256);
    float dec[4][3];
    // signed formats (float/snorm) get coordinates shifted by -0.5 so negative values and the sign are exercised too
    const bool sgn = F->k == K_F32 || F->k == K_F16 || F->k == K_SN16 || F->k == K_SN8;
    for (int i = 0; i < 4; i++) {
        float v[3] = {V[i][0] - (sgn ? 0.5f : 0), V[i][1] - (sgn ? 0.5f : 0), V[i][2] - (sgn ? 0.5f : 0)};
        venc(F, v, VB.p + (size_t)i * F->stride, dec[i]);
    }
    hflush(&VB);
    for (int i = 0; i < 6; i++) { ((uint16_t *)IB16.p)[i] = (uint16_t)IDX[i]; ((uint32_t *)IB32.p)[i] = IDX[i]; }
    hflush(&IB16);
    hflush(&IB32);
    VkTransformMatrixKHR xm;
    memcpy(xm.matrix, XF, sizeof XF);
    memcpy(XB.p, &xm, sizeof xm);
    hflush(&XB);
    say("decoded v0=(%.6f,%.6f,%.6f) v1=(%.6f,%.6f,%.6f) v2=(%.6f,%.6f,%.6f) v3=(%.6f,%.6f,%.6f)", dec[0][0], dec[0][1], dec[0][2], dec[1][0],
        dec[1][1], dec[1][2], dec[2][0], dec[2][1], dec[2][2], dec[3][0], dec[3][1], dec[3][2]);
    struct HT h = mk_ht();
    const float TZ = 2.0f;
    static const char *const cn[4] = {"idx16", "idx32", "idx16+xform", "idx32+xform"};
    int bad = 0, bad_cases = 0;
    char res[160] = "";
    size_t rl = 0;
    for (int c = 0; c < 4; c++) {
        phase(cn[c]);
        bool i32 = c & 1, xf = c >= 2;
        VkAccelerationStructureGeometryKHR g = {VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR};
        g.geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR;
        g.flags = VK_GEOMETRY_OPAQUE_BIT_KHR;
        g.geometry.triangles.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR;
        g.geometry.triangles.vertexFormat = F->f;
        g.geometry.triangles.vertexData.deviceAddress = VB.a;
        g.geometry.triangles.vertexStride = F->stride;
        g.geometry.triangles.maxVertex = 3;
        g.geometry.triangles.indexType = i32 ? VK_INDEX_TYPE_UINT32 : VK_INDEX_TYPE_UINT16;
        g.geometry.triangles.indexData.deviceAddress = i32 ? IB32.a : IB16.a;
        g.geometry.triangles.transformData.deviceAddress = xf ? XB.a : 0;
        uint32_t prims = 2;
        VkAccelerationStructureBuildSizesInfoKHR sz = sizes_of_g(BLASF, FT, &g, &prims, 1);
        struct Buf S = mkbuf_g(up(sz.accelerationStructureSize, 256), KB64), SC = mkbuf(sz.buildScratchSize + SA);
        fill(&S, 0x5A);
        VkAccelerationStructureKHR as = mkas(&S, 0, S.size, GENF);
        begin(cbs[0]);
        rec_build_g(cbs[0], BLASF, FT, false, VK_NULL_HANDLE, as, &g, &(VkAccelerationStructureBuildRangeInfoKHR){2, 0, 0, 0}, 1, up(SC.a, SA));
        end(cbs[0]);
        double ms = submit_wait(cbs[0], "vfmt build");
        int64_t gf = -1;
        uint64_t gb = guard(&S, sz.accelerationStructureSize, S.msize, 0x5A, &gf);
        // expected rays: transformed decoded vertices, 5 barycentric points per triangle + 4 misses
        float P[4][3];
        for (int i = 0; i < 4; i++)
            for (int r = 0; r < 3; r++)
                P[i][r] = xf ? XF[r][0] * dec[i][0] + XF[r][1] * dec[i][1] + XF[r][2] * dec[i][2] + XF[r][3] : dec[i][r];
        struct Exp ex[16];
        uint32_t nr = 0;
        for (int t = 0; t < 2; t++)
            for (int w = 0; w < 5; w++) {
                const uint32_t *ix = IDX + 3 * t;
                float x = 0, y = 0, z = 0;
                for (int q = 0; q < 3; q++) { x += W[w][q] * P[ix[q]][0]; y += W[w][q] * P[ix[q]][1]; z += W[w][q] * P[ix[q]][2]; }
                struct Exp e = {x, y, 1, z + TZ, 21, t, 1};
                ex[nr++] = e;
            }
        ex[nr++] = miss_at(-3.0f, 0.5f);
        ex[nr++] = miss_at(4.0f, 0.5f);
        ex[nr++] = miss_at(0.5f, -3.0f);
        ex[nr++] = miss_at(0.5f, 4.0f);
        char what[48];
        snprintf(what, sizeof what, "%s", cn[c]);
        int b = trace_blas(&h, as_addr(as), TZ, 21, ex, nr, what);
        say("case %s as=%llu build_scratch=%llu build_ms=%.2f rays_bad=%d of %u past_size_bytes=%llu%s", cn[c],
            (unsigned long long)sz.accelerationStructureSize, (unsigned long long)sz.buildScratchSize, ms, b, nr, (unsigned long long)gb,
            gb ? " AS_BEYOND_SIZE" : "");
        bad += b + (gb ? 1 : 0);
        bad_cases += b || gb;
        rl += snprintf(res + rl, sizeof res - rl, "%s%s=%s", rl ? " " : "", cn[c], b || gb ? "bad" : "ok");
        rmas(as);
    }
    if (bad) { snprintf(why, cap, "%d of 4 cases wrong (%s)%s", bad_cases, res, feat ? "" : " feature=0"); return 1; }
    snprintf(why, cap, "4 cases (idx16/idx32, +/- 3x4 transform) hit/t/prim match the CPU decode%s", feat ? "" : " feature=0");
    return 0;
}

// ================================================================ arm: multigeom
// One BLAS, 5 triangle geometries (own vertex/index buffers; geometry 2 with 16-bit indices, geometry 3 with a 3x4 transform),
// BUILD then 50 in-place UPDATEs; rays check geometry index + primitive index + t.
static int arm_multigeom(char *why, size_t cap) {
    phase("multigeom-setup");
    struct Mesh m[5] = {mk_mesh(4, 4, -3.0f, -0.9f, -2.0f, 0.9f), mk_mesh(8, 4, -1.8f, -0.9f, -0.8f, 0.9f), mk_mesh(6, 6, -0.6f, -0.9f, 0.6f, 0.9f),
                        mk_mesh(3, 5, -1.0f, -1.8f, 1.0f, 1.8f), mk_mesh(10, 2, 2.0f, -0.9f, 3.0f, 0.9f)};
    // geometry 3: x' = 0.5x + 1.3, y' = 0.5y, z' = z + 0.7  -> occupies [0.8,1.8] x [-0.9,0.9]
    static const float XF3[3][4] = {{0.5f, 0, 0, 1.3f}, {0, 0.5f, 0, 0}, {0, 0, 1, 0.7f}};
    struct Buf XB = mkbuf(256);
    memcpy(XB.p, XF3, sizeof XF3);
    hflush(&XB);
    struct Buf I16 = mkbuf(up((VkDeviceSize)m[2].nt * 6, 256));
    for (uint32_t i = 0; i < m[2].nt * 3; i++) ((uint16_t *)I16.p)[i] = (uint16_t)((uint32_t *)m[2].ib.p)[i];
    hflush(&I16);
    VkAccelerationStructureGeometryKHR g[5];
    VkAccelerationStructureBuildRangeInfoKHR rg[5];
    uint32_t prims[5];
    for (int k = 0; k < 5; k++) {
        mesh_z(&m[k], 3.0f + 0.2f * k, 0.1f, (float)k);
        g[k] = mesh_geom(&m[k]);
        if (k == 2) { g[k].geometry.triangles.indexType = VK_INDEX_TYPE_UINT16; g[k].geometry.triangles.indexData.deviceAddress = I16.a; }
        if (k == 3) g[k].geometry.triangles.transformData.deviceAddress = XB.a;
        prims[k] = m[k].nt;
        VkAccelerationStructureBuildRangeInfoKHR r = {m[k].nt, 0, 0, 0};
        rg[k] = r;
    }
    VkAccelerationStructureBuildSizesInfoKHR sz = sizes_of_g(BLASF, FB | AU, g, prims, 5);
    const VkDeviceSize drv = sz.accelerationStructureSize, vb = up(drv, 256);
    say("blas geoms=5 tris=%u/%u/%u/%u/%u (geom2 idx16, geom3 transform) as=%llu build_scratch=%llu update_scratch=%llu", prims[0], prims[1],
        prims[2], prims[3], prims[4], (unsigned long long)drv, (unsigned long long)sz.buildScratchSize, (unsigned long long)sz.updateScratchSize);
    struct Buf S = mkbuf_g(vb, KB64);
    VkDeviceSize smax = sz.buildScratchSize > sz.updateScratchSize ? sz.buildScratchSize : sz.updateScratchSize;
    struct Buf SC = mkbuf(up(smax, KB64) + 2 * KB64);
    VkDeviceSize sb = up(SC.a, KB64 > SA ? KB64 : SA) - SC.a;
    fill(&S, 0x5A);
    fill(&SC, 0x5A);
    VkAccelerationStructureKHR as = mkas(&S, 0, vb, GENF);
    struct HT h = mk_ht();
    begin(cbs[0]);
    rec_build_g(cbs[0], BLASF, FB | AU, false, VK_NULL_HANDLE, as, g, rg, 5, SC.a + sb);
    end(cbs[0]);
    double tb = submit_wait(cbs[0], "initial build");
    int64_t f_as = -1, f_sc = -1;
    uint64_t g_as = guard(&S, drv, S.msize, 0x5A, &f_as), g_sc = 0;
    fill(&SC, 0x5A);
    struct Exp ex[48];
    int bad_total = 0, bad_iters = 0;
    double ms = 0, ms_max = 0;
    for (int it = 0; it <= 50; it++) {
        phase(it ? "multigeom-update" : "multigeom-build");
        if (it) {
            for (int k = 0; k < 5; k++) mesh_z(&m[k], 3.0f + 0.2f * k + 0.3f * sinf(0.13f * it + k), 0.1f, 0.41f * it + k);
            begin(cbs[0]);
            rec_build_g(cbs[0], BLASF, FB | AU, true, as, as, g, rg, 5, SC.a + sb);
            end(cbs[0]);
            double t = submit_wait(cbs[0], "update");
            ms += t;
            if (t > ms_max) ms_max = t;
        }
        uint32_t nr = 0;
        for (int k = 0; k < 5; k++)
            for (uint32_t s = 0; s < 6; s++) {
                struct Exp e = mesh_pt(&m[k], s, 0, 0, k == 3 ? XF3[2][3] : 0, 300);
                if (k == 3) { e.x = XF3[0][0] * e.x + XF3[0][3]; e.y = XF3[1][1] * e.y + XF3[1][3]; }
                e.geomp1 = k + 1;
                ex[nr++] = e;
            }
        static const float mx[6] = {-1.9f, -0.7f, 0.7f, 1.9f, 3.5f, 0.0f}, my[6] = {0, 0, 0, 0, 0, 1.5f};
        for (int s = 0; s < 6; s++) ex[nr++] = miss_at(mx[s], my[s]);
        char what[32];
        snprintf(what, sizeof what, it ? "update %d" : "build", it);
        int b = trace_blas(&h, as_addr(as), 0, 300, ex, nr, what);
        bad_total += b;
        bad_iters += b != 0;
        g_as += guard(&S, drv, S.msize, 0x5A, &f_as);
        if (it) g_sc += guard(&SC, sb + sz.updateScratchSize, SC.size, 0x5A, &f_sc);
    }
    say("time multigeom build_ms=%.2f updates=50 ms_avg=%.2f ms_max=%.2f bad_iters=%d bad_rays=%d of %d", tb, ms / 50, ms_max, bad_iters, bad_total, 51 * 36);
    say("guards storage_bytes_hit=%llu first_storage_off=%lld scratch_bytes_hit=%llu first_scratch_off=%lld%s", (unsigned long long)g_as,
        (long long)f_as, (unsigned long long)g_sc, f_sc < 0 ? -1ll : (long long)(f_sc - (int64_t)sb), g_as ? " AS_BEYOND_SIZE" : "");
    if (bad_total || g_as || g_sc) {
        snprintf(why, cap, "wrong rays %d (%d of 51 traces), storage guard %llu, scratch guard %llu", bad_total, bad_iters, (unsigned long long)g_as,
                 (unsigned long long)g_sc);
        return 1;
    }
    snprintf(why, cap, "5-geometry BLAS: build + 50 in-place refits, geometry/primitive index + t correct, nothing past the driver sizes");
    return 0;
}

// ================================================================ v3 arms (refitchain, copyupd, bufclone, copyext, aliasupd)
// Which compat-layer strategies are safe on a driver whose UPDATE src != dst / CLONE / COMPACT misbehave. One step runner:
// every step (BUILD / UPDATE / CLONE / COMPACT / vkCmdCopyBuffer) prints a `step=N begin` marker first (logcat GROUP_ faults
// can be lined up with it), is measured against a snapshot of the destination buffer (incl. its GPU-mapped guard tail), of the
// source buffer and against a pattern-filled scratch buffer, and its destination is traced (24 hits + 8 misses vs the CPU).
// The first failing counted step is kept in FIRSTFAIL and named by the verdict (also by a VKERROR / TIMEOUT / crash line).
// A view: GENERIC handle on (buffer, offset) with a created size; cur = size of the AS it holds now (BEYOND_SIZE reference).
struct V { struct Buf *b; VkDeviceSize off, create, cur; VkAccelerationStructureKHR h; const char *name; };
enum Op { OP_BUILD, OP_UPDATE, OP_CLONE, OP_COMPACT, OP_BUFCOPY };
static const char *const OPN[] = {"BUILD", "UPDATE", "CLONE", "COMPACT", "BUFCOPY"};
struct St {
    enum Op op;
    struct V *src, *dst;
    const VkAccelerationStructureGeometryKHR *g;
    uint32_t prims;
    VkBuildAccelerationStructureFlagsKHR fl;
    VkDeviceSize size;        // expected size of the result: BUILD accel size, UPDATE/CLONE the source's, COMPACT compacted, BUFCOPY bytes
    VkDeviceSize scr;         // scratch the op may use (build / update scratch size); 0 = no scratch
    const struct Mesh *m;     // rays through the destination after the step (NULL: no trace)
    bool wipe_src;            // wipe the source storage (0xC3, cur bytes) before tracing the destination
    bool brk;                 // break phase (rebuild-heals): printed, never counted, never FIRSTFAIL
};
static struct {
    int n, bad, consec, slow, brk_n, brk_bad, counted, stop_at;
    bool stopped;
    double ms_sum, ms_max;
    uint8_t *sd, *ss;
    size_t cap;
    struct Buf SC;
    VkDeviceSize sb;
    struct HT h;
    VkDeviceSize drv, compacted, xval;   // hi_is candidates (xname/xval: one extra, e.g. clone-alias h1_create)
    const char *xname;
} RS;
#define CUSTOM3 14
#define STOP_CONSEC 10   // a counted chain stops after this many consecutive failing steps (each broken refit can cost a ~0.5 s queue reset)

static void rs_init(VkDeviceSize scratch_max) {
    RS.SC = mkbuf_g(up(scratch_max, KB64) + 2 * KB64, 4 * MB);   // 4 MB guard: wild scratch writes are measured, not faulted
    RS.sb = up(RS.SC.a, KB64 > SA ? KB64 : SA) - RS.SC.a;
    RS.h = mk_ht();
    RS.stop_at = STOP_CONSEC;
}
static void snapcap(size_t need) {   // two host snapshot buffers (destination, source) of at least need bytes
    if (need <= RS.cap) return;
    free(RS.sd);
    free(RS.ss);
    RS.sd = malloc(need);
    RS.ss = malloc(need);
    if (!RS.sd || !RS.ss) die(3, "malloc %zu", need);
    RS.cap = need;
}
// Bytes of b in [from, to) that are not pat: count + highest offset + 1 (absolute, -1 = none). 4 KB memcmp fast path.
static void patscan(const struct Buf *b, VkDeviceSize from, VkDeviceSize to, uint8_t pat, uint64_t *n, int64_t *hi) {
    static uint8_t blk[4096];
    static int bp = -1;
    if (bp != pat) { memset(blk, pat, sizeof blk); bp = pat; }
    if (to > b->msize) to = b->msize;
    for (VkDeviceSize o = from; o < to;) {
        VkDeviceSize e = (o / 4096 + 1) * 4096;
        if (e > to) e = to;
        if (memcmp(b->p + o, blk, e - o))
            for (VkDeviceSize k = o; k < e; k++) if (b->p[k] != pat) { (*n)++; *hi = (int64_t)k + 1; }
        o = e;
    }
}
// Changed bytes of b vs snap in [from, to): count, lowest offset, highest offset + 1 (absolute; -1 = none).
struct Dif { uint64_t n; int64_t lo, hi; };
static struct Dif dif(const struct Buf *b, const uint8_t *snap, VkDeviceSize from, VkDeviceSize to) {
    struct Dif d = {0, -1, -1};
    if (to > b->msize) to = b->msize;
    for (VkDeviceSize o = from; o < to;) {
        VkDeviceSize e = (o / 4096 + 1) * 4096;
        if (e > to) e = to;
        if (memcmp(b->p + o, snap + o, e - o))
            for (VkDeviceSize k = o; k < e; k++)
                if (b->p[k] != snap[k]) { if (d.lo < 0) d.lo = (int64_t)k; d.hi = (int64_t)k + 1; d.n++; }
        o = e;
    }
    return d;
}
static void rs_state(struct Mesh *m, int k) { mesh_z(m, 3.0f + 0.3f * sinf(0.11f * k), 0.15f, 0.37f * k); }

// Runs one step; returns 1 if it failed (any flag or wrong ray).
static int run_step(const struct St *s) {
    static char ph[96];
    struct V *d = s->dst, *sr = s->src;
    int n = ++RS.n;
    snprintf(ph, sizeof ph, "step=%d %s %s->%s", n, OPN[s->op], sr ? sr->name : "new", d->name);
    phase(ph);
    const bool same = !sr || sr->b == d->b;
    snapcap(d->b->msize > (sr ? sr->b->msize : 0) ? d->b->msize : sr->b->msize);
    hinval(d->b);
    memcpy(RS.sd, d->b->p, d->b->msize);
    if (!same) { hinval(sr->b); memcpy(RS.ss, sr->b->p, sr->b->msize); }
    if (s->scr) fill(&RS.SC, 0x5A);
    say("step=%d begin %s %s->%s%s", n, OPN[s->op], sr ? sr->name : "new", d->name, s->brk ? " (break phase)" : "");
    VkBuildAccelerationStructureFlagsKHR fl = s->fl;
    bool rec = true;
    /* NEG:skip */
    begin(cbs[0]);
    fullbar(cbs[0]);
    if (rec) switch (s->op) {
    case OP_BUILD:
    case OP_UPDATE:
        rec_build_g(cbs[0], BLASF, fl, s->op == OP_UPDATE, sr ? sr->h : VK_NULL_HANDLE, d->h, s->g,
                    &(VkAccelerationStructureBuildRangeInfoKHR){s->prims, 0, 0, 0}, 1, RS.SC.a + RS.sb);
        break;
    case OP_CLONE:
    case OP_COMPACT: {
        VkCopyAccelerationStructureInfoKHR ci = {VK_STRUCTURE_TYPE_COPY_ACCELERATION_STRUCTURE_INFO_KHR, NULL, sr->h, d->h,
            s->op == OP_COMPACT ? VK_COPY_ACCELERATION_STRUCTURE_MODE_COMPACT_KHR : VK_COPY_ACCELERATION_STRUCTURE_MODE_CLONE_KHR};
        pCopy(cbs[0], &ci);
        break;
    }
    case OP_BUFCOPY: {
        VkBufferCopy r = {sr->off, d->off, s->size};
        vkCmdCopyBuffer(cbs[0], sr->b->b, d->b->b, 1, &r);
        break;
    }
    }
    end(cbs[0]);
    double ms = submit_wait(cbs[0], ph);
    /* NEG:after */
    // ---- measure
    hinval(d->b);
    struct Dif below = dif(d->b, RS.sd, 0, d->off), in = dif(d->b, RS.sd, d->off, d->b->msize);
    struct Dif bst = dif(d->b, RS.sd, d->off + d->create, d->b->size), past = dif(d->b, RS.sd, d->b->size, d->b->msize);
    struct Dif sch = {0, -1, -1};
    if (!same) { hinval(sr->b); sch = dif(sr->b, RS.ss, 0, sr->b->msize); }
    uint64_t scr_n = 0;
    int64_t scr_hi = -1;
    if (s->scr) {
        hinval(&RS.SC);   // outside [sb, sb + scr): below the scratch base and past the allowed scratch size (incl. the 4 MB guard)
        patscan(&RS.SC, 0, RS.sb, 0x5A, &scr_n, &scr_hi);
        patscan(&RS.SC, RS.sb + s->scr, RS.SC.msize, 0x5A, &scr_n, &scr_hi);
        if (scr_hi >= 0) scr_hi -= (int64_t)RS.sb;
    }
    const int64_t lo = in.lo < 0 ? -1 : in.lo - (int64_t)d->off, hi = in.lo < 0 ? -1 : in.hi - (int64_t)d->off;
    const int64_t past_hi = past.n ? past.hi - (int64_t)d->b->size : -1;
    char fl_s[200];
    size_t l = 0;
    fl_s[0] = 0;
    #define FLG(c, t) do { if (c) l += snprintf(fl_s + l, sizeof fl_s - l, "%s,", t); } while (0)
    FLG(below.n, "DST_BELOW");
    FLG(hi > (int64_t)s->size, s->op == OP_COMPACT ? "DST_BEYOND_COMPACT" : "DST_BEYOND_SIZE");
    FLG(hi > (int64_t)d->create, "DST_BEYOND_STORAGE");
    FLG(past.n, "PAST_BUFFER_WRITE");
    FLG(scr_n, "SCR_OUTSIDE");
    FLG(sch.n, "SRC_CHANGED");
    // hi_is: which known size the written end equals
    char his[160];
    size_t hl = 0;
    his[0] = 0;
    // exact match "name", or "name-N" when the written end is N < 256 bytes short of it (unchanged tail bytes are invisible to a diff)
    #define HIS(c, t) do { int64_t c_ = (int64_t)(c); if (hi >= 0 && c_ && hi <= c_ && c_ - hi < 256 && hl < sizeof his) { \
        if (hi == c_) hl += snprintf(his + hl, sizeof his - hl, "%s,", t); \
        else hl += snprintf(his + hl, sizeof his - hl, "%s-%lld,", t, (long long)(c_ - hi)); } } while (0)
    HIS(s->size, "size");
    HIS(RS.drv, "accel_size");
    HIS(RS.compacted, "compacted");
    HIS(up(RS.compacted, KB64), "compacted_64k");
    HIS(sr ? sr->create : 0, "src_create");
    HIS(d->create, "dst_create");
    HIS(d->b->size - d->off, "dst_rest");
    HIS(RS.xval, RS.xname);
    if (hl) his[hl - 1] = 0; else snprintf(his, sizeof his, "-");
    d->cur = s->size;
    if (s->wipe_src && sr) { memset(sr->b->p + sr->off, 0xC3, sr->cur); hflush(sr->b); }
    int bad = 0;
    uint32_t nr = 0;
    if (s->m) {
        struct Exp ex[48];
        nr = mesh_rays(s->m, ex, 24, CUSTOM3, 0);
        VkDeviceAddress ta = as_addr(d->h);
        /* NEG:trace */
        bad = trace_blas(&RS.h, ta, 0, CUSTOM3, ex, nr, ph);
    }
    FLG(bad, "RAYS_BAD");
    if (l) fl_s[l - 1] = 0; else snprintf(fl_s, sizeof fl_s, "-");
    #undef FLG
    #undef HIS
    say("step=%d %s %s->%s%s dst_off=%llu dst_create=%llu size=%llu dst_written=[%lld,%lld) hi_is=%s beyond_storage_bytes=%llu "
        "past_end_bytes=%llu past_end_hi=%lld scr_outside_bytes=%llu scr_hi=%lld src_changed_bytes=%llu ms=%.2f rays_bad=%d/%u flags=%s",
        n, OPN[s->op], sr ? sr->name : "new", d->name, s->brk ? " (break)" : "", (unsigned long long)d->off, (unsigned long long)d->create,
        (unsigned long long)s->size, (long long)lo, (long long)hi, his, (unsigned long long)bst.n, (unsigned long long)past.n,
        (long long)past_hi, (unsigned long long)scr_n, (long long)scr_hi, (unsigned long long)sch.n, ms, bad, nr, fl_s);
    const int fail = l != 0;
    if (ms > 250) RS.slow++;
    if (s->brk) { RS.brk_n++; RS.brk_bad += fail; return fail; }
    RS.counted++;
    RS.ms_sum += ms;
    if (ms > RS.ms_max) RS.ms_max = ms;
    if (fail) {
        RS.bad++;
        RS.consec++;
        if (!FIRSTFAIL[0])
            snprintf(FIRSTFAIL, sizeof FIRSTFAIL, "step=%d %s %s->%s written=[%lld,%lld) past_end_hi=%lld rays_bad=%d flags=%s", n, OPN[s->op],
                     sr ? sr->name : "new", d->name, (long long)lo, (long long)hi, (long long)past_hi, bad, fl_s);
    } else RS.consec = 0;
    if (RS.consec >= RS.stop_at && !RS.stopped) {
        RS.stopped = true;
        say("STOP chain after %d consecutive failing steps (at step=%d)", RS.consec, n);
    }
    return fail;
}
static int s_build(struct V *d, const struct Mesh *m, const VkAccelerationStructureGeometryKHR *g, VkBuildAccelerationStructureFlagsKHR fl,
                   const VkAccelerationStructureBuildSizesInfoKHR *sz, bool brk) {
    struct St s = {OP_BUILD, NULL, d, g, m->nt, fl, sz->accelerationStructureSize, sz->buildScratchSize, m, false, brk};
    return run_step(&s);
}
static int s_update(struct V *src, struct V *d, const struct Mesh *m, const VkAccelerationStructureGeometryKHR *g,
                    VkBuildAccelerationStructureFlagsKHR fl, const VkAccelerationStructureBuildSizesInfoKHR *sz, bool wipe, bool brk) {
    struct St s = {OP_UPDATE, src, d, g, m->nt, fl, src->cur, sz->updateScratchSize, m, wipe, brk};
    return run_step(&s);
}
static int s_copy(enum Op op, struct V *src, struct V *d, VkDeviceSize size, const struct Mesh *m) {
    struct St s = {op, src, d, NULL, 0, 0, size, 0, m, true, false};
    return run_step(&s);
}
static VkQueryPool QP3;
static VkDeviceSize query_compacted(VkAccelerationStructureKHR as) {
    if (!pWriteProps) die(2, "vkCmdWriteAccelerationStructuresPropertiesKHR not exposed");
    if (!QP3) {
        VkQueryPoolCreateInfo qpi = {VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO, NULL, 0, VK_QUERY_TYPE_ACCELERATION_STRUCTURE_COMPACTED_SIZE_KHR, 1, 0};
        CK(vkCreateQueryPool(dev, &qpi, NULL, &QP3));
    }
    begin(cbs[0]);
    vkCmdResetQueryPool(cbs[0], QP3, 0, 1);
    fullbar(cbs[0]);
    pWriteProps(cbs[0], 1, &as, VK_QUERY_TYPE_ACCELERATION_STRUCTURE_COMPACTED_SIZE_KHR, QP3, 0);
    end(cbs[0]);
    submit_wait(cbs[0], "compacted-size query");
    uint64_t c = 0;
    CK(vkGetQueryPoolResults(dev, QP3, 0, 1, sizeof c, &c, sizeof c, VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT));
    return c;
}
static void rs_end(void) { if (QP3) { vkDestroyQueryPool(dev, QP3, NULL); QP3 = VK_NULL_HANDLE; } }
// Common verdict: FAIL names the first failing counted step.
static int rs_verdict(char *why, size_t cap, const char *ok, const char *extra) {
    say("summary steps=%d counted=%d bad_steps=%d slow_steps_over_250ms=%d ms_avg=%.2f ms_max=%.2f break_steps=%d break_bad=%d stopped=%d",
        RS.n, RS.counted, RS.bad, RS.slow, RS.counted ? RS.ms_sum / RS.counted : 0, RS.ms_max, RS.brk_n, RS.brk_bad, RS.stopped);
    rs_end();
    if (RS.bad) {
        snprintf(why, cap, "first failing %s; %d of %d counted steps bad%s%s", FIRSTFAIL, RS.bad, RS.counted,
                 RS.stopped ? " (chain stopped after 10 consecutive)" : "", extra);
        return 1;
    }
    snprintf(why, cap, "%s%s", ok, extra);
    return 0;
}

// ================================================================ arm: refitchain
// BLAS 2048 tris, FAST_TRACE|ALLOW_UPDATE (0x5, as the game), each handle = GENERIC view over its whole VkBuffer (driver size
// rounded to 256), 9 MB GPU-mapped guard tail per buffer. Vertices move between refits; every destination is traced.
static int arm_refitchain(int sub, char *why, size_t cap) {   // 0 inplace-after-other 1 pingpong-built 2 pingpong-built-wipe 3 other-into-built 4 rebuild-heals
    phase("refitchain-setup");
    const VkBuildAccelerationStructureFlagsKHR FL = FT | AU;
    struct Mesh M = mk_mesh(32, 32, -1.5f, -1.0f, 1.5f, 1.0f), M2 = mk_mesh(16, 8, -1.0f, -0.8f, 1.0f, 0.8f);
    rs_state(&M, 0);
    mesh_z(&M2, 2.5f, 0.1f, 1.0f);
    VkAccelerationStructureGeometryKHR g = mesh_geom(&M), g2 = mesh_geom(&M2);
    VkAccelerationStructureBuildSizesInfoKHR sz = sizes_of(BLASF, FL, &g, M.nt), sz2 = sizes_of(BLASF, FL, &g2, M2.nt);
    const VkDeviceSize drv = sz.accelerationStructureSize, vb = up(drv, 256);
    if (sz2.accelerationStructureSize > vb) die(2, "small geometry larger than the big one (%llu > %llu): layout invalid",
                                                (unsigned long long)sz2.accelerationStructureSize, (unsigned long long)vb);
    static struct Buf B[3];
    struct V v[3];
    static const char *const nm[3] = {"A", "B", "A'"};
    const int NH = sub == 4 ? 3 : 2;
    for (int k = 0; k < NH; k++) {
        B[k] = mkbuf_g(vb, 9 * MB);
        fill(&B[k], 0x5A);
        v[k] = (struct V){&B[k], 0, vb, 0, mkas(&B[k], 0, vb, GENF), nm[k]};
    }
    VkDeviceSize smax = sz.buildScratchSize > sz.updateScratchSize ? sz.buildScratchSize : sz.updateScratchSize;
    if (sz2.buildScratchSize > smax) smax = sz2.buildScratchSize;
    rs_init(smax);
    RS.drv = drv;
    say("blas tris=%u flags=0x%x accel_size=%llu build_scratch=%llu update_scratch=%llu | %d handles GENERIC whole VkBuffer created=%llu "
        "guard_tail=%llu | small geometry tris=%u accel_size=%llu | scratch guard=%llu", M.nt, FL, (unsigned long long)drv,
        (unsigned long long)sz.buildScratchSize, (unsigned long long)sz.updateScratchSize, NH, (unsigned long long)vb,
        (unsigned long long)(B[0].msize - B[0].size), M2.nt, (unsigned long long)sz2.accelerationStructureSize,
        (unsigned long long)(RS.SC.msize - RS.SC.size));
    struct V *A = &v[0], *Bv = &v[1], *A2 = &v[2];
    char extra[160] = "";
    switch (sub) {
    case 0:   // inplace-after-other: BUILD A; UPDATE A->B (B never built); 30 in-place UPDATEs of B
        s_build(A, &M, &g, FL, &sz, false);
        rs_state(&M, 1);
        s_update(A, Bv, &M, &g, FL, &sz, false, false);
        for (int k = 2; k <= 31 && !RS.stopped; k++) { rs_state(&M, k); s_update(Bv, Bv, &M, &g, FL, &sz, false, false); }
        return rs_verdict(why, cap, "BUILD A, UPDATE A->B (B never built), 30 in-place UPDATEs of B: all correct, nothing outside", "");
    case 1:
    case 2:   // pingpong-built(-wipe): both handles BUILD-made, then 60 refits A->B, B->A
        s_build(A, &M, &g, FL, &sz, false);
        s_build(Bv, &M, &g, FL, &sz, false);
        for (int k = 1; k <= 60 && !RS.stopped; k++) {
            rs_state(&M, k);
            if (k & 1) s_update(A, Bv, &M, &g, FL, &sz, sub == 2, false);
            else s_update(Bv, A, &M, &g, FL, &sz, sub == 2, false);
        }
        return rs_verdict(why, cap, sub == 2 ? "60 refits between two BUILD-made handles, source wiped before each trace: all correct"
                                             : "60 refits between two BUILD-made handles: all correct, nothing outside", "");
    case 3:   // other-into-built: B built with a smaller, different geometry, then UPDATE A->B, 10 in-place UPDATEs of B
        s_build(A, &M, &g, FL, &sz, false);
        s_build(Bv, &M2, &g2, FL, &sz2, false);
        rs_state(&M, 1);
        s_update(A, Bv, &M, &g, FL, &sz, false, false);
        for (int k = 2; k <= 11 && !RS.stopped; k++) { rs_state(&M, k); s_update(Bv, Bv, &M, &g, FL, &sz, false, false); }
        return rs_verdict(why, cap, "UPDATE A->B onto a B built with 256 other tris + 10 in-place UPDATEs of B: all correct", "");
    default: {   // rebuild-heals
        s_build(A, &M, &g, FL, &sz, false);
        rs_state(&M, 1);
        s_update(A, Bv, &M, &g, FL, &sz, false, true);
        rs_state(&M, 2);
        s_update(Bv, A2, &M, &g, FL, &sz, false, true);
        rs_state(&M, 3);
        s_update(A2, Bv, &M, &g, FL, &sz, false, true);
        const int brk_bad = RS.brk_bad;
        say("break phase done: %d of %d break steps bad (expected on the blob: B->A' and A'->B), slow_steps=%d", brk_bad, RS.brk_n, RS.slow);
        rs_state(&M, 4);
        s_build(A2, &M, &g, FL, &sz, false);
        s_build(Bv, &M, &g, FL, &sz, false);
        for (int k = 5; k <= 34 && !RS.stopped; k++) {
            rs_state(&M, k);
            s_update(A2, A2, &M, &g, FL, &sz, false, false);
            s_update(Bv, Bv, &M, &g, FL, &sz, false, false);
        }
        if (!RS.stopped) { rs_state(&M, 35); s_update(A, A, &M, &g, FL, &sz, false, false); }   // control: A was never broken
        snprintf(extra, sizeof extra, "; break phase %d of 3 steps bad (break_reproduced=%d)", brk_bad, brk_bad > 0);
        return rs_verdict(why, cap, "after the break phase, fresh BUILDs of A' and B + 30 in-place refits each + A control refit: all correct", extra);
    }
    }
}

// ================================================================ arm: copyupd
// Source S = view with EXACTLY the driver size (its own buffer, 1 MB guard), BUILD; CLONE / COMPACT into D (GENERIC view at
// 1 MB of an 8 MB buffer, created 7 MB, 9 MB guard); trace D (source wiped); 30 in-place UPDATEs of D; BUILD D; 10 more.
static int arm_copyupd(bool compact, char *why, size_t cap) {
    phase("copyupd-setup");
    if (!pCopy) die(2, "vkCmdCopyAccelerationStructureKHR not exposed");
    const VkBuildAccelerationStructureFlagsKHR FL = compact ? FT | AU | AC : FT | AU;
    struct Mesh M = mk_mesh(32, 32, -1.5f, -1.0f, 1.5f, 1.0f);
    rs_state(&M, 0);
    VkAccelerationStructureGeometryKHR g = mesh_geom(&M);
    VkAccelerationStructureBuildSizesInfoKHR sz = sizes_of(BLASF, FL, &g, M.nt);
    const VkDeviceSize drv = sz.accelerationStructureSize;
    if (drv > 7 * MB) die(2, "accel size %llu > 7 MB: layout invalid", (unsigned long long)drv);
    struct Buf S = mkbuf_g(up(drv, 256), MB), D = mkbuf_g(8 * MB, 9 * MB);
    fill(&S, 0x5A);
    fill(&D, 0x5A);
    struct V vs = {&S, 0, drv, 0, mkas(&S, 0, drv, GENF), "S"}, vd = {&D, MB, 7 * MB, 0, mkas(&D, MB, 7 * MB, GENF), "D"};
    rs_init(sz.buildScratchSize > sz.updateScratchSize ? sz.buildScratchSize : sz.updateScratchSize);
    RS.drv = drv;
    say("blas tris=%u flags=0x%x accel_size=%llu build_scratch=%llu update_scratch=%llu | S created=%llu (exact) off=0 | D off=%llu created=%llu "
        "buffer=8388608 guard_tail=%llu", M.nt, FL, (unsigned long long)drv, (unsigned long long)sz.buildScratchSize,
        (unsigned long long)sz.updateScratchSize, (unsigned long long)drv, (unsigned long long)MB, (unsigned long long)(7 * MB),
        (unsigned long long)(D.msize - D.size));
    s_build(&vs, &M, &g, FL, &sz, false);
    VkDeviceSize c = 0;
    if (compact) {
        c = query_compacted(vs.h);
        RS.compacted = c;
        say("compacted=%llu accel_size=%llu%s", (unsigned long long)c, (unsigned long long)drv, c == 0 || c > drv ? " COMPACTED_SIZE_ODD" : "");
        if (c == 0 || c > drv) die(1, "compacted size %llu is 0 or > accel size %llu", (unsigned long long)c, (unsigned long long)drv);
    }
    s_copy(compact ? OP_COMPACT : OP_CLONE, &vs, &vd, compact ? c : drv, &M);
    const int copy_bad = RS.bad;
    int upd1 = 0, upd2 = 0, bld = 0;
    for (int k = 1; k <= 30 && !RS.stopped; k++) { rs_state(&M, k); upd1 += s_update(&vd, &vd, &M, &g, FL, &sz, false, false); }
    if (!RS.stopped) { rs_state(&M, 31); bld = s_build(&vd, &M, &g, FL, &sz, false); }
    for (int k = 32; k <= 41 && !RS.stopped; k++) { rs_state(&M, k); upd2 += s_update(&vd, &vd, &M, &g, FL, &sz, false, false); }
    char extra[200];
    snprintf(extra, sizeof extra, "; copy=%s updates_after_copy_bad=%d/30 build_of_copy=%s updates_after_build_bad=%d/10", copy_bad ? "bad" : "ok",
             upd1, bld ? "bad" : "ok", upd2);
    return rs_verdict(why, cap, compact ? "COMPACT of an exact-size source + 30 in-place refits + BUILD + 10 refits: all correct"
                                        : "CLONE of an exact-size source + 30 in-place refits + BUILD + 10 refits: all correct", extra);
}

// ================================================================ arm: bufclone
// Can the layer clone / compact with a plain vkCmdCopyBuffer? A = rest-of-buffer view at 5701632 of an 8 MB buffer, BUILD
// (FAST_TRACE|ALLOW_UPDATE|ALLOW_COMPACTION); copy the compacted size (pass 1) / the accel size (pass 2) to 6357760 of another
// buffer (offset mod 4096 and mod 65536 = 768, source 0); new GENERIC view H there (never built); wipe A; trace H; 10 in-place
// UPDATEs of H; BUILD H.
static int arm_bufclone(char *why, size_t cap) {
    phase("bufclone-setup");
    XUSE = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    const VkBuildAccelerationStructureFlagsKHR FL = FT | AU | AC;
    const VkDeviceSize SOFF = 5701632, DOFF = 6348288 + 256 * 37, SREST = 8 * MB - SOFF, DREST = 8 * MB - DOFF;
    struct Mesh M = mk_mesh(32, 32, -1.5f, -1.0f, 1.5f, 1.0f);
    rs_state(&M, 0);
    VkAccelerationStructureGeometryKHR g = mesh_geom(&M);
    VkAccelerationStructureBuildSizesInfoKHR sz = sizes_of(BLASF, FL, &g, M.nt);
    const VkDeviceSize drv = sz.accelerationStructureSize;
    if (drv > DREST) die(2, "accel size %llu > destination rest %llu: layout invalid", (unsigned long long)drv, (unsigned long long)DREST);
    struct Buf S = mkbuf_g(8 * MB, MB), D = mkbuf_g(8 * MB, 9 * MB);
    rs_init(sz.buildScratchSize > sz.updateScratchSize ? sz.buildScratchSize : sz.updateScratchSize);
    RS.drv = drv;
    struct V va = {&S, SOFF, SREST, 0, mkas(&S, SOFF, SREST, GENF), "A"};
    say("blas tris=%u flags=0x%x accel_size=%llu build_scratch=%llu update_scratch=%llu | A off=%llu created=%llu (rest) | H off=%llu created=%llu "
        "(rest) mod4096=%llu mod65536=%llu guard_tail=%llu", M.nt, FL, (unsigned long long)drv, (unsigned long long)sz.buildScratchSize,
        (unsigned long long)sz.updateScratchSize, (unsigned long long)SOFF, (unsigned long long)SREST, (unsigned long long)DOFF,
        (unsigned long long)DREST, (unsigned long long)(DOFF % 4096), (unsigned long long)(DOFF % 65536), (unsigned long long)(D.msize - D.size));
    char res[2][120];
    for (int pass = 0; pass < 2; pass++) {
        phase(pass ? "bufclone-pass-accel" : "bufclone-pass-compacted");
        fill(&S, 0x5A);
        fill(&D, 0x5A);
        rs_state(&M, 0);
        const int b0 = RS.bad;
        int bb = s_build(&va, &M, &g, FL, &sz, false);
        const VkDeviceSize c = query_compacted(va.h);
        RS.compacted = c;
        const VkDeviceSize nb = pass ? drv : c;
        say("pass=%s compacted=%llu accel_size=%llu copy_bytes=%llu", pass ? "accel" : "compacted", (unsigned long long)c, (unsigned long long)drv,
            (unsigned long long)nb);
        if (c == 0 || c > drv) die(1, "compacted size %llu is 0 or > accel size %llu", (unsigned long long)c, (unsigned long long)drv);
        struct V vh = {&D, DOFF, DREST, 0, mkas(&D, DOFF, DREST, GENF), "H"};
        struct St cp = {OP_BUFCOPY, &va, &vh, NULL, 0, 0, nb, 0, &M, true, false};
        int cb = run_step(&cp);
        int ub = 0, hb = 0;
        for (int k = 1; k <= 10 && !RS.stopped; k++) { rs_state(&M, k); ub += s_update(&vh, &vh, &M, &g, FL, &sz, false, false); }
        if (!RS.stopped) { rs_state(&M, 11); hb = s_build(&vh, &M, &g, FL, &sz, false); }
        snprintf(res[pass], sizeof res[pass], "%s(%llu B): build=%s copy_trace=%s updates_bad=%d/10 build_H=%s", pass ? "accel" : "compacted",
                 (unsigned long long)nb, bb ? "bad" : "ok", cb ? "bad" : "ok", ub, hb ? "bad" : "ok");
        say("pass summary %s (bad steps this pass %d)", res[pass], RS.bad - b0);
        rmas(vh.h);
        if (RS.stopped) { snprintf(res[1], sizeof res[1], "not run (stopped)"); break; }
    }
    char extra[260];
    snprintf(extra, sizeof extra, "; %s | %s", res[0], res[1]);
    return rs_verdict(why, cap, "vkCmdCopyBuffer of a built AS to another offset is a valid AS (rays), refittable and rebuildable", extra);
}

// ================================================================ arm: copyext
// Written extent of the blob's own CLONE / COMPACT in layouts the layer could choose. Source BLAS FAST_TRACE|ALLOW_COMPACTION,
// 10000 tris then 100 tris (two passes, as copy). S = 8 MB buffer (1 MB guard), D = 8 MB buffer (9 MB guard).
//  clone-exactsrc  : source view = offset 0, created exactly accel size; dst = rest-of-buffer view at 6 MB (2 MB left).
//  compact-restdst : same source; COMPACT into the 6 MB rest-of-buffer view.
//  compact-exactdst: same source; dst view created exactly compacted rounded up to 256, ending at 4 MB (pattern after it).
//  clone-alias     : BUILD through H1 = rest-of-buffer view at 5701632; H2 = same buffer + offset, created exactly accel size,
//                    never built; CLONE with src = H2 into the 6 MB rest-of-buffer view.
static int arm_copyext(int sub, char *why, size_t cap) {
    phase("copyext-setup");
    if (!pCopy) die(2, "vkCmdCopyAccelerationStructureKHR not exposed");
    const bool compact = sub == 1 || sub == 2;
    const VkBuildAccelerationStructureFlagsKHR FL = FT | AC;
    struct Buf S = mkbuf_g(8 * MB, MB), D = mkbuf_g(8 * MB, 9 * MB);
    static const uint32_t NX[2] = {100, 10}, NY[2] = {50, 5};
    VkDeviceSize smax = 0;
    struct Mesh Ms[2];
    VkAccelerationStructureGeometryKHR gs[2];
    VkAccelerationStructureBuildSizesInfoKHR szs[2];
    for (int p = 0; p < 2; p++) {
        Ms[p] = mk_mesh(NX[p], NY[p], -2, -1, 2, 1);
        mesh_z(&Ms[p], 3.0f, 0.2f, 0.3f + p);
        gs[p] = mesh_geom(&Ms[p]);
        szs[p] = sizes_of(BLASF, FL, &gs[p], Ms[p].nt);
        if (szs[p].buildScratchSize > smax) smax = szs[p].buildScratchSize;
    }
    rs_init(smax);
    char res[2][160];
    for (int pass = 0; pass < 2; pass++) {
        phase(pass ? "copyext-pass-100" : "copyext-pass-10000");
        const VkDeviceSize drv = szs[pass].accelerationStructureSize;
        if (drv > 2 * MB) die(2, "tris=%u: accel size %llu > 2 MB destination room: layout invalid", Ms[pass].nt, (unsigned long long)drv);
        fill(&S, 0x5A);
        fill(&D, 0x5A);
        RS.drv = drv;
        RS.compacted = 0;
        RS.xval = 0;
        const VkDeviceSize SOFF = sub == 3 ? 5701632 : 0;
        struct V vb = {&S, SOFF, sub == 3 ? 8 * MB - SOFF : drv, 0, VK_NULL_HANDLE, sub == 3 ? "H1" : "S"};
        vb.h = mkas(&S, vb.off, vb.create, GENF);
        struct V va = vb;
        if (sub == 3) { va.create = drv; va.h = mkas(&S, SOFF, drv, GENF); va.name = "H2"; RS.xname = "h1_create"; RS.xval = vb.create; }
        s_build(&vb, &Ms[pass], &gs[pass], FL, &szs[pass], false);
        va.cur = drv;
        const VkDeviceSize c = query_compacted(vb.h);
        RS.compacted = c;
        say("tris=%u accel_size=%llu compacted=%llu build_scratch=%llu src %s off=%llu created=%llu%s%s", Ms[pass].nt, (unsigned long long)drv,
            (unsigned long long)c, (unsigned long long)szs[pass].buildScratchSize, va.name, (unsigned long long)va.off, (unsigned long long)va.create,
            sub == 3 ? " (built through H1 created=" : "", sub == 3 ? "rest of buffer)" : "");
        if (compact && (c == 0 || c > drv)) die(1, "compacted size %llu is 0 or > accel size %llu", (unsigned long long)c, (unsigned long long)drv);
        VkDeviceSize dcreate = 2 * MB, doff = 6 * MB;
        if (sub == 2) { dcreate = up(c, 256); doff = 4 * MB - dcreate; }
        struct V vd = {&D, doff, dcreate, 0, mkas(&D, doff, dcreate, GENF), "D"};
        const int b0 = RS.bad;
        s_copy(compact ? OP_COMPACT : OP_CLONE, &va, &vd, compact ? c : drv, &Ms[pass]);
        snprintf(res[pass], sizeof res[pass], "tris=%u %s=%llu %s", Ms[pass].nt, compact ? "compacted" : "accel_size",
                 (unsigned long long)(compact ? c : drv), RS.bad > b0 ? "bad" : "ok");
        rmas(vd.h);
        if (sub == 3) rmas(va.h);
        rmas(vb.h);
    }
    char extra[340];
    snprintf(extra, sizeof extra, "; %s | %s", res[0], res[1]);
    static const char *const okm[4] = {"CLONE from an exact-size source stays inside [dst, dst+accel_size), copy traced correctly",
                                       "COMPACT into a rest-of-buffer view stays inside [dst, dst+compacted), copy traced correctly",
                                       "COMPACT into an exact (compacted, 256-rounded) view stays inside it, copy traced correctly",
                                       "CLONE with an exact-size alias handle (never built) as source: inside [dst, dst+accel_size), copy traced correctly"};
    return rs_verdict(why, cap, okm[sub], extra);
}

// ================================================================ arm: aliasupd
// BUILD through H1 (rest-of-buffer view at 5701632 of an 8 MB buffer, 9 MB guard); 10 in-place UPDATEs through H2 (same
// buffer + offset, created exactly the accel size, never built); then 10 in-place UPDATEs through H1 again.
static int arm_aliasupd(char *why, size_t cap) {
    phase("aliasupd-setup");
    const VkBuildAccelerationStructureFlagsKHR FL = FT | AU;
    struct Mesh M = mk_mesh(32, 32, -1.5f, -1.0f, 1.5f, 1.0f);
    rs_state(&M, 0);
    VkAccelerationStructureGeometryKHR g = mesh_geom(&M);
    VkAccelerationStructureBuildSizesInfoKHR sz = sizes_of(BLASF, FL, &g, M.nt);
    const VkDeviceSize drv = sz.accelerationStructureSize, SOFF = 5701632, SREST = 8 * MB - SOFF;
    if (drv > SREST) die(2, "accel size %llu > rest %llu: layout invalid", (unsigned long long)drv, (unsigned long long)SREST);
    struct Buf S = mkbuf_g(8 * MB, 9 * MB);
    fill(&S, 0x5A);
    rs_init(sz.buildScratchSize > sz.updateScratchSize ? sz.buildScratchSize : sz.updateScratchSize);
    RS.drv = drv;
    struct V h1 = {&S, SOFF, SREST, 0, mkas(&S, SOFF, SREST, GENF), "H1"}, h2 = {&S, SOFF, drv, 0, mkas(&S, SOFF, drv, GENF), "H2"};
    say("blas tris=%u flags=0x%x accel_size=%llu build_scratch=%llu update_scratch=%llu | H1 off=%llu created=%llu (rest) | H2 same offset "
        "created=%llu (exact, never built) | same_address=%d guard_tail=%llu", M.nt, FL, (unsigned long long)drv, (unsigned long long)sz.buildScratchSize,
        (unsigned long long)sz.updateScratchSize, (unsigned long long)SOFF, (unsigned long long)SREST, (unsigned long long)drv,
        as_addr(h1.h) == as_addr(h2.h), (unsigned long long)(S.msize - S.size));
    s_build(&h1, &M, &g, FL, &sz, false);
    h2.cur = drv;
    int u2 = 0, u1 = 0;
    for (int k = 1; k <= 10 && !RS.stopped; k++) { rs_state(&M, k); u2 += s_update(&h2, &h2, &M, &g, FL, &sz, false, false); }
    for (int k = 11; k <= 20 && !RS.stopped; k++) { rs_state(&M, k); u1 += s_update(&h1, &h1, &M, &g, FL, &sz, false, false); }
    char extra[120];
    snprintf(extra, sizeof extra, "; updates_via_H2_bad=%d/10 updates_via_H1_after_bad=%d/10", u2, u1);
    return rs_verdict(why, cap, "BUILD via H1, 10 in-place refits via the never-built exact alias H2, 10 via H1 again: all correct", extra);
}

// ================================================================ main
int main(int argc, char **argv) {
    const char *arm = argc > 1 ? argv[1] : "";
    const char *sub = argc > 2 ? argv[2] : "";
    bool lab3 = argc > 3 && (!strcmp(arm, "overlap") || !strcmp(arm, "copy"));
    snprintf(LABEL, sizeof LABEL, "%s%s%s%s%s", arm, *sub ? ":" : "", sub, lab3 ? ":" : "", lab3 ? argv[3] : "");
    struct sigaction sa;
    memset(&sa, 0, sizeof sa);
    sa.sa_handler = on_alarm;
    sigaction(SIGALRM, &sa, NULL);
    sa.sa_handler = on_crash;
    sigaction(SIGSEGV, &sa, NULL);
    sigaction(SIGBUS, &sa, NULL);
    sigaction(SIGILL, &sa, NULL);
    sigaction(SIGFPE, &sa, NULL);
    sigaction(SIGABRT, &sa, NULL);
    setvbuf(stdout, NULL, _IOLBF, 0);
    int which = !strcmp(arm, "sizes") ? 1 : !strcmp(arm, "extent") ? 2 : !strcmp(arm, "overlap") ? 3 : !strcmp(arm, "update") ? 4
              : !strcmp(arm, "mutate") ? 5 : !strcmp(arm, "race") ? 6 : !strcmp(arm, "copy") ? 7 : !strcmp(arm, "updsrc") ? 8
              : !strcmp(arm, "formats") ? 9 : !strcmp(arm, "vfmt") ? 10 : !strcmp(arm, "multigeom") ? 11 : !strcmp(arm, "refitchain") ? 12
              : !strcmp(arm, "copyupd") ? 13 : !strcmp(arm, "bufclone") ? 14 : !strcmp(arm, "copyext") ? 15 : !strcmp(arm, "aliasupd") ? 16 : 0;
    int subi = 0;
    if (which == 3) subi = !strcmp(sub, "batch") ? 0 : !strcmp(sub, "serial") ? 1 : -1;
    if (which == 4) subi = !*sub || !strcmp(sub, "single") ? 0 : !strcmp(sub, "batch") ? 1 : !strcmp(sub, "tlasupd") ? 2 : -1;
    if (which == 5) subi = !*sub || !strcmp(sub, "generic") ? 0 : !strcmp(sub, "recreate") ? 1 : -1;
    if (which == 6) subi = !strcmp(sub, "nobarrier") ? 0 : !strcmp(sub, "barrier") ? 1 : !strcmp(sub, "twocmd") ? 2 : !strcmp(sub, "twocmdnb") ? 3 : -1;
    bool blas_type = which == 3 && argc > 3 && !strcmp(argv[3], "blas");
    if (which == 3 && argc > 3 && !blas_type) subi = -1;
    if (which == 7) subi = !strcmp(sub, "clone") ? 0 : !strcmp(sub, "compact") ? 1 : -1;
    if (which == 8) subi = !*sub ? 0 : !strcmp(sub, "big") ? 1 : !strcmp(sub, "bigdst") ? 2 : -1;
    bool copy_big = which == 7 && argc > 3 && !strcmp(argv[3], "big");
    if (which == 7 && argc > 3 && !copy_big) subi = -1;
    const struct VFmt *vf = which == 10 ? vfmt_by_name(sub) : NULL;
    if (which == 10 && !vf) subi = -1;
    static const char *const RCN[5] = {"inplace-after-other", "pingpong-built", "pingpong-built-wipe", "other-into-built", "rebuild-heals"};
    static const char *const CEN[4] = {"clone-exactsrc", "compact-restdst", "compact-exactdst", "clone-alias"};
    if (which == 12) { subi = -1; for (int i = 0; i < 5; i++) if (!strcmp(sub, RCN[i])) subi = i; }
    if (which == 13) subi = !strcmp(sub, "clone") ? 0 : !strcmp(sub, "compact") ? 1 : -1;
    if (which == 15) { subi = -1; for (int i = 0; i < 4; i++) if (!strcmp(sub, CEN[i])) subi = i; }
    if ((which == 14 || which == 16) && *sub) subi = -1;
    if (!which || subi < 0) {
        printf("usage: as-probe sizes | extent | overlap batch|serial [blas] | update single|batch|tlasupd | mutate generic|recreate |\n"
               "                race nobarrier|barrier|twocmd|twocmdnb [grid=1024] [iters=300] |\n"
               "                copy clone|compact [big] | updsrc [big|bigdst] | formats | vfmt <format> | multigeom |\n"
               "                refitchain inplace-after-other|pingpong-built|pingpong-built-wipe|other-into-built|rebuild-heals |\n"
               "                copyupd clone|compact | bufclone | copyext clone-exactsrc|compact-restdst|compact-exactdst|clone-alias | aliasupd\n"
               "  vfmt formats:");
        for (int i = 0; i < NVFMT; i++) printf(" %s", VFMTS[i].name);
        printf("\n");
        die(2, "bad arguments");
    }
    phase("instance");
    VkApplicationInfo app = {VK_STRUCTURE_TYPE_APPLICATION_INFO, NULL, "as-probe", 1, NULL, 0, VK_API_VERSION_1_2};
    VkInstanceCreateInfo ici = {VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, NULL, 0, &app, 0, NULL, 0, NULL};
    VkInstance inst;
    CK(vkCreateInstance(&ici, NULL, &inst));
    uint32_t npd = 1;
    VkResult er = vkEnumeratePhysicalDevices(inst, &npd, &pd);
    if ((er != VK_SUCCESS && er != VK_INCOMPLETE) || !npd) die(3, "no physical device (%d)", er);
    VkPhysicalDeviceDriverProperties drv = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES};
    asp.pNext = &drv;
    VkPhysicalDeviceProperties2 p2 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, &asp};
    vkGetPhysicalDeviceProperties2(pd, &p2);
    say("device \"%s\" api %u.%u.%u driver \"%s\" \"%s\" driverVersion 0x%x minScratchAlign %u maxPrimitiveCount %llu maxInstanceCount %llu",
        p2.properties.deviceName, VK_API_VERSION_MAJOR(p2.properties.apiVersion), VK_API_VERSION_MINOR(p2.properties.apiVersion),
        VK_API_VERSION_PATCH(p2.properties.apiVersion), drv.driverName, drv.driverInfo, p2.properties.driverVersion,
        asp.minAccelerationStructureScratchOffsetAlignment, (unsigned long long)asp.maxPrimitiveCount, (unsigned long long)asp.maxInstanceCount);
    SA = asp.minAccelerationStructureScratchOffsetAlignment ? asp.minAccelerationStructureScratchOffsetAlignment : 1;
    VkPhysicalDeviceRayQueryFeaturesKHR rqf = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_QUERY_FEATURES_KHR};
    VkPhysicalDeviceAccelerationStructureFeaturesKHR asf = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR, &rqf};
    VkPhysicalDeviceVulkan12Features v12 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES, &asf};
    VkPhysicalDeviceFeatures2 f2 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, &v12};
    vkGetPhysicalDeviceFeatures2(pd, &f2);
    uint32_t ne = 0;
    vkEnumerateDeviceExtensionProperties(pd, NULL, &ne, NULL);
    VkExtensionProperties *ext = calloc(ne ? ne : 1, sizeof *ext);
    vkEnumerateDeviceExtensionProperties(pd, NULL, &ne, ext);
    bool has_as = false, has_rq = false, has_dho = false;
    for (uint32_t i = 0; i < ne; i++) {
        has_as |= !strcmp(ext[i].extensionName, "VK_KHR_acceleration_structure");
        has_rq |= !strcmp(ext[i].extensionName, "VK_KHR_ray_query");
        has_dho |= !strcmp(ext[i].extensionName, "VK_KHR_deferred_host_operations");
    }
    if (!has_as || !has_rq || !has_dho || !asf.accelerationStructure || !rqf.rayQuery || !v12.bufferDeviceAddress)
        die(2, "missing: as_ext %d rq_ext %d dho_ext %d asFeat %u rqFeat %u bda %u", has_as, has_rq, has_dho, asf.accelerationStructure,
            rqf.rayQuery, v12.bufferDeviceAddress);
    uint32_t nq = 0, qf = UINT32_MAX;
    vkGetPhysicalDeviceQueueFamilyProperties(pd, &nq, NULL);
    VkQueueFamilyProperties *qp = calloc(nq, sizeof *qp);
    vkGetPhysicalDeviceQueueFamilyProperties(pd, &nq, qp);
    for (uint32_t i = 0; i < nq && qf == UINT32_MAX; i++) if (qp[i].queueFlags & VK_QUEUE_COMPUTE_BIT) qf = i;
    if (qf == UINT32_MAX) die(2, "no compute queue");
    float prio = 1.0f;
    VkDeviceQueueCreateInfo qci = {VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO, NULL, 0, qf, 1, &prio};
    VkPhysicalDeviceRayQueryFeaturesKHR en_rq = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_QUERY_FEATURES_KHR, NULL, VK_TRUE};
    VkPhysicalDeviceAccelerationStructureFeaturesKHR en_as = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR, &en_rq, VK_TRUE};
    VkPhysicalDeviceVulkan12Features en_12 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES, &en_as};
    en_12.bufferDeviceAddress = VK_TRUE;
    const char *dexts[] = {"VK_KHR_acceleration_structure", "VK_KHR_ray_query", "VK_KHR_deferred_host_operations"};
    VkDeviceCreateInfo dci = {VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, &en_12, 0, 1, &qci, 0, NULL, 3, dexts, NULL};
    phase("device");
    CK(vkCreateDevice(pd, &dci, NULL, &dev));
    vkGetPhysicalDeviceMemoryProperties(pd, &memp);
#define LD(var, name) do { var = (void *)vkGetDeviceProcAddr(dev, name); if (!var) die(3, "no entry point %s", name); } while (0)
    LD(pBDA, "vkGetBufferDeviceAddress");
    LD(pSizes, "vkGetAccelerationStructureBuildSizesKHR");
    LD(pCreateAS, "vkCreateAccelerationStructureKHR");
    LD(pDestroyAS, "vkDestroyAccelerationStructureKHR");
    LD(pBuild, "vkCmdBuildAccelerationStructuresKHR");
    LD(pASAddr, "vkGetAccelerationStructureDeviceAddressKHR");
    pCopy = (PFN_vkCmdCopyAccelerationStructureKHR)vkGetDeviceProcAddr(dev, "vkCmdCopyAccelerationStructureKHR");
    pWriteProps = (PFN_vkCmdWriteAccelerationStructuresPropertiesKHR)vkGetDeviceProcAddr(dev, "vkCmdWriteAccelerationStructuresPropertiesKHR");
    vkGetDeviceQueue(dev, qf, 0, &queue);
    VkCommandPoolCreateInfo cpi = {VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO, NULL, VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT, qf};
    CK(vkCreateCommandPool(dev, &cpi, NULL, &pool));
    VkCommandBufferAllocateInfo cai = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, NULL, pool, VK_COMMAND_BUFFER_LEVEL_PRIMARY, 2};
    CK(vkAllocateCommandBuffers(dev, &cai, cbs));
    VkFenceCreateInfo fci = {VK_STRUCTURE_TYPE_FENCE_CREATE_INFO, NULL, 0};
    CK(vkCreateFence(dev, &fci, NULL, &fence));

    VkDescriptorSetLayout dsl = VK_NULL_HANDLE;
    VkShaderModule sm = VK_NULL_HANDLE;
    VkDescriptorPool dp = VK_NULL_HANDLE;
    if (which != 1 && which != 9) {
        phase("pipeline");
        double t0 = now_ms();
        VkDescriptorSetLayoutBinding bnd[3] = {
            {0, VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL},
            {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL},
            {2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL}};
        VkDescriptorSetLayoutCreateInfo dli = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, NULL, 0, 3, bnd};
        CK(vkCreateDescriptorSetLayout(dev, &dli, NULL, &dsl));
        VkPushConstantRange pcr = {VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(struct PC)};
        VkPipelineLayoutCreateInfo pli = {VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, NULL, 0, 1, &dsl, 1, &pcr};
        CK(vkCreatePipelineLayout(dev, &pli, NULL, &pl));
        VkShaderModuleCreateInfo smi = {VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, NULL, 0, sizeof as_probe_spv, as_probe_spv};
        CK(vkCreateShaderModule(dev, &smi, NULL, &sm));
        VkComputePipelineCreateInfo cpci = {VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO, NULL, 0,
            {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, NULL, 0, VK_SHADER_STAGE_COMPUTE_BIT, sm, "main", NULL}, pl, VK_NULL_HANDLE, 0};
        CK(vkCreateComputePipelines(dev, VK_NULL_HANDLE, 1, &cpci, NULL, &rqpipe));
        VkDescriptorPoolSize ps[2] = {{VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 1}, {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2}};
        VkDescriptorPoolCreateInfo dpi = {VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO, NULL, 0, 1, 2, ps};
        CK(vkCreateDescriptorPool(dev, &dpi, NULL, &dp));
        VkDescriptorSetAllocateInfo dai = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, NULL, dp, 1, &dsl};
        CK(vkAllocateDescriptorSets(dev, &dai, &ds));
        RB = mkbuf(16 * MAXRAYS);
        OB = mkbuf(16 + 16 * MAXRAYS);
        VkDescriptorBufferInfo rbi = {RB.b, 0, VK_WHOLE_SIZE}, obi = {OB.b, 0, VK_WHOLE_SIZE};
        VkWriteDescriptorSet w[2] = {
            {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, NULL, ds, 1, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, NULL, &rbi, NULL},
            {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, NULL, ds, 2, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, NULL, &obi, NULL}};
        vkUpdateDescriptorSets(dev, 2, w, 0, NULL);
        say("time pipeline ms=%.1f", now_ms() - t0);
    }

    char why[300] = "";
    double t0 = now_ms();
    int code = which == 1 ? arm_sizes(why, sizeof why)
             : which == 2 ? arm_extent(why, sizeof why)
             : which == 3 ? arm_overlap(subi == 0, blas_type, why, sizeof why)
             : which == 4 ? arm_update(subi, why, sizeof why)
             : which == 5 ? arm_mutate(subi == 1, why, sizeof why)
             : which == 7 ? arm_copy(subi == 1, copy_big, why, sizeof why)
             : which == 8 ? (subi ? arm_updsrc_big(subi == 2, why, sizeof why) : arm_updsrc(why, sizeof why))
             : which == 9 ? arm_formats(why, sizeof why)
             : which == 10 ? arm_vfmt(vf, why, sizeof why)
             : which == 11 ? arm_multigeom(why, sizeof why)
             : which == 12 ? arm_refitchain(subi, why, sizeof why)
             : which == 13 ? arm_copyupd(subi == 1, why, sizeof why)
             : which == 14 ? arm_bufclone(why, sizeof why)
             : which == 15 ? arm_copyext(subi, why, sizeof why)
             : which == 16 ? arm_aliasupd(why, sizeof why)
             : arm_race(subi, argc > 3 ? (uint32_t)atoi(argv[3]) : 1024, argc > 4 ? atoi(argv[4]) : 300, why, sizeof why);
    double t_arm = now_ms() - t0;
    phase("teardown");
    char final_msg[160];
    snprintf(final_msg, sizeof final_msg, "AS-PROBE %s TIMEOUT watchdog 60 s in teardown (arm verdict was %s)\n", LABEL, VERD[code]);
    memcpy(wd_msg, final_msg, sizeof final_msg);
    CK(vkDeviceWaitIdle(dev));
    while (nases) rmas(ases[nases - 1]);
    for (int i = 0; i < nbufs; i++) { vkDestroyBuffer(dev, bufs[i].b, NULL); vkFreeMemory(dev, bufs[i].m, NULL); }
    if (rqpipe) vkDestroyPipeline(dev, rqpipe, NULL);
    if (sm) vkDestroyShaderModule(dev, sm, NULL);
    if (dp) vkDestroyDescriptorPool(dev, dp, NULL);
    if (pl) vkDestroyPipelineLayout(dev, pl, NULL);
    if (dsl) vkDestroyDescriptorSetLayout(dev, dsl, NULL);
    vkDestroyFence(dev, fence, NULL);
    vkDestroyCommandPool(dev, pool, NULL);
    vkDestroyDevice(dev, NULL);
    vkDestroyInstance(inst, NULL);
    alarm(0);
    size_t l = strlen(why);
    snprintf(why + l, sizeof why - l, " (arm %.0f ms)", t_arm);
    die(code, "%s", why);
}

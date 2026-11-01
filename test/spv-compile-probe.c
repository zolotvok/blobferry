// spv-compile-probe: does the vendor shader compiler take this compute SPIR-V? Control (D3D12 through vkd3d-proton)
// dies at start because the LLVM inside vulkan.mali.so aborts on one compute shader ("LLVM ERROR: Cannot select ...
// #405 ... i26", docs/games/control.md § Diagnosis plan, R1/C8/C9). For each .spv a forked child creates the instance
// and the device, builds the descriptor set layouts from the module's own DescriptorSet/Binding decorations and
// creates the compute pipeline. An abort inside the compiler kills only the child and is reported as a result:
//   RESULT <file> compiled | aborted (signal or exit inside the compile) | vkerror <VkResult> | setup <why> | timeout
// with the compile time; the child's stderr (where LLVM writes its error) is printed above the RESULT line.
// The device: every feature the driver reports on (Vulkan 1.1/1.2/1.3 + the extension feature structs below), the
// vkd3d-relevant extensions (-x: every reported extension except the WSI ones; -b: robustBufferAccess(2) and
// robustImageAccess off, as vkd3d-proton runs). The layout, in vkd3d-proton's shape:
//   - bindings that alias several descriptor types (vkd3d's set 1 binding 2: 2D/Cube/texel-buffer/storage images and
//     the BindlessCBV arrays) become one VK_EXT_mutable_descriptor_type binding with vkd3d's type list
//     (SAMPLED_IMAGE, UNIFORM_TEXEL_BUFFER, STORAGE_IMAGE, STORAGE_TEXEL_BUFFER, UNIFORM_BUFFER, STORAGE_BUFFER; -m used:
//     only the aliased types);
//   - runtime arrays become partially bound update-after-bind bindings (the highest binding of a set also with a
//     variable count), count min(vkd3d's 1000000 / 2048 samplers, the driver's update-after-bind limits) or -n;
//   - -P SETS makes those sets push-descriptor sets (vkd3d's root CBVs with MALI_COMPAT_VENDOR_ID=5143: -P 2);
//   - the push-constant range is the size of the PushConstant block (RootConstants), or -c BYTES; stages ALL.
// -l LAYOUT.spv takes the layout from another module (union with the compiled one): spirv-reduce candidates are then
// compiled against the original shader's layout, so removing a variable does not change the layout under test.
// Usage: spv-compile-probe [-l LAYOUT.spv] [-P SET[,SET]] [-m vkd3d|used] [-n COUNT] [-c BYTES] [-x] [-b] [-e ENTRY]
//                          [-t SECONDS (default 60)] FILE.spv...
// Exit: 0 compiled, 1 the compile did not return (signal/exit inside vkCreateComputePipelines), 2 setup error,
// 3 vkCreateComputePipelines returned an error, 4 timeout; with several files the highest code.
// Run on the ICD alone: VK_DRIVER_FILES=<bridge icd.json> HYBRIS_USE_VENDOR_NAMESPACE=1 VK_LOADER_LAYERS_DISABLE=~implicit~
// Through the compat layer: VK_DRIVER_FILES=<bridge icd.json> HYBRIS_USE_VENDOR_NAMESPACE=1 ENABLE_MALI_COMPAT=1
// [MALI_COMPAT_SPV_FIX=1 ...]. Interestingness wrapper for spirv-reduce: test/spv-reduce-interesting.sh.
// [Claude Opus 5.5] 2026-10-30
#define _GNU_SOURCE
#include <errno.h>
#include <poll.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/resource.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>
#include <vulkan/vulkan.h>

// ---- options
static const char *opt_layout, *opt_entry;
static uint32_t opt_push_sets, opt_count, opt_pc_bytes;
static int opt_mutable_used, opt_all_ext, opt_no_robust, opt_timeout = 60;

static uint64_t now_ns(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (uint64_t)t.tv_sec * 1000000000ull + (uint64_t)t.tv_nsec;
}

// ---- SPIR-V reflection
struct spv {
    uint32_t *w;      // the module
    size_t n;         // words
    uint32_t bound;
    uint32_t *def;    // word index of the instruction that defines an id (types, constants, variables)
    uint32_t *set, *bind, *stride;
    uint8_t *has_set, *has_bind, *bufblock;
    uint32_t entry_id, local[3];
    char entry[64];
};

#define MAXB 128
struct binding {
    uint32_t set, binding, count, ntypes;
    int runtime;
    VkDescriptorType types[8];
};
static struct binding binds[MAXB];
static uint32_t nbinds, pc_size;
static int layout_done;   // -l: the layout module is reflected; additions from the compiled module get a note

static const uint32_t *ins(const struct spv *s, uint32_t id) {
    return id < s->bound && s->def[id] ? &s->w[s->def[id]] : NULL;
}

static uint32_t const_value(const struct spv *s, uint32_t id) {
    const uint32_t *c = ins(s, id);   // OpConstant / OpSpecConstant: type, id, value
    return c && ((c[0] & 0xffff) == 43 || (c[0] & 0xffff) == 50) ? c[3] : 1;
}

static uint32_t member_offset(const struct spv *s, uint32_t st, uint32_t m) {
    for (size_t i = 5; i < s->n; i += s->w[i] >> 16) {
        const uint32_t *o = &s->w[i];
        if ((o[0] & 0xffff) == 72 && o[1] == st && o[2] == m && o[3] == 35) return o[4];   // OpMemberDecorate Offset
    }
    return 0;
}

static uint32_t type_size(const struct spv *s, uint32_t ty, int depth) {
    const uint32_t *d = ins(s, ty);
    if (!d || depth > 16) return 0;
    switch (d[0] & 0xffff) {
    case 20: return 4;                                              // bool
    case 21: case 22: return d[2] / 8;                              // int, float
    case 23: case 24: return d[3] * type_size(s, d[2], depth + 1);  // vector, matrix (column count x column)
    case 28: {                                                      // array
        uint32_t st = s->stride[ty] ? s->stride[ty] : type_size(s, d[2], depth + 1);
        return const_value(s, d[3]) * st;
    }
    case 30: {                                                      // struct: end of the last member
        uint32_t end = 0, members = (d[0] >> 16) - 2;
        for (uint32_t m = 0; m < members; m++) {
            uint32_t e = member_offset(s, ty, m) + type_size(s, d[2 + m], depth + 1);
            if (e > end) end = e;
        }
        return end;
    }
    case 32: return 8;                                              // physical storage buffer pointer
    default: return 0;
    }
}

static int spv_load(const char *path, struct spv *s) {
    memset(s, 0, sizeof *s);
    FILE *f = fopen(path, "rb");
    if (!f) { printf("setup: cannot open %s: %s\n", path, strerror(errno)); return -1; }
    fseek(f, 0, SEEK_END);
    long len = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (len < 20 || len % 4) { printf("setup: %s: %ld bytes is not a SPIR-V module\n", path, len); fclose(f); return -1; }
    s->w = malloc(len);
    s->n = len / 4;
    if (fread(s->w, 4, s->n, f) != s->n) { printf("setup: %s: short read\n", path); fclose(f); return -1; }
    fclose(f);
    if (s->w[0] != 0x07230203) { printf("setup: %s: bad SPIR-V magic 0x%08x\n", path, s->w[0]); return -1; }
    s->bound = s->w[3];
    if (!s->bound || s->bound > (1u << 24)) { printf("setup: %s: bad id bound %u\n", path, s->bound); return -1; }
    s->def = calloc(s->bound, 4); s->set = calloc(s->bound, 4); s->bind = calloc(s->bound, 4);
    s->stride = calloc(s->bound, 4); s->has_set = calloc(s->bound, 1); s->has_bind = calloc(s->bound, 1);
    s->bufblock = calloc(s->bound, 1);
    for (size_t i = 5; i < s->n;) {
        const uint32_t *o = &s->w[i];
        uint32_t op = o[0] & 0xffff, wc = o[0] >> 16;
        if (!wc || i + wc > s->n) { printf("setup: %s: truncated instruction at word %zu\n", path, i); return -1; }
        uint32_t rid = 0;
        if ((op >= 19 && op <= 39) || op == 5341) rid = wc > 1 ? o[1] : 0;     // types (incl. acceleration structure)
        else if (op == 43 || op == 50 || op == 59) rid = wc > 2 ? o[2] : 0;     // OpConstant, OpSpecConstant, OpVariable
        if (rid && rid < s->bound) s->def[rid] = (uint32_t)i;
        if (op == 71 && wc > 2 && o[1] < s->bound) {                            // OpDecorate
            if (o[2] == 34 && wc > 3) s->set[o[1]] = o[3], s->has_set[o[1]] = 1;
            else if (o[2] == 33 && wc > 3) s->bind[o[1]] = o[3], s->has_bind[o[1]] = 1;
            else if (o[2] == 3) s->bufblock[o[1]] = 1;
            else if (o[2] == 6 && wc > 3) s->stride[o[1]] = o[3];
        } else if (op == 15 && wc > 3 && o[1] == 5 && !s->entry_id) {           // OpEntryPoint GLCompute
            const char *name = (const char *)&o[3];
            size_t max = (wc - 3) * 4;
            if (!opt_entry || strncmp(name, opt_entry, max) == 0) {
                s->entry_id = o[2];
                snprintf(s->entry, sizeof s->entry, "%.*s", (int)strnlen(name, max), name);
            }
        } else if (op == 16 && wc >= 6 && o[2] == 17) {                        // OpExecutionMode LocalSize
            if (o[1] == s->entry_id || !s->local[0]) memcpy(s->local, &o[3], 12);
        }
        i += wc;
    }
    return 0;
}

static void add_binding(uint32_t set, uint32_t b, VkDescriptorType t, uint32_t count, int runtime) {
    struct binding *x = NULL;
    for (uint32_t i = 0; i < nbinds; i++)
        if (binds[i].set == set && binds[i].binding == b) x = &binds[i];
    if (!x) {
        if (nbinds == MAXB) return;
        x = &binds[nbinds++];
        memset(x, 0, sizeof *x);
        x->set = set, x->binding = b;
    }
    uint32_t k = 0;
    while (k < x->ntypes && x->types[k] != t) k++;
    if (k == x->ntypes && layout_done)
        printf("note: set %u binding %u: type %d is not in the -l layout (a different module?)\n", set, b, t);
    if (k == x->ntypes && k < 8) x->types[x->ntypes++] = t;
    if (count > x->count) x->count = count;
    x->runtime |= runtime;
}

// the descriptor type of a resource variable; -1 for anything else
static int var_descriptor(const struct spv *s, const uint32_t *v, VkDescriptorType *t, uint32_t *count, int *runtime) {
    const uint32_t *p = ins(s, v[1]);   // OpTypePointer: id, storage class, pointee
    if (!p || (p[0] & 0xffff) != 32) return -1;
    uint32_t sc = v[3], ty = p[3];
    *count = 1, *runtime = 0;
    const uint32_t *d;
    for (;;) {
        d = ins(s, ty);
        if (!d) return -1;
        if ((d[0] & 0xffff) == 28) *count *= const_value(s, d[3]), ty = d[2];
        else if ((d[0] & 0xffff) == 29) *runtime = 1, ty = d[2];
        else break;
    }
    uint32_t op = d[0] & 0xffff;
    if (sc == 12) { *t = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; return 0; }
    if (sc == 2) { *t = s->bufblock[ty] ? VK_DESCRIPTOR_TYPE_STORAGE_BUFFER : VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER; return 0; }
    if (sc != 0) return -1;
    if (op == 26) { *t = VK_DESCRIPTOR_TYPE_SAMPLER; return 0; }
    if (op == 5341) { *t = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR; return 0; }
    int combined = 0;
    if (op == 27) { combined = 1; d = ins(s, d[2]); if (!d) return -1; op = d[0] & 0xffff; }
    if (op != 25) return -1;
    uint32_t dim = d[3], sampled = d[7];   // OpTypeImage: id, sampled type, Dim, Depth, Arrayed, MS, Sampled, Format
    if (dim == 5) *t = sampled == 2 ? VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER : VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER;
    else if (dim == 6) *t = VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT;
    else if (combined) *t = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    else *t = sampled == 2 ? VK_DESCRIPTOR_TYPE_STORAGE_IMAGE : VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
    return 0;
}

static void reflect(const struct spv *s) {
    for (size_t i = 5; i < s->n; i += s->w[i] >> 16) {
        const uint32_t *v = &s->w[i];
        if ((v[0] & 0xffff) != 59 || (v[0] >> 16) < 4) continue;   // OpVariable
        uint32_t id = v[2];
        if (v[3] == 9) {                                              // PushConstant
            const uint32_t *p = ins(s, v[1]);
            uint32_t sz = p ? type_size(s, p[3], 0) : 0;
            if (sz > pc_size) pc_size = sz;
            continue;
        }
        VkDescriptorType t;
        uint32_t count;
        int runtime;
        if (var_descriptor(s, v, &t, &count, &runtime)) continue;
        if (!s->has_set[id] || !s->has_bind[id]) { printf("note: resource %%%u has no DescriptorSet/Binding, skipped\n", id); continue; }
        add_binding(s->set[id], s->bind[id], t, count, runtime);
    }
}

static const char *dtype_name(VkDescriptorType t) {
    switch (t) {
    case VK_DESCRIPTOR_TYPE_SAMPLER: return "SAMPLER";
    case VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER: return "COMBINED_IMAGE_SAMPLER";
    case VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE: return "SAMPLED_IMAGE";
    case VK_DESCRIPTOR_TYPE_STORAGE_IMAGE: return "STORAGE_IMAGE";
    case VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER: return "UNIFORM_TEXEL_BUFFER";
    case VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER: return "STORAGE_TEXEL_BUFFER";
    case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER: return "UNIFORM_BUFFER";
    case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER: return "STORAGE_BUFFER";
    case VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT: return "INPUT_ATTACHMENT";
    case VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR: return "ACCELERATION_STRUCTURE";
    case VK_DESCRIPTOR_TYPE_MUTABLE_EXT: return "MUTABLE";
    default: return "?";
    }
}

// ---- device: extensions and feature structs
struct extfeat { const char *name; VkStructureType stype; size_t size; };
static const struct extfeat ext_table[] = {   // also the default (vkd3d-relevant) extension list
    {VK_EXT_MUTABLE_DESCRIPTOR_TYPE_EXTENSION_NAME, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MUTABLE_DESCRIPTOR_TYPE_FEATURES_EXT,
     sizeof(VkPhysicalDeviceMutableDescriptorTypeFeaturesEXT)},
    {"VK_VALVE_mutable_descriptor_type", VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MUTABLE_DESCRIPTOR_TYPE_FEATURES_EXT,
     sizeof(VkPhysicalDeviceMutableDescriptorTypeFeaturesEXT)},   // only when the EXT is missing (same struct)
    {VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME, 0, 0},
    {VK_EXT_ROBUSTNESS_2_EXTENSION_NAME, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ROBUSTNESS_2_FEATURES_EXT,
     sizeof(VkPhysicalDeviceRobustness2FeaturesEXT)},
    {VK_EXT_SHADER_ATOMIC_FLOAT_EXTENSION_NAME, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_ATOMIC_FLOAT_FEATURES_EXT,
     sizeof(VkPhysicalDeviceShaderAtomicFloatFeaturesEXT)},
    {VK_EXT_SHADER_ATOMIC_FLOAT_2_EXTENSION_NAME, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_ATOMIC_FLOAT_2_FEATURES_EXT,
     sizeof(VkPhysicalDeviceShaderAtomicFloat2FeaturesEXT)},
    {VK_EXT_SHADER_IMAGE_ATOMIC_INT64_EXTENSION_NAME, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_IMAGE_ATOMIC_INT64_FEATURES_EXT,
     sizeof(VkPhysicalDeviceShaderImageAtomicInt64FeaturesEXT)},
    {VK_KHR_WORKGROUP_MEMORY_EXPLICIT_LAYOUT_EXTENSION_NAME,
     VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_WORKGROUP_MEMORY_EXPLICIT_LAYOUT_FEATURES_KHR,
     sizeof(VkPhysicalDeviceWorkgroupMemoryExplicitLayoutFeaturesKHR)},
    {VK_KHR_SHADER_SUBGROUP_UNIFORM_CONTROL_FLOW_EXTENSION_NAME,
     VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_SUBGROUP_UNIFORM_CONTROL_FLOW_FEATURES_KHR,
     sizeof(VkPhysicalDeviceShaderSubgroupUniformControlFlowFeaturesKHR)},
    {VK_NV_COMPUTE_SHADER_DERIVATIVES_EXTENSION_NAME, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COMPUTE_SHADER_DERIVATIVES_FEATURES_NV,
     sizeof(VkPhysicalDeviceComputeShaderDerivativesFeaturesNV)},
};
#define NEXT (sizeof ext_table / sizeof ext_table[0])

static int wsi_ext(const char *n) {   // -x skips these: they need surface instance extensions
    static const char *const bad[] = {"swapchain", "present", "display", "surface", "full_screen", "android", "win32"};
    for (size_t i = 0; i < sizeof bad / sizeof bad[0]; i++)
        if (strstr(n, bad[i])) return 1;
    return 0;
}

#define CK(x) do { VkResult r_ = (x); if (r_) { printf("setup: %s failed: %d\n", #x, r_); return 2; } } while (0)

static uint32_t limit_for(VkDescriptorType t, const VkPhysicalDeviceDescriptorIndexingProperties *ip,
                          const VkPhysicalDeviceLimits *l, int uab) {
    switch (t) {
    case VK_DESCRIPTOR_TYPE_SAMPLER:
        return uab ? ip->maxPerStageDescriptorUpdateAfterBindSamplers : l->maxPerStageDescriptorSamplers;
    case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER:
        return uab ? ip->maxPerStageDescriptorUpdateAfterBindUniformBuffers : l->maxPerStageDescriptorUniformBuffers;
    case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER:
        return uab ? ip->maxPerStageDescriptorUpdateAfterBindStorageBuffers : l->maxPerStageDescriptorStorageBuffers;
    case VK_DESCRIPTOR_TYPE_STORAGE_IMAGE: case VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER:
        return uab ? ip->maxPerStageDescriptorUpdateAfterBindStorageImages : l->maxPerStageDescriptorStorageImages;
    case VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT:
        return uab ? ip->maxPerStageDescriptorUpdateAfterBindInputAttachments : l->maxPerStageDescriptorInputAttachments;
    default:   // sampled images, combined, uniform texel buffers
        return uab ? ip->maxPerStageDescriptorUpdateAfterBindSampledImages : l->maxPerStageDescriptorSampledImages;
    }
}

static int uab_supported(VkDescriptorType t, const VkPhysicalDeviceVulkan12Features *f) {
    switch (t) {
    case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER: return f->descriptorBindingUniformBufferUpdateAfterBind;
    case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER: return f->descriptorBindingStorageBufferUpdateAfterBind;
    case VK_DESCRIPTOR_TYPE_STORAGE_IMAGE: return f->descriptorBindingStorageImageUpdateAfterBind;
    case VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER: return f->descriptorBindingUniformTexelBufferUpdateAfterBind;
    case VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER: return f->descriptorBindingStorageTexelBufferUpdateAfterBind;
    case VK_DESCRIPTOR_TYPE_SAMPLER: case VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE: case VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER:
        return f->descriptorBindingSampledImageUpdateAfterBind;
    default: return 0;
    }
}

static const VkDescriptorType vkd3d_mutable[] = {
    VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
    VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
};

// The child: everything Vulkan happens here. Returns 2 on a setup error (the reason printed); after the compile it
// reports through the status pipe and leaves with _exit (no teardown inside the vendor driver).
static int child(const char *path, int status_fd) {
    struct spv s, lay;
    if (opt_layout) {
        if (spv_load(opt_layout, &lay)) return 2;
        reflect(&lay);
        layout_done = 1;
    }
    if (spv_load(path, &s)) return 2;
    reflect(&s);
    if (!s.entry_id) { printf("setup: %s has no GLCompute entry point%s%s\n", path, opt_entry ? " named " : "", opt_entry ? opt_entry : ""); return 2; }

    VkApplicationInfo app = {VK_STRUCTURE_TYPE_APPLICATION_INFO, NULL, "spv-compile-probe", 1, NULL, 0, VK_API_VERSION_1_3};
    VkInstanceCreateInfo ici = {VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, NULL, 0, &app};
    VkInstance inst;
    CK(vkCreateInstance(&ici, NULL, &inst));
    uint32_t n = 1;
    VkPhysicalDevice pd;
    VkResult er = vkEnumeratePhysicalDevices(inst, &n, &pd);
    if ((er && er != VK_INCOMPLETE) || !n) { printf("setup: no physical device (%d)\n", er); return 2; }
    VkPhysicalDeviceDescriptorIndexingProperties ip = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_INDEXING_PROPERTIES};
    VkPhysicalDeviceProperties2 p2 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, &ip};
    vkGetPhysicalDeviceProperties2(pd, &p2);
    const VkPhysicalDeviceProperties *props = &p2.properties;
    if (props->apiVersion < VK_API_VERSION_1_2) { printf("setup: device API %u.%u is below 1.2\n",
        VK_API_VERSION_MAJOR(props->apiVersion), VK_API_VERSION_MINOR(props->apiVersion)); return 2; }

    // extensions: the table's (vkd3d-relevant) ones, or with -x every reported one except WSI
    uint32_t ne = 0;
    vkEnumerateDeviceExtensionProperties(pd, NULL, &ne, NULL);
    VkExtensionProperties *ep = calloc(ne ? ne : 1, sizeof *ep);
    vkEnumerateDeviceExtensionProperties(pd, NULL, &ne, ep);
    const char **en = calloc(ne + 1, sizeof *en);
    uint32_t nen = 0;
    int have_ext_mutable = 0, mutable_on = 0, push_on = 0;
    for (uint32_t i = 0; i < ne; i++)
        if (!strcmp(ep[i].extensionName, VK_EXT_MUTABLE_DESCRIPTOR_TYPE_EXTENSION_NAME)) have_ext_mutable = 1;
    for (uint32_t i = 0; i < ne; i++) {
        const char *x = ep[i].extensionName;
        int in_table = 0;
        for (size_t k = 0; k < NEXT; k++) if (!strcmp(x, ext_table[k].name)) in_table = 1;
        if (!strcmp(x, "VK_VALVE_mutable_descriptor_type") && have_ext_mutable) continue;
        if (in_table || (opt_all_ext && !wsi_ext(x))) en[nen++] = x;
    }
    // feature chain: core 1.1/1.2/1.3 + the table's structs of enabled extensions, everything the driver reports on
    VkPhysicalDeviceVulkan11Features f11 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES};
    VkPhysicalDeviceVulkan12Features f12 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES, &f11};
    VkPhysicalDeviceVulkan13Features f13 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES, &f12};
    VkPhysicalDeviceFeatures2 f2 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
                                    props->apiVersion >= VK_API_VERSION_1_3 ? (void *)&f13 : (void *)&f12};
    VkPhysicalDeviceRobustness2FeaturesEXT *rob2 = NULL;
    VkPhysicalDeviceMutableDescriptorTypeFeaturesEXT *mut = NULL;
    for (uint32_t i = 0; i < nen; i++)
        for (size_t k = 0; k < NEXT; k++) {
            if (strcmp(en[i], ext_table[k].name)) continue;
            if (!strcmp(en[i], VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME)) push_on = 1;
            if (!ext_table[k].size) continue;
            VkBaseOutStructure *b = calloc(1, ext_table[k].size);
            b->sType = ext_table[k].stype;
            b->pNext = (VkBaseOutStructure *)f2.pNext;
            f2.pNext = b;
            if (b->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ROBUSTNESS_2_FEATURES_EXT) rob2 = (void *)b;
            if (b->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MUTABLE_DESCRIPTOR_TYPE_FEATURES_EXT) mut = (void *)b;
        }
    vkGetPhysicalDeviceFeatures2(pd, &f2);
    if (opt_no_robust) {
        f2.features.robustBufferAccess = VK_FALSE;
        f13.robustImageAccess = VK_FALSE;
        if (rob2) rob2->robustBufferAccess2 = rob2->robustImageAccess2 = VK_FALSE;
    }
    mutable_on = mut && mut->mutableDescriptorType;

    uint32_t nq = 0, qf = UINT32_MAX;
    vkGetPhysicalDeviceQueueFamilyProperties(pd, &nq, NULL);
    VkQueueFamilyProperties qp[16];
    if (nq > 16) nq = 16;
    vkGetPhysicalDeviceQueueFamilyProperties(pd, &nq, qp);
    for (uint32_t i = 0; i < nq && qf == UINT32_MAX; i++) if (qp[i].queueFlags & VK_QUEUE_COMPUTE_BIT) qf = i;
    if (qf == UINT32_MAX) { printf("setup: no compute queue family\n"); return 2; }
    float prio = 1.0f;
    VkDeviceQueueCreateInfo qci = {VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO, NULL, 0, qf, 1, &prio};
    VkDeviceCreateInfo dci = {VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, &f2, 0, 1, &qci, 0, NULL, nen, en, NULL};
    VkDevice dev;
    CK(vkCreateDevice(pd, &dci, NULL, &dev));
    printf("device: %s, driver 0x%x, API %u.%u.%u, %u extensions on%s (mutable %s, push descriptors %s), "
           "robustBufferAccess %s, robustness2 %s\n", props->deviceName, props->driverVersion,
           VK_API_VERSION_MAJOR(props->apiVersion), VK_API_VERSION_MINOR(props->apiVersion),
           VK_API_VERSION_PATCH(props->apiVersion), nen, opt_all_ext ? " (-x)" : "", mutable_on ? "yes" : "no",
           push_on ? "yes" : "no", f2.features.robustBufferAccess ? "on" : "off",
           rob2 ? (rob2->robustBufferAccess2 ? "on" : "off") : "n/a");

    // set layouts
    uint32_t maxset = 0;
    for (uint32_t i = 0; i < nbinds; i++) if (binds[i].set + 1 > maxset) maxset = binds[i].set + 1;
    if (maxset > 32) { printf("setup: descriptor set %u is too high\n", maxset - 1); return 2; }
    VkDescriptorSetLayout sl[32];
    for (uint32_t set = 0; set < maxset; set++) {
        int push = (opt_push_sets >> set) & 1;
        VkDescriptorSetLayoutBinding lb[MAXB];
        VkDescriptorBindingFlags fl[MAXB];
        VkMutableDescriptorTypeListEXT ml[MAXB];
        VkDescriptorType mtypes[MAXB][14];
        uint32_t nb = 0, any_mut = 0, any_uab = 0, top = 0;
        for (uint32_t i = 0; i < nbinds; i++) if (binds[i].set == set && binds[i].binding >= top) top = binds[i].binding;
        for (uint32_t b = 0; b <= top; b++)   // in binding order
            for (uint32_t i = 0; i < nbinds; i++) {
                const struct binding *x = &binds[i];
                if (x->set != set || x->binding != b) continue;
                VkDescriptorType t = x->types[0];
                ml[nb] = (VkMutableDescriptorTypeListEXT){0, mtypes[nb]};
                if (x->ntypes > 1) {
                    if (!mutable_on) { printf("setup: set %u binding %u aliases %u descriptor types and the device has no "
                                              "mutable descriptors\n", set, b, x->ntypes); return 2; }
                    t = VK_DESCRIPTOR_TYPE_MUTABLE_EXT, any_mut = 1;
                    uint32_t c = 0;
                    if (!opt_mutable_used)
                        for (size_t k = 0; k < sizeof vkd3d_mutable / sizeof vkd3d_mutable[0]; k++) mtypes[nb][c++] = vkd3d_mutable[k];
                    for (uint32_t k = 0; k < x->ntypes; k++) {   // the aliased types not in vkd3d's list (or all with -m used)
                        uint32_t j = 0;
                        while (j < c && mtypes[nb][j] != x->types[k]) j++;
                        if (j == c && c < 14) mtypes[nb][c++] = x->types[k];
                    }
                    ml[nb].descriptorTypeCount = c;
                }
                const VkDescriptorType *tl = x->ntypes > 1 ? mtypes[nb] : &t;
                uint32_t ntl = x->ntypes > 1 ? ml[nb].descriptorTypeCount : 1;
                uint32_t count = x->count;
                fl[nb] = 0;
                if (x->runtime) {
                    if (push) { printf("setup: push set %u has a runtime array at binding %u\n", set, b); return 2; }
                    // a mutable binding is vkd3d's resource heap: update-after-bind and its size follow the image
                    // types (the UBO limits of a mobile driver would shrink it to a handful)
                    static const VkDescriptorType heap_types[] = {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE};
                    const VkDescriptorType *ct = t == VK_DESCRIPTOR_TYPE_MUTABLE_EXT ? heap_types : tl;
                    uint32_t nct = t == VK_DESCRIPTOR_TYPE_MUTABLE_EXT ? 2 : ntl;
                    int uab = f12.descriptorBindingPartiallyBound;
                    for (uint32_t k = 0; k < nct; k++) uab &= uab_supported(ct[k], &f12);
                    count = opt_count ? opt_count : (t == VK_DESCRIPTOR_TYPE_SAMPLER ? 2048 : 1000000);
                    if (!opt_count)
                        for (uint32_t k = 0; k < nct; k++) {
                            uint32_t lim = limit_for(ct[k], &ip, &props->limits, uab);
                            if (lim && lim < count) count = lim;
                        }
                    if (f12.descriptorBindingPartiallyBound) fl[nb] |= VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT;
                    if (uab) fl[nb] |= VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT, any_uab = 1;
                    if (uab && f12.descriptorBindingUpdateUnusedWhilePending)
                        fl[nb] |= VK_DESCRIPTOR_BINDING_UPDATE_UNUSED_WHILE_PENDING_BIT;
                    if (b == top && f12.descriptorBindingVariableDescriptorCount)
                        fl[nb] |= VK_DESCRIPTOR_BINDING_VARIABLE_DESCRIPTOR_COUNT_BIT;
                }
                lb[nb] = (VkDescriptorSetLayoutBinding){b, t, count, VK_SHADER_STAGE_ALL, NULL};
                printf("layout: set %u%s binding %u %s", set, push ? " (push)" : "", b, dtype_name(t));
                if (x->ntypes > 1) {
                    printf(" {");
                    for (uint32_t k = 0; k < ntl; k++) printf("%s%s", k ? "," : "", dtype_name(tl[k]));
                    printf("}");
                }
                printf(" x%u%s%s%s%s\n", count, x->runtime ? " runtime" : "",
                       fl[nb] & VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT ? " partially-bound" : "",
                       fl[nb] & VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT ? " update-after-bind" : "",
                       fl[nb] & VK_DESCRIPTOR_BINDING_VARIABLE_DESCRIPTOR_COUNT_BIT ? " variable-count" : "");
                nb++;
            }
        if (push && !push_on) { printf("setup: -P %u but the device has no VK_KHR_push_descriptor\n", set); return 2; }
        VkMutableDescriptorTypeCreateInfoEXT mci = {VK_STRUCTURE_TYPE_MUTABLE_DESCRIPTOR_TYPE_CREATE_INFO_EXT, NULL, nb, ml};
        VkDescriptorSetLayoutBindingFlagsCreateInfo bfi = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO,
                                                           any_mut ? &mci : NULL, nb, fl};
        VkDescriptorSetLayoutCreateInfo dsl = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, &bfi,
            (push ? VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR : 0) |
            (any_uab ? VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT : 0), nb, lb};
        VkDescriptorSetLayoutSupport sup = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_SUPPORT};
        vkGetDescriptorSetLayoutSupport(dev, &dsl, &sup);
        if (!sup.supported) printf("layout: set %u: vkGetDescriptorSetLayoutSupport says unsupported (creating it anyway)\n", set);
        CK(vkCreateDescriptorSetLayout(dev, &dsl, NULL, &sl[set]));
    }
    uint32_t pcs = opt_pc_bytes ? opt_pc_bytes : (pc_size + 3) & ~3u;
    if (pcs > props->limits.maxPushConstantsSize) pcs = props->limits.maxPushConstantsSize;
    VkPushConstantRange pcr = {VK_SHADER_STAGE_COMPUTE_BIT, 0, pcs};
    VkPipelineLayoutCreateInfo pli = {VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, NULL, 0, maxset, sl, pcs ? 1 : 0, &pcr};
    VkPipelineLayout pl;
    CK(vkCreatePipelineLayout(dev, &pli, NULL, &pl));
    VkShaderModuleCreateInfo smi = {VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, NULL, 0, s.n * 4, s.w};
    VkShaderModule sm;
    CK(vkCreateShaderModule(dev, &smi, NULL, &sm));
    printf("shader: %s, %zu bytes, entry %s, local size %ux%ux%u, push constants %u bytes%s%s\n", path, s.n * 4, s.entry,
           s.local[0], s.local[1], s.local[2], pcs, opt_layout ? ", layout from " : "", opt_layout ? opt_layout : "");
    fflush(stdout);

    VkComputePipelineCreateInfo cpi = {VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO, NULL, 0,
        {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, NULL, 0, VK_SHADER_STAGE_COMPUTE_BIT, sm, s.entry, NULL}, pl};
    VkPipeline pipe;
    char msg[64];
    int len = snprintf(msg, sizeof msg, "S %llu\n", (unsigned long long)now_ns());
    if (write(status_fd, msg, len) != len) return 2;
    VkResult r = vkCreateComputePipelines(dev, VK_NULL_HANDLE, 1, &cpi, NULL, &pipe);
    len = snprintf(msg, sizeof msg, "R %d %llu\n", r, (unsigned long long)now_ns());
    if (write(status_fd, msg, len) != len) return 2;
    fflush(stdout);
    _exit(0);
}

// ---- the parent: one child per file, its stderr captured, a timeout
static int run_one(const char *path) {
    int sp[2], ep[2];
    if (pipe(sp) || pipe(ep)) { printf("RESULT %s setup pipe: %s\n", path, strerror(errno)); return 2; }
    fflush(stdout);
    fflush(stderr);
    pid_t pid = fork();
    if (pid < 0) { printf("RESULT %s setup fork: %s\n", path, strerror(errno)); return 2; }
    if (pid == 0) {
        close(sp[0]);
        close(ep[0]);
        dup2(ep[1], 2);
        close(ep[1]);
        setvbuf(stdout, NULL, _IOLBF, 0);
        struct rlimit nocore = {1, 1};   // 1, not 0: the kernel then skips pipe core handlers too (abort time, no cores)
        setrlimit(RLIMIT_CORE, &nocore);
        int rc = child(path, sp[1]);
        fflush(stdout);
        _exit(rc);
    }
    close(sp[1]);
    close(ep[1]);
    char st[256] = "";
    size_t stn = 0, ecap = 1 << 16, en = 0;
    char *eb = malloc(ecap);
    struct pollfd pf[2] = {{sp[0], POLLIN, 0}, {ep[0], POLLIN, 0}};
    int open_fds = 2, timed_out = 0;
    uint64_t deadline = now_ns() + (uint64_t)opt_timeout * 1000000000ull;
    while (open_fds) {
        uint64_t t = now_ns();
        int ms = t >= deadline ? 0 : (int)((deadline - t) / 1000000) + 1;
        int pr = poll(pf, 2, ms);
        if (pr < 0 && errno == EINTR) continue;
        if (pr == 0 && !timed_out) { timed_out = 1; kill(pid, SIGKILL); deadline = now_ns() + 5000000000ull; continue; }
        if (pr == 0) break;   // pipes still open 5 s after the kill (a grandchild holds them): give up reading
        for (int i = 0; i < 2; i++) {
            if (pf[i].fd < 0 || !(pf[i].revents & (POLLIN | POLLHUP | POLLERR))) continue;
            char buf[4096];
            ssize_t r = read(pf[i].fd, buf, sizeof buf);
            if (r <= 0) { close(pf[i].fd); pf[i].fd = -1; open_fds--; continue; }
            if (i == 0) {
                size_t c = (size_t)r < sizeof st - 1 - stn ? (size_t)r : sizeof st - 1 - stn;
                memcpy(st + stn, buf, c);
                stn += c;
                st[stn] = 0;
            } else if (en < (16u << 20)) {
                if (en + r > ecap) { while (en + r > ecap) ecap *= 2; eb = realloc(eb, ecap); }
                memcpy(eb + en, buf, r);
                en += r;
            }
        }
    }
    int ws = 0;
    waitpid(pid, &ws, 0);
    uint64_t t_end = now_ns();
    for (int i = 0; i < 2; i++) if (pf[i].fd >= 0) close(pf[i].fd);

    if (en) {   // the child's stderr: driver noise and the compiler's message
        printf("---- child stderr (%zu bytes) ----\n", en);
        if (en <= 65536) fwrite(eb, 1, en, stdout);
        else {
            fwrite(eb, 1, 8192, stdout);
            printf("\n[... %zu bytes skipped ...]\n", en - 8192 - 57344);
            fwrite(eb + en - 57344, 1, 57344, stdout);
        }
        if (eb[en - 1] != '\n') printf("\n");
        printf("---- end of child stderr ----\n");
    }
    free(eb);
    unsigned long long t_s = 0, t_r = 0;
    int vr = 0, started = 0, returned = 0;
    char *sl = strstr(st, "S "), *rl = strstr(st, "R ");
    if (sl && sscanf(sl, "S %llu", &t_s) == 1) started = 1;
    if (rl && sscanf(rl, "R %d %llu", &vr, &t_r) == 2) returned = 1;
    const char *base = strrchr(path, '/') ? strrchr(path, '/') + 1 : path;
    char how[96];
    if (WIFSIGNALED(ws)) snprintf(how, sizeof how, "signal %d (%s)", WTERMSIG(ws), strsignal(WTERMSIG(ws)));
    else snprintf(how, sizeof how, "exit %d", WEXITSTATUS(ws));
    if (returned) {
        double ms = (t_r - t_s) / 1e6;
        if (vr == VK_SUCCESS) { printf("RESULT %s compiled in %.1f ms\n", base, ms); return 0; }
        printf("RESULT %s vkerror %d after %.1f ms\n", base, vr, ms);
        return 3;
    }
    if (timed_out) {
        printf("RESULT %s timeout after %d s %s (killed)\n", base, opt_timeout,
               started ? "inside vkCreateComputePipelines" : "before the compile");
        return 4;
    }
    if (started) {
        printf("RESULT %s aborted: %s after %.1f ms inside vkCreateComputePipelines\n", base, how, (t_end - t_s) / 1e6);
        return 1;
    }
    printf("RESULT %s setup failed (%s) before the compile\n", base, how);
    return 2;
}

static void usage(void) {
    printf("usage: spv-compile-probe [-l LAYOUT.spv] [-P SET[,SET]] [-m vkd3d|used] [-n COUNT] [-c BYTES] [-x] [-b]\n"
           "                         [-e ENTRY] [-t SECONDS] FILE.spv...\n"
           "exit: 0 compiled, 1 aborted inside the compile, 2 setup error, 3 VkResult error, 4 timeout\n");
}

int main(int argc, char **argv) {
    int c;
    while ((c = getopt(argc, argv, "l:P:m:n:c:xbe:t:h")) != -1) {
        switch (c) {
        case 'l': opt_layout = optarg; break;
        case 'P':
            for (char *p = optarg; *p;) {
                unsigned long v = strtoul(p, &p, 10);
                if (v < 32) opt_push_sets |= 1u << v;
                if (*p) p++;
            }
            break;
        case 'm':
            if (!strcmp(optarg, "used")) opt_mutable_used = 1;
            else if (strcmp(optarg, "vkd3d")) { usage(); return 2; }
            break;
        case 'n': opt_count = strtoul(optarg, NULL, 0); break;
        case 'c': opt_pc_bytes = strtoul(optarg, NULL, 0); break;
        case 'x': opt_all_ext = 1; break;
        case 'b': opt_no_robust = 1; break;
        case 'e': opt_entry = optarg; break;
        case 't': opt_timeout = atoi(optarg) > 0 ? atoi(optarg) : 60; break;
        default: usage(); return c == 'h' ? 0 : 2;
        }
    }
    if (optind >= argc) { usage(); return 2; }
    signal(SIGPIPE, SIG_IGN);
    int worst = 0;
    for (int i = optind; i < argc; i++) {
        nbinds = 0, pc_size = 0;
        int rc = run_one(argv[i]);
        if (rc > worst) worst = rc;
    }
    return worst;
}

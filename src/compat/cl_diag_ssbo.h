// SPDX-License-Identifier: GPL-3.0-or-later
// cl_diag_ssbo.h: diagnostic MALI_COMPAT_SSBO_ALIGN_LOG (hook table sal_proc). Included by mali_compat_layer.c.
// ---- MALI_COMPAT_SSBO_ALIGN_LOG=1 (diagnostic, default off; docs/agents/test_plans/requests/re-village-align2-1107.md
// "Alternative E2'"): counts the STORAGE_BUFFER / STORAGE_BUFFER_DYNAMIC descriptors the app writes with offset % 64 != 0
// (the driver's real minStorageBufferOffsetAlignment is 64, MALI_COMPAT_FAKE_ALIGN reports 16), histogram by remainder
// 16/32/48/other, plus every SSBO descriptor written. Paths: vkUpdateDescriptorSets (writes; copies carry no offset),
// vkUpdateDescriptorSetWithTemplate, vkCmdPushDescriptorSet, vkCmdPushDescriptorSet2, vkCmdPushDescriptorSetWithTemplate(2)
// (the KHR names too) and vkGetDescriptorEXT (descriptor buffers: address % 64). One stderr line at most every 10 s
// ("mali-compat: ssbo-align: ..."). Off: no hook is handed out (only the g_sal_on test in vkGetDeviceProcAddr). The hooks
// wrap whatever the rest of vkGetDeviceProcAddr hands out (this layer's own hook or the driver's), as sync-diag does.
// Nothing is changed. [night keeper, Claude Opus 5.5] 2026-11-10
// One next pointer per entry-point NAME (core and KHR aliases may resolve to different entry points below).
enum { SAL_UPD, SAL_UPD_TPL, SAL_UPD_TPL_K, SAL_PUSH_K, SAL_PUSH, SAL_PUSH2, SAL_PUSH2_K, SAL_PUSH_TPL, SAL_PUSH_TPL_K,
       SAL_PUSH_TPL2, SAL_PUSH_TPL2_K, SAL_TPL_NEW, SAL_TPL_NEW_K, SAL_TPL_DEL, SAL_TPL_DEL_K, SAL_GETDESC, SAL_N };
_Static_assert(SAL_N <= SAL_NEXT_MAX, "DeviceData.sal_next too small");
enum { SALP_SET, SALP_TPL, SALP_PUSH, SALP_DBUF, SALP_N }; // per path: sets, set templates, push (all kinds), vkGetDescriptorEXT
static bool g_sal_on;
static __thread bool tl_sal_inner;
static PFN_vkVoidFunction g_sal_next_any[SAL_N]; // first resolution (fallback for a call through another device's pointer)
static _Atomic uint64_t g_sal_cnt[SALP_N][5];    // [path][0 = written, 1 = %64 == 16, 2 = 32, 3 = 48, 4 = other != 0]
static _Atomic uint64_t g_sal_due;
static pthread_mutex_t g_sal_print_m = PTHREAD_MUTEX_INITIALIZER;

static void sal_init_once(void) {
    const char *e = getenv("MALI_COMPAT_SSBO_ALIGN_LOG");
    g_sal_on = e && *e && strcmp(e, "0") != 0;
    if (!g_sal_on) return;
    atomic_store(&g_sal_due, now_ns() + 10000000000ull);
    fprintf(stderr, "mali-compat: ssbo-align-log on (MALI_COMPAT_SSBO_ALIGN_LOG): STORAGE_BUFFER(_DYNAMIC) descriptors with "
            "offset %% 64 != 0 in sets/templates/push/vkGetDescriptorEXT, one line per 10 s\n");
}
static void sal_init(void) {
    static pthread_once_t once = PTHREAD_ONCE_INIT;
    pthread_once(&once, sal_init_once);
}
static inline bool sal_is_ssbo(VkDescriptorType t) {
    return t == VK_DESCRIPTOR_TYPE_STORAGE_BUFFER || t == VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC;
}
static inline void sal_add(uint64_t loc[5], VkDeviceSize off) {
    uint32_t r = (uint32_t)(off & 63);
    loc[0]++;
    if (r) loc[r == 16 ? 1 : r == 32 ? 2 : r == 48 ? 3 : 4]++;
}
static void sal_tick(void) {
    uint64_t now = now_ns(), due = atomic_load_explicit(&g_sal_due, memory_order_relaxed);
    if (now < due || !atomic_compare_exchange_strong(&g_sal_due, &due, now + 10000000000ull)) return;
    if (pthread_mutex_trylock(&g_sal_print_m)) return;
    static uint64_t last[SALP_N][5], last_ns;
    uint64_t cur[SALP_N][5], w[5] = {0}, t[5] = {0}, wp[SALP_N][2];
    for (int p = 0; p < SALP_N; p++) {
        for (int b = 0; b < 5; b++) {
            cur[p][b] = atomic_load_explicit(&g_sal_cnt[p][b], memory_order_relaxed);
            w[b] += cur[p][b] - last[p][b];
            t[b] += cur[p][b];
        }
        wp[p][0] = cur[p][0] - last[p][0];
        wp[p][1] = (cur[p][1] - last[p][1]) + (cur[p][2] - last[p][2]) + (cur[p][3] - last[p][3]) + (cur[p][4] - last[p][4]);
    }
    double secs = last_ns ? (double)(now - last_ns) / 1e9 : 10.0;
    memcpy(last, cur, sizeof last);
    last_ns = now;
    pthread_mutex_unlock(&g_sal_print_m);
    #define U(x) (unsigned long long)(x)
    fprintf(stderr, "mali-compat: ssbo-align: last %.1f s: written %llu, offset%%64!=0 %llu (16:%llu 32:%llu 48:%llu other:%llu; "
            "written/!=0 set %llu/%llu tpl %llu/%llu push %llu/%llu dbuf %llu/%llu) | total written %llu, !=0 %llu "
            "(16:%llu 32:%llu 48:%llu other:%llu)\n", secs, U(w[0]), U(w[1] + w[2] + w[3] + w[4]), U(w[1]), U(w[2]), U(w[3]),
            U(w[4]), U(wp[SALP_SET][0]), U(wp[SALP_SET][1]), U(wp[SALP_TPL][0]), U(wp[SALP_TPL][1]), U(wp[SALP_PUSH][0]),
            U(wp[SALP_PUSH][1]), U(wp[SALP_DBUF][0]), U(wp[SALP_DBUF][1]), U(t[0]), U(t[1] + t[2] + t[3] + t[4]), U(t[1]),
            U(t[2]), U(t[3]), U(t[4]));
    #undef U
}
static void sal_flush(int path, const uint64_t loc[5]) {
    if (loc[0])
        for (int b = 0; b < 5; b++)
            if (loc[b]) atomic_fetch_add_explicit(&g_sal_cnt[path][b], loc[b], memory_order_relaxed);
    sal_tick();
}
static void sal_writes(int path, uint32_t n, const VkWriteDescriptorSet *w) {
    uint64_t loc[5] = {0};
    for (uint32_t i = 0; w && i < n; i++)
        if (sal_is_ssbo(w[i].descriptorType) && w[i].pBufferInfo)
            for (uint32_t k = 0; k < w[i].descriptorCount; k++) sal_add(loc, w[i].pBufferInfo[k].offset);
    sal_flush(path, loc);
}

// Templates: only their SSBO entries are kept ((VkDevice, handle) -> entries), from vkCreateDescriptorUpdateTemplate on.
struct SalTplEnt { uint32_t count; size_t offset, stride; };
struct SalTpl { VkDevice dev; uint64_t h; uint32_t n; struct SalTplEnt *e; struct SalTpl *next; };
#define SAL_TPL_BUCKETS 256
static struct SalTpl *g_sal_tpl[SAL_TPL_BUCKETS];
static pthread_mutex_t g_sal_tpl_m = PTHREAD_MUTEX_INITIALIZER;
static void sal_tpl_del_locked(VkDevice dev, uint64_t h) {
    for (struct SalTpl **pp = &g_sal_tpl[handle_bucket(h, SAL_TPL_BUCKETS)]; *pp; pp = &(*pp)->next)
        if ((*pp)->h == h && (*pp)->dev == dev) { struct SalTpl *t = *pp; *pp = t->next; free(t->e); free(t); return; }
}
static void sal_tpl_new(VkDevice dev, VkDescriptorUpdateTemplate tpl, const VkDescriptorUpdateTemplateCreateInfo *ci) {
    uint32_t n = 0;
    for (uint32_t i = 0; i < ci->descriptorUpdateEntryCount; i++) n += sal_is_ssbo(ci->pDescriptorUpdateEntries[i].descriptorType);
    struct SalTpl *t = n ? calloc(1, sizeof *t) : NULL;
    if (t && !(t->e = calloc(n, sizeof *t->e))) { free(t); t = NULL; }
    if (t) {
        t->dev = dev;
        t->h = (uint64_t)tpl;
        for (uint32_t i = 0; i < ci->descriptorUpdateEntryCount; i++) {
            const VkDescriptorUpdateTemplateEntry *s = &ci->pDescriptorUpdateEntries[i];
            if (sal_is_ssbo(s->descriptorType)) t->e[t->n++] = (struct SalTplEnt){s->descriptorCount, s->offset, s->stride};
        }
    }
    pthread_mutex_lock(&g_sal_tpl_m);
    sal_tpl_del_locked(dev, (uint64_t)tpl); // a reused handle whose destroy was not seen
    if (t) {
        uint32_t b = handle_bucket(t->h, SAL_TPL_BUCKETS);
        t->next = g_sal_tpl[b];
        g_sal_tpl[b] = t;
    }
    pthread_mutex_unlock(&g_sal_tpl_m);
}
static void sal_tpl_data(int path, VkDevice dev, VkDescriptorUpdateTemplate tpl, const void *data) {
    uint64_t loc[5] = {0};
    if (data) {
        pthread_mutex_lock(&g_sal_tpl_m);
        for (struct SalTpl *t = g_sal_tpl[handle_bucket((uint64_t)tpl, SAL_TPL_BUCKETS)]; t; t = t->next) {
            if (t->h != (uint64_t)tpl || t->dev != dev) continue;
            for (uint32_t i = 0; i < t->n; i++)
                for (uint32_t k = 0; k < t->e[i].count; k++)
                    sal_add(loc, ((const VkDescriptorBufferInfo *)((const uint8_t *)data + t->e[i].offset +
                                                                     k * t->e[i].stride))->offset);
            break;
        }
        pthread_mutex_unlock(&g_sal_tpl_m);
    }
    sal_flush(path, loc);
}

static PFN_vkVoidFunction sal_next(struct DeviceData *d, int kind) {
    PFN_vkVoidFunction f = d ? d->sal_next[kind] : NULL;
    return f ? f : g_sal_next_any[kind];
}
static VKAPI_ATTR void VKAPI_CALL sal_UpdateDescriptorSets(VkDevice device, uint32_t nw, const VkWriteDescriptorSet *w,
                                                           uint32_t nc, const VkCopyDescriptorSet *c) {
    sal_writes(SALP_SET, nw, w);
    ((PFN_vkUpdateDescriptorSets)sal_next(get_device_data(device), SAL_UPD))(device, nw, w, nc, c);
}
static inline void sal_upd_tpl(int kind, VkDevice device, VkDescriptorSet set, VkDescriptorUpdateTemplate t, const void *data) {
    sal_tpl_data(SALP_TPL, device, t, data);
    ((PFN_vkUpdateDescriptorSetWithTemplate)sal_next(get_device_data(device), kind))(device, set, t, data);
}
static inline void sal_push(int kind, VkCommandBuffer cmd, VkPipelineBindPoint bp, VkPipelineLayout layout, uint32_t set,
                            uint32_t n, const VkWriteDescriptorSet *w) {
    sal_writes(SALP_PUSH, n, w);
    ((PFN_vkCmdPushDescriptorSetKHR)sal_next(get_device_data_by_cmd(cmd), kind))(cmd, bp, layout, set, n, w);
}
static inline void sal_push2(int kind, VkCommandBuffer cmd, const VkPushDescriptorSetInfo *i) {
    if (i) sal_writes(SALP_PUSH, i->descriptorWriteCount, i->pDescriptorWrites);
    ((PFN_vkCmdPushDescriptorSet2)sal_next(get_device_data_by_cmd(cmd), kind))(cmd, i);
}
static inline void sal_push_tpl(int kind, VkCommandBuffer cmd, VkDescriptorUpdateTemplate t, VkPipelineLayout layout,
                                uint32_t set, const void *data) {
    struct DeviceData *d = get_device_data_by_cmd(cmd);
    sal_tpl_data(SALP_PUSH, d ? d->device : VK_NULL_HANDLE, t, data);
    ((PFN_vkCmdPushDescriptorSetWithTemplate)sal_next(d, kind))(cmd, t, layout, set, data);
}
static inline void sal_push_tpl2(int kind, VkCommandBuffer cmd, const VkPushDescriptorSetWithTemplateInfo *i) {
    struct DeviceData *d = get_device_data_by_cmd(cmd);
    if (i) sal_tpl_data(SALP_PUSH, d ? d->device : VK_NULL_HANDLE, i->descriptorUpdateTemplate, i->pData);
    ((PFN_vkCmdPushDescriptorSetWithTemplate2)sal_next(d, kind))(cmd, i);
}
static inline VkResult sal_tpl_create(int kind, VkDevice device, const VkDescriptorUpdateTemplateCreateInfo *ci,
                                      const VkAllocationCallbacks *a, VkDescriptorUpdateTemplate *t) {
    VkResult r = ((PFN_vkCreateDescriptorUpdateTemplate)sal_next(get_device_data(device), kind))(device, ci, a, t);
    if (r == VK_SUCCESS && ci && t) sal_tpl_new(device, *t, ci);
    return r;
}
static inline void sal_tpl_destroy(int kind, VkDevice device, VkDescriptorUpdateTemplate t, const VkAllocationCallbacks *a) {
    if (t) {
        pthread_mutex_lock(&g_sal_tpl_m);
        sal_tpl_del_locked(device, (uint64_t)t);
        pthread_mutex_unlock(&g_sal_tpl_m);
    }
    ((PFN_vkDestroyDescriptorUpdateTemplate)sal_next(get_device_data(device), kind))(device, t, a);
}
// one wrapper per name (core and KHR), each with its own next pointer
static VKAPI_ATTR void VKAPI_CALL sal_UpdateDescriptorSetWithTemplate(
    VkDevice device, VkDescriptorSet set, VkDescriptorUpdateTemplate t, const void *data) {
    sal_upd_tpl(SAL_UPD_TPL, device, set, t, data);
}
static VKAPI_ATTR void VKAPI_CALL sal_UpdateDescriptorSetWithTemplateKHR(
    VkDevice device, VkDescriptorSet set, VkDescriptorUpdateTemplate t, const void *data) {
    sal_upd_tpl(SAL_UPD_TPL_K, device, set, t, data);
}
static VKAPI_ATTR void VKAPI_CALL sal_CmdPushDescriptorSet(
    VkCommandBuffer cmd, VkPipelineBindPoint bp, VkPipelineLayout layout, uint32_t set, uint32_t n, const VkWriteDescriptorSet *w) {
    sal_push(SAL_PUSH, cmd, bp, layout, set, n, w);
}
static VKAPI_ATTR void VKAPI_CALL sal_CmdPushDescriptorSetKHR(
    VkCommandBuffer cmd, VkPipelineBindPoint bp, VkPipelineLayout layout, uint32_t set, uint32_t n, const VkWriteDescriptorSet *w) {
    sal_push(SAL_PUSH_K, cmd, bp, layout, set, n, w);
}
static VKAPI_ATTR void VKAPI_CALL sal_CmdPushDescriptorSet2(
    VkCommandBuffer cmd, const VkPushDescriptorSetInfo *i) {
    sal_push2(SAL_PUSH2, cmd, i);
}
static VKAPI_ATTR void VKAPI_CALL sal_CmdPushDescriptorSet2KHR(
    VkCommandBuffer cmd, const VkPushDescriptorSetInfo *i) {
    sal_push2(SAL_PUSH2_K, cmd, i);
}
static VKAPI_ATTR void VKAPI_CALL sal_CmdPushDescriptorSetWithTemplate(
    VkCommandBuffer cmd, VkDescriptorUpdateTemplate t, VkPipelineLayout layout, uint32_t set, const void *data) {
    sal_push_tpl(SAL_PUSH_TPL, cmd, t, layout, set, data);
}
static VKAPI_ATTR void VKAPI_CALL sal_CmdPushDescriptorSetWithTemplateKHR(
    VkCommandBuffer cmd, VkDescriptorUpdateTemplate t, VkPipelineLayout layout, uint32_t set, const void *data) {
    sal_push_tpl(SAL_PUSH_TPL_K, cmd, t, layout, set, data);
}
static VKAPI_ATTR void VKAPI_CALL sal_CmdPushDescriptorSetWithTemplate2(
    VkCommandBuffer cmd, const VkPushDescriptorSetWithTemplateInfo *i) {
    sal_push_tpl2(SAL_PUSH_TPL2, cmd, i);
}
static VKAPI_ATTR void VKAPI_CALL sal_CmdPushDescriptorSetWithTemplate2KHR(
    VkCommandBuffer cmd, const VkPushDescriptorSetWithTemplateInfo *i) {
    sal_push_tpl2(SAL_PUSH_TPL2_K, cmd, i);
}
static VKAPI_ATTR VkResult VKAPI_CALL sal_CreateDescriptorUpdateTemplate(
    VkDevice device, const VkDescriptorUpdateTemplateCreateInfo *ci, const VkAllocationCallbacks *a, VkDescriptorUpdateTemplate *t) {
    return sal_tpl_create(SAL_TPL_NEW, device, ci, a, t);
}
static VKAPI_ATTR VkResult VKAPI_CALL sal_CreateDescriptorUpdateTemplateKHR(
    VkDevice device, const VkDescriptorUpdateTemplateCreateInfo *ci, const VkAllocationCallbacks *a, VkDescriptorUpdateTemplate *t) {
    return sal_tpl_create(SAL_TPL_NEW_K, device, ci, a, t);
}
static VKAPI_ATTR void VKAPI_CALL sal_DestroyDescriptorUpdateTemplate(
    VkDevice device, VkDescriptorUpdateTemplate t, const VkAllocationCallbacks *a) {
    sal_tpl_destroy(SAL_TPL_DEL, device, t, a);
}
static VKAPI_ATTR void VKAPI_CALL sal_DestroyDescriptorUpdateTemplateKHR(
    VkDevice device, VkDescriptorUpdateTemplate t, const VkAllocationCallbacks *a) {
    sal_tpl_destroy(SAL_TPL_DEL_K, device, t, a);
}
static VKAPI_ATTR void VKAPI_CALL sal_GetDescriptorEXT(VkDevice device, const VkDescriptorGetInfoEXT *gi, size_t size, void *out) {
    if (gi && sal_is_ssbo(gi->type) && gi->data.pStorageBuffer) {
        uint64_t loc[5] = {0};
        sal_add(loc, gi->data.pStorageBuffer->address);
        sal_flush(SALP_DBUF, loc);
    }
    ((PFN_vkGetDescriptorEXT)sal_next(get_device_data(device), SAL_GETDESC))(device, gi, size, out);
}
#define SAL_N2(name, kind) {"vk" #name, kind, (PFN_vkVoidFunction)sal_##name}
static const struct { const char *name; uint8_t kind; PFN_vkVoidFunction fn; } g_sal_names[] = {
    SAL_N2(UpdateDescriptorSets, SAL_UPD),
    SAL_N2(UpdateDescriptorSetWithTemplate, SAL_UPD_TPL), SAL_N2(UpdateDescriptorSetWithTemplateKHR, SAL_UPD_TPL_K),
    SAL_N2(CmdPushDescriptorSet, SAL_PUSH), SAL_N2(CmdPushDescriptorSetKHR, SAL_PUSH_K),
    SAL_N2(CmdPushDescriptorSet2, SAL_PUSH2), SAL_N2(CmdPushDescriptorSet2KHR, SAL_PUSH2_K),
    SAL_N2(CmdPushDescriptorSetWithTemplate, SAL_PUSH_TPL), SAL_N2(CmdPushDescriptorSetWithTemplateKHR, SAL_PUSH_TPL_K),
    SAL_N2(CmdPushDescriptorSetWithTemplate2, SAL_PUSH_TPL2), SAL_N2(CmdPushDescriptorSetWithTemplate2KHR, SAL_PUSH_TPL2_K),
    SAL_N2(CreateDescriptorUpdateTemplate, SAL_TPL_NEW), SAL_N2(CreateDescriptorUpdateTemplateKHR, SAL_TPL_NEW_K),
    SAL_N2(DestroyDescriptorUpdateTemplate, SAL_TPL_DEL), SAL_N2(DestroyDescriptorUpdateTemplateKHR, SAL_TPL_DEL_K),
    SAL_N2(GetDescriptorEXT, SAL_GETDESC),
};
#undef SAL_N2
// Called near the top of vkGetDeviceProcAddr while MALI_COMPAT_SSBO_ALIGN_LOG is on: the counting wrapper around the entry
// point the rest of vkGetDeviceProcAddr hands out. NULL = not ours, or the name does not exist below.
static PFN_vkVoidFunction sal_proc(VkDevice device, const char *pName) {
    if (!strstr(pName, "Descriptor")) return NULL;
    uint32_t k = 0;
    while (k < sizeof g_sal_names / sizeof g_sal_names[0] && strcmp(g_sal_names[k].name, pName)) k++;
    if (k == sizeof g_sal_names / sizeof g_sal_names[0]) return NULL;
    struct DeviceData *d = get_device_data(device);
    if (!d) return NULL;
    tl_sal_inner = true;
    PFN_vkVoidFunction inner = vkGetDeviceProcAddr(device, pName);
    tl_sal_inner = false;
    if (!inner) return NULL;
    int kind = g_sal_names[k].kind;
    pthread_mutex_lock(&g_sal_tpl_m);
    if (!d->sal_next[kind]) d->sal_next[kind] = inner;
    if (!g_sal_next_any[kind]) g_sal_next_any[kind] = inner;
    pthread_mutex_unlock(&g_sal_tpl_m);
    return g_sal_names[k].fn;
}


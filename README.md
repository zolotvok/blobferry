# Blobferry

**Run Windows games on an Android phone's own Mali Vulkan driver, from a Linux (glibc) userspace.**

> **Experimental.** Tested on one phone only (vivo X300 Pro, Mali-G1-Ultra MC12, vendor driver `v1.r54p1`). Expect
> rendering bugs, hangs and crashes. Nothing here is endorsed by Arm, the phone vendor or the DroidDeck project.

Blobferry ferries Vulkan calls from Linux programs (Proton/Wine with DXVK or vkd3d-proton, under FEX) to the
proprietary Mali driver ("the blob") that ships on the phone, instead of an open-source driver such as PanVK. It was
built for [DroidDeck](https://github.com/Droid-Deck/DroidDeck), a SteamOS-like app for Android, and is used through
the [zolotvok/DroidDeck](https://github.com/zolotvok/DroidDeck) fork (branch `mali-panvk`). By default Steam and
gamescope stay on PanVK there and only the games run on the vendor driver. An experimental setting (Performance >
Mali system driver: "Steam and gamescope on it too") runs the whole session on Blobferry as well: gamescope, Xwayland
(glamor on Mesa Zink) and the Steam client, with Steam's menus drawn in software. Tested on 2026-10-30: Steam Big
Picture complete and responsive, Oblivion launched from it and ran its menu at ~92 FPS. See
[Session on the vendor driver](#session-on-the-vendor-driver).

**Using it:** DroidDeck from the fork, version 0.4.0-mali.1 or newer, bundles Blobferry. On a Mali phone it installs
the files into its Linux runtime at every session start, and "Mali system driver (bridge)" appears on its Runtime
driver page. No adb or zip import is needed. The steps below are for building and testing Blobferry itself.

## How it works

```
game (DXVK / vkd3d-proton, Proton, FEX)
  └─ Vulkan loader (glibc)
       ├─ VK_LAYER_DROIDDECK_mali_compat   src/compat: makes DXVK/vkd3d-proton work on the Mali driver
       ├─ Arm vulkan-wsi-layer             Wayland swapchains on dma-buf (patched, see patches/)
       └─ libvulkan_mali_bridge.so         src/icd: Khronos ICD shim
            └─ libhybris                   bionic linker inside the glibc process
                 └─ /vendor/lib64/hw/vulkan.mali.so   the phone's own driver (never shipped)
```

The vendor driver is loaded into each Vulkan process. There is no IPC or call serialisation, so the CPU cost stays
close to native.

**ICD shim (`src/icd/mali_bridge_icd.c`):** loads the Android Vulkan HAL through libhybris and exposes it as a normal
Khronos ICD.

**Compat layer (`src/compat/mali_compat_layer.c`):** fills the gaps between what DXVK and vkd3d-proton expect and what
the Mali driver offers:
- BC1-BC7 textures decoded on the GPU (compute or fragment shaders), with AFRC-compressed shadow images to save memory;
- SPIR-V patching that removes ClipDistance/CullDistance where the driver lacks them;
- feature masking and a polygon-mode override;
- emulated `VK_EXT_memory_budget` and `VK_EXT_map_memory_placed`;
- a fix for a vendor driver hang: a fragment shader that writes `gl_FragDepth` in a pass without a depth attachment
  (vkd3d-proton's depth-to-colour copy does this) gets that output removed, which Vulkan discards anyway;
- optional render-target compression: STORAGE usage turns compression off on this driver, so colour targets are
  created without it, and the ones the game really uses as UAVs are learned per game;
- D3D11 stream output without DXVK's geometry shader (the vertex shader captures instead), which costs ~4 ms per draw
  on this driver;
- against long pipeline compiles: a persistent pipeline cache and asynchronous compiles with stand-in pipelines;
- for the session: emulated `VK_EXT_physical_device_drm` and dma-buf export, which gamescope needs and the driver
  lacks;
- a built-in GPU profiler: per render pass / dispatch GPU time, and the pass a GPU hang is stuck in.

Most of these are opt-in, see [Switches](#switches).

The layer is implicit and turns on only with `ENABLE_MALI_COMPAT=1` (the DroidDeck fork sets it for bridge games and for
the session on Blobferry).

## Status

State on 2026-11-02, from the measurements up to 2026-11-01 (one phone, one person testing). Every FPS figure carries its
date, its settings and the Android thermal status (0-4): the GPU clock cap falls from thermal 2 on, so numbers at thermal 3-4
are not comparable with cool ones. "User" marks something the tester reported without a meter. Games with no measured
current state are left out.

| Game | API | Latest measured state |
|---|---|---|
| The Elder Scrolls IV: Oblivion | DX9 → DXVK | in game ~89 FPS, menu 87-89 (older measurement; date and thermal not recorded). With the session itself on Blobferry: menu ~92 FPS (2026-10-30) |
| Tomb Raider (2013) | DX11 → DXVK | built-in benchmark, DXVK pipeline libraries off, layer GPU timing on: 2026-11-01 12:59 min 44.6 / avg 60.5 / max 71.4 FPS at thermal 0-2; 2026-11-01 14:30 min 41.3 / avg 55.3 / max 72.1 at thermal 3 (CPU/GPU 87 °C, skin 46.0 °C, with two extra timing instruments in the layers) |
| Yooka-Laylee and the Impossible Lair | DX11 → DXVK (Unity) | 2026-10-30, `UnityGraphicsQuality=0`, 1600x720: 80-90 FPS in game with `MALI_COMPAT_XFB_VS=1`, 19-20 without (thermal not recorded). One run went black at 0 FPS after Continue, not reproduced |
| The Long Dark | DX11 → DXVK (Unity 6) | game's own Ultra preset, 1600x720, layer GPU timing on, 2026-11-01 15:07-15:12: in the house 38.9 FPS (GPU clock 1029-1163 MHz average, thermal 1, 68 °C, skin 41.3 °C); outdoors 28.6 FPS (735-779 MHz, cap 1066, thermal 2, 72 °C, skin 43.4 °C); main menu 36.5-39.3 |
| Hob | DX11 → DXVK (32-bit) | 2026-10-31: starts to the menu and plays, autoplay 244 s min 24 / avg 38.7 / max 45 FPS (thermal not recorded). Loading is intermittent, see [notes](#hob) |
| The Witcher 3 (Steam `next-gen` build 14504303, DX11 exe) | DX11 → DXVK | 2026-11-01 17:09, after Continue: 20.9-22.0 FPS at thermal 1 (CPU/GPU 84 °C, skin 42.5 °C). 2026-10-30, low settings, 1600x720, White Orchard: 18.3-18.7 FPS average standing at the spawn view (thermal not logged), 29.2 FPS in one cool view (GPU clock ~1240 MHz, thermal 1-2); at the Kaer Morhen training ground 13.9 FPS at thermal 3-4 (GPU clock 312 MHz average). Needs `NOATT_AREA`, `PIPELINE_CACHE` and `ASYNC_PSO`, see [notes](#the-witcher-3) |
| The Witcher 3 (classic v1.32) | DX11 → DXVK | older measurement (date not recorded): 41-48 FPS indoors, 19-29 FPS outdoors at thermal 3 (GPU 75-82 °C) |
| The Witcher 3 (current release) | DX12 → vkd3d-proton | 2026-10-28 21:06 22.2 FPS in White Orchard (thermal not recorded); 2026-10-30 averages 19.3 and 18.3 FPS (thermal 1 → 2 and 2 → 3). Not the exe a launch starts any more on the `next-gen` build |
| Horizon Zero Dawn | DX12 → vkd3d-proton | built-in benchmark, 1600x720 Custom preset: 2026-11-01 13:25 avg 15 / min 7 / max 35 FPS (CPU 15, GPU 24, thermal 3); 2026-11-01 14:55 14-26 FPS at thermal 2, then a 0-FPS stall at benchmark + 2:56 with no result screen. Static meshes (buildings, props, cliffs) were missing in every run that checked it (last 2026-10-29), see [notes](#horizon-zero-dawn) |
| Control | DX12 → vkd3d-proton | plays on the Low preset: 2026-11-01 17:53-17:58 gameplay min 23.0 / avg 29.4 / max 39.9 FPS at thermal 3 (CPU/GPU max 105 °C, skin 46.1 °C); about 20 FPS after the janitor mini-cutscene. Shadow flicker, a freeze and a pause problem are open, see [notes](#control) |
| Watch Dogs | DX11 → DXVK (Ubisoft Connect) | starts and reaches gameplay with the current Ubisoft Connect in the prefix and `MALI_COMPAT_WLS_CLAMP=all`: 2026-11-01 22:52, prologue, 60 s on the DXVK HUD min 23.7 / mean 26.8 / max 31.9 FPS at thermal 4 (CPU/GPU 71-75 °C). User: lowering the resolution in the game's options crashed it |
| Watch Dogs 2 | DX11 → DXVK (Ubisoft Connect) | playable (user report) with `MALI_COMPAT_WLS_CLAMP` (six shader hashes) and serialized submissions. User, evening of 2026-11-01, no meter, thermal not measured: the protagonist's head is drawn, ~40 FPS at the start falling to ~15 and ~8 in firefights, crash after cutting the power in the server room (cause not measured), see [notes](#watch-dogs-2) |

### Control

- **Preset:** Low. The tester's `renderer.ini` (md5 e7b8377d) has 13 quality keys at 0 (texture resolution and MSAA among them).
  User, 2026-11-01 17:4x: the graphics bugs are the same with Low, so Low is the phone setting. On the previous, higher
  preset Control ran at 12-24 FPS (2026-10-31 to 2026-11-01 16:55, thermal 2-3 where recorded).
- **Env of the last run** (2026-11-01 17:5x; not a minimal set, the keys were not bisected): `MALI_COMPAT_NO_SUSPEND_RESUME=1`
  (with it the 3D scene is drawn; without it the 2026-10-31 run showed no 3D and the intro video in small rects),
  `FEX_DISKCACHE=0` in the guest (without it FEX's disk cache deadlocked the game's UI thread, black at 0 FPS, 2026-10-31), `MALI_COMPAT_SPV_FIX=1`,
  `VKD3D_SHADER_OVERRIDE=Z:/root/control-override`, `VKD3D_CONFIG=nodxr`, `VKD3D_SHADER_CACHE_PATH=0`,
  `PROTON_NO_ESYNC=1 PROTON_NO_FSYNC=1 PROTON_NO_NTSYNC=1 WINEESYNC=0`, `MALI_WSI_FPS=1`; that run also had
  `MALI_COMPAT_GPU_TIME=1` (measuring) and `MALI_COMPAT_FORCE_COHERENT=1` (experiment).
- **The override is a stub.** The vendor shader compiler aborts (LLVM "Cannot select") on one large compute shader
  (`bf125ef61cc75d97`); `SPV_FIX` does not remove that. The override replaces it with an empty 8x8x1 compute shader that
  writes nothing. The stub was not dispatched in the traced in-game scene (2026-10-31, 0 trace lines); whether the picture
  lacks something because of it is not known.
- **Pad:** with the DroidDeck build "padapply3" (APK md5 7ebd8716, DroidDeck side, not Blobferry) the stick worked all of
  the 2026-11-01 17:5x run without a guide-button press (one run). Earlier builds lost the virtual pad in gameplay after
  Steam re-created it.
- **FPS steps:** 2026-11-01 17:22 (user run, Low): 39 FPS while walking, then at the janitor mini-cutscene (17:22:46-51)
  a drop that stayed: 19.4-21.8 FPS, mean 20.9, at 17:28 (thermal 2, CPU/GPU 68-74 °C, skin 44.3 °C, GPU clock mean 622 MHz).
  Measured at the step: records per frame 15 → 150-190, draws per frame 36 → 717-929, GPU time per frame 4 → 29-55 ms.
  User: not the heat. Whether the game streamed in more, or the cutscene scene is simply heavier, is not separated.
- **Open:**
  - Shadow flicker in the security booth (one light pool flips bright and dark frame by frame). Not fixed. Ruled out
    by measurement: a barrier between the split render-pass parts (`SUSRES_BARRIER`), AFBC/AFRC on the depth atlas,
    GPU timing, caster culling. `FORCE_COHERENT` is untested there: the run that had it froze before the booth.
  - Freeze at the security gate, 2026-11-01 17:58:48: 0 FPS, process alive, thermal 3, GPU memory 6226 MB. Cause not measured.
  - Opening the pause menu in gameplay ended the game in 4 of 6 tries (2026-11-01 06:00 to 16:46, none of them on Low);
    one death (14:08) was a SIGKILL (exit code 9). Who sent it is not measured (low memory is suspected); not re-tested on Low.

### The Witcher 3

The `next-gen` build starts the DX11 exe (`bin/x64`) first. Switches that the current runs need: `NOATT_AREA=4096x4096`
(DXVK's 16384x16384x256 attachment-less passes hung the GPU at the Kaer Morhen training ground), `PIPELINE_CACHE=1` and
`ASYNC_PSO=50` (single pipeline compiles of 3.7-9.7 s stalled the game at 0 FPS in fights and new places), `XFB_VS=1`. The
frame is GPU-bound (84-92 % busy, 2026-10-30), and at ~1.1 GHz about 37 % of the GPU time scales with the core clock (measured on a pair
of views at ~1240 and ~1040 MHz), so a hot phone costs frame rate. Do not use `GT_TRACE=all` here (crash at start).

### Horizon Zero Dawn

Needs DroidDeck's "DX12: no descriptor buffers" (black screen otherwise), `MALI_COMPAT_NO_SUSPEND_RESUME=1` (blocky sky) and
`MALI_COMPAT_PIPELINE_CACHE=2` (load 2:35 on 2026-11-01 instead of ~11 min). The best combined average measured is 16 FPS
(2026-10-29); it sits at the CPU column of the result screen (CPU 15-16, GPU 19-28). The benchmark completed in most runs up
to 2026-11-01 13:25; some earlier runs froze in the load or the benchmark, and the 2026-11-01 14:55 run stalled. The 2026-11-01 runs did not record whether static meshes are drawn.

### Hob

The launcher (`HobLauncher.exe`) needs the MFC 2012 runtime, so the Epic sidecar starts `HOB.exe` directly. Loading hit a
race in Hob's job pool: with the plain environment 0 of 3 loads finished (2026-10-30). With `FEX_STRICTINPROCESSSPLITLOCKS=1`
and `WINE_CPU_TOPOLOGY=4:4,5,6,7` 3 of 3 finished in 5.6-9.6 s (2026-10-31). That is one series of three.

### Watch Dogs 2

State on the phone from 2026-11-01 18:21 (layer 950546af): `MALI_COMPAT_WLS_CLAMP=e0ff8b4d,46aad60b,653351e8,ae9a9376,92ae90c3,25785cbd`
together with the vendor driver's serialized-submissions debug option (`MALI_DEBUG_CONFIG`), `BUDGET_MB=2048`,
`PIPELINE_CACHE=1` and `ASYNC_PSO=50`. The head was missing with serialization alone (2026-11-01 11:29, ~20 FPS by the
user's account, 16850 GPU group errors in the log). The clamp without serialization has not been run on this game. The log pulled after
the evening session belongs to a later, short launch, so that session's crash is not in the data.

## Session on the vendor driver

gamescope and the Steam client can run on Blobferry too (experimental; DroidDeck's setting sets all of this up):
- `MALI_COMPAT_DRM_NODE=/dev/mali0`: gamescope refuses a Vulkan device without DRM nodes;
- `MALI_COMPAT_DMABUF_EXPORT=1`: gamescope's nested backend hands its output images to the parent compositor as
  dma-bufs, which the driver can import but not export;
- `MALI_COMPAT_GS_HIDE_PROCS=Xwayland`: Xwayland's glamor (on Mesa Zink) then draws its quads without geometry
  shaders (an idea from [MaliBridge](https://github.com/ASEbrahim/MaliBridge), see [NOTICE.md](NOTICE.md));
- Steam with `-cef-disable-gpu` (menus in software); with GPU menus it is untested.

`test/session-probe.c` lists what gamescope asks of the driver, `test/export-probe.c` checks the dma-buf export.
Known open item: Steam's GPU topology query fails on the session (`failed create vulkan instance: -7`), harmless for
the UI.

## Building

The scripts cross-build for aarch64 glibc with clang. For now they expect the development layout this project
came from:
- a sibling checkout of [panvk-kbase-android](https://github.com/zenithblue-oss/panvk-kbase-android) with its Debian
  bookworm arm64 sysroot, Mesa headers and host wayland-scanner under `../panvk-kbase-android/work/`;
- the Android NDK 29 (`ld.lld`) under `~/Android/Sdk/ndk/29.0.14206865`.

```sh
git submodule update --init
./build-libhybris.sh configure && ./build-libhybris.sh make && ./build-libhybris.sh install
./build-wsi.sh          # exports the submodule, applies patches/vulkan-wsi-layer, builds the WSI layer
./build-bridge.sh build # ICD shim, compat layer, test probes
./build-bridge.sh zip   # build/mali_bridge/Mali_Bridge.zip for DroidDeck's driver import page
```

`build-bridge.sh install-driver` and `deploy` push the results to a connected phone with adb. They leave a `.dev`
marker so that a DroidDeck build that bundles Blobferry keeps the deployed files; delete
`usr/lib/mali_bridge/.dev` (runtime) and `linux_vulkan_drivers/Mali_Bridge/.dev` (app files) to go back to the bundled
build. `./package.sh` packs the built files into `build/dist/blobferry-<commit>.zip` with an adb install script
(`install-adb.sh`), the licenses, `SOURCE.txt` (the commit, the submodule commits and the patches) and `MD5SUMS`;
`BLOBFERRY_VERSION=<name> ./package.sh` names a release and refuses a dirty tree. The build scripts never change the
submodules: `build-libhybris.sh configure` and `build-wsi.sh` build an export of the submodule commit with the
`patches/` applied, so a clean checkout rebuilds the released binaries bit for bit (the WSI build maps its source
paths with `-ffile-prefix-map`, see `build-wsi.sh`). DroidDeck's `tools/build_local.sh` bundles that package
(`DROIDDECK_BLOBFERRY=<zip or folder>`).

## Switches

Environment variables, read when a process creates its Vulkan instance. Unset or `0` means off unless noted.
`<names>` is a comma-separated list of process-name prefixes (`/proc/self/comm`).

**Games:**

| Switch | Effect |
|---|---|
| `MALI_COMPAT_FRAGDEPTH_FIX=0` | turn off the FragDepth fix (on by default) |
| `MALI_COMPAT_BC_AFRC=0` | uncompressed BC shadow images (needs much more GPU memory) |
| `MALI_COMPAT_RT_COMPRESS=1` | render-target compression; learned UAV targets in `/root/mali-rt-keep-<process>.txt` |
| `MALI_COMPAT_XFB_VS=1` | D3D11 stream output: a passthrough capture geometry shader is dropped and the vertex shader captures (Unity's GPU skinning; no effect in The Long Dark, Unity 6 skins in compute: 0 stream-output pipelines in its log) |
| `MALI_COMPAT_NOATT_AREA=<W>x<H>` | clamp DXVK's attachment-less passes (16384x16384, 256 layers) to this area and one layer; they hung the GPU in The Witcher 3 (`4096x4096`) |
| `MALI_COMPAT_PIPELINE_CACHE=1` | persistent pipeline cache per process in `/root/.cache/mali-compat/<process>.pcache` (the driver keeps none under libhybris); fine with DXVK, gave wrong pipelines with vkd3d-proton (HZD). `=2`: in memory only, one cache per compile thread |
| `MALI_COMPAT_PIPELINE_CACHE_THREADS=<names>` | the threads that use that cache (default `AsyncPso`, `*` = all; `dxvk-cs,MainThread` for DXVK) |
| `MALI_COMPAT_ASYNC_PSO=<ms>` | compile graphics pipelines on worker threads and wait at most `<ms>`; a slower one returns a stand-in whose draws are dropped until it is ready (pop-in instead of a stall) |
| `MALI_COMPAT_ASYNC_THREADS=<n>` | worker threads for `ASYNC_PSO` (default 2) |
| `MALI_COMPAT_VENDOR_ID=<hex>` | report another PCI vendor ID (driverID stays Arm); `5143` makes vkd3d-proton use push descriptors for root CBVs |
| `MALI_COMPAT_SPV_FIX=1` | two SPIR-V rewrites against vendor-compiler aborts (u32→u64→float conversions, uniform `BroadcastFirst`); for Control the vendor-compiler abort stays with it (2026-10-30); Control runs with a stub shader override instead, see [Control](#control) |
| `MALI_COMPAT_SPV_DONTUNROLL=1` | mark every loop DontUnroll (experiment against slow compiles) |
| `MALI_COMPAT_NO_SUSPEND_RESUME=1` | turn vkd3d-proton's suspended/resumed render passes into plain passes (fixed HZD's blocky sky, needed by Control); installs the render-pass hooks itself, no `GPU_TIME` needed and no timing work without it |
| `MALI_COMPAT_SUSRES_BARRIER=1` | with `NO_SUSPEND_RESUME` only: one memory barrier (attachment write → load/test/sample) before each converted resuming part, so the split parts are ordered as the spec requires (K-CS1, in the layer `bridge/build/stage-old+decouple+kcs1/`). Tried on Control's shadow flicker at the security booth (2026-10-31 13:4x, barrier before 188064 of 188064 resumed parts): the flicker stayed (user report), so it is not the fix |
| `MALI_COMPAT_WLS_CLAMP=<hashes>\|all` | clamp every dynamic index into a Workgroup-storage array (DXVK's TGSM) with `UMin(index, length-1)` before the access chain; in-range indices are unchanged, an out-of-range one (undefined behaviour in D3D11/Vulkan) stays inside the array. On Mali, workgroup memory is ordinary GPU address space, so such an index walked into neighbouring memory and caused the vendor driver's GPU MMU translation faults. `all` or `1`: every shader module; else a comma list of hex hashes (FNV-1a of the SPIR-V as the app passed it, the `GPU_TIME` / `DUMP_HASH` hash). Prints `wls-clamp-1 active` once per device and one `wls-clamp: module <hash>` line per clamped module. **Watch Dogs 2:** `MALI_COMPAT_WLS_CLAMP=e0ff8b4d,46aad60b,653351e8,ae9a9376,92ae90c3,25785cbd` (the compute shaders that faulted); run together with the vendor's serialized-submissions debug option, the protagonist's head is back (user report, 2026-11-01), ~40 FPS at the start falling to ~15, ~8 in firefights (user, no meter); the clamp alone has not been run on Watch Dogs 2. **Watch Dogs:** `MALI_COMPAT_WLS_CLAMP=all` (with the current Ubisoft Connect in the prefix); starts and reaches gameplay, 23.7-31.9 FPS (mean 26.8) in the prologue over 60 s on the DXVK HUD (2026-11-01 22:52, thermal 4); without it GPU translation faults from the first frame, flashing stale frames and a load hang |
| `MALI_COMPAT_FORCE_COMP=1` | give attachments without STORAGE their compression back when vkd3d-proton asks for none (experiment) |
| `MALI_COMPAT_DEPTH_NOCOMP=1` | create sampled depth/stencil images uncompressed (experiment) |
| `MALI_COMPAT_FORCE_COHERENT=1` | allocations from the non-coherent cached host memory type go to the coherent one of the same heap (experiment). Control: in the env of the 2026-11-01 17:5x run, which froze at the security gate before the flicker room, so its effect on the flicker is untested; HZD's env has it as a default (no change in its missing static meshes) |
| `MALI_COMPAT_FAKE_OCCLUSION=1` | every occlusion query reads as visible (experiment) |
| `MALI_COMPAT_BUDGET_MB=<MiB>` | `VK_EXT_memory_budget` with live accounting: each heap's `heapUsage` is its live usage (added on allocate, subtracted on free) and `heapBudget` is capped at this size. 0 or unset = the default report: `heapBudget` = min(heap, 6 GiB) and every heap's `heapUsage` = the bytes allocated so far (never decreases). Logged once at init |
| `MALI_COMPAT_BUDGET_LOG_S=<s>` | with `BUDGET_MB` only: interval of the budget log line `mali-compat: budget heap0 <used>/<budget> MiB (peak) ..., <n> allocs, <q> queries` (default 30 s); printed only with `MALI_COMPAT_GPU_TIME=1` or `MALI_COMPAT_DEBUG` |
| `MALI_COMPAT_MEMLOG=1` | memory event log `/root/mali-memlog-<comm>.tsv`, one TSV line per allocate / map / unmap / free / failed allocate and every 100th queue submit (`=2`: without the opaque-capture-address query); buffered, written once per second (diagnostic) |
| `MALI_COMPAT_MEMLOG_MB=<MiB>` | size cap of that log per process (default 64 MiB), then one `TRUNC` line and the log stops |
| `MALI_WSI_ASYNC_PRESENT=1` | WSI layer only, from `patches/vulkan-wsi-layer/0003-*.patch` (in the shipped layer since 2026-11-02, off by default; measured on Control: +8 % fps, +22 % fps per GPU GHz, plan control-0410 r2). Also uses the presentation thread for swapchains without explicit sync, so `vkQueuePresentKHR` returns after the fence submit instead of waiting for the frame's GPU work on the app's present thread; set (0 or 1) it prints one `mali-wsi: async present:` line per swapchain |

**Session** (see [Session on the vendor driver](#session-on-the-vendor-driver)):

| Switch | Effect |
|---|---|
| `MALI_COMPAT_DRM_NODE=<node>` | report `VK_EXT_physical_device_drm` with this character device (e.g. `/dev/mali0`) as primary and render node |
| `MALI_COMPAT_DMABUF_EXPORT=1` | emulate dma-buf export: exportable linear images get memory from a dma-heap, imported into the driver; an allocation that imports a dma-buf keeps its import |
| `MALI_COMPAT_DMABUF_HEAP=<path>` | the dma-heap for that (default `/dev/dma_heap/system`) |
| `MALI_COMPAT_GS_HIDE_PROCS=<names>` | hide `geometryShader` from these processes (`Xwayland`) |
| `MALI_COMPAT_FP16_HIDE_PROCS=<names>` | hide `shaderFloat16`/`shaderInt16` from these processes (diagnostic) |
| `MALI_COMPAT_DMABUF_PEEK=1` | every 5 s, how much of each exported buffer is non-zero; logs external image creations (diagnostic) |

**Debugging:**

| Switch | Effect |
|---|---|
| `MALI_BRIDGE_DEBUG=1` | ICD shim logs to stderr |
| `MALI_BRIDGE_HAL=<path>` | load another HAL instead of `/vendor/lib64/hw/vulkan.mali.so` |
| `MALI_COMPAT_DEBUG=1`, `MALI_COMPAT_STATS=1` | compat layer logging and counters (`STATS=every`: report on every BC upload instead of every 5 s) |
| `MALI_WSI_NO_10BIT=1` | hide 10-bit surface formats |
| `MALI_COMPAT_GPU_TIME=1` | GPU profiler: top passes/dispatches every 5 s, compile times, and STUCK lines for hangs in `/root/mali-gputime-<thread>.txt` |
| `MALI_COMPAT_GT_TRACE=<hash>` | with the profiler: log every pass that uses the shader with this hash (`all`: every pass and dispatch) |
| `MALI_COMPAT_WAIT_CALLERS=1` | with the profiler: `wait caller:` lines in each 5-s report, one per (waiting thread, caller lib+offset, fence kind `cmd`/`nocb-wait`/`nocb`/`untracked`/`timeline`, submitting thread): count, ms and % of wall, mean/max, fence age (wait start minus submit) and a duration histogram for `vkWaitForFences`/`vkWaitSemaphores`. Acquire/present do not reach this layer (the WSI layer above implements them; its own fence waits show under its threads; `MALI_WSI_PACE=1` times acquire/flip). Read once at instance creation; without it the code runs no extra work |
| `MALI_COMPAT_SKIP_HASH=<hash>` | with the profiler: drop the draws of pipelines with this shader (finds the draw that makes a pass slow) |
| `MALI_COMPAT_DUMP_HASH=<hash>[,..]` | write shaders with these hashes to `/root/mali-spv-<hash>.spv` |
| `MALI_COMPAT_DESC_CAPTURE=<cs hashes>` | debug only: for every dispatch of the compute pipelines with these shader hashes (hex, comma list, the `GPU_TIME` `cs:` hash), write the resolved set-0 bindings (buffer device address, view range/format, buffer size, memory) and the object lifetimes (destroy/free/reset, submit, fence wait) as TSV rows to `MALI_COMPAT_DESC_CAPTURE_FILE` (default `/root/mali-desc-<pid>.tsv`), capped at `MALI_COMPAT_DESC_CAPTURE_MB` (default 512); counters every 10 s on stderr. Costs time and disk, off by default |

## Repository layout

- `src/icd/`: ICD shim
- `src/compat/`: compat layer; `bc/` holds the BC decode shaders and their generated SPIR-V headers
- `patches/vulkan-wsi-layer/`: changes to the Arm WSI layer (`*.patch` are applied by `build-wsi.sh` in name order, `*-staged` variants are held back)
- `patches/libhybris/`: change to libhybris (race-free lazy init of static bionic locks), see [NOTICE.md](NOTICE.md)
- `patches/wine/`: a source patch for Wine's ntdll (async I/O status block written after its memory was freed, seen with Watch Dogs 2); written without a Wine tree at hand, not built or tested, not applied by anything here
- `test/`: small Vulkan probes (incl. `rt-comp-probe.c` for render-target compression, `xfb-probe.c` for stream
  output, `session-probe.c` and `export-probe.c` for the session) and SPIR-V patch tests (`spirv-patch-check.sh`,
  `fragdepth-host.c`, `xfb-vs-host.c`)
- `third_party/`: libhybris, vulkan-wsi-layer and android-headers as git submodules

## How it was made

Most of the code, the tests and the debugging were done by AI coding agents, directed and tested on the phone by
zolotvok: Claude Opus 5.5, Fable 5.1 and Claude Sonnet 5.5 (Anthropic), Gemini 3.8 Flash and Gemini 4 Argon (Google,
the latter for reviews). Code comments carry a `[model]` tag where an agent wrote or changed something.

## License

Blobferry's own code is licensed under the GNU General Public License, version 3 or (at your option) any later version
(GPL-3.0-or-later), see [LICENSE](LICENSE). Versions up to and including commit 4077514 (the last pushed before the
relicense) remain available under the MIT license. Third-party parts keep their own licenses, and credits are listed in
[NOTICE.md](NOTICE.md). Blobferry ships no vendor blobs: the proprietary Mali driver stays on the phone and is loaded from
there at run time.

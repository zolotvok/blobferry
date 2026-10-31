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

In short (2026-10-30):

| Game | API | Result |
|---|---|---|
| The Elder Scrolls IV: Oblivion | DX9 → DXVK | in game, ~89 FPS; also runs with the session itself on Blobferry |
| Tomb Raider (2013) | DX11 → DXVK | benchmark avg 54 FPS |
| Yooka-Laylee and the Impossible Lair | DX11 → DXVK (Unity) | in game, 80-90 FPS with `MALI_COMPAT_XFB_VS=1` (19-20 without) |
| The Witcher 3 (classic) | DX11 → DXVK | in game, 41-48 FPS indoors, 19-29 outdoors |
| The Witcher 3 (current release) | DX11 → DXVK | in game, ~18-22 FPS at low settings; with `NOATT_AREA` (GPU hang otherwise), `PIPELINE_CACHE` and `ASYNC_PSO` (compile stalls) |
| The Witcher 3 (current release) | DX12 → vkd3d-proton | in game, ~17-22 FPS; the prologue runs, one GPU hang left later in the game |
| Horizon Zero Dawn | DX12 → vkd3d-proton | benchmark completes (avg 14-16 FPS); the static world geometry (buildings, props, cliffs) is missing |
| Control | DX12 → vkd3d-proton | starts, intro and 3D visible, plays at ~12-13 FPS (autoplay 3 min, exit 0). Recipe: `MALI_COMPAT_NO_SUSPEND_RESUME=1` with one of `GPU_TIME`/`ASYNC_PSO`/`NOATT_AREA`/`GT_TRACE` on (its hooks do the work), plus `FEX_DISKCACHE=0` in the guest; the compute shader that aborted the vendor compiler is replaced through `VKD3D_SHADER_OVERRIDE` |

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
./build-wsi.sh          # applies patches/vulkan-wsi-layer, builds the WSI layer
./build-bridge.sh build # ICD shim, compat layer, test probes
./build-bridge.sh zip   # build/mali_bridge/Mali_Bridge.zip for DroidDeck's driver import page
```

`build-bridge.sh install-driver` and `deploy` push the results to a connected phone with adb. They leave a `.dev`
marker so that a DroidDeck build that bundles Blobferry keeps the deployed files; delete
`usr/lib/mali_bridge/.dev` (runtime) and `linux_vulkan_drivers/Mali_Bridge/.dev` (app files) to go back to the bundled
build. `./package.sh` packs the built files into `build/dist/blobferry-<commit>.zip` with an adb install script
(`install-adb.sh`) and the licenses. DroidDeck's `tools/build_local.sh` bundles that package
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
| `MALI_COMPAT_XFB_VS=1` | D3D11 stream output: a passthrough capture geometry shader is dropped and the vertex shader captures (Unity's GPU skinning) |
| `MALI_COMPAT_NOATT_AREA=<W>x<H>` | clamp DXVK's attachment-less passes (16384x16384, 256 layers) to this area and one layer; they hung the GPU in The Witcher 3 (`4096x4096`) |
| `MALI_COMPAT_PIPELINE_CACHE=1` | persistent pipeline cache per process in `/root/.cache/mali-compat/<process>.pcache` (the driver keeps none under libhybris); fine with DXVK, gave wrong pipelines with vkd3d-proton (HZD). `=2`: in memory only, one cache per compile thread |
| `MALI_COMPAT_PIPELINE_CACHE_THREADS=<names>` | the threads that use that cache (default `AsyncPso`, `*` = all; `dxvk-cs,MainThread` for DXVK) |
| `MALI_COMPAT_ASYNC_PSO=<ms>` | compile graphics pipelines on worker threads and wait at most `<ms>`; a slower one returns a stand-in whose draws are dropped until it is ready (pop-in instead of a stall) |
| `MALI_COMPAT_ASYNC_THREADS=<n>` | worker threads for `ASYNC_PSO` (default 2) |
| `MALI_COMPAT_VENDOR_ID=<hex>` | report another PCI vendor ID (driverID stays Arm); `5143` makes vkd3d-proton use push descriptors for root CBVs |
| `MALI_COMPAT_SPV_FIX=1` | two SPIR-V rewrites against vendor-compiler aborts (u32→u64→float conversions, uniform `BroadcastFirst`); for Control, not enough yet |
| `MALI_COMPAT_SPV_DONTUNROLL=1` | mark every loop DontUnroll (experiment against slow compiles) |
| `MALI_COMPAT_NO_SUSPEND_RESUME=1` | turn vkd3d-proton's suspended/resumed render passes into plain passes (fixed HZD's blocky sky); uses the hooks of `GPU_TIME`, `ASYNC_PSO` or `NOATT_AREA`, so one of those must be on |
| `MALI_COMPAT_FORCE_COMP=1` | give attachments without STORAGE their compression back when vkd3d-proton asks for none (experiment) |
| `MALI_COMPAT_DEPTH_NOCOMP=1` | create sampled depth/stencil images uncompressed (experiment) |
| `MALI_COMPAT_FORCE_COHERENT=1` | allocations from the non-coherent cached host memory type go to the coherent one of the same heap (experiment) |
| `MALI_COMPAT_FAKE_OCCLUSION=1` | every occlusion query reads as visible (experiment) |

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
| `MALI_COMPAT_SKIP_HASH=<hash>` | with the profiler: drop the draws of pipelines with this shader (finds the draw that makes a pass slow) |
| `MALI_COMPAT_DUMP_HASH=<hash>[,..]` | write shaders with these hashes to `/root/mali-spv-<hash>.spv` |

## Repository layout

- `src/icd/`: ICD shim
- `src/compat/`: compat layer; `bc/` holds the BC decode shaders and their generated SPIR-V headers
- `patches/vulkan-wsi-layer/`: changes to the Arm WSI layer
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

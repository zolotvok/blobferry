# Blobferry

**Run Windows games on an Android phone's own Mali Vulkan driver, from a Linux (glibc) userspace.**

> **Experimental.** Tested on one phone only (vivo X300 Pro, Mali-G1-Ultra MC12, vendor driver `v1.r54p1`). Expect
> rendering bugs, hangs and crashes. Nothing here is endorsed by Arm, the phone vendor or the DroidDeck project.

Blobferry ferries Vulkan calls from Linux programs (Proton/Wine with DXVK or vkd3d-proton, under FEX) to the
proprietary Mali driver ("the blob") that ships on the phone, instead of an open-source driver such as PanVK. It was
built for [DroidDeck](https://github.com/Droid-Deck/DroidDeck), a SteamOS-like app for Android, and is used through
the [zolotvok/DroidDeck](https://github.com/zolotvok/DroidDeck) fork (branch `mali-panvk`): there Steam and gamescope
stay on PanVK and only the games run on the vendor driver.

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
- a built-in GPU profiler: per render pass / dispatch GPU time, and the pass a GPU hang is stuck in.

The layer is implicit and turns on only with `ENABLE_MALI_COMPAT=1` (the DroidDeck fork sets it for bridge games).

## Status

In short (2026-10-27):

| Game | API | Result |
|---|---|---|
| The Elder Scrolls IV: Oblivion | DX9 → DXVK | in game, ~89 FPS |
| Tomb Raider (2013) | DX11 → DXVK | benchmark avg 54 FPS |
| The Witcher 3 (classic) | DX11 → DXVK | in game, ~22 FPS |
| Horizon Zero Dawn | DX12 → vkd3d-proton | benchmark completes (avg 13 FPS); much of the world geometry is missing |
| The Witcher 3 (current release) | DX12 → vkd3d-proton | in game, ~20-24 FPS; the prologue runs, one GPU hang left later in the game |

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

## Debug switches

| Switch | Effect |
|---|---|
| `MALI_BRIDGE_DEBUG=1` | ICD shim logs to stderr |
| `MALI_BRIDGE_HAL=<path>` | load another HAL instead of `/vendor/lib64/hw/vulkan.mali.so` |
| `MALI_COMPAT_DEBUG=1`, `MALI_COMPAT_STATS=1` | compat layer logging and counters |
| `MALI_COMPAT_BC_AFRC=0` | uncompressed BC shadow images (needs much more GPU memory) |
| `MALI_WSI_NO_10BIT=1` | hide 10-bit surface formats |
| `MALI_COMPAT_FRAGDEPTH_FIX=0` | turn off the FragDepth fix (on by default) |
| `MALI_COMPAT_RT_COMPRESS=1` | render-target compression; learned UAV targets in `/root/mali-rt-keep-<process>.txt` |
| `MALI_COMPAT_GPU_TIME=1` | GPU profiler: top passes/dispatches every 5 s and STUCK lines for hangs in `/root/mali-gputime-<thread>.txt` |
| `MALI_COMPAT_GT_TRACE=<hash>` | with the profiler: log every pass that uses the shader with this hash |
| `MALI_COMPAT_DUMP_HASH=<hash>[,..]` | write shaders with these hashes to `/root/mali-spv-<hash>.spv` |
| `MALI_COMPAT_DEPTH_NOCOMP=1` | create sampled depth/stencil images uncompressed (experiment) |

## Repository layout

- `src/icd/`: ICD shim
- `src/compat/`: compat layer; `bc/` holds the BC decode shaders and their generated SPIR-V headers
- `patches/vulkan-wsi-layer/`: changes to the Arm WSI layer
- `test/`: small Vulkan probes (incl. `rt-comp-probe.c` for render-target compression) and SPIR-V patch tests
  (`spirv-patch-check.sh`, `fragdepth-host.c`)
- `third_party/`: libhybris, vulkan-wsi-layer and android-headers as git submodules

## How it was made

Most of the code, the tests and the debugging were done by AI coding agents, directed and tested on the phone by
zolotvok: Claude Opus 5.5 and Claude Sonnet 5.5 (Anthropic), Gemini 3.8 Flash and Gemini 4 Argon (Google, the latter
for reviews). Code comments carry a `[model]` tag where an agent wrote or changed something.

## License

MIT, see [LICENSE](LICENSE). Third-party parts and credits are listed in [NOTICE.md](NOTICE.md).

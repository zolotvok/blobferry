# NOTICE

Blobferry's own code (the ICD shim, the compat layer, build scripts, tests and docs) is MIT-licensed, see `LICENSE`.
The parts below come from other projects and keep their own licenses.

## BC1-BC7 decode shaders (`src/compat/bc/`)

`panvk_bc_{s3tc,rgtc,bc6,bc7}.comp` and the SPIR-V headers generated from them come from the PanVK BC decoders of
[panvk-kbase-android](https://github.com/zenithblue-oss/panvk-kbase-android) (Mesa patches, MIT). Blobferry changes a
few lines per shader (4-15). `gen_spv.sh` regenerates the `*_spv.h` headers with glslangValidator.

## Arm vulkan-wsi-layer (`third_party/vulkan-wsi-layer`, git submodule)

Copyright (c) Arm Limited, MIT, from <https://gitlab.freedesktop.org/mesa/vulkan-wsi-layer>. The submodule stays
at the upstream commit; Blobferry's changes are in `patches/vulkan-wsi-layer/` (MIT, like the layer).

## libhybris (`third_party/libhybris`, git submodule)

The [wmww fork](https://github.com/wmww/libhybris), used unchanged. Mixed licenses (Apache-2.0, LGPL-2.1 and
others, see the `LICENSE.*` files in the submodule). Binaries built from it must come with its source or a pointer to
the exact commit.

## android-headers (`third_party/android-headers`, git submodule)

From [Halium](https://github.com/Halium/android-headers): AOSP headers, mostly Apache-2.0 (see each file's header),
used unchanged as build headers for libhybris.

## Not included

The vendor Mali Vulkan driver (`/vendor/lib64/hw/vulkan.mali.so`) is proprietary. Blobferry loads it from the
phone at run time and never ships, copies or redistributes it.

## Prior art

Running the phone's vendor Vulkan driver from a glibc Linux guest through libhybris is what tawc and
[ARLinux](https://github.com/taowen/arlinux-rootfs) do. Blobferry builds on that idea with its own ICD shim and a
compatibility layer for DXVK and vkd3d-proton on Mali.

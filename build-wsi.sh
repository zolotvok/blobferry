#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-or-later
# Builds the Vulkan WSI layer with Wayland and dma_buf_heaps allocator (system heap)
# for aarch64 glibc using clang and the bookworm sysroot. [Gemini 3.8 Flash] 2026-10-26
set -eu

B=$(cd "$(dirname "$0")" && pwd)
SR=$B/../panvk-kbase-android/work/sysroot-bookworm-arm64
NDK=$HOME/Android/Sdk/ndk/29.0.14206865/toolchains/llvm/prebuilt/linux-x86_64/bin
GCCDIR=$(ls -d "$SR"/usr/lib/gcc/aarch64-linux-gnu/*/ | head -1)
SCANNER=$B/../panvk-kbase-android/work/host-wayland/bin/wayland-scanner
PROTOCOLS=$B/../panvk-kbase-android/work/mesa/subprojects/wayland-protocols-1.41
SRC=$B/third_party/vulkan-wsi-layer
BUILDDIR=$B/build/vulkan-wsi-layer
STAGE=$B/build/mali_bridge/stage/usr/lib/mali_bridge

mkdir -p "$BUILDDIR"

# Our changes live in patches/vulkan-wsi-layer; the submodule stays at the upstream commit. Apply what is not applied yet.
for p in "$B"/patches/vulkan-wsi-layer/*.patch; do
  git -C "$SRC" apply --reverse --check "$p" 2>/dev/null || git -C "$SRC" apply "$p"
done

# Set up pkg-config environment for cross compilation
export PKG_CONFIG_DIR=
export PKG_CONFIG_LIBDIR="$SR/usr/lib/aarch64-linux-gnu/pkgconfig:$SR/usr/share/pkgconfig:$SR/usr/local/lib/pkgconfig"
export PKG_CONFIG_SYSROOT_DIR="$SR"

FLAGS="--target=aarch64-linux-gnu --sysroot=$SR --gcc-install-dir=$GCCDIR -Wno-error=unused-command-line-argument"
LINK_FLAGS="--target=aarch64-linux-gnu --sysroot=$SR --gcc-install-dir=$GCCDIR --ld-path=$NDK/ld.lld -Wl,-rpath,/usr/lib/aarch64-linux-gnu:/usr/local/lib -Wl,-z,nodelete"

cmake -S "$SRC" -B "$BUILDDIR" -G Ninja \
  -DCMAKE_SYSTEM_NAME=Linux \
  -DCMAKE_SYSTEM_PROCESSOR=aarch64 \
  -DCMAKE_C_COMPILER=clang \
  -DCMAKE_CXX_COMPILER=clang++ \
  -DCMAKE_C_FLAGS="$FLAGS -Wno-error" \
  -DCMAKE_CXX_FLAGS="$FLAGS -Wno-error" \
  -DCMAKE_SHARED_LINKER_FLAGS="$LINK_FLAGS" \
  -DCMAKE_EXE_LINKER_FLAGS="$LINK_FLAGS" \
  -DVULKAN_CXX_INCLUDE="$B/../panvk-kbase-android/work/mesa/include" \
  -DKERNEL_HEADER_DIR="$SR/usr/include" \
  -DBUILD_WSI_HEADLESS=0 \
  -DBUILD_WSI_WAYLAND=1 \
  -DBUILD_WSI_DISPLAY=0 \
  -DSELECT_EXTERNAL_ALLOCATOR=dma_buf_heaps \
  -DWSIALLOC_MEMORY_HEAP_NAME="system" \
  -DWAYLAND_SCANNER_EXEC="$SCANNER" \
  -DWAYLAND_PROTOCOLS_DIR="$PROTOCOLS" \
  -DENABLE_WAYLAND_FIFO_PRESENTATION_THREAD=ON   # [Claude Opus 5.5] present_image (and its CPU wait for the GPU when gamescope offers no explicit sync) off the app thread

ninja -C "$BUILDDIR"

mkdir -p "$STAGE"
cp "$BUILDDIR/libVkLayer_window_system_integration.so" "$STAGE/"
# Fix JSON library path to absolute path in stage and adjust api_version + enable_environment
OUT=$B/build/mali_bridge
IMPLICIT_DIR="$OUT/stage/usr/share/vulkan/implicit_layer.d"
mkdir -p "$IMPLICIT_DIR"
sed -e 's|"library_path": ".*"|"library_path": "/usr/lib/mali_bridge/libVkLayer_window_system_integration.so"|' \
    -e 's|"api_version": ".*"|"api_version": "1.3.0"|' \
    -e 's|"type": "GLOBAL"|"type": "GLOBAL",\n        "enable_environment": { "ENABLE_MALI_WSI": "1" },\n        "disable_environment": { "DISABLE_MALI_WSI": "1" }|' \
  "$BUILDDIR/VkLayer_window_system_integration.json" > "$STAGE/VkLayer_window_system_integration.json"

cp "$STAGE/VkLayer_window_system_integration.json" "$IMPLICIT_DIR/VkLayer_window_system_integration.json"

echo "WSI layer built successfully: $STAGE/libVkLayer_window_system_integration.so"

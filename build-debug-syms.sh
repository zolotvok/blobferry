#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-or-later
# Debug-info twins of the shipped Blobferry binaries, for symbolizing simpleperf data down to source lines.
# [keeper, Claude Opus 5.5] 2026-11-05
#
# Builds compat layer, ICD (pinned git revs, see COMPAT_REV/ICD_REV), WSI layer, libhybris-common and the q linker
# plugin (submodule HEAD + patches, as the release scripts do) with the SAME
# flags as build-bridge.sh / build-wsi.sh / build-libhybris.sh plus -g, into build/debug-syms/ (the normal build dirs
# and the stage are not touched). -g does not change the generated code, so the debug twin has the same .text/.rodata/
# .data as the shipped file; the script checks that section by section against the reference files (the shipped
# 1102 package, md5s below) and writes build/debug-syms/MANIFEST.tsv: name, shipped md5, shipped build-id, debug file,
# verdict (SAME-CODE = the twin's line tables are valid for the shipped binary's addresses).
# The shipped files themselves are not stripped (.symtab present), so function-level symbols work without this;
# the twins only add file:line and inlined frames.
#
# Usage: bridge/build-debug-syms.sh [all|compat|icd|wsi|hybris|check]      (default all; logs in build/debug-syms/*.log)
set -eu
B=$(cd "$(dirname "$0")" && pwd)
SR=$B/../panvk-kbase-android/work/sysroot-bookworm-arm64
NDK=$HOME/Android/Sdk/ndk/29.0.14206865/toolchains/llvm/prebuilt/linux-x86_64/bin
GCCDIR=$(ls -d "$SR"/usr/lib/gcc/aarch64-linux-gnu/*/ | head -1)
T="--target=aarch64-linux-gnu --sysroot=$SR --gcc-install-dir=$GCCDIR --ld-path=$NDK/ld.lld"
D=$B/build/debug-syms
REF=$B/build/dist/blobferry-65652c9-phone1102/rootfs/usr/lib   # the package the phone runs (md5s checked below)
mkdir -p "$D"

# name | shipped md5 (prefix) | reference file | debug twin
REFS="compat|950546af|$REF/mali_bridge/libVkLayer_droiddeck_mali_compat.so|$D/libVkLayer_droiddeck_mali_compat.so
icd|caf6c3c0|$REF/mali_bridge/libvulkan_mali_bridge.so|$D/libvulkan_mali_bridge.so
wsi|16e16c96|$REF/mali_bridge/libVkLayer_window_system_integration.so|$D/libVkLayer_window_system_integration.so
hybris|5206ca54|$REF/hybris/libhybris-common.so.1.0.0|$D/libhybris-common.so.1.0.0
q|4eb5242f|$REF/hybris/libhybris/linker/q.so|$D/q.so"

# compat and ICD are built from pinned git revisions of bridge/ (the working tree moves on: K-WC1 changed the compat
# source on 11-05 10:12). dfd12f8 = "source of the shipped layer (950546af)", 08b9b13 = "the shipped ICD caf6c3c0".
COMPAT_REV=${COMPAT_REV:-dfd12f8}
ICD_REV=${ICD_REV:-08b9b13}
export_rev() { # rev path -> $D/src-<rev>/<path>
  local dst=$D/src-$1
  rm -rf "$dst/$2" && mkdir -p "$dst"
  git -C "$B" archive --format=tar "$1" "$2" | tar -x -C "$dst"
  echo "$dst/$2"
}
build_compat() {
  local S
  S=$(export_rev "$COMPAT_REV" src/compat)
  clang $T -O2 -g -Wall -fPIC -shared -fvisibility=hidden -Wl,-soname,libVkLayer_droiddeck_mali_compat.so -Wl,-Bsymbolic \
    -Wl,--no-undefined -Wl,-z,nodelete -I"$S" -I"$B/../panvk-kbase-android/work/mesa/include" \
    "$S/mali_compat_layer.c" -o "$D/libVkLayer_droiddeck_mali_compat.so" -lpthread
}

build_icd() {
  local S
  S=$(export_rev "$ICD_REV" src/icd)
  clang $T -O2 -g -Wall -fPIC -shared -fvisibility=hidden -Wl,-soname,libvulkan_mali_bridge.so -Wl,-Bsymbolic \
    -Wl,--no-undefined -Wl,-z,nodelete "$S/mali_bridge_icd.c" -o "$D/libvulkan_mali_bridge.so" \
    -L"$B/build/libhybris-install/usr/lib/hybris" -lhybris-common -Wl,-rpath,/usr/lib/hybris
}

# Same steps as build-wsi.sh (fresh export of the submodule HEAD + patches/vulkan-wsi-layer/*.patch, same prefix maps,
# Release with plain -O2), with -g added; own work dir.
build_wsi() {
  local SUB=$B/third_party/vulkan-wsi-layer SRC=$D/wsi/src BLD=$D/wsi/build
  rm -rf "$D/wsi" && mkdir -p "$SRC" "$BLD"
  git -C "$SUB" archive --format=tar HEAD | tar -x -C "$SRC"
  for p in "$B"/patches/vulkan-wsi-layer/*.patch; do patch -p1 -s -N --no-backup-if-mismatch -d "$SRC" < "$p"; done
  export PKG_CONFIG_DIR= PKG_CONFIG_SYSROOT_DIR="$SR"
  export PKG_CONFIG_LIBDIR="$SR/usr/lib/aarch64-linux-gnu/pkgconfig:$SR/usr/share/pkgconfig:$SR/usr/local/lib/pkgconfig"
  local PMAP="-ffile-prefix-map=$SRC=/tmp/blobferry-wsi-async-rel/src -ffile-prefix-map=$BLD=/tmp/blobferry-wsi-async-rel/build"
  local FLAGS="--target=aarch64-linux-gnu --sysroot=$SR --gcc-install-dir=$GCCDIR -Wno-error=unused-command-line-argument $PMAP"
  local LF="--target=aarch64-linux-gnu --sysroot=$SR --gcc-install-dir=$GCCDIR --ld-path=$NDK/ld.lld -Wl,-rpath,/usr/lib/aarch64-linux-gnu:/usr/local/lib -Wl,-z,nodelete"
  cmake -S "$SRC" -B "$BLD" -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_FLAGS_RELEASE="-O2 -g" -DCMAKE_CXX_FLAGS_RELEASE="-O2 -g" \
    -DCMAKE_SYSTEM_NAME=Linux -DCMAKE_SYSTEM_PROCESSOR=aarch64 -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ \
    -DCMAKE_C_FLAGS="$FLAGS -Wno-error" -DCMAKE_CXX_FLAGS="$FLAGS -Wno-error" -DCMAKE_SHARED_LINKER_FLAGS="$LF" \
    -DCMAKE_EXE_LINKER_FLAGS="$LF" -DVULKAN_CXX_INCLUDE="$B/../panvk-kbase-android/work/mesa/include" \
    -DKERNEL_HEADER_DIR="$SR/usr/include" -DBUILD_WSI_HEADLESS=0 -DBUILD_WSI_WAYLAND=1 -DBUILD_WSI_DISPLAY=0 \
    -DSELECT_EXTERNAL_ALLOCATOR=dma_buf_heaps -DWSIALLOC_MEMORY_HEAP_NAME="system" \
    -DWAYLAND_SCANNER_EXEC="$B/../panvk-kbase-android/work/host-wayland/bin/wayland-scanner" \
    -DWAYLAND_PROTOCOLS_DIR="$B/../panvk-kbase-android/work/mesa/subprojects/wayland-protocols-1.41" \
    -DENABLE_WAYLAND_FIFO_PRESENTATION_THREAD=ON
  ninja -C "$BLD"
  cp "$BLD/libVkLayer_window_system_integration.so" "$D/"
}

# Same steps as build-libhybris.sh configure + make (incl. the four objects built without _FORTIFY_SOURCE that make
# the shipped 5206ca54), with -g added; own copy dir.
build_hybris() {
  local COPY=$D/libhybris-src SRC=$D/libhybris-src/hybris
  rm -rf "$COPY" && mkdir -p "$COPY"
  git -C "$B/third_party/libhybris" archive --format=tar HEAD | tar -x -C "$COPY"
  for p in "$B"/patches/libhybris/*.patch; do [ -e "$p" ] && patch -p1 -s -N --no-backup-if-mismatch -d "$COPY" < "$p"; done
  export CC="clang $T -Wl,--no-as-needed" CXX="clang++ $T -Wl,--no-as-needed"
  export AR=$NDK/llvm-ar RANLIB=$NDK/llvm-ranlib STRIP=$NDK/llvm-strip NM=$NDK/llvm-nm
  # __FILE__ strings: the shipped build ran in build/libhybris-src; map this copy's path onto it (same .rodata size,
  # so the same .text addresses)
  local PM="-ffile-prefix-map=$COPY=$B/build/libhybris-src"
  local CF="-O2 -g $PM -D_FORTIFY_SOURCE=2 -Wno-error" CXF="-O2 -g $PM -D_FORTIFY_SOURCE=2 -Wno-error -Wno-non-pod-varargs"
  export CFLAGS="$CF" CXXFLAGS="$CXF"
  (cd "$SRC" && NOCONFIGURE=1 ./autogen.sh && ./configure --host=aarch64-linux-gnu --prefix=/usr/lib/hybris \
     --libdir=/usr/lib/hybris --with-android-headers="$B/third_party/android-headers" --enable-arch=arm64 \
     --enable-mali-quirks --enable-property-cache \
     --with-default-hybris-ld-library-path=/vendor/lib64/egl:/vendor/lib64/hw:/vendor/lib64:/system/lib64)
  local NOFORT="libhybris_common_la-tls_patcher.lo legacy_properties/libhybris_common_la-cache.lo legacy_properties/libhybris_common_la-properties.lo legacy_properties/libhybris_common_la-runtime_cache.lo"
  make -C "$SRC/common" SUBDIRS=. CFLAGS="-O2 -g $PM -Wno-error" CXXFLAGS="$CXF" $NOFORT
  make -C "$SRC/common" -j"$(nproc)" SUBDIRS=. CFLAGS="$CF" CXXFLAGS="$CXF" libhybris-common.la
  make -C "$SRC/common/q" -j"$(nproc)" CFLAGS="$CF" CXXFLAGS="$CXF"
  cp "$SRC/common/.libs/libhybris-common.so.1.0.0" "$D/"
  cp "$(find "$SRC/common/q" -name q.so -path '*.libs*' | head -1)" "$D/q.so"
}

# Same code? Compare the loaded sections byte for byte (debug sections and the build-id note may differ).
check() {
  printf 'name\tshipped_md5\tbuild_id\tdebug_file\tverdict\n' > "$D/MANIFEST.tsv"
  echo "$REFS" | while IFS='|' read -r name md5 ref dbg; do
    local got bid verdict=SAME-CODE
    got=$(md5sum "$ref" | cut -c1-8)
    bid=$("$NDK/llvm-readelf" -n "$ref" | awk '/Build ID/{print $3}')
    [ "$got" = "$md5" ] || verdict="REF-MD5-MISMATCH($got)"
    if [ ! -f "$dbg" ]; then verdict="NOT-BUILT"
    else
      for s in .text .rodata .data .data.rel.ro .plt .eh_frame; do
        if "$NDK/llvm-readelf" -S "$ref" | grep -q " $s "; then
          "$NDK/llvm-objcopy" -O binary --only-section=$s "$ref" "$D/.cmp-ref" &&
            "$NDK/llvm-objcopy" -O binary --only-section=$s "$dbg" "$D/.cmp-dbg" &&
            [ -s "$D/.cmp-ref" ] && cmp -s "$D/.cmp-ref" "$D/.cmp-dbg" && continue
          # -g can reorder a few instructions inside one function (clang scheduling noise, seen in libhybris: 1-8
          # words of ~48k). Same section size and <= 32 differing words: same layout, line tables valid within a
          # few instructions -> SAME-LAYOUT(n); anything else DIFFERS.
          local nw
          nw=$(python3 -c 'import sys;a=open(sys.argv[1],"rb").read();b=open(sys.argv[2],"rb").read()
print(-1 if len(a)!=len(b) else sum(a[i:i+4]!=b[i:i+4] for i in range(0,len(a),4)))' "$D/.cmp-ref" "$D/.cmp-dbg" 2>/dev/null || echo -1)
          if [ "$s" = .text ] && [ "$nw" -ge 0 ] && [ "$nw" -le 32 ]; then verdict="SAME-LAYOUT($nw words)"
          else verdict="DIFFERS($s)"; fi
        fi
      done
      rm -f "$D/.cmp-ref" "$D/.cmp-dbg"
      "$NDK/llvm-readelf" -S "$dbg" | grep -q ' .debug_line ' || verdict="$verdict,NO-DWARF"
    fi
    printf '%s\t%s\t%s\t%s\t%s\n' "$name" "$got" "$bid" "$dbg" "$verdict" | tee -a "$D/MANIFEST.tsv"
  done
}

case ${1:-all} in
  compat) build_compat >"$D/compat.log" 2>&1 ;;
  icd) build_icd >"$D/icd.log" 2>&1 ;;
  wsi) build_wsi >"$D/wsi.log" 2>&1 ;;
  hybris) build_hybris >"$D/hybris.log" 2>&1 ;;
  check) ;;
  all) build_compat >"$D/compat.log" 2>&1; build_icd >"$D/icd.log" 2>&1; build_wsi >"$D/wsi.log" 2>&1
       build_hybris >"$D/hybris.log" 2>&1 ;;
  *) echo "usage: $0 [all|compat|icd|wsi|hybris|check]" >&2; exit 2 ;;
esac
check

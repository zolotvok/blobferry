#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-or-later
# Minimal libhybris (common + bionic Q linker plugin) for aarch64 glibc, cross-built with clang and
# the Debian bookworm arm64 sysroot used for PanVK. [Claude Sonnet 5.5] 2026-10-26
# Usage: bridge/build-libhybris.sh configure|make|install   (log: bridge/build/libhybris-<step>.log)
set -eu
B=$(cd "$(dirname "$0")" && pwd)
SR=$B/../panvk-kbase-android/work/sysroot-bookworm-arm64
NDK=$HOME/Android/Sdk/ndk/29.0.14206865/toolchains/llvm/prebuilt/linux-x86_64/bin
SUB=$B/third_party/libhybris
COPY=$B/build/libhybris-src
SRC=$COPY/hybris
HDR=$B/third_party/android-headers
OUT=$B/build/libhybris-install
GCCDIR=$(ls -d "$SR"/usr/lib/gcc/aarch64-linux-gnu/*/ | head -1)
mkdir -p "$B/build" "$OUT"
T="--target=aarch64-linux-gnu --sysroot=$SR --gcc-install-dir=$GCCDIR --ld-path=$NDK/ld.lld"
export CC="clang $T -Wl,--no-as-needed" CXX="clang++ $T -Wl,--no-as-needed"
export AR=$NDK/llvm-ar RANLIB=$NDK/llvm-ranlib STRIP=$NDK/llvm-strip NM=$NDK/llvm-nm
export CFLAGS="-O2 -D_FORTIFY_SOURCE=2 -Wno-error" CXXFLAGS="-O2 -D_FORTIFY_SOURCE=2 -Wno-error -Wno-non-pod-varargs"
step=${1:-configure}
# Our changes live in patches/libhybris; the submodule stays clean at the fork's commit. configure starts from a fresh
# export of the submodule's committed HEAD into build/libhybris-src (uncommitted edits in the submodule are not used)
# with the *.patch files applied in name order; make and install build in that copy. [keeper, Claude Opus 5.5] 2026-11-04
if [ "$step" = configure ]; then
  rm -rf "$COPY" && mkdir -p "$COPY"
  git -C "$SUB" archive --format=tar HEAD | tar -x -C "$COPY"
  for p in "$B"/patches/libhybris/*.patch; do
    [ -e "$p" ] || continue
    patch -p1 -s -N --no-backup-if-mismatch -d "$COPY" < "$p"
  done
fi
[ -d "$SRC" ] || { echo "build-libhybris: run '$0 configure' first" >&2; exit 1; }
cd "$SRC"
case $step in
  configure)
    NOCONFIGURE=1 ./autogen.sh >"$B/build/libhybris-autogen.log" 2>&1
    ./configure --host=aarch64-linux-gnu --prefix=/usr/lib/hybris --libdir=/usr/lib/hybris \
      --with-android-headers="$HDR" --enable-arch=arm64 \
      --enable-mali-quirks --enable-property-cache \
      --with-default-hybris-ld-library-path=/vendor/lib64/egl:/vendor/lib64/hw:/vendor/lib64:/system/lib64 \
      >"$B/build/libhybris-configure.log" 2>&1
    grep -n 'HAS_ANDROID_11_0_0' config.h Makefile 2>/dev/null | head -3 ;;
  make)
    # Reproducibility: the libhybris-common in the package and on the phone (md5 5206ca54) has four objects compiled
    # without -D_FORTIFY_SOURCE=2: the first build (2026-10-26) had no fortify flag, stopped at hooks.c, and make never
    # rebuilt the objects that had compiled. These four are built first without the flag, so that a clean build gives
    # 5206ca54 bit for bit (the only difference is fprintf/snprintf/strcpy instead of the _chk variants in them).
    # BLOBFERRY_HYBRIS_FORTIFY_ALL=1 builds them with fortify as well (md5 5d5ad149, not run on a phone).
    # [keeper, Claude Opus 5.5] 2026-11-04
    NOFORT="libhybris_common_la-tls_patcher.lo legacy_properties/libhybris_common_la-cache.lo legacy_properties/libhybris_common_la-properties.lo legacy_properties/libhybris_common_la-runtime_cache.lo"
    (cd common && for o in $NOFORT; do rm -f "$o" "$(dirname "$o")/.libs/$(basename "$o" .lo).o"; done; rm -f libhybris-common.la)
    [ "${BLOBFERRY_HYBRIS_FORTIFY_ALL:-0}" = 1 ] || \
      make -C common SUBDIRS=. CFLAGS="-O2 -Wno-error" CXXFLAGS="$CXXFLAGS" $NOFORT >"$B/build/libhybris-make.log" 2>&1
    # CXXFLAGS/CFLAGS are baked into the Makefiles at configure time, so pass them on the make command line
    make -C common -j"$(nproc)" SUBDIRS=. CFLAGS="$CFLAGS" CXXFLAGS="$CXXFLAGS" libhybris-common.la >>"$B/build/libhybris-make.log" 2>&1
    make -C common/q -j"$(nproc)" CFLAGS="$CFLAGS" CXXFLAGS="$CXXFLAGS" >>"$B/build/libhybris-make.log" 2>&1 ;;
  install)
    make -C common install-libLTLIBRARIES DESTDIR="$OUT" >"$B/build/libhybris-install.log" 2>&1
    make -C common/q install DESTDIR="$OUT" >>"$B/build/libhybris-install.log" 2>&1
    find "$OUT" -type f | head -20 ;;
esac

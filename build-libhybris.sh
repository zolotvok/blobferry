#!/usr/bin/env bash
# Minimal libhybris (common + bionic Q linker plugin) for aarch64 glibc, cross-built with clang and
# the Debian bookworm arm64 sysroot used for PanVK. [Claude Sonnet 5.5] 2026-10-26
# Usage: bridge/build-libhybris.sh configure|make|install   (log: bridge/build/libhybris-<step>.log)
set -eu
B=$(cd "$(dirname "$0")" && pwd)
SR=$B/../panvk-kbase-android/work/sysroot-bookworm-arm64
NDK=$HOME/Android/Sdk/ndk/29.0.14206865/toolchains/llvm/prebuilt/linux-x86_64/bin
SRC=$B/third_party/libhybris/hybris
HDR=$B/third_party/android-headers
OUT=$B/build/libhybris-install
GCCDIR=$(ls -d "$SR"/usr/lib/gcc/aarch64-linux-gnu/*/ | head -1)
mkdir -p "$B/build" "$OUT"
T="--target=aarch64-linux-gnu --sysroot=$SR --gcc-install-dir=$GCCDIR --ld-path=$NDK/ld.lld"
export CC="clang $T -Wl,--no-as-needed" CXX="clang++ $T -Wl,--no-as-needed"
export AR=$NDK/llvm-ar RANLIB=$NDK/llvm-ranlib STRIP=$NDK/llvm-strip NM=$NDK/llvm-nm
export CFLAGS="-O2 -D_FORTIFY_SOURCE=2 -Wno-error" CXXFLAGS="-O2 -D_FORTIFY_SOURCE=2 -Wno-error -Wno-non-pod-varargs"
step=${1:-configure}
cd "$SRC"
case $step in
  configure)
    [ -x ./configure ] || NOCONFIGURE=1 ./autogen.sh >"$B/build/libhybris-autogen.log" 2>&1
    ./configure --host=aarch64-linux-gnu --prefix=/usr/lib/hybris --libdir=/usr/lib/hybris \
      --with-android-headers="$HDR" --enable-arch=arm64 \
      --enable-mali-quirks --enable-property-cache \
      --with-default-hybris-ld-library-path=/vendor/lib64/egl:/vendor/lib64/hw:/vendor/lib64:/system/lib64 \
      >"$B/build/libhybris-configure.log" 2>&1
    grep -n 'HAS_ANDROID_11_0_0' config.h Makefile 2>/dev/null | head -3 ;;
  make)
    # CXXFLAGS/CFLAGS are baked into the Makefiles at configure time, so pass them on the make command line
    make -C common -j"$(nproc)" SUBDIRS=. CFLAGS="$CFLAGS" CXXFLAGS="$CXXFLAGS" libhybris-common.la >"$B/build/libhybris-make.log" 2>&1
    make -C common/q -j"$(nproc)" CFLAGS="$CFLAGS" CXXFLAGS="$CXXFLAGS" >>"$B/build/libhybris-make.log" 2>&1 ;;
  install)
    make -C common install-libLTLIBRARIES DESTDIR="$OUT" >"$B/build/libhybris-install.log" 2>&1
    make -C common/q install DESTDIR="$OUT" >>"$B/build/libhybris-install.log" 2>&1
    find "$OUT" -type f | head -20 ;;
esac

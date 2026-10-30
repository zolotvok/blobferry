#!/usr/bin/env bash
# Builds the Mali bridge ICD shim + hal-probe for aarch64 glibc (host clang, PanVK's Debian bookworm sysroot) and
# deploys them into the DroidDeck rootfs. [Claude Sonnet 5.5] 2026-10-26
# Usage: bridge/build-bridge.sh build|zip|install-driver|deploy   (needs build-libhybris.sh install done; deploy logs to ../adb-device-changes.txt)
set -eu
B=$(cd "$(dirname "$0")" && pwd)
SR=$B/../panvk-kbase-android/work/sysroot-bookworm-arm64
NDK=$HOME/Android/Sdk/ndk/29.0.14206865/toolchains/llvm/prebuilt/linux-x86_64/bin
GCCDIR=$(ls -d "$SR"/usr/lib/gcc/aarch64-linux-gnu/*/ | head -1)
T="--target=aarch64-linux-gnu --sysroot=$SR --gcc-install-dir=$GCCDIR --ld-path=$NDK/ld.lld"
LIBDIR=$B/build/libhybris-install/usr/lib/hybris
OUT=$B/build/mali_bridge
STAGE=$OUT/stage/usr/lib/mali_bridge
LINK="-L$LIBDIR -lhybris-common -Wl,-rpath,/usr/lib/hybris"
mkdir -p "$STAGE"
case ${1:-build} in
  build)
    clang $T -O2 -Wall -fPIC -shared -fvisibility=hidden -Wl,-soname,libvulkan_mali_bridge.so -Wl,-Bsymbolic -Wl,--no-undefined -Wl,-z,nodelete \
      "$B/src/icd/mali_bridge_icd.c" -o "$STAGE/libvulkan_mali_bridge.so" $LINK
    clang $T -O2 -Wall -fPIC -shared -fvisibility=hidden -Wl,-soname,libVkLayer_droiddeck_mali_compat.so -Wl,-Bsymbolic -Wl,--no-undefined -Wl,-z,nodelete \
      -I"$B/src/compat" -I"$B/../panvk-kbase-android/work/mesa/include" \
      "$B/src/compat/mali_compat_layer.c" -o "$STAGE/libVkLayer_droiddeck_mali_compat.so" -lpthread
    clang $T -O2 -Wall "$B/test/hal-probe.c" -o "$OUT/hal-probe" $LINK
    clang $T -O2 -Wall -pthread "$B/test/icd-threads.c" -o "$OUT/icd-threads" -lvulkan
    clang $T -O2 -Wall -fPIC -shared "$B/test/segv-report.c" -o "$STAGE/segv-report.so"   # LD_PRELOAD crash reporter (no gdb in the guest)
    cp "$OUT/icd-threads" "$STAGE/icd-threads"   # test tools ride along in /usr/lib/mali_bridge (no new rootfs dirs)
    clang $T -O2 -Wall "$B/test/img-probe.c" -o "$STAGE/img-probe" -lvulkan   # which dma-buf image create infos the HAL takes
    clang $T -O2 -Wall -I"$B/../panvk-kbase-android/work/mesa/include" "$B/test/map-probe.c" -o "$STAGE/map-probe" -lvulkan   # can the vendor mappings be moved (placed maps)
    clang $T -O2 -Wall "$B/test/bc-probe.c" -o "$STAGE/bc-probe" -lvulkan   # does the compat layer decode BC uploads
    clang $T -O2 -Wall -I"$B/test" "$B/test/pushdesc-probe.c" -o "$STAGE/pushdesc-probe" -lvulkan   # push-descriptor UBO in vkd3d's root-CBV layout
    clang $T -O2 -Wall -I"$B/test" "$B/test/xfb-probe.c" -o "$STAGE/xfb-probe" -lvulkan   # cost of stream output (GS + transform feedback)
    clang $T -O2 -Wall -I"$B/../panvk-kbase-android/work/mesa/include" "$B/test/afrc-probe.c" -o "$STAGE/afrc-probe" -lvulkan   # AFRC rates for the BC shadow formats
    # vkprobe.exe: an x86-64 Windows program for Proton (llvm-mingw in tools-llvm-mingw; Vulkan headers from the sysroot)
    MINGW=$(ls "$B"/../tools-llvm-mingw/llvm-mingw-*/bin/x86_64-w64-mingw32-clang 2>/dev/null | head -1)
    if [ -n "$MINGW" ]; then
      "$MINGW" -O2 -Wall -idirafter "$SR/usr/include" "$B/test/vkprobe.c" -o "$OUT/vkprobe.exe"
      cp "$OUT/vkprobe.exe" "$STAGE/vkprobe.exe"
    else echo "build-bridge: no llvm-mingw under tools-llvm-mingw, vkprobe.exe not built" >&2; fi
    # The JSON lives outside /usr/share/vulkan/icd.d on purpose: the loader must only see the bridge when asked.
    cat >"$STAGE/mali_bridge_icd.json" <<'EOF'
{
  "file_format_version": "1.0.0",
  "ICD": {
    "library_path": "/usr/lib/mali_bridge/libvulkan_mali_bridge.so",
    "api_version": "1.3.0"
  }
}
EOF
    # Implicit compat layer manifest
    IMPLICIT_DIR="$OUT/stage/usr/share/vulkan/implicit_layer.d"
    mkdir -p "$IMPLICIT_DIR"
    cat >"$STAGE/VkLayer_droiddeck_mali_compat.json" <<'EOF_COMPAT'
{
  "file_format_version": "1.0.0",
  "layer": {
    "name": "VK_LAYER_DROIDDECK_mali_compat",
    "type": "GLOBAL",
    "library_path": "/usr/lib/mali_bridge/libVkLayer_droiddeck_mali_compat.so",
    "api_version": "1.3.0",
    "implementation_version": "1",
    "description": "Mali Compatibility Layer for DXVK / Proton",
    "functions": {
      "vkNegotiateLoaderLayerInterfaceVersion": "vkNegotiateLoaderLayerInterfaceVersion"
    },
    "enable_environment": {
      "ENABLE_MALI_COMPAT": "1"
    },
    "disable_environment": {
      "DISABLE_MALI_COMPAT": "1"
    }
  }
}
EOF_COMPAT
    cp "$STAGE/VkLayer_droiddeck_mali_compat.json" "$IMPLICIT_DIR/VkLayer_droiddeck_mali_compat.json"
    file "$STAGE/libvulkan_mali_bridge.so" "$STAGE/libVkLayer_droiddeck_mali_compat.so" "$OUT/hal-probe" "$OUT/icd-threads" | cut -c1-150 ;;
  zip)
    # Importable driver zip for the app's "Runtime driver" page (Import Turnip zip...): the shim plus a meta.json. The app writes
    # its own icd.json. libhybris is not in the zip: it lives in the runtime at /usr/lib/hybris (build-libhybris.sh install).
    cat >"$STAGE/meta.json" <<'EOF2'
{"schemaVersion": 1, "kind": "linux-vulkan-icd", "name": "Mali system driver (bridge)", "driverVersion": "v1.r54p1", "libc": "glibc", "minGlibc": "2.36"}
EOF2
    rm -f "$OUT/Mali_Bridge.zip"
    (cd "$STAGE" && zip -q -j "$OUT/Mali_Bridge.zip" libvulkan_mali_bridge.so meta.json)
    unzip -l "$OUT/Mali_Bridge.zip" | tail -4 ;;
  install-driver)
    # Same result as importing Mali_Bridge.zip on the Runtime driver page (LinuxVulkanDriverManager.adopt), without the file picker.
    D=files/linux_vulkan_drivers/Mali_Bridge
    adb push -q "$STAGE/libvulkan_mali_bridge.so" /data/local/tmp/libvulkan_mali_bridge.so
    adb shell "run-as com.droiddeck.launcher sh -c 'mkdir -p $D && cp /data/local/tmp/libvulkan_mali_bridge.so $D/ && chmod 755 $D/libvulkan_mali_bridge.so && touch $D/.dev'; rm /data/local/tmp/libvulkan_mali_bridge.so"
    adb shell "run-as com.droiddeck.launcher sh -c 'cat > $D/icd.json'" <<EOF3
{
  "file_format_version": "1.0.0",
  "ICD": {
    "library_path": "/data/user/0/com.droiddeck.launcher/$D/libvulkan_mali_bridge.so",
    "api_version": "1.3.0"
  }
}
EOF3
    adb shell "run-as com.droiddeck.launcher sh -c 'cat > $D/meta.json'" <<'EOF3'
{"schemaVersion": 1, "kind": "linux-vulkan-icd", "name": "Mali system driver (bridge)", "driverVersion": "v1.r54p1", "libc": "glibc", "minGlibc": "2.36", "sourceLibraryName": "libvulkan_mali_bridge.so"}
EOF3
    echo "$(date +%H:%M) bridge: install $D (shim + icd.json + meta.json) into the app's Linux driver dir; remove with: adb shell run-as com.droiddeck.launcher rm -r $D" >> "$B/../adb-device-changes.txt"
    adb shell "run-as com.droiddeck.launcher ls -l $D" ;;
  deploy)
    tar -C "$OUT/stage" -cf "$OUT/mali_bridge.tar" usr
    adb push -q "$OUT/mali_bridge.tar" /data/local/tmp/mali_bridge.tar
    # .dev: DroidDeck builds that bundle Blobferry restage it at every session start; this marker keeps the deployed build
    adb shell "run-as com.droiddeck.launcher sh -c 'tar -C files/linuxfs -xf /data/local/tmp/mali_bridge.tar && touch files/linuxfs/usr/lib/mali_bridge/.dev'; rm /data/local/tmp/mali_bridge.tar"
    echo "$(date +%H:%M) bridge: deploy usr/lib/mali_bridge (ICD shim + JSON) into rootfs files/linuxfs, .dev marker (app keeps it); no session restart; back to the bundled build: rm files/linuxfs/usr/lib/mali_bridge/.dev" >> "$B/../adb-device-changes.txt"
    adb shell "run-as com.droiddeck.launcher ls -l files/linuxfs/usr/lib/mali_bridge" ;;
esac

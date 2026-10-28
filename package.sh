#!/usr/bin/env bash
# package.sh  pack the built bridge into build/dist/blobferry-<commit>.zip (run build-libhybris.sh install, build-wsi.sh
# and build-bridge.sh build first). The zip holds:
#   rootfs/            files for the DroidDeck guest rootfs (shim, compat + WSI layers and their manifests, libhybris)
#   Mali_Bridge.zip    the driver zip for DroidDeck's Runtime driver page ("Import Turnip zip...")
#   install-adb.sh     installs both on a phone over adb (needs a debuggable DroidDeck build: run-as)
#   INSTALL.txt, LICENSE, NOTICE.md, licenses/libhybris/
# The vendor Vulkan driver is not in it: the bridge loads the phone's own. [Claude Opus 5.5] 2026-10-28
set -eu
B=$(cd "$(dirname "$0")" && pwd)
S=$B/build/mali_bridge/stage
H=$B/build/libhybris-install/usr/lib/hybris
V=$(git -C "$B" rev-parse --short HEAD)
[ -n "$(git -C "$B" status --porcelain -- src)" ] && V=$V-dirty
OUT=$B/build/dist/blobferry-$V
rm -rf "$OUT" && mkdir -p "$OUT/rootfs/usr/lib/mali_bridge" "$OUT/rootfs/usr/share/vulkan/implicit_layer.d" "$OUT/rootfs/usr/lib/hybris" \
                           "$OUT/licenses/libhybris"
for f in libvulkan_mali_bridge.so libVkLayer_droiddeck_mali_compat.so libVkLayer_window_system_integration.so \
         mali_bridge_icd.json VkLayer_droiddeck_mali_compat.json VkLayer_window_system_integration.json; do
    cp "$S/usr/lib/mali_bridge/$f" "$OUT/rootfs/usr/lib/mali_bridge/"
done
cp "$S"/usr/share/vulkan/implicit_layer.d/*.json "$OUT/rootfs/usr/share/vulkan/implicit_layer.d/"
cp -P "$H"/libhybris-common.so* "$OUT/rootfs/usr/lib/hybris/"
mkdir -p "$OUT/rootfs/usr/lib/hybris/libhybris/linker" && cp "$H/libhybris/linker/q.so" "$OUT/rootfs/usr/lib/hybris/libhybris/linker/"
rm -f "$OUT/rootfs/usr/lib/hybris/"*.la
(cd "$S/usr/lib/mali_bridge" && zip -q -j "$OUT/Mali_Bridge.zip" libvulkan_mali_bridge.so meta.json)
cp "$B/LICENSE" "$B/NOTICE.md" "$OUT/"
cp "$B"/third_party/libhybris/LICENSE.* "$OUT/licenses/libhybris/"
echo "libhybris source: https://github.com/wmww/libhybris commit $(git -C "$B/third_party/libhybris" rev-parse HEAD)" > "$OUT/licenses/libhybris/SOURCE.txt"

cat > "$OUT/install-adb.sh" <<'EOF'
#!/usr/bin/env bash
# install-adb.sh: install Blobferry into DroidDeck on the phone connected over adb. Needs a debuggable DroidDeck build
# with the bridge support (github.com/zolotvok/DroidDeck, branch mali-panvk); stop the DroidDeck session first.
set -eu
D=$(cd "$(dirname "$0")" && pwd)
P=com.droiddeck.launcher
T=/data/local/tmp/blobferry
tar -C "$D/rootfs" -cf /tmp/blobferry-rootfs.tar usr
adb shell "mkdir -p $T"
adb push -q /tmp/blobferry-rootfs.tar $T/rootfs.tar
adb push -q "$D/rootfs/usr/lib/mali_bridge/libvulkan_mali_bridge.so" $T/
adb shell "run-as $P sh -c 'tar -C files/linuxfs -xf $T/rootfs.tar && mkdir -p files/linux_vulkan_drivers/Mali_Bridge && cp $T/libvulkan_mali_bridge.so files/linux_vulkan_drivers/Mali_Bridge/ && chmod 755 files/linux_vulkan_drivers/Mali_Bridge/libvulkan_mali_bridge.so'"
adb shell "run-as $P sh -c 'cat > files/linux_vulkan_drivers/Mali_Bridge/icd.json'" <<J
{"file_format_version": "1.0.0", "ICD": {"library_path": "/data/user/0/$P/files/linux_vulkan_drivers/Mali_Bridge/libvulkan_mali_bridge.so", "api_version": "1.3.0"}}
J
unzip -p "$D/Mali_Bridge.zip" meta.json | adb shell "run-as $P sh -c 'cat > files/linux_vulkan_drivers/Mali_Bridge/meta.json'"
adb shell "rm -r $T"
rm -f /tmp/blobferry-rootfs.tar
echo "Installed. In DroidDeck: Components > Runtime driver > Mali system driver (bridge), then start a session."
EOF
chmod +x "$OUT/install-adb.sh"

cat > "$OUT/INSTALL.txt" <<EOF
Blobferry $V: run Windows games on the phone's own Mali Vulkan driver in DroidDeck (experimental)

Tested on one phone only (vivo X300 Pro, Mali-G1-Ultra MC12, vendor driver v1.r54p1).

Needs
- DroidDeck with the bridge support: github.com/zolotvok/DroidDeck, branch mali-panvk (the official DroidDeck
  does not have it). Steam and the desktop keep running on PanVK; only the games use the vendor driver.
- adb, and a debuggable DroidDeck build for install-adb.sh (it uses run-as).

DroidDeck 0.4.0-mali.1 and newer from the fork bundle Blobferry and install it by themselves; this package is for
other builds and for testing a Blobferry build.

Install
1. Stop the DroidDeck session.
2. ./install-adb.sh  (copies rootfs/ into DroidDeck's Linux rootfs and installs the driver).
   Mali_Bridge.zip alone (Runtime driver page, "Import Turnip zip...") installs only the shim; the layers and
   libhybris in rootfs/ are needed as well, so use the script.
3. Components > Runtime driver > "Mali system driver (bridge)", start a session, launch a game.

Not importable into Winlator, GameHub or GameNative: the shim is a glibc Linux Vulkan driver that loads the Android
driver through libhybris inside DroidDeck's rootfs, and the layers are installed as Vulkan implicit layers there.

License: MIT (LICENSE); third-party parts and credits in NOTICE.md and licenses/.
EOF
(cd "$B/build/dist" && rm -f "blobferry-$V.zip" && zip -q -r -y "blobferry-$V.zip" "blobferry-$V")
ls -l "$B/build/dist/blobferry-$V.zip"
unzip -l "$B/build/dist/blobferry-$V.zip"

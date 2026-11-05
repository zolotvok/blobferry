#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-or-later
# package.sh  pack the built bridge into build/dist/blobferry-<commit>.zip (run build-libhybris.sh install, build-wsi.sh
# and build-bridge.sh build first). The zip holds:
#   rootfs/            files for the DroidDeck guest rootfs (shim, compat + WSI layers and their manifests, libhybris)
#   Mali_Bridge.zip    the driver zip for DroidDeck's Runtime driver page ("Import Turnip zip...")
#   install-adb.sh     installs both on a phone over adb (needs a debuggable DroidDeck build: run-as)
#   INSTALL.txt, SOURCE.txt (commit, submodules, patches), MD5SUMS, LICENSE, NOTICE.md, licenses/{libhybris,vulkan-wsi-layer}/
# Next to the zip: blobferry-<V>.zip.md5 and blobferry-<V>.zip.sha256 (the release assets, see the end).
# The vendor Vulkan driver is not in it: the bridge loads the phone's own. [Claude Opus 5.5] 2026-10-28
set -eu
B=$(cd "$(dirname "$0")" && pwd)
# BLOBFERRY_STAGE=<dir> packs another stage dir with the same layout (usr/lib/mali_bridge + usr/share/vulkan/...), e.g. a
# stage-* build that is on the phone; BLOBFERRY_VERSION=<name> then names the package. [keeper, Claude Opus 5.5] 2026-11-02
S=${BLOBFERRY_STAGE:-$B/build/mali_bridge/stage}
H=$B/build/libhybris-install/usr/lib/hybris
C=$(git -C "$B" rev-parse HEAD)
V=$(git -C "$B" rev-parse --short HEAD)
# Dirty = any change to a tracked file or a submodule (the build scripts export the submodules' committed HEADs, but a
# dirty tree means the package may not match the commit it names). A named release (BLOBFERRY_VERSION) must be clean.
# [keeper, Claude Opus 5.5] 2026-11-04
DIRTY=$(git -C "$B" status --porcelain --untracked-files=no --ignore-submodules=none)
[ -n "$DIRTY" ] && V=$V-dirty
if [ -n "${BLOBFERRY_VERSION:-}" ]; then
    [ -z "$DIRTY" ] || [ "${BLOBFERRY_ALLOW_DIRTY:-0}" = 1 ] || { echo "package.sh: tree is dirty, a named release needs a clean commit:" >&2; echo "$DIRTY" >&2; exit 1; }
    V=$BLOBFERRY_VERSION
fi
OUT=$B/build/dist/blobferry-$V
rm -rf "$OUT" && mkdir -p "$OUT/rootfs/usr/lib/mali_bridge" "$OUT/rootfs/usr/share/vulkan/implicit_layer.d" "$OUT/rootfs/usr/lib/hybris" \
                           "$OUT/licenses/libhybris" "$OUT/licenses/vulkan-wsi-layer"
for f in libvulkan_mali_bridge.so libVkLayer_droiddeck_mali_compat.so libVkLayer_window_system_integration.so \
         mali_bridge_icd.json VkLayer_droiddeck_mali_compat.json VkLayer_window_system_integration.json; do
    cp "$S/usr/lib/mali_bridge/$f" "$OUT/rootfs/usr/lib/mali_bridge/"
done
cp "$S"/usr/share/vulkan/implicit_layer.d/*.json "$OUT/rootfs/usr/share/vulkan/implicit_layer.d/"
cp -P "$H"/libhybris-common.so* "$OUT/rootfs/usr/lib/hybris/"
mkdir -p "$OUT/rootfs/usr/lib/hybris/libhybris/linker" && cp "$H/libhybris/linker/q.so" "$OUT/rootfs/usr/lib/hybris/libhybris/linker/"
rm -f "$OUT/rootfs/usr/lib/hybris/"*.la
# Inner driver zip: fixed file times (the commit time) and no extra attributes, so the same files give the same bytes
T=$(git -C "$B" log -1 --format=%ct HEAD)
MZ=$OUT.mz; rm -rf "$MZ"; mkdir -p "$MZ"; cp "$S/usr/lib/mali_bridge/libvulkan_mali_bridge.so" "$S/usr/lib/mali_bridge/meta.json" "$MZ/"; touch -d "@$T" "$MZ"/*
(cd "$MZ" && TZ=UTC zip -q -X -j "$OUT/Mali_Bridge.zip" libvulkan_mali_bridge.so meta.json); rm -rf "$MZ"
cp "$B/LICENSE" "$B/NOTICE.md" "$OUT/"
cp "$B"/third_party/libhybris/LICENSE.* "$OUT/licenses/libhybris/"
cp "$B"/patches/libhybris/*.patch "$OUT/licenses/libhybris/"
echo "libhybris source: https://github.com/wmww/libhybris commit $(git -C "$B/third_party/libhybris" rev-parse HEAD), plus the patches in this folder (applied by build-libhybris.sh); full build source: https://github.com/zolotvok/blobferry commit $C" > "$OUT/licenses/libhybris/SOURCE.txt"
cp "$B/third_party/vulkan-wsi-layer/LICENSE" "$OUT/licenses/vulkan-wsi-layer/"
cp "$B"/patches/vulkan-wsi-layer/*.patch "$OUT/licenses/vulkan-wsi-layer/"
echo "vulkan-wsi-layer source: https://gitlab.freedesktop.org/mesa/vulkan-wsi-layer commit $(git -C "$B/third_party/vulkan-wsi-layer" rev-parse HEAD), plus the patches in this folder (applied in name order by build-wsi.sh; MIT, like the layer); full build source: https://github.com/zolotvok/blobferry commit $C" > "$OUT/licenses/vulkan-wsi-layer/SOURCE.txt"
{
    echo "Blobferry $V"
    echo "Source: https://github.com/zolotvok/blobferry commit $C${DIRTY:+ (built from a dirty tree)}"
    echo "Submodules (build-libhybris.sh and build-wsi.sh build an export of these commits plus patches/<name>/*.patch):"
    git -C "$B" submodule status | sed 's/^/  /'
    echo "Patches applied at build time:"
    (cd "$B" && md5sum patches/libhybris/*.patch patches/vulkan-wsi-layer/*.patch | sed 's/^/  /')
    echo "Build: build-libhybris.sh configure, make, install; build-wsi.sh; build-bridge.sh build, zip; package.sh"
} > "$OUT/SOURCE.txt"

cat > "$OUT/install-adb.sh" <<'EOF'
#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-or-later
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
Source: https://github.com/zolotvok/blobferry commit $C (SOURCE.txt); checksums in MD5SUMS.

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

License: GPL-3.0-or-later (LICENSE); third-party parts and credits in NOTICE.md and licenses/.
EOF
(cd "$OUT" && find . -type f ! -name MD5SUMS | LC_ALL=C sort | xargs md5sum > MD5SUMS)
# Same zip bytes for the same files: file times set to the commit time, no extra attributes, sorted order.
find "$OUT" -exec touch -h -d "@$T" {} +
(cd "$B/build/dist" && rm -f "blobferry-$V.zip" && find "blobferry-$V" | LC_ALL=C sort | TZ=UTC zip -q -X -y "blobferry-$V.zip" -@)
(cd "$B/build/dist" && md5sum "blobferry-$V.zip" > "blobferry-$V.zip.md5")
# Release asset contract (DroidDeck's Blobferry updater, gpu/BlobferryReleases + droiddeckctl blobferry install): a
# GitHub release tagged <V> carries blobferry-<V>.zip and blobferry-<V>.zip.sha256 (sha256sum format); the app checks
# the sha256 against GitHub's asset digest, then the zip's MD5SUMS, SOURCE.txt and the AArch64 libraries.
# [app, Claude Opus 5.5] 2026-11-05
(cd "$B/build/dist" && sha256sum "blobferry-$V.zip" > "blobferry-$V.zip.sha256")
ls -l "$B/build/dist/blobferry-$V.zip"; cat "$B/build/dist/blobferry-$V.zip.md5" "$B/build/dist/blobferry-$V.zip.sha256"
unzip -l "$B/build/dist/blobferry-$V.zip"

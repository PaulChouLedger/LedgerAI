#!/bin/bash
# Inject the aura-first-boot identity-reset service onto a freshly-cloned
# Jetson disk. Works on either a block device (the destination NVMe via
# its USB enclosure) or a raw image file (uses losetup to expose
# partitions).
#
# Run AFTER `dd if=<source> of=<target>` finishes so the target has the
# Jetson partition table; this script just mounts the rootfs partition
# and drops the service files in.
#
# Usage:
#   sudo ./inject_first_boot.sh /dev/sdc            # block device
#   sudo ./inject_first_boot.sh /mnt/aura1/master.img  # image file

set -euo pipefail

TARGET="${1:-}"
if [ -z "$TARGET" ]; then
    echo "usage: $0 <block-device-or-image-file>" >&2
    exit 1
fi

HERE=$(cd "$(dirname "$0")" && pwd)
SCRIPT="$HERE/aura-first-boot.sh"
UNIT="$HERE/aura-first-boot.service"
[ -f "$SCRIPT" ] || { echo "missing $SCRIPT" >&2; exit 1; }
[ -f "$UNIT" ]   || { echo "missing $UNIT" >&2; exit 1; }

# Set up the target as a block device. Image files go through losetup so
# the kernel sees their partitions as separate /dev/loopNpM nodes.
LOOP=""
if [ -b "$TARGET" ]; then
    DEV="$TARGET"
elif [ -f "$TARGET" ]; then
    LOOP=$(losetup --find --show --partscan "$TARGET")
    DEV="$LOOP"
    echo "[inject] image $TARGET attached as $LOOP"
else
    echo "$TARGET is neither a block device nor a regular file" >&2
    exit 1
fi

MNT=""
cleanup() {
    if [ -n "${MNT:-}" ] && mountpoint -q "$MNT"; then umount "$MNT"; fi
    if [ -n "${MNT:-}" ] && [ -d "$MNT" ]; then rmdir "$MNT"; fi
    if [ -n "$LOOP" ]; then losetup -d "$LOOP" || true; fi
}
trap cleanup EXIT

# Find the largest ext4 partition — that's the Jetson rootfs (sdN1 in the
# 15-partition layout).
ROOTFS=$(lsblk -lnpb -o NAME,FSTYPE,SIZE "$DEV" \
         | awk '$2=="ext4"{print $3, $1}' \
         | sort -nr | head -1 | awk '{print $2}')

if [ -z "$ROOTFS" ]; then
    echo "no ext4 partition found on $DEV" >&2
    exit 1
fi
echo "[inject] rootfs partition: $ROOTFS"

MNT=$(mktemp -d)
mount "$ROOTFS" "$MNT"

install -d -m 0755 "$MNT/usr/local/bin"
install -m 0755 "$SCRIPT" "$MNT/usr/local/bin/aura-first-boot.sh"
install -m 0644 "$UNIT"   "$MNT/etc/systemd/system/aura-first-boot.service"

# Enable the unit by hand (we can't run `systemctl enable` against a
# foreign rootfs, so we create the symlink directly).
install -d -m 0755 "$MNT/etc/systemd/system/multi-user.target.wants"
ln -sfn /etc/systemd/system/aura-first-boot.service \
        "$MNT/etc/systemd/system/multi-user.target.wants/aura-first-boot.service"

sync

echo "[inject] aura-first-boot service installed on $ROOTFS"
echo "[inject] on first boot of the new puck:"
echo "         - new hostname derived from MAC"
echo "         - machine-id, ssh host keys, tailscale state regenerated"
echo "         - aura per-puck state cleared (voicelines kept)"

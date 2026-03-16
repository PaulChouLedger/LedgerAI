#!/bin/bash
# Mount a 32GB RAM disk for Piper training data (zero I/O latency)
# Run once: sudo bash setup_ramdisk.sh
set -e

MOUNT_POINT="/mnt/ramdisk"
SIZE="32G"

mkdir -p "$MOUNT_POINT"
mount -t tmpfs -o size=$SIZE tmpfs "$MOUNT_POINT"
chown paul:paul "$MOUNT_POINT"

# Add to fstab for persistence across reboots
if ! grep -q "$MOUNT_POINT" /etc/fstab; then
    echo "tmpfs $MOUNT_POINT tmpfs size=$SIZE,uid=1000,gid=1000 0 0" >> /etc/fstab
    echo "Added to /etc/fstab"
fi

echo "32GB RAM disk mounted at $MOUNT_POINT"
echo "Free RAM after mount:"
free -h | head -2

#!/bin/bash
# Restore a master Jetson image to one or more destination block devices
# in parallel, then auto-inject the aura-first-boot identity-reset
# service onto each. Designed for the "clone factory" workflow: one image
# file on internal storage, N USB-NVMe enclosures plugged in, one command.
#
# Usage:
#   sudo ./restore_image.sh /mnt/aura1/master.img /dev/sdc /dev/sdd /dev/sde
#
# At USB 3.2 Gen 2 speeds, ~3 destinations in parallel saturate a single
# USB controller. More than that and per-disk throughput drops; better to
# batch in groups of 3.

set -euo pipefail

IMG="${1:-}"
shift || true
if [ -z "$IMG" ] || [ "$#" -lt 1 ]; then
    echo "usage: $0 <image-file> <dest-device> [<dest-device> ...]" >&2
    exit 1
fi
[ -f "$IMG" ] || { echo "image $IMG not found" >&2; exit 1; }

HERE=$(cd "$(dirname "$0")" && pwd)
INJECT="$HERE/inject_first_boot.sh"
[ -x "$INJECT" ] || { echo "missing $INJECT" >&2; exit 1; }

# Sanity check every target before kicking anything off — we don't want
# to start writes to /dev/sdX and discover halfway through that /dev/sdY
# was a typo for the source disk.
for DEV in "$@" ; do
    if [ ! -b "$DEV" ]; then
        echo "$DEV is not a block device" >&2; exit 1
    fi
    # Refuse to clobber a mounted disk.
    if findmnt -n -S "$DEV" >/dev/null 2>&1 || \
       findmnt -n -S "${DEV}1" >/dev/null 2>&1 ; then
        echo "$DEV (or a partition) is currently mounted — aborting" >&2
        exit 1
    fi
done

echo "[factory] image: $IMG"
echo "[factory] targets: $*"
echo "[factory] starting parallel restore..."

PIDS=()
for DEV in "$@" ; do
    LOG="/tmp/restore_$(basename "$DEV").log"
    (
        echo "[$DEV] dd start $(date -Iseconds)" | tee -a "$LOG"
        dd if="$IMG" of="$DEV" bs=64M conv=fsync status=progress 2>>"$LOG"
        echo "[$DEV] dd done $(date -Iseconds)" | tee -a "$LOG"
        # Re-read the partition table the kernel cached before dd ran.
        partprobe "$DEV" 2>/dev/null || true
        sleep 2
        "$INJECT" "$DEV" 2>&1 | tee -a "$LOG"
        echo "[$DEV] DONE $(date -Iseconds)" | tee -a "$LOG"
    ) &
    PIDS+=($!)
    echo "[factory] -> $DEV  (pid $!, log $LOG)"
done

echo "[factory] waiting for ${#PIDS[@]} parallel restores..."
FAIL=0
for p in "${PIDS[@]}" ; do
    if ! wait "$p"; then FAIL=$((FAIL+1)); fi
done

if [ "$FAIL" -gt 0 ]; then
    echo "[factory] $FAIL of ${#PIDS[@]} restores FAILED — check /tmp/restore_*.log" >&2
    exit 1
fi
echo "[factory] all ${#PIDS[@]} restores complete. Pull the disks and install."

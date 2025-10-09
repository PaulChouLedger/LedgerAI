#!/bin/bash
#
# Flash ReSpeaker Firmware Helper
#
# Temporarily removes udev rule, flashes firmware, then restores rule
#
# Usage:
#   sudo bash scripts/flash_firmware.sh
#

set -e

FIRMWARE_PATH="$HOME/usb_4_mic_array/1_channel_firmware.bin"
UDEV_RULE="/etc/udev/rules.d/99-respeaker.rules"
UDEV_BACKUP="/tmp/99-respeaker.rules.backup"

echo ""
echo "================================================================================"
echo "  🔧 RESPEAKER FIRMWARE FLASH HELPER"
echo "================================================================================"
echo ""

# Check if running as root
if [ "$EUID" -ne 0 ]; then 
    echo "  ❌ ERROR: Please run with sudo"
    echo ""
    echo "  Usage:"
    echo "    sudo bash scripts/flash_firmware.sh"
    echo ""
    exit 1
fi

# Get actual user
ACTUAL_USER=${SUDO_USER:-$USER}
ACTUAL_HOME=$(eval echo ~$ACTUAL_USER)
FIRMWARE_PATH="$ACTUAL_HOME/usb_4_mic_array/1_channel_firmware.bin"

# Check if firmware exists
if [ ! -f "$FIRMWARE_PATH" ]; then
    echo "  ❌ Firmware not found: $FIRMWARE_PATH"
    echo ""
    echo "  Please ensure 1_channel_firmware.bin is in ~/usb_4_mic_array/"
    echo ""
    exit 1
fi

echo "[1/5] Backing up udev rule..."
if [ -f "$UDEV_RULE" ]; then
    cp "$UDEV_RULE" "$UDEV_BACKUP"
    echo "     ✅ Backed up to $UDEV_BACKUP"
else
    echo "     ℹ️  No udev rule to backup"
fi

echo "[2/5] Removing udev rule (temporarily)..."
if [ -f "$UDEV_RULE" ]; then
    rm "$UDEV_RULE"
    udevadm control --reload-rules
    echo "     ✅ Udev rule removed"
fi

echo "[3/5] Flashing firmware..."
cd "$ACTUAL_HOME/usb_4_mic_array"
sudo -u $ACTUAL_USER python dfu.py --download 1_channel_firmware.bin

echo "[4/5] Waiting for device to reboot..."
sleep 3

echo "[5/5] Restoring udev rule..."
if [ -f "$UDEV_BACKUP" ]; then
    cp "$UDEV_BACKUP" "$UDEV_RULE"
    udevadm control --reload-rules
    udevadm trigger
    rm "$UDEV_BACKUP"
    echo "     ✅ Udev rule restored"
fi

echo ""
echo "================================================================================"
echo "  ✅ FIRMWARE FLASH COMPLETE"
echo "================================================================================"
echo ""
echo "  The ReSpeaker should now be running single-channel firmware."
echo ""
echo "  Next steps:"
echo "    1. Unplug and replug the USB device"
echo "    2. Run: python3 aura-control/listener.py"
echo ""
echo "================================================================================"
echo ""


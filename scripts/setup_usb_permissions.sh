#!/bin/bash
#
# Setup USB Permissions for ReSpeaker 4 Mic Array
#
# This creates a udev rule that allows non-root access to the ReSpeaker
# so the listener can configure hardware DSP settings without sudo.
#
# Usage:
#   sudo bash scripts/setup_usb_permissions.sh
#

set -e

echo ""
echo "================================================================================"
echo "  🔧 RESPEAKER USB PERMISSIONS SETUP"
echo "================================================================================"
echo ""
echo "  This will create a udev rule to allow non-root access to ReSpeaker."
echo "  You only need to run this ONCE (survives reboots)."
echo ""
echo "================================================================================"
echo ""

# Check if running as root
if [ "$EUID" -ne 0 ]; then 
    echo "  ❌ ERROR: Please run with sudo"
    echo ""
    echo "  Usage:"
    echo "    sudo bash scripts/setup_usb_permissions.sh"
    echo ""
    exit 1
fi

# Get the actual user (not root)
ACTUAL_USER=${SUDO_USER:-$USER}
echo "[1/4] Detected user: $ACTUAL_USER"

# Create udev rule
UDEV_RULE_FILE="/etc/udev/rules.d/99-respeaker.rules"
echo "[2/4] Creating udev rule: $UDEV_RULE_FILE"

cat > $UDEV_RULE_FILE << 'EOF'
# ReSpeaker 4 Mic Array - Allow user access for DSP configuration
# Vendor: 0x2886 (Seeed), Product: 0x0018 (ReSpeaker 4 Mic Array)
SUBSYSTEM=="usb", ATTR{idVendor}=="2886", ATTR{idProduct}=="0018", MODE="0666", GROUP="plugdev"
EOF

echo "     ✅ Created: $UDEV_RULE_FILE"

# Reload udev rules
echo "[3/4] Reloading udev rules..."
udevadm control --reload-rules
udevadm trigger

echo "     ✅ Rules reloaded"

# Add user to plugdev group (if not already)
echo "[4/4] Adding user to 'plugdev' group..."
usermod -a -G plugdev $ACTUAL_USER

echo "     ✅ User added to group"

echo ""
echo "================================================================================"
echo "  ✅ SETUP COMPLETE"
echo "================================================================================"
echo ""
echo "  The ReSpeaker USB device is now accessible without sudo."
echo ""
echo "  IMPORTANT: You may need to:"
echo "    1. Log out and log back in (for group changes to take effect)"
echo "    2. Unplug and replug the ReSpeaker USB device"
echo ""
echo "  After that, listener.py can auto-configure hardware DSP!"
echo ""
echo "================================================================================"
echo ""


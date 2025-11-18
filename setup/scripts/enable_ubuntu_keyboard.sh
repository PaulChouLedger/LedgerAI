#!/bin/bash
#
# Re-enable Ubuntu On-Screen Keyboard (for testing/recovery)
#
# This script re-enables Ubuntu's default on-screen keyboard.
#
# Usage:
#   sudo bash setup/scripts/enable_ubuntu_keyboard.sh
#

set -e

echo ""
echo "================================================================================"
echo "  ⌨️  RE-ENABLE UBUNTU ON-SCREEN KEYBOARD"
echo "================================================================================"
echo ""

# Check if running as root
if [ "$EUID" -ne 0 ]; then 
    echo "  ❌ ERROR: Please run with sudo"
    echo ""
    echo "  Usage:"
    echo "    sudo bash setup/scripts/enable_ubuntu_keyboard.sh"
    echo ""
    exit 1
fi

# Re-enable onboard
if [ -f "/etc/xdg/autostart/onboard.desktop" ]; then
    sed -i 's/^Hidden=true/Hidden=false/' /etc/xdg/autostart/onboard.desktop || true
    sed -i 's/^NoDisplay=true/NoDisplay=false/' /etc/xdg/autostart/onboard.desktop || true
    echo "  ✅ Re-enabled onboard"
fi

# Re-enable caribou
if [ -f "/etc/xdg/autostart/caribou.desktop" ]; then
    sed -i 's/^Hidden=true/Hidden=false/' /etc/xdg/autostart/caribou.desktop || true
    sed -i 's/^NoDisplay=true/NoDisplay=false/' /etc/xdg/autostart/caribou.desktop || true
    echo "  ✅ Re-enabled caribou"
fi

# Remove user-level disabled files
# Detect the actual user (works even when run via sudo)
AURA_USER="${SUDO_USER:-$USER}"
AURA_HOME="/home/$AURA_USER"
if [ -d "$AURA_HOME/.config/autostart" ]; then
    rm -f "$AURA_HOME/.config/autostart/onboard.desktop"
    rm -f "$AURA_HOME/.config/autostart/caribou.desktop"
    echo "  ✅ Removed user-level disabled files for user: $AURA_USER"
fi

echo ""
echo "  ✅ Ubuntu keyboard re-enabled"
echo ""


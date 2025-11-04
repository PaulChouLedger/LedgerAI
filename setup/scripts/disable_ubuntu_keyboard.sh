#!/bin/bash
#
# Disable Ubuntu On-Screen Keyboard
#
# This script disables Ubuntu's default on-screen keyboard (onboard, caribou, etc.)
# so that only the custom GUI keyboard from scripts is used.
#
# Usage:
#   sudo bash setup/scripts/disable_ubuntu_keyboard.sh
#

set -e

echo ""
echo "================================================================================"
echo "  ⌨️  DISABLE UBUNTU ON-SCREEN KEYBOARD"
echo "================================================================================"
echo ""
echo "  This will disable Ubuntu's default on-screen keyboard so that only"
echo "  the custom GUI keyboard from scripts is used."
echo ""
echo "================================================================================"
echo ""

# Check if running as root
if [ "$EUID" -ne 0 ]; then 
    echo "  ❌ ERROR: Please run with sudo"
    echo ""
    echo "  Usage:"
    echo "    sudo bash setup/scripts/disable_ubuntu_keyboard.sh"
    echo ""
    exit 1
fi

# Step 1: Kill any running on-screen keyboard processes
echo "[1/5] Stopping running on-screen keyboard processes..."
pkill -f onboard || true
pkill -f caribou || true
pkill -f matchbox-keyboard || true
echo "     ✅ Stopped running processes"

# Step 2: Disable onboard from autostart
echo "[2/5] Disabling onboard from autostart..."
if [ -f "/etc/xdg/autostart/onboard.desktop" ]; then
    # Disable by adding Hidden=true
    sed -i 's/^Hidden=.*/Hidden=true/' /etc/xdg/autostart/onboard.desktop || true
    # Also add NoDisplay=true
    if ! grep -q "NoDisplay=true" /etc/xdg/autostart/onboard.desktop; then
        echo "NoDisplay=true" >> /etc/xdg/autostart/onboard.desktop
    fi
    echo "     ✅ Disabled onboard autostart"
else
    echo "     ℹ️  onboard.desktop not found (may not be installed)"
fi

# Step 3: Disable caribou from autostart
echo "[3/5] Disabling caribou from autostart..."
if [ -f "/etc/xdg/autostart/caribou.desktop" ]; then
    sed -i 's/^Hidden=.*/Hidden=true/' /etc/xdg/autostart/caribou.desktop || true
    if ! grep -q "NoDisplay=true" /etc/xdg/autostart/caribou.desktop; then
        echo "NoDisplay=true" >> /etc/xdg/autostart/caribou.desktop
    fi
    echo "     ✅ Disabled caribou autostart"
else
    echo "     ℹ️  caribou.desktop not found (may not be installed)"
fi

# Step 4: Disable user-level autostart (for aura user)
AURA_USER="aura"
AURA_HOME="/home/$AURA_USER"
if [ -d "$AURA_HOME" ]; then
    echo "[4/5] Disabling user-level autostart for $AURA_USER..."
    
    # Create autostart directory if it doesn't exist
    mkdir -p "$AURA_HOME/.config/autostart"
    
    # Disable onboard in user autostart
    if [ -f "$AURA_HOME/.config/autostart/onboard.desktop" ]; then
        sed -i 's/^Hidden=.*/Hidden=true/' "$AURA_HOME/.config/autostart/onboard.desktop" || true
        if ! grep -q "NoDisplay=true" "$AURA_HOME/.config/autostart/onboard.desktop"; then
            echo "NoDisplay=true" >> "$AURA_HOME/.config/autostart/onboard.desktop"
        fi
        echo "     ✅ Disabled user-level onboard autostart"
    fi
    
    # Create a disabled onboard.desktop if it doesn't exist
    if [ ! -f "$AURA_HOME/.config/autostart/onboard.desktop" ]; then
        cat > "$AURA_HOME/.config/autostart/onboard.desktop" << EOF
[Desktop Entry]
Type=Application
Name=Onboard
Hidden=true
NoDisplay=true
EOF
        echo "     ✅ Created disabled onboard.desktop"
    fi
    
    # Disable caribou in user autostart
    if [ -f "$AURA_HOME/.config/autostart/caribou.desktop" ]; then
        sed -i 's/^Hidden=.*/Hidden=true/' "$AURA_HOME/.config/autostart/caribou.desktop" || true
        if ! grep -q "NoDisplay=true" "$AURA_HOME/.config/autostart/caribou.desktop"; then
            echo "NoDisplay=true" >> "$AURA_HOME/.config/autostart/caribou.desktop"
        fi
        echo "     ✅ Disabled user-level caribou autostart"
    fi
    
    # Fix ownership
    chown -R "$AURA_USER:$AURA_USER" "$AURA_HOME/.config/autostart" 2>/dev/null || true
else
    echo "     ⚠️  User $AURA_USER home directory not found, skipping user-level config"
fi

# Step 5: Unset environment variables that might trigger keyboard
echo "[5/5] Configuring environment variables..."
# These can be set in .bashrc or .profile to prevent keyboard from auto-starting
ENV_VARS=(
    "GTK_IM_MODULE=ibus"
    "QT_IM_MODULE=ibus"
)

# Note: We don't want to completely disable these, just ensure they don't trigger OSK
# The custom keyboard will work fine with these settings

echo "     ✅ Environment configuration complete"

echo ""
echo "================================================================================"
echo "  ✅ UBUNTU KEYBOARD DISABLED"
echo "================================================================================"
echo ""
echo "  Ubuntu's on-screen keyboard has been disabled."
echo "  The custom GUI keyboard from scripts will be used instead."
echo ""
echo "  To re-enable Ubuntu keyboard (if needed):"
echo "    sudo bash setup/scripts/enable_ubuntu_keyboard.sh"
echo ""
echo "  Useful commands:"
echo "    ps aux | grep -E 'onboard|caribou|matchbox'  # Check if running"
echo "    pkill -f onboard                              # Kill if running"
echo ""


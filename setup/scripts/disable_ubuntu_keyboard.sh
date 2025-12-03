#!/bin/bash
#
# Temporarily Stop Ubuntu On-Screen Keyboard
#
# This script temporarily stops Ubuntu's default on-screen keyboard processes.
# It does NOT permanently disable them - they will restart on next login/boot.
# This is meant to be called by main.py while Aura is running.
#
# Usage:
#   bash setup/scripts/disable_ubuntu_keyboard.sh
#   (No sudo needed - only kills user processes)
#

echo "[Keyboard] Stopping Ubuntu on-screen keyboard processes..."

# Kill any running on-screen keyboard processes (user-level, no sudo needed)
# Use SIGKILL (-9) for more aggressive killing
pkill -9 -f "onboard|caribou|matchbox-keyboard|gnome-shell.*keyboard|ibus.*keyboard" 2>/dev/null || true

# Also try individual kills as fallback
pkill -9 -f onboard 2>/dev/null || true
pkill -9 -f caribou 2>/dev/null || true
pkill -9 -f matchbox-keyboard 2>/dev/null || true

# Try to stop systemd user services if they exist
systemctl --user stop onboard.service 2>/dev/null || true

# Check if any are still running
if pgrep -f "onboard|caribou|matchbox-keyboard" > /dev/null; then
    echo "[Keyboard] ⚠️  Some keyboard processes still running (may need sudo)"
else
    echo "[Keyboard] ✅ Ubuntu keyboard processes stopped"
fi


#!/bin/bash
#
# Install ReSpeaker Auto-Tuning Service
#
# This creates a systemd service that automatically configures
# the ReSpeaker hardware DSP on every boot.
#
# Usage:
#   sudo bash scripts/install_auto_tune.sh
#

set -e

echo ""
echo "================================================================================"
echo "  🔧 RESPEAKER AUTO-TUNE SERVICE INSTALLER"
echo "================================================================================"
echo ""
echo "  This will create a systemd service that runs on boot to configure"
echo "  the ReSpeaker hardware DSP (no manual tuning needed!)."
echo ""
echo "================================================================================"
echo ""

# Check if running as root
if [ "$EUID" -ne 0 ]; then 
    echo "  ❌ ERROR: Please run with sudo"
    echo ""
    echo "  Usage:"
    echo "    sudo bash scripts/install_auto_tune.sh"
    echo ""
    exit 1
fi

SERVICE_FILE="/etc/systemd/system/respeaker-tuning.service"
SOURCE_SERVICE="$(dirname "$0")/respeaker-tuning.service"

echo "[1/4] Copying service file..."
cp "$SOURCE_SERVICE" "$SERVICE_FILE"
echo "     ✅ Copied to $SERVICE_FILE"

echo "[2/4] Reloading systemd..."
systemctl daemon-reload
echo "     ✅ Systemd reloaded"

echo "[3/4] Enabling service (auto-start on boot)..."
systemctl enable respeaker-tuning.service
echo "     ✅ Service enabled"

echo "[4/4] Starting service now..."
systemctl start respeaker-tuning.service
echo "     ✅ Service started"

echo ""
echo "================================================================================"
echo "  ✅ INSTALLATION COMPLETE"
echo "================================================================================"
echo ""
echo "  The ReSpeaker will now be automatically configured on every boot!"
echo ""
echo "  Useful commands:"
echo "    sudo systemctl status respeaker-tuning    # Check service status"
echo "    sudo systemctl restart respeaker-tuning   # Manually re-tune"
echo "    sudo journalctl -u respeaker-tuning       # View service logs"
echo "    sudo systemctl disable respeaker-tuning   # Disable auto-tune"
echo ""
echo "  The listener will now work without sudo - just run:"
echo "    python3 aura-control/main.py"
echo ""
echo "================================================================================"
echo ""


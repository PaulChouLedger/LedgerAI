#!/bin/bash
# install.sh -- Install / upgrade Aura v2 systemd service
#
# Usage:
#   cd aura/services
#   sudo bash install.sh
#
# This replaces any existing aura.service with the v2 entry point.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SERVICE_SRC="$SCRIPT_DIR/aura-v2.service"
SERVICE_DST="/etc/systemd/system/aura.service"

if [ "$(id -u)" -ne 0 ]; then
    echo "Error: must run as root (sudo bash install.sh)"
    exit 1
fi

if [ ! -f "$SERVICE_SRC" ]; then
    echo "Error: $SERVICE_SRC not found"
    exit 1
fi

# Check if v1 service is running and stop it
if systemctl is-active --quiet aura.service 2>/dev/null; then
    echo "[install] Stopping existing aura.service..."
    systemctl stop aura.service
fi

# Symlink (auto-updates on git pull) rather than copy
echo "[install] Linking $SERVICE_SRC -> $SERVICE_DST"
ln -sf "$SERVICE_SRC" "$SERVICE_DST"

systemctl daemon-reload
systemctl enable aura.service

echo "[install] Aura v2 service installed and enabled"
echo ""
echo "Commands:"
echo "  sudo systemctl start aura.service    # Start now"
echo "  sudo systemctl stop aura.service     # Stop"
echo "  sudo systemctl restart aura.service  # Restart"
echo "  journalctl -u aura.service -f        # View logs"

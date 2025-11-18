#!/bin/bash
# Quick fix script to update xvf3800-tuning.service with correct user and paths
# Usage: bash fix_xvf3800_service.sh

set -e

AURA_USER="${SUDO_USER:-$USER}"
AURA_HOME="/home/$AURA_USER"
LEDGERAI_DIR="$AURA_HOME/LedgerAI"

# Detect Python command
if command -v python3.10 &> /dev/null; then
    PYTHON_CMD="python3.10"
elif command -v python3.9 &> /dev/null; then
    PYTHON_CMD="python3.9"
elif command -v python3.8 &> /dev/null; then
    PYTHON_CMD="python3.8"
else
    PYTHON_CMD="python3"
fi

SYSTEMD_SERVICE="/etc/systemd/system/xvf3800-tuning.service"

if [ ! -f "$SYSTEMD_SERVICE" ]; then
    echo "❌ Service file not found at $SYSTEMD_SERVICE"
    exit 1
fi

echo "Fixing xvf3800-tuning.service..."
echo "  User: $AURA_USER"
echo "  Home: $AURA_HOME"
echo "  Python: $PYTHON_CMD"
echo "  LedgerAI: $LEDGERAI_DIR"
echo ""

# Extract preset from current service file (default to agc_20_ec)
PRESET_ARG=$(grep "^ExecStart=" "$SYSTEMD_SERVICE" | sed -n 's/.*tune_xvf3800.py[[:space:]]*\([^[:space:]]*\).*/\1/p' || echo "agc_20_ec")
if [ -z "$PRESET_ARG" ] || [ "$PRESET_ARG" = "$LEDGERAI_DIR/setup/scripts/tune_xvf3800.py" ]; then
    PRESET_ARG="agc_20_ec"
fi

echo "  Preset: $PRESET_ARG"
echo ""

# Create backup
sudo cp "$SYSTEMD_SERVICE" "${SYSTEMD_SERVICE}.bak.$(date +%s)"

# Update service file
sudo sed -i "s|^User=.*|User=$AURA_USER|g" "$SYSTEMD_SERVICE"
sudo sed -i "s|^ExecStart=.*|ExecStart=$PYTHON_CMD $LEDGERAI_DIR/setup/scripts/tune_xvf3800.py $PRESET_ARG|g" "$SYSTEMD_SERVICE"

# Reload systemd
sudo systemctl daemon-reload

echo "✅ Service file updated!"
echo ""
echo "Verifying service file:"
cat "$SYSTEMD_SERVICE" | grep -E "^(User|ExecStart)="
echo ""
echo "To test the service:"
echo "  sudo systemctl start xvf3800-tuning.service"
echo "  sudo systemctl status xvf3800-tuning.service"


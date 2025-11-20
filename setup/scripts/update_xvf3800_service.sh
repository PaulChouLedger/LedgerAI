#!/bin/bash
# Quick script to update xvf3800-tuning.service with latest changes
# This updates the service file without running the full install script

set -e

# Detect user
AURA_USER="${SUDO_USER:-$USER}"
AURA_HOME="/home/$AURA_USER"
LEDGERAI_DIR="$AURA_HOME/LedgerAI"

# Detect Python
if command -v python3.10 &> /dev/null; then
    PYTHON_CMD="python3.10"
elif command -v python3.9 &> /dev/null; then
    PYTHON_CMD="python3.9"
elif command -v python3.8 &> /dev/null; then
    PYTHON_CMD="python3.8"
else
    PYTHON_CMD="python3"
fi

SERVICE_FILE="$LEDGERAI_DIR/setup/scripts/xvf3800-tuning.service"
SYSTEMD_SERVICE="/etc/systemd/system/xvf3800-tuning.service"

if [ ! -f "$SERVICE_FILE" ]; then
    echo "❌ Service template not found at $SERVICE_FILE"
    exit 1
fi

echo "🔄 Updating xvf3800-tuning.service..."

# Create temporary service file
TEMP_SERVICE="/tmp/xvf3800-tuning.service"
cp "$SERVICE_FILE" "$TEMP_SERVICE"

# Replace placeholders
sed -i "s|__AURA_USER__|$AURA_USER|g" "$TEMP_SERVICE"
sed -i "s|__PYTHON_CMD__|$PYTHON_CMD|g" "$TEMP_SERVICE"
sed -i "s|__LEDGERAI_DIR__|$LEDGERAI_DIR|g" "$TEMP_SERVICE"

# Handle legacy values
sed -i "s|User=aura|User=$AURA_USER|g" "$TEMP_SERVICE"
sed -i "s|User=ledger|User=$AURA_USER|g" "$TEMP_SERVICE"
sed -i "s|/home/aura|$AURA_HOME|g" "$TEMP_SERVICE"
sed -i "s|/home/ledger|$AURA_HOME|g" "$TEMP_SERVICE"
sed -i "s|/usr/bin/python3|$PYTHON_CMD|g" "$TEMP_SERVICE" || true

# Extract and preserve preset argument
PRESET_ARG=$(grep "^ExecStart=" "$TEMP_SERVICE" | sed -n 's/.*tune_xvf3800.py[[:space:]]*\([^[:space:]]*\).*/\1/p' || echo "agc_20_ec")
if [ -z "$PRESET_ARG" ] || [ "$PRESET_ARG" = "$LEDGERAI_DIR/setup/scripts/tune_xvf3800.py" ] || [ "$PRESET_ARG" = "__LEDGERAI_DIR__/setup/scripts/tune_xvf3800.py" ]; then
    PRESET_ARG="agc_20_ec"
fi

# Update ExecStart with -B flag (ensures latest script is always used)
sed -i "s|^ExecStart=.*|ExecStart=$PYTHON_CMD -B $LEDGERAI_DIR/setup/scripts/tune_xvf3800.py $PRESET_ARG|g" "$TEMP_SERVICE"

# Copy to systemd
sudo cp "$TEMP_SERVICE" "$SYSTEMD_SERVICE"
rm -f "$TEMP_SERVICE"

# Reload systemd
sudo systemctl daemon-reload

echo "✅ Service updated successfully!"
echo ""
echo "The service will:"
echo "  - Run on boot (WantedBy=multi-user.target)"
echo "  - Use latest script version (with -B flag, no bytecode cache)"
echo "  - Apply LED disable settings automatically"
echo ""
echo "To test immediately:"
echo "  sudo systemctl restart xvf3800-tuning.service"
echo ""
echo "To check status:"
echo "  sudo systemctl status xvf3800-tuning.service"
echo "  journalctl -u xvf3800-tuning.service -n 50"


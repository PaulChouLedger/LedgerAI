#!/bin/bash
# Quick fix to add onnxruntime environment variables to systemd service

set -e

echo "=========================================="
echo "  Fixing onnxruntime in systemd service"
echo "=========================================="
echo ""

SERVICE_FILE="/etc/systemd/system/aura.service"

if [ ! -f "$SERVICE_FILE" ]; then
    echo "[ERROR] aura.service not found at $SERVICE_FILE"
    exit 1
fi

echo "[INFO] Current service file:"
grep -E "Environment=|ExecStart=" "$SERVICE_FILE" | head -5
echo ""

# Check if environment variables already exist
if grep -q "ORT_DISABLE_CPUINFO" "$SERVICE_FILE"; then
    echo "[INFO] ✅ ORT environment variables already present in service file"
    echo "[INFO] Reloading systemd..."
    sudo systemctl daemon-reload
    echo "[INFO] ✅ Done"
    exit 0
fi

echo "[INFO] Adding ORT environment variables to service file..."
echo ""

# Create backup
sudo cp "$SERVICE_FILE" "$SERVICE_FILE.backup.$(date +%s)"
echo "[INFO] ✅ Backup created"

# Add environment variables after XAUTHORITY line
sudo sed -i '/^Environment="XAUTHORITY=/a # CRITICAL: Set onnxruntime environment variables to prevent CPU detection crashes\nEnvironment="ORT_DISABLE_CPUINFO=1"\nEnvironment="ORT_LOG_LEVEL=3"' "$SERVICE_FILE"

echo "[INFO] ✅ Environment variables added"
echo ""

# Reload systemd
sudo systemctl daemon-reload
echo "[INFO] ✅ Systemd reloaded"
echo ""

# Show updated service file
echo "[INFO] Updated service file (relevant lines):"
grep -E "Environment=|ExecStart=" "$SERVICE_FILE" | head -8
echo ""

echo "=========================================="
echo "  Fix Complete!"
echo "=========================================="
echo ""
echo "The systemd service now includes:"
echo "  Environment=\"ORT_DISABLE_CPUINFO=1\""
echo "  Environment=\"ORT_LOG_LEVEL=3\""
echo ""
echo "To apply:"
echo "  sudo systemctl restart aura.service"
echo "  sudo systemctl status aura.service"
echo ""

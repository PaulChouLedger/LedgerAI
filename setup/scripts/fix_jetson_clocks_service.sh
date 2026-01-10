#!/bin/bash
# Fix jetson-maxn-power.service - find correct paths and update service file

set -e

echo "🔧 Fixing jetson-maxn-power.service..."

# Find where jetson_clocks is located
JETSON_CLOCKS_PATH=""
if command -v jetson_clocks >/dev/null 2>&1; then
    JETSON_CLOCKS_PATH=$(which jetson_clocks)
elif [ -f "/usr/bin/jetson_clocks" ]; then
    JETSON_CLOCKS_PATH="/usr/bin/jetson_clocks"
fi

if [ -z "$JETSON_CLOCKS_PATH" ]; then
    echo "❌ ERROR: jetson_clocks not found!"
    echo "   Please ensure Jetson utilities are installed"
    exit 1
fi

echo "✅ Found jetson_clocks at: $JETSON_CLOCKS_PATH"

# Find where nvpmodel is located (optional)
NVPMODEL_PATH=""
if command -v nvpmodel >/dev/null 2>&1; then
    NVPMODEL_PATH=$(which nvpmodel)
elif [ -f "/usr/bin/nvpmodel" ]; then
    NVPMODEL_PATH="/usr/bin/nvpmodel"
fi

# Create service file
SERVICE_FILE="/etc/systemd/system/jetson-maxn-power.service"

echo "📝 Creating/updating service file: $SERVICE_FILE"

if [ -n "$NVPMODEL_PATH" ]; then
    echo "✅ Found nvpmodel at: $NVPMODEL_PATH"
    sudo tee "$SERVICE_FILE" > /dev/null << EOFSERVICE
[Unit]
Description=Set Jetson to MAXN Power Mode
After=multi-user.target

[Service]
Type=oneshot
ExecStart=/bin/sh -c "$NVPMODEL_PATH -m 0; $JETSON_CLOCKS_PATH"
RemainAfterExit=yes

[Install]
WantedBy=multi-user.target
EOFSERVICE
else
    echo "⚠️  nvpmodel not found - creating service with jetson_clocks only"
    sudo tee "$SERVICE_FILE" > /dev/null << EOFSERVICE
[Unit]
Description=Set Jetson to Maximum Clocks
After=multi-user.target

[Service]
Type=oneshot
ExecStart=$JETSON_CLOCKS_PATH
RemainAfterExit=yes

[Install]
WantedBy=multi-user.target
EOFSERVICE
fi

# Reload systemd
echo "🔄 Reloading systemd..."
sudo systemctl daemon-reload

# Enable service
echo "✅ Enabling service..."
sudo systemctl enable jetson-maxn-power.service

# Start service
echo "🚀 Starting service..."
if sudo systemctl start jetson-maxn-power.service; then
    echo "✅ Service started successfully!"
else
    echo "⚠️  Service start had issues - checking status..."
    sudo systemctl status jetson-maxn-power.service --no-pager || true
    exit 1
fi

echo ""
echo "✅ jetson-maxn-power.service configured!"
echo ""
echo "Check status with:"
echo "  sudo systemctl status jetson-maxn-power.service"
echo ""
echo "Clocks will now be set to maximum on every boot"

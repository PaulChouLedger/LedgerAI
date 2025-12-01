#!/bin/bash
# List all available Jetson power modes and their wattage

echo "=========================================="
echo "  Jetson Power Modes"
echo "=========================================="
echo ""

# Check if nvpmodel exists
if ! command -v nvpmodel >/dev/null 2>&1; then
    echo "❌ nvpmodel command not found"
    exit 1
fi

# Show current mode
echo "Current Power Mode:"
sudo nvpmodel -q
echo ""

# Try to read config file to show all modes
CONFIG_FILE="/etc/nvpmodel/nvpmodel_p3767_0000_super.conf"
if [ -f "$CONFIG_FILE" ]; then
    echo "Available Power Modes in config file:"
    echo "----------------------------------------"
    # Extract power mode sections
    grep -E "^POWER_MODEL|^POWER_MODEL_NAME|^POWER_MODEL_ID" "$CONFIG_FILE" | head -50
    echo ""
    echo "Full power mode details:"
    echo "----------------------------------------"
    # Show power mode blocks
    awk '/^POWER_MODEL/,/^}/' "$CONFIG_FILE" | grep -E "POWER_MODEL|POWER_MODEL_NAME|POWER_MODEL_ID|POWER_MODEL_TDP" | head -100
else
    echo "⚠️  Config file not found at: $CONFIG_FILE"
    echo ""
    echo "Trying to list modes from nvpmodel:"
    # Try to query each mode (0-10 typically)
    for mode in {0..10}; do
        echo -n "Mode $mode: "
        sudo nvpmodel -m $mode -q 2>&1 | grep -i "power\|watt\|tdp" | head -1 || echo "Unknown"
    done
fi

echo ""
echo "=========================================="
echo "  Common Jetson Orin NX Power Modes:"
echo "=========================================="
echo "  Mode 0 (MAXN):     ~25W - Maximum performance"
echo "  Mode 1:            ~10W - Balanced (current)"
echo "  Mode 2:            ~15W - Performance"
echo "  Mode 3:            ~20W - High performance"
echo ""
echo "Note: Actual wattage depends on workload and cooling."
echo "      Lower modes = quieter fan = better audio quality"
echo ""


#!/bin/bash
# Revert all hardware modifications made to the system
# This removes udev rules, systemd services, and resets ReSpeaker

set -e

echo "=========================================="
echo "  REVERTING HARDWARE MODIFICATIONS"
echo "=========================================="
echo ""

# Check if running with sudo
if [ "$EUID" -ne 0 ]; then
    echo "❌ This script must be run with sudo"
    echo "Usage: sudo bash scripts/revert_hardware_setup.sh"
    exit 1
fi

# 1. Remove udev rules
echo "🔧 Step 1: Removing udev rules..."
if [ -f /etc/udev/rules.d/99-respeaker.rules ]; then
    rm /etc/udev/rules.d/99-respeaker.rules
    echo "   ✅ Removed /etc/udev/rules.d/99-respeaker.rules"
else
    echo "   ℹ️  No udev rules found"
fi

# Reload udev rules
if command -v udevadm &> /dev/null; then
    udevadm control --reload-rules
    udevadm trigger
    echo "   ✅ Reloaded udev rules"
fi

# 2. Remove systemd service (if exists)
echo ""
echo "🔧 Step 2: Removing systemd services..."

SERVICE_FILES=(
    "/etc/systemd/system/aura-listener.service"
    "/etc/systemd/system/ledgerai.service"
    "/etc/systemd/system/aura.service"
)

SERVICE_FOUND=false
for service_file in "${SERVICE_FILES[@]}"; do
    if [ -f "$service_file" ]; then
        SERVICE_FOUND=true
        service_name=$(basename "$service_file")
        
        # Stop service if running
        if systemctl is-active --quiet "$service_name" 2>/dev/null; then
            systemctl stop "$service_name"
            echo "   ✅ Stopped $service_name"
        fi
        
        # Disable service if enabled
        if systemctl is-enabled --quiet "$service_name" 2>/dev/null; then
            systemctl disable "$service_name"
            echo "   ✅ Disabled $service_name"
        fi
        
        # Remove service file
        rm "$service_file"
        echo "   ✅ Removed $service_file"
    fi
done

if [ "$SERVICE_FOUND" = false ]; then
    echo "   ℹ️  No systemd services found"
fi

# Reload systemd
if command -v systemctl &> /dev/null; then
    systemctl daemon-reload
    echo "   ✅ Reloaded systemd"
fi

# 3. Information about ReSpeaker hardware settings
echo ""
echo "🔧 Step 3: ReSpeaker hardware settings..."
echo "   ℹ️  Hardware AGC/DSP settings are temporary and reset on reboot"
echo "   ℹ️  No persistent configuration to remove"

# 4. Optional: Remove USB device permissions from user groups
echo ""
echo "🔧 Step 4: User group permissions..."
USER_TO_CHECK="${SUDO_USER:-$USER}"

if groups "$USER_TO_CHECK" | grep -q 'plugdev'; then
    echo "   ℹ️  User '$USER_TO_CHECK' is in 'plugdev' group"
    echo "   ⚠️  Not removing (may be needed for other devices)"
else
    echo "   ℹ️  User not in special USB groups"
fi

# Summary
echo ""
echo "=========================================="
echo "  ✅ HARDWARE MODIFICATIONS REVERTED"
echo "=========================================="
echo ""
echo "What was removed:"
echo "  • udev rules for ReSpeaker USB permissions"
echo "  • systemd services for auto-start"
echo ""
echo "What remains:"
echo "  • User group memberships (plugdev, etc.)"
echo "  • Installed Python packages"
echo "  • Docker containers"
echo ""
echo "Next steps:"
echo "  1. Reboot to ensure all changes take effect:"
echo "     sudo reboot"
echo ""
echo "  2. To fully reset ReSpeaker hardware:"
echo "     Unplug and replug the USB device"
echo ""
echo "=========================================="


#!/bin/bash
# Quick diagnostic script for XVF3800 tuning service
# Usage: ./check_xvf_service.sh

echo "🔍 Checking XVF3800 Tuning Service..."
echo ""

echo "1️⃣  Service Status:"
systemctl status xvf3800-tuning.service --no-pager -l | head -12
echo ""

echo "2️⃣  Service Enabled:"
if systemctl is-enabled xvf3800-tuning.service >/dev/null 2>&1; then
    echo "   ✅ Enabled (runs on boot)"
else
    echo "   ❌ Not enabled (won't run on boot)"
fi
echo ""

echo "3️⃣  Recent Logs (last 20 lines):"
echo "   (Look for LED-related messages and errors)"
journalctl -u xvf3800-tuning.service -n 20 --no-pager | tail -20
echo ""

echo "4️⃣  USB Device Detection:"
if lsusb | grep -qi "reSpeaker\|XVF3800\|UACDemo"; then
    echo "   ✅ Device found:"
    lsusb | grep -i "reSpeaker\|XVF3800\|UACDemo"
else
    echo "   ⚠️  Device not found in lsusb"
fi
echo ""

echo "5️⃣  xvf_host Binary:"
XVF_HOST_PATH="$HOME/reSpeaker_XVF3800_USB_4MIC_ARRAY/host_control/jetson/xvf_host"
if [ -f "$XVF_HOST_PATH" ]; then
    echo "   ✅ Found: $XVF_HOST_PATH"
    ls -lh "$XVF_HOST_PATH"
else
    echo "   ❌ Not found at: $XVF_HOST_PATH"
    echo "   💡 Check if XVF3800 SDK is installed"
fi
echo ""

echo "6️⃣  LED Status Check:"
echo "   To test LED disable manually, run:"
echo "   python3 -B ~/LedgerAI/setup/scripts/tune_xvf3800.py agc_20_ec"
echo ""

echo "7️⃣  Service Configuration:"
echo "   Service file: /etc/systemd/system/xvf3800-tuning.service"
if [ -f "/etc/systemd/system/xvf3800-tuning.service" ]; then
    echo "   ✅ Service file exists"
    echo "   Current preset:"
    grep "ExecStart" /etc/systemd/system/xvf3800-tuning.service | grep -o "agc_[^ ]*\|balanced_beam\|ultra_sensitive" | head -1
else
    echo "   ❌ Service file not found"
fi
echo ""

echo "💡 Quick Fixes:"
echo "   - Restart service: sudo systemctl restart xvf3800-tuning.service"
echo "   - View live logs: journalctl -u xvf3800-tuning.service -f"
echo "   - Test manually: python3 -B ~/LedgerAI/setup/scripts/tune_xvf3800.py agc_20_ec"


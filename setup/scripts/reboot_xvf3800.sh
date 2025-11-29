#!/bin/bash
# Software reboot of XVF3800 using xvf_host REBOOT command
# This simulates pressing the physical reset button

set +e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${GREEN}==========================================${NC}"
echo -e "${GREEN}  Software Reboot - XVF3800${NC}"
echo -e "${GREEN}==========================================${NC}"
echo ""

# Find xvf_host
XVF_HOST_PATH="$HOME/reSpeaker_XVF3800_USB_4MIC_ARRAY/host_control/jetson/xvf_host"

if [ ! -f "$XVF_HOST_PATH" ]; then
    echo -e "${RED}❌ xvf_host not found at: $XVF_HOST_PATH${NC}"
    exit 1
fi

chmod +x "$XVF_HOST_PATH" 2>/dev/null || true

# Check if device is present
if ! lsusb | grep -q "2886:" && ! lsusb | grep -qi "seeed\|reSpeaker\|XVF3800"; then
    echo -e "${RED}❌ XVF3800 USB device not found${NC}"
    exit 1
fi

echo -e "${YELLOW}[1]${NC} Device found"
lsusb | grep -i "seeed\|reSpeaker\|XVF3800" | head -1 | sed 's/^/  /'
echo ""

# Send REBOOT command
echo -e "${YELLOW}[2]${NC} Sending REBOOT command to device..."
echo "  Command: $XVF_HOST_PATH REBOOT 1"
if sudo "$XVF_HOST_PATH" REBOOT 1 2>&1; then
    echo -e "  ✅ REBOOT command sent successfully"
else
    EXIT_CODE=$?
    echo -e "  ⚠️  REBOOT command returned exit code: $EXIT_CODE"
    echo "  (This may be normal - device may reboot immediately)"
fi
echo ""

# Wait for device to disappear and reappear
echo -e "${YELLOW}[3]${NC} Waiting for device to reboot..."
sleep 2

# Check if device disappeared (rebooting)
if ! lsusb | grep -q "2886:" && ! lsusb | grep -qi "seeed\|reSpeaker\|XVF3800"; then
    echo "  ✅ Device disappeared (rebooting)"
else
    echo "  ⚠️  Device still visible (may have rebooted quickly)"
fi

# Wait for device to reappear
echo "  Waiting for device to reappear..."
for i in {1..15}; do
    if lsusb | grep -q "2886:" || lsusb | grep -qi "seeed\|reSpeaker\|XVF3800"; then
        echo -e "  ✅ Device reappeared (attempt $i/15)"
        break
    fi
    sleep 1
done
echo ""

# Wait for ALSA to detect
echo -e "${YELLOW}[4]${NC} Waiting for ALSA to detect device..."
for i in {1..10}; do
    if arecord -l 2>/dev/null | grep -qi "XVF3800\|reSpeaker"; then
        echo -e "  ✅ Device visible to ALSA (attempt $i/10)"
        break
    fi
    sleep 1
done
echo ""

# Wait a bit more for full initialization
echo -e "${YELLOW}[5]${NC} Waiting for device to fully initialize..."
sleep 3
echo ""

echo -e "${GREEN}==========================================${NC}"
echo -e "${GREEN}  Reboot Complete${NC}"
echo -e "${GREEN}==========================================${NC}"
echo ""
echo "Device should now be in the same state as after physical reset button press"
echo "Test microphone capture:"
echo "  python3 ~/LedgerAI/setup/scripts/test_transcription.py"
echo ""


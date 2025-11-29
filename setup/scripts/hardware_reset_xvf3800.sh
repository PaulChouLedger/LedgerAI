#!/bin/bash
# Full hardware reset of XVF3800 USB device
# This simulates unplugging/replugging the device

set +e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${GREEN}==========================================${NC}"
echo -e "${GREEN}  Full Hardware Reset - XVF3800${NC}"
echo -e "${GREEN}==========================================${NC}"
echo ""

# Find USB device
USB_LINE=$(lsusb | grep "2886:" | head -1)
if [ -z "$USB_LINE" ]; then
    USB_LINE=$(lsusb | grep -i "seeed\|reSpeaker\|XVF3800" | head -1)
fi

if [ -z "$USB_LINE" ]; then
    echo -e "${RED}❌ XVF3800 USB device not found${NC}"
    exit 1
fi

BUS=$(echo "$USB_LINE" | awk '{print $2}')
DEV=$(echo "$USB_LINE" | awk '{print $4}' | sed 's/://')
VID_PID=$(echo "$USB_LINE" | grep -o "[0-9a-f]\{4\}:[0-9a-f]\{4\}")

echo -e "${YELLOW}[1]${NC} Found device:"
echo "  $USB_LINE"
echo "  Bus: $BUS, Device: $DEV"
echo "  VID:PID: $VID_PID"
echo ""

# Method 1: Try usbreset utility (if available)
echo -e "${YELLOW}[2]${NC} Attempting USB reset via usbreset utility..."
if command -v usbreset >/dev/null 2>&1; then
    echo "  Using usbreset utility..."
    sudo usbreset "$VID_PID" 2>&1
    if [ $? -eq 0 ]; then
        echo -e "  ✅ USB reset successful"
        sleep 3
    else
        echo -e "  ⚠️  usbreset failed, trying alternative methods..."
    fi
else
    echo "  usbreset utility not found, trying sysfs method..."
fi
echo ""

# Method 2: Find correct sysfs path and reset
if [ -z "$USB_RESET_SUCCESS" ]; then
    echo -e "${YELLOW}[3]${NC} Finding USB device in sysfs..."
    
    # USB devices in sysfs use hierarchical paths like 1-1.4, not 001-004
    # Find the device by VID:PID
    SYSFS_PATH=$(find /sys/bus/usb/devices -name "idVendor" -exec grep -l "$(echo $VID_PID | cut -d: -f1)" {} \; 2>/dev/null | head -1 | xargs dirname)
    
    if [ -n "$SYSFS_PATH" ]; then
        echo "  Found sysfs path: $SYSFS_PATH"
        
        # Try authorized reset
        if [ -f "$SYSFS_PATH/authorized" ]; then
            echo "  Resetting via authorized attribute..."
            echo 0 | sudo tee "$SYSFS_PATH/authorized" >/dev/null 2>&1
            sleep 1
            echo 1 | sudo tee "$SYSFS_PATH/authorized" >/dev/null 2>&1
            sleep 2
            echo -e "  ✅ USB reset via authorized attribute"
        fi
        
        # Try power/control reset
        if [ -f "$SYSFS_PATH/power/control" ]; then
            echo "  Resetting via power control..."
            echo "suspend" | sudo tee "$SYSFS_PATH/power/control" >/dev/null 2>&1
            sleep 1
            echo "on" | sudo tee "$SYSFS_PATH/power/control" >/dev/null 2>&1
            sleep 2
            echo -e "  ✅ USB reset via power control"
        fi
    else
        echo -e "  ⚠️  Could not find sysfs path for device"
    fi
    echo ""
fi

# Method 3: Unbind/rebind USB driver
echo -e "${YELLOW}[4]${NC} Attempting USB driver unbind/rebind..."
if [ -n "$SYSFS_PATH" ]; then
    DRIVER_PATH=$(readlink "$SYSFS_PATH/driver" 2>/dev/null | xargs basename 2>/dev/null)
    if [ -n "$DRIVER_PATH" ] && [ "$DRIVER_PATH" != "usb" ]; then
        DEVICE_ID=$(basename "$SYSFS_PATH")
        echo "  Unbinding driver: $DRIVER_PATH"
        echo "$DEVICE_ID" | sudo tee "/sys/bus/usb/drivers/$DRIVER_PATH/unbind" >/dev/null 2>&1
        sleep 2
        echo "  Rebinding driver: $DRIVER_PATH"
        echo "$DEVICE_ID" | sudo tee "/sys/bus/usb/drivers/$DRIVER_PATH/bind" >/dev/null 2>&1
        sleep 2
        echo -e "  ✅ USB driver rebind attempted"
    else
        echo -e "  ⚠️  Could not determine driver or driver is 'usb' (cannot unbind)"
    fi
else
    echo -e "  ⚠️  Skipping (sysfs path not found)"
fi
echo ""

# Wait for device to reappear
echo -e "${YELLOW}[5]${NC} Waiting for device to reinitialize..."
for i in {1..10}; do
    if lsusb | grep -q "$VID_PID" || lsusb | grep -qi "seeed\|reSpeaker\|XVF3800"; then
        echo -e "  ✅ Device detected (attempt $i/10)"
        break
    fi
    sleep 1
done
echo ""

# Wait for ALSA to detect
echo -e "${YELLOW}[6]${NC} Waiting for ALSA to detect device..."
for i in {1..10}; do
    if arecord -l 2>/dev/null | grep -qi "XVF3800\|reSpeaker"; then
        echo -e "  ✅ Device visible to ALSA (attempt $i/10)"
        break
    fi
    sleep 1
done
echo ""

# Reconfigure GPIO and DSP
echo -e "${YELLOW}[7]${NC} Reconfiguring GPIO and DSP..."
TUNE_SCRIPT="$HOME/LedgerAI/setup/scripts/tune_xvf3800.py"
if [ -f "$TUNE_SCRIPT" ]; then
    echo "  Running: sudo python3 $TUNE_SCRIPT agc_20_ec"
    sudo python3 "$TUNE_SCRIPT" agc_20_ec 2>&1 | grep -E "\[GPIO\]|\[LED\]|✅|❌|⚠️|Profile Complete" || true
    echo -e "  ✅ Configuration complete"
else
    echo -e "  ${RED}❌ Tuning script not found: $TUNE_SCRIPT${NC}"
fi
echo ""

echo -e "${GREEN}==========================================${NC}"
echo -e "${GREEN}  Hardware Reset Complete${NC}"
echo -e "${GREEN}==========================================${NC}"
echo ""
echo "Test microphone capture:"
echo "  python3 ~/LedgerAI/setup/scripts/test_transcription.py"
echo ""


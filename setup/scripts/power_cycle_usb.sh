#!/bin/bash
# Power cycle USB device to simulate unplug/replug
# This cuts power to the USB port and restores it

set +e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${GREEN}==========================================${NC}"
echo -e "${GREEN}  USB Power Cycle - XVF3800${NC}"
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
echo "  Bus: $BUS, Device: $DEV, VID:PID: $VID_PID"
echo ""

# Method 1: Try uhubctl (USB hub power control) - most reliable
echo -e "${YELLOW}[2]${NC} Attempting USB hub power control via uhubctl..."
if command -v uhubctl >/dev/null 2>&1; then
    echo "  uhubctl found - attempting power cycle..."
    # List USB hubs
    HUBS=$(uhubctl 2>/dev/null | grep -i "hub" | head -1)
    if [ -n "$HUBS" ]; then
        # Try to find the hub and port for this device
        # This is complex - uhubctl needs hub number and port number
        # For now, try common configurations
        for hub in 1 2 3; do
            for port in 1 2 3 4; do
                echo "  Trying hub $hub, port $port..."
                uhubctl -l $hub -p $port -a 0 >/dev/null 2>&1
                sleep 2
                if ! lsusb | grep -q "$VID_PID"; then
                    echo "  ✅ Power cut successful (hub $hub, port $port)"
                    sleep 2
                    uhubctl -l $hub -p $port -a 1 >/dev/null 2>&1
                    sleep 3
                    if lsusb | grep -q "$VID_PID"; then
                        echo "  ✅ Power restored successfully"
                        POWER_CYCLE_SUCCESS=true
                        break 2
                    fi
                fi
            done
        done
        if [ -z "$POWER_CYCLE_SUCCESS" ]; then
            echo "  ⚠️  Could not determine correct hub/port"
        fi
    else
        echo "  ⚠️  No USB hubs found"
    fi
else
    echo "  uhubctl not installed"
    echo "  Install: sudo apt-get install uhubctl"
fi
echo ""

# Method 2: Try sysfs power control
if [ -z "$POWER_CYCLE_SUCCESS" ]; then
    echo -e "${YELLOW}[3]${NC} Attempting sysfs power control..."
    
    # Find device in sysfs
    SYSFS_DEVICE=$(find /sys/bus/usb/devices -name "idVendor" -exec grep -l "$(echo $VID_PID | cut -d: -f1)" {} \; 2>/dev/null | head -1)
    if [ -n "$SYSFS_DEVICE" ]; then
        SYSFS_PATH=$(dirname "$SYSFS_DEVICE")
        echo "  Found sysfs path: $SYSFS_PATH"
        
        # Try power/control
        if [ -f "$SYSFS_PATH/power/control" ]; then
            echo "  Cutting power via power/control..."
            echo "suspend" | sudo tee "$SYSFS_PATH/power/control" >/dev/null 2>&1
            sleep 2
            if ! lsusb | grep -q "$VID_PID"; then
                echo "  ✅ Power cut successful"
                sleep 2
                echo "on" | sudo tee "$SYSFS_PATH/power/control" >/dev/null 2>&1
                sleep 3
                if lsusb | grep -q "$VID_PID"; then
                    echo "  ✅ Power restored successfully"
                    POWER_CYCLE_SUCCESS=true
                else
                    echo "  ⚠️  Power restore may have failed"
                fi
            else
                echo "  ⚠️  Power cut may not have worked (device still visible)"
            fi
        else
            echo "  ⚠️  power/control not available"
        fi
        
        # Try removing/adding device
        if [ -z "$POWER_CYCLE_SUCCESS" ] && [ -f "$SYSFS_PATH/remove" ]; then
            echo "  Attempting device remove/add..."
            DEVICE_ID=$(basename "$SYSFS_PATH")
            echo "$DEVICE_ID" | sudo tee "$SYSFS_PATH/remove" >/dev/null 2>&1
            sleep 2
            if ! lsusb | grep -q "$VID_PID"; then
                echo "  ✅ Device removed"
                sleep 2
                # Trigger rescan
                for parent in $(dirname "$SYSFS_PATH"); do
                    if [ -f "$parent/rescan" ]; then
                        echo 1 | sudo tee "$parent/rescan" >/dev/null 2>&1
                        break
                    fi
                done
                sleep 3
                if lsusb | grep -q "$VID_PID"; then
                    echo "  ✅ Device re-added"
                    POWER_CYCLE_SUCCESS=true
                else
                    echo "  ⚠️  Device may need physical unplug/replug"
                fi
            else
                echo "  ⚠️  Device remove may not have worked"
            fi
        fi
    else
        echo "  ⚠️  Could not find device in sysfs"
    fi
    echo ""
fi

# Wait for device to fully reinitialize
if [ -n "$POWER_CYCLE_SUCCESS" ]; then
    echo -e "${YELLOW}[4]${NC} Waiting for device to fully reinitialize..."
    sleep 3
    
    # Wait for ALSA
    for i in {1..10}; do
        if arecord -l 2>/dev/null | grep -qi "XVF3800\|reSpeaker"; then
            echo -e "  ✅ Device visible to ALSA (attempt $i/10)"
            break
        fi
        sleep 1
    done
    echo ""
    
    echo -e "${GREEN}==========================================${NC}"
    echo -e "${GREEN}  Power Cycle Complete${NC}"
    echo -e "${GREEN}==========================================${NC}"
    echo ""
    echo "Device should now be in the same state as after physical unplug/replug"
    echo "Test microphone capture:"
    echo "  python3 ~/LedgerAI/setup/scripts/test_transcription.py"
    echo ""
else
    echo -e "${RED}==========================================${NC}"
    echo -e "${RED}  Power Cycle Failed${NC}"
    echo -e "${RED}==========================================${NC}"
    echo ""
    echo "Could not power cycle device programmatically."
    echo "Options:"
    echo "  1. Install uhubctl: sudo apt-get install uhubctl"
    echo "  2. Physically unplug/replug the device"
    echo ""
fi


#!/bin/bash
# Reset USB audio device to fix boot-time capture issues
# This simulates what happens when you unplug/replug the device

set +e # Don't exit on error

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}==========================================${NC}"
echo -e "${GREEN}  USB Audio Device Reset${NC}"
echo -e "${GREEN}==========================================${NC}"
echo ""

# Find XVF3800 USB device - try multiple methods
USB_DEVICE=""

# Method 1: Try VID:PID 2886:0018 (standard)
USB_DEVICE=$(lsusb | grep "2886:0018" | head -1)

# Method 2: Try by name (Seeed Studio or reSpeaker)
if [ -z "$USB_DEVICE" ]; then
    USB_DEVICE=$(lsusb | grep -i "seeed\|reSpeaker\|XVF3800" | head -1)
fi

# Method 3: Try alternative VID:PID (some devices use different PIDs)
if [ -z "$USB_DEVICE" ]; then
    USB_DEVICE=$(lsusb | grep "2886:" | head -1)
fi

if [ -z "$USB_DEVICE" ]; then
    echo -e "${RED}❌ XVF3800 USB device not found in lsusb${NC}"
    echo "  Available USB devices:"
    lsusb | sed 's/^/    /'
    echo ""
    echo "  But ALSA shows the device exists. Trying to find via ALSA..."
    
    # Try to find via ALSA device path
    ALSA_DEVICE=$(arecord -l 2>/dev/null | grep -i "XVF3800\|reSpeaker" | head -1)
    if [ -n "$ALSA_DEVICE" ]; then
        echo -e "  ${YELLOW}⚠️  Found in ALSA but not in lsusb - device may be in use${NC}"
        echo -e "  ${YELLOW}   This is OK - we can still reset GPIO and run tuning${NC}"
        USB_DEVICE="ALSA_ONLY"
    else
        echo -e "  ${RED}❌ Not found in ALSA either${NC}"
        exit 1
    fi
fi

# Extract bus and device numbers (if USB device found)
if [ "$USB_DEVICE" != "ALSA_ONLY" ]; then
    BUS=$(echo "$USB_DEVICE" | awk '{print $2}')
    DEV=$(echo "$USB_DEVICE" | awk '{print $4}' | sed 's/://')
    echo -e "${YELLOW}[1]${NC} Found XVF3800 at USB $BUS:$DEV"
    echo "  Device: $USB_DEVICE"
else
    BUS=""
    DEV=""
    echo -e "${YELLOW}[1]${NC} Found XVF3800 in ALSA (USB device info unavailable)"
fi

echo -e "${YELLOW}[1]${NC} Found XVF3800 at USB $BUS:$DEV"
echo ""

# Method 1: Reset USB device via sysfs (requires root)
if [ -n "$BUS" ] && [ -n "$DEV" ]; then
    echo -e "${YELLOW}[2]${NC} Attempting USB device reset..."
    USB_RESET_PATH="/sys/bus/usb/devices/$BUS-$DEV/authorized"
    
    if [ -f "$USB_RESET_PATH" ]; then
        echo "  Resetting USB device..."
        echo 0 | sudo tee "$USB_RESET_PATH" >/dev/null 2>&1
        sleep 1
        echo 1 | sudo tee "$USB_RESET_PATH" >/dev/null 2>&1
        sleep 2
        echo -e "  ✅ USB device reset attempted"
    else
        echo -e "  ⚠️  USB reset path not found: $USB_RESET_PATH"
    fi
    echo ""
    
    # Method 2: Unbind/rebind USB driver (more aggressive)
    echo -e "${YELLOW}[3]${NC} Attempting USB driver unbind/rebind..."
    USB_DRIVER_PATH="/sys/bus/usb/drivers/usb/$BUS-$DEV"
    
    if [ -d "$USB_DRIVER_PATH" ]; then
        echo "  Unbinding USB driver..."
        echo "$BUS-$DEV" | sudo tee /sys/bus/usb/drivers/usb/unbind >/dev/null 2>&1
        sleep 1
        echo "  Rebinding USB driver..."
        echo "$BUS-$DEV" | sudo tee /sys/bus/usb/drivers/usb/bind >/dev/null 2>&1
        sleep 2
        echo -e "  ✅ USB driver rebind attempted"
    else
        echo -e "  ⚠️  USB driver path not found: $USB_DRIVER_PATH"
    fi
    echo ""
else
    echo -e "${YELLOW}[2-3]${NC} Skipping USB reset (device info unavailable, but ALSA shows device exists)"
    echo "  This is OK - we can still configure GPIO and DSP"
    echo ""
fi

# Method 3: Reinitialize GPIO pins after reset
echo -e "${YELLOW}[4]${NC} Reinitializing GPIO pins..."
XVF_HOST_PATH="$HOME/reSpeaker_XVF3800_USB_4MIC_ARRAY/host_control/jetson/xvf_host"

if [ -f "$XVF_HOST_PATH" ]; then
    chmod +x "$XVF_HOST_PATH" 2>/dev/null || true
    
    # Wait a bit for device to be ready
    sleep 1
    
    # Set GPIO pins
    echo "  Setting X0D30 (mic mute) = 0..."
    "$XVF_HOST_PATH" GPO_WRITE_VALUE 30 0 2>/dev/null && echo "    ✅ Mic unmuted" || echo "    ⚠️  Failed"
    
    echo "  Setting X0D31 (amp enable) = 0..."
    "$XVF_HOST_PATH" GPO_WRITE_VALUE 31 0 2>/dev/null && echo "    ✅ Amp enabled" || echo "    ⚠️  Failed"
    
    # Verify
    sleep 0.5
    GPIO_OUTPUT=$("$XVF_HOST_PATH" GPO_READ_VALUES 2>&1)
    gpo_line=$(echo "$GPIO_OUTPUT" | grep 'GPO_READ_VALUES' | head -1)
    if [ -n "$gpo_line" ]; then
        values_str=$(echo "$gpo_line" | sed 's/GPO_READ_VALUES //')
        values=($values_str)
        if [ ${#values[@]} -ge 5 ]; then
            x0d30=${values[1]}
            x0d31=${values[2]}
            echo "  📊 GPIO state: X0D30=$x0d30, X0D31=$x0d31"
            if [ "$x0d30" == "0" ] && [ "$x0d31" == "0" ]; then
                echo -e "  ✅ GPIO pins correctly configured"
            else
                echo -e "  ⚠️  GPIO pins may not be correct"
            fi
        fi
    fi
else
    echo -e "  ⚠️  xvf_host not found: $XVF_HOST_PATH"
fi
echo ""

# Method 4: Run full tuning script
echo -e "${YELLOW}[5]${NC} Running full tuning script..."
TUNE_SCRIPT="$HOME/LedgerAI/setup/scripts/tune_xvf3800.py"
if [ -f "$TUNE_SCRIPT" ]; then
    echo "  Running: python3 $TUNE_SCRIPT agc_20_ec"
    sudo python3 "$TUNE_SCRIPT" agc_20_ec 2>&1 | grep -E "\[GPIO\]|\[LED\]|✅|❌|⚠️" || true
    echo -e "  ✅ Tuning script completed"
else
    echo -e "  ⚠️  Tuning script not found: $TUNE_SCRIPT"
fi
echo ""

# Wait for ALSA to re-detect
echo -e "${YELLOW}[6]${NC} Waiting for ALSA to re-detect device..."
sleep 2

if arecord -l 2>/dev/null | grep -q -i "XVF3800\|reSpeaker"; then
    echo -e "  ✅ Device visible to ALSA"
    arecord -l 2>/dev/null | grep -i "XVF3800\|reSpeaker" | sed 's/^/    /'
else
    echo -e "  ⚠️  Device not yet visible to ALSA"
fi
echo ""

echo -e "${GREEN}==========================================${NC}"
echo -e "${GREEN}  Reset Complete${NC}"
echo -e "${GREEN}==========================================${NC}"
echo ""
echo "Test microphone capture:"
echo "  python3 ~/LedgerAI/setup/scripts/test_transcription.py"
echo ""


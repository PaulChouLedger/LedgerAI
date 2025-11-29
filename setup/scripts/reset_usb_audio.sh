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

# Find XVF3800 USB device
USB_DEVICE=$(lsusb | grep "2886:0018" | head -1)
if [ -z "$USB_DEVICE" ]; then
    echo -e "${RED}❌ XVF3800 USB device not found${NC}"
    exit 1
fi

# Extract bus and device numbers
BUS=$(echo "$USB_DEVICE" | awk '{print $2}')
DEV=$(echo "$USB_DEVICE" | awk '{print $4}' | sed 's/://')

echo -e "${YELLOW}[1]${NC} Found XVF3800 at USB $BUS:$DEV"
echo ""

# Method 1: Reset USB device via sysfs (requires root)
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


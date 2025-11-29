#!/bin/bash
# Diagnose arecord I/O errors
# Helps identify why hardware device access is failing

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  arecord I/O Error Diagnostics${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Check if device argument provided
DEVICE="${1:-hw:3,0}"
echo -e "${YELLOW}[INFO]${NC} Testing device: $DEVICE"
echo ""

# 1. List all audio devices
echo -e "${GREEN}[STEP 1]${NC} Listing all audio devices..."
echo "----------------------------------------"
aplay -l 2>/dev/null || echo "aplay -l failed"
echo ""
arecord -l 2>/dev/null || echo "arecord -l failed"
echo ""

# 2. Check if device is busy
echo -e "${GREEN}[STEP 2]${NC} Checking if device is in use..."
echo "----------------------------------------"
if lsof 2>/dev/null | grep -q "snd"; then
    echo "Processes using audio devices:"
    lsof 2>/dev/null | grep "snd" | head -10
else
    echo "No processes found using audio devices"
fi
echo ""

# 3. Check device permissions
echo -e "${GREEN}[STEP 3]${NC} Checking device permissions..."
echo "----------------------------------------"
if [ -d "/dev/snd" ]; then
    echo "Audio device files:"
    ls -la /dev/snd/ | head -20
else
    echo "/dev/snd directory not found"
fi
echo ""

# 4. Check user groups
echo -e "${GREEN}[STEP 4]${NC} Checking user groups..."
echo "----------------------------------------"
echo "Current user: $(whoami)"
echo "Groups: $(groups)"
if groups | grep -q audio; then
    echo -e "${GREEN}✅${NC} User is in 'audio' group"
else
    echo -e "${RED}❌${NC} User is NOT in 'audio' group"
    echo "   Fix: sudo usermod -aG audio $(whoami)"
fi
echo ""

# 5. Try different device access methods
echo -e "${GREEN}[STEP 5]${NC} Testing different device access methods..."
echo "----------------------------------------"

# Try hw: device
echo -n "Testing hw:$DEVICE... "
if timeout 2 arecord -D "$DEVICE" -f S16_LE -r 16000 -c 1 -d 1 /dev/null 2>&1 | grep -q "error\|Error\|ERROR"; then
    echo -e "${RED}❌ FAILED${NC}"
    timeout 2 arecord -D "$DEVICE" -f S16_LE -r 16000 -c 1 -d 1 /dev/null 2>&1 | head -3
else
    echo -e "${GREEN}✅ OK${NC}"
fi

# Extract card number
CARD_NUM=$(echo "$DEVICE" | sed 's/hw:\([0-9]*\),.*/\1/')
echo -n "Testing plug:hw:$CARD_NUM,0... "
if timeout 2 arecord -D "plug:hw:$CARD_NUM,0" -f S16_LE -r 16000 -c 1 -d 1 /dev/null 2>&1 | grep -q "error\|Error\|ERROR"; then
    echo -e "${RED}❌ FAILED${NC}"
    timeout 2 arecord -D "plug:hw:$CARD_NUM,0" -f S16_LE -r 16000 -c 1 -d 1 /dev/null 2>&1 | head -3
else
    echo -e "${GREEN}✅ OK${NC}"
fi

# Try default
echo -n "Testing default device... "
if timeout 2 arecord -D default -f S16_LE -r 16000 -c 1 -d 1 /dev/null 2>&1 | grep -q "error\|Error\|ERROR"; then
    echo -e "${RED}❌ FAILED${NC}"
    timeout 2 arecord -D default -f S16_LE -r 16000 -c 1 -d 1 /dev/null 2>&1 | head -3
else
    echo -e "${GREEN}✅ OK${NC}"
fi
echo ""

# 6. Check ALSA configuration
echo -e "${GREEN}[STEP 6]${NC} Checking ALSA configuration..."
echo "----------------------------------------"
if [ -f "$HOME/.asoundrc" ]; then
    echo -e "${YELLOW}Found:${NC} $HOME/.asoundrc"
    echo "First 20 lines:"
    head -20 "$HOME/.asoundrc"
else
    echo "No ~/.asoundrc found"
fi
echo ""

# 7. Check for PipeWire
echo -e "${GREEN}[STEP 7]${NC} Checking PipeWire status..."
echo "----------------------------------------"
if command -v wpctl >/dev/null 2>&1; then
    echo "PipeWire status (wpctl):"
    wpctl status 2>/dev/null | head -30 || echo "Could not get PipeWire status"
    echo ""
    echo "PipeWire sources (XVF3800/reSpeaker):"
    wpctl status 2>/dev/null | grep -i "XVF3800\|reSpeaker" || echo "  (none found)"
    echo ""
    echo "PipeWire sinks (UACDemoV1.0):"
    wpctl status 2>/dev/null | grep -i "UACDemo\|Jieli" || echo "  (none found)"
else
    echo "wpctl not found - PipeWire may not be installed"
fi
echo ""

# 7.5. Check PipeWire services
echo -e "${GREEN}[STEP 7.5]${NC} Checking PipeWire services..."
echo "----------------------------------------"
if systemctl --user is-active --quiet pipewire.service 2>/dev/null; then
    echo -e "${GREEN}✅${NC} pipewire.service is active"
else
    echo -e "${RED}❌${NC} pipewire.service is not active"
    echo "   Start: systemctl --user start pipewire.service"
fi

if systemctl --user is-active --quiet wireplumber.service 2>/dev/null; then
    echo -e "${GREEN}✅${NC} wireplumber.service is active"
else
    echo -e "${RED}❌${NC} wireplumber.service is not active"
    echo "   Start: systemctl --user start wireplumber.service"
fi
echo ""

# 7.6. Check PipeWire nodes (detailed)
if command -v pw-cli >/dev/null 2>&1; then
    echo -e "${GREEN}[STEP 7.6]${NC} Checking PipeWire nodes (detailed)..."
    echo "----------------------------------------"
    echo "Audio input nodes:"
    pw-cli list-objects Node 2>/dev/null | grep -A 10 "XVF3800\|reSpeaker" | head -20 || echo "  (none found)"
    echo ""
fi

# 8. Recommendations
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  Recommendations${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo "1. Try using 'plug:' instead of 'hw:' for automatic format conversion:"
echo "   arecord -D plug:hw:$CARD_NUM,0 -f S16_LE -r 16000 -c 1 test.wav"
echo ""
echo "2. Try using 'default' device:"
echo "   arecord -D default -f S16_LE -r 16000 -c 1 test.wav"
echo ""
echo "3. Check PipeWire device status:"
echo "   wpctl status | grep -i 'XVF3800\|reSpeaker'"
echo "   wpctl status | grep -i 'sink\|source'"
echo ""
echo "4. Restart PipeWire services if needed:"
echo "   systemctl --user restart pipewire.service"
echo "   systemctl --user restart wireplumber.service"
echo ""
echo "5. Check if another process is using the device:"
echo "   sudo fuser -v /dev/snd/*"
echo "   lsof | grep snd"
echo ""
echo "6. Kill processes using audio (if safe):"
echo "   sudo fuser -k /dev/snd/*"
echo ""
echo "7. Try with sudo (to test permissions):"
echo "   sudo arecord -D $DEVICE -f S16_LE -r 16000 -c 1 test.wav"
echo ""
echo "8. Check USB device connection:"
echo "   lsusb | grep -i audio"
echo "   dmesg | tail -20 | grep -i audio"
echo ""
echo "9. Check PipeWire logs if issues persist:"
echo "   journalctl --user -u pipewire.service -n 50"
echo "   journalctl --user -u wireplumber.service -n 50"
echo ""


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

# 7.1. Check device states (suspended/idle/active) using pw-cli
echo -e "${GREEN}[STEP 7.1]${NC} Checking device states (suspended/idle/active)..."
echo "----------------------------------------"
if command -v pw-cli >/dev/null 2>&1; then
    echo "XVF3800 device states:"
    echo ""
    
    # Extract node IDs for XVF3800 devices (look for "id 42" pattern)
    NODE_IDS=$(pw-cli list-objects Node 2>/dev/null | grep -B 5 "XVF3800\|reSpeaker" | grep "^[[:space:]]*id [0-9]" | awk '{print $2}' | sort -u)
    
    if [ -n "$NODE_IDS" ]; then
        for NODE_ID in $NODE_IDS; do
            # Get full node info
            NODE_INFO=$(pw-cli info "$NODE_ID" 2>/dev/null)
            
            if [ -z "$NODE_INFO" ]; then
                continue
            fi
            
            # Extract node properties
            NODE_NAME=$(echo "$NODE_INFO" | grep "node.name" | head -1 | sed 's/.*= "\(.*\)"/\1/')
            NODE_DESC=$(echo "$NODE_INFO" | grep "node.description" | head -1 | sed 's/.*= "\(.*\)"/\1/')
            MEDIA_CLASS=$(echo "$NODE_INFO" | grep "media.class" | head -1 | sed 's/.*= "\(.*\)"/\1/')
            
            # Check suspend-on-idle setting
            SUSPEND_ON_IDLE=$(echo "$NODE_INFO" | grep "device.suspend-on-idle" | head -1 | sed 's/.*= "\(.*\)"/\1/' || echo "not set")
            
            # Check port information (input/output ports)
            INPUT_PORTS=$(echo "$NODE_INFO" | grep "input ports:" | head -1 | sed 's/.*input ports: \([0-9]*\)\/\([0-9]*\).*/\1\/\2/' || echo "0/0")
            OUTPUT_PORTS=$(echo "$NODE_INFO" | grep "output ports:" | head -1 | sed 's/.*output ports: \([0-9]*\)\/\([0-9]*\).*/\1\/\2/' || echo "0/0")
            
            echo "Node ID $NODE_ID:"
            [ -n "$NODE_DESC" ] && echo "  Description: $NODE_DESC"
            [ -n "$NODE_NAME" ] && echo "  Name: $NODE_NAME"
            [ -n "$MEDIA_CLASS" ] && echo "  Type: $MEDIA_CLASS"
            echo "  Input ports: $INPUT_PORTS"
            echo "  Output ports: $OUTPUT_PORTS"
            echo "  Suspend-on-idle: $SUSPEND_ON_IDLE"
            
            # Determine status based on port counts and media class
            # Note: In PipeWire, Audio/Source (microphone) OUTPUTS audio, so it has OUTPUT ports
            #       Audio/Sink (speaker) RECEIVES audio, so it has INPUT ports
            if echo "$MEDIA_CLASS" | grep -q "Source\|Input"; then
                # For Audio/Source (microphone): 
                # - Input ports: 0/0 is NORMAL (it doesn't receive audio)
                # - Output ports: >0 means it's OUTPUTTING audio (capturing)
                OUTPUT_COUNT=$(echo "$OUTPUT_PORTS" | cut -d'/' -f1)
                OUTPUT_TOTAL=$(echo "$OUTPUT_PORTS" | cut -d'/' -f2)
                if [ "$OUTPUT_COUNT" -gt 0 ]; then
                    echo "  Status: ✅ ACTIVE (has $OUTPUT_COUNT/$OUTPUT_TOTAL output port(s) - capturing audio)"
                    echo "  Note: Input ports 0/0 is normal for Audio/Source (microphones don't receive audio)"
                else
                    echo "  Status: ⚠️  IDLE/SUSPENDED (no output ports - not capturing)"
                    echo "  Note: Input ports 0/0 is normal, but 0 output ports means not active"
                fi
            elif echo "$MEDIA_CLASS" | grep -q "Sink\|Output"; then
                # For Audio/Sink (speaker):
                # - Output ports: 0/0 is NORMAL (it doesn't output audio)
                # - Input ports: >0 means it's RECEIVING audio (playing)
                INPUT_COUNT=$(echo "$INPUT_PORTS" | cut -d'/' -f1)
                INPUT_TOTAL=$(echo "$INPUT_PORTS" | cut -d'/' -f2)
                if [ "$INPUT_COUNT" -gt 0 ]; then
                    echo "  Status: ✅ ACTIVE (has $INPUT_COUNT/$INPUT_TOTAL input port(s) - playing audio)"
                    echo "  Note: Output ports 0/0 is normal for Audio/Sink (speakers don't output audio)"
                else
                    echo "  Status: ⚠️  IDLE/SUSPENDED (no input ports - not playing)"
                    echo "  Note: Output ports 0/0 is normal, but 0 input ports means not active"
                fi
            else
                echo "  Status: ℹ️  Unknown media class"
            fi
            
            # For input nodes specifically, check if they can capture
            if echo "$MEDIA_CLASS" | grep -q "Source\|Input"; then
                echo "  (Input/Source node)"
                # Check if device is actually accessible via ALSA
                ALSA_NAME=$(echo "$NODE_NAME" | sed 's/alsa_\(input\|output\)\.//' | sed 's/\..*//')
                if [ -n "$ALSA_NAME" ]; then
                    echo "  ALSA device: $ALSA_NAME"
                    # Check if ALSA can see this device
                    if arecord -l 2>/dev/null | grep -q "$ALSA_NAME\|XVF3800"; then
                        echo "  ALSA status: ✅ Device visible to ALSA"
                    else
                        echo "  ALSA status: ⚠️  Device not visible to ALSA"
                    fi
                fi
            fi
            echo ""
        done
    else
        echo "  (no XVF3800 nodes found)"
    fi
    
    # Alternative: Use pw-dump for more detailed JSON output
    if command -v pw-dump >/dev/null 2>&1; then
        echo "Device states (from pw-dump - more detailed):"
        pw-dump 2>/dev/null | jq -r '.[] | select(.info.props."node.name" | contains("XVF3800") or contains("reSpeaker")) | "\(.id): \(.info.props."node.name") - State: \(.info.state // "unknown")"' 2>/dev/null || {
            # Fallback if jq not available
            pw-dump 2>/dev/null | grep -A 30 "XVF3800\|reSpeaker" | grep -E '"id"|"name"|"state"' | head -20 | sed 's/^/  /' || echo "  (could not get state)"
        }
    fi
    echo ""
else
    echo "pw-cli not found - cannot check detailed device states"
    echo "Install: sudo apt install pipewire-cli"
    echo ""
    echo "Alternative: Check using pw-dump:"
    if command -v pw-dump >/dev/null 2>&1; then
        pw-dump 2>/dev/null | grep -A 20 "XVF3800\|reSpeaker" | grep -E '"state"|"suspend"' | head -10 || echo "  (could not get state)"
    else
        echo "  pw-dump also not found"
    fi
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

# 7.6. Check PipeWire nodes (detailed with state)
if command -v pw-cli >/dev/null 2>&1; then
    echo -e "${GREEN}[STEP 7.6]${NC} Checking PipeWire nodes (detailed with state)..."
    echo "----------------------------------------"
    echo "XVF3800 node details (showing state):"
    # Get node IDs for XVF3800 devices
    NODE_IDS=$(pw-cli list-objects Node 2>/dev/null | grep -B 5 "XVF3800\|reSpeaker" | grep '"id"' | head -5 | sed 's/.*"id": \([0-9]*\).*/\1/')
    
    if [ -n "$NODE_IDS" ]; then
        for NODE_ID in $NODE_IDS; do
            echo ""
            echo "Node ID $NODE_ID:"
            pw-cli info "$NODE_ID" 2>/dev/null | grep -E "id|name|state|suspend|media\.class|device\.suspend-on-idle" | head -15 || echo "  (could not get info for node $NODE_ID)"
        done
    else
        echo "  (no XVF3800 nodes found)"
    fi
    echo ""
    
    # Alternative: use pw-dump if available
    if command -v pw-dump >/dev/null 2>&1; then
        echo "Using pw-dump for detailed state:"
        pw-dump 2>/dev/null | grep -A 30 "XVF3800\|reSpeaker" | grep -E "id|name|state|suspend|media\.class" | head -20 || echo "  (could not dump state)"
    fi
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


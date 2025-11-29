#!/bin/bash
# Replace PulseAudio with PipeWire
# Run this script to migrate from PulseAudio to PipeWire immediately

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  Replace PulseAudio with PipeWire${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

# Check if running as root
if [ "$EUID" -eq 0 ]; then 
    echo -e "${RED}ERROR:${NC} Please run this script as a regular user (not root)"
    exit 1
fi

AURA_HOME="${HOME:-/home/ledger}"

# Step 1: Stop and disable PulseAudio
echo -e "${YELLOW}[STEP 1]${NC} Stopping PulseAudio..."
systemctl --user stop pulseaudio 2>/dev/null || true
systemctl --user stop pulseaudio.socket 2>/dev/null || true
systemctl --user disable pulseaudio 2>/dev/null || true
systemctl --user disable pulseaudio.socket 2>/dev/null || true
sudo systemctl stop pulseaudio 2>/dev/null || true
sudo systemctl disable pulseaudio 2>/dev/null || true

# Kill any running PulseAudio processes
echo -e "${YELLOW}[STEP 2]${NC} Killing PulseAudio processes..."
pkill -9 pulseaudio 2>/dev/null || true
sleep 1

# Step 2: Install PipeWire
echo -e "${YELLOW}[STEP 3]${NC} Installing PipeWire..."
sudo apt update
sudo apt install -y \
    pipewire \
    pipewire-pulse \
    wireplumber \
    libspa-0.2-bluetooth \
    libspa-0.2-jack || {
    echo -e "${RED}ERROR:${NC} Failed to install PipeWire packages"
    exit 1
}

# Step 3: Remove PulseAudio
echo -e "${YELLOW}[STEP 4]${NC} Removing PulseAudio packages..."
sudo apt remove -y pulseaudio pulseaudio-utils 2>/dev/null || true
sudo apt autoremove -y 2>/dev/null || true

# Step 4: Configure PipeWire to start on boot
echo -e "${YELLOW}[STEP 5]${NC} Configuring PipeWire to start on boot..."
systemctl --user enable pipewire.service 2>/dev/null || true
systemctl --user enable pipewire-pulse.service 2>/dev/null || true
systemctl --user enable wireplumber.service 2>/dev/null || true

# Step 5: Start PipeWire
echo -e "${YELLOW}[STEP 6]${NC} Starting PipeWire services..."
systemctl --user start pipewire.service 2>/dev/null || true
systemctl --user start pipewire-pulse.service 2>/dev/null || true
systemctl --user start wireplumber.service 2>/dev/null || true

# Wait for PipeWire to initialize
echo -e "${YELLOW}[STEP 7]${NC} Waiting for PipeWire to initialize..."
sleep 3

# Step 6: Configure PipeWire to prevent USB audio suspension
echo -e "${YELLOW}[STEP 8]${NC} Configuring PipeWire to prevent USB audio suspension..."
PIPEWIRE_CONFIG_DIR="$AURA_HOME/.config/pipewire"
mkdir -p "$PIPEWIRE_CONFIG_DIR/pipewire-pulse.d"

# Create custom config to prevent USB device suspension
cat > "$PIPEWIRE_CONFIG_DIR/pipewire-pulse.d/99-usb-audio-no-suspend.conf" << 'EOFPIPEWIRE'
# Prevent USB audio devices from auto-suspending
# This ensures USB microphones stay IDLE instead of SUSPENDED
context.properties = {
    default.clock.rate = 48000
    default.clock.quantum = 1024
    default.clock.min-quantum = 32
    default.clock.max-quantum = 8192
}

# Disable suspend-on-idle for USB audio devices
pulse.properties = {
    server.address = [
        "unix:native"
        "unix:/tmp/pulse-socket"
    ]
    server.dont-migrate = true
    server.allow-pulseaudio-override = false
}

# Module configuration
pulse.rules = [
    {
        matches = [
            {
                device.name = "~alsa.*"
            }
        ]
        actions = {
            update-props = {
                device.suspend-on-idle = false
            }
        }
    }
]
EOFPIPEWIRE

chmod 644 "$PIPEWIRE_CONFIG_DIR/pipewire-pulse.d/99-usb-audio-no-suspend.conf"
echo -e "${GREEN}✅${NC} PipeWire configuration created"

# Step 7: Restart PipeWire to apply configuration
echo -e "${YELLOW}[STEP 9]${NC} Restarting PipeWire to apply configuration..."
systemctl --user restart pipewire.service 2>/dev/null || true
systemctl --user restart pipewire-pulse.service 2>/dev/null || true
sleep 2

# Step 8: Verify PipeWire is working
echo -e "${YELLOW}[STEP 10]${NC} Verifying PipeWire installation..."
if systemctl --user is-active --quiet pipewire.service; then
    echo -e "${GREEN}✅${NC} PipeWire service is active"
else
    echo -e "${RED}⚠️${NC} PipeWire service not active - may need manual start"
fi

if command -v pactl >/dev/null 2>&1; then
    if pactl info 2>/dev/null | grep -q "pipewire"; then
        echo -e "${GREEN}✅${NC} PipeWire is active (pactl shows pipewire)"
    else
        echo -e "${YELLOW}⚠️${NC} PipeWire may not be fully initialized"
    fi
    
    echo ""
    echo "Available audio sources:"
    pactl list short sources 2>/dev/null | head -5 || echo "  (none found)"
    echo ""
    echo "Available audio sinks:"
    pactl list short sinks 2>/dev/null | head -5 || echo "  (none found)"
fi

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  Migration Complete!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "PipeWire has replaced PulseAudio."
echo "USB audio devices should now stay IDLE instead of SUSPENDED."
echo ""
echo "To verify:"
echo "  pactl list short sources | grep XVF3800"
echo "  (Should show IDLE instead of SUSPENDED)"
echo ""
echo "If you need to restart PipeWire:"
echo "  systemctl --user restart pipewire.service"
echo "  systemctl --user restart pipewire-pulse.service"
echo ""


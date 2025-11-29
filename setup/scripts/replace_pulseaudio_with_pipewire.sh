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

# Step 2: Install PipeWire (standalone, no PulseAudio compatibility)
echo -e "${YELLOW}[STEP 3]${NC} Installing PipeWire (standalone)..."
sudo apt update
sudo apt install -y \
    pipewire \
    wireplumber \
    libspa-0.2-bluetooth \
    libspa-0.2-jack \
    pipewire-audio-client-libraries || {
    echo -e "${RED}ERROR:${NC} Failed to install PipeWire packages"
    exit 1
}

echo -e "${GREEN}✅${NC} PipeWire installed (standalone mode - no PulseAudio compatibility)"
echo -e "${YELLOW}[INFO]${NC} Using PipeWire native commands (wpctl/pw-cli)"

# Step 3: Remove PulseAudio completely (including pipewire-pulse)
echo -e "${YELLOW}[STEP 4]${NC} Removing PulseAudio and pipewire-pulse completely..."
# Remove PulseAudio server and client tools
sudo apt remove -y pulseaudio pulseaudio-utils 2>/dev/null || true
# Remove pipewire-pulse if installed (PulseAudio compatibility layer)
sudo apt remove -y pipewire-pulse 2>/dev/null || true
sudo apt autoremove -y 2>/dev/null || true
echo -e "${GREEN}✅${NC} PulseAudio and pipewire-pulse removed - using PipeWire standalone"

# Step 4: Configure PipeWire to start on boot
echo -e "${YELLOW}[STEP 5]${NC} Configuring PipeWire to start on boot..."
systemctl --user enable pipewire.service 2>/dev/null || true
systemctl --user enable wireplumber.service 2>/dev/null || true
echo -e "${GREEN}✅${NC} PipeWire services enabled (standalone mode)"

# Step 5: Start PipeWire services in correct order
echo -e "${YELLOW}[STEP 6]${NC} Starting PipeWire services (standalone)..."
echo "  Starting pipewire.service..."
systemctl --user start pipewire.service 2>/dev/null || true
sleep 1

echo "  Starting wireplumber.service..."
systemctl --user start wireplumber.service 2>/dev/null || true
sleep 2

# Wait for PipeWire to fully initialize
echo -e "${YELLOW}[STEP 7]${NC} Waiting for PipeWire to initialize..."
sleep 2

# Verify PipeWire is running
if systemctl --user is-active --quiet pipewire.service; then
    echo -e "${GREEN}✅${NC} pipewire.service is active"
else
    echo -e "${RED}⚠️${NC} pipewire.service not active - may need manual start"
fi

if systemctl --user is-active --quiet wireplumber.service; then
    echo -e "${GREEN}✅${NC} wireplumber.service is active"
else
    echo -e "${RED}⚠️${NC} wireplumber.service not active - may need manual start"
fi

# Verify wpctl is available (PipeWire native command)
if command -v wpctl >/dev/null 2>&1; then
    echo -e "${GREEN}✅${NC} wpctl is available (PipeWire native command)"
else
    echo -e "${RED}⚠️${NC} wpctl not found - may need to install wireplumber"
fi

# Step 6: Configure Wireplumber to prevent USB audio suspension
echo -e "${YELLOW}[STEP 9]${NC} Configuring Wireplumber to prevent USB audio suspension..."
WIREPLUMBER_CONFIG_DIR="$AURA_HOME/.config/wireplumber"
mkdir -p "$WIREPLUMBER_CONFIG_DIR/main.lua.d"

# Create Wireplumber policy to prevent USB audio device suspension
cat > "$WIREPLUMBER_CONFIG_DIR/main.lua.d/99-usb-audio-no-suspend.lua" << 'EOFWIREPLUMBER'
-- Prevent USB audio devices from auto-suspending
-- This ensures USB microphones stay IDLE instead of SUSPENDED

alsa_monitor.rules = {
  {
    matches = {
      {
        { "device.name", "matches", "alsa.*" },
      },
    },
    apply_properties = {
      ["device.suspend-on-idle"] = false,
    },
  },
}

-- Also prevent suspension for USB audio devices specifically
alsa_monitor.rules = {
  {
    matches = {
      {
        { "device.name", "matches", "alsa.*" },
      },
    },
    apply_properties = {
      ["device.suspend-on-idle"] = false,
    },
  },
  {
    matches = {
      {
        { "node.name", "matches", "alsa.*" },
      },
    },
    apply_properties = {
      ["node.suspend-on-idle"] = false,
    },
  },
}
EOFWIREPLUMBER

chmod 644 "$WIREPLUMBER_CONFIG_DIR/main.lua.d/99-usb-audio-no-suspend.lua"
echo -e "${GREEN}✅${NC} Wireplumber configuration created"

# Step 7: Restart PipeWire services to apply configuration
echo -e "${YELLOW}[STEP 10]${NC} Restarting PipeWire services to apply configuration..."
echo "  Restarting wireplumber.service..."
systemctl --user restart wireplumber.service 2>/dev/null || true
sleep 1
echo "  Restarting pipewire.service..."
systemctl --user restart pipewire.service 2>/dev/null || true
sleep 2

# Verify devices are active (using wpctl - PipeWire native)
echo -e "${YELLOW}[INFO]${NC} Checking device status with wpctl..."
if command -v wpctl >/dev/null 2>&1; then
    XVF3800_DEVICES=$(wpctl status 2>/dev/null | grep -i "XVF3800\|reSpeaker" | wc -l)
    if [ "$XVF3800_DEVICES" -gt 0 ]; then
        echo -e "${GREEN}✅${NC} XVF3800 devices found in PipeWire"
        wpctl status 2>/dev/null | grep -i "XVF3800\|reSpeaker" | head -5
    else
        echo -e "${YELLOW}⚠️${NC} XVF3800 devices not found"
    fi
fi

# Step 8: Verify PipeWire is working
echo -e "${YELLOW}[STEP 11]${NC} Verifying PipeWire installation..."
if systemctl --user is-active --quiet pipewire.service; then
    echo -e "${GREEN}✅${NC} PipeWire service is active"
else
    echo -e "${RED}⚠️${NC} PipeWire service not active - may need manual start"
fi

# Note: pipewire-pulse should provide pactl, but wpctl is the native PipeWire command
if ! command -v pactl >/dev/null 2>&1; then
    echo -e "${YELLOW}[INFO]${NC} pactl not available - using wpctl (PipeWire native) instead"
    echo -e "${YELLOW}[INFO]${NC} This is normal - wpctl is the recommended PipeWire command"
fi

# Use wpctl (PipeWire native) for verification
if command -v wpctl >/dev/null 2>&1; then
    echo ""
    echo "PipeWire status (wpctl - native):"
    wpctl status 2>/dev/null | head -30 || echo "  (wpctl not working)"
    echo ""
    echo "XVF3800 devices:"
    wpctl status 2>/dev/null | grep -i "XVF3800\|reSpeaker" || echo "  (none found)"
else
    echo ""
    echo "⚠️  wpctl not available - PipeWire may not be properly installed"
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
echo "  wpctl status | grep XVF3800"
echo "  (Should show device as active/available)"
echo ""
echo "Note: Using PipeWire standalone (no PulseAudio compatibility)"
echo "      All audio operations use wpctl (PipeWire native command)"
echo ""
echo "If you need to restart PipeWire:"
echo "  systemctl --user restart wireplumber.service"
echo "  systemctl --user restart pipewire.service"
echo "  systemctl --user restart pipewire-pulse.service"
echo ""


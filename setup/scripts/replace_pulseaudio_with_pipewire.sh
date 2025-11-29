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
    libspa-0.2-jack \
    pipewire-audio-client-libraries || {
    echo -e "${RED}ERROR:${NC} Failed to install PipeWire packages"
    exit 1
}

# Ensure pactl is available (pipewire-pulse provides it, but may need symlink)
if ! command -v pactl >/dev/null 2>&1; then
    echo -e "${YELLOW}[INFO]${NC} pactl not found, checking if pipewire-pulse provides it..."
    if [ -f "/usr/bin/pw-cli" ]; then
        echo -e "${YELLOW}[INFO]${NC} Using PipeWire native commands (wpctl/pw-cli) instead of pactl"
    fi
fi

# Step 3: Remove PulseAudio (but keep pulseaudio-utils for pactl command)
echo -e "${YELLOW}[STEP 4]${NC} Removing PulseAudio server (keeping pulseaudio-utils for pactl)..."
# Remove PulseAudio server but keep pulseaudio-utils (contains pactl command)
sudo apt remove -y pulseaudio 2>/dev/null || true
# Check if pulseaudio-utils is installed, if not install it (needed for pactl)
if ! command -v pactl >/dev/null 2>&1; then
    echo -e "${YELLOW}[INFO]${NC} Installing pulseaudio-utils for pactl command (client tools only)..."
    sudo apt install -y pulseaudio-utils 2>/dev/null || {
        echo -e "${YELLOW}[INFO]${NC} pulseaudio-utils not available - will use wpctl instead"
    }
fi
sudo apt autoremove -y 2>/dev/null || true

# Step 4: Configure PipeWire to start on boot
echo -e "${YELLOW}[STEP 5]${NC} Configuring PipeWire to start on boot..."
systemctl --user enable pipewire.service 2>/dev/null || true
systemctl --user enable pipewire-pulse.service 2>/dev/null || true
systemctl --user enable wireplumber.service 2>/dev/null || true

# Step 5: Start PipeWire services in correct order
echo -e "${YELLOW}[STEP 6]${NC} Starting PipeWire services..."
echo "  Starting pipewire.service..."
systemctl --user start pipewire.service 2>/dev/null || true
sleep 1

echo "  Starting wireplumber.service..."
systemctl --user start wireplumber.service 2>/dev/null || true
sleep 1

echo "  Starting pipewire-pulse.service..."
systemctl --user start pipewire-pulse.service 2>/dev/null || true
sleep 2

# Verify pipewire-pulse is running (provides pactl compatibility)
echo -e "${YELLOW}[STEP 7]${NC} Verifying pipewire-pulse is running..."
if systemctl --user is-active --quiet pipewire-pulse.service; then
    echo -e "${GREEN}✅${NC} pipewire-pulse.service is active"
else
    echo -e "${RED}⚠️${NC} pipewire-pulse.service is not active - starting..."
    systemctl --user start pipewire-pulse.service
    sleep 2
    if systemctl --user is-active --quiet pipewire-pulse.service; then
        echo -e "${GREEN}✅${NC} pipewire-pulse.service started successfully"
    else
        echo -e "${RED}⚠️${NC} Failed to start pipewire-pulse.service"
        echo -e "${YELLOW}[INFO]${NC} Check logs: journalctl --user -u pipewire-pulse.service"
    fi
fi

# Wait for PipeWire to fully initialize
echo -e "${YELLOW}[STEP 8]${NC} Waiting for PipeWire to initialize..."
sleep 3

# Verify pactl is available (provided by pipewire-pulse)
if command -v pactl >/dev/null 2>&1; then
    echo -e "${GREEN}✅${NC} pactl is available (pipewire-pulse compatibility)"
    # Test if pactl can connect to pipewire-pulse
    if pactl info 2>/dev/null | grep -q "pipewire\|PipeWire"; then
        echo -e "${GREEN}✅${NC} pactl is connected to PipeWire"
    else
        echo -e "${YELLOW}⚠️${NC} pactl found but may not be connected to PipeWire"
    fi
else
    echo -e "${YELLOW}⚠️${NC} pactl not found - wpctl will be used instead"
    echo -e "${YELLOW}[INFO]${NC} This is OK - wpctl is the native PipeWire command"
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
        { "device.name", "matches", "alsa.*usb*" },
      },
    },
    apply_properties = {
      ["device.suspend-on-idle"] = false,
      ["device.session.suspend-timeout-seconds"] = 0,
    },
  },
}
EOFWIREPLUMBER

chmod 644 "$WIREPLUMBER_CONFIG_DIR/main.lua.d/99-usb-audio-no-suspend.lua"
echo -e "${GREEN}✅${NC} Wireplumber configuration created"

# Step 7: Restart PipeWire and Wireplumber to apply configuration
echo -e "${YELLOW}[STEP 10]${NC} Restarting PipeWire and Wireplumber to apply configuration..."
systemctl --user restart wireplumber.service 2>/dev/null || true
systemctl --user restart pipewire.service 2>/dev/null || true
systemctl --user restart pipewire-pulse.service 2>/dev/null || true
sleep 3

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
    echo "PipeWire status (wpctl):"
    wpctl status 2>/dev/null | head -20 || echo "  (wpctl not working)"
    echo ""
    echo "XVF3800 devices:"
    wpctl status 2>/dev/null | grep -i "XVF3800\|reSpeaker" || echo "  (none found)"
fi

# Also check pactl if available (pipewire-pulse compatibility)
if command -v pactl >/dev/null 2>&1; then
    echo ""
    echo "PipeWire sources (pactl - pipewire-pulse compatibility):"
    pactl list short sources 2>/dev/null | head -5 || echo "  (none found)"
    echo ""
    echo "PipeWire sinks (pactl):"
    pactl list short sinks 2>/dev/null | head -5 || echo "  (none found)"
    echo ""
    echo "XVF3800 sources (pactl):"
    pactl list short sources 2>/dev/null | grep -i "XVF3800\|reSpeaker" || echo "  (none found)"
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
if command -v pactl >/dev/null 2>&1; then
    echo "  pactl list short sources | grep XVF3800"
fi
echo "  (Should show device as available, not suspended)"
echo ""
echo "Note: wpctl is the native PipeWire command"
echo "      pactl is provided by pipewire-pulse for compatibility"
echo ""
echo "If you need to restart PipeWire:"
echo "  systemctl --user restart wireplumber.service"
echo "  systemctl --user restart pipewire.service"
echo "  systemctl --user restart pipewire-pulse.service"
echo ""


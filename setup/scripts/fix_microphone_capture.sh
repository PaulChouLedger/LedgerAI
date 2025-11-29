#!/bin/bash
# Fix microphone capture issues after install_aura_bootable.sh
# This script applies all necessary fixes to allow direct ALSA access

set -e

echo "=========================================="
echo "  Fixing Microphone Capture"
echo "=========================================="
echo ""

# 1. Comment out module-alsa-source in PulseAudio config
echo "[1] Configuring PulseAudio to allow direct ALSA access..."
PULSE_CONFIG="/etc/pulse/default.pa"

if [ -f "$PULSE_CONFIG" ]; then
    # Backup config
    if [ ! -f "$PULSE_CONFIG.bak" ]; then
        sudo cp "$PULSE_CONFIG" "$PULSE_CONFIG.bak"
        echo "  ✅ Backed up PulseAudio config"
    fi
    
    # Comment out module-alsa-source if not already commented
    if grep -q "^load-module module-alsa-source" "$PULSE_CONFIG"; then
        sudo sed -i 's/^load-module module-alsa-source/# Modified - Allow direct ALSA access\n# load-module module-alsa-source/' "$PULSE_CONFIG"
        echo "  ✅ Commented out module-alsa-source"
    else
        echo "  ✅ module-alsa-source already commented out"
    fi
    
    # Set suspend timeout to 0
    if grep -q "load-module module-suspend-on-idle" "$PULSE_CONFIG"; then
        sudo sed -i 's/load-module module-suspend-on-idle.*timeout=[0-9]*/load-module module-suspend-on-idle timeout=0/' "$PULSE_CONFIG" || \
        sudo sed -i 's/^load-module module-suspend-on-idle$/load-module module-suspend-on-idle timeout=0/' "$PULSE_CONFIG" || true
        echo "  ✅ Set suspend timeout to 0"
    fi
else
    echo "  ⚠️  PulseAudio config not found"
fi

# 2. Restart PulseAudio
echo ""
echo "[2] Restarting PulseAudio..."
if pgrep -x pulseaudio > /dev/null; then
    pulseaudio --kill 2>/dev/null || sudo killall pulseaudio 2>/dev/null || true
    sleep 2
    echo "  ✅ PulseAudio stopped"
fi

# Wait a moment before restarting
sleep 1

# PulseAudio will auto-start, but let's start it explicitly
pulseaudio --start 2>/dev/null || true
sleep 2
echo "  ✅ PulseAudio restarted"

# 3. Resume microphone source
echo ""
echo "[3] Resuming XVF3800 microphone source..."
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESUME_SCRIPT="$SCRIPT_DIR/resume_xvf3800_source.sh"

if [ -f "$RESUME_SCRIPT" ]; then
    bash "$RESUME_SCRIPT"
else
    echo "  ⚠️  Resume script not found, trying manual resume..."
    SOURCE_NAME=$(pactl list short sources 2>/dev/null | grep -i "XVF3800\|reSpeaker" | grep "input" | awk '{print $2}' | head -1)
    if [ -n "$SOURCE_NAME" ]; then
        pactl suspend-source "$SOURCE_NAME" 0 2>/dev/null && echo "  ✅ Source resumed" || echo "  ⚠️  Failed to resume source"
    else
        echo "  ⚠️  XVF3800 source not found"
    fi
fi

# 4. Check ALSA devices
echo ""
echo "[4] Checking ALSA devices..."
if command -v arecord >/dev/null 2>&1; then
    echo "  ALSA capture devices:"
    arecord -l 2>/dev/null | grep -i "XVF3800\|reSpeaker\|card" | head -5 || echo "    No XVF3800 found"
fi

# 5. Test with PulseAudio stopped (to check if it's blocking)
echo ""
echo "[5] Testing with PulseAudio stopped..."
pulseaudio --kill 2>/dev/null || true
sleep 2

python3 -c "
import sounddevice as sd
devices = sd.query_devices()
xvf_found = False
for i, dev in enumerate(devices):
    if 'XVF3800' in dev['name'] or 'reSpeaker' in dev['name']:
        print(f'  ✅ Found: Device {i}: {dev[\"name\"]} ({dev[\"max_input_channels\"]} in, {dev[\"max_output_channels\"]} out)')
        xvf_found = True
if not xvf_found:
    print('  ❌ XVF3800 not found in sounddevice devices')
    print('  Available input devices:')
    for i, dev in enumerate(devices):
        if dev['max_input_channels'] > 0:
            print(f'    Device {i}: {dev[\"name\"]}')
" 2>&1 || echo "  ⚠️  Failed to query devices"

# Restart PulseAudio
pulseaudio --start 2>/dev/null || true
sleep 1

echo ""
echo "=========================================="
echo "  Next Steps"
echo "=========================================="
echo ""
echo "If microphone still doesn't appear:"
echo "  1. Try stopping PulseAudio completely:"
echo "     pulseaudio --kill"
echo "     # Then test: python3 -c \"import sounddevice as sd; print(sd.query_devices())\""
echo ""
echo "  2. Check if device is visible to ALSA:"
echo "     arecord -l | grep -i xvf3800"
echo ""
echo "  3. Try unplugging and replugging the USB device"
echo ""
echo "  4. Check USB device:"
echo "     lsusb | grep -i seeed"
echo ""


#!/bin/bash
# Diagnose why microphone capture is not working after install_aura_bootable.sh

echo "=========================================="
echo "  Audio Capture Diagnostic"
echo "=========================================="
echo ""

# Check if PulseAudio is running
echo "[1] Checking PulseAudio status..."
if pgrep -x pulseaudio > /dev/null; then
    echo "  ✅ PulseAudio is running"
    echo "  ⚠️  PulseAudio may be blocking direct ALSA access"
    echo ""
    echo "  PulseAudio sources:"
    pactl list short sources 2>/dev/null | grep -i "XVF3800\|reSpeaker" || echo "    No XVF3800 sources found"
    echo ""
    echo "  PulseAudio source status:"
    pactl list short sources 2>/dev/null | grep -i "XVF3800\|reSpeaker" | awk '{print "    " $1 ": " $2 " - " $NF}' || echo "    No XVF3800 sources found"
else
    echo "  ❌ PulseAudio is NOT running"
fi
echo ""

# Check ALSA devices
echo "[2] Checking ALSA devices..."
if command -v arecord >/dev/null 2>&1; then
    echo "  ALSA capture devices:"
    arecord -l 2>/dev/null | grep -i "XVF3800\|reSpeaker\|card" || echo "    No XVF3800 devices found"
else
    echo "  ❌ arecord not found"
fi
echo ""

# Check if PulseAudio has exclusive control
echo "[3] Checking PulseAudio configuration..."
PULSE_CONFIG="/etc/pulse/default.pa"
if [ -f "$PULSE_CONFIG" ]; then
    echo "  Checking for 'load-module module-alsa-source' (may block direct ALSA access)..."
    if grep -q "load-module module-alsa-source" "$PULSE_CONFIG"; then
        echo "  ⚠️  Found: PulseAudio is loading ALSA source module"
        echo "     This may prevent direct ALSA access"
    else
        echo "  ✅ No ALSA source module found"
    fi
    
    echo "  Checking for 'load-module module-suspend-on-idle'..."
    if grep -q "load-module module-suspend-on-idle" "$PULSE_CONFIG"; then
        echo "  ⚠️  Found: PulseAudio suspends devices on idle"
        SUSPEND_TIMEOUT=$(grep "load-module module-suspend-on-idle" "$PULSE_CONFIG" | grep -o "timeout=[0-9]*" || echo "default")
        echo "     Timeout: $SUSPEND_TIMEOUT"
    else
        echo "  ✅ No suspend-on-idle module found"
    fi
else
    echo "  ⚠️  PulseAudio config not found at $PULSE_CONFIG"
fi
echo ""

# Check user groups
echo "[4] Checking user permissions..."
CURRENT_USER="${SUDO_USER:-$USER}"
echo "  Current user: $CURRENT_USER"
echo "  Groups: $(groups)"
if groups | grep -q audio; then
    echo "  ✅ User is in 'audio' group"
else
    echo "  ❌ User is NOT in 'audio' group"
    echo "     Run: sudo usermod -aG audio $CURRENT_USER"
    echo "     Then logout/login"
fi
echo ""

# Check for .asoundrc
echo "[5] Checking for .asoundrc..."
ASOUNDRC="$HOME/.asoundrc"
if [ -f "$ASOUNDRC" ]; then
    echo "  ⚠️  Found .asoundrc at $ASOUNDRC"
    echo "     This may interfere with microphone capture"
    echo "     Contents:"
    cat "$ASOUNDRC" | sed 's/^/     /'
else
    echo "  ✅ No .asoundrc found"
fi
echo ""

# Test direct ALSA access
echo "[6] Testing direct ALSA access..."
if command -v arecord >/dev/null 2>&1; then
    XVF_CARD=$(arecord -l 2>/dev/null | grep -i "XVF3800\|reSpeaker" | head -1 | sed -n 's/.*card \([0-9]*\):.*/\1/p')
    if [ -n "$XVF_CARD" ]; then
        echo "  Found XVF3800 at card $XVF_CARD"
        echo "  Testing direct ALSA capture (2 seconds)..."
        timeout 2 arecord -D "hw:$XVF_CARD,0" -f S16_LE -r 16000 -c 1 /tmp/test_alsa.wav 2>&1 | head -5
        if [ -f "/tmp/test_alsa.wav" ]; then
            SIZE=$(stat -c%s /tmp/test_alsa.wav 2>/dev/null || stat -f%z /tmp/test_alsa.wav 2>/dev/null)
            if [ "$SIZE" -gt 0 ]; then
                echo "  ✅ Direct ALSA capture works (file size: $SIZE bytes)"
            else
                echo "  ❌ Direct ALSA capture failed (empty file)"
            fi
            rm -f /tmp/test_alsa.wav
        else
            echo "  ❌ Direct ALSA capture failed (no file created)"
        fi
    else
        echo "  ❌ XVF3800 device not found in ALSA"
    fi
else
    echo "  ❌ arecord not available"
fi
echo ""

# Recommendations
echo "=========================================="
echo "  Recommendations"
echo "=========================================="
echo ""
if pgrep -x pulseaudio > /dev/null; then
    echo "⚠️  PulseAudio is running and may be blocking direct ALSA access"
    echo ""
    echo "Option 1: Stop PulseAudio temporarily (for testing):"
    echo "  pulseaudio --kill"
    echo "  # Then test microphone capture"
    echo "  # Restart PulseAudio: pulseaudio --start"
    echo ""
    echo "Option 2: Configure PulseAudio to not block ALSA:"
    echo "  Edit /etc/pulse/default.pa and comment out:"
    echo "  # load-module module-alsa-source"
    echo "  # Then restart: pulseaudio --kill && pulseaudio --start"
    echo ""
    echo "Option 3: Use PulseAudio for capture (adds latency):"
    echo "  Configure sounddevice to use PulseAudio backend"
    echo ""
fi

if [ -f "$ASOUNDRC" ]; then
    echo "⚠️  .asoundrc exists and may interfere"
    echo "  Consider removing it: rm $ASOUNDRC"
    echo ""
fi

echo "To test microphone after fixes:"
echo "  python3 -c \"import sounddevice as sd; print(sd.query_devices())\""
echo "  # Look for XVF3800 device"
echo ""


#!/bin/bash
# Test script to check and set ALSA default audio device
# Usage: ./test_alsa_default.sh

echo "=========================================="
echo "  ALSA Default Audio Device Test"
echo "=========================================="
echo ""

# Check current .asoundrc
ASOUNDRC="$HOME/.asoundrc"
if [ -f "$ASOUNDRC" ]; then
    echo "📄 Current .asoundrc:"
    cat "$ASOUNDRC"
    echo ""
else
    echo "⚠️  No .asoundrc file found"
    echo ""
fi

# List available audio devices
echo "🔍 Available ALSA devices:"
aplay -l 2>/dev/null | grep "^card" || echo "No devices found"
echo ""

# Check for UACDemo device (both variants)
if aplay -l 2>/dev/null | grep -qE "UACDemoV1\.0|UACDemoV10"; then
    CARD_NUM=$(aplay -l 2>/dev/null | grep -E "UACDemoV1\.0|UACDemoV10" | sed -n 's/.*card \([0-9]*\):.*/\1/p' | head -1)
    DEVICE_NAME=$(aplay -l 2>/dev/null | grep -E "UACDemoV1\.0|UACDemoV10" | sed -n 's/.*card [0-9]*: \([^,]*\).*/\1/p' | head -1)
    
    echo "✅ Found UACDemo device:"
    echo "   Card: $CARD_NUM"
    echo "   Name: $DEVICE_NAME"
    echo ""
    
    # Check if it's set as default
    if [ -f "$ASOUNDRC" ] && grep -q "defaults.pcm.card.*$CARD_NUM" "$ASOUNDRC" 2>/dev/null; then
        echo "✅ Device is set as default in .asoundrc"
    else
        echo "⚠️  Device is NOT set as default"
        echo ""
        echo "To set it as default, run:"
        echo "  bash setup/scripts/set_default_audio_on_boot.sh"
        echo ""
        echo "Or manually create ~/.asoundrc:"
        echo "  defaults.pcm.card $CARD_NUM"
        echo "  defaults.ctl.card $CARD_NUM"
    fi
    echo ""
    
    # Test playback with default device
    echo "🎵 Testing playback with default device:"
    echo "   aplay -D default test.wav"
    echo ""
    echo "   (This should use card $CARD_NUM if .asoundrc is correct)"
    echo ""
    
    # Test playback with explicit device
    echo "🎵 Testing playback with explicit device:"
    echo "   aplay -D plughw:$CARD_NUM,0 test.wav"
    echo ""
    echo "   (This should always work - same as speaker.py uses)"
else
    echo "⚠️  UACDemo device not found"
    echo "   Make sure device is connected"
fi

echo "=========================================="
echo "  Current ALSA Default"
echo "=========================================="
echo ""

# Check what ALSA thinks is default
if [ -f "$ASOUNDRC" ]; then
    DEFAULT_CARD=$(grep "defaults.pcm.card" "$ASOUNDRC" | awk '{print $2}')
    if [ -n "$DEFAULT_CARD" ]; then
        echo "Default card from .asoundrc: $DEFAULT_CARD"
        DEVICE_INFO=$(aplay -l 2>/dev/null | grep "card $DEFAULT_CARD:" | head -1)
        if [ -n "$DEVICE_INFO" ]; then
            echo "Device: $DEVICE_INFO"
        else
            echo "⚠️  Card $DEFAULT_CARD not found in aplay -l"
        fi
    fi
else
    echo "No .asoundrc - ALSA will use system default (usually card 0)"
fi

echo ""
echo "=========================================="
echo "  Quick Fix"
echo "=========================================="
echo ""
echo "To set UACDemo as default, run:"
echo "  bash setup/scripts/set_default_audio_on_boot.sh"
echo ""
echo "Or manually:"
echo "  echo 'defaults.pcm.card 0' > ~/.asoundrc"
echo "  echo 'defaults.ctl.card 0' >> ~/.asoundrc"
echo ""
echo "(Replace 0 with your actual card number if different)"

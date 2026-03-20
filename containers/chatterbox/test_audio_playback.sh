#!/bin/bash
# Test audio playback with the same device selection as speaker.py

echo "=========================================="
echo "  Audio Playback Test"
echo "=========================================="
echo ""

# Find the audio device (same logic as speaker.py)
echo "🔍 Detecting audio output device..."
DEVICE_NAME=""
CARD_INDEX=""

# Try to find UACDemoV1.0 first
if aplay -l 2>/dev/null | grep -q "UACDemoV1.0"; then
    CARD_INDEX=$(aplay -l 2>/dev/null | grep "UACDemoV1.0" | head -1 | sed -n 's/.*card \([0-9]*\):.*/\1/p')
    DEVICE_NAME="UACDemoV1.0"
    echo "✅ Found: $DEVICE_NAME (card $CARD_INDEX)"
# Fallback: find any USB audio device
elif aplay -l 2>/dev/null | grep -q "USB Audio"; then
    CARD_INDEX=$(aplay -l 2>/dev/null | grep "USB Audio" | grep -E "0 in|out" | head -1 | sed -n 's/.*card \([0-9]*\):.*/\1/p')
    DEVICE_NAME=$(aplay -l 2>/dev/null | grep "USB Audio" | grep -E "0 in|out" | head -1 | sed -n 's/.*card [0-9]*: \([^,]*\).*/\1/p')
    echo "✅ Found: $DEVICE_NAME (card $CARD_INDEX)"
else
    echo "⚠️  No USB audio device found, using default"
    CARD_INDEX=""
    DEVICE_NAME="default"
fi

echo ""
echo "📋 Available ALSA devices:"
aplay -l 2>/dev/null | grep "^card"

echo ""
echo "=========================================="
echo "  Testing Playback"
echo "=========================================="
echo ""

# Find WAV files to test
WAV_FILES=(
    "test_output_basic_synthesis.wav"
    "test_output_voice_cloning_synthesis.wav"
    "../assets/voice_samples/sample.wav"
)

for wav_file in "${WAV_FILES[@]}"; do
    if [ ! -f "$wav_file" ]; then
        continue
    fi
    
    echo "🎵 Testing: $wav_file"
    
    # Get file info
    if command -v soxi &> /dev/null; then
        echo "   Format: $(soxi -t "$wav_file" 2>/dev/null || echo 'unknown')"
        echo "   Sample rate: $(soxi -r "$wav_file" 2>/dev/null || echo 'unknown') Hz"
        echo "   Channels: $(soxi -c "$wav_file" 2>/dev/null || echo 'unknown')"
    fi
    
    # Test with detected device (same as speaker.py)
    if [ -n "$CARD_INDEX" ]; then
        echo "   Using device: plughw:$CARD_INDEX,0 (same as speaker.py)"
        if aplay -D "plughw:$CARD_INDEX,0" "$wav_file" 2>&1; then
            echo "   ✅ Playback completed"
        else
            echo "   ❌ Playback failed"
        fi
    else
        echo "   Using default device"
        if aplay "$wav_file" 2>&1; then
            echo "   ✅ Playback completed"
        else
            echo "   ❌ Playback failed"
        fi
    fi
    
    echo ""
done

echo "=========================================="
echo "  Device Information"
echo "=========================================="
echo ""

if [ -n "$CARD_INDEX" ]; then
    echo "Detected device: $DEVICE_NAME (card $CARD_INDEX)"
    echo ""
    echo "To test manually, use:"
    echo "  aplay -D plughw:$CARD_INDEX,0 your_file.wav"
    echo ""
    echo "Or test with raw PCM (like speaker.py does):"
    echo "  echo 'test' | aplay -D plughw:$CARD_INDEX,0 -f S16_LE -r 22050 -c 1"
else
    echo "No specific device detected - using default"
    echo ""
    echo "To test manually, use:"
    echo "  aplay your_file.wav"
fi

echo ""
echo "=========================================="
echo "  Troubleshooting"
echo "=========================================="
echo ""

# Check if device is busy
if [ -n "$CARD_INDEX" ]; then
    echo "Checking if device is in use..."
    if lsof 2>/dev/null | grep -q "snd"; then
        echo "⚠️  Audio device may be in use:"
        lsof 2>/dev/null | grep "snd" | head -5
    else
        echo "✅ Device appears to be free"
    fi
fi

# Check permissions
echo ""
echo "Checking permissions..."
if [ -r /dev/snd/controlC* ] 2>/dev/null || [ -r /dev/snd/pcmC*D*p ] 2>/dev/null; then
    echo "✅ Have read access to audio devices"
else
    echo "⚠️  May not have access to audio devices"
    echo "   Try: sudo usermod -a -G audio $USER"
    echo "   Then logout and login again"
fi

# Check PulseAudio status
echo ""
if command -v pulseaudio &> /dev/null; then
    if pulseaudio --check 2>/dev/null; then
        echo "ℹ️  PulseAudio is running (this is OK - speaker.py uses ALSA directly)"
    else
        echo "ℹ️  PulseAudio is not running (this is OK - speaker.py uses ALSA directly)"
    fi
fi

echo ""
echo "=========================================="
echo "  Summary"
echo "=========================================="
echo ""
echo "If playback completed but you heard no sound:"
echo "  1. Check volume: amixer -c $CARD_INDEX sget PCM"
echo "  2. Check device is not muted: amixer -c $CARD_INDEX sset PCM unmute"
echo "  3. Try different device: aplay -l to list all devices"
echo "  4. Check physical connections (speakers/headphones)"
echo ""
echo "The PulseAudio error you saw is harmless - it's from a library"
echo "trying to use PulseAudio, but speaker.py uses ALSA directly."

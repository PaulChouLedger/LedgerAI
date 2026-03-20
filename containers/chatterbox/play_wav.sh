#!/bin/bash
# play_wav.sh - Play WAV file using same device selection as speaker.py
# Usage: ./play_wav.sh file.wav

if [ $# -eq 0 ]; then
    echo "Usage: $0 <wav_file>"
    echo "Example: $0 test_output_basic_synthesis.wav"
    exit 1
fi

WAV_FILE="$1"

if [ ! -f "$WAV_FILE" ]; then
    echo "❌ File not found: $WAV_FILE"
    exit 1
fi

# Auto-detect device (same logic as speaker.py)
CARD_INDEX=""
DEVICE_NAME=""

# Try to find UACDemoV1.0 first
if aplay -l 2>/dev/null | grep -q "UACDemoV1.0"; then
    CARD_INDEX=$(aplay -l 2>/dev/null | grep "UACDemoV1.0" | head -1 | sed -n 's/.*card \([0-9]*\):.*/\1/p')
    DEVICE_NAME="UACDemoV1.0"
# Fallback: find any USB audio device
elif aplay -l 2>/dev/null | grep -q "USB Audio"; then
    CARD_INDEX=$(aplay -l 2>/dev/null | grep "USB Audio" | grep -E "0 in|out" | head -1 | sed -n 's/.*card \([0-9]*\):.*/\1/p')
    DEVICE_NAME=$(aplay -l 2>/dev/null | grep "USB Audio" | grep -E "0 in|out" | head -1 | sed -n 's/.*card [0-9]*: \([^,]*\).*/\1/p')
fi

if [ -n "$CARD_INDEX" ]; then
    echo "🎵 Playing: $WAV_FILE"
    echo "🔊 Using device: $DEVICE_NAME (card $CARD_INDEX) - same as speaker.py"
    aplay -D "plughw:$CARD_INDEX,0" "$WAV_FILE"
else
    echo "⚠️  No USB audio device detected, using default"
    echo "🎵 Playing: $WAV_FILE"
    aplay "$WAV_FILE"
fi

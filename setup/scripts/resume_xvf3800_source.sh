#!/bin/bash
# Resume/activate XVF3800 microphone source in PulseAudio
# This fixes the SUSPENDED state that prevents microphone capture

# Find XVF3800 source
SOURCE_NAME=$(pactl list short sources 2>/dev/null | grep -i "XVF3800\|reSpeaker" | grep "input" | awk '{print $2}' | head -1)

if [ -z "$SOURCE_NAME" ]; then
    echo "[Audio] ❌ XVF3800 source not found in PulseAudio"
    exit 1
fi

echo "[Audio] Found XVF3800 source: $SOURCE_NAME"

# Check current state
CURRENT_STATE=$(pactl list short sources 2>/dev/null | grep "$SOURCE_NAME" | awk '{print $NF}')

if [ "$CURRENT_STATE" = "SUSPENDED" ]; then
    echo "[Audio] Source is SUSPENDED - resuming..."
    
    # Resume the source
    if pactl suspend-source "$SOURCE_NAME" 0 2>/dev/null; then
        echo "[Audio] ✅ Source resumed successfully"
    else
        echo "[Audio] ❌ Failed to resume source"
        exit 1
    fi
else
    echo "[Audio] Source state: $CURRENT_STATE"
    echo "[Audio] ✅ Source is already active"
fi

# Verify it's active
sleep 0.5
NEW_STATE=$(pactl list short sources 2>/dev/null | grep "$SOURCE_NAME" | awk '{print $NF}')
if [ "$NEW_STATE" != "SUSPENDED" ]; then
    echo "[Audio] ✅ Microphone is now active and ready for capture"
    exit 0
else
    echo "[Audio] ⚠️  Source is still suspended - PulseAudio may need restart"
    echo "[Audio] Try: pulseaudio --kill && pulseaudio --start"
    exit 1
fi


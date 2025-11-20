#!/bin/bash
# Set default audio output to UACDemoV1.0 on boot
# This script is called by the aura.service systemd unit
# It configures both ALSA and PulseAudio to use UACDemoV1.0 as default

# Get user home directory (script runs as the service user)
AURA_HOME="${HOME:-/home/ledger}"

# Wait a bit for audio devices to be ready (they may not be immediately available on boot)
sleep 2

# Check if UACDemoV1.0 audio device exists
# Redirect stderr and handle failures gracefully
if aplay -l 2>/dev/null | grep -q "UACDemoV1.0" 2>/dev/null; then
    CARD_NUM=$(aplay -l 2>/dev/null | grep "UACDemoV1.0" | sed -n 's/.*card \([0-9]*\):.*/\1/p' | head -1)
    
    if [ -n "$CARD_NUM" ]; then
        # Create/update .asoundrc for ALSA
        cat > "$AURA_HOME/.asoundrc" << EOF
pcm.!default {
    type hw
    card $CARD_NUM
    device 0
}

ctl.!default {
    type hw
    card $CARD_NUM
}
EOF
        # Ensure proper ownership
        chmod 644 "$AURA_HOME/.asoundrc" 2>/dev/null || true
        
        # Set PulseAudio default sink (if available) - uses dynamic name matching
        if command -v pactl >/dev/null 2>&1; then
            # Wait for PulseAudio to be ready
            for i in {1..5}; do
                if pactl list short sinks 2>/dev/null | grep -q .; then
                    break
                fi
                sleep 1
            done
            
            SINK_NAME=$(pactl list short sinks 2>/dev/null | grep -i "UACDemoV1.0\|UACDemo" | cut -f2 | head -1)
            if [ -n "$SINK_NAME" ]; then
                pactl set-default-sink "$SINK_NAME" 2>/dev/null || true
            fi
        fi
    fi
else
    # Device not found - this is OK, it might not be plugged in
    # Don't fail the service, just exit silently
    exit 0
fi

exit 0


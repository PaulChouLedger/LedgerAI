#!/bin/bash
# Set default audio output to UACDemoV1.0 on boot
# This script is called by the aura.service systemd unit
# It configures both ALSA and PipeWire to use UACDemoV1.0 as default
# Uses PipeWire native commands (wpctl) - no PulseAudio compatibility layer

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
        
        # Set PipeWire default sink (using wpctl - PipeWire native command)
        if command -v wpctl >/dev/null 2>&1; then
            # Use PipeWire native wpctl command
            # Wait for PipeWire to be ready
            for i in {1..10}; do
                if wpctl status 2>/dev/null | grep -q .; then
                    break
                fi
                sleep 1
            done
            
            # Find sink using wpctl
            # wpctl status shows sinks with format: " *   42. Sink Name" or "      42. Sink Name"
            SINK_LINE=$(wpctl status 2>/dev/null | grep -i "Jieli_Technology_UACDemoV1.0\|UACDemoV1.0\|UACDemo" | grep -i "sink" | head -1)
            SINK_ID=$(echo "$SINK_LINE" | sed -n 's/.*[^0-9]\([0-9][0-9]*\)\. .*/\1/p')
            
            if [ -n "$SINK_ID" ]; then
                echo "[Audio] Setting PipeWire default sink to ID: $SINK_ID" >&2
                # wpctl set-default sets default sink by ID
                if wpctl set-default "$SINK_ID" 2>/dev/null; then
                    echo "[Audio] ✅ Default sink set successfully (using wpctl)" >&2
                else
                    echo "[Audio] ⚠️  Failed to set default sink with wpctl" >&2
                    echo "[Audio] 💡 Try manually: wpctl set-default $SINK_ID" >&2
                fi
            else
                echo "[Audio] ⚠️  UACDemoV1.0 sink not found with wpctl" >&2
                echo "[Audio] Available sinks:" >&2
                wpctl status 2>/dev/null | grep -i "sink" | head -5 >&2 || true
            fi
        else
            echo "[Audio] ⚠️  wpctl not found - PipeWire may not be installed" >&2
        fi
    fi
else
    # Device not found - this is OK, it might not be plugged in
    # Don't fail the service, just exit silently
    exit 0
fi

exit 0


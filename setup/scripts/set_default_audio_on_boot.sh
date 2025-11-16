#!/bin/bash
# Set default audio output on boot
# This script is called by aura.service ExecStartPre to ensure audio output persists

# Get user home directory from environment or use default
USER_HOME="${HOME:-/home/${SUDO_USER:-$USER}}"

# Set default ALSA device to UACDemoV1.0 (if present)
if aplay -l 2>/dev/null | grep -q "UACDemoV1.0"; then
    CARD_NUM=$(aplay -l 2>/dev/null | grep "UACDemoV1.0" | sed -n 's/.*card \([0-9]*\):.*/\1/p' | head -1)
    if [ -n "$CARD_NUM" ]; then
        # Create/update .asoundrc for ALSA
        cat > "$USER_HOME/.asoundrc" << EOF
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
        chmod 644 "$USER_HOME/.asoundrc" 2>/dev/null || true
        echo "[Audio] ✅ ALSA default set to card $CARD_NUM (UACDemoV1.0)"
        
        # Set PulseAudio default sink (if available)
        if command -v pactl >/dev/null 2>&1; then
            # Wait a moment for PulseAudio to be ready
            sleep 1
            
            # Find sink name dynamically
            SINK_NAME=$(pactl list short sinks 2>/dev/null | grep -i "UACDemoV1.0\|UACDemo" | cut -f2 | head -1)
            if [ -n "$SINK_NAME" ]; then
                # Set XDG_RUNTIME_DIR for PulseAudio
                export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/run/user/$(id -u)}"
                
                # Try to set default sink
                if pactl set-default-sink "$SINK_NAME" 2>/dev/null; then
                    echo "[Audio] ✅ PulseAudio default sink set to: $SINK_NAME"
                else
                    echo "[Audio] ⚠️ Could not set PulseAudio default sink (may need user session)"
                fi
            else
                echo "[Audio] ⚠️ UACDemoV1.0 PulseAudio sink not found (may appear later)"
            fi
        fi
    fi
else
    echo "[Audio] ⚠️ UACDemoV1.0 device not found - using system default"
fi


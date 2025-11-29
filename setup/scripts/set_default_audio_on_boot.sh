#!/bin/bash
# Set default audio output to UACDemoV1.0 on boot (ALSA only)
# This script is called by the aura.service systemd unit
# Sets ALSA default output card via .asoundrc (output only, won't affect microphone)

# Get user home directory (script runs as the service user)
AURA_HOME="${HOME:-/home/ledger}"

# Wait a bit for audio devices to be ready (they may not be immediately available on boot)
sleep 2

# Check if UACDemoV1.0 audio device exists
# Redirect stderr and handle failures gracefully
if aplay -l 2>/dev/null | grep -q "UACDemoV1.0" 2>/dev/null; then
    CARD_NUM=$(aplay -l 2>/dev/null | grep "UACDemoV1.0" | sed -n 's/.*card \([0-9]*\):.*/\1/p' | head -1)
    
    if [ -n "$CARD_NUM" ]; then
        # Set ALSA default output card (output only, not input - won't interfere with microphone)
        # This helps other applications use the correct output device
        # speaker.py auto-detects by device name, but system default is still useful
        ASOUNDRC="$AURA_HOME/.asoundrc"
        if [ ! -f "$ASOUNDRC" ] || ! grep -q "defaults.pcm.card.*$CARD_NUM" "$ASOUNDRC" 2>/dev/null; then
            # Create minimal .asoundrc with output-only default (doesn't affect input/microphone)
            cat > "$ASOUNDRC" << EOF
# ALSA default output card (output only - does not affect microphone input)
# Set by set_default_audio_on_boot.sh
# This only sets the default OUTPUT card, not input, so it won't interfere with microphone capture
defaults.pcm.card $CARD_NUM
defaults.ctl.card $CARD_NUM
EOF
            echo "[Audio] ✅ Set ALSA default output card to $CARD_NUM (UACDemoV1.0)" >&2
            echo "[Audio]    Created minimal .asoundrc (output only, won't affect microphone)" >&2
        else
            echo "[Audio] ✅ ALSA default output card already set to $CARD_NUM" >&2
        fi
    fi
else
    # Device not found - this is OK, it might not be plugged in
    # Don't fail the service, just exit silently
    exit 0
fi

exit 0


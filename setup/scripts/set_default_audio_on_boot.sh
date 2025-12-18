#!/bin/bash
# Set default audio output to UACDemoV1.0 on boot (ALSA and PulseAudio)
# This script is called by the aura.service systemd unit
# Sets ALSA default output card via .asoundrc and PulseAudio default sink
# (output only, won't affect microphone)

# Get user home directory (script runs as the service user)
AURA_HOME="${HOME:-/home/ledger}"

# Wait a bit for audio devices to be ready (they may not be immediately available on boot)
sleep 2

# Check if UACDemoV1.0 or UACDemoV10 audio device exists
# Handle both variants: "UACDemoV1.0" and "UACDemoV10" (with/without dot)
# Redirect stderr and handle failures gracefully
if aplay -l 2>/dev/null | grep -qE "UACDemoV1\.0|UACDemoV10" 2>/dev/null; then
    # Try UACDemoV1.0 first, then UACDemoV10 (without dot)
    CARD_NUM=$(aplay -l 2>/dev/null | grep -E "UACDemoV1\.0|UACDemoV10" | sed -n 's/.*card \([0-9]*\):.*/\1/p' | head -1)
    
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
            DEVICE_NAME=$(aplay -l 2>/dev/null | grep -E "UACDemoV1\.0|UACDemoV10" | sed -n 's/.*card [0-9]*: \([^,]*\).*/\1/p' | head -1)
            echo "[Audio] ✅ Set ALSA default output card to $CARD_NUM ($DEVICE_NAME)" >&2
            echo "[Audio]    Created minimal .asoundrc (output only, won't affect microphone)" >&2
        else
            echo "[Audio] ✅ ALSA default output card already set to $CARD_NUM" >&2
        fi
        
        # Set PulseAudio default sink to UACDemoV1.0 (if PulseAudio is available)
        if command -v pactl >/dev/null 2>&1; then
            # Find the PulseAudio sink for UACDemoV1.0 or UACDemoV10
            # Use awk to properly parse sink blocks and match by description or card number
            SINK_NAME=$(pactl list sinks 2>/dev/null | awk -v card="$CARD_NUM" '
                BEGIN { in_sink=0; sink_name=""; sink_desc=""; sink_card=""; match_found=0 }
                /^Sink #/ { 
                    # Process previous sink if we found a match
                    if (in_sink && sink_name != "" && match_found) {
                        print sink_name
                        exit
                    }
                    # Start new sink block
                    in_sink=1
                    sink_name=""
                    sink_desc=""
                    sink_card=""
                    match_found=0
                }
                /^[[:space:]]*Name:[[:space:]]*/ && in_sink { 
                    sink_name=$2
                }
                /^[[:space:]]*Description:[[:space:]]*/ && in_sink { 
                    # Get everything after "Description: "
                    sink_desc=substr($0, index($0, "Description:") + 13)
                }
                /^[[:space:]]*alsa.card =/ && in_sink { 
                    gsub(/"/, "", $3)
                    sink_card=$3
                }
                # Check if this sink matches our criteria (after we have the name)
                in_sink && sink_name != "" {
                    # Match if description contains UACDemo (but not XVF3800) OR card number matches
                    if ((sink_desc ~ /UACDemo/ && sink_desc !~ /XVF3800/) || 
                        (sink_card == card && sink_desc !~ /XVF3800/)) {
                        match_found=1
                    }
                }
                /^$/ && in_sink {
                    # End of sink block - check if we found a match
                    if (match_found && sink_name != "") {
                        print sink_name
                        exit
                    }
                    in_sink=0
                }
                END {
                    # Process last sink if needed
                    if (in_sink && match_found && sink_name != "") {
                        print sink_name
                    }
                }
            ')
            
            # Fallback: Simple method if awk didn't work
            if [ -z "$SINK_NAME" ]; then
                # Look for sink name that contains UACDemoV1.0 in the name itself
                SINK_NAME=$(pactl list sinks 2>/dev/null | awk '
                    /^[[:space:]]*Name:/ { sink_name=$2 }
                    /UACDemo/ && !/XVF3800/ && sink_name != "" { print sink_name; exit }
                ')
            fi
            
            if [ -n "$SINK_NAME" ]; then
                # Set as default sink
                if pactl set-default-sink "$SINK_NAME" 2>/dev/null; then
                    DEVICE_NAME=$(aplay -l 2>/dev/null | grep -E "UACDemoV1\.0|UACDemoV10" | sed -n 's/.*card [0-9]*: \([^,]*\).*/\1/p' | head -1)
                    echo "[Audio] ✅ Set PulseAudio default sink to $SINK_NAME ($DEVICE_NAME)" >&2
                else
                    echo "[Audio] ⚠️  Failed to set PulseAudio default sink (may need PulseAudio restart)" >&2
                fi
            else
                echo "[Audio] ⚠️  Could not find PulseAudio sink for UACDemoV1.0" >&2
                echo "[Audio]    PulseAudio may not be running or device not yet available" >&2
            fi
        else
            echo "[Audio] ℹ️  PulseAudio (pactl) not available - skipping PulseAudio default sink setup" >&2
        fi
    fi
else
    # Device not found - this is OK, it might not be plugged in
    # Don't fail the service, just exit silently
    exit 0
fi

exit 0

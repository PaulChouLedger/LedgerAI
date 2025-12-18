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
            # Use the simpler "short" format which is more reliable
            SINK_NAME=$(pactl list sinks short 2>/dev/null | awk '
                $2 ~ /UACDemo/ && $2 !~ /XVF3800/ { 
                    print $2
                    exit
                }
            ' | head -1 | tr -d '\n\r' | xargs)
            
            # Fallback: Parse full sink list if short format didn't work
            if [ -z "$SINK_NAME" ]; then
                SINK_NAME=$(pactl list sinks 2>/dev/null | awk -v card="$CARD_NUM" '
                    BEGIN { in_sink=0; sink_name=""; sink_desc=""; sink_card="" }
                    /^Sink #/ { 
                        # Start new sink block
                        in_sink=1
                        sink_name=""
                        sink_desc=""
                        sink_card=""
                    }
                    /^[[:space:]]*Name:[[:space:]]*/ && in_sink { 
                        sink_name=$2
                    }
                    /^[[:space:]]*Description:[[:space:]]*/ && in_sink { 
                        sink_desc=substr($0, index($0, "Description:") + 13)
                    }
                    /^[[:space:]]*alsa.card =/ && in_sink { 
                        gsub(/"/, "", $3)
                        sink_card=$3
                    }
                    /^$/ && in_sink {
                        # End of sink block - check if this matches
                        if (sink_name != "" && 
                            ((sink_desc ~ /UACDemo/ && sink_desc !~ /XVF3800/) || 
                             (sink_card == card && sink_desc !~ /XVF3800/))) {
                            print sink_name
                            exit
                        }
                        in_sink=0
                    }
                ' | head -1 | tr -d '\n\r' | xargs)
            fi
            
            # Clean up sink name (remove any extra whitespace/newlines)
            SINK_NAME=$(echo -n "$SINK_NAME" | tr -d '\n\r' | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')
            
            if [ -n "$SINK_NAME" ]; then
                # Debug: Show sink name (with quotes to see any hidden characters)
                echo "[Audio] 🔍 Found PulseAudio sink: '$SINK_NAME'" >&2
                
                # Check if PulseAudio is running
                if ! pactl info >/dev/null 2>&1; then
                    echo "[Audio] ⚠️  PulseAudio is not running - cannot set default sink" >&2
                    echo "[Audio]    Try: pulseaudio --start" >&2
                else
                    # Check if sink is suspended and resume it first
                    SINK_STATE=$(pactl list sinks 2>/dev/null | grep -A 5 "^[[:space:]]*Name: $SINK_NAME" | grep "^[[:space:]]*State:" | awk '{print $2}')
                    if [ "$SINK_STATE" = "SUSPENDED" ]; then
                        echo "[Audio] 🔄 Resuming suspended sink..." >&2
                        pactl suspend-sink "$SINK_NAME" 0 2>/dev/null || true
                        sleep 0.5  # Give it a moment to resume
                    fi
                    
                    # Verify sink exists and get its index
                    SINK_INDEX=$(pactl list sinks short 2>/dev/null | grep "[[:space:]]$SINK_NAME$" | awk '{print $1}')
                    
                    if [ -z "$SINK_INDEX" ]; then
                        echo "[Audio] ⚠️  Sink '$SINK_NAME' not found in PulseAudio" >&2
                        echo "[Audio]    Available sinks:" >&2
                        pactl list sinks short 2>/dev/null | head -5 | sed 's/^/[Audio]      /' >&2
                    else
                        echo "[Audio] 🔍 Sink found at index: $SINK_INDEX" >&2
                        
                        # Try using sink index instead of name (more reliable)
                        if pactl set-default-sink "$SINK_INDEX" 2>/dev/null; then
                            CURRENT_DEFAULT=$(pactl info 2>/dev/null | grep "Default Sink:" | sed 's/Default Sink: //' | tr -d '\n\r')
                            DEVICE_NAME=$(aplay -l 2>/dev/null | grep -E "UACDemoV1\.0|UACDemoV10" | sed -n 's/.*card [0-9]*: \([^,]*\).*/\1/p' | head -1)
                            echo "[Audio] ✅ Set PulseAudio default sink to $SINK_NAME ($DEVICE_NAME)" >&2
                        else
                            # Fallback: Try with sink name directly
                            ERROR_OUTPUT=$(pactl set-default-sink "$SINK_NAME" 2>&1)
                            EXIT_CODE=$?
                            
                            if [ $EXIT_CODE -eq 0 ]; then
                                CURRENT_DEFAULT=$(pactl info 2>/dev/null | grep "Default Sink:" | sed 's/Default Sink: //' | tr -d '\n\r')
                                DEVICE_NAME=$(aplay -l 2>/dev/null | grep -E "UACDemoV1\.0|UACDemoV10" | sed -n 's/.*card [0-9]*: \([^,]*\).*/\1/p' | head -1)
                                echo "[Audio] ✅ Set PulseAudio default sink to $SINK_NAME ($DEVICE_NAME)" >&2
                            else
                                echo "[Audio] ⚠️  Failed to set PulseAudio default sink" >&2
                                if [ -n "$ERROR_OUTPUT" ]; then
                                    echo "[Audio]    Error: $ERROR_OUTPUT" >&2
                                fi
                                echo "[Audio]    Note: This is optional - ALSA default is already set" >&2
                                echo "[Audio]    ALSA playback (aplay, speaker.py) will work regardless" >&2
                            fi
                        fi
                    fi
                fi
            else
                echo "[Audio] ⚠️  Could not find PulseAudio sink for UACDemoV1.0" >&2
                echo "[Audio]    PulseAudio may not be running or device not yet available" >&2
                echo "[Audio]    Debug: Run 'pactl list sinks | grep -A 5 UACDemo' to see available sinks" >&2
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

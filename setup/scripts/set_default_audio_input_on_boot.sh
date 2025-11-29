#!/bin/bash
# Set default audio input to ReSpeaker XVF3800 on boot
# This script is called by the aura.service systemd unit
# It configures PulseAudio to use ReSpeaker XVF3800 as default input source

# Get user home directory (script runs as the service user)
AURA_HOME="${HOME:-/home/ledger}"

# Wait a bit for audio devices to be ready (they may not be immediately available on boot)
sleep 2

# Check if ReSpeaker XVF3800 audio device exists
# Redirect stderr and handle failures gracefully
if arecord -l 2>/dev/null | grep -q "XVF3800\|reSpeaker\|ArrayUAC10" 2>/dev/null; then
    CARD_NUM=$(arecord -l 2>/dev/null | grep -i "XVF3800\|reSpeaker\|ArrayUAC10" | sed -n 's/.*card \([0-9]*\):.*/\1/p' | head -1)
    
    if [ -n "$CARD_NUM" ]; then
        # Set PulseAudio default source (if available) - uses dynamic name matching
        if command -v pactl >/dev/null 2>&1; then
            # Wait longer for PulseAudio to be ready (USB devices can take time)
            for i in {1..10}; do
                if pactl list short sources 2>/dev/null | grep -q .; then
                    break
                fi
                sleep 1
            done
            
            # Try multiple methods to find the source
            # Priority: Match constant pattern "Seeed_Studio_reSpeaker_XVF3800_4-Mic_Array" (works across all device IDs)
            # Method 1: Look for "Seeed_Studio_reSpeaker_XVF3800_4-Mic_Array" pattern (constant part, ignores device ID)
            SOURCE_NAME=$(pactl list short sources 2>/dev/null | grep -i "alsa_input.*Seeed_Studio_reSpeaker_XVF3800_4-Mic_Array" | awk '{print $2}' | head -1)
            
            # Method 2: Look for "Seeed.*Studio.*reSpeaker.*XVF3800" (more flexible pattern matching)
            if [ -z "$SOURCE_NAME" ]; then
                SOURCE_NAME=$(pactl list short sources 2>/dev/null | grep -iE "alsa_input.*Seeed.*Studio.*reSpeaker.*XVF3800|alsa_input.*Seeed_Studio_reSpeaker_XVF3800" | awk '{print $2}' | head -1)
            fi
            
            # Method 3: Look for XVF3800 in input source name (fallback if Seeed pattern doesn't match)
            if [ -z "$SOURCE_NAME" ]; then
                SOURCE_NAME=$(pactl list short sources 2>/dev/null | grep -i "alsa_input.*XVF3800" | awk '{print $2}' | head -1)
            fi
            
            # Method 4: Look for reSpeaker in input source name
            if [ -z "$SOURCE_NAME" ]; then
                SOURCE_NAME=$(pactl list short sources 2>/dev/null | grep -i "alsa_input.*reSpeaker" | awk '{print $2}' | head -1)
            fi
            
            # Method 5: Look for ArrayUAC10 in input source name (alternative name)
            if [ -z "$SOURCE_NAME" ]; then
                SOURCE_NAME=$(pactl list short sources 2>/dev/null | grep -i "alsa_input.*ArrayUAC10" | awk '{print $2}' | head -1)
            fi
            
            # Method 6: Look for USB input devices with "analog-stereo" (last resort fallback)
            if [ -z "$SOURCE_NAME" ]; then
                SOURCE_NAME=$(pactl list short sources 2>/dev/null | grep -i "alsa_input.*usb.*analog-stereo" | awk '{print $2}' | head -1)
            fi
            
            if [ -n "$SOURCE_NAME" ]; then
                echo "[Audio] Setting PulseAudio default source to: $SOURCE_NAME" >&2
                
                # Try to set default source
                SET_SUCCESS=false
                
                # Method 1: Try directly
                if pactl set-default-source "$SOURCE_NAME" 2>/dev/null; then
                    echo "[Audio] ✅ Default source set successfully" >&2
                    SET_SUCCESS=true
                else
                    # Method 2: Try with XDG_RUNTIME_DIR if available
                    if [ -n "$XDG_RUNTIME_DIR" ] && [ -S "$XDG_RUNTIME_DIR/pulse/native" ]; then
                        export PULSE_RUNTIME_PATH="$XDG_RUNTIME_DIR/pulse"
                        if pactl set-default-source "$SOURCE_NAME" 2>/dev/null; then
                            echo "[Audio] ✅ Default source set successfully (using XDG_RUNTIME_DIR)" >&2
                            SET_SUCCESS=true
                        fi
                    fi
                    
                    # Method 3: Try finding PulseAudio socket in common locations
                    if [ "$SET_SUCCESS" = false ]; then
                        for runtime_dir in /run/user/*; do
                            if [ -S "$runtime_dir/pulse/native" ]; then
                                export PULSE_RUNTIME_PATH="$runtime_dir/pulse"
                                if pactl set-default-source "$SOURCE_NAME" 2>/dev/null; then
                                    echo "[Audio] ✅ Default source set successfully (found socket in $runtime_dir)" >&2
                                    SET_SUCCESS=true
                                    break
                                fi
                            fi
                        done
                    fi
                    
                    if [ "$SET_SUCCESS" = false ]; then
                        echo "[Audio] ⚠️  Failed to set default source" >&2
                        echo "[Audio] 💡 You may need to run manually:" >&2
                        echo "[Audio]    pactl set-default-source \"$SOURCE_NAME\"" >&2
                    fi
                fi
            else
                echo "[Audio] ⚠️  ReSpeaker XVF3800 source not found in PulseAudio" >&2
                echo "[Audio] Available sources:" >&2
                pactl list short sources 2>/dev/null | awk '{print "  " $2}' >&2 || true
            fi
        fi
    fi
else
    # Device not found - this is OK, it might not be plugged in
    # Don't fail the service, just exit silently
    exit 0
fi

exit 0


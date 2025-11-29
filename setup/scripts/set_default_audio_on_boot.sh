#!/bin/bash
# Set default audio output to UACDemoV1.0 on boot (optional PulseAudio only)
# This script is called by the aura.service systemd unit
# Note: PulseAudio is optional - speaker.py uses ALSA as fallback
# Note: No .asoundrc is created - PortAudio handles audio directly

# Get user home directory (script runs as the service user)
AURA_HOME="${HOME:-/home/ledger}"

# Wait a bit for audio devices to be ready (they may not be immediately available on boot)
sleep 2

# Check if UACDemoV1.0 audio device exists
# Redirect stderr and handle failures gracefully
if aplay -l 2>/dev/null | grep -q "UACDemoV1.0" 2>/dev/null; then
    CARD_NUM=$(aplay -l 2>/dev/null | grep "UACDemoV1.0" | sed -n 's/.*card \([0-9]*\):.*/\1/p' | head -1)
    
    if [ -n "$CARD_NUM" ]; then
        # Don't create .asoundrc - let PortAudio handle audio directly
        # .asoundrc can interfere with microphone capture
        # Volume control is handled via ALSA amixer commands in speaker.py
        
        # Set PulseAudio default sink (if available) - uses dynamic name matching
        # Note: PulseAudio is optional - speaker.py falls back to ALSA if not available
        if command -v pactl >/dev/null 2>&1; then
            # Wait longer for PulseAudio to be ready (USB devices can take time)
            for i in {1..10}; do
                if pactl list short sinks 2>/dev/null | grep -q .; then
                    break
                fi
                sleep 1
            done
            
            # Try multiple methods to find the sink
            # Priority: Match constant pattern "Jieli_Technology_UACDemoV1.0" (works across all device IDs)
            # Method 1: Look for "Jieli_Technology_UACDemoV1.0" pattern (constant part, ignores device ID)
            SINK_NAME=$(pactl list short sinks 2>/dev/null | grep -i "Jieli_Technology_UACDemoV1.0" | awk '{print $2}' | head -1)
            
            # Method 2: Look for "Jieli.*Technology.*UACDemoV1.0" (more flexible pattern matching)
            if [ -z "$SINK_NAME" ]; then
                SINK_NAME=$(pactl list short sinks 2>/dev/null | grep -iE "Jieli.*Technology.*UACDemoV1\.0|Jieli_Technology_UACDemoV1\.0" | awk '{print $2}' | head -1)
            fi
            
            # Method 3: Look for UACDemoV1.0 anywhere (fallback if Jieli pattern doesn't match)
            if [ -z "$SINK_NAME" ]; then
                SINK_NAME=$(pactl list short sinks 2>/dev/null | grep -i "UACDemoV1.0" | awk '{print $2}' | head -1)
            fi
            
            # Method 4: Look for Jieli Technology with UACDemo (without version)
            if [ -z "$SINK_NAME" ]; then
                SINK_NAME=$(pactl list short sinks 2>/dev/null | grep -i "Jieli.*Technology.*UACDemo\|Jieli.*UACDemo" | awk '{print $2}' | head -1)
            fi
            
            # Method 5: Look for UACDemo anywhere (broader fallback)
            if [ -z "$SINK_NAME" ]; then
                SINK_NAME=$(pactl list short sinks 2>/dev/null | grep -i "UACDemo" | awk '{print $2}' | head -1)
            fi
            
            # Method 6: Look for USB audio devices with "analog-stereo" (last resort fallback)
            if [ -z "$SINK_NAME" ]; then
                SINK_NAME=$(pactl list short sinks 2>/dev/null | grep -i "usb.*analog-stereo" | awk '{print $2}' | head -1)
            fi
            
            if [ -n "$SINK_NAME" ]; then
                echo "[Audio] Setting PulseAudio default sink to: $SINK_NAME" >&2
                
                # Try to set default sink
                SET_SUCCESS=false
                
                # Method 1: Try directly
                if pactl set-default-sink "$SINK_NAME" 2>/dev/null; then
                    echo "[Audio] ✅ Default sink set successfully" >&2
                    SET_SUCCESS=true
                else
                    # Method 2: Try with XDG_RUNTIME_DIR if available
                    if [ -n "$XDG_RUNTIME_DIR" ] && [ -S "$XDG_RUNTIME_DIR/pulse/native" ]; then
                        export PULSE_RUNTIME_PATH="$XDG_RUNTIME_DIR/pulse"
                        if pactl set-default-sink "$SINK_NAME" 2>/dev/null; then
                            echo "[Audio] ✅ Default sink set successfully (using XDG_RUNTIME_DIR)" >&2
                            SET_SUCCESS=true
                        fi
                    fi
                    
                    # Method 3: Try finding PulseAudio socket in common locations
                    if [ "$SET_SUCCESS" = false ]; then
                        for runtime_dir in /run/user/*; do
                            if [ -S "$runtime_dir/pulse/native" ]; then
                                export PULSE_RUNTIME_PATH="$runtime_dir/pulse"
                                if pactl set-default-sink "$SINK_NAME" 2>/dev/null; then
                                    echo "[Audio] ✅ Default sink set successfully (found socket in $runtime_dir)" >&2
                                    SET_SUCCESS=true
                                    break
                                fi
                            fi
                        done
                    fi
                    
                    if [ "$SET_SUCCESS" = false ]; then
                        echo "[Audio] ⚠️  Failed to set default sink" >&2
                        echo "[Audio] 💡 You may need to run manually:" >&2
                        echo "[Audio]    pactl set-default-sink \"$SINK_NAME\"" >&2
                    fi
                fi
            else
                echo "[Audio] ⚠️  UACDemoV1.0 sink not found in PulseAudio" >&2
                echo "[Audio] Available sinks:" >&2
                pactl list short sinks 2>/dev/null | awk '{print "  " $2}' >&2 || true
            fi
        fi
    fi
else
    # Device not found - this is OK, it might not be plugged in
    # Don't fail the service, just exit silently
    exit 0
fi

exit 0


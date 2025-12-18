#!/bin/bash
# Test script to debug PulseAudio sink detection and setting

echo "=========================================="
echo "  PulseAudio Sink Test"
echo "=========================================="
echo ""

# Check if PulseAudio is running
if ! pactl info >/dev/null 2>&1; then
    echo "❌ PulseAudio is not running"
    echo "   Try: pulseaudio --start"
    exit 1
else
    echo "✅ PulseAudio is running"
fi

echo ""
echo "🔍 Finding UACDemo device..."
CARD_NUM=$(aplay -l 2>/dev/null | grep -E "UACDemoV1\.0|UACDemoV10" | sed -n 's/.*card \([0-9]*\):.*/\1/p' | head -1)

if [ -z "$CARD_NUM" ]; then
    echo "❌ UACDemo device not found"
    exit 1
fi

echo "✅ Found UACDemo on card $CARD_NUM"
echo ""

echo "📋 All PulseAudio sinks:"
pactl list sinks | grep -E "^Sink #|^[[:space:]]*Name:|^[[:space:]]*Description:|^[[:space:]]*alsa.card ="
echo ""

echo "🔍 Testing sink detection (same as set_default_audio_on_boot.sh)..."
SINK_NAME=$(pactl list sinks 2>/dev/null | awk -v card="$CARD_NUM" '
    BEGIN { in_sink=0; sink_name=""; sink_desc=""; sink_card=""; match_found=0 }
    /^Sink #/ { 
        if (in_sink && sink_name != "" && match_found) {
            print sink_name
            exit
        }
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
        sink_desc=substr($0, index($0, "Description:") + 13)
    }
    /^[[:space:]]*alsa.card =/ && in_sink { 
        gsub(/"/, "", $3)
        sink_card=$3
    }
    in_sink && sink_name != "" {
        if ((sink_desc ~ /UACDemo/ && sink_desc !~ /XVF3800/) || 
            (sink_card == card && sink_desc !~ /XVF3800/)) {
            match_found=1
        }
    }
    /^$/ && in_sink {
        if (match_found && sink_name != "") {
            print sink_name
            exit
        }
        in_sink=0
    }
    END {
        if (in_sink && match_found && sink_name != "") {
            print sink_name
        }
    }
')

if [ -z "$SINK_NAME" ]; then
    echo "❌ Could not find sink name"
    echo ""
    echo "Trying fallback method..."
    SINK_NAME=$(pactl list sinks 2>/dev/null | awk '
        /^[[:space:]]*Name:/ { sink_name=$2 }
        /UACDemo/ && !/XVF3800/ && sink_name != "" { print sink_name; exit }
    ')
fi

if [ -n "$SINK_NAME" ]; then
    echo "✅ Found sink: $SINK_NAME"
    echo ""
    
    echo "🔧 Testing set-default-sink..."
    if pactl set-default-sink "$SINK_NAME" 2>&1; then
        echo "✅ Command succeeded"
        
        # Verify
        CURRENT_DEFAULT=$(pactl info 2>/dev/null | grep "Default Sink:" | sed 's/Default Sink: //')
        echo "   Current default: $CURRENT_DEFAULT"
        
        if [ "$CURRENT_DEFAULT" = "$SINK_NAME" ]; then
            echo "✅ Default sink is correctly set!"
        else
            echo "⚠️  Default sink doesn't match (expected: $SINK_NAME)"
        fi
    else
        echo "❌ Command failed"
        echo "   Error output:"
        pactl set-default-sink "$SINK_NAME" 2>&1
    fi
else
    echo "❌ Could not find sink name"
    echo ""
    echo "Manual search:"
    pactl list sinks | grep -B 2 -A 2 "UACDemo"
fi

echo ""
echo "=========================================="

#!/bin/bash
# collect_wake_word_data.sh - Helper script for collecting wake word training data
# Usage: ./collect_wake_word_data.sh [wake-word|not-wake-word|tts-echo|noise]

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

DATA_TYPE="${1:-wake-word}"
TRAINING_DIR=~/precise-training
DURATION=2  # seconds
SAMPLE_RATE=16000

# Create directory structure
mkdir -p "$TRAINING_DIR"/{wake-word,not-wake-word,tts-echo,noise}

print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_prompt() {
    echo -e "${YELLOW}[PROMPT]${NC} $1"
}

# Detect audio device
AUDIO_DEVICE="plughw:0,0"
if ! arecord -l | grep -q "card 0"; then
    print_info "Detecting audio device..."
    AUDIO_DEVICE=$(arecord -l | head -1 | grep -oP 'card \K[0-9]+' | head -1)
    if [ -z "$AUDIO_DEVICE" ]; then
        AUDIO_DEVICE="plughw:0,0"
    else
        AUDIO_DEVICE="plughw:${AUDIO_DEVICE},0"
    fi
fi

print_info "Using audio device: $AUDIO_DEVICE"
print_info "Recording directory: $TRAINING_DIR/$DATA_TYPE"
print_info ""

case "$DATA_TYPE" in
    wake-word)
        print_prompt "Recording 'Hey Aura' samples"
        print_prompt "Press ENTER to start recording, then say 'Hey Aura'"
        print_prompt "Press Ctrl+C to stop"
        echo ""
        
        counter=1
        while true; do
            filename=$(printf "%s/%s/sample_%03d.wav" "$TRAINING_DIR" "$DATA_TYPE" $counter)
            print_info "Recording sample $counter... (Press Ctrl+C to stop)"
            read -p "Press ENTER to record (2 seconds)..." dummy
            
            arecord -D "$AUDIO_DEVICE" -f S16_LE -r $SAMPLE_RATE -c 1 -d $DURATION "$filename" 2>/dev/null
            
            if [ -f "$filename" ]; then
                # Check file size (should be > 1000 bytes for valid audio)
                size=$(stat -f%z "$filename" 2>/dev/null || stat -c%s "$filename" 2>/dev/null)
                if [ "$size" -gt 1000 ]; then
                    print_success "Saved: $filename"
                    counter=$((counter + 1))
                else
                    print_info "Recording too short, retrying..."
                    rm -f "$filename"
                fi
            else
                print_info "Recording failed, retrying..."
            fi
            echo ""
        done
        ;;
        
    not-wake-word)
        print_prompt "Recording 'not wake word' samples"
        print_prompt "Say anything EXCEPT 'Hey Aura'"
        print_prompt "Examples: 'Hey you', 'What time is it', 'Turn on lights'"
        print_prompt "Press Ctrl+C to stop"
        echo ""
        
        counter=1
        while true; do
            filename=$(printf "%s/%s/sample_%03d.wav" "$TRAINING_DIR" "$DATA_TYPE" $counter)
            print_info "Recording sample $counter... (Press Ctrl+C to stop)"
            read -p "Press ENTER to record (2 seconds)..." dummy
            
            arecord -D "$AUDIO_DEVICE" -f S16_LE -r $SAMPLE_RATE -c 1 -d $DURATION "$filename" 2>/dev/null
            
            if [ -f "$filename" ]; then
                size=$(stat -f%z "$filename" 2>/dev/null || stat -c%s "$filename" 2>/dev/null)
                if [ "$size" -gt 1000 ]; then
                    print_success "Saved: $filename"
                    counter=$((counter + 1))
                else
                    print_info "Recording too short, retrying..."
                    rm -f "$filename"
                fi
            else
                print_info "Recording failed, retrying..."
            fi
            echo ""
        done
        ;;
        
    tts-echo)
        print_prompt "Recording TTS/echo samples"
        print_prompt "This will record TTS audio from speakers (echo)"
        print_prompt "Steps:"
        print_prompt "  1. Start recording"
        print_prompt "  2. Trigger TTS response (ask Aura a question)"
        print_prompt "  3. Let TTS play through speakers"
        print_prompt "  4. Stop recording after TTS finishes"
        print_prompt ""
        print_prompt "Press Ctrl+C to stop"
        echo ""
        
        counter=1
        while true; do
            filename=$(printf "%s/%s/tts_echo_%03d.wav" "$TRAINING_DIR" "$DATA_TYPE" $counter)
            print_info "Recording TTS echo sample $counter..."
            print_info "Press ENTER to start recording, then trigger TTS"
            read -p "Press ENTER to start (will record for 10 seconds)..." dummy
            
            # Record for 10 seconds (longer for TTS responses)
            arecord -D "$AUDIO_DEVICE" -f S16_LE -r $SAMPLE_RATE -c 1 -d 10 "$filename" 2>/dev/null &
            RECORD_PID=$!
            
            print_info "Recording... (PID: $RECORD_PID)"
            print_info "Now trigger TTS response (ask Aura a question)"
            print_info "Press ENTER when TTS finishes to stop recording..."
            read dummy
            
            # Stop recording
            kill $RECORD_PID 2>/dev/null || true
            wait $RECORD_PID 2>/dev/null || true
            
            if [ -f "$filename" ]; then
                size=$(stat -f%z "$filename" 2>/dev/null || stat -c%s "$filename" 2>/dev/null)
                if [ "$size" -gt 10000 ]; then
                    print_success "Saved: $filename"
                    counter=$((counter + 1))
                else
                    print_info "Recording too short, retrying..."
                    rm -f "$filename"
                fi
            else
                print_info "Recording failed, retrying..."
            fi
            echo ""
        done
        ;;
        
    noise)
        print_prompt "Recording background noise samples"
        print_prompt "Record ambient noise (room tone, fan, etc.)"
        print_prompt "Press Ctrl+C to stop"
        echo ""
        
        counter=1
        while true; do
            filename=$(printf "%s/%s/noise_%03d.wav" "$TRAINING_DIR" "$DATA_TYPE" $counter)
            print_info "Recording noise sample $counter... (Press Ctrl+C to stop)"
            read -p "Press ENTER to record (3 seconds)..." dummy
            
            arecord -D "$AUDIO_DEVICE" -f S16_LE -r $SAMPLE_RATE -c 1 -d 3 "$filename" 2>/dev/null
            
            if [ -f "$filename" ]; then
                size=$(stat -f%z "$filename" 2>/dev/null || stat -c%s "$filename" 2>/dev/null)
                if [ "$size" -gt 1000 ]; then
                    print_success "Saved: $filename"
                    counter=$((counter + 1))
                else
                    print_info "Recording too short, retrying..."
                    rm -f "$filename"
                fi
            else
                print_info "Recording failed, retrying..."
            fi
            echo ""
        done
        ;;
        
    *)
        echo "Usage: $0 [wake-word|not-wake-word|tts-echo|noise]"
        echo ""
        echo "Data types:"
        echo "  wake-word      - Record 'Hey Aura' samples (positive data)"
        echo "  not-wake-word  - Record other speech (negative data)"
        echo "  tts-echo       - Record TTS audio from speakers (critical for echo rejection)"
        echo "  noise          - Record background noise"
        exit 1
        ;;
esac


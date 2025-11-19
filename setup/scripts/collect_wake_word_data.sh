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

print_error() {
    echo -e "\033[0;31m[ERROR]${NC} $1"
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
        echo ""
        
        # Check for assets directory
        ASSETS_DIR="$HOME/LedgerAI/assets/voice_samples"
        if [ ! -d "$ASSETS_DIR" ]; then
            ASSETS_DIR="$HOME/LedgerAI/assets/prompts"
        fi
        
        # Ask user for mode
        echo "Select recording mode:"
        echo "  1. Manual - You trigger TTS manually (ask Aura a question)"
        echo "  2. Auto (audio file) - Play sample audio from assets automatically"
        echo "  3. Auto (TTS API) - Generate TTS automatically via ElevenLabs API"
        read -p "Enter mode (1/2/3) [default: 1]: " mode
        mode="${mode:-1}"
        
        case "$mode" in
            2)
                # Auto mode: Play audio files
                if [ ! -d "$ASSETS_DIR" ] || [ -z "$(ls -A "$ASSETS_DIR"/*.wav 2>/dev/null)" ]; then
                    print_error "No audio files found in $ASSETS_DIR"
                    print_info "Falling back to manual mode..."
                    mode=1
                else
                    print_info "Found audio files in $ASSETS_DIR"
                    AUDIO_FILES=("$ASSETS_DIR"/*.wav)
                    print_info "Will play ${#AUDIO_FILES[@]} audio files automatically"
                fi
                ;;
            3)
                # Auto mode: TTS API
                if [ ! -f "$HOME/LedgerAI/.env" ]; then
                    print_error ".env file not found - cannot use TTS API"
                    print_info "Falling back to manual mode..."
                    mode=1
                else
                    # Check for ElevenLabs API key
                    if ! grep -q "ELEVENLABS_API_KEY" "$HOME/LedgerAI/.env" 2>/dev/null; then
                        print_error "ELEVENLABS_API_KEY not found in .env"
                        print_info "Falling back to manual mode..."
                        mode=1
                    else
                        print_info "TTS API mode enabled"
                        # Load API key
                        export $(grep "ELEVENLABS_API_KEY" "$HOME/LedgerAI/.env" | xargs)
                        export $(grep "ELEVENLABS_VOICE_ID" "$HOME/LedgerAI/.env" | xargs 2>/dev/null || echo "ELEVENLABS_VOICE_ID=default")
                    fi
                fi
                ;;
        esac
        
        print_prompt "Press Ctrl+C to stop"
        echo ""
        
        counter=1
        while true; do
            filename=$(printf "%s/%s/tts_echo_%03d.wav" "$TRAINING_DIR" "$DATA_TYPE" $counter)
            
            if [ "$mode" = "1" ]; then
                # Manual mode
                print_info "Recording TTS echo sample $counter..."
                print_info "Press ENTER to start recording, then trigger TTS"
                read -p "Press ENTER to start (will record for 10 seconds)..." dummy
                
                # Record for 10 seconds
                arecord -D "$AUDIO_DEVICE" -f S16_LE -r $SAMPLE_RATE -c 1 -d 10 "$filename" 2>/dev/null &
                RECORD_PID=$!
                
                print_info "Recording... (PID: $RECORD_PID)"
                print_info "Now trigger TTS response (ask Aura a question)"
                print_info "Press ENTER when TTS finishes to stop recording..."
                read dummy
                
                # Stop recording
                kill $RECORD_PID 2>/dev/null || true
                wait $RECORD_PID 2>/dev/null || true
                
            elif [ "$mode" = "2" ]; then
                # Auto mode: Play audio file
                audio_file="${AUDIO_FILES[$((counter - 1))]}"
                if [ -z "$audio_file" ] || [ ! -f "$audio_file" ]; then
                    print_info "All audio files played. Starting over..."
                    counter=1
                    audio_file="${AUDIO_FILES[0]}"
                fi
                
                print_info "Recording TTS echo sample $counter..."
                print_info "Will play: $(basename "$audio_file")"
                read -p "Press ENTER to start (will record for 10 seconds)..." dummy
                
                # Start recording
                arecord -D "$AUDIO_DEVICE" -f S16_LE -r $SAMPLE_RATE -c 1 -d 10 "$filename" 2>/dev/null &
                RECORD_PID=$!
                
                # Small delay to ensure recording started
                sleep 0.2
                
                # Play audio file
                print_info "Playing audio file..."
                aplay -q "$audio_file" 2>/dev/null || {
                    print_error "Failed to play audio file"
                    kill $RECORD_PID 2>/dev/null || true
                    continue
                }
                
                # Wait a bit for echo to settle
                sleep 0.5
                
                # Stop recording
                kill $RECORD_PID 2>/dev/null || true
                wait $RECORD_PID 2>/dev/null || true
                
            elif [ "$mode" = "3" ]; then
                # Auto mode: TTS API
                # Predefined TTS phrases (including "Hey Aura" for critical testing)
                TTS_PHRASES=(
                    "Hey Aura, what time is it?"
                    "Hello, this is a test response."
                    "The weather today is sunny and warm."
                    "I can help you with various tasks."
                    "This is an automated TTS sample for training."
                    "Hey Aura, how are you today?"
                    "Thank you for using the voice assistant."
                    "I'm here to assist you with your questions."
                )
                
                phrase="${TTS_PHRASES[$((counter - 1))]}"
                if [ -z "$phrase" ]; then
                    print_info "All phrases used. Starting over..."
                    counter=1
                    phrase="${TTS_PHRASES[0]}"
                fi
                
                print_info "Recording TTS echo sample $counter..."
                print_info "Will generate TTS: \"$phrase\""
                read -p "Press ENTER to start (will record for 10 seconds)..." dummy
                
                # Start recording
                arecord -D "$AUDIO_DEVICE" -f S16_LE -r $SAMPLE_RATE -c 1 -d 10 "$filename" 2>/dev/null &
                RECORD_PID=$!
                
                # Small delay to ensure recording started
                sleep 0.2
                
                # Generate and play TTS using helper script
                print_info "Generating TTS..."
                SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
                python3 "$SCRIPT_DIR/play_tts.py" "$phrase" 2>/dev/null || {
                    print_error "TTS generation failed"
                    kill $RECORD_PID 2>/dev/null || true
                    continue
                }
                
                # Wait a bit for echo to settle
                sleep 0.5
                
                # Stop recording
                kill $RECORD_PID 2>/dev/null || true
                wait $RECORD_PID 2>/dev/null || true
            fi
            
            # Validate recording
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


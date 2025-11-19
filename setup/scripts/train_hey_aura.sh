#!/bin/bash
# train_hey_aura.sh - Automated training script for "Hey Aura" wake word model
# Usage: ./train_hey_aura.sh [epochs]

set -e

EPOCHS="${1:-50}"
TRAINING_DIR=~/precise-training
MODEL_NAME=hey-aura

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

echo "=========================================="
echo "  Training 'Hey Aura' Wake Word Model"
echo "=========================================="
echo ""

# Check if virtual environment is activated
if [ -z "$VIRTUAL_ENV" ]; then
    if [ -d ~/aura-env ]; then
        print_info "Activating virtual environment..."
        source ~/aura-env/bin/activate
    else
        print_error "Virtual environment not found. Please activate it manually:"
        print_error "  source ~/aura-env/bin/activate"
        exit 1
    fi
fi

# Check training data exists
if [ ! -d "$TRAINING_DIR/wake-word" ]; then
    print_error "Training data not found at $TRAINING_DIR"
    print_info "Please collect training data first:"
    print_info "  ./collect_wake_word_data.sh wake-word"
    print_info "  ./collect_wake_word_data.sh not-wake-word"
    print_info "  ./collect_wake_word_data.sh tts-echo"
    print_info "  ./collect_wake_word_data.sh noise"
    exit 1
fi

# Count samples (use find to handle empty directories and hidden files)
wake_word_count=$(find "$TRAINING_DIR/wake-word" -maxdepth 1 -name "*.wav" -type f 2>/dev/null | wc -l)
not_wake_word_count=$(find "$TRAINING_DIR/not-wake-word" -maxdepth 1 -name "*.wav" -type f 2>/dev/null | wc -l)
tts_echo_count=$(find "$TRAINING_DIR/tts-echo" -maxdepth 1 -name "*.wav" -type f 2>/dev/null | wc -l)
noise_count=$(find "$TRAINING_DIR/noise" -maxdepth 1 -name "*.wav" -type f 2>/dev/null | wc -l)

# Debug: Show actual files found
print_info "Debug: Checking actual files in directories..."
if [ "$wake_word_count" -gt 0 ]; then
    print_info "  Wake word files found:"
    find "$TRAINING_DIR/wake-word" -maxdepth 1 -name "*.wav" -type f 2>/dev/null | head -5 | while read f; do
        print_info "    - $(basename "$f")"
    done
else
    print_warning "  No wake word files found in $TRAINING_DIR/wake-word"
    print_info "  Directory contents:"
    ls -la "$TRAINING_DIR/wake-word/" 2>/dev/null | head -10 || print_info "    (directory empty or not accessible)"
fi

print_info "Training data summary:"
print_info "  Wake word samples: $wake_word_count"
print_info "  Not wake word samples: $not_wake_word_count"
print_info "  TTS/echo samples: $tts_echo_count"
print_info "  Noise samples: $noise_count"
echo ""

# Check minimum requirements
if [ "$wake_word_count" -lt 50 ]; then
    print_warning "Low number of wake word samples ($wake_word_count). Recommended: 200+"
fi

if [ "$tts_echo_count" -lt 20 ]; then
    print_warning "Low number of TTS/echo samples ($tts_echo_count). Recommended: 100+"
    print_warning "Echo rejection may not work well without sufficient TTS data!"
fi

# Prepare combined negative data
print_info "Preparing negative training data..."
COMBINED_NEGATIVE="$TRAINING_DIR/combined-negative"
mkdir -p "$COMBINED_NEGATIVE"
rm -f "$COMBINED_NEGATIVE"/*.wav

# Copy negative samples
if [ "$not_wake_word_count" -gt 0 ]; then
    cp "$TRAINING_DIR/not-wake-word"/*.wav "$COMBINED_NEGATIVE/" 2>/dev/null || true
fi

if [ "$tts_echo_count" -gt 0 ]; then
    cp "$TRAINING_DIR/tts-echo"/*.wav "$COMBINED_NEGATIVE/" 2>/dev/null || true
fi

if [ "$noise_count" -gt 0 ]; then
    cp "$TRAINING_DIR/noise"/*.wav "$COMBINED_NEGATIVE/" 2>/dev/null || true
fi

combined_count=$(ls "$COMBINED_NEGATIVE"/*.wav 2>/dev/null | wc -l)
print_success "Combined negative data: $combined_count samples"

if [ "$combined_count" -eq 0 ]; then
    print_error "No negative training data found!"
    exit 1
fi

# Check precise-train is available (as command, Python module, or wrapper)
PRECISE_TRAIN_CMD=""
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

if command -v precise-train &> /dev/null; then
    PRECISE_TRAIN_CMD="precise-train"
    print_info "Found precise-train command"
elif [ -f "$VIRTUAL_ENV/bin/precise-train" ]; then
    PRECISE_TRAIN_CMD="$VIRTUAL_ENV/bin/precise-train"
    print_info "Found precise-train in venv bin"
elif python3 -c "import precise.train" 2>/dev/null; then
    # Try using Python module directly
    PRECISE_TRAIN_CMD="python3 -m precise.train"
    print_info "Found precise.train module"
elif python3 -c "from precise import train" 2>/dev/null; then
    # Alternative import path
    PRECISE_TRAIN_CMD="python3 -c 'from precise import train; import sys; train.main()'"
    print_info "Found precise.train via alternative import"
elif [ -f "$SCRIPT_DIR/precise_train_wrapper.py" ]; then
    # Use wrapper script as fallback
    PRECISE_TRAIN_CMD="python3 $SCRIPT_DIR/precise_train_wrapper.py"
    print_info "Using precise_train_wrapper.py"
else
    print_error "precise-train not found!"
    echo ""
    print_info "Diagnostics:"
    # Check if precise package is installed
    if python3 -c "import precise" 2>/dev/null; then
        print_info "  ✅ precise package is installed"
        print_info "  ⚠️  But precise-train command not found"
        print_info "  💡 Try: pip install --upgrade --force-reinstall precise-runner"
    else
        print_info "  ❌ precise package is NOT installed"
    fi
    
    if python3 -c "import precise_runner" 2>/dev/null; then
        print_info "  ✅ precise-runner is installed"
        print_info "  💡 Try: pip install --upgrade --force-reinstall precise-runner"
        print_info "  Or check: ls ~/aura-env/bin/ | grep precise"
    else
        print_info "  ❌ precise-runner is NOT installed"
        print_info "  💡 Install: pip install precise-runner"
    fi
    
    print_info ""
    print_info "Installation options:"
    print_info "  1. pip install --upgrade --force-reinstall precise-runner"
    print_info "  2. pip install --ignore-installed precise"
    print_info "  3. Or use fix script: ./fix_numpy_scipy_compatibility.sh"
    exit 1
fi

print_info "Using: $PRECISE_TRAIN_CMD"

# Train model
echo ""
print_info "Training model with $EPOCHS epochs..."
print_info "This may take a while (10-30 minutes depending on data size)..."
echo ""

# Change to training directory - precise-train expects directories in current dir
print_info "Setting up training environment..."
cd "$TRAINING_DIR"
print_info "Working directory: $(pwd)"

# Verify wake-word directory exists and has files
if [ ! -d "wake-word" ]; then
    print_error "Wake word directory missing!"
    print_info "Expected: $TRAINING_DIR/wake-word"
    exit 1
fi

# Re-count files in current directory (after cd)
wake_word_count_local=$(find wake-word -maxdepth 1 -name "*.wav" -type f 2>/dev/null | wc -l)
if [ "$wake_word_count_local" -eq 0 ]; then
    print_error "Wake word directory is empty!"
    print_info "Directory: $(pwd)/wake-word"
    print_info "Contents:"
    ls -la wake-word/ 2>/dev/null | head -10 || print_info "  (directory empty)"
    print_info "Please collect wake word samples first:"
    print_info "  ./collect_wake_word_data.sh wake-word"
    exit 1
fi

# Mycroft Precise expects directories named "wake-word" and "not-wake-word" in current dir
# It does NOT accept directories as command-line arguments
# Format: precise-train model.net -e N
# The tool automatically looks for "wake-word" and "not-wake-word" directories

# Prepare not-wake-word directory from combined-negative
# Mycroft Precise expects "not-wake-word" directory, not "combined-negative"
print_info "Preparing not-wake-word directory for training..."
COMBINED_NEGATIVE_FULL="$TRAINING_DIR/combined-negative"

if [ ! -d "$COMBINED_NEGATIVE_FULL" ] || [ "$combined_count" -eq 0 ]; then
    print_error "Combined negative data missing or empty!"
    print_info "Expected: $COMBINED_NEGATIVE_FULL with .wav files"
    exit 1
fi

# Backup existing not-wake-word if it exists and is different
if [ -d "not-wake-word" ]; then
    existing_count=$(find not-wake-word -maxdepth 1 -name "*.wav" -type f 2>/dev/null | wc -l)
    if [ "$existing_count" -gt 0 ] && [ "$existing_count" -ne "$combined_count" ]; then
        print_info "Backing up existing not-wake-word ($existing_count files)..."
        mv not-wake-word "not-wake-word.backup.$(date +%s)"
    fi
fi

# Copy combined-negative to not-wake-word
print_info "Copying combined-negative to not-wake-word..."
rm -rf not-wake-word
cp -r "$COMBINED_NEGATIVE_FULL" not-wake-word
not_wake_word_count_local=$(find not-wake-word -maxdepth 1 -name "*.wav" -type f 2>/dev/null | wc -l)
print_info "✅ Copied $not_wake_word_count_local files to not-wake-word/"

# Verify both directories are ready
print_info "Training directories ready:"
print_info "  - wake-word/ ($wake_word_count_local files)"
print_info "  - not-wake-word/ ($not_wake_word_count_local files)"

# Build the training command
# Based on help output, the format shows: :-e --epochs int 10
# This means both -e and --epochs should work, but let's try --epochs first
# Format: precise-train model.net --epochs N
# Directories are automatically detected from current directory
TRAIN_ARGS="${MODEL_NAME}.net --epochs $EPOCHS"
print_info "Training command: $PRECISE_TRAIN_CMD $TRAIN_ARGS"
print_info "Note: precise-train will automatically use wake-word/ and not-wake-word/ directories"

# Try different argument formats
# The prettyparse patch might not handle all formats correctly
# Let's try each format and capture output to see what works
print_info "Attempting training with different argument formats..."
TRAINING_SUCCESS=false
LAST_ERROR=""

# Try 1: --epochs as separate argument (most standard)
print_info "Trying: $PRECISE_TRAIN_CMD ${MODEL_NAME}.net --epochs $EPOCHS"
if TRAIN_OUTPUT=$($PRECISE_TRAIN_CMD "${MODEL_NAME}.net" "--epochs" "$EPOCHS" 2>&1); then
    print_success "Training complete!"
    TRAINING_SUCCESS=true
else
    LAST_ERROR="$TRAIN_OUTPUT"
    print_info "  ❌ Failed: $(echo "$TRAIN_OUTPUT" | head -3 | tr '\n' ' ')"
fi

# Try 2: -e as separate argument
if [ "$TRAINING_SUCCESS" = false ]; then
    print_info "Trying: $PRECISE_TRAIN_CMD ${MODEL_NAME}.net -e $EPOCHS"
    if TRAIN_OUTPUT=$($PRECISE_TRAIN_CMD "${MODEL_NAME}.net" "-e" "$EPOCHS" 2>&1); then
        print_success "Training complete!"
        TRAINING_SUCCESS=true
    else
        LAST_ERROR="$TRAIN_OUTPUT"
        print_info "  ❌ Failed: $(echo "$TRAIN_OUTPUT" | head -3 | tr '\n' ' ')"
    fi
fi

# Try 3: --epochs=50 format (equals sign)
if [ "$TRAINING_SUCCESS" = false ]; then
    print_info "Trying: $PRECISE_TRAIN_CMD ${MODEL_NAME}.net --epochs=$EPOCHS"
    if TRAIN_OUTPUT=$($PRECISE_TRAIN_CMD "${MODEL_NAME}.net" "--epochs=$EPOCHS" 2>&1); then
        print_success "Training complete!"
        TRAINING_SUCCESS=true
    else
        LAST_ERROR="$TRAIN_OUTPUT"
        print_info "  ❌ Failed: $(echo "$TRAIN_OUTPUT" | head -3 | tr '\n' ' ')"
    fi
fi

# Try 4: -e=50 format (equals sign)
if [ "$TRAINING_SUCCESS" = false ]; then
    print_info "Trying: $PRECISE_TRAIN_CMD ${MODEL_NAME}.net -e=$EPOCHS"
    if TRAIN_OUTPUT=$($PRECISE_TRAIN_CMD "${MODEL_NAME}.net" "-e=$EPOCHS" 2>&1); then
        print_success "Training complete!"
        TRAINING_SUCCESS=true
    else
        LAST_ERROR="$TRAIN_OUTPUT"
        print_info "  ❌ Failed: $(echo "$TRAIN_OUTPUT" | head -3 | tr '\n' ' ')"
    fi
fi

# Try 5: No epochs argument (use default of 10)
if [ "$TRAINING_SUCCESS" = false ]; then
    print_warning "All epochs formats failed, trying without epochs (will use default: 10)"
    print_info "Trying: $PRECISE_TRAIN_CMD ${MODEL_NAME}.net"
    if TRAIN_OUTPUT=$($PRECISE_TRAIN_CMD "${MODEL_NAME}.net" 2>&1); then
        print_success "Training complete (used default epochs: 10)!"
        print_warning "Note: Used default epochs instead of requested $EPOCHS"
        TRAINING_SUCCESS=true
    else
        LAST_ERROR="$TRAIN_OUTPUT"
        print_info "  ❌ Failed: $(echo "$TRAIN_OUTPUT" | head -5 | tr '\n' ' ')"
        print_info ""
        print_info "Last error output:"
        echo "$LAST_ERROR" | head -10 | while read line; do
            print_info "  $line"
        done
    fi
fi

if [ "$TRAINING_SUCCESS" = false ]; then
    print_error "Training failed with all argument formats!"
    print_info ""
    print_info "Full error output from last attempt:"
    echo "$LAST_ERROR"
    print_info ""
    print_info "Troubleshooting:"
    print_info "1. Re-patch prettyparse (may need updated patch):"
    print_info "   cd ~/LedgerAI/setup/scripts"
    print_info "   python3 patch_prettyparse.py"
    print_info ""
    print_info "2. Check precise-train help: $PRECISE_TRAIN_CMD --help"
    print_info ""
    print_info "3. Verify training data exists:"
    print_info "   - ls $TRAINING_DIR/wake-word/*.wav | wc -l"
    print_info "   - ls $TRAINING_DIR/not-wake-word/*.wav | wc -l"
    print_info ""
    print_info "4. Try running manually to see exact error:"
    print_info "   cd $TRAINING_DIR"
    print_info "   $PRECISE_TRAIN_CMD ${MODEL_NAME}.net --epochs $EPOCHS"
    print_info ""
    print_info "5. Check if prettyparse is correctly patched:"
    print_info "   python3 -c \"from prettyparse import create_parser, add_to_parser; print('OK')\""
    exit 1
fi

# Convert to .pb format
echo ""
print_info "Converting to .pb format..."
if command -v precise-convert &> /dev/null; then
    if precise-convert "${MODEL_NAME}.net" "${MODEL_NAME}.pb"; then
        print_success "Conversion complete!"
    else
        print_warning "Conversion failed (model may still work with .net format)"
    fi
else
    print_warning "precise-convert not found. Model will be in .net format"
    print_info "You may need to convert manually or use .net format"
fi

# Test model if test data exists
if [ -d "$TRAINING_DIR/test" ]; then
    echo ""
    print_info "Testing model on validation set..."
    if [ -d "$TRAINING_DIR/test/wake-word" ] && [ -d "$TRAINING_DIR/test/combined-negative" ]; then
        # Prepare test negative data
        mkdir -p "$TRAINING_DIR/test/combined-negative"
        rm -f "$TRAINING_DIR/test/combined-negative"/*.wav
        cp "$TRAINING_DIR/test/not-wake-word"/*.wav "$TRAINING_DIR/test/combined-negative/" 2>/dev/null || true
        cp "$TRAINING_DIR/test/tts-echo"/*.wav "$TRAINING_DIR/test/combined-negative/" 2>/dev/null || true
        cp "$TRAINING_DIR/test/noise"/*.wav "$TRAINING_DIR/test/combined-negative/" 2>/dev/null || true
        
        if command -v precise-test &> /dev/null; then
            precise-test "${MODEL_NAME}.net" \
                test/wake-word \
                test/combined-negative
        else
            print_warning "precise-test not found, skipping validation"
        fi
    else
        print_warning "Test data structure incomplete, skipping validation"
    fi
else
    print_warning "Test directory not found, skipping validation"
fi

# Install model
echo ""
print_info "Installing model..."
MODEL_DIR=~/precise-models
mkdir -p "$MODEL_DIR"

if [ -f "${MODEL_NAME}.pb" ]; then
    cp "${MODEL_NAME}.pb" "$MODEL_DIR/"
    chmod 644 "$MODEL_DIR/${MODEL_NAME}.pb"
    print_success "Model installed: $MODEL_DIR/${MODEL_NAME}.pb"
elif [ -f "${MODEL_NAME}.net" ]; then
    # Try to use .net format (may need conversion later)
    cp "${MODEL_NAME}.net" "$MODEL_DIR/${MODEL_NAME}.net"
    print_success "Model installed: $MODEL_DIR/${MODEL_NAME}.net"
    print_warning "Note: Model is in .net format. You may need to convert to .pb"
else
    print_error "Model file not found after training!"
    exit 1
fi

echo ""
print_success "✅ Training complete!"
echo ""
print_info "Next steps:"
print_info "  1. Configure model path in Settings → AI Model Settings"
print_info "     Path: $MODEL_DIR/${MODEL_NAME}.pb (or .net)"
print_info "  2. Restart Aura"
print_info "  3. Test wake word detection"
print_info "  4. Verify TTS rejection (should NOT trigger on TTS audio)"
echo ""


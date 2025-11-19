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

# Count samples
wake_word_count=$(ls "$TRAINING_DIR/wake-word"/*.wav 2>/dev/null | wc -l)
not_wake_word_count=$(ls "$TRAINING_DIR/not-wake-word"/*.wav 2>/dev/null | wc -l)
tts_echo_count=$(ls "$TRAINING_DIR/tts-echo"/*.wav 2>/dev/null | wc -l)
noise_count=$(ls "$TRAINING_DIR/noise"/*.wav 2>/dev/null | wc -l)

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

cd "$TRAINING_DIR"

# Use the detected command (may be "precise-train" or "python3 -m precise.train")
if $PRECISE_TRAIN_CMD -e "$EPOCHS" "${MODEL_NAME}.net" \
    wake-word \
    "$COMBINED_NEGATIVE"; then
    print_success "Training complete!"
else
    print_error "Training failed!"
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


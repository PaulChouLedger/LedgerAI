#!/bin/bash
# Step 4: Train Piper TTS on the Olga dataset.
#
# Runs on the RTX workstation. Training takes ~1-2 days for 8+ hours of audio.
# Produces ONNX weights that can be deployed directly to Jetson.
#
# Prerequisites:
#   git clone https://github.com/rhasspy/piper.git /home/paul/piper
#   cd /home/paul/piper/src/python && pip install -e .
#   pip install piper-phonemize onnxruntime
#
# Usage:
#   ./train_piper.sh              # Start training from scratch
#   ./train_piper.sh --resume     # Resume from last checkpoint

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DATASET_DIR="$SCRIPT_DIR/piper_dataset"
OUTPUT_DIR="$SCRIPT_DIR/output"
PIPER_DIR="${PIPER_DIR:-/home/paul/piper}"

# Piper medium quality model (good balance of speed and quality)
# Options: x-low, low, medium, high
QUALITY="medium"

mkdir -p "$OUTPUT_DIR"

echo "============================================"
echo "Piper TTS Voice Training"
echo "============================================"
echo "Dataset:  $DATASET_DIR"
echo "Output:   $OUTPUT_DIR"
echo "Quality:  $QUALITY"
echo ""

# Check prerequisites
if [ ! -d "$PIPER_DIR" ]; then
    echo "Piper not found at $PIPER_DIR"
    echo "Install with:"
    echo "  git clone https://github.com/rhasspy/piper.git $PIPER_DIR"
    echo "  cd $PIPER_DIR/src/python && pip install -e ."
    exit 1
fi

if [ ! -d "$DATASET_DIR/wavs" ]; then
    echo "Dataset not found at $DATASET_DIR/wavs"
    echo "Run build_piper_dataset.py first."
    exit 1
fi

# Count training files
NUM_FILES=$(ls "$DATASET_DIR/wavs/"*.wav 2>/dev/null | wc -l)
echo "Training files: $NUM_FILES"
echo ""

# Step 1: Preprocess dataset (compute mel spectrograms, phonemize text)
echo "[Step 1/3] Preprocessing dataset..."
python3 -m piper_train.preprocess \
    --language en-us \
    --input-dir "$DATASET_DIR" \
    --output-dir "$OUTPUT_DIR/preprocessed" \
    --dataset-format ljspeech \
    --single-speaker \
    --sample-rate 22050

# Step 2: Train
echo ""
echo "[Step 2/3] Training (this will take 1-2 days)..."
echo "Monitor with: tail -f $OUTPUT_DIR/training.log"
echo ""

RESUME_FLAG=""
if [ "$1" = "--resume" ] && [ -d "$OUTPUT_DIR/checkpoints" ]; then
    LATEST_CKPT=$(ls -t "$OUTPUT_DIR/checkpoints/"*.ckpt 2>/dev/null | head -1)
    if [ -n "$LATEST_CKPT" ]; then
        RESUME_FLAG="--resume-from-checkpoint $LATEST_CKPT"
        echo "Resuming from: $LATEST_CKPT"
    fi
fi

python3 -m piper_train \
    --dataset-dir "$OUTPUT_DIR/preprocessed" \
    --accelerator gpu \
    --devices 1 \
    --batch-size 16 \
    --validation-split 0.05 \
    --num-test-examples 5 \
    --max-epochs 10000 \
    --quality "$QUALITY" \
    --checkpoint-epochs 250 \
    --precision 16 \
    $RESUME_FLAG \
    2>&1 | tee "$OUTPUT_DIR/training.log"

# Step 3: Export to ONNX
echo ""
echo "[Step 3/3] Exporting to ONNX..."
BEST_CKPT=$(ls -t "$OUTPUT_DIR"/lightning_logs/*/checkpoints/*.ckpt 2>/dev/null | head -1)
if [ -z "$BEST_CKPT" ]; then
    BEST_CKPT=$(ls -t "$OUTPUT_DIR/checkpoints/"*.ckpt 2>/dev/null | head -1)
fi

if [ -z "$BEST_CKPT" ]; then
    echo "No checkpoint found! Training may not have completed."
    exit 1
fi

echo "Exporting checkpoint: $BEST_CKPT"
python3 -m piper_train.export_onnx \
    "$BEST_CKPT" \
    "$OUTPUT_DIR/aura_olga.onnx"

# Copy config for Piper inference
cp "$DATASET_DIR/config.json" "$OUTPUT_DIR/aura_olga.onnx.json"

echo ""
echo "============================================"
echo "Training complete!"
echo "============================================"
echo ""
echo "ONNX model: $OUTPUT_DIR/aura_olga.onnx"
echo "Config:     $OUTPUT_DIR/aura_olga.onnx.json"
echo ""
echo "To deploy to Jetson:"
echo "  scp $OUTPUT_DIR/aura_olga.onnx ledger@puck2:/home/ledger/Aura4/aura-control/voices/"
echo "  scp $OUTPUT_DIR/aura_olga.onnx.json ledger@puck2:/home/ledger/Aura4/aura-control/voices/"
echo ""
echo "Then update speaker.py to use Piper instead of Kokoro+RVC."

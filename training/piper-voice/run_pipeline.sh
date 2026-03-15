#!/bin/bash
# Full pipeline: Generate sentences → Synthesize via ElevenLabs → Build dataset → Train Piper
#
# Before running:
#   1. Set your API key:  export ELEVENLABS_API_KEY="sk-..."
#   2. Set voice ID:      export ELEVENLABS_VOICE_ID="iy0lEidUIpheWxyur2p8"
#   3. Install deps:      pip install elevenlabs requests librosa soundfile numpy
#   4. Install Piper:     git clone https://github.com/rhasspy/piper.git ~/piper
#                         cd ~/piper/src/python && pip install -e .
#
# Usage:
#   ./run_pipeline.sh          # Run everything
#   ./run_pipeline.sh step2    # Resume from ElevenLabs synthesis
#   ./run_pipeline.sh step3    # Resume from dataset building
#   ./run_pipeline.sh step4    # Resume from Piper training
#
# Estimated timeline:
#   Step 1: ~1 second (generate sentences)
#   Step 2: ~3-5 hours (ElevenLabs synthesis of 10,000 sentences)
#   Step 3: ~10 minutes (convert + build dataset)
#   Step 4: ~1-2 days (Piper training on RTX)

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

START_STEP="${1:-step1}"

if [[ "$START_STEP" == "step1" ]] || [[ "$START_STEP" == "all" ]]; then
    echo ""
    echo "========== STEP 1: Generate sentences =========="
    python3 generate_sentences.py --count 10000 --output sentences.txt
fi

if [[ "$START_STEP" == "step1" ]] || [[ "$START_STEP" == "step2" ]] || [[ "$START_STEP" == "all" ]]; then
    echo ""
    echo "========== STEP 2: Synthesize via ElevenLabs =========="
    if [ -z "$ELEVENLABS_API_KEY" ]; then
        echo "ERROR: Set ELEVENLABS_API_KEY first."
        echo "  export ELEVENLABS_API_KEY='your-key-here'"
        exit 1
    fi
    python3 synthesize_elevenlabs.py --input sentences.txt --output-dir dataset --resume
fi

if [[ "$START_STEP" =~ ^(step1|step2|step3|all)$ ]]; then
    echo ""
    echo "========== STEP 3: Build Piper dataset =========="
    python3 build_piper_dataset.py --input-dir dataset --output-dir piper_dataset
fi

if [[ "$START_STEP" =~ ^(step1|step2|step3|step4|all)$ ]]; then
    echo ""
    echo "========== STEP 4: Train Piper =========="
    ./train_piper.sh
fi

echo ""
echo "Pipeline complete! Deploy with:"
echo "  scp output/aura_olga.onnx ledger@puck2:/home/ledger/Aura4/aura-control/voices/"
echo "  scp output/aura_olga.onnx.json ledger@puck2:/home/ledger/Aura4/aura-control/voices/"

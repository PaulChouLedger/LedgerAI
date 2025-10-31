#!/bin/bash
# Quick verification and build script

MODEL_DIR="$HOME/LedgerAI/llm-container/models/Llama/Llama-3.2-1B-Instruct"

echo "🔍 Verifying model download..."
echo ""

if [ ! -d "$MODEL_DIR" ]; then
    echo "❌ Model directory not found: $MODEL_DIR"
    exit 1
fi

echo "✅ Model directory exists"
echo ""
echo "📁 Checking required files..."

required_files=("config.json" "tokenizer.json" "tokenizer_config.json")
missing_files=()

for file in "${required_files[@]}"; do
    if [ -f "$MODEL_DIR/$file" ]; then
        echo "  ✅ $file"
    else
        echo "  ❌ $file (missing)"
        missing_files+=("$file")
    fi
done

# Check for model files (safetensors or pytorch)
if ls "$MODEL_DIR"/*.safetensors 1> /dev/null 2>&1 || ls "$MODEL_DIR"/pytorch_model*.bin 1> /dev/null 2>&1; then
    echo "  ✅ Model weights found"
else
    echo "  ⚠️  Model weights not found (checking...)"
    ls -lh "$MODEL_DIR"/*.safetensors "$MODEL_DIR"/pytorch_model*.bin 2>/dev/null | head -3
fi

echo ""

if [ ${#missing_files[@]} -eq 0 ]; then
    echo "✅ Model looks complete!"
    echo ""
    echo "🚀 Ready to build engine. Run:"
    echo ""
    echo "cd ~/LedgerAI/llm-container"
    echo "docker run --rm -it --gpus all \\"
    echo "  -v \$(pwd)/models:/models \\"
    echo "  -v \$(pwd)/scripts:/scripts \\"
    echo "  dustynv/tensorrt_llm:0.12-r36.4.0 \\"
    echo "  bash /scripts/build_tensorrt_engine.sh llama-3.2-1b /models/Llama/Llama-3.2-1B-Instruct"
    echo ""
else
    echo "❌ Model is incomplete. Missing files:"
    for file in "${missing_files[@]}"; do
        echo "  - $file"
    done
    echo ""
    echo "Please complete the download first."
fi


#!/bin/bash
# Farsight Server Setup — installs dependencies and downloads the model.
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MODEL_DIR="$SCRIPT_DIR/models"

echo "[farsight] Installing Python dependencies..."
pip3 install flask huggingface-hub

echo "[farsight] Installing llama-cpp-python with CUDA support..."
CMAKE_ARGS="-DGGML_CUDA=on" pip3 install llama-cpp-python --force-reinstall --no-cache-dir

echo "[farsight] Downloading Qwen2.5-7B-Instruct Q4_K_M..."
mkdir -p "$MODEL_DIR"
huggingface-cli download bartowski/Qwen2.5-7B-Instruct-GGUF \
    Qwen2.5-7B-Instruct-Q4_K_M.gguf \
    --local-dir "$MODEL_DIR"

echo ""
echo "[farsight] Setup complete. Start with:"
echo "  python3 $SCRIPT_DIR/server.py"

#!/bin/bash
# Native Medical LLM server — runs llm-medical-container/container_rest.py directly on Jetson
# Replaces Docker llm-medical container for GPU performance
set -e

# Auto-detect repo root (works whether cloned as LedgerAI or Aura4)
LEDGER_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$LEDGER_DIR/llm-medical-container"

# Create symlinks if missing (container_rest.py expects /shared and /models)
[ -L /shared ] || sudo ln -sf "$LEDGER_DIR/shared" /shared
[ -L /models ] || sudo ln -sf "$LEDGER_DIR/llm-container/models" /models
[ -L /app ]    || sudo ln -sf "$LEDGER_DIR/llm-medical-container" /app
mkdir -p "$LEDGER_DIR/data/input"
mkdir -p "$LEDGER_DIR/data/embeddings"
mkdir -p "$LEDGER_DIR/data/memory"

# Set environment
export SIMPLE_MODEL_PATH=/models/Qwen2.5-7B-Instruct-Q4_K_M.gguf
export SIMPLE_CHAT_FORMAT=chatml
export SIMPLE_N_CTX=8192
export LLM_STOP='<|im_end|>'
export SHOW_REASONING_DEBUG=false
export RAG_MODE=CPU
export NVIDIA_VISIBLE_DEVICES=all

# Load API keys from .env
if [ -f "$LEDGER_DIR/.env" ]; then
    set -a
    source "$LEDGER_DIR/.env"
    set +a
fi

echo "[Native Medical LLM] Starting Qwen2.5-7B Q4_K_M on port 11434..."
exec "$HOME/aura-env/bin/python3" container_rest.py

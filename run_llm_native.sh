#!/bin/bash
# Native LLM server — runs container_rest.py directly on Jetson
set -e

# Auto-detect repo root (works whether cloned as LedgerAI or Aura4)
LEDGER_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$LEDGER_DIR/containers/llm"

# Create symlinks if missing (container_rest.py expects /shared and /models)
[ -L /shared ] || sudo ln -sf "$LEDGER_DIR/shared" /shared
[ -L /models ] || sudo ln -sf "$LEDGER_DIR/containers/llm/models" /models
[ -L /app ]    || sudo ln -sf "$LEDGER_DIR/containers/llm" /app
mkdir -p "$LEDGER_DIR/data/input"
mkdir -p "$LEDGER_DIR/data/embeddings"

# Set environment
export BASE_MODEL_PATH=/models/Qwen2.5-7B-Instruct-Q5_K_M.gguf
export COT_MODEL_PATH=/models/Qwen2.5-7B-Instruct-Q5_K_M.gguf
export SIMPLE_MODEL_PATH=/models/Qwen2.5-7B-Instruct-Q5_K_M.gguf
export SIMPLE_CHAT_FORMAT=chatml
export SIMPLE_N_CTX=12288
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

echo "[Native LLM] Starting Qwen2.5-7B Q5_K_M (ctx=12288) on port 11434..."
export PYTHONUNBUFFERED=1
exec "$HOME/aura-env/bin/python3" -u container_rest.py

#!/bin/bash
# Native Memory server — runs container_rest.py directly on Jetson
set -e

# Auto-detect repo root (works whether cloned as LedgerAI or Aura4)
LEDGER_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$LEDGER_DIR/containers/memory"

# Create symlinks if missing (container_rest.py expects /shared, /app, /app/data)
[ -L /shared ] || sudo ln -sf "$LEDGER_DIR/shared" /shared
[ -L /app ]    || sudo ln -sf "$LEDGER_DIR/containers/memory" /app
mkdir -p "$LEDGER_DIR/data/memory"

# Set environment
export PORT=11438
export MEMORY_DIR="$LEDGER_DIR/data/memory"
export WHISPER_SERVICE_URL=http://localhost:5000
export LLM_SERVICE_URL=http://localhost:11434
export AUDIO_DEVICE_NAME=reSpeaker
export NVIDIA_VISIBLE_DEVICES=all
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# Load API keys from .env
if [ -f "$LEDGER_DIR/.env" ]; then
    set -a
    source "$LEDGER_DIR/.env"
    set +a
fi

echo "[Native Memory] Starting memory service on port 11438..."
export PYTHONUNBUFFERED=1
exec "$HOME/aura-env/bin/python3" -u container_rest.py

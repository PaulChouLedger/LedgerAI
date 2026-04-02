#!/bin/bash
# Native Whisper server — runs container_rest.py directly on Jetson
set -e

# Auto-detect repo root (works whether cloned as LedgerAI or Aura4)
LEDGER_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$LEDGER_DIR/containers/whisper"

# Create symlinks if missing (container_rest.py expects /shared and /app)
[ -L /shared ] || sudo ln -sf "$LEDGER_DIR/shared" /shared
[ -L /app ]    || sudo ln -sf "$LEDGER_DIR/containers/whisper" /app

# Whisper model is cached in the user's HuggingFace hub directory.
# Models are cached in the user's HuggingFace hub directory.
# If models exist in /root/.cache (legacy), symlink so container_rest.py finds them.
NATIVE_CACHE="$HOME/.cache/huggingface/hub"
ROOT_CACHE="/root/.cache/huggingface/hub"
mkdir -p "$NATIVE_CACHE"

# Symlink root cache → native cache so container_rest.py finds models regardless of path
if [ -d "$ROOT_CACHE" ] && [ ! -L "$ROOT_CACHE" ]; then
    # Models exist in /root — leave them (whisper code reads from /root/.cache)
    true
elif [ ! -d "$ROOT_CACHE" ]; then
    sudo mkdir -p "$(dirname "$ROOT_CACHE")"
    sudo ln -sf "$NATIVE_CACHE" "$ROOT_CACHE" 2>/dev/null || true
fi

# Set environment
export WHISPER_MODEL="${WHISPER_MODEL:-distil-whisper/distil-large-v3.5-ct2}"
export NVIDIA_VISIBLE_DEVICES=all
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# Load API keys from .env
if [ -f "$LEDGER_DIR/.env" ]; then
    set -a
    source "$LEDGER_DIR/.env"
    set +a
fi

echo "[Native Whisper] Starting ${WHISPER_MODEL} on port 5000..."
export PYTHONUNBUFFERED=1
exec "$HOME/aura-env/bin/python3" -u container_rest.py

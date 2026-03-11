#!/bin/bash
# start_aura.sh — Boot script for Aura v9
# Waits for X display, starts docker containers, launches aura.py

set -e

# Detect X display
for i in $(seq 1 30); do
    SOCK=$(ls /tmp/.X11-unix/ 2>/dev/null | grep "^X" | head -1)
    if [ -n "$SOCK" ]; then
        NUM=$(echo "$SOCK" | sed "s/X//")
        export DISPLAY=":$NUM"
        break
    fi
    sleep 1
done
export DISPLAY="${DISPLAY:-:0}"

# Allow local connections
xhost +local: 2>/dev/null || true

# Hide GNOME panels and cursor on the round display
gsettings set org.gnome.desktop.screensaver lock-enabled false 2>/dev/null || true
gsettings set org.gnome.desktop.lockdown disable-lock-screen true 2>/dev/null || true
gsettings set org.gnome.desktop.screensaver idle-activation-enabled false 2>/dev/null || true
gsettings set org.gnome.desktop.session idle-delay 0 2>/dev/null || true

# Hide cursor (unclutter if available, otherwise move it off-screen)
if command -v unclutter &>/dev/null; then
    unclutter -idle 0.1 -root &
else
    xdotool mousemove 9999 9999 2>/dev/null || true
fi

# Activate virtualenv
source /home/ledger/aura-env/bin/activate

# Prevent HuggingFace from checking for model updates (no internet)
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# PyTorch/OpenMP: prevent worker threads from spin-waiting on CPU when idle.
# Without this, 7 OMP threads burn 100% CPU each after any inference call.
export OMP_WAIT_POLICY=PASSIVE
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export GOTO_NUM_THREADS=2

# Farsight: remote GPU server for Perpetual deep thinking (via Tailscale)
export AURA_FARSIGHT_URL="http://100.76.191.92:11435"

# Ensure docker containers are running (only generic LLM — medical shares port)
cd /home/ledger/Aura4/setup
docker compose up -d whisper llm-generic memory 2>/dev/null \
    || docker-compose up -d whisper llm-generic memory 2>/dev/null || true

# Small delay for containers to bind ports
sleep 3

# Launch Aura
cd /home/ledger/Aura4/aura-control/v9
exec python3 -u aura.py

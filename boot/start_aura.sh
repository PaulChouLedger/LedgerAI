#!/bin/bash
# start_aura.sh — Boot script for Aura v9 (fully native, no containers)
# Waits for X display, starts all services natively, launches aura.py
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

# Set XAUTHORITY (GDM puts it in a non-standard location on Jetson)
if [ -f "/run/user/$(id -u)/gdm/Xauthority" ]; then
    export XAUTHORITY="/run/user/$(id -u)/gdm/Xauthority"
fi

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

# ── Maximize XVF3800 mic capture volume ────────────────────────
# The ALSA default is 52/60 (-8dB). Max it to 60/60 (0dB) for best SNR.
amixer -c 1 cset numid=10 60,60 2>/dev/null || true

# ── Kill ALL stale processes from prior boots ──────────────────
# Without this, zombie python3/aplay/ffmpeg processes hold the mic
# and audio devices, causing "Device or resource busy" on restart.
# Use --older-than to avoid killing ourselves (we just started).
# SIGKILL (-9) for container_rest: SIGTERM leaves CUDA contexts alive,
# blocking GPU VRAM for seconds and causing the new LLM to fail model load.
MY_PID=$$
pkill -f "python3.*aura\.py" --older-than 5s 2>/dev/null || true
pkill -9 -f "python3.*container_rest\.py" 2>/dev/null || true
pkill -f "aplay.*plughw" 2>/dev/null || true
pkill -f "ffmpeg.*plughw" 2>/dev/null || true
pkill -f "unclutter" 2>/dev/null || true
# Release any lingering ALSA device handles (all cards, not just card 1)
for dev in /dev/snd/pcmC*c /dev/snd/pcmC*p; do
    [ -e "$dev" ] && fuser -k "$dev" 2>/dev/null || true
done
sleep 2

# Wait for GPU VRAM to be released by killed CUDA processes.
# CUDA driver cleanup after SIGKILL can take several seconds on Jetson
# (unified memory, no discrete VRAM to free instantly).
for i in $(seq 1 10); do
    # Check if any python3 container_rest processes still linger
    if ! pgrep -f "python3.*container_rest" >/dev/null 2>&1; then
        break
    fi
    echo "[start_aura] Waiting for stale GPU processes to exit ($i/10)..."
    pkill -9 -f "python3.*container_rest\.py" 2>/dev/null || true
    sleep 1
done

# ── Start native LLM FIRST — needs GPU memory before other services ──
cd /home/ledger/Aura4
nohup bash run_llm_native.sh > /tmp/aura-llm.log 2>&1 &

# Wait for LLM to load model and claim GPU memory (7B Q4 ≈ 4.4GB on Jetson)
echo "[start_aura] Waiting for LLM to load..."
for i in $(seq 1 60); do
    if curl -sf http://localhost:11434/health >/dev/null 2>&1; then
        echo "[start_aura] LLM is up after ${i}s"
        break
    fi
    sleep 1
done

# ── Start native Whisper ──
echo "[start_aura] Starting native Whisper..."
nohup bash run_whisper_native.sh > /tmp/aura-whisper.log 2>&1 &

# ── Start native Memory service ──
echo "[start_aura] Starting native Memory service..."
nohup bash run_memory_native.sh > /tmp/aura-memory.log 2>&1 &

# Wait for Whisper and Memory to come up
echo "[start_aura] Waiting for Whisper + Memory..."
for i in $(seq 1 30); do
    WHISPER_UP=false
    MEMORY_UP=false
    curl -sf http://localhost:5000/health >/dev/null 2>&1 && WHISPER_UP=true
    curl -sf http://localhost:11438/health >/dev/null 2>&1 && MEMORY_UP=true
    if $WHISPER_UP && $MEMORY_UP; then
        echo "[start_aura] All services up after ${i}s"
        break
    fi
    sleep 1
done

# Ensure data dir is writable (sudo git reset can make files root-owned)
chown -R ledger:ledger /home/ledger/Aura4/data 2>/dev/null || true

# Launch Aura
cd /home/ledger/Aura4/aura
exec python3 -u aura.py

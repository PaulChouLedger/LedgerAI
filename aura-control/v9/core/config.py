"""
core.config -- Constants, paths, palette, hardware config.

Single source of truth for anything that was previously scattered across
module-level globals in carbon_demo.py, listener.py, speaker.py, etc.
Edit here; everything else reads from this module.
"""

from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
WORKSPACE_ROOT = Path(__file__).resolve().parents[3]      # /home/ledger/LedgerAI
AURA_ROOT      = Path(__file__).resolve().parents[1]      # aura-control/v9
DATA_DIR       = WORKSPACE_ROOT / "data"
ASSETS_DIR     = WORKSPACE_ROOT / "assets"
VOICES_DIR     = AURA_ROOT.parent / "voices" / "wav"       # aura-control/voices/wav
SETTINGS_FILE  = DATA_DIR / "app_settings.json"
TOKEN_USAGE_FILE = DATA_DIR / "token_usage.json"

# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------
DISPLAY = os.environ.get("DISPLAY", ":0")
SCREEN_W = 1080
SCREEN_H = 1080
DISPLAY_DIAM_MM = float(os.environ.get("AURA_DISPLAY_DIAM_MM", "70.0"))

# ---------------------------------------------------------------------------
# Audio / microphone
# ---------------------------------------------------------------------------
PULSE_SOURCE = os.environ.get(
    "AURA_PULSE_SOURCE",
    "alsa_input.usb-Seeed_Studio_reSpeaker_XVF3800_4-Mic_Array_"
    "101991441254700003-00.analog-stereo",
)
SAMPLE_RATE      = 16000
FRAME_MS         = 30
SAMPLE_WIDTH      = 2       # bytes (s16le)
VAD_MODE         = 2
MAX_RECORD_S     = 10
END_SILENCE_MS   = 650
MAX_SILENCE_MS   = 900
IDLE_BACKOFF_S   = 0.01     # prevents busy-loop when audio stalls

# Derived
BYTES_PER_FRAME  = int(SAMPLE_RATE * (FRAME_MS / 1000.0) * SAMPLE_WIDTH)
MAX_FRAMES       = int((MAX_RECORD_S * 1000) / FRAME_MS)

# ---------------------------------------------------------------------------
# Whisper
# ---------------------------------------------------------------------------
WHISPER_MODEL_SIZE = os.environ.get("AURA_WHISPER_MODEL", "base")
WHISPER_DEVICE     = os.environ.get("AURA_WHISPER_DEVICE", "cuda").strip().lower()
WHISPER_COMPUTE    = os.environ.get("AURA_WHISPER_COMPUTE", "float16").strip().lower()
WHISPER_BEAM       = int(os.environ.get("AURA_WHISPER_BEAM", "2"))
WHISPER_BEST_OF    = int(os.environ.get("AURA_WHISPER_BEST_OF", "2"))

# ---------------------------------------------------------------------------
# TTS
# ---------------------------------------------------------------------------
TTS_DEVICE   = os.environ.get("AURA_TTS_DEVICE", "cuda")
TTS_OUT_WAV  = Path("/tmp/aura_tts.wav")
TTS_VOLUME   = float(os.environ.get("TTS_VOLUME", "0.90"))   # 90% default
TTS_STEPS    = int(os.environ.get("AURA_TTS_STEPS", "50"))    # 50 ≈ 25s on Jetson

# ---------------------------------------------------------------------------
# Container endpoints
# ---------------------------------------------------------------------------
WHISPER_URL  = os.environ.get("AURA_WHISPER_URL",  "http://localhost:5000")
LLM_URL      = os.environ.get("AURA_LLM_URL",      "http://localhost:11434")
MEMORY_URL   = os.environ.get("AURA_MEMORY_URL",   "http://localhost:11438")
RAG_URL      = os.environ.get("AURA_RAG_URL",      "http://localhost:11435")

# ---------------------------------------------------------------------------
# Wake word
# ---------------------------------------------------------------------------
WAKE_WORDS = [
    "aura", "ora", "or a", "oura", "laura",
    "all right", "alright", "we're", "were",
]

# ---------------------------------------------------------------------------
# Palette  (Qt-free: stored as (R, G, B, A) tuples)
# GUI code converts these to QColor at import time.
# ---------------------------------------------------------------------------
PAL_BLUE       = (35, 165, 255, 255)
PAL_GOLD       = (208, 178, 112, 255)
PAL_GRAPHITE   = (22, 22, 24, 255)
PAL_MUTE_GREEN = (90, 200, 130, 255)
PAL_MUTE_RED   = (220, 75, 65, 255)

# ---------------------------------------------------------------------------
# Dock defaults  (Topics Center is always available; these are the initial
# complications shown on the perimeter ring before the user customizes)
# ---------------------------------------------------------------------------
DEFAULT_DOCK = ["Topics Center", "Settings", "Mute", "Aura Concierge"]
MAX_DOCK_SLOTS = 4      # 4 major complications + 4 domain glyphs interspersed

# ---------------------------------------------------------------------------
# Boot / Falcon animation
# ---------------------------------------------------------------------------
FIXED_ROTATION_DEG = float(os.environ.get("AURA_ROTATION_DEG", "-130.0"))
BOOT_MUSIC_PATH    = ASSETS_DIR / "AuraIntro.mp3"
BOOT_PROMPTS_DIR   = ASSETS_DIR / "boot_prompts"
VOICE_PROFILES_DIR = DATA_DIR / "voice_profiles"

# Timing
BOOT_TOTAL_S       = float(os.environ.get("AURA_BOOT_TOTAL_S", "120.0"))
BOOT_HOLD_S        = float(os.environ.get("AURA_BOOT_HOLD_S", "2.5"))
BOOT_FADE_S        = float(os.environ.get("AURA_BOOT_FADE_S", "3.0"))
BOOT_SERVICE_TIMEOUT_S = float(os.environ.get("AURA_BOOT_SVC_TIMEOUT", "60.0"))
BOOT_MIC_TIMEOUT_S = float(os.environ.get("AURA_BOOT_MIC_TIMEOUT", "45.0"))

# Capture
BOOT_CAPTURE_MAX_S      = 5.0      # max seconds for a single voice capture
BOOT_CAPTURE_SILENCE_S  = 0.8      # silence before ending capture
BOOT_CAPTURE_TIMEOUT_S  = 8.0      # give up waiting for speech

# Speaker embedding (resemblyzer)
EMBEDDING_DIM             = 256
EMBEDDING_MATCH_THRESHOLD = 0.75

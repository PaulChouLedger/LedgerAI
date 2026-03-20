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
WORKSPACE_ROOT = Path(__file__).resolve().parents[2]      # repo root
AURA_ROOT      = Path(__file__).resolve().parents[1]      # aura/
DATA_DIR       = WORKSPACE_ROOT / "data"
ASSETS_DIR     = WORKSPACE_ROOT / "assets"
VOICES_DIR     = WORKSPACE_ROOT / "voices"
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
    "101991441253700168-00.analog-stereo",
)
# ALSA output device — bypass PipeWire/PulseAudio, play directly to hardware.
# "plughw:" handles sample-rate and channel conversion automatically.
ALSA_PLAYBACK_DEVICE = os.environ.get(
    "AURA_ALSA_DEVICE", "plughw:CARD=UACDemoV10,DEV=0"
)
SAMPLE_RATE      = 16000
FRAME_MS         = 30
SAMPLE_WIDTH      = 2       # bytes (s16le)
VAD_MODE         = 2
MAX_RECORD_S     = 10
END_SILENCE_MS   = 650
MAX_SILENCE_MS   = 900
IDLE_BACKOFF_S   = 0.01     # prevents busy-loop when audio stalls
MIC_GAIN         = float(os.environ.get("AURA_MIC_GAIN", "1.5"))  # digital gain on captured audio (XVF3800 has built-in AGC)

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
TTS_VOLUME   = float(os.environ.get("TTS_VOLUME", "0.85"))   # ALSA volume (0.85 = 85%)
TTS_GAIN     = float(os.environ.get("TTS_GAIN", "3.6"))     # digital gain applied to audio before playback
TTS_STEPS    = int(os.environ.get("AURA_TTS_STEPS", "50"))    # 50 ≈ 25s on Jetson

# XTTS v2 voice cloning (legacy — kept for reference, replaced by Piper)
XTTS_REFS_DIR      = WORKSPACE_ROOT / "voices" / "xtts_refs"
XTTS_SAMPLE_RATE   = 24000
XTTS_TEMPERATURE   = float(os.environ.get("AURA_XTTS_TEMP", "0.65"))
XTTS_REP_PENALTY   = float(os.environ.get("AURA_XTTS_REP_PENALTY", "10.0"))
XTTS_LENGTH_PENALTY = float(os.environ.get("AURA_XTTS_LEN_PENALTY", "1.0"))

# Piper TTS (VITS-based, ~63MB ONNX, <1s synthesis on CPU)
PIPER_MODEL_PATH   = WORKSPACE_ROOT / "voices" / "aura_olga_2249.onnx"
PIPER_SAMPLE_RATE  = 22050
PIPER_LENGTH_SCALE = float(os.environ.get("AURA_PIPER_LENGTH_SCALE", "1.15"))
PIPER_NOISE_SCALE  = float(os.environ.get("AURA_PIPER_NOISE_SCALE", "0.667"))
PIPER_NOISE_W      = float(os.environ.get("AURA_PIPER_NOISE_W", "0.8"))

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
# Color schemes  (runtime-switchable watch face themes)
# ---------------------------------------------------------------------------
COLOR_SCHEMES = {
    "rafael": {
        # Background
        "bg_base":       (16, 30, 54),
        "bg_fill":       (10, 18, 38),        # window.py fallback fill
        "bg_style":      "lacquer",            # lacquer (blue) vs radial (red)
        "bg_emboss":     (12, 24, 42),
        "bg_thread":     (235, 238, 242),
        "bg_thread_strong": (245, 248, 252),
        "stylesheet_bg": "black",
        "fade_overlay":  (0, 0, 0),
        "rotation_fill": None,                 # None = no pre-rotation fill needed
        # Stars
        "star_white":    (210, 225, 248),
        "star_gold":     (180, 210, 245),
        # Nebula
        "nebula_core":   (60, 130, 220),
        "nebula_mid":    (35, 80, 160),
        "nebula_deep":   (15, 35, 90),
        "nebula_edge":   (10, 20, 55),
        "nebula_bright": (100, 180, 255),
        # Mist
        "mist_color":    (150, 185, 225),
        # Ticks / center ring
        "tick_color":    (195, 215, 240),
        "ring_main":     (145, 175, 215),
        "ring_hi":       (200, 220, 245),
        # Bezel
        "bezel_hi":      (200, 215, 235),
        # Ring palette
        "ring_palette":  "blue",
        # Speaking highlights
        "speak_hi":      (190, 218, 248),
    },
    "ferrari": {
        # Background
        "bg_base":       (54, 10, 16),
        "bg_fill":       (88, 14, 30),
        "bg_style":      "radial",
        "bg_emboss":     (255, 210, 210),
        "bg_thread":     (255, 190, 190),
        "bg_thread_strong": (255, 190, 190),
        "stylesheet_bg": "#580e1e",
        "fade_overlay":  (115, 28, 48),
        "rotation_fill": (88, 14, 30),         # pre-rotation fill to prevent black corners
        # Stars
        "star_white":    (248, 210, 210),
        "star_gold":     (245, 180, 180),
        # Nebula
        "nebula_core":   (220, 60, 60),
        "nebula_mid":    (160, 35, 35),
        "nebula_deep":   (90, 15, 15),
        "nebula_edge":   (55, 10, 10),
        "nebula_bright": (255, 100, 100),
        # Mist
        "mist_color":    (225, 150, 150),
        # Ticks / center ring  (gold/champagne — classic Ferrari pairing)
        "tick_color":    (225, 195, 130),
        "ring_main":     (200, 170, 100),
        "ring_hi":       (240, 215, 150),
        # Bezel
        "bezel_hi":      (230, 205, 140),
        # Ring palette
        "ring_palette":  "red",
        # Speaking highlights
        "speak_hi":      (248, 200, 190),
    },
}

DEFAULT_COLOR_SCHEME = os.environ.get("AURA_COLOR_SCHEME", "rafael")

# ---------------------------------------------------------------------------
# Dock defaults  (Topics Center is always available; these are the initial
# complications shown on the perimeter ring before the user customizes)
# ---------------------------------------------------------------------------
DEFAULT_DOCK = ["Topics Center", "Settings", "Mute"]
MAX_DOCK_SLOTS = 3      # 3 complications + 3 domain glyphs interspersed

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
BOOT_CAPTURE_SILENCE_S  = 1.5      # silence before ending capture
BOOT_CAPTURE_TIMEOUT_S  = 8.0      # give up waiting for speech

# Speaker embedding (resemblyzer)
EMBEDDING_DIM             = 256
EMBEDDING_MATCH_THRESHOLD = 0.45

# ---------------------------------------------------------------------------
# Aura Perpetual (background rumination engine)
# ---------------------------------------------------------------------------
BRIEFINGS_DIR              = DATA_DIR / "briefings"
PERPETUAL_IDLE_THRESHOLD_S = float(os.environ.get("AURA_PERPETUAL_IDLE_S", "120.0"))
PERPETUAL_MAX_ITERATIONS   = int(os.environ.get("AURA_PERPETUAL_MAX_ITER", "5"))
PERPETUAL_CONVERGENCE      = float(os.environ.get("AURA_PERPETUAL_CONVERGENCE", "0.92"))
PERPETUAL_BRIEFING_COOLDOWN_S = 86400   # 24 hours between briefings
PERPETUAL_QUESTION_COOLDOWN_S = 3600   # 1 hour between proactive questions
PERPETUAL_7B_MODEL_PATH    = os.environ.get(
    "AURA_PERPETUAL_7B_MODEL",
    "/models/extra/Qwen2.5-7B-Instruct-Q4_K_M.gguf",
)

# Farsight: remote GPU server for deep thinking (enterprise mode)
# If set, Perpetual farms out LLM calls to this endpoint instead of local model swap.
# If unset or empty, falls back to local Puck inference (swap to 7B on CPU).
FARSIGHT_URL = os.environ.get("AURA_FARSIGHT_URL", "http://100.76.191.92:11435")  # Farsight RTX PRO 6000 Blackwell (72B Qwen)

# Farsight TTS: pre-synthesize briefing audio on remote GPU for higher quality
FARSIGHT_TTS_STEPS = int(os.environ.get("AURA_FARSIGHT_TTS_STEPS", "200"))  # diffusion steps

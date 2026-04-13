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
END_SILENCE_MS   = 400
MAX_SILENCE_MS   = 600
IDLE_BACKOFF_S   = 0.01     # prevents busy-loop when audio stalls
MIC_GAIN         = float(os.environ.get("AURA_MIC_GAIN", "3.0"))  # digital gain on captured audio (XVF3800 AGC handles most amplification)

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
TTS_VOLUME   = float(os.environ.get("TTS_VOLUME", "0.75"))   # ALSA volume (0.75 = 75%)
TTS_GAIN     = float(os.environ.get("TTS_GAIN", "4.7"))     # digital gain applied to audio before playback
TTS_STEPS    = int(os.environ.get("AURA_TTS_STEPS", "50"))    # 50 ≈ 25s on Jetson

# Piper TTS (VITS-based, ~63MB ONNX, <1s synthesis on CPU)
PIPER_MODEL_PATH   = WORKSPACE_ROOT / "voices" / "aura_olga_27499.onnx"
PIPER_SAMPLE_RATE  = 22050
PIPER_LENGTH_SCALE = float(os.environ.get("AURA_PIPER_LENGTH_SCALE", "1.05"))
PIPER_NOISE_SCALE  = float(os.environ.get("AURA_PIPER_NOISE_SCALE", "0.85"))
PIPER_NOISE_W      = float(os.environ.get("AURA_PIPER_NOISE_W", "0.9"))

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
    "rose_quartz": {
        # Background — warm rose-pink, visible not black
        "bg_base":       (62, 38, 52),
        "bg_fill":       (72, 42, 60),
        "bg_style":      "radial",
        "bg_emboss":     (80, 50, 68),
        "bg_thread":     (230, 210, 225),
        "bg_thread_strong": (240, 220, 235),
        "stylesheet_bg": "#482a3c",
        "fade_overlay":  (72, 42, 60),
        "rotation_fill": (72, 42, 60),
        # Stars — warm pink-white
        "star_white":    (240, 215, 235),
        "star_gold":     (225, 195, 220),
        # Nebula — rose quartz pink, bright and pronounced
        "nebula_core":   (200, 140, 180),
        "nebula_mid":    (160, 95, 135),
        "nebula_deep":   (100, 55, 82),
        "nebula_edge":   (72, 42, 60),
        "nebula_bright": (225, 180, 210),
        # Mist
        "mist_color":    (210, 180, 200),
        # Ticks / center ring — rose quartz pink (200,200,230 tinted warmer)
        "tick_color":    (220, 195, 225),
        "ring_main":     (200, 170, 205),
        "ring_hi":       (235, 215, 240),
        # Bezel
        "bezel_hi":      (225, 205, 230),
        # Ring palette
        "ring_palette":  "pink",
        # Speaking highlights
        "speak_hi":      (230, 205, 230),
    },
    "steel": {
        # Flagship enclosure: true gunmetal gray, near-neutral
        # Background — deep neutral charcoal
        "bg_base":       (20, 20, 22),
        "bg_fill":       (14, 14, 15),
        "bg_style":      "lacquer",
        "bg_emboss":     (26, 27, 28),
        "bg_thread":     (188, 190, 194),       # pure silver threads
        "bg_thread_strong": (210, 212, 215),
        "stylesheet_bg": "black",
        "fade_overlay":  (0, 0, 0),
        "rotation_fill": None,
        # Stars — warm silver
        "star_white":    (215, 215, 218),
        "star_gold":     (182, 182, 186),
        # Nebula — true gray, barely any blue
        "nebula_core":   (82, 84, 90),
        "nebula_mid":    (56, 57, 62),
        "nebula_deep":   (34, 34, 37),
        "nebula_edge":   (20, 20, 22),
        "nebula_bright": (132, 134, 142),
        # Mist — neutral gray
        "mist_color":    (150, 152, 158),
        # Ticks / center ring — bright silver
        "tick_color":    (192, 194, 200),
        "ring_main":     (140, 142, 150),
        "ring_hi":       (210, 212, 218),
        # Bezel — polished silver
        "bezel_hi":      (198, 200, 206),
        # Ring palette
        "ring_palette":  "steel",
        # Speaking highlights — silver flash
        "speak_hi":      (196, 198, 206),
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

# ---------------------------------------------------------------------------
# Presence (proactive voice initiation)
# ---------------------------------------------------------------------------
PRESENCE_RMS_QUIET          = 0.003
PRESENCE_RMS_ACTIVE         = 0.008
PRESENCE_WINDOW_SIZE        = 12       # 12 samples × 5s = 60s sliding window
PRESENCE_MIN_SILENCE_S      = 1800     # 30 min quiet before greeting triggers
PRESENCE_GREETING_COOLDOWN  = 7200     # 2 hours between greetings
MORNING_BRIEFING_HOUR_MIN   = 6
MORNING_BRIEFING_HOUR_MAX   = 11
IDLE_COMMENT_COOLDOWN_S     = 2700     # 45 min between idle comments
IDLE_COMMENT_MIN_SILENCE_S  = 900      # 15 min silence before idle comment
IDLE_COMMENT_MAX_SILENCE_S  = 1800     # 30 min max (user may have left)
IDLE_SESSION_WINDOW_S       = 7200     # 2 hours — user spoke within this window
TELEGRAM_ALERT_COOLDOWN_S   = 1800     # 30 min between alerts
TELEGRAM_ALERT_MAX_HOUR     = 2        # max 2 alerts per hour
PROACTIVE_DAILY_BUDGET      = 4        # max unprompted utterances per day

# ---------------------------------------------------------------------------
# Household engagement
# ---------------------------------------------------------------------------
HOUSEHOLD_PROFILES_FILE         = DATA_DIR / "household_profiles.json"
HOUSEHOLD_IDENTIFY_COOLDOWN     = 30      # seconds between re-identifying same voice
HOUSEHOLD_UNKNOWN_GREET_COOLDOWN = 300    # 5 min between greeting unknown visitors
HOUSEHOLD_CONVERSATION_MODE_S   = 30      # seconds to bypass wake word after proactive speech
HOUSEHOLD_GREETING_COOLDOWN     = 1800    # 30 min before re-greeting same known user

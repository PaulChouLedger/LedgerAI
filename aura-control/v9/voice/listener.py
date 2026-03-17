"""
voice.listener -- Mic capture → VAD → Whisper transcription.

Runs on a daemon thread.  Communicates via bus events:
    emits:  "transcript.ready"  text=str
            "listener.vad"      active=bool
            "listener.state"    state=str  ("waiting"|"listening"|"transcribing")
    reads:  "tts.playing"       (echo gate — skip while Aura is speaking)
            "mute.toggled"      (skip while muted)

Extracted from core/listener.py.  Zero Qt imports.
"""

from __future__ import annotations

import io
import os
import re
import time
import threading
import subprocess
from typing import Optional

import numpy as np
import requests
import sounddevice as sd
import soundfile as sf
import torch
from scipy.fft import rfft, rfftfreq

from services.diaglog import heard as _diag_heard, rejected as _diag_rejected

from core.bus import bus
from core.config import (
    SAMPLE_RATE, WHISPER_URL,
)
from core.state import state
from services.memlog import memlog
from voice.wake import heard_wake, should_respond, strip_wake

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

FRAME_SIZE          = int(SAMPLE_RATE * 0.032)      # ~512 samples, 32ms
SILENCE_TIMEOUT     = 0.2                           # seconds
VAD_START_THRESH    = 0.25
VAD_SILENCE_THRESH  = 0.10
MIN_AUDIO_SAMPLES   = 2000                          # ~125ms

DEVICE_NAME         = "reSpeaker"
MIC_CHANNEL         = 0                             # XVF3800 channel 0 = beamformed

# Advanced filter thresholds (calibrated for XVF3800 + beamforming)
SPEECH_ZCR_MAX        = 0.50
SPEECH_FLATNESS_MAX   = 0.75
SPEECH_CENTROID_MIN   = 200.0
SPEECH_CENTROID_MAX   = 5000.0
SPEECH_BAND_MIN       = 0.03
SPEECH_DURATION_MIN   = 0.3
SPEECH_HIGH_FREQ_MAX  = 0.25
SPEECH_RMS_MIN        = 0.0005
SPEECH_RMS_MAX        = 0.90
SPEECH_PEAK_MIN       = 0.0008

CONTEXT_DEPTH = 6

# Common Whisper hallucinations on silence/noise — reject these outright
WHISPER_HALLUCINATIONS = {
    "you", "bye", "bye.", "thank you", "thank you.", "thanks.",
    "thanks", "yeah", "yes", "no", "okay", "ok", "hmm", "hm",
    "oh", "ah", "uh", "um", "so", "the", "a", "i", "it",
    "the end", "the end.", "thanks for watching", "thanks for watching.",
    "thank you for watching", "thank you for watching.",
    "subscribe", "like and subscribe",
    "you're welcome", "you're welcome.",
}

# ---------------------------------------------------------------------------
# Silero VAD (loaded once, used by all Listener instances)
# ---------------------------------------------------------------------------

_vad_model = None
_vad_lock = threading.Lock()


def _get_vad():
    global _vad_model
    if _vad_model is None:
        with _vad_lock:
            if _vad_model is None:
                _vad_model, _ = torch.hub.load(
                    "snakers4/silero-vad", "silero_vad", onnx=False
                )
    return _vad_model


# ---------------------------------------------------------------------------
# Audio feature analysis
# ---------------------------------------------------------------------------

def calculate_audio_features(chunk: np.ndarray, sr: int = SAMPLE_RATE) -> dict:
    """Compute spectral features for speech/noise discrimination."""
    audio = chunk.astype(np.float32)
    if audio.max() > 1.5:
        audio = audio / 32768.0

    rms = float(np.sqrt(np.mean(audio ** 2)))
    peak = float(np.max(np.abs(audio)))

    # Zero-crossing rate
    signs = np.sign(audio)
    zcr = float(np.mean(np.abs(np.diff(signs)) > 0))

    # Spectral analysis
    N = len(audio)
    if N < 64:
        return {"rms": rms, "peak": peak, "zcr": zcr,
                "centroid": 0, "flatness": 1.0, "band_energy": 0, "high_freq": 0}

    spec = np.abs(rfft(audio))
    freqs = rfftfreq(N, 1.0 / sr)
    total = np.sum(spec) + 1e-12

    centroid = float(np.sum(freqs * spec) / total)

    geo_mean = np.exp(np.mean(np.log(spec + 1e-12)))
    ari_mean = np.mean(spec)
    flatness = float(geo_mean / (ari_mean + 1e-12))

    band_mask = (freqs >= 300) & (freqs <= 3400)
    band_energy = float(np.sum(spec[band_mask]) / total) if np.any(band_mask) else 0.0

    hi_mask = freqs >= 6000
    high_freq = float(np.sum(spec[hi_mask]) / total) if np.any(hi_mask) else 0.0

    return {
        "rms": rms, "peak": peak, "zcr": zcr,
        "centroid": centroid, "flatness": flatness,
        "band_energy": band_energy, "high_freq": high_freq,
    }


def is_likely_speech(features: dict, duration: Optional[float] = None) -> tuple:
    """Multi-threshold check.  Returns (ok, reason)."""
    if features["rms"] < SPEECH_RMS_MIN:
        return False, "rms_too_low"
    if features["rms"] > SPEECH_RMS_MAX:
        return False, "rms_clipping"
    if features["peak"] < SPEECH_PEAK_MIN:
        return False, "peak_too_low"
    if features["zcr"] > SPEECH_ZCR_MAX:
        return False, "zcr_noise"
    if features["centroid"] < SPEECH_CENTROID_MIN:
        return False, "centroid_low"
    if features["centroid"] > SPEECH_CENTROID_MAX:
        return False, "centroid_high"
    if features["flatness"] > SPEECH_FLATNESS_MAX:
        return False, "flat_noise"
    if features["band_energy"] < SPEECH_BAND_MIN:
        return False, "low_band"
    if features["high_freq"] > SPEECH_HIGH_FREQ_MAX:
        return False, "hiss"
    if duration is not None and duration < SPEECH_DURATION_MIN:
        return False, "too_short"
    return True, "ok"


# ---------------------------------------------------------------------------
# Device detection
# ---------------------------------------------------------------------------

def find_device_index(max_retries: int = 10) -> Optional[int]:
    """Find XVF3800 microphone index with retry logic."""
    for attempt in range(max_retries):
        try:
            devices = sd.query_devices()
            for idx, dev in enumerate(devices):
                if DEVICE_NAME in (dev.get("name") or ""):
                    print(f"[listener] Found {DEVICE_NAME} at index {idx}")
                    return idx
        except Exception as e:
            print(f"[listener] Device query error (attempt {attempt+1}): {e}")
        delay = min(1.0 * (2 ** attempt), 5.0)
        time.sleep(delay)
    print(f"[listener] {DEVICE_NAME} not found after {max_retries} retries")
    return None


def _find_alsa_card(name_fragment: str = "Array", max_retries: int = 10) -> Optional[str]:
    """Discover the ALSA hw: device for the reSpeaker by scanning /proc/asound/cards.

    Returns e.g. 'hw:1,0' — the card number may change across reboots
    depending on USB enumeration order, so never hardcode it.
    """
    for attempt in range(max_retries):
        try:
            with open("/proc/asound/cards") as f:
                for line in f:
                    # Lines look like: " 1 [Array          ]: USB-Audio - ..."
                    line = line.strip()
                    if not line or not line[0].isdigit():
                        continue
                    parts = line.split("[", 1)
                    if len(parts) < 2:
                        continue
                    card_num = parts[0].strip()
                    if name_fragment in line:
                        dev = f"hw:{card_num},0"
                        print(f"[listener] Found mic ALSA card: {dev} ({line.strip()})")
                        return dev
        except Exception as e:
            print(f"[listener] ALSA card scan error (attempt {attempt+1}): {e}")
        delay = min(1.0 * (2 ** attempt), 5.0)
        time.sleep(delay)
    print(f"[listener] ALSA card '{name_fragment}' not found after {max_retries} retries")
    return None


# ---------------------------------------------------------------------------
# Whisper HTTP
# ---------------------------------------------------------------------------

def transcribe(audio: np.ndarray, sr: int = SAMPLE_RATE) -> str:
    """POST audio to Whisper container, return transcribed text."""
    # Final speech filter
    feats = calculate_audio_features(audio, sr)
    dur = len(audio) / sr
    ok, reason = is_likely_speech(feats, dur)
    if not ok:
        print(f"[listener] Rejected (post-filter): {reason}")
        return ""

    # Encode as WAV
    buf = io.BytesIO()
    sf.write(buf, audio, sr, format="WAV", subtype="PCM_16")
    buf.seek(0)

    try:
        resp = requests.post(
            f"{WHISPER_URL}/transcribe",
            files={"audio": ("audio.wav", buf, "audio/wav")},
            timeout=15,
        )
        if resp.status_code != 200:
            print(f"[listener] Whisper HTTP {resp.status_code}")
            return ""
        text = resp.json().get("text", "").strip()
        return text
    except Exception as e:
        print(f"[listener] Whisper error: {e}")
        return ""


def warmup_whisper():
    """Send 1s silence to prime Whisper JIT."""
    silence = np.zeros(SAMPLE_RATE, dtype=np.float32)
    try:
        transcribe(silence, SAMPLE_RATE)
        print("[listener] Whisper warmed up")
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Listener class
# ---------------------------------------------------------------------------

class Listener:
    """Daemon-thread listener: mic → VAD → Whisper → bus events."""

    # Adaptive volume: measure ambient noise and adjust output volume.
    # EMA smoothing over ~10s worth of 32ms frames (α ≈ 0.003).
    _AMBIENT_EMA_ALPHA = 0.003
    _AMBIENT_EMIT_INTERVAL = 5.0   # emit bus event every 5s

    def __init__(self) -> None:
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._playing = False
        self._muted = False
        self._prompt_history: list = []
        self._last_active_ts: float = 0.0
        self._ambient_rms: float = 0.005   # start with quiet assumption
        self._ambient_last_emit: float = 0.0

        # OpenWakeWord detector (lazy init)
        self._wake_detector = None

        # Bus subscriptions
        bus.on("tts.started", self._on_tts_start)
        bus.on("tts.finished", self._on_tts_end)
        bus.on("mute.toggled", self._on_mute)

    # ----- Bus callbacks -----

    def _on_tts_start(self, **_kw):
        self._playing = True

    def _on_tts_end(self, **_kw):
        self._playing = False

    def _on_mute(self, muted: bool = False, **_kw):
        self._muted = muted
        if muted:
            print("[listener] MUTED — mic stream paused")
        else:
            print("[listener] UNMUTED — mic stream resumed")

    # ----- Control -----

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True, name="listener")
        self._thread.start()

    def stop(self):
        self._stop.set()

    # ----- Main loop -----

    def _run(self):
        memlog.delta("listener: before VAD load")
        vad = _get_vad()
        memlog.delta("listener: VAD loaded")
        warmup_whisper()
        memlog.delta("listener: Whisper warmed up")

        # Try to init wake word detector
        wake_enabled = state.wake_word_enabled
        if wake_enabled:
            try:
                from voice._openwakeword import create_detector
                self._wake_detector = create_detector()
                print("[listener] OpenWakeWord detector ready")
            except Exception as e:
                print(f"[listener] Wake word init failed (continuing without): {e}")
                self._wake_detector = None

        # Open mic stream — use alsaaudio directly (PortAudio can't see ReSpeaker)
        mic_dev = _find_alsa_card("Array")
        if mic_dev is None:
            print("[listener] No reSpeaker mic found — cannot listen")
            return
        try:
            from voice.alsa_mic import AlsaMic
            stream = AlsaMic(
                device=mic_dev,
                rate=SAMPLE_RATE,
                period_size=FRAME_SIZE,
            )
        except Exception as e:
            print(f"[listener] Cannot open mic stream ({mic_dev}): {e}")
            return

        print("[listener] Listening...")
        bus.emit("listener.state", state="waiting")
        listening_active = not wake_enabled

        try:
            while not self._stop.is_set():
                # Read a frame
                try:
                    data, overflowed = stream.read(FRAME_SIZE)
                except Exception:
                    time.sleep(0.01)
                    continue

                if self._muted:
                    # Mic is muted — stop the stream entirely for privacy
                    stream.stop()
                    while self._muted and not self._stop.is_set():
                        time.sleep(0.1)
                    if not self._stop.is_set():
                        stream.start()
                    continue

                if self._playing:
                    continue

                # Extract mono channel
                if data.ndim > 1:
                    mono = data[:, MIC_CHANNEL].astype(np.float32) / 32768.0
                else:
                    mono = data.astype(np.float32) / 32768.0

                # Stage 1: Wake word gate
                if wake_enabled and not listening_active:
                    if self._wake_detector:
                        try:
                            conf = self._wake_detector.process(mono)
                            if conf > 0.5:
                                listening_active = True
                                vad.reset_states()
                                bus.emit("listener.state", state="listening")
                                print("[listener] Wake word detected!")
                        except Exception:
                            pass
                    continue

                # Emit mic level for VU meter (cheap RMS, every frame)
                _rms = float(np.sqrt(np.mean(mono ** 2)))
                bus.emit("mic.level", rms=_rms)

                # Stage 2: VAD
                tensor = torch.from_numpy(mono)
                vad_prob = float(vad(tensor, SAMPLE_RATE).detach())

                if vad_prob < VAD_START_THRESH:
                    # Ambient noise measurement (silence frames only)
                    a = self._AMBIENT_EMA_ALPHA
                    self._ambient_rms = (1 - a) * self._ambient_rms + a * _rms
                    now = time.time()
                    if now - self._ambient_last_emit >= self._AMBIENT_EMIT_INTERVAL:
                        self._ambient_last_emit = now
                        bus.emit("ambient.level", rms=self._ambient_rms)
                    continue

                # Speech detected — start recording
                bus.emit("listener.state", state="listening")
                bus.emit("listener.vad", active=True)
                buffer = [data]
                silence_start: Optional[float] = None

                # Record until silence
                while not self._stop.is_set():
                    try:
                        data2, _ = stream.read(FRAME_SIZE)
                    except Exception:
                        break

                    buffer.append(data2)

                    if data2.ndim > 1:
                        mono2 = data2[:, MIC_CHANNEL].astype(np.float32) / 32768.0
                    else:
                        mono2 = data2.astype(np.float32) / 32768.0

                    tensor2 = torch.from_numpy(mono2)
                    vp = float(vad(tensor2, SAMPLE_RATE).detach())

                    if vp < VAD_SILENCE_THRESH:
                        if silence_start is None:
                            silence_start = time.time()
                        elif (time.time() - silence_start) >= SILENCE_TIMEOUT:
                            break
                    else:
                        silence_start = None

                # Process recorded audio
                bus.emit("listener.vad", active=False)
                bus.emit("listener.state", state="transcribing")

                full = np.concatenate(buffer)
                if full.ndim > 1:
                    audio = full[:, MIC_CHANNEL].astype(np.float32) / 32768.0
                else:
                    audio = full.astype(np.float32) / 32768.0

                if len(audio) < MIN_AUDIO_SAMPLES:
                    bus.emit("listener.state", state="waiting")
                    vad.reset_states()
                    if wake_enabled:
                        listening_active = False
                    continue

                # Advanced speech filter
                feats = calculate_audio_features(audio)
                dur = len(audio) / SAMPLE_RATE
                ok, reason = is_likely_speech(feats, dur)
                if not ok:
                    print(f"[listener] Rejected: {reason}")
                    _diag_rejected("(audio)", reason)
                    bus.emit("listener.state", state="waiting")
                    vad.reset_states()
                    if wake_enabled:
                        listening_active = False
                    continue

                # Transcribe
                text = transcribe(audio)
                vad.reset_states()

                if text:
                    # Drop Whisper hallucinations (common phantom transcripts)
                    clean_lower = text.strip().lower().rstrip(".,!?")
                    if (len(text.strip()) < 3 or
                            clean_lower in WHISPER_HALLUCINATIONS or
                            text.strip().lower() in WHISPER_HALLUCINATIONS):
                        print(f"[listener] Rejected (hallucination): '{text}'")
                        _diag_rejected(text, "hallucination")
                        bus.emit("listener.state", state="waiting")
                        if wake_enabled:
                            listening_active = False
                        continue
                    print(f"[mic] \"{text}\"")
                    _diag_heard(text)
                    # When wake word is disabled, respond to everything
                    # When enabled, use wake word + context window logic
                    if not wake_enabled or should_respond(text, self._last_active_ts):
                        clean = strip_wake(text) if heard_wake(text) else text
                        if clean:
                            self._last_active_ts = time.time()
                            self._prompt_history.append(clean)
                            if len(self._prompt_history) > CONTEXT_DEPTH:
                                self._prompt_history = self._prompt_history[-CONTEXT_DEPTH:]
                            bus.emit("transcript.ready", text=clean)
                    else:
                        print(f"[listener] Ignored (no wake/context): '{text[:60]}'")

                bus.emit("listener.state", state="waiting")
                if wake_enabled:
                    listening_active = False

        finally:
            try:
                stream.stop()
                stream.close()
            except Exception:
                pass
            if self._wake_detector and hasattr(self._wake_detector, "cleanup"):
                self._wake_detector.cleanup()
            print("[listener] Stopped")

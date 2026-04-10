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
from scipy.signal import butter, sosfilt

from services.diaglog import heard as _diag_heard, rejected as _diag_rejected

from core.bus import bus
from core.config import (
    SAMPLE_RATE, WHISPER_URL, MIC_GAIN,
)
from core.state import state
from services.memlog import memlog
from voice.wake import heard_wake, should_respond, strip_wake

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

FRAME_SIZE          = int(SAMPLE_RATE * 0.032)      # ~512 samples, 32ms
SILENCE_TIMEOUT     = 0.5                            # seconds — aggressive for fast turn-taking
VAD_START_THRESH    = 0.30                          # balanced: 0.25 too sensitive, 0.45 missed real speech at 4ft
VAD_SILENCE_THRESH  = 0.10
MIN_AUDIO_SAMPLES   = 2000                          # ~125ms
MIN_RMS_TO_RECORD   = 0.001                         # very low for cross-device audio
MIN_RECORD_DURATION = 0.6                           # seconds — sub-600ms is never real speech

DEVICE_NAME         = "reSpeaker"
MIC_CHANNEL         = 0                             # XVF3800 channel 0 (AEC-processed: beamformed + echo cancellation)

# Advanced filter thresholds (calibrated for XVF3800 + beamforming)
SPEECH_ZCR_MAX        = 0.50
SPEECH_FLATNESS_MAX   = 0.75
SPEECH_CENTROID_MIN   = 200.0
SPEECH_CENTROID_MAX   = 5000.0
SPEECH_BAND_MIN       = 0.03

# Barge-in: interrupt Aura when user speaks over her
# Strategy: compare ch0 (AEC) vs ch1 (raw). During speaker-only playback,
# ch0/ch1 ratio ≈ 0.5 (AEC provides ~2x suppression). When human speaks,
# their voice appears equally on both channels, pushing ratio toward 1.0.
BARGEIN_VAD_THRESH    = 0.70          # VAD on ch0 must detect speech
BARGEIN_FRAMES        = 8             # consecutive frames (~256ms)
BARGEIN_RATIO_MIN     = 0.75          # ch0/ch1 ratio above this = human voice present (speaker-only ≈ 0.5)
BARGEIN_RMS_MIN       = 0.06          # minimum ch0 RMS to even consider (above ambient noise)
BARGEIN_AEC_WARMUP    = 1.0           # seconds after TTS starts before enabling barge-in
SPEECH_DURATION_MIN   = 0.2
SPEECH_HIGH_FREQ_MAX  = 0.40
SPEECH_RMS_MIN        = 0.0005
SPEECH_RMS_MAX        = 0.90
SPEECH_PEAK_MIN       = 0.0008

CONTEXT_DEPTH = 6

# Whisper confidence gating — reject low-confidence transcriptions
WHISPER_MIN_LOG_PROB     = -0.7   # avg_log_prob below this → likely hallucination/confabulation
WHISPER_MAX_NO_SPEECH    = 1.1    # disabled — VAD already gates speech; Whisper nsp is unreliable

# Bandpass filter for speech (80-7500 Hz) — removes fan rumble + high-freq hiss
_BANDPASS_SOS = butter(4, [80.0, 7500.0], btype="bandpass", fs=SAMPLE_RATE, output="sos")

# Common Whisper hallucinations on silence/noise — reject these outright
WHISPER_HALLUCINATIONS = {
    "bye", "bye.", "hmm", "hm",
    "oh", "ah", "uh", "um",
    "the end", "the end.", "thanks for watching", "thanks for watching.",
    "thank you for watching", "thank you for watching.",
    "subscribe", "like and subscribe",
    "this is a conversation", "this is a conversation.",
}

# Regex patterns for hallucinations that vary slightly each time
# (e.g. "Thank you, Oro.", "Thank you, sir.", "Thanks for watching!")
_HALLUCINATION_PATTERNS = [
    re.compile(r"^thank you[,.]?\s+\w+[.!]?$", re.IGNORECASE),
    re.compile(r"^thanks for \w+[.!]?$", re.IGNORECASE),
    re.compile(r"^(please )?subscribe[.!]?$", re.IGNORECASE),
    # Catch repeated phrases (Whisper INITIAL_PROMPT leak) e.g. "X. X. X. X."
    re.compile(r"^(.{4,40}?)[\s.!?,]*(?:\1[\s.!?,]*){2,}$", re.IGNORECASE),
    # Catch "this is a conversation" with any surrounding text
    re.compile(r"this is a conversation", re.IGNORECASE),
]

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
# Whisper — in-process (Phase B) with HTTP fallback
# ---------------------------------------------------------------------------

# Try in-process engine first; fall back to HTTP if unavailable
_USE_IN_PROCESS_WHISPER = True

def _get_whisper_engine():
    """Lazy import to avoid circular deps during boot."""
    try:
        from voice.whisper_engine import whisper_engine
        return whisper_engine
    except Exception:
        return None


def transcribe(audio: np.ndarray, sr: int = SAMPLE_RATE) -> tuple[str, float, float]:
    """Transcribe audio via in-process Whisper (or HTTP fallback)."""
    # Final speech filter
    feats = calculate_audio_features(audio, sr)
    dur = len(audio) / sr
    ok, reason = is_likely_speech(feats, dur)
    if not ok:
        print(f"[listener] Rejected (post-filter): {reason}")
        return "", -1.0, 1.0

    # Apply bandpass filter to clean audio before Whisper
    filtered = sosfilt(_BANDPASS_SOS, audio).astype(np.float32)

    # RMS-normalize: target 0.1, cap gain at 15x
    rms_raw = float(np.sqrt(np.mean(filtered ** 2)))
    if rms_raw > 0.001:
        gain = min(0.1 / rms_raw, 15.0)
        filtered = np.clip(filtered * gain, -1.0, 1.0)

    # Save diagnostic WAV (last 3 recordings)
    try:
        _diag_idx = getattr(transcribe, "_diag_idx", 0)
        _diag_path = f"/tmp/aura_whisper_diag_{_diag_idx % 3}.wav"
        sf.write(_diag_path, filtered, sr, format="WAV", subtype="PCM_16")
        transcribe._diag_idx = _diag_idx + 1
        rms_filtered = float(np.sqrt(np.mean(filtered ** 2)))
        print(f"[listener] Diag WAV saved: {_diag_path} (rms={rms_filtered:.4f}, dur={dur:.1f}s)")
    except Exception:
        pass

    # ── In-process path (no HTTP, no WAV encode) ──
    engine = _get_whisper_engine() if _USE_IN_PROCESS_WHISPER else None
    if engine and engine.loaded:
        return engine.transcribe(filtered, sr)

    # ── HTTP fallback ──
    buf = io.BytesIO()
    sf.write(buf, filtered, sr, format="WAV", subtype="PCM_16")
    buf.seek(0)
    try:
        resp = requests.post(
            f"{WHISPER_URL}/transcribe",
            files={"audio": ("audio.wav", buf, "audio/wav")},
            timeout=15,
        )
        if resp.status_code != 200:
            print(f"[listener] Whisper HTTP {resp.status_code}")
            return "", -1.0, 1.0
        data = resp.json()
        return (
            data.get("text", "").strip(),
            data.get("avg_log_prob", -1.0),
            data.get("no_speech_prob", 1.0),
        )
    except Exception as e:
        print(f"[listener] Whisper error: {e}")
        return "", -1.0, 1.0


def warmup_whisper():
    """Prime Whisper JIT with 1s silence."""
    engine = _get_whisper_engine() if _USE_IN_PROCESS_WHISPER else None
    if engine and engine.loaded:
        engine.warmup()
        return
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

    # Echo gate holdoff: keep mic suppressed for this many seconds after
    # TTS finishes, so the room reverb / speaker tail doesn't trigger VAD.
    _ECHO_HOLDOFF_S = 0.5

    def _on_tts_start(self, **_kw):
        print(f"[listener] tts.started → echo gate ON")
        self._playing = True
        self._tts_start_ts = time.time()

    def _on_tts_end(self, **_kw):
        print(f"[listener] tts.finished → holdoff {self._ECHO_HOLDOFF_S}s")
        # Delay un-muting so residual room audio doesn't trigger VAD
        def _delayed_release():
            time.sleep(self._ECHO_HOLDOFF_S)
            # Only release if no new TTS started during the holdoff
            if not state.playing:
                self._playing = False
                print("[listener] Echo gate released")
            else:
                print("[listener] Echo gate holdoff: state.playing still True, NOT releasing")
        threading.Thread(target=_delayed_release, daemon=True,
                         name="echo-holdoff").start()

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
        from voice.alsa_mic import AlsaMic
        stream = None
        for _attempt in range(15):
            try:
                stream = AlsaMic(
                    device=mic_dev,
                    rate=SAMPLE_RATE,
                    period_size=FRAME_SIZE,
                )
                break
            except Exception as e:
                print(f"[listener] Cannot open mic stream ({mic_dev}): {e}")
                if _attempt < 14:
                    time.sleep(3.0)
        if stream is None:
            print("[listener] Mic unavailable after 15 attempts — cannot listen")
            return

        print("[listener] Listening...")
        bus.emit("listener.state", state="waiting")
        listening_active = not wake_enabled
        _bargein_count = 0

        _heartbeat_ts = time.time()
        try:
            while not self._stop.is_set():
                # Heartbeat: log every 30s so we know the thread is alive
                _hb_now = time.time()
                if (_hb_now - _heartbeat_ts) >= 30.0:
                    _heartbeat_ts = _hb_now
                    print(f"[listener] heartbeat (playing={self._playing}, "
                          f"state.playing={state.playing}, muted={self._muted})")

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

                # Barge-in detection: while Aura is speaking, monitor ch0
                # (AEC-processed) for real human speech. The XVF3800 DSP
                # cancels the speaker output on ch0, so any signal above
                # threshold is the user trying to interrupt.
                if self._playing or state.playing:
                    # Wait for AEC to converge before checking for barge-in
                    _tts_elapsed = time.time() - getattr(self, '_tts_start_ts', 0)
                    if _tts_elapsed < BARGEIN_AEC_WARMUP:
                        _bargein_count = 0
                        continue

                    # Extract both channels for ratio comparison
                    if data.ndim > 1:
                        _bi_ch0 = data[:, 0].astype(np.float32) / 32768.0
                        _bi_ch1 = data[:, 1].astype(np.float32) / 32768.0
                    else:
                        _bi_ch0 = data.astype(np.float32) / 32768.0
                        _bi_ch1 = _bi_ch0
                    _bi_rms0 = float(np.sqrt(np.mean(_bi_ch0 ** 2)))
                    _bi_rms1 = float(np.sqrt(np.mean(_bi_ch1 ** 2)))
                    _bi_ratio = _bi_rms0 / max(_bi_rms1, 0.0001)
                    _bi_tensor = torch.from_numpy(_bi_ch0)
                    _bi_vad = float(vad(_bi_tensor, SAMPLE_RATE).detach())

                    # Human speech: ratio→1.0 (voice on both channels equally)
                    # Speaker only: ratio≈0.5 (AEC suppresses ch0)
                    if (_bi_vad >= BARGEIN_VAD_THRESH
                            and _bi_rms0 >= BARGEIN_RMS_MIN
                            and _bi_ratio >= BARGEIN_RATIO_MIN):
                        _bargein_count += 1
                    else:
                        _bargein_count = max(0, _bargein_count - 1)

                    if _bargein_count >= BARGEIN_FRAMES:
                        print(f"[listener] 🔊 BARGE-IN detected (vad={_bi_vad:.2f}, "
                              f"ch0={_bi_rms0:.4f}, ratio={_bi_ratio:.2f}, frames={_bargein_count})")
                        bus.emit("bargein")
                        _bargein_count = 0
                        # Brief pause for speaker to stop, then resume normal listening
                        time.sleep(0.15)
                        vad.reset_states()
                        continue

                    # Still playing, no barge-in — suppress normal processing
                    _now = time.time()
                    if not hasattr(self, '_echo_gate_log_ts') or (_now - self._echo_gate_log_ts) > 2.0:
                        self._echo_gate_log_ts = _now
                        print(f"[listener] Echo gate (vad={_bi_vad:.2f}, rms0={_bi_rms0:.4f}, "
                              f"ratio={_bi_ratio:.2f}, cnt={_bargein_count})")
                    continue

                # Extract mono channel + apply digital mic gain
                if data.ndim > 1:
                    mono = data[:, MIC_CHANNEL].astype(np.float32) / 32768.0
                else:
                    mono = data.astype(np.float32) / 32768.0
                if MIC_GAIN != 1.0:
                    mono = np.clip(mono * MIC_GAIN, -1.0, 1.0)

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
                self._mid_rec_bargein_sent = False
                buffer = [data]
                silence_start: Optional[float] = None
                _rec_start = time.time()
                _MAX_RECORD_S = 15.0  # hard cap — no utterance > 15s
                print(f"[listener] Recording started (vad={vad_prob:.3f}, rms={_rms:.4f})")

                # Record until silence
                while not self._stop.is_set():
                    # Mute pressed mid-recording — discard everything
                    if self._muted:
                        print("[listener] Mute during recording — discarding buffer")
                        buffer.clear()
                        break

                    # Mid-recording: if TTS started playing, this is a barge-in
                    # in progress. Keep recording — ch0 AEC removes speaker output.
                    # Emit bargein to stop TTS so we capture clean audio.
                    if self._playing or state.playing:
                        if not hasattr(self, '_mid_rec_bargein_sent') or not self._mid_rec_bargein_sent:
                            print("[listener] TTS playing mid-recording — barge-in, stopping TTS")
                            bus.emit("bargein")
                            self._mid_rec_bargein_sent = True
                        # Continue recording (ch0 AEC handles echo)

                    try:
                        data2, _ = stream.read(FRAME_SIZE)
                    except Exception:
                        break

                    buffer.append(data2)

                    if data2.ndim > 1:
                        mono2 = data2[:, MIC_CHANNEL].astype(np.float32) / 32768.0
                    else:
                        mono2 = data2.astype(np.float32) / 32768.0
                    if MIC_GAIN != 1.0:
                        mono2 = np.clip(mono2 * MIC_GAIN, -1.0, 1.0)

                    tensor2 = torch.from_numpy(mono2)
                    vp = float(vad(tensor2, SAMPLE_RATE).detach())

                    # Hard cap on recording duration
                    if (time.time() - _rec_start) >= _MAX_RECORD_S:
                        print(f"[listener] Recording hit {_MAX_RECORD_S}s cap — stopping")
                        break

                    if vp < VAD_SILENCE_THRESH:
                        if silence_start is None:
                            silence_start = time.time()
                        elif (time.time() - silence_start) >= SILENCE_TIMEOUT:
                            break
                    else:
                        silence_start = None

                # Process recorded audio
                _rec_dur = time.time() - _rec_start
                print(f"[listener] Recording ended: {_rec_dur:.1f}s, "
                      f"{len(buffer)} frames, buffer_cleared={len(buffer)==0}")
                bus.emit("listener.vad", active=False)

                # If buffer was cleared (mute during recording), skip
                if not buffer:
                    bus.emit("listener.state", state="waiting")
                    vad.reset_states()
                    if wake_enabled:
                        listening_active = False
                    continue

                # Too-short recordings are never real speech — skip Whisper
                if _rec_dur < MIN_RECORD_DURATION:
                    print(f"[listener] Rejected (too short): {_rec_dur:.1f}s < {MIN_RECORD_DURATION}s")
                    bus.emit("listener.state", state="waiting")
                    vad.reset_states()
                    if wake_enabled:
                        listening_active = False
                    continue

                # Check RMS of full recording — very quiet = ambient noise, not speech
                _full_tmp = np.concatenate(buffer)
                if _full_tmp.ndim > 1:
                    _full_mono = _full_tmp[:, MIC_CHANNEL].astype(np.float32) / 32768.0
                else:
                    _full_mono = _full_tmp.astype(np.float32) / 32768.0
                _full_rms = float(np.sqrt(np.mean(_full_mono ** 2)))
                if _full_rms < MIN_RMS_TO_RECORD:
                    print(f"[listener] Rejected (too quiet): rms={_full_rms:.4f} < {MIN_RMS_TO_RECORD}")
                    bus.emit("listener.state", state="waiting")
                    vad.reset_states()
                    if wake_enabled:
                        listening_active = False
                    continue

                # Final echo gate check: if TTS kicked in right at end of
                # recording, discard before spending time on transcription.
                if self._playing or state.playing:
                    print("[listener] TTS active after recording — discarding")
                    bus.emit("listener.state", state="waiting")
                    vad.reset_states()
                    if wake_enabled:
                        listening_active = False
                    continue

                bus.emit("listener.state", state="transcribing")

                full = np.concatenate(buffer)
                if full.ndim > 1:
                    audio = full[:, MIC_CHANNEL].astype(np.float32) / 32768.0
                else:
                    audio = full.astype(np.float32) / 32768.0
                if MIC_GAIN != 1.0:
                    audio = np.clip(audio * MIC_GAIN, -1.0, 1.0)

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
                text, avg_log_prob, no_speech_prob = transcribe(audio)
                vad.reset_states()

                if text:
                    # Confidence gate — reject low-confidence transcriptions
                    if avg_log_prob < WHISPER_MIN_LOG_PROB or no_speech_prob > WHISPER_MAX_NO_SPEECH:
                        print(f"[listener] Rejected (low confidence): '{text}' "
                              f"(log_prob={avg_log_prob:.2f}, nsp={no_speech_prob:.2f})")
                        _diag_rejected(text, f"low_confidence lp={avg_log_prob:.2f} nsp={no_speech_prob:.2f}")
                        bus.emit("listener.state", state="waiting")
                        if wake_enabled:
                            listening_active = False
                        continue

                    # Drop Whisper hallucinations (common phantom transcripts)
                    clean_lower = text.strip().lower().rstrip(".,!?")
                    _is_hallucination = (
                        len(text.strip()) < 3
                        or clean_lower in WHISPER_HALLUCINATIONS
                        or text.strip().lower() in WHISPER_HALLUCINATIONS
                        or any(p.match(text.strip()) for p in _HALLUCINATION_PATTERNS)
                    )
                    if _is_hallucination:
                        print(f"[listener] Rejected (hallucination): '{text}'")
                        _diag_rejected(text, "hallucination")
                        bus.emit("listener.state", state="waiting")
                        if wake_enabled:
                            listening_active = False
                        continue
                    print(f"[mic] \"{text}\" (conf={avg_log_prob:.2f}, nsp={no_speech_prob:.2f})")
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

"""
voice.speaker -- TTS queue → XTTS v2 voice cloning → DeepFilterNet cleanup → playback.

Runs on a daemon thread.  Communicates via bus events:
    listens: "llm.sentence"   text=str
    emits:   "tts.started", "tts.finished"
             "speaker.state"  state=str ("idle"|"synthesizing"|"playing")

Uses XTTS v2 (zero-shot voice cloning from 15 reference clips, 24kHz output)
with DeepFilterNet neural noise suppression post-processing.
Zero Qt imports.
"""

from __future__ import annotations

import glob as _glob
import os

# Must be set before any coqui/TTS imports to avoid interactive prompts
os.environ["COQUI_TOS_AGREED"] = "1"
os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"

import queue
import random
import re
import subprocess
import threading
import time
import wave
from typing import Optional

import numpy as np
from dotenv import load_dotenv

from core.bus import bus
from core.config import (
    WORKSPACE_ROOT, TTS_VOLUME, TTS_GAIN, VOICES_DIR, FARSIGHT_URL,
    XTTS_REFS_DIR, XTTS_SAMPLE_RATE, XTTS_TEMPERATURE, XTTS_REP_PENALTY,
    XTTS_LENGTH_PENALTY,
)
from core.state import state
from services.diaglog import said as _diag_said
from services.memlog import memlog

# ---------------------------------------------------------------------------
# .env (API keys live here)
# ---------------------------------------------------------------------------

_dotenv = WORKSPACE_ROOT / ".env"
if _dotenv.exists():
    load_dotenv(str(_dotenv))

# ---------------------------------------------------------------------------
# XTTS v2 voice cloning (lazy-loaded, replaces Kokoro+RVC)
# ---------------------------------------------------------------------------

_xtts_model = None
_xtts_gpt_cond = None      # cached speaker conditioning latents
_xtts_speaker_emb = None    # cached speaker embedding


def _get_xtts():
    """Lazy-load XTTS v2 and pre-compute speaker embedding from reference clips.

    Speaker embedding is computed once from 15 reference WAVs and cached for all
    subsequent inference calls — avoids re-reading clips on every synthesis.
    XTTS v2 runs on CUDA (~3GB VRAM, 24kHz output).
    """
    global _xtts_model, _xtts_gpt_cond, _xtts_speaker_emb
    if _xtts_model is not None:
        return _xtts_model

    memlog.delta("speaker: before XTTS load")

    import torch
    from TTS.api import TTS

    _xtts_model = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to("cuda")

    # Pre-compute speaker embedding from reference clips (one-time cost)
    refs = sorted(_glob.glob(str(XTTS_REFS_DIR / "ref_*.wav")))
    if not refs:
        print(f"[speaker] WARNING: No reference WAVs in {XTTS_REFS_DIR}")
    else:
        _xtts_gpt_cond, _xtts_speaker_emb = (
            _xtts_model.synthesizer.tts_model.get_conditioning_latents(audio_path=refs)
        )
        print(f"[speaker] XTTS v2 loaded on CUDA ({len(refs)} refs, embedding cached)")

    memlog.delta("speaker: XTTS loaded")
    return _xtts_model


# ---------------------------------------------------------------------------
# DeepFilterNet neural noise suppression (lazy-loaded)
# ---------------------------------------------------------------------------

_deepfilter_model = None
_deepfilter_state = None


def _get_deepfilter():
    """Lazy-load DeepFilterNet3 for neural noise suppression."""
    global _deepfilter_model, _deepfilter_state
    if _deepfilter_model is not None:
        return _deepfilter_model, _deepfilter_state

    memlog.delta("speaker: before DeepFilterNet load")

    # Patch torchaudio for DeepFilterNet compatibility (PyTorch 2.6+)
    import torchaudio
    if not hasattr(torchaudio, 'backend'):
        import types
        import sys as _sys
        from collections import namedtuple
        torchaudio.backend = types.ModuleType('torchaudio.backend')
        torchaudio.backend.common = types.ModuleType('torchaudio.backend.common')
        torchaudio.backend.common.AudioMetaData = namedtuple(
            'AudioMetaData',
            ['sample_rate', 'num_frames', 'num_channels', 'bits_per_sample', 'encoding'],
        )
        _sys.modules['torchaudio.backend'] = torchaudio.backend
        _sys.modules['torchaudio.backend.common'] = torchaudio.backend.common

    from df.enhance import init_df
    _deepfilter_model, _deepfilter_state, _ = init_df()
    print(f"[speaker] DeepFilterNet loaded (sr={_deepfilter_state.sr()})")
    memlog.delta("speaker: DeepFilterNet loaded")
    return _deepfilter_model, _deepfilter_state


def _deepfilter_clean(audio_np: np.ndarray, sr: int) -> np.ndarray:
    """Clean audio using DeepFilterNet neural noise suppression.

    Handles sample-rate conversion (XTTS 24kHz ↔ DeepFilterNet 48kHz).
    """
    import torch
    import torchaudio

    from df.enhance import enhance

    df_model, df_state = _get_deepfilter()
    dfn_sr = df_state.sr()

    t = torch.from_numpy(audio_np).unsqueeze(0).float()
    if sr != dfn_sr:
        t = torchaudio.functional.resample(t, sr, dfn_sr)
    enhanced = enhance(df_model, df_state, t)
    if sr != dfn_sr:
        enhanced = torchaudio.functional.resample(enhanced, dfn_sr, sr)
    return enhanced.squeeze().numpy()


# ---------------------------------------------------------------------------
# Local model readiness — tracks whether XTTS+DeepFilterNet are loaded
# ---------------------------------------------------------------------------

_local_tts_ready = threading.Event()


def is_local_tts_ready() -> bool:
    """Check if the local TTS pipeline (XTTS v2) is loaded and warm."""
    return _local_tts_ready.is_set()


def warm_local_tts_background():
    """Load XTTS v2 in a background thread. Non-blocking.

    The thread sets _local_tts_ready once the model is in VRAM.
    Speaker._stream_synth will use Farsight as fallback until this completes.
    """
    def _warm():
        try:
            _get_xtts()
            _local_tts_ready.set()
            print("[speaker] Local TTS pipeline warm (XTTS ready)")
        except Exception as e:
            print(f"[speaker] Local TTS warmup failed: {e}")
            _local_tts_ready.set()
    threading.Thread(target=_warm, daemon=True, name="tts-local-warm").start()


# ---------------------------------------------------------------------------
# Farsight RTX TTS fallback — used while local models load
# ---------------------------------------------------------------------------

def _farsight_available() -> bool:
    """Quick check: can we reach the Farsight RTX TTS endpoint?"""
    if not FARSIGHT_URL:
        return False
    try:
        import requests
        resp = requests.get(f"{FARSIGHT_URL}/health", timeout=2)
        return resp.status_code == 200
    except Exception:
        return False


_farsight_ok: Optional[bool] = None  # cached after first check


def _check_farsight_once() -> bool:
    """Check Farsight availability once and cache the result."""
    global _farsight_ok
    if _farsight_ok is not None:
        return _farsight_ok
    _farsight_ok = _farsight_available()
    if _farsight_ok:
        print("[speaker] Farsight RTX TTS available — will use as fallback during warmup")
    else:
        print("[speaker] Farsight RTX not reachable — local TTS only")
    return _farsight_ok


def _farsight_synth_to_file(text: str, out_path: str) -> bool:
    """Synthesize text on the remote Farsight RTX and save as WAV.

    Returns True on success.  The RTX is always warm (never restarted),
    so this is fast (~200-500ms for a sentence over local network).
    """
    import requests
    try:
        resp = requests.post(
            f"{FARSIGHT_URL}/perpetual/synthesize",
            json={"text": text, "steps": 50},  # lower steps for speed
            timeout=15,
        )
        if resp.status_code != 200:
            return False
        with open(out_path, "wb") as f:
            f.write(resp.content)
        return True
    except Exception as e:
        print(f"[speaker] Farsight TTS error: {e}")
        return False


# ---------------------------------------------------------------------------
# Text processing
# ---------------------------------------------------------------------------

_CLEAN_RE = [
    (re.compile(r"^#{1,6}\s+", re.M), ""),
    (re.compile(r"#{1,6}(?=\s|$)"), ""),
    (re.compile(r"\*\*([^*]+)\*\*"), r"\1"),
    (re.compile(r"\*([^*\n]+)\*"), r"\1"),
    (re.compile(r"\*\*+"), ""),
    (re.compile(r"(?<!\w)\*(?!\w)"), ""),
    (re.compile(r"([a-zA-Z0-9])([.!?])([a-zA-Z-])"), r"\1\2 \3"),
    (re.compile(r"([,.!?:;])([a-zA-Z])"), r"\1 \2"),
    (re.compile(r" {2,}"), " "),
]


def _clean_tts_text(text: str) -> str:
    if not text:
        return text
    for pat, repl in _CLEAN_RE:
        text = pat.sub(repl, text)
    return text.strip()


def preprocess(text: str) -> str:
    text = re.sub(r"<sentence_start>|<sentence_end>|<pause>", "", text)
    return _clean_tts_text(text).strip()


# ---------------------------------------------------------------------------
# Volume control
# ---------------------------------------------------------------------------

_volume_set = False
_current_vol_pct = 0

# Adaptive volume: maps ambient RMS to ALSA mixer percentage.
# Quiet room (RMS ~0.002) → 30%, noisy room (RMS ~0.03+) → 90%.
_ADAPTIVE_VOL_MIN = 30
_ADAPTIVE_VOL_MAX = 90
_AMBIENT_RMS_QUIET = 0.002   # typical quiet room
_AMBIENT_RMS_LOUD = 0.03     # TV, conversation nearby


def _set_volume(vol_pct: int = 0):
    """Set playback volume (ALSA amixer).

    If vol_pct is 0, uses TTS_VOLUME from config (initial setup).
    Otherwise sets the given percentage directly.
    """
    global _volume_set, _current_vol_pct
    if vol_pct == 0:
        if _volume_set:
            return
        vol_pct = int(TTS_VOLUME * 100) if TTS_VOLUME <= 2.0 else int(TTS_VOLUME)

    if vol_pct == _current_vol_pct:
        return

    for ctrl in ("PCM", "Speaker", "Master"):
        try:
            subprocess.run(
                ["amixer", "-D", "hw:CARD=UACDemoV10", "sset", ctrl, f"{vol_pct}%"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                check=True, timeout=2,
            )
            _volume_set = True
            _current_vol_pct = vol_pct
            return
        except Exception:
            continue


def _on_ambient_level(rms: float = 0.0, **_kw):
    """Adjust output volume based on ambient noise level."""
    # Linear interpolation between quiet and loud thresholds
    t = (rms - _AMBIENT_RMS_QUIET) / (_AMBIENT_RMS_LOUD - _AMBIENT_RMS_QUIET)
    t = max(0.0, min(1.0, t))
    vol = int(_ADAPTIVE_VOL_MIN + t * (_ADAPTIVE_VOL_MAX - _ADAPTIVE_VOL_MIN))
    if vol != _current_vol_pct:
        print(f"[speaker] Adaptive volume: {vol}% (ambient RMS={rms:.4f})")
        _set_volume(vol)


bus.on("ambient.level", _on_ambient_level)


# ---------------------------------------------------------------------------
# Audio normalization (consistent volume across clauses)
# ---------------------------------------------------------------------------

# Target RMS level for normalization (-26 dBFS ≈ 0.05).
# Using fixed-target RMS instead of peak normalization prevents volume jumps
# between clauses (a clause with one loud spike won't crush overall volume).
# TTS_GAIN (default 1.7) is applied on top for final output level.
_TARGET_RMS = 0.05


def _normalize_audio(audio_np: np.ndarray) -> np.ndarray:
    """Normalize audio to a consistent RMS level, then apply TTS_GAIN."""
    if audio_np.size == 0:
        return audio_np
    rms = float(np.sqrt(np.mean(audio_np ** 2)))
    if rms < 1e-8:
        return audio_np
    gain = (_TARGET_RMS / rms) * TTS_GAIN
    audio_np = audio_np * gain
    return np.clip(audio_np, -1.0, 1.0)


# ---------------------------------------------------------------------------
# WAV file synthesis (kept for external callers like boot tour)
# ---------------------------------------------------------------------------

def _synth_to_file(text: str, style: str, out_path: str) -> float:
    """Synthesize *text* to a WAV file using XTTS v2 + DeepFilterNet.

    Returns wall-clock ms.
    """
    tts = _get_xtts()
    clean = re.sub(r"<[^>]+>", "", text).strip()
    clean = clean.replace("°C", "degrees Celsius").replace("°F", "degrees Fahrenheit")
    if not clean:
        return 0.0

    t0 = time.perf_counter()

    # Use cached speaker embedding for fast inference
    out = tts.synthesizer.tts_model.inference(
        clean, "en", _xtts_gpt_cond, _xtts_speaker_emb,
        temperature=XTTS_TEMPERATURE,
        repetition_penalty=XTTS_REP_PENALTY,
        length_penalty=XTTS_LENGTH_PENALTY,
    )
    audio_np = np.array(out["wav"], dtype=np.float32)

    if audio_np.size == 0:
        return 0.0

    # Consistent RMS normalization + gain (DeepFilterNet removed from live
    # pipeline — XTTS output is clean synthetic audio, doesn't need it.
    # Pre-baked boot/thinking WAVs still have DeepFilterNet from offline gen.)
    audio_np = _normalize_audio(audio_np)

    pcm16 = (audio_np * 32767.0).astype(np.int16)

    with wave.open(out_path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(XTTS_SAMPLE_RATE)
        wf.writeframes(pcm16.tobytes())

    return (time.perf_counter() - t0) * 1000


# ---------------------------------------------------------------------------
# Speaker class — clause-pipelined XTTS synthesis → aplay
# ---------------------------------------------------------------------------

# Sentinel objects for the unified work queue
_SENTINEL_WAV = "wav"       # item is a WAV file path
_SENTINEL_TEXT = "text"     # item is (text, style) to synthesize+stream


class Speaker:
    """Clause-pipelined TTS playback: XTTS v2 → DeepFilterNet → aplay.

    Architecture:
      - Single worker thread pulls from a unified queue
      - For text: synthesizes each clause via XTTS v2, cleans with DeepFilterNet,
        plays via aplay — clause N+1 synthesizes while N plays
      - For WAV files: plays via aplay (thinking fillers, tour, briefings)
      - Sentence N+1 synthesis begins as soon as N finishes playing
    """

    # Pre-generated thinking fillers (loaded once at import)
    _THINKING_DIR = WORKSPACE_ROOT / "assets" / "thinking_fillers"
    _thinking_wavs: list[str] = sorted(_glob.glob(str(_THINKING_DIR / "think_*.wav")))
    _breath_wavs: list[str] = sorted(_glob.glob(str(_THINKING_DIR / "breath_*.wav")))

    def __init__(self) -> None:
        # Unified work queue: items are (_SENTINEL_WAV, path) or (_SENTINEL_TEXT, (text, style))
        self._work_q: queue.Queue = queue.Queue()
        self._worker_thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._interrupted = threading.Event()   # set by interrupt() to abort playback
        self._muted = False                      # inviolable: nothing plays when True
        self._current_aplay: Optional[subprocess.Popen] = None  # track aplay for kill

        # Keep legacy queues as thin wrappers for external code that checks them
        self._sentence_q = self._work_q  # compatibility alias
        self._wav_q = self._work_q       # compatibility alias

        bus.on("llm.sentence", self._on_sentence)
        bus.on("mute.toggled", self._on_mute_toggled)

    def _on_mute_toggled(self, muted: bool = False, **_kw):
        """When muted, immediately interrupt all playback and block new work."""
        self._muted = muted
        if muted:
            self.interrupt()

    def _on_sentence(self, text: str = "", style: str = "", **_kw):
        if self._muted:
            return  # inviolable: nothing queued while muted
        text = preprocess(text)
        if text and not re.match(r"^[\s.,!?]+$", text):
            self._work_q.put((_SENTINEL_TEXT, (text, style or "neutral")))

    def enqueue(self, text: str, style: str = "neutral"):
        """Manually enqueue text (for intro prompts, etc.)."""
        if self._muted:
            return
        text = preprocess(text)
        if text:
            self._work_q.put((_SENTINEL_TEXT, (text, style)))

    def enqueue_wav(self, wav_path: str):
        """Push a pre-synthesized WAV directly to the playback queue."""
        if self._muted:
            return
        if os.path.exists(wav_path):
            self._work_q.put((_SENTINEL_WAV, wav_path))
        else:
            print(f"[speaker] Pre-synth WAV not found: {wav_path}")

    # Minimum filler size (~3s at 48kHz stereo 16-bit ≈ 150kB).
    # Shorter ones tend to be terse single words ("So", "Well") that sound awkward.
    _MIN_FILLER_BYTES = 148000

    # Words that signal a complex query needing longer think time
    _COMPLEX_SIGNALS = re.compile(
        r'\b(how|why|explain|compare|difference|recipe|steps?|instructions?'
        r'|history|describe|analyze|what happens|tell me about)\b', re.I
    )

    def _estimate_complexity(self, query: str) -> str:
        """Estimate if a query is quick or complex based on simple heuristics.

        Returns 'quick' (breath filler) or 'complex' (verbal filler).
        """
        words = query.split()
        # Very short queries (1-4 words) are almost always quick
        if len(words) <= 4:
            return "quick"
        # Long queries or those with complex signal words need more time
        if len(words) >= 10 or self._COMPLEX_SIGNALS.search(query):
            return "complex"
        return "quick"

    def play_thinking_filler(self, query: str = ""):
        """Queue a filler sound while LLM generates.

        Short/simple queries get a breath intake (~1.5s).
        Complex queries get a longer verbal filler (~3-4s) to buy more time.
        """
        if self._muted:
            return
        complexity = self._estimate_complexity(query) if query else "quick"
        if complexity == "quick" and self._breath_wavs:
            wav = random.choice(self._breath_wavs)
        elif self._thinking_wavs:
            good = [w for w in self._thinking_wavs
                    if os.path.getsize(w) >= self._MIN_FILLER_BYTES]
            wav = random.choice(good or self._thinking_wavs)
        elif self._breath_wavs:
            wav = random.choice(self._breath_wavs)
        else:
            return
        print(f"[speaker] Thinking filler ({complexity}): {os.path.basename(wav)}")
        self._work_q.put((_SENTINEL_WAV, wav))

    # ----- Control -----

    def start(self):
        if self._worker_thread and self._worker_thread.is_alive():
            return
        self._stop.clear()
        self._worker_thread = threading.Thread(
            target=self._worker_loop, daemon=True, name="speaker-worker",
        )
        self._worker_thread.start()

    def stop(self):
        self._stop.set()

    def interrupt(self):
        """Immediately kill playback, flush pending work, and silence output."""
        # Kill any running aplay process
        proc = getattr(self, '_current_aplay', None)
        if proc and proc.poll() is None:
            try:
                proc.kill()
            except Exception:
                pass
        # Flush the work queue
        while not self._work_q.empty():
            try:
                self._work_q.get_nowait()
            except queue.Empty:
                break
        # Signal synth threads to abort
        self._interrupted.set()
        state.playing = False
        bus.emit("tts.finished")
        bus.emit("speaker.state", state="idle")

    # ----- Unified worker: handles both streaming synth and WAV playback -----

    def _worker_loop(self):
        """Single thread: pull work items, either stream-synth or play WAV."""
        from core.config import ALSA_PLAYBACK_DEVICE
        _set_volume()
        print(f"[speaker] Streaming worker started (aplay → {ALSA_PLAYBACK_DEVICE})")

        while not self._stop.is_set():
            try:
                item = self._work_q.get(timeout=0.5)
            except queue.Empty:
                continue

            # Unpack: new format (_SENTINEL, payload) or legacy (text, style)
            if isinstance(item, tuple) and len(item) == 2 and item[0] in (_SENTINEL_WAV, _SENTINEL_TEXT):
                kind, payload = item
            elif isinstance(item, tuple):
                # Legacy: (text, style) from old callers
                kind, payload = _SENTINEL_TEXT, item
            elif isinstance(item, str):
                # Legacy: bare WAV path
                kind, payload = _SENTINEL_WAV, item
            else:
                continue

            # Inviolable mute invariant: discard all work while muted
            if self._muted:
                continue

            if kind == _SENTINEL_WAV:
                self._play_wav(payload, ALSA_PLAYBACK_DEVICE)
            elif kind == _SENTINEL_TEXT:
                text, style = payload
                if not text or text.lower() in {"uh", "hmm", "um", "<silence>"}:
                    continue
                self._stream_synth(text, style, ALSA_PLAYBACK_DEVICE)

    @staticmethod
    def _emit_amplitude_envelope(wav_path: str, interrupt_event: threading.Event):
        """Pre-compute amplitude envelope from WAV and emit bus events synced to playback."""
        try:
            with wave.open(wav_path) as wf:
                sr = wf.getframerate()
                frames = wf.readframes(wf.getnframes())
                n_ch = wf.getnchannels()
            audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
            if n_ch > 1:
                audio = audio[::n_ch]  # take first channel

            # Compute RMS envelope in 30ms windows
            win = max(1, int(sr * 0.030))
            hop = max(1, int(sr * 0.020))  # 20ms hop → 50 updates/sec
            envelope = []
            for i in range(0, len(audio) - win, hop):
                rms = float(np.sqrt(np.mean(audio[i:i+win] ** 2)))
                envelope.append(rms)

            if not envelope:
                return

            # Normalize to 0–1
            peak = max(envelope)
            if peak > 1e-6:
                envelope = [v / peak for v in envelope]

            # Emit amplitude in real-time sync with playback
            t0 = time.perf_counter()
            hop_sec = hop / sr
            for idx, level in enumerate(envelope):
                if interrupt_event.is_set():
                    break
                target_time = t0 + idx * hop_sec
                wait = target_time - time.perf_counter()
                if wait > 0:
                    time.sleep(wait)
                bus.emit("tts.amplitude", level=level)

            # Reset to zero after playback
            bus.emit("tts.amplitude", level=0.0)
        except Exception:
            bus.emit("tts.amplitude", level=0.0)

    def _play_wav(self, wav_path: str, alsa_device: str):
        """Play a pre-rendered WAV file via aplay (interruptible)."""
        self._interrupted.clear()
        bus.emit("tts.started")
        bus.emit("speaker.state", state="playing")
        state.playing = True

        # Start amplitude envelope emitter in background
        amp_thread = threading.Thread(
            target=self._emit_amplitude_envelope,
            args=(wav_path, self._interrupted),
            daemon=True, name="amp-envelope",
        )
        amp_thread.start()

        try:
            proc = subprocess.Popen(
                ["aplay", "-D", alsa_device, wav_path],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            self._current_aplay = proc
            proc.wait()
            self._current_aplay = None
            time.sleep(0.05)
        except Exception as e:
            print(f"[speaker] WAV playback error: {e}")
        finally:
            self._current_aplay = None
            bus.emit("tts.amplitude", level=0.0)
            state.playing = False
            if not self._interrupted.is_set():
                bus.emit("tts.finished")
                bus.emit("speaker.state", state="idle")

    # Regex to split text into clauses at natural break points
    # Splits after sentence-ending punctuation or at commas/semicolons for long text
    _CLAUSE_SPLIT = re.compile(r'(?<=[.!?])\s+|(?<=[,;])\s+(?=\S{15,})')

    def _split_clauses(self, text: str) -> list[str]:
        """Split text into clause-sized chunks for pipelined synthesis.

        Short text (<80 chars) is kept whole to preserve natural prosody.
        Longer text is split at sentence boundaries (. ! ?) so that the
        first clause starts playing while subsequent clauses synthesize.
        """
        if len(text) < 80:
            return [text]

        # Split at sentence boundaries first
        parts = re.split(r'(?<=[.!?])\s+', text)

        # Merge very short fragments back into neighbors
        merged = []
        for p in parts:
            p = p.strip()
            if not p:
                continue
            if merged and len(merged[-1]) < 40:
                merged[-1] = merged[-1] + " " + p
            else:
                merged.append(p)

        # If last fragment is tiny, merge it back
        if len(merged) > 1 and len(merged[-1]) < 25:
            merged[-2] = merged[-2] + " " + merged[-1]
            merged.pop()

        return merged if merged else [text]

    # Rotating WAV slots for clause-level pipelining
    _CLAUSE_WAV_SLOTS = [f"/tmp/aura_clause_{i}.wav" for i in range(4)]

    def _stream_synth(self, text: str, style: str, alsa_device: str):
        """Synthesize text with clause-level pipelining.

        Splits text into clauses, synthesizes each to a WAV file, and plays
        via aplay. Synthesis of clause N+1 overlaps with playback of clause N
        using a background synth thread.

        If local XTTS isn't loaded yet and Farsight RTX is reachable, uses
        Farsight for synthesis (RTX is always warm — never restarted).
        """
        clean = re.sub(r"<[^>]+>", "", text).strip()
        clean = clean.replace("°C", "degrees Celsius").replace("°F", "degrees Fahrenheit")
        if not clean:
            return

        # If local TTS not ready, try Farsight RTX as fallback
        if not _local_tts_ready.is_set() and _check_farsight_once():
            self._stream_synth_farsight(clean, alsa_device)
            return

        tts = _get_xtts()

        clauses = self._split_clauses(clean)
        preview = clean[:70] + ("..." if len(clean) > 70 else "")
        t0 = time.perf_counter()

        bus.emit("tts.started")
        bus.emit("speaker.state", state="playing")
        state.playing = True

        # WAV queue: synth thread produces paths, main plays them
        wav_ready: queue.Queue = queue.Queue(maxsize=4)
        synth_done = threading.Event()
        synth_stats = {"clauses": 0, "samples": 0, "first_ms": 0.0}

        def _synth_clauses():
            """Background: synthesize each clause via XTTS v2 + DeepFilterNet."""
            slot = 0
            for clause in clauses:
                clause = clause.strip()
                if not clause:
                    continue
                out_path = self._CLAUSE_WAV_SLOTS[slot % len(self._CLAUSE_WAV_SLOTS)]
                slot += 1
                try:
                    out = tts.synthesizer.tts_model.inference(
                        clause, "en", _xtts_gpt_cond, _xtts_speaker_emb,
                        temperature=XTTS_TEMPERATURE,
                        repetition_penalty=XTTS_REP_PENALTY,
                        length_penalty=XTTS_LENGTH_PENALTY,
                    )
                    audio_np = np.array(out["wav"], dtype=np.float32)
                    if audio_np.size == 0:
                        continue

                    # Consistent RMS normalization + gain
                    audio_np = _normalize_audio(audio_np)

                    pcm16 = (audio_np * 32767.0).astype(np.int16)

                    with wave.open(out_path, "wb") as wf:
                        wf.setnchannels(1)
                        wf.setsampwidth(2)
                        wf.setframerate(XTTS_SAMPLE_RATE)
                        wf.writeframes(pcm16.tobytes())

                    if synth_stats["clauses"] == 0:
                        synth_stats["first_ms"] = (time.perf_counter() - t0) * 1000
                    synth_stats["clauses"] += 1
                    synth_stats["samples"] += len(pcm16)
                    wav_ready.put(out_path, timeout=60)

                except Exception as e:
                    print(f"[speaker] Clause synth error: {e}")
            synth_done.set()

        # Start synth in background
        synth_t = threading.Thread(target=_synth_clauses, daemon=True, name="clause-synth")
        synth_t.start()

        # Play WAVs as they arrive (clause N plays while N+1 synthesizes)
        self._interrupted.clear()
        try:
            while not synth_done.is_set() or not wav_ready.empty():
                if self._interrupted.is_set():
                    break
                try:
                    wav_path = wav_ready.get(timeout=0.5)
                except queue.Empty:
                    continue
                if self._interrupted.is_set():
                    break
                try:
                    amp_t = threading.Thread(
                        target=self._emit_amplitude_envelope,
                        args=(wav_path, self._interrupted),
                        daemon=True, name="amp-clause",
                    )
                    amp_t.start()
                    proc = subprocess.Popen(
                        ["aplay", "-D", alsa_device, wav_path],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
                    self._current_aplay = proc
                    proc.wait()
                    self._current_aplay = None
                except Exception as e:
                    print(f"[speaker] Clause play error: {e}")
        finally:
            self._current_aplay = None
            bus.emit("tts.amplitude", level=0.0)
            synth_t.join(timeout=30)
            state.playing = False
            if not self._interrupted.is_set():
                bus.emit("tts.finished")
                bus.emit("speaker.state", state="idle")

        total_ms = (time.perf_counter() - t0) * 1000
        audio_ms = synth_stats["samples"] / XTTS_SAMPLE_RATE * 1000 if synth_stats["samples"] else 0
        pending = self._work_q.qsize()
        pipeline_info = f" [pending={pending}]" if pending else ""
        print(f"[speaker] Pipelined: {total_ms:.0f}ms total, {audio_ms:.0f}ms audio, "
              f"first={synth_stats['first_ms']:.0f}ms, {synth_stats['clauses']} clauses"
              f" → \"{preview}\"{pipeline_info}")
        _diag_said(preview, synth_stats['first_ms'])

    # ----- Farsight RTX fallback path -----

    def _stream_synth_farsight(self, text: str, alsa_device: str):
        """Synthesize via Farsight RTX when local XTTS isn't loaded yet.

        The RTX is always warm (enterprise GPU, never restarted), so latency
        is just network round-trip + synthesis time (~200-500ms per clause).
        Local XTTS loads quietly in the background and takes over once ready.
        """
        t0 = time.perf_counter()
        preview = text[:70] + ("..." if len(text) > 70 else "")
        print(f"[speaker] Using Farsight RTX TTS (local warming up): \"{preview}\"")

        bus.emit("tts.started")
        bus.emit("speaker.state", state="playing")
        state.playing = True
        self._interrupted.clear()

        clauses = self._split_clauses(text)
        try:
            for i, clause in enumerate(clauses):
                if self._interrupted.is_set():
                    break
                clause = clause.strip()
                if not clause:
                    continue
                out_path = self._CLAUSE_WAV_SLOTS[i % len(self._CLAUSE_WAV_SLOTS)]
                ok = _farsight_synth_to_file(clause, out_path)
                if not ok:
                    print(f"[speaker] Farsight clause failed, waiting for local TTS...")
                    _local_tts_ready.wait(timeout=60)
                    _synth_to_file(clause, "neutral", out_path)
                if self._interrupted.is_set():
                    break
                # Play the clause
                try:
                    proc = subprocess.Popen(
                        ["aplay", "-D", alsa_device, out_path],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
                    self._current_aplay = proc
                    proc.wait()
                    self._current_aplay = None
                except Exception as e:
                    print(f"[speaker] Farsight clause play error: {e}")
        finally:
            self._current_aplay = None
            state.playing = False
            if not self._interrupted.is_set():
                bus.emit("tts.finished")
                bus.emit("speaker.state", state="idle")

        total_ms = (time.perf_counter() - t0) * 1000
        print(f"[speaker] Farsight TTS: {total_ms:.0f}ms total, {len(clauses)} clauses")

    # ----- Warmup -----

    def warmup(self):
        """Pre-load XTTS v2 (marks _local_tts_ready when done)."""
        try:
            _get_xtts()
            _local_tts_ready.set()
            print("[speaker] Warmup complete (XTTS ready)")
        except Exception as e:
            _local_tts_ready.set()  # don't block forever
            print(f"[speaker] Warmup failed (non-fatal): {e}")

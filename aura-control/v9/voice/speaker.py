"""
voice.speaker -- TTS queue → synthesis → playback.

Runs on a daemon thread.  Communicates via bus events:
    listens: "llm.sentence"   text=str
    emits:   "tts.started", "tts.finished"
             "speaker.state"  state=str ("idle"|"synthesizing"|"playing")

Supports ChatterboxTTS (local GPU).  Zero Qt imports.
Extracted from core/speaker.py.
"""

from __future__ import annotations

import hashlib
import inspect
import os
import pickle
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
from core.config import WORKSPACE_ROOT, VOICES_DIR, TTS_VOLUME
from core.state import state
from services.memlog import memlog

# ---------------------------------------------------------------------------
# .env (API keys live here)
# ---------------------------------------------------------------------------

_dotenv = WORKSPACE_ROOT / ".env"
if _dotenv.exists():
    load_dotenv(str(_dotenv))

# ---------------------------------------------------------------------------
# Voice cloning config — 8 style-specific reference samples
# ---------------------------------------------------------------------------

VOICE_REFS = {
    "neutral":    VOICES_DIR / "ref_neutral.wav",
    "warm":       VOICES_DIR / "ref_warm.wav",
    "assertive":  VOICES_DIR / "ref_assertive.wav",
    "empathy":    VOICES_DIR / "ref_empathy.wav",
    "playful":    VOICES_DIR / "ref_playful.wav",
    "energy":     VOICES_DIR / "ref_energy.wav",
    "soft":       VOICES_DIR / "ref_soft.wav",
    "technical":  VOICES_DIR / "ref_technical.wav",
}

STYLE_TO_REF = {
    "default": "neutral", "neutral": "neutral",
    "warm": "warm", "friendly": "warm",
    "assertive": "assertive", "empathy": "empathy",
    "playful": "playful", "energy": "energy",
    "soft": "soft", "technical": "technical",
}

# Per-style TTS tuning
# exaggeration: how expressive/emotional (0=flat, 0.7+=very animated)
# cfg_weight:   how closely to match the reference voice (0=ignore, 0.8+=tight clone)
# Higher values = warmer, more human, closer to the reference recordings.
STYLE_PARAMS = {
    "neutral":    dict(exaggeration=0.50, cfg_weight=0.70),
    "warm":       dict(exaggeration=0.65, cfg_weight=0.80),
    "empathy":    dict(exaggeration=0.70, cfg_weight=0.80),
    "assertive":  dict(exaggeration=0.55, cfg_weight=0.70),
    "technical":  dict(exaggeration=0.40, cfg_weight=0.65),
    "playful":    dict(exaggeration=0.75, cfg_weight=0.80),
    "energy":     dict(exaggeration=0.80, cfg_weight=0.85),
    "soft":       dict(exaggeration=0.55, cfg_weight=0.75),
}

# Legacy single-sample fallback (env var override)
CHATTERBOX_VOICE_SAMPLE = os.getenv("CHATTERBOX_VOICE_SAMPLE", "")

VOICE_CACHE_DIR = WORKSPACE_ROOT / "data" / "voice_cache"
VOICE_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Audio settings
# ---------------------------------------------------------------------------

PCM_SAMPLE_RATE = 22050

# ALSA output device detection
ALSA_CONTROLS = ["PCM", "Speaker", "Master"]


def _detect_output_device():
    """Auto-detect USB audio output device (UACDemoV1.0 preferred)."""
    try:
        out = subprocess.check_output(["aplay", "-l"], text=True, timeout=3)
        for line in out.splitlines():
            if "UACDemoV1.0" in line:
                m = re.search(r"card (\d+):", line)
                if m:
                    return "UACDemoV1.0", int(m.group(1))
            if "USB Audio" in line:
                m = re.search(r"card (\d+):", line)
                if m:
                    return "USB_Audio", int(m.group(1))
    except Exception:
        pass
    return None, None


OUTPUT_DEVICE_NAME, OUTPUT_CARD_INDEX = _detect_output_device()

# ---------------------------------------------------------------------------
# ChatterboxTTS (lazy-loaded)
# ---------------------------------------------------------------------------

_chatterbox = None
_chatterbox_is_turbo = False
_voice_embedding = None


def _get_chatterbox():
    global _chatterbox, _chatterbox_is_turbo
    if _chatterbox is not None:
        return _chatterbox
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    memlog.delta("speaker: before ChatterboxTTS load")
    # Prefer Turbo variant (same as v1 — better voice cloning)
    try:
        from chatterbox.tts_turbo import ChatterboxTurboTTS
        _chatterbox = ChatterboxTurboTTS.from_pretrained(device=device)
        _chatterbox_is_turbo = True
        print(f"[speaker] ChatterboxTurboTTS initialized (device={device})")
        memlog.delta("speaker: ChatterboxTurboTTS loaded")
        return _chatterbox
    except Exception as e:
        print(f"[speaker] Turbo TTS not available ({e}), falling back to standard")
        memlog.delta("speaker: Turbo failed, trying standard")
    # Fallback to standard ChatterboxTTS
    try:
        from chatterbox.tts import ChatterboxTTS
    except ImportError:
        from chatterbox import ChatterboxTTS
    try:
        _chatterbox = ChatterboxTTS.from_pretrained(device=device)
    except (AttributeError, TypeError):
        _chatterbox = ChatterboxTTS()
    print(f"[speaker] ChatterboxTTS initialized (device={device})")
    memlog.delta("speaker: ChatterboxTTS loaded")
    return _chatterbox


def _get_voice_embedding(cb, sample_path: str):
    """Load or create cached voice embedding."""
    global _voice_embedding
    if _voice_embedding is not None:
        return _voice_embedding

    st = os.stat(sample_path)
    key = hashlib.md5(f"{sample_path}:{st.st_mtime}:{st.st_size}".encode()).hexdigest()
    cache = VOICE_CACHE_DIR / f"voice_embedding_{key}.pkl"
    if cache.exists():
        try:
            _voice_embedding = pickle.loads(cache.read_bytes())
            return _voice_embedding
        except Exception:
            pass

    for fn_name in ("extract_voice_embedding", "get_voice_embedding"):
        fn = getattr(cb, fn_name, None)
        if fn:
            try:
                _voice_embedding = fn(sample_path)
                cache.write_bytes(pickle.dumps(_voice_embedding))
                return _voice_embedding
            except Exception:
                pass
    return None


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
# TTS generation
# ---------------------------------------------------------------------------

def _resolve_voice_sample(style: str) -> Optional[str]:
    """Pick the voice reference WAV for a given style."""
    # Env var override → use that single sample for everything
    if CHATTERBOX_VOICE_SAMPLE and os.path.exists(CHATTERBOX_VOICE_SAMPLE):
        return CHATTERBOX_VOICE_SAMPLE

    # Style-based: map to one of the 8 reference samples
    ref_key = STYLE_TO_REF.get(style.lower().strip(), "neutral")
    ref_path = VOICE_REFS.get(ref_key, VOICE_REFS["neutral"])
    if ref_path.exists():
        return str(ref_path)

    # Final fallback: try neutral
    neutral = VOICE_REFS["neutral"]
    if neutral.exists():
        return str(neutral)

    return None


def _generate_tts_audio(text: str, style: str = "neutral"):
    """Generate PCM int16 mono chunks from ChatterboxTTS."""
    cb = _get_chatterbox()
    clean = re.sub(r"<[^>]+>", "", text).strip()
    clean = clean.replace("°C", "degrees Celsius").replace("°F", "degrees Fahrenheit")
    if not clean:
        return

    # Per-style parameters (with bounded jitter for natural variation)
    sp = STYLE_PARAMS.get(style, STYLE_PARAMS["neutral"])
    base_ex = sp.get("exaggeration", 0.50)
    base_cfg = sp.get("cfg_weight", 0.70)
    base_temp = float(os.getenv("CHATTERBOX_BASE_TEMPERATURE", "0.85"))
    base_top_p = float(os.getenv("CHATTERBOX_BASE_TOP_P", "0.95"))
    base_pace = float(os.getenv("CHATTERBOX_BASE_PACE", "1.0"))

    def jitter(v, amt):
        return max(0.0, v * random.uniform(1.0 - amt, 1.0 + amt))

    # Build kwargs based on what this ChatterboxTTS build supports
    synth_fn = getattr(cb, "generate", None) or cb.synthesize
    try:
        sig = inspect.signature(synth_fn)
        supported = set(sig.parameters.keys())
    except Exception:
        supported = set()

    # Log supported params (first call only)
    if not hasattr(_generate_tts_audio, "_logged"):
        print(f"[speaker] ChatterboxTTS.generate params: {sorted(supported)}")
        _generate_tts_audio._logged = True

    kwargs = {}
    if "exaggeration" in supported:
        kwargs["exaggeration"] = jitter(base_ex, 0.08)
    if "cfg_weight" in supported:
        kwargs["cfg_weight"] = jitter(base_cfg, 0.08)
    if "temperature" in supported:
        kwargs["temperature"] = jitter(base_temp, 0.08)
    if "top_p" in supported:
        kwargs["top_p"] = jitter(base_top_p, 0.03)
    if "pace" in supported:
        kwargs["pace"] = jitter(base_pace, 0.03)
    elif "speed" in supported:
        kwargs["speed"] = jitter(base_pace, 0.03)
    if "norm_loudness" in supported:
        kwargs["norm_loudness"] = True
    if "repetition_penalty" in supported:
        kwargs["repetition_penalty"] = 1.15   # reduce monotone repetitive patterns

    # Voice cloning — pick reference sample by style
    if state.chatterbox_voice_cloning_enabled:
        sample = _resolve_voice_sample(style)
        if sample:
            emb = _get_voice_embedding(cb, sample)
            if emb is not None and ("voice_embedding" in supported or "embedding" in supported):
                key = "voice_embedding" if "voice_embedding" in supported else "embedding"
                kwargs[key] = emb
                print(f"[speaker] Voice cloning: using cached embedding for {style}")
            elif "audio_prompt_path" in supported:
                kwargs["audio_prompt_path"] = sample
                print(f"[speaker] Voice cloning: using audio_prompt_path={os.path.basename(sample)}")
            else:
                print(f"[speaker] Voice cloning: no supported method (supported={sorted(supported)})")
        else:
            print(f"[speaker] Voice cloning: no sample found for style={style}")

    print(f"[speaker] Synthesis kwargs: {list(kwargs.keys())}")
    try:
        audio = synth_fn(clean, **kwargs)
    except TypeError:
        print("[speaker] Synthesis with kwargs failed, retrying without")
        audio = synth_fn(clean)

    # Convert to PCM int16 mono
    try:
        import torch
        if isinstance(audio, torch.Tensor):
            audio_np = audio.detach().float().cpu().numpy()
        else:
            audio_np = np.asarray(audio, dtype=np.float32)
    except ImportError:
        audio_np = np.asarray(audio, dtype=np.float32)

    audio_np = np.squeeze(audio_np).astype(np.float32)

    # RMS normalization + peak clamp
    rms = float(np.sqrt(np.mean(audio_np * audio_np))) if audio_np.size else 0.0
    if rms > 1e-8:
        gain = min(0.18 / rms, 60.0)
        audio_np *= gain

    peak = float(np.max(np.abs(audio_np))) if audio_np.size else 0.0
    if peak > 1e-8:
        audio_np = audio_np / peak * 0.98

    audio_np = np.clip(audio_np, -1.0, 1.0)
    pcm16 = (audio_np * 32767.0).astype(np.int16)

    CHUNK = 2048
    for i in range(0, len(pcm16), CHUNK):
        yield pcm16[i:i + CHUNK].tobytes()


# ---------------------------------------------------------------------------
# Volume control
# ---------------------------------------------------------------------------

_volume_set = False


def _set_volume():
    """Set playback volume (PulseAudio preferred, ALSA fallback)."""
    global _volume_set
    if _volume_set:
        return
    vol_pct = int(TTS_VOLUME * 100) if TTS_VOLUME <= 2.0 else int(TTS_VOLUME)

    # PulseAudio
    try:
        result = subprocess.run(
            ["pactl", "get-default-sink"],
            capture_output=True, text=True, check=True, timeout=2,
        )
        sink = result.stdout.strip()
        if sink:
            subprocess.run(
                ["pactl", "set-sink-volume", sink, f"{vol_pct}%"],
                check=True, timeout=2,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            _volume_set = True
            return
    except Exception:
        pass

    # ALSA fallback
    if OUTPUT_CARD_INDEX is not None:
        for ctrl in ALSA_CONTROLS:
            try:
                subprocess.run(
                    ["amixer", "-c", str(OUTPUT_CARD_INDEX), "sset", ctrl, f"{vol_pct}%"],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                    check=True, timeout=2,
                )
                _volume_set = True
                return
            except Exception:
                continue


# ---------------------------------------------------------------------------
# WAV slot pool (for pipelined synthesis — rotate so playback and synthesis
# never collide on the same file)
# ---------------------------------------------------------------------------

_TTS_WAV_SLOTS = [f"/tmp/aura_tts_slot{i}.wav" for i in range(4)]


def _synth_to_file(text: str, style: str, out_path: str) -> float:
    """Synthesize *text* to a WAV file. Returns wall-clock ms."""
    t0 = time.perf_counter()
    stream = _generate_tts_audio(text, style=style)
    first_chunk = next(stream, None)
    if not first_chunk:
        return 0.0

    with wave.open(out_path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(PCM_SAMPLE_RATE)
        wf.writeframes(first_chunk)
        for chunk in stream:
            if chunk:
                wf.writeframes(chunk)
        # 250ms silence tail to prevent end clipping
        tail = int(PCM_SAMPLE_RATE * 0.25)
        wf.writeframes(b"\x00\x00" * tail)

    return (time.perf_counter() - t0) * 1000


# ---------------------------------------------------------------------------
# Speaker class — pipelined: synth N+1 while playing N
# ---------------------------------------------------------------------------

class Speaker:
    """Pipelined TTS playback: sentence queue → synth thread → play thread.

    A background synth thread pulls sentences and writes WAV files to rotating
    slots.  The main speaker thread plays each WAV as soon as it's ready, so
    synthesis of the *next* sentence overlaps with playback of the current one.
    """

    # Cycle through all 8 voice styles for LLM responses — each uses a
    # different reference WAV for natural variety across sentences.
    _LLM_STYLES = [
        "neutral", "warm", "empathy", "soft",
        "energy", "assertive", "technical", "playful",
    ]

    def __init__(self) -> None:
        self._sentence_q: queue.Queue = queue.Queue()   # (text, style) from LLM / enqueue
        self._wav_q: queue.Queue = queue.Queue(maxsize=2)  # ready WAV paths
        self._play_thread: Optional[threading.Thread] = None
        self._synth_thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._style_idx = 0  # for cycling LLM response styles

        bus.on("llm.sentence", self._on_sentence)

    def _on_sentence(self, text: str = "", style: str = "", **_kw):
        text = preprocess(text)
        if text and not re.match(r"^[\s.,!?]+$", text):
            # If no explicit style provided (LLM responses), cycle through refs
            if not style:
                style = self._LLM_STYLES[self._style_idx % len(self._LLM_STYLES)]
                self._style_idx += 1
            self._sentence_q.put((text, style))

    def enqueue(self, text: str, style: str = "neutral"):
        """Manually enqueue text (for intro prompts, etc.)."""
        text = preprocess(text)
        if text:
            self._sentence_q.put((text, style))

    def enqueue_wav(self, wav_path: str):
        """Push a pre-synthesized WAV directly to the playback queue.

        Bypasses the synthesis thread entirely — used for welcome sentences
        that were pre-synthesized during boot warmup.
        """
        if os.path.exists(wav_path):
            self._wav_q.put(wav_path, timeout=30)
        else:
            print(f"[speaker] Pre-synth WAV not found: {wav_path}")

    # ----- Control -----

    def start(self):
        if self._play_thread and self._play_thread.is_alive():
            return
        self._stop.clear()
        self._synth_thread = threading.Thread(
            target=self._synth_loop, daemon=True, name="speaker-synth",
        )
        self._play_thread = threading.Thread(
            target=self._play_loop, daemon=True, name="speaker-play",
        )
        self._synth_thread.start()
        self._play_thread.start()

    def stop(self):
        self._stop.set()

    # ----- Synth thread: sentence queue → WAV files → wav queue -----

    def _synth_loop(self):
        """Pull sentences, synthesize to rotating WAV slots, push to play queue."""
        print("[speaker] Synth loop started")
        slot = 0
        while not self._stop.is_set():
            try:
                item = self._sentence_q.get(timeout=0.5)
            except queue.Empty:
                continue

            if isinstance(item, tuple):
                sentence, style = item
            else:
                sentence, style = item, "neutral"

            if not sentence or sentence.lower() in {"uh", "hmm", "um", "<silence>"}:
                continue

            out_path = _TTS_WAV_SLOTS[slot % len(_TTS_WAV_SLOTS)]
            slot += 1

            preview = sentence[:70] + ("..." if len(sentence) > 70 else "")
            print(f"[speaker] Synthesizing ({style}): \"{preview}\"")

            try:
                ms = _synth_to_file(sentence, style, out_path)
                if ms > 0:
                    queued = self._wav_q.qsize()
                    pending = self._sentence_q.qsize()
                    pipeline_info = f" [play_q={queued}, pending={pending}]" if (queued or pending) else ""
                    print(f"[speaker] Synth done: {ms:.0f}ms → \"{preview}\"{pipeline_info}")
                    self._wav_q.put(out_path, timeout=60)
                else:
                    print(f"[speaker] Synth produced no audio: \"{preview}\"")
            except Exception as e:
                import traceback
                print(f"[speaker] Synth error: {e}")
                traceback.print_exc()

    # ----- Play thread: wav queue → aplay (direct ALSA) -----

    def _play_loop(self):
        """Play WAVs as they arrive from the synth thread."""
        from core.config import ALSA_PLAYBACK_DEVICE
        _set_volume()
        print(f"[speaker] Playback loop started (aplay → {ALSA_PLAYBACK_DEVICE})")

        while not self._stop.is_set():
            try:
                wav_path = self._wav_q.get(timeout=0.5)
            except queue.Empty:
                continue

            bus.emit("tts.started")
            bus.emit("speaker.state", state="playing")
            state.playing = True
            try:
                subprocess.run(
                    ["aplay", "-D", ALSA_PLAYBACK_DEVICE, wav_path],
                    check=False,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                time.sleep(0.15)  # drain time for PipeWire/Pulse
            except Exception as e:
                print(f"[speaker] Playback error: {e}")
            finally:
                state.playing = False
                bus.emit("tts.finished")
                bus.emit("speaker.state", state="idle")

    # ----- Warmup -----

    def warmup(self, style: str = "neutral"):
        """Pre-load ChatterboxTTS and optionally prime voice prompt."""
        try:
            cb = _get_chatterbox()
            kw = {}
            sample = _resolve_voice_sample(style)
            if sample:
                synth_fn = getattr(cb, "generate", None) or cb.synthesize
                try:
                    sig = inspect.signature(synth_fn)
                    if "audio_prompt_path" in sig.parameters:
                        kw["audio_prompt_path"] = sample
                except Exception:
                    pass
            try:
                synth_fn = getattr(cb, "generate", None) or cb.synthesize
                synth_fn("Hello.", **kw)
            except Exception:
                pass
            print("[speaker] Warmup complete")
        except Exception as e:
            print(f"[speaker] Warmup failed (non-fatal): {e}")

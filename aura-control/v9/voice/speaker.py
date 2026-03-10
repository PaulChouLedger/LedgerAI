"""
voice.speaker -- TTS queue → Kokoro synthesis → streaming playback.

Runs on a daemon thread.  Communicates via bus events:
    listens: "llm.sentence"   text=str
    emits:   "tts.started", "tts.finished"
             "speaker.state"  state=str ("idle"|"synthesizing"|"playing")

Uses Kokoro TTS (82M parameter model, 24kHz output).
Streams audio chunks directly to aplay stdin — playback begins from the
first phoneme segment (~100-200ms) instead of waiting for full synthesis.
Zero Qt imports.
"""

from __future__ import annotations

import glob as _glob
import os
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
from core.config import WORKSPACE_ROOT, TTS_VOLUME, VOICES_DIR
from core.state import state
from services.memlog import memlog

# ---------------------------------------------------------------------------
# .env (API keys live here)
# ---------------------------------------------------------------------------

_dotenv = WORKSPACE_ROOT / ".env"
if _dotenv.exists():
    load_dotenv(str(_dotenv))

# ---------------------------------------------------------------------------
# Kokoro TTS config
# ---------------------------------------------------------------------------

# Custom voice pack: style vectors fine-tuned on actress samples via RTX training
_CUSTOM_VOICE_PATH = str(VOICES_DIR.parent / "aura_actress.pt")
# Full fine-tuned model: decoder + predictor weights (Level 3 training)
_FULL_FINETUNE_PATH = str(VOICES_DIR.parent / "aura_full.pt")
KOKORO_VOICE = os.environ.get("AURA_KOKORO_VOICE", _CUSTOM_VOICE_PATH if os.path.exists(_CUSTOM_VOICE_PATH) else "af_heart")
KOKORO_SPEED = float(os.environ.get("AURA_KOKORO_SPEED", "1.0"))
KOKORO_SAMPLE_RATE = 24000

# ---------------------------------------------------------------------------
# Kokoro TTS (lazy-loaded)
# ---------------------------------------------------------------------------

_kokoro_pipe = None


def _get_kokoro():
    """Lazy-load the Kokoro pipeline (downloads model on first use).

    Runs on CPU to avoid GPU contention with LLM/Whisper containers.
    The 82M model is small enough for fast CPU inference on Jetson.
    """
    global _kokoro_pipe
    if _kokoro_pipe is not None:
        return _kokoro_pipe
    memlog.delta("speaker: before Kokoro load")
    import torch
    from kokoro import KPipeline
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _kokoro_pipe = KPipeline(lang_code="a", repo_id="hexgrad/Kokoro-82M")
    if device == "cuda" and hasattr(_kokoro_pipe, "model") and _kokoro_pipe.model is not None:
        try:
            _kokoro_pipe.model = _kokoro_pipe.model.to(device)
            print(f"[speaker] Kokoro TTS initialized on CUDA (voice={KOKORO_VOICE})")
        except RuntimeError as e:
            print(f"[speaker] CUDA failed ({e}), falling back to CPU")
            _kokoro_pipe.model = _kokoro_pipe.model.cpu()
    else:
        print(f"[speaker] Kokoro TTS initialized on {device} (voice={KOKORO_VOICE})")

    # NOTE: Full decoder/predictor fine-tune (Level 3) produced garbled output
    # (peak amplitude 10x lower than stock, audio artifacts). The training loss
    # converged but the model degraded — likely needs longer training or different
    # hyperparams. For now, use stock Kokoro + custom style vectors only.
    # The style vectors alone capture the actress's timbre effectively.
    # if os.path.exists(_FULL_FINETUNE_PATH):
    #     ckpt = torch.load(_FULL_FINETUNE_PATH, ...)
    #     _kokoro_pipe.model.decoder.load_state_dict(ckpt["decoder"])
    #     _kokoro_pipe.model.predictor.load_state_dict(ckpt["predictor"])

    memlog.delta("speaker: Kokoro loaded")
    return _kokoro_pipe


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


def _set_volume():
    """Set playback volume (ALSA amixer)."""
    global _volume_set
    if _volume_set:
        return
    vol_pct = int(TTS_VOLUME * 100) if TTS_VOLUME <= 2.0 else int(TTS_VOLUME)

    for ctrl in ("PCM", "Speaker", "Master"):
        try:
            subprocess.run(
                ["amixer", "sset", ctrl, f"{vol_pct}%"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                check=True, timeout=2,
            )
            _volume_set = True
            return
        except Exception:
            continue


# ---------------------------------------------------------------------------
# WAV file synthesis (kept for external callers like boot tour)
# ---------------------------------------------------------------------------

def _synth_to_file(text: str, style: str, out_path: str) -> float:
    """Synthesize *text* to a WAV file. Returns wall-clock ms."""
    pipe = _get_kokoro()
    clean = re.sub(r"<[^>]+>", "", text).strip()
    clean = clean.replace("°C", "degrees Celsius").replace("°F", "degrees Fahrenheit")
    if not clean:
        return 0.0

    t0 = time.perf_counter()
    all_audio = []
    for _gs, _ps, audio in pipe(clean, voice=KOKORO_VOICE, speed=KOKORO_SPEED):
        if audio is not None and len(audio) > 0:
            all_audio.append(audio)

    if not all_audio:
        return 0.0

    audio_np = np.concatenate(all_audio).astype(np.float32)
    peak = float(np.max(np.abs(audio_np))) if audio_np.size else 0.0
    if peak > 1e-8:
        audio_np = audio_np / peak * 0.95
    audio_np = np.clip(audio_np, -1.0, 1.0)
    pcm16 = (audio_np * 32767.0).astype(np.int16)

    with wave.open(out_path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(KOKORO_SAMPLE_RATE)
        wf.writeframes(pcm16.tobytes())
        tail = int(KOKORO_SAMPLE_RATE * 0.25)
        wf.writeframes(b"\x00\x00" * tail)

    return (time.perf_counter() - t0) * 1000


# ---------------------------------------------------------------------------
# Speaker class — streaming: Kokoro chunks → aplay stdin (no WAV files)
# ---------------------------------------------------------------------------

# Sentinel objects for the unified work queue
_SENTINEL_WAV = "wav"       # item is a WAV file path
_SENTINEL_TEXT = "text"     # item is (text, style) to synthesize+stream


class Speaker:
    """Streaming TTS playback: Kokoro chunks pipe directly to aplay stdin.

    Architecture:
      - Single worker thread pulls from a unified queue
      - For text: opens aplay with raw PCM stdin, streams Kokoro chunks
        as they arrive — playback starts from the first phoneme segment
      - For WAV files: plays via aplay (thinking fillers, tour, briefings)
      - Sentence N+1 synthesis begins as soon as N finishes playing
    """

    # Pre-generated thinking fillers (loaded once at import)
    _THINKING_DIR = WORKSPACE_ROOT / "assets" / "thinking_fillers"
    _thinking_wavs: list[str] = sorted(_glob.glob(str(_THINKING_DIR / "think_*.wav")))

    def __init__(self) -> None:
        # Unified work queue: items are (_SENTINEL_WAV, path) or (_SENTINEL_TEXT, (text, style))
        self._work_q: queue.Queue = queue.Queue()
        self._worker_thread: Optional[threading.Thread] = None
        self._stop = threading.Event()

        # Keep legacy queues as thin wrappers for external code that checks them
        self._sentence_q = self._work_q  # compatibility alias
        self._wav_q = self._work_q       # compatibility alias

        bus.on("llm.sentence", self._on_sentence)

    def _on_sentence(self, text: str = "", style: str = "", **_kw):
        text = preprocess(text)
        if text and not re.match(r"^[\s.,!?]+$", text):
            self._work_q.put((_SENTINEL_TEXT, (text, style or "neutral")))

    def enqueue(self, text: str, style: str = "neutral"):
        """Manually enqueue text (for intro prompts, etc.)."""
        text = preprocess(text)
        if text:
            self._work_q.put((_SENTINEL_TEXT, (text, style)))

    def enqueue_wav(self, wav_path: str):
        """Push a pre-synthesized WAV directly to the playback queue."""
        if os.path.exists(wav_path):
            self._work_q.put((_SENTINEL_WAV, wav_path))
        else:
            print(f"[speaker] Pre-synth WAV not found: {wav_path}")

    def play_thinking_filler(self):
        """Queue a random pre-generated thinking filler for instant playback."""
        if not self._thinking_wavs:
            return
        wav = random.choice(self._thinking_wavs)
        print(f"[speaker] Thinking filler: {os.path.basename(wav)}")
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

            if kind == _SENTINEL_WAV:
                self._play_wav(payload, ALSA_PLAYBACK_DEVICE)
            elif kind == _SENTINEL_TEXT:
                text, style = payload
                if not text or text.lower() in {"uh", "hmm", "um", "<silence>"}:
                    continue
                self._stream_synth(text, style, ALSA_PLAYBACK_DEVICE)

    def _play_wav(self, wav_path: str, alsa_device: str):
        """Play a pre-rendered WAV file via aplay."""
        bus.emit("tts.started")
        bus.emit("speaker.state", state="playing")
        state.playing = True
        try:
            subprocess.run(
                ["aplay", "-D", alsa_device, wav_path],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            time.sleep(0.05)
        except Exception as e:
            print(f"[speaker] WAV playback error: {e}")
        finally:
            state.playing = False
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
        """
        pipe = _get_kokoro()
        clean = re.sub(r"<[^>]+>", "", text).strip()
        clean = clean.replace("°C", "degrees Celsius").replace("°F", "degrees Fahrenheit")
        if not clean:
            return

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
            """Background: synthesize each clause to a rotating WAV slot."""
            slot = 0
            for clause in clauses:
                clause = clause.strip()
                if not clause:
                    continue
                out_path = self._CLAUSE_WAV_SLOTS[slot % len(self._CLAUSE_WAV_SLOTS)]
                slot += 1
                try:
                    all_audio = []
                    for _gs, _ps, audio in pipe(clause, voice=KOKORO_VOICE, speed=KOKORO_SPEED):
                        if audio is not None and len(audio) > 0:
                            all_audio.append(audio)
                    if not all_audio:
                        continue

                    audio_np = np.concatenate(all_audio).astype(np.float32)
                    peak = float(np.max(np.abs(audio_np))) if audio_np.size else 0.0
                    if peak > 1e-8:
                        audio_np = audio_np / peak * 0.95
                    audio_np = np.clip(audio_np, -1.0, 1.0)
                    pcm16 = (audio_np * 32767.0).astype(np.int16)

                    with wave.open(out_path, "wb") as wf:
                        wf.setnchannels(1)
                        wf.setsampwidth(2)
                        wf.setframerate(KOKORO_SAMPLE_RATE)
                        wf.writeframes(pcm16.tobytes())
                        # Small silence tail
                        wf.writeframes(b"\x00\x00" * int(KOKORO_SAMPLE_RATE * 0.1))

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
        try:
            while not synth_done.is_set() or not wav_ready.empty():
                try:
                    wav_path = wav_ready.get(timeout=0.5)
                except queue.Empty:
                    continue
                try:
                    subprocess.run(
                        ["aplay", "-D", alsa_device, wav_path],
                        check=False,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
                except Exception as e:
                    print(f"[speaker] Clause play error: {e}")
        finally:
            synth_t.join(timeout=30)
            state.playing = False
            bus.emit("tts.finished")
            bus.emit("speaker.state", state="idle")

        total_ms = (time.perf_counter() - t0) * 1000
        audio_ms = synth_stats["samples"] / KOKORO_SAMPLE_RATE * 1000 if synth_stats["samples"] else 0
        pending = self._work_q.qsize()
        pipeline_info = f" [pending={pending}]" if pending else ""
        print(f"[speaker] Pipelined: {total_ms:.0f}ms total, {audio_ms:.0f}ms audio, "
              f"first={synth_stats['first_ms']:.0f}ms, {synth_stats['clauses']} clauses"
              f" → \"{preview}\"{pipeline_info}")

    # ----- Warmup -----

    def warmup(self):
        """Pre-load Kokoro model."""
        try:
            _get_kokoro()
            print("[speaker] Warmup complete")
        except Exception as e:
            print(f"[speaker] Warmup failed (non-fatal): {e}")

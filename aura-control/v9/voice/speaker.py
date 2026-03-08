"""
voice.speaker -- TTS queue → Kokoro synthesis → playback.

Runs on a daemon thread.  Communicates via bus events:
    listens: "llm.sentence"   text=str
    emits:   "tts.started", "tts.finished"
             "speaker.state"  state=str ("idle"|"synthesizing"|"playing")

Uses Kokoro TTS (82M parameter model, 24kHz output).
Zero Qt imports.
"""

from __future__ import annotations

import os
import queue
import re
import subprocess
import threading
import time
import wave
from typing import Optional

import numpy as np
from dotenv import load_dotenv

from core.bus import bus
from core.config import WORKSPACE_ROOT, TTS_VOLUME
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

# Single consistent voice for the entire system
KOKORO_VOICE = os.environ.get("AURA_KOKORO_VOICE", "af_heart")
KOKORO_SPEED = float(os.environ.get("AURA_KOKORO_SPEED", "1.0"))
KOKORO_SAMPLE_RATE = 24000

# ---------------------------------------------------------------------------
# Kokoro TTS (lazy-loaded)
# ---------------------------------------------------------------------------

_kokoro_pipe = None


def _get_kokoro():
    """Lazy-load the Kokoro pipeline (downloads model on first use)."""
    global _kokoro_pipe
    if _kokoro_pipe is not None:
        return _kokoro_pipe
    memlog.delta("speaker: before Kokoro load")
    from kokoro import KPipeline
    _kokoro_pipe = KPipeline(lang_code="a", repo_id="hexgrad/Kokoro-82M")
    print(f"[speaker] Kokoro TTS initialized (voice={KOKORO_VOICE})")
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
# TTS generation
# ---------------------------------------------------------------------------

def _generate_tts_audio(text: str, style: str = "neutral"):
    """Generate PCM int16 mono audio from Kokoro TTS.

    Style parameter is accepted for API compatibility but Kokoro uses
    a single consistent voice (KOKORO_VOICE) for all output.
    """
    pipe = _get_kokoro()
    clean = re.sub(r"<[^>]+>", "", text).strip()
    clean = clean.replace("°C", "degrees Celsius").replace("°F", "degrees Fahrenheit")
    if not clean:
        return

    # Kokoro generates audio in chunks (one per grapheme-phoneme segment)
    all_audio = []
    for _gs, _ps, audio in pipe(clean, voice=KOKORO_VOICE, speed=KOKORO_SPEED):
        if audio is not None and len(audio) > 0:
            all_audio.append(audio)

    if not all_audio:
        return

    audio_np = np.concatenate(all_audio).astype(np.float32)

    # Normalize
    peak = float(np.max(np.abs(audio_np))) if audio_np.size else 0.0
    if peak > 1e-8:
        audio_np = audio_np / peak * 0.95

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
        wf.setframerate(KOKORO_SAMPLE_RATE)
        wf.writeframes(first_chunk)
        for chunk in stream:
            if chunk:
                wf.writeframes(chunk)
        # 250ms silence tail to prevent end clipping
        tail = int(KOKORO_SAMPLE_RATE * 0.25)
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

    def __init__(self) -> None:
        self._sentence_q: queue.Queue = queue.Queue()   # (text, style) from LLM / enqueue
        self._wav_q: queue.Queue = queue.Queue(maxsize=2)  # ready WAV paths
        self._play_thread: Optional[threading.Thread] = None
        self._synth_thread: Optional[threading.Thread] = None
        self._stop = threading.Event()

        bus.on("llm.sentence", self._on_sentence)

    def _on_sentence(self, text: str = "", style: str = "", **_kw):
        text = preprocess(text)
        if text and not re.match(r"^[\s.,!?]+$", text):
            self._sentence_q.put((text, style or "neutral"))

    def enqueue(self, text: str, style: str = "neutral"):
        """Manually enqueue text (for intro prompts, etc.)."""
        text = preprocess(text)
        if text:
            self._sentence_q.put((text, style))

    def enqueue_wav(self, wav_path: str):
        """Push a pre-synthesized WAV directly to the playback queue."""
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
        print("[speaker] Synth loop started (Kokoro TTS)")
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
            print(f"[speaker] Synthesizing: \"{preview}\"")

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
                time.sleep(0.05)
            except Exception as e:
                print(f"[speaker] Playback error: {e}")
            finally:
                state.playing = False
                bus.emit("tts.finished")
                bus.emit("speaker.state", state="idle")

    # ----- Warmup -----

    def warmup(self):
        """Pre-load Kokoro model."""
        try:
            _get_kokoro()
            print("[speaker] Warmup complete")
        except Exception as e:
            print(f"[speaker] Warmup failed (non-fatal): {e}")

"""
voice.speaker -- TTS queue → Kokoro synthesis → RVC voice conversion → playback.

Runs on a daemon thread.  Communicates via bus events:
    listens: "llm.sentence"   text=str
    emits:   "tts.started", "tts.finished"
             "speaker.state"  state=str ("idle"|"synthesizing"|"playing")

Uses Kokoro TTS (82M parameter model, 24kHz output) for speech synthesis,
then RVC (Retrieval-based Voice Conversion) to convert to the target voice.
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
from core.config import WORKSPACE_ROOT, TTS_VOLUME, VOICES_DIR, FARSIGHT_URL
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
# RVC voice conversion config (post-processing after Kokoro)
# ---------------------------------------------------------------------------

_RVC_MODEL_PATH = str(VOICES_DIR.parent / "rvc" / "aura_olga.pth")
_RVC_INDEX_PATH = str(VOICES_DIR.parent / "rvc" / "aura_olga.index")
RVC_ENABLED = os.environ.get("AURA_RVC_ENABLED", "1") == "1" and os.path.exists(_RVC_MODEL_PATH)

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
# RVC voice conversion (lazy-loaded)
# ---------------------------------------------------------------------------

_rvc_engine = None


def _get_rvc():
    """Lazy-load the RVC voice conversion model."""
    global _rvc_engine
    if _rvc_engine is not None:
        return _rvc_engine
    if not RVC_ENABLED:
        return None
    try:
        memlog.delta("speaker: before RVC load")
        from rvc_python.infer import RVCInference
        _rvc_engine = RVCInference(device="cuda")
        _rvc_engine.load_model(
            _RVC_MODEL_PATH,
            version="v2",
            index_path=_RVC_INDEX_PATH if os.path.exists(_RVC_INDEX_PATH) else "",
        )
        _rvc_engine.set_params(
            f0method="rmvpe",
            f0up_key=0,
            index_rate=0.75,
            protect=0.33,
        )
        print(f"[speaker] RVC voice conversion loaded (model={os.path.basename(_RVC_MODEL_PATH)})")
        memlog.delta("speaker: RVC loaded")
    except Exception as e:
        print(f"[speaker] RVC load failed (will use Kokoro only): {e}")
        _rvc_engine = None
    return _rvc_engine


_RVC_VOLUME = float(os.environ.get("AURA_RVC_VOLUME", "0.30"))  # RVC output is very hot


def _rvc_denoise(audio: np.ndarray, sr: int) -> np.ndarray:
    """Spectral noise gate: estimate noise floor from quiet frames,
    then suppress frequency bins below the noise floor."""
    from scipy.signal import stft, istft

    nperseg = 1024
    f, t, Zxx = stft(audio, fs=sr, nperseg=nperseg)

    magnitude = np.abs(Zxx)
    # Estimate noise floor from the quietest 15% of frames
    frame_energy = np.mean(magnitude, axis=0)
    thresh_idx = max(1, int(len(frame_energy) * 0.15))
    quietest = np.argsort(frame_energy)[:thresh_idx]
    noise_profile = np.mean(magnitude[:, quietest], axis=1, keepdims=True)

    # Spectral gate: attenuate bins below 3x noise floor (aggressive)
    gate = np.clip((magnitude - 3.0 * noise_profile) / (magnitude + 1e-10), 0, 1)
    Zxx_clean = Zxx * gate

    _, cleaned = istft(Zxx_clean, fs=sr, nperseg=nperseg)
    # Match original length
    if len(cleaned) < len(audio):
        cleaned = np.pad(cleaned, (0, len(audio) - len(cleaned)))
    return cleaned[: len(audio)]


def _rvc_convert(wav_path: str) -> str:
    """Run RVC voice conversion on a WAV file. Applies spectral denoising
    and volume scaling to suppress hiss introduced by RVC."""
    rvc = _get_rvc()
    if rvc is None:
        return wav_path
    out_path = wav_path.replace(".wav", "_rvc.wav")
    try:
        rvc.infer_file(wav_path, out_path)

        with wave.open(out_path, "rb") as r:
            params = r.getparams()
            pcm = np.frombuffer(r.readframes(r.getnframes()), dtype=np.int16)

        audio = pcm.astype(np.float32)
        audio = _rvc_denoise(audio, params.framerate)
        audio = audio * _RVC_VOLUME
        pcm_out = np.clip(audio, -32768, 32767).astype(np.int16)

        with wave.open(out_path, "wb") as w:
            w.setparams(params)
            w.writeframes(pcm_out.tobytes())

        return out_path
    except Exception as e:
        print(f"[speaker] RVC conversion failed: {e}")
        return wav_path  # fallback to unconverted


# ---------------------------------------------------------------------------
# Local model readiness — tracks whether Kokoro+RVC are loaded
# ---------------------------------------------------------------------------

_local_tts_ready = threading.Event()


def is_local_tts_ready() -> bool:
    """Check if the local TTS pipeline (Kokoro+RVC) is loaded and warm."""
    return _local_tts_ready.is_set()


def warm_local_tts_background():
    """Load Kokoro+RVC in a background thread. Non-blocking.

    The thread sets _local_tts_ready once both models are in VRAM.
    Speaker._stream_synth will use Farsight as fallback until this completes.
    """
    def _warm():
        try:
            _get_kokoro()
            if RVC_ENABLED:
                _get_rvc()
            _local_tts_ready.set()
            print("[speaker] Local TTS pipeline warm (Kokoro+RVC ready)")
        except Exception as e:
            print(f"[speaker] Local TTS warmup failed: {e}")
            # Still mark as ready so we don't block forever — will error on use
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
        # Run through local RVC if available (Farsight voice != Olga)
        if RVC_ENABLED and _rvc_engine is not None:
            rvc_out = _rvc_convert(out_path)
            if rvc_out != out_path:
                os.replace(rvc_out, out_path)
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

def _kokoro_phonetic_fix(text: str) -> str:
    """Fix Kokoro TTS mispronunciations via phonetic substitutions."""
    # Kokoro says "Aurura" for "Aura" — use phonetic spelling
    text = re.sub(r'\bAura\b', 'Awe-ruh', text)
    text = re.sub(r'\baura\b', 'awe-ruh', text)
    text = re.sub(r'\bAURA\b', 'AWE-RUH', text)
    return text


def _synth_to_file(text: str, style: str, out_path: str) -> float:
    """Synthesize *text* to a WAV file. Returns wall-clock ms."""
    pipe = _get_kokoro()
    clean = re.sub(r"<[^>]+>", "", text).strip()
    clean = clean.replace("°C", "degrees Celsius").replace("°F", "degrees Fahrenheit")
    clean = _kokoro_phonetic_fix(clean)
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

    # Trim trailing low-energy tail before RVC to prevent hallucinated words.
    # Kokoro sometimes emits faint artifacts after the last phoneme; RVC
    # amplifies these into audible speech fragments.
    if RVC_ENABLED and audio_np.size > 0:
        win = int(KOKORO_SAMPLE_RATE * 0.03)          # 30ms windows
        rms_thresh = 0.02                              # ~-34 dB
        # Walk backwards to find last window with real speech energy
        end = len(audio_np)
        while end > win:
            chunk = audio_np[end - win : end]
            if np.sqrt(np.mean(chunk ** 2)) > rms_thresh:
                break
            end -= win
        # Keep a tiny 30ms fade-out pad, then zero
        pad = min(int(KOKORO_SAMPLE_RATE * 0.03), len(audio_np) - end)
        audio_np = audio_np[: end + pad]

    pcm16 = (audio_np * 32767.0).astype(np.int16)

    with wave.open(out_path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(KOKORO_SAMPLE_RATE)
        wf.writeframes(pcm16.tobytes())
        # Only add trailing silence when RVC is off (RVC hallucinates over silence)
        if not RVC_ENABLED:
            tail = int(KOKORO_SAMPLE_RATE * 0.25)
            wf.writeframes(b"\x00\x00" * tail)

    # RVC voice conversion post-processing
    if RVC_ENABLED:
        rvc_out = _rvc_convert(out_path)
        if rvc_out != out_path:
            os.replace(rvc_out, out_path)

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

    def play_thinking_filler(self):
        """Queue a breath intake sound for instant playback while LLM generates."""
        if self._muted:
            return
        # Prefer breath sounds (natural, voice-neutral) over verbal fillers
        if self._breath_wavs:
            wav = random.choice(self._breath_wavs)
        elif self._thinking_wavs:
            good = [w for w in self._thinking_wavs
                    if os.path.getsize(w) >= self._MIN_FILLER_BYTES]
            wav = random.choice(good or self._thinking_wavs)
        else:
            return
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

    def _play_wav(self, wav_path: str, alsa_device: str):
        """Play a pre-rendered WAV file via aplay (interruptible)."""
        self._interrupted.clear()
        bus.emit("tts.started")
        bus.emit("speaker.state", state="playing")
        state.playing = True
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

        If local Kokoro isn't loaded yet and Farsight RTX is reachable, uses
        Farsight for synthesis (RTX is always warm — never restarted).
        """
        clean = re.sub(r"<[^>]+>", "", text).strip()
        clean = clean.replace("°C", "degrees Celsius").replace("°F", "degrees Fahrenheit")
        clean = _kokoro_phonetic_fix(clean)
        if not clean:
            return

        # If local TTS not ready, try Farsight RTX as fallback
        if not _local_tts_ready.is_set() and _check_farsight_once():
            self._stream_synth_farsight(clean, alsa_device)
            return

        pipe = _get_kokoro()

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

                    # Trim trailing low-energy tail before RVC
                    if RVC_ENABLED and audio_np.size > 0:
                        win = int(KOKORO_SAMPLE_RATE * 0.03)
                        rms_thresh = 0.02
                        end = len(audio_np)
                        while end > win:
                            chunk = audio_np[end - win : end]
                            if np.sqrt(np.mean(chunk ** 2)) > rms_thresh:
                                break
                            end -= win
                        pad = min(int(KOKORO_SAMPLE_RATE * 0.03), len(audio_np) - end)
                        audio_np = audio_np[: end + pad]

                    pcm16 = (audio_np * 32767.0).astype(np.int16)

                    with wave.open(out_path, "wb") as wf:
                        wf.setnchannels(1)
                        wf.setsampwidth(2)
                        wf.setframerate(KOKORO_SAMPLE_RATE)
                        wf.writeframes(pcm16.tobytes())
                        if not RVC_ENABLED:
                            wf.writeframes(b"\x00\x00" * int(KOKORO_SAMPLE_RATE * 0.1))

                    # RVC voice conversion post-processing
                    if RVC_ENABLED:
                        rvc_out = _rvc_convert(out_path)
                        if rvc_out != out_path:
                            os.replace(rvc_out, out_path)

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
            synth_t.join(timeout=30)
            state.playing = False
            if not self._interrupted.is_set():
                bus.emit("tts.finished")
                bus.emit("speaker.state", state="idle")

        total_ms = (time.perf_counter() - t0) * 1000
        audio_ms = synth_stats["samples"] / KOKORO_SAMPLE_RATE * 1000 if synth_stats["samples"] else 0
        pending = self._work_q.qsize()
        pipeline_info = f" [pending={pending}]" if pending else ""
        print(f"[speaker] Pipelined: {total_ms:.0f}ms total, {audio_ms:.0f}ms audio, "
              f"first={synth_stats['first_ms']:.0f}ms, {synth_stats['clauses']} clauses"
              f" → \"{preview}\"{pipeline_info}")

    # ----- Farsight RTX fallback path -----

    def _stream_synth_farsight(self, text: str, alsa_device: str):
        """Synthesize via Farsight RTX when local Kokoro isn't loaded yet.

        The RTX is always warm (enterprise GPU, never restarted), so latency
        is just network round-trip + synthesis time (~200-500ms per clause).
        Local Kokoro loads quietly in the background and takes over once ready.
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
                    # Fall through to local — block until ready
                    _local_tts_ready.wait(timeout=60)
                    _synth_to_file(clause, "neutral", out_path)
                else:
                    # Farsight returns raw Kokoro — run through local RVC
                    # (only if RVC model is already loaded; don't block on lazy-load)
                    if RVC_ENABLED and _rvc_engine is not None:
                        rvc_out = _rvc_convert(out_path)
                        if rvc_out != out_path:
                            os.replace(rvc_out, out_path)
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
        """Pre-load Kokoro and RVC models (marks _local_tts_ready when done)."""
        try:
            _get_kokoro()
            if RVC_ENABLED:
                _get_rvc()
            _local_tts_ready.set()
            print(f"[speaker] Warmup complete (RVC={'on' if RVC_ENABLED and _rvc_engine else 'off'})")
        except Exception as e:
            _local_tts_ready.set()  # don't block forever
            print(f"[speaker] Warmup failed (non-fatal): {e}")

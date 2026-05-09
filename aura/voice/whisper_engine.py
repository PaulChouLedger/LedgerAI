"""
voice.whisper_engine -- In-process faster-whisper for zero-overhead STT.

Replaces the HTTP call to the Whisper Flask service.  Passes numpy arrays
directly — no WAV encoding, no HTTP, no temp files.

Thread-safe via core.gpu.gpu_lock.
"""

from __future__ import annotations

import os
import time
from typing import Optional

import numpy as np

from core.gpu import gpu_lock

# ── Config (mirrors containers/whisper/container_rest.py) ──

MODEL_NAME = os.environ.get(
    "WHISPER_MODEL", "distil-whisper/distil-large-v3.5-ct2"
)
BEAM_SIZE = 1           # greedy — fastest for short conversational utterances
TEMPERATURE = 0.0
PATIENCE = 1.0
LENGTH_PENALTY = 1.0
# Run on CPU — Jetson Orin NX unified memory was OOMing when LLM (10 GB) and
# Whisper shared the GPU pool, leaving Whisper in a degraded state that
# transcribed every utterance as garbage. CPU int8_float32 is fast enough on
# the 8 Cortex-A78AE cores (~1.5-2x realtime for short utterances). Pure int8
# isn't supported on aarch64 CPUs (no Intel MKL/AVX); int8_float32 keeps the
# small int8 weights with float32 compute, which works on every CPU backend.
DEVICE = "cpu"
COMPUTE_TYPE = "int8_float32"

INITIAL_PROMPT = (
    "Paul is talking to Aura, a voice assistant. "
    "Bob Carella, Mussolini, Hitler, Ledger, $LEDGER."
)


class WhisperEngine:
    """In-process faster-whisper wrapper with GPU locking."""

    def __init__(self) -> None:
        self._model = None
        self._loaded = False

    @property
    def loaded(self) -> bool:
        return self._loaded

    def load(self) -> bool:
        """Load faster-whisper model onto GPU.  Called once during boot."""
        if self._loaded:
            return True
        try:
            from faster_whisper import WhisperModel

            hf_cache = os.path.expanduser("~/.cache/huggingface/hub")
            print(f"[whisper_engine] Loading {MODEL_NAME} ({COMPUTE_TYPE})...")
            t0 = time.time()
            self._model = WhisperModel(
                MODEL_NAME,
                device=DEVICE,
                compute_type=COMPUTE_TYPE,
                download_root=hf_cache,
            )
            self._loaded = True
            print(f"[whisper_engine] Model loaded in {time.time()-t0:.1f}s")
            return True
        except Exception as e:
            print(f"[whisper_engine] Failed to load: {e}")
            import traceback
            traceback.print_exc()
            return False

    def transcribe(
        self,
        audio: np.ndarray,
        sr: int = 16000,
        initial_prompt: Optional[str] = None,
    ) -> tuple[str, float, float]:
        """Transcribe float32 numpy audio directly.

        Args:
            audio: mono float32 numpy array at `sr` Hz
            sr: sample rate (should be 16000 — listener already resamples)
            initial_prompt: override for conditioning prompt

        Returns:
            (text, avg_log_prob, no_speech_prob)
        """
        if not self._loaded or self._model is None:
            print("[whisper_engine] Model not loaded!")
            return "", -1.0, 1.0

        prompt = initial_prompt or INITIAL_PROMPT
        t0 = time.time()

        # No gpu_lock — Whisper runs on CPU now, so it can transcribe in
        # parallel with the GPU LLM. Lets the next utterance start being
        # decoded while the previous reply is still streaming out.
        try:
            segments, info = self._model.transcribe(
                audio,
                language="en",
                beam_size=BEAM_SIZE,
                temperature=TEMPERATURE,
                patience=PATIENCE,
                length_penalty=LENGTH_PENALTY,
                initial_prompt=prompt,
                condition_on_previous_text=False,
                no_speech_threshold=0.6,
                log_prob_threshold=-1.0,
            )
            segment_list = list(segments)
        except RuntimeError as e:
            print(f"[whisper_engine] Runtime error: {e}")
            return "", -1.0, 1.0

        # Post-processing outside the lock
        text = " ".join(s.text.strip() for s in segment_list if s.text.strip())
        avg_log_prob = (
            float(np.mean([s.avg_logprob for s in segment_list]))
            if segment_list
            else -1.0
        )
        no_speech_prob = (
            float(np.mean([s.no_speech_prob for s in segment_list]))
            if segment_list
            else 1.0
        )
        dur = len(audio) / sr
        elapsed = time.time() - t0
        print(
            f"[whisper_engine] '{text}' ({elapsed:.2f}s, {dur:.1f}s audio, "
            f"conf={avg_log_prob:.2f}, nsp={no_speech_prob:.2f})"
        )
        return text, avg_log_prob, no_speech_prob

    def warmup(self) -> None:
        """Send 1s silence to prime CUDA JIT."""
        if self._loaded:
            silence = np.zeros(16000, dtype=np.float32)
            self.transcribe(silence)
            print("[whisper_engine] Warmed up")


# Module-level singleton
whisper_engine = WhisperEngine()

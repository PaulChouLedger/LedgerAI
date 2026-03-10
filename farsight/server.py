"""
Farsight Server — Remote GPU backend for Aura Pucks.

Runs on a high-end GPU workstation (e.g. RTX PRO 6000 Blackwell, 96GB VRAM).
Provides two services:
  1. /perpetual/chat     — Deep LLM reasoning (72B model)
  2. /perpetual/synthesize — High-quality voice-cloned TTS (Chatterbox, 200 steps)

Start:
    python3 server.py

The server listens on port 11435 by default.
Pucks connect via Tailscale mesh VPN (zero-config WireGuard).
"""

from __future__ import annotations

import os
import time

from flask import Flask, request, jsonify

app = Flask(__name__)

# ---------------------------------------------------------------------------
# LLM (lazy-loaded)
# ---------------------------------------------------------------------------

_llm = None
_llm_lock = None

MODEL_PATH = os.environ.get(
    "FARSIGHT_MODEL",
    "/home/paul/.cache/huggingface/hub/models--bartowski--Qwen2.5-7B-Instruct-GGUF/"
    "snapshots/*/Qwen2.5-7B-Instruct-Q4_K_M.gguf",
)


def _get_llm():
    """Lazy-load the LLM model."""
    global _llm, _llm_lock
    import threading
    if _llm_lock is None:
        _llm_lock = threading.Lock()
    if _llm is not None:
        return _llm

    import glob
    paths = glob.glob(MODEL_PATH)
    model_path = paths[0] if paths else MODEL_PATH

    print(f"[farsight] Loading LLM: {model_path}")
    t0 = time.time()

    from llama_cpp import Llama
    import torch
    n_gpu = -1 if torch.cuda.is_available() else 0
    _llm = Llama(
        model_path=model_path,
        n_ctx=4096,
        n_gpu_layers=n_gpu,
        n_threads=os.cpu_count() or 8,
        n_batch=512,
        verbose=False,
    )
    print(f"[farsight] LLM loaded in {time.time() - t0:.1f}s (GPU layers: {n_gpu})")
    return _llm


@app.route("/perpetual/chat", methods=["POST"])
def perpetual_chat():
    """LLM inference endpoint for Aura Perpetual rumination."""
    data = request.get_json(silent=True) or {}
    system_prompt = data.get("system_prompt", "You are a helpful assistant.")
    user_prompt = data.get("prompt", "")
    max_tokens = int(data.get("max_tokens", 512))

    if not user_prompt:
        return jsonify({"error": "No prompt provided"}), 400

    llm = _get_llm()
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    with _llm_lock:
        t0 = time.time()
        resp = llm.create_chat_completion(
            messages=messages,
            max_tokens=max_tokens,
            temperature=0.7,
            top_p=0.9,
            stream=False,
        )
        elapsed = time.time() - t0

    text = resp["choices"][0]["message"]["content"].strip()
    tokens = resp.get("usage", {}).get("completion_tokens", 0)
    print(f"[farsight] Chat: {tokens} tokens in {elapsed:.1f}s ({tokens/elapsed:.0f} tok/s)")
    return jsonify({"response": text})


# ---------------------------------------------------------------------------
# TTS (Chatterbox voice cloning — lazy-loaded)
# ---------------------------------------------------------------------------

_chatterbox = None


def _get_chatterbox():
    """Lazy-load Chatterbox TTS on GPU."""
    global _chatterbox
    if _chatterbox is not None:
        return _chatterbox

    print("[farsight-tts] Loading Chatterbox TTS on CUDA...")
    t0 = time.time()
    from chatterbox.tts import ChatterboxTTS
    _chatterbox = ChatterboxTTS.from_pretrained(device="cuda")
    print(f"[farsight-tts] Chatterbox loaded in {time.time() - t0:.1f}s")
    return _chatterbox


@app.route("/perpetual/synthesize", methods=["POST"])
def perpetual_synthesize():
    """High-quality TTS with optional voice cloning for briefing delivery.

    Request JSON:
        text:         str  — Text to synthesize
        voice_sample: str  — (optional) Base64-encoded WAV for voice cloning
        steps:        int  — (optional) Diffusion steps (default 200)

    Returns: WAV audio (24kHz mono)
    """
    import base64
    import io
    import tempfile
    import wave
    import numpy as np
    import torch
    from flask import Response

    data = request.get_json(silent=True) or {}
    text = data.get("text", "")
    voice_b64 = data.get("voice_sample")
    steps = int(data.get("steps", 200))

    if not text:
        return jsonify({"error": "No text provided"}), 400

    try:
        model = _get_chatterbox()

        # Prepare voice reference
        voice_path = None
        if voice_b64:
            wav_bytes = base64.b64decode(voice_b64)
            tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            tmp.write(wav_bytes)
            tmp.close()
            voice_path = tmp.name
            print(f"[farsight-tts] Voice reference: {len(wav_bytes)} bytes")

        # Synthesize
        t0 = time.time()
        print(f"[farsight-tts] Synthesizing ({steps} steps): \"{text[:80]}...\"")

        kwargs = dict(exaggeration=0.5, cfg_weight=0.5)
        if voice_path:
            kwargs["audio_prompt_path"] = voice_path
        wav_tensor = model.generate(text, **kwargs)

        elapsed = time.time() - t0
        print(f"[farsight-tts] Synthesis: {elapsed:.1f}s")

        # Convert to WAV
        if isinstance(wav_tensor, torch.Tensor):
            audio_np = wav_tensor.cpu().numpy().squeeze()
        else:
            audio_np = np.array(wav_tensor).squeeze()

        peak = float(np.max(np.abs(audio_np))) if audio_np.size else 1.0
        if peak > 1e-8:
            audio_np = audio_np / peak * 0.95
        pcm16 = (np.clip(audio_np, -1.0, 1.0) * 32767).astype(np.int16)

        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(24000)
            wf.writeframes(pcm16.tobytes())
            wf.writeframes(b"\x00\x00" * int(24000 * 0.3))  # tail silence

        wav_data = buf.getvalue()
        duration = len(pcm16) / 24000
        print(f"[farsight-tts] Output: {duration:.1f}s, {len(wav_data)} bytes")

        if voice_path:
            os.unlink(voice_path)

        return Response(wav_data, mimetype="audio/wav")

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "service": "farsight",
        "llm_loaded": _llm is not None,
        "tts_loaded": _chatterbox is not None,
    })


if __name__ == "__main__":
    port = int(os.environ.get("FARSIGHT_PORT", "11435"))
    print(f"[farsight] Starting on port {port}")
    app.run(host="0.0.0.0", port=port, threaded=True)

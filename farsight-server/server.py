"""
Farsight Server — Remote GPU inference for Aura Perpetual.

Loads Qwen2.5-72B-Instruct Q4_K_M on GPU via llama-cpp-python and exposes
a /perpetual/chat endpoint that Pucks can POST to for deep thinking.

Usage:
    python3 server.py

Expects the model at: ./models/Qwen2.5-72B-Instruct-Q4_K_M.gguf
Download with:
    huggingface-cli download bartowski/Qwen2.5-72B-Instruct-GGUF \
        Qwen2.5-72B-Instruct-Q4_K_M.gguf --local-dir ./models
"""

import os
import threading
import time

from flask import Flask, request, jsonify
from llama_cpp import Llama

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
MODEL_PATH = os.path.join(MODEL_DIR, "Qwen2.5-72B-Instruct-Q4_K_M.gguf")

# GPU layers: -1 = offload entire model to GPU (no CPU inference)
N_GPU_LAYERS = int(os.environ.get("FARSIGHT_GPU_LAYERS", "-1"))
N_CTX = int(os.environ.get("FARSIGHT_CTX", "8192"))
N_BATCH = int(os.environ.get("FARSIGHT_BATCH", "2048"))
N_THREADS = int(os.environ.get("FARSIGHT_THREADS", "8"))  # 7800X3D = 8 physical cores
FLASH_ATTN = os.environ.get("FARSIGHT_FLASH_ATTN", "1") == "1"
USE_MMAP = os.environ.get("FARSIGHT_MMAP", "1") == "1"
HOST = os.environ.get("FARSIGHT_HOST", "0.0.0.0")
PORT = int(os.environ.get("FARSIGHT_PORT", "11435"))

# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

llm = None
llm_lock = threading.Lock()
model_loaded = False


def load_model():
    global llm, model_loaded
    if not os.path.exists(MODEL_PATH):
        print(f"[farsight] ERROR: Model not found at {MODEL_PATH}")
        print(f"[farsight] Download it with:")
        print(f"  huggingface-cli download bartowski/Qwen2.5-7B-Instruct-GGUF \\")
        print(f"    Qwen2.5-7B-Instruct-Q4_K_M.gguf --local-dir {MODEL_DIR}")
        return False

    print(f"[farsight] Loading model: {MODEL_PATH}")
    print(f"[farsight] GPU layers: {N_GPU_LAYERS}, context: {N_CTX}, batch: {N_BATCH}")
    print(f"[farsight] flash_attn: {FLASH_ATTN}, mmap: {USE_MMAP}, threads: {N_THREADS}")
    start = time.time()

    llm = Llama(
        model_path=MODEL_PATH,
        n_gpu_layers=N_GPU_LAYERS,
        n_ctx=N_CTX,
        n_batch=N_BATCH,
        n_threads=N_THREADS,
        use_mmap=USE_MMAP,
        flash_attn=FLASH_ATTN,
        verbose=False,
    )

    elapsed = time.time() - start
    model_loaded = True
    print(f"[farsight] Model loaded in {elapsed:.1f}s")
    return True


# ---------------------------------------------------------------------------
# Flask app
# ---------------------------------------------------------------------------

app = Flask(__name__)


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok" if model_loaded else "loading",
        "model": os.path.basename(MODEL_PATH),
        "model_loaded": model_loaded,
        "gpu_layers": N_GPU_LAYERS,
        "context_size": N_CTX,
        "batch_size": N_BATCH,
        "flash_attn": FLASH_ATTN,
        "use_mmap": USE_MMAP,
        "threads": N_THREADS,
    })


@app.route("/perpetual/chat", methods=["POST"])
def perpetual_chat():
    """Direct LLM call for Aura Perpetual — matches Puck endpoint spec."""
    if not model_loaded or llm is None:
        return jsonify({"error": "model not loaded"}), 503

    data = request.get_json(silent=True) or {}
    system_prompt = data.get("system_prompt", "You are a helpful assistant.")
    user_prompt = data.get("prompt", "")
    max_tokens = data.get("max_tokens", 512)

    if not user_prompt:
        return jsonify({"error": "prompt required"}), 400

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    try:
        with llm_lock:
            resp = llm.create_chat_completion(
                messages=messages,
                max_tokens=max_tokens,
                temperature=0.7,
                top_p=0.9,
                stream=False,
            )
        text = resp["choices"][0]["message"]["content"].strip()
        return jsonify({"response": text})
    except Exception as e:
        print(f"[farsight] Chat error: {e}")
        return jsonify({"error": str(e)}), 500


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# TTS Pre-synthesis (Chatterbox voice cloning — lazy-loaded)
# Added by Puck-Claude for high-quality briefing audio
# ---------------------------------------------------------------------------

_chatterbox = None


def _get_chatterbox():
    """Lazy-load Chatterbox TTS on GPU."""
    global _chatterbox
    if _chatterbox is not None:
        return _chatterbox

    print("[farsight-tts] Loading Chatterbox TTS on CUDA...")
    import time as _t
    t0 = _t.time()
    from chatterbox.tts import ChatterboxTTS
    _chatterbox = ChatterboxTTS.from_pretrained(device="cuda")
    print(f"[farsight-tts] Chatterbox loaded in {_t.time() - t0:.1f}s")
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


if __name__ == "__main__":
    if not load_model():
        print("[farsight] Cannot start without model. Exiting.")
        raise SystemExit(1)

    print(f"[farsight] Farsight server starting on {HOST}:{PORT}")
    app.run(host=HOST, port=PORT, threaded=True, debug=False)

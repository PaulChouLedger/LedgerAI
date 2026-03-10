"""
Farsight TTS endpoint — /perpetual/synthesize

Add this to the Farsight Flask server on the RTX workstation.
Uses Chatterbox TTS with voice cloning for high-quality briefing audio.

Usage: Import and register with the Flask app in server.py:
    from synthesize_endpoint import register_tts_endpoint
    register_tts_endpoint(app)
"""

from __future__ import annotations

import base64
import io
import tempfile
import time
import wave

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Chatterbox TTS (lazy-loaded — heavy model, only load when first needed)
# ---------------------------------------------------------------------------

_chatterbox_model = None


def _get_chatterbox():
    """Lazy-load ChatterboxTTS model on GPU."""
    global _chatterbox_model
    if _chatterbox_model is not None:
        return _chatterbox_model

    print("[farsight-tts] Loading Chatterbox TTS model on CUDA...")
    t0 = time.time()
    from chatterbox.tts import ChatterboxTTS

    _chatterbox_model = ChatterboxTTS.from_pretrained(device="cuda")
    print(f"[farsight-tts] Chatterbox loaded in {time.time() - t0:.1f}s")
    return _chatterbox_model


def register_tts_endpoint(app):
    """Register /perpetual/synthesize on the given Flask app."""
    from flask import request, Response, jsonify

    @app.route("/perpetual/synthesize", methods=["POST"])
    def perpetual_synthesize():
        """Synthesize high-quality briefing audio with optional voice cloning.

        Request JSON:
            text:         str  — Text to synthesize
            voice_sample: str  — (optional) Base64-encoded WAV of the user's voice
            steps:        int  — (optional) Diffusion steps (default 200)

        Returns: WAV audio file (24kHz mono)
        """
        data = request.get_json(silent=True) or {}
        text = data.get("text", "")
        voice_b64 = data.get("voice_sample")
        steps = int(data.get("steps", 200))

        if not text:
            return jsonify({"error": "No text provided"}), 400

        try:
            model = _get_chatterbox()

            # Prepare voice reference (if provided)
            voice_path = None
            if voice_b64:
                # Decode base64 WAV to temp file
                wav_bytes = base64.b64decode(voice_b64)
                tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                tmp.write(wav_bytes)
                tmp.close()
                voice_path = tmp.name
                print(f"[farsight-tts] Voice reference: {len(wav_bytes)} bytes")

            # Synthesize with Chatterbox
            t0 = time.time()
            print(f"[farsight-tts] Synthesizing ({steps} steps): \"{text[:80]}...\"")

            if voice_path:
                # Voice cloning mode
                wav_tensor = model.generate(
                    text,
                    audio_prompt_path=voice_path,
                    exaggeration=0.5,
                    cfg_weight=0.5,
                    num_steps=steps,
                )
            else:
                # Default voice (no cloning)
                wav_tensor = model.generate(
                    text,
                    exaggeration=0.5,
                    cfg_weight=0.5,
                    num_steps=steps,
                )

            elapsed = time.time() - t0
            print(f"[farsight-tts] Synthesis complete: {elapsed:.1f}s")

            # Convert tensor to WAV bytes
            if isinstance(wav_tensor, torch.Tensor):
                audio_np = wav_tensor.cpu().numpy().squeeze()
            else:
                audio_np = np.array(wav_tensor).squeeze()

            # Normalize
            peak = float(np.max(np.abs(audio_np))) if audio_np.size else 1.0
            if peak > 1e-8:
                audio_np = audio_np / peak * 0.95
            pcm16 = (np.clip(audio_np, -1.0, 1.0) * 32767).astype(np.int16)

            # Write WAV to buffer
            buf = io.BytesIO()
            with wave.open(buf, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(24000)  # Chatterbox outputs at 24kHz
                wf.writeframes(pcm16.tobytes())
                # Tail silence to prevent clipping
                tail = int(24000 * 0.3)
                wf.writeframes(b"\x00\x00" * tail)

            wav_data = buf.getvalue()
            duration = len(pcm16) / 24000
            print(f"[farsight-tts] Output: {duration:.1f}s audio, {len(wav_data)} bytes")

            # Clean up temp file
            if voice_path:
                import os
                os.unlink(voice_path)

            return Response(wav_data, mimetype="audio/wav")

        except Exception as e:
            import traceback
            traceback.print_exc()
            return jsonify({"error": str(e)}), 500

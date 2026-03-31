import os
import sys
os.environ['NUMPY_DISABLE_ABI_COMPATIBILITY'] = '1'

from flask import Flask, request, jsonify
from faster_whisper import WhisperModel
import soundfile as sf
import numpy as np
import scipy.signal
import tempfile
import time

# ============================================================================
# CONFIGURATION
# ============================================================================

MODEL_NAME = os.environ.get("WHISPER_MODEL", "distil-whisper/distil-large-v3.5-ct2")

CACHE_DIR = "/app/cache/whisper"

BEAM_SIZE = 5
TEMPERATURE = 0.0
PATIENCE = 1.0
LENGTH_PENALTY = 1.0
COMPUTE_TYPE = "int8"

# Simple conversational prompt — no domain-specific biasing
INITIAL_PROMPT = "This is a conversation."

# ============================================================================
# END CONFIGURATION
# ============================================================================

app = Flask(__name__)

print(f"[Whisper] Loading model: {MODEL_NAME} ({COMPUTE_TYPE})")

# Map model names to HuggingFace repo directories
MODEL_MAPPING = {
    "distil-small.en": "models--Systran--faster-distil-whisper-small.en",
    "small.en": "models--Systran--faster-small-whisper.en",
    "medium.en": "models--Systran--faster-medium-whisper.en",
    "base.en": "models--Systran--faster-base-whisper.en",
    "large-v3-turbo": "models--mobiuslabsgmbh--faster-whisper-large-v3-turbo",
    "distil-large-v3": "models--Systran--faster-distil-whisper-large-v3",
    "distil-whisper/distil-large-v3.5-ct2": "models--distil-whisper--distil-large-v3.5-ct2",
}

model_repo = MODEL_MAPPING.get(MODEL_NAME, f"models--Systran--faster-{MODEL_NAME.replace('.', '-')}")
model_cache_path = f"/root/.cache/huggingface/hub/{model_repo}"
if os.path.exists(model_cache_path):
    print(f"[Whisper] Model found in cache")
else:
    print(f"[Whisper] Model not cached, will download")

try:
    model = WhisperModel(MODEL_NAME, device="cuda", compute_type=COMPUTE_TYPE,
                         download_root="/root/.cache/huggingface/hub")
    print(f"[Whisper] Model loaded ({COMPUTE_TYPE})")
except Exception as e:
    print(f"[Whisper] {COMPUTE_TYPE} failed: {e}, trying int8_float16...")
    try:
        model = WhisperModel(MODEL_NAME, device="cuda", compute_type="int8_float16",
                             download_root="/root/.cache/huggingface/hub")
        print(f"[Whisper] Model loaded (int8_float16 fallback)")
    except Exception as e2:
        print(f"[Whisper] FATAL: GPU loading failed: {e2}")
        raise RuntimeError("GPU initialization failed")

# Timing statistics
timing_stats = {
    "total_requests": 0,
    "total_processing_time": 0,
    "total_transcription_time": 0,
    "total_audio_duration": 0,
    "min_transcription_time": float('inf'),
    "max_transcription_time": 0,
}


def preprocess_audio(path, target_sr=16000):
    audio, sr = sf.read(path)
    if len(audio.shape) > 1:
        audio = np.mean(audio, axis=1)
    if sr != target_sr:
        audio = scipy.signal.resample_poly(audio, target_sr, sr)
    return audio.astype(np.float32)


@app.route("/transcribe", methods=["POST"])
def transcribe():
    request_start = time.time()

    if "audio" not in request.files:
        return jsonify({"error": "No audio file uploaded"}), 400

    audio_file = request.files["audio"]
    custom_prompt = request.form.get("initial_prompt", INITIAL_PROMPT)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(audio_file.read())
        tmp_path = tmp.name

    try:
        audio = preprocess_audio(tmp_path)
        audio_duration = len(audio) / 16000

        if np.sqrt(np.mean(audio**2)) < 0.001:
            print(f"[Whisper] Audio very quiet ({audio_duration:.1f}s)")

        t0 = time.time()
        text = ""

        try:
            segments, info = model.transcribe(
                audio,
                language="en",
                beam_size=BEAM_SIZE,
                temperature=TEMPERATURE,
                patience=PATIENCE,
                length_penalty=LENGTH_PENALTY,
                initial_prompt=custom_prompt,
            )
            segment_list = list(segments)
            text = " ".join([s.text.strip() for s in segment_list if s.text.strip()])
            avg_log_prob = np.mean([s.avg_logprob for s in segment_list]) if segment_list else -1.0
            no_speech_prob = np.mean([s.no_speech_prob for s in segment_list]) if segment_list else 1.0
            print(f"[Whisper] '{text}' ({time.time()-t0:.2f}s, {audio_duration:.1f}s audio, conf={avg_log_prob:.2f}, nsp={no_speech_prob:.2f})")

        except RuntimeError as e:
            error_str = str(e)
            print(f"[Whisper] Runtime error: {error_str}")

            if "out of memory" in error_str.lower():
                import gc
                gc.collect()
                try:
                    import torch
                    torch.cuda.empty_cache()
                except ImportError:
                    pass

                try:
                    segments, info = model.transcribe(
                        audio, language="en", beam_size=BEAM_SIZE,
                        temperature=TEMPERATURE, patience=PATIENCE,
                        length_penalty=LENGTH_PENALTY, initial_prompt=custom_prompt,
                    )
                    text = " ".join([s.text.strip() for s in list(segments) if s.text.strip()])
                    print(f"[Whisper] OOM retry OK: '{text}'")
                except Exception:
                    text = ""
            else:
                text = ""

        except Exception as e:
            print(f"[Whisper] Transcription failed: {e}")
            text = ""

        transcription_time = time.time() - t0
        total_time = time.time() - request_start

        timing_stats["total_requests"] += 1
        timing_stats["total_processing_time"] += total_time
        timing_stats["total_transcription_time"] += transcription_time
        timing_stats["total_audio_duration"] += audio_duration
        timing_stats["min_transcription_time"] = min(timing_stats["min_transcription_time"], transcription_time)
        timing_stats["max_transcription_time"] = max(timing_stats["max_transcription_time"], transcription_time)

        os.remove(tmp_path)

        return jsonify({
            "text": text,
            "avg_log_prob": round(avg_log_prob, 4) if 'avg_log_prob' in dir() else -1.0,
            "no_speech_prob": round(no_speech_prob, 4) if 'no_speech_prob' in dir() else 1.0,
            "timing": {
                "request_to_completion": round(total_time, 3),
                "model_transcription": round(transcription_time, 3),
                "audio_duration": round(audio_duration, 3),
            },
        })
    except Exception as e:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        return jsonify({"error": str(e)}), 500


@app.route("/health", methods=["GET"])
def health():
    n = timing_stats["total_requests"]
    stats = {}
    if n > 0:
        stats = {
            "avg_transcription_time": round(timing_stats["total_transcription_time"] / n, 3),
            "avg_audio_duration": round(timing_stats["total_audio_duration"] / n, 3),
        }
    return jsonify({
        "status": "healthy",
        "model": MODEL_NAME,
        "compute_type": COMPUTE_TYPE,
        "beam_size": BEAM_SIZE,
        "requests_processed": n,
        "timing_stats": stats,
    })


@app.route("/models/available", methods=["GET"])
def get_available_models():
    hub_cache_dir = "/root/.cache/huggingface/hub"
    dir_to_model = {v: k for k, v in MODEL_MAPPING.items()}
    available = []

    if os.path.exists(hub_cache_dir):
        for item in os.listdir(hub_cache_dir):
            if item in dir_to_model and os.path.isdir(os.path.join(hub_cache_dir, item)):
                available.append(dir_to_model[item])

    return jsonify({
        "available_models": sorted(available),
        "current_model": MODEL_NAME,
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

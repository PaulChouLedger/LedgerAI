import os
# Set NumPy compatibility before importing
os.environ['NUMPY_DISABLE_ABI_COMPATIBILITY'] = '1'

from flask import Flask, request, jsonify
from faster_whisper import WhisperModel
import soundfile as sf
import numpy as np
import scipy.signal
import tempfile
import time

app = Flask(__name__)
# Use faster-whisper with same model as transcription tuner
# Set cache directory to avoid re-downloading models
cache_dir = "/app/cache/whisper"
print(f"[Whisper] 🚀 Loading faster-whisper model: distil-small.en")
print(f"[Whisper] 📁 Cache directory: {cache_dir}")

# Check if cache directory exists and what's in it
import os
if os.path.exists(cache_dir):
    print(f"[Whisper] 📁 Cache directory exists")
    cache_contents = os.listdir(cache_dir)
    print(f"[Whisper] 📁 Cache contents: {cache_contents}")
else:
    print(f"[Whisper] ⚠️ Cache directory does not exist")

# Check Hugging Face cache
hf_cache = "/root/.cache/huggingface"
if os.path.exists(hf_cache):
    print(f"[Whisper] 📁 HF cache exists: {hf_cache}")
    hf_contents = os.listdir(hf_cache)
    print(f"[Whisper] 📁 HF cache contents: {hf_contents}")
else:
    print(f"[Whisper] ⚠️ HF cache does not exist: {hf_cache}")

# Check if model is available in the built-in cache
model_name = os.getenv("WHISPER_MODEL", "distil-small.en")
print(f"[Whisper] 📋 Using model: {model_name}")

# Map model names to their actual HuggingFace repo names
model_mapping = {
    "distil-small.en": "models--Systran--faster-distil-whisper-small.en",
    "small.en": "models--Systran--faster-small-whisper.en",
    "base.en": "models--Systran--faster-base-whisper.en"
}

# Get the actual model repo name
model_repo = model_mapping.get(model_name, f"models--Systran--faster-{model_name.replace('.', '-')}")
model_cache_path = f"/root/.cache/huggingface/hub/{model_repo}"

if os.path.exists(model_cache_path):
    print(f"[Whisper] ✅ Model found in built-in cache: {model_repo}")
else:
    print(f"[Whisper] ⚠️ Model not found in built-in cache: {model_repo}")
    print(f"[Whisper] 📁 Available models:")
    hf_hub_dir = "/root/.cache/huggingface/hub"
    if os.path.exists(hf_hub_dir):
        for item in os.listdir(hf_hub_dir):
            if item.startswith("models--"):
                print(f"[Whisper]   - {item}")
    else:
        print(f"[Whisper]   - No HuggingFace hub directory found")

# Let faster_whisper handle CUDA/PyTorch initialization internally
print(f"[Whisper] 🚀 Initializing faster-whisper with GPU support...")

# faster_whisper will handle GPU detection and initialization
print(f"[Whisper] ✅ Ready to initialize faster-whisper model")

# Load GPU model - NO CPU FALLBACK
try:
    model = WhisperModel(model_name, device="cuda", compute_type="float16")
    print(f"[Whisper] ✅ GPU model '{model_name}' loaded successfully from built-in cache")
except Exception as e:
    print(f"[Whisper] ❌ GPU model loading failed: {e}")
    print(f"[Whisper] 💥 FATAL: GPU required - no CPU fallback available")
    raise RuntimeError("GPU initialization failed - GPU is required for this container")

# Timing statistics tracking
timing_stats = {
    "total_requests": 0,
    "total_processing_time": 0,
    "total_transcription_time": 0,
    "total_audio_duration": 0,
    "min_transcription_time": float('inf'),
    "max_transcription_time": 0
}

def preprocess_audio(path, target_sr=16000):
    audio, sr = sf.read(path)

    # Convert stereo to mono if necessary
    if len(audio.shape) > 1:
        audio = np.mean(audio, axis=1)

    # Resample if not 16kHz
    if sr != target_sr:
        audio = scipy.signal.resample_poly(audio, target_sr, sr)

    return audio.astype(np.float32)

@app.route("/transcribe", methods=["POST"])
def transcribe():
    # Start timing from request receipt
    request_start_time = time.time()
    
    if "audio" not in request.files:
        return jsonify({"error": "No audio file uploaded"}), 400

    audio_file = request.files["audio"]
    
    # Time file processing
    file_processing_start = time.time()
    
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(audio_file.read())
        tmp_path = tmp.name
    
    file_processing_time = time.time() - file_processing_start

    try:
        # Time audio preprocessing
        preprocessing_start = time.time()
        audio = preprocess_audio(tmp_path)
        preprocessing_time = time.time() - preprocessing_start
        
        # Debug audio properties
        print(f"[Whisper] 🔍 Audio properties:")
        print(f"  📊 Shape: {audio.shape}")
        print(f"  📊 Duration: {len(audio) / 16000:.3f}s")
        print(f"  📊 Min/Max: {audio.min():.4f}/{audio.max():.4f}")
        print(f"  📊 RMS: {np.sqrt(np.mean(audio**2)):.4f}")
        print(f"  📊 Non-zero samples: {np.count_nonzero(audio)}/{len(audio)}")
        import sys
        sys.stdout.flush()
        
        # Check if audio is too quiet or silent
        if np.sqrt(np.mean(audio**2)) < 0.001:
            print(f"[Whisper] ⚠️ Audio is very quiet (RMS < 0.001)")
            sys.stdout.flush()
        if np.count_nonzero(audio) < len(audio) * 0.1:
            print(f"[Whisper] ⚠️ Audio has very few non-zero samples")
            sys.stdout.flush()
        
        # Time model transcription (using faster-whisper like transcription tuner)
        transcription_start = time.time()
        print(f"[Whisper] 🧠 Starting transcription at {transcription_start:.6f}")
        
        # Initialize text variable
        text = ""
        
        try:
            print(f"[Whisper] 🧠 Model parameters: language='en', beam_size=5")
            sys.stdout.flush()
            segments, _ = model.transcribe(audio, language="en", beam_size=5)
            print(f"[Whisper] 🧠 Transcription completed, processing segments...")
            sys.stdout.flush()
            
            # Debug: Check segments
            segment_list = list(segments)
            print(f"[Whisper] 🔍 Found {len(segment_list)} segments")
            sys.stdout.flush()
            for i, segment in enumerate(segment_list):
                print(f"[Whisper] 🔍 Segment {i}: '{segment.text}' (start={segment.start:.2f}, end={segment.end:.2f})")
                sys.stdout.flush()
            
            text = " ".join([s.text.strip() for s in segment_list if s.text.strip()])
            print(f"[Whisper] 📝 Final text: '{text}'")
            sys.stdout.flush()
            
        except RuntimeError as e:
            print(f"[Whisper] ❌ Runtime error: {e}")
            if "cuDNN" in str(e) or "CUDNN_STATUS" in str(e):
                print(f"[Whisper] 🔍 cuDNN error - GPU/CUDA initialization issue")
                # Try to clear GPU memory and retry once
                import torch
                torch.cuda.empty_cache()
                print(f"[Whisper] 🔄 Cleared GPU memory, retrying...")
                try:
                    segments, _ = model.transcribe(audio, language="en", beam_size=5)
                    segment_list = list(segments)
                    text = " ".join([s.text.strip() for s in segment_list if s.text.strip()])
                    print(f"[Whisper] ✅ Retry successful: '{text}'")
                    sys.stdout.flush()
                except Exception as retry_error:
                    print(f"[Whisper] ❌ Retry also failed: {retry_error}")
                    sys.stdout.flush()
                    text = ""
            else:
                sys.stdout.flush()
                text = ""
        except Exception as e:
            print(f"[Whisper] ❌ Transcription failed: {e}")
            print(f"[Whisper] ❌ Error type: {type(e).__name__}")
            import traceback
            print(f"[Whisper] ❌ Traceback: {traceback.format_exc()}")
            sys.stdout.flush()
            text = ""
            
        transcription_time = time.time() - transcription_start
        
        # Calculate total processing time
        total_time = time.time() - request_start_time
        
        # Calculate audio duration
        audio_duration = len(audio) / 16000  # Assuming 16kHz sample rate
        
        # Calculate efficiency (real-time factor)
        efficiency = audio_duration / transcription_time if transcription_time > 0 else 0
        
        # Update timing statistics
        timing_stats["total_requests"] += 1
        timing_stats["total_processing_time"] += total_time
        timing_stats["total_transcription_time"] += transcription_time
        timing_stats["total_audio_duration"] += audio_duration
        timing_stats["min_transcription_time"] = min(timing_stats["min_transcription_time"], transcription_time)
        timing_stats["max_transcription_time"] = max(timing_stats["max_transcription_time"], transcription_time)
        
        # Print timing information to console (like transcription tuner)
        print(f"[Whisper] ⏱️ TIMING METRICS:")
        print(f"  📥 Request to completion: {total_time:.3f}s")
        print(f"  📁 File processing: {file_processing_time:.3f}s")
        print(f"  🔧 Audio preprocessing: {preprocessing_time:.3f}s")
        print(f"  🧠 Model transcription: {transcription_time:.3f}s")
        print(f"  📊 Audio duration: {audio_duration:.3f}s")
        print(f"  ⚡ Efficiency RTF: {efficiency:.2f}x")
        print(f"  📝 Transcribed: \"{text}\"")
        print()  # Empty line for readability
        sys.stdout.flush()
        
        # Clean up temp file
        os.remove(tmp_path)
        
        # Return comprehensive timing information
        return jsonify({
            "text": text,
            "timing": {
                "request_to_completion": round(total_time, 3),
                "file_processing": round(file_processing_time, 3),
                "audio_preprocessing": round(preprocessing_time, 3),
                "model_transcription": round(transcription_time, 3),
                "audio_duration": round(audio_duration, 3),
                "efficiency_rtf": round(efficiency, 2)
            }
        })
    except Exception as e:
        # Clean up temp file on error
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        return jsonify({"error": str(e)}), 500

@app.route("/health", methods=["GET"])
def health():
    """Health endpoint with timing statistics"""
    if timing_stats["total_requests"] == 0:
        return jsonify({
            "status": "healthy",
            "model": "distil-small.en",
            "requests_processed": 0,
            "message": "No requests processed yet"
        })
    
    avg_processing_time = timing_stats["total_processing_time"] / timing_stats["total_requests"]
    avg_transcription_time = timing_stats["total_transcription_time"] / timing_stats["total_requests"]
    avg_audio_duration = timing_stats["total_audio_duration"] / timing_stats["total_requests"]
    overall_efficiency = timing_stats["total_audio_duration"] / timing_stats["total_transcription_time"] if timing_stats["total_transcription_time"] > 0 else 0
    
    return jsonify({
        "status": "healthy",
        "model": "distil-small.en",
        "requests_processed": timing_stats["total_requests"],
        "timing_stats": {
            "avg_processing_time": round(avg_processing_time, 3),
            "avg_transcription_time": round(avg_transcription_time, 3),
            "avg_audio_duration": round(avg_audio_duration, 3),
            "min_transcription_time": round(timing_stats["min_transcription_time"], 3),
            "max_transcription_time": round(timing_stats["max_transcription_time"], 3),
            "overall_efficiency_rtf": round(overall_efficiency, 2)
        }
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

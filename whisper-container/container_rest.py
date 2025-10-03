from flask import Flask, request, jsonify
from whisper_trt import load_trt_model
import soundfile as sf
import numpy as np
import scipy.signal
import tempfile
import os
import time

# TensorRT optimizations and memory fixes
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TRT_LOGGER_VERBOSITY'] = '1'
os.environ['MALLOC_CHECK_'] = '0'  # Disable malloc debugging
os.environ['PYTHONMALLOC'] = 'malloc'  # Use system malloc
os.environ['PYTHONUNBUFFERED'] = '1'
os.environ['PYTHONHASHSEED'] = '0'  # Fix hash randomization
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'  # Disable .pyc files
os.environ['PYTHONIOENCODING'] = 'utf-8'  # Fix encoding issues

app = Flask(__name__)

# Load TensorRT optimized model with memory management
print("[Whisper] 🔧 Loading TensorRT model...")
print("[Whisper] 🔧 Setting up TensorRT environment...")

# Clear any existing CUDA context and set memory limits
try:
    import torch
    import gc
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()  # Clear IPC resources
        # Set memory fraction to prevent OOM
        torch.cuda.set_per_process_memory_fraction(0.7)  # Reduced from 0.8
        gc.collect()  # Force garbage collection
        print("[Whisper] 🔧 Cleared CUDA cache and set memory limit")
except Exception as e:
    print(f"[Whisper] ⚠️ CUDA setup warning: {e}")

# Load TensorRT model - no fallbacks
print("[Whisper] 🔧 Loading TensorRT model with memory protection...")
print(f"[Whisper] 🔍 Cache directories:")
print(f"[Whisper] 🔍 /root/.cache/whisper: {os.path.exists('/root/.cache/whisper')}")
print(f"[Whisper] 🔍 /root/.cache/whisper_trt: {os.path.exists('/root/.cache/whisper_trt')}")
if os.path.exists('/root/.cache/whisper'):
    print(f"[Whisper] 🔍 Whisper cache contents: {os.listdir('/root/.cache/whisper')}")
if os.path.exists('/root/.cache/whisper_trt'):
    print(f"[Whisper] 🔍 WhisperTRT cache contents: {os.listdir('/root/.cache/whisper_trt')}")

# Check if we can find the cached model files
import glob
whisper_files = glob.glob('/root/.cache/whisper/*.pt')
whisper_trt_files = glob.glob('/root/.cache/whisper_trt/*.pth')
print(f"[Whisper] 🔍 Found whisper files: {whisper_files}")
print(f"[Whisper] 🔍 Found whisper_trt files: {whisper_trt_files}")

model = load_trt_model("base.en")
print("[Whisper] ✅ TensorRT model loaded successfully")

# Add cleanup function for graceful shutdown
import atexit
import signal

def cleanup():
    print("[Whisper] 🧹 Cleaning up resources...")
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        gc.collect()
    except:
        pass

atexit.register(cleanup)
signal.signal(signal.SIGTERM, lambda s, f: cleanup())
signal.signal(signal.SIGINT, lambda s, f: cleanup())

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
        
        # Time model transcription (using whisper_trt API)
        transcription_start = time.time()
        start_timestamp = time.strftime('%H:%M:%S') + f".{int((transcription_start % 1) * 1000000):06d}"
        print(f"[Whisper] 🚀 TRANSCRIPTION PROCESSING START: {start_timestamp}")
        
        print(f"[Whisper] 🔍 Audio data shape: {audio.shape}, duration: {len(audio) / 16000:.3f}s")
        print(f"[Whisper] 🧠 Starting WhisperTRT transcription at {time.time():.6f}")
        
        result = model.transcribe(audio)
        text = result["text"].strip()
        
        transcription_end = time.time()
        end_timestamp = time.strftime('%H:%M:%S') + f".{int((transcription_end % 1) * 1000000):06d}"
        print(f"[Whisper] ✅ TRANSCRIPTION PROCESSING END: {end_timestamp}")
        
        transcription_time = transcription_end - transcription_start
        print(f"[Whisper] 🧠 WhisperTRT transcription completed at {transcription_end:.6f}, latency: {transcription_time:.6f}s")
        
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
        print(f"  🧠 WhisperTRT processing latency: {transcription_time:.3f}s")
        print(f"  📊 Total audio duration: {audio_duration:.3f}s")
        print(f"  ⚡ Efficiency RTF: {efficiency:.2f}x")
        print(f"[Whisper] 📜 Transcribed Text: \"{text}\"")
        print()  # Empty line for readability
        
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

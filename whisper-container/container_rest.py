from flask import Flask, request, jsonify
from whisper_trt import load_trt_model
import soundfile as sf
import numpy as np
import scipy.signal
import tempfile
import os
import time

app = Flask(__name__)
# Load TensorRT optimized model
model = load_trt_model("base.en")

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

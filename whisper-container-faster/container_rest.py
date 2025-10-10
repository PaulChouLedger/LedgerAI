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
import json
from pathlib import Path

# === Medical Vocabulary Management ===
def load_medical_terms():
    """Load medical terms from JSON file and build initial prompt"""
    medical_terms_file = Path("/app/medical_terms.json")
    
    if not medical_terms_file.exists():
        print("[Whisper] ⚠️ Medical terms file not found, using basic prompt")
        return "This is a medical conversation. Proper names and technical medical terms are important."
    
    try:
        with open(medical_terms_file, 'r') as f:
            terms_dict = json.load(f)
        
        # Sample terms from each category to build a concise but comprehensive prompt
        # We'll include high-priority organ systems and common terms
        priority_systems = ["cardiovascular", "respiratory", "gastrointestinal", 
                           "endocrine", "neurological", "common_symptoms", "medications"]
        
        sampled_terms = []
        for system in priority_systems:
            if system in terms_dict:
                # Take first 5 terms from each priority system
                sampled_terms.extend(terms_dict[system][:5])
        
        # Build the prompt
        terms_str = ", ".join(sampled_terms[:40])  # Limit to ~40 terms to keep prompt reasonable
        prompt = f"This is a medical conversation. Common terms include: {terms_str}. Proper names and technical medical terms are important."
        
        print(f"[Whisper] 📚 Loaded {sum(len(v) for v in terms_dict.values())} medical terms from {len(terms_dict)} categories")
        return prompt
        
    except Exception as e:
        print(f"[Whisper] ⚠️ Error loading medical terms: {e}")
        return "This is a medical conversation. Proper names and technical medical terms are important."

def save_medical_term(term, category="learned"):
    """Add a new medical term to the vocabulary (for future learning capability)"""
    medical_terms_file = Path("/app/medical_terms.json")
    
    try:
        with open(medical_terms_file, 'r') as f:
            terms_dict = json.load(f)
        
        if category not in terms_dict:
            terms_dict[category] = []
        
        if term.lower() not in [t.lower() for t in terms_dict[category]]:
            terms_dict[category].append(term)
            
            with open(medical_terms_file, 'w') as f:
                json.dump(terms_dict, f, indent=2)
            
            print(f"[Whisper] 📝 Added new term '{term}' to category '{category}'")
            return True
    except Exception as e:
        print(f"[Whisper] ⚠️ Error saving term: {e}")
    
    return False

# === Transcription Configuration ===
# Tune these parameters for your needs:
BEAM_SIZE = 10                     # Higher = better accuracy, slower (5=fast, 10=balanced, 20=best) - MUST be int
TEMPERATURE = 0.0                  # 0.0 = deterministic, 0.1+ = more creative
PATIENCE = 1.0                     # Wait time for better results (increased for better accuracy)
LENGTH_PENALTY = 1.0               # Slight penalty to prevent cutting off words

# Load medical terms and build initial prompt dynamically
INITIAL_PROMPT = load_medical_terms()

# Performance vs Accuracy Guide:
# BEAM_SIZE=5:  ~1x latency, good accuracy
# BEAM_SIZE=10: ~2x latency, better accuracy (current)
# BEAM_SIZE=20: ~4x latency, best accuracy

app = Flask(__name__)

# Check if model is available in the built-in cache
import os
model_name = os.getenv("WHISPER_MODEL", "distil-small.en")
cache_dir = "/app/cache/whisper"

print(f"[Whisper] 🚀 Loading faster-whisper model: {model_name}")
print(f"[Whisper] 📁 Cache directory: {cache_dir}")

# Map model names to their actual HuggingFace repo names
model_mapping = {
    "distil-small.en": "models--Systran--faster-distil-whisper-small.en",      # Fast, lower accuracy
    "small.en": "models--Systran--faster-small-whisper.en",                    # Better accuracy
    "medium.en": "models--Systran--faster-medium-whisper.en",                  # Much better for names
    "base.en": "models--Systran--faster-base-whisper.en",                      # Basic model
    "large-v3-turbo": "models--mobiuslabsgmbh--faster-whisper-large-v3-turbo", # Best accuracy, higher latency
    "distil-large-v3": "models--Systran--faster-distil-whisper-large-v3"       # Excellent accuracy, low latency ⭐ RECOMMENDED
}

# Get the actual model repo name
model_repo = model_mapping.get(model_name, f"models--Systran--faster-{model_name.replace('.', '-')}")
model_cache_path = f"/root/.cache/huggingface/hub/{model_repo}"

if os.path.exists(model_cache_path):
    print(f"[Whisper] ✅ Model found in cache: {model_name}")
else:
    print(f"[Whisper] ⚠️ Model not in cache, will download: {model_name}")

# Load GPU model - NO CPU FALLBACK
try:
    model = WhisperModel(model_name, device="cuda", compute_type="int8_float16", download_root="/root/.cache/huggingface/hub")
    print(f"[Whisper] ✅ GPU model '{model_name}' loaded successfully from HuggingFace cache")
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
    
    # Check for custom initial_prompt (for guiding spelling of names)
    custom_prompt = request.form.get("initial_prompt", INITIAL_PROMPT)
    
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
        
        # Audio properties (minimal logging)
        audio_duration = len(audio) / 16000
        print(f"[Whisper] 🔍 Audio duration: {audio_duration:.3f}s")
        
        # Check if audio is too quiet or silent (minimal logging)
        if np.sqrt(np.mean(audio**2)) < 0.001:
            print(f"[Whisper] ⚠️ Audio is very quiet")
        
        # Time model transcription
        transcription_start = time.time()
        
        # Initialize text variable
        text = ""
        
        try:
            # Using configurable transcription parameters
            print(f"[Whisper] 🧠 Starting transcription...")
            segments, info = model.transcribe(
                audio, 
                language="en",
                beam_size=BEAM_SIZE,
                temperature=TEMPERATURE,
                patience=PATIENCE,
                length_penalty=LENGTH_PENALTY,
                initial_prompt=custom_prompt
            )
            # Process segments
            segment_list = list(segments)
            text = " ".join([s.text.strip() for s in segment_list if s.text.strip()])
            print(f"[Whisper] 📝 Transcribed: '{text}'")
            
        except RuntimeError as e:
            print(f"[Whisper] ❌ Runtime error: {e}")
            if "cuDNN" in str(e) or "CUDNN_STATUS" in str(e):
                print(f"[Whisper] 🔍 cuDNN error - GPU/CUDA initialization issue")
                # Try to clear GPU memory and retry once
                import torch
                torch.cuda.empty_cache()
                print(f"[Whisper] 🔄 Cleared GPU memory, retrying...")
                try:
                    segments, info = model.transcribe(
                        audio, 
                        language="en",
                        beam_size=BEAM_SIZE,
                        temperature=TEMPERATURE,
                        initial_prompt=custom_prompt
                    )
                    segment_list = list(segments)
                    text = " ".join([s.text.strip() for s in segment_list if s.text.strip()])
                    print(f"[Whisper] ✅ Retry successful: '{text}'")
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
        
        # Show model transcription timing (important for performance monitoring)
        print(f"[Whisper] ⏱️ Model transcription: {transcription_time:.3f}s")
        
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

@app.route("/add_medical_term", methods=["POST"])
def add_medical_term():
    """Add a new medical term to the vocabulary"""
    data = request.json
    
    if not data or "term" not in data:
        return jsonify({"error": "Missing 'term' in request"}), 400
    
    term = data["term"]
    category = data.get("category", "learned")
    
    success = save_medical_term(term, category)
    
    if success:
        # Reload the prompt with new terms
        global INITIAL_PROMPT
        INITIAL_PROMPT = load_medical_terms()
        
        return jsonify({
            "success": True,
            "message": f"Added term '{term}' to category '{category}'",
            "updated_prompt": INITIAL_PROMPT
        })
    else:
        return jsonify({
            "success": False,
            "message": "Term already exists or error occurred"
        })

@app.route("/medical_terms", methods=["GET"])
def get_medical_terms():
    """Get all medical terms organized by category"""
    medical_terms_file = Path("/app/medical_terms.json")
    
    if not medical_terms_file.exists():
        return jsonify({"error": "Medical terms file not found"}), 404
    
    try:
        with open(medical_terms_file, 'r') as f:
            terms_dict = json.load(f)
        
        # Calculate statistics
        total_terms = sum(len(v) for v in terms_dict.values())
        
        return jsonify({
            "total_terms": total_terms,
            "categories": len(terms_dict),
            "terms_by_category": {k: len(v) for k, v in terms_dict.items()},
            "current_prompt": INITIAL_PROMPT,
            "all_terms": terms_dict
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

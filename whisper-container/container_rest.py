import os
import sys
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
import requests
from pathlib import Path

# === Medical Vocabulary Management ===
def get_medical_prompt():
    """
    Generate a medical conversational prompt to guide transcription.
    Includes common medical terms and context to improve accuracy.
    """
    return (
        "This is a medical conversation. Common medical terms include: "
        "pneumonia, antibiotic, diagnosis, treatment, symptom, patient, "
        "medication, dosage, infection, fever, pain, blood pressure, "
        "heart rate, respiratory, cardiovascular, gastrointestinal, "
        "neurological, diabetes, hypertension, asthma, allergy, "
        "prescription, therapy, examination, test, result, procedure, "
        "surgery, recovery, discharge, follow-up. "
        "Proper names of medications, medical conditions, and technical medical terms are important."
    )

# ============================================================================
# CONFIGURATION - Tune these values for your needs
# ============================================================================

# === Model Selection ===
MODEL_NAME = os.environ.get("WHISPER_MODEL", "distil-whisper/distil-large-v3.5-ct2")
# Options: "distil-small.en", "small.en", "medium.en", "base.en", "large-v3-turbo", "distil-large-v3", "distil-whisper/distil-large-v3.5-ct2"
# Accuracy: distil-small < small < medium < distil-large-v3 < distil-large-v3.5 ≈ large-v3-turbo
# Latency: distil-small (fastest) < small < medium < distil-large-v3 < distil-large-v3.5 < large-v3-turbo (slowest)
# Memory: Smaller models use less GPU memory
# Recommendation: "distil-small.en" for fastest (default), "distil-whisper/distil-large-v3.5-ct2" for best accuracy/speed balance

CACHE_DIR = "/app/cache/whisper"  # Model cache directory (internal use)

# === Transcription Parameters ===

BEAM_SIZE = 1
# Range: 1-20 (integer)
# Accuracy: Higher = better accuracy (more candidate paths evaluated)
# Latency: Higher = slower (exponential increase: 5≈1x, 10≈2x, 20≈4x latency)
# Memory: Higher = more GPU memory required
# Trade-off: 1=fastest/lowest latency, 5=fast/good, 10=balanced, 20=best accuracy/slowest
# ⚡ MINIMUM LATENCY: Set to 1 for fastest transcription

TEMPERATURE = 0.0
# Range: 0.0-1.0
# Accuracy: 0.0 = deterministic (best for accuracy), >0.0 = more variable outputs
# Latency: No significant impact
# Use: Keep at 0.0 for consistent, accurate transcriptions

PATIENCE = 1.0
# Range: 0.0-2.0
# Accuracy: Higher = better accuracy (waits longer for better results)
# Latency: Higher = slower (waits longer before finalizing)
# Trade-off: 1.0=minimum for beam_size=1 (required constraint: beam_size * patience >= 1)
# ⚡ MINIMUM LATENCY: Set to 1.0 (minimum required with beam_size=1)

LENGTH_PENALTY = 1.0
# Range: 0.0-2.0
# Accuracy: 1.0 = neutral, <1.0 = prefers shorter outputs, >1.0 = prefers longer outputs
# Latency: Minimal impact
# Use: 1.0 for balanced, adjust if outputs are too short/long

# === Compute Type (GPU Quantization) ===
COMPUTE_TYPE = "int8"
# Options: "int8" (fastest, lowest memory), "int8_float16" (compatible), "float16" (higher memory)
# Accuracy: int8 ≈ int8_float16 ≈ float16 (minimal difference, <1% accuracy loss with int8)
# Latency: int8 (fastest) < int8_float16 < float16 (slowest)
# Memory: int8 (lowest) < int8_float16 < float16 (highest)
# Recommendation: "int8" for best speed/memory with minimal accuracy loss

# === Initial Prompt Configuration ===
AUTO_DETECT_CONTAINER = True  # Automatically detect medical vs generic container
INITIAL_PROMPT_MEDICAL = None  # Auto-generated medical prompt (set at startup)
INITIAL_PROMPT_GENERIC = "This is a conversation."  # General conversation prompt
INITIAL_PROMPT_FALLBACK = "This is a conversation."  # Fallback if detection fails

# Purpose: Guides model on context/domain (e.g., medical terms, technical jargon)
# Accuracy: Including domain-specific terms improves recognition of specialized vocabulary
# Latency: Minimal impact (only affects first few tokens)
# Auto-detection: If AUTO_DETECT_CONTAINER=True, prompt is set based on active LLM container

# ============================================================================
# END CONFIGURATION
# ============================================================================

app = Flask(__name__)

# === Container Detection and Initial Prompt Setup ===
def detect_llm_container_type():
    """
    Detect if medical or generic LLM container is running.
    Returns: "medical", "generic", or "unknown"
    """
    try:
        # Both containers use port 11434, check health endpoint
        health_url = "http://localhost:11434/health"
        response = requests.get(health_url, timeout=2)
        if response.status_code == 200:
            health_data = response.json()
            # Check if it's medical container (has medical-specific fields)
            if "medical" in str(health_data).lower() or "navigator" in str(health_data).lower():
                return "medical"
            # Generic container indicators
            elif "generic" in str(health_data).lower() or "conversation" in str(health_data).lower():
                return "generic"
    except Exception as e:
        print(f"[Whisper] ⚠️ Could not detect container type: {e}")
    
    return "unknown"

def get_initial_prompt():
    """
    Get the appropriate initial prompt based on container type.
    Returns the prompt string to use for transcription.
    """
    if not AUTO_DETECT_CONTAINER:
        # Use fallback if auto-detection is disabled
        return INITIAL_PROMPT_FALLBACK
    
    container_type = detect_llm_container_type()
    
    if container_type == "medical":
        # Generate medical prompt with common medical terms
        medical_prompt = get_medical_prompt()
        print(f"[Whisper] 🏥 Medical container detected - using medical prompt")
        return medical_prompt
    elif container_type == "generic":
        print(f"[Whisper] 💬 Generic container detected - using general conversation prompt")
        return INITIAL_PROMPT_GENERIC
    else:
        print(f"[Whisper] ⚠️ Container type unknown - using fallback prompt")
        return INITIAL_PROMPT_FALLBACK

# Initialize the prompt based on detected container
INITIAL_PROMPT = get_initial_prompt()
print(f"[Whisper] 📝 Initial prompt set: '{INITIAL_PROMPT[:80]}...'")

print(f"[Whisper] 🚀 Loading faster-whisper model: {MODEL_NAME}")
print(f"[Whisper] 📁 Cache directory: {CACHE_DIR}")

# Map model names to their actual HuggingFace repo names
MODEL_MAPPING = {
    "distil-small.en": "models--Systran--faster-distil-whisper-small.en",      # Fast, lower accuracy
    "small.en": "models--Systran--faster-small-whisper.en",                    # Better accuracy
    "medium.en": "models--Systran--faster-medium-whisper.en",                  # Much better for names
    "base.en": "models--Systran--faster-base-whisper.en",                      # Basic model
    "large-v3-turbo": "models--mobiuslabsgmbh--faster-whisper-large-v3-turbo", # Best accuracy, higher latency
    "distil-large-v3": "models--Systran--faster-distil-whisper-large-v3",      # Excellent accuracy, low latency
    "distil-whisper/distil-large-v3.5-ct2": "models--distil-whisper--distil-large-v3.5-ct2"  # Best accuracy/speed balance (1.5x faster than turbo, better short-form accuracy)
}

# Get the actual model repo name
model_repo = MODEL_MAPPING.get(MODEL_NAME, f"models--Systran--faster-{MODEL_NAME.replace('.', '-')}")
model_cache_path = f"/root/.cache/huggingface/hub/{model_repo}"

# Check if model exists in cache
if os.path.exists(model_cache_path):
    print(f"[Whisper] ✅ Model found in cache: {MODEL_NAME}")
else:
    print(f"[Whisper] ⚠️ Model not in cache, will download: {MODEL_NAME}")

# Load GPU model optimized for speed and accuracy
try:
    model = WhisperModel(MODEL_NAME, device="cuda", compute_type=COMPUTE_TYPE, download_root="/root/.cache/huggingface/hub")
    print(f"[Whisper] ✅ GPU model '{MODEL_NAME}' loaded with {COMPUTE_TYPE} quantization (optimized for speed + accuracy)")
except Exception as e:
    print(f"[Whisper] ⚠️ {COMPUTE_TYPE} loading failed: {e}, trying int8_float16 fallback...")
    try:
        # Fallback to int8_float16 if primary compute type fails
        model = WhisperModel(MODEL_NAME, device="cuda", compute_type="int8_float16", download_root="/root/.cache/huggingface/hub")
        print(f"[Whisper] ✅ GPU model '{MODEL_NAME}' loaded with int8_float16 quantization (fallback)")
    except Exception as e2:
        print(f"[Whisper] ❌ GPU model loading failed: {e2}")
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
    # Use custom prompt from request, or fall back to configured INITIAL_PROMPT
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
            # Clear GPU cache before transcription to prevent OOM
            try:
                import torch
                torch.cuda.empty_cache()
            except ImportError:
                # torch not available - skip cache clearing (faster-whisper handles memory internally)
                pass
            
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
            error_str = str(e)
            print(f"[Whisper] ❌ Runtime error: {error_str}")
            
            # Handle out-of-memory errors - only reduce quality as last resort
            if "out of memory" in error_str.lower():
                print(f"[Whisper] 🔍 CUDA out of memory - attempting aggressive memory cleanup...")
                try:
                    import torch
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                    # Force garbage collection
                    import gc
                    gc.collect()
                    torch.cuda.empty_cache()
                except ImportError:
                    # torch not available - use gc only
                    import gc
                    gc.collect()
                
                # First retry: same settings but with fresh memory
                print(f"[Whisper] 🔄 Retrying with same settings after memory cleanup...")
                try:
                    segments, info = model.transcribe(
                        audio, 
                        language="en",
                        beam_size=BEAM_SIZE,
                        temperature=TEMPERATURE,
                        patience=PATIENCE,
                        length_penalty=LENGTH_PENALTY,
                        initial_prompt=custom_prompt
                    )
                    segment_list = list(segments)
                    text = " ".join([s.text.strip() for s in segment_list if s.text.strip()])
                    print(f"[Whisper] ✅ Retry successful after memory cleanup: '{text}'")
                except Exception as retry_error:
                    print(f"[Whisper] ⚠️ Retry failed: {retry_error}")
                    # Only reduce beam_size as last resort (sacrifices accuracy)
                    reduced_beam_size = max(5, BEAM_SIZE // 2)  # Don't go below 5 to maintain accuracy
                    print(f"[Whisper] 🔄 Last resort: retrying with beam_size={reduced_beam_size} (reduced accuracy)...")
                    try:
                        segments, info = model.transcribe(
                            audio, 
                            language="en",
                            beam_size=reduced_beam_size,
                            temperature=TEMPERATURE,
                            patience=PATIENCE,
                            length_penalty=LENGTH_PENALTY,
                            initial_prompt=custom_prompt
                        )
                        segment_list = list(segments)
                        text = " ".join([s.text.strip() for s in segment_list if s.text.strip()])
                        print(f"[Whisper] ✅ Retry with reduced beam_size successful: '{text}'")
                    except Exception as final_error:
                        print(f"[Whisper] ❌ All retry attempts failed: {final_error}")
                        sys.stdout.flush()
                        text = ""
            elif "cuDNN" in error_str or "CUDNN_STATUS" in error_str:
                print(f"[Whisper] 🔍 cuDNN error - GPU/CUDA initialization issue")
                # Try to clear GPU memory and retry once
                try:
                    import torch
                    torch.cuda.empty_cache()
                    print(f"[Whisper] 🔄 Cleared GPU memory, retrying...")
                except ImportError:
                    print(f"[Whisper] 🔄 Retrying without torch cache clearing...")
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
            "model": MODEL_NAME,
            "compute_type": COMPUTE_TYPE,
            "beam_size": BEAM_SIZE,
            "requests_processed": 0,
            "message": "No requests processed yet"
        })
    
    avg_processing_time = timing_stats["total_processing_time"] / timing_stats["total_requests"]
    avg_transcription_time = timing_stats["total_transcription_time"] / timing_stats["total_requests"]
    avg_audio_duration = timing_stats["total_audio_duration"] / timing_stats["total_requests"]
    overall_efficiency = timing_stats["total_audio_duration"] / timing_stats["total_transcription_time"] if timing_stats["total_transcription_time"] > 0 else 0
    
    return jsonify({
        "status": "healthy",
        "model": MODEL_NAME,
        "compute_type": COMPUTE_TYPE,
        "beam_size": BEAM_SIZE,
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
    """Medical terms are now embedded in prompt (no longer stored in JSON)"""
    return jsonify({
        "success": False,
        "message": "Medical terms are now embedded in the prompt. Edit get_medical_prompt() function to add terms."
    }), 400

@app.route("/medical_terms", methods=["GET"])
def get_medical_terms():
    """Get current medical prompt (no longer uses JSON file)"""
    return jsonify({
        "message": "Medical terms are now embedded in the prompt (no JSON file)",
        "current_prompt": INITIAL_PROMPT,
        "prompt_type": "medical" if "medical" in INITIAL_PROMPT.lower() else "generic"
    })

@app.route("/models/available", methods=["GET"])
def get_available_models():
    """Get list of available Whisper models baked into the container"""
    hub_cache_dir = "/root/.cache/huggingface/hub"
    available_models = []
    
    # Reverse mapping: directory name -> model name
    dir_to_model = {
        "models--Systran--faster-distil-whisper-small.en": "distil-small.en",
        "models--Systran--faster-small-whisper.en": "small.en",
        "models--Systran--faster-medium-whisper.en": "medium.en",
        "models--Systran--faster-base-whisper.en": "base.en",
        "models--mobiuslabsgmbh--faster-whisper-large-v3-turbo": "large-v3-turbo",
        "models--Systran--faster-distil-whisper-large-v3": "distil-large-v3",
        "models--distil-whisper--distil-large-v3.5-ct2": "distil-whisper/distil-large-v3.5-ct2"
    }
    
    # Scan the hub cache directory for baked-in models
    if os.path.exists(hub_cache_dir):
        try:
            for item in os.listdir(hub_cache_dir):
                # Check if this directory matches a known model pattern
                if item in dir_to_model:
                    model_dir = os.path.join(hub_cache_dir, item)
                    # Verify it's actually a directory (models are stored as directories)
                    if os.path.isdir(model_dir):
                        # Check if it has content (snapshots subdirectory for Systran models, or direct content for distil-whisper)
                        has_content = False
                        if os.path.exists(os.path.join(model_dir, "snapshots")):
                            # Systran models have snapshots/ subdirectory
                            snapshots_dir = os.path.join(model_dir, "snapshots")
                            if os.path.isdir(snapshots_dir) and os.listdir(snapshots_dir):
                                has_content = True
                        else:
                            # Direct content (like distil-whisper models)
                            if os.listdir(model_dir):
                                has_content = True
                        
                        if has_content:
                            model_name = dir_to_model[item]
                            available_models.append(model_name)
                            print(f"[Models] Found baked-in model: {model_name} (from {item})")
        except Exception as e:
            print(f"[Models] ⚠️ Error scanning models: {e}")
            import traceback
            traceback.print_exc()
    
    # Always include distil-small.en as it's the default fallback
    if "distil-small.en" not in available_models:
        available_models.append("distil-small.en")
    
    return jsonify({
        "available_models": sorted(available_models),
        "current_model": MODEL_NAME,
        "model_cache_dir": hub_cache_dir
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

# === container_rest.py — Aura Medical Container ===
# Minimal Flask API that delegates all logic to AdvancedMedicalNavigator
# All medical logic is handled by the fine-tuned LLM in advanced_medical_navigator.py

from flask import Flask, request, jsonify, stream_with_context, Response
import os
import threading
import logging
import json
import sys

# Add shared directory to path for base class import
sys.path.insert(0, '/shared')

# Import shared base class
from llm_base import BaseLLMContainer

# Import medical navigator (handles all logic)
from advanced_medical_navigator import AdvancedMedicalNavigator

app = Flask(__name__)

# Suppress verbose logging
werkzeug_logger = logging.getLogger('werkzeug')
werkzeug_logger.setLevel(logging.WARNING)

# === Initialize Base Container ===
base_container = BaseLLMContainer(
    service_name="aura-llm-medical",
    default_model_path="/models/Qwen2.5-1.5B-Instruct.Q4_K_M.gguf"
)

# === Model/LLM Config (using base class, but keeping for backward compatibility) ===
LLM_TEMPERATURE_SIMPLE = base_container.LLM_TEMPERATURE_SIMPLE
LLM_TOP_P = base_container.LLM_TOP_P
LLM_TOP_K = base_container.LLM_TOP_K
LLM_REPEAT_PENALTY = base_container.LLM_REPEAT_PENALTY
LLM_NUM_PREDICT_DEFAULT = base_container.LLM_NUM_PREDICT_DEFAULT
SIMPLE_N_CTX = base_container.SIMPLE_N_CTX
SIMPLE_CHAT_FORMAT = base_container.SIMPLE_CHAT_FORMAT
N_THREADS = base_container.N_THREADS
N_BATCH = base_container.N_BATCH
CACHE_PROMPT = True

# Use base class model resolution
SIMPLE_MODEL_PATH = base_container.resolve_model_path()

# Reference to LLM instance (will be set by base_container.load_model())
llm_simple = None

# === LLM Wrapper (using base class) ===
def extract_llm_response_content(response) -> str:
    """Extract text content from LLM response"""
    return base_container.extract_llm_response_content(response)

def llm_chat_simple(messages, max_tokens=None, temperature=None, stream=False, **kwargs):
    """Wrapper for LLM chat completion"""
    return base_container.llm_chat_simple(messages, max_tokens, temperature, stream, **kwargs)

# === Medical Navigator Instance ===
medical_navigator = None

def get_medical_navigator():
    """Get or create medical navigator instance"""
    global medical_navigator
    if medical_navigator is None:
        medical_navigator = AdvancedMedicalNavigator(llm_chat_fn=llm_chat_simple)
    return medical_navigator

# === Health Check ===
# Register health check using base class, with additional navigator info
@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint"""
    try:
        response = base_container.health_check_response({
            "navigator_loaded": medical_navigator is not None
        })
        status_code = 200 if response.get("status") == "ok" else 500
        return jsonify(response), status_code
    except Exception as e:
        return jsonify({
            "status": "error",
            "service": "aura-llm-medical",
            "error": str(e)
        }), 500

# === Medical Chat Endpoint ===
@app.route("/chat-tts", methods=["POST"])
def chat_tts():
    """Medical chat endpoint - delegates to AdvancedMedicalNavigator"""
    data = request.get_json()
    prompt = (data.get("prompt") or "").strip()
    session_id = (data.get("chat_id") or data.get("session_id") or "default").strip()
    stream = data.get("stream", True)  # Default to streaming for TTS
    
    if not prompt:
        return jsonify({"error": "Missing prompt"}), 400
    
    print(f"[Medical] 💬 Session: {session_id}, Prompt: '{prompt[:50]}...', Stream: {stream}")
    
    try:
        navigator = get_medical_navigator()
        
        if stream:
            # Streaming mode
            def generate_response():
                try:
                    result, token_stream = navigator.process_message(session_id, prompt, stream=True)
                    
                    # Wrap token stream with sentence tags for TTS compatibility using base class
                    for tagged_token in base_container.sentence_tag_stream(token_stream):
                        if tagged_token:
                            yield f"{tagged_token}\n"
                    
                    print(f"[Medical] ✅ Streamed response complete")
                except Exception as e:
                    print(f"[Medical] ❌ Error: {e}")
                    import traceback
                    traceback.print_exc()
                    yield "<sentence_start>\nI apologize, I encountered an error processing your request.\n<sentence_end>\n"
            
            return Response(
                stream_with_context(generate_response()),
                mimetype="text/plain",
                headers={
                    "Cache-Control": "no-cache",
                    "X-Accel-Buffering": "no",
                    "Connection": "keep-alive"
                }
            )
        else:
            # Non-streaming mode
            result = navigator.process_message(session_id, prompt, stream=False)
            response_text = result.get('response', '') or result.get('message', '') or result.get('question', '')
            return jsonify({"response": response_text})
            
    except Exception as e:
        print(f"[Medical] ❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

# === Generic Chat Endpoint (for compatibility) ===
@app.route("/chat-tg", methods=["POST"])
def chat_tg():
    """Generic chat endpoint - delegates to AdvancedMedicalNavigator"""
    return chat_tts()

if __name__ == "__main__":
    print("[Medical] 🚀 Starting Aura Medical LLM Container...")
    
    # Load model with GPU acceleration using base class
    print(f"[Medical] 📦 Loading model: {SIMPLE_MODEL_PATH}")
    n_gpu_layers = -1  # Offload all layers to GPU
    print(f"[Medical] 🚀 GPU acceleration: {n_gpu_layers} layers offloaded to GPU")
    
    # Override base class load_model to add GPU support
    from llama_cpp import Llama
    base_container.model_path = SIMPLE_MODEL_PATH
    base_container.llm_simple = Llama(
        model_path=SIMPLE_MODEL_PATH,
        n_ctx=SIMPLE_N_CTX,
        n_threads=N_THREADS,
        n_batch=N_BATCH,
        n_gpu_layers=n_gpu_layers,
        cache_prompt=CACHE_PROMPT,
        chat_format=SIMPLE_CHAT_FORMAT,
        use_mlock=True,
        use_mmap=True,
        verbose=False
    )
    base_container._model_loaded = True
    llm_simple = base_container.llm_simple  # Set global reference
    print(f"[Medical] ✅ Model loaded: {SIMPLE_MODEL_PATH}")
    
    # Initialize medical navigator
    print(f"[Medical] 🔧 Initializing AdvancedMedicalNavigator...")
    get_medical_navigator()
    print(f"[Medical] ✅ Medical Navigator initialized")
    
    print("[Medical] ✅ Medical Container ready!")
    print("[Medical] 🌐 Starting Flask server on 0.0.0.0:11434...")
    
    app.run(host="0.0.0.0", port=11434, threaded=True, debug=False)

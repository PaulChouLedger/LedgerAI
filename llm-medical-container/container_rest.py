# === container_rest.py — Aura Medical Container ===
# Minimal Flask API that delegates all logic to AdvancedMedicalNavigator
# All medical logic is handled by the fine-tuned LLM in advanced_medical_navigator.py

from flask import Flask, request, jsonify, stream_with_context, Response
from llama_cpp import Llama
import os
import threading
import logging
import json

# Import medical navigator (handles all logic)
from advanced_medical_navigator import AdvancedMedicalNavigator

app = Flask(__name__)

# Suppress verbose logging
werkzeug_logger = logging.getLogger('werkzeug')
werkzeug_logger.setLevel(logging.WARNING)

# === Thread Safety ===
llm_lock = threading.Lock()

# === Model/LLM Config ===
LLM_TEMPERATURE_SIMPLE = 0.7
LLM_TOP_P = 0.95
LLM_TOP_K = 40
LLM_REPEAT_PENALTY = 1.1
LLM_NUM_PREDICT_DEFAULT = 800
SIMPLE_N_CTX = 4096
SIMPLE_CHAT_FORMAT = "qwen"
N_THREADS = 8
N_BATCH = 256
CACHE_PROMPT = True

# === Model Path Resolution ===
def _resolve_model_path():
    """Resolve model path from app_settings.json or environment"""
    try:
        settings_path = "/app/data/app_settings.json"
        if os.path.isfile(settings_path):
            with open(settings_path, "r") as f:
                data = json.load(f)
                name = (data.get("llm_model") or "").strip()
                if name:
                    candidate = f"/models/{name}" if not name.startswith("/") else name
                    if os.path.isfile(candidate):
                        print(f"[Medical] 🎯 Using model from settings: {candidate}")
                        return candidate
    except Exception as e:
        print(f"[Medical] ⚠️ Failed reading app settings: {e}")
    
    env_path = os.getenv("SIMPLE_MODEL_PATH", "")
    if env_path and os.path.isfile(env_path):
        print(f"[Medical] 🛟 Using model from environment: {env_path}")
        return env_path
    
    fallback = "/models/Qwen2.5-1.5B-Instruct.Q4_K_M.gguf"
    print(f"[Medical] 🛟 Using default model: {fallback}")
    return fallback

SIMPLE_MODEL_PATH = _resolve_model_path()
llm_simple = None

# === LLM Wrapper ===
def extract_llm_response_content(response) -> str:
    """Extract text content from LLM response"""
    if isinstance(response, dict):
        if 'choices' in response and len(response['choices']) > 0:
            return response['choices'][0]['message']['content']
        elif 'content' in response:
            return response['content']
    return str(response)

def llm_chat_simple(messages, max_tokens=None, temperature=None, stream=False, **kwargs):
    """Wrapper for LLM chat completion"""
    if temperature is None:
        temperature = float(LLM_TEMPERATURE_SIMPLE)
    
    if max_tokens is None:
        max_tokens = int(LLM_NUM_PREDICT_DEFAULT)
    
    generation_params = {
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": kwargs.pop("top_p", float(LLM_TOP_P)),
        "top_k": kwargs.pop("top_k", int(LLM_TOP_K)),
        "repeat_penalty": kwargs.pop("repeat_penalty", float(LLM_REPEAT_PENALTY)),
        "stream": stream,
        **kwargs
    }
    
    generation_params["stop"] = []
    
    with llm_lock:
        try:
            response = llm_simple.create_chat_completion(**generation_params)
            if stream:
                return response
            return extract_llm_response_content(response)
        except Exception as e:
            print(f"[Medical] ❌ Error in llm_chat_simple: {e}")
            if stream:
                return iter([])
            return ""

# === Medical Navigator Instance ===
medical_navigator = None

def get_medical_navigator():
    """Get or create medical navigator instance"""
    global medical_navigator
    if medical_navigator is None:
        medical_navigator = AdvancedMedicalNavigator(llm_chat_fn=llm_chat_simple)
    return medical_navigator

# === Health Check ===
@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint"""
    try:
        simple_loaded = llm_simple is not None
        navigator_loaded = medical_navigator is not None
        
        return jsonify({
            "status": "ok",
            "service": "aura-llm-medical",
            "models": {
                "simple_loaded": simple_loaded,
                "simple_path": SIMPLE_MODEL_PATH
            },
            "navigator_loaded": navigator_loaded
        })
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
                    
                    # Wrap token stream with sentence tags for TTS compatibility
                    sentence_open = False
                    sentence_buffer = ""
                    SENTENCE_ENDINGS = ('.', '!', '?')
                    
                    # Send initial sentence_start tag
                    yield "<sentence_start>\n"
                    sentence_open = True
                    
                    # Stream tokens as they're generated
                    for token in token_stream:
                        if token:  # Skip empty tokens
                            token_clean = token.strip()
                            sentence_buffer += token
                            
                            # Check if token ends a sentence
                            if token_clean and token_clean[-1] in SENTENCE_ENDINGS:
                                # End current sentence
                                yield f"{token}\n"
                                yield "<sentence_end>\n"
                                sentence_buffer = ""
                                sentence_open = False
                                # Start next sentence immediately
                                yield "<sentence_start>\n"
                                sentence_open = True
                            else:
                                # Continue current sentence
                                yield f"{token}\n"
                    
                    # Close any remaining sentence
                    if sentence_open and sentence_buffer.strip():
                        yield "<sentence_end>\n"
                    
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
    
    # Load model
    print(f"[Medical] 📦 Loading model: {SIMPLE_MODEL_PATH}")
    n_gpu_layers = -1  # Offload all layers to GPU
    print(f"[Medical] 🚀 GPU acceleration: {n_gpu_layers} layers offloaded to GPU")
    
    llm_simple = Llama(
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
    print(f"[Medical] ✅ Model loaded: {SIMPLE_MODEL_PATH}")
    
    # Initialize medical navigator
    print(f"[Medical] 🔧 Initializing AdvancedMedicalNavigator...")
    get_medical_navigator()
    print(f"[Medical] ✅ Medical Navigator initialized")
    
    print("[Medical] ✅ Medical Container ready!")
    print("[Medical] 🌐 Starting Flask server on 0.0.0.0:11434...")
    
    app.run(host="0.0.0.0", port=11434, threaded=True, debug=False)

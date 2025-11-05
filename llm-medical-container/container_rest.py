# === container_rest.py — Aura Medical Container (Clinician Mode Architecture)
# All requests are handled by the clinician mode which provides:
# - Medical symptom assessment with adaptive diagnostic engine
# - Medical knowledge queries with RAG integration
# - Casual greetings and general medical conversation
# - Comprehensive physician-like medical assistance

from flask import Flask, request, jsonify, stream_with_context, Response
from llama_cpp import Llama
from dotenv import load_dotenv
import os, re, json, string, threading, time
from datetime import datetime, timedelta
from glob import glob
import requests

# Note: Validation functions removed - ML system handles all validation

# Import clinician mode for comprehensive medical assistance
from clinician_mode import ClinicianSession, is_clinician_trigger, get_clinician_session, handle_clinician_response

# Import modular RAG client (supports both GPU and CPU modes)
from rag import get_rag_client

app = Flask(__name__)
load_dotenv()

# === Thread Safety ===
llm_lock = threading.Lock()

# === Health Check Endpoint ===
@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint to verify models are loaded"""
    try:
        # Check if model is loaded
        simple_loaded = llm_simple is not None
        
        return jsonify({
            "status": "ok",
            "service": "aura-llm",
            "models": {
                "simple_loaded": simple_loaded,
                "simple_path": SIMPLE_MODEL_PATH
            }
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "service": "aura-llm",
            "error": str(e)
        }), 500

# === CPU FAISS Auto-Ingestion Endpoints ===
@app.route('/cpu-faiss/ingest', methods=['POST'])
def cpu_faiss_ingest():
    """Trigger CPU FAISS auto-ingestion manually"""
    try:
        # Get RAG client instance
        from rag import get_rag_client
        rag_client = get_rag_client()
        
        if not rag_client or not hasattr(rag_client, '_auto_ingest') or rag_client._auto_ingest is None:
            return jsonify({'error': 'CPU FAISS auto-ingestion not available'}), 500
        
        # Trigger manual scan
        result = rag_client._auto_ingest.scan_and_process()
        
        return jsonify({
            'status': 'success',
            'processed': result['processed'],
            'skipped': result['skipped'],
            'errors': result['errors'],
            'total_chunks': result['total_chunks'],
            'message': 'CPU FAISS auto-ingestion completed'
        })
        
    except Exception as e:
        logger.error(f"Error in CPU FAISS auto-ingestion: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/cpu-faiss/status', methods=['GET'])
def cpu_faiss_status():
    """Get CPU FAISS status"""
    try:
        # Get RAG client instance
        from rag import get_rag_client
        rag_client = get_rag_client()
        
        if not rag_client or not hasattr(rag_client, '_auto_ingest') or rag_client._auto_ingest is None:
            return jsonify({'error': 'CPU FAISS auto-ingestion not available'}), 500
        
        auto_ingest = rag_client._auto_ingest
        
        return jsonify({
            'status': 'active',
            'watching': auto_ingest.watching,
            'total_chunks': len(auto_ingest.chunks),
            'processed_files': len(auto_ingest.state.get('processed_files', {})),
            'input_directory': str(auto_ingest.input_dir),
            'cpu_embeddings_directory': str(auto_ingest.cpu_embeddings_dir),
            'model_name': auto_ingest.model_name
        })
        
    except Exception as e:
        logger.error(f"Error getting CPU FAISS status: {e}")
        return jsonify({'error': str(e)}), 500

# === Model Config ===
# Model configuration from .env only - no fallback defaults
SIMPLE_MODEL_PATH = os.environ["SIMPLE_MODEL_PATH"]
SIMPLE_N_CTX = int(os.environ["SIMPLE_N_CTX"])
SIMPLE_CHAT_FORMAT = os.environ["SIMPLE_CHAT_FORMAT"]

# Models will be loaded in __main__ block to prevent double loading
import os
import time

llm_simple = None

# Note: TRIAGE_DEFS is loaded automatically by triage.py when imported

# === Normalize text helper (used by router) ===
def normalize_text(text: str) -> str:
    """Normalize text for comparison"""
    if not text:
        return ""
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = ' '.join(text.split())
    return text

def clear_session_state(session_id: str):
    """Clear session state from storage"""
    try:
        # Create data/sessions directory if it doesn't exist
        sessions_dir = os.path.join("data", "sessions")
        os.makedirs(sessions_dir, exist_ok=True)
        
        session_file = f"session_{session_id}.json"
        session_path = os.path.join(sessions_dir, session_file)
        
        if os.path.exists(session_path):
            os.remove(session_path)
            print(f"[Container] 🗑️ Cleared session file: {session_path}")
        else:
            print(f"[Container] ℹ️ No session file to clear: {session_path}")
    except Exception as e:
        print(f"[Container] ⚠️ Error clearing session state: {e}")


def extract_llm_response_content(response) -> str:
    """
    Centralized extraction of text content from LLM response
    Handles both dict (JSON) and string formats from llama.cpp
    
    Args:
        response: LLM response (dict or string)
        
    Returns:
        Extracted text content
    """
    # If response is a dict (JSON response from LLM)
    if isinstance(response, dict):
        # Standard OpenAI-style response format
        if 'choices' in response and len(response['choices']) > 0:
            return response['choices'][0]['message']['content']
        # Alternative content format
        elif 'content' in response:
            return response['content']
    
    # If response is already a string, return it directly
    return str(response)


def stream_llm_response(messages, max_tokens=100):
    """
    Global streaming wrapper for LLM responses
    Yields text chunks as they're generated, reducing initial latency
    
    Args:
        messages: Chat messages for LLM
        max_tokens: Maximum tokens to generate
        
    Yields:
        Text chunks from LLM as they're generated
    """
    try:
        stream = llm_chat(messages, max_tokens=max_tokens, stream=True)
        
        for chunk in stream:
            # Extract content from streaming chunk
            if isinstance(chunk, dict):
                if 'choices' in chunk and len(chunk['choices']) > 0:
                    delta = chunk['choices'][0].get('delta', {})
                    content = delta.get('content', '')
                    if content:
                        yield content
    except Exception as e:
        print(f"[Container] ❌ Streaming error: {e}")
        yield ""


# === Non-streaming chat endpoint for Telegram ===
@app.route("/chat-tg", methods=["POST"])
def chat_tg():
    """
    Non-streaming chat endpoint for Telegram bot
    Uses SAME routing and logic as /chat, just returns single response instead of streaming
    """
    data = request.get_json()
    prompt = (data.get("prompt") or "").strip()
    session_id = (data.get("chat_id") or data.get("session_id") or "telegram_session").strip()
    do_reset = bool(data.get("reset"))
    
    if not prompt:
        return jsonify({"response": "Please describe your symptoms."})
    
    # Handle reset commands (same as /chat)
    prompt_norm = normalize_text(prompt)
    RESET_KEYWORDS = {"reset", "restart", "new session"}
    if any(k in prompt_norm for k in RESET_KEYWORDS):
        do_reset = True
    
    print(f"[Telegram] 💬 Session: {session_id}, Prompt: '{prompt[:50]}{'...' if len(prompt) > 50 else ''}', Reset: {do_reset}")
    
    # Handle session reset - PROPERLY CLEAR SESSION STATE
    if do_reset:
        print(f"[Telegram] 🔄 Resetting session: {session_id}")
        
        # Clear session state properly
        try:
            # Reset clinician session if exists
            from clinician_mode import reset_clinician_session
            reset_clinician_session(session_id)
            print(f"[Telegram] ✅ Clinician session reset: {session_id}")
        except Exception as e:
            print(f"[Telegram] ⚠️ Error resetting clinician session: {e}")
        
        # Always return reset confirmation
        return jsonify({"response": "Session reset. Start again with your symptoms."})
    
    try:
        # All requests go to clinician mode
        print(f"[Telegram] 🎯 Using clinician mode for all requests")
        
        # Helper to collect streamed response into single string
        def collect_stream(generator):
            """Collect streamed response and clean it"""
            response_parts = []
            for chunk in filter_think_blocks(generator):
                chunk = chunk.strip()
                if chunk:
                    # Remove sentence markers
                    chunk = chunk.replace('<sentence_start>', '').replace('<sentence_end>', '')
                    chunk = chunk.strip()
                    if chunk:
                        response_parts.append(chunk)
            return ' '.join(response_parts).strip()
        
        # Dispatch to unified medical mode
        try:
            # For Telegram, we don't need immediate fillers since it's text-based
            # Engine debug output will flow through naturally (no duplication needed)
            print(f"[Container] 🔍 Telegram request: '{prompt[:50]}{'...' if len(prompt) > 50 else ''}'")
            response = handle_clinician_response(prompt, session_id, llm_chat, llm_chat_simple)
            print(f"[Container] ✅ Telegram response processed")
            print(f"[Container] 🔍 Response type: {type(response)}")
            if isinstance(response, dict):
                print(f"[Container] 🔍 Response keys: {list(response.keys())}")
                if 'debug' in response:
                    print(f"[Container] 🔍 Debug info present: {response['debug'] is not None}")
                else:
                    print(f"[Container] ⚠️ No debug key in response")
            
            # Check if response includes question (dict) or is simple text (str)
            if isinstance(response, dict):
                # Return question + debug info for Telegram
                debug_info = response.get('debug')
                print(f"[Container] 🔍 Debug info in response: {debug_info is not None}")
                if debug_info:
                    print(f"[Container] 🔍 Debug info keys: {list(debug_info.keys())}")
                    if 'engine_debug_output' in debug_info:
                        print(f"[Container] 🔍 Engine debug output: {len(debug_info['engine_debug_output'])} lines")
                        if debug_info['engine_debug_output']:
                            print(f"[Container] 🔍 First few debug lines: {debug_info['engine_debug_output'][:3]}")
                        else:
                            print(f"[Container] ⚠️ Engine debug output is empty")
                    else:
                        print(f"[Container] ⚠️ No engine_debug_output key in debug info")
                
                # Format response for Telegram
                # Handle empathetic statement + question (with pause indicator)
                response_text = ""
                if response.get('message'):
                    response_text = response['message']
                
                if response.get('question'):
                    if response_text:
                        # Add pause indicator if both message and question exist
                        if response.get('has_pause'):
                            response_text += "\n\n"  # Extra spacing for pause
                        else:
                            response_text += "\n"
                    response_text += response['question']
                
                # Fallback if neither exists
                if not response_text:
                    response_text = response.get('question', response.get('message', ''))
                
                telegram_response = {
                    "response": response_text,
                    "debug": debug_info  # Include debug info if available
                }
                return jsonify(telegram_response)
            else:
                # Simple text response
                return jsonify({"response": response})
        except Exception as e:
            print(f"[Container] ❌ Error in clinician mode (non-streaming): {e}")
            print(f"[Container] 📋 Error type: {type(e).__name__}")
            print(f"[Container] 📍 Error location: {e.__traceback__.tb_frame.f_code.co_filename}:{e.__traceback__.tb_lineno}")
            print(f"[Container] 🔍 Full traceback:")
            import traceback
            traceback.print_exc()
            
            # NO FALLBACKS - re-raise the actual error
            raise e
            
    except Exception as e:
        print(f"[Telegram] ❌ Error in chat-simple: {e}")
        print(f"[Telegram] 📋 Error type: {type(e).__name__}")
        print(f"[Telegram] 📍 Error location: {e.__traceback__.tb_frame.f_code.co_filename}:{e.__traceback__.tb_lineno}")
        print(f"[Telegram] 🔍 Full traceback:")
        import traceback
        traceback.print_exc()
        # NO FALLBACKS - re-raise the actual error
        raise e


# === Streaming chat endpoint for TTS/Voice ===
@app.route("/chat-tts", methods=["POST"])
def chat_tts():
    """
    Main chat endpoint using modular architecture

    Routes requests to CLINICIAN mode for all interactions:
    - Casual greetings and general conversation
    - Medical knowledge queries with GPU-accelerated RAG
    - Symptom assessment with adaptive diagnostic engine
    - OLDCARTS-based questioning with guideline matching
    """
    data = request.get_json()
    prompt = (data.get("prompt") or "").strip()
    session_id = (data.get("chat_id") or data.get("session_id") or "").strip() or None
    do_reset = bool(data.get("reset"))

    # Handle reset commands
    prompt_norm = normalize_text(prompt)
    RESET_KEYWORDS = {"reset", "restart", "new session"}
    if any(k in prompt_norm for k in RESET_KEYWORDS):
        do_reset = True

    if not prompt:
        return jsonify({"error": "Missing prompt"}), 400

    print(f"[Aura-LLM] 💬 Session: {session_id}, Prompt: '{prompt[:50]}{'...' if len(prompt) > 50 else ''}', Reset: {do_reset}")

    # Handle session reset (simplified - just return reset message)
    if do_reset:
        if prompt_norm in RESET_KEYWORDS:
            def generate_reset():
                yield "<sentence_start>\n🔄 Session reset. Start again with your symptoms.\n<sentence_end>\n"
            # Filter think blocks at container level (though unlikely here)
            return Response(stream_with_context(filter_think_blocks(generate_reset())), mimetype="text/plain")

    # All requests go to clinician mode
    print(f"[Aura-LLM] 🎯 Using clinician mode for all requests")

    # Dispatch to clinician mode
    def generate_clinician():
        try:
            print("[Container] 🔄 Using dynamic medical assessment for CLINICIAN")
            
            # Process the actual response
            response = handle_clinician_response(prompt, session_id, llm_chat, llm_chat_simple)
            
            print(f"[Container] ✅ Got response from unified medical session")
            
            # Check if response is dict or simple text (str)
            if isinstance(response, dict):
                # Handle empathetic statement + question (with pause)
                if response.get('message') and response.get('question'):
                    # Stream empathetic statement first
                    message_text = response.get('message', '')
                    yield "<sentence_start>\n"
                    yield f"{message_text}\n"
                    yield "<sentence_end>\n"
                    
                    # Add pause if indicated
                    if response.get('has_pause'):
                        yield "<pause>\n"  # Pause marker for TTS
                    
                    # Then stream question
                    question_text = response.get('question', '')
                    yield "<sentence_start>\n"
                    yield f"{question_text}\n"
                    yield "<sentence_end>\n"
                elif response.get('question'):
                    # Just question
                    question_text = response.get('question', '')
                    yield "<sentence_start>\n"
                    yield f"{question_text}\n"
                    yield "<sentence_end>\n"
                elif response.get('message'):
                    # Just message
                    message_text = response.get('message', '')
                    yield "<sentence_start>\n"
                    yield f"{message_text}\n"
                    yield "<sentence_end>\n"
                else:
                    # Fallback
                    yield "<sentence_start>\n"
                    yield "I'm processing your response...\n"
                    yield "<sentence_end>\n"
            elif isinstance(response, str):
                # Simple text response (no filler)
                yield "<sentence_start>\n"
                yield f"{response}\n"
                yield "<sentence_end>\n"
            else:
                # Fallback
                yield "<sentence_start>\n"
                yield "I'm processing your response...\n"
                yield "<sentence_end>\n"
        except Exception as e:
            print(f"[Container] ❌ Error in clinician mode: {e}")
            print(f"[Container] 📋 Error type: {type(e).__name__}")
            print(f"[Container] 📍 Error location: {e.__traceback__.tb_frame.f_code.co_filename}:{e.__traceback__.tb_lineno}")
            print(f"[Container] 🔍 Full traceback:")
            import traceback
            traceback.print_exc()
            
            # NO FALLBACKS - re-raise the actual error
            raise e

    # Filter think blocks at container level
    return Response(stream_with_context(filter_think_blocks(generate_clinician())), mimetype="text/plain")





# === Stream Filtering with Garbage Detection ===
def filter_think_blocks(generator):
    """
    Filter and validate streaming output from all modes
    
    - Filters <think> tags (if model uses them)
    - Detects repetitive garbage output (e.g., "333333...")
    - Provides fallback response if garbage detected
    """
    from collections import Counter
    
    accumulated_output = []
    garbage_detected = False
    
    for token in generator:
        if token and token.strip():
            accumulated_output.append(token)
            
            # Early garbage detection - check every 100 chars
            full_output = ''.join(accumulated_output)
            
            # Extract just the text content (without sentence tags)
            import re
            text_only = re.sub(r'<sentence_start>|<sentence_end>|\n', '', full_output)
            
            if len(text_only) > 50 and len(text_only) % 100 < 20:  # Check periodically
                char_counts = Counter(text_only.lower())
                if char_counts:
                    most_common_char, most_common_count = char_counts.most_common(1)[0]
                    repetition_ratio = most_common_count / len(text_only)
                    
                    if repetition_ratio > 0.6:  # 60%+ same character = garbage
                        print(f"[Container] ⚠️ GARBAGE DETECTED: char='{most_common_char}', ratio={repetition_ratio:.2f}, output='{text_only[:100]}'")
                        garbage_detected = True
                        break  # Stop consuming stream
            
            yield token
    
    # If garbage was detected, provide fallback response
    if garbage_detected:
        print(f"[Container] 🔄 Using fallback response due to garbage detection")
        # Clear any previous output and send fallback
        yield "<sentence_start>\nI'm sorry, I had trouble processing that. Could you tell me more about what's going on?\n<sentence_end>\n"



# === Helper Functions ===

def load_state(session_id: str) -> dict:
    """Load session state from file"""
    import json
    import os
    
    if not session_id:
        return {}
    
    state_file = f"/app/data/sessions/{session_id}.json"
    
    try:
        if os.path.exists(state_file):
            with open(state_file, 'r') as f:
                return json.load(f)
    except Exception as e:
        print(f"[State] ❌ Error loading state for {session_id}: {e}")
    
    return {}

def save_state(state: dict, session_id: str) -> None:
    """Save session state to file"""
    import json
    import os
    
    if not session_id:
        return
    
    # Ensure directory exists
    os.makedirs("/app/data/sessions", exist_ok=True)
    
    state_file = f"/app/data/sessions/{session_id}.json"
    
    try:
        with open(state_file, 'w') as f:
            json.dump(state, f, indent=2)
    except Exception as e:
        print(f"[State] ❌ Error saving state for {session_id}: {e}")

def reset_session_state(session_id: str) -> dict:
    """Reset session state while preserving user name"""
    state = load_state(session_id)
    user_name = state.get("user_name")

    reset_state = {
        "condition": None, "step_index": 0, "answers": [], "flags": {},
        "last_key": None, "user_name": user_name,
        "active_pathway": None, "entered_pathway": False,
        "updated_at": None, "phrasing_history": [], "detailed_symptoms": [],
        "original_complaint": None, "expanded_prompt": None, "mode": None
    }

    save_state(reset_state, session_id)
    print(f"[Aura-LLM] 🔄 Session reset for session_id: {session_id}")
    return reset_state


def llm_chat(messages, max_tokens=None, temperature=None, stream=False, **kwargs):
    """
    Wrapper for LLM chat completion with thread safety and speed optimizations
    
    Args:
        messages: Chat messages
        max_tokens: Max tokens to generate (default: 100)
        temperature: Sampling temperature (default: use model config)
        stream: Enable streaming (default: False)
        **kwargs: Additional LLM parameters
    """
    # Apply centralized speed optimizations
    if temperature is None:
        temperature = float(os.environ["LLM_TEMPERATURE_SIMPLE"])
    
    # Handle max_tokens: use LLM_NUM_PREDICT as default if not provided
    if max_tokens is None:
        num_predict_env = os.getenv("LLM_NUM_PREDICT")
        if num_predict_env and num_predict_env.isdigit():
            max_tokens = int(num_predict_env)
        else:
            raise ValueError("LLM_NUM_PREDICT must be set in environment")
    
    generation_params = {
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": kwargs.pop("top_p", float(os.environ["LLM_TOP_P"])),
        "top_k": kwargs.pop("top_k", int(os.environ["LLM_TOP_K"])),
        "repeat_penalty": kwargs.pop("repeat_penalty", float(os.environ["LLM_REPEAT_PENALTY"])),
        "presence_penalty": kwargs.pop("presence_penalty", float(os.environ["LLM_PRESENCE_PENALTY"])),
        "frequency_penalty": kwargs.pop("frequency_penalty", float(os.environ["LLM_FREQUENCY_PENALTY"])),
        "stream": stream,
        **kwargs
    }
    # Optional stop sequences from environment
    stop_env = os.getenv("LLM_STOP", "").strip()
    stop_sequences = [s for s in stop_env.split(",") if s] if stop_env else []
    
    # Add reasoning-specific stop sequences to prevent internal reasoning
    # These are patterns that indicate reasoning/explanation, not the actual question
    reasoning_stop_sequences = [
        "\n\nHere's a",
        "\n\nHere is a",
        "\n\nAlternatively:",
        "\n\nThis question uses",
        "\n\nIt also",
        "\n\nwhich are",
        "\n\nThis uses",
        "\n\nAlternatively,",
        "\nAlternatively:",
        "\nHere's a",
        "\nHere is a",
    ]
    
    # Combine environment stop sequences with reasoning stop sequences
    generation_params["stop"] = stop_sequences + reasoning_stop_sequences
    
    with llm_lock:
        try:
            if llm_simple is None:
                raise RuntimeError("No LLM model available (simple model not loaded)")
            
            response = llm_simple.create_chat_completion(**generation_params)
            # If streaming, return the generator directly
            if stream:
                return response
            # For non-streaming, extract and return just the text content
            # This makes llm_chat() easier to use (returns strings, not dicts)
            return extract_llm_response_content(response)
        except Exception as e:
            print(f"[LLM] ❌ Error in llm_chat: {e}")
            if stream:
                # Return empty generator for streaming
                return iter([])
            return ""  # Return empty string on error


def llm_chat_simple(messages, max_tokens=None, temperature=None, stream=False, **kwargs):
    """
    Wrapper for SIMPLE LLM (Llama-1B) chat completion - for templates and validation
    
    Args:
        messages: Chat messages
        max_tokens: Max tokens to generate (default from LLM_NUM_PREDICT)
        temperature: Sampling temperature (default: use model config)
        stream: Enable streaming (default: False)
        **kwargs: Additional LLM parameters
    """
    if temperature is None:
        temperature = float(os.environ["LLM_TEMPERATURE_SIMPLE"])
    
    # Handle max_tokens: use LLM_NUM_PREDICT as default if not provided
    if max_tokens is None:
        num_predict_env = os.getenv("LLM_NUM_PREDICT")
        if num_predict_env and num_predict_env.isdigit():
            max_tokens = int(num_predict_env)
        else:
            raise ValueError("LLM_NUM_PREDICT must be set in environment")
    
    generation_params = {
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": kwargs.pop("top_p", float(os.environ["LLM_TOP_P"])),
        "top_k": kwargs.pop("top_k", int(os.environ["LLM_TOP_K"])),
        "repeat_penalty": kwargs.pop("repeat_penalty", float(os.environ["LLM_REPEAT_PENALTY"])),
        "presence_penalty": kwargs.pop("presence_penalty", float(os.environ["LLM_PRESENCE_PENALTY"])),
        "frequency_penalty": kwargs.pop("frequency_penalty", float(os.environ["LLM_FREQUENCY_PENALTY"])),
        "stream": stream,
        **kwargs
    }
    # Optional stop sequences from environment
    stop_env = os.getenv("LLM_STOP", "").strip()
    stop_sequences = [s for s in stop_env.split(",") if s] if stop_env else []
    
    # Add reasoning-specific stop sequences to prevent internal reasoning
    reasoning_stop_sequences = [
        "\n\nHere's a",
        "\n\nHere is a",
        "\n\nAlternatively:",
        "\n\nThis question uses",
        "\n\nIt also",
        "\n\nwhich are",
        "\n\nThis uses",
        "\n\nAlternatively,",
        "\nAlternatively:",
        "\nHere's a",
        "\nHere is a",
    ]
    
    # Combine environment stop sequences with reasoning stop sequences
    generation_params["stop"] = stop_sequences + reasoning_stop_sequences
    
    with llm_lock:  # Shared lock for both models
        try:
            response = llm_simple.create_chat_completion(**generation_params)
            if stream:
                return response
            return extract_llm_response_content(response)
        except Exception as e:
            print(f"[LLM-Simple] ❌ Error in llm_chat_simple: {e}")
            if stream:
                return iter([])
            return ""


def llm_chat_once(messages, **kwargs):
    """Single LLM call for NLG rewriting (used by triage)"""
    return llm_chat(messages, **kwargs)


# === Server Startup ===

if __name__ == "__main__":
    # Load model ONLY when running as main script (prevents double loading on import)
    
    print(f"[LLM] 🚀 Loading model: {SIMPLE_MODEL_PATH}")
    print(f"[LLM] ⚙️  Config: n_ctx={SIMPLE_N_CTX}, format={SIMPLE_CHAT_FORMAT}")
    
    # Check if model file exists and get file info
    if not os.path.exists(SIMPLE_MODEL_PATH):
        print(f"[LLM] ❌ Model file not found: {SIMPLE_MODEL_PATH}")
        exit(1)
    else:
        # Get file size and modification time
        file_stat = os.stat(SIMPLE_MODEL_PATH)
        file_size_mb = file_stat.st_size / (1024 * 1024)
        mod_time = time.ctime(file_stat.st_mtime)
        print(f"[LLM] 📁 Model file found locally: {file_size_mb:.1f}MB, modified: {mod_time}")
        print(f"[LLM] 🔍 File path: {SIMPLE_MODEL_PATH}")
    
    print(f"[LLM] 🧠 Initializing Llama model (this may take a while for large models)...")
    start_time = time.time()
    llm_simple = Llama(
        model_path=SIMPLE_MODEL_PATH,
        n_ctx=SIMPLE_N_CTX,
        n_gpu_layers=32,  # Use fewer layers for simple model on Orin32
        n_threads=6,
        chat_format=SIMPLE_CHAT_FORMAT,
        use_mlock=True,
        use_mmap=True,
        verbose=False,
        temperature=float(os.environ["LLM_TEMPERATURE_SIMPLE"]),
        top_p=float(os.environ["LLM_TOP_P"]),
        top_k=int(os.environ["LLM_TOP_K"]),
        repeat_penalty=float(os.environ["LLM_REPEAT_PENALTY"])
    )
    load_time = time.time() - start_time
    print(f"[LLM] ✅ Simple model loaded: {SIMPLE_MODEL_PATH} (took {load_time:.1f}s)")
    
    print("[Aura-LLM] 🚀 Starting Aura LLM Container (Modular Architecture)")
    print("[Aura-LLM] 📋 Configuration:")
    print("  - CLINICIAN: Intelligent medical assistant with adaptive diagnostic engine")
    print("    • Handles casual greetings, medical knowledge queries, and symptom assessment")
    print("    • Uses GPU-accelerated RAG for medical knowledge and guideline matching")
    print("    • Using Llama-3.2-1B model for all tasks")

    app.run(host='0.0.0.0', port=11434, debug=False)


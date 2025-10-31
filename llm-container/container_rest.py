# === container_rest.py — Aura Unified LLM Container
# Handles both generic and medical conversations based on USE_MEDICAL_MODE toggle
# 
# Medical Mode (USE_MEDICAL_MODE=true):
# - Medical symptom assessment with adaptive diagnostic engine
# - Medical knowledge queries with RAG integration
# - Casual greetings and general medical conversation
# - Comprehensive physician-like medical assistance
#
# Generic Mode (USE_MEDICAL_MODE=false):
# - General conversation
# - RAG-powered document Q&A
# - Flexible LLM interactions

from flask import Flask, request, jsonify, stream_with_context, Response
from dotenv import load_dotenv
import os, re, json, string, threading, time
from datetime import datetime, timedelta
from glob import glob
import requests

# TensorRT-LLM imports
try:
    from tensorrt_llm_wrapper import TensorRTLLMWrapper
    from tensorrt_models_config import get_engine_dir, get_tokenizer_dir, validate_engine_dir
    TENSORRT_LLM_AVAILABLE = True
except ImportError as e:
    print(f"[TensorRT-LLM] ⚠️ TensorRT-LLM wrapper not available: {e}")
    TENSORRT_LLM_AVAILABLE = False
    TensorRTLLMWrapper = None

# Note: Validation functions removed - ML system handles all validation

# Medical/Generic mode toggle
USE_MEDICAL_MODE = os.getenv("USE_MEDICAL_MODE", "true").lower() == "true"  # Default to medical mode

# Import clinician mode for medical conversations (only if enabled)
if USE_MEDICAL_MODE:
    try:
        from clinician_mode import ClinicianSession, is_clinician_trigger, get_clinician_session, handle_clinician_response
        MEDICAL_MODE_AVAILABLE = True
        print(f"[Container] ✅ Medical mode ENABLED - All requests handled by clinician mode")
    except ImportError as e:
        print(f"[Container] ❌ Medical mode enabled but clinician_mode not available: {e}")
        MEDICAL_MODE_AVAILABLE = False
        USE_MEDICAL_MODE = False  # Fallback to generic mode
else:
    MEDICAL_MODE_AVAILABLE = False
    print(f"[Container] ℹ️ Medical mode DISABLED - Generic conversation mode active")

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
# Single model (Qwen3-4B-Instruct) for all tasks
SIMPLE_MODEL_PATH = os.getenv("SIMPLE_MODEL_PATH", "/models/Qwen3-4B-Instruct-2507-Q4_K_M.gguf")
SIMPLE_N_CTX = int(os.getenv("SIMPLE_N_CTX", "2048"))
SIMPLE_CHAT_FORMAT = os.getenv("SIMPLE_CHAT_FORMAT", "qwen")

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


# === Medical Chat Endpoint ===
@app.route("/chat-medical", methods=["POST"])
def chat_medical():
    """
    Medical conversation endpoint
    Handles medical symptom assessment, knowledge queries, and clinician interactions
    """
    data = request.get_json()
    prompt = (data.get("prompt") or "").strip()
    session_id = (data.get("chat_id") or data.get("session_id") or "default").strip()
    do_reset = bool(data.get("reset"))
    
    if not MEDICAL_MODE_AVAILABLE:
        return jsonify({"error": "Medical mode not available. Ensure USE_MEDICAL_MODE=true and medical modules are loaded."}), 503
    
    if not prompt:
        return jsonify({"error": "Prompt is required"}), 400
    
    # Handle reset
    prompt_norm = normalize_text(prompt)
    RESET_KEYWORDS = {"reset", "restart", "new session"}
    if any(k in prompt_norm for k in RESET_KEYWORDS):
        do_reset = True
    
    if do_reset:
        try:
            from clinician_mode import reset_clinician_session
            reset_clinician_session(session_id)
        except Exception as e:
            print(f"[Medical] ⚠️ Error resetting session: {e}")
        return jsonify({"response": "Session reset. Start again with your symptoms."})
    
    try:
        print(f"[Medical] 💬 Session: {session_id}, Prompt: '{prompt[:50]}...'")
        response = handle_clinician_response(prompt, session_id, llm_chat, llm_chat_simple)
        
        # Format response
        if isinstance(response, dict):
            response_text = response.get('message', '')
            if response.get('question'):
                if response_text:
                    response_text += "\n\n" if response.get('has_pause') else "\n"
                response_text += response['question']
            if not response_text:
                response_text = response.get('question', response.get('message', ''))
            
            return jsonify({
                "response": response_text,
                "debug": response.get('debug')
            })
        else:
            return jsonify({"response": response})
            
    except Exception as e:
        print(f"[Medical] ❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# === Generic Chat Endpoint ===
@app.route("/chat-generic", methods=["POST"])
def chat_generic():
    """
    Generic conversation endpoint
    Handles general conversation, RAG Q&A, and non-medical interactions
    """
    data = request.get_json()
    prompt = (data.get("prompt") or "").strip()
    session_id = (data.get("chat_id") or data.get("session_id") or "default").strip()
    stream = data.get("stream", False)
    max_tokens = data.get("max_tokens", 200)
    use_rag = data.get("use_rag", True)
    
    if not prompt:
        return jsonify({"error": "Prompt is required"}), 400
    
    try:
        print(f"[Generic] 💬 Session: {session_id}, Prompt: '{prompt[:50]}...'")
        
        # Get RAG context if enabled
        rag_context = None
        if use_rag:
            try:
                rag_client = get_rag_client()
                if rag_client:
                    results = rag_client.search(query=prompt, k=3)
                    if results:
                        rag_context = "\n\n".join([f"[{i+1}] {r['text']}" for i, r in enumerate(results)])
                        print(f"[Generic] ✅ RAG: Retrieved {len(results)} chunks")
            except Exception as e:
                print(f"[Generic] ⚠️ RAG search failed: {e}")
        
        # Build messages
        messages = [
            {"role": "system", "content": "You are a helpful AI assistant. Be concise, friendly, and helpful."},
            {"role": "user", "content": prompt}
        ]
        if rag_context:
            messages[0]["content"] += f"\n\nRelevant context:\n{rag_context}"
        
        # Get response
        if stream:
            def generate():
                response_gen = llm_chat_simple(messages, max_tokens=max_tokens, stream=True, temperature=0.7)
                for chunk in response_gen:
                    if isinstance(chunk, dict) and 'choices' in chunk:
                        delta = chunk['choices'][0].get('delta', {})
                        content = delta.get('content', '')
                        if content:
                            yield content
                    elif isinstance(chunk, str):
                        yield chunk
            return Response(stream_with_context(generate()), mimetype='text/plain')
        else:
            response = llm_chat_simple(messages, max_tokens=max_tokens, stream=False, temperature=0.7)
            
            # Extract content
            if isinstance(response, dict):
                content = response.get('choices', [{}])[0].get('message', {}).get('content', '') or str(response)
            else:
                content = response
            
            return jsonify({
                "response": content,
                "session_id": session_id,
                "used_rag": rag_context is not None
            })
            
    except Exception as e:
        print(f"[Generic] ❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# === Non-streaming chat endpoint for Telegram (routes based on USE_MEDICAL_MODE) ===
@app.route("/chat-tg", methods=["POST"])
def chat_tg():
    """
    Non-streaming chat endpoint for Telegram bot
    Routes to /chat-medical or /chat-generic based on USE_MEDICAL_MODE toggle
    """
    # Route to appropriate endpoint based on toggle
    if USE_MEDICAL_MODE and MEDICAL_MODE_AVAILABLE:
        print(f"[Telegram] 🎯 Routing to medical endpoint")
        return chat_medical()
    else:
        print(f"[Telegram] 🎯 Routing to generic endpoint")
        return chat_generic()


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

    # Route based on USE_MEDICAL_MODE toggle
    if USE_MEDICAL_MODE and MEDICAL_MODE_AVAILABLE:
        print(f"[Aura-LLM] 🎯 Medical mode ENABLED - Routing to clinician mode")
        
        # Dispatch to clinician mode
        def generate_clinician():
            try:
                print("[Container] 🔄 Using dynamic medical assessment for CLINICIAN")
                
                # Check if this will be a simple operation (no filler needed)
                def will_use_simple_llm(prompt_text):
                    """Predict if the operation will use simple patterns (no filler needed)"""
                    prompt_lower = prompt_text.lower().strip()
                    
                    # Simple operations (no filler needed):
                    # - Age answers: "35", "35 years old", "thirty five"
                    # - Sex answers: "male", "female", "man", "woman"
                    # - Simple clarifications
                    
                    # Age patterns
                    age_patterns = [
                        r'^\d+\.?$',  # Just numbers: "35" or "35."
                        r'^\d+\s*years?\s*old\.?$',  # "35 years old" or "35 years old."
                        r'^i\'?m\s+\d+\.?$',  # "I'm 35" or "I'm 35."
                        r'^i\s+am\s+\d+\.?$',  # "I am 35" or "I am 35."
                        r'^(thirty|forty|fifty|sixty|seventy|eighty|ninety)',  # "thirty five"
                        r'^(twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety)\s+(one|two|three|four|five|six|seven|eight|nine)$'
                    ]
                    
                    # Sex patterns
                    sex_patterns = [
                        r'^(male|female|man|woman|m|f)\.?$',
                        r'^(i am|i\'m)\s+(male|female|a man|a woman)\.?$'
                    ]
                    
                    # Check for age patterns
                    import re
                    for pattern in age_patterns:
                        if re.match(pattern, prompt_lower):
                            return True
                    
                    # Check for sex patterns
                    for pattern in sex_patterns:
                        if re.match(pattern, prompt_lower):
                            return True
                    
                    # Default to complex operation (needs filler)
                    return False
                
                # Determine if we need a filler based on predicted operation complexity
                will_use_simple = will_use_simple_llm(prompt)
                
                if will_use_simple:
                    # Simple operation - no filler needed
                    print(f"[Container] ⚡ Simple operation - no filler needed")
                else:
                    # Complex operation - use filler
                    from thinking_fillers import get_filler
                    immediate_filler = get_filler('question_generation', use_audio=True)
                    filler_text = immediate_filler['text']
                    print(f"[Container] 💬 IMMEDIATE filler for complex operation: '{filler_text}'")
                    
                    # Stream filler immediately
                    yield "<sentence_start>\n"
                    yield f"{filler_text}\n"
                    yield "<sentence_end>\n"
                
                # Now process the actual response in the background
                response = handle_clinician_response(prompt, session_id, llm_chat, llm_chat_simple)
                
                print(f"[Container] ✅ Got response from unified medical session")
                
                # Check if response includes filler (dict) or is simple text (str)
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
    
    else:
        # Generic mode - handle general conversation
        print(f"[Aura-LLM] 🎯 Generic mode ENABLED - Handling general conversation")
        
        def generate_generic():
            try:
                print("[Container] 💬 Generic conversation mode")
                
                # Simple generic conversation using LLM
                messages = [
                    {"role": "system", "content": "You are a helpful AI assistant. Be concise, friendly, and helpful."},
                    {"role": "user", "content": prompt}
                ]
                
                # Use simple LLM for generic conversation
                response = llm_chat_simple(messages, max_tokens=200, temperature=0.7)
                
                if isinstance(response, dict):
                    # Extract content from OpenAI-format response
                    content = response.get('choices', [{}])[0].get('message', {}).get('content', '')
                    if not content:
                        content = str(response)
                else:
                    content = response
                
                yield "<sentence_start>\n"
                yield f"{content}\n"
                yield "<sentence_end>\n"
                
            except Exception as e:
                print(f"[Container] ❌ Error in generic mode: {e}")
                import traceback
                traceback.print_exc()
                yield "<sentence_start>\n"
                yield "I apologize, but I encountered an error processing your request.\n"
                yield "<sentence_end>\n"
        
        return Response(stream_with_context(filter_think_blocks(generate_generic())), mimetype="text/plain")





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


def llm_chat(messages, max_tokens=100, temperature=None, stream=False, **kwargs):
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
    
    generation_params = {
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": kwargs.pop("top_p", float(os.getenv("LLM_TOP_P", "0.85"))),
        "top_k": kwargs.pop("top_k", int(os.getenv("LLM_TOP_K", "30"))),
        "repeat_penalty": kwargs.pop("repeat_penalty", float(os.getenv("LLM_REPEAT_PENALTY", "1.15"))),
        "presence_penalty": kwargs.pop("presence_penalty", float(os.getenv("LLM_PRESENCE_PENALTY", "0.0"))),
        "frequency_penalty": kwargs.pop("frequency_penalty", float(os.getenv("LLM_FREQUENCY_PENALTY", "0.0"))),
        "stream": stream,
        **kwargs
    }
    # Optional stop sequences and num_predict override
    stop_env = os.getenv("LLM_STOP", "").strip()
    if stop_env:
        generation_params["stop"] = [s for s in stop_env.split(",") if s]
    num_predict_env = os.getenv("LLM_NUM_PREDICT")
    if num_predict_env and num_predict_env.isdigit():
        generation_params["max_tokens"] = int(num_predict_env)
    
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


def llm_chat_simple(messages, max_tokens=100, temperature=None, stream=False, **kwargs):
    """
    Wrapper for SIMPLE LLM (Llama-1B) chat completion - for templates and validation
    
    Args:
        messages: Chat messages
        max_tokens: Max tokens to generate (default: 100)
        temperature: Sampling temperature (default: use model config)
        stream: Enable streaming (default: False)
        **kwargs: Additional LLM parameters
    """
    if temperature is None:
        temperature = float(os.environ["LLM_TEMPERATURE_SIMPLE"])
    
    generation_params = {
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": kwargs.pop("top_p", float(os.getenv("LLM_TOP_P", "0.85"))),
        "top_k": kwargs.pop("top_k", int(os.getenv("LLM_TOP_K", "30"))),
        "repeat_penalty": kwargs.pop("repeat_penalty", float(os.getenv("LLM_REPEAT_PENALTY", "1.15"))),
        "presence_penalty": kwargs.pop("presence_penalty", float(os.getenv("LLM_PRESENCE_PENALTY", "0.0"))),
        "frequency_penalty": kwargs.pop("frequency_penalty", float(os.getenv("LLM_FREQUENCY_PENALTY", "0.0"))),
        "stream": stream,
        **kwargs
    }
    stop_env = os.getenv("LLM_STOP", "").strip()
    if stop_env:
        generation_params["stop"] = [s for s in stop_env.split(",") if s]
    num_predict_env = os.getenv("LLM_NUM_PREDICT")
    if num_predict_env and num_predict_env.isdigit():
        generation_params["max_tokens"] = int(num_predict_env)
    
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
    
    if not TENSORRT_LLM_AVAILABLE:
        print("[LLM] ❌ TensorRT-LLM is not available. Cannot start container.")
        exit(1)
    
    # Get TensorRT-LLM engine directory
    engine_dir = get_engine_dir()
    tokenizer_dir = get_tokenizer_dir()
    
    print(f"[LLM] 🚀 Loading TensorRT-LLM engine from: {engine_dir}")
    print(f"[LLM] ⚙️  Config: format={SIMPLE_CHAT_FORMAT}")
    
    # Validate engine directory
    if not validate_engine_dir(engine_dir):
        print(f"[LLM] ❌ Invalid engine directory: {engine_dir}")
        print(f"[LLM] 💡 Ensure TensorRT-LLM engine is built and available at: {engine_dir}")
        exit(1)
    
    print(f"[LLM] 🧠 Initializing TensorRT-LLM model (this may take a while for large models)...")
    start_time = time.time()
    
    try:
        llm_simple = TensorRTLLMWrapper(
            engine_dir=engine_dir,
            tokenizer_dir=tokenizer_dir
        )
        load_time = time.time() - start_time
        print(f"[LLM] ✅ TensorRT-LLM model loaded: {engine_dir} (took {load_time:.1f}s)")
    except Exception as e:
        print(f"[LLM] ❌ Failed to load TensorRT-LLM model: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
    
    print("[Aura-LLM] 🚀 Starting Aura LLM Container (Modular Architecture)")
    print("[Aura-LLM] 📋 Configuration:")
    print("  - CLINICIAN: Intelligent medical assistant with adaptive diagnostic engine")
    print("    • Handles casual greetings, medical knowledge queries, and symptom assessment")
    print("    • Uses GPU-accelerated RAG for medical knowledge and guideline matching")
    print("    • Using Llama-3.2-1B model for all tasks")

    app.run(host='0.0.0.0', port=11434, debug=False)


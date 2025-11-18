# === container_rest.py — Aura Medical Container (Direct Routing Architecture)
# Simplified architecture using Advanced Medical Navigator:
# - Advanced Medical Navigator (default, only option) - hybrid LLM/RAG/FAISS
#
# Architecture:
# container_rest.py → advanced_medical_navigator.py
#
# No intermediate layers - cleaner, simpler, easier to debug.

from flask import Flask, request, jsonify, stream_with_context, Response
from llama_cpp import Llama
import os, re, json, string, threading, time
from datetime import datetime, timedelta
from glob import glob
import requests
from typing import Dict, Callable

# Import modular RAG client (supports both GPU and CPU modes)
from rag import get_rag_client
import numpy as np

# RAG Embedding API wrapper
class RAGEmbeddingAPI:
    """Wrapper for RAG client's embedding service"""
    def __init__(self):
        self.rag_client = get_rag_client()
    
    def encode(self, texts: list) -> list:
        """Encode texts to embeddings"""
        embeddings = self.rag_client.embed(texts)
        if embeddings:
            return [np.array(emb, dtype=np.float32) for emb in embeddings]
        else:
            raise RuntimeError("RAG embedding failed")

AdvancedMedicalNavigator = None
MEDICAL_NAVIGATOR_AVAILABLE = False


def _ensure_medical_navigator_import():
    global AdvancedMedicalNavigator, MEDICAL_NAVIGATOR_AVAILABLE
    if MEDICAL_NAVIGATOR_AVAILABLE:
        return True
    try:
        from advanced_medical_navigator import AdvancedMedicalNavigator as ImportedNavigator
        AdvancedMedicalNavigator = ImportedNavigator
        MEDICAL_NAVIGATOR_AVAILABLE = True
        print("[Container] ✅ Advanced Medical Navigator imported")
        return True
    except ImportError as e:
        MEDICAL_NAVIGATOR_AVAILABLE = False
        print(f"[Container] ⚠️ Medical Navigator not available: {e}")
        return False

app = Flask(__name__)

# === Singleton Instances (Expensive to Create, Reused Across Sessions) ===
_global_medical_rule_engine = None
_global_medical_navigator = None

def get_medical_rule_engine(embedding_api):
    """Get or create singleton medical rule engine (expensive FAISS indexing, reuse!)"""
    global _global_medical_rule_engine
    
    if _global_medical_rule_engine is None:
        print("[Container] 🔧 Initializing Medical Rule Engine (one-time FAISS indexing)...")
        try:
            from ml.medical_rule_engine import MedicalRuleEngine
            _global_medical_rule_engine = MedicalRuleEngine(embedding_model=embedding_api)
            print(f"[Container] ✅ Medical Rule Engine initialized (FAISS indexes built once)")
        except Exception as e:
            print(f"[Container] ❌ Failed to initialize Medical Rule Engine: {e}")
            return None
    else:
        print(f"[Container] ♻️  Reusing Medical Rule Engine (FAISS already built)")
    
    return _global_medical_rule_engine

def get_medical_navigator(llm_chat_fn, embedding_model=None):
    """Get or create singleton medical navigator (FAISS + LLM decision)"""
    global _global_medical_navigator
    if not _ensure_medical_navigator_import():
        raise RuntimeError("Advanced Medical Navigator not available")

    if _global_medical_navigator is None:
        print("[Container] 🔧 Initializing Advanced Medical Navigator (FAISS + LLM, one-time setup)...")
        _global_medical_navigator = AdvancedMedicalNavigator(
            llm_chat_fn=llm_chat_fn,
            embedding_model=embedding_model
        )
        print(f"[Container] ✅ Advanced Medical Navigator initialized (FAISS + LLM)")
    
    return _global_medical_navigator

# === Session Management (Per-User State) ===
active_sessions: Dict[str, Dict] = {}

def get_or_create_session(session_id: str) -> Dict:
    """Get or create session for specific user"""
    global active_sessions
    
    if session_id not in active_sessions:
        print(f"[Container] 🔧 Creating new session: {session_id}")
        active_sessions[session_id] = {
            'created_at': datetime.now(),
            'last_activity': datetime.now()
        }
    else:
        print(f"[Container] 🔄 Reusing session: {session_id}")
        active_sessions[session_id]['last_activity'] = datetime.now()
    
    return active_sessions[session_id]

def reset_session(session_id: str):
    """Reset session state"""
    global _global_medical_navigator
    
    print(f"[Container] 🔄 Resetting session: {session_id}")
    
    if _global_medical_navigator and session_id in _global_medical_navigator.sessions:
        del _global_medical_navigator.sessions[session_id]
        print(f"[Container] ✅ Medical navigator session deleted")
    
    # Clear session from active_sessions
    if session_id in active_sessions:
        del active_sessions[session_id]

def cleanup_inactive_sessions():
    """Clean up old inactive sessions (>2 hours)"""
    global active_sessions
    cutoff_time = datetime.now() - timedelta(hours=2)
    sessions_to_remove = []
    
    for session_id, session in active_sessions.items():
        if session.get('last_activity', datetime.now()) < cutoff_time:
            sessions_to_remove.append(session_id)
    
    for session_id in sessions_to_remove:
        del active_sessions[session_id]
        if _global_medical_navigator and session_id in _global_medical_navigator.sessions:
            del _global_medical_navigator.sessions[session_id]
    
    if sessions_to_remove:
        print(f"[Container] 🗑️  Cleaned up {len(sessions_to_remove)} inactive sessions")

# === Thread Safety ===
llm_lock = threading.Lock()

# === Global instances (initialized at startup) ===
rag_api = None

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

# === Model/LLM Config (hardcoded for easy tuning) ===
LLM_TEMPERATURE_SIMPLE = 0.4
LLM_TOP_P = 0.95
LLM_TOP_K = 40
LLM_REPEAT_PENALTY = 1.1
LLM_PRESENCE_PENALTY = 0.0
LLM_FREQUENCY_PENALTY = 0.0
LLM_NUM_PREDICT_DEFAULT = 220
SIMPLE_N_CTX = 1024
SIMPLE_CHAT_FORMAT = "qwen"
N_THREADS = 8
N_BATCH = 256  # Reduced from 512 for faster generation (smaller batches = lower latency)
CACHE_PROMPT = True

# RAG Mode toggle: "CPU", "GPU", or "OFF" (resolved from app_settings.json if present)
def _resolve_rag_mode():
    try:
        settings_path = "/app/data/app_settings.json"
        if os.path.isfile(settings_path):
            with open(settings_path, "r") as f:
                data = json.load(f)
                mode = (data.get("rag_mode") or "").strip().upper()
                if mode in ("CPU", "GPU", "OFF"):
                    print(f"[LLM] 🎛️ RAG_MODE from settings: {mode}")
                    return mode
                elif mode:
                    print(f"[LLM] ⚠️ Invalid rag_mode '{mode}' in settings; using default")
    except Exception as e:
        print(f"[LLM] ⚠️ Failed reading rag_mode from settings: {e}")
    default_mode = "CPU"
    print(f"[LLM] 🛟 Using default RAG_MODE: {default_mode}")
    return default_mode

RAG_MODE = _resolve_rag_mode()

# === Model Config (auto-resolve from app settings for model path only) ===
def _resolve_model_path():
    """
    Determine model path priority:
    1) app_settings.json llm_model (filename) -> /models/<filename> if exists
    2) SIMPLE_MODEL_PATH from env
    3) Default fallback
    """
    try:
        settings_path = "/app/data/app_settings.json"
        if os.path.isfile(settings_path):
            with open(settings_path, "r") as f:
                data = json.load(f)
                name = (data.get("llm_model") or "").strip()
                if name:
                    candidate = f"/models/{name}" if not name.startswith("/") else name
                    if os.path.isfile(candidate):
                        print(f"[LLM] 🎯 Using model from settings: {candidate}")
                        return candidate
                    else:
                        print(f"[LLM] ⚠️ Model from settings not found: {candidate}")
    except Exception as e:
        print(f"[LLM] ⚠️ Failed reading app settings: {e}")
    fallback = "/models/Qwen2.5-1.5B-Instruct.Q4_K_M.gguf"
    print(f"[LLM] 🛟 Using default model: {fallback}")
    return fallback

SIMPLE_MODEL_PATH = _resolve_model_path()

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
    
    # Handle session reset
    if do_reset:
        print(f"[Telegram] 🔄 Resetting session: {session_id}")
        reset_session(session_id)
        return jsonify({"response": "Session reset. Start again with your symptoms."})
    
    try:
        # Get or create session
        get_or_create_session(session_id)
        
        # Cleanup inactive sessions periodically (10% chance)
        import random
        if random.randint(1, 10) == 1:
            cleanup_inactive_sessions()
        
        # Advanced Medical Navigator is the default and only option
        if not _ensure_medical_navigator_import():
            raise RuntimeError("Advanced Medical Navigator not available")
        
        print(f"[Container] 🔍 Telegram request: '{prompt[:50]}{'...' if len(prompt) > 50 else ''}'")
        print(f"[Telegram] 🔀 Using Advanced Medical Navigator")
        
        # Initialize singleton (FAISS + LLM decision)
        navigator = get_medical_navigator(llm_chat, embedding_model=rag_api)
        
        # Process message through navigator
        response = navigator.process_message(session_id=session_id, user_message=prompt)
        print(f"[Container] ✅ Navigator response processed")
        
        # Format response for Telegram
        print(f"[Container] 🔍 Response type: {type(response)}")
        
        if isinstance(response, dict):
            print(f"[Container] 🔍 Response keys: {list(response.keys())}")
            if 'debug' in response and response['debug']:
                debug = response['debug']
                engine_debug = debug.get('engine') if isinstance(debug, dict) else None
                if engine_debug:
                    print(engine_debug)
                internal_debug = debug.get('internal') if isinstance(debug, dict) else None
                if internal_debug:
                    for line in internal_debug:
                        print(line)
            
            # Extract response text from dict
            response_text = response.get('response', '')  # Navigator uses 'response'
            if not response_text:
                response_text = response.get('message', '')  # Fallback for 'message' key
            if not response_text:
                response_text = response.get('question', '')  # Or 'question'
            
            # Handle message + question format
            if response.get('message') and response.get('question'):
                response_text = response['message']
                if response.get('has_pause'):
                    response_text += "\n\n"
                else:
                    response_text += "\n"
                response_text += response['question']
            
            telegram_response = {
                "response": response_text,
                "debug": response.get('debug') or response.get('metadata')
            }
            return jsonify(telegram_response)
        else:
            # Simple text response
            return jsonify({"response": str(response)})
            
    except Exception as e:
        print(f"[Container] ❌ Error processing request: {e}")
        print(f"[Container] 📋 Error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# === Streaming chat endpoint for TTS/Voice ===
@app.route("/chat-tts", methods=["POST"])
def chat_tts():
    """
    Main chat endpoint using modular architecture

    Routes requests to CLINICIAN mode for all interactions:
    - Casual greetings and general conversation
    - Medical knowledge queries with GPU-accelerated RAG
    - Symptom assessment with medical navigator
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

    # Handle session reset
    if do_reset:
        reset_session(session_id)
        def generate_reset():
            yield "<sentence_start>\n🔄 Session reset. Start again with your symptoms.\n<sentence_end>\n"
        return Response(stream_with_context(filter_think_blocks(generate_reset())), mimetype="text/plain")

    # Get or create session
    get_or_create_session(session_id)
    
    # Advanced Medical Navigator is the default and only option
    if not _ensure_medical_navigator_import():
        raise RuntimeError("Advanced Medical Navigator not available")

    # Dispatch to medical engine with streaming
    def generate_medical_response():
        try:
            # rag_api is module-level global (initialized at startup)
            print(f"[TTS] 🔀 Using Advanced Medical Navigator (streaming)")
                
                # Initialize singleton (LLM-only, no medical_rule_engine or embedding_model needed)
            navigator = get_medical_navigator(llm_chat, embedding_model=rag_api)
                
            # STREAMING: Use streaming mode to get tokens as they're generated
            result = navigator.process_message(session_id=session_id, user_message=prompt, stream=True)
            
            # Check if streaming is supported (returns tuple of (response_dict, token_stream))
            if isinstance(result, tuple) and len(result) == 2:
                response_dict, token_stream = result
                print(f"[Container] ✅ Streaming enabled - tokens will be yielded as generated")
                
                # Stream tokens with sentence splitting - each sentence gets its own tags
                # Note: Session storage is handled by the navigator's token_stream generator
                # Process token stream and split into sentences for natural TTS playback
                sentence_buffer = ""
                sentence_open = False
                
                for token in token_stream:
                    token_stripped = token.strip()
                    
                    # Special handling for standalone dashes: they start new sentences for list items
                    if token_stripped == '-':
                        # Close previous sentence if open
                        if sentence_open:
                            yield "<sentence_end>\n"
                            sentence_buffer = ""
                        # Start new sentence for list item (dash is first word)
                        sentence_open = True
                        yield "<sentence_start>\n"
                        yield f"{token}\n"
                        sentence_buffer = token
                        continue
                    
                    # Normal token processing
                    if not sentence_open:
                        sentence_open = True
                        yield "<sentence_start>\n"
                    
                    # Ensure newline after each token so HTTP clients using iter_lines() receive tokens promptly
                    yield f"{token}\n"
                    sentence_buffer += token
                    
                    # Check if we've reached a sentence boundary
                    # 1. Sentence endings: . ! ? (period, exclamation, question mark)
                    if token_stripped and token_stripped[-1] in ('.', '!', '?'):
                        yield "<sentence_end>\n"
                        sentence_buffer = ""
                        sentence_open = False
                    # 2. Colons: split for list items (e.g., "include:" starts a list)
                    elif token_stripped and token_stripped[-1] == ':':
                        yield "<sentence_end>\n"
                        sentence_buffer = ""
                        sentence_open = False
                
                # Close any remaining sentence
                if sentence_open:
                    yield "<sentence_end>\n"
                
                print(f"[Container] ✅ Streamed response complete (session updated by navigator)")
                return
            
            # Fallback: Non-streaming mode (original behavior)
            print(f"[Container] ⚠️ Streaming not available - using blocking mode")
            response = navigator.process_message(session_id=session_id, user_message=prompt, stream=False)
            
            print(f"[Container] ✅ Got response from medical engine")
            
            # Stream response to TTS - send as chunks for batching
            if isinstance(response, dict):
                # Handle empathetic statement + question (with pause)
                if response.get('message') and response.get('question'):
                    # Stream empathetic statement first (as single chunk for batching)
                    message_text = response.get('message', '')
                    yield "<sentence_start>\n"
                    yield f"{message_text}\n"
                    yield "<sentence_end>\n"
                    
                    # Add pause if indicated
                    if response.get('has_pause'):
                        yield "<pause>\n"  # Pause marker for TTS
                    
                    # Then stream question (as single chunk for batching)
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
                elif response.get('response'):
                    # Navigator uses 'response' key - split into sentences for better batching
                    response_text = response.get('response', '')
                    # Split by sentence boundaries (periods, exclamation, question marks, colons, dashes)
                    # Split on sentence endings and list markers
                    sentences = re.split(r'([.!?]+\s+|:\s+|-\s+)', response_text)
                    current_sentence = ""
                    for part in sentences:
                        current_sentence += part
                        part_stripped = part.strip()
                        # If we have a complete sentence ending, yield it
                        if part_stripped and part_stripped[-1] in '.!?:':
                            yield "<sentence_start>\n"
                            yield f"{current_sentence.strip()}\n"
                            yield "<sentence_end>\n"
                            current_sentence = ""
                        # Also split on standalone dashes (list items)
                        elif part_stripped == '-' or (len(part_stripped) == 1 and part_stripped == '-'):
                            # Close previous sentence if any
                            if current_sentence.strip() and not current_sentence.strip().endswith('-'):
                                prev_sentence = current_sentence.rstrip('-').strip()
                                if prev_sentence:
                                    yield "<sentence_start>\n"
                                    yield f"{prev_sentence}\n"
                                    yield "<sentence_end>\n"
                            # Start new sentence with dash
                            yield "<sentence_start>\n"
                            yield f"-\n"
                            current_sentence = ""
                    # Yield any remaining text
                    if current_sentence.strip():
                        yield "<sentence_start>\n"
                        yield f"{current_sentence.strip()}\n"
                        yield "<sentence_end>\n"
                else:
                    # Fallback
                    yield "<sentence_start>\n"
                    yield "I'm processing your response...\n"
                    yield "<sentence_end>\n"
            elif isinstance(response, str):
                # Simple text response - split into sentences for better batching
                # Split by sentence boundaries (periods, exclamation, question marks, colons, dashes)
                sentences = re.split(r'([.!?]+\s+|:\s+|-\s+)', response)
                current_sentence = ""
                for part in sentences:
                    current_sentence += part
                    part_stripped = part.strip()
                    # If we have a complete sentence ending, yield it
                    if part_stripped and part_stripped[-1] in '.!?:':
                        yield "<sentence_start>\n"
                        yield f"{current_sentence.strip()}\n"
                        yield "<sentence_end>\n"
                        current_sentence = ""
                    # Also split on standalone dashes (list items)
                    elif part_stripped == '-' or (len(part_stripped) == 1 and part_stripped == '-'):
                        # Close previous sentence if any
                        if current_sentence.strip() and not current_sentence.strip().endswith('-'):
                            prev_sentence = current_sentence.rstrip('-').strip()
                            if prev_sentence:
                                yield "<sentence_start>\n"
                                yield f"{prev_sentence}\n"
                                yield "<sentence_end>\n"
                        # Start new sentence with dash
                        yield "<sentence_start>\n"
                        yield f"-\n"
                        current_sentence = ""
                # Yield any remaining text
                if current_sentence.strip():
                    yield "<sentence_start>\n"
                    yield f"{current_sentence.strip()}\n"
                    yield "<sentence_end>\n"
            else:
                # Fallback
                yield "<sentence_start>\n"
                yield "I'm processing your response...\n"
                yield "<sentence_end>\n"
        except Exception as e:
            print(f"[Container] ❌ Error in medical engine: {e}")
            import traceback
            traceback.print_exc()
            raise e

    # Filter think blocks at container level
    return Response(stream_with_context(filter_think_blocks(generate_medical_response())), mimetype="text/plain")





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
        temperature = float(LLM_TEMPERATURE_SIMPLE)
    
    # Handle max_tokens: use LLM_NUM_PREDICT as default if not provided
    if max_tokens is None:
        max_tokens = int(LLM_NUM_PREDICT_DEFAULT)
    
    generation_params = {
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": kwargs.pop("top_p", float(LLM_TOP_P)),
        "top_k": kwargs.pop("top_k", int(LLM_TOP_K)),
        "repeat_penalty": kwargs.pop("repeat_penalty", float(LLM_REPEAT_PENALTY)),
        "presence_penalty": kwargs.pop("presence_penalty", float(LLM_PRESENCE_PENALTY)),
        "frequency_penalty": kwargs.pop("frequency_penalty", float(LLM_FREQUENCY_PENALTY)),
        "stream": stream,
        **kwargs
    }
    # Optional stop sequences from environment
    stop_sequences = []
    
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
        temperature = float(LLM_TEMPERATURE_SIMPLE)
    
    # Handle max_tokens: use LLM_NUM_PREDICT as default if not provided
    if max_tokens is None:
        max_tokens = int(LLM_NUM_PREDICT_DEFAULT)
    
    generation_params = {
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": kwargs.pop("top_p", float(LLM_TOP_P)),
        "top_k": kwargs.pop("top_k", int(LLM_TOP_K)),
        "repeat_penalty": kwargs.pop("repeat_penalty", float(LLM_REPEAT_PENALTY)),
        "presence_penalty": kwargs.pop("presence_penalty", float(LLM_PRESENCE_PENALTY)),
        "frequency_penalty": kwargs.pop("frequency_penalty", float(LLM_FREQUENCY_PENALTY)),
        "stream": stream,
        **kwargs
    }
    # Optional stop sequences from environment
    stop_sequences = []
    
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
    # Initialize RAG embedding API (module-level variable, no global needed in __main__ block)
    if RAG_MODE in ("CPU", "GPU"):
        print(f"[Container] 🔧 Initializing RAG embedding API (mode: {RAG_MODE})...")
        try:
            rag_api = RAGEmbeddingAPI()
            test_embedding = rag_api.encode(["test"])
            rag_client = get_rag_client()
            rag_mode_display = "GPU" if rag_client.use_gpu else "CPU"
            print(f"[Container] ✅ RAG embedding API initialized ({rag_mode_display} mode)")
            if not rag_client.use_gpu:
                # Show CPU index status
                if hasattr(rag_client, '_cpu_chunks'):
                    chunk_count = len(rag_client._cpu_chunks) if rag_client._cpu_chunks else 0
                    print(f"[Container] 📊 RAG CPU index: {chunk_count} chunks available")
        except Exception as e:
            print(f"[Container] ⚠️ RAG API not available: {e}")
            rag_api = None
    else:
        print("[Container] ℹ️ RAG_MODE=OFF — skipping RAG embedding API initialization")
        rag_api = None
    
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
        n_gpu_layers=-1,  # -1 = offload all layers to GPU for maximum acceleration
        n_threads=N_THREADS,
        n_batch=N_BATCH,
        cache_prompt=CACHE_PROMPT,
        chat_format=SIMPLE_CHAT_FORMAT,
        use_mlock=True,
        use_mmap=True,
        verbose=False,
        temperature=float(LLM_TEMPERATURE_SIMPLE),
        top_p=float(LLM_TOP_P),
        top_k=int(LLM_TOP_K),
        repeat_penalty=float(LLM_REPEAT_PENALTY)
    )
    load_time = time.time() - start_time
    print(f"[LLM] ✅ Simple model loaded: {SIMPLE_MODEL_PATH} (took {load_time:.1f}s)")
    
    print("[Aura-LLM] 🚀 Starting Aura LLM Container (Direct Routing Architecture)")
    print("[Aura-LLM] 📋 Configuration:")
    
    # Advanced Medical Navigator is the default and only option
    _ensure_medical_navigator_import()
    print("  - MODE: Advanced Medical Navigator (Hybrid LLM/RAG/FAISS)")
    print("    • Natural conversation flow with guideline-based assessment")
    print("    • Dynamic condition ranking and smart question selection")
    print("    • On-demand guideline loading for low latency")
    print("    • FAISS semantic matching and anatomical filtering")
    print("    • Multi-category support with fuzzy fallback")
    print("    • Uses local CPU FAISS for medical knowledge")
    print("    • Single LLM model (Qwen2.5-1.5B) for all tasks")

    app.run(host='0.0.0.0', port=11434, debug=False)


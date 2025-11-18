# === container_rest.py — Aura Generic Conversational Container ===
# Provides general conversation with RAG-powered knowledge

from flask import Flask, request, jsonify, stream_with_context, Response
from llama_cpp import Llama
import os, threading, atexit
import requests
from typing import List, Optional
from collections import Counter
import re

# Conversation management for passive listening and keyword activation
from conversation_manager import ConversationMemoryIndex, ConversationOrchestrator

# Import modular RAG client (supports both GPU and CPU modes)
from rag import get_rag_client

app = Flask(__name__)

# === Thread Safety ===
llm_lock = threading.Lock()

# === Model/LLM Config (hardcoded for easy tuning) ===
LLM_TEMPERATURE_SIMPLE = 0.7
LLM_TOP_P = 0.95
LLM_TOP_K = 40
LLM_REPEAT_PENALTY = 1.1
LLM_NUM_PREDICT_DEFAULT = 150  # Reduced from 300 for faster responses (shorter = faster)
SIMPLE_N_CTX = 1024
SIMPLE_CHAT_FORMAT = "qwen"
N_THREADS = 8
N_BATCH = 256  # Reduced from 512 for faster generation (smaller batches = lower latency)
CACHE_PROMPT = True

# RAG Mode toggle: "CPU", "GPU", or "OFF" (resolved from app_settings.json if present)
def _resolve_rag_mode():
    try:
        import json
        settings_path = "/app/data/app_settings.json"
        if os.path.isfile(settings_path):
            with open(settings_path, "r") as f:
                data = json.load(f)
                mode = (data.get("rag_mode") or "").strip().upper()
                if mode in ("CPU", "GPU", "OFF"):
                    print(f"[Generic] 🎛️ RAG_MODE from settings: {mode}")
                    return mode
                elif mode:
                    print(f"[Generic] ⚠️ Invalid rag_mode '{mode}' in settings; using default")
    except Exception as e:
        print(f"[Generic] ⚠️ Failed reading rag_mode from settings: {e}")
    default_mode = "CPU"
    print(f"[Generic] 🛟 Using default RAG_MODE: {default_mode}")
    return default_mode

RAG_MODE = _resolve_rag_mode()

# Conversation/activation config (no env)
ACTIVATION_KEYWORDS = ["hey aura"]
ACTIVATION_WINDOW_SECONDS = 15.0
ACTIVATION_COOLDOWN_SECONDS = 3.0
CONVERSATION_MEMORY_DIR = "data/learning/conversation_memory"
CONVERSATION_MEMORY_PERSIST_EVERY = 10
CONVERSATION_MEMORY_MAX_ENTRIES = 5000
CONVERSATION_MEMORY_TOP_K = 3
CONVERSATION_MEMORY_MIN_SCORE = 0.35

# === Model Path Resolution (app_settings.json or fallback) ===
def _resolve_model_path():
    """
    Determine model path priority:
    1) app_settings.json llm_model (filename) -> /models/<filename> if exists
    2) SIMPLE_MODEL_PATH from env
    3) Default fallback
    """
    # 1) App settings override (mounted at /app/data/app_settings.json)
    try:
        import json
        settings_path = "/app/data/app_settings.json"
        if os.path.isfile(settings_path):
            with open(settings_path, "r") as f:
                data = json.load(f)
                name = (data.get("llm_model") or "").strip()
                if name:
                    candidate = f"/models/{name}" if not name.startswith("/") else name
                    if os.path.isfile(candidate):
                        print(f"[Generic] 🎯 Using model from settings: {candidate}")
                        return candidate
                    else:
                        print(f"[Generic] ⚠️ Model from settings not found: {candidate}")
    except Exception as e:
        print(f"[Generic] ⚠️ Failed reading app settings: {e}")
    # 3) Fallback
    fallback = "/models/Qwen2.5-1.5B-Instruct.Q4_K_M.gguf"
    print(f"[Generic] 🛟 Using default model: {fallback}")
    return fallback

SIMPLE_MODEL_PATH = _resolve_model_path()

llm_simple = None

# === Conversation Memory / Activation Config ===
# === Health Check Endpoint ===
@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint to verify models are loaded"""
    try:
        simple_loaded = llm_simple is not None
        
        return jsonify({
            "status": "ok",
            "service": "aura-llm-generic",
            "models": {
                "simple_loaded": simple_loaded,
                "simple_path": SIMPLE_MODEL_PATH
            }
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "service": "aura-llm-generic",
            "error": str(e)
        }), 500

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
            print(f"[LLM] ❌ Error in llm_chat_simple: {e}")
            if stream:
                return iter([])
            return ""

# === Conversational Logic ===
def handle_conversation(
    prompt: str, session_id: str, memory_context: Optional[str] = None, stream: bool = False
):
    """
    Handle general conversation with optional RAG
    
    Args:
        prompt: User's message
        session_id: Session identifier
        memory_context: Optional conversation memory context
        stream: If True, returns a generator that yields tokens. If False, returns complete response string.
    
    Returns:
        If stream=False: Complete response string
        If stream=True: Generator that yields tokens as they're generated
    """
    
    # Try RAG first for knowledge queries (CPU or GPU) if enabled
    # Skip RAG for simple conversational queries to reduce latency
    rag_context = ""
    if RAG_MODE in ("CPU", "GPU"):
        print(f"[Generic] 🔍 RAG_MODE={RAG_MODE} - checking if query should use RAG...")
        
        # Only use RAG for queries that seem like knowledge/document questions
        # Simple conversational queries don't need RAG (faster response)
        # Skip RAG for personal/conversational queries (day, schedule, how are you, etc.)
        personal_keywords = ['my day', 'my schedule', 'my calendar', 'how are you', 'how am i', 
                          'what am i', 'when am i', 'where am i', 'tell me about me']
        is_personal_query = any(keyword in prompt.lower() for keyword in personal_keywords)
        
        # Use RAG for knowledge/document queries, but skip for personal/conversational
        knowledge_keywords = ['what is', 'what are', 'how does', 'explain', 'tell me about',
                            'document', 'file', 'information about', 'details about']
        is_knowledge_query = (any(keyword in prompt.lower() for keyword in knowledge_keywords) 
                            and not is_personal_query)
        
        print(f"[Generic] 🔍 Query analysis: is_personal={is_personal_query}, is_knowledge={is_knowledge_query}")
        
        if is_knowledge_query:
            try:
                print(f"[Generic] 🔍 RAG search triggered for knowledge query: '{prompt[:50]}...'")
                rag_client = get_rag_client()
                rag_mode = "GPU" if rag_client.use_gpu else "CPU"
                print(f"[Generic] 🔍 RAG mode: {rag_mode}")
                
                # Check if RAG client has any embeddings
                if hasattr(rag_client, '_cpu_chunks') and rag_client._cpu_chunks:
                    print(f"[Generic] 📊 RAG index: {len(rag_client._cpu_chunks)} chunks available")
                elif hasattr(rag_client, '_cpu_index') and rag_client._cpu_index:
                    index_size = rag_client._cpu_index.ntotal if hasattr(rag_client._cpu_index, 'ntotal') else 0
                    print(f"[Generic] 📊 RAG index: {index_size} vectors available")
                else:
                    print(f"[Generic] ⚠️ RAG index appears empty - no embeddings loaded")
                
                results = rag_client.search(query=prompt, k=3)
                
                if results and len(results) > 0:
                    print(f"[Generic] ✅ RAG found {len(results)} results")
                    for i, result in enumerate(results[:3], 1):
                        score = result.get('score', 0)
                        text_preview = result.get('text', '')[:50]
                        # Extract file name from metadata
                        file_name = "unknown"
                        if isinstance(result.get('metadata'), dict):
                            file_path = result['metadata'].get('file_path', '')
                            if file_path:
                                from pathlib import Path
                                file_name = Path(file_path).name
                            else:
                                file_name = result['metadata'].get('document_name', 'unknown')
                        print(f"[Generic]   [{i}] Score: {score:.3f}, File: {file_name}, Preview: '{text_preview}...'")
                    rag_context = "\n".join(
                        [r.get("text", "") for r in results[:3] if r.get("text")]
                    )
                    print(f"[Generic] ✅ Using RAG context ({len(rag_context)} chars) for LLM response")
                else:
                    print(f"[Generic] ⚠️ RAG search returned no results (index may be empty or query doesn't match)")
            except Exception as e:
                print(f"[Generic] ⚠️ RAG failed, using direct LLM: {e}")
                import traceback
                traceback.print_exc()
        else:
            print(f"[Generic] ⏭️ Skipping RAG for conversational query (faster response)")
    else:
        print(f"[Generic] ⏭️ RAG_MODE={RAG_MODE} - RAG disabled")
    
    contextual_sections: List[str] = []
    if rag_context:
        contextual_sections.append(f"Knowledge context:\n{rag_context}")
        print(f"[Generic] 📝 LLM prompt includes RAG context")
    if memory_context:
        contextual_sections.append(f"Conversation memory:\n{memory_context}")
        print(f"[Generic] 📝 LLM prompt includes conversation memory")
    combined_context = "\n\n".join(contextual_sections).strip()
    
    if not combined_context:
        print(f"[Generic] 📝 LLM prompt: direct conversation (no RAG, no memory)")

    if combined_context:
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant. Use the provided knowledge context "
                    "and conversation memory to answer the user's question.\n\n"
                    f"{combined_context}\n\n"
                    f"User question: {prompt}\n\n"
                    "If the provided context does not contain the answer, acknowledge "
                    "that and answer based on your general knowledge."
                ),
            }
        ]
        return llm_chat_simple(messages, max_tokens=150, stream=stream)

    # Fallback to direct LLM conversation without external context
    system_prompt = (
        "You are a helpful, friendly assistant. Keep responses concise and conversational."
    )
    if memory_context:
        system_prompt += f"\n\nConversation memory you can reference:\n{memory_context}"
    messages = [
        {
            "role": "system",
            "content": system_prompt,
        },
        {
            "role": "user",
            "content": prompt,
        },
    ]

    return llm_chat_simple(messages, max_tokens=300, stream=stream)


def _embed_texts(texts: List[str]) -> List[List[float]]:
    if not texts:
        return []
    try:
        rag_client = get_rag_client()
        return rag_client.embed(texts)
    except Exception as exc:
        print(f"[Memory] ⚠️ Failed to generate embeddings: {exc}")
        return []


conversation_memory = ConversationMemoryIndex(
    storage_dir=CONVERSATION_MEMORY_DIR,
    persist_every=CONVERSATION_MEMORY_PERSIST_EVERY,
    max_entries=CONVERSATION_MEMORY_MAX_ENTRIES,
)

conversation_orchestrator = ConversationOrchestrator(
    memory_index=conversation_memory,
    embed_fn=_embed_texts,
    conversation_handler=handle_conversation,
    activation_keywords=ACTIVATION_KEYWORDS,
    activation_window=ACTIVATION_WINDOW_SECONDS,
    activation_cooldown=ACTIVATION_COOLDOWN_SECONDS,
    memory_top_k=CONVERSATION_MEMORY_TOP_K,
    memory_min_score=CONVERSATION_MEMORY_MIN_SCORE,
)

atexit.register(conversation_orchestrator.flush_memory)

# === Chat Endpoints ===
@app.route("/chat-tg", methods=["POST"])
def chat_tg():
    """Non-streaming chat endpoint for Telegram"""
    data = request.get_json()
    prompt = (data.get("prompt") or "").strip()
    session_id = (data.get("chat_id") or data.get("session_id") or "default").strip()
    
    if not prompt:
        return jsonify({"response": "Please provide a message."})
    
    print(f"[Generic] 💬 Session: {session_id}, Prompt: '{prompt[:50]}...'")
    
    try:
        response = handle_conversation(prompt, session_id)
        return jsonify({"response": response})
    except Exception as e:
        print(f"[Generic] ❌ Error: {e}")
        return jsonify({"response": "I apologize, I encountered an error processing your request."})

@app.route("/chat-tts", methods=["POST"])
def chat_tts():
    """Streaming chat endpoint for TTS/Voice - streams tokens as they're generated"""
    data = request.get_json()
    prompt = (data.get("prompt") or "").strip()
    session_id = (data.get("chat_id") or data.get("session_id") or None)
    
    if not prompt:
        return jsonify({"error": "Missing prompt"}), 400
    
    print(f"[Generic] 💬 Streaming Session: {session_id}, Prompt: '{prompt[:50]}...'")
    
    def generate_response():
        try:
            # Use streaming mode to get tokens as they're generated
            result = handle_conversation(prompt, session_id or "default", stream=True)
            
            # Check if result is a generator (streaming)
            if hasattr(result, '__iter__') and not isinstance(result, str):
                print(f"[Generic] ✅ Streaming enabled - tokens will be yielded as generated")
                normalized_chunks = _normalize_stream_chunks(result)
                word_stream = _word_stream_from_chunks(normalized_chunks)
                sentence_stream = _sentence_tag_stream(word_stream)
                for token in sentence_stream:
                    yield f"{token}\n"
                print(f"[Generic] ✅ Streamed response complete")
            else:
                # Fallback: non-streaming (result is a string)
                print(f"[Generic] ⚠️ Streaming not available - yielding complete response")
                fallback_text = result if isinstance(result, str) and result else "I apologize, I encountered an error."
                normalized_chunks = _normalize_stream_chunks(iter([fallback_text]))
                word_stream = _word_stream_from_chunks(normalized_chunks)
                sentence_stream = _sentence_tag_stream(word_stream)
                for token in sentence_stream:
                    yield f"{token}\n"
        except Exception as e:
            print(f"[Generic] ❌ Error: {e}")
            import traceback
            traceback.print_exc()
            fallback_text = "I apologize, I encountered an error."
            normalized_chunks = _normalize_stream_chunks(iter([fallback_text]))
            word_stream = _word_stream_from_chunks(normalized_chunks)
            sentence_stream = _sentence_tag_stream(word_stream)
            for token in sentence_stream:
                yield f"{token}\n"
    
    return Response(
        stream_with_context(filter_think_blocks(generate_response())),
        mimetype="text/plain",
    )


# === Streaming Helpers =======================================================

WORD_BOUNDARY_CHARS = [' ', '.', ',', '!', '?', ':', ';', '-', '(', ')', '[', ']']
SENTENCE_ENDINGS = ('.', '!', '?')


def _normalize_stream_chunks(chunk_iter):
    """
    Normalize mixed-type streaming chunks (dicts, strings) to plain strings.
    """
    for chunk in chunk_iter:
        if isinstance(chunk, dict):
            if 'choices' in chunk and len(chunk['choices']) > 0:
                delta = chunk['choices'][0].get('delta', {})
                content = delta.get('content', '')
                if content:
                    yield content
            elif 'content' in chunk:
                content = chunk.get('content', '')
                if content:
                    yield content
        elif isinstance(chunk, str):
            if chunk:
                yield chunk
        else:
            yield str(chunk)


def _find_word_boundary(buffer: str):
    """
    Return the index of the first word boundary character in buffer, or None.
    """
    for idx, char in enumerate(buffer):
        if char in WORD_BOUNDARY_CHARS:
            return idx
    return None


def _word_stream_from_chunks(chunk_iter):
    """
    Buffer raw LLM chunks until we reach a word boundary, then yield the word.
    Ensures downstream consumers receive complete words (no sub-word splits).
    """
    buffer = ""
    for chunk in chunk_iter:
        if not chunk:
            continue
        buffer += chunk
        while True:
            boundary_idx = _find_word_boundary(buffer)
            if boundary_idx is None:
                break
            word = buffer[:boundary_idx + 1]
            buffer = buffer[boundary_idx + 1:]
            if word:
                yield word
    if buffer:
        yield buffer


def _sentence_tag_stream(word_stream):
    """
    Wrap word stream with <sentence_start>/<sentence_end> markers, splitting on sentence boundaries.
    Each complete sentence/phrase gets its own tags for natural TTS playback.
    """
    sentence_buffer = ""
    sentence_open = False
    
    for word in word_stream:
        word_stripped = word.strip()
        
        # Special handling for standalone dashes: they start new sentences for list items
        if word_stripped == '-':
            # Close previous sentence if open
            if sentence_open:
                yield "<sentence_end>"
                sentence_buffer = ""
            # Start new sentence for list item (dash is first word)
            sentence_open = True
            yield "<sentence_start>"
            yield word
            sentence_buffer = word
            continue
        
        # Normal word processing
        if not sentence_open:
            sentence_open = True
            yield "<sentence_start>"
        
        yield word
        sentence_buffer += word
        
        # Check if we've reached a sentence boundary
        # 1. Sentence endings: . ! ? (period, exclamation, question mark)
        if word_stripped and word_stripped[-1] in SENTENCE_ENDINGS:
            yield "<sentence_end>"
            sentence_buffer = ""
            sentence_open = False
        # 2. Colons: split for list items (e.g., "include:" starts a list)
        elif word_stripped and word_stripped[-1] == ':':
            yield "<sentence_end>"
            sentence_buffer = ""
            sentence_open = False
    
    # Close any remaining sentence
    if sentence_open:
        yield "<sentence_end>"


def filter_think_blocks(generator):
    """
    Filter streaming output to remove <think> blocks and detect garbage output.
    Mirrors the medical container behavior for parity.
    """
    accumulated_output = []
    garbage_detected = False
    
    for token in generator:
        if token and token.strip():
            accumulated_output.append(token)
            
            full_output = ''.join(accumulated_output)
            text_only = re.sub(r'<sentence_start>|<sentence_end>|\n', '', full_output)
            
            if len(text_only) > 50 and len(text_only) % 100 < 20:
                char_counts = Counter(text_only.lower())
                if char_counts:
                    most_common_char, most_common_count = char_counts.most_common(1)[0]
                    repetition_ratio = most_common_count / len(text_only)
                    
                    if repetition_ratio > 0.6:
                        print(f"[Generic] ⚠️ GARBAGE DETECTED: char='{most_common_char}', ratio={repetition_ratio:.2f}, output='{text_only[:100]}'")
                        garbage_detected = True
                        break
        
        yield token
    
    if garbage_detected:
        print(f"[Generic] 🔄 Using fallback response due to garbage detection")
        yield "<sentence_start>\nI'm sorry, I had trouble processing that. Could you tell me more about what's going on?\n<sentence_end>\n"


@app.route("/voice/transcript", methods=["POST"])
def voice_transcript():
    """
    Passive transcript ingestion endpoint.
    
    Accepts continuous text from the SST pipeline, indexes it for long-term memory,
    and returns an LLM response only when an activation keyword window is open.
    """
    data = request.get_json(silent=True) or {}
    text = (data.get("text") or "").strip()
    session_id = str(
        data.get("session_id") or data.get("chat_id") or data.get("conversation_id") or "default"
    )
    is_final = bool(data.get("is_final", True))
    timestamp = data.get("timestamp")
    metadata = data.get("metadata") or {}

    if not text:
        return jsonify({"error": "Missing text"}), 400

    result = conversation_orchestrator.process_chunk(
        session_id=session_id,
        text=text,
        is_final=is_final,
        timestamp=timestamp,
        metadata=metadata,
    )

    return jsonify(
        {
            "status": "ok",
            "session_id": session_id,
            **result,
        }
    )

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
        print(f"[Generic] ❌ Error in CPU FAISS auto-ingestion: {e}")
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
            'processed_files': len(auto_ingest.state.get('processed_files', {}))
        })
        
    except Exception as e:
        print(f"[Generic] ❌ Error getting CPU FAISS status: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    print("[Generic] 🚀 Starting Aura Generic LLM Container...")
    
    # Load model with GPU acceleration
    print(f"[Generic] 📦 Loading model: {SIMPLE_MODEL_PATH}")
    # Offload all layers to GPU for maximum acceleration (set to 0 to disable GPU)
    # For Jetson, offloading all layers typically provides best performance
    n_gpu_layers = -1  # -1 = offload all layers to GPU, 0 = CPU only
    print(f"[Generic] 🚀 GPU acceleration: {n_gpu_layers} layers offloaded to GPU")
    llm_simple = Llama(
        model_path=SIMPLE_MODEL_PATH,
        n_ctx=SIMPLE_N_CTX,
        n_threads=N_THREADS,
        n_batch=N_BATCH,
        n_gpu_layers=n_gpu_layers,  # Enable GPU acceleration
        cache_prompt=CACHE_PROMPT,
        chat_format=SIMPLE_CHAT_FORMAT,
        use_mlock=True,
        use_mmap=True,
        verbose=False
    )
    print(f"[Generic] ✅ Model loaded: {SIMPLE_MODEL_PATH}")
    
    print("[Generic] ✅ LLM Container ready!")
    print("[Generic] 🌐 Starting Flask server on 0.0.0.0:11434...")
    
    app.run(host="0.0.0.0", port=11434, threaded=True, debug=False)

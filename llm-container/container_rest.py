# === container_rest.py — Aura Generic Conversational Container ===
# Provides general conversation with RAG-powered knowledge

from flask import Flask, request, jsonify, stream_with_context, Response
from llama_cpp import Llama
import os, threading, atexit
import requests
from typing import List, Optional
from collections import Counter
import re
import logging
import json

# Conversation management for passive listening and keyword activation
from conversation_manager import ConversationMemoryIndex, ConversationOrchestrator

# Import modular RAG client (supports both GPU and CPU modes)
from rag import get_rag_client

app = Flask(__name__)

# Suppress verbose logging for status/health endpoints
werkzeug_logger = logging.getLogger('werkzeug')
werkzeug_logger.setLevel(logging.WARNING)  # Only log warnings and errors, not info requests

# === Thread Safety ===
llm_lock = threading.Lock()

# === Model/LLM Config (hardcoded for easy tuning) ===
LLM_TEMPERATURE_SIMPLE = 0.7
LLM_TOP_P = 0.95
LLM_TOP_K = 40
LLM_REPEAT_PENALTY = 1.1
LLM_NUM_PREDICT_DEFAULT = 800  # Increased to allow comprehensive responses (can be overridden via LLM_NUM_PREDICT env var)
SIMPLE_N_CTX = 4096  # Reduced from 8192 to decrease latency (sufficient for RAG context + responses up to 1500 tokens)
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

# === Streaming/Text Processing Config ===
WORD_BOUNDARY_CHARS = [' ', '.', ',', '!', '?', ':', ';', '-', '(', ')', '[', ']']
SENTENCE_ENDINGS = ('.', '!', '?')

# === Response Generation Config ===
MAX_TOKENS_RAG_MODE = 1500  # Max tokens when using RAG context (increased for comprehensive responses)
MAX_TOKENS_DIRECT_MODE = 1200  # Max tokens for direct conversation (increased for comprehensive responses)

# === Model Path Resolution (app_settings.json or fallback) ===
def _resolve_model_path():
    """
    Determine model path priority:
    1) app_settings.json llm_model (filename) -> /models/<filename> if exists
    2) SIMPLE_MODEL_PATH from env (set by Dockerfile)
    3) Default fallback (matches Dockerfile)
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
    
    # 2) Use environment variable (set by Dockerfile) as fallback
    env_path = os.getenv("SIMPLE_MODEL_PATH", "")
    if env_path and os.path.isfile(env_path):
        print(f"[Generic] 🛟 Using model from environment: {env_path}")
        return env_path
    
    # 3) Final fallback (matches Dockerfile default)
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
        
        # Normalize contractions to handle variations like "what's" -> "what is"
        contractions_map = {
            "what's": "what is",
            "what're": "what are",
            "who's": "who is",
            "where's": "where is",
            "when's": "when is",
            "why's": "why is",
            "how's": "how is",
            "how're": "how are",
            "how'd": "how did",
            "how'll": "how will",
            "that's": "that is",
            "there's": "there is",
            "here's": "here is",
            "it's": "it is",
            "i'm": "i am",
            "you're": "you are",
            "we're": "we are",
            "they're": "they are",
            "he's": "he is",
            "she's": "she is",
        }
        normalized_prompt = prompt.lower()
        for contraction, expansion in contractions_map.items():
            normalized_prompt = normalized_prompt.replace(contraction, expansion)
        
        # Skip RAG for personal/conversational queries (day, schedule, how are you, etc.)
        personal_keywords = ['my day', 'my schedule', 'my calendar', 'how are you', 'how am i', 
                          'what am i', 'when am i', 'where am i', 'tell me about me']
        is_personal_query = any(keyword in normalized_prompt for keyword in personal_keywords)
        
        # Only use RAG if search actually returns results (require actual relevance, not just substring match)
        # This ensures RAG is only used when there's actually relevant content to inject
        rag_client = None
        rag_context = ""
        rag_results = []
        
        if not is_personal_query:
            try:
                rag_client = get_rag_client()
                if rag_client:
                    # Quick check: does RAG have content at all?
                    has_content = rag_client.quick_content_match(prompt)
                    if has_content:
                        print(f"[Generic] 🔍 Query may match RAG content - performing search...")
                        
                        # Check if RAG client has any embeddings
                        if hasattr(rag_client, '_cpu_chunks') and rag_client._cpu_chunks:
                            print(f"[Generic] 📊 RAG index: {len(rag_client._cpu_chunks)} chunks available")
                        elif hasattr(rag_client, '_cpu_index') and rag_client._cpu_index:
                            index_size = rag_client._cpu_index.ntotal if hasattr(rag_client._cpu_index, 'ntotal') else 0
                            print(f"[Generic] 📊 RAG index: {index_size} vectors available")
                        else:
                            print(f"[Generic] ⚠️ RAG index appears empty - no embeddings loaded")
                        
                        # Search for relevant results (defaults: k=3, threshold=0.35 for balanced relevance)
                        rag_results = rag_client.search(query=prompt)
                        
                        # Only use RAG if search actually returns results above threshold
                        if rag_results and len(rag_results) > 0:
                            print(f"[Generic] ✅ RAG found {len(rag_results)} relevant results (threshold=0.35) - will inject context")
                        else:
                            print(f"[Generic] 🔍 RAG search returned no results above threshold - skipping RAG injection")
                    else:
                        print(f"[Generic] 🔍 Query doesn't match RAG content - skipping RAG (faster response)")
                else:
                    print(f"[Generic] ⚠️ RAG client not available")
            except Exception as e:
                print(f"[Generic] ⚠️ RAG check failed: {e}")
                rag_client = None
        
        should_use_rag = (rag_results and len(rag_results) > 0)
        print(f"[Generic] 🔍 Query analysis: is_personal={is_personal_query}, should_use_rag={should_use_rag}")
        
        if should_use_rag and rag_results:
            try:
                print(f"[Generic] 🔍 RAG injection triggered: '{prompt[:50]}...'")
                rag_mode = "GPU" if rag_client.use_gpu else "CPU"
                print(f"[Generic] 🔍 RAG mode: {rag_mode}")
                
                for i, result in enumerate(rag_results, 1):
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
                
                # Build RAG context, limit each result to 800 chars
                MAX_CHARS_PER_RESULT = 800
                rag_chunks = []
                for r in rag_results:
                    text = r.get("text", "")
                    if text:
                        # Truncate if too long, but try to break at word boundary
                        if len(text) > MAX_CHARS_PER_RESULT:
                            truncated = text[:MAX_CHARS_PER_RESULT]
                            # Try to break at last space to avoid cutting words
                            last_space = truncated.rfind(' ')
                            if last_space > MAX_CHARS_PER_RESULT * 0.8:  # Only if we're not losing too much
                                truncated = truncated[:last_space] + "..."
                            else:
                                truncated = truncated + "..."
                            rag_chunks.append(truncated)
                        else:
                            rag_chunks.append(text)
                rag_context = "\n".join(rag_chunks)
                print(f"[Generic] ✅ Using RAG context ({len(rag_context)} chars, ~{len(rag_context)//4} tokens) for LLM response")
            except Exception as e:
                print(f"[Generic] ⚠️ RAG failed, using direct LLM: {e}")
                import traceback
                traceback.print_exc()
                rag_context = ""  # Clear context on error
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
                    "You are a helpful, knowledgeable assistant. Provide comprehensive, well-structured responses that thoroughly address the user's question.\n\n"
                    f"{combined_context}\n\n"
                    "RESPONSE GUIDELINES:\n"
                    "- Provide thorough, comprehensive answers that cover the topic in depth\n"
                    "- Structure your response with clear sections and subsections when appropriate\n"
                    "- Explain the 'why' behind recommendations, not just the 'what'\n"
                    "- Cover different scenarios, types, or variations when relevant\n"
                    "- Include important context, disclaimers, or safety information when appropriate\n"
                    "- Use formatting like **bold text** for emphasis and clear section breaks\n"
                    "- Reference the knowledge context above if it contains relevant information that improves your answer\n"
                    "- Do NOT force irrelevant information from the context into your response\n"
                    "- If the context is not relevant to the question, ignore it and answer using your general knowledge\n"
                    "- Be conversational and natural, but prioritize completeness and helpfulness\n\n"
                    f"User question: {prompt}"
                ),
            }
        ]
        return llm_chat_simple(messages, max_tokens=MAX_TOKENS_RAG_MODE, stream=stream)

    # Fallback to direct LLM conversation without external context
    system_prompt = (
        "You are a helpful, knowledgeable assistant. Provide comprehensive, well-structured responses that thoroughly address the user's question.\n\n"
        "RESPONSE GUIDELINES:\n"
        "- Provide thorough, comprehensive answers that cover the topic in depth\n"
        "- Structure your response with clear sections and subsections when appropriate\n"
        "- Explain the 'why' behind recommendations, not just the 'what'\n"
        "- Cover different scenarios, types, or variations when relevant\n"
        "- Include important context, disclaimers, or safety information when appropriate\n"
        "- Use formatting like **bold text** for emphasis and clear section breaks\n"
        "- Be conversational and natural, but prioritize completeness and helpfulness"
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

    # Use standard max_tokens - matches LLM_NUM_PREDICT_DEFAULT
    return llm_chat_simple(messages, max_tokens=MAX_TOKENS_DIRECT_MODE, stream=stream)


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
    """Streaming chat endpoint - streams tokens as they're generated for faster response"""
    data = request.get_json()
    prompt = (data.get("prompt") or "").strip()
    session_id = (data.get("chat_id") or data.get("session_id") or "default").strip()
    
    # Auto-detect streaming capability:
    # 1. Explicit stream parameter takes precedence
    # 2. Check Accept header for SSE support
    # 3. Default to True for better UX (clients that can't handle it should explicitly set stream=false)
    explicit_stream = data.get("stream")
    if explicit_stream is not None:
        stream = explicit_stream
    else:
        # Check if client supports SSE (text/event-stream)
        accept_header = request.headers.get("Accept", "")
        supports_sse = "text/event-stream" in accept_header or "text/event-stream" in accept_header.lower()
        # Default to streaming for better UX - clients that can't handle it should set stream=false
        stream = True  # Default to streaming for better perceived performance
        if not supports_sse:
            # If no explicit preference and client doesn't advertise SSE support, 
            # we could default to False, but let's be optimistic and default to True
            # Clients that break can explicitly set stream=false
            pass
    
    if not prompt:
        return jsonify({"response": "Please provide a message."})
    
    print(f"[Generic] 💬 Session: {session_id}, Prompt: '{prompt[:50]}...', Stream: {stream}")
    
    if stream:
        # Streaming mode: return Server-Sent Events (SSE) format
        def generate_streaming_response():
            try:
                # Use streaming mode to get tokens as they're generated
                result = handle_conversation(prompt, session_id, stream=True)
                
                # Check if result is a generator (streaming)
                if hasattr(result, '__iter__') and not isinstance(result, str):
                    print(f"[Generic] ✅ Streaming enabled - tokens will be yielded as generated")
                    normalized_chunks = _normalize_stream_chunks(result)
                    word_stream = _word_stream_from_chunks(normalized_chunks)
                    
                    # Stream words as JSON chunks for incremental display
                    accumulated = ""
                    for word in word_stream:
                        accumulated += word
                        # Send incremental JSON updates
                        yield f"data: {json.dumps({'response': accumulated, 'done': False})}\n\n"
                    
                    # Send final message
                    yield f"data: {json.dumps({'response': accumulated, 'done': True})}\n\n"
                    print(f"[Generic] ✅ Streamed response complete")
                else:
                    # Fallback: non-streaming (result is a string)
                    print(f"[Generic] ⚠️ Streaming not available - sending complete response")
                    fallback_text = result if isinstance(result, str) and result else "I apologize, I encountered an error."
                    yield f"data: {json.dumps({'response': fallback_text, 'done': True})}\n\n"
            except Exception as e:
                print(f"[Generic] ❌ Error: {e}")
                import traceback
                traceback.print_exc()
                error_msg = "I apologize, I encountered an error processing your request."
                yield f"data: {json.dumps({'response': error_msg, 'done': True})}\n\n"
        
        return Response(
            stream_with_context(generate_streaming_response()),
            mimetype="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no"  # Disable nginx buffering
            }
        )
    else:
        # Non-streaming mode: return complete response (backward compatibility)
        try:
            response = handle_conversation(prompt, session_id, stream=False)
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
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # Disable nginx/proxy buffering
            "Connection": "keep-alive"  # Keep connection open for streaming
        }
    )


# === Streaming Helpers =======================================================


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
    Filters out whitespace-only tokens.
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
            # Only yield non-empty words (filter out whitespace-only tokens)
            if word and word.strip():
                yield word
    # Only yield remaining buffer if it's not just whitespace
    if buffer and buffer.strip():
        yield buffer


def _sentence_tag_stream(word_stream):
    """
    Wrap word stream with <sentence_start>/<sentence_end> markers, splitting on sentence boundaries.
    Each complete sentence/phrase gets its own tags for natural TTS playback.
    Expands abbreviations like "e.g." → "for example", "i.e." → "that is", "etc." → "etcetera".
    
    IMPORTANT: This processes words incrementally with minimal buffering (1 token lookahead),
    allowing tokens to be sent to TTS as they're generated.
    """
    sentence_buffer = ""
    sentence_open = False
    prev_word = None
    buffered_word = None  # One-token lookahead buffer for multi-token abbreviations
    
    # Abbreviation expansions (abbrev -> full text)
    abbrev_expansions = {
        'e.g.': 'for example',
        'i.e.': 'that is',
        'etc.': 'etcetera',
        'vs.': 'versus',
        'dr.': 'doctor',
        'mr.': 'mister',
        'mrs.': 'missus',
        'ms.': 'miss',
        'prof.': 'professor',
        'sr.': 'senior',
        'jr.': 'junior',
    }
    
    # Multi-token abbreviation patterns (first part -> (second part, expansion))
    multi_token_abbrevs = {
        'e.': ('g.', 'for example'),  # e.g.
        'i.': ('e.', 'that is'),  # i.e.
    }
    
    def yield_word(word_to_yield):
        """Helper to yield a word, expanding abbreviations if needed"""
        nonlocal sentence_buffer, sentence_open
        
        word_stripped = word_to_yield.strip()
        
        # Special handling for standalone dashes: they start new sentences for list items
        if word_stripped == '-':
            # Close previous sentence if open
            if sentence_open:
                yield "<sentence_end>"
                sentence_buffer = ""
            # Start new sentence for list item (dash is first word)
            sentence_open = True
            yield "<sentence_start>"
            yield word_to_yield
            sentence_buffer = word_to_yield
            return
        
        # Normal word processing
        if not sentence_open:
            sentence_open = True
            yield "<sentence_start>"
        
        # Check if this is a single-token abbreviation that should be expanded
        word_lower = word_stripped.lower().rstrip(',').rstrip(')').rstrip(']').rstrip('}')
        if word_lower in abbrev_expansions:
            # Replace with expansion, preserving trailing punctuation
            trailing_punct = ""
            for char in reversed(word_stripped):
                if not char.isalnum() and char != '.':
                    trailing_punct = char + trailing_punct
                else:
                    break
            expansion_text = abbrev_expansions[word_lower] + trailing_punct
            yield expansion_text
            sentence_buffer += expansion_text
        else:
            # Normal word - yield as-is
            yield word_to_yield
            sentence_buffer += word_to_yield
        
        # Check if we've reached a sentence boundary
        # 1. Sentence endings: . ! ? (period, exclamation, question mark)
        if word_stripped and word_stripped[-1] in SENTENCE_ENDINGS:
            # Check if this might be part of an abbreviation
            is_abbreviation = False
            
            # Check if word is a known single-token abbreviation
            word_lower = word_stripped.lower().rstrip(',').rstrip(')').rstrip(']').rstrip('}')
            if word_lower in abbrev_expansions:
                is_abbreviation = True
            # Check if it's a single letter followed by period (like "e." or "i.")
            # Remove leading punctuation for detection
            word_clean = word_stripped.lstrip('(').lstrip('[').lstrip('{').lower()
            if len(word_clean) == 2 and word_clean[0].isalpha() and word_clean[-1] == '.':
                # Check if this could be the first part of a multi-token abbreviation
                if word_clean in multi_token_abbrevs:
                    is_abbreviation = True  # Don't end sentence yet, wait for next token
                # Also check if previous word was also short, likely abbreviation
                elif prev_word and len(prev_word.strip()) <= 3:
                    is_abbreviation = True
            
            # Only end sentence if it's not an abbreviation
            if not is_abbreviation:
                yield "<sentence_end>"
                sentence_buffer = ""
                sentence_open = False
        # 2. Colons: split for list items (e.g., "include:" starts a list)
        elif word_stripped and word_stripped[-1] == ':':
            yield "<sentence_end>"
            sentence_buffer = ""
            sentence_open = False
    
    # Process the word stream
    for word in word_stream:
        # Skip whitespace-only tokens (shouldn't happen after _word_stream_from_chunks fix, but double-check)
        if not word or not word.strip():
            continue
        
        word_stripped = word.strip()
        
        # If we have a buffered word, check if current word completes a multi-token abbreviation
        if buffered_word:
            buffered_stripped = buffered_word.strip()
            buffered_clean = buffered_stripped.lstrip('(').lstrip('[').lstrip('{').lower()
            
            # Check if buffered word could be first part of multi-token abbreviation
            if buffered_clean in multi_token_abbrevs:
                expected_part, expansion = multi_token_abbrevs[buffered_clean]
                word_clean = word_stripped.lstrip(',').lstrip(' ').lower()
                
                if word_clean == expected_part:
                    # Complete multi-token abbreviation detected - expand it
                    # Preserve trailing punctuation from current word
                    trailing_punct = ""
                    for char in reversed(word_stripped):
                        if not char.isalnum() and char != '.':
                            trailing_punct = char + trailing_punct
                        else:
                            break
                    
                    expansion_text = expansion + trailing_punct
                    # Yield the expansion
                    for item in yield_word(expansion_text):
                        yield item
                    buffered_word = None
                    prev_word = word
                    continue
                else:
                    # Not the expected continuation - yield buffered word normally
                    for item in yield_word(buffered_word):
                        yield item
                    buffered_word = None
            else:
                # Buffered word wasn't part of abbreviation - yield it normally
                for item in yield_word(buffered_word):
                    yield item
                buffered_word = None
        
        # Check if current word could be first part of multi-token abbreviation
        word_clean = word_stripped.lstrip('(').lstrip('[').lstrip('{').lower()
        if len(word_clean) == 2 and word_clean[0].isalpha() and word_clean[-1] == '.':
            if word_clean in multi_token_abbrevs:
                # Buffer this word to check next token
                buffered_word = word
                prev_word = word
                continue
        
        # Normal processing - not part of multi-token abbreviation
        for item in yield_word(word):
            yield item
        prev_word = word
    
    # Process any remaining buffered word
    if buffered_word:
        for item in yield_word(buffered_word):
            yield item
    
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
    """Get CPU FAISS status (called by GUI every 15 seconds)"""
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
    
    # Pre-initialize RAG client at container startup (reduces first-query latency)
    if RAG_MODE in ("CPU", "GPU"):
        print(f"[Generic] 🔍 Pre-initializing RAG client (RAG_MODE={RAG_MODE})...")
        try:
            from rag import get_rag_client
            rag_client = get_rag_client()
            print(f"[Generic] ✅ RAG client pre-initialized: {rag_client._mode}")
            if hasattr(rag_client, '_cpu_chunks') and rag_client._cpu_chunks:
                print(f"[Generic] 📊 RAG index ready: {len(rag_client._cpu_chunks)} chunks available")
            elif hasattr(rag_client, '_cpu_index') and rag_client._cpu_index:
                index_size = rag_client._cpu_index.ntotal if hasattr(rag_client._cpu_index, 'ntotal') else 0
                print(f"[Generic] 📊 RAG index ready: {index_size} vectors in index")
        except Exception as e:
            print(f"[Generic] ⚠️ RAG client pre-initialization failed: {e}")
            print("[Generic] 💡 RAG will be initialized on first use (may add latency to first query)")
    else:
        print(f"[Generic] ⏭️ RAG_MODE={RAG_MODE} - skipping RAG initialization")
    
    print("[Generic] ✅ LLM Container ready!")
    print("[Generic] 🌐 Starting Flask server on 0.0.0.0:11434...")
    
    app.run(host="0.0.0.0", port=11434, threaded=True, debug=False)

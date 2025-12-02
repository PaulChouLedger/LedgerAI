# === container_rest.py — Aura Generic Conversational Container ===
# Provides general conversation with RAG-powered knowledge

from flask import Flask, request, jsonify, stream_with_context, Response
import os, threading, atexit, time
import requests
from typing import List, Optional
from collections import Counter
import re
import logging
import json
import sys

# Add shared directory to path for base class and RAG imports
sys.path.insert(0, '/shared')

# Import shared base class
from llm_base import BaseLLMContainer

# Conversation management for passive listening and keyword activation
from conversation_manager import ConversationMemoryIndex, ConversationOrchestrator

# Import modular RAG client from shared (supports both GPU and CPU modes)
from rag import get_rag_client

app = Flask(__name__)

# Suppress verbose logging for status/health endpoints
werkzeug_logger = logging.getLogger('werkzeug')
werkzeug_logger.setLevel(logging.WARNING)  # Only log warnings and errors, not info requests

# === Initialize Base Container ===
base_container = BaseLLMContainer(
    service_name="aura-llm-generic",
    default_model_path="/models/Qwen2.5-1.5B-Instruct.Q4_K_M.gguf"
)

# Override default parameters for generic container
base_container.LLM_NUM_PREDICT_DEFAULT = 800  # Increased for comprehensive responses
base_container.SIMPLE_N_CTX = 4096  # Reduced from 8192 for lower latency
base_container.N_BATCH = 256  # Reduced for faster generation

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

# Use base class model resolution
SIMPLE_MODEL_PATH = base_container.resolve_model_path()

# Reference to LLM instance (will be set by base_container.load_model())
llm_simple = None

# === Conversation Memory / Activation Config ===
# === Health Check Endpoint ===
# Register health check using base class
base_container.register_health_check(app)

# Wrapper functions for backward compatibility
def extract_llm_response_content(response) -> str:
    """Extract text content from LLM response"""
    return base_container.extract_llm_response_content(response)

def llm_chat_simple(messages, max_tokens=None, temperature=None, stream=False, **kwargs):
    """Wrapper for LLM chat completion"""
    return base_container.llm_chat_simple(messages, max_tokens, temperature, stream, **kwargs)

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
        memory_rag_results = []  # Results from memory container
        
        if not is_personal_query:
            # Check document RAG (files/documents)
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
                        
                        # Search for relevant results (uses RAG_SEARCH_THRESHOLD and RAG_SEARCH_K from rag_client config)
                        rag_results = rag_client.search(query=prompt)
                        
                        # Get threshold from RAG client config for logging
                        try:
                            from rag.rag_client import RAG_SEARCH_THRESHOLD
                            threshold_display = RAG_SEARCH_THRESHOLD
                        except ImportError:
                            threshold_display = "default"
                        
                        # Only use RAG if search actually returns results above threshold
                        if rag_results and len(rag_results) > 0:
                            print(f"[Generic] ✅ RAG found {len(rag_results)} relevant results (threshold={threshold_display}) - will inject context")
                        else:
                            print(f"[Generic] 🔍 RAG search returned no results above threshold - skipping RAG injection")
                    else:
                        print(f"[Generic] 🔍 Query doesn't match RAG content - skipping RAG (faster response)")
                else:
                    print(f"[Generic] ⚠️ RAG client not available")
            except Exception as e:
                print(f"[Generic] ⚠️ RAG check failed: {e}")
                rag_client = None
            
            # Check memory container RAG (stored conversations)
            try:
                memory_container_url = os.environ.get('MEMORY_CONTAINER_URL', 'http://localhost:11438')
                # Quick check: does memory container have relevant conversations?
                quick_match_response = requests.post(
                    f"{memory_container_url}/rag/quick-match",
                    json={"query": prompt},
                    timeout=2
                )
                if quick_match_response.status_code == 200:
                    quick_match_data = quick_match_response.json()
                    if quick_match_data.get('has_match', False):
                        print(f"[Generic] 🔍 Query may match stored conversations - performing memory RAG search...")
                        # Search memory container for relevant conversations
                        memory_rag_response = requests.post(
                            f"{memory_container_url}/rag/search",
                            json={
                                "query": prompt,
                                "k": 3,  # Get top 3 relevant conversations
                                "threshold": 0.35
                            },
                            timeout=5
                        )
                        if memory_rag_response.status_code == 200:
                            memory_rag_data = memory_rag_response.json()
                            memory_rag_results = memory_rag_data.get('results', [])
                            if memory_rag_results and len(memory_rag_results) > 0:
                                print(f"[Generic] ✅ Memory RAG found {len(memory_rag_results)} relevant conversations - will inject context")
                            else:
                                print(f"[Generic] 🔍 Memory RAG search returned no results above threshold")
                        else:
                            print(f"[Generic] ⚠️ Memory RAG search failed: HTTP {memory_rag_response.status_code}")
                    else:
                        print(f"[Generic] 🔍 Query doesn't match stored conversations - skipping memory RAG")
                else:
                    # Memory container not available or error - continue without it
                    pass
            except requests.exceptions.RequestException as e:
                # Memory container not available - this is OK, continue without it
                pass
            except Exception as e:
                print(f"[Generic] ⚠️ Memory RAG check failed: {e}")
        
        should_use_rag = (rag_results and len(rag_results) > 0) or (memory_rag_results and len(memory_rag_results) > 0)
        should_use_memory_rag = (memory_rag_results and len(memory_rag_results) > 0)
        print(f"[Generic] 🔍 Query analysis: is_personal={is_personal_query}, should_use_rag={should_use_rag}, should_use_memory_rag={should_use_memory_rag}")
        
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
                
                # Build RAG context with improved formatting and relevance ordering
                MAX_CHARS_PER_RESULT = 1200
                rag_chunks = []
                
                # Sort results by score (highest first) for better context ordering
                sorted_results = sorted(rag_results, key=lambda x: x.get('score', 0), reverse=True)
                
                for i, r in enumerate(sorted_results, 1):
                    text = r.get("text", "")
                    score = r.get("score", 0)
                    metadata = r.get("metadata", {})
                    
                    if text:
                        # Extract source information
                        source_name = "unknown"
                        if isinstance(metadata, dict):
                            file_path = metadata.get('file_path', '')
                            if file_path:
                                from pathlib import Path
                                source_name = Path(file_path).name
                            else:
                                source_name = metadata.get('document_name', 'unknown')
                        
                        # Truncate if too long, but try to break at sentence boundary
                        if len(text) > MAX_CHARS_PER_RESULT:
                            truncated = text[:MAX_CHARS_PER_RESULT]
                            # Try to break at last sentence boundary
                            last_period = max(
                                truncated.rfind('. '),
                                truncated.rfind('! '),
                                truncated.rfind('? ')
                            )
                            if last_period > MAX_CHARS_PER_RESULT * 0.7:  # Only if we're not losing too much
                                truncated = truncated[:last_period + 1] + "..."
                            else:
                                # Fall back to word boundary
                                last_space = truncated.rfind(' ')
                                if last_space > MAX_CHARS_PER_RESULT * 0.8:
                                    truncated = truncated[:last_space] + "..."
                                else:
                                    truncated = truncated + "..."
                            text = truncated
                        
                        # Format chunk (minimal metadata, let LLM focus on content)
                        # Only include source if available, relevance score is implicit in ordering
                        if source_name != "unknown":
                            formatted_chunk = f"{text}\n[Source: {source_name}]"
                        else:
                            formatted_chunk = text
                        rag_chunks.append(formatted_chunk)
                
                # Join with clear separators
                rag_context = "\n\n---\n\n".join(rag_chunks)
                print(f"[Generic] ✅ Using document RAG context ({len(rag_context)} chars, ~{len(rag_context)//4} tokens) from {len(rag_chunks)} chunks for LLM response")
            except Exception as e:
                print(f"[Generic] ⚠️ RAG failed, using direct LLM: {e}")
                import traceback
                traceback.print_exc()
                rag_context = ""  # Clear context on error
        
        # Add memory RAG results (stored conversations) to context (works independently)
        if memory_rag_results and len(memory_rag_results) > 0:
            try:
                print(f"[Generic] 🔍 Memory RAG injection: '{prompt[:50]}...'")
                memory_chunks = []
                MAX_CHARS_PER_RESULT = 1200  # Same as document RAG
                
                # Sort memory results by score (highest first)
                sorted_memory_results = sorted(memory_rag_results, key=lambda x: x.get('score', 0), reverse=True)
                
                for i, result in enumerate(sorted_memory_results, 1):
                    text = result.get('text', '')
                    score = result.get('score', 0)
                    metadata = result.get('metadata', {})
                    
                    if text:
                        # Extract conversation info
                        conv_id = metadata.get('conversation_id', 'unknown')
                        timestamp = metadata.get('datetime', metadata.get('timestamp', ''))
                        source = metadata.get('source', 'unknown')
                        
                        # Format memory chunk
                        if timestamp:
                            memory_chunk = f"[Previous conversation ({timestamp})]: {text}"
                        else:
                            memory_chunk = f"[Previous conversation]: {text}"
                        
                        # Truncate if too long
                        if len(memory_chunk) > MAX_CHARS_PER_RESULT:
                            truncated = memory_chunk[:MAX_CHARS_PER_RESULT]
                            last_period = max(
                                truncated.rfind('. '),
                                truncated.rfind('! '),
                                truncated.rfind('? ')
                            )
                            if last_period > MAX_CHARS_PER_RESULT * 0.8:
                                truncated = truncated[:last_period + 1] + "..."
                            else:
                                truncated = truncated + "..."
                            memory_chunk = truncated
                        
                        memory_chunks.append(memory_chunk)
                        print(f"[Generic]   [Memory {i}] Score: {score:.3f}, Source: {source}, Preview: '{text[:50]}...'")
                
                if memory_chunks:
                    memory_context = "\n\n---\n\n".join(memory_chunks)
                    # Combine with document RAG context if it exists
                    if rag_context:
                        rag_context = f"{rag_context}\n\n---\n\n[Stored Conversations]\n{memory_context}"
                    else:
                        rag_context = f"[Stored Conversations]\n{memory_context}"
                    print(f"[Generic] ✅ Added memory RAG context ({len(memory_context)} chars) from {len(memory_chunks)} conversations")
            except Exception as e:
                print(f"[Generic] ⚠️ Memory RAG context building failed: {e}")
                import traceback
                traceback.print_exc()
        
        if rag_context:
            print(f"[Generic] ✅ Using combined RAG context ({len(rag_context)} chars, ~{len(rag_context)//4} tokens) for LLM response")
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
        # Check if RAG context is present (more authoritative than general knowledge)
        has_rag_context = "Knowledge context:" in combined_context
        
        # Detect if user is asking for instructions/steps
        instruction_keywords = ['how to', 'how do i', 'steps', 'step by step', 'instructions', 'guide me', 'walk me through', 'show me how']
        is_instruction_request = any(keyword in prompt.lower() for keyword in instruction_keywords)
        
        if has_rag_context:
            # Dynamic prompt construction with Aura Vision identity
            if is_instruction_request:
                system_content = (
                    f"{combined_context}\n\n"
                    "You are Aura Vision, an AI agent created by Ledger AI Quantum Corporation. "
                    "You act as a proactive AI agent guiding users to better outcomes through gentle guidance.\n\n"
                    "Guidelines:\n"
                    "- Provide a clear, step-by-step response (numbered steps)\n"
                    "- Keep each step concise and actionable\n"
                    "- Synthesize information from the context sections naturally\n"
                    "- Integrate information from multiple sections when relevant\n"
                    "- Rephrase and explain in your own words rather than copying text\n"
                    "- If the context doesn't fully address the question, supplement appropriately\n"
                    "- Be conversational and friendly, like Siri or Alexa"
                )
            else:
                system_content = (
                    f"{combined_context}\n\n"
                    "You are Aura Vision, an AI agent created by Ledger AI Quantum Corporation. "
                    "You act as a proactive AI agent guiding users to better outcomes through gentle guidance.\n\n"
                    "Guidelines:\n"
                    "- Keep responses short and conversational, like Siri or Alexa (2-3 sentences typically)\n"
                    "- Be friendly, helpful, and concise\n"
                    "- Synthesize information from the context sections naturally\n"
                    "- Integrate information from multiple sections when relevant\n"
                    "- Rephrase and explain in your own words rather than copying text\n"
                    "- If the context doesn't fully address the question, supplement appropriately\n"
                    "- Avoid lengthy explanations unless specifically requested"
                )
        else:
            # No RAG context, use standard prompt with Aura Vision identity
            if is_instruction_request:
                system_content = (
                    f"{combined_context}\n\n"
                    "You are Aura Vision, an AI agent created by Ledger AI Quantum Corporation. "
                    "You act as a proactive AI agent guiding users to better outcomes through gentle guidance.\n\n"
                    "Provide a clear, step-by-step response (numbered steps). Keep each step concise and actionable. "
                    "Be conversational and friendly, like Siri or Alexa."
                )
            else:
                system_content = (
                    f"{combined_context}\n\n"
                    "You are Aura Vision, an AI agent created by Ledger AI Quantum Corporation. "
                    "You act as a proactive AI agent guiding users to better outcomes through gentle guidance.\n\n"
                    "Keep your response short and conversational, like Siri or Alexa (2-3 sentences typically). "
                    "Be friendly, helpful, and concise. Avoid lengthy explanations unless specifically requested."
                )
        
        messages = [
            {
                "role": "system",
                "content": system_content,
            },
            {
                "role": "user",
                "content": prompt,
            }
        ]
        return llm_chat_simple(messages, max_tokens=MAX_TOKENS_RAG_MODE, stream=stream)

    # Fallback to direct LLM conversation without external context
    # Detect if user is asking for instructions/steps
    instruction_keywords = ['how to', 'how do i', 'steps', 'step by step', 'instructions', 'guide me', 'walk me through', 'show me how']
    is_instruction_request = any(keyword in prompt.lower() for keyword in instruction_keywords)
    
    if is_instruction_request:
        system_prompt = (
            "You are Aura Vision, an AI agent created by Ledger AI Quantum Corporation. "
            "You act as a proactive AI agent guiding users to better outcomes through gentle guidance.\n\n"
            "Provide a clear, step-by-step response (numbered steps) to the user's question. "
            "Keep each step concise and actionable. Be conversational and friendly, like Siri or Alexa."
        )
    else:
        system_prompt = (
            "You are Aura Vision, an AI agent created by Ledger AI Quantum Corporation. "
            "You act as a proactive AI agent guiding users to better outcomes through gentle guidance.\n\n"
            "Keep your response short and conversational, like Siri or Alexa (2-3 sentences typically). "
            "Be friendly, helpful, and concise. Avoid lengthy explanations unless specifically requested."
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
                    # DO NOT clean here - cleaning happens in web interface right before displaying
                    # This preserves raw text structure for bubble detection (numbered items, etc.)
                    accumulated = ""
                    for word in word_stream:
                        accumulated += word
                        # Send raw accumulated text during streaming (no cleaning)
                        yield f"data: {json.dumps({'response': accumulated, 'done': False})}\n\n"
                    
                    # Send final message with raw text (cleaning happens in web interface)
                    yield f"data: {json.dumps({'response': accumulated, 'done': True})}\n\n"
                    print(f"[Generic] ✅ Streamed response complete")
                else:
                    # Fallback: non-streaming (result is a string)
                    print(f"[Generic] ⚠️ Streaming not available - sending complete response")
                    fallback_text = result if isinstance(result, str) and result else "I apologize, I encountered an error."
                    cleaned_fallback = _clean_text_formatting(fallback_text)
                    yield f"data: {json.dumps({'response': cleaned_fallback, 'done': True})}\n\n"
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
            cleaned_response = _clean_text_formatting(response)
            return jsonify({"response": cleaned_response})
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
    
    session_id = session_id or "default"
    print(f"[Generic] 💬 Streaming Session: {session_id}, Prompt: '{prompt[:50]}...'")
    
    # Build conversation memory context for this prompt (with fallback)
    memory_context = None
    try:
        memory_context = conversation_orchestrator._build_memory_context(prompt)
        if memory_context:
            print(f"[Generic] 📝 Retrieved conversation memory context for session {session_id}")
    except Exception as e:
        print(f"[Generic] ⚠️ Failed to retrieve conversation memory (continuing without it): {e}")
        memory_context = None  # Fallback: continue without memory context
    
    # Store user's prompt in conversation memory for future reference (with fallback)
    try:
        conversation_orchestrator._store_in_memory(
            prompt,
            {
                "session_id": session_id,
                "timestamp": time.time(),
                "type": "user_message",
            },
        )
    except Exception as e:
        print(f"[Generic] ⚠️ Failed to store user message in memory (continuing): {e}")
        # Fallback: continue without storing (non-critical operation)
    
    def generate_response():
        full_response_text = ""  # Accumulate full response for memory storage
        try:
            # Use streaming mode to get tokens as they're generated, with memory context
            result = handle_conversation(prompt, session_id, memory_context=memory_context, stream=True)
            
            # Check if result is a generator (streaming)
            if hasattr(result, '__iter__') and not isinstance(result, str):
                print(f"[Generic] ✅ Streaming enabled - tokens will be yielded as generated")
                
                # Debug: Consume all chunks to see what we get
                raw_chunks = []
                try:
                    for chunk in result:
                        raw_chunks.append(chunk)
                        if len(raw_chunks) <= 3:  # Debug first 3 chunks
                            print(f"[Generic] 🔍 DEBUG: Raw chunk {len(raw_chunks)}: type={type(chunk).__name__}")
                            if isinstance(chunk, dict):
                                if 'choices' in chunk and chunk['choices']:
                                    choice = chunk['choices'][0]
                                    delta = choice.get('delta', {})
                                    finish_reason = choice.get('finish_reason')
                                    print(f"[Generic] 🔍 DEBUG: Chunk {len(raw_chunks)} - delta keys: {list(delta.keys())}, finish_reason: {finish_reason}")
                                    if 'content' in delta:
                                        print(f"[Generic] 🔍 DEBUG: Chunk {len(raw_chunks)} content: {repr(delta['content'][:100])}")
                    print(f"[Generic] 🔍 DEBUG: Total raw chunks received: {len(raw_chunks)}")
                    if len(raw_chunks) == 0:
                        print(f"[Generic] ⚠️ DEBUG: Raw LLM iterator is EMPTY")
                    elif len(raw_chunks) == 1:
                        # Check if the single chunk has a finish_reason
                        if isinstance(raw_chunks[0], dict) and 'choices' in raw_chunks[0]:
                            choice = raw_chunks[0]['choices'][0]
                            finish_reason = choice.get('finish_reason')
                            if finish_reason:
                                print(f"[Generic] ⚠️ DEBUG: Model stopped after 1 chunk with finish_reason: {finish_reason}")
                            else:
                                print(f"[Generic] ⚠️ DEBUG: Model only generated 1 chunk (role only) - no finish_reason")
                except Exception as peek_error:
                    print(f"[Generic] ⚠️ DEBUG: Error consuming iterator: {peek_error}")
                    import traceback
                    traceback.print_exc()
                
                # Re-create iterator from collected chunks for processing
                print(f"[Generic] 🔍 DEBUG: About to normalize {len(raw_chunks)} chunks")
                normalized_chunks = _normalize_stream_chunks(iter(raw_chunks))
                word_stream = _word_stream_from_chunks(normalized_chunks)
                sentence_stream = _sentence_tag_stream(word_stream)
                
                token_count = 0
                chunk_count = 0
                normalized_count = 0
                for token in sentence_stream:
                    token_count += 1
                    chunk_count += 1
                    if chunk_count <= 5:  # Debug first 5 tokens
                        print(f"[Generic] 🔍 DEBUG: Token {chunk_count}: {repr(token[:50])}")
                    # Accumulate tokens for memory storage (skip control tags)
                    if not (token.startswith('<') and token.endswith('>')):
                        full_response_text += token
                    yield f"{token}\n"
                
                if token_count == 0:
                    print(f"[Generic] ⚠️ WARNING: No tokens generated by LLM - sending empty response")
                    print(f"[Generic] 🔍 DEBUG: Raw chunks: {len(raw_chunks)}, Normalized chunks processed: {chunk_count}, Tokens yielded: {token_count}")
                    # Send empty sentence tags to indicate completion
                    yield "<sentence_start>\n"
                    yield "<sentence_end>\n"
                else:
                    print(f"[Generic] ✅ Streamed response complete ({token_count} tokens)")
                
                # Store assistant's response in conversation memory after streaming completes (with fallback)
                if full_response_text.strip():
                    try:
                        conversation_orchestrator._store_in_memory(
                            full_response_text.strip(),
                            {
                                "session_id": session_id,
                                "timestamp": time.time(),
                                "type": "assistant_response",
                            },
                        )
                        print(f"[Generic] 💾 Stored assistant response in conversation memory")
                    except Exception as e:
                        print(f"[Generic] ⚠️ Failed to store assistant response in memory (continuing): {e}")
                        # Fallback: continue without storing (non-critical operation)
            else:
                # Fallback: non-streaming (result is a string)
                print(f"[Generic] ⚠️ Streaming not available - yielding complete response")
                fallback_text = result if isinstance(result, str) and result else "I apologize, I encountered an error."
                full_response_text = fallback_text  # Store for memory
                normalized_chunks = _normalize_stream_chunks(iter([fallback_text]))
                word_stream = _word_stream_from_chunks(normalized_chunks)
                sentence_stream = _sentence_tag_stream(word_stream)
                for token in sentence_stream:
                    yield f"{token}\n"
                
                # Store assistant's response in conversation memory (with fallback)
                if full_response_text.strip():
                    try:
                        conversation_orchestrator._store_in_memory(
                            full_response_text.strip(),
                            {
                                "session_id": session_id,
                                "timestamp": time.time(),
                                "type": "assistant_response",
                            },
                        )
                        print(f"[Generic] 💾 Stored assistant response in conversation memory")
                    except Exception as e:
                        print(f"[Generic] ⚠️ Failed to store assistant response in memory (continuing): {e}")
                        # Fallback: continue without storing (non-critical operation)
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

# Import shared text cleaning utility (if available)
# Note: In container environment, we define it locally for simplicity
# The function is identical to aura-control/utils/text_cleaning.py for consistency
def _clean_text_formatting(text: str) -> str:
    """
    Clean and normalize text formatting for better readability.
    Matches TTS cleaning logic for consistency between chat and voice output.
    Removes markdown formatting, fixes spacing, removes hashtags and asterisks.
    
    This function is identical to aura-control/utils/text_cleaning.py::clean_text_formatting()
    to ensure chat and TTS use the same formatting.
    """
    if not text:
        return text
    
    # Remove markdown headers (hashtags at start of line)
    text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)
    # Remove standalone hashtags
    text = re.sub(r'#{1,6}(?=\s|$)', '', text)
    
    # Remove markdown bold/italic (asterisks)
    text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)  # **text** -> text
    text = re.sub(r'\*([^*\n]+)\*', r'\1', text)  # *text* -> text
    # Remove standalone asterisks (markdown formatting artifacts)
    text = re.sub(r'\*\*+', '', text)  # Remove multiple asterisks
    text = re.sub(r'(?<!\w)\*(?!\w)', '', text)  # Remove single asterisks not part of words
    
    # Fix missing spaces after punctuation
    text = re.sub(r'([a-zA-Z0-9])([.!?])([a-zA-Z-])', r'\1\2 \3', text)  # word.word -> word. word
    text = re.sub(r'([,.!?:;])([a-zA-Z])', r'\1 \2', text)  # word,word -> word, word
    text = re.sub(r'([a-zA-Z0-9])(\()', r'\1 \2', text)  # word(word -> word (word
    text = re.sub(r'(\))([a-zA-Z0-9])', r'\1 \2', text)  # word)word -> word) word
    
    # Normalize multiple spaces
    text = re.sub(r' {2,}', ' ', text)
    
    return text.strip()


def _normalize_stream_chunks(chunk_iter):
    """
    Normalize mixed-type streaming chunks (dicts, strings) to plain strings.
    Note: First chunk from llama.cpp often only has 'role' in delta - this is normal.
    Subsequent chunks will have 'content' in delta.
    """
    chunk_idx = 0
    chunks_with_content = 0
    for chunk in chunk_iter:
        chunk_idx += 1
        if chunk_idx <= 5:  # Debug first 5 chunks
            print(f"[Generic] 🔍 DEBUG: _normalize_stream_chunks received chunk {chunk_idx}: type={type(chunk).__name__}")
        
        if isinstance(chunk, dict):
            if 'choices' in chunk and len(chunk['choices']) > 0:
                choice = chunk['choices'][0]
                delta = choice.get('delta', {})
                content = delta.get('content', '')
                finish_reason = choice.get('finish_reason')
                
                if chunk_idx <= 5:
                    print(f"[Generic] 🔍 DEBUG: Chunk {chunk_idx} delta keys: {list(delta.keys())}, has_content={bool(content)}, finish_reason={finish_reason}")
                
                # Check if generation finished early
                if finish_reason and finish_reason != 'null':
                    print(f"[Generic] ⚠️ DEBUG: LLM finished early with reason: {finish_reason}")
                    if not content:
                        print(f"[Generic] ⚠️ DEBUG: No content in final chunk - model may have stopped generating")
                
                if content:
                    chunks_with_content += 1
                    if chunk_idx <= 5:
                        print(f"[Generic] 🔍 DEBUG: Extracted content from delta: {repr(content[:50])}")
                    yield content
                elif chunk_idx <= 5:
                    # First chunk often only has 'role' - this is normal, just log it
                    print(f"[Generic] 🔍 DEBUG: Chunk {chunk_idx} has no content (only role/metadata) - skipping")
            elif 'content' in chunk:
                content = chunk.get('content', '')
                if content:
                    chunks_with_content += 1
                    if chunk_idx <= 5:
                        print(f"[Generic] 🔍 DEBUG: Extracted content from chunk: {repr(content[:50])}")
                    yield content
            elif chunk_idx <= 5:
                print(f"[Generic] 🔍 DEBUG: Dict chunk has no content: {list(chunk.keys())}")
        elif isinstance(chunk, str):
            if chunk:
                chunks_with_content += 1
                if chunk_idx <= 5:
                    print(f"[Generic] 🔍 DEBUG: String chunk: {repr(chunk[:50])}")
                yield chunk
        else:
            chunks_with_content += 1
            if chunk_idx <= 5:
                print(f"[Generic] 🔍 DEBUG: Unknown chunk type, converting to string: {repr(str(chunk)[:50])}")
            yield str(chunk)
    
    if chunk_idx == 0:
        print(f"[Generic] ⚠️ DEBUG: _normalize_stream_chunks received NO chunks from iterator")
    elif chunks_with_content == 0:
        print(f"[Generic] ⚠️ DEBUG: _normalize_stream_chunks received {chunk_idx} chunks but NONE had content")
        print(f"[Generic] 🔍 DEBUG: This usually means the LLM stopped generating after the first chunk (role only)")
    else:
        print(f"[Generic] 🔍 DEBUG: _normalize_stream_chunks processed {chunk_idx} chunks, {chunks_with_content} had content")


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
    """Wrapper for base class sentence tagging"""
    return base_container.sentence_tag_stream(word_stream)
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
        # 2. Colons: Don't split on colons - they're usually part of list headers that should stay with content
        # This prevents awkward splits like "**Symptoms:" being a separate sentence
        # Colons will naturally be part of the sentence and cleaned up in post-processing
        # (Removed colon-based splitting to prevent awkward chunking)
    
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
    
    # Load model with GPU acceleration using base class
    print(f"[Generic] 📦 Loading model: {SIMPLE_MODEL_PATH}")
    # Offload all layers to GPU for maximum acceleration (set to 0 to disable GPU)
    # For Jetson, offloading all layers typically provides best performance
    n_gpu_layers = -1  # -1 = offload all layers to GPU, 0 = CPU only
    print(f"[Generic] 🚀 GPU acceleration: {n_gpu_layers} layers offloaded to GPU")
    
    # Override base class load_model to add GPU support
    from llama_cpp import Llama
    base_container.model_path = SIMPLE_MODEL_PATH
    base_container.llm_simple = Llama(
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
    base_container._model_loaded = True
    llm_simple = base_container.llm_simple  # Set global reference
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

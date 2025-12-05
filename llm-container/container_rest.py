# === container_rest.py — Aura Generic Conversational Container ===
# Provides general conversation with RAG-powered knowledge

from flask import Flask, request, jsonify, stream_with_context, Response
import os, threading, atexit, time
import requests
from typing import List, Dict, Optional
from collections import Counter
import re
import logging
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

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
    default_model_path="/models/phi-2.Q4_K_M.gguf"
)

# Override default parameters for generic container
base_container.LLM_NUM_PREDICT_DEFAULT = 800  # Increased for comprehensive responses
base_container.SIMPLE_N_CTX = 4096  # Reduced from 8192 for lower latency
base_container.N_BATCH = 256  # Reduced for faster generation
# Override chat format for phi-2 (phi-2 uses chatml format)
base_container.SIMPLE_CHAT_FORMAT = os.getenv('SIMPLE_CHAT_FORMAT', 'chatml')

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

# === Debug Mode: Show LLM Reasoning ===
# Set SHOW_REASONING_DEBUG=true to make LLM show its reasoning step-by-step in the output (visible chain-of-thought)
# 
# IMPORTANT: We CANNOT see the LLM's internal reasoning (it's a black box - we only see input/output).
# However, we CAN make the LLM show its reasoning PROCESS in the OUTPUT using chain-of-thought prompting.
# This will make the LLM explicitly show: what it's analyzing, what it finds in each chunk, etc.
#
# Note: Using phi-2 (2.7B parameters) - optimized for reasoning tasks and should follow step-by-step instructions well.
SHOW_REASONING_DEBUG = os.environ.get('SHOW_REASONING_DEBUG', 'false').lower() == 'true'
if SHOW_REASONING_DEBUG:
    print(f"[Generic] 🔍 SHOW_REASONING_DEBUG is ENABLED - phi-2 will show step-by-step reasoning in response")

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
    # Check if model is loaded before calling
    if not base_container._model_loaded or base_container.llm_simple is None:
        print(f"[Generic] ⚠️ ERROR: Model not loaded! _model_loaded={base_container._model_loaded}, llm_simple={base_container.llm_simple is not None}")
        if stream:
            return iter([])
        return ""
    
    # Reduced debug logging for performance
    result = base_container.llm_chat_simple(messages, max_tokens, temperature, stream, **kwargs)
    return result

# === Filler Phrases ===
def get_filler_phrase() -> str:
    """Get a random filler phrase to use while processing (reduces perceived latency)"""
    import random
    filler_phrases = [
        "One moment, let me think about that.",
        "Let me search through my knowledge base for that information.",
        "Give me a moment to recall the details.",
        "Let me check what I know about that.",
        "I'll need a moment to look that up for you.",
        "Let me think about that for a second.",
        "One moment, I'm searching through my knowledge base.",
        "Let me gather the relevant information for you.",
        "Give me a moment to process that question.",
        "I'll need a second to find the right information.",
    ]
    return random.choice(filler_phrases)

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
        needs_filler_phrase = False  # Flag to indicate if we should yield filler phrase before LLM response
        memory_rag_results = []  # Results from memory container
        memory_rag_failed = False  # Track if memory RAG failed (timeout, error, etc.)
        
        if not is_personal_query:
            # Detect if query is asking for "what else" or additional information
            is_followup_query = any(phrase in prompt.lower() for phrase in ['what else', 'anything else', 'more about', 'additional', 'other'])
            
            # Parallelize document RAG and memory RAG searches for better latency
            def search_document_rag():
                """Search document RAG in parallel"""
                try:
                    client = get_rag_client()
                    if client:
                        has_content = client.quick_content_match(prompt)
                        if has_content:
                            print(f"[Generic] 🔍 Query may match RAG content - performing search...")
                            if hasattr(client, '_cpu_chunks') and client._cpu_chunks:
                                print(f"[Generic] 📊 RAG index: {len(client._cpu_chunks)} chunks available")
                            elif hasattr(client, '_cpu_index') and client._cpu_index:
                                index_size = client._cpu_index.ntotal if hasattr(client._cpu_index, 'ntotal') else 0
                                print(f"[Generic] 📊 RAG index: {index_size} vectors available")
                            results = client.search(query=prompt)
                            if results and len(results) > 0:
                                try:
                                    from rag.rag_client import RAG_SEARCH_THRESHOLD
                                    threshold_display = RAG_SEARCH_THRESHOLD
                                except ImportError:
                                    threshold_display = "default"
                                print(f"[Generic] ✅ RAG found {len(results)} relevant results (threshold={threshold_display}) - will inject context")
                            else:
                                print(f"[Generic] 🔍 RAG search returned no results above threshold - skipping RAG injection")
                            return client, results
                        else:
                            print(f"[Generic] 🔍 Query doesn't match RAG content - skipping RAG (faster response)")
                    return None, []
                except Exception as e:
                    print(f"[Generic] ⚠️ RAG check failed: {e}")
                    return None, []
            
            def search_memory_rag():
                """Search memory RAG in parallel"""
                try:
                    memory_container_url = os.environ.get('MEMORY_CONTAINER_URL', 'http://localhost:11438')
                    # For follow-up queries, skip quick-match and go straight to search (faster)
                    if is_followup_query:
                        print(f"[Generic] 🔍 Follow-up query detected - skipping quick-match, performing direct search...")
                        quick_match_response = None
                    else:
                        # Quick check: does memory container have relevant conversations?
                        # Reduced timeout from 2s to 500ms for faster response
                        try:
                            quick_match_response = requests.post(
                                f"{memory_container_url}/rag/quick-match",
                                json={"query": prompt},
                                timeout=0.5  # Reduced from 2s
                            )
                        except requests.exceptions.Timeout:
                            print(f"[Generic] ⚠️ Memory container quick-match timeout - index may be rebuilding, skipping memory RAG")
                            quick_match_response = None
                    
                    if quick_match_response is None or (quick_match_response and quick_match_response.status_code == 200 and quick_match_response.json().get('has_match', False)):
                        print(f"[Generic] 🔍 Query may match stored conversations - performing memory RAG search...")
                        from rag.rag_client import RAG_SEARCH_THRESHOLD, RAG_SEARCH_K
                        memory_rag_threshold = RAG_SEARCH_THRESHOLD
                        memory_rag_k = RAG_SEARCH_K
                        
                        # Reduced timeout from 15s to 3s (more realistic)
                        try:
                            memory_rag_response = requests.post(
                                f"{memory_container_url}/rag/search",
                                json={
                                    "query": prompt,
                                    "k": memory_rag_k * 2,  # Get more candidates for LLM scoring
                                    "threshold": memory_rag_threshold * 0.8,  # Lower threshold to get more candidates
                                    "rerank": True  # Enable re-ranking to pre-filter irrelevant conversations
                                },
                                timeout=3  # Reduced from 15s to 3s
                            )
                        except requests.exceptions.Timeout:
                            print(f"[Generic] ⚠️ Memory RAG search timeout (>3s) - index may be rebuilding, continuing without memory context")
                            memory_rag_response = None
                        
                        if memory_rag_response and memory_rag_response.status_code == 200:
                            memory_rag_data = memory_rag_response.json()
                            return memory_rag_data.get('results', []), memory_rag_threshold, memory_rag_k
                    return [], None, None
                except Exception as e:
                    print(f"[Generic] ⚠️ Memory RAG check failed: {e}")
                    return [], None, None
            
            # Run both searches in parallel
            with ThreadPoolExecutor(max_workers=2) as executor:
                doc_future = executor.submit(search_document_rag)
                memory_future = executor.submit(search_memory_rag)
                
                # Get document RAG results
                rag_client, rag_results = doc_future.result()
                
                # Get memory RAG candidates (pre-filtered by re-ranking, but need LLM scoring to distinguish questions from answers)
                memory_rag_candidates, memory_rag_threshold, memory_rag_k = memory_future.result()
            
            # Process memory RAG candidates with LLM scoring
            # Re-ranking pre-filters irrelevant conversations, but LLM scoring distinguishes questions from answers
            memory_rag_results = []
            if memory_rag_candidates and memory_rag_threshold and memory_rag_k:
                # For follow-up queries, skip LLM scoring and use re-ranked scores only (faster)
                if is_followup_query:
                    print(f"[Generic] ⚡ Follow-up query - using re-ranked scores only (skipping LLM scoring for speed)")
                    scored_candidates = sorted(memory_rag_candidates, key=lambda x: x.get('score', 0), reverse=True)
                    memory_rag_results = [
                        r for r in scored_candidates 
                        if r.get('score', 0) >= memory_rag_threshold
                    ][:memory_rag_k]
                else:
                    # Use LLM scoring to distinguish questions from answers
                    print(f"[Generic] 🤖 Using LLM to score {len(memory_rag_candidates)} conversation candidates (pre-filtered by re-ranking)...")
                    
                    # Build prompt for LLM to score conversations
                    conversations_text = ""
                    for i, candidate in enumerate(memory_rag_candidates, 1):
                        conv_text = candidate.get('text', '')
                        conversations_text += f"{i}. {conv_text}\n"
                    
                    # Pre-filter: Detect obvious questions using heuristics
                    question_words = {'who', 'what', 'where', 'when', 'why', 'how', 'which', 'whose', 'whom', 'do', 'does', 'did', 'is', 'are', 'was', 'were', 'can', 'could', 'will', 'would', 'should', 'may', 'might', 'remember'}
                    filtered_candidates = []
                    for candidate in memory_rag_candidates:
                        text = candidate.get('text', '').strip()
                        # Check if it's a question: starts with question word or "do you remember", ends with "?"
                        is_question = False
                        text_lower = text.lower()
                        if text.endswith('?'):
                            # Check if starts with question word
                            first_word = text_lower.split()[0] if text_lower.split() else ""
                            if first_word in question_words or text_lower.startswith('do you remember') or text_lower.startswith('remember'):
                                is_question = True
                        
                        # Only include if it's NOT a question (has actual information)
                        if not is_question:
                            filtered_candidates.append(candidate)
                        else:
                            print(f"[Generic]   [Pre-filter] ❌ Excluded question: '{text[:60]}...'")
                    
                    # If all candidates were questions, skip LLM scoring and don't inject
                    if not filtered_candidates:
                        print(f"[Generic] ⚠️ All memory RAG candidates are questions - skipping injection to prevent hallucination")
                        memory_rag_results = []
                    else:
                        # Use filtered candidates for LLM scoring
                        conversations_text = ""
                        for i, candidate in enumerate(filtered_candidates, 1):
                            conv_text = candidate.get('text', '')
                            conversations_text += f"{i}. {conv_text}\n"
                        
                        scoring_prompt = f"""Rate how well each previous conversation answers the user's question.

User's question: "{prompt}"

Previous conversations (pre-filtered to exclude questions):
{conversations_text}

For each conversation, assign a score from 0.0 to 1.0:
- 1.0 = Perfectly answers the question with relevant information (e.g., "Elizabeth Martinez is an ICU nurse at Memorial Hermann")
- 0.7-0.9 = Mostly answers the question, contains relevant information
- 0.4-0.6 = Partially relevant, some useful information
- 0.1-0.3 = Minimally relevant, little useful information
- 0.0 = Not relevant, doesn't answer the question

IMPORTANT: All conversations here have been pre-filtered to exclude questions. Only score based on how well they provide actual information/answers.

Return ONLY a JSON array with exactly {len(filtered_candidates)} scores in order.
Example: [0.8, 0.9, 0.7, 0.6, 0.5]

JSON array only:"""
                    
                        try:
                            # Call LLM for scoring (non-streaming, fast)
                            scoring_messages = [{"role": "user", "content": scoring_prompt}]
                            scoring_response = llm_chat_simple(
                                scoring_messages,
                                max_tokens=100,  # Reduced from 200 to 100 for faster response
                                temperature=0.3,
                                stream=False
                            )
                            
                            # Parse LLM response to extract scores
                            import json
                            json_match = re.search(r'\[[\d\.,\s]+\]', scoring_response)
                            if json_match:
                                scores = json.loads(json_match.group())
                                print(f"[Generic] ✅ LLM returned {len(scores)} scores")
                            else:
                                try:
                                    scores = json.loads(scoring_response.strip())
                                except:
                                    print(f"[Generic] ⚠️ Failed to parse LLM scores, using re-ranked scores as fallback")
                                    scores = [c.get('score', 0.0) for c in filtered_candidates]
                            
                            # Ensure we have the right number of scores
                            if len(scores) != len(filtered_candidates):
                                print(f"[Generic] ⚠️ LLM returned {len(scores)} scores but expected {len(filtered_candidates)}, using re-ranked scores as fallback")
                                scores = [c.get('score', 0.0) for c in filtered_candidates]
                            
                            # Update candidates with LLM scores
                            scored_candidates = []
                            for i, (candidate, llm_score) in enumerate(zip(filtered_candidates, scores)):
                                scored_candidates.append({
                                    **candidate,
                                    'score': float(llm_score),
                                    'original_re_ranked_score': candidate.get('score', 0.0),
                                    'llm_score': float(llm_score)
                                })
                                print(f"[Generic]   [{i+1}] LLM score: {llm_score:.3f} (re-ranked: {candidate.get('score', 0.0):.3f}): '{candidate.get('text', '')[:60]}...'")
                            
                            # Sort by LLM score (descending)
                            scored_candidates.sort(key=lambda x: x['score'], reverse=True)
                            
                            # Filter by threshold and return top k
                            # Use higher threshold (0.5) to filter out low-quality results
                            answer_threshold = max(memory_rag_threshold, 0.5)  # At least 0.5 to ensure quality answers
                            memory_rag_results = [
                                r for r in scored_candidates 
                                if r.get('score', 0) >= answer_threshold
                            ][:memory_rag_k]
                        except Exception as e:
                            print(f"[Generic] ⚠️ LLM scoring failed: {e}, falling back to re-ranked scores")
                            import traceback
                            traceback.print_exc()
                            # Fallback to re-ranked scores (use filtered candidates, not all candidates)
                            scored_candidates = sorted(filtered_candidates, key=lambda x: x.get('score', 0), reverse=True)
                            # Use higher threshold for fallback too
                            answer_threshold = max(memory_rag_threshold, 0.5)
                            memory_rag_results = [
                                r for r in scored_candidates 
                                if r.get('score', 0) >= answer_threshold
                            ][:memory_rag_k]
                
                if memory_rag_results and len(memory_rag_results) > 0:
                    print(f"[Generic] ✅ Memory RAG found {len(memory_rag_results)} relevant conversations (from {len(memory_rag_candidates)} candidates, threshold={max(memory_rag_threshold, 0.5):.2f}) - will inject context")
                else:
                    print(f"[Generic] 🔍 Memory RAG search returned no answer-like conversations (all were questions or below threshold) - skipping injection to prevent hallucination")
        
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
                    full_text = result.get('text', '')
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
                    print(f"[Generic]   [{i}] FULL CHUNK TEXT: '{full_text}'")  # Log full chunk for debugging
                
                # Detect if this is a list question early (for context handling)
                list_keywords = ['who are', 'who were', 'list all', 'list the', 'what are the', 'what are', 'name all', 'name the']
                is_list_query = any(keyword in prompt.lower() for keyword in list_keywords)
                
                # SIMPLIFIED: Trust final LLM to reason through chunks internally
                # No pre-filtering - let LLM understand query and extract valid information from all chunks
                # RAG semantic search already filtered by relevance, now LLM will internally reason through chunks
                print(f"[Generic] 📋 Using {len(rag_results)} RAG chunks - LLM will internally reason and extract valid information")
                
                # Use original RAG results, sorted by semantic similarity score
                sorted_results = sorted(rag_results, key=lambda x: x.get('score', 0), reverse=True)
                
                # Limit number of chunks to avoid token bloat (but keep more for list questions)
                max_chunks = 8 if is_list_query else 6
                sorted_results = sorted_results[:max_chunks]
                
                print(f"[Generic] 📋 Using top {len(sorted_results)} chunks (max {max_chunks}) for LLM reasoning")
                
                # Build RAG context - let LLM reason through all chunks
                MAX_CHARS_PER_RESULT = 1800 if is_list_query else 1200
                rag_chunks = []
                
                # Process chunks - let LLM reason through them internally
                for i, r in enumerate(sorted_results, 1):
                    text = r.get("text", "")
                    score = r.get("score", 0)
                    metadata = r.get("metadata", {})
                    
                    if text:
                        # Extract source information for reference
                        source_name = "unknown"
                        if isinstance(metadata, dict):
                            file_path = metadata.get('file_path', '')
                            if file_path:
                                from pathlib import Path
                                source_name = Path(file_path).name
                            else:
                                source_name = metadata.get('document_name', 'unknown')
                        
                        # Only truncate if extremely long (let LLM handle most filtering)
                        if len(text) > MAX_CHARS_PER_RESULT:
                            # Try to break at sentence boundary
                            truncated = text[:MAX_CHARS_PER_RESULT]
                            last_period = max(
                                truncated.rfind('. '),
                                truncated.rfind('! '),
                                truncated.rfind('? ')
                            )
                            if last_period > MAX_CHARS_PER_RESULT * 0.7:
                                text = truncated[:last_period + 1] + "..."
                            else:
                                text = truncated + "..."
                        
                        # Format chunk - LLM will internally reason and extract valid information
                        if source_name != "unknown":
                            formatted_chunk = f"{text}\n[Source: {source_name}]"
                        else:
                            formatted_chunk = text
                        rag_chunks.append(formatted_chunk)
                
                # Join with clear separators
                rag_context = "\n\n---\n\n".join(rag_chunks)
                print(f"[Generic] ✅ Using document RAG context ({len(rag_context)} chars, ~{len(rag_context)//4} tokens) from {len(rag_chunks)} chunks for LLM response")
                print(f"[Generic] 📄 FULL RAG CONTEXT SENT TO LLM:\n{rag_context}\n")  # Log full context for debugging
            except Exception as e:
                print(f"[Generic] ⚠️ RAG failed, using direct LLM: {e}")
                import traceback
                traceback.print_exc()
                rag_context = ""  # Clear context on error
        
        # Add memory RAG results (stored conversations) to context (works independently)
        # Apply same filtering logic as document RAG
        if memory_rag_results and len(memory_rag_results) > 0:
            try:
                print(f"[Generic] 🔍 Memory RAG injection: '{prompt[:50]}...'")
                memory_chunks = []
                MAX_CHARS_PER_RESULT = 1200  # Same as document RAG
                
                # Get threshold for filtering (same as document RAG)
                try:
                    from rag.rag_client import RAG_SEARCH_THRESHOLD
                    memory_rag_threshold = RAG_SEARCH_THRESHOLD
                except ImportError:
                    memory_rag_threshold = 0.30  # Default fallback
                
                # Sort memory results by score (highest first)
                sorted_memory_results = sorted(memory_rag_results, key=lambda x: x.get('score', 0), reverse=True)
                
                # Filter by threshold (same logic as document RAG)
                filtered_memory_results = [
                    r for r in sorted_memory_results 
                    if r.get('score', 0) >= memory_rag_threshold
                ]
                
                if not filtered_memory_results:
                    print(f"[Generic] 🔍 Memory RAG: All results below threshold={memory_rag_threshold:.2f}, skipping injection")
                    memory_rag_results = []  # Clear results
                else:
                    for i, result in enumerate(filtered_memory_results, 1):
                        text = result.get('text', '')
                        score = result.get('score', 0)
                        original_score = result.get('original_score', score)
                        keyword_score = result.get('keyword_score', 0)
                        is_question = result.get('is_question', False)
                        penalty = result.get('penalty', 0)
                        boost = result.get('boost', 0)
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
                            
                            # Truncate if too long (same logic as document RAG)
                            if len(memory_chunk) > MAX_CHARS_PER_RESULT:
                                truncated = memory_chunk[:MAX_CHARS_PER_RESULT]
                                # Try to break at sentence boundary (same as document RAG)
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
                                memory_chunk = truncated
                            
                            memory_chunks.append(memory_chunk)
                            threshold_status = "✅" if score >= memory_rag_threshold else "❌"
                            score_details = f"Score: {score:.3f} (semantic: {original_score:.3f}, keyword: {keyword_score:.3f}"
                            if penalty > 0:
                                score_details += f", -{penalty:.2f} penalty"
                            if boost > 0:
                                score_details += f", +{boost:.2f} boost"
                            score_details += f")"
                            is_answer_like = result.get('is_answer_like', False)
                            if is_answer_like:
                                question_marker = "📝 Answer-like"
                            else:
                                question_marker = "❓ Question" if is_question else "📝 Statement"
                            # Show more text for debugging (80 chars instead of 50)
                            print(f"[Generic]   [Memory {i}] {threshold_status} {question_marker} {score_details}, Source: {source}, Preview: '{text[:80]}...'")
                
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
        
        # Detect if user is asking for a list (e.g., "who are", "list all", "what are the")
        list_keywords = ['who are', 'who were', 'list all', 'list the', 'what are the', 'what are', 'name all', 'name the']
        is_list_request = any(keyword in prompt.lower() for keyword in list_keywords)
        
        if has_rag_context:
            # Dynamic prompt construction with Aura Vision identity
            # IMPORTANT: Include the prompt in the system message (matches working commit 1927b467c106120dd4e1231f600eccdaa5a93f08)
            if is_instruction_request:
                system_content = (
                    f"{combined_context}\n\n"
                    "You are Aura Vision, an AI agent created by Ledger AI Quantum Corporation. "
                    "You act as a proactive AI agent guiding users to better outcomes through gentle guidance.\n\n"
                    f"Based on the context provided above, answer the following question: {prompt}\n\n"
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
                # Add warning if memory RAG failed
                memory_warning = ""
                if memory_rag_failed:
                    memory_warning = "\n⚠️ IMPORTANT: Conversation memory context is unavailable (memory container may be rebuilding index). " \
                                   "Only provide information you are certain about from the provided context. " \
                                   "Do NOT make up or guess information about people, places, or facts. " \
                                   "If you don't have reliable information, say so rather than speculating.\n\n"
                
                # Extract person names from query for explicit instruction
                query_person_names = re.findall(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b', prompt)
                person_instruction = ""
                if query_person_names:
                    person_list = ", ".join(query_person_names)
                    person_instruction = f"\n\n⚠️ CRITICAL: The user is asking about {person_list}. ONLY use information from the context that specifically mentions {person_list}. DO NOT confuse information about {person_list} with information about other people mentioned in the context. If a context section mentions multiple people, only extract and use the information that pertains to {person_list}.\n"
                
                # Add list-specific instructions if this is a list question
                list_instruction = ""
                if is_list_request:
                    list_instruction = (
                        "\n🚨 CRITICAL: This question asks for a LIST or MULTIPLE items. "
                        "You MUST read EVERY context section from BEGINNING to END completely - analyze each section thoroughly and in detail. "
                        "Do NOT stop reading a section once you find some items - continue reading until the section ends to find ALL relevant items. "
                        "Missing even one item is a critical error - completeness is essential. "
                        "Only include items that have the EXACT relationship or category being asked about. "
                        "When listing people: Include their titles/roles when mentioned. "
                        "Format your answer naturally, clearly introducing the list.\n"
                    )
                
                # Build response length guideline - prioritize completeness for lists and debug mode
                if SHOW_REASONING_DEBUG:
                    # Debug mode: Allow longer responses to show reasoning
                    response_length_guideline = "- When debug mode is enabled, show your complete reasoning process - length is not a concern\n"
                elif is_list_request:
                    response_length_guideline = "- Keep items brief but include ALL items from the RAG context (completeness over brevity for lists)\n"
                else:
                    response_length_guideline = "- Keep responses short and conversational, like Siri or Alexa (2-3 sentences typically)\n"
                
                system_content = (
                    f"{combined_context}\n\n"
                    "You are Aura Vision, an AI agent created by Ledger AI Quantum Corporation. "
                    "You act as a proactive AI agent guiding users to better outcomes through gentle guidance.\n\n"
                    f"{memory_warning}"
                    f"{person_instruction}"
                    f"{list_instruction}"
                    "Instructions:\n"
                    "- Read ALL context sections THOROUGHLY from beginning to end - analyze each section in detail\n"
                    "- Do not skip any part of the context sections - read them completely\n"
                    "- Extract ALL information that directly answers the question\n"
                    "- For list questions: Find EVERY item mentioned across ALL sections - missing any is a serious error\n"
                    "- Be precise: Only include items with the EXACT relationship being asked about\n"
                    "- Format your answer naturally and clearly\n"
                    "- Include titles/roles when listing people"
                )
                
                # Build user message - add debug instructions if enabled
                user_content = prompt
                if SHOW_REASONING_DEBUG:
                    print(f"[Generic] 🔍 DEBUG MODE ENABLED - LLM will show step-by-step reasoning (will be logged, not spoken)")
                    user_content = (
                        f"⚠️ MANDATORY FORMAT: Show your reasoning, then your answer in separate sections.\n\n"
                        f"First, show your reasoning:\n"
                        f"STEP 1 - State what the user is asking\n"
                        f"STEP 2 - Analyze each context section (read EVERY section completely, state what you found in each)\n"
                        f"STEP 3 - Extract ALL relevant information (for lists, find EVERY item from ALL sections)\n\n"
                        f"Then, after showing your reasoning, write:\n"
                        f"---ANSWER---\n"
                        f"[Provide only your final answer here, no reasoning]\n"
                        f"---END ANSWER---\n\n"
                        f"Now answer: {prompt}"
                    )
                
            # For phi-2 with chatml format, separate system and user messages
            messages = [
                {
                    "role": "system",
                    "content": system_content,
                },
                {
                    "role": "user",
                    "content": user_content,
                }
            ]
            
            # DEBUG: Log full system prompt being sent to LLM
            print(f"\n{'='*80}")
            print(f"[LLM Reasoning Debug] 📤 FULL SYSTEM PROMPT BEING SENT TO LLM:")
            print(f"{'='*80}")
            print(system_content)
            print(f"{'='*80}\n")
            
            # Don't wrap the iterator - let base_container's debug_iterator handle logging
            # The base class already wraps it with debug logging
            return llm_chat_simple(messages, max_tokens=MAX_TOKENS_RAG_MODE, stream=stream)
        else:
            # No RAG context, use standard prompt with Aura Vision identity
            # Check if memory RAG was attempted but found no useful information
            no_useful_memory = (memory_rag_results is not None and len(memory_rag_results) == 0) or (memory_rag_candidates and len(memory_rag_candidates) > 0 and not memory_rag_results)
            memory_note = ""
            if no_useful_memory:
                memory_note = "\n⚠️ IMPORTANT: No useful information was found in conversation memory (only questions were found, no actual answers). DO NOT make up or guess information. If you don't have reliable information about what was asked, say so clearly rather than providing generic or speculative responses.\n\n"
            
            if is_instruction_request:
                system_content = (
                    f"{combined_context}\n\n"
                    "You are Aura Vision, an AI agent created by Ledger AI Quantum Corporation. "
                    "You act as a proactive AI agent guiding users to better outcomes through gentle guidance.\n\n"
                    f"{memory_note}"
                    "Provide a clear, step-by-step response (numbered steps). Keep each step concise and actionable. "
                    "Be conversational and friendly, like Siri or Alexa."
                )
            else:
                # Add warning if memory RAG failed or no useful information found
                memory_warning = ""
                if memory_rag_failed:
                    memory_warning = "\n\n⚠️ IMPORTANT: Conversation memory context is unavailable (memory container may be rebuilding index). " \
                                   "Only provide information you are certain about from the provided context. " \
                                   "Do NOT make up or guess information about people, places, or facts. " \
                                   "If you don't have reliable information, say so rather than speculating.\n\n"
                elif no_useful_memory:
                    memory_warning = "\n\n⚠️ IMPORTANT: No useful information was found in conversation memory (only questions were found, no actual answers). " \
                                   "DO NOT make up or guess information. If you don't have reliable information about what was asked, " \
                                   "say so clearly (e.g., 'I don't have that information in my memory') rather than providing generic or speculative responses.\n\n"
                
                system_content = (
                    f"{combined_context}\n\n"
                    "You are Aura Vision, an AI agent created by Ledger AI Quantum Corporation. "
                    "You act as a proactive AI agent guiding users to better outcomes through gentle guidance.\n\n"
                    f"{memory_warning}"
                    "IMPORTANT: Use the conversation memory provided above to answer the user's question. "
                    "If the memory contains relevant information, provide that information in your response. "
                    "If you notice a misspelling or typo, briefly acknowledge it but still answer the actual question asked.\n\n"
                    "Keep your response short and conversational, like Siri or Alexa (2-3 sentences typically). "
                    "Be friendly, helpful, and concise. Avoid lengthy explanations unless specifically requested."
                )
        
        # When only memory context (no RAG), use separate user message
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
        
            # Reduced debug logging for performance
        
        # If streaming and we used LLM scoring, yield filler phrase first
        if stream and needs_filler_phrase:
            filler_phrase = get_filler_phrase()
            print(f"[Generic] 💭 Yielding filler phrase before response: '{filler_phrase}'")
            
            # Create wrapper generator that yields filler phrase first, then LLM response
            def response_with_filler():
                # Yield filler phrase with sentence tags
                yield "<sentence_start>\n"
                for word in filler_phrase.split():
                    yield f"{word} "
                yield "<sentence_end>\n"
                
                # Then yield from actual LLM response
                llm_response = llm_chat_simple(messages, max_tokens=MAX_TOKENS_RAG_MODE, stream=True)
                
                # If debug mode is enabled, filter out reasoning and only stream the answer
                if SHOW_REASONING_DEBUG:
                    buffer = ""
                    in_answer = False
                    answer_started = False
                    
                    for chunk in llm_response:
                        buffer += chunk
                        
                        # Check if we've reached the answer section
                        if "---ANSWER---" in buffer and not in_answer:
                            # Extract everything before answer for logging
                            answer_marker = buffer.find("---ANSWER---")
                            reasoning_text = buffer[:answer_marker].strip()
                            if reasoning_text:
                                print(f"\n{'='*80}")
                                print(f"[Generic] 🔍 [LLM Reasoning Debug] FULL REASONING OUTPUT:")
                                print(f"{'='*80}")
                                print(reasoning_text)
                                print(f"{'='*80}\n")
                            # Start streaming from after the marker
                            in_answer = True
                            answer_started = True
                            # Yield everything after ---ANSWER---
                            answer_start_pos = buffer.find("---ANSWER---") + len("---ANSWER---")
                            remaining = buffer[answer_start_pos:].lstrip()
                            if remaining and "---END ANSWER---" not in remaining:
                                yield remaining
                            buffer = ""  # Clear buffer after extracting reasoning
                        
                        # If we're in answer section, check for end marker
                        if in_answer:
                            if "---END ANSWER---" in buffer:
                                # Extract answer before end marker
                                end_pos = buffer.find("---END ANSWER---")
                                answer_text = buffer[:end_pos].strip()
                                if answer_text:
                                    yield answer_text
                                break
                            elif answer_started:
                                # Stream chunks normally in answer section
                                yield chunk
                        # If not in answer yet, just buffer (don't yield)
                    
                    # If we never found answer marker, log full response and yield it
                    if not in_answer and buffer:
                        print(f"\n{'='*80}")
                        print(f"[Generic] 🔍 [LLM Reasoning Debug] FULL RESPONSE (no answer marker found):")
                        print(f"{'='*80}")
                        print(buffer)
                        print(f"{'='*80}\n")
                        # Yield full response as fallback
                        for char in buffer:
                            yield char
                else:
                    # Not debug mode, stream normally
                    for chunk in llm_response:
                        yield chunk
            
            return response_with_filler()
        
        # For streaming with debug mode, filter out reasoning
        if stream and SHOW_REASONING_DEBUG:
            def filter_reasoning():
                buffer = ""
                in_answer = False
                answer_started = False
                
                llm_response = llm_chat_simple(messages, max_tokens=MAX_TOKENS_RAG_MODE, stream=True)
                
                for chunk in llm_response:
                    buffer += chunk
                    
                    # Check if we've reached the answer section
                    if "---ANSWER---" in buffer and not in_answer:
                        # Extract everything before answer for logging
                        answer_marker = buffer.find("---ANSWER---")
                        reasoning_text = buffer[:answer_marker].strip()
                        if reasoning_text:
                            print(f"\n{'='*80}")
                            print(f"[Generic] 🔍 [LLM Reasoning Debug] FULL REASONING OUTPUT:")
                            print(f"{'='*80}")
                            print(reasoning_text)
                            print(f"{'='*80}\n")
                        # Start streaming from after the marker
                        in_answer = True
                        answer_started = True
                        # Yield everything after ---ANSWER---
                        answer_start_pos = buffer.find("---ANSWER---") + len("---ANSWER---")
                        remaining = buffer[answer_start_pos:].lstrip()
                        if remaining and "---END ANSWER---" not in remaining:
                            yield remaining
                        buffer = ""  # Clear buffer after extracting reasoning
                    
                    # If we're in answer section, check for end marker
                    if in_answer:
                        if "---END ANSWER---" in buffer:
                            # Extract answer before end marker
                            end_pos = buffer.find("---END ANSWER---")
                            answer_text = buffer[:end_pos].strip()
                            if answer_text:
                                yield answer_text
                            break
                        elif answer_started:
                            # Stream chunks normally in answer section
                            yield chunk
                    # If not in answer yet, just buffer (don't yield)
                
                # If we never found answer marker, log full response and yield it
                if not in_answer and buffer:
                    print(f"\n{'='*80}")
                    print(f"[Generic] 🔍 [LLM Reasoning Debug] FULL RESPONSE (no answer marker found):")
                    print(f"{'='*80}")
                    print(buffer)
                    print(f"{'='*80}\n")
                    # Yield full response as fallback
                    for char in buffer:
                        yield char
            
            return filter_reasoning()
        
        # Don't wrap the iterator - let base_container's debug_iterator handle logging
        # The base class already wraps it with debug logging
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
        system_prompt += (
            f"\n\nConversation memory you can reference:\n{memory_context}\n\n"
            "IMPORTANT: Use the conversation memory above to answer the user's question. "
            "If the memory contains relevant information, provide that information in your response. "
            "If you notice a misspelling or typo, briefly acknowledge it but still answer the actual question asked."
        )
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
    
    # Reduced debug logging for performance

    # Use standard max_tokens - matches LLM_NUM_PREDICT_DEFAULT
    # Don't wrap the iterator - let base_container's debug_iterator handle logging
    # The base class already wraps it with debug logging
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
    print(f"[Generic] 📝 FULL TRANSCRIBED QUERY: '{prompt}'")  # Log full query for debugging
    
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
            # Check if RAG will be used BEFORE processing (to play filler phrase during RAG)
            will_use_rag = False
            if RAG_MODE in ("CPU", "GPU"):
                try:
                    # Quick check: will document RAG be used?
                    client = get_rag_client()
                    if client:
                        has_doc_content = client.quick_content_match(prompt)
                        if has_doc_content:
                            will_use_rag = True
                            print(f"[Generic] ✅ Document RAG will be used - prefiltering confirmed match")
                    
                    # Quick check: will memory RAG be used?
                    memory_container_url = os.environ.get('MEMORY_CONTAINER_URL', 'http://localhost:11438')
                    try:
                        quick_match_response = requests.post(
                            f"{memory_container_url}/rag/quick-match",
                            json={"query": prompt},
                            timeout=0.5
                        )
                        if quick_match_response and quick_match_response.status_code == 200:
                            has_memory_content = quick_match_response.json().get('has_match', False)
                            if has_memory_content:
                                will_use_rag = True
                                print(f"[Generic] ✅ Memory RAG will be used - prefiltering confirmed match")
                    except requests.exceptions.Timeout:
                        pass  # Timeout means we'll skip memory RAG, no filler needed
                    except Exception as e:
                        print(f"[Generic] ⚠️ Memory RAG quick-match check failed: {e}")
                except Exception as e:
                    print(f"[Generic] ⚠️ RAG pre-check failed: {e}")
            
            # If RAG will be used, yield filler phrase first (RAG processing happens during playback)
            if will_use_rag:
                filler_phrase = get_filler_phrase()
                print(f"[Generic] 💭 Yielding filler phrase before RAG processing: '{filler_phrase}'")
                # Yield filler phrase with proper sentence tags - must be complete before LLM response
                yield "<sentence_start>\n"
                words = filler_phrase.split()
                for i, word in enumerate(words):
                    if i < len(words) - 1:
                        yield f"{word} "
                    else:
                        yield f"{word}"
                yield "\n<sentence_end>\n"
                # Small delay to ensure filler phrase is fully processed before LLM response starts
                time.sleep(0.1)  # 100ms delay to ensure TTS starts processing filler phrase
            
            # Use streaming mode to get tokens as they're generated, with memory context
            result = handle_conversation(prompt, session_id, memory_context=memory_context, stream=True)
            
            # Check if result is a generator (streaming)
            if hasattr(result, '__iter__') and not isinstance(result, str):
                # Reduced debug logging for performance
                normalized_chunks = _normalize_stream_chunks(result)
                word_stream = _word_stream_from_chunks(normalized_chunks)
                sentence_stream = _sentence_tag_stream(word_stream)
                token_count = 0
                try:
                    first_token = next(sentence_stream)
                    token_count += 1
                    # Yield the first token
                    if not (first_token.startswith('<') and first_token.endswith('>')):
                        full_response_text += first_token
                    yield f"{first_token}\n"
                    # Continue with rest
                    for token in sentence_stream:
                        token_count += 1
                        # Accumulate tokens for memory storage (skip control tags)
                        if not (token.startswith('<') and token.endswith('>')):
                            full_response_text += token
                        yield f"{token}\n"
                except StopIteration:
                    # Yield empty sentence tags so speaker knows stream ended
                    yield "<sentence_start>\n"
                    yield "<sentence_end>\n"
                except Exception as e:
                    print(f"[Generic] ⚠️ ERROR iterating sentence_stream: {e}")
                    import traceback
                    traceback.print_exc()
                    # Yield empty sentence tags so speaker knows stream ended
                    yield "<sentence_start>\n"
                    yield "<sentence_end>\n"
                
                if token_count == 0:
                    print(f"[Generic] ⚠️ WARNING: No tokens yielded from sentence_stream!")
                
                print(f"[Generic] ✅ Streamed response complete (yielded {token_count} tokens)")
                
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
    Matches the working version from commit d4a5c540a1a3da07a7ea5a2403155adf0d7e79e7
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
    Matches the working version from commit d4a5c540a1a3da07a7ea5a2403155adf0d7e79e7
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
        
        # Detect sentence endings based on punctuation (for immediate TTS processing)
        # Check if word ends with sentence-ending punctuation
        word_ends_with_punct = False
        sentence_ending_punct = None
        for punct in SENTENCE_ENDINGS:
            if word_to_yield.rstrip().endswith(punct):
                word_ends_with_punct = True
                sentence_ending_punct = punct
                break
        
        # If sentence ending detected, close current sentence and start new one
        # This allows TTS to start processing first sentence while LLM generates subsequent ones
        if word_ends_with_punct and sentence_open:
            # Close current sentence
                yield "<sentence_end>"
                sentence_buffer = ""
                sentence_open = False
            # Note: Don't immediately start new sentence - wait for next word
            # This prevents empty sentences if stream ends
    
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
    print(f"[Generic] 🔍 SHOW_REASONING_DEBUG = {SHOW_REASONING_DEBUG} (from environment)")
    
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

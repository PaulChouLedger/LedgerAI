# === container_rest.py — Aura Generic Conversational Container ===
# Provides general conversation with RAG-powered knowledge
#
# Dual-Model Architecture:
# - Base Model (Q4_K_M_base): Used for conversational queries (no RAG)
#   - Natural, friendly responses using general knowledge
#   - Loaded at startup for fast conversational responses
# - CoT Model (Q4_K_M-rag-cot): Used for RAG queries (with Chain of Thought)
#   - Structured extraction and reasoning for document-based queries
#   - Pre-loaded at startup to eliminate first-query latency
# - Model selection is automatic based on RAG context detection

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

# === Dual-Model Configuration ===
# Base model for conversational queries (no RAG)
BASE_MODEL_PATH = os.getenv('BASE_MODEL_PATH', "/models/Qwen2.5-1.5B-Instruct.Q4_K_M_base.gguf")
# CoT-trained model for RAG queries (with Chain of Thought reasoning)
COT_MODEL_PATH = os.getenv('COT_MODEL_PATH', "/models/Qwen2.5-1.5B-Instruct.Q4_K_M-rag-cot.gguf")

print(f"[Generic] 🔧 Dual-Model Configuration:")
print(f"[Generic]    BASE_MODEL_PATH: {BASE_MODEL_PATH}")
print(f"[Generic]    COT_MODEL_PATH: {COT_MODEL_PATH}")

# === Initialize Base Container (for conversational queries) ===
base_container = BaseLLMContainer(
    service_name="aura-llm-generic-base",
    default_model_path=BASE_MODEL_PATH
)

# === Initialize CoT Container (for RAG queries - pre-loaded at startup) ===
cot_container = BaseLLMContainer(
    service_name="aura-llm-generic-cot",
    default_model_path=COT_MODEL_PATH
)

# Override default parameters for base container (conversational)
base_container.LLM_NUM_PREDICT_DEFAULT = 800  # Increased for comprehensive responses
base_container.SIMPLE_N_CTX = 8192  # Match training MAX_SEQ_LENGTH (was 2048, causing truncation)
base_container.N_BATCH = 256  # Reduced for faster generation
base_container.SIMPLE_CHAT_FORMAT = os.getenv('SIMPLE_CHAT_FORMAT', 'chatml')

# Override default parameters for CoT container (RAG queries)
cot_container.LLM_NUM_PREDICT_DEFAULT = 2048  # Higher for CoT reasoning + final answer
cot_container.SIMPLE_N_CTX = 8192  # Match training MAX_SEQ_LENGTH (was 2048, causing truncation)
cot_container.N_BATCH = 256  # Same batch size
cot_container.SIMPLE_CHAT_FORMAT = os.getenv('SIMPLE_CHAT_FORMAT', 'chatml')

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
MAX_TOKENS_RAG_MODE = 250  # Max tokens when using RAG context (enforces concise 2-3 sentence answers)
MAX_TOKENS_RAG_MODE_LIST = 800  # Increased for CoT reasoning + final answer (reasoning ~400-500 tokens, answer ~100-200 tokens)
MAX_TOKENS_DIRECT_MODE = 600  # Max tokens for direct conversation (allows longer responses including lists)
MAX_TOKENS_DIRECT_MODE_LIST = 800  # Increased for CoT reasoning + final answer

# === Debug Mode: Show LLM Reasoning ===
# Set SHOW_REASONING_DEBUG=true to make LLM show its reasoning step-by-step in the output (visible chain-of-thought)
# 
# IMPORTANT: We CANNOT see the LLM's internal reasoning (it's a black box - we only see input/output).
# However, we CAN make the LLM show its reasoning PROCESS in the OUTPUT using chain-of-thought prompting.
# This will make the LLM explicitly show: what it's analyzing, what it finds in each chunk, etc.
#
# Note: Using Qwen2.5-1.5B - optimized for instruction following and conversational tasks.
SHOW_REASONING_DEBUG = os.environ.get('SHOW_REASONING_DEBUG', 'false').lower() == 'true'
if SHOW_REASONING_DEBUG:
    print(f"[Generic] 🔍 SHOW_REASONING_DEBUG is ENABLED - Qwen2.5 will show step-by-step reasoning in response")

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

def llm_chat_simple(messages, max_tokens=None, temperature=None, stream=False, use_cot_model=False, **kwargs):
    """
    Wrapper for LLM chat completion with dual-model support.
    
    Args:
        messages: Chat messages
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        stream: Whether to stream responses
        use_cot_model: If True, use CoT-trained model for RAG queries; otherwise use base model
        **kwargs: Additional arguments
    
    Returns:
        LLM response (string or iterator if streaming)
    """
    # Select container based on model type
    if use_cot_model:
        container = cot_container
        model_name = "CoT (RAG)"
        # CoT model should be pre-loaded at startup (check if it failed to load)
        if not container._model_loaded or container.llm_simple is None:
            print(f"[Generic] ⚠️ CoT model not loaded! Attempting emergency load: {COT_MODEL_PATH}")
            from llama_cpp import Llama
            n_gpu_layers = -1  # Offload all layers to GPU
            container.model_path = COT_MODEL_PATH
            container.llm_simple = Llama(
                model_path=COT_MODEL_PATH,
                n_ctx=container.SIMPLE_N_CTX,
                n_threads=1,  # Use 1 thread for deterministic output (temperature=0 alone isn't enough)
                n_batch=container.N_BATCH,
                n_gpu_layers=n_gpu_layers,
                cache_prompt=CACHE_PROMPT,
                chat_format=container.SIMPLE_CHAT_FORMAT,
                use_mlock=True,
                use_mmap=True,
                verbose=False
            )
            container._model_loaded = True
            print(f"[Generic] ✅ CoT model emergency-loaded: {COT_MODEL_PATH}")
    else:
        container = base_container
        model_name = "Base (Conversational)"
    
    # Check if model is loaded
    if not container._model_loaded or container.llm_simple is None:
        print(f"[Generic] ⚠️ ERROR: {model_name} model not loaded! _model_loaded={container._model_loaded}")
        if stream:
            return iter([])
        return ""
    
    # Call the selected container's chat method
    result = container.llm_chat_simple(messages, max_tokens, temperature, stream, **kwargs)
    return result


def hard_stop_after_first_final_answer(stream_iter):
    """
    Hard-stop a streaming LLM iterator once we have produced the FIRST complete sentence
    after the first occurrence of 'FINAL ANSWER:' (case-insensitive, handles variations).
    
    This prevents the model from continuing and emitting a second/hallucinated answer
    (or re-starting REASONING) after it already produced a correct final answer.
    
    Works on a chunk stream (strings or dict chunks) as returned by llm_chat_simple().
    """
    import re
    seen_final_answer_marker = False
    final_answer_text = ""
    accumulated_text = ""  # Track all text seen so far for pattern matching

    def _extract_text_from_chunk(chunk) -> str:
        # Match _normalize_stream_chunks behavior, but keep it local to avoid import order issues.
        if isinstance(chunk, dict):
            if 'choices' in chunk and len(chunk['choices']) > 0:
                delta = chunk['choices'][0].get('delta', {})
                return delta.get('content', '') or ''
            if 'content' in chunk:
                return chunk.get('content', '') or ''
            return ''
        if isinstance(chunk, str):
            return chunk
        return str(chunk)

    # Case-insensitive pattern to match "FINAL ANSWER:" variations (with or without colon after "ANSWER")
    # Examples: "FINAL ANSWER:", "Final Answer:", "Final ANSWER:", "FINAL ANSWER" (no colon)
    final_answer_pattern = re.compile(r'final\s+answer\s*:?', re.IGNORECASE)

    for chunk in stream_iter:
        text = _extract_text_from_chunk(chunk)
        accumulated_text += text if text else ""
        
        if not seen_final_answer_marker:
            # Look for FIRST "FINAL ANSWER" marker (case-insensitive, handles variations)
            if text:
                # Check if this chunk contains the marker
                match = final_answer_pattern.search(accumulated_text)
                if match:
                    seen_final_answer_marker = True
                    # Extract text after the marker
                    marker_end = match.end()
                    after = accumulated_text[marker_end:].lstrip()
                    final_answer_text += after
                    print(f"[Generic] 🛑 [Hard Stop] FINAL ANSWER marker detected at position {marker_end}, starting collection")
        else:
            if text:
                final_answer_text += text

        # Always yield the original chunk unchanged.
        yield chunk

        # If we're in FINAL ANSWER collection, stop after the first complete sentence.
        # Heuristic: end at first '.', '?', or '!' after we've accumulated some non-trivial content.
        if seen_final_answer_marker:
            stripped = final_answer_text.strip()
            if len(stripped) >= 8 and any(p in stripped for p in (".", "?", "!")):
                # Stop at the earliest sentence-ending punctuation.
                print(f"[Generic] 🛑 [Hard Stop] Stopping stream after first complete sentence: '{stripped[:100]}...'")
                break

# === RAG Chunk Filtering ===
# Cache for sentence transformer model (lazy loading)
_semantic_model = None

def _get_semantic_model():
    """Lazy load sentence transformer model for semantic similarity"""
    global _semantic_model
    if _semantic_model is None:
        try:
            from sentence_transformers import SentenceTransformer
            _semantic_model = SentenceTransformer('all-distilroberta-v1')
            print("[Generic] ✅ Loaded semantic model for chunk filtering")
        except Exception as e:
            print(f"[Generic] ⚠️ Failed to load semantic model: {e}, using substring matching only")
            _semantic_model = False  # Mark as failed to avoid retrying
    return _semantic_model if _semantic_model is not False else None

def extract_relevant_substrings(text: str, query: str, entity_names: list = None, use_semantic: bool = True) -> str:
    """
    Extract sentences/paragraphs from text that are relevant to the query using:
    1. Substring matching (direct matches for query terms and entity names)
    2. Semantic similarity (related/relevant information)
    
    This is more inclusive - includes both direct matches and semantically related content.
    The LLM will do final scoring to determine what to include in the answer.
    
    Args:
        text: Full chunk text to filter
        query: Original user query
        entity_names: List of entity names to look for (e.g., ["LedgerAI", "Ledger AI"])
        use_semantic: Whether to use semantic similarity (default: True)
    
    Returns:
        Filtered text containing relevant sentences/paragraphs (both direct matches and semantically related)
    """
    import numpy as np
    
    if not text or not query:
        return text
    
    # Extract key query terms (co-founder, founder, etc.)
    query_lower = query.lower()
    query_terms = []
    
    # Look for relationship terms
    relationship_patterns = [
        r'\bco-?founder\w*\b',
        r'\bfounder\w*\b',
        r'\bco-?found\w*\b',
        r'\bemployee\w*\b',
        r'\bmanager\w*\b',
        r'\bdirector\w*\b',
        r'\bofficer\w*\b',
        r'\bceo\b',
        r'\bcfo\b',
        r'\bcto\b',
        r'\bambassador\w*\b',
    ]
    
    for pattern in relationship_patterns:
        matches = re.findall(pattern, query_lower, re.IGNORECASE)
        query_terms.extend([m.lower() for m in matches])
    
    # If no relationship terms found, extract important words from query
    if not query_terms:
        # Extract words that are likely important (3+ chars, not common words)
        common_words = {'the', 'are', 'who', 'what', 'where', 'when', 'how', 'is', 'at', 'of', 'and', 'or'}
        words = re.findall(r'\b\w{3,}\b', query_lower)
        query_terms = [w for w in words if w not in common_words][:5]  # Top 5 important words
    
    # Extract entity names from query if not provided
    if entity_names is None:
        entity_names = []
        # Look for capitalized phrases (likely entity names)
        entity_patterns = [
            r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b',  # "Ledger AI", "Company Name"
            r'\b[A-Z][a-z]+[A-Z][a-z]+\b',  # "LedgerAI" (camelCase)
        ]
        for pattern in entity_patterns:
            matches = re.findall(pattern, query)
            entity_names.extend(matches)
    
    # Normalize entity names (remove duplicates, handle variations)
    entity_names = list(set([e.lower() for e in entity_names if len(e) > 2]))
    
    # If no entity names found, try to extract from common patterns
    if not entity_names:
        # Look for "of X" or "at X" patterns
        of_pattern = r'\bof\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b'
        at_pattern = r'\bat\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b'
        matches = re.findall(of_pattern, query, re.IGNORECASE) + re.findall(at_pattern, query, re.IGNORECASE)
        entity_names = [m.lower() for m in matches if len(m) > 2]
    
    # Split text into sentences and paragraphs
    sentences = re.split(r'([.!?]\s+)', text)
    # Recombine sentences with their punctuation
    sentences = [sentences[i] + (sentences[i+1] if i+1 < len(sentences) else '') 
                 for i in range(0, len(sentences), 2)]
    sentences = [s.strip() for s in sentences if s.strip()]
    
    # Also try paragraph-level extraction (split by double newlines)
    paragraphs = re.split(r'\n\n+', text)
    paragraphs = [p.strip() for p in paragraphs if p.strip()]
    
    # Collect all candidate parts (sentences and paragraphs)
    candidates = []
    for s in sentences:
        if len(s) > 10:  # Only consider substantial sentences
            candidates.append(('sentence', s))
    for p in paragraphs:
        if len(p) > 20:  # Only consider substantial paragraphs
            candidates.append(('paragraph', p))
    
    if not candidates:
        return text
    
    # Score each candidate using both substring and semantic matching
    scored_candidates = []
    
    # Get semantic model if available
    semantic_model = _get_semantic_model() if use_semantic else None
    
    # Compute query embedding once if using semantic matching
    query_embedding = None
    if semantic_model:
        try:
            query_embedding = semantic_model.encode([query], convert_to_numpy=True)[0]
        except Exception as e:
            print(f"[Generic] ⚠️ Semantic encoding failed: {e}, using substring matching only")
            semantic_model = None
    
    for candidate_type, candidate_text in candidates:
        candidate_lower = candidate_text.lower()
        
        # Substring matching score
        substring_score = 0.0
        has_query_term = False
        has_entity = False
        
        if query_terms:
            for term in query_terms:
                if term in candidate_lower:
                    substring_score += 0.3
                    has_query_term = True
        
        if entity_names:
            for entity in entity_names:
                if entity in candidate_lower:
                    substring_score += 0.4
                    has_entity = True
        
        # Bonus for having both query term and entity (direct match)
        if has_query_term and has_entity:
            substring_score += 0.3
        
        # Semantic similarity score
        semantic_score = 0.0
        if semantic_model and query_embedding is not None:
            try:
                candidate_embedding = semantic_model.encode([candidate_text], convert_to_numpy=True)[0]
                # Cosine similarity
                similarity = np.dot(query_embedding, candidate_embedding) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(candidate_embedding)
                )
                semantic_score = max(0.0, similarity)  # Normalize to 0-1
            except Exception as e:
                # If semantic encoding fails, just use substring score
                pass
        
        # Combined score (weighted: 60% substring, 40% semantic)
        # This ensures we get both direct matches and related content
        combined_score = (substring_score * 0.6) + (semantic_score * 0.4)
        
        scored_candidates.append({
            'text': candidate_text,
            'type': candidate_type,
            'substring_score': substring_score,
            'semantic_score': semantic_score,
            'combined_score': combined_score
        })
    
    # Sort by combined score (highest first)
    scored_candidates.sort(key=lambda x: x['combined_score'], reverse=True)
    
    # Include candidates that meet threshold (be inclusive - let LLM do final filtering)
    # Lower threshold to include both direct matches and semantically related content
    threshold = 0.15  # Low threshold to be inclusive
    relevant_parts = []
    
    for candidate in scored_candidates:
        if candidate['combined_score'] >= threshold:
            # Avoid duplicates
            if candidate['text'] not in relevant_parts:
                relevant_parts.append(candidate['text'])
    
    # If we found relevant parts, join them; otherwise return original
    if relevant_parts:
        filtered_text = ' '.join(relevant_parts)
        # Ensure we have reasonable amount of context
        if len(filtered_text) < 50:
            # Too short, return original to avoid losing all context
            return text
        return filtered_text
    else:
        # No matches found - return original text (better to have some context than none)
        return text

# === Filler Phrases ===
def get_filler_phrase() -> str:
    """
    Extract only sentences/paragraphs from text that contain both query terms and entity names.
    This helps filter out irrelevant information before passing to LLM.
    
    Args:
        text: Full chunk text to filter
        query: Original user query
        entity_names: List of entity names to look for (e.g., ["LedgerAI", "Ledger AI"])
    
    Returns:
        Filtered text containing only relevant sentences/paragraphs
    """
    if not text or not query:
        return text
    
    # Extract key query terms (co-founder, founder, etc.)
    query_lower = query.lower()
    query_terms = []
    
    # Look for relationship terms
    relationship_patterns = [
        r'\bco-?founder\w*\b',
        r'\bfounder\w*\b',
        r'\bco-?found\w*\b',
        r'\bemployee\w*\b',
        r'\bmanager\w*\b',
        r'\bdirector\w*\b',
        r'\bofficer\w*\b',
        r'\bceo\b',
        r'\bcfo\b',
        r'\bcto\b',
        r'\bambassador\w*\b',
    ]
    
    for pattern in relationship_patterns:
        matches = re.findall(pattern, query_lower, re.IGNORECASE)
        query_terms.extend([m.lower() for m in matches])
    
    # If no relationship terms found, extract important words from query
    if not query_terms:
        # Extract words that are likely important (3+ chars, not common words)
        common_words = {'the', 'are', 'who', 'what', 'where', 'when', 'how', 'is', 'at', 'of', 'and', 'or'}
        words = re.findall(r'\b\w{3,}\b', query_lower)
        query_terms = [w for w in words if w not in common_words][:5]  # Top 5 important words
    
    # Extract entity names from query if not provided
    if entity_names is None:
        entity_names = []
        # Look for capitalized phrases (likely entity names)
        entity_patterns = [
            r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b',  # "Ledger AI", "Company Name"
            r'\b[A-Z]{2,}\b',  # "AI", "LLC", etc.
        ]
        for pattern in entity_patterns:
            matches = re.findall(pattern, query)
            entity_names.extend(matches)
    
    # Normalize entity names (remove duplicates, handle variations)
    entity_names = list(set([e.lower() for e in entity_names if len(e) > 2]))
    
    # If no entity names found, try to extract from common patterns
    if not entity_names:
        # Look for "of X" or "at X" patterns
        of_pattern = r'\bof\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b'
        at_pattern = r'\bat\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b'
        matches = re.findall(of_pattern, query, re.IGNORECASE) + re.findall(at_pattern, query, re.IGNORECASE)
        entity_names = [m.lower() for m in matches if len(m) > 2]
    
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

# === Secondary Filler (CoT/Filtering Phase) ===
def get_cot_filler_phrase() -> str:
    """Get a short filler phrase to play when CoT mode starts (before FINAL ANSWER filtering)."""
    import random
    filler_phrases = [
        "Okay—processing that now.",
        "Got it. One more second while I extract the answer.",
        "Alright—pulling the exact answer from the context.",
        "One moment—finalizing the answer.",
    ]
    return random.choice(filler_phrases)

# === Conversational Logic ===
def validate_query(prompt: str) -> tuple:
    """
    Validate query to detect malformed/incomplete transcriptions.
    
    Returns:
        (is_valid, error_message)
        - is_valid: True if query is valid, False if it's malformed
        - error_message: Clarification message if invalid, None if valid
    """
    if not prompt or len(prompt.strip()) < 3:
        return False, "I'm sorry, I didn't understand your question. Can you repeat it?"
    
    prompt_lower = prompt.lower().strip()
    
    # Normalize contractions BEFORE checking for question words
    # This ensures "what's" is recognized as "what is" for validation
    contractions_map = {
        "what's": "what is",
        "who's": "who is",
        "where's": "where is",
        "when's": "when is",
        "why's": "why is",
        "how's": "how is",
        "which's": "which is",
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
    for contraction, expansion in contractions_map.items():
        prompt_lower = prompt_lower.replace(contraction, expansion)
    
    # Check for queries starting with fragments (common transcription errors)
    # These indicate the question word was cut off: "or the co-founders" instead of "who are the co-founders"
    #
    # NOTE: Some valid questions legitimately start with a preposition, e.g.:
    # - "on what chain is the ledger token on?"
    # - "in which year was X founded?"
    # - "of what does it consist?"
    #
    # We only treat a leading preposition as a fragment if it's NOT immediately followed by a question word.
    fragment_starters = ['or ', 'and ', 'the ', 'a ', 'an ', 'at ', 'to ', 'for ', 'with ', 'by ']
    preposition_starters = ['of ', 'in ', 'on ']
    starts_with_fragment = any(prompt_lower.startswith(frag) for frag in fragment_starters)
    starts_with_preposition = any(prompt_lower.startswith(prep) for prep in preposition_starters)
    if starts_with_fragment:
        print(f"[Generic] ⚠️ Query validation failed: starts with fragment - '{prompt[:50]}...'")
        return False, "I'm sorry, I didn't understand your question. Can you repeat it?"
    if starts_with_preposition:
        # Allow "on what ...", "in which ...", "of what ..." etc.
        second_word = prompt_lower.split(maxsplit=2)[1] if len(prompt_lower.split()) >= 2 else ""
        allowed_after_preposition = {
            "who", "what", "where", "when", "why", "how", "which", "whose", "whom",
            "do", "does", "did", "is", "are", "was", "were", "can", "could", "will",
            "would", "should", "may", "might",
        }
        if second_word not in allowed_after_preposition:
            print(f"[Generic] ⚠️ Query validation failed: starts with fragment-like preposition - '{prompt[:50]}...'")
            return False, "I'm sorry, I didn't understand your question. Can you repeat it?"
    
    # Check for queries that are just noun phrases without question words
    # Valid question words
    question_words = ['who', 'what', 'where', 'when', 'why', 'how', 'which', 'whose', 'whom', 
                      'do', 'does', 'did', 'is', 'are', 'was', 'were', 'can', 'could', 'will', 
                      'would', 'should', 'may', 'might', 'tell', 'explain', 'describe', 'list', 
                      'show', 'give', 'find', 'search']
    
    # Check if query contains any question words
    # After contraction normalization, check for question words at start or with word boundaries
    # Use word boundary regex to catch question words at start, middle, or end
    import re
    has_question_word = any(
        re.search(r'\b' + re.escape(qw) + r'\b', prompt_lower)  # Word boundary match
        for qw in question_words
    )
    
    # Check for imperative patterns (valid commands)
    imperative_patterns = ['tell me', 'show me', 'give me', 'find', 'search', 'list', 'explain']
    has_imperative = any(prompt_lower.startswith(imp) for imp in imperative_patterns)
    
    # If no question word and no imperative, and it's a short phrase, likely malformed
    if not has_question_word and not has_imperative:
        # Allow very short conversational phrases (greetings, etc.)
        conversational_short = any(phrase in prompt_lower for phrase in [
            'hello', 'hi', 'hey', 'thanks', 'thank you', 'bye', 'goodbye', 'ok', 'okay', 'yes', 'no'
        ])
        if not conversational_short and len(prompt.split()) <= 5:
            # Short phrase without question word or imperative - likely fragment
            print(f"[Generic] ⚠️ Query validation failed: no question word/imperative, short phrase - '{prompt[:50]}...'")
            return False, "I'm sorry, I didn't understand your question. Can you repeat it?"
    
    # Check for queries that are just punctuation or very short
    if len(prompt.strip()) <= 2 or prompt.strip() in ['.', '?', '!', ',', ';', ':']:
        return False, "I'm sorry, I didn't understand your question. Can you repeat it?"

    # Catch incomplete questions that end in a dangling preposition (common STT truncation)
    # Examples:
    # - "Did Bob Parella work for?"
    # - "Who did he work with?"
    dangling_patterns = [
        r'\b(work|worked|working)\s+for\??\s*$',   # "work for?"
        r'\b(go|went|going)\s+to\??\s*$',          # "go to?"
        r'\b(live|lived|living)\s+in\??\s*$',      # "live in?"
        r'\b(born)\s+in\??\s*$',                   # "born in?"
        r'\b(graduate|graduated)\s+from\??\s*$',   # "graduated from?"
        r'\b(about)\??\s*$',                       # "... about?"
    ]
    if any(re.search(pat, prompt_lower) for pat in dangling_patterns):
        print(f"[Generic] ⚠️ Query validation failed: dangling/incomplete ending - '{prompt[:60]}...'")
        return False, "I'm sorry, I didn't understand your question. Can you repeat it?"
    
    return True, None

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
    
    # Validate query before processing (catch malformed transcriptions)
    is_valid, error_message = validate_query(prompt)
    if not is_valid:
        print(f"[Generic] 🚫 Query validation failed - returning clarification message")
        if stream:
            def clarification_response():
                yield "<sentence_start>\n"
                yield f"{error_message}\n"
                yield "<sentence_end>\n"
            return clarification_response()
        else:
            return error_message
    
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
        
        # No bypass phrases - all queries use quick_content_match to determine RAG usage
        # quick_content_match extracts only key terms (names, important nouns) and skips everyday words
        bypass_hit = None
        is_personal_query = False
        
        # Determine if query is conversational (for response formatting only, not RAG triggering)
        conversational_phrases = [
            'thank you', 'thanks', 'thank', 'thanks a lot', 'thank you very much',
            'goodbye', 'bye', 'see you', 'see ya', 'farewell',
            'you\'re welcome', 'no problem', 'my pleasure', 'anytime',
            'hello', 'hi', 'hey', 'greetings',
            'how are you', 'how\'s it going', 'how\'s everything', 'how do you do',
            'ok', 'okay', 'sure', 'alright', 'got it', 'understood',
            'yes', 'yeah', 'yep', 'no', 'nope',
            'please', 'excuse me', 'sorry', 'pardon'
        ]
        is_conversational = any(phrase in normalized_prompt for phrase in conversational_phrases)
        
        # SIMPLIFIED: Use quick_content_match as primary RAG trigger (fast substring/fuzzy match)
        # This is more reliable than pattern matching - if content exists, use RAG
        rag_client = None
        rag_context = ""
        rag_results = []
        needs_filler_phrase = False  # Flag to indicate if we should yield filler phrase before LLM response
        memory_rag_results = []  # Results from memory container
        memory_rag_failed = False  # Track if memory RAG failed (timeout, error, etc.)
        
        # Check if RAG will be used BEFORE doing the search (for filler phrase timing)
        # PRIMARY DECISION: Use quick_content_match first, only bypass if no content AND clear everyday phrase
        will_use_rag = False
        try:
            client = get_rag_client()
            if client:
                has_content = client.quick_content_match(prompt)
                if has_content:
                    will_use_rag = True
                    print(f"[Generic] ✅ Document RAG will be used (quick_content_match=True)")
                else:
                    print(f"[Generic] 🔍 Document RAG quick_content_match: no relevant content found")
        except Exception as e:
            print(f"[Generic] ⚠️ RAG pre-check failed: {e}")
        
        # Filler phrase is now yielded in generate_response() BEFORE calling handle_conversation()
        # This ensures it plays immediately while RAG search happens
        # Use RAG if quick_content_match found content (extracts only key terms, skips everyday words)
        # NOTE: Filler phrase is no longer yielded here - it's yielded in generate_response() before this function is called
        # Check if RAG should be used - quick_content_match is the only decision maker
        should_use_rag_for_search = False
        if RAG_MODE in ("CPU", "GPU"):
            try:
                client = get_rag_client()
                if client:
                    has_content = client.quick_content_match(prompt)
                    if has_content:
                        should_use_rag_for_search = True
                        print(f"[Generic] 🔍 quick_content_match=True - performing RAG search...")
                    else:
                        should_use_rag_for_search = False
                        print(f"[Generic] 🔍 quick_content_match=False - skipping RAG (no key terms found in documents)")
            except Exception as e:
                print(f"[Generic] ⚠️ RAG check failed in handle_conversation: {e}")
        
        if should_use_rag_for_search:
            # Detect if query is asking for "what else" or additional information
            is_followup_query = any(phrase in prompt.lower() for phrase in ['what else', 'anything else', 'more about', 'additional', 'other'])
            
            # Parallelize document RAG and memory RAG searches for better latency
            def search_document_rag():
                """Search document RAG in parallel - uses quick_content_match as the only trigger"""
                try:
                    client = get_rag_client()
                    if client:
                        has_content = client.quick_content_match(prompt)
                        if has_content:
                            print(f"[Generic] 🔍 quick_content_match=True - performing RAG search...")
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
                            print(f"[Generic] 🔍 quick_content_match=False - skipping document RAG")
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
                    question_words = {'who', 'what', 'where', 'when', 'why', 'how', 'which', 'whose', 'whom', 'do', 'does', 'did', 'is', 'are', 'was', 'were', 'can', 'could', 'will', 'would', 'should', 'may', 'might', 'remember', 'for the', 'or the'}
                    question_phrases = {'for the', 'or the', 'who are', 'who is', 'what are', 'what is', 'tell me', 'do you remember', 'remember'}
                    filtered_candidates = []
                    for candidate in memory_rag_candidates:
                        text = candidate.get('text', '').strip()
                        # Check if it's a question: starts with question word/phrase, ends with "?", or is a question-like pattern
                        is_question = False
                        text_lower = text.lower().strip()
                        
                        # Check if it ends with "?"
                        if text.endswith('?'):
                            first_word = text_lower.split()[0] if text_lower.split() else ""
                            if first_word in question_words:
                                is_question = True
                        
                        # Check if it starts with question words/phrases (even without "?")
                        if text_lower.split():
                            first_word = text_lower.split()[0]
                            first_two_words = ' '.join(text_lower.split()[:2]) if len(text_lower.split()) >= 2 else first_word
                            
                            # Check for question phrases like "for the", "or the", "who are", etc.
                            if first_two_words in question_phrases or first_word in question_words:
                                is_question = True
                        
                        # Check for question patterns: "for the X", "or the X" (common transcription errors)
                        # Also check if text contains "co-founder" or similar with "for the"/"or the" prefix (common in transcriptions)
                        if text_lower.startswith('for the ') or text_lower.startswith('or the '):
                            is_question = True
                        elif 'co-founder' in text_lower and (text_lower.startswith('for ') or text_lower.startswith('or ')):
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
                        # Additional filter: Skip very short or non-actionable statements for "how to" queries
                        if any(phrase in prompt.lower() for phrase in ['how can', 'how do', 'how to', 'how should', 'what should', 'what can']):
                            actionable_candidates = []
                            for candidate in filtered_candidates:
                                text = candidate.get('text', '').strip()
                                # Check if it contains actionable advice (has verbs like "improve", "optimize", "reduce", etc.)
                                actionable_verbs = ['improve', 'optimize', 'reduce', 'increase', 'enhance', 'implement', 'focus', 'consider', 'try', 'use', 'apply', 'develop', 'create', 'build', 'establish']
                                # Check if it's long enough to contain advice (at least 50 chars) or contains actionable verbs
                                if len(text) >= 50 or any(verb in text.lower() for verb in actionable_verbs):
                                    actionable_candidates.append(candidate)
                                else:
                                    print(f"[Generic]   [Pre-filter] ❌ Excluded non-actionable statement: '{text[:60]}...'")
                            
                            if not actionable_candidates:
                                print(f"[Generic] ⚠️ All memory RAG candidates are non-actionable statements - skipping injection for 'how to' query")
                                memory_rag_results = []
                            else:
                                # Use actionable candidates for LLM scoring
                                filtered_candidates = actionable_candidates
                        
                        # Use filtered candidates for LLM scoring (if we still have candidates)
                        if filtered_candidates and not memory_rag_results:
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
        print(f"[Generic] 🔍 Query analysis: is_personal={is_personal_query}, is_conversational={is_conversational}, should_use_rag={should_use_rag}, should_use_memory_rag={should_use_memory_rag}")
        
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
                # Also detect plural nouns that indicate lists (co-founders, founders, employees, members, etc.)
                list_indicators = ['co-founders', 'founders', 'employees', 'members', 'team', 'people', 'individuals']
                is_list_query = any(keyword in prompt.lower() for keyword in list_keywords) or any(indicator in prompt.lower() for indicator in list_indicators)
                
                # SIMPLIFIED: Trust final LLM to reason through chunks internally
                # No pre-filtering - let LLM understand query and extract valid information from all chunks
                # RAG semantic search already filtered by relevance, now LLM will internally reason through chunks
                print(f"[Generic] 📋 Using {len(rag_results)} RAG chunks - LLM will internally reason and extract valid information")
                
                # Use original RAG results, sorted by semantic similarity score
                sorted_results = sorted(rag_results, key=lambda x: x.get('score', 0), reverse=True)
                
                # Limit number of chunks to avoid token bloat (but keep more for list questions to ensure completeness)
                # Match test script: use more chunks for list queries to ensure all items are found (test script uses full context)
                max_chunks = 20 if is_list_query else 6  # Increased to 20 for list queries to match test script completeness
                sorted_results = sorted_results[:max_chunks]
                
                print(f"[Generic] 📋 Using top {len(sorted_results)} chunks (max {max_chunks}) for LLM reasoning")
                
                # Build RAG context - let LLM reason through all chunks
                # Use 3000 chars for all queries - important info can appear anywhere in a chunk
                # (e.g., education info for David Lara appears late in his bio)
                MAX_CHARS_PER_RESULT = 3000
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
                        
                        # For list queries, don't filter by entity - include ALL chunks to ensure completeness
                        # The model will scan through all chunks and extract matching items
                        # Entity filtering can cause missing items (e.g., Jorge Guinovart might be in a chunk that doesn't explicitly mention "LedgerAI")
                        if not is_list_query and any(term in prompt.lower() for term in ['co-founder', 'founder', 'employee', 'manager', 'director', 'officer']):
                            # Only filter for non-list queries - for list queries, include all chunks
                            # Extract entity names from query for basic filtering
                            entity_names = []
                            of_pattern = r'\bof\s+([A-Z][A-Za-z]*(?:\s+[A-Z][A-Za-z]*)*)\b'
                            at_pattern = r'\bat\s+([A-Z][A-Za-z]*(?:\s+[A-Z][A-Za-z]*)*)\b'
                            matches = re.findall(of_pattern, prompt) + re.findall(at_pattern, prompt)
                            entity_names.extend(matches)
                            entity_names = list(set([e for e in entity_names if len(e) > 2]))
                            
                            # Normalize for matching
                            if entity_names:
                                normalized_entities = []
                                for entity in entity_names:
                                    normalized_entities.append(entity.lower())
                                    normalized_entities.append(entity.lower().replace(' ', ''))
                                entity_names_normalized = list(set(normalized_entities))
                                
                                # Basic check: does chunk mention the entity at all?
                                text_lower = text.lower()
                                entity_mentioned = False
                                for entity_norm in entity_names_normalized:
                                    if entity_norm in text_lower:
                                        entity_mentioned = True
                                        break
                                    # Check capitalized versions
                                    if entity.replace(' ', '') in text or entity in text:
                                        entity_mentioned = True
                                        break
                                
                                if not entity_mentioned:
                                    print(f"[Generic] ⚠️ Chunk {i}: Does not mention query entity, skipping")
                                    continue
                                
                                print(f"[Generic] ✅ Chunk {i}: Mentions query entity, including full chunk ({len(text)} chars)")
                        
                        # For list queries, always include chunk (let model scan all chunks for completeness)
                        if is_list_query:
                            print(f"[Generic] ✅ Chunk {i}: Including for list query ({len(text)} chars) - model will scan for all matching items")
                        
                        # For list queries, don't truncate - we need full chunks to extract all items
                        # Match test script: use full chunks without truncation for list queries
                        # Only truncate if extremely long (let LLM handle most filtering)
                        if not is_list_query and len(text) > MAX_CHARS_PER_RESULT:
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
                        # For list queries, keep full text (match test script - no truncation for completeness)
                        # Only truncate if extremely long (over 5000 chars) to avoid token bloat
                        elif is_list_query and len(text) > 5000:
                            # Only truncate if extremely long (over 5000 chars) - try to break at sentence boundary
                            truncated = text[:5000]
                            last_period = max(
                                truncated.rfind('. '),
                                truncated.rfind('! '),
                                truncated.rfind('? ')
                            )
                            if last_period > 4500:  # Only if we can break near the end
                                text = truncated[:last_period + 1] + "..."
                            else:
                                text = truncated + "..."
                        # For list queries under 5000 chars, use full text (no truncation)
                        
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
                MAX_CHARS_PER_RESULT = 3000  # Same as document RAG - important info can appear anywhere
                
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
                            # Extract conversation info (for logging only, not included in prompt)
                            conv_id = metadata.get('conversation_id', 'unknown')
                            timestamp = metadata.get('datetime', metadata.get('timestamp', ''))
                            source = metadata.get('source', 'unknown')
                            
                            # Format memory chunk - use only the text content, no metadata wrapper
                            # This prevents LLM from including timestamps/metadata in spoken responses
                            memory_chunk = text
                            
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
                    # Note: No metadata wrapper - just the text content to prevent LLM from including it in responses
                    if rag_context:
                        rag_context = f"{rag_context}\n\n---\n\n{memory_context}"
                    else:
                        rag_context = memory_context
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
    # NOTE: Conversation memory is already included in rag_context from memory container API
    # The memory_context parameter is deprecated - memory container is the only source
    combined_context = "\n\n".join(contextual_sections).strip()
    
    if not combined_context:
        print(f"[Generic] 📝 LLM prompt: direct conversation (no RAG, no memory)")

    if combined_context:
        # Check if RAG context is present (more authoritative than general knowledge)
        has_rag_context = "Knowledge context:" in combined_context
        
        # Detect if user is asking for instructions/steps
        instruction_keywords = ['how to', 'how do i', 'steps', 'step by step', 'instructions', 'guide me', 'walk me through', 'show me how']
        is_instruction_request = any(keyword in prompt.lower() for keyword in instruction_keywords)
        
        # Detect if user is asking for a list (e.g., "who are", "list all", "what are the", "co-founders", "founders")
        list_keywords = ['who are', 'who were', 'list all', 'list the', 'what are the', 'what are', 'what were', 'name all', 'name the']
        # Also detect plural nouns that indicate lists (co-founders, founders, employees, members, etc.)
        list_indicators = ['co-founders', 'founders', 'employees', 'members', 'team', 'people', 'individuals']
        is_list_request = any(keyword in prompt.lower() for keyword in list_keywords) or any(indicator in prompt.lower() for indicator in list_indicators)
        
        # Detect if query asks for a specific set (should find ALL, not limit to TOP 3)
        specific_set_indicators = ['co-founders', 'founders', 'co-founder', 'founder', 'all the', 'all of the', 'every']
        is_specific_set_query = any(indicator in prompt.lower() for indicator in specific_set_indicators)
        
        if has_rag_context:
            # Dynamic prompt construction with Aura Vision identity
            # IMPORTANT: Include the prompt in the system message (matches working commit 1927b467c106120dd4e1231f600eccdaa5a93f08)
            if is_instruction_request:
                # Simplified instruction request handling
                # Only add question instruction if not a conversational query
                question_instruction = "" if is_conversational else "\n\nAlways end your response with a brief, natural question (do not include 'follow up' or 'follow-up' in the question text). Examples: 'Would you like more information about this?' or 'Is there anything else I can help you with?'"
                system_content = (
                    f"{combined_context}\n\n"
                    "You are Aura Vision, an AI agent created by Ledger AI Quantum Corporation. "
                    "You act as a proactive AI agent guiding users to better outcomes through gentle guidance.\n\n"
                    "CRITICAL RULES:\n"
                    "- Only provide logical, factual responses. Avoid hallucination at all costs.\n"
                    "- CRITICAL QUERY VALIDATION: Before responding, you MUST first evaluate if the query makes logical sense:\n"
                    "  1. Check if the query contains nonsensical combinations (e.g., 'recipe for rest and efforts' - 'rest and efforts' is not a real recipe name)\n"
                    "  2. Check if key terms in the query are coherent and refer to real concepts (e.g., asking for a recipe for something that doesn't exist)\n"
                    "  3. Check if the query is incomplete, unclear, or contains transcription errors\n"
                    "  4. If the query does NOT make logical sense, DO NOT force it into a response. Instead, politely ask: 'I'm not sure I understand your question. Could you please repeat or rephrase it?'\n"
                    "- CRITICAL: Before responding, check if the query is an incomplete sentence (starts with 'and', 'but', 'or', 'so', 'then', 'also', 'make sure', 'ensure', etc.). If so, ask for clarification instead of answering.\n"
                    "- IMPORTANT: Commands and instructions like 'Give me X', 'Tell me about Y', 'Show me Z' are VALID requests and should be answered normally using your general knowledge.\n"
                    "- If the user's query is unclear, nonsensical, or doesn't make logical sense, DO NOT force it into a response.\n"
                    "- Instead, politely ask the user to clarify or repeat their question. Example: 'I'm not sure I understand. Could you please rephrase your question or provide more context?'\n"
                    "- Never invent facts, names, dates, or details that aren't in the provided context.\n"
                    "- CRITICAL: DO NOT treat incomplete sentences as questions. If the query is incomplete (starts with 'and', 'but', 'or', etc.), ask for clarification rather than making up an answer.\n"
                    "- IMPORTANT: Valid commands like 'Give me X', 'Tell me about Y' are acceptable and should be answered using general knowledge.\n"
                    "- CRITICAL: DO NOT make up information to answer nonsensical queries. If a query asks for something that doesn't exist (like 'recipe for rest and efforts'), ask for clarification instead of inventing a response.\n\n"
                    "Use ONLY information from the context above. Do not invent facts.\n"
                    "If information is missing, say 'I don't have that information'.\n\n"
                    f"Answer: {prompt}\n\n"
                    "CRITICAL: Keep your response VERY SHORT - maximum 2-3 sentences. "
                    "Provide a clear, concise step-by-step response with only essential information. "
                    "Be conversational and friendly. If more detail is needed, the user will ask."
                    f"{question_instruction}"
                )
                # Build user message for instruction requests
                if SHOW_REASONING_DEBUG:
                    user_content = (
                        f"Show your reasoning, then provide your answer.\n\n"
                        f"Question: {prompt}"
                    )
                else:
                    user_content = (
                        f"Answer this question BRIEFLY (2-3 sentences maximum):\n"
                        f"- If the 'Knowledge context' sections above contain relevant information that answers the query, use that information.\n"
                        f"- If the context does NOT contain relevant information (e.g., it's about unrelated topics or only mentions keywords without answering), use your general knowledge to provide a helpful answer.\n"
                        f"- Provide ONLY essential information - no lengthy explanations or multiple examples.\n"
                        f"- If more detail is needed, the user will ask.\n\n"
                        f"Question: {prompt}"
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
                    person_instruction = f"\n⚠️ Only use information about {person_list} from the context. Do not confuse with other people.\n"
                
                # Simplified list instruction
                list_instruction = ""
                if is_list_request:
                    if is_specific_set_query:
                        # For specific set queries (like "co-founders"), find ALL items, not just TOP 3
                        list_instruction = (
                            "\n📋 LIST QUESTION:\n"
                            "Read all sections completely. Extract ALL items that directly match the query. "
                            "CRITICAL: Find and list ALL items that match the query - do NOT limit the number. "
                            "Read through the entire context carefully to ensure you find every relevant item. "
                            "Only include information explicitly stated in the context.\n"
                        )
                    else:
                        # For general list queries, limit to TOP 3
                        list_instruction = (
                            "\n📋 LIST QUESTION:\n"
                            "Read all sections completely. Extract items that directly match the query. "
                            "CRITICAL: Limit your response to the TOP 3 most relevant items only. "
                            "Do NOT list more than 3 items. "
                            "If there are more items available, mention that more information can be provided if needed. "
                        "Only include information explicitly stated in the context.\n"
                    )
                
                # Build response length guideline - simplified
                if SHOW_REASONING_DEBUG:
                    response_length_guideline = "- Show your reasoning process.\n"
                elif is_list_request:
                    # Only add question instruction if not a conversational query
                    question_instruction = "" if is_conversational else "\n- MANDATORY: End your response with a brief, natural question. This is REQUIRED. Examples: 'Would you like more information about this?' or 'Is there anything else I can help you with?'\n"
                    if is_specific_set_query:
                        # For specific set queries, find ALL items
                        response_length_guideline = (
                            "- CRITICAL: List ALL items that match the query. "
                            "Do NOT limit the number - find every item that matches. "
                            "Read through the entire context completely to ensure you don't miss any items. "
                            "Format as a natural list or numbered list.\n"
                            f"{question_instruction}"
                        )
                    else:
                        # For general list queries, limit to TOP 3
                        response_length_guideline = (
                            "- CRITICAL: List ONLY the top 3 most relevant items that match the query. "
                            "Do NOT exceed 3 items. "
                            "If more items exist, briefly mention that more information is available if needed. "
                            "Format as a numbered or bulleted list (1-3 items maximum).\n"
                        f"{question_instruction}"
                    )
                else:
                    # Only add question instruction if not a conversational query
                    question_instruction = "" if is_conversational else "\n6. MANDATORY: End your response with a brief, natural question. This is REQUIRED. Examples: 'Would you like more information about this?' or 'Is there anything else I can help you with?'\n"
                    response_length_guideline = (
                        "- CRITICAL: Keep responses VERY SHORT - maximum 2-3 sentences total.\n"
                        "- Provide only the essential information needed to answer the question.\n"
                        "- Do NOT provide lengthy explanations, multiple examples, or extensive background.\n"
                        "- If the user wants more details, they will ask - offer to provide more information in your closing question.\n"
                        f"{question_instruction}"
                    )
                
                # Simplified reasoning - only for debug mode
                reasoning_instructions = ""
                if SHOW_REASONING_DEBUG:
                    reasoning_instructions = (
                        "\n🧠 REASONING:\n"
                        "Use only information from the context. Do not invent facts.\n"
                    )
                
                rag_scoring_instructions = ""  # Removed for simplicity
                
                # Ultra-simplified RAG instructions - prevent hallucination
                # Check if query asks about a specific entity (company, person, etc.)
                query_has_entity = any(word in prompt.lower() for word in ['of ', 'at ', 'from '])
                entity_instruction = ""
                if query_has_entity:
                    # Extract entity name from query for explicit instruction
                    # Handle abbreviations with periods (e.g., "Ledger A.I." not "Ledger A")
                    # Match entity name including periods and common abbreviations
                    entity_match = re.search(r'\bof\s+([A-Z][A-Za-z]*(?:\s+[A-Z][A-Za-z.]*)*)', prompt)
                    if entity_match:
                        entity_name = entity_match.group(1).strip()
                        # Normalize entity name variations (e.g., "Ledger A.I." = "LedgerAI" = "Ledger AI")
                        entity_name_normalized = entity_name.replace('.', '').replace(' ', '').lower()
                        entity_instruction = (
                            f"\n⚠️ CRITICAL: The query asks about '{entity_name}'. "
                            f"Extract information where the text explicitly states relationships to '{entity_name}' or its variations (e.g., '{entity_name.replace('.', '')}', '{entity_name.replace('.', ' ')}'). "
                            f"DO NOT exclude information due to minor name variations (spaces, periods, capitalization). "
                            f"Only exclude information about clearly different entities.\n"
                        )
                
                simple_instructions = (
                    "\nINSTRUCTIONS:\n"
                    "1. Read all sections (separated by '---') completely from start to finish.\n"
                    "2. CRITICAL QUERY VALIDATION: Before responding, you MUST first evaluate if the query makes logical sense:\n"
                    "   a. Check if the query contains nonsensical combinations (e.g., 'recipe for rest and efforts' - 'rest and efforts' is not a real recipe name)\n"
                    "   b. Check if key terms in the query are coherent and refer to real concepts (e.g., asking for a recipe for something that doesn't exist)\n"
                    "   c. Check if the query is incomplete, unclear, or contains transcription errors\n"
                    "   d. Check if the query is an incomplete sentence (starts with 'and', 'but', 'or', 'so', 'then', 'also', 'plus', 'make sure', 'ensure', etc.)\n"
                    "   e. IMPORTANT: Commands and instructions like 'Give me X', 'Tell me about Y', 'Show me Z' are VALID requests and should be answered normally.\n"
                    "   f. Check if the query is missing context from a previous conversation\n"
                    "   If the query does NOT make logical sense (e.g., asks for something that doesn't exist, or is truly incomplete), DO NOT force it into a response. Instead, politely ask: 'I'm not sure I understand your question. Could you please repeat or rephrase it?'\n"
                    "3. Check if the context above actually contains information relevant to answering the query.\n"
                    "4. If the context contains relevant information that answers the query:\n"
                    "   - Extract ONLY information that directly answers the query from the context.\n"
                    "   - For relationship queries (co-founders, employees, etc.): ONLY include people whose relationship is explicitly stated as being TO the entity mentioned in the query.\n"
                    "   - DO NOT invent, guess, hallucinate, or use information not in the context.\n"
                    "   - CRITICAL: DO NOT invent product names, company names, or entity names. Only use names that are EXPLICITLY written in the context above.\n"
                    "   - CRITICAL: DO NOT create variations of names (e.g., if context says 'AuraVision', do NOT say 'Ledger Vision' or any other variation). Use EXACTLY the names as written in the context.\n"
                    "   - CRITICAL: If a name is not in the context, do NOT make up a similar-sounding name. Say 'I don't have that information' instead.\n"
                    "   - Only provide logical, factual responses based on the provided context.\n"
                    "5. If the context does NOT contain relevant information (e.g., context is about unrelated topics, or only mentions keywords but doesn't answer the query):\n"
                    "   - Use your general knowledge ONLY if the query is clear and logical.\n"
                    "   - CRITICAL: Even with general knowledge, DO NOT invent specific product names, company names, or entity names that aren't common public knowledge.\n"
                    "   - If the query is unclear or doesn't make sense, ask for clarification instead of guessing.\n"
                    "   - Be clear and informative based on common knowledge, but only if the query is well-formed.\n"
                    "   - Keep it BRIEF - provide only essential information, not lengthy explanations.\n"
                    "6. Format your answer naturally and concisely.\n"
                    "7. REMEMBER: Maximum 2-3 sentences. If more detail is needed, the user will ask.\n"
                    "8. ANTI-HALLUCINATION RULE: Never make up facts, names, dates, or details. If you don't know something, say so clearly rather than inventing information.\n"
                    "9. CRITICAL ANTI-HALLUCINATION: Never invent product names, company names, or entity names. Only use names that are EXPLICITLY written in the context. If a name is not in the context, say 'I don't have that information' rather than guessing or creating variations.\n"
                    )
                
                # No examples - LLM must use only the RAG context provided
                few_shot_examples = ""
                
                # Use CoT format ONLY when RAG is triggered (has_rag_context = True)
                # This matches the user's request: "use CoT when RAG is triggered and use basic LLM conversational mode if RAG not triggered"
                # IMPORTANT: This prompt MUST match the training prompt exactly (from test_rag_cot_model_colab.py)
                # Match test script SYSTEM_PROMPT exactly (from test_rag_cot_model_colab.py)
                # This ensures production behavior matches test results
                cot_system_prompt = (
                    "You are a precise data extraction bot.\n\n"
                    "ALWAYS START WITH REASONING:\n"
                    "Begin every response with \"REASONING:\" - this is MANDATORY.\n\n"
                    "1. REASONING: For each relevant item found in the context:\n"
                    "   - Item: [What you found]\n"
                    "   - Evidence: \"[Verbatim quote from context]\"\n"
                    "   - Action: [KEEP] if it matches the query, otherwise [DISCARD].\n\n"
                    "2. End scan with: - End of scan.\n\n"
                    "3. FINAL ANSWER: based ONLY on [KEEP] items.\n\n"
                    "CRITICAL RULES (APPLY TO ALL QUERIES):\n\n"
                    "EVIDENCE:\n"
                    "- Evidence MUST be EXACT verbatim quote from context - do NOT paraphrase or fabricate.\n"
                    "- You MUST evaluate ALL relevant items in the context before ending the scan.\n"
                    "- CRITICAL: Read through the ENTIRE context completely from start to finish - do NOT stop scanning early.\n"
                    "- CRITICAL: Continue scanning until you reach the VERY END of the context - do NOT stop when you find matches.\n"
                    "- CRITICAL: Items may appear at the VERY END of long contexts - you MUST scan until the absolute end.\n"
                    "- Scan systematically through all chunks, paragraphs, and sections.\n"
                    "- In complex contexts with many entities, scan ALL entities before ending.\n"
                    "- Entities may appear late in the context - continue scanning until the very end.\n"
                    "- Do NOT end scan until you have checked EVERY relevant item in the context.\n"
                    "- Do NOT stop scanning when you find matches - continue until the END of context.\n"
                    "- Do NOT assume you've found all items - always scan to the very end.\n"
                    "- Items may appear at the very end - you MUST scan ALL items before ending.\n\n"
                    "KEEP/DISCARD:\n"
                    "- Items marked [DISCARD] must NEVER appear in FINAL ANSWER.\n"
                    "- FINAL ANSWER must ONLY include items marked [KEEP].\n"
                    "- FINAL ANSWER must include ALL items marked [KEEP] - do not omit any.\n"
                    "- If you mark an item [KEEP] in reasoning, it MUST appear in FINAL ANSWER.\n\n"
                    "MATCHING (PREVENTS HALLUCINATION - STRICT VERBATIM RULE - UNIVERSAL PRINCIPLE):\n"
                    "- UNIVERSAL PRINCIPLE: Query term MUST appear verbatim (exact word-for-word) in evidence for [KEEP].\n"
                    "- This principle applies to ALL query types: roles, names, dates, numbers, locations, products, services, entities, etc.\n"
                    "- CRITICAL: You MUST check if the query term ACTUALLY appears in the evidence - do NOT infer, assume, or hallucinate.\n"
                    "- CRITICAL: If the query term does NOT appear verbatim in evidence → [DISCARD] (NO exceptions, NO inference, NO assumptions, NO memorization, NO hallucination).\n"
                    "- CRITICAL: Different terms are NOT matches - only exact verbatim presence matters.\n"
                    "- CRITICAL: If query mentions a specific entity (e.g., \"co-founders of LedgerAI\"), BOTH the role/item AND the entity name must appear verbatim in evidence.\n"
                    "- CRITICAL: If query asks for \"X of CompanyY\", evidence must say \"X of CompanyY\" verbatim - partial matches are NOT sufficient.\n"
                    "- If query term appears verbatim in evidence → [KEEP] (regardless of other roles/info mentioned).\n"
                    "- CRITICAL: Do NOT memorize specific combinations. Apply the verbatim principle universally to ALL queries and ALL items.\n"
                    "- CRITICAL: The verbatim matching rule is UNIVERSAL - it applies to ALL queries regardless of what item is being asked about.\n"
                    "- DO NOT infer or assume relationships - only use explicitly stated information.\n"
                    "- DO NOT use context clues - only verbatim presence of query term matters.\n"
                    "- DO NOT memorize item combinations - apply the verbatim principle to every query.\n"
                    "- The same verbatim matching principle applies to ALL items: roles, names, dates, numbers, locations, products, services, entities, etc.\n\n"
                    "EMPTY RESULTS:\n"
                    "- If ALL items are marked [DISCARD], FINAL ANSWER must indicate no matches found.\n\n"
                    "OUTPUT FORMAT:\n"
                    "- FINAL ANSWER must include ONLY the information explicitly requested in the query - nothing more, nothing less.\n"
                    "- CRITICAL: FINAL ANSWER must ONLY include items marked [KEEP] in reasoning - do NOT include items marked [DISCARD].\n"
                    "- CRITICAL: If an item was marked [DISCARD] in reasoning, it must NOT appear in FINAL ANSWER.\n"
                    "- Include ONLY what is requested - exclude extra words or context not explicitly requested.\n"
                    "- If query asks for a list, include ALL matching items found in the context (do not omit any).\n"
                    "- Preserve verbatim information from evidence - do NOT paraphrase (e.g., if evidence says \"50 developers\", do NOT change to \"50 employees\").\n\n"
                    "FORMAT REQUIREMENTS (CRITICAL - MUST FOLLOW EXACTLY - NO EXCEPTIONS):\n"
                    "- REASONING: must start with exactly \"REASONING:\" on its own line followed by a newline.\n"
                    "- CRITICAL: After \"REASONING:\" you MUST press Enter/Newline - do NOT put Item on same line.\n"
                    "- CRITICAL: Do NOT use one-line format like \"REASONING: Item:Evidence - ...\" - this is FORBIDDEN.\n"
                    "- CRITICAL: Do NOT combine Item/Evidence/Action on one line - each MUST be on separate lines.\n"
                    "- Each item MUST have three separate lines with line breaks between them:\n"
                    "  Line 1: - Item: [name or value]\n"
                    "  [NEWLINE REQUIRED HERE]\n"
                    "  Line 2: - Evidence: \"[verbatim quote]\"\n"
                    "  [NEWLINE REQUIRED HERE]\n"
                    "  Line 3: - Action: [KEEP] or [DISCARD]\n"
                    "  [NEWLINE REQUIRED HERE before next Item]\n"
                    "- CRITICAL: Do NOT use formats like \"Item:Evidence - ...\" or \"Item:Evidence, Action:...\" - these are FORBIDDEN.\n"
                    "- CRITICAL: Always use the three-line format with proper line breaks.\n"
                    "- CRITICAL: Do NOT use format like \"REASONING:- Item:... - Evidence:... - Action:...\".\n"
                    "- CRITICAL: Do NOT use format like \"REASONING:Item:...\" (no space after colon).\n"
                    "- CRITICAL: You MUST use line breaks (press Enter) between REASONING:, Item:, Evidence:, and Action:.\n"
                    "- FORBIDDEN FORMATS (NEVER USE THESE):\n"
                    "  ❌ REASONING:- Item:... - Evidence:... - Action:...\n"
                    "  ❌ REASONING:Item:... (no space, no newline)\n"
                    "  ❌ REASONING: - Item:... (space but no newline)\n"
                    "- REQUIRED FORMAT (ALWAYS USE THIS):\n"
                    "  ✅ REASONING:\n"
                    "  ✅ - Item: John Smith\n"
                    "  ✅ - Evidence: \"John Smith is the CEO of TechCorp\"\n"
                    "  ✅ - Action: [KEEP]\n"
                    "- This format is MANDATORY for ALL queries - no exceptions, no variations.\n\n"
                    "CRITICAL - STOP AFTER FINAL ANSWER:\n"
                    "- Once you provide FINAL ANSWER, STOP generating immediately.\n"
                    "- Do NOT continue with any further analysis, reasoning, or generation.\n"
                    "- Do NOT add explanations, clarifications, or additional information after FINAL ANSWER.\n"
                    "- Do NOT continue scanning or processing after FINAL ANSWER.\n"
                    "- FINAL ANSWER is the END of your response - nothing comes after it.\n"
                    "- The response MUST end with FINAL ANSWER - no continuation.\n\n"
                    "MANDATORY: After providing the FINAL ANSWER, you MUST end with a brief, natural question that ends with a question mark (?). "
                    "This is REQUIRED - the very last sentence of your response must be a question ending with '?'. "
                    "Examples: 'Would you like more information about this?' or 'Is there anything else I can help you with?' "
                    "Do not include the phrase 'follow up' or 'follow-up' in your question - just ask naturally. "
                    "Make it flow naturally with the conversation topic. "
                    "CRITICAL: The last character of your entire response must be a question mark (?)."
                )
                
                # Build system and user messages - match training format EXACTLY
                # Training format:
                #   System: CoT prompt with CRITICAL RULES
                #   User: "Knowledge context: ...\n---\nQuestion: ..."
                system_content = cot_system_prompt
                
                # User message matches training format exactly
                user_content = f"Knowledge context: {combined_context}\n---\nQuestion: {prompt}"
                
            # For chatml format, separate system and user messages
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
            
            # Match test script parameters EXACTLY (from test_rag_cot_model_colab.py)
            # Test script uses: temperature=0.0, top_p=1.0, repeat_penalty=1.1, max_tokens=2048, stop=["<|im_end|>", "\n\n\n"]
            max_tokens_limit = 2048  # Always use 2048 for RAG queries to match test script
            # Use CoT model for RAG queries (dual-model architecture)
            # Only use CoT model if RAG found content (quick_content_match=True)
            llm_response = llm_chat_simple(
                messages,
                max_tokens=max_tokens_limit,
                temperature=0.0,  # Match test script (0.0 for deterministic output)
                stream=stream,
                use_cot_model=True,  # RAG found content - use CoT model
                top_p=1.0,  # Match test script (disable top_p sampling)
                repeat_penalty=1.1,  # Match test script (lower repeat penalty)
                stop=["<|im_end|>", "\n\n\n"],  # Match test script stop sequences
            )
            if stream:
                # Let CoT filter handle stopping after first FINAL ANSWER (it has logic to detect post-answer markers)
                # The hard stop was cutting off too early and preventing proper CoT filtering
                yield from llm_response
            else:
                return llm_response
        else:
            # No RAG context, use standard prompt with Aura Vision identity
            # Check if memory RAG was attempted but found no useful information
            no_useful_memory = (memory_rag_results is not None and len(memory_rag_results) == 0) or (memory_rag_candidates and len(memory_rag_candidates) > 0 and not memory_rag_results)
            memory_note = ""
            if no_useful_memory:
                memory_note = "\n⚠️ IMPORTANT: No useful information was found in conversation memory (only questions were found, no actual answers). DO NOT make up or guess information. If you don't have reliable information about what was asked, say so clearly rather than providing generic or speculative responses.\n\n"
            
            if is_instruction_request:
                # Only require question for informational queries, not conversational ones
                question_instruction_memory = "" if is_conversational else (
                    "\n\nMANDATORY: Always end your response with a brief, natural question that ends with a question mark (?). "
                    "The very last sentence of your response must be a question ending with '?'. "
                    "Do not include 'follow up' or 'follow-up' in the question text. "
                    "Examples: 'Would you like more information about this?' or 'Is there anything else I can help you with?' "
                    "CRITICAL: The last character of your entire response must be a question mark (?)."
                )
                system_content = (
                    f"{combined_context}\n\n"
                    "You are Aura Vision, an AI agent created by Ledger AI Quantum Corporation. "
                    "You act as a proactive AI agent guiding users to better outcomes through gentle guidance.\n\n"
                    f"{memory_note}"
                    "CRITICAL: Keep your response VERY SHORT - maximum 2-3 sentences or a brief numbered list (3-4 steps max). "
                    "Provide only essential steps. Keep each step concise and actionable. "
                    "Be conversational and friendly, like Siri or Alexa. "
                    "If more detail is needed, the user will ask."
                    f"{question_instruction_memory}"
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
                
                # NO CoT instructions for non-RAG queries - use basic conversational mode
                # CoT is ONLY used when RAG is triggered (has_rag_context = True)
                
                # Only require question for informational queries, not conversational ones
                question_instruction_no_memory = "" if is_conversational else (
                    "\n\nMANDATORY: Your response MUST end with a brief, natural question that ends with a question mark (?). "
                    "This is REQUIRED - the very last sentence of your response must be a question ending with '?'. "
                    "Examples: 'Would you like more information about this?' "
                    "or 'Is there anything else I can help you with?' or 'Need more details on this?' "
                    "Do not include the phrase 'follow up' or 'follow-up' in your question - just ask naturally. "
                    "Make it flow naturally with the conversation topic. This question is in addition to your 2-3 sentence answer. "
                    "CRITICAL: The last character of your entire response must be a question mark (?)."
                )
                system_content = (
                    f"{combined_context}\n\n"
                    "You are Aura Vision, an AI agent created by Ledger AI Quantum Corporation. "
                    "You act as a proactive AI agent guiding users to better outcomes through gentle guidance.\n\n"
                    "CRITICAL RULES:\n"
                    "- Only provide logical, factual responses. Avoid hallucination at all costs.\n"
                    "- IMPORTANT: Commands and instructions like 'Give me X', 'Tell me about Y', 'Show me Z' are VALID requests and should be answered normally using your general knowledge.\n"
                    "- If the user's query is unclear, nonsensical, or doesn't make logical sense, DO NOT force it into a response.\n"
                    "- Instead, politely ask the user to clarify or repeat their question.\n"
                    "- For general knowledge questions (recipes, facts, etc.), use your general knowledge to provide helpful answers.\n"
                    "- Never invent facts, names, dates, or details that aren't in the provided context or common knowledge.\n\n"
                    f"{memory_warning}"
                    "IMPORTANT: Use the conversation memory provided above to answer the user's question. "
                    "If the memory contains relevant information, provide that information in your response. "
                    "If you notice a misspelling or typo, briefly acknowledge it but still answer the actual question asked.\n\n"
                    "CRITICAL: Keep your response VERY SHORT - maximum 2-3 sentences total. "
                    "Get straight to the point with ONLY the essential information needed to answer the question. "
                    "DO NOT provide lengthy explanations, multiple examples, extensive background, or detailed elaborations. "
                    "If the user wants more information, they will ask - you can offer to provide more details in your closing question. "
                    "Be concise, informative, friendly, and conversational."
                    f"{question_instruction_no_memory}"
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
        # Skip filler phrase for conversational queries
        if stream and needs_filler_phrase and not is_conversational:
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
                # Use higher token limit for list questions to ensure all items are included
                max_tokens_limit = MAX_TOKENS_RAG_MODE_LIST if is_list_request else MAX_TOKENS_RAG_MODE
                llm_response = llm_chat_simple(messages, max_tokens=max_tokens_limit, stream=True)
                
                # Always filter structured format to extract only Final Answer for TTS
                # The LLM may output structured format (Known Facts, Reasoning Steps, Final Answer, Confidence)
                # We need to extract only the Final Answer section
                # Also filter out scoring debug sections (---SCORING--- to ---END SCORING---)
                buffer = ""
                in_final_answer = False
                final_answer_started = False
                in_scoring = False
                reasoning_buffer = ""  # Buffer for logging reasoning in debug mode
                scoring_buffer = ""  # Buffer for logging scoring in debug mode
                
                # Track if we're in scoring section - don't yield anything until we're past it
                in_scoring_section = False
                answer_started = False
                accumulated_buffer = ""  # Accumulate chunks until we find the answer
                
                for chunk in llm_response:
                    accumulated_buffer += chunk
                    
                    # Early detection: If buffer contains scoring patterns, don't yield until we find the answer
                    if not answer_started:
                        # Check for scoring markers or patterns (including "Final Answer:" which is still part of scoring format)
                        if re.search(r'\b(SCORING|Item\s+\d+|Person\s+\d+|Score:|Include:|Reason:|Text:|Final\s+Answer:)\b', accumulated_buffer, re.IGNORECASE):
                            in_scoring_section = True
                        
                        # Check if we've reached the actual answer (after "Final Answer:" marker or after scoring section)
                        # Look for the actual list starting with "The co-founders" or a numbered list like "1. Paul" or just "Paul Chou" (first name in list)
                        # Also check for patterns that indicate we're past scoring: "The co-founders", "are:", "include:", etc.
                        # IMPORTANT: Also detect when scoring format is mixed with answer (e.g., "Item 1: NameText:'Name is description'")
                        # In this case, look for pattern like "[Name] is [description]" which indicates the actual answer content
                        answer_patterns = [
                            r'\bThe\s+co-founders\s+(of|are)',
                            r'\bco-founders\s+(of|are)',
                            r'\bfounders\s+(of|are)',
                            r'^\d+\.\s+[A-Z][a-z]+\s+[A-Z][a-z]+',  # "1. Paul Chou"
                            r'^[A-Z][a-z]+\s+[A-Z][a-z]+\s+\(',  # "Paul Chou (Co-Founder"
                            r'^[A-Z][a-z]+\s+[A-Z][a-z]+\s+is\s+a\s+co-founder',  # "Paul Chou is a co-founder"
                            # Detect when answer content appears after scoring format (e.g., "Item 1: NameText:'Name is description'")
                            r'[A-Z][a-z]+\s+[A-Z][a-z]+\s+is\s+(a|an|the)\s+[a-z]+',  # "Bob Carella is a driving force"
                            r'[A-Z][a-z]+\s+[A-Z][a-z]+\s+is\s+[a-z]+',  # "Bob Carella is strategic"
                        ]
                        answer_found = False
                        for pattern in answer_patterns:
                            if re.search(pattern, accumulated_buffer, re.IGNORECASE | re.MULTILINE):
                                answer_started = True
                                in_scoring_section = False
                                # Extract only the answer part - find the actual list start
                                answer_match = re.search(pattern, accumulated_buffer, re.IGNORECASE | re.MULTILINE)
                                if answer_match:
                                    # If we found answer content, extract from that point
                                    # But first, try to clean up any scoring prefixes that might be right before it
                                    start_pos = answer_match.start()
                                    # Look backwards to see if there's scoring format we should remove
                                    before_match = accumulated_buffer[:start_pos]
                                    # Remove common scoring prefixes
                                    before_match = re.sub(r'Item\s+\d+:\s*', '', before_match, flags=re.IGNORECASE)
                                    before_match = re.sub(r'Person\s+\d+:\s*', '', before_match, flags=re.IGNORECASE)
                                    before_match = re.sub(r'Text:\s*[\'"]?', '', before_match, flags=re.IGNORECASE)
                                    before_match = re.sub(r'Score:\s*(HIGH|MEDIUM|LOW)\s*', '', before_match, flags=re.IGNORECASE)
                                    before_match = re.sub(r'Include:\s*(YES|NO)\s*', '', before_match, flags=re.IGNORECASE)
                                    before_match = re.sub(r'Reason:.*?(?=\n|$)', '', before_match, flags=re.IGNORECASE | re.MULTILINE | re.DOTALL)
                                    # If there's still content before the match, check if it's just scoring format
                                    if before_match.strip() and not re.search(r'[A-Z][a-z]+\s+[A-Z][a-z]+', before_match):
                                        # It's likely just scoring format, start from the answer match
                                        accumulated_buffer = accumulated_buffer[start_pos:]
                                    else:
                                        # Keep some context before
                                        accumulated_buffer = before_match + accumulated_buffer[start_pos:]
                                # Also remove "Final Answer:" if it's still there
                                accumulated_buffer = re.sub(r'Final\s+Answer:\s*', '', accumulated_buffer, flags=re.IGNORECASE)
                                # Now set buffer to the cleaned answer and continue processing
                                buffer = accumulated_buffer
                                answer_found = True
                                break
                        
                        if not answer_found:
                            # Still in scoring section, don't process this chunk yet
                            continue
                    
                    # From here on, we're processing the answer (either just found it or already processing it)
                    # Only add to buffer if answer has started
                    if answer_started:
                        buffer += chunk
                    else:
                        # Still in scoring section, skip this chunk
                        continue
                    
                    # First, filter out scoring sections (catch variations: ---SCORING---, SCORING-, -SCORING-, etc.)
                    # Look for scoring markers (flexible matching)
                    scoring_markers = ["---SCORING---", "SCORING-", "-SCORING-", "SCORING"]
                    scoring_start_pos = -1
                    scoring_marker = None
                    for marker in scoring_markers:
                        if marker in buffer and not in_scoring:
                            scoring_start_pos = buffer.find(marker)
                            scoring_marker = marker
                            break
                    
                    if scoring_start_pos >= 0 and not in_scoring:
                        in_scoring = True
                        # Extract scoring section for logging
                        end_markers = ["---END SCORING---", "END SCORING-", "-END SCORING-", "END SCORING"]
                        end_pos = -1
                        end_marker = None
                        for em in end_markers:
                            if em in buffer[scoring_start_pos:]:
                                end_pos = buffer.find(em, scoring_start_pos)
                                end_marker = em
                                break
                        
                        if end_pos >= 0:
                            scoring_buffer = buffer[scoring_start_pos:end_pos]
                            buffer = buffer[:scoring_start_pos] + buffer[end_pos + len(end_marker):]
                            in_scoring = False
                            # Log scoring for debugging
                            if scoring_buffer:
                                print(f"\n{'='*80}")
                                print(f"[Generic] 🔍 [LLM Scoring Debug] SCORING OUTPUT:")
                                print(f"{'='*80}")
                                print(scoring_buffer)
                                print(f"{'='*80}\n")
                        else:
                            # Still waiting for end marker, remove what we have so far
                            buffer = buffer[:scoring_start_pos]
                    
                    # Check for end marker while in scoring section
                    if in_scoring:
                        end_markers = ["---END SCORING---", "END SCORING-", "-END SCORING-", "END SCORING"]
                        for em in end_markers:
                            if em in buffer:
                                end_pos = buffer.find(em)
                                scoring_buffer += buffer[:end_pos]
                                buffer = buffer[end_pos + len(em):]
                                in_scoring = False
                                # Log scoring for debugging
                                if scoring_buffer:
                                    print(f"\n{'='*80}")
                                    print(f"[Generic] 🔍 [LLM Scoring Debug] SCORING OUTPUT:")
                                    print(f"{'='*80}")
                                    print(scoring_buffer)
                                    print(f"{'='*80}\n")
                                break
                    
                    # Skip chunks while in scoring section
                    if in_scoring:
                        continue
                    
                    # Detect scoring sections by content patterns (even without explicit markers)
                    # Look for patterns like "Item X:", "Person X:", "Score: HIGH/MEDIUM/LOW", "Include: YES/NO", "Reason:"
                    scoring_content_pattern = r"Item\s+\d+:|Person\s+\d+:|Score:\s*(HIGH|MEDIUM|LOW)|Include:\s*(YES|NO)|Reason:"
                    if re.search(scoring_content_pattern, buffer, re.IGNORECASE):
                        # This looks like scoring output - find where it ends
                        # Look for end markers or transition to answer
                        # Also look for actual answer content patterns (person name + description)
                        end_patterns = ["---END SCORING---", "END SCORING", "Now answer:", "Final Answer:", "---ANSWER---", "The co-founders", "The founders", "Ledger AI's co-founders"]
                        # Also check for answer content that might be mixed with scoring format
                        answer_content_pattern = r'[A-Z][a-z]+\s+[A-Z][a-z]+\s+is\s+(a|an|the|strategic|driving|visionary|master|dynamic)'
                        found_end = False
                        for ep in end_patterns:
                            if ep in buffer:
                                end_pos = buffer.find(ep)
                                # Extract scoring section for logging
                                scoring_section = buffer[:end_pos]
                                if scoring_section:
                                    print(f"\n{'='*80}")
                                    print(f"[Generic] 🔍 [LLM Scoring Debug] SCORING OUTPUT (detected by pattern):")
                                    print(f"{'='*80}")
                                    print(scoring_section)
                                    print(f"{'='*80}\n")
                                # Remove scoring section
                                buffer = buffer[end_pos + len(ep):].lstrip()
                                found_end = True
                                break
                        # If no explicit end marker, check if answer content appears (even if mixed with scoring)
                        if not found_end and re.search(answer_content_pattern, buffer, re.IGNORECASE):
                            # Find the first occurrence of answer content
                            answer_match = re.search(answer_content_pattern, buffer, re.IGNORECASE)
                            if answer_match:
                                # Extract scoring section before answer for logging
                                scoring_section = buffer[:answer_match.start()]
                                if scoring_section and re.search(scoring_content_pattern, scoring_section, re.IGNORECASE):
                                    print(f"\n{'='*80}")
                                    print(f"[Generic] 🔍 [LLM Scoring Debug] SCORING OUTPUT (detected before answer):")
                                    print(f"{'='*80}")
                                    print(scoring_section)
                                    print(f"{'='*80}\n")
                                # Keep answer content, but clean up any scoring format that's right before it
                                answer_start = answer_match.start()
                                # Look for scoring format immediately before the answer
                                before_answer = buffer[:answer_start]
                                # Remove scoring prefixes
                                before_answer = re.sub(r'Item\s+\d+:\s*', '', before_answer, flags=re.IGNORECASE)
                                before_answer = re.sub(r'Person\s+\d+:\s*', '', before_answer, flags=re.IGNORECASE)
                                before_answer = re.sub(r'Text:\s*[\'"]?', '', before_answer, flags=re.IGNORECASE)
                                before_answer = re.sub(r'Score:\s*(HIGH|MEDIUM|LOW)\s*', '', before_answer, flags=re.IGNORECASE)
                                before_answer = re.sub(r'Include:\s*(YES|NO)\s*', '', before_answer, flags=re.IGNORECASE)
                                before_answer = re.sub(r'Reason:.*?(?=\n|$)', '', before_answer, flags=re.IGNORECASE | re.MULTILINE | re.DOTALL)
                                # Reconstruct buffer with cleaned prefix + answer
                                buffer = before_answer + buffer[answer_start:]
                                found_end = True
                        if not found_end:
                            # Still in scoring section, clear buffer and skip this chunk
                            buffer = ""
                            continue
                    
                    # Additional aggressive filter: remove any remaining scoring format patterns
                    # Look for lines that start with scoring keywords followed by scoring content
                    lines = buffer.split('\n')
                    filtered_lines = []
                    skip_until_answer = False
                    for line in lines:
                        # Check if this line starts scoring format (Item X:, Person X:, Score:, Include:, Reason:, Text:)
                        if re.match(r'^\s*(Item\s+\d+:|Person\s+\d+:|Score:|Include:|Reason:|Text:)\s*', line, re.IGNORECASE):
                            skip_until_answer = True
                            continue
                        # Check if we've reached the actual answer
                        if re.search(r'^(The|Ledger|co-founders|founders|Based on|According to)', line, re.IGNORECASE) and skip_until_answer:
                            skip_until_answer = False
                        # Skip lines while in scoring section
                        if skip_until_answer:
                            continue
                        filtered_lines.append(line)
                    buffer = '\n'.join(filtered_lines)
                    
                    # Final cleanup: remove any remaining scoring patterns (more aggressive)
                    # Remove entire lines that contain scoring format
                    buffer = re.sub(r'Item\s+\d+:.*?(?=\n|$)', '', buffer, flags=re.IGNORECASE | re.MULTILINE)
                    buffer = re.sub(r'Person\s+\d+:.*?(?=\n|$)', '', buffer, flags=re.IGNORECASE | re.MULTILINE)
                    buffer = re.sub(r'Text:\s*[\'"].*?[\'"].*?(?=\n|$)', '', buffer, flags=re.IGNORECASE | re.MULTILINE | re.DOTALL)
                    buffer = re.sub(r'Score:\s*(HIGH|MEDIUM|LOW).*?(?=\n|$)', '', buffer, flags=re.IGNORECASE | re.MULTILINE)
                    buffer = re.sub(r'Include:\s*(YES|NO).*?(?=\n|$)', '', buffer, flags=re.IGNORECASE | re.MULTILINE)
                    buffer = re.sub(r'Reason:.*?(?=\n|$)', '', buffer, flags=re.IGNORECASE | re.MULTILINE)
                    
                    # Remove scoring markers
                    buffer = re.sub(r'-*\s*SCORING\s*-*', '', buffer, flags=re.IGNORECASE)
                    buffer = re.sub(r'-*\s*END\s+SCORING\s*-*', '', buffer, flags=re.IGNORECASE)
                    buffer = re.sub(r'Final\s+Answer:\s*', '', buffer, flags=re.IGNORECASE)
                    
                    # Remove inline scoring format that's mixed with answer content
                    # Pattern: "Item 1: NameText:'Name is description'" -> "Name is description"
                    buffer = re.sub(r'Item\s+\d+:\s*([A-Z][a-z]+\s+[A-Z][a-z]+)Text:\s*[\'"]?', r'\1 ', buffer, flags=re.IGNORECASE)
                    buffer = re.sub(r'Person\s+\d+:\s*([A-Z][a-z]+\s+[A-Z][a-z]+)Text:\s*[\'"]?', r'\1 ', buffer, flags=re.IGNORECASE)
                    # Pattern: "Score: HIGHReason:" -> remove
                    buffer = re.sub(r'Score:\s*(HIGH|MEDIUM|LOW)\s*Reason:\s*', '', buffer, flags=re.IGNORECASE)
                    # Pattern: "Include: YESItem 2:" -> remove
                    buffer = re.sub(r'Include:\s*(YES|NO)\s*(Item|Person)\s+\d+:\s*', '', buffer, flags=re.IGNORECASE)
                    # Pattern: "'Score: HIGHReason:" -> remove
                    buffer = re.sub(r'[\'"]\s*Score:\s*(HIGH|MEDIUM|LOW)\s*Reason:\s*', '', buffer, flags=re.IGNORECASE)
                    # Pattern: "Text:'Name is description'" -> "Name is description"
                    buffer = re.sub(r'Text:\s*[\'"]', '', buffer, flags=re.IGNORECASE)
                    
                    # Remove any standalone scoring keywords that might have leaked through
                    buffer = re.sub(r'\b(Item\s+\d+|Person\s+\d+|Score:|Include:|Reason:|Text:)\b', '', buffer, flags=re.IGNORECASE)
                    
                    # Remove quotes around scoring text that might remain
                    buffer = re.sub(r'[\'"]\s*(Item|Person|Score|Include|Reason|Text):', '', buffer, flags=re.IGNORECASE)
                    buffer = re.sub(r'[\'"]\s*(HIGH|MEDIUM|LOW|YES|NO)\b', '', buffer, flags=re.IGNORECASE)
                    
                    # Clean up any double spaces or weird spacing left after removals
                    buffer = re.sub(r'\s+', ' ', buffer)  # Multiple spaces to single space
                    buffer = re.sub(r'\s+([,.!?;:])', r'\1', buffer)  # Remove space before punctuation
                    buffer = re.sub(r'([,.!?;:])\s*([,.!?;:])', r'\1\2', buffer)  # Remove duplicate punctuation
                    
                    # Check for structured format markers
                    # Look for "Final Answer:" or "---ANSWER---" markers
                    # BUT only if we've already detected the answer started (to avoid false positives from scoring section)
                    final_answer_marker = None
                    if answer_started and "Final Answer:" in buffer and not in_final_answer:
                        final_answer_marker = buffer.find("Final Answer:")
                        # Extract everything before Final Answer for logging (if debug mode)
                        if SHOW_REASONING_DEBUG:
                            reasoning_buffer = buffer[:final_answer_marker].strip()
                    elif answer_started and "---ANSWER---" in buffer and not in_final_answer:
                        final_answer_marker = buffer.find("---ANSWER---")
                        # Extract everything before answer for logging (if debug mode)
                        if SHOW_REASONING_DEBUG:
                            reasoning_buffer = buffer[:final_answer_marker].strip()
                    # Also check if buffer contains actual answer content (not just markers)
                    elif answer_started and not in_final_answer:
                        # Check if buffer contains actual answer patterns
                        answer_content_patterns = [
                            r'\bThe\s+co-founders\s+(of|are)',
                            r'\bco-founders\s+(of|are)',
                            r'^\d+\.\s+[A-Z][a-z]+\s+[A-Z][a-z]+',
                        ]
                        for pattern in answer_content_patterns:
                            if re.search(pattern, buffer, re.IGNORECASE | re.MULTILINE):
                                in_final_answer = True
                                final_answer_started = True
                                break
                    
                    # If we found Final Answer section, start extracting
                    if final_answer_marker is not None and not in_final_answer:
                        if SHOW_REASONING_DEBUG and reasoning_buffer:
                            print(f"\n{'='*80}")
                            print(f"[Generic] 🔍 [LLM Reasoning Debug] FULL REASONING OUTPUT:")
                            print(f"{'='*80}")
                            print(reasoning_buffer)
                            print(f"{'='*80}\n")
                        
                        in_final_answer = True
                        final_answer_started = True
                        
                        # Extract text after "Final Answer:" or "---ANSWER---"
                        if "Final Answer:" in buffer:
                            answer_start_pos = buffer.find("Final Answer:") + len("Final Answer:")
                        else:
                            answer_start_pos = buffer.find("---ANSWER---") + len("---ANSWER---")
                        
                        remaining = buffer[answer_start_pos:].lstrip()
                        # Check if we've already hit Confidence section (end of Final Answer)
                        if "Confidence:" in remaining:
                            confidence_pos = remaining.find("Confidence:")
                            answer_text = remaining[:confidence_pos].strip()
                            if answer_text:
                                yield answer_text
                            break
                        elif "---END ANSWER---" in remaining:
                            end_pos = remaining.find("---END ANSWER---")
                            answer_text = remaining[:end_pos].strip()
                            if answer_text:
                                yield answer_text
                            break
                        elif remaining:
                            yield remaining
                        buffer = ""  # Clear buffer after extracting
                    
                    # If we're in Final Answer section, check for end markers
                    if in_final_answer:
                        # Check for end markers: "Confidence:" or "---END ANSWER---"
                        if "Confidence:" in buffer:
                            confidence_pos = buffer.find("Confidence:")
                            answer_text = buffer[:confidence_pos].strip()
                            if answer_text:
                                yield answer_text
                            break
                        elif "---END ANSWER---" in buffer:
                            end_pos = buffer.find("---END ANSWER---")
                            answer_text = buffer[:end_pos].strip()
                            if answer_text:
                                yield answer_text
                            break
                        elif final_answer_started:
                            # Stream chunks normally in Final Answer section (but check for Confidence: in each chunk)
                            if "Confidence:" in chunk:
                                # Chunk contains end marker, extract answer part only
                                confidence_pos = chunk.find("Confidence:")
                                answer_part = chunk[:confidence_pos].strip()
                                if answer_part:
                                    yield answer_part
                                break
                            else:
                                yield chunk
                    # If not in Final Answer yet, just buffer (don't yield)
                
                # If we never found Final Answer marker, log full response and extract answer if possible
                if not in_final_answer and buffer:
                    if SHOW_REASONING_DEBUG:
                        print(f"\n{'='*80}")
                        print(f"[Generic] 🔍 [LLM Reasoning Debug] FULL RESPONSE (no Final Answer marker found):")
                        print(f"{'='*80}")
                        print(buffer)
                        print(f"{'='*80}\n")
                    
                    # Try to extract Final Answer section even if marker wasn't found
                    if "Final Answer:" in buffer:
                        answer_start = buffer.find("Final Answer:") + len("Final Answer:")
                        remaining = buffer[answer_start:].lstrip()
                        if "Confidence:" in remaining:
                            confidence_pos = remaining.find("Confidence:")
                            answer_text = remaining[:confidence_pos].strip()
                            if answer_text:
                                yield answer_text
                        else:
                            # Yield everything after Final Answer:
                            yield remaining
                    else:
                        # Fallback: yield full response
                        for char in buffer:
                            yield char
            
            return response_with_filler()
        
        # For streaming with debug mode, filter out reasoning and scoring
        if stream and SHOW_REASONING_DEBUG:
            def filter_reasoning():
                buffer = ""
                in_answer = False
                answer_started = False
                in_scoring = False
                scoring_buffer = ""
                
                # Use higher token limit for list questions to ensure all items are included
                max_tokens_limit = MAX_TOKENS_RAG_MODE_LIST if is_list_request else MAX_TOKENS_RAG_MODE
                llm_response = llm_chat_simple(messages, max_tokens=max_tokens_limit, stream=True)
                
                # Track if we're in scoring section - don't yield anything until we're past it
                in_scoring_section_debug = False
                answer_started_debug = False
                
                for chunk in llm_response:
                    buffer += chunk
                    
                    # Early detection: If buffer contains scoring patterns, don't yield until we find the answer
                    if not answer_started_debug:
                        # Check for scoring markers or patterns (including "Final Answer:" which is still part of scoring format)
                        if re.search(r'\b(SCORING|Item\s+\d+|Person\s+\d+|Score:|Include:|Reason:|Text:|Final\s+Answer:)\b', buffer, re.IGNORECASE):
                            in_scoring_section_debug = True
                        
                        # Check if we've reached the actual answer (after "Final Answer:" marker)
                        # Look for the actual list starting with "The co-founders" or a numbered list
                        if re.search(r'\b(The\s+co-founders|The\s+founders|Ledger\s+AI[\'"]?s\s+co-founders|co-founders\s+are|founders\s+are|The\s+[A-Z][a-z]+\s+co-founders|^\d+\.\s+[A-Z][a-z]+)\b', buffer, re.IGNORECASE | re.MULTILINE):
                            answer_started_debug = True
                            in_scoring_section_debug = False
                            # Extract only the answer part - find the actual list start
                            answer_match = re.search(r'\b(The\s+co-founders|The\s+founders|Ledger\s+AI[\'"]?s\s+co-founders|co-founders\s+are|founders\s+are|The\s+[A-Z][a-z]+\s+co-founders|^\d+\.\s+[A-Z][a-z]+)\b', buffer, re.IGNORECASE | re.MULTILINE)
                            if answer_match:
                                buffer = buffer[answer_match.start():]
                            # Also remove "Final Answer:" if it's still there
                            buffer = re.sub(r'Final\s+Answer:\s*', '', buffer, flags=re.IGNORECASE)
                        
                        # If still in scoring section, don't yield
                        if in_scoring_section_debug and not answer_started_debug:
                            continue
                    
                    # First, filter out scoring sections (catch variations: ---SCORING---, SCORING-, -SCORING-, etc.)
                    scoring_markers = ["---SCORING---", "SCORING-", "-SCORING-", "SCORING"]
                    scoring_start_pos = -1
                    scoring_marker = None
                    for marker in scoring_markers:
                        if marker in buffer and not in_scoring:
                            scoring_start_pos = buffer.find(marker)
                            scoring_marker = marker
                            break
                    
                    if scoring_start_pos >= 0 and not in_scoring:
                        in_scoring = True
                        end_markers = ["---END SCORING---", "END SCORING-", "-END SCORING-", "END SCORING"]
                        end_pos = -1
                        end_marker = None
                        for em in end_markers:
                            if em in buffer[scoring_start_pos:]:
                                end_pos = buffer.find(em, scoring_start_pos)
                                end_marker = em
                                break
                        
                        if end_pos >= 0:
                            scoring_buffer = buffer[scoring_start_pos:end_pos]
                            buffer = buffer[:scoring_start_pos] + buffer[end_pos + len(end_marker):]
                            in_scoring = False
                            if scoring_buffer:
                                print(f"\n{'='*80}")
                                print(f"[Generic] 🔍 [LLM Scoring Debug] SCORING OUTPUT:")
                                print(f"{'='*80}")
                                print(scoring_buffer)
                                print(f"{'='*80}\n")
                        else:
                            buffer = buffer[:scoring_start_pos]
                    
                    if in_scoring:
                        end_markers = ["---END SCORING---", "END SCORING-", "-END SCORING-", "END SCORING"]
                        for em in end_markers:
                            if em in buffer:
                                end_pos = buffer.find(em)
                                scoring_buffer += buffer[:end_pos]
                                buffer = buffer[end_pos + len(em):]
                                in_scoring = False
                                if scoring_buffer:
                                    print(f"\n{'='*80}")
                                    print(f"[Generic] 🔍 [LLM Scoring Debug] SCORING OUTPUT:")
                                    print(f"{'='*80}")
                                    print(scoring_buffer)
                                    print(f"{'='*80}\n")
                                break
                    
                    if in_scoring:
                        continue
                    
                    # Detect scoring sections by content patterns (even without explicit markers)
                    scoring_content_pattern = r"Person\s+\d+:|Score:\s*(HIGH|MEDIUM|LOW)|Include:\s*(YES|NO)|Reason:"
                    if re.search(scoring_content_pattern, buffer, re.IGNORECASE):
                        # This looks like scoring output - find where it ends
                        end_patterns = ["---END SCORING---", "END SCORING", "Now answer:", "Final Answer:", "---ANSWER---", "The co-founders", "The founders", "Ledger AI's co-founders"]
                        found_end = False
                        for ep in end_patterns:
                            if ep in buffer:
                                end_pos = buffer.find(ep)
                                scoring_section = buffer[:end_pos]
                                if scoring_section:
                                    print(f"\n{'='*80}")
                                    print(f"[Generic] 🔍 [LLM Scoring Debug] SCORING OUTPUT (detected by pattern):")
                                    print(f"{'='*80}")
                                    print(scoring_section)
                                    print(f"{'='*80}\n")
                                buffer = buffer[end_pos + len(ep):].lstrip()
                                found_end = True
                                break
                        if not found_end:
                            buffer = ""
                            continue
                    
                    # Additional aggressive filter: remove any remaining scoring format patterns
                    lines = buffer.split('\n')
                    filtered_lines = []
                    skip_until_answer = False
                    for line in lines:
                        if re.match(r'^\s*(Person\s+\d+:|Score:|Include:|Reason:|Text:)\s*', line, re.IGNORECASE):
                            skip_until_answer = True
                            continue
                        if re.search(r'^(The|Ledger|co-founders|founders)', line, re.IGNORECASE) and skip_until_answer:
                            skip_until_answer = False
                        if skip_until_answer:
                            continue
                        filtered_lines.append(line)
                    buffer = '\n'.join(filtered_lines)
                    
                    # Final cleanup: remove any remaining scoring patterns
                    buffer = re.sub(r'Person\s+\d+:.*?(?=\n|$)', '', buffer, flags=re.IGNORECASE | re.MULTILINE)
                    buffer = re.sub(r'Score:\s*(HIGH|MEDIUM|LOW).*?(?=\n|$)', '', buffer, flags=re.IGNORECASE | re.MULTILINE)
                    buffer = re.sub(r'Include:\s*(YES|NO).*?(?=\n|$)', '', buffer, flags=re.IGNORECASE | re.MULTILINE)
                    buffer = re.sub(r'Reason:.*?(?=\n|$)', '', buffer, flags=re.IGNORECASE | re.MULTILINE)
                    
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
        # Use higher token limit for list questions to ensure all items are included
        max_tokens_limit = MAX_TOKENS_RAG_MODE_LIST if is_list_request else MAX_TOKENS_RAG_MODE
        llm_response = llm_chat_simple(messages, max_tokens=max_tokens_limit, stream=stream)
        if stream:
            yield from llm_response
        else:
            return llm_response
    else:
        # No RAG context - will fall through to fallback section below
        pass

    # Fallback to direct LLM conversation without external context
    # Detect if user is asking for instructions/steps
    instruction_keywords = ['how to', 'how do i', 'steps', 'step by step', 'instructions', 'guide me', 'walk me through', 'show me how']
    is_instruction_request = any(keyword in prompt.lower() for keyword in instruction_keywords)
    
    # Detect if this is a list request (same logic as RAG mode)
    list_keywords = ['who are', 'who were', 'list all', 'list the', 'what are the', 'what are', 'what were', 'name all', 'name the']
    list_indicators = ['co-founders', 'founders', 'employees', 'members', 'team', 'people', 'individuals']
    is_list_request_direct = any(keyword in prompt.lower() for keyword in list_keywords) or any(indicator in prompt.lower() for indicator in list_indicators)
    
    # Check if this is a conversational phrase (thank you, goodbye, etc.) - skip follow-up question
    normalized_prompt_fallback = prompt.lower()
    conversational_phrases_fallback = [
        'thank you', 'thanks', 'thank', 'thanks a lot', 'thank you very much',
        'goodbye', 'bye', 'see you', 'see ya', 'farewell',
        'you\'re welcome', 'no problem', 'my pleasure', 'anytime',
        'hello', 'hi', 'hey', 'greetings',
        'how are you', 'how\'s it going', 'how\'s everything', 'how do you do',
        'ok', 'okay', 'sure', 'alright', 'got it', 'understood',
        'yes', 'yeah', 'yep', 'no', 'nope',
        'please', 'excuse me', 'sorry', 'pardon'
    ]
    is_conversational_fallback = any(phrase in normalized_prompt_fallback for phrase in conversational_phrases_fallback)
    
    # Exclude information-seeking questions from being marked as conversational
    information_seeking_patterns_fallback = [
        'do you know', 'who is', 'who are', 'who was', 'who were',
        'what is', 'what are', 'what was', 'what were',
        'where is', 'where are', 'where was', 'where were',
        'when is', 'when are', 'when was', 'when were',
        'why is', 'why are', 'why was', 'why were',
        'how is', 'how are', 'how was', 'how were',
        'tell me about', 'tell me who', 'tell me what', 'tell me where',
        'can you tell me', 'could you tell me', 'would you tell me'
    ]
    # If query contains information-seeking patterns, it's NOT conversational
    if any(pattern in normalized_prompt_fallback for pattern in information_seeking_patterns_fallback):
        is_conversational_fallback = False
    
    if is_instruction_request:
        # Only add question instruction if not a conversational query
        question_instruction = "" if is_conversational_fallback else "\n\nAlways end your response with a brief, natural question (do not include 'follow up' or 'follow-up' in the question text). Examples: 'Would you like more information about this?' or 'Is there anything else I can help you with?'"
        system_prompt = (
            "You are Aura Vision, an AI agent created by Ledger AI Quantum Corporation. "
            "You act as a proactive AI agent guiding users to better outcomes through gentle guidance.\n\n"
            "CRITICAL RULES:\n"
            "- Only provide logical, factual responses. Avoid hallucination at all costs.\n"
            "- CRITICAL QUERY VALIDATION: Before responding, you MUST first evaluate if the query makes logical sense:\n"
            "  1. Check if the query contains nonsensical combinations (e.g., 'recipe for rest and efforts' - 'rest and efforts' is not a real recipe name)\n"
            "  2. Check if key terms in the query are coherent and refer to real concepts (e.g., asking for a recipe for something that doesn't exist)\n"
            "  3. Check if the query is incomplete, unclear, or contains transcription errors\n"
            "  4. If the query does NOT make logical sense, DO NOT force it into a response. Instead, politely ask: 'I'm not sure I understand your question. Could you please repeat or rephrase it?'\n"
            "- CRITICAL: Before responding, check if the query is an incomplete sentence (starts with 'and', 'but', 'or', 'so', 'then', 'also', 'make sure', 'ensure', etc.) or an instruction rather than a question. If so, ask for clarification instead of answering.\n"
            "- If the user's query is unclear, nonsensical, or doesn't make logical sense, DO NOT force it into a response.\n"
            "- Instead, politely ask the user to clarify or repeat their question. Example: 'I'm not sure I understand. Could you please rephrase your question or provide more context?'\n"
            "- Never invent facts, names, dates, or details.\n"
            "- CRITICAL: DO NOT invent product names, company names, or entity names. Only use names that you know from common public knowledge.\n"
            "- CRITICAL: DO NOT create variations of names. If you don't know a specific name, say 'I don't have that information' rather than guessing or creating variations.\n"
            "- CRITICAL: DO NOT treat instructions or incomplete sentences as questions. If the query is an instruction or incomplete, ask for clarification rather than making up an answer.\n"
            "- CRITICAL: DO NOT make up information to answer nonsensical queries. If a query asks for something that doesn't exist (like 'recipe for rest and efforts'), ask for clarification instead of inventing a response.\n\n"
            "Provide a clear, step-by-step response (numbered steps) to the user's question. "
            "Keep each step concise and actionable. Be conversational and friendly, like Siri or Alexa."
            f"{question_instruction}"
        )
    elif is_conversational_fallback:
        # For non-conversational phrases (thank you, goodbye, etc.), just respond naturally without follow-up question
        system_prompt = (
            "You are Aura Vision, an AI agent created by Ledger AI Quantum Corporation. "
            "You act as a proactive AI agent guiding users to better outcomes through gentle guidance.\n\n"
            "CRITICAL RULES:\n"
            "- Only provide logical, factual responses. Avoid hallucination at all costs.\n"
            "- CRITICAL QUERY VALIDATION: Before responding, you MUST first evaluate if the query makes logical sense:\n"
            "  1. Check if the query contains nonsensical combinations (e.g., 'recipe for rest and efforts' - 'rest and efforts' is not a real recipe name)\n"
            "  2. Check if key terms in the query are coherent and refer to real concepts (e.g., asking for a recipe for something that doesn't exist)\n"
            "  3. Check if the query is incomplete, unclear, or contains transcription errors\n"
            "  4. If the query does NOT make logical sense, DO NOT force it into a response. Instead, politely ask: 'I'm not sure I understand your question. Could you please repeat or rephrase it?'\n"
            "- CRITICAL: Before responding, check if the query is an incomplete sentence (starts with 'and', 'but', 'or', 'so', 'then', 'also', 'make sure', 'ensure', etc.) or an instruction rather than a question. If so, ask for clarification instead of answering.\n"
            "- If the user's query is unclear, nonsensical, or doesn't make logical sense, DO NOT force it into a response.\n"
            "- Instead, politely ask the user to clarify or repeat their question. Example: 'I'm not sure I understand. Could you please rephrase your question or provide more context?'\n"
            "- Never invent facts, names, dates, or details.\n"
            "- CRITICAL: DO NOT invent product names, company names, or entity names. Only use names that you know from common public knowledge.\n"
            "- CRITICAL: DO NOT create variations of names. If you don't know a specific name, say 'I don't have that information' rather than guessing or creating variations.\n"
            "- CRITICAL: DO NOT treat instructions or incomplete sentences as questions. If the query is an instruction or incomplete, ask for clarification rather than making up an answer.\n"
            "- CRITICAL: DO NOT make up information to answer nonsensical queries. If a query asks for something that doesn't exist (like 'recipe for rest and efforts'), ask for clarification instead of inventing a response.\n\n"
            "Keep your response SHORT and natural - 1-2 sentences maximum. "
            "Be friendly, helpful, and conversational. Respond naturally to the user's phrase. "
            "Do NOT add a follow-up question - just respond appropriately to what they said."
        )
    else:
        # For conversational queries (actual questions), include follow-up question
        # Check if this is a list request and add list-specific instructions
        list_instruction = ""
        if is_list_request_direct:
            list_instruction = (
                "\n📋 LIST QUESTION DETECTED:\n"
                "CRITICAL: Limit your response to ONLY the top 3 most relevant items. "
                "Do NOT exceed 3 items. "
                "Format as a numbered list (1, 2, 3) or bullet points. "
                "If more items exist, briefly mention that more information is available if needed. "
                "Keep each item concise (1-2 sentences per item).\n\n"
            )
        
        # Only add question instruction if not a conversational query
        question_instruction = "" if is_conversational_fallback else (
            "\nMANDATORY: Your response MUST end with a brief, natural question that ends with a question mark (?). "
            "This is REQUIRED - the very last sentence of your response must be a question ending with '?'. "
            "Do not skip it. Examples: 'Would you like more information about this?' "
            "or 'Is there anything else I can help you with?' or 'Need more details on this?' "
            "Do not include the phrase 'follow up' or 'follow-up' in your question - just ask naturally. "
            "Make it flow naturally with the conversation topic. "
            "CRITICAL: The last character of your entire response must be a question mark (?)."
        )
        
        response_length_guideline = ""
        if is_list_request_direct:
            response_length_guideline = (
                "CRITICAL: List ONLY the top 3 most relevant items. "
                "Do NOT exceed 3 items. "
                "If more items exist, mention that more information is available if needed. "
                "Keep each item concise (1-2 sentences per item)."
            )
        else:
            response_length_guideline = (
                "CRITICAL: Keep your response VERY SHORT - maximum 2-3 sentences total. "
                "Provide ONLY essential information - no lengthy explanations, multiple examples, or extensive background. "
                "If the user wants more details, they will ask. Be friendly, helpful, and concise. "
                "Avoid lengthy explanations, excessive background details, or multiple examples."
            )
        
        system_prompt = (
            "You are Aura Vision, an AI agent created by Ledger AI Quantum Corporation. "
            "You act as a proactive AI agent guiding users to better outcomes through gentle guidance.\n\n"
            "CRITICAL RULES:\n"
            "- Only provide logical, factual responses. Avoid hallucination at all costs.\n"
            "- CRITICAL QUERY VALIDATION: Before responding, you MUST first evaluate if the query makes logical sense:\n"
            "  1. Check if the query contains nonsensical combinations (e.g., 'recipe for rest and efforts' - 'rest and efforts' is not a real recipe name)\n"
            "  2. Check if key terms in the query are coherent and refer to real concepts (e.g., asking for a recipe for something that doesn't exist)\n"
            "  3. Check if the query is incomplete, unclear, or contains transcription errors\n"
            "  4. If the query does NOT make logical sense, DO NOT force it into a response. Instead, politely ask: 'I'm not sure I understand your question. Could you please repeat or rephrase it?'\n"
            "- CRITICAL: Before responding, check if the query is an incomplete sentence (starts with 'and', 'but', 'or', 'so', 'then', 'also', 'make sure', 'ensure', etc.) or an instruction rather than a question. If so, ask for clarification instead of answering.\n"
            "- If the user's query is unclear, nonsensical, or doesn't make logical sense, DO NOT force it into a response.\n"
            "- Instead, politely ask the user to clarify or repeat their question. Example: 'I'm not sure I understand. Could you please rephrase your question or provide more context?'\n"
            "- Never invent facts, names, dates, or details.\n"
            "- CRITICAL: DO NOT invent product names, company names, or entity names. Only use names that you know from common public knowledge or that are explicitly mentioned in conversation.\n"
            "- CRITICAL: DO NOT create variations of names. If you don't know a specific name, say 'I don't have that information' rather than guessing or creating variations.\n"
            "- CRITICAL: DO NOT treat instructions or incomplete sentences as questions. If the query is an instruction or incomplete, ask for clarification rather than making up an answer.\n"
            "- CRITICAL: DO NOT make up information to answer nonsensical queries. If a query asks for something that doesn't exist (like 'recipe for rest and efforts'), ask for clarification instead of inventing a response.\n\n"
            f"{list_instruction}"
            f"{response_length_guideline}\n\n"
            f"{question_instruction}"
        )
    
    # NOTE: Conversation memory should ONLY come from memory container API
    # Memory context is already included in RAG context if available
    # The memory_context parameter is deprecated - do not use it
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

    # Use higher token limit for list questions to ensure all items are included
    # Don't wrap the iterator - let base_container's debug_iterator handle logging
    # The base class already wraps it with debug logging
    max_tokens_limit = MAX_TOKENS_DIRECT_MODE_LIST if is_list_request_direct else MAX_TOKENS_DIRECT_MODE
    llm_response = llm_chat_simple(messages, max_tokens=max_tokens_limit, stream=stream)
    if stream:
        yield from llm_response
    else:
        return llm_response


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
    # NOTE: Conversation memory should ONLY be evaluated by memory container, not LLM container
    # The memory container API is called separately in the RAG processing section
    memory_context = None
    # Removed local conversation memory evaluation - only use memory container API
    # if memory_context:
    #     print(f"[Generic] 📝 Retrieved conversation memory context for session {session_id}")
    
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
            # Determine if query is conversational (for response formatting only, not RAG triggering)
            normalized_prompt = prompt.lower()
            conversational_phrases = [
                'thank you', 'thanks', 'thank', 'thanks a lot', 'thank you very much',
                'goodbye', 'bye', 'see you', 'see ya', 'farewell',
                'you\'re welcome', 'no problem', 'my pleasure', 'anytime',
                'hello', 'hi', 'hey', 'greetings',
                'how are you', 'how\'s it going', 'how\'s everything', 'how do you do',
                'ok', 'okay', 'sure', 'alright', 'got it', 'understood',
                'yes', 'yeah', 'yep', 'no', 'nope',
                'please', 'excuse me', 'sorry', 'pardon'
            ]
            is_conversational = any(phrase in normalized_prompt for phrase in conversational_phrases)

            # ------------------------------------------------------------------
            # RAG decision (simplified + predictable)
            # ------------------------------------------------------------------
            # Goal:
            # - Remove brittle "rules" (conversational vs info-seeking heuristics) from RAG gating.
            # - Keep a small everyday-language bypass list (e.g., "how are you") that *never* uses RAG.
            # - Otherwise, use ONLY a quick substring/fuzzy check to decide if RAG should run.
            #
            # This makes behavior deterministic:
            #   bypass_hit -> no RAG
            #   else quick_content_match -> RAG
            #   else -> no RAG
            # PRIMARY DECISION: Use quick_content_match to determine if RAG should be used
            # quick_content_match extracts only key terms (names, important nouns) and skips everyday words
            will_use_rag = False
            print(f"[Generic] 🔍 [RAG Decision] Starting RAG decision check - RAG_MODE={RAG_MODE}")

            if RAG_MODE in ("CPU", "GPU"):
                try:
                    # PRIMARY: Quick substring match on all RAG documents (extracts only key terms, skips everyday words)
                    client = get_rag_client()
                    if client:
                        has_doc_content = client.quick_content_match(prompt)
                        print(f"[Generic] 🔍 [RAG Decision] quick_content_match result: {has_doc_content}")
                        if has_doc_content:
                            will_use_rag = True
                            print(f"[Generic] ✅ [RAG Decision] Document RAG will be used (quick_content_match=True)")
                        else:
                            will_use_rag = False
                            print(f"[Generic] 🔍 [RAG Decision] Document RAG quick_content_match: no relevant content found")
                    else:
                        print(f"[Generic] ⚠️ [RAG Decision] RAG client not available")
                    
                    # Quick check: does memory RAG have relevant content?
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
                                print(f"[Generic] ✅ Memory RAG will be used - quick_match found relevant content")
                    except requests.exceptions.Timeout:
                        pass  # Timeout means we'll skip memory RAG, no filler needed
                    except Exception as e:
                        print(f"[Generic] ⚠️ Memory RAG quick-match check failed: {e}")
                except Exception as e:
                    print(f"[Generic] ⚠️ RAG pre-check failed: {e}")
            
            # Yield filler phrase IMMEDIATELY after detecting RAG will be used (before calling handle_conversation)
            # This ensures it plays while RAG search happens in the background
            print(f"[Generic] 🔍 [Filler Phrase] Checking if filler phrase should be yielded - will_use_rag={will_use_rag}")
            filler_phrase_yielded = False
            # If we're going to do RAG, emit a filler phrase to cover retrieval latency
            if will_use_rag:
                filler_phrase = get_filler_phrase()
                print(f"[Generic] ✅ [Filler Phrase] DECISION: Yielding filler phrase IMMEDIATELY after RAG detection")
                print(f"[Generic] 💭 [Filler Phrase] Filler phrase: '{filler_phrase}'")
                # IMPORTANT: Stream protocol is line-delimited. Each yielded chunk must end with '\n'
                # and tags should NOT be prefixed by a newline, otherwise the speaker parser can miss text.
                print(f"[Generic] 💭 [Filler Phrase] Yielding <sentence_start> tag (line-delimited)")
                yield "<sentence_start>\n"
                print(f"[Generic] 💭 [Filler Phrase] Yielding filler phrase text (line-delimited): '{filler_phrase}'")
                yield f"{filler_phrase}\n"
                print(f"[Generic] 💭 [Filler Phrase] Yielding <sentence_end> tag (line-delimited)")
                yield "<sentence_end>\n"
                filler_phrase_yielded = True
                print(f"[Generic] ✅ [Filler Phrase] Filler phrase yield complete")
            else:
                print(f"[Generic] ⏭️ [Filler Phrase] Skipping filler phrase - will_use_rag={will_use_rag}, is_conversational={is_conversational}")
            
            # Use streaming mode to get tokens as they're generated, with memory context
            # The filler phrase was already yielded above, handle_conversation() will skip its own
            print(f"[Generic] 🔍 [Stream] Calling handle_conversation() with stream=True (filler_phrase_yielded={filler_phrase_yielded})")
            result = handle_conversation(prompt, session_id, memory_context=memory_context, stream=True)
            
            # Check if result is a generator (streaming)
            if hasattr(result, '__iter__') and not isinstance(result, str):
                print(f"[Generic] 🔍 [Stream] Processing stream from handle_conversation()")
                # Special-case: some internal generators (e.g., validate_query() clarification)
                # already yield sentence tags (<sentence_start>/<sentence_end>). If we run those through
                # _word_stream_from_chunks + _sentence_tag_stream we can swallow or distort them,
                # resulting in a silent 0-token output. Detect and pass through directly.
                normalized_chunks = _normalize_stream_chunks(result)
                normalized_chunks = iter(normalized_chunks)
                first_chunk = None
                try:
                    first_chunk = next(normalized_chunks)
                except StopIteration:
                    first_chunk = None

                def _chain_first(first, rest_iter):
                    if first is not None:
                        yield first
                    for x in rest_iter:
                        yield x

                if isinstance(first_chunk, str) and "<sentence_start>" in first_chunk:
                    print("[Generic] ✅ [Stream] Detected pre-tagged sentence stream (passthrough)")
                    token_count = 0
                    full_response_text = ""
                    for chunk in _chain_first(first_chunk, normalized_chunks):
                        if not chunk:
                            continue
                        token_count += 1
                        # Accumulate non-control text for memory storage
                        stripped = chunk.strip()
                        if stripped and stripped not in ("<sentence_start>", "<sentence_end>"):
                            full_response_text += stripped + " "
                        yield chunk if chunk.endswith("\n") else (chunk + "\n")
                    if token_count == 0:
                        print("[Generic] ⚠️ WARNING: No tokens yielded from pre-tagged stream!")
                        # Fallback: speak a generic clarification
                        yield "<sentence_start>\n"
                        yield "I'm sorry, I didn't catch that. Can you repeat your question?\n"
                        yield "<sentence_end>\n"
                    print(f"[Generic] ✅ Streamed response complete (yielded {token_count} tokens)")
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
                    return

                # Normal path: word/sentence streaming for raw model output
                normalized_chunks = _chain_first(first_chunk, normalized_chunks)
                word_stream = _word_stream_from_chunks(normalized_chunks)
                sentence_stream = _sentence_tag_stream(word_stream)
                token_count = 0
                try:
                    print(f"[Generic] 🔍 [Stream] Getting first token from sentence_stream...")
                    first_token = next(sentence_stream)
                    token_count += 1
                    print(f"[Generic] ✅ [Stream] First token received: '{first_token[:50]}...' (type: {type(first_token)})")
                    # Yield the first token
                    if not (first_token.startswith('<') and first_token.endswith('>')):
                        full_response_text += first_token
                    print(f"[Generic] 💭 [Stream] Yielding first token: '{first_token[:50]}...'")
                    yield f"{first_token}\n"
                    # Continue with rest
                    for token in sentence_stream:
                        token_count += 1
                        # Accumulate tokens for memory storage (skip control tags)
                        if not (token.startswith('<') and token.endswith('>')):
                            full_response_text += token
                        yield f"{token}\n"
                except StopIteration:
                    # Instead of yielding empty tags (silent), speak a clarification.
                    yield "<sentence_start>\n"
                    yield "I'm sorry, I didn't catch that. Can you repeat your question?\n"
                    yield "<sentence_end>\n"
                except Exception as e:
                    print(f"[Generic] ⚠️ ERROR iterating sentence_stream: {e}")
                    import traceback
                    traceback.print_exc()
                    # Speak a clarification instead of silence
                    yield "<sentence_start>\n"
                    yield "I'm sorry, I didn't catch that. Can you repeat your question?\n"
                    yield "<sentence_end>\n"
                
                if token_count == 0:
                    print(f"[Generic] ⚠️ WARNING: No tokens yielded from sentence_stream!")
                    # Speak a clarification (belt-and-suspenders)
                    yield "<sentence_start>\n"
                    yield "I'm sorry, I didn't catch that. Can you repeat your question?\n"
                    yield "<sentence_end>\n"
                
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
        stream_with_context(filter_cot_reasoning(filter_think_blocks(generate_response()))),
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
    
    CRITICAL: Punctuation-only tokens (like trailing quotes) are merged with the previous sentence
    to prevent TTS artifacts from standalone punctuation.
    """
    sentence_buffer = ""
    sentence_open = False
    prev_word = None
    buffered_word = None  # One-token lookahead buffer for multi-token abbreviations
    pending_sentence_end = False  # Track if we detected sentence-ending punctuation but haven't closed yet (for trailing punctuation)
    
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
        nonlocal sentence_buffer, sentence_open, pending_sentence_end
        
        word_stripped = word_to_yield.strip()
        
        # Special handling for standalone dashes: they start new sentences for list items
        # Note: This will be called from yield_word, but we can't know if content follows.
        # The main loop will handle skipping dashes that don't have content.
        if word_stripped == '-':
            # Close previous sentence if open
            if sentence_open:
                yield "<sentence_end>"
                sentence_buffer = ""
                pending_sentence_end = False
            # Start new sentence for list item (dash is first word)
            # If no content follows, this will be skipped as empty by the speaker
            sentence_open = True
            yield "<sentence_start>"
            yield word_to_yield
            sentence_buffer = word_to_yield
            return
        
        # Normal word processing
        if not sentence_open:
            sentence_open = True
            pending_sentence_end = False  # Reset flag when starting new sentence
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
        
        # If sentence ending detected, mark it but don't close yet - wait to see if next token is trailing punctuation
        # This allows trailing quotes/punctuation to be grouped with the sentence
        if word_ends_with_punct and sentence_open:
            pending_sentence_end = True  # Mark that we should close after checking next token
            # Don't close yet - will be handled in main loop after checking next token
    
    # Process the word stream
    for word in word_stream:
        # Skip whitespace-only tokens (shouldn't happen after _word_stream_from_chunks fix, but double-check)
        if not word or not word.strip():
            continue
        
        word_stripped = word.strip()
        
        # Skip standalone dashes that are formatting artifacts (no meaningful content)
        # These often appear as list formatting but without actual list items
        if word_stripped == '-' and not sentence_open and not pending_sentence_end:
            # Standalone dash with no context - skip it (likely formatting artifact)
            prev_word = word
            continue
        
        # CRITICAL FIX: If we detected sentence-ending punctuation in previous word,
        # check if current token is trailing punctuation-only. If so, include it in current sentence before closing.
        # This prevents trailing quotes/punctuation from becoming separate sentences.
        if pending_sentence_end and sentence_open:
            # Check if this token is only punctuation (no alphanumeric characters)
            is_punctuation_only = word_stripped and not any(c.isalnum() for c in word_stripped)
            if is_punctuation_only:
                # This is trailing punctuation - add it to current sentence, then close
                yield word
                sentence_buffer += word
                yield "<sentence_end>"
                sentence_buffer = ""
                sentence_open = False
                pending_sentence_end = False
                prev_word = word
                continue
            else:
                # Next token is actual content - close previous sentence and start new one
                yield "<sentence_end>"
                sentence_buffer = ""
                sentence_open = False
                pending_sentence_end = False
                # Fall through to process this word as start of new sentence
        
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
    
    # Close any remaining sentence (including if we detected sentence-ending punctuation but stream ended)
    if sentence_open or pending_sentence_end:
        yield "<sentence_end>"


def filter_cot_reasoning(generator):
    """
    Filter streaming output to buffer and skip CoT REASONING section until FINAL ANSWER is found.
    Only yields content after FINAL ANSWER: marker.
    Also filters out items that were marked [DISCARD] in the reasoning section.
    Works with sentence-tagged token stream.
    Extraction logic matches test_rag_cot_model_colab.py exactly.
    STREAMS tokens incrementally for low TTS latency (doesn't wait for full response).
    
    IMPORTANT: Only applies CoT filtering if REASONING: is detected. Otherwise, passes through
    the response directly (for conversational queries that don't use CoT).
    """
    text_buffer = ""  # Accumulate text for marker detection
    reasoning_buffer = ""  # Buffer reasoning section to extract DISCARD items
    found_final_answer = False
    discarded_items = set()
    kept_items = []  # Store KEEP items from reasoning (ensure all are included in final answer)
    answer_buffer = []  # Buffer answer tokens for cleaning
    collecting_answer = False
    is_cot_response = False  # Track if this is a CoT response (has REASONING:)
    cot_detected = False  # Track if we've detected CoT format
    # Some upstream paths may yield a filler phrase sentence before the model starts emitting CoT.
    # We want that ONE filler sentence to pass through immediately for TTS latency, but we must
    # NOT pass through the model's own sentence-tagged REASONING lines.
    preface_sentence_active = False
    preface_sentence_parts = []
    preface_sentence_used = False
    cot_filler_emitted = False
    final_sentence_started = False  # ensure we only emit <sentence_start> once for final answer
    
    def extract_text(token):
        """Extract text content from token (removing sentence tags)"""
        if not token:
            return ""
        # Remove sentence tags and newlines for marker detection
        text = token.replace("<sentence_start>", "").replace("<sentence_end>", "").replace("\n", " ").strip()
        return text
    
    def extract_discarded_items(reasoning_text):
        """Extract names/items that were marked [DISCARD] in reasoning section"""
        discarded = set()
        # Match complete item blocks: - Item: [Name] ... - Action: [ACTION]
        # Use non-greedy matching to stop at the next "- Item:" or end
        pattern = r'- Item:\s*([^-]+?)\s*-\s*(?:Evidence:[^-]*?)?\s*-\s*Action:\s*(\[[^\]]+\])'
        matches = re.finditer(pattern, reasoning_text, re.IGNORECASE | re.DOTALL)
        
        for match in matches:
            item_name = match.group(1).strip()
            action = match.group(2)
            
            # Only extract if action is [DISCARD]
            if re.search(r'\[\s*DISCARD\s*\]', action, re.IGNORECASE):
                # Clean up item name
                item_name = re.sub(r'^["\']|["\']$', '', item_name)  # Remove quotes
                item_name = item_name.strip()
                if item_name:
                    discarded.add(item_name.lower())
                    # Debug: log what we're extracting
                    print(f"[Generic] 🔍 [CoT Filter] Extracted DISCARD item: '{item_name}' (action: '{action}')")
        return discarded
    
    def extract_kept_items(reasoning_text):
        """Extract names/items that were marked [KEEP] in reasoning section"""
        kept_items = []
        # Strategy: Split by both " - Item:" and "Item:" (with or without dash) to handle all formats
        # This is more robust than regex matching and handles Evidence sections with dashes/quotes
        # Handle both formats: "- Item:" and "Item:" (without dash, like "Item:Liam Hugill")
        segments = re.split(r'\s*-\s*Item:|\bItem:', reasoning_text, flags=re.IGNORECASE)
        print(f"[Generic] 🔍 [CoT Filter] Split reasoning into {len(segments)} segments")
        for i, segment in enumerate(segments[1:], 1):  # Skip first segment (before first Item)
            # Check if this segment has Action:[KEEP]
            if re.search(r'Action:\s*\[\s*KEEP\s*\]', segment, re.IGNORECASE):
                # Extract name (everything before " - Evidence:" or " - Action:" or just "Evidence:" or "Action:")
                # Use non-greedy match to stop at first occurrence
                name_match = re.match(r'^([^-]+?)(?:\s*-\s*(?:Evidence|Action):|(?:Evidence|Action):)', segment, re.IGNORECASE)
                if name_match:
                    item_name = name_match.group(1).strip()
                    # Clean up item name
                    item_name = re.sub(r'^["\']|["\']$', '', item_name)  # Remove quotes
                    item_name = re.sub(r'\s*-\s*Role:.*$', '', item_name, flags=re.IGNORECASE)
                    item_name = item_name.strip()
                    if item_name and item_name not in kept_items:
                        kept_items.append(item_name)
                        print(f"[Generic] 🔍 [CoT Filter] Extracted KEEP item: '{item_name}' (segment {i})")
                    else:
                        print(f"[Generic] ⚠️ [CoT Filter] Skipped empty/duplicate item name from segment {i}: '{item_name}'")
                else:
                    print(f"[Generic] ⚠️ [CoT Filter] No name match found in segment {i} (has KEEP): '{segment[:100]}...'")
        
        # Fallback: If still no items, try regex pattern matching (for edge cases)
        if not kept_items:
            # Pattern: - Item: [Name] - Evidence: ... - Action: [KEEP]
            pattern = r'- Item:\s*([^-]+?)(?:(?!\s*-\s*Item:).)*?-\s*Action:\s*\[\s*KEEP\s*\]'
            matches = re.finditer(pattern, reasoning_text, re.IGNORECASE | re.DOTALL)
            for match in matches:
                item_name = match.group(1).strip()
                item_name = re.sub(r'^["\']|["\']$', '', item_name)
                item_name = re.sub(r'\s*-\s*Role:.*$', '', item_name, flags=re.IGNORECASE)
                item_name = re.sub(r'\s*-\s*Evidence:.*$', '', item_name, flags=re.IGNORECASE)
                item_name = re.sub(r'\s*-\s*$', '', item_name)
                item_name = item_name.strip()
                if item_name and item_name not in kept_items:
                    kept_items.append(item_name)
                    print(f"[Generic] 🔍 [CoT Filter] Extracted KEEP item (regex fallback): '{item_name}'")
        
        return kept_items
    
    # Buffer tokens to detect if this is a CoT response
    token_buffer_list = []  # Store tokens to replay if not CoT
    detection_buffer = ""  # Text for CoT detection
    
    # First pass: detect CoT by buffering initial tokens
    for token in generator:
        # Special-case: allow exactly one *preface* sentence (our injected filler phrase)
        # to pass through immediately before CoT detection buffers the stream.
        if isinstance(token, str) and not preface_sentence_used:
            if token.strip() == "<sentence_start>":
                preface_sentence_active = True
                preface_sentence_parts = [token]
                continue
            if preface_sentence_active:
                preface_sentence_parts.append(token)
                if token.strip() == "<sentence_end>":
                    # Decide whether to pass this preface sentence through immediately
                    preface_text = "".join(
                        t for t in preface_sentence_parts
                        if isinstance(t, str) and t.strip() not in ("<sentence_start>", "<sentence_end>")
                    )
                    preface_text_l = preface_text.lower()
                    # Must NOT contain CoT markers
                    contains_cot_marker = ("reasoning:" in preface_text_l) or ("final answer" in preface_text_l)
                    filler_patterns = [
                        "let me think", "give me a moment", "let me recall", "let me check",
                        "one moment", "just a second", "hold on", "thinking",
                        "i'll need a moment", "i will need a moment", "give me a moment to",
                        "let me look that up", "let me search", "let me find",
                        "let me gather", "let me process", "gather the relevant", "process that question",
                        "one moment, let me", "give me a moment to process", "let me think about that",
                    ]
                    looks_like_filler = any(p in preface_text_l for p in filler_patterns)

                    if looks_like_filler and not contains_cot_marker:
                        for t in preface_sentence_parts:
                            yield t
                        preface_sentence_used = True
                    else:
                        # Not a filler phrase (or it contains CoT markers) → treat as normal stream
                        for t in preface_sentence_parts:
                            token_buffer_list.append(t)
                            tc = extract_text(t)
                            if tc:
                                detection_buffer += tc
                                if not tc.rstrip().endswith(('.', ',', '!', '?', ':', ';', ' ', '\n')):
                                    detection_buffer += " "
                    preface_sentence_active = False
                    preface_sentence_parts = []
                continue

        token_buffer_list.append(token)
        text_content = extract_text(token)
        
        if text_content:
            detection_buffer += text_content
            if not text_content.rstrip().endswith(('.', ',', '!', '?', ':', ';', ' ', '\n')):
                detection_buffer += " "
        
        # Check for CoT markers (case-insensitive)
        detection_buffer_lower = detection_buffer.lower()
        has_reasoning = "reasoning:" in detection_buffer_lower
        
        if has_reasoning:
            is_cot_response = True
            cot_detected = True
            print(f"[Generic] 🔍 [CoT Filter] CoT response detected - applying reasoning filter")
            
            # IMPORTANT: Suppress model-generated filler phrases if we already yielded one
            # The filler phrase should play DURING RAG processing, not after
            # Find reasoning marker (preserve original case from detection_buffer)
            reasoning_pos = detection_buffer_lower.find("reasoning:")
            if reasoning_pos != -1:
                # Extract the actual marker with original case
                reasoning_marker = detection_buffer[reasoning_pos:reasoning_pos+9]
            else:
                reasoning_marker = "REASONING:"  # Fallback
            pre_reasoning_content = detection_buffer.split(reasoning_marker)[0].strip()
            # Store pre-reasoning content for potential fallback if REASONING extraction fails
            stored_pre_reasoning = pre_reasoning_content if pre_reasoning_content else None
            if pre_reasoning_content:
                # Check if pre-reasoning content looks like a direct answer (e.g., "The CEO of Ledger AI is Paul Chou.")
                # If it does, we should extract it as a fallback if REASONING extraction fails
                # But still suppress it from being spoken (we'll use REASONING extraction instead)
                looks_like_answer = any(pattern in pre_reasoning_content.lower() for pattern in [
                    "is ", "are ", "was ", "were ", "the ", "ceo of", "founder of", "co-founder of"
                ]) and len(pre_reasoning_content) < 200  # Short, direct answer format
                
                if looks_like_answer:
                    print(f"[Generic] 💭 [CoT Filter] Pre-reasoning content looks like direct answer: '{pre_reasoning_content[:100]}...' (will suppress but may use as fallback if REASONING extraction fails)")
                else:
                    print(f"[Generic] 💭 [CoT Filter] Suppressing pre-reasoning content: '{pre_reasoning_content[:50]}...' (CoT responses must start with REASONING:)")
                # Don't yield - suppress everything before REASONING:
            
            # Use only the REASONING part for CoT processing
            text_buffer = reasoning_marker + detection_buffer.split(reasoning_marker)[-1]
            break
        elif len(detection_buffer) > 300:
            # After 300 chars with no CoT markers, assume non-CoT
            if not any(marker in detection_buffer_lower for marker in ["reasoning:", "final answer:", "- item:", "- evidence:", "- action:"]):
                is_cot_response = False
                cot_detected = True
                print(f"[Generic] 🔍 [CoT Filter] Non-CoT response detected - passing through directly")
                # Yield all buffered tokens and continue passing through
                for buffered_token in token_buffer_list:
                    yield buffered_token
                # Pass through remaining tokens
                for remaining_token in generator:
                    yield remaining_token
                return  # Exit early for non-CoT responses
    
    # If we get here and CoT was detected, process CoT response
    if not is_cot_response:
        # Shouldn't happen, but safety check
        for token in token_buffer_list:
            yield token
        return
    
    # CoT response detected - continue processing with remaining tokens
    # text_buffer already has initial content
    # CRITICAL: Do NOT replay buffered tokens - they contain REASONING section which should be suppressed
    # Only process NEW tokens from generator (everything after detection buffer)
    for token in generator:
        text_content = extract_text(token)
        
        # CoT response - apply filtering logic
        if not found_final_answer:
            # Still looking for FINAL ANSWER - buffer everything but DO NOT YIELD
            # CRITICAL: Suppress ALL tokens until FINAL ANSWER is found (including REASONING section and sentence tags)
            # Add token text directly (don't add extra space, tokens might already have spacing)
            if text_content:
                text_buffer += text_content
                # Add space only if token doesn't end with punctuation or space
                if not text_content.rstrip().endswith(('.', ',', '!', '?', ':', ';', ' ', '\n')):
                    text_buffer += " "
            
            # Track reasoning section
            if "REASONING:" in text_buffer or "Reasoning:" in text_buffer:
                if "REASONING:" in text_buffer:
                    reasoning_buffer = "REASONING:" + text_buffer.split("REASONING:")[-1]
                elif "Reasoning:" in text_buffer:
                    reasoning_buffer = "Reasoning:" + text_buffer.split("Reasoning:")[-1]
            
            # Check for FINAL ANSWER marker (handle case where it might be split across tokens)
            # Also check for variations
            marker_found = False
            marker = None
            if "FINAL ANSWER:" in text_buffer:
                marker_found = True
                marker = "FINAL ANSWER:"
            elif "Final Answer:" in text_buffer:
                marker_found = True
                marker = "Final Answer:"
            elif "FINAL ANSWER" in text_buffer and ":" in text_buffer:
                # Handle case where colon might be in next token
                final_pos = text_buffer.find("FINAL ANSWER")
                if final_pos != -1 and final_pos + len("FINAL ANSWER") < len(text_buffer):
                    # Check if colon is nearby
                    remaining = text_buffer[final_pos + len("FINAL ANSWER"):]
                    if ":" in remaining[:5]:  # Colon within 5 chars
                        marker_found = True
                        marker = "FINAL ANSWER:"
            
            # CRITICAL: Do NOT yield any tokens (including sentence tags) until FINAL ANSWER is found
            # This suppresses the entire REASONING section from being spoken
            # Suppress ALL tokens: content tokens, sentence tags (<sentence_start>, <sentence_end>), everything
            # Only continue to next token if FINAL ANSWER not found yet
            if not marker_found:
                continue  # Skip ALL tokens (content + sentence tags) until FINAL ANSWER
            
            if marker_found:
                found_final_answer = True
                
                # Emit SECOND filler phrase BEFORE CoT filter processing begins (extraction/debug logging)
                # This plays while the filter extracts DISCARD/KEEP items and logs debug output
                if not cot_filler_emitted:
                    cot_filler_emitted = True
                    cot_filler = get_cot_filler_phrase()
                    print(f"[Generic] 💭 [CoT Filter] Emitting CoT filler phrase (before extraction): '{cot_filler}'")
                    yield "<sentence_start>\n"
                    yield f"{cot_filler}\n"
                    yield "<sentence_end>\n"
                
                # Extract reasoning section to find DISCARD items
                if reasoning_buffer:
                    # Extract reasoning up to FINAL ANSWER
                    if marker in reasoning_buffer:
                        reasoning_text = reasoning_buffer.split(marker)[0]
                    else:
                        reasoning_text = reasoning_buffer
                else:
                    # Fallback: extract from text_buffer
                    text_before = text_buffer.split(marker)[0]
                    if "REASONING:" in text_before:
                        reasoning_text = "REASONING:" + text_before.split("REASONING:")[-1]
                    elif "Reasoning:" in text_before:
                        reasoning_text = "Reasoning:" + text_before.split("Reasoning:")[-1]
                    else:
                        reasoning_text = text_before
                
                discarded_items = extract_discarded_items(reasoning_text)
                kept_items = extract_kept_items(reasoning_text)  # Extract KEEP items (update outer scope variable)
                
                # DEBUG: Print FULL raw response (for debugging)
                print(f"\n{'='*80}")
                print(f"[Generic] 📋 [CoT Debug] FULL RAW TEXT BUFFER (first 2000 chars):")
                print(f"{'='*80}")
                clean_buffer = text_buffer.replace("<sentence_start>", "").replace("<sentence_end>", "").strip()
                print(clean_buffer[:2000])
                print(f"{'='*80}\n")
                
                # DEBUG: Print reasoning section
                print(f"\n{'='*80}")
                print(f"[Generic] 🧠 [CoT Reasoning Debug] REASONING OUTPUT:")
                print(f"{'='*80}")
                clean_reasoning = reasoning_text.replace("<sentence_start>", "").replace("<sentence_end>", "").strip()
                print(clean_reasoning[:2000])  # Increased to 2000 chars to see more
                print(f"{'='*80}")
                if discarded_items:
                    print(f"[Generic] 🚫 [CoT Reasoning Debug] Items marked DISCARD: {discarded_items}")
                else:
                    print(f"[Generic] 🚫 [CoT Reasoning Debug] No DISCARD items found")
                if kept_items:
                    print(f"[Generic] ✅ [CoT Reasoning Debug] Items marked KEEP ({len(kept_items)}): {kept_items}")
                else:
                    print(f"[Generic] ⚠️  [CoT Reasoning Debug] WARNING: No KEEP items extracted! This may indicate extraction failure.")
                    # Fallback: try to extract all items with KEEP action using simpler pattern
                    print(f"[Generic] 🔄 [CoT Reasoning Debug] Attempting fallback extraction...")
                    # Look for all "Action:[KEEP]" occurrences and extract preceding Item names
                    keep_matches = list(re.finditer(r'Action:\s*\[\s*KEEP\s*\]', clean_reasoning, re.IGNORECASE))
                    print(f"[Generic] 🔍 [CoT Reasoning Debug] Found {len(keep_matches)} Action:[KEEP] markers")
                    for i, keep_match in enumerate(keep_matches):
                        # Find the preceding "Item:" before this KEEP action
                        text_before = clean_reasoning[:keep_match.start()]
                        # Find the last "Item:" before this KEEP
                        item_match = list(re.finditer(r'- Item:\s*([^-]+?)(?:\s*-\s*(?:Evidence|Action):)', text_before, re.IGNORECASE))
                        if item_match:
                            last_item = item_match[-1]
                            item_name = last_item.group(1).strip()
                            item_name = re.sub(r'^["\']|["\']$', '', item_name)
                            item_name = item_name.strip()
                            if item_name and item_name not in kept_items:
                                kept_items.append(item_name)
                                print(f"[Generic] 🔍 [CoT Filter] Fallback extracted KEEP item: '{item_name}'")
                    if kept_items:
                        print(f"[Generic] ✅ [CoT Reasoning Debug] Fallback extraction found {len(kept_items)} KEEP items: {kept_items}")
                print(f"{'='*80}\n")
                
                # Extract initial answer text from buffer.
                # IMPORTANT: use the FIRST occurrence of the marker so later leaked "FINAL ANSWER:" blocks
                # (or post-answer REASONING) can't override the correct answer.
                initial_answer = text_buffer.split(marker, 1)[-1].strip()
                if initial_answer:
                    # Clean up initial answer
                    initial_answer = re.sub(r'\[(KEEP|DISCARD|Action|Result)\]', '', initial_answer, flags=re.IGNORECASE)
                    initial_answer = re.sub(r'(?m)^- .*$', '', initial_answer).strip()
                    answer_buffer.append(initial_answer)
                
                collecting_answer = True
                # IMPORTANT: do NOT yield <sentence_start> here.
                # We only emit <sentence_start> right before we output the cleaned final answer,
                # otherwise the speaker will buffer in silence while we filter.
                continue
        
        # CRITICAL: If we haven't found FINAL ANSWER yet, continue to next token (suppress all REASONING tokens)
        # This ensures REASONING section is NEVER yielded - only FINAL ANSWER content is yielded
        if not found_final_answer:
            continue
        
        # After FINAL ANSWER found, buffer answer tokens (need enough text to clean/truncate safely).
        # CRITICAL: Stop collecting as soon as we see the model start emitting ANY new sections.
        # This prevents leaks like "... andREASONING: I" (no whitespace before REASONING).
        if found_final_answer and collecting_answer:
            if text_content:
                next_piece = text_content
                # Fast, robust marker detection (case-insensitive; catches attached markers like "andREASONING:")
                probe = ("".join(answer_buffer) + next_piece)
                probe_l = probe.lower()
                if (
                    "reasoning:" in probe_l
                    or "final answer:" in probe_l
                    or "final_answer:" in probe_l
                    or "end of scan" in probe_l
                    or "search continues" in probe_l
                    or "i'm sorry, but i don't" in probe_l
                    or "i don't have specific information" in probe_l
                ):
                    print("[Generic] ⚠️ [CoT Filter] Detected post-answer marker/leak while collecting; stopping stream to prevent hallucination")
                    collecting_answer = False
                    # Stop the generator entirely - don't process any more tokens
                    # This prevents the model from continuing and generating a second answer
                    break
                answer_buffer.append(next_piece)
            continue
    
    # After stream ends, clean and yield final answer (matches test script extraction)
    if found_final_answer and answer_buffer:
        # DEBUG: Print raw answer buffer before cleaning
        raw_answer = " ".join(answer_buffer).strip()
        print(f"\n{'='*80}")
        print(f"[Generic] 📝 [CoT Debug] RAW ANSWER BUFFER (before cleaning):")
        print(f"{'='*80}")
        print(raw_answer[:500])
        print(f"{'='*80}\n")
        
        final_answer = raw_answer
        
        # Cleanup matches test script exactly
        final_answer = re.sub(r'\[(KEEP|DISCARD|Action|Result)\]', '', final_answer, flags=re.IGNORECASE)
        final_answer = re.sub(r'(?m)^- .*$', '', final_answer).strip()  # Remove bulleted lines
        final_answer = re.sub(r'- End of scan\.?\s*', '', final_answer, flags=re.IGNORECASE)
        # If the model "bleeds" extra sections into FINAL ANSWER, hard-truncate them.
        # We've seen outputs like: "<final>. REASONING: ... DISCARD ... Final Answer: ... University of Arizona."
        # For TTS, we only want the first FINAL ANSWER span.
        # CRITICAL: Find the FIRST occurrence of any truncation marker and cut everything after it
        # Must catch REASONING even when attached to previous word (e.g., "andREASONING:")
        # Remove word boundaries (\b) from REASONING patterns to catch attached instances
        trunc_markers = [
            r'REASONING:\s*I\s+will',     # "REASONING: I will..." (common pattern after first answer)
            r'REASONING:\s*I\b',          # "REASONING: I" (without "will")
            r'REASONING:\s*[A-Z]',        # "REASONING: [Any capital letter]" (no word boundary - catches attached)
            r'REASONING:',                 # "REASONING:" (catch any occurrence, even attached like "andREASONING:")
            r'Reasoning:\s*I\s+will',     # "Reasoning: I will..."
            r'Reasoning:\s*I\b',           # "Reasoning: I"
            r'Reasoning:\s*[A-Z]',         # "Reasoning: [Any capital letter]"
            r'Reasoning:',                  # "Reasoning:" (catch any occurrence)
            r'\bFINAL\s+ANSWER:\s*[A-Z]',  # "FINAL ANSWER: [Any capital letter]" (second FINAL ANSWER)
            r'\bFinal\s+Answer:\s*[A-Z]',  # "Final Answer: [Any capital letter]"
            r'\bFinal\s+answer:\s*[A-Z]',  # "Final answer: [Any capital letter]"
            r'\bSEARCH\s+CONTINUES',       # "SEARCH CONTINUES"
            r'\bEND\s+OF\s+SCAN',          # "END OF SCAN" (if it appears after first answer)
            r'\bDISCARD\b\s*Final\s*Answer', # "DISCARD Final Answer"
            r"I'm\s+sorry,\s+but\s+I\s+don't", # Apology after hallucination
            r"I\s+don't\s+have\s+specific\s+information", # "I don't have specific information..."
        ]
        
        # Find the earliest truncation marker
        earliest_pos = len(final_answer)
        for pattern in trunc_markers:
            matches = list(re.finditer(pattern, final_answer, re.IGNORECASE))
            for match in matches:
                if match.start() < earliest_pos:
                    earliest_pos = match.start()
        
        # If we found a truncation marker, cut everything after it
        # BUT: Don't cut mid-sentence - find the last sentence boundary before the marker
        if earliest_pos < len(final_answer):
            print(f"[Generic] ✂️  [CoT Filter] Found truncation marker at position {earliest_pos}")
            # Find the last sentence-ending punctuation before the marker
            # Look for . ! ? followed by space or end of string, within last 200 chars before marker
            search_start = max(0, earliest_pos - 200)
            text_before_marker = final_answer[search_start:earliest_pos]
            
            # Find last sentence boundary (period, exclamation, question mark)
            sentence_end_match = None
            for punct in ['.', '!', '?']:
                # Look for punctuation followed by space or end
                pattern = re.escape(punct) + r'(?:\s+|$)'
                matches = list(re.finditer(pattern, text_before_marker))
                if matches:
                    last_match = matches[-1]
                    if sentence_end_match is None or last_match.end() > sentence_end_match.end():
                        sentence_end_match = last_match
            
            if sentence_end_match:
                # Truncate at the sentence boundary (adjust for search_start offset)
                truncate_at = search_start + sentence_end_match.end()
                print(f"[Generic] ✂️  [CoT Filter] Truncating at sentence boundary (position {truncate_at}) instead of mid-sentence")
                final_answer = final_answer[:truncate_at].strip()
            else:
                # No sentence boundary found - truncate at marker but try to clean up
                print(f"[Generic] ✂️  [CoT Filter] Truncating at position {earliest_pos} (no sentence boundary found)")
                final_answer = final_answer[:earliest_pos].strip()
                # Remove trailing incomplete phrases (common after truncation)
                final_answer = re.sub(r'\s+from\s+the\s*\.?\s*$', '', final_answer, flags=re.IGNORECASE)
                final_answer = re.sub(r'\s+at\s+the\s*\.?\s*$', '', final_answer, flags=re.IGNORECASE)
                final_answer = re.sub(r'\s+in\s+the\s*\.?\s*$', '', final_answer, flags=re.IGNORECASE)
                final_answer = re.sub(r'\s+to\s+the\s*\.?\s*$', '', final_answer, flags=re.IGNORECASE)
                final_answer = re.sub(r'\s+the\s*\.?\s*$', '', final_answer, flags=re.IGNORECASE)
                final_answer = re.sub(r'\s+\.\s*$', '', final_answer)  # Remove trailing period if sentence incomplete
                # If we truncated right after a conjunction, drop it ("... and")
                final_answer = re.sub(r'\b(and|or)\s*$', '', final_answer, flags=re.IGNORECASE).strip()
        
        # Clean up trailing punctuation/whitespace after truncation
        final_answer = re.sub(r'\s+', ' ', final_answer).strip()
        final_answer = re.sub(r'\s+\.', '.', final_answer)
        final_answer = re.sub(r'\s+,', ',', final_answer)
        # Remove trailing incomplete sentences (common after truncation)
        final_answer = re.sub(r'\s+-\s*$', '', final_answer)  # Remove trailing " -"
        final_answer = re.sub(r'\s+\.\.\.\s*$', '', final_answer)  # Remove trailing "..."
        final_answer = final_answer.strip(" \t\n\r")
        
        # POST-PROCESSING: Clean up formatting (deterministic, no model confusion)
        # 1. Remove role titles for "Who is [ROLE]?" queries (e.g., "The CTO" → just the name)
        # Pattern: "The [ROLE]" or "[ROLE]" at start of answer
        final_answer = re.sub(r'^(The\s+)?(CEO|CTO|CFO|CMO|COO|Chief\s+\w+\s+Officer|President|Director|Manager|Lead|Head\s+of\s+\w+)(\s+is\s+|\s+of\s+|\s+at\s+)?', '', final_answer, flags=re.IGNORECASE)
        
        # 2. Remove dates/years from amounts (e.g., "$50 million in 2023" → "$50 million")
        # Pattern: "in 20XX" or "for 20XX" or "(20XX)" after amounts
        final_answer = re.sub(r'\s+(in|for|during)\s+(19|20)\d{2}\b', '', final_answer, flags=re.IGNORECASE)
        final_answer = re.sub(r'\s*\(19|20\d{2}\)', '', final_answer)  # Remove "(2023)" style dates
        
        # 3. Remove standalone years (e.g., "2023" when not part of a date)
        # Only remove if it's clearly a year (4 digits) and not part of a larger number
        final_answer = re.sub(r'\b(19|20)\d{2}\b(?=\s|$|\.|,|;|:)', '', final_answer)
        
        # CRITICAL: Ensure ALL KEEP items are included and ALL DISCARD items are removed
        # Step 1: Remove ALL DISCARD items from final answer
        if discarded_items:
            for discarded_name in discarded_items:
                name_parts = discarded_name.split()
                if len(name_parts) > 1:
                    pattern = r'\b' + r'\s+'.join([re.escape(part) for part in name_parts]) + r'\b'
                else:
                    pattern = r'\b' + re.escape(discarded_name) + r'\b'
                final_answer = re.sub(pattern, '', final_answer, flags=re.IGNORECASE)
            # Clean up extra spaces and commas after filtering
            final_answer = re.sub(r'\s+', ' ', final_answer)
            final_answer = re.sub(r',\s*,', ',', final_answer)
            final_answer = re.sub(r',\s*and\s*,', ' and ', final_answer)
            final_answer = re.sub(r'^\s*,\s*', '', final_answer)
            final_answer = re.sub(r'\s*,\s*$', '', final_answer)
            if final_answer.strip():
                print(f"[Generic] ✂️  [CoT Reasoning Debug] Filtered FINAL ANSWER (removed DISCARD items): {final_answer[:200]}...")
        
        # Step 2: Ensure ALL KEEP items are included in final answer
        if kept_items:
            # Extract all names from final answer (case-insensitive)
            final_answer_lower = final_answer.lower()
            missing_items = []
            print(f"[Generic] 🔍 [CoT Reasoning Debug] Checking {len(kept_items)} KEEP items against final answer: '{final_answer[:200]}...'")
            for kept_item in kept_items:
                kept_lower = kept_item.lower()
                # Check if item appears in final answer (handle variations)
                item_parts = kept_lower.split()
                if len(item_parts) > 1:
                    # Multi-word name: check if all parts appear together
                    pattern = r'\b' + r'\s+'.join([re.escape(part) for part in item_parts]) + r'\b'
                    found = re.search(pattern, final_answer_lower)
                    if not found:
                        missing_items.append(kept_item)
                        print(f"[Generic] ⚠️  [CoT Reasoning Debug] Missing: '{kept_item}' (pattern: {pattern})")
                    else:
                        print(f"[Generic] ✅ [CoT Reasoning Debug] Found: '{kept_item}' in final answer")
                else:
                    # Single word: simple check
                    if kept_lower not in final_answer_lower:
                        missing_items.append(kept_item)
                        print(f"[Generic] ⚠️  [CoT Reasoning Debug] Missing: '{kept_item}' (single word)")
                    else:
                        print(f"[Generic] ✅ [CoT Reasoning Debug] Found: '{kept_item}' in final answer")
            
            if missing_items:
                print(f"[Generic] ⚠️  [CoT Reasoning Debug] Missing KEEP items in final answer: {missing_items}")
                # Add missing items to final answer
                # Extract existing names from answer to preserve order (check ALL kept_items, not just multi-word)
                existing_names = []
                for kept_item in kept_items:
                    kept_lower = kept_item.lower()
                    item_parts = kept_lower.split()
                    if len(item_parts) > 1:
                        # Multi-word name: check if all parts appear together
                        pattern = r'\b' + r'\s+'.join([re.escape(part) for part in item_parts]) + r'\b'
                        if re.search(pattern, final_answer_lower):
                            existing_names.append(kept_item)
                    else:
                        # Single word: simple check
                        if kept_lower in final_answer_lower:
                            existing_names.append(kept_item)
                
                # Combine existing and missing items, maintaining order from kept_items
                # CRITICAL: Include ALL items from kept_items (they were marked KEEP, so they must be in final answer)
                # Build all_items in the order they appear in kept_items
                all_items_ordered = []
                # First, add all items from kept_items in order (ensures all KEEP items are included)
                for item in kept_items:  # Use kept_items order (as extracted from reasoning)
                    if item not in all_items_ordered:
                        all_items_ordered.append(item)
                
                # Verify: all kept_items should be in all_items_ordered
                if len(all_items_ordered) != len(kept_items):
                    print(f"[Generic] ⚠️  [CoT Reasoning Debug] WARNING: all_items_ordered ({len(all_items_ordered)}) != kept_items ({len(kept_items)})")
                    # Force add any missing items
                    for item in kept_items:
                        if item not in all_items_ordered:
                            all_items_ordered.append(item)
                            print(f"[Generic] 🔧 [CoT Reasoning Debug] Force-added missing item: '{item}'")
                
                all_items = all_items_ordered
                
                # Reconstruct answer with all KEEP items
                if len(all_items) == 1:
                    final_answer = all_items[0]
                elif len(all_items) == 2:
                    final_answer = f"{all_items[0]} and {all_items[1]}"
                else:
                    # Multiple items: "X, Y, and Z"
                    items_str = ", ".join(all_items[:-1]) + f", and {all_items[-1]}"
                    # Try to preserve original answer structure if it has a prefix
                    if ":" in final_answer or "are" in final_answer.lower():
                        # Extract prefix (e.g., "The co-founders of Ledger AI are:")
                        prefix_match = re.match(r'^([^:]+:?\s*)', final_answer)
                        if prefix_match:
                            prefix = prefix_match.group(1)
                            final_answer = prefix + items_str
                        else:
                            final_answer = items_str
                    else:
                        final_answer = items_str
                
                print(f"[Generic] ✅ [CoT Reasoning Debug] Added missing KEEP items: {final_answer[:200]}...")
            else:
                print(f"[Generic] ✅ [CoT Reasoning Debug] All KEEP items present in final answer")
        
        if final_answer.strip():
            print(f"[Generic] 📝 [CoT Reasoning Debug] Final FINAL ANSWER: {final_answer[:200]}...")
            # Note: Second filler phrase is now emitted earlier (before CoT extraction/debug logging)
            # to play during the filtering phase

            # Now emit the actual final answer
            if not final_sentence_started:
                final_sentence_started = True
                yield "<sentence_start>\n"
            # Yield cleaned answer word by word
            words = final_answer.strip().split()
            for i, word in enumerate(words):
                if i < len(words) - 1:
                    yield word + " "
                else:
                    yield word
            yield "\n<sentence_end>\n"
        else:
            # Empty answer - provide fallback
            print(f"[Generic] ⚠️ [CoT Reasoning Debug] FINAL ANSWER is empty after cleaning - providing fallback")
            fallback = "I don't understand. Could you please repeat or rephrase your question?"
            for word in fallback.split():
                yield word + " "
            yield "\n<sentence_end>\n"
    elif not found_final_answer:
        # No FINAL ANSWER found - debug what we actually received
        print(f"\n{'='*80}")
        print(f"[Generic] ⚠️ [CoT Reasoning Debug] No FINAL ANSWER found in model output")
        print(f"{'='*80}")
        print(f"[Generic] 📝 Full text buffer (first 2000 chars): {text_buffer[:2000]}")
        print(f"{'='*80}")
        print(f"[Generic] 🔍 Checking for markers:")
        print(f"  - 'FINAL ANSWER:' in buffer: {'FINAL ANSWER:' in text_buffer}")
        print(f"  - 'Final Answer:' in buffer: {'Final Answer:' in text_buffer}")
        print(f"  - 'FINAL' in buffer: {'FINAL' in text_buffer}")
        print(f"  - 'ANSWER' in buffer: {'ANSWER' in text_buffer}")
        if "REASONING:" in text_buffer or "Reasoning:" in text_buffer:
            print(f"  - REASONING found: YES")
            reasoning_snippet = text_buffer.split("REASONING:")[-1] if "REASONING:" in text_buffer else text_buffer.split("Reasoning:")[-1]
            print(f"  - After REASONING (first 500 chars): {reasoning_snippet[:500]}")
        else:
            print(f"  - REASONING found: NO")
        print(f"{'='*80}\n")
        
        # Try fallback extraction (like test script)
        temp_response = text_buffer.strip()
        if temp_response.startswith('t'):
            temp_response = temp_response[1:].strip()
        
        clean_response = ""
        if "FINAL ANSWER:" in temp_response:
            clean_response = temp_response.split("FINAL ANSWER:")[-1].strip()
            print(f"[Generic] ✅ Found FINAL ANSWER: using fallback extraction")
        elif "Final Answer:" in temp_response:
            clean_response = temp_response.split("Final Answer:")[-1].strip()
            print(f"[Generic] ✅ Found Final Answer: using fallback extraction")
        elif "- End of scan." in temp_response:
            clean_response = temp_response.split("- End of scan.")[-1].strip()
            print(f"[Generic] ✅ Found End of scan: using fallback extraction")
        else:
            # Last resort: extract KEEP items from reasoning and construct answer
            if reasoning_buffer or ("REASONING:" in temp_response or "Reasoning:" in temp_response):
                reasoning_text = reasoning_buffer if reasoning_buffer else temp_response
                if "REASONING:" in reasoning_text:
                    reasoning_text = "REASONING:" + reasoning_text.split("REASONING:")[-1]
                elif "Reasoning:" in reasoning_text:
                    reasoning_text = "Reasoning:" + reasoning_text.split("Reasoning:")[-1]
                
                kept_items = extract_kept_items(reasoning_text)
                if kept_items:
                    # Construct natural answer from kept items
                    if len(kept_items) == 1:
                        clean_response = kept_items[0]
                    elif len(kept_items) == 2:
                        clean_response = f"{kept_items[0]} and {kept_items[1]}"
                    else:
                        # Multiple items: "X, Y, and Z"
                        items_str = ", ".join(kept_items[:-1]) + f", and {kept_items[-1]}"
                        clean_response = items_str
                    print(f"[Generic] ✅ Constructed answer from KEEP items: {clean_response[:200]}...")
                else:
                    # Try last block as fallback
                    blocks = temp_response.split('\n\n')
                    if len(blocks) > 1:
                        clean_response = blocks[-1].strip()
                        print(f"[Generic] ⚠️ Using last block as fallback: {clean_response[:200]}")
        
        if clean_response.strip():
            # Clean up the extracted response
            clean_response = re.sub(r'\[(KEEP|DISCARD|Action|Result)\]', '', clean_response, flags=re.IGNORECASE)
            clean_response = re.sub(r'(?m)^- .*$', '', clean_response).strip()
            if clean_response.strip():
                print(f"[Generic] 📝 [CoT Reasoning Debug] Extracted answer via fallback: {clean_response[:200]}...")
                yield "<sentence_start>\n"
                words = clean_response.strip().split()
                for i, word in enumerate(words):
                    if i < len(words) - 1:
                        yield word + " "
                    else:
                        yield word
                yield "\n<sentence_end>\n"
                return
        
        # Final fallback
        print(f"[Generic] ⚠️ [CoT Reasoning Debug] No FINAL ANSWER found - providing fallback")
        fallback = "I don't understand. Could you please repeat or rephrase your question?"
        yield "<sentence_start>\n"
        for word in fallback.split():
            yield word + " "
        yield "\n<sentence_end>\n"
    


def filter_think_blocks(generator):
    """
    Filter streaming output to remove <think> blocks and detect garbage output.
    Mirrors the medical container behavior for parity.
    NOTE: Reasoning filtering is handled by filter_cot_reasoning - this function only handles garbage detection.
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
    
    # Load BASE model at startup (for conversational queries)
    print(f"[Generic] 📦 Loading BASE model (conversational): {BASE_MODEL_PATH}")
    # Offload all layers to GPU for maximum acceleration (set to 0 to disable GPU)
    # For Jetson, offloading all layers typically provides best performance
    n_gpu_layers = -1  # -1 = offload all layers to GPU, 0 = CPU only
    print(f"[Generic] 🚀 GPU acceleration: {n_gpu_layers} layers offloaded to GPU")
    
    # Override base class load_model to add GPU support
    from llama_cpp import Llama
    base_container.model_path = BASE_MODEL_PATH
    base_container.llm_simple = Llama(
        model_path=BASE_MODEL_PATH,
        n_ctx=base_container.SIMPLE_N_CTX,
        n_threads=N_THREADS,
        n_batch=base_container.N_BATCH,
        n_gpu_layers=n_gpu_layers,  # Enable GPU acceleration
        cache_prompt=CACHE_PROMPT,
        chat_format=base_container.SIMPLE_CHAT_FORMAT,
        use_mlock=True,
        use_mmap=True,
        verbose=False
    )
    base_container._model_loaded = True
    llm_simple = base_container.llm_simple  # Set global reference
    print(f"[Generic] ✅ BASE model loaded: {BASE_MODEL_PATH}")
    
    # Pre-load CoT model at startup (eliminates first-query latency)
    print(f"[Generic] 📦 Loading CoT model (RAG queries): {COT_MODEL_PATH}")
    cot_container.model_path = COT_MODEL_PATH
    cot_container.llm_simple = Llama(
        model_path=COT_MODEL_PATH,
        n_ctx=cot_container.SIMPLE_N_CTX,
        n_threads=1,  # Use 1 thread for deterministic output (temperature=0 alone isn't enough)
        n_batch=cot_container.N_BATCH,
        n_gpu_layers=n_gpu_layers,  # Enable GPU acceleration
        cache_prompt=CACHE_PROMPT,
        chat_format=cot_container.SIMPLE_CHAT_FORMAT,
        use_mlock=True,
        use_mmap=True,
        verbose=False
    )
    cot_container._model_loaded = True
    print(f"[Generic] ✅ CoT model loaded: {COT_MODEL_PATH}")
    
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

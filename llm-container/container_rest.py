# === container_rest.py — Aura LLM Container (Modular Architecture)
# Routes requests to appropriate conversation modes:
# - CASUAL: Simple greetings
# - THINKER: Knowledge queries with RAG
# - TRIAGE: Hardcoded diagnostic system (baseline)
# - CLINICIAN: RAG-powered intelligent diagnosis (future)

from flask import Flask, request, jsonify, stream_with_context, Response
from llama_cpp import Llama
from dotenv import load_dotenv
import os, re, json, string, threading, time
from datetime import datetime, timedelta
from glob import glob
from nlg import rewrite as nlg_rewrite
import requests

# Import centralized validation utilities
from validation import (
    match_answer_option, check_typo_similarity, normalize_text, 
    tokenize, normalize_yes_no_response, get_generic_onset_answers,
    match_flexible_time, MIN_MATCH
)

# Import modular conversation modes
from router import route_prompt, ConversationMode, format_mode_info
from casual import handle_casual, stream_casual_response
from thinker import handle_thinker
from triage import detect_condition, process_triage_step, generate_triage_completion, load_state, save_state, get_intro, apply_synonym_expansion, substitute_name, TRIAGE_DEFS, get_steps, is_valid_answer
# Import unified medical mode for comprehensive medical assistance
from unified_medical_mode import UnifiedMedicalSession, is_unified_medical_trigger, get_unified_medical_session, handle_unified_medical_response

# RAG functionality moved to separate RAG container (port 11435)
RAG_SERVICE_URL = "http://localhost:11435"

app = Flask(__name__)
load_dotenv()

# === Thread Safety ===
llm_lock = threading.Lock()

# === Dual Model Config (Optimized for Orin NX) ===
# Complex model (Mistral-7B) for diagnostic reasoning
MODEL_PATH = os.getenv("MODEL_PATH", "/models/Mistral-7B-Instruct-v0.3.Q4_K_M.gguf")
N_CTX = int(os.getenv("N_CTX", "8192"))
CHAT_FORMAT = os.getenv("CHAT_FORMAT", "mistral-instruct")

# Simple model (Llama-1B) for templates/validation
SIMPLE_MODEL_PATH = os.getenv("SIMPLE_MODEL_PATH", "/models/Llama-3.2-1B-Instruct-Q4_K_M.gguf")
SIMPLE_N_CTX = int(os.getenv("SIMPLE_N_CTX", "2048"))
SIMPLE_CHAT_FORMAT = os.getenv("SIMPLE_CHAT_FORMAT", "llama-3")

# Complex model config
model_config = {
    "model_path": MODEL_PATH,
    "n_ctx": N_CTX,
    "n_gpu_layers": -1,
    "n_threads": 6,
    "chat_format": CHAT_FORMAT,
    "use_mlock": True,
    "use_mmap": True,
    "verbose": False,
    "temperature": float(os.getenv("LLM_TEMPERATURE", "0.6")),
    "top_p": float(os.getenv("LLM_TOP_P", "0.85")),
    "top_k": int(os.getenv("LLM_TOP_K", "30")),
    "repeat_penalty": float(os.getenv("LLM_REPEAT_PENALTY", "1.15")),
}

# Simple model config
simple_model_config = {
    "model_path": SIMPLE_MODEL_PATH,
    "n_ctx": SIMPLE_N_CTX,
    "n_gpu_layers": -1,
    "n_threads": 4,  # Fewer threads for simple model
    "chat_format": SIMPLE_CHAT_FORMAT,
    "use_mlock": True,
    "use_mmap": True,
    "verbose": False,
    "temperature": float(os.getenv("LLM_TEMPERATURE", "0.6")),
    "top_p": float(os.getenv("LLM_TOP_P", "0.85")),
    "top_k": int(os.getenv("LLM_TOP_K", "30")),
    "repeat_penalty": float(os.getenv("LLM_REPEAT_PENALTY", "1.15")),
}

print(f"[LLM] 🚀 Loading COMPLEX model: {MODEL_PATH}")
print(f"[LLM] ⚙️  Config: n_ctx={N_CTX}, format={CHAT_FORMAT}")
llm = Llama(**model_config)
print(f"[LLM] ✅ Complex model (Mistral-7B) loaded")

print(f"[LLM] 🚀 Loading SIMPLE model: {SIMPLE_MODEL_PATH}")
print(f"[LLM] ⚙️  Config: n_ctx={SIMPLE_N_CTX}, format={SIMPLE_CHAT_FORMAT}")
llm_simple = Llama(**simple_model_config)
print(f"[LLM] ✅ Simple model (Llama-1B) loaded")

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
        state = reset_session_state(session_id)
        if prompt_norm in RESET_KEYWORDS:
            return jsonify({"response": "Session reset. Start again with your symptoms."})
    
    try:
        # Route to appropriate mode (SAME as /chat)
        state = load_state(session_id)
        mode, updated_state = route_prompt(prompt_norm, state, session_id, llm_chat)
        save_state(updated_state, session_id)
        
        print(f"[Telegram] 🎯 Routed to mode: {mode.upper()}")
        
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
        
        # Dispatch to mode handler (SAME modes as /chat)
        if mode == ConversationMode.CASUAL:
            response = collect_stream(stream_casual_response(prompt_norm, llm_chat, session_id))
            if not response:
                response = "Hello! How can I help you today?"
            return jsonify({"response": response})
        
        elif mode == ConversationMode.THINKER:
            response = collect_stream(handle_thinker(prompt_norm, llm_chat, session_id))
            if not response:
                response = "I don't have information about that."
            return jsonify({"response": response})
        
        elif mode == ConversationMode.TRIAGE:
            # Check if this is a NEW triage session
            if updated_state.get('is_new_triage'):
                # Clear the new session flag
                updated_state['is_new_triage'] = False
                updated_state['step_index'] = 1
                updated_state['last_key'] = get_steps(updated_state['condition'], updated_state)[0].get('key')
                save_state(updated_state, session_id)
                
                condition = updated_state.get('condition')
                steps = get_steps(condition, updated_state)
                
                # Get intro and first question
                intro = substitute_name(TRIAGE_DEFS[condition].get("intro", ""), updated_state.get("user_name"))
                first_question = substitute_name(steps[0].get('question', ''), updated_state.get('user_name'))
                
                response = ""
                if intro:
                    response += intro + " "
                response += first_question
                return jsonify({"response": response})
            else:
                # Continue existing triage
                question, final_state = process_triage_step(prompt, updated_state, session_id, llm_chat)
                save_state(final_state, session_id)
                return jsonify({"response": question})
        
        elif mode == ConversationMode.UNIFIED_MEDICAL:
            try:
                # For Telegram, we don't need immediate fillers since it's text-based
                response = handle_unified_medical_response(prompt, session_id, llm_chat, llm_chat_simple)
                
                # Check if response includes question (dict) or is simple text (str)
                if isinstance(response, dict):
                    # Return question + debug info for Telegram
                    telegram_response = {
                        "response": response.get('question', response.get('message', '')),
                        "debug": response.get('debug')  # Include debug info if available
                    }
                    return jsonify(telegram_response)
                else:
                    # Simple text response
                    return jsonify({"response": response})
            except Exception as e:
                print(f"[Container] ❌ Error in unified medical mode (non-streaming): {e}")
                print(f"[Container] 📋 Error type: {type(e).__name__}")
                print(f"[Container] 📍 Error location: {e.__traceback__.tb_frame.f_code.co_filename}:{e.__traceback__.tb_lineno}")
                print(f"[Container] 🔍 Full traceback:")
                import traceback
                traceback.print_exc()
                
                # NO FALLBACKS - re-raise the actual error
                raise e
        
        else:
            return jsonify({"response": "I'm sorry, I didn't understand that."})
            
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

    Routes requests to appropriate conversation mode:
    - CASUAL: Simple greetings
    - THINKER: Knowledge queries with RAG
    - TRIAGE: Hardcoded diagnostic system (baseline)
    - CLINICIAN: RAG-powered intelligent diagnosis (future)
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
        state = reset_session_state(session_id)
        if prompt_norm in RESET_KEYWORDS:
            def generate_reset():
                yield "<sentence_start>\n🔄 Session reset. Start again with your symptoms.\n<sentence_end>\n"
            # Filter think blocks at container level (though unlikely here)
            return Response(stream_with_context(filter_think_blocks(generate_reset())), mimetype="text/plain")

    # Route to appropriate mode
    state = load_state(session_id)
    mode, updated_state = route_prompt(prompt_norm, state, session_id, llm_chat)
    save_state(updated_state, session_id)

    print(f"[Aura-LLM] 🎯 Routed to mode: {mode.upper()}")

    # Dispatch to mode handler
    if mode == ConversationMode.CASUAL:
        def generate_casual():
            for token in stream_casual_response(prompt_norm, llm_chat, session_id):
                yield token
        # Filter think blocks at container level
        return Response(stream_with_context(filter_think_blocks(generate_casual())), mimetype="text/plain")

    elif mode == ConversationMode.THINKER:
        def generate_thinker():
            for token in handle_thinker(prompt_norm, llm_chat, session_id):
                yield token
        # Filter think blocks at container level
        return Response(stream_with_context(filter_think_blocks(generate_thinker())), mimetype="text/plain")

    elif mode == ConversationMode.TRIAGE:
        print(f"[Triage] 🔍 Mode: TRIAGE, condition={updated_state.get('condition')}, is_new={updated_state.get('is_new_triage')}, step_index={updated_state.get('step_index')}")

        # Check if this is a NEW triage session (just detected condition)
        if updated_state.get('is_new_triage'):
            print(f"[Triage] 🆕 NEW triage session - asking first question")
            # Clear the new session flag AND SAVE immediately
            updated_state['is_new_triage'] = False
            updated_state['step_index'] = 1  # We're asking step 0, so next answer goes to step 1
            updated_state['last_key'] = get_steps(updated_state['condition'], updated_state)[0].get('key')
            save_state(updated_state, session_id)  # CRITICAL: Save state before generating response

            condition = updated_state.get('condition')
            steps = get_steps(condition, updated_state)

            def generate_new_triage():
                # Use NLG for intro and first question
                intro = substitute_name(TRIAGE_DEFS[condition].get("intro", ""), updated_state.get("user_name"))
                if intro:
                    intro_nlg = nlg_rewrite(intro, "intro", {
                        "name": updated_state.get("user_name"),
                        "condition": condition
                    }, updated_state.get("phrasing_history"), llm_chat_once)
                    yield f"<sentence_start>\n{intro_nlg}\n<sentence_end>\n"
                    
                    # Update phrasing_history with intro
                    if "phrasing_history" not in updated_state:
                        updated_state["phrasing_history"] = []
                    updated_state["phrasing_history"].append(intro_nlg)

                raw_q = substitute_name(steps[0].get('question', ''), updated_state.get('user_name'))
                q_nlg = nlg_rewrite(raw_q, "question", {
                    "name": updated_state.get("user_name"),
                    "condition": condition,
                    "key": steps[0].get('key'),
                    "allowed_answers": list(steps[0].get('answers', {}).keys())
                }, updated_state.get("phrasing_history"), llm_chat_once)
                yield f"<sentence_start>\n{q_nlg}\n<sentence_end>\n"
                
                # CRITICAL: Update phrasing_history with the first question
                updated_state["phrasing_history"].append(q_nlg)
                updated_state["phrasing_history"] = updated_state["phrasing_history"][-10:]
                save_state(updated_state, session_id)

            # Filter think blocks at container level
            return Response(stream_with_context(filter_think_blocks(generate_new_triage())), mimetype="text/plain")

        # Continue existing triage - process user's answer
        else:
            print(f"[Triage] 🔄 Continuing triage - processing answer: '{prompt}'")
            def generate_triage_continue():
                try:
                    question, final_state = process_triage_step(prompt, updated_state, session_id, llm_chat)
                    save_state(final_state, session_id)
                    yield f"<sentence_start>\n{question}\n<sentence_end>\n"
                except Exception as e:
                    print(f"[Aura-LLM] ❌ Error in triage: {e}")
                    import traceback
                    traceback.print_exc()
                    print(f"[Aura-LLM] 🔍 Error details: {type(e).__name__}: {str(e)}")
                    yield f"<sentence_start>\nI'm sorry, there was an error processing your triage.\n<sentence_end>\n"
            # Filter think blocks at container level
            return Response(stream_with_context(filter_think_blocks(generate_triage_continue())), mimetype="text/plain")

    elif mode == ConversationMode.UNIFIED_MEDICAL:
        def generate_unified_medical():
            try:
                print("[Container] 🔄 Using dynamic medical assessment for UNIFIED_MEDICAL")
                
                # Check if this will be a simple operation (Llama-1B) or complex operation (Mistral-7B/RAG)
                def will_use_simple_llm(prompt_text):
                    """Predict if the operation will use Llama-1B (simple) or Mistral-7B (complex)"""
                    prompt_lower = prompt_text.lower().strip()
                    
                    # Simple operations that use Llama-1B:
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
                    
                    # Default to complex operation (Mistral-7B/RAG)
                    return False
                
                # Determine if we need a filler based on predicted LLM usage
                will_use_simple = will_use_simple_llm(prompt)
                
                if will_use_simple:
                    # Simple operation (Llama-1B) - no filler needed
                    print(f"[Container] ⚡ Simple operation (Llama-1B) - no filler needed")
                else:
                    # Complex operation (Mistral-7B/RAG) - use filler
                    from thinking_fillers import get_filler
                    immediate_filler = get_filler('question_generation', use_audio=True)
                    filler_text = immediate_filler['text']
                    print(f"[Container] 💬 IMMEDIATE filler for complex operation (Mistral-7B/RAG): '{filler_text}'")
                    
                    # Stream filler immediately
                    yield "<sentence_start>\n"
                    yield f"{filler_text}\n"
                    yield "<sentence_end>\n"
                
                # Now process the actual response in the background
                response = handle_unified_medical_response(prompt, session_id, llm_chat, llm_chat_simple)
                
                print(f"[Container] ✅ Got response from unified medical session")
                
                # Check if response includes filler (dict) or is simple text (str)
                if isinstance(response, dict) and 'question' in response:
                    # Stream the actual question as SEPARATE sentence
                    question_text = response.get('question', '')
                    yield "<sentence_start>\n"
                    yield f"{question_text}\n"
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
                print(f"[Container] ❌ Error in unified medical mode: {e}")
                print(f"[Container] 📋 Error type: {type(e).__name__}")
                print(f"[Container] 📍 Error location: {e.__traceback__.tb_frame.f_code.co_filename}:{e.__traceback__.tb_lineno}")
                print(f"[Container] 🔍 Full traceback:")
                import traceback
                traceback.print_exc()
                
                # NO FALLBACKS - re-raise the actual error
                raise e

        # Filter think blocks at container level
        return Response(stream_with_context(filter_think_blocks(generate_unified_medical())), mimetype="text/plain")

    elif mode == ConversationMode.THINKER:
        def generate_thinker():
            try:
                # handle_thinker already yields streaming chunks with sentence markers
                for chunk in handle_thinker(prompt, llm_chat, session_id):
                    yield chunk
            except Exception as e:
                print(f"[Container] ❌ Error in thinker mode: {e}")
                import traceback
                traceback.print_exc()
                yield f"<sentence_start>\nI'm sorry, I encountered an error processing your query.\n<sentence_end>\n"

        # Filter think blocks at container level
        return Response(stream_with_context(filter_think_blocks(generate_thinker())), mimetype="text/plain")

    else:
        # Fallback
        def generate_fallback():
            yield "<sentence_start>\nHello! I'm AuraVision, your friendly personal assistant. How can I help you today?\n<sentence_end>\n"
        # Filter think blocks at container level
        return Response(stream_with_context(filter_think_blocks(generate_fallback())), mimetype="text/plain")




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
        temperature = model_config.get("temperature", 0.6)
    
    generation_params = {
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": kwargs.pop("top_p", model_config.get("top_p", 0.85)),
        "top_k": kwargs.pop("top_k", model_config.get("top_k", 30)),
        "repeat_penalty": kwargs.pop("repeat_penalty", model_config.get("repeat_penalty", 1.15)),
        "stream": stream,
        **kwargs
    }
    
    with llm_lock:
        try:
            response = llm.create_chat_completion(**generation_params)
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
        temperature = simple_model_config.get("temperature", 0.6)
    
    generation_params = {
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": kwargs.pop("top_p", simple_model_config.get("top_p", 0.85)),
        "top_k": kwargs.pop("top_k", simple_model_config.get("top_k", 30)),
        "repeat_penalty": kwargs.pop("repeat_penalty", simple_model_config.get("repeat_penalty", 1.15)),
        "stream": stream,
        **kwargs
    }
    
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
    print("[Aura-LLM] 🚀 Starting Aura LLM Container (Modular Architecture)")
    print("[Aura-LLM] 📋 Available modes:")
    print("  - CASUAL: Simple greetings")
    print("  - THINKER: Knowledge queries with RAG")
    print("  - TRIAGE: Hardcoded diagnostic system")
    print("  - CLINICIAN: RAG-powered intelligent diagnosis (framework)")

    app.run(host='0.0.0.0', port=11434, debug=False)


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

# === Model Config (Optimized for Orin NX) ===
MODEL_PATH = os.getenv("MODEL_PATH", "/models/Llama-3.2-3B-Instruct-Q4_K_M.gguf")
N_CTX = int(os.getenv("N_CTX", "2048"))  # Increased for medical mode with RAG guidelines

model_config = {
    "model_path": MODEL_PATH,
    "n_ctx": N_CTX,
    "n_gpu_layers": -1,      # All layers on GPU (Orin NX has sufficient VRAM)
    "n_threads": 6,          # Increased from 4 (Orin NX has 6 performance cores)
    "chat_format": os.getenv("CHAT_FORMAT", "llama-3"),
    "use_mlock": True,
    "use_mmap": True,
    "verbose": False,
    # Speed optimizations - applied globally
    "temperature": float(os.getenv("LLM_TEMPERATURE", "0.6")),
    "top_p": float(os.getenv("LLM_TOP_P", "0.85")),
    "top_k": int(os.getenv("LLM_TOP_K", "30")),
    "repeat_penalty": float(os.getenv("LLM_REPEAT_PENALTY", "1.15")),
}

print(f"[LLM] 🚀 Loading model: {MODEL_PATH}")
print(f"[LLM] ⚙️  Config: n_ctx={N_CTX}, n_gpu_layers=-1, n_threads=6")
llm = Llama(**model_config)
print(f"[LLM] ✅ Model loaded successfully")

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
                response = handle_unified_medical_response(prompt, session_id, llm_chat)
                # llm_chat() now returns strings, so response is already extracted
                return jsonify({"response": response})
            except Exception as e:
                print(f"[Container] ❌ Error in unified medical mode (non-streaming): {e}")
                import traceback
                traceback.print_exc()
                
                # CRITICAL: Clear medical mode state to prevent lock
                print(f"[Container] 🔓 Clearing medical mode state due to error")
                state_to_clear = load_state(session_id)
                if 'dynamic_assessment' in state_to_clear:
                    del state_to_clear['dynamic_assessment']
                if 'mode' in state_to_clear:
                    del state_to_clear['mode']
                save_state(state_to_clear, session_id)
                print(f"[Container] ✅ Medical mode cleared - user can try again or ask other questions")
                
                return jsonify({"response": "I'm sorry, I encountered an error processing your medical query. Please try again or consult a healthcare professional."})
        
        else:
            return jsonify({"response": "I'm sorry, I didn't understand that."})
            
    except Exception as e:
        print(f"[Telegram] ❌ Error in chat-simple: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"response": "I'm sorry, there was an error processing your request."})


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
                # Use the unified medical session to process the query
                response = handle_unified_medical_response(prompt, session_id, llm_chat)
                
                # response is already a string from the session
                print(f"[Container] ✅ Got response from unified medical session")
                
                # Wrap in sentence markers for TTS
                yield "<sentence_start>\n"
                yield f"{response}\n"
                yield "<sentence_end>\n"
            except Exception as e:
                print(f"[Container] ❌ Error in unified medical mode: {e}")
                import traceback
                traceback.print_exc()
                
                # CRITICAL: Clear medical mode state to prevent lock
                print(f"[Container] 🔓 Clearing medical mode state due to error")
                state_to_clear = load_state(session_id)
                if 'dynamic_assessment' in state_to_clear:
                    del state_to_clear['dynamic_assessment']
                if 'mode' in state_to_clear:
                    del state_to_clear['mode']
                save_state(state_to_clear, session_id)
                print(f"[Container] ✅ Medical mode cleared - user can try again or ask other questions")
                
                # Return error message
                yield f"<sentence_start>\nI'm sorry, I encountered an error processing your medical query. Please try again or consult a healthcare professional.\n<sentence_end>\n"

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


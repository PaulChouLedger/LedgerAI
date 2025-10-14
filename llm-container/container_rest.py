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

# Import triage utilities for centralized validation
from triage import (
    apply_synonym_expansion, normalize_yes_no_response, get_generic_onset_answers,
    match_flexible_time, tokenize, get_steps, load_state, save_state, MIN_MATCH
)

# Import modular conversation modes
from router import route_prompt, ConversationMode, format_mode_info
from casual import handle_casual, stream_casual_response
from thinker import handle_thinker
from triage import detect_condition, process_triage_step, generate_triage_completion, load_state, save_state, get_intro, get_steps, apply_synonym_expansion, normalize_text, substitute_name, TRIAGE_DEFS
from clinician import ClinicianSession, is_clinician_trigger, create_clinician_session

# RAG functionality moved to separate RAG container (port 11435)
RAG_SERVICE_URL = "http://localhost:11435"

app = Flask(__name__)
load_dotenv()

# === Thread Safety ===
llm_lock = threading.Lock()

# === Model Config ===
MODEL_PATH = os.getenv("MODEL_PATH", "/models/Llama-3.2-3B-Instruct-Q4_K_M.gguf")
N_CTX = int(os.getenv("N_CTX", "2048"))

# Model configuration to minimize thinking behavior
model_config = {
    "model_path": MODEL_PATH,
    "n_ctx": N_CTX,
    "n_gpu_layers": 32,
    "n_threads": 4,
    "chat_format": os.getenv("CHAT_FORMAT", "llama-3"),
    "use_mlock": True,
    "use_mmap": True,
    "verbose": False,
    "temperature": float(os.getenv("LLM_TEMPERATURE", "0.7")),  # Lower for more focused responses
    "top_p": float(os.getenv("LLM_TOP_P", "0.9")),             # Reduce creativity
    "top_k": int(os.getenv("LLM_TOP_K", "40")),                # Limit vocabulary choices
    "repeat_penalty": float(os.getenv("LLM_REPEAT_PENALTY", "1.1")),  # Discourage repetition
}

# Llama 3.2 models don't use thinking tags - no special configuration needed

llm = Llama(**model_config)

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
        mode, updated_state = route_prompt(prompt_norm, state, session_id)
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
        
        elif mode == ConversationMode.CLINICIAN:
            clinician = create_clinician_session(session_id, prompt, llm_chat)
            opening = clinician.start_session()
            return jsonify({"response": opening})
        
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
    mode, updated_state = route_prompt(prompt_norm, state, session_id)
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

                raw_q = substitute_name(steps[0].get('question', ''), updated_state.get('user_name'))
                q_nlg = nlg_rewrite(raw_q, "question", {
                    "name": updated_state.get("user_name"),
                    "condition": condition,
                    "key": steps[0].get('key'),
                    "allowed_answers": list(steps[0].get('answers', {}).keys())
                }, updated_state.get("phrasing_history"), llm_chat_once)
                yield f"<sentence_start>\n{q_nlg}\n<sentence_end>\n"

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
                    yield f"<sentence_start>\nI'm sorry, there was an error processing your triage.\n<sentence_end>\n"
            # Filter think blocks at container level
            return Response(stream_with_context(filter_think_blocks(generate_triage_continue())), mimetype="text/plain")

    elif mode == ConversationMode.CLINICIAN:
        def generate_clinician():
            clinician = create_clinician_session(session_id, prompt, llm_chat)
            opening = clinician.start_session()
            yield f"<sentence_start>\n{opening}\n<sentence_end>\n"
        # Filter think blocks at container level
        return Response(stream_with_context(filter_think_blocks(generate_clinician())), mimetype="text/plain")

    else:
        # Fallback
        def generate_fallback():
            yield "<sentence_start>\nHello! I'm AuraVision, your friendly personal assistant. How can I help you today?\n<sentence_end>\n"
        # Filter think blocks at container level
        return Response(stream_with_context(filter_think_blocks(generate_fallback())), mimetype="text/plain")


# === Stream Filtering (Simplified - Llama models don't use think tags) ===
def filter_think_blocks(generator):
    """
    Simple pass-through filter for models that don't use think tags
    Llama models generate clean responses without internal reasoning
    """
    for token in generator:
        if token and token.strip():
            yield token

# === Centralized Answer Validation ===

def match_answer_option(ans_norm: str, valid_map: Dict[str, str], use_synonyms: bool = True, key: str = None) -> Tuple[Optional[str], float]:
    """Match answer to options with fuzzy matching and typo correction"""
    ans_expanded = apply_synonym_expansion(ans_norm) if use_synonyms else ans_norm

    # Normalize yes/no first
    normalized_response = normalize_yes_no_response(ans_expanded)
    if normalized_response in ["yes", "no"]:
        if "yes" in valid_map and "no" in valid_map:
            return normalized_response, 1.0

    # Generic onset answers
    if key == "onset" and (not valid_map or len(valid_map) == 0):
        valid_map = get_generic_onset_answers()

    # Flexible time matching
    time_match = match_flexible_time(ans_expanded, valid_map)
    if time_match:
        return time_match

    # Improved fuzzy matching with typo correction
    ans_tokens = set(tokenize(ans_expanded))
    best, score = None, 0.0

    for opt in valid_map:
        opt_tokens = set(tokenize(opt))

        # Exact match gets highest score
        if ans_expanded == opt:
            return opt, 1.0

        # Token overlap matching
        overlap = len(ans_tokens & opt_tokens)

        if overlap > 0:
            base_score = overlap / float(len(opt_tokens)) if opt_tokens else 0
            length_bonus = len(opt_tokens) * 0.1

            if overlap == len(ans_tokens) and overlap == len(opt_tokens):
                exact_bonus = 0.5
            elif overlap == len(opt_tokens):
                exact_bonus = 0.3
            else:
                exact_bonus = 0

            final_score = base_score + length_bonus + exact_bonus
        else:
            # Check for common typos and close matches
            typo_score = check_typo_similarity(ans_expanded, opt)
            final_score = typo_score

        if final_score > score:
            best, score = opt, final_score

    return best, score


def check_typo_similarity(ans: str, opt: str) -> float:
    """Check for typo similarity using character-level matching"""
    # Simple character-level similarity for common typos
    if len(ans) == len(opt):
        # Same length - check for single character differences
        diff_count = sum(1 for a, o in zip(ans, opt) if a != o)
        if diff_count <= 1:  # Allow 1 character difference
            return 0.8

    # Check for common typos (e.g., "roght" -> "right")
    common_typos = {
        "roght": "right",
        "recieve": "receive",
        "seperate": "separate",
        "occured": "occurred",
        "definately": "definitely",
    }

    if ans in common_typos and common_typos[ans] == opt:
        return 0.9

    return 0.0


def is_valid_answer(condition: str, key: str, answer: str, state: Dict[str, Any]) -> bool:
    """Validate if answer is acceptable for given question"""
    ans_norm = normalize_text(answer)
    print(f"[Triage] 🔍 Validating answer: key='{key}', ans='{answer}' (norm='{ans_norm}')")

    if key and key.startswith("clarify_") and state.get("pending_clarify") and state["pending_clarify"].get("key") == key:
        opt, score = match_answer_option(ans_norm, state["pending_clarify"].get("answers", {}), use_synonyms=False, key=key)
        print(f"[Triage] 🔍 Clarify validation: opt='{opt}', score={score}, threshold={MIN_MATCH}")
        return opt and score >= MIN_MATCH

    steps = get_steps(condition, state)
    print(f"[Triage] 🔍 Found {len(steps)} steps for condition '{condition}'")

    for s in steps:
        if isinstance(s, dict) and s.get("key") == key:
            answers = s.get("answers", {})
            print(f"[Triage] 🔍 Checking step with key '{key}', answers: {answers}")

            # For onset questions with empty answers, use generic onset validation
            if not answers and key == "onset":
                print(f"[Triage] 🔍 Onset question - using generic time validation")
                generic_onset_answers = get_generic_onset_answers()
                opt, score = match_answer_option(ans_norm, generic_onset_answers, key=key)
                print(f"[Triage] 🔍 Time matching: opt='{opt}', score={score}, threshold={MIN_MATCH}")
                return opt and score >= MIN_MATCH

            # For other empty answers, accept any answer (rare case)
            if not answers:
                print(f"[Triage] ⚠️ Empty answers for '{key}' - accepting any answer")
                return True

            opt, score = match_answer_option(ans_norm, answers, key=key)
            print(f"[Triage] 🔍 Answer matching: opt='{opt}', score={score}, threshold={MIN_MATCH}")
            return opt and score >= MIN_MATCH

    print(f"[Triage] ❌ No step found with key '{key}'")
    return False


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


def llm_chat(messages, **kwargs):
    """Wrapper for LLM chat completion with thread safety"""
    with llm_lock:
        try:
            return llm.create_chat_completion(messages=messages, **kwargs)
        except Exception as e:
            print(f"[LLM] ❌ Error in llm_chat: {e}")
            return {"choices": [{"message": {"content": ""}}]}


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


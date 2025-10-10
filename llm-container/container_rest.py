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

# Import modular conversation modes
from router import route_prompt, ConversationMode, format_mode_info
from casual import handle_casual, stream_casual_response
from thinker import handle_thinker
from triage import detect_condition, process_triage_step, generate_triage_completion, load_state, save_state, get_intro, get_steps, apply_synonym_expansion, normalize_text, substitute_name
from clinician import ClinicianSession, is_clinician_trigger, create_clinician_session

# RAG functionality moved to separate RAG container (port 11435)
RAG_SERVICE_URL = "http://localhost:11435"

app = Flask(__name__)
load_dotenv()

# === Thread Safety ===
llm_lock = threading.Lock()

# === Model Config ===
MODEL_PATH = os.getenv("MODEL_PATH", "/models/qwen2.5-1.5b-instruct-q4_0.gguf")
N_CTX = int(os.getenv("N_CTX", "2048"))
llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=N_CTX,
    n_gpu_layers=32,
    n_threads=4,
    chat_format=os.getenv("CHAT_FORMAT", "qwen"),
    use_mlock=True,
    use_mmap=True,
    verbose=False,
)

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
@app.route("/chat-simple", methods=["POST"])
def chat_simple():
    """Non-streaming chat endpoint for Telegram bot"""
    data = request.get_json()
    prompt = data.get("prompt", "").strip()
    session_id = data.get("chat_id", "telegram_session")
    reset = data.get("reset", False)
    
    if reset:
        # Clear session state
        state = {"condition": None, "step_index": 0, "answers": [], "flags": {},
                "last_key": None, "user_name": None, "active_pathway": None, 
                "entered_pathway": False, "detailed_symptoms": [], 
                "original_complaint": None, "expanded_prompt": None}
        save_state(state, session_id)
        return jsonify({"response": "Session reset. Start again with your symptoms."})
    
    if not prompt:
        return jsonify({"response": "Please describe your symptoms."})
    
    # Process the prompt and return a single response
    try:
        state = load_state(session_id)
        
        # Check if there's an active triage session first
        if state.get("condition"):
            # Continue existing triage - don't detect new conditions
            question, updated_state = process_triage_step(prompt, state, session_id)
            return jsonify({"response": question})
        else:
            # No active triage - check for new conditions
            condition = detect_condition(prompt, session_id)

            if condition:
                # New triage session - initialize state and ask first question
                state.update({"condition": condition, "step_index": 0, "answers": [], "flags": {},
                             "last_key": None, "user_name": None, "active_pathway": None,
                             "entered_pathway": False, "detailed_symptoms": [prompt],
                             "original_complaint": prompt, "expanded_prompt": prompt})
                save_state(state, session_id)

                # Get intro and first question
                steps = get_steps(condition, state)
                intro = substitute_name(TRIAGE_DEFS[condition].get("intro", ""), state.get("user_name"))
                first_question = substitute_name(steps[0].get('question', ''), state.get('user_name'))

                response = ""
                if intro:
                    response += intro + " "
                response += first_question

                return jsonify({"response": response})
            else:
                # Casual conversation - no triage active
                casual_responses = [
                    "Hello! How can I help you today?",
                    "Hi there! What can I do for you?",
                    "Good to see you! How are you feeling?",
                    "Hello! I'm here to help with any medical concerns you might have.",
                    "Hi! Feel free to describe any symptoms you're experiencing."
                ]
                import random
                return jsonify({"response": random.choice(casual_responses)})
            
    except Exception as e:
        print(f"[Aura-LLM] ❌ Error in chat-simple: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"response": "I'm sorry, there was an error processing your request."})


# === Chat endpoint (modular architecture) ===
@app.route("/chat", methods=["POST"])
def chat():
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
            return Response(stream_with_context(generate_reset()), mimetype="text/plain")

    # Route to appropriate mode
    state = load_state(session_id)
    mode, updated_state = route_prompt(prompt_norm, state, session_id)
    save_state(updated_state, session_id)

    print(f"[Aura-LLM] 🎯 Routed to mode: {mode.upper()}")

    # Dispatch to mode handler
    if mode == ConversationMode.CASUAL:
        return stream_casual_response(prompt_norm, session_id)

    elif mode == ConversationMode.THINKER:
        def generate_thinker():
            for token in handle_thinker(prompt_norm, llm_chat, session_id):
                yield token
        return Response(stream_with_context(generate_thinker()), mimetype="text/plain")

    elif mode == ConversationMode.TRIAGE:
        # Continue existing triage
        if updated_state.get("condition") and not updated_state.get('is_new_triage'):
            def generate_triage_continue():
                try:
                    question, final_state = process_triage_step(prompt, updated_state, session_id)
                    save_state(final_state, session_id)
                    yield f"<sentence_start>\n{question}\n<sentence_end>\n"
                except Exception as e:
                    print(f"[Aura-LLM] ❌ Error in triage: {e}")
                    import traceback
                    traceback.print_exc()
                    yield f"<sentence_start>\nI'm sorry, there was an error processing your triage.\n<sentence_end>\n"
            return Response(stream_with_context(generate_triage_continue()), mimetype="text/plain")

        # NEW triage - ask first question (don't process initial complaint as answer)
        else:
            condition = updated_state.get('condition')
            steps = get_steps(condition, updated_state)
            p_expanded = apply_synonym_expansion(normalize_text(prompt))

            # Initialize triage state (OLD LOGIC - step_index = 1, not 0!)
            updated_state.update({
                "condition": condition,
                "step_index": 1,  # We're asking step 0, so next will be step 1
                "answers": [],
                "flags": {},
                "last_key": steps[0].get("key"),
                "active_pathway": None,
                "entered_pathway": False,
                "phrasing_history": updated_state.get("phrasing_history", []),
                "original_complaint": prompt,
                "expanded_prompt": p_expanded,
                "detailed_symptoms": updated_state.get("detailed_symptoms", [])
            })
            save_state(updated_state, session_id)

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

            return Response(stream_with_context(generate_new_triage()), mimetype="text/plain")

    elif mode == ConversationMode.CLINICIAN:
        def generate_clinician():
            clinician = create_clinician_session(session_id, prompt)
            opening = clinician.start_session()
            yield f"<sentence_start>\n{opening}\n<sentence_end>\n"
        return Response(stream_with_context(generate_clinician()), mimetype="text/plain")

    else:
        # Fallback
        def generate_fallback():
            yield "<sentence_start>\nHello! I'm AuraVision, your friendly personal assistant. How can I help you today?\n<sentence_end>\n"
        return Response(stream_with_context(generate_fallback()), mimetype="text/plain")


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

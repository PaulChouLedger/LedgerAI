# === container_rest.py — Aura deterministic triage
# (organ-system JSON-driven, per-step severity + fuzzy matching + synonym expansion
#  + SOAP-style clinician recap + JSON-based name placeholders with multi-word name support
#  + priority keys + outcomes inheritance + pathway selection + clarify routing + debug logging + casual mode) ===

from flask import Flask, request, jsonify, stream_with_context, Response
from llama_cpp import Llama
from dotenv import load_dotenv
import os, re, json, string, threading, time
from datetime import datetime, timedelta
from glob import glob
from nlg import rewrite as nlg_rewrite

# RAG module - will be initialized after LLM warm-up
RAG_AVAILABLE = False
RAG_INITIALIZED = False

# RAG functions (loaded only when RAG is initialized)
get_rag_fn = None
search_medical_info_fn = None
smart_search_medical_info_fn = None

def initialize_rag_safely():
    """Initialize RAG system safely after LLM is loaded"""
    global RAG_AVAILABLE, RAG_INITIALIZED, get_rag_fn, search_medical_info_fn, smart_search_medical_info_fn
    
    # Prevent multiple initializations
    if RAG_INITIALIZED:
        print("[Aura-LLM] ℹ️ RAG already initialized, skipping...")
        return
        
    # Add initialization lock to prevent race conditions
    import threading
    init_lock = getattr(initialize_rag_safely, '_lock', None)
    if init_lock is None:
        init_lock = threading.Lock()
        initialize_rag_safely._lock = init_lock
    
    with init_lock:
        if RAG_INITIALIZED:
            print("[Aura-LLM] ℹ️ RAG already initialized (locked), skipping...")
            return
        
    try:
        print("[Aura-LLM] 🔍 Initializing RAG system...")
        time.sleep(3)  # Reduced wait time
        
        from rag import get_rag, search_medical_info, smart_search_medical_info
        get_rag_fn = get_rag
        search_medical_info_fn = search_medical_info
        smart_search_medical_info_fn = smart_search_medical_info
        
        RAG_AVAILABLE = True
        RAG_INITIALIZED = True
        print("[Aura-LLM] ✅ RAG module loaded and initialized")
    except ImportError as e:
        print(f"[Aura-LLM] ⚠️ RAG module not available: {e}")
        RAG_AVAILABLE = False
    except Exception as e:
        print(f"[Aura-LLM] ❌ RAG initialization failed: {e}")
        RAG_AVAILABLE = False

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

MIN_MATCH = float(os.getenv("TRIAGE_MIN_MATCH", "0.6"))

STATE_DIR = os.getenv("TRIAGE_STATE_DIR", "/app/state")
os.makedirs(STATE_DIR, exist_ok=True)
def get_state_path(session_id: str | None):
    if session_id:
        safe = re.sub(r"[^A-Za-z0-9_.-]", "_", session_id)
        return os.path.join(STATE_DIR, f"triage_state_{safe}.json")
    return os.path.join(STATE_DIR, "triage_state.json")

# === Load triage defs ===
TRIAGE_DEFS = {}
triage_dir = os.getenv("TRIAGE_DEFINITIONS_DIR", "/app/triage_defs")
print(f"[Aura-LLM] 🔍 Loading triage definitions from: {triage_dir}")
if os.path.isdir(triage_dir):
    for path in glob(os.path.join(triage_dir, "*.json")):
        try:
            with open(path, "r") as f:
                data = json.load(f)
                TRIAGE_DEFS.update(data)
                print(f"[Aura-LLM] ✅ Loaded triage defs: {os.path.basename(path)}")
                print(f"[Aura-LLM] 🔍 Loaded conditions: {list(data.keys())}")
        except Exception as e:
            print(f"[Aura-LLM] ⚠️ Failed to load triage defs {path}: {e}")
else:
    print(f"[Aura-LLM] ❌ Triage definitions directory not found: {triage_dir}")
print(f"[Aura-LLM] 🔍 Total loaded conditions: {len(TRIAGE_DEFS)}")

# === State helpers ===
def load_state(session_id: str | None = None):
    path = get_state_path(session_id)
    if os.path.exists(path):
        try:
            return json.load(open(path, "r"))
        except Exception:
            pass
    return {
        "condition": None, "step_index": 0, "answers": [],
        "flags": {}, "last_key": None, "user_name": None,
        "active_pathway": None, "entered_pathway": False,
        "updated_at": None,
        "phrasing_history": []
    }

def save_state(state, session_id: str | None = None):
    state["updated_at"] = datetime.utcnow().isoformat()
    path = get_state_path(session_id)
    json.dump(state, open(path, "w"))

def triage_is_stale(state, minutes=5):
    try:
        return datetime.utcnow() - datetime.fromisoformat(state.get("updated_at") or "") > timedelta(minutes=minutes)
    except Exception:
        return False

# === Utils ===
def normalize_text(t): 
    # Remove emojis and other Unicode characters, keep only ASCII letters, numbers, and spaces
    import re
    # Remove emojis and non-ASCII characters
    t = re.sub(r'[^\x00-\x7F]+', '', t)
    # Remove punctuation and convert to lowercase
    return t.lower().translate(str.maketrans("", "", string.punctuation)).strip()
def tokenize(t): return normalize_text(t).split()

# === Synonym expansion ===
def apply_synonym_expansion(text):
    """Apply synonym expansion to normalize medical terms"""
    # Load synonyms from all files
    synonym_files = [
        "/app/synonyms/gi_synonyms.json",
        "/app/synonyms/gu_synonyms.json", 
        "/app/synonyms/neuro_synonyms.json",
        "/app/synonyms/cardio_synonyms.json",
        "/app/synonyms/derm_synonyms.json",
        "/app/synonyms/endocrine_synonyms.json",
        "/app/synonyms/resp_synonyms.json",
        "/app/synonyms/renal_synonyms.json"
    ]
    
    synonyms = {}
    for file_path in synonym_files:
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r') as f:
                    file_synonyms = json.load(f)
                    synonyms.update(file_synonyms)
            except Exception as e:
                print(f"[Aura-LLM] ⚠️ Failed to load synonyms from {file_path}: {e}")
    
    # Apply synonym expansion - prioritize longer phrases first, stop after first match
    expanded_text = text
    # Sort variations by length (longest first) to avoid partial replacements
    all_variations = []
    for standard_term, variations in synonyms.items():
        for variation in variations:
            all_variations.append((len(variation), variation, standard_term))
    
    # Sort by length descending
    all_variations.sort(key=lambda x: x[0], reverse=True)
    
    for length, variation, standard_term in all_variations:
        # Use word boundaries to avoid partial matches (e.g., "weak" in "weakness")
        pattern = r'\b' + re.escape(variation) + r'\b'
        if re.search(pattern, expanded_text, re.IGNORECASE):
            print(f"[Aura-LLM] 🔄 Synonym expansion: '{variation}' -> '{standard_term}'")
            # Use case-insensitive replacement with word boundaries
            expanded_text = re.sub(pattern, standard_term, expanded_text, flags=re.IGNORECASE)
            # Stop after first match to avoid nested replacements
            break
    
    return expanded_text

def substitute_name(text, user_name):
    if not text: return text
    if "{name}" in text:
        if user_name:
            return text.replace("{name}", user_name)
        else:
            # Remove {name} and clean up any resulting punctuation issues
            text = text.replace("{name}, ", "").replace("{name}", "")
            # Clean any leading stray punctuation (e.g., leading comma when name missing)
            cleaned = re.sub(r"^[,;:\-]\s*", "", text)
            cleaned = re.sub(r"\s{2,}", " ", cleaned).strip()
            return cleaned
    # Clean any leading stray punctuation (e.g., leading comma when name missing)
    cleaned = re.sub(r"^[,;:\-]\s*", "", text)
    cleaned = re.sub(r"\s{2,}", " ", cleaned).strip()
    return cleaned


# === Name extraction ===
def extract_name(prompt):
    m = re.search(r"(?:my name is|i am|i'm|my name's)\s+([A-Za-z .'-]+)", prompt, re.IGNORECASE)
    if not m: 
        return None
    raw = m.group(1).strip()
    
    # Check if the extracted text matches any medical terms in synonym files
    # This is more dynamic than a hardcoded list
    raw_lower = raw.lower()
    
    # Load all synonym files to check against
    import os
    import json
    
    synonym_files = [
        "synonyms/cardio_synonyms.json",
        "synonyms/derm_synonyms.json", 
        "synonyms/endocrine_synonyms.json",
        "synonyms/gi_synonyms.json",
        "synonyms/gu_synonyms.json",
        "synonyms/neuro_synonyms.json",
        "synonyms/renal_synonyms.json",
        "synonyms/resp_synonyms.json"
    ]
    
    # Check if the extracted text matches any medical terms
    for synonym_file in synonym_files:
        try:
            file_path = os.path.join("synonyms", synonym_file)
            if os.path.exists(file_path):
                with open(file_path, 'r') as f:
                    synonyms = json.load(f)
                    # Check if any key or value in the synonyms matches
                    for key, values in synonyms.items():
                        if (raw_lower == key.lower() or 
                            any(raw_lower == val.lower() for val in values) or
                            any(raw_lower in val.lower() for val in values)):
                            return None  # This is a medical term, not a name
        except Exception:
            continue  # Skip if file doesn't exist or can't be read
    
    # Split on medical keywords or common separators, but be more careful with "and"
    raw = re.split(r"\b(having|with|experiencing|suffering|complaining|reporting)\b|,|\.", raw, 1)[0].strip()
    # Handle "and" more carefully - only split if it's followed by medical terms
    if " and " in raw:
        # Check if "and" is followed by medical terms
        and_parts = raw.split(" and ", 1)
        if len(and_parts) > 1:
            after_and = and_parts[1].lower()
            medical_after_and = any(term in after_and for term in ["pain", "ache", "headache", "chest", "abdominal", "stomach", "nausea", "dizzy", "fever", "cough", "having", "experiencing", "suffering"])
            if medical_after_and:
                raw = and_parts[0].strip()
    
    # Also handle "I'm" patterns
    if " i'm " in raw.lower() or raw.lower().endswith(" i'm"):
        raw = raw.split(" i'm", 1)[0].strip()
    if " im " in raw.lower() or raw.lower().endswith(" im"):
        raw = raw.split(" im", 1)[0].strip()
    
    parts = raw.split()
    if not parts or len(parts) > 3: 
        return None
    blacklist = {"pain","cough","fever","dizziness","weakness","nausea","vomiting","abdominal","chest","done","finished","complete","over"}
    if any(p.lower() in blacklist for p in parts): 
        return None
    fixed = [p.capitalize() for p in parts]
    result = " ".join(fixed)
    print(f"[Aura-LLM] ✅ Extracted name: '{result}'")
    return result

# === Condition detection ===
def detect_condition(prompt, session_id: str | None = None):
    p = normalize_text(prompt)
    print(f"[Aura-LLM] 🔍 Original prompt: '{prompt}'")
    print(f"[Aura-LLM] 🔍 Normalized prompt: '{p}'")
    name = extract_name(prompt)
    if name:
        state = load_state(session_id)
        state["user_name"] = name
        save_state(state, session_id)
        print(f"[Aura-LLM] 👤 User name set: {name}")
    
    # Check for casual greetings first - don't trigger triage for these
    # Only block if it's JUST a greeting without any medical content
    casual_greetings = ["hello", "hi", "hey", "good morning", "good afternoon", "good evening", "hello aura", "hi aura", "hey aura", "good morning aura", "good afternoon aura", "good evening aura"]
    if any(greeting in p for greeting in casual_greetings):
        # Check if there are any medical symptoms mentioned
        medical_keywords = ["pain", "hurt", "ache", "symptom", "problem", "issue", "concern", "worried", "sick", "ill", "unwell"]
        has_medical_content = any(keyword in p for keyword in medical_keywords)
        
        if not has_medical_content:
            print(f"[Aura-LLM] 💬 Casual greeting detected: '{p}' -> no triage trigger")
            return None
        else:
            print(f"[Aura-LLM] 💬 Greeting with medical content detected: '{p}' -> proceeding with triage")
    
    # Apply synonym expansion
    p_expanded = apply_synonym_expansion(p)
    print(f"[Aura-LLM] 🔄 Expanded prompt: '{p_expanded}'")
    print(f"[Aura-LLM] 🔍 Checking for triage triggers in: '{p_expanded}'")
    
    # Initialize detailed symptoms array for main complaint tracking
    state = load_state(session_id)
    if "detailed_symptoms" not in state:
        state["detailed_symptoms"] = []
    
    # Add the initial expanded prompt to the array
    if p_expanded and p_expanded not in state["detailed_symptoms"]:
        state["detailed_symptoms"].append(p_expanded)
        print(f"[Aura-LLM] 📝 Detailed symptoms array: {state['detailed_symptoms']}")
    
    save_state(state, session_id)
        
    for cond, data in TRIAGE_DEFS.items():
        triggers = data.get("triggers", [])
        for trig in triggers:
            trig_norm = normalize_text(trig)
            # Try exact match first
            if trig_norm in p_expanded:
                print(f"[Aura-LLM] ✅ Detected condition: {cond} (exact match: '{trig}')")
                # Use detailed_symptom from JSON if available, otherwise use expanded prompt
                if "detailed_symptom" in TRIAGE_DEFS[cond]:
                    detailed_symptom = TRIAGE_DEFS[cond]["detailed_symptom"]
                    if detailed_symptom not in state["detailed_symptoms"]:
                        state["detailed_symptoms"].append(detailed_symptom)
                        print(f"[Aura-LLM] 📝 Detailed symptoms array: {state['detailed_symptoms']}")
                        save_state(state, session_id)
                return cond
            # Try fuzzy match with 0.6 threshold
            ans_tokens = set(tokenize(p_expanded))
            trig_tokens = set(tokenize(trig_norm))
            overlap = len(ans_tokens & trig_tokens) / float(len(trig_tokens)) if trig_tokens else 0
            if overlap >= MIN_MATCH:
                print(f"[Aura-LLM] ✅ Detected condition: {cond} (fuzzy match: '{trig}' = {overlap:.2f})")
                # Use detailed_symptom from JSON if available, otherwise use expanded prompt
                if "detailed_symptom" in TRIAGE_DEFS[cond]:
                    detailed_symptom = TRIAGE_DEFS[cond]["detailed_symptom"]
                    if detailed_symptom not in state["detailed_symptoms"]:
                        state["detailed_symptoms"].append(detailed_symptom)
                        print(f"[Aura-LLM] 📝 Detailed symptoms array: {state['detailed_symptoms']}")
                        save_state(state, session_id)
                return cond
    return None

# === Answer matching ===
def match_answer_option(prompt, answers):
    """Match user input against answer options with fuzzy matching"""
    # Apply synonym expansion to the user input
    prompt_expanded = apply_synonym_expansion(prompt)
    
    best_match = None
    best_score = 0
    
    for option in answers.keys():
        option_norm = normalize_text(option)
        
        # Try exact match first
        if option_norm in prompt_expanded:
            return option, 1.0
            
        # Try fuzzy match
        prompt_tokens = set(tokenize(prompt_expanded))
        option_tokens = set(tokenize(option_norm))
        
        if option_tokens:
            overlap = len(prompt_tokens & option_tokens) / float(len(option_tokens))
            if overlap > best_score:
                best_score = overlap
                best_match = option
    
    return best_match, best_score

def normalize_yes_no_response(text):
    """Normalize natural yes/no responses to standard yes/no"""
    text_lower = text.lower().strip()
    
    # Check for negative responses FIRST (more specific patterns)
    if any(phrase in text_lower for phrase in [
        "no", "nope", "nah", "not", "don't", "do not", "haven't", "have not",
        "i don't", "i do not", "i haven't", "i have not",
        "i don't have", "i do not have", "i don't feel", "i do not feel",
        "i don't experience", "i do not experience", "i am not", "i'm not"
    ]):
        return "no"
    
    # Check for specific negative patterns that might be confused with positive
    if text_lower in ["i dont", "i don't", "i do not", "i havent", "i haven't", "i have not"]:
        return "no"
    
    # Then check for positive responses (less specific patterns)
    if any(phrase in text_lower for phrase in [
        "yes", "yea", "yeah", "yep", "yup", "sure", "ok", "okay",
        "i do", "i have", "i am", "i feel", "i experience",
        "i do have", "i do feel", "i do experience",
        "i have been", "i am having", "i am experiencing",
        "i do have been", "i do have had", "i do have been having"
    ]):
        return "yes"
    
    return text

def get_generic_onset_answers():
    """Get standard onset answers that apply to all conditions"""
    return {
        "within the last hour": "emergency",
        "within the last few hours": "emergency", 
        "today": "urgent",
        "yesterday": "urgent",
        "a few days ago": "urgent",
        "a week ago": "non_urgent",
        "unknown": "urgent"
    }

def match_flexible_time(ans_expanded, valid_map):
    """Match flexible time patterns like '3 hours ago', '5 days ago', etc."""
    import re
    
    # Pattern to match numerical time expressions
    time_pattern = r'(\d+)\s*(minute|hour|day|week|month)s?\s*ago'
    match = re.search(time_pattern, ans_expanded, re.IGNORECASE)
    
    if not match:
        return None
    
    number = int(match.group(1))
    unit = match.group(2).lower()
    
    # Map time units to severity based on duration
    if unit in ['minute', 'hour']:
        # Minutes and hours are emergency
        if unit == 'minute' or (unit == 'hour' and number <= 6):
            return "within the last hour", 1.0
        elif unit == 'hour' and number <= 12:
            return "within the last few hours", 1.0
        else:
            return "today", 1.0
    elif unit == 'day':
        # Days are urgent
        if number == 1:
            return "yesterday", 1.0
        elif number <= 7:
            return "a few days ago", 1.0
        else:
            return "a week ago", 1.0
    elif unit == 'week':
        # Weeks are non-urgent
        if number == 1:
            return "a week ago", 1.0
        else:
            return "last week", 1.0
    elif unit == 'month':
        # Months are non-urgent
        return "last week", 1.0
    
    return None

def match_answer_option(ans_norm, valid_map, use_synonyms=True, key=None):
    # Apply synonym expansion to the answer (skip for pending_clarify)
    ans_expanded = apply_synonym_expansion(ans_norm) if use_synonyms else ans_norm
    
    # First try to normalize yes/no responses
    normalized_response = normalize_yes_no_response(ans_expanded)
    if normalized_response in ["yes", "no"]:
        # Check if the valid_map contains yes/no options
        if "yes" in valid_map and "no" in valid_map:
            return normalized_response, 1.0
    
    # For onset questions, use generic onset answers if not explicitly defined
    if key == "onset" and (not valid_map or len(valid_map) == 0):
        valid_map = get_generic_onset_answers()
    
    # Check for flexible time patterns (e.g., "3 hours ago", "5 days ago", "2 weeks ago")
    time_match = match_flexible_time(ans_expanded, valid_map)
    if time_match:
        return time_match
    
    ans_tokens = set(tokenize(ans_expanded))
    best, score = None, 0.0
    for opt in valid_map:
        opt_tokens = set(tokenize(opt))
        overlap = len(ans_tokens & opt_tokens)
        
        # Calculate score with length preference for more specific matches
        if overlap > 0:
            # Base score is the overlap ratio
            base_score = overlap / float(len(opt_tokens)) if opt_tokens else 0
            # Bonus for longer matches (more specific) - multiply by length
            length_bonus = len(opt_tokens) * 0.1
            # Bonus for exact matches or very high overlap
            if overlap == len(ans_tokens) and overlap == len(opt_tokens):
                exact_bonus = 0.5  # Exact match bonus
            elif overlap == len(opt_tokens):
                exact_bonus = 0.3  # Answer contains all option tokens
            else:
                exact_bonus = 0
                
            final_score = base_score + length_bonus + exact_bonus
        else:
            final_score = 0
            
        if final_score > score: 
            best, score = opt, final_score
            
    return best, score

def match_all_options(ans_norm, valid_map):
    # Apply synonym expansion to the answer
    ans_expanded = apply_synonym_expansion(ans_norm)
    
    ans_tokens = set(tokenize(ans_expanded))
    matches = []
    for opt in valid_map:
        opt_tokens = set(tokenize(opt))
        overlap = len(ans_tokens & opt_tokens) / float(len(opt_tokens)) if opt_tokens else 0
        if overlap >= MIN_MATCH:
            matches.append(opt)
    
    # Special handling for compound answers like "nausea and sensitivity to sound"
    if not matches and "and" in ans_expanded:
        # Try to match individual components
        components = [comp.strip() for comp in ans_expanded.split("and")]
        for comp in components:
            comp_tokens = set(tokenize(comp))
            for opt in valid_map:
                opt_tokens = set(tokenize(opt))
                overlap = len(comp_tokens & opt_tokens) / float(len(opt_tokens)) if opt_tokens else 0
                if overlap >= MIN_MATCH and opt not in matches:
                    matches.append(opt)
    
    return matches

def add_phrasing_fingerprint(state, text):
    """Add a short fingerprint of the text to phrasing_history to avoid repetition."""
    if not text: return
    # Create a simple fingerprint (first 100 chars, normalized)
    fp = re.sub(r"\s+", " ", text.strip().lower())[:100]
    if fp and fp not in (state.get("phrasing_history") or []):
        state.setdefault("phrasing_history", []).append(fp)
        # Keep only last 10 to avoid memory bloat
        if len(state["phrasing_history"]) > 10:
            state["phrasing_history"] = state["phrasing_history"][-10:]

def llm_chat_once(messages, gen_kwargs):
    """Non-stream single completion via llama_cpp with thread safety."""
    with llm_lock:
        try:
            resp = llm.create_chat_completion(messages=messages, **{k: v for k, v in gen_kwargs.items() if v is not None})
            return resp
        except Exception as e:
            print(f"[LLM] ❌ Error in llm_chat_once: {e}")
            return {"choices":[{"message":{"content":""}}]}

def get_steps(cond, state):
    steps = TRIAGE_DEFS[cond].get("steps", [])
    if state.get("active_pathway") and "pathways" in TRIAGE_DEFS[cond]:
        steps = TRIAGE_DEFS[cond]["pathways"][state["active_pathway"]].get("steps", steps)
    return steps

def get_pathway_detailed_symptom(cond, pathway, state):
    """Get detailed symptom description from pathway definition"""
    # Check if pathway has detailed_symptom defined
    if "pathways" in TRIAGE_DEFS[cond] and pathway in TRIAGE_DEFS[cond]["pathways"]:
        pathway_def = TRIAGE_DEFS[cond]["pathways"][pathway]
        if "detailed_symptom" in pathway_def:
            return pathway_def["detailed_symptom"]
    
    # Generic fallback: generate from condition and pathway name
    cond_name = cond.replace("_", " ")
    pathway_name = pathway.replace("_pathway", "").replace("_", " ")
    return f"{pathway_name} {cond_name}"

def is_valid_answer(cond, key, ans, state):
    ans_norm = normalize_text(ans)
    # Validate inline clarify answers against pending clarify map
    if key and key.startswith("clarify_") and state.get("pending_clarify") and state["pending_clarify"].get("key") == key:
        opt, score = match_answer_option(ans_norm, state["pending_clarify"].get("answers", {}), use_synonyms=False, key=key)
        return opt and score >= MIN_MATCH
    steps = get_steps(cond, state)
    for s in steps:
        if isinstance(s, dict) and s.get("key") == key:
            opt, score = match_answer_option(ans_norm, s.get("answers", {}), key=key)
            return opt and score >= MIN_MATCH
    return False

def update_flags_from_answer(cond, key, ans, state, session_id=None):
    ans_norm = normalize_text(ans)
    # Handle inline clarify answers first
    if key and key.startswith("clarify_") and state.get("pending_clarify") and state["pending_clarify"].get("key") == key:
        opt, score = match_answer_option(ans_norm, state["pending_clarify"].get("answers", {}), use_synonyms=False, key=key)
        if not opt or score < MIN_MATCH: return
        sev = state["pending_clarify"]["answers"][opt]
        if isinstance(sev, str) and sev.endswith("_pathway"):
            state["active_pathway"] = sev
            state["step_index"] = 0
            state["answers"] = []
            state["last_key"] = None
            state["entered_pathway"] = False
            print(f"[Aura-LLM] 🔀 Clarify routed → {sev}")
            
            # Store the clarify answer for potential use in recap building
            # The similarity-based main complaint selection will handle finding the most detailed match
            state["clarify_answer"] = opt
            
            # Add pathway-specific detailed symptom to the array
            # Get detailed symptom from pathway definition or use generic fallback
            detailed_symptom = get_pathway_detailed_symptom(cond, sev, state)
            
            # Add to detailed symptoms array
            if "detailed_symptoms" not in state:
                state["detailed_symptoms"] = []
            if detailed_symptom not in state["detailed_symptoms"]:
                state["detailed_symptoms"].append(detailed_symptom)
                print(f"[Aura-LLM] 📝 Detailed symptoms array: {state['detailed_symptoms']}")
                save_state(state, session_id)
                
        state.pop("pending_clarify", None)
        return
    steps = get_steps(cond, state)
    print(f"[Aura-LLM] 🔍 update_flags_from_answer: cond={cond}, key={key}, ans={ans}")
    for s in steps:
        if isinstance(s, dict) and s.get("key") == key:
            print(f"[Aura-LLM] 🔍 Found matching step: {s}")
            opt, score = match_answer_option(ans_norm, s.get("answers", {}), key=key)
            print(f"[Aura-LLM] 🔍 Matched option: {opt}, score: {score}")
            if not opt or score < MIN_MATCH: 
                print(f"[Aura-LLM] ❌ No valid match found")
                return
            
            # Handle empty answers dictionary (generic onset questions)
            if not s["answers"] or opt not in s["answers"]:
                # Use generic onset answers for empty dictionaries
                if key == "onset":
                    generic_answers = get_generic_onset_answers()
                    sev = generic_answers.get(opt, "urgent")  # Default to urgent if not found
                    print(f"[Aura-LLM] 🔍 Using generic onset mapping: {opt} -> {sev}")
                else:
                    print(f"[Aura-LLM] ❌ No answers defined for key: {key}")
                    return
            else:
                sev = s["answers"][opt]
                print(f"[Aura-LLM] 🔍 Severity/pathway: {sev}")

            # Inline clarify object with followup_question
            if isinstance(sev, dict) and sev.get("followup_question"):
                clarify_key = f"clarify_{key}"
                state["pending_clarify"] = {
                    "key": clarify_key,
                    "question": sev.get("followup_question", ""),
                    "answers": sev.get("answers", {})
                }
                state["last_key"] = clarify_key
                print(f"[Aura-LLM] ❓ Queued inline clarify step for '{key}' → {clarify_key}")
                return

            # Clarify routing → immediate redirection
            if key.startswith("clarify_"):
                if isinstance(sev, str) and sev.endswith("_pathway"):
                    state["active_pathway"] = sev
                    state["step_index"] = 0
                    state["answers"] = []
                    state["last_key"] = None
                    state["entered_pathway"] = False
                    print(f"[Aura-LLM] 🔀 Clarify routed → {sev}")
                return

            # Normal pathway - check if it's a pathway by looking in the pathways section
            print(f"[Aura-LLM] 🔍 Checking pathway routing: sev={sev}, type={type(sev)}")
            is_pathway = isinstance(sev, str) and "pathways" in TRIAGE_DEFS[cond] and sev in TRIAGE_DEFS[cond]["pathways"]
            print(f"[Aura-LLM] 🔍 Is pathway: {is_pathway}")
            if is_pathway:
                state["active_pathway"] = sev
                state["step_index"] = 0
                state["answers"] = []
                state["last_key"] = None
                state["entered_pathway"] = False
                print(f"[Aura-LLM] 🔀 Pathway selected: {sev}")
                
                # Add detailed symptom for pathway - generic approach
                if "detailed_symptoms" not in state:
                    state["detailed_symptoms"] = []
                
                # Create detailed symptom using generic pathway name
                pathway_name = sev.replace("_", " ")
                detailed_symptom = pathway_name
                
                if detailed_symptom not in state["detailed_symptoms"]:
                    state["detailed_symptoms"].append(detailed_symptom)
                    print(f"[Aura-LLM] 📝 Detailed symptoms array: {state['detailed_symptoms']}")
                    save_state(state, session_id)
            else:
                print(f"[Aura-LLM] 🔍 Not a pathway, setting flag: {key}={sev}")
                state["flags"].setdefault(cond, {})[key] = sev

# === Classification ===
def classify_response(cond, flags):
    vals = list(flags.get(cond, {}).values())
    if "emergency" in vals: return "emergency"
    if "urgent" in vals: return "urgent"
    if vals and all(v=="non_urgent" for v in vals): return "non_urgent"
    return "urgent"

# === Recap ===
def pretty_join(parts, conj="and"):
    if not parts: return ""
    # Normalize spaces in each part to prevent multiple spaces
    parts = [re.sub(r'\s+', ' ', part.strip()) for part in parts]
    if len(parts) == 1: return parts[0]
    if len(parts) == 2: return f"{parts[0]} {conj} {parts[1]}"
    return ", ".join(parts[:-1]) + f", {conj} {parts[-1]}"

def build_recap(cond, answers, flags, severity, session_id=None):
    state = load_state(session_id)
    steps = get_steps(cond, state)
    pk = TRIAGE_DEFS[cond].get("priority_keys", [])

    if state.get("active_pathway") and "pathways" in TRIAGE_DEFS[cond]:
        path = TRIAGE_DEFS[cond]["pathways"][state["active_pathway"]]
        steps = path.get("steps", steps)
        pk = path.get("priority_keys", pk)
        print(f"[Aura-LLM] 📝 Recap built from pathway: {state['active_pathway']}")
        

    positives, negatives, priority_positives, priority_negatives = [], [], [], []
    def _strip_prefix(text: str) -> str:
        # Remove leading "You reported/denied" (case-insensitive), extra spaces, and trailing punctuation
        t = re.sub(r"(?i)^\s*you\s+(reported|denied)\s+", "", text or "").strip()
        t = t.rstrip(". ")
        return t
    for s, raw in zip(steps, answers):
        if not isinstance(s, dict): continue
        key, templ = s.get("key"), s.get("recap_template","{answer}")
        valid_map = s.get("answers", {})
        ans_norm = normalize_text(raw)
        opts = match_all_options(ans_norm, valid_map) or []
        if not opts:
            opt_single, _ = match_answer_option(ans_norm, valid_map, key=key)
            if opt_single: opts = [opt_single]
        # Map yes/no to reported/denied; otherwise join multiple options
        if len(opts) == 1 and opts[0] in ("yes", "no"):
            ans_out = "reported" if opts[0] == "yes" else "denied"
        elif opts:
            # For compound answers, use "reported" since they're positive findings
            # Remove redundant overlapping options - improved logic
            clean_opts = []
            for opt in opts:
                # Skip if this option is a subset of another option
                is_redundant = False
                for other_opt in opts:
                    if opt != other_opt and opt in other_opt:
                        is_redundant = True
                        break
                if not is_redundant:
                    clean_opts.append(opt)
            
            # Additional cleanup: remove options that contain words not mentioned by user
            if len(clean_opts) > 1:
                # Find the most specific option (longest) and use that
                clean_opts = [max(clean_opts, key=len)]
            
            # Debug logging for redundancy removal
            print(f"[Aura-LLM] 🔍 Original options: {opts}")
            print(f"[Aura-LLM] 🔍 Clean options: {clean_opts}")
            
            # Special handling for timing questions - don't add "reported" prefix
            if key in ["onset", "when", "timing", "duration"]:
                if clean_opts:
                    ans_out = pretty_join(clean_opts, 'and')
                else:
                    ans_out = pretty_join(opts, 'and')
            else:
                if clean_opts:
                    ans_out = f"reported {pretty_join(clean_opts, 'and')}"
                else:
                    ans_out = f"reported {pretty_join(opts, 'and')}"
        else:
            # If no match found, check if it's a positive or negative response
            normalized = normalize_yes_no_response(raw)
            if normalized == "yes":
                ans_out = f"reported {raw}"
            elif normalized == "no":
                ans_out = "denied"
            else:
                ans_out = raw
        # Handle pathway routing - show actual answer, not pathway names
        if ans_out.endswith("_pathway"):
            # Extract the pathway name without "_pathway" suffix for display
            display_name = ans_out.replace("_pathway", "").replace("_", " ").title()
            line = templ.format(answer=display_name).strip()
        elif re.match(r"^\s*You\s+\{answer\}\s+", templ, flags=re.IGNORECASE) and ans_out not in ("reported", "denied"):
        # If template is of the form "You {answer} X" and ans_out is not reported/denied,
        # rewrite to "You reported X with ans_out" for better readability
            tail = re.sub(r"^\s*You\s+\{answer\}\s+", "", templ, flags=re.IGNORECASE).strip()
            line = f"You reported {tail} with {ans_out}"
        else:
            # Handle special cases for denial answers
            if ans_out.lower() in ["none", "no", "neither"]:
                # For denial answers, we need to determine what was being asked about
                # Look at the question text to determine what was denied
                question_text = s.get("question", "").lower()
                if "chest pain" in question_text and "palpitations" in question_text:
                    line = "denied chest pain and palpitations"
                elif "chest pain" in question_text:
                    line = "denied chest pain"
                elif "palpitations" in question_text:
                    line = "denied palpitations"
                elif "shortness of breath" in question_text:
                    line = "denied shortness of breath"
                elif "fainting" in question_text or "faint" in question_text:
                    line = "denied fainting"
                elif "dizziness" in question_text:
                    line = "denied dizziness"
                else:
                    line = f"denied {ans_out}"
            else:
                # Use the raw answer for display
                line = templ.format(answer=ans_out).strip()
            
        # Debug logging for recap generation
        print(f"[Aura-LLM] 🔍 Recap step: key='{key}', ans_out='{ans_out}', line='{line}', is_priority={key in pk}")

        # Check if this is a denial (none, no, neither answers)
        is_negative = "denied" in line.lower() or ans_out.lower() in ["none", "no", "neither"]

        if key in pk:
            if is_negative:
                priority_negatives.append(_strip_prefix(line))
            else:
                priority_positives.append(_strip_prefix(line))
        elif is_negative:
            negatives.append(_strip_prefix(line))
        else:
            positives.append(_strip_prefix(line))

    parts = []
    # De-duplicate while preserving order
    # De-duplicate while preserving order
    def _dedup(seq):
        seen = set(); out = []
        for item in seq:
            if item and item not in seen:
                seen.add(item); out.append(item)
        return out
    positives = _dedup(positives)
    negatives = _dedup(negatives)
    priority_positives = _dedup(priority_positives)
    priority_negatives = _dedup(priority_negatives)

    # Debug logging for recap categorization
    print(f"[Aura-LLM] 🔍 Recap categorization:")
    print(f"[Aura-LLM] 🔍 Priority positives: {priority_positives}")
    print(f"[Aura-LLM] 🔍 Priority negatives: {priority_negatives}")
    print(f"[Aura-LLM] 🔍 Regular positives: {positives}")
    print(f"[Aura-LLM] 🔍 Regular negatives: {negatives}")
    
    # Extract main complaint using detailed symptoms array
    # Find the most detailed and similar symptom to the original complaint
    original_complaint = state.get("original_complaint", "").lower()
    detailed_symptoms = state.get("detailed_symptoms", [])
    print(f"[Aura-LLM] 📝 Final detailed symptoms array: {detailed_symptoms}")
    
    main_complaint = None
    best_score = 0
    
    if detailed_symptoms and original_complaint:
        # Use the detailed symptoms array to find the best match
        for symptom in detailed_symptoms:
            symptom_lower = symptom.lower()
            # Calculate similarity score based on word overlap
            original_words = set(original_complaint.split())
            symptom_words = set(symptom_lower.split())
            
            # Count overlapping words
            overlap = len(original_words.intersection(symptom_words))
            # Prefer longer matches (more detailed) - increase weight for specificity
            length_bonus = len(symptom_words) * 0.3
            # Bonus for anatomical specificity (arm, leg, side, etc.)
            anatomical_bonus = 0
            if any(word in symptom_lower for word in ['arm', 'leg', 'side', 'right', 'left', 'upper', 'lower']):
                anatomical_bonus = 2.0
            score = overlap + length_bonus + anatomical_bonus
            
            if score > best_score:
                best_score = score
                main_complaint = symptom
    
    # Fallback to condition name if no good match
    if not main_complaint:
        main_complaint = cond.replace("_", " ").replace("suspected", "").strip()
    
    # Separate timing from other symptoms (excluding the main complaint)
    timing_info = []
    other_positives = []
    for pos in priority_positives:
        # Skip the main complaint (could be first item or location-specific complaint)
        if pos == main_complaint:
            continue
        # Check if this is timing information (starts with time words or contains time patterns)
        is_timing = False
        pos_lower = pos.lower()
        
        # Skip simple yes/no answers - they're not timing
        if pos_lower in ["yes", "no", "reported yes", "reported no", "denied"]:
            is_timing = False
        # Check for timing patterns at the beginning
        elif any(pos_lower.startswith(time_word) for time_word in ["symptoms began", "pain began", "swelling began", "dizziness began", "episode occurred"]):
            is_timing = True
        # Check for time words at the end
        elif any(pos_lower.endswith(time_word) for time_word in ["ago", "hours ago", "days ago", "today", "yesterday"]):
            is_timing = True
        # Check for specific timing patterns
        elif any(time_pattern in pos_lower for time_pattern in ["within the last", "a few days ago", "a week ago"]):
            is_timing = True
            
        if is_timing:
            timing_info.append(pos)
        else:
            other_positives.append(pos)
    
    # Debug logging after timing separation
    print(f"[Aura-LLM] 🔍 Main complaint: {main_complaint}")
    print(f"[Aura-LLM] 🔍 Priority positives for location detection: {priority_positives}")
    print(f"[Aura-LLM] 🔍 Expanded prompt: {state.get('expanded_prompt', '')}")
    print(f"[Aura-LLM] 🔍 Original complaint: {state.get('original_complaint', '')}")
    print(f"[Aura-LLM] 🔍 Timing info: {timing_info}")
    print(f"[Aura-LLM] 🔍 Other positives: {other_positives}")
    
    # Build main recap sentence
    if other_positives:
        main_sentence = f"You reported {main_complaint} with associated {pretty_join(other_positives, 'and')}"
    else:
        main_sentence = f"You reported {main_complaint}"
    
    # Add timing if present
    if timing_info:
        # Check if timing already has "starting" in it
        timing_str = pretty_join(timing_info, 'and')
        if not timing_str.lower().startswith("starting"):
            main_sentence += f" starting {timing_str}"
        else:
            main_sentence += f" {timing_str}"
    
    parts.append(main_sentence + ".")
    
    # Clean up positives to avoid redundancy with main complaint
    clean_positives = []
    for pos in positives:
        # Skip if this positive is already covered in the main complaint or other_positives
        if pos != main_complaint and pos not in other_positives:
            # Skip if it's "none", "no", "neither" - these should be in negatives
            pos_lower = pos.lower()
            if pos_lower not in ["none", "no", "neither", "reported none", "reported no", "reported neither"]:
                clean_positives.append(pos)
    
    if clean_positives:
        # Remove "reported" prefix if it's already in the text
        clean_positives_text = []
        for pos in clean_positives:
            if pos.lower().startswith("reported "):
                clean_positives_text.append(pos[9:])  # Remove "reported " prefix
            else:
                clean_positives_text.append(pos)
        parts.append("You also reported " + pretty_join(clean_positives_text, "and") + ".")
    
    # Handle denied symptoms - combine priority and regular negatives
    all_negatives = priority_negatives + negatives
    if all_negatives:
        # Clean up the negative descriptions to be more specific
        clean_negatives = []
        for neg in all_negatives:
            # Remove "denied" prefix if present and clean up the text
            neg_clean = neg.lower().replace("denied", "").replace("you", "").strip()
            if neg_clean and neg_clean not in ["none", "no", "neither"]:
                clean_negatives.append(neg_clean)
        
        if clean_negatives:
            parts.append("You denied " + pretty_join(clean_negatives, "and") + ".")
    summary = " ".join(parts).strip()
    # Cleanup: normalize spaces, collapse duplicate punctuation, fix spaces before punctuation, tidy capitalization after commas
    summary = re.sub(r"\s+", " ", summary)  # Normalize multiple spaces to single space
    summary = re.sub(r"\s+([.,;:])", r"\1", summary)
    summary = re.sub(r"([.!?]){2,}", r"\1", summary)
    summary = re.sub(r",\s+\.", ".", summary)

    # Get clinical summary from JSON if available
    clinical_summary = TRIAGE_DEFS[cond].get("clinical_summary", "")

    recap_tpl = TRIAGE_DEFS[cond].get("recap","{summary} Overall this is classified as {severity}.")
    return substitute_name(
        recap_tpl.format(summary=summary,severity=severity,clinical_summary=clinical_summary,name=state.get("user_name") or ""),
        state.get("user_name")
    )

def split_recap_into_chunks(recap_text):
    """Split recap text into smaller chunks for better TTS streaming"""
    # Split on sentence boundaries, but avoid splitting on initials/abbreviations
    # Use a more sophisticated approach to handle J.K. Rowling, Dr. Smith, etc.
    
    # First, protect initials and abbreviations by temporarily replacing them
    protected_text = recap_text
    
    # Protect initials pattern (A.B., J.K., etc.)
    initials_pattern = r'\b[A-Z]\.(?:[A-Z]\.)*'
    initials_matches = re.findall(initials_pattern, protected_text)
    for i, match in enumerate(initials_matches):
        protected_text = protected_text.replace(match, f"__INITIALS_{i}__")
    
    # Now split on sentence boundaries
    sentences = re.split(r'(?<=[.!?])\s+', protected_text.strip())
    
    # Restore the initials
    for i, match in enumerate(initials_matches):
        for j, sentence in enumerate(sentences):
            sentences[j] = sentence.replace(f"__INITIALS_{i}__", match)
    
    # Filter out empty sentences
    sentences = [s.strip() for s in sentences if s.strip()]
    
    # Group sentences together to create fewer, larger chunks (max 5 chunks)
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        # If adding this sentence would make the chunk too long, start a new chunk
        if current_chunk and len(current_chunk + " " + sentence) > 200:
            chunks.append(current_chunk.strip())
            current_chunk = sentence
        else:
            if current_chunk:
                current_chunk += " " + sentence
            else:
                current_chunk = sentence
    
    # Add the last chunk if it exists
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    # If we still have too many chunks, combine them further
    if len(chunks) > 5:
        # Combine chunks to get down to 5 or fewer
        combined_chunks = []
        chunk_size = len(chunks) // 5 + (1 if len(chunks) % 5 > 0 else 0)
        
        for i in range(0, len(chunks), chunk_size):
            group = chunks[i:i + chunk_size]
            combined_chunks.append(" ".join(group))
        
        chunks = combined_chunks
    
    return chunks

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
        return {"response": "Session reset. Start again with your symptoms."}
    
    if not prompt:
        return {"response": "Please describe your symptoms."}
    
    # Process the prompt and return a single response
    try:
        state = load_state(session_id)
        
        # Check if there's an active triage session first
        if state.get("condition"):
            # Continue existing triage - don't detect new conditions
            return process_triage_step(prompt, session_id)
        else:
            # No active triage - check for new conditions
            condition = detect_condition(prompt, session_id)
            
            if condition:
                # New triage session
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
                
                return {"response": response}
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
                return {"response": random.choice(casual_responses)}
            
    except Exception as e:
        print(f"[Aura-LLM] ❌ Error in chat-simple: {e}")
        return {"response": "I'm sorry, there was an error processing your request."}

def generate_triage_completion(state, session_id):
    """Generate triage completion recap and outcome - shared between voice and Telegram"""
    cond = state["condition"]
    sev = classify_response(cond, state["flags"])
    path = TRIAGE_DEFS[cond]
    active = state.get("active_pathway")
    outcomes = path.get("outcomes", {})
    
    if active and "pathways" in path and active in path["pathways"]:
        outcomes = path["pathways"][active].get("outcomes", outcomes)
        print(f"[Aura-LLM] 🎯 Outcome taken from pathway: {active}")
    
    # Generate recap using the same logic as voice interface
    recap = build_recap(cond, state["answers"], state["flags"], sev, session_id)
    
    # Apply NLG rewriting to recap (same as voice interface)
    from nlg import rewrite
    recap_nlg = rewrite(recap, "recap", {
        "name": state.get("user_name"),
        "condition": cond,
        "pathway": state.get("active_pathway")
    }, state.get("phrasing_history", []), 
    lambda messages, gen_kwargs: {"content": recap})  # Simple fallback
    
    # Get outcome (same as voice interface)
    raw_outcome = outcomes.get(sev, 'Follow up with a doctor.')
    outcome = substitute_name(raw_outcome, state.get("user_name"))
    
    # Apply NLG rewriting to outcome (same as voice interface)
    outcome_nlg = rewrite(outcome, "outcome", {
        "name": state.get("user_name"),
        "condition": cond,
        "pathway": state.get("active_pathway")
    }, state.get("phrasing_history", []),
    lambda messages, gen_kwargs: {"content": outcome})  # Simple fallback
    
    # Combine recap and outcome
    response = f"{recap_nlg}\n\n{outcome_nlg}"
    
    # Reset session after completion
    state = {"condition": None, "step_index": 0, "answers": [], "flags": {},
            "last_key": None, "user_name": state.get("user_name"), "active_pathway": None, 
            "entered_pathway": False, "detailed_symptoms": [], 
            "original_complaint": None, "expanded_prompt": None}
    save_state(state, session_id)
    
    return response

def process_triage_step(prompt, session_id):
    """Process a single triage step and return response"""
    state = load_state(session_id)
    
    if not state.get("condition"):
        return {"response": "Please describe your symptoms to begin triage."}
    
    # Process the answer and get the next question
    try:
        # Add answer to state (same as voice interface)
        state["answers"].append(prompt)
        
        # Update flags from the answer
        update_flags_from_answer(state["condition"], state.get("last_key"), prompt, state, session_id)
        
        # Advance to the next step
        current_step_index = state.get("step_index", 0)
        state["step_index"] = current_step_index + 1
        save_state(state, session_id)
        
        # Get the next step
        steps = get_steps(state["condition"], state)
        
        if state["step_index"] < len(steps):
            # Get the current step
            current_step = steps[state["step_index"]]
            question = current_step.get("question", "")
            
            # Apply NLG rewriting if needed
            from nlg import rewrite
            rewritten_question = rewrite(
                question,
                "question",
                {
                    "name": state.get("user_name"),
                    "condition": state["condition"],
                    "key": current_step.get("key"),
                    "allowed_answers": list(current_step.get("answers", {}).keys())
                },
                state.get("phrasing_history", []),
                lambda messages, gen_kwargs: {"content": question}  # Simple fallback
            )
            
            # Substitute name in the question
            final_question = substitute_name(rewritten_question, state.get("user_name"))
            
            return {"response": final_question}
        else:
            # Triage is complete, generate final recap
            # Use the EXACT same logic as the voice interface by calling the same function
            try:
                # Call the same completion logic as the voice interface
                response = generate_triage_completion(state, session_id)
                return {"response": response}
                
            except Exception as e:
                print(f"[Aura-LLM] ❌ Error generating recap: {e}")
                # Fallback response
                return {"response": f"Based on your symptoms, please seek medical attention. Your symptoms require evaluation by a healthcare provider."}
            
    except Exception as e:
        print(f"[Aura-LLM] ❌ Error in process_triage_step: {e}")
        return {"response": "I'm sorry, there was an error processing your response."}

# === Chat endpoint ===
@app.route("/chat", methods=["POST"])
def chat():
    data=request.get_json(); prompt=(data.get("prompt") or "").strip()
    session_id = (data.get("chat_id") or data.get("session_id") or "").strip() or None
    do_reset = bool(data.get("reset"))
    prompt_norm = normalize_text(prompt)
    RESET_KEYWORDS = {"reset","restart","new session"}
    if any(k in prompt_norm for k in RESET_KEYWORDS):
        do_reset = True
    if not prompt: return jsonify({"error":"Missing prompt"}),400
    print(f"[Aura-LLM] 💬 Session: {session_id}, Prompt: '{prompt[:50]}{'...' if len(prompt) > 50 else ''}', Reset: {do_reset}")
    state=load_state(session_id)

    if do_reset:
        user=state.get("user_name")
        state={"condition":None,"step_index":0,"answers":[],"flags":{},
               "last_key":None,"user_name":user,
               "active_pathway":None,"entered_pathway":False,
               "updated_at":None,"phrasing_history":[],"detailed_symptoms":[]}
        save_state(state, session_id)
        print(f"[Aura-LLM] 🔄 Session reset for session_id: {session_id}")
        print(f"[Aura-LLM] 🔄 Reset state: {state}")
        print(f"[Aura-LLM] 🔄 Preserved user name: {user}")
        # If the user explicitly sent a reset command, acknowledge and stop
        if prompt_norm in RESET_KEYWORDS:
            def generate_reset():
                yield f"<sentence_start>\n🔄 Session reset. Start again with your symptoms.\n<sentence_end>\n"
            return Response(stream_with_context(generate_reset()),mimetype="text/plain")

    if state.get("condition") and triage_is_stale(state):
        state.update({"condition":None,"step_index":0,"answers":[],"flags":{},
                      "last_key":None,"active_pathway":None,"entered_pathway":False})
        save_state(state, session_id)

    if state.get("condition"):
        print(f"[Aura-LLM] 🔍 Continuing triage with condition: {state.get('condition')}")
        print(f"[Aura-LLM] 🔍 State: {state}")
        cond=state["condition"]; idx=state["step_index"]
        steps=get_steps(cond,state)
        step_list=[s if isinstance(s,dict) else {"key":None,"question":str(s)} for s in steps]
        cur_step=step_list[idx-1] if idx>0 else step_list[0]
        cur_key=state.get("last_key") or cur_step.get("key"); answer=prompt

        def generate():
            nonlocal state,cond,idx,cur_step,cur_key,answer
            if cur_key and not is_valid_answer(cond,cur_key,answer,state):
                yield f"<sentence_start>\nI didn't quite catch that. {substitute_name(cur_step.get('question',''),state.get('user_name'))}\n<sentence_end>\n"; return
            state["answers"].append(answer)
            if cur_key: update_flags_from_answer(cond,cur_key,answer,state,session_id); save_state(state, session_id)

            # Ask queued inline clarify question
            if state.get("pending_clarify") and state.get("last_key") == state["pending_clarify"].get("key"):
                raw_q = state["pending_clarify"].get("question", "")
                raw_q = substitute_name(raw_q,state.get('user_name'))
                q = nlg_rewrite(raw_q, "clarify", {"name":state.get("user_name"),"condition":cond,"pathway":state.get("active_pathway"),"key":state["pending_clarify"].get("key"),"allowed_answers": list(state["pending_clarify"].get("answers",{}).keys())}, state.get("phrasing_history"), llm_chat_once)
                add_phrasing_fingerprint(state, q)
                yield f"<sentence_start>\n{q}\n<sentence_end>\n"; return

            # Enter pathway if redirected
            if state.get("active_pathway") and not state.get("entered_pathway"):
                path=TRIAGE_DEFS[cond]["pathways"][state["active_pathway"]]
                state.update({"step_index":1,"answers":[],"last_key":path["steps"][0].get("key"),"entered_pathway":True})
                save_state(state, session_id)
                print(f"[Aura-LLM] 🚪 Entering pathway: {state['active_pathway']}")
                if "intro" in path:
                    raw = substitute_name(path.get('intro',''),state.get('user_name'))
                    intro = nlg_rewrite(raw, "intro", {"name":state.get("user_name"),"condition":cond,"pathway":state.get("active_pathway")}, state.get("phrasing_history"), llm_chat_once)
                    add_phrasing_fingerprint(state, intro)
                    yield f"<sentence_start>\n{intro}\n<sentence_end>\n"
                raw_q = substitute_name(path['steps'][0]['question'],state.get('user_name'))
                q = nlg_rewrite(raw_q, "question", {"name":state.get("user_name"),"condition":cond,"pathway":state.get("active_pathway"),"key":path['steps'][0].get('key'),"allowed_answers": list(path['steps'][0].get('answers',{}).keys())}, state.get("phrasing_history"), llm_chat_once)
                add_phrasing_fingerprint(state, q)
                yield f"<sentence_start>\n{q}\n<sentence_end>\n"; return

            if idx < len(step_list):
                nxt=step_list[idx]; state.update({"step_index":idx+1,"last_key":nxt.get("key")}); save_state(state, session_id)
                print(f"[Aura-LLM] 🔍 Asking step {idx+1}/{len(step_list)}: {nxt.get('key')} - {nxt.get('question')}")
                raw_q = substitute_name(nxt.get('question',''),state.get('user_name'))
                q = nlg_rewrite(raw_q, "question", {"name":state.get("user_name"),"condition":cond,"pathway":state.get("active_pathway"),"key":nxt.get('key'),"allowed_answers": list(nxt.get('answers',{}).keys())}, state.get("phrasing_history"), llm_chat_once)
                add_phrasing_fingerprint(state, q)
                yield f"<sentence_start>\n{q}\n<sentence_end>\n"; return

            sev=classify_response(cond,state["flags"])
            path=TRIAGE_DEFS[cond]; active=state.get("active_pathway")
            outcomes=path.get("outcomes",{})
            if active and "pathways" in path and active in path["pathways"]:
                outcomes=path["pathways"][active].get("outcomes",outcomes)
                print(f"[Aura-LLM] 🎯 Outcome taken from pathway: {active}")
            recap=build_recap(cond,state["answers"],state["flags"],sev,session_id)
            recap_nlg = nlg_rewrite(recap, "recap", {"name":state.get("user_name"),"condition":cond,"pathway":state.get("active_pathway")}, state.get("phrasing_history"), llm_chat_once)
            add_phrasing_fingerprint(state, recap_nlg)
            
            # Split recap into chunks for better TTS streaming
            print(f"[Aura-LLM] 🔍 Original recap_nlg: '{recap_nlg}'")
            recap_chunks = split_recap_into_chunks(recap_nlg)
            print(f"[Aura-LLM] 🔍 Recap chunks: {recap_chunks}")
            print(f"[Aura-LLM] 🔄 Streaming recap in {len(recap_chunks)} chunks for better TTS latency")
            for chunk in recap_chunks:
                if chunk.strip():
                    yield f"<sentence_start>\n{chunk.strip()}\n<sentence_end>\n"
            
            raw_out = substitute_name(outcomes.get(sev,'Follow up with a doctor.'),state.get('user_name'))
            out_nlg = nlg_rewrite(raw_out, "outcome", {"name":state.get("user_name"),"condition":cond,"pathway":state.get("active_pathway")}, state.get("phrasing_history"), llm_chat_once)
            add_phrasing_fingerprint(state, out_nlg)
            yield f"<sentence_start>\n{out_nlg}\n<sentence_end>\n"
            user=state.get("user_name")
            state.update({"condition":None,"step_index":0,"answers":[],"flags":{},
                          "last_key":None,"user_name":user,
                          "active_pathway":None,"entered_pathway":False,
                          "detailed_symptoms":[],"original_complaint":None,"expanded_prompt":None})
            save_state(state, session_id)

        return Response(stream_with_context(generate()),mimetype="text/plain")

    # New triage or casual
    condition=detect_condition(prompt, session_id); state=load_state(session_id)
    print(f"[Aura-LLM] 🔍 Detected condition: {condition}")
    def generate():
        nonlocal condition, prompt, state
        if not condition:
            # Try RAG first for non-triage queries (only if RAG is initialized)
            used_rag = False
            final_prompt = prompt
            
            print(f"[Aura-LLM] 🔍 RAG status: AVAILABLE={RAG_AVAILABLE}, INITIALIZED={RAG_INITIALIZED}")
            print(f"[Aura-LLM] 🔍 RAG functions: get_rag_fn={get_rag_fn is not None}, search_medical_info_fn={search_medical_info_fn is not None}, smart_search_medical_info_fn={smart_search_medical_info_fn is not None}")
            if RAG_AVAILABLE and RAG_INITIALIZED and smart_search_medical_info_fn:
                print(f"[Aura-LLM] 🔍 About to call smart_search_medical_info_fn")
                try:
                    used_rag, final_prompt = smart_search_medical_info_fn(prompt, k=3)
                    if used_rag:
                        print(f"[Aura-LLM] 🔍 Using RAG for query: '{prompt[:50]}{'...' if len(prompt) > 50 else ''}'")
                    else:
                        print(f"[Aura-LLM] 💬 Using regular chat for query: '{prompt[:50]}{'...' if len(prompt) > 50 else ''}'")
                except Exception as e:
                    print(f"[Aura-LLM] ⚠️ RAG error, falling back to regular chat: {e}")
                    used_rag = False
                    final_prompt = prompt
            else:
                print(f"[Aura-LLM] 💬 Using regular chat for query: '{prompt[:50]}{'...' if len(prompt) > 50 else ''}' (RAG not available)")
            
            # Prepare system message based on whether RAG was used
            if used_rag:
                system_msg = "I am AuraVision, your personal assistant. Use the provided information to give accurate, helpful responses. Answer based on the context provided, and if the information doesn't fully answer the question, say so clearly."
            else:
                # Check for casual greetings
                casual_greetings = ["hello aura", "hi aura", "hey aura", "good morning aura", "good afternoon aura", "good evening aura"]
                if any(greeting in prompt_norm for greeting in casual_greetings):
                    system_msg = "I am AuraVision, your friendly personal assistant. Respond warmly to greetings and ask how I can help."
                else:
                    system_msg = "I am AuraVision, your friendly personal assistant."
            
            msgs = [{"role": "system", "content": system_msg}, {"role": "user", "content": final_prompt}]
            
            try:
                stream=llm.create_chat_completion(messages=msgs,stream=True)
                casual_buf=""
                for ch in stream:
                    tok=ch.get("choices",[{}])[0].get("delta",{}).get("content","")
                    if not tok: continue; 
                    casual_buf+=tok
                    if re.search(r"[.!?]['\")\]]?\s*$",casual_buf):
                        yield f"<sentence_start>\n{casual_buf.strip()}\n<sentence_end>\n"; casual_buf=""
                # Yield any remaining content
                if casual_buf.strip():
                    yield f"<sentence_start>\n{casual_buf.strip()}\n<sentence_end>\n"
            except Exception as e:
                print(f"[Aura-LLM] ❌ Error in casual mode: {e}")
                yield f"<sentence_start>\nHello! I'm AuraVision, your friendly personal assistant. How can I help you today?\n<sentence_end>\n"
            return
        steps=get_steps(condition,state)
        # Get the expanded prompt from detect_condition
        p_expanded = apply_synonym_expansion(normalize_text(prompt))
        state.update({"condition":condition,"step_index":1,"answers":[],"flags":{},
                      "last_key":steps[0].get("key"),"active_pathway":None,"entered_pathway":False,
                      "phrasing_history":state.get("phrasing_history",[]),
                      "original_complaint":prompt,
                      "expanded_prompt":p_expanded,
                      "detailed_symptoms":state.get("detailed_symptoms", [])})
        save_state(state, session_id)
        
        # Use NLG for intro and first question
        intro=substitute_name(TRIAGE_DEFS[condition].get("intro",""),state.get("user_name"))
        if intro:
            intro_nlg = nlg_rewrite(intro, "intro", {"name":state.get("user_name"),"condition":condition}, state.get("phrasing_history"), llm_chat_once)
            add_phrasing_fingerprint(state, intro_nlg)
            yield f"<sentence_start>\n{intro_nlg}\n<sentence_end>\n"
        
        raw_q = substitute_name(steps[0].get('question',''),state.get('user_name'))
        q_nlg = nlg_rewrite(raw_q, "question", {"name":state.get("user_name"),"condition":condition,"key":steps[0].get('key'),"allowed_answers": list(steps[0].get('answers',{}).keys())}, state.get("phrasing_history"), llm_chat_once)
        add_phrasing_fingerprint(state, q_nlg)
        yield f"<sentence_start>\n{q_nlg}\n<sentence_end>\n"

    return Response(stream_with_context(generate()),mimetype="text/plain")

# === RAG Endpoints ===
@app.route("/rag/search", methods=["POST"])
def rag_search():
    """Search medical information using RAG"""
    if not RAG_AVAILABLE or not RAG_INITIALIZED or not search_medical_info_fn:
        return jsonify({"error": "RAG module not available or not initialized"}), 500
    
    try:
        data = request.get_json()
        query = data.get("query", "").strip()
        k = data.get("k", 3)
        
        if not query:
            return jsonify({"error": "Query is required"}), 400
        
        # Search medical information
        augmented_prompt = search_medical_info_fn(query, k)
        
        return jsonify({
            "query": query,
            "augmented_prompt": augmented_prompt,
            "k": k
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/rag/init", methods=["POST"])
def rag_init():
    """Initialize RAG system"""
    try:
        initialize_rag_safely()
        if RAG_AVAILABLE:
            return jsonify({"status": "success", "message": "RAG initialized"})
        else:
            return jsonify({"status": "failed", "message": "RAG initialization failed"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/rag/stats", methods=["GET"])
def rag_stats():
    """Get RAG system statistics"""
    if not RAG_AVAILABLE or not RAG_INITIALIZED or not get_rag_fn:
        return jsonify({"error": "RAG module not available or not initialized"}), 500
    
    try:
        rag = get_rag_fn()
        stats = rag.get_stats()
        return jsonify(stats)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__=="__main__":
    print("[Aura-LLM] 🚑 Aura triage running with clarify routing, SOAP recap, debug logs, and casual mode")
    app.run(host="0.0.0.0",port=11434)

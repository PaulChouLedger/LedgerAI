# === container_rest.py — Aura deterministic triage
# (organ-system JSON-driven, per-step severity + fuzzy matching + synonym expansion
#  + SOAP-style clinician recap + JSON-based name placeholders with multi-word name support
#  + priority keys + outcomes inheritance + pathway selection + clarify routing + debug logging + casual mode) ===

from flask import Flask, request, jsonify, stream_with_context, Response
from llama_cpp import Llama
from dotenv import load_dotenv
import os, re, json, string
from datetime import datetime, timedelta
from glob import glob
from nlg import rewrite as nlg_rewrite

app = Flask(__name__)
load_dotenv()

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
    
    # Apply synonym expansion
    expanded_text = text
    for standard_term, variations in synonyms.items():
        for variation in variations:
            if variation.lower() in expanded_text.lower():
                expanded_text = expanded_text.replace(variation, standard_term)
                print(f"[Aura-LLM] 🔄 Expanded '{variation}' -> '{standard_term}'")
    
    return expanded_text

def substitute_name(text, user_name):
    if not text: return text
    if "{name}" in text:
        if user_name:
            return text.replace("{name}", user_name)
        return text.replace("{name}, ", "").replace("{name}", "")
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
    # Split on medical keywords or common separators, but be more careful with "and"
    raw = re.split(r"\b(having|with|experiencing|suffering|complaining|reporting)\b|,|\.", raw, 1)[0].strip()
    # Handle "and" more carefully - only split if it's followed by medical terms
    if " and " in raw:
        # Check if "and" is followed by medical terms
        and_parts = raw.split(" and ", 1)
        if len(and_parts) > 1:
            after_and = and_parts[1].lower()
            medical_after_and = any(term in after_and for term in ["pain", "ache", "headache", "chest", "abdominal", "stomach", "nausea", "dizzy", "fever", "cough"])
            if medical_after_and:
                raw = and_parts[0].strip()
    
    parts = raw.split()
    if not parts or len(parts) > 3: 
        return None
    blacklist = {"pain","cough","fever","dizziness","weakness","nausea","vomiting","abdominal","chest"}
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
    casual_greetings = ["hello aura", "hi aura", "hey aura", "good morning aura", "good afternoon aura", "good evening aura"]
    if any(greeting in p for greeting in casual_greetings):
        print(f"[Aura-LLM] 💬 Casual greeting detected: '{p}' -> no triage trigger")
        return None
    
    # Apply synonym expansion
    p_expanded = apply_synonym_expansion(p)
    print(f"[Aura-LLM] 🔄 Expanded prompt: '{p_expanded}'")
    
    for cond, data in TRIAGE_DEFS.items():
        triggers = data.get("triggers", [])
        for trig in triggers:
            trig_norm = normalize_text(trig)
            # Try exact match first
            if trig_norm in p_expanded:
                print(f"[Aura-LLM] ✅ Exact match trigger '{trig}' for '{cond}'")
                return cond
            # Try fuzzy match with 0.6 threshold
            ans_tokens = set(tokenize(p_expanded))
            trig_tokens = set(tokenize(trig_norm))
            overlap = len(ans_tokens & trig_tokens) / float(len(trig_tokens)) if trig_tokens else 0
            if overlap >= MIN_MATCH:
                print(f"[Aura-LLM] ✅ Fuzzy match trigger '{trig}' for '{cond}' (score: {overlap:.2f})")
                return cond
    return None

# === Answer matching ===
def match_answer_option(ans_norm, valid_map):
    ans_tokens = set(tokenize(ans_norm))
    best, score = None, 0.0
    for opt in valid_map:
        opt_tokens = set(tokenize(opt))
        overlap = len(ans_tokens & opt_tokens) / float(len(opt_tokens)) if opt_tokens else 0
        if overlap > score: best, score = opt, overlap
    print(f"[Aura-LLM] 🔎 Fuzzy match: ans='{ans_norm}' -> opt='{best}' score={score:.2f}")
    return best, score

def match_all_options(ans_norm, valid_map):
    ans_tokens = set(tokenize(ans_norm))
    matches = []
    for opt in valid_map:
        opt_tokens = set(tokenize(opt))
        overlap = len(ans_tokens & opt_tokens) / float(len(opt_tokens)) if opt_tokens else 0
        if overlap >= MIN_MATCH:
            matches.append(opt)
    print(f"[Aura-LLM] 🔎 Multi-match: ans='{ans_norm}' -> opts={matches}")
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
    """Non-stream single completion via llama_cpp."""
    try:
        resp = llm.create_chat_completion(messages=messages, **{k: v for k, v in gen_kwargs.items() if v is not None})
        return resp
    except Exception as e:
        return {"choices":[{"message":{"content":""}}]}

def get_steps(cond, state):
    steps = TRIAGE_DEFS[cond].get("steps", [])
    if state.get("active_pathway") and "pathways" in TRIAGE_DEFS[cond]:
        steps = TRIAGE_DEFS[cond]["pathways"][state["active_pathway"]].get("steps", steps)
    return steps

def is_valid_answer(cond, key, ans, state):
    ans_norm = normalize_text(ans)
    # Validate inline clarify answers against pending clarify map
    if key and key.startswith("clarify_") and state.get("pending_clarify") and state["pending_clarify"].get("key") == key:
        opt, score = match_answer_option(ans_norm, state["pending_clarify"].get("answers", {}))
        return opt and score >= MIN_MATCH
    steps = get_steps(cond, state)
    for s in steps:
        if isinstance(s, dict) and s.get("key") == key:
            opt, score = match_answer_option(ans_norm, s.get("answers", {}))
            return opt and score >= MIN_MATCH
    return False

def update_flags_from_answer(cond, key, ans, state):
    ans_norm = normalize_text(ans)
    # Handle inline clarify answers first
    if key and key.startswith("clarify_") and state.get("pending_clarify") and state["pending_clarify"].get("key") == key:
        opt, score = match_answer_option(ans_norm, state["pending_clarify"].get("answers", {}))
        if not opt or score < MIN_MATCH: return
        sev = state["pending_clarify"]["answers"][opt]
        if isinstance(sev, str) and sev.endswith("_pathway"):
            state["active_pathway"] = sev
            state["step_index"] = 0
            state["answers"] = []
            state["last_key"] = None
            state["entered_pathway"] = False
            print(f"[Aura-LLM] 🔀 Clarify routed → {sev}")
        state.pop("pending_clarify", None)
        return
    steps = get_steps(cond, state)
    for s in steps:
        if isinstance(s, dict) and s.get("key") == key:
            opt, score = match_answer_option(ans_norm, s.get("answers", {}))
            if not opt or score < MIN_MATCH: return
            sev = s["answers"][opt]

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

            # Normal pathway
            if isinstance(sev, str) and sev.endswith("_pathway"):
                state["active_pathway"] = sev
                state["step_index"] = 0
                state["answers"] = []
                state["last_key"] = None
                state["entered_pathway"] = False
                print(f"[Aura-LLM] 🔀 Pathway selected: {sev}")
            else:
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
    if len(parts) == 1: return parts[0]
    if len(parts) == 2: return f"{parts[0]} {conj} {parts[1]}"
    return ", ".join(parts[:-1]) + f", {conj} {parts[-1]}"

def build_recap(cond, answers, flags, severity):
    state = load_state()
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
            opt_single, _ = match_answer_option(ans_norm, valid_map)
            if opt_single: opts = [opt_single]
        # Map yes/no to reported/denied; otherwise join multiple options
        if len(opts) == 1 and opts[0] in ("yes", "no"):
            ans_out = "reported" if opts[0] == "yes" else "denied"
        else:
            ans_out = pretty_join(opts, "and") if opts else raw
        # Special handling for location steps - show actual location, not pathway names
        if key == "location":
            if ans_out.endswith("_pathway"):
                # Extract the pathway name without "_pathway" suffix for display
                display_name = ans_out.replace("_pathway", "").replace("_", " ").title()
                line = templ.format(answer=display_name).strip()
            else:
                # Use the raw answer for location (e.g., "left lower quadrant")
                line = templ.format(answer=raw).strip()
        # If template is of the form "You {answer} X" and ans_out is not reported/denied,
        # rewrite to "You reported X with ans_out" for better readability
        elif re.match(r"^\s*You\s+\{answer\}\s+", templ, flags=re.IGNORECASE) and ans_out not in ("reported", "denied"):
            tail = re.sub(r"^\s*You\s+\{answer\}\s+", "", templ, flags=re.IGNORECASE).strip()
            line = f"You reported {tail} with {ans_out}"
        else:
            line = templ.format(answer=ans_out).strip()

        if key in pk:
            if "denied" in line.lower():
                priority_negatives.append(_strip_prefix(line))
            else:
                priority_positives.append(_strip_prefix(line))
        elif "denied" in line.lower():
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

    # Priority first, clinician-style phrasing without colons
    if priority_positives:
        parts.append("You reported key symptoms including " + pretty_join(priority_positives, "and") + ".")
    # Only include denied key symptoms if some key symptoms were reported
    if priority_positives and priority_negatives:
        parts.append("You denied key symptoms of " + pretty_join(priority_negatives, "or") + ".")
    if positives: parts.append("You reported " + pretty_join(positives, "and") + ".")
    if negatives: parts.append("You denied " + pretty_join(negatives, "or") + ".")
    summary = " ".join(parts).strip()
    # Cleanup: collapse duplicate punctuation, fix spaces before punctuation, tidy capitalization after commas
    summary = re.sub(r"\s+([.,;:])", r"\1", summary)
    summary = re.sub(r"([.!?]){2,}", r"\1", summary)
    summary = re.sub(r",\s+\.", ".", summary)

    recap_tpl = TRIAGE_DEFS[cond].get("recap","{summary} Overall this is classified as {severity}.")
    return substitute_name(
        recap_tpl.format(summary=summary,severity=severity,name=state.get("user_name") or ""),
        state.get("user_name")
    )

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
               "updated_at":None,"phrasing_history":[]}
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
        cond=state["condition"]; idx=state["step_index"]
        steps=get_steps(cond,state)
        step_list=[s if isinstance(s,dict) else {"key":None,"question":str(s)} for s in steps]
        cur_step=step_list[idx-1] if idx>0 else step_list[0]
        cur_key=state.get("last_key") or cur_step.get("key"); answer=prompt

        def generate():
            nonlocal state,cond,idx,cur_step,cur_key,answer
            if cur_key and not is_valid_answer(cond,cur_key,answer,state):
                yield f"<sentence_start>\nI didn’t quite catch that. {substitute_name(cur_step.get('question',''),state.get('user_name'))}\n<sentence_end>\n"; return
            state["answers"].append(answer)
            if cur_key: update_flags_from_answer(cond,cur_key,answer,state); save_state(state, session_id)

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
            recap=build_recap(cond,state["answers"],state["flags"],sev)
            recap_nlg = nlg_rewrite(recap, "recap", {"name":state.get("user_name"),"condition":cond,"pathway":state.get("active_pathway")}, state.get("phrasing_history"), llm_chat_once)
            add_phrasing_fingerprint(state, recap_nlg)
            yield f"<sentence_start>\n{recap_nlg}\n<sentence_end>\n"
            raw_out = substitute_name(outcomes.get(sev,'Follow up with a doctor.'),state.get('user_name'))
            out_nlg = nlg_rewrite(raw_out, "outcome", {"name":state.get("user_name"),"condition":cond,"pathway":state.get("active_pathway")}, state.get("phrasing_history"), llm_chat_once)
            add_phrasing_fingerprint(state, out_nlg)
            yield f"<sentence_start>\n{out_nlg}\n<sentence_end>\n"
            user=state.get("user_name")
            state.update({"condition":None,"step_index":0,"answers":[],"flags":{},
                          "last_key":None,"user_name":user,
                          "active_pathway":None,"entered_pathway":False})
            save_state(state, session_id)

        return Response(stream_with_context(generate()),mimetype="text/plain")

    # New triage or casual
    condition=detect_condition(prompt, session_id); state=load_state(session_id)
    print(f"[Aura-LLM] 🔍 Detected condition: {condition}")
    def generate():
        nonlocal condition, prompt, state
        if not condition:
            # Check for casual greetings
            casual_greetings = ["hello aura", "hi aura", "hey aura", "good morning aura", "good afternoon aura", "good evening aura"]
            if any(greeting in prompt_norm for greeting in casual_greetings):
                msgs=[{"role":"system","content":"I am AuraVision, your friendly personal assistant. Respond warmly to greetings and ask how I can help."},
                      {"role":"user","content":prompt}]
            else:
                msgs=[{"role":"system","content":"I am AuraVision, your friendly personal assistant."},
                      {"role":"user","content":prompt}]
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
        state.update({"condition":condition,"step_index":1,"answers":[],"flags":{},
                      "last_key":steps[0].get("key"),"active_pathway":None,"entered_pathway":False,
                      "phrasing_history":state.get("phrasing_history",[])})
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

if __name__=="__main__":
    print("[Aura-LLM] 🚑 Aura triage running with clarify routing, SOAP recap, debug logs, and casual mode")
    app.run(host="0.0.0.0",port=11434)

from flask import Flask, request, jsonify, stream_with_context, Response
from llama_cpp import Llama
from dotenv import load_dotenv
import os, re, json, string
from datetime import datetime, timedelta
from collections import deque
from glob import glob

app = Flask(__name__)
load_dotenv()

# === Model Config ===
MODEL_PATH = os.getenv("MODEL_PATH", "/models/qwen2.5-1.5b-instruct-q4_0.gguf")
N_CTX = int(os.getenv("N_CTX", "1024"))
CHAT_FORMAT = os.getenv("CHAT_FORMAT", "qwen")
MODEL_NAME = os.path.basename(MODEL_PATH) if MODEL_PATH else "unknown"

# === Load the model ===
print(f"[Aura-LLM] 🧠 Loading model: {MODEL_NAME}")
llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=N_CTX,
    n_gpu_layers=32,
    n_threads=4,
    chat_format=CHAT_FORMAT,
    use_mlock=True,
    use_mmap=True,
    verbose=False,
)

# === Persistence ===
STATE_DIR = os.getenv("TRIAGE_STATE_DIR", "/app/state")
os.makedirs(STATE_DIR, exist_ok=True)
TRIAGE_STATE_PATH = os.path.join(STATE_DIR, "triage_state.json")

recent_casual_responses = deque(maxlen=5)

# === Load all organ system JSONs ===
TRIAGE_DEFS = {}
triage_dir = os.getenv("TRIAGE_DEFINITIONS_DIR", "/app/triage_defs")
if os.path.isdir(triage_dir):
    for path in glob(os.path.join(triage_dir, "*.json")):
        try:
            with open(path, "r") as f:
                data = json.load(f)
                TRIAGE_DEFS.update(data)
                print(f"[Aura-LLM] ✅ Loaded triage definitions from {os.path.basename(path)}")
        except Exception as e:
            print(f"[Aura-LLM] ⚠️ Failed to load {path}: {e}")
else:
    print(f"[Aura-LLM] ⚠️ No triage definitions directory found at {triage_dir}")

# === State helpers ===
def load_state():
    """Load triage state from persistent storage"""
    if os.path.exists(TRIAGE_STATE_PATH):
        try:
            with open(TRIAGE_STATE_PATH, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return {
        "condition": None,
        "step_index": 0,
        "answers": [],
        "flags": {},
        "last_key": None,
        "started_at": None,
        "updated_at": None,
    }

def save_state(state):
    """Save triage state to persistent storage"""
    state["updated_at"] = datetime.utcnow().isoformat()
    with open(TRIAGE_STATE_PATH, "w") as f:
        json.dump(state, f)

def triage_is_stale(state, minutes=5) -> bool:
    """Check if triage state is stale and should be reset"""
    try:
        if not state.get("updated_at"):
            return False
        last = datetime.fromisoformat(state["updated_at"])
        return datetime.utcnow() - last > timedelta(minutes=minutes)
    except Exception:
        return False

# === Helpers ===
def clean_sentence(text: str) -> str:
    """Clean sentence by removing AuraVision references and normalizing whitespace"""
    return re.sub(
        r"\s+",
        " ",
        re.sub(r"\bAuraVision\b|\bAura\b|\bLaura\b", "", text, flags=re.IGNORECASE),
    ).strip()

def normalize_text(text: str) -> str:
    """Lowercase, strip punctuation for consistent trigger/answer matching"""
    return text.lower().translate(str.maketrans("", "", string.punctuation)).strip()

def detect_condition(prompt: str):
    """Robust trigger detection using JSON definitions"""
    p = normalize_text(prompt)
    for cond, data in TRIAGE_DEFS.items():
        for trig in data.get("triggers", []):
            t = normalize_text(trig)
            if not t:
                continue
            # Always do substring match (covers multi- and single-word)
            if t in p:
                print(f"[Aura-LLM] 🔎 Matched trigger '{trig}' for condition '{cond}' in: {p}")
                return cond
    print(f"[Aura-LLM] ❌ No triage trigger matched for: {p}")
    return None

def is_valid_answer(condition, key, answer: str) -> bool:
    """Check if user answer matches expected patterns for the condition"""
    patterns = TRIAGE_DEFS[condition].get("valid_patterns", {}).get(key, [])
    ans = normalize_text(answer)
    return any(p in ans for p in patterns) or ans in {"yes", "no", "nope"}

def update_flags_from_answer(condition, key, answer, state):
    """Update condition flags based on user answer"""
    ans = normalize_text(answer)
    flags = state.get("flags") or {}
    if condition not in flags:
        flags[condition] = {}
    patterns = TRIAGE_DEFS[condition].get("valid_patterns", {}).get(key, [])
    for pat in patterns:
        if pat in ans:
            flags[condition][pat] = True
    if ans in {"no", "nope"}:
        flags[condition][key + "_negative"] = True
    state["flags"] = flags

def classify_response(condition, flags):
    """Classify response severity based on flags and rules"""
    rules = TRIAGE_DEFS[condition].get("flag_rules", {})
    cond_flags = flags.get(condition, {})

    for rule in rules.get("emergency", []):
        if "all" in rule and all(cond_flags.get(f) for f in rule["all"]):
            return "emergency"
        if "any" in rule and any(cond_flags.get(f) for f in rule["any"]):
            return "emergency"

    for rule in rules.get("urgent", []):
        if "all" in rule and all(cond_flags.get(f) for f in rule["all"]):
            return "urgent"
        if "any" in rule and any(cond_flags.get(f) for f in rule["any"]):
            return "urgent"

    for rule in rules.get("non_urgent", []):
        if "all" in rule and all(cond_flags.get(f) for f in rule["all"]):
            return "non_urgent"
        if "any" in rule and any(cond_flags.get(f) for f in rule["any"]):
            return "non_urgent"

    return "urgent"

def build_recap(condition, answers, flags, severity):
    """Build recap message with severity classification"""
    recap_tpl = TRIAGE_DEFS[condition].get("recap", "You mentioned: {summary}.")
    summary = ", ".join(answers)
    return recap_tpl.format(summary=summary) + f" Overall this is classified as {severity}."

# === Casual system prompt ===
CASUAL_SYSTEM_PROMPT = {
    "role": "system",
    "content": (
        "I am AuraVision, your friendly personal assistant ready to assist you with any task or questions you may have. "
        "When chatting casually, hold a natural, human-like conversation. Be concise but warm."
    ),
}

# === Health endpoint (preserved from original) ===
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "healthy", "service": "llm"}), 200

# === Chat endpoint ===
@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    prompt = (data.get("prompt") or "").strip()
    if not prompt:
        return jsonify({"error": "Missing prompt"}), 400

    state = load_state()
    if state.get("condition") and triage_is_stale(state):
        state = {
            "condition": None,
            "step_index": 0,
            "answers": [],
            "flags": {},
            "last_key": None,
            "started_at": None,
            "updated_at": None,
        }
        save_state(state)

    print(f"[Aura-LLM] 💬 User: {prompt}")

    # Active triage
    if state.get("condition"):
        condition = state["condition"]
        idx = state["step_index"]
        steps = TRIAGE_DEFS[condition].get("steps", [])
        keys = TRIAGE_DEFS[condition].get("keys", [])

        answer = prompt
        key = state.get("last_key")

        def generate():
            nonlocal state, condition, idx, steps, keys, answer, key
            if key and not is_valid_answer(condition, key, answer):
                retry = "I didn't quite catch that. " + steps[idx - 1]
                yield "<sentence_start>\n" + retry + "\n<sentence_end>\n"
                return

            state["answers"].append(answer)
            if key:
                update_flags_from_answer(condition, key, answer, state)
            save_state(state)

            if idx < len(steps):
                q = steps[idx]
                state["step_index"] += 1
                state["last_key"] = keys[idx]
                save_state(state)
                yield "<sentence_start>\n" + q + "\n<sentence_end>\n"
                return

            flags = state["flags"]
            severity = classify_response(condition, flags)
            recap = build_recap(condition, state["answers"], flags, severity)
            outcome = TRIAGE_DEFS[condition]["outcomes"][severity]
            yield "<sentence_start>\n" + recap + "\n<sentence_end>\n"
            yield "<sentence_start>\n" + outcome + "\n<sentence_end>\n"

            state = {
                "condition": None,
                "step_index": 0,
                "answers": [],
                "flags": {},
                "last_key": None,
                "started_at": None,
                "updated_at": None,
            }
            save_state(state)

        return Response(stream_with_context(generate()), mimetype="text/plain")

    # Otherwise detect new complaint
    condition = detect_condition(prompt)

    def generate():
        nonlocal state, condition
        if not condition:
            print("[Aura-LLM] 💬 Casual conversation mode")
            msgs = [CASUAL_SYSTEM_PROMPT] + [{"role": "user", "content": prompt}]
            stream = llm.create_chat_completion(messages=msgs, stream=True)
            buf = ""
            for ch in stream:
                tok = ch.get("choices", [{}])[0].get("delta", {}).get("content", "")
                if not tok:
                    continue
                buf += tok
                if re.search(r"[.!?]['\")\]]?\s*$", buf) and len(buf.strip()) > 5:
                    sent = clean_sentence(buf)
                    if sent in recent_casual_responses:
                        continue
                    recent_casual_responses.append(sent)
                    yield "<sentence_start>\n" + sent + "\n<sentence_end>\n"
                    buf = ""
            return

        print(f"[Aura-LLM] 🩺 Starting triage for {condition}")
        steps = TRIAGE_DEFS[condition].get("steps", [])
        keys = TRIAGE_DEFS[condition].get("keys", [])
        state = {
            "condition": condition,
            "step_index": 1,
            "answers": [],
            "flags": {},
            "last_key": keys[0] if keys else None,
            "started_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat(),
        }
        save_state(state)
        intro = TRIAGE_DEFS[condition].get(
            "intro",
            f"I'm sorry you're experiencing {condition.replace('_',' ')}. Let me ask you a few questions.",
        )
        if steps:
            yield "<sentence_start>\n" + intro + "\n<sentence_end>\n"
            yield "<sentence_start>\n" + steps[0] + "\n<sentence_end>\n"

    return Response(stream_with_context(generate()), mimetype="text/plain")


if __name__ == "__main__":
    print("[Aura-LLM] 🚑 Aura triage (Organ-system JSON-driven, robust triggers)")
    app.run(host="0.0.0.0", port=11434)
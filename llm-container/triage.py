#!/usr/bin/env python3
"""
Aura TRIAGE Mode - Complete Working Version (Restored from 48de2ca)

All triage functions restored to working state with proper:
- SOAP-style recap generation
- Detailed symptoms tracking
- Priority key handling
- Pathway support
"""

import os
import json
import re
import string
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from glob import glob

# Global triage definitions dictionary
TRIAGE_DEFS = {}
MIN_MATCH = 0.6

def load_triage_definitions(triage_dir="/app/triage_defs"):
    """Load all triage definitions from JSON files"""
    global TRIAGE_DEFS
    
    if not os.path.isdir(triage_dir):
        print(f"[Triage] ❌ Triage definitions directory not found: {triage_dir}")
        return
    
    print(f"[Triage] 🔍 Loading triage definitions from: {triage_dir}")
    
    for path in glob(os.path.join(triage_dir, "*.json")):
        try:
            with open(path, "r") as f:
                data = json.load(f)
                TRIAGE_DEFS.update(data)
                print(f"[Triage] ✅ Loaded triage defs: {os.path.basename(path)}")
                print(f"[Triage] 🔍 Loaded conditions: {list(data.keys())}")
        except Exception as e:
            print(f"[Triage] ⚠️ Failed to load triage defs {path}: {e}")
    
    print(f"[Triage] 🔍 Total loaded conditions: {len(TRIAGE_DEFS)}")


# Load triage definitions on module import
load_triage_definitions(os.getenv("TRIAGE_DEFINITIONS_DIR", "/app/triage_defs"))


# === State Management ===

def get_state_path(session_id: str) -> str:
    """Get file path for session state"""
    STATE_DIR = os.getenv("TRIAGE_STATE_DIR", "/app/state")
    os.makedirs(STATE_DIR, exist_ok=True)
    if session_id:
        safe = re.sub(r"[^A-Za-z0-9_.-]", "_", session_id)
        return os.path.join(STATE_DIR, f"triage_state_{safe}.json")
    return os.path.join(STATE_DIR, "triage_state.json")


def load_state(session_id: str) -> Dict[str, Any]:
    """Load triage state from disk"""
    path = get_state_path(session_id)
    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                return json.load(f)
        except Exception:
            pass
    
    return {
        "condition": None,
        "step_index": 0,
        "answers": [],
        "flags": {},
        "last_key": None,
        "user_name": None,
        "active_pathway": None,
        "entered_pathway": False,
        "updated_at": None,
        "phrasing_history": [],
        "detailed_symptoms": [],
        "original_complaint": None,
        "expanded_prompt": None
    }


def save_state(state: Dict[str, Any], session_id: str):
    """Save triage state to disk"""
    state["updated_at"] = datetime.utcnow().isoformat()
    path = get_state_path(session_id)
    with open(path, "w") as f:
        json.dump(state, f)


def triage_is_stale(state: Dict[str, Any], minutes: int = 5) -> bool:
    """Check if triage session is stale"""
    try:
        return datetime.utcnow() - datetime.fromisoformat(state.get("updated_at") or "") > timedelta(minutes=minutes)
    except Exception:
        return False


# === Text Processing ===

def normalize_text(text: str) -> str:
    """Normalize text for comparison"""
    if not text:
        return ""
    # Remove emojis and non-ASCII
    text = re.sub(r'[^\x00-\x7F]+', '', text)
    # Remove punctuation and lowercase
    return text.lower().translate(str.maketrans('', '', string.punctuation)).strip()


def tokenize(text: str) -> List[str]:
    """Tokenize normalized text"""
    return normalize_text(text).split()


def substitute_name(text: str, user_name: str) -> str:
    """Substitute {name} placeholder with user's name"""
    if not text:
        return text
    if "{name}" in text:
        if user_name:
            return text.replace("{name}", user_name)
        else:
            text = text.replace("{name}, ", "").replace("{name}", "")
            cleaned = re.sub(r"^[,;:\-]\s*", "", text)
            cleaned = re.sub(r"\s{2,}", " ", cleaned).strip()
            return cleaned
    cleaned = re.sub(r"^[,;:\-]\s*", "", text)
    cleaned = re.sub(r"\s{2,}", " ", cleaned).strip()
    return cleaned


def pretty_join(parts: List[str], conj: str = "and") -> str:
    """Join list with proper grammar"""
    if not parts:
        return ""
    parts = [re.sub(r'\s+', ' ', part.strip()) for part in parts]
    if len(parts) == 1:
        return parts[0]
    if len(parts) == 2:
        return f"{parts[0]} {conj} {parts[1]}"
    return ", ".join(parts[:-1]) + f", {conj} {parts[-1]}"


# === Synonym Expansion ===

def apply_synonym_expansion(text: str) -> str:
    """Apply synonym expansion to normalize medical terms"""
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
                print(f"[Triage] ⚠️ Failed to load synonyms from {file_path}: {e}")
    
    expanded_text = text
    all_variations = []
    for standard_term, variations in synonyms.items():
        for variation in variations:
            all_variations.append((len(variation), variation, standard_term))
    
    all_variations.sort(key=lambda x: x[0], reverse=True)
    
    for length, variation, standard_term in all_variations:
        pattern = r'\b' + re.escape(variation) + r'\b'
        if re.search(pattern, expanded_text, re.IGNORECASE):
            expanded_text = re.sub(pattern, standard_term, expanded_text, flags=re.IGNORECASE)
            break
    
    return expanded_text


# === Answer Matching ===

def normalize_yes_no_response(text: str) -> str:
    """Normalize natural yes/no responses"""
    text_lower = text.lower().strip()
    
    # Negative responses first
    if any(phrase in text_lower for phrase in [
        "no", "nope", "nah", "not", "don't", "do not", "haven't", "have not",
        "i don't", "i do not", "i haven't", "i have not",
        "i don't have", "i do not have", "i don't feel", "i do not feel",
        "i don't experience", "i do not experience", "i am not", "i'm not"
    ]):
        return "no"
    
    if text_lower in ["i dont", "i don't", "i do not", "i havent", "i haven't", "i have not"]:
        return "no"
    
    # Positive responses
    if any(phrase in text_lower for phrase in [
        "yes", "yea", "yeah", "yep", "yup", "sure", "ok", "okay",
        "i do", "i have", "i am", "i feel", "i experience",
        "i do have", "i do feel", "i do experience",
        "i have been", "i am having", "i am experiencing"
    ]):
        return "yes"
    
    return text


def get_generic_onset_answers() -> Dict[str, str]:
    """Get standard onset answers"""
    return {
        "within the last hour": "emergency",
        "within the last few hours": "emergency",
        "today": "urgent",
        "yesterday": "urgent",
        "a few days ago": "urgent",
        "a week ago": "non_urgent",
        "unknown": "urgent"
    }


def match_flexible_time(ans_expanded: str, valid_map: Dict[str, str]) -> Optional[Tuple[str, float]]:
    """Match flexible time patterns like '3 hours ago'"""
    time_pattern = r'(\d+)\s*(minute|hour|day|week|month)s?\s*ago'
    match = re.search(time_pattern, ans_expanded, re.IGNORECASE)
    
    if not match:
        return None
    
    number = int(match.group(1))
    unit = match.group(2).lower()
    
    if unit in ['minute', 'hour']:
        if unit == 'minute' or (unit == 'hour' and number <= 6):
            return "within the last hour", 1.0
        elif unit == 'hour' and number <= 12:
            return "within the last few hours", 1.0
        else:
            return "today", 1.0
    elif unit == 'day':
        if number == 1:
            return "yesterday", 1.0
        elif number <= 7:
            return "a few days ago", 1.0
        else:
            return "a week ago", 1.0
    elif unit == 'week':
        if number == 1:
            return "a week ago", 1.0
        else:
            return "last week", 1.0
    elif unit == 'month':
        return "last week", 1.0
    
    return None


def match_answer_option(ans_norm: str, valid_map: Dict[str, str], use_synonyms: bool = True, key: str = None) -> Tuple[Optional[str], float]:
    """Match answer to options with fuzzy matching"""
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
    
    ans_tokens = set(tokenize(ans_expanded))
    best, score = None, 0.0
    
    for opt in valid_map:
        opt_tokens = set(tokenize(opt))
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
            final_score = 0
            
        if final_score > score:
            best, score = opt, final_score
            
    return best, score


def match_all_options(ans_norm: str, valid_map: Dict[str, str]) -> List[str]:
    """Match all applicable options"""
    ans_expanded = apply_synonym_expansion(ans_norm)
    ans_tokens = set(tokenize(ans_expanded))
    matches = []
    
    for opt in valid_map:
        opt_tokens = set(tokenize(opt))
        overlap = len(ans_tokens & opt_tokens) / float(len(opt_tokens)) if opt_tokens else 0
        if overlap >= MIN_MATCH:
            matches.append(opt)
    
    # Handle compound answers
    if not matches and "and" in ans_expanded:
        components = [comp.strip() for comp in ans_expanded.split("and")]
        for comp in components:
            comp_tokens = set(tokenize(comp))
            for opt in valid_map:
                opt_tokens = set(tokenize(opt))
                overlap = len(comp_tokens & opt_tokens) / float(len(opt_tokens)) if opt_tokens else 0
                if overlap >= MIN_MATCH and opt not in matches:
                    matches.append(opt)
    
    return matches


# === Pathway and Flag Management ===

def get_intro(cond: str, state: Dict[str, Any]) -> str:
    """Get intro message for condition"""
    intro = TRIAGE_DEFS[cond].get("intro", "Let me ask you a few questions.")
    # Substitute {name} placeholder if present
    user_name = state.get("user_name")
    if user_name and "{name}" in intro:
        intro = intro.replace("{name}", user_name)
    elif "{name}" in intro:
        intro = intro.replace("{name}, ", "")  # Remove name placeholder if no name
    return intro


def get_steps(cond: str, state: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Get steps for condition (considers active pathway)"""
    steps = TRIAGE_DEFS[cond].get("steps", [])
    if state.get("active_pathway") and "pathways" in TRIAGE_DEFS[cond]:
        steps = TRIAGE_DEFS[cond]["pathways"][state["active_pathway"]].get("steps", steps)
    return steps


def get_pathway_detailed_symptom(cond: str, pathway: str, state: Dict[str, Any]) -> str:
    """Get detailed symptom from pathway"""
    if "pathways" in TRIAGE_DEFS[cond] and pathway in TRIAGE_DEFS[cond]["pathways"]:
        pathway_def = TRIAGE_DEFS[cond]["pathways"][pathway]
        if "detailed_symptom" in pathway_def:
            return pathway_def["detailed_symptom"]
    
    cond_name = cond.replace("_", " ")
    pathway_name = pathway.replace("_pathway", "").replace("_", " ")
    return f"{pathway_name} {cond_name}"


def is_valid_answer(cond: str, key: str, ans: str, state: Dict[str, Any]) -> bool:
    """Check if answer is valid for question"""
    ans_norm = normalize_text(ans)
    
    if key and key.startswith("clarify_") and state.get("pending_clarify") and state["pending_clarify"].get("key") == key:
        opt, score = match_answer_option(ans_norm, state["pending_clarify"].get("answers", {}), use_synonyms=False, key=key)
        return opt and score >= MIN_MATCH
    
    steps = get_steps(cond, state)
    for s in steps:
        if isinstance(s, dict) and s.get("key") == key:
            opt, score = match_answer_option(ans_norm, s.get("answers", {}), key=key)
            return opt and score >= MIN_MATCH
    
    return False


def update_flags_from_answer(cond: str, key: str, ans: str, state: Dict[str, Any], session_id: str):
    """Update flags based on answer"""
    ans_norm = normalize_text(ans)
    
    # Handle inline clarify answers
    if key and key.startswith("clarify_") and state.get("pending_clarify") and state["pending_clarify"].get("key") == key:
        opt, score = match_answer_option(ans_norm, state["pending_clarify"].get("answers", {}), use_synonyms=False, key=key)
        if not opt or score < MIN_MATCH:
            return
        
        sev = state["pending_clarify"]["answers"][opt]
        if isinstance(sev, str) and sev.endswith("_pathway"):
            state["active_pathway"] = sev
            state["step_index"] = 0
            state["answers"] = []
            state["last_key"] = None
            state["entered_pathway"] = False
            print(f"[Triage] 🔀 Clarify routed → {sev}")
            
            detailed_symptom = get_pathway_detailed_symptom(cond, sev, state)
            if "detailed_symptoms" not in state:
                state["detailed_symptoms"] = []
            if detailed_symptom not in state["detailed_symptoms"]:
                state["detailed_symptoms"].append(detailed_symptom)
                print(f"[Triage] 📝 Detailed symptoms array: {state['detailed_symptoms']}")
                save_state(state, session_id)
                
        state.pop("pending_clarify", None)
        return
    
    steps = get_steps(cond, state)
    
    for s in steps:
        if isinstance(s, dict) and s.get("key") == key:
            opt, score = match_answer_option(ans_norm, s.get("answers", {}), key=key)
            
            if not opt or score < MIN_MATCH:
                return
            
            # Handle empty answers (generic onset)
            if not s["answers"] or opt not in s["answers"]:
                if key == "onset":
                    generic_answers = get_generic_onset_answers()
                    sev = generic_answers.get(opt, "urgent")
                else:
                    return
            else:
                sev = s["answers"][opt]

            # Inline clarify with followup_question
            if isinstance(sev, dict) and sev.get("followup_question"):
                clarify_key = f"clarify_{key}"
                state["pending_clarify"] = {
                    "key": clarify_key,
                    "question": sev.get("followup_question", ""),
                    "answers": sev.get("answers", {})
                }
                state["last_key"] = clarify_key
                return

            # Clarify routing
            if key.startswith("clarify_"):
                if isinstance(sev, str) and sev.endswith("_pathway"):
                    state["active_pathway"] = sev
                    state["step_index"] = 0
                    state["answers"] = []
                    state["last_key"] = None
                    state["entered_pathway"] = False
                return

            # Normal pathway
            is_pathway = isinstance(sev, str) and "pathways" in TRIAGE_DEFS[cond] and sev in TRIAGE_DEFS[cond]["pathways"]
            
            if is_pathway:
                state["active_pathway"] = sev
                state["step_index"] = 0
                state["answers"] = []
                state["last_key"] = None
                state["entered_pathway"] = False
                
                if "detailed_symptoms" not in state:
                    state["detailed_symptoms"] = []
                
                pathway_name = sev.replace("_", " ")
                detailed_symptom = pathway_name
                
                if detailed_symptom not in state["detailed_symptoms"]:
                    state["detailed_symptoms"].append(detailed_symptom)
                    save_state(state, session_id)
            else:
                state["flags"].setdefault(cond, {})[key] = sev


# === Severity Classification ===

def classify_response(cond: str, flags: Dict[str, Any]) -> str:
    """Classify severity based on flags"""
    vals = list(flags.get(cond, {}).values())
    if "emergency" in vals:
        return "emergency"
    if "urgent" in vals:
        return "urgent"
    if vals and all(v == "non_urgent" for v in vals):
        return "non_urgent"
    return "urgent"


# === Recap Generation ===

def build_recap(cond: str, answers: List[str], flags: Dict[str, Any], severity: str, session_id: str = None) -> str:
    """Build comprehensive SOAP-style recap"""
    state = load_state(session_id)
    steps = get_steps(cond, state)
    pk = TRIAGE_DEFS[cond].get("priority_keys", [])

    if state.get("active_pathway") and "pathways" in TRIAGE_DEFS[cond]:
        path = TRIAGE_DEFS[cond]["pathways"][state["active_pathway"]]
        steps = path.get("steps", steps)
        pk = path.get("priority_keys", pk)
        print(f"[Triage] 📝 Recap built from pathway: {state['active_pathway']}")

    positives, negatives, priority_positives, priority_negatives = [], [], [], []
    
    def _strip_prefix(text: str) -> str:
        t = re.sub(r"(?i)^\s*you\s+(reported|denied)\s+", "", text or "").strip()
        t = t.rstrip(". ")
        return t
    
    for s, raw in zip(steps, answers):
        if not isinstance(s, dict):
            continue
        key, templ = s.get("key"), s.get("recap_template", "{answer}")
        valid_map = s.get("answers", {})
        ans_norm = normalize_text(raw)
        opts = match_all_options(ans_norm, valid_map) or []
        
        if not opts:
            opt_single, _ = match_answer_option(ans_norm, valid_map, key=key)
            if opt_single:
                opts = [opt_single]
        
        # Map yes/no to reported/denied
        if len(opts) == 1 and opts[0] in ("yes", "no"):
            ans_out = "reported" if opts[0] == "yes" else "denied"
        elif opts:
            clean_opts = []
            for opt in opts:
                is_redundant = False
                for other_opt in opts:
                    if opt != other_opt and opt in other_opt:
                        is_redundant = True
                        break
                if not is_redundant:
                    clean_opts.append(opt)
            
            if len(clean_opts) > 1:
                clean_opts = [max(clean_opts, key=len)]
            
            # Timing questions
            if key in ["onset", "when", "timing", "duration"]:
                ans_out = pretty_join(clean_opts or opts, 'and')
            else:
                ans_out = f"reported {pretty_join(clean_opts or opts, 'and')}"
        else:
            normalized = normalize_yes_no_response(raw)
            if normalized == "yes":
                ans_out = f"reported {raw}"
            elif normalized == "no":
                ans_out = "denied"
            else:
                ans_out = raw
        
               # Handle pathway routing
        if ans_out.endswith("_pathway"):
            display_name = ans_out.replace("_pathway", "").replace("_", " ").title()
            line = templ.format(answer=display_name).strip()
        elif re.match(r"^\s*You\s+\{answer\}\s+", templ, flags=re.IGNORECASE) and ans_out not in ("reported", "denied"):
            tail = re.sub(r"^\s*You\s+\{answer\}\s+", "", templ, flags=re.IGNORECASE).strip()
            line = f"You reported {tail} with {ans_out}"
        else:
            # Use the template as-is (JSON files should handle proper formatting)
            line = templ.format(answer=ans_out).strip()
        
        # Categorize
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
    
    # Deduplicate
    def _dedup(seq):
        seen = set()
        out = []
        for item in seq:
            if item and item not in seen:
                seen.add(item)
                out.append(item)
        return out
    
    positives = _dedup(positives)
    negatives = _dedup(negatives)
    priority_positives = _dedup(priority_positives)
    priority_negatives = _dedup(priority_negatives)
    
    # Extract main complaint from detailed symptoms
    original_complaint = (state.get("original_complaint") or "").lower()
    detailed_symptoms = state.get("detailed_symptoms", []) or []
    
    main_complaint = None
    best_score = 0
    
    if detailed_symptoms and original_complaint:
        for symptom in detailed_symptoms:
            symptom_lower = symptom.lower()
            original_words = set(original_complaint.split())
            symptom_words = set(symptom_lower.split())
            
            overlap = len(original_words.intersection(symptom_words))
            length_bonus = len(symptom_words) * 0.3
            anatomical_bonus = 0
            if any(word in symptom_lower for word in ['arm', 'leg', 'side', 'right', 'left', 'upper', 'lower']):
                anatomical_bonus = 2.0
            score = overlap + length_bonus + anatomical_bonus
            
            if score > best_score:
                best_score = score
                main_complaint = symptom
    
    if not main_complaint:
        main_complaint = cond.replace("_", " ").replace("suspected", "").strip()
    
    # Separate timing
    timing_info = []
    other_positives = []
    for pos in priority_positives:
        if pos == main_complaint:
            continue
        pos_lower = pos.lower()
        is_timing = (
            any(pos_lower.startswith(tw) for tw in ["symptoms began", "pain began", "swelling began"]) or
            any(pos_lower.endswith(tw) for tw in ["ago", "hours ago", "days ago", "today", "yesterday"]) or
            any(tp in pos_lower for tp in ["within the last", "a few days ago", "a week ago"])
        )
        if is_timing:
            timing_info.append(pos)
        else:
            other_positives.append(pos)
    
    # Build main sentence
    parts = []
    if other_positives:
        main_sentence = f"You reported {main_complaint} with associated {pretty_join(other_positives, 'and')}"
    else:
        main_sentence = f"You reported {main_complaint}"
    
    if timing_info:
        timing_str = pretty_join(timing_info, 'and')
        if not timing_str.lower().startswith("starting"):
            main_sentence += f" starting {timing_str}"
        else:
            main_sentence += f" {timing_str}"
    
    parts.append(main_sentence + ".")
    
    # Additional positives
    clean_positives = [p for p in positives if p != main_complaint and p not in other_positives and p.lower() not in ["none", "no", "neither"]]
    if clean_positives:
        clean_positives_text = [p[9:] if p.lower().startswith("reported ") else p for p in clean_positives]
        parts.append("You also reported " + pretty_join(clean_positives_text, "and") + ".")
    
    # Negatives
    all_negatives = priority_negatives + negatives
    if all_negatives:
        clean_negatives = [n.lower().replace("denied", "").replace("you", "").strip() for n in all_negatives]
        clean_negatives = [n for n in clean_negatives if n and n not in ["none", "no", "neither"]]
        if clean_negatives:
            parts.append("You denied " + pretty_join(clean_negatives, "and") + ".")
    
    summary = " ".join(parts).strip()
    summary = re.sub(r"\s+", " ", summary)
    summary = re.sub(r"\s+([.,;:])", r"\1", summary)
    summary = re.sub(r"([.!?]){2,}", r"\1", summary)
    summary = re.sub(r",\s+\.", ".", summary)
    
    clinical_summary = TRIAGE_DEFS[cond].get("clinical_summary", "")
    recap_tpl = TRIAGE_DEFS[cond].get("recap", "{summary} Overall this is classified as {severity}.")
    
    return substitute_name(
        recap_tpl.format(summary=summary, severity=severity, clinical_summary=clinical_summary, name=state.get("user_name") or ""),
        state.get("user_name")
    )


# === Condition Detection ===

def detect_condition(prompt: str, session_id: str = None) -> Optional[str]:
    """Detect medical condition from prompt"""
    p = normalize_text(prompt)
    
    # Check for casual greetings
    casual_greeting_patterns = [
        r'\bhello\b', r'\bhi\b', r'\bhey\b', r'\bhowdy\b',
        r'\bgood morning\b', r'\bgood afternoon\b', r'\bgood evening\b'
    ]
    
    knowledge_indicators = ["tell me", "what is", "who is", "explain", "describe", "information about"]
    is_knowledge_query = any(indicator in p for indicator in knowledge_indicators)
    
    if not is_knowledge_query:
        is_casual_greeting = any(re.search(pattern, p) for pattern in casual_greeting_patterns)
        if is_casual_greeting:
            medical_keywords = ["pain", "hurt", "ache", "symptom", "problem", "issue"]
            has_medical_content = any(keyword in p for keyword in medical_keywords)
            if not has_medical_content:
                return None
    
    # Apply synonym expansion
    p_expanded = apply_synonym_expansion(p)
    
    # Initialize detailed symptoms
    state = load_state(session_id)
    if "detailed_symptoms" not in state:
        state["detailed_symptoms"] = []
    
    if p_expanded and p_expanded not in state["detailed_symptoms"]:
        state["detailed_symptoms"].append(p_expanded)
    
    save_state(state, session_id)
    
    for cond, data in TRIAGE_DEFS.items():
        triggers = data.get("triggers", [])
        for trig in triggers:
            trig_norm = normalize_text(trig)
            
            if trig_norm in p_expanded:
                if "detailed_symptom" in TRIAGE_DEFS[cond]:
                    detailed_symptom = TRIAGE_DEFS[cond]["detailed_symptom"]
                    if detailed_symptom not in state["detailed_symptoms"]:
                        state["detailed_symptoms"].append(detailed_symptom)
                        save_state(state, session_id)
                return cond
            
            # Fuzzy match
            ans_tokens = set(tokenize(p_expanded))
            trig_tokens = set(tokenize(trig_norm))
            overlap = len(ans_tokens & trig_tokens) / float(len(trig_tokens)) if trig_tokens else 0
            if overlap >= MIN_MATCH:
                if "detailed_symptom" in TRIAGE_DEFS[cond]:
                    detailed_symptom = TRIAGE_DEFS[cond]["detailed_symptom"]
                    if detailed_symptom not in state["detailed_symptoms"]:
                        state["detailed_symptoms"].append(detailed_symptom)
                        save_state(state, session_id)
                return cond
    
    return None


# === Triage Step Processing ===

def process_triage_step(prompt: str, state: Dict[str, Any], session_id: str) -> Tuple[str, Dict[str, Any]]:
    """Process triage step and return next question"""
    condition = state.get("condition")
    if not condition:
        return "Please describe your symptoms to begin triage.", state

    # Get current step info
    current_step_index = state.get("step_index", 0)
    steps = get_steps(condition, state)
    step_list = [s if isinstance(s, dict) else {"key": None, "question": str(s)} for s in steps]
    current_step = step_list[current_step_index] if current_step_index < len(step_list) else None

    if current_step:
        current_key = state.get("last_key") or current_step.get("key")

        # Validate answer before accepting (OLD LOGIC)
        if current_key and not is_valid_answer(condition, current_key, prompt, state):
            print(f"[Triage] ❌ Invalid answer '{prompt}' for question '{current_key}'")
            print(f"[Triage] 🔄 Re-asking question (expected one of: {list(current_step.get('answers', {}).keys())})")
            return f"I didn't quite catch that. {substitute_name(current_step.get('question', ''), state.get('user_name'))}", state

        # Answer is valid - add it and update flags
        state["answers"].append(prompt)

        if current_key:
            update_flags_from_answer(condition, current_key, prompt, state, session_id)

        # Advance to next step
        state["step_index"] = current_step_index + 1
    
    # Get next step
    if state["step_index"] < len(steps):
        next_step = steps[state["step_index"]]
        question = next_step.get("question", "")

        # Apply NLG rewriting (using simple fallback like old version)
        from nlg import rewrite
        def llm_chat_once_fallback(messages, **kwargs):
            """Simple fallback for NLG rewriting - just return the question"""
            return {"content": question}

        rewritten_question = rewrite(
            question,
            "question",
            {
                "name": state.get("user_name"),
                "condition": state["condition"],
                "key": next_step.get("key"),
                "allowed_answers": list(next_step.get("answers", {}).keys())
            },
            state.get("phrasing_history", []),
            llm_chat_once_fallback
        )

        final_question = substitute_name(rewritten_question, state.get("user_name"))
        return final_question, state

    else:
        # Triage complete
        recap_response = generate_triage_completion(state, session_id)
        return recap_response, state


def generate_triage_completion(state: Dict[str, Any], session_id: str) -> str:
    """Generate final triage completion"""
    condition = state.get("condition")
    answers = state.get("answers", [])
    flags = state.get("flags", {})
    
    if not condition:
        return "I'm sorry, there was an error processing your triage."
    
    # Classify severity
    severity = classify_response(condition, flags)
    
    # Build recap
    recap = build_recap(condition, answers, flags, severity, session_id)
    
    # Get outcome
    active_pathway = state.get("active_pathway")
    outcomes = TRIAGE_DEFS[condition].get("outcomes", {})
    
    if active_pathway and "pathways" in TRIAGE_DEFS[condition]:
        pathway_outcomes = TRIAGE_DEFS[condition]["pathways"][active_pathway].get("outcomes", {})
        if pathway_outcomes:
            outcomes = pathway_outcomes
    
    if severity in outcomes:
        outcome = outcomes[severity]
    else:
        if severity == "emergency":
            outcome = "Seek emergency medical care immediately (call 911 or go to nearest ER)."
        elif severity == "urgent":
            outcome = "Seek medical care within 2-4 hours (urgent care or ER if symptoms worsen)."
        else:
            outcome = "Schedule appointment with primary care physician within 24-48 hours."
    
    outcome = substitute_name(outcome, state.get("user_name"))
    
    return f"{recap} {outcome}"


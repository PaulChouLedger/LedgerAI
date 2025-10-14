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


# tokenize moved to validation.py


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

# Import centralized validation from validation module
from validation import match_answer_option, check_typo_similarity, MIN_MATCH, get_generic_onset_answers, match_flexible_time, tokenize, normalize_yes_no_response


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
                opt, score = match_answer_option(ans_norm, generic_onset_answers, key=key, 
                                                synonym_expansion_fn=apply_synonym_expansion)
                print(f"[Triage] 🔍 Time matching: opt='{opt}', score={score}, threshold={MIN_MATCH}")
                return opt and score >= MIN_MATCH

            # For other empty answers, accept any answer (rare case)
            if not answers:
                print(f"[Triage] ⚠️ Empty answers for '{key}' - accepting any answer")
                return True

            opt, score = match_answer_option(ans_norm, answers, key=key, 
                                            synonym_expansion_fn=apply_synonym_expansion)
            print(f"[Triage] 🔍 Answer matching: opt='{opt}', score={score}, threshold={MIN_MATCH}")
            return opt and score >= MIN_MATCH

    print(f"[Triage] ❌ No step found with key '{key}'")
    return False


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
    print(f"[Triage] 🔍 Validating answer: key='{key}', ans='{ans}' (norm='{ans_norm}')")

    if key and key.startswith("clarify_") and state.get("pending_clarify") and state["pending_clarify"].get("key") == key:
        opt, score = match_answer_option(ans_norm, state["pending_clarify"].get("answers", {}), use_synonyms=False, key=key)
        print(f"[Triage] 🔍 Clarify validation: opt='{opt}', score={score}, threshold={MIN_MATCH}")
        return opt and score >= MIN_MATCH

    steps = get_steps(cond, state)
    print(f"[Triage] 🔍 Found {len(steps)} steps for condition '{cond}'")

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
        
        # Determine answer format based on template pattern
        # Templates like "You {answer} X" expect "reported"/"denied"
        # Templates like "X {answer}" expect the actual answer text
        
        template_expects_reported_denied = re.match(r"^\s*You\s+\{answer\}\s+", templ, flags=re.IGNORECASE)
        
        if len(opts) == 1 and opts[0] in ("yes", "no"):
            # Yes/no questions
            if template_expects_reported_denied:
                ans_out = "reported" if opts[0] == "yes" else "denied"
            else:
                # For templates like "X {answer}", don't use reported/denied
                ans_out = opts[0]
        elif opts:
            # Multiple options or specific answers
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
            
            # For templates expecting reported/denied, add prefix
            if template_expects_reported_denied:
                # Check if this is a positive or negative response
                normalized = normalize_yes_no_response(raw)
                if normalized == "no":
                    ans_out = "denied"
                else:
                    ans_out = f"reported {pretty_join(clean_opts or opts, 'and')}"
            else:
                # Template expects actual answer, no prefix
                ans_out = pretty_join(clean_opts or opts, 'and')
        else:
            # No match found - fallback
            normalized = normalize_yes_no_response(raw)
            if normalized == "yes":
                ans_out = "reported" if template_expects_reported_denied else "yes"
            elif normalized == "no":
                ans_out = "denied" if template_expects_reported_denied else "no"
            else:
                ans_out = raw
        
        # Handle pathway routing
        if ans_out.endswith("_pathway"):
            display_name = ans_out.replace("_pathway", "").replace("_", " ").title()
            line = templ.format(answer=display_name).strip()
        else:
            # Use the template as-is
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
        # Clean other_positives to remove sentence prefixes and restructure
        cleaned_other_positives = []
        for pos in other_positives:
            pos_clean = pos
            # Remove common sentence starters
            prefixes_to_remove = [
                "You described the pain as ",
                "You described ",
                "You reported that ",
                "You reported ",
                "You mentioned that ",
                "You mentioned ",
                "You said that ",
                "You said ",
                "that the pain ",
                "that pain ",
                "that ",
                "the pain is ",
                "pain "
            ]
            for prefix in prefixes_to_remove:
                if pos_clean.lower().startswith(prefix.lower()):
                    pos_clean = pos_clean[len(prefix):]
                    break
            
            # Capitalize first letter if needed
            if pos_clean and pos_clean[0].islower() and not pos_clean.lower().startswith(('arm', 'leg', 'chest')):
                pos_clean = pos_clean[0].lower() + pos_clean[1:]
            
            cleaned_other_positives.append(pos_clean)
        
        # Build proper sentence structure
        main_sentence = f"You reported {main_complaint}"

        # Add symptom descriptions
        symptom_descriptions = []
        for symptom in cleaned_other_positives:
            if symptom.lower().startswith(('heavy', 'severe', 'mild', 'moderate', 'sharp', 'dull', 'burning', 'crushing')):
                symptom_descriptions.append(f"described as {symptom}")
            elif symptom.lower().startswith(('worsens', 'improves', 'radiates', 'accompanied')):
                symptom_descriptions.append(symptom)
            else:
                symptom_descriptions.append(symptom)

        if symptom_descriptions:
            if len(symptom_descriptions) == 1:
                main_sentence += f" {symptom_descriptions[0]}"
            else:
                main_sentence += f" {pretty_join(symptom_descriptions, 'and')}"

        # Add timing information
        if timing_info:
            timing_str = pretty_join(timing_info, 'and')
            # Clean up timing strings
            timing_str = timing_str.replace("Onset was ", "").replace("starting Onset was ", "starting ")
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

def detect_condition(prompt: str, session_id: str = None, llm_chat_fn=None) -> Optional[str]:
    """Detect medical condition from prompt"""
    p = normalize_text(prompt)
    
    print(f"[Triage] 🔍 Detecting condition from: '{prompt}' (normalized: '{p}')")
    
    # Filter out casual conversation responses
    # Single-word responses like "bad", "good", "okay", "fine" are NOT medical conditions
    casual_responses = ["bad", "good", "okay", "fine", "well", "great", "terrible", "awful", "not great", "so so"]
    if p in casual_responses:
        print(f"[Triage] 💬 Casual response detected: '{p}' - not a medical condition")
        return None
    
    # Require minimum length for medical conditions (at least 2 words or contains medical keywords)
    words = p.split()
    medical_keywords = ["pain", "hurt", "ache", "symptom", "problem", "issue", "bleeding", "fever", "dizzy", "nausea", "vomiting"]
    has_medical_keyword = any(keyword in p for keyword in medical_keywords)
    
    if len(words) == 1 and not has_medical_keyword:
        print(f"[Triage] 💬 Single word without medical keyword: '{p}' - not detecting condition")
        return None
    
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
            if not has_medical_keyword:
                return None
    
    # CRITICAL: Check if there's already an active triage session
    # Don't detect new conditions if already in triage
    state = load_state(session_id)
    if state.get("condition"):
        print(f"[Triage] 🔒 Active triage for '{state['condition']}' - not detecting new conditions")
        return None
    
    # Use LLM-based intent classification for smarter detection
    if llm_chat_fn:
        print(f"[Triage] 🧠 Using LLM intent classifier")
        from intent_classifier import detect_medical_intent, map_condition_to_triage
        
        # Build conversation history from state
        conversation_history = []
        if state.get("phrasing_history"):
            for phrase in state.get("phrasing_history", [])[-3:]:
                conversation_history.append({"role": "assistant", "content": phrase})
        
        # Detect intent using LLM
        intent = detect_medical_intent(prompt, conversation_history, llm_chat_fn)
        
        if not intent.get("is_medical"):
            print(f"[Triage] 💬 Not medical: intent={intent.get('intent')}, confidence={intent.get('confidence')}")
            return None
        
        # Map LLM condition category to triage definition
        condition_category = intent.get("condition_category")
        if condition_category:
            mapped_condition = map_condition_to_triage(condition_category)
            if mapped_condition:
                print(f"[Triage] ✅ LLM detected: '{condition_category}' → mapped to '{mapped_condition}'")
                # Initialize detailed symptoms from extracted symptoms
                if "detailed_symptoms" not in state:
                    state["detailed_symptoms"] = []
                extracted = intent.get("extracted_symptoms", [])
                for symptom in extracted:
                    if symptom not in state["detailed_symptoms"]:
                        state["detailed_symptoms"].append(symptom)
                state["original_complaint"] = prompt
                save_state(state, session_id)
                return mapped_condition
    
    # Fallback to pattern matching if LLM not available or didn't detect condition
    print(f"[Triage] 🔍 Using fallback pattern matching")
    
    # Apply synonym expansion
    p_expanded = apply_synonym_expansion(p)
    print(f"[Triage] 🔍 After synonym expansion: '{p_expanded}'")
    
    # Initialize detailed symptoms
    if "detailed_symptoms" not in state:
        state["detailed_symptoms"] = []
    
    if p_expanded and p_expanded not in state["detailed_symptoms"]:
        state["detailed_symptoms"].append(p_expanded)
    
    save_state(state, session_id)
    
    # Find ALL potential matches and return the BEST one (not just first)
    best_match = None
    best_score = 0.0
    
    for cond, data in TRIAGE_DEFS.items():
        triggers = data.get("triggers", [])
        for trig in triggers:
            trig_norm = normalize_text(trig)
            
            # Exact match - highest priority
            if trig_norm in p_expanded:
                print(f"[Triage] ✅ Exact match: '{cond}' for trigger '{trig_norm}'")
                if "detailed_symptom" in TRIAGE_DEFS[cond]:
                    detailed_symptom = TRIAGE_DEFS[cond]["detailed_symptom"]
                    if detailed_symptom not in state["detailed_symptoms"]:
                        state["detailed_symptoms"].append(detailed_symptom)
                        save_state(state, session_id)
                return cond
            
            # Token-based fuzzy match
            ans_tokens = set(tokenize(p_expanded))
            trig_tokens = set(tokenize(trig_norm))
            overlap = len(ans_tokens & trig_tokens) / float(len(trig_tokens)) if trig_tokens else 0
            
            # Also check character-level similarity for typos (e.g., "abdomina" → "abdominal")
            p_words = p_expanded.split()
            trig_words = trig_norm.split()
            typo_match_score = 0.0
            
            for p_word in p_words:
                for trig_word in trig_words:
                    typo_score = check_typo_similarity(p_word, trig_word)
                    if typo_score > typo_match_score:
                        typo_match_score = typo_score
            
            # Calculate combined score (prioritize token overlap over typos)
            combined_score = (overlap * 1.5) + typo_match_score
            
            # Track best match
            if combined_score >= MIN_MATCH and combined_score > best_score:
                best_match = cond
                best_score = combined_score
                print(f"[Triage] 🔍 Candidate: '{cond}' (score={combined_score:.2f}, overlap={overlap:.2f}, typo={typo_match_score:.2f})")
    
    # Return best match if found
    if best_match:
        print(f"[Triage] ✅ Best match: '{best_match}' (score={best_score:.2f})")
        if "detailed_symptom" in TRIAGE_DEFS[best_match]:
            detailed_symptom = TRIAGE_DEFS[best_match]["detailed_symptom"]
            if detailed_symptom not in state["detailed_symptoms"]:
                state["detailed_symptoms"].append(detailed_symptom)
                save_state(state, session_id)
        return best_match
    
    return None


# === Triage Step Processing ===

def process_triage_step(prompt: str, state: Dict[str, Any], session_id: str, llm_chat_fn=None) -> Tuple[str, Dict[str, Any]]:
    """Process triage step and return next question"""
    condition = state.get("condition")
    if not condition:
        return "Please describe your symptoms to begin triage.", state

    # Get current step info
    # step_index represents "the next step to ask", so we're processing the previous step's answer
    next_step_index = state.get("step_index", 0)
    steps = get_steps(condition, state)
    step_list = [s if isinstance(s, dict) else {"key": None, "question": str(s)} for s in steps]
    
    # last_key tells us which question we actually asked and need to validate against
    last_key = state.get("last_key")

    print(f"[Triage] 🔍 Processing answer: condition={condition}, next_step_index={next_step_index}, last_key={last_key}")
    print(f"[Triage] 🔍 Total steps: {len(steps)}")

    # If we have a last_key, we're processing an answer to a previous question
    if last_key:
        print(f"[Triage] 🔍 Validating answer '{prompt}' for question key '{last_key}'")

        # Validate answer against the question we asked (last_key)
        if not is_valid_answer(condition, last_key, prompt, state):
            print(f"[Triage] ❌ Invalid answer '{prompt}' for question '{last_key}'")
            
            # Find the step that matches last_key to re-ask the correct question
            step_to_reask = None
            for s in step_list:
                if s.get("key") == last_key:
                    step_to_reask = s
                    break
            
            if step_to_reask:
                expected_answers = list(step_to_reask.get('answers', {}).keys())
                print(f"[Triage] 🔄 Re-asking question for key '{last_key}' (expected one of: {expected_answers})")
                return f"I didn't quite catch that. {substitute_name(step_to_reask.get('question', ''), state.get('user_name'))}", state
            else:
                # Fallback - this shouldn't happen
                print(f"[Triage] ⚠️ Could not find step with key '{last_key}'")
                return "I didn't quite catch that. Could you repeat your answer?", state

        # Answer is valid - add it and update flags
        print(f"[Triage] ✅ Valid answer for key '{last_key}', adding to state")
        state["answers"].append(prompt)
        update_flags_from_answer(condition, last_key, prompt, state, session_id)

        # We've processed this answer, continue to ask next question below
        print(f"[Triage] 🔄 Answer processed, will ask step {next_step_index}")
    
    # Check for pending clarify questions first
    if state.get("pending_clarify"):
        clarify_data = state["pending_clarify"]
        question = clarify_data.get("question", "")
        state["last_key"] = clarify_data.get("key")
        print(f"[Triage] ❓ Asking pending clarify question: {question}")
    # Otherwise, ask next main step question
    elif next_step_index < len(steps):
        next_step = steps[next_step_index]
        question = next_step.get("question", "")

        # Update last_key and step_index for the question we're about to ask
        state["last_key"] = next_step.get("key")
        state["step_index"] = next_step_index + 1  # Next time, we'll ask the following question
        print(f"[Triage] 📝 Asking step {next_step_index}, key='{state['last_key']}', next_step_index will be {state['step_index']}")
    else:
        question = None

    # Apply NLG rewriting if we have a question to ask
    if question:
        # Apply NLG rewriting (using simple fallback like old version)
        from nlg import rewrite
        def llm_chat_once_fallback(messages, **kwargs):
            """Simple fallback for NLG rewriting - just return the question"""
            return {"content": question}

        # Determine the question key for NLG context
        question_key = state.get("last_key")
        allowed_answers = []

        # Get allowed answers for the question
        if state.get("pending_clarify"):
            allowed_answers = list(state["pending_clarify"].get("answers", {}).keys())
        elif next_step_index < len(steps):
            next_step = steps[next_step_index]
            allowed_answers = list(next_step.get("answers", {}).keys())

        rewritten_question = rewrite(
            question,
            "question",
            {
                "name": state.get("user_name"),
                "condition": state["condition"],
                "key": question_key,
                "allowed_answers": allowed_answers
            },
            state.get("phrasing_history", []),
            llm_chat_once_fallback
        )

        final_question = substitute_name(rewritten_question, state.get("user_name"))
        return final_question, state
    else:
        # No question to ask - check if triage is complete
        if not state.get("pending_clarify") and next_step_index >= len(steps):
            # Triage complete - generate recap and reset session
            recap_response = generate_triage_completion(state, session_id, llm_chat_fn)
            # Reload state after reset
            reset_state = load_state(session_id)
            return recap_response, reset_state
        else:
            # No question to ask but triage not complete - this shouldn't happen
            print(f"[Triage] ⚠️ No question to ask but triage not complete")
            return "I'm having trouble with the next question. Could you describe your symptoms again?", state


def generate_triage_completion(state: Dict[str, Any], session_id: str, llm_chat_fn=None) -> str:
    """Generate final triage completion and reset session"""
    condition = state.get("condition")
    answers = state.get("answers", [])
    flags = state.get("flags", {})
    
    if not condition:
        return "I'm sorry, there was an error processing your triage."
    
    # Classify severity
    severity = classify_response(condition, flags)
    
    # Build recap
    recap = build_recap(condition, answers, flags, severity, session_id)

    # Generate sophisticated LLM-based outcome instead of hardcoded responses
    if llm_chat_fn:
        outcome = _generate_llm_outcome(condition, severity, answers, flags, state, llm_chat_fn)
    else:
        # Fallback to existing logic if LLM not available
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
    
    # IMPORTANT: Reset session state after completion to prevent infinite loop
    # Preserve user name across resets
    user_name = state.get("user_name")
    reset_state = {
        "condition": None, "step_index": 0, "answers": [], "flags": {},
        "last_key": None, "user_name": user_name,
        "active_pathway": None, "entered_pathway": False,
        "updated_at": None, "phrasing_history": [], "detailed_symptoms": [],
        "original_complaint": None, "expanded_prompt": None, "mode": None
    }
    save_state(reset_state, session_id)
    print(f"[Triage] ✅ Triage completed - session reset for {session_id}")
    
    return f"{recap} {outcome}"


def _generate_llm_outcome(condition: str, severity: str, answers: List[str], flags: Dict, state: Dict, llm_chat_fn) -> str:
    """
    Generate sophisticated triage outcome using LLM

    Args:
        condition: Medical condition detected
        severity: Classified severity level
        answers: User's answers to triage questions
        flags: Additional context flags
        state: Full session state
        llm_chat_fn: LLM chat function

    Returns:
        Sophisticated outcome recommendation
    """
    # Build context for LLM
    user_name = state.get("user_name", "patient")
    original_complaint = state.get("original_complaint", "")
    chief_complaint = state.get("detailed_symptoms", [""])[0] if state.get("detailed_symptoms") else original_complaint

    # Build summary of answers for context
    answers_summary = []
    for i, answer in enumerate(answers):
        answers_summary.append(f"Q{i+1}: {answer}")

    answers_context = "\n".join(answers_summary) if answers_summary else "No specific answers provided."

    # Build clinical context
    clinical_summary = TRIAGE_DEFS[condition].get("clinical_summary", "")
    
    # Get the JSON-defined outcome for this severity level (as guidance)
    active_pathway = state.get("active_pathway")
    outcomes = TRIAGE_DEFS[condition].get("outcomes", {})
    
    if active_pathway and "pathways" in TRIAGE_DEFS[condition]:
        pathway_outcomes = TRIAGE_DEFS[condition]["pathways"][active_pathway].get("outcomes", {})
        if pathway_outcomes:
            outcomes = pathway_outcomes
    
    json_outcome = outcomes.get(severity, "")

    system_prompt = """You are a medical triage AI providing clinical assessments and care recommendations.

CRITICAL INSTRUCTIONS:
- You MUST provide probable diagnoses based on the symptoms and triage data
- Use clinical language: "This is concerning for [diagnosis]" or "These symptoms suggest [diagnosis]"
- Be specific about the medical condition (e.g., "acute cholangitis", "appendicitis", "myocardial infarction")
- State the severity clearly (emergency, urgent, or non-urgent)
- Provide specific, actionable next steps
- Include timeframes for seeking care
- For emergencies, emphasize calling 911 or going to ER immediately
- Be conversational but clinically accurate
- Do NOT say "I can't provide a diagnosis" - you are providing a triage assessment

Example emergency response: "Your symptoms of fever, jaundice, and vomiting are concerning for acute cholangitis. This is a medical emergency. Please call 911 or go to the nearest emergency room immediately."
Example urgent response: "Your symptoms suggest acute appendicitis and require urgent evaluation. Please go to an urgent care or emergency room within 2-4 hours."
"""

    user_prompt = f"""Patient: {user_name}
Chief Complaint: {chief_complaint}
Condition: {condition.replace('_', ' ').title()}
Severity: {severity.title()}
Clinical Assessment: {clinical_summary}

Patient's Answers:
{answers_context}

Clinical Guidance (from triage protocol):
{json_outcome}

Based on this triage assessment, provide a specific clinical assessment with probable diagnosis and care recommendation.

Your response must:
1. State what the symptoms are concerning for (probable diagnosis)
2. Clearly state the severity level (emergency/urgent/non-urgent)
3. Provide specific next steps (call 911, ER, urgent care, or scheduled appointment)
4. Include timeframes if appropriate"""

    try:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        response = llm_chat_fn(
            messages=messages,
            max_tokens=300,
            temperature=0.7,
            stream=False
        )

        content = response.get("choices", [{}])[0].get("message", {}).get("content", "")
        outcome = content.strip() if content else _get_fallback_outcome(severity)

        # Ensure it's addressed to the patient if name is provided
        if user_name and not user_name.lower() in ["patient", "user"]:
            # LLM might not use the name, so add it if needed
            if not user_name.lower() in outcome.lower():
                outcome = f"{user_name}, {outcome}"

        return outcome

    except Exception as e:
        print(f"[Triage] ❌ Error generating LLM outcome: {e}")
        return _get_fallback_outcome(severity)


def _get_fallback_outcome(severity: str) -> str:
    """Fallback outcome when LLM fails"""
    if severity == "emergency":
        return "Seek emergency medical care immediately (call 911 or go to nearest ER)."
    elif severity == "urgent":
        return "Seek medical care within 2-4 hours (urgent care or ER if symptoms worsen)."
    else:
        return "Schedule appointment with primary care physician within 24-48 hours."


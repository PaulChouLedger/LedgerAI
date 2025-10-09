#!/usr/bin/env python3
"""
Aura TRIAGE Mode - Hardcoded Medical Diagnostic System

Current baseline diagnostic system using:
- JSON-based condition definitions (triage_defs/)
- Structured question flow
- Severity scoring
- SOAP-style clinician recap

This is the working baseline that CLINICIAN mode will eventually replace.
"""

import os
import json
import re
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


# === Core Triage Functions ===

def detect_condition(prompt: str, session_id: str | None = None) -> Optional[str]:
    """
    Detect medical condition from prompt using triage definitions

    Args:
        prompt: User's input
        session_id: Optional session identifier

    Returns:
        Detected condition name or None
    """
    p = normalize_text(prompt)
    print(f"[Triage] 🔍 Original prompt: '{prompt}'")
    print(f"[Triage] 🔍 Normalized prompt: '{p}'")

    # Check for casual greetings first - don't trigger triage for these
    import re
    casual_greeting_patterns = [
        r'\bhello\b', r'\bhi\b', r'\bhey\b', r'\bhowdy\b',
        r'\bgood morning\b', r'\bgood afternoon\b', r'\bgood evening\b',
        r'\bhello aura\b', r'\bhi aura\b', r'\bhey aura\b'
    ]

    # Check if it's a knowledge query - DON'T trigger triage for these
    knowledge_indicators = ["tell me", "what is", "who is", "explain", "describe", "information about",
                           "details about", "everything about", "all about"]
    is_knowledge_query = any(indicator in p for indicator in knowledge_indicators)

    if is_knowledge_query:
        print(f"[Triage] 💬 Knowledge query detected - not triggering triage")
        return None

    # Only check for greetings if it's NOT a knowledge query
    is_casual_greeting = any(re.search(pattern, p) for pattern in casual_greeting_patterns)

    if is_casual_greeting:
        # Check if there are any medical symptoms mentioned
        medical_keywords = ["pain", "hurt", "ache", "symptom", "problem", "issue", "concern", "worried", "sick", "ill", "unwell"]
        has_medical_content = any(keyword in p for keyword in medical_keywords)

        if not has_medical_content:
            print(f"[Triage] 💬 Casual greeting detected: '{p}' -> no triage trigger")
            return None
        else:
            print(f"[Triage] 💬 Greeting with medical content detected: '{p}' -> proceeding with triage")

    # Apply synonym expansion
    p_expanded = apply_synonym_expansion(p)
    print(f"[Triage] 🔄 Expanded prompt: '{p_expanded}'")
    print(f"[Triage] 🔍 Checking for triage triggers in: '{p_expanded}'")

    # Check each condition's triggers
    for condition_name, condition_def in TRIAGE_DEFS.items():
        triggers = condition_def.get('triggers', [])

        for trigger in triggers:
            trigger_normalized = normalize_text(trigger)

            # Check for exact match or substring match
            if trigger_normalized in p_expanded or p_expanded in trigger_normalized:
                print(f"[Triage] ✅ Matched trigger '{trigger}' for condition '{condition_name}'")
                return condition_name

    print(f"[Triage] ❌ No condition matched for prompt: '{p_expanded}'")
    return None


def process_triage_step(prompt: str, state: Dict[str, Any], session_id: str) -> Tuple[str, Dict[str, Any]]:
    """
    Process a single triage step answer

    Args:
        prompt: User's answer to current question
        state: Current triage state
        session_id: Session identifier

    Returns:
        Tuple of (next_question, updated_state)
    """
    condition = state.get("condition")
    if not condition:
        return "Please describe your symptoms to begin triage.", state

    # Add answer to state
    state["answers"].append(prompt)

    # Update flags from the answer
    current_step_index = state.get("step_index", 0)
    steps = get_steps(condition, state)
    current_step = steps[current_step_index] if current_step_index < len(steps) else None

    if current_step:
        update_flags_from_answer(condition, current_step.get("key"), prompt, state, session_id)

    # Advance to next step
    state["step_index"] = current_step_index + 1

    # Get next step
    if state["step_index"] < len(steps):
        # Get current step
        next_step = steps[state["step_index"]]
        question = next_step.get("question", "")

        # Apply NLG rewriting if needed
        from nlg import rewrite
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
            lambda messages, gen_kwargs: {"content": question}  # Simple fallback
        )

        # Substitute name in question
        final_question = substitute_name(rewritten_question, state.get("user_name"))
        return final_question, state

    else:
        # Triage complete - generate final recap
        recap_response = generate_triage_completion(state, session_id)
        return recap_response, state


def generate_triage_completion(state: Dict[str, Any], session_id: str) -> str:
    """
    Generate final triage completion with SOAP-style recap

    Args:
        state: Final triage state
        session_id: Session identifier

    Returns:
        Complete triage summary and recommendation
    """
    condition = state.get("condition")
    answers = state.get("answers", [])
    flags = state.get("flags", {})
    user_name = state.get("user_name")

    if not condition:
        return "I'm sorry, there was an error processing your triage."

    # Build comprehensive recap
    recap_parts = []

    # Subjective section
    if user_name:
        recap_parts.append(f"**Subjective:** {user_name} reported symptoms consistent with {condition.replace('_', ' ')}.")

    # Add detailed symptoms
    if state.get("detailed_symptoms"):
        detailed_symptoms_text = "; ".join(state["detailed_symptoms"])
        recap_parts.append(f"**Detailed Symptoms:** {detailed_symptoms_text}")

    # Add answers in structured format
    if answers:
        recap_parts.append("**History Provided:**")
        for i, answer in enumerate(answers):
            recap_parts.append(f"  {i+1}. {answer}")

    # Assessment (severity and urgency)
    severity = "unknown"
    urgency = "non_urgent"
    for key, value in flags.items():
        if key in ["emergency", "urgent"]:
            urgency = key
        elif key in ["severe", "moderate", "mild"]:
            severity = key

    recap_parts.append(f"**Assessment:** {severity.title()} severity, {urgency.replace('_', ' ').title()} priority.")

    # Plan (recommendation)
    if urgency == "emergency":
        recommendation = "Seek emergency medical care immediately (call 911 or go to nearest ER)."
    elif urgency == "urgent":
        recommendation = "Seek medical care within 2-4 hours (urgent care or ER if symptoms worsen)."
    else:
        recommendation = "Schedule appointment with primary care physician within 24-48 hours."

    recap_parts.append(f"**Plan:** {recommendation}")

    # Add any specific flags
    if flags:
        flag_summary = []
        for key, value in flags.items():
            if value:  # Only show positive flags
                flag_summary.append(key.replace('_', ' ').title())
        if flag_summary:
            recap_parts.append(f"**Key Findings:** {', '.join(flag_summary)}")

    # Compile full recap
    full_recap = "\n\n".join(recap_parts)

    # Reset state for next session
    state.update({
        "condition": None,
        "step_index": 0,
        "answers": [],
        "flags": {},
        "last_key": None,
        "active_pathway": None,
        "entered_pathway": False
    })

    return full_recap


def get_steps(condition: str, state: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Get triage steps for a condition, considering pathways

    Args:
        condition: Condition name
        state: Current triage state

    Returns:
        List of step dictionaries
    """
    if condition not in TRIAGE_DEFS:
        return []

    condition_def = TRIAGE_DEFS[condition]
    steps = condition_def.get('steps', [])

    # Check for pathway selection
    active_pathway = state.get('active_pathway')
    if active_pathway and 'pathways' in condition_def:
        pathway_steps = condition_def['pathways'].get(active_pathway, {}).get('steps', [])
        if pathway_steps:
            return pathway_steps

    return steps


def update_flags_from_answer(condition: str, last_key: str, answer: str, state: Dict[str, Any], session_id: str):
    """
    Update triage flags based on answer to previous question

    Args:
        condition: Current condition
        last_key: Key of the question that was answered
        answer: User's answer
        state: Current triage state
        session_id: Session identifier
    """
    if condition not in TRIAGE_DEFS:
        return

    condition_def = TRIAGE_DEFS[condition]
    steps = get_steps(condition, state)

    # Find the step that was just answered
    for step in steps:
        if step.get("key") == last_key:
            answers_dict = step.get("answers", {})

            # Check for exact match first
            if answer in answers_dict:
                flag = answers_dict[answer]
                if flag:
                    state["flags"][flag] = True
                    print(f"[Triage] 🚩 Set flag: {flag}")
                return

            # Check for fuzzy matching
            for expected_answer, flag in answers_dict.items():
                if fuzzy_match_answer(answer, expected_answer):
                    if flag:
                        state["flags"][flag] = True
                        print(f"[Triage] 🚩 Fuzzy matched '{answer}' → '{expected_answer}' → flag: {flag}")
                    return


def fuzzy_match_answer(user_answer: str, expected_answer: str) -> bool:
    """
    Check if user answer matches expected answer using fuzzy matching

    Args:
        user_answer: What user said
        expected_answer: Expected answer from triage definition

    Returns:
        True if matches (exact or fuzzy)
    """
    user_lower = user_answer.lower().strip()
    expected_lower = expected_answer.lower().strip()

    # Exact match
    if user_lower == expected_lower:
        return True

    # Fuzzy matching for common variations
    if expected_lower == "yes":
        return user_lower in ["yes", "yeah", "yep", "sure", "ok", "okay", "y"]
    elif expected_lower == "no":
        return user_lower in ["no", "nope", "nah", "not", "n"]

    return False


# === Helper Functions ===

def normalize_text(text: str) -> str:
    """Normalize text for comparison (lowercase, strip punctuation)"""
    if not text:
        return ""

    # Convert to lowercase
    text = text.lower()

    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))

    # Normalize whitespace
    text = ' '.join(text.split())

    return text


def apply_synonym_expansion(text: str) -> str:
    """
    Apply synonym expansion to improve trigger matching

    Args:
        text: Text to expand

    Returns:
        Text with synonyms added
    """
    synonyms = {
        "chest": ["chest", "thoracic", "heart"],
        "pain": ["pain", "hurt", "ache", "discomfort", "pressure"],
        "head": ["head", "cephalic", "cranial"],
        "stomach": ["stomach", "abdominal", "belly", "gastric"],
        "breathing": ["breathing", "respiration", "breath"],
        "difficulty": ["difficulty", "trouble", "problem", "hard"],
        "shortness": ["shortness", "short"],
        "breath": ["breath", "breathing", "respiration"],
    }

    expanded = text.lower()

    for key, syn_list in synonyms.items():
        for synonym in syn_list:
            if synonym in expanded:
                # Add other synonyms to improve matching
                for other_syn in syn_list:
                    if other_syn != synonym:
                        expanded += f" {other_syn}"

    return expanded


def substitute_name(text: str, name: str) -> str:
    """
    Substitute name placeholders in text

    Args:
        text: Text with placeholders
        name: User's name

    Returns:
        Text with names substituted
    """
    if not name:
        return text

    # Replace common placeholders
    replacements = {
        "{name}": name,
        "{Name}": name,
        "{{name}}": name,
        "{{Name}}": name,
    }

    for placeholder, replacement in replacements.items():
        text = text.replace(placeholder, replacement)

    return text


def triage_is_stale(state: Dict[str, Any]) -> bool:
    """
    Check if triage session is stale (too old)

    Args:
        state: Triage state

    Returns:
        True if stale
    """
    if not state.get("condition"):
        return False

    updated_at = state.get("updated_at")
    if not updated_at:
        return True

    try:
        updated_time = datetime.fromisoformat(updated_at)
        now = datetime.now()

        # Consider stale if more than 24 hours old
        if (now - updated_time).total_seconds() > 24 * 60 * 60:
            return True

    except (ValueError, TypeError):
        return True

    return False


def load_state(session_id: str) -> Dict[str, Any]:
    """
    Load session state from file

    Args:
        session_id: Session identifier

    Returns:
        Session state dictionary
    """
    if not session_id:
        return {"condition": None, "step_index": 0, "answers": [], "flags": {},
                "last_key": None, "user_name": None, "active_pathway": None,
                "entered_pathway": False, "updated_at": None, "phrasing_history": [],
                "detailed_symptoms": [], "original_complaint": None, "expanded_prompt": None}

    state_file = os.path.join("session_states", f"{session_id}.json")

    try:
        if os.path.exists(state_file):
            with open(state_file, 'r') as f:
                state = json.load(f)

            # Check if state is stale
            if triage_is_stale(state):
                print(f"[Triage] 🕰️  Session {session_id} is stale, resetting")
                state = {"condition": None, "step_index": 0, "answers": [], "flags": {},
                        "last_key": None, "user_name": None, "active_pathway": None,
                        "entered_pathway": False, "updated_at": None, "phrasing_history": [],
                        "detailed_symptoms": [], "original_complaint": None, "expanded_prompt": None}

            return state
        else:
            return {"condition": None, "step_index": 0, "answers": [], "flags": {},
                   "last_key": None, "user_name": None, "active_pathway": None,
                   "entered_pathway": False, "updated_at": None, "phrasing_history": [],
                   "detailed_symptoms": [], "original_complaint": None, "expanded_prompt": None}
    except Exception as e:
        print(f"[Triage] ❌ Error loading state for session {session_id}: {e}")
        return {"condition": None, "step_index": 0, "answers": [], "flags": {},
               "last_key": None, "user_name": None, "active_pathway": None,
               "entered_pathway": False, "updated_at": None, "phrasing_history": [],
               "detailed_symptoms": [], "original_complaint": None, "expanded_prompt": None}


def save_state(state: Dict[str, Any], session_id: str):
    """
    Save session state to file

    Args:
        state: State to save
        session_id: Session identifier
    """
    if not session_id:
        return

    # Ensure session_states directory exists
    os.makedirs("session_states", exist_ok=True)

    state_file = os.path.join("session_states", f"{session_id}.json")

    try:
        # Add timestamp
        state["updated_at"] = datetime.now().isoformat()

        with open(state_file, 'w') as f:
            json.dump(state, f, indent=2)

    except Exception as e:
        print(f"[Triage] ❌ Error saving state for session {session_id}: {e}")


# Initialize triage definitions on import
load_triage_definitions()


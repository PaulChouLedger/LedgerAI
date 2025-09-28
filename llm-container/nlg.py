import os
import re
from typing import Callable, Dict, List, Literal, Optional

Role = Literal["intro","question","clarify","recap","outcome"]

def _fingerprint(text: str) -> str:
    text = re.sub(r"\s+", " ", (text or "").strip()).lower()
    return text[:120]

def rewrite(
    text: str,
    role: Role,
    context: Dict,
    phrasing_history: Optional[List[str]],
    chat_fn: Callable[[list, dict], Dict]
) -> str:
    """Paraphrase the given text with clinician tone while preserving facts.

    - text: the sentence to rewrite
    - role: where this text is used (intro, question, clarify, recap, outcome)
    - context: may include { name, condition, pathway, key, allowed_answers }
    - phrasing_history: prior short phrases to avoid repeating
    - chat_fn: callable(messages, gen_kwargs) -> dict completion (non-stream)
    """
    if not text:
        return text

    if os.getenv("NLG_ENABLED", "1") not in ("1", "true", "TRUE", "yes"):
        return text
    
    print(f"[NLG] 🔄 Rewriting text: '{text}'")
    print(f"[NLG] 🔄 Role: {role}, Context: {context}")
    print(f"[NLG] 🔄 Phrasing history: {phrasing_history}")

    allowed = context.get("allowed_answers") or []
    name = context.get("name") or ""
    
    # Count name usage in recent history to reduce repetition
    name_count = 0
    if phrasing_history:
        for phrase in phrasing_history[-5:]:  # Check last 5 phrases
            if name and name.lower() in phrase.lower():
                name_count += 1
    
    # Build prompt
    system = (
        "You are a medical assistant. Your task is to rewrite the provided text to sound natural and professional while preserving all clinical facts exactly. "
        "CRITICAL: You must return ONLY the rewritten text. Do not include any instructions, meta-commentary, or explanations. "
        "Preserve clinical facts exactly. Avoid repetition and canned phrasing. "
        "Do not add medical advice beyond what is given. Use second-person voice. "
        "CRITICAL: Use the patient's name ONLY in intros and final recaps/outcomes. "
        "For questions, NEVER use the name - just ask directly. "
        "For clarify questions, NEVER use the name. "
        "If the name has been used recently, avoid using it again even in intros/recaps. "
        "IMPORTANT: Do NOT add redundant questions. If the text already asks about a symptom, do not add another question about the same symptom. "
        "CRITICAL: For factual questions (age, timing, history, yes/no facts), preserve the exact meaning. "
        "Do NOT change 'Are you older than 50?' to 'Do you feel older than 50?' - these have different meanings. "
        "IMPORTANT: Do NOT add professional titles like 'nurse', 'doctor', 'physician' unless they are in the original text. "
        "Keep the language neutral and professional without adding unnecessary titles. "
        "CRITICAL: Do NOT add unnecessary words like 'acknowledge', 'understand', 'recognize' unless they improve clarity. "
        "If the original text is already clear and clinical, make minimal changes. "
        "IMPORTANT: For questions, if the original is already clear and direct, return it unchanged. Do not make simple questions more complex or wordy. "
        "CRITICAL: Make questions clear and contextual. If a patient mentions 'dark stools', don't ask 'Do you have bloody stools?' - ask about the specific color or characteristics instead. "
        "FOR RECAPS: Preserve ALL clinical details including specific symptoms, denied symptoms, and clinical assessments. Do not summarize away important medical information. "
        "CRITICAL: NEVER add symptoms that are not mentioned in the original text. Do not hallucinate or invent symptoms like 'dark stools', 'nausea', or 'abdominal pain' unless they are explicitly stated. "
        "CRITICAL: NEVER add demographic information like age, gender, or other personal details unless they are explicitly mentioned in the original text. Do not invent ages like '50 years old' or other demographic facts. "
        "IMPORTANT: Do NOT add temporal references like 'today', 'now', 'currently', 'at this time' unless they are in the original text. Keep the language timeless and clinical. "
        "CRITICAL: Do NOT add markdown formatting like **bold**, *italics*, or other formatting unless it is in the original text. Keep the output clean and plain text. NEVER use **, *, or any markdown syntax. Output must be plain text only. "
        "IMPORTANT: Preserve anatomical terms exactly. Do NOT change 'both' to 'both sides' when referring to limbs. Keep 'arm', 'leg', 'both' as specified in the original text. "
        "CRITICAL: Vary your vocabulary - avoid repetitive words like 'affected', 'suggest', 'concerning' in rapid succession. Use synonyms and varied phrasing. "
        "ABSOLUTELY FORBIDDEN: Never use 'X can suggest Y. Let's clarify' pattern - this is repetitive and robotic. "
        "FOR PATHWAY INTROS: Make them natural and varied. Use diverse approaches like 'Let me ask about X', 'I need to know more about X', 'Tell me about X', 'Now I need to know about X', 'Let's check for X', or simply ask the first question directly. "
        "AVOID THESE OVERUSED PHRASES: 'can suggest', 'Let's clarify', 'Let's check', 'We need to know'. Use fresh, varied language instead. "
        "CRITICAL: Do NOT repeat the same phrase multiple times. If you find yourself repeating text, stop and provide a single, clear version. "
        "ABSOLUTELY FORBIDDEN: Do not repeat the same sentence or phrase more than once. Each sentence should be unique and add new information. "
        "EXTREME ANTI-REPETITION: If you detect that you are repeating the same text pattern (like 'left lower quadrant pain with fever or bowel changes suggests diverticulitis'), STOP immediately and provide a completely different formulation. "
        "FOR OUTCOMES: Keep it brief and single. Do not repeat the same diagnostic phrase multiple times. One clear statement is sufficient."
    )

    # Few-shot style hints per role
    role_hint = {
        "intro": "Rewrite to acknowledge the patient and transition to the next step naturally. Use the name once at the start only if provided. For pathway intros, vary the approach - NEVER use 'X can suggest Y. Let's clarify'. Use fresh phrases like 'Let me ask about X', 'Tell me about X', or ask directly.",
        "question": "Rewrite the question to be clear and direct. Do not use the patient's name. Keep it concise and clinical. Vary vocabulary to avoid repetition.",
        "clarify": "Rewrite as a short, clear follow-up question. Do not use the patient's name. Start with 'Do you...' or 'Are you...'",
        "recap": "Rewrite as a clinical summary. Use the name once at start only if provided. Preserve all clinical details exactly.",
        "outcome": "Rewrite as a clear disposition statement. Use the name once at start only if provided. Do not repeat symptoms already mentioned. Keep it brief - one clear sentence only. Never repeat the same phrase multiple times."
    }.get(role, "Rewrite clearly and briefly.")

    history = phrasing_history or []
    avoid = "; ".join(history[-5:]) if history else ""
    
    # Check for redundant questions in history
    if role in ("question", "clarify") and history:
        recent_questions = [h.lower() for h in history[-3:] if "?" in h]
        if any("headache" in q for q in recent_questions) and "headache" in text.lower():
            print(f"[NLG] ⚠️ Detected potential redundant headache question")
            print(f"[NLG] ⚠️ Recent questions: {recent_questions}")
            print(f"[NLG] ⚠️ Current text: {text}")
            # Return original text without NLG processing to avoid redundancy
            return text
    
    
    # Check for factual questions that should not be rephrased
    factual_patterns = [
        r"are you older than \d+",
        r"are you over \d+", 
        r"are you under \d+",
        r"are you \d+ years old",
        r"did the .* reach its worst",
        r"was the .* preceded by",
        r"do you have a history of",
        r"have you had",
        r"did you experience",
        r"were you involved in",
        r"is this the worst",
        r"did this start",
        r"has this been going on",
        r"how long have you had"
    ]
    
    text_lower = text.lower()
    for pattern in factual_patterns:
        if re.search(pattern, text_lower):
            print(f"[NLG] 🔒 Preserving factual question: '{text}'")
            return text

    # Determine if we should use the name based on role and recent usage
    should_use_name = role in ("intro", "recap", "outcome") and name_count < 1
    
    user_content = f"""
Text to rewrite: "{text}"

Role: {role}
Style: {role_hint}
Name: {name if should_use_name else "None"}
Avoid repeating: {avoid}

Rewrite the text above to be natural and professional while preserving all clinical facts exactly. Return only the rewritten text, not instructions or meta-commentary.
"""

    # Check for repetitive patterns and add context
    if role == "intro" and "suggest" in text.lower() and "clarify" in text.lower():
        print(f"[NLG] ⚠️ Detected repetitive 'suggest...clarify' pattern in intro")
        print(f"[NLG] ⚠️ Original text: {text}")
        user_content += f"\n\nIMPORTANT: The original text uses repetitive 'suggest...clarify' pattern. Rewrite it completely differently - use varied vocabulary and fresh phrasing."
    
    if role == "question" and "during these episodes" in text.lower():
        recent_questions = [h.lower() for h in history[-3:] if "during these episodes" in h]
        if len(recent_questions) >= 2:  # If we've used this phrase recently
            print(f"[NLG] ⚠️ Detected repetitive 'during these episodes' pattern")
            print(f"[NLG] ⚠️ Recent questions: {recent_questions}")
            user_content += f"\n\nIMPORTANT: Avoid using 'during these episodes' as it's been used repeatedly. Rewrite with different phrasing like 'Do you experience...' or 'Have you noticed...'"
    
    if role == "outcome":
        text_lower = text.lower()
        if "suggests" in text_lower and text_lower.count("suggests") > 1:
            print(f"[NLG] ⚠️ Detected repetitive 'suggests' pattern in outcome")
            print(f"[NLG] ⚠️ Original text: {text}")
            user_content += f"\n\nCRITICAL: The original text repeats diagnostic phrases. Rewrite as ONE clear, brief statement without repetition."
        
        if ("cardiac emergency" in text_lower or "call 911" in text_lower) and any("emergency" in h.lower() for h in history[-2:]):
            print(f"[NLG] ⚠️ Detected repetitive emergency messaging")
            print(f"[NLG] ⚠️ Original text: {text}")
            user_content += f"\n\nIMPORTANT: The previous outcome already mentioned emergency care. Rewrite to avoid repetition - focus on a single, clear directive."

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": str(user_content)}
    ]

    gen_kwargs = {
        "temperature": float(os.getenv("NLG_TEMPERATURE", "0.5")),
        "top_p": float(os.getenv("NLG_TOP_P", "0.85")),
        "max_tokens": int(os.getenv("NLG_MAX_TOKENS", "128"))
    }

    try:
        result = chat_fn(messages, gen_kwargs)
        content = (
            result.get("choices", [{}])[0]
                  .get("message", {})
                  .get("content", "")
        )
        text_out = (content or text).strip()
        print(f"[NLG] 🔄 Raw NLG output: '{text_out}'")
        
        # Post-process to remove name if it shouldn't be used
        if not should_use_name and name and name.lower() in text_out.lower():
            # Remove name from the beginning of the text (with comma)
            text_out = re.sub(rf"^{re.escape(name)},\s*", "", text_out, flags=re.IGNORECASE)
            # Remove name from the beginning of the text (without comma)
            text_out = re.sub(rf"^{re.escape(name)}\s+", "", text_out, flags=re.IGNORECASE)
            # Remove "You reported your name is [name]" pattern
            text_out = re.sub(rf"You reported your name is {re.escape(name)}", "You reported", text_out, flags=re.IGNORECASE)
            # Remove "You reported [name]" pattern
            text_out = re.sub(rf"You reported {re.escape(name)}", "You reported", text_out, flags=re.IGNORECASE)
            print(f"[NLG] 🚫 Removed name '{name}' from text: '{text_out}'")
        # Basic cleanup
        text_out = re.sub(r"\s+([.,;:!?])", r"\1", text_out)
        text_out = re.sub(r"\s{2,}", " ", text_out).strip()
        
        # Clean up awkward emergency messaging
        if role == "outcome":
            # Fix redundant "if your symptoms concern" phrasing
            text_out = re.sub(r"if your symptoms concern a serious condition such as", "for serious conditions such as", text_out, flags=re.IGNORECASE)
            text_out = re.sub(r"if your symptoms are concerning for", "for", text_out, flags=re.IGNORECASE)
            # Simplify emergency messaging
            text_out = re.sub(r"Please seek immediate emergency care or call 911 for", "Please seek immediate emergency care or call 911 for", text_out)
            text_out = re.sub(r"Please seek immediate care or call 911 for", "Please seek immediate emergency care or call 911 for", text_out)
        # Ensure question style for question/clarify
        if role in ("question","clarify") and not text_out.endswith(("?",".")):
            text_out += "?"
        print(f"[NLG] ✅ Final NLG output: '{text_out}'")
        return text_out
    except Exception:
        return text



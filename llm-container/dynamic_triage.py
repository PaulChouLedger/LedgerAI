#!/usr/bin/env python3
"""
Dynamic Triage Question Generation & Validation
Replaces rigid JSON questions with LLM-generated natural conversations
"""

import json
import re
from typing import Dict, List, Optional, Callable, Any


def generate_dynamic_question(
    step: Dict[str, Any],
    context: Dict[str, Any],
    conversation_history: List[str],
    llm_chat_fn: Callable
) -> str:
    """
    Generate a natural, context-aware question using LLM
    
    Args:
        step: Current step from JSON (provides guidance, not rigid template)
        context: Conversation context (condition, pathway, prior answers)
        conversation_history: Recent exchanges
        llm_chat_fn: LLM chat function
        
    Returns:
        Natural question string
    """
    step_key = step.get("key", "")
    base_question = step.get("question", "")
    
    # Build context for LLM
    prior_answers = context.get("prior_answers", [])
    condition = context.get("condition", "")
    pathway = context.get("pathway", "")
    
    system_prompt = """You are a medical triage assistant conducting a natural conversation with a patient.

CRITICAL INSTRUCTIONS:
- Generate ONE natural, conversational question based on the guidance provided
- Be empathetic and professional
- Keep questions SHORT - one sentence maximum
- Adapt your phrasing to the conversation flow
- Do NOT repeat questions that have already been asked
- Make the patient feel heard and cared for
- Use "you" and "your" - speak directly to the patient
- Do NOT add explanations or multiple questions

Examples:
Guidance: "Ask about onset"
Good: "When did this start?"
Good: "How long have you been experiencing this?"
Bad: "When did the abdominal pain start? I need to know the timing to assess severity."

Guidance: "Ask about fever"
Good: "Have you had any fever?"
Good: "Do you feel feverish?"
Bad: "Do you have fever? This is important for diagnosis."
"""

    # Build conversation context
    conversation_text = ""
    if conversation_history:
        recent = conversation_history[-3:]
        conversation_text = "\n".join(recent)
    
    prior_answers_text = ""
    if prior_answers:
        prior_answers_text = "Already asked about:\n" + "\n".join(f"- {ans}" for ans in prior_answers[-3:])
    
    user_prompt = f"""Medical Context:
- Condition: {condition}
- Current focus: {pathway}

{prior_answers_text}

Question Guidance: {base_question}
Key to assess: {step_key}

Generate a natural, conversational question to ask the patient. Output ONLY the question, nothing else."""

    try:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        response = llm_chat_fn(
            messages=messages,
            max_tokens=50,
            temperature=0.7,
            stream=False
        )
        
        content = response.get("choices", [{}])[0].get("message", {}).get("content", "")
        question = content.strip()
        
        # Clean up any extra punctuation or formatting
        question = question.strip('"\'')
        if not question.endswith('?'):
            question += '?'
        
        print(f"[DynamicTriage] 💬 Generated question: {question}")
        return question
    
    except Exception as e:
        print(f"[DynamicTriage] ❌ Error generating question: {e}")
        # Fallback to base question
        return base_question


def validate_and_extract_answer(
    user_response: str,
    step: Dict[str, Any],
    context: Dict[str, Any],
    llm_chat_fn: Callable
) -> Dict[str, Any]:
    """
    Use LLM to validate answer and extract information
    
    Args:
        user_response: What the user said
        step: Current step definition (for guidance)
        context: Conversation context
        llm_chat_fn: LLM function
        
    Returns:
        {
            "is_valid": bool,
            "extracted_value": str,
            "severity_flag": str or None,
            "confidence": float
        }
    """
    step_key = step.get("key", "")
    expected_answers = step.get("answers", {})
    question = step.get("question", "")
    
    system_prompt = """You are a medical information extractor for a triage system.

Your job is to:
1. Determine if the user answered the question
2. Extract the relevant medical information
3. Classify the response based on the expected answer types

CRITICAL RULES - BE VERY PERMISSIVE:
- DEFAULT TO ACCEPTING RESPONSES - if you have ANY doubt, mark as valid
- Accept single words: "right", "left", "upper", "lower" are ALL valid location answers
- Accept partial answers: "right" = "right side"
- Accept natural variations: "yesterday", "today", "few days ago", "last week"
- Accept typos: "yesterat" = "yesterday", "roght" = "right"
- For yes/no: accept "yeah", "yep", "nope", "nah", "uh huh", etc.
- For location: "left", "right", "upper", "lower", "middle", "center" are ALL valid
- For timing: ANY time expression is valid
- Only mark as invalid if the response is completely nonsensical or off-topic

IMPORTANT: Single-word location answers like "right", "left", "upper", "lower" are ALWAYS VALID.

Respond in JSON format ONLY:
{
  "is_valid": true/false,
  "extracted_value": "normalized answer",
  "severity_flag": "emergency" or "urgent" or "non_urgent" or null,
  "confidence": 0.95
}

Examples:
User: "right" for "Where is the pain?"
→ {"is_valid": true, "extracted_value": "right side", "severity_flag": null, "confidence": 0.95}

User: "yesterday" for "When did it start?"
→ {"is_valid": true, "extracted_value": "yesterday", "severity_flag": null, "confidence": 0.95}

User: "yesterat" for "When did it start?"
→ {"is_valid": true, "extracted_value": "yesterday", "severity_flag": null, "confidence": 0.9}

User: "yeah" for "Do you have fever?"
→ {"is_valid": true, "extracted_value": "yes", "severity_flag": "urgent", "confidence": 0.95}

User: "upper" for "Where is the pain?"
→ {"is_valid": true, "extracted_value": "upper abdomen", "severity_flag": null, "confidence": 0.95}

User: "not sure" for any question
→ {"is_valid": false, "extracted_value": null, "severity_flag": null, "confidence": 0.3}

User: "what?" or "huh?" for any question
→ {"is_valid": false, "extracted_value": null, "severity_flag": null, "confidence": 0.1}
"""

    # Build expected answer info
    expected_info = ""
    if expected_answers:
        answer_types = list(expected_answers.keys())
        expected_info = f"Expected answer types: {', '.join(answer_types)}"
        
        # Add severity mappings if available
        severity_map = []
        for ans_key, severity in expected_answers.items():
            if severity:
                severity_map.append(f"'{ans_key}' → {severity}")
        if severity_map:
            expected_info += f"\nSeverity mapping: {', '.join(severity_map)}"
    
    user_prompt = f"""Question asked: {question}
Question type: {step_key}

{expected_info}

User's response: "{user_response}"

Validate this response and extract the information. Return JSON only."""

    try:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        response = llm_chat_fn(
            messages=messages,
            max_tokens=100,
            temperature=0.3,
            stream=False
        )
        
        content = response.get("choices", [{}])[0].get("message", {}).get("content", "")
        
        # Extract JSON
        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group())
            print(f"[DynamicTriage] ✅ LLM Validation Result: {result}")
            
            # If LLM says invalid but it looks reasonable, use fallback to double-check
            if not result.get("is_valid"):
                print(f"[DynamicTriage] ⚠️ LLM marked as invalid, double-checking with fallback...")
                fallback_result = _fallback_validation(user_response, step_key, expected_answers)
                if fallback_result.get("is_valid"):
                    print(f"[DynamicTriage] ✅ Fallback says valid: {fallback_result}")
                    return fallback_result
            
            return result
        else:
            print(f"[DynamicTriage] ⚠️ No JSON in response: {content}")
            # Use fallback validation
            return _fallback_validation(user_response, step_key, expected_answers)
    
    except Exception as e:
        print(f"[DynamicTriage] ❌ Error validating answer: {e}")
        # Fallback to simple heuristic validation
        return _fallback_validation(user_response, step_key, expected_answers)


def _fallback_validation(user_response: str, step_key: str, expected_answers: Dict) -> Dict[str, Any]:
    """
    Simple heuristic validation when LLM fails
    Be very permissive - accept most reasonable responses
    """
    response_lower = user_response.lower().strip()
    
    print(f"[DynamicTriage] 🔍 Fallback validation: response='{response_lower}', step_key='{step_key}'")
    
    # Location questions - accept directional words
    if "location" in step_key.lower() or "where" in step_key.lower() or "side" in step_key.lower():
        location_words = ["left", "right", "upper", "lower", "middle", "center", "top", "bottom", "diffuse", "quadrant"]
        if any(word in response_lower for word in location_words):
            print(f"[DynamicTriage] ✅ Fallback: Location answer detected")
            return {
                "is_valid": True,
                "extracted_value": user_response,
                "severity_flag": None,
                "confidence": 0.8
            }
    
    # Timing questions - accept temporal words (CHECK FIRST before yes/no)
    # Important: Check timing BEFORE yes/no to avoid "yesteaday" → "yes" confusion
    if "onset" in step_key.lower() or "when" in step_key.lower() or "start" in step_key.lower():
        # Check for temporal patterns more carefully
        temporal_patterns = [
            "today", "yesterday", "day", "week", "hour", "ago", 
            "morning", "evening", "night", "month", "year",
            "recent", "while", "just", "started"
        ]
        
        # Also check for typos like "yesteaday", "yesturday", etc.
        if any(pattern in response_lower for pattern in temporal_patterns):
            print(f"[DynamicTriage] ✅ Fallback: Temporal answer detected")
            # Try to normalize common typos
            normalized = user_response
            if "yest" in response_lower and "day" in response_lower:
                normalized = "yesterday"
            elif "2day" in response_lower or "toda" in response_lower:
                normalized = "today"
            
            return {
                "is_valid": True,
                "extracted_value": normalized,
                "severity_flag": None,
                "confidence": 0.8
            }
        
        # Also check for fuzzy time expressions
        if len(response_lower) >= 4 and any(c.isdigit() for c in response_lower):
            # Contains numbers, likely a time expression
            print(f"[DynamicTriage] ✅ Fallback: Numeric time expression detected")
            return {
                "is_valid": True,
                "extracted_value": user_response,
                "severity_flag": None,
                "confidence": 0.7
            }
    
    # Yes/No questions - accept variations (ONLY if not a timing question)
    if expected_answers and set(expected_answers.keys()) & {"yes", "no"}:
        yes_words = ["yes", "yeah", "yep", "yup", "uh huh", "sure", "definitely"]
        no_words = ["no", "nope", "nah", "not really", "negative"]
        
        if any(word in response_lower for word in yes_words):
            severity = expected_answers.get("yes")
            return {
                "is_valid": True,
                "extracted_value": "yes",
                "severity_flag": severity,
                "confidence": 0.9
            }
        elif any(word in response_lower for word in no_words):
            severity = expected_answers.get("no")
            return {
                "is_valid": True,
                "extracted_value": "no",
                "severity_flag": severity,
                "confidence": 0.9
            }
    
    # If response is not just "not sure" or "don't know", accept it
    uncertain_phrases = ["not sure", "don't know", "dunno", "idk", "unclear", "unsure"]
    if not any(phrase in response_lower for phrase in uncertain_phrases):
        # Default to accepting if it has some content
        if len(response_lower.split()) >= 1 and len(response_lower) >= 2:
            return {
                "is_valid": True,
                "extracted_value": user_response,
                "severity_flag": None,
                "confidence": 0.7
            }
    
    # Last resort - mark as invalid
    return {
        "is_valid": False,
        "extracted_value": None,
        "severity_flag": None,
        "confidence": 0.0
    }


def map_severity_to_flag(severity_label: str, step_key: str) -> str:
    """
    Map severity label to flag key for state tracking
    
    Args:
        severity_label: "emergency", "urgent", "non_urgent"
        step_key: The question key (e.g., "fever", "jaundice")
        
    Returns:
        Flag key for state (e.g., "fever_emergency")
    """
    if not severity_label or severity_label == "non_urgent":
        return step_key
    
    return f"{step_key}_{severity_label}"


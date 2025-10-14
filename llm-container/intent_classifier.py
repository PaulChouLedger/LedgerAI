#!/usr/bin/env python3
"""
Intelligent Intent Classifier
Uses LLM to understand conversation context and detect medical intent
Replaces rigid pattern matching with context-aware classification
"""

import json
import re
from typing import Dict, List, Optional, Callable


def detect_medical_intent(
    prompt: str,
    conversation_history: Optional[List[Dict]] = None,
    llm_chat_fn: Optional[Callable] = None
) -> Dict:
    """
    Use LLM to intelligently detect user's medical intent
    
    Args:
        prompt: User's current message
        conversation_history: Recent conversation context
        llm_chat_fn: LLM chat function
        
    Returns:
        Dict with:
        - is_medical: bool - Is this a medical symptom/condition?
        - condition_category: str - Likely medical condition category
        - confidence: float - Confidence score 0.0-1.0
        - intent: str - Overall intent (medical_symptom, casual_response, clarification, etc.)
        - extracted_symptoms: List[str] - Key symptoms mentioned
    """
    if not llm_chat_fn:
        # Fallback to simple keyword detection if no LLM available
        return _fallback_intent_detection(prompt)
    
    # Build conversation context
    context_text = ""
    if conversation_history and len(conversation_history) > 0:
        recent = conversation_history[-3:]  # Last 3 exchanges
        context_lines = []
        for msg in recent:
            role = msg.get('role', 'user')
            content = msg.get('content', '')
            context_lines.append(f"{role.capitalize()}: {content}")
        context_text = "\n".join(context_lines)
    
    system_prompt = """You are a medical conversation analyzer for a triage system.
Analyze the user's message in context and determine their intent.

CRITICAL RULES:
- Be context-aware: "bad" after "How's your day?" is NOT medical
- "chest pain", "abdominal pain", "headache" ARE medical
- Single words like "bad", "good", "fine", "okay" are usually casual unless context says otherwise
- Location answers like "left side", "upper abdomen" during active triage are clarifications, NOT new conditions
- Typos are common: "abdomina pain" = "abdominal pain"

Respond ONLY with valid JSON in this exact format:
{
  "is_medical": true or false,
  "condition_category": "chest_pain" or "abdominal_pain" or null,
  "confidence": 0.95,
  "intent": "medical_symptom" or "casual_response" or "clarification" or "greeting",
  "extracted_symptoms": ["fever", "nausea"]
}

Do NOT add any text before or after the JSON."""

    user_prompt = f"""Conversation so far:
{context_text if context_text else "(No prior context)"}

Current user message: "{prompt}"

Analyze this message and provide your assessment in JSON format."""

    try:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        response = llm_chat_fn(
            messages=messages,
            max_tokens=150,
            temperature=0.3,  # Lower temperature for more consistent classification
            stream=False
        )
        
        content = response.get("choices", [{}])[0].get("message", {}).get("content", "")
        
        # Parse JSON response
        try:
            # Extract JSON from response (handle cases where LLM adds extra text)
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                intent_data = json.loads(json_match.group())
                print(f"[Intent] ✅ Classified: {intent_data}")
                return intent_data
            else:
                print(f"[Intent] ⚠️ No JSON found in response: {content}")
                return _fallback_intent_detection(prompt)
                
        except json.JSONDecodeError as e:
            print(f"[Intent] ❌ JSON parse error: {e}, content: {content}")
            return _fallback_intent_detection(prompt)
    
    except Exception as e:
        print(f"[Intent] ❌ Error in LLM classification: {e}")
        return _fallback_intent_detection(prompt)


def _fallback_intent_detection(prompt: str) -> Dict:
    """Fallback intent detection using simple keyword matching"""
    p = prompt.lower().strip()
    
    # Check for medical keywords
    medical_keywords = ["pain", "hurt", "ache", "bleeding", "fever", "dizzy", "nausea", "vomiting", 
                       "cough", "shortness of breath", "chest", "abdomen", "head", "stomach"]
    
    has_medical = any(keyword in p for keyword in medical_keywords)
    
    # Casual single-word responses
    casual_words = ["bad", "good", "okay", "fine", "well", "great", "terrible", "awful", "yes", "no"]
    is_casual = p in casual_words
    
    # Default classification
    if has_medical and len(p.split()) >= 2:
        return {
            "is_medical": True,
            "condition_category": None,  # Would need mapping
            "confidence": 0.7,
            "intent": "medical_symptom",
            "extracted_symptoms": [p]
        }
    elif is_casual:
        return {
            "is_medical": False,
            "condition_category": None,
            "confidence": 0.8,
            "intent": "casual_response",
            "extracted_symptoms": []
        }
    else:
        return {
            "is_medical": False,
            "condition_category": None,
            "confidence": 0.5,
            "intent": "unclear",
            "extracted_symptoms": []
        }


def map_condition_to_triage(condition_category: str) -> Optional[str]:
    """
    Map LLM-detected condition category to triage definition key
    
    Args:
        condition_category: Category from LLM (e.g., "chest_pain", "abdominal_pain")
        
    Returns:
        Triage definition key or None
    """
    # Direct mapping
    condition_mapping = {
        "chest_pain": "chest_pain",
        "abdominal_pain": "abdominal_pain",
        "headache": "headache",
        "weakness": "weakness",
        "shortness_of_breath": "shortness_of_breath",
        "palpitations": "palpitations",
        "dizziness": "dizziness",
        "syncope": "syncope",
        "fever": "uri",  # Fever alone often suggests URI
        "cough": "cough",
        "nausea": "abdominal_pain",  # Nausea often GI-related
        "vomiting": "abdominal_pain",
        # Add more mappings as needed
    }
    
    return condition_mapping.get(condition_category)


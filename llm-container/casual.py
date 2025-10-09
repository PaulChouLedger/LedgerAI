#!/usr/bin/env python3
"""
Aura CASUAL Mode - Simple Greetings and Small Talk

Handles:
- Greetings: "Hello", "Hi", "How are you?"
- Small talk: "How's the weather?", "What's up?"
- Brief acknowledgments

Does NOT handle:
- Knowledge queries (→ THINKER mode)
- Medical symptoms (→ TRIAGE/CLINICIAN mode)
"""

import requests
import random

def is_casual_trigger(prompt: str) -> bool:
    """
    Check if prompt should trigger CASUAL mode
    
    Args:
        prompt: Normalized prompt (lowercase)
        
    Returns:
        True if casual greeting/small talk
    """
    import re
    
    # Simple greeting patterns (word boundaries to avoid false matches)
    greeting_patterns = [
        r'^\s*\bhello\b\s*$',
        r'^\s*\bhi\b\s*$',
        r'^\s*\bhey\b\s*$',
        r'^\s*\bgood morning\b\s*$',
        r'^\s*\bgood afternoon\b\s*$',
        r'^\s*\bgood evening\b\s*$',
        r'^\s*\bhow are you\b\s*$',
        r'^\s*\bhows it going\b\s*$',
        r'^\s*\bwhats up\b\s*$',
        # With "aura" suffix
        r'^\s*\bhello aura\b\s*$',
        r'^\s*\bhi aura\b\s*$',
        r'^\s*\bhey aura\b\s*$',
    ]
    
    prompt_lower = prompt.lower().strip()
    
    return any(re.search(pattern, prompt_lower) for pattern in greeting_patterns)


def handle_casual(prompt: str, session_id: str = None) -> str:
    """
    Generate casual response for greetings and small talk
    
    Args:
        prompt: User's greeting/small talk
        session_id: Optional session identifier
        
    Returns:
        Friendly greeting response
    """
    print(f"[CASUAL] 💬 Handling greeting: '{prompt}'")
    
    responses = [
        "Hello! How can I help you today?",
        "Hi there! What can I do for you?",
        "Good to see you! How are you feeling?",
        "Hello! I'm here to help with any questions or concerns you might have.",
        "Hi! Feel free to ask me anything.",
        "Hey! What's on your mind?",
    ]
    
    return random.choice(responses)


def stream_casual_response(prompt: str, session_id: str = None):
    """
    Generate streaming casual response (for consistency with other modes)
    
    Args:
        prompt: User's greeting
        session_id: Optional session identifier
        
    Yields:
        Streamed response chunks
    """
    response = handle_casual(prompt, session_id)
    
    # Split into sentences for streaming
    sentences = response.split('. ')
    
    for sentence in sentences:
        if sentence.strip():
            # Add period back if it was removed
            if not sentence.endswith('.') and not sentence.endswith('?') and not sentence.endswith('!'):
                sentence += '.'
            yield f"<sentence_start>\n{sentence.strip()}\n<sentence_end>\n"


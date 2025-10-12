#!/usr/bin/env python3
"""
Aura CASUAL Mode - Unrestricted Friendly Conversation

Handles:
- Greetings and small talk
- General conversation
- Follow-up questions
- Casual inquiries

This mode uses full LLM for natural, unrestricted conversation.
Will auto-switch to THINKER mode if user asks knowledge/information questions.
"""

import requests
import random
import re

def filter_thinking_tags(text: str) -> str:
    """Remove thinking/reasoning tags and verbose internal monologue"""
    if not text:
        return text
    
    # Remove <think> or </think> tags
    text = re.sub(r'</?think>', '', text, flags=re.IGNORECASE)
    
    # Remove thinking patterns (lines that are clearly internal reasoning)
    thinking_patterns = [
        r"I need to respond.*?\.",
        r"I should (?:keep|make|try|be).*?\.",
        r"Let me (?:think|consider|try).*?\.",
        r"Maybe (?:I|something).*?\.",
        r"Since they .*?\.",
        r"That sounds (?:good|right|perfect)\.?",
        r"Alright,? that'?s? .*?\.",
        r"It'?s? (?:friendly|helpful|good|solid).*?\."
    ]
    
    for pattern in thinking_patterns:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE)
    
    # Remove emoji-only responses
    text = re.sub(r'^\s*[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF]+\s*$', '', text)
    
    # Clean up multiple spaces/newlines
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    
    return text

def is_casual_trigger(prompt: str) -> bool:
    """
    Check if prompt should START with CASUAL mode (simple greetings only)
    
    Args:
        prompt: Normalized prompt (lowercase)
        
    Returns:
        True if simple greeting to initiate conversation
    """
    # Only match VERY simple greetings to start casual mode
    # Once in casual mode, it's unrestricted until user asks knowledge query
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


def handle_casual(prompt: str, llm_chat_fn, session_id: str = None):
    """
    Generate casual response using full LLM (unrestricted conversation)
    
    Args:
        prompt: User's message
        llm_chat_fn: LLM chat function for generating responses
        session_id: Optional session identifier
        
    Yields:
        Streamed response tokens
    """
    print(f"[CASUAL] 💬 Handling conversation: '{prompt[:50]}...'")
    
    # System prompt for casual conversation
    system_prompt = (
        "You are Aura, a friendly and helpful AI assistant. "
        "Respond directly and naturally in 1-2 short sentences. "
        "Do NOT explain your thinking process or reasoning. "
        "Do NOT use phrases like 'I need to', 'Let me think', 'Maybe I should'. "
        "Just respond conversationally and warmly. "
        "Be concise, empathetic, and approachable. "
        "If asked about medical symptoms, suggest they describe their symptoms. "
        "Example: User: 'Hi' → You: 'Hello! How can I help you today?'"
    )
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt}
    ]
    
    try:
        response = llm_chat_fn(
            messages=messages,
            max_tokens=50,  # Short, concise responses (1-2 sentences)
            temperature=0.6,  # Slightly lower for more focused responses
            stream=True,
            stop=["\n\n", "Let me", "I need", "I should", "Maybe"]  # Stop on verbose patterns
        )
        
        buffer = []
        for chunk in response:
            if 'choices' in chunk:
                delta = chunk['choices'][0].get('delta', {})
                content = delta.get('content', '')
                
                if content:
                    buffer.append(content)
                    
                    # Detect sentence boundaries for streaming
                    full_text = ''.join(buffer)
                    sentences = re.split(r'([.!?]\s+)', full_text)
                    
                    # Stream complete sentences
                    while len(sentences) > 2:  # At least one complete sentence
                        sentence = sentences.pop(0) + (sentences.pop(0) if sentences else '')
                        
                        # Filter out thinking/reasoning before streaming
                        cleaned = filter_thinking_tags(sentence)
                        if cleaned.strip():
                            yield f"<sentence_start>\n{cleaned.strip()}\n<sentence_end>\n"
                        
                        buffer = [s for s in sentences]
        
        # Stream any remaining text (filtered)
        remaining = ''.join(buffer).strip()
        if remaining:
            cleaned = filter_thinking_tags(remaining)
            if cleaned:
                yield f"<sentence_start>\n{cleaned}\n<sentence_end>\n"
            
    except Exception as e:
        print(f"[CASUAL] ❌ Error in LLM generation: {e}")
        # Fallback response
        yield f"<sentence_start>\nI'm here to help! What would you like to know?\n<sentence_end>\n"


def stream_casual_response(prompt: str, llm_chat_fn, session_id: str = None):
    """
    Stream casual response (wrapper for handle_casual)
    
    Args:
        prompt: User's message
        llm_chat_fn: LLM chat function
        session_id: Optional session identifier
        
    Yields:
        Streamed response chunks
    """
    for chunk in handle_casual(prompt, llm_chat_fn, session_id):
        yield chunk


"""
Simple chat utilities for Aura voice assistant.
RAG processing is now handled by the rag-container service.
"""
import re
from collections import deque
from typing import List


# === Filler phrases for natural conversation ===
FILLER_PHRASES = [
    "I'm happy to help!",
    "Let me think about that...",
    "Sure, one moment...",
    "Alright, let's take a look.",
    "Hold on while I check.",
    "Great question! Let me see..."
]
recent_fillers = deque(maxlen=3)

SHORT_PROMPTS = [
    "hello", "hi", "hey", "good morning", "good afternoon", "good evening",
    "thanks", "thank you", "ok", "cool", "interesting"
]


def is_short_prompt(text: str) -> bool:
    """Check if the text is a short greeting or acknowledgment."""
    return any(text.lower().strip() == sp for sp in SHORT_PROMPTS)


def get_filler() -> str:
    """Get a natural filler phrase that hasn't been used recently."""
    for phrase in FILLER_PHRASES:
        if phrase not in recent_fillers:
            recent_fillers.append(phrase)
            return phrase
    return "One sec..."


def split_into_sentences(text: str) -> List[str]:
    """Split text into sentences for TTS processing."""
    # Simple sentence splitting on punctuation
    sentences = re.split(r'[.!?]+', text)
    return [s.strip() for s in sentences if s.strip()]


def should_use_filler(text: str) -> bool:
    """Determine if we should use a filler phrase for this input."""
    return not is_short_prompt(text) and len(text.strip()) > 10

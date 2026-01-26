"""
Shared text cleaning utility for consistent formatting across chat and TTS.
Used by both llm-container (chat formatting) and aura-control (TTS formatting).
"""

import re


def clean_text_formatting(text: str) -> str:
    """
    Clean and normalize text formatting for better readability.
    Removes markdown formatting, fixes spacing, removes hashtags and asterisks.
    
    This function is used by both:
    - Chat interface (llm-container/container_rest.py)
    - TTS playback (aura-control/core/speaker.py)
    
    Args:
        text: Input text to clean
        
    Returns:
        Cleaned text with markdown removed and spacing fixed
    """
    if not text:
        return text
    
    # Remove markdown headers (hashtags at start of line)
    text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)
    # Remove standalone hashtags
    text = re.sub(r'#{1,6}(?=\s|$)', '', text)
    
    # Remove markdown bold/italic (asterisks)
    text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)  # **text** -> text
    text = re.sub(r'\*([^*\n]+)\*', r'\1', text)  # *text* -> text
    # Remove standalone asterisks (markdown formatting artifacts)
    text = re.sub(r'\*\*+', '', text)  # Remove multiple asterisks
    text = re.sub(r'(?<!\w)\*(?!\w)', '', text)  # Remove single asterisks not part of words
    
    # Remove LaTeX/math formatting (before other processing)
    # Remove LaTeX commands like \text{}, \frac{}, etc.
    text = re.sub(r'\\text\{([^}]+)\}', r'\1', text)  # \text{or} -> or
    text = re.sub(r'\\frac\{([^}]+)\}\{([^}]+)\}', r'\1/\2', text)  # \frac{3}{4} -> 3/4
    # Remove LaTeX math delimiters \( and \)
    text = re.sub(r'\\\(', '', text)  # Remove \(
    text = re.sub(r'\\\)', '', text)  # Remove \)
    # Remove other common LaTeX commands
    text = re.sub(r'\\[a-zA-Z]+\{([^}]*)\}', r'\1', text)  # \command{text} -> text
    text = re.sub(r'\\[a-zA-Z]+', '', text)  # Remove remaining LaTeX commands
    
    # Fix missing spaces after punctuation
    text = re.sub(r'([a-zA-Z0-9])([.!?])([a-zA-Z-])', r'\1\2 \3', text)  # word.word -> word. word
    text = re.sub(r'([,.!?:;])([a-zA-Z])', r'\1 \2', text)  # word,word -> word, word
    text = re.sub(r'([a-zA-Z0-9])(\()', r'\1 \2', text)  # word(word -> word (word
    text = re.sub(r'(\))([a-zA-Z0-9])', r'\1 \2', text)  # word)word -> word) word
    
    # Normalize multiple spaces
    text = re.sub(r' {2,}', ' ', text)
    
    return text.strip()


"""
Shared Fuzzy Matching Utilities
Common fuzzy matching functions for RAG systems to handle transcription errors
"""

import re
from typing import List
from difflib import SequenceMatcher

# Common stop words for query processing
STOP_WORDS = {
    'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
    'i', 'you', 'he', 'she', 'it', 'we', 'they', 'is', 'are', 'was', 'were',
    'do', 'does', 'did', 'how', 'what', 'when', 'where', 'why',
    'can', 'could', 'should', 'would', 'may', 'might', 'must'
}


def fuzzy_match_term(term: str, text: str, threshold: float = 0.75) -> bool:
    """
    Check if a term fuzzy matches any word in the text.
    Handles transcription errors by using fuzzy string matching.
    
    Args:
        term: Term to search for
        text: Text to search in
        threshold: Minimum similarity ratio (0.0-1.0) for a match
    
    Returns:
        True if term fuzzy matches any word in text
    """
    # Extract all words from text (3+ characters to avoid matching common words)
    text_words = re.findall(r'\b\w{3,}\b', text.lower())
    term_lower = term.lower()
    
    # First try exact match (fastest)
    if term_lower in text_words:
        return True
    
    # Then try fuzzy match for transcription errors
    for word in text_words:
        # Only fuzzy match words of similar length (avoid false positives)
        if abs(len(word) - len(term_lower)) <= 2:
            similarity = SequenceMatcher(None, term_lower, word).ratio()
            if similarity >= threshold:
                return True
    
    return False


def extract_key_terms(query: str, min_word_length: int = 2) -> List[str]:
    """
    Extract key terms from a query, removing stop words.
    
    Args:
        query: Search query
        min_word_length: Minimum word length to include
    
    Returns:
        List of key terms (non-stop words)
    """
    query_lower = query.lower()
    words = re.findall(r'\b\w{' + str(min_word_length) + r',}\b', query_lower)
    key_terms = [w for w in words if w not in STOP_WORDS]
    return key_terms


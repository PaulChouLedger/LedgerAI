#!/usr/bin/env python3
"""
Centralized Validation Module
Provides shared validation and matching functions for all conversation modes
Avoids circular imports by being independent of container_rest and triage
"""

from typing import Dict, List, Tuple, Optional, Any
import string

# Constants
MIN_MATCH = 0.6

# === Utility Functions ===

def normalize_text(text: str) -> str:
    """Normalize text for comparison"""
    if not text:
        return ""
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = ' '.join(text.split())
    return text


def tokenize(text: str) -> List[str]:
    """Tokenize normalized text"""
    return normalize_text(text).split()


def normalize_yes_no_response(text: str) -> str:
    """Normalize natural yes/no responses"""
    text_lower = text.lower().strip()
    
    # Negative responses first
    if any(phrase in text_lower for phrase in [
        "no", "nope", "nah", "not", "don't", "do not", "haven't", "have not",
        "i don't", "i do not", "i haven't", "i have not",
        "i don't have", "i do not have", "i don't feel", "i do not feel",
        "i don't experience", "i do not experience", "i am not", "i'm not"
    ]):
        return "no"
    
    if text_lower in ["i dont", "i don't", "i do not", "i havent", "i haven't", "i have not"]:
        return "no"
    
    # Positive responses
    if any(phrase in text_lower for phrase in [
        "yes", "yea", "yeah", "yep", "yup", "sure", "ok", "okay",
        "i do", "i have", "i am", "i feel", "i experience",
        "i do have", "i do feel", "i do experience",
        "i have been", "i am having", "i am experiencing"
    ]):
        return "yes"
    
    return text


def get_generic_onset_answers() -> Dict[str, str]:
    """Get standard onset answers"""
    return {
        "within the last hour": "emergency",
        "within the last few hours": "emergency",
        "today": "urgent",
        "yesterday": "urgent",
        "a few days ago": "urgent",
        "a week ago": "non_urgent",
        "unknown": "urgent"
    }


def match_flexible_time(ans_expanded: str, valid_map: Dict[str, str]) -> Optional[Tuple[str, float]]:
    """Match flexible time patterns like '3 hours ago'"""
    import re
    
    time_pattern = r'(\d+)\s*(minute|hour|day|week|month)s?\s*ago'
    match = re.search(time_pattern, ans_expanded)
    
    if match:
        amount = int(match.group(1))
        unit = match.group(2)
        
        if unit == "minute":
            return "within the last hour", 1.0
        elif unit == "hour":
            if amount <= 2:
                return "within the last few hours", 1.0
            else:
                return "today", 1.0
        elif unit == "day":
            if amount == 1:
                return "yesterday", 1.0
            elif amount <= 3:
                return "a few days ago", 1.0
            elif amount <= 7:
                return "a week ago", 1.0
        elif unit == "week":
            return "a week ago", 1.0
        elif unit == "month":
            return "a week ago", 1.0
    
    # Handle common time phrases
    if "today" in ans_expanded or "this morning" in ans_expanded or "this afternoon" in ans_expanded:
        return "today", 1.0
    if "yesterday" in ans_expanded:
        return "yesterday", 1.0
    if "hour" in ans_expanded:
        if "few hours" in ans_expanded or "2 hours" in ans_expanded or "3 hours" in ans_expanded:
            return "within the last few hours", 1.0
        return "within the last hour", 1.0
    if "days" in ans_expanded or "day" in ans_expanded:
        if "few days" in ans_expanded or "2 days" in ans_expanded or "3 days" in ans_expanded:
            return "a few days ago", 1.0
        return "a few days ago", 1.0
    if "week" in ans_expanded:
        return "a week ago", 1.0
    if "last week" in ans_expanded:
        return "last week", 1.0
    
    return None


# === Centralized Answer Validation ===

def check_typo_similarity(ans: str, opt: str) -> float:
    """
    Check for typo similarity using character-level matching
    Handles all typos algorithmically without manual dictionary
    """
    # Exact match
    if ans == opt:
        return 1.0
    
    # Same length - check character differences
    if len(ans) == len(opt):
        diff_count = sum(1 for a, o in zip(ans, opt) if a != o)
        if diff_count == 1:
            # Single character typo (e.g., "roght" → "right")
            return 0.9
        elif diff_count == 2:
            # Two character typos (e.g., "abdmoinal" → "abdominal")
            return 0.7
    
    # Prefix/suffix matching (missing/extra characters)
    if len(ans) > 0 and len(opt) > 0:
        # Missing character(s) at end (e.g., "abdomina" → "abdominal")
        if opt.startswith(ans) and len(opt) - len(ans) <= 2:
            return 0.85
        
        # Extra character(s) at end
        if ans.startswith(opt) and len(ans) - len(opt) <= 2:
            return 0.85
        
        # Character position matching for similar-length words
        if abs(len(ans) - len(opt)) <= 2:
            min_len = min(len(ans), len(opt))
            max_len = max(len(ans), len(opt))
            
            # Count matching characters at same positions
            matching = sum(1 for i in range(min_len) if ans[i] == opt[i])
            similarity = matching / float(max_len)
            
            # High similarity threshold
            if similarity >= 0.85:  # 85%+ character match
                return 0.8
            elif similarity >= 0.75:  # 75%+ character match
                return 0.6

    return 0.0


def match_answer_option(ans_norm: str, valid_map: Dict[str, str], use_synonyms: bool = True, key: str = None, 
                       synonym_expansion_fn=None, normalize_yes_no_fn=None) -> Tuple[Optional[str], float]:
    """
    Match answer to options with fuzzy matching and typo correction
    
    Args:
        ans_norm: Normalized answer text
        valid_map: Dictionary of valid answers
        use_synonyms: Whether to use synonym expansion
        key: Question key (for context)
        synonym_expansion_fn: Optional synonym expansion function
        normalize_yes_no_fn: Optional yes/no normalization function
        
    Returns:
        Tuple of (matched_option, confidence_score)
    """
    ans_expanded = ans_norm
    if use_synonyms and synonym_expansion_fn:
        ans_expanded = synonym_expansion_fn(ans_norm)

    # Normalize yes/no first
    if normalize_yes_no_fn:
        normalized_response = normalize_yes_no_fn(ans_expanded)
    else:
        normalized_response = normalize_yes_no_response(ans_expanded)
        
    if normalized_response in ["yes", "no"]:
        if "yes" in valid_map and "no" in valid_map:
            return normalized_response, 1.0

    # Generic onset answers
    if key == "onset" and (not valid_map or len(valid_map) == 0):
        valid_map = get_generic_onset_answers()

    # Flexible time matching
    time_match = match_flexible_time(ans_expanded, valid_map)
    if time_match:
        return time_match

    # Improved fuzzy matching with typo correction
    ans_tokens = set(tokenize(ans_expanded))
    best, score = None, 0.0

    for opt in valid_map:
        opt_tokens = set(tokenize(opt))

        # Exact match gets highest score
        if ans_expanded == opt:
            return opt, 1.0

        # Token overlap matching
        overlap = len(ans_tokens & opt_tokens)

        if overlap > 0:
            base_score = overlap / float(len(opt_tokens)) if opt_tokens else 0
            length_bonus = len(opt_tokens) * 0.1

            if overlap == len(ans_tokens) and overlap == len(opt_tokens):
                exact_bonus = 0.5
            elif overlap == len(opt_tokens):
                exact_bonus = 0.3
            else:
                exact_bonus = 0

            final_score = base_score + length_bonus + exact_bonus
        else:
            # Check for common typos and close matches
            typo_score = check_typo_similarity(ans_expanded, opt)
            final_score = typo_score

        if final_score > score:
            best, score = opt, final_score

    return best, score


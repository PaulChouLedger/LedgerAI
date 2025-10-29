#!/usr/bin/env python3
"""
Generic Fuzzy Matcher
Handles common typos, misspellings, and transcription errors for improved text matching
"""

import re
from typing import Dict, List, Tuple, Optional
from difflib import SequenceMatcher
from fuzzywuzzy import fuzz, process

class FuzzyMatcher:
    """
    Generic fuzzy matching for handling typos, misspellings, and transcription errors
    """
    
    def __init__(self):
        # Common typo corrections (general-purpose)
        self.common_typo_corrections = {
            # Common word typos
            'teh': 'the',
            'adn': 'and',
            'taht': 'that',
            'recieve': 'receive',
            'seperate': 'separate',
            'occured': 'occurred',
            'definately': 'definitely',
            'accomodate': 'accommodate',
            'existance': 'existence',
            'maintainance': 'maintenance',
            'seige': 'siege',
            'acheive': 'achieve',
            'neccessary': 'necessary',
            'occassion': 'occasion',
            'begining': 'beginning',
            'mispell': 'misspell',
            'transcripton': 'transcription',
            'transcripton': 'transcription',
            'transcriptin': 'transcription',
            'transcriptio': 'transcription',
        }
        
        # Phonetic/sound-alike mappings for common words
        self.phonetic_mappings = {
            'their': ['there', 'they\'re'],
            'there': ['their', 'they\'re'],
            'to': ['too', 'two'],
            'too': ['to', 'two'],
            'two': ['to', 'too'],
            'than': ['then'],
            'then': ['than'],
            'accept': ['except'],
            'except': ['accept'],
            'affect': ['effect'],
            'effect': ['affect'],
            'complement': ['compliment'],
            'compliment': ['complement'],
        }
    
    def fuzzy_correct(self, text: str, similarity_threshold: float = 0.6, 
                     custom_dictionary: Optional[List[str]] = None) -> str:
        """
        Apply fuzzy correction to text, handling typos and misspellings
        
        Args:
            text: Input text potentially containing typos
            similarity_threshold: Minimum similarity score for fuzzy matching (0.0-1.0)
            custom_dictionary: Optional list of correct words to match against
            
        Returns:
            Corrected text with typos fixed
        """
        corrected_text = text.lower()
        
        # 1. EXACT TYPO CORRECTIONS (fastest)
        for typo, correction in self.common_typo_corrections.items():
            pattern = r'\b' + re.escape(typo) + r'\b'
            corrected_text = re.sub(pattern, correction, corrected_text, flags=re.IGNORECASE)
        
        # 2. PHONETIC MAPPINGS (moderate speed)
        words = corrected_text.split()
        corrected_words = []
        
        for word in words:
            corrected_word = self._apply_phonetic_correction(word)
            corrected_words.append(corrected_word)
        
        corrected_text = ' '.join(corrected_words)
        
        # 3. FUZZY SIMILARITY MATCHING (if custom dictionary provided)
        if custom_dictionary:
            words = corrected_text.split()
            final_words = []
            
            for word in words:
                if len(word) >= 4:  # Only fuzzy match longer words
                    corrected_word = self._apply_fuzzy_correction(
                        word, custom_dictionary, similarity_threshold
                    )
                    final_words.append(corrected_word)
                else:
                    final_words.append(word)
            
            corrected_text = ' '.join(final_words)
        
        return corrected_text
    
    def _apply_phonetic_correction(self, word: str) -> str:
        """Apply phonetic/sound-alike corrections"""
        word_clean = word.lower().strip()
        
        for correct_term, variants in self.phonetic_mappings.items():
            if word_clean in variants:
                return correct_term
        
        return word
    
    def _apply_fuzzy_correction(self, word: str, dictionary: List[str], 
                                threshold: float) -> str:
        """Apply fuzzy string matching against a dictionary"""
        word_clean = word.lower().strip()
        
        best_match = word_clean
        best_score = 0.0
        
        for dict_word in dictionary:
            dict_word_lower = dict_word.lower()
            # Calculate similarity using SequenceMatcher
            similarity = SequenceMatcher(None, word_clean, dict_word_lower).ratio()
            
            if similarity > threshold and similarity > best_score:
                best_score = similarity
                best_match = dict_word_lower
        
        return best_match if best_score > threshold else word_clean
    
    def fuzzy_search(self, query: str, candidates: List[str], 
                    threshold: int = 70, limit: int = 5) -> List[Tuple[str, int]]:
        """
        Fuzzy search for best matches in a list of candidates
        
        Args:
            query: Search query (potentially misspelled)
            candidates: List of candidate strings to search
            threshold: Minimum similarity score (0-100)
            limit: Maximum number of results to return
            
        Returns:
            List of tuples (matched_string, similarity_score)
        """
        # Use fuzzywuzzy for fast fuzzy matching
        matches = process.extract(
            query, 
            candidates, 
            scorer=fuzz.token_sort_ratio,
            limit=limit
        )
        
        # Filter by threshold
        filtered_matches = [
            (match[0], match[1]) 
            for match in matches 
            if match[1] >= threshold
        ]
        
        return filtered_matches
    
    def fuzzy_match(self, text1: str, text2: str, threshold: float = 0.8) -> bool:
        """
        Check if two strings are fuzzy matches
        
        Args:
            text1: First string
            text2: Second string
            threshold: Minimum similarity ratio (0.0-1.0)
            
        Returns:
            True if strings match above threshold
        """
        similarity = SequenceMatcher(None, text1.lower(), text2.lower()).ratio()
        return similarity >= threshold
    
    def normalize_text(self, text: str) -> str:
        """
        Normalize text for better matching (remove punctuation, lowercase, etc.)
        
        Args:
            text: Input text
            
        Returns:
            Normalized text
        """
        # Remove punctuation
        text = re.sub(r'[^\w\s]', '', text)
        # Normalize whitespace
        text = ' '.join(text.split())
        # Lowercase
        return text.lower()
    
    def add_custom_corrections(self, corrections: Dict[str, str]):
        """
        Add custom typo corrections
        
        Args:
            corrections: Dictionary mapping typos to corrections
        """
        self.common_typo_corrections.update(corrections)
    
    def add_custom_phonetic_mappings(self, mappings: Dict[str, List[str]]):
        """
        Add custom phonetic mappings
        
        Args:
            mappings: Dictionary mapping correct terms to variants
        """
        self.phonetic_mappings.update(mappings)


# Global instance for easy import
_fuzzy_matcher_instance = None

def get_fuzzy_matcher() -> FuzzyMatcher:
    """Get or create global fuzzy matcher instance"""
    global _fuzzy_matcher_instance
    if _fuzzy_matcher_instance is None:
        _fuzzy_matcher_instance = FuzzyMatcher()
    return _fuzzy_matcher_instance

# Test the fuzzy matcher
if __name__ == "__main__":
    fuzzy_matcher = FuzzyMatcher()
    
    # Test cases
    test_cases = [
        "I recieve teh document",
        "This is definately seperate",
        "The transcripton occured",
        "Neccessary maintanance",
        "Begining of acheive",
    ]
    
    print("🧪 Testing Generic Fuzzy Matcher")
    print("=" * 50)
    
    for original in test_cases:
        corrected = fuzzy_matcher.fuzzy_correct(original)
        print(f"Original:  '{original}'")
        print(f"Corrected: '{corrected}'")
        print()
    
    # Test fuzzy search
    print("\n🔍 Testing Fuzzy Search")
    print("=" * 50)
    candidates = [
        "artificial intelligence",
        "machine learning",
        "deep learning",
        "neural networks",
        "natural language processing"
    ]
    
    queries = ["artifical inteligence", "machin lernig", "deep lerning"]
    for query in queries:
        matches = fuzzy_matcher.fuzzy_search(query, candidates, threshold=60)
        print(f"Query: '{query}'")
        print(f"Matches: {matches}")
        print()



#!/usr/bin/env python3
"""
Fuzzy Medical Term Matcher
Handles common medical typos and variants for improved complaint categorization
"""

import re
from typing import Dict, List, Tuple
from difflib import SequenceMatcher

class FuzzyMedicalMatcher:
    """
    Fuzzy matching for medical terms to handle common typos and variants
    Uses global FAISS index terms when available, otherwise falls back to hardcoded list
    """
    
    def __init__(self, indexed_medical_terms: set = None):
        """
        Initialize fuzzy matcher
        
        Args:
            indexed_medical_terms: Optional set of all patient_friendly terms from FAISS index
                                   If provided, fuzzy matching will use these instead of hardcoded list
        """
        self.indexed_medical_terms = indexed_medical_terms
        # Common medical term typos and variants
        self.medical_typo_corrections = {
            # Abdominal variations
            'abodminal': 'abdominal',
            'abdomnial': 'abdominal', 
            'abdomninal': 'abdominal',
            'abdomenal': 'abdominal',
            'abodmen': 'abdomen',  # Common typo: "abodmen" -> "abdomen"
            'abdominal': 'abdominal',  # Correct spelling included
            'abdomen': 'abdomen',  # Correct spelling included
            
            # Chest variations
            'cheste': 'chest',
            'chest': 'chest',
            
            # Heart variations
            'hart': 'heart',
            'heart': 'heart',
            
            # Common medical terms
            'stomac': 'stomach',
            'stomache': 'stomach',
            'stomach': 'stomach',
            
            'bely': 'belly',
            'belly': 'belly',
            
            'headach': 'headache',
            'headace': 'headache',
            'headache': 'headache',
            
            'nausea': 'nausea',
            'nauseous': 'nauseous',
            'naseous': 'nauseous',
            'naseaus': 'nauseous',
            
            'vomiting': 'vomiting',
            'vomiting': 'vomiting',
            'vom': 'vomiting',
            
            'dizzy': 'dizzy',
            'dizy': 'dizzy',
            'dizziness': 'dizziness',
            
            'breathing': 'breathing',
            'brething': 'breathing',
            'breathng': 'breathing'
        }
        
        # Phonetic/sound-alike mappings
        self.phonetic_mappings = {
            'abdomen': ['abdoman', 'abdomin', 'abdomun'],
            'abdominal': ['abdomnal', 'abdominel', 'abdomnul'],
            'stomach': ['stomac', 'stomache', 'stomak'],
            'chest': ['cheste', 'chesst'],
            'heart': ['hart', 'hearth'],
            'nausea': ['nausious', 'naseous', 'naseaus'],
            'vomiting': ['vommiting', 'vomiting', 'vomting'],
            'headache': ['headach', 'headace', 'hedache'],
            'breathing': ['brething', 'breathng', 'breathin']
        }
    
    def fuzzy_correct_medical_terms(self, text: str, similarity_threshold: float = 0.6) -> str:
        """
        Apply fuzzy correction to medical terms in text
        
        Args:
            text: Input text potentially containing medical typos
            similarity_threshold: Minimum similarity score for fuzzy matching (0.0-1.0)
            
        Returns:
            Corrected text with medical typos fixed
        """
        corrected_text = text.lower()
        
        # 1. EXACT TYPO CORRECTIONS (fastest)
        for typo, correction in self.medical_typo_corrections.items():
            # Use word boundary matching to avoid partial word corrections
            pattern = r'\b' + re.escape(typo) + r'\b'
            corrected_text = re.sub(pattern, correction, corrected_text, flags=re.IGNORECASE)
        
        # 2. PHONETIC MAPPINGS (moderate speed)
        words = corrected_text.split()
        corrected_words = []
        
        for word in words:
            corrected_word = self._apply_phonetic_correction(word)
            corrected_words.append(corrected_word)
        
        corrected_text = ' '.join(corrected_words)
        
        # 3. FUZZY SIMILARITY MATCHING (slowest, for remaining unmatched terms)
        # Only apply to words that are likely medical term typos (longer words, not common English)
        words = corrected_text.split()
        final_words = []
        
        for word in words:
            word_clean = word.lower().strip()
            # Only fuzzy match words that:
            # 1. Are at least 5 characters (to avoid matching common words like "part", "side")
            # 2. Are not already in our typo corrections or phonetic mappings
            # 3. Are likely medical terms (not common English words)
            if len(word_clean) >= 5 and word_clean not in self.medical_typo_corrections:
                # Check if it's in phonetic mappings
                in_phonetic = any(word_clean in variants for variants in self.phonetic_mappings.values())
                if not in_phonetic:
                    corrected_word = self._apply_fuzzy_correction(word, similarity_threshold)
                    final_words.append(corrected_word)
                else:
                    final_words.append(word)
            else:
                final_words.append(word)
        
        return ' '.join(final_words)
    
    def _apply_phonetic_correction(self, word: str) -> str:
        """Apply phonetic/sound-alike corrections"""
        word_clean = word.lower().strip()
        
        for correct_term, variants in self.phonetic_mappings.items():
            if word_clean in variants:
                return correct_term
        
        return word
    
    def _apply_fuzzy_correction(self, word: str, threshold: float) -> str:
        """Apply fuzzy string matching for medical terms"""
        word_clean = word.lower().strip()
        
        # Common English words that should NOT be fuzzy matched (location prepositions, body parts, etc.)
        common_words_whitelist = {
            'part', 'lower', 'upper', 'left', 'right', 'middle', 'center', 'around', 'near', 'your', 'my', 'the', 'a', 'an',
            'of', 'in', 'on', 'at', 'to', 'for', 'with', 'from', 'by', 'about', 'into', 'onto', 'over', 'under',
            'side', 'sides', 'area', 'areas', 'place', 'places', 'spot', 'spots', 'region', 'regions', 'zone', 'zones',
            'top', 'bottom', 'front', 'back', 'rear', 'behind', 'above', 'below', 'between', 'among', 'through',
            'ribs', 'rib', 'groin', 'belly', 'button', 'abdomen', 'chest', 'shoulder', 'shoulders', 'blade', 'blades'
        }
        
        # Skip fuzzy matching for common words
        if word_clean in common_words_whitelist:
            return word_clean
        
        # Use indexed terms from FAISS if available, otherwise fall back to hardcoded list
        if self.indexed_medical_terms:
            medical_terms = list(self.indexed_medical_terms)
        else:
            # Fallback to hardcoded list if no index provided
            medical_terms = [
                'abdominal', 'stomach', 'heart', 'headache', 'nausea', 
                'vomiting', 'breathing', 'dizzy', 'pain', 'ache'
            ]
        
        best_match = word_clean
        best_score = 0.0
        
        for medical_term in medical_terms:
            # Normalize medical term for comparison
            medical_term_clean = medical_term.lower().strip()
            
            # Calculate similarity using SequenceMatcher
            similarity = SequenceMatcher(None, word_clean, medical_term_clean).ratio()
            
            # Require higher similarity for shorter words to avoid false matches
            min_threshold = threshold
            if len(word_clean) <= 5 and len(medical_term_clean) <= 5:
                # Short words need very high similarity (0.8+) to avoid false matches
                min_threshold = max(threshold, 0.8)
            
            if similarity > min_threshold and similarity > best_score:
                best_score = similarity
                best_match = medical_term_clean  # Return normalized version
        
        # Only return correction if similarity is significantly high
        if best_score < 0.8:
            return word_clean
        
        return best_match
    
    def get_corrected_organ_keywords(self, text: str) -> List[str]:
        """
        Get corrected organ system keywords from potentially misspelled text
        
        Returns:
            List of corrected organ system keywords found
        """
        corrected_text = self.fuzzy_correct_medical_terms(text)
        
        organ_keywords = {
            'GI': ['abdominal', 'stomach', 'belly', 'gut', 'bowel', 'intestine', 'gastrointestinal'],
            'CARDIO': ['chest', 'heart', 'cardiac', 'coronary', 'myocardial'],
            'NEURO': ['head', 'headache', 'brain', 'neurological', 'cerebral', 'migraine'],
            'MSK': ['back', 'joint', 'muscle', 'bone', 'spine', 'musculoskeletal'],
            'RENAL': ['kidney', 'urinary', 'bladder', 'flank', 'renal'],
            'DERM': ['skin', 'rash', 'lesion', 'dermatological'],
            'GYN': ['pelvic', 'menstrual', 'gynecological']
        }
        
        found_keywords = []
        for organ_system, keywords in organ_keywords.items():
            for keyword in keywords:
                if keyword in corrected_text:
                    found_keywords.append((organ_system, keyword))
        
        return found_keywords

# Test the fuzzy matcher
if __name__ == "__main__":
    fuzzy_matcher = FuzzyMedicalMatcher()
    
    # Test cases
    test_cases = [
        "I have abodminal pain",  # Should correct to "abdominal"
        "cheste pain and hart problems",  # Should correct to "chest" and "heart"
        "stomache ache with naseaus",  # Should correct to "stomach" and "nauseous"
        "severe headach and dizy spells",  # Should correct to "headache" and "dizzy"
        "brething difficulties"  # Should correct to "breathing"
    ]
    
    print("🧪 Testing Fuzzy Medical Term Matcher")
    print("=" * 50)
    
    for original in test_cases:
        corrected = fuzzy_matcher.fuzzy_correct_medical_terms(original)
        keywords = fuzzy_matcher.get_corrected_organ_keywords(original)
        
        print(f"Original:  '{original}'")
        print(f"Corrected: '{corrected}'")
        print(f"Keywords:  {keywords}")
        print()

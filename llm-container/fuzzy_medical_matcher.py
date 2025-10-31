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
    """
    
    def __init__(self):
        # Common medical term typos and variants
        self.medical_typo_corrections = {
            # Abdominal variations
            'abodminal': 'abdominal',
            'abdomnial': 'abdominal', 
            'abdomninal': 'abdominal',
            'abdomenal': 'abdominal',
            'abdominal': 'abdominal',  # Correct spelling included
            
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
        words = corrected_text.split()
        final_words = []
        
        for word in words:
            if len(word) >= 4:  # Only fuzzy match longer words
                corrected_word = self._apply_fuzzy_correction(word, similarity_threshold)
                final_words.append(corrected_word)
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
        
        # Common medical terms for fuzzy matching
        medical_terms = [
            'abdominal', 'stomach', 'chest', 'heart', 'headache', 'nausea', 
            'vomiting', 'breathing', 'dizzy', 'belly', 'pain', 'ache'
        ]
        
        best_match = word_clean
        best_score = 0.0
        
        for medical_term in medical_terms:
            # Calculate similarity using SequenceMatcher
            similarity = SequenceMatcher(None, word_clean, medical_term).ratio()
            
            if similarity > threshold and similarity > best_score:
                best_score = similarity
                best_match = medical_term
        
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

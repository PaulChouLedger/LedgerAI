#!/usr/bin/env python3
"""
Medical Rule Engine - Simplified Universal Approach
Uses medical_rules.json and unified similarity function for all scoring
"""

import json
import os
import numpy as np
from typing import Dict, Any, List, Optional
from pathlib import Path

class MedicalRuleEngine:
    """
    Universal medical rule engine
    - Uses medical_rules.json for anatomical filtering
    - Unified similarity function for all OLDCARTS elements
    """
    
    def __init__(self, embedding_model=None):
        self.embedding_model = embedding_model
        self.medical_rules = self._load_medical_rules()
    
    def _load_medical_rules(self) -> Dict:
        """Load medical_rules.json"""
        current_file = Path(__file__).resolve()
        config_dir = current_file.parent.parent / 'config'
        json_path = config_dir / 'medical_rules.json'
        
        try:
            with open(json_path, 'r') as f:
                rules = json.load(f)
            return rules
        except Exception as e:
            print(f"[MedicalRules] ⚠️ Error loading rules: {e}")
            return {}
    
    def _get_condition_anatomical_type(self, condition_name: str, organ_system: str) -> Optional[str]:
        """Get anatomical type from medical_rules.json"""
        if not self.medical_rules or organ_system not in self.medical_rules:
            return None
        
        for anatomical_type, condition_list in self.medical_rules[organ_system].items():
            if condition_name in condition_list:
                return anatomical_type
        return None
    
    def _extract_directional_component(self, normalized_text: str, raw_text: str = None) -> Optional[str]:
        """
        Extract directional component using medical_rules.json structure
        Checks normalized category name against conditions in medical_rules.json
        """
        text_to_check = normalized_text.lower()
        if raw_text:
            text_to_check += " " + raw_text.lower()
        
        # Check for directional indicators (universal across all body systems)
        if any(word in text_to_check for word in ['right', 'ruq', 'rlq']):
            return 'right'
        elif any(word in text_to_check for word in ['left', 'luq', 'llq']):
            return 'left'
        elif any(word in text_to_check for word in ['bilateral', 'both sides', 'both']):
            return 'bilateral'
        elif any(word in text_to_check for word in ['midline', 'center', 'central', 'middle', 'epigastric', 'suprapubic']):
            return 'midline'
        
        return None
    
    def _normalize_with_synonyms(self, patient_text: str, synonyms: dict, oldcarts_element: str) -> str:
        """Normalize patient text using synonym file"""
        if oldcarts_element not in synonyms:
            return patient_text.lower().strip()
        
        patient_lower = patient_text.lower()
        element_synonyms = synonyms[oldcarts_element]
        
        # Exact substring matching
        for standard_term, synonym_list in element_synonyms.items():
            for synonym in synonym_list:
                if synonym.lower() in patient_lower:
                    return standard_term
        
        # Semantic matching fallback
        if self.embedding_model:
            best_match = None
            best_similarity = 0.0
            threshold = 0.65
            
            all_synonyms = []
            synonym_texts = []
            for standard_term, synonym_list in element_synonyms.items():
                for synonym in synonym_list:
                    all_synonyms.append((standard_term, synonym))
                    synonym_texts.append(synonym.lower())
            
            if all_synonyms:
                try:
                    all_texts = [patient_lower] + synonym_texts
                    embeddings = self.embedding_model.encode(all_texts)
                    patient_emb = embeddings[0]
                    
                    for i, (standard_term, synonym) in enumerate(all_synonyms):
                        synonym_emb = embeddings[i + 1]
                        similarity = float(np.dot(patient_emb, synonym_emb) / 
                                          (np.linalg.norm(patient_emb) * np.linalg.norm(synonym_emb)))
                        
                        if similarity > best_similarity:
                            best_similarity = similarity
                            best_match = standard_term
                    
                    if best_similarity >= threshold:
                        return best_match
                except Exception as e:
                    pass
        
        return patient_text.lower().strip()
    
    def compute_unified_similarity(self, patient_text: str, guideline_text: str, 
                                   condition_name: str, organ_system: str = None, 
                                   oldcarts_element: str = None, structured_oldcarts: dict = None,
                                   pre_normalized_text: str = None) -> Dict[str, Any]:
        """
        UNIFIED similarity function used for ALL OLDCARTS elements
        
        Flow:
        1. Raw semantic similarity (embeddings)
        2. Normalization (with semantic fallback)
        3. Word match boost (normalized text vs structured_oldcarts)
        """
        # STEP 1: Raw semantic similarity
        raw_similarity = 0.0
        if self.embedding_model:
            try:
                embeddings = self.embedding_model.encode([patient_text.lower(), guideline_text])
                raw_similarity = float(np.dot(embeddings[0], embeddings[1]) / 
                                      (np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])))
            except Exception as e:
                pass
        
        # STEP 2: Normalization
        if pre_normalized_text:
            normalized_text = pre_normalized_text
        else:
            normalized_text = patient_text.lower()
            if organ_system and oldcarts_element:
                synonym_file = f"synonyms/{organ_system.lower()}_synonyms_oldcarts.json"
                synonym_path = os.path.join(os.path.dirname(__file__), '..', synonym_file)
                
                if os.path.exists(synonym_path):
                    try:
                        with open(synonym_path, 'r') as f:
                            synonyms = json.load(f)
                        normalized_text = self._normalize_with_synonyms(patient_text, synonyms, oldcarts_element)
                    except Exception as e:
                        pass
        
        # STEP 3: Word match boost
        word_match_boost = 0.0
        if structured_oldcarts and oldcarts_element:
            word_match_boost = self._compute_word_match_boost(
                patient_text, normalized_text, guideline_text,
                organ_system, oldcarts_element, structured_oldcarts,
                condition_name
            )
        
        # STEP 4: Combine results
        final_similarity = raw_similarity + word_match_boost
        final_similarity = max(0.0, min(1.0, final_similarity))
        
        return {
            'similarity': final_similarity,
            'raw_similarity': raw_similarity,
            'word_match_boost': word_match_boost,
            'normalized_text': normalized_text,
            'method': 'unified_similarity'
        }
    
    def _compute_word_match_boost(self, patient_text: str, normalized_text: str,
                                  guideline_text: str, organ_system: str, 
                                  oldcarts_element: str, structured_oldcarts: dict,
                                  condition_name: str) -> float:
        """
        Compute word match boost using medical_rules.json and structured_oldcarts
        """
        if oldcarts_element not in structured_oldcarts:
            return 0.0
        
        element_data = structured_oldcarts[oldcarts_element]
        includes_terms = element_data.get('includes', [])
        excludes_terms = element_data.get('excludes', [])
        
        # STEP 1: Location-specific medical_rules.json check
        base_boost = 0.0
        if oldcarts_element == 'location' and condition_name and organ_system:
            anatomical_type = self._get_condition_anatomical_type(condition_name, organ_system)
            patient_direction = self._extract_directional_component(normalized_text, patient_text)
            
            if anatomical_type and patient_direction:
                # Boost matches, penalize opposites, keep vague/bilateral intact
                if anatomical_type == 'right_only':
                    if patient_direction == 'right':
                        base_boost = 0.3  # Match
                    elif patient_direction == 'left':
                        return -0.3  # Penalty
                elif anatomical_type == 'left_only':
                    if patient_direction == 'left':
                        base_boost = 0.3  # Match
                    elif patient_direction == 'right':
                        return -0.3  # Penalty
                # bilateral and midline: base_boost stays 0.0 (compatible)
        
        # STEP 2: Check excludes
        normalized_lower = normalized_text.lower()
        patient_lower = patient_text.lower()
        
        for term in excludes_terms:
            term_lower = term.lower()
            if (term_lower in normalized_lower or normalized_lower in term_lower or
                term_lower in patient_lower or patient_lower in term_lower):
                return -0.3  # Penalty
        
        # STEP 3: Check includes
        for term in includes_terms:
            term_lower = term.lower()
            
            # Exact match
            if (term_lower == normalized_lower or term_lower in normalized_lower or 
                normalized_lower in term_lower):
                return min(base_boost + 0.5, 0.5)
            
            # Word overlap
            term_words = set(term_lower.split())
            normalized_words = set(normalized_lower.replace('_', ' ').split())
            matching_words = normalized_words.intersection(term_words)
            
            if len(matching_words) >= 2:
                match_boost = min(0.2 * len(matching_words), 0.4)
                return min(base_boost + match_boost, 0.5)
        
        # STEP 4: Semantic matching fallback
        if self.embedding_model and includes_terms:
            try:
                all_texts = [normalized_lower] + [term.lower() for term in includes_terms]
                embeddings = self.embedding_model.encode(all_texts)
                normalized_emb = embeddings[0]
                
                best_similarity = 0.0
                for i, term in enumerate(includes_terms):
                    term_emb = embeddings[i + 1]
                    similarity = float(np.dot(normalized_emb, term_emb) / 
                                     (np.linalg.norm(normalized_emb) * np.linalg.norm(term_emb)))
                    if similarity > best_similarity:
                        best_similarity = similarity
                
                if best_similarity >= 0.65:
                    match_boost = min(0.2 + (best_similarity - 0.65) * 0.5, 0.4)
                    return min(base_boost + match_boost, 0.5)
            except Exception as e:
                pass
        
        # Return base_boost if any (from medical_rules.json)
        return base_boost
    
    def filter_guidelines_by_location(self, patient_answer: str, guidelines: List[Dict], 
                                     organ_system: str) -> List[Dict]:
        """
        Filter guidelines using medical_rules.json based on location
        
        Returns:
            Filtered guidelines (boost matches, rule out opposites, keep vague/bilateral intact)
        """
        if not organ_system or not self.medical_rules:
            return guidelines
        
        # Normalize patient answer
        synonym_file = f"synonyms/{organ_system.lower()}_synonyms_oldcarts.json"
        synonym_path = os.path.join(os.path.dirname(__file__), '..', synonym_file)
        
        normalized_answer = patient_answer.lower()
        if os.path.exists(synonym_path):
            try:
                with open(synonym_path, 'r') as f:
                    synonyms = json.load(f)
                normalized_answer = self._normalize_with_synonyms(patient_answer, synonyms, 'location')
            except Exception as e:
                pass
        
        patient_direction = self._extract_directional_component(normalized_answer, patient_answer)
        if not patient_direction:
            return guidelines  # No direction found, keep all
        
        filtered = []
        for guideline in guidelines:
            condition_name = guideline.get('data', {}).get('condition', guideline.get('name', ''))
            anatomical_type = self._get_condition_anatomical_type(condition_name, organ_system)
            
            if not anatomical_type:
                filtered.append(guideline)  # Unknown type, keep it
                continue
            
            # Check compatibility
            if anatomical_type == 'right_only':
                if patient_direction == 'left':
                    continue  # Rule out
            elif anatomical_type == 'left_only':
                if patient_direction == 'right':
                    continue  # Rule out
            
            # Keep: matches, bilateral, midline, vague
            filtered.append(guideline)
        
        return filtered

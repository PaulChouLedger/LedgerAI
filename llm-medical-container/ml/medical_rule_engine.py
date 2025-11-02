#!/usr/bin/env python3
"""
Medical Rule Engine - Simplified Universal Approach
Uses medical_rules.json and unified similarity function for all scoring
"""

import json
import os
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
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
        self.term_embeddings = {}
        self._build_term_indexes()
    
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
    
    def _build_term_indexes(self):
        """Build FAISS indexes for all OLDCARTS terms from guidelines."""
        if not self.embedding_model:
            print("[FAISS] ⚠️ No embedding model available, skipping term index building")
            return
        
        print("[FAISS] 🔨 Building term indexes...")
        
        # Collect all terms from guidelines
        all_terms = {
            'onset': set(), 'location': set(), 'duration': set(), 'character': set(),
            'aggravating': set(), 'relieving': set(), 'timing': set(), 'severity': set()
        }
        
        # Load guidelines and extract terms
        guidelines_path = os.path.join(os.path.dirname(__file__), '..', 'medical', 'guidelines')
        if not os.path.exists(guidelines_path):
            print(f"[FAISS] ⚠️ Guidelines path does not exist: {guidelines_path}")
            return
        
        guideline_count = 0
        for root, dirs, files in os.walk(guidelines_path):
            for file in files:
                if file.endswith('.json'):
                    try:
                        with open(os.path.join(root, file), 'r') as f:
                            guideline = json.load(f)
                            # Try both possible structures
                            structured = guideline.get('key_features', {}).get('structured_oldcarts', {})
                            if not structured:
                                structured = guideline.get('data', {}).get('key_features', {}).get('structured_oldcarts', {})
                            
                            if structured:
                                guideline_count += 1
                                for element, data in structured.items():
                                    if isinstance(data, dict) and 'includes' in data and element in all_terms:
                                        for term in data['includes']:
                                            # Support new structure where terms can be dicts {medical, patient_friendly}
                                            if isinstance(term, dict):
                                                medical = term.get('medical')
                                                if isinstance(medical, str) and medical.strip():
                                                    all_terms[element].add(medical.lower())
                                            elif isinstance(term, str):
                                                all_terms[element].add(term.lower())
                    except Exception as e:
                        print(f"[FAISS] ⚠️ Could not load guideline {file}: {e}")
        
        print(f"[FAISS] 📚 Loaded {guideline_count} guidelines")
        
        # Add synonyms to FAISS index and build synonym-to-medical mapping
        synonym_to_medical_mapping = {
            'onset': {}, 'location': {}, 'duration': {}, 'character': {},
            'aggravating': {}, 'relieving': {}, 'timing': {}, 'severity': {}
        }
        
        synonyms_dir = os.path.join(os.path.dirname(__file__), '..', 'synonyms')
        if os.path.exists(synonyms_dir):
            for synonym_file in os.listdir(synonyms_dir):
                if synonym_file.endswith('_synonyms_oldcarts.json'):
                    try:
                        synonym_path = os.path.join(synonyms_dir, synonym_file)
                        with open(synonym_path, 'r') as f:
                            synonyms = json.load(f)
                        
                        # Add all synonyms to term lists and build mapping
                        for element, synonym_dict in synonyms.items():
                            if element in all_terms:
                                for medical_term, synonym_list in synonym_dict.items():
                                    # Add medical term itself
                                    all_terms[element].add(medical_term.lower())
                                    # Map medical term to itself
                                    synonym_to_medical_mapping[element][medical_term.lower()] = medical_term.lower()
                                    # Add all patient-friendly synonyms and map them back to medical term
                                    for synonym in synonym_list:
                                        all_terms[element].add(synonym.lower())
                                        synonym_to_medical_mapping[element][synonym.lower()] = medical_term.lower()
                    except Exception as e:
                        print(f"[FAISS] ⚠️ Could not load synonyms from {synonym_file}: {e}")
        
        # Build FAISS indexes for each element
        for element, terms in all_terms.items():
            if terms:
                terms_list = list(terms)
                try:
                    embeddings = self.embedding_model.encode(terms_list)
                    embeddings = np.asarray(embeddings, dtype='float32')
                    
                    # Normalize embeddings for cosine similarity (required for IndexFlatIP)
                    faiss.normalize_L2(embeddings)
                    
                    # Create FAISS index
                    index = faiss.IndexFlatIP(embeddings.shape[1])  # Inner product for cosine similarity
                    index.add(embeddings)
                    
                    self.term_embeddings[element] = {
                        'terms': terms_list,
                        'embeddings': embeddings,
                        'index': index,
                        'synonym_to_medical': synonym_to_medical_mapping[element]
                    }
                    
                    print(f"[FAISS] ✅ Built index for {element}: {len(terms_list)} terms")
                except Exception as e:
                    print(f"[FAISS] ⚠️ Error building index for {element}: {e}")
                    import traceback
                    traceback.print_exc()
    
    def _normalize_term_list(self, terms: List[Any]) -> List[str]:
        """Normalize guideline term lists that may contain strings or {medical, patient_friendly} dicts."""
        normalized: List[str] = []
        for term in terms or []:
            if isinstance(term, dict):
                medical = term.get('medical')
                if isinstance(medical, str) and medical.strip():
                    normalized.append(medical.strip().lower())
            elif isinstance(term, str):
                normalized.append(term.strip().lower())
        return normalized
    
    def find_matching_terms_faiss(self, prompt: str, element: str, threshold: float = 0.65) -> List[str]:
        """Find matching terms using ONLY FAISS semantic similarity."""
        if element not in self.term_embeddings or not self.embedding_model:
            return []
        
        matches = []
        
        try:
            # Encode prompt
            prompt_embedding = self.embedding_model.encode([prompt])
            prompt_embedding = np.asarray(prompt_embedding, dtype='float32')
            
            # Normalize for cosine similarity (required for IndexFlatIP)
            faiss.normalize_L2(prompt_embedding)
            
            # Search FAISS index
            scores, indices = self.term_embeddings[element]['index'].search(
                prompt_embedding, k=10
            )
            
            # Filter by threshold and map synonyms back to medical terms
            synonym_to_medical = self.term_embeddings[element].get('synonym_to_medical', {})
            for score, idx in zip(scores[0], indices[0]):
                if score >= threshold:
                    term = self.term_embeddings[element]['terms'][idx]
                    # Map synonym back to medical term if available
                    medical_term = synonym_to_medical.get(term, term)
                    if medical_term not in matches:
                        matches.append(medical_term)
            
            return matches
        except Exception as e:
            print(f"[FAISS] ⚠️ Error in term matching: {e}")
            import traceback
            traceback.print_exc()
            return []
    
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
                    embeddings = np.asarray(embeddings, dtype='float32')
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
                embeddings = np.asarray(embeddings, dtype='float32')
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
        Compute word match boost using FAISS and medical_rules.json
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
        
        # STEP 2: FAISS-based term matching (only for the specific OLDCARTS element)
        if oldcarts_element in self.term_embeddings:
            # Get matches for THIS specific element only
            all_faiss_matches = self.find_matching_terms_faiss(patient_text, oldcarts_element, threshold=0.7)
            
            # Filter to only includes/excludes from THIS guideline
            excludes_lower = self._normalize_term_list(excludes_terms)
            includes_lower = self._normalize_term_list(includes_terms)
            
            # Check excludes (must be in both FAISS matches AND this guideline's excludes)
            matching_excludes = [term for term in all_faiss_matches if term.lower() in excludes_lower]
            if matching_excludes:
                return -0.3  # Penalty for exclude matches
            
            # Check includes (must be in both FAISS matches AND this guideline's includes)
            matching_includes = [term for term in all_faiss_matches if term.lower() in includes_lower]
            if matching_includes:
                # Calculate boost based on number of matches
                match_boost = min(0.1 * len(matching_includes), 0.4)
                return min(base_boost + match_boost, 0.5)
        
        # STEP 3: Fallback to exact matching if FAISS not available
        normalized_lower = normalized_text.lower()
        patient_lower = patient_text.lower()
        
        # Check excludes
        for term in self._normalize_term_list(excludes_terms):
            term_lower = term
            if (term_lower in normalized_lower or normalized_lower in term_lower or
                term_lower in patient_lower or patient_lower in term_lower):
                return -0.3  # Penalty
        
        # Check includes
        for term in self._normalize_term_list(includes_terms):
            term_lower = term
            
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
        
        # Return base_boost if any (from medical_rules.json)
        return base_boost
    
    def filter_guidelines_by_location(self, patient_answer: str, guidelines: List[Dict], 
                                     organ_system: str) -> List[Dict]:
        """
        UNIVERSAL: Filter guidelines using FAISS + medical_rules.json for ALL organ systems
        
        Flow:
        1. Use FAISS to find location matches across all medical conditions
        2. Extract direction from FAISS-matched terms
        3. Apply medical_rules.json filtering universally:
           - right → show right_only + bilateral + midline, rule out left_only
           - left → show left_only + bilateral + midline, rule out right_only
           - bilateral/midline → show all (compatible with any direction)
        
        Returns:
            Filtered guidelines based on anatomical compatibility
        """
        if not organ_system or not self.medical_rules:
            return guidelines
        
        # STEP 1: Use FAISS to find location matches (universal across all organ systems)
        patient_direction = None
        if 'location' in self.term_embeddings:
            # Find matching location terms using FAISS
            location_matches = self.find_matching_terms_faiss(patient_answer, 'location', threshold=0.65)
            
            if location_matches:
                # Extract direction from FAISS-matched terms
                patient_direction = self._extract_directional_component_from_terms(location_matches, patient_answer)
        
        # Fallback to simple keyword extraction if FAISS didn't find matches
        if not patient_direction:
            normalized_answer = patient_answer.lower()
            synonym_file = f"synonyms/{organ_system.lower()}_synonyms_oldcarts.json"
            synonym_path = os.path.join(os.path.dirname(__file__), '..', synonym_file)
            
            if os.path.exists(synonym_path):
                try:
                    with open(synonym_path, 'r') as f:
                        synonyms = json.load(f)
                    normalized_answer = self._normalize_with_synonyms(patient_answer, synonyms, 'location')
                except Exception:
                    pass
            
            patient_direction = self._extract_directional_component(normalized_answer, patient_answer)
        
        if not patient_direction:
            return guidelines  # No direction found, keep all
        
        # STEP 2: Apply medical_rules.json filtering UNIVERSALLY across all organ systems
        filtered = []
        for guideline in guidelines:
            condition_name = guideline.get('data', {}).get('condition', guideline.get('name', ''))
            anatomical_type = self._get_condition_anatomical_type(condition_name, organ_system)
            
            if not anatomical_type:
                filtered.append(guideline)  # Unknown type, keep it
                continue
            
            # UNIVERSAL medical_rules.json logic for ALL organ systems:
            # GI, CARDIO, PULMONARY, MSK, DERM, NEURO, RENAL, GU, GYN
            if anatomical_type == 'right_only':
                if patient_direction == 'left':
                    continue  # Rule out left_only when patient says "right"
                # Keep: right matches right_only, bilateral, midline, vague
            elif anatomical_type == 'left_only':
                if patient_direction == 'right':
                    continue  # Rule out left_only when patient says "right"
                # Keep: left matches left_only, bilateral, midline, vague
            # bilateral and midline: always keep (compatible with all directions)
            # This works for ALL organ systems: GI, CARDIO, PULMONARY, MSK, DERM, NEURO, RENAL, GU, GYN
            
            filtered.append(guideline)
        
        return filtered
    
    def _extract_directional_component_from_terms(self, matched_terms: List[str], raw_text: str = None) -> Optional[str]:
        """
        UNIVERSAL: Extract directional component from FAISS-matched location terms
        
        Works across ALL organ systems and medical conditions by analyzing
        the actual matched terms (e.g., "right side", "right lower quadrant", "left chest")
        to determine anatomical direction.
        
        Args:
            matched_terms: List of terms found by FAISS (e.g., ["right side", "right lower quadrant"])
            raw_text: Original patient text for additional context
            
        Returns:
            Direction: 'right', 'left', 'bilateral', 'midline', or None
        """
        combined_text = ' '.join(matched_terms).lower()
        if raw_text:
            combined_text += ' ' + raw_text.lower()
        
        # UNIVERSAL directional detection across ALL organ systems:
        # GI, CARDIO, PULMONARY, MSK, DERM, NEURO, RENAL, GU, GYN
        
        # Right-sided indicators
        if any(word in combined_text for word in ['right', 'ruq', 'rlq', 'right side', 'right sided']):
            return 'right'
        
        # Left-sided indicators  
        elif any(word in combined_text for word in ['left', 'luq', 'llq', 'left side', 'left sided']):
            return 'left'
        
        # Bilateral indicators
        elif any(word in combined_text for word in ['bilateral', 'both sides', 'both', 'symmetrical', 'all over', 'everywhere']):
            return 'bilateral'
        
        # Midline indicators
        elif any(word in combined_text for word in ['midline', 'center', 'central', 'middle', 'epigastric', 'suprapubic', 'periumbilical']):
            return 'midline'
        
        return None
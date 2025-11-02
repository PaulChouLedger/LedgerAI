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
        self.term_embeddings = {}  # Global index (legacy, kept for backward compatibility)
        self.term_embeddings_by_category = {}  # Category-specific indexes: {category: {element: {...}}}
        self.synonym_cache = {}  # Cache loaded synonym files to avoid repeated I/O
        self.active_category = None  # Currently active category
        self._build_category_specific_indexes()
    
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
    
    def _build_category_specific_indexes(self):
        """Build FAISS indexes separately for each category/organ system."""
        if not self.embedding_model:
            print("[FAISS] ⚠️ No embedding model available, skipping term index building")
            return
        
        print("[FAISS] 🔨 Building category-specific indexes...")
        
        # Map category names to organ system directories and synonym file prefixes
        category_to_dir = {
            'gastrointestinal': ('GI', 'gi'),
            'cardiovascular': ('CARDIO', 'cardio'),
            'respiratory': ('PULMONARY', 'resp'),
            'neurological': ('NEURO', 'neuro'),
            'musculoskeletal': ('MSK', 'msk'),
            'renal': ('RENAL', 'renal'),
            'genitourinary': ('GU', 'gu'),
            'gynecological': ('GYN', 'gyn'),  # May not have synonym file
            'dermatological': ('DERM', 'derm')
        }
        
        guidelines_path = os.path.join(os.path.dirname(__file__), '..', 'medical', 'guidelines')
        if not os.path.exists(guidelines_path):
            print(f"[FAISS] ⚠️ Guidelines path does not exist: {guidelines_path}")
            return
        
        # Build index for each category
        for category, (organ_system_dir, synonym_prefix) in category_to_dir.items():
            category_path = os.path.join(guidelines_path, organ_system_dir)
            if not os.path.exists(category_path):
                continue
            
            print(f"[FAISS] 🔨 Building index for {category} ({organ_system_dir})...")
            
            # Collect terms for this category only
            all_terms = {
                'onset': set(), 'location': set(), 'duration': set(), 'character': set(),
                'aggravating': set(), 'relieving': set(), 'timing': set(), 'severity': set()
            }
            term_to_conditions = {}
            
            # Load guidelines from this category only
            guideline_count = 0
            for file in os.listdir(category_path):
                if file.endswith('.json'):
                    try:
                        with open(os.path.join(category_path, file), 'r') as f:
                            guideline = json.load(f)
                            condition_name = guideline.get('condition', guideline.get('name', ''))
                            
                            structured = guideline.get('key_features', {}).get('structured_oldcarts', {})
                            if not structured:
                                structured = guideline.get('data', {}).get('key_features', {}).get('structured_oldcarts', {})
                            
                            if structured:
                                guideline_count += 1
                                for element, data in structured.items():
                                    if isinstance(data, dict) and 'includes' in data and element in all_terms:
                                        for term in data['includes']:
                                            medical_term = None
                                            if isinstance(term, dict):
                                                medical_term = term.get('medical')
                                                if isinstance(medical_term, str) and medical_term.strip():
                                                    medical_term = medical_term.strip().lower()
                                            elif isinstance(term, str):
                                                medical_term = term.strip().lower()
                                            
                                            if medical_term:
                                                all_terms[element].add(medical_term)
                                                key = (element, medical_term)
                                                if key not in term_to_conditions:
                                                    term_to_conditions[key] = set()
                                                term_to_conditions[key].add(condition_name)
                    except Exception as e:
                        print(f"[FAISS] ⚠️ Could not load guideline {file}: {e}")
            
            # Add category-specific synonyms
            synonym_to_medical_mapping = {
                'onset': {}, 'location': {}, 'duration': {}, 'character': {},
                'aggravating': {}, 'relieving': {}, 'timing': {}, 'severity': {}
            }
            
            synonyms_dir = os.path.join(os.path.dirname(__file__), '..', 'synonyms')
            synonym_file = f"{synonym_prefix}_synonyms_oldcarts.json"
            synonym_path = os.path.join(synonyms_dir, synonym_file)
            
            if os.path.exists(synonym_path):
                try:
                    with open(synonym_path, 'r') as f:
                        synonyms = json.load(f)
                    
                    for element, synonym_dict in synonyms.items():
                        if element in all_terms:
                            for medical_term, synonym_list in synonym_dict.items():
                                all_terms[element].add(medical_term.lower())
                                synonym_to_medical_mapping[element][medical_term.lower()] = medical_term.lower()
                                for synonym in synonym_list:
                                    all_terms[element].add(synonym.lower())
                                    synonym_to_medical_mapping[element][synonym.lower()] = medical_term.lower()
                except Exception as e:
                    print(f"[FAISS] ⚠️ Could not load synonyms from {synonym_file}: {e}")
            else:
                print(f"[FAISS] ℹ️ No synonym file found for {category} ({synonym_file}), continuing without synonyms")
            
            # Build FAISS indexes for this category
            category_indexes = {}
            for element, terms in all_terms.items():
                if terms:
                    terms_list = list(terms)
                    try:
                        embeddings = self.embedding_model.encode(terms_list)
                        embeddings = np.asarray(embeddings, dtype='float32')
                        faiss.normalize_L2(embeddings)
                        
                        index = faiss.IndexFlatIP(embeddings.shape[1])
                        index.add(embeddings)
                        
                        # Build term-to-conditions mapping for this element
                        element_term_to_conditions = {}
                        for term in terms_list:
                            key = (element, term)
                            if key in term_to_conditions:
                                element_term_to_conditions[term] = term_to_conditions[key]
                        
                        category_indexes[element] = {
                            'terms': terms_list,
                            'embeddings': embeddings,
                            'index': index,
                            'synonym_to_medical': synonym_to_medical_mapping[element],
                            'term_to_conditions': element_term_to_conditions
                        }
                    except Exception as e:
                        print(f"[FAISS] ⚠️ Error building index for {category}/{element}: {e}")
                        import traceback
                        traceback.print_exc()
            
            self.term_embeddings_by_category[category] = category_indexes
            print(f"[FAISS] ✅ Built index for {category}: {guideline_count} guidelines, {sum(len(idx['terms']) for idx in category_indexes.values())} total terms")
        
        # Also build global index for backward compatibility (used when category not yet determined)
        print(f"[FAISS] 🔨 Building global index (for initial parsing)...")
        self._build_global_index()
    
    def _build_global_index(self):
        """Build global index from all guidelines (for initial parsing before category is determined)."""
        all_terms = {
            'onset': set(), 'location': set(), 'duration': set(), 'character': set(),
            'aggravating': set(), 'relieving': set(), 'timing': set(), 'severity': set()
        }
        term_to_conditions = {}
        
        guidelines_path = os.path.join(os.path.dirname(__file__), '..', 'medical', 'guidelines')
        guideline_count = 0
        
        for root, dirs, files in os.walk(guidelines_path):
            for file in files:
                if file.endswith('.json'):
                    try:
                        with open(os.path.join(root, file), 'r') as f:
                            guideline = json.load(f)
                            condition_name = guideline.get('condition', guideline.get('name', ''))
                            
                            structured = guideline.get('key_features', {}).get('structured_oldcarts', {})
                            if not structured:
                                structured = guideline.get('data', {}).get('key_features', {}).get('structured_oldcarts', {})
                            
                            if structured:
                                guideline_count += 1
                                for element, data in structured.items():
                                    if isinstance(data, dict) and 'includes' in data and element in all_terms:
                                        for term in data['includes']:
                                            medical_term = None
                                            if isinstance(term, dict):
                                                medical_term = term.get('medical')
                                                if isinstance(medical_term, str) and medical_term.strip():
                                                    medical_term = medical_term.strip().lower()
                                            elif isinstance(term, str):
                                                medical_term = term.strip().lower()
                                            
                                            if medical_term:
                                                all_terms[element].add(medical_term)
                                                key = (element, medical_term)
                                                if key not in term_to_conditions:
                                                    term_to_conditions[key] = set()
                                                term_to_conditions[key].add(condition_name)
                    except Exception:
                        pass
        
        # Add all synonyms (global index includes all)
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
                        
                        for element, synonym_dict in synonyms.items():
                            if element in all_terms:
                                for medical_term, synonym_list in synonym_dict.items():
                                    all_terms[element].add(medical_term.lower())
                                    synonym_to_medical_mapping[element][medical_term.lower()] = medical_term.lower()
                                    for synonym in synonym_list:
                                        all_terms[element].add(synonym.lower())
                                        synonym_to_medical_mapping[element][synonym.lower()] = medical_term.lower()
                    except Exception:
                        pass
        
        # Build global indexes
        for element, terms in all_terms.items():
            if terms:
                terms_list = list(terms)
                try:
                    embeddings = self.embedding_model.encode(terms_list)
                    embeddings = np.asarray(embeddings, dtype='float32')
                    faiss.normalize_L2(embeddings)
                    
                    index = faiss.IndexFlatIP(embeddings.shape[1])
                    index.add(embeddings)
                    
                    element_term_to_conditions = {}
                    for term in terms_list:
                        key = (element, term)
                        if key in term_to_conditions:
                            element_term_to_conditions[term] = term_to_conditions[key]
                    
                    self.term_embeddings[element] = {
                        'terms': terms_list,
                        'embeddings': embeddings,
                        'index': index,
                        'synonym_to_medical': synonym_to_medical_mapping[element],
                        'term_to_conditions': element_term_to_conditions
                    }
                except Exception:
                    pass
        
        print(f"[FAISS] ✅ Built global index: {guideline_count} guidelines")
    
    def set_active_category(self, category: str):
        """Switch to category-specific indexes once category is determined."""
        self.active_category = category
        if category in self.term_embeddings_by_category:
            # Switch term_embeddings to category-specific
            self.term_embeddings = self.term_embeddings_by_category[category]
            total_terms = sum(len(idx['terms']) for idx in self.term_embeddings.values())
            elements_with_indexes = list(self.term_embeddings.keys())
            print(f"[FAISS] 🔀 Switched to {category} category index")
            print(f"[FAISS] 📊 Category index stats: {total_terms} total terms across {len(elements_with_indexes)} elements: {elements_with_indexes}")
            # Print detailed breakdown per element
            for element in elements_with_indexes:
                term_count = len(self.term_embeddings[element]['terms'])
                print(f"[FAISS]   - {element}: {term_count} terms")
        else:
            print(f"[FAISS] ⚠️ Category {category} not found, keeping global index")
            print(f"[FAISS] 📊 Available categories: {list(self.term_embeddings_by_category.keys())}")
    
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
    
    def find_matching_terms_faiss(self, prompt: str, element: str, threshold: float = 0.65, 
                                   return_scores: bool = False, active_condition_names: set = None) -> List[str]:
        """
        Find matching terms using ONLY FAISS semantic similarity.
        Uses category-specific index if category is set, otherwise uses global index.
        
        Args:
            prompt: Patient answer text
            element: OLDCARTS element (location, aggravating, etc.)
            threshold: Minimum similarity score (0.0-1.0)
            return_scores: If True, store scores in self._last_faiss_scores
            active_condition_names: Optional set of condition names to filter results (if None, returns all matches)
                                    Note: If category-specific index is used, this further filters within that category
        
        Returns:
            List of matching medical terms (filtered to active conditions if provided)
        """
        # Use category-specific index if available, otherwise global index
        indexes_to_use = self.term_embeddings
        index_type = "category-specific" if self.active_category else "global"
        
        if element not in indexes_to_use or not self.embedding_model:
            return []
        
        # Debug: Show which index is being used (only once per search to avoid spam)
        if not hasattr(self, '_last_index_debug') or self._last_index_debug != (self.active_category, element):
            term_count = len(indexes_to_use[element]['terms'])
            category_info = f"{self.active_category} category" if self.active_category else "global"
            print(f"[FAISS] 🔍 Using {category_info} index for {element} ({term_count} terms, {index_type})")
            self._last_index_debug = (self.active_category, element)
        
        matches = []
        match_scores = {}
        
        try:
            # Encode prompt
            prompt_embedding = self.embedding_model.encode([prompt])
            prompt_embedding = np.asarray(prompt_embedding, dtype='float32')
            
            # Normalize for cosine similarity (required for IndexFlatIP)
            faiss.normalize_L2(prompt_embedding)
            
            # Search FAISS index (category-specific if category is set)
            # Increase k to ensure we get enough matches after filtering
            k = 20 if active_condition_names else 10
            scores, indices = indexes_to_use[element]['index'].search(
                prompt_embedding, k=k
            )
            
            # Filter by threshold and map synonyms back to medical terms
            synonym_to_medical = indexes_to_use[element].get('synonym_to_medical', {})
            term_to_conditions = indexes_to_use[element].get('term_to_conditions', {})
            
            # Debug: show which index is being used (only print occasionally to avoid spam)
            # Removed frequent debug print - was causing output spam
            
            for score, idx in zip(scores[0], indices[0]):
                if score >= threshold:
                    term = indexes_to_use[element]['terms'][idx]
                    # Map synonym back to medical term if available
                    medical_term = synonym_to_medical.get(term, term)
                    
                    # Filter by active conditions if provided
                    if active_condition_names is not None:
                        # Check if this term is used by any active condition
                        term_conditions = term_to_conditions.get(term, set())
                        # If term has no condition mapping (e.g., synonyms), include it (universal terms)
                        # Otherwise, only include if used by active conditions
                        if term_conditions and not term_conditions.intersection(active_condition_names):
                            # This term is not used by any active condition - skip it
                            continue
                    
                    if medical_term not in matches:
                        matches.append(medical_term)
                    # Store score for debug purposes
                    if return_scores:
                        # Keep highest score if multiple synonyms map to same medical term
                        if medical_term not in match_scores or score > match_scores[medical_term]:
                            match_scores[medical_term] = float(score)
            
            # If debug mode, attach scores to function attribute (hacky but works)
            if return_scores:
                self._last_faiss_scores = match_scores
                # Also print for immediate debugging
                print(f"[FAISS] 🔍 Scores for '{prompt}' in {element}: {match_scores}")
            
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
        """Normalize patient text using FAISS semantic matching"""
        if oldcarts_element not in synonyms:
            return patient_text.lower().strip()
        
        # Use FAISS to find the best matching medical term
        faiss_matches = self.find_matching_terms_faiss(patient_text, oldcarts_element, threshold=0.75)
        if faiss_matches:
            # Return the best match (first one with highest score)
            return faiss_matches[0]
        
        return patient_text.lower().strip()
    
    def compute_unified_similarity(self, patient_text: str, guideline_text: str, 
                                   condition_name: str, organ_system: str = None, 
                                   oldcarts_element: str = None, structured_oldcarts: dict = None,
                                   pre_normalized_text: str = None, precomputed_similarity: float = None,
                                   active_condition_names: set = None) -> Dict[str, Any]:
        """
        UNIFIED similarity function used for ALL OLDCARTS elements
        
        Flow:
        1. Raw semantic similarity (embeddings)
        2. Normalization (with semantic fallback)
        3. Word match boost (normalized text vs structured_oldcarts)
        """
        # STEP 1: Raw semantic similarity
        raw_similarity = 0.0
        if precomputed_similarity is not None:
            # Use pre-computed similarity from batch embeddings
            raw_similarity = precomputed_similarity
        elif self.embedding_model:
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
                # Check synonym cache first
                cache_key = f"{organ_system.lower()}_{oldcarts_element}"
                if cache_key in self.synonym_cache:
                    synonyms = self.synonym_cache[cache_key]
                else:
                    synonym_file = f"synonyms/{organ_system.lower()}_synonyms_oldcarts.json"
                    synonym_path = os.path.join(os.path.dirname(__file__), '..', synonym_file)
                    
                    if os.path.exists(synonym_path):
                        try:
                            with open(synonym_path, 'r') as f:
                                all_synonyms = json.load(f)
                            synonyms = all_synonyms.get(oldcarts_element, {})
                            self.synonym_cache[cache_key] = synonyms
                        except Exception as e:
                            synonyms = {}
                    else:
                        synonyms = {}
                
                if synonyms:
                    normalized_text = self._normalize_with_synonyms(patient_text, synonyms, oldcarts_element)
        
        # STEP 3: Word match boost
        word_match_boost = 0.0
        if structured_oldcarts and oldcarts_element:
            word_match_boost = self._compute_word_match_boost(
                patient_text, normalized_text, guideline_text,
                organ_system, oldcarts_element, structured_oldcarts,
                condition_name, active_condition_names=active_condition_names
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
                                  condition_name: str, active_condition_names: set = None) -> float:
        """
        Simplified word match boost: detect matches vs mismatches, boost or don't boost accordingly.
        """
        if oldcarts_element not in structured_oldcarts:
            return 0.0
        
        element_data = structured_oldcarts[oldcarts_element]
        includes_terms = element_data.get('includes', [])
        excludes_terms = element_data.get('excludes', [])
        
        normalized_lower = normalized_text.lower()
        patient_lower = patient_text.lower()
        includes_lower = self._normalize_term_list(includes_terms)
        excludes_lower = self._normalize_term_list(excludes_terms)
        
        # STEP 1: Check excludes first (immediate penalty)
        for term in excludes_lower:
            if (term in normalized_lower or normalized_lower in term or
                term in patient_lower or patient_lower in term):
                return -0.3  # Penalty
        
        # STEP 2: For location, check for anatomical mismatches before boosting (universal for all organ systems)
        # This applies during ANY location question, including clarification questions
        if oldcarts_element == 'location':
            # Extract anatomical components from patient's normalized text (using medical_rules.json)
            patient_components = self._extract_anatomical_components(normalized_text)
            
            # Check for anatomical mismatch at the guideline level (if ALL anatomically-specific location terms are mismatched, apply penalty)
            if patient_components:
                anatomically_specific_terms = []  # Terms with anatomical components
                all_specific_mismatched = True  # All anatomically-specific terms are mismatched
                has_matching_term = False
                
                for term in includes_lower:
                    condition_components = self._extract_anatomical_components(term)
                    
                    # If term has anatomical components, check against patient's components
                    if condition_components:
                        anatomically_specific_terms.append(term)
                        if self._are_anatomical_opposites(patient_components, condition_components):
                            # This anatomically-specific term is a mismatch
                            continue
                        else:
                            # Found at least one non-mismatched anatomically-specific term - don't apply penalty
                            all_specific_mismatched = False
                            # Check if it also matches (exact/substring)
                            if (term == normalized_lower or term in normalized_lower or 
                                normalized_lower in term or term in patient_lower or 
                                patient_lower in term):
                                has_matching_term = True
                                # Exact match gets full boost
                                if term == normalized_lower or normalized_lower == term:
                                    return 0.5
                                # Substring match gets partial boost
                                return 0.3
                    else:
                        # Term has no anatomical components (e.g., "abdomen", "right side")
                        # Check if it matches via substring/exact (these are general terms)
                        if (term == normalized_lower or term in normalized_lower or 
                            normalized_lower in term or term in patient_lower or 
                            patient_lower in term):
                            has_matching_term = True
                            # Exact match gets full boost
                            if term == normalized_lower or normalized_lower == term:
                                return 0.5
                            # Substring match gets partial boost
                            return 0.3
                
                # If we have anatomically-specific terms AND ALL of them are mismatched, apply penalty
                # This happens during clarification when patient specifies a precise location
                if anatomically_specific_terms and all_specific_mismatched and not has_matching_term:
                    print(f"[Anatomical Mismatch] ⚠️ Penalty applied: Patient '{normalized_text}' ({patient_components}) vs condition location terms {anatomically_specific_terms} (all mismatched)")
                    return -0.3  # Penalty for anatomical mismatch (e.g., RUQ vs RLQ during clarification)
            
            # Fallback: Check each term individually for exact/substring matches
            for term in includes_lower:
                condition_components = self._extract_anatomical_components(term)
                
                # If both have anatomical components, check if they're opposites
                if patient_components and condition_components:
                    if self._are_anatomical_opposites(patient_components, condition_components):
                        # Anatomical mismatch - skip this term
                        continue
                
                # Exact or substring match (and not opposite)
                if (term == normalized_lower or term in normalized_lower or 
                    normalized_lower in term or term in patient_lower or 
                    patient_lower in term):
                    # Exact match gets full boost
                    if term == normalized_lower or normalized_lower == term:
                        return 0.5
                    # Substring match gets partial boost
                    return 0.3
        
        # STEP 3: FAISS-based term matching (for all elements)
        if oldcarts_element in self.term_embeddings:
            all_faiss_matches = self.find_matching_terms_faiss(
                patient_text, oldcarts_element, threshold=0.7, active_condition_names=active_condition_names
            )
            
            # Check excludes
            matching_excludes = [term for term in all_faiss_matches if term.lower() in excludes_lower]
            if matching_excludes:
                return -0.3
            
            # Check includes
            matching_includes = [term for term in all_faiss_matches if term.lower() in includes_lower]
            if matching_includes:
                # For location, check for anatomical mismatches (universal)
                if oldcarts_element == 'location':
                    patient_components = self._extract_anatomical_components(normalized_text)
                    if patient_components:
                        # Check if ALL matched terms are mismatched
                        all_faiss_mismatched = True
                        valid_matches = []
                        
                        for matched_term in matching_includes:
                            condition_components = self._extract_anatomical_components(matched_term.lower())
                            if condition_components:
                                if self._are_anatomical_opposites(patient_components, condition_components):
                                    # Mismatch - skip this term
                                    continue
                                else:
                                    # Found at least one non-mismatched term
                                    all_faiss_mismatched = False
                                    valid_matches.append(matched_term)
                            else:
                                # No anatomical components in term - consider it valid
                                all_faiss_mismatched = False
                                valid_matches.append(matched_term)
                        
                        # If ALL FAISS matches are anatomically mismatched, apply penalty
                        if all_faiss_mismatched and len(valid_matches) == 0:
                            return -0.3  # Penalty for anatomical mismatch
                        
                        # Use only valid (non-mismatched) matches for boost calculation
                        matching_includes = valid_matches
                
                # Good match - boost based on number
                if matching_includes:
                    match_boost = min(0.1 * len(matching_includes), 0.4)
                    return match_boost
        
        # STEP 4: Fallback exact matching
        for term in includes_lower:
            # Exact match
            if term == normalized_lower or normalized_lower == term:
                return 0.5
            
            # Substring match
            if (term in normalized_lower or normalized_lower in term or
                term in patient_lower or patient_lower in term):
                return 0.3
        
        return 0.0
    
    def _extract_anatomical_components(self, text: str) -> Dict[str, str]:
        """
        Universal: Extract all anatomical components from text using medical_rules.json.
        Returns dict with keys: 'quadrant', 'horizontal', 'vertical', 'anterior_posterior'
        """
        if not self.medical_rules or 'anatomical_components' not in self.medical_rules:
            return {}
        
        text_lower = text.lower()
        components = {}
        anatomical = self.medical_rules.get('anatomical_components', {})
        
        # Extract quadrant (GI: right_upper, right_lower, etc.)
        quadrant_patterns = anatomical.get('quadrant_patterns', {})
        for quadrant_key, patterns in quadrant_patterns.items():
            if any(pattern in text_lower for pattern in patterns):
                components['quadrant'] = quadrant_key
                break
        
        # Extract horizontal direction (left/right)
        horizontal = anatomical.get('directional_keywords', {}).get('horizontal', {})
        for direction, keywords in horizontal.items():
            if any(keyword in text_lower for keyword in keywords):
                components['horizontal'] = direction
                break
        
        # Extract vertical direction (upper/lower)
        vertical = anatomical.get('directional_keywords', {}).get('vertical', {})
        for direction, keywords in vertical.items():
            if any(keyword in text_lower for keyword in keywords):
                components['vertical'] = direction
                break
        
        # Extract anterior/posterior
        anterior_posterior = anatomical.get('directional_keywords', {}).get('anterior_posterior', {})
        for direction, keywords in anterior_posterior.items():
            if any(keyword in text_lower for keyword in keywords):
                components['anterior_posterior'] = direction
                break
        
        return components
    
    def _are_anatomical_opposites(self, components1: Dict[str, str], components2: Dict[str, str]) -> bool:
        """
        Universal: Check if two sets of anatomical components are opposites using medical_rules.json.
        Works for all organ systems: GI (quadrants), CARDIO/PULMONARY (chest), MSK (limbs), etc.
        
        Also handles quadrant vs. vertical comparison (e.g., "right_upper" quadrant vs. "upper" vertical).
        """
        if not self.medical_rules or 'anatomical_opposites' not in self.medical_rules:
            return False
        
        opposites = self.medical_rules.get('anatomical_opposites', {})
        
        # Extract vertical from quadrant if needed (e.g., "right_upper" → "upper")
        def extract_vertical_from_quadrant(quadrant_key):
            if not quadrant_key or '_' not in quadrant_key:
                return None
            parts = quadrant_key.split('_')
            if len(parts) >= 2:
                if parts[1] in ['upper', 'lower']:
                    return parts[1]
                # Handle "right_upper_quadrant" format
                if len(parts) >= 3 and parts[1] in ['upper', 'lower']:
                    return parts[1]
            return None
        
        # Get vertical components (direct or extracted from quadrant)
        vertical1 = components1.get('vertical')
        if not vertical1 and 'quadrant' in components1:
            vertical1 = extract_vertical_from_quadrant(components1['quadrant'])
        
        vertical2 = components2.get('vertical')
        if not vertical2 and 'quadrant' in components2:
            vertical2 = extract_vertical_from_quadrant(components2['quadrant'])
        
        # Check vertical opposites (most important for upper/lower quadrant distinction)
        if vertical1 and vertical2:
            vertical_opposites = opposites.get('vertical', {})
            opposite_list = vertical_opposites.get(vertical1, [])
            if vertical2 in opposite_list:
                return True
        
        # Check quadrants (GI) - full quadrant comparison
        if 'quadrant' in components1 and 'quadrant' in components2:
            quadrant_opposites = opposites.get('quadrants', {})
            opposite_list = quadrant_opposites.get(components1['quadrant'], [])
            if components2['quadrant'] in opposite_list:
                return True
        
        # Check horizontal (left/right) - universal for all systems
        # Extract horizontal from quadrant if needed
        horizontal1 = components1.get('horizontal')
        if not horizontal1 and 'quadrant' in components1:
            quadrant_parts = components1['quadrant'].split('_')
            if quadrant_parts and quadrant_parts[0] in ['left', 'right']:
                horizontal1 = quadrant_parts[0]
        
        horizontal2 = components2.get('horizontal')
        if not horizontal2 and 'quadrant' in components2:
            quadrant_parts = components2['quadrant'].split('_')
            if quadrant_parts and quadrant_parts[0] in ['left', 'right']:
                horizontal2 = quadrant_parts[0]
        
        if horizontal1 and horizontal2:
            horizontal_opposites = opposites.get('horizontal', {})
            opposite_list = horizontal_opposites.get(horizontal1, [])
            if horizontal2 in opposite_list:
                return True
        
        # Check anterior/posterior (front/back)
        if 'anterior_posterior' in components1 and 'anterior_posterior' in components2:
            ap_opposites = opposites.get('anterior_posterior', {})
            opposite_list = ap_opposites.get(components1['anterior_posterior'], [])
            if components2['anterior_posterior'] in opposite_list:
                return True
        
        return False
    
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
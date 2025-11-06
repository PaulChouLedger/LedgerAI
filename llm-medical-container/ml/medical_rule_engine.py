#!/usr/bin/env python3
"""
Medical Rule Engine - Simplified Universal Approach
Uses medical_rules.json and semantic similarity matching for all scoring
"""

# ============================================================================
# Section 1: Configuration (Top)
# ============================================================================

import json
import os
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from typing import Dict, Any, List, Optional
from pathlib import Path

# Import ML enhancer
try:
    from ml.semantic_ml_enhancer import SemanticMLEnhancer
    ML_ENHANCER_AVAILABLE = True
except ImportError:
    ML_ENHANCER_AVAILABLE = False
    print("[MedicalRules] ⚠️ ML enhancer not available. Semantic matching will use base scores only.")


class MedicalRuleEngine:
    """
    Universal medical rule engine
    - Uses medical_rules.json for anatomical filtering
    - Semantic similarity matching for all OLDCARTS elements
    """
    
    # ============================================================================
    # Section 2: Initialization
    # ============================================================================
    
    def __init__(self, embedding_model=None, enable_ml_learning: bool = None):
        self.embedding_model = embedding_model
        self.medical_rules = self._load_medical_rules()
        self.term_embeddings_by_category = {}  # Category-specific indexes: {category: {element: {...}}}
        # REMOVED: synonym_cache - synonym files no longer used
        self.active_category = None  # Currently active category
        self.term_embeddings = {}  # Current index (global or category-specific)
        self.global_term_embeddings = {}  # Preserved global index (for multi-category matching)
        
        # Initialize ML enhancer
        if enable_ml_learning is None:
            # Check environment variable
            enable_ml_learning = os.environ.get('ENABLE_ML_LEARNING', 'false').lower() == 'true'
        
        self.ml_enhancer = None
        if ML_ENHANCER_AVAILABLE and enable_ml_learning:
            try:
                self.ml_enhancer = SemanticMLEnhancer(enable_learning=enable_ml_learning)
                print("[MedicalRules] ✅ ML enhancer initialized")
            except Exception as e:
                print(f"[MedicalRules] ⚠️ Failed to initialize ML enhancer: {e}")
                self.ml_enhancer = None
        
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
        
        # Get enabled categories from environment variable (same as adaptive_diagnostic_engine)
        enabled_categories_env = os.environ.get('ENABLED_MEDICAL_CATEGORIES', 'GI').strip()
        enabled_categories = [cat.strip().upper() for cat in enabled_categories_env.split(',') if cat.strip()]
        
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
        
        # Build index for each category (only if enabled)
        for category, (organ_system_dir, synonym_prefix) in category_to_dir.items():
            # Filter by enabled categories
            if enabled_categories and organ_system_dir.upper() not in enabled_categories:
                print(f"[FAISS] ⏭️  Skipping {category} ({organ_system_dir}) - not in enabled categories")
                continue
            category_path = os.path.join(guidelines_path, organ_system_dir)
            if not os.path.exists(category_path):
                continue
            
            print(f"[FAISS] 🔨 Building index for {category} ({organ_system_dir})...")
            
            # Collect terms for this category only
            all_terms = {
                'onset': set(), 'location': set(), 'duration': set(), 'character': set(),
                'aggravating': set(), 'relieving': set(), 'timing': set(), 'severity': set(),
                'associated': set(), 'radiation': set()
            }
            term_to_conditions = {}
            synonym_to_medical_mapping = {
                'onset': {}, 'location': {}, 'duration': {}, 'character': {},
                'aggravating': {}, 'relieving': {}, 'timing': {}, 'severity': {},
                'associated': {}, 'radiation': {}
            }
            
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
                                            # Use patient_friendly terms for indexing (patients speak in patient_friendly terms)
                                            patient_friendly_term = None
                                            medical_term = None
                                            
                                            if isinstance(term, dict):
                                                # Get patient_friendly first (preferred for semantic matching)
                                                patient_friendly_term = term.get('patient_friendly')
                                                if isinstance(patient_friendly_term, str) and patient_friendly_term.strip():
                                                    patient_friendly_term = patient_friendly_term.strip()
                                                # Also get medical term for mapping
                                                medical_term = term.get('medical')
                                                if isinstance(medical_term, str) and medical_term.strip():
                                                    medical_term = medical_term.strip().lower()
                                            elif isinstance(term, str):
                                                # Fallback: use string as both patient_friendly and medical
                                                patient_friendly_term = term.strip()
                                                medical_term = term.strip().lower()
                                            
                                            # Index patient_friendly term (what patients actually say)
                                            if patient_friendly_term:
                                                all_terms[element].add(patient_friendly_term)
                                                # Map patient_friendly -> medical term for result mapping
                                                key = (element, patient_friendly_term)
                                                if key not in term_to_conditions:
                                                    term_to_conditions[key] = set()
                                                term_to_conditions[key].add(condition_name)
                                                # Store mapping from patient_friendly to medical term (for returning medical terms)
                                                if medical_term and medical_term != patient_friendly_term.lower():
                                                    if element not in synonym_to_medical_mapping:
                                                        synonym_to_medical_mapping[element] = {}
                                                    synonym_to_medical_mapping[element][patient_friendly_term.lower()] = medical_term
                    except Exception as e:
                        print(f"[FAISS] ⚠️ Could not load guideline {file}: {e}")
            
            # REMOVED: Synonym file loading - no longer needed since we index patient_friendly terms directly
            # The patient_friendly -> medical mapping is built from guidelines themselves
            
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
                        
                        # DEBUG: Show sample terms to verify synonyms are included
                        if element == 'location' and 'right upper quadrant' in synonym_to_medical_mapping[element]:
                            sample_synonyms = [t for t in terms_list[:20] if 'right' in t.lower() or 'upper' in t.lower()]
                            print(f"[FAISS] 🔍 Sample location terms in index: {sample_synonyms[:10]}")
                    except Exception as e:
                        print(f"[FAISS] ⚠️ Error building index for {category}/{element}: {e}")
                        import traceback
                        traceback.print_exc()
            
            self.term_embeddings_by_category[category] = category_indexes
            print(f"[FAISS] ✅ Built index for {category}: {guideline_count} guidelines, {sum(len(idx['terms']) for idx in category_indexes.values())} total terms")
        
        # Also build global index for backward compatibility (used when category not yet determined)
        print(f"[FAISS] 🔨 Building global index (for initial parsing)...")
        self._build_global_index()
        # Preserve global index separately for multi-category matching
        # Note: FAISS indexes can't be deep copied, so we just reference the same dict
        # This is safe because we only read from it, never modify it
        self.global_term_embeddings = self.term_embeddings
    
    def get_all_indexed_terms(self) -> set:
        """
        Get all patient_friendly terms from all categories and elements in the FAISS indexes.
        This can be used by fuzzy matcher to match against actual guideline terms.
        
        Returns:
            Set of all patient_friendly terms indexed across all categories and OLDCARTS elements
        """
        all_terms = set()
        
        # Collect terms from category-specific indexes
        for category, category_indexes in self.term_embeddings_by_category.items():
            for element, index_data in category_indexes.items():
                if 'terms' in index_data:
                    all_terms.update(index_data['terms'])
        
        # Also collect terms from global index
        for element, index_data in self.term_embeddings.items():
            if 'terms' in index_data:
                all_terms.update(index_data['terms'])
        
        return all_terms
    
    def _build_global_index(self):
        """Build global index from all guidelines (for initial parsing before category is determined)."""
        all_terms = {
            'onset': set(), 'location': set(), 'duration': set(), 'character': set(),
            'aggravating': set(), 'relieving': set(), 'timing': set(), 'severity': set(),
            'associated': set(), 'radiation': set()
        }
        term_to_conditions = {}
        synonym_to_medical_mapping = {
            'onset': {}, 'location': {}, 'duration': {}, 'character': {},
            'aggravating': {}, 'relieving': {}, 'timing': {}, 'severity': {},
            'associated': {}, 'radiation': {}
        }
        
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
                                            # Use patient_friendly terms for indexing (patients speak in patient_friendly terms)
                                            patient_friendly_term = None
                                            medical_term = None
                                            
                                            if isinstance(term, dict):
                                                # Get patient_friendly first (preferred for semantic matching)
                                                patient_friendly_term = term.get('patient_friendly')
                                                if isinstance(patient_friendly_term, str) and patient_friendly_term.strip():
                                                    patient_friendly_term = patient_friendly_term.strip()
                                                # Also get medical term for mapping
                                                medical_term = term.get('medical')
                                                if isinstance(medical_term, str) and medical_term.strip():
                                                    medical_term = medical_term.strip().lower()
                                            elif isinstance(term, str):
                                                # Fallback: use string as both patient_friendly and medical
                                                patient_friendly_term = term.strip()
                                                medical_term = term.strip().lower()
                                            
                                            # Index patient_friendly term (what patients actually say)
                                            if patient_friendly_term:
                                                all_terms[element].add(patient_friendly_term)
                                                # Map patient_friendly -> medical term for result mapping
                                                key = (element, patient_friendly_term)
                                                if key not in term_to_conditions:
                                                    term_to_conditions[key] = set()
                                                term_to_conditions[key].add(condition_name)
                                                # Store mapping from patient_friendly to medical term (for returning medical terms)
                                                if medical_term and medical_term != patient_friendly_term.lower():
                                                    if element not in synonym_to_medical_mapping:
                                                        synonym_to_medical_mapping[element] = {}
                                                    synonym_to_medical_mapping[element][patient_friendly_term.lower()] = medical_term
                    except Exception:
                        pass
        
        # REMOVED: Synonym file loading - no longer needed since we index patient_friendly terms directly
        # The patient_friendly -> medical mapping is built from guidelines themselves
        
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
                        'synonym_to_medical': synonym_to_medical_mapping.get(element, {}),
                        'term_to_conditions': element_term_to_conditions
                    }
                except Exception:
                    pass
        
        print(f"[FAISS] ✅ Built global index: {guideline_count} guidelines")
    
    def _merge_category_indexes(self, categories: List[str]) -> Dict:
        """Merge multiple category-specific indexes into a single combined index."""
        merged_indexes = {}
        all_elements = set()
        
        # Collect all elements from all categories
        for category in categories:
            if category in self.term_embeddings_by_category:
                all_elements.update(self.term_embeddings_by_category[category].keys())
        
        # Merge indexes for each element
        for element in all_elements:
            all_terms = []
            all_embeddings_list = []
            all_term_to_conditions = {}
            all_synonym_to_medical = {}
            
            # Collect terms and embeddings from all categories for this element
            for category in categories:
                if category in self.term_embeddings_by_category:
                    category_indexes = self.term_embeddings_by_category[category]
                    if element in category_indexes:
                        index_data = category_indexes[element]
                        terms = index_data.get('terms', [])
                        embeddings = index_data.get('embeddings', None)
                        term_to_conditions = index_data.get('term_to_conditions', {})
                        synonym_to_medical = index_data.get('synonym_to_medical', {})
                        
                        # Add terms and embeddings (avoid duplicates)
                        for term_idx, term in enumerate(terms):
                            if term not in all_terms:
                                embedding_to_add = None
                                if embeddings is not None and term_idx < len(embeddings):
                                    # Use the embedding at the same index as the term
                                    embedding_to_add = embeddings[term_idx]
                                elif self.embedding_model:
                                    # Fallback: encode on the fly if needed
                                    embedding_to_add = self.embedding_model.encode([term])[0]
                                else:
                                    print(f"[FAISS] ⚠️ No embedding available for term: {term}, skipping")
                                    continue  # Skip this term if no embedding available
                                
                                # Only add if we have an embedding
                                if embedding_to_add is not None:
                                    all_terms.append(term)
                                    all_embeddings_list.append(embedding_to_add)
                                    
                                    # Merge term_to_conditions
                                    if term in term_to_conditions:
                                        if term not in all_term_to_conditions:
                                            all_term_to_conditions[term] = set()
                                        if isinstance(term_to_conditions[term], set):
                                            all_term_to_conditions[term].update(term_to_conditions[term])
                                        else:
                                            all_term_to_conditions[term].add(term_to_conditions[term])
                                    
                                    # Merge synonym_to_medical (later categories override earlier ones)
                                    if term.lower() in synonym_to_medical:
                                        all_synonym_to_medical[term.lower()] = synonym_to_medical[term.lower()]
            
            # Build merged FAISS index for this element
            if all_terms and all_embeddings_list:
                try:
                    embeddings_array = np.vstack(all_embeddings_list).astype('float32')
                    faiss.normalize_L2(embeddings_array)
                    
                    index = faiss.IndexFlatIP(embeddings_array.shape[1])
                    index.add(embeddings_array)
                    
                    # Convert sets to lists for term_to_conditions
                    term_to_conditions_final = {}
                    for term, conditions in all_term_to_conditions.items():
                        if isinstance(conditions, set):
                            term_to_conditions_final[term] = conditions
                        else:
                            term_to_conditions_final[term] = {conditions}
                    
                    merged_indexes[element] = {
                        'terms': all_terms,
                        'embeddings': embeddings_array,
                        'index': index,
                        'synonym_to_medical': all_synonym_to_medical,
                        'term_to_conditions': term_to_conditions_final
                    }
                except Exception as e:
                    print(f"[FAISS] ⚠️ Error merging indexes for element {element}: {e}")
        
        return merged_indexes
    
    def set_active_category(self, category = None):
        """
        Switch to category-specific indexes once category is determined.
        
        Args:
            category: Can be:
                - str: Single category name
                - List[str]: Multiple categories (will merge their indexes)
                - None: Reset to global index
        """
        if category is None:
            # Reset to global index
            if self.global_term_embeddings:
                self.term_embeddings = self.global_term_embeddings
                self.active_category = None
                total_terms = sum(len(idx['terms']) for idx in self.term_embeddings.values())
                elements_with_indexes = list(self.term_embeddings.keys())
                print(f"[FAISS] 🔀 Reset to global index")
                print(f"[FAISS] 📊 Global index stats: {total_terms} total terms across {len(elements_with_indexes)} elements: {elements_with_indexes}")
            else:
                print(f"[FAISS] ⚠️ Global index not available")
            return
        
        # Handle list of categories (multi-category merge)
        if isinstance(category, list):
            if len(category) == 1:
                category = category[0]  # Single category, treat as string
            else:
                # Multiple categories: merge their indexes
                self.active_category = ','.join(category)
                merged_indexes = self._merge_category_indexes(category)
                if merged_indexes:
                    self.term_embeddings = merged_indexes
                    total_terms = sum(len(idx['terms']) for idx in self.term_embeddings.values())
                    elements_with_indexes = list(self.term_embeddings.keys())
                    print(f"[FAISS] 🔀 Merged indexes for {len(category)} categories: {', '.join(category)}")
                    print(f"[FAISS] 📊 Merged index stats: {total_terms} total terms across {len(elements_with_indexes)} elements: {elements_with_indexes}")
                    # Print detailed breakdown per element
                    for element in elements_with_indexes:
                        term_count = len(self.term_embeddings[element]['terms'])
                        print(f"[FAISS]   - {element}: {term_count} terms")
                else:
                    print(f"[FAISS] ⚠️ Failed to merge categories, using global index")
                    if self.global_term_embeddings:
                        self.term_embeddings = self.global_term_embeddings
                return
        
        # Handle single category (string)
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
    
    # ============================================================================
    # Section 3: Semantic Similarity (core matching)
    # ============================================================================
    
    def find_matching_terms_faiss(self, prompt: str, element: str, threshold: float = 0.65, 
                                   return_scores: bool = False, active_condition_names: set = None) -> List[str]:
        """
        Find matching terms using ONLY FAISS semantic similarity.
        Uses category-specific index if category is set, otherwise uses global index.
        ML enhancement can improve scores and adjust thresholds dynamically.
        
        Args:
            prompt: Patient answer text
            element: OLDCARTS element (location, aggravating, etc.)
            threshold: Minimum similarity score (0.0-1.0) - may be adjusted by ML
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
        
        # Get adaptive threshold from ML enhancer if available
        if self.ml_enhancer:
            category = self.active_category or 'default'
            adaptive_threshold = self.ml_enhancer.get_adaptive_threshold(element, category)
            # Use the more permissive threshold (lower value)
            threshold = min(threshold, adaptive_threshold)
        
        # Debug: Show which index is being used (only once per search to avoid spam)
        if not hasattr(self, '_last_index_debug') or self._last_index_debug != (self.active_category, element):
            term_count = len(indexes_to_use[element]['terms'])
            category_info = f"{self.active_category} category" if self.active_category else "global"
            ml_info = " (ML-enhanced)" if self.ml_enhancer else ""
            print(f"[FAISS] 🔍 Using {category_info} index for {element} ({term_count} terms, {index_type}){ml_info}")
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
            
            # FIRST PASS: Store ALL scores (including below threshold) if return_scores=True
            # Apply ML enhancement to improve scores
            improved_scores = {}
            for score, idx in zip(scores[0], indices[0]):
                term = indexes_to_use[element]['terms'][idx]
                raw_score = float(score)
                
                # Apply ML enhancement if available
                if self.ml_enhancer:
                    try:
                        category = self.active_category or 'default'
                        guideline_text = term  # Use term as guideline text for comparison
                        improved_score = self.ml_enhancer.improve_similarity_score(
                            raw_score, prompt, guideline_text, element, category, prompt_embedding[0]
                        )
                        improved_scores[term] = improved_score
                    except Exception:
                        improved_scores[term] = raw_score
                else:
                    improved_scores[term] = raw_score
                
                # Store in match_scores for return_scores
                if return_scores:
                    if term not in match_scores or improved_scores[term] > match_scores[term]:
                        match_scores[term] = improved_scores[term]
                    # Also store medical term score
                    medical_term = synonym_to_medical.get(term, term)
                    if medical_term not in match_scores or improved_scores[term] > match_scores[medical_term]:
                        match_scores[medical_term] = improved_scores[term]
            
            # SECOND PASS: Filter by threshold using improved scores and build matches list
            for idx in indices[0]:
                term = indexes_to_use[element]['terms'][idx]
                score = improved_scores.get(term, scores[0][list(indices[0]).index(idx)])
                
                if score >= threshold:
                    # Map patient_friendly back to medical term if available
                    medical_term = synonym_to_medical.get(term, term)
                    
                    # Filter by active conditions if provided
                    if active_condition_names is not None:
                        # Check if this term is used by any active condition
                        term_conditions = term_to_conditions.get(term, set())
                        # If term has no condition mapping, include it (universal terms)
                        # Otherwise, only include if used by active conditions
                        if term_conditions and not term_conditions.intersection(active_condition_names):
                            # This term is not used by any active condition - skip it
                            continue
                    
                    # Return patient_friendly term (what was actually indexed and matched)
                    if term not in matches:
                        matches.append(term)
            
            # Store scores for debugging purposes
            if return_scores:
                self._last_faiss_scores = match_scores
                # Also print for immediate debugging (show patient_friendly terms that matched)
                patient_friendly_scores = {k: v for k, v in match_scores.items() if k in matches}
                print(f"[FAISS] 🔍 Scores for '{prompt}' in {element}: {patient_friendly_scores}")
            
            return matches
        except Exception as e:
            print(f"[FAISS] ⚠️ Error in term matching: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def compute_semantic_similarity(self, patient_text: str, guideline_text: str, 
                                   condition_name: str, organ_system: str = None, 
                                   oldcarts_element: str = None, structured_oldcarts: dict = None,
                                   pre_normalized_text: str = None, precomputed_similarity: float = None,
                                   active_condition_names: set = None) -> Dict[str, Any]:
        """
        Simple semantic similarity match for all OLDCARTS elements (EXCEPT location)
        
        Algorithm:
        - User response → semantically compare to guideline terms → return highest similarity score
        - No synonym normalization (embedding model handles this)
        - No word match boost (pure semantic similarity)
        
        Location uses separate score_location_answer() function due to directional component handling.
        
        Args:
            patient_text: Raw user response (e.g., "I have a tummy ache")
            guideline_text: Space-joined medical terms from guideline OLDCARTS element (e.g., "abdominal pain stomach ache")
            condition_name: Name of condition (for debugging)
            organ_system: Organ system (optional, not used)
            oldcarts_element: OLDCARTS element (optional, not used)
            structured_oldcarts: Full guideline structure (optional, not used)
            pre_normalized_text: Not used (kept for compatibility)
            precomputed_similarity: Pre-computed embedding similarity (for optimization)
            active_condition_names: Not used (kept for compatibility)
        
        Returns:
            Dict with 'similarity' (0.0-1.0) - pure semantic similarity score
        """
        # Compute semantic similarity (embeddings only)
        raw_similarity = 0.0
        embeddings = None
        
        if precomputed_similarity is not None:
            # Use pre-computed similarity from batch embeddings (optimization)
            raw_similarity = precomputed_similarity
        elif self.embedding_model:
            try:
                # Encode patient text and guideline text
                embeddings = self.embedding_model.encode([patient_text.lower(), guideline_text])
                embeddings = np.asarray(embeddings, dtype='float32')
                
                # Cosine similarity
                raw_similarity = float(np.dot(embeddings[0], embeddings[1]) / 
                                      (np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])))
            except Exception as e:
                pass
        
        # Clamp to [0, 1]
        raw_similarity = max(0.0, min(1.0, raw_similarity))
        
        # Apply ML enhancement if available
        final_similarity = raw_similarity
        if self.ml_enhancer and oldcarts_element:
            try:
                # Get category for ML enhancement
                category = self.active_category or 'default'
                
                # Use ML to improve similarity score
                patient_embedding = embeddings[0] if embeddings is not None else None
                final_similarity = self.ml_enhancer.improve_similarity_score(
                    raw_similarity, patient_text, guideline_text,
                    oldcarts_element, category, patient_embedding
                )
            except Exception as e:
                print(f"[MedicalRules] ⚠️ ML enhancement error: {e}")
                final_similarity = raw_similarity
        
        return {
            'similarity': final_similarity,
            'raw_similarity': raw_similarity,  # Base similarity before ML enhancement
            'word_match_boost': 0.0,  # Not used in simplified version
            'normalized_text': patient_text.lower(),  # Not actually normalized (kept for compatibility)
            'method': 'semantic_similarity_ml' if self.ml_enhancer else 'semantic_similarity'
        }
    
    # REMOVED: _normalize_with_synonyms - synonym files no longer used, FAISS handles semantic matching directly
    
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
        
        # STEP 2: Word matching for non-location elements
        # NOTE: Location element uses separate algorithm (score_location_answer), not semantic similarity
        # So we don't need anatomical mismatch detection here - that's handled in location-specific code
        
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
                # NOTE: Location element uses separate algorithm, so no anatomical mismatch check here
                # Good match - boost based on number
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
    
    # ============================================================================
    # Section 4: Location Processing (special handling)
    # ============================================================================
    
    def filter_guidelines_by_location(self, patient_answer: str, guidelines: List[Dict], 
                                     organ_system: str) -> List[Dict]:
        """
        SEPARATE ALGORITHM: Filter guidelines using medical_rules.json for location ONLY
        
        This is a separate algorithm from semantic similarity matching, specifically for location filtering.
        Uses FAISS + medical_rules.json anatomical_type to filter incompatible guidelines.
        
        Flow:
        1. Use FAISS to find location matches across all medical conditions
        2. Extract direction from FAISS-matched terms
        3. Apply medical_rules.json filtering using anatomical_type from guidelines:
           - right → show right_only + bilateral + midline + vague, rule out left_only
           - left → show left_only + bilateral + midline + vague, rule out right_only
           - bilateral/midline/vague → show all (compatible with any direction)
        
        Returns:
            Filtered guidelines based on anatomical compatibility using medical_rules.json
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
            # Use raw patient answer directly (no synonym normalization needed - FAISS handles semantic matching)
            patient_direction = self._extract_directional_component(patient_answer.lower(), patient_answer)
        
        if not patient_direction:
            return guidelines  # No direction found, keep all
        
        # STEP 2: Apply anatomical filtering using anatomical_type from guidelines
        filtered = []
        for guideline in guidelines:
            # Get anatomical_type directly from guideline location element
            anatomical_type = self._get_anatomical_type_from_guideline(guideline)
            
            if not anatomical_type:
                filtered.append(guideline)  # Unknown type, keep it
                continue
            
            # Map guideline anatomical_type to filtering category
            filter_category = self._map_anatomical_type_to_filter_category(anatomical_type)
            
            # UNIVERSAL filtering logic for ALL organ systems using medical_rules.json:
            # GI, CARDIO, PULMONARY, MSK, DERM, NEURO, RENAL, GU, GYN
            if filter_category == 'right_only':
                if patient_direction == 'left':
                    continue  # Rule out when patient says "left"
                # Keep: right matches right_only, bilateral, midline, vague
            elif filter_category == 'left_only':
                if patient_direction == 'right':
                    continue  # Rule out when patient says "right"
                # Keep: left matches left_only, bilateral, midline, vague
            # bilateral, midline, and vague: always keep (compatible with all directions)
            # This works for ALL organ systems: GI, CARDIO, PULMONARY, MSK, DERM, NEURO, RENAL, GU, GYN
            
            filtered.append(guideline)
        
        return filtered
    
    def score_location_answer(self, patient_text: str, guideline_text: str, 
                             condition_name: str, organ_system: str,
                             location_data: Dict, pre_normalized_text: str = None,
                             precomputed_similarity: float = None,
                             active_condition_names: set = None) -> Dict[str, Any]:
        """
        SEPARATE ALGORITHM: Score location answers using medical_rules.json (NOT semantic similarity)
        
        This is specifically for location element scoring, separate from semantic similarity matching.
        Uses FAISS + anatomical matching + word matching, but does NOT use semantic similarity matching.
        
        Flow:
        1. Raw semantic similarity (embeddings)
        2. Normalization (with synonyms)
        3. Location-specific word matching (using medical_rules.json for anatomical compatibility)
        
        Returns:
            Dictionary with similarity, raw_similarity, word_match_boost, normalized_text
        """
        # STEP 1: Raw semantic similarity
        raw_similarity = 0.0
        if precomputed_similarity is not None:
            raw_similarity = precomputed_similarity
        elif self.embedding_model:
            try:
                embeddings = self.embedding_model.encode([patient_text.lower(), guideline_text])
                embeddings = np.asarray(embeddings, dtype='float32')
                raw_similarity = float(np.dot(embeddings[0], embeddings[1]) / 
                                      (np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])))
            except Exception as e:
                pass
        
        # STEP 2: Normalization (simplified - no synonym files needed)
        if pre_normalized_text:
            normalized_text = pre_normalized_text
        else:
            normalized_text = patient_text.lower()  # Simple lowercase normalization
        
        # STEP 3: Location-specific word matching (using medical_rules.json)
        word_match_boost = self._compute_location_word_match(
            patient_text, normalized_text, location_data, organ_system,
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
            'method': 'location_specific'
        }
    
    def _compute_location_word_match(self, patient_text: str, normalized_text: str,
                                    location_data: Dict, organ_system: str,
                                    condition_name: str, active_condition_names: set = None) -> float:
        """
        Location-specific word matching using medical_rules.json for anatomical compatibility.
        This is separate from semantic similarity matching.
        """
        if not location_data:
            return 0.0
        
        includes_terms = location_data.get('includes', [])
        excludes_terms = location_data.get('excludes', [])
        
        normalized_lower = normalized_text.lower()
        patient_lower = patient_text.lower()
        includes_lower = self._normalize_term_list(includes_terms)
        excludes_lower = self._normalize_term_list(excludes_terms)
        
        # STEP 1: Check excludes first (immediate penalty)
        for term in excludes_lower:
            if (term in normalized_lower or normalized_lower in term or
                term in patient_lower or patient_lower in term):
                return -0.3  # Penalty
        
        # STEP 2: Extract anatomical components and check for matches/mismatches
        patient_components = self._extract_anatomical_components(normalized_text)
        
        # Check for anatomical matches and mismatches using medical_rules.json
        anatomically_specific_terms = []
        all_specific_mismatched = True
        has_matching_term = False
        
        for term in includes_lower:
            condition_components = self._extract_anatomical_components(term)
            
            # If term has anatomical components, check against patient's components
            if condition_components:
                anatomically_specific_terms.append(term)
                if patient_components:
                    if self._are_anatomical_opposites(patient_components, condition_components):
                        # This anatomically-specific term is a mismatch
                        continue
                    else:
                        # Found at least one non-mismatched anatomically-specific term
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
        if anatomically_specific_terms and all_specific_mismatched and not has_matching_term:
            return -0.3  # Penalty for anatomical mismatch
        
        # STEP 3: FAISS-based term matching
        if 'location' in self.term_embeddings:
            all_faiss_matches = self.find_matching_terms_faiss(
                patient_text, 'location', threshold=0.7, active_condition_names=active_condition_names
            )
            
            # Check excludes
            matching_excludes = [term for term in all_faiss_matches if term.lower() in excludes_lower]
            if matching_excludes:
                return -0.3
            
            # Check includes
            matching_includes = [term for term in all_faiss_matches if term.lower() in includes_lower]
            if matching_includes:
                # Check for anatomical mismatches in FAISS matches
                if patient_components:
                    all_faiss_mismatched = True
                    valid_matches = []
                    
                    for matched_term in matching_includes:
                        condition_components = self._extract_anatomical_components(matched_term.lower())
                        if condition_components:
                            if self._are_anatomical_opposites(patient_components, condition_components):
                                continue  # Mismatch - skip
                            else:
                                all_faiss_mismatched = False
                                valid_matches.append(matched_term)
                        else:
                            all_faiss_mismatched = False
                            valid_matches.append(matched_term)
                    
                    if all_faiss_mismatched and len(valid_matches) == 0:
                        return -0.3  # Penalty for anatomical mismatch
                    
                    matching_includes = valid_matches
                
                # Good match - boost based on number
                if matching_includes:
                    match_boost = min(0.1 * len(matching_includes), 0.4)
                    return match_boost
        
        # STEP 4: Fallback exact matching
        for term in includes_lower:
            condition_components = self._extract_anatomical_components(term)
            
            # If both have anatomical components, check if they're opposites
            if patient_components and condition_components:
                if self._are_anatomical_opposites(patient_components, condition_components):
                    continue  # Anatomical mismatch - skip
            
            # Exact or substring match (and not opposite)
            if (term == normalized_lower or term in normalized_lower or 
                normalized_lower in term or term in patient_lower or 
                patient_lower in term):
                # Exact match gets full boost
                if term == normalized_lower or normalized_lower == term:
                    return 0.5
                # Substring match gets partial boost
                return 0.3
        
        return 0.0  # No match found
    
    # ============================================================================
    # Section 5: Utilities
    # ============================================================================
    
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
    
    def _get_anatomical_type_from_guideline(self, guideline: Dict) -> Optional[str]:
        """Get anatomical_type directly from guideline location element"""
        structured_oldcarts = guideline.get('data', {}).get('key_features', {}).get('structured_oldcarts', {})
        if not structured_oldcarts:
            structured_oldcarts = guideline.get('key_features', {}).get('structured_oldcarts', {})
        
        location_data = structured_oldcarts.get('location')
        if isinstance(location_data, dict):
            return location_data.get('anatomical_type')
        return None
    
    def _map_anatomical_type_to_components(self, anatomical_type: str) -> Dict[str, Any]:
        """
        Map guideline anatomical_type values to component dictionary using existing medical_rules.json structures.
        Uses _extract_anatomical_components() for parsing, then adds bilateral/vague flags from directional_keywords.
        Returns dict with keys: 'vertical', 'horizontal', 'bilateral', 'vague'
        """
        if not anatomical_type or not self.medical_rules:
            return {}
        
        # Use existing _extract_anatomical_components() to parse anatomical_type as text
        # This handles quadrant_patterns (right_upper, left_lower) and directional_keywords (upper, lower, right, left)
        components = self._extract_anatomical_components(anatomical_type)
        
        # Check for bilateral/vague flags from directional_keywords
        anatomical = self.medical_rules.get('anatomical_components', {})
        directional_keywords = anatomical.get('directional_keywords', {})
        
        # Check bilateral keywords
        bilateral_keywords = directional_keywords.get('bilateral', {})
        if bilateral_keywords:
            for direction, keywords in bilateral_keywords.items():
                if any(keyword in anatomical_type.lower() for keyword in keywords):
                    components['bilateral'] = True
                    # Bilateral has no horizontal direction
                    if 'horizontal' in components:
                        del components['horizontal']
        
        # Check vague keywords
        vague_keywords = directional_keywords.get('vague', {})
        if vague_keywords:
            for direction, keywords in vague_keywords.items():
                if any(keyword in anatomical_type.lower() for keyword in keywords):
                    components['vague'] = True
                    # Vague has no directional components
                    if 'horizontal' in components:
                        del components['horizontal']
                    if 'vertical' in components:
                        del components['vertical']
                    if 'quadrant' in components:
                        del components['quadrant']
        
        return components
    
    def _map_anatomical_type_to_filter_category(self, anatomical_type: str) -> str:
        """
        Map guideline anatomical_type values to filter categories.
        Guideline values: right_lower, right_upper, left_lower, left_upper, bilateral, vague, upper, lower
        Filter categories: right_only, left_only, bilateral, vague
        Note: "midline" is treated as "bilateral" (compatible with left or right)
        Note: "vague" means truly diffuse with no directional component (compatible with any location)
        """
        if not anatomical_type:
            return 'vague'  # Default: keep all (truly diffuse, no directional component)
        
        anatomical_type_lower = anatomical_type.lower()
        
        # Map quadrant-based to right_only/left_only
        if anatomical_type_lower in ['right_lower', 'right_upper', 'right']:
            return 'right_only'
        elif anatomical_type_lower in ['left_lower', 'left_upper', 'left']:
            return 'left_only'
        elif anatomical_type_lower in ['vague', 'diffuse']:
            # Vague: truly diffuse, no directional component at all
            return 'vague'
        elif anatomical_type_lower in ['bilateral', 'both', 'midline', 'center', 'central']:
            # Bilateral and midline: compatible with left or right (but still has some directional meaning)
            return 'bilateral'
        elif anatomical_type_lower in ['upper', 'lower']:
            # Upper/lower are ambiguous - treat as vague (no horizontal component)
            return 'vague'
        else:
            # Unknown - default to vague (keep all)
            return 'vague'
    
    def _extract_directional_component(self, normalized_text: str, raw_text: str = None) -> Optional[str]:
        """
        Extract directional component using medical_rules.json structure
        Checks normalized category name against conditions in medical_rules.json
        """
        if not self.medical_rules or 'anatomical_components' not in self.medical_rules:
            return None
        
        text_to_check = normalized_text.lower()
        if raw_text:
            text_to_check += " " + raw_text.lower()
        
        anatomical = self.medical_rules.get('anatomical_components', {})
        
        # Check horizontal direction from medical_rules.json
        horizontal_keywords = anatomical.get('directional_keywords', {}).get('horizontal', {})
        for direction, keywords in horizontal_keywords.items():
            if any(word in text_to_check for word in keywords):
                return direction
        
        # Check quadrant patterns (includes RUQ, RLQ, LUQ, LLQ)
        quadrant_patterns = anatomical.get('quadrant_patterns', {})
        for quadrant, patterns in quadrant_patterns.items():
            if any(pattern in text_to_check for pattern in patterns):
                # Extract horizontal from quadrant (e.g., "right_upper" -> "right")
                if quadrant.startswith('right'):
                    return 'right'
                elif quadrant.startswith('left'):
                    return 'left'
        
        # Check for bilateral from medical_rules.json
        bilateral_keywords = anatomical.get('directional_keywords', {}).get('bilateral', {})
        for direction, keywords in bilateral_keywords.items():
            if any(word in text_to_check for word in keywords):
                return direction
        
        # Check anatomical_regions for midline indicators
        anatomical_regions = anatomical.get('anatomical_regions', {})
        for region_name, region_data in anatomical_regions.items():
            if region_name in text_to_check:
                # If region has no horizontal component, it's midline
                if region_data.get('horizontal') is None:
                    return 'midline'
        
        # Check for midline/center keywords from medical_rules.json
        midline_keywords = anatomical.get('directional_keywords', {}).get('midline', {})
        for direction, keywords in midline_keywords.items():
            if any(word in text_to_check for word in keywords):
                return direction
        
        return None
    
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
        
        # Bilateral indicators - check from medical_rules.json
        if self.medical_rules and 'anatomical_components' in self.medical_rules:
            anatomical = self.medical_rules.get('anatomical_components', {})
            bilateral_keywords = anatomical.get('directional_keywords', {}).get('bilateral', {})
            for direction, keywords in bilateral_keywords.items():
                if any(word in combined_text for word in keywords):
                    return direction
        
        # Midline indicators - check anatomical_regions from medical_rules.json
        if self.medical_rules and 'anatomical_components' in self.medical_rules:
            anatomical = self.medical_rules.get('anatomical_components', {})
            anatomical_regions = anatomical.get('anatomical_regions', {})
            for region_name, region_data in anatomical_regions.items():
                if region_name in combined_text:
                    # If region has no horizontal component, it's midline
                    if region_data.get('horizontal') is None:
                        return 'midline'
            
            # Check for midline/center keywords from medical_rules.json
            midline_keywords = anatomical.get('directional_keywords', {}).get('midline', {})
            for direction, keywords in midline_keywords.items():
                if any(word in combined_text for word in keywords):
                    return direction
        
        return None
    
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
        
        # Extract vertical components (direct or from quadrant)
        vertical1 = components1.get('vertical')
        if not vertical1 and 'quadrant' in components1:
            parts = components1['quadrant'].split('_')
            if len(parts) >= 2 and parts[1] in ['upper', 'lower']:
                vertical1 = parts[1]
            elif len(parts) >= 3 and parts[1] in ['upper', 'lower']:
                vertical1 = parts[1]
        
        vertical2 = components2.get('vertical')
        if not vertical2 and 'quadrant' in components2:
            parts = components2['quadrant'].split('_')
            if len(parts) >= 2 and parts[1] in ['upper', 'lower']:
                vertical2 = parts[1]
            elif len(parts) >= 3 and parts[1] in ['upper', 'lower']:
                vertical2 = parts[1]
        
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
    
    # ============================================================================
    # Section 6: ML Learning and Training
    # ============================================================================
    
    def record_match_for_learning(self, patient_text: str, guideline_text: str, element: str,
                                  similarity: float, was_successful: bool, embedding: Optional[np.ndarray] = None):
        """
        Record a match for ML learning
        
        Args:
            patient_text: Patient response text
            guideline_text: Guideline term text
            element: OLDCARTS element type
            similarity: Similarity score
            was_successful: Whether this was a successful match
            embedding: Optional embedding vector
        """
        if not self.ml_enhancer:
            return
        
        try:
            category = self.active_category or 'default'
            if was_successful:
                self.ml_enhancer.record_successful_match(
                    patient_text, guideline_text, element, category, similarity, embedding
                )
            else:
                self.ml_enhancer.record_failed_match(
                    patient_text, guideline_text, element, category, similarity, embedding
                )
        except Exception as e:
            print(f"[MedicalRules] ⚠️ Error recording match for learning: {e}")
    
    def train_ml_models(self, epochs: int = 10):
        """
        Train ML models on collected data
        
        Args:
            epochs: Number of training epochs
        """
        if not self.ml_enhancer:
            print("[MedicalRules] ⚠️ ML enhancer not available")
            return
        
        try:
            self.ml_enhancer.train_similarity_model(epochs=epochs)
            self.ml_enhancer.save_all()
            print("[MedicalRules] ✅ ML model training complete")
        except Exception as e:
            print(f"[MedicalRules] ⚠️ Error training ML models: {e}")
    
    def get_ml_stats(self) -> Dict[str, Any]:
        """Get ML enhancement statistics"""
        if not self.ml_enhancer:
            return {}
        
        return self.ml_enhancer.get_stats()
    
    def adjust_threshold_for_element(self, element: str, success_rate: float):
        """
        Adjust threshold for an element based on success rate
        
        Args:
            element: OLDCARTS element type
            success_rate: Success rate (0-1) for this element
        """
        if not self.ml_enhancer:
            return
        
        try:
            category = self.active_category or 'default'
            self.ml_enhancer.adjust_threshold(element, category, success_rate)
        except Exception as e:
            print(f"[MedicalRules] ⚠️ Error adjusting threshold: {e}")
   

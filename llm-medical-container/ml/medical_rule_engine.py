#!/usr/bin/env python3
"""
Medical Rule Engine
Combines hardcoded rules with ML predictions for anatomical relationships
"""

import json
import joblib
import re
import os
import numpy as np
from typing import Dict, Any, List

# Optional ML trainer import
try:
    from .location_ml_trainer import LocationMLTrainer
    ML_TRAINER_AVAILABLE = True
except ImportError:
    try:
        from location_ml_trainer import LocationMLTrainer
        ML_TRAINER_AVAILABLE = True
    except ImportError:
        ML_TRAINER_AVAILABLE = False
        LocationMLTrainer = None

class MedicalRuleEngine:
    """
    Medical rule engine for anatomical relationships
    Combines hardcoded rules with ML predictions
    """
    
    def _get_condition_anatomical_type(self, condition_name: str, organ_system: str) -> str:
        """Get anatomical type from medical_rules.json for a condition"""
        if not self.medical_rules:
            print(f"[MedicalRules] ⚠️ No medical rules loaded")
            return None
            
        if organ_system not in self.medical_rules:
            print(f"[MedicalRules] ⚠️ Organ system '{organ_system}' not found in medical rules")
            return None
        
        organ_rules = self.medical_rules[organ_system]
        print(f"[MedicalRules] 🔍 Looking for '{condition_name}' in {organ_system} rules")
        print(f"[MedicalRules] 📋 Available anatomical types: {list(organ_rules.keys())}")
        
        for anatomical_type, condition_list in organ_rules.items():
            if condition_name in condition_list:
                print(f"[MedicalRules] ✅ Found in {anatomical_type}: {condition_name}")
                return anatomical_type
        
        print(f"[MedicalRules] ❌ '{condition_name}' not found in medical rules")
        print(f"[MedicalRules] 📋 Condition list sample (first 3): {[c[:30] for c in condition_list[:3]] if condition_list else 'empty'}")
        return None
    
    def _check_anatomical_compatibility(self, patient_text: str, exclude_term: str, condition_name: str, organ_system: str) -> bool:
        """
        Check if patient text and exclude term are anatomically compatible
        Returns True if they are compatible (no penalty), False if incompatible (penalty applies)
        
        Uses medical_rules.json to determine anatomical types:
        - right_only condition + patient says "left" → incompatible (but opposite sides already handled)
        - left_only condition + patient says "right" → incompatible
        """
        patient_lower = patient_text.lower()
        exclude_lower = exclude_term.lower()
        
        # Extract sides from both
        patient_has_right = 'right' in patient_lower
        patient_has_left = 'left' in patient_lower
        exclude_has_right = 'right' in exclude_lower
        exclude_has_left = 'left' in exclude_lower
        
        # Get condition's anatomical type
        anatomical_type = self._get_condition_anatomical_type(condition_name, organ_system)
        
        print(f"[WordMatch] 🔍 Anatomical check: condition='{condition_name}', type='{anatomical_type}'")
        print(f"[WordMatch]   Patient: right={patient_has_right}, left={patient_has_left}")
        print(f"[WordMatch]   Exclude: right={exclude_has_right}, left={exclude_has_left}")
        
        # If condition is right_only and patient says left, they're talking about opposite side
        # This shouldn't match anyway (opposite side), so skip penalty
        if anatomical_type == 'right_only' and patient_has_left and exclude_has_left:
            print(f"[WordMatch] ⏭️  SKIP: Condition is right_only, patient/exclude both mention left")
            return True
        
        if anatomical_type == 'left_only' and patient_has_right and exclude_has_right:
            print(f"[WordMatch] ⏭️  SKIP: Condition is left_only, patient/exclude both mention right")
            return True
        
        return False  # Compatible for semantic check
    
    def __init__(self, ml_model_path: str = "ml/location_ml_model.pkl", embedding_model=None):
        self.ml_model = None
        self.ml_trainer = None
        self.embedding_model = embedding_model  # Store embedding model for semantic similarity
        
        # Initialize ML trainer if available
        if ML_TRAINER_AVAILABLE:
            try:
                self.ml_trainer = LocationMLTrainer()
                # Load ML model if available
                try:
                    self.ml_model = self.ml_trainer.load_model(ml_model_path)
                    print(f"✅ ML model loaded from {ml_model_path}")
                except FileNotFoundError:
                    print(f"⚠️ ML model not found at {ml_model_path}, using hardcoded rules only")
            except Exception as e:
                print(f"⚠️ ML trainer initialization failed: {e}, using hardcoded rules only")
                self.ml_trainer = None
        else:
            print("⚠️ Medical Rule Engine not available: No module named 'location_ml_trainer'")
        
        # Load hardcoded medical rules
        self.medical_rules = self._load_medical_rules()
    
    def _load_medical_rules(self) -> Dict:
        """
        Load medical rules from JSON file
        """
        import json
        from pathlib import Path
        
        # Get path to medical_rules.json relative to this file
        current_file = Path(__file__).resolve()
        config_dir = current_file.parent.parent / 'config'
        json_path = config_dir / 'medical_rules.json'
        
        # Debug path resolution
        print(f"[MedicalRules] 🔍 Current file: {current_file}")
        print(f"[MedicalRules] 🔍 Config dir: {config_dir}")
        print(f"[MedicalRules] 🔍 JSON path: {json_path}")
        print(f"[MedicalRules] 🔍 Exists: {json_path.exists()}")
        print(f"[MedicalRules] 🔍 Config dir exists: {config_dir.exists()}")
        if config_dir.exists():
            print(f"[MedicalRules] 🔍 Config dir contents: {list(config_dir.iterdir())}")
        
        try:
            with open(json_path, 'r') as f:
                rules = json.load(f)
            print(f"[MedicalRules] ✅ Loaded medical rules from {json_path}")
            print(f"[MedicalRules] 📊 Organ systems: {len(rules)}")
            return rules
        except FileNotFoundError:
            print(f"[MedicalRules] ⚠️ JSON file not found at {json_path}")
            print(f"[MedicalRules] Using empty rules dict")
            return {}
        except json.JSONDecodeError as e:
            print(f"[MedicalRules] ❌ Error parsing JSON: {e}")
            print(f"[MedicalRules] Using empty rules dict")
            return {}
        except Exception as e:
            print(f"[MedicalRules] ❌ Unexpected error loading rules: {e}")
            print(f"[MedicalRules] Using empty rules dict")
            return {}
    
    def compute_unified_similarity(self, patient_text: str, guideline_text: str, 
                                   condition_name: str, organ_system: str = None, 
                                   oldcarts_element: str = None, structured_oldcarts: dict = None,
                                   return_word_match_only: bool = False, pre_normalized_text: str = None) -> Dict[str, Any]:
        """
        UNIFIED similarity computation used in ALL cases (scoring, clarification, etc.)
        
        Flow:
        1. Raw semantic similarity (embeddings)
        2. Normalization (with semantic fallback if exact match fails)
        3. Word match boost (normalized text vs structured_oldcarts)
        
        Args:
            patient_text: Raw patient answer
            guideline_text: Guideline text for semantic similarity
            condition_name: Condition name
            organ_system: Organ system (e.g., "gi")
            oldcarts_element: OLDCARTS element (e.g., "location")
            structured_oldcarts: Structured OLDCARTS dict
            return_word_match_only: If True, only return word_match_boost (for clarification matching)
            pre_normalized_text: If provided, skip normalization step (optimization)
        
        Returns:
            Dict with similarity, word_match_boost, normalized_text, etc.
        """
        # STEP 1: Raw semantic similarity (only if needed for scoring)
        raw_similarity = 0.0
        if not return_word_match_only and self.embedding_model:
            try:
                embeddings = self.embedding_model.encode([patient_text.lower(), guideline_text])
                raw_similarity = float(np.dot(embeddings[0], embeddings[1]) / (np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])))
            except Exception as e:
                print(f"[Unified] ⚠️ Error computing raw semantic similarity: {e}")
        
        # STEP 2: Normalization (with semantic fallback) - skip if pre-normalized provided
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
                        print(f"[Unified] ⚠️ Error in normalization: {e}")
        
        # STEP 3: Word match boost (normalized text vs structured_oldcarts)
        # Pass normalized_text to avoid duplicate normalization inside _compute_word_match_boost
        word_match_boost = 0.0
        if structured_oldcarts and oldcarts_element:
            word_match_boost = self._compute_word_match_boost(
                patient_text,  # Raw text (for backward compatibility)
                guideline_text,
                organ_system,
                oldcarts_element,
                structured_oldcarts,
                condition_name,
                pre_normalized_text=normalized_text  # Pass pre-normalized text to avoid duplicate work
            )
        
        # STEP 4: Combine results
        if return_word_match_only:
            # For clarification matching, only return word_match_boost
            return {
                'word_match_boost': word_match_boost,
                'normalized_text': normalized_text,
                'has_match': word_match_boost > 0
            }
        
        # For scoring, combine raw similarity + word match boost
        final_similarity = raw_similarity + word_match_boost
        final_similarity = max(0.0, min(1.0, final_similarity))  # Clamp to 0-1
        
        return {
            'similarity': final_similarity,
            'raw_similarity': raw_similarity,
            'word_match_boost': word_match_boost,
            'normalized_text': normalized_text,
            'method': 'unified_similarity',
            'confidence': 'high' if final_similarity >= 0.7 else 'medium' if final_similarity >= 0.3 else 'low',
            'reasoning': f'Raw semantic ({raw_similarity:.3f}) + word match boost ({word_match_boost:.3f}) = {final_similarity:.3f}',
            'anatomical_type': 'unknown'
        }
    
    def get_enhanced_similarity(self, patient_text: str, guideline_text: str, 
                              condition_name: str, organ_system: str = None, oldcarts_element: str = None, structured_oldcarts: dict = None) -> Dict[str, Any]:
        """
        Enhanced similarity - now uses unified computation
        """
        return self.compute_unified_similarity(
            patient_text, guideline_text, condition_name,
            organ_system, oldcarts_element, structured_oldcarts,
            return_word_match_only=False
        )
    
    def _compute_embedding_similarity(self, patient_text: str, guideline_text: str, organ_system: str = None, oldcarts_element: str = None, structured_oldcarts: dict = None, condition_name: str = None) -> Dict[str, Any]:
        """
        DEPRECATED: Use compute_unified_similarity() instead
        Kept for backward compatibility - internally calls unified function
        """
        # Delegate to unified function
        result = self.compute_unified_similarity(
            patient_text, guideline_text, condition_name or "",
            organ_system, oldcarts_element, structured_oldcarts,
            return_word_match_only=False
        )
        
        # Return in expected format for backward compatibility
        return {
            'similarity': result['similarity'],
            'method': result['method'],
            'confidence': result['confidence'],
            'reasoning': result['reasoning'],
            'anatomical_type': result['anatomical_type'],
            'semantic_score': result['raw_similarity']
        }
    
    def _compute_word_match_boost(self, patient_text: str, guideline_text: str, organ_system: str = None, oldcarts_element: str = None, structured_oldcarts: dict = None, condition_name: str = None, pre_normalized_text: str = None) -> float:
        """
        Compute word-based similarity boost for direct matches
        
        Normalizes patient answer using appropriate synonym file and section,
        then checks if normalized answer appears in structured OLDCARTS data.
        
        Args:
            pre_normalized_text: If provided, use this instead of normalizing again (optimization)
        
        Example: "right side of my belly" -> normalized to "right side" -> match in includes array
        """
        # Skip normalization for demographics
        if oldcarts_element in ['age', 'sex', 'biological_sex']:
            return 0.0
        
        if not organ_system or not oldcarts_element or not structured_oldcarts:
            # Fallback to simple word matching if no context provided
            return self._simple_word_match_boost(patient_text, guideline_text)
        
        try:
            # Use pre-normalized text if provided (optimization to avoid duplicate normalization)
            if pre_normalized_text:
                normalized_answer = pre_normalized_text
            else:
                # Load appropriate synonym file and normalize
                synonym_file = f"synonyms/{organ_system.lower()}_synonyms_oldcarts.json"
                synonym_path = os.path.join(os.path.dirname(__file__), '..', synonym_file)
                
                if not os.path.exists(synonym_path):
                    print(f"[WordMatch] ⚠️ Synonym file not found: {synonym_path}")
                    return self._simple_word_match_boost(patient_text, guideline_text)
                
                with open(synonym_path, 'r') as f:
                    synonyms = json.load(f)
                
                normalized_answer = self._normalize_with_synonyms(patient_text, synonyms, oldcarts_element)
            
            print(f"[WordMatch] 🔄 Normalization: '{patient_text}' → '{normalized_answer}'")
            
            # Load synonyms if needed (for excludes semantic check - only if not pre-normalized)
            # Synonyms already loaded above if not pre-normalized, so only load if pre-normalized
            if pre_normalized_text:
                synonym_file = f"synonyms/{organ_system.lower()}_synonyms_oldcarts.json"
                synonym_path = os.path.join(os.path.dirname(__file__), '..', synonym_file)
                if os.path.exists(synonym_path):
                    with open(synonym_path, 'r') as f:
                        synonyms = json.load(f)
            
            if oldcarts_element not in structured_oldcarts:
                print(f"[WordMatch] ⚠️ No structured data for {oldcarts_element}")
                return self._simple_word_match_boost(patient_text, guideline_text)
            
            element_data = structured_oldcarts[oldcarts_element]
            if not isinstance(element_data, dict):
                print(f"[WordMatch] ⚠️ Invalid structured data format for {oldcarts_element}")
                return self._simple_word_match_boost(patient_text, guideline_text)
            
            includes_terms = element_data.get('includes', [])
            excludes_terms = element_data.get('excludes', [])
            
            print(f"[WordMatch] 📋 Includes terms: {includes_terms}")
            print(f"[WordMatch] 📋 Excludes terms: {excludes_terms}")
            
            # normalized_answer already set above (from pre_normalized_text or normalization)
            normalized_lower = normalized_answer.lower().strip()
            normalized_words = normalized_answer.lower().replace('_', ' ').split()
            
            # Check excludes FIRST using normalized text (consistent with includes)
            if excludes_terms:
                for term in excludes_terms:
                    term_lower = term.lower()
                    
                    # Check 1: Exact or substring match using normalized text
                    if normalized_lower == term_lower or term_lower in normalized_lower or normalized_lower in term_lower:
                        print(f"[WordMatch] ⛔ EXCLUDE MATCH (exact/substring): '{normalized_answer}' matches exclude term '{term}'")
                        print(f"[WordMatch]   Applying penalty: -0.3")
                        return -0.3  # PENALTY for excluded term
                    
                    # Check 2: Word overlap - see if any word from normalized answer appears in exclude term
                    term_words = set(term_lower.split())
                    normalized_words_set = set(normalized_words)
                    matching_words = normalized_words_set.intersection(term_words)
                    
                    if len(matching_words) >= 1:
                        print(f"[WordMatch] ⛔ EXCLUDE MATCH (word overlap): '{normalized_answer}' has matching words '{matching_words}' with exclude term '{term}'")
                        print(f"[WordMatch]   Applying penalty: -0.3")
                        return -0.3  # PENALTY for excluded term
                    
                    # Check 3: Semantic similarity with embedding model (use medical_rules.json for compatibility)
                    if self.embedding_model and condition_name and organ_system:
                        try:
                            # Check anatomical compatibility using medical_rules.json
                            is_compatible = self._check_anatomical_compatibility(patient_text, term, condition_name, organ_system)
                            if is_compatible:
                                print(f"[WordMatch] ⏭️  SKIP: Anatomically incompatible (using medical_rules.json) - no penalty")
                                continue
                            
                            # Proceed with semantic similarity check (using normalized text for consistency)
                            embeddings = self.embedding_model.encode([normalized_answer.lower(), term])
                            similarity = float(np.dot(embeddings[0], embeddings[1]) / (np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])))
                            
                            print(f"[WordMatch] 🔍 Exclude check (semantic): '{normalized_answer}' vs '{term}' = {similarity:.2f}")
                            
                            # Penalize if semantic similarity exceeds threshold
                            if similarity > 0.4:
                                print(f"[WordMatch] ⛔ EXCLUDE MATCH (semantic): '{normalized_answer}' has similarity ({similarity:.2f}) with exclude term '{term}'")
                                print(f"[WordMatch]   Applying penalty: -0.3")
                                return -0.3  # PENALTY for excluded term
                        except Exception as e:
                            print(f"[WordMatch] ⚠️ Error computing exclude similarity: {e}")
            
            # NOW check includes using normalized text
            
            # Check for match in includes terms (using normalized text)
            found_match = False
            for term in includes_terms:
                term_lower = term.lower()
                
                # Exact match: normalized answer matches term exactly or as substring
                if normalized_lower == term_lower or normalized_lower in term_lower or term_lower in normalized_lower:
                    print(f"[WordMatch] ✅ Exact match found: '{normalized_answer}' ↔ '{term}'")
                    print(f"[WordMatch]   Progressive boost: +0.5 (exact match)")
                    return 0.5  # Highest boost for exact match
                
                # Check word overlap: see if any word from normalized answer appears in term
                term_words = set(term_lower.split())
                normalized_words_set = set(normalized_words)
                matching_words = normalized_words_set.intersection(term_words)
                
                if len(matching_words) >= 1:  # Any word match
                    # Progressive boost based on number of matching words
                    # 1 word = 0.2, 2 words = 0.3, 3+ words = 0.4
                    boost = min(0.2 * len(matching_words), 0.4)
                    print(f"[WordMatch] ✅ Word match ({len(matching_words)} words): '{normalized_answer}' ↔ '{term}'")
                    print(f"[WordMatch]   Matching words: {matching_words}")
                    print(f"[WordMatch]   Progressive boost: +{boost:.2f}")
                    return boost
            
            # STEP 3: Fallback to semantic matching if no exact/word match found
            # This handles cases where normalized term (e.g., "ruq_pain") semantically matches includes term (e.g., "right upper quadrant")
            if self.embedding_model and not found_match and includes_terms:
                best_similarity = 0.0
                best_term = None
                semantic_threshold = 0.65
                
                try:
                    # Batch encode: normalized answer + all includes terms (more efficient)
                    all_texts = [normalized_lower] + [term.lower() for term in includes_terms]
                    embeddings = self.embedding_model.encode(all_texts)
                    normalized_emb = embeddings[0]
                    
                    # Compare against all includes terms semantically
                    for i, term in enumerate(includes_terms):
                        term_emb = embeddings[i + 1]
                        similarity = float(np.dot(normalized_emb, term_emb) / (np.linalg.norm(normalized_emb) * np.linalg.norm(term_emb)))
                        
                        if similarity > best_similarity:
                            best_similarity = similarity
                            best_term = term
                    
                    if best_similarity >= semantic_threshold:
                        # Progressive boost based on similarity
                        boost = min(0.2 + (best_similarity - semantic_threshold) * 0.5, 0.4)
                        print(f"[WordMatch] ✅ Semantic match found: '{normalized_answer}' ↔ '{best_term}' (similarity: {best_similarity:.3f})")
                        print(f"[WordMatch]   Progressive boost: +{boost:.2f}")
                        return boost
                except Exception as e:
                    print(f"[WordMatch] ⚠️ Error in semantic matching: {e}")
            
            print(f"[WordMatch] ❌ No word matches found")
            print(f"[WordMatch]   Normalized: '{normalized_answer}'")
            print(f"[WordMatch]   Available includes terms: {includes_terms}")
            return 0.0
            
        except Exception as e:
            print(f"[WordMatch] ❌ Error in synonym-based matching: {e}")
            return self._simple_word_match_boost(patient_text, guideline_text)
    
    def _normalize_with_synonyms(self, patient_text: str, synonyms: dict, oldcarts_element: str) -> str:
        """
        Normalize patient text using synonym file for specific OLDCARTS element
        Uses exact substring matching first, then semantic matching as fallback
        """
        if oldcarts_element not in synonyms:
            return patient_text.lower().strip()
        
        patient_lower = patient_text.lower()
        element_synonyms = synonyms[oldcarts_element]
        
        # STEP 1: Try exact substring matching first (faster)
        for standard_term, synonym_list in element_synonyms.items():
            for synonym in synonym_list:
                if synonym in patient_lower:
                    print(f"[WordMatch] 🔄 Synonym match (exact): '{synonym}' → '{standard_term}'")
                    return standard_term
        
        # STEP 2: Fallback to semantic matching if no exact match found
        if self.embedding_model:
            best_match = None
            best_similarity = 0.0
            similarity_threshold = 0.65  # Threshold for semantic match
            
            # Collect all synonyms for semantic comparison
            all_synonyms = []
            synonym_texts = []
            for standard_term, synonym_list in element_synonyms.items():
                for synonym in synonym_list:
                    all_synonyms.append((standard_term, synonym))
                    synonym_texts.append(synonym.lower())
            
            if all_synonyms:
                try:
                    # Batch encode: patient text + all synonyms
                    all_texts = [patient_lower] + synonym_texts
                    embeddings = self.embedding_model.encode(all_texts)
                    patient_emb = embeddings[0]
                    
                    # Compare against all synonyms
                    for i, (standard_term, synonym) in enumerate(all_synonyms):
                        synonym_emb = embeddings[i + 1]
                        similarity = float(np.dot(patient_emb, synonym_emb) / (np.linalg.norm(patient_emb) * np.linalg.norm(synonym_emb)))
                        
                        if similarity > best_similarity:
                            best_similarity = similarity
                            best_match = standard_term
                    
                    if best_similarity >= similarity_threshold:
                        print(f"[WordMatch] 🔄 Synonym match (semantic): '{patient_text}' → '{best_match}' (similarity: {best_similarity:.3f})")
                        return best_match
                    else:
                        print(f"[WordMatch] ⚠️ No semantic match found (best: {best_similarity:.3f} < {similarity_threshold})")
                except Exception as e:
                    print(f"[WordMatch] ⚠️ Error in semantic normalization: {e}")
        
        # No synonym match found (exact or semantic), return original text
        print(f"[WordMatch] ⚠️ No synonym match found for: '{patient_text}'")
        return patient_text.lower().strip()
    
    def _simple_word_match_boost(self, patient_text: str, guideline_text: str) -> float:
        """Simple word matching fallback without synonym normalization"""
        patient_lower = patient_text.lower().strip()
        guideline_lower = guideline_text.lower()
        
        # Direct word match boost
        if patient_lower in guideline_lower:
            return 0.3  # Strong boost for exact match
        
        # Check for partial matches (patient words in guideline)
        patient_words = patient_lower.split()
        guideline_words = guideline_lower.split()
        
        # Count how many patient words appear in guideline
        matches = sum(1 for word in patient_words if word in guideline_words)
        total_words = len(patient_words)
        
        if total_words > 0:
            match_ratio = matches / total_words
            if match_ratio >= 0.5:  # At least half the words match
                return 0.2  # Medium boost for partial match
            elif match_ratio >= 0.25:  # At least a quarter match
                return 0.1  # Small boost for weak match
        
        return 0.0  # No boost
    
    def _compute_embedding_similarity(self, patient_text: str, guideline_text: str, organ_system: str = None, oldcarts_element: str = None, structured_oldcarts: dict = None, condition_name: str = None) -> Dict[str, Any]:
        """
        Compute semantic similarity using embedding model
        This provides deep semantic understanding without needing synonym lists
        
        Args:
            patient_text: Patient's natural language response
            guideline_text: Guideline text to compare against
            
        Returns:
            Dictionary with similarity score and metadata
        """
        if not self.embedding_model:
            # Fallback to simple similarity calculation
            return {
                'similarity': 0.2,  # Low similarity fallback
                'reasoning': 'No embedding model available - using fallback',
                'method': 'fallback'
            }
        
        try:
            # Encode both texts into embeddings
            # Duration uses same word-match boost logic as location (via structured_oldcarts.includes/excludes)
            embeddings = self.embedding_model.encode([patient_text, guideline_text])
            patient_emb = embeddings[0]
            guideline_emb = embeddings[1]
            
            # STEP 1: Compute raw semantic similarity first
            raw_similarity = float(np.dot(patient_emb, guideline_emb) / (np.linalg.norm(patient_emb) * np.linalg.norm(guideline_emb)))
            print(f"[Embedding] 📊 Raw semantic similarity: {raw_similarity:.4f}")
            
            # STEP 2: Normalize and check word matches for boost/penalty
            word_match_boost = self._compute_word_match_boost(patient_text, guideline_text, organ_system, oldcarts_element, structured_oldcarts, condition_name)
            boost_applied = False
            
            # Apply boost/penalty regardless of sign (positive = boost, negative = penalty)
            if word_match_boost != 0:
                raw_similarity = raw_similarity + word_match_boost
                raw_similarity = max(0.0, min(1.0, raw_similarity))  # Clamp between 0 and 1
                boost_applied = True
                if word_match_boost > 0:
                    print(f"[Embedding] 🎯 Word match boost applied: +{word_match_boost:.3f}, total: {raw_similarity:.4f}")
                else:
                    print(f"[Embedding] ⛔ Word match penalty applied: {word_match_boost:.3f}, total: {raw_similarity:.4f}")
            
            similarity = raw_similarity
            
            # Ensure similarity is between 0 and 1
            similarity = max(0.0, min(1.0, similarity))
            
            # DEBUG: Show what's being compared
            print(f"[Embedding] 🔍 Comparing:")
            print(f"[Embedding]   Patient: '{patient_text}'")
            print(f"[Embedding]   Guideline: '{guideline_text[:80]}...'")
            print(f"[Embedding]   Patient vector shape: {patient_emb.shape}, first 5 values: {patient_emb[:5]}")
            print(f"[Embedding]   Guideline vector shape: {guideline_emb.shape}, first 5 values: {guideline_emb[:5]}")
            print(f"[Embedding]   Dot product: {np.dot(patient_emb, guideline_emb):.4f}")
            print(f"[Embedding]   Patient norm: {np.linalg.norm(patient_emb):.4f}")
            print(f"[Embedding]   Guideline norm: {np.linalg.norm(guideline_emb):.4f}")
            print(f"[Embedding]   Raw similarity: {similarity:.4f}")
            
            # Determine method and confidence based on score
            if similarity >= 0.85:
                method = 'embedding_excellent_match'
                confidence = 'high'
            elif similarity >= 0.70:
                method = 'embedding_good_match'
                confidence = 'high'
            elif similarity >= 0.50:
                method = 'embedding_moderate_match'
                confidence = 'medium'
            elif similarity >= 0.30:
                method = 'embedding_weak_match'
                confidence = 'low'
            else:
                method = 'embedding_no_match'
                confidence = 'low'
            
            print(f"[Embedding]   Method: {method}, Confidence: {confidence}")
            
            return {
                'similarity': similarity,
                'method': method,
                'confidence': confidence,
                'reasoning': f'Embedding-based semantic similarity: {similarity:.3f}',
                'word_match_boost_applied': boost_applied
            }
            
        except Exception as e:
            print(f"❌ Embedding similarity failed: {e}")
            return {
                'similarity': 0.0,
                'method': 'embedding_error',
                'confidence': 'low',
                'reasoning': f'Embedding computation failed: {e}'
            }
    


    def get_semantic_similarity(self, patient_text: str, guideline_text: str) -> Dict[str, Any]:
        """
        Simple semantic similarity for trigger matching (not anatomical rules)
        This is used for matching chief complaint triggers, not anatomical location
        """
        # For trigger matching, use simple word overlap and semantic similarity
        # This is about matching "abdominal pain" to "abdominal pain", not anatomical location
        
        # Simple word overlap similarity
        patient_words = set(patient_text.lower().split())
        guideline_words = set(guideline_text.lower().split())
        
        if not patient_words or not guideline_words:
            return {
                'similarity': 0.0,
                'method': 'no_words',
                'confidence': 'low',
                'reasoning': 'No words to compare'
            }
        
        # Jaccard similarity for word overlap
        intersection = len(patient_words.intersection(guideline_words))
        union = len(patient_words.union(guideline_words))
        jaccard_similarity = intersection / union if union > 0 else 0.0
        
        # Exact match bonus
        if patient_text.lower() == guideline_text.lower():
            similarity = 1.0
            method = 'exact_match'
        # Substring match bonus
        elif patient_text.lower() in guideline_text.lower() or guideline_text.lower() in patient_text.lower():
            similarity = 0.8
            method = 'substring_match'
        # Word overlap
        elif jaccard_similarity > 0.5:
            similarity = jaccard_similarity
            method = 'word_overlap'
        # Low similarity
        else:
            similarity = jaccard_similarity * 0.5  # Penalty for low overlap
            method = 'low_overlap'
        
        return {
            'similarity': similarity,
            'method': method,
            'confidence': 'medium',
            'reasoning': f'Trigger matching: {method}'
        }

# Example usage
if __name__ == "__main__":
    engine = MedicalRuleEngine()
    
    # Test cases
    test_cases = [
        ("left sided pain", "right lower quadrant pain", "Acute Appendicitis", "GI"),
        ("right sided pain", "right lower quadrant pain", "Acute Appendicitis", "GI"),
        ("bilateral pain", "unilateral chest pain", "Pneumothorax", "CARDIO"),
        ("chest pain", "epigastric pain", "Acute Pancreatitis", "GI")
    ]
    
    print("🧪 Testing Medical Rule Engine:")
    for patient_text, guideline_text, condition, organ_system in test_cases:
        result = engine.get_enhanced_similarity(patient_text, guideline_text, condition, organ_system)
        print(f"\n📋 Test: '{patient_text}' vs '{guideline_text}' ({condition})")
        print(f"   Similarity: {result['similarity']:.3f}")
        print(f"   Method: {result['method']}")
        print(f"   Reasoning: {result['reasoning']}")
        print(f"   Anatomical Type: {result['anatomical_type']}")

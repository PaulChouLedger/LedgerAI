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
    
    def get_enhanced_similarity(self, patient_text: str, guideline_text: str, 
                              condition_name: str, organ_system: str = None, oldcarts_element: str = None, structured_oldcarts: dict = None) -> Dict[str, Any]:
        """
        Enhanced similarity with SEMANTIC SIMILARITY FIRST, anatomical rules as modifiers
        
        CRITICAL FIX: Use semantic similarity as PRIMARY scoring method,
        anatomical rules only as fallbacks or modifiers for inconclusive cases
        """
        
        # Use patient text directly - semantic similarity handles all variations naturally
        # No need for synonym normalization - embeddings understand "below ribs" = "upper quadrant" etc.
        patient_text_for_scoring = patient_text.lower()
        
        # COMPUTE SEMANTIC SIMILARITY WITH WORD-MATCH BOOST
        # Word-match boost handles ALL anatomical discrimination via structured_oldcarts:
        # - Includes terms → match → boost (+0.3, +0.2, or +0.1)
        # - Excludes terms → match → penalty (-0.3)
        # - Example: "right side" + Diverticulitis (excludes: "right side") → -0.3 penalty
        semantic_result = self._compute_embedding_similarity(patient_text_for_scoring, guideline_text, organ_system, oldcarts_element, structured_oldcarts)
        semantic_score = semantic_result['similarity']
        
        # Return semantic score as-is - word-match boost already handled discrimination
        return {
            'similarity': semantic_score,
            'method': 'semantic_similarity',
            'confidence': 'high' if semantic_score >= 0.7 else 'medium' if semantic_score >= 0.3 else 'low',
            'reasoning': f'Semantic similarity with word-match boost: {semantic_result["reasoning"]}',
            'anatomical_type': 'unknown',  # No longer needed - handled by structured_oldcarts
            'semantic_score': semantic_score
        }
    
    def _compute_word_match_boost(self, patient_text: str, guideline_text: str, organ_system: str = None, oldcarts_element: str = None, structured_oldcarts: dict = None) -> float:
        """
        Compute word-based similarity boost for direct matches
        
        Normalizes patient answer using appropriate synonym file and section,
        then checks if normalized answer appears in structured OLDCARTS data.
        
        Example: "right side of my belly" -> normalized to "right side" -> match in includes array
        """
        # Skip normalization for demographics
        if oldcarts_element in ['age', 'sex', 'biological_sex']:
            return 0.0
        
        if not organ_system or not oldcarts_element or not structured_oldcarts:
            # Fallback to simple word matching if no context provided
            return self._simple_word_match_boost(patient_text, guideline_text)
        
        try:
            # Load appropriate synonym file
            synonym_file = f"synonyms/{organ_system.lower()}_synonyms_oldcarts.json"
            synonym_path = os.path.join(os.path.dirname(__file__), '..', synonym_file)
            
            if not os.path.exists(synonym_path):
                print(f"[WordMatch] ⚠️ Synonym file not found: {synonym_path}")
                return self._simple_word_match_boost(patient_text, guideline_text)
            
            with open(synonym_path, 'r') as f:
                synonyms = json.load(f)
            
            # Normalize patient answer using synonym file
            normalized_answer = self._normalize_with_synonyms(patient_text, synonyms, oldcarts_element)
            
            print(f"[WordMatch] 🔄 Normalization: '{patient_text}' → '{normalized_answer}'")
            
            # Get structured OLDCARTS data for this element
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
            
            normalized_lower = normalized_answer.lower().strip()
            normalized_words = normalized_answer.lower().replace('_', ' ').split()
            
            # CRITICAL: Check excludes FIRST - opposite matches should get PENALTY
            for term in excludes_terms:
                term_lower = term.lower()
                # Check if patient answer matches an exclude term (opposite side)
                if any(word in term_lower for word in normalized_words):
                    print(f"[WordMatch] ⛔ EXCLUDE MATCH: '{normalized_answer}' matches exclude term '{term}'")
                    print(f"[WordMatch]   Applying penalty: -0.3")
                    return -0.3  # PENALTY for opposite side
            
            # Check for match in includes terms
            for term in includes_terms:
                if normalized_lower in term.lower() or term.lower() in normalized_lower:
                    print(f"[WordMatch] ✅ Exact match found: '{normalized_answer}' ↔ '{term}'")
                    return 0.3  # Strong boost for exact match
            
            # Check for partial word matches using normalized words
            best_match_ratio = 0.0
            best_matching_term = None
            
            for term in includes_terms:
                term_words = term.lower().split()
                matches = sum(1 for word in normalized_words if word in term_words)
                total_words = len(normalized_words)
                
                if total_words > 0:
                    match_ratio = matches / total_words
                    if match_ratio > best_match_ratio:
                        best_match_ratio = match_ratio
                        best_matching_term = term
            
            if best_match_ratio >= 0.5:  # At least half the words match
                print(f"[WordMatch] ✅ Partial match: {best_match_ratio:.1%} with '{best_matching_term}'")
                print(f"[WordMatch]   Normalized words: {normalized_words}")
                return 0.2  # Medium boost for partial match
            elif best_match_ratio >= 0.25:  # At least a quarter match
                print(f"[WordMatch] 🔍 Weak match: {best_match_ratio:.1%} with '{best_matching_term}'")
                print(f"[WordMatch]   Normalized words: {normalized_words}")
                return 0.1  # Small boost for weak match
            
            print(f"[WordMatch] ❌ No significant word matches found")
            print(f"[WordMatch]   Normalized words: {normalized_words}")
            print(f"[WordMatch]   Available includes terms: {includes_terms}")
            return 0.0
            
        except Exception as e:
            print(f"[WordMatch] ❌ Error in synonym-based matching: {e}")
            return self._simple_word_match_boost(patient_text, guideline_text)
    
    def _normalize_with_synonyms(self, patient_text: str, synonyms: dict, oldcarts_element: str) -> str:
        """Normalize patient text using synonym file for specific OLDCARTS element"""
        if oldcarts_element not in synonyms:
            return patient_text.lower().strip()
        
        patient_lower = patient_text.lower()
        element_synonyms = synonyms[oldcarts_element]
        
        # Find the best matching synonym category
        for standard_term, synonym_list in element_synonyms.items():
            for synonym in synonym_list:
                if synonym in patient_lower:
                    print(f"[WordMatch] 🔄 Synonym match: '{synonym}' → '{standard_term}'")
                    return standard_term
        
        # No synonym match found, return original text
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
    
    def _extract_anatomical_features(self, text: str) -> Dict:
        """
        Extract anatomical features from text
        """
        text_lower = text.lower()
        
        return {
            'has_right_quadrant': bool(re.search(r'right.*quadrant|ruq|rlq|right.*side', text_lower)),
            'has_left_quadrant': bool(re.search(r'left.*quadrant|luq|llq|left.*side', text_lower)),
            'has_bilateral': bool(re.search(r'bilateral|either side|both sides|unilateral', text_lower)),
            'has_midline': bool(re.search(r'midline|epigastric|periumbilical|central', text_lower)),
            'has_flank': bool(re.search(r'flank|side', text_lower)),
            'has_chest': bool(re.search(r'chest|thoracic', text_lower)),
            'has_back': bool(re.search(r'back|posterior', text_lower)),
            'has_upper': bool(re.search(r'upper|superior', text_lower)),
            'has_lower': bool(re.search(r'lower|inferior', text_lower)),
            'has_anterior': bool(re.search(r'anterior|front', text_lower)),
            'has_posterior': bool(re.search(r'posterior|back', text_lower)),
            'has_radiates': bool(re.search(r'radiates|referred', text_lower)),
            'has_migrates': bool(re.search(r'migrates|moves|travels', text_lower)),
            'has_localizes': bool(re.search(r'localizes|localized|focal', text_lower)),
            'has_diffuse': bool(re.search(r'diffuse|widespread|generalized', text_lower)),
            'spatial_term_count': len(re.findall(r'quadrant|side|flank|epigastric|midline|chest|back', text_lower))
        }
    
    def _compute_duration_similarity(self, patient_text: str, guideline_text: str) -> Dict[str, Any]:
        """
        Special handling for duration/time-based similarity
        
        Args:
            patient_text: Patient's duration response (e.g., "1 day", "2 hours", "constant")
            guideline_text: Guideline text containing duration information
            
        Returns:
            Duration-specific similarity result or None if not applicable
        """
        # Check if this looks like a duration question
        duration_indicators = ['day', 'hour', 'minute', 'week', 'month', 'constant', 'intermittent', 'episodic', 'persistent']
        patient_lower = patient_text.lower()
        guideline_lower = guideline_text.lower()
        
        # Only apply duration logic if patient text contains time references
        if not any(indicator in patient_lower for indicator in duration_indicators):
            return None
        
        # Extract time references from patient text
        patient_time = self._extract_time_reference(patient_text)
        if not patient_time:
            return None
        
        # Extract time references from guideline text
        guideline_times = self._extract_guideline_times(guideline_text)
        if not guideline_times:
            return None
        
        # Compute duration similarity
        similarity = self._match_duration_times(patient_time, guideline_times)
        
        return {
            'similarity': similarity,
            'method': 'duration_specialized',
            'confidence': 'high' if similarity > 0.7 else 'medium',
            'reasoning': f'Duration match: {patient_time} vs {guideline_times}',
            'anatomical_type': 'duration',
            'semantic_score': similarity
        }
    
    def _extract_time_reference(self, text: str) -> Dict[str, Any]:
        """Extract time reference from patient text"""
        text_lower = text.lower().strip()
        
        # Parse various time formats
        time_patterns = [
            (r'(\d+)\s*days?', 'days'),
            (r'(\d+)\s*hours?', 'hours'), 
            (r'(\d+)\s*minutes?', 'minutes'),
            (r'(\d+)\s*weeks?', 'weeks'),
            (r'(\d+)\s*months?', 'months'),
            (r'constant', 'constant'),
            (r'intermittent', 'intermittent'),
            (r'episodic', 'episodic'),
            (r'persistent', 'persistent')
        ]
        
        for pattern, unit in time_patterns:
            match = re.search(pattern, text_lower)
            if match:
                if unit in ['constant', 'intermittent', 'episodic', 'persistent']:
                    return {'type': 'qualitative', 'value': unit, 'text': text}
                else:
                    return {'type': 'quantitative', 'value': int(match.group(1)), 'unit': unit, 'text': text}
        
        return None
    
    def _extract_guideline_times(self, text: str) -> List[Dict[str, Any]]:
        """Extract time references from guideline text"""
        text_lower = text.lower()
        times = []
        
        # Look for specific duration patterns in medical text
        duration_patterns = [
            (r'(\d+)\s*to\s*(\d+)\s*hours?', 'hours_range'),
            (r'(\d+)\s*-\s*(\d+)\s*hours?', 'hours_range'),
            (r'(\d+)\s*to\s*(\d+)\s*days?', 'days_range'),
            (r'(\d+)\s*-\s*(\d+)\s*days?', 'days_range'),
            (r'(\d+)\s*to\s*(\d+)\s*minutes?', 'minutes_range'),
            (r'(\d+)\s*-\s*(\d+)\s*minutes?', 'minutes_range'),
            (r'(\d+)\s*hours?', 'hours'),
            (r'(\d+)\s*days?', 'days'),
            (r'(\d+)\s*minutes?', 'minutes'),
            (r'(\d+)\s*weeks?', 'weeks'),
            (r'(\d+)\s*months?', 'months'),
            (r'constant', 'constant'),
            (r'intermittent', 'intermittent'),
            (r'episodic', 'episodic'),
            (r'persistent', 'persistent'),
            (r'lasting\s*(\d+)\s*hours?', 'hours'),
            (r'lasting\s*(\d+)\s*days?', 'days'),
            (r'lasting\s*(\d+)\s*minutes?', 'minutes')
        ]
        
        for pattern, unit in duration_patterns:
            matches = re.finditer(pattern, text_lower)
            for match in matches:
                if 'range' in unit:
                    # Handle ranges like "6-12 hours"
                    start_val = int(match.group(1))
                    end_val = int(match.group(2))
                    times.append({
                        'type': 'range',
                        'start': start_val,
                        'end': end_val,
                        'unit': unit.replace('_range', ''),
                        'text': match.group(0)
                    })
                else:
                    # Handle single values
                    value = int(match.group(1)) if match.group(1) else 0
                    times.append({
                        'type': 'single',
                        'value': value,
                        'unit': unit,
                        'text': match.group(0)
                    })
        
        return times
    
    def _match_duration_times(self, patient_time: Dict[str, Any], guideline_times: List[Dict[str, Any]]) -> float:
        """Match patient time against guideline times"""
        if not patient_time or not guideline_times:
            return 0.0
        
        # Convert patient time to hours for comparison
        patient_hours = self._convert_to_hours(patient_time)
        if patient_hours is None:
            return 0.0
        
        best_match = 0.0
        
        for guideline_time in guideline_times:
            if guideline_time['type'] == 'range':
                # Check if patient time falls within range
                guideline_start = self._convert_to_hours(guideline_time)
                guideline_end = self._convert_to_hours(guideline_time)
                
                if guideline_start <= patient_hours <= guideline_end:
                    # Perfect match within range
                    match_score = 1.0
                elif patient_hours < guideline_start:
                    # Patient time is shorter - partial match
                    match_score = max(0.0, 1.0 - (guideline_start - patient_hours) / guideline_start)
                else:
                    # Patient time is longer - partial match
                    match_score = max(0.0, 1.0 - (patient_hours - guideline_end) / guideline_end)
                
                best_match = max(best_match, match_score)
                
            elif guideline_time['type'] == 'single':
                # Check single value match
                guideline_hours = self._convert_to_hours(guideline_time)
                if guideline_hours is not None:
                    if patient_hours == guideline_hours:
                        match_score = 1.0
                    else:
                        # Calculate similarity based on ratio
                        ratio = min(patient_hours, guideline_hours) / max(patient_hours, guideline_hours)
                        match_score = ratio
                    
                    best_match = max(best_match, match_score)
        
        return best_match
    
    def _convert_to_hours(self, time_ref: Dict[str, Any]) -> float:
        """Convert time reference to hours for comparison"""
        if time_ref['type'] == 'qualitative':
            # Handle qualitative terms
            if time_ref['value'] == 'constant':
                return 24.0  # Assume constant means all day
            elif time_ref['value'] in ['intermittent', 'episodic']:
                return 0.5  # Short episodes
            elif time_ref['value'] == 'persistent':
                return 12.0  # Persistent but not necessarily constant
            else:
                return 0.0
        
        elif time_ref['type'] == 'quantitative':
            # Convert to hours
            value = time_ref['value']
            unit = time_ref['unit']
            
            if unit == 'hours':
                return float(value)
            elif unit == 'days':
                return float(value * 24)
            elif unit == 'minutes':
                return float(value) / 60.0
            elif unit == 'weeks':
                return float(value * 24 * 7)
            elif unit == 'months':
                return float(value * 24 * 30)  # Approximate
            else:
                return 0.0
        
        elif time_ref['type'] == 'range':
            # For ranges, return the midpoint
            start_hours = self._convert_to_hours({'type': 'quantitative', 'value': time_ref['start'], 'unit': time_ref['unit']})
            end_hours = self._convert_to_hours({'type': 'quantitative', 'value': time_ref['end'], 'unit': time_ref['unit']})
            return (start_hours + end_hours) / 2.0
        
        return 0.0
    
    def _compute_embedding_similarity(self, patient_text: str, guideline_text: str, organ_system: str = None, oldcarts_element: str = None, structured_oldcarts: dict = None) -> Dict[str, Any]:
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
            # SPECIAL HANDLING FOR DURATION: Extract and normalize time references
            duration_similarity = self._compute_duration_similarity(patient_text, guideline_text)
            if duration_similarity is not None:
                return duration_similarity
            
            # Encode both texts into embeddings
            embeddings = self.embedding_model.encode([patient_text, guideline_text])
            patient_emb = embeddings[0]
            guideline_emb = embeddings[1]
            
            # Compute cosine similarity
            similarity = float(np.dot(patient_emb, guideline_emb) / (np.linalg.norm(patient_emb) * np.linalg.norm(guideline_emb)))
            
            # BOOST SIMILARITY FOR DIRECT WORD MATCHES
            word_match_boost = self._compute_word_match_boost(patient_text, guideline_text, organ_system, oldcarts_element, structured_oldcarts)
            boost_applied = False
            if word_match_boost > 0:
                similarity = min(1.0, similarity + word_match_boost)
                boost_applied = True
                print(f"[Embedding] 🎯 Word match boost: +{word_match_boost:.3f}")
            
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

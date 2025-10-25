#!/usr/bin/env python3
"""
Medical Rule Engine
Combines hardcoded rules with ML predictions for anatomical relationships
"""

import json
import joblib
import re
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
    
    def __init__(self, ml_model_path: str = "ml/location_ml_model.pkl"):
        self.ml_model = None
        self.ml_trainer = None
        
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
        Load hardcoded medical rules
        """
        return {
            'GI': {
                'bilateral': [
                    'Acute Gastroenteritis', 'Severe Constipation', 'IBD Flare', 'IBS',
                    'Acute Mesenteric Ischemia'
                ],
                'midline': [
                    'Peptic Ulcer Disease', 'Acute Gastritis', 'Acute Pancreatitis',
                    'Gastric Outlet Obstruction'
                ],
                'right_only': [
                    'Acute Appendicitis', 'Acute Cholecystitis', 'Biliary Colic',
                    'Acute Cholangitis', 'Acute Hepatitis'
                ],
                'left_only': [
                    'Acute Diverticulitis', 'Sigmoid Volvulus'
                ]
            },
            'GU': {
                'bilateral': [
                    'Kidney Stone', 'UTI/Pyelonephritis'
                ],
                'midline': [
                    'Bladder Infection', 'Urethritis'
                ]
            },
            'CARDIO': {
                'bilateral': [
                    'Pneumothorax', 'Pleural Effusion', 'Pleurisy'
                ],
                'midline': [
                    'Aortic Dissection', 'Aortic Stenosis'
                ]
            },
            'PULMONARY': {
                'bilateral': [
                    'Pneumothorax', 'Pleural Effusion'
                ],
                'midline': [
                    'Epiglottitis'
                ]
            }
        }
    
    def get_anatomical_type(self, condition_name: str, organ_system: str = None) -> str:
        """
        Get anatomical type for condition using hardcoded rules first
        """
        # Check hardcoded rules first
        if organ_system and organ_system in self.medical_rules:
            for anatomical_type, conditions in self.medical_rules[organ_system].items():
                if condition_name in conditions:
                    return anatomical_type
        
        # Check all organ systems if no specific system provided
        for system, rules in self.medical_rules.items():
            for anatomical_type, conditions in rules.items():
                if condition_name in conditions:
                    return anatomical_type
        
        return 'unknown'
    
    def get_enhanced_similarity(self, patient_text: str, guideline_text: str, 
                              condition_name: str, organ_system: str = None) -> Dict[str, Any]:
        """
        Enhanced similarity with SEMANTIC SIMILARITY FIRST, anatomical rules as modifiers
        
        CRITICAL FIX: Use semantic similarity as PRIMARY scoring method,
        anatomical rules only as fallbacks or modifiers for inconclusive cases
        """
        
        # 1. COMPUTE SEMANTIC SIMILARITY FIRST (Primary scoring method)
        semantic_result = self._compute_semantic_similarity(patient_text, guideline_text)
        semantic_score = semantic_result['similarity']
        
        # 2. Get anatomical type for validation/modification
        anatomical_type = self.get_anatomical_type(condition_name, organ_system)
        
        # 3. HIGH SEMANTIC SIMILARITY - Don't let anatomical rules override perfect matches
        if semantic_score >= 0.7:  # High semantic similarity (70%+)
            # For high similarity, only check for anatomical opposites as a safety check
            if anatomical_type in ['right_only', 'left_only'] and self._is_anatomical_opposite(patient_text, guideline_text):
                return {
                    'similarity': 0.0,
                    'method': 'anatomical_override',
                    'confidence': 'high', 
                    'reasoning': f'High semantic similarity ({semantic_score:.2f}) overridden by anatomical opposite',
                    'anatomical_type': anatomical_type,
                    'semantic_score': semantic_score
                }
            else:
                # Use semantic similarity for high matches
                return {
                    'similarity': semantic_score,
                    'method': 'semantic_similarity',
                    'confidence': 'high',
                    'reasoning': f'High semantic similarity: {semantic_result["reasoning"]}',
                    'anatomical_type': anatomical_type,
                    'semantic_score': semantic_score
                }
        
        # 4. MEDIUM SEMANTIC SIMILARITY - Blend with anatomical rules
        elif semantic_score >= 0.3:  # Medium semantic similarity (30-70%)
            
            # Check for anatomical opposites first (overrides everything)
            if anatomical_type in ['right_only', 'left_only'] and self._is_anatomical_opposite(patient_text, guideline_text):
                return {
                    'similarity': 0.0,
                    'method': 'anatomical_opposite',
                    'confidence': 'high',
                    'reasoning': 'Anatomical opposite detected',
                    'anatomical_type': anatomical_type,
                    'semantic_score': semantic_score
                }
            
            # Blend semantic similarity with anatomical rules
            if anatomical_type == 'bilateral':
                # For bilateral conditions, average semantic similarity with bilateral bonus
                blended_score = (semantic_score * 0.7) + (0.5 * 0.3)  # Weight semantic 70%, anatomical 30%
                return {
                    'similarity': min(blended_score, 0.8),  # Cap at 80% for medium semantic
                    'method': 'semantic_bilateral_blend',
                    'confidence': 'high',
                    'reasoning': f'Semantic ({semantic_score:.2f}) blended with bilateral rule',
                    'anatomical_type': anatomical_type,
                    'semantic_score': semantic_score
                }
            
            elif anatomical_type == 'midline':
                # For midline conditions, average semantic similarity with midline bonus
                blended_score = (semantic_score * 0.8) + (0.4 * 0.2)  # Weight semantic 80%, anatomical 20%  
                return {
                    'similarity': min(blended_score, 0.7),  # Cap at 70% for medium semantic
                    'method': 'semantic_midline_blend',
                    'confidence': 'high',
                    'reasoning': f'Semantic ({semantic_score:.2f}) blended with midline rule',
                    'anatomical_type': anatomical_type,
                    'semantic_score': semantic_score
                }
            
            elif anatomical_type in ['right_only', 'left_only']:
                # For unilateral conditions, use semantic similarity with slight boost for same side
                boosted_score = min(semantic_score * 1.1, 0.75)  # 10% boost, cap at 75%
                return {
                    'similarity': boosted_score,
                    'method': 'semantic_same_side',
                    'confidence': 'medium',
                    'reasoning': f'Semantic similarity ({semantic_score:.2f}) with same-side boost',
                    'anatomical_type': anatomical_type,
                    'semantic_score': semantic_score
                }
            
            else:
                # Unknown anatomical type - use semantic similarity directly
                return {
                    'similarity': semantic_score,
                    'method': 'semantic_similarity',
                    'confidence': 'medium',
                    'reasoning': f'Semantic similarity: {semantic_result["reasoning"]}',
                    'anatomical_type': anatomical_type,
                    'semantic_score': semantic_score
                }
        
        # 5. LOW SEMANTIC SIMILARITY - Use anatomical rules as fallback
        else:  # semantic_score < 0.3
            
            # Check for anatomical opposites (should be 0%)
            if anatomical_type in ['right_only', 'left_only'] and self._is_anatomical_opposite(patient_text, guideline_text):
                return {
                    'similarity': 0.0,
                    'method': 'anatomical_opposite',
                    'confidence': 'high',
                    'reasoning': 'Anatomical opposite detected',
                    'anatomical_type': anatomical_type,
                    'semantic_score': semantic_score
                }
            
            # For low semantic similarity, use anatomical rules as fallback
            if anatomical_type == 'bilateral':
                return {
                    'similarity': 0.5,
                    'method': 'bilateral_rule_fallback',
                    'confidence': 'medium',
                    'reasoning': f'Low semantic ({semantic_score:.2f}) - bilateral fallback',
                    'anatomical_type': anatomical_type,
                    'semantic_score': semantic_score
                }
            
            elif anatomical_type == 'midline':
                return {
                    'similarity': 0.4,
                    'method': 'midline_rule_fallback', 
                    'confidence': 'medium',
                    'reasoning': f'Low semantic ({semantic_score:.2f}) - midline fallback',
                    'anatomical_type': anatomical_type,
                    'semantic_score': semantic_score
                }
            
            elif anatomical_type in ['right_only', 'left_only']:
                return {
                    'similarity': 0.3,
                    'method': 'same_side_fallback',
                    'confidence': 'low',
                    'reasoning': f'Low semantic ({semantic_score:.2f}) - same side fallback',
                    'anatomical_type': anatomical_type,
                    'semantic_score': semantic_score
                }
            
            # 6. Use ML prediction if available (for unknown anatomical type)
            if self.ml_model:
                ml_result = self._get_ml_prediction(patient_text, guideline_text, condition_name, organ_system)
                return {
                    'similarity': ml_result['similarity'],
                    'method': 'ml_prediction_fallback',
                    'confidence': 'medium',
                    'reasoning': f"Low semantic ({semantic_score:.2f}) - ML fallback: {ml_result['predicted_type']}",
                    'anatomical_type': ml_result['predicted_type'],
                    'semantic_score': semantic_score
                }
            
            # 7. Final fallback - use low semantic similarity
            return {
                'similarity': max(semantic_score, 0.1),  # Minimum 10%
                'method': 'semantic_low',
                'confidence': 'low',
                'reasoning': f'Low semantic similarity: {semantic_result["reasoning"]}',
                'anatomical_type': 'unknown',
                'semantic_score': semantic_score
            }
    
    def _is_anatomical_opposite(self, patient_text: str, guideline_text: str) -> bool:
        """
        Check for anatomical opposites - only true opposites, not same-side matches
        """
        patient_lower = patient_text.lower()
        guideline_lower = guideline_text.lower()
        
        # First check for same-side matches (should NOT be opposites)
        same_side_matches = [
            ('left', 'left'), ('right', 'right'),
            ('upper', 'upper'), ('lower', 'lower'),
            ('anterior', 'anterior'), ('posterior', 'posterior')
        ]
        
        for patient_term, guideline_term in same_side_matches:
            if patient_term in patient_lower and guideline_term in guideline_lower:
                return False  # Same side = NOT opposite
        
        # Then check for true opposites only
        opposites = [
            ('left', 'right'), ('right', 'left'),
            ('upper', 'lower'), ('lower', 'upper'),
            ('anterior', 'posterior'), ('posterior', 'anterior')
        ]
        
        for patient_term, guideline_term in opposites:
            if patient_term in patient_lower and guideline_term in guideline_lower:
                return True
        
        return False
    
    def _get_ml_prediction(self, patient_text: str, guideline_text: str, 
                          condition_name: str, organ_system: str) -> Dict:
        """
        Get ML prediction for anatomical type
        """
        # If ML trainer is not available, return default prediction
        if not self.ml_trainer or not self.ml_model:
            return {
                'predicted_type': 'unilateral',
                'confidence': 0.5,
                'similarity_score': 0.5
            }
        
        # Extract anatomical features from guideline text
        anatomical_features = self._extract_anatomical_features(guideline_text)
        
        # Get ML prediction
        prediction = self.ml_trainer.predict_anatomical_type(
            self.ml_model, guideline_text, organ_system, anatomical_features
        )
        
        # Convert prediction to similarity score
        if prediction['predicted_type'] == 'bilateral':
            similarity = 0.5
        elif prediction['predicted_type'] == 'midline':
            similarity = 0.4
        elif prediction['predicted_type'] in ['right_only', 'left_only']:
            # Check for opposites
            if self._is_anatomical_opposite(patient_text, guideline_text):
                similarity = 0.0
            else:
                similarity = 0.3
        else:
            similarity = 0.2  # Unknown type
        
        return {
            'predicted_type': prediction['predicted_type'],
            'confidence': prediction['confidence'],
            'similarity': similarity
        }
    
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
    
    def _compute_semantic_similarity(self, patient_text: str, guideline_text: str) -> Dict[str, Any]:
        """
        Compute semantic similarity between patient input and guideline text
        This is the PRIMARY scoring method for OLDCARTS matching
        """
        # Normalize inputs
        patient_lower = patient_text.lower().strip()
        guideline_lower = guideline_text.lower().strip()
        
        if not patient_lower or not guideline_lower:
            return {
                'similarity': 0.0,
                'method': 'no_text',
                'confidence': 'low',
                'reasoning': 'No text to compare'
            }
        
        # 1. EXACT MATCH - Highest score
        if patient_lower == guideline_lower:
            return {
                'similarity': 1.0,
                'method': 'exact_match',
                'confidence': 'high',
                'reasoning': 'Exact text match'
            }
        
        # 2. CONTRADICTION DETECTION - Look for explicit contradictions
        if self._detect_contradiction(patient_lower, guideline_lower):
            return {
                'similarity': 0.1,  # Very low but not zero
                'method': 'contradiction',
                'confidence': 'high', 
                'reasoning': 'Text contains contradictory information'
            }
        
        # 3. ADVANCED SEMANTIC MATCHING - Check for conceptual matches
        patient_words = set(patient_lower.split())
        guideline_words = set(guideline_lower.split())
        
        # Remove common stop words for better matching
        stop_words = {'the', 'and', 'or', 'in', 'on', 'at', 'to', 'of', 'a', 'an', 'is', 'are', 'was', 'were', 'pain', 'typically'}
        patient_words = patient_words - stop_words
        guideline_words = guideline_words - stop_words
        
        if not patient_words or not guideline_words:
            similarity = 0.2
            method = 'no_meaningful_words'
        else:
            # 4. MEDICAL CONCEPT MATCHING - Map synonymous medical terms
            conceptual_similarity = self._compute_medical_concept_similarity(patient_words, guideline_words)
            
            # 5. WORD OVERLAP ANALYSIS
            intersection = len(patient_words.intersection(guideline_words))
            union = len(patient_words.union(guideline_words))
            jaccard = intersection / union if union > 0 else 0.0
            
            # 6. SEMANTIC KEYWORD MATCHING
            semantic_boost = self._compute_semantic_keyword_boost(patient_words, guideline_words)
            
            # 7. SUBSTRING MATCHING BONUS
            substring_bonus = 0.0
            if patient_lower in guideline_lower or guideline_lower in patient_lower:
                substring_bonus = 0.3  # 30% bonus for substring matches
            
            # Combine all similarity measures
            similarity = min(
                max(conceptual_similarity, jaccard) + semantic_boost + substring_bonus, 
                0.95  # Cap at 95% for non-exact matches
            )
            
            # Assign method based on primary contributor
            if conceptual_similarity >= 0.6:
                method = 'medical_concept_match'
            elif substring_bonus > 0:
                method = 'substring_match'
            elif jaccard >= 0.5:
                method = 'high_word_overlap'
            elif jaccard >= 0.2:
                method = 'medium_word_overlap'
            else:
                method = 'low_word_overlap'
        
        return {
            'similarity': similarity,
            'method': method,
            'confidence': 'medium',
            'reasoning': f'Semantic analysis: {method}'
        }
    
    def _detect_contradiction(self, patient_text: str, guideline_text: str) -> bool:
        """
        Detect explicit contradictions between patient input and guideline
        """
        # Common contradiction patterns
        contradictions = [
            # Location contradictions
            (['left'], ['right']),
            (['right'], ['left']),  
            (['upper'], ['lower']),
            (['lower'], ['upper']),
            
            # Localization contradictions  
            (['not localized', 'not local', 'diffuse', 'widespread'], ['localized', 'focal', 'specific']),
            (['localized', 'focal', 'specific'], ['not localized', 'not local', 'diffuse', 'widespread']),
            
            # Severity contradictions
            (['mild', 'slight'], ['severe', 'intense', 'excruciating']),
            (['severe', 'intense'], ['mild', 'slight']),
            
            # Timing contradictions
            (['constant', 'continuous'], ['intermittent', 'comes and goes']),
            (['intermittent', 'comes and goes'], ['constant', 'continuous']),
        ]
        
        for patient_terms, guideline_terms in contradictions:
            patient_has = any(term in patient_text for term in patient_terms)
            guideline_has = any(term in guideline_text for term in guideline_terms)
            
            if patient_has and guideline_has:
                return True
        
        return False
    
    def _compute_medical_concept_similarity(self, patient_words: set, guideline_words: set) -> float:
        """
        Compute similarity based on medical concept mapping
        Maps patient language to medical terminology
        """
        # Medical concept mappings - patient language to medical terms
        concept_mappings = {
            # Location mappings
            'left': ['left', 'llq', 'luq'],
            'right': ['right', 'rlq', 'ruq'], 
            'lower': ['lower', 'llq', 'rlq', 'hypogastric', 'suprapubic'],
            'upper': ['upper', 'luq', 'ruq', 'epigastric'],
            'part': ['quadrant', 'area', 'region', 'zone'],
            'side': ['side', 'lateral', 'quadrant'],
            
            # Character mappings
            'sharp': ['sharp', 'stabbing', 'knife-like', 'piercing'],
            'dull': ['dull', 'aching', 'throbbing'],
            'burning': ['burning', 'searing', 'hot'],
            'cramping': ['cramping', 'colicky', 'spasmodic'],
            
            # Timing mappings
            'sudden': ['sudden', 'acute', 'abrupt'],
            'gradual': ['gradual', 'insidious', 'slow'],
            'constant': ['constant', 'continuous', 'persistent'],
            'comes': ['intermittent', 'episodic', 'comes'],
            'goes': ['intermittent', 'episodic', 'goes'],
            
            # Localization mappings
            'localized': ['localized', 'focal', 'specific'],
            'diffuse': ['diffuse', 'widespread', 'generalized'],
            'not': ['not', 'no', 'without'],
        }
        
        # Calculate conceptual matches
        conceptual_score = 0.0
        total_patient_concepts = 0
        
        for patient_word in patient_words:
            total_patient_concepts += 1
            
            # Check if patient word maps to any guideline concepts
            if patient_word in concept_mappings:
                mapped_concepts = concept_mappings[patient_word]
                
                # Check if any mapped concept appears in guideline
                for concept in mapped_concepts:
                    if any(concept in guideline_word for guideline_word in guideline_words):
                        conceptual_score += 1.0
                        break  # Only count once per patient word
                    # Also check exact matches
                    elif concept in guideline_words:
                        conceptual_score += 1.0
                        break
            
            # Direct word matches get full score
            elif patient_word in guideline_words:
                conceptual_score += 1.0
        
        # Normalize by total patient concepts
        if total_patient_concepts > 0:
            return conceptual_score / total_patient_concepts
        else:
            return 0.0
    
    def _compute_semantic_keyword_boost(self, patient_words: set, guideline_words: set) -> float:
        """
        Compute semantic boost based on meaningful medical keyword matches
        """
        # Important medical location keywords get higher weight
        location_keywords = {
            'quadrant', 'rlq', 'llq', 'ruq', 'luq', 'epigastric', 'periumbilical', 
            'flank', 'groin', 'chest', 'abdomen', 'abdominal'
        }
        
        # Character keywords
        character_keywords = {
            'sharp', 'dull', 'cramping', 'burning', 'stabbing', 'aching', 'throbbing'
        }
        
        # Timing keywords
        timing_keywords = {
            'constant', 'intermittent', 'continuous', 'episodic', 'waves'
        }
        
        boost = 0.0
        
        # Location keyword matches get high boost
        location_matches = patient_words.intersection(guideline_words).intersection(location_keywords)
        boost += len(location_matches) * 0.2  # 20% boost per location keyword match
        
        # Character keyword matches get medium boost  
        character_matches = patient_words.intersection(guideline_words).intersection(character_keywords)
        boost += len(character_matches) * 0.15  # 15% boost per character keyword match
        
        # Timing keyword matches get medium boost
        timing_matches = patient_words.intersection(guideline_words).intersection(timing_keywords) 
        boost += len(timing_matches) * 0.1  # 10% boost per timing keyword match
        
        return min(boost, 0.3)  # Cap semantic boost at 30%
    
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

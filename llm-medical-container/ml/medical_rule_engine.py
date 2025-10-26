#!/usr/bin/env python3
"""
Medical Rule Engine
Combines hardcoded rules with ML predictions for anatomical relationships
"""

import json
import joblib
import re
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
                              condition_name: str, organ_system: str = None, oldcarts_element: str = None) -> Dict[str, Any]:
        """
        Enhanced similarity with SEMANTIC SIMILARITY FIRST, anatomical rules as modifiers
        
        CRITICAL FIX: Use semantic similarity as PRIMARY scoring method,
        anatomical rules only as fallbacks or modifiers for inconclusive cases
        """
        
        # Use patient text directly - semantic similarity handles all variations naturally
        # No need for synonym normalization - embeddings understand "below ribs" = "upper quadrant" etc.
        patient_text_for_scoring = patient_text.lower()
        
        # 1. COMPUTE SEMANTIC SIMILARITY FIRST (Primary scoring method)
        # Use embedding model for deep semantic similarity
        semantic_result = self._compute_embedding_similarity(patient_text_for_scoring, guideline_text)
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
                # Check if patient's side matches condition's side
                patient_lower = patient_text_for_scoring.lower()
                guideline_lower = guideline_text.lower()
                
                # Check basic side match (embedding similarity handles quadrant matching)
                patient_has_left = any(term in patient_lower for term in ['left', 'llq', 'luq'])
                patient_has_right = any(term in patient_lower for term in ['right', 'rlq', 'ruq'])
                
                if anatomical_type == 'left_only' and patient_has_left:
                    # Patient mentions left, condition is left-only → Same side match
                    # Use semantic score (embedding already captured quadrant specificity)
                    return {
                        'similarity': max(semantic_score, 0.3),
                        'method': 'same_side_fallback',
                        'confidence': 'medium',
                        'reasoning': f'Low semantic ({semantic_score:.2f}) - same side fallback (left)',
                        'anatomical_type': anatomical_type,
                        'semantic_score': semantic_score
                    }
                elif anatomical_type == 'right_only' and patient_has_right:
                    # Patient mentions right, condition is right-only → Same side match
                    # Use semantic score (embedding already captured quadrant specificity)
                    return {
                        'similarity': max(semantic_score, 0.3),
                        'method': 'same_side_fallback',
                        'confidence': 'medium',
                        'reasoning': f'Low semantic ({semantic_score:.2f}) - same side fallback (right)',
                        'anatomical_type': anatomical_type,
                        'semantic_score': semantic_score
                    }
                elif (anatomical_type == 'left_only' and patient_has_right) or (anatomical_type == 'right_only' and patient_has_left):
                    # Clear anatomical mismatch → Very low score
                    return {
                        'similarity': 0.05,
                        'method': 'anatomical_mismatch',
                        'confidence': 'high',
                        'reasoning': f'Low semantic ({semantic_score:.2f}) + anatomical mismatch ({anatomical_type} vs patient side)',
                        'anatomical_type': anatomical_type,
                        'semantic_score': semantic_score
                    }
                else:
                    # No clear anatomical information from patient → Use semantic similarity
                    return {
                        'similarity': semantic_score,
                        'method': 'semantic_only',
                        'confidence': 'low',
                        'reasoning': f'Low semantic ({semantic_score:.2f}) - no clear anatomical info from patient',
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
    
    def _compute_embedding_similarity(self, patient_text: str, guideline_text: str) -> Dict[str, Any]:
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
            # Fallback to traditional semantic similarity
            return self._compute_semantic_similarity(patient_text, guideline_text)
        
        try:
            # Encode both texts into embeddings
            embeddings = self.embedding_model.encode([patient_text, guideline_text])
            patient_emb = embeddings[0]
            guideline_emb = embeddings[1]
            
            # Compute cosine similarity
            similarity = float(np.dot(patient_emb, guideline_emb) / (np.linalg.norm(patient_emb) * np.linalg.norm(guideline_emb)))
            
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
                'reasoning': f'Embedding-based semantic similarity: {similarity:.3f}'
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

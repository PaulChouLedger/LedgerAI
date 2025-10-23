#!/usr/bin/env python3
"""
Medical Rule Engine
Combines hardcoded rules with ML predictions for anatomical relationships
"""

import json
import joblib
import re
from typing import Dict, Any, List
from location_ml_trainer import LocationMLTrainer

class MedicalRuleEngine:
    """
    Medical rule engine for anatomical relationships
    Combines hardcoded rules with ML predictions
    """
    
    def __init__(self, ml_model_path: str = "location_ml_model.pkl"):
        self.ml_model = None
        self.ml_trainer = LocationMLTrainer()
        
        # Load ML model if available
        try:
            self.ml_model = self.ml_trainer.load_model(ml_model_path)
            print(f"✅ ML model loaded from {ml_model_path}")
        except FileNotFoundError:
            print(f"⚠️ ML model not found at {ml_model_path}, using hardcoded rules only")
        
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
        Enhanced similarity with medical rules and ML
        """
        
        # 1. Check hardcoded rules first
        anatomical_type = self.get_anatomical_type(condition_name, organ_system)
        
        if anatomical_type == 'bilateral':
            return {
                'similarity': 0.5,
                'method': 'bilateral_rule',
                'confidence': 'high',
                'reasoning': 'Bilateral condition - can occur on either side',
                'anatomical_type': anatomical_type
            }
        
        elif anatomical_type == 'midline':
            return {
                'similarity': 0.4,
                'method': 'midline_rule',
                'confidence': 'high',
                'reasoning': 'Midline condition - not side-specific',
                'anatomical_type': anatomical_type
            }
        
        elif anatomical_type in ['right_only', 'left_only']:
            # Check for anatomical opposites
            if self._is_anatomical_opposite(patient_text, guideline_text):
                return {
                    'similarity': 0.0,
                    'method': 'anatomical_opposite',
                    'confidence': 'high',
                    'reasoning': 'Anatomical opposite detected',
                    'anatomical_type': anatomical_type
                }
            else:
                return {
                    'similarity': 0.3,
                    'method': 'same_side',
                    'confidence': 'medium',
                    'reasoning': 'Same anatomical side',
                    'anatomical_type': anatomical_type
                }
        
        # 2. Use ML prediction if available
        if self.ml_model:
            ml_result = self._get_ml_prediction(patient_text, guideline_text, condition_name, organ_system)
            if ml_result['confidence'] > 0.7:
                return {
                    'similarity': ml_result['similarity'],
                    'method': 'ml_prediction',
                    'confidence': 'medium',
                    'reasoning': f"ML prediction: {ml_result['predicted_type']}",
                    'anatomical_type': ml_result['predicted_type']
                }
        
        # 3. Fallback to semantic similarity
        semantic_score = self._compute_semantic_similarity(patient_text, guideline_text)
        return {
            'similarity': semantic_score,
            'method': 'semantic_fallback',
            'confidence': 'low',
            'reasoning': 'Using semantic similarity fallback',
            'anatomical_type': 'unknown'
        }
    
    def _is_anatomical_opposite(self, patient_text: str, guideline_text: str) -> bool:
        """
        Check for anatomical opposites
        """
        patient_lower = patient_text.lower()
        guideline_lower = guideline_text.lower()
        
        # Check for hard opposites
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
    
    def _compute_semantic_similarity(self, text1: str, text2: str) -> float:
        """
        Compute semantic similarity (placeholder - would use embeddings)
        """
        # Simple word overlap for now
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union) if union else 0.0

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

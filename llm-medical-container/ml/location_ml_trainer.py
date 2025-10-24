#!/usr/bin/env python3
"""
Location ML Trainer
Handles training and prediction for anatomical location relationships
"""

import joblib
import numpy as np
from typing import Dict, Any, List, Optional
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
import os

class LocationMLTrainer:
    """
    ML trainer for anatomical location relationships
    Trains models to predict anatomical types (bilateral, unilateral, midline)
    """
    
    def __init__(self):
        self.model = None
        self.vectorizer = None
        self.feature_names = []
        
    def load_model(self, model_path: str) -> Optional[Any]:
        """
        Load a trained ML model from file
        
        Args:
            model_path: Path to the model file
            
        Returns:
            Loaded model or None if not found
        """
        try:
            if os.path.exists(model_path):
                model_data = joblib.load(model_path)
                if isinstance(model_data, dict):
                    self.model = model_data.get('model')
                    self.vectorizer = model_data.get('vectorizer')
                    self.feature_names = model_data.get('feature_names', [])
                else:
                    self.model = model_data
                return self.model
            else:
                print(f"⚠️ Model file not found: {model_path}")
                return None
        except Exception as e:
            print(f"⚠️ Error loading model: {e}")
            return None
    
    def save_model(self, model_path: str, model: Any, vectorizer: Any = None, feature_names: List[str] = None):
        """
        Save a trained model to file
        
        Args:
            model_path: Path to save the model
            model: Trained model
            vectorizer: Text vectorizer
            feature_names: List of feature names
        """
        try:
            model_data = {
                'model': model,
                'vectorizer': vectorizer,
                'feature_names': feature_names or []
            }
            joblib.dump(model_data, model_path)
            print(f"✅ Model saved to {model_path}")
        except Exception as e:
            print(f"⚠️ Error saving model: {e}")
    
    def train_model(self, training_data: List[Dict], model_path: str = "ml/location_ml_model.pkl"):
        """
        Train a new ML model on anatomical location data
        
        Args:
            training_data: List of training examples with 'text', 'organ_system', 'anatomical_type'
            model_path: Path to save the trained model
        """
        try:
            if not training_data:
                print("⚠️ No training data provided")
                return None
            
            # Extract features
            texts = [item['text'] for item in training_data]
            organ_systems = [item['organ_system'] for item in training_data]
            anatomical_types = [item['anatomical_type'] for item in training_data]
            
            # Create text features
            self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
            text_features = self.vectorizer.fit_transform(texts)
            
            # Create organ system features (one-hot encoding)
            organ_system_features = np.array([[1 if os == organ_system else 0 for os in set(organ_systems)] 
                                            for organ_system in organ_systems])
            
            # Combine features
            combined_features = np.hstack([text_features.toarray(), organ_system_features])
            
            # Train model
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
            self.model.fit(combined_features, anatomical_types)
            
            # Save model
            self.save_model(model_path, self.model, self.vectorizer, self.feature_names)
            
            print(f"✅ Model trained on {len(training_data)} examples")
            return self.model
            
        except Exception as e:
            print(f"⚠️ Error training model: {e}")
            return None
    
    def predict_anatomical_type(self, model: Any, guideline_text: str, 
                               organ_system: str, anatomical_features: List[str]) -> Dict:
        """
        Predict anatomical type using trained model
        
        Args:
            model: Trained model
            guideline_text: Text from medical guideline
            organ_system: Organ system (e.g., 'GI', 'Cardiac')
            anatomical_features: List of anatomical features
            
        Returns:
            Dictionary with prediction results
        """
        try:
            if not model or not self.vectorizer:
                # Fallback to rule-based prediction
                return self._rule_based_prediction(guideline_text, organ_system, anatomical_features)
            
            # Prepare features
            text_features = self.vectorizer.transform([guideline_text])
            
            # Create organ system features
            all_organ_systems = ['GI', 'Cardiac', 'Pulmonary', 'Neurological', 'Renal', 'Musculoskeletal']
            organ_system_features = np.array([[1 if os == organ_system else 0 for os in all_organ_systems]])
            
            # Combine features
            combined_features = np.hstack([text_features.toarray(), organ_system_features])
            
            # Make prediction
            prediction = model.predict(combined_features)[0]
            probabilities = model.predict_proba(combined_features)[0]
            confidence = max(probabilities)
            
            return {
                'predicted_type': prediction,
                'confidence': confidence,
                'probabilities': dict(zip(model.classes_, probabilities))
            }
            
        except Exception as e:
            print(f"⚠️ Error in prediction: {e}")
            return self._rule_based_prediction(guideline_text, organ_system, anatomical_features)
    
    def _rule_based_prediction(self, guideline_text: str, organ_system: str, anatomical_features: List[str]) -> Dict:
        """
        Fallback rule-based prediction when ML model is not available
        
        Args:
            guideline_text: Text from medical guideline
            organ_system: Organ system
            anatomical_features: List of anatomical features
            
        Returns:
            Dictionary with rule-based prediction
        """
        text_lower = guideline_text.lower()
        
        # Rule-based classification
        bilateral_indicators = ['both', 'bilateral', 'left and right', 'upper and lower']
        midline_indicators = ['midline', 'center', 'central', 'middle']
        unilateral_indicators = ['left', 'right', 'upper', 'lower', 'anterior', 'posterior']
        
        bilateral_score = sum(1 for indicator in bilateral_indicators if indicator in text_lower)
        midline_score = sum(1 for indicator in midline_indicators if indicator in text_lower)
        unilateral_score = sum(1 for indicator in unilateral_indicators if indicator in text_lower)
        
        if bilateral_score > midline_score and bilateral_score > unilateral_score:
            predicted_type = 'bilateral'
            confidence = 0.8
        elif midline_score > bilateral_score and midline_score > unilateral_score:
            predicted_type = 'midline'
            confidence = 0.8
        else:
            predicted_type = 'unilateral'
            confidence = 0.7
        
        return {
            'predicted_type': predicted_type,
            'confidence': confidence,
            'probabilities': {
                'bilateral': bilateral_score / (bilateral_score + midline_score + unilateral_score + 1),
                'midline': midline_score / (bilateral_score + midline_score + unilateral_score + 1),
                'unilateral': unilateral_score / (bilateral_score + midline_score + unilateral_score + 1)
            }
        }
    
    def create_sample_training_data(self) -> List[Dict]:
        """
        Create sample training data for initial model training
        
        Returns:
            List of training examples
        """
        return [
            # GI examples
            {'text': 'right lower quadrant pain', 'organ_system': 'GI', 'anatomical_type': 'unilateral'},
            {'text': 'left upper quadrant pain', 'organ_system': 'GI', 'anatomical_type': 'unilateral'},
            {'text': 'bilateral abdominal pain', 'organ_system': 'GI', 'anatomical_type': 'bilateral'},
            {'text': 'midline epigastric pain', 'organ_system': 'GI', 'anatomical_type': 'midline'},
            
            # Cardiac examples
            {'text': 'left chest pain', 'organ_system': 'Cardiac', 'anatomical_type': 'unilateral'},
            {'text': 'bilateral chest discomfort', 'organ_system': 'Cardiac', 'anatomical_type': 'bilateral'},
            {'text': 'central chest pain', 'organ_system': 'Cardiac', 'anatomical_type': 'midline'},
            
            # Pulmonary examples
            {'text': 'right lung pain', 'organ_system': 'Pulmonary', 'anatomical_type': 'unilateral'},
            {'text': 'bilateral chest pain', 'organ_system': 'Pulmonary', 'anatomical_type': 'bilateral'},
            {'text': 'central chest tightness', 'organ_system': 'Pulmonary', 'anatomical_type': 'midline'},
        ]

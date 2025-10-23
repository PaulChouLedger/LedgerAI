#!/usr/bin/env python3
"""
Location ML Trainer
Trains ML model for anatomical type prediction
"""

import json
import csv
import re
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

class LocationMLTrainer:
    """
    Train ML model for anatomical type prediction
    """
    
    def __init__(self, data_file: str = "location_ml_data.csv"):
        self.data_file = data_file
        self.model = None
        self.vectorizer = None
        self.feature_names = []
        
    def load_training_data(self) -> Tuple[List[Dict], List[str], List[str]]:
        """
        Load training data from CSV
        """
        print("📊 Loading training data...")
        
        data = []
        with open(self.data_file, 'r', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                data.append(row)
        
        print(f"✅ Loaded {len(data)} training examples")
        
        # Extract features and labels
        X = []
        y = []
        
        for row in data:
            # Extract anatomical features
            features = self._extract_features(row)
            X.append(features)
            y.append(row['anatomical_type'])
        
        return data, X, y
    
    def _extract_features(self, row: Dict) -> List[float]:
        """
        Extract features for ML model
        """
        features = []
        
        # 1. Anatomical features from extraction
        anatomical_features = eval(row['anatomical_features'])  # Convert string to dict
        
        # Add boolean features
        for key, value in anatomical_features.items():
            if isinstance(value, bool):
                features.append(1.0 if value else 0.0)
            elif isinstance(value, int):
                features.append(float(value))
        
        # 2. Text features from location text
        location_text = row['location_text']
        text_features = self._extract_text_features(location_text)
        features.extend(text_features)
        
        # 3. Organ system features
        organ_system = row['organ_system']
        organ_features = self._extract_organ_features(organ_system)
        features.extend(organ_features)
        
        return features
    
    def _extract_text_features(self, text: str) -> List[float]:
        """
        Extract text-based features
        """
        if not text:
            return [0.0] * 10  # Return zeros if no text
        
        text_lower = text.lower()
        
        features = [
            len(text),  # Text length
            text.count(' '),  # Word count
            text.count('right'),  # Right mentions
            text.count('left'),  # Left mentions
            text.count('bilateral'),  # Bilateral mentions
            text.count('unilateral'),  # Unilateral mentions
            text.count('midline'),  # Midline mentions
            text.count('quadrant'),  # Quadrant mentions
            text.count('chest'),  # Chest mentions
            text.count('flank')  # Flank mentions
        ]
        
        return features
    
    def _extract_organ_features(self, organ_system: str) -> List[float]:
        """
        Extract organ system features (one-hot encoding)
        """
        organ_systems = ['GI', 'CARDIO', 'PULMONARY', 'GU', 'GYN']
        features = [0.0] * len(organ_systems)
        
        if organ_system in organ_systems:
            features[organ_systems.index(organ_system)] = 1.0
        
        return features
    
    def train_model(self, X: List[List[float]], y: List[str]) -> RandomForestClassifier:
        """
        Train Random Forest model
        """
        print("🤖 Training ML model...")
        
        # Convert to numpy arrays
        X_array = np.array(X)
        y_array = np.array(y)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_array, y_array, test_size=0.2, random_state=42, stratify=y_array
        )
        
        # Train Random Forest
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=42
        )
        
        model.fit(X_train, y_train)
        
        # Evaluate model
        train_score = model.score(X_train, y_train)
        test_score = model.score(X_test, y_test)
        
        print(f"📈 Training accuracy: {train_score:.3f}")
        print(f"📈 Test accuracy: {test_score:.3f}")
        
        # Cross-validation
        cv_scores = cross_val_score(model, X_array, y_array, cv=5)
        print(f"📈 Cross-validation: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
        
        # Classification report
        y_pred = model.predict(X_test)
        print("\n📊 Classification Report:")
        print(classification_report(y_test, y_pred))
        
        # Feature importance
        feature_importance = model.feature_importances_
        print(f"\n🔍 Top 10 Most Important Features:")
        for i, importance in enumerate(sorted(feature_importance, reverse=True)[:10]):
            print(f"   {i+1}. Importance: {importance:.3f}")
        
        return model
    
    def save_model(self, model: RandomForestClassifier, output_file: str = "location_ml_model.pkl"):
        """
        Save trained model
        """
        joblib.dump(model, output_file)
        print(f"💾 Model saved to {output_file}")
    
    def load_model(self, model_file: str = "location_ml_model.pkl") -> RandomForestClassifier:
        """
        Load trained model
        """
        model = joblib.load(model_file)
        print(f"📂 Model loaded from {model_file}")
        return model
    
    def predict_anatomical_type(self, model: RandomForestClassifier, 
                               location_text: str, organ_system: str, 
                               anatomical_features: Dict) -> Dict:
        """
        Predict anatomical type for new data
        """
        # Extract features using the same method as training
        features = self._extract_features({
            'location_text': location_text,
            'organ_system': organ_system,
            'anatomical_features': str(anatomical_features)
        })
        
        # Ensure feature count matches training data
        expected_features = model.n_features_in_
        if len(features) != expected_features:
            print(f"⚠️ Feature mismatch: got {len(features)}, expected {expected_features}")
            # Pad with zeros if needed
            while len(features) < expected_features:
                features.append(0.0)
            # Truncate if too many
            features = features[:expected_features]
        
        # Make prediction
        prediction = model.predict([features])[0]
        probabilities = model.predict_proba([features])[0]
        
        # Get class names
        class_names = model.classes_
        prob_dict = {class_names[i]: probabilities[i] for i in range(len(class_names))}
        
        return {
            'predicted_type': prediction,
            'confidence': max(probabilities),
            'probabilities': prob_dict
        }

# Example usage
if __name__ == "__main__":
    trainer = LocationMLTrainer()
    
    # Load data
    data, X, y = trainer.load_training_data()
    
    if len(data) > 0:
        # Train model
        model = trainer.train_model(X, y)
        
        # Save model
        trainer.save_model(model)
        
        # Test prediction
        sample_features = {
            'has_right_quadrant': True,
            'has_left_quadrant': False,
            'has_bilateral': False,
            'has_midline': False,
            'has_chest': False,
            'has_flank': False,
            'spatial_term_count': 2
        }
        
        prediction = trainer.predict_anatomical_type(
            model, "right lower quadrant pain", "GI", sample_features
        )
        
        print(f"\n🎯 Sample Prediction:")
        print(f"   Input: 'right lower quadrant pain' (GI)")
        print(f"   Predicted: {prediction['predicted_type']}")
        print(f"   Confidence: {prediction['confidence']:.3f}")
        print(f"   Probabilities: {prediction['probabilities']}")
        
    else:
        print("❌ No training data available")

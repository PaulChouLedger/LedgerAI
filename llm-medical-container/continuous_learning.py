#!/usr/bin/env python3
"""
Continuous Learning System
Background model updates and performance monitoring
"""

import json
import os
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib

class ContinuousLearning:
    """
    Continuous learning system for ML model updates
    """
    
    def __init__(self, 
                 learning_data_dir: str = "./data/learning",
                 model_dir: str = "./data/models",
                 retrain_threshold: int = 50,
                 performance_threshold: float = 0.8):
        """
        Initialize continuous learning system
        
        Args:
            learning_data_dir: Directory containing learning data
            model_dir: Directory for storing models
            retrain_threshold: Number of new examples needed for retraining
            performance_threshold: Minimum performance threshold for model updates
        """
        self.learning_data_dir = Path(learning_data_dir)
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        self.retrain_threshold = retrain_threshold
        self.performance_threshold = performance_threshold
        
        # Learning state
        self.is_learning = False
        self.learning_thread = None
        self.current_model_version = 1
        self.last_retrain_time = None
        
        # Performance tracking
        self.performance_history = []
        self.model_versions = []
        
        # Start background learning
        self.start_background_learning()
        
        print(f"[Continuous Learning] 🧠 Initialized with retrain threshold: {retrain_threshold}")
        print(f"[Continuous Learning] 📊 Performance threshold: {performance_threshold}")
    
    def start_background_learning(self):
        """Start background learning thread"""
        if not self.is_learning:
            self.is_learning = True
            self.learning_thread = threading.Thread(target=self._background_learning_loop)
            self.learning_thread.daemon = True
            self.learning_thread.start()
            print(f"[Continuous Learning] 🔄 Background learning started")
    
    def stop_background_learning(self):
        """Stop background learning thread"""
        self.is_learning = False
        if self.learning_thread:
            self.learning_thread.join()
        print(f"[Continuous Learning] ⏹️ Background learning stopped")
    
    def _background_learning_loop(self):
        """Background learning loop"""
        while self.is_learning:
            try:
                # Check if retraining is needed
                if self._should_retrain():
                    print(f"[Continuous Learning] 🔄 Retraining triggered")
                    self._retrain_model()
                
                # Update performance metrics
                self._update_performance_metrics()
                
                # Sleep for 1 hour
                time.sleep(3600)
                
            except Exception as e:
                print(f"[Continuous Learning] ❌ Background learning error: {e}")
                time.sleep(3600)
    
    def _should_retrain(self) -> bool:
        """Check if model should be retrained"""
        try:
            # Count new feedback data
            feedback_file = self.learning_data_dir / "feedback.json"
            if not feedback_file.exists():
                return False
            
            with open(feedback_file, 'r') as f:
                feedback_data = json.load(f)
            
            # Count recent feedback (last 24 hours)
            recent_count = 0
            cutoff_time = datetime.now() - timedelta(hours=24)
            
            for feedback in feedback_data:
                feedback_time = datetime.fromisoformat(feedback['timestamp'])
                if feedback_time > cutoff_time:
                    recent_count += 1
            
            print(f"[Continuous Learning] 📊 Recent feedback: {recent_count}/{self.retrain_threshold}")
            return recent_count >= self.retrain_threshold
            
        except Exception as e:
            print(f"[Continuous Learning] ❌ Error checking retrain condition: {e}")
            return False
    
    def _retrain_model(self):
        """Retrain ML model with new data"""
        try:
            print(f"[Continuous Learning] 🔄 Starting model retraining...")
            
            # 1. Collect training data
            training_data = self._collect_training_data()
            if not training_data:
                print(f"[Continuous Learning] ⚠️ No training data available")
                return
            
            # 2. Train new model
            new_model = self._train_model(training_data)
            if not new_model:
                print(f"[Continuous Learning] ❌ Model training failed")
                return
            
            # 3. Validate performance
            performance = self._validate_model(new_model, training_data)
            if performance['accuracy'] < self.performance_threshold:
                print(f"[Continuous Learning] ⚠️ New model performance too low: {performance['accuracy']:.3f}")
                return
            
            # 4. Save new model
            new_version = self.current_model_version + 1
            model_path = self.model_dir / f"location_ml_model_v{new_version}.pkl"
            joblib.dump(new_model, model_path)
            
            # 5. Update model version
            self.current_model_version = new_version
            self.last_retrain_time = datetime.now()
            
            # 6. Save model metadata
            self._save_model_metadata(new_version, performance)
            
            print(f"[Continuous Learning] ✅ Model retrained successfully (v{new_version})")
            print(f"[Continuous Learning] 📊 Performance: {performance}")
            
        except Exception as e:
            print(f"[Continuous Learning] ❌ Retraining error: {e}")
    
    def _collect_training_data(self) -> List[Dict]:
        """Collect training data from learning files"""
        training_data = []
        
        try:
            # Load feedback data
            feedback_file = self.learning_data_dir / "feedback.json"
            if feedback_file.exists():
                with open(feedback_file, 'r') as f:
                    feedback_data = json.load(f)
                
                for feedback in feedback_data:
                    if feedback.get('accuracy') is not None:
                        training_data.append({
                            'type': 'feedback',
                            'data': feedback
                        })
            
            # Load prediction data
            predictions_file = self.learning_data_dir / "predictions.json"
            if predictions_file.exists():
                with open(predictions_file, 'r') as f:
                    predictions_data = json.load(f)
                
                for prediction in predictions_data:
                    training_data.append({
                        'type': 'prediction',
                        'data': prediction
                    })
            
            print(f"[Continuous Learning] 📊 Collected {len(training_data)} training examples")
            return training_data
            
        except Exception as e:
            print(f"[Continuous Learning] ❌ Error collecting training data: {e}")
            return []
    
    def _train_model(self, training_data: List[Dict]) -> Optional[RandomForestClassifier]:
        """Train new ML model"""
        try:
            # Extract features and labels
            X = []
            y = []
            
            for item in training_data:
                if item['type'] == 'feedback':
                    # Use feedback data for training
                    feedback = item['data']
                    if feedback.get('accuracy') is not None:
                        # Create features from feedback
                        features = self._extract_features_from_feedback(feedback)
                        if features:
                            X.append(features)
                            y.append(feedback['accuracy'])
            
            if len(X) < 10:  # Need minimum examples
                print(f"[Continuous Learning] ⚠️ Insufficient training data: {len(X)} examples")
                return None
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Train model
            model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=2,
                min_samples_leaf=1,
                random_state=42
            )
            
            model.fit(X_train, y_train)
            
            # Test performance
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            print(f"[Continuous Learning] 📊 Model accuracy: {accuracy:.3f}")
            return model
            
        except Exception as e:
            print(f"[Continuous Learning] ❌ Model training error: {e}")
            return None
    
    def _extract_features_from_feedback(self, feedback: Dict) -> List[float]:
        """Extract features from feedback data"""
        try:
            features = []
            
            # Basic features
            features.append(feedback.get('accuracy', 0.0))
            features.append(len(feedback.get('user_feedback', '')))
            features.append(len(feedback.get('condition_name', '')))
            
            # Prediction features
            prediction = feedback.get('prediction', {})
            features.append(prediction.get('similarity', 0.0))
            features.append(1.0 if prediction.get('method') == 'ml_prediction' else 0.0)
            features.append(1.0 if prediction.get('confidence') == 'high' else 0.0)
            
            # Organ system features
            organ_system = feedback.get('organ_system', '')
            organ_systems = ['GI', 'CARDIO', 'PULMONARY', 'GU', 'GYN']
            for system in organ_systems:
                features.append(1.0 if system in organ_system else 0.0)
            
            return features
            
        except Exception as e:
            print(f"[Continuous Learning] ❌ Feature extraction error: {e}")
            return []
    
    def _validate_model(self, model: RandomForestClassifier, training_data: List[Dict]) -> Dict:
        """Validate model performance"""
        try:
            # Extract test data
            X_test = []
            y_test = []
            
            for item in training_data:
                if item['type'] == 'feedback':
                    feedback = item['data']
                    if feedback.get('accuracy') is not None:
                        features = self._extract_features_from_feedback(feedback)
                        if features:
                            X_test.append(features)
                            y_test.append(feedback['accuracy'])
            
            if len(X_test) < 5:
                return {'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1_score': 0.0}
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted')
            recall = recall_score(y_test, y_pred, average='weighted')
            f1 = f1_score(y_test, y_pred, average='weighted')
            
            return {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1
            }
            
        except Exception as e:
            print(f"[Continuous Learning] ❌ Model validation error: {e}")
            return {'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1_score': 0.0}
    
    def _save_model_metadata(self, version: int, performance: Dict):
        """Save model metadata"""
        try:
            metadata = {
                'version': version,
                'timestamp': datetime.now().isoformat(),
                'performance': performance,
                'retrain_threshold': self.retrain_threshold,
                'performance_threshold': self.performance_threshold
            }
            
            metadata_file = self.model_dir / "model_metadata.json"
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            print(f"[Continuous Learning] 💾 Model metadata saved")
            
        except Exception as e:
            print(f"[Continuous Learning] ❌ Metadata save error: {e}")
    
    def _update_performance_metrics(self):
        """Update performance metrics"""
        try:
            # Load performance data
            performance_file = self.learning_data_dir / "performance.json"
            if not performance_file.exists():
                return
            
            with open(performance_file, 'r') as f:
                performance_data = json.load(f)
            
            # Calculate recent performance
            recent_data = []
            cutoff_time = datetime.now() - timedelta(hours=24)
            
            for metric in performance_data:
                metric_time = datetime.fromisoformat(metric['timestamp'])
                if metric_time > cutoff_time:
                    recent_data.append(metric)
            
            if recent_data:
                # Calculate average performance
                avg_accuracy = np.mean([m['value'] for m in recent_data if m['metric_name'] == 'accuracy'])
                
                self.performance_history.append({
                    'timestamp': datetime.now().isoformat(),
                    'accuracy': avg_accuracy,
                    'data_points': len(recent_data)
                })
                
                print(f"[Continuous Learning] 📊 Recent performance: {avg_accuracy:.3f}")
            
        except Exception as e:
            print(f"[Continuous Learning] ❌ Performance update error: {e}")
    
    def get_learning_status(self) -> Dict[str, Any]:
        """Get current learning status"""
        return {
            'is_learning': self.is_learning,
            'current_model_version': self.current_model_version,
            'last_retrain_time': self.last_retrain_time.isoformat() if self.last_retrain_time else None,
            'retrain_threshold': self.retrain_threshold,
            'performance_threshold': self.performance_threshold,
            'performance_history_count': len(self.performance_history)
        }

# Example usage
if __name__ == "__main__":
    learning = ContinuousLearning()
    
    # Get status
    status = learning.get_learning_status()
    print(f"📊 Learning Status: {status}")
    
    # Stop learning
    learning.stop_background_learning()

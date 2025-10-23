#!/usr/bin/env python3
"""
Learning Data Collector
Collects and stores learning data for continuous ML improvement
"""

import json
import os
from datetime import datetime
from typing import Dict, Any, List, Optional
from pathlib import Path
import threading
import time

class LearningDataCollector:
    """
    Collects learning data for ML model improvement
    """
    
    def __init__(self, data_dir: str = "./data/learning"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Learning data files
        self.feedback_file = self.data_dir / "feedback.json"
        self.predictions_file = self.data_dir / "predictions.json"
        self.performance_file = self.data_dir / "performance.json"
        
        # In-memory queues for real-time collection
        self.feedback_queue = []
        self.predictions_queue = []
        self.performance_queue = []
        
        # Threading for background saving
        self.save_thread = None
        self.is_running = False
        
        # Start background saving
        self.start_background_saving()
        
        print(f"[Learning] 📊 Learning data collector initialized")
        print(f"[Learning] 📁 Data directory: {self.data_dir}")
    
    def collect_prediction_feedback(self, 
                                   prediction: Dict[str, Any], 
                                   user_feedback: str, 
                                   accuracy: Optional[float] = None,
                                   condition_name: str = "",
                                   organ_system: str = "") -> None:
        """
        Collect feedback on ML predictions
        
        Args:
            prediction: ML prediction result
            user_feedback: User's response/feedback
            accuracy: Accuracy score (0.0-1.0)
            condition_name: Medical condition name
            organ_system: Organ system (GI, CARDIO, etc.)
        """
        feedback_data = {
            'timestamp': datetime.now().isoformat(),
            'prediction': prediction,
            'user_feedback': user_feedback,
            'accuracy': accuracy,
            'condition_name': condition_name,
            'organ_system': organ_system,
            'session_id': self._get_current_session_id()
        }
        
        self.feedback_queue.append(feedback_data)
        print(f"[Learning] 📝 Feedback collected: {condition_name} ({organ_system})")
    
    def collect_prediction(self, 
                          patient_text: str, 
                          guideline_text: str, 
                          condition_name: str,
                          similarity: float, 
                          method: str, 
                          confidence: str,
                          anatomical_type: str) -> None:
        """
        Collect ML prediction data
        
        Args:
            patient_text: Patient's input text
            guideline_text: Guideline text
            condition_name: Medical condition
            similarity: Similarity score (0.0-1.0)
            method: Method used (hardcoded_rule, ml_prediction, etc.)
            confidence: Confidence level (high, medium, low)
            anatomical_type: Anatomical type (bilateral, midline, etc.)
        """
        prediction_data = {
            'timestamp': datetime.now().isoformat(),
            'patient_text': patient_text,
            'guideline_text': guideline_text,
            'condition_name': condition_name,
            'similarity': similarity,
            'method': method,
            'confidence': confidence,
            'anatomical_type': anatomical_type,
            'session_id': self._get_current_session_id()
        }
        
        self.predictions_queue.append(prediction_data)
        print(f"[Learning] 🎯 Prediction collected: {condition_name} (similarity: {similarity:.3f})")
    
    def collect_performance_metric(self, 
                                 metric_name: str, 
                                 value: float, 
                                 condition_name: str = "",
                                 organ_system: str = "") -> None:
        """
        Collect performance metrics
        
        Args:
            metric_name: Name of the metric (accuracy, precision, recall, etc.)
            value: Metric value
            condition_name: Medical condition
            organ_system: Organ system
        """
        performance_data = {
            'timestamp': datetime.now().isoformat(),
            'metric_name': metric_name,
            'value': value,
            'condition_name': condition_name,
            'organ_system': organ_system,
            'session_id': self._get_current_session_id()
        }
        
        self.performance_queue.append(performance_data)
        print(f"[Learning] 📈 Performance metric: {metric_name} = {value:.3f}")
    
    def start_background_saving(self):
        """Start background thread for saving data"""
        if not self.is_running:
            self.is_running = True
            self.save_thread = threading.Thread(target=self._background_save_loop)
            self.save_thread.daemon = True
            self.save_thread.start()
            print(f"[Learning] 🔄 Background saving started")
    
    def stop_background_saving(self):
        """Stop background saving thread"""
        self.is_running = False
        if self.save_thread:
            self.save_thread.join()
        print(f"[Learning] ⏹️ Background saving stopped")
    
    def _background_save_loop(self):
        """Background loop for saving data"""
        while self.is_running:
            try:
                # Save data every 5 minutes
                time.sleep(300)
                self._save_all_data()
            except Exception as e:
                print(f"[Learning] ❌ Background save error: {e}")
                time.sleep(60)
    
    def _save_all_data(self):
        """Save all collected data to files"""
        try:
            # Save feedback data
            if self.feedback_queue:
                self._append_to_json_file(self.feedback_file, self.feedback_queue)
                self.feedback_queue.clear()
            
            # Save prediction data
            if self.predictions_queue:
                self._append_to_json_file(self.predictions_file, self.predictions_queue)
                self.predictions_queue.clear()
            
            # Save performance data
            if self.performance_queue:
                self._append_to_json_file(self.performance_file, self.performance_queue)
                self.performance_queue.clear()
            
            print(f"[Learning] 💾 Data saved to files")
        except Exception as e:
            print(f"[Learning] ❌ Save error: {e}")
    
    def _append_to_json_file(self, file_path: Path, data: List[Dict]):
        """Append data to JSON file"""
        # Load existing data
        existing_data = []
        if file_path.exists():
            try:
                with open(file_path, 'r') as f:
                    existing_data = json.load(f)
            except:
                existing_data = []
        
        # Append new data
        existing_data.extend(data)
        
        # Save back to file
        with open(file_path, 'w') as f:
            json.dump(existing_data, f, indent=2)
    
    def _get_current_session_id(self) -> str:
        """Get current session ID"""
        # Try to get from environment or use default
        return os.environ.get('SESSION_ID', 'unknown')
    
    def get_learning_stats(self) -> Dict[str, Any]:
        """Get learning statistics"""
        stats = {
            'feedback_count': len(self.feedback_queue),
            'predictions_count': len(self.predictions_queue),
            'performance_count': len(self.performance_queue),
            'files': {
                'feedback_file': str(self.feedback_file),
                'predictions_file': str(self.predictions_file),
                'performance_file': str(self.performance_file)
            }
        }
        
        # Count existing data in files
        for file_path in [self.feedback_file, self.predictions_file, self.performance_file]:
            if file_path.exists():
                try:
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                        stats[f'{file_path.stem}_file_count'] = len(data)
                except:
                    stats[f'{file_path.stem}_file_count'] = 0
        
        return stats
    
    def export_learning_data(self, output_file: str = "learning_data_export.json"):
        """Export all learning data to a single file"""
        export_data = {
            'export_timestamp': datetime.now().isoformat(),
            'feedback_data': [],
            'predictions_data': [],
            'performance_data': []
        }
        
        # Load feedback data
        if self.feedback_file.exists():
            try:
                with open(self.feedback_file, 'r') as f:
                    export_data['feedback_data'] = json.load(f)
            except:
                pass
        
        # Load predictions data
        if self.predictions_file.exists():
            try:
                with open(self.predictions_file, 'r') as f:
                    export_data['predictions_data'] = json.load(f)
            except:
                pass
        
        # Load performance data
        if self.performance_file.exists():
            try:
                with open(self.performance_file, 'r') as f:
                    export_data['performance_data'] = json.load(f)
            except:
                pass
        
        # Save export file
        export_path = self.data_dir / output_file
        with open(export_path, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        print(f"[Learning] 📤 Learning data exported to {export_path}")
        return export_path

# Example usage
if __name__ == "__main__":
    collector = LearningDataCollector()
    
    # Test data collection
    collector.collect_prediction_feedback(
        prediction={'similarity': 0.8, 'method': 'ml_prediction'},
        user_feedback="The prediction was accurate",
        accuracy=0.9,
        condition_name="Acute Appendicitis",
        organ_system="GI"
    )
    
    collector.collect_prediction(
        patient_text="right lower quadrant pain",
        guideline_text="right lower quadrant pain",
        condition_name="Acute Appendicitis",
        similarity=0.8,
        method="hardcoded_rule",
        confidence="high",
        anatomical_type="right_only"
    )
    
    collector.collect_performance_metric(
        metric_name="accuracy",
        value=0.85,
        condition_name="Acute Appendicitis",
        organ_system="GI"
    )
    
    # Get stats
    stats = collector.get_learning_stats()
    print(f"\n📊 Learning Stats: {stats}")
    
    # Export data
    export_path = collector.export_learning_data()
    print(f"📤 Data exported to: {export_path}")
    
    # Stop background saving
    collector.stop_background_saving()

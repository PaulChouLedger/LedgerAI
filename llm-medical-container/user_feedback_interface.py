#!/usr/bin/env python3
"""
User Feedback Interface
Rating system for ML predictions and user feedback collection
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
import threading
import time

class UserFeedbackInterface:
    """
    User feedback interface for collecting ratings and feedback
    """
    
    def __init__(self, 
                 data_dir: str = "./data/learning",
                 feedback_file: str = "user_feedback.json",
                 rating_scale: int = 5):
        """
        Initialize user feedback interface
        
        Args:
            data_dir: Directory for storing feedback data
            feedback_file: File to store user feedback
            rating_scale: Rating scale (1-5 stars)
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.feedback_file = self.data_dir / feedback_file
        self.rating_scale = rating_scale
        
        # Feedback collection
        self.feedback_queue = []
        self.is_collecting = False
        self.collection_thread = None
        
        # Start feedback collection
        self.start_feedback_collection()
        
        print(f"[User Feedback] 💬 Initialized with {rating_scale}-star rating scale")
    
    def start_feedback_collection(self):
        """Start feedback collection thread"""
        if not self.is_collecting:
            self.is_collecting = True
            self.collection_thread = threading.Thread(target=self._feedback_collection_loop)
            self.collection_thread.daemon = True
            self.collection_thread.start()
            print(f"[User Feedback] 🔄 Feedback collection started")
    
    def stop_feedback_collection(self):
        """Stop feedback collection thread"""
        self.is_collecting = False
        if self.collection_thread:
            self.collection_thread.join()
        print(f"[User Feedback] ⏹️ Feedback collection stopped")
    
    def _feedback_collection_loop(self):
        """Background feedback collection loop"""
        while self.is_collecting:
            try:
                # Process feedback queue
                self._process_feedback_queue()
                
                # Sleep for 30 seconds
                time.sleep(30)
                
            except Exception as e:
                print(f"[User Feedback] ❌ Collection error: {e}")
                time.sleep(60)
    
    def _process_feedback_queue(self):
        """Process feedback queue and save to file"""
        if self.feedback_queue:
            try:
                # Load existing feedback
                existing_feedback = []
                if self.feedback_file.exists():
                    with open(self.feedback_file, 'r') as f:
                        existing_feedback = json.load(f)
                
                # Add new feedback
                existing_feedback.extend(self.feedback_queue)
                
                # Save to file
                with open(self.feedback_file, 'w') as f:
                    json.dump(existing_feedback, f, indent=2)
                
                print(f"[User Feedback] 💾 Saved {len(self.feedback_queue)} feedback items")
                self.feedback_queue.clear()
                
            except Exception as e:
                print(f"[User Feedback] ❌ Feedback processing error: {e}")
    
    def collect_prediction_rating(self, 
                                 prediction_id: str,
                                 prediction: Dict[str, Any],
                                 user_rating: int,
                                 user_comment: str = "",
                                 condition_name: str = "",
                                 organ_system: str = "") -> bool:
        """
        Collect user rating for ML prediction
        
        Args:
            prediction_id: Unique identifier for prediction
            prediction: ML prediction result
            user_rating: User rating (1-5 stars)
            user_comment: Optional user comment
            condition_name: Medical condition name
            organ_system: Organ system
            
        Returns:
            bool: True if feedback collected successfully
        """
        try:
            # Validate rating
            if not (1 <= user_rating <= self.rating_scale):
                print(f"[User Feedback] ❌ Invalid rating: {user_rating} (must be 1-{self.rating_scale})")
                return False
            
            # Create feedback entry
            feedback_entry = {
                'prediction_id': prediction_id,
                'timestamp': datetime.now().isoformat(),
                'prediction': prediction,
                'user_rating': user_rating,
                'user_comment': user_comment,
                'condition_name': condition_name,
                'organ_system': organ_system,
                'session_id': self._get_current_session_id(),
                'feedback_type': 'prediction_rating'
            }
            
            # Add to queue
            self.feedback_queue.append(feedback_entry)
            
            print(f"[User Feedback] ⭐ Rating collected: {user_rating}/5 stars ({condition_name})")
            return True
            
        except Exception as e:
            print(f"[User Feedback] ❌ Rating collection error: {e}")
            return False
    
    def collect_accuracy_feedback(self, 
                                 prediction_id: str,
                                 predicted_accuracy: float,
                                 actual_accuracy: float,
                                 user_comment: str = "",
                                 condition_name: str = "",
                                 organ_system: str = "") -> bool:
        """
        Collect accuracy feedback for ML prediction
        
        Args:
            prediction_id: Unique identifier for prediction
            predicted_accuracy: Predicted accuracy score
            actual_accuracy: Actual accuracy score
            user_comment: Optional user comment
            condition_name: Medical condition name
            organ_system: Organ system
            
        Returns:
            bool: True if feedback collected successfully
        """
        try:
            # Create feedback entry
            feedback_entry = {
                'prediction_id': prediction_id,
                'timestamp': datetime.now().isoformat(),
                'predicted_accuracy': predicted_accuracy,
                'actual_accuracy': actual_accuracy,
                'accuracy_difference': abs(predicted_accuracy - actual_accuracy),
                'user_comment': user_comment,
                'condition_name': condition_name,
                'organ_system': organ_system,
                'session_id': self._get_current_session_id(),
                'feedback_type': 'accuracy_feedback'
            }
            
            # Add to queue
            self.feedback_queue.append(feedback_entry)
            
            print(f"[User Feedback] 📊 Accuracy feedback collected: {actual_accuracy:.3f} ({condition_name})")
            return True
            
        except Exception as e:
            print(f"[User Feedback] ❌ Accuracy feedback error: {e}")
            return False
    
    def collect_general_feedback(self, 
                                 feedback_type: str,
                                 user_comment: str,
                                 rating: Optional[int] = None,
                                 condition_name: str = "",
                                 organ_system: str = "") -> bool:
        """
        Collect general user feedback
        
        Args:
            feedback_type: Type of feedback (suggestion, bug_report, feature_request, etc.)
            user_comment: User comment
            rating: Optional rating (1-5 stars)
            condition_name: Medical condition name
            organ_system: Organ system
            
        Returns:
            bool: True if feedback collected successfully
        """
        try:
            # Create feedback entry
            feedback_entry = {
                'timestamp': datetime.now().isoformat(),
                'feedback_type': feedback_type,
                'user_comment': user_comment,
                'rating': rating,
                'condition_name': condition_name,
                'organ_system': organ_system,
                'session_id': self._get_current_session_id(),
                'feedback_category': 'general_feedback'
            }
            
            # Add to queue
            self.feedback_queue.append(feedback_entry)
            
            print(f"[User Feedback] 💬 General feedback collected: {feedback_type} ({condition_name})")
            return True
            
        except Exception as e:
            print(f"[User Feedback] ❌ General feedback error: {e}")
            return False
    
    def collect_system_feedback(self, 
                               system_component: str,
                               performance_rating: int,
                               user_comment: str = "",
                               condition_name: str = "",
                               organ_system: str = "") -> bool:
        """
        Collect system performance feedback
        
        Args:
            system_component: System component (diagnostic_engine, ml_model, etc.)
            performance_rating: Performance rating (1-5 stars)
            user_comment: Optional user comment
            condition_name: Medical condition name
            organ_system: Organ system
            
        Returns:
            bool: True if feedback collected successfully
        """
        try:
            # Validate rating
            if not (1 <= performance_rating <= self.rating_scale):
                print(f"[User Feedback] ❌ Invalid rating: {performance_rating} (must be 1-{self.rating_scale})")
                return False
            
            # Create feedback entry
            feedback_entry = {
                'timestamp': datetime.now().isoformat(),
                'system_component': system_component,
                'performance_rating': performance_rating,
                'user_comment': user_comment,
                'condition_name': condition_name,
                'organ_system': organ_system,
                'session_id': self._get_current_session_id(),
                'feedback_type': 'system_feedback'
            }
            
            # Add to queue
            self.feedback_queue.append(feedback_entry)
            
            print(f"[User Feedback] 🔧 System feedback collected: {system_component} ({performance_rating}/5 stars)")
            return True
            
        except Exception as e:
            print(f"[User Feedback] ❌ System feedback error: {e}")
            return False
    
    def get_feedback_summary(self) -> Dict[str, Any]:
        """Get feedback summary"""
        try:
            # Load existing feedback
            existing_feedback = []
            if self.feedback_file.exists():
                with open(self.feedback_file, 'r') as f:
                    existing_feedback = json.load(f)
            
            # Calculate summary
            total_feedback = len(existing_feedback) + len(self.feedback_queue)
            
            # Count by type
            feedback_types = {}
            for feedback in existing_feedback:
                feedback_type = feedback.get('feedback_type', 'unknown')
                feedback_types[feedback_type] = feedback_types.get(feedback_type, 0) + 1
            
            # Count by rating
            ratings = [f.get('user_rating', 0) for f in existing_feedback if f.get('user_rating')]
            avg_rating = sum(ratings) / len(ratings) if ratings else 0.0
            
            return {
                'total_feedback': total_feedback,
                'pending_feedback': len(self.feedback_queue),
                'feedback_types': feedback_types,
                'average_rating': avg_rating,
                'rating_scale': self.rating_scale,
                'is_collecting': self.is_collecting
            }
            
        except Exception as e:
            print(f"[User Feedback] ❌ Summary error: {e}")
            return {}
    
    def _get_current_session_id(self) -> str:
        """Get current session ID"""
        return os.environ.get('SESSION_ID', 'unknown')
    
    def export_feedback_data(self, output_file: str = "user_feedback_export.json") -> Path:
        """Export all feedback data"""
        try:
            # Load existing feedback
            existing_feedback = []
            if self.feedback_file.exists():
                with open(self.feedback_file, 'r') as f:
                    existing_feedback = json.load(f)
            
            # Add pending feedback
            all_feedback = existing_feedback + self.feedback_queue
            
            # Create export data
            export_data = {
                'export_timestamp': datetime.now().isoformat(),
                'total_feedback': len(all_feedback),
                'feedback_data': all_feedback
            }
            
            # Save export file
            export_path = self.data_dir / output_file
            with open(export_path, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            print(f"[User Feedback] 📤 Feedback data exported to {export_path}")
            return export_path
            
        except Exception as e:
            print(f"[User Feedback] ❌ Export error: {e}")
            return None

# Example usage
if __name__ == "__main__":
    feedback_interface = UserFeedbackInterface()
    
    # Test feedback collection
    feedback_interface.collect_prediction_rating(
        prediction_id="pred_123",
        prediction={'similarity': 0.8, 'method': 'ml_prediction'},
        user_rating=4,
        user_comment="The prediction was accurate",
        condition_name="Acute Appendicitis",
        organ_system="GI"
    )
    
    feedback_interface.collect_accuracy_feedback(
        prediction_id="pred_123",
        predicted_accuracy=0.8,
        actual_accuracy=0.85,
        user_comment="Close prediction",
        condition_name="Acute Appendicitis",
        organ_system="GI"
    )
    
    feedback_interface.collect_general_feedback(
        feedback_type="suggestion",
        user_comment="The system could be faster",
        rating=3,
        condition_name="Acute Appendicitis",
        organ_system="GI"
    )
    
    feedback_interface.collect_system_feedback(
        system_component="diagnostic_engine",
        performance_rating=4,
        user_comment="Good performance overall",
        condition_name="Acute Appendicitis",
        organ_system="GI"
    )
    
    # Get summary
    summary = feedback_interface.get_feedback_summary()
    print(f"📊 Feedback Summary: {summary}")
    
    # Export data
    export_path = feedback_interface.export_feedback_data()
    print(f"📤 Data exported to: {export_path}")
    
    # Stop collection
    feedback_interface.stop_feedback_collection()

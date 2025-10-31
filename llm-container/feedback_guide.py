#!/usr/bin/env python3
"""
Feedback Guide - How to provide feedback to the learning system
"""

from adaptive_diagnostic_engine import AdaptiveDiagnosticEngine
import uuid

class FeedbackGuide:
    """
    Guide for providing feedback to the learning system
    """
    
    def __init__(self):
        self.engine = None
        self.session_id = str(uuid.uuid4())
        
    def initialize_engine(self):
        """Initialize the diagnostic engine"""
        try:
            self.engine = AdaptiveDiagnosticEngine()
            print("✅ Diagnostic engine initialized")
            return True
        except Exception as e:
            print(f"❌ Failed to initialize engine: {e}")
            return False
    
    def provide_prediction_feedback(self, 
                                  patient_text: str,
                                  guideline_text: str,
                                  condition_name: str,
                                  user_rating: int,
                                  user_comment: str = ""):
        """
        Provide feedback on ML prediction
        
        Args:
            patient_text: Patient's input text
            guideline_text: Guideline text
            condition_name: Medical condition name
            user_rating: User rating (1-5 stars)
            user_comment: Optional user comment
        """
        if not self.engine:
            print("❌ Engine not initialized")
            return False
        
        try:
            # Generate prediction ID
            prediction_id = f"pred_{uuid.uuid4().hex[:8]}"
            
            # Get ML prediction
            similarity = self.engine._compute_enhanced_location_similarity(
                patient_text, guideline_text, condition_name
            )
            
            # Create prediction result
            prediction = {
                'similarity': similarity,
                'patient_text': patient_text,
                'guideline_text': guideline_text,
                'condition_name': condition_name
            }
            
            # Collect user feedback
            success = self.engine.collect_user_feedback(
                prediction_id=prediction_id,
                prediction=prediction,
                user_rating=user_rating,
                user_comment=user_comment,
                condition_name=condition_name
            )
            
            if success:
                print(f"✅ Feedback collected: {condition_name} - {user_rating}/5 stars")
                print(f"   Comment: {user_comment}")
                return True
            else:
                print("❌ Failed to collect feedback")
                return False
                
        except Exception as e:
            print(f"❌ Error providing feedback: {e}")
            return False
    
    def provide_accuracy_feedback(self,
                                 condition_name: str,
                                 predicted_accuracy: float,
                                 actual_accuracy: float,
                                 user_comment: str = ""):
        """
        Provide accuracy feedback
        
        Args:
            condition_name: Medical condition name
            predicted_accuracy: Predicted accuracy score
            actual_accuracy: Actual accuracy score
            user_comment: Optional user comment
        """
        if not self.engine:
            print("❌ Engine not initialized")
            return False
        
        try:
            # Generate prediction ID
            prediction_id = f"acc_{uuid.uuid4().hex[:8]}"
            
            # Collect accuracy feedback
            success = self.engine.collect_accuracy_feedback(
                prediction_id=prediction_id,
                predicted_accuracy=predicted_accuracy,
                actual_accuracy=actual_accuracy,
                user_comment=user_comment,
                condition_name=condition_name
            )
            
            if success:
                print(f"✅ Accuracy feedback collected: {condition_name}")
                print(f"   Predicted: {predicted_accuracy:.3f}, Actual: {actual_accuracy:.3f}")
                print(f"   Comment: {user_comment}")
                return True
            else:
                print("❌ Failed to collect accuracy feedback")
                return False
                
        except Exception as e:
            print(f"❌ Error providing accuracy feedback: {e}")
            return False
    
    def get_learning_status(self):
        """Get current learning system status"""
        if not self.engine:
            print("❌ Engine not initialized")
            return None
        
        try:
            status = self.engine.get_learning_status()
            print("📊 Learning System Status:")
            print(f"  Medical Rule Engine: {'✅' if status['medical_rule_engine'] else '❌'}")
            print(f"  Learning Collector: {'✅' if status['learning_collector'] else '❌'}")
            print(f"  Continuous Learning: {'✅' if status['continuous_learning'] else '❌'}")
            print(f"  Performance Monitor: {'✅' if status['performance_monitor'] else '❌'}")
            print(f"  User Feedback: {'✅' if status['user_feedback'] else '❌'}")
            
            return status
            
        except Exception as e:
            print(f"❌ Error getting status: {e}")
            return None

# Example usage and demonstration
def demonstrate_feedback_system():
    """Demonstrate how to use the feedback system"""
    print("🎯 FEEDBACK SYSTEM DEMONSTRATION")
    print("=" * 50)
    
    # Initialize feedback guide
    guide = FeedbackGuide()
    
    # Initialize engine
    if not guide.initialize_engine():
        return
    
    print("\n1. PROVIDING PREDICTION FEEDBACK")
    print("-" * 30)
    
    # Example 1: Good prediction
    guide.provide_prediction_feedback(
        patient_text="right lower quadrant pain",
        guideline_text="right lower quadrant pain",
        condition_name="Acute Appendicitis",
        user_rating=5,
        user_comment="Excellent prediction - very accurate"
    )
    
    # Example 2: Poor prediction
    guide.provide_prediction_feedback(
        patient_text="left sided pain",
        guideline_text="right lower quadrant pain",
        condition_name="Acute Appendicitis",
        user_rating=2,
        user_comment="Wrong side - should be right side for appendicitis"
    )
    
    print("\n2. PROVIDING ACCURACY FEEDBACK")
    print("-" * 30)
    
    # Example 1: Good accuracy
    guide.provide_accuracy_feedback(
        condition_name="Acute Appendicitis",
        predicted_accuracy=0.8,
        actual_accuracy=0.85,
        user_comment="Close prediction, very good"
    )
    
    # Example 2: Poor accuracy
    guide.provide_accuracy_feedback(
        condition_name="Acute Cholecystitis",
        predicted_accuracy=0.9,
        actual_accuracy=0.6,
        user_comment="Overestimated accuracy significantly"
    )
    
    print("\n3. CHECKING LEARNING STATUS")
    print("-" * 30)
    
    # Get learning status
    guide.get_learning_status()
    
    print("\n✅ FEEDBACK SYSTEM DEMONSTRATION COMPLETE!")
    print("💡 The system will now use this feedback to improve future predictions")

if __name__ == "__main__":
    demonstrate_feedback_system()

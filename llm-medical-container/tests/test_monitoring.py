#!/usr/bin/env python3
"""
Test script to demonstrate monitoring system
"""

import json
import os
from pathlib import Path

def create_test_data():
    """Create sample test data to demonstrate monitoring"""
    
    # Create data directory - use the correct path for monitoring system
    data_dir = Path("../data/learning")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Sample predictions data
    predictions_data = [
        {
            "timestamp": "2025-10-22T20:00:00Z",
            "patient_text": "right lower quadrant pain",
            "guideline_text": "right lower quadrant pain",
            "condition_name": "Acute Appendicitis",
            "similarity": 0.95,
            "method": "ml_prediction",
            "confidence": 0.88,
            "anatomical_type": "right_only",
            "organ_system": "GI"
        },
        {
            "timestamp": "2025-10-22T20:05:00Z",
            "patient_text": "left sided pain",
            "guideline_text": "right upper quadrant pain",
            "condition_name": "Acute Cholecystitis",
            "similarity": 0.0,
            "method": "hardcoded_rules",
            "confidence": 0.95,
            "anatomical_type": "right_only",
            "organ_system": "GI"
        }
    ]
    
    # Sample feedback data
    feedback_data = [
        {
            "timestamp": "2025-10-22T20:10:00Z",
            "prediction_id": "pred_001",
            "user_rating": 5,
            "user_comment": "Excellent prediction",
            "condition_name": "Acute Appendicitis",
            "organ_system": "GI"
        },
        {
            "timestamp": "2025-10-22T20:15:00Z",
            "prediction_id": "pred_002",
            "user_rating": 4,
            "user_comment": "Good anatomical exclusion",
            "condition_name": "Acute Cholecystitis",
            "organ_system": "GI"
        }
    ]
    
    # Sample performance data
    performance_data = [
        {
            "timestamp": "2025-10-22T20:00:00Z",
            "prediction": 0.95,
            "confidence": 0.88,
            "method": "ml_prediction",
            "condition_name": "Acute Appendicitis",
            "organ_system": "GI",
            "accuracy": 0.92
        },
        {
            "timestamp": "2025-10-22T20:05:00Z",
            "prediction": 0.0,
            "confidence": 0.95,
            "method": "hardcoded_rules",
            "condition_name": "Acute Cholecystitis",
            "organ_system": "GI",
            "accuracy": 0.98
        }
    ]
    
    # Save data files
    with open(data_dir / "predictions.json", "w") as f:
        json.dump(predictions_data, f, indent=2)
    
    with open(data_dir / "feedback.json", "w") as f:
        json.dump(feedback_data, f, indent=2)
    
    with open(data_dir / "performance.json", "w") as f:
        json.dump(performance_data, f, indent=2)
    
    print("✅ Test data created successfully!")
    print(f"📁 Data directory: {data_dir.absolute()}")
    print(f"📊 Files created: predictions.json, feedback.json, performance.json")

def main():
    """Main test function"""
    print("🧪 Testing Monitoring System")
    print("=" * 50)
    
    # Create test data
    create_test_data()
    
    print("\n📊 Now run monitoring scripts:")
    print("python learning_tracker.py")
    print("python performance_dashboard.py")
    print("python feedback_guide.py")

if __name__ == "__main__":
    main()

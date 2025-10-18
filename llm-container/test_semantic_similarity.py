#!/usr/bin/env python3

import sys
import os
sys.path.append('/app')

# Import the adaptive diagnostic engine
from adaptive_diagnostic_engine import AdaptiveDiagnosticEngine
import numpy as np

def test_location_similarity():
    print("🧪 Testing Semantic Similarity for Location Matching")
    print("=" * 60)
    
    # Test cases
    test_cases = [
        {
            "user_response": "left lower belly pain towards my pelvis",
            "guideline_location": "LEFT LOWER QUADRANT (LLQ) - key differentiator from appendicitis (RLQ). LOCALIZED and CONSTANT.",
            "expected": "High similarity (should match Diverticulitis)"
        },
        {
            "user_response": "left lower belly pain towards my pelvis", 
            "guideline_location": "EPIGASTRIC, midline upper abdomen. No radiation typically.",
            "expected": "Low similarity (should NOT match Peptic Ulcer)"
        },
        {
            "user_response": "upper right side under my ribs",
            "guideline_location": "RIGHT UPPER QUADRANT (RUQ) pain with CVA tenderness.",
            "expected": "High similarity (should match Cholecystitis)"
        },
        {
            "user_response": "upper right side under my ribs",
            "guideline_location": "LEFT LOWER QUADRANT (LLQ) - key differentiator from appendicitis (RLQ).",
            "expected": "Low similarity (should NOT match Diverticulitis)"
        }
    ]
    
    try:
        # Initialize the engine (this will load the embedding model)
        print("🔄 Initializing Adaptive Diagnostic Engine...")
        engine = AdaptiveDiagnosticEngine()
        
        if not engine.embedding_model:
            print("❌ Embedding model not available - cannot test similarity")
            return
            
        print("✅ Embedding model loaded successfully")
        print()
        
        # Test each case
        for i, case in enumerate(test_cases, 1):
            print(f"Test {i}: {case['expected']}")
            print(f"User: \"{case['user_response']}\"")
            print(f"Guideline: \"{case['guideline_location']}\"")
            
            try:
                similarity = engine._compute_similarity(case['user_response'], case['guideline_location'])
                print(f"Similarity: {similarity:.3f}")
                print(f"Threshold: 0.75")
                print(f"Match: {'✅ YES' if similarity > 0.75 else '❌ NO'}")
                
                if similarity > 0.75:
                    print("🎯 This should rank HIGH in differential diagnosis")
                else:
                    print("📉 This should rank LOW in differential diagnosis")
                    
            except Exception as e:
                print(f"❌ Error computing similarity: {e}")
            
            print("-" * 60)
        
        print("\n🎯 Expected Results:")
        print("- 'left lower belly pain towards my pelvis' should match Diverticulitis (LLQ)")
        print("- 'left lower belly pain towards my pelvis' should NOT match Peptic Ulcer (epigastric)")
        print("- This should fix the ranking issue where Diverticulitis was #5 instead of #1")
        
    except Exception as e:
        print(f"❌ Error initializing engine: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_location_similarity()

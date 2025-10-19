#!/usr/bin/env python3

import sys
import os
sys.path.append('/app')

# Import the adaptive diagnostic engine and RAG embedding API
from adaptive_diagnostic_engine import AdaptiveDiagnosticEngine
from unified_medical_mode import RAGEmbeddingAPI

def test_complex_descriptions():
    print("🧪 Testing Complex Description Handling")
    print("=" * 70)
    
    try:
        # Initialize the embedding API first
        print("🔄 Initializing RAG Embedding API...")
        embedding_api = RAGEmbeddingAPI()
        
        # Initialize the engine with the embedding model
        print("🔄 Initializing Adaptive Diagnostic Engine...")
        engine = AdaptiveDiagnosticEngine(embedding_model=embedding_api)
        
        if not engine.embedding_model:
            print("❌ Embedding model not available - cannot test similarity")
            return
            
        print("✅ Embedding model loaded successfully")
        print()
        
        # Test cases for complex descriptions
        test_cases = [
            # Complex descriptions that should be ACCEPTED
            {
                "user_response": "sharp pain in the lower left part of my belly that goes towards my pelvis",
                "guideline_location": "LEFT LOWER QUADRANT (LLQ) - key differentiator from appendicitis (RLQ). LOCALIZED and CONSTANT.",
                "expected": "ACCEPT (Complex Diverticulitis - should get anatomical bonus)"
            },
            {
                "user_response": "dull ache in my upper right side just below my ribcage",
                "guideline_location": "RIGHT UPPER QUADRANT (RUQ) pain with CVA tenderness.",
                "expected": "ACCEPT (Complex Cholecystitis - should get anatomical bonus)"
            },
            {
                "user_response": "severe cramping pain in the lower left abdomen that radiates to my back",
                "guideline_location": "LEFT LOWER QUADRANT (LLQ) - key differentiator from appendicitis (RLQ). LOCALIZED and CONSTANT.",
                "expected": "ACCEPT (Complex Diverticulitis - should get anatomical bonus)"
            },
            {
                "user_response": "burning sensation in the upper right part of my stomach under the ribs",
                "guideline_location": "RIGHT UPPER QUADRANT (RUQ) pain with CVA tenderness.",
                "expected": "ACCEPT (Complex Cholecystitis - should get anatomical bonus)"
            },
            
            # Simple descriptions for comparison
            {
                "user_response": "left lower quadrant pain",
                "guideline_location": "LEFT LOWER QUADRANT (LLQ) - key differentiator from appendicitis (RLQ). LOCALIZED and CONSTANT.",
                "expected": "ACCEPT (Simple Diverticulitis - should get high boost)"
            },
            {
                "user_response": "upper right abdomen",
                "guideline_location": "RIGHT UPPER QUADRANT (RUQ) pain with CVA tenderness.",
                "expected": "ACCEPT (Simple Cholecystitis - should get high boost)"
            },
            
            # Wrong locations that should be REJECTED
            {
                "user_response": "sharp pain in the lower left part of my belly that goes towards my pelvis",
                "guideline_location": "EPIGASTRIC, midline upper abdomen. No radiation typically.",
                "expected": "REJECT (Wrong location - no anatomical match)"
            },
            {
                "user_response": "dull ache in my upper right side just below my ribcage",
                "guideline_location": "LEFT LOWER QUADRANT (LLQ) - key differentiator from appendicitis (RLQ).",
                "expected": "REJECT (Wrong location - no anatomical match)"
            }
        ]
        
        # Test each case
        accepted_scores = []
        rejected_scores = []
        
        for i, case in enumerate(test_cases, 1):
            print(f"Test {i}: {case['expected']}")
            print(f"User: \"{case['user_response']}\"")
            print(f"Guideline: \"{case['guideline_location']}\"")
            
            try:
                similarity = engine._compute_enhanced_location_similarity(
                    case['user_response'], 
                    case['guideline_location']
                )
                print(f"Final similarity: {similarity:.3f}")
                
                # Test against threshold
                threshold = 0.70
                if similarity > threshold:
                    accepted_scores.append(similarity)
                    print(f"✅ ACCEPTED (>{threshold})")
                else:
                    rejected_scores.append(similarity)
                    print(f"❌ REJECTED (<{threshold})")
                    
            except Exception as e:
                print(f"❌ Error computing similarity: {e}")
            
            print("-" * 70)
        
        # Analyze results
        print("\n📊 COMPLEX DESCRIPTION ANALYSIS:")
        print("=" * 50)
        
        if accepted_scores:
            print(f"✅ ACCEPTED scores: {[f'{s:.3f}' for s in accepted_scores]}")
            print(f"   Min: {min(accepted_scores):.3f}, Max: {max(accepted_scores):.3f}, Avg: {sum(accepted_scores)/len(accepted_scores):.3f}")
        
        if rejected_scores:
            print(f"❌ REJECTED scores: {[f'{s:.3f}' for s in rejected_scores]}")
            print(f"   Min: {min(rejected_scores):.3f}, Max: {max(rejected_scores):.3f}, Avg: {sum(rejected_scores)/len(rejected_scores):.3f}")
        
        # Check if complex descriptions are properly handled
        print(f"\n🎯 COMPLEX DESCRIPTION HANDLING:")
        print(f"   Complex descriptions should get anatomical bonus (+0.05)")
        print(f"   This should push them above 0.70 threshold")
        print(f"   Wrong locations should remain below threshold")
        
        print("\n🎯 SUMMARY:")
        print("Enhanced keyword matching should:")
        print("✅ Give anatomical bonus for 'left', 'right', 'upper', 'lower', 'quadrant'")
        print("✅ Handle complex descriptions with extra words")
        print("✅ Accept precise location matches regardless of complexity")
        print("❌ Reject wrong locations even with complex descriptions")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_complex_descriptions()

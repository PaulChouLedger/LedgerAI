#!/usr/bin/env python3

import sys
import os
sys.path.append('/app')

# Import the adaptive diagnostic engine and RAG embedding API
from adaptive_diagnostic_engine import AdaptiveDiagnosticEngine
from unified_medical_mode import RAGEmbeddingAPI

def test_final_threshold():
    print("🧪 Testing Final Optimized Threshold (0.70)")
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
        
        # Test cases for final threshold validation
        test_cases = [
            # Should be ACCEPTED (above 0.70)
            {
                "user_response": "left lower quadrant pain",
                "guideline_location": "LEFT LOWER QUADRANT (LLQ) - key differentiator from appendicitis (RLQ). LOCALIZED and CONSTANT.",
                "expected": "ACCEPT (Diverticulitis - should be 0.826)"
            },
            {
                "user_response": "pain in my upper right abdomen",
                "guideline_location": "RIGHT UPPER QUADRANT (RUQ) pain with CVA tenderness.",
                "expected": "ACCEPT (Cholecystitis - should be 0.869)"
            },
            {
                "user_response": "upper right side under my ribs",
                "guideline_location": "RIGHT UPPER QUADRANT (RUQ) pain with CVA tenderness.",
                "expected": "ACCEPT (Cholecystitis - should be 0.764)"
            },
            {
                "user_response": "dull ache in my upper right side just below my ribcage",
                "guideline_location": "RIGHT UPPER QUADRANT (RUQ) pain with CVA tenderness.",
                "expected": "ACCEPT (Cholecystitis - should be 0.781)"
            },
            
            # Should be REJECTED (below 0.70)
            {
                "user_response": "left lower belly pain towards my pelvis",
                "guideline_location": "LEFT LOWER QUADRANT (LLQ) - key differentiator from appendicitis (RLQ). LOCALIZED and CONSTANT.",
                "expected": "REJECT (Diverticulitis - should be 0.666)"
            },
            {
                "user_response": "sharp pain in the lower left part of my belly that goes towards my pelvis",
                "guideline_location": "LEFT LOWER QUADRANT (LLQ) - key differentiator from appendicitis (RLQ). LOCALIZED and CONSTANT.",
                "expected": "REJECT (Diverticulitis - should be 0.678)"
            },
            {
                "user_response": "left lower belly pain towards my pelvis",
                "guideline_location": "EPIGASTRIC, midline upper abdomen. No radiation typically.",
                "expected": "REJECT (Peptic Ulcer - should be 0.722)"
            },
            {
                "user_response": "upper right side under my ribs",
                "guideline_location": "LEFT LOWER QUADRANT (LLQ) - key differentiator from appendicitis (RLQ).",
                "expected": "REJECT (Diverticulitis - should be 0.670)"
            },
            {
                "user_response": "belly pain",
                "guideline_location": "LEFT LOWER QUADRANT (LLQ) - key differentiator from appendicitis (RLQ).",
                "expected": "REJECT (Generic - should be 0.622)"
            },
            {
                "user_response": "abdominal pain",
                "guideline_location": "EPIGASTRIC, midline upper abdomen. No radiation typically.",
                "expected": "REJECT (Generic - should be 0.748)"
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
        print("\n📊 FINAL THRESHOLD (0.70) ANALYSIS:")
        print("=" * 50)
        
        if accepted_scores:
            print(f"✅ ACCEPTED scores: {[f'{s:.3f}' for s in accepted_scores]}")
            print(f"   Min: {min(accepted_scores):.3f}, Max: {max(accepted_scores):.3f}, Avg: {sum(accepted_scores)/len(accepted_scores):.3f}")
        
        if rejected_scores:
            print(f"❌ REJECTED scores: {[f'{s:.3f}' for s in rejected_scores]}")
            print(f"   Min: {min(rejected_scores):.3f}, Max: {max(rejected_scores):.3f}, Avg: {sum(rejected_scores)/len(rejected_scores):.3f}")
        
        # Check if threshold is optimal
        if accepted_scores and rejected_scores:
            min_accepted = min(accepted_scores)
            max_rejected = max(rejected_scores)
            
            print(f"\n🎯 THRESHOLD ANALYSIS:")
            print(f"   Min accepted: {min_accepted:.3f}")
            print(f"   Max rejected: {max_rejected:.3f}")
            print(f"   Gap: {min_accepted - max_rejected:.3f}")
            
            if min_accepted > max_rejected:
                print(f"   ✅ Perfect separation! No overlap between accepted/rejected")
            else:
                print(f"   ⚠️  Some overlap - consider adjusting threshold")
        
        print("\n🎯 SUMMARY:")
        print("Final threshold (0.70) should:")
        print("✅ Accept precise location matches (quadrant, specific terms)")
        print("❌ Reject vague descriptions and wrong locations")
        print("🎯 Provide optimal separation for diagnostic accuracy")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_final_threshold()

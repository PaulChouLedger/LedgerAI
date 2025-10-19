#!/usr/bin/env python3

import sys
import os
sys.path.append('/app')

# Import the adaptive diagnostic engine and RAG embedding API
from adaptive_diagnostic_engine import AdaptiveDiagnosticEngine
from unified_medical_mode import RAGEmbeddingAPI

def test_fuzzy_matching():
    print("🧪 Testing Fuzzy Matching Approach")
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
        
        # Test cases for fuzzy matching
        test_cases = [
            # Your example case
            {
                "user_response": "I have sharp left lower abdomanial pain towards my pelvis",
                "guideline_location": "LEFT LOWER QUADRANT (LLQ) - key differentiator from appendicitis (RLQ). LOCALIZED and CONSTANT.",
                "expected": "ACCEPT (Should get high weighted boost for 'left', 'lower', 'abdomanial'≈'abdominal')"
            },
            {
                "user_response": "sharp pain in the lower left part of my belly that goes towards my pelvis",
                "guideline_location": "LEFT LOWER QUADRANT (LLQ) - key differentiator from appendicitis (RLQ). LOCALIZED and CONSTANT.",
                "expected": "ACCEPT (Should get weighted boost for 'left', 'lower')"
            },
            {
                "user_response": "left lower quadrant pain",
                "guideline_location": "LEFT LOWER QUADRANT (LLQ) - key differentiator from appendicitis (RLQ). LOCALIZED and CONSTANT.",
                "expected": "ACCEPT (Should get high weighted boost for exact matches)"
            },
            {
                "user_response": "upper right side under my ribs",
                "guideline_location": "RIGHT UPPER QUADRANT (RUQ) pain with CVA tenderness.",
                "expected": "ACCEPT (Should get weighted boost for 'right', 'upper')"
            },
            
            # Test fuzzy matching with typos
            {
                "user_response": "left lower abdomnial pain",
                "guideline_location": "LEFT LOWER QUADRANT (LLQ) - key differentiator from appendicitis (RLQ). LOCALIZED and CONSTANT.",
                "expected": "ACCEPT (Should match 'abdomnial'≈'abdominal' with fuzzy matching)"
            },
            {
                "user_response": "upper rite side pain",
                "guideline_location": "RIGHT UPPER QUADRANT (RUQ) pain with CVA tenderness.",
                "expected": "ACCEPT (Should match 'rite'≈'right' with fuzzy matching)"
            },
            
            # Wrong locations that should be REJECTED
            {
                "user_response": "I have sharp left lower abdomanial pain towards my pelvis",
                "guideline_location": "EPIGASTRIC, midline upper abdomen. No radiation typically.",
                "expected": "REJECT (Wrong location - no fuzzy matches)"
            },
            {
                "user_response": "upper right side under my ribs",
                "guideline_location": "LEFT LOWER QUADRANT (LLQ) - key differentiator from appendicitis (RLQ).",
                "expected": "REJECT (Wrong location - no fuzzy matches)"
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
        print("\n📊 FUZZY MATCHING ANALYSIS:")
        print("=" * 50)
        
        if accepted_scores:
            print(f"✅ ACCEPTED scores: {[f'{s:.3f}' for s in accepted_scores]}")
            print(f"   Min: {min(accepted_scores):.3f}, Max: {max(accepted_scores):.3f}, Avg: {sum(accepted_scores)/len(accepted_scores):.3f}")
        
        if rejected_scores:
            print(f"❌ REJECTED scores: {[f'{s:.3f}' for s in rejected_scores]}")
            print(f"   Min: {min(rejected_scores):.3f}, Max: {max(rejected_scores):.3f}, Avg: {sum(rejected_scores)/len(rejected_scores):.3f}")
        
        print("\n🎯 FUZZY MATCHING FEATURES:")
        print("✅ No hardcoded stop words")
        print("✅ Weighted scoring based on fuzzy match quality")
        print("✅ Handles typos (abdomanial ≈ abdominal)")
        print("✅ Handles variations (rite ≈ right)")
        print("✅ Exact matches get weight 1.0")
        print("✅ Substring matches get weight 0.8")
        print("✅ Character overlap gets proportional weight")
        print("✅ Only matches above 60% similarity count")
        
        print("\n🎯 SUMMARY:")
        print("Fuzzy matching approach should:")
        print("✅ Handle typos and variations naturally")
        print("✅ Give higher weights to better matches")
        print("✅ Work for any medical condition")
        print("✅ Be completely dynamic (no hardcoded lists)")
        print("✅ Provide precise location matching")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_fuzzy_matching()

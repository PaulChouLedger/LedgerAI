#!/usr/bin/env python3

import sys
import os
sys.path.append('/app')

# Import the adaptive diagnostic engine and RAG embedding API
from adaptive_diagnostic_engine import AdaptiveDiagnosticEngine
from unified_medical_mode import RAGEmbeddingAPI

def test_enhanced_location_similarity():
    print("🧪 Testing Enhanced Location Similarity")
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
        
        # Test cases for enhanced location similarity
        test_cases = [
            # Should be HIGH similarity with boost
            {
                "user_response": "left lower belly pain towards my pelvis",
                "guideline_location": "LEFT LOWER QUADRANT (LLQ) - key differentiator from appendicitis (RLQ). LOCALIZED and CONSTANT.",
                "expected": "HIGH (Diverticulitis - exact matches: left, lower, abdomen)"
            },
            {
                "user_response": "left lower quadrant pain",
                "guideline_location": "LEFT LOWER QUADRANT (LLQ) - key differentiator from appendicitis (RLQ). LOCALIZED and CONSTANT.",
                "expected": "HIGH (Diverticulitis - exact matches: left, lower, quadrant)"
            },
            {
                "user_response": "upper right side under my ribs",
                "guideline_location": "RIGHT UPPER QUADRANT (RUQ) pain with CVA tenderness.",
                "expected": "HIGH (Cholecystitis - exact matches: right, upper, ribs)"
            },
            
            # Should be REJECTED (contradictory terms)
            {
                "user_response": "left lower belly pain towards my pelvis",
                "guideline_location": "RIGHT UPPER QUADRANT (RUQ) pain with CVA tenderness.",
                "expected": "REJECTED (contradictory: left vs right, lower vs upper)"
            },
            {
                "user_response": "upper right side under my ribs",
                "guideline_location": "LEFT LOWER QUADRANT (LLQ) - key differentiator from appendicitis (RLQ).",
                "expected": "REJECTED (contradictory: right vs left, upper vs lower)"
            },
            
            # Should be MEDIUM (no exact matches, fallback to semantic)
            {
                "user_response": "belly pain",
                "guideline_location": "LEFT LOWER QUADRANT (LLQ) - key differentiator from appendicitis (RLQ).",
                "expected": "MEDIUM (generic vs specific - no exact matches)"
            }
        ]
        
        # Test each case
        for i, case in enumerate(test_cases, 1):
            print(f"Test {i}: {case['expected']}")
            print(f"User: \"{case['user_response']}\"")
            print(f"Guideline: \"{case['guideline_location']}\"")
            
            try:
                # Test enhanced similarity
                enhanced_similarity = engine._compute_enhanced_location_similarity(
                    case['user_response'], 
                    case['guideline_location']
                )
                print(f"Enhanced similarity: {enhanced_similarity:.3f}")
                
                # Test regular similarity for comparison
                regular_similarity = engine._compute_similarity(
                    case['user_response'], 
                    case['guideline_location']
                )
                print(f"Regular similarity: {regular_similarity:.3f}")
                
                # Show improvement
                if enhanced_similarity > 0:
                    improvement = enhanced_similarity - regular_similarity
                    if improvement > 0:
                        print(f"📈 Improvement: +{improvement:.3f}")
                    elif improvement < 0:
                        print(f"📉 Change: {improvement:.3f}")
                
                # Categorize result
                if enhanced_similarity == 0.0:
                    print("🚫 REJECTED (contradictory terms)")
                elif enhanced_similarity >= 0.7:
                    print("✅ HIGH similarity")
                elif enhanced_similarity >= 0.5:
                    print("📊 MEDIUM similarity")
                else:
                    print("📉 LOW similarity")
                    
            except Exception as e:
                print(f"❌ Error computing similarity: {e}")
            
            print("-" * 70)
        
        print("\n🎯 SUMMARY:")
        print("Enhanced location similarity should:")
        print("✅ Boost scores for exact anatomical matches")
        print("🚫 Reject contradictory terms (left vs right, upper vs lower)")
        print("📊 Fall back to semantic similarity for generic terms")
        print("🎯 Provide more precise differentiation than pure semantic similarity")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_enhanced_location_similarity()

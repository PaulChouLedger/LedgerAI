#!/usr/bin/env python3

import sys
import os
sys.path.append('/app')

# Import the adaptive diagnostic engine and RAG embedding API
from adaptive_diagnostic_engine import AdaptiveDiagnosticEngine
from unified_medical_mode import RAGEmbeddingAPI

def test_pure_semantic_location():
    print("🧪 Testing Pure Semantic Location Similarity")
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
        
        # Test cases for pure semantic similarity
        test_cases = [
            # Should be HIGH similarity
            {
                "user_response": "left lower belly pain towards my pelvis",
                "guideline_location": "LEFT LOWER QUADRANT (LLQ) - key differentiator from appendicitis (RLQ). LOCALIZED and CONSTANT.",
                "expected": "HIGH (Diverticulitis match)"
            },
            {
                "user_response": "left lower quadrant pain",
                "guideline_location": "LEFT LOWER QUADRANT (LLQ) - key differentiator from appendicitis (RLQ). LOCALIZED and CONSTANT.",
                "expected": "HIGH (Diverticulitis match)"
            },
            {
                "user_response": "upper right side under my ribs",
                "guideline_location": "RIGHT UPPER QUADRANT (RUQ) pain with CVA tenderness.",
                "expected": "HIGH (Cholecystitis match)"
            },
            {
                "user_response": "pain in my upper right abdomen",
                "guideline_location": "RIGHT UPPER QUADRANT (RUQ) pain with CVA tenderness.",
                "expected": "HIGH (Cholecystitis match)"
            },
            
            # Should be LOW similarity (different locations)
            {
                "user_response": "left lower belly pain towards my pelvis",
                "guideline_location": "EPIGASTRIC, midline upper abdomen. No radiation typically.",
                "expected": "LOW (Peptic Ulcer - wrong location)"
            },
            {
                "user_response": "upper right side under my ribs",
                "guideline_location": "LEFT LOWER QUADRANT (LLQ) - key differentiator from appendicitis (RLQ).",
                "expected": "LOW (Diverticulitis - wrong location)"
            },
            
            # Edge cases
            {
                "user_response": "belly pain",
                "guideline_location": "LEFT LOWER QUADRANT (LLQ) - key differentiator from appendicitis (RLQ).",
                "expected": "MEDIUM (generic vs specific)"
            },
            {
                "user_response": "abdominal pain",
                "guideline_location": "EPIGASTRIC, midline upper abdomen. No radiation typically.",
                "expected": "MEDIUM (generic vs specific)"
            },
            
            # Complex descriptions
            {
                "user_response": "sharp pain in the lower left part of my belly that goes towards my pelvis",
                "guideline_location": "LEFT LOWER QUADRANT (LLQ) - key differentiator from appendicitis (RLQ). LOCALIZED and CONSTANT.",
                "expected": "HIGH (Complex Diverticulitis match)"
            },
            {
                "user_response": "dull ache in my upper right side just below my ribcage",
                "guideline_location": "RIGHT UPPER QUADRANT (RUQ) pain with CVA tenderness.",
                "expected": "HIGH (Complex Cholecystitis match)"
            }
        ]
        
        # Test each case
        high_scores = []
        low_scores = []
        medium_scores = []
        
        for i, case in enumerate(test_cases, 1):
            print(f"Test {i}: {case['expected']}")
            print(f"User: \"{case['user_response']}\"")
            print(f"Guideline: \"{case['guideline_location']}\"")
            
            try:
                similarity = engine._compute_enhanced_location_similarity(
                    case['user_response'], 
                    case['guideline_location']
                )
                print(f"Pure semantic similarity: {similarity:.3f}")
                
                # Categorize scores
                if "HIGH" in case['expected']:
                    high_scores.append(similarity)
                    print(f"🎯 HIGH similarity expected")
                elif "LOW" in case['expected']:
                    low_scores.append(similarity)
                    print(f"📉 LOW similarity expected")
                else:
                    medium_scores.append(similarity)
                    print(f"📊 MEDIUM similarity expected")
                    
            except Exception as e:
                print(f"❌ Error computing similarity: {e}")
            
            print("-" * 70)
        
        # Analyze results
        print("\n📊 PURE SEMANTIC ANALYSIS:")
        print("=" * 50)
        
        if high_scores:
            print(f"🎯 HIGH similarity scores: {[f'{s:.3f}' for s in high_scores]}")
            print(f"   Min: {min(high_scores):.3f}, Max: {max(high_scores):.3f}, Avg: {sum(high_scores)/len(high_scores):.3f}")
        
        if low_scores:
            print(f"📉 LOW similarity scores: {[f'{s:.3f}' for s in low_scores]}")
            print(f"   Min: {min(low_scores):.3f}, Max: {max(low_scores):.3f}, Avg: {sum(low_scores)/len(low_scores):.3f}")
        
        if medium_scores:
            print(f"📊 MEDIUM similarity scores: {[f'{s:.3f}' for s in medium_scores]}")
            print(f"   Min: {min(medium_scores):.3f}, Max: {max(medium_scores):.3f}, Avg: {sum(medium_scores)/len(medium_scores):.3f}")
        
        # Suggest threshold
        if high_scores and low_scores:
            min_high = min(high_scores)
            max_low = max(low_scores)
            suggested_threshold = (min_high + max_low) / 2
            
            print(f"\n💡 SUGGESTED THRESHOLD: {suggested_threshold:.3f}")
            print(f"   (Midpoint between min HIGH {min_high:.3f} and max LOW {max_low:.3f})")
            
            # Test current threshold
            current_threshold = 0.65
            high_above_threshold = sum(1 for s in high_scores if s > current_threshold)
            low_below_threshold = sum(1 for s in low_scores if s < current_threshold)
            
            print(f"\n🔍 CURRENT THRESHOLD ({current_threshold}) ANALYSIS:")
            print(f"   HIGH scores above {current_threshold}: {high_above_threshold}/{len(high_scores)}")
            print(f"   LOW scores below {current_threshold}: {low_below_threshold}/{len(low_scores)}")
            
            if high_above_threshold < len(high_scores):
                print(f"   ⚠️  {len(high_scores) - high_above_threshold} HIGH scores are being rejected!")
            if low_below_threshold < len(low_scores):
                print(f"   ⚠️  {len(low_scores) - low_below_threshold} LOW scores are being accepted!")
        
        print("\n🎯 SUMMARY:")
        print("Pure semantic similarity should:")
        print("✅ Handle complex location descriptions naturally")
        print("✅ Understand medical terminology without hardcoded lists")
        print("✅ Work for any medical condition")
        print("✅ Provide nuanced similarity scores")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_pure_semantic_location()

#!/usr/bin/env python3

import sys
import os
sys.path.append('/app')

# Import the adaptive diagnostic engine and RAG embedding API
from adaptive_diagnostic_engine import AdaptiveDiagnosticEngine
from unified_medical_mode import RAGEmbeddingAPI

def test_comprehensive_fuzzy_matching():
    print("🧪 Comprehensive Fuzzy Matching Test")
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
        
        # Comprehensive test cases
        test_cases = [
            # Your edge case
            {
                "user_response": "I have sharp left lower abdomanial pain towards my pelvis",
                "guideline_location": "LEFT LOWER QUADRANT (LLQ) - key differentiator from appendicitis (RLQ). LOCALIZED and CONSTANT.",
                "expected": "ACCEPT (Edge case - should get fuzzy boost for 'left', 'lower')",
                "category": "EDGE_CASE"
            },
            
            # Exact matches (should get high boost)
            {
                "user_response": "left lower quadrant pain",
                "guideline_location": "LEFT LOWER QUADRANT (LLQ) - key differentiator from appendicitis (RLQ). LOCALIZED and CONSTANT.",
                "expected": "ACCEPT (Exact matches - should get high boost)",
                "category": "EXACT_MATCH"
            },
            {
                "user_response": "right upper quadrant",
                "guideline_location": "RIGHT UPPER QUADRANT (RUQ) pain with CVA tenderness.",
                "expected": "ACCEPT (Exact matches - should get high boost)",
                "category": "EXACT_MATCH"
            },
            
            # Typos and variations
            {
                "user_response": "left lower abdomnial pain",
                "guideline_location": "LEFT LOWER QUADRANT (LLQ) - key differentiator from appendicitis (RLQ). LOCALIZED and CONSTANT.",
                "expected": "ACCEPT (Typo - 'abdomnial' should match 'abdominal')",
                "category": "TYPO"
            },
            {
                "user_response": "upper rite side under ribs",
                "guideline_location": "RIGHT UPPER QUADRANT (RUQ) pain with CVA tenderness.",
                "expected": "ACCEPT (Typo - 'rite' should match 'right')",
                "category": "TYPO"
            },
            {
                "user_response": "lower left quadrent pain",
                "guideline_location": "LEFT LOWER QUADRANT (LLQ) - key differentiator from appendicitis (RLQ). LOCALIZED and CONSTANT.",
                "expected": "ACCEPT (Typo - 'quadrent' should match 'quadrant')",
                "category": "TYPO"
            },
            
            # Complex descriptions
            {
                "user_response": "sharp pain in the lower left part of my belly that goes towards my pelvis",
                "guideline_location": "LEFT LOWER QUADRANT (LLQ) - key differentiator from appendicitis (RLQ). LOCALIZED and CONSTANT.",
                "expected": "ACCEPT (Complex - should get boost for 'left', 'lower')",
                "category": "COMPLEX"
            },
            {
                "user_response": "dull ache in my upper right side just below my ribcage",
                "guideline_location": "RIGHT UPPER QUADRANT (RUQ) pain with CVA tenderness.",
                "expected": "ACCEPT (Complex - should get boost for 'right', 'upper')",
                "category": "COMPLEX"
            },
            
            # Wrong locations (should be rejected)
            {
                "user_response": "I have sharp left lower abdomanial pain towards my pelvis",
                "guideline_location": "EPIGASTRIC, midline upper abdomen. No radiation typically.",
                "expected": "REJECT (Wrong location - left vs upper)",
                "category": "WRONG_LOCATION"
            },
            {
                "user_response": "upper right side under my ribs",
                "guideline_location": "LEFT LOWER QUADRANT (LLQ) - key differentiator from appendicitis (RLQ).",
                "expected": "REJECT (Wrong location - right vs left, upper vs lower)",
                "category": "WRONG_LOCATION"
            },
            {
                "user_response": "chest pain in the center",
                "guideline_location": "LEFT LOWER QUADRANT (LLQ) - key differentiator from appendicitis (RLQ).",
                "expected": "REJECT (Wrong location - chest vs abdomen)",
                "category": "WRONG_LOCATION"
            },
            
            # Generic descriptions (should be rejected)
            {
                "user_response": "belly pain",
                "guideline_location": "LEFT LOWER QUADRANT (LLQ) - key differentiator from appendicitis (RLQ).",
                "expected": "REJECT (Too generic - no specific location)",
                "category": "GENERIC"
            },
            {
                "user_response": "abdominal pain",
                "guideline_location": "EPIGASTRIC, midline upper abdomen. No radiation typically.",
                "expected": "REJECT (Too generic - no specific location)",
                "category": "GENERIC"
            },
            
            # Edge cases with minimal matches
            {
                "user_response": "pain in my left side",
                "guideline_location": "LEFT LOWER QUADRANT (LLQ) - key differentiator from appendicitis (RLQ). LOCALIZED and CONSTANT.",
                "expected": "ACCEPT (Minimal match - 'left' should be enough)",
                "category": "MINIMAL_MATCH"
            },
            {
                "user_response": "upper abdomen pain",
                "guideline_location": "RIGHT UPPER QUADRANT (RUQ) pain with CVA tenderness.",
                "expected": "ACCEPT (Minimal match - 'upper' should be enough)",
                "category": "MINIMAL_MATCH"
            }
        ]
        
        # Test each case
        results = {
            "ACCEPT": [],
            "REJECT": [],
            "EDGE_CASE": [],
            "EXACT_MATCH": [],
            "TYPO": [],
            "COMPLEX": [],
            "WRONG_LOCATION": [],
            "GENERIC": [],
            "MINIMAL_MATCH": []
        }
        
        for i, case in enumerate(test_cases, 1):
            print(f"Test {i}: {case['expected']}")
            print(f"Category: {case['category']}")
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
                    result = "ACCEPT"
                    print(f"✅ ACCEPTED (>{threshold})")
                else:
                    result = "REJECT"
                    print(f"❌ REJECTED (<{threshold})")
                
                # Store results by category
                results[result].append(similarity)
                results[case['category']].append(similarity)
                    
            except Exception as e:
                print(f"❌ Error computing similarity: {e}")
            
            print("-" * 70)
        
        # Analyze results by category
        print("\n📊 COMPREHENSIVE ANALYSIS BY CATEGORY:")
        print("=" * 60)
        
        categories = [
            ("EDGE_CASE", "Your specific edge case"),
            ("EXACT_MATCH", "Exact word matches"),
            ("TYPO", "Typos and variations"),
            ("COMPLEX", "Complex descriptions"),
            ("WRONG_LOCATION", "Wrong locations (should reject)"),
            ("GENERIC", "Generic descriptions (should reject)"),
            ("MINIMAL_MATCH", "Minimal matches (should accept)")
        ]
        
        for category, description in categories:
            if results[category]:
                scores = results[category]
                accepted = sum(1 for s in scores if s > 0.70)
                print(f"\n{category}: {description}")
                print(f"   Scores: {[f'{s:.3f}' for s in scores]}")
                print(f"   Accepted: {accepted}/{len(scores)} (>{0.70})")
                print(f"   Avg: {sum(scores)/len(scores):.3f}")
        
        # Overall analysis
        print(f"\n📊 OVERALL RESULTS:")
        print(f"   ACCEPTED: {len(results['ACCEPT'])} cases")
        print(f"   REJECTED: {len(results['REJECT'])} cases")
        
        if results['ACCEPT']:
            print(f"   ACCEPTED scores: {[f'{s:.3f}' for s in results['ACCEPT']]}")
            print(f"   Min: {min(results['ACCEPT']):.3f}, Max: {max(results['ACCEPT']):.3f}, Avg: {sum(results['ACCEPT'])/len(results['ACCEPT']):.3f}")
        
        if results['REJECT']:
            print(f"   REJECTED scores: {[f'{s:.3f}' for s in results['REJECT']]}")
            print(f"   Min: {min(results['REJECT']):.3f}, Max: {max(results['REJECT']):.3f}, Avg: {sum(results['REJECT'])/len(results['REJECT']):.3f}")
        
        # Threshold analysis
        if results['ACCEPT'] and results['REJECT']:
            min_accepted = min(results['ACCEPT'])
            max_rejected = max(results['REJECT'])
            gap = min_accepted - max_rejected
            
            print(f"\n🎯 THRESHOLD ANALYSIS:")
            print(f"   Min accepted: {min_accepted:.3f}")
            print(f"   Max rejected: {max_rejected:.3f}")
            print(f"   Gap: {gap:.3f}")
            
            if gap > 0:
                print(f"   ✅ Perfect separation! No overlap between accepted/rejected")
            else:
                print(f"   ⚠️  Some overlap - consider adjusting threshold or weights")
        
        print("\n🎯 SUMMARY:")
        print("Comprehensive fuzzy matching should:")
        print("✅ Handle your edge case with typos")
        print("✅ Accept exact matches with high scores")
        print("✅ Handle typos and variations")
        print("✅ Accept complex descriptions")
        print("❌ Reject wrong locations")
        print("❌ Reject generic descriptions")
        print("🎯 Provide good separation for diagnostic accuracy")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_comprehensive_fuzzy_matching()

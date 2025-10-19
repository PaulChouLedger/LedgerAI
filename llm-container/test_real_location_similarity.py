#!/usr/bin/env python3

import sys
import os
sys.path.append('/app')

# Import the adaptive diagnostic engine and RAG embedding API
from adaptive_diagnostic_engine import AdaptiveDiagnosticEngine
from unified_medical_mode import RAGEmbeddingAPI

def test_real_location_similarity():
    print("🧪 Real Location Similarity Test - Using Actual Guideline LOCATION Sections")
    print("=" * 80)
    
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
        
        # Load real guidelines and extract LOCATION sections
        print("🔄 Loading real guidelines and extracting LOCATION sections...")
        test_cases = []
        
        # Test with real guideline data
        for guideline_name in engine.all_guidelines:
            guideline_data = engine.all_guidelines[guideline_name]
            
            # Extract LOCATION section from the guideline
            classic_presentation = guideline_data.get('key_features', {}).get('classic_presentation', '')
            if not classic_presentation:
                print(f"⚠️ No classic_presentation found in {guideline_name}")
                continue
                
            location_section = engine._extract_oldcarts_section(classic_presentation, 'L')
            if not location_section:
                print(f"⚠️ No LOCATION section found in {guideline_name}")
                continue
                
            print(f"📋 {guideline_name}: {location_section[:100]}...")
            
            # Create test cases based on the actual LOCATION content
            if 'appendicitis' in guideline_name.lower():
                test_cases.extend([
                    {
                        "user_response": "I have severe pain in my lower right abdomen that started suddenly",
                        "guideline_name": guideline_name,
                        "location_section": location_section,
                        "expected": "ACCEPT (Appendicitis - RLQ pain)",
                        "category": "GI_APPENDICITIS"
                    },
                    {
                        "user_response": "left side abdominal pain",
                        "guideline_name": guideline_name,
                        "location_section": location_section,
                        "expected": "REJECT (Appendicitis - wrong side)",
                        "category": "GI_APPENDICITIS"
                    }
                ])
            elif 'cholecystitis' in guideline_name.lower():
                test_cases.extend([
                    {
                        "user_response": "I have pain in my upper right abdomen under my ribs",
                        "guideline_name": guideline_name,
                        "location_section": location_section,
                        "expected": "ACCEPT (Cholecystitis - RUQ pain)",
                        "category": "GI_CHOLECYSTITIS"
                    },
                    {
                        "user_response": "left upper quadrant pain",
                        "guideline_name": guideline_name,
                        "location_section": location_section,
                        "expected": "REJECT (Cholecystitis - wrong side)",
                        "category": "GI_CHOLECYSTITIS"
                    }
                ])
            elif 'diverticulitis' in guideline_name.lower():
                test_cases.extend([
                    {
                        "user_response": "I have sharp left lower belly pain towards my pelvis",
                        "guideline_name": guideline_name,
                        "location_section": location_section,
                        "expected": "ACCEPT (Diverticulitis - LLQ pain)",
                        "category": "GI_DIVERTICULITIS"
                    },
                    {
                        "user_response": "right lower abdominal pain",
                        "guideline_name": guideline_name,
                        "location_section": location_section,
                        "expected": "REJECT (Diverticulitis - wrong side)",
                        "category": "GI_DIVERTICULITIS"
                    }
                ])
            elif 'myocardial' in guideline_name.lower() or 'heart attack' in guideline_name.lower():
                test_cases.extend([
                    {
                        "user_response": "crushing chest pain that radiates to my left arm",
                        "guideline_name": guideline_name,
                        "location_section": location_section,
                        "expected": "ACCEPT (MI - chest pain)",
                        "category": "CARDIO_MI"
                    },
                    {
                        "user_response": "abdominal pain",
                        "guideline_name": guideline_name,
                        "location_section": location_section,
                        "expected": "REJECT (MI - wrong location)",
                        "category": "CARDIO_MI"
                    }
                ])
        
        print(f"✅ Created {len(test_cases)} test cases from real guidelines")
        print()
        
        # Run tests with real data
        results = {
            "ACCEPT": [],
            "REJECT": [],
            "GI_APPENDICITIS": [],
            "GI_CHOLECYSTITIS": [],
            "GI_DIVERTICULITIS": [],
            "CARDIO_MI": []
        }
        
        for i, case in enumerate(test_cases, 1):
            print(f"Test {i}: {case['expected']}")
            print(f"Category: {case['category']}")
            print(f"User: \"{case['user_response']}\"")
            print(f"Guideline: \"{case['location_section']}\"")
            
            try:
                similarity = engine._compute_enhanced_location_similarity(
                    case['user_response'], 
                    case['location_section']
                )
                print(f"Final similarity: {similarity:.3f}")
                
                # Test against threshold (matches engine's SEMANTIC_THRESHOLD)
                threshold = 0.60
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
                import traceback
                traceback.print_exc()
            
            print("-" * 80)
        
        # Analysis
        print("\n📊 ANALYSIS BY CATEGORY:")
        print("=" * 80)
        
        for category, scores in results.items():
            if scores:
                print(f"{category}:")
                print(f"   Scores: {[f'{s:.3f}' for s in scores]}")
                accepted = sum(1 for s in scores if s > 0.60)
                print(f"   Accepted: {accepted}/{len(scores)} (>{0.60})")
                print(f"   Avg: {sum(scores)/len(scores):.3f}")
                print()
        
        # Overall results
        all_accepted = results["ACCEPT"]
        all_rejected = results["REJECT"]
        
        print("📊 OVERALL RESULTS:")
        print(f"   ACCEPTED: {len(all_accepted)} cases")
        print(f"   REJECTED: {len(all_rejected)} cases")
        
        if all_accepted:
            print(f"   ACCEPTED scores: {[f'{s:.3f}' for s in all_accepted]}")
            print(f"   Min: {min(all_accepted):.3f}, Max: {max(all_accepted):.3f}, Avg: {sum(all_accepted)/len(all_accepted):.3f}")
        
        if all_rejected:
            print(f"   REJECTED scores: {[f'{s:.3f}' for s in all_rejected]}")
            print(f"   Min: {min(all_rejected):.3f}, Max: {max(all_rejected):.3f}, Avg: {sum(all_rejected)/len(all_rejected):.3f}")
        
        # Threshold analysis
        if all_accepted and all_rejected:
            min_accepted = min(all_accepted)
            max_rejected = max(all_rejected)
            gap = min_accepted - max_rejected
            print(f"\n🎯 THRESHOLD ANALYSIS:")
            print(f"   Min accepted: {min_accepted:.3f}")
            print(f"   Max rejected: {max_rejected:.3f}")
            print(f"   Gap: {gap:.3f}")
            if gap > 0:
                print(f"   ✅ Perfect separation! No overlap between accepted/rejected")
            else:
                print(f"   ⚠️ Overlap detected! Some cases misclassified")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_real_location_similarity()

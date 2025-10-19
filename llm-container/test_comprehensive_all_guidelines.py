#!/usr/bin/env python3

import sys
import os
sys.path.append('/app')

# Import the adaptive diagnostic engine and RAG embedding API
from adaptive_diagnostic_engine import AdaptiveDiagnosticEngine
from unified_medical_mode import RAGEmbeddingAPI

def test_comprehensive_all_guidelines():
    print("🧪 Comprehensive Test - All GI & Cardiovascular Guidelines")
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
        real_test_cases = []
        
        # Test with real guideline data
        for guideline in engine.all_guidelines:
            guideline_name = guideline['name']
            guideline_data = guideline['data']
            
            # Extract LOCATION section from the guideline
            location_section = engine._extract_oldcarts_section(guideline_data, 'LOCATION')
            if not location_section:
                continue
                
            # Create test cases based on the actual LOCATION content
            if 'appendicitis' in guideline_name.lower():
                real_test_cases.extend([
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
                real_test_cases.extend([
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
                real_test_cases.extend([
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
                real_test_cases.extend([
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
        
        print(f"✅ Created {len(real_test_cases)} test cases from real guidelines")
        print()
        
        # Run tests with real data
        test_cases = real_test_cases
            {
                "user_response": "sharp pain in my right lower belly",
                "guideline_location": "RIGHT LOWER QUADRANT (RLQ) pain, classically starting periumbilical then migrating to RLQ.",
                "expected": "ACCEPT (Appendicitis - right lower)",
                "category": "GI_APPENDICITIS"
            },
            {
                "user_response": "left side abdominal pain",
                "guideline_location": "RIGHT LOWER QUADRANT (RLQ) pain, classically starting periumbilical then migrating to RLQ.",
                "expected": "REJECT (Appendicitis - wrong side)",
                "category": "GI_APPENDICITIS"
            },
            
            # ACUTE CHOLECYSTITIS
            {
                "user_response": "I have pain in my upper right abdomen under my ribs",
                "guideline_location": "RIGHT UPPER QUADRANT (RUQ) pain with CVA tenderness.",
                "expected": "ACCEPT (Cholecystitis - RUQ pain)",
                "category": "GI_CHOLECYSTITIS"
            },
            {
                "user_response": "upper right side pain that radiates to my back",
                "guideline_location": "RIGHT UPPER QUADRANT (RUQ) pain with CVA tenderness.",
                "expected": "ACCEPT (Cholecystitis - upper right)",
                "category": "GI_CHOLECYSTITIS"
            },
            {
                "user_response": "left upper quadrant pain",
                "guideline_location": "RIGHT UPPER QUADRANT (RUQ) pain with CVA tenderness.",
                "expected": "REJECT (Cholecystitis - wrong side)",
                "category": "GI_CHOLECYSTITIS"
            },
            
            # ACUTE PANCREATITIS
            {
                "user_response": "severe pain in my upper abdomen that goes to my back",
                "guideline_location": "EPIGASTRIC pain radiating to back, often described as 'boring' or 'penetrating'.",
                "expected": "ACCEPT (Pancreatitis - epigastric)",
                "category": "GI_PANCREATITIS"
            },
            {
                "user_response": "burning pain in the center of my upper belly",
                "guideline_location": "EPIGASTRIC pain radiating to back, often described as 'boring' or 'penetrating'.",
                "expected": "ACCEPT (Pancreatitis - epigastric)",
                "category": "GI_PANCREATITIS"
            },
            {
                "user_response": "lower abdominal pain",
                "guideline_location": "EPIGASTRIC pain radiating to back, often described as 'boring' or 'penetrating'.",
                "expected": "REJECT (Pancreatitis - wrong location)",
                "category": "GI_PANCREATITIS"
            },
            
            # DIVERTICULITIS
            {
                "user_response": "I have sharp left lower belly pain towards my pelvis",
                "guideline_location": "LEFT LOWER QUADRANT (LLQ) - key differentiator from appendicitis (RLQ). LOCALIZED and CONSTANT.",
                "expected": "ACCEPT (Diverticulitis - LLQ pain)",
                "category": "GI_DIVERTICULITIS"
            },
            {
                "user_response": "left lower quadrant pain that's constant",
                "guideline_location": "LEFT LOWER QUADRANT (LLQ) - key differentiator from appendicitis (RLQ). LOCALIZED and CONSTANT.",
                "expected": "ACCEPT (Diverticulitis - LLQ)",
                "category": "GI_DIVERTICULITIS"
            },
            {
                "user_response": "right lower abdominal pain",
                "guideline_location": "LEFT LOWER QUADRANT (LLQ) - key differentiator from appendicitis (RLQ). LOCALIZED and CONSTANT.",
                "expected": "REJECT (Diverticulitis - wrong side)",
                "category": "GI_DIVERTICULITIS"
            },
            
            # PEPTIC ULCER DISEASE
            {
                "user_response": "burning pain in my upper middle abdomen",
                "guideline_location": "EPIGASTRIC, midline upper abdomen. No radiation typically.",
                "expected": "ACCEPT (Peptic Ulcer - epigastric)",
                "category": "GI_PEPTIC_ULCER"
            },
            {
                "user_response": "upper central abdominal pain",
                "guideline_location": "EPIGASTRIC, midline upper abdomen. No radiation typically.",
                "expected": "ACCEPT (Peptic Ulcer - epigastric)",
                "category": "GI_PEPTIC_ULCER"
            },
            {
                "user_response": "lower left abdominal pain",
                "guideline_location": "EPIGASTRIC, midline upper abdomen. No radiation typically.",
                "expected": "REJECT (Peptic Ulcer - wrong location)",
                "category": "GI_PEPTIC_ULCER"
            },
            
            # BILIARY COLIC
            {
                "user_response": "intermittent pain in my upper right abdomen",
                "guideline_location": "RIGHT UPPER QUADRANT (RUQ) or epigastric, often radiating to right shoulder or back.",
                "expected": "ACCEPT (Biliary Colic - RUQ)",
                "category": "GI_BILIARY_COLIC"
            },
            {
                "user_response": "upper right side pain that comes and goes",
                "guideline_location": "RIGHT UPPER QUADRANT (RUQ) or epigastric, often radiating to right shoulder or back.",
                "expected": "ACCEPT (Biliary Colic - upper right)",
                "category": "GI_BILIARY_COLIC"
            },
            
            # GASTROENTERITIS
            {
                "user_response": "cramping pain all over my abdomen",
                "guideline_location": "GENERALIZED abdominal pain, often crampy and intermittent.",
                "expected": "ACCEPT (Gastroenteritis - generalized)",
                "category": "GI_GASTROENTERITIS"
            },
            {
                "user_response": "diffuse abdominal pain with cramps",
                "guideline_location": "GENERALIZED abdominal pain, often crampy and intermittent.",
                "expected": "ACCEPT (Gastroenteritis - generalized)",
                "category": "GI_GASTROENTERITIS"
            },
            
            # ===== CARDIOVASCULAR GUIDELINES =====
            
            # ACUTE MYOCARDIAL INFARCTION (HEART ATTACK)
            {
                "user_response": "crushing chest pain that radiates to my left arm",
                "guideline_location": "CHEST pain, often described as pressure, heaviness, or crushing. May radiate to left arm, jaw, or back.",
                "expected": "ACCEPT (MI - chest pain)",
                "category": "CARDIO_MI"
            },
            {
                "user_response": "severe pressure in my chest",
                "guideline_location": "CHEST pain, often described as pressure, heaviness, or crushing. May radiate to left arm, jaw, or back.",
                "expected": "ACCEPT (MI - chest pressure)",
                "category": "CARDIO_MI"
            },
            {
                "user_response": "abdominal pain",
                "guideline_location": "CHEST pain, often described as pressure, heaviness, or crushing. May radiate to left arm, jaw, or back.",
                "expected": "REJECT (MI - wrong location)",
                "category": "CARDIO_MI"
            },
            
            # UNSTABLE ANGINA
            {
                "user_response": "chest pain that feels like pressure",
                "guideline_location": "CHEST pain, typically substernal, described as pressure, squeezing, or heaviness.",
                "expected": "ACCEPT (Angina - chest pressure)",
                "category": "CARDIO_ANGINA"
            },
            {
                "user_response": "squeezing sensation in my chest",
                "guideline_location": "CHEST pain, typically substernal, described as pressure, squeezing, or heaviness.",
                "expected": "ACCEPT (Angina - chest squeezing)",
                "category": "CARDIO_ANGINA"
            },
            
            # PULMONARY EMBOLISM
            {
                "user_response": "sharp chest pain that gets worse when I breathe",
                "guideline_location": "CHEST pain, often pleuritic (worsens with breathing), may be sharp or stabbing.",
                "expected": "ACCEPT (PE - chest pain)",
                "category": "CARDIO_PE"
            },
            {
                "user_response": "chest pain that hurts when I take deep breaths",
                "guideline_location": "CHEST pain, often pleuritic (worsens with breathing), may be sharp or stabbing.",
                "expected": "ACCEPT (PE - pleuritic chest pain)",
                "category": "CARDIO_PE"
            },
            
            # AORTIC DISSECTION
            {
                "user_response": "tearing chest pain that goes to my back",
                "guideline_location": "CHEST pain, classically described as 'tearing' or 'ripping', often radiating to back.",
                "expected": "ACCEPT (Aortic Dissection - tearing chest pain)",
                "category": "CARDIO_AORTIC_DISSECTION"
            },
            {
                "user_response": "severe chest pain that feels like tearing",
                "guideline_location": "CHEST pain, classically described as 'tearing' or 'ripping', often radiating to back.",
                "expected": "ACCEPT (Aortic Dissection - tearing chest pain)",
                "category": "CARDIO_AORTIC_DISSECTION"
            },
            
            # PERICARDITIS
            {
                "user_response": "chest pain that gets better when I lean forward",
                "guideline_location": "CHEST pain, often sharp and pleuritic, may improve with sitting up and leaning forward.",
                "expected": "ACCEPT (Pericarditis - chest pain)",
                "category": "CARDIO_PERICARDITIS"
            },
            {
                "user_response": "sharp chest pain that's worse when lying down",
                "guideline_location": "CHEST pain, often sharp and pleuritic, may improve with sitting up and leaning forward.",
                "expected": "ACCEPT (Pericarditis - chest pain)",
                "category": "CARDIO_PERICARDITIS"
            },
            
            # ===== EDGE CASES AND TYPOS =====
            
            # Typos in GI conditions
            {
                "user_response": "left lower abdomnial pain",
                "guideline_location": "LEFT LOWER QUADRANT (LLQ) - key differentiator from appendicitis (RLQ). LOCALIZED and CONSTANT.",
                "expected": "ACCEPT (Diverticulitis - typo in abdominal)",
                "category": "TYPO_GI"
            },
            {
                "user_response": "upper rite side pain",
                "guideline_location": "RIGHT UPPER QUADRANT (RUQ) pain with CVA tenderness.",
                "expected": "ACCEPT (Cholecystitis - typo in right)",
                "category": "TYPO_GI"
            },
            
            # Typos in Cardiovascular conditions
            {
                "user_response": "chest pain that radites to my arm",
                "guideline_location": "CHEST pain, often described as pressure, heaviness, or crushing. May radiate to left arm, jaw, or back.",
                "expected": "ACCEPT (MI - typo in radiates)",
                "category": "TYPO_CARDIO"
            },
            {
                "user_response": "chest presure and tightness",
                "guideline_location": "CHEST pain, typically substernal, described as pressure, squeezing, or heaviness.",
                "expected": "ACCEPT (Angina - typo in pressure)",
                "category": "TYPO_CARDIO"
            },
            
            # Complex descriptions
            {
                "user_response": "I have severe sharp pain in the lower left part of my belly that goes towards my pelvis and gets worse when I move",
                "guideline_location": "LEFT LOWER QUADRANT (LLQ) - key differentiator from appendicitis (RLQ). LOCALIZED and CONSTANT.",
                "expected": "ACCEPT (Diverticulitis - complex description)",
                "category": "COMPLEX_GI"
            },
            {
                "user_response": "crushing chest pain that feels like an elephant sitting on my chest and radiates down my left arm to my fingers",
                "guideline_location": "CHEST pain, often described as pressure, heaviness, or crushing. May radiate to left arm, jaw, or back.",
                "expected": "ACCEPT (MI - complex description)",
                "category": "COMPLEX_CARDIO"
            },
            
            # Wrong locations (should be rejected)
            {
                "user_response": "chest pain",
                "guideline_location": "LEFT LOWER QUADRANT (LLQ) - key differentiator from appendicitis (RLQ). LOCALIZED and CONSTANT.",
                "expected": "REJECT (Wrong - chest vs abdomen)",
                "category": "WRONG_LOCATION"
            },
            {
                "user_response": "abdominal pain",
                "guideline_location": "CHEST pain, often described as pressure, heaviness, or crushing. May radiate to left arm, jaw, or back.",
                "expected": "REJECT (Wrong - abdomen vs chest)",
                "category": "WRONG_LOCATION"
            },
            {
                "user_response": "left side pain",
                "guideline_location": "RIGHT LOWER QUADRANT (RLQ) pain, classically starting periumbilical then migrating to RLQ.",
                "expected": "REJECT (Wrong - left vs right)",
                "category": "WRONG_LOCATION"
            },
            
            # Generic descriptions (should be rejected)
            {
                "user_response": "belly pain",
                "guideline_location": "LEFT LOWER QUADRANT (LLQ) - key differentiator from appendicitis (RLQ). LOCALIZED and CONSTANT.",
                "expected": "REJECT (Too generic)",
                "category": "GENERIC"
            },
            {
                "user_response": "chest discomfort",
                "guideline_location": "CHEST pain, often described as pressure, heaviness, or crushing. May radiate to left arm, jaw, or back.",
                "expected": "REJECT (Too generic)",
                "category": "GENERIC"
            }
        ]
        
        # Test each case
        results = {
            "ACCEPT": [],
            "REJECT": [],
            "GI_APPENDICITIS": [],
            "GI_CHOLECYSTITIS": [],
            "GI_PANCREATITIS": [],
            "GI_DIVERTICULITIS": [],
            "GI_PEPTIC_ULCER": [],
            "GI_BILIARY_COLIC": [],
            "GI_GASTROENTERITIS": [],
            "CARDIO_MI": [],
            "CARDIO_ANGINA": [],
            "CARDIO_PE": [],
            "CARDIO_AORTIC_DISSECTION": [],
            "CARDIO_PERICARDITIS": [],
            "TYPO_GI": [],
            "TYPO_CARDIO": [],
            "COMPLEX_GI": [],
            "COMPLEX_CARDIO": [],
            "WRONG_LOCATION": [],
            "GENERIC": []
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
                threshold = 0.75
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
            
            print("-" * 80)
        
        # Analyze results by category
        print("\n📊 COMPREHENSIVE ANALYSIS BY CATEGORY:")
        print("=" * 80)
        
        categories = [
            ("GI_APPENDICITIS", "Acute Appendicitis"),
            ("GI_CHOLECYSTITIS", "Acute Cholecystitis"),
            ("GI_PANCREATITIS", "Acute Pancreatitis"),
            ("GI_DIVERTICULITIS", "Acute Diverticulitis"),
            ("GI_PEPTIC_ULCER", "Peptic Ulcer Disease"),
            ("GI_BILIARY_COLIC", "Biliary Colic"),
            ("GI_GASTROENTERITIS", "Acute Gastroenteritis"),
            ("CARDIO_MI", "Acute Myocardial Infarction"),
            ("CARDIO_ANGINA", "Unstable Angina"),
            ("CARDIO_PE", "Pulmonary Embolism"),
            ("CARDIO_AORTIC_DISSECTION", "Aortic Dissection"),
            ("CARDIO_PERICARDITIS", "Acute Pericarditis"),
            ("TYPO_GI", "GI Typos"),
            ("TYPO_CARDIO", "Cardio Typos"),
            ("COMPLEX_GI", "Complex GI Descriptions"),
            ("COMPLEX_CARDIO", "Complex Cardio Descriptions"),
            ("WRONG_LOCATION", "Wrong Locations (should reject)"),
            ("GENERIC", "Generic Descriptions (should reject)")
        ]
        
        for category, description in categories:
            if results[category]:
                scores = results[category]
                accepted = sum(1 for s in scores if s > 0.75)
                print(f"\n{category}: {description}")
                print(f"   Scores: {[f'{s:.3f}' for s in scores]}")
                print(f"   Accepted: {accepted}/{len(scores)} (>{0.75})")
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
        
        # Performance by system
        print(f"\n🎯 SYSTEM PERFORMANCE:")
        print(f"   GI Guidelines: {len([c for c in test_cases if c['category'].startswith('GI_')])} tests")
        print(f"   Cardio Guidelines: {len([c for c in test_cases if c['category'].startswith('CARDIO_')])} tests")
        print(f"   Typos: {len([c for c in test_cases if c['category'].startswith('TYPO_')])} tests")
        print(f"   Complex: {len([c for c in test_cases if c['category'].startswith('COMPLEX_')])} tests")
        print(f"   Wrong/Generic: {len([c for c in test_cases if c['category'] in ['WRONG_LOCATION', 'GENERIC']])} tests")
        
        print("\n🎯 SUMMARY:")
        print("Comprehensive fuzzy matching across all guidelines should:")
        print("✅ Handle all GI conditions (appendicitis, cholecystitis, pancreatitis, etc.)")
        print("✅ Handle all cardiovascular conditions (MI, angina, PE, etc.)")
        print("✅ Handle typos and variations across all conditions")
        print("✅ Accept complex descriptions")
        print("❌ Reject wrong locations and generic descriptions")
        print("🎯 Provide consistent performance across all medical specialties")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_comprehensive_all_guidelines()

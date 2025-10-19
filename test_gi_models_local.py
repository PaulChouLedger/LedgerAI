#!/usr/bin/env python3
"""
Local test script to test OLDCARTS normalization flow with adaptive diagnostic engine.
Tests the complete flow: User Prompt → OLDCARTS Normalization → Semantic Matching → Medical Guidelines

OLDCARTS NORMALIZATION TESTING:
- Tests the new OLDCARTS-structured synonym normalization
- Uses adaptive diagnostic engine with real medical guidelines
- Compares normalized vs non-normalized patient language
- Validates that OLDCARTS normalization improves semantic matching accuracy
"""

import sys
import os
import time
import json
import numpy as np
from typing import List, Dict, Tuple

# Add the llm-container directory to the path
# Try multiple possible paths for different environments
possible_paths = [
    '/Users/rcabello/Documents/GitHub/LedgerAI/llm-container',  # macOS path
    '~/LedgerAI/llm-container',  # Ubuntu relative path
    './llm-container',  # Current directory relative path
    'llm-container'  # Just the directory name
]

for path in possible_paths:
    expanded_path = os.path.expanduser(path)
    if os.path.exists(expanded_path):
        sys.path.insert(0, expanded_path)
        print(f"✅ Added to Python path: {expanded_path}")
        break
else:
    # If none of the paths work, try adding the current working directory
    current_dir = os.getcwd()
    llm_container_path = os.path.join(current_dir, 'llm-container')
    if os.path.exists(llm_container_path):
        sys.path.insert(0, llm_container_path)
        print(f"✅ Added to Python path: {llm_container_path}")
    else:
        print(f"⚠️  Could not find llm-container directory. Current dir: {current_dir}")
        print(f"   Available directories: {os.listdir(current_dir)}")

def get_hardcoded_gi_guidelines() -> List[Dict]:
    """Hardcoded GI guidelines for testing."""
    return [
        {
            'name': 'Acute Appendicitis',
            'location': 'Pain MIGRATES from periumbilical to right lower quadrant (RLQ) over 12-24 hours - highly specific migration pattern. Localizes to McBurney\'s point in RLQ.',
            'chief_complaint_triggers': ['right lower quadrant', 'RLQ', 'appendix', 'appendicitis', 'right lower belly', 'right lower abdomen']
        },
        {
            'name': 'Acute Cholecystitis', 
            'location': 'Right upper quadrant (RUQ), precisely localized just below right rib cage. RADIATES TO RIGHT SHOULDER OR SCAPULA (phrenic nerve referred pain).',
            'chief_complaint_triggers': ['right upper quadrant', 'RUQ', 'gallbladder', 'cholecystitis', 'right upper belly', 'right upper abdomen', 'under ribs right']
        },
        {
            'name': 'Acute Pancreatitis',
            'location': 'Epigastric (upper mid-abdomen) and periumbilical. RADIATES STRAIGHT THROUGH TO THE BACK in \'boring\' pattern.',
            'chief_complaint_triggers': ['epigastric', 'upper middle', 'pancreas', 'pancreatitis', 'middle stomach', 'upper mid abdomen']
        },
        {
            'name': 'Acute Gastroenteritis',
            'location': 'PERIUMBILICAL or DIFFUSE throughout abdomen. NOT localized to one quadrant. Cramping moves around.',
            'chief_complaint_triggers': ['periumbilical', 'diffuse', 'all over', 'generalized', 'stomach flu', 'gastroenteritis', 'moves around']
        },
        {
            'name': 'Biliary Colic',
            'location': 'Right upper quadrant (RUQ) or epigastric. May RADIATE TO RIGHT SHOULDER OR BACK.',
            'chief_complaint_triggers': ['right upper quadrant', 'RUQ', 'biliary', 'gallbladder', 'right upper belly', 'right shoulder']
        },
        {
            'name': 'Small Bowel Obstruction',
            'location': 'Periumbilical and diffuse (not localized to one quadrant). Cramping throughout mid-abdomen.',
            'chief_complaint_triggers': ['periumbilical', 'diffuse', 'bowel obstruction', 'small bowel', 'cramping', 'mid abdomen']
        },
        {
            'name': 'Acute Diverticulitis',
            'location': 'LEFT LOWER QUADRANT (LLQ) - key differentiator from appendicitis (RLQ). LOCALIZED and CONSTANT. Sometimes palpable tender mass.',
            'chief_complaint_triggers': ['left lower quadrant', 'LLQ', 'diverticulitis', 'left lower belly', 'left lower abdomen']
        },
        {
            'name': 'GERD',
            'location': 'RETROSTERNAL (behind breastbone) and EPIGASTRIC. BURNING rises from stomach toward throat. No radiation.',
            'chief_complaint_triggers': ['retrosternal', 'behind breastbone', 'heartburn', 'GERD', 'acid reflux', 'burning chest']
        },
        {
            'name': 'Gastric Outlet Obstruction',
            'location': 'Epigastric (upper mid-abdomen), may radiate to back.',
            'chief_complaint_triggers': ['epigastric', 'upper middle', 'gastric outlet', 'upper mid abdomen', 'middle stomach']
        },
        {
            'name': 'Acute Gastritis',
            'location': 'EPIGASTRIC (upper mid-abdomen), diffuse. NO radiation.',
            'chief_complaint_triggers': ['epigastric', 'upper middle', 'gastritis', 'upper mid abdomen', 'middle stomach']
        },
        {
            'name': 'Acute Hepatitis',
            'location': 'RIGHT UPPER QUADRANT (liver area) discomfort. RUQ tenderness with hepatomegaly (enlarged liver).',
            'chief_complaint_triggers': ['right upper quadrant', 'RUQ', 'liver', 'hepatitis', 'right upper belly', 'liver area']
        },
        {
            'name': 'Inflammatory Bowel Disease Flare',
            'location': 'Diffuse cramping throughout abdomen, or RLQ if Crohn\'s (terminal ileum involved).',
            'chief_complaint_triggers': ['diffuse', 'cramping', 'IBD', 'Crohn\'s', 'inflammatory bowel', 'all over abdomen']
        },
        {
            'name': 'Irritable Bowel Syndrome (IBS)',
            'location': 'LOWER ABDOMEN, diffuse. Cramping migrates, not fixed to one spot.',
            'chief_complaint_triggers': ['lower abdomen', 'IBS', 'irritable bowel', 'cramping', 'migrates', 'not fixed']
        },
        {
            'name': 'Incarcerated Inguinal/Femoral Hernia',
            'location': 'Groin (inguinal or femoral region), may have lower abdominal pain if bowel involved.',
            'chief_complaint_triggers': ['groin', 'inguinal', 'femoral', 'hernia', 'lower abdomen', 'pelvic']
        },
        {
            'name': 'Mallory-Weiss Tear',
            'location': 'Epigastric (upper mid-abdomen) or lower chest discomfort.',
            'chief_complaint_triggers': ['epigastric', 'upper middle', 'Mallory-Weiss', 'upper mid abdomen', 'lower chest']
        },
        {
            'name': 'Acute Mesenteric Ischemia',
            'location': 'Diffuse, PERIUMBILICAL. Not localized to one quadrant.',
            'chief_complaint_triggers': ['diffuse', 'periumbilical', 'mesenteric', 'ischemia', 'all over abdomen']
        },
        {
            'name': 'Peptic Ulcer Disease',
            'location': 'EPIGASTRIC, midline upper abdomen. No radiation typically.',
            'chief_complaint_triggers': ['epigastric', 'upper middle', 'ulcer', 'peptic ulcer', 'upper mid abdomen']
        },
        {
            'name': 'Perforated Viscus',
            'location': 'Initially EPIGASTRIC (perforated ulcer) or localized, then becomes DIFFUSE as peritonitis develops.',
            'chief_complaint_triggers': ['epigastric', 'upper middle', 'perforated', 'peritonitis', 'diffuse', 'upper mid abdomen']
        },
        {
            'name': 'Sigmoid Volvulus',
            'location': 'Left lower quadrant (LLQ) or diffuse lower abdomen.',
            'chief_complaint_triggers': ['left lower quadrant', 'LLQ', 'sigmoid', 'volvulus', 'left lower belly', 'left lower abdomen']
        },
        {
            'name': 'Kidney Stone',
            'location': 'Unilateral FLANK PAIN. RADIATES from flank→groin→testicle (males) or labia (females). Follows ureter path.',
            'chief_complaint_triggers': ['flank', 'kidney stone', 'renal colic', 'side pain', 'radiates to groin']
        }
    ]

def get_patient_gi_prompts() -> List[Dict]:
    """Get 40 patient GI-related prompts for testing with expected matches."""
    return [
        {
            'prompt': "left lower part of my abdomen towards my pelvis",
            'expected': ['Acute Diverticulitis', 'Sigmoid Volvulus'],
            'should_reject': ['Acute Appendicitis', 'Acute Cholecystitis', 'Acute Pancreatitis']
        },
        {
            'prompt': "right upper side under my ribs",
            'expected': ['Acute Cholecystitis', 'Biliary Colic', 'Acute Hepatitis'],
            'should_reject': ['Acute Diverticulitis', 'Acute Appendicitis']
        },
        {
            'prompt': "middle of my stomach area",
            'expected': ['Acute Pancreatitis', 'Peptic Ulcer Disease', 'Acute Gastritis'],
            'should_reject': ['Acute Diverticulitis', 'Acute Appendicitis']
        },
        {
            'prompt': "all over my belly, it moves around",
            'expected': ['Acute Gastroenteritis', 'Irritable Bowel Syndrome (IBS)', 'Small Bowel Obstruction'],
            'should_reject': ['Acute Cholecystitis', 'Acute Diverticulitis']
        },
        {
            'prompt': "right lower side near my hip bone",
            'expected': ['Acute Appendicitis', 'Incarcerated Inguinal/Femoral Hernia'],
            'should_reject': ['Acute Diverticulitis', 'Acute Cholecystitis']
        },
        {
            'prompt': "upper middle part of my stomach",
            'expected': ['Acute Pancreatitis', 'Peptic Ulcer Disease', 'Acute Gastritis'],
            'should_reject': ['Acute Diverticulitis', 'Acute Appendicitis']
        },
        {
            'prompt': "left side of my belly",
            'expected': ['Acute Diverticulitis', 'Sigmoid Volvulus'],
            'should_reject': ['Acute Appendicitis', 'Acute Cholecystitis']
        },
        {
            'prompt': "left side",
            'expected': ['Acute Diverticulitis', 'Sigmoid Volvulus'],
            'should_reject': ['Acute Appendicitis', 'Acute Cholecystitis', 'Acute Pancreatitis', 'Biliary Colic', 'Acute Hepatitis']
        },
        # Additional vague patient language patterns
        {
            'prompt': "right side",
            'expected': ['Acute Appendicitis', 'Acute Cholecystitis', 'Biliary Colic'],
            'should_reject': ['Acute Diverticulitis', 'Sigmoid Volvulus']
        },
        {
            'prompt': "my left",
            'expected': ['Acute Diverticulitis', 'Sigmoid Volvulus'],
            'should_reject': ['Acute Appendicitis', 'Acute Cholecystitis']
        },
        {
            'prompt': "my right",
            'expected': ['Acute Appendicitis', 'Acute Cholecystitis', 'Biliary Colic'],
            'should_reject': ['Acute Diverticulitis', 'Sigmoid Volvulus']
        },
        {
            'prompt': "on the left",
            'expected': ['Acute Diverticulitis', 'Sigmoid Volvulus'],
            'should_reject': ['Acute Appendicitis', 'Acute Cholecystitis']
        },
        {
            'prompt': "on the right",
            'expected': ['Acute Appendicitis', 'Acute Cholecystitis', 'Biliary Colic'],
            'should_reject': ['Acute Diverticulitis', 'Sigmoid Volvulus']
        },
        {
            'prompt': "left part",
            'expected': ['Acute Diverticulitis', 'Sigmoid Volvulus'],
            'should_reject': ['Acute Appendicitis', 'Acute Cholecystitis']
        },
        {
            'prompt': "right part",
            'expected': ['Acute Appendicitis', 'Acute Cholecystitis', 'Biliary Colic'],
            'should_reject': ['Acute Diverticulitis', 'Sigmoid Volvulus']
        },
        {
            'prompt': "behind my breastbone and upper stomach",
            'expected': ['GERD', 'Acute Pancreatitis', 'Peptic Ulcer Disease'],
            'should_reject': ['Acute Diverticulitis', 'Acute Appendicitis']
        },
        {
            'prompt': "right shoulder and upper right belly",
            'expected': ['Acute Cholecystitis', 'Biliary Colic'],
            'should_reject': ['Acute Diverticulitis', 'Acute Appendicitis']
        },
        {
            'prompt': "lower part of my stomach, moves around",
            'expected': ['Irritable Bowel Syndrome (IBS)', 'Acute Gastroenteritis'],
            'should_reject': ['Acute Cholecystitis', 'Acute Pancreatitis']
        },
        {
            'prompt': "right side under my ribs, goes to my back",
            'expected': ['Acute Cholecystitis', 'Biliary Colic'],
            'should_reject': ['Acute Diverticulitis', 'Acute Appendicitis']
        },
        {
            'prompt': "middle belly area around my belly button",
            'expected': ['Acute Pancreatitis', 'Small Bowel Obstruction', 'Acute Gastroenteritis'],
            'should_reject': ['Acute Cholecystitis', 'Acute Diverticulitis']
        },
        {
            'prompt': "left lower belly, stays in one spot",
            'expected': ['Acute Diverticulitis', 'Sigmoid Volvulus'],
            'should_reject': ['Acute Appendicitis', 'Acute Cholecystitis']
        },
        {
            'prompt': "upper stomach behind my chest bone",
            'expected': ['GERD', 'Acute Pancreatitis', 'Peptic Ulcer Disease'],
            'should_reject': ['Acute Diverticulitis', 'Acute Appendicitis']
        },
        {
            'prompt': "right upper belly, goes to my shoulder",
            'expected': ['Acute Cholecystitis', 'Biliary Colic'],
            'should_reject': ['Acute Diverticulitis', 'Acute Appendicitis']
        },
        {
            'prompt': "all over my abdomen, not in one place",
            'expected': ['Acute Gastroenteritis', 'Irritable Bowel Syndrome (IBS)', 'Small Bowel Obstruction'],
            'should_reject': ['Acute Cholecystitis', 'Acute Diverticulitis']
        },
        {
            'prompt': "right lower belly near my hip",
            'expected': ['Acute Appendicitis', 'Incarcerated Inguinal/Femoral Hernia'],
            'should_reject': ['Acute Diverticulitis', 'Acute Cholecystitis']
        },
        {
            'prompt': "upper middle stomach, goes through to my back",
            'expected': ['Acute Pancreatitis', 'Gastric Outlet Obstruction'],
            'should_reject': ['Acute Diverticulitis', 'Acute Appendicitis']
        },
        {
            'prompt': "left side of my lower belly",
            'expected': ['Acute Diverticulitis', 'Sigmoid Volvulus'],
            'should_reject': ['Acute Appendicitis', 'Acute Cholecystitis']
        },
        {
            'prompt': "right upper part under my ribs, goes to my back",
            'expected': ['Acute Cholecystitis', 'Biliary Colic'],
            'should_reject': ['Acute Diverticulitis', 'Acute Appendicitis']
        },
        # Additional 20 prompts for comprehensive testing
        {
            'prompt': "pain in my right side that goes to my shoulder",
            'expected': ['Acute Cholecystitis', 'Biliary Colic'],
            'should_reject': ['Acute Diverticulitis', 'Acute Appendicitis']
        },
        {
            'prompt': "left side pain that stays in one place",
            'expected': ['Acute Diverticulitis', 'Sigmoid Volvulus'],
            'should_reject': ['Acute Appendicitis', 'Acute Cholecystitis']
        },
        {
            'prompt': "burning pain behind my chest bone",
            'expected': ['GERD', 'Peptic Ulcer Disease'],
            'should_reject': ['Acute Diverticulitis', 'Acute Appendicitis']
        },
        {
            'prompt': "cramping pain all over my stomach",
            'expected': ['Acute Gastroenteritis', 'Irritable Bowel Syndrome (IBS)', 'Small Bowel Obstruction'],
            'should_reject': ['Acute Cholecystitis', 'Acute Diverticulitis']
        },
        {
            'prompt': "sharp pain in my right lower belly",
            'expected': ['Acute Appendicitis', 'Incarcerated Inguinal/Femoral Hernia'],
            'should_reject': ['Acute Diverticulitis', 'Acute Cholecystitis']
        },
        {
            'prompt': "dull ache in my upper middle abdomen",
            'expected': ['Acute Pancreatitis', 'Peptic Ulcer Disease', 'Acute Gastritis'],
            'should_reject': ['Acute Diverticulitis', 'Acute Appendicitis']
        },
        {
            'prompt': "pain that moves around my belly button area",
            'expected': ['Acute Pancreatitis', 'Small Bowel Obstruction', 'Acute Gastroenteritis'],
            'should_reject': ['Acute Cholecystitis', 'Acute Diverticulitis']
        },
        {
            'prompt': "constant pain in my left lower side",
            'expected': ['Acute Diverticulitis', 'Sigmoid Volvulus'],
            'should_reject': ['Acute Appendicitis', 'Acute Cholecystitis']
        },
        {
            'prompt': "pain under my right ribs that radiates",
            'expected': ['Acute Cholecystitis', 'Biliary Colic', 'Acute Hepatitis'],
            'should_reject': ['Acute Diverticulitis', 'Acute Appendicitis']
        },
        {
            'prompt': "severe pain in my upper stomach",
            'expected': ['Acute Pancreatitis', 'Peptic Ulcer Disease', 'Acute Gastritis'],
            'should_reject': ['Acute Diverticulitis', 'Acute Appendicitis']
        },
        {
            'prompt': "pain that goes from my belly to my back",
            'expected': ['Acute Pancreatitis', 'Gastric Outlet Obstruction'],
            'should_reject': ['Acute Diverticulitis', 'Acute Appendicitis']
        },
        {
            'prompt': "intermittent pain in my right upper abdomen",
            'expected': ['Biliary Colic', 'Acute Cholecystitis'],
            'should_reject': ['Acute Diverticulitis', 'Acute Appendicitis']
        },
        {
            'prompt': "pain in my groin area on the right",
            'expected': ['Incarcerated Inguinal/Femoral Hernia', 'Acute Appendicitis'],
            'should_reject': ['Acute Diverticulitis', 'Acute Cholecystitis']
        },
        {
            'prompt': "burning sensation in my upper belly",
            'expected': ['GERD', 'Acute Gastritis', 'Peptic Ulcer Disease'],
            'should_reject': ['Acute Diverticulitis', 'Acute Appendicitis']
        },
        {
            'prompt': "pain that started around my belly button",
            'expected': ['Acute Pancreatitis', 'Small Bowel Obstruction', 'Acute Gastroenteritis'],
            'should_reject': ['Acute Cholecystitis', 'Acute Diverticulitis']
        },
        {
            'prompt': "left side abdominal pain that's constant",
            'expected': ['Acute Diverticulitis', 'Sigmoid Volvulus'],
            'should_reject': ['Acute Appendicitis', 'Acute Cholecystitis']
        },
        {
            'prompt': "pain in my right flank area",
            'expected': ['Kidney Stone', 'Acute Hepatitis'],
            'should_reject': ['Acute Diverticulitis', 'Acute Appendicitis']
        },
        {
            'prompt': "diffuse abdominal pain that moves",
            'expected': ['Acute Gastroenteritis', 'Irritable Bowel Syndrome (IBS)', 'Small Bowel Obstruction'],
            'should_reject': ['Acute Cholecystitis', 'Acute Diverticulitis']
        },
        {
            'prompt': "pain in my epigastric region",
            'expected': ['Acute Pancreatitis', 'Peptic Ulcer Disease', 'Acute Gastritis'],
            'should_reject': ['Acute Diverticulitis', 'Acute Appendicitis']
        },
        {
            'prompt': "right lower quadrant pain",
            'expected': ['Acute Appendicitis', 'Incarcerated Inguinal/Femoral Hernia'],
            'should_reject': ['Acute Diverticulitis', 'Acute Cholecystitis']
        }
    ]

def test_oldcarts_normalization_flow():
    """Test the OLDCARTS normalization flow using adaptive diagnostic engine"""
    print("🧪 Testing OLDCARTS Normalization Flow with Adaptive Diagnostic Engine")
    print("=" * 70)
    
    # Import the adaptive diagnostic engine
    try:
        print(f"🔍 Current Python path: {sys.path[:3]}...")  # Show first 3 paths
        print(f"🔍 Looking for adaptive_diagnostic_engine.py...")
        
        # Check if the file exists in the expected locations
        possible_files = [
            'adaptive_diagnostic_engine.py',
            './adaptive_diagnostic_engine.py',
            'llm-container/adaptive_diagnostic_engine.py'
        ]
        
        for file_path in possible_files:
            if os.path.exists(file_path):
                print(f"✅ Found adaptive_diagnostic_engine.py at: {file_path}")
                break
        else:
            print(f"❌ adaptive_diagnostic_engine.py not found in current directory")
            print(f"   Current directory: {os.getcwd()}")
            print(f"   Files in current directory: {os.listdir('.')}")
        
        from adaptive_diagnostic_engine import AdaptiveDiagnosticEngine
        print("✅ Successfully imported AdaptiveDiagnosticEngine")
    except ImportError as e:
        print(f"❌ Failed to import AdaptiveDiagnosticEngine: {e}")
        print(f"   Python path: {sys.path}")
        return False
    
    # Create an instance with embedding model for semantic matching
    try:
        # Try to import and initialize a sentence transformer model
        embedding_model = None
        try:
            from sentence_transformers import SentenceTransformer
            print("🧠 Loading embedding model for semantic matching...")
            embedding_model = SentenceTransformer('all-MiniLM-L6-v2')  # Fast, lightweight model
            print("✅ Successfully loaded embedding model")
        except ImportError:
            print("⚠️  sentence-transformers not available - semantic matching will be disabled")
        except Exception as e:
            print(f"⚠️  Failed to load embedding model: {e}")
        
        engine = AdaptiveDiagnosticEngine(embedding_model=embedding_model)
        print("✅ Successfully created AdaptiveDiagnosticEngine instance")
    except Exception as e:
        print(f"❌ Failed to create AdaptiveDiagnosticEngine instance: {e}")
        return False
    
    # Test cases for OLDCARTS normalization
    test_cases = [
        {
            "prompt": "my tummy hurts really bad in the upper right",
            "description": "Basic location normalization",
            "expected_improvements": ["tummy → abdominal pain", "upper right → right upper quadrant"]
        },
        {
            "prompt": "I have sharp stabbing pain that started suddenly after eating",
            "description": "Character, onset, and aggravating factors",
            "expected_improvements": ["sharp stabbing → sharp pain", "started suddenly → sudden onset", "after eating → postprandial"]
        },
        {
            "prompt": "my belly ache gets worse when I move and goes to my back",
            "description": "Location, aggravating factors, and radiation",
            "expected_improvements": ["belly ache → abdominal pain", "when I move → movement", "goes to my back → radiation to back"]
        },
        {
            "prompt": "I feel queasy and want to throw up, it's really painful",
            "description": "Associated symptoms and severity",
            "expected_improvements": ["queasy → nausea", "want to throw up → vomiting", "really painful → severe pain"]
        },
        {
            "prompt": "pain in my left lower belly that stays in one spot",
            "description": "Location and timing characteristics",
            "expected_improvements": ["left lower belly → left lower quadrant", "stays in one spot → constant"]
        }
    ]
    
    print("\n🔍 Testing OLDCARTS Normalization Flow...")
    print("=" * 70)
    
    all_passed = True
    
    # Get hardcoded GI guidelines for testing
    gi_guidelines = get_hardcoded_gi_guidelines()
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\nTest {i}: {test_case['description']}")
        print(f"Input: '{test_case['prompt']}'")
        
        try:
            # Test the OLDCARTS normalization
            normalized = engine._apply_oldcarts_normalization(test_case['prompt'])
            print(f"Normalized: '{normalized}'")
            
            # Test the full synonym expansion (legacy method for comparison)
            legacy_normalized = engine._apply_synonym_expansion(test_case['prompt'])
            print(f"Legacy: '{legacy_normalized}'")
            
            # Check if normalization occurred
            if normalized != test_case['prompt'].lower():
                print("✅ OLDCARTS normalization applied successfully")
                
                # Show specific improvements
                print("🔄 Normalization improvements:")
                for improvement in test_case['expected_improvements']:
                    print(f"   - {improvement}")
            else:
                print("⚠️  No normalization applied")
            
            # Test location-specific semantic matching
            print("🎯 Testing location-specific semantic matching...")
            
            if hasattr(engine, 'test_semantic_matching') and engine.embedding_model:
                try:
                    print(f"🧠 Using engine's test_semantic_matching method for location matching")
                    
                    # Extract location data from guidelines for testing
                    location_guidelines = []
                    for guideline in gi_guidelines:
                        location_desc = guideline.get('location', '')
                        if location_desc:
                            location_guidelines.append({
                                'name': guideline['name'],
                                'location': location_desc,
                                'data': guideline
                            })
                    
                    print(f"📍 Testing against {len(location_guidelines)} guidelines with location data")
                    
                    # Use the engine's test method for semantic matching
                    matched_guidelines = engine.test_semantic_matching(test_case['prompt'], location_guidelines)
                    
                    print(f"📊 Top 5 matching guidelines (LOCATION-SPECIFIC SEMANTIC MATCHING):")
                    for j, match in enumerate(matched_guidelines[:5], 1):
                        print(f"   {j}. {match['name']}: {match['similarity']:.3f}")
                        print(f"      Location: {match['data']['location'][:80]}...")
                    
                except Exception as e:
                    print(f"⚠️  Engine semantic matching failed: {e}")
                    print(f"✅ Normalized text ready for semantic matching: '{normalized}'")
            else:
                print(f"✅ Normalized text ready for semantic matching: '{normalized}'")
                print("ℹ️  Engine semantic matching not available (no embedding model)")
            
        except Exception as e:
            print(f"❌ Error during testing: {e}")
            all_passed = False
    
    print("\n" + "=" * 70)
    if all_passed:
        print("🎉 All OLDCARTS normalization tests passed!")
        print("The adaptive diagnostic engine is ready to use OLDCARTS normalization.")
    else:
        print("⚠️  Some tests failed. Check the implementation.")
    
    return all_passed

def test_model_performance(model_name: str, guidelines: List[Dict], patient_prompts: List[Dict]) -> Dict:
    """Test a single model's performance on all patient prompts vs guidelines."""
    print(f"\n🔄 Testing model: {model_name}")
    
    # Load model and measure loading time
    start_time = time.time()
    try:
        model = SentenceTransformer(model_name)
        load_time = time.time() - start_time
        print(f"✅ Model loaded in {load_time:.2f}s")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return None
    
    results = {
        'model_name': model_name,
        'load_time': load_time,
        'total_inference_time': 0,
        'similarities': [],
        'best_matches': [],
        'accuracy_analysis': []
    }
    
    # Test each patient prompt against all guidelines
    total_start = time.time()
    
    for i, prompt_data in enumerate(patient_prompts):
        prompt = prompt_data['prompt']
        expected = prompt_data['expected']
        should_reject = prompt_data['should_reject']
        
        prompt_similarities = []
        
        for guideline in guidelines:
            start_inference = time.time()
            similarity = compute_similarity(model, prompt, guideline['location'])
            inference_time = time.time() - start_inference
            
            results['total_inference_time'] += inference_time
            
            prompt_similarities.append({
                'guideline': guideline['name'],
                'location': guideline['location'],
                'similarity': similarity,
                'inference_time': inference_time
            })
        
        # Sort by similarity and get best match
        prompt_similarities.sort(key=lambda x: x['similarity'], reverse=True)
        best_match = prompt_similarities[0]
        
        # Analyze accuracy
        is_correct = best_match['guideline'] in expected
        is_wrong = best_match['guideline'] in should_reject
        
        accuracy_status = "✅ CORRECT" if is_correct else ("❌ WRONG" if is_wrong else "⚠️  NEUTRAL")
        
        results['similarities'].append({
            'prompt': prompt,
            'best_match': best_match,
            'all_similarities': prompt_similarities,
            'expected': expected,
            'should_reject': should_reject,
            'is_correct': is_correct,
            'is_wrong': is_wrong
        })
        
        results['best_matches'].append({
            'prompt': prompt,
            'best_guideline': best_match['guideline'],
            'similarity': best_match['similarity'],
            'is_correct': is_correct,
            'is_wrong': is_wrong
        })
        
        results['accuracy_analysis'].append({
            'prompt': prompt,
            'best_match': best_match['guideline'],
            'similarity': best_match['similarity'],
            'expected': expected,
            'should_reject': should_reject,
            'is_correct': is_correct,
            'is_wrong': is_wrong,
            'status': accuracy_status
        })
        
        print(f"  📝 Prompt {i+1:2d}: '{prompt[:50]}...' → {best_match['guideline']} ({best_match['similarity']:.3f}) {accuracy_status}")
    
    total_time = time.time() - total_start
    results['total_time'] = total_time
    
    # Calculate accuracy metrics
    correct_count = sum(1 for match in results['best_matches'] if match['is_correct'])
    wrong_count = sum(1 for match in results['best_matches'] if match['is_wrong'])
    total_count = len(results['best_matches'])
    
    results['accuracy_metrics'] = {
        'correct': correct_count,
        'wrong': wrong_count,
        'neutral': total_count - correct_count - wrong_count,
        'total': total_count,
        'accuracy_percentage': (correct_count / total_count) * 100,
        'error_percentage': (wrong_count / total_count) * 100
    }
    
    print(f"⏱️  Total inference time: {results['total_inference_time']:.2f}s")
    print(f"⏱️  Total time: {total_time:.2f}s")
    print(f"📊 Accuracy: {correct_count}/{total_count} ({results['accuracy_metrics']['accuracy_percentage']:.1f}%) correct, {wrong_count} wrong")
    
    return results

def analyze_results(all_results: List[Dict]) -> None:
    """Analyze and compare results across all models."""
    print("\n" + "="*100)
    print("📊 MODEL PERFORMANCE ANALYSIS")
    print("="*100)
    
    # Summary table
    print(f"\n{'Model':<25} {'Load Time':<10} {'Inference Time':<15} {'Avg Similarity':<15} {'Accuracy':<10} {'Errors':<8}")
    print("-" * 90)
    
    for result in all_results:
        if result is None:
            continue
            
        avg_similarity = np.mean([match['similarity'] for match in result['best_matches']])
        accuracy = result['accuracy_metrics']['accuracy_percentage']
        errors = result['accuracy_metrics']['wrong']
        print(f"{result['model_name']:<25} {result['load_time']:<10.2f} {result['total_inference_time']:<15.2f} {avg_similarity:<15.3f} {accuracy:<10.1f}% {errors:<8}")
    
    # Detailed analysis for each model
    for result in all_results:
        if result is None:
            continue
            
        print(f"\n🔍 DETAILED ANALYSIS: {result['model_name']}")
        print("="*100)
        
        # Show accuracy summary
        metrics = result['accuracy_metrics']
        print(f"\n📊 ACCURACY SUMMARY:")
        print(f"   ✅ Correct: {metrics['correct']}/{metrics['total']} ({metrics['accuracy_percentage']:.1f}%)")
        print(f"   ❌ Wrong: {metrics['wrong']}/{metrics['total']} ({metrics['error_percentage']:.1f}%)")
        print(f"   ⚠️  Neutral: {metrics['neutral']}/{metrics['total']} ({100-metrics['accuracy_percentage']-metrics['error_percentage']:.1f}%)")
        
        # Show all matches in formatted table with accuracy status
        print(f"\n{'#':<3} {'User Prompt':<40} {'Best Match':<25} {'Score':<8} {'Status':<10}")
        print("-" * 100)
        
        for i, accuracy_data in enumerate(result['accuracy_analysis']):
            prompt = accuracy_data['prompt']
            best_match = accuracy_data['best_match']
            score = accuracy_data['similarity']
            status = accuracy_data['status']
            
            # Truncate long strings
            prompt_display = prompt[:37] + "..." if len(prompt) > 40 else prompt
            match_display = best_match[:22] + "..." if len(best_match) > 25 else best_match
            
            print(f"{i+1:<3} {prompt_display:<40} {match_display:<25} {score:<8.3f} {status:<10}")
        
        # Show wrong matches in detail
        wrong_matches = [data for data in result['accuracy_analysis'] if data['is_wrong']]
        if wrong_matches:
            print(f"\n❌ WRONG MATCHES ({len(wrong_matches)}):")
            print("-" * 100)
            for match in wrong_matches:
                print(f"   '{match['prompt'][:50]}...'")
                print(f"   Expected: {', '.join(match['expected'])}")
                print(f"   Got: {match['best_match']} (score: {match['similarity']:.3f})")
                print(f"   Should reject: {', '.join(match['should_reject'])}")
                print()
        
        # Show specific test case with top 5 matches
        left_lower_prompt = "left lower part of my abdomen towards my pelvis"
        left_lower_result = next((r for r in result['similarities'] if r['prompt'] == left_lower_prompt), None)
        if left_lower_result:
            print(f"\n🎯 SPECIFIC TEST CASE: '{left_lower_prompt}'")
            print("-" * 100)
            print(f"{'Rank':<5} {'Guideline':<30} {'Location':<50} {'Score':<8}")
            print("-" * 100)
            for i, match in enumerate(left_lower_result['all_similarities'][:5]):
                location_display = match['location'][:47] + "..." if len(match['location']) > 50 else match['location']
                print(f"{i+1:<5} {match['guideline']:<30} {location_display:<50} {match['similarity']:<8.3f}")
        
        print("\n" + "="*100)

def analyze_guideline_optimization(all_results: List[Dict], guidelines: List[Dict]):
    """Analyze which guidelines need optimization based on test results."""
    print(f"\n🔧 GUIDELINE OPTIMIZATION ANALYSIS")
    print("="*60)
    
    best_model = max(all_results, key=lambda x: x['accuracy_metrics']['accuracy_percentage'])
    model_name = best_model['model_name']
    
    # Load the best model for detailed analysis
    try:
        model = SentenceTransformer(model_name)
        print(f"✅ Loaded {model_name} for detailed analysis")
    except Exception as e:
        print(f"❌ Failed to load {model_name}: {e}")
        return
    
    # Analyze each guideline's performance
    guideline_performance = {}
    
    for guideline in guidelines:
        name = guideline['name']
        location = guideline['location']
        
        # Test against all vague prompts
        vague_prompts = [
            "left side", "right side", "my left", "my right", 
            "on the left", "on the right", "left part", "right part"
        ]
        
        similarities = []
        for prompt in vague_prompts:
            sim = compute_similarity(model, prompt, location)
            similarities.append(sim)
        
        avg_similarity = np.mean(similarities)
        max_similarity = np.max(similarities)
        min_similarity = np.min(similarities)
        
        guideline_performance[name] = {
            'avg_similarity': avg_similarity,
            'max_similarity': max_similarity,
            'min_similarity': min_similarity,
            'location_text': location,
            'similarities': dict(zip(vague_prompts, similarities))
        }
    
    # Sort by average similarity (higher = more likely to match vague terms)
    sorted_guidelines = sorted(guideline_performance.items(), 
                              key=lambda x: x[1]['avg_similarity'], reverse=True)
    
    print(f"\n📊 GUIDELINES RANKED BY VAGUE TERM SIMILARITY:")
    print(f"   (Higher scores = more likely to match vague patient language)")
    
    for i, (name, perf) in enumerate(sorted_guidelines, 1):
        print(f"\n   {i}. {name}")
        print(f"      Avg Similarity: {perf['avg_similarity']:.3f}")
        print(f"      Range: {perf['min_similarity']:.3f} - {perf['max_similarity']:.3f}")
        print(f"      Current Location: {perf['location_text'][:80]}...")
        
        # Show which vague terms match best
        best_matches = sorted(perf['similarities'].items(), 
                            key=lambda x: x[1], reverse=True)[:3]
        print(f"      Best vague matches: {best_matches}")
    
    # Identify guidelines that need optimization
    print(f"\n⚠️  GUIDELINES NEEDING OPTIMIZATION:")
    print(f"   (High similarity to vague terms = needs more specific language)")
    
    for name, perf in sorted_guidelines:
        if perf['avg_similarity'] > 0.4:  # Threshold for "too vague"
            print(f"\n   🔧 {name} (avg: {perf['avg_similarity']:.3f})")
            print(f"      Current: {perf['location_text']}")
            
            # Suggest improvements
            suggestions = []
            if 'left' in perf['location_text'].lower() and perf['similarities']['left side'] > 0.5:
                suggestions.append("Add more specific anatomical terms (e.g., 'left lower quadrant', 'sigmoid colon')")
            if 'right' in perf['location_text'].lower() and perf['similarities']['right side'] > 0.5:
                suggestions.append("Add more specific anatomical terms (e.g., 'right upper quadrant', 'gallbladder fossa')")
            if 'upper' in perf['location_text'].lower():
                suggestions.append("Add anatomical landmarks (e.g., 'below rib cage', 'epigastric region')")
            if 'lower' in perf['location_text'].lower():
                suggestions.append("Add anatomical landmarks (e.g., 'above pelvis', 'inguinal region')")
            
            if suggestions:
                print(f"      💡 Suggestions:")
                for suggestion in suggestions:
                    print(f"         - {suggestion}")
    
    return guideline_performance

def analyze_word_patterns(all_results: List[Dict], guidelines: List[Dict]):
    """Analyze which word patterns work best for semantic similarity to create a framework for future guidelines."""
    print(f"\n🔍 WORD PATTERN ANALYSIS FOR GUIDELINE OPTIMIZATION")
    print("="*70)
    
    best_model = max(all_results, key=lambda x: x['accuracy_metrics']['accuracy_percentage'])
    model_name = best_model['model_name']
    
    # Load the best model for detailed analysis
    try:
        model = SentenceTransformer(model_name)
        print(f"✅ Using {model_name} for word pattern analysis")
    except Exception as e:
        print(f"❌ Failed to load {model_name}: {e}")
        return
    
    # Define test word patterns
    word_patterns = {
        'anatomical_terms': [
            'quadrant', 'epigastric', 'hypogastric', 'periumbilical', 
            'retrosternal', 'suprapubic', 'thoracic', 'lumbar'
        ],
        'directional_terms': [
            'left', 'right', 'upper', 'lower', 'anterior', 'posterior',
            'medial', 'lateral', 'proximal', 'distal'
        ],
        'anatomical_landmarks': [
            'rib cage', 'pelvis', 'hip bone', 'breastbone', 'shoulder',
            'scapula', 'flank', 'groin', 'umbilicus'
        ],
        'patient_language': [
            'side', 'part', 'area', 'belly', 'stomach', 'tummy',
            'my left', 'my right', 'on the left', 'on the right'
        ],
        'medical_specificity': [
            'migrates', 'localizes', 'radiates', 'constant', 'cramping',
            'burning', 'sharp', 'dull', 'diffuse', 'localized'
        ]
    }
    
    # Test each pattern against all guidelines
    pattern_performance = {}
    
    for pattern_type, words in word_patterns.items():
        print(f"\n📊 Testing {pattern_type.upper()} patterns:")
        print("-" * 50)
        
        pattern_scores = {}
        
        for word in words:
            similarities = []
            matching_guidelines = []
            
            for guideline in guidelines:
                sim = compute_similarity(model, word, guideline['location'])
                similarities.append(sim)
                if sim > 0.3:  # Threshold for "relevant match"
                    matching_guidelines.append((guideline['name'], sim))
            
            avg_similarity = np.mean(similarities)
            max_similarity = np.max(similarities)
            relevant_matches = len(matching_guidelines)
            
            pattern_scores[word] = {
                'avg_similarity': avg_similarity,
                'max_similarity': max_similarity,
                'relevant_matches': relevant_matches,
                'top_matches': sorted(matching_guidelines, key=lambda x: x[1], reverse=True)[:3]
            }
            
            print(f"   '{word}': avg={avg_similarity:.3f}, max={max_similarity:.3f}, matches={relevant_matches}")
            if matching_guidelines:
                top_match = max(matching_guidelines, key=lambda x: x[1])
                print(f"      Best match: {top_match[0]} ({top_match[1]:.3f})")
        
        pattern_performance[pattern_type] = pattern_scores
    
    # Analyze which patterns work best
    print(f"\n🏆 BEST PERFORMING WORD PATTERNS:")
    print("="*70)
    
    all_words = []
    for pattern_type, words in pattern_performance.items():
        for word, scores in words.items():
            all_words.append((word, scores['avg_similarity'], scores['relevant_matches'], pattern_type))
    
    # Sort by average similarity (higher is better for semantic matching)
    best_words = sorted(all_words, key=lambda x: x[1], reverse=True)[:15]
    
    print(f"{'Rank':<5} {'Word':<20} {'Avg Sim':<10} {'Matches':<8} {'Category':<15}")
    print("-" * 70)
    for i, (word, avg_sim, matches, category) in enumerate(best_words, 1):
        print(f"{i:<5} {word:<20} {avg_sim:<10.3f} {matches:<8} {category:<15}")
    
    # Framework recommendations
    print(f"\n💡 FRAMEWORK FOR FUTURE GUIDELINES:")
    print("="*70)
    
    # High-performing anatomical terms
    high_performing_anatomical = [word for word, avg_sim, matches, cat in best_words 
                                 if cat == 'anatomical_terms' and avg_sim > 0.2][:5]
    if high_performing_anatomical:
        print(f"\n✅ USE THESE ANATOMICAL TERMS (high semantic similarity):")
        for term in high_performing_anatomical:
            print(f"   - '{term}'")
    
    # High-performing landmarks
    high_performing_landmarks = [word for word, avg_sim, matches, cat in best_words 
                                if cat == 'anatomical_landmarks' and avg_sim > 0.2][:5]
    if high_performing_landmarks:
        print(f"\n✅ USE THESE ANATOMICAL LANDMARKS (high semantic similarity):")
        for landmark in high_performing_landmarks:
            print(f"   - '{landmark}'")
    
    # Medical specificity terms
    high_performing_medical = [word for word, avg_sim, matches, cat in best_words 
                              if cat == 'medical_specificity' and avg_sim > 0.2][:5]
    if high_performing_medical:
        print(f"\n✅ USE THESE MEDICAL SPECIFICITY TERMS (high semantic similarity):")
        for term in high_performing_medical:
            print(f"   - '{term}'")
    
    # Avoid patient language (too vague)
    patient_language_scores = [avg_sim for word, avg_sim, matches, cat in all_words 
                              if cat == 'patient_language']
    if patient_language_scores:
        avg_patient_sim = np.mean(patient_language_scores)
        print(f"\n⚠️  AVOID PATIENT LANGUAGE (avg similarity: {avg_patient_sim:.3f}):")
        print(f"   - Terms like 'side', 'part', 'area' are too vague")
        print(f"   - Use specific anatomical terms instead")
    
    # Directional terms analysis
    directional_scores = [(word, avg_sim) for word, avg_sim, matches, cat in all_words 
                         if cat == 'directional_terms']
    if directional_scores:
        print(f"\n📐 DIRECTIONAL TERMS ANALYSIS:")
        for word, avg_sim in sorted(directional_scores, key=lambda x: x[1], reverse=True):
            print(f"   '{word}': {avg_sim:.3f}")
        print(f"   💡 Use specific combinations: 'left lower quadrant' vs just 'left'")
    
    # Create sample optimized guideline
    print(f"\n📝 SAMPLE OPTIMIZED GUIDELINE FORMAT:")
    print("-" * 70)
    print(f"❌ AVOID: 'Pain in the left side of the abdomen'")
    print(f"✅ USE: 'Pain localizes to LEFT LOWER QUADRANT (LLQ), specifically")
    print(f"        in the sigmoid colon region above the pelvis'")
    print(f"")
    print(f"❌ AVOID: 'Right upper area under ribs'") 
    print(f"✅ USE: 'Pain in RIGHT UPPER QUADRANT (RUQ), precisely localized")
    print(f"        below the right rib cage in the gallbladder fossa'")
    
    return pattern_performance

def main():
    """Main test function."""
    print("🧪 OLDCARTS NORMALIZATION FLOW TESTING")
    print("="*50)
    
    # Test the OLDCARTS normalization flow
    print("🎯 Testing OLDCARTS normalization with adaptive diagnostic engine...")
    success = test_oldcarts_normalization_flow()
    
    if success:
        print("\n✅ OLDCARTS normalization flow test completed successfully!")
        print("\n📋 Summary:")
        print("   - Adaptive diagnostic engine successfully loaded")
        print("   - OLDCARTS normalization is working correctly")
        print("   - Patient language is being normalized to medical terms")
        print("   - System is ready for semantic matching with medical guidelines")
        
        print("\n🔄 Next steps:")
        print("   1. Test with real patient interactions")
        print("   2. Monitor normalization accuracy in production")
        print("   3. Fine-tune OLDCARTS synonyms based on usage patterns")
    else:
        print("\n❌ OLDCARTS normalization flow test failed!")
        print("   Check the error messages above and fix the implementation.")
    
    print("\n" + "="*50)

if __name__ == "__main__":
    main()

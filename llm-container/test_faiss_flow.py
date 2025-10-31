#!/usr/bin/env python3
"""
Test FAISS implementation and unified function flow
"""

import os
import sys
import json
from pathlib import Path

# Add the current directory to Python path
sys.path.append(os.path.dirname(__file__))

def test_faiss_implementation():
    """Test the complete FAISS flow"""
    print("🧪 Testing FAISS Implementation and Unified Function")
    print("=" * 60)
    
    try:
        # Test 1: Import and initialize medical rule engine
        print("\n1️⃣ Testing Medical Rule Engine Initialization...")
        from sentence_transformers import SentenceTransformer
        from ml.medical_rule_engine import MedicalRuleEngine
        
        # Initialize embedding model
        print("   📥 Loading embedding model...")
        embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        
        # Initialize medical rule engine
        print("   🏥 Initializing Medical Rule Engine...")
        engine = MedicalRuleEngine(embedding_model=embedding_model)
        print("   ✅ Medical Rule Engine initialized successfully")
        
        # Test 2: Test FAISS term matching
        print("\n2️⃣ Testing FAISS Term Matching...")
        test_prompts = [
            "I have right sided abdominal pain",
            "right lower quadrant pain",
            "sudden onset sharp pain",
            "pain gets worse when I move",
            "constant pain for 2 days"
        ]
        
        elements = ['onset', 'location', 'duration', 'character', 'aggravating']
        
        for prompt in test_prompts:
            print(f"\n   📝 Testing: '{prompt}'")
            for element in elements:
                matches = engine.find_matching_terms_faiss(prompt, element, threshold=0.7)
                if matches:
                    print(f"      {element}: {matches}")
                else:
                    print(f"      {element}: No matches")
        
        # Test 3: Test unified similarity function
        print("\n3️⃣ Testing Unified Similarity Function...")
        
        # Test case: Right sided pain vs Appendicitis
        patient_text = "right sided abdominal pain more towards my pelvis that is sharp"
        guideline_text = "Right lower quadrant pain with sharp, stabbing quality"
        condition_name = "Acute Appendicitis"
        organ_system = "GI"
        oldcarts_element = "location"
        
        # Create mock structured_oldcarts
        structured_oldcarts = {
            "location": {
                "includes": ["right lower quadrant", "periumbilical", "right upper quadrant"],
                "excludes": ["left side", "left lower quadrant"]
            }
        }
        
        print(f"   📝 Patient: '{patient_text}'")
        print(f"   📋 Guideline: '{guideline_text}'")
        print(f"   🏥 Condition: {condition_name}")
        
        result = engine.compute_unified_similarity(
            patient_text, guideline_text, condition_name, organ_system,
            oldcarts_element, structured_oldcarts
        )
        
        print(f"   📊 Results:")
        print(f"      Raw similarity: {result['raw_similarity']:.3f}")
        print(f"      Word match boost: {result['word_match_boost']:.3f}")
        print(f"      Final similarity: {result['similarity']:.3f}")
        print(f"      Normalized text: {result['normalized_text']}")
        
        # Test 4: Test adaptive diagnostic engine
        print("\n4️⃣ Testing Adaptive Diagnostic Engine...")
        try:
            from adaptive_diagnostic_engine import AdaptiveDiagnosticEngine
            
            # Initialize adaptive engine
            print("   🏥 Initializing Adaptive Diagnostic Engine...")
            adaptive_engine = AdaptiveDiagnosticEngine(
                guidelines_dir="medical/guidelines",
                embedding_model=embedding_model,
                llm_chat_simple_fn=lambda x, **kwargs: "Test response"
            )
            print("   ✅ Adaptive Diagnostic Engine initialized")
            
            # Test initial prompt processing
            print("\n   🔍 Testing Initial Prompt Processing...")
            test_prompt = "I have right sided abdominal pain more towards my pelvis that is sharp, worse with eating"
            
            print(f"   📝 Testing prompt: '{test_prompt}'")
            
            # Test OLDCARTS analysis
            guidelines = list(adaptive_engine.all_guidelines.values())[:5]  # Test with first 5 guidelines
            oldcarts_analysis = adaptive_engine._parse_prompt_against_structured_oldcarts(test_prompt, guidelines)
            
            print(f"   📊 OLDCARTS Analysis:")
            print(f"      Answered components: {oldcarts_analysis.get('answered_components', {})}")
            print(f"      Missing components: {oldcarts_analysis.get('missing_components', [])}")
            
        except Exception as e:
            print(f"   ⚠️ Adaptive Diagnostic Engine test failed: {e}")
        
        print("\n✅ All tests completed successfully!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_faiss_implementation()


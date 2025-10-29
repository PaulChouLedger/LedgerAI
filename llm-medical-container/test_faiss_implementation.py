#!/usr/bin/env python3
"""
Test FAISS implementation for term matching
"""

import os
import sys
sys.path.append('/app/ml')

from sentence_transformers import SentenceTransformer
from ml.medical_rule_engine import MedicalRuleEngine

def test_faiss_implementation():
    """Test FAISS-based term matching"""
    print("🔨 Testing FAISS implementation...")
    
    # Initialize embedding model
    print("📥 Loading embedding model...")
    embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    
    # Initialize medical rule engine
    print("🏥 Initializing Medical Rule Engine...")
    engine = MedicalRuleEngine(embedding_model=embedding_model)
    
    # Test term matching
    test_prompts = [
        "I have abdominal pain",
        "right lower quadrant pain",
        "sudden onset sharp pain",
        "pain gets worse when I move",
        "constant pain for 2 days"
    ]
    
    elements = ['onset', 'location', 'duration', 'character', 'aggravating']
    
    print("\n🧪 Testing term matching:")
    for prompt in test_prompts:
        print(f"\n📝 Prompt: '{prompt}'")
        for element in elements:
            matches = engine.find_matching_terms_faiss(prompt, element, threshold=0.7)
            if matches:
                print(f"  {element}: {matches}")
            else:
                print(f"  {element}: No matches")
    
    print("\n✅ FAISS implementation test completed!")

if __name__ == "__main__":
    test_faiss_implementation()


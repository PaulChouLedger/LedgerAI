#!/usr/bin/env python3
"""
Test if embeddings are actually different for different texts
"""

import numpy as np
from sentence_transformers import SentenceTransformer

def test_embedding_differences():
    """Test if different texts produce different embeddings"""
    
    # Load the same model as the engine
    model = SentenceTransformer("all-mpnet-base-v2")
    
    # Test cases
    test_cases = [
        ("chest pain", "abdominal pain"),
        ("chest pain", "chest pain"),  # Same text
        ("left arm", "right arm"),
        ("upper right abdomen", "lower left abdomen"),
        ("crushing chest pain", "mild chest discomfort")
    ]
    
    print("🧪 Testing embedding differences...")
    print("=" * 60)
    
    for text1, text2 in test_cases:
        # Generate embeddings
        emb1 = model.encode([text1])[0]
        emb2 = model.encode([text2])[0]
        
        # Compute similarity
        similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        
        # Check if embeddings are identical
        are_identical = np.allclose(emb1, emb2, atol=1e-6)
        
        print(f"Text 1: '{text1}'")
        print(f"Text 2: '{text2}'")
        print(f"  Embedding 1 norm: {np.linalg.norm(emb1):.6f}")
        print(f"  Embedding 2 norm: {np.linalg.norm(emb2):.6f}")
        print(f"  Are identical: {are_identical}")
        print(f"  Cosine similarity: {similarity:.6f}")
        print(f"  First 5 values emb1: {emb1[:5]}")
        print(f"  First 5 values emb2: {emb2[:5]}")
        print("-" * 40)
    
    # Test with your actual test case
    print("\n🔍 Testing your actual case:")
    text1 = "crushing chest pain that radiates to my left arm"
    text2 = "RETROSTERNAL or left chest. RADIATES to left arm, jaw, neck, back."
    
    emb1 = model.encode([text1])[0]
    emb2 = model.encode([text2])[0]
    similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
    
    print(f"Text 1: '{text1}'")
    print(f"Text 2: '{text2}'")
    print(f"  Similarity: {similarity:.6f}")
    print(f"  Embedding 1 first 10: {emb1[:10]}")
    print(f"  Embedding 2 first 10: {emb2[:10]}")
    print(f"  Are identical: {np.allclose(emb1, emb2, atol=1e-6)}")

if __name__ == "__main__":
    test_embedding_differences()

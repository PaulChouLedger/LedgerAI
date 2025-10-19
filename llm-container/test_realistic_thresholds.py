#!/usr/bin/env python3
"""
Test with realistic thresholds for medical text matching
"""

import numpy as np
import requests
import re
from collections import Counter

def get_embeddings(texts):
    """Get embeddings from RAG container"""
    response = requests.post(
        "http://localhost:11435/embed",
        json={"texts": texts},
        timeout=5
    )
    
    if response.status_code != 200:
        raise RuntimeError(f"Embedding API failed: {response.status_code}")
    
    return response.json()["embeddings"]

def cosine_similarity(emb1, emb2):
    """Standard cosine similarity"""
    return np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))

def jaccard_similarity(text1, text2):
    """Jaccard similarity based on word overlap"""
    words1 = set(re.findall(r'\b\w+\b', text1.lower()))
    words2 = set(re.findall(r'\b\w+\b', text2.lower()))
    
    intersection = len(words1.intersection(words2))
    union = len(words1.union(words2))
    
    return intersection / union if union > 0 else 0

def hybrid_similarity(text1, text2, emb1, emb2, cosine_weight=0.3, jaccard_weight=0.7):
    """Combine cosine and Jaccard similarity"""
    cosine_sim = cosine_similarity(emb1, emb2)
    jaccard_sim = jaccard_similarity(text1, text2)
    
    return cosine_weight * cosine_sim + jaccard_weight * jaccard_sim

def test_with_realistic_thresholds():
    """Test with realistic thresholds for medical matching"""
    
    # Test cases with expected results
    test_cases = [
        # (text1, text2, expected_result, description)
        ("chest pain", "chest pain", "ACCEPT", "Same text"),
        ("chest pain", "abdominal pain", "REJECT", "Different body parts"),
        ("left arm pain", "right arm pain", "ACCEPT", "Different sides - same body part"),
        ("upper right abdomen", "lower left abdomen", "ACCEPT", "Different quadrants - same body part"),
        ("crushing chest pain", "mild chest discomfort", "ACCEPT", "Different severity - same body part"),
        ("crushing chest pain that radiates to my left arm", "RETROSTERNAL or left chest. RADIATES to left arm, jaw, neck, back.", "ACCEPT", "Your test case - should match"),
        ("abdominal pain", "chest pain", "REJECT", "Different body parts - should not match"),
        ("left lower quadrant pain", "right lower quadrant pain", "ACCEPT", "Different sides - same quadrant type"),
        ("severe headache", "mild headache", "ACCEPT", "Different severity - same symptom"),
        ("headache", "stomach pain", "REJECT", "Completely different symptoms")
    ]
    
    print("🧪 Testing with Realistic Thresholds")
    print("=" * 80)
    
    # Test different threshold combinations
    threshold_configs = [
        {"cosine": 0.4, "jaccard": 0.3, "hybrid": 0.4, "name": "Realistic (0.4/0.3/0.4)"},
        {"cosine": 0.5, "jaccard": 0.4, "hybrid": 0.5, "name": "Moderate (0.5/0.4/0.5)"},
        {"cosine": 0.6, "jaccard": 0.5, "hybrid": 0.6, "name": "Strict (0.6/0.5/0.6)"},
        {"cosine": 0.3, "jaccard": 0.2, "hybrid": 0.3, "name": "Lenient (0.3/0.2/0.3)"}
    ]
    
    for config in threshold_configs:
        print(f"\n📊 {config['name']} Thresholds:")
        print("-" * 50)
        
        correct_predictions = 0
        total_predictions = 0
        
        for text1, text2, expected, description in test_cases:
            # Get embeddings
            embeddings = get_embeddings([text1, text2])
            emb1 = np.array(embeddings[0])
            emb2 = np.array(embeddings[1])
            
            # Calculate similarities
            cosine_sim = cosine_similarity(emb1, emb2)
            jaccard_sim = jaccard_similarity(text1, text2)
            hybrid_sim = hybrid_similarity(text1, text2, emb1, emb2)
            
            # Apply thresholds
            cosine_result = "ACCEPT" if cosine_sim > config["cosine"] else "REJECT"
            jaccard_result = "ACCEPT" if jaccard_sim > config["jaccard"] else "REJECT"
            hybrid_result = "ACCEPT" if hybrid_sim > config["hybrid"] else "REJECT"
            
            # Check if predictions match expected
            cosine_correct = cosine_result == expected
            jaccard_correct = jaccard_result == expected
            hybrid_correct = hybrid_result == expected
            
            if cosine_correct:
                correct_predictions += 1
            total_predictions += 1
            
            # Show results for important cases
            if description in ["Your test case - should match", "Different body parts - should not match"]:
                print(f"  {description}:")
                print(f"    Cosine: {cosine_sim:.3f} → {cosine_result} {'✅' if cosine_correct else '❌'}")
                print(f"    Jaccard: {jaccard_sim:.3f} → {jaccard_result} {'✅' if jaccard_correct else '❌'}")
                print(f"    Hybrid: {hybrid_sim:.3f} → {hybrid_result} {'✅' if hybrid_correct else '❌'}")
        
        # Calculate accuracy
        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
        print(f"\n  📈 Cosine Accuracy: {accuracy:.1%} ({correct_predictions}/{total_predictions})")
    
    # Find optimal thresholds
    print("\n" + "=" * 80)
    print("🎯 Finding Optimal Thresholds")
    print("=" * 80)
    
    # Test your specific case with different thresholds
    text1 = "crushing chest pain that radiates to my left arm"
    text2 = "RETROSTERNAL or left chest. RADIATES to left arm, jaw, neck, back."
    
    embeddings = get_embeddings([text1, text2])
    emb1 = np.array(embeddings[0])
    emb2 = np.array(embeddings[1])
    
    cosine_sim = cosine_similarity(emb1, emb2)
    jaccard_sim = jaccard_similarity(text1, text2)
    hybrid_sim = hybrid_similarity(text1, text2, emb1, emb2)
    
    print(f"Your test case similarities:")
    print(f"  Cosine: {cosine_sim:.3f}")
    print(f"  Jaccard: {jaccard_sim:.3f}")
    print(f"  Hybrid: {hybrid_sim:.3f}")
    print()
    
    # Find thresholds that would accept this case
    print("Thresholds that would ACCEPT your test case:")
    print(f"  Cosine threshold: < {cosine_sim:.3f}")
    print(f"  Jaccard threshold: < {jaccard_sim:.3f}")
    print(f"  Hybrid threshold: < {hybrid_sim:.3f}")
    
    # Recommended thresholds
    print("\n🎯 Recommended Thresholds:")
    print(f"  Cosine: {cosine_sim - 0.05:.2f} (slightly below your case)")
    print(f"  Jaccard: {jaccard_sim - 0.05:.2f} (slightly below your case)")
    print(f"  Hybrid: {hybrid_sim - 0.05:.2f} (slightly below your case)")

if __name__ == "__main__":
    test_with_realistic_thresholds()

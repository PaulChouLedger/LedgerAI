#!/usr/bin/env python3
"""
Test hybrid similarity: Cosine + Jaccard (no hardcoded medical terms)
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
    # Normalize and tokenize
    words1 = set(re.findall(r'\b\w+\b', text1.lower()))
    words2 = set(re.findall(r'\b\w+\b', text2.lower()))
    
    intersection = len(words1.intersection(words2))
    union = len(words1.union(words2))
    
    return intersection / union if union > 0 else 0

def tf_idf_similarity(text1, text2):
    """TF-IDF based similarity (no hardcoded terms)"""
    # Tokenize
    words1 = re.findall(r'\b\w+\b', text1.lower())
    words2 = re.findall(r'\b\w+\b', text2.lower())
    
    # Count frequencies
    freq1 = Counter(words1)
    freq2 = Counter(words2)
    
    # Get all unique words
    all_words = set(words1 + words2)
    
    # Calculate TF-IDF vectors (simplified - treating each text as a document)
    tf_idf1 = []
    tf_idf2 = []
    
    for word in all_words:
        # TF: term frequency in document
        tf1 = freq1.get(word, 0) / len(words1) if words1 else 0
        tf2 = freq2.get(word, 0) / len(words2) if words2 else 0
        
        # IDF: inverse document frequency (simplified - just use word length as proxy)
        # Longer words are more specific, shorter words are more common
        idf = len(word) / 10.0  # Normalize by average word length
        
        tf_idf1.append(tf1 * idf)
        tf_idf2.append(tf2 * idf)
    
    # Convert to numpy arrays
    tf_idf1 = np.array(tf_idf1)
    tf_idf2 = np.array(tf_idf2)
    
    # Calculate cosine similarity of TF-IDF vectors
    if np.linalg.norm(tf_idf1) == 0 or np.linalg.norm(tf_idf2) == 0:
        return 0
    
    return np.dot(tf_idf1, tf_idf2) / (np.linalg.norm(tf_idf1) * np.linalg.norm(tf_idf2))

def hybrid_similarity(text1, text2, emb1, emb2, cosine_weight=0.5, jaccard_weight=0.5):
    """Combine cosine and Jaccard similarity"""
    cosine_sim = cosine_similarity(emb1, emb2)
    jaccard_sim = jaccard_similarity(text1, text2)
    
    # Weighted combination
    return cosine_weight * cosine_sim + jaccard_weight * jaccard_sim

def adaptive_hybrid_similarity(text1, text2, emb1, emb2):
    """Adaptive weighting based on text characteristics"""
    cosine_sim = cosine_similarity(emb1, emb2)
    jaccard_sim = jaccard_similarity(text1, text2)
    
    # Analyze text characteristics
    words1 = re.findall(r'\b\w+\b', text1.lower())
    words2 = re.findall(r'\b\w+\b', text2.lower())
    
    # If texts are very different in length, favor Jaccard
    length_ratio = min(len(words1), len(words2)) / max(len(words1), len(words2))
    
    # If many common words, favor Jaccard
    common_words = len(set(words1).intersection(set(words2)))
    total_words = len(set(words1).union(set(words2)))
    word_overlap_ratio = common_words / total_words if total_words > 0 else 0
    
    # Adaptive weights
    if word_overlap_ratio > 0.3:  # High word overlap
        cosine_weight = 0.3
        jaccard_weight = 0.7
    elif length_ratio < 0.5:  # Very different lengths
        cosine_weight = 0.3
        jaccard_weight = 0.7
    else:  # Balanced case
        cosine_weight = 0.5
        jaccard_weight = 0.5
    
    return cosine_weight * cosine_sim + jaccard_weight * jaccard_sim

def test_hybrid_similarities():
    """Test different hybrid approaches"""
    
    # Your test case
    text1 = "crushing chest pain that radiates to my left arm"
    text2 = "RETROSTERNAL or left chest. RADIATES to left arm, jaw, neck, back."
    
    print("🧪 Testing Hybrid Similarity Methods (No Hardcoding)")
    print("=" * 70)
    print(f"Text 1: '{text1}'")
    print(f"Text 2: '{text2}'")
    print()
    
    # Get embeddings
    embeddings = get_embeddings([text1, text2])
    emb1 = np.array(embeddings[0])
    emb2 = np.array(embeddings[1])
    
    # Test different methods
    methods = {
        "Cosine Similarity": cosine_similarity(emb1, emb2),
        "Jaccard Similarity": jaccard_similarity(text1, text2),
        "TF-IDF Similarity": tf_idf_similarity(text1, text2),
        "Hybrid (50% cosine + 50% jaccard)": hybrid_similarity(text1, text2, emb1, emb2, 0.5, 0.5),
        "Hybrid (30% cosine + 70% jaccard)": hybrid_similarity(text1, text2, emb1, emb2, 0.3, 0.7),
        "Hybrid (70% cosine + 30% jaccard)": hybrid_similarity(text1, text2, emb1, emb2, 0.7, 0.3),
        "Adaptive Hybrid": adaptive_hybrid_similarity(text1, text2, emb1, emb2)
    }
    
    for method, score in methods.items():
        status = "✅ ACCEPT" if score > 0.6 else "❌ REJECT"
        print(f"{method:40}: {score:.3f} {status}")
    
    print()
    print("🔍 Text Analysis:")
    words1 = set(re.findall(r'\b\w+\b', text1.lower()))
    words2 = set(re.findall(r'\b\w+\b', text2.lower()))
    
    common_words = words1.intersection(words2)
    unique_to_1 = words1 - words2
    unique_to_2 = words2 - words1
    
    print(f"  Text 1 words: {len(words1)}")
    print(f"  Text 2 words: {len(words2)}")
    print(f"  Common words: {len(common_words)} ({sorted(common_words)})")
    print(f"  Unique to text1: {len(unique_to_1)} ({sorted(unique_to_1)})")
    print(f"  Unique to text2: {len(unique_to_2)} ({sorted(unique_to_2)})")
    print(f"  Word overlap ratio: {len(common_words) / len(words1.union(words2)):.3f}")
    
    # Test with more cases
    print("\n" + "=" * 70)
    print("🧪 Testing Multiple Cases")
    print("=" * 70)
    
    test_cases = [
        ("chest pain", "chest pain", "Same text"),
        ("chest pain", "abdominal pain", "Different body parts"),
        ("left arm pain", "right arm pain", "Different sides"),
        ("upper right abdomen", "lower left abdomen", "Different quadrants"),
        ("crushing chest pain", "mild chest discomfort", "Different severity")
    ]
    
    for text1, text2, description in test_cases:
        embeddings = get_embeddings([text1, text2])
        emb1 = np.array(embeddings[0])
        emb2 = np.array(embeddings[1])
        
        cosine_sim = cosine_similarity(emb1, emb2)
        jaccard_sim = jaccard_similarity(text1, text2)
        hybrid_sim = hybrid_similarity(text1, text2, emb1, emb2, 0.3, 0.7)
        adaptive_sim = adaptive_hybrid_similarity(text1, text2, emb1, emb2)
        
        print(f"\n{description}:")
        print(f"  Cosine: {cosine_sim:.3f}")
        print(f"  Jaccard: {jaccard_sim:.3f}")
        print(f"  Hybrid (30/70): {hybrid_sim:.3f}")
        print(f"  Adaptive: {adaptive_sim:.3f}")

if __name__ == "__main__":
    test_hybrid_similarities()

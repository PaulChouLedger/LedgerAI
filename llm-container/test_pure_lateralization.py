#!/usr/bin/env python3
"""
Test pure left vs right similarity without full system initialization
"""

import requests
import numpy as np

def test_pure_lateralization():
    """Test pure left vs right similarity using RAG container API"""
    
    print('Testing pure left vs right similarity...')
    
    # Test pure words
    response = requests.post(
        "http://localhost:11435/embed",
        json={"texts": ["left", "right"]},
        timeout=5
    )
    
    if response.status_code != 200:
        print(f"❌ RAG API error: {response.status_code}")
        return
    
    embeddings = response.json()['embeddings']
    left_emb = np.array(embeddings[0])
    right_emb = np.array(embeddings[1])
    
    # Calculate cosine similarity
    similarity = np.dot(left_emb, right_emb) / (np.linalg.norm(left_emb) * np.linalg.norm(right_emb))
    print(f'Pure left vs right similarity: {similarity:.3f}')
    
    # Test with some context words
    test_pairs = [
        ('left', 'right'),
        ('left side', 'right side'), 
        ('left arm', 'right arm'),
        ('left leg', 'right leg'),
        ('left chest', 'right chest'),
        ('left upper', 'right upper'),
        ('left lower', 'right lower'),
        ('left quadrant', 'right quadrant'),
        ('left upper quadrant', 'right upper quadrant'),
        ('left lower quadrant', 'right lower quadrant')
    ]
    
    print('\nDetailed results:')
    for pair1, pair2 in test_pairs:
        response = requests.post(
            "http://localhost:11435/embed",
            json={"texts": [pair1, pair2]},
            timeout=5
        )
        
        if response.status_code == 200:
            embeddings = response.json()['embeddings']
            emb1 = np.array(embeddings[0])
            emb2 = np.array(embeddings[1])
            sim = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
            print(f'{pair1:20} vs {pair2:20}: {sim:.3f}')
        else:
            print(f'{pair1:20} vs {pair2:20}: ERROR')

if __name__ == "__main__":
    test_pure_lateralization()

#!/usr/bin/env python3
"""
Test script to check if generated embeddings are compatible with FAISS
"""

import os
import numpy as np
import faiss
from pathlib import Path

def test_embeddings():
    """Test the generated embeddings and index"""
    
    # Paths
    embeddings_dir = Path("data/embeddings")
    index_path = embeddings_dir / "index.faiss"
    chunks_path = embeddings_dir / "doc_chunks.npy"
    
    print("🔍 Testing embeddings compatibility...")
    
    # Check if files exist
    if not index_path.exists():
        print(f"❌ Index file not found: {index_path}")
        return False
        
    if not chunks_path.exists():
        print(f"❌ Chunks file not found: {chunks_path}")
        return False
    
    print(f"✅ Found index: {index_path}")
    print(f"✅ Found chunks: {chunks_path}")
    
    # Load index
    try:
        index = faiss.read_index(str(index_path))
        print(f"✅ Index loaded successfully")
        print(f"📊 Index type: {type(index)}")
        print(f"📊 Index dimension: {index.d}")
        print(f"📊 Index total vectors: {index.ntotal}")
        print(f"📊 Index metric type: {index.metric_type}")
        print(f"📊 Index is trained: {index.is_trained}")
    except Exception as e:
        print(f"❌ Failed to load index: {e}")
        return False
    
    # Load chunks
    try:
        chunks = np.load(chunks_path, allow_pickle=True)
        print(f"✅ Chunks loaded successfully")
        print(f"📦 Number of chunks: {len(chunks)}")
    except Exception as e:
        print(f"❌ Failed to load chunks: {e}")
        return False
    
    # Test search with random query
    try:
        print(f"🔍 Testing search with random query...")
        test_query = np.random.random((1, index.d)).astype(np.float32)
        print(f"📊 Test query shape: {test_query.shape}, dtype: {test_query.dtype}")
        print(f"📊 Test query OWNDATA: {test_query.flags.owndata}, base: {test_query.base is None}")
        
        distances, indices = index.search(test_query, min(3, index.ntotal))
        print(f"✅ Search successful!")
        print(f"📊 Results - distances: {distances}, indices: {indices}")
        
        # Test with actual sentence transformer embedding
        print(f"🔍 Testing with sentence transformer...")
        from sentence_transformers import SentenceTransformer
        encoder = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
        test_text = "heart disease cardiovascular"
        embedding = encoder.encode([test_text], convert_to_numpy=True)
        embedding = embedding.astype(np.float32)
        
        print(f"📊 ST embedding shape: {embedding.shape}, dtype: {embedding.dtype}")
        print(f"📊 ST embedding OWNDATA: {embedding.flags.owndata}, base: {embedding.base is None}")
        
        distances, indices = index.search(embedding, min(3, index.ntotal))
        print(f"✅ Sentence transformer search successful!")
        print(f"📊 Results - distances: {distances}, indices: {indices}")
        
        # Show actual chunks
        for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
            if idx < len(chunks):
                chunk = chunks[idx]
                print(f"📄 Result {i+1}: distance={dist:.4f}, chunk='{chunk[:100]}...'")
        
    except Exception as e:
        print(f"❌ Search failed: {e}")
        return False
    
    return True

if __name__ == "__main__":
    success = test_embeddings()
    if success:
        print("✅ All tests passed - embeddings are compatible with standard FAISS")
    else:
        print("❌ Tests failed - embeddings have compatibility issues")

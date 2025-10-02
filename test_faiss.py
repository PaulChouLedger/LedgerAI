#!/usr/bin/env python3
"""
Quick test of FAISS installation and embeddings
"""

def test_faiss():
    print("🧪 Testing FAISS installation...")
    
    try:
        import faiss
        print(f"✅ FAISS version: {faiss.__version__}")
        
        # Test embeddings
        import os
        if os.path.exists("data/embeddings/index.faiss"):
            index = faiss.read_index("data/embeddings/index.faiss")
            print(f"✅ Index loaded: {index.ntotal} vectors")
        else:
            print("❌ Index file not found")
            return False
            
        if os.path.exists("data/embeddings/doc_chunks.npy"):
            import numpy as np
            chunks = np.load("data/embeddings/doc_chunks.npy", allow_pickle=True)
            print(f"✅ Chunks loaded: {len(chunks)} documents")
        else:
            print("❌ Chunks file not found")
            return False
            
        print("🎉 FAISS setup is ready!")
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    test_faiss()

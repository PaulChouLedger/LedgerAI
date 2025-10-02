#!/usr/bin/env python3
"""
Debug RAG system to identify the numpy array issue
"""

def debug_rag():
    print("🔍 Debugging RAG system...")
    
    try:
        # Test imports
        print("1. Testing imports...")
        import numpy as np
        print(f"   ✅ NumPy version: {np.__version__}")
        
        import faiss
        print(f"   ✅ FAISS version: {faiss.__version__}")
        
        from sentence_transformers import SentenceTransformer
        print("   ✅ SentenceTransformers imported")
        
        # Test sentence transformer
        print("\n2. Testing SentenceTransformer...")
        encoder = SentenceTransformer('all-MiniLM-L6-v2')
        print("   ✅ Model loaded")
        
        # Test encoding
        print("\n3. Testing encoding...")
        query = "chest pain symptoms"
        
        # Try different encoding methods
        print("   Method 1: encode([query])")
        try:
            embedding1 = encoder.encode([query])
            print(f"   ✅ Shape: {embedding1.shape}, Type: {type(embedding1)}, Dtype: {embedding1.dtype}")
        except Exception as e:
            print(f"   ❌ Error: {e}")
        
        print("   Method 2: encode([query], convert_to_numpy=True)")
        try:
            embedding2 = encoder.encode([query], convert_to_numpy=True)
            print(f"   ✅ Shape: {embedding2.shape}, Type: {type(embedding2)}, Dtype: {embedding2.dtype}")
        except Exception as e:
            print(f"   ❌ Error: {e}")
        
        print("   Method 3: encode(query) single string")
        try:
            embedding3 = encoder.encode(query)
            print(f"   ✅ Shape: {embedding3.shape}, Type: {type(embedding3)}, Dtype: {embedding3.dtype}")
        except Exception as e:
            print(f"   ❌ Error: {e}")
        
        # Test FAISS
        print("\n4. Testing FAISS...")
        try:
            # Load index
            index = faiss.read_index("data/embeddings/index.faiss")
            print(f"   ✅ Index loaded: {index.ntotal} vectors, dimension: {index.d}")
            
            # Test search with working embedding
            if 'embedding1' in locals():
                test_embedding = np.array(embedding1, dtype=np.float32)
                if len(test_embedding.shape) == 1:
                    test_embedding = test_embedding.reshape(1, -1)
                
                print(f"   Testing search with shape: {test_embedding.shape}")
                distances, indices = index.search(test_embedding, 3)
                print(f"   ✅ Search successful: {len(indices[0])} results")
                
        except Exception as e:
            print(f"   ❌ FAISS error: {e}")
        
        print("\n🎉 Debug complete!")
        
    except Exception as e:
        print(f"❌ Debug failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_rag()
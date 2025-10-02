#!/usr/bin/env python3
"""
Simple RAG test that can run inside Docker container
Tests basic FAISS functionality without external dependencies
"""

import os
import sys
import time

def test_rag_basic():
    """Basic test of RAG components"""
    print("🧪 Testing RAG system (basic)...")
    
    # Check if embeddings exist
    index_path = "data/embeddings/index.faiss"
    chunks_path = "data/embeddings/doc_chunks.npy"
    
    print(f"📁 Checking files...")
    print(f"   Index: {os.path.exists(index_path)} ({index_path})")
    print(f"   Chunks: {os.path.exists(chunks_path)} ({chunks_path})")
    
    if not os.path.exists(index_path) or not os.path.exists(chunks_path):
        print("❌ Required files missing")
        return False
    
    # Test imports
    try:
        print("📦 Testing imports...")
        import faiss
        print(f"   FAISS: ✅ v{faiss.__version__}")
        
        import numpy as np
        print(f"   NumPy: ✅ v{np.__version__}")
        
        # Test FAISS index loading
        print("🔍 Testing FAISS index...")
        index = faiss.read_index(index_path)
        print(f"   Index size: {index.ntotal} vectors")
        
        # Test chunks loading
        print("📄 Testing chunks...")
        chunks = np.load(chunks_path, allow_pickle=True)
        print(f"   Chunks: {len(chunks)} documents")
        
        # Test basic search
        print("🔍 Testing search...")
        # Create a dummy query vector (same dimension as index)
        d = index.d  # dimension
        query_vector = np.random.random((1, d)).astype('float32')
        
        start_time = time.time()
        distances, indices = index.search(query_vector, 3)
        search_time = time.time() - start_time
        
        print(f"   Search time: {search_time:.3f}s")
        print(f"   Results: {len(indices[0])} chunks found")
        
        # Show sample results
        for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
            if idx < len(chunks):
                chunk_preview = str(chunks[idx])[:100] + "..." if len(str(chunks[idx])) > 100 else str(chunks[idx])
                print(f"   {i+1}. Distance: {dist:.3f}, Chunk: {chunk_preview}")
        
        print("✅ Basic RAG test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

def test_sentence_transformers():
    """Test sentence transformers availability"""
    print("🤖 Testing Sentence Transformers...")
    
    try:
        from sentence_transformers import SentenceTransformer
        
        # Use a lightweight model
        model_name = "all-MiniLM-L6-v2"
        print(f"   Loading model: {model_name}")
        
        start_time = time.time()
        encoder = SentenceTransformer(model_name)
        load_time = time.time() - start_time
        
        print(f"   Load time: {load_time:.2f}s")
        
        # Test encoding
        test_text = "chest pain symptoms"
        start_time = time.time()
        embedding = encoder.encode([test_text])[0]
        encode_time = time.time() - start_time
        
        print(f"   Encoding time: {encode_time:.3f}s")
        print(f"   Embedding shape: {embedding.shape}")
        
        print("✅ Sentence Transformers test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Sentence Transformers test failed: {e}")
        return False

def main():
    """Run all basic tests"""
    print("🚀 Basic RAG System Test")
    print("=" * 40)
    
    # Test 1: Basic FAISS functionality
    basic_ok = test_rag_basic()
    
    # Test 2: Sentence Transformers
    st_ok = test_sentence_transformers()
    
    # Summary
    print("\n" + "=" * 40)
    print("📋 Test Summary:")
    print(f"   Basic FAISS: {'✅ PASS' if basic_ok else '❌ FAIL'}")
    print(f"   Sentence Transformers: {'✅ PASS' if st_ok else '❌ FAIL'}")
    
    if basic_ok and st_ok:
        print("\n🎉 All basic tests passed! RAG components are working.")
        print("\n📝 Next steps:")
        print("   1. Start LLM container: docker-compose up llm-container")
        print("   2. Test RAG endpoints: curl http://localhost:11434/rag/stats")
        return True
    else:
        print("\n⚠️ Some tests failed. Check the output above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

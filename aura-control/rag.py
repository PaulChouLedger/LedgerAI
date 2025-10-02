"""
Aura RAG Module - FAISS-GPU based retrieval for medical triage
Optimized for Jetson Orin NX with GPU acceleration
"""

import numpy as np
import faiss
import os
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any
import time

class AuraRAG:
    def __init__(self, 
                 index_path: str = "data/embeddings/index.faiss",
                 chunks_path: str = "data/embeddings/doc_chunks.npy",
                 model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize Aura RAG system with FAISS-GPU
        
        Args:
            index_path: Path to FAISS index file
            chunks_path: Path to document chunks numpy file
            model_name: Sentence transformer model name
        """
        self.index_path = index_path
        self.chunks_path = chunks_path
        self.model_name = model_name
        
        # Initialize components
        self.index = None
        self.chunks = None
        self.encoder = None
        self.gpu_available = False
        
        # Load components
        self._load_components()
    
    def _load_components(self):
        """Load FAISS index, chunks, and encoder model"""
        print("[RAG] 🔧 Loading RAG components...")
        
        # Load FAISS index
        if os.path.exists(self.index_path):
            self.index = faiss.read_index(self.index_path)
            print(f"[RAG] ✅ Loaded FAISS index: {self.index_path}")
            
            # Optimize CPU FAISS for Jetson (use multiple threads)
            faiss.omp_set_num_threads(4)  # Use 4 threads on Jetson Orin NX
            print("[RAG] 🔧 Set FAISS to use 4 CPU threads for optimal Jetson performance")
        else:
            raise FileNotFoundError(f"FAISS index not found: {self.index_path}")
        
        # Load document chunks
        if os.path.exists(self.chunks_path):
            self.chunks = np.load(self.chunks_path, allow_pickle=True)
            print(f"[RAG] ✅ Loaded {len(self.chunks)} document chunks")
        else:
            raise FileNotFoundError(f"Document chunks not found: {self.chunks_path}")
        
        # Load sentence transformer
        self.encoder = SentenceTransformer(self.model_name)
        print(f"[RAG] ✅ Loaded encoder: {self.model_name}")
        
        # Check GPU availability (FAISS-GPU not available on ARM64/Jetson)
        try:
            self.gpu_available = faiss.get_num_gpus() > 0
            if self.gpu_available:
                print(f"[RAG] 🚀 GPU available: {faiss.get_num_gpus()} devices")
                # Move index to GPU
                self._move_to_gpu()
            else:
                print("[RAG] 💻 Using CPU FAISS (optimized for Jetson)")
        except Exception as e:
            print(f"[RAG] 💻 Using CPU FAISS (GPU not available on ARM64): {e}")
            self.gpu_available = False
    
    def _move_to_gpu(self):
        """Move FAISS index to GPU for faster search"""
        try:
            if self.gpu_available:
                # Create GPU resource
                res = faiss.StandardGpuResources()
                # Move index to GPU
                self.index = faiss.index_cpu_to_gpu(res, 0, self.index)
                print("[RAG] 🚀 Index moved to GPU")
        except Exception as e:
            print(f"[RAG] ⚠️ Failed to move to GPU: {e}, using CPU")
            self.gpu_available = False
    
    def retrieve(self, query: str, k: int = 3) -> List[Dict[str, Any]]:
        """
        Retrieve relevant document chunks for a query
        
        Args:
            query: User query string
            k: Number of chunks to retrieve
            
        Returns:
            List of relevant document chunks with metadata
        """
        start_time = time.time()
        
        # Encode query with error handling
        try:
            query_embedding = self.encoder.encode([query], convert_to_numpy=True)
            print(f"[RAG] 🔍 Query embedding shape: {query_embedding.shape}, type: {type(query_embedding)}")
            
            # Handle different output formats
            if isinstance(query_embedding, list):
                query_embedding = np.array(query_embedding, dtype=np.float32)
            elif not isinstance(query_embedding, np.ndarray):
                query_embedding = np.array(query_embedding, dtype=np.float32)
            
            # Ensure 2D array for FAISS
            if len(query_embedding.shape) == 1:
                query_embedding = query_embedding.reshape(1, -1)
            elif query_embedding.shape[0] > 1:
                query_embedding = query_embedding[0:1]  # Take first embedding
            
            # Ensure float32 dtype
            query_embedding = query_embedding.astype(np.float32)
            
            print(f"[RAG] 🔍 Final embedding shape: {query_embedding.shape}, dtype: {query_embedding.dtype}")
            
        except Exception as e:
            print(f"[RAG] ❌ Encoding error: {e}")
            raise Exception(f"Failed to encode query: {e}")
        
        # Search FAISS index
        distances, indices = self.index.search(query_embedding, k)
        
        # Format results
        results = []
        for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
            if idx < len(self.chunks):
                chunk = self.chunks[idx]
                results.append({
                    'chunk': chunk,
                    'score': float(1.0 / (1.0 + distance)),  # Convert distance to similarity
                    'rank': i + 1,
                    'distance': float(distance)
                })
        
        retrieval_time = time.time() - start_time
        print(f"[RAG] 🔍 Retrieved {len(results)} chunks in {retrieval_time:.3f}s")
        
        return results
    
    def augment_prompt(self, user_query: str, context_chunks: List[Dict[str, Any]]) -> str:
        """
        Augment user query with retrieved context
        
        Args:
            user_query: Original user query
            context_chunks: Retrieved document chunks
            
        Returns:
            Augmented prompt with context
        """
        if not context_chunks:
            return user_query
        
        # Build context from chunks
        context_parts = []
        for i, chunk_info in enumerate(context_chunks, 1):
            chunk = chunk_info['chunk']
            score = chunk_info['score']
            context_parts.append(f"{i}. {chunk} (relevance: {score:.2f})")
        
        context = "\n".join(context_parts)
        
        # Create augmented prompt
        augmented_prompt = f"""Based on the following medical information:

{context}

Please answer this question: {user_query}

Provide a helpful, accurate response based on the medical information above."""
        
        return augmented_prompt
    
    def search_and_augment(self, query: str, k: int = 3) -> str:
        """
        Complete RAG pipeline: search + augment
        
        Args:
            query: User query
            k: Number of chunks to retrieve
            
        Returns:
            Augmented prompt ready for LLM
        """
        # Retrieve relevant chunks
        chunks = self.retrieve(query, k)
        
        # Augment prompt with context
        augmented_prompt = self.augment_prompt(query, chunks)
        
        return augmented_prompt
    
    def get_stats(self) -> Dict[str, Any]:
        """Get RAG system statistics"""
        return {
            'index_size': self.index.ntotal if self.index else 0,
            'chunks_loaded': len(self.chunks) if self.chunks is not None else 0,
            'gpu_available': self.gpu_available,
            'model_name': self.model_name,
            'index_path': self.index_path
        }

# Global RAG instance
rag_instance = None

def get_rag() -> AuraRAG:
    """Get or create global RAG instance"""
    global rag_instance
    if rag_instance is None:
        rag_instance = AuraRAG()
    return rag_instance

def search_medical_info(query: str, k: int = 3) -> str:
    """
    Convenience function for medical information retrieval
    
    Args:
        query: Medical query
        k: Number of relevant chunks to retrieve
        
    Returns:
        Augmented prompt with medical context
    """
    rag = get_rag()
    return rag.search_and_augment(query, k)

# Test function
def test_rag():
    """Test RAG functionality"""
    print("[RAG] 🧪 Testing RAG system...")
    
    try:
        rag = AuraRAG()
        stats = rag.get_stats()
        print(f"[RAG] 📊 Stats: {stats}")
        
        # Test query
        test_query = "chest pain symptoms"
        results = rag.retrieve(test_query, k=2)
        print(f"[RAG] 🔍 Test query: '{test_query}'")
        print(f"[RAG] 📄 Retrieved {len(results)} chunks")
        
        for i, result in enumerate(results):
            print(f"[RAG] {i+1}. Score: {result['score']:.3f}, Distance: {result['distance']:.3f}")
            print(f"[RAG]    Chunk: {result['chunk'][:100]}...")
        
        return True
        
    except Exception as e:
        print(f"[RAG] ❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    test_rag()

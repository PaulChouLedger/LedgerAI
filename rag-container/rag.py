#!/usr/bin/env python3
"""
Aura RAG Module - FAISS-based retrieval for document search
Optimized for Jetson Orin NX with GPU acceleration via faiss_lite
"""

import os
import numpy as np
import faiss
from typing import List, Dict, Any
import time

# Configure for GPU acceleration (faiss_lite container)
os.environ['OMP_NUM_THREADS'] = '4'

from sentence_transformers import SentenceTransformer

# Use faiss_lite Python wrapper (the working C++ functions)
import sys
sys.path.append('/opt/faiss_lite')
from faiss_lite import cudaKNN, cudaL2Norm, cudaAllocMapped
print("[RAG] ✅ faiss_lite Python wrapper loaded - using CUDA functions")

class AuraRAG:
    def __init__(self, 
                 index_path: str = "data/embeddings/index.faiss",
                 vectors_path: str = "data/embeddings/vectors.npy",
                 chunks_path: str = "data/embeddings/doc_chunks.npy",
                 model_name: str = "all-MiniLM-L6-v2",
                 relevance_threshold: float = 0.3):
        """
        Initialize Aura RAG system with FAISS
        
        Args:
            index_path: Path to FAISS index file
            chunks_path: Path to document chunks numpy file
            model_name: Sentence transformer model name
        """
        self.index_path = index_path
        self.vectors_path = vectors_path
        self.chunks_path = chunks_path
        self.model_name = model_name
        self.relevance_threshold = relevance_threshold
        
        # Initialize components
        self.index = None
        self.chunks = None
        self.encoder = None
        
        # Load components
        self._load_components()
        
        # Convert FAISS index to numpy arrays for faiss_lite CUDA functions
        self._prepare_cuda_data()
    
    def _load_components(self):
        """Load FAISS index, chunks, and encoder model"""
        print("[RAG] 🔧 Loading RAG components...")
        
        # Load FAISS index
        try:
            if os.path.exists(self.index_path):
                print(f"[RAG] 🔧 Loading FAISS index from: {self.index_path}")
                print(f"[RAG] 🔍 FAISS version: {faiss.__version__}")
                self.index = faiss.read_index(self.index_path)
                print(f"[RAG] ✅ Loaded FAISS index: {self.index.ntotal} vectors")
                print(f"[RAG] 🔍 Index dimension: {self.index.d}")
                print(f"[RAG] 🔍 Index type: {type(self.index)}")
                print(f"[RAG] 🔍 Index is_trained: {self.index.is_trained}")
                print(f"[RAG] 🔍 Index metric_type: {self.index.metric_type}")
            else:
                raise FileNotFoundError(f"FAISS index not found: {self.index_path}")
        except Exception as e:
            print(f"[RAG] ❌ Failed to load FAISS index: {e}")
            raise
        
        # Load document chunks
        try:
            if os.path.exists(self.chunks_path):
                print(f"[RAG] 🔧 Loading document chunks from: {self.chunks_path}")
                self.chunks = np.load(self.chunks_path, allow_pickle=True)
                print(f"[RAG] ✅ Loaded {len(self.chunks)} document chunks")
            else:
                raise FileNotFoundError(f"Document chunks not found: {self.chunks_path}")
        except Exception as e:
            print(f"[RAG] ❌ Failed to load document chunks: {e}")
            raise
        
        # Load sentence transformer
        try:
            print(f"[RAG] 🔧 Loading sentence transformer: {self.model_name}")
            # Let sentence-transformers auto-detect the best device
            self.encoder = SentenceTransformer(
                self.model_name, 
                trust_remote_code=True
            )
            print(f"[RAG] ✅ Loaded sentence transformer: {self.model_name}")
            print(f"[RAG] 🔍 Sentence transformer device: {self.encoder.device}")
        except Exception as e:
            print(f"[RAG] ❌ Failed to load sentence transformer: {e}")
            # Try alternative loading method
            try:
                print(f"[RAG] 🔧 Trying alternative loading method...")
                self.encoder = SentenceTransformer(self.model_name)
                print(f"[RAG] ✅ Loaded sentence transformer (alternative method): {self.model_name}")
                print(f"[RAG] 🔍 Sentence transformer device: {self.encoder.device}")
            except Exception as e2:
                print(f"[RAG] ❌ Alternative loading also failed: {e2}")
                raise e
    
    def _prepare_cuda_data(self):
        """Prepare raw vectors for faiss_lite CUDA functions"""
        try:
            print("[RAG] 🔧 Loading raw vectors for faiss_lite...")
            
            # Load raw vectors from file (much simpler than reconstructing from FAISS index)
            if os.path.exists(self.vectors_path):
                vectors = np.load(self.vectors_path)
                print(f"[RAG] 🔍 Loaded vectors shape: {vectors.shape}, dtype: {vectors.dtype}")
                
                # Ensure float32
                if vectors.dtype != np.float32:
                    vectors = vectors.astype(np.float32)
                
                # Allocate CUDA memory for vectors
                self.cuda_vectors = cudaAllocMapped(vectors.shape, np.float32)
                self.cuda_vectors['array'][:] = vectors
                
                # Initialize vector norms (None for inner product metric)
                self.cuda_vector_norms = None
                
                # Pre-compute L2 norms if using L2 metric
                if self.index.metric_type == faiss.METRIC_L2:
                    n_vectors = vectors.shape[0]
                    self.cuda_vector_norms = cudaAllocMapped((n_vectors,), np.float32)
                    result = cudaL2Norm(
                        self.cuda_vectors['ptr'], 4,  # 4 bytes for float32
                        n_vectors, self.index.d,
                        self.cuda_vector_norms['ptr'], True, None
                    )
                    if not result:
                        print("[RAG] ⚠️ Failed to compute L2 norms")
                        self.cuda_vector_norms = None
                
                print(f"[RAG] ✅ CUDA data prepared: {vectors.shape[0]} vectors, {vectors.shape[1]} dimensions")
            else:
                print(f"[RAG] ❌ Raw vectors file not found: {self.vectors_path}")
                raise Exception(f"Raw vectors file not found: {self.vectors_path}")
                
        except Exception as e:
            print(f"[RAG] ❌ Failed to prepare CUDA data: {e}")
            raise
    
    def _search_with_faiss_lite(self, query_embedding, k):
        """Search using faiss_lite CUDA functions"""
        if self.cuda_vectors is None:
            print("[RAG] ❌ CUDA vectors not prepared")
            return None, None
        
        try:
            print(f"[RAG] 🔍 Query embedding: shape={query_embedding.shape}, dtype={query_embedding.dtype}")
            print(f"[RAG] 🔍 CUDA vectors: shape={self.cuda_vectors['array'].shape}, dtype={self.cuda_vectors['array'].dtype}")
            
            # Allocate CUDA memory for query
            cuda_query = cudaAllocMapped(query_embedding.shape, np.float32)
            cuda_query['array'][:] = query_embedding
            
            # Allocate CUDA memory for results
            cuda_distances = cudaAllocMapped((1, k), np.float32)
            cuda_indices = cudaAllocMapped((1, k), np.int64)
            
            # Call cudaKNN
            n_vectors = self.index.ntotal
            metric = 0 if self.index.metric_type == faiss.METRIC_INNER_PRODUCT else 1
            
            print(f"[RAG] 🔍 Calling cudaKNN: n={n_vectors}, d={self.index.d}, k={k}, metric={metric}")
            
            # Prepare vector norms pointer - use proper ctypes float pointer
            import ctypes
            vector_norms_ptr = ctypes.POINTER(ctypes.c_float)()  # NULL float pointer
            if self.cuda_vector_norms is not None:
                vector_norms_ptr = self.cuda_vector_norms['ptr']
            
            result = cudaKNN(
                self.cuda_vectors['ptr'],     # vectors
                cuda_query['ptr'],            # queries
                4,                            # dsize (float32 = 4 bytes)
                n_vectors,                    # n (number of vectors)
                1,                            # m (number of queries)
                self.index.d,                 # d (dimension)
                k,                            # k (number of results)
                metric,                       # metric type
                vector_norms_ptr,             # vector norms (NULL for inner product)
                cuda_distances['ptr'],        # output distances
                cuda_indices['ptr'],          # output indices
                ctypes.c_void_p(0)            # stream (NULL)
            )
            
            print(f"[RAG] 🔍 cudaKNN result: {result}")
            
            if result:
                distances = cuda_distances['array'].copy()
                indices = cuda_indices['array'].copy()
                print(f"[RAG] ✅ faiss_lite CUDA search successful: distances={distances}, indices={indices}")
                return distances, indices
            else:
                print("[RAG] ❌ faiss_lite CUDA search returned False")
                return None, None
                
        except Exception as e:
            print(f"[RAG] ❌ faiss_lite search error: {e}")
            import traceback
            traceback.print_exc()
            return None, None
    
    def search(self, query: str, k: int = 3) -> List[Dict[str, Any]]:
        """
        Search documents using RAG - simplified like container test
        
        Args:
            query: Search query
            k: Number of results to return
            
        Returns:
            List of relevant document chunks with metadata
        """
        if not query or not isinstance(query, str):
            return []
        
        try:
            # Simple encoding like the container test
            query_embedding = self.encoder.encode([query], convert_to_numpy=True)
            query_embedding = query_embedding.astype(np.float32)
            
            print(f"[RAG] 🔍 Query embedding shape: {query_embedding.shape}, dtype: {query_embedding.dtype}")
            
            # Use faiss_lite CUDA functions (the working approach)
            print(f"[RAG] 🔧 Using faiss_lite CUDA search...")
            distances, indices = self._search_with_faiss_lite(query_embedding, k)
            
            if distances is None:
                raise Exception("faiss_lite CUDA search failed")
            
            # Format results
            results = []
            for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
                idx = int(idx)
                if idx < len(self.chunks):
                    chunk = self.chunks[idx]
                    similarity_score = float(1.0 / (1.0 + distance))
                    
                    if similarity_score >= self.relevance_threshold:
                        results.append({
                            'chunk': chunk,
                            'score': similarity_score,
                            'distance': float(distance),
                            'rank': i + 1
                        })
            
            return results
            
        except Exception as e:
            print(f"[RAG] ❌ Search error: {e}")
            return []
    
    def get_stats(self) -> Dict[str, Any]:
        """Get RAG system statistics"""
        return {
            'index_size': self.index.ntotal if self.index else 0,
            'chunks_loaded': len(self.chunks) if self.chunks is not None else 0,
            'model_name': self.model_name,
            'index_path': self.index_path,
            'status': 'ready' if self.index is not None and self.chunks is not None and self.encoder is not None else 'not_ready'
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
    Search medical information and return augmented prompt
    
    Args:
        query: Medical query
        k: Number of relevant chunks to retrieve
        
    Returns:
        Augmented prompt with medical context
    """
    rag = get_rag()
    results = rag.search(query, k)
    
    if not results:
        return query
    
    # Build context from chunks
    context_parts = []
    for i, result in enumerate(results, 1):
        chunk = result['chunk']
        score = result['score']
        context_parts.append(f"{i}. {chunk} (relevance: {score:.2f})")
    
    context = "\n".join(context_parts)
    
    # Create augmented prompt
    augmented_prompt = f"""Based on the following information:

{context}

Please answer this question: {query}

Provide a helpful, accurate response based on the information above."""
    
    return augmented_prompt

def smart_search_medical_info(query: str, k: int = 3) -> tuple[bool, str]:
    """
    Smart search that decides whether to use RAG
    
    Args:
        query: User query
        k: Number of chunks to retrieve
        
    Returns:
        Tuple of (used_rag, prompt)
    """
    # Simple heuristic: use RAG for questions
    if '?' in query or any(word in query.lower() for word in ['what', 'who', 'where', 'when', 'why', 'how']):
        rag = get_rag()
        results = rag.search(query, k)
        
        if results:
            augmented_prompt = search_medical_info(query, k)
            return True, augmented_prompt
    
    return False, query
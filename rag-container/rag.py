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
import ctypes as C
import torch

# Configure for GPU acceleration (faiss_lite container)
os.environ['OMP_NUM_THREADS'] = '4'

from sentence_transformers import SentenceTransformer

# Try to import the faiss_lite CUDA functions
try:
    from cuda import cuda, nvrtc
    from cuda.cudart import (
        cudaMallocManaged, 
        cudaHostAlloc,
        cudaHostAllocMapped,
        cudaHostGetDevicePointer,
        cudaMemAttachGlobal, 
        cudaGetLastError,
        cudaGetErrorString,
        cudaError_t
    )
    
    # Load the faiss_lite library
    try:
        _lib = C.CDLL('/opt/faiss_lite/build/libfaiss_lite.so')
        print("[RAG] 🔍 faiss_lite library loaded from /opt/faiss_lite/build/libfaiss_lite.so")
    except Exception as lib_e:
        print(f"[RAG] ❌ Failed to load faiss_lite library: {lib_e}")
        raise
    
    def _cudaKNN(name='cudaKNN'):
        func = _lib[name]
        func.argtypes = [
            C.c_void_p, # vectors
            C.c_void_p, # queries
            C.c_int,    # dsize
            C.c_int,    # n
            C.c_int,    # m
            C.c_int,    # d
            C.c_int,    # k
            C.c_int,    # metric
            C.POINTER(C.c_float), # vector_norms
            C.POINTER(C.c_float), # out_distances
            C.POINTER(C.c_longlong), # out_indices
            C.c_void_p, # cudaStream_t
        ]
        func.restype = C.c_bool
        return func
    
    cudaKNN = _cudaKNN()
    FAISS_LITE_AVAILABLE = True
    print("[RAG] ✅ faiss_lite CUDA functions loaded successfully")
    
except ImportError as e:
    print(f"[RAG] ⚠️ faiss_lite CUDA functions not available: {e}")
    FAISS_LITE_AVAILABLE = False

class AuraRAG:
    def __init__(self, 
                 index_path: str = "data/embeddings/index.faiss",
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
        self.chunks_path = chunks_path
        self.model_name = model_name
        self.relevance_threshold = relevance_threshold
        
        # Initialize components
        self.index = None
        self.chunks = None
        self.encoder = None
        
        # Load components
        self._load_components()
    
    def _allocate_cuda_memory(self, data, dtype=np.float32):
        """Allocate CUDA memory for data and return CUDA pointer"""
        if not FAISS_LITE_AVAILABLE:
            return data
        
        try:
            # Get data size
            if isinstance(data, np.ndarray):
                size = data.nbytes
                shape = data.shape
            else:
                size = len(data) * np.dtype(dtype).itemsize
                shape = (len(data),)
            
            # Allocate CUDA managed memory
            err, ptr = cudaMallocManaged(size, cudaMemAttachGlobal)
            if err != cudaError_t.cudaSuccess:
                print(f"[RAG] ❌ CUDA allocation failed: {cudaGetErrorString(err)[1]}")
                return data
            
            # Create CUDA array that points to the managed memory
            cuda_array = np.ctypeslib.as_array(C.cast(ptr, C.POINTER(C.c_float)), shape=shape)
            cuda_array[:] = data.astype(dtype)
            
            print(f"[RAG] 🔍 Allocated CUDA memory: {size} bytes, shape: {shape}")
            print(f"[RAG] 🔍 CUDA array properties - OWNDATA: {cuda_array.flags.owndata}, base: {cuda_array.base is None}")
            return cuda_array
            
        except Exception as e:
            print(f"[RAG] ❌ CUDA memory allocation failed: {e}")
            return data
    
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
    
    def search(self, query: str, k: int = 3) -> List[Dict[str, Any]]:
        """
        Search documents using RAG
        
        Args:
            query: Search query
            k: Number of results to return
            
        Returns:
            List of relevant document chunks with metadata
        """
        if not query or not isinstance(query, str):
            return []
        
        try:
            # Encode query with explicit conversion to numpy
            query_embedding = self.encoder.encode(
                query, 
                convert_to_numpy=True,
                normalize_embeddings=False,
                show_progress_bar=False
            )
            
            # Force conversion to numpy array and ensure proper type
            print(f"[RAG] 🔍 Raw embedding type: {type(query_embedding)}")
            print(f"[RAG] 🔍 Raw embedding shape: {getattr(query_embedding, 'shape', 'no shape')}")
            
            if hasattr(query_embedding, 'cpu'):
                # Handle PyTorch tensors
                print(f"[RAG] 🔍 Converting PyTorch tensor to numpy")
                query_embedding = query_embedding.cpu().detach().numpy()
            elif hasattr(query_embedding, 'numpy'):
                # Handle tensorflow tensors
                print(f"[RAG] 🔍 Converting TensorFlow tensor to numpy")
                query_embedding = query_embedding.numpy()
            elif not isinstance(query_embedding, np.ndarray):
                print(f"[RAG] 🔍 Converting other type to numpy array")
                query_embedding = np.array(query_embedding)
            
            print(f"[RAG] 🔍 After conversion - type: {type(query_embedding)}, shape: {query_embedding.shape}")
            
            # CRITICAL: Create a completely independent numpy array for FAISS
            # FAISS requires arrays that own their data (OWNDATA=True) and no base
            
            # Force a completely independent array by using buffer copy
            if len(query_embedding.shape) == 1:
                # For 1D arrays, create new 2D array from buffer
                data = query_embedding.astype(np.float32).tobytes()
                query_embedding = np.frombuffer(data, dtype=np.float32).reshape(1, -1)
            else:
                # For 2D arrays, flatten, copy via buffer, reshape
                data = query_embedding.astype(np.float32).flatten().tobytes()
                query_embedding = np.frombuffer(data, dtype=np.float32).reshape(query_embedding.shape)
            
            # Final verification
            print(f"[RAG] 🔍 Final array properties - OWNDATA: {query_embedding.flags.owndata}, base: {query_embedding.base is None}")
            print(f"[RAG] 🔍 Final array shape: {query_embedding.shape}, dtype: {query_embedding.dtype}")
            
            # Validate embedding before FAISS search
            if not isinstance(query_embedding, np.ndarray):
                raise ValueError(f"Expected numpy array, got {type(query_embedding)}")
            
            if query_embedding.dtype != np.float32:
                query_embedding = query_embedding.astype(np.float32)
            
            # Debug embedding details
            print(f"[RAG] 🔍 Query embedding shape: {query_embedding.shape}, dtype: {query_embedding.dtype}")
            print(f"[RAG] 🔍 Query embedding type: {type(query_embedding)}")
            print(f"[RAG] 🔍 Is contiguous: {query_embedding.flags.c_contiguous}")
            print(f"[RAG] 🔍 About to search FAISS index with query shape: {query_embedding.shape}")
            
            # Search FAISS index with error handling
            try:
                # Ensure we have a proper numpy array for FAISS
                if not isinstance(query_embedding, np.ndarray):
                    print(f"[RAG] ❌ Expected numpy array, got {type(query_embedding)}")
                    raise ValueError(f"Expected numpy array, got {type(query_embedding)}")
                
                # Double-check the array properties
                if not query_embedding.flags.c_contiguous:
                    query_embedding = np.ascontiguousarray(query_embedding)
                    print(f"[RAG] 🔧 Made array contiguous")
                
                # Verify FAISS index is ready
                if not self.index.is_trained:
                    print(f"[RAG] ❌ FAISS index is not trained")
                    raise ValueError("FAISS index is not trained")
                
                print(f"[RAG] 🔍 Final embedding for FAISS: shape={query_embedding.shape}, dtype={query_embedding.dtype}, contiguous={query_embedding.flags.c_contiguous}")
                
                # Additional debugging for FAISS
                print(f"[RAG] 🔍 Embedding memory address: {query_embedding.__array_interface__ if hasattr(query_embedding, '__array_interface__') else 'No array interface'}")
                print(f"[RAG] 🔍 Embedding flags: {query_embedding.flags}")
                print(f"[RAG] 🔍 Embedding base: {query_embedding.base is None}")
                
                # Try different FAISS search methods
                try:
                    # Method 1: Try CUDA memory allocation if available
                    if FAISS_LITE_AVAILABLE:
                        print(f"[RAG] 🔧 Trying CUDA memory allocation...")
                        cuda_ptr = self._allocate_cuda_memory(query_embedding)
                        print(f"[RAG] 🔧 CUDA array type: {type(cuda_ptr)}, original type: {type(query_embedding)}")
                        if isinstance(cuda_ptr, np.ndarray) and cuda_ptr is not query_embedding:  # Successfully allocated CUDA memory
                            print(f"[RAG] 🔧 Using CUDA memory for FAISS search...")
                            distances, indices = self.index.search(cuda_ptr, k)
                            print(f"[RAG] ✅ CUDA memory FAISS search successful")
                        else:
                            raise Exception("CUDA memory allocation failed")
                    else:
                        raise Exception("FAISS lite not available")
                        
                except Exception as e1:
                    print(f"[RAG] ❌ CUDA search failed: {e1}")
                    try:
                        # Method 2: Standard search
                        print(f"[RAG] 🔧 Trying standard FAISS search...")
                        distances, indices = self.index.search(query_embedding, k)
                        print(f"[RAG] ✅ Standard FAISS search successful")
                    except Exception as e2:
                        print(f"[RAG] ❌ Standard search failed: {e2}")
                        try:
                            # Method 3: Try with different array format
                            print(f"[RAG] 🔧 Trying with explicit C-order array...")
                            query_c_array = np.asarray(query_embedding, dtype=np.float32, order='C')
                            distances, indices = self.index.search(query_c_array, k)
                            print(f"[RAG] ✅ C-order FAISS search successful")
                        except Exception as e3:
                            print(f"[RAG] ❌ C-order search failed: {e3}")
                            raise e1
                print(f"[RAG] ✅ FAISS search completed successfully")
            except Exception as e:
                print(f"[RAG] ❌ FAISS search failed: {e}")
                print(f"[RAG] 🔍 Query embedding details: shape={query_embedding.shape}, dtype={query_embedding.dtype}")
                print(f"[RAG] 🔍 Index details: type={type(self.index)}, trained={self.index.is_trained}")
                print(f"[RAG] 🔍 Index dimension: {self.index.d}")
                print(f"[RAG] 🔍 Query dimension: {query_embedding.shape[1] if len(query_embedding.shape) > 1 else query_embedding.shape[0]}")
                print(f"[RAG] 🔍 Index metric type: {self.index.metric_type}")
                print(f"[RAG] 🔍 Index total vectors: {self.index.ntotal}")
                
                # Try to understand what FAISS expects
                print(f"[RAG] 🔍 Testing FAISS with a simple array...")
                test_array = np.random.random((1, self.index.d)).astype(np.float32)
                print(f"[RAG] 🔍 Test array properties - OWNDATA: {test_array.flags.owndata}, base: {test_array.base is None}")
                try:
                    test_distances, test_indices = self.index.search(test_array, min(k, self.index.ntotal))
                    print(f"[RAG] ✅ Test search successful - FAISS index is working")
                except Exception as test_e:
                    print(f"[RAG] ❌ Test search failed: {test_e}")
                    print(f"[RAG] 🔍 This suggests the FAISS index itself has issues")
                    
                    # Try to create a new simple index to test
                    print(f"[RAG] 🔧 Creating test index to verify FAISS functionality...")
                    try:
                        test_index = faiss.IndexFlatIP(self.index.d)
                        test_vectors = np.random.random((10, self.index.d)).astype(np.float32)
                        test_index.add(test_vectors)
                        test_query = np.random.random((1, self.index.d)).astype(np.float32)
                        test_d, test_i = test_index.search(test_query, 3)
                        print(f"[RAG] ✅ Test index search successful - FAISS is working")
                        print(f"[RAG] 🔍 Test results: distances={test_d}, indices={test_i}")
                    except Exception as test_index_e:
                        print(f"[RAG] ❌ Test index creation failed: {test_index_e}")
                        print(f"[RAG] 🔍 This suggests a fundamental FAISS installation issue")
                
                # Try to reinitialize the index if it seems corrupted
                try:
                    print(f"[RAG] 🔧 Attempting to reload FAISS index...")
                    self.index = faiss.read_index(self.index_path)
                    distances, indices = self.index.search(query_embedding, k)
                    print(f"[RAG] ✅ FAISS search successful after reload")
                except Exception as e2:
                    print(f"[RAG] ❌ FAISS search failed even after reload: {e2}")
                    raise e
            
            # Debug FAISS results
            print(f"[RAG] 🔍 FAISS search results - distances shape: {distances.shape}, indices shape: {indices.shape}")
            print(f"[RAG] 🔍 Distances: {distances[0]}, Indices: {indices[0]}")
            
            # Format results
            results = []
            # Convert to numpy arrays to ensure proper handling
            distances_array = np.array(distances[0])
            indices_array = np.array(indices[0])
            
            for i, (distance, idx) in enumerate(zip(distances_array, indices_array)):
                # Convert to Python int to avoid array comparison issues
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

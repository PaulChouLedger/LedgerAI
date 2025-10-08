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
import re
from difflib import SequenceMatcher

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
        self.cuda_vectors = None
        self.cuda_vector_norms = None
        self.cuda_query_buffer = None  # Reusable query buffer to avoid allocation on every search
        
        # Load components
        self._load_components()
        
        # Validate that critical components loaded before preparing CUDA data
        if self.index is None or self.chunks is None or self.encoder is None:
            error_msg = f"Critical components failed to load: index={self.index is not None}, chunks={self.chunks is not None}, encoder={self.encoder is not None}"
            print(f"[RAG] ❌ {error_msg}")
            raise RuntimeError(error_msg)
        
        # Convert FAISS index to numpy arrays for faiss_lite CUDA functions
        try:
            self._prepare_cuda_data()
            
            # Validate that CUDA data was actually prepared
            if self.cuda_vectors is None:
                raise RuntimeError("CUDA vectors not initialized - _prepare_cuda_data() failed silently")
            
            # Force GPU prefetch for managed memory (fixes lazy-loading bug)
            print(f"[RAG] 🔧 Prefetching all vectors to GPU memory...")
            try:
                import faiss_lite
                # Prefetch ALL vectors to GPU (not lazy - forces immediate load)
                faiss_lite.cudaMemPrefetchAsync(
                    self.cuda_vectors_ptr,
                    self.cuda_vectors.nbytes,
                    device=0  # GPU 0
                )
                # Synchronize to ensure prefetch completes
                faiss_lite.cudaDeviceSynchronize()
                print(f"[RAG] ✅ All {self.index.ntotal} vectors prefetched to GPU")
            except Exception as e:
                print(f"[RAG] ⚠️ Prefetch failed (will use lazy loading): {e}")
                # Fallback: warmup search to trigger loading
                warmup_query = self.encoder.encode(["warmup"], convert_to_numpy=True).astype(np.float32)
                warmup_norms = np.linalg.norm(warmup_query, axis=1, keepdims=True)
                warmup_query = warmup_query / warmup_norms
                self._search_with_faiss_lite(warmup_query, k=10)
                print(f"[RAG] ✅ Warmup search completed")
            
            print(f"[RAG] ✅ Initialization complete: index ready, CUDA data ready, encoder ready")
            print(f"[RAG] 🎯 System status: {self.index.ntotal} vectors indexed, {len(self.chunks)} chunks loaded")
            
        except Exception as e:
            print(f"[RAG] ❌ CUDA data preparation failed during init: {e}")
            print(f"[RAG] ⚠️ RAG system CANNOT function without CUDA data")
            raise
    
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
        
        # Load sentence transformer - CUDA required
        import torch
        
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available - GPU required for RAG")
        
        print(f"[RAG] 🔧 Loading sentence transformer on CUDA: {self.model_name}")
        
        # Handle PyTorch meta tensor bug: load on CPU first, then move to CUDA
        # This avoids "Cannot copy out of meta tensor" error
        try:
            self.encoder = SentenceTransformer(self.model_name, device='cuda')
        except Exception as e:
            if "meta tensor" in str(e).lower():
                print(f"[RAG] 🔧 Meta tensor detected - loading via CPU→CUDA")
                self.encoder = SentenceTransformer(self.model_name, device='cpu')
                self.encoder = self.encoder.to('cuda')
            else:
                raise
        
        print(f"[RAG] ✅ Loaded on CUDA: {self.model_name}")
        print(f"[RAG] 🔍 Device: {self.encoder.device}")
    
    def _prepare_cuda_data(self):
        """Prepare raw vectors for faiss_lite CUDA functions"""
        try:
            print("[RAG] 🔧 Loading raw vectors for faiss_lite...")
            print(f"[RAG] 🔍 Looking for vectors at: {self.vectors_path}")
            print(f"[RAG] 🔍 Vectors file exists: {os.path.exists(self.vectors_path)}")
            
            # Load raw vectors from file (much simpler than reconstructing from FAISS index)
            if not os.path.exists(self.vectors_path):
                error_msg = f"Raw vectors file not found: {self.vectors_path}"
                print(f"[RAG] ❌ {error_msg}")
                print(f"[RAG] 💡 Run 'python3 scripts/rebuild_embeddings.py' to generate vectors.npy")
                raise FileNotFoundError(error_msg)
            
            vectors = np.load(self.vectors_path)
            print(f"[RAG] 🔍 Loaded vectors shape: {vectors.shape}, dtype: {vectors.dtype}")
            
            # Validate vectors
            if len(vectors.shape) != 2:
                raise ValueError(f"Vectors must be 2D array, got shape {vectors.shape}")
            
            if vectors.shape[1] != self.index.d:
                raise ValueError(f"Vector dimension {vectors.shape[1]} doesn't match index dimension {self.index.d}")
            
            # Ensure float32
            if vectors.dtype != np.float32:
                print(f"[RAG] 🔄 Converting vectors from {vectors.dtype} to float32")
                vectors = vectors.astype(np.float32)
            
            # Allocate CUDA memory for vectors
            print(f"[RAG] 🔧 Allocating CUDA memory for {vectors.shape[0]} vectors...")
            self.cuda_vectors = cudaAllocMapped(vectors.shape, np.float32)
            self.cuda_vectors['array'][:] = vectors
            print(f"[RAG] ✅ CUDA vectors allocated: ptr={self.cuda_vectors['ptr']}")
            
            # Initialize vector norms (None for inner product metric)
            self.cuda_vector_norms = None
            
            # Pre-compute L2 norms if using L2 metric
            if self.index.metric_type == faiss.METRIC_L2:
                n_vectors = vectors.shape[0]
                print(f"[RAG] 🔧 Pre-computing L2 norms for {n_vectors} vectors...")
                self.cuda_vector_norms = cudaAllocMapped((n_vectors,), np.float32)
                result = cudaL2Norm(
                    self.cuda_vectors['ptr'], 4,  # 4 bytes for float32
                    n_vectors, self.index.d,
                    self.cuda_vector_norms['ptr'], True, None
                )
                if not result:
                    print("[RAG] ⚠️ Failed to compute L2 norms, will use NULL pointer")
                    self.cuda_vector_norms = None
                else:
                    print(f"[RAG] ✅ L2 norms computed")
            
            # Pre-allocate query buffer for reuse across searches (avoids repeated allocations)
            print(f"[RAG] 🔧 Pre-allocating query buffer...")
            self.cuda_query_buffer = cudaAllocMapped((1, self.index.d), np.float32)
            print(f"[RAG] ✅ Query buffer allocated: {self.cuda_query_buffer['array'].shape}")
            
            # Pre-allocate result buffers (max k=10 for most queries)
            self.max_k = 10
            self.cuda_distances_buffer = cudaAllocMapped((1, self.max_k), np.float32)
            self.cuda_indices_buffer = cudaAllocMapped((1, self.max_k), np.int64)
            print(f"[RAG] ✅ Result buffers allocated: distances={self.cuda_distances_buffer['array'].shape}, indices={self.cuda_indices_buffer['array'].shape}")
            
            print(f"[RAG] ✅ CUDA data prepared: {vectors.shape[0]} vectors, {vectors.shape[1]} dimensions")
            print(f"[RAG] 🔍 Metric type: {self.index.metric_type} ({'Inner Product' if self.index.metric_type == 0 else 'L2'})")
            
        except FileNotFoundError as e:
            print(f"[RAG] ❌ File not found: {e}")
            print(f"[RAG] 💡 Make sure to run rebuild_embeddings.py first!")
            raise
        except Exception as e:
            print(f"[RAG] ❌ Failed to prepare CUDA data: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def _search_with_faiss_lite(self, query_embedding, k):
        """Search using faiss_lite CUDA functions"""
        if self.cuda_vectors is None:
            print("[RAG] ❌ CUDA vectors not prepared")
            return None, None
        
        # Note: Do NOT call torch.cuda.empty_cache() here!
        # It will invalidate the cudaAllocMapped memory used by faiss_lite
        
        try:
            print(f"[RAG] 🔍 Query embedding: shape={query_embedding.shape}, dtype={query_embedding.dtype}")
            print(f"[RAG] 🔍 Query embedding preview: {query_embedding[0][:5]}")  # Check if non-zero
            print(f"[RAG] 🔍 CUDA vectors: shape={self.cuda_vectors['array'].shape}, dtype={self.cuda_vectors['array'].dtype}")
            
            # Reuse pre-allocated query buffer (avoid repeated allocations that corrupt memory)
            self.cuda_query_buffer['array'][:] = query_embedding
            
            # Verify query was copied correctly
            print(f"[RAG] 🔍 CUDA query preview: {self.cuda_query_buffer['array'][0][:5]}")
            
            # Use pre-allocated result buffers or allocate new ones if k > max_k
            if k <= self.max_k:
                # Reuse pre-allocated buffers
                cuda_distances = self.cuda_distances_buffer
                cuda_indices = self.cuda_indices_buffer
                print(f"[RAG] 🔧 Reusing pre-allocated result buffers (k={k} <= max_k={self.max_k})")
            else:
                # Allocate larger buffers if needed
                print(f"[RAG] 🔧 Allocating larger buffers (k={k} > max_k={self.max_k})")
                cuda_distances = cudaAllocMapped((1, k), np.float32)
                cuda_indices = cudaAllocMapped((1, k), np.int64)
            
            # Call cudaKNN
            n_vectors = self.index.ntotal
            metric = 0 if self.index.metric_type == faiss.METRIC_INNER_PRODUCT else 1
            
            print(f"[RAG] 🔍 Calling cudaKNN: n={n_vectors}, d={self.index.d}, k={k}, metric={metric}")
            
            # Prepare vector norms pointer - for inner product metric, we can pass NULL
            import ctypes
            if self.cuda_vector_norms is not None:
                vector_norms_ptr = self.cuda_vector_norms['ptr']
            else:
                # For inner product metric, pass NULL pointer
                vector_norms_ptr = ctypes.cast(0, ctypes.POINTER(ctypes.c_float))
            
            # Cast pointers to the correct types
            distances_ptr = ctypes.cast(cuda_distances['ptr'], ctypes.POINTER(ctypes.c_float))
            indices_ptr = ctypes.cast(cuda_indices['ptr'], ctypes.POINTER(ctypes.c_longlong))
            
            result = cudaKNN(
                self.cuda_vectors['ptr'],           # vectors
                self.cuda_query_buffer['ptr'],      # queries (reuse buffer)
                4,                                  # dsize (float32 = 4 bytes)
                n_vectors,                          # n (number of vectors)
                1,                                  # m (number of queries)
                self.index.d,                       # d (dimension)
                k,                                  # k (number of results)
                metric,                             # metric type
                vector_norms_ptr,                   # vector norms (NULL for inner product)
                distances_ptr,                      # output distances
                indices_ptr,                        # output indices
                ctypes.c_void_p(0)                  # stream (NULL)
            )
            
            print(f"[RAG] 🔍 cudaKNN result: {result}")
            
            if result:
                # Copy only the requested k results (not the entire buffer)
                distances = cuda_distances['array'][:, :k].copy()
                indices = cuda_indices['array'][:, :k].copy()
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
    
    def _extract_person_name(self, query: str) -> str:
        """
        Extract person name from queries like "Who is X?", "Tell me about Y", etc.
        
        Args:
            query: User query
            
        Returns:
            Extracted name or empty string if no name found
        """
        # Common patterns for name queries (case-insensitive for entire pattern)
        patterns = [
            r"who (?:is|was) ([a-z]+(?:\s+[a-z]+)*)",              # "Who is david lara"
            r"who'?s ([a-z]+(?:\s+[a-z]+)*)",                      # "Who's bob carella"
            r"tell me about ([a-z]+(?:\s+[a-z]+)*)",               # "Tell me about paul"
            r"about ([a-z]+(?:\s+[a-z]+)*)",                       # "About jorge"
            r"describe ([a-z]+(?:\s+[a-z]+)*)",                    # "Describe liam"
            r"what (?:do you know )?about ([a-z]+(?:\s+[a-z]+)*)", # "What about X"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                # Capitalize name for matching (Rafael Cabello, not rafael cabello)
                name = match.group(1).strip()
                name = ' '.join(word.capitalize() for word in name.split())
                print(f"[RAG] 🔍 Detected name query: '{name}'")
                return name
        
        return ""
    
    def search(self, query: str, k: int = 3) -> List[Dict[str, Any]]:
        """
        Search documents using RAG with hybrid keyword filtering
        
        Args:
            query: Search query
            k: Number of results to return
            
        Returns:
            List of relevant document chunks with metadata
        """
        if not query or not isinstance(query, str):
            return []
        
        # Extract person name if this is a biographical query
        person_name = self._extract_person_name(query)
        use_keyword_filter = bool(person_name)
        
        if use_keyword_filter:
            print(f"[RAG] 🔍 Hybrid search enabled: filtering for '{person_name}'")
        
        try:
            # Encode query - ensure it's on CPU for numpy conversion
            import torch
            with torch.no_grad():
                query_embedding = self.encoder.encode([query], convert_to_numpy=True, show_progress_bar=False)
                query_embedding = query_embedding.astype(np.float32)
            
            # Normalize for Inner Product metric (required for cosine similarity)
            # Do NOT use faiss.normalize_L2() - it causes "input not a numpy array" errors
            # Use manual normalization instead
            if self.index.metric_type == faiss.METRIC_INNER_PRODUCT:
                # Manual L2 normalization
                norms = np.linalg.norm(query_embedding, axis=1, keepdims=True)
                query_embedding = query_embedding / norms
                print(f"[RAG] 🔧 Query manually normalized for Inner Product metric")
            
            print(f"[RAG] 🔍 Query embedding shape: {query_embedding.shape}, dtype: {query_embedding.dtype}")
            print(f"[RAG] 🔍 Query embedding preview: {query_embedding[0][:5]}")  # First 5 values
            
            # Use faiss_lite CUDA functions (the working approach)
            print(f"[RAG] 🔧 Using faiss_lite CUDA search...")
            
            # Request more results if using keyword filter (we'll filter down)
            search_k = k * 3 if use_keyword_filter else k
            distances, indices = self._search_with_faiss_lite(query_embedding, search_k)
            
            if distances is None:
                raise Exception("faiss_lite CUDA search failed")
            
            # Format results with detailed debugging and deduplication
            results = []
            seen_indices = set()  # Track seen indices to avoid duplicates
            print(f"[RAG] 🔍 Processing {len(distances[0])} search results...")
            
            for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
                idx = int(idx)
                
                # Skip duplicate indices
                if idx in seen_indices:
                    print(f"[RAG] 🔍 Result {i+1}: idx={idx} (DUPLICATE - skipping)")
                    continue
                    
                if idx < len(self.chunks):
                    chunk = self.chunks[idx]
                    similarity_score = float(1.0 / (1.0 + distance))
                    
                    print(f"[RAG] 🔍 Result {i+1}: idx={idx}, distance={distance:.4f}, score={similarity_score:.4f}, threshold={self.relevance_threshold}")
                    print(f"[RAG] 🔍 Chunk preview: '{chunk[:100]}...'")
                    
                    # Apply keyword filter if enabled (HYBRID SEARCH with fuzzy matching)
                    if use_keyword_filter:
                        # Use fuzzy matching to handle typos/variations (0.65 catches Raphael↔Rafael)
                        if not self._fuzzy_name_search(person_name, chunk, threshold=0.65):
                            print(f"[RAG] 🔍 Filtered out: no fuzzy match for '{person_name}' in chunk")
                            continue
                        else:
                            print(f"[RAG] ✅ Fuzzy name match found for '{person_name}'")
                    
                    if similarity_score >= self.relevance_threshold:
                        results.append({
                            'chunk': chunk,
                            'score': similarity_score,
                            'distance': float(distance),
                            'rank': len(results) + 1
                        })
                        seen_indices.add(idx)
                        print(f"[RAG] ✅ Added to results (above threshold)")
                        
                        # Stop if we have enough results
                        if len(results) >= k:
                            break
                    else:
                        print(f"[RAG] ❌ Below threshold, skipping")
                else:
                    print(f"[RAG] ❌ Invalid index {idx} (max: {len(self.chunks)-1})")
            
            print(f"[RAG] 🔍 Final results: {len(results)} unique documents above threshold")
            
            # If we have fewer results than requested, try to get more diverse results
            if len(results) < k and len(results) > 0 and not use_keyword_filter:
                print(f"[RAG] 🔧 Only found {len(results)} unique results, requesting more for diversity...")
                # Try to get additional results by increasing k and filtering out seen indices
                additional_k = k * 2  # Request more results
                additional_distances, additional_indices = self._search_with_faiss_lite(query_embedding, additional_k)
                
                if additional_distances is not None:
                    for i, (distance, idx) in enumerate(zip(additional_distances[0], additional_indices[0])):
                        idx = int(idx)
                        if idx not in seen_indices and idx < len(self.chunks):
                            chunk = self.chunks[idx]
                            similarity_score = float(1.0 / (1.0 + distance))
                            
                            if similarity_score >= self.relevance_threshold:
                                results.append({
                                    'chunk': chunk,
                                    'score': similarity_score,
                                    'distance': float(distance),
                                    'rank': len(results) + 1
                                })
                                seen_indices.add(idx)
                                print(f"[RAG] 🔧 Added additional result: idx={idx}, score={similarity_score:.4f}")
                                
                                if len(results) >= k:
                                    break
            
            return results
            
        except Exception as e:
            print(f"[RAG] ❌ Search error: {e}")
            return []
    
    def _fuzzy_name_search(self, person_name: str, chunk: str, threshold: float = 0.65) -> bool:
        """
        Check if a chunk contains a fuzzy match for the person name
        Handles typos like "Bob Corella" matching "Bob Carella"
        
        Args:
            person_name: Person name from query (e.g., "Bob Corella")
            chunk: Document chunk
            threshold: Similarity threshold (0.0 to 1.0)
            
        Returns:
            True if fuzzy match found, False otherwise
        """
        # Method 1: Check full name similarity with all name-like sequences in chunk
        chunk_names = re.findall(r'[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+', chunk)
        for chunk_name in chunk_names:
            similarity = SequenceMatcher(None, person_name.lower(), chunk_name.lower()).ratio()
            if similarity >= threshold:
                print(f"[RAG] 🔍 Fuzzy full name match: '{person_name}' ~ '{chunk_name}' (similarity: {similarity:.3f})")
                return True
        
        # Method 2: Check individual word matches (for partial names)
        query_words = person_name.split()
        chunk_words = re.findall(r'\b[A-Z][a-z]+\b', chunk)
        
        matches = 0
        for query_word in query_words:
            for chunk_word in chunk_words:
                similarity = SequenceMatcher(None, query_word.lower(), chunk_word.lower()).ratio()
                if similarity >= threshold:
                    print(f"[RAG] 🔍 Fuzzy word match: '{query_word}' ~ '{chunk_word}' (similarity: {similarity:.3f})")
                    matches += 1
                    break
        
        # Require at least 2 words to match (first + last name)
        if matches >= min(2, len(query_words)):
            return True
        
        return False

    def diagnose_index_issues(self) -> Dict[str, Any]:
        """Diagnose potential issues with the FAISS index"""
        issues = []
        
        if self.index is None:
            issues.append("FAISS index not loaded")
            return {'issues': issues, 'status': 'critical'}
        
        # Check for duplicate vectors in the index
        try:
            # Get some sample vectors from the index
            if hasattr(self.index, 'reconstruct'):
                sample_indices = [0, 1, 2] if self.index.ntotal >= 3 else list(range(self.index.ntotal))
                vectors = []
                for idx in sample_indices:
                    vector = self.index.reconstruct(idx)
                    vectors.append(vector)
                
                # Check for identical vectors
                identical_count = 0
                for i in range(len(vectors)):
                    for j in range(i + 1, len(vectors)):
                        if np.array_equal(vectors[i], vectors[j]):
                            identical_count += 1
                
                if identical_count > 0:
                    issues.append(f"Found {identical_count} identical vectors in index")
                
        except Exception as e:
            issues.append(f"Error checking index vectors: {e}")
        
        # Check chunks for duplicates
        if self.chunks is not None:
            chunk_set = set(self.chunks)
            if len(chunk_set) != len(self.chunks):
                issues.append(f"Found {len(self.chunks) - len(chunk_set)} duplicate chunks")
        
        # Check index statistics
        if self.index.ntotal != len(self.chunks):
            issues.append(f"Index size ({self.index.ntotal}) doesn't match chunks ({len(self.chunks)})")
        
        return {
            'issues': issues,
            'status': 'healthy' if not issues else 'issues_found',
            'index_size': self.index.ntotal,
            'chunks_count': len(self.chunks) if self.chunks is not None else 0
        }

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
    Search medical information and return augmented prompt with fuzzy name matching
    
    Args:
        query: Medical query
        k: Number of relevant chunks to retrieve
        
    Returns:
        Augmented prompt with medical context
    """
    rag = get_rag()
    results = rag.search(query, k)
    
    print(f"[RAG] 🔍 search_medical_info called with query: '{query}'")
    print(f"[RAG] 🔍 Found {len(results)} results")
    
    if not results:
        print(f"[RAG] ⚠️ No results found, returning original query")
        return query
    
    # Check for fuzzy name matches in results
    fuzzy_matches_found = False
    for result in results:
        if rag._fuzzy_name_search(query, result['chunk']):
            fuzzy_matches_found = True
            break
    
    # Build context from chunks
    context_parts = []
    for i, result in enumerate(results, 1):
        chunk = result['chunk']
        score = result['score']
        context_parts.append(f"{i}. {chunk} (relevance: {score:.2f})")
        print(f"[RAG] 🔍 Adding chunk {i}: score={score:.3f}, preview='{chunk[:50]}...'")
    
    context = "\n".join(context_parts)
    
    # Create enhanced augmented prompt with name variation handling
    if fuzzy_matches_found:
        augmented_prompt = f"""Based on the following information:

{context}

Please answer this question: {query}

IMPORTANT: The query may contain name variations or slight misspellings. Look for similar names in the context above (e.g., "Corrella" might refer to "Carella", "Corella", etc.). Provide a helpful, accurate response based on the information above, making reasonable connections between similar names."""
    else:
        augmented_prompt = f"""Based on the following information:

{context}

Please answer this question: {query}

Provide a helpful, accurate response based on the information above."""
    
    print(f"[RAG] 🔍 Augmented prompt length: {len(augmented_prompt)} characters")
    print(f"[RAG] 🔍 Augmented prompt preview: '{augmented_prompt[:200]}...'")
    print(f"[RAG] 🔍 Fuzzy name matching: {'enabled' if fuzzy_matches_found else 'not needed'}")
    
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
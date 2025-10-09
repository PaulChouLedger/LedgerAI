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
            
            # Force all vectors into GPU memory with warmup search
            # Note: cudaMemPrefetchAsync not available in faiss_lite, using warmup search instead
            print(f"[RAG] 🔧 Warming up GPU with test search ({self.index.ntotal} vectors)...")
            warmup_query = self.encoder.encode(["warmup"], convert_to_numpy=True).astype(np.float32)
            warmup_norms = np.linalg.norm(warmup_query, axis=1, keepdims=True)
            warmup_query = warmup_query / warmup_norms
            self._search_with_faiss_lite(warmup_query, k=min(20, self.index.ntotal))
            
            # Synchronize CUDA to ensure warmup is complete before accepting queries
            import torch
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                print(f"[RAG] ✅ GPU warmup completed and synchronized - vectors ready for queries")
            else:
                print(f"[RAG] ✅ GPU warmup completed - vectors ready for queries")
            
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
        Extract person name from queries like "Who is X?", "Tell me about Y", 
        "Where does X work?", etc.
        
        Args:
            query: User query
            
        Returns:
            Extracted name or empty string if no name found
        """
        # Common patterns for name queries (case-insensitive)
        # Use [a-z] since query is already lowercased
        patterns = [
            # Multi-word names (e.g., "bob carella", "david lara")  
            r"who (?:is|was) ([a-z]+(?: [a-z]+)+)",                           # "who is david lara"
            r"who'?s ([a-z]+(?: [a-z]+)+)",                                    # "who's bob carella"
            r"tell me (?:about|everything .*? about) ([a-z]+(?: [a-z]+)+)",   # "tell me about/everything you know about paul chou"
            r"(?:all )?(?:the )?information .*? about ([a-z]+(?: [a-z]+)+)",  # "all the information you know about X"
            r"everything .*? about ([a-z]+(?: [a-z]+)+)",                      # "everything about X"
            r"about ([a-z]+(?: [a-z]+)+)",                                     # "about jorge guinovart"
            r"describe ([a-z]+(?: [a-z]+)+)",                                  # "describe liam hugill"
            r"what (?:do you know )?about ([a-z]+(?: [a-z]+)+)",              # "what about X" or "what do you know about X"
            # Location/work/contact queries about multi-word names
            r"where (?:does|did|is) ([a-z]+(?: [a-z]+)+) (?:work|live|study)",  # "where does david lara work"
            r"what (?:is|was) ([a-z]+(?: [a-z]+)+)'?s? (?:job|role|position|title|company|workplace)",  # "what is paul chou's job"
            r"(?:where|what) (?:can|do) (?:i |you )?(?:find|contact|reach) ([a-z]+(?: [a-z]+)+)",  # "where can i find bob carella"
            # Single names (e.g., "raphael", "peter")
            r"who (?:is|was) ([a-z]+)\??$",                                    # "who is raphael?"
            r"who'?s ([a-z]+)\??$",                                            # "who's raphael?"
            r"tell me (?:about|everything .*? about) ([a-z]+)\??$",           # "tell me about raphael?"
            r"(?:all )?(?:the )?information .*? about ([a-z]+)\??$",          # "information about X"
            r"everything .*? about ([a-z]+)\??$",                              # "everything about X"
            r"about ([a-z]+)\??$",                                             # "about raphael?"
            r"describe ([a-z]+)\??$",                                          # "describe raphael?"
            # Location/work/contact queries about single names
            r"where (?:does|did|is) ([a-z]+) (?:work|live|study)",            # "where does raphael work"
            r"what (?:is|was) ([a-z]+)'?s? (?:job|role|position|title|company|workplace)",  # "what is paul's job"
            r"(?:where|what) (?:can|do) (?:i |you )?(?:find|contact|reach) ([a-z]+)",  # "where can i contact bob"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                name = match.group(1)
                # Capitalize each word for matching against document text
                name_capitalized = ' '.join(word.capitalize() for word in name.split())
                print(f"[RAG] 🔍 Detected name query: '{name_capitalized}' (from '{name}')")
                return name_capitalized
        
        return ""
    
    def _extract_medical_term(self, query: str) -> str:
        """
        Extract medical terms from queries like "What is myocardial infarction?", 
        "Tell me about diabetes", "Explain COPD", etc.
        
        Args:
            query: User query
            
        Returns:
            Extracted medical term or empty string if no term found
        """
        # Patterns for medical/technical queries
        patterns = [
            r"what is ([a-z][a-z\s]+?)(?:\?|$)",                          # "What is myocardial infarction?"
            r"what's ([a-z][a-z\s]+?)(?:\?|$)",                           # "What's diabetes mellitus?"
            r"define ([a-z][a-z\s]+?)(?:\?|$)",                           # "Define hypertension"
            r"tell me about ([a-z][a-z\s]+?)(?:\?|$)",                    # "Tell me about COPD"
            r"explain ([a-z][a-z\s]+?)(?:\?|$)",                          # "Explain pneumonia"
            r"describe ([a-z][a-z\s]+?)(?:\?|$)",                         # "Describe asthma"
            r"information (?:on|about) ([a-z][a-z\s]+?)(?:\?|$)",        # "Information on heart disease"
            r"symptoms of ([a-z][a-z\s]+?)(?:\?|$)",                      # "Symptoms of COVID-19"
            r"treatment for ([a-z][a-z\s]+?)(?:\?|$)",                    # "Treatment for depression"
            r"diagnosis of ([a-z][a-z\s]+?)(?:\?|$)",                     # "Diagnosis of cancer"
        ]
        
        query_lower = query.lower()
        
        for pattern in patterns:
            match = re.search(pattern, query_lower)
            if match:
                term = match.group(1).strip()
                # Filter out very short or common words
                if len(term) > 3 and term not in ['that', 'this', 'there', 'their', 'those']:
                    print(f"[RAG] 🔍 Detected medical term query: '{term}'")
                    return term
        
        return ""
    
    def _extract_key_terms(self, query: str) -> List[str]:
        """
        Extract key terms from ANY query for keyword filtering
        
        Extracts:
        - Capitalized names (e.g., "Bob Carella", "Rafael")
        - Multi-word technical terms (e.g., "myocardial infarction", "neural pathways")
        - Significant single words (nouns, medical terms, 4+ chars)
        
        Filters out:
        - Common stop words (the, is, how, what, etc.)
        - Very short words (< 4 chars)
        - Question words
        
        Args:
            query: User query
            
        Returns:
            List of key terms to use for filtering
        """
        # Stop words to exclude (common question words and articles)
        stop_words = {
            'what', 'how', 'why', 'when', 'where', 'who', 'does', 'is', 'are', 'was', 'were',
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with',
            'from', 'about', 'like', 'this', 'that', 'these', 'those', 'can', 'could', 'would',
            'should', 'will', 'do', 'did', 'have', 'has', 'had', 'be', 'been', 'being', 'me',
            'you', 'your', 'my', 'it', 'its', 'tell', 'explain', 'describe', 'define'
        }
        
        key_terms = []
        
        # 1. Check for capitalized names (highest priority)
        person_name = self._extract_person_name(query)
        if person_name:
            key_terms.append(person_name)
            return key_terms  # Names are specific enough, use only the name
        
        # 2. Extract multi-word phrases (technical terms like "heart disease", "brain function")
        # Look for 2-3 word phrases of significant words
        words = re.findall(r'\b[a-zA-Z]+\b', query.lower())
        
        for i in range(len(words) - 1):
            # Check if this could be a multi-word term
            word1, word2 = words[i], words[i + 1]
            
            if (word1 not in stop_words and word2 not in stop_words and 
                len(word1) >= 4 and len(word2) >= 4):
                phrase = f"{word1} {word2}"
                key_terms.append(phrase)
                
                # Also try 3-word phrases
                if i + 2 < len(words):
                    word3 = words[i + 2]
                    if word3 not in stop_words and len(word3) >= 4:
                        phrase3 = f"{word1} {word2} {word3}"
                        key_terms.append(phrase3)
        
        # 3. Extract significant single words (medical terms, nouns, etc.)
        for word in words:
            if (word not in stop_words and 
                len(word) >= 4 and 
                word not in key_terms):  # Don't duplicate if already in a phrase
                key_terms.append(word)
        
        # Remove duplicates while preserving order, prioritize longer terms
        seen = set()
        unique_terms = []
        for term in sorted(key_terms, key=len, reverse=True):
            if term not in seen:
                seen.add(term)
                unique_terms.append(term)
        
        # Return top 5 most significant terms (longer = more specific)
        return unique_terms[:5]
    
    def search(self, query: str, k: int = 3) -> List[Dict[str, Any]]:
        """
        Search documents using RAG with intelligent hybrid filtering
        
        Strategy:
        1. Extract key terms from ANY query (names, medical terms, technical words)
        2. Filter chunks that contain those terms (fast O(n) scan)
        3. Semantic search on filtered chunks (or all if no filter matches)
        
        Works with queries like:
        - "Who is Bob Carella?" (name extraction)
        - "What is myocardial infarction?" (medical term)
        - "How does the brain function?" (key term: brain)
        - "Tell me about diabetes treatment" (key terms: diabetes, treatment)
        
        Args:
            query: Search query
            k: Number of results to return
            
        Returns:
            List of relevant document chunks with metadata
        """
        if not query or not isinstance(query, str):
            return []
        
        # Extract key terms from query (names, medical terms, significant words)
        key_terms = self._extract_key_terms(query)
        
        if key_terms:
            print(f"[RAG] 🔍 Key terms detected: {key_terms}")
            # Pre-filter chunks by keyword match BEFORE semantic search
            filtered_indices = self._filter_chunks_by_terms(key_terms)
            
            if filtered_indices:
                print(f"[RAG] 🔍 Keyword filter: {len(filtered_indices)}/{len(self.chunks)} chunks contain key terms")
                # Do semantic search on filtered subset (much faster!)
                return self._search_filtered_chunks(query, filtered_indices, k)
            else:
                print(f"[RAG] ⚠️ No chunks contain key terms, using full semantic search")
        
        # Regular semantic search for non-name queries
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
            
            distances, indices = self._search_with_faiss_lite(query_embedding, k)
            
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
            if len(results) < k and len(results) > 0:
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
    
    def _filter_chunks_by_name(self, person_name: str, threshold: float = 0.65) -> List[int]:
        """
        Fast keyword filter: returns indices of chunks that contain fuzzy name matches
        This is O(n) in chunk count but avoids expensive semantic search on irrelevant chunks
        
        Args:
            person_name: Name to search for (e.g., "Bob Corella")
            threshold: Character similarity threshold (default 0.65 for phonetic variations)
            
        Returns:
            List of chunk indices that match
        """
        matching_indices = []
        query_words = person_name.split()
        
        print(f"[RAG] 🔍 Scanning {len(self.chunks)} chunks for '{person_name}'...")
        
        for idx, chunk in enumerate(self.chunks):
            if self._fuzzy_name_search(person_name, chunk, threshold):
                matching_indices.append(idx)
        
        return matching_indices
    
    def _filter_chunks_by_terms(self, key_terms: List[str], threshold: float = 0.75) -> List[int]:
        """
        Filter chunks by multiple key terms (OR logic with fuzzy matching)
        Returns chunks that contain ANY of the key terms
        
        Args:
            key_terms: List of key terms to search for (e.g., ["brain", "neural pathways"])
            threshold: Fuzzy match threshold (default 0.75)
            
        Returns:
            List of chunk indices that contain at least one key term
        """
        if not key_terms:
            return []
        
        matching_indices = set()  # Use set to avoid duplicates
        
        print(f"[RAG] 🔍 Scanning {len(self.chunks)} chunks for terms: {key_terms}")
        
        for term in key_terms:
            term_lower = term.lower()
            
            # Check if this is a name (capitalized multi-word)
            if ' ' in term and term[0].isupper():
                # Use name-specific fuzzy matching
                for idx, chunk in enumerate(self.chunks):
                    if self._fuzzy_name_search(term, chunk, threshold=0.65):
                        matching_indices.add(idx)
            else:
                # Use general term matching (case-insensitive)
                for idx, chunk in enumerate(self.chunks):
                    chunk_lower = chunk.lower()
                    
                    # Try exact match first (fastest)
                    if term_lower in chunk_lower:
                        matching_indices.add(idx)
                        continue
                    
                    # Try fuzzy matching for misspellings
                    if ' ' in term_lower:
                        # Multi-word term: match all words with fuzzy logic
                        term_words = term_lower.split()
                        chunk_words = re.findall(r'\b[a-z]+\b', chunk_lower)
                        
                        matches = 0
                        for term_word in term_words:
                            for chunk_word in chunk_words:
                                similarity = SequenceMatcher(None, term_word, chunk_word).ratio()
                                if similarity >= threshold:
                                    matches += 1
                                    break
                        
                        if matches >= len(term_words):
                            matching_indices.add(idx)
                    else:
                        # Single word: fuzzy match against chunk words
                        chunk_words = re.findall(r'\b[a-z]+\b', chunk_lower)
                        for chunk_word in chunk_words:
                            similarity = SequenceMatcher(None, term_lower, chunk_word).ratio()
                            if similarity >= threshold:
                                matching_indices.add(idx)
                                break
        
        result = sorted(list(matching_indices))
        print(f"[RAG] 🔍 Found {len(result)} chunks matching key terms")
        return result
    
    def _filter_chunks_by_term(self, medical_term: str, threshold: float = 0.75) -> List[int]:
        """
        Fast keyword filter: returns indices of chunks that contain medical term matches
        Uses case-insensitive fuzzy matching to handle variations like:
        - "myocardial infarction" vs "Myocardial Infarction"
        - "MI" vs "myocardial infarction" (if both present)
        
        Args:
            medical_term: Medical term to search for (e.g., "myocardial infarction")
            threshold: Character similarity threshold (default 0.75 for medical terms)
            
        Returns:
            List of chunk indices that match
        """
        matching_indices = []
        term_lower = medical_term.lower()
        term_words = term_lower.split()
        
        print(f"[RAG] 🔍 Scanning {len(self.chunks)} chunks for medical term '{medical_term}'...")
        
        for idx, chunk in enumerate(self.chunks):
            chunk_lower = chunk.lower()
            
            # First try exact match (fastest)
            if term_lower in chunk_lower:
                matching_indices.append(idx)
                continue
            
            # Try fuzzy matching for each term word
            chunk_words = re.findall(r'\b[a-z]+\b', chunk_lower)
            matches = 0
            
            for term_word in term_words:
                best_similarity = 0.0
                for chunk_word in chunk_words:
                    similarity = SequenceMatcher(None, term_word, chunk_word).ratio()
                    if similarity > best_similarity:
                        best_similarity = similarity
                
                if best_similarity >= threshold:
                    matches += 1
            
            # If all term words fuzzy match, include this chunk
            if matches >= len(term_words):
                matching_indices.append(idx)
        
        return matching_indices
    
    def _search_filtered_chunks(self, query: str, filtered_indices: List[int], k: int) -> List[Dict[str, Any]]:
        """
        Return filtered chunks sorted by relevance
        Since we've already filtered by keyword, just return those chunks
        
        Args:
            query: Search query  
            filtered_indices: List of chunk indices to return
            k: Number of results to return
            
        Returns:
            List of filtered chunks
        """
        print(f"[RAG] 🔍 Returning {len(filtered_indices)} keyword-matched chunks...")
        
        results = []
        for i, idx in enumerate(filtered_indices[:k]):
            chunk = self.chunks[idx]
            print(f"[RAG] 🔍 Result {i+1}: idx={idx}")
            print(f"[RAG] 🔍 Chunk preview: '{chunk[:100]}...'")
            
            results.append({
                'chunk': chunk,
                'score': 1.0,  # High score since it matched keywords
                'distance': 0.0,
                'rank': i + 1
            })
            print(f"[RAG] ✅ Added to results")
        
        print(f"[RAG] ✅ Returning {len(results)} keyword-filtered results")
        return results
    
    def _phonetic_match(self, word1: str, word2: str) -> bool:
        """
        Check if two words sound the same using Metaphone phonetic algorithm
        
        Args:
            word1: First word
            word2: Second word
            
        Returns:
            True if words have same phonetic encoding
        """
        # Simple Metaphone implementation (consonant-based)
        def metaphone(word):
            """Simplified Metaphone algorithm"""
            word = word.upper()
            if not word:
                return ""
            
            # Phonetic transformations
            result = []
            prev = ''
            
            for i, char in enumerate(word):
                # Skip vowels except at start
                if char in 'AEIOU':
                    if i == 0:
                        result.append(char)
                    continue
                
                # Handle consonant pairs and transformations
                if char == 'C':
                    if i + 1 < len(word) and word[i + 1] == 'H':
                        result.append('X')
                    elif i + 1 < len(word) and word[i + 1] in 'EIY':
                        result.append('S')
                    else:
                        result.append('K')
                elif char == 'G':
                    if i + 1 < len(word) and word[i + 1] in 'EIY':
                        result.append('J')
                    else:
                        result.append('G')
                elif char == 'P' and i + 1 < len(word) and word[i + 1] == 'H':
                    result.append('F')
                elif char == 'Q':
                    result.append('K')
                elif char == 'S' and i + 1 < len(word) and word[i + 1] == 'H':
                    result.append('X')
                elif char == 'T' and i + 1 < len(word) and word[i + 1] == 'H':
                    result.append('0')  # TH sound
                elif char == 'W' and prev in 'AEIOU':
                    continue  # Silent W after vowel
                elif char == 'X':
                    result.append('KS')
                elif char == 'Y' and prev not in 'AEIOU':
                    result.append('Y')
                elif char in 'BDFHJKLMNPRSTVZ':
                    # Skip doubles
                    if not result or result[-1] != char:
                        result.append(char)
                
                prev = char
            
            return ''.join(result)[:4]  # Return first 4 chars
        
        return metaphone(word1) == metaphone(word2)
    
    def _fuzzy_name_search(self, person_name: str, chunk: str, threshold: float = 0.65) -> bool:
        """
        Check if a chunk contains a fuzzy match for the person name using:
        1. Phonetic matching (Metaphone) - handles "Rafael"/"Raphael", "Smith"/"Smyth"
        2. Character-level similarity - handles typos like "Corella"/"Carella"
        3. Proximity check - ensures name words appear TOGETHER (not scattered across chunk)
        
        This prevents false matches like finding "Bob Cabello" when document has
        "Bob Kabeyo" and "Rafael Cabello" as separate people.
        
        Args:
            person_name: Person name from query (e.g., "Bob Corella")
            chunk: Document chunk
            threshold: Character similarity threshold (0.0 to 1.0, default 0.65 for phonetic variations)
            
        Returns:
            True if fuzzy match found, False otherwise
        """
        print(f"[RAG] 🔍 Fuzzy search: looking for '{person_name}' in chunk")
        
        # Extract ALL capitalized words from chunk WITH their positions
        chunk_words = []
        for match in re.finditer(r'\b[A-Z][a-z]+\b', chunk):
            chunk_words.append({
                'word': match.group(),
                'position': match.start()
            })
        
        if not chunk_words:
            print(f"[RAG] 🔍 No capitalized words found in chunk")
            return False
        
        print(f"[RAG] 🔍 Found {len(chunk_words)} capitalized words: {[w['word'] for w in chunk_words[:10]]}")  # Show first 10
        
        # Split query name into individual words
        query_words = person_name.split()
        print(f"[RAG] 🔍 Query words: {query_words}")
        
        # For each query word, find ALL possible matches in the chunk
        query_word_matches = []
        
        for query_idx, query_word in enumerate(query_words):
            matches_for_this_word = []
            
            for chunk_word_info in chunk_words:
                chunk_word = chunk_word_info['word']
                position = chunk_word_info['position']
                
            best_similarity = 0.0
                match_type = None
                
                # Try phonetic matching first (sounds the same?)
                if self._phonetic_match(query_word, chunk_word):
                    best_similarity = 1.0
                    match_type = "phonetic"
                else:
                    # Fall back to character-level similarity
                similarity = SequenceMatcher(None, query_word.lower(), chunk_word.lower()).ratio()
                
                    # Additional length check to prevent short words from matching
                    len_query = len(query_word)
                    len_chunk = len(chunk_word)
                    len_diff = abs(len_query - len_chunk)
                    
                    # Penalize if length difference is too large (>40% of the longer word)
                    max_len = max(len_query, len_chunk)
                    if len_diff > max_len * 0.4:
                        similarity *= 0.5  # Heavily penalize length mismatches
                    
                    # Also require minimum absolute character matches (at least 70% of shorter word)
                    min_len = min(len_query, len_chunk)
                    if similarity * max_len < min_len * 0.7:
                        similarity = 0.0  # Not enough actual character overlap
                    
                    best_similarity = similarity
                    match_type = "character"
                
                if best_similarity >= threshold:
                    matches_for_this_word.append({
                        'chunk_word': chunk_word,
                        'position': position,
                        'similarity': best_similarity,
                        'match_type': match_type
                    })
            
            query_word_matches.append(matches_for_this_word)
        
        # Check if ALL query words have at least one match
        if any(len(matches) == 0 for matches in query_word_matches):
            missing_words = [query_words[i] for i, matches in enumerate(query_word_matches) if len(matches) == 0]
            print(f"[RAG] ❌ Missing matches for: {missing_words}")
            return False
        
        # Now check if there's a valid sequence where matched words appear close together
        # For a 2-word name like "Bob Cabello", the matched words should be within ~50 characters
        MAX_NAME_SPAN = 50  # Maximum characters between first and last word of a name
        
        def check_proximity(match_combo):
            """Check if a combination of matches forms a valid name (words close together)"""
            positions = [m['position'] for m in match_combo]
            positions.sort()
            span = positions[-1] - positions[0]
            return span <= MAX_NAME_SPAN
        
        # Try all combinations of matches to find one where words are close together
        import itertools
        all_combinations = list(itertools.product(*query_word_matches))
        
        valid_combinations = []
        for combo in all_combinations:
            if check_proximity(combo):
                valid_combinations.append(combo)
        
        if valid_combinations:
            # Found at least one valid combination where all words match and are close together
            best_combo = valid_combinations[0]
            matched_pairs = []
            for i, match in enumerate(best_combo):
                matched_pairs.append(f"{query_words[i]}~{match['chunk_word']}")
                if match['match_type'] == "phonetic":
                    print(f"[RAG] ✅ Phonetic match: '{query_words[i]}' ~ '{match['chunk_word']}' (sounds alike)")
                else:
                    print(f"[RAG] ✅ Character match: '{query_words[i]}' ~ '{match['chunk_word']}' (similarity: {match['similarity']:.3f})")
            
            positions = sorted([m['position'] for m in best_combo])
            span = positions[-1] - positions[0]
            print(f"[RAG] ✅ All words matched and close together (span: {span} chars): {matched_pairs}")
            return True
        else:
            print(f"[RAG] ❌ Words match individually but not close together (not a valid name)")
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
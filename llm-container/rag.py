"""
Aura RAG Module - FAISS-GPU based retrieval for medical triage
Optimized for Jetson Orin NX with GPU acceleration
"""

import os
import numpy as np
import faiss
from typing import List, Dict, Any
import time

# Force CPU-only mode for Jetson compatibility
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['OMP_NUM_THREADS'] = '4'

# Additional PyTorch isolation
os.environ['TORCH_USE_CUDA_DSA'] = '0'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

# Fix transformers cache warning
os.environ['HF_HOME'] = './cache/huggingface'
os.environ['TRANSFORMERS_CACHE'] = './cache/transformers'

import torch
torch.set_num_threads(4)
torch.set_num_interop_threads(4)

from sentence_transformers import SentenceTransformer

class AuraRAG:
    def __init__(self, 
                 index_path: str = "data/embeddings/index.faiss",
                 chunks_path: str = "data/embeddings/doc_chunks.npy",
                 model_name: str = "all-MiniLM-L6-v2",
                 relevance_threshold: float = 0.3):
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
        self.relevance_threshold = relevance_threshold
        
        # Initialize components
        self.index = None
        self.chunks = None
        self.encoder = None
        self.gpu_available = False
        
        # Load components
        self._load_components()
    
    def _load_components(self):
        """Load FAISS index, chunks, and encoder model with robust error handling"""
        # Prevent duplicate loading
        if self.index is not None and self.chunks is not None and self.encoder is not None:
            print("[RAG] 🔧 Components already loaded, skipping...")
            return
            
        print("[RAG] 🔧 Loading RAG components...")
        
        # Add loading state tracking
        self._loading_state = {
            'index_loaded': False,
            'chunks_loaded': False,
            'encoder_loaded': False,
            'errors': []
        }
        
        # Load FAISS index with comprehensive error handling
        try:
            if os.path.exists(self.index_path):
                print(f"[RAG] 🔧 Loading FAISS index from: {self.index_path}")
                self.index = faiss.read_index(self.index_path)
                
                # Validate index
                if self.index.ntotal == 0:
                    raise ValueError("FAISS index is empty (0 vectors)")
                
                # Optimize CPU FAISS for Jetson (use multiple threads)
                faiss.omp_set_num_threads(4)  # Use 4 threads on Jetson Orin NX
                print(f"[RAG] ✅ Loaded FAISS index: {self.index_path} ({self.index.ntotal} vectors)")
                self._loading_state['index_loaded'] = True
            else:
                raise FileNotFoundError(f"FAISS index not found: {self.index_path}")
        except Exception as e:
            error_msg = f"Failed to load FAISS index: {e}"
            print(f"[RAG] ❌ {error_msg}")
            self._loading_state['errors'].append(error_msg)
            raise
        
        # Load document chunks with comprehensive error handling
        try:
            if os.path.exists(self.chunks_path):
                print(f"[RAG] 🔧 Loading document chunks from: {self.chunks_path}")
                self.chunks = np.load(self.chunks_path, allow_pickle=True)
                
                # Validate chunks
                if len(self.chunks) == 0:
                    raise ValueError("Document chunks file is empty")
                
                # Ensure chunks are strings
                for i, chunk in enumerate(self.chunks):
                    if not isinstance(chunk, str):
                        print(f"[RAG] ⚠️ Warning: Chunk {i} is not a string: {type(chunk)}")
                
                print(f"[RAG] ✅ Loaded {len(self.chunks)} document chunks")
                self._loading_state['chunks_loaded'] = True
            else:
                raise FileNotFoundError(f"Document chunks not found: {self.chunks_path}")
        except Exception as e:
            error_msg = f"Failed to load document chunks: {e}"
            print(f"[RAG] ❌ {error_msg}")
            self._loading_state['errors'].append(error_msg)
            raise
        
        # Load sentence transformer (force CPU for Jetson compatibility)
        try:
            # Set environment variables before importing torch
            os.environ['OMP_NUM_THREADS'] = '4'
            os.environ['MKL_NUM_THREADS'] = '4'
            
            import torch
            
            # Force CPU-only mode to avoid Jetson GPU tensor issues
            torch.set_num_threads(4)  # Optimize for Jetson CPU
            device = 'cpu'
            
            # Additional safety measures
            torch.backends.cudnn.enabled = False
            torch.backends.cuda.matmul.allow_tf32 = False
            
            print(f"[RAG] 🔧 Loading sentence transformer: {self.model_name}")
            
            # Method 1: Try loading with explicit offline mode
            try:
                # Set offline mode to prevent downloads
                os.environ['TRANSFORMERS_OFFLINE'] = '1'
                os.environ['HF_HUB_OFFLINE'] = '1'
                
                self.encoder = SentenceTransformer(
                    self.model_name,
                    device='cpu',
                    trust_remote_code=True,
                    cache_folder='/root/.cache/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2'
                )
                
                # Test the model
                print("[RAG] 🔧 Testing encoder with dummy input...")
                dummy_input = ["test sentence"]
                _ = self.encoder.encode(dummy_input)
                
                print(f"[RAG] ✅ Loaded encoder: {self.model_name} (device: cpu, threads: 4)")
                
            except Exception as e1:
                print(f"[RAG] 🔄 Method 1 failed: {e1}")
                
                # Method 2: Try with fresh download and proper initialization
                print("[RAG] 🔄 Trying fresh download method...")
                os.environ['TRANSFORMERS_OFFLINE'] = '0'  # Allow downloads
                os.environ['HF_HUB_OFFLINE'] = '0'
                
                # Clear potentially corrupted cache
                import shutil
                cache_dir = './cache/sentence_transformers'
                if os.path.exists(cache_dir):
                    shutil.rmtree(cache_dir)
                    print("[RAG] 🧹 Cleared corrupted cache")
                
                # Load with explicit model initialization
                self.encoder = SentenceTransformer(
                    self.model_name,
                    device='cpu',
                    trust_remote_code=True
                )
                
                # Ensure proper model state
                self.encoder.eval()
                
                # Test the model
                print("[RAG] 🔧 Testing encoder with dummy input...")
                dummy_input = ["test sentence"]
                _ = self.encoder.encode(dummy_input)
                
                print(f"[RAG] ✅ Loaded encoder with fresh download: {self.model_name}")
            
        except Exception as e:
            error_msg = f"Failed to load sentence transformer: {e}"
            print(f"[RAG] ❌ {error_msg}")
            self._loading_state['errors'].append(error_msg)
            print("[RAG] ⚠️ Using keyword-based fallback search")
            self.encoder = None
        
        # Final validation
        self._validate_components()
        
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
    
    def _validate_components(self):
        """Validate all loaded components and report health status"""
        print("[RAG] 🔍 Validating RAG components...")
        
        # Check index
        if self.index is None:
            print("[RAG] ❌ FAISS index not loaded")
        else:
            print(f"[RAG] ✅ FAISS index: {self.index.ntotal} vectors, dimension: {self.index.d}")
        
        # Check chunks
        if self.chunks is None:
            print("[RAG] ❌ Document chunks not loaded")
        else:
            print(f"[RAG] ✅ Document chunks: {len(self.chunks)} chunks")
        
        # Check encoder
        if self.encoder is None:
            print("[RAG] ❌ Sentence transformer not loaded (keyword search only)")
        else:
            print(f"[RAG] ✅ Sentence transformer: {self.model_name}")
        
        # Overall health status
        health_score = sum([
            self.index is not None,
            self.chunks is not None,
            self.encoder is not None
        ]) / 3.0
        
        print(f"[RAG] 📊 Health score: {health_score:.1%}")
        
        if health_score < 1.0:
            print(f"[RAG] ⚠️ RAG system partially functional (health: {health_score:.1%})")
            if self._loading_state['errors']:
                print(f"[RAG] 🔍 Errors encountered: {self._loading_state['errors']}")
    
    def get_health_status(self) -> dict:
        """Get comprehensive health status of RAG system"""
        return {
            'index_loaded': self.index is not None,
            'chunks_loaded': self.chunks is not None,
            'encoder_loaded': self.encoder is not None,
            'gpu_available': self.gpu_available,
            'health_score': sum([
                self.index is not None,
                self.chunks is not None,
                self.encoder is not None
            ]) / 3.0,
            'errors': getattr(self, '_loading_state', {}).get('errors', []),
            'index_size': self.index.ntotal if self.index else 0,
            'chunks_count': len(self.chunks) if self.chunks is not None else 0
        }
    
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
    
    def retrieve(self, query: str, k: int = 3, max_retries: int = 2) -> List[Dict[str, Any]]:
        """
        Hybrid retrieval: semantic + keyword for maximum intelligence
        Includes retry mechanism for reliability
        
        Args:
            query: User query string
            k: Number of chunks to retrieve
            max_retries: Maximum number of retry attempts
            
        Returns:
            List of relevant document chunks with metadata
        """
        start_time = time.time()
        
        # Validate inputs
        if not query or not isinstance(query, str):
            print(f"[RAG] ❌ Invalid query: {query}")
            return []
        
        # Check system health
        health = self.get_health_status()
        if health['health_score'] < 0.5:
            print(f"[RAG] ⚠️ RAG system health low ({health['health_score']:.1%}), using keyword search")
            return self._keyword_search(query, k)
        
        # Check if encoder is available
        if self.encoder is None:
            print("[RAG] ⚠️ Encoder not available, using keyword fallback search")
            return self._keyword_search(query, k)
        
        # Retry mechanism for semantic search
        for attempt in range(max_retries + 1):
            try:
                print(f"[RAG] 🔍 Semantic search attempt {attempt + 1}/{max_retries + 1}")
                semantic_results = self._semantic_search(query, k)
                
                if semantic_results:
                    elapsed = time.time() - start_time
                    print(f"[RAG] ✅ Semantic search found {len(semantic_results)} results in {elapsed:.2f}s")
                    return semantic_results
                else:
                    print("[RAG] 🔄 Semantic search found no results, trying keyword search...")
                    return self._keyword_search(query, k)
                    
            except Exception as e:
                print(f"[RAG] ❌ Semantic search attempt {attempt + 1} failed: {e}")
                if attempt < max_retries:
                    print(f"[RAG] 🔄 Retrying in 0.5s...")
                    time.sleep(0.5)
                else:
                    print(f"[RAG] ❌ All semantic search attempts failed, using keyword search")
                    return self._keyword_search(query, k)
    
    def _semantic_search(self, query: str, k: int = 3) -> List[Dict[str, Any]]:
        """
        Pure semantic search using sentence transformer
        
        Args:
            query: Search query
            k: Number of results to return
            
        Returns:
            List of relevant document chunks with metadata
        """
        start_time = time.time()
        
        # Encode query with robust error handling
        try:
            # Ensure query is a string and not empty
            if not query or not isinstance(query, str):
                print(f"[RAG] ❌ Invalid query: {query}")
                return self._keyword_search(query, k)
            
            # Clean and prepare query
            query = query.strip()
            if not query:
                print("[RAG] ❌ Empty query after cleaning")
                return self._keyword_search(query, k)
            
            print(f"[RAG] 🔍 Encoding query: '{query}'")
            
            # Simple CPU-only encoding with proper input format
            query_embedding = self.encoder.encode(query, convert_to_numpy=True, show_progress_bar=False)
            print(f"[RAG] 🔍 Query embedding shape: {query_embedding.shape}, type: {type(query_embedding)}")
            
            # Ensure numpy array
            if not isinstance(query_embedding, np.ndarray):
                query_embedding = np.array(query_embedding)
            
            # Ensure float32 dtype for FAISS compatibility
            query_embedding = query_embedding.astype(np.float32)
            
            # Ensure 2D array for FAISS
            if len(query_embedding.shape) == 1:
                query_embedding = query_embedding.reshape(1, -1)
            elif query_embedding.shape[0] > 1:
                query_embedding = query_embedding[0:1]  # Take first embedding
            
            print(f"[RAG] 🔍 Final embedding shape: {query_embedding.shape}, dtype: {query_embedding.dtype}")
            
        except Exception as e:
            print(f"[RAG] ❌ Encoding error: {e}")
            print("[RAG] 🔄 Falling back to keyword search")
            return self._keyword_search(query, k)
        
        # Search FAISS index with error handling
        try:
            distances, indices = self.index.search(query_embedding, k)
        except Exception as e:
            print(f"[RAG] ❌ FAISS search error: {e}")
            return []
        
        # Format results with enhanced relevance filtering
        results = []
        for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
            if idx < len(self.chunks):
                chunk = self.chunks[idx]
                # Convert L2 distance to similarity score (0-1 range)
                similarity_score = float(1.0 / (1.0 + distance))
                
                # Enhanced relevance check: verify the chunk actually contains query terms
                query_words = set(query.lower().split())
                chunk_words = set(chunk.lower().split())
                word_overlap = len(query_words.intersection(chunk_words))
                word_relevance = word_overlap / len(query_words) if query_words else 0
                
                # Combined relevance: semantic similarity + word overlap
                combined_relevance = (similarity_score * 0.7) + (word_relevance * 0.3)
                
                # Only include results with both semantic and word relevance
                if combined_relevance >= self.relevance_threshold and word_relevance > 0.1:
                    results.append({
                        'chunk': chunk,
                        'score': combined_relevance,
                        'semantic_score': similarity_score,
                        'word_relevance': word_relevance,
                        'rank': i + 1,
                        'distance': float(distance)
                    })
                    
                    # Debug: Show actual chunk content with relevance breakdown
                    print(f"[RAG] 📄 Chunk {i+1} (combined: {combined_relevance:.3f}, semantic: {similarity_score:.3f}, words: {word_relevance:.3f}):")
                    print(f"[RAG] 📄 Content: {chunk[:200]}{'...' if len(chunk) > 200 else ''}")
                    print(f"[RAG] 📄 ---")
                else:
                    print(f"[RAG] 🚫 Chunk {i+1} filtered out (combined: {combined_relevance:.3f}, words: {word_relevance:.3f})")
        
        retrieval_time = time.time() - start_time
        print(f"[RAG] 🔍 Retrieved {len(results)} chunks in {retrieval_time:.3f}s")
        
        return results
    
    def _keyword_search(self, query: str, k: int = 3) -> List[Dict[str, Any]]:
        """
        Fallback keyword-based search when sentence transformer is not available
        
        Args:
            query: Search query
            k: Number of results to return
            
        Returns:
            List of relevant chunks based on keyword matching
        """
        print(f"[RAG] 🔍 Keyword search for: '{query}'")
        
        # Enhanced keyword matching with name recognition
        query_words = set(query.lower().split())
        results = []
        
        # Extract potential names from query (capitalized words)
        query_names = [word for word in query.split() if word[0].isupper()]
        
        for i, chunk in enumerate(self.chunks):
            # Handle both string and dict chunk formats
            if isinstance(chunk, str):
                chunk_text = chunk.lower()
                chunk_original = chunk
            else:
                chunk_text = chunk.get('text', '').lower()
                chunk_original = chunk.get('text', '')
            
            chunk_words = set(chunk_text.split())
            
            # Calculate keyword overlap
            overlap = len(query_words.intersection(chunk_words))
            word_relevance = overlap / len(query_words) if query_words else 0
            
            # Name matching bonus (exact name matches get higher scores)
            name_bonus = 0
            for name in query_names:
                if name.lower() in chunk_text:
                    name_bonus += 0.3  # Significant bonus for name matches
            
            # Combined relevance with name bonus
            relevance_score = word_relevance + name_bonus
            
            # Only include results with meaningful relevance
            if relevance_score > 0.2:  # Higher threshold for keyword search
                results.append({
                    'chunk': chunk_original,
                    'score': relevance_score,
                    'word_relevance': word_relevance,
                    'name_bonus': name_bonus,
                    'rank': len(results) + 1,
                    'distance': 1.0 - relevance_score
                })
        
        # Sort by relevance and return top k
        results.sort(key=lambda x: x['score'], reverse=True)
        return results[:k]
    
    def _analyze_query_intent(self, query: str) -> dict:
        """
        Analyze query intent using dynamic patterns and NLP techniques
        
        Args:
            query: User query string
            
        Returns:
            Dictionary with intent analysis results
        """
        query_lower = query.lower().strip()
        words = query_lower.split()
        
        # Basic query characteristics
        analysis = {
            'word_count': len(words),
            'has_question_word': any(w in words for w in ['what', 'who', 'where', 'when', 'why', 'how', 'which', 'can', 'could', 'would', 'should', 'is', 'are', 'was', 'were', 'do', 'does', 'did']),
            'is_greeting': False,
            'is_conversational': False,
            'is_informational': False,
            'confidence': 0.0
        }
        
        # Detect greetings and conversational patterns
        greeting_patterns = ['hello', 'hi', 'hey', 'good morning', 'good afternoon', 'good evening']
        conversational_patterns = ['how are you', 'what\'s up', 'thanks', 'thank you', 'bye', 'goodbye']
        
        if any(pattern in query_lower for pattern in greeting_patterns + conversational_patterns):
            analysis['is_greeting'] = True
            analysis['is_conversational'] = True
            analysis['confidence'] = 0.9
            return analysis
        
        # Detect informational queries (questions seeking factual information)
        informational_starters = ['what is', 'who is', 'who was', 'tell me about', 'explain', 'describe', 'what are', 'how do', 'how does', 'how can', 'what does', 'what did', 'what was', 'what were', 'where is', 'where are', 'when is', 'when was', 'why is', 'why are']
        if any(query_lower.startswith(starter) for starter in informational_starters):
            analysis['is_informational'] = True
            analysis['confidence'] = 0.8
        
        # Check for question words and patterns
        if analysis['has_question_word']:
            analysis['is_informational'] = True
            analysis['confidence'] = max(analysis['confidence'], 0.6)
        
        # Detect imperative/request patterns
        imperative_patterns = ['tell me', 'explain', 'describe', 'show me', 'give me']
        if any(pattern in query_lower for pattern in imperative_patterns):
            analysis['is_informational'] = True
            analysis['confidence'] = max(analysis['confidence'], 0.7)
        
        return analysis
    
    def _has_document_relevance(self, query: str) -> float:
        """
        Dynamically assess if query might be relevant to available documents
        Uses a lightweight similarity check against document content
        
        Args:
            query: User query string
            
        Returns:
            Relevance score (0.0 to 1.0)
        """
        if not hasattr(self, 'chunks') or self.chunks is None:
            return 0.0
        
        query_words = set(query.lower().split())
        
        # Sample a subset of chunks for efficiency (first 50 chunks)
        sample_chunks = self.chunks[:min(50, len(self.chunks))]
        
        relevance_scores = []
        for chunk in sample_chunks:
            chunk_words = set(chunk.lower().split())
            
            # Calculate word overlap
            overlap = len(query_words.intersection(chunk_words))
            if overlap > 0:
                # Normalize by query length
                score = overlap / len(query_words)
                relevance_scores.append(score)
        
        # Return average relevance if any matches found
        if relevance_scores:
            return sum(relevance_scores) / len(relevance_scores)
        
        return 0.0
    
    def should_use_rag(self, query: str) -> bool:
        """
        Dynamically determine if RAG should be used based on query analysis
        More inclusive approach: any question or complex query should trigger RAG
        
        Args:
            query: User query string
            
        Returns:
            Boolean indicating if RAG should be used
        """
        # Skip very short queries (but allow single words if they're questions)
        words = query.split()
        if len(words) < 1:
            print(f"[RAG] 🚫 Empty query")
            return False
        
        # Analyze query intent
        intent = self._analyze_query_intent(query)
        print(f"[RAG] 🔍 Intent analysis: {intent}")
        
        # Skip only casual greetings (not informational greetings)
        if intent['is_greeting'] and not intent['is_informational']:
            print(f"[RAG] 🚫 Casual greeting detected: '{query}'")
            return False
        
        # Use RAG for ANY question (not just medical)
        if intent['has_question_word']:
            print(f"[RAG] ✅ Question detected: '{query}'")
            return True
        
        # Use RAG for informational queries (lowered threshold)
        if intent['is_informational'] and intent['confidence'] >= 0.3:
            print(f"[RAG] ✅ Informational query: '{query}'")
            return True
        
        # Use RAG for complex queries based on length
        if len(words) >= 4:  # Longer queries are more likely to be informational
            print(f"[RAG] ✅ Complex query (length: {len(words)}): '{query}'")
            return True
        
        # For shorter queries, do quick relevance check
        if len(words) >= 2:
            doc_relevance = self._has_document_relevance(query)
            print(f"[RAG] 🔍 Document relevance score: {doc_relevance:.3f}")
            if doc_relevance > 0.05:  # Lowered threshold for relevance
                print(f"[RAG] ✅ Document relevance found: '{query}'")
                return True
        
        # Default to using RAG for unclear cases (more inclusive)
        print(f"[RAG] ✅ Defaulting to RAG for query: '{query}'")
        return True
    
    def smart_retrieve(self, query: str, k: int = 3) -> tuple[bool, List[Dict[str, Any]]]:
        """
        Smart retrieval that decides whether to use RAG and returns results
        Includes quick relevance pre-check to reduce latency
        
        Args:
            query: User query string
            k: Number of chunks to retrieve
            
        Returns:
            Tuple of (should_use_rag, results)
        """
        if not self.should_use_rag(query):
            return False, []
        
        # Quick relevance pre-check to avoid expensive semantic search if not needed
        print(f"[RAG] 🔍 Quick relevance check for: '{query}'")
        quick_relevance = self._has_document_relevance(query)
        print(f"[RAG] 🔍 Quick relevance score: {quick_relevance:.3f}")
        
        # If quick check shows no relevance, skip expensive search
        if quick_relevance < 0.02:  # Very low threshold for quick check
            print(f"[RAG] 🚫 Quick check: No document relevance for: '{query}'")
            return False, []
            
        # Proceed with full semantic search
        print(f"[RAG] 🔍 Proceeding with semantic search for: '{query}'")
        results = self.retrieve(query, k)
        
        # If no results meet the relevance threshold, don't use RAG
        if not results:
            print(f"[RAG] 🚫 No relevant results found for query: '{query}'")
            return False, []
            
        print(f"[RAG] ✅ Found {len(results)} relevant results for query: '{query}'")
        return True, results
    
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
        augmented_prompt = f"""Based on the following information:

{context}

Please answer this question: {user_query}

Provide a helpful, accurate response based on the information above."""
        
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

def smart_search_medical_info(query: str, k: int = 3) -> tuple[bool, str]:
    """
    Smart wrapper that decides whether to use RAG and returns augmented prompt
    
    Args:
        query: User query string
        k: Number of chunks to retrieve
        
    Returns:
        Tuple of (used_rag, prompt) where prompt is either augmented or original
    """
    rag = get_rag()
    should_use, results = rag.smart_retrieve(query, k)
    
    if should_use and results:
        augmented_prompt = rag.augment_prompt(query, results)
        return True, augmented_prompt
    else:
        return False, query

# Test function
def test_rag():
    """Test RAG functionality"""
    print("[RAG] 🧪 Testing RAG system...")
    
    try:
        rag = get_rag()  # Use global instance instead of creating new one
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

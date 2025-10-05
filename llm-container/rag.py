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
            
            self.encoder = SentenceTransformer(self.model_name, device=device)
            # Don't call .to('cpu') again - it's already on CPU from the device parameter
            print(f"[RAG] ✅ Loaded encoder: {self.model_name} (device: {device}, threads: 4)")
            
        except Exception as e:
            print(f"[RAG] ❌ Failed to load sentence transformer: {e}")
            # Fallback: create a dummy encoder that returns zeros
            self.encoder = None
            print("[RAG] ⚠️ Using fallback encoder (no semantic search)")
        
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
        
        # Check if encoder is available
        if self.encoder is None:
            print("[RAG] ⚠️ Encoder not available, cannot perform semantic search")
            return []
        
        # Encode query with robust error handling
        try:
            # Simple CPU-only encoding
            query_embedding = self.encoder.encode([query], convert_to_numpy=True)
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
            raise Exception(f"Failed to encode query: {e}")
        
        # Search FAISS index with error handling
        try:
            distances, indices = self.index.search(query_embedding, k)
        except Exception as e:
            print(f"[RAG] ❌ FAISS search error: {e}")
            return []
        
        # Format results with relevance filtering
        results = []
        for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
            if idx < len(self.chunks):
                chunk = self.chunks[idx]
                # Convert L2 distance to similarity score (0-1 range)
                similarity_score = float(1.0 / (1.0 + distance))
                
                # Only include results above relevance threshold
                if similarity_score >= self.relevance_threshold:
                    results.append({
                        'chunk': chunk,
                        'score': similarity_score,
                        'rank': i + 1,
                        'distance': float(distance)
                    })
                    
                    # Debug: Show actual chunk content
                    print(f"[RAG] 📄 Chunk {i+1} (score: {similarity_score:.3f}):")
                    print(f"[RAG] 📄 Content: {chunk[:200]}{'...' if len(chunk) > 200 else ''}")
                    print(f"[RAG] 📄 Full content: {chunk}")
                    print(f"[RAG] 📄 ---")
        
        retrieval_time = time.time() - start_time
        print(f"[RAG] 🔍 Retrieved {len(results)} chunks in {retrieval_time:.3f}s")
        
        return results
    
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
            'has_question_word': any(w in words for w in ['what', 'who', 'where', 'when', 'why', 'how', 'which']),
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
        informational_starters = ['what is', 'who is', 'who was', 'tell me about', 'explain', 'describe']
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
        
        Args:
            query: User query string
            
        Returns:
            Boolean indicating if RAG should be used
        """
        # Skip very short queries
        if len(query.split()) < 2:
            print(f"[RAG] 🚫 Query too short: '{query}'")
            return False
        
        # Analyze query intent
        intent = self._analyze_query_intent(query)
        print(f"[RAG] 🔍 Intent analysis: {intent}")
        
        # Skip greetings and casual conversation
        if intent['is_greeting'] or intent['is_conversational']:
            print(f"[RAG] 🚫 Casual conversation detected: '{query}'")
            return False
        
        # Use RAG for informational queries with high confidence
        if intent['is_informational'] and intent['confidence'] >= 0.6:
            print(f"[RAG] ✅ High-confidence informational query: '{query}'")
            return True
        
        # For borderline cases, check document relevance
        if intent['confidence'] >= 0.4:
            doc_relevance = self._has_document_relevance(query)
            print(f"[RAG] 🔍 Document relevance score: {doc_relevance:.3f}")
            if doc_relevance > 0.1:  # Found some word overlap with documents
                print(f"[RAG] ✅ Document relevance found: '{query}'")
                return True
        
        # Default to not using RAG for unclear intent
        print(f"[RAG] 🚫 Unclear intent, skipping RAG: '{query}'")
        return False
    
    def smart_retrieve(self, query: str, k: int = 3) -> tuple[bool, List[Dict[str, Any]]]:
        """
        Smart retrieval that decides whether to use RAG and returns results
        
        Args:
            query: User query string
            k: Number of chunks to retrieve
            
        Returns:
            Tuple of (should_use_rag, results)
        """
        if not self.should_use_rag(query):
            return False, []
            
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

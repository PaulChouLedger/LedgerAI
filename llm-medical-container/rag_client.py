"""
RAG Client - Modular RAG system with GPU/CPU fallback
Supports both external RAG container (GPU) and internal FAISS (CPU)
"""

import os
import requests
import numpy as np
from typing import List, Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

# Configuration
RAG_ENABLED = os.environ.get('RAG_ENABLED', 'false').lower() == 'true'
RAG_SERVICE_URL = os.environ.get('RAG_SERVICE_URL', 'http://localhost:11435')
RAG_TIMEOUT = int(os.environ.get('RAG_TIMEOUT', '10'))

class RAGClient:
    """
    Unified RAG client that supports both GPU (external container) and CPU (local) modes
    
    GPU Mode:
        - Uses HTTP API calls to external RAG container
        - Slower due to network overhead
        - Better for distributed systems
    
    CPU Mode:
        - Direct in-process FAISS operations
        - Faster (no network overhead)
        - Simpler (no external dependencies)
        - All operations happen locally within the LLM container
    """
    
    def __init__(self, use_gpu: bool = None):
        """
        Initialize RAG client
        
        Args:
            use_gpu: Force GPU mode (True) or CPU mode (False). If None, uses RAG_ENABLED env var
        """
        self.use_gpu = RAG_ENABLED if use_gpu is None else use_gpu
        self._cpu_rag = None
        self._mode = "GPU (External RAG Container - HTTP API)" if self.use_gpu else "CPU (Local FAISS - In-Process)"
        
        logger.info(f"[RAG Client] Initialized in {self._mode} mode")
        
        if self.use_gpu:
            self._check_rag_service()
        else:
            # CPU mode: Initialize local FAISS - NO HTTP calls needed!
            self._initialize_cpu_rag()
    
    def _check_rag_service(self) -> bool:
        """Check if external RAG service is available"""
        try:
            response = requests.get(f"{RAG_SERVICE_URL}/health", timeout=2)
            if response.status_code == 200:
                logger.info(f"[RAG Client] ✅ RAG service available at {RAG_SERVICE_URL}")
                return True
            else:
                logger.warning(f"[RAG Client] ⚠️ RAG service unhealthy: {response.status_code}")
                return False
        except Exception as e:
            logger.error(f"[RAG Client] ❌ RAG service unavailable: {e}")
            logger.info(f"[RAG Client] 🔄 Falling back to CPU mode")
            self.use_gpu = False
            self._mode = "CPU (Local FAISS - Fallback)"
            self._initialize_cpu_rag()
            return False
    
    def _initialize_cpu_rag(self):
        """Initialize local CPU-based RAG system"""
        try:
            from sentence_transformers import SentenceTransformer
            import faiss
            
            logger.info("[RAG Client] 🔧 Initializing CPU RAG system...")
            
            # Use the same model as the GPU version for consistency
            self._embedding_model = SentenceTransformer('all-distilroberta-v1')
            self._embedding_dim = 768
            
            # Initialize FAISS CPU index
            self._cpu_index = None
            self._cpu_chunks = []
            self._cpu_metadata = []
            
            # Try to load pre-existing index
            self._load_cpu_index()
            
            logger.info("[RAG Client] ✅ CPU RAG system initialized")
            
        except ImportError as e:
            logger.error(f"[RAG Client] ❌ Failed to import CPU RAG dependencies: {e}")
            logger.error("[RAG Client] Install with: pip install sentence-transformers faiss-cpu")
            raise
        except Exception as e:
            logger.error(f"[RAG Client] ❌ Failed to initialize CPU RAG: {e}")
            raise
    
    def _load_cpu_index(self):
        """Load existing FAISS index from disk"""
        index_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'embeddings')
        
        try:
            import faiss
            import pickle
            
            faiss_index_path = os.path.join(index_path, 'faiss_index.bin')
            metadata_path = os.path.join(index_path, 'metadata.pkl')
            
            if os.path.exists(faiss_index_path) and os.path.exists(metadata_path):
                self._cpu_index = faiss.read_index(faiss_index_path)
                
                with open(metadata_path, 'rb') as f:
                    data = pickle.load(f)
                    self._cpu_chunks = data.get('chunks', [])
                    self._cpu_metadata = data.get('metadata', [])
                
                logger.info(f"[RAG Client] ✅ Loaded {len(self._cpu_chunks)} chunks from CPU index")
            else:
                logger.warning("[RAG Client] ⚠️ No existing CPU index found")
                # Create empty index
                import faiss
                self._cpu_index = faiss.IndexFlatL2(self._embedding_dim)
                
        except Exception as e:
            logger.error(f"[RAG Client] ❌ Failed to load CPU index: {e}")
            # Create empty index as fallback
            import faiss
            self._cpu_index = faiss.IndexFlatL2(self._embedding_dim)
    
    def search(self, query: str, k: int = 5, threshold: float = 0.3) -> List[Dict]:
        """
        Search for relevant medical information
        
        Args:
            query: Search query
            k: Number of results to return
            threshold: Similarity threshold (0-1)
        
        Returns:
            List of search results with text, score, and metadata
        """
        if self.use_gpu:
            return self._search_gpu(query, k, threshold)
        else:
            return self._search_cpu(query, k, threshold)
    
    def _search_gpu(self, query: str, k: int, threshold: float) -> List[Dict]:
        """Search using external RAG container (GPU)"""
        try:
            response = requests.post(
                f"{RAG_SERVICE_URL}/rag/search",
                json={"query": query, "k": k, "threshold": threshold},
                timeout=RAG_TIMEOUT
            )
            
            if response.status_code == 200:
                data = response.json()
                return data.get('results', [])
            else:
                logger.error(f"[RAG Client] GPU search failed: {response.status_code}")
                return []
                
        except requests.exceptions.Timeout:
            logger.error(f"[RAG Client] GPU search timeout after {RAG_TIMEOUT}s")
            return []
        except Exception as e:
            logger.error(f"[RAG Client] GPU search error: {e}")
            return []
    
    def _search_cpu(self, query: str, k: int, threshold: float) -> List[Dict]:
        """Search using local CPU FAISS"""
        try:
            if self._cpu_index is None or len(self._cpu_chunks) == 0:
                logger.warning("[RAG Client] CPU index is empty")
                return []
            
            # Generate query embedding
            query_embedding = self._embedding_model.encode([query])[0]
            query_embedding = np.array([query_embedding]).astype('float32')
            
            # Search FAISS index
            distances, indices = self._cpu_index.search(query_embedding, k)
            
            # Convert distances to similarity scores (L2 distance -> cosine similarity approximation)
            # Lower distance = higher similarity
            scores = 1 / (1 + distances[0])
            
            # Filter by threshold and build results
            results = []
            for idx, score in zip(indices[0], scores):
                if idx < len(self._cpu_chunks) and score >= threshold:
                    results.append({
                        'text': self._cpu_chunks[idx],
                        'score': float(score),
                        'metadata': self._cpu_metadata[idx] if idx < len(self._cpu_metadata) else {}
                    })
            
            logger.info(f"[RAG Client] CPU search found {len(results)} results (threshold={threshold})")
            return results
            
        except Exception as e:
            logger.error(f"[RAG Client] CPU search error: {e}")
            return []
    
    def embed(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for texts
        
        Args:
            texts: List of texts to embed
        
        Returns:
            List of embedding vectors
        """
        if self.use_gpu:
            return self._embed_gpu(texts)
        else:
            return self._embed_cpu(texts)
    
    def _embed_gpu(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using external RAG container (GPU)"""
        try:
            response = requests.post(
                f"{RAG_SERVICE_URL}/embed",
                json={"texts": texts},
                timeout=RAG_TIMEOUT
            )
            
            if response.status_code == 200:
                data = response.json()
                return data.get('embeddings', [])
            else:
                logger.error(f"[RAG Client] GPU embedding failed: {response.status_code}")
                return []
                
        except Exception as e:
            logger.error(f"[RAG Client] GPU embedding error: {e}")
            return []
    
    def _embed_cpu(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using local CPU model"""
        try:
            embeddings = self._embedding_model.encode(texts)
            return embeddings.tolist()
        except Exception as e:
            logger.error(f"[RAG Client] CPU embedding error: {e}")
            return []
    
    def get_guideline(self, guideline_name: str) -> Dict:
        """
        Get all chunks for a specific guideline
        
        Args:
            guideline_name: Name of the guideline
        
        Returns:
            Dictionary with guideline chunks and metadata
        """
        if self.use_gpu:
            return self._get_guideline_gpu(guideline_name)
        else:
            return self._get_guideline_cpu(guideline_name)
    
    def _get_guideline_gpu(self, guideline_name: str) -> Dict:
        """Get guideline using external RAG container (GPU)"""
        try:
            response = requests.get(
                f"{RAG_SERVICE_URL}/rag/guideline/{guideline_name}",
                timeout=RAG_TIMEOUT
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"[RAG Client] GPU guideline fetch failed: {response.status_code}")
                return {}
                
        except Exception as e:
            logger.error(f"[RAG Client] GPU guideline fetch error: {e}")
            return {}
    
    def _get_guideline_cpu(self, guideline_name: str) -> Dict:
        """Get guideline from local CPU index"""
        try:
            # Filter chunks by guideline name in metadata
            guideline_chunks = []
            for i, metadata in enumerate(self._cpu_metadata):
                if metadata.get('guideline_name') == guideline_name:
                    guideline_chunks.append({
                        'text': self._cpu_chunks[i],
                        'metadata': metadata
                    })
            
            return {
                'guideline_name': guideline_name,
                'chunks': guideline_chunks,
                'total': len(guideline_chunks)
            }
            
        except Exception as e:
            logger.error(f"[RAG Client] CPU guideline fetch error: {e}")
            return {}
    
    def get_mode(self) -> str:
        """Get current RAG mode"""
        return self._mode
    
    def is_gpu_enabled(self) -> bool:
        """Check if GPU mode is enabled"""
        return self.use_gpu


# Singleton instance
_rag_client = None

def get_rag_client(use_gpu: bool = None) -> RAGClient:
    """Get or create RAG client singleton"""
    global _rag_client
    if _rag_client is None:
        _rag_client = RAGClient(use_gpu=use_gpu)
    return _rag_client


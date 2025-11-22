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
RAG_MODE = os.environ.get('RAG_MODE', 'CPU').upper()  # GPU = RAG container, CPU = CPU FAISS
RAG_SERVICE_URL = os.environ.get('RAG_SERVICE_URL', 'http://localhost:11435')
RAG_TIMEOUT = int(os.environ.get('RAG_TIMEOUT', '10'))

class RAGClient:
    """
    Unified RAG client that supports both GPU (external container) and CPU (local) modes
    
    GPU Mode:
        - Uses HTTP API calls to external RAG container
        - GPU-accelerated FAISS for faster searches on large datasets
        - Network overhead for small queries, but faster for large batches
        - Better for distributed systems and production deployments
    
    CPU Mode:
        - Direct in-process FAISS operations
        - Faster for small queries (no network overhead)
        - Simpler (no external dependencies)
        - All operations happen locally within the LLM container
        - Better for development and systems without GPU
    """
    
    def __init__(self, use_gpu: bool = None):
        """
        Initialize RAG client
        
        Args:
            use_gpu: Force GPU mode (True) or CPU mode (False). If None, uses RAG_MODE env var
        """
        self.use_gpu = (RAG_MODE == 'GPU') if use_gpu is None else use_gpu
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
        """Initialize local CPU-based RAG system with auto-ingestion"""
        try:
            from sentence_transformers import SentenceTransformer
            import faiss
            
            print("[RAG Client] 🔧 Initializing CPU RAG system with auto-ingestion...")
            logger.info("[RAG Client] 🔧 Initializing CPU RAG system with auto-ingestion...")
            
            # Use all-distilroberta-v1 (benchmarked as best performing model)
            print("[RAG Client] 📥 Loading embedding model: all-distilroberta-v1...")
            self._embedding_model = SentenceTransformer('all-distilroberta-v1')
            self._embedding_dim = 768
            print("[RAG Client] ✅ Embedding model loaded")
            
            # Initialize FAISS CPU index
            self._cpu_index = None
            self._cpu_chunks = []
            self._cpu_metadata = []
            
            # Initialize auto-ingestion system
            print("[RAG Client] 🔄 Initializing auto-ingestion system...")
            self._initialize_auto_ingestion()
            
            # Try to load pre-existing index
            print("[RAG Client] 📂 Loading existing embeddings/index...")
            self._load_cpu_index()
            
            # Show summary
            chunk_count = len(self._cpu_chunks) if self._cpu_chunks else 0
            index_size = self._cpu_index.ntotal if self._cpu_index and hasattr(self._cpu_index, 'ntotal') else 0
            print(f"[RAG Client] ✅ CPU RAG system initialized: {chunk_count} chunks, {index_size} vectors in index")
            logger.info(f"[RAG Client] ✅ CPU RAG system initialized: {chunk_count} chunks, {index_size} vectors in index")
            
        except ImportError as e:
            logger.error(f"[RAG Client] ❌ Failed to import CPU RAG dependencies: {e}")
            logger.error("[RAG Client] Install with: pip install sentence-transformers faiss-cpu")
            raise
        except Exception as e:
            logger.error(f"[RAG Client] ❌ Failed to initialize CPU RAG: {e}")
            raise
    
    def _load_cpu_index(self):
        """Load existing FAISS index from disk"""
        print("[RAG Client] 📂 Attempting to load CPU FAISS index from disk...")
        index_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'embeddings')
        print(f"[RAG Client] 📂 Index path: {index_path}")
        
        try:
            import faiss
            import pickle
            
            faiss_index_path = os.path.join(index_path, 'faiss_index.bin')
            metadata_path = os.path.join(index_path, 'metadata.pkl')
            
            print(f"[RAG Client] 📂 Checking for index: {faiss_index_path}")
            print(f"[RAG Client] 📂 Checking for metadata: {metadata_path}")
            
            if os.path.exists(faiss_index_path) and os.path.exists(metadata_path):
                print("[RAG Client] ✅ Found existing index files, loading...")
                self._cpu_index = faiss.read_index(faiss_index_path)
                
                with open(metadata_path, 'rb') as f:
                    data = pickle.load(f)
                    self._cpu_chunks = data.get('chunks', [])
                    self._cpu_metadata = data.get('metadata', [])
                
                print(f"[RAG Client] ✅ Loaded {len(self._cpu_chunks)} chunks from CPU index")
                logger.info(f"[RAG Client] ✅ Loaded {len(self._cpu_chunks)} chunks from CPU index")
            else:
                print("[RAG Client] ⚠️ No existing CPU index found - creating empty index")
                logger.warning("[RAG Client] ⚠️ No existing CPU index found")
                # Create empty index
                import faiss
                self._cpu_index = faiss.IndexFlatL2(self._embedding_dim)
                
        except Exception as e:
            print(f"[RAG Client] ❌ Failed to load CPU index: {e}")
            logger.error(f"[RAG Client] ❌ Failed to load CPU index: {e}")
            import traceback
            traceback.print_exc()
            # Create empty index as fallback
            import faiss
            self._cpu_index = faiss.IndexFlatL2(self._embedding_dim)
    
    def _initialize_auto_ingestion(self):
        """Initialize auto-ingestion system for CPU FAISS"""
        try:
            # Import the auto-ingestion system
            from .cpu_faiss_auto_ingest import CPUFAISSAutoIngest
            
            print("[RAG Client] 🔄 Initializing auto-ingestion system...")
            # Initialize auto-ingestion
            self._auto_ingest = CPUFAISSAutoIngest()
            
            # Load existing embeddings
            print("[RAG Client] 📂 Loading existing embeddings...")
            if self._auto_ingest.load_existing_embeddings():
                print("[RAG Client] ✅ Loaded existing embeddings via auto-ingestion")
                logger.info("[RAG Client] ✅ Loaded existing embeddings via auto-ingestion")
            else:
                print("[RAG Client] ⚠️ No existing embeddings found, will scan input directory")
                logger.info("[RAG Client] ⚠️ No existing embeddings found, will process on first scan")
            
            # Run initial scan to process any missing files
            print("[RAG Client] 🔍 Running initial scan for missing files...")
            scan_result = self._auto_ingest.scan_and_process()
            if scan_result['processed'] > 0:
                print(f"[RAG Client] ✅ Initial scan processed {scan_result['processed']} file(s)")
                # Reload embeddings after processing
                self._auto_ingest.load_existing_embeddings()
                # Update our local references
                self._cpu_chunks = self._auto_ingest.chunks
                self._cpu_metadata = self._auto_ingest.metadata
                # Rebuild index if we have chunks
                if self._cpu_chunks and len(self._cpu_chunks) > 0:
                    print(f"[RAG Client] 🔧 Rebuilding FAISS index with {len(self._cpu_chunks)} chunks...")
                    self._rebuild_cpu_index()
            else:
                print(f"[RAG Client] ℹ️ Initial scan: {scan_result['processed']} processed, {scan_result['skipped']} skipped")
            
            # Start file watching
            self._auto_ingest.start_watching()
            print(f"[RAG Client] 👀 Started auto-ingestion file watching: {self._auto_ingest.input_dir}")
            logger.info("[RAG Client] 👀 Started auto-ingestion file watching")
            
        except ImportError as e:
            logger.warning(f"[RAG Client] ⚠️ Auto-ingestion not available: {e}")
            self._auto_ingest = None
        except Exception as e:
            logger.error(f"[RAG Client] ❌ Failed to initialize auto-ingestion: {e}")
            self._auto_ingest = None
    
    def quick_content_match(self, query: str) -> bool:
        """
        Quick substring match to check if query terms appear in RAG content.
        Much faster than full semantic search - used to decide if RAG should be used.
        
        Args:
            query: Search query
        
        Returns:
            True if query terms match any RAG content, False otherwise
        """
        if not query or not query.strip():
            return False
        
        # Extract key terms from query (remove common stop words)
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'is', 'are', 'was', 'were', 'do', 'does', 'did', 'how', 'what', 'when', 'where', 'why', 'can', 'could', 'should', 'would', 'may', 'might', 'must'}
        query_lower = query.lower()
        # Extract words (2+ characters) that aren't stop words
        import re
        words = re.findall(r'\b\w{2,}\b', query_lower)
        key_terms = [w for w in words if w not in stop_words]
        
        if not key_terms:
            return False
        
        # Quick substring match against RAG chunks
        if self.use_gpu:
            # For GPU mode, we'd need to check via API, but that's slow
            # Instead, assume GPU has content if service is available
            try:
                response = requests.get(f"{RAG_SERVICE_URL}/rag/stats", timeout=1)
                if response.status_code == 200:
                    stats = response.json()
                    return stats.get('chunks_loaded', 0) > 0
            except:
                pass
            return False
        else:
            # CPU mode: quick substring match against chunk text
            if not self._cpu_chunks or len(self._cpu_chunks) == 0:
                # Try to reload if empty
                if self._auto_ingest:
                    if self._auto_ingest.load_existing_embeddings():
                        self._cpu_chunks = self._auto_ingest.chunks
                        self._cpu_metadata = self._auto_ingest.metadata
            
            if not self._cpu_chunks or len(self._cpu_chunks) == 0:
                return False
            
            # Check if any key term appears in any chunk text
            # For medical/technical queries, check more chunks (up to 500) to catch relevant content
            # This is still fast (substring match) compared to full semantic search
            chunks_to_check = min(500, len(self._cpu_chunks))
            matches_found = 0
            for i in range(chunks_to_check):
                # Chunks are strings, not dictionaries
                chunk = self._cpu_chunks[i]
                if isinstance(chunk, dict):
                    chunk_text = chunk.get('text', '').lower()
                else:
                    chunk_text = str(chunk).lower()
                
                # Check if multiple key terms match (more confident match)
                matching_terms = sum(1 for term in key_terms if term in chunk_text)
                if matching_terms >= 2:  # At least 2 key terms match
                    return True
                elif matching_terms == 1 and matches_found == 0:  # First single match
                    matches_found = 1
            
            # If we found at least one single-term match, use RAG (better than nothing)
            return matches_found > 0
    
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
        print(f"[RAG Client] 🚀 GPU search: query='{query[:50]}...', k={k}, threshold={threshold}")
        try:
            response = requests.post(
                f"{RAG_SERVICE_URL}/rag/search",
                json={"query": query, "k": k, "threshold": threshold},
                timeout=RAG_TIMEOUT
            )
            
            if response.status_code == 200:
                data = response.json()
                results = data.get('results', [])
                print(f"[RAG Client] ✅ GPU search returned {len(results)} results")
                if results:
                    for i, result in enumerate(results, 1):
                        score = result.get('score', 0)
                        text_preview = result.get('text', '')[:50]
                        print(f"[RAG Client]   [{i}] Score: {score:.3f}, Preview: '{text_preview}...'")
                return results
            else:
                print(f"[RAG Client] ❌ GPU search failed: HTTP {response.status_code}")
                logger.error(f"[RAG Client] GPU search failed: {response.status_code}")
                return []
                
        except requests.exceptions.Timeout:
            print(f"[RAG Client] ⏱️ GPU search timeout after {RAG_TIMEOUT}s")
            logger.error(f"[RAG Client] GPU search timeout after {RAG_TIMEOUT}s")
            return []
        except Exception as e:
            print(f"[RAG Client] ❌ GPU search error: {e}")
            logger.error(f"[RAG Client] GPU search error: {e}")
            return []
    
    def _search_cpu(self, query: str, k: int, threshold: float) -> List[Dict]:
        """Search using local CPU FAISS"""
        print(f"[RAG Client] 🔍 CPU search: query='{query[:50]}...', k={k}, threshold={threshold}")
        try:
            if self._cpu_index is None or len(self._cpu_chunks) == 0:
                print(f"[RAG Client] ⚠️ CPU index is empty (no documents indexed)")
                logger.warning("[RAG Client] CPU index is empty")
                return []
            
            print(f"[RAG Client] 📊 CPU index: {len(self._cpu_chunks)} chunks available")
            
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
            
            print(f"[RAG Client] ✅ CPU search found {len(results)} results (threshold={threshold})")
            if results:
                for i, result in enumerate(results, 1):
                    # Extract file name from metadata
                    file_name = "unknown"
                    if isinstance(result.get('metadata'), dict):
                        file_path = result['metadata'].get('file_path', '')
                        if file_path:
                            from pathlib import Path
                            file_name = Path(file_path).name
                        else:
                            file_name = result['metadata'].get('document_name') or result['metadata'].get('guideline_name', 'unknown')
                    print(f"[RAG Client]   [{i}] Score: {result['score']:.3f}, File: {file_name}, Preview: '{result['text'][:50]}...'")
            logger.info(f"[RAG Client] CPU search found {len(results)} results (threshold={threshold})")
            return results
            
        except Exception as e:
            print(f"[RAG Client] ❌ CPU search error: {e}")
            logger.error(f"[RAG Client] CPU search error: {e}")
            return []
    
    def _rebuild_cpu_index(self):
        """Rebuild FAISS index from current chunks"""
        try:
            import faiss
            if not self._cpu_chunks or len(self._cpu_chunks) == 0:
                print("[RAG Client] ⚠️ No chunks to rebuild index")
                return
            
            print(f"[RAG Client] 🔧 Generating embeddings for {len(self._cpu_chunks)} chunks...")
            embeddings = self._embedding_model.encode(self._cpu_chunks)
            embeddings = np.array(embeddings).astype('float32')
            
            print(f"[RAG Client] 🔧 Creating FAISS index...")
            self._cpu_index = faiss.IndexFlatL2(self._embedding_dim)
            self._cpu_index.add(embeddings)
            
            print(f"[RAG Client] ✅ Rebuilt FAISS index: {self._cpu_index.ntotal} vectors")
            logger.info(f"[RAG Client] ✅ Rebuilt FAISS index: {self._cpu_index.ntotal} vectors")
        except Exception as e:
            print(f"[RAG Client] ❌ Failed to rebuild CPU index: {e}")
            logger.error(f"[RAG Client] ❌ Failed to rebuild CPU index: {e}")
            import traceback
            traceback.print_exc()
    
    def embed(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for texts
        
        Args:
            texts: List of texts to embed
        
        Returns:
            List of embedding vectors
        """
        if len(texts) > 0:
            print(f"[RAG Client] 🔤 Generating embeddings: {len(texts)} text(s), mode={'GPU' if self.use_gpu else 'CPU'}")
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
        print(f"[RAG Client] 🚀 Initializing RAG client (first call)...")
        print(f"[RAG Client] 🔧 RAG_MODE={RAG_MODE}, use_gpu={use_gpu}")
        _rag_client = RAGClient(use_gpu=use_gpu)
        print(f"[RAG Client] ✅ RAG client initialized: {_rag_client._mode}")
    return _rag_client


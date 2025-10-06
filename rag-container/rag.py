#!/usr/bin/env python3
"""
Aura RAG Module - FAISS-based retrieval for document search
Optimized for Jetson Orin NX with CPU-only operation
"""

import os
import numpy as np
import faiss
from typing import List, Dict, Any
import time

# Force CPU-only mode for Jetson compatibility
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['OMP_NUM_THREADS'] = '4'

from sentence_transformers import SentenceTransformer

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
    
    def _load_components(self):
        """Load FAISS index, chunks, and encoder model"""
        print("[RAG] 🔧 Loading RAG components...")
        
        # Load FAISS index
        try:
            if os.path.exists(self.index_path):
                print(f"[RAG] 🔧 Loading FAISS index from: {self.index_path}")
                self.index = faiss.read_index(self.index_path)
                print(f"[RAG] ✅ Loaded FAISS index: {self.index.ntotal} vectors")
                print(f"[RAG] 🔍 Index dimension: {self.index.d}")
                print(f"[RAG] 🔍 Index type: {type(self.index)}")
                print(f"[RAG] 🔍 Index is_trained: {self.index.is_trained}")
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
            self.encoder = SentenceTransformer(self.model_name, device='cpu')
            print(f"[RAG] ✅ Loaded sentence transformer: {self.model_name}")
        except Exception as e:
            print(f"[RAG] ❌ Failed to load sentence transformer: {e}")
            raise
    
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
            # Encode query
            query_embedding = self.encoder.encode(query, convert_to_numpy=True)
            
            # Ensure it's a numpy array
            if not isinstance(query_embedding, np.ndarray):
                query_embedding = np.array(query_embedding)
            
            query_embedding = query_embedding.astype(np.float32)
            
            if len(query_embedding.shape) == 1:
                query_embedding = query_embedding.reshape(1, -1)
            
            # Ensure contiguous memory layout for FAISS
            query_embedding = np.ascontiguousarray(query_embedding)
            
            # Debug embedding shape and type
            print(f"[RAG] 🔍 Query embedding shape: {query_embedding.shape}, dtype: {query_embedding.dtype}")
            
            # Search FAISS index
            print(f"[RAG] 🔍 About to search FAISS index with query shape: {query_embedding.shape}")
            print(f"[RAG] 🔍 Query embedding type: {type(query_embedding)}")
            print(f"[RAG] 🔍 Query embedding dtype: {query_embedding.dtype}")
            
            distances, indices = self.index.search(query_embedding, k)
            
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

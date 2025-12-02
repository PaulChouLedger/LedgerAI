"""
Memory RAG Client - RAG functionality for stored conversations
Uses MemoryManager's FAISS index to search and retrieve relevant past conversations
"""

import os
import sys
import numpy as np
from typing import List, Dict, Optional
import logging

# Add shared directory to path for fuzzy matching utilities
sys.path.insert(0, '/shared')
from rag.fuzzy_utils import fuzzy_match_term, extract_key_terms

logger = logging.getLogger(__name__)

# RAG Search Configuration
MEMORY_RAG_SEARCH_THRESHOLD = float(os.environ.get('MEMORY_RAG_SEARCH_THRESHOLD', '0.35'))  # Similarity threshold (0-1)
MEMORY_RAG_SEARCH_K = int(os.environ.get('MEMORY_RAG_SEARCH_K', '3'))  # Number of results to return (default: 3)

class MemoryRAGClient:
    """
    RAG client for searching stored conversations in the memory container.
    Uses the MemoryManager's FAISS index to find relevant past conversations.
    """
    
    def __init__(self, memory_manager):
        """
        Initialize Memory RAG client
        
        Args:
            memory_manager: MemoryManager instance to use for searching
        """
        self.memory_manager = memory_manager
        logger.info("[Memory RAG] Initialized Memory RAG client")
    
    def _fuzzy_match_term(self, term: str, text: str, threshold: float = 0.75) -> bool:
        """
        Check if a term fuzzy matches any word in the text.
        Uses shared fuzzy matching utility.
        """
        return fuzzy_match_term(term, text, threshold)
    
    def quick_content_match(self, query: str) -> bool:
        """
        Quick substring/fuzzy match to check if query terms appear in stored conversations.
        Uses exact substring matching first (fast), then fuzzy matching for transcription errors.
        Much faster than full semantic search - used to decide if RAG should be used.
        
        Args:
            query: Search query
        
        Returns:
            True if query terms match any stored conversation, False otherwise
        """
        if not query or not query.strip():
            return False
        
        # Extract key terms from query (remove common stop words)
        key_terms = extract_key_terms(query, min_word_length=2)
        
        if not key_terms:
            return False
        
        # Quick substring/fuzzy match against stored conversations
        if not self.memory_manager.conversations or len(self.memory_manager.conversations) == 0:
            return False
        
        # Check if any key term appears in any conversation (exact match first, then fuzzy)
        # Check up to 500 conversations to catch relevant content
        # This is still fast (substring/fuzzy match) compared to full semantic search
        conversations_to_check = min(500, len(self.memory_manager.conversations))
        exact_matches = 0
        fuzzy_matches = 0
        
        for i in range(conversations_to_check):
            conv = self.memory_manager.conversations[i]
            conv_text = conv.get('text', '').lower()
            
            # First try exact substring matching (fastest)
            exact_matching_terms = sum(1 for term in key_terms if term in conv_text)
            if exact_matching_terms >= 2:  # At least 2 key terms match exactly
                return True
            elif exact_matching_terms == 1 and exact_matches == 0:  # First single exact match
                exact_matches = 1
            
            # If no exact match, try fuzzy matching for transcription errors
            if exact_matching_terms == 0:
                fuzzy_matching_terms = sum(1 for term in key_terms if self._fuzzy_match_term(term, conv_text, threshold=0.75))
                if fuzzy_matching_terms >= 2:  # At least 2 key terms fuzzy match
                    return True
                elif fuzzy_matching_terms == 1 and fuzzy_matches == 0:  # First single fuzzy match
                    fuzzy_matches = 1
        
        # If we found at least one match (exact or fuzzy), use RAG
        return (exact_matches > 0) or (fuzzy_matches > 0)
    
    def search(self, query: str, k: int = None, threshold: float = None) -> List[Dict]:
        """
        Search for relevant stored conversations
        
        Args:
            query: Search query
            k: Number of results to return (default: MEMORY_RAG_SEARCH_K)
            threshold: Similarity threshold 0-1 (default: MEMORY_RAG_SEARCH_THRESHOLD)
        
        Returns:
            List of search results with text, score, and metadata
        """
        # Use configuration defaults if not provided
        if k is None:
            k = MEMORY_RAG_SEARCH_K
        if threshold is None:
            threshold = MEMORY_RAG_SEARCH_THRESHOLD
        
        logger.info(f"[Memory RAG] 🔍 Searching stored conversations: query='{query[:50]}...', k={k}, threshold={threshold}")
        
        # Use MemoryManager's search_similar method
        search_results = self.memory_manager.search_similar(query, k=k, threshold=threshold)
        
        # Convert to RAG format (similar to LLM container's RAG client)
        results = []
        for result in search_results:
            conv = result.get('conversation', {})
            score = result.get('score', 0.0)
            metadata = result.get('metadata', {})
            
            # Format as RAG result
            rag_result = {
                'text': conv.get('text', ''),
                'score': score,
                'metadata': {
                    'conversation_id': conv.get('id', ''),
                    'timestamp': conv.get('timestamp', 0),
                    'datetime': conv.get('datetime', ''),
                    'source': conv.get('source', 'unknown'),
                    **metadata
                }
            }
            results.append(rag_result)
        
        logger.info(f"[Memory RAG] ✅ Found {len(results)} relevant conversations (threshold: {threshold})")
        return results
    
    def embed(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for texts using MemoryManager's embedding model
        
        Args:
            texts: List of texts to embed
        
        Returns:
            List of embedding vectors
        """
        try:
            embeddings = self.memory_manager.embedding_model.encode(texts)
            return embeddings.tolist()
        except Exception as e:
            logger.error(f"[Memory RAG] ❌ Embedding error: {e}")
            return []


# Singleton instance
_memory_rag_client = None

def get_memory_rag_client(memory_manager=None) -> Optional[MemoryRAGClient]:
    """
    Get or create Memory RAG client singleton
    
    Args:
        memory_manager: MemoryManager instance (required for first call)
    
    Returns:
        MemoryRAGClient instance or None if memory_manager not provided
    """
    global _memory_rag_client
    if _memory_rag_client is None:
        if memory_manager is None:
            logger.warning("[Memory RAG] ⚠️ MemoryManager not provided, cannot initialize RAG client")
            return None
        logger.info("[Memory RAG] 🚀 Initializing Memory RAG client (first call)...")
        _memory_rag_client = MemoryRAGClient(memory_manager)
        logger.info("[Memory RAG] ✅ Memory RAG client initialized")
    return _memory_rag_client


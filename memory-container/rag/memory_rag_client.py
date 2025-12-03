"""
Memory RAG Client - RAG functionality for stored conversations
Uses MemoryManager's FAISS index to search and retrieve relevant past conversations
"""

import os
import sys
import numpy as np
from typing import List, Dict, Optional
import logging

# Import fuzzy matching utilities from shared directory
# Ensure /shared is in path (must be first to take precedence)
if '/shared' not in sys.path:
    sys.path.insert(0, '/shared')

# Verify shared directory is mounted
if not os.path.exists('/shared'):
    raise ImportError(
        "❌ /shared directory not found! "
        "Make sure docker-compose.yml mounts ../shared:/shared. "
        "Check: docker exec setup-memory-1 ls -la /shared"
    )

if not os.path.exists('/shared/rag/fuzzy_utils.py'):
    shared_contents = os.listdir('/shared') if os.path.exists('/shared') else []
    raise ImportError(
        f"❌ Cannot find /shared/rag/fuzzy_utils.py. "
        f"Shared dir exists: {os.path.exists('/shared')}, "
        f"Contents: {shared_contents}. "
        f"Make sure the shared directory is properly mounted in docker-compose.yml"
    )

# Import from shared rag package using importlib for more reliable loading
# This works even if the package structure isn't perfect
import importlib.util
fuzzy_utils_path = '/shared/rag/fuzzy_utils.py'
spec = importlib.util.spec_from_file_location("fuzzy_utils", fuzzy_utils_path)
if spec is None or spec.loader is None:
    raise ImportError(
        f"❌ Failed to load fuzzy_utils from {fuzzy_utils_path}. "
        f"File exists: {os.path.exists(fuzzy_utils_path)}"
    )
fuzzy_utils_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fuzzy_utils_module)
fuzzy_match_term = fuzzy_utils_module.fuzzy_match_term
extract_key_terms = fuzzy_utils_module.extract_key_terms

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
        
        Requires at least 2 key terms to match (exact or fuzzy) to prevent false positives.
        
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
        
        # Extract person names from query (capitalized multi-word names)
        # If query contains person names, we MUST find at least one name in conversations
        import re
        question_words = {'who', 'what', 'where', 'when', 'why', 'how', 'which', 'whose', 'whom', 'do', 'does', 'did', 'is', 'are', 'was', 'were', 'can', 'could', 'will', 'would', 'should', 'may', 'might'}
        query_names = re.findall(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b', query)
        query_names_lower = [name.lower() for name in query_names]
        
        # Also extract individual capitalized words (excluding question words)
        query_capitalized_words = re.findall(r'\b([A-Z][a-z]+)\b', query)
        query_capitalized_lower = [w.lower() for w in query_capitalized_words if w.lower() not in question_words]
        
        # If we have person names, we need to find at least one in the conversations
        has_person_name = len(query_names_lower) > 0 or len(query_capitalized_lower) >= 2
        
        # Quick substring/fuzzy match against stored conversations
        if not self.memory_manager.conversations or len(self.memory_manager.conversations) == 0:
            return False
        
        # Check if any key term appears in any conversation (exact match first, then fuzzy)
        # Check up to 500 conversations to catch relevant content
        # This is still fast (substring/fuzzy match) compared to full semantic search
        conversations_to_check = min(500, len(self.memory_manager.conversations))
        exact_matches = 0
        fuzzy_matches = 0
        name_found = False  # Track if person name was found (if query has names)
        
        for i in range(conversations_to_check):
            conv = self.memory_manager.conversations[i]
            conv_text = conv.get('text', '').lower()
            
            # If query has person names, check if at least one name appears in this conversation
            if has_person_name:
                # Check full names first (e.g., "John Smith" vs "John Smyth" - handles spelling variations)
                for name in query_names_lower:
                    name_words = name.split()
                    if len(name_words) >= 2:
                        # Check if at least 2 words of the name appear in conversation (using fuzzy matching for spelling variations)
                        # Use fuzzy matching for all name words to handle transcription/spelling errors
                        first_name_match = name_words[0] in conv_text or self._fuzzy_match_term(name_words[0], conv_text, threshold=0.75)
                        last_name_match = name_words[-1] in conv_text or self._fuzzy_match_term(name_words[-1], conv_text, threshold=0.75)
                        
                        # If both first and last name match (exact or fuzzy), we found the person
                        if first_name_match and last_name_match:
                            name_found = True
                            break
                        # Or if at least 2 words match (exact or fuzzy)
                        elif sum(1 for word in name_words if word in conv_text or self._fuzzy_match_term(word, conv_text, threshold=0.75)) >= 2:
                            name_found = True
                            break
                
                # If full name not found, check individual capitalized words with fuzzy matching
                if not name_found and query_capitalized_lower:
                    for cap_word in query_capitalized_lower:
                        # Use fuzzy matching to handle spelling variations (e.g., transcription errors)
                        word_match = cap_word in conv_text or self._fuzzy_match_term(cap_word, conv_text, threshold=0.75)
                        if word_match:
                            # Check if at least one other capitalized word also appears (to avoid false positives)
                            other_caps = [w for w in query_capitalized_lower if w != cap_word]
                            if len(other_caps) == 0:
                                name_found = True
                                break
                            else:
                                # At least one other capitalized word should also match
                                other_match = any(w in conv_text or self._fuzzy_match_term(w, conv_text, threshold=0.75) for w in other_caps)
                                if other_match:
                                    name_found = True
                                    break
            
            # Check ALL key terms with fuzzy matching (handles transcription errors and spelling variations)
            # Count both exact and fuzzy matches together
            matching_terms = 0
            for term in key_terms:
                if term in conv_text:
                    matching_terms += 1  # Exact match
                elif self._fuzzy_match_term(term, conv_text, threshold=0.75):
                    matching_terms += 1  # Fuzzy match
            
            if matching_terms >= 2:  # At least 2 key terms match (exact or fuzzy)
                # If query has person names, require name match too
                if has_person_name and not name_found:
                    continue  # Skip this conversation, no name match
                exact_matches = 2  # Found 2+ matches, definitely use RAG
                break  # Found good match, no need to continue
            elif matching_terms == 1:
                # Track single match, but continue looking for better matches
                if exact_matches == 0:
                    # If query has person names, require name match too
                    if has_person_name and not name_found:
                        continue  # Skip this conversation, no name match
                    exact_matches = 1
        
        # If query has person names but none were found, don't use RAG
        if has_person_name and not name_found:
            return False
        
        # Require at least 2 matches (exact or fuzzy) to use RAG - prevents false positives
        # This ensures RAG is only used when there's actual relevant content
        # Note: exact_matches now counts both exact and fuzzy matches combined
        if exact_matches >= 2:
            return True
        elif exact_matches == 1 and fuzzy_matches == 1:
            # One match from first pass + one from second pass = 2 total matches, use RAG
            return True
        
        # Not enough matches found - skip RAG
        return False
    
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


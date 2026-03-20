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
    
    def _rerank_results(self, query: str, results: List[Dict], top_k: int = None) -> List[Dict]:
        """
        Re-rank search results using fuzzy keyword matching and entity detection
        Same logic as document RAG to ensure consistent filtering.
        
        Improves relevance by considering query-chunk interaction, handles transcription errors,
        and prioritizes results with matching person names/entities.
        """
        if not results:
            return results
        
        import re
        
        # Extract key terms from query (non-stopwords)
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were'}
        query_terms = [w.lower() for w in re.findall(r'\b\w+\b', query.lower()) if w not in stop_words and len(w) > 2]
        
        # Extract person names/entities from query (capitalized words, 2+ words)
        query_names = re.findall(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b', query)
        query_names_lower = [name.lower() for name in query_names]
        
        # Also extract individual capitalized words (excluding question words)
        question_words = {'who', 'what', 'where', 'when', 'why', 'how', 'which', 'whose', 'whom', 'do', 'does', 'did', 'is', 'are', 'was', 'were', 'can', 'could', 'will', 'would', 'should', 'may', 'might'}
        query_capitalized_words = re.findall(r'\b([A-Z][a-z]+)\b', query)
        query_capitalized_lower = [w.lower() for w in query_capitalized_words if w.lower() not in question_words]
        
        # Handle lowercase queries (e.g., from speech-to-text: "do you know who bob carella is?")
        # If no capitalized words found, check for potential name phrases after question words
        if not query_capitalized_lower and len(query_terms) >= 2:
            query_lower = query.lower()
            words = re.findall(r'\b\w+\b', query_lower)
            
            # Look for patterns like "who is X Y" or "who X Y" where X and Y could be names
            for i in range(len(words) - 1):
                if words[i] in ['who', 'what']:
                    next_idx = i + 1
                    if next_idx < len(words) and words[next_idx] == 'is':
                        next_idx += 1
                    
                    if next_idx + 1 < len(words):
                        potential_name_words = []
                        for j in range(next_idx, min(next_idx + 3, len(words))):
                            word = words[j]
                            if word not in stop_words and word not in question_words and len(word) > 2:
                                potential_name_words.append(word)
                            else:
                                break
                        
                        if len(potential_name_words) >= 2:
                            query_capitalized_lower.extend(potential_name_words)
                            logger.info(f"[Memory RAG Pre-filter] 🔍 Detected potential names from lowercase query: {potential_name_words}")
                            break
        
        # Pre-filter: Only include chunks that have at least one query term match (fuzzy)
        # CRITICAL: For queries with names, require at least one capitalized word (name) match
        logger.info(f"[Memory RAG Pre-filter] 🔍 Starting pre-filter: {len(results)} conversations, query: '{query[:50]}...'")
        filtered_results = []
        for i, result in enumerate(results, 1):
            text = result.get('text', '').lower()
            original_text = result.get('text', '')
            semantic_score = result.get('score', 0.0)
            
            # For queries with capitalized words (names), REQUIRE name matches
            # For multi-word names (2+ words), require at least 2 matches to ensure proper name matching
            # This ensures chunks about different people are excluded (e.g., "Bob Corella" shouldn't match "Bob Smith")
            has_name_match = False
            matched_name_words = []
            if query_capitalized_lower:
                # Check ALL capitalized words (don't break early - need to verify all name parts match)
                # This is critical for fuzzy matching typos like "Corella" vs "Carella"
                for cap_word in query_capitalized_lower:
                    if self._fuzzy_match_term(cap_word, text, threshold=0.75):
                        matched_name_words.append(cap_word)
                
                # For multi-word names (2+ words), require at least 2 matches
                # This ensures both first and last name match (handles typos like "Corella" vs "Carella")
                if len(query_capitalized_lower) >= 2:
                    if len(matched_name_words) >= 2:
                        has_name_match = True
                        logger.debug(f"[Memory RAG Pre-filter] ✅ Name match: {len(matched_name_words)}/{len(query_capitalized_lower)} name words fuzzy matched: {matched_name_words}")
                    else:
                        logger.debug(f"[Memory RAG Pre-filter] ❌ Insufficient name matches: only {len(matched_name_words)}/{len(query_capitalized_lower)} words matched (expected at least 2)")
                else:
                    # Single word name - just need one match
                    has_name_match = len(matched_name_words) > 0
                    if has_name_match:
                        logger.debug(f"[Memory RAG Pre-filter] ✅ Name match: '{matched_name_words[0]}' fuzzy matched")
            
            # If query has names but chunk has no name match, exclude it
            if query_capitalized_lower and not has_name_match:
                logger.debug(f"[Memory RAG Pre-filter] ❌ EXCLUDED: Query has names {query_capitalized_lower} but conversation has no name match")
                continue
            
            # For queries without names, check for other query terms
            if not query_capitalized_lower:
                has_query_term = False
                matched_term = None
                for term in query_terms:
                    if self._fuzzy_match_term(term, text, threshold=0.75):
                        has_query_term = True
                        matched_term = term
                        break
                
                if not has_query_term:
                    logger.debug(f"[Memory RAG Pre-filter] ❌ EXCLUDED: No query term matches found (terms: {query_terms})")
                    continue
            
            filtered_results.append(result)
        
        if not filtered_results:
            logger.warning(f"[Memory RAG Pre-filter] ⚠️ All conversations filtered out - no query term matches found")
            return []
        
        logger.info(f"[Memory RAG Pre-filter] ✅ {len(filtered_results)}/{len(results)} conversations passed pre-filter")
        
        # Score each result based on keyword matches and semantic score
        reranked = []
        for result in filtered_results:
            text = result.get('text', '').lower()
            original_text = result.get('text', '')
            semantic_score = result.get('score', 0.0)
            
            # Count keyword matches using fuzzy matching
            keyword_matches = sum(1 for term in query_terms if self._fuzzy_match_term(term, text, threshold=0.75))
            keyword_score = keyword_matches / max(len(query_terms), 1) if query_terms else 0
            
            # Check for person name matches (critical for "Who is X?" queries)
            # Works for both capitalized and lowercase names in query/chunks
            name_match_boost = 0.0
            text_lower = original_text.lower()
            
            # Try to match against extracted capitalized names first (more reliable)
            if query_names:
                result_names = re.findall(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b', original_text)
                result_names_lower = [name.lower() for name in result_names]
                
                for query_name in query_names_lower:
                    query_name_words = query_name.split()
                    
                    # 1. Check for exact full name match (in extracted capitalized names)
                    if query_name in result_names_lower:
                        name_match_boost = 0.25
                        break
                    
                    # 2. Check for fuzzy full name match (in extracted capitalized names)
                    for result_name in result_names_lower:
                        if all(self._fuzzy_match_term(word, result_name, threshold=0.75) for word in query_name_words if len(word) > 2):
                            name_match_boost = 0.20
                            break
                    if name_match_boost > 0:
                        break
                    
                    # 3. Check for partial name matches using fuzzy matching on full text (handles lowercase names in chunks)
                    if len(query_name_words) >= 2:
                        matching_words = sum(1 for word in query_name_words if len(word) > 2 and self._fuzzy_match_term(word, text_lower, threshold=0.75))
                        if matching_words >= 2:
                            name_match_boost = 0.15
                            break
                        elif matching_words == 1 and len(query_name_words) == 2:
                            name_match_boost = 0.10
                            break
            
            # Also handle queries with individual capitalized words (from lowercase queries or partial names)
            # Use fuzzy matching directly on text - works for both capitalized and lowercase names in chunks
            if name_match_boost == 0.0 and query_capitalized_lower and len(query_capitalized_lower) >= 2:
                # Treat 2+ capitalized words as potential name and check if they appear in text (case-insensitive fuzzy)
                matching_words = sum(1 for word in query_capitalized_lower if len(word) > 2 and self._fuzzy_match_term(word, text_lower, threshold=0.75))
                if matching_words >= 2:
                    name_match_boost = 0.15  # Moderate boost - found multiple name words in chunk
                elif matching_words == 1 and len(query_capitalized_lower) == 2:
                    name_match_boost = 0.10  # Small boost - found one word of two-word name
            
            # Combined score: 70% semantic, 30% keyword, plus name match boost
            base_score = 0.7 * semantic_score + 0.3 * keyword_score
            combined_score = base_score + name_match_boost
            combined_score = min(1.0, combined_score)
            
            reranked.append({
                **result,
                'score': combined_score,
                'original_score': semantic_score,
                'keyword_score': keyword_score,
                'name_match_boost': name_match_boost
            })
        
        # Sort by combined score (descending)
        reranked.sort(key=lambda x: x['score'], reverse=True)
        
        # Return top_k if specified
        if top_k is not None:
            return reranked[:top_k]
        
        return reranked
    
    def search(self, query: str, k: int = None, threshold: float = None, rerank: bool = True) -> List[Dict]:
        """
        Search for relevant stored conversations with re-ranking (same logic as document RAG)
        
        Args:
            query: Search query
            k: Number of results to return (default: MEMORY_RAG_SEARCH_K)
            threshold: Similarity threshold 0-1 (default: MEMORY_RAG_SEARCH_THRESHOLD)
            rerank: Whether to apply re-ranking with pre-filtering (default: True)
        
        Returns:
            List of search results with text, score, and metadata
        """
        # Use configuration defaults if not provided
        if k is None:
            k = MEMORY_RAG_SEARCH_K
        if threshold is None:
            threshold = MEMORY_RAG_SEARCH_THRESHOLD
        
        logger.info(f"[Memory RAG] 🔍 Searching stored conversations: query='{query[:50]}...', k={k}, threshold={threshold}")
        
        # Get more candidates for re-ranking (same as document RAG)
        search_k = k * 2 if rerank else k
        search_threshold = 0.0 if rerank else threshold  # Get all candidates when re-ranking
        
        # Use MemoryManager's search_similar method
        search_results = self.memory_manager.search_similar(query, k=search_k, threshold=search_threshold)
        
        # Convert to RAG format
        results = []
        for result in search_results:
            conv = result.get('conversation', {})
            score = result.get('score', 0.0)
            metadata = result.get('metadata', {})
            
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
        
        # Re-rank results if enabled (includes pre-filtering)
        if rerank and results:
            results = self._rerank_results(query, results, top_k=k)
            # Re-apply threshold after re-ranking
            filtered_results = []
            for r in results:
                effective_threshold = threshold * 0.85 if r.get('name_match_boost', 0) > 0 else threshold
                if r['score'] >= effective_threshold:
                    filtered_results.append(r)
            results = filtered_results
        
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


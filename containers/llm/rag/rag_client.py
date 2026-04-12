"""
RAG Client - Modular RAG system with GPU/CPU fallback
Supports both external RAG container (GPU) and internal FAISS (CPU)
"""

import os
import requests
import numpy as np
from typing import List, Dict, Optional, Tuple
import logging
from difflib import SequenceMatcher
import math

logger = logging.getLogger(__name__)

# Configuration
RAG_MODE = os.environ.get('RAG_MODE', 'CPU').upper()  # GPU = RAG container, CPU = CPU FAISS
RAG_SERVICE_URL = os.environ.get('RAG_SERVICE_URL', 'http://localhost:11435')
RAG_TIMEOUT = int(os.environ.get('RAG_TIMEOUT', '10'))

# RAG Search Configuration
RAG_SEARCH_THRESHOLD = float(os.environ.get('RAG_SEARCH_THRESHOLD', '0.15'))  # Similarity threshold (0-1), lower = more results (lowered to 0.15 for better recall)
RAG_SEARCH_K = int(os.environ.get('RAG_SEARCH_K', '5'))  # Number of results to return (default: 5)

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
    
    def __init__(self, use_gpu: bool = None, base_dir: str | None = None):
        """
        Initialize RAG client

        Args:
            use_gpu: Force GPU mode (True) or CPU mode (False). If None, uses RAG_MODE env var
            base_dir: Override base data directory (default: /app/data)
        """
        self._base_dir = base_dir
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
            
            logger.info("[RAG Client] 🔧 Initializing CPU RAG system with auto-ingestion...")
            
            # Use all-distilroberta-v1 (benchmarked as best performing model)
            self._embedding_model = SentenceTransformer('all-distilroberta-v1')
            self._embedding_dim = 768
            
            # Initialize FAISS CPU index
            self._cpu_index = None
            self._cpu_chunks = []
            self._cpu_metadata = []
            
            # Initialize auto-ingestion system
            self._initialize_auto_ingestion()
            
            # Try to load pre-existing index
            self._load_cpu_index()
            
            logger.info("[RAG Client] ✅ CPU RAG system initialized with auto-ingestion")
            
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
        # Use absolute path to match auto-ingest system (/app/data/embeddings)
        # This matches the path used in cpu_faiss_auto_ingest.py
        index_path = f"{self._base_dir}/embeddings" if self._base_dir else "/app/data/embeddings"
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
                print(f"[RAG Client] ⚠️ No existing CPU index found at {index_path}")
                print(f"[RAG Client] ⚠️ Index exists: {os.path.exists(faiss_index_path)}, Metadata exists: {os.path.exists(metadata_path)}")
                logger.warning("[RAG Client] ⚠️ No existing CPU index found")
                
                # Check if auto-ingestion has chunks but no index - build index on startup
                if self._auto_ingest and hasattr(self._auto_ingest, 'chunks') and len(self._auto_ingest.chunks) > 0:
                    print(f"[RAG Client] 🔧 Found {len(self._auto_ingest.chunks)} chunks from auto-ingestion, building index on startup...")
                    self._cpu_chunks = self._auto_ingest.chunks
                    self._cpu_metadata = self._auto_ingest.metadata if hasattr(self._auto_ingest, 'metadata') else []
                    self._rebuild_cpu_index()
                    print(f"[RAG Client] ✅ Index built on startup with {len(self._cpu_chunks)} chunks")
                else:
                    # Create empty index (use IndexFlatIP to match _rebuild_cpu_index)
                    import faiss
                    self._cpu_index = faiss.IndexFlatIP(self._embedding_dim)
                
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
            self._auto_ingest = CPUFAISSAutoIngest(base_dir=self._base_dir)
            
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
    
    def _fuzzy_match_term(self, term: str, text: str, threshold: float = 0.75) -> bool:
        """
        Check if a term fuzzy matches any word in the text.
        Handles transcription errors by using fuzzy string matching.
        Also handles common name spelling variations (e.g., "Raphael" vs "Rafael").
        
        Args:
            term: Term to search for
            text: Text to search in
            threshold: Minimum similarity ratio (0.0-1.0) for a match
        
        Returns:
            True if term fuzzy matches any word in text
        """
        import re
        # Extract all words from text (3+ characters to avoid matching common words)
        text_words = re.findall(r'\b\w{3,}\b', text.lower())
        term_lower = term.lower()
        
        # First try exact match (fastest)
        if term_lower in text_words:
            return True
        
        # Handle common name spelling variations (e.g., "Raphael" vs "Rafael")
        # These are common variations that should match even if similarity is slightly below threshold
        common_name_variations = {
            'raphael': ['rafael', 'raphael'],
            'rafael': ['raphael', 'rafael'],
            'michael': ['michael', 'micheal'],
            'micheal': ['michael', 'micheal'],
            'stephen': ['stephen', 'steven'],
            'steven': ['stephen', 'steven'],
            'catherine': ['catherine', 'katherine'],
            'katherine': ['catherine', 'katherine'],
        }
        
        # Check if term is a known name variation
        if term_lower in common_name_variations:
            variations = common_name_variations[term_lower]
            for variation in variations:
                if variation in text_words:
                    return True
        
        # Then try fuzzy match for transcription errors
        for word in text_words:
            # Only fuzzy match words of similar length (avoid false positives)
            # For names (4+ chars), allow length difference of up to 2 chars
            # For shorter words, require exact length match
            max_length_diff = 2 if len(term_lower) >= 4 else 1
            if abs(len(word) - len(term_lower)) <= max_length_diff:
                similarity = SequenceMatcher(None, term_lower, word).ratio()
                if similarity >= threshold:
                    return True
        
        return False
    
    def quick_content_match(self, query: str) -> bool:
        """
        Quick substring/fuzzy match to check if query terms appear in RAG content.
        Uses exact substring matching first (fast), then fuzzy matching for transcription errors.
        Much faster than full semantic search - used to decide if RAG should be used.
        
        Args:
            query: Search query
        
        Returns:
            True if query terms match any RAG content, False otherwise
        """
        if not query or not query.strip():
            return False
        
        # Extract ONLY key terms (skip all everyday words)
        # Focus on: capitalized names, important nouns, technical terms
        import re
        
        # Comprehensive stop words list (all everyday/conversational words)
        stop_words = {
            # Articles and prepositions
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'from',
            # Pronouns
            'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them', 'your', 'my', 'his', 'her', 'its', 'our', 'their',
            # Common verbs
            'is', 'are', 'was', 'were', 'do', 'does', 'did', 'have', 'has', 'had', 'be', 'been', 'being', 'get', 'got', 'gets',
            # Question words
            'how', 'what', 'when', 'where', 'why', 'which', 'who', 'whom', 'whose',
            # Modal verbs
            'can', 'could', 'should', 'would', 'may', 'might', 'must', 'will', 'shall',
            # Conversational words (ALL query framing words)
            'tell', 'me', 'about', 'know', 'knows', 'show', 'give', 'find', 'search', 'explain', 'describe',
            'list', 'say', 'says', 'said', 'talk', 'talks', 'discuss', 'discusses', 'please', 'help',
            # Temporal/contextual words (common but not content-specific)
            'last', 'next', 'this', 'that', 'these', 'those', 'previous', 'current', 'recent', 'past', 'future',
            'today', 'yesterday', 'tomorrow', 'week', 'month', 'year', 'time', 'times'
        }
        
        # First, extract capitalized names (highest priority - these are always key terms)
        # Multi-word names: "Bob Carella", "Liam Hugill", "Paul Chou"
        capitalized_names = re.findall(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b', query)
        # Single capitalized words (likely names): "Liam", "Bob", "Paul"
        single_capitalized = re.findall(r'\b([A-Z][a-z]+)\b', query)
        # Filter out question words from single capitalized
        question_words = {'who', 'what', 'where', 'when', 'why', 'how', 'which', 'whose', 'whom'}
        single_capitalized = [w for w in single_capitalized if w.lower() not in question_words]
        
        # Combine all names (multi-word first, then single)
        key_terms = [name.lower() for name in capitalized_names] + [name.lower() for name in single_capitalized]
        
        # If we found names, use only those (names are specific enough)
        if key_terms:
            print(f"[RAG Client] 🔍 Extracted key terms (names only): {key_terms}")
        else:
            # No names found - extract other key terms (nouns, technical terms)
            # Extract words (3+ characters) that aren't stop words
            query_lower = query.lower()
            words = re.findall(r'\b\w{3,}\b', query_lower)  # Minimum 3 chars to avoid very short words
            key_terms = [w for w in words if w not in stop_words]
            # Sort by length (longest first) - prioritize specific terms
            key_terms = sorted(key_terms, key=len, reverse=True)
            print(f"[RAG Client] Extracted key terms (no names found): {key_terms}")
        
        if not key_terms:
            print(f"[RAG Client] ⚠️ No key terms extracted from query: '{query}'")
            return False
        
        # Quick substring/fuzzy match against RAG chunks
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
            # CPU mode: quick substring/fuzzy match against chunk text
            if not self._cpu_chunks or len(self._cpu_chunks) == 0:
                # Try to reload if empty
                if self._auto_ingest:
                    if self._auto_ingest.load_existing_embeddings():
                        self._cpu_chunks = self._auto_ingest.chunks
                        self._cpu_metadata = self._auto_ingest.metadata
            
            if not self._cpu_chunks or len(self._cpu_chunks) == 0:
                return False
            
            # Check if any key term appears in any chunk text (exact match first, then fuzzy)
            # For medical/technical queries, check more chunks (up to 500) to catch relevant content
            # This is still fast (substring/fuzzy match) compared to full semantic search
            chunks_to_check = min(500, len(self._cpu_chunks))
            
            # Prioritize longer/more specific terms (e.g., "vasopressin" over "tell")
            # The longest term is most likely to be the actual subject of the query
            primary_term = key_terms[0] if key_terms else None  # Longest term (already sorted)
            
            found_primary_match = False
            found_secondary_matches = 0
            
            for i in range(chunks_to_check):
                # Chunks are strings, not dictionaries
                chunk = self._cpu_chunks[i]
                if isinstance(chunk, dict):
                    chunk_text = chunk.get('text', '').lower()
                else:
                    chunk_text = str(chunk).lower()
                
                # First check if the primary (longest/most specific) term matches
                # This is the most important check - if "Liam" or "Bob Carella" isn't in the docs, don't use RAG
                # Handle both single-word terms (e.g., "liam") and multi-word terms (e.g., "bob carella")
                if primary_term:
                    # For multi-word terms (names like "bob carella"), check if all words appear in chunk
                    if ' ' in primary_term:
                        # Multi-word term: check if all words appear (in order, but allow some flexibility)
                        words = primary_term.split()
                        # Check if all words appear in chunk (allows some word order flexibility)
                        if all(word in chunk_text for word in words):
                            found_primary_match = True
                        # Also try fuzzy match for transcription errors
                        elif self._fuzzy_match_term(primary_term, chunk_text, threshold=0.75):
                            found_primary_match = True
                    else:
                        # Single-word term: simple substring or fuzzy match
                        if primary_term in chunk_text:
                            found_primary_match = True
                        elif self._fuzzy_match_term(primary_term, chunk_text, threshold=0.75):
                            found_primary_match = True
                
                # Also check other key terms (for queries with multiple specific terms)
                if len(key_terms) > 1:
                    for term in key_terms[1:]:  # Skip primary term
                        if term in chunk_text:
                            found_secondary_matches += 1
                        elif self._fuzzy_match_term(term, chunk_text, threshold=0.75):
                            found_secondary_matches += 1
                
                # If we found the primary term, we can return early (most specific match found)
                if found_primary_match:
                        return True
            
            # Only use RAG if we found the primary (most specific) term
            # This ensures queries like "Tell me about vasopressin" only trigger RAG if "vasopressin" is actually in the documents
            # Secondary matches are only used for multi-term queries where the primary term is also found
            return found_primary_match
    
    def _expand_query(self, query: str) -> str:
        """
        Expand query with basic linguistic variations to improve retrieval
        
        Generic approach: handles common word forms and variations
        """
        import re
        
        # Basic linguistic expansions (generic, not domain-specific)
        # Handle common word variations
        query_lower = query.lower()
        
        # Simple pluralization/singularization handling
        # This is generic and works for any domain
        words = query.split()
        expanded_words = []
        
        for word in words:
            word_lower = word.lower()
            # Add common variations
            if word_lower.endswith('s') and len(word_lower) > 3:
                # Try singular form
                singular = word_lower[:-1]
                if singular not in expanded_words:
                    expanded_words.append(singular)
            elif len(word_lower) > 3:
                # Try plural form
                plural = word_lower + 's'
                if plural not in expanded_words:
                    expanded_words.append(plural)
            expanded_words.append(word_lower)
        
        # Use original query for embedding (expansions handled in keyword matching)
        return query
    
    def _rerank_results(self, query: str, results: List[Dict], top_k: int = None) -> List[Dict]:
        """
        Re-rank search results using fuzzy keyword matching and entity detection
        
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
        # Pattern matches full names like "John Smith", "Jane Doe", etc.
        query_names = re.findall(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b', query)
        query_names_lower = [name.lower() for name in query_names]
        
        # Also extract individual capitalized words (first names, last names separately)
        # e.g., "John Smith" -> ["John", "Smith"]
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
                            print(f"[RAG Pre-filter] 🔍 Detected potential names from lowercase query: {potential_name_words}")
                            break
        
        # Pre-filter: Only include chunks that have at least one query term match (fuzzy)
        # This prevents irrelevant chunks from being analyzed
        # CRITICAL: For queries with names, require at least one capitalized word (name) match
        print(f"[RAG Pre-filter] 🔍 Starting pre-filter: {len(results)} chunks, query: '{query[:50]}...'")
        print(f"[RAG Pre-filter] 🔍 Query terms: {query_terms}, Capitalized words: {query_capitalized_lower}")
        filtered_results = []
        for result in results:
            text = result.get('text', '').lower()
            original_text = result.get('text', '')
            
            # DEBUG: For name queries, show what name words appear in the chunk
            if query_capitalized_lower:
                chunk_name_words = []
                for cap_word in query_capitalized_lower:
                    # Check if word appears in chunk (exact or fuzzy)
                    if cap_word in text:
                        chunk_name_words.append(f"{cap_word}(exact)")
                    elif self._fuzzy_match_term(cap_word, text, threshold=0.65):
                        chunk_name_words.append(f"{cap_word}(fuzzy)")
                if chunk_name_words:
                    print(f"[RAG Pre-filter] 🔍 Chunk contains name words: {chunk_name_words} in '{original_text[:60]}...'")
                else:
                    print(f"[RAG Pre-filter] 🔍 Chunk does NOT contain any name words from query in '{original_text[:60]}...'")
            
            # For queries with capitalized words (names), REQUIRE name matches
            # For multi-word names (2+ words), require at least 2 matches to ensure proper name matching
            # This ensures chunks about different people are excluded (e.g., "Bob Corella" shouldn't match "Bob Smith")
            has_name_match = False
            matched_name_words = []
            if query_capitalized_lower:
                # Check ALL capitalized words (don't break early - need to verify all name parts match)
                # Use lower threshold for name matching to handle spelling variations like "Raphael" vs "Rafael"
                for cap_word in query_capitalized_lower:
                    # Try with standard threshold first
                    if self._fuzzy_match_term(cap_word, text, threshold=0.75):
                        matched_name_words.append(cap_word)
                    # If no match, try with lower threshold (handles name variations)
                    elif self._fuzzy_match_term(cap_word, text, threshold=0.70):
                        matched_name_words.append(cap_word)
                        print(f"[RAG Pre-filter] 🔍 Name word matched with lower threshold: '{cap_word}'")
                
                # For multi-word names (2+ words), require at least 2 matches OR 1 strong match (last name)
                # This ensures both first and last name match (handles typos like "Corella" vs "Carella")
                # BUT also allows single strong matches (e.g., "Rafael" vs "Raphael" - last name "Cabello" matches)
                if len(query_capitalized_lower) >= 2:
                    if len(matched_name_words) >= 2:
                        has_name_match = True
                        print(f"[RAG Pre-filter] ✅ Name match: {len(matched_name_words)}/{len(query_capitalized_lower)} name words fuzzy matched: {matched_name_words} in '{original_text[:60]}...'")
                    elif len(matched_name_words) == 1:
                        # Single match: check if it's the last name (usually more distinctive) OR if it's a strong match
                        matched_word = matched_name_words[0]
                        # Last name is typically the last word in the query
                        is_last_name = matched_word == query_capitalized_lower[-1]
                        # Check if it's a strong fuzzy match (high similarity)
                        matched_word_idx = query_capitalized_lower.index(matched_word)
                        # Re-check with higher threshold to ensure strong match
                        strong_match = self._fuzzy_match_term(matched_word, text, threshold=0.85)
                        if is_last_name or strong_match:
                            has_name_match = True
                            print(f"[RAG Pre-filter] ✅ Name match (single strong): '{matched_word}' matched ({'last name' if is_last_name else 'strong match'}) in '{original_text[:60]}...'")
                        else:
                            # For last name, be more lenient - if it matches at all, allow it
                            if is_last_name:
                                has_name_match = True
                                print(f"[RAG Pre-filter] ✅ Name match (last name lenient): '{matched_word}' matched in '{original_text[:60]}...'")
                            else:
                                print(f"[RAG Pre-filter] ❌ Insufficient name matches: only 1/{len(query_capitalized_lower)} words matched (weak match) in '{original_text[:60]}...'")
                    else:
                        # No matches at 0.75 or 0.70 - try even lower threshold for name variations
                        # This handles cases like "Raphael" vs "Rafael" where similarity might be lower
                        for cap_word in query_capitalized_lower:
                            if self._fuzzy_match_term(cap_word, text, threshold=0.65):  # Very lenient for name variations
                                matched_name_words.append(cap_word)
                                print(f"[RAG Pre-filter] 🔍 Name word matched with very low threshold: '{cap_word}'")
                        
                        if len(matched_name_words) >= 1:
                            # If we found at least one match (even with low threshold), check if it's the last name
                            matched_word = matched_name_words[0]
                            is_last_name = matched_word == query_capitalized_lower[-1]
                            if is_last_name or len(matched_name_words) >= 2:
                                has_name_match = True
                                print(f"[RAG Pre-filter] ✅ Name match (lenient threshold): {len(matched_name_words)}/{len(query_capitalized_lower)} name words matched: {matched_name_words} in '{original_text[:60]}...'")
                            else:
                                print(f"[RAG Pre-filter] ❌ Insufficient name matches: only {len(matched_name_words)}/{len(query_capitalized_lower)} words matched (expected at least 1) in '{original_text[:60]}...'")
                        else:
                            print(f"[RAG Pre-filter] ❌ Insufficient name matches: only {len(matched_name_words)}/{len(query_capitalized_lower)} words matched (expected at least 1) in '{original_text[:60]}...'")
                else:
                    # Single word name - use lower threshold for fuzzy matching (handles spelling variations like "Raphael" vs "Rafael")
                    # For single-word names, we're more lenient since there's no last name to confirm
                    if len(matched_name_words) > 0:
                        has_name_match = True
                        print(f"[RAG Pre-filter] ✅ Name match: '{matched_name_words[0]}' fuzzy matched in '{original_text[:60]}...'")
                    else:
                        # Try with lower threshold for single-word names (handles common spelling variations)
                        for cap_word in query_capitalized_lower:
                            if self._fuzzy_match_term(cap_word, text, threshold=0.70):  # Lower threshold for single-word names
                                has_name_match = True
                                matched_name_words.append(cap_word)
                                print(f"[RAG Pre-filter] ✅ Name match (lower threshold): '{cap_word}' fuzzy matched in '{original_text[:60]}...'")
                                break
                        if not has_name_match:
                            print(f"[RAG Pre-filter] ❌ No name match found for '{query_capitalized_lower[0]}' in '{original_text[:60]}...'")
            
            # If query has names but chunk has no name match, exclude it
            if query_capitalized_lower and not has_name_match:
                print(f"[RAG Pre-filter] ❌ Excluded (query has names but chunk has no name match): '{original_text[:60]}...'")
                continue

            # Require a minimum number of query term overlaps (handles general relevance)
            match_count = 0
            for term in query_terms:
                if self._fuzzy_match_term(term, text, threshold=0.75):
                    match_count += 1
            if query_terms:
                required_matches = min(
                    len(query_terms),
                    max(2, math.ceil(len(query_terms) * 0.35))
                )
            else:
                required_matches = 0

            if match_count < required_matches:
                print(f"[RAG Pre-filter] ❌ Excluded (insufficient term overlap: {match_count}/{required_matches}) '{original_text[:60]}...'")
                continue
            else:
                print(f"[RAG Pre-filter] ✅ Term overlap: {match_count}/{required_matches} matches for '{original_text[:60]}...'")
            
            filtered_results.append(result)
        
        if not filtered_results:
            print(f"[RAG Pre-filter] ⚠️ All chunks filtered out - no query term matches found")
            logger.warning(f"[RAG Pre-filter] ⚠️ All chunks filtered out - no query term matches found")
            return []
        
        print(f"[RAG Pre-filter] ✅ {len(filtered_results)}/{len(results)} chunks passed pre-filter")
        logger.info(f"[RAG Pre-filter] ✅ {len(filtered_results)}/{len(results)} chunks passed pre-filter")
        
        # Score each result based on keyword matches and semantic score
        reranked = []
        for result in filtered_results:
            text = result.get('text', '').lower()
            original_text = result.get('text', '')  # Keep original for name extraction
            semantic_score = result.get('score', 0.0)
            
            # Count keyword matches using fuzzy matching (handles transcription errors)
            keyword_matches = sum(1 for term in query_terms if self._fuzzy_match_term(term, text, threshold=0.75))
            keyword_score = keyword_matches / max(len(query_terms), 1) if query_terms else 0
            
            # Check for person name matches (critical for "Who is X?" queries)
            # Works for both capitalized and lowercase names in query/chunks
            name_match_boost = 0.0
            text_lower = original_text.lower()
            
            # Try to match against extracted capitalized names first (more reliable)
            if query_names:
                # Extract names from result text (full names) - capitalized only
                result_names = re.findall(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b', original_text)
                result_names_lower = [name.lower() for name in result_names]
                
                # Check if any query name matches any result name (exact or fuzzy)
                for query_name in query_names_lower:
                    query_name_words = query_name.split()
                    
                    # 1. Check for exact full name match (in extracted capitalized names)
                    if query_name in result_names_lower:
                        name_match_boost = 0.25  # Strong boost for exact name match
                        break
                    
                    # 2. Check for fuzzy full name match (in extracted capitalized names)
                    for result_name in result_names_lower:
                        # Check if all query name words appear in result name (fuzzy)
                        if all(self._fuzzy_match_term(word, result_name, threshold=0.75) for word in query_name_words if len(word) > 2):
                            name_match_boost = 0.20  # Boost for fuzzy name match
                            break
                    if name_match_boost > 0:
                        break
                    
                    # 3. Check for partial name matches using fuzzy matching on full text (handles lowercase names in chunks)
                    # This handles cases where names might be lowercase in the chunk
                    if len(query_name_words) >= 2:
                        matching_words = sum(1 for word in query_name_words if len(word) > 2 and self._fuzzy_match_term(word, text_lower, threshold=0.75))
                        if matching_words >= 2:
                            name_match_boost = 0.15  # Moderate boost for partial name match
                            break
                        elif matching_words == 1 and len(query_name_words) == 2:
                            name_match_boost = 0.10  # Small boost for single word match in 2-word name
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
            combined_score = min(1.0, combined_score)  # Cap at 1.0
            
            reranked.append({
                **result,
                'score': combined_score,
                'original_score': semantic_score,
                'keyword_score': keyword_score,
                'name_match_boost': name_match_boost
            })
            
            # Debug logging for re-ranking (only for first few results to avoid spam)
            if len(reranked) <= 3:
                logger.debug(f"[RAG Re-rank] Score: {combined_score:.3f} (semantic: {semantic_score:.3f}, keyword: {keyword_score:.3f}, name_boost: {name_match_boost:.3f}) - '{original_text[:60]}...'")
        
        # Sort by combined score (descending)
        reranked.sort(key=lambda x: x['score'], reverse=True)
        
        # Return top_k if specified
        if top_k is not None:
            return reranked[:top_k]
        
        return reranked
    
    def search(self, query: str, k: int = None, threshold: float = None, rerank: bool = True) -> List[Dict]:
        """
        Search for relevant information with improved retrieval
        
        Args:
            query: Search query
            k: Number of results to return (default: RAG_SEARCH_K)
            threshold: Similarity threshold 0-1 (default: RAG_SEARCH_THRESHOLD)
            rerank: Whether to re-rank results for better relevance (default: True)
        
        Returns:
            List of search results with text, score, and metadata
        """
        # Use configuration defaults if not provided
        if k is None:
            k = RAG_SEARCH_K
        if threshold is None:
            threshold = RAG_SEARCH_THRESHOLD
        
        # Expand query for better retrieval
        expanded_query = self._expand_query(query)
        
        # Detect if query contains names (for name queries, search more candidates)
        import re
        query_names = re.findall(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b', query)
        query_capitalized_words = re.findall(r'\b([A-Z][a-z]+)\b', query)
        question_words = {'who', 'what', 'where', 'when', 'why', 'how', 'which', 'whose', 'whom', 'do', 'does', 'did', 'is', 'are', 'was', 'were', 'can', 'could', 'will', 'would', 'should', 'may', 'might'}
        query_capitalized_lower = [w.lower() for w in query_capitalized_words if w.lower() not in question_words]
        has_name_query = len(query_names) > 0 or (len(query_capitalized_lower) >= 1 and any(len(w) > 3 for w in query_capitalized_lower))
        
        # For name queries, search many more candidates and use very low threshold
        # This ensures we find chunks that mention the person even if they're not semantically similar
        if has_name_query:
            search_k = max(k * 10, 100)  # Search 10x more candidates for name queries (or at least 100)
            search_threshold = 0.0  # No threshold for name queries - we'll filter by name matching instead
            print(f"[RAG Client] 🔍 Name query detected - searching {search_k} candidates with threshold={search_threshold}")
        else:
            search_k = k * 2  # Normal queries: 2x candidates
            search_threshold = threshold * 0.8  # Slightly lower threshold for re-ranking
        
        # Search with expanded query (but use original for embedding)
        if self.use_gpu:
            results = self._search_gpu(expanded_query, search_k, search_threshold)
        else:
            results = self._search_cpu(expanded_query, search_k, search_threshold)
        
        # Re-rank results if enabled (includes pre-filtering)
        if rerank and results:
            print(f"[RAG Client] 🔍 Pre-filtering and re-ranking {len(results)} results for query: '{query[:50]}...'")
            results = self._rerank_results(query, results, top_k=k)
            print(f"[RAG Client] ✅ After pre-filter and re-rank: {len(results)} results remaining")
            # Re-apply threshold after re-ranking
            # Use slightly lower threshold for chunks with name matches (they're more likely to be relevant)
            filtered_results = []
            for r in results:
                score = r.get('score', 0)
                name_boost = r.get('name_match_boost', 0)
                # If chunk has name match boost, use 15% lower threshold (e.g., 0.255 instead of 0.30)
                # This helps include relevant chunks that mention the person but don't have perfect semantic match
                effective_threshold = threshold * 0.85 if name_boost > 0 else threshold
                if score >= effective_threshold:
                    filtered_results.append(r)
                else:
                    logger.debug(f"[RAG] Filtered out chunk (score: {score:.3f} < threshold: {effective_threshold:.3f}, name_boost: {name_boost:.3f})")
            results = filtered_results
        
        return results[:k]  # Return top k results
    
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
            # Try to reload index if it might be empty but files exist
            if (self._cpu_index is None or len(self._cpu_chunks) == 0) and self._auto_ingest:
                print("[RAG Client] 🔄 Index appears empty, attempting to reload...")
                self._load_cpu_index()
                # Also try reloading from auto-ingest
                if self._auto_ingest.load_existing_embeddings():
                    self._cpu_chunks = self._auto_ingest.chunks
                    self._cpu_metadata = self._auto_ingest.metadata
                    if self._cpu_chunks and len(self._cpu_chunks) > 0:
                        self._rebuild_cpu_index()
            
            if self._cpu_index is None or len(self._cpu_chunks) == 0:
                print(f"[RAG Client] ⚠️ CPU index is empty (no documents indexed)")
                logger.warning("[RAG Client] CPU index is empty")
                return []
            
            print(f"[RAG Client] 📊 CPU index: {len(self._cpu_chunks)} chunks available")
            
            # Generate query embedding
            query_embedding = self._embedding_model.encode([query])[0]
            query_embedding = np.array([query_embedding]).astype('float32')
            
            # Detect index type and handle accordingly
            # IndexFlatIP returns similarity scores (higher = more similar)
            # IndexFlatL2 returns distances (lower = more similar)
            index_type = type(self._cpu_index).__name__
            is_inner_product = 'IP' in index_type or 'InnerProduct' in index_type
            
            if is_inner_product:
                # For IndexFlatIP: normalize query embedding for cosine similarity
                import faiss
                query_embedding = query_embedding.reshape(1, -1)
                faiss.normalize_L2(query_embedding)
                # Search returns similarity scores (higher = more similar)
            search_results, indices = self._cpu_index.search(query_embedding.reshape(1, -1), k)
            
            # Convert to similarity scores
            if is_inner_product:
                # IndexFlatIP: results are already similarity scores (inner product = cosine similarity when normalized)
                # Values range from -1 to 1, but typically 0 to 1 for normalized embeddings
                scores = search_results[0]
            else:
                # IndexFlatL2: results are distances, convert to similarity
                # Lower distance = higher similarity
                scores = 1 / (1 + search_results[0])
            
            # Filter by threshold and build results
            # First, collect ALL matches (including below threshold) for debugging
            all_matches = []
            for idx, score in zip(indices[0], scores):
                if idx < len(self._cpu_chunks):
                    all_matches.append({
                        'text': self._cpu_chunks[idx],
                        'score': float(score),
                        'metadata': self._cpu_metadata[idx] if idx < len(self._cpu_metadata) else {}
                    })
            
            # Show all matches for debugging (even below threshold)
            print(f"[RAG Client] 🔍 DEBUG: All {len(all_matches)} matches (showing top {min(k, len(all_matches))}):")
            for i, match in enumerate(all_matches[:k], 1):
                # Extract file name from metadata
                file_name = "unknown"
                if isinstance(match.get('metadata'), dict):
                    file_path = match['metadata'].get('file_path', '')
                    if file_path:
                        from pathlib import Path
                        file_name = Path(file_path).name
                    else:
                        file_name = match['metadata'].get('document_name', 'unknown')
                threshold_status = "✅" if match['score'] >= threshold else "❌"
                print(f"[RAG Client]   [{i}] {threshold_status} Score: {match['score']:.3f} (threshold: {threshold:.3f}), File: {file_name}, Preview: '{match['text'][:50]}...'")
            
            # Filter by threshold
            results = [match for match in all_matches if match['score'] >= threshold]
            
            print(f"[RAG Client] ✅ CPU search found {len(results)} results above threshold={threshold} (out of {len(all_matches)} total matches)")
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
                            file_name = result['metadata'].get('document_name', 'unknown')
                    print(f"[RAG Client]   [{i}] Score: {result['score']:.3f}, File: {file_name}, Preview: '{result['text'][:50]}...'")
            logger.info(f"[RAG Client] CPU search found {len(results)} results (threshold={threshold})")
            return results
            
        except Exception as e:
            print(f"[RAG Client] ❌ CPU search error: {e}")
            logger.error(f"[RAG Client] ❌ CPU search error: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def _rebuild_cpu_index(self):
        """Rebuild FAISS index from current chunks (triggered by auto-ingest on document upload)"""
        try:
            import faiss
            if not self._cpu_chunks or len(self._cpu_chunks) == 0:
                print("[RAG Client] ⚠️ No chunks to rebuild index")
                return
            
            print(f"[RAG Client] 🔧 Generating embeddings for {len(self._cpu_chunks)} chunks...")
            embeddings = self._embedding_model.encode(self._cpu_chunks)
            embeddings = np.array(embeddings).astype('float32')
            
            # Normalize embeddings for cosine similarity (required for IndexFlatIP)
            faiss.normalize_L2(embeddings)
            print(f"[RAG Client] ✅ Normalized embeddings for cosine similarity")
            
            print(f"[RAG Client] 🔧 Creating FAISS index (IndexFlatIP for cosine similarity)...")
            self._cpu_index = faiss.IndexFlatIP(self._embedding_dim)
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

def get_rag_client(use_gpu: bool = None, base_dir: str | None = None) -> RAGClient:
    """Get or create RAG client singleton"""
    global _rag_client
    if _rag_client is None:
        print(f"[RAG Client] Initializing RAG client (first call)...")
        print(f"[RAG Client] RAG_MODE={RAG_MODE}, use_gpu={use_gpu}, base_dir={base_dir}")
        _rag_client = RAGClient(use_gpu=use_gpu, base_dir=base_dir)
        print(f"[RAG Client] RAG client initialized: {_rag_client._mode}")
    return _rag_client


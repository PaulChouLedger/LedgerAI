"""
RAG Client - Modular RAG system with GPU/CPU fallback
Supports both external RAG container (GPU) and internal FAISS (CPU)
"""

import os
import requests
import numpy as np
from typing import List, Dict, Optional, Tuple
import logging
from .fuzzy_utils import fuzzy_match_term, extract_key_terms

logger = logging.getLogger(__name__)

# Configuration
RAG_MODE = os.environ.get('RAG_MODE', 'CPU').upper()  # GPU = RAG container, CPU = CPU FAISS
RAG_SERVICE_URL = os.environ.get('RAG_SERVICE_URL', 'http://localhost:11435')
RAG_TIMEOUT = int(os.environ.get('RAG_TIMEOUT', '10'))

# RAG Search Configuration
RAG_SEARCH_THRESHOLD = float(os.environ.get('RAG_SEARCH_THRESHOLD', '0.10'))  # Similarity threshold — low to ensure company queries always hit RAG
RAG_SEARCH_K = int(os.environ.get('RAG_SEARCH_K', '3'))  # Number of results to return (default: 3)

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
    
    # CODE VERSION: v2.0 - Updated threshold=0.0 handling to return ALL chunks
    _CODE_VERSION = "v2.0"
    
    def __init__(self, use_gpu: bool = None):
        """
        Initialize RAG client
        
        Args:
            use_gpu: Force GPU mode (True) or CPU mode (False). If None, uses RAG_MODE env var
        """
        self.use_gpu = (RAG_MODE == 'GPU') if use_gpu is None else use_gpu
        self._cpu_rag = None
        self._mode = "GPU (External RAG Container - HTTP API)" if self.use_gpu else "CPU (Local FAISS - In-Process)"
        
        print(f"[RAG Client] ✅ CODE VERSION {self._CODE_VERSION} LOADED - threshold=0.0 returns ALL chunks")
        logger.info(f"[RAG Client] Initialized in {self._mode} mode (Code version: {self._CODE_VERSION})")
        
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
            # Force CPU to avoid CUDA conflicts with llama-cpp on Jetson
            self._embedding_model = SentenceTransformer('all-distilroberta-v1', device='cpu')
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
        index_path = "/app/data/embeddings"
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
    
    def _fuzzy_match_term(self, term: str, text: str, threshold: float = 0.75) -> bool:
        """
        Check if a term fuzzy matches any word in the text.
        Uses shared fuzzy matching utility.
        """
        return fuzzy_match_term(term, text, threshold)
    
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
        
        # Extract key terms from query (remove common stop words)
        key_terms = extract_key_terms(query, min_word_length=2)
        
        if not key_terms:
            return False
        
        # Extract person names from query (capitalized multi-word names)
        # If query contains person names, we MUST find at least one name in the document
        import re
        question_words = {'who', 'what', 'where', 'when', 'why', 'how', 'which', 'whose', 'whom', 'do', 'does', 'did', 'is', 'are', 'was', 'were', 'can', 'could', 'will', 'would', 'should', 'may', 'might'}
        query_names = re.findall(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b', query)
        query_names_lower = [name.lower() for name in query_names]
        
        # Also extract individual capitalized words (excluding question words)
        query_capitalized_words = re.findall(r'\b([A-Z][a-z]+)\b', query)
        query_capitalized_lower = [w.lower() for w in query_capitalized_words if w.lower() not in question_words]
        
        # Handle lowercase queries (e.g., from speech-to-text)
        # If no capitalized words found, check for potential name phrases after question words
        if not query_capitalized_lower and len(key_terms) >= 2:
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
                            # Use same stop_words set from extract_key_terms
                            stop_words_set = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were'}
                            if word not in stop_words_set and word not in question_words and len(word) > 2:
                                potential_name_words.append(word)
                            else:
                                break
                        
                        if len(potential_name_words) >= 2:
                            query_capitalized_lower.extend(potential_name_words)
                            break
        
        # If we have person names, we need to find at least one in the document
        has_person_name = len(query_names_lower) > 0 or len(query_capitalized_lower) >= 2
        
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
            exact_matches = 0
            fuzzy_matches = 0
            name_found = False  # Track if person name was found (if query has names)
            
            for i in range(chunks_to_check):
                # Chunks are strings, not dictionaries
                chunk = self._cpu_chunks[i]
                if isinstance(chunk, dict):
                    chunk_text = chunk.get('text', '').lower()
                else:
                    chunk_text = str(chunk).lower()
                
                # If query has person names, check if at least one name appears in this chunk
                if has_person_name:
                    # Check full names first (e.g., "John Smith" vs "John Smyth" - handles spelling variations)
                    for name in query_names_lower:
                        name_words = name.split()
                        if len(name_words) >= 2:
                            # Check if at least 2 words of the name appear in chunk (using fuzzy matching for spelling variations)
                            # Use fuzzy matching for all name words to handle transcription/spelling errors
                            first_name_match = name_words[0] in chunk_text or fuzzy_match_term(name_words[0], chunk_text, threshold=0.75)
                            last_name_match = name_words[-1] in chunk_text or fuzzy_match_term(name_words[-1], chunk_text, threshold=0.75)
                            
                            # If both first and last name match (exact or fuzzy), we found the person
                            if first_name_match and last_name_match:
                                name_found = True
                                break
                            # Or if at least 2 words match (exact or fuzzy)
                            elif sum(1 for word in name_words if word in chunk_text or fuzzy_match_term(word, chunk_text, threshold=0.75)) >= 2:
                                name_found = True
                                break
                    
                    # If full name not found, check individual capitalized words with fuzzy matching
                    if not name_found and query_capitalized_lower:
                        for cap_word in query_capitalized_lower:
                            # Use fuzzy matching to handle spelling variations (e.g., transcription errors)
                            word_match = cap_word in chunk_text or fuzzy_match_term(cap_word, chunk_text, threshold=0.75)
                            if word_match:
                                # Check if at least one other capitalized word also appears (to avoid false positives)
                                other_caps = [w for w in query_capitalized_lower if w != cap_word]
                                if len(other_caps) == 0:
                                    name_found = True
                                    break
                                else:
                                    # At least one other capitalized word should also match
                                    other_match = any(w in chunk_text or fuzzy_match_term(w, chunk_text, threshold=0.75) for w in other_caps)
                                    if other_match:
                                        name_found = True
                                        break
                
                # Check ALL key terms with fuzzy matching (handles transcription errors and spelling variations)
                # Count both exact and fuzzy matches together
                matching_terms = 0
                for term in key_terms:
                    if term in chunk_text:
                        matching_terms += 1  # Exact match
                    elif fuzzy_match_term(term, chunk_text, threshold=0.75):
                        matching_terms += 1  # Fuzzy match
                
                if matching_terms >= 2:  # At least 2 key terms match (exact or fuzzy)
                    # If query has person names, require name match too
                    if has_person_name and not name_found:
                        continue  # Skip this chunk, no name match
                    # Found 2+ matches (exact or fuzzy), definitely use RAG
                    exact_matches = 2
                    fuzzy_matches = 0  # Reset fuzzy_matches since we're using combined count
                    break  # Found good match, no need to continue
                elif matching_terms == 1:
                    # Track single match, but continue looking for better matches
                    if exact_matches == 0:
                        # If query has person names, require name match too
                        if has_person_name and not name_found:
                            continue  # Skip this chunk, no name match
                        exact_matches = 1
                        fuzzy_matches = 0  # Reset fuzzy_matches since we're using combined count
            
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
    
    def _fuzzy_match_term(self, term: str, text: str, threshold: float = 0.75) -> bool:
        """
        Check if a term fuzzy matches any word in the text.
        Handles transcription errors by using fuzzy string matching.
        
        Args:
            term: Term to search for
            text: Text to search in
            threshold: Minimum similarity ratio (0.0-1.0) for a match
        
        Returns:
            True if term fuzzy matches any word in text
        """
        # Use the imported fuzzy_match_term function
        return fuzzy_match_term(term, text, threshold)
    
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
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were',
                      # 2026-08-18: question words and pronouns were missing here,
                      # so 'how are you doing?' kept every chunk containing 'how'
                      # and fed board-meeting notes in as context. The correct list
                      # already existed 20 lines below as question_words, used only
                      # for name extraction.
                      'who', 'what', 'where', 'when', 'why', 'how', 'which', 'whose', 'whom',
                      'do', 'does', 'did', 'can', 'could', 'will', 'would', 'should',
                      'you', 'your', 'yours', 'me', 'my', 'mine', 'we', 'our', 'us',
                      'it', 'its', 'that', 'this', 'these', 'those', 'they', 'them',
                      'be', 'been', 'being', 'have', 'has', 'had', 'am', 'doing',
                      'going', 'just', 'about', 'there', 'here', 'from', 'not', 'got'}
        query_terms = [w.lower() for w in re.findall(r'\b\w+\b', query.lower()) if w not in stop_words and len(w) > 2]
        
        # Extract person names/entities from query (capitalized words, 2+ words)
        # Pattern matches full names like "John Smith", "Jane Doe", etc.
        query_names = re.findall(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b', query)
        query_names_lower = [name.lower() for name in query_names]
        
        # Also extract individual capitalized words (first names, last names separately)
        # e.g., "John Smith" -> ["John", "Smith"]
        # EXCLUDE question words at start of query (Who, What, Where, etc.) and common verbs/action words
        question_words = {'who', 'what', 'where', 'when', 'why', 'how', 'which', 'whose', 'whom', 'do', 'does', 'did', 'is', 'are', 'was', 'were', 'can', 'could', 'will', 'would', 'should', 'may', 'might'}
        # Exclude common action verbs/imperatives that are often capitalized at sentence start
        action_words = {'summarize', 'tell', 'list', 'show', 'explain', 'describe', 'give', 'find', 'search', 'provide', 'identify', 'name', 'count'}
        query_capitalized_words = re.findall(r'\b([A-Z][a-z]+)\b', query)
        query_capitalized_lower = [w.lower() for w in query_capitalized_words if w.lower() not in question_words and w.lower() not in action_words]
        
        # Handle lowercase queries (e.g., from speech-to-text: "do you know who john doe is?")
        # If no capitalized words found, check for potential name phrases after question words
        if not query_capitalized_lower and len(query_terms) >= 2:
            query_lower = query.lower()
            words = re.findall(r'\b\w+\b', query_lower)
            
            # Look for patterns like "who is X Y" or "who X Y" where X and Y could be names
            # Find words after "who", "who is", "tell me about", etc.
            for i in range(len(words) - 1):
                # Pattern: "who [is] X Y" or "tell me about X Y"
                if words[i] in ['who', 'what']:
                    # Get next 1-3 words after "who" or "who is" as potential names
                    next_idx = i + 1
                    if next_idx < len(words) and words[next_idx] == 'is':
                        next_idx += 1
                    
                    # Extract 2-3 words that could be a name
                    if next_idx + 1 < len(words):
                        potential_name_words = []
                        for j in range(next_idx, min(next_idx + 3, len(words))):
                            word = words[j]
                            # Skip stop words and question words
                            if word not in stop_words and word not in question_words and len(word) > 2:
                                potential_name_words.append(word)
                            else:
                                break
                        
                        # If we found 2+ potential name words, treat them as names
                        if len(potential_name_words) >= 2:
                            query_capitalized_lower.extend(potential_name_words)
                            print(f"[RAG Pre-filter] 🔍 Detected potential names from lowercase query: {potential_name_words}")
                            break
        
        # Pre-filter: Only include chunks that have at least one query term match (fuzzy)
        # This prevents irrelevant chunks from being analyzed
        # CRITICAL: For queries with names, require at least one capitalized word (name) match
        print(f"[RAG Pre-filter] 🔍 Starting pre-filter: {len(results)} chunks, query: '{query[:50]}...'")
        print(f"[RAG Pre-filter] 🔍 Query terms: {query_terms}, Capitalized words (names): {query_capitalized_lower}")
        filtered_results = []
        for i, result in enumerate(results, 1):
            text = result.get('text', '').lower()
            original_text = result.get('text', '')
            semantic_score = result.get('score', 0.0)
            
            # Show chunk preview for debugging (reduced verbosity)
            print(f"[RAG Pre-filter] 📄 Chunk {i}/{len(results)} (score: {semantic_score:.3f}): '{original_text[:100]}...'")
            
            # For queries with capitalized words (names), REQUIRE name matches
            # For multi-word names (2+ words), require at least 2 matches to ensure proper name matching
            # This ensures chunks about different people are excluded (e.g., "Bob Corella" shouldn't match "Bob Smith")
            has_name_match = False
            matched_name_words = []
            if query_capitalized_lower:
                # Check ALL capitalized words (don't break early - need to verify all name parts match)
                # This is critical for fuzzy matching typos like "Doe" vs "Doe"
                for cap_word in query_capitalized_lower:
                    if self._fuzzy_match_term(cap_word, text, threshold=0.75):
                        matched_name_words.append(cap_word)
                
                # For multi-word names (2+ words), require at least 2 matches
                # This ensures both first and last name match (handles typos in names)
                if len(query_capitalized_lower) >= 2:
                    if len(matched_name_words) >= 2:
                        has_name_match = True
                        print(f"[RAG Pre-filter] ✅ Name match: {len(matched_name_words)}/{len(query_capitalized_lower)} name words fuzzy matched: {matched_name_words}")
                    else:
                        print(f"[RAG Pre-filter] ❌ Insufficient name matches: only {len(matched_name_words)}/{len(query_capitalized_lower)} words matched (expected at least 2)")
                else:
                    # Single word name - just need one match
                    has_name_match = len(matched_name_words) > 0
                    if has_name_match:
                        print(f"[RAG Pre-filter] ✅ Name match: '{matched_name_words[0]}' fuzzy matched in chunk text")
            
            # Check for query term matches (always check, regardless of names)
            has_query_term = False
            matched_term = None
            for term in query_terms:
                if self._fuzzy_match_term(term, text, threshold=0.75):
                    has_query_term = True
                    matched_term = term
                    print(f"[RAG Pre-filter] ✅ Query term match: '{term}' found in chunk text")
                    break
            
            # If query has names (2+ capitalized words = likely a real name), require name match
            # Single capitalized word might be a name OR just a capitalized word at sentence start
            # For 2+ capitalized words, we're confident it's a name, so require name matching
            if len(query_capitalized_lower) >= 2:
                # Multi-word name: require name match to prevent false positives
                if not has_name_match:
                    print(f"[RAG Pre-filter] ❌ EXCLUDED: Query has multi-word name {query_capitalized_lower} but chunk has no name match")
                    continue
            elif len(query_capitalized_lower) == 1:
                # Single capitalized word: could be a name OR just sentence-start capitalization
                # Require either name match OR query term match
                if not has_name_match and not has_query_term:
                    print(f"[RAG Pre-filter] ❌ EXCLUDED: Query has capitalized word {query_capitalized_lower} but chunk has no name or query term match")
                    continue
            else:
                # No capitalized words: require query term match
                if not has_query_term:
                    print(f"[RAG Pre-filter] ❌ EXCLUDED: No query term matches found (terms: {query_terms})")
                    continue
            
            # Show what matched for debugging
            match_info = ', '.join(matched_name_words) if matched_name_words else matched_term
            print(f"[RAG Pre-filter] ✅ INCLUDED: Chunk passed pre-filter (matched: {match_info})")
            filtered_results.append(result)
        
        if not filtered_results:
            # Fallback: return top 3 by semantic score rather than nothing.
            # The embedding model already scored these as relevant — the pre-filter
            # may have been too strict (e.g. name typos from STT).
            print(f"[RAG Pre-filter] ⚠️ All chunks filtered out — falling back to top 3 by semantic score")
            logger.warning(f"[RAG Pre-filter] ⚠️ All chunks filtered out — falling back to top 3 by semantic score")
            filtered_results = results[:3]
        
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
        
        # Search with expanded query (but use original for embedding)
        # Use a lowered threshold to get more candidates for re-ranking, but not 0.0
        search_threshold = threshold * 0.5 if rerank else threshold
        # Use k*2 for initial search (pre-filter will narrow it down)
        search_k = k * 2
        print(f"[RAG Client] ✅ CODE VERSION v2.0: Updated search() - search_threshold={search_threshold}, search_k={search_k}")
        if self.use_gpu:
            results = self._search_gpu(expanded_query, search_k, search_threshold)
        else:
            results = self._search_cpu(expanded_query, search_k, search_threshold)
        
        # Re-rank results if enabled (includes pre-filtering)
        if rerank and results:
            print(f"[RAG Client] 🔍 Pre-filtering and re-ranking {len(results)} results for query: '{query[:50]}...'")
            results = self._rerank_results(query, results, top_k=k)
            # Re-apply threshold after re-ranking
            filtered_results = []
            for r in results:
                effective_threshold = threshold * 0.85 if r.get('name_match_boost', 0) > 0 else threshold
                if r['score'] >= effective_threshold:
                    filtered_results.append(r)
            results = filtered_results
            print(f"[RAG Client] ✅ Filtered to {len(results)} results above threshold={threshold}")
        
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
                query_embedding = query_embedding.flatten()
            
            # Search FAISS index
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
        """Rebuild FAISS index from current chunks"""
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

def get_rag_client(use_gpu: bool = None) -> RAGClient:
    """Get or create RAG client singleton"""
    global _rag_client
    if _rag_client is None:
        print(f"[RAG Client] 🚀 Initializing RAG client (first call)...")
        print(f"[RAG Client] 🔧 RAG_MODE={RAG_MODE}, use_gpu={use_gpu}")
        _rag_client = RAGClient(use_gpu=use_gpu)
        print(f"[RAG Client] ✅ RAG client initialized: {_rag_client._mode}")
    return _rag_client


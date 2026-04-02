#!/usr/bin/env python3
"""
Memory Manager - Handles vectorization and storage of conversations
"""

import os
import json
import time
import pickle
import hashlib
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from datetime import datetime
import threading
import logging

logger = logging.getLogger(__name__)
# Set up logger for this module
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('[%(asctime)s] [MemoryManager] [%(levelname)s] %(message)s', 
                                datefmt='%Y-%m-%d %H:%M:%S')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

class MemoryManager:
    """
    Manages conversation memory with vectorization and semantic search
    """
    
    def __init__(self, memory_dir: str = "/app/data/memory"):
        """
        Initialize memory manager
        
        Args:
            memory_dir: Directory to store memory data
        """
        self.memory_dir = Path(memory_dir)
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        
        # Memory storage paths
        self.conversations_file = self.memory_dir / "conversations.jsonl"
        self.embeddings_file = self.memory_dir / "embeddings.npy"
        self.metadata_file = self.memory_dir / "metadata.pkl"
        self.index_file = self.memory_dir / "memory_index.faiss"
        
        # Initialize embedding model
        self.model_name = "all-distilroberta-v1"
        self.embedding_dim = 768
        logger.info(f"🔧 Loading embedding model: {self.model_name}")
        self.embedding_model = SentenceTransformer(self.model_name, device="cpu")
        logger.info(f"✅ Embedding model loaded")
        
        # Memory storage
        self.conversations: List[Dict] = []
        self.embeddings: np.ndarray = None
        self.metadata: List[Dict] = []
        self.index: Optional[faiss.Index] = None
        
        # Thread safety
        self.lock = threading.Lock()  # For general operations
        self.rebuild_lock = threading.Lock()  # For index rebuild operations
        self.rebuild_in_progress = False  # Track if rebuild is happening
        self.last_rebuild_count = 0  # Track when last rebuild happened
        
        # Rebuild configuration (can be adjusted based on usage patterns)
        # Lower values = more frequent rebuilds (better index optimization, more CPU usage)
        # Higher values = less frequent rebuilds (less CPU usage, but index may be less optimal)
        self.rebuild_interval = int(os.environ.get('MEMORY_REBUILD_INTERVAL', '10'))  # Default: every 10 conversations
        
        # Load existing memory
        self._load_memory()
        
        # Initialize FAISS index
        self._initialize_index()
    
    def _load_memory(self):
        """Load existing conversations and embeddings from disk"""
        logger.info(f"📂 Loading existing memory from {self.memory_dir}")
        
        # Load conversations
        if self.conversations_file.exists():
            try:
                with open(self.conversations_file, 'r') as f:
                    self.conversations = [json.loads(line) for line in f if line.strip()]
                logger.info(f"✅ Loaded {len(self.conversations)} conversations from disk")
            except Exception as e:
                logger.warning(f"⚠️ Failed to load conversations: {e}")
                self.conversations = []
        else:
            logger.debug("📂 No existing conversations file found")
            self.conversations = []
        
        # Load embeddings and metadata
        if self.embeddings_file.exists() and self.metadata_file.exists():
            try:
                self.embeddings = np.load(self.embeddings_file)
                with open(self.metadata_file, 'rb') as f:
                    self.metadata = pickle.load(f)
                logger.info(f"✅ Loaded {len(self.metadata)} embeddings from disk")
            except Exception as e:
                logger.warning(f"⚠️ Failed to load embeddings: {e}")
                self.embeddings = None
                self.metadata = []
        else:
            logger.debug("📂 No existing embeddings found")
            self.embeddings = None
            self.metadata = []
    
    def _initialize_index(self):
        """Initialize or load FAISS index"""
        if self.index_file.exists() and self.embeddings is not None and len(self.embeddings) > 0:
            try:
                print(f"[Memory] 📂 Loading FAISS index from {self.index_file}")
                self.index = faiss.read_index(str(self.index_file))
                print(f"[Memory] ✅ Loaded index with {self.index.ntotal} vectors")
            except Exception as e:
                print(f"[Memory] ⚠️ Failed to load index, creating new: {e}")
                self._rebuild_index()
        else:
            self._rebuild_index()
    
    def _rebuild_index(self, background: bool = False):
        """
        Rebuild FAISS index from current embeddings.
        
        Args:
            background: If True, runs in background thread without blocking
        """
        # Check if rebuild is already in progress
        if self.rebuild_in_progress:
            if not background:
                logger.debug("🔧 Index rebuild already in progress, skipping")
            return
        
        def do_rebuild():
            """Internal rebuild function that can run in background"""
            with self.rebuild_lock:
                if self.rebuild_in_progress:
                    return  # Another thread started rebuild
                
                self.rebuild_in_progress = True
                try:
                    # Get current state (snapshot for thread safety)
                    with self.lock:
                        if self.embeddings is None or len(self.embeddings) == 0:
                            logger.warning("⚠️ No embeddings to index")
                            new_index = faiss.IndexFlatIP(self.embedding_dim)
                        else:
                            embeddings_snapshot = self.embeddings.copy()
                            num_vectors = len(embeddings_snapshot)
                            
                            logger.info(f"🔧 Rebuilding FAISS index with {num_vectors} vectors (background: {background})")
                    
                    # Normalize embeddings for cosine similarity (only if we have embeddings)
                    if self.embeddings is not None and len(self.embeddings) > 0:
                        embeddings_normalized = embeddings_snapshot.copy()
                        faiss.normalize_L2(embeddings_normalized)
                        
                        # Create new index (don't modify existing one yet)
                        new_index = faiss.IndexFlatIP(self.embedding_dim)
                        new_index.add(embeddings_normalized)
                        
                        logger.info(f"✅ New index built with {new_index.ntotal} vectors")
                    
                    # Atomically swap indices (quick operation)
                    # IMPORTANT: Take final snapshot right before swap to include any conversations
                    # that were added during the rebuild process
                    # Minimize lock time - do heavy operations (disk write) outside lock
                    with self.lock:
                        # Check if index size changed during rebuild (new conversations added)
                        current_index_size = self.index.ntotal if self.index else 0
                        if new_index.ntotal < current_index_size:
                            # New conversations were added during rebuild - need to include them
                            logger.info(f"🔧 Index grew during rebuild ({current_index_size} vectors in old index, {new_index.ntotal} in new)")
                            # Get the additional embeddings that were added during rebuild
                            missing_count = len(self.embeddings) - new_index.ntotal
                            if missing_count > 0:
                                # Get the tail of embeddings that weren't in the snapshot
                                additional_embeddings = self.embeddings[-missing_count:]
                                additional_normalized = additional_embeddings.copy()
                                faiss.normalize_L2(additional_normalized)
                                new_index.add(additional_normalized)
                                logger.info(f"✅ Added {missing_count} conversations that were added during rebuild")
                        
                        # Atomic swap (very fast - just pointer assignment)
                        old_index = self.index
                        self.index = new_index
                        # Don't save to disk inside lock - do it outside to minimize lock time
                    
                    # Save new index to disk OUTSIDE lock (this is the slow part)
                    # This allows searches to proceed while we write to disk
                    faiss.write_index(new_index, str(self.index_file))
                    logger.info(f"✅ Index swapped and saved (old: {old_index.ntotal if old_index else 0}, new: {new_index.ntotal} vectors)")
                    
                except Exception as e:
                    logger.error(f"❌ Failed to rebuild index: {e}")
                    import traceback
                    traceback.print_exc()
                finally:
                    self.rebuild_in_progress = False
        
        if background:
            # Run in background thread (non-blocking)
            rebuild_thread = threading.Thread(target=do_rebuild, daemon=True, name="IndexRebuild")
            rebuild_thread.start()
            logger.info("🔧 Started background index rebuild thread")
        else:
            # Run synchronously (blocking, for startup)
            do_rebuild()
    
    def store_conversation(self, text: str, timestamp: Optional[float] = None, 
                          source: str = "background", metadata: Optional[Dict] = None) -> str:
        """
        Store a conversation snippet with vectorization
        
        Args:
            text: Conversation text to store
            timestamp: Unix timestamp (default: current time)
            source: Source of conversation (e.g., "background", "wake_word")
            metadata: Additional metadata
        
        Returns:
            Conversation ID
        """
        if not text or not text.strip():
            logger.warning("⚠️ Empty text provided, skipping storage")
            return None
        
        with self.lock:
            timestamp = timestamp or time.time()
            conv_id = hashlib.md5(f"{text}{timestamp}".encode()).hexdigest()[:16]
            
            # Create conversation entry
            conversation = {
                "id": conv_id,
                "text": text.strip(),
                "timestamp": timestamp,
                "datetime": datetime.fromtimestamp(timestamp).isoformat(),
                "source": source,
                "metadata": metadata or {}
            }
            
            # Store conversation
            self.conversations.append(conversation)
            
            # Append to file
            try:
                with open(self.conversations_file, 'a') as f:
                    f.write(json.dumps(conversation) + '\n')
            except Exception as e:
                logger.error(f"❌ Failed to save conversation to file: {e}")
            
            # Generate embedding
            embedding = self.embedding_model.encode([text])[0]
            embedding = np.array([embedding]).astype('float32')
            
            # Update embeddings
            if self.embeddings is None:
                self.embeddings = embedding
            else:
                self.embeddings = np.vstack([self.embeddings, embedding])
            
            # Update metadata
            self.metadata.append({
                "conversation_id": conv_id,
                "timestamp": timestamp,
                "source": source,
                "text_length": len(text),
                **(metadata or {})
            })
            
            # Save embeddings and metadata
            self._save_embeddings()
            
            # Use incremental index updates for immediate availability
            # Periodically rebuild in background for optimal performance
            if self.index is not None:
                embedding_normalized = embedding.copy().reshape(1, -1)  # Ensure 2D shape
                faiss.normalize_L2(embedding_normalized)
                self.index.add(embedding_normalized)
                # Save updated index
                faiss.write_index(self.index, str(self.index_file))
                
                # Trigger background rebuild periodically (configurable interval)
                # Schedule rebuild asynchronously to avoid blocking query processing
                conversations_since_rebuild = len(self.conversations) - self.last_rebuild_count
                if conversations_since_rebuild >= self.rebuild_interval and not self.rebuild_in_progress:
                    # Schedule rebuild in separate thread with delay to avoid blocking current query
                    # Delay ensures swap happens during TTS generation (after LLM response), not during query processing
                    def schedule_rebuild_async():
                        import time
                        # Increased delay to 3s - this ensures swap happens during TTS generation
                        # TTS typically takes 5-15s, so swap will occur during that time, not during query processing
                        logger.info(f"📅 Scheduled background index rebuild (will start in 3s to avoid blocking query processing)")
                        time.sleep(3.0)  # Delay to let query processing and LLM response generation complete
                        if not self.rebuild_in_progress:  # Double-check after delay
                            # Check system load before rebuilding to avoid impacting inference
                            if self._should_rebuild_now():
                                logger.info(f"🔧 Starting background index rebuild (every {self.rebuild_interval} conversations, current: {len(self.conversations)})")
                                self.last_rebuild_count = len(self.conversations)
                                self._rebuild_index(background=True)
                            else:
                                logger.info(f"⏸️ Skipping rebuild - system under load (will retry later)")
                                # Reset counter to try again soon
                                self.last_rebuild_count = len(self.conversations) - (self.rebuild_interval // 2)
                    
                    # Start rebuild scheduling in background thread (non-blocking)
                    # This returns immediately, allowing query processing to continue
                    schedule_thread = threading.Thread(target=schedule_rebuild_async, daemon=True, name="ScheduleIndexRebuild")
                    schedule_thread.start()
                    # Don't wait for thread - return immediately so query can continue
            else:
                # Index not initialized yet - this should only happen on startup
                logger.warning("⚠️ Index not initialized, skipping incremental update (will be built on next startup)")
            
            logger.info(f"✅ Stored conversation: '{text[:50]}...' (ID: {conv_id}, source: {source})")
            return conv_id
    
    def _should_rebuild_now(self) -> bool:
        """
        Check if system load is low enough to safely rebuild index.
        Returns True if rebuild should proceed, False if system is too busy.
        """
        try:
            import psutil
            # Get CPU usage over last second
            cpu_percent = psutil.cpu_percent(interval=0.1)
            
            # Get memory usage
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            
            # Only rebuild if CPU usage is below 70% and memory usage is below 85%
            # This ensures we don't impact inference or other critical processes
            cpu_threshold = 70.0
            memory_threshold = 85.0
            
            if cpu_percent > cpu_threshold:
                logger.debug(f"⏸️ CPU usage too high ({cpu_percent:.1f}% > {cpu_threshold}%) - skipping rebuild")
                return False
            
            if memory_percent > memory_threshold:
                logger.debug(f"⏸️ Memory usage too high ({memory_percent:.1f}% > {memory_threshold}%) - skipping rebuild")
                return False
            
            logger.debug(f"✅ System load acceptable (CPU: {cpu_percent:.1f}%, Memory: {memory_percent:.1f}%) - rebuild OK")
            return True
            
        except ImportError:
            # psutil not available - proceed with rebuild (assume system can handle it)
            logger.debug("⚠️ psutil not available - proceeding with rebuild (install psutil for load checking)")
            return True
        except Exception as e:
            # Error checking load - proceed with rebuild (better than blocking)
            logger.debug(f"⚠️ Error checking system load: {e} - proceeding with rebuild")
            return True
    
    def _save_embeddings(self):
        """Save embeddings and metadata to disk"""
        try:
            if self.embeddings is not None:
                np.save(self.embeddings_file, self.embeddings)
            with open(self.metadata_file, 'wb') as f:
                pickle.dump(self.metadata, f)
        except Exception as e:
            logger.error(f"[Memory] Failed to save embeddings: {e}")
    
    def search_similar(self, query: str, k: int = 5, threshold: float = 0.5) -> List[Dict]:
        """
        Search for similar conversations
        
        Args:
            query: Search query
            k: Number of results
            threshold: Similarity threshold (0-1)
        
        Returns:
            List of similar conversations with scores
        """
        logger.debug(f"🔍 Searching for similar conversations (query: '{query[:50]}...', k={k}, threshold={threshold})")
        
        if self.index is None or self.index.ntotal == 0:
            logger.warning("⚠️ FAISS index is empty, cannot search")
            return []
        
        with self.lock:
            # Generate query embedding
            logger.debug("🔢 Generating query embedding...")
            query_embedding = self.embedding_model.encode([query])[0]
            query_embedding = np.array([query_embedding]).astype('float32').reshape(1, -1)
            
            # Normalize for cosine similarity
            faiss.normalize_L2(query_embedding)
            logger.debug("✅ Query embedding normalized")
            
            # Search
            logger.debug(f"🔎 Searching FAISS index ({self.index.ntotal} vectors)...")
            scores, indices = self.index.search(query_embedding, k)
            logger.debug(f"📊 Search returned {len(scores[0])} candidates")
            
            # Build results
            results = []
            for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
                if idx < len(self.conversations) and score >= threshold:
                    conv = self.conversations[idx]
                    results.append({
                        "conversation": conv,
                        "score": float(score),
                        "metadata": self.metadata[idx] if idx < len(self.metadata) else {}
                    })
                    logger.debug(f"  [{i+1}] Score: {score:.3f} >= {threshold} ✅ - '{conv.get('text', '')[:50]}...'")
                elif idx < len(self.conversations):
                    logger.debug(f"  [{i+1}] Score: {score:.3f} < {threshold} ❌ - Skipped")
            
            logger.info(f"✅ Found {len(results)} similar conversations (threshold: {threshold})")
            return results
    
    def search_recent(self, hours: int = 24, limit: int = 50) -> List[Dict]:
        """
        Get recent conversations
        
        Args:
            hours: Number of hours to look back
            limit: Maximum number of results
        
        Returns:
            List of recent conversations
        """
        cutoff_time = time.time() - (hours * 3600)
        
        recent = [
            conv for conv in self.conversations
            if conv.get("timestamp", 0) >= cutoff_time
        ]
        
        # Sort by timestamp (newest first)
        recent.sort(key=lambda x: x.get("timestamp", 0), reverse=True)
        
        return recent[:limit]
    
    def delete_conversations(self, conversation_ids: List[str]) -> Dict:
        """
        Delete conversations by their IDs
        
        Args:
            conversation_ids: List of conversation IDs to delete
        
        Returns:
            Dict with deletion statistics
        """
        if not conversation_ids:
            return {"deleted": 0, "not_found": []}
        
        with self.lock:
            # Create a set for faster lookup
            ids_to_delete = set(conversation_ids)
            found_ids = set()
            
            # Build new lists/arrays excluding items to delete
            new_conversations = []
            new_metadata = []
            new_embeddings_list = []
            
            # Track which IDs were actually found
            for i, conv in enumerate(self.conversations):
                conv_id = conv.get("id")
                if conv_id in ids_to_delete:
                    found_ids.add(conv_id)
                    # Skip this conversation (don't add to new lists)
                    continue
                
                # Keep this conversation
                new_conversations.append(conv)
                
                # Keep corresponding metadata
                if i < len(self.metadata):
                    new_metadata.append(self.metadata[i])
                
                # Keep corresponding embedding
                if self.embeddings is not None and i < len(self.embeddings):
                    new_embeddings_list.append(self.embeddings[i])
            
            # Check for not found IDs
            not_found = list(ids_to_delete - found_ids)
            deleted_count = len(found_ids)
            
            # Update in-memory data
            self.conversations = new_conversations
            
            # Update metadata
            self.metadata = new_metadata
            
            # Update embeddings (reconstruct numpy array)
            if new_embeddings_list:
                self.embeddings = np.vstack(new_embeddings_list).astype('float32')
            else:
                self.embeddings = None
            
            # Save updated data to disk
            if deleted_count > 0:
                # Rewrite conversations file
                try:
                    with open(self.conversations_file, 'w') as f:
                        for conv in self.conversations:
                            f.write(json.dumps(conv) + '\n')
                    logger.info(f"✅ Rewrote conversations file after deletion")
                except Exception as e:
                    logger.error(f"❌ Failed to save conversations after deletion: {e}")
                
                # Save embeddings and metadata
                self._save_embeddings()
                
                # Rebuild index (in background to avoid blocking)
                logger.info(f"🔧 Rebuilding index after deleting {deleted_count} conversations...")
                self._rebuild_index(background=True)
            
            logger.info(f"✅ Deleted {deleted_count} conversation(s), {len(not_found)} not found")
            return {
                "deleted": deleted_count,
                "not_found": not_found
            }
    
    def delete_all_conversations(self) -> Dict:
        """
        Delete all conversations
        
        Returns:
            Dict with deletion statistics
        """
        with self.lock:
            count = len(self.conversations)
            
            # Clear all data
            self.conversations = []
            self.embeddings = None
            self.metadata = []
            
            # Clear files
            try:
                # Clear conversations file
                if self.conversations_file.exists():
                    with open(self.conversations_file, 'w') as f:
                        pass  # Truncate file
                    logger.info(f"✅ Cleared conversations file")
                
                # Remove embeddings and metadata files
                if self.embeddings_file.exists():
                    self.embeddings_file.unlink()
                    logger.info(f"✅ Removed embeddings file")
                
                if self.metadata_file.exists():
                    self.metadata_file.unlink()
                    logger.info(f"✅ Removed metadata file")
                
                # Remove index file
                if self.index_file.exists():
                    self.index_file.unlink()
                    logger.info(f"✅ Removed index file")
                
            except Exception as e:
                logger.error(f"❌ Failed to clear files: {e}")
            
            # Rebuild empty index
            self._rebuild_index(background=False)
            
            logger.info(f"✅ Deleted all {count} conversation(s)")
            return {
                "deleted": count
            }
    
    def get_stats(self) -> Dict:
        """Get memory statistics"""
        return {
            "total_conversations": len(self.conversations),
            "total_embeddings": len(self.metadata) if self.metadata else 0,
            "index_size": self.index.ntotal if self.index else 0,
            "memory_dir": str(self.memory_dir),
            "oldest_conversation": min([c.get("timestamp", 0) for c in self.conversations]) if self.conversations else None,
            "newest_conversation": max([c.get("timestamp", 0) for c in self.conversations]) if self.conversations else None
        }


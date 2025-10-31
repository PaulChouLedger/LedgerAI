#!/usr/bin/env python3
"""
CPU FAISS Auto-Ingestion System
Monitors data/input/ for new guideline files and automatically updates embeddings
"""

import os
import time
import json
import pickle
import hashlib
from pathlib import Path
from typing import Dict, List, Any
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

class CPUFAISSAutoIngest:
    """Auto-ingestion system for CPU FAISS"""
    
    def __init__(self):
        self.input_dir = Path("../data/input")  # Relative to llm-medical-container
        self.cpu_embeddings_dir = Path("../data/embeddings")  # Relative to llm-medical-container
        self.model_name = "all-distilroberta-v1"
        self.embedding_dimension = 768
        
        # Initialize components
        self.model = None
        self.index = None
        self.chunks = []
        self.metadata = []
        self.state_file = Path("data/cpu_ingest_state.json")
        self.state = self._load_state()
        
        # Ensure directories exist
        self.input_dir.mkdir(parents=True, exist_ok=True)
        self.cpu_embeddings_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize model
        self._load_model()
        
        # Initialize file watcher
        self.observer = None
        self.watching = False
    
    def _load_model(self):
        """Load the sentence transformer model"""
        try:
            self.model = SentenceTransformer(self.model_name)
            print(f"[Auto-Ingest] ✅ Loaded model: {self.model_name}")
        except Exception as e:
            print(f"[Auto-Ingest] ❌ Failed to load model: {e}")
            raise
    
    def _load_state(self) -> Dict:
        """Load ingestion state"""
        if self.state_file.exists():
            try:
                with open(self.state_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"[Auto-Ingest] ⚠️ Failed to load state: {e}")
        return {"processed_files": {}, "last_scan": None}
    
    def _save_state(self):
        """Save ingestion state"""
        try:
            with open(self.state_file, 'w') as f:
                json.dump(self.state, f, indent=2)
        except Exception as e:
            print(f"[Auto-Ingest] ⚠️ Failed to save state: {e}")
    
    def _get_file_hash(self, file_path: Path) -> str:
        """Get file hash for change detection"""
        try:
            with open(file_path, 'rb') as f:
                return hashlib.md5(f.read()).hexdigest()
        except Exception as e:
            print(f"[Auto-Ingest] ⚠️ Failed to get hash for {file_path}: {e}")
            return ""
    
    def chunk_text(self, text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        """Chunk text into overlapping segments"""
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk = " ".join(words[i:i + chunk_size])
            if chunk.strip():
                chunks.append(chunk.strip())
        
        return chunks
    
    def _process_file(self, file_path: Path) -> bool:
        """Process a single guideline file"""
        try:
            # Check if file was already processed
            file_hash = self._get_file_hash(file_path)
            if file_path.name in self.state["processed_files"]:
                if self.state["processed_files"][file_path.name]["hash"] == file_hash:
                    return False  # Already processed, no changes
            
            print(f"[Auto-Ingest] 📄 Processing: {file_path.name}")
            
            # Read file content
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Extract guideline name from filename
            guideline_name = file_path.stem.replace("GUIDELINE_", "")
            
            # Chunk the content
            chunks = self.chunk_text(content)
            
            # Generate embeddings
            embeddings = self.model.encode(chunks)
            
            # Create metadata for each chunk
            chunk_metadata = []
            for i, chunk in enumerate(chunks):
                chunk_metadata.append({
                    "chunk_id": f"{guideline_name}_{i}",
                    "guideline_name": guideline_name,
                    "chunk_index": i,
                    "text": chunk,
                    "file_path": str(file_path)
                })
            
            # Add to existing data
            if isinstance(self.chunks, list):
                self.chunks.extend(chunks)
            else:
                # Convert numpy array to list if needed
                self.chunks = list(self.chunks) + chunks
            
            self.metadata.extend(chunk_metadata)
            
            # Update state
            self.state["processed_files"][file_path.name] = {
                "hash": file_hash,
                "processed_at": time.time(),
                "chunks": len(chunks)
            }
            
            print(f"[Auto-Ingest] ✅ Processed {file_path.name}: {len(chunks)} chunks")
            return True
            
        except Exception as e:
            print(f"[Auto-Ingest] ❌ Error processing {file_path.name}: {e}")
            return False
    
    def _generate_embeddings(self) -> np.ndarray:
        """Generate embeddings for all chunks"""
        if not self.chunks:
            return np.array([])
        
        print(f"[Auto-Ingest] 🔧 Generating embeddings for {len(self.chunks)} chunks...")
        embeddings = self.model.encode(self.chunks)
        print(f"[Auto-Ingest] ✅ Generated embeddings: {embeddings.shape}")
        return embeddings
    
    def _save_embeddings_cpu_format(self):
        """Save embeddings in CPU FAISS format"""
        if not self.chunks or not self.metadata:
            print("[Auto-Ingest] ⚠️ No data to save")
            return
        
        try:
            # Generate embeddings
            embeddings = self._generate_embeddings()
            
            if embeddings.size == 0:
                print("[Auto-Ingest] ⚠️ No embeddings to save")
                return
            
            # Create FAISS index
            index = faiss.IndexFlatIP(self.embedding_dimension)  # Inner product for cosine similarity
            index.add(embeddings.astype('float32'))
            
            # Save CPU FAISS format
            faiss.write_index(index, str(self.cpu_embeddings_dir / "faiss_index.bin"))
            
            # Save metadata
            metadata = {
                "total_chunks": len(self.chunks),
                "embedding_dimension": self.embedding_dimension,
                "model_name": self.model_name,
                "chunks": self.chunks,
                "metadata": self.metadata
            }
            
            with open(self.cpu_embeddings_dir / "metadata.pkl", 'wb') as f:
                pickle.dump(metadata, f)
            
            print(f"[Auto-Ingest] ✅ Saved CPU FAISS embeddings: {len(self.chunks)} chunks")
            
        except Exception as e:
            print(f"[Auto-Ingest] ❌ Failed to save CPU embeddings: {e}")
            raise
    
    def scan_and_process(self) -> Dict[str, Any]:
        """Scan input directory and process new/modified files"""
        print("[Auto-Ingest] 🔍 Scanning for new/modified files...")
        
        processed_count = 0
        skipped_count = 0
        error_count = 0
        
        # Find all guideline files
        guideline_files = list(self.input_dir.glob("GUIDELINE_*.txt"))
        
        for file_path in guideline_files:
            try:
                if self._process_file(file_path):
                    processed_count += 1
                else:
                    skipped_count += 1
            except Exception as e:
                print(f"[Auto-Ingest] ❌ Error processing {file_path.name}: {e}")
                error_count += 1
        
        # Save embeddings if any files were processed
        if processed_count > 0:
            self._save_embeddings_cpu_format()
            self._save_state()
        
        result = {
            "processed": processed_count,
            "skipped": skipped_count,
            "errors": error_count,
            "total_chunks": len(self.chunks)
        }
        
        print(f"[Auto-Ingest] 📊 Scan complete: {processed_count} processed, {skipped_count} skipped, {error_count} errors")
        return result
    
    def load_existing_embeddings(self) -> bool:
        """Load existing embeddings from disk"""
        try:
            index_path = self.cpu_embeddings_dir / "faiss_index.bin"
            metadata_path = self.cpu_embeddings_dir / "metadata.pkl"
            
            if not index_path.exists() or not metadata_path.exists():
                print("[Auto-Ingest] ⚠️ No existing embeddings found")
                return False
            
            # Load metadata
            with open(metadata_path, 'rb') as f:
                metadata = pickle.load(f)
            
            self.chunks = metadata.get("chunks", [])
            self.metadata = metadata.get("metadata", [])
            
            print(f"[Auto-Ingest] ✅ Loaded existing embeddings: {len(self.chunks)} chunks")
            return True
            
        except Exception as e:
            print(f"[Auto-Ingest] ❌ Failed to load existing embeddings: {e}")
            return False
    
    def start_watching(self):
        """Start file system watching"""
        if self.watching:
            return
        
        try:
            from watchdog.observers import Observer
            from watchdog.events import FileSystemEventHandler
            
            class GuidelineHandler(FileSystemEventHandler):
                def __init__(self, auto_ingest):
                    self.auto_ingest = auto_ingest
                
                def on_created(self, event):
                    if not event.is_directory and event.src_path.endswith('.txt'):
                        self.auto_ingest.scan_and_process()
                
                def on_modified(self, event):
                    if not event.is_directory and event.src_path.endswith('.txt'):
                        self.auto_ingest.scan_and_process()
            
            self.observer = Observer()
            self.observer.schedule(GuidelineHandler(self), str(self.input_dir), recursive=False)
            self.observer.start()
            self.watching = True
            
            print("[Auto-Ingest] 👀 Started file watching")
            
        except ImportError:
            print("[Auto-Ingest] ⚠️ Watchdog not available, file watching disabled")
        except Exception as e:
            print(f"[Auto-Ingest] ❌ Failed to start watching: {e}")
    
    def stop_watching(self):
        """Stop file system watching"""
        if self.observer and self.watching:
            self.observer.stop()
            self.observer.join()
            self.watching = False
            print("[Auto-Ingest] 🛑 Stopped file watching")

if __name__ == "__main__":
    # Test the auto-ingestion system
    auto_ingest = CPUFAISSAutoIngest()
    
    # Load existing embeddings
    auto_ingest.load_existing_embeddings()
    
    # Scan and process
    result = auto_ingest.scan_and_process()
    print(f"Result: {result}")

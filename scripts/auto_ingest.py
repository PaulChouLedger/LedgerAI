#!/usr/bin/env python3
"""
Auto-ingest pipeline for RAG system
Monitors data/input/ directory and automatically processes new files
"""
import os
import time
import json
import hashlib
import shutil
import numpy as np
import faiss
from pathlib import Path
from typing import List, Dict, Set
from datetime import datetime
from sentence_transformers import SentenceTransformer
import PyPDF2
import docx
import re

class AutoIngestPipeline:
    def __init__(self, 
                 input_dir: str = "data/input",
                 parsed_dir: str = "data/parsed", 
                 embeddings_dir: str = "data/embeddings",
                 state_file: str = "data/ingest_state.json",
                 model_name: str = "all-MiniLM-L6-v2",
                 chunk_size: int = 400,
                 chunk_overlap: int = 50):
        
        self.input_dir = Path(input_dir)
        self.parsed_dir = Path(parsed_dir)
        self.embeddings_dir = Path(embeddings_dir)
        self.state_file = Path(state_file)
        self.model_name = model_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Create directories
        self.input_dir.mkdir(parents=True, exist_ok=True)
        self.parsed_dir.mkdir(parents=True, exist_ok=True)
        self.embeddings_dir.mkdir(parents=True, exist_ok=True)
        
        # Load sentence transformer
        print(f"[AutoIngest] 🔧 Loading sentence transformer: {model_name}")
        self.encoder = SentenceTransformer(model_name)
        
        # Load or create state
        self.state = self.load_state()
        
        print(f"[AutoIngest] ✅ Initialized auto-ingest pipeline")
        print(f"  📁 Input: {self.input_dir}")
        print(f"  📁 Parsed: {self.parsed_dir}")
        print(f"  📁 Embeddings: {self.embeddings_dir}")
    
    def load_state(self) -> Dict:
        """Load processing state from disk"""
        if self.state_file.exists():
            try:
                with open(self.state_file, 'r') as f:
                    state = json.load(f)
                print(f"[AutoIngest] 📋 Loaded state: {len(state.get('processed_files', {}))} files tracked")
                return state
            except Exception as e:
                print(f"[AutoIngest] ⚠️ Error loading state: {e}")
        
        return {
            "processed_files": {},  # filename -> {hash, timestamp, chunks_count}
            "last_scan": None,
            "total_chunks": 0
        }
    
    def save_state(self):
        """Save processing state to disk"""
        try:
            with open(self.state_file, 'w') as f:
                json.dump(self.state, f, indent=2)
        except Exception as e:
            print(f"[AutoIngest] ❌ Error saving state: {e}")
    
    def get_file_hash(self, file_path: Path) -> str:
        """Calculate MD5 hash of a file"""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    
    def extract_text_from_pdf(self, pdf_path: Path) -> str:
        """Extract text from PDF file"""
        try:
            text = ""
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
            return text.strip()
        except Exception as e:
            print(f"[AutoIngest] ❌ Error extracting PDF {pdf_path}: {e}")
            return ""
    
    def extract_text_from_docx(self, docx_path: Path) -> str:
        """Extract text from DOCX file"""
        try:
            doc = docx.Document(docx_path)
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return text.strip()
        except Exception as e:
            print(f"[AutoIngest] ❌ Error extracting DOCX {docx_path}: {e}")
            return ""
    
    def extract_text_from_txt(self, txt_path: Path) -> str:
        """Extract text from TXT file"""
        try:
            with open(txt_path, 'r', encoding='utf-8') as f:
                return f.read().strip()
        except Exception as e:
            print(f"[AutoIngest] ❌ Error extracting TXT {txt_path}: {e}")
            return ""
    
    def extract_text(self, file_path: Path) -> str:
        """Extract text from various file formats"""
        suffix = file_path.suffix.lower()
        
        if suffix == '.pdf':
            return self.extract_text_from_pdf(file_path)
        elif suffix == '.docx':
            return self.extract_text_from_docx(file_path)
        elif suffix in ['.txt', '.md']:
            return self.extract_text_from_txt(file_path)
        else:
            print(f"[AutoIngest] ⚠️ Unsupported file format: {suffix}")
            return ""
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        if not text:
            return ""
        
        # Normalize whitespace
        text = re.sub(r'\n\s*\n', '\n\n', text)  # Normalize paragraph breaks
        text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
        text = text.strip()
        
        return text
    
    def chunk_text(self, text: str) -> List[str]:
        """Split text into overlapping chunks"""
        if len(text) <= self.chunk_size:
            return [text.strip()]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + self.chunk_size
            
            # If we're not at the end, try to break at a sentence boundary
            if end < len(text):
                # Look for sentence endings within the last 100 characters
                search_start = max(start + self.chunk_size - 100, start)
                sentence_end = -1
                
                for i in range(end, search_start, -1):
                    if text[i] in '.!?':
                        # Make sure it's not an abbreviation
                        if i + 1 < len(text) and text[i + 1] in ' \n\t':
                            sentence_end = i + 1
                            break
                
                if sentence_end > 0:
                    end = sentence_end
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            # Move start position with overlap
            start = end - self.chunk_overlap
            if start >= len(text):
                break
        
        return chunks
    
    def process_file(self, file_path: Path) -> bool:
        """Process a single file through the pipeline"""
        print(f"[AutoIngest] 📄 Processing: {file_path.name}")
        
        try:
            # Extract text
            raw_text = self.extract_text(file_path)
            if not raw_text:
                print(f"[AutoIngest] ⚠️ No text extracted from {file_path.name}")
                return False
            
            # Clean text
            clean_text = self.clean_text(raw_text)
            if not clean_text:
                print(f"[AutoIngest] ⚠️ No clean text from {file_path.name}")
                return False
            
            # Save parsed text
            parsed_file = self.parsed_dir / f"{file_path.stem}.txt"
            with open(parsed_file, 'w', encoding='utf-8') as f:
                f.write(clean_text)
            
            # Generate chunks
            chunks = self.chunk_text(clean_text)
            if not chunks:
                print(f"[AutoIngest] ⚠️ No chunks generated from {file_path.name}")
                return False
            
            print(f"[AutoIngest] ✅ Generated {len(chunks)} chunks from {file_path.name}")
            return True
            
        except Exception as e:
            print(f"[AutoIngest] ❌ Error processing {file_path.name}: {e}")
            return False
    
    def rebuild_embeddings(self):
        """Rebuild the entire FAISS index and doc_chunks from all parsed files"""
        print(f"[AutoIngest] 🔧 Rebuilding embeddings from all parsed files...")
        
        # Collect all chunks from parsed files
        all_chunks = []
        
        for parsed_file in self.parsed_dir.glob("*.txt"):
            try:
                with open(parsed_file, 'r', encoding='utf-8') as f:
                    text = f.read()
                
                chunks = self.chunk_text(text)
                all_chunks.extend(chunks)
                print(f"[AutoIngest] 📄 Added {len(chunks)} chunks from {parsed_file.name}")
                
            except Exception as e:
                print(f"[AutoIngest] ❌ Error reading {parsed_file}: {e}")
        
        if not all_chunks:
            print(f"[AutoIngest] ⚠️ No chunks found to embed")
            return False
        
        print(f"[AutoIngest] 🔢 Total chunks to embed: {len(all_chunks)}")
        
        # Generate embeddings
        print(f"[AutoIngest] 🧠 Generating embeddings...")
        embeddings = self.encoder.encode(all_chunks, convert_to_numpy=True, show_progress_bar=True)
        
        # Create FAISS index
        print(f"[AutoIngest] 🔍 Creating FAISS index...")
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings.astype('float32'))
        
        # Save FAISS index
        index_path = self.embeddings_dir / "index.faiss"
        faiss.write_index(index, str(index_path))
        
        # Save document chunks
        chunks_path = self.embeddings_dir / "doc_chunks.npy"
        chunks_array = np.array(all_chunks, dtype=object)
        np.save(chunks_path, chunks_array)
        
        # Update state
        self.state["total_chunks"] = len(all_chunks)
        self.save_state()
        
        print(f"[AutoIngest] ✅ Embeddings rebuilt successfully!")
        print(f"  📊 Total chunks: {len(all_chunks)}")
        print(f"  📁 Index: {index_path}")
        print(f"  📁 Chunks: {chunks_path}")
        
        return True
    
    def scan_for_new_files(self) -> List[Path]:
        """Scan input directory for new or modified files"""
        new_files = []
        
        if not self.input_dir.exists():
            print(f"[AutoIngest] ⚠️ Input directory not found: {self.input_dir}")
            return new_files
        
        # Supported file extensions
        supported_extensions = {'.pdf', '.docx', '.txt', '.md'}
        
        for file_path in self.input_dir.iterdir():
            if not file_path.is_file():
                continue
            
            if file_path.suffix.lower() not in supported_extensions:
                continue
            
            # Calculate current file hash
            try:
                current_hash = self.get_file_hash(file_path)
            except Exception as e:
                print(f"[AutoIngest] ❌ Error hashing {file_path.name}: {e}")
                continue
            
            # Check if file is new or modified
            file_key = file_path.name
            stored_info = self.state["processed_files"].get(file_key)
            
            if not stored_info or stored_info.get("hash") != current_hash:
                new_files.append(file_path)
                print(f"[AutoIngest] 🆕 {'New' if not stored_info else 'Modified'} file: {file_path.name}")
        
        return new_files
    
    def process_new_files(self) -> bool:
        """Process any new files and rebuild embeddings if needed"""
        new_files = self.scan_for_new_files()
        
        if not new_files:
            print(f"[AutoIngest] ✅ No new files to process")
            return False
        
        print(f"[AutoIngest] 🔄 Processing {len(new_files)} new/modified files...")
        
        processed_any = False
        
        for file_path in new_files:
            if self.process_file(file_path):
                # Update state for successfully processed file
                file_hash = self.get_file_hash(file_path)
                self.state["processed_files"][file_path.name] = {
                    "hash": file_hash,
                    "timestamp": datetime.now().isoformat(),
                    "processed": True
                }
                processed_any = True
            else:
                print(f"[AutoIngest] ❌ Failed to process {file_path.name}")
        
        if processed_any:
            # Rebuild embeddings with all files
            self.rebuild_embeddings()
            
            # Update scan timestamp
            self.state["last_scan"] = datetime.now().isoformat()
            self.save_state()
            
            print(f"[AutoIngest] 🎉 Auto-ingest completed successfully!")
            return True
        else:
            print(f"[AutoIngest] ❌ No files were successfully processed")
            return False
    
    def run_once(self):
        """Run the auto-ingest pipeline once"""
        print(f"[AutoIngest] 🚀 Starting auto-ingest scan...")
        self.process_new_files()
    
    def run_continuous(self, interval: int = 60):
        """Run the auto-ingest pipeline continuously"""
        print(f"[AutoIngest] 🔄 Starting continuous monitoring (interval: {interval}s)")
        
        try:
            while True:
                self.run_once()
                print(f"[AutoIngest] 😴 Sleeping for {interval} seconds...")
                time.sleep(interval)
        except KeyboardInterrupt:
            print(f"[AutoIngest] ⛔ Stopping continuous monitoring")

def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Auto-ingest pipeline for RAG system")
    parser.add_argument("--continuous", "-c", action="store_true", 
                       help="Run continuously (default: run once)")
    parser.add_argument("--interval", "-i", type=int, default=60,
                       help="Scan interval in seconds (default: 60)")
    parser.add_argument("--rebuild", "-r", action="store_true",
                       help="Force rebuild of all embeddings")
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = AutoIngestPipeline()
    
    if args.rebuild:
        print("[AutoIngest] 🔧 Force rebuilding embeddings...")
        pipeline.rebuild_embeddings()
    elif args.continuous:
        pipeline.run_continuous(args.interval)
    else:
        pipeline.run_once()

if __name__ == "__main__":
    main()

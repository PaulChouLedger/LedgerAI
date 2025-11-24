"""
Simple cuDF-based document processing for RAG pipeline.
Handles chunking, cleaning, and metadata extraction.
"""
import os
import cudf
import pandas as pd
from typing import List, Dict, Any
from bs4 import BeautifulSoup

from aura_rag.config import AuraRAGConfig


class DocumentProcessor:
    """Simple document processor using cuDF for GPU-accelerated text processing."""
    
    def __init__(self, config: AuraRAGConfig):
        self.config = config
        self.chunk_size = config.chunk_size
        self.chunk_overlap = 50  # Fixed reasonable value
        self.sentence_boundary_chars = ".!?"  # Fixed reasonable value
        self.min_chunk_size = 32  # Fixed reasonable value
    
    def clean_text(self, text: str) -> str:
        """Clean HTML and normalize text."""
        if not text:
            return ""
        
        # Remove HTML tags
        soup = BeautifulSoup(text, "html.parser")
        clean_text = soup.get_text(separator=" ")
        
        # Basic normalization
        clean_text = " ".join(clean_text.split())
        return clean_text.strip()
    
    def chunk_text(self, text: str, doc_id: str) -> List[Dict[str, Any]]:
        """Split text into overlapping chunks with metadata."""
        if not text:
            return []
        
        chunks = []
        start = 0
        chunk_idx = 0
        
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            chunk_text = text[start:end]
            
            # Try to break at sentence boundary
            if end < len(text):
                last_sentence = -1
                for char in self.sentence_boundary_chars:
                    pos = chunk_text.rfind(char)
                    if pos > last_sentence:
                        last_sentence = pos
                
                if last_sentence > start + self.chunk_size // 2:
                    end = start + last_sentence + 1
                    chunk_text = text[start:end]
            
            # Only add chunk if it meets minimum size requirement
            if len(chunk_text.strip()) >= self.min_chunk_size:
                chunks.append({
                    "doc_id": doc_id,
                    "chunk_id": f"{doc_id}_chunk_{chunk_idx}",
                    "text": chunk_text.strip(),
                    "start_pos": start,
                    "end_pos": end,
                    "chunk_index": chunk_idx
                })
            
            start = end - self.chunk_overlap
            chunk_idx += 1
        
        return chunks
    
    def process_documents(self, input_dir: str) -> cudf.DataFrame:
        """Process all documents (.txt, .pdf) in directory and return cuDF DataFrame."""
        if not os.path.exists(input_dir):
            print(f"Input directory does not exist: {input_dir}")
            return cudf.DataFrame()
            
        all_chunks = []
        
        for filename in os.listdir(input_dir):
            if not filename.endswith(('.txt', '.pdf')):
                continue
                
            file_path = os.path.join(input_dir, filename)
            doc_id = os.path.splitext(filename)[0]
            
            try:
                if filename.endswith('.txt'):
                    with open(file_path, 'r', encoding='utf-8') as f:
                        text = f.read()
                elif filename.endswith('.pdf'):
                    # Simple PDF handling - in production you'd use pdfminer
                    try:
                        from pdfminer.high_level import extract_text
                        text = extract_text(file_path)
                    except ImportError:
                        print(f"PDF processing not available - install pdfminer.six")
                        text = f"PDF content from {filename} (PDF processing not available)"
                    except Exception as e:
                        print(f"Error processing PDF {filename}: {e}")
                        text = f"PDF content from {filename} (processing failed)"
                
                clean_text = self.clean_text(text)
                chunks = self.chunk_text(clean_text, doc_id)
                all_chunks.extend(chunks)
                
            except Exception as e:
                print(f"Error processing {filename}: {e}")
                continue
        
        if not all_chunks:
            return cudf.DataFrame()
        
        # Convert to cuDF for GPU processing
        df = pd.DataFrame(all_chunks)
        return cudf.from_pandas(df)

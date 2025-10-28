#!/usr/bin/env python3
"""
Optimize RAG Architecture - Single Vector Store for Both Containers

This script implements the optimal architecture where:
1. All guidelines are pre-processed to data/input/
2. Single vectorization pipeline creates embeddings in data/embeddings/
3. Both CPU FAISS (llm-medical-container) and GPU FAISS (rag-container) use same data
4. Only difference is RAG_MODE toggle (cpu vs gpu)
"""

import os
import json
import numpy as np
import faiss
from pathlib import Path
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
import time

class OptimizedRAGArchitecture:
    """
    Optimized RAG architecture with shared vector store
    """
    
    def __init__(self, 
                 input_dir: str = "data/input",
                 embeddings_dir: str = "data/embeddings",
                 model_name: str = "all-distilroberta-v1"):
        
        self.input_dir = Path(input_dir)
        self.embeddings_dir = Path(embeddings_dir)
        self.model_name = model_name
        
        # Create directories
        self.embeddings_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize encoder
        self.encoder = None
        self.index = None
        self.chunks = []
        self.chunk_metadata = []
        
        print(f"[OptimizedRAG] 📂 Input dir: {self.input_dir}")
        print(f"[OptimizedRAG] 📂 Embeddings dir: {self.embeddings_dir}")
    
    def load_encoder(self):
        """Load sentence transformer model"""
        print(f"[OptimizedRAG] 🔧 Loading encoder: {self.model_name}")
        self.encoder = SentenceTransformer(self.model_name)
        print(f"[OptimizedRAG] ✅ Encoder loaded")
    
    def process_guidelines(self) -> List[str]:
        """Process all guideline files from data/input/"""
        print(f"[OptimizedRAG] 📚 Processing guidelines from {self.input_dir}")
        
        guideline_files = list(self.input_dir.glob("GUIDELINE_*.txt"))
        print(f"[OptimizedRAG] Found {len(guideline_files)} guideline files")
        
        all_chunks = []
        
        for file_path in guideline_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Extract condition name from filename
                condition_name = file_path.stem.replace("GUIDELINE_", "")
                
                # Split into chunks (same logic as RAG container)
                chunks = self._chunk_text(content, condition_name)
                all_chunks.extend(chunks)
                
                print(f"[OptimizedRAG] ✅ Processed {file_path.name}: {len(chunks)} chunks")
                
            except Exception as e:
                print(f"[OptimizedRAG] ❌ Error processing {file_path.name}: {e}")
        
        print(f"[OptimizedRAG] 📊 Total chunks: {len(all_chunks)}")
        return all_chunks
    
    def _chunk_text(self, text: str, condition_name: str, chunk_size: int = 400, overlap: int = 50) -> List[Dict]:
        """Split text into overlapping chunks"""
        chunks = []
        
        # Split by paragraphs first
        paragraphs = text.split('\n\n')
        current_chunk = ""
        
        for paragraph in paragraphs:
            if len(current_chunk) + len(paragraph) <= chunk_size:
                current_chunk += paragraph + "\n\n"
            else:
                if current_chunk.strip():
                    chunks.append({
                        'text': current_chunk.strip(),
                        'condition': condition_name,
                        'chunk_id': len(chunks)
                    })
                current_chunk = paragraph + "\n\n"
        
        # Add final chunk
        if current_chunk.strip():
            chunks.append({
                'text': current_chunk.strip(),
                'condition': condition_name,
                'chunk_id': len(chunks)
            })
        
        return chunks
    
    def create_embeddings(self, chunks: List[Dict]) -> np.ndarray:
        """Create embeddings for all chunks"""
        print(f"[OptimizedRAG] 🔧 Creating embeddings for {len(chunks)} chunks")
        
        texts = [chunk['text'] for chunk in chunks]
        embeddings = self.encoder.encode(texts, convert_to_numpy=True)
        
        print(f"[OptimizedRAG] ✅ Created embeddings: {embeddings.shape}")
        return embeddings
    
    def build_faiss_index(self, embeddings: np.ndarray) -> faiss.Index:
        """Build FAISS index for fast similarity search"""
        print(f"[OptimizedRAG] 🔧 Building FAISS index")
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        
        # Create index
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
        
        # Add vectors to index
        index.add(embeddings)
        
        print(f"[OptimizedRAG] ✅ FAISS index built: {index.ntotal} vectors")
        return index
    
    def save_embeddings(self, embeddings: np.ndarray, chunks: List[Dict], index: faiss.Index):
        """Save all data to shared embeddings directory"""
        print(f"[OptimizedRAG] 💾 Saving embeddings and metadata")
        
        # Save raw vectors
        vectors_path = self.embeddings_dir / "vectors.npy"
        np.save(vectors_path, embeddings)
        print(f"[OptimizedRAG] ✅ Saved vectors: {vectors_path}")
        
        # Save FAISS index
        index_path = self.embeddings_dir / "index.faiss"
        faiss.write_index(index, str(index_path))
        print(f"[OptimizedRAG] ✅ Saved FAISS index: {index_path}")
        
        # Save chunks
        chunks_path = self.embeddings_dir / "doc_chunks.npy"
        np.save(chunks_path, chunks)
        print(f"[OptimizedRAG] ✅ Saved chunks: {chunks_path}")
        
        # Save metadata
        metadata = {
            'total_chunks': len(chunks),
            'embedding_dimension': embeddings.shape[1],
            'model_name': self.model_name,
            'created_at': time.time(),
            'conditions': list(set(chunk['condition'] for chunk in chunks))
        }
        
        metadata_path = self.embeddings_dir / "chunk_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"[OptimizedRAG] ✅ Saved metadata: {metadata_path}")
        
        print(f"[OptimizedRAG] 📊 Summary:")
        print(f"  - Total chunks: {len(chunks)}")
        print(f"  - Embedding dimension: {embeddings.shape[1]}")
        print(f"  - Conditions: {len(metadata['conditions'])}")
        print(f"  - Files created: 4")
    
    def optimize_architecture(self):
        """Run the complete optimization pipeline"""
        print("\n" + "="*80)
        print("  🚀 OPTIMIZING RAG ARCHITECTURE")
        print("="*80)
        
        start_time = time.time()
        
        # Step 1: Load encoder
        self.load_encoder()
        
        # Step 2: Process guidelines
        chunks = self.process_guidelines()
        
        # Step 3: Create embeddings
        embeddings = self.create_embeddings(chunks)
        
        # Step 4: Build FAISS index
        index = self.build_faiss_index(embeddings)
        
        # Step 5: Save everything
        self.save_embeddings(embeddings, chunks, index)
        
        end_time = time.time()
        duration = end_time - start_time
        
        print("\n" + "="*80)
        print("  ✅ OPTIMIZATION COMPLETE!")
        print("="*80)
        print(f"  Duration: {duration:.2f} seconds")
        print(f"  Chunks processed: {len(chunks)}")
        print(f"  Embeddings created: {embeddings.shape}")
        print(f"  Files saved to: {self.embeddings_dir}")
        print("\n  🎯 Next steps:")
        print("  1. Both containers can now use the same data/embeddings/")
        print("  2. Only difference is RAG_MODE toggle (cpu vs gpu)")
        print("  3. No more duplicate JSON processing!")
        print("="*80 + "\n")


def main():
    """Main execution"""
    optimizer = OptimizedRAGArchitecture()
    optimizer.optimize_architecture()


if __name__ == "__main__":
    main()

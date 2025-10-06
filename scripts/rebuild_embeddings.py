#!/usr/bin/env python3
"""
Rebuild RAG embeddings with current FAISS-GPU setup
"""

import os
import sys
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from pathlib import Path
import time

def rebuild_embeddings():
    """Rebuild FAISS index and document chunks"""
    print("🔄 Rebuilding RAG embeddings...")
    
    # Paths
    data_dir = Path("data")
    embeddings_dir = data_dir / "embeddings"
    parsed_dir = data_dir / "parsed"
    input_dir = data_dir / "input"
    
    # Create embeddings directory
    embeddings_dir.mkdir(exist_ok=True)
    
    # Check for parsed text
    parsed_files = list(parsed_dir.glob("*.txt"))
    if not parsed_files:
        print("❌ No parsed text files found in data/parsed/")
        return False
    
    print(f"📄 Found {len(parsed_files)} parsed files")
    
    # Load sentence transformer
    print("🧠 Loading sentence transformer model...")
    
    # Use local model directory directly
    local_model_path = "rag-container/models--sentence-transformers--all-MiniLM-L6-v2/snapshots/c9745ed1d9f207416be6d2e6f8de32d1f16199bf"
    if os.path.exists(local_model_path):
        print(f"📁 Using local model: {local_model_path}")
        
        # Set environment variables to use local cache and avoid downloads
        os.environ['HF_HOME'] = os.path.abspath('rag-container')
        os.environ['TRANSFORMERS_CACHE'] = os.path.abspath('rag-container')
        os.environ['HF_HUB_OFFLINE'] = '1'
        os.environ['TRANSFORMERS_OFFLINE'] = '1'
        
        # Use the local model path directly
        encoder = SentenceTransformer(local_model_path, device='cuda')
        print(f"✅ Loaded local model: {local_model_path}")
    else:
        # Fallback to standard model
        model_name = "all-MiniLM-L6-v2"
        encoder = SentenceTransformer(model_name, device='cuda')
        print(f"✅ Loaded model: {model_name}")
    
    # Read all parsed text
    all_texts = []
    for file_path in parsed_files:
        print(f"📖 Reading {file_path.name}...")
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read().strip()
            if text:
                all_texts.append(text)
    
    if not all_texts:
        print("❌ No text content found in parsed files")
        return False
    
    print(f"📝 Total text length: {sum(len(text) for text in all_texts)} characters")
    
    # Split into chunks (simple chunking)
    chunk_size = 1000  # characters per chunk
    chunks = []
    
    for text in all_texts:
        # Simple chunking by character count
        for i in range(0, len(text), chunk_size):
            chunk = text[i:i + chunk_size].strip()
            if len(chunk) > 100:  # Only keep substantial chunks
                chunks.append(chunk)
    
    print(f"📦 Created {len(chunks)} text chunks")
    
    # Generate embeddings
    print("🔢 Generating embeddings...")
    start_time = time.time()
    
    embeddings = encoder.encode(chunks, convert_to_numpy=True, show_progress_bar=True)
    embeddings = embeddings.astype(np.float32)
    
    print(f"⏱️ Embedding generation took {time.time() - start_time:.2f} seconds")
    print(f"🔢 Embedding shape: {embeddings.shape}")
    
    # Create FAISS index
    print("🔍 Creating FAISS index...")
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
    
    # Normalize embeddings for cosine similarity
    faiss.normalize_L2(embeddings)
    index.add(embeddings)
    
    print(f"✅ FAISS index created with {index.ntotal} vectors")
    
    # Save index, raw vectors, and chunks
    index_path = embeddings_dir / "index.faiss"
    vectors_path = embeddings_dir / "vectors.npy"
    chunks_path = embeddings_dir / "doc_chunks.npy"
    
    print(f"💾 Saving index to {index_path}")
    faiss.write_index(index, str(index_path))
    
    print(f"💾 Saving raw vectors to {vectors_path}")
    np.save(vectors_path, embeddings.astype(np.float32))
    
    print(f"💾 Saving chunks to {chunks_path}")
    np.save(chunks_path, np.array(chunks))
    
    print("✅ Embeddings rebuilt successfully!")
    print(f"📊 Index: {index.ntotal} vectors, dimension: {dimension}")
    print(f"📦 Chunks: {len(chunks)} text chunks")
    
    return True

if __name__ == "__main__":
    success = rebuild_embeddings()
    sys.exit(0 if success else 1)

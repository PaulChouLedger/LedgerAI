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
    
    # Split into chunks with smart overlap for better context
    chunk_size = 1000  # characters per chunk
    overlap = 200  # overlap between chunks to avoid splitting important content
    chunks = []
    
    for text in all_texts:
        # Smart chunking with overlap
        text_len = len(text)
        start = 0
        
        while start < text_len:
            # Get chunk with overlap
            end = start + chunk_size
            chunk = text[start:end].strip()
            
            # Only keep substantial chunks
            if len(chunk) > 100:
                chunks.append(chunk)
            
            # Move forward by chunk_size - overlap
            start += (chunk_size - overlap)
            
            # If we're at the end, break
            if end >= text_len:
                break
    
    print(f"📦 Created {len(chunks)} text chunks")
    
    # Verify important content is in chunks
    print("\n🔍 Verifying important names are in chunks...")
    important_names = ["Bob Carella", "David Lara", "Paul Chou"]
    for name in important_names:
        found_in = []
        for i, chunk in enumerate(chunks):
            if name in chunk:
                found_in.append(i)
        if found_in:
            print(f"  ✅ '{name}' found in {len(found_in)} chunk(s): {found_in[:3]}...")
        else:
            print(f"  ❌ '{name}' NOT FOUND in any chunks!")
    
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
    
    # Test the index with sample queries
    print("\n🧪 Testing index with sample queries...")
    test_queries = ["Who is David Lara?", "Who is Bob Carella?", "What is AuraVision?"]
    
    for query in test_queries:
        query_embedding = encoder.encode([query], convert_to_numpy=True).astype(np.float32)
        faiss.normalize_L2(query_embedding)
        distances, indices = index.search(query_embedding, 3)
        
        print(f"\n  Query: '{query}'")
        print(f"  Top 3 results:")
        for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
            idx = int(idx)
            if idx < len(chunks):
                preview = chunks[idx][:100].replace('\n', ' ')
                print(f"    {i+1}. idx={idx}, distance={dist:.4f}, preview: '{preview}...'")
        
        # Check for duplicate indices
        if len(set(indices[0])) < len(indices[0]):
            print(f"  ⚠️ WARNING: Search returned duplicate indices!")
        elif all(indices[0] == 0):
            print(f"  ❌ CRITICAL: All results point to index 0!")
        else:
            print(f"  ✅ Search returns diverse results")
    
    return True

if __name__ == "__main__":
    success = rebuild_embeddings()
    sys.exit(0 if success else 1)

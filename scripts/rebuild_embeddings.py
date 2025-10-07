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
    
    # Split into chunks with paragraph-aware splitting
    chunk_size = 1000  # target characters per chunk
    overlap = 200  # overlap between chunks
    chunks = []
    
    for text in all_texts:
        # Split by double newlines (paragraph boundaries) first
        paragraphs = text.split('\n\n')
        
        current_chunk = ""
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            
            # If adding this paragraph would exceed chunk size
            if len(current_chunk) + len(para) > chunk_size and current_chunk:
                # Save current chunk
                if len(current_chunk) > 100:
                    chunks.append(current_chunk.strip())
                
                # Start new chunk with overlap from previous
                # Take last 'overlap' characters from previous chunk
                if len(current_chunk) > overlap:
                    current_chunk = current_chunk[-overlap:] + "\n\n" + para
                else:
                    current_chunk = para
            else:
                # Add paragraph to current chunk
                if current_chunk:
                    current_chunk += "\n\n" + para
                else:
                    current_chunk = para
        
        # Don't forget the last chunk
        if current_chunk and len(current_chunk) > 100:
            chunks.append(current_chunk.strip())
    
    print(f"📦 Created {len(chunks)} text chunks")
    
    # Verify important content is in chunks
    print("\n🔍 Verifying important names are in chunks...")
    important_names = ["Bob Carella", "David Lara", "Paul Chou"]
    name_chunks = {}  # Store chunk indices for later testing
    
    for name in important_names:
        found_in = []
        for i, chunk in enumerate(chunks):
            if name in chunk:
                found_in.append(i)
        
        name_chunks[name] = found_in
        
        if found_in:
            print(f"  ✅ '{name}' found in {len(found_in)} chunk(s): {found_in}")
            
            # Validate that chunks actually START with or prominently feature this person
            for chunk_idx in found_in:
                chunk = chunks[chunk_idx]
                name_pos = chunk.find(name)
                chunk_start = chunk[:100].replace('\n', ' ')
                
                # Check if name appears early in the chunk (within first 200 chars)
                if name_pos < 200:
                    print(f"     ✅ Chunk {chunk_idx}: '{name}' at position {name_pos}")
                    print(f"        Start: '{chunk_start}...'")
                else:
                    print(f"     ⚠️ Chunk {chunk_idx}: '{name}' at position {name_pos} (late in chunk!)")
                    print(f"        Start: '{chunk_start}...'")
                    # Show where the name actually appears
                    context_start = max(0, name_pos - 30)
                    context_end = min(len(chunk), name_pos + 100)
                    name_context = chunk[context_start:context_end].replace('\n', ' ')
                    print(f"        Name context: '...{name_context}...'")
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
        distances, indices = index.search(query_embedding, 10)  # Get top 10 to see ranking
        
        print(f"\n  Query: '{query}'")
        print(f"  Top 10 results:")
        
        # Extract the person's name from query if present
        query_name = None
        for name in important_names:
            if name in query:
                query_name = name
                break
        
        expected_chunks = name_chunks.get(query_name, []) if query_name else []
        
        for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
            idx = int(idx)
            if idx < len(chunks):
                preview = chunks[idx][:100].replace('\n', ' ')
                
                # Check if this chunk contains the expected name
                contains_name = ""
                if query_name and query_name in chunks[idx]:
                    contains_name = f" ✅ CONTAINS '{query_name}'"
                elif idx in expected_chunks:
                    contains_name = f" ✅ Expected chunk"
                
                print(f"    {i+1}. idx={idx}, distance={dist:.4f}{contains_name}")
                if i < 3:  # Show preview for top 3
                    print(f"       preview: '{preview}...'")
        
        # Check if expected chunks are in top 3
        if expected_chunks:
            top_3_indices = [int(idx) for idx in indices[0][:3]]
            found_in_top_3 = any(chunk_idx in top_3_indices for chunk_idx in expected_chunks)
            if found_in_top_3:
                print(f"  ✅ Found '{query_name}' in top 3 results")
            else:
                print(f"  ⚠️ WARNING: '{query_name}' NOT in top 3!")
                # Show where it actually ranked
                for chunk_idx in expected_chunks:
                    if chunk_idx in indices[0]:
                        rank = list(indices[0]).index(chunk_idx) + 1
                        print(f"     '{query_name}' chunk {chunk_idx} ranked #{rank}")
        
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

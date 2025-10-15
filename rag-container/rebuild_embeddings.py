#!/usr/bin/env python3
"""
Rebuild RAG embeddings with current FAISS-GPU setup

Supports multiple document types:
- Person bios (team pages, about us sections)
- Technical documentation (APIs, guides)
- General content (articles, reports)

Chunking strategies:
1. Content-aware: Detects person bios, headers, section breaks
2. Paragraph-aware: Respects paragraph boundaries
3. Fixed-size: Falls back to simple character-based chunking
"""

import os
import sys
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from pathlib import Path
import time

# === Configuration ===
CHUNK_SIZE = 1000  # Target characters per chunk
OVERLAP = 200  # Overlap between chunks to preserve context
USE_SMART_CHUNKING = True  # Use content-aware chunking (True) or simple splitting (False)
DETECT_PERSON_BIOS = True  # Enable person bio detection for team/about pages

# === Usage Guide ===
# For your document type, adjust these settings:
#
# 1. TEAM/BIO PAGES (like this PDF):
#    USE_SMART_CHUNKING = True
#    DETECT_PERSON_BIOS = True
#
# 2. TECHNICAL DOCS (APIs, guides):
#    USE_SMART_CHUNKING = True
#    DETECT_PERSON_BIOS = False
#    (Will use paragraph-aware chunking)
#
# 3. GENERAL CONTENT (articles, reports):
#    USE_SMART_CHUNKING = False
#    DETECT_PERSON_BIOS = False  
#    (Will use simple fixed-size chunking)
#
# 4. MIXED CONTENT:
#    USE_SMART_CHUNKING = True
#    DETECT_PERSON_BIOS = True
#    (Will auto-detect and use best strategy per section)

def rebuild_embeddings(data_root="/app/data"):
    """Rebuild FAISS index and document chunks"""
    print("🔄 Rebuilding RAG embeddings...")
    
    # Paths (container-safe)
    data_dir = Path(data_root)
    embeddings_dir = data_dir / "embeddings"
    parsed_dir = data_dir / "parsed"
    input_dir = data_dir / "input"
    
    # Create embeddings directory
    embeddings_dir.mkdir(parents=True, exist_ok=True)
    
    # Check for parsed text
    parsed_files = list(parsed_dir.glob("*.txt"))
    if not parsed_files:
        print("❌ No parsed text files found in data/parsed/")
        return False
    
    print(f"📄 Found {len(parsed_files)} parsed files")
    
    # Load sentence transformer
    print("🧠 Loading sentence transformer model...")
    
    # Model is pre-downloaded in container (WiFi-independent)
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
    
    # Split into chunks based on configuration
    import re
    chunks = []
    
    print(f"\n📐 Chunking strategy: {'Smart content-aware' if USE_SMART_CHUNKING else 'Simple fixed-size'}")
    print(f"   Chunk size: {CHUNK_SIZE} chars, Overlap: {OVERLAP} chars")
    print(f"   Bio detection: {'Enabled' if DETECT_PERSON_BIOS else 'Disabled'}")
    
    for text in all_texts:
        # Try smart chunking if enabled
        bio_starts = []
        
        if USE_SMART_CHUNKING and DETECT_PERSON_BIOS:
            # Multiple patterns to detect section breaks (works for various document types)
            # Note: Use \s* to handle leading whitespace before names
            patterns = [
                re.compile(r'\n\s*([A-Z][a-z]+ [A-Z][a-z]+) is (a|an|the) ', re.MULTILINE),  # "Name is a/an/the"
                re.compile(r'\n\s*([A-Z][a-z]+ [A-Z][a-z]+) was (a|an|the) ', re.MULTILINE),  # "Name was"
                re.compile(r'\n\s*([A-Z][a-z]+ [A-Z][a-z]+) has been ', re.MULTILINE),  # "Name has been"
                re.compile(r'\n\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+) is (a|an|the) ', re.MULTILINE),  # Multi-word names
            ]
            
            # Combine all pattern matches
            bio_pattern = patterns[0]  # Use first pattern as primary
            
            # Find all bio starts
            for match in bio_pattern.finditer(text):
                bio_starts.append((match.start(), match.group(1)))
            
            if bio_starts:
                print(f"\n🔍 Found {len(bio_starts)} person bios in text")
                for pos, name in bio_starts:
                    print(f"  - {name} at position {pos}")
        
        # If we found bio markers and smart chunking is enabled, use them to split
        if bio_starts and USE_SMART_CHUNKING:
            last_pos = 0
            
            for i, (start_pos, name) in enumerate(bio_starts):
                # Get text from last position to current bio start
                if last_pos < start_pos:
                    before_bio = text[last_pos:start_pos].strip()
                    # Split this section normally if it's large
                    if len(before_bio) > CHUNK_SIZE:
                        # Chunk the pre-bio content
                        for j in range(0, len(before_bio), CHUNK_SIZE - OVERLAP):
                            chunk = before_bio[j:j + CHUNK_SIZE].strip()
                            if len(chunk) > 100:
                                chunks.append(chunk)
                    elif len(before_bio) > 100:
                        chunks.append(before_bio)
                
                # Get bio content (from this bio start to next bio start or end)
                if i < len(bio_starts) - 1:
                    next_pos = bio_starts[i + 1][0]
                else:
                    next_pos = len(text)
                
                bio_content = text[start_pos:next_pos].strip()
                
                # If bio is longer than chunk size, split it with overlap
                if len(bio_content) > CHUNK_SIZE:
                    for j in range(0, len(bio_content), CHUNK_SIZE - OVERLAP):
                        chunk = bio_content[j:j + CHUNK_SIZE].strip()
                        if len(chunk) > 100:
                            chunks.append(chunk)
                            print(f"  ✅ Created chunk for {name} (part {j // (CHUNK_SIZE - OVERLAP) + 1})")
                else:
                    chunks.append(bio_content)
                    print(f"  ✅ Created chunk for {name}")
                
                last_pos = next_pos
            
            # Handle any remaining text after last bio
            if last_pos < len(text):
                remaining = text[last_pos:].strip()
                if len(remaining) > 100:
                    chunks.append(remaining)
        else:
            # No bio markers found or smart chunking disabled, fall back to paragraph splitting
            print(f"\n📄 Using paragraph-aware chunking (no bio markers found or disabled)")
            paragraphs = text.split('\n\n')
            current_chunk = ""
            
            for para in paragraphs:
                para = para.strip()
                if not para:
                    continue
                
                if len(current_chunk) + len(para) > CHUNK_SIZE and current_chunk:
                    if len(current_chunk) > 100:
                        chunks.append(current_chunk.strip())
                    
                    if len(current_chunk) > OVERLAP:
                        current_chunk = current_chunk[-OVERLAP:] + "\n\n" + para
                    else:
                        current_chunk = para
                else:
                    if current_chunk:
                        current_chunk += "\n\n" + para
                    else:
                        current_chunk = para
            
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
    
    # Generate embeddings (all at once like before - this ALWAYS worked)
    print("🔢 Generating embeddings...")
    start_time = time.time()
    
    # Use sentence_transformers default (this creates proper numpy arrays)
    embeddings = encoder.encode(chunks, convert_to_numpy=True, show_progress_bar=True)
    
    print(f"⏱️ Embedding generation took {time.time() - start_time:.2f} seconds")
    print(f"🔢 Embedding shape: {embeddings.shape}")
    print(f"🔢 Original dtype: {embeddings.dtype}")
    
    # Convert to float32 if needed (sentence_transformers sometimes returns float64)
    if embeddings.dtype != np.float32:
        print(f"🔧 Converting {embeddings.dtype} → float32")
        embeddings = embeddings.astype(np.float32, copy=False)
    
    # Normalize for cosine similarity using FAISS built-in
    print("🔧 Normalizing embeddings for cosine similarity...")
    faiss.normalize_L2(embeddings)  # In-place normalization
    
    print(f"✅ Embeddings ready:")
    print(f"   Type: {type(embeddings)}")
    print(f"   Dtype: {embeddings.dtype}")
    print(f"   Shape: {embeddings.shape}")
    print(f"   C-contiguous: {embeddings.flags['C_CONTIGUOUS']}")
    
    # Create FAISS index
    print("🔍 Creating FAISS index...")
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
    
    # Add to FAISS index
    print(f"🔍 Adding {embeddings.shape[0]} vectors to FAISS index...")
    index.add(embeddings)
    
    print(f"✅ FAISS index created with {index.ntotal} vectors")
    
    # Save index, raw vectors, and chunks
    index_path = embeddings_dir / "index.faiss"
    vectors_path = embeddings_dir / "vectors.npy"
    chunks_path = embeddings_dir / "doc_chunks.npy"
    
    print(f"💾 Saving index to {index_path}")
    faiss.write_index(index, str(index_path))
    
    print(f"💾 Saving raw vectors to {vectors_path}")
    np.save(vectors_path, embeddings)
    
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
        # Use FAISS built-in normalization (most reliable)
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

#!/usr/bin/env python3
"""
Rebuild RAG embeddings with current FAISS-GPU setup

IMPORTANT: This script uses all-mpnet-base-v2 (768 dimensions) to match
the RAG container configuration. Do not change the model without updating
rag.py to match!

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

def rebuild_embeddings(data_root="data"):
    """Rebuild FAISS index and document chunks"""
    print("🔄 Rebuilding RAG embeddings...")
    
    # Paths (host default, container can override)
    data_dir = Path(data_root)
    embeddings_dir = data_dir / "embeddings"
    parsed_dir = data_dir / "parsed"
    input_dir = data_dir / "input"
    
    # Create embeddings directory
    embeddings_dir.mkdir(parents=True, exist_ok=True)
    
    # Check for parsed text (RAG ingest already copied everything from data/input to data/parsed)
    parsed_files = list(parsed_dir.glob("*.txt"))
    if not parsed_files:
        print("❌ No parsed text files found in data/parsed/")
        return False
    
    print(f"📄 Found {len(parsed_files)} parsed files")
    
    # Load sentence transformer
    print("🧠 Loading sentence transformer model...")
    
    # Use the same model as RAG container (all-mpnet-base-v2)
    import os
    model_name = "all-mpnet-base-v2"  # Must match rag.py model_name
    
    # Check for local model directory first
    local_model_path = f"rag-container/models--sentence-transformers--{model_name.replace('-', '--')}/snapshots"
    if os.path.exists(local_model_path):
        # Find the actual snapshot directory
        snapshots = [d for d in os.listdir(local_model_path) if os.path.isdir(os.path.join(local_model_path, d))]
        if snapshots:
            full_local_path = os.path.join(local_model_path, snapshots[0])
            print(f"📁 Using local model: {full_local_path}")
            
            # Set environment to use local cache and avoid downloads
            os.environ['HF_HOME'] = os.path.abspath('rag-container')
            os.environ['TRANSFORMERS_CACHE'] = os.path.abspath('rag-container')
            os.environ['HF_HUB_OFFLINE'] = '1'
            os.environ['TRANSFORMERS_OFFLINE'] = '1'
            
            encoder = SentenceTransformer(full_local_path, device='cuda')
            print(f"✅ Loaded local model: {full_local_path}")
        else:
            # No snapshots found, download the model
            print(f"📥 No local snapshots found, downloading {model_name}...")
            encoder = SentenceTransformer(model_name, device='cuda')
            print(f"✅ Loaded model: {model_name}")
    else:
        # No local model directory, download the model
        print(f"📥 No local model found, downloading {model_name}...")
        encoder = SentenceTransformer(model_name, device='cuda')
        print(f"✅ Loaded model: {model_name}")
    
    # Read all parsed text with source tracking
    all_texts = []
    text_sources = []  # Track which file each text came from
    
    for file_path in parsed_files:
        print(f"📖 Reading {file_path.name}...")
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read().strip()
            if text:
                all_texts.append(text)
                text_sources.append(file_path.name)
    
    if not all_texts:
        print("❌ No text content found in parsed files")
        return False
    
    print(f"📝 Total text length: {sum(len(text) for text in all_texts)} characters")
    
    # Split into chunks based on configuration
    import re
    chunks = []
    chunk_metadata = []  # NEW: Track metadata for each chunk
    
    # Helper function to add chunk with metadata
    def add_chunk(chunk_text, source_file, guideline_name=None, section_type="unknown"):
        """Add chunk and its metadata"""
        chunks.append(chunk_text)
        
        metadata = {
            "chunk_id": len(chunks) - 1,
            "source_file": source_file,
            "char_length": len(chunk_text)
        }
        
        # Add guideline-specific metadata
        if guideline_name:
            metadata["guideline_name"] = guideline_name
            metadata["is_medical_guideline"] = True
            
            # Detect section type from chunk content
            chunk_lower = chunk_text.lower()
            if "diagnostic questioning strategy" in chunk_lower or "question 1:" in chunk_lower:
                metadata["section_type"] = "diagnostic_questions"
            elif "red flag" in chunk_lower or "emergency warning" in chunk_lower:
                metadata["section_type"] = "red_flags"
            elif "differential diagnos" in chunk_lower:
                metadata["section_type"] = "differentials"
            elif "classic presentation" in chunk_lower:
                metadata["section_type"] = "presentation"
            else:
                metadata["section_type"] = section_type
        else:
            metadata["is_medical_guideline"] = False
            metadata["section_type"] = "general_content"
        
        chunk_metadata.append(metadata)
    
    print(f"\n📐 Chunking strategy: {'Smart content-aware' if USE_SMART_CHUNKING else 'Simple fixed-size'}")
    print(f"   Chunk size: {CHUNK_SIZE} chars, Overlap: {OVERLAP} chars")
    print(f"   Bio detection: {'Enabled' if DETECT_PERSON_BIOS else 'Disabled'}")
    
    for text_idx, text in enumerate(all_texts):
        source_file = text_sources[text_idx]
        
        # Extract guideline metadata if this is a GUIDELINE file
        guideline_name = None
        if source_file.startswith('GUIDELINE_'):
            # Extract from first few lines (header might be after separator line)
            first_lines = '\n'.join(text.split('\n')[:5])  # Check first 5 lines
            import re
            match = re.search(r'DIAGNOSTIC GUIDELINE:\s*([^\n]+)', first_lines, re.IGNORECASE)
            if match:
                guideline_name = match.group(1).strip()
                print(f"\n📋 Processing medical guideline: {guideline_name} (from {source_file})")
            else:
                print(f"\n⚠️ File {source_file} looks like a guideline but no header found - checking content...")
        
        print(f"\n📄 Processing {source_file}...")
        file_chunks_start = len(chunks)  # Track where this file's chunks start
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
                                add_chunk(chunk, source_file, guideline_name, "pre_bio")
                    elif len(before_bio) > 100:
                        add_chunk(before_bio, source_file, guideline_name, "pre_bio")
                
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
                            add_chunk(chunk, source_file, guideline_name, "person_bio")
                            print(f"  ✅ Created chunk for {name} (part {j // (CHUNK_SIZE - OVERLAP) + 1})")
                else:
                    add_chunk(bio_content, source_file, guideline_name, "person_bio")
                    print(f"  ✅ Created chunk for {name}")
                
                last_pos = next_pos
            
            # Handle any remaining text after last bio
            if last_pos < len(text):
                remaining = text[last_pos:].strip()
                if len(remaining) > 100:
                    add_chunk(remaining, source_file, guideline_name, "remaining")
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
                        add_chunk(current_chunk.strip(), source_file, guideline_name, "paragraph")
                    
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
                add_chunk(current_chunk.strip(), source_file, guideline_name, "paragraph")
    
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
    
    # Save chunk metadata
    metadata_path = embeddings_dir / "chunk_metadata.json"
    print(f"💾 Saving chunk metadata to {metadata_path}")
    
    import json
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(chunk_metadata, f, indent=2, ensure_ascii=False)
    
    # Count medical guideline chunks
    guideline_chunks = [m for m in chunk_metadata if m.get('is_medical_guideline')]
    guidelines_found = set([m['guideline_name'] for m in guideline_chunks if 'guideline_name' in m])
    
    print("✅ Embeddings rebuilt successfully!")
    print(f"📊 Index: {index.ntotal} vectors, dimension: {dimension}")
    print(f"📦 Chunks: {len(chunks)} text chunks")
    print(f"📋 Medical guidelines: {len(guidelines_found)} guidelines, {len(guideline_chunks)} chunks")
    
    if guidelines_found:
        print(f"   Guidelines indexed:")
        for gname in sorted(guidelines_found):
            gcount = len([m for m in guideline_chunks if m.get('guideline_name') == gname])
            print(f"   - {gname}: {gcount} chunks")
    
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

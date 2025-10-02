#!/usr/bin/env python3
"""
Generate doc_chunks.npy from existing text data for RAG system
"""
import os
import numpy as np
import re
from typing import List

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """
    Split text into overlapping chunks
    
    Args:
        text: Input text to chunk
        chunk_size: Target size of each chunk in characters
        overlap: Number of characters to overlap between chunks
        
    Returns:
        List of text chunks
    """
    if len(text) <= chunk_size:
        return [text.strip()]
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        
        # If we're not at the end, try to break at a sentence boundary
        if end < len(text):
            # Look for sentence endings within the last 100 characters
            search_start = max(start + chunk_size - 100, start)
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
        start = end - overlap
        if start >= len(text):
            break
    
    return chunks

def process_text_file(file_path: str) -> List[str]:
    """Process a text file and return chunks"""
    print(f"📄 Processing: {file_path}")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # Clean up the text
        text = re.sub(r'\n\s*\n', '\n\n', text)  # Normalize paragraph breaks
        text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
        text = text.strip()
        
        if not text:
            print(f"⚠️  Empty file: {file_path}")
            return []
        
        # Generate chunks
        chunks = chunk_text(text, chunk_size=400, overlap=50)
        print(f"✅ Generated {len(chunks)} chunks from {file_path}")
        
        return chunks
        
    except Exception as e:
        print(f"❌ Error processing {file_path}: {e}")
        return []

def generate_doc_chunks():
    """Generate doc_chunks.npy from available text data"""
    print("🔧 Generating document chunks for RAG system...")
    
    # Check current directory
    if not os.path.exists("data"):
        print("❌ data/ directory not found. Please run from LedgerAI root directory.")
        return False
    
    all_chunks = []
    
    # Process parsed text file
    parsed_file = "data/parsed/ledgerai.txt"
    if os.path.exists(parsed_file):
        chunks = process_text_file(parsed_file)
        all_chunks.extend(chunks)
    else:
        print(f"⚠️  Parsed file not found: {parsed_file}")
    
    # You can add more text files here if needed
    # For example, if you have other medical documents:
    # additional_files = ["data/other_medical_docs.txt"]
    # for file_path in additional_files:
    #     if os.path.exists(file_path):
    #         chunks = process_text_file(file_path)
    #         all_chunks.extend(chunks)
    
    if not all_chunks:
        print("❌ No text chunks generated. Please check your input files.")
        return False
    
    # Save chunks to numpy file
    output_path = "data/embeddings/doc_chunks.npy"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    try:
        # Convert to numpy array
        chunks_array = np.array(all_chunks, dtype=object)
        np.save(output_path, chunks_array)
        
        print(f"✅ Saved {len(all_chunks)} chunks to {output_path}")
        print(f"📊 File size: {os.path.getsize(output_path)} bytes")
        
        # Show sample chunks
        print(f"\n📝 Sample chunks:")
        for i, chunk in enumerate(all_chunks[:3]):
            print(f"  {i+1}. {chunk[:100]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ Error saving chunks: {e}")
        return False

def verify_chunks():
    """Verify the generated chunks file"""
    chunks_path = "data/embeddings/doc_chunks.npy"
    
    if not os.path.exists(chunks_path):
        print(f"❌ Chunks file not found: {chunks_path}")
        return False
    
    try:
        chunks = np.load(chunks_path, allow_pickle=True)
        print(f"✅ Verification: Loaded {len(chunks)} chunks")
        print(f"📄 Sample chunk: {chunks[0][:150]}...")
        return True
    except Exception as e:
        print(f"❌ Error verifying chunks: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Starting document chunk generation...")
    
    if generate_doc_chunks():
        print("\n🔍 Verifying generated chunks...")
        if verify_chunks():
            print("\n🎉 Document chunks generated successfully!")
            print("💡 You can now restart your Aura system to use RAG.")
        else:
            print("\n❌ Chunk verification failed.")
    else:
        print("\n❌ Failed to generate document chunks.")

#!/usr/bin/env python3
"""
Convert GPU FAISS embeddings to CPU FAISS format
Places files in the correct non-nested data/embeddings/ directory
"""

import os
import pickle
import numpy as np
import faiss
from pathlib import Path

def convert_gpu_to_cpu_faiss():
    """Convert GPU FAISS format to CPU FAISS format"""
    
    print("\n" + "="*80)
    print("  🔄 CONVERTING EMBEDDINGS FOR CPU FAISS")
    print("="*80)
    
    # Source and target directories
    source_dir = Path("data/embeddings")
    target_dir = Path("data/embeddings")  # Same directory, different file formats
    
    print(f"[CPU FAISS Converter] 📂 Source: {source_dir}")
    print(f"[CPU FAISS Converter] 📂 Target: {target_dir}")
    
    # Load source embeddings
    print("[CPU FAISS Converter] 🔧 Loading source embeddings...")
    
    try:
        # Load FAISS index
        faiss_index_path = source_dir / "index.faiss"
        index = faiss.read_index(str(faiss_index_path))
        print(f"[CPU FAISS Converter] ✅ Loaded FAISS index: {index.ntotal} vectors")
        
        # Load chunks
        chunks_path = source_dir / "doc_chunks.npy"
        chunks = np.load(chunks_path, allow_pickle=True)
        print(f"[CPU FAISS Converter] ✅ Loaded chunks: {len(chunks)}")
        
        # Load metadata
        metadata_path = source_dir / "chunk_metadata.json"
        import json
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        print(f"[CPU FAISS Converter] ✅ Loaded metadata: {metadata['total_chunks']} chunks")
        
    except Exception as e:
        print(f"[CPU FAISS Converter] ❌ Failed to load source embeddings: {e}")
        return False
    
    # Convert to CPU FAISS format
    print("[CPU FAISS Converter] 🔧 Converting to CPU FAISS format...")
    
    try:
        # Save CPU FAISS index (same format, just different filename)
        cpu_index_path = target_dir / "faiss_index.bin"
        faiss.write_index(index, str(cpu_index_path))
        print(f"[CPU FAISS Converter] ✅ Saved CPU FAISS index: {cpu_index_path}")
        
        # Save CPU metadata (pickle format)
        cpu_metadata = {
            "total_chunks": len(chunks),
            "embedding_dimension": index.d,
            "model_name": metadata.get("model_name", "all-distilroberta-v1"),
            "chunks": chunks.tolist(),
            "metadata": metadata  # Use the full metadata structure
        }
        
        cpu_metadata_path = target_dir / "metadata.pkl"
        with open(cpu_metadata_path, 'wb') as f:
            pickle.dump(cpu_metadata, f)
        print(f"[CPU FAISS Converter] ✅ Saved CPU metadata: {cpu_metadata_path}")
        
    except Exception as e:
        print(f"[CPU FAISS Converter] ❌ Failed to convert: {e}")
        return False
    
    # Verify conversion
    print("[CPU FAISS Converter] 🔍 Verifying conversion...")
    
    try:
        # Verify CPU index
        cpu_index = faiss.read_index(str(cpu_index_path))
        print(f"[CPU FAISS Converter] ✅ CPU index verification: {cpu_index.ntotal} vectors")
        
        # Verify CPU metadata
        with open(cpu_metadata_path, 'rb') as f:
            cpu_meta = pickle.load(f)
        print(f"[CPU FAISS Converter] ✅ CPU metadata verification: {cpu_meta['total_chunks']} chunks")
        
    except Exception as e:
        print(f"[CPU FAISS Converter] ❌ Verification failed: {e}")
        return False
    
    # Summary
    print(f"\n[CPU FAISS Converter] 📊 Conversion Summary:")
    print(f"  - Source vectors: {index.ntotal}")
    print(f"  - Source chunks: {len(chunks)}")
    print(f"  - CPU index file: {cpu_index_path}")
    print(f"  - CPU metadata file: {cpu_metadata_path}")
    print(f"  - Embedding dimension: {index.d}")
    print(f"  - Model: {metadata.get('model_name', 'all-distilroberta-v1')}")
    
    print(f"\n[CPU FAISS Converter] ✅ Conversion complete!")
    print(f"  The llm-generic-container can now use CPU FAISS with the shared embeddings.")
    
    print("="*80)
    print("  ✅ CPU FAISS CONVERSION COMPLETE!")
    print("="*80)
    print("  Both containers can now use the same underlying data:")
    print("  - GPU FAISS (rag-container): Uses data/embeddings/")
    print("  - CPU FAISS (llm-generic-container): Uses data/embeddings/")
    print("  - Same embeddings, different file formats")
    print("="*80)
    
    return True

if __name__ == "__main__":
    convert_gpu_to_cpu_faiss()
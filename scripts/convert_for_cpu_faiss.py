#!/usr/bin/env python3
"""
Convert optimized embeddings to CPU FAISS format

This script converts the shared embeddings (created by optimize_rag_architecture.py)
to the format expected by the CPU FAISS implementation in llm-medical-container.
"""

import os
import json
import pickle
import numpy as np
import faiss
from pathlib import Path

def convert_embeddings_for_cpu_faiss():
    """Convert shared embeddings to CPU FAISS format"""
    
    # Source files (from optimize_rag_architecture.py)
    source_dir = Path("data/embeddings")
    
    # Target directory (for llm-medical-container CPU FAISS)
    target_dir = Path("llm-medical-container/data/embeddings")
    target_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"[CPU FAISS Converter] 📂 Source: {source_dir}")
    print(f"[CPU FAISS Converter] 📂 Target: {target_dir}")
    
    # Load source files
    print(f"[CPU FAISS Converter] 🔧 Loading source embeddings...")
    
    # Load FAISS index
    index_path = source_dir / "index.faiss"
    if not index_path.exists():
        raise FileNotFoundError(f"Source index not found: {index_path}")
    
    index = faiss.read_index(str(index_path))
    print(f"[CPU FAISS Converter] ✅ Loaded FAISS index: {index.ntotal} vectors")
    
    # Load chunks
    chunks_path = source_dir / "doc_chunks.npy"
    if not chunks_path.exists():
        raise FileNotFoundError(f"Source chunks not found: {chunks_path}")
    
    chunks = np.load(chunks_path, allow_pickle=True)
    print(f"[CPU FAISS Converter] ✅ Loaded chunks: {len(chunks)}")
    
    # Load metadata
    metadata_path = source_dir / "chunk_metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Source metadata not found: {metadata_path}")
    
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    print(f"[CPU FAISS Converter] ✅ Loaded metadata: {metadata['total_chunks']} chunks")
    
    # Convert to CPU FAISS format
    print(f"[CPU FAISS Converter] 🔧 Converting to CPU FAISS format...")
    
    # Save FAISS index as .bin file
    cpu_index_path = target_dir / "faiss_index.bin"
    faiss.write_index(index, str(cpu_index_path))
    print(f"[CPU FAISS Converter] ✅ Saved CPU FAISS index: {cpu_index_path}")
    
    # Prepare metadata for pickle format
    cpu_metadata = {
        'chunks': chunks.tolist(),  # Convert numpy array to list
        'metadata': metadata,
        'total_chunks': len(chunks),
        'embedding_dimension': index.d,
        'model_name': metadata.get('model_name', 'all-distilroberta-v1')
    }
    
    # Save metadata as pickle file
    cpu_metadata_path = target_dir / "metadata.pkl"
    with open(cpu_metadata_path, 'wb') as f:
        pickle.dump(cpu_metadata, f)
    print(f"[CPU FAISS Converter] ✅ Saved CPU metadata: {cpu_metadata_path}")
    
    # Verify files
    print(f"[CPU FAISS Converter] 🔍 Verifying conversion...")
    
    # Test loading the CPU FAISS index
    test_index = faiss.read_index(str(cpu_index_path))
    print(f"[CPU FAISS Converter] ✅ CPU index verification: {test_index.ntotal} vectors")
    
    # Test loading the CPU metadata
    with open(cpu_metadata_path, 'rb') as f:
        test_metadata = pickle.load(f)
    print(f"[CPU FAISS Converter] ✅ CPU metadata verification: {len(test_metadata['chunks'])} chunks")
    
    print(f"\n[CPU FAISS Converter] 📊 Conversion Summary:")
    print(f"  - Source vectors: {index.ntotal}")
    print(f"  - Source chunks: {len(chunks)}")
    print(f"  - CPU index file: {cpu_index_path}")
    print(f"  - CPU metadata file: {cpu_metadata_path}")
    print(f"  - Embedding dimension: {index.d}")
    print(f"  - Model: {metadata.get('model_name', 'all-distilroberta-v1')}")
    
    print(f"\n[CPU FAISS Converter] ✅ Conversion complete!")
    print(f"  The llm-medical-container can now use CPU FAISS with the shared embeddings.")
    
    return True

def main():
    """Main execution"""
    print("\n" + "="*80)
    print("  🔄 CONVERTING EMBEDDINGS FOR CPU FAISS")
    print("="*80)
    
    try:
        convert_embeddings_for_cpu_faiss()
        
        print("\n" + "="*80)
        print("  ✅ CPU FAISS CONVERSION COMPLETE!")
        print("="*80)
        print("  Both containers can now use the same underlying data:")
        print("  - GPU FAISS (rag-container): Uses data/embeddings/")
        print("  - CPU FAISS (llm-medical-container): Uses llm-medical-container/data/embeddings/")
        print("  - Same embeddings, different file formats")
        print("="*80 + "\n")
        
    except Exception as e:
        print(f"\n[CPU FAISS Converter] ❌ Conversion failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    main()

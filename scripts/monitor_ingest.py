#!/usr/bin/env python3
"""
Monitor script for auto-ingest pipeline
Shows current status and statistics
"""
import json
import os
from pathlib import Path
from datetime import datetime
import numpy as np
import faiss

def load_state():
    """Load ingest state"""
    state_file = Path("data/ingest_state.json")
    if state_file.exists():
        with open(state_file, 'r') as f:
            return json.load(f)
    return {}

def check_embeddings():
    """Check embeddings files"""
    embeddings_dir = Path("data/embeddings")
    
    index_path = embeddings_dir / "index.faiss"
    chunks_path = embeddings_dir / "doc_chunks.npy"
    
    info = {}
    
    if index_path.exists():
        try:
            index = faiss.read_index(str(index_path))
            info["faiss_vectors"] = index.ntotal
            info["faiss_dimension"] = index.d
            info["faiss_size_mb"] = index_path.stat().st_size / (1024 * 1024)
        except Exception as e:
            info["faiss_error"] = str(e)
    else:
        info["faiss_exists"] = False
    
    if chunks_path.exists():
        try:
            chunks = np.load(chunks_path, allow_pickle=True)
            info["chunks_count"] = len(chunks)
            info["chunks_size_mb"] = chunks_path.stat().st_size / (1024 * 1024)
            info["sample_chunk"] = chunks[0][:100] + "..." if len(chunks) > 0 else ""
        except Exception as e:
            info["chunks_error"] = str(e)
    else:
        info["chunks_exists"] = False
    
    return info

def show_status():
    """Show current auto-ingest status"""
    print("🔍 Aura Auto-Ingest Status")
    print("=" * 50)
    
    # Check directories
    dirs = {
        "Input": Path("data/input"),
        "Parsed": Path("data/parsed"),
        "Embeddings": Path("data/embeddings")
    }
    
    print("\n📁 Directories:")
    for name, path in dirs.items():
        if path.exists():
            count = len(list(path.glob("*"))) if path.is_dir() else 0
            print(f"  ✅ {name}: {path} ({count} files)")
        else:
            print(f"  ❌ {name}: {path} (not found)")
    
    # Load state
    state = load_state()
    
    print("\n📊 Processing State:")
    if state:
        processed_files = state.get("processed_files", {})
        print(f"  📄 Processed files: {len(processed_files)}")
        print(f"  📅 Last scan: {state.get('last_scan', 'Never')}")
        print(f"  🔢 Total chunks: {state.get('total_chunks', 0)}")
        
        if processed_files:
            print(f"\n📋 Processed Files:")
            for filename, info in processed_files.items():
                timestamp = info.get('timestamp', 'Unknown')
                if timestamp != 'Unknown':
                    try:
                        dt = datetime.fromisoformat(timestamp)
                        timestamp = dt.strftime("%Y-%m-%d %H:%M:%S")
                    except:
                        pass
                print(f"    📄 {filename} - {timestamp}")
    else:
        print("  ⚠️ No state file found")
    
    # Check embeddings
    print("\n🧠 Embeddings Status:")
    embeddings_info = check_embeddings()
    
    if "faiss_vectors" in embeddings_info:
        print(f"  ✅ FAISS Index: {embeddings_info['faiss_vectors']} vectors")
        print(f"     Dimension: {embeddings_info['faiss_dimension']}")
        print(f"     Size: {embeddings_info['faiss_size_mb']:.1f} MB")
    elif "faiss_error" in embeddings_info:
        print(f"  ❌ FAISS Error: {embeddings_info['faiss_error']}")
    else:
        print(f"  ❌ FAISS Index: Not found")
    
    if "chunks_count" in embeddings_info:
        print(f"  ✅ Document Chunks: {embeddings_info['chunks_count']} chunks")
        print(f"     Size: {embeddings_info['chunks_size_mb']:.1f} MB")
        if embeddings_info.get("sample_chunk"):
            print(f"     Sample: {embeddings_info['sample_chunk']}")
    elif "chunks_error" in embeddings_info:
        print(f"  ❌ Chunks Error: {embeddings_info['chunks_error']}")
    else:
        print(f"  ❌ Document Chunks: Not found")
    
    # Check input files
    input_dir = Path("data/input")
    if input_dir.exists():
        input_files = list(input_dir.glob("*"))
        supported_files = [f for f in input_files if f.suffix.lower() in {'.pdf', '.docx', '.txt', '.md'}]
        
        print(f"\n📥 Input Directory:")
        print(f"  📄 Total files: {len(input_files)}")
        print(f"  ✅ Supported files: {len(supported_files)}")
        
        if supported_files:
            print(f"  📋 Files:")
            for file_path in supported_files:
                size_mb = file_path.stat().st_size / (1024 * 1024)
                print(f"    📄 {file_path.name} ({size_mb:.1f} MB)")

if __name__ == "__main__":
    show_status()

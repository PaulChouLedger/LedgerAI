#!/usr/bin/env python3
"""
Sync common code from llm-container to llm-medical-container
Keeps RAG files in sync between generic and medical containers
Note: container_rest.py is no longer synced - medical container has its own minimal version
"""

import os
import shutil
import sys
from pathlib import Path

def get_workspace_root():
    """Get workspace root directory"""
    script_dir = Path(__file__).parent
    return script_dir.parent.parent

def sync_file(source, dest, description):
    """Sync a file from source to destination"""
    source_path = Path(source)
    dest_path = Path(dest)
    
    if not source_path.exists():
        print(f"⚠️  Source file not found: {source_path}")
        return False
    
    # Create destination directory if needed
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Check if files are different
    if dest_path.exists():
        if source_path.read_bytes() == dest_path.read_bytes():
            print(f"✓ {description} - already in sync")
            return True
    
    # Copy file
    try:
        shutil.copy2(source_path, dest_path)
        print(f"✅ Synced {description}: {source_path.name}")
        return True
    except Exception as e:
        print(f"❌ Failed to sync {description}: {e}")
        return False

def sync_rag_directory(source_dir, dest_dir):
    """Sync RAG directory files"""
    source_path = Path(source_dir)
    dest_path = Path(dest_dir)
    
    if not source_path.exists():
        print(f"⚠️  Source RAG directory not found: {source_path}")
        return False
    
    # Files to sync from RAG directory
    rag_files = [
        'rag_client.py',
        'cpu_faiss_auto_ingest.py',
        'optimize_rag_architecture.py',
        '__init__.py',
    ]
    
    synced_count = 0
    for filename in rag_files:
        source_file = source_path / filename
        dest_file = dest_path / filename
        
        if source_file.exists():
            if sync_file(source_file, dest_file, f"RAG/{filename}"):
                synced_count += 1
        else:
            print(f"⚠️  RAG file not found: {source_file}")
    
    return synced_count > 0


def main():
    """Main sync function"""
    workspace_root = get_workspace_root()
    
    generic_dir = workspace_root / "llm-container"
    medical_dir = workspace_root / "llm-medical-container"
    
    if not generic_dir.exists():
        print(f"❌ Generic container directory not found: {generic_dir}")
        sys.exit(1)
    
    if not medical_dir.exists():
        print(f"❌ Medical container directory not found: {medical_dir}")
        sys.exit(1)
    
    print("🔄 Syncing common code from llm-container to llm-medical-container...")
    print()
    
    # Sync RAG directory
    print("📦 Syncing RAG files...")
    generic_rag = generic_dir / "rag"
    medical_rag = medical_dir / "rag"
    sync_rag_directory(generic_rag, medical_rag)
    print()
    
    # Sync other common files if they exist
    common_files = [
        ("requirements.txt", "requirements.txt"),
    ]
    
    print("📋 Syncing other common files...")
    for source_name, dest_name in common_files:
        source_file = generic_dir / source_name
        dest_file = medical_dir / dest_name
        if source_file.exists():
            sync_file(source_file, dest_file, source_name)
    print()
    
    print("✅ Sync complete!")
    print()
    print("💡 Note: container_rest.py is no longer synced")
    print("   Medical container has its own minimal container_rest.py that delegates to AdvancedMedicalNavigator")

if __name__ == "__main__":
    main()


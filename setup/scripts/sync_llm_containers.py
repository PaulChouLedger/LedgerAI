#!/usr/bin/env python3
"""
Sync common code from llm-container to llm-medical-container
Keeps container_rest.py and RAG files in sync between generic and medical containers
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

def sync_container_rest_sections(source_file, dest_file):
    """
    Sync container_rest.py from generic to medical, preserving medical-specific imports and code
    """
    source_path = Path(source_file)
    dest_path = Path(dest_file)
    
    if not source_path.exists():
        print(f"⚠️  Source container_rest.py not found: {source_path}")
        return False
    
    if not dest_path.exists():
        print(f"⚠️  Destination container_rest.py not found: {dest_path}")
        return False
    
    # Read both files
    source_content = source_path.read_text()
    dest_content = dest_path.read_text()
    
    # Check if files are already identical (excluding medical-specific parts)
    # For now, we'll do a direct sync since the user wants changes applied in parallel
    # Medical-specific code should be in separate files (advanced_medical_navigator.py)
    
    # Simple approach: sync the file directly
    # The medical container should import medical-specific functionality from other files
    try:
        # Create backup
        backup_path = dest_path.with_suffix('.py.backup')
        if dest_path.exists():
            shutil.copy2(dest_path, backup_path)
            print(f"   Created backup: {backup_path.name}")
        
        # Copy source to destination
        shutil.copy2(source_path, dest_path)
        
        # Check if there are medical-specific imports that need to be preserved
        # Look for medical-specific imports in the backup
        if backup_path.exists():
            backup_content = backup_path.read_text()
            medical_imports = []
            for line in backup_content.split('\n'):
                if 'advanced_medical_navigator' in line or 'fuzzy_medical_matcher' in line:
                    medical_imports.append(line)
            
            # If medical imports exist, add them back
            if medical_imports:
                dest_content = dest_path.read_text()
                # Find where to insert medical imports (after standard imports)
                lines = dest_content.split('\n')
                insert_idx = 0
                for i, line in enumerate(lines):
                    if line.startswith('from rag import') or line.startswith('import rag'):
                        insert_idx = i + 1
                        break
                
                # Insert medical imports
                for import_line in medical_imports:
                    if import_line not in dest_content:
                        lines.insert(insert_idx, import_line)
                        insert_idx += 1
                
                dest_path.write_text('\n'.join(lines))
                print(f"   Preserved medical-specific imports")
        
        print(f"✅ Synced container_rest.py")
        return True
    except Exception as e:
        print(f"❌ Failed to sync container_rest.py: {e}")
        # Restore backup if sync failed
        if backup_path.exists():
            shutil.copy2(backup_path, dest_path)
            print(f"   Restored from backup")
        return False

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
    
    # Sync container_rest.py
    print("📄 Syncing container_rest.py...")
    generic_container_rest = generic_dir / "container_rest.py"
    medical_container_rest = medical_dir / "container_rest.py"
    sync_container_rest_sections(generic_container_rest, medical_container_rest)
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
    print("💡 Tip: Medical-specific code should be in advanced_medical_navigator.py")
    print("   container_rest.py should contain common functionality only.")

if __name__ == "__main__":
    main()


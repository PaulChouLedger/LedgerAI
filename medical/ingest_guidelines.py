#!/usr/bin/env python3
"""
Medical Guideline Ingestion Pipeline

Converts scraped guidelines to RAG-ready format and triggers ingestion.
Works with the existing RAG auto-ingest system.
"""

import os
import json
import shutil
import requests
from pathlib import Path
from typing import List, Dict

class GuidelineIngestionPipeline:
    """
    Manages ingestion of medical guidelines into RAG system
    """
    
    def __init__(self, 
                 guidelines_dir: str = None,
                 rag_input_dir: str = None,
                 rag_service_url: str = "http://localhost:11435"):
        # Default to repo root paths (using absolute paths)
        if guidelines_dir is None:
            script_dir = Path(__file__).resolve().parent  # /path/to/LedgerAI/medical
            repo_root = script_dir.parent  # /path/to/LedgerAI
            guidelines_dir = repo_root / "data" / "input" / "medical_guidelines"
        
        if rag_input_dir is None:
            script_dir = Path(__file__).resolve().parent
            repo_root = script_dir.parent
            rag_input_dir = repo_root / "data" / "input"
        
        self.guidelines_dir = Path(guidelines_dir).resolve()
        self.rag_input_dir = Path(rag_input_dir).resolve()
        self.rag_service_url = rag_service_url
        
        print(f"[Ingest] ✅ Pipeline initialized")
        print(f"[Ingest]    Guidelines: {self.guidelines_dir}")
        print(f"[Ingest]    RAG input:  {self.rag_input_dir}")
    
    def check_guidelines_ready(self) -> int:
        """
        Check if guideline files exist in data/input/
        
        Returns:
            Number of guideline .txt files found
        """
        txt_files = list(self.rag_input_dir.glob("NIH_MedlinePlus_*.txt"))
        
        if not txt_files:
            print(f"[Ingest] ⚠️ No guideline .txt files found in {self.rag_input_dir}")
            print(f"[Ingest] 💡 Run guideline_scraper.py first to generate guidelines")
            return 0
        
        print(f"\n[Ingest] 📂 Found {len(txt_files)} guideline files in {self.rag_input_dir}")
        for txt_file in txt_files[:5]:  # Show first 5
            print(f"[Ingest]    ✅ {txt_file.name}")
        if len(txt_files) > 5:
            print(f"[Ingest]    ... and {len(txt_files) - 5} more")
        
        return len(txt_files)
    
    def trigger_rag_ingest(self) -> bool:
        """
        Trigger RAG container to ingest new files
        
        Returns:
            True if successful
        """
        try:
            print(f"\n[Ingest] 🔄 Step 1: Extracting text from files...")
            
            response = requests.post(
                f"{self.rag_service_url}/rag/ingest",
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"[Ingest] ✅ Text extraction complete:")
                print(f"[Ingest]    Processed: {result.get('processed', 0)} files")
                print(f"[Ingest]    Skipped:   {result.get('skipped', 0)} files")
            else:
                print(f"[Ingest] ❌ RAG ingest failed: HTTP {response.status_code}")
                return False
            
            # Step 2: Rebuild embeddings
            print(f"\n[Ingest] 🔄 Step 2: Building embeddings and FAISS index...")
            print(f"[Ingest] 💡 Running: docker exec rag-container python3 /app/rebuild_embeddings.py")
            
            import subprocess
            rebuild_result = subprocess.run(
                ["docker", "exec", "rag-container", "python3", "/app/rebuild_embeddings.py"],
                capture_output=True,
                text=True,
                timeout=120
            )
            
            if rebuild_result.returncode == 0:
                print(f"[Ingest] ✅ Embeddings rebuilt successfully!")
                # Parse output to show chunk count
                output_lines = rebuild_result.stdout.split('\n')
                for line in output_lines:
                    if 'Created' in line and 'chunks' in line:
                        print(f"[Ingest]    {line.strip()}")
                    elif 'vectors' in line or 'Chunks:' in line:
                        print(f"[Ingest]    {line.strip()}")
                return True
            else:
                print(f"[Ingest] ❌ Rebuild failed:")
                print(rebuild_result.stderr)
                return False
                
        except Exception as e:
            print(f"[Ingest] ❌ Error during ingestion: {e}")
            print(f"[Ingest] 💡 Make sure RAG container is running: docker-compose ps")
            return False
    
    def run_full_pipeline(self):
        """Execute complete ingestion pipeline"""
        print("\n" + "="*80)
        print("  📚 MEDICAL GUIDELINE INGESTION PIPELINE")
        print("="*80 + "\n")
        
        # Step 1: Check if guidelines exist
        file_count = self.check_guidelines_ready()
        
        if file_count == 0:
            print("\n❌ No guideline files found\n")
            print("💡 Run: python3 medical/guideline_scraper.py\n")
            return False
        
        print(f"\n[Pipeline] ✅ Found {file_count} guideline files ready for RAG")
        
        # Step 2: Trigger RAG ingestion
        success = self.trigger_rag_ingest()
        
        if success:
            print("\n" + "="*80)
            print("  ✅ PIPELINE COMPLETE!")
            print("="*80)
            print("\n  Medical guidelines are now available in RAG system!")
            print("  The unified medical mode can now use them for dynamic questioning.\n")
            print("="*80 + "\n")
        else:
            print("\n⚠️ RAG ingestion failed - you may need to restart RAG container\n")
        
        return success


def main():
    """Main execution"""
    pipeline = GuidelineIngestionPipeline()
    pipeline.run_full_pipeline()


if __name__ == "__main__":
    main()


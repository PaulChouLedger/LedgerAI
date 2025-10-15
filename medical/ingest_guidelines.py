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
        self.rag_ready_dir = self.guidelines_dir / "rag_ready"
        self.rag_input_dir = Path(rag_input_dir).resolve()
        self.rag_service_url = rag_service_url
        
        print(f"[Ingest] ✅ Pipeline initialized")
        print(f"[Ingest]    Guidelines: {self.guidelines_dir}")
        print(f"[Ingest]    RAG input:  {self.rag_input_dir}")
    
    def copy_to_rag_input(self) -> int:
        """
        Copy RAG-ready guideline files to RAG input directory
        
        Returns:
            Number of files copied
        """
        if not self.rag_ready_dir.exists():
            print(f"[Ingest] ⚠️ No RAG-ready directory found: {self.rag_ready_dir}")
            print(f"[Ingest] 💡 Run guideline_scraper.py first to generate guidelines")
            return 0
        
        txt_files = list(self.rag_ready_dir.glob("*.txt"))
        
        if not txt_files:
            print(f"[Ingest] ⚠️ No .txt files found in {self.rag_ready_dir}")
            return 0
        
        print(f"\n[Ingest] 📥 Copying {len(txt_files)} guidelines to RAG input...")
        
        copied = 0
        for txt_file in txt_files:
            dest = self.rag_input_dir / txt_file.name
            try:
                shutil.copy2(txt_file, dest)
                print(f"[Ingest] ✅ Copied: {txt_file.name}")
                copied += 1
            except Exception as e:
                print(f"[Ingest] ❌ Error copying {txt_file.name}: {e}")
        
        return copied
    
    def trigger_rag_ingest(self) -> bool:
        """
        Trigger RAG container to ingest new files
        
        Returns:
            True if successful
        """
        try:
            print(f"\n[Ingest] 🔄 Triggering RAG auto-ingest...")
            
            response = requests.post(
                f"{self.rag_service_url}/rag/ingest",
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"[Ingest] ✅ RAG ingest complete:")
                print(f"[Ingest]    Processed: {result.get('processed', 0)} files")
                print(f"[Ingest]    Skipped:   {result.get('skipped', 0)} files")
                print(f"[Ingest]    Total chunks: {result.get('total_chunks', 0)}")
                return True
            else:
                print(f"[Ingest] ❌ RAG ingest failed: HTTP {response.status_code}")
                return False
                
        except Exception as e:
            print(f"[Ingest] ❌ Error triggering RAG ingest: {e}")
            print(f"[Ingest] 💡 Make sure RAG container is running: docker-compose ps")
            return False
    
    def run_full_pipeline(self):
        """Execute complete ingestion pipeline"""
        print("\n" + "="*80)
        print("  📚 MEDICAL GUIDELINE INGESTION PIPELINE")
        print("="*80 + "\n")
        
        # Step 1: Copy guidelines to RAG input
        copied = self.copy_to_rag_input()
        
        if copied == 0:
            print("\n❌ No files to ingest\n")
            return False
        
        print(f"\n[Pipeline] ✅ Copied {copied} guideline files")
        
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


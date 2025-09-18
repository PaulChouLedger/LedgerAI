# core/auto_ingest.py — Optional auto ingestion on startup

import os
from core.data_ingestion import ingest_all_supported_files  # ✅ Updated import
from core.context import build_faiss_index

def setup_context(auto_ingest=True):
    """
    Auto-ingests input files (TXT, PDF, HTML) and rebuilds context index.
    """
    if auto_ingest:
        print("[Aura/setup] 🧠 Auto-ingesting new files (TXT, PDF, HTML)...")
        ingest_all_supported_files()  # ✅ Updated function name

    print("[Aura/setup] 🧠 Building FAISS context index...")
    build_faiss_index()

#!/usr/bin/env python3
"""
ingest_watcher.py — Standalone RAG auto-ingest daemon.

Decouples FAISS ingest from the LLM service. Watches data/input/ for new
files (dropped by AuraConnect BLE transfers, manual scp, etc.) and updates
the embeddings index in place. Survives LLM restarts, GUI crashes, BLE
session churn.

Run via systemd as aura-ingest.service. The LLM service reads the same
on-disk index at startup, so ingest is always live regardless of LLM state.
"""

from __future__ import annotations

import os
import signal
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"

sys.path.insert(0, str(REPO_ROOT / "containers" / "llm"))

from rag.cpu_faiss_auto_ingest import CPUFAISSAutoIngest


def main() -> int:
    print(f"[ingest-watcher] base_dir={DATA_DIR}", flush=True)
    ai = CPUFAISSAutoIngest(base_dir=str(DATA_DIR))
    ai.load_existing_embeddings()

    print("[ingest-watcher] initial scan", flush=True)
    ai.scan_and_process()

    ai.start_watching()

    stop = False
    def _term(_sig, _frm):
        nonlocal stop
        stop = True
    signal.signal(signal.SIGTERM, _term)
    signal.signal(signal.SIGINT, _term)

    while not stop:
        time.sleep(1)

    ai.stop_watching()
    print("[ingest-watcher] stopped", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

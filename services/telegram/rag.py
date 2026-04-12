"""
rag -- RAG (Retrieval-Augmented Generation) for TG bot.

Thin wrapper around the shared RAG client, configured for RTX paths.
Searches local FAISS index and returns context for LLM prompt injection.
"""

from __future__ import annotations

import logging
import sys
import os
from pathlib import Path

log = logging.getLogger(__name__)

# Add the LLM container path so we can import the RAG module
_rag_path = str(Path(__file__).resolve().parent.parent.parent / "containers" / "llm")
if _rag_path not in sys.path:
    sys.path.insert(0, _rag_path)

# Base data dir — on RTX this is the repo's data/ directory
_BASE_DIR = str(Path(__file__).resolve().parent.parent.parent / "data")

_client = None
_init_failed = False


def _get_client():
    """Lazy-init RAG client singleton."""
    global _client, _init_failed
    if _client is not None:
        return _client
    if _init_failed:
        return None
    try:
        os.environ.setdefault("RAG_MODE", "CPU")
        from rag import get_rag_client
        _client = get_rag_client(use_gpu=False, base_dir=_BASE_DIR)
        log.info("RAG client initialized (base_dir=%s, chunks=%d)",
                 _BASE_DIR, len(getattr(_client, '_cpu_chunks', [])))
        return _client
    except Exception as e:
        log.error("RAG init failed: %s", e)
        _init_failed = True
        return None


def search(query: str, k: int = 3, threshold: float = 0.15) -> list[dict]:
    """Search RAG index. Returns list of {text, score, metadata} dicts."""
    client = _get_client()
    if not client:
        return []
    try:
        return client.search(query=query, k=k, threshold=threshold)
    except Exception as e:
        log.warning("RAG search failed: %s", e)
        return []


def format_context(results: list[dict], max_chars: int = 2000) -> str:
    """Format RAG results into a context block for system prompt injection."""
    if not results:
        return ""
    lines = []
    total = 0
    for r in results:
        text = r.get("text", "").strip()
        source = r.get("metadata", {}).get("document_name", "")
        if not text:
            continue
        entry = f"[{source}] {text}" if source else text
        if total + len(entry) > max_chars:
            break
        lines.append(entry)
        total += len(entry)
    if not lines:
        return ""
    return "RELEVANT KNOWLEDGE:\n" + "\n---\n".join(lines)


def rag_context_for(query: str, k: int = 3, max_chars: int = 2000) -> str:
    """One-call convenience: search + format. Returns empty string if nothing found."""
    results = search(query, k=k)
    return format_context(results, max_chars=max_chars)

"""
RAG Module for LLM Medical Container

This module contains all RAG-related components:
- RAG Client for CPU/GPU FAISS operations
- Auto-ingestion system for dynamic updates
- Conversion scripts for different FAISS formats
- Architecture optimization tools
"""

from .rag_client import RAGClient, get_rag_client

__all__ = [
    'RAGClient',
    'get_rag_client'
]

__version__ = "1.0.0"

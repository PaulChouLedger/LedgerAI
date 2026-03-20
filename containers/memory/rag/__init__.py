"""
RAG Module for Memory Container

This module provides RAG functionality for searching stored conversations
and injecting relevant past conversations into LLM responses.
"""

from .memory_rag_client import MemoryRAGClient, get_memory_rag_client

__all__ = [
    'MemoryRAGClient',
    'get_memory_rag_client'
]

__version__ = "1.0.0"


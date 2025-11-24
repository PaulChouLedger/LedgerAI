"""
Simple LlamaIndex VectorStore adapter for cuVS integration.
"""
from typing import List, Dict, Any, Optional
from llama_index.core.vector_stores.types import (
    VectorStore,
    VectorStoreQuery,
    VectorStoreQueryResult,
    MetadataFilters,
    MetadataFilter,
    FilterOperator,
)
from llama_index.core.schema import TextNode, BaseNode
import numpy as np

from aura_rag.vector_store import VectorStore as CuVSVectorStore


class CuVSVectorStoreAdapter(VectorStore):
    """LlamaIndex adapter for cuVS vector store."""
    
    def __init__(self, vector_store: CuVSVectorStore):
        self.vector_store = vector_store
    
    def add(self, nodes: List[BaseNode]) -> List[str]:
        """Add nodes to vector store (not implemented for read-only)."""
        raise NotImplementedError("This is a read-only vector store")
    
    def delete(self, ref_doc_id: str, **kwargs) -> None:
        """Delete nodes by ref_doc_id (not implemented for read-only)."""
        raise NotImplementedError("This is a read-only vector store")
    
    def query(self, query: VectorStoreQuery, **kwargs) -> VectorStoreQueryResult:
        """Query the vector store."""
        if not query.query_embedding:
            return VectorStoreQueryResult(nodes=[], similarities=[], ids=[])
        
        # Convert query embedding to numpy array
        query_embedding = np.array(query.query_embedding)
        
        # Search using cuVS
        results = self.vector_store.search(
            query_embedding=query_embedding,
            k=query.similarity_top_k or 3,
            score_threshold=self.vector_store.score_threshold
        )
        
        # Convert results to LlamaIndex format
        nodes = []
        similarities = []
        ids = []
        
        for result in results:
            # Create TextNode
            node = TextNode(
                text=result["text"],
                metadata={
                    "doc_id": result["doc_id"],
                    "chunk_id": result["chunk_id"],
                    "similarity": result["similarity"],
                    "rank": result["rank"]
                }
            )
            
            nodes.append(node)
            similarities.append(result["similarity"])
            ids.append(result["chunk_id"])
        
        return VectorStoreQueryResult(
            nodes=nodes,
            similarities=similarities,
            ids=ids
        )
    
    def persist(self, persist_path: str, fs=None) -> None:
        """Persist vector store (already handled by cuVS)."""
        pass

"""
Simple cuVS-based vector store with metadata support.
"""
import os
import pickle
import numpy as np
import cudf
from typing import List, Dict, Any, Tuple
from cuvs.neighbors import NearestNeighbors

from .config import AuraRAGConfig


class VectorStore:
    """Simple vector store using cuVS for GPU-accelerated similarity search."""
    
    def __init__(self, config: AuraRAGConfig):
        self.config = config
        self.index_path = config.index_path
        self.n_neighbors = 10  # Fixed reasonable value
        self.metric = "cosine"  # Fixed reasonable value
        self.score_threshold = config.score_threshold
        
        self.index_file = os.path.join(self.index_path, "cuvs_index.pkl")
        self.metadata_file = os.path.join(self.index_path, "metadata.parquet")
        
        os.makedirs(self.index_path, exist_ok=True)
        
        self.index = None
        self.metadata = None
        self._load_index()
    
    def _load_index(self):
        """Load existing index and metadata if available."""
        if os.path.exists(self.index_file) and os.path.exists(self.metadata_file):
            print("Loading existing cuVS index...")
            with open(self.index_file, "rb") as f:
                self.index = pickle.load(f)
            self.metadata = cudf.read_parquet(self.metadata_file)
            print(f"Loaded index with {len(self.metadata)} vectors")
        else:
            print("No existing index found, will create new one")
    
    def build_index(self, embeddings: np.ndarray, metadata_df: cudf.DataFrame):
        """Build cuVS index from embeddings and metadata."""
        print(f"Building cuVS index with {len(embeddings)} vectors...")
        
        # Create cuVS NearestNeighbors index
        self.index = NearestNeighbors(
            n_neighbors=self.n_neighbors,
            metric=self.metric
        )
        
        # Fit the index
        self.index.fit(embeddings.astype(np.float32))
        
        # Store metadata
        self.metadata = metadata_df.copy()
        
        # Save to disk
        self._save_index()
        print("cuVS index built and saved successfully")
    
    def _save_index(self):
        """Save index and metadata to disk."""
        with open(self.index_file, "wb") as f:
            pickle.dump(self.index, f)
        
        self.metadata.to_parquet(self.metadata_file)
        print(f"Index saved to {self.index_path}")
    
    def search(self, query_embedding: np.ndarray, k: int = None, score_threshold: float = None) -> List[Dict[str, Any]]:
        """Search for similar vectors."""
        if self.index is None:
            return []
        
        # Use config defaults if not provided
        if k is None:
            k = self.n_neighbors
        if score_threshold is None:
            score_threshold = self.score_threshold
        
        # Search for nearest neighbors
        distances, indices = self.index.kneighbors(
            query_embedding.reshape(1, -1).astype(np.float32), 
            n_neighbors=k
        )
        
        results = []
        for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
            # Convert distance to similarity score (cosine similarity)
            similarity = 1 - distance
            
            if similarity >= score_threshold and idx < len(self.metadata):
                metadata_row = self.metadata.iloc[idx]
                results.append({
                    "text": metadata_row["text"],
                    "doc_id": metadata_row["doc_id"],
                    "chunk_id": metadata_row["chunk_id"],
                    "similarity": float(similarity),
                    "rank": i + 1
                })
        
        return results
    
    def get_stats(self) -> Dict[str, Any]:
        """Get vector store statistics."""
        if self.metadata is None:
            return {"total_vectors": 0, "total_docs": 0}
        
        return {
            "total_vectors": len(self.metadata),
            "total_docs": self.metadata["doc_id"].nunique(),
            "index_path": self.index_path
        }

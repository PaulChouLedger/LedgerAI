"""
Simple RAG engine that orchestrates the entire pipeline.
"""
import os
import requests
from typing import List, Dict, Any, Optional
from llama_index.core import VectorStoreIndex, ServiceContext
from llama_index.core.llms import LLM
from llama_index.core.embeddings import BaseEmbedding

from .config import AuraRAGConfig
from .document_processor import DocumentProcessor
from .embedding_engine import EmbeddingEngine
from .vector_store import VectorStore
from .llamaindex_adapter import CuVSVectorStoreAdapter


class SimpleLLM(LLM):
    """Simple LLM wrapper for the existing LLM container."""
    
    def __init__(self, config: AuraRAGConfig):
        super().__init__()
        self.config = config
        self.llm_url = config.llm_url
        self.timeout = config.llm_timeout
        self.max_retries = 3  # Fixed reasonable value
        self.retry_delay = 1.0  # Fixed reasonable value
    
    def complete(self, prompt: str, **kwargs) -> str:
        """Complete text using the LLM container."""
        for attempt in range(self.max_retries + 1):
            try:
                response = requests.post(
                    f"{self.llm_url}/chat",
                    json={"prompt": prompt},
                    timeout=self.timeout
                )
                response.raise_for_status()
                return response.json().get("response", "")
            except Exception as e:
                if attempt < self.max_retries:
                    import time
                    time.sleep(self.retry_delay)
                    continue
                return f"Error calling LLM after {self.max_retries + 1} attempts: {e}"
    
    def stream_complete(self, prompt: str, **kwargs):
        """Stream completion (not implemented)."""
        raise NotImplementedError("Streaming not implemented")


class SimpleEmbedding(BaseEmbedding):
    """Simple embedding wrapper for the embedding engine."""
    
    def __init__(self, embedding_engine: EmbeddingEngine):
        super().__init__()
        self.embedding_engine = embedding_engine
    
    def _get_query_embedding(self, query: str) -> List[float]:
        """Get query embedding."""
        embedding = self.embedding_engine.encode(query)
        return embedding[0].tolist()
    
    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get text embeddings."""
        embeddings = self.embedding_engine.encode_batch(texts)
        return embeddings.tolist()


class RAGEngine:
    """Simple RAG engine that orchestrates the entire pipeline."""
    
    def __init__(self, config: AuraRAGConfig):
        self.config = config
        
        # Initialize components with config
        self.doc_processor = DocumentProcessor(config)
        self.embedding_engine = EmbeddingEngine(config)
        self.vector_store = VectorStore(config)
        
        # LlamaIndex components
        self.llm = SimpleLLM(config)
        self.embedding = SimpleEmbedding(self.embedding_engine)
        self.service_context = ServiceContext.from_defaults(
            llm=self.llm,
            embed_model=self.embedding
        )
        
        # Initialize index
        self._setup_index()
    
    def _setup_index(self):
        """Setup the vector index."""
        if self.vector_store.index is None:
            print("Building vector index from documents...")
            self._build_index()
        else:
            print("Using existing vector index")
        
        # Create LlamaIndex adapter
        vector_store_adapter = CuVSVectorStoreAdapter(self.vector_store)
        self.index = VectorStoreIndex.from_vector_store(
            vector_store_adapter,
            service_context=self.service_context
        )
    
    def _build_index(self):
        """Build vector index from documents."""
        # Process documents
        print(f"Processing documents from {self.config.input_dir}")
        chunks_df = self.doc_processor.process_documents(self.config.input_dir)
        
        if chunks_df.empty:
            print("No documents found to process")
            return
        
        # Generate embeddings
        print(f"Generating embeddings for {len(chunks_df)} chunks")
        texts = chunks_df["text"].to_pandas().tolist()
        embeddings = self.embedding_engine.encode_batch(texts)
        
        # Build vector store
        self.vector_store.build_index(embeddings, chunks_df)
    
    def query(self, question: str, top_k: int = None) -> str:
        """Query the RAG system."""
        try:
            # Use config default if not provided
            if top_k is None:
                top_k = self.config.top_k
                
            # Create query engine
            query_engine = self.index.as_query_engine(
                similarity_top_k=top_k,
                response_mode="compact"  # Fixed reasonable value
            )
            
            # Get response
            response = query_engine.query(question)
            return str(response)
            
        except Exception as e:
            return f"Error processing query: {e}"
    
    def get_stats(self) -> Dict[str, Any]:
        """Get RAG system statistics."""
        return {
            "vector_store": self.vector_store.get_stats(),
            "input_dir": self.config.input_dir,
            "index_path": self.config.index_path,
            "llm_url": self.config.llm_url,
            "config": self.config.to_dict()
        }

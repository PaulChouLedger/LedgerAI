"""
Configuration module for Aura RAG system.
Uses Pydantic for validation and type safety.
Only the most critical performance knobs are exposed.
"""
from typing import Literal
from pydantic import BaseModel, Field, validator
import os


class AuraRAGConfig(BaseModel):
    """Main configuration class - only critical performance parameters."""
    
    # === CRITICAL PERFORMANCE KNOBS ===
    
    # Document Processing
    chunk_size: int = Field(
        default=512,
        ge=64,
        le=2048,
        description="Size of text chunks - affects retrieval quality vs speed"
    )
    
    # Embeddings
    embedding_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        description="Embedding model - affects quality vs speed"
    )
    
    embedding_batch_size: int = Field(
        default=32,
        ge=1,
        le=128,
        description="Batch size for embeddings - affects GPU memory vs speed"
    )
    
    # Vector Search
    top_k: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Number of chunks to retrieve - affects context richness"
    )
    
    score_threshold: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Similarity threshold - affects result quality vs quantity"
    )
    
    # LLM Integration
    llm_url: str = Field(
        default="http://localhost:11434",
        description="LLM container URL"
    )
    
    llm_timeout: int = Field(
        default=30,
        ge=5,
        le=120,
        description="LLM request timeout in seconds"
    )
    
    # Storage
    input_dir: str = Field(
        default="/shared/input",
        description="Directory containing input documents"
    )
    
    index_path: str = Field(
        default="/shared/vector_index",
        description="Path to store vector index"
    )
    
    # Global
    debug: bool = Field(
        default=False,
        description="Enable debug logging"
    )
    
    @classmethod
    def from_env(cls) -> "AuraRAGConfig":
        """Create configuration from environment variables."""
        return cls(
            chunk_size=int(os.getenv("CHUNK_SIZE", "512")),
            embedding_model=os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
            embedding_batch_size=int(os.getenv("EMBEDDING_BATCH_SIZE", "32")),
            top_k=int(os.getenv("TOP_K", "3")),
            score_threshold=float(os.getenv("SCORE_THRESHOLD", "0.7")),
            llm_url=os.getenv("LLM_URL", "http://localhost:11434"),
            llm_timeout=int(os.getenv("LLM_TIMEOUT", "30")),
            input_dir=os.getenv("INPUT_DIR", "/shared/input"),
            index_path=os.getenv("INDEX_PATH", "/shared/vector_index"),
            debug=os.getenv("DEBUG", "false").lower() == "true",
        )
    
    def to_dict(self) -> dict:
        """Convert configuration to dictionary."""
        return self.dict()
    
    def save_to_file(self, filepath: str) -> None:
        """Save configuration to JSON file."""
        import json
        with open(filepath, 'w') as f:
            json.dump(self.dict(), f, indent=2)
    
    @classmethod
    def load_from_file(cls, filepath: str) -> "AuraRAGConfig":
        """Load configuration from JSON file."""
        import json
        with open(filepath, 'r') as f:
            data = json.load(f)
        return cls(**data)


# Global configuration instance (optional - can be used for default config)
# config = AuraRAGConfig.from_env()

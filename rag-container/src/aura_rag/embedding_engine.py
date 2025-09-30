"""
Simple GPU-accelerated embedding engine using HuggingFace transformers.
"""
import torch
import numpy as np
from typing import List, Union
from transformers import AutoTokenizer, AutoModel

from .config import AuraRAGConfig


class EmbeddingEngine:
    """Simple GPU embedding engine using HuggingFace transformers."""
    
    def __init__(self, config: AuraRAGConfig):
        self.config = config
        self.model_name = config.embedding_model
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.max_length = 512  # Fixed reasonable value
        self.normalize_embeddings = True  # Fixed reasonable value
        
        print(f"Loading embedding model: {self.model_name} on {self.device}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name).to(self.device)
        self.model.eval()
        
        print(f"Embedding model loaded successfully")
    
    def encode(self, texts: Union[str, List[str]]) -> np.ndarray:
        """Encode text(s) to embeddings."""
        if isinstance(texts, str):
            texts = [texts]
        
        # Tokenize
        inputs = self.tokenizer(
            texts, 
            padding=True, 
            truncation=True, 
            max_length=self.max_length,
            return_tensors="pt"
        ).to(self.device)
        
        # Get embeddings
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Use mean pooling of last hidden states
            embeddings = outputs.last_hidden_state.mean(dim=1)
            # Normalize for cosine similarity if configured
            if self.normalize_embeddings:
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        
        return embeddings.cpu().numpy()
    
    def encode_batch(self, texts: List[str], batch_size: int = None) -> np.ndarray:
        """Encode texts in batches for memory efficiency."""
        if batch_size is None:
            batch_size = self.config.embedding_batch_size
            
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            embeddings = self.encode(batch)
            all_embeddings.append(embeddings)
        
        return np.vstack(all_embeddings)

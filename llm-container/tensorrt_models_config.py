"""
TensorRT-LLM Models Configuration
Maps model names to engine directories and tokenizer configs
"""

import os
from typing import Dict, Optional

# Base path for TensorRT-LLM engines
TENSORRT_ENGINES_BASE = os.getenv("TENSORRT_ENGINES_BASE", "/models/tensorrt-llm")

# Model configurations
MODEL_CONFIGS: Dict[str, Dict] = {
    "qwen3-4b": {
        "engine_dir": f"{TENSORRT_ENGINES_BASE}/qwen3-4b-instruct",
        "tokenizer_dir": f"{TENSORRT_ENGINES_BASE}/qwen3-4b-instruct",
        "chat_format": "qwen",
        "context_window": 2048,
    },
    "qwen3-4b-2507": {
        "engine_dir": f"{TENSORRT_ENGINES_BASE}/qwen3-4b-instruct-2507",
        "tokenizer_dir": f"{TENSORRT_ENGINES_BASE}/qwen3-4b-instruct-2507",
        "chat_format": "qwen",
        "context_window": 2048,
    },
    "llama-3.2-1b": {
        "engine_dir": f"{TENSORRT_ENGINES_BASE}/llama-3.2-1b-instruct",
        "tokenizer_dir": f"{TENSORRT_ENGINES_BASE}/llama-3.2-1b-instruct",
        "chat_format": "llama-3",
        "context_window": 2048,
    },
    "llama-3.1-8b": {
        "engine_dir": f"{TENSORRT_ENGINES_BASE}/llama-3.1-8b-instruct",
        "tokenizer_dir": f"{TENSORRT_ENGINES_BASE}/llama-3.1-8b-instruct",
        "chat_format": "llama-3.1",
        "context_window": 8192,
    },
    "llama-3.1-8b-instruct": {
        "engine_dir": f"{TENSORRT_ENGINES_BASE}/llama-3.1-8b-instruct",
        "tokenizer_dir": f"{TENSORRT_ENGINES_BASE}/llama-3.1-8b-instruct",
        "chat_format": "llama-3.1",
        "context_window": 8192,
    },
    "qwen2.5-coder-7b": {
        "engine_dir": f"{TENSORRT_ENGINES_BASE}/qwen2.5-coder-7b-instruct",
        "tokenizer_dir": f"{TENSORRT_ENGINES_BASE}/qwen2.5-coder-7b-instruct",
        "chat_format": "qwen2",
        "context_window": 32768,  # Qwen2.5 supports longer context
    },
    "qwen2.5-coder-7b-instruct": {
        "engine_dir": f"{TENSORRT_ENGINES_BASE}/qwen2.5-coder-7b-instruct",
        "tokenizer_dir": f"{TENSORRT_ENGINES_BASE}/qwen2.5-coder-7b-instruct",
        "chat_format": "qwen2",
        "context_window": 32768,
    },
}


def get_model_config(model_name: Optional[str] = None) -> Dict:
    """
    Get configuration for a model
    
    Args:
        model_name: Model name (e.g., "qwen3-4b"). If None, uses SIMPLE_MODEL_NAME env var
    
    Returns:
        Model configuration dict
    """
    if model_name is None:
        model_name = os.getenv("SIMPLE_MODEL_NAME", "llama-3.2-1b")  # Best for 1-2s latency
    
    model_name = model_name.lower()
    
    if model_name not in MODEL_CONFIGS:
        # Try to construct from SIMPLE_MODEL_PATH
        model_path = os.getenv("SIMPLE_MODEL_PATH", "")
        if model_path:
            # Extract model name from path
            base_name = os.path.basename(model_path).lower()
            # Try to match partial names
            for key, config in MODEL_CONFIGS.items():
                if key in base_name or base_name in key:
                    print(f"[Config] 🔍 Matched '{base_name}' to config '{key}'")
                    return config
        
        # Default fallback
        print(f"[Config] ⚠️ Model '{model_name}' not found, using default 'qwen3-4b-2507'")
        return MODEL_CONFIGS["qwen3-4b-2507"]
    
    return MODEL_CONFIGS[model_name]


def get_engine_dir(model_name: Optional[str] = None) -> str:
    """Get engine directory for a model"""
    config = get_model_config(model_name)
    engine_dir = config["engine_dir"]
    
    # Allow override via env var
    env_engine_dir = os.getenv("TENSORRT_ENGINE_DIR")
    if env_engine_dir:
        engine_dir = env_engine_dir
    
    return engine_dir


def get_tokenizer_dir(model_name: Optional[str] = None) -> str:
    """Get tokenizer directory for a model"""
    config = get_model_config(model_name)
    tokenizer_dir = config.get("tokenizer_dir", config["engine_dir"])
    
    # Allow override via env var
    env_tokenizer_dir = os.getenv("TENSORRT_TOKENIZER_DIR")
    if env_tokenizer_dir:
        tokenizer_dir = env_tokenizer_dir
    
    return tokenizer_dir


def validate_engine_dir(engine_dir: str) -> bool:
    """Check if engine directory exists and is valid"""
    if not os.path.exists(engine_dir):
        print(f"[Config] ❌ Engine directory not found: {engine_dir}")
        return False
    
    # Check for required files
    required_files = ["config.json"]  # TensorRT-LLM engines typically have this
    for req_file in required_files:
        req_path = os.path.join(engine_dir, req_file)
        if not os.path.exists(req_path):
            print(f"[Config] ⚠️ Required file not found: {req_path}")
            # Don't fail - some engines might have different structure
    
    return True


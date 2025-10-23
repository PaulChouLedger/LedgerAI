# =====================================
# Aura LLM Medical Container Configuration
# =====================================

import os

# === Temperature Configuration ===
# Granular temperature controls for different LLM operations
LLM_TEMPERATURE_SIMPLE = float(os.environ.get('LLM_TEMPERATURE_SIMPLE', '0.1'))      # For basic questions (L, C, T, S, O, D)
LLM_TEMPERATURE_COMPLEX = float(os.environ.get('LLM_TEMPERATURE_COMPLEX', '0.1'))     # For critical questions (A, R)
LLM_TEMPERATURE_NORMALIZATION = float(os.environ.get('LLM_TEMPERATURE_NORMALIZATION', '0.1'))  # For text normalization
LLM_TEMPERATURE_CREATIVE = float(os.environ.get('LLM_TEMPERATURE_CREATIVE', '0.6'))       # For creative responses
LLM_TEMPERATURE_ANALYSIS = float(os.environ.get('LLM_TEMPERATURE_ANALYSIS', '0.3'))       # For analysis tasks

# === Model Configuration ===
MODEL_PATH = os.environ.get('MODEL_PATH', '/models/Mistral-7B-Instruct-v0.3.Q4_K_M.gguf')
SIMPLE_MODEL_PATH = os.environ.get('SIMPLE_MODEL_PATH', '/models/Llama-3.2-1B-Instruct-Q4_K_M.gguf')
N_CTX = int(os.environ.get('N_CTX', '8192'))
SIMPLE_N_CTX = int(os.environ.get('SIMPLE_N_CTX', '2048'))

# === RAG Configuration ===
RAG_ENABLED = os.environ.get('RAG_ENABLED', 'true').lower() == 'true'
RAG_SERVICE_URL = os.environ.get('RAG_SERVICE_URL', 'http://localhost:11435')
RAG_TIMEOUT = int(os.environ.get('RAG_TIMEOUT', '10'))

# === Service Configuration ===
LLM_PORT = int(os.environ.get('LLM_PORT', '11434'))

# === Logging ===
LOG_LEVEL = os.environ.get('LOG_LEVEL', 'INFO')

# Use pre-built dustynv llama.cpp image
FROM dustynv/llama_cpp:b5283-r36.4-cu128-24.04

# Set working directory
WORKDIR /app

# Install required packages FIRST (this layer will be cached)
# This ensures packages are only installed once and cached for future builds
RUN pip3 install \
    flask \
    python-dotenv \
    numpy>=1.26.0 \
    requests \
    sentence-transformers \
    faiss-cpu \
    scikit-learn \
    joblib \
    fuzzywuzzy \
    python-Levenshtein \
    watchdog \
    PyPDF2==3.0.1 \
    python-docx==0.8.11 \
    openpyxl>=3.1.0 \
    --index-url https://pypi.org/simple


# Pre-download sentence transformer model (cached layer)
# Model will be cached in default location (/root/.cache/huggingface/) which is baked into image
RUN python3 -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-distilroberta-v1')" && \
    echo "✅ all-distilroberta-v1 model pre-downloaded into image cache"

# Copy models (cached - these are large and rarely change)
# Both 1.5B and 3B models are available - switch via SIMPLE_MODEL_PATH environment variable
# Default: 1.5B model (faster), fallback: 3B model (better reasoning)
COPY models/Qwen2.5-1.5B-Instruct.Q4_K_M.gguf /models/Qwen2.5-1.5B-Instruct.Q4_K_M.gguf
#COPY models/qwen2.5-3b-instruct-q4_k_m.gguf /models/qwen2.5-3b-instruct-q4_k_m.gguf
# === GENERIC CONVERSATIONAL CONTAINER ===
# This container provides general conversation with RAG-powered knowledge:
# - General conversational responses
# - RAG API integration for document Q&A
# - Simple, fast LLM interactions
# - No medical assessment components

# Copy Python application files
COPY container_rest.py ./
COPY conversation_manager.py ./

# Note: RAG module is now in /shared/rag/ (mounted via docker-compose.yml)

# Models already copied above (cached layer)

# === Environment defaults (can be overridden by .env or -e) ===
ENV SIMPLE_MODEL_PATH=/models/Qwen2.5-1.5B-Instruct.Q4_K_M.gguf
# To switch to 3B model, override with: /models/qwen2.5-3b-instruct-q4_k_m.gguf
ENV SIMPLE_CHAT_FORMAT=chatml
ENV SIMPLE_N_CTX=8192
ENV LLM_STOP=<|im_end|>

# Expose port for API
EXPOSE 11434

# Run the API (loads .env inside container if mounted)
CMD ["python3", "/app/container_rest.py"]

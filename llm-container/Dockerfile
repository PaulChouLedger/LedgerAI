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
    --index-url https://pypi.org/simple


# Pre-download sentence transformer model (cached layer)
RUN python3 -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-distilroberta-v1')" && \
    echo "✅ all-distilroberta-v1 model pre-downloaded into image"

# Copy models (cached - these are large and rarely change)
COPY models/Llama-3.2-1B-Instruct-Q4_K_M.gguf /models/Llama-3.2-1B-Instruct-Q4_K_M.gguf
COPY models/Nemotron-Mini-4B-Instruct-Q4_K_M.gguf /models/Nemotron-Mini-4B-Instruct-Q4_K_M.gguf

# === GENERIC CONVERSATIONAL CONTAINER ===
# This container provides general conversation with RAG-powered knowledge:
# - General conversational responses
# - RAG API integration for document Q&A
# - Simple, fast LLM interactions
# - No medical assessment components

# Copy Python application files
COPY container_rest.py ./

# Copy RAG module (organized structure)
COPY rag/ /app/rag/

# Models already copied above (cached layer)

# === Environment defaults (can be overridden by .env or -e) ===
ENV SIMPLE_MODEL_PATH=/models/Nemotron-Mini-4B-Instruct-Q4_K_M.gguf
ENV SIMPLE_CHAT_FORMAT=chatml
ENV SIMPLE_N_CTX=2048
ENV LLM_STOP=<|im_end|>,</s>

# Expose port for API
EXPOSE 11436

# Run the API (loads .env inside container if mounted)
CMD ["python3", "/app/container_rest.py"]

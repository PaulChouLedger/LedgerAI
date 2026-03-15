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

# Copy model (single Qwen2.5-7B Q8_0 for all tasks — conversation + RAG/CoT)
COPY models/qwen2.5-7b-instruct-q8_0*.gguf /models/
# === GENERIC CONVERSATIONAL CONTAINER ===
# This container provides general conversation with RAG-powered knowledge:
# - General conversational responses
# - RAG API integration for document Q&A
# - Simple, fast LLM interactions
# - No medical assessment components

# Copy Python application files
COPY container_rest.py ./
COPY conversation_manager.py ./
COPY cot_filter.py ./
COPY session_state.py ./
COPY rag_summary.py ./
COPY rag_cot.py ./
COPY conversation.py ./

# Note: RAG module is now in /shared/rag/ (mounted via docker-compose.yml)

# Models already copied above (cached layer)

# === Environment defaults (can be overridden by .env or -e) ===
# Single 7B model for all tasks (conversation + RAG/CoT)
ENV BASE_MODEL_PATH=/models/qwen2.5-7b-instruct-q8_0-00001-of-00003.gguf
ENV COT_MODEL_PATH=/models/qwen2.5-7b-instruct-q8_0-00001-of-00003.gguf
ENV SIMPLE_MODEL_PATH=/models/qwen2.5-7b-instruct-q8_0-00001-of-00003.gguf
ENV SIMPLE_CHAT_FORMAT=chatml
ENV SIMPLE_N_CTX=16384
ENV LLM_STOP=<|im_end|>

# Expose port for API
EXPOSE 11434

# Run the API (loads .env inside container if mounted)
CMD ["python3", "/app/container_rest.py"]

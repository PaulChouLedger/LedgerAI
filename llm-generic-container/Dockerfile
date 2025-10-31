# Use pre-built TensorRT-LLM image
FROM dustynv/tensorrt_llm:0.12-r36.4.0

# Set working directory
WORKDIR /app

# Install required packages
RUN pip3 install --no-cache-dir \
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
    huggingface-hub \
    transformers \
    accelerate \
    safetensors

# Pre-download sentence transformer model (cached layer)
RUN python3 -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-distilroberta-v1')" && \
    echo "✅ all-distilroberta-v1 model pre-downloaded into image"

# Copy models (cached - these are large and rarely change)
# Note: Models should be pre-converted to TensorRT format
COPY models/ /models/

# === GENERIC LLM CONTAINER WITH TENSORRT-LLM ===
# This container provides:
# - Qwen2.5-7B-Instruct for RAG and complex questions (TensorRT-LLM)
# - Llama-3.2-1B for simple tasks and greetings (TensorRT-LLM)
# - RAG integration for document-based Q&A
# - Streaming and non-streaming endpoints
# - CPU FAISS auto-ingestion for document indexing
# - Fuzzy matching for typos and transcription errors

# Copy Python application files
COPY container_rest.py ./
COPY fuzzy_matcher.py ./
COPY tensorrt_llm_wrapper.py ./
COPY tensorrt_models_config.py ./
COPY rag/ /app/rag/

# === Environment defaults (can be overridden by .env or -e) ===
ENV MODEL_PATH_COMPLEX=/models/qwen2.5-7b-instruct-trt/engine
ENV MODEL_PATH_SIMPLE=/models/llama-3.2-1b-instruct-trt/engine
ENV CHAT_FORMAT_COMPLEX=qwen
ENV CHAT_FORMAT_SIMPLE=llama-3
ENV N_CTX_COMPLEX=4096
ENV N_CTX_SIMPLE=2048

# Expose port for API
EXPOSE 11435

# Run the API (loads .env inside container if mounted)
CMD ["python3", "/app/container_rest.py"]

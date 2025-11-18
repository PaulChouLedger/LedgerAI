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
COPY models/Qwen2.5-1.5B-Instruct.Q4_K_M-medical.gguf /models/Qwen2.5-1.5B-Instruct.Q4_K_M-medical.gguf

# Copy static files (cached - these change less frequently than Python code)
COPY ml/ /app/ml/
COPY medical/guidelines/ /app/medical/guidelines/
COPY config/ /app/config/
# ML files (location_ml_model.pkl, location_ml_data.csv) - kept for future ML training but not actively used
# oldcarts_keywords.json - kept for compatibility

# === DIRECT ROUTING ARCHITECTURE ===
# This container provides simplified medical chatbot with direct routing:
# - Advanced Medical Navigator (USE_MEDICAL_NAVIGATOR=true) - hybrid LLM/RAG/FAISS
# - Adaptive diagnostic engine (default) - guideline-based assessment
# - No intermediate routing layers - cleaner, simpler, easier to debug
# - Smart LLM-based question generation and answer extraction
# - Fuzzy medical matching for typo correction
# - RAG integration for semantic similarity and medical knowledge
# - Single LLM support (Qwen2.5-1.5B for all tasks)
# - Session management built into container_rest.py

# Copy all Python application files in one layer (frequently changed - no caching)
# This ensures that any change to any Python file only invalidates this single layer
COPY container_rest.py advanced_medical_navigator.py fuzzy_medical_matcher.py ./

# Copy RAG module (organized structure)
COPY rag/ /app/rag/

# Note: medical_terms.json is mounted at /shared/medical_terms.json (see docker-compose.yml)

# Models already copied above (cached layer)

# === Environment defaults (can be overridden by .env or -e) ===
ENV SIMPLE_MODEL_PATH=/models/Qwen2.5-1.5B-Instruct.Q4_K_M-medical.gguf
ENV SIMPLE_CHAT_FORMAT=qwen
ENV SIMPLE_N_CTX=2048
ENV LLM_STOP=<|eot_id|>


# Expose port for API
EXPOSE 11434

# Run the API (loads .env inside container if mounted)
CMD ["python3", "/app/container_rest.py"]

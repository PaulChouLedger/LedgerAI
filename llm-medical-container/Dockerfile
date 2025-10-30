# Use pre-built dustynv llama.cpp image
FROM dustynv/llama_cpp:0.3.9-r36.4.0-cu128-24.04

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

# Ensure latest llama-cpp-python for GGUF compatibility
RUN pip3 install --upgrade llama-cpp-python

# Pre-download sentence transformer model (cached layer)
RUN python3 -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-distilroberta-v1')" && \
    echo "✅ all-distilroberta-v1 model pre-downloaded into image"

# Copy models (cached - these are large and rarely change)
COPY models/Llama-3.2-1B-Instruct-Q4_K_M.gguf /models/Llama-3.2-1B-Instruct-Q4_K_M.gguf
COPY models/Qwen3-4B-Instruct-2507-Q4_K_M.gguf /models/Qwen3-4B-Instruct-2507-Q4_K_M.gguf

# Copy static files (cached - these change less frequently than Python code)
COPY synonyms/ /app/synonyms/
COPY ml/ /app/ml/
COPY medical/guidelines/ /app/medical/guidelines/
COPY config/ /app/config/
# ML files (location_ml_model.pkl, location_ml_data.csv) - kept for future ML training but not actively used
# oldcarts_keywords.json - kept for compatibility

# === CLINICIAN MODE ARCHITECTURE ===
# This container provides a focused medical chatbot with:
# - Clinician mode for all medical interactions (symptoms, knowledge, greetings)
# - Adaptive diagnostic engine with guideline-based assessment
# - Smart LLM-based age extraction for natural language processing
# - Fuzzy medical matching for typo correction and robust input handling
# - RAG API integration for semantic similarity and medical knowledge
# - Single LLM support (Llama-3.2-1B for all tasks)
# - Complete session management with proper reset functionality

# Copy all Python application files in one layer (frequently changed - no caching)
# This ensures that any change to any Python file only invalidates this single layer
COPY container_rest.py clinician_mode.py adaptive_diagnostic_engine.py thinking_fillers.py feedback_guide.py fuzzy_medical_matcher.py ./

# Copy RAG module (organized structure)
COPY rag/ /app/rag/

# Note: medical_terms.json is mounted at /shared/medical_terms.json (see docker-compose.yml)

# Models already copied above (cached layer)

# === Environment defaults (can be overridden by .env or -e) ===
ENV SIMPLE_MODEL_PATH=/models/Llama-3.2-1B-Instruct-Q4_K_M.gguf
ENV SIMPLE_CHAT_FORMAT=llama-3
ENV SIMPLE_N_CTX=2048

# Expose port for API
EXPOSE 11434

# Run the API (loads .env inside container if mounted)
CMD ["python3", "/app/container_rest.py"]

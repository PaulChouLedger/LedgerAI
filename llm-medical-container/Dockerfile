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
    --index-url https://pypi.org/simple

# Pre-download sentence transformer model (cached layer)
RUN python3 -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-distilroberta-v1')" && \
    echo "✅ all-distilroberta-v1 model pre-downloaded into image"

# Copy models (cached - these are large and rarely change)
COPY models/Llama-3.2-1B-Instruct-Q4_K_M.gguf /models/Llama-3.2-1B-Instruct-Q4_K_M.gguf
COPY models/Mistral-7B-Instruct-v0.3.Q4_K_M.gguf /models/Mistral-7B-Instruct-v0.3.Q4_K_M.gguf

# Copy static files (cached - these change less frequently than Python code)
COPY synonyms/ /app/synonyms/
COPY ml/ /app/ml/
COPY medical/guidelines/ /app/medical/guidelines/
# ML files (location_ml_model.pkl, location_ml_data.csv) - kept for future ML training but not actively used
# config/medical_term_mappings.json - deprecated, all data now in synonyms/
# oldcarts_keywords.json - kept for compatibility

# === CLINICIAN MODE ARCHITECTURE ===
# This container provides a focused medical chatbot with:
# - Clinician mode for all medical interactions (symptoms, knowledge, greetings)
# - Adaptive diagnostic engine with guideline-based assessment
# - Smart LLM-based age extraction for natural language processing
# - Fuzzy medical matching for typo correction and robust input handling
# - RAG API integration for semantic similarity and medical knowledge
# - Dual LLM support (Llama-1B for simple tasks, Mistral-7B for complex reasoning)
# - Complete session management with proper reset functionality

# Copy all Python application files in one layer (frequently changed - no caching)
# This ensures that any change to any Python file only invalidates this single layer
COPY container_rest.py clinician_mode.py adaptive_diagnostic_engine.py thinking_fillers.py feedback_guide.py fuzzy_medical_matcher.py ./

# Copy RAG module (organized structure)
COPY rag/ /app/rag/

# Note: medical_terms.json is mounted at /shared/medical_terms.json (see docker-compose.yml)

# Models already copied above (cached layer)

# === Environment defaults (can be overridden by .env or -e) ===
ENV MODEL_PATH=/models/Mistral-7B-Instruct-v0.3.Q4_K_M.gguf
ENV CHAT_FORMAT=mistral-instruct
ENV N_CTX=8192

# Expose port for API
EXPOSE 11434

# Run the API (loads .env inside container if mounted)
CMD ["python3", "/app/container_rest.py"]

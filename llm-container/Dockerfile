# Use DustyNV TensorRT-LLM base image (Jetson optimized)
# This image includes TensorRT-LLM pre-installed and configured for Jetson Orin
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
    watchdog


# Pre-download sentence transformer model (cached layer)
RUN python3 -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-distilroberta-v1')" && \
    echo "✅ all-distilroberta-v1 model pre-downloaded into image"

# Copy TensorRT-LLM engines (if pre-built)
# Note: TensorRT-LLM engines must be built separately using trtllm-build
# Engines should be placed in /models/tensorrt-llm/{model-name}/
# Example structure:
#   /models/tensorrt-llm/qwen3-4b-instruct-2507/
#     - config.json
#     - engine files...
RUN mkdir -p /models/tensorrt-llm

# Copy static files (cached - these change less frequently than Python code)
COPY synonyms/ /app/synonyms/
COPY ml/ /app/ml/
COPY medical/guidelines/ /app/medical/guidelines/
COPY config/ /app/config/
# ML files (location_ml_model.pkl, location_ml_data.csv) - kept for future ML training but not actively used
# oldcarts_keywords.json - kept for compatibility

# === UNIFIED LLM CONTAINER ARCHITECTURE ===
# This container provides both medical and generic conversation modes:
# 
# Medical Mode (USE_MEDICAL_MODE=true):
# - Clinician mode for all medical interactions (symptoms, knowledge, greetings)
# - Adaptive diagnostic engine with guideline-based assessment
# - OLDCARTS-based questioning with guideline matching
# 
# Generic Mode (USE_MEDICAL_MODE=false):
# - General conversation handling
# - RAG-powered document Q&A
# 
# Both modes use:
# - TensorRT-LLM for optimized GPU inference (Jetson Orin)
# - RAG API integration for semantic similarity and knowledge retrieval
# - Complete session management with proper reset functionality

# Copy all Python application files in one layer (frequently changed - no caching)
# This ensures that any change to any Python file only invalidates this single layer
COPY container_rest.py clinician_mode.py adaptive_diagnostic_engine.py thinking_fillers.py feedback_guide.py fuzzy_medical_matcher.py tensorrt_llm_wrapper.py tensorrt_models_config.py ./

# Copy RAG module (organized structure)
COPY rag/ /app/rag/

# Note: medical_terms.json is mounted at /shared/medical_terms.json (see docker-compose.yml)

# Models already copied above (cached layer)

# === Environment defaults (can be overridden by .env or -e) ===
# TensorRT-LLM configuration
ENV TENSORRT_ENGINES_BASE=/models/tensorrt-llm
ENV SIMPLE_MODEL_NAME=llama-3.2-1b
ENV SIMPLE_CHAT_FORMAT=llama-3
ENV SIMPLE_N_CTX=2048

# Expose port for API
EXPOSE 11434

# Run the API (loads .env inside container if mounted)
CMD ["python3", "/app/container_rest.py"]

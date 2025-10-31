# TensorRT-LLM Container - Using DustyNV pre-built TensorRT-LLM container
# Based on Jetson AI Lab: https://www.jetson-ai-lab.com/tensorrt_llm.html
# 
# Note: TensorRT-LLM pip installation is NOT supported on Jetson/Tegra systems
# The dustynv/tensorrt_llm container has TensorRT-LLM pre-built and optimized for Jetson
# Conversion scripts are located at /opt/TensorRT-LLM/examples/llama/
FROM dustynv/tensorrt_llm:0.12-r36.4.0

# Set working directory
WORKDIR /app

# TensorRT-LLM is already installed in the base image
# CUDA environment is already configured

# Install additional required packages for our application
# Note: PyTorch, TensorRT-LLM, transformers, and safetensors are already installed in the base image
# Use PyPI index explicitly to avoid Jetson repository timeouts
# --ignore-installed ensures we get the versions we need even if some packages exist
# Note: Don't reinstall safetensors or transformers as they may conflict with base image versions
RUN pip3 install --no-cache-dir --ignore-installed \
    --index-url https://pypi.org/simple/ \
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
# Make this optional - if it fails, the model will download at runtime
RUN python3 -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-distilroberta-v1')" && \
    echo "✅ all-distilroberta-v1 model pre-downloaded into image" || \
    echo "⚠️  Model pre-download skipped - will download at runtime"

# Create directory for TensorRT-LLM engines
RUN mkdir -p /models/tensorrt-llm

# Copy static files (cached - these change less frequently than Python code)
COPY synonyms/ /app/synonyms/
COPY ml/ /app/ml/
COPY medical/guidelines/ /app/medical/guidelines/
COPY config/ /app/config/

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
COPY container_rest.py clinician_mode.py adaptive_diagnostic_engine.py thinking_fillers.py feedback_guide.py fuzzy_medical_matcher.py tensorrt_llm_wrapper.py tensorrt_models_config.py ./

# Copy RAG module (organized structure)
COPY rag/ /app/rag/

# Copy build scripts (for engine building)
COPY scripts/ /app/scripts/

# Note: medical_terms.json is mounted at /shared/medical_terms.json (see docker-compose.yml)

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

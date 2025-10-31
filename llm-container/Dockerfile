# TensorRT-LLM Container - Installed via pip (following official installation guide)
# Base: DustyNV PyTorch image (Jetson optimized, includes CUDA)
# Reference: https://nvidia.github.io/TensorRT-LLM/installation/linux.html
FROM dustynv/pytorch:2.7-r36.4.0-cu128-24.04

# Set working directory
WORKDIR /app

# Set CUDA_HOME (required by TensorRT-LLM)
# DustyNV images typically have CUDA in /usr/local/cuda or /usr/local/cuda-12.x
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# Install prerequisites per TensorRT-LLM documentation
# Note: For Jetson, some packages may already be installed
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Optional: Install MPI (for distributed inference if needed)
# RUN apt-get update && apt-get install -y libopenmpi-dev && rm -rf /var/lib/apt/lists/*

# Upgrade pip and setuptools (required by TensorRT-LLM)
# Use PyPI index for TensorRT-LLM installation (not Jetson-specific index)
RUN pip3 install --upgrade pip setuptools --index-url https://pypi.org/simple/

# Install TensorRT-LLM via pip
# This will download and install TensorRT-LLM with all dependencies
# Use PyPI index explicitly to avoid Jetson repository timeouts
# Review license terms of third-party open source projects before use
RUN pip3 install --no-cache-dir tensorrt_llm --index-url https://pypi.org/simple/

# Install additional required packages for our application
# Note: PyTorch is already installed in the base image
# Use PyPI index for standard packages
RUN pip3 install --no-cache-dir \
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
    watchdog \
    transformers \
    safetensors

# Pre-download sentence transformer model (cached layer)
RUN python3 -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-distilroberta-v1')" && \
    echo "✅ all-distilroberta-v1 model pre-downloaded into image"

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

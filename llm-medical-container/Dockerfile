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
COPY ml/location_ml_model.pkl .
COPY ml/location_ml_data.csv .
COPY config/medical_rules.json .
COPY config/medical_term_mappings.json .
COPY oldcarts_keywords.json .

# === CLINICIAN MODE ARCHITECTURE ===
# This container provides a focused medical chatbot with:
# - Clinician mode for all medical interactions (symptoms, knowledge, greetings)
# - Adaptive diagnostic engine with guideline-based assessment
# - RAG API integration for semantic similarity and medical knowledge
# - Dual LLM support (Llama-1B for simple tasks, Mistral-7B for complex reasoning)

# Copy Python application files
COPY container_rest.py .
COPY clinician_mode.py .
COPY adaptive_diagnostic_engine.py .
COPY thinking_fillers.py .
COPY rag_client.py .
COPY feedback_guide.py .

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

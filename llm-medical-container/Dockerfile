# Use pre-built dustynv llama.cpp image
FROM dustynv/llama_cpp:0.3.9-r36.4.0-cu128-24.04

# Set working directory
WORKDIR /app

# === CLINICIAN MODE ARCHITECTURE ===
# This container provides a focused medical chatbot with:
# - Clinician mode for all medical interactions (symptoms, knowledge, greetings)
# - Adaptive diagnostic engine with guideline-based assessment
# - RAG API integration for semantic similarity and medical knowledge
# - Dual LLM support (Llama-1B for simple tasks, Mistral-7B for complex reasoning)

# Copy application logic (clinician mode architecture)
# validation.py removed - ML system handles all validation
COPY container_rest.py .
COPY clinician_mode.py .
COPY adaptive_diagnostic_engine.py .
COPY thinking_fillers.py .
COPY rag_client.py .
COPY synonyms/ /app/synonyms/

# Copy ML system components
COPY ml/ /app/ml/
COPY feedback_guide.py .
    
# Copy medical guidelines (JSON files with chief complaint triggers and scoring)
# These are used by the adaptive diagnostic engine for multi-guideline evaluation
# Note: guidelines should be in llm-medical-container/medical/guidelines/ (inside build context)
COPY medical/guidelines/ /app/medical/guidelines/

# Copy ML models and data
COPY ml/location_ml_model.pkl .
COPY ml/location_ml_data.csv .
COPY config/medical_rules.json .
COPY config/medical_term_mappings.json .

# Copy OLDCARTS keywords file
COPY oldcarts_keywords.json .

# Note: medical_terms.json is mounted at /shared/medical_terms.json (see docker-compose.yml)


# Copy example models (others can be mounted at runtime)
#COPY models/qwen2.5-1.5b-instruct-q4_0.gguf /models/qwen2.5-1.5b-instruct-q4_0.gguf
#COPY models/Llama-3.2-3B-Instruct-Q4_K_M.gguf /models/Llama-3.2-3B-Instruct-Q4_K_M.gguf
COPY models/Llama-3.2-1B-Instruct-Q4_K_M.gguf /models/Llama-3.2-1B-Instruct-Q4_K_M.gguf
#COPY models/Llama-3.2-8B-Instruct-Q4_K_M.gguf /models/Llama-3.2-8B-Instruct-Q4_K_M.gguf
COPY models/Mistral-7B-Instruct-v0.3.Q4_K_M.gguf /models/Mistral-7B-Instruct-v0.3.Q4_K_M.gguf

# Install Python dependencies (LLM + Learning System)
RUN pip3 install --no-cache-dir --ignore-installed \
    flask \
    python-dotenv \
    numpy>=1.26.0 \
    requests \
    sentence-transformers \
    faiss-cpu \
    scikit-learn \
    joblib \
    --index-url https://pypi.org/simple

# Pre-download all-distilroberta-v1 model (benchmarked as best, makes it WiFi-independent for CPU mode)
RUN python3 -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-distilroberta-v1')" && \
    echo "✅ all-distilroberta-v1 model pre-downloaded into image"


# === Environment defaults (can be overridden by .env or -e) ===
ENV MODEL_PATH=/models/Mistral-7B-Instruct-v0.3.Q4_K_M.gguf
ENV CHAT_FORMAT=mistral-instruct
ENV N_CTX=8192

# Expose port for API
EXPOSE 11434

# Run the API (loads .env inside container if mounted)
CMD ["python3", "/app/container_rest.py"]

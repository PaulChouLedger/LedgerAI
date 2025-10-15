# Use NVIDIA Jetson-compatible llama.cpp image
FROM llama_cpp:r36.4.tegra-aarch64-cu126-22.04

# Set working directory
WORKDIR /app

# Copy application logic (modular architecture)
COPY validation.py .
COPY intent_classifier.py .
COPY dynamic_triage.py .
COPY outcome_cache.py .
COPY container_rest.py .
COPY nlg.py .
COPY router.py .
COPY casual.py .
COPY thinker.py .
COPY triage.py .
COPY unified_medical_mode.py .
COPY medical_rag.py .
COPY adaptive_diagnostic_engine.py .
COPY triage_defs/ /app/triage_defs/
COPY synonyms/ /app/synonyms/

# Copy medical guidelines (JSON files with chief complaint triggers and scoring)
# These are used by the adaptive diagnostic engine for multi-guideline evaluation
# Note: guidelines should be in llm-container/medical/guidelines/ (inside build context)
COPY medical/guidelines/ /app/medical/guidelines/

# Note: medical_terms.json is mounted at /shared/medical_terms.json (see docker-compose.yml)


# Copy example models (others can be mounted at runtime)
#COPY models/qwen2.5-1.5b-instruct-q4_0.gguf /models/qwen2.5-1.5b-instruct-q4_0.gguf
#COPY models/Llama-3.2-3B-Instruct-Q4_K_M.gguf /models/Llama-3.2-3B-Instruct-Q4_K_M.gguf
COPY models/Llama-3.2-1B-Instruct-Q4_K_M.gguf /models/Llama-3.2-1B-Instruct-Q4_K_M.gguf

# Install Python dependencies (LLM only)
RUN pip3 install --no-cache-dir --ignore-installed \
    flask \
    python-dotenv \
    numpy==1.24.4 \
    --index-url https://pypi.org/simple


# === Environment defaults (can be overridden by .env or -e) ===
ENV MODEL_PATH=/models/Llama-3.2-1B-Instruct-Q4_K_M.gguf
ENV CHAT_FORMAT=llama-3
ENV N_CTX=1024

# Expose port for API
EXPOSE 11434

# Run the API (loads .env inside container if mounted)
CMD ["python3", "/app/container_rest.py"]

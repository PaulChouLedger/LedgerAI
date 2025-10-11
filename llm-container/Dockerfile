# Use NVIDIA Jetson-compatible llama.cpp image
FROM llama_cpp:r36.4.tegra-aarch64-cu126-22.04

# Set working directory
WORKDIR /app

# Copy application logic (modular architecture)
COPY container_rest.py .
COPY nlg.py .
COPY router.py .
COPY casual.py .
COPY thinker.py .
COPY triage.py .
COPY clinician.py .
COPY llm_inference.py .
COPY triage_defs/ /app/triage_defs/
COPY synonyms/ /app/synonyms/


# Copy example models (others can be mounted at runtime)
COPY models/qwen2.5-1.5b-instruct-q4_0.gguf /models/qwen2.5-1.5b-instruct-q4_0.gguf
COPY models/Qwen_Qwen3-1.7B-Q4_K_M.gguf /models/Qwen_Qwen3-1.7B-Q4_K_M.gguf

# Install Python dependencies (LLM only)
RUN pip3 install --no-cache-dir --ignore-installed \
    flask \
    python-dotenv \
    numpy==1.24.4 \
    --index-url https://pypi.org/simple


# === Environment defaults (can be overridden by .env or -e) ===
ENV MODEL_PATH=/models/qwen2.5-1.5b-instruct-q4_0.gguf
ENV CHAT_FORMAT=qwen
ENV N_CTX=1024

# Expose port for API
EXPOSE 11434

# Run the API (loads .env inside container if mounted)
CMD ["python3", "/app/container_rest.py"]

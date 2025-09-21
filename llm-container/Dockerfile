# Use NVIDIA Jetson-compatible llama.cpp image
FROM dustynv/llama_cpp:0.3.7-r36.4.0

# Set working directory
WORKDIR /app

# Copy application logic
COPY container_rest.py .
COPY triage_definitions.json .

# Copy example models (others can be mounted at runtime)
COPY models/qwen2.5-1.5b-instruct-q4_0.gguf /models/qwen2.5-1.5b-instruct-q4_0.gguf
COPY models/smollm2-1.7b-instruct-q4_k_m.gguf /models/smollm2-1.7b-instruct-q4_k_m.gguf
COPY models/DeepSeek-R1-Distill-Qwen-1.5B-Medical-COT.i1-Q4_K_M.gguf /models/DeepSeek-R1-Distill-Qwen-1.5B-Medical-COT.i1-Q4_K_M.gguf

# Install Python dependencies
RUN pip3 install --no-cache-dir --ignore-installed flask python-dotenv --index-url https://pypi.org/simple

# === Environment defaults (can be overridden by .env or -e) ===
ENV MODEL_PATH=/models/qwen2.5-1.5b-instruct-q4_0.gguf
ENV CHAT_FORMAT=qwen
ENV N_CTX=1024

# Expose port for API
EXPOSE 11434

# Run the API (loads .env inside container if mounted)
CMD ["python3", "/app/container_rest.py"]

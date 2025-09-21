# Use NVIDIA Jetson-compatible llama.cpp image
FROM dustynv/llama_cpp:0.3.7-r36.4.0

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

# Install uv package manager
RUN pip3 install --no-cache-dir uv --ignore-installed --index-url https://pypi.org/simple

# Copy package files
COPY pyproject.toml /app/
COPY src/ /app/src/
COPY models/mistral.gguf /models/mistral.gguf

# Copy example models (others can be mounted at runtime)
COPY models/qwen2.5-1.5b-instruct-q4_0.gguf /models/qwen2.5-1.5b-instruct-q4_0.gguf
COPY models/smollm2-1.7b-instruct-q4_k_m.gguf /models/smollm2-1.7b-instruct-q4_k_m.gguf
COPY models/DeepSeek-R1-Distill-Qwen-1.5B-Medical-COT.i1-Q4_K_M.gguf /models/DeepSeek-R1-Distill-Qwen-1.5B-Medical-COT.i1-Q4_K_M.gguf

# === Environment defaults (can be overridden by .env or -e) ===
ENV MODEL_PATH=/models/qwen2.5-1.5b-instruct-q4_0.gguf
ENV CHAT_FORMAT=qwen
ENV N_CTX=1024

# Install the package using uv
RUN uv sync

# Expose port for API
EXPOSE 11434

CMD ["uv", "run", "/app/src/aura_llm/container_rest.py"]

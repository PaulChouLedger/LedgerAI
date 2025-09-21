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
COPY models /models

# Copy triage definitions and create state directory
COPY src/aura_llm/*.json /app/triage_defs/
RUN mkdir -p /app/state

# === Environment defaults (can be overridden by .env or -e) ===
ENV MODEL_PATH=/models/qwen2.5-1.5b-instruct-q4_0.gguf
ENV CHAT_FORMAT=qwen
ENV N_CTX=1024
ENV TRIAGE_STATE_DIR=/app/state
ENV TRIAGE_DEFINITIONS_DIR=/app/triage_defs

# Install the package using uv
RUN uv sync

# Expose port for API
EXPOSE 11434

CMD ["uv", "run", "/app/src/aura_llm/container_rest.py"]

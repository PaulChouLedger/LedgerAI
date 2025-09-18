FROM dustynv/llama_cpp:0.3.7-r36.4.0

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

# Install uv package manager
RUN pip3 install --no-cache-dir uv --ignore-installed --index-url https://pypi.org/simple

# Copy package files
COPY pyproject.toml /app/
COPY src/ /app/src/
COPY models/mistral.gguf /models/mistral.gguf
COPY models/qwen2.5-1.5b-instruct-q4_0.gguf /models/qwen2.5-1.5b-instruct-q4_0.gguf

ENV MODEL_NAME qwen2.5-1.5b-instruct-q4_0

# Install the package using uv
RUN uv sync

EXPOSE 11434

CMD ["uv", "run", "/app/src/aura_llm/container_rest.py"]

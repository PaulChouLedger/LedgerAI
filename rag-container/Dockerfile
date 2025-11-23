# Modern RAG container with cuDF, cuVS, and LlamaIndex
FROM rapidsai/rapidsai:23.08a-cuda11.8.0-py3.10

# Switch to root for system package installation
USER root

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

# Install uv package manager
RUN pip3 install --no-cache-dir uv --ignore-installed --index-url https://pypi.org/simple

# Copy package files
COPY pyproject.toml /app/
COPY src/ /app/src/

# Install the package using uv
RUN uv sync

CMD ["uv", "run", "/app/src/aura_rag/container_rest.py"]

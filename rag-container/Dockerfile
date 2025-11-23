# Modern RAG container with cuDF, cuVS, and LlamaIndex
FROM rapidsai/rapidsai:23.08a-cuda11.8.0-py3.10

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

# Install uv package manager
RUN pip install --upgrade pip --break-system-packages && \
    pip install --break-system-packages uv --extra-index-url=https://pypi.nvidia.com

# Copy package files
COPY pyproject.toml /app/
COPY src/ /app/src/

# Install the package using uv
RUN uv sync

CMD ["uv", "run", "/app/src/aura_rag/container_rest.py"]

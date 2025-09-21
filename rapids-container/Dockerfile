# Modern RAG container with cuDF, cuVS, and LlamaIndex
FROM rapidsai/rapidsai:24.02-cuda12.0-runtime-ubuntu22.04

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

CMD ["uv", "run", "/app/src/aura_rapids/container_rest.py"]

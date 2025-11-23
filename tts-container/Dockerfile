# Dockerfile — aura-tts using Piper TTS as base
FROM dustynv/piper-tts:r36.4.0

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

# Install uv package manager
RUN pip3 install --no-cache-dir uv --ignore-installed --index-url https://pypi.org/simple

# Copy package files
COPY pyproject.toml /app/
COPY src/ /app/src/
# Ensure model + config are in the correct path inside the image
COPY models /models

# Install the package using uv
RUN uv sync

EXPOSE 5002

CMD ["uv", "run", "/app/src/aura_tts/container_rest.py"]

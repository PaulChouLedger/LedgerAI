FROM dustynv/whisper_trt:r36.3.0

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

# Install uv package manager
RUN pip3 install --no-cache-dir uv --ignore-installed --index-url https://pypi.org/simple

# Copy package files
COPY pyproject.toml /app/
COPY src/ /app/src/
#COPY cache/whisper /root/.cache/whisper
#COPY cache/whisper_trt /root/.cache/whisper_trt

# Patch both whisper_trt and OpenAI whisper cache dirs
RUN rm -f /root/.cache/whisper_trt && mkdir -p /root/.cache/whisper_trt && \
    rm -f /root/.cache/whisper && mkdir -p /root/.cache/whisper && \
    sed -i 's/os.makedirs(_CACHE_DIR)/os.makedirs(_CACHE_DIR, exist_ok=True)/' /opt/whisper_trt/whisper_trt/cache.py

# Install the package using uv
# RUN uv sync
RUN pip install .

EXPOSE 5000
CMD ["python", "/app/src/aura_whisper/container_rest.py"]

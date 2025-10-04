# Use NVIDIA PyTorch base image (includes PyTorch and CUDA)
FROM nvcr.io/nvidia/pytorch:23.10-py3

WORKDIR /app

# Copy REST API and build script
COPY container_rest.py /app/container_rest.py
COPY build_ctranslate2.sh /app/build_ctranslate2.sh

# Install system dependencies for CTranslate2
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    && rm -rf /var/lib/apt/lists/*

# Build CTranslate2 from source
RUN chmod +x /app/build_ctranslate2.sh && /app/build_ctranslate2.sh

# Install faster-whisper and REST API dependencies
RUN pip3 install faster-whisper flask soundfile scipy

# Create cache directories
RUN mkdir -p /app/cache/whisper /root/.cache/huggingface

EXPOSE 5000
CMD ["python3", "/app/container_rest.py"]

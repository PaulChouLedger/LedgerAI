# Use NVIDIA PyTorch base image (includes PyTorch and CUDA)
FROM nvcr.io/nvidia/pytorch:23.10-py3

WORKDIR /app

# Copy REST API
COPY container_rest.py /app/container_rest.py

# Install CTranslate2 from Jetson index (pre-built ARM64 wheels)
RUN pip3 install --index-url https://pypi.jetson-ai-lab.io/jp6/cu128 ctranslate2

# Install faster-whisper and REST API dependencies
RUN pip3 install faster-whisper flask soundfile scipy

# Create cache directory
RUN mkdir -p /app/cache/whisper

EXPOSE 5000
CMD ["python3", "/app/container_rest.py"]

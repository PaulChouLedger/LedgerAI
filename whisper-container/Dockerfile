# Use the pre-built dustynv faster-whisper image
FROM dustynv/faster-whisper:r36.4.0-cu128-24.04

WORKDIR /app

# Note: medical_terms.json is mounted at /shared/medical_terms.json (see docker-compose.yml)

# Install REST API dependencies
# Note: Base image should have torch, but ensure it's accessible for GPU memory management
RUN pip3 install --ignore-installed --index-url https://pypi.org/simple flask soundfile scipy

# === Baked-in Models ===

# distil-small.en (fast, lower accuracy) - commented out
RUN mkdir -p /root/.cache/huggingface/hub/models--Systran--faster-distil-whisper-small.en/snapshots/ef77d90526ccd62cde3808ee70626a01e5cf83e4
COPY models--Systran--faster-distil-whisper-small.en/snapshots/ef77d90526ccd62cde3808ee70626a01e5cf83e4/* /root/.cache/huggingface/hub/models--Systran--faster-distil-whisper-small.en/snapshots/ef77d90526ccd62cde3808ee70626a01e5cf83e4/

# distil-large-v3.5-ct2 (best accuracy/speed balance - 1.5x faster than turbo, better short-form accuracy) ⭐ RECOMMENDED
#COPY models--distil-whisper--distil-large-v3.5-ct2/ /root/.cache/huggingface/hub/models--distil-whisper--distil-large-v3.5-ct2/

# Copy REST API (Python code changes - keep as last layer for better caching)
COPY container_rest.py /app/container_rest.py

EXPOSE 5000
CMD ["python3", "/app/container_rest.py"]
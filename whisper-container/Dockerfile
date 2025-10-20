# Use the pre-built dustynv faster-whisper image
FROM dustynv/faster-whisper:r36.4.0-cu128-24.04

WORKDIR /app

# Copy REST API
COPY container_rest.py /app/container_rest.py

# Note: medical_terms.json is mounted at /shared/medical_terms.json (see docker-compose.yml)

# Install REST API dependencies
RUN pip3 install --ignore-installed --index-url https://pypi.org/simple flask soundfile scipy

# === Baked-in Models ===

# distil-small.en (fast, lower accuracy) - commented out
RUN mkdir -p /root/.cache/huggingface/hub/models--Systran--faster-distil-whisper-small.en/snapshots/ef77d90526ccd62cde3808ee70626a01e5cf83e4
COPY models--Systran--faster-distil-whisper-small.en/snapshots/ef77d90526ccd62cde3808ee70626a01e5cf83e4/* /root/.cache/huggingface/hub/models--Systran--faster-distil-whisper-small.en/snapshots/ef77d90526ccd62cde3808ee70626a01e5cf83e4/

# distil-large-v3 (excellent accuracy, low latency) - commented out
# COPY models--Systran--faster-distil-whisper-large-v3/ /root/.cache/huggingface/hub/models--Systran--faster-distil-whisper-large-v3/

EXPOSE 5000
CMD ["python3", "/app/container_rest.py"]
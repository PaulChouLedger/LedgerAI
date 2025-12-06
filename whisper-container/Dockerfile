# Use the pre-built dustynv faster-whisper image
FROM dustynv/faster-whisper:r36.4.0-cu128-24.04

WORKDIR /app

# Copy REST API
COPY container_rest.py /app/container_rest.py

# Note: medical_terms.json is mounted at /shared/medical_terms.json (see docker-compose.yml)

# Install REST API dependencies
# Note: Base image should have torch, but ensure it's accessible for GPU memory management
RUN pip3 install --ignore-installed --index-url https://pypi.org/simple flask soundfile scipy

# === Baked-in Models ===

# distil-small.en (fast, lower accuracy) - commented out
RUN mkdir -p /root/.cache/huggingface/hub/models--Systran--faster-distil-whisper-small.en/snapshots/ef77d90526ccd62cde3808ee70626a01e5cf83e4
COPY models--Systran--faster-distil-whisper-small.en/snapshots/ef77d90526ccd62cde3808ee70626a01e5cf83e4/* /root/.cache/huggingface/hub/models--Systran--faster-distil-whisper-small.en/snapshots/ef77d90526ccd62cde3808ee70626a01e5cf83e4/

# large-v3-turbo (best accuracy, higher latency)
COPY models--mobiuslabsgmbh--faster-whisper-large-v3-turbo/ /root/.cache/huggingface/hub/models--mobiuslabsgmbh--faster-whisper-large-v3-turbo/

EXPOSE 5000
CMD ["python3", "/app/container_rest.py"]
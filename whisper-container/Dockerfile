# Use the jetson-containers built faster-whisper image
FROM faster-whisper:r36.4.tegra-aarch64-cu126-22.04

WORKDIR /app

# Copy REST API
COPY container_rest.py /app/container_rest.py

# Copy shared medical terms (single source of truth at repo_root/shared/)
COPY ../shared/medical_terms.json /app/medical_terms.json

# Install REST API dependencies
RUN pip3 install --ignore-installed flask soundfile scipy

# === Baked-in Models ===

# distil-small.en (fast, lower accuracy) - commented out
RUN mkdir -p /root/.cache/huggingface/hub/models--Systran--faster-distil-whisper-small.en/snapshots/ef77d90526ccd62cde3808ee70626a01e5cf83e4
COPY models--Systran--faster-distil-whisper-small.en/snapshots/ef77d90526ccd62cde3808ee70626a01e5cf83e4/* /root/.cache/huggingface/hub/models--Systran--faster-distil-whisper-small.en/snapshots/ef77d90526ccd62cde3808ee70626a01e5cf83e4/

# distil-large-v3 (excellent accuracy, low latency) - commented out
# COPY models--Systran--faster-distil-whisper-large-v3/ /root/.cache/huggingface/hub/models--Systran--faster-distil-whisper-large-v3/

EXPOSE 5000
CMD ["python3", "/app/container_rest.py"]
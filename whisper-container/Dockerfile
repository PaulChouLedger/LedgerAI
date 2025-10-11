# Use the jetson-containers built faster-whisper image
FROM faster-whisper:r36.4.tegra-aarch64-cu126-22.04

WORKDIR /app

# Copy REST API and medical vocabulary
COPY container_rest.py /app/container_rest.py
COPY medical_terms.json /app/medical_terms.json

# Install REST API dependencies
RUN pip3 install --ignore-installed flask soundfile scipy

# === Baked-in Models ===

# distil-small.en (fast, lower accuracy) - commented out
 RUN mkdir -p /root/.cache/huggingface/hub/models--Systran--faster-distil-whisper-small.en/snapshots/ef77d90526ccd62cde3808ee70626a01e5cf83e4
 COPY models--Systran--faster-distil-whisper-small.en/snapshots/ef77d90526ccd62cde3808ee70626a01e5cf83e4/* /root/.cache/huggingface/hub/models--Systran--faster-distil-whisper-small.en/snapshots/ef77d90526ccd62cde3808ee70626a01e5cf83e4/

# large-v3-turbo (best accuracy, higher latency) - commented out due to latency
# RUN mkdir -p /root/.cache/huggingface/hub/models--mobiuslabsgmbh--faster-whisper-large-v3-turbo
# COPY models--mobiuslabsgmbh--faster-whisper-large-v3-turbo/ /root/.cache/huggingface/hub/models--mobiuslabsgmbh--faster-whisper-large-v3-turbo/

# distil-large-v3 (excellent accuracy, low latency) - ACTIVE
# RUN mkdir -p /root/.cache/huggingface/hub/models--Systran--faster-distil-whisper-large-v3
# COPY models--Systran--faster-distil-whisper-large-v3/ /root/.cache/huggingface/hub/models--Systran--faster-distil-whisper-large-v3/

EXPOSE 5000
CMD ["python3", "/app/container_rest.py"]
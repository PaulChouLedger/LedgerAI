# Use the jetson-containers built faster-whisper image
FROM faster-whisper:r36.4.tegra-aarch64-cu126-22.04

WORKDIR /app

# Copy REST API
COPY container_rest.py /app/container_rest.py

# Copy the pre-downloaded model
COPY models--Systran--faster-distil-small-en /root/.cache/huggingface/hub/models--Systran--faster-distil-small-en

# Install REST API dependencies
RUN pip3 install --ignore-installed flask soundfile scipy

EXPOSE 5000
CMD ["python3", "/app/container_rest.py"]
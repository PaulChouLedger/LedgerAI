# Use the jetson-containers built faster-whisper image
FROM faster-whisper:r36.4.tegra-aarch64-cu126-22.04

WORKDIR /app

# Copy REST API
COPY container_rest.py /app/container_rest.py

# Install REST API dependencies
RUN pip3 install --ignore-installed flask soundfile scipy

# Create a simple model directory
RUN mkdir -p /app/models/distil-small.en

# Copy the model files directly to a simple directory
COPY models--Systran--faster-distil-whisper-small.en/snapshots/ef77d90526ccd62cde3808ee70626a01e5cf83e4/* /app/models/distil-small.en/

EXPOSE 5000
CMD ["python3", "/app/container_rest.py"]
# Use the jetson-containers built faster-whisper image
FROM faster-whisper:r36.4.tegra-aarch64-cu126-22.04

WORKDIR /app

# Copy REST API
COPY container_rest.py /app/container_rest.py

# Install REST API dependencies
RUN pip3 install flask soundfile scipy

# Create cache directories
RUN mkdir -p /app/cache/whisper /root/.cache/huggingface

EXPOSE 5000
CMD ["python3", "/app/container_rest.py"]
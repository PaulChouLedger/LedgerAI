# Use dustynv faster-whisper base image directly
FROM dustynv/faster-whisper:r36.4.0-cu128-24.04

# Set build arguments for compatibility
ARG FORCE_BUILD=off
ARG FASTER_WHISPER_VERSION
ARG FASTER_WHISPER_BRANCH

WORKDIR /app

# Copy REST API and official dustynv scripts
COPY container_rest.py /app/container_rest.py
COPY *.sh /tmp/faster-whisper/

# Install faster-whisper using the official dustynv method
RUN /tmp/faster-whisper/install.sh || /tmp/faster-whisper/build.sh

# Install REST API dependencies
RUN pip3 install flask soundfile scipy

# Create cache directories
RUN mkdir -p /app/cache/whisper /root/.cache/huggingface

EXPOSE 5000
CMD ["python3", "/app/container_rest.py"]
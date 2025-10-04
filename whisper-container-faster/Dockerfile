# Use dustynv faster-whisper base image (includes all dependencies)
FROM dustynv/faster-whisper:r36.4.0-cu128-24.04

WORKDIR /app

# Copy REST API
COPY container_rest.py /app/container_rest.py

# Install only Flask for the REST API (soundfile already included)
RUN pip3 install --index-url https://pypi.org/simple flask

# Create cache directory
RUN mkdir -p /app/cache/whisper

EXPOSE 5000
CMD ["python3", "/app/container_rest.py"]

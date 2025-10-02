FROM nvcr.io/nvidia/pytorch:23.10-py3

WORKDIR /app

# Copy REST API
COPY container_rest.py /app/container_rest.py

# Install faster-whisper and dependencies
RUN pip3 install --break-system-packages --ignore-installed --no-cache-dir --index-url https://pypi.org/simple \
    faster-whisper flask soundfile numpy==1.26.4 scipy

EXPOSE 5000
CMD ["python3", "/app/container_rest.py"]

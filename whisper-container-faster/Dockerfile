FROM nvcr.io/nvidia/pytorch:23.10-py3

WORKDIR /app

# Copy REST API
COPY container_rest.py /app/container_rest.py

# Copy build script
COPY build_ctranslate2.sh /app/build_ctranslate2.sh

# Install faster-whisper and dependencies
RUN pip3 install --break-system-packages --ignore-installed --no-cache-dir --index-url https://pypi.org/simple \
    faster-whisper flask soundfile numpy==1.24.4 scipy

# Build CTranslate2 for optimal performance
RUN chmod +x /app/build_ctranslate2.sh && \
    /app/build_ctranslate2.sh

# Force reinstall NumPy 1.24.4 after CTranslate2 build (fixes NumPy 2.x compatibility)
RUN pip3 install --break-system-packages --force-reinstall --no-cache-dir numpy==1.24.4

# Create cache directory
RUN mkdir -p /app/cache/whisper

# Set CUDA environment variables
ENV CUDA_VISIBLE_DEVICES=0
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Set NumPy compatibility
ENV NUMPY_DISABLE_ABI_COMPATIBILITY=1

EXPOSE 5000
CMD ["python3", "/app/container_rest.py"]

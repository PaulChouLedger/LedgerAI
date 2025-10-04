# Try a more recent base image with better cuDNN support
FROM nvcr.io/nvidia/pytorch:24.02-py3

WORKDIR /app

# Copy REST API
COPY container_rest.py /app/container_rest.py

# Copy build script
COPY build_ctranslate2.sh /app/build_ctranslate2.sh

# Install faster-whisper and dependencies
RUN pip3 install --break-system-packages --ignore-installed --no-cache-dir --index-url https://pypi.org/simple \
    faster-whisper flask soundfile numpy==1.24.4 "scipy<1.11"

# Install additional CUDA/cuDNN packages if needed
RUN pip3 install --break-system-packages --no-cache-dir \
    nvidia-cudnn-cu12 \
    nvidia-cublas-cu12 \
    nvidia-cusparse-cu12 \
    nvidia-cufft-cu12 \
    nvidia-curand-cu12 \
    nvidia-cusolver-cu12 \
    nvidia-cusparse-cu12 \
    nvidia-nccl-cu12 \
    nvidia-nvtx-cu12 || echo "Some CUDA packages may not be available"

# Build CTranslate2 for optimal performance
RUN chmod +x /app/build_ctranslate2.sh && \
    /app/build_ctranslate2.sh

# Force reinstall NumPy 1.24.4 and compatible SciPy after CTranslate2 build (fixes NumPy 2.x compatibility)
RUN pip3 install --break-system-packages --force-reinstall --no-cache-dir numpy==1.24.4 "scipy<1.11"

# Create cache directory
RUN mkdir -p /app/cache/whisper

# Set CUDA environment variables
ENV CUDA_VISIBLE_DEVICES=0
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Additional CUDA/cuDNN environment variables for better GPU initialization
ENV CUDA_CACHE_DISABLE=0
ENV CUDA_CACHE_MAXSIZE=2147483648
ENV CUDNN_VERSION=8.9.0

# Set NumPy compatibility
ENV NUMPY_DISABLE_ABI_COMPATIBILITY=1

# Check cuDNN installation and PyTorch CUDA support
RUN python3 -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('cuDNN version:', torch.backends.cudnn.version()); print('PyTorch version:', torch.__version__); print('CUDA version:', torch.version.cuda)"

# Check if cuDNN libraries are available
RUN find /usr -name "*cudnn*" -type f 2>/dev/null | head -10 || echo "No cuDNN libraries found"

# Check CUDA installation
RUN nvcc --version || echo "nvcc not found"

EXPOSE 5000
CMD ["python3", "/app/container_rest.py"]

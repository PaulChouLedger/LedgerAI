# Chatterbox-TTS Container
# Installs Chatterbox from source to avoid dependency conflicts
# Using Jetson-compatible base image with PyTorch pre-installed
FROM dustynv/pytorch:2.6-r36.4.0-cu128-24.04

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip, setuptools, and wheel (fixes Python 3.12 compatibility issues)
RUN pip3 install --upgrade --index-url https://pypi.org/simple pip setuptools wheel

# Verify PyTorch is installed and CUDA-enabled
RUN python3 -c "import torch; assert hasattr(torch.version, 'cuda') and torch.version.cuda is not None, 'CUDA PyTorch required'; print(f'✅ PyTorch {torch.__version__} with CUDA {torch.version.cuda}')"

# Install numpy first with Jetson index (has pre-built ARM64 wheels, avoids Python 3.12 build issues)
RUN pip3 install --index-url https://pypi.jetson-ai-lab.dev/jp6/cu128 --extra-index-url https://pypi.org/simple \
    "numpy>=1.24.0,<1.26.0" || \
    pip3 install --index-url https://pypi.org/simple --only-binary numpy "numpy>=1.24.0,<1.26.0"

# Install remaining Chatterbox dependencies (standard PyPI packages)
RUN pip3 install --index-url https://pypi.org/simple \
    librosa==0.11.0 \
    transformers==4.46.3 \
    diffusers==0.29.0 \
    safetensors==0.5.3 \
    conformer==0.3.2 \
    resemble-perth==1.0.1 \
    s3tokenizer \
    pykakasi==2.3.0 \
    jaconv \
    soundfile \
    scipy \
    typer==0.9.0

# Clone and install Chatterbox from source (skip dependencies to avoid conflicts)
# Note: russian-text-stresser may not be available on PyPI - skipping (optional for Russian support)
# Install spacy first (Chatterbox dependency)
RUN pip3 install --index-url https://pypi.org/simple spacy==3.6.1 || true

# Clone Chatterbox and install (keep source for editable install)
RUN git clone https://github.com/resemble-ai/chatterbox.git /app/chatterbox && \
    cd /app/chatterbox && \
    pip3 install --index-url https://pypi.org/simple -e . --no-deps && \
    python3 -c "from chatterbox import ChatterboxTTS; print('✅ Chatterbox installed successfully')"

# Install Flask for REST API
RUN pip3 install --index-url https://pypi.org/simple flask requests python-dotenv

# Copy application files
COPY container_rest.py /app/container_rest.py

# Create directories for voice samples and cache
RUN mkdir -p /app/voice_samples /app/voice_cache

# Expose port for API
EXPOSE 11437

# Run the API
CMD ["python3", "/app/container_rest.py"]


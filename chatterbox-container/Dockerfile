# Chatterbox-TTS Container
# Installs Chatterbox from source to avoid dependency conflicts
# Using CUDA 12.6 devel image - PyTorch 2.6 supports CUDA 12.6
FROM nvidia/cuda:12.6.0-devel-ubuntu22.04

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3-pip \
    git \
    build-essential \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Create symlink for python
RUN ln -s /usr/bin/python3.10 /usr/bin/python

# Upgrade pip
RUN pip3 install --upgrade pip setuptools wheel

# Install PyTorch 2.6 with CUDA 12.6 support (required by Chatterbox)
# CRITICAL: This will FAIL if CUDA PyTorch cannot be installed (NO CPU FALLBACK)
RUN pip3 install --no-cache-dir torch==2.6.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu126 && \
    python3 -c "import torch; assert hasattr(torch.version, 'cuda') and torch.version.cuda is not None, 'CUDA PyTorch required'"

# Install Chatterbox dependencies before cloning
# Note: Install compatible versions to avoid conflicts
RUN pip3 install \
    "numpy>=1.24.0,<1.26.0" \
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
RUN pip3 install spacy==3.6.1 || true

# Clone Chatterbox and install (keep source for editable install)
RUN git clone https://github.com/resemble-ai/chatterbox.git /app/chatterbox && \
    cd /app/chatterbox && \
    pip3 install -e . --no-deps && \
    python3 -c "from chatterbox import ChatterboxTTS; print('✅ Chatterbox installed successfully')"

# Install Flask for REST API
RUN pip3 install flask requests python-dotenv

# Copy application files
COPY container_rest.py /app/container_rest.py

# Create directories for voice samples and cache
RUN mkdir -p /app/voice_samples /app/voice_cache

# Expose port for API
EXPOSE 11437

# Run the API
CMD ["python3", "/app/container_rest.py"]


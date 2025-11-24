# Chatterbox-TTS Container
# Installs Chatterbox from source to avoid dependency conflicts
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

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

# Install PyTorch with CUDA support first (required by Chatterbox)
# Using CUDA 12.1 compatible version
RUN pip3 install torch==2.6.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu121

# Install Chatterbox dependencies before cloning
RUN pip3 install \
    numpy>=1.24.0,<1.26.0 \
    librosa==0.11.0 \
    transformers==4.46.3 \
    diffusers==0.29.0 \
    safetensors==0.5.3 \
    conformer==0.3.2 \
    resemble-perth==1.0.1 \
    s3tokenizer \
    pykakasi==2.3.0 \
    jaconv \
    gradio==5.44.1 \
    soundfile \
    scipy

# Clone and install Chatterbox from source
RUN git clone https://github.com/resemble-ai/chatterbox.git /tmp/chatterbox && \
    cd /tmp/chatterbox && \
    pip3 install -e . && \
    rm -rf /tmp/chatterbox

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


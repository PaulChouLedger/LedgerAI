# Chatterbox-TTS Container
# Installs Chatterbox from source to avoid dependency conflicts
# Using devel image to ensure CUDA headers/libraries are available for PyTorch
FROM nvidia/cuda:12.1.0-devel-ubuntu22.04

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
# PyTorch 2.6.0 doesn't exist yet - using latest available CUDA 12.1 version
# CRITICAL: This will FAIL if CUDA PyTorch cannot be installed (NO CPU FALLBACK)
# Note: We check for CUDA support in PyTorch (torch.version.cuda), not runtime availability
# Runtime CUDA availability is checked when container runs (requires GPU)
RUN echo "Installing PyTorch with CUDA 12.1 support..." && \
    pip3 install --no-cache-dir torch torchaudio --index-url https://download.pytorch.org/whl/cu121 && \
    echo "Checking installed PyTorch package..." && \
    pip3 show torch | grep -E "Name|Version|Location" && \
    echo "Verifying CUDA support in PyTorch..." && \
    python3 << 'PYEOF'
import torch
import sys

print(f"PyTorch version: {torch.__version__}")
print(f"torch.version attributes: {dir(torch.version)}")

# Check for CUDA support - try multiple ways
has_cuda_attr = hasattr(torch.version, 'cuda')
cuda_version = getattr(torch.version, 'cuda', None)

# Also check if CUDA is in the package name or build info
try:
    import torch._C
    has_cuda_build = hasattr(torch._C, '_cuda_getDeviceCount')
except:
    has_cuda_build = False

print(f"Has torch.version.cuda attribute: {has_cuda_attr}")
print(f"CUDA version value: {cuda_version}")
print(f"Has CUDA in build: {has_cuda_build}")

# Verify it's CUDA-enabled PyTorch
if not has_cuda_attr:
    print("❌ ERROR: torch.version.cuda attribute missing - CPU-only PyTorch installed")
    sys.exit(1)

if cuda_version is None or cuda_version == '':
    print("❌ ERROR: CUDA version is None or empty - CPU-only PyTorch installed")
    sys.exit(1)

# Additional check - verify it's not CPU-only build
if not has_cuda_build and 'cpu' in torch.__version__.lower():
    print("❌ ERROR: CPU-only PyTorch detected in version string")
    sys.exit(1)

print(f"✅ PyTorch {torch.__version__} with CUDA {cuda_version} support installed")
PYEOF
    || (echo "❌ FATAL: Failed to install CUDA-enabled PyTorch. CPU fallback not allowed." && exit 1)

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


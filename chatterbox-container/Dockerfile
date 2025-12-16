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

# CRITICAL: Create constraints file to prevent PyTorch installation/upgrade
# Get current PyTorch version and pin it
RUN python3 -c "import torch; print(torch.__version__)" > /tmp/pytorch_version.txt && \
    TORCH_VER=$(cat /tmp/pytorch_version.txt) && \
    echo "torch==${TORCH_VER}" > /tmp/constraints.txt && \
    echo "torchvision" >> /tmp/constraints.txt && \
    echo "torchaudio" >> /tmp/constraints.txt && \
    echo "Constraints file created:" && cat /tmp/constraints.txt

# Install numpy - use 1.26.0+ which has Python 3.12 wheels on PyPI (avoids slow Jetson index)
# Use constraints to prevent PyTorch changes
RUN pip3 install --index-url https://pypi.org/simple \
    --constraint /tmp/constraints.txt \
    "numpy>=1.26.0,<2.0.0"

# CRITICAL: Base image already has PyTorch CUDA - DO NOT install or upgrade it
# Install packages that might pull in PyTorch with --no-deps to prevent PyTorch installation
RUN pip3 install --index-url https://pypi.org/simple \
    --no-deps \
    transformers==4.46.3 \
    diffusers==0.29.0

# Install non-PyTorch dependencies for transformers and diffusers
# CRITICAL: Install accelerate with --no-deps since it might pull in PyTorch
RUN pip3 install --index-url https://pypi.org/simple \
    --no-deps \
    accelerate || true

# Install other dependencies (these shouldn't pull in PyTorch)
# Use constraints to prevent PyTorch installation
RUN pip3 install --index-url https://pypi.org/simple \
    --constraint /tmp/constraints.txt \
    packaging \
    filelock \
    huggingface-hub \
    regex \
    requests \
    tokenizers \
    tqdm \
    pyyaml \
    pillow \
    psutil \
    safetensors==0.5.3 || true

# Check PyTorch after installing dependencies
RUN python3 -c "import torch; v=torch.__version__; cuda=hasattr(torch.version, 'cuda') and torch.version.cuda is not None; print(f'After deps: {v}, CUDA: {cuda}'); assert cuda, f'PyTorch lost after deps install! Version: {v}'"

# Install other Chatterbox dependencies (these shouldn't pull in PyTorch)
# CRITICAL: Use --no-deps and install dependencies manually to prevent PyTorch installation
RUN pip3 install --index-url https://pypi.org/simple \
    --no-deps \
    librosa==0.11.0 \
    conformer==0.3.2 \
    resemble-perth==1.0.1 \
    s3tokenizer \
    pykakasi==2.3.0 \
    jaconv \
    soundfile \
    scipy \
    typer==0.9.0

# Check PyTorch after installing Chatterbox deps
RUN python3 -c "import torch; v=torch.__version__; cuda=hasattr(torch.version, 'cuda') and torch.version.cuda is not None; print(f'After Chatterbox deps: {v}, CUDA: {cuda}'); assert cuda, f'PyTorch lost after Chatterbox deps! Version: {v}'"

# Install dependencies for librosa and other packages (excluding torch)
# Use constraints to prevent PyTorch installation
RUN pip3 install --index-url https://pypi.org/simple \
    --constraint /tmp/constraints.txt \
    audioread \
    decorator \
    numba \
    llvmlite \
    resampy \
    pooch || true

# Check PyTorch after installing librosa deps
RUN python3 -c "import torch; v=torch.__version__; cuda=hasattr(torch.version, 'cuda') and torch.version.cuda is not None; print(f'After librosa deps: {v}, CUDA: {cuda}'); assert cuda, f'PyTorch lost after librosa deps! Version: {v}'"

# CRITICAL: Re-verify PyTorch CUDA was not overwritten
# If this fails, something is still installing PyTorch - check which package
RUN python3 << 'EOF'
import torch
import sys
import subprocess

print(f"Checking PyTorch after all installations: {torch.__version__}")
has_cuda = hasattr(torch.version, 'cuda') and torch.version.cuda is not None

if not has_cuda:
    print(f"❌ ERROR: PyTorch CUDA was lost!")
    print(f"   Current version: {torch.__version__}")
    print(f"   This indicates a package installed CPU PyTorch.")
    print(f"\n   Checking which packages are installed:")
    result = subprocess.run(['pip3', 'list'], capture_output=True, text=True)
    torch_packages = [line for line in result.stdout.split('\n') if 'torch' in line.lower()]
    if torch_packages:
        print(f"   Found torch-related packages:")
        for line in torch_packages:
            print(f"     {line}")
    else:
        print(f"   No torch packages found in pip list")
    sys.exit(1)
else:
    print(f"✅ PyTorch {torch.__version__} with CUDA {torch.version.cuda} preserved")
EOF

# Clone and install Chatterbox from source (skip dependencies to avoid conflicts)
# Note: russian-text-stresser may not be available on PyPI - skipping (optional for Russian support)
# Install spacy first (Chatterbox dependency)
# Use constraints to prevent PyTorch installation
RUN pip3 install --index-url https://pypi.org/simple \
    --constraint /tmp/constraints.txt \
    spacy==3.6.1 || true

# Clone Chatterbox and install (keep source for editable install)
RUN git clone https://github.com/resemble-ai/chatterbox.git /app/chatterbox && \
    cd /app/chatterbox && \
    pip3 install --index-url https://pypi.org/simple -e . --no-deps && \
    python3 -c "from chatterbox import ChatterboxTTS; print('✅ Chatterbox installed successfully')"

# Install Flask for REST API
# Use constraints to prevent PyTorch installation (though these shouldn't need it)
RUN pip3 install --index-url https://pypi.org/simple \
    --constraint /tmp/constraints.txt \
    flask requests python-dotenv

# Copy application files
COPY container_rest.py /app/container_rest.py

# Create directories for voice samples and cache
RUN mkdir -p /app/voice_samples /app/voice_cache

# Expose port for API
EXPOSE 11437

# Run the API
CMD ["python3", "/app/container_rest.py"]


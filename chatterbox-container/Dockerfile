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
    libsoxr0 \
    libsoxr-dev \
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

# Install torchaudio and torchvision if not already present (required by s3tokenizer)
# Use constraints on torch only to prevent PyTorch upgrade, but allow torchaudio/torchvision installation
RUN python3 << 'EOF'
import torch
import subprocess
import sys

# Check if torchaudio is installed
try:
    import torchaudio
    print(f"✅ torchaudio {torchaudio.__version__} already installed")
except ImportError:
    print("⚠️  torchaudio not found, installing...")
    # Create constraints file with only torch (to prevent PyTorch upgrade)
    torch_ver = torch.__version__
    with open('/tmp/torch_only_constraints.txt', 'w') as f:
        f.write(f"torch=={torch_ver}\n")
    
    # Install torchaudio matching PyTorch version
    result = subprocess.run([
        'pip3', 'install', '--index-url', 'https://pypi.org/simple',
        '--constraint', '/tmp/torch_only_constraints.txt',
        'torchaudio'
    ], capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"❌ Failed to install torchaudio: {result.stderr}")
        sys.exit(1)
    
    # Verify installation
    import torchaudio
    print(f"✅ torchaudio {torchaudio.__version__} installed successfully")

# Check if torchvision is installed
try:
    import torchvision
    print(f"✅ torchvision {torchvision.__version__} already installed")
except ImportError:
    print("⚠️  torchvision not found, installing...")
    # Create constraints file with only torch (to prevent PyTorch upgrade)
    torch_ver = torch.__version__
    with open('/tmp/torch_only_constraints.txt', 'w') as f:
        f.write(f"torch=={torch_ver}\n")
    
    # Install torchvision matching PyTorch version
    result = subprocess.run([
        'pip3', 'install', '--index-url', 'https://pypi.org/simple',
        '--constraint', '/tmp/torch_only_constraints.txt',
        'torchvision'
    ], capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"❌ Failed to install torchvision: {result.stderr}")
        sys.exit(1)
    
    # Verify installation
    import torchvision
    print(f"✅ torchvision {torchvision.__version__} installed successfully")

# Verify PyTorch CUDA is still intact
assert hasattr(torch.version, 'cuda') and torch.version.cuda is not None, \
    f'PyTorch CUDA lost after torchaudio/torchvision install! Version: {torch.__version__}'
print(f"✅ PyTorch {torch.__version__} with CUDA {torch.version.cuda} preserved")
EOF

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
# CRITICAL: Install tokenizers version compatible with transformers==4.46.3 (requires >=0.20,<0.21)
RUN pip3 install --index-url https://pypi.org/simple \
    --constraint /tmp/constraints.txt \
    packaging \
    filelock \
    huggingface-hub \
    regex \
    requests \
    "tokenizers>=0.20,<0.21" \
    tqdm \
    pyyaml \
    pillow \
    psutil \
    safetensors==0.5.3 \
    einops || true

# Check PyTorch after installing dependencies
RUN python3 -c "import torch; v=torch.__version__; cuda=hasattr(torch.version, 'cuda') and torch.version.cuda is not None; print(f'After deps: {v}, CUDA: {cuda}'); assert cuda, f'PyTorch lost after deps install! Version: {v}'"

# Install resemble-perth WITH dependencies (needed for PerthImplicitWatermarker)
# soxr is required for PerthImplicitWatermarker (libsoxr-dev installed above)
# Use constraints to prevent PyTorch installation
RUN pip3 install --index-url https://pypi.org/simple \
    --constraint /tmp/constraints.txt \
    resemble-perth==1.0.1 \
    soxr

# Install other Chatterbox dependencies (these shouldn't pull in PyTorch)
# CRITICAL: Use --no-deps and install dependencies manually to prevent PyTorch installation
RUN pip3 install --index-url https://pypi.org/simple \
    --no-deps \
    librosa==0.11.0 \
    conformer==0.3.2 \
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
    pooch \
    lazy_loader \
    joblib \
    msgpack || true

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
# Install omegaconf (required by chatterbox, doesn't conflict with PyTorch)
RUN pip3 install --index-url https://pypi.org/simple \
    --constraint /tmp/constraints.txt \
    "omegaconf>=2.3.0"

# Install spacy (Chatterbox dependency, optional for some features)
# Use constraints to prevent PyTorch installation
RUN pip3 install --index-url https://pypi.org/simple \
    --constraint /tmp/constraints.txt \
    spacy==3.6.1 || true

# Clone Chatterbox and install (keep source for editable install)
RUN git clone https://github.com/resemble-ai/chatterbox.git /app/chatterbox && \
    cd /app/chatterbox && \
    pip3 install --index-url https://pypi.org/simple \
    --constraint /tmp/constraints.txt \
    -e . --no-deps && \
    python3 -c "from chatterbox import ChatterboxTTS; print('✅ Chatterbox installed successfully')"

# Pre-download ChatterboxTTS models (cached layer - models are large ~2-3GB)
# This ensures models are available in the image and avoids first-request download delays
# Models will be cached in ~/.cache/huggingface/ which is baked into the image
RUN python3 << 'EOF'
import os
import sys
from pathlib import Path
import time

# Enable HuggingFace progress bars and verbose logging
os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '0'  # Show progress bars
os.environ['TRANSFORMERS_VERBOSITY'] = 'info'  # Verbose logging
os.environ['HF_HUB_DOWNLOAD_TIMEOUT'] = '1800'  # 30 minutes timeout (models are large)

# Set HuggingFace cache directory to a known location
cache_dir = Path.home() / '.cache' / 'huggingface'
os.environ['HF_HOME'] = str(cache_dir.parent)
print(f"[Chatterbox] 📦 Using HuggingFace cache: {cache_dir}")

try:
    print("[Chatterbox] 📥 Pre-downloading ChatterboxTTS models from HuggingFace...")
    print("[Chatterbox] ⏳ This may take several minutes (models are ~2-3GB)...")
    print("[Chatterbox] 💡 You should see download progress below")
    print("[Chatterbox] 💡 Build will continue even if download fails (models can download at runtime)")
    print("[Chatterbox] " + "="*70)
    
    start_time = time.time()
    
    from chatterbox.tts import ChatterboxTTS
    import torch
    
    # Detect device (use CPU for build-time download)
    device = "cpu"  # Use CPU during build, will use CUDA at runtime
    
    print(f"[Chatterbox] 🔄 Calling ChatterboxTTS.from_pretrained(device={device})...")
    print("[Chatterbox] 💡 HuggingFace will show download progress (downloading/loading files)...")
    
    # Track cache size before
    cache_size_before = 0
    if cache_dir.exists():
        cache_size_before = sum(f.stat().st_size for f in cache_dir.rglob('*') if f.is_file())
        print(f"[Chatterbox] 📊 Cache size before: {cache_size_before / (1024**3):.2f} GB")
    
    model = ChatterboxTTS.from_pretrained(device=device)
    
    elapsed = time.time() - start_time
    print("[Chatterbox] " + "="*70)
    print(f"[Chatterbox] ✅ Models pre-downloaded successfully (took {elapsed:.1f} seconds)")
    
    # Verify models are cached
    if cache_dir.exists():
        cache_size_after = sum(f.stat().st_size for f in cache_dir.rglob('*') if f.is_file())
        downloaded = cache_size_after - cache_size_before
        print(f"[Chatterbox] 📊 Cache size after: {cache_size_after / (1024**3):.2f} GB")
        if downloaded > 0:
            print(f"[Chatterbox] 📥 Downloaded: {downloaded / (1024**3):.2f} GB")
        print(f"[Chatterbox] 💡 Models cached in {cache_dir} and baked into image")
    else:
        print(f"[Chatterbox] ⚠️  Cache directory {cache_dir} does not exist after download")
        
except Exception as e:
    elapsed = time.time() - start_time if 'start_time' in locals() else 0
    print("[Chatterbox] " + "="*70)
    print(f"[Chatterbox] ⚠️  Model pre-download failed after {elapsed:.1f} seconds: {e}")
    import traceback
    traceback.print_exc()
    print("[Chatterbox] 💡 Models will be downloaded on first request instead")
    print("[Chatterbox] 💡 Ensure container has internet access on first use")
    # Don't fail the build if download fails - allow runtime download
    # But make it clear in the logs that models weren't pre-downloaded
    sys.exit(0)  # Exit with success to not fail the build
EOF

# Verify models were downloaded (informational - don't fail if missing)
RUN python3 << 'EOF'
from pathlib import Path
import os

cache_dir = Path.home() / '.cache' / 'huggingface'
if cache_dir.exists():
    cache_size = sum(f.stat().st_size for f in cache_dir.rglob('*') if f.is_file())
    print(f"[Chatterbox] ✅ Model cache verified: {cache_size / (1024**3):.2f} GB in {cache_dir}")
    print("[Chatterbox] 💡 Models are baked into image - no download needed at runtime")
else:
    print("[Chatterbox] ⚠️  Model cache not found - models will download on first request")
    print("[Chatterbox] 💡 This is OK if pre-download failed, but will cause delays on first use")
EOF

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


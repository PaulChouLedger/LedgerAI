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

# Install scikit-learn (required by librosa for voice cloning/trim effects)
# Use constraints to prevent PyTorch installation
RUN pip3 install --index-url https://pypi.org/simple \
    --constraint /tmp/constraints.txt \
    scikit-learn || true

# Clone Chatterbox and install (keep source for editable install)
RUN git clone https://github.com/resemble-ai/chatterbox.git /app/chatterbox && \
    cd /app/chatterbox && \
    pip3 install --index-url https://pypi.org/simple \
    --constraint /tmp/constraints.txt \
    -e . --no-deps && \
    python3 -c "from chatterbox import ChatterboxTTS; print('✅ Chatterbox installed successfully')"

# Pre-download ChatterboxTTS models using huggingface_hub with hf_transfer
# Model repository: ResembleAI/chatterbox (~2-3GB total, ~15 files)
# WARNING: Large .safetensors/.pt files can take 15-30+ minutes EACH on slow connections
# Progress may appear "stuck" at a file count (e.g., "4/15") - this is normal, it's downloading large files
# To skip: docker build --build-arg SKIP_MODEL_DOWNLOAD=1 (models will download at runtime instead)
ARG SKIP_MODEL_DOWNLOAD=0

# Install huggingface_hub first
RUN pip3 install --index-url https://pypi.org/simple huggingface_hub[hf_transfer]

# Download models (conditionally based on SKIP_MODEL_DOWNLOAD)
# Create download script first
RUN cat > /tmp/download_models.py << 'EOF'
import os
import sys
from pathlib import Path
import time
from huggingface_hub import snapshot_download

# Set HuggingFace cache directory and enable hf_transfer for faster downloads
cache_dir = Path.home() / '.cache' / 'huggingface'
os.environ['HF_HOME'] = str(cache_dir.parent)
os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '0'  # Show progress bars
os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '1'  # Use hf_transfer if available

# Set longer timeouts for large files
os.environ['HF_HUB_DOWNLOAD_TIMEOUT'] = '3600'  # 1 hour per file

print(f"[Chatterbox] 📦 Using HuggingFace cache: {cache_dir}")
print("[Chatterbox] 📥 Pre-downloading ChatterboxTTS models from HuggingFace...")
print("[Chatterbox] 💡 Model: ResembleAI/chatterbox")
print("[Chatterbox] ⏳ Large files - may take 20-40 minutes on slow connections...")
print("[Chatterbox] 💡 Progress bars will show per-file downloads")
print("[Chatterbox] " + "="*70)

start_time = time.time()

# Track cache size before
cache_size_before = 0
if cache_dir.exists():
    cache_size_before = sum(f.stat().st_size for f in cache_dir.rglob('*') if f.is_file())
    print(f"[Chatterbox] 📊 Cache size before: {cache_size_before / (1024**3):.2f} GB")

try:
    # Download with verbose logging - this will show progress for each file
    print("[Chatterbox] 🔄 Starting download (you'll see progress bars for each file)...")
    print("[Chatterbox] 💡 Large .safetensors/.pt files may take 5-15 minutes each")
    
    snapshot_download(
        repo_id="ResembleAI/chatterbox",
        cache_dir=str(cache_dir),
        local_files_only=False,
        max_workers=1  # Single worker to avoid overwhelming connection
    )
    
    elapsed = time.time() - start_time
    print("[Chatterbox] " + "="*70)
    print(f"[Chatterbox] ✅ Models downloaded successfully (took {elapsed/60:.1f} minutes)")
    
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
        
except KeyboardInterrupt:
    elapsed = time.time() - start_time if 'start_time' in locals() else 0
    print("[Chatterbox] " + "="*70)
    print(f"[Chatterbox] ⚠️  Download interrupted by user after {elapsed/60:.1f} minutes")
    print("[Chatterbox] 💡 Partial download may be cached - will resume on next build")
    sys.exit(0)
except Exception as e:
    elapsed = time.time() - start_time if 'start_time' in locals() else 0
    print("[Chatterbox] " + "="*70)
    print(f"[Chatterbox] ⚠️  Model pre-download failed after {elapsed/60:.1f} minutes: {e}")
    import traceback
    traceback.print_exc()
    print("[Chatterbox] 💡 Models will be downloaded on first request instead")
    print("[Chatterbox] 💡 Partial downloads are cached and will resume")
    print("[Chatterbox] 💡 Ensure container has internet access on first use")
    sys.exit(0)  # Exit with success to not fail the build
EOF

# Run the download script conditionally
RUN if [ "$SKIP_MODEL_DOWNLOAD" = "1" ]; then \
        echo "[Chatterbox] ⏭️  Skipping model pre-download (will download at runtime)"; \
    else \
        python3 /tmp/download_models.py || true; \
    fi

# Verify ChatterboxTTS can load the downloaded models (only if we downloaded them)
RUN cat > /tmp/verify_models.py << 'EOF'
import os
from pathlib import Path

cache_dir = Path.home() / '.cache' / 'huggingface'
os.environ['HF_HOME'] = str(cache_dir.parent)

try:
    print("[Chatterbox] 🔍 Verifying models can be loaded...")
    from chatterbox.tts import ChatterboxTTS
    
    # Try to load (this should use cached models now)
    print("[Chatterbox] 🔄 Testing model loading (should be fast - models are cached)...")
    model = ChatterboxTTS.from_pretrained(device="cpu")  # Use CPU for verification
    print("[Chatterbox] ✅ Models verified - ChatterboxTTS loaded successfully")
except Exception as e:
    print(f"[Chatterbox] ⚠️  Model verification failed: {e}")
    print("[Chatterbox] 💡 Models may still download at runtime")
    # Don't fail build - models might work at runtime
EOF

RUN if [ "$SKIP_MODEL_DOWNLOAD" != "1" ]; then \
        python3 /tmp/verify_models.py || true; \
    else \
        echo "[Chatterbox] ⏭️  Skipping model verification (models not pre-downloaded)"; \
    fi


# Install Flask for REST API
# Use constraints to prevent PyTorch installation (though these shouldn't need it)
RUN pip3 install --index-url https://pypi.org/simple \
    --constraint /tmp/constraints.txt \
    flask requests python-dotenv

# Copy application files
COPY container_rest.py /app/container_rest.py
COPY test_voice_cloning_api.py /app/test_voice_cloning_api.py

# Create directories for voice samples and cache
RUN mkdir -p /app/voice_samples /app/voice_cache

# Expose port for API
EXPOSE 11437

# Run the API
CMD ["python3", "/app/container_rest.py"]


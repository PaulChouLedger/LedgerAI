#!/usr/bin/env python3
"""
Pre-download ChatterboxTTS models to avoid CAS service errors at runtime.
This script downloads the models from HuggingFace and caches them locally.
"""

import os
import sys
from pathlib import Path
import time

# Set up environment for HuggingFace
cache_dir = Path.home() / '.cache' / 'huggingface'
os.makedirs(cache_dir, exist_ok=True)
os.environ['HF_HOME'] = str(cache_dir.parent)
os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '0'
os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '1'
os.environ['HF_HUB_DOWNLOAD_TIMEOUT'] = '3600'  # 1 hour timeout

print("="*70)
print("ChatterboxTTS Model Pre-Download Script")
print("="*70)
print(f"📦 Cache directory: {cache_dir}")
print(f"📥 Model repository: ResembleAI/chatterbox")
print(f"⏳ This may take 20-40 minutes depending on network speed...")
print("="*70)
print()

try:
    from huggingface_hub import snapshot_download
except ImportError:
    print("❌ huggingface_hub not installed")
    print("💡 Install it with: pip install huggingface_hub")
    sys.exit(1)

# Check if models are already cached
model_cache = cache_dir / 'hub' / 'models--ResembleAI--chatterbox'
if model_cache.exists():
    cache_size = sum(f.stat().st_size for f in model_cache.rglob('*') if f.is_file())
    print(f"✅ Models already cached: {cache_size / (1024**3):.2f} GB")
    print(f"📁 Location: {model_cache}")
    print()
    response = input("Re-download models? (y/N): ")
    if response.lower() != 'y':
        print("✅ Using existing cached models")
        sys.exit(0)

print("🔄 Starting download...")
print("💡 Large .safetensors/.pt files may take 5-15 minutes each")
print()

start_time = time.time()

try:
    snapshot_download(
        repo_id="ResembleAI/chatterbox",
        cache_dir=str(cache_dir),
        local_files_only=False,
        max_workers=1  # Single worker to avoid overwhelming connection
    )
    
    elapsed = time.time() - start_time
    print()
    print("="*70)
    print(f"✅ Models downloaded successfully!")
    print(f"⏱️  Time taken: {elapsed/60:.1f} minutes")
    
    # Verify download
    if model_cache.exists():
        cache_size = sum(f.stat().st_size for f in model_cache.rglob('*') if f.is_file())
        print(f"📊 Cache size: {cache_size / (1024**3):.2f} GB")
        print(f"📁 Location: {model_cache}")
        print()
        print("✅ Models are now cached and ready to use!")
        print("💡 ChatterboxTTS will use these cached models on next startup")
    else:
        print("⚠️  Warning: Model cache directory not found after download")
        
except KeyboardInterrupt:
    elapsed = time.time() - start_time
    print()
    print("="*70)
    print(f"⚠️  Download interrupted after {elapsed/60:.1f} minutes")
    print("💡 Partial download may be cached - will resume on next run")
    sys.exit(1)
    
except Exception as e:
    elapsed = time.time() - start_time
    print()
    print("="*70)
    print(f"❌ Download failed after {elapsed/60:.1f} minutes")
    print(f"Error: {e}")
    print()
    print("💡 Troubleshooting:")
    print("   1. Check internet connectivity: ping 8.8.8.8")
    print("   2. Test HuggingFace access: curl https://huggingface.co")
    print("   3. Check firewall/proxy settings")
    print("   4. Ensure sufficient disk space (~3GB needed)")
    print("   5. Try again later (service may be temporarily down)")
    sys.exit(1)

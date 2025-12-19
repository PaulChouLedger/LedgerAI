#!/usr/bin/env python3
"""
Pre-download ChatterboxTTS models to avoid CAS service errors at runtime.
This script downloads the models from HuggingFace and caches them locally.
"""

import os
import sys
from pathlib import Path
import time
import signal

# Set up environment for HuggingFace
cache_dir = Path.home() / '.cache' / 'huggingface'
os.makedirs(cache_dir, exist_ok=True)
os.environ['HF_HOME'] = str(cache_dir.parent)
os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '0'
os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '1'  # Use hf_transfer for faster downloads
os.environ['HF_HUB_DOWNLOAD_TIMEOUT'] = '7200'  # 2 hour timeout (increased for large files)
os.environ['HF_HUB_CACHE'] = str(cache_dir)  # Explicit cache location

# Check if hf_transfer is available
try:
    import hf_transfer
    print("✅ hf_transfer is available - will use for faster downloads")
except ImportError:
    print("⚠️  hf_transfer not available - install with: pip install hf_transfer")
    print("   Downloads will be slower but should still work")

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
    print("💡 Install it with:")
    print("   pip install huggingface_hub[hf_transfer]")
    print("   OR")
    print("   pip install -r requirements_host.txt")
    print()
    print("💡 Note: This is only needed if running on host machine.")
    print("   The Dockerfile already installs this inside the container.")
    sys.exit(1)

# Check if models are already cached
model_cache = cache_dir / 'hub' / 'models--ResembleAI--chatterbox'
if model_cache.exists():
    cache_size = sum(f.stat().st_size for f in model_cache.rglob('*') if f.is_file())
    print(f"✅ Models found in cache: {cache_size / (1024**3):.2f} GB")
    print(f"📁 Location: {model_cache}")
    
    # Check for incomplete downloads
    incomplete_files = []
    for file_path in model_cache.rglob('*.tmp') or model_cache.rglob('*.incomplete'):
        incomplete_files.append(file_path)
    
    if incomplete_files:
        print(f"⚠️  Found {len(incomplete_files)} incomplete download(s) - will resume")
        print("💡 Partial downloads will be automatically resumed")
    else:
        print("✅ Models appear to be complete")
    
    print()
    response = input("Re-download models? (y/N): ")
    if response.lower() != 'y':
        print("✅ Using existing cached models")
        sys.exit(0)
    else:
        print("🔄 Will re-download (existing cache will be used as fallback)")

print("🔄 Starting download...")
print("💡 Large .safetensors/.pt files may take 5-15 minutes each")
print("💡 Downloads will automatically resume if interrupted")
print()

start_time = time.time()
last_progress_time = time.time()
progress_timeout = 300  # 5 minutes without progress = timeout

def check_progress():
    """Check if download is making progress"""
    global last_progress_time
    current_time = time.time()
    if current_time - last_progress_time > progress_timeout:
        print()
        print("⚠️  WARNING: No progress for 5 minutes - download may be stuck")
        print("💡 This could indicate:")
        print("   - Network connectivity issues")
        print("   - HuggingFace CAS service problems")
        print("   - Large file taking longer than expected")
        print("💡 The download will continue, but you may want to:")
        print("   - Check your internet connection")
        print("   - Try again later if service is down")
        print("   - Press Ctrl+C to cancel and retry")
        last_progress_time = current_time

# Retry logic with exponential backoff
max_retries = 3
retry_delay = 10  # Start with 10 seconds
download_success = False

for attempt in range(1, max_retries + 1):
    try:
        print(f"🔄 Download attempt {attempt}/{max_retries}...")
        if attempt > 1:
            print(f"⏳ Waiting {retry_delay} seconds before retry...")
            time.sleep(retry_delay)
            retry_delay *= 2  # Exponential backoff
        
        snapshot_download(
            repo_id="ResembleAI/chatterbox",
            cache_dir=str(cache_dir),
            local_files_only=False,
            max_workers=1,  # Single worker to avoid overwhelming connection
            resume_download=True,  # Resume partial downloads
            tqdm_class=None  # Use default progress bar
        )
        
        download_success = True
        break  # Success, exit retry loop
        
    except KeyboardInterrupt:
        elapsed = time.time() - start_time
        print()
        print("="*70)
        print(f"⚠️  Download interrupted by user after {elapsed/60:.1f} minutes")
        print("💡 Partial download is cached - will resume on next run")
        print("💡 Run the script again to continue downloading")
        sys.exit(1)
        
    except Exception as e:
        error_str = str(e)
        elapsed = time.time() - start_time
        
        # Check if it's a CAS/service error
        is_cas_error = any(term in error_str.lower() for term in [
            'cas service', 'reqwest', 'connection', 'timeout', 
            'network', 'failed after', 'retries'
        ])
        
        if is_cas_error and attempt < max_retries:
            print()
            print(f"⚠️  CAS/Network error (attempt {attempt}/{max_retries}):")
            print(f"   {error_str[:200]}")
            print(f"💡 Retrying in {retry_delay} seconds...")
            print(f"💡 Elapsed time: {elapsed/60:.1f} minutes")
            continue  # Retry
        else:
            # Last attempt or non-network error - raise
            print()
            print("="*70)
            print(f"❌ Download failed after {elapsed/60:.1f} minutes")
            print(f"Error: {error_str}")
            if attempt >= max_retries:
                print()
                print("💡 All retry attempts exhausted")
            raise

if not download_success:
    print()
    print("="*70)
    print("❌ Download failed after all retry attempts")
    sys.exit(1)

try:
    
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
    print("   6. Partial downloads are cached - run script again to resume")
    print()
    print("💡 Alternative: Let Docker build handle download (may be more reliable)")
    print("   docker build -t chatterbox-tts .")
    sys.exit(1)

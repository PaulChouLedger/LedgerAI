#!/usr/bin/env python3
"""
Simple script to download Whisper models for Docker build.
Downloads both:
- distil-large-v3.5-ct2 (best accuracy/speed balance - 1.5x faster than turbo, better short-form accuracy ⭐ RECOMMENDED)
- faster-distil-whisper-small.en (fast, lower accuracy)

Downloads directly to whisper-container/models--*/

Usage:
    python3 download_model_simple.py

Then SCP the model directories to the remote host.
"""

import os
import sys
import shutil
from pathlib import Path

# Get script directory
SCRIPT_DIR = Path(__file__).parent.absolute()

# Models to download
MODELS = [
    {
        "name": "distil-whisper/distil-large-v3.5-ct2",
        "target_dir": SCRIPT_DIR / "models--distil-whisper--distil-large-v3.5-ct2",
        "cache_name": "models--distil-whisper--distil-large-v3.5-ct2",
        "description": "best accuracy/speed balance - 1.5x faster than turbo, better short-form accuracy ⭐ RECOMMENDED"
    },
    {
        "name": "Systran/faster-distil-whisper-small.en",
        "target_dir": SCRIPT_DIR / "models--Systran--faster-distil-whisper-small.en",
        "cache_name": "models--Systran--faster-distil-whisper-small.en",
        "description": "fast, lower accuracy"
    }
]

print("=" * 70)
print(f"📥 Downloading Whisper Models for Docker Build")
print("=" * 70)
print()

# Check if huggingface_hub is available
try:
    from huggingface_hub import snapshot_download
    print("✅ huggingface_hub found")
except ImportError:
    print("⚠️  huggingface_hub not installed. Installing...")
    os.system("pip3 install huggingface_hub")
    from huggingface_hub import snapshot_download

# Download each model
downloaded_models = []
for i, model_config in enumerate(MODELS, 1):
    model_name = model_config["name"]
    target_dir = model_config["target_dir"]
    description = model_config["description"]
    
    print()
    print("=" * 70)
    print(f"[{i}/{len(MODELS)}] 📥 Downloading: {model_name}")
    print(f"   Description: {description}")
    print(f"📂 Target: {target_dir}")
    print("=" * 70)
    print()
    
    print(f"🔄 Downloading model (this may take several minutes)...")
    print()
    
    # Download model using huggingface_hub
    try:
        download_path = snapshot_download(
            repo_id=model_name,
            local_dir=target_dir,
            local_dir_use_symlinks=False
        )
        print()
        print(f"✅ Model downloaded to: {download_path}")
        downloaded_models.append({
            "name": model_name,
            "target_dir": target_dir
        })
    except Exception as e:
        print(f"❌ Error downloading model: {e}")
        sys.exit(1)

# Summary
print()
print("=" * 70)
print("✅ ALL MODELS DOWNLOADED SUCCESSFULLY!")
print("=" * 70)
print()
print("📁 Downloaded models:")
for model_info in downloaded_models:
    print(f"   • {model_info['name']}")
    print(f"     Location: {model_info['target_dir']}")
print()
print("📤 To transfer via SCP (entire directory):")
print(f"   cd {SCRIPT_DIR}")
print(f"   scp -r models--*/ user@remote:/path/to/whisper-container/")
print()
print("Or compress first (recommended):")
print(f"   cd {SCRIPT_DIR}")
for model_info in downloaded_models:
    model_dir_name = model_info['cache_name']
    print(f"   tar -czf {model_dir_name}.tar.gz {model_dir_name}/")
print(f"   scp models--*.tar.gz user@remote:/path/to/whisper-container/")
print("   # On remote host:")
print(f"   cd whisper-container")
print(f"   tar -xzf models--*.tar.gz")
print()


#!/usr/bin/env python3
"""
Simple script to download Whisper models for Docker build.
Downloads both:
- faster-whisper-large-v3-turbo (best accuracy, higher latency)
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
        "name": "large-v3-turbo",
        "target_dir": SCRIPT_DIR / "models--mobiuslabsgmbh--faster-whisper-large-v3-turbo",
        "cache_name": "models--mobiuslabsgmbh--faster-whisper-large-v3-turbo",
        "description": "best accuracy, higher latency"
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

# Check if faster-whisper is available
try:
    from faster_whisper import WhisperModel
    print("✅ faster-whisper found")
except ImportError:
    print("⚠️  faster-whisper not installed. Installing...")
    os.system("pip3 install faster-whisper")
    from faster_whisper import WhisperModel

# Use default cache directory
cache_dir = os.path.expanduser("~/.cache/huggingface/hub")

# Download each model
downloaded_models = []
for i, model_config in enumerate(MODELS, 1):
    model_name = model_config["name"]
    target_dir = model_config["target_dir"]
    cache_name = model_config["cache_name"]
    description = model_config["description"]
    
    print()
    print("=" * 70)
    print(f"[{i}/{len(MODELS)}] 📥 Downloading: {model_name}")
    print(f"   Description: {description}")
    print(f"📂 Target: {target_dir}")
    print("=" * 70)
    print()
    
    expected_cache = os.path.join(cache_dir, cache_name)
    
    print(f"🔄 Downloading model (this may take several minutes)...")
    print(f"   Model will be cached at: {expected_cache}")
    print()
    
    # Download model
    model = WhisperModel(
        model_name,
        device="cpu",  # CPU is fine for download
        compute_type="int8",
        download_root=cache_dir
    )
    
    print()
    print("✅ Model downloaded successfully!")
    print()
    
    # Copy to target directory
    if os.path.exists(expected_cache):
        print(f"📋 Copying model from cache to: {target_dir}")
        
        # Remove existing directory if it exists
        if target_dir.exists():
            print(f"   Removing existing directory...")
            shutil.rmtree(target_dir)
        
        # Copy entire directory structure
        shutil.copytree(expected_cache, target_dir)
        
        print(f"✅ Model copied successfully!")
        downloaded_models.append({
            "name": cache_name,
            "target_dir": target_dir
        })
    else:
        print(f"❌ Error: Model not found in cache at {expected_cache}")
        print(f"   Check cache directory: {cache_dir}")
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


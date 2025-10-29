#!/usr/bin/env python3
"""
TensorRT-LLM Model Conversion Script
Converts HuggingFace models to TensorRT engines
"""

import os
import subprocess
import sys
from pathlib import Path

def download_huggingface_model(model_name: str, output_dir: str):
    """Download model from HuggingFace"""
    print(f"[Conversion] 📥 Downloading {model_name}...")
    
    try:
        from huggingface_hub import snapshot_download
        
        model_path = snapshot_download(
            repo_id=model_name,
            local_dir=output_dir,
            local_dir_use_symlinks=False
        )
        print(f"[Conversion] ✅ Downloaded to: {model_path}")
        return model_path
    except ImportError:
        print("[Conversion] ❌ huggingface_hub not installed. Installing...")
        subprocess.run([sys.executable, "-m", "pip", "install", "huggingface_hub"])
        return download_huggingface_model(model_name, output_dir)

def convert_to_tensorrt(model_path: str, output_path: str, model_type: str):
    """Convert HuggingFace model to TensorRT engine"""
    print(f"[Conversion] 🔄 Converting {model_type} to TensorRT...")
    
    # TensorRT-LLM conversion command
    # This is a simplified example - actual commands depend on model architecture
    cmd = [
        "python", "-m", "tensorrt_llm.entrypoints.trtllm_build",
        "--model_dir", model_path,
        "--output_dir", output_path,
        "--dtype", "float16",
        "--max_batch_size", "1",
        "--max_input_len", "2048",
        "--max_output_len", "1024"
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"[Conversion] ✅ TensorRT engine created: {output_path}")
            return True
        else:
            print(f"[Conversion] ❌ Conversion failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"[Conversion] ❌ Conversion error: {e}")
        return False

def main():
    """Main conversion process"""
    print("[Conversion] 🚀 TensorRT-LLM Model Conversion")
    print("=" * 50)
    
    # Create directories
    models_dir = Path("models")
    tensorrt_dir = Path("tensorrt_engines")
    models_dir.mkdir(exist_ok=True)
    tensorrt_dir.mkdir(exist_ok=True)
    
    # Models to convert
    models = [
        {
            "name": "meta-llama/Llama-3.2-1B-Instruct",
            "type": "llama",
            "output": "llama-3.2-1b-instruct.engine"
        },
        {
            "name": "mistralai/Mistral-7B-Instruct-v0.3", 
            "type": "mistral",
            "output": "mistral-7b-instruct-v0.3.engine"
        }
    ]
    
    for model in models:
        print(f"\n[Conversion] 🔄 Processing {model['name']}")
        
        # Download model
        model_path = download_huggingface_model(model['name'], f"models/{model['type']}")
        
        # Convert to TensorRT
        output_path = tensorrt_dir / model['output']
        success = convert_to_tensorrt(model_path, str(output_path), model['type'])
        
        if success:
            print(f"[Conversion] ✅ {model['name']} conversion completed")
        else:
            print(f"[Conversion] ❌ {model['name']} conversion failed")
    
    print(f"\n[Conversion] 🎉 Conversion process completed!")
    print(f"[Conversion] 📁 TensorRT engines saved in: {tensorrt_dir}")

if __name__ == "__main__":
    main()

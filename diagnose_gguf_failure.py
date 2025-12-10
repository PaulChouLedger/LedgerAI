#!/usr/bin/env python3
"""
Diagnostic script to identify why GGUF conversion fails
Run this in Colab after training to compare with working medical bot setup
"""

import os
import json
import subprocess
import sys

print("=" * 80)
print("GGUF Conversion Failure Diagnostic")
print("=" * 80)
print()

# Check both model directories
medical_dir = "gguf_model"
rag_dir = "gguf_model_rag_analysis"

print("1. Checking merged model directories:")
print("-" * 80)

for dir_name, dir_path in [("Medical", medical_dir), ("RAG", rag_dir)]:
    print(f"\n{dir_name} model directory: {dir_path}")
    if os.path.exists(dir_path):
        files = os.listdir(dir_path)
        print(f"  ✅ Exists with {len(files)} files")
        
        # Check for key files
        has_config = any("config.json" in f for f in files)
        has_model = any("model.safetensors" in f or "pytorch_model" in f for f in files)
        has_tokenizer = any("tokenizer" in f for f in files)
        
        print(f"  - config.json: {'✅' if has_config else '❌'}")
        print(f"  - model file: {'✅' if has_model else '❌'}")
        print(f"  - tokenizer: {'✅' if has_tokenizer else '❌'}")
        
        # Show file sizes
        print(f"  File sizes:")
        for f in sorted(files)[:10]:
            file_path = os.path.join(dir_path, f)
            if os.path.isfile(file_path):
                size_mb = os.path.getsize(file_path) / 1024 / 1024
                print(f"    - {f}: {size_mb:.2f} MB")
    else:
        print(f"  ❌ Does not exist")

print("\n" + "=" * 80)
print("2. Comparing model configurations:")
print("-" * 80)

# Check if we can read configs
for dir_name, dir_path in [("Medical", medical_dir), ("RAG", rag_dir)]:
    config_path = os.path.join(dir_path, "config.json")
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            print(f"\n{dir_name} model config:")
            print(f"  - Model type: {config.get('model_type', 'unknown')}")
            print(f"  - Architecture: {config.get('architectures', ['unknown'])[0]}")
            print(f"  - Vocab size: {config.get('vocab_size', 'unknown')}")
            print(f"  - Hidden size: {config.get('hidden_size', 'unknown')}")
            print(f"  - Num layers: {config.get('num_hidden_layers', 'unknown')}")
        except Exception as e:
            print(f"  ❌ Could not read config: {e}")

print("\n" + "=" * 80)
print("3. Checking system resources:")
print("-" * 80)

# Disk space
try:
    result = subprocess.run(["df", "-h", "."], capture_output=True, text=True, check=True)
    print("Disk space:")
    print(result.stdout)
except:
    print("Could not check disk space")

# Memory
try:
    result = subprocess.run(["free", "-h"], capture_output=True, text=True, check=True)
    print("Memory:")
    print(result.stdout)
except:
    print("Could not check memory (may not be available)")

print("\n" + "=" * 80)
print("4. Checking llama.cpp setup:")
print("-" * 80)

if os.path.exists("llama.cpp"):
    print("✅ llama.cpp directory exists")
    converter_script = "llama.cpp/unsloth_convert_hf_to_gguf.py"
    if os.path.exists(converter_script):
        print(f"✅ Converter script exists: {converter_script}")
        # Check script size
        size = os.path.getsize(converter_script) / 1024
        print(f"   Script size: {size:.2f} KB")
    else:
        print(f"❌ Converter script missing: {converter_script}")
else:
    print("❌ llama.cpp directory not found")

print("\n" + "=" * 80)
print("5. Testing manual conversion (RAG model):")
print("-" * 80)

if os.path.exists(rag_dir) and os.path.exists("llama.cpp/unsloth_convert_hf_to_gguf.py"):
    print("Attempting manual conversion to see actual error...")
    print("Command: python llama.cpp/unsloth_convert_hf_to_gguf.py --outfile test.gguf --outtype bf16 gguf_model_rag_analysis")
    print()
    
    try:
        # Run without capturing output to see the real error
        result = subprocess.run(
            f"python llama.cpp/unsloth_convert_hf_to_gguf.py --outfile test.gguf --outtype bf16 {rag_dir}",
            shell=True,
            check=False,
            capture_output=False  # Show output in real-time
        )
        
        if result.returncode == 0:
            print("\n✅ Manual conversion succeeded!")
            print("   The issue might be with unsloth's wrapper function.")
        else:
            print(f"\n❌ Manual conversion failed (return code: {result.returncode})")
            print("   The error above shows the actual problem.")
    except Exception as e:
        print(f"\n❌ Error running conversion: {e}")
        import traceback
        traceback.print_exc()
else:
    print("⚠️  Cannot test - missing required files/directories")

print("\n" + "=" * 80)
print("6. Recommendations:")
print("-" * 80)

if os.path.exists(rag_dir):
    files = os.listdir(rag_dir)
    has_model = any("model.safetensors" in f or "pytorch_model" in f for f in files)
    
    if not has_model:
        print("❌ No model file found in merged directory")
        print("   → The merge step may have failed")
        print("   → Try restarting runtime and running training again")
    else:
        print("✅ Model file exists")
        print("   → The merge step succeeded")
        print("   → The issue is likely in the llama.cpp conversion")
        print("   → Try:")
        print("     1. Restart Colab runtime")
        print("     2. Run conversion manually (see step 5 above)")
        print("     3. Check if medical bot conversion works in same session")
        print("     4. Compare file sizes between medical and RAG merged models")

print("\n" + "=" * 80)


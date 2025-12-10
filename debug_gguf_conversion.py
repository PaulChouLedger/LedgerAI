#!/usr/bin/env python3
"""
Debug script to manually run GGUF conversion and see actual error messages
Run this in Colab after training if GGUF conversion fails
"""

import os
import subprocess
import sys

# Configuration - update these to match your training
OUTPUT_DIR = "outputs_rag_analysis"
GGUF_OUTPUT_DIR = "gguf_model_rag_analysis"

print("=" * 80)
print("Debugging GGUF Conversion")
print("=" * 80)
print()

# Check if merged model exists
if not os.path.exists(GGUF_OUTPUT_DIR):
    print(f"❌ Directory {GGUF_OUTPUT_DIR} does not exist!")
    print(f"   The model merge step may have failed.")
    print(f"   Try running the training script again.")
    sys.exit(1)

print(f"✅ Directory {GGUF_OUTPUT_DIR} exists")
files = os.listdir(GGUF_OUTPUT_DIR)
print(f"📁 Files in directory ({len(files)} total):")
for f in files[:20]:
    file_path = os.path.join(GGUF_OUTPUT_DIR, f)
    if os.path.isfile(file_path):
        size = os.path.getsize(file_path) / 1024 / 1024  # MB
        print(f"   - {f} ({size:.2f} MB)")
    else:
        print(f"   - {f}/ (directory)")

# Check for required files
required_files = ["config.json", "model.safetensors"]
missing = []
for req_file in required_files:
    if not any(req_file in f for f in files):
        missing.append(req_file)

if missing:
    print(f"\n⚠️  Missing required files: {missing}")
    print(f"   The model merge may not have completed successfully.")
else:
    print(f"\n✅ All required files present")

# Check disk space
print(f"\n💾 Disk space check:")
try:
    result = subprocess.run(["df", "-h", "."], capture_output=True, text=True, check=True)
    print(result.stdout)
except:
    print("   Could not check disk space")

# Check memory
print(f"\n🧠 Memory check:")
try:
    result = subprocess.run(["free", "-h"], capture_output=True, text=True, check=True)
    print(result.stdout)
except:
    print("   Could not check memory (may not be available on this system)")

# Try manual conversion to see actual error
print(f"\n" + "=" * 80)
print("Attempting manual conversion to see actual error...")
print("=" * 80)

# Check if llama.cpp directory exists
if not os.path.exists("llama.cpp"):
    print("❌ llama.cpp directory not found!")
    print("   Unsloth should have created this. Try restarting runtime.")
    sys.exit(1)

# Try to run the conversion manually
conversion_script = "llama.cpp/unsloth_convert_hf_to_gguf.py"
if not os.path.exists(conversion_script):
    print(f"❌ Conversion script not found: {conversion_script}")
    sys.exit(1)

print(f"✅ Found conversion script: {conversion_script}")
print(f"\n🔧 Running conversion (this will show the actual error)...")
print(f"   Command: python {conversion_script} --outfile test.gguf --outtype bf16 {GGUF_OUTPUT_DIR}")
print()

try:
    # Run without capturing output so we can see the error
    result = subprocess.run(
        f"python {conversion_script} --outfile test.gguf --outtype bf16 {GGUF_OUTPUT_DIR}",
        shell=True,
        check=False,  # Don't raise on error, we want to see the output
        capture_output=False  # Show output in real-time
    )
    
    if result.returncode == 0:
        print(f"\n✅ Manual conversion succeeded!")
        print(f"   The issue might be with unsloth's wrapper. Try using the manual conversion.")
    else:
        print(f"\n❌ Manual conversion failed with return code: {result.returncode}")
        print(f"   The error above shows what went wrong.")
        
except Exception as e:
    print(f"\n❌ Error running conversion: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 80)
print("Debugging complete")
print("=" * 80)


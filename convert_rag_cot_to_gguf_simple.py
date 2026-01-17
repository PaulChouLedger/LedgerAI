#!/usr/bin/env python3
"""
Simple GGUF Conversion Script (Fallback Only)

NOTE: This script is normally NOT needed!
The training script (train_rag_cot_colab.py) automatically converts to GGUF during training.

Only use this script if:
- Automatic conversion during training failed
- You need to re-convert an already-trained model
- You're working with a model trained before the auto-conversion feature

Uses Unsloth's save_pretrained_gguf() - the same method used in train_rag_cot_colab.py
This is the most reliable method as it matches exactly what the training script does.
"""

import os
import sys
import glob
import shutil
import subprocess

# Configuration
HF_MODEL_PATH = "outputs_rag_cot"  # LoRA adapters
GGUF_OUTPUT_DIR = "gguf_model_rag_cot"
QUANTIZATION = "q4_k_m"  # Q4_K_M quantization
EXPECTED_FILENAME = "Qwen2.5-1.5B-Instruct.Q4_K_M-rag-cot.gguf"

print("=" * 80)
print("Simple GGUF Conversion (Using Training Script Method)")
print("=" * 80)
print(f"Source: {HF_MODEL_PATH}")
print(f"Output: {GGUF_OUTPUT_DIR}")
print(f"Quantization: {QUANTIZATION}")
print()

# GPU Check with auto-fix for CPU-only PyTorch
print("=" * 80)
print("GPU Check")
print("=" * 80)

# First check if GPU hardware is available (nvidia-smi)
gpu_hardware_available = False
try:
    import subprocess
    result = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=5)
    if result.returncode == 0:
        gpu_hardware_available = True
        print(f"   ✅ GPU hardware detected (nvidia-smi works)")
        # Extract GPU name from nvidia-smi output
        for line in result.stdout.split('\n'):
            if 'NVIDIA' in line or 'A100' in line or 'T4' in line or 'V100' in line:
                print(f"   {line.strip()}")
    else:
        print(f"   ⚠️  nvidia-smi returned error")
except (FileNotFoundError, subprocess.TimeoutExpired):
    print(f"   ⚠️  nvidia-smi not available or timed out")
except Exception as e:
    print(f"   ⚠️  nvidia-smi check failed: {e}")

# Check PyTorch CUDA support
try:
    import torch
    print(f"   PyTorch version: {torch.__version__}")
    
    # Check if PyTorch version is too old (needs 2.6+ for torch.int1 support)
    torch_version_parts = torch.__version__.split('+')[0].split('.')
    torch_major = int(torch_version_parts[0])
    torch_minor = int(torch_version_parts[1]) if len(torch_version_parts) > 1 else 0
    needs_upgrade = torch_major < 2 or (torch_major == 2 and torch_minor < 6)
    
    # Check if torch.int1 exists (more reliable check)
    has_int1 = hasattr(torch, 'int1')
    
    if needs_upgrade or not has_int1:
        if torch.cuda.is_available():
            print(f"   ⚠️  PyTorch {torch.__version__} is too old (needs 2.6+ for unsloth compatibility)")
            print(f"   🔧 Upgrading PyTorch to 2.6+ with CUDA support...")
        else:
            print(f"   ⚠️  PyTorch {torch.__version__} is too old and CUDA not available")
            if not gpu_hardware_available:
                print(f"   ❌ No GPU hardware detected")
                sys.exit(1)
            print(f"   🔧 Installing PyTorch 2.6+ with CUDA support...")
        
        print(f"   (Note: You may need to restart runtime after installation)")
        try:
            # Check for xformers dependency conflict
            try:
                import pkg_resources
                xformers_pkg = pkg_resources.get_distribution("xformers")
                xformers_version = xformers_pkg.version
                print(f"   ⚠️  xformers {xformers_version} detected")
                print(f"   (You may see a version conflict warning - this is usually OK)")
            except:
                pass
            
            # Install PyTorch 2.6+ with CUDA (needed for torch.int1 support)
            # Try CUDA 12.4 first (matches system), fallback to 12.1
            print(f"   Installing PyTorch 2.6+ with CUDA 12.4 support...")
            try:
                subprocess.check_call([
                    sys.executable, "-m", "pip", "install", 
                    "torch>=2.6.0", "torchvision", "torchaudio", 
                    "--index-url", "https://download.pytorch.org/whl/cu124",
                    "--upgrade"
                ], stderr=subprocess.DEVNULL)
                print(f"   ✅ Installed PyTorch with CUDA 12.4")
            except:
                print(f"   ⚠️  CUDA 12.4 not available, trying CUDA 12.1...")
                subprocess.check_call([
                    sys.executable, "-m", "pip", "install", 
                    "torch>=2.6.0", "torchvision", "torchaudio", 
                    "--index-url", "https://download.pytorch.org/whl/cu121",
                    "--upgrade"
                ], stderr=subprocess.DEVNULL)
                print(f"   ✅ Installed PyTorch with CUDA 12.1")
            print(f"   ✅ PyTorch upgraded!")
            print(f"   ")
            print(f"   ⚠️  IMPORTANT: Runtime restart required!")
            print(f"   ")
            print(f"   Please do the following:")
            print(f"   1. Runtime > Restart runtime (or Runtime > Restart and run all)")
            print(f"   2. Re-run this script")
            print(f"   ")
            print(f"   This is necessary because PyTorch is already loaded in memory.")
            print(f"   After restart, PyTorch will load with torch.int1 support.")
            sys.exit(0)  # Exit gracefully - user needs to restart
        except Exception as install_err:
            print(f"   ❌ Failed to upgrade PyTorch: {install_err}")
            print(f"   ")
            print(f"   💡 Manual installation:")
            print(f"   Run in a new cell:")
            print(f"   !pip install torch>=2.6.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124 --upgrade")
            print(f"   Then: Runtime > Restart runtime")
            print(f"   Then: Re-run this script")
            sys.exit(1)
    
    # If we reach here, PyTorch is 2.6+ but might be CPU-only
    if 'cpu' in torch.__version__.lower() or not torch.cuda.is_available():
        print(f"   ⚠️  PyTorch {torch.__version__} detected but CUDA not available!")
        if gpu_hardware_available:
            print(f"   🔧 GPU hardware is available - installing PyTorch with CUDA support...")
            print(f"   (Note: You may need to restart runtime after installation)")
            try:
                # Install PyTorch 2.6+ with CUDA (needed for torch.int1 support)
                # Try CUDA 12.4 first (matches system), fallback to 12.1
                print(f"   Installing PyTorch 2.6+ with CUDA 12.4 support...")
                try:
                    subprocess.check_call([
                        sys.executable, "-m", "pip", "install", 
                        "torch>=2.6.0", "torchvision", "torchaudio", 
                        "--index-url", "https://download.pytorch.org/whl/cu124",
                        "--upgrade"
                    ], stderr=subprocess.DEVNULL)
                    print(f"   ✅ Installed PyTorch with CUDA 12.4")
                except:
                    print(f"   ⚠️  CUDA 12.4 not available, trying CUDA 12.1...")
                    subprocess.check_call([
                        sys.executable, "-m", "pip", "install", 
                        "torch>=2.6.0", "torchvision", "torchaudio", 
                        "--index-url", "https://download.pytorch.org/whl/cu121",
                        "--upgrade"
                    ], stderr=subprocess.DEVNULL)
                    print(f"   ✅ Installed PyTorch with CUDA 12.1")
                print(f"   ✅ PyTorch with CUDA installed!")
                print(f"   ")
                print(f"   ⚠️  IMPORTANT: Runtime restart required!")
                print(f"   ")
                print(f"   Please do the following:")
                print(f"   1. Runtime > Restart runtime (or Runtime > Restart and run all)")
                print(f"   2. Re-run this script")
                print(f"   ")
                print(f"   This is necessary because PyTorch is already loaded in memory.")
                print(f"   After restart, PyTorch will load with CUDA support.")
                sys.exit(0)  # Exit gracefully - user needs to restart
            except Exception as install_err:
                print(f"   ❌ Failed to install PyTorch with CUDA: {install_err}")
                print(f"   ")
                print(f"   💡 Manual installation:")
                print(f"   Run in a new cell:")
                print(f"   !pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 --upgrade")
                print(f"   Then: Runtime > Restart runtime")
                print(f"   Then: Re-run this script")
                sys.exit(1)
    
    print(f"   CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   CUDA version: {torch.version.cuda}")
        print(f"   GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            memory_gb = props.total_memory / (1024**3)
            print(f"   GPU {i}: {props.name} ({memory_gb:.1f} GB)")
        print(f"   ✅ GPU ready for conversion")
    else:
        if gpu_hardware_available:
            print(f"   ❌ GPU hardware available but PyTorch CUDA not working!")
            print(f"   💡 Try: Runtime > Restart runtime, then re-run this script")
            sys.exit(1)
        else:
            print(f"   ❌ No GPU detected!")
            print(f"   💡 Check: Runtime > Change runtime type > Hardware accelerator: GPU")
            print(f"   Then restart runtime and re-run this script")
            sys.exit(1)
except ImportError:
    print(f"   ⚠️  PyTorch not installed - will install with CUDA...")
    if gpu_hardware_available:
        try:
            import subprocess
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", 
                "torch", "torchvision", "torchaudio", 
                "--index-url", "https://download.pytorch.org/whl/cu121",
                "-q"
            ])
            import torch
            print(f"   ✅ PyTorch with CUDA installed: {torch.__version__}")
        except Exception as e:
            print(f"   ❌ Failed to install PyTorch: {e}")
            sys.exit(1)
    else:
        print(f"   ❌ No GPU hardware detected - cannot proceed")
        sys.exit(1)
except Exception as e:
    print(f"   ❌ GPU check error: {e}")
    sys.exit(1)

# Load model using Unsloth (same as training script)
print("\n" + "=" * 80)
print("Loading Model with Unsloth")
print("=" * 80)
try:
    from unsloth import FastLanguageModel
    
    print(f"Loading model from: {HF_MODEL_PATH}")
    print("   (Unsloth automatically merges LoRA adapters during GGUF conversion)")
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=HF_MODEL_PATH,
        max_seq_length=4096,  # Match training
        dtype=None,  # Auto detection
        load_in_4bit=True,
    )
    print("✅ Model loaded")
    
except Exception as e:
    print(f"❌ Failed to load model: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Convert to GGUF (same method as training script)
print("\n" + "=" * 80)
print("Converting to GGUF Format")
print("=" * 80)

# Ensure output directory exists
os.makedirs(GGUF_OUTPUT_DIR, exist_ok=True)

print(f"Converting to GGUF with {QUANTIZATION} quantization...")
print("   (This automatically merges LoRA adapters and quantizes in one step)")
print("   (This may take several minutes...)")

# Pre-install llama.cpp to avoid Unsloth's broken build process
print("\nPre-installing llama.cpp (workaround for Unsloth build issues)...")
try:
    if not os.path.exists("llama.cpp"):
        print("   Cloning llama.cpp repository...")
        subprocess.check_call([
            "git", "clone", "--depth", "1", 
            "https://github.com/ggerganov/llama.cpp.git"
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print("   ✅ llama.cpp cloned")
    else:
        print("   ✅ llama.cpp already exists")
    
    # Build llama.cpp with correct CMake options (without deprecated LLAMA_CURL)
    llama_cpp_build = os.path.join("llama.cpp", "build")
    if not os.path.exists(os.path.join(llama_cpp_build, "bin", "quantize")):
        print("   Building llama.cpp (this may take a few minutes)...")
        current_dir = os.getcwd()
        os.chdir("llama.cpp")
        try:
            # Build with CMake (correct options, no deprecated LLAMA_CURL)
            subprocess.check_call([
                "cmake", "-B", "build", 
                "-DCMAKE_BUILD_TYPE=Release",
                "-DBUILD_SHARED_LIBS=OFF"
            ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            subprocess.check_call([
                "cmake", "--build", "build", "--config", "Release", "-j"
            ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            os.chdir(current_dir)
            print("   ✅ llama.cpp built successfully")
        except Exception as build_err:
            os.chdir(current_dir)
            print(f"   ⚠️  llama.cpp build failed: {build_err}")
            print("   Will let Unsloth try its own build (may fail)")
    else:
        print("   ✅ llama.cpp already built")
        
except Exception as e:
    print(f"   ⚠️  Pre-installation failed: {e}")
    print("   Will let Unsloth try its own build (may fail)")

try:
    # This is the EXACT same method used in train_rag_cot_colab.py
    # If llama.cpp is pre-built, Unsloth will use it instead of trying to build
    model.save_pretrained_gguf(
        GGUF_OUTPUT_DIR,
        tokenizer,
        quantization_method=QUANTIZATION
    )
    print("✅ GGUF conversion completed")
except Exception as e:
    print(f"❌ GGUF conversion failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Wait for file system sync
import time
time.sleep(5)

# Find and rename GGUF file
print("\n" + "=" * 80)
print("Locating GGUF File")
print("=" * 80)

gguf_files = []
# Check in output directory
if os.path.exists(GGUF_OUTPUT_DIR):
    gguf_files = [f for f in os.listdir(GGUF_OUTPUT_DIR) if f.endswith(".gguf")]
    # Also check subdirectories
    for root, dirs, files in os.walk(GGUF_OUTPUT_DIR):
        for file in files:
            if file.endswith(".gguf"):
                full_path = os.path.join(root, file)
                rel_path = os.path.relpath(full_path, GGUF_OUTPUT_DIR)
                if rel_path not in gguf_files:
                    gguf_files.append(rel_path)

# Also check root directory
root_gguf_files = [f for f in os.listdir(".") if f.endswith(".gguf")]
if root_gguf_files:
    print(f"⚠️  Found GGUF files in root directory: {root_gguf_files}")
    for root_file in root_gguf_files:
        dest = os.path.join(GGUF_OUTPUT_DIR, root_file)
        shutil.move(root_file, dest)
        gguf_files.append(root_file)
        print(f"   Moved {root_file} to {GGUF_OUTPUT_DIR}")

if not gguf_files:
    print("❌ No GGUF files found after conversion!")
    print("   Check the output directory manually")
    sys.exit(1)

# Find the quantized file (should have Q4_K_M in name)
quantized_files = [f for f in gguf_files if QUANTIZATION.lower() in f.lower()]
if quantized_files:
    source_file = os.path.join(GGUF_OUTPUT_DIR, quantized_files[0])
    if not os.path.isabs(source_file) and not os.path.exists(source_file):
        # Try absolute path
        source_file = os.path.abspath(os.path.join(GGUF_OUTPUT_DIR, quantized_files[0]))
    
    final_file = os.path.join(GGUF_OUTPUT_DIR, EXPECTED_FILENAME)
    
    if source_file != final_file:
        if os.path.exists(final_file):
            os.remove(final_file)
        shutil.move(source_file, final_file)
        print(f"✅ Renamed to: {EXPECTED_FILENAME}")
    else:
        print(f"✅ File already named correctly: {EXPECTED_FILENAME}")
    
    file_size_mb = os.path.getsize(final_file) / (1024 * 1024)
    print(f"\n✅ GGUF Model Ready!")
    print(f"   File: {final_file}")
    print(f"   Size: {file_size_mb:.2f} MB")
    print(f"   Quantization: {QUANTIZATION}")
else:
    print(f"⚠️  Quantized file not found, but found: {gguf_files}")
    print(f"   Using first file: {gguf_files[0]}")
    source_file = os.path.join(GGUF_OUTPUT_DIR, gguf_files[0])
    file_size_mb = os.path.getsize(source_file) / (1024 * 1024)
    print(f"   File: {source_file}")
    print(f"   Size: {file_size_mb:.2f} MB")

print("\n" + "=" * 80)
print("✅ Conversion Complete!")
print("=" * 80)
print("This used the EXACT same method as train_rag_cot_colab.py")
print("The model should work correctly with CoT reasoning.")

#!/usr/bin/env python3
"""
Manual GGUF Conversion Script for Colab
========================================
Use this script when Unsloth's automatic GGUF conversion fails.
This uses llama.cpp's official converter which is more reliable.
"""

import os
import subprocess
import sys
import glob
import shutil
import time

# Configuration
HF_MODEL_PATH = "outputs_rag_cot"
GGUF_OUTPUT_DIR = "gguf_model_rag_cot"
QUANTIZATION = "q4_k_m"  # Q4_K_M quantization
EXPECTED_FILENAME = "Qwen2.5-1.5B-Instruct.Q4_K_M-rag-cot.gguf"

# Colab path handling: ensure we're in /content directory
if os.path.exists("/content"):
    # We're in Colab - change to /content directory
    os.chdir("/content")
    print("📁 Changed working directory to /content (Colab)")
elif os.path.exists("/Users"):
    # We're on macOS - stay in current directory
    pass
else:
    # Try to find content directory
    if os.path.exists("./content"):
        os.chdir("./content")
        print("📁 Changed working directory to ./content")

print("=" * 80)
print("Manual GGUF Conversion Script")
print("=" * 80)
print(f"Current directory: {os.getcwd()}")
print(f"Source: {HF_MODEL_PATH}")
print(f"Output: {GGUF_OUTPUT_DIR}")
print(f"Quantization: {QUANTIZATION}")
print()

# GPU Diagnostic Check
print("=" * 80)
print("GPU Diagnostic Check")
print("=" * 80)

# First check if GPU hardware is available (nvidia-smi)
gpu_hardware_available = False
try:
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
    
    if 'cpu' in torch.__version__.lower():
        print(f"   ⚠️  PyTorch CPU-only version detected!")
        if gpu_hardware_available:
            print(f"   🔧 GPU hardware is available but PyTorch doesn't have CUDA support")
            print(f"   Installing PyTorch with CUDA support...")
            try:
                # Install PyTorch with CUDA 12.1 (works with most Colab GPUs)
                subprocess.check_call([
                    sys.executable, "-m", "pip", "install", 
                    "torch", "torchvision", "torchaudio", 
                    "--index-url", "https://download.pytorch.org/whl/cu121",
                    "-q", "--upgrade"
                ])
                print(f"   ✅ PyTorch with CUDA installed - reloading...")
                # Force reload torch
                import importlib
                import sys
                if 'torch' in sys.modules:
                    del sys.modules['torch']
                import torch
                print(f"   New PyTorch version: {torch.__version__}")
            except Exception as install_err:
                print(f"   ❌ Failed to install PyTorch with CUDA: {install_err}")
                print(f"   💡 Try manually: !pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
                raise
    
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
        else:
            print(f"   ❌ No GPU detected!")
            print(f"   💡 Check: Runtime > Change runtime type > Hardware accelerator: GPU")
            print(f"   Then restart runtime and re-run this script")
except ImportError:
    print(f"   ⚠️  PyTorch not installed - will install with CUDA...")
    if gpu_hardware_available:
        try:
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
            raise
except Exception as e:
    print(f"   ⚠️  GPU check error: {e}")
print()

# Step 1: Ensure output directory exists
os.makedirs(GGUF_OUTPUT_DIR, exist_ok=True)
print(f"✅ Output directory ready: {os.path.abspath(GGUF_OUTPUT_DIR)}")

# Step 2: Check if model needs merging (LoRA adapters vs full model)
print("\n" + "=" * 80)
print("Checking Model Format")
print("=" * 80)

needs_merging = False
merged_model_path = HF_MODEL_PATH

# Check if config.json exists (indicates full model)
if not os.path.exists(os.path.join(HF_MODEL_PATH, "config.json")):
    print("⚠️  config.json not found - model appears to be LoRA adapters")
    
    # Check if Unsloth already created a merged model in gguf_model_rag_cot
    # (Unsloth's save_pretrained_gguf creates a merged model there before converting)
    potential_merged = os.path.join(GGUF_OUTPUT_DIR, "model.safetensors")
    if os.path.exists(potential_merged) or os.path.exists(os.path.join(GGUF_OUTPUT_DIR, "config.json")):
        print(f"   ✅ Found merged model in {GGUF_OUTPUT_DIR} (from Unsloth merge step)")
        merged_model_path = GGUF_OUTPUT_DIR
        needs_merging = False
    else:
        print("   Will merge adapters first, then convert")
        needs_merging = True
        merged_model_path = "outputs_rag_cot_merged"
else:
    print("✅ Full model detected (config.json found)")

# Step 3: Try Unsloth conversion first (may work after runtime restart)
print("\n" + "=" * 80)
print("Attempt 1: Unsloth Conversion")
print("=" * 80)

try:
    import torch
    # Better GPU detection
    has_cuda = torch.cuda.is_available()
    if has_cuda:
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"✅ GPU detected: {gpu_name} ({gpu_memory:.1f} GB)")
    else:
        print("⚠️  torch.cuda.is_available() returned False")
        print("   This means PyTorch was compiled without CUDA support")
        print("   💡 The script should have auto-installed PyTorch with CUDA above")
        print("   If you see this, PyTorch installation may have failed")
        raise RuntimeError("No GPU available - PyTorch doesn't have CUDA support")
    
    if not has_cuda:
        raise RuntimeError("No GPU available")
    
    from unsloth import FastLanguageModel
    
    print("Loading model...")
    # Unsloth's save_pretrained_gguf automatically merges adapters during conversion
    # So we don't need to merge manually - just load and convert
    model, tokenizer = FastLanguageModel.from_pretrained(HF_MODEL_PATH)
    print("✅ Model loaded")
    
    print(f"Starting GGUF conversion with {QUANTIZATION}...")
    print("   Note: Unsloth will automatically merge LoRA adapters during conversion")
    model.save_pretrained_gguf(
        GGUF_OUTPUT_DIR,
        tokenizer,
        quantization_method=QUANTIZATION
    )
    print("✅ Unsloth conversion completed")
    
    # Wait for file system sync
    time.sleep(5)
    
    # Find GGUF files
    gguf_files = []
    if os.path.exists(GGUF_OUTPUT_DIR):
        gguf_files = glob.glob(f"{GGUF_OUTPUT_DIR}/**/*.gguf", recursive=True)
    if not gguf_files:
        gguf_files = glob.glob("*.gguf")
    
    if gguf_files:
        print(f"✅ Found {len(gguf_files)} GGUF file(s)")
        old_file = gguf_files[0]
        new_file = os.path.join(GGUF_OUTPUT_DIR, EXPECTED_FILENAME)
        
        if os.path.exists(new_file):
            os.remove(new_file)
        
        shutil.move(old_file, new_file)
        file_size_mb = os.path.getsize(new_file) / (1024 * 1024)
        print(f"✅ GGUF saved as: {EXPECTED_FILENAME}")
        print(f"   Size: {file_size_mb:.2f} MB")
        print(f"   Location: {new_file}")
        sys.exit(0)
    else:
        print("⚠️  No GGUF files found after Unsloth conversion")
        raise Exception("Unsloth conversion produced no files")
        
except Exception as e:
    print(f"❌ Unsloth conversion failed: {e}")
    print("\n" + "=" * 80)
    print("Attempt 2: llama.cpp Official Converter")
    print("=" * 80)
    
    # Step 3: Check for merged model or merge adapters if needed
    # IMPORTANT: Verify the merged model is actually fine-tuned, not just base model
    merged_in_gguf_dir = os.path.join(GGUF_OUTPUT_DIR, "config.json")
    merged_model_is_fine_tuned = False
    
    if os.path.exists(merged_in_gguf_dir):
        # Check if this merged model has training artifacts (indicates it's fine-tuned)
        training_state = os.path.join(GGUF_OUTPUT_DIR, "training_state.json")
        training_args = os.path.join(GGUF_OUTPUT_DIR, "training_args.json")
        adapter_config = os.path.join(HF_MODEL_PATH, "adapter_config.json")
        
        # If source has adapters but merged model has no training artifacts, it's likely base model
        if os.path.exists(adapter_config) and not os.path.exists(training_state) and not os.path.exists(training_args):
            print(f"⚠️  Found merged model in {GGUF_OUTPUT_DIR}, but it appears to be BASE model, not fine-tuned!")
            print(f"   Source model has LoRA adapters but merged model has no training artifacts")
            print(f"   Will delete and re-merge with adapters...")
            # Delete the base model files
            import glob
            for safetensor in glob.glob(os.path.join(GGUF_OUTPUT_DIR, "*.safetensors")):
                os.remove(safetensor)
                print(f"   Deleted: {os.path.basename(safetensor)}")
            if os.path.exists(merged_in_gguf_dir):
                os.remove(merged_in_gguf_dir)
            if os.path.exists(os.path.join(GGUF_OUTPUT_DIR, "tokenizer_config.json")):
                os.remove(os.path.join(GGUF_OUTPUT_DIR, "tokenizer_config.json"))
            print(f"   ✅ Cleaned up base model files")
            needs_merging = True
            merged_model_is_fine_tuned = False
        else:
            print(f"✅ Found merged model in {GGUF_OUTPUT_DIR} (from Unsloth merge step)")
            model_path_for_conversion = GGUF_OUTPUT_DIR
            needs_merging = False
            merged_model_is_fine_tuned = True
    elif needs_merging and not os.path.exists(os.path.join(merged_model_path, "config.json")):
        print("Merging LoRA adapters into full model (required for llama.cpp converter)...")
        
        # Verify GPU is available
        import torch
        if not torch.cuda.is_available():
            print("❌ No GPU available - cannot merge adapters")
            print("   💡 Make sure Colab runtime has GPU enabled:")
            print("      Runtime > Change runtime type > Hardware accelerator: GPU")
            print("      Then restart runtime and re-run this script")
            raise RuntimeError("No GPU available - GPU required for merging LoRA adapters")
        
        gpu_name = torch.cuda.get_device_name(0)
        print(f"   ✅ Using GPU: {gpu_name}")
        
        try:
            from unsloth import FastLanguageModel
            print(f"   Loading model with LoRA adapters from: {HF_MODEL_PATH}")
            model, tokenizer = FastLanguageModel.from_pretrained(HF_MODEL_PATH)
            print(f"   Merging LoRA adapters into base model...")
            model = model.merge_and_unload()
            merge_success = True
        except Exception as merge_err:
            print(f"❌ Failed to merge adapters: {merge_err}")
            print(f"   Error details:")
            import traceback
            traceback.print_exc()
            raise RuntimeError(f"Failed to merge adapters: {merge_err}")
        
        if merge_success:
            os.makedirs(merged_model_path, exist_ok=True)
            print(f"   Saving merged model to: {merged_model_path}")
            model.save_pretrained(merged_model_path, safe_serialization=True)
            tokenizer.save_pretrained(merged_model_path)
            
            # Copy training artifacts if they exist (to mark this as fine-tuned)
            training_state_src = os.path.join(HF_MODEL_PATH, "training_state.json")
            training_args_src = os.path.join(HF_MODEL_PATH, "training_args.json")
            if os.path.exists(training_state_src):
                import shutil
                shutil.copy(training_state_src, os.path.join(merged_model_path, "training_state.json"))
            if os.path.exists(training_args_src):
                import shutil
                shutil.copy(training_args_src, os.path.join(merged_model_path, "training_args.json"))
            
            # Verify merged model size (should be ~2.9GB for Qwen 1.5B, not 0.55GB)
            import glob
            merged_files = glob.glob(os.path.join(merged_model_path, "*.safetensors"))
            if merged_files:
                total_size = sum(os.path.getsize(f) for f in merged_files) / (1024**3)
                print(f"✅ Adapters merged to {merged_model_path}")
                print(f"   Merged model size: {total_size:.2f} GB (should be ~2.9GB for Qwen 1.5B)")
                if total_size < 1.0:
                    print(f"   ⚠️  WARNING: Merged model seems too small - merge may have failed!")
            model_path_for_conversion = merged_model_path
    else:
        # Use merged model path if available, otherwise use original
        model_path_for_conversion = merged_model_path if os.path.exists(os.path.join(merged_model_path, "config.json")) else HF_MODEL_PATH
    
    # Verify which model we're converting
    print(f"\n{'='*80}")
    print("Model Conversion Verification")
    print(f"{'='*80}")
    print(f"   Source model path: {HF_MODEL_PATH}")
    print(f"   Model to convert: {model_path_for_conversion}")
    
    # Check if source has adapters
    adapter_config = os.path.join(HF_MODEL_PATH, "adapter_config.json")
    has_adapters = os.path.exists(adapter_config)
    
    # Check if target has training artifacts
    training_state = os.path.join(model_path_for_conversion, "training_state.json")
    training_args = os.path.join(model_path_for_conversion, "training_args.json")
    has_training_artifacts = os.path.exists(training_state) or os.path.exists(training_args)
    
    if has_adapters and not has_training_artifacts:
        print(f"   ⚠️  WARNING: Source has LoRA adapters but target has no training artifacts!")
        print(f"   This suggests the model to convert is the BASE model, not fine-tuned.")
        print(f"   The conversion will produce a base model, not a fine-tuned one.")
        print(f"   💡 Solution: Delete merged model in {GGUF_OUTPUT_DIR} and re-run to merge adapters")
    elif has_adapters and has_training_artifacts:
        print(f"   ✅ Source has adapters and target has training artifacts - looks good!")
    elif not has_adapters:
        print(f"   ℹ️  Source appears to be a full model (not LoRA adapters)")
    
    # Step 4: Use llama.cpp's official converter
    try:
        print("Cloning llama.cpp repository...")
        if os.path.exists("llama.cpp"):
            print("   llama.cpp directory already exists, using it...")
        else:
            subprocess.check_call([
                "git", "clone", "--depth", "1", 
                "https://github.com/ggerganov/llama.cpp.git"
            ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print("✅ llama.cpp cloned")
        
        # Install Python dependencies
        print("Installing Python dependencies...")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", 
            "-r", "llama.cpp/requirements.txt", "-q"
        ])
        print("✅ Dependencies installed")
        
        # Convert HF to GGUF (bf16 first)
        print("Converting HuggingFace model to GGUF (bf16)...")
        # Try different possible script names
        possible_scripts = [
            os.path.join("llama.cpp", "convert_hf_to_gguf.py"),  # Most common
            os.path.join("llama.cpp", "convert-hf-to-gguf.py"),  # Alternative
            os.path.join("llama.cpp", "convert.py"),              # Generic
        ]
        
        convert_script = None
        for script_path in possible_scripts:
            if os.path.exists(script_path):
                convert_script = script_path
                print(f"   Found converter: {script_path}")
                break
        
        if not convert_script:
            # List files in llama.cpp to help debug
            llama_files = os.listdir("llama.cpp")
            print(f"   Files in llama.cpp: {llama_files[:10]}...")
            raise FileNotFoundError(f"llama.cpp converter script not found. Tried: {possible_scripts}")
        
        # Convert to f16 first (llama.cpp converter doesn't support direct quantization)
        print("Converting HuggingFace model to GGUF (f16)...")
        # Construct output filename
        base_gguf_name = "model.gguf"  # Default name
        # Try to get model name from config
        try:
            import json
            config_path = os.path.join(model_path_for_conversion, "config.json")
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config = json.load(f)
                    model_name = config.get("model_type", "model")
                    if "qwen" in model_name.lower():
                        base_gguf_name = "qwen2.5-1.5b-instruct-f16.gguf"
        except:
            pass
        
        base_gguf_path = os.path.join(GGUF_OUTPUT_DIR, base_gguf_name)
        
        # llama.cpp converter uses --outfile, not --outdir
        print(f"   Converting model from: {model_path_for_conversion}")
        subprocess.check_call([
            sys.executable, convert_script,
            model_path_for_conversion,
            "--outfile", base_gguf_path,
            "--outtype", "f16"
        ])
        print("✅ HF to GGUF (f16) conversion completed")
        direct_quant = False
        
        # Verify the converted GGUF file exists
        time.sleep(2)
        if not os.path.exists(base_gguf_path):
            # Try to find any GGUF file in the output directory
            gguf_files = glob.glob(f"{GGUF_OUTPUT_DIR}/*.gguf")
            if not gguf_files:
                gguf_files = glob.glob(f"{GGUF_OUTPUT_DIR}/**/*.gguf", recursive=True)
            
            if not gguf_files:
                raise FileNotFoundError(f"No GGUF file found after conversion. Expected: {base_gguf_path}")
            
            base_gguf_path = gguf_files[0]
        
        print(f"✅ Found base GGUF: {os.path.basename(base_gguf_path)}")
        # Convert to absolute path
        base_gguf = os.path.abspath(base_gguf_path)
        if not os.path.exists(base_gguf):
            raise FileNotFoundError(f"Base GGUF file not found: {base_gguf}")
        print(f"   Full path: {base_gguf}")
        
        # Step 4: Quantize to Q4_K_M (if not already quantized)
        if direct_quant and QUANTIZATION.lower() in base_gguf.lower():
            print(f"✅ Already quantized to {QUANTIZATION}")
            quantized_file = base_gguf
        else:
            print(f"Quantizing to {QUANTIZATION}...")
            # Ensure we use the correct absolute path
            # In Colab, this should be /content/gguf_model_rag_cot/...
            if not os.path.isabs(GGUF_OUTPUT_DIR):
                # Make it absolute relative to current working directory
                gguf_output_dir_abs = os.path.abspath(GGUF_OUTPUT_DIR)
            else:
                gguf_output_dir_abs = GGUF_OUTPUT_DIR
            quantized_file = os.path.join(gguf_output_dir_abs, EXPECTED_FILENAME)
            if os.path.exists(quantized_file):
                os.remove(quantized_file)
            
            # Try to use Python quantizer if available (most reliable)
            quantize_py = os.path.join("llama.cpp", "quantize.py")
            quantize_success = False
            
            if os.path.exists(quantize_py):
                print("   Using Python quantizer script...")
                try:
                    result = subprocess.run([
                        sys.executable, quantize_py,
                        base_gguf,
                        quantized_file,
                        QUANTIZATION
                    ], capture_output=True, text=True, timeout=600)
                    
                    if result.returncode == 0 and os.path.exists(quantized_file):
                        file_size_mb = os.path.getsize(quantized_file) / (1024 * 1024)
                        print(f"✅ Quantization completed using Python script")
                        print(f"   Output size: {file_size_mb:.2f} MB")
                        quantize_success = True
                    else:
                        print(f"   ⚠️  Python quantizer failed:")
                        if result.stdout:
                            print(f"      stdout: {result.stdout[:200]}")
                        if result.stderr:
                            print(f"      stderr: {result.stderr[:200]}")
                except subprocess.TimeoutExpired:
                    print(f"   ⚠️  Python quantizer timed out after 10 minutes")
                except Exception as py_err:
                    print(f"   ⚠️  Python quantizer failed: {py_err}")
            
            # Also try alternative script locations
            if not quantize_success:
                alt_scripts = [
                    os.path.join("llama.cpp", "examples", "quantize", "quantize.py"),
                    os.path.join("llama.cpp", "scripts", "quantize.py"),
                ]
                for alt_script in alt_scripts:
                    if os.path.exists(alt_script):
                        print(f"   Trying alternative script: {alt_script}")
                        try:
                            result = subprocess.run([
                                sys.executable, alt_script,
                                base_gguf,
                                quantized_file,
                                QUANTIZATION
                            ], capture_output=True, text=True, timeout=600)
                            
                            if result.returncode == 0 and os.path.exists(quantized_file):
                                file_size_mb = os.path.getsize(quantized_file) / (1024 * 1024)
                                print(f"✅ Quantization completed using alternative script")
                                print(f"   Output size: {file_size_mb:.2f} MB")
                                quantize_success = True
                                break
                        except Exception as alt_err:
                            print(f"   ⚠️  Alternative script failed: {alt_err}")
                            continue
            if not quantize_success:
                # First check if quantize binary already exists
                quantize_binary = None
                possible_paths = [
                    os.path.join("llama.cpp", "build", "bin", "quantize"),
                    os.path.join("llama.cpp", "build", "quantize"),
                    os.path.join("llama.cpp", "quantize"),
                ]
                
                for path in possible_paths:
                    if os.path.exists(path) and os.access(path, os.X_OK):
                        quantize_binary = os.path.abspath(path)
                        print(f"   Found existing quantize binary: {quantize_binary}")
                        break
                
                if not quantize_binary:
                    # Try building quantize binary with CMake (without deprecated options)
                    print("   Building quantize tool with CMake...")
                    current_dir = os.getcwd()
                    os.chdir("llama.cpp")
                    try:
                        # Build with CMake (remove deprecated LLAMA_CURL option)
                        subprocess.check_call([
                            "cmake", "-B", "build", 
                            "-DCMAKE_BUILD_TYPE=Release",
                            "-DBUILD_SHARED_LIBS=OFF"
                        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                        subprocess.check_call([
                            "cmake", "--build", "build", "--config", "Release", "-j"
                        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                        os.chdir("..")
                        
                        quantize_script = os.path.join("llama.cpp", "build", "bin", "quantize")
                        if not os.path.exists(quantize_script):
                            quantize_script = os.path.join("llama.cpp", "build", "quantize")
                        
                        if os.path.exists(quantize_script):
                            quantize_binary = os.path.abspath(quantize_script)
                        else:
                            raise FileNotFoundError("quantize binary not found after build")
                    except Exception as build_err:
                        os.chdir("..")
                        print(f"   ⚠️  CMake build failed: {build_err}")
                        # Will fall through to llama-cpp-python
                        quantize_binary = None
                
                if quantize_binary:
                    try:
                        result = subprocess.run([
                            quantize_binary,
                            base_gguf,
                            quantized_file,
                            QUANTIZATION
                        ], capture_output=True, text=True, timeout=600)
                        
                        if result.returncode == 0 and os.path.exists(quantized_file):
                            file_size_mb = os.path.getsize(quantized_file) / (1024 * 1024)
                            print(f"✅ Quantization completed using binary")
                            print(f"   Output size: {file_size_mb:.2f} MB")
                            quantize_success = True
                        else:
                            raise subprocess.CalledProcessError(result.returncode, quantize_binary, result.stderr)
                    except Exception as bin_err:
                        print(f"   ⚠️  Binary quantization failed: {bin_err}")
                        # Fall through to llama-cpp-python
                
                if not quantize_success:
                    # Last resort: Try using llama-cpp-python for quantization
                    try:
                        print("   Trying llama-cpp-python quantization...")
                        import llama_cpp
                        from llama_cpp import llama_model_quantize_default_params, llama_model_quantize
                        
                        # Use the same absolute path logic as before
                        if not os.path.isabs(GGUF_OUTPUT_DIR):
                            gguf_output_dir_abs = os.path.abspath(GGUF_OUTPUT_DIR)
                        else:
                            gguf_output_dir_abs = GGUF_OUTPUT_DIR
                        quantized_file = os.path.join(gguf_output_dir_abs, EXPECTED_FILENAME)
                        if os.path.exists(quantized_file):
                            os.remove(quantized_file)
                        
                        # Verify input file exists
                        if not os.path.exists(base_gguf):
                            raise FileNotFoundError(f"Input GGUF file not found: {base_gguf}")
                        
                        # Ensure output directory exists
                        os.makedirs(os.path.dirname(quantized_file), exist_ok=True)
                        
                        print(f"   Input: {base_gguf} ({os.path.getsize(base_gguf) / (1024*1024):.2f} MB)")
                        print(f"   Output: {quantized_file}")
                        
                        # Use llama-cpp-python's quantization
                        params = llama_model_quantize_default_params()
                        params.ftype = {
                            "q4_k_m": 3,  # Q4_K_M type
                            "q4_0": 2,
                            "q8_0": 7
                        }.get(QUANTIZATION.lower(), 3)
                        
                        # llama_model_quantize requires bytes paths
                        # Also ensure paths are absolute and correct
                        base_gguf_abs = os.path.abspath(base_gguf)
                        quantized_file_abs = os.path.abspath(quantized_file)
                        
                        # Ensure the directory exists
                        os.makedirs(os.path.dirname(quantized_file_abs), exist_ok=True)
                        
                        base_gguf_bytes = base_gguf_abs.encode('utf-8')
                        quantized_file_bytes = quantized_file_abs.encode('utf-8')
                        
                        llama_model_quantize(base_gguf_bytes, quantized_file_bytes, params)
                        
                        # Verify output was created (check both the specified path and common alternatives)
                        if not os.path.exists(quantized_file_abs):
                            # Try to find the file - it might have been created in a different location
                            possible_locations = [
                                quantized_file_abs,
                                quantized_file,  # Original path
                                os.path.join(os.getcwd(), GGUF_OUTPUT_DIR, EXPECTED_FILENAME),  # Current working dir
                                os.path.join("/content", GGUF_OUTPUT_DIR, EXPECTED_FILENAME),  # Colab content dir
                                os.path.join("/", GGUF_OUTPUT_DIR, EXPECTED_FILENAME),  # Root level (wrong but possible)
                                os.path.join(os.getcwd(), EXPECTED_FILENAME),  # Current directory (no subdir)
                            ]
                            found = False
                            for loc in possible_locations:
                                if os.path.exists(loc):
                                    quantized_file_abs = loc
                                    quantized_file = loc
                                    found = True
                                    print(f"   Found quantized file at: {loc}")
                                    break
                            
                            if not found:
                                raise FileNotFoundError(f"Quantized file was not created. Checked: {possible_locations}")
                        else:
                            quantized_file = quantized_file_abs
                        
                        print(f"✅ Quantization completed using llama-cpp-python")
                        print(f"   Output size: {os.path.getsize(quantized_file) / (1024*1024):.2f} MB")
                        quantize_success = True
                    except Exception as py_err:
                        print(f"   ⚠️  Python quantization also failed: {py_err}")
                        print(f"   Using base GGUF file (f16) - renaming to expected name")
                        quantized_file = os.path.join(GGUF_OUTPUT_DIR, EXPECTED_FILENAME)
                        if os.path.exists(quantized_file):
                            os.remove(quantized_file)
                        shutil.move(base_gguf, quantized_file)
                        print(f"   ⚠️  Note: File is f16, not {QUANTIZATION}. You may need to quantize later.")
        
        # Ensure quantized_file is defined
        if 'quantized_file' not in locals():
            quantized_file = base_gguf
        
        # Rename to expected filename if needed
        # Use absolute path for final file
        if not os.path.isabs(GGUF_OUTPUT_DIR):
            gguf_output_dir_final = os.path.abspath(GGUF_OUTPUT_DIR)
        else:
            gguf_output_dir_final = GGUF_OUTPUT_DIR
        final_file = os.path.join(gguf_output_dir_final, EXPECTED_FILENAME)
        
        # Normalize paths for comparison
        quantized_file_abs = os.path.abspath(quantized_file)
        final_file_abs = os.path.abspath(final_file)
        
        if quantized_file_abs != final_file_abs:
            if os.path.exists(final_file_abs):
                os.remove(final_file_abs)
            # Ensure target directory exists
            os.makedirs(os.path.dirname(final_file_abs), exist_ok=True)
            shutil.move(quantized_file_abs, final_file_abs)
            quantized_file = final_file_abs
        else:
            quantized_file = quantized_file_abs
        
        # Final check: search for file in common wrong locations and move to correct one
        if not os.path.exists(quantized_file):
            print(f"⚠️  File not found at expected location: {quantized_file}")
            print("   Searching for file in common locations...")
            
            search_locations = [
                os.path.join("/", GGUF_OUTPUT_DIR, EXPECTED_FILENAME),  # Root level
                os.path.join("/content", GGUF_OUTPUT_DIR, EXPECTED_FILENAME),  # Colab content
                os.path.join(os.getcwd(), GGUF_OUTPUT_DIR, EXPECTED_FILENAME),  # Current dir
                os.path.join(os.getcwd(), EXPECTED_FILENAME),  # Current dir, no subdir
            ]
            
            found_file = None
            for loc in search_locations:
                if os.path.exists(loc):
                    found_file = loc
                    print(f"   ✅ Found file at: {loc}")
                    break
            
            if found_file:
                # Move to correct location
                correct_location = os.path.join(os.getcwd(), GGUF_OUTPUT_DIR, EXPECTED_FILENAME)
                os.makedirs(os.path.dirname(correct_location), exist_ok=True)
                if os.path.exists(correct_location):
                    os.remove(correct_location)
                shutil.move(found_file, correct_location)
                quantized_file = correct_location
                print(f"   ✅ Moved file to correct location: {correct_location}")
            else:
                print(f"   ❌ File not found in any of these locations: {search_locations}")
        
        # Verify file
        if os.path.exists(quantized_file):
            file_size_mb = os.path.getsize(quantized_file) / (1024 * 1024)
            print(f"✅ GGUF saved as: {EXPECTED_FILENAME}")
            print(f"   Size: {file_size_mb:.2f} MB")
            print(f"   Location: {quantized_file}")
            
            # Clean up base GGUF if different
            if base_gguf != quantized_file and os.path.exists(base_gguf):
                os.remove(base_gguf)
                print(f"   Cleaned up base GGUF file")
            
            print("\n" + "=" * 80)
            print("✅ CONVERSION SUCCESSFUL!")
            print("=" * 80)
            sys.exit(0)
        else:
            raise FileNotFoundError("Quantized file not found after quantization")
            
    except Exception as e2:
        print(f"❌ llama.cpp conversion also failed: {e2}")
        import traceback
        traceback.print_exc()
        print("\n" + "=" * 80)
        print("❌ ALL CONVERSION METHODS FAILED")
        print("=" * 80)
        print("Your model is saved in HuggingFace format at:")
        print(f"   {HF_MODEL_PATH}/")
        print("\nYou can:")
        print("1. Download the HuggingFace model and convert locally")
        print("2. Try restarting the Colab runtime and running this script again")
        print("3. Use a different conversion tool")
        sys.exit(1)

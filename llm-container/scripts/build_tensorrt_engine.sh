#!/bin/bash
# ============================================================================
# TensorRT-LLM Engine Build Script
# Builds TensorRT-LLM engines for supported models
# ============================================================================

set -e

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default paths
TENSORRT_ENGINES_BASE="${TENSORRT_ENGINES_BASE:-/models/tensorrt-llm}"
MODELS_BASE="${MODELS_BASE:-/models}"

# Default model (can be overridden)
MODEL_NAME="${MODEL_NAME:-qwen3-4b-2507}"

print_header() {
    echo ""
    echo "========================================================================"
    echo "   $1"
    echo "========================================================================"
    echo ""
}

# Function to build Qwen engine
build_qwen_engine() {
    local model_name=$1
    local model_path=$2
    local engine_dir=$3
    local context_window=${4:-2048}
    
    print_header "Building Qwen TensorRT-LLM Engine"
    
    echo -e "${BLUE}Model:${NC} $model_name"
    echo -e "${BLUE}Source:${NC} $model_path"
    echo -e "${BLUE}Engine Output:${NC} $engine_dir"
    echo -e "${BLUE}Context Window:${NC} $context_window"
    echo ""
    
    # Check if source model exists
    if [ ! -d "$model_path" ] && [ ! -f "$model_path" ]; then
        echo -e "${RED}❌ Source model not found: $model_path${NC}"
        echo ""
        echo -e "${YELLOW}Debugging info:${NC}"
        echo "  Model path: $model_path"
        echo "  Parent directory exists: $([ -d "$(dirname "$model_path")" ] && echo "✅ Yes" || echo "❌ No")"
        if [ -d "$(dirname "$model_path")" ]; then
            echo "  Parent directory contents:"
            ls -la "$(dirname "$model_path")" 2>/dev/null | head -10 || echo "  (cannot list)"
        fi
        echo ""
        echo -e "${YELLOW}💡 Common issues:${NC}"
        echo "  1. Model not downloaded yet - run 'hf download' first"
        echo "  2. Volume mount path mismatch - check docker -v path"
        echo "  3. Wrong model path - verify the exact directory name"
        echo ""
        echo -e "${YELLOW}💡 To download the model:${NC}"
        echo "  hf download meta-llama/Llama-3.2-1B-Instruct --local-dir $model_path"
        return 1
    fi
    
    # Create engine directory
    mkdir -p "$engine_dir"
    
    echo -e "${GREEN}🚀 Starting TensorRT-LLM build...${NC}"
    echo ""
    
    # Build TensorRT-LLM engine
    # Optimized for low latency (1-2s target)
    # Note: max_seq_len = max_input_len + generation length
    max_seq_len=$((context_window + 256))  # Input context + output generation
    
    trtllm-build \
        --checkpoint_dir "$model_path" \
        --output_dir "$engine_dir" \
        --gemm_plugin float16 \
        --gpt_attention_plugin float16 \
        --context_fmha enable \
        --remove_input_padding enable \
        --max_batch_size 1 \
        --max_input_len $context_window \
        --max_seq_len $max_seq_len \
        --max_beam_width 1 \
        --builder_opt 3 \
        || {
            echo -e "${RED}❌ TensorRT-LLM build failed${NC}"
            return 1
        }
    
    echo ""
    echo -e "${GREEN}✅ Engine built successfully: $engine_dir${NC}"
}

# Function to build Llama engine
build_llama_engine() {
    local model_name=$1
    local model_path=$2
    local engine_dir=$3
    local context_window=${4:-2048}
    
    print_header "Building Llama TensorRT-LLM Engine"
    
    echo -e "${BLUE}Model:${NC} $model_name"
    echo -e "${BLUE}Source:${NC} $model_path"
    echo -e "${BLUE}Engine Output:${NC} $engine_dir"
    echo -e "${BLUE}Context Window:${NC} $context_window"
    echo ""
    
    # Check if source model exists
    if [ ! -d "$model_path" ] && [ ! -f "$model_path" ]; then
        echo -e "${RED}❌ Source model not found: $model_path${NC}"
        echo ""
        echo -e "${YELLOW}Debugging info:${NC}"
        echo "  Model path: $model_path"
        echo "  Parent directory exists: $([ -d "$(dirname "$model_path")" ] && echo "✅ Yes" || echo "❌ No")"
        echo "  Parent directory contents:"
        if [ -d "$(dirname "$model_path")" ]; then
            ls -la "$(dirname "$model_path")" 2>/dev/null | head -10 || echo "  (cannot list)"
        fi
        echo ""
        echo -e "${YELLOW}💡 Common issues:${NC}"
        echo "  1. Model not downloaded yet - run 'hf download' first"
        echo "  2. Volume mount path mismatch - check docker -v path"
        echo "  3. Wrong model path - verify the exact directory name"
        echo ""
        echo -e "${YELLOW}💡 To download the model:${NC}"
        echo "  hf download meta-llama/Llama-3.2-1B-Instruct --local-dir $model_path"
        return 1
    fi
    
    # Verify model weights exist (TensorRT-LLM requirement)
    echo -e "${BLUE}📋 Verifying model files...${NC}"
    has_safetensors=false
    has_pytorch=false
    
    if ls "$model_path"/*.safetensors 1> /dev/null 2>&1; then
        has_safetensors=true
        echo "  ✅ Found .safetensors files"
    fi
    
    if ls "$model_path"/pytorch_model*.bin 1> /dev/null 2>&1; then
        has_pytorch=true
        echo "  ✅ Found pytorch_model files"
    fi
    
    if [ "$has_safetensors" = false ] && [ "$has_pytorch" = false ]; then
        echo -e "${RED}❌ No model weight files found!${NC}"
        echo ""
        echo "Model directory contents:"
        ls -lh "$model_path" | head -10
        echo ""
        echo -e "${YELLOW}💡 TensorRT-LLM requires model weights in one of these formats:${NC}"
        echo "  - .safetensors files (preferred)"
        echo "  - pytorch_model*.bin files"
        echo ""
        echo "The model may not have downloaded completely. Re-run the download."
        return 1
    fi
    echo ""
    
    # Create engine directory
    mkdir -p "$engine_dir"
    
    echo -e "${GREEN}🚀 Starting TensorRT-LLM build...${NC}"
    echo ""
    
    # Fix config.json if missing required fields (required by TensorRT-LLM)
    config_file="$model_path/config.json"
    if [ -f "$config_file" ]; then
        needs_fix=false
        if ! grep -q '"architecture"' "$config_file" 2>/dev/null; then
            needs_fix=true
        fi
        if ! grep -q '"dtype"' "$config_file" 2>/dev/null; then
            needs_fix=true
        fi
        
        if [ "$needs_fix" = true ]; then
            echo -e "${YELLOW}⚠️  Fixing config.json (adding required fields)...${NC}"
            python3 << EOF
import json

config_path = "$config_file"
try:
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    fixed = []
    if 'architecture' not in config:
        config['architecture'] = 'LlamaForCausalLM'
        fixed.append('architecture')
    
    if 'dtype' not in config:
        config['dtype'] = 'float16'
        fixed.append('dtype')
    
    if fixed:
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"✅ Added fields: {', '.join(fixed)}")
    else:
        print("✅ All required fields already exist")
except Exception as e:
    print(f"⚠️  Could not fix config: {e}")
EOF
        fi
        echo ""
    fi
    
    # Build TensorRT-LLM engine
    # Optimized for low latency (1-2s target)
    # Note: max_seq_len = max_input_len + generation length
    max_seq_len=$((context_window + 256))  # Input context + output generation
    
    echo -e "${BLUE}Building with max_seq_len=${max_seq_len} (input=${context_window} + generation=256)${NC}"
    echo ""
    
    # Show model directory structure for debugging
    echo -e "${BLUE}📁 Model directory structure:${NC}"
    ls -lh "$model_path" | grep -E "\.(safetensors|bin)$|config\.json" | head -5
    echo ""
    
    # TensorRT-LLM requires checkpoint format, not raw HuggingFace
    # Convert HuggingFace model to TensorRT-LLM checkpoint format first
    checkpoint_dir="$engine_dir/checkpoint"
    
    echo -e "${BLUE}🔧 Converting HuggingFace model to TensorRT-LLM checkpoint format...${NC}"
    echo ""
    
    # Check if checkpoint already exists and is valid (has been properly converted)
    # TensorRT-LLM may check both root and rank0/ locations
    checkpoint_marker="$checkpoint_dir/.tensorrt_llm_converted"
    checkpoint_weights="$checkpoint_dir/model.safetensors"
    checkpoint_config="$checkpoint_dir/config.json"
    
    # Also check for alternative weight file names
    if [ ! -f "$checkpoint_weights" ]; then
        checkpoint_weights="$checkpoint_dir/pytorch_model.bin"
    fi
    
    # Check rank0/ as well (TensorRT-LLM may prefer this location)
    rank0_weights="$checkpoint_dir/rank0/model.safetensors"
    if [ ! -f "$rank0_weights" ]; then
        rank0_weights="$checkpoint_dir/rank0/pytorch_model.bin"
    fi
    
    # Checkpoint is valid if it has weights (either in root or rank0/) and config.json
    weights_exist=false
    if [ -f "$checkpoint_weights" ] || [ -f "$rank0_weights" ]; then
        weights_exist=true
    fi
    
    if [ -d "$checkpoint_dir" ] && \
       [ -f "$checkpoint_config" ] && \
       [ "$weights_exist" = true ] && \
       [ -f "$checkpoint_marker" ]; then
        echo -e "${GREEN}✅ Checkpoint already exists and is properly converted, skipping conversion${NC}"
        echo ""
    else
        # Remove incomplete checkpoint if it exists
        if [ -d "$checkpoint_dir" ]; then
            echo -e "${YELLOW}⚠️  Removing incomplete checkpoint directory (will re-convert properly)...${NC}"
            rm -rf "$checkpoint_dir"
        fi
        echo -e "${BLUE}   Converting: $model_path → $checkpoint_dir${NC}"
        
        # CRITICAL: Disable exit on error BEFORE any operations that might fail
        # This prevents silent exits during conversion attempts
        # Use set +e explicitly (safest way, doesn't depend on command substitution)
        echo "   DEBUG: About to disable exit-on-error..."
        set +e || echo "   WARNING: set +e failed (this shouldn't happen)"
        echo "   DEBUG: Exit-on-error disabled (set +e) - status: $?"
        
        echo "   DEBUG: About to start conversion process..."
        
        # Verify we can create the directory
        echo "   DEBUG: Creating checkpoint directory..."
        mkdir -p "$checkpoint_dir" 2>&1
        mkdir_status=$?
        if [ $mkdir_status -ne 0 ]; then
            echo -e "${RED}❌ Failed to create checkpoint directory: $checkpoint_dir${NC}"
            echo "   Check permissions and disk space"
            # Restore set -e before returning
            set -e
            return 1
        fi
        echo "   ✅ Directory created: $checkpoint_dir"
        
        # Use TensorRT-LLM's conversion utility
        # Based on Jetson AI Lab: https://www.jetson-ai-lab.com/tensorrt_llm.html
        # Scripts are in /opt/TensorRT-LLM/examples/llama/
        conversion_success=false
        
        # set +e already done above, just ensure trap is set
        trap 'echo "ERROR: Conversion step failed at line $LINENO"' ERR
        
        echo -e "${BLUE}   Starting conversion process...${NC}"
        echo "   DEBUG: About to search for convert_checkpoint.py"
        echo "   DEBUG: Current directory: $(pwd)"
        echo "   DEBUG: Model path exists: $([ -d "$model_path" ] && echo "yes" || echo "no")"
        echo "   DEBUG: Checkpoint dir exists: $([ -d "$checkpoint_dir" ] && echo "yes" || echo "no")"
        
        # Method 1: Look for convert_checkpoint.py (pip-installed TensorRT-LLM)
        # With pip installation, scripts are typically in site-packages
        convert_script=""
        
        # Try to find it using Python - use explicit error handling
        echo -e "${BLUE}   Searching for convert_checkpoint.py...${NC}"
        
        # Use a temp file to capture output and avoid command substitution issues
        python_found_path=""
        
        # Run Python script to find conversion script
        python3 << 'PYEOF' > /tmp/find_convert_script.log 2>&1
import sys
try:
    import tensorrt_llm
    import os
    package_path = os.path.dirname(tensorrt_llm.__file__)
    # Check /opt/TensorRT-LLM first (Jetson AI Lab standard location)
    opt_path = "/opt/TensorRT-LLM/examples/llama/convert_checkpoint.py"
    if os.path.exists(opt_path):
        print(opt_path)
        sys.exit(0)
    
    # Check package installation locations
    convert_path = os.path.join(package_path, "models", "llama", "convert_checkpoint.py")
    if os.path.exists(convert_path):
        print(convert_path)
        sys.exit(0)
    # Also check examples
    convert_path = os.path.join(package_path, "examples", "llama", "convert_checkpoint.py")
    if os.path.exists(convert_path):
        print(convert_path)
        sys.exit(0)
except Exception as e:
    # Log the error but don't fail
    print(f"Search failed: {e}", file=sys.stderr)
    pass
sys.exit(1)
PYEOF
        
        python_search_status=$?
        if [ $python_search_status -eq 0 ]; then
            # Python script succeeded, read the result
            if [ -f /tmp/find_convert_script.log ]; then
                python_found_path=$(cat /tmp/find_convert_script.log | grep -v "Search failed" | head -1)
                if [ -n "$python_found_path" ] && [ -f "$python_found_path" ]; then
                    echo -e "${GREEN}   ✅ Found: $python_found_path${NC}"
                else
                    python_found_path=""
                fi
            fi
        else
            # Python script failed, that's okay - we'll use fallback
            echo -e "${YELLOW}   ⚠️  Could not search via Python import, checking common locations...${NC}"
            python_found_path=""
        fi
        
        if [ -n "$python_found_path" ] && [ -f "$python_found_path" ]; then
            convert_script="$python_found_path"
        else
            # Fallback: try common locations
            # Based on Jetson AI Lab docs: https://www.jetson-ai-lab.com/tensorrt_llm.html
            # Scripts are typically in /opt/TensorRT-LLM/examples/llama/
            echo -e "${BLUE}   Checking common locations...${NC}"
            possible_paths=(
                "/opt/TensorRT-LLM/examples/llama/convert_checkpoint.py"
                "/opt/TensorRT-LLM/examples/llama/convert_llama_weights_to_tensorrt_llm.py"
                "/usr/local/lib/python3.12/site-packages/tensorrt_llm/models/llama/convert_checkpoint.py"
                "/usr/local/lib/python3.11/site-packages/tensorrt_llm/models/llama/convert_checkpoint.py"
                "/usr/local/lib/python3.10/dist-packages/tensorrt_llm/models/llama/convert_checkpoint.py"
                "/usr/local/lib/python3.10/dist-packages/tensorrt_llm/examples/llama/convert_checkpoint.py"
                "/workspace/examples/llama/convert_checkpoint.py"
            )
            
            for path in "${possible_paths[@]}"; do
                if [ -f "$path" ]; then
                    convert_script="$path"
                    echo -e "${GREEN}   ✅ Found at: $path${NC}"
                    break
                fi
            done
            
            if [ -z "$convert_script" ]; then
                echo -e "${YELLOW}   ⚠️  Official conversion script not found, using transformers fallback${NC}"
            fi
        fi
        
        if [ -n "$convert_script" ]; then
            echo -e "${BLUE}   Attempting Method 1: Official convert_checkpoint.py${NC}"
            echo -e "${BLUE}   Using: $convert_script${NC}"
            echo ""
            echo "   DEBUG: About to execute: python3 $convert_script --model_dir $model_path --output_dir $checkpoint_dir --dtype float16"
            
            # Try running the conversion script
            # Based on Jetson AI Lab docs, scripts accept --model_dir and --output_dir
            # Note: May fail with Python 3.10 compatibility issues
            echo -e "${BLUE}   Running conversion script...${NC}"
            
            # Capture both stdout and stderr, and check exit code explicitly
            python3 "$convert_script" \
                --model_dir "$model_path" \
                --output_dir "$checkpoint_dir" \
                --dtype float16 \
                > /tmp/trtllm_convert.log 2>&1
            
            convert_exit_code=$?
            
            # Always show the log output
            echo "   Conversion script output:"
            cat /tmp/trtllm_convert.log
            echo ""
            
            if [ $convert_exit_code -eq 0 ]; then
                echo -e "${GREEN}   ✅ Conversion script executed successfully${NC}"
                # Verify checkpoint structure (weights and config in root, matching official TensorRT-LLM format)
                if [ -f "$checkpoint_dir/model.safetensors" ] || [ -f "$checkpoint_dir/model.bin" ] || [ -f "$checkpoint_dir/pytorch_model.bin" ]; then
                    if [ -f "$checkpoint_dir/config.json" ]; then
                        # Create marker file to indicate successful conversion
                        touch "$checkpoint_marker"
                        echo "Converted using convert_checkpoint.py" > "$checkpoint_marker"
                        conversion_success=true
                        echo -e "${GREEN}   ✅ Official conversion script succeeded${NC}"
                    else
                        echo -e "${YELLOW}   ⚠️  Conversion script ran but config.json not found, trying alternative...${NC}"
                    fi
                else
                    echo -e "${YELLOW}   ⚠️  Conversion script ran but weights not found, trying alternative...${NC}"
                fi
            else
                echo -e "${YELLOW}   ⚠️  convert_checkpoint.py failed with exit code $convert_exit_code${NC}"
                echo "   Full error log:"
                cat /tmp/trtllm_convert.log
                echo ""
                echo -e "${YELLOW}   Trying alternative conversion method...${NC}"
            fi
        fi
        
        # Method 2: Use transformers to re-save (ensures proper format, avoids tensorrt_llm import)
        if [ "$conversion_success" = false ]; then
            echo ""
            echo -e "${BLUE}   Attempting Method 2: Transformers fallback (avoids tensorrt_llm import issues)...${NC}"
            echo "   DEBUG: Running Python transformers conversion..."
            
            # Write Python script to temp file for better error handling
            cat > /tmp/trtllm_transformers_convert.py << 'PYEOF'
import os
import sys
import shutil
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_path = os.environ.get('MODEL_PATH', '')
checkpoint_dir = os.environ.get('CHECKPOINT_DIR', '')

print(f"DEBUG: model_path={model_path}, checkpoint_dir={checkpoint_dir}")

if not model_path or not checkpoint_dir:
    print("❌ Environment variables not set", file=sys.stderr)
    sys.exit(1)

try:
    print(f"Loading from: {model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    print(f"Saving to: {checkpoint_dir}")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Save model and tokenizer to checkpoint directory
    # TensorRT-LLM expects weights and config.json directly in checkpoint_dir (not in rank0/ subdirectory)
    # This matches the official llama.sh script structure
    model.save_pretrained(checkpoint_dir, safe_serialization=True)
    tokenizer.save_pretrained(checkpoint_dir)
    
    # Verify the checkpoint structure matches TensorRT-LLM expectations
    config_path = os.path.join(checkpoint_dir, "config.json")
    weights_path = os.path.join(checkpoint_dir, "model.safetensors")
    
    if not os.path.exists(weights_path):
        # Try alternative naming
        weights_path = os.path.join(checkpoint_dir, "pytorch_model.bin")
        if not os.path.exists(weights_path):
            print(f"❌ ERROR: No weights file found in checkpoint directory")
            sys.exit(1)
    
    if not os.path.exists(config_path):
        print(f"❌ ERROR: config.json not found in checkpoint directory")
        sys.exit(1)
    
    # Verify config.json is valid JSON
    import json
    try:
        with open(config_path, 'r') as f:
            config_data = json.load(f)
        config_size = os.path.getsize(config_path)
        print(f"✅ Verified config.json (size: {config_size} bytes, keys: {len(config_data)} fields)")
    except Exception as e:
        print(f"❌ ERROR: config.json is invalid: {e}")
        sys.exit(1)
    
    weights_size = os.path.getsize(weights_path)
    print(f"✅ Verified weights file: {os.path.basename(weights_path)} (size: {weights_size:,} bytes)")
    
    # TensorRT-LLM's from_checkpoint might expect rank-based structure even for single GPU
    # Create rank0/ subdirectory with weights and config (TensorRT-LLM may check both locations)
    rank0_dir = os.path.join(checkpoint_dir, "rank0")
    os.makedirs(rank0_dir, exist_ok=True)
    
    # Copy weights to rank0/ (TensorRT-LLM might look here first)
    rank0_weights = os.path.join(rank0_dir, os.path.basename(weights_path))
    if not os.path.exists(rank0_weights):
        print(f"Creating rank0/ structure for TensorRT-LLM compatibility...")
        shutil.copy2(weights_path, rank0_weights)
        print(f"✅ Copied weights to rank0/: {rank0_weights}")
    
    # Copy config.json to rank0/ as well
    rank0_config = os.path.join(rank0_dir, "config.json")
    if not os.path.exists(rank0_config):
        shutil.copy2(config_path, rank0_config)
        print(f"✅ Copied config.json to rank0/")
    
    print(f"✅ Checkpoint structure created: files in both root and rank0/ (TensorRT-LLM compatibility)")
    
    # TensorRT-LLM may also need tokenizer files in root
    # (already saved by tokenizer.save_pretrained above)
    
    # Create marker file to indicate successful conversion
    marker_file = os.path.join(checkpoint_dir, ".tensorrt_llm_converted")
    with open(marker_file, 'w') as f:
        f.write("Converted using transformers.save_pretrained() - files in both root and rank0/ for TensorRT-LLM compatibility\n")
    
    print("✅ Model re-saved successfully with TensorRT-LLM checkpoint structure")
    print(f"✅ Marker file created: {marker_file}")
    sys.exit(0)
except Exception as e:
    print(f"❌ Failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
PYEOF
            
            echo "   DEBUG: Executing transformers conversion script..."
            MODEL_PATH="$model_path" CHECKPOINT_DIR="$checkpoint_dir" python3 /tmp/trtllm_transformers_convert.py > /tmp/trtllm_transformers_convert.log 2>&1
            transformers_exit_code=$?
            
            # Always show the log output
            echo "   Transformers conversion output:"
            cat /tmp/trtllm_transformers_convert.log
            echo ""
            
            if [ $transformers_exit_code -eq 0 ]; then
                # Verify checkpoint structure (weights and config in root, matching official TensorRT-LLM format)
                if [ -f "$checkpoint_dir/model.safetensors" ] || [ -f "$checkpoint_dir/model.bin" ] || [ -f "$checkpoint_dir/pytorch_model.bin" ]; then
                    if [ -f "$checkpoint_dir/config.json" ]; then
                        touch "$checkpoint_marker"
                        echo "Converted using transformers.save_pretrained()" > "$checkpoint_marker"
                        conversion_success=true
                        echo -e "${GREEN}   ✅ Transformers conversion succeeded${NC}"
                    else
                        echo -e "${RED}   ❌ config.json not found in checkpoint directory${NC}"
                        conversion_success=false
                    fi
                else
                    echo -e "${RED}   ❌ Weights not found in checkpoint directory${NC}"
                    conversion_success=false
                fi
            else
                echo -e "${RED}   ❌ Transformers conversion failed with exit code $transformers_exit_code${NC}"
                echo "   Full error log:"
                cat /tmp/trtllm_transformers_convert.log
                conversion_success=false
            fi
        fi
        
        # Re-enable exit on error and remove trap
        trap - ERR
        set -e
        echo "   DEBUG: Exit-on-error re-enabled (set -e)"
        
        if [ "$conversion_success" = false ]; then
            echo ""
            echo -e "${RED}❌ Failed to convert HuggingFace model to checkpoint format${NC}"
            echo ""
            echo -e "${YELLOW}💡 Manual conversion steps:${NC}"
            echo "  1. Check TensorRT-LLM documentation for conversion tools"
            echo "  2. Verify model files are complete"
            echo "  3. Try using nvidia-modelopt to convert the model"
            return 1
        fi
        
        echo ""
        echo -e "${GREEN}✅ Conversion complete${NC}"
        echo ""
    fi
    
    # Build TensorRT-LLM engine from checkpoint
    echo -e "${BLUE}🔧 Building TensorRT-LLM engine from checkpoint...${NC}"
    echo ""
    
    # Verify checkpoint structure before building
    # TensorRT-LLM may check both root and rank0/ locations
    echo "   DEBUG: Verifying checkpoint structure..."
    echo "   DEBUG: Checking checkpoint_dir: $checkpoint_dir"
    
    # Check for weights in both locations (root first, then rank0/)
    weights_found=""
    config_found=""
    
    if [ -f "$checkpoint_dir/model.safetensors" ]; then
        weights_found="$checkpoint_dir/model.safetensors"
        echo "   ✅ Found weights in root: model.safetensors"
    elif [ -f "$checkpoint_dir/pytorch_model.bin" ]; then
        weights_found="$checkpoint_dir/pytorch_model.bin"
        echo "   ✅ Found weights in root: pytorch_model.bin"
    elif [ -f "$checkpoint_dir/rank0/model.safetensors" ]; then
        weights_found="$checkpoint_dir/rank0/model.safetensors"
        echo "   ✅ Found weights in rank0/: model.safetensors"
    elif [ -f "$checkpoint_dir/rank0/pytorch_model.bin" ]; then
        weights_found="$checkpoint_dir/rank0/pytorch_model.bin"
        echo "   ✅ Found weights in rank0/: pytorch_model.bin"
    else
        echo -e "${RED}   ❌ Weights not found in checkpoint directory (checked root and rank0/)${NC}"
        return 1
    fi
    
    # Check for config.json (prefer rank0/ if it exists, otherwise root)
    if [ -f "$checkpoint_dir/rank0/config.json" ]; then
        config_found="$checkpoint_dir/rank0/config.json"
        config_size=$(stat -f%z "$config_found" 2>/dev/null || stat -c%s "$config_found" 2>/dev/null || echo "unknown")
        echo "   ✅ Found config.json in rank0/ (size: $config_size bytes)"
    elif [ -f "$checkpoint_dir/config.json" ]; then
        config_found="$checkpoint_dir/config.json"
        config_size=$(stat -f%z "$config_found" 2>/dev/null || stat -c%s "$config_found" 2>/dev/null || echo "unknown")
        echo "   ✅ Found config.json in root (size: $config_size bytes)"
    else
        echo -e "${RED}   ❌ config.json not found in checkpoint directory (checked root and rank0/)${NC}"
        return 1
    fi
    
    echo "   ✅ Checkpoint structure verified"
    echo "   DEBUG: Using checkpoint_dir: $checkpoint_dir"
    echo "   DEBUG: TensorRT-LLM will look for weights in both root and rank0/ locations"
    echo ""
    
    # Build and capture both stdout/stderr and exit code
    set +e  # Don't exit on error
    trtllm-build \
        --checkpoint_dir "$checkpoint_dir" \
        --model_cls_name LlamaForCausalLM \
        --output_dir "$engine_dir" \
        --gemm_plugin float16 \
        --gpt_attention_plugin float16 \
        --context_fmha enable \
        --remove_input_padding enable \
        --max_batch_size 1 \
        --max_input_len $context_window \
        --max_seq_len $max_seq_len \
        --max_beam_width 1 \
        --builder_opt 3 \
        2>&1 | tee /tmp/trtllm_build.log
    build_exit_code=${PIPESTATUS[0]}  # Get trtllm-build exit code, not tee's
    set -e  # Re-enable exit on error
    
    # Check if build succeeded
    if [ $build_exit_code -eq 0 ]; then
        # Also verify engine files were created
        if [ -f "$engine_dir/rank0.engine" ] || [ -d "$engine_dir/rank0" ]; then
            echo ""
            echo -e "${GREEN}✅ Engine build succeeded!${NC}"
            return 0
        else
            echo ""
            echo -e "${YELLOW}⚠️  Build reported success but engine files not found${NC}"
            echo "Checking checkpoint directory structure..."
            ls -la "$checkpoint_dir" | head -10
            return 1
        fi
    else
        echo ""
        echo -e "${RED}❌ TensorRT-LLM engine build failed (exit code: $build_exit_code)${NC}"
        echo ""
        
        # Check for specific errors
        if grep -q "assert os.path.isfile(weights_path)" /tmp/trtllm_build.log 2>/dev/null; then
            echo -e "${YELLOW}💡 Error: TensorRT-LLM couldn't find weights file${NC}"
            echo ""
            echo "TensorRT-LLM expects weights in a specific checkpoint format."
            echo "The checkpoint directory should have model weights in the expected location."
            echo ""
            echo "Checkpoint directory contents:"
            ls -lh "$checkpoint_dir" | head -10
            echo ""
            echo "rank0/ subdirectory contents:"
            if [ -d "$checkpoint_dir/rank0" ]; then
                ls -lh "$checkpoint_dir/rank0"
            else
                echo "  ❌ rank0/ directory does not exist!"
            fi
            echo ""
            echo -e "${YELLOW}💡 Debugging: TensorRT-LLM may be looking for weights in a different path${NC}"
            echo "   Trying to find what TensorRT-LLM expects..."
            echo ""
            
            # Check common weight file patterns TensorRT-LLM might look for
            weight_patterns=(
                "$checkpoint_dir/rank0/model.safetensors"
                "$checkpoint_dir/rank0/model.bin"
                "$checkpoint_dir/rank0/pytorch_model.bin"
                "$checkpoint_dir/rank0/pytorch_model.safetensors"
                "$checkpoint_dir/model.safetensors"
                "$checkpoint_dir/model.bin"
            )
            
            echo "Checking for weight files in expected locations:"
            found_weights=false
            for pattern in "${weight_patterns[@]}"; do
                if [ -f "$pattern" ]; then
                    echo "  ✅ Found: $pattern ($(ls -lh "$pattern" | awk '{print $5}'))"
                    found_weights=true
                fi
            done
            
            if [ "$found_weights" = false ]; then
                echo "  ❌ No weight files found in expected locations"
            fi
            echo ""
            echo ""
            echo -e "${BLUE}🔍 Attempting to inspect TensorRT-LLM's expected weights path...${NC}"
            echo ""
            
            # Try to inspect what TensorRT-LLM is looking for by checking the source code pattern
            python3 << EOF
import os
checkpoint_dir = "$checkpoint_dir"

# Common patterns TensorRT-LLM might use to construct weights_path
# Based on TensorRT-LLM checkpoint format conventions
patterns_to_check = []

# Pattern 1: rank0/model.safetensors (current structure)
patterns_to_check.append(os.path.join(checkpoint_dir, "rank0", "model.safetensors"))

# Pattern 2: rank0/model.bin (alternate format)
patterns_to_check.append(os.path.join(checkpoint_dir, "rank0", "model.bin"))

# Pattern 3: rank0/pytorch_model.safetensors
patterns_to_check.append(os.path.join(checkpoint_dir, "rank0", "pytorch_model.safetensors"))

# Pattern 4: rank0/pytorch_model.bin
patterns_to_check.append(os.path.join(checkpoint_dir, "rank0", "pytorch_model.bin"))

# Pattern 5: Direct in checkpoint (without rank0/)
patterns_to_check.append(os.path.join(checkpoint_dir, "model.safetensors"))
patterns_to_check.append(os.path.join(checkpoint_dir, "model.bin"))

print("Expected paths TensorRT-LLM might be checking:")
for i, path in enumerate(patterns_to_check, 1):
    exists = os.path.isfile(path)
    status = "✅ EXISTS" if exists else "❌ NOT FOUND"
    print(f"  {i}. {path} - {status}")
    if exists:
        size = os.path.getsize(path) / (1024**3)  # Size in GB
        print(f"     Size: {size:.2f} GB")

print("\nActual checkpoint structure:")
if os.path.isdir(checkpoint_dir):
    print(f"  Root: {checkpoint_dir}")
    for item in os.listdir(checkpoint_dir):
        item_path = os.path.join(checkpoint_dir, item)
        if os.path.isdir(item_path):
            print(f"    📁 {item}/")
            if item == "rank0":
                for subitem in os.listdir(item_path):
                    subitem_path = os.path.join(item_path, subitem)
                    if os.path.isfile(subitem_path):
                        size = os.path.getsize(subitem_path)
                        if size > 1024**3:
                            size_str = f"{size/(1024**3):.2f} GB"
                        else:
                            size_str = f"{size/(1024**2):.2f} MB"
                        print(f"      📄 {subitem} ({size_str})")
        elif os.path.isfile(item_path):
            size = os.path.getsize(item_path)
            if size > 1024**2:
                size_str = f"{size/(1024**2):.2f} MB"
            else:
                size_str = f"{size/1024:.2f} KB"
            print(f"    📄 {item} ({size_str})")
EOF
            
            echo ""
            echo -e "${YELLOW}💡 Solution: The checkpoint may need to be re-converted using the official script${NC}"
            echo "   However, the official conversion script has Python 3.10 compatibility issues."
            echo "   Try one of these approaches:"
            echo ""
            echo "   1. Delete checkpoint and retry (script will use alternative conversion):"
            echo "      rm -rf $checkpoint_dir"
            echo ""
            echo "   2. Manually create symlink if TensorRT-LLM expects different naming:"
            echo "      Check the path patterns above and create symlinks if needed"
            echo ""
            echo "   3. Use a TensorRT-LLM container with Python 3.11+ for official conversion"
        else
            echo "Check the error messages above for details."
        fi
        return 1
    fi
}

# Main build function
build_engine() {
    local model_name=$1
    local model_path=$2
    local build_status=0
    
    # Temporarily disable exit-on-error for function calls
    # This allows us to check return codes explicitly
    set +e
    
    case "$model_name" in
        qwen3-4b|qwen3-4b-2507|qwen*)
            engine_dir="$TENSORRT_ENGINES_BASE/qwen3-4b-instruct-2507"
            build_qwen_engine "$model_name" "$model_path" "$engine_dir" 2048
            build_status=$?
            ;;
        llama-3.2-1b|llama3.2*)
            engine_dir="$TENSORRT_ENGINES_BASE/llama-3.2-1b-instruct"
            echo "DEBUG: About to call build_llama_engine"
            build_llama_engine "$model_name" "$model_path" "$engine_dir" 2048
            build_status=$?
            echo "DEBUG: build_llama_engine returned: $build_status"
            ;;
        llama-3.1-8b|llama-3.1-8b-instruct|llama3.1*)
            engine_dir="$TENSORRT_ENGINES_BASE/llama-3.1-8b-instruct"
            build_llama_engine "$model_name" "$model_path" "$engine_dir" 8192
            build_status=$?
            ;;
        qwen2.5-coder-7b|qwen2.5-coder-7b-instruct|qwen2.5*)
            engine_dir="$TENSORRT_ENGINES_BASE/qwen2.5-coder-7b-instruct"
            # Qwen2.5 uses same build process as Qwen but with longer context
            build_qwen_engine "$model_name" "$model_path" "$engine_dir" 32768
            build_status=$?
            ;;
        *)
            echo -e "${RED}❌ Unknown model: $model_name${NC}"
            echo -e "${YELLOW}Supported models:${NC}"
            echo "  - qwen3-4b-2507"
            echo "  - llama-3.2-1b"
            echo "  - llama-3.1-8b-instruct"
            echo "  - qwen2.5-coder-7b-instruct"
            build_status=1
            ;;
    esac
    
    # Restore exit-on-error
    set -e
    
    # Return the build status
    if [ $build_status -ne 0 ]; then
        echo -e "${RED}❌ Engine build failed with status: $build_status${NC}"
        return $build_status
    fi
    
    return 0
}

# Check if running in TensorRT-LLM container or has trtllm-build
if ! command -v trtllm-build &> /dev/null; then
    echo -e "${YELLOW}⚠️  trtllm-build not found in PATH${NC}"
    echo -e "${YELLOW}💡 This script should be run inside the TensorRT-LLM container:${NC}"
    echo ""
    echo "  docker run --rm -it --gpus all \\"
    echo "    -v /path/to/models:/models \\"
    echo "    -v \$(pwd)/scripts:/scripts \\"
    echo "    dustynv/tensorrt_llm:0.12-r36.4.0 \\"
    echo "    bash /scripts/build_tensorrt_engine.sh"
    echo ""
    exit 1
fi

# Parse arguments
if [ $# -eq 0 ]; then
    print_header "TensorRT-LLM Engine Builder"
    
    echo "Usage:"
    echo "  $0 <model_name> <model_path>"
    echo ""
    echo "Examples:"
    echo "  $0 qwen3-4b-2507 /models/Qwen/Qwen3-4B-Instruct"
    echo "  $0 llama-3.2-1b /models/Llama/Llama-3.2-1B-Instruct"
    echo ""
    echo "Or set environment variables:"
    echo "  MODEL_NAME=qwen3-4b-2507"
    echo "  MODEL_PATH=/models/Qwen/Qwen3-4B-Instruct"
    echo "  TENSORRT_ENGINES_BASE=/models/tensorrt-llm"
    echo ""
    
    # Try to use environment variables
    if [ -n "$MODEL_NAME" ] && [ -n "$MODEL_PATH" ]; then
        echo -e "${BLUE}Using environment variables:${NC}"
        echo "  MODEL_NAME=$MODEL_NAME"
        echo "  MODEL_PATH=$MODEL_PATH"
        echo ""
        build_engine "$MODEL_NAME" "$MODEL_PATH"
    else
        echo -e "${YELLOW}No model specified. Please provide model name and path.${NC}"
        exit 1
    fi
else
    MODEL_NAME=${1:-$MODEL_NAME}
    MODEL_PATH=${2:-$MODEL_PATH}
    
    if [ -z "$MODEL_PATH" ]; then
        echo -e "${RED}❌ Model path not provided${NC}"
        echo "Usage: $0 <model_name> <model_path>"
        exit 1
    fi
    
    build_engine "$MODEL_NAME" "$MODEL_PATH"
fi

print_header "Build Complete!"
echo -e "${GREEN}✅ TensorRT-LLM engine is ready${NC}"
echo ""
echo "Next steps:"
echo "1. Ensure the engine directory is mounted in docker-compose.yml"
echo "2. Set TENSORRT_ENGINES_BASE or TENSORRT_ENGINE_DIR in .env"
echo "3. Restart the container"


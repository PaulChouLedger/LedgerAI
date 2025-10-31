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
    
    # TensorRT-LLM 0.12 supports --hf_model_dir for HuggingFace models
    # Try this first, then fall back to --checkpoint_dir if needed
    echo -e "${BLUE}🔧 Attempting build with Llama model class...${NC}"
    echo ""
    
    build_success=false
    
    # Attempt 1: Use --hf_model_dir (TensorRT-LLM 0.12+ supports HuggingFace directly)
    echo -e "${BLUE}   Attempt 1: Using --hf_model_dir (HuggingFace format)...${NC}"
    if trtllm-build \
        --hf_model_dir "$model_path" \
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
        2>&1 | tee /tmp/trtllm_build_attempt1.log; then
        echo ""
        echo -e "${GREEN}✅ Build succeeded with --hf_model_dir!${NC}"
        build_success=true
    fi
    
    # Attempt 2: Use --checkpoint_dir with model_cls_name (if Attempt 1 failed)
    if [ "$build_success" = false ]; then
        echo ""
        echo -e "${BLUE}   Attempt 2: Using --checkpoint_dir with model class...${NC}"
        if trtllm-build \
            --checkpoint_dir "$model_path" \
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
            2>&1 | tee /tmp/trtllm_build_attempt2.log; then
            echo ""
            echo -e "${GREEN}✅ Build succeeded with --checkpoint_dir!${NC}"
            build_success=true
        fi
    fi
    
    if [ "$build_success" = true ]; then
        return 0
    fi
    
    # Both attempts failed - provide diagnostic info
    echo ""
    echo -e "${YELLOW}⚠️  Both build attempts failed${NC}"
    echo ""
    
    # Check for weights_path error
    if grep -q "assert os.path.isfile(weights_path)" /tmp/trtllm_build_attempt*.log 2>/dev/null; then
        echo -e "${YELLOW}💡 Error: TensorRT-LLM couldn't find weights in expected format${NC}"
        echo ""
        echo "This TensorRT-LLM version may require converting HuggingFace models first."
        echo "Try using the convert_checkpoint.py script:"
        echo ""
        echo "  python3 /usr/local/lib/python3.10/dist-packages/tensorrt_llm/models/llama/convert_checkpoint.py \\"
        echo "    --model_dir $model_path \\"
        echo "    --output_dir $engine_dir/checkpoint \\"
        echo "    --dtype float16"
        echo ""
        echo "Then rebuild with: --checkpoint_dir $engine_dir/checkpoint"
    else
        echo -e "${RED}❌ TensorRT-LLM build failed${NC}"
        echo "Check the error messages above for details."
    fi
    
    return 1
}

# Main build function
build_engine() {
    local model_name=$1
    local model_path=$2
    
    case "$model_name" in
        qwen3-4b|qwen3-4b-2507|qwen*)
            engine_dir="$TENSORRT_ENGINES_BASE/qwen3-4b-instruct-2507"
            build_qwen_engine "$model_name" "$model_path" "$engine_dir" 2048
            ;;
        llama-3.2-1b|llama3.2*)
            engine_dir="$TENSORRT_ENGINES_BASE/llama-3.2-1b-instruct"
            build_llama_engine "$model_name" "$model_path" "$engine_dir" 2048
            ;;
        llama-3.1-8b|llama-3.1-8b-instruct|llama3.1*)
            engine_dir="$TENSORRT_ENGINES_BASE/llama-3.1-8b-instruct"
            build_llama_engine "$model_name" "$model_path" "$engine_dir" 8192
            ;;
        qwen2.5-coder-7b|qwen2.5-coder-7b-instruct|qwen2.5*)
            engine_dir="$TENSORRT_ENGINES_BASE/qwen2.5-coder-7b-instruct"
            # Qwen2.5 uses same build process as Qwen but with longer context
            build_qwen_engine "$model_name" "$model_path" "$engine_dir" 32768
            ;;
        *)
            echo -e "${RED}❌ Unknown model: $model_name${NC}"
            echo -e "${YELLOW}Supported models:${NC}"
            echo "  - qwen3-4b-2507"
            echo "  - llama-3.2-1b"
            echo "  - llama-3.1-8b-instruct"
            echo "  - qwen2.5-coder-7b-instruct"
            return 1
            ;;
    esac
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


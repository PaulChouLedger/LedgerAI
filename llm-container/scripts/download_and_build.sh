#!/bin/bash
# ============================================================================
# Complete workflow: Download model and build TensorRT-LLM engine
# ============================================================================

set -e

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Default paths (relative to script location)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LEDGERAI_DIR="$(dirname "$SCRIPT_DIR")"
MODELS_DIR="${MODELS_DIR:-$LEDGERAI_DIR/models}"

MODEL_NAME="${MODEL_NAME:-llama-3.2-1b}"
HF_MODEL_NAME="${HF_MODEL_NAME:-meta-llama/Llama-3.2-1B-Instruct}"

print_header() {
    echo ""
    echo "========================================================================"
    echo "   $1"
    echo "========================================================================"
    echo ""
}

# Step 1: Download model
download_model() {
    print_header "Step 1: Downloading Model from HuggingFace"
    
    local model_dir="$MODELS_DIR/Llama/Llama-3.2-1B-Instruct"
    
    echo -e "${BLUE}Model:${NC} $HF_MODEL_NAME"
    echo -e "${BLUE}Download to:${NC} $model_dir"
    echo ""
    
    # Check if already downloaded
    if [ -d "$model_dir" ] && [ -f "$model_dir/config.json" ]; then
        echo -e "${GREEN}✅ Model already downloaded${NC}"
        return 0
    fi
    
    # Create directory
    mkdir -p "$model_dir"
    
    echo -e "${YELLOW}📥 Downloading model (this may take a while)...${NC}"
    echo ""
    
    # Check if hf command exists
    if ! command -v hf &> /dev/null; then
        echo -e "${RED}❌ 'hf' command not found${NC}"
        echo ""
        echo "Install it with:"
        echo "  pip install huggingface-hub"
        echo ""
        echo "Or use huggingface-cli (deprecated but works):"
        echo "  pip install huggingface-cli"
        return 1
    fi
    
    # Download model
    hf download "$HF_MODEL_NAME" \
        --local-dir "$model_dir" \
        --local-dir-use-symlinks False \
        || {
            echo ""
            echo -e "${RED}❌ Download failed${NC}"
            echo ""
            echo -e "${YELLOW}💡 Troubleshooting:${NC}"
            echo "  1. Ensure you're logged in: hf login"
            echo "  2. Request access at: https://huggingface.co/$HF_MODEL_NAME"
            echo "  3. Check your internet connection"
            return 1
        }
    
    echo ""
    echo -e "${GREEN}✅ Model downloaded successfully${NC}"
}

# Step 2: Build engine
build_engine() {
    print_header "Step 2: Building TensorRT-LLM Engine"
    
    local model_path="$MODELS_DIR/Llama/Llama-3.2-1B-Instruct"
    local script_path="$SCRIPT_DIR/build_tensorrt_engine.sh"
    
    echo -e "${BLUE}Model path:${NC} $model_path"
    echo -e "${BLUE}Script path:${NC} $script_path"
    echo ""
    
    # Verify model exists
    if [ ! -d "$model_path" ] || [ ! -f "$model_path/config.json" ]; then
        echo -e "${RED}❌ Model not found at: $model_path${NC}"
        echo "Please download the model first (run this script or download manually)"
        return 1
    fi
    
    # Verify script exists
    if [ ! -f "$script_path" ]; then
        echo -e "${RED}❌ Build script not found: $script_path${NC}"
        return 1
    fi
    
    echo -e "${YELLOW}🚀 Starting Docker build (this may take 10-30 minutes)...${NC}"
    echo ""
    
    # Build using docker
    docker run --rm -it --gpus all \
        -v "$MODELS_DIR:/models" \
        -v "$SCRIPT_DIR:/scripts" \
        dustynv/tensorrt_llm:0.12-r36.4.0 \
        bash /scripts/build_tensorrt_engine.sh "$MODEL_NAME" "$model_path"
}

# Main workflow
main() {
    print_header "TensorRT-LLM Model Download & Build"
    
    echo -e "${BLUE}Configuration:${NC}"
    echo "  Models directory: $MODELS_DIR"
    echo "  Model: $HF_MODEL_NAME"
    echo "  Build target: $MODEL_NAME"
    echo ""
    
    # Check if user wants to skip download
    if [ "$1" == "--skip-download" ]; then
        echo -e "${YELLOW}Skipping download (using existing model)${NC}"
        echo ""
    else
        # Step 1: Download
        download_model || exit 1
        echo ""
    fi
    
    # Step 2: Build
    build_engine || exit 1
    
    print_header "Complete!"
    echo -e "${GREEN}✅ Model downloaded and engine built successfully${NC}"
    echo ""
    echo "Next steps:"
    echo "1. Update .env with: SIMPLE_MODEL_NAME=llama-3.2-1b"
    echo "2. Mount the engine directory in docker-compose.yml"
    echo "3. Restart the container"
    echo ""
}

# Run main
main "$@"


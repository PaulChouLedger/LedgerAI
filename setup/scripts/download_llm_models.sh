#!/bin/bash
# Standalone script to download LLM models for containers
# Usage: bash download_llm_models.sh

set -e  # Exit on error

echo "=========================================="
echo "  LLM Models Download"
echo "=========================================="
echo ""

# Detect user (works even when run via sudo)
AURA_USER="${SUDO_USER:-$USER}"
AURA_HOME="/home/$AURA_USER"
LEDGERAI_DIR="$AURA_HOME/LedgerAI"

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

print_info() {
    echo -e "${YELLOW}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if LedgerAI directory exists
if [ ! -d "$LEDGERAI_DIR" ]; then
    print_error "LedgerAI directory not found at $LEDGERAI_DIR"
    exit 1
fi

# Create models directories if they don't exist
print_info "Creating models directories..."
mkdir -p "$LEDGERAI_DIR/llm-container/models" 2>/dev/null || true
mkdir -p "$LEDGERAI_DIR/llm-medical-container/models" 2>/dev/null || true
print_success "Models directories ready"

# Download Qwen2.5 model for generic LLM container
GENERIC_MODEL_PATH="$LEDGERAI_DIR/llm-container/models/Qwen2.5-1.5B-Instruct.Q4_K_M.gguf"
if [ -f "$GENERIC_MODEL_PATH" ]; then
    print_success "Generic LLM model already exists: $GENERIC_MODEL_PATH"
    print_info "   File size: $(du -h "$GENERIC_MODEL_PATH" | cut -f1)"
else
    print_info "Downloading Qwen2.5-1.5B-Instruct model for generic LLM container..."
    print_info "   Source: HuggingFace (RichardErkhov/unsloth_-_Qwen2.5-1.5B-Instruct-gguf)"
    print_info "   This is a large file (~1GB) - may take several minutes depending on connection..."
    echo ""
    
    if wget -q --show-progress -O "$GENERIC_MODEL_PATH" https://huggingface.co/RichardErkhov/unsloth_-_Qwen2.5-1.5B-Instruct-gguf/resolve/main/Qwen2.5-1.5B-Instruct.Q4_K_M.gguf; then
        print_success "Generic LLM model downloaded successfully"
        print_info "   Location: $GENERIC_MODEL_PATH"
        print_info "   File size: $(du -h "$GENERIC_MODEL_PATH" | cut -f1)"
    else
        print_error "Failed to download generic LLM model"
        print_info "   You can download manually:"
        print_info "   cd $LEDGERAI_DIR/llm-container/models"
        print_info "   wget https://huggingface.co/RichardErkhov/unsloth_-_Qwen2.5-1.5B-Instruct-gguf/resolve/main/Qwen2.5-1.5B-Instruct.Q4_K_M.gguf"
        print_info "   Note: Docker build will fail if model is missing"
        exit 1
    fi
fi

echo ""
print_success "✅ LLM model download complete!"
print_info ""
print_info "Next steps:"
print_info "   1. Build Docker containers: cd $LEDGERAI_DIR/setup && docker compose build"
print_info "   2. Start containers: docker compose up -d"


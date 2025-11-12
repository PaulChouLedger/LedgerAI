#!/bin/bash
# Setup script for Unsloth fine-tuning on Jetson
# This script installs all required dependencies

set -e

echo "🚀 Setting up Unsloth for fine-tuning on Jetson..."
echo ""

# Check if running on Jetson
if [ ! -f /etc/nv_tegra_release ]; then
    echo "⚠️  Warning: This script is designed for Jetson devices"
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}' | cut -d. -f1,2)
echo "📦 Python version: $PYTHON_VERSION"

# Check CUDA
if command -v nvidia-smi &> /dev/null; then
    echo "✅ NVIDIA GPU detected"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
else
    echo "⚠️  Warning: nvidia-smi not found. CUDA may not be available."
fi

echo ""
echo "📥 Installing Unsloth from Jetson AI Lab PyPI..."

# Install Unsloth from Jetson AI Lab
pip3 install --index-url https://pypi.jetson-ai-lab.io/jp6/cu129 \
    unsloth-2025.7.9-py3-none-any.whl || {
    echo "❌ Failed to install unsloth wheel. Trying direct install..."
    pip3 install --index-url https://pypi.jetson-ai-lab.io/jp6/cu129 unsloth
}

# Install unsloth_zoo (required dependency)
echo ""
echo "📥 Installing unsloth_zoo..."
pip3 install unsloth_zoo

echo ""
echo "📥 Installing additional dependencies..."

# Install PyTorch (adjust URL based on your CUDA version)
# For CUDA 12.1 (common on Jetson)
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 || {
    echo "⚠️  PyTorch installation failed. You may need to install manually."
}

# Install transformers and related packages
# Note: Using specific versions to avoid compatibility issues with unsloth
# transformers 4.40.0-4.45.x is compatible (avoid 4.46+ which removed top_k_top_p_filtering)
pip3 install "transformers>=4.40.0,<4.46.0" \
    datasets>=2.14.0 \
    "trl>=0.7.0,<0.8.0" \
    peft>=0.8.0 \
    accelerate>=0.27.0 \
    bitsandbytes>=0.41.0 \
    scipy \
    sentencepiece \
    unsloth_zoo

echo ""
echo "✅ Verifying installation..."

# Test imports
python3 -c "
import sys
try:
    # Try importing unsloth - may have patching errors but should still work for SFT
    from unsloth import FastLanguageModel
    print('✅ Unsloth imported successfully')
except (ImportError, IndexError, AttributeError) as e:
    error_msg = str(e)
    if 'IndexError' in str(type(e).__name__) or 'list index out of range' in error_msg:
        print('⚠️  Unsloth patching encountered an error (this may be non-critical for SFT)')
        print('   Attempting to continue anyway...')
        try:
            # Try to import again - sometimes the patching error doesn't prevent usage
            from unsloth import FastLanguageModel
            print('✅ Unsloth import succeeded on retry')
        except Exception as e2:
            print(f'❌ Unsloth import failed: {e2}')
            error_msg = str(e2)
            if 'top_k_top_p_filtering' in error_msg:
                print('   This is a transformers version compatibility issue.')
                print('   Solution: pip install "transformers>=4.40.0,<4.46.0"')
            else:
                print('   This may be a compatibility issue. Try:')
                print('   1. pip install "transformers>=4.40.0,<4.46.0"')
                print('   2. pip install "trl>=0.7.0,<0.8.0"')
                print('   3. pip install --upgrade unsloth')
            sys.exit(1)
    else:
        print(f'❌ Unsloth import failed: {e}')
        sys.exit(1)

try:
    import torch
    print(f'✅ PyTorch {torch.__version__} imported successfully')
    if torch.cuda.is_available():
        print(f'✅ CUDA available: {torch.cuda.get_device_name(0)}')
    else:
        print('⚠️  CUDA not available')
except ImportError as e:
    print(f'❌ PyTorch import failed: {e}')
    exit(1)

try:
    from transformers import AutoTokenizer
    print('✅ Transformers imported successfully')
except ImportError as e:
    print(f'❌ Transformers import failed: {e}')
    exit(1)
"

if [ $? -eq 0 ]; then
    echo ""
    echo "🎉 Setup completed successfully!"
    echo ""
    echo "Next steps:"
    echo "1. Ensure your dataset is ready: medical_sft_dataset.json (in LLM_tuning directory)"
    echo "2. Run fine-tuning from LLM_tuning directory:"
    echo "   cd LLM_tuning"
    echo "   python3 finetune_unsloth.py --model_name unsloth/Llama-3.2-1B-Instruct-bnb-4bit"
    echo ""
else
    echo ""
    echo "❌ Setup failed. Please check the errors above."
    exit 1
fi


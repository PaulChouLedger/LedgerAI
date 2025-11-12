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
echo "🧹 Cleaning up existing unsloth installations (if any)..."

# Uninstall existing unsloth and unsloth_zoo to ensure clean installation
pip3 uninstall -y unsloth unsloth_zoo 2>/dev/null || {
    echo "   (No existing installations found - this is fine)"
}

echo ""
echo "📥 Installing Unsloth from Jetson AI Lab PyPI (cu126)..."
echo "   (Installing unsloth package - pip will automatically install all dependencies including unsloth_zoo)"

# Install Unsloth from the cu126 index
# Note: Use package name, not wheel filename - pip will find the wheel automatically
pip3 install --index-url https://pypi.jetson-ai-lab.io/jp6/cu126 \
    unsloth==2025.7.9 || {
    echo "⚠️  Version-specific install failed, trying latest version..."
    pip3 install --index-url https://pypi.jetson-ai-lab.io/jp6/cu126 unsloth
}

echo ""
echo "📥 Installing PyTorch..."

# Install PyTorch (adjust URL based on your CUDA version)
# For CUDA 12.1 (common on Jetson)
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 || {
    echo "⚠️  PyTorch installation failed. You may need to install manually."
}

echo ""
echo "📥 Installing additional dependencies with version constraints..."

# Install other dependencies with version constraints to avoid compatibility issues
# Note: The unsloth wheel may have installed some of these, but we constrain versions
# to avoid known issues (transformers 4.46+ removed top_k_top_p_filtering, trl 0.8+ has patching issues)
pip3 install "transformers>=4.40.0,<4.46.0" \
    datasets>=2.14.0 \
    "trl>=0.7.0,<0.8.0" \
    peft>=0.8.0 \
    accelerate>=0.27.0 \
    bitsandbytes>=0.41.0 \
    scipy \
    sentencepiece

echo ""
echo "✅ Verifying installation..."

# Test imports
python3 -c "
import sys
try:
    # Try importing unsloth - may have patching errors but should still work for SFT
    from unsloth import FastLanguageModel
    print('✅ Unsloth imported successfully')
except (ImportError, IndexError, AttributeError, ModuleNotFoundError) as e:
    error_msg = str(e)
    if 'unsloth_zoo' in error_msg or ('No module named' in error_msg and 'unsloth_zoo' in error_msg):
        print('❌ unsloth_zoo is not properly installed.')
        print('   The package should have installed unsloth_zoo automatically. Try reinstalling:')
        print('   pip install --force-reinstall --no-cache-dir --index-url https://pypi.jetson-ai-lab.io/jp6/cu126 unsloth==2025.7.9')
        print('   Or install manually: pip install unsloth_zoo')
        sys.exit(1)
    elif 'IndexError' in str(type(e).__name__) or 'list index out of range' in error_msg:
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
                print('   Solution: pip install transformers==4.45.2')
            else:
                print('   This may be a compatibility issue. Try:')
                print('   1. pip install transformers==4.45.2')
                print('   2. pip install trl==0.7.11')
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


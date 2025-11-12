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

# Check transformers version - will force reinstall with compatible version later
echo ""
echo "🧹 Checking transformers version..."
CURRENT_TRANSFORMERS=$(pip3 show transformers 2>/dev/null | grep "^Version:" | awk '{print $2}' || echo "not installed")
if [ "$CURRENT_TRANSFORMERS" != "not installed" ]; then
    echo "   Current version: $CURRENT_TRANSFORMERS"
    echo "   (Will ensure compatible version <4.46.0 is installed)"
fi

echo ""
echo "📥 Installing PyTorch..."

# Install PyTorch (adjust URL based on your CUDA version)
# For CUDA 12.1 (common on Jetson)
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 || {
    echo "⚠️  PyTorch installation failed. You may need to install manually."
}

echo ""
echo "📥 Installing transformers with compatible version FIRST..."

# CRITICAL: Install transformers FIRST with compatible version (4.46+ removed top_k_top_p_filtering)
# This must be done before unsloth/unsloth_zoo to avoid dependency conflicts
echo "   Installing compatible transformers version (4.40.0-4.45.x)..."
pip3 install --force-reinstall --no-cache-dir "transformers>=4.40.0,<4.46.0" || {
    echo "   ⚠️  Version range install failed, trying specific version..."
    pip3 install --force-reinstall --no-cache-dir transformers==4.45.2
}

echo ""
echo "📥 Installing trl with compatible version..."

# Install trl with compatible version BEFORE unsloth
pip3 install --force-reinstall --no-cache-dir "trl>=0.7.0,<0.8.0" || {
    echo "   ⚠️  Version range install failed, trying specific version..."
    pip3 install --force-reinstall --no-cache-dir trl==0.7.11
}

echo ""
echo "📥 Installing Unsloth from Jetson AI Lab PyPI (cu126)..."
echo "   (Installing unsloth package, then unsloth_zoo with compatible version)"

# Install Unsloth from the cu126 index
# Note: Use package name, not wheel filename - pip will find the wheel automatically
pip3 install --index-url https://pypi.jetson-ai-lab.io/jp6/cu126 \
    unsloth==2025.7.9 || {
    echo "⚠️  Version-specific install failed, trying latest version..."
    pip3 install --index-url https://pypi.jetson-ai-lab.io/jp6/cu126 unsloth
}

# Check if unsloth_zoo is needed (it may be included in unsloth 2025.7.9)
# If not, we'll install it but need to be careful about version conflicts
echo ""
echo "📥 Checking if unsloth_zoo is needed..."
python3 -c "import unsloth_zoo" 2>/dev/null && {
    echo "   ✅ unsloth_zoo is already available"
} || {
    echo "   Installing unsloth_zoo (skipping dependency resolution to avoid conflicts)..."
    # Try installing without dependencies first to avoid pulling incompatible transformers
    pip3 install --no-deps unsloth_zoo 2>/dev/null || {
        echo "   ⚠️  Could not install unsloth_zoo without dependencies"
        echo "   Will try with transformers constraint..."
        # Create a temporary constraints file
        echo "transformers<4.46.0" > /tmp/transformers_constraint.txt
        pip3 install --constraint /tmp/transformers_constraint.txt unsloth_zoo 2>/dev/null || {
            echo "   ⚠️  unsloth_zoo installation failed - may need manual installation"
            echo "   You can try: pip3 install --no-deps unsloth_zoo"
        }
        rm -f /tmp/transformers_constraint.txt
    }
}

echo ""
echo "📥 Installing additional dependencies..."

# Install other dependencies with version constraints
echo "   Installing other dependencies..."
pip3 install datasets>=2.14.0 \
    peft>=0.8.0 \
    accelerate>=0.27.0 \
    bitsandbytes>=0.41.0 \
    scipy \
    sentencepiece \
    "fsspec>=2023.1.0,<=2025.9.0" || {
    echo "   ⚠️  Some dependencies failed to install, but continuing..."
}

echo ""
echo "🧹 Clearing Python cache to ensure fresh imports..."
# Find site-packages directory dynamically
SITE_PACKAGES=$(python3 -c "import site; print(site.getsitepackages()[0])" 2>/dev/null || echo "")
if [ -n "$SITE_PACKAGES" ] && [ -d "$SITE_PACKAGES" ]; then
    find "$SITE_PACKAGES" -type d -name __pycache__ -exec rm -r {} + 2>/dev/null || true
    find "$SITE_PACKAGES" -name "*.pyc" -delete 2>/dev/null || true
    echo "   Cleared cache in $SITE_PACKAGES"
else
    echo "   Could not locate site-packages, skipping cache clear"
fi

echo ""
echo "✅ Verifying installation..."

# Verify transformers version first
echo "   Checking transformers version..."
TRANSFORMERS_VERSION=$(python3 -c "import transformers; print(transformers.__version__)" 2>/dev/null || echo "not installed")
echo "   Transformers version: $TRANSFORMERS_VERSION"
if [ "$TRANSFORMERS_VERSION" != "not installed" ]; then
    # Check if version is 4.46 or higher
    if python3 -c "from packaging import version; import sys; v='$TRANSFORMERS_VERSION'; sys.exit(0 if version.parse(v) < version.parse('4.46.0') else 1)" 2>/dev/null; then
        echo "   ✅ Transformers version is compatible"
    else
        echo "   ❌ Transformers version $TRANSFORMERS_VERSION is incompatible!"
        echo "   Attempting to fix..."
        pip3 install --force-reinstall --no-cache-dir transformers==4.45.2
    fi
fi

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
                print('   Current transformers version is incompatible (4.46+).')
                print('   Solution: pip install --force-reinstall --no-cache-dir transformers==4.45.2')
                print('   Or: pip install --force-reinstall --no-cache-dir \"transformers>=4.40.0,<4.46.0\"')
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


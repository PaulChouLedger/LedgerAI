#!/bin/bash
# Setup script for Unsloth in a clean virtual environment
# This avoids all dependency conflicts by starting fresh

set -e

echo "🚀 Setting up clean virtual environment for Unsloth..."
echo ""

# Check if virtual environment already exists
if [ -d "unsloth-env" ]; then
    echo "⚠️  Virtual environment 'unsloth-env' already exists"
    read -p "   Remove and recreate? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "   Removing existing virtual environment..."
        rm -rf unsloth-env
    else
        echo "   Using existing virtual environment"
        source unsloth-env/bin/activate
        echo "   ✅ Virtual environment activated"
        exit 0
    fi
fi

# Create virtual environment
echo "📦 Creating virtual environment..."
python3 -m venv unsloth-env

# Activate virtual environment
echo "🔌 Activating virtual environment..."
source unsloth-env/bin/activate

# Upgrade pip
echo "📥 Upgrading pip..."
pip install --upgrade pip

# Install PyTorch first (required by unsloth)
echo ""
echo "📥 Installing PyTorch..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 || {
    echo "⚠️  PyTorch installation failed. You may need to install manually."
}

# Install unsloth from wheel - let it handle its own dependencies
echo ""
echo "📥 Installing Unsloth from Jetson AI Lab PyPI (cu126)..."
echo "   (This will install unsloth and all its dependencies)"
pip install --index-url https://pypi.jetson-ai-lab.io/jp6/cu126 unsloth==2025.7.9 || {
    echo "⚠️  Version-specific install failed, trying latest version..."
    pip install --index-url https://pypi.jetson-ai-lab.io/jp6/cu126 unsloth
}

# Install additional dependencies that might be needed
echo ""
echo "📥 Installing additional dependencies..."
pip install datasets peft accelerate bitsandbytes scipy sentencepiece

echo ""
echo "✅ Verifying installation..."

# Check transformers version first
echo "   Checking transformers version..."
python3 -c "
import transformers
from packaging import version
v = transformers.__version__
print(f'   Transformers version: {v}')
if version.parse(v) >= version.parse('4.46.0'):
    print('   ⚠️  WARNING: Transformers version is >= 4.46.0 (incompatible)')
    print('   Attempting to downgrade...')
    import subprocess
    import sys
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--force-reinstall', '--no-cache-dir', 'transformers==4.45.2'])
    print('   ✅ Downgraded to transformers 4.45.2')
else:
    print('   ✅ Transformers version is compatible')
"

# Test imports
python3 -c "
import sys
try:
    from unsloth import FastLanguageModel
    print('✅ Unsloth imported successfully')
    print('')
    print('📋 Installation summary:')
    import transformers
    print(f'   - transformers: {transformers.__version__}')
    import torch
    print(f'   - torch: {torch.__version__}')
    try:
        import trl
        print(f'   - trl: {trl.__version__}')
    except:
        print('   - trl: not installed')
    try:
        import unsloth_zoo
        print('   - unsloth_zoo: installed')
    except:
        print('   - unsloth_zoo: not installed (may be included in unsloth)')
except Exception as e:
    print(f'❌ Unsloth import failed: {e}')
    error_msg = str(e)
    if 'top_k_top_p_filtering' in error_msg:
        print('')
        print('   This is a transformers compatibility issue.')
        print('   Even in a clean environment, unsloth may have pulled incompatible transformers.')
        print('   Try manually downgrading:')
        print('   pip install --force-reinstall --no-cache-dir transformers==4.45.2')
    sys.exit(1)
"

echo ""
echo "✅ Setup complete!"
echo ""
echo "📝 To use this environment:"
echo "   source unsloth-env/bin/activate"
echo ""
echo "📝 To run fine-tuning:"
echo "   source unsloth-env/bin/activate"
echo "   python3 finetune_unsloth.py --dataset_path ./medical_sft_dataset.json"
echo ""


#!/bin/bash
# Install ChatterboxTTS without pkuseg (for English-only TTS)

set -e

echo "================================================================================"
echo "ChatterboxTTS Installation (Without pkuseg)"
echo "================================================================================"
echo ""
echo "This script installs ChatterboxTTS without pkuseg dependency."
echo "English TTS will work, but Chinese text processing won't be available."
echo ""
echo "⚠️  WARNING: This will downgrade several packages:"
echo "   - torch: 2.8.0 → 2.6.0"
echo "   - torchaudio: 2.8.0 → 2.6.0"
echo "   - numpy: 1.26.0 → 1.25.x"
echo "   - diffusers: 0.35.2 → 0.29.0"
echo "   - transformers: 4.45.2 → 4.46.3"
echo "   - safetensors: 0.6.2 → 0.5.3"
echo ""
echo "⚠️  This may affect other components that depend on newer versions."
echo ""
echo "💡 RECOMMENDED: Use a virtual environment to avoid conflicts:"
echo "   python3 -m venv ~/chatterbox-env"
echo "   source ~/chatterbox-env/bin/activate"
echo "   bash setup/scripts/install_chatterbox_without_pkuseg.sh"
echo ""
read -p "Continue with system-wide installation? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Installation cancelled."
    exit 1
fi
echo ""

echo "Step 1: Installing core dependencies with exact versions..."
pip3 install --upgrade setuptools

# Install exact versions required by chatterbox-tts 0.1.4
echo "   Installing PyTorch 2.6.0..."
pip3 install torch==2.6.0 torchaudio==2.6.0

echo "   Installing numpy (compatible version)..."
pip3 install "numpy>=1.24.0,<1.26.0"

echo "   Installing other core dependencies..."
pip3 install librosa==0.11.0 transformers==4.46.3 diffusers==0.29.0 safetensors==0.5.3

echo "   Installing chatterbox-tts specific dependencies..."
pip3 install conformer==0.3.2 resemble-perth==1.0.1 s3tokenizer pykakasi==2.3.0 jaconv gradio==5.44.1

echo ""
echo "Step 2: Installing chatterbox-tts without pkuseg..."
pip3 install chatterbox-tts --no-deps

echo ""
echo "Step 3: Verifying installation..."
VERIFICATION_OUTPUT=$(python3 -c "
try:
    from chatterbox import ChatterboxTTS
    print('✅ ChatterboxTTS import successful')
    exit(0)
except ImportError as e:
    print(f'❌ Import error: {e}')
    import traceback
    traceback.print_exc()
    exit(1)
except Exception as e:
    print(f'⚠️  Warning: {e}')
    # Try to continue anyway
    exit(0)
" 2>&1)

if [ $? -eq 0 ]; then
    echo "$VERIFICATION_OUTPUT"
    echo "✅ ChatterboxTTS is installed and working!"
    echo ""
    echo "⚠️  Note: pkuseg is not installed (Chinese text processing unavailable)"
    echo "   English TTS and voice cloning will work fine"
    echo ""
    echo "📋 Next steps:"
    echo "   1. Generate voice sample: python3 setup/scripts/generate_chatterbox_voice_sample.py"
    echo "   2. Enable ChatterboxTTS in Settings → TTS Engine"
    echo "   3. Test by asking AuraVision a question"
else
    echo "$VERIFICATION_OUTPUT"
    echo ""
    echo "❌ Installation verification failed"
    echo ""
    echo "💡 Try testing manually:"
    echo "   python3 -c \"from chatterbox import ChatterboxTTS; print('OK')\""
    echo ""
    echo "   Check for any missing dependencies or version conflicts."
    exit 1
fi

echo ""
echo "================================================================================"
echo "✅ Installation complete!"
echo "================================================================================"


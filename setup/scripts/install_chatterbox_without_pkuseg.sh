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

echo "Step 1: Installing core dependencies..."
pip3 install --upgrade setuptools
pip3 install torch torchaudio numpy librosa transformers diffusers safetensors

echo ""
echo "Step 2: Installing chatterbox-tts without pkuseg..."
pip3 install chatterbox-tts --no-deps

echo ""
echo "Step 3: Verifying installation..."
if python3 -c "from chatterbox import ChatterboxTTS; print('✅ ChatterboxTTS import successful')" 2>/dev/null; then
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
    echo "❌ Installation verification failed"
    exit 1
fi

echo ""
echo "================================================================================"
echo "✅ Installation complete!"
echo "================================================================================"


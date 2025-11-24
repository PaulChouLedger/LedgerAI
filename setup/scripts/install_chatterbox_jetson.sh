#!/bin/bash
# Install ChatterboxTTS on Jetson with workarounds for pkuseg compatibility

set -e

echo "================================================================================"
echo "ChatterboxTTS Installation Script for Jetson"
echo "================================================================================"
echo ""

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}' | cut -d. -f1,2)
echo "🐍 Python version: $PYTHON_VERSION"

if [[ $(echo "$PYTHON_VERSION >= 3.12" | bc -l 2>/dev/null || echo "0") == "1" ]]; then
    echo "⚠️  Warning: Python 3.12+ may have compatibility issues"
fi

echo ""
echo "Step 1: Installing setuptools (provides distutils compatibility)..."
pip3 install --upgrade setuptools

echo ""
echo "Step 2: Attempting to install pkuseg with workaround..."
if pip3 install pkuseg --no-build-isolation 2>&1 | tee /tmp/pkuseg_install.log; then
    echo "✅ pkuseg installed successfully"
else
    echo "⚠️  pkuseg installation failed, but continuing..."
    echo "   (pkuseg is only needed for Chinese text processing)"
fi

echo ""
echo "Step 3: Installing chatterbox-tts..."
if pip3 install chatterbox-tts 2>&1 | tee /tmp/chatterbox_install.log; then
    echo "✅ chatterbox-tts installed successfully"
else
    echo ""
    echo "⚠️  Standard installation failed, trying without pkuseg dependency..."
    pip3 install torch torchaudio
    pip3 install chatterbox-tts --no-deps || {
        echo "❌ Installation failed"
        exit 1
    }
fi

echo ""
echo "Step 4: Verifying installation..."
if python3 -c "from chatterbox import ChatterboxTTS; print('✅ ChatterboxTTS import successful')" 2>/dev/null; then
    echo "✅ ChatterboxTTS is installed and working!"
    echo ""
    echo "📋 Next steps:"
    echo "   1. Generate voice sample: python3 setup/scripts/generate_chatterbox_voice_sample.py"
    echo "   2. Enable ChatterboxTTS in Settings → TTS Engine"
    echo "   3. Test by asking AuraVision a question"
else
    echo "❌ Installation verification failed"
    echo "   Check logs: /tmp/pkuseg_install.log and /tmp/chatterbox_install.log"
    exit 1
fi

echo ""
echo "================================================================================"
echo "✅ Installation complete!"
echo "================================================================================"


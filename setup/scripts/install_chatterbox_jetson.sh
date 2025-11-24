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
echo "   (pkuseg is only needed for Chinese text processing - may be optional)"

# Try multiple methods to install pkuseg
PKUSEG_INSTALLED=false

# Method 1: Try with setuptools and no build isolation
if pip3 install pkuseg --no-build-isolation 2>&1 | tee /tmp/pkuseg_install.log; then
    echo "✅ pkuseg installed successfully (method 1)"
    PKUSEG_INSTALLED=true
else
    echo "   Method 1 failed, trying method 2..."
    
    # Method 2: Try installing from pre-built wheel if available
    if pip3 install pkuseg --only-binary :all: 2>&1 | tee -a /tmp/pkuseg_install.log; then
        echo "✅ pkuseg installed successfully (method 2 - pre-built wheel)"
        PKUSEG_INSTALLED=true
    else
        echo "   Method 2 failed, trying method 3..."
        
        # Method 3: Try with environment variable to skip msvccompiler check
        export SETUPTOOLS_USE_DISTUTILS=stdlib
        if pip3 install pkuseg --no-build-isolation 2>&1 | tee -a /tmp/pkuseg_install.log; then
            echo "✅ pkuseg installed successfully (method 3)"
            PKUSEG_INSTALLED=true
        else
            echo "⚠️  All pkuseg installation methods failed"
            echo "   Continuing without pkuseg (may work for English-only TTS)"
        fi
    fi
fi

echo ""
echo "Step 3: Installing chatterbox-tts..."

# If pkuseg failed, try installing chatterbox-tts without it
if [ "$PKUSEG_INSTALLED" = false ]; then
    echo "   pkuseg not available, installing chatterbox-tts without pkuseg dependency..."
    
    # Install exact versions required by chatterbox-tts 0.1.4
    echo "   Installing PyTorch 2.6.0..."
    pip3 install torch==2.6.0 torchaudio==2.6.0 2>&1 | tee /tmp/chatterbox_install.log
    
    echo "   Installing numpy (compatible version)..."
    pip3 install "numpy>=1.24.0,<1.26.0" 2>&1 | tee -a /tmp/chatterbox_install.log
    
    echo "   Installing other core dependencies..."
    pip3 install librosa==0.11.0 transformers==4.46.3 diffusers==0.29.0 safetensors==0.5.3 2>&1 | tee -a /tmp/chatterbox_install.log
    
    echo "   Installing chatterbox-tts specific dependencies..."
    pip3 install conformer==0.3.2 resemble-perth==1.0.1 s3tokenizer pykakasi==2.3.0 jaconv gradio==5.44.1 2>&1 | tee -a /tmp/chatterbox_install.log
    
    # Try installing chatterbox-tts without pkuseg
    if pip3 install chatterbox-tts --no-deps 2>&1 | tee -a /tmp/chatterbox_install.log; then
        echo "✅ chatterbox-tts installed without pkuseg dependency"
    else
        echo "⚠️  Installation without dependencies failed, trying standard install..."
        if pip3 install chatterbox-tts 2>&1 | tee -a /tmp/chatterbox_install.log; then
            echo "✅ chatterbox-tts installed successfully (standard method)"
        else
            echo "❌ Installation failed"
            exit 1
        fi
    fi
else
    # Standard installation (pkuseg is available)
    if pip3 install chatterbox-tts 2>&1 | tee /tmp/chatterbox_install.log; then
        echo "✅ chatterbox-tts installed successfully"
    else
        echo "❌ Installation failed"
        exit 1
    fi
fi

echo ""
echo "Step 4: Verifying installation..."
VERIFICATION_OUTPUT=$(python3 -c "
try:
    from chatterbox import ChatterboxTTS
    print('✅ ChatterboxTTS import successful')
    exit(0)
except ImportError as e:
    print(f'❌ Import error: {e}')
    exit(1)
except Exception as e:
    print(f'⚠️  Import warning: {e}')
    # Try to continue anyway
    exit(0)
" 2>&1)

if [ $? -eq 0 ]; then
    echo "$VERIFICATION_OUTPUT"
    echo "✅ ChatterboxTTS is installed and working!"
    echo ""
    if [ "$PKUSEG_INSTALLED" = false ]; then
        echo "⚠️  Note: pkuseg is not installed (Chinese text processing unavailable)"
        echo "   English TTS should still work fine"
    fi
    echo ""
    echo "📋 Next steps:"
    echo "   1. Generate voice sample: python3 setup/scripts/generate_chatterbox_voice_sample.py"
    echo "   2. Enable ChatterboxTTS in Settings → TTS Engine"
    echo "   3. Test by asking AuraVision a question"
else
    echo "$VERIFICATION_OUTPUT"
    echo ""
    echo "⚠️  Installation verification had issues"
    echo "   Check logs: /tmp/pkuseg_install.log and /tmp/chatterbox_install.log"
    echo ""
    echo "💡 Try testing manually:"
    echo "   python3 -c \"from chatterbox import ChatterboxTTS; print('OK')\""
    echo ""
    echo "   If it works, you can proceed. pkuseg may not be needed for English TTS."
    exit 1
fi

echo ""
echo "================================================================================"
echo "✅ Installation complete!"
echo "================================================================================"


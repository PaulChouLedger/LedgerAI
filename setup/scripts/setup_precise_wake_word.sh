#!/bin/bash
# Setup Mycroft Precise Wake Word Detection for Jetson
# This is the MOST RELIABLE option for Jetson devices

set -e

echo "=========================================="
echo "Mycroft Precise Wake Word Setup"
echo "=========================================="
echo ""
echo "Mycroft Precise is highly recommended for Jetson - very reliable!"
echo ""

# Check if we're in a virtual environment
if [ -z "$VIRTUAL_ENV" ]; then
    echo "⚠️  Not in a virtual environment"
    echo "💡 Activate your virtual environment first:"
    echo "   source ~/aura-env/bin/activate"
    echo ""
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Install precise-runner and precise-engine
echo "[1/3] Installing precise-runner and precise-engine..."
pip install precise-runner precise-engine

# Download default model
echo ""
echo "[2/3] Downloading wake word model..."
MODEL_DIR="$HOME/precise-models"
mkdir -p "$MODEL_DIR"
cd "$MODEL_DIR"

if [ ! -f "hey-mycroft.pb" ]; then
    echo "Downloading hey-mycroft.pb..."
    wget -q https://github.com/MycroftAI/precise-data/raw/models/hey-mycroft.pb
    echo "✅ Model downloaded to: $MODEL_DIR/hey-mycroft.pb"
else
    echo "✅ Model already exists: $MODEL_DIR/hey-mycroft.pb"
fi

# Test installation
echo ""
echo "[3/3] Testing installation..."
python3 -c "from precise_runner import PreciseEngine; print('✅ Precise imported successfully!')" || {
    echo "❌ Precise import failed"
    exit 1
}

# Check if precise-engine executable is available
if command -v precise-engine &> /dev/null; then
    echo "✅ precise-engine executable found: $(which precise-engine)"
else
    echo "⚠️  precise-engine executable not found in PATH"
    echo "💡 Try: pip install --upgrade precise-engine"
    echo "💡 Or check if it's installed: pip show precise-engine"
fi

echo ""
echo "=========================================="
echo "✅ Setup Complete!"
echo "=========================================="
echo ""
echo "Model location: $MODEL_DIR/hey-mycroft.pb"
echo ""
echo "To use a custom model:"
echo "  1. Train at: https://github.com/MycroftAI/mycroft-precise"
echo "  2. Place .pb file in: $MODEL_DIR/"
echo "  3. Update model path in Settings → AI Model Settings"
echo ""
echo "Precise will be automatically used when you restart Aura!"
echo ""


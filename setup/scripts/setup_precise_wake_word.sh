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

# Install precise-runner (Python package)
echo "[1/4] Installing precise-runner (Python package)..."
pip install precise-runner

# Download precise-engine binary for Jetson (aarch64)
echo ""
echo "[2/4] Downloading precise-engine binary for Jetson..."
PRECISE_DIR="$HOME/.mycroft/precise"
mkdir -p "$PRECISE_DIR"

if [ ! -f "$PRECISE_DIR/precise-engine/precise-engine" ]; then
    cd /tmp
    echo "Downloading precise-engine for aarch64..."
    wget -q https://github.com/MycroftAI/mycroft-precise/releases/download/v0.3.0/precise-all_0.3.0_aarch64.tar.gz || {
        echo "❌ Failed to download precise-engine"
        echo "💡 Try manually:"
        echo "   wget https://github.com/MycroftAI/mycroft-precise/releases/download/v0.3.0/precise-all_0.3.0_aarch64.tar.gz"
        exit 1
    }
    tar xzf precise-all_0.3.0_aarch64.tar.gz
    mv precise "$PRECISE_DIR/precise-engine"
    chmod +x "$PRECISE_DIR/precise-engine/precise-engine"
    rm precise-all_0.3.0_aarch64.tar.gz
    echo "✅ precise-engine installed to: $PRECISE_DIR/precise-engine/precise-engine"
else
    echo "✅ precise-engine already exists: $PRECISE_DIR/precise-engine/precise-engine"
fi

# Download default model
echo ""
echo "[3/4] Downloading wake word model..."
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
echo "[4/4] Testing installation..."
python3 -c "from precise_runner import PreciseEngine; print('✅ Precise imported successfully!')" || {
    echo "❌ Precise import failed"
    exit 1
}

# Check if precise-engine executable is available
if [ -f "$PRECISE_DIR/precise-engine/precise-engine" ] && [ -x "$PRECISE_DIR/precise-engine/precise-engine" ]; then
    echo "✅ precise-engine executable found: $PRECISE_DIR/precise-engine/precise-engine"
else
    echo "⚠️  precise-engine executable not found or not executable"
    echo "💡 Check: ls -la $PRECISE_DIR/precise-engine/precise-engine"
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


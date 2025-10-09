#!/bin/bash
#
# LedgerAI - Install Dependencies
# Simple script to install all required dependencies
#

set -e

echo "=================================================="
echo "  LedgerAI - Dependency Installer"
echo "=================================================="
echo ""

# Detect OS
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    OS="linux"
elif [[ "$OSTYPE" == "darwin"* ]]; then
    OS="macos"
else
    echo "❌ Unsupported OS: $OSTYPE"
    exit 1
fi

echo "📋 Detected OS: $OS"
echo ""

# ============================================
# System Dependencies
# ============================================
echo "📦 Installing system dependencies..."
echo ""

if [ "$OS" = "linux" ]; then
    sudo apt-get update
    sudo apt-get install -y \
        python3 \
        python3-pip \
        python3-dev \
        python3-venv \
        build-essential \
        portaudio19-dev \
        libasound2-dev \
        libsndfile1 \
        ffmpeg \
        sox \
        alsa-utils \
        pulseaudio \
        libudev-dev \
        libusb-1.0-0-dev \
        python3-pyqt5 \
        python3-pyqt5.qtmultimedia \
        docker.io \
        docker-compose
    
    # Add user to docker group
    sudo usermod -aG docker $USER
    echo "⚠️  You may need to log out and back in for docker group changes"
    
elif [ "$OS" = "macos" ]; then
    # Check for Homebrew
    if ! command -v brew &> /dev/null; then
        echo "📥 Installing Homebrew..."
        /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    fi
    
    brew install \
        python@3.11 \
        portaudio \
        ffmpeg \
        sox
    
    echo "⚠️  Please install Docker Desktop manually from:"
    echo "    https://www.docker.com/products/docker-desktop"
fi

echo "✅ System dependencies installed"
echo ""

# ============================================
# Python Dependencies
# ============================================
echo "🐍 Installing Python dependencies..."
echo ""

# Upgrade pip
python3 -m pip install --upgrade pip

# Install PyTorch
echo "Installing PyTorch..."
if [ "$OS" = "macos" ]; then
    pip3 install torch torchvision torchaudio
else
    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
fi

# Install audio libraries
echo "Installing audio libraries..."
pip3 install \
    sounddevice \
    soundfile \
    numpy \
    scipy

# Install PyQt5 for macOS
if [ "$OS" = "macos" ]; then
    echo "Installing PyQt5..."
    pip3 install PyQt5
fi

# Install core dependencies
echo "Installing core dependencies..."
pip3 install \
    requests \
    python-dotenv \
    elevenlabs \
    pyusb \
    flask \
    werkzeug

# Install optional dependencies
echo "Installing optional dependencies..."
pip3 install \
    google-api-python-client \
    google-auth \
    google-auth-oauthlib \
    google-auth-httplib2 \
    python-telegram-bot || echo "⚠️  Some optional packages failed (non-critical)"

echo "✅ Python dependencies installed"
echo ""

# ============================================
# Docker Containers
# ============================================
echo "🐳 Building Docker containers..."
echo ""

if command -v docker &> /dev/null; then
    docker compose build
    echo "✅ Docker containers built"
else
    echo "⚠️  Docker not available - skipping container build"
    echo "    Install Docker and run: docker compose build"
fi
echo ""

# ============================================
# Setup (Linux only - ReSpeaker)
# ============================================
if [ "$OS" = "linux" ] && [ -f "scripts/setup_usb_permissions.sh" ]; then
    echo "🎤 Setup ReSpeaker USB permissions? (y/n)"
    read -r response
    if [[ "$response" =~ ^[Yy]$ ]]; then
        sudo bash scripts/setup_usb_permissions.sh
    fi
fi

# ============================================
# Create directories
# ============================================
echo "📁 Creating required directories..."
mkdir -p shared/input_audio
mkdir -p shared/output_audio
mkdir -p data/embeddings
mkdir -p data/parsed
mkdir -p data/input
mkdir -p rag-container/cache
echo "✅ Directories created"
echo ""

# ============================================
# Summary
# ============================================
echo "=================================================="
echo "  ✅ Installation Complete!"
echo "=================================================="
echo ""
echo "Next steps:"
echo ""
echo "1. Configure environment variables:"
echo "   nano llm-container/.env"
echo "   (Add your ELEVENLABS_API_KEY)"
echo ""
echo "2. Start Docker containers:"
echo "   docker compose up -d"
echo ""
echo "3. Run the application:"
echo "   python3 aura-control/main.py"
echo ""
if [ "$OS" = "linux" ]; then
    echo "⚠️  Log out and back in for docker group changes!"
    echo ""
fi
echo "=================================================="


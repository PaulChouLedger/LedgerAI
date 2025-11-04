#!/bin/bash
#
# LedgerAI - Jetson Installation Script
# 
# Installs only HOST dependencies needed to run LedgerAI on Jetson devices.
# Heavy lifting (LLM, Whisper, RAG) runs in Docker containers.
#
# Usage:
#   bash install_jetson.sh
#

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[✓]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[⚠]${NC} $1"
}

log_error() {
    echo -e "${RED}[✗]${NC} $1"
}

print_header() {
    echo ""
    echo "================================================================================"
    echo "  $1"
    echo "================================================================================"
    echo ""
}

command_exists() {
    command -v "$1" >/dev/null 2>&1
}

print_header "LedgerAI - Jetson NX Installation"

log_info "This script installs HOST dependencies for Jetson devices"
log_info "Containers include: Whisper, LLM inference, and RAG"
echo ""

# ============================================
# Check if running on Jetson
# ============================================
if [ ! -f /etc/nv_tegra_release ]; then
    log_warning "This doesn't appear to be a Jetson device"
    echo -n "Continue anyway? (y/n) "
    read -r response
    if [[ ! "$response" =~ ^[Yy]$ ]]; then
        exit 0
    fi
else
    JETSON_VERSION=$(cat /etc/nv_tegra_release)
    log_info "Detected Jetson: $JETSON_VERSION"
fi

# ============================================
# System Dependencies (Host only)
# ============================================
print_header "Installing System Dependencies (Host)"

log_info "Updating package lists..."
sudo apt-get update

log_info "Installing Python 3 and essential packages (including git)..."
sudo apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    python3-venv \
    build-essential \
    git

# Audio libraries (for aura-control/listener.py if running on host)
log_info "Installing audio libraries..."
sudo apt-get install -y \
    portaudio19-dev \
    libasound2-dev \
    libsndfile1 \
    alsa-utils \
    pulseaudio

# USB device access (for ReSpeaker)
log_info "Installing USB libraries..."
sudo apt-get install -y \
    libudev-dev \
    libusb-1.0-0-dev

# PyQt5 for GUI (if running GUI on host)
log_info "Installing PyQt5..."
sudo apt-get install -y \
    python3-pyqt5 \
    python3-pyqt5.qtmultimedia

log_success "System packages installed"

# ============================================
# Docker Installation
# ============================================
print_header "Checking Docker Installation"

if command_exists docker; then
    DOCKER_VERSION=$(docker --version)
    log_success "Docker already installed: $DOCKER_VERSION"
else
    log_info "Installing Docker..."
    log_warning "Using docker.io (compatible with Jetson)"
    
    sudo apt-get install -y docker.io docker-compose
    
    # Add user to docker group
    sudo usermod -aG docker $USER
    
    log_success "Docker installed"
    log_warning "You must LOG OUT and LOG BACK IN for docker group changes to take effect"
fi

# ============================================
# Jetson Container Tools Installation
# ============================================
print_header "Installing Jetson Container Tools"

# Check if jetson-containers already exists
if [ -d "$HOME/jetson-containers" ]; then
    log_info "jetson-containers directory already exists"
    echo -n "Reinstall Jetson container tools? (y/n) "
    read -r response
    if [[ "$response" =~ ^[Yy]$ ]]; then
        log_info "Reinstalling Jetson container tools..."
        cd "$HOME/jetson-containers"
        bash install.sh
    else
        log_info "Skipping Jetson container tools installation"
    fi
else
    log_info "Installing Jetson container tools (required for dustynv/* base images)..."
    log_info "This provides the base images used by Aura containers"
    
    cd "$HOME"
    git clone https://github.com/dusty-nv/jetson-containers
    cd jetson-containers
    bash install.sh
    
    if [ $? -eq 0 ]; then
        log_success "Jetson container tools installed"
    else
        log_error "Jetson container tools installation failed"
        log_warning "You may need to install manually: git clone https://github.com/dusty-nv/jetson-containers && bash jetson-containers/install.sh"
    fi
fi

# ============================================
# Python Dependencies (Host only)
# ============================================
print_header "Installing Python Dependencies (Host)"

log_info "These are only for aura-control (GUI/listener)"
log_info "Container dependencies are in their respective Dockerfiles"
echo ""

# Upgrade pip
python3 -m pip install --upgrade pip

# PyTorch (for VAD in listener.py)
log_info "Installing PyTorch for Jetson..."
# Use Jetson-specific PyTorch wheel if available
# For now, use CPU version (Jetson has its own CUDA torch in nvidia repos)
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Audio libraries
log_info "Installing audio libraries..."
pip3 install sounddevice soundfile numpy scipy

# Core dependencies
log_info "Installing core dependencies..."
pip3 install \
    requests \
    python-dotenv \
    elevenlabs \
    pyusb

# PyQt5 (if not from apt)
log_info "Installing PyQt5 via pip (backup)..."
pip3 install PyQt5 || log_warning "PyQt5 pip install failed (system version will be used)"

log_success "Python dependencies installed"

# ============================================
# Create Required Directories
# ============================================
print_header "Creating Required Directories"

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

mkdir -p "$SCRIPT_DIR/shared/input_audio"
mkdir -p "$SCRIPT_DIR/shared/output_audio"
mkdir -p "$SCRIPT_DIR/data/embeddings"
mkdir -p "$SCRIPT_DIR/data/parsed"
mkdir -p "$SCRIPT_DIR/data/input"
mkdir -p "$SCRIPT_DIR/rag-container/cache"

log_success "Directories created"

# ============================================
# Setup ReSpeaker (if connected)
# ============================================
print_header "ReSpeaker USB Setup"

if lsusb | grep -q "2886:0018"; then
    log_info "ReSpeaker 4-Mic Array detected!"
    
    if [ -f "$SCRIPT_DIR/scripts/setup_usb_permissions.sh" ]; then
        log_info "Setting up USB permissions..."
        sudo bash "$SCRIPT_DIR/scripts/setup_usb_permissions.sh"
    fi
    
    if [ -f "$SCRIPT_DIR/scripts/install_auto_tune.sh" ]; then
        echo -n "Install ReSpeaker auto-tune service? (y/n) "
        read -r response
        if [[ "$response" =~ ^[Yy]$ ]]; then
            sudo bash "$SCRIPT_DIR/scripts/install_auto_tune.sh"
        fi
    fi
else
    log_warning "ReSpeaker not detected (connect and rerun if needed)"
fi

# ============================================
# Environment Setup
# ============================================
print_header "Environment Configuration"

log_info "Aura uses a single .env file at the workspace root"
log_info "Use './aura_config.sh' to configure all settings interactively"

if [ ! -f "$SCRIPT_DIR/.env" ]; then
    if [ -f "$SCRIPT_DIR/.env.example" ]; then
        log_info "Creating .env from template..."
        cp "$SCRIPT_DIR/.env.example" "$SCRIPT_DIR/.env"
        log_success "Created .env file at workspace root"
        log_warning "Run './aura_config.sh' to configure your settings (API keys, etc.)"
    else
        log_warning ".env.example not found - you may need to create .env manually"
        log_info "Run './aura_config.sh' to set up your configuration"
    fi
else
    log_info ".env file already exists at workspace root"
    log_info "Run './aura_config.sh' to view or modify settings"
fi

# ============================================
# Summary
# ============================================
print_header "Installation Complete!"

echo "📋 Next Steps:"
echo ""
echo "1. Configure your settings (API keys, etc.):"
echo "   cd $SCRIPT_DIR"
echo "   ./aura_config.sh"
echo "   (Interactive configuration manager)"
echo ""
echo "2. Build Docker containers:"
echo "   cd $SCRIPT_DIR/setup"
echo "   docker compose build"
echo ""
echo "3. Start the containers:"
echo "   cd $SCRIPT_DIR/setup"
echo "   docker compose up -d"
echo ""
echo "4. Run the main application:"
echo "   cd $SCRIPT_DIR"
echo "   python3 aura-control/core/main.py"
echo ""

if ! command_exists docker || ! groups | grep -q docker; then
    log_warning "IMPORTANT: Log out and log back in for docker permissions!"
    echo ""
fi

echo "🔧 Useful Commands:"
echo ""
echo "  docker compose up -d          # Start containers in background"
echo "  docker compose logs -f        # View container logs"
echo "  docker compose down           # Stop containers"
echo "  docker compose build          # Rebuild containers"
echo ""

echo "📝 Notes for Jetson:"
echo ""
echo "  - Heavy compute (LLM, Whisper, RAG) runs in containers"
echo "  - aura-control (GUI/listener) runs on host"
echo "  - Containers use ~6GB RAM total"
echo "  - Models are quantized for efficiency"
echo ""

log_success "Setup complete!"


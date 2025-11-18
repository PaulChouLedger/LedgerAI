#!/bin/bash
# Standalone script to install Mycroft Precise for wake word detection
# Usage: bash install_mycroft_precise.sh

set -e  # Exit on error

echo "=========================================="
echo "  Mycroft Precise Installation"
echo "=========================================="
echo ""

# Detect user (works even when run via sudo)
AURA_USER="${SUDO_USER:-$USER}"
AURA_HOME="/home/$AURA_USER"

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

print_info() {
    echo -e "${YELLOW}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Activate virtual environment if it exists
if [ -d "$AURA_HOME/aura-env" ]; then
    print_info "Activating virtual environment..."
    source "$AURA_HOME/aura-env/bin/activate"
else
    print_error "Virtual environment not found at $AURA_HOME/aura-env"
    print_info "Please run the full installation script first, or activate your Python environment manually"
    exit 1
fi

# Step 1: Install precise-runner (Python package)
print_info "Installing precise-runner Python package..."
if pip install precise-runner precise-engine 2>&1 | tee /tmp/precise_install.log; then
    print_success "precise-runner installed successfully"
else
    print_error "precise-runner installation had issues (check /tmp/precise_install.log)"
    print_info "   Wake word detection may not work until this is resolved"
fi

# Step 2: Download precise-engine binary for ARM64/Jetson
print_info "Downloading precise-engine binary for ARM64/Jetson..."
PRECISE_ENGINE_DIR="$AURA_HOME/.mycroft/precise/precise-engine"
mkdir -p "$PRECISE_ENGINE_DIR"

# Check if precise-engine already exists
if [ -f "$PRECISE_ENGINE_DIR/precise-engine" ] && [ -x "$PRECISE_ENGINE_DIR/precise-engine" ]; then
    print_success "precise-engine already exists at $PRECISE_ENGINE_DIR/precise-engine"
else
    cd /tmp
    if wget -q --show-progress https://github.com/MycroftAI/mycroft-precise/releases/download/v0.3.0/precise-all_0.3.0_aarch64.tar.gz; then
        print_info "Extracting precise-engine..."
        tar xzf precise-all_0.3.0_aarch64.tar.gz
        if [ -d "precise" ] && [ -f "precise/precise-engine" ]; then
            cp -r precise/* "$PRECISE_ENGINE_DIR/"
            chmod +x "$PRECISE_ENGINE_DIR/precise-engine"
            print_success "precise-engine installed to $PRECISE_ENGINE_DIR/precise-engine"
            rm -rf precise precise-all_0.3.0_aarch64.tar.gz
        else
            print_error "precise-engine not found in extracted archive"
        fi
    else
        print_error "Failed to download precise-engine binary"
        print_info "   You can download manually:"
        print_info "   cd ~ && wget https://github.com/MycroftAI/mycroft-precise/releases/download/v0.3.0/precise-all_0.3.0_aarch64.tar.gz"
        print_info "   tar xzf precise-all_0.3.0_aarch64.tar.gz"
        print_info "   mkdir -p ~/.mycroft/precise/precise-engine"
        print_info "   cp -r precise/* ~/.mycroft/precise/precise-engine/"
    fi
fi

# Step 3: Download wake word model (hey-mycroft.pb)
print_info "Downloading wake word model (hey-mycroft.pb)..."
MODEL_DIR="$AURA_HOME/precise-models"
mkdir -p "$MODEL_DIR"

if [ -f "$MODEL_DIR/hey-mycroft.pb" ]; then
    print_success "Wake word model already exists at $MODEL_DIR/hey-mycroft.pb"
else
    if wget -q --show-progress -O "$MODEL_DIR/hey-mycroft.pb" https://github.com/MycroftAI/precise-data/raw/models/hey-mycroft.pb; then
        print_success "Wake word model downloaded to $MODEL_DIR/hey-mycroft.pb"
    else
        print_error "Failed to download wake word model"
        print_info "   You can download manually:"
        print_info "   mkdir -p ~/precise-models"
        print_info "   wget -O ~/precise-models/hey-mycroft.pb https://github.com/MycroftAI/precise-data/raw/models/hey-mycroft.pb"
    fi
fi

# Step 4: Create symlink in home directory for compatibility
if [ ! -f "$AURA_HOME/hey-mycroft.pb" ] && [ -f "$MODEL_DIR/hey-mycroft.pb" ]; then
    ln -s "$MODEL_DIR/hey-mycroft.pb" "$AURA_HOME/hey-mycroft.pb"
    print_success "Created symlink: $AURA_HOME/hey-mycroft.pb -> $MODEL_DIR/hey-mycroft.pb"
fi

# Step 5: Verify installation
print_info ""
print_info "Verifying Mycroft Precise installation..."

VERIFICATION_PASSED=true

if python3 -c "from precise_runner import PreciseEngine, PreciseRunner" 2>/dev/null; then
    print_success "Mycroft Precise Python package is importable"
else
    print_error "Mycroft Precise Python package not importable"
    VERIFICATION_PASSED=false
fi

if [ -x "$PRECISE_ENGINE_DIR/precise-engine" ]; then
    print_success "precise-engine binary is executable"
else
    print_error "precise-engine binary not found or not executable"
    VERIFICATION_PASSED=false
fi

if [ -f "$MODEL_DIR/hey-mycroft.pb" ] || [ -f "$AURA_HOME/hey-mycroft.pb" ]; then
    print_success "Wake word model file found"
else
    print_error "Wake word model file not found"
    VERIFICATION_PASSED=false
fi

echo ""
if [ "$VERIFICATION_PASSED" = true ]; then
    print_success "✅ Mycroft Precise installation complete!"
    print_info "   You can now enable wake word detection in Settings → AI Model Settings"
else
    print_error "⚠️  Installation completed with some issues"
    print_info "   Please check the errors above and resolve them"
    exit 1
fi

echo ""
print_info "Installation locations:"
print_info "   Python package: $(python3 -c 'import precise_runner; print(precise_runner.__file__)' 2>/dev/null || echo 'Not found')"
print_info "   Binary: $PRECISE_ENGINE_DIR/precise-engine"
print_info "   Model: $MODEL_DIR/hey-mycroft.pb"


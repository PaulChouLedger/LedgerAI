#!/bin/bash
# setup_precise_wake_word.sh - Download and install Mycroft Precise engine binary for Jetson
# Usage: ./setup_precise_wake_word.sh

set -e

echo "=========================================="
echo "  Mycroft Precise Engine Setup"
echo "=========================================="
echo ""

# Configuration
AURA_USER="${SUDO_USER:-$USER}"
AURA_HOME="/home/$AURA_USER"
PRECISE_ENGINE_DIR="$AURA_HOME/.mycroft/precise/precise-engine"
MODEL_DIR="$AURA_HOME/precise-models"

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

# Step 1: Download precise-engine binary
print_info "Step 1: Downloading precise-engine binary for ARM64/Jetson..."
mkdir -p "$PRECISE_ENGINE_DIR"

# Check if already exists
if [ -f "$PRECISE_ENGINE_DIR/precise-engine" ] && [ -x "$PRECISE_ENGINE_DIR/precise-engine" ]; then
    print_success "precise-engine already exists at $PRECISE_ENGINE_DIR/precise-engine"
    
    # Verify it's a binary
    if file "$PRECISE_ENGINE_DIR/precise-engine" | grep -q "ELF"; then
        print_success "Verified: It's a binary executable (not a script)"
    else
        print_error "Warning: File exists but may not be a binary"
        print_info "Re-downloading..."
        rm -f "$PRECISE_ENGINE_DIR/precise-engine"
    fi
fi

if [ ! -f "$PRECISE_ENGINE_DIR/precise-engine" ]; then
    cd /tmp
    print_info "Downloading from GitHub releases..."
    
    if wget -q --show-progress https://github.com/MycroftAI/mycroft-precise/releases/download/v0.3.0/precise-all_0.3.0_aarch64.tar.gz; then
        print_info "Extracting archive..."
        tar xzf precise-all_0.3.0_aarch64.tar.gz
        
        if [ -d "precise" ] && [ -f "precise/precise-engine" ]; then
            cp -r precise/* "$PRECISE_ENGINE_DIR/"
            chmod +x "$PRECISE_ENGINE_DIR/precise-engine"
            print_success "precise-engine installed to $PRECISE_ENGINE_DIR/precise-engine"
            rm -rf precise precise-all_0.3.0_aarch64.tar.gz
        else
            print_error "precise-engine not found in extracted archive"
            print_info "Archive structure may have changed"
            exit 1
        fi
    else
        print_error "Failed to download precise-engine binary"
        print_info "Check your internet connection and try again"
        exit 1
    fi
fi

# Step 2: Download wake word model
print_info ""
print_info "Step 2: Downloading wake word model (hey-mycroft.pb)..."
mkdir -p "$MODEL_DIR"

if [ -f "$MODEL_DIR/hey-mycroft.pb" ]; then
    print_success "Wake word model already exists at $MODEL_DIR/hey-mycroft.pb"
else
    print_info "Downloading model from GitHub..."
    if wget -q --show-progress -O "$MODEL_DIR/hey-mycroft.pb" https://github.com/MycroftAI/precise-data/raw/models/hey-mycroft.pb; then
        print_success "Wake word model downloaded to $MODEL_DIR/hey-mycroft.pb"
    else
        print_error "Failed to download wake word model"
        print_info "You can download manually:"
        print_info "  mkdir -p $MODEL_DIR"
        print_info "  wget -O $MODEL_DIR/hey-mycroft.pb https://github.com/MycroftAI/precise-data/raw/models/hey-mycroft.pb"
    fi
fi

# Step 3: Create symlink in home directory for compatibility
if [ ! -f "$AURA_HOME/hey-mycroft.pb" ] && [ -f "$MODEL_DIR/hey-mycroft.pb" ]; then
    ln -s "$MODEL_DIR/hey-mycroft.pb" "$AURA_HOME/hey-mycroft.pb"
    print_success "Created symlink: $AURA_HOME/hey-mycroft.pb -> $MODEL_DIR/hey-mycroft.pb"
fi

# Step 4: Verify installation
print_info ""
print_info "Step 3: Verifying installation..."

# Check binary
if [ -x "$PRECISE_ENGINE_DIR/precise-engine" ]; then
    BINARY_TYPE=$(file "$PRECISE_ENGINE_DIR/precise-engine" | grep -o "ELF.*" || echo "unknown")
    print_success "precise-engine binary is executable"
    print_info "  Type: $BINARY_TYPE"
    print_info "  Location: $PRECISE_ENGINE_DIR/precise-engine"
else
    print_error "precise-engine binary not found or not executable"
    exit 1
fi

# Check model
if [ -f "$MODEL_DIR/hey-mycroft.pb" ] || [ -f "$AURA_HOME/hey-mycroft.pb" ]; then
    MODEL_SIZE=$(du -h "$MODEL_DIR/hey-mycroft.pb" 2>/dev/null | cut -f1 || echo "unknown")
    print_success "Wake word model file found"
    print_info "  Size: $MODEL_SIZE"
    print_info "  Location: $MODEL_DIR/hey-mycroft.pb"
else
    print_error "Wake word model file not found"
    print_info "  Download it manually (see instructions above)"
fi

# Step 5: Test binary (optional)
print_info ""
print_info "Step 4: Testing binary..."
if [ -f "$MODEL_DIR/hey-mycroft.pb" ]; then
    print_info "Running: $PRECISE_ENGINE_DIR/precise-engine --help"
    if "$PRECISE_ENGINE_DIR/precise-engine" --help > /dev/null 2>&1; then
        print_success "Binary test passed!"
    else
        print_error "Binary test failed (may still work, but check output above)"
    fi
else
    print_info "Skipping binary test (model file not found)"
fi

echo ""
echo "=========================================="
echo "  Setup Complete!"
echo "=========================================="
echo ""
echo "✅ precise-engine binary: $PRECISE_ENGINE_DIR/precise-engine"
if [ -f "$MODEL_DIR/hey-mycroft.pb" ]; then
    echo "✅ Wake word model: $MODEL_DIR/hey-mycroft.pb"
else
    echo "⚠️  Wake word model: Not found (download manually)"
fi
echo ""
echo "The wake word detector should now be able to find the binary."
echo "Restart Aura to use wake word detection."
echo ""


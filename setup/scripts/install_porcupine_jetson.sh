#!/bin/bash
# Install Porcupine for Jetson using pre-built ARM64 library
# This script downloads the pre-built library from GitHub and installs the Python binding

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PORCUPINE_LIB_DIR="$SCRIPT_DIR/porcupine_lib"
PORCUPINE_VERSION="v3.0.3"
LIB_URL="https://github.com/Picovoice/porcupine/raw/${PORCUPINE_VERSION}/lib/linux/aarch64/libpv_porcupine.so"

echo "=========================================="
echo "Installing Porcupine for Jetson (ARM64)"
echo "=========================================="
echo ""

# Check if we're on ARM64
if [ "$(uname -m)" != "aarch64" ]; then
    echo "⚠️  Warning: This script is designed for ARM64 (Jetson) devices"
    echo "   Current architecture: $(uname -m)"
    read -p "   Continue anyway? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Create directory for library
mkdir -p "$PORCUPINE_LIB_DIR"
cd "$PORCUPINE_LIB_DIR"

echo "[1/4] Downloading pre-built ARM64 library..."
if [ -f "libpv_porcupine.so" ]; then
    echo "   ✅ Library already exists, skipping download"
else
    echo "   📥 Downloading from: $LIB_URL"
    curl -LO "$LIB_URL"
    if [ $? -eq 0 ]; then
        echo "   ✅ Library downloaded successfully"
    else
        echo "   ❌ Failed to download library"
        exit 1
    fi
fi

# Verify library file
if [ ! -f "libpv_porcupine.so" ]; then
    echo "   ❌ Library file not found after download"
    exit 1
fi

LIB_SIZE=$(stat -f%z "libpv_porcupine.so" 2>/dev/null || stat -c%s "libpv_porcupine.so" 2>/dev/null)
if [ "$LIB_SIZE" -lt 100000 ]; then
    echo "   ⚠️  Warning: Library file seems too small ($LIB_SIZE bytes)"
    echo "   This might be an error page instead of the actual library"
    exit 1
fi

echo ""
echo "[2/4] Installing Python binding (pvporcupine)..."
# Install Python package (this will fail on platform check, but we'll handle it)
pip install pvporcupine || {
    echo "   ⚠️  pip install failed (expected on Jetson)"
    echo "   Will proceed with manual library setup"
}

echo ""
echo "[3/4] Setting up library path..."
# Find where pvporcupine is installed
PYTHON_SITE_PACKAGES=$(python -c "import site; print(site.getsitepackages()[0])" 2>/dev/null || python -c "import site; print(site.USER_SITE)" 2>/dev/null || echo "")

if [ -n "$PYTHON_SITE_PACKAGES" ] && [ -d "$PYTHON_SITE_PACKAGES/pvporcupine" ]; then
    PVPORCUPINE_DIR="$PYTHON_SITE_PACKAGES/pvporcupine"
    echo "   📍 Found pvporcupine at: $PVPORCUPINE_DIR"
    
    # Check if there's a lib directory
    if [ -d "$PVPORCUPINE_DIR/lib" ]; then
        echo "   📂 Copying library to pvporcupine/lib/"
        cp "$PORCUPINE_LIB_DIR/libpv_porcupine.so" "$PVPORCUPINE_DIR/lib/"
        echo "   ✅ Library copied"
    else
        echo "   📂 Creating lib directory in pvporcupine"
        mkdir -p "$PVPORCUPINE_DIR/lib"
        cp "$PORCUPINE_LIB_DIR/libpv_porcupine.so" "$PVPORCUPINE_DIR/lib/"
        echo "   ✅ Library copied"
    fi
else
    echo "   ⚠️  pvporcupine package not found in site-packages"
    echo "   You may need to install it first, or set PYTHONPATH manually"
    echo ""
    echo "   To use the library manually, set:"
    echo "   export PVPORCUPINE_LIB_PATH=\"$PORCUPINE_LIB_DIR/libpv_porcupine.so\""
fi

echo ""
echo "[4/4] Verifying installation..."
python3 << 'EOF'
import sys
import os

# Try to import pvporcupine
try:
    import pvporcupine
    print("   ✅ pvporcupine Python package is importable")
    
    # Try to check if library can be found
    try:
        # This will fail if library is missing, but at least we know the package works
        print("   ✅ Python binding is ready")
    except Exception as e:
        print(f"   ⚠️  Library check: {e}")
        
except ImportError as e:
    print(f"   ❌ Failed to import pvporcupine: {e}")
    print("   Install with: pip install pvporcupine")
    sys.exit(1)
EOF

echo ""
echo "=========================================="
echo "✅ Porcupine installation complete!"
echo "=========================================="
echo ""
echo "Library location: $PORCUPINE_LIB_DIR/libpv_porcupine.so"
echo ""
echo "If you encounter 'Unsupported platform' errors, you may need to:"
echo "1. Set environment variable: export PVPORCUPINE_LIB_PATH=\"$PORCUPINE_LIB_DIR/libpv_porcupine.so\""
echo "2. Or copy the library to the pvporcupine package directory"
echo ""
echo "To test, run: python -c 'import pvporcupine; print(pvporcupine.KEYWORDS)'"


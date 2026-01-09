#!/bin/bash
# Script to copy onnxruntime 1.23.2 from working device to problematic device
# Run this on the WORKING device to prepare files for transfer

set -e

echo "=========================================="
echo "  Copy onnxruntime 1.23.2 from Working Device"
echo "=========================================="
echo ""
echo "This script should be run on the WORKING device to prepare files for transfer."
echo ""

# Check if we're in a virtual environment
if [ -z "$VIRTUAL_ENV" ] && [ -d "$HOME/aura-env" ]; then
    echo "[INFO] Activating virtual environment..."
    source "$HOME/aura-env/bin/activate"
fi

# Find the onnxruntime installation
echo "[STEP] 1. Finding onnxruntime 1.23.2 installation..."
ONNXRUNTIME_PATH=$(python3 -c "import onnxruntime; import os; print(os.path.dirname(onnxruntime.__file__))" 2>/dev/null || echo "")
if [ -z "$ONNXRUNTIME_PATH" ]; then
    echo "[ERROR] ❌ Could not find onnxruntime installation"
    exit 1
fi

RUNTIME_VERSION=$(python3 -c "import onnxruntime; print(onnxruntime.__version__)" 2>/dev/null || echo "unknown")
echo "[INFO] Found onnxruntime at: $ONNXRUNTIME_PATH"
echo "[INFO] Runtime version: $RUNTIME_VERSION"

if [ "$RUNTIME_VERSION" != "1.23.2" ]; then
    echo "[WARNING] ⚠️  Runtime version is $RUNTIME_VERSION, not 1.23.2"
    echo "[INFO]    Continuing anyway..."
fi

# Find dist-info
SITE_PACKAGES=$(dirname "$ONNXRUNTIME_PATH")
DIST_INFO=$(find "$SITE_PACKAGES" -maxdepth 1 -name "onnxruntime-1.23.2.dist-info" -type d 2>/dev/null | head -1)
if [ -z "$DIST_INFO" ]; then
    echo "[WARNING] ⚠️  Could not find onnxruntime-1.23.2.dist-info"
    echo "[INFO]    Will try to create a package from the installed files"
fi

# Create a temporary directory for packaging
TEMP_DIR=$(mktemp -d)
echo ""
echo "[STEP] 2. Creating package archive..."
echo "[INFO] Temporary directory: $TEMP_DIR"

# Copy onnxruntime package
echo "[INFO] Copying onnxruntime package..."
cp -r "$ONNXRUNTIME_PATH" "$TEMP_DIR/onnxruntime" 2>/dev/null || {
    echo "[ERROR] ❌ Failed to copy onnxruntime package"
    exit 1
}

# Copy dist-info if found
if [ -n "$DIST_INFO" ]; then
    echo "[INFO] Copying dist-info..."
    cp -r "$DIST_INFO" "$TEMP_DIR/" 2>/dev/null || true
fi

# Create archive
ARCHIVE_NAME="onnxruntime-1.23.2-linux-arm64.tar.gz"
ARCHIVE_PATH="$HOME/$ARCHIVE_NAME"
echo "[INFO] Creating archive: $ARCHIVE_PATH"
cd "$TEMP_DIR"
tar -czf "$ARCHIVE_PATH" onnxruntime* 2>/dev/null || {
    echo "[ERROR] ❌ Failed to create archive"
    rm -rf "$TEMP_DIR"
    exit 1
}

# Cleanup
rm -rf "$TEMP_DIR"

echo ""
echo "[INFO] ✅ Archive created: $ARCHIVE_PATH"
echo ""
echo "=========================================="
echo "  Next Steps"
echo "=========================================="
echo ""
echo "1. Transfer the archive to the problematic device:"
echo "   scp $ARCHIVE_PATH ledger@<problematic-device-ip>:~/"
echo ""
echo "2. On the problematic device, extract and install:"
echo "   cd ~"
echo "   tar -xzf $ARCHIVE_NAME"
echo "   cd onnxruntime-1.23.2-linux-arm64"
echo "   pip install --force-reinstall --no-deps ./onnxruntime"
echo "   pip install onnxruntime-gpu>=1.23.0  # Install metapackage"
echo ""
echo "   OR use the install script:"
echo "   bash ~/LedgerAI/setup/scripts/install_onnxruntime_1.23.2.sh"
echo ""

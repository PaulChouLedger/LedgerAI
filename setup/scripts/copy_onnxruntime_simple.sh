#!/bin/bash
# Simple script to copy onnxruntime from working device
# Run this on the WORKING device - doesn't require importing onnxruntime

set -e

echo "=========================================="
echo "  Copy onnxruntime from Working Device"
echo "=========================================="
echo ""
echo "This script should be run on the WORKING device."
echo "It will create a tarball of the onnxruntime installation."
echo ""

# Check if we're in a virtual environment
if [ -z "$VIRTUAL_ENV" ] && [ -d "$HOME/aura-env" ]; then
    echo "[INFO] Activating virtual environment..."
    source "$HOME/aura-env/bin/activate"
fi

# Find site-packages directory
SITE_PACKAGES=""
if [ -n "$VIRTUAL_ENV" ]; then
    SITE_PACKAGES="$VIRTUAL_ENV/lib/python3.10/site-packages"
elif [ -d "$HOME/aura-env" ]; then
    SITE_PACKAGES="$HOME/aura-env/lib/python3.10/site-packages"
else
    echo "[ERROR] ❌ Could not find virtual environment"
    echo "[INFO]   Please activate your virtual environment or set VIRTUAL_ENV"
    exit 1
fi

if [ ! -d "$SITE_PACKAGES" ]; then
    echo "[ERROR] ❌ Site-packages directory not found: $SITE_PACKAGES"
    exit 1
fi

echo "[INFO] Using site-packages: $SITE_PACKAGES"
echo ""

# Check if onnxruntime is installed
if [ ! -d "$SITE_PACKAGES/onnxruntime" ]; then
    echo "[ERROR] ❌ onnxruntime package not found in $SITE_PACKAGES"
    echo "[INFO]   Please install onnxruntime first"
    exit 1
fi

# Check version from dist-info
VERSION="unknown"
if [ -d "$SITE_PACKAGES/onnxruntime-1.23.2.dist-info" ]; then
    VERSION="1.23.2"
elif [ -d "$SITE_PACKAGES/onnxruntime-1.23.0.dist-info" ]; then
    VERSION="1.23.0"
else
    # Try to find any onnxruntime dist-info
    DIST_INFO=$(find "$SITE_PACKAGES" -maxdepth 1 -name "onnxruntime-*.dist-info" -type d 2>/dev/null | head -1)
    if [ -n "$DIST_INFO" ]; then
        VERSION=$(basename "$DIST_INFO" | sed 's/onnxruntime-\(.*\)\.dist-info/\1/')
    fi
fi

echo "[STEP] 1. Found onnxruntime installation..."
echo "[INFO]   Package directory: $SITE_PACKAGES/onnxruntime"
echo "[INFO]   Version: $VERSION"
echo ""

# Create archive
ARCHIVE_NAME="onnxruntime-${VERSION}-linux-arm64.tar.gz"
ARCHIVE_PATH="$HOME/$ARCHIVE_NAME"

echo "[STEP] 2. Creating archive..."
cd "$SITE_PACKAGES"

# Create temporary directory for packaging
TEMP_DIR=$(mktemp -d)
echo "[INFO]   Temporary directory: $TEMP_DIR"

# Copy onnxruntime package
echo "[INFO]   Copying onnxruntime package..."
cp -r onnxruntime "$TEMP_DIR/" || {
    echo "[ERROR] ❌ Failed to copy onnxruntime package"
    rm -rf "$TEMP_DIR"
    exit 1
}

# Copy dist-info if found
DIST_INFO=$(find . -maxdepth 1 -name "onnxruntime-*.dist-info" -type d 2>/dev/null | head -1)
if [ -n "$DIST_INFO" ]; then
    echo "[INFO]   Copying dist-info: $(basename $DIST_INFO)"
    cp -r "$DIST_INFO" "$TEMP_DIR/" || true
fi

# Create archive
echo "[INFO]   Creating archive: $ARCHIVE_PATH"
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

# Show file size
ARCHIVE_SIZE=$(du -h "$ARCHIVE_PATH" | cut -f1)
echo "[INFO]   Archive size: $ARCHIVE_SIZE"
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
echo "   cd ~/aura-env/lib/python3.10/site-packages"
echo "   cp -r ~/onnxruntime* ."
echo ""
echo "   OR install directly:"
echo "   cd ~"
echo "   tar -xzf $ARCHIVE_NAME"
echo "   cd onnxruntime-${VERSION}-linux-arm64"
echo "   pip install --force-reinstall --no-deps ./onnxruntime"
echo ""
echo "3. Verify installation:"
echo "   pip show onnxruntime"
echo ""

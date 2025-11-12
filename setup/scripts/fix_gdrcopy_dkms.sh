#!/bin/bash
# fix_gdrcopy_dkms.sh
# Fixes GDRCopy build error by installing dkms or disabling GDRCopy

set -e

JETSON_CONTAINERS_DIR="${HOME}/jetson-containers"

if [ ! -d "$JETSON_CONTAINERS_DIR" ]; then
    echo "❌ jetson-containers directory not found at $JETSON_CONTAINERS_DIR"
    echo "Please ensure jetson-containers is cloned in your home directory"
    exit 1
fi

cd "$JETSON_CONTAINERS_DIR"

echo "🔍 Searching for GDRCopy build files..."

# Find GDRCopy build/install scripts
GDRCOPY_BUILD=$(find . -path "*/build/build_gdrcopy.sh" 2>/dev/null | head -1)
GDRCOPY_INSTALL=$(find . -path "*/install/install_gdrcopy.sh" 2>/dev/null | head -1)
CUDASTACK_DOCKERFILE=$(find . -path "*/cudastack/Dockerfile" 2>/dev/null | head -1)

echo "📋 Found files:"
[ -n "$GDRCOPY_BUILD" ] && echo "  - Build script: $GDRCOPY_BUILD"
[ -n "$GDRCOPY_INSTALL" ] && echo "  - Install script: $GDRCOPY_INSTALL"
[ -n "$CUDASTACK_DOCKERFILE" ] && echo "  - Dockerfile: $CUDASTACK_DOCKERFILE"

# Strategy: Install dkms in the Dockerfile before GDRCopy build
if [ -f "$CUDASTACK_DOCKERFILE" ]; then
    echo ""
    echo "🔧 Patching Dockerfile to install dkms..."
    
    # Create backup
    cp "$CUDASTACK_DOCKERFILE" "${CUDASTACK_DOCKERFILE}.bak.$(date +%Y%m%d_%H%M%S)"
    echo "✅ Backup created: ${CUDASTACK_DOCKERFILE}.bak.*"
    
    # Check if dkms installation already exists
    if grep -q "apt-get.*install.*dkms" "$CUDASTACK_DOCKERFILE"; then
        echo "⚠️  dkms installation already present in Dockerfile"
    else
        # Add dkms installation before GDRCopy build section
        # Look for the section that handles GDRCopy
        if grep -q "WITH_GDRCOPY" "$CUDASTACK_DOCKERFILE"; then
            # Add dkms installation right before GDRCopy installation
            sed -i '/WITH_GDRCOPY.*1.*GDRCopy/i\        RUN apt-get update \&\& apt-get install -y dkms || true \\' "$CUDASTACK_DOCKERFILE"
            echo "✅ Added dkms installation to Dockerfile"
        else
            # Add at the beginning of RUN commands section
            sed -i '/^RUN /a\RUN apt-get update \&\& apt-get install -y dkms || true' "$CUDASTACK_DOCKERFILE" | head -1
            echo "✅ Added dkms installation to Dockerfile (generic location)"
        fi
    fi
fi

# Also patch build script if it exists
if [ -f "$GDRCOPY_BUILD" ]; then
    echo ""
    echo "🔧 Patching GDRCopy build script..."
    
    # Create backup
    cp "$GDRCOPY_BUILD" "${GDRCOPY_BUILD}.bak.$(date +%Y%m%d_%H%M%S)"
    
    # Check if dkms check already exists
    if grep -q "dkms" "$GDRCOPY_BUILD"; then
        echo "⚠️  dkms already referenced in build script"
    else
        # Add dkms installation at the beginning
        sed -i '1a\# Install dkms if not present\nif ! command -v dkms &> /dev/null; then\n    apt-get update && apt-get install -y dkms || true\nfi' "$GDRCOPY_BUILD"
        echo "✅ Added dkms installation check to build script"
    fi
fi

# Patch install script if it exists
if [ -f "$GDRCOPY_INSTALL" ]; then
    echo ""
    echo "🔧 Patching GDRCopy install script..."
    
    # Create backup
    cp "$GDRCOPY_INSTALL" "${GDRCOPY_INSTALL}.bak.$(date +%Y%m%d_%H%M%S)"
    
    # Check if dkms check already exists
    if grep -q "dkms" "$GDRCOPY_INSTALL"; then
        echo "⚠️  dkms already referenced in install script"
    else
        # Add dkms installation at the beginning
        sed -i '1a\# Install dkms if not present\nif ! command -v dkms &> /dev/null; then\n    apt-get update && apt-get install -y dkms || true\nfi' "$GDRCOPY_INSTALL"
        echo "✅ Added dkms installation check to install script"
    fi
fi

echo ""
echo "✅ Patch complete!"
echo ""
echo "📝 Next steps:"
echo "   1. Rebuild the container:"
echo "      cd ~/jetson-containers"
echo "      jetson-containers build unsloth:r36.4.tegra-aarch64-cu126-22.04-cudastack_standard"
echo ""
echo "   2. Or if you don't need GDRCopy, build with it disabled:"
echo "      jetson-containers build --build-arg WITH_GDRCOPY=0 unsloth:r36.4.tegra-aarch64-cu126-22.04-cudastack_standard"
echo ""
echo "💡 Tip: GDRCopy is only needed for multi-GPU setups. For single Jetson devices, you can disable it."


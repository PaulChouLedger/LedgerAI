#!/bin/bash
# build_unsloth_without_gdrcopy.sh
# Builds unsloth container without GDRCopy (recommended for single-GPU Jetson)

set -e

JETSON_CONTAINERS_DIR="${HOME}/jetson-containers"

if [ ! -d "$JETSON_CONTAINERS_DIR" ]; then
    echo "❌ jetson-containers directory not found at $JETSON_CONTAINERS_DIR"
    echo "Please ensure jetson-containers is cloned in your home directory"
    exit 1
fi

cd "$JETSON_CONTAINERS_DIR"

echo "🚀 Building unsloth container WITHOUT GDRCopy..."
echo "   (GDRCopy is only needed for multi-GPU setups)"
echo ""

# Build with GDRCopy disabled
jetson-containers build \
  --tag unsloth:r36.4.tegra-aarch64-cu126-22.04-cudastack_standard \
  --build-arg WITH_GDRCOPY=0 \
  unsloth:r36.4.tegra-aarch64-cu126-22.04-cudastack_standard

echo ""
echo "✅ Build complete!"
echo "   Image: unsloth:r36.4.tegra-aarch64-cu126-22.04-cudastack_standard"


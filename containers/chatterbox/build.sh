#!/bin/bash
# Build script for Chatterbox container using standard Docker (avoids PyTorch source compilation)
# This script builds the container directly using the Dockerfile, which uses a pre-built PyTorch image

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=========================================="
echo "  Building Chatterbox-TTS Container"
echo "=========================================="
echo ""
echo "This build uses a pre-built PyTorch image to avoid source compilation."
echo "Building from: $SCRIPT_DIR"
echo ""

# Check if Docker is available
if ! command -v docker &> /dev/null; then
    echo "❌ Error: Docker is not installed or not in PATH"
    exit 1
fi

# Check if NVIDIA Docker runtime is available (for Jetson)
if docker info 2>/dev/null | grep -q "nvidia"; then
    echo "✅ NVIDIA Docker runtime detected"
    USE_NVIDIA_RUNTIME="--runtime=nvidia"
else
    echo "⚠️  NVIDIA Docker runtime not detected - GPU support may be limited"
    USE_NVIDIA_RUNTIME=""
fi

# Build arguments
BUILD_ARGS=(
    "--network=host"
    "--shm-size=8g"
    "-t" "chatterbox-tts:latest"
    "-f" "Dockerfile"
)

# Optional: Skip model download to speed up build
if [ "${SKIP_MODEL_DOWNLOAD:-0}" = "1" ]; then
    BUILD_ARGS+=("--build-arg" "SKIP_MODEL_DOWNLOAD=1")
    echo "ℹ️  Model download will be skipped (models will download at runtime)"
fi

# Build the container
echo "🔨 Building container..."
echo "Command: docker build ${BUILD_ARGS[*]} ."
echo ""

if docker build "${BUILD_ARGS[@]}" .; then
    echo ""
    echo "=========================================="
    echo "✅ Build completed successfully!"
    echo "=========================================="
    echo ""
    echo "To run the container:"
    echo "  docker run -d --name chatterbox-tts $USE_NVIDIA_RUNTIME --network=host \\"
    echo "    -v \$(pwd)/../assets/voice_samples:/app/voice_samples \\"
    echo "    -v \$(pwd)/../data/voice_cache:/app/voice_cache \\"
    echo "    chatterbox-tts:latest"
    echo ""
    echo "Or use docker-compose:"
    echo "  cd ../setup && docker compose up -d chatterbox-tts"
    echo ""
else
    echo ""
    echo "=========================================="
    echo "❌ Build failed!"
    echo "=========================================="
    echo ""
    echo "Troubleshooting:"
    echo "1. Ensure Docker has enough disk space (container needs ~10-15GB)"
    echo "2. Check Docker logs: docker logs <container-id>"
    echo "3. Try building with model download skipped: SKIP_MODEL_DOWNLOAD=1 ./build.sh"
    echo "4. Ensure NVIDIA Docker runtime is installed for Jetson devices"
    exit 1
fi

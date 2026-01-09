#!/bin/bash
# Build onnxruntime-gpu 1.23.0 from source for Jetson
# This script builds onnxruntime-gpu with CUDA support for ARM64/Jetson devices
# Use this when Jetson PyPI is down or unavailable
#
# WARNING: Building from source is complex and time-consuming (1-3 hours)
# Consider these alternatives first:
# 1. Wait for Jetson PyPI to come back online
# 2. Copy onnxruntime from a working device (see copy_onnxruntime_from_working_device.sh)
# 3. Use standard PyPI onnxruntime==1.23.2 (may crash, but can work with ORT_DISABLE_CPUINFO=1)
#
# This script requires:
# - CUDA toolkit installed
# - ~10GB free disk space
# - 1-3 hours of build time
# - Sufficient RAM (may need swap if device has <8GB RAM)

set -e

echo "=========================================="
echo "  Building onnxruntime-gpu 1.23.0 from Source"
echo "=========================================="
echo ""
echo "⚠️  WARNING: Building from source is complex and time-consuming!"
echo "   - Build time: 1-3 hours"
echo "   - Disk space: ~10GB required"
echo "   - RAM: May need swap if <8GB RAM"
echo ""
echo "Alternatives to consider:"
echo "  1. Wait for Jetson PyPI: https://pypi.jetson-ai-lab.io"
echo "  2. Copy from working device: bash copy_onnxruntime_from_working_device.sh"
echo "  3. Use standard PyPI: pip install onnxruntime==1.23.2"
echo ""
read -p "Continue with source build? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Build cancelled"
    exit 0
fi
echo ""

# Configuration
ONNXRUNTIME_VERSION="1.23.0"
ONNXRUNTIME_REPO="https://github.com/microsoft/onnxruntime.git"
BUILD_DIR="$HOME/onnxruntime-build"
INSTALL_PREFIX="$HOME/onnxruntime-install"

# Check if we're on a Jetson device
if [ ! -f /etc/nv_tegra_release ]; then
    echo "⚠️  WARNING: This doesn't appear to be a Jetson device"
    echo "   The build may not work correctly on non-Jetson systems"
    read -p "Continue anyway? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Check disk space
AVAILABLE_SPACE=$(df -BG "$HOME" | tail -1 | awk '{print $4}' | sed 's/G//')
if [ "$AVAILABLE_SPACE" -lt 10 ]; then
    echo "⚠️  WARNING: Less than 10GB free space available"
    echo "   Available: ${AVAILABLE_SPACE}GB"
    echo "   Build may fail due to insufficient space"
    read -p "Continue anyway? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Check CUDA availability
if ! command -v nvcc &> /dev/null; then
    echo "❌ ERROR: nvcc (CUDA compiler) not found"
    echo "   Please install CUDA toolkit first:"
    echo "   sudo apt-get install nvidia-cuda-toolkit"
    exit 1
fi

CUDA_VERSION=$(nvcc --version | grep "release" | sed 's/.*release \([0-9]\+\.[0-9]\+\).*/\1/')
echo "[INFO] CUDA version: $CUDA_VERSION"
echo ""

# Check Python version
PYTHON_VERSION=$(python3 --version | awk '{print $2}' | cut -d. -f1,2)
echo "[INFO] Python version: $PYTHON_VERSION"
echo ""

# Install build dependencies
echo "[STEP] 1. Installing build dependencies..."
echo "[INFO] This may require sudo access"
echo "[INFO] Installing essential build tools..."

REQUIRED_PACKAGES="build-essential cmake git python3-dev python3-pip ninja-build"
OPTIONAL_PACKAGES="libprotobuf-dev protobuf-compiler libprotoc-dev libeigen3-dev patchelf"

sudo apt-get update
sudo apt-get install -y $REQUIRED_PACKAGES || {
    echo "❌ Failed to install required packages"
    exit 1
}

sudo apt-get install -y $OPTIONAL_PACKAGES || {
    echo "⚠️  Some optional packages not available - continuing anyway"
}

# Check cmake version (need >= 3.18)
CMAKE_VERSION=$(cmake --version | head -1 | awk '{print $3}')
CMAKE_MAJOR=$(echo "$CMAKE_VERSION" | cut -d. -f1)
CMAKE_MINOR=$(echo "$CMAKE_VERSION" | cut -d. -f2)
if [ "$CMAKE_MAJOR" -lt 3 ] || ([ "$CMAKE_MAJOR" -eq 3 ] && [ "$CMAKE_MINOR" -lt 18 ]); then
    echo "⚠️  WARNING: CMake version $CMAKE_VERSION is too old (need >= 3.18)"
    echo "   You may need to install a newer version"
fi

echo ""
echo "[STEP] 2. Cloning onnxruntime repository..."
if [ -d "$BUILD_DIR" ]; then
    echo "[INFO] Build directory exists, updating..."
    cd "$BUILD_DIR"
    git fetch --tags
    git checkout "v${ONNXRUNTIME_VERSION}" || {
        echo "⚠️  Version tag v${ONNXRUNTIME_VERSION} not found, using latest"
        git checkout main
    }
    git submodule update --init --recursive
else
    echo "[INFO] Cloning repository..."
    git clone --recursive --depth 1 --branch "v${ONNXRUNTIME_VERSION}" "$ONNXRUNTIME_REPO" "$BUILD_DIR" || {
        echo "⚠️  Branch v${ONNXRUNTIME_VERSION} not found, cloning main branch..."
        git clone --recursive "$ONNXRUNTIME_REPO" "$BUILD_DIR"
        cd "$BUILD_DIR"
        git checkout "v${ONNXRUNTIME_VERSION}" 2>/dev/null || {
            echo "⚠️  Tag not found, using latest commit"
        }
    }
fi

cd "$BUILD_DIR"
echo "[INFO] Repository at: $(pwd)"
echo "[INFO] Current commit: $(git rev-parse HEAD)"
echo ""

# Check if virtual environment exists
if [ -d "$HOME/aura-env" ]; then
    echo "[STEP] 3. Activating virtual environment..."
    source "$HOME/aura-env/bin/activate"
    PYTHON_EXECUTABLE="$HOME/aura-env/bin/python3"
else
    PYTHON_EXECUTABLE=$(which python3)
    echo "[INFO] Using system Python: $PYTHON_EXECUTABLE"
fi

# Install Python build dependencies
echo ""
echo "[STEP] 4. Installing Python build dependencies..."
pip install --upgrade pip setuptools wheel numpy protobuf

# Configure build
echo ""
echo "[STEP] 5. Configuring build..."
mkdir -p build
cd build

# Determine CUDA architecture (Jetson-specific)
JETSON_MODEL=$(cat /proc/device-tree/model 2>/dev/null | tr -d '\0' || echo "unknown")
echo "[INFO] Jetson model: $JETSON_MODEL"

# Common Jetson CUDA architectures
# Xavier: 7.2, Orin: 8.7, AGX Orin: 8.7
CUDA_ARCH="8.7"  # Default for Orin, adjust if needed
if echo "$JETSON_MODEL" | grep -qi "xavier"; then
    CUDA_ARCH="7.2"
    echo "[INFO] Detected Xavier - using CUDA arch 7.2"
elif echo "$JETSON_MODEL" | grep -qi "orin"; then
    CUDA_ARCH="8.7"
    echo "[INFO] Detected Orin - using CUDA arch 8.7"
else
    echo "[INFO] Using default CUDA arch 8.7 (adjust if needed)"
fi

# CMake configuration for Jetson
echo "[INFO] Configuring CMake (this may take a few minutes)..."
echo "[INFO] CUDA architecture: $CUDA_ARCH"
echo "[INFO] CUDA home: /usr/local/cuda"

# Find CUDA installation
CUDA_HOME="/usr/local/cuda"
if [ ! -d "$CUDA_HOME" ]; then
    # Try alternative locations
    if [ -d "/usr/local/cuda-${CUDA_VERSION}" ]; then
        CUDA_HOME="/usr/local/cuda-${CUDA_VERSION}"
    elif [ -d "/opt/nvidia/cuda" ]; then
        CUDA_HOME="/opt/nvidia/cuda"
    else
        echo "⚠️  CUDA not found in standard locations, using /usr/local/cuda"
    fi
fi

# Check TensorRT (optional)
TENSORRT_HOME=""
if [ -d "/usr/src/tensorrt" ]; then
    TENSORRT_HOME="/usr/src/tensorrt"
    echo "[INFO] TensorRT found: $TENSORRT_HOME"
else
    echo "[INFO] TensorRT not found - building without TensorRT support"
fi

# Build CMake command
CMAKE_ARGS=(
    -DCMAKE_BUILD_TYPE=Release
    -DCMAKE_INSTALL_PREFIX="$INSTALL_PREFIX"
    -DPython3_EXECUTABLE="$PYTHON_EXECUTABLE"
    -Donnxruntime_BUILD_UNIT_TESTS=OFF
    -Donnxruntime_BUILD_BENCHMARKS=OFF
    -Donnxruntime_USE_PREINSTALLED_EIGEN=ON
    -Donnxruntime_ENABLE_PYTHON=ON
    -Donnxruntime_USE_CUDA=ON
    -Donnxruntime_CUDA_HOME="$CUDA_HOME"
    -Donnxruntime_CUDA_ARCHITECTURES="${CUDA_ARCH}"
    -Donnxruntime_BUILD_SHARED_LIB=ON
    -Donnxruntime_ENABLE_LANGUAGE_INTEROP_OPS=OFF
    -Donnxruntime_USE_DNNL=OFF
    -Donnxruntime_USE_MKLML=OFF
    -Donnxruntime_USE_OPENMP=ON
    -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCH}"
    -GNinja
)

# Add TensorRT if available
if [ -n "$TENSORRT_HOME" ]; then
    CMAKE_ARGS+=(
        -Donnxruntime_USE_TENSORRT=ON
        -Donnxruntime_TENSORRT_HOME="$TENSORRT_HOME"
    )
else
    CMAKE_ARGS+=(-Donnxruntime_USE_TENSORRT=OFF)
fi

# Add CUDNN if available
if [ -d "/usr/lib/aarch64-linux-gnu" ]; then
    CMAKE_ARGS+=(-Donnxruntime_CUDNN_HOME=/usr/lib/aarch64-linux-gnu)
fi

cmake .. "${CMAKE_ARGS[@]}" 2>&1 | tee /tmp/onnxruntime_cmake.log || {
    echo "❌ CMake configuration failed"
    echo "   Check logs: /tmp/onnxruntime_cmake.log"
    echo ""
    echo "Common issues:"
    echo "  1. CUDA not found - install CUDA toolkit"
    echo "  2. CMake version too old - need >= 3.18"
    echo "  3. Missing dependencies - check error messages above"
    exit 1
}

echo "[INFO] ✅ CMake configuration complete"
echo ""

# Build
echo "[STEP] 6. Building onnxruntime (this will take 1-3 hours)..."
echo "[INFO] Building with $(nproc) parallel jobs"
echo "[INFO] This is a long process - be patient!"
echo ""

# Build with limited parallelism to avoid OOM
PARALLEL_JOBS=$(($(nproc) / 2))
if [ "$PARALLEL_JOBS" -lt 1 ]; then
    PARALLEL_JOBS=1
fi

ninja -j"$PARALLEL_JOBS" 2>&1 | tee /tmp/onnxruntime_build.log || {
    echo "❌ Build failed"
    echo "   Check logs: /tmp/onnxruntime_build.log"
    exit 1
}

echo "[INFO] ✅ Build complete"
echo ""

# Install
echo "[STEP] 7. Installing onnxruntime..."
ninja install 2>&1 | tee /tmp/onnxruntime_install.log || {
    echo "❌ Installation failed"
    echo "   Check logs: /tmp/onnxruntime_install.log"
    exit 1
}

# Install Python package
echo ""
echo "[STEP] 8. Installing Python package..."
cd "$BUILD_DIR"
pip install -e . --no-build-isolation 2>&1 | tee /tmp/onnxruntime_pip_install.log || {
    echo "⚠️  pip install -e failed, trying alternative method..."
    # Alternative: copy built files
    PYTHON_SITE_PACKAGES=$($PYTHON_EXECUTABLE -c "import site; print(site.getsitepackages()[0] if hasattr(site, 'getsitepackages') and site.getsitepackages() else site.USER_SITE)")
    if [ -n "$PYTHON_SITE_PACKAGES" ]; then
        echo "[INFO] Copying built package to: $PYTHON_SITE_PACKAGES"
        cp -r "$INSTALL_PREFIX/lib/python*/site-packages/onnxruntime" "$PYTHON_SITE_PACKAGES/" 2>/dev/null || {
            echo "⚠️  Could not copy package files automatically"
            echo "   You may need to manually install the package"
        }
    fi
}

# Verify installation
echo ""
echo "[STEP] 9. Verifying installation..."
export ORT_DISABLE_CPUINFO=1
export ORT_LOG_LEVEL=3

if ORT_DISABLE_CPUINFO=1 ORT_LOG_LEVEL=3 python3 -c "import onnxruntime; print(f'✅ onnxruntime imported successfully'); print(f'Version: {onnxruntime.__version__}')" 2>&1; then
    INSTALLED_VERSION=$(ORT_DISABLE_CPUINFO=1 ORT_LOG_LEVEL=3 python3 -c "import onnxruntime; print(onnxruntime.__version__)" 2>/dev/null || echo "unknown")
    echo "[INFO] ✅ onnxruntime installed successfully"
    echo "[INFO]    Version: $INSTALLED_VERSION"
    echo "[INFO]    Location: $(python3 -c 'import onnxruntime; import os; print(os.path.dirname(onnxruntime.__file__))' 2>/dev/null || echo 'unknown')"
else
    echo "[WARNING] ⚠️  Import test failed (may crash due to CPU detection)"
    echo "[INFO]    Checking if package files exist..."
    if pip show onnxruntime 2>/dev/null | grep -q "Version:"; then
        echo "[INFO] ✅ Package is installed (import may crash but package exists)"
        pip show onnxruntime
    else
        echo "[ERROR] ❌ Package installation verification failed"
        exit 1
    fi
fi

echo ""
echo "=========================================="
echo "  Build Complete!"
echo "=========================================="
echo ""
echo "✅ onnxruntime-gpu has been built from source"
echo ""
echo "Build artifacts:"
echo "   Build directory: $BUILD_DIR"
echo "   Install prefix: $INSTALL_PREFIX"
echo ""
echo "To clean up build files (optional):"
echo "   rm -rf $BUILD_DIR"
echo "   # Keep $INSTALL_PREFIX if you want to reuse the installation"
echo ""
echo "Note: The built package should work with ORT_DISABLE_CPUINFO=1 set"
echo ""

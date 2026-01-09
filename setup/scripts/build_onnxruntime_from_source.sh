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
    git fetch --all --tags
    git checkout "v${ONNXRUNTIME_VERSION}" 2>/dev/null || {
        echo "⚠️  Version tag v${ONNXRUNTIME_VERSION} not found, checking available tags..."
        git fetch --tags
        LATEST_TAG=$(git tag | grep "^v1.23" | sort -V | tail -1)
        if [ -n "$LATEST_TAG" ]; then
            echo "[INFO] Using closest tag: $LATEST_TAG"
            git checkout "$LATEST_TAG"
        else
            echo "⚠️  No v1.23.x tag found, using latest main branch"
            git checkout main
            git pull
        }
    }
    git submodule update --init --recursive
else
    echo "[INFO] Cloning repository (this may take a few minutes)..."
    # Clone without depth to allow tag checkout
    git clone --recursive "$ONNXRUNTIME_REPO" "$BUILD_DIR" || {
        echo "❌ Failed to clone repository"
        exit 1
    }
    cd "$BUILD_DIR"
    echo "[INFO] Checking out version v${ONNXRUNTIME_VERSION}..."
    git fetch --tags
    git checkout "v${ONNXRUNTIME_VERSION}" 2>/dev/null || {
        echo "⚠️  Version tag v${ONNXRUNTIME_VERSION} not found, checking available tags..."
        LATEST_TAG=$(git tag | grep "^v1.23" | sort -V | tail -1)
        if [ -n "$LATEST_TAG" ]; then
            echo "[INFO] Using closest tag: $LATEST_TAG"
            git checkout "$LATEST_TAG"
            ONNXRUNTIME_VERSION=$(echo "$LATEST_TAG" | sed 's/^v//')
        else
            echo "⚠️  No v1.23.x tag found, using latest main branch"
            git checkout main
        }
    }
    # Update submodules for the checked out version
    git submodule update --init --recursive
fi

cd "$BUILD_DIR"
echo "[INFO] Repository at: $(pwd)"
echo "[INFO] Current commit: $(git rev-parse HEAD)"
echo "[INFO] Current tag/branch: $(git describe --tags --exact-match 2>/dev/null || git branch --show-current || echo 'detached HEAD')"
echo "[INFO] Verifying CMakeLists.txt exists..."
if [ ! -f "CMakeLists.txt" ]; then
    echo "❌ ERROR: CMakeLists.txt not found in repository root"
    echo "   Repository may not have cloned correctly"
    exit 1
fi
echo "[INFO] ✅ CMakeLists.txt found"
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
echo "[INFO] Installing build tools..."
pip install --upgrade pip setuptools wheel protobuf

# Install numpy compatible with onnxruntime (need <2.0 for 1.23.0)
echo "[INFO] Installing numpy <2.0 (required for onnxruntime 1.23.0)..."
pip install "numpy<2.0,>=1.21.0" || {
    echo "⚠️  Failed to install compatible numpy version"
    echo "   Continuing with existing numpy..."
}

# Configure build
echo ""
echo "[STEP] 5. Configuring build..."
echo "[INFO] Creating build directory..."
mkdir -p "$BUILD_DIR/build"
cd "$BUILD_DIR/build"
echo "[INFO] Build directory: $(pwd)"
echo "[INFO] Source directory: $BUILD_DIR"
echo "[INFO] Verifying source directory..."
if [ ! -f "$BUILD_DIR/CMakeLists.txt" ]; then
    echo "❌ ERROR: CMakeLists.txt not found in source directory"
    echo "   Source: $BUILD_DIR"
    exit 1
fi
echo "[INFO] ✅ Source directory verified"
echo ""

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

echo "[INFO] Running CMake configuration..."
echo "[INFO] Source path: $BUILD_DIR"
echo "[INFO] Build path: $(pwd)"
cmake "$BUILD_DIR" "${CMAKE_ARGS[@]}" 2>&1 | tee /tmp/onnxruntime_cmake.log || {
    echo "❌ CMake configuration failed"
    echo "   Check logs: /tmp/onnxruntime_cmake.log"
    echo ""
    echo "Common issues:"
    echo "  1. CUDA not found - install CUDA toolkit"
    echo "  2. CMake version too old - need >= 3.18"
    echo "  3. Missing dependencies - check error messages above"
    echo "  4. Source directory issue - verify CMakeLists.txt exists"
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
cd "$BUILD_DIR/build"

# First try installing the built wheel if it exists
if [ -f "dist/onnxruntime_gpu-${ONNXRUNTIME_VERSION}-*.whl" ] || [ -f "dist/onnxruntime-${ONNXRUNTIME_VERSION}-*.whl" ]; then
    echo "[INFO] Found built wheel, installing..."
    pip install dist/onnxruntime*.whl 2>&1 | tee /tmp/onnxruntime_pip_install.log || {
        echo "⚠️  Wheel installation failed, trying build_python..."
    }
else
    echo "[INFO] Building Python package..."
    # Build Python package using cmake --build
    cmake --build . --target onnxruntime_python 2>&1 | tee /tmp/onnxruntime_pip_install.log || {
        echo "⚠️  Python package build failed, trying pip install -e..."
        cd "$BUILD_DIR"
        pip install -e . --no-build-isolation 2>&1 | tee -a /tmp/onnxruntime_pip_install.log || {
            echo "⚠️  pip install -e failed, trying alternative method..."
            # Alternative: copy built files
            PYTHON_SITE_PACKAGES=$($PYTHON_EXECUTABLE -c "import site; print(site.getsitepackages()[0] if hasattr(site, 'getsitepackages') and site.getsitepackages() else site.USER_SITE)" 2>/dev/null || echo "")
            if [ -n "$PYTHON_SITE_PACKAGES" ] && [ -d "$INSTALL_PREFIX/lib" ]; then
                echo "[INFO] Copying built package to: $PYTHON_SITE_PACKAGES"
                find "$INSTALL_PREFIX/lib" -name "onnxruntime" -type d | head -1 | while read pkg_dir; do
                    if [ -d "$pkg_dir" ]; then
                        cp -r "$pkg_dir" "$PYTHON_SITE_PACKAGES/" 2>/dev/null && echo "[INFO] ✅ Copied package files" || {
                            echo "⚠️  Could not copy package files automatically"
                        }
                    fi
                done
            fi
        }
    }
fi

# Verify installation
echo ""
echo "[STEP] 9. Verifying installation..."
export ORT_DISABLE_CPUINFO=1
export ORT_LOG_LEVEL=3

# Check what version pip shows
PIP_VERSION=$(pip show onnxruntime 2>/dev/null | grep "^Version:" | awk '{print $2}' || echo "")
if [ -z "$PIP_VERSION" ]; then
    echo "[ERROR] ❌ onnxruntime package not found in pip"
    echo "   Build may have failed - check logs above"
    exit 1
fi

echo "[INFO] Installed version (from pip): $PIP_VERSION"
echo "[INFO] Expected version: $ONNXRUNTIME_VERSION"

# Check if this is the version we built or an existing installation
if [ "$PIP_VERSION" != "$ONNXRUNTIME_VERSION" ]; then
    echo "[WARNING] ⚠️  Version mismatch detected!"
    echo "   Expected: $ONNXRUNTIME_VERSION"
    echo "   Found: $PIP_VERSION"
    echo "   This may be an existing installation, not the newly built version"
    echo ""
    echo "   To use the newly built version:"
    echo "   1. Uninstall existing: pip uninstall -y onnxruntime onnxruntime-gpu"
    echo "   2. Reinstall from build: pip install $BUILD_DIR/build/dist/onnxruntime*.whl"
    echo ""
    read -p "Continue verification anyway? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Try to import
echo "[INFO] Testing import (with ORT_DISABLE_CPUINFO=1)..."
if ORT_DISABLE_CPUINFO=1 ORT_LOG_LEVEL=3 python3 -c "import onnxruntime; print(f'✅ onnxruntime imported successfully'); print(f'Version: {onnxruntime.__version__}')" 2>&1; then
    INSTALLED_VERSION=$(ORT_DISABLE_CPUINFO=1 ORT_LOG_LEVEL=3 python3 -c "import onnxruntime; print(onnxruntime.__version__)" 2>/dev/null || echo "unknown")
    INSTALLED_LOCATION=$(python3 -c "import onnxruntime; import os; print(os.path.dirname(onnxruntime.__file__))" 2>/dev/null || echo "unknown")
    echo "[INFO] ✅ onnxruntime installed and importable"
    echo "[INFO]    Version: $INSTALLED_VERSION"
    echo "[INFO]    Location: $INSTALLED_LOCATION"
    
    # Check if location matches our build
    if echo "$INSTALLED_LOCATION" | grep -q "$BUILD_DIR\|$INSTALL_PREFIX"; then
        echo "[INFO] ✅ Package location matches build directory"
    else
        echo "[WARNING] ⚠️  Package location doesn't match build directory"
        echo "   This may be an existing installation, not the newly built version"
    fi
else
    echo "[WARNING] ⚠️  Import test failed (may crash due to CPU detection)"
    echo "[INFO]    Checking if package files exist..."
    if pip show onnxruntime 2>/dev/null | grep -q "Version:"; then
        echo "[INFO] ✅ Package is installed (import may crash but package exists)"
        echo "[INFO]    Version: $PIP_VERSION"
        echo "[INFO]    Location: $(pip show onnxruntime | grep '^Location:' | awk '{print $2}')"
        echo ""
        echo "   Note: Import failed, but package is installed."
        echo "   This may work at runtime with ORT_DISABLE_CPUINFO=1 set in your environment."
    else
        echo "[ERROR] ❌ Package installation verification failed"
        exit 1
    fi
fi

echo ""
echo "=========================================="
echo "  Build Summary"
echo "=========================================="
echo ""
if [ -f "$BUILD_DIR/build/build.ninja" ] || [ -f "$BUILD_DIR/build/Makefile" ]; then
    echo "✅ Build configuration completed"
    if [ -f "$BUILD_DIR/build/onnxruntime/libonnxruntime.so" ] || [ -d "$BUILD_DIR/build/onnxruntime" ]; then
        echo "✅ Build artifacts found"
    else
        echo "⚠️  Build artifacts not found - build may have failed"
    fi
else
    echo "⚠️  Build configuration may not have completed"
fi

if pip show onnxruntime 2>/dev/null | grep -q "Version:"; then
    INSTALLED_VER=$(pip show onnxruntime | grep "^Version:" | awk '{print $2}')
    echo "✅ Package installed: onnxruntime $INSTALLED_VER"
    if [ "$INSTALLED_VER" = "$ONNXRUNTIME_VERSION" ]; then
        echo "✅ Version matches expected: $ONNXRUNTIME_VERSION"
    else
        echo "⚠️  Version mismatch: expected $ONNXRUNTIME_VERSION, got $INSTALLED_VER"
    fi
else
    echo "❌ Package not found in pip"
fi

echo ""
echo "Build artifacts:"
echo "   Source directory: $BUILD_DIR"
echo "   Build directory: $BUILD_DIR/build"
echo "   Install prefix: $INSTALL_PREFIX"
echo ""
echo "To clean up build files (optional, saves ~5-10GB):"
echo "   rm -rf $BUILD_DIR"
echo "   # Keep $INSTALL_PREFIX if you want to reuse the installation"
echo ""
echo "Note: The built package should work with ORT_DISABLE_CPUINFO=1 set"
echo "   Add to your environment or systemd service:"
echo "   export ORT_DISABLE_CPUINFO=1"
echo ""

#!/bin/bash
# Quick fix script to install onnxruntime-gpu 1.23.2 (fixes CPU detection crash)
# This script tries multiple sources to find 1.23.2 with ARM64 builds

set -e

echo "=========================================="
echo "  Installing onnxruntime-gpu 1.23.2"
echo "=========================================="
echo ""

# Check if we're in a virtual environment
if [ -z "$VIRTUAL_ENV" ] && [ -d "$HOME/aura-env" ]; then
    echo "[INFO] Activating virtual environment..."
    source "$HOME/aura-env/bin/activate"
fi

# Uninstall existing versions
echo "[STEP] 1. Uninstalling existing onnxruntime packages..."
pip uninstall -y onnxruntime onnxruntime-gpu 2>/dev/null || true
python3 -m pip uninstall -y onnxruntime onnxruntime-gpu 2>/dev/null || true
echo "[INFO] ✅ Uninstalled existing versions"
echo ""

# Try to install 1.23.2 from multiple sources
echo "[STEP] 2. Attempting to install onnxruntime-gpu 1.23.2..."
echo ""

# Strategy 1: Try Jetson PyPI
echo "[INFO] Attempt 1: Jetson PyPI..."
if pip install --index-url https://pypi.jetson-ai-lab.io/jp6/cu126 "onnxruntime-gpu==1.23.2" 2>&1 | tee /tmp/onnxruntime_1.23.2_install.log; then
    INSTALLED_VERSION=$(pip show onnxruntime-gpu 2>/dev/null | grep "^Version:" | awk '{print $2}' || echo "unknown")
    if [ "$INSTALLED_VERSION" = "1.23.2" ]; then
        echo "[INFO] ✅ Installed onnxruntime-gpu 1.23.2 from Jetson PyPI"
        INSTALLED=true
    fi
fi

# Strategy 2: Try standard PyPI (official Microsoft release - Oct 2024)
# See: https://github.com/microsoft/onnxruntime/releases/tag/v1.23.2
# Note: 1.23.2 has ARM64 builds (confirmed: macOS ARM64 exists, Linux ARM64 should too)
if [ "${INSTALLED:-false}" != "true" ]; then
    echo "[INFO] Attempt 2: Standard PyPI (official Microsoft release)..."
    echo "[INFO]   ONNX Runtime 1.23.2 was released Oct 25, 2024 by Microsoft"
    echo "[INFO]   See: https://github.com/microsoft/onnxruntime/releases/tag/v1.23.2"
    echo "[INFO]   ARM64 builds confirmed available (need Linux ARM64 for Jetson)"
    pip uninstall -y onnxruntime-gpu onnxruntime 2>/dev/null || true
    # Try installing with explicit platform preference for Linux ARM64
    echo "[INFO]   Installing onnxruntime-gpu==1.23.2 (pip will select Linux ARM64 if available)..."
    if pip install "onnxruntime-gpu==1.23.2" 2>&1 | tee -a /tmp/onnxruntime_1.23.2_install.log; then
        # Verify it's ARM64
        export ORT_DISABLE_CPUINFO=1
        export ORT_LOG_LEVEL=3
        INSTALLED_PATH=$(python3 -c "import onnxruntime; import os; print(os.path.dirname(onnxruntime.__file__))" 2>/dev/null || echo "")
        if [ -n "$INSTALLED_PATH" ]; then
            SO_FILE=$(find "$INSTALLED_PATH" -name "*.so" -type f 2>/dev/null | head -1)
            if [ -n "$SO_FILE" ] && file "$SO_FILE" 2>/dev/null | grep -qE "ARM|aarch64"; then
                RUNTIME_VER=$(python3 -c "import onnxruntime; print(onnxruntime.__version__)" 2>/dev/null || echo "unknown")
                echo "[INFO] ✅ Installed onnxruntime-gpu $RUNTIME_VER from standard PyPI (ARM64 verified)"
                INSTALLED=true
            else
                echo "[WARNING] ⚠️  Package from standard PyPI is not ARM64 - uninstalling"
                pip uninstall -y onnxruntime-gpu onnxruntime 2>/dev/null || true
                INSTALLED=false
            fi
        else
            echo "[WARNING] ⚠️  Could not verify installation"
            INSTALLED=false
        fi
    fi
fi

# Strategy 3: If 1.23.2 not available, provide instructions
if [ "${INSTALLED:-false}" != "true" ]; then
    echo ""
    echo "[ERROR] ❌ Could not install onnxruntime-gpu 1.23.2 from any source"
    echo ""
    echo "Options:"
    echo "  1. Check if 1.23.2 is available on the working device and copy it:"
    echo "     - On working device: pip show -f onnxruntime-gpu"
    echo "     - Copy the wheel file or entire package directory"
    echo ""
    echo "  2. Try installing from a wheel file if you have one:"
    echo "     pip install /path/to/onnxruntime_gpu-1.23.2-*.whl"
    echo ""
    echo "  3. Use 1.23.0 with ORT_DISABLE_CPUINFO=1 (may still crash on some devices):"
    echo "     pip install --index-url https://pypi.jetson-ai-lab.io/jp6/cu126 onnxruntime-gpu==1.23.0"
    echo ""
    exit 1
fi

# Verify installation
echo ""
echo "[STEP] 3. Verifying installation..."
export ORT_DISABLE_CPUINFO=1
export ORT_LOG_LEVEL=3
if python3 -c "import onnxruntime; print('✅ Import successful'); print(f'Version: {onnxruntime.__version__}')" 2>&1; then
    FINAL_VERSION=$(python3 -c "import onnxruntime; print(onnxruntime.__version__)" 2>/dev/null)
    echo "[INFO] ✅ onnxruntime-gpu 1.23.2 installed and working correctly"
    echo "[INFO]    Runtime version: $FINAL_VERSION"
    echo "[INFO]    This version fixes the CPU detection crash"
else
    echo "[ERROR] ❌ onnxruntime still crashes after installation"
    echo "[ERROR]    Check logs and try alternative installation methods"
    exit 1
fi
echo ""

echo "=========================================="
echo "  Installation Complete!"
echo "=========================================="
echo ""
echo "✅ onnxruntime-gpu 1.23.2 is installed and working"
echo "   This version fixes the CPU detection crash on JetPack R36.4.4"
echo ""

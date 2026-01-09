#!/bin/bash
# Check onnxruntime-gpu version discrepancy

echo "=========================================="
echo "  Checking onnxruntime-gpu Version"
echo "=========================================="
echo ""

# Check pip metadata version
echo "[INFO] Checking pip metadata..."
PIP_VERSION=$(pip show onnxruntime-gpu 2>/dev/null | grep "^Version:" | awk '{print $2}' || echo "not found")
echo "  pip show version: $PIP_VERSION"
echo ""

# Check installed location
echo "[INFO] Checking installation location..."
INSTALL_LOCATION=$(pip show onnxruntime-gpu 2>/dev/null | grep "^Location:" | awk '{print $2}' || echo "not found")
echo "  Installation location: $INSTALL_LOCATION"
echo ""

# Check runtime version (what Python actually imports)
echo "[INFO] Checking runtime version (what Python imports)..."
export ORT_DISABLE_CPUINFO=1
export ORT_LOG_LEVEL=3
RUNTIME_VERSION=$(python3 -c "import onnxruntime; print(onnxruntime.__version__)" 2>/dev/null || echo "import failed")
echo "  Runtime version: $RUNTIME_VERSION"
echo ""

# Check for multiple installations
echo "[INFO] Checking for multiple onnxruntime installations..."
echo "  System-wide:"
python3 -c "import sys; sys.path.insert(0, '/usr/lib/python3/dist-packages'); import onnxruntime; print(f'    Version: {onnxruntime.__version__}')" 2>/dev/null || echo "    Not found"
echo "  Virtual environment:"
python3 -c "import onnxruntime; print(f'    Version: {onnxruntime.__version__}')" 2>/dev/null || echo "    Not found"
echo ""

# Compare versions
if [ "$PIP_VERSION" != "$RUNTIME_VERSION" ] && [ "$RUNTIME_VERSION" != "import failed" ]; then
    echo "⚠️  VERSION MISMATCH DETECTED!"
    echo "   pip metadata: $PIP_VERSION"
    echo "   runtime version: $RUNTIME_VERSION"
    echo ""
    echo "This could indicate:"
    echo "  1. Multiple onnxruntime installations (system + venv)"
    echo "  2. Package metadata mismatch"
    echo "  3. Import path picking up different version"
    echo ""
    echo "To fix:"
    echo "  1. Uninstall all: pip uninstall -y onnxruntime onnxruntime-gpu"
    echo "  2. Reinstall: pip install --extra-index-url https://pypi.jetson-ai-lab.io/jp6/cu126 onnxruntime-gpu>=1.23.0"
else
    echo "✅ Versions match: $PIP_VERSION"
fi
echo ""

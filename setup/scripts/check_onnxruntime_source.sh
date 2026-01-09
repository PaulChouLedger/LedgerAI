#!/bin/bash
# Check where onnxruntime-gpu is installed from

echo "=========================================="
echo "  Checking onnxruntime-gpu Installation Source"
echo "=========================================="
echo ""

# Check pip metadata
echo "[INFO] Checking pip metadata..."
PIP_VERSION=$(pip show onnxruntime-gpu 2>/dev/null | grep "^Version:" | awk '{print $2}' || echo "not installed")
PIP_LOCATION=$(pip show onnxruntime-gpu 2>/dev/null | grep "^Location:" | awk '{print $2}' || echo "not found")
echo "  Version: $PIP_VERSION"
echo "  Location: $PIP_LOCATION"
echo ""

# Check where it was installed from
echo "[INFO] Checking installation source..."
INSTALLED_FROM=$(pip show onnxruntime-gpu 2>/dev/null | grep -A 5 "Location:" | grep -E "Installed|Files" || echo "unknown")
echo "  Installation info: $INSTALLED_FROM"
echo ""

# Check runtime version
echo "[INFO] Checking runtime version..."
export ORT_DISABLE_CPUINFO=1
export ORT_LOG_LEVEL=3
RUNTIME_VERSION=$(python3 -c "import onnxruntime; print(onnxruntime.__version__)" 2>/dev/null || echo "import failed")
echo "  Runtime version: $RUNTIME_VERSION"
echo ""

# Check where the actual module file is
echo "[INFO] Checking actual module file location..."
MODULE_FILE=$(python3 -c "import onnxruntime; import os; print(os.path.dirname(onnxruntime.__file__))" 2>/dev/null || echo "not found")
echo "  Module file: $MODULE_FILE"
echo ""

# Check if it's in venv or system
if echo "$MODULE_FILE" | grep -q "aura-env\|venv"; then
    echo "  ✅ Installed in virtual environment"
elif echo "$MODULE_FILE" | grep -q "/usr/lib\|/usr/local/lib"; then
    echo "  ⚠️  Installed system-wide (not in venv)"
else
    echo "  Location: $MODULE_FILE"
fi
echo ""

# Check what versions are available from different sources
echo "[INFO] Checking available versions from different PyPI sources..."
echo ""
echo "  Standard PyPI (pypi.org):"
pip index versions onnxruntime-gpu --index-url https://pypi.org/simple 2>&1 | head -5 || echo "    Could not query"
echo ""

echo "  Jetson PyPI (pypi.jetson-ai-lab.io/jp6/cu126):"
pip index versions onnxruntime-gpu --extra-index-url https://pypi.jetson-ai-lab.io/jp6/cu126 2>&1 | head -5 || echo "    Could not query"
echo ""

# Check if there are multiple installations
echo "[INFO] Checking for multiple installations..."
echo "  System-wide:"
python3 -c "import sys; sys.path.insert(0, '/usr/lib/python3/dist-packages'); import onnxruntime; print(f'    Version: {onnxruntime.__version__}'); print(f'    Location: {onnxruntime.__file__}')" 2>/dev/null || echo "    Not found"
echo "  Virtual environment:"
python3 -c "import onnxruntime; print(f'    Version: {onnxruntime.__version__}'); print(f'    Location: {onnxruntime.__file__}')" 2>/dev/null || echo "    Not found"
echo ""

echo "=========================================="
echo "  Summary"
echo "=========================================="
echo ""
if [ "$PIP_VERSION" != "$RUNTIME_VERSION" ] && [ "$RUNTIME_VERSION" != "import failed" ]; then
    echo "⚠️  VERSION MISMATCH:"
    echo "   pip metadata: $PIP_VERSION"
    echo "   runtime: $RUNTIME_VERSION"
    echo ""
    echo "This suggests:"
    echo "  1. Multiple installations (system + venv)"
    echo "  2. Python is importing from a different location than pip installed to"
    echo "  3. Package metadata mismatch"
    echo ""
    echo "If runtime version is 1.23.2 but pip shows 1.23.0:"
    echo "  - It's likely installed system-wide or from standard PyPI"
    echo "  - Standard PyPI's onnxruntime-gpu may not work on ARM64/Jetson"
    echo "  - Jetson PyPI only has 1.23.0 (Jetson-optimized)"
fi
echo ""

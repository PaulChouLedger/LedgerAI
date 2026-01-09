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
SYSTEM_VERSION=$(python3 -c "import sys; sys.path.insert(0, '/usr/lib/python3/dist-packages'); import onnxruntime; print(onnxruntime.__version__)" 2>/dev/null || echo "Not found")
SYSTEM_PATH=$(python3 -c "import sys; sys.path.insert(0, '/usr/lib/python3/dist-packages'); import onnxruntime; import os; print(os.path.dirname(onnxruntime.__file__))" 2>/dev/null || echo "Not found")
echo "    Version: $SYSTEM_VERSION"
echo "    Path: $SYSTEM_PATH"
echo "  Virtual environment:"
VENV_VERSION=$(python3 -c "import onnxruntime; print(onnxruntime.__version__)" 2>/dev/null || echo "Not found")
VENV_PATH=$(python3 -c "import onnxruntime; import os; print(os.path.dirname(onnxruntime.__file__))" 2>/dev/null || echo "Not found")
echo "    Version: $VENV_VERSION"
echo "    Path: $VENV_PATH"
echo ""

# Determine which one Python is actually using
echo "[INFO] Determining which installation Python is using..."
ACTUAL_PATH=$(python3 -c "import onnxruntime; import os; print(os.path.dirname(onnxruntime.__file__))" 2>/dev/null || echo "unknown")
echo "  Actual import path: $ACTUAL_PATH"
echo ""

# Check architecture of the binary files
echo "[INFO] Checking binary architecture (ARM64 vs x86_64)..."
if [ -n "$ACTUAL_PATH" ] && [ "$ACTUAL_PATH" != "unknown" ]; then
    # Find .so files in the onnxruntime directory
    SO_FILES=$(find "$ACTUAL_PATH" -name "*.so" -type f 2>/dev/null | head -3)
    if [ -n "$SO_FILES" ]; then
        echo "  Checking architecture of binary files:"
        for SO_FILE in $SO_FILES; do
            if [ -f "$SO_FILE" ]; then
                ARCH=$(file "$SO_FILE" 2>/dev/null | grep -oE "ARM|aarch64|x86_64|Intel 80386" || echo "unknown")
                echo "    $(basename $SO_FILE): $ARCH"
            fi
        done
    else
        echo "  ⚠️  No .so files found to check architecture"
    fi
else
    echo "  ⚠️  Cannot determine path to check architecture"
fi
echo ""

# Check if it's from Jetson PyPI or standard PyPI
echo "[INFO] Checking installation source..."
if [ -n "$ACTUAL_PATH" ] && [ "$ACTUAL_PATH" != "unknown" ]; then
    # Find the site-packages directory
    SITE_PACKAGES=$(dirname "$ACTUAL_PATH")
    # Look specifically for onnxruntime-gpu dist-info (not other packages)
    DIST_INFO=$(find "$SITE_PACKAGES" -maxdepth 1 -name "onnxruntime*.dist-info" -type d 2>/dev/null | head -1)
    if [ -z "$DIST_INFO" ]; then
        # Fallback: search more broadly
        DIST_INFO=$(find "$SITE_PACKAGES" -name "*onnxruntime*.dist-info" -type d 2>/dev/null | head -1)
    fi
    if [ -n "$DIST_INFO" ]; then
        echo "  Distribution info: $(basename $DIST_INFO)"
        # Check METADATA for version and source
        if [ -f "$DIST_INFO/METADATA" ]; then
            METADATA_VERSION=$(grep "^Version:" "$DIST_INFO/METADATA" 2>/dev/null | awk '{print $2}' || echo "unknown")
            echo "  Metadata version: $METADATA_VERSION"
        fi
        # Check if there's a direct_url.json (pip 20.1+) that shows source
        if [ -f "$DIST_INFO/direct_url.json" ]; then
            SOURCE_URL=$(grep -o '"url": "[^"]*"' "$DIST_INFO/direct_url.json" 2>/dev/null | cut -d'"' -f4 || echo "unknown")
            echo "  Source URL: $SOURCE_URL"
            if echo "$SOURCE_URL" | grep -q "jetson-ai-lab"; then
                echo "  ✅ Installed from Jetson PyPI (Jetson-optimized)"
            elif echo "$SOURCE_URL" | grep -q "pypi.org"; then
                echo "  ⚠️  Installed from standard PyPI"
            fi
        fi
        # Check RECORD for wheel filename
        if [ -f "$DIST_INFO/RECORD" ]; then
            # Look for wheel filename in RECORD (first line often has it)
            WHEEL_NAME=$(head -1 "$DIST_INFO/RECORD" 2>/dev/null | grep -oE "onnxruntime[^,]*\.whl" | head -1 || echo "")
            if [ -n "$WHEEL_NAME" ]; then
                echo "  Wheel file: $WHEEL_NAME"
                # Check if wheel name contains platform tags
                if echo "$WHEEL_NAME" | grep -q "linux_aarch64\|manylinux.*aarch64"; then
                    echo "  ✅ ARM64/Jetson build detected in wheel name"
                elif echo "$WHEEL_NAME" | grep -q "linux_x86_64\|manylinux.*x86_64"; then
                    echo "  ⚠️  x86_64 build detected in wheel name (not optimal for Jetson)"
                fi
            fi
        fi
    else
        echo "  ⚠️  Could not find onnxruntime dist-info directory"
        echo "  (This might indicate a system package or manual installation)"
    fi
fi
echo ""

# Compare versions
if [ "$PIP_VERSION" != "$RUNTIME_VERSION" ] && [ "$RUNTIME_VERSION" != "import failed" ]; then
    echo "⚠️  VERSION MISMATCH DETECTED!"
    echo "   pip metadata: $PIP_VERSION"
    echo "   runtime version: $RUNTIME_VERSION"
    echo ""
    echo "Analysis:"
    if [ "$RUNTIME_VERSION" = "1.23.2" ] && [ "$PIP_VERSION" = "1.23.0" ]; then
        echo "  - Python is importing 1.23.2 but pip metadata shows 1.23.0"
        echo "  - This suggests a version mismatch in package metadata"
        echo ""
        # Check if binaries are ARM64 (which would indicate Jetson build)
        if echo "$(file "$ACTUAL_PATH"/*.so 2>/dev/null | head -1)" | grep -q "ARM\|aarch64"; then
            echo "  ✅ Binary architecture is ARM64/aarch64 (Jetson-optimized)"
            echo "  - This means 1.23.2 IS likely a Jetson build (not x86_64)"
            echo "  - The version mismatch is likely just metadata inconsistency"
            echo ""
            echo "✅ CONCLUSION: 1.23.2 appears to be Jetson-optimized and working correctly"
            echo "   The pip metadata showing 1.23.0 is likely outdated or incorrect"
        else
            echo "  ⚠️  Could not verify binary architecture"
            echo "  - 1.23.2 might be from standard PyPI (x86_64, not optimal)"
            echo "  - Or it could be Jetson-optimized with metadata mismatch"
        fi
    else
        echo "  - Multiple onnxruntime installations detected"
        echo "  - Python import path is picking up a different version than pip installed"
    fi
    echo ""
    echo "To fix and use Jetson-optimized version:"
    echo "  1. Uninstall all: pip uninstall -y onnxruntime onnxruntime-gpu"
    echo "  2. Uninstall system-wide: sudo pip3 uninstall -y onnxruntime onnxruntime-gpu"
    echo "  3. Reinstall Jetson version: pip install --index-url https://pypi.jetson-ai-lab.io/jp6/cu126 --extra-index-url https://pypi.org/simple onnxruntime-gpu==1.23.0"
else
    echo "✅ Versions match: $PIP_VERSION"
fi
echo ""

#!/bin/bash
# Find where onnxruntime-gpu 1.23.2 actually came from

echo "=========================================="
echo "  Finding onnxruntime-gpu 1.23.2 Source"
echo "=========================================="
echo ""

# Get the actual import path
ACTUAL_PATH=$(python3 -c "import onnxruntime; import os; print(os.path.dirname(onnxruntime.__file__))" 2>/dev/null || echo "unknown")
echo "[INFO] onnxruntime import path: $ACTUAL_PATH"
echo ""

# Find dist-info directory
SITE_PACKAGES=$(dirname "$ACTUAL_PATH")
echo "[INFO] Searching for dist-info in: $SITE_PACKAGES"
echo ""

# Find onnxruntime dist-info
DIST_INFO=$(find "$SITE_PACKAGES" -maxdepth 1 -name "*onnxruntime*.dist-info" -type d 2>/dev/null | head -1)
if [ -z "$DIST_INFO" ]; then
    echo "❌ Could not find onnxruntime dist-info directory"
    exit 1
fi

echo "[INFO] Found dist-info: $(basename $DIST_INFO)"
echo ""

# Check METADATA for version
if [ -f "$DIST_INFO/METADATA" ]; then
    echo "[INFO] METADATA file contents (relevant lines):"
    grep -E "^Version:|^Name:|^Home-page:|^Author:" "$DIST_INFO/METADATA" 2>/dev/null | head -10
    echo ""
fi

# Check direct_url.json (pip 20.1+ shows installation source)
if [ -f "$DIST_INFO/direct_url.json" ]; then
    echo "[INFO] Installation source (direct_url.json):"
    cat "$DIST_INFO/direct_url.json" 2>/dev/null | python3 -m json.tool 2>/dev/null || cat "$DIST_INFO/direct_url.json"
    echo ""
else
    echo "[INFO] No direct_url.json found (pip < 20.1 or manual installation)"
    echo ""
fi

# Check INSTALLER file
if [ -f "$DIST_INFO/INSTALLER" ]; then
    INSTALLER=$(cat "$DIST_INFO/INSTALLER" 2>/dev/null)
    echo "[INFO] Installer: $INSTALLER"
    echo ""
fi

# Check RECORD for wheel filename
if [ -f "$DIST_INFO/RECORD" ]; then
    echo "[INFO] Checking RECORD file for wheel information..."
    # Look for wheel filename in first few lines
    WHEEL_INFO=$(head -5 "$DIST_INFO/RECORD" 2>/dev/null | grep -oE "onnxruntime[^,]*\.whl" | head -1 || echo "")
    if [ -n "$WHEEL_INFO" ]; then
        echo "  Wheel file pattern: $WHEEL_INFO"
        # Extract platform tags from wheel name
        if echo "$WHEEL_INFO" | grep -q "linux_aarch64\|manylinux.*aarch64"; then
            echo "  ✅ ARM64/Jetson platform tag detected"
        elif echo "$WHEEL_INFO" | grep -q "linux_x86_64\|manylinux.*x86_64"; then
            echo "  ⚠️  x86_64 platform tag detected"
        fi
    fi
    echo ""
fi

# Check if there's a WHEEL file
if [ -f "$DIST_INFO/WHEEL" ]; then
    echo "[INFO] WHEEL metadata:"
    cat "$DIST_INFO/WHEEL" 2>/dev/null | head -10
    echo ""
fi

# Check pip's installation log if available
echo "[INFO] Checking for pip installation logs..."
if [ -f "/tmp/onnxruntime_install.log" ]; then
    echo "  Found /tmp/onnxruntime_install.log"
    echo "  Installation source from log:"
    grep -E "Looking in indexes|Collecting|Downloading|from.*pypi" /tmp/onnxruntime_install.log 2>/dev/null | head -5
    echo ""
fi

# Check what standard PyPI has
echo "[INFO] Checking standard PyPI for onnxruntime-gpu 1.23.2..."
STANDARD_VERSIONS=$(pip index versions onnxruntime-gpu --index-url https://pypi.org/simple 2>&1 | grep -A 5 "Available versions" || echo "")
if echo "$STANDARD_VERSIONS" | grep -q "1.23.2"; then
    echo "  ✅ Standard PyPI has 1.23.2"
    echo "  Checking if it has ARM64 builds..."
    # Try to see what platforms are available
    pip download --no-deps --index-url https://pypi.org/simple "onnxruntime-gpu==1.23.2" 2>&1 | grep -E "onnxruntime.*\.whl|platform" | head -5 || echo "    Could not check platforms"
else
    echo "  ❌ Standard PyPI does not have 1.23.2"
fi
echo ""

echo "=========================================="
echo "  Summary"
echo "=========================================="
echo ""
echo "Most likely scenarios for 1.23.2 on working device:"
echo "  1. Installed from standard PyPI when it had ARM64 builds (unlikely)"
echo "  2. Manually installed from a wheel file or different source"
echo "  3. Upgraded from 1.23.0 using 'pip install --upgrade' which may have"
echo "     pulled from standard PyPI if Jetson PyPI wasn't specified"
echo "  4. Installed before Jetson PyPI was available/configured"
echo ""
echo "Since binaries are ARM64, 1.23.2 is likely:"
echo "  - A Jetson-optimized build from a different source"
echo "  - Or standard PyPI happened to have ARM64 builds for 1.23.2"
echo ""

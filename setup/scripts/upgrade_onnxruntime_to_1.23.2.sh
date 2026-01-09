#!/bin/bash
# Upgrade onnxruntime-gpu to version 1.23.2+ to fix CPU detection crash on JetPack R36.4.4

set -e

echo "=========================================="
echo "  Upgrading onnxruntime-gpu to 1.23.2+"
echo "=========================================="
echo ""

# Check current version
echo "[INFO] Checking current onnxruntime-gpu version..."
CURRENT_VERSION=$(pip show onnxruntime-gpu 2>/dev/null | grep "^Version:" | awk '{print $2}' || echo "not installed")
echo "  Current version: $CURRENT_VERSION"
echo ""

# Check runtime version (what actually gets imported)
echo "[INFO] Checking runtime version..."
export ORT_DISABLE_CPUINFO=1
export ORT_LOG_LEVEL=3
RUNTIME_VERSION=$(python3 -c "import onnxruntime; print(onnxruntime.__version__)" 2>/dev/null || echo "import failed/crashes")
echo "  Runtime version: $RUNTIME_VERSION"
echo ""

# Check if upgrade is needed
if [ "$RUNTIME_VERSION" = "1.23.2" ] || [ "$RUNTIME_VERSION" = "import failed/crashes" ]; then
    if [ "$RUNTIME_VERSION" = "import failed/crashes" ]; then
        echo "[INFO] ⚠️  onnxruntime crashes on import - upgrade needed"
    else
        echo "[INFO] ✅ Already running 1.23.2 (working version)"
        echo "[INFO]    But pip metadata shows $CURRENT_VERSION - fixing metadata..."
    fi
else
    echo "[INFO] Current runtime version: $RUNTIME_VERSION"
    if [ "$RUNTIME_VERSION" != "1.23.2" ] && [ "$RUNTIME_VERSION" != "import failed/crashes" ]; then
        echo "[INFO] Version $RUNTIME_VERSION detected - checking if upgrade needed..."
    fi
fi
echo ""

# Uninstall existing versions
echo "[STEP] 1. Uninstalling existing onnxruntime packages..."
pip uninstall -y onnxruntime onnxruntime-gpu 2>/dev/null || true
python3 -m pip uninstall -y onnxruntime onnxruntime-gpu 2>/dev/null || true
echo "[INFO] ✅ Uninstalled existing versions"
echo ""

# Install version 1.23.2+ (fixes the crash)
echo "[STEP] 2. Installing onnxruntime-gpu >=1.23.2 (fixes CPU detection crash)..."
if pip install --extra-index-url https://pypi.jetson-ai-lab.io/jp6/cu126 "onnxruntime-gpu>=1.23.2" 2>&1 | tee /tmp/onnxruntime_upgrade.log; then
    INSTALLED_VERSION=$(pip show onnxruntime-gpu 2>/dev/null | grep "^Version:" | awk '{print $2}' || echo "unknown")
    echo "[INFO] ✅ Installed version: $INSTALLED_VERSION"
else
    echo "[WARNING] ⚠️  Version >=1.23.2 not available, trying >=1.23.0..."
    if pip install --extra-index-url https://pypi.jetson-ai-lab.io/jp6/cu126 "onnxruntime-gpu>=1.23.0" 2>&1 | tee /tmp/onnxruntime_upgrade.log; then
        INSTALLED_VERSION=$(pip show onnxruntime-gpu 2>/dev/null | grep "^Version:" | awk '{print $2}' || echo "unknown")
        echo "[INFO] ✅ Installed version: $INSTALLED_VERSION"
        echo "[WARNING] ⚠️  Version 1.23.0 may crash on JetPack R36.4.4 - ensure ORT_DISABLE_CPUINFO=1 is set"
    else
        echo "[ERROR] ❌ Failed to install onnxruntime-gpu"
        echo "[ERROR]    Check logs: /tmp/onnxruntime_upgrade.log"
        exit 1
    fi
fi
echo ""

# Verify installation
echo "[STEP] 3. Verifying installation..."
export ORT_DISABLE_CPUINFO=1
export ORT_LOG_LEVEL=3
if python3 -c "import onnxruntime; print('✅ Import successful'); print(f'Version: {onnxruntime.__version__}')" 2>&1; then
    FINAL_VERSION=$(python3 -c "import onnxruntime; print(onnxruntime.__version__)" 2>/dev/null)
    echo "[INFO] ✅ onnxruntime-gpu working correctly"
    echo "[INFO]    Runtime version: $FINAL_VERSION"
    
    if [ "$FINAL_VERSION" = "1.23.2" ] || [ "$FINAL_VERSION" \> "1.23.2" ]; then
        echo "[INFO] ✅ Version $FINAL_VERSION fixes the CPU detection crash"
    else
        echo "[WARNING] ⚠️  Version $FINAL_VERSION may still crash - ensure ORT_DISABLE_CPUINFO=1 is set"
    fi
else
    echo "[ERROR] ❌ onnxruntime still crashes after upgrade"
    echo "[ERROR]    This may require additional fixes or a different version"
    exit 1
fi
echo ""

echo "=========================================="
echo "  Upgrade Complete!"
echo "=========================================="
echo ""
echo "onnxruntime-gpu has been upgraded to fix the CPU detection crash."
echo ""
echo "Next steps:"
echo "  1. Ensure systemd service has ORT_DISABLE_CPUINFO=1 (run fix_onnxruntime_service.sh)"
echo "  2. Restart Aura service: sudo systemctl restart aura.service"
echo "  3. Check logs: journalctl -u aura.service -f"
echo ""

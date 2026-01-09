#!/bin/bash
# Diagnostic script to identify why onnxruntime crashes on this device but not others

echo "=========================================="
echo "  onnxruntime Crash Diagnostic"
echo "=========================================="
echo ""

echo "[INFO] Checking system information..."
echo ""

# Check JetPack version
if [ -f /etc/nv_tegra_release ]; then
    echo "JetPack Version:"
    cat /etc/nv_tegra_release
    echo ""
else
    echo "⚠️  Not a Jetson device or /etc/nv_tegra_release not found"
    echo ""
fi

# Check OS version
echo "OS Version:"
cat /etc/os-release | grep -E "VERSION|PRETTY_NAME"
echo ""

# Check Python version
echo "Python Version:"
python3 --version
echo ""

# Check onnxruntime-gpu version
echo "onnxruntime-gpu Version:"
pip show onnxruntime-gpu 2>/dev/null | grep -E "Name|Version|Location" || echo "⚠️  onnxruntime-gpu not found"
echo ""

# Check NumPy version
echo "NumPy Version:"
pip show numpy 2>/dev/null | grep -E "Name|Version" || echo "⚠️  NumPy not found"
echo ""

# Check gcc-toolset version
echo "GCC Toolset:"
if [ -d "/opt/rh/gcc-toolset-14" ]; then
    echo "✅ gcc-toolset-14 found"
    /opt/rh/gcc-toolset-14/root/usr/bin/gcc --version 2>/dev/null || echo "⚠️  gcc not accessible"
else
    echo "⚠️  gcc-toolset-14 not found"
fi
echo ""

# Check system libraries
echo "System Libraries:"
echo "glibc version:"
ldd --version | head -1
echo ""

# Check if environment variables are set in systemd service
echo "Checking systemd service configuration:"
if [ -f /etc/systemd/system/aura.service ]; then
    echo "✅ aura.service found"
    echo ""
    echo "Environment variables in service:"
    grep -E "ORT_DISABLE|ORT_LOG" /etc/systemd/system/aura.service || echo "⚠️  ORT environment variables not found in service file"
    echo ""
    echo "ExecStart command:"
    grep "^ExecStart=" /etc/systemd/system/aura.service || echo "⚠️  ExecStart not found"
else
    echo "⚠️  aura.service not found"
fi
echo ""

# Test importing onnxruntime with environment variables
echo "Testing onnxruntime import..."
echo ""

# Test 1: Without environment variables
echo "[TEST 1] Importing onnxruntime WITHOUT environment variables:"
python3 -c "import onnxruntime; print('✅ Import successful')" 2>&1 | head -5 || echo "❌ Import failed"
echo ""

# Test 2: With environment variables set in shell
echo "[TEST 2] Importing onnxruntime WITH environment variables (shell level):"
export ORT_DISABLE_CPUINFO=1
export ORT_LOG_LEVEL=3
python3 -c "import onnxruntime; print('✅ Import successful')" 2>&1 | head -5 || echo "❌ Import failed"
echo ""

# Test 3: With environment variables set in Python
echo "[TEST 3] Importing onnxruntime WITH environment variables (Python level):"
python3 << 'PYEOF'
import os
os.environ['ORT_DISABLE_CPUINFO'] = '1'
os.environ['ORT_LOG_LEVEL'] = '3'
try:
    import onnxruntime
    print('✅ Import successful')
except Exception as e:
    print(f'❌ Import failed: {e}')
PYEOF
echo ""

# Test 4: Check when environment variables are read
echo "[TEST 4] Checking when onnxruntime reads environment variables:"
python3 << 'PYEOF'
import os
import sys

# Set before any imports
os.environ['ORT_DISABLE_CPUINFO'] = '1'
os.environ['ORT_LOG_LEVEL'] = '3'

print(f"Environment variables set: ORT_DISABLE_CPUINFO={os.environ.get('ORT_DISABLE_CPUINFO')}")
print(f"Environment variables set: ORT_LOG_LEVEL={os.environ.get('ORT_LOG_LEVEL')}")

try:
    # This is where the crash happens
    import onnxruntime
    print('✅ onnxruntime imported successfully')
    print(f"onnxruntime version: {onnxruntime.__version__}")
except Exception as e:
    print(f'❌ Import failed: {e}')
    import traceback
    traceback.print_exc()
PYEOF
echo ""

# Check OpenWakeWord import
echo "[TEST 5] Testing OpenWakeWord import:"
export ORT_DISABLE_CPUINFO=1
export ORT_LOG_LEVEL=3
python3 -c "import openwakeword; print('✅ OpenWakeWord import successful')" 2>&1 | head -10 || echo "❌ OpenWakeWord import failed"
echo ""

echo "=========================================="
echo "  Diagnostic Complete"
echo "=========================================="
echo ""
echo "Key things to check:"
echo "1. Compare JetPack/OS version with working devices"
echo "2. Compare onnxruntime-gpu version"
echo "3. Check if gcc-toolset-14 is present (this device) vs absent (working devices)"
echo "4. Verify environment variables are set before Python starts (not just in Python code)"
echo ""

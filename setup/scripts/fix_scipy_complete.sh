#!/bin/bash
# fix_scipy_complete.sh - Complete scipy reinstallation (for corrupted installations)
# Usage: ./fix_scipy_complete.sh

set -e

echo "=========================================="
echo "  Complete Scipy Reinstallation"
echo "=========================================="
echo ""

# Activate virtual environment if it exists
if [ -f ~/aura-env/bin/activate ]; then
    source ~/aura-env/bin/activate
    echo "[INFO] Activated aura-env"
fi

echo "[STEP 1] Removing all scipy-related packages..."
pip uninstall -y scipy scipy-base scipy-core 2>&1 | grep -v "WARNING:" || true

echo ""
echo "[STEP 2] Removing scipy directories manually (in case of corruption)..."
SCIPY_DIRS=(
    "$VIRTUAL_ENV/lib/python3.10/site-packages/scipy"
    "$VIRTUAL_ENV/lib/python3.10/site-packages/scipy-*"
)
for dir in "${SCIPY_DIRS[@]}"; do
    if [ -d "$dir" ] || ls $dir 2>/dev/null; then
        echo "   Removing: $dir"
        rm -rf $dir 2>/dev/null || true
    fi
done

# Also remove any .pyc files
find "$VIRTUAL_ENV/lib/python3.10/site-packages" -name "*scipy*" -type f -delete 2>/dev/null || true
find "$VIRTUAL_ENV/lib/python3.10/site-packages" -name "*scipy*" -type d -exec rm -rf {} + 2>/dev/null || true

echo "✅ Cleaned up scipy files"

echo ""
echo "[STEP 3] Cleaning pip cache..."
pip cache purge 2>&1 | tail -3 || true

echo ""
echo "[STEP 4] Installing system dependencies..."
sudo apt-get update -qq
sudo apt-get install -y \
    libblas-dev \
    liblapack-dev \
    libatlas-base-dev \
    gfortran \
    libopenblas-dev \
    python3-dev \
    2>&1 | tail -5 || echo "   (some packages may already be installed)"

echo ""
echo "[STEP 5] Upgrading pip, setuptools, wheel..."
pip install --upgrade pip setuptools wheel 2>&1 | tail -3

echo ""
echo "[STEP 6] Installing numpy first (required for scipy)..."
if pip install --no-cache-dir --force-reinstall --upgrade "numpy>=1.24.0" 2>&1 | tee /tmp/numpy_install.log; then
    echo "✅ numpy installed"
    python3 -c "import numpy; print(f'   numpy version: {numpy.__version__}')" 2>&1
else
    echo "⚠️  numpy installation had issues"
fi

echo ""
echo "[STEP 7] Installing scipy (this may take a while on Jetson)..."
echo "   Note: Building scipy from source on ARM can take 10-30 minutes"
echo "   Consider using pre-built wheels if available..."

# Try to install scipy with pre-built wheels first
if pip install --no-cache-dir --upgrade "scipy>=1.10.0" 2>&1 | tee /tmp/scipy_install.log; then
    echo "✅ scipy installed successfully"
else
    echo "⚠️  Standard installation failed, trying with --no-build-isolation..."
    pip install --no-cache-dir --upgrade --no-build-isolation "scipy>=1.10.0" 2>&1 | tail -10
fi

echo ""
echo "[STEP 8] Verifying scipy installation..."
python3 -c "
import sys
print('Python version:', sys.version)
print('')

try:
    import scipy
    print(f'✅ scipy imported successfully')
    
    # Check version
    try:
        version = scipy.__version__
        print(f'✅ scipy version: {version}')
    except AttributeError:
        print('⚠️  scipy.__version__ not available (unusual but may still work)')
        # Try alternative
        try:
            import scipy.version
            version = scipy.version.version
            print(f'   Alternative version check: {version}')
        except:
            pass
    
    print(f'✅ scipy location: {scipy.__file__}')
    
    # Check scipy.fft
    try:
        from scipy import fft
        print(f'✅ scipy.fft module exists')
        
        if hasattr(fft, 'rfft'):
            print(f'✅ scipy.fft.rfft exists')
        else:
            print(f'❌ scipy.fft.rfft not found')
            print(f'   Available in fft: {[x for x in dir(fft) if not x.startswith(\"_\")][:15]}')
    except ImportError as e:
        print(f'❌ scipy.fft not available: {e}')
        
    # Test the actual import we need
    try:
        from scipy.fft import rfft, rfftfreq
        print(f'✅ scipy.fft.rfft and rfftfreq import successfully')
        
        # Test functionality
        import numpy as np
        test_data = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        result = rfft(test_data)
        print(f'✅ rfft works: result shape {result.shape}')
        
        freqs = rfftfreq(len(test_data), 1.0/16000)
        print(f'✅ rfftfreq works: {freqs[0]:.1f} Hz to {freqs[-1]:.1f} Hz')
        
        print('')
        print('✅✅✅ ALL CHECKS PASSED! scipy is working correctly!')
    except ImportError as e:
        print(f'❌ Import test failed: {e}')
        print('   The code has fallback imports, but scipy.fft is preferred')
        exit(1)
    except Exception as e:
        print(f'❌ Functionality test failed: {e}')
        import traceback
        traceback.print_exc()
        exit(1)
        
except ImportError as e:
    print(f'❌ scipy import failed: {e}')
    import traceback
    traceback.print_exc()
    exit(1)
except Exception as e:
    print(f'❌ Unexpected error: {e}')
    import traceback
    traceback.print_exc()
    exit(1)
" 2>&1

if [ $? -eq 0 ]; then
    echo ""
    echo "✅✅✅ Fix complete! scipy is fully functional."
else
    echo ""
    echo "⚠️  Verification failed. Options:"
    echo "   1. Try: pip install --upgrade --force-reinstall scipy"
    echo "   2. Check if system packages are installed:"
    echo "      sudo apt-get install -y libblas-dev liblapack-dev gfortran"
    echo "   3. The code has fallback imports, so it may still work"
    echo "   4. Check logs: /tmp/scipy_install.log"
fi

echo ""
echo "=========================================="
echo "  Fix Complete"
echo "=========================================="


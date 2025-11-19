#!/bin/bash
# fix_scipy_fft.sh - Fix scipy.fft import errors
# Usage: ./fix_scipy_fft.sh

set -e

echo "=========================================="
echo "  Fixing scipy.fft ImportError"
echo "=========================================="
echo ""

# Activate virtual environment if it exists
if [ -f ~/aura-env/bin/activate ]; then
    source ~/aura-env/bin/activate
    echo "[INFO] Activated aura-env"
fi

echo "[STEP 1] Checking current scipy installation..."
python3 -c "
try:
    import scipy
    print(f'✅ scipy version: {scipy.__version__}')
    print(f'   Location: {scipy.__file__}')
    
    # Check if scipy.fft exists
    try:
        from scipy import fft
        print(f'✅ scipy.fft module exists')
        
        # Check if rfft exists
        if hasattr(fft, 'rfft'):
            print(f'✅ scipy.fft.rfft exists')
        else:
            print(f'❌ scipy.fft.rfft not found')
            print(f'   Available: {[x for x in dir(fft) if not x.startswith(\"_\")][:10]}')
    except ImportError as e:
        print(f'❌ scipy.fft module not found: {e}')
        
    # Check scipy.fftpack (older versions)
    try:
        from scipy import fftpack
        if hasattr(fftpack, 'rfft'):
            print(f'✅ scipy.fftpack.rfft exists (older API)')
    except:
        pass
        
except Exception as e:
    print(f'❌ scipy import failed: {e}')
    exit(1)
" 2>&1

echo ""
echo "[STEP 2] Uninstalling scipy..."
pip uninstall -y scipy 2>&1 | grep -v "WARNING:" || true

echo ""
echo "[STEP 3] Installing system dependencies (may be required)..."
sudo apt-get install -y libblas-dev liblapack-dev libatlas-base-dev gfortran 2>&1 | tail -5 || echo "   (some packages may already be installed)"

echo ""
echo "[STEP 4] Reinstalling scipy (compatible version)..."
# Install scipy 1.10.0 or later (has scipy.fft with rfft)
if pip install --no-cache-dir --force-reinstall --upgrade "scipy>=1.10.0" 2>&1 | tee /tmp/scipy_reinstall.log; then
    echo "✅ scipy reinstalled successfully"
else
    echo "⚠️  scipy installation had issues, trying without version constraint..."
    pip install --no-cache-dir --force-reinstall --upgrade scipy 2>&1 | tail -5
fi

echo ""
echo "[STEP 5] Verifying scipy.fft installation..."
python3 -c "
try:
    from scipy.fft import rfft, rfftfreq
    print('✅ scipy.fft.rfft imports successfully')
    print('✅ scipy.fft.rfftfreq imports successfully')
    
    # Test that they work
    import numpy as np
    test_data = np.array([1.0, 2.0, 3.0, 4.0])
    result = rfft(test_data)
    print(f'✅ rfft works: {result}')
    
    freqs = rfftfreq(len(test_data), 1.0/16000)
    print(f'✅ rfftfreq works: {freqs[:3]}')
    
    print('✅ All checks passed!')
except ImportError as e:
    print(f'❌ Import failed: {e}')
    print('   Trying fallback imports...')
    try:
        from scipy.fftpack import rfft
        from numpy.fft import rfftfreq
        print('✅ Fallback imports work (older scipy version)')
        print('   Note: listener.py has fallback code for this')
    except Exception as e2:
        print(f'❌ Fallback also failed: {e2}')
        exit(1)
except Exception as e:
    print(f'❌ Verification failed: {e}')
    import traceback
    traceback.print_exc()
    exit(1)
" 2>&1

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Fix complete! scipy.fft should now work."
else
    echo ""
    echo "⚠️  Fix incomplete. The code has fallback imports, but you may want to:"
    echo "   1. Check scipy version: python3 -c 'import scipy; print(scipy.__version__)'"
    echo "   2. Try: pip install --upgrade scipy"
    echo "   3. Check logs: /tmp/scipy_reinstall.log"
fi

echo ""
echo "=========================================="
echo "  Fix Complete"
echo "=========================================="


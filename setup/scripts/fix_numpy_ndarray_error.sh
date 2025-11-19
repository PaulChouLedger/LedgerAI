#!/bin/bash
# fix_numpy_ndarray_error.sh - Fix numpy.ndarray AttributeError
# This error usually indicates a corrupted numpy installation or namespace collision

set -e

echo "=========================================="
echo "  Fixing numpy.ndarray AttributeError"
echo "=========================================="
echo ""

# Activate virtual environment if it exists
if [ -f ~/aura-env/bin/activate ]; then
    source ~/aura-env/bin/activate
    echo "[INFO] Activated aura-env"
fi

echo "[STEP 1] Checking for conflicting numpy.py files..."
# Check if there's a numpy.py file in the current directory or Python path
CONFLICTING_FILES=$(find . -maxdepth 2 -name "numpy.py" -type f 2>/dev/null || true)
if [ -n "$CONFLICTING_FILES" ]; then
    echo "⚠️  Found conflicting numpy.py files:"
    echo "$CONFLICTING_FILES"
    echo "   These will be removed (they conflict with the real numpy package)"
    echo "$CONFLICTING_FILES" | xargs rm -f
    echo "✅ Removed conflicting files"
else
    echo "✅ No conflicting numpy.py files found"
fi

echo ""
echo "[STEP 2] Uninstalling all numpy-related packages..."
pip uninstall -y numpy numpy-base numpy-core 2>&1 | grep -v "WARNING:" || true

echo ""
echo "[STEP 3] Cleaning pip cache..."
pip cache purge 2>&1 | tail -3 || true

echo ""
echo "[STEP 4] Reinstalling numpy (clean install)..."
# Install numpy without any cache to ensure clean installation
if pip install --no-cache-dir --force-reinstall numpy==1.26.4 2>&1 | tee /tmp/numpy_reinstall.log; then
    echo "✅ numpy 1.26.4 installed"
else
    echo "⚠️  numpy 1.26.4 installation had issues, trying latest compatible version..."
    pip install --no-cache-dir --force-reinstall numpy 2>&1 | tail -5
fi

echo ""
echo "[STEP 5] Verifying numpy installation..."
python3 -c "
import numpy as np
print(f'✅ numpy version: {np.__version__}')
print(f'✅ numpy location: {np.__file__}')
print(f'✅ numpy.ndarray exists: {hasattr(np, \"ndarray\")}')
if hasattr(np, 'ndarray'):
    print(f'✅ numpy.ndarray type: {type(np.ndarray)}')
    arr = np.array([1, 2, 3])
    print(f'✅ Can create array: {arr}')
    print(f'✅ Array is ndarray: {isinstance(arr, np.ndarray)}')
    print('✅ numpy is working correctly!')
else:
    print('❌ numpy.ndarray still missing - this is unusual')
    print('   numpy type:', type(np))
    print('   numpy dir:', list(dir(np))[:20])
    exit(1)
" 2>&1

if [ $? -eq 0 ]; then
    echo ""
    echo "[STEP 6] Testing torch import..."
    python3 -c "
import torch
print(f'✅ torch version: {torch.__version__}')
print('✅ torch imports successfully!')
" 2>&1 && echo "✅ All checks passed!" || echo "⚠️  torch still has issues (may need torch reinstall)"
else
    echo ""
    echo "❌ numpy verification failed"
    echo "   This may require more aggressive fixes:"
    echo "   1. pip uninstall numpy torch"
    echo "   2. pip cache purge"
    echo "   3. pip install numpy==1.26.4"
    echo "   4. pip install torch (compatible version)"
fi

echo ""
echo "=========================================="
echo "  Fix Complete"
echo "=========================================="


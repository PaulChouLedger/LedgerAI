#!/bin/bash
# diagnose_numpy_ndarray.sh - Diagnose numpy.ndarray AttributeError
# Usage: ./diagnose_numpy_ndarray.sh

set -e

echo "=========================================="
echo "  Diagnosing numpy.ndarray AttributeError"
echo "=========================================="
echo ""

# Activate virtual environment if it exists
if [ -f ~/aura-env/bin/activate ]; then
    source ~/aura-env/bin/activate
    echo "[INFO] Activated aura-env"
fi

echo "[STEP 1] Checking for conflicting numpy files..."
echo "Searching for numpy.py files in current directory and Python path..."
find . -name "numpy.py" -type f 2>/dev/null | head -5 || echo "  No numpy.py files found in current directory"
python3 -c "import sys; [print(f'  {p}') for p in sys.path if 'numpy' in str(p)]" 2>/dev/null || true

echo ""
echo "[STEP 2] Checking numpy installation..."
python3 -c "
import sys
import os

# Check if numpy is in path
print('Python path:')
for p in sys.path:
    print(f'  {p}')

print('')
print('Checking for numpy module...')
try:
    import numpy
    print(f'✅ numpy imported successfully')
    print(f'   Location: {numpy.__file__}')
    print(f'   Version: {numpy.__version__}')
    print(f'   Type: {type(numpy)}')
    print(f'   Dir (first 20): {list(dir(numpy))[:20]}')
except Exception as e:
    print(f'❌ Failed to import numpy: {e}')
    sys.exit(1)

print('')
print('Checking for numpy.ndarray...')
try:
    ndarray = numpy.ndarray
    print(f'✅ numpy.ndarray exists: {ndarray}')
    print(f'   Type: {type(ndarray)}')
    print(f'   Is class: {isinstance(ndarray, type)}')
except AttributeError as e:
    print(f'❌ numpy.ndarray not found: {e}')
    print('   This is the problem!')
    print('')
    print('Checking what numpy actually is...')
    print(f'   numpy type: {type(numpy)}')
    print(f'   numpy module: {numpy}')
    print(f'   numpy.__dict__ keys: {list(numpy.__dict__.keys())[:30]}')
    sys.exit(1)

print('')
print('Testing numpy.ndarray creation...')
try:
    arr = numpy.array([1, 2, 3])
    print(f'✅ Can create array: {arr}')
    print(f'   Array type: {type(arr)}')
    print(f'   Is ndarray: {isinstance(arr, numpy.ndarray)}')
except Exception as e:
    print(f'❌ Failed to create array: {e}')
    sys.exit(1)
" 2>&1

echo ""
echo "[STEP 3] Checking for multiple numpy installations..."
python3 -c "
import sys
import importlib.util

# Check all numpy installations
numpy_paths = []
for path in sys.path:
    numpy_path = os.path.join(path, 'numpy')
    if os.path.exists(numpy_path):
        numpy_paths.append(numpy_path)
    numpy_py = os.path.join(path, 'numpy.py')
    if os.path.exists(numpy_py):
        numpy_paths.append(numpy_py)

if len(numpy_paths) > 1:
    print(f'⚠️  Found {len(numpy_paths)} numpy installations:')
    for p in numpy_paths:
        print(f'   {p}')
else:
    print(f'✅ Found 1 numpy installation: {numpy_paths[0] if numpy_paths else \"None\"}')
" 2>&1 || true

echo ""
echo "[STEP 4] Checking torch compatibility..."
python3 -c "
try:
    import torch
    print(f'✅ torch version: {torch.__version__}')
    print(f'   torch location: {torch.__file__}')
except Exception as e:
    print(f'❌ torch import failed: {e}')
    import traceback
    traceback.print_exc()
" 2>&1 || true

echo ""
echo "=========================================="
echo "  Diagnosis Complete"
echo "=========================================="


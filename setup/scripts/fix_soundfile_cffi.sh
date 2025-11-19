#!/bin/bash
# fix_soundfile_cffi.sh - Fix corrupted cffi_backend binary
# The error "file too short" indicates a corrupted binary extension

set -e

echo "=========================================="
echo "  Fixing soundfile/cffi ImportError"
echo "=========================================="
echo ""

# Activate virtual environment if it exists
if [ -f ~/aura-env/bin/activate ]; then
    source ~/aura-env/bin/activate
    echo "[INFO] Activated aura-env"
fi

echo "[STEP 1] Checking corrupted file..."
CORRUPTED_FILE="$VIRTUAL_ENV/lib/python3.10/site-packages/_cffi_backend.cpython-310-aarch64-linux-gnu.so"
if [ -f "$CORRUPTED_FILE" ]; then
    FILE_SIZE=$(stat -c%s "$CORRUPTED_FILE" 2>/dev/null || stat -f%z "$CORRUPTED_FILE" 2>/dev/null || echo "unknown")
    echo "⚠️  Found corrupted file: $CORRUPTED_FILE"
    echo "   Size: $FILE_SIZE bytes (should be much larger)"
    echo "   Removing corrupted file..."
    rm -f "$CORRUPTED_FILE"
    echo "✅ Removed corrupted file"
else
    echo "✅ No corrupted file found at expected location"
fi

echo ""
echo "[STEP 2] Uninstalling soundfile and cffi..."
pip uninstall -y soundfile cffi 2>&1 | grep -v "WARNING:" || true

echo ""
echo "[STEP 3] Cleaning pip cache..."
pip cache purge 2>&1 | tail -3 || true

echo ""
echo "[STEP 4] Reinstalling cffi (must be installed first)..."
if pip install --no-cache-dir --force-reinstall --upgrade cffi 2>&1 | tee /tmp/cffi_reinstall.log; then
    echo "✅ cffi reinstalled successfully"
else
    echo "⚠️  cffi installation had issues"
    echo "   Trying with build isolation disabled..."
    pip install --no-cache-dir --force-reinstall --upgrade --no-build-isolation cffi 2>&1 | tail -5
fi

echo ""
echo "[STEP 5] Verifying cffi installation..."
python3 -c "
try:
    import _cffi_backend
    print('✅ _cffi_backend imports successfully')
    print(f'   Location: {_cffi_backend.__file__}')
    import os
    size = os.path.getsize(_cffi_backend.__file__)
    print(f'   File size: {size} bytes')
    if size < 1000:
        print('   ⚠️  File seems too small, may still be corrupted')
    else:
        print('   ✅ File size looks good')
except Exception as e:
    print(f'❌ _cffi_backend import failed: {e}')
    exit(1)
" 2>&1

if [ $? -ne 0 ]; then
    echo ""
    echo "⚠️  cffi verification failed, trying alternative installation..."
    echo "[STEP 5b] Installing system libffi-dev (may be required)..."
    sudo apt-get install -y libffi-dev 2>&1 | tail -3 || echo "   (skipping if already installed)"
    
    echo ""
    echo "[STEP 5c] Reinstalling cffi with system libraries..."
    pip install --no-cache-dir --force-reinstall --upgrade cffi 2>&1 | tail -5
fi

echo ""
echo "[STEP 6] Reinstalling soundfile..."
if pip install --no-cache-dir --force-reinstall --upgrade soundfile 2>&1 | tee /tmp/soundfile_reinstall.log; then
    echo "✅ soundfile reinstalled successfully"
else
    echo "⚠️  soundfile installation had issues"
    echo "   Trying with build isolation disabled..."
    pip install --no-cache-dir --force-reinstall --upgrade --no-build-isolation soundfile 2>&1 | tail -5
fi

echo ""
echo "[STEP 7] Verifying soundfile installation..."
python3 -c "
try:
    import soundfile as sf
    print('✅ soundfile imports successfully')
    print(f'   Version: {sf.__version__}')
    print(f'   Location: {sf.__file__}')
    print('✅ All checks passed!')
except Exception as e:
    print(f'❌ soundfile import failed: {e}')
    import traceback
    traceback.print_exc()
    exit(1)
" 2>&1

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Fix complete! soundfile should now work."
else
    echo ""
    echo "❌ Fix incomplete. Additional steps:"
    echo "   1. Check if libsndfile1-dev is installed: sudo apt-get install libsndfile1-dev"
    echo "   2. Try: pip install --force-reinstall --no-binary :all: soundfile"
    echo "   3. Check logs: /tmp/cffi_reinstall.log and /tmp/soundfile_reinstall.log"
fi

echo ""
echo "=========================================="
echo "  Fix Complete"
echo "=========================================="


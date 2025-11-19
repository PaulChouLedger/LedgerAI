#!/bin/bash
# fix_certifi.sh - Fix corrupted certifi package
# Usage: ./fix_certifi.sh

set -e

echo "=========================================="
echo "  Fixing certifi AttributeError"
echo "=========================================="
echo ""

# Activate virtual environment if it exists
if [ -f ~/aura-env/bin/activate ]; then
    source ~/aura-env/bin/activate
    echo "[INFO] Activated aura-env"
fi

echo "[STEP 1] Checking current certifi installation..."
python3 -c "
try:
    import certifi
    print(f'✅ certifi imported')
    print(f'   Location: {certifi.__file__}')
    try:
        print(f'   Version: {certifi.__version__}')
    except AttributeError:
        print('   ⚠️  No __version__ attribute')
    
    # Check for where() function
    if hasattr(certifi, 'where'):
        print(f'✅ certifi.where exists')
        try:
            path = certifi.where()
            print(f'   Certificate path: {path}')
            import os
            if os.path.exists(path):
                print(f'   ✅ Certificate file exists')
            else:
                print(f'   ❌ Certificate file missing!')
        except Exception as e:
            print(f'   ❌ certifi.where() failed: {e}')
    else:
        print(f'❌ certifi.where() not found - this is the problem!')
        print(f'   Available attributes: {[x for x in dir(certifi) if not x.startswith(\"_\")][:10]}')
except Exception as e:
    print(f'❌ certifi import failed: {e}')
    import traceback
    traceback.print_exc()
" 2>&1

echo ""
echo "[STEP 2] Uninstalling certifi and related packages..."
pip uninstall -y certifi httpx elevenlabs requests urllib3 2>&1 | grep -v "WARNING:" || true

echo ""
echo "[STEP 3] Removing corrupted certifi files..."
CERTIFI_DIRS=(
    "$VIRTUAL_ENV/lib/python3.10/site-packages/certifi"
    "$VIRTUAL_ENV/lib/python3.10/site-packages/certifi-*"
)
for dir in "${CERTIFI_DIRS[@]}"; do
    if [ -d "$dir" ] || ls $dir 2>/dev/null; then
        echo "   Removing: $dir"
        rm -rf $dir 2>/dev/null || true
    fi
done

# Remove any .pyc files
find "$VIRTUAL_ENV/lib/python3.10/site-packages" -name "*certifi*" -type f -delete 2>/dev/null || true
find "$VIRTUAL_ENV/lib/python3.10/site-packages" -name "*certifi*" -type d -exec rm -rf {} + 2>/dev/null || true

echo "✅ Cleaned up certifi files"

echo ""
echo "[STEP 4] Cleaning pip cache..."
pip cache purge 2>&1 | tail -3 || true

echo ""
echo "[STEP 5] Reinstalling certifi (must be first)..."
if pip install --no-cache-dir --force-reinstall --upgrade certifi 2>&1 | tee /tmp/certifi_reinstall.log; then
    echo "✅ certifi reinstalled"
else
    echo "⚠️  certifi installation had issues"
    pip install --no-cache-dir --force-reinstall certifi 2>&1 | tail -5
fi

echo ""
echo "[STEP 6] Verifying certifi installation..."
python3 -c "
try:
    import certifi
    print(f'✅ certifi imported successfully')
    
    if not hasattr(certifi, 'where'):
        print('❌ certifi.where still missing!')
        exit(1)
    
    cert_path = certifi.where()
    print(f'✅ certifi.where() works: {cert_path}')
    
    import os
    if os.path.exists(cert_path):
        size = os.path.getsize(cert_path)
        print(f'✅ Certificate file exists ({size} bytes)')
        if size < 1000:
            print('   ⚠️  File seems too small, may be corrupted')
        else:
            print('   ✅ File size looks good')
    else:
        print(f'❌ Certificate file missing at: {cert_path}')
        exit(1)
    
    print('✅ certifi is working correctly!')
except Exception as e:
    print(f'❌ Verification failed: {e}')
    import traceback
    traceback.print_exc()
    exit(1)
" 2>&1

if [ $? -ne 0 ]; then
    echo ""
    echo "⚠️  certifi verification failed"
    exit 1
fi

echo ""
echo "[STEP 7] Reinstalling httpx and elevenlabs..."
pip install --no-cache-dir --force-reinstall --upgrade httpx elevenlabs 2>&1 | tail -5

echo ""
echo "[STEP 8] Verifying httpx and elevenlabs..."
python3 -c "
try:
    import httpx
    print(f'✅ httpx imported successfully')
    
    import elevenlabs
    print(f'✅ elevenlabs imported successfully')
    
    # Test that they can create clients
    try:
        from elevenlabs import ElevenLabs
        # Don't actually create client (needs API key), just check import
        print('✅ ElevenLabs class available')
    except Exception as e:
        print(f'⚠️  ElevenLabs import check: {e}')
    
    print('✅ All checks passed!')
except Exception as e:
    print(f'❌ Verification failed: {e}')
    import traceback
    traceback.print_exc()
    exit(1)
" 2>&1

if [ $? -eq 0 ]; then
    echo ""
    echo "✅✅✅ Fix complete! certifi, httpx, and elevenlabs should now work."
else
    echo ""
    echo "⚠️  Some packages may still have issues. Try:"
    echo "   pip install --upgrade --force-reinstall certifi httpx elevenlabs"
fi

echo ""
echo "=========================================="
echo "  Fix Complete"
echo "=========================================="


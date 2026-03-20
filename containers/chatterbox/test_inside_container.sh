#!/bin/bash
# Test script to run inside the container to debug startup issues

echo "=== Testing Container Environment ==="
echo ""

echo "1. Checking Python:"
python3 --version
echo ""

echo "2. Checking if container_rest.py exists:"
ls -la /app/container_rest.py
echo ""

echo "3. Testing Python import:"
python3 -c "import sys; print(f'Python path: {sys.path}')"
echo ""

echo "4. Testing PyTorch import:"
python3 -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
echo ""

echo "5. Testing Chatterbox import:"
python3 -c "
try:
    from chatterbox.tts import ChatterboxTTS
    print('✅ Imported from chatterbox.tts')
except ImportError as e:
    print(f'❌ Import from chatterbox.tts failed: {e}')
    try:
        from chatterbox import ChatterboxTTS
        print('✅ Imported from chatterbox')
    except ImportError as e2:
        print(f'❌ Import from chatterbox failed: {e2}')
"
echo ""

echo "6. Testing if Flask can import:"
python3 -c "from flask import Flask; print('✅ Flask imported')"
echo ""

echo "7. Checking if /app/chatterbox exists:"
ls -la /app/chatterbox 2>&1 || echo "❌ /app/chatterbox does not exist"
echo ""

echo "8. Trying to run container_rest.py directly:"
cd /app && python3 container_rest.py &
PID=$!
sleep 3
if ps -p $PID > /dev/null; then
    echo "✅ container_rest.py is running (PID: $PID)"
    kill $PID
else
    echo "❌ container_rest.py crashed or exited"
fi


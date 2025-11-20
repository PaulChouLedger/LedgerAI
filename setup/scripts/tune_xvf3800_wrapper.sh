#!/bin/bash
# Wrapper script for tune_xvf3800.py that waits for USB device to be ready
# This ensures the device is available before attempting to configure it

MAX_WAIT=30  # Maximum seconds to wait for device
WAIT_INTERVAL=1  # Check every second
PRESET="${1:-agc_20_ec}"  # Default preset

# Get paths from environment or use defaults
LEDGERAI_DIR="${LEDGERAI_DIR:-$HOME/LedgerAI}"
PYTHON_CMD="${PYTHON_CMD:-python3.10}"

echo "[XVF3800 Wrapper] Waiting for USB device to be ready..."
echo "[XVF3800 Wrapper] Will wait up to ${MAX_WAIT} seconds..."

# Wait for device to appear in lsusb
for i in $(seq 1 $MAX_WAIT); do
    if lsusb | grep -qi "reSpeaker\|XVF3800\|UACDemo"; then
        echo "[XVF3800 Wrapper] ✅ Device found after ${i} seconds"
        
        # Additional small delay to ensure device is fully initialized
        sleep 2
        
        # Run the tuning script
        echo "[XVF3800 Wrapper] Running tuning script with preset: ${PRESET}"
        "$PYTHON_CMD" -B "$LEDGERAI_DIR/setup/scripts/tune_xvf3800.py" "$PRESET"
        exit $?
    fi
    
    if [ $((i % 5)) -eq 0 ]; then
        echo "[XVF3800 Wrapper] ⏳ Still waiting... (${i}/${MAX_WAIT}s)"
    fi
    
    sleep $WAIT_INTERVAL
done

echo "[XVF3800 Wrapper] ⚠️  Device not found after ${MAX_WAIT} seconds"
echo "[XVF3800 Wrapper] Attempting to run tuning script anyway (may fail)..."
"$PYTHON_CMD" -B "$LEDGERAI_DIR/setup/scripts/tune_xvf3800.py" "$PRESET"
exit $?


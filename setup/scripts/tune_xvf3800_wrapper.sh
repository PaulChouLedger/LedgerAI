#!/bin/bash
# Wrapper script for tune_xvf3800.py that waits for USB device to be ready
# This ensures the device is available before attempting to configure it
# Uses dynamic path resolution - no hardcoded paths needed!

MAX_WAIT=30  # Maximum seconds to wait for device
WAIT_INTERVAL=1  # Check every second
PRESET="${1:-agc_20_ec}"  # Default preset

# Dynamically find LedgerAI directory
# Method 1: Find by locating this script (most reliable)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LEDGERAI_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Method 2: If that doesn't work, try common locations
if [ ! -f "$LEDGERAI_DIR/setup/scripts/tune_xvf3800.py" ]; then
    # Try user's home directory
    if [ -f "$HOME/LedgerAI/setup/scripts/tune_xvf3800.py" ]; then
        LEDGERAI_DIR="$HOME/LedgerAI"
    # Try /home/ledger (common Jetson setup)
    elif [ -f "/home/ledger/LedgerAI/setup/scripts/tune_xvf3800.py" ]; then
        LEDGERAI_DIR="/home/ledger/LedgerAI"
    # Try /home/aura (common setup)
    elif [ -f "/home/aura/LedgerAI/setup/scripts/tune_xvf3800.py" ]; then
        LEDGERAI_DIR="/home/aura/LedgerAI"
    else
        echo "[XVF3800 Wrapper] ❌ Error: Could not find LedgerAI directory"
        echo "[XVF3800 Wrapper]    Checked: $LEDGERAI_DIR"
        echo "[XVF3800 Wrapper]    Checked: $HOME/LedgerAI"
        echo "[XVF3800 Wrapper]    Script location: $SCRIPT_DIR"
        exit 1
    fi
fi

# Detect user from LedgerAI directory ownership (for OTA updates - no User= field needed)
# This allows the service to work seamlessly after git pull
ACTUAL_USER=$(stat -c "%U" "$LEDGERAI_DIR" 2>/dev/null || echo "")
if [ -z "$ACTUAL_USER" ] || [ "$ACTUAL_USER" = "root" ]; then
    # Fallback: try to detect from $HOME or use current user
    if [ -n "$HOME" ] && [ "$HOME" != "/root" ]; then
        ACTUAL_USER=$(basename "$HOME")
    else
        ACTUAL_USER=$(whoami)
    fi
fi

# If running as root but LedgerAI is owned by another user, switch to that user
CURRENT_USER=$(whoami)
if [ "$CURRENT_USER" = "root" ] && [ -n "$ACTUAL_USER" ] && [ "$ACTUAL_USER" != "root" ]; then
    echo "[XVF3800 Wrapper] 🔄 Switching to user: $ACTUAL_USER (owner of LedgerAI directory)"
    exec su - "$ACTUAL_USER" -c "\"$LEDGERAI_DIR/setup/scripts/tune_xvf3800_wrapper.sh\" \"$PRESET\""
    exit $?
fi

# Get Python command from environment or detect it
PYTHON_CMD="${PYTHON_CMD:-$(command -v python3.10 2>/dev/null || command -v python3 2>/dev/null || echo 'python3')}"

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


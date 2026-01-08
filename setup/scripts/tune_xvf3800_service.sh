#!/bin/bash
# Wrapper script for xvf3800-tuning.service
# Handles device detection, path resolution, and executes tuning script

set +e  # Don't exit on errors during device detection loops

PRESET="${1:-agc_20_ec}"

# Wait for USB device to appear (up to 30 seconds)
echo "[xvf3800-tuning] Waiting for USB device..."
for i in {1..30}; do
    if lsusb | grep -q "2886:" || lsusb | grep -qi "seeed\|reSpeaker\|XVF3800"; then
        echo "[xvf3800-tuning] USB device detected"
        break
    fi
    sleep 1
done

# Additional wait for device to fully initialize
# Device needs time to be ready for xvf_host commands after USB detection
sleep 3

# Wait for ALSA to detect the device (up to 10 seconds)
echo "[xvf3800-tuning] Waiting for ALSA device..."
ALSA_DETECTED=false
for i in {1..10}; do
    if arecord -l 2>/dev/null | grep -q -i "XVF3800\|reSpeaker"; then
        echo "[xvf3800-tuning] ALSA device detected"
        ALSA_DETECTED=true
        break
    fi
    sleep 1
done

if [ "$ALSA_DETECTED" = "false" ]; then
    echo "[xvf3800-tuning] WARNING: ALSA device not detected, but continuing anyway"
    echo "[xvf3800-tuning] Device may still be initializing..."
fi

# Additional wait after ALSA detection for device to be fully ready
echo "[xvf3800-tuning] Waiting additional time for device to be ready for xvf_host..."
sleep 2

# Try to find LedgerAI directory
# Check common user home directories
LEDGERAI_DIR=""
for home_dir in "/home/ledger" "/home/aura" "$HOME"; do
    if [ -n "$home_dir" ] && [ -d "$home_dir/LedgerAI" ] && [ -f "$home_dir/LedgerAI/setup/scripts/tune_xvf3800.py" ]; then
        LEDGERAI_DIR="$home_dir/LedgerAI"
        echo "[xvf3800-tuning] Found LedgerAI directory: $LEDGERAI_DIR"
        break
    fi
done

if [ -z "$LEDGERAI_DIR" ]; then
    echo "[xvf3800-tuning] ERROR: Could not find LedgerAI directory"
    echo "[xvf3800-tuning] Checked: /home/ledger/LedgerAI, /home/aura/LedgerAI, ${HOME:-<not set>}/LedgerAI"
    exit 1
fi

# Verify xvf_host exists
if [ -f "$LEDGERAI_DIR/setup/scripts/tune_xvf3800.py" ]; then
    # The Python script will check for xvf_host itself, but let's verify the expected path
    EXPECTED_XVF_HOST=""
    for home_dir in "/home/ledger" "/home/aura" "$HOME"; do
        if [ -n "$home_dir" ] && [ -f "$home_dir/reSpeaker_XVF3800_USB_4MIC_ARRAY/host_control/jetson/xvf_host" ]; then
            EXPECTED_XVF_HOST="$home_dir/reSpeaker_XVF3800_USB_4MIC_ARRAY/host_control/jetson/xvf_host"
            echo "[xvf3800-tuning] Found xvf_host at: $EXPECTED_XVF_HOST"
            break
        fi
    done
    if [ -z "$EXPECTED_XVF_HOST" ]; then
        echo "[xvf3800-tuning] WARNING: xvf_host not found in expected location"
        echo "[xvf3800-tuning] The Python script will provide more details if it can't find it"
    fi
fi

# Final check: Verify device is still visible before proceeding
echo "[xvf3800-tuning] Verifying device is accessible..."
if ! lsusb | grep -q "2886:" && ! lsusb | grep -qi "seeed\|reSpeaker\|XVF3800"; then
    echo "[xvf3800-tuning] ERROR: Device no longer visible in USB device list"
    echo "[xvf3800-tuning] Device may have disconnected or needs to be replugged"
    exit 1
fi

# Change to LedgerAI directory and run tuning script
cd "$LEDGERAI_DIR" || exit 1

# Run the tuning script (now enable strict error checking)
set -e
echo "[xvf3800-tuning] Running tuning script with preset: $PRESET"
python3 -B setup/scripts/tune_xvf3800.py "$PRESET"

exit $?

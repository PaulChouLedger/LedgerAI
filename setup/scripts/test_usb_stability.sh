#!/bin/bash
# Test USB connection stability with ReSpeaker

echo "=================================="
echo "  USB STABILITY TEST - ReSpeaker"
echo "=================================="
echo ""
echo "This test monitors USB disconnects/reconnects"
echo "Touch the USB isolator box during the test"
echo "Press Ctrl+C to stop"
echo ""

# Monitor kernel messages for USB events
echo "Monitoring USB events (touch your USB isolator now)..."
echo ""

# Run dmesg in follow mode and filter for USB
sudo dmesg -w | grep --line-buffered -i "usb\|2886:0018\|respeaker" &
DMESG_PID=$!

# Also monitor lsusb to see if device disappears
echo "Monitoring device presence..."
while true; do
    if lsusb | grep -q "2886:0018"; then
        echo -ne "\r✅ ReSpeaker connected   "
    else
        echo -ne "\r❌ ReSpeaker DISCONNECTED"
    fi
    sleep 0.5
done

# Cleanup on exit
trap "kill $DMESG_PID 2>/dev/null; exit" INT TERM


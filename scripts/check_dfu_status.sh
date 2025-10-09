#!/bin/bash
#
# Check if ReSpeaker is in DFU mode
#

echo ""
echo "Checking for ReSpeaker devices..."
echo ""
echo "Normal mode (2886:0018):"
lsusb | grep "2886:0018" || echo "  Not found"
echo ""
echo "DFU mode (should be different ID, often 20b1:xxxx or similar):"
lsusb | grep -i "dfu" || echo "  Not found"
echo ""
echo "All USB devices:"
lsusb
echo ""

# Check if dfu-util is installed
if command -v dfu-util &> /dev/null; then
    echo "Checking with dfu-util:"
    sudo dfu-util -l
else
    echo "dfu-util not installed. Install with: sudo apt-get install dfu-util"
fi


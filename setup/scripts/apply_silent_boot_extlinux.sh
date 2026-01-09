#!/bin/bash
# Quick script to apply silent boot to ExtLinux
# Based on your current configuration

set -e

EXTLINUX_CONFIG="/boot/extlinux/extlinux.conf"

if [ ! -f "$EXTLINUX_CONFIG" ]; then
    echo "Error: ExtLinux config not found at $EXTLINUX_CONFIG"
    exit 1
fi

# Backup original
if [ ! -f "${EXTLINUX_CONFIG}.bak" ]; then
    echo "Creating backup: ${EXTLINUX_CONFIG}.bak"
    sudo cp "$EXTLINUX_CONFIG" "${EXTLINUX_CONFIG}.bak"
fi

echo "Current configuration:"
grep "^[[:space:]]*APPEND" "$EXTLINUX_CONFIG" || echo "APPEND line not found"

echo ""
echo "Applying silent boot configuration..."
echo ""

# Method: Use sed to modify the APPEND line
# 1. Add 'quiet' if not present
# 2. Change mminit_loglevel=4 to mminit_loglevel=0
# 3. Add loglevel=0 if not present

sudo sed -i.bak2 \
    -e 's/\(APPEND[[:space:]]*${cbootargs}\)/\1 quiet/' \
    -e 's/mminit_loglevel=4/mminit_loglevel=0/g' \
    -e 's/\(APPEND.*\)quiet\(.*\)loglevel=[0-9]/\1quiet\2loglevel=0/' \
    -e '/APPEND.*quiet.*loglevel=/!s/\(APPEND.*quiet\)/\1 loglevel=0/' \
    "$EXTLINUX_CONFIG"

echo "Modified configuration:"
grep "^[[:space:]]*APPEND" "$EXTLINUX_CONFIG" || echo "APPEND line not found"

echo ""
echo "✅ Silent boot configuration applied!"
echo ""
echo "Changes:"
echo "  - Added 'quiet' parameter"
echo "  - Changed mminit_loglevel=4 → mminit_loglevel=0"
echo "  - Added loglevel=0"
echo ""
echo "Backup saved at: ${EXTLINUX_CONFIG}.bak"
echo ""
echo "To apply changes, reboot:"
echo "  sudo reboot"

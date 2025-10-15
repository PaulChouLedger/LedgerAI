#!/bin/bash
# Fix Python 3.9+ compatibility issue in ReSpeaker tuning library
# Changes deprecated array.tostring() to array.tobytes()

set -e

TUNING_FILE="$HOME/usb_4_mic_array/tuning.py"

echo "========================================"
echo "  FIX TUNING LIBRARY COMPATIBILITY"
echo "========================================"
echo ""

# Check if file exists
if [ ! -f "$TUNING_FILE" ]; then
    echo "❌ File not found: $TUNING_FILE"
    echo ""
    echo "Please install the tuning library first:"
    echo "  git clone https://github.com/respeaker/usb_4_mic_array.git ~/usb_4_mic_array"
    echo ""
    exit 1
fi

# Check if tostring exists
if ! grep -q "tostring" "$TUNING_FILE"; then
    echo "✅ Library already patched or doesn't need patching"
    exit 0
fi

echo "📝 Backing up original file..."
cp "$TUNING_FILE" "$TUNING_FILE.backup"
echo "   ✅ Backup saved to: $TUNING_FILE.backup"
echo ""

echo "🔧 Applying Python 3.9+ compatibility fix..."
# Replace tostring() with tobytes()
sed -i 's/\.tostring()/.tobytes()/g' "$TUNING_FILE"

# Verify the change
if grep -q "tobytes" "$TUNING_FILE"; then
    echo "   ✅ Successfully replaced tostring() with tobytes()"
    echo ""
    echo "Changes made:"
    grep -n "tobytes" "$TUNING_FILE"
    echo ""
else
    echo "   ❌ Failed to apply patch"
    echo "   Restoring backup..."
    mv "$TUNING_FILE.backup" "$TUNING_FILE"
    exit 1
fi

echo "========================================"
echo "  ✅ TUNING LIBRARY FIXED"
echo "========================================"
echo ""
echo "The ReSpeaker tuning library is now compatible with Python 3.9+"
echo ""
echo "You can now run:"
echo "  sudo python3 scripts/tune_respeaker.py show"
echo "  sudo python3 scripts/tune_respeaker.py reset"
echo ""
echo "To restore original:"
echo "  cp $TUNING_FILE.backup $TUNING_FILE"
echo ""


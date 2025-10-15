#!/bin/bash
#
# Setup ALSA configuration for ReSpeaker on Jetson
# Fixes mmap and buffer alignment issues on ARM
#
# Usage:
#   bash setup_alsa_jetson.sh
#

set -e

echo "================================================================================"
echo "  ReSpeaker ALSA Setup for Jetson"
echo "================================================================================"
echo ""

# Check if ReSpeaker is connected
if ! lsusb | grep -q "2886:0018"; then
    echo "⚠️  Warning: ReSpeaker 4-Mic Array not detected"
    echo "   Connect the device and run this script again"
    echo ""
fi

# Backup existing .asoundrc if it exists
if [ -f ~/.asoundrc ]; then
    echo "📋 Backing up existing ~/.asoundrc to ~/.asoundrc.backup"
    cp ~/.asoundrc ~/.asoundrc.backup
fi

# Copy ALSA configuration
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

if [ -f "$SCRIPT_DIR/.asoundrc" ]; then
    echo "📝 Installing ALSA configuration to ~/.asoundrc"
    cp "$SCRIPT_DIR/.asoundrc" ~/.asoundrc
    echo "✅ ALSA configuration installed"
else
    echo "❌ Error: .asoundrc not found in $SCRIPT_DIR"
    exit 1
fi

# Also set ALSA environment variables for PortAudio
echo ""
echo "📝 Setting ALSA environment variables..."

# Add to .bashrc if not already there
if ! grep -q "ALSA_CARD=ArrayUAC10" ~/.bashrc 2>/dev/null; then
    echo "" >> ~/.bashrc
    echo "# ALSA configuration for ReSpeaker on Jetson" >> ~/.bashrc
    echo "export ALSA_CARD=ArrayUAC10" >> ~/.bashrc
    echo "export AUDIODEV=hw:ArrayUAC10,0" >> ~/.bashrc
    echo "# Disable ALSA mmap for ARM compatibility" >> ~/.bashrc
    echo "export LIBASOUND_THREAD_SAFE=0" >> ~/.bashrc
    echo "" >> ~/.bashrc
    echo "✅ Added ALSA variables to ~/.bashrc"
else
    echo "✅ ALSA variables already in ~/.bashrc"
fi

echo ""
echo "================================================================================"
echo "  Setup Complete!"
echo "================================================================================"
echo ""
echo "📋 Next steps:"
echo ""
echo "1. Reload your shell or run:"
echo "   source ~/.bashrc"
echo ""
echo "2. Verify ALSA sees the device:"
echo "   arecord -l"
echo ""
echo "3. Test recording (Ctrl+C to stop):"
echo "   arecord -D hw:ArrayUAC10,0 -f S16_LE -r 16000 -c 6 test.wav"
echo ""
echo "4. Run LedgerAI:"
echo "   python3 aura-control/main.py"
echo ""
echo "================================================================================"


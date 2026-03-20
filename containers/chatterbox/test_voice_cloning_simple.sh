#!/bin/bash
# Simple voice cloning test - checks file existence first

CHATTERBOX_URL="${CHATTERBOX_URL:-http://localhost:11437}"
VOICE_FILE="audio2.wav"

echo "Testing voice cloning with $VOICE_FILE..."
echo ""

# Check if file exists on host
if [ -f "$HOME/LedgerAI/assets/prompts/$VOICE_FILE" ]; then
    echo "✅ File exists on host: $HOME/LedgerAI/assets/prompts/$VOICE_FILE"
else
    echo "❌ File NOT found on host: $HOME/LedgerAI/assets/prompts/$VOICE_FILE"
    echo "💡 Make sure the file exists before testing"
    exit 1
fi

echo ""
echo "Sending synthesis request with voice cloning..."
echo ""

# Test with voice cloning
curl -X POST "$CHATTERBOX_URL/synthesize" \
  -H "Content-Type: application/json" \
  -d "{\"text\": \"Hello, this is a test of voice cloning with audio2.wav. The voice should match the sample.\", \"voice_sample\": \"$VOICE_FILE\"}" \
  --output test_audio2_cloned.wav \
  --write-out "\nHTTP Status: %{http_code}\n" \
  --silent --show-error

if [ -f test_audio2_cloned.wav ]; then
    echo ""
    echo "✅ Audio file created: test_audio2_cloned.wav"
    ls -lh test_audio2_cloned.wav
    echo ""
    echo "💡 Check container logs to see if voice cloning was used:"
    echo "   docker logs setup-chatterbox-tts-1 | tail -20"
else
    echo ""
    echo "❌ Audio file not created - check container logs for errors"
fi

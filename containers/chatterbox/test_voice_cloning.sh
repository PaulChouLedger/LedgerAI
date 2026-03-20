#!/bin/bash
# Test voice cloning with ChatterboxTTS

CHATTERBOX_URL="${CHATTERBOX_URL:-http://localhost:11437}"
VOICE_SAMPLE="${1:-audio2.wav}"  # Default to audio2.wav from prompts directory
TEXT="${2:-Hello, this is a test of voice cloning with ChatterboxTTS.}"  # Default test text

echo "="*70
echo "ChatterboxTTS Voice Cloning Test"
echo "="*70
echo "URL: $CHATTERBOX_URL"
echo "Voice Sample: $VOICE_SAMPLE"
echo "Text: $TEXT"
echo "="*70
echo

# Test 1: Without voice cloning (default voice)
echo "Test 1: Synthesizing WITHOUT voice cloning (default voice)..."
curl -X POST "$CHATTERBOX_URL/synthesize" \
  -H "Content-Type: application/json" \
  -d "{\"text\": \"$TEXT\"}" \
  --output test_default_voice.wav \
  --silent --show-error

if [ $? -eq 0 ] && [ -f test_default_voice.wav ]; then
    echo "✅ Default voice synthesis successful: test_default_voice.wav"
    ls -lh test_default_voice.wav
else
    echo "❌ Default voice synthesis failed"
fi

echo
echo "="*70
echo

# Test 2: With voice cloning
echo "Test 2: Synthesizing WITH voice cloning (using $VOICE_SAMPLE)..."
curl -X POST "$CHATTERBOX_URL/synthesize" \
  -H "Content-Type: application/json" \
  -d "{\"text\": \"$TEXT\", \"voice_sample\": \"$VOICE_SAMPLE\"}" \
  --output test_cloned_voice.wav \
  --silent --show-error

if [ $? -eq 0 ] && [ -f test_cloned_voice.wav ]; then
    echo "✅ Voice cloning synthesis successful: test_cloned_voice.wav"
    ls -lh test_cloned_voice.wav
    echo
    echo "💡 Compare the two files to hear the difference:"
    echo "   - test_default_voice.wav (default voice)"
    echo "   - test_cloned_voice.wav (cloned from $VOICE_SAMPLE)"
else
    echo "❌ Voice cloning synthesis failed"
    echo "💡 Check container logs: docker logs setup-chatterbox-tts-1"
fi

echo
echo "="*70
echo "Test complete!"

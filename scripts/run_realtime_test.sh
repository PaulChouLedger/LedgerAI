#!/bin/bash

# Run Real-time Microphone Speech Benchmark on Jetson
# This will test both faster-whisper and whisper-container with live speech

echo "🚀 Starting Real-time Microphone Speech Benchmark"
echo "================================================"

# Check if we're in the right directory
if [ ! -f "scripts/realtime_microphone_benchmark.py" ]; then
    echo "❌ Please run this from the LedgerAI root directory"
    exit 1
fi

# Install dependencies if needed
echo "📦 Installing dependencies..."
pip install faster-whisper torch torchaudio soundfile numpy scipy requests psutil sounddevice

# Make script executable
chmod +x scripts/realtime_microphone_benchmark.py

echo "🎤 This will test both models with:"
echo "  1. Existing audio files (sample.wav, startup_test.wav, etc.)"
echo "  2. Live microphone input (5 seconds of recording)"
echo ""
echo "🔧 Make sure your microphone is working and speak clearly!"
echo ""

# Check if whisper container is running
echo "🔍 Checking whisper container..."
if curl -s http://localhost:5000/health > /dev/null 2>&1; then
    echo "✅ Whisper container is running - will test both models"
else
    echo "⚠️ Whisper container not running - will test faster-whisper only"
    echo "   To test both models, start container with: docker compose up whisper"
fi

echo ""
echo "🎯 Starting benchmark..."
echo ""

# Run the benchmark
python scripts/realtime_microphone_benchmark.py

echo ""
echo "🎉 Real-time microphone test complete!"
echo "📄 Results saved to: realtime_microphone_benchmark.json"
echo ""
echo "📋 To view results:"
echo "  cat realtime_microphone_benchmark.json | jq '.comparison'"
echo ""
echo "📋 To start whisper container for full comparison:"
echo "  docker compose up whisper"

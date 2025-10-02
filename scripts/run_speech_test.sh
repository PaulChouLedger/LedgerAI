#!/bin/bash

# Run Speech Accuracy & Latency Test on Jetson
# This will test both faster-whisper and whisper-container with real speech

echo "🚀 Starting Speech Accuracy & Latency Test"
echo "=========================================="

# Check if we're in the right directory
if [ ! -f "scripts/speech_accuracy_benchmark.py" ]; then
    echo "❌ Please run this from the LedgerAI root directory"
    exit 1
fi

# Install dependencies if needed
echo "📦 Installing dependencies..."
pip install faster-whisper torch torchaudio soundfile numpy scipy requests psutil

# Make script executable
chmod +x scripts/speech_accuracy_benchmark.py

echo "🔬 Running speech accuracy benchmark..."
echo "This will test both faster-whisper and whisper-container with real speech samples"
echo ""

# Run the benchmark
python scripts/speech_accuracy_benchmark.py

echo ""
echo "🎉 Speech accuracy test complete!"
echo "📄 Results saved to: speech_accuracy_benchmark.json"
echo ""
echo "📋 To view results:"
echo "  cat speech_accuracy_benchmark.json | jq '.comparison'"
echo ""
echo "📋 To start whisper container for comparison:"
echo "  docker compose up whisper"

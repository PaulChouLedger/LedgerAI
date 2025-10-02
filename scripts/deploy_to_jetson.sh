#!/bin/bash

# Deploy Whisper Benchmarking to Jetson
# Run this script on your Jetson device

echo "🚀 Deploying Whisper Benchmarking to Jetson"
echo "=========================================="

# Check if we're in the LedgerAI directory
if [ ! -f "scripts/quick_whisper_benchmark.py" ]; then
    echo "❌ Please run this from the LedgerAI root directory"
    exit 1
fi

# Install dependencies
echo "📦 Installing dependencies..."
pip install faster-whisper torch torchaudio soundfile numpy scipy requests psutil

# Make scripts executable
echo "🔧 Setting up scripts..."
chmod +x scripts/quick_whisper_benchmark.py
chmod +x scripts/realistic_whisper_benchmark.py
chmod +x scripts/test_whisper_container.py
chmod +x scripts/compare_whisper_models.py

echo "✅ Deployment complete!"
echo ""
echo "🎯 Available benchmarks on Jetson:"
echo "  • Quick test: python scripts/quick_whisper_benchmark.py"
echo "  • Realistic test: python scripts/realistic_whisper_benchmark.py"
echo "  • Container test: python scripts/test_whisper_container.py"
echo "  • Full comparison: python scripts/compare_whisper_models.py"
echo ""
echo "📋 To run benchmarks:"
echo "  1. Test faster-whisper: python scripts/realistic_whisper_benchmark.py"
echo "  2. Start whisper container: docker compose up whisper"
echo "  3. Run comparison: python scripts/compare_whisper_models.py"

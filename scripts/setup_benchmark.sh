#!/bin/bash

# Setup script for Whisper Benchmarking

echo "🚀 Setting up Whisper Benchmarking Environment"
echo "=============================================="

# Check if we're in the right directory
if [ ! -f "scripts/benchmark_whisper.py" ]; then
    echo "❌ Please run this script from the LedgerAI root directory"
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d "benchmark-env" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv benchmark-env
    if [ $? -ne 0 ]; then
        echo "❌ Failed to create virtual environment"
        exit 1
    fi
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source benchmark-env/bin/activate

# Install dependencies
echo "📥 Installing benchmark dependencies..."
pip install --upgrade pip
pip install -r scripts/requirements_benchmark.txt

if [ $? -ne 0 ]; then
    echo "❌ Failed to install dependencies"
    deactivate
    exit 1
fi

# Make scripts executable
echo "🔧 Setting up scripts..."
chmod +x scripts/quick_whisper_benchmark.py
chmod +x scripts/test_whisper_container.py
chmod +x scripts/compare_whisper_models.py

echo "✅ Setup complete!"
echo ""
echo "🎯 Available benchmarks:"
echo "  • Quick faster-whisper test: python scripts/quick_whisper_benchmark.py"
echo "  • Whisper container test: python scripts/test_whisper_container.py"
echo "  • Full comparison: python scripts/compare_whisper_models.py"
echo ""
echo "📋 To start benchmarking:"
echo "  1. Activate environment: source benchmark-env/bin/activate"
echo "  2. Run quick test: python scripts/quick_whisper_benchmark.py"
echo "  3. (Optional) Start whisper container: docker compose up whisper"
echo "  4. Run full comparison: python scripts/compare_whisper_models.py"

deactivate

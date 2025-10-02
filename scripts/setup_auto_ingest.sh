#!/bin/bash
# Setup script for Aura Auto-Ingest Pipeline

set -e

echo "🚀 Setting up Aura Auto-Ingest Pipeline..."

# Check if we're in the right directory
if [ ! -f "scripts/auto_ingest.py" ]; then
    echo "❌ Please run this script from the LedgerAI root directory"
    exit 1
fi

# Install Python dependencies
echo "📦 Installing Python dependencies..."
pip install -r scripts/requirements_ingest.txt

# Create directories
echo "📁 Creating directories..."
mkdir -p data/input
mkdir -p data/parsed
mkdir -p data/embeddings

# Test the pipeline
echo "🧪 Testing the pipeline..."
python scripts/auto_ingest.py --help

echo "✅ Auto-ingest pipeline setup complete!"
echo ""
echo "📋 Usage:"
echo "  # Run once:"
echo "  python scripts/auto_ingest.py"
echo ""
echo "  # Run continuously (every 60 seconds):"
echo "  python scripts/auto_ingest.py --continuous"
echo ""
echo "  # Run continuously (every 30 seconds):"
echo "  python scripts/auto_ingest.py --continuous --interval 30"
echo ""
echo "  # Force rebuild all embeddings:"
echo "  python scripts/auto_ingest.py --rebuild"
echo ""
echo "📁 Supported file types: PDF, DOCX, TXT, MD"
echo "📁 Drop files in: data/input/"
echo "📁 Parsed files: data/parsed/"
echo "📁 Embeddings: data/embeddings/"

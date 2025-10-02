#!/bin/bash
# Install FAISS-GPU for Jetson Orin NX

echo "🔧 Installing FAISS-GPU for Jetson Orin NX..."

# Check if we're on Jetson
if [ -f /etc/nv_tegra_release ]; then
    echo "✅ Jetson detected"
    cat /etc/nv_tegra_release
else
    echo "⚠️ Not running on Jetson - this script is optimized for Jetson Orin NX"
fi

# Check CUDA availability
echo "🔍 Checking CUDA availability..."
nvcc --version 2>/dev/null
if [ $? -eq 0 ]; then
    echo "✅ CUDA available"
else
    echo "❌ CUDA not found - FAISS-GPU requires CUDA"
    exit 1
fi

# Install FAISS-GPU
echo "📦 Installing FAISS-GPU..."
pip3 install faiss-gpu

# Test installation
echo "🧪 Testing FAISS-GPU installation..."
python3 -c "
import faiss
print('FAISS version:', faiss.__version__)
print('GPU devices:', faiss.get_num_gpus())
if faiss.get_num_gpus() > 0:
    print('✅ FAISS-GPU working!')
else:
    print('⚠️ No GPU devices detected')
"

echo "🎉 FAISS-GPU installation complete!"
echo ""
echo "Next steps:"
echo "1. Run: python3 scripts/test_rag.py"
echo "2. Start LLM container: docker-compose up llm-container"
echo "3. Test RAG API endpoints"
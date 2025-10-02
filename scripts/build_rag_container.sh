#!/bin/bash
# Build LLM container with FAISS RAG support

echo "🚀 Building LLM container with FAISS RAG support..."

# Navigate to project root
cd "$(dirname "$0")/.."

# Ensure RAG module is in llm-container directory
if [ ! -f "llm-container/rag.py" ]; then
    echo "📁 Copying RAG module to container..."
    cp aura-control/rag.py llm-container/rag.py
fi

# Build the container
echo "🔨 Building Docker container..."
cd llm-container
docker build --no-cache -t aura-llm-rag .

if [ $? -eq 0 ]; then
    echo "✅ Container built successfully!"
    echo ""
    echo "📋 Next steps:"
    echo "1. Update docker-compose.yml to use 'aura-llm-rag' image"
    echo "2. Start container: docker-compose up llm-container"
    echo "3. Test RAG endpoints:"
    echo "   curl http://localhost:11434/rag/stats"
    echo "   curl -X POST http://localhost:11434/rag/search -H 'Content-Type: application/json' -d '{\"query\": \"chest pain symptoms\", \"k\": 3}'"
else
    echo "❌ Container build failed!"
    exit 1
fi

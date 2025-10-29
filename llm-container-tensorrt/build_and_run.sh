#!/bin/bash
# TensorRT-LLM Container Build and Run Script

set -e

echo "[TensorRT-LLM] 🚀 Building TensorRT-LLM Container"

# Check if NVIDIA Docker runtime is available
if ! docker info | grep -q nvidia; then
    echo "[TensorRT-LLM] ⚠️  NVIDIA Docker runtime not detected"
    echo "[TensorRT-LLM] 💡 Make sure nvidia-docker2 is installed"
fi

# Build the container
echo "[TensorRT-LLM] 🔨 Building Docker image..."
docker-compose -f llm-container-tensorrt/docker-compose.yml build

# Create shared directory if it doesn't exist
mkdir -p shared/input_audio shared/output_audio

# Start the container
echo "[TensorRT-LLM] 🚀 Starting TensorRT-LLM container..."
docker-compose -f llm-container-tensorrt/docker-compose.yml up -d

# Wait for container to be ready
echo "[TensorRT-LLM] ⏳ Waiting for container to be ready..."
sleep 10

# Check health
echo "[TensorRT-LLM] 🏥 Checking container health..."
curl -s http://localhost:11435/health | jq . || echo "[TensorRT-LLM] ⚠️  Health check failed"

echo "[TensorRT-LLM] ✅ TensorRT-LLM container is running on port 11435"
echo "[TensorRT-LLM] 📋 Available endpoints:"
echo "[TensorRT-LLM]   - GET  http://localhost:11435/health"
echo "[TensorRT-LLM]   - POST http://localhost:11435/load-model"
echo "[TensorRT-LLM]   - POST http://localhost:11435/generate"
echo "[TensorRT-LLM]   - GET  http://localhost:11435/models"

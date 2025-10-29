# TensorRT-LLM Container Documentation

## Overview

This container provides high-performance LLM inference using NVIDIA's TensorRT-LLM framework, built on the optimized `dustynv/tensorrt_llm:0.12-r36.4.0` base image for maximum performance and compatibility.

## Features

- **High Performance**: TensorRT-LLM optimized inference with dustynv optimizations
- **GPU Acceleration**: Full CUDA support with NVIDIA GPUs
- **Pre-optimized**: Uses dustynv's tested and optimized TensorRT-LLM build
- **Model Flexibility**: Support for various model formats
- **REST API**: Simple HTTP interface for integration
- **Auto-loading**: Automatic model discovery and loading

## Architecture

Based on the [jetson-containers TensorRT-LLM implementation](https://github.com/dusty-nv/jetson-containers/tree/master/packages/llm/tensorrt_optimizer/tensorrt_llm), this container provides:

1. **TensorRT-LLM Engine**: Core inference engine (dustynv optimized)
2. **REST API**: Flask-based web service
3. **Model Management**: Dynamic model loading/unloading
4. **Health Monitoring**: Built-in health checks

## Base Image Benefits

Using `dustynv/tensorrt_llm:0.12-r36.4.0` provides:

- **Pre-compiled TensorRT-LLM**: No build time required
- **Optimized for Jetson**: Tested on NVIDIA hardware
- **Latest Features**: Includes latest TensorRT-LLM improvements
- **Stable Dependencies**: All dependencies pre-configured

## Quick Start

```bash
# Build and run the container
cd llm-container-tensorrt
./build_and_run.sh
```

## API Endpoints

### Health Check
```bash
GET /health
```

### Load Model
```bash
POST /load-model
Content-Type: application/json

{
  "model_path": "/models/Mistral-7B-Instruct-v0.3.Q4_K_M.gguf",
  "model_name": "mistral-7b"
}
```

### Generate Text
```bash
POST /generate
Content-Type: application/json

{
  "prompt": "What is the capital of France?",
  "model_name": "mistral-7b",
  "max_tokens": 100
}
```

### List Models
```bash
GET /models
```

## Model Conversion

To use TensorRT-LLM, models need to be converted from their original format:

1. **From HuggingFace**: Use the TensorRT-LLM conversion scripts
2. **From GGUF**: Convert to TensorRT format using provided tools
3. **From ONNX**: Use TensorRT-LLM's ONNX import capabilities

## Performance Benefits

- **3-5x faster inference** compared to standard PyTorch
- **Lower memory usage** through TensorRT optimizations
- **Batch processing** support for multiple requests
- **Dynamic batching** for optimal throughput

## Integration with LedgerAI

This container can be integrated with the existing LedgerAI system by:

1. Updating the `llm-medical-container` to use TensorRT-LLM API
2. Modifying the RAG client to route requests to this container
3. Implementing fallback mechanisms for compatibility

## Requirements

- NVIDIA GPU with CUDA support
- NVIDIA Docker runtime (nvidia-docker2)
- Docker Compose
- At least 8GB GPU memory for 7B models

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**: Reduce batch size or use smaller models
2. **Model Loading Failed**: Check model format compatibility
3. **Performance Issues**: Verify GPU utilization and TensorRT installation

### Debugging

```bash
# Check container logs
docker logs llm-tensorrt-1

# Monitor GPU usage
nvidia-smi

# Test API endpoints
curl http://localhost:11435/health
```

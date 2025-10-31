# TensorRT-LLM on Jetson - Installation Notes

## ⚠️ Important: Pip Installation Not Supported

**TensorRT-LLM pip installation is NOT supported on Jetson/Tegra systems.**

The error you encountered:
```
RuntimeError: TensorRT does not currently build wheels for Tegra systems
```

This is expected - NVIDIA does not provide pip wheels for Tegra architectures.

## Solution: Use Pre-built Containers

We use **DustyNV's TensorRT-LLM container** which has TensorRT-LLM pre-built and optimized for Jetson:

```dockerfile
FROM dustynv/tensorrt_llm:0.12-r36.4.0
```

This container includes:
- ✅ TensorRT-LLM pre-compiled for Jetson Orin
- ✅ CUDA environment configured
- ✅ Python 3.10+ with all TensorRT-LLM dependencies
- ✅ `trtllm-build` command available
- ✅ TensorRT-LLM Python API available

## Checkpoint Conversion Workaround

Since the official `convert_checkpoint.py` script may have Python 3.10 compatibility issues, our build script includes a fallback method:

1. **Method 1**: Try official `convert_checkpoint.py` (if it works)
2. **Method 2**: Use `transformers.save_pretrained()` with manual `rank0/` structure creation

The fallback method creates the correct checkpoint format that TensorRT-LLM expects:
```
checkpoint/
  ├── config.json
  ├── rank0/
  │   ├── model.safetensors
  │   └── config.json
```

## Building the Container

```bash
cd ~/LedgerAI/llm-container
docker build -t ledger-llm-container .
```

## Building Engines

```bash
docker run --rm -it --gpus all \
  -v $(pwd)/models:/models \
  ledger-llm-container \
  bash /app/scripts/build_tensorrt_engine.sh llama-3.2-1b /models/Llama/Llama-3.2-1B-Instruct
```

## Alternative: NGC Container

If you prefer NVIDIA's official container:
```dockerfile
FROM nvcr.io/nvidia/tensorrt-llm:0.12.0-py3
```

However, this may not be optimized for Jetson and may require additional configuration.


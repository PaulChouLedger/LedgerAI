# Building TensorRT-LLM Engines

This guide explains how to build TensorRT-LLM engines for use with the LLM container.

## Prerequisites

1. **Source Model**: You need the model in HuggingFace format (not GGUF)
   - For Qwen models: Download from HuggingFace (e.g., `Qwen/Qwen3-4B-Instruct`)
   - For Llama models: Download from HuggingFace (e.g., `meta-llama/Llama-3.2-1B-Instruct`)

2. **TensorRT-LLM Container**: The build must run inside the TensorRT-LLM container
   - Base image: `dustynv/tensorrt_llm:0.12-r36.4.0`

## Quick Start

### Method 1: Using the Build Script

```bash
# Run inside TensorRT-LLM container
docker run --rm -it --gpus all \
  -v /path/to/models:/models \
  -v $(pwd)/llm-container/scripts:/scripts \
  dustynv/tensorrt_llm:0.12-r36.4.0 \
  bash /scripts/build_tensorrt_engine.sh qwen3-4b-2507 /models/Qwen/Qwen3-4B-Instruct
```

### Method 2: Manual Build

```bash
# Run inside TensorRT-LLM container
docker run --rm -it --gpus all \
  -v /path/to/models:/models \
  dustynv/tensorrt_llm:0.12-r36.4.0 \
  bash

# Inside container:
trtllm-build \
  --checkpoint_dir /models/Qwen/Qwen3-4B-Instruct \
  --output_dir /models/tensorrt-llm/qwen3-4b-instruct-2507 \
  --gemm_plugin float16 \
  --gpt_attention_plugin float16 \
  --context_fmha enable \
  --max_batch_size 1 \
  --max_input_len 2048 \
  --max_output_len 512 \
  --max_beam_width 1 \
  --builder_opt 3
```

## Supported Models

| Model Name | Config Key | Context Window | Example Path |
|------------|-----------|----------------|--------------|
| Qwen3-4B-Instruct | `qwen3-4b-2507` | 2048 | `/models/Qwen/Qwen3-4B-Instruct` |
| Llama-3.2-1B-Instruct | `llama-3.2-1b` | 2048 | `/models/Llama/Llama-3.2-1B-Instruct` |
| Llama-3.1-8B-Instruct | `llama-3.1-8b` | 8192 | `/models/Llama/Llama-3.1-8B-Instruct` |

## Build Parameters Explained

- `--gemm_plugin float16`: Use FP16 for matrix operations (faster, less memory)
- `--gpt_attention_plugin float16`: Use FP16 for attention (faster)
- `--context_fmha enable`: Enable Flash Attention (faster attention)
- `--max_batch_size 1`: Maximum batch size (1 for single requests)
- `--max_input_len 2048`: Maximum input context window
- `--max_output_len 512`: Maximum tokens to generate
- `--builder_opt 3`: Optimization level (3 = highest)

## Output Structure

After building, the engine directory should contain:
```
/models/tensorrt-llm/qwen3-4b-instruct-2507/
  ├── config.json
  ├── engine files (.engine, .plan, etc.)
  └── tokenizer files (if copied)
```

## Mounting Engines in Docker

Update `docker-compose.yml` to mount the engines:

```yaml
services:
  llm:
    volumes:
      - /path/to/tensorrt-llm:/models/tensorrt-llm
```

Or set in `.env`:
```bash
TENSORRT_ENGINES_BASE=/models/tensorrt-llm
TENSORRT_ENGINE_DIR=/models/tensorrt-llm/qwen3-4b-instruct-2507
```

## Troubleshooting

### "trtllm-build not found"
- Ensure you're running inside the TensorRT-LLM container
- Check that the container image includes TensorRT-LLM

### "Out of memory during build"
- Reduce `--max_input_len` or `--max_output_len`
- Use `--builder_opt 1` or `2` instead of `3`
- Ensure Jetson has enough swap space

### "Model checkpoint not found"
- Verify the source model path is correct
- Ensure model is in HuggingFace format (not GGUF)
- Check file permissions

### "Engine directory already exists"
- Remove or rename existing engine directory
- Or specify a different `--output_dir`

## Converting GGUF to HuggingFace Format

If you only have GGUF models, you'll need to convert them first:

```python
# Install: pip install transformers accelerate
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load GGUF model (using llama.cpp Python bindings)
# ... load model ...

# Save as HuggingFace format
model.save_pretrained("/path/to/output")
tokenizer.save_pretrained("/path/to/output")
```

Note: This conversion is resource-intensive and may require the full model to be loaded in RAM.

## Next Steps

1. Build the engine using the script or manual commands
2. Verify the engine directory exists and contains required files
3. Update docker-compose.yml to mount the engine directory
4. Set environment variables in .env
5. Restart the container and verify it loads the engine


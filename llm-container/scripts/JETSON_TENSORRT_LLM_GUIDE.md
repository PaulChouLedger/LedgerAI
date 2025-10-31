# TensorRT-LLM on Jetson - Quick Reference

Based on [Jetson AI Lab TensorRT-LLM Guide](https://www.jetson-ai-lab.com/tensorrt_llm.html)

## Standard Script Locations

The `dustynv/tensorrt_llm:0.12-r36.4.0` container includes TensorRT-LLM with scripts in:

- `/opt/TensorRT-LLM/examples/llama/` - Llama-specific examples and scripts
- `/opt/TensorRT-LLM/llama.sh` - Automated Llama build script

## Quick Build (Using Provided Script)

The container includes an automated script for Llama-7B:

```bash
jetson-containers run \
  -e HUGGINGFACE_TOKEN=YOUR_API_KEY \
  -e FORCE_BUILD=on \
  dustynv/tensorrt_llm:0.12-r36.4.0 \
    /opt/TensorRT-LLM/llama.sh
```

## Manual Build (What Our Script Does)

For custom models like Llama-3.2-1B, our script:

1. **Finds conversion script**: Checks `/opt/TensorRT-LLM/examples/llama/convert_checkpoint.py`
2. **Converts checkpoint**: HuggingFace → TensorRT-LLM format
3. **Builds engine**: Uses `trtllm-build` with optimized settings

## OpenAI-Compatible Server

After building, you can start an OpenAI-compatible server:

```bash
jetson-containers run \
  dustynv/tensorrt_llm:0.12-r36.4.0 \
  python3 /opt/TensorRT-LLM/examples/apps/openai_server.py \
    /data/models/tensorrt_llm/your-model-engine
```

This provides a standard OpenAI API endpoint at `http://localhost:8000`.

## Key Differences from Generic TensorRT-LLM

1. **Script Location**: Uses `/opt/TensorRT-LLM/` instead of Python site-packages
2. **Jetson Optimized**: Container is pre-optimized for Jetson Orin
3. **Example Scripts**: Includes ready-to-use example scripts

## Our Build Script Integration

Our `build_tensorrt_engine.sh` now checks:
1. `/opt/TensorRT-LLM/examples/llama/convert_checkpoint.py` (Jetson standard)
2. Python site-packages locations (fallback)
3. Transformers fallback (if official script unavailable)

This ensures compatibility with the Jetson AI Lab standard container setup.


# TensorRT-LLM Build Troubleshooting

## Issue: `AssertionError` - weights_path not found

TensorRT-LLM is looking for model weights but can't find them. This usually means:

1. **Model needs checkpoint conversion first**
   - TensorRT-LLM might need models converted from HuggingFace format to TensorRT-LLM checkpoint format
   - Use `convert_checkpoint.py` or similar tools

2. **Model file structure issue**
   - Verify all model files downloaded correctly
   - Check for `model.safetensors` or `pytorch_model.bin` files

## Quick Check on Jetson

```bash
# Check model files
ls -lh ~/LedgerAI/llm-container/models/Llama/Llama-3.2-1B-Instruct/

# Should see:
# - config.json
# - model.safetensors (or pytorch_model.bin)
# - tokenizer.json
# - tokenizer_config.json
# - etc.
```

## Solution 1: Convert to TensorRT-LLM Checkpoint Format

TensorRT-LLM might require converting the HuggingFace model first:

```bash
# Inside TensorRT-LLM container
python3 -m tensorrt_llm.models.llama.convert_checkpoint \
    --model_dir /models/Llama/Llama-3.2-1B-Instruct \
    --output_dir /models/Llama/Llama-3.2-1B-Instruct-converted \
    --dtype float16
```

Then use the converted directory for building.

## Solution 2: Use Nemo Checkpoint Format

If available, use Nemo checkpoint format instead:

```bash
# Convert HuggingFace → Nemo → TensorRT-LLM
```

## Solution 3: Check TensorRT-LLM Version Compatibility

The error might indicate a version mismatch. Check:

```bash
# In TensorRT-LLM container
python3 -c "import tensorrt_llm; print(tensorrt_llm.__version__)"
```

## Solution 4: Alternative - Use Pre-built Engines

If conversion is too complex, consider:
- Using pre-built TensorRT-LLM engines from NVIDIA/HuggingFace
- Using a different model format (GGUF → convert → TensorRT-LLM)

## Current Error Analysis

The error `assert os.path.isfile(weights_path)` in `from_checkpoint()` suggests:
- TensorRT-LLM expects weights in a specific file location
- The checkpoint directory structure might not match expectations
- Model might need preprocessing/conversion step

Check the TensorRT-LLM documentation for your specific version:
- https://nvidia.github.io/TensorRT-LLM/


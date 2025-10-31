# Building TensorRT-LLM Engine

## Prerequisites

1. **Build the container** (if not already built):
   ```bash
   cd ~/LedgerAI/llm-container
   docker build -t ledger-llm-container .
   ```

2. **Ensure model is downloaded**:
   - Model should be in `~/LedgerAI/llm-container/models/Llama/Llama-3.2-1B-Instruct/`
   - Should contain: `config.json`, `model.safetensors`, `tokenizer.json`, etc.

## Build Engine Command

```bash
cd ~/LedgerAI/llm-container

docker run --rm -it --gpus all \
  -v $(pwd)/models:/models \
  -v $(pwd)/scripts:/scripts \
  ledger-llm-container \
  bash /app/scripts/build_tensorrt_engine.sh llama-3.2-1b /models/Llama/Llama-3.2-1B-Instruct
```

## Command Breakdown

- `--rm`: Remove container after build
- `--it`: Interactive terminal
- `--gpus all`: Enable GPU access (required for TensorRT-LLM)
- `-v $(pwd)/models:/models`: Mount models directory
- `-v $(pwd)/scripts:/scripts`: Mount scripts directory
- `ledger-llm-container`: Container name
- `bash /app/scripts/build_tensorrt_engine.sh`: Run the build script
- `llama-3.2-1b`: Model name (used for configuration)
- `/models/Llama/Llama-3.2-1B-Instruct`: Path to HuggingFace model in container

## Output Location

The engine will be built to:
```
~/LedgerAI/llm-container/models/tensorrt-llm/llama-3.2-1b-instruct/
```

## Alternative Models

For other models, adjust the model name and path:

```bash
# Example: Qwen model
docker run --rm -it --gpus all \
  -v $(pwd)/models:/models \
  -v $(pwd)/scripts:/scripts \
  ledger-llm-container \
  bash /app/scripts/build_tensorrt_engine.sh qwen3-4b /models/Qwen/Qwen3-4B-Instruct-2507
```

## Troubleshooting

If the build fails:
1. Check that the model directory contains all required files
2. Verify GPU is accessible: `nvidia-smi` should work
3. Check container logs for specific errors
4. Ensure sufficient disk space (engines can be large)


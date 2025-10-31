# Building Specific Models

This guide provides exact commands for building TensorRT-LLM engines for specific models.

## Meta Llama 3.1-8B-Instruct

Based on: [Meta's Llama 3.1 models collection](https://huggingface.co/collections/meta-llama/metas-llama-31-models-and-evals)

### Download Model

```bash
# Using huggingface-cli (recommended)
huggingface-cli download meta-llama/Llama-3.1-8B-Instruct \
  --local-dir /models/Llama/Llama-3.1-8B-Instruct \
  --local-dir-use-symlinks False

# Or using git lfs
git lfs install
git clone https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct /models/Llama/Llama-3.1-8B-Instruct
```

### Build Engine

```bash
# Using the build script
docker run --rm -it --gpus all \
  -v /path/to/models:/models \
  -v $(pwd)/llm-container/scripts:/scripts \
  dustynv/tensorrt_llm:0.12-r36.4.0 \
  bash /scripts/build_tensorrt_engine.sh llama-3.1-8b-instruct /models/Llama/Llama-3.1-8B-Instruct

# Or manual build
docker run --rm -it --gpus all \
  -v /path/to/models:/models \
  dustynv/tensorrt_llm:0.12-r36.4.0 \
  bash

# Inside container:
trtllm-build \
  --checkpoint_dir /models/Llama/Llama-3.1-8B-Instruct \
  --output_dir /models/tensorrt-llm/llama-3.1-8b-instruct \
  --gemm_plugin float16 \
  --gpt_attention_plugin float16 \
  --context_fmha enable \
  --max_batch_size 1 \
  --max_input_len 8192 \
  --max_output_len 512 \
  --max_beam_width 1 \
  --builder_opt 3
```

**Model Specs:**
- Size: ~16GB (HuggingFace format)
- Context Window: 8192 tokens
- Chat Format: `llama-3.1`
- Best for: General conversation, complex reasoning

**Jetson Orin Considerations:**
- This is an 8B model - may be tight on 16GB Jetson Orin
- Consider using `--builder_opt 1` or `2` if build fails
- Reduce `--max_input_len` to 4096 if memory constrained

---

## Qwen2.5-Coder-7B-Instruct

Based on: [Qwen2.5-Coder-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct)

### Download Model

```bash
# Using huggingface-cli
huggingface-cli download Qwen/Qwen2.5-Coder-7B-Instruct \
  --local-dir /models/Qwen/Qwen2.5-Coder-7B-Instruct \
  --local-dir-use-symlinks False

# Or using git lfs
git lfs install
git clone https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct /models/Qwen/Qwen2.5-Coder-7B-Instruct
```

### Build Engine

```bash
# Using the build script
docker run --rm -it --gpus all \
  -v /path/to/models:/models \
  -v $(pwd)/llm-container/scripts:/scripts \
  dustynv/tensorrt_llm:0.12-r36.4.0 \
  bash /scripts/build_tensorrt_engine.sh qwen2.5-coder-7b-instruct /models/Qwen/Qwen2.5-Coder-7B-Instruct

# Or manual build (with reduced context for Jetson)
docker run --rm -it --gpus all \
  -v /path/to/models:/models \
  dustynv/tensorrt_llm:0.12-r36.4.0 \
  bash

# Inside container (note: reduced context window for Jetson)
trtllm-build \
  --checkpoint_dir /models/Qwen/Qwen2.5-Coder-7B-Instruct \
  --output_dir /models/tensorrt-llm/qwen2.5-coder-7b-instruct \
  --gemm_plugin float16 \
  --gpt_attention_plugin float16 \
  --context_fmha enable \
  --max_batch_size 1 \
  --max_input_len 4096 \
  --max_output_len 512 \
  --max_beam_width 1 \
  --builder_opt 2
```

**Model Specs:**
- Size: ~15.2GB (HuggingFace format)
- Original Context Window: 32768 tokens
- **Recommended for Jetson: 4096-8192 tokens** (reduce to fit memory)
- Chat Format: `qwen2`
- Best for: Code generation, programming assistance

**Jetson Orin Considerations:**
- 7B model - will be tight on 16GB Jetson Orin
- **Important:** Full 32k context is too large - reduce to 4096 or 8192
- Use `--builder_opt 2` instead of `3` for stability
- May need quantization or INT8 for 16GB Jetson

---

## Comparison Table

| Model | Size | Context | Best For | Jetson 16GB |
|-------|------|---------|----------|-------------|
| Llama-3.1-8B-Instruct | ~16GB | 8192 | General, reasoning | ⚠️ Tight fit |
| Qwen2.5-Coder-7B-Instruct | ~15.2GB | 32768→4096* | Code generation | ⚠️ Tight fit |

\* Reduced for Jetson compatibility

## Recommended Models for Jetson Orin 16GB

For better compatibility with 16GB Jetson Orin, consider:

1. **Llama-3.2-1B-Instruct** - ✅ Well-suited (smaller)
2. **Qwen3-4B-Instruct** - ✅ Better fit (medium)
3. **Llama-3.1-8B-Instruct** (INT8 quantized) - ⚠️ If needed
4. **Qwen2.5-Coder-7B-Instruct** (INT8 quantized) - ⚠️ If needed

## Building with INT8 Quantization

For larger models on Jetson, consider INT8 quantization:

```bash
trtllm-build \
  --checkpoint_dir /models/Llama/Llama-3.1-8B-Instruct \
  --output_dir /models/tensorrt-llm/llama-3.1-8b-instruct-int8 \
  --gemm_plugin int8 \
  --gpt_attention_plugin int8 \
  --max_input_len 4096 \
  --max_output_len 512 \
  --builder_opt 2
```

## Next Steps

1. Download the model using `huggingface-cli` or `git lfs`
2. Build the engine using the commands above
3. Update `.env` with the model configuration:
   ```bash
   SIMPLE_MODEL_NAME=llama-3.1-8b-instruct
   # or
   SIMPLE_MODEL_NAME=qwen2.5-coder-7b-instruct
   ```
4. Mount the engine directory in `docker-compose.yml`
5. Restart the container


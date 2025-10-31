# Building Fast Model for 1-2 Second Latency

## Recommended Model: **Llama-3.2-1B-Instruct**

This is the **best model for 1-2 second latency** on Jetson Orin 16GB.

### Why Llama-3.2-1B-Instruct?

- ✅ **Smallest model**: ~700MB-1GB (vs 4-8GB for others)
- ✅ **Fastest inference**: Optimized for low-latency
- ✅ **Good quality**: Despite small size, performs well for medical conversations
- ✅ **Fits Jetson**: Perfect for 16GB Jetson Orin
- ✅ **TensorRT-LLM optimized**: Builds very fast engines

### Expected Performance

| Metric | Target | Typical Result |
|--------|--------|----------------|
| **First token latency** | <500ms | ~300-500ms |
| **Full response (50 tokens)** | 1-2s | ~1-1.5s |
| **Memory usage** | Low | ~2-3GB |
| **Context window** | 2048 | 2048 tokens |

---

## Step 1: Download Model

```bash
# Using huggingface-cli (recommended)
huggingface-cli download meta-llama/Llama-3.2-1B-Instruct \
  --local-dir /models/Llama/Llama-3.2-1B-Instruct \
  --local-dir-use-symlinks False

# Or using git lfs
git lfs install
git clone https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct /models/Llama/Llama-3.2-1B-Instruct
```

**Note**: You need HuggingFace access for Llama models. Request access at: https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct

---

## Step 2: Build TensorRT-LLM Engine (Optimized for Speed)

```bash
# Using the build script
docker run --rm -it --gpus all \
  -v /path/to/models:/models \
  -v $(pwd)/llm-container/scripts:/scripts \
  dustynv/tensorrt_llm:0.12-r36.4.0 \
  bash /scripts/build_tensorrt_engine.sh llama-3.2-1b /models/Llama/Llama-3.2-1B-Instruct

# Or manual build with speed optimizations
docker run --rm -it --gpus all \
  -v /path/to/models:/models \
  dustynv/tensorrt_llm:0.12-r36.4.0 \
  bash

# Inside container - optimized for low latency:
trtllm-build \
  --checkpoint_dir /models/Llama/Llama-3.2-1B-Instruct \
  --output_dir /models/tensorrt-llm/llama-3.2-1b-instruct \
  --gemm_plugin float16 \
  --gpt_attention_plugin float16 \
  --context_fmha enable \
  --max_batch_size 1 \
  --max_input_len 2048 \
  --max_output_len 256 \
  --max_beam_width 1 \
  --builder_opt 3 \
  --remove_input_padding enable
```

### Speed Optimization Parameters

- `--max_output_len 256`: Limit output for faster generation
- `--remove_input_padding enable`: Remove padding for faster processing
- `--builder_opt 3`: Maximum optimization level
- `float16`: Faster than INT8, good quality

---

## Step 3: Configure Environment

Update `.env` file:

```bash
# Fast model configuration (1-2s latency)
SIMPLE_MODEL_NAME=llama-3.2-1b
SIMPLE_CHAT_FORMAT=llama-3
SIMPLE_N_CTX=2048

# Generation parameters (optimized for speed)
LLM_TEMPERATURE_SIMPLE=0.6
LLM_TOP_P=0.9
LLM_TOP_K=40
LLM_REPEAT_PENALTY=1.15
LLM_NUM_PREDICT=100  # Limit tokens for faster responses
LLM_STOP=\n\n
```

---

## Step 4: Update Docker Compose

Ensure the engine directory is mounted:

```yaml
services:
  llm:
    volumes:
      - /path/to/tensorrt-llm:/models/tensorrt-llm
    environment:
      - TENSORRT_ENGINES_BASE=/models/tensorrt-llm
      - SIMPLE_MODEL_NAME=llama-3.2-1b
```

---

## Alternative Fast Models

If Llama-3.2-1B doesn't meet quality requirements, consider:

### 1. Qwen3-4B-Instruct (Balanced)
- **Latency**: ~2-3s
- **Size**: ~2.5GB
- **Quality**: Better than 1B, still fast
- **Build**: Same process, use `qwen3-4b-2507`

### 2. Llama-3.1-8B-Instruct (INT8 Quantized)
- **Latency**: ~3-4s
- **Size**: ~8GB (quantized)
- **Quality**: Best quality, slower
- **Build**: Add `--gemm_plugin int8 --gpt_attention_plugin int8`

---

## Performance Tips

1. **Limit max tokens**: Set `LLM_NUM_PREDICT=100` for faster responses
2. **Use FP16**: Fastest without significant quality loss
3. **Disable streaming**: Non-streaming can be faster for short responses
4. **Optimize context**: Keep context window at 2048 (default)

---

## Verification

After building and starting the container, verify latency:

```bash
# Test endpoint
curl -X POST http://localhost:11434/chat-generic \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello, how are you?", "max_tokens": 50}'

# Check logs for timing
docker logs llm-container | grep "latency\|time"
```

Expected: Response in **1-2 seconds** for typical medical questions.

---

## Troubleshooting

### Still too slow (>3s)?
- Reduce `LLM_NUM_PREDICT` to 50-80
- Check GPU utilization: `sudo tegrastats`
- Ensure TensorRT-LLM engine was built correctly

### Out of memory?
- 1B model should fit easily
- Check: `free -h` to see memory usage
- Clear other processes if needed

### Poor quality?
- Increase `LLM_TEMPERATURE_SIMPLE` to 0.7
- Or switch to Qwen3-4B for better quality (slightly slower)


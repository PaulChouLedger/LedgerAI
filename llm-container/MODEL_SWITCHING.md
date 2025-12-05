# Switching Between Qwen2.5 Models (1.5B vs 3B)

Both models are available in the container. Switch between them using the `SIMPLE_MODEL_PATH` environment variable.

## Available Models

1. **Qwen2.5-1.5B-Instruct** (`/models/Qwen2.5-1.5B-Instruct.Q4_K_M.gguf`)
   - Faster inference
   - Lower memory usage
   - Good for simple queries

2. **Qwen2.5-3B-Instruct** (`/models/qwen2.5-3b-instruct-q4_k_m.gguf`)
   - Better reasoning capabilities
   - More accurate complex queries
   - Higher memory usage
   - Default model

## Switch via Docker Compose

Edit `setup/docker-compose.yml` and add to the `llm-generic` service environment:

```yaml
llm-generic:
  environment:
    - SIMPLE_MODEL_PATH=/models/Qwen2.5-1.5B-Instruct.Q4_K_M.gguf      # Switch to 1.5B
    # OR
    - SIMPLE_MODEL_PATH=/models/qwen2.5-3b-instruct-q4_k_m.gguf        # Use 3B (default)
```

Then restart:
```bash
cd setup
docker-compose restart llm-generic
```

## Switch via .env File

Add to `.env`:
```bash
SIMPLE_MODEL_PATH=/models/Qwen2.5-1.5B-Instruct.Q4_K_M.gguf      # For 1.5B
# OR
SIMPLE_MODEL_PATH=/models/qwen2.5-3b-instruct-q4_k_m.gguf        # For 3B (default)
```

## Current Default

Default is set to **3B model** for better reasoning on complex queries like co-founder extraction.


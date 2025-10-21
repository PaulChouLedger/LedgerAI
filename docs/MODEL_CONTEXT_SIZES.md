# Model Context Sizes Reference

**Different models have different optimal context sizes!**

---

## 🎯 Why Context Size Matters

**Context size** = How much text the model can "remember" at once

```
Small context (2048):
- Fast
- Less memory
- Good for simple tasks
- Example: Templates, validation

Large context (32768+):
- Slower
- More memory
- Can handle long conversations
- Example: Complex diagnostics
```

---

## 📊 Recommended Context Sizes by Model

### Mistral Models

| Model | Recommended N_CTX | Max N_CTX | Use Case |
|-------|-------------------|-----------|----------|
| **Mistral-7B-Instruct-v0.3** | 8192 | 32768 | Medical reasoning |
| **Mistral-Nemo-12B** | 8192 | 128000 | Complex medical cases |
| **Mistral-Small-22B** | 8192 | 32768 | Advanced diagnostics |

**Config:**
```bash
MODEL_PATH=/models/Mistral-7B-Instruct-v0.3.Q4_K_M.gguf
CHAT_FORMAT=mistral-instruct
N_CTX=8192      # Good balance
# N_CTX=32768   # For very long conversations
```

---

### Llama Models

| Model | Recommended N_CTX | Max N_CTX | Use Case |
|-------|-------------------|-----------|----------|
| **Llama-3.1-8B-Instruct** | 8192 | 128000 | Medical reasoning |
| **Llama-3.2-3B-Instruct** | 4096 | 128000 | Lightweight reasoning |
| **Llama-3.2-1B-Instruct** | 2048 | 8192 | Simple tasks, templates |

**Config:**
```bash
# Complex model
MODEL_PATH=/models/Llama-3.1-8B-Instruct-Q4_K_M.gguf
CHAT_FORMAT=llama-3
N_CTX=8192

# Simple model
SIMPLE_MODEL_PATH=/models/Llama-3.2-1B-Instruct-Q4_K_M.gguf
SIMPLE_CHAT_FORMAT=llama-3
SIMPLE_N_CTX=2048
```

---

### Gemma Models

| Model | Recommended N_CTX | Max N_CTX | Use Case |
|-------|-------------------|-----------|----------|
| **Gemma-2-9B-Instruct** | 8192 | 8192 | Medical reasoning |
| **Gemma-2-2B-Instruct** | 2048 | 8192 | Simple tasks |

**Config:**
```bash
MODEL_PATH=/models/gemma-2-9b-it-Q6_K_L.gguf
CHAT_FORMAT=gemma
N_CTX=8192

SIMPLE_MODEL_PATH=/models/gemma-2-2b-it-Q8_0.gguf
SIMPLE_CHAT_FORMAT=gemma
SIMPLE_N_CTX=2048
```

---

### Qwen Models

| Model | Recommended N_CTX | Max N_CTX | Use Case |
|-------|-------------------|-----------|----------|
| **Qwen2.5-7B-Instruct** | 8192 | 32768 | Medical reasoning |
| **Qwen2.5-14B-Instruct** | 8192 | 32768 | Advanced diagnostics |

**Config:**
```bash
MODEL_PATH=/models/Qwen2.5-7B-Instruct-Q4_K_M.gguf
CHAT_FORMAT=qwen
N_CTX=8192
```

---

### Phi Models

| Model | Recommended N_CTX | Max N_CTX | Use Case |
|-------|-------------------|-----------|----------|
| **Phi-3.5-Mini-Instruct** | 4096 | 128000 | Lightweight, long context |
| **Phi-3-Medium-Instruct** | 4096 | 128000 | Balanced performance |

**Config:**
```bash
MODEL_PATH=/models/Phi-3.5-mini-instruct-Q4_K_M.gguf
CHAT_FORMAT=phi
N_CTX=4096
# N_CTX=128000  # For very long medical histories
```

---

## 🎚️ How to Choose Context Size

### For Medical Use (Recommended Settings)

```bash
# Option 1: Balanced (recommended)
N_CTX=8192
SIMPLE_N_CTX=2048

# Option 2: Long conversations
N_CTX=16384
SIMPLE_N_CTX=4096

# Option 3: Very long medical histories
N_CTX=32768
SIMPLE_N_CTX=8192
```

### Trade-offs

| Context Size | Speed | Memory | Use When |
|--------------|-------|--------|----------|
| **2048** | ⚡⚡⚡ Fast | Low | Simple templates |
| **4096** | ⚡⚡ Good | Medium | Short consultations |
| **8192** | ⚡ OK | High | Standard medical (recommended) |
| **16384** | 🐌 Slow | Very High | Long consultations |
| **32768+** | 🐌🐌 Very Slow | Extreme | Multi-visit summaries |

---

## ⚙️ How to Configure

### Method 1: Using Config Manager

```bash
./aura_config.sh

# Choose option 3 (Configure LLM models)
# You'll be prompted for context size
```

### Method 2: Edit .env Directly

```bash
nano .env

# Change these lines:
N_CTX=8192              # For complex model
SIMPLE_N_CTX=2048       # For simple model
```

### Method 3: Model-Specific Presets

Create a preset for each model you use:

```bash
# .env.mistral7b
MODEL_PATH=/models/Mistral-7B-Instruct-v0.3.Q4_K_M.gguf
N_CTX=8192
CHAT_FORMAT=mistral-instruct

# .env.llama8b
MODEL_PATH=/models/Llama-3.1-8B-Instruct-Q4_K_M.gguf
N_CTX=8192
CHAT_FORMAT=llama-3

# Load preset:
cp .env.mistral7b .env
docker-compose restart llm
```

---

## 🧪 Testing Context Sizes

### Test 1: Does it fit in memory?

```bash
# Start LLM container with your context size
docker-compose restart llm

# Watch logs
docker logs -f aura-llm

# Look for:
✅ [LLM] ✅ Complex model loaded: /models/...
❌ [LLM] ❌ Out of memory error
```

### Test 2: Speed test

```bash
# Small context (fast)
N_CTX=4096
# vs
# Large context (slow)
N_CTX=32768

# Run same conversation, compare response time
```

### Test 3: Long conversation test

```bash
# Set large context
N_CTX=16384

# Test with long medical history
# Does it remember earlier parts of conversation?
```

---

## 📋 Quick Reference Table

| Model Family | Default N_CTX | Max N_CTX | Simple N_CTX |
|--------------|---------------|-----------|--------------|
| **Mistral** | 8192 | 32768 | 2048 |
| **Llama 3.1** | 8192 | 128000 | 2048 |
| **Gemma 2** | 8192 | 8192 | 2048 |
| **Qwen 2.5** | 8192 | 32768 | 2048 |
| **Phi 3.5** | 4096 | 128000 | 2048 |

---

## 💡 Pro Tips

### 1. Start Conservative

```bash
# First time with a new model:
N_CTX=4096

# Works well? Increase:
N_CTX=8192

# Still good? Go higher if needed:
N_CTX=16384
```

### 2. Match Your Use Case

```bash
# Quick symptom checks (5-10 questions):
N_CTX=4096

# Standard OLDCARTS assessment (10-20 questions):
N_CTX=8192

# Complex multi-system review (20+ questions):
N_CTX=16384

# Long-term patient monitoring (multiple sessions):
N_CTX=32768
```

### 3. Monitor Memory Usage

```bash
# Check GPU memory
nvidia-smi

# If memory is tight, reduce context:
N_CTX=4096  # Uses less memory
```

### 4. Different Sizes for Different Models

```bash
# Complex model: Large context for reasoning
MODEL_PATH=/models/Mistral-7B-Instruct-v0.3.Q4_K_M.gguf
N_CTX=8192

# Simple model: Small context for speed
SIMPLE_MODEL_PATH=/models/Llama-3.2-1B-Instruct-Q4_K_M.gguf
SIMPLE_N_CTX=2048
```

---

## 🆘 Troubleshooting

### Problem: Out of memory error

```bash
# Reduce context size
N_CTX=4096  # Try smaller

# Or reduce GPU layers
N_GPU_LAYERS=35  # Don't use all layers
```

### Problem: Very slow responses

```bash
# You might have context too large
N_CTX=32768  # This is slow!

# Try smaller:
N_CTX=8192   # Much faster
```

### Problem: Model "forgets" earlier conversation

```bash
# Context too small for conversation length
N_CTX=2048   # Too small!

# Increase:
N_CTX=8192   # Better memory
```

---

## 🔗 See Also

- `UNIFIED_CONFIG_GUIDE.md` - Complete configuration guide
- `RAG_GPU_CPU_TOGGLE.md` - RAG performance options
- `.env.example` - All configuration options

---

**Remember:** Different models, different sizes! Check the table and adjust `N_CTX` accordingly! 🎯


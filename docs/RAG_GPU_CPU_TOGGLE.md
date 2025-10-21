# RAG Container: GPU vs CPU Toggle

**Choose between GPU-accelerated FAISS (fast) or CPU FAISS (no GPU needed)**

---

## 🎯 The Choice

```
┌─────────────────────────────────────────────────────────┐
│                                                          │
│  RAG_ENABLED=true          vs       RAG_ENABLED=false  │
│                                                          │
│  GPU FAISS (Fast)                   CPU FAISS (Slow)    │
│  ✅ Faster searches                  ✅ No GPU needed    │
│  ✅ Better for production            ✅ Works anywhere   │
│  ❌ Requires GPU                      ❌ Slower          │
│  ❌ More complex setup                ❌ Limited scale   │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

---

## 📊 What's the Difference?

### Option 1: GPU RAG (RAG_ENABLED=true)

**How it works:**
```
Patient query
    ↓
LLM Container (Port 11434)
    ↓
Calls RAG Container (Port 11435) ← Separate container
    ↓
RAG uses GPU-accelerated FAISS
    ↓
Returns results FAST
```

**Characteristics:**
- ⚡ **Fast** - GPU acceleration
- 🏗️ **Scalable** - Separate service
- 🎯 **Better** - Optimized FAISS implementation
- 💾 **More memory** - GPU VRAM required

**Use when:**
- You have GPU available (Jetson, NVIDIA GPU)
- Running in production
- Need fast response times
- Have many documents in RAG

---

### Option 2: CPU RAG (RAG_ENABLED=false)

**How it works:**
```
Patient query
    ↓
LLM Container (Port 11434)
    ↓
Uses built-in CPU FAISS ← Same container
    ↓
Returns results (slower)
```

**Characteristics:**
- 🐌 **Slower** - CPU only
- 📦 **Simple** - One less container
- 💻 **Works anywhere** - No GPU needed
- 💾 **Less memory** - Uses system RAM

**Use when:**
- No GPU available
- Development/testing on laptop
- Small document collection
- Simplicity > speed

---

## ⚙️ Configuration

### In `.env` file:

```bash
# ============================================================================
# RAG CONTAINER - GPU vs CPU Toggle
# ============================================================================

# RAG Mode Selection
# - true: Use RAG container with GPU-accelerated FAISS (faster, recommended)
# - false: Use local CPU FAISS within LLM container (slower, no GPU needed)
RAG_ENABLED=true

# RAG Container URL (when RAG_ENABLED=true)
RAG_SERVICE_URL=http://localhost:11435

# RAG request timeout (seconds)
RAG_TIMEOUT=10
```

---

## 🚀 How to Switch

### Method 1: Config Manager

```bash
./aura_config.sh

# Future: Will have RAG toggle option
# For now, edit manually or use Method 2
```

### Method 2: Edit .env

```bash
nano .env

# Change this line:
RAG_ENABLED=true    # GPU RAG (fast)
# or
RAG_ENABLED=false   # CPU RAG (slow)

# Save and restart
docker-compose restart
```

### Method 3: Quick Command

```bash
# Enable GPU RAG
sed -i 's/RAG_ENABLED=false/RAG_ENABLED=true/' .env
docker-compose restart

# Disable (use CPU)
sed -i 's/RAG_ENABLED=true/RAG_ENABLED=false/' .env
docker-compose restart
```

---

## 📈 Performance Comparison

### Benchmark: Search 1000 Documents

| Mode | Search Time | Memory | Setup |
|------|-------------|--------|-------|
| **GPU FAISS** | ~50ms | 2GB GPU | Complex |
| **CPU FAISS** | ~500ms | 2GB RAM | Simple |

### For Medical Use

**Typical RAG query:** "chest pain guidelines"

```bash
# GPU RAG (RAG_ENABLED=true)
Search: 50ms
Total response: 2-3 seconds

# CPU RAG (RAG_ENABLED=false)  
Search: 500ms
Total response: 3-4 seconds
```

**Impact:** ~1 second slower with CPU RAG per query

---

## 🎯 Which Should You Use?

### Use GPU RAG (RAG_ENABLED=true) When:

✅ You have NVIDIA GPU (Jetson Orin, RTX, etc.)
✅ Running in production
✅ Need best performance
✅ Have >100 documents in RAG
✅ Multiple users/queries

**Example setups:**
- Jetson Orin NX/AGX (your setup!)
- Desktop with NVIDIA GPU
- Cloud GPU instance (AWS p3, GCP with GPU)

### Use CPU RAG (RAG_ENABLED=false) When:

✅ Development on laptop without GPU
✅ Testing on MacBook
✅ Simple demos
✅ Small document collection (<50 docs)
✅ Single user testing

**Example setups:**
- MacBook (no NVIDIA GPU)
- Linux laptop without GPU
- Cloud CPU-only instance
- Windows desktop without GPU

---

## 🔧 Setup for Each Mode

### Setup 1: GPU RAG (Recommended for Production)

```yaml
# docker-compose.yml (already configured)
services:
  llm:
    environment:
      - RAG_ENABLED=true
      - RAG_SERVICE_URL=http://localhost:11435
  
  rag:
    # RAG container with GPU
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
```

```bash
# .env
RAG_ENABLED=true
RAG_SERVICE_URL=http://localhost:11435

# Start both containers
docker-compose up -d llm rag
```

### Setup 2: CPU RAG (Simpler)

```bash
# .env
RAG_ENABLED=false

# Only need LLM container (RAG built-in)
docker-compose up -d llm

# RAG container not needed!
```

---

## 🧪 Testing

### Test 1: Verify Which Mode is Active

```bash
# Start Aura
python main.py

# Check logs
docker logs aura-llm | grep RAG

# GPU RAG (RAG_ENABLED=true):
[RAG] ✅ Using external RAG container: http://localhost:11435

# CPU RAG (RAG_ENABLED=false):
[RAG] ⚠️ Using local CPU FAISS (external RAG disabled)
```

### Test 2: Speed Test

```bash
# Ask a question that requires RAG:
"What are the symptoms of pancreatitis?"

# Time how long it takes

# GPU: ~2-3 seconds total
# CPU: ~3-4 seconds total
```

### Test 3: Verify RAG Container

```bash
# If RAG_ENABLED=true, container should be running:
docker ps | grep aura-rag

# Should see:
aura-rag   Up X minutes   0.0.0.0:11435->11435/tcp

# If RAG_ENABLED=false, container not needed
```

---

## 💡 Hybrid Approach

You can even switch modes dynamically!

```bash
# Development (CPU, simple):
RAG_ENABLED=false
docker-compose up -d llm

# Production (GPU, fast):
RAG_ENABLED=true
docker-compose up -d llm rag
```

---

## 🆘 Troubleshooting

### Problem: RAG_ENABLED=true but container won't start

**Symptoms:**
```bash
docker logs aura-rag
# ERROR: CUDA not available
```

**Solution:**
```bash
# Check GPU availability
nvidia-smi

# If no GPU, use CPU mode instead:
RAG_ENABLED=false
```

### Problem: RAG_ENABLED=false but searches are slow

**This is expected!** CPU FAISS is slower.

**Solutions:**
1. **Use GPU mode** (if you have GPU):
   ```bash
   RAG_ENABLED=true
   docker-compose up -d rag
   ```

2. **Reduce document count** (fewer docs = faster):
   ```bash
   # Remove unused documents from data/input/
   ```

3. **Accept slower speed** (CPU limitations)

### Problem: RAG container timeout

**Symptoms:**
```bash
[LLM] ❌ RAG request timeout
```

**Solution:**
```bash
# Increase timeout in .env:
RAG_TIMEOUT=30  # Was 10, now 30 seconds

# Or switch to CPU mode (built-in):
RAG_ENABLED=false
```

---

## 📋 Quick Reference

### Enable GPU RAG (Fast)

```bash
# 1. Edit .env
RAG_ENABLED=true
RAG_SERVICE_URL=http://localhost:11435

# 2. Start RAG container
docker-compose up -d rag

# 3. Restart LLM
docker-compose restart llm
```

### Enable CPU RAG (Simple)

```bash
# 1. Edit .env
RAG_ENABLED=false

# 2. Stop RAG container (not needed)
docker-compose stop rag

# 3. Restart LLM
docker-compose restart llm
```

### Check Current Mode

```bash
# View setting
grep RAG_ENABLED .env

# Check logs
docker logs aura-llm | grep -i "rag"
```

---

## 🎓 Summary

| Aspect | GPU RAG (true) | CPU RAG (false) |
|--------|----------------|-----------------|
| **Speed** | ⚡ Fast (~50ms) | 🐌 Slow (~500ms) |
| **Setup** | Complex (2 containers) | Simple (1 container) |
| **Requirements** | NVIDIA GPU | Any CPU |
| **Memory** | GPU VRAM | System RAM |
| **Best For** | Production | Development |
| **Scalability** | Excellent | Limited |

**Recommendation:**
- **GPU available?** Use `RAG_ENABLED=true` (faster!)
- **No GPU?** Use `RAG_ENABLED=false` (simpler!)

---

## 🔗 See Also

- `UNIFIED_CONFIG_GUIDE.md` - Complete configuration
- `MODEL_CONTEXT_SIZES.md` - Context size reference
- `docker-compose.yml` - Container setup

---

**Your setup (Jetson Orin):** Use `RAG_ENABLED=true` for best performance! 🚀


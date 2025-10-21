# RAG Container Toggle Guide

## Overview

The LLM Medical Container now supports **two RAG modes**:

1. **GPU Mode** (Default): Uses external RAG container with FAISS GPU acceleration
2. **CPU Mode** (Fallback): Uses local FAISS CPU within LLM container

This allows you to run the system with or without the RAG container, providing flexibility for different hardware configurations.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    RAG Mode Selection                        │
└─────────────────────────────────────────────────────────────┘
                              │
                    ┌─────────┴─────────┐
                    │                   │
        ┌───────────▼──────────┐  ┌────▼──────────────────┐
        │   GPU Mode (Fast)    │  │  CPU Mode (Portable)  │
        │                      │  │                       │
        │ External RAG Container│  │ Local FAISS in LLM   │
        │ Port: 11435          │  │ Container             │
        │ FAISS GPU            │  │ FAISS CPU             │
        │ Faster embeddings    │  │ No external service   │
        └──────────────────────┘  └───────────────────────┘
```

## Configuration

### Method 1: Environment Variable

```bash
# Enable GPU mode (use RAG container)
export RAG_ENABLED=true

# Enable CPU mode (local FAISS)
export RAG_ENABLED=false
```

### Method 2: Configuration File

Create a `.env` file in the `llm-medical-container` directory:

```env
# Use GPU mode (RAG container)
RAG_ENABLED=true
RAG_SERVICE_URL=http://localhost:11435
RAG_TIMEOUT=10
```

Or for CPU mode:

```env
# Use CPU mode (local FAISS)
RAG_ENABLED=false
```

### Method 3: Docker Compose

```yaml
services:
  llm-container:
    environment:
      - RAG_ENABLED=true  # or false for CPU mode
      - RAG_SERVICE_URL=http://rag-container:11435
```

## Usage Examples

### Running with GPU Mode (RAG Container)

```bash
# Terminal 1: Start RAG container
cd rag-container
docker-compose up

# Terminal 2: Start LLM container with GPU mode
cd llm-medical-container
export RAG_ENABLED=true
python container_rest.py
```

### Running with CPU Mode (No RAG Container)

```bash
# Just start LLM container with CPU mode
cd llm-medical-container
export RAG_ENABLED=false
python container_rest.py
```

## Performance Comparison

| Feature | GPU Mode | CPU Mode |
|---------|----------|----------|
| **Search Speed** | Faster (GPU acceleration) | Moderate (CPU only) |
| **API Overhead** | ⚠️ HTTP calls (slower) | ✅ In-process (faster) |
| **Total Latency** | Network + GPU processing | CPU processing only |
| **Memory** | More (2 containers) | Less (1 container) |
| **Setup** | Requires RAG container | Self-contained |
| **Hardware** | Needs GPU | Works anywhere |
| **Embedding** | all-distilroberta-v1 (GPU) | all-distilroberta-v1 (CPU) |
| **FAISS** | GPU index | CPU index (FlatL2) |
| **Communication** | HTTP REST API | Direct function calls |

### Latency Breakdown

**GPU Mode (External RAG Container):**
```
Total Latency = HTTP Request + Network Transfer + GPU Processing + HTTP Response
              = ~10-50ms    + ~5-10ms         + ~5-20ms        + ~5-10ms
              = ~25-90ms total
```

**CPU Mode (In-Process):**
```
Total Latency = CPU Processing
              = ~20-100ms total
```

> **Note**: CPU mode can actually be **faster** for small queries because it avoids HTTP/network overhead! GPU mode shines with large batch operations.

## Automatic Fallback

The system automatically falls back to CPU mode if:
- `RAG_ENABLED=true` but RAG service is unavailable
- RAG service health check fails
- RAG service times out

```python
[RAG Client] ❌ RAG service unavailable: Connection refused
[RAG Client] 🔄 Falling back to CPU mode
[RAG Client] ✅ CPU RAG system initialized
```

## API Compatibility

Both modes provide the same API interface:

```python
from rag_client import get_rag_client

rag = get_rag_client()

# Search medical information
results = rag.search("chest pain symptoms", k=5)

# Generate embeddings
embeddings = rag.embed(["text1", "text2"])

# Get guideline
guideline = rag.get_guideline("chest_pain_assessment")

# Check current mode
print(f"RAG Mode: {rag.get_mode()}")
# Output: "GPU (External RAG Container)" or "CPU (Local FAISS)"
```

## Troubleshooting

### RAG Container Not Available

```
[RAG Client] ❌ RAG service unavailable: Connection refused
```

**Solution**: Either start the RAG container or set `RAG_ENABLED=false`

### CPU Mode Dependencies Missing

```
[RAG Client] ❌ Failed to import CPU RAG dependencies
```

**Solution**: Install dependencies:
```bash
pip install sentence-transformers faiss-cpu
```

### Empty CPU Index

```
[RAG Client] ⚠️ No existing CPU index found
```

**Solution**: The CPU index is built from `data/embeddings/`. Either:
1. Copy embeddings from RAG container
2. Run the ingest script to build embeddings

## Migration Guide

### From RAG Container to CPU Mode

1. Export embeddings from RAG container:
   ```bash
   docker cp rag-container:/data/embeddings ./data/
   ```

2. Set CPU mode:
   ```bash
   export RAG_ENABLED=false
   ```

3. Restart LLM container

### From CPU Mode to RAG Container

1. Start RAG container:
   ```bash
   cd rag-container
   docker-compose up
   ```

2. Set GPU mode:
   ```bash
   export RAG_ENABLED=true
   ```

3. Restart LLM container

## Best Practices

1. **Production**: Use GPU mode for best performance
2. **Development**: Use CPU mode for simpler setup
3. **Testing**: Use CPU mode to avoid RAG container dependency
4. **Demo**: Use GPU mode for fastest response times

## Future Enhancements

- [ ] Dynamic mode switching without restart
- [ ] Hybrid mode (GPU for search, CPU for embeddings)
- [ ] Index synchronization between GPU and CPU modes
- [ ] Performance metrics and mode recommendation


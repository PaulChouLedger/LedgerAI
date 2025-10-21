# RAG Modes: Detailed Comparison

## Architecture Comparison

### GPU Mode (External RAG Container)
```
┌─────────────────────────────────────────────────────────────┐
│                    LLM Medical Container                     │
│                                                              │
│  ┌────────────────┐                                         │
│  │  User Request  │                                         │
│  └───────┬────────┘                                         │
│          │                                                   │
│          ▼                                                   │
│  ┌──────────────────┐                                       │
│  │  RAG Client      │                                       │
│  │  (rag_client.py) │                                       │
│  └───────┬──────────┘                                       │
│          │                                                   │
│          │  HTTP POST /rag/search                           │
│          │  {"query": "...", "k": 5}                        │
│          │                                                   │
│          ▼                                                   │
└──────────┼───────────────────────────────────────────────────┘
           │
           │  Network/HTTP
           │  ~10-50ms overhead
           │
           ▼
┌──────────┼───────────────────────────────────────────────────┐
│          ▼                                                    │
│  ┌──────────────────┐         RAG Container                  │
│  │  Flask Endpoint  │         (Port 11435)                   │
│  └───────┬──────────┘                                        │
│          │                                                    │
│          ▼                                                    │
│  ┌──────────────────┐                                        │
│  │  FAISS GPU Index │  ← GPU Accelerated                    │
│  └───────┬──────────┘                                        │
│          │                                                    │
│          ▼                                                    │
│  ┌────────────────────────┐                                  │
│  │  SentenceTransformer   │  ← GPU Embeddings               │
│  │  (all-distilroberta-v1)│                                 │
│  └───────┬────────────────┘                                  │
│          │                                                    │
│          │  HTTP Response                                    │
│          │  {"results": [...]}                               │
│          │                                                    │
└──────────┼────────────────────────────────────────────────────┘
           │
           ▼
┌──────────┼───────────────────────────────────────────────────┐
│          ▼                                                    │
│  ┌──────────────────┐                                        │
│  │  Response to LLM │                                        │
│  └──────────────────┘                                        │
│                                                               │
└───────────────────────────────────────────────────────────────┘
```

**Total Path**: User → LLM Container → HTTP → RAG Container → GPU Processing → HTTP Response → LLM Container → User

---

### CPU Mode (In-Process)
```
┌─────────────────────────────────────────────────────────────┐
│                    LLM Medical Container                     │
│                                                              │
│  ┌────────────────┐                                         │
│  │  User Request  │                                         │
│  └───────┬────────┘                                         │
│          │                                                   │
│          ▼                                                   │
│  ┌──────────────────┐                                       │
│  │  RAG Client      │                                       │
│  │  (rag_client.py) │                                       │
│  └───────┬──────────┘                                       │
│          │                                                   │
│          │  Direct function call (in-process)               │
│          │  No HTTP overhead!                               │
│          │                                                   │
│          ▼                                                   │
│  ┌──────────────────┐                                       │
│  │  FAISS CPU Index │  ← CPU Processing                    │
│  └───────┬──────────┘                                       │
│          │                                                   │
│          ▼                                                   │
│  ┌────────────────────────┐                                 │
│  │  SentenceTransformer   │  ← CPU Embeddings              │
│  │  (all-distilroberta-v1)│                                │
│  └───────┬────────────────┘                                 │
│          │                                                   │
│          ▼                                                   │
│  ┌──────────────────┐                                       │
│  │  Response to LLM │                                       │
│  └──────────────────┘                                       │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

**Total Path**: User → LLM Container → Local Processing → User

---

## Key Differences

### 1. **Communication Method**

**GPU Mode:**
- HTTP REST API calls
- JSON serialization/deserialization
- Network stack overhead
- Port 11435

**CPU Mode:**
- Direct Python function calls
- In-memory data passing
- No serialization overhead
- No network

### 2. **Code Flow**

**GPU Mode:**
```python
# Inside LLM container
def search(query):
    # HTTP call to external container
    response = requests.post(
        "http://localhost:11435/rag/search",
        json={"query": query, "k": 5}
    )
    return response.json()['results']
```

**CPU Mode:**
```python
# Inside LLM container
def search(query):
    # Direct in-process call
    embeddings = self._embedding_model.encode([query])
    distances, indices = self._cpu_index.search(embeddings, 5)
    return self._build_results(indices, distances)
```

### 3. **Dependencies**

**GPU Mode:**
- Requires RAG container running
- Requires network connectivity
- Requires HTTP server (Flask)
- Depends on external service health

**CPU Mode:**
- Self-contained within LLM container
- No external dependencies
- No HTTP server needed
- Independent operation

### 4. **Error Handling**

**GPU Mode:**
```python
# Can fail due to:
- RAG container not running
- Network issues
- Port conflicts
- Timeout errors
- HTTP 500 errors
```

**CPU Mode:**
```python
# Can only fail due to:
- Python exceptions
- Memory issues
- Index corruption
```

## When to Use Each Mode

### Use GPU Mode When:
- ✅ You have a GPU available
- ✅ You need maximum search performance
- ✅ You're running distributed microservices
- ✅ You want to scale RAG independently
- ✅ Multiple containers need to share RAG

### Use CPU Mode When:
- ✅ You don't have a GPU
- ✅ You want simplest possible setup
- ✅ You're developing/testing locally
- ✅ You want fastest response for small queries
- ✅ You want single-container deployment
- ✅ You want to avoid network overhead

## Performance Benchmarks (Estimated)

### Small Query (Single sentence)
- **GPU Mode**: ~50-90ms (Network: 30ms, GPU: 20-60ms)
- **CPU Mode**: ~30-70ms (CPU only: 30-70ms)
- **Winner**: CPU Mode (avoids network overhead)

### Medium Query (Paragraph)
- **GPU Mode**: ~60-100ms (Network: 30ms, GPU: 30-70ms)
- **CPU Mode**: ~80-150ms (CPU only: 80-150ms)
- **Winner**: GPU Mode (starts to benefit from GPU)

### Large Query (Multiple paragraphs)
- **GPU Mode**: ~70-120ms (Network: 30ms, GPU: 40-90ms)
- **CPU Mode**: ~150-300ms (CPU only: 150-300ms)
- **Winner**: GPU Mode (significant GPU advantage)

### Batch Queries (10+ queries)
- **GPU Mode**: ~100-200ms (Network: 30ms, GPU parallel: 70-170ms)
- **CPU Mode**: ~500-1500ms (CPU sequential: 500-1500ms)
- **Winner**: GPU Mode (massive parallelization benefit)

## Recommendations

### Development & Testing
```bash
export RAG_ENABLED=false  # Use CPU mode
```
**Why**: Simpler setup, no need to manage multiple containers

### Production (High Volume)
```bash
export RAG_ENABLED=true   # Use GPU mode
```
**Why**: Better performance for sustained high query volume

### Production (Low Volume / No GPU)
```bash
export RAG_ENABLED=false  # Use CPU mode
```
**Why**: CPU mode is actually faster for sporadic single queries

### Deployment Scenarios

| Scenario | Recommended Mode | Reason |
|----------|------------------|--------|
| Local Development | CPU | Simplicity |
| Demo/Presentation | CPU | No dependencies |
| High-Traffic API | GPU | Sustained performance |
| Edge Device (Jetson) | Mixed | GPU if available, CPU fallback |
| Cloud (with GPU) | GPU | Better hardware utilization |
| Cloud (CPU-only) | CPU | No alternative |
| Kubernetes | GPU | Scalable microservices |


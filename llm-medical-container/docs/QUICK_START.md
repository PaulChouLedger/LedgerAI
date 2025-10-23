# Quick Start: RAG Toggle System

## One-Line Toggle

```bash
# Use GPU (RAG Container)
export RAG_ENABLED=true && python container_rest.py

# Use CPU (Local FAISS)
export RAG_ENABLED=false && python container_rest.py
```

## That's It!

Your entire LedgerAI system now automatically adapts to use either:
- **GPU mode**: External RAG container with FAISS GPU (fast for large batches)
- **CPU mode**: Local FAISS CPU within LLM container (fast for single queries, no dependencies)

## Quick Test

```python
from rag_client import get_rag_client

rag = get_rag_client()
print(f"Mode: {rag.get_mode()}")
results = rag.search("chest pain symptoms", k=5)
print(f"Found {len(results)} results")
```

## See Also

- `RAG_TOGGLE_GUIDE.md` - Full usage guide
- `RAG_MODES_COMPARISON.md` - Performance comparison
- `INTEGRATION_SUMMARY.md` - What was changed


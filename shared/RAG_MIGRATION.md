# RAG Module Migration to Shared

## Summary
Moved the RAG module from container-specific directories to `shared/rag/` to eliminate duplication, following the same pattern as `llm_base.py`.

## Changes Made

### 1. Moved RAG to Shared
- ✅ Copied `llm-container/rag/` → `shared/rag/` (using generic container's version as it's more complete)
- ✅ Updated `shared/rag/__init__.py` to reflect shared status
- ✅ Created `shared/rag/README.md` with documentation

### 2. Updated Container Imports
- ✅ Generic container (`llm-container/container_rest.py`):
  - Already has `sys.path.insert(0, '/shared')` for base class
  - Imports `from rag import get_rag_client` now resolve to `/shared/rag/`
  
- ✅ Medical container (`llm-medical-container/container_rest.py`):
  - Already has `sys.path.insert(0, '/shared')` for base class
  - Will use shared RAG if it imports RAG in the future

### 3. Updated Dockerfiles
- ✅ `llm-container/Dockerfile`: Removed `COPY rag/ /app/rag/` (now uses `/shared/rag/`)
- ✅ `llm-medical-container/Dockerfile`: Removed `COPY rag/ /app/rag/` (now uses `/shared/rag/`)
- ✅ Both containers mount `/shared` via `docker-compose.yml`

### 4. Deleted Sync Script
- ✅ Deleted `setup/scripts/sync_llm_containers.py` (no longer needed)
- ✅ Updated git hooks (`pre-commit`, `post-merge`) to remove sync script references
- ✅ Updated `install_aura_bootable.sh` comments

## Architecture

```
shared/
├── llm_base.py          # Shared LLM base class
├── rag/                 # Shared RAG module (NEW)
│   ├── __init__.py
│   ├── rag_client.py
│   ├── cpu_faiss_auto_ingest.py
│   ├── convert_for_cpu_faiss.py
│   ├── optimize_rag_architecture.py
│   └── README.md
└── medical_terms.json   # Shared medical terms
```

Both containers import from shared:
```python
import sys
sys.path.insert(0, '/shared')
from llm_base import BaseLLMContainer
from rag import get_rag_client
```

## Benefits

1. **No Duplication**: RAG code exists in one place (`shared/rag/`)
2. **Consistency**: Both containers use identical RAG implementation
3. **Easier Maintenance**: Fixes and improvements in one location
4. **No Sync Needed**: Eliminated need for `sync_llm_containers.py` script
5. **Cleaner Architecture**: Follows same pattern as `llm_base.py`

## Migration Notes

- The generic container's RAG was used as the source (it had more features)
- Both containers' local `rag/` directories can be removed (they're no longer used)
- Docker volumes mount `/shared` so changes are immediately available
- No container rebuild needed for RAG changes (just restart)

## Testing

To verify the migration:
1. Both containers should import RAG from `/shared/rag/`
2. RAG functionality should work identically in both containers
3. No import errors when containers start


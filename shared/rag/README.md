# RAG Module - Shared

This directory contains the shared RAG (Retrieval-Augmented Generation) module used by both generic and medical LLM containers.

## 📁 Directory Structure

```
rag/
├── __init__.py                    # RAG module initialization
├── rag_client.py                  # Main RAG client (CPU/GPU FAISS)
├── cpu_faiss_auto_ingest.py      # Auto-ingestion system
├── convert_for_cpu_faiss.py      # GPU→CPU format conversion
├── optimize_rag_architecture.py  # Architecture optimization
└── README.md                     # This file
```

## 🔧 Components

### 1. RAG Client (`rag_client.py`)
- **Purpose**: Unified RAG client supporting both CPU and GPU FAISS
- **Features**: 
  - Automatic CPU/GPU mode detection
  - Embedding generation and similarity search
  - Auto-ingestion integration
  - Fuzzy matching for transcription errors
- **Usage**: `from rag import get_rag_client`

### 2. Auto-Ingestion (`cpu_faiss_auto_ingest.py`)
- **Purpose**: Monitor `data/input/` for new guideline files
- **Features**:
  - File system watching with `watchdog`
  - Automatic embedding generation
  - Dynamic index updates
  - State persistence

### 3. Format Conversion (`convert_for_cpu_faiss.py`)
- **Purpose**: Convert GPU FAISS format to CPU FAISS format
- **Features**:
  - Loads GPU format files (`index.faiss`, `vectors.npy`, etc.)
  - Converts to CPU format (`faiss_index.bin`, `metadata.pkl`)
  - Verification and validation

### 4. Architecture Optimization (`optimize_rag_architecture.py`)
- **Purpose**: Build optimized vector database from text files
- **Features**:
  - Text chunking and embedding generation
  - FAISS index creation
  - Metadata generation
  - Performance optimization

## 🚀 Usage

### Basic RAG Operations
```python
from rag import get_rag_client

# Get RAG client (auto-detects CPU/GPU mode)
rag_client = get_rag_client()

# Search for relevant chunks
results = rag_client.search(
    query="chest pain",
    k=10,
    threshold=0.2
)
```

### Auto-Ingestion
```python
from rag.cpu_faiss_auto_ingest import CPUFAISSAutoIngest

# Initialize auto-ingestion
auto_ingest = CPUFAISSAutoIngest()

# Start watching for new files
auto_ingest.start_watching()
```

## 📦 Location

This module is located in `/shared/rag/` and is mounted into both LLM containers via Docker volumes:
- `llm-generic` container
- `llm-medical` container

Both containers import from `/shared` using:
```python
import sys
sys.path.insert(0, '/shared')
from rag import get_rag_client
```

## 🔄 Migration from Container-Specific RAG

Previously, RAG was duplicated in:
- `llm-container/rag/`
- `llm-medical-container/rag/`

Now it's unified in `shared/rag/` to eliminate duplication and ensure consistency.

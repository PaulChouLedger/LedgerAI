# RAG Module

This directory contains all RAG (Retrieval-Augmented Generation) related components for the LLM Medical Container.

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

# Start watching for file changes
auto_ingest.start_watching()

# Manual scan
result = auto_ingest.scan_and_process()
```

## 📊 Data Flow

1. **Source**: JSON guidelines in `medical/guidelines/`
2. **Text Conversion**: Converted to `GUIDELINE_*.txt` files in `data/input/`
3. **Vectorization**: Processed by `optimize_rag_architecture.py`
4. **Storage**: Saved in `data/embeddings/` (both GPU and CPU formats)
5. **Retrieval**: Accessed by `rag_client.py` for similarity search

## 🔄 Auto-Ingestion Flow

1. **File Monitoring**: Watchdog monitors `data/input/` for changes
2. **Change Detection**: New/modified `GUIDELINE_*.txt` files detected
3. **Processing**: Files chunked and embedded using SentenceTransformer
4. **Index Update**: FAISS index updated with new embeddings
5. **State Persistence**: Processing state saved for consistency

## 🎯 Integration

The RAG module integrates with:
- **Adaptive Diagnostic Engine**: Uses RAG for guideline retrieval
- **Clinician Mode**: Provides medical knowledge via RAG
- **Docker Containers**: Shared data directory via volume mounts

## 📈 Performance

- **CPU FAISS**: Local processing, no network overhead
- **GPU FAISS**: External container, GPU acceleration
- **Auto-Ingestion**: Real-time updates without restarts
- **Shared Data**: Single source of truth for both formats

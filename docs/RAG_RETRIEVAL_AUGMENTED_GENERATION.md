# RAG (Retrieval Augmented Generation) Component

## Overview

The RAG component provides semantic search over document embeddings using FAISS (Facebook AI Similarity Search). It supports both **GPU-accelerated** (external container) and **CPU-based** (embedded in LLM containers) implementations.

## Architecture

### GPU Mode (External Container)
- **Location**: `rag-container/rag.py`
- **Port**: `11435`
- **Acceleration**: CUDA via `faiss_lite` C++ wrapper
- **Embedding Model**: `all-distilroberta-v1` (SentenceTransformers)

### CPU Mode (Embedded)
- **Location**: `llm-container/rag/` and `llm-medical-container/rag/`
- **Implementation**: CPU FAISS within LLM containers
- **Auto-ingestion**: Monitors `data/input/` for new files
- **Real-time Indexing**: Chunks processed as files added

## Core Components

### 1. FAISS Index

**Structure**:
- **Index File**: `data/embeddings/index.faiss`
- **Vectors File**: `data/embeddings/vectors.npy`
- **Chunks File**: `data/embeddings/doc_chunks.npy`
- **Metadata File**: `data/embeddings/chunk_metadata.json`

**Index Properties**:
- **Metric**: Inner Product (cosine similarity)
- **Dimension**: 768 (all-distilroberta-v1 embedding size)
- **Type**: FAISS IndexFlatIP (exact search)

### 2. GPU Acceleration (faiss_lite)

**CUDA Functions**:
- `cudaKNN`: K-nearest neighbor search on GPU
- `cudaL2Norm`: L2 norm computation
- `cudaAllocMapped`: CUDA memory allocation

**Data Preparation**:
1. Loads raw vectors from `vectors.npy`
2. Allocates CUDA memory via `cudaAllocMapped`
3. Pre-computes L2 norms (if using L2 metric)
4. Pre-allocates query buffers for reuse
5. Warms up GPU with test search

**Memory Management**:
- Pre-allocated query buffer (reused across searches)
- Pre-allocated result buffers (distances, indices)
- Avoids repeated allocations that corrupt memory

### 3. Embedding Generation

**Model**: SentenceTransformer `all-distilroberta-v1`

**Process**:
1. Loads model on GPU (CUDA required)
2. Encodes query text to 768-dim vector
3. Normalizes for Inner Product metric (L2 normalization)
4. Searches FAISS index for similar vectors

**GPU Loading**:
- Handles PyTorch meta tensor bug
- Loads on CPU first, then moves to CUDA if needed
- Falls back gracefully on errors

## Search Logic

### 1. Semantic Search

**Function**: `search(query, k=3, disable_keyword_filter=False)`

**Basic Flow**:
1. **Query Encoding**: Converts query to embedding vector
2. **Normalization**: L2-normalizes for cosine similarity
3. **FAISS Search**: Finds k nearest neighbors
4. **Result Formatting**: Returns chunks with scores

**Scoring**:
- **Distance**: FAISS inner product distance
- **Similarity**: `1.0 / (1.0 + distance)` (converted to 0-1 scale)
- **Threshold**: Only results above `relevance_threshold` (0.3) returned

### 2. Hybrid Keyword + Semantic Search

**Intelligent Filtering**:
1. **Key Term Extraction**:
   - Extracts person names (capitalized words)
   - Extracts medical terms (multi-word phrases)
   - Filters stop words
   - Prioritizes longer, more specific terms

2. **Keyword Pre-filtering**:
   - Scans all chunks for key terms
   - Fast O(n) string matching
   - Fuzzy matching for name variations
   - Returns indices of matching chunks

3. **Semantic Search on Filtered Subset**:
   - Only searches pre-filtered chunks
   - Much faster than full index search
   - Maintains semantic relevance

### 3. Fuzzy Name Matching

**Function**: `_fuzzy_name_search(person_name, chunk, threshold=0.65)`

**Algorithms**:
1. **Phonetic Matching** (Metaphone):
   - Checks if words sound the same
   - Handles "Rafael"/"Raphael", "Smith"/"Smyth"
   - Consonant-based encoding

2. **Character Similarity**:
   - SequenceMatcher for edit distance
   - Length penalty for mismatches
   - Minimum character overlap (70%)

3. **Proximity Check**:
   - Ensures name words appear together (<50 chars apart)
   - Prevents false matches from scattered words
   - Validates complete name presence

**Name Extraction**:
- Detects patterns: "who is X", "tell me about Y", "where does X work"
- Handles multi-word names: "Bob Carella", "David Lara"
- Case-insensitive matching

### 4. Medical Term Extraction

**Function**: `_extract_medical_term(query)`

**Patterns**:
- "What is [term]?"
- "Tell me about [term]"
- "Explain [term]"
- "Symptoms of [term]"
- "Treatment for [term]"

**Filtering**:
- Minimum length: 4 characters
- Excludes common words: "that", "this", "there"
- Prioritizes technical terms

## Data Ingestion

### GPU Mode Ingestion

**Endpoint**: `POST /rag/ingest`

**Process**:
1. **File Scanning**: Monitors `data/input/` directory
2. **PDF Extraction**: Extracts text from PDFs
3. **File Copying**: Copies TXT files to `data/parsed/`
4. **Chunking**: Splits documents into chunks (size: configurable)
5. **Embedding**: Generates embeddings for each chunk
6. **Index Update**: Adds vectors to FAISS index
7. **State Tracking**: Records processed files

### CPU Mode Auto-Ingestion

**Location**: `llm-container/rag/` and `llm-medical-container/rag/`

**Features**:
- **Auto-scan**: Periodically scans `data/input/`
- **Incremental**: Only processes new files
- **State Persistence**: Tracks processed files
- **Background Thread**: Non-blocking ingestion

**Endpoints**:
- `POST /cpu-faiss/ingest`: Manual trigger
- `GET /cpu-faiss/status`: Status check

**Chunk Storage**:
- In-memory: Python list of chunks
- Metadata: File paths, chunk indices
- Embeddings: CPU FAISS index

### Embedding Rebuild

**Process**:
1. **Scan Parsed Files**: Reads all files from `data/parsed/`
2. **Chunk Generation**: Splits documents into overlapping chunks
3. **Embedding Generation**: Creates embeddings via SentenceTransformer
4. **Index Building**: Builds new FAISS index
5. **Save Artifacts**: Saves index, vectors, chunks, metadata

**Script**: `setup/scripts/rebuild_embeddings_host.py`

## Guideline-Specific Features

### Medical Guideline Retrieval

**Function**: `get_all_chunks_from_guideline(guideline_name)`

**Purpose**: Retrieves ALL chunks from a specific medical guideline

**Process**:
1. Uses chunk metadata to find guideline chunks
2. Returns all chunks (not just top-k)
3. Perfect for medical assessment (needs all diagnostic questions)

**Metadata Fields**:
- `guideline_name`: Name of guideline
- `is_medical_guideline`: Boolean flag
- `file_path`: Source file path

## Search Strategies

### 1. Pure Semantic Search

**When Used**:
- No keyword filter matches
- `disable_keyword_filter=True` flag
- Medical guideline queries

**Process**:
- Direct FAISS search on full index
- Returns top-k results by similarity

### 2. Keyword Pre-filtering

**When Used**:
- Key terms detected in query
- Name queries (e.g., "who is Bob Carella?")
- Medical term queries (e.g., "what is diabetes?")

**Process**:
1. Extract key terms
2. Filter chunks containing terms
3. Semantic search on filtered subset
4. Returns keyword-matched chunks

**Benefits**:
- Faster search (smaller subset)
- More relevant results
- Handles name variations

### 3. Smart Content Matching

**Function**: `quick_content_match(query)`

**Purpose**: Decides if RAG should be used

**Process**:
1. Generates query embedding
2. Searches with k=1 (fast check)
3. If similarity > threshold → use RAG
4. Otherwise → skip RAG (faster response)

**Benefits**:
- Avoids unnecessary RAG calls
- Faster for conversational queries
- Only uses RAG when relevant

## API Endpoints

### GPU Mode (RAG Container)

**`POST /rag/search`**:
- Semantic search over documents
- Returns top-k results with scores

**`POST /rag/augment`**:
- Augments prompt with RAG context
- Formats context for LLM prompt

**`POST /rag/ingest`**:
- Triggers file ingestion
- Processes new files from `data/input/`

**`POST /rag/reload`**:
- Reloads index with new embeddings
- Updates chunks and metadata

**`GET /rag/stats`**:
- Returns index statistics
- Chunk count, index size

**`GET /rag/health`**:
- Health check
- GPU availability status

### CPU Mode (LLM Containers)

**`POST /cpu-faiss/ingest`**:
- Triggers CPU FAISS ingestion
- Processes files from `data/input/`

**`GET /cpu-faiss/status`**:
- Returns ingestion status
- Chunk count, processed files

## Configuration

### RAG Mode Toggle

**Location**: `app_settings.json`

**Options**:
- `"CPU"`: Embedded CPU FAISS (default)
- `"GPU"`: External GPU container
- `"OFF"`: Disabled

**Resolution**:
```python
def _resolve_rag_mode():
    # 1. Check app_settings.json
    # 2. Fallback to "CPU"
```

### Relevance Threshold

**Default**: `0.3`

**Purpose**: Filters out low-relevance results

**Adjustment**:
- Lower = more results (more noise)
- Higher = fewer results (higher quality)

## Performance Optimization

### GPU Mode

**Optimizations**:
1. Pre-allocated CUDA buffers
2. Vector norms pre-computed
3. GPU warmup on startup
4. Batch query processing (future)

### CPU Mode

**Optimizations**:
1. In-memory chunk storage
2. Incremental ingestion
3. Background processing
4. State persistence

## Code Locations

- **GPU Implementation**: `rag-container/rag.py`
- **CPU Implementation**: `llm-container/rag/` and `llm-medical-container/rag/`
- **Embedding Rebuild**: `setup/scripts/rebuild_embeddings_host.py`
- **Ingestion**: `rag-container/ingest.py`

## Dependencies

- `faiss`: FAISS library (CPU or GPU)
- `faiss_lite`: CUDA wrapper for GPU acceleration
- `sentence-transformers`: Embedding generation
- `torch`: PyTorch for CUDA
- `numpy`: Array operations


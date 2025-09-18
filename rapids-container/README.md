# Aura RAPIDS RAG Service

Modern, GPU-accelerated RAG (Retrieval-Augmented Generation) service using cuDF, cuVS, and LlamaIndex.

## Architecture

```
Raw Documents → cuDF Processing → GPU Embeddings → cuVS Index → LlamaIndex → LLM Response
```

## Components

- **Document Processor**: cuDF-based chunking and cleaning
- **Embedding Engine**: GPU-accelerated HuggingFace transformers
- **Vector Store**: cuVS for fast similarity search with metadata
- **LlamaIndex Adapter**: Custom VectorStore adapter for cuVS
- **RAG Engine**: Orchestrates the entire pipeline

## API Endpoints

### `POST /rag`
Main RAG endpoint - takes user query, returns LLM response with context.

**Request:**
```json
{
  "query": "What is AuraVision?",
  "top_k": 3
}
```

**Response:**
```json
{
  "query": "What is AuraVision?",
  "response": "AuraVision is...",
  "top_k": 3
}
```

### `GET /health`
Health check with system statistics.

### `POST /rebuild`
Rebuild the vector index from documents.

### `GET /stats`
Get RAG system statistics.

## Usage

1. Place documents in `/shared/input/` (supports .txt, .pdf)
2. Start the service - it will automatically build the index
3. Query via `/rag` endpoint
4. Rebuild index when documents change via `/rebuild`

## Dependencies

- cuDF: GPU-accelerated DataFrame processing
- cuVS: GPU-accelerated vector similarity search
- LlamaIndex: RAG framework
- HuggingFace Transformers: GPU embeddings
- Flask: REST API

Simple, fast, GPU-native RAG pipeline.

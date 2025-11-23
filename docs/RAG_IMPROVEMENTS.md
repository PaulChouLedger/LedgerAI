# RAG System Improvements

## Overview
This document describes the improvements made to the RAG (Retrieval Augmented Generation) system to enhance vectorization, retrieval quality, and natural context injection into the LLM.

## Key Improvements

### 1. Semantic-Aware Chunking
**Location**: `llm-container/rag/cpu_faiss_auto_ingest.py`

**Changes**:
- Replaced simple word-based chunking with semantic-aware chunking
- Respects paragraph boundaries (double newlines)
- Maintains sentence boundaries within chunks
- Preserves context through intelligent overlap
- Filters out very short chunks (artifacts)

**Benefits**:
- Chunks are more semantically coherent
- Better context preservation across chunk boundaries
- Improved retrieval quality due to better chunk structure

### 2. Query Expansion
**Location**: `llm-container/rag/rag_client.py`

**Changes**:
- Added generic query expansion (no domain-specific hardcoding)
- Handles basic linguistic variations (pluralization/singularization)
- Works generically for any application domain

**Benefits**:
- Improved recall for related terms
- Generic approach works across all domains
- No hardcoded domain-specific terms

### 3. Re-ranking of Results
**Location**: `llm-container/rag/rag_client.py`

**Changes**:
- Added `_rerank_results()` method that combines semantic and keyword scores
- 70% weight on semantic similarity, 30% on keyword matching
- Retrieves more candidates (k*2) then re-ranks to top k

**Benefits**:
- Better relevance ordering of retrieved chunks
- Hybrid scoring improves precision
- More relevant chunks appear first in context

### 4. Improved Context Formatting
**Location**: `llm-container/container_rest.py`, `llm-medical-container/container_rest.py`

**Changes**:
- Results sorted by relevance score before injection
- Minimal metadata formatting (source only if available)
- Clear separators between chunks (`---`)
- Sentence-boundary aware truncation (preserves meaning)
- Increased chunk size limit (1200 chars) for better context

**Benefits**:
- LLM receives context in relevance order
- Less noise from metadata
- Better context preservation through sentence-aware truncation

### 5. Dynamic Prompt Engineering
**Location**: `llm-container/container_rest.py`, `llm-medical-container/container_rest.py`

**Changes**:
- Removed all hardcoded domain-specific instructions
- Generic, dynamic prompts that work for any application
- Minimal, focused guidelines
- Context-driven prompt construction

**Benefits**:
- Works generically across all domains
- No hardcoded assumptions about content type
- LLM can adapt to any context naturally

## Technical Details

### Chunking Strategy
- **Primary split**: Paragraph boundaries (`\n\n+`)
- **Secondary split**: Sentence boundaries (`.`, `!`, `?`)
- **Overlap**: Configurable (default 50 words)
- **Min chunk size**: 10 words (filters artifacts)

### Re-ranking Algorithm
```python
combined_score = 0.7 * semantic_score + 0.3 * keyword_score
```
- Semantic score: From embedding similarity
- Keyword score: Term overlap ratio
- Results sorted by combined score

### Context Injection Flow
1. Query expansion (generic linguistic variations)
2. Semantic search (retrieve k*2 candidates)
3. Re-ranking (combine semantic + keyword scores)
4. Filter by threshold
5. Sort by relevance
6. Format with minimal metadata
7. Inject into LLM with dynamic prompts

## Configuration

All improvements use existing configuration:
- `RAG_SEARCH_K`: Number of results (default: 5)
- `RAG_SEARCH_THRESHOLD`: Similarity threshold (default: 0.30)
- `MAX_CHARS_PER_RESULT`: Chunk size limit (default: 1200)

## Backward Compatibility

All changes are backward compatible:
- Existing embeddings continue to work
- No changes to API interfaces
- Gradual improvement as new documents are ingested

## Future Enhancements

Potential improvements (not yet implemented):
- Hybrid search (semantic + BM25 keyword search)
- Cross-encoder re-ranking for better precision
- Query reformulation using LLM
- Adaptive chunk sizing based on document type
- Metadata-aware retrieval (filter by document type, date, etc.)


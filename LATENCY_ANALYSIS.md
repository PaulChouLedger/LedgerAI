# Latency Analysis & Optimization Opportunities

## Current Latency Bottlenecks

### 1. **Sequential API Calls** (Major Impact)
**Current Flow:**
```
Document RAG Search → Memory Quick-Match → Memory RAG Search → LLM Scoring → Response
```

**Time Breakdown:**
- Document RAG search: ~100-500ms (FAISS CPU search)
- Memory quick-match: ~50-200ms (HTTP request, 2s timeout)
- Memory RAG search: ~200-1000ms (HTTP request, 15s timeout)
- LLM scoring: ~500-2000ms (blocking LLM call)
- **Total sequential wait: ~850-3700ms**

**Optimization:** Parallelize document RAG and memory RAG searches using `concurrent.futures` or `asyncio`.

### 2. **Blocking LLM Scoring** (Major Impact)
**Current:** LLM scoring for memory RAG happens synchronously before response generation.

**Time:** ~500-2000ms (depends on model speed)

**Optimization Options:**
- Skip LLM scoring for follow-up queries (use semantic scores only)
- Make LLM scoring optional/configurable
- Use faster model for scoring (if available)
- Cache scoring results for similar queries

### 3. **High Timeout Values** (Moderate Impact)
**Current:**
- Memory quick-match: 2s timeout
- Memory RAG search: 15s timeout (very high!)

**Issue:** Even if requests complete quickly, high timeouts can cause delays if network is slow.

**Optimization:** Reduce timeouts to more realistic values (500ms for quick-match, 3s for search).

### 4. **Sentence Extraction Processing** (Minor Impact)
**Current:** Regex-based sentence splitting and fuzzy matching for each chunk.

**Time:** ~10-50ms per chunk (depends on chunk size)

**Optimization:** 
- Cache sentence splits
- Use faster regex patterns
- Limit sentence extraction to top N chunks only

### 5. **Pre-filter Logging** (Minor Impact - Already Fixed)
**Status:** ✅ Reduced from full text to 100-char previews

### 6. **Debug Logging** (Minor Impact - Already Fixed)
**Status:** ✅ Removed verbose debug statements

### 7. **Multiple Sequential HTTP Requests**
**Current:** Each HTTP request waits for the previous one to complete.

**Optimization:** Use connection pooling and parallel requests.

## Recommended Optimizations (Priority Order)

### Priority 1: Parallelize RAG Searches
**Impact:** High (can save 200-1500ms)
**Effort:** Medium

```python
# Use concurrent.futures to run searches in parallel
from concurrent.futures import ThreadPoolExecutor

with ThreadPoolExecutor(max_workers=2) as executor:
    doc_rag_future = executor.submit(rag_client.search, prompt)
    memory_rag_future = executor.submit(memory_container_search, prompt)
    
    rag_results = doc_rag_future.result()
    memory_rag_candidates = memory_rag_future.result()
```

### Priority 2: Make LLM Scoring Optional/Faster
**Impact:** High (can save 500-2000ms)
**Effort:** Low

- Add config flag to skip LLM scoring
- Use semantic scores only for follow-up queries
- Reduce LLM scoring max_tokens (currently 200, could be 50)

### Priority 3: Reduce Timeouts
**Impact:** Medium (prevents hanging on slow network)
**Effort:** Low

- Quick-match: 2s → 500ms
- Memory RAG search: 15s → 3s
- Add retry logic for transient failures

### Priority 4: Skip Quick-Match for Follow-ups
**Impact:** Medium (saves 50-200ms)
**Effort:** Low

- For "what else" queries, skip quick-match and go straight to search
- Quick-match is mainly useful for initial queries

### Priority 5: Cache Sentence Splits
**Impact:** Low (saves 10-50ms per chunk)
**Effort:** Medium

- Cache regex splits per chunk
- Only re-process if chunk text changes

## Estimated Latency Improvements

| Optimization | Current Time | Optimized Time | Savings |
|--------------|--------------|----------------|---------|
| Parallel RAG searches | 300-1500ms | 300-1500ms (max of both) | 200-1500ms |
| Skip LLM scoring (optional) | 500-2000ms | 0ms | 500-2000ms |
| Reduce timeouts | 2-15s (worst case) | 0.5-3s (worst case) | 1.5-12s (worst case) |
| Skip quick-match (follow-ups) | 50-200ms | 0ms | 50-200ms |
| **Total Potential Savings** | **850-3700ms** | **~300-1500ms** | **~550-2200ms** |

## Implementation Notes

1. **Parallelization:** Use `ThreadPoolExecutor` for I/O-bound operations (HTTP requests, FAISS searches)
2. **LLM Scoring:** Make it configurable via environment variable or settings
3. **Timeouts:** Make them configurable per endpoint
4. **Caching:** Use LRU cache for sentence splits and quick-match results


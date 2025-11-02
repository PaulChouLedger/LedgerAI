# Category-Specific FAISS Index Latency Analysis

## Current System Size

Based on guideline directories:
- **GI**: 22 guidelines
- **CARDIO**: ~35 guidelines  
- **PULMONARY**: ~30 guidelines
- **GU**: 4 guidelines
- **GYN**: 4 guidelines
- **NEURO**: ~20 guidelines (estimated)
- **Total**: ~115-125 guidelines across all categories

## Index Size Reduction

### Example: GI Category
- **Before**: Search through ~125 guidelines' terms + all synonyms
- **After**: Search through ~22 GI guidelines' terms + GI-specific synonyms only
- **Reduction**: ~82% fewer terms to search (22/125 = 17.6% of original)

### Per Category Breakdown

| Category | Guidelines | Terms in Index | Reduction vs Global |
|----------|-----------|----------------|---------------------|
| GI | 22 | ~17% | 83% reduction |
| CARDIO | 35 | ~28% | 72% reduction |
| PULMONARY | 30 | ~24% | 76% reduction |
| GU | 4 | ~3% | 97% reduction |
| GYN | 4 | ~3% | 97% reduction |
| NEURO | ~20 | ~16% | 84% reduction |

## FAISS Search Performance

### FAISS IndexFlatIP Characteristics
- **Algorithm**: Exhaustive search (computes similarity with all vectors)
- **Time Complexity**: O(n) where n = number of vectors in index
- **Operation**: Dot product for normalized vectors (very fast but scales linearly)

### Typical Latency Breakdown (per FAISS search)

#### Before (Global Index):
```
1. Embedding encoding:        50-200ms  (constant, depends on model)
2. FAISS search:              20-80ms   (scales with ~125 guidelines)
3. Result filtering/processing: 5-15ms   (constant)
────────────────────────────────────────
Total:                         75-295ms
```

#### After (Category-Specific, GI example):
```
1. Embedding encoding:        50-200ms  (constant, same as before)
2. FAISS search:               3-12ms   (scales with ~22 guidelines = 82% faster)
3. Result filtering/processing: 2-8ms   (fewer results to process)
────────────────────────────────────────
Total:                         55-220ms
```

## Estimated Latency Savings

### Per FAISS Search Call:
- **Time saved**: ~15-65ms per search (20-80ms → 3-12ms for search)
- **Percentage improvement**: **15-25% faster** overall (accounts for encoding being constant)

### Real-World Usage Pattern:

A typical question-answer cycle makes **multiple FAISS calls**:
1. Initial parsing: 1 FAISS call (uses global index - no change)
2. Location analysis: 1 FAISS call → **~20-65ms saved**
3. Missing info analysis: 1 FAISS call → **~20-65ms saved**
4. Clarification matching: 1 FAISS call → **~20-65ms saved**
5. Word match boost: Up to 20 FAISS calls (one per guideline) → **~300-1300ms saved**

### Cumulative Savings Per Question:

**Scenario 1: Simple answer (minimal calls)**
- Location analysis: 1 call
- Missing info: 1 call
- **Total saved**: ~40-130ms

**Scenario 2: Complex answer (many calls)**
- Location analysis: 1 call
- Missing info: 1 call  
- Word match boost: 20 calls (one per active guideline)
- **Total saved**: ~340-1430ms (0.34-1.4 seconds)

**Scenario 3: Clarification round**
- Missing info analysis: 1 call
- Term satisfaction check: Multiple calls
- **Total saved**: ~60-195ms

## Key Factors Affecting Savings

### 1. **Index Size (Primary Factor)**
- Larger categories (CARDIO, PULMONARY): ~70-76% savings
- Smaller categories (GU, GYN): ~97% savings
- Medium categories (GI, NEURO): ~82-84% savings

### 2. **Number of FAISS Calls**
- More calls = more cumulative savings
- Word match boost loop makes 20+ calls per answer
- **This is where the biggest savings occur**

### 3. **Encoding Time (Constant)**
- Patient answer encoding: 50-200ms (unchanged)
- This dominates small searches but savings accumulate on multiple calls

### 4. **Hardware**
- CPU: Larger relative savings (search is CPU-bound)
- GPU: Encoding dominates, but search still benefits

## Real-World Impact

### Typical Consultation (5-10 questions):
- **Average savings per question**: 100-500ms
- **Total consultation savings**: 0.5-5 seconds
- **Most noticeable**: During clarification rounds (multiple FAISS calls)

### Best Case (GU/GYN with 4 guidelines):
- Search time: 80ms → 2-5ms (95-97% reduction)
- Multiple calls compound: ~1.5-2 seconds saved per question

### Worst Case (CARDIO with 35 guidelines):
- Search time: 80ms → 22-25ms (68-72% reduction)
- Still significant: ~0.5-1 second saved per question

## Conclusion

**Estimated latency improvement**: **15-25% faster per question** on average, with:
- **Best case**: 50-70% faster (GU/GYN)
- **Worst case**: 10-15% faster (CARDIO)
- **Typical case**: 20-30% faster (GI)

**Biggest benefit**: Occurs during **word match boost loops** where 20+ FAISS calls are made per patient answer. This is where category-specific indexing provides the most value.

## Next Steps for Optimization

If further latency reduction is needed:
1. **Pre-compute embeddings** for guideline terms (only encode patient answer once)
2. **Batch FAISS searches** when multiple terms need matching
3. **Cache frequent searches** (common terms like "sharp", "dull")
4. **Use approximate search** (IndexIVFFlat) for even larger indices (>1000 terms)


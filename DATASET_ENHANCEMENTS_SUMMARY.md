# Dataset Enhancements for Multiple Entity Extraction

## Date: 2025-01-16

## Problem
Training logs showed poor multiple entity extraction performance:
- Only 10% complete extractions (2/20 examples)
- 45% partial extractions (missing 1-2 entities)
- 35% complete failures
- Model stops after finding first 2-3 entities instead of extracting ALL

## Root Cause Analysis
1. **Round-robin distribution**: Items distributed evenly across chunks using `i % num_chunks`
2. **Insufficient multi-entity examples**: Only 2-4 items per query, often in same chunk
3. **Lack of scattered distribution**: Items not explicitly scattered across multiple chunks
4. **Insufficient list query examples**: Not enough examples forcing complete extraction

## Enhancements Implemented

### 1. Enhanced Distribution Logic for List/Entity Queries
**Location**: `generate_rag_dataset_v2.py` lines 675-720

**Changes**:
- **Before**: Round-robin distribution (`i % num_chunks`) - items distributed evenly
- **After**: Explicit scattering across multiple chunks for list/entity queries
  - For 2-3 items: Put in 2 different chunks
  - For 4+ items: Put in 3+ different chunks
  - Ensures model MUST read multiple chunks to find ALL items

**Code Logic**:
```python
if query_type in ["list", "entity"]:
    # Scatter items across chunks, ensuring multiple chunks have items
    num_chunks_with_items = min(len(relevant_info), max(2, num_chunks))
    chunks_with_items = random.sample(range(num_chunks), num_chunks_with_items)
    # Distribute items across selected chunks
```

### 2. Increased Entity Count for Multi-Chunk Patterns
**Location**: `generate_rag_dataset_v2.py` lines 580-588

**Changes**:
- **Before**: `num_relevant_items = random.randint(2, 4)` for all patterns
- **After**: For `multi_chunk` and `role_filtering` patterns:
  - Entity queries: Generate 3-4 entities (not just 2)
  - Forces model to extract multiple entities, not just first match

### 3. Increased List Item Count
**Location**: `generate_rag_dataset_v2.py` lines 589-608

**Changes**:
- **Before**: `num_relevant_items = random.randint(2, 4)` for all patterns
- **After**: For `multi_chunk` and `mixed_content` patterns:
  - List queries: Generate 3-4 items (not just 2)
  - Forces model to extract ALL items, not stop after first match

### 4. Prioritized List/Entity Queries in Multi-Chunk Patterns
**Location**: `generate_rag_dataset_v2.py` lines 558-570

**Changes**:
- **Before**: Random query template selection
- **After**: For `multi_chunk` and `role_filtering` patterns:
  - 70% chance of selecting list/entity query templates
  - These patterns specifically designed to teach complete multi-entity extraction

### 5. Increased Pattern Counts for Multi-Entity Patterns
**Location**: `generate_rag_dataset_v2.py` lines 830-842

**Changes**:
- **Before**:
  - `multi_chunk`: 1200 examples (19.2%)
  - `role_filtering`: 900 examples (14.4%)
- **After**:
  - `multi_chunk`: 1500 examples (24.0%) - **+300 examples**
  - `role_filtering`: 1200 examples (19.2%) - **+300 examples**
- **Compensated by reducing**:
  - `mixed_content`: 900 → 700 (-200)
  - `cross_entity`: 900 → 800 (-100)
  - `synthesis`: 600 → 550 (-50)
  - `not_found`: 600 → 550 (-50)
  - `comparison`: 450 → 400 (-50)
  - `relationship`: 450 → 400 (-50)
  - `analytical`: 250 → 150 (-100)
- **Total**: Still 6250 examples

## Expected Impact

### Before Enhancements:
- Items distributed round-robin (evenly across chunks)
- 2-4 items per query, often in same chunk
- Model could find 2-3 items in first chunk and stop
- Only 10% complete extraction rate

### After Enhancements:
- Items explicitly scattered across 2-3 chunks
- 3-4 items per query (more for multi-chunk patterns)
- Model MUST read multiple chunks to find ALL items
- **Expected**: 50-70% complete extraction rate

## Key Improvements

1. **Scattered Distribution**: Items no longer round-robin - explicitly placed in different chunks
2. **More Multi-Entity Examples**: 43.2% of dataset (2700/6250) focused on multi-entity extraction
3. **Higher Item Counts**: 3-4 items per query (instead of 2-4 with bias toward 2)
4. **Pattern Prioritization**: Multi-chunk patterns prioritize list/entity queries (70% chance)

## Verification

### Pattern Distribution:
```
mixed_content       :  700 ( 11.2%)
multi_chunk         : 1500 ( 24.0%) ← INCREASED
role_filtering      : 1200 ( 19.2%) ← INCREASED
cross_entity        :  800 ( 12.8%)
synthesis           :  550 (  8.8%)
not_found           :  550 (  8.8%)
comparison          :  400 (  6.4%)
relationship        :  400 (  6.4%)
analytical          :  150 (  2.4%)
─────────────────────────────────────
Total               : 6250 (100.0%)
```

### Multi-Entity Focus:
- `multi_chunk` + `role_filtering` = 2700 examples (43.2% of dataset)
- These patterns now prioritize list/entity queries (70% chance)
- Items explicitly scattered across chunks

## Next Steps

1. **Regenerate Dataset**:
   ```bash
   python generate_rag_dataset_v2.py
   ```

2. **Verify Distribution**:
   - Check that list/entity queries have items in multiple chunks
   - Verify 3-4 items per query for multi-chunk patterns
   - Confirm items are scattered (not all in same chunk)

3. **Retrain Model**:
   - Use enhanced dataset with scattered multi-entity examples
   - Monitor multiple entity extraction during training
   - Expected improvement: 50-70% complete extraction rate

4. **Evaluate**:
   - Run comprehensive evaluation after training
   - Compare with previous results
   - Verify improvement in multiple entity extraction

## Technical Details

### Distribution Algorithm (List/Entity Queries):
1. Determine number of chunks with items: `min(len(items), max(2, num_chunks))`
2. Randomly select which chunks will contain items
3. Distribute items across selected chunks using modulo
4. Ensures items are in different chunks, forcing complete reading

### Example:
- Query: "who are the managers of TechCorp?" (4 managers expected)
- Chunks: 4 chunks total
- Distribution:
  - Manager 1: Chunk 1
  - Manager 2: Chunk 2
  - Manager 3: Chunk 3
  - Manager 4: Chunk 1 (second item in Chunk 1)
- Model MUST read all 4 chunks to find all 4 managers

## Notes

- Total dataset size remains 6250 examples
- All existing patterns preserved, just rebalanced
- Enhanced distribution only applies to list/entity queries
- Other query types use original distribution logic
- System prompt already enhanced with list query instructions (previous update)

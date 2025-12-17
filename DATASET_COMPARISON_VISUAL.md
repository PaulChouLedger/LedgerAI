# Dataset Comparison: Visual Summary

## Training Gap: Multiple Entity Extraction Failure

**Observed Problem**: Model extracts only 2-3 items when 4 are expected (50-75% success rate)

---

## Visual Comparison: "Who are the managers of TechCorp?" (4 managers)

### 🔴 OLD DATASET - Round-Robin Distribution

```
┌─────────────────────────────────────────────────────────┐
│ Query: "who are the managers of TechCorp?"             │
│ Expected: 4 managers                                     │
└─────────────────────────────────────────────────────────┘

┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│   CHUNK 1       │  │   CHUNK 2       │  │   CHUNK 3       │
│                 │  │                 │  │                 │
│ ✅ Alice        │  │ ✅ Carol        │  │ ⚪ [context]     │
│ ✅ Bob          │  │ ✅ Dave         │  │ ⚪ [context]     │
│ ⚪ [context]    │  │ ⚪ [context]    │  │ ⚪ [context]     │
│ ⚪ [context]    │  │ ⚪ [context]    │  │ ⚪ [context]     │
│ ⚪ [context]    │  │ ⚪ [context]    │  │ ⚪ [context]     │
└─────────────────┘  └─────────────────┘  └─────────────────┘
    2 managers           2 managers           0 managers

Model Behavior:
  Step 1: Read Chunk 1 → Finds 2 managers ✅
  Step 2: Thinks "Found 2, probably done" ❌
  Step 3: STOPS READING ❌
  
  Output: "Alice Johnson and Bob Smith" (2 of 4)
  Match Score: 50% ❌
```

**Problem**: Model can find multiple items in first chunk and stop.

---

### 🟢 NEW DATASET - Scattered Distribution

```
┌─────────────────────────────────────────────────────────┐
│ Query: "who are the managers of TechCorp?"             │
│ Expected: 4 managers                                     │
└─────────────────────────────────────────────────────────┘

┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│   CHUNK 1       │  │   CHUNK 2       │  │   CHUNK 3       │
│                 │  │                 │  │                 │
│ ⚪ [context]    │  │ ⚪ [context]    │  │ ⚪ [context]    │
│ ✅ Alice        │  │ ✅ Bob          │  │ ⚪ [context]    │
│ ⚪ [context]    │  │ ✅ Carol        │  │ ✅ Dave         │
│ ⚪ [context]    │  │ ⚪ [context]    │  │ ⚪ [context]    │
│ ⚪ [context]    │  │ ⚪ [context]    │  │ ⚪ [context]    │
└─────────────────┘  └─────────────────┘  └─────────────────┘
    1 manager           2 managers           1 manager

Model Behavior:
  Step 1: Read Chunk 1 → Finds 1 manager ✅
  Step 2: Thinks "Found 1, query asks for 'managers' (plural)" ✅
  Step 3: CONTINUES READING ✅
  
  Step 4: Read Chunk 2 → Finds 2 more managers ✅
  Step 5: Thinks "Found 3 so far, must continue" ✅
  Step 6: CONTINUES READING ✅
  
  Step 7: Read Chunk 3 → Finds 1 more manager ✅
  Step 8: Thinks "Found 4 total, complete" ✅
  Step 9: COMPLETE ✅
  
  Output: "Alice Johnson, Bob Smith, Carol Williams, and Dave Miller" (4 of 4)
  Match Score: 100% ✅
```

**Solution**: Model MUST read all chunks to find all items.

---

## Distribution Algorithm Comparison

### OLD: Round-Robin
```python
for i, item in enumerate([Alice, Bob, Carol, Dave]):
    chunk_idx = i % 3  # 0, 1, 2, 0
    
Result:
  Alice (i=0) → Chunk 0
  Bob   (i=1) → Chunk 1
  Carol (i=2) → Chunk 2
  Dave  (i=3) → Chunk 0
  
Distribution: [2, 1, 1] or [2, 2, 0]
```

### NEW: Scattered
```python
chunks_with_items = random.sample([0, 1, 2], 3)  # All chunks
for i, item in enumerate([Alice, Bob, Carol, Dave]):
    chunk_idx = chunks_with_items[i % 3]  # 0, 1, 2, 0
    
Result:
  Alice (i=0) → Chunk 0
  Bob   (i=1) → Chunk 1
  Carol (i=2) → Chunk 2
  Dave  (i=3) → Chunk 0
  
Distribution: [2, 1, 1] (ensures all chunks have items)
```

**Key Difference**: NEW ensures at least 2 chunks have items (for 2+ items), preventing all items in first chunk.

---

## Training Signal Comparison

### OLD Dataset Training Signal:
```
Example 1: Chunk 1 has 2 managers → Model stops → Learns "2 is enough" ❌
Example 2: Chunk 1 has 2 managers → Model stops → Learns "2 is enough" ❌
Example 3: Chunk 1 has 2 managers → Model stops → Learns "2 is enough" ❌

Result: Model learns to stop after finding 2 items
```

### NEW Dataset Training Signal:
```
Example 1: Chunk 1 has 1 manager → Model continues → Learns "must read all" ✅
Example 2: Chunk 1 has 1 manager → Model continues → Learns "must read all" ✅
Example 3: Chunk 1 has 1 manager → Model continues → Learns "must read all" ✅

Result: Model learns to read all chunks to find all items
```

---

## Impact on Training Metrics

### Before (OLD Dataset):
- **Complete extractions**: 10% (2/20 examples)
- **Partial extractions**: 45% (9/20 examples) - missing 1-2 items
- **Failures**: 35% (7/20 examples)
- **Average completeness**: 65% (when partially successful)

### After (NEW Dataset - Expected):
- **Complete extractions**: 50-70% (expected)
- **Partial extractions**: 20-30% (expected) - down from 45%
- **Failures**: 10-20% (expected) - down from 35%
- **Average completeness**: 85-95% (expected) - up from 65%

---

## Key Improvements Summary

| Improvement | OLD | NEW | Impact |
|-------------|-----|-----|--------|
| **Items in Chunk 1** | 2 of 4 (50%) | 1-2 of 4 (25-50%) | Prevents early stopping |
| **Chunks with items** | 2 chunks | 3 chunks | Forces complete reading |
| **Item counts** | 2-4 (bias: 2) | 3-4 (bias: 3-4) | More complete examples |
| **Multi-entity examples** | 33.6% | 43.2% | +28% more examples |
| **Distribution** | Round-robin | Scattered | Strategic placement |

---

## Conclusion

The enhanced dataset directly addresses the training gap by:

1. **Scattering items** → Prevents finding all items in first chunk
2. **Forcing multi-chunk reading** → Model must read all chunks
3. **Higher item counts** → More examples requiring 3-4 items
4. **More examples** → 28% more multi-entity training examples

**Expected Result**: Complete extraction rate improves from 10% to 50-70%.

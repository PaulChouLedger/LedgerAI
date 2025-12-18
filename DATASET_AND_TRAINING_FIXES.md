# Dataset and Training Fixes - Implementation Guide

## 🔍 Root Cause Analysis

### Issue #1: Model Stops After First Few Entities
**Evidence:**
- Model extracts 2/4 co-founders (50% accuracy)
- Missing entities are in Chunk 2 (model didn't read it completely)
- Model doesn't mention chunks in response

**Root Cause:**
- Dataset may not emphasize "extract ALL" strongly enough
- Model learned to stop after finding some entities
- System prompt mentions "extract ALL" but examples may not reinforce it

### Issue #2: Answer Type Classification Bias
**Evidence:**
- Model defaults to "comparison" (9.5% of dataset)
- Should be "relationship", "analytical", "process" → outputs "comparison"

**Root Cause:**
- No explicit query pattern → answer_type mapping in system prompt
- Model infers from vague descriptions

### Issue #3: Role Filtering Issues
**Evidence:**
- Model includes "Will Specht" (Head of Engineering) when asked for "co-founders"

**Root Cause:**
- Model not learning exact role matching
- Dataset may not have enough negative examples (wrong role = exclude)

---

## 🔧 Dataset Generator Fixes

### Fix #1: Strengthen "Extract ALL" Emphasis in System Prompt

**Current System Prompt:**
```
2. Extract ALL matching items (do NOT stop after first match)
```

**Enhanced System Prompt:**
```
2. Extract ALL matching items (do NOT stop after first match)

CRITICAL: Partial extraction is WRONG and will be penalized.

Examples:
- Query: "who are the co-founders of TechCorp?"
  - WRONG: {"items": ["John Smith"]}  ❌ Only 1 of 4 co-founders
  - CORRECT: {"items": ["John Smith", "Sarah Jones", "Mike Brown", "Alice White"]}  ✅ All 4 co-founders

- Query: "list the features of ProductX"
  - WRONG: {"items": ["feature1", "feature2"]}  ❌ Only 2 of 5 features
  - CORRECT: {"items": ["feature1", "feature2", "feature3", "feature4", "feature5"]}  ✅ All 5 features

If you find 4 entities, include all 4. If you find 10 items, include all 10.
Stopping after finding some items is INCORRECT.
```

### Fix #2: Increase Multi-Entity Examples

**Current:**
```python
"multi_chunk": 1500,
```

**Fix:**
```python
"multi_chunk": 2500,  # Increased from 1500 - 40% more examples
```

**Ensure:**
- All multi_chunk examples have 3-5 entities across 2-4 chunks
- Entities are distributed across chunks (not all in first chunk)
- Model must read multiple chunks to get complete answer

### Fix #3: Add Explicit "Extract ALL" Examples

Add examples that explicitly show:
1. Reading all chunks
2. Extracting all entities
3. What happens if you stop early (wrong)

### Fix #4: Add Answer Type Mapping Examples

Add to system prompt:
```
ANSWER TYPE SELECTION (CRITICAL):
- "how are X and Y related?" → answer_type: "relationship"
- "what is the connection between X and Y?" → answer_type: "relationship"
- "why did X [action]?" → answer_type: "analytical"
- "what caused X to [action]?" → answer_type: "analytical"
- "what is the difference between X and Y?" → answer_type: "comparison"
- "compare X and Y" → answer_type: "comparison"
- "how does [process] work?" → answer_type: "process"
- "what is the process for X?" → answer_type: "process"
- "who are the [role] of X?" → answer_type: "entities"
- "list the [items] of X" → answer_type: "list"
```

### Fix #5: Ensure Multi-Chunk Distribution

**Current Issue:** Some multi-chunk examples may put all entities in first chunk
**Fix:** Always distribute entities across chunks for multi_chunk pattern

```python
# In generate_rag_dataset_v3_json.py
if pattern_type == "multi_chunk" and query_type in ["entity", "list"]:
    # CRITICAL: Distribute items across chunks - don't put all in first chunk
    # This forces model to read all chunks
    items_per_chunk = max(1, len(relevant_info) // num_chunks)
    # Ensure at least 2 chunks have entities
    if len(relevant_info) >= 3:
        # Distribute: chunk 1 gets some, chunk 2 gets some, etc.
        # Don't put all in chunk 1
```

---

## 🎯 Training Script Fixes

### Fix #1: Increase LoRA Rank

**Current:**
```python
LORA_R = 6
```

**Fix:**
```python
LORA_R = 8  # Increased from 6 - more capacity for multi-entity patterns
# Or even 10 if model size allows
```

**Reasoning:** Multi-entity extraction across chunks is a complex pattern requiring more model capacity.

### Fix #2: Add Answer Type Examples in System Prompt

The system prompt already has answer_type mapping (we added it), but ensure it's in every training example.

### Fix #3: Monitor Extraction Completeness During Training

Add a callback that tracks:
- Average number of entities extracted vs expected
- Answer type classification accuracy
- Early stopping if completeness < 80%

### Fix #4: Increase Epochs if Needed

**Current:** 7 epochs
**Consider:** 10 epochs if model needs more training

---

## 🧪 Additional Diagnostic Tests

### Test A: Chunk Distribution Impact
**Purpose:** Verify model reads entities from later chunks

```python
test = {
    "query": "who are the co-founders of X?",
    "chunks": [
        {"text": "Entity A in chunk 1", "score": 0.85},
        {"text": "Entity B in chunk 2", "score": 0.85},
        {"text": "Entity C in chunk 3", "score": 0.85},
    ],
    "expected": ["Entity A", "Entity B", "Entity C"]
}
# Check if model extracts all 3 or stops after chunk 1
```

### Test B: Answer Type Confidence
**Purpose:** Measure how confident model is in answer_type selection

### Test C: Role Filtering Edge Cases
**Purpose:** Test exact role matching with similar roles

```python
test = {
    "query": "who are the co-founders?",
    "chunks": [
        {"text": "John is Co-Founder"},  # ✅
        {"text": "Jane is CEO"},  # ❌
        {"text": "Mike is Co-Founder and CEO"},  # ✅ (has Co-Founder)
    ],
    "expected": ["John", "Mike"],
    "should_exclude": ["Jane"]
}
```

---

## 📋 Implementation Checklist

### Before Next Training Session

- [ ] **Update dataset generator:**
  - [ ] Increase multi_chunk examples to 2500
  - [ ] Strengthen "extract ALL" in system prompt
  - [ ] Add answer_type mapping examples
  - [ ] Ensure multi-chunk entities are distributed (not all in chunk 1)
  - [ ] Add explicit "extract ALL" examples

- [ ] **Update training script:**
  - [ ] Increase LoRA rank to 8-10
  - [ ] Add extraction completeness monitoring
  - [ ] Add answer_type accuracy tracking
  - [ ] Consider increasing epochs to 10

- [ ] **Run diagnostics on current model:**
  - [ ] Run comprehensive_model_diagnostics.py
  - [ ] Document all issues
  - [ ] Create baseline metrics

- [ ] **Regenerate dataset:**
  - [ ] Run generate_rag_dataset_v3_json.py with fixes
  - [ ] Verify dataset has 2500+ multi-chunk examples
  - [ ] Verify system prompt has "extract ALL" emphasis

- [ ] **Train new model:**
  - [ ] Use updated dataset
  - [ ] Use updated training config (LoRA rank 8-10)
  - [ ] Monitor extraction completeness during training

- [ ] **Test new model:**
  - [ ] Run comprehensive diagnostics
  - [ ] Compare with baseline
  - [ ] Target: >90% multi-entity extraction accuracy

---

## 🎯 Success Metrics

### Multi-Entity Extraction
- **Current:** 50% (2/4 co-founders)
- **Target:** >90%
- **Gap:** +40%

### Answer Type Classification
- **Current:** ~60% (frequently defaults to "comparison")
- **Target:** >85%
- **Gap:** +25%

### Role Filtering
- **Current:** ~75% (includes non-matching roles)
- **Target:** >95%
- **Gap:** +20%

---

## 🚀 Priority Order

1. **HIGH:** Update dataset generator (increase multi_chunk, strengthen prompt)
2. **HIGH:** Increase LoRA rank in training script
3. **MEDIUM:** Add monitoring during training
4. **MEDIUM:** Run comprehensive diagnostics
5. **LOW:** Increase epochs (only if needed)

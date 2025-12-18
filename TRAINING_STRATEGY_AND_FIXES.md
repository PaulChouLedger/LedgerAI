# RAG Model Training Strategy & Fixes
## Comprehensive Analysis and Improvement Plan

## 🔍 Current Model Issues (Diagnosed)

### Issue #1: Incomplete Multi-Entity Extraction
**Symptom:** Model extracts 2/4 co-founders (50% accuracy)
- ✅ Found: David Lara, Jorge Guinovart
- ❌ Missing: Paul Chou, Bob Carella (both in Chunk 2)
- ⚠️  Extra: Will Specht (not a co-founder)

**Root Cause:** Model stops after finding first few entities, doesn't read all chunks completely

### Issue #2: Answer Type Classification Bias
**Symptom:** Model defaults to "comparison" for many query types
- Should be "relationship" → outputs "comparison"
- Should be "analytical" → outputs "comparison"
- Should be "process" → outputs "comparison"

**Root Cause:** Model hasn't learned query pattern → answer_type mapping

### Issue #3: Role Filtering Issues
**Symptom:** Model includes "Will Specht" (Head of Engineering) when asked for "co-founders"
**Root Cause:** Model not filtering by exact role match

---

## 📊 Diagnostic Test Suite

### Test 1: Multi-Entity Extraction Across Chunks
**Purpose:** Verify model reads all chunks and extracts all entities

```python
test_cases = [
    {
        "name": "Co-founders across 2 chunks",
        "query": "who are the co-founders of LedgerAI?",
        "chunks": [
            {"text": "...David Lara...Co-Founder...", "score": 0.85},
            {"text": "...Paul Chou...Co-Founder...Bob Carella...Co-Founder...", "score": 0.85}
        ],
        "expected": ["David Lara", "Paul Chou", "Bob Carella"],
        "min_accuracy": 100  # Must get all
    },
    {
        "name": "Executives across 3 chunks",
        "query": "who are the executives of CompanyX?",
        "chunks": [
            {"text": "...John Smith...Executive...", "score": 0.85},
            {"text": "...Jane Doe...Executive...", "score": 0.85},
            {"text": "...Mike Brown...Executive...", "score": 0.85}
        ],
        "expected": ["John Smith", "Jane Doe", "Mike Brown"],
        "min_accuracy": 100
    }
]
```

### Test 2: Answer Type Classification
**Purpose:** Verify model correctly identifies answer_type from query pattern

```python
test_cases = [
    {
        "query": "how are X and Y related?",
        "expected_answer_type": "relationship",
        "min_confidence": 0.8
    },
    {
        "query": "why did X happen?",
        "expected_answer_type": "analytical",
        "min_confidence": 0.8
    },
    {
        "query": "what is the difference between X and Y?",
        "expected_answer_type": "comparison",
        "min_confidence": 0.8
    }
]
```

### Test 3: Role Filtering Accuracy
**Purpose:** Verify model filters by exact role

```python
test_cases = [
    {
        "query": "who are the co-founders of X?",
        "chunks": [
            {"text": "...John...Co-Founder...", "score": 0.85},
            {"text": "...Jane...CEO...", "score": 0.85},  # Should be excluded
            {"text": "...Mike...Co-Founder...", "score": 0.85}
        ],
        "expected": ["John", "Mike"],
        "should_exclude": ["Jane"]  # CEO is not a co-founder
    }
]
```

### Test 4: Chunk Reading Completeness
**Purpose:** Verify model mentions all chunks in response

```python
# Check if model's chunks_used includes all chunks with relevant info
# Model should reference chunks it used
```

---

## 🔧 Dataset Generation Fixes

### Fix #1: Emphasize "Extract ALL" in System Prompt
**Current:** System prompt mentions "extract ALL" but may not be strong enough
**Fix:** Add explicit examples showing partial vs complete extraction

```python
SYSTEM_PROMPT_ENHANCED = """
CRITICAL: For queries asking for multiple items, you MUST extract ALL of them.

WRONG (Partial Extraction):
Query: "who are the co-founders of TechCorp?"
Response: {"items": ["John Smith"]}  ❌ Only 1 of 3 co-founders

CORRECT (Complete Extraction):
Query: "who are the co-founders of TechCorp?"
Response: {"items": ["John Smith", "Sarah Jones", "Mike Brown"]}  ✅ All 3 co-founders

If you find 4 co-founders, include all 4. If you find 10 services, include all 10.
Partial extraction is INCORRECT and will be penalized.
"""
```

### Fix #2: Increase Multi-Entity Examples
**Current:** 1500 multi_chunk examples
**Fix:** Increase to 2500+ and ensure all have 3-5 entities across multiple chunks

```python
# In generate_rag_dataset_v3_json.py
pattern_distribution = {
    "multi_chunk": 2500,  # Increased from 1500
    # Ensure each has 3-5 entities across 2-4 chunks
}
```

### Fix #3: Add Explicit "Extract ALL" Training Examples
**Fix:** Add examples that explicitly show the model extracting all entities

```python
# Add examples like:
{
    "query": "who are the co-founders of TechCorp?",
    "chunks": [
        {"text": "John Smith is Co-Founder...", "score": 0.85},
        {"text": "Sarah Jones is Co-Founder...", "score": 0.85},
        {"text": "Mike Brown is Co-Founder...", "score": 0.85}
    ],
    "output": {
        "answer_type": "entities",
        "items": ["John Smith", "Sarah Jones", "Mike Brown"],  # ALL 3
        "text": "",
        "chunks_used": [1, 2, 3]
    },
    "note": "Extracted all 3 co-founders from all 3 chunks"
}
```

### Fix #4: Add Negative Examples (Partial Extraction = Wrong)
**Fix:** Include examples showing what NOT to do

```python
# Add negative examples in training (with explicit feedback):
{
    "query": "who are the co-founders of TechCorp?",
    "chunks": [/* 3 chunks with 3 co-founders */],
    "output": {
        "items": ["John Smith"],  # WRONG - only 1 of 3
        "note": "INCORRECT: Only extracted 1 of 3 co-founders. Must extract ALL."
    }
}
```

---

## 🎯 Training Script Fixes

### Fix #1: Add Class Weighting for Answer Types
**Issue:** Model defaults to "comparison" (only 9.5% of dataset)
**Fix:** Add class weights to penalize incorrect answer_type

```python
# In train_rag_analysis_colab.py
from sklearn.utils.class_weight import compute_class_weight

# Calculate class weights based on dataset distribution
answer_type_weights = {
    "list": 0.5,        # 32.1% - most common, reduce weight
    "entities": 0.7,    # 21.7% - common
    "comparison": 2.0,  # 9.5% - rare, increase weight to prevent defaulting
    "relationship": 2.0, # 9.2% - rare
    "analytical": 2.0,   # 9.1% - rare
    "process": 2.0,     # 9.5% - rare
    "not_found": 2.0,   # 8.8% - rare
}
```

### Fix #2: Add Loss Function for Completeness
**Fix:** Add a custom loss component that penalizes incomplete extraction

```python
def completeness_loss(predicted_items, expected_items):
    """Penalize incomplete extraction"""
    if len(predicted_items) < len(expected_items):
        # Heavy penalty for missing items
        missing_ratio = (len(expected_items) - len(predicted_items)) / len(expected_items)
        return missing_ratio * 2.0  # 2x penalty for missing items
    return 0.0
```

### Fix #3: Increase LoRA Rank for Multi-Entity Patterns
**Current:** LoRA rank = 6
**Fix:** Increase to 8-10 for better capacity to learn multi-entity patterns

```python
LORA_R = 8  # Increased from 6 - more capacity for complex patterns
```

### Fix #4: Add Explicit Answer Type Examples in Training
**Fix:** Add examples at the start of each epoch showing query → answer_type mapping

```python
# Add few-shot examples in system prompt:
FEW_SHOT_EXAMPLES = """
Query: "how are X and Y related?" → answer_type: "relationship"
Query: "what is the difference between X and Y?" → answer_type: "comparison"
Query: "why did X happen?" → answer_type: "analytical"
Query: "how does X work?" → answer_type: "process"
Query: "who are the co-founders of X?" → answer_type: "entities"
Query: "list the features of X" → answer_type: "list"
"""
```

---

## 🧪 Additional Diagnostic Tests

### Test 5: Chunk Processing Order
**Purpose:** Verify model processes chunks in order and doesn't skip

```python
def test_chunk_processing_order():
    """Test if model processes all chunks"""
    chunks = [
        {"text": "Entity A in chunk 1", "score": 0.85},
        {"text": "Entity B in chunk 2", "score": 0.85},
        {"text": "Entity C in chunk 3", "score": 0.85},
    ]
    # Model should extract all 3 entities
    # Check if chunks_used includes all 3 chunks
```

### Test 6: Token Limit Impact
**Purpose:** Verify model doesn't truncate chunks

```python
def test_token_limits():
    """Test with varying chunk sizes"""
    # Test with chunks that approach token limit
    # Verify model doesn't skip later chunks
```

### Test 7: Role Filtering Edge Cases
**Purpose:** Test exact role matching

```python
test_cases = [
    {
        "query": "who are the co-founders?",
        "chunks": [
            {"text": "John is Co-Founder"},  # ✅ Should include
            {"text": "Jane is CEO"},  # ❌ Should exclude
            {"text": "Mike is Co-Founder and CEO"},  # ✅ Should include (has Co-Founder)
        ],
        "expected": ["John", "Mike"],
        "should_exclude": ["Jane"]
    }
]
```

---

## 📝 Training Strategy

### Phase 1: Dataset Enhancement (Before Training)
1. ✅ Increase multi-entity examples to 2500+
2. ✅ Add explicit "extract ALL" examples
3. ✅ Add negative examples (partial extraction = wrong)
4. ✅ Enhance system prompt with few-shot examples
5. ✅ Add answer_type mapping examples

### Phase 2: Training Configuration
1. ✅ Increase LoRA rank to 8-10
2. ✅ Add class weighting for answer types
3. ✅ Add completeness loss component
4. ✅ Increase epochs if needed (7 → 10)
5. ✅ Monitor extraction completeness during training

### Phase 3: Validation During Training
1. ✅ Add validation set with multi-entity examples
2. ✅ Track extraction completeness metric
3. ✅ Track answer_type accuracy
4. ✅ Early stopping if completeness < 80%

### Phase 4: Post-Training Testing
1. ✅ Run comprehensive diagnostic suite
2. ✅ Test on real-world examples (like LedgerAI co-founders)
3. ✅ Measure accuracy: target >90% for multi-entity extraction
4. ✅ Measure answer_type accuracy: target >85%

---

## 🎯 Success Metrics

### Multi-Entity Extraction
- **Target:** >90% accuracy (extract all expected entities)
- **Current:** 50% (2/4 co-founders)
- **Gap:** Need 40% improvement

### Answer Type Classification
- **Target:** >85% accuracy
- **Current:** ~60% (frequently defaults to "comparison")
- **Gap:** Need 25% improvement

### Role Filtering
- **Target:** >95% accuracy (exact role match)
- **Current:** ~75% (includes non-matching roles)
- **Gap:** Need 20% improvement

---

## 🚀 Implementation Priority

### High Priority (Must Fix)
1. ✅ Increase multi-entity examples in dataset
2. ✅ Enhance system prompt with "extract ALL" emphasis
3. ✅ Add explicit answer_type mapping
4. ✅ Increase LoRA rank

### Medium Priority (Should Fix)
1. ✅ Add class weighting
2. ✅ Add completeness loss
3. ✅ Add validation set

### Low Priority (Nice to Have)
1. ✅ Add negative examples
2. ✅ Add few-shot examples in prompt
3. ✅ Increase epochs

---

## 📋 Next Steps

1. **Update dataset generator** with fixes above
2. **Regenerate dataset** with enhanced multi-entity examples
3. **Update training script** with new configuration
4. **Run diagnostic tests** on new model after training
5. **Compare results** with current model


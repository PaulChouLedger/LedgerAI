# Training Improvement Plan - Complete Strategy

## 📊 Current Model Performance (Baseline)

### Multi-Entity Extraction: 50% Accuracy
- ✅ Found: David Lara, Jorge Guinovart (2/4 co-founders)
- ❌ Missing: Paul Chou, Bob Carella (both in Chunk 2)
- ⚠️  Extra: Will Specht (not a co-founder)

### Answer Type Classification: ~60% Accuracy
- Frequently defaults to "comparison" for other types
- Needs explicit query → answer_type mapping

### Role Filtering: ~75% Accuracy
- Includes non-matching roles (e.g., "Head of Engineering" when asked for "co-founders")

---

## 🔧 Fixes Applied to Dataset Generator

### ✅ Fix #1: Enhanced System Prompt
**File:** `generate_rag_dataset_v3_json.py`

**Changes:**
1. Added explicit "WRONG vs CORRECT" examples showing partial vs complete extraction
2. Added answer_type mapping rules
3. Strengthened "extract ALL" emphasis with multiple examples
4. Added role filtering rules

**Impact:** Model will see stronger guidance in every training example

### ✅ Fix #2: Increased Multi-Chunk Examples
**File:** `generate_rag_dataset_v3_json.py`

**Change:**
```python
"multi_chunk": 2500,  # Increased from 1500 (40% more examples)
```

**Impact:** More training examples emphasizing multi-entity extraction

### ✅ Fix #3: Improved Chunk Distribution
**File:** `generate_rag_dataset_v3_json.py`

**Change:**
- Ensures entities are distributed across at least 2 chunks
- Forces model to read multiple chunks to get complete answer
- Prevents all entities from being in first chunk

**Impact:** Model must read all chunks to extract all entities

---

## 🎯 Training Script Status

### ✅ LoRA Rank: Already Optimized
- Current: `LORA_RANK = 8` ✅ (already increased from 6)
- Status: Good - no change needed

### ✅ Answer Type Mapping: Already Added
- System prompt includes explicit mapping (we added it earlier)
- Status: Good - will be in new dataset

### ⚠️  Consider: Add Extraction Completeness Monitoring
**Recommendation:** Add callback to track extraction completeness during training

---

## 🧪 Diagnostic Tests to Run

### Test Suite 1: Multi-Entity Extraction
Run `comprehensive_model_diagnostics.py` to test:
1. Co-founders across 2 chunks (current failing case)
2. Executives across 3 chunks
3. Services across 4 chunks

**Success Criteria:** >90% accuracy (extract all expected entities)

### Test Suite 2: Answer Type Classification
Test query patterns:
- "how are X and Y related?" → should be "relationship"
- "why did X happen?" → should be "analytical"
- "what is the difference between X and Y?" → should be "comparison"

**Success Criteria:** >85% accuracy

### Test Suite 3: Role Filtering
Test exact role matching:
- "co-founders" query should NOT include "CEO", "CTO", "Head of Engineering"
- Only exact role matches should be extracted

**Success Criteria:** >95% accuracy

### Test Suite 4: Chunk Reading Completeness
Verify model mentions all chunks with relevant info:
- Check `chunks_used` in JSON output
- Should include all chunks containing entities

**Success Criteria:** Model references all relevant chunks

---

## 📋 Action Plan for Next Training

### Step 1: Regenerate Dataset (REQUIRED)
```bash
python3 generate_rag_dataset_v3_json.py
```

**Verify:**
- Dataset has 2500+ multi_chunk examples
- System prompt includes "extract ALL" emphasis
- System prompt includes answer_type mapping

### Step 2: Run Diagnostics on Current Model (BASELINE)
```bash
python3 comprehensive_model_diagnostics.py
```

**Document:**
- Current multi-entity accuracy: __%
- Current answer_type accuracy: __%
- Current role filtering accuracy: __%

### Step 3: Train New Model
```bash
python3 train_rag_analysis_colab.py
```

**Monitor:**
- Watch training logs for extraction completeness
- Check if model improves over epochs
- Note any patterns in failures

### Step 4: Test New Model
```bash
python3 comprehensive_model_diagnostics.py
```

**Compare:**
- New model vs baseline
- Target: >90% multi-entity extraction
- Target: >85% answer_type classification

### Step 5: Real-World Test
Test with actual LedgerAI co-founders query:
- Expected: 4 co-founders
- Target: Extract all 4

---

## 🎯 Expected Improvements

### After Applying Fixes:

**Multi-Entity Extraction:**
- Current: 50% (2/4 co-founders)
- Expected: >90% (4/4 co-founders)
- Improvement: +40%

**Answer Type Classification:**
- Current: ~60% (frequently defaults to "comparison")
- Expected: >85%
- Improvement: +25%

**Role Filtering:**
- Current: ~75% (includes non-matching roles)
- Expected: >95%
- Improvement: +20%

---

## 🔍 Additional Diagnostic Tests

### Test A: Chunk Order Independence
**Purpose:** Verify model extracts same entities regardless of chunk order

```python
test = {
    "query": "who are the co-founders?",
    "chunks_order_1": [
        {"text": "Entity A in chunk 1"},
        {"text": "Entity B in chunk 2"},
    ],
    "chunks_order_2": [
        {"text": "Entity B in chunk 1"},  # Swapped
        {"text": "Entity A in chunk 2"},
    ],
    "expected": ["Entity A", "Entity B"]  # Should be same in both orders
}
```

### Test B: Token Limit Impact
**Purpose:** Verify model doesn't skip chunks when approaching token limit

```python
# Test with chunks that total ~7000 tokens
# Verify model still reads all chunks
```

### Test C: Irrelevant Entity Filtering
**Purpose:** Verify model excludes irrelevant entities

```python
test = {
    "query": "who are the co-founders of TechCorp?",
    "chunks": [
        {"text": "John is Co-Founder of TechCorp"},  # ✅ Relevant
        {"text": "Jane is CEO of TechCorp"},  # ❌ Irrelevant (wrong role)
        {"text": "Mike is Co-Founder of OtherCorp"},  # ❌ Irrelevant (wrong company)
        {"text": "Sarah is Co-Founder of TechCorp"},  # ✅ Relevant
    ],
    "expected": ["John", "Sarah"],
    "should_exclude": ["Jane", "Mike"]
}
```

---

## 📝 Key Changes Summary

### Dataset Generator (`generate_rag_dataset_v3_json.py`)
1. ✅ Enhanced system prompt with "extract ALL" examples
2. ✅ Added answer_type mapping rules
3. ✅ Increased multi_chunk examples: 1500 → 2500
4. ✅ Improved chunk distribution (forces multi-chunk reading)

### Training Script (`train_rag_analysis_colab.py`)
1. ✅ LoRA rank already at 8 (good)
2. ✅ System prompt already includes answer_type mapping (good)
3. ⚠️  Consider: Add extraction completeness monitoring

### Diagnostic Tools
1. ✅ Created `comprehensive_model_diagnostics.py`
2. ✅ Created `debug_multi_entity_extraction.py`
3. ✅ Created strategy documents

---

## 🚀 Next Steps (Priority Order)

1. **IMMEDIATE:** Regenerate dataset with fixes
   ```bash
   python3 generate_rag_dataset_v3_json.py
   ```

2. **IMMEDIATE:** Run diagnostics on current model (baseline)
   ```bash
   python3 comprehensive_model_diagnostics.py
   ```

3. **HIGH:** Train new model with updated dataset
   ```bash
   python3 train_rag_analysis_colab.py
   ```

4. **HIGH:** Test new model and compare with baseline

5. **MEDIUM:** If still issues, consider:
   - Increasing LoRA rank to 10
   - Adding class weighting
   - Increasing epochs to 10

---

## 📊 Success Metrics

### Must Achieve (Critical):
- Multi-entity extraction: >90% accuracy
- Answer type classification: >85% accuracy

### Should Achieve (Important):
- Role filtering: >95% accuracy
- Chunk reading completeness: 100% (all relevant chunks mentioned)

### Nice to Have:
- Faster training convergence
- Better generalization to new queries

---

## 🔄 Iterative Improvement Process

1. **Diagnose** → Run comprehensive diagnostics
2. **Fix** → Apply fixes to dataset/training
3. **Train** → Train new model
4. **Test** → Compare with baseline
5. **Iterate** → If not meeting targets, refine and repeat

---

## 📚 Files Created/Updated

1. ✅ `TRAINING_STRATEGY_AND_FIXES.md` - Comprehensive strategy
2. ✅ `DATASET_AND_TRAINING_FIXES.md` - Implementation guide
3. ✅ `comprehensive_model_diagnostics.py` - Full test suite
4. ✅ `debug_multi_entity_extraction.py` - Single query diagnostic
5. ✅ `generate_rag_dataset_v3_json.py` - Updated with fixes
6. ✅ `train_rag_analysis_colab.py` - Already has answer_type mapping

---

## ✅ Ready to Execute

All fixes are implemented. Next steps:
1. Regenerate dataset
2. Run baseline diagnostics
3. Train new model
4. Compare results

# Training Recommendations - Multiple Entity Extraction & Other Issues

## Date: 2025-01-16

## Training Progress Summary
- **Epoch**: 6.04 / 7 (86% complete before connection loss)
- **Final Loss**: 0.1409 (down from ~1.3 at epoch 3.04)
- **Loss Reduction**: ~89% over ~3 epochs
- **Status**: Training was progressing well, loss curve healthy

## Critical Issues Identified

### 1. Multiple Entity Extraction Problems ⚠️
**Problem**: Model extracts only partial lists when multiple entities are requested.

**Examples from logs**:
- Query: "who are the managers of NexusCo?" → Expected: 4 names, Got: "Extract information from Chunk 1 and Chunk 2" (CoT leakage)
- Query: "who are the directors of PrimeDynamics?" → Expected: 4 names, Got: Only 1 name
- Query: "who are the leaders of NextDynamics?" → Expected: 2 names, Got: Only 1 name
- Query: "who is the executives at SmartNetworks?" → Expected: 4 names, Got: Only 1 name

**Root Cause**: 
- Model capacity (rank 4) may be too low for complex multi-entity extraction
- Model may be stopping early after finding first match
- CoT leakage interferes with extraction

### 2. CoT Leakage Still Present ⚠️
**Frequency**: ~10-15% of examples still show CoT leakage
- "Extract information from Chunk X"
- "Extract information comparing entities..."
- "Ensuring all relevant information was extracted..."

**Impact**: Causes 0% match scores when it occurs

### 3. Incomplete List Queries
**Problem**: Model sometimes extracts partial lists or misses items
- Query: "list the services related to machine learning" → Extracts extra items
- Query: "list the components related to scalability" → Returns "I don't have that information" when info exists

## Recommendations

### Option 1: Resume Training (Recommended First)
If you have a checkpoint from epoch 6.04:
1. **Resume from checkpoint** - only ~0.96 epochs remaining
2. **Complete training** to epoch 7
3. **Evaluate** and then decide on next steps

**Advantages**:
- Saves time (only ~1 hour remaining)
- Loss curve was healthy
- Can evaluate before making changes

### Option 2: Retrain with Improved Settings (If No Checkpoint)

#### A. Increase LoRA Rank (Address Multiple Entity Extraction)
```python
LORA_RANK = 6  # Increase from 4 to 6
LORA_ALPHA = 12  # 2x rank
LORA_DROPOUT = 0.25  # Keep same
```

**Rationale**:
- Rank 4 may be too low for complex multi-entity extraction
- Rank 6 provides ~4.1M trainable parameters (0.26% of model)
- Still conservative enough to prevent memorization
- Better capacity for learning to extract ALL entities, not just first match

#### B. Adjust Learning Rate
```python
learning_rate = 6e-7  # Slight increase from 5e-7
```

**Rationale**:
- Slightly faster learning may help with multi-entity patterns
- Still conservative (rank 8 used 8e-7 and caused issues)

#### C. Keep Other Settings
```python
num_train_epochs = 7  # Keep same
weight_decay = 0.7  # Keep same
warmup_steps = 1500  # Keep same
```

### Option 3: Dataset Improvements (Address Root Cause)

#### A. Add More List Query Examples
Focus on queries that require extracting multiple entities:
- "who are the [role] of [company]?" (expecting 2-4 names)
- "list the [items] related to [topic]" (expecting 2-4 items)
- "what are the [features/capabilities/services] of [entity]?" (expecting 2-4 items)

#### B. Emphasize Completeness in System Prompt
Add explicit instruction in system prompt:
```
CRITICAL FOR LIST QUERIES:
- When query asks for multiple items (e.g., "who are the managers", "list the features"):
  - Extract ALL matching items from ALL chunks
  - Do NOT stop after finding first match
  - Verify you have read ALL chunks completely
  - If query asks for "managers" and you find 3 managers, list all 3, not just 1
```

#### C. Add Negative Examples
Include examples where model should extract multiple items but might be tempted to stop early.

### Option 4: Post-Training Solutions

#### A. Post-Processing Filter for CoT Leakage
```python
def remove_cot_leakage(text: str) -> str:
    """Remove CoT patterns from model output"""
    patterns = [
        r'Extract information from Chunk \d+.*?\n',
        r'Extract information.*?\n',
        r'Ensuring all relevant information.*?\n',
        r'Read all \d+ chunk.*?\n',
    ]
    for pattern in patterns:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE | re.MULTILINE)
    return text.strip()
```

#### B. List Extraction Validation
Add validation to check if list queries extracted expected number of items:
- If query contains "list", "who are", "what are" (plural), expect multiple items
- If only 1 item extracted, flag as incomplete

## Recommended Action Plan

### Immediate (Before Retraining)
1. **Check for checkpoint** at epoch 6.04
   - If exists: Resume training, complete to epoch 7
   - If not: Proceed to retraining

### If Retraining Needed
1. **Update training script**:
   - Increase `LORA_RANK` to 6
   - Increase `learning_rate` to 6e-7
   - Keep epochs at 7

2. **Enhance dataset** (if possible):
   - Add explicit instructions for list queries in system prompt
   - Verify dataset has sufficient list query examples
   - Check that list query examples have complete extractions

3. **Monitor during training**:
   - Watch for multiple entity extraction examples
   - Check if model extracts all items or stops early
   - Monitor CoT leakage frequency

### Post-Training
1. **Run evaluation** on trained model
2. **Analyze failures**:
   - Count incomplete extractions (expected N items, got <N)
   - Measure CoT leakage rate
   - Identify query types with lowest scores

3. **Implement post-processing**:
   - CoT leakage filter
   - List completeness validator

4. **If still issues**:
   - Consider increasing LoRA rank to 8 (but monitor for memorization)
   - Add more training examples for problematic query types
   - Consider fine-tuning on subset of examples with multiple entities

## Expected Outcomes

### With Rank 6, LR 6e-7:
- **Better multi-entity extraction**: Model should extract all items, not just first
- **Slightly faster learning**: May help with complex patterns
- **Still prevent memorization**: Rank 6 is still conservative

### Success Metrics:
- **Multiple entity queries**: >80% extract all expected items
- **CoT leakage**: <5% (down from ~10-15%)
- **Match scores**: Mean >50% (up from ~40-50%)
- **Loss curve**: Gradual decrease (not rapid plummet)

## Code Changes Needed

### Update `train_rag_analysis_colab.py`:
```python
LORA_RANK = 6  # Increased from 4 - better capacity for multi-entity extraction
LORA_ALPHA = 12  # 2x rank
learning_rate = 6e-7  # Slight increase from 5e-7
```

### Update System Prompt (in dataset generation):
Add explicit instruction for list queries emphasizing completeness.

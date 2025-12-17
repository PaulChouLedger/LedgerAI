# Retraining Guide - Multiple Entity Extraction Fix

## Date: 2025-01-16

## Training Status
- **Previous Run**: Reached epoch 6.04/7 (86% complete) before connection loss
- **Final Loss**: 0.1409 (healthy progression from 1.3)
- **Issue**: Multiple entity extraction problems identified

## Critical Issues

### 1. Multiple Entity Extraction Failure ⚠️
**Problem**: Model extracts only 1 entity when multiple are expected.

**Examples**:
- "who are the managers of NexusCo?" → Expected: 4 names, Got: Only CoT leakage
- "who are the directors of PrimeDynamics?" → Expected: 4 names, Got: Only 1 name
- "who is the executives at SmartNetworks?" → Expected: 4 names, Got: Only 1 name

**Root Cause**: Rank 4 too low for complex multi-entity extraction patterns.

### 2. CoT Leakage (~10-15%)
Still present, causing 0% match scores when it occurs.

### 3. Incomplete List Queries
Model sometimes extracts partial lists or misses items entirely.

## Updated Training Configuration

### Changes Made to `train_rag_analysis_colab.py`:

```python
LORA_RANK = 6  # Increased from 4
LORA_ALPHA = 12  # 2x rank
LORA_DROPOUT = 0.25  # Keep same
learning_rate = 6e-7  # Increased from 5e-7
num_train_epochs = 7  # Keep same
```

### Rationale:
- **Rank 6**: ~4.1M trainable parameters (0.26% of model)
  - Better capacity for multi-entity extraction
  - Still conservative (rank 8 caused memorization)
  - Should learn to extract ALL items, not just first match

- **LR 6e-7**: Slightly faster learning
  - Helps with complex multi-entity patterns
  - Still conservative (rank 8 used 8e-7 and caused issues)

## Expected Improvements

### With Rank 6:
1. **Better Multi-Entity Extraction**
   - Model should extract all expected items
   - Less likely to stop after first match
   - Better pattern recognition for list queries

2. **Still Prevent Memorization**
   - Rank 6 is still conservative
   - Loss should decrease gradually (not plummet)
   - Better generalization than rank 8

3. **CoT Leakage**
   - May decrease slightly with better capacity
   - Post-processing filter still recommended

## Training Instructions

### Option 1: Resume from Checkpoint (If Available)
If you have a checkpoint from epoch 6.04:
```python
# In Colab, modify training script to resume:
from transformers import Trainer

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    # ... other args
)

# Resume from checkpoint
trainer.train(resume_from_checkpoint="./unsloth_output/checkpoint-XXXX")
```

### Option 2: Fresh Training (Recommended)
1. **Upload updated `train_rag_analysis_colab.py`** to Colab
2. **Verify dataset** is still correct (no CoT leakage)
3. **Start training**:
   ```python
   !python train_rag_analysis_colab.py
   ```

## Monitoring During Training

### Watch For:
1. **Loss Curve**:
   - Should decrease gradually (~0.1-0.2 per epoch)
   - Final loss around 0.10-0.15 at epoch 7
   - **Stop if**: Loss drops >95% in first 2 epochs (memorization)

2. **Multi-Entity Examples**:
   - Check training monitor outputs
   - Verify model extracts ALL items, not just first
   - Look for "who are the [role]" queries

3. **CoT Leakage**:
   - Should decrease to <10%
   - Monitor for "Extract information from Chunk X" patterns

## Post-Training Evaluation

### Run Evaluation:
```python
!python evaluate_trained_model_colab.py
```

### Check Specifically:
1. **Multiple Entity Queries**:
   - "who are the [role] of [company]?"
   - "list the [items] related to [topic]"
   - Verify all expected items are extracted

2. **CoT Leakage Rate**:
   - Should be <10% (ideally <5%)
   - If still high, implement post-processing filter

3. **Match Scores**:
   - Mean should be >50%
   - Multiple entity queries should score >70%

## If Issues Persist

### If Multi-Entity Extraction Still Fails:
1. **Increase LoRA rank to 8** (but monitor closely for memorization)
2. **Add more training examples** specifically for list queries
3. **Enhance system prompt** with explicit multi-entity instructions

### If CoT Leakage Persists:
1. **Implement post-processing filter** (see `TRAINING_RECOMMENDATIONS.md`)
2. **Add negative examples** to training data
3. **Increase LoRA dropout** to 0.30

### If Loss Still Plummets:
1. **Reduce learning rate** to 5e-7 or 4e-7
2. **Reduce LoRA rank** back to 4
3. **Increase weight_decay** to 0.8

## Success Criteria

Training is successful if:
- ✅ Loss decreases gradually (not >95% in <2 epochs)
- ✅ Multi-entity queries extract ALL expected items (>80% success rate)
- ✅ CoT leakage <10%
- ✅ Mean match score >50%
- ✅ Model generalizes (not just memorizing)

## Next Steps After Training

1. **Evaluate** trained model
2. **Analyze failures** by query type
3. **Implement post-processing** if needed
4. **Fine-tune** on problematic examples if necessary


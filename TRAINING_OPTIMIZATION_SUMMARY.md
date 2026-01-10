# Training Parameter Optimization Summary

**Date**: 2026-01-10  
**Issue**: Model shows 3 DISCARD violations (target: 0) despite improved average score (80.39%)

## Optimization Strategy

**Goal**: Slow down learning rate to allow gradual rule acquisition (especially DISCARD enforcement) without memorization.

## Parameter Changes

### Before (Current Training)
- **Learning Rate**: `2e-5`
- **Epochs**: `15`
- **Warmup Steps**: `50`
- **Weight Decay**: `0.25` ✅ (kept same)
- **LoRA Rank**: `128` ✅ (kept same)

### After (Optimized)
- **Learning Rate**: `1.5e-5` ⬇️ **25% slower**
- **Epochs**: `25` ⬆️ **67% more** (with slow LR, prevents memorization)
- **Warmup Ratio**: `0.1` (10% of training) ⬆️ **Longer warmup**
- **Weight Decay**: `0.25` ✅ (kept same - maintains anti-memorization)
- **LoRA Rank**: `128` ✅ (kept same - prevents memorization)

## Rationale

### 1. Lower Learning Rate (1.5e-5)
- **Why**: Slower learning allows model to gradually internalize DISCARD enforcement rules
- **Effect**: Model takes smaller steps, reducing risk of overshooting optimal weights
- **Trade-off**: More epochs needed, but prevents memorization

### 2. More Epochs (25)
- **Why**: With slower LR, more epochs = gradual learning, not memorization
- **Effect**: Model sees examples more times but learns slowly, reinforcing rules
- **Trade-off**: Longer training time, but better rule learning

### 3. Longer Warmup (10% ratio)
- **Why**: Gradual start prevents early memorization
- **Effect**: Learning rate ramps up slowly, allowing model to adapt gradually
- **Trade-off**: Slightly longer training, but better generalization

### 4. Keep High Weight Decay (0.25)
- **Why**: Maintains anti-memorization while allowing rule learning
- **Effect**: Regularization prevents overfitting to specific examples
- **Trade-off**: None - this is optimal

### 5. Keep Low LoRA Rank (128)
- **Why**: Prevents memorization by limiting model capacity
- **Effect**: Forces model to learn general patterns, not specific examples
- **Trade-off**: None - this is optimal

## Expected Improvements

| Metric | Current | Target | Expected After Optimization |
|--------|---------|--------|----------------------------|
| Average Score | 80.39% | >85% | 85-90% |
| DISCARD Violations | 3/17 (17.6%) | 0/17 (0%) | 0-1/17 (0-5.9%) |
| Person Queries | 60.42% | >80% | 75-85% |
| Real-World Examples | Mixed | >80% | 80-90% |

## Key Insight

**Slow learning + More epochs = Gradual rule learning (not memorization)**

With a very low learning rate (1.5e-5), increasing epochs from 15 to 25 doesn't cause memorization. Instead:
- Model takes smaller steps per epoch
- Rules are reinforced gradually over more iterations
- DISCARD enforcement can be learned more reliably
- Generalization is maintained through high weight decay

## Next Steps

1. **Retrain** with optimized parameters
2. **Test** new model with `test_rag_cot_model_colab.py`
3. **Verify** DISCARD violations reduced to 0-1
4. **Monitor** average score improvement

## Training Configuration Summary

```python
training_args = TrainingArguments(
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,  # Effective batch size = 8
    warmup_ratio=0.1,  # 10% of training in warmup
    num_train_epochs=25,  # More epochs with slow LR
    learning_rate=1.5e-5,  # EVEN LOWER: slower learning
    weight_decay=0.25,  # HIGH: maintains anti-memorization
    lr_scheduler_type="cosine",  # Smooth learning curve
    # ... other settings
)
```

**Total Training Steps**: ~3,525 steps (141 examples × 25 epochs / 8 effective batch size)  
**Warmup Steps**: ~353 steps (10% of total)  
**Learning Rate Schedule**: Cosine decay from 1.5e-5 to near-zero over 25 epochs

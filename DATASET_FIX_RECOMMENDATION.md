# Dataset Fix Recommendation

**Date**: 2026-01-10  
**Issue**: 30 training examples have potential DISCARD violations

## Problem Identified

The training dataset has **30 examples** where items marked `[DISCARD]` in REASONING may still appear in FINAL ANSWER. This teaches the model to violate the DISCARD rule.

## Root Cause

**Not a training configuration issue** - the anti-memorization settings are correct.  
**The issue is dataset quality** - the training examples themselves violate the rule they're meant to teach.

## Solution Approach

### Option 1: Fix Dataset First (Recommended)

1. **Review and fix all 30 violations manually**
   - Identify true violations vs. false positives (contextual mentions)
   - Remove DISCARD items from FINAL ANSWER
   - Ensure FINAL ANSWER only includes [KEEP] items

2. **Strengthen system prompt DISCARD emphasis**
   - Current: 4 mentions of "discard"
   - Target: 8-10 mentions with stronger language
   - Add explicit examples in system prompt

3. **Add more explicit DISCARD examples**
   - Current: 6 examples with empty/minimal FINAL ANSWER (all DISCARD)
   - Target: 15-20 examples showing proper DISCARD enforcement

### Option 2: Slower Learning (Without Memorization)

If dataset is too large to fix quickly, consider **even slower learning**:

```python
# In train_rag_cot_colab.py - KEEP anti-memorization settings
num_train_epochs=25,        # More epochs (was 15)
learning_rate=1.5e-5,       # Even LOWER (was 2e-5) - slower learning
weight_decay=0.25,          # Keep same (HIGH - anti-memorization)
warmup_steps=100,           # Longer warmup (was 50)
warmup_ratio=0.1,           # 10% of training in warmup
lr_scheduler_type="cosine", # Keep cosine decay
```

**Rationale**:
- **Lower LR (1.5e-5)**: Even slower learning prevents memorization
- **More epochs (25)**: With slow LR, more epochs = more gradual learning (not memorization)
- **Longer warmup**: More gradual start = better generalization
- **Keep high weight decay**: Maintains anti-memorization

### Option 3: Hybrid Approach (Best)

1. **Fix critical violations** (top 10-15 examples with clearest violations)
2. **Retrain with slower learning rate** (1.5e-5, 25 epochs)
3. **Test and iterate**

## Recommendations

### Immediate Actions

1. **Fix Dataset Violations** (Priority 1)
   - Review the 30 examples identified
   - Fix at least the most egregious violations (10-15 examples)
   - Ensure "No co-founders" examples have minimal FINAL ANSWER

2. **Strengthen System Prompt** (Priority 2)
   - Increase "discard" mentions from 4 to 8-10
   - Add explicit example: "If REASONING has 'Item: X - Action: [DISCARD]', X must NOT appear in FINAL ANSWER"
   - Emphasize "NEVER appear" multiple times

3. **Adjust Training for Slower Learning** (Priority 3)
   - Lower LR to 1.5e-5 or 1e-5
   - Increase epochs to 25-30 (with slow LR, this prevents memorization)
   - Longer warmup (100 steps or 10% ratio)

### Expected Improvements

| Metric | Current | After Fix | Target |
|--------|---------|-----------|--------|
| Dataset Violations | 30 | <5 | 0 |
| DISCARD Violations (Test) | 2/17 (11.8%) | 0-1/17 | 0/17 |
| Average Score | 74.75% | 80-85% | >85% |

## Conclusion

**Keep anti-memorization settings** - they're correct.

**Fix dataset quality first** - the training examples are teaching the wrong behavior.

**Use slower learning** - lower LR (1.5e-5) with more epochs (25) prevents memorization while allowing gradual rule learning.

The issue is not training speed, but dataset quality. Once dataset is fixed, slower learning (lower LR) will help the model learn the DISCARD rule more reliably without memorization.

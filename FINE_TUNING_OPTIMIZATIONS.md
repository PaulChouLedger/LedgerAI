# Fine-Tuning Optimizations for Better Results

## 1. JSON Output Format ✅ (Implemented)

**Why it helps:**
- Structured format is easier for model to learn
- Forces model to extract ALL items (array must be complete)
- Reduces ambiguity (clear structure vs. natural language)
- Post-processing can convert to natural language

**Implementation:**
- Dataset generator: `generate_rag_dataset_v3_json.py`
- Output format: `{"answer_type": "...", "items": [...], "text": "...", "chunks_used": [...]}`

## 2. Simplified System Prompt ✅ (Implemented)

**Why it helps:**
- Removes CoT complexity that confuses small models
- Focuses on core task: extract → JSON
- Shorter prompt = less confusion, better learning

**Changes:**
- Removed 6-step CoT instructions
- Added clear JSON format examples
- Emphasized "extract ALL items" requirement

## 3. Additional Training Optimizations

### A. Loss Function Modifications

**Current:** Standard cross-entropy loss

**Optimizations to try:**

1. **Focal Loss for Imbalanced Classes**
   - Penalize "I don't have information" responses less (they're common)
   - Focus learning on actual extractions
   ```python
   # In training script
   from torch.nn import functional as F
   
   def focal_loss(pred, target, alpha=0.25, gamma=2.0):
       ce_loss = F.cross_entropy(pred, target, reduction='none')
       pt = torch.exp(-ce_loss)
       focal_loss = alpha * (1 - pt) ** gamma * ce_loss
       return focal_loss.mean()
   ```

2. **Token-Level Weighting**
   - Weight important tokens (entity names, JSON structure) higher
   - Reduce weight on common words ("the", "and", etc.)

3. **Completeness Loss**
   - Add penalty if model outputs partial extraction
   - Reward complete extractions (all entities found)

### B. Training Strategy Improvements

1. **Curriculum Learning**
   - Start with simple examples (1 entity, 1 chunk)
   - Gradually increase complexity (multiple entities, multiple chunks)
   - Helps model learn incrementally

2. **Hard Negative Mining**
   - Focus training on examples model struggles with
   - Re-sample failed examples more frequently

3. **Data Augmentation**
   - Paraphrase queries (same meaning, different wording)
   - Shuffle chunk order (teach model order doesn't matter)
   - Add noise to irrelevant chunks (teach model to ignore)

### C. Model Architecture Tweaks

1. **LoRA Configuration**
   ```python
   # Current: rank=6, alpha=12, dropout=0.25
   # Try:
   LORA_RANK = 8  # Slightly higher capacity
   LORA_ALPHA = 16  # 2x rank
   LORA_DROPOUT = 0.3  # More regularization
   ```

2. **Target Modules**
   - Current: All attention + MLP layers
   - Try: Only attention layers (q_proj, k_proj, v_proj, o_proj)
   - Or: Add embedding layers for better token understanding

3. **Gradient Accumulation**
   - Accumulate gradients over multiple batches
   - Simulates larger batch size without memory increase
   ```python
   gradient_accumulation_steps = 4  # Process 4 batches before update
   ```

### D. Learning Rate Schedule

**Current:** Linear warmup + decay

**Optimizations:**

1. **Cosine Annealing with Restarts**
   - Helps escape local minima
   - Better for fine-tuning

2. **OneCycle Policy**
   - Fast learning rate increase, then decrease
   - Often better for small datasets

3. **Differential Learning Rates**
   - Higher LR for LoRA layers
   - Lower LR for base model (if unfreezing)

### E. Regularization Enhancements

1. **Label Smoothing**
   - Prevents overconfidence
   - Better generalization
   ```python
   label_smoothing_factor = 0.1  # 10% smoothing
   ```

2. **Dropout Variations**
   - Increase dropout in attention layers
   - Add dropout to embeddings

3. **Weight Decay**
   - Current: 0.7 (already high)
   - Try: 0.8-1.0 for stronger regularization

## 4. Evaluation and Monitoring

### A. Custom Metrics

1. **Extraction Completeness Score**
   - Measure: % of expected entities actually extracted
   - Target: >90% for multi-entity queries

2. **JSON Validity Rate**
   - Measure: % of outputs that are valid JSON
   - Target: 100%

3. **Chunk Coverage**
   - Measure: % of relevant chunks actually used
   - Target: 100% (all relevant chunks should be used)

### B. Early Stopping

```python
# Stop if validation loss doesn't improve
early_stopping_patience = 3  # epochs
# Stop if match score plateaus
match_score_patience = 5  # epochs
```

## 5. Post-Processing Pipeline

**Convert JSON to Natural Language:**

```python
def json_to_natural_language(json_output: str) -> str:
    """Convert JSON output to natural language for user display"""
    try:
        data = json.loads(json_output)
        
        if data["answer_type"] == "not_found":
            return data["text"]
        
        if data["answer_type"] in ["entities", "list"]:
            items = data["items"]
            if len(items) == 0:
                return "I don't have that information in the provided documents"
            elif len(items) == 1:
                return items[0]
            elif len(items) == 2:
                return f"{items[0]} and {items[1]}"
            else:
                return ", ".join(items[:-1]) + f", and {items[-1]}"
        
        elif data["answer_type"] in ["comparison", "analytical", "relationship", "process"]:
            return data["text"]
        
        return json_output  # Fallback to raw JSON
    except:
        return json_output  # If JSON parsing fails, return as-is
```

## 6. Recommended Training Configuration

```python
# LoRA Configuration
LORA_RANK = 8  # Increased from 6
LORA_ALPHA = 16  # 2x rank
LORA_DROPOUT = 0.3  # Increased regularization

# Training Arguments
num_train_epochs = 5  # Reduced from 7 (prevent overfitting)
learning_rate = 5e-7  # Slightly lower (more conservative)
weight_decay = 0.8  # Increased regularization
warmup_steps = 2000  # More warmup
gradient_accumulation_steps = 4  # Simulate larger batch
label_smoothing_factor = 0.1  # Prevent overconfidence

# Evaluation
eval_strategy = "steps"
eval_steps = 500  # Evaluate every 500 steps
save_strategy = "steps"
save_steps = 500
load_best_model_at_end = True
metric_for_best_model = "match_score"  # Custom metric
```

## 7. Implementation Priority

**High Priority (Do First):**
1. ✅ JSON output format (done)
2. ✅ Simplified system prompt (done)
3. Curriculum learning
4. Custom completeness loss

**Medium Priority:**
5. Focal loss
6. Gradient accumulation
7. Label smoothing
8. Better evaluation metrics

**Low Priority (If Above Don't Work):**
9. Architecture changes
10. Advanced LR schedules
11. Data augmentation

## 8. Expected Improvements

**With JSON Format + Simplified Prompt:**
- Extraction completeness: 25% → 70-80%
- JSON validity: N/A → 95%+
- Match scores: 12% → 50-60%

**With All Optimizations:**
- Extraction completeness: 70-80% → 85-90%
- Match scores: 50-60% → 70-80%
- CoT leakage: 23% → <5%

## 9. Next Steps

1. **Generate JSON dataset**: Run `generate_rag_dataset_v3_json.py`
2. **Update training script**: Add JSON parsing, custom metrics
3. **Train with optimizations**: Use recommended config
4. **Evaluate**: Check extraction completeness, JSON validity
5. **Post-process**: Convert JSON to natural language for display

# CoT Toggle Training Pipeline

This pipeline trains the Qwen2.5-1.5B model to conditionally use Chain of Thought (CoT) reasoning:
- **WITH CoT** when RAG context is provided (uses CoT system prompt)
- **WITHOUT CoT** for conversational queries (uses conversational system prompt)

## Overview

The model learns to toggle CoT behavior based on the system prompt:
- **CoT System Prompt** → Model generates REASONING: ... FINAL ANSWER: format
- **Conversational System Prompt** → Model generates natural, direct responses

## Pipeline Steps

### 1. Generate Conversational Dataset

Creates conversational examples without CoT:

```bash
python3 create_conversational_dataset.py
```

This generates `conversational_dataset.json` with ~2000 examples of:
- Recipes
- General knowledge questions
- Greetings and small talk
- Natural conversational responses

### 2. Merge Datasets

Merges RAG+CoT dataset with conversational dataset:

```bash
python3 merge_cot_toggle_dataset.py
```

This creates `rag_cot_toggle_training_dataset.json` with:
- 50% RAG+CoT examples (with CoT system prompt)
- 50% Conversational examples (without CoT system prompt)

### 3. Train Model

Trains the model on the mixed dataset:

```bash
# In Google Colab or local environment with GPU
python3 train_cot_toggle_colab.py
```

**Requirements:**
- GPU (CUDA) recommended
- Install dependencies: `pip install unsloth trl peft accelerate bitsandbytes datasets`

**Training Configuration:**
- Model: Qwen2.5-1.5B-Instruct
- LoRA Rank: 128 (prevents memorization)
- Epochs: 3
- Learning Rate: 2e-4
- Batch Size: 2 (with gradient accumulation: 4)

**Outputs:**
- `outputs_cot_toggle/` - LoRA adapters
- `outputs_cot_toggle_merged/` - Merged model (16-bit)
- `gguf_model_cot_toggle/` - GGUF format (q4_k_m quantization)

## Dataset Structure

### RAG+CoT Examples (with CoT)
```json
{
  "messages": [
    {
      "role": "system",
      "content": "You are a precise data extraction bot.\n1. Start with REASONING:..."
    },
    {
      "role": "user",
      "content": "Knowledge context: ...\n---\nQuestion: Who are the co-founders?"
    },
    {
      "role": "assistant",
      "content": "REASONING:\n- Item: ...\nFINAL ANSWER: ..."
    }
  ]
}
```

### Conversational Examples (without CoT)
```json
{
  "messages": [
    {
      "role": "system",
      "content": "You are Aura Vision, an AI agent..."
    },
    {
      "role": "user",
      "content": "Give me a recipe for cooked chicken."
    },
    {
      "role": "assistant",
      "content": "Here's a simple recipe: Season chicken with salt and pepper..."
    }
  ]
}
```

## Expected Behavior After Training

### With RAG (CoT System Prompt)
**Input:**
- System: CoT extraction bot prompt
- User: "Knowledge context: ...\n---\nQuestion: Who are the co-founders?"

**Output:**
```
REASONING:
- Item: Person A
  - Evidence: "..."
  - Action: [KEEP]
- End of scan.

FINAL ANSWER:
The co-founders are Person A and Person B.
```

### Without RAG (Conversational System Prompt)
**Input:**
- System: Aura Vision conversational prompt
- User: "Give me a recipe for cooked chicken."

**Output:**
```
Here's a simple recipe: Season chicken with salt and pepper, then cook in a pan over medium heat for 6-8 minutes per side until golden and cooked through. Would you like more details?
```

## Deployment

After training, update `llm-container/container_rest.py` to:
1. Use CoT system prompt when `has_rag_context = True`
2. Use conversational system prompt when `has_rag_context = False`
3. Deploy the new GGUF model to `/models/`

## Testing

Use the dedicated test script to verify CoT toggle behavior:

```bash
# In Colab or local environment
python3 test_cot_toggle_colab.py
```

The test script verifies:
1. **RAG Scenarios** (should use CoT):
   - "Who are the co-founders of Ledger AI?"
   - Should generate REASONING: ... FINAL ANSWER: format
   - Should extract from RAG context correctly

2. **Conversational Scenarios** (should NOT use CoT):
   - "Give me a recipe for cooked chicken."
   - "What is the capital of France?"
   - Should respond naturally without CoT format

**Test Output:**
- CoT behavior correctness (should match expected mode)
- RAG extraction accuracy (for RAG scenarios)
- Natural response quality (for conversational scenarios)

## Troubleshooting

### Model Still Uses CoT for Conversational Queries
- Check that system prompt is correctly set (no CoT instructions)
- Verify training dataset has sufficient conversational examples
- May need more training epochs or adjust learning rate

### Model Doesn't Use CoT for RAG Queries
- Check that CoT system prompt is being used
- Verify RAG context format matches training: "Knowledge context: ...\n---\nQuestion: ..."
- May need to retrain with more RAG examples

### Imbalanced Dataset
Current dataset has ~135 RAG examples vs 2000 conversational. This is acceptable, but ideally:
- Add more RAG examples to balance (target: 50/50 or 60/40)
- Or adjust merge ratio in `merge_cot_toggle_dataset.py`

## Files

- `create_conversational_dataset.py` - Generates conversational examples
- `merge_cot_toggle_dataset.py` - Merges datasets
- `train_cot_toggle_colab.py` - Training script
- `test_cot_toggle_colab.py` - Test script for CoT toggle behavior
- `rag_cot_toggle_training_dataset.json` - Final merged dataset
- `conversational_dataset.json` - Conversational examples only

## Next Steps

1. ✅ Generate conversational dataset
2. ✅ Merge with RAG+CoT dataset
3. ✅ Create training script
4. ✅ Create test script
5. ⏳ Train model (run in Colab or local GPU)
6. ⏳ Test model behavior with `test_cot_toggle_colab.py`
7. ⏳ Deploy to production
8. ✅ Update llm-container to use conditional CoT (already done)

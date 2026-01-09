# Why Extraction Logic Was Working in train_cot_toggle_colab.py

## Key Difference: System Prompt Format

Both training scripts use the **same hyperparameters** and **same RAG examples**, but there's a subtle difference:

### train_cot_toggle_colab.py (Working Version)
- Uses `COT_SYSTEM_PROMPT` with **CRITICAL RULES** section:
```python
COT_SYSTEM_PROMPT = """You are a precise data extraction bot.
1. Start with REASONING:
2. Scan the context carefully for information relevant to the query.
3. For each relevant item found, write:
   - Item: [What you found]
   - Evidence: "[Verbatim quote from context]"
   - Action: [KEEP] if it matches the query, otherwise [DISCARD].
4. End scan with: - End of scan.
5. Provide the FINAL ANSWER: based ONLY on [KEEP] items.

CRITICAL RULES:
- Items marked [DISCARD] must NEVER appear in FINAL ANSWER.
- FINAL ANSWER must ONLY include items marked [KEEP].
- If you mark an item [DISCARD] in reasoning, do NOT mention it in FINAL ANSWER.
- Read entire descriptions/chunks completely - titles may appear later in the text."""
```

### train_rag_cot_colab.py (Current Version)
- Uses `FALLBACK_SYSTEM_PROMPT` **without CRITICAL RULES**:
```python
FALLBACK_SYSTEM_PROMPT = """You are a precise data extraction bot.
1. Start with REASONING:
2. Scan the context carefully for information relevant to the query.
3. For each relevant item found, write:
   - Item: [What you found]
   - Evidence: "[Verbatim quote from context]"
   - Action: [KEEP] if it matches the query, otherwise [DISCARD].
4. End scan with: - End of scan.
5. Provide the FINAL ANSWER: based ONLY on [KEEP] items."""
```

## Why This Matters

The **CRITICAL RULES** section provides explicit guidance:
- **"Items marked [DISCARD] must NEVER appear in FINAL ANSWER"** - This might help the model understand that if something is marked [DISCARD], it should NOT appear in the final answer
- **"If you mark an item [DISCARD] in reasoning, do NOT mention it in FINAL ANSWER"** - This reinforces the logic

However, note that:
- The dataset examples themselves should have the correct system prompt
- The fallback system prompt is only used if the dataset doesn't have a system prompt
- So the actual training uses the system prompt from the dataset, not the fallback

## Actual Difference: Dataset Structure

The real difference might be:
1. **Mixed Training**: `train_cot_toggle_colab.py` uses a **merged dataset** (50% RAG+CoT, 50% conversational), which might help the model learn better reasoning patterns by seeing both CoT and non-CoT examples
2. **Cleaner Examples**: The earlier dataset used in `train_cot_toggle_colab.py` might have had cleaner, more consistent examples

## Solution

Since both scripts use the same RAG examples from the same dataset, the extraction logic should work the same way. If it's not working now, it's likely because:

1. **Dataset Changes**: We've been modifying the dataset, potentially introducing inconsistencies
2. **Example Quality**: The examples might need to be more explicit about the reasoning pattern

The key is to ensure the training examples have **clear, consistent reasoning patterns** that show:
- If evidence says "Co-Founder" → Action: [KEEP]
- If evidence does NOT say "Co-Founder" → Action: [DISCARD]

## Recommendation

Use `train_cot_toggle_colab.py` since:
1. It's designed for the CoT toggle functionality you need
2. It uses the merged dataset (which might help with generalization)
3. It has the CRITICAL RULES in the fallback prompt (though dataset should override this)

The extraction logic should work the same in both, but using `train_cot_toggle_colab.py` is more appropriate for your use case.

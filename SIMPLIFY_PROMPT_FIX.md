# Simplify System Prompt Fix

## Issue Identified

After adding many "CRITICAL RULES" to the system prompt, the model regressed to 0% accuracy and started hallucinating. The complex prompt with multiple rules may have confused the model.

## Solution: Restore Original Simple Prompt

The original prompt from `train_rag_cot_colab.py` (lines 54-62) that yielded accurate results is much simpler:

```
You are a precise data extraction bot.
1. Start with REASONING:
2. Scan the context carefully for information relevant to the query.
3. For each relevant item found, write:
   - Item: [What you found]
   - Evidence: "[Verbatim quote from context]"
   - Action: [KEEP] if it matches the query, otherwise [DISCARD].
4. End scan with: - End of scan.
5. Provide the FINAL ANSWER: based ONLY on [KEEP] items.
```

## Key Insights

1. **Simplicity works**: The original prompt is clear and concise
2. **Evidence instruction implies exact extraction**: "Evidence: '[Verbatim quote from context]'" already instructs exact extraction
3. **Less is more**: Adding too many "CRITICAL" rules may confuse the model
4. **Structured format**: The numbered steps provide clear structure without being overwhelming

## Changes Made

1. Restored all 152 system prompts to the simple version
2. Removed all "CRITICAL RULES" additions:
   - ❌ Removed: "CRITICAL: FINAL ANSWER must include ALL items marked [KEEP]"
   - ❌ Removed: "CRITICAL: Scan the ENTIRE context from start to finish"
   - ❌ Removed: "CRITICAL ANTI-HALLUCINATION: You MUST extract information EXACTLY..."
3. Kept the core structure that worked originally

## Expected Improvement

After retraining with the simple prompt:
- **Current**: 0% accuracy (hallucinating with complex prompt)
- **Target**: Back to 75-100% accuracy (like before with simple prompt)
- **Key fix**: Simpler prompt = less confusion = better results

## Next Steps

1. Retrain model with simplified prompt (152 RAG examples)
2. Test again to verify accuracy restored
3. If needed, make minimal adjustments (not adding many rules)

# Co-Founder Reasoning Logic Fix

## Critical Issue Identified

The model is DISCARDing items that explicitly say "Co-Founder" with nonsensical reasons:
- **Bob Carella**: Evidence: "As Co-Founder and Chief Financial Officer of LedgerAI" → Action: [DISCARD] (Reason: Co-founder of LedgerAI, not co-founder of LedgerAI). ❌ WRONG
- **Paul Chou**: Evidence: "As CEO and Co-Founder of LedgerAI" → Action: [DISCARD] (Reason: Co-founder of LedgerAI, not co-founder). ❌ WRONG
- **David Lara**: Evidence: "As Co-Founder and Chief Operating Officer of LedgerAI" → Action: [DISCARD] (Reason: CO-FOUNDER, not co-founder). ❌ WRONG (case sensitivity?)
- **Jorge Guinovart**: Evidence: "As Co-Founder and Chief Marketing Officer of LedgerAI" → Action: [DISCARD] (Reason: Co-founder, not initial co-founder). ❌ WRONG

The model is clearly confused about what "Co-Founder" means and is discarding items with nonsensical reasons.

## Root Cause

The training examples have correct logic (they KEEP items with "Co-Founder"), but the model may have learned incorrect patterns from:
1. Over-complicated reasoning in some examples
2. Inconsistent formatting (e.g., "Co-Founder" vs "co-founder" vs "CO-FOUNDER")
3. Not enough explicit examples showing: "If evidence says 'Co-Founder' → KEEP"

## Solution: Add Explicit Co-Founder Examples

Added 5 new clear examples that emphasize:
- **If evidence says "Co-Founder" → Action: [KEEP]** (no exceptions)
- **If evidence says "CEO and Co-Founder" → Action: [KEEP]**
- **If evidence says "Co-Founder and [Other Role]" → Action: [KEEP]**
- **Only discard if evidence does NOT say "Co-Founder"**

## Dataset Status

- **Total RAG+CoT examples**: 157 (152 original + 5 new clear Co-Founder examples)
- **All system prompts**: Simple version (restored from original that worked)
- **Merged dataset**: 285 total (157 RAG + 128 conversational)

## Expected Improvement

After retraining with these clear examples:
- **Current**: 50% accuracy (discarding co-founders with wrong reasons)
- **Target**: 75-100% accuracy (correctly KEEPing all co-founders)
- **Key fix**: Model will learn: "Co-Founder" in evidence = [KEEP]

## Next Steps

1. Retrain model with updated dataset (157 RAG examples)
2. Test again to verify co-founder reasoning is correct
3. Verify all 4 co-founders are found in LedgerAI test

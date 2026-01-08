# Extraction Accuracy Improvements

## Issues Identified

From test results, the model struggled with:
1. **Complex multi-chunk contexts** - LedgerAI test: 50% accuracy (vs TechCorp: 100%)
2. **Contradictory reasoning** - Marking "CO-founder" then DISCARDing it
3. **Name truncation** - "Jorge Guinovar" instead of "Jorge Guinovart"
4. **Missing co-founders** - Bob Carella and Jorge Guinovart not found
5. **Incorrect inclusions** - Albert Soler (External Counsel) included

## Improvements Made

### 1. Added Complex Multi-Chunk Examples

Created 5 new complex training examples that address the issues:

**Example 1: QuantumTech (Similar to LedgerAI test)**
- 3 long chunks with noise
- 4 co-founders mixed with 4 non-co-founders
- Tests full name handling and clear KEEP/DISCARD logic

**Example 2: TechFlow Innovations**
- Multiple chunks with name variations
- Tests extraction from different chunk positions

**Example 3: DataFlow Systems (Late-Title Pattern)**
- Very long descriptions with co-founder title appearing later
- Tests reading complete descriptions

**Example 4: CloudScale Technologies (Stress Test)**
- Multiple chunks with many non-co-founders
- Tests filtering logic with many distractors

**Example 5: InnovateAI Solutions (Late-Title Pattern)**
- Long single chunk with co-founder title late in description
- Tests thorough reading of long contexts

### 2. Updated Training Dataset

- **Before**: 135 RAG examples
- **After**: 140 RAG examples (135 original + 5 new complex)
- **Merged Dataset**: 280 total (140 RAG + 140 conversational)

### 3. Prioritized Complex Examples

The merge script now prioritizes complex examples (longer contexts, multiple chunks) to ensure they're included in training.

## Expected Improvements

After retraining with the improved dataset, the model should:

1. ✅ **Better handle long multi-chunk contexts** - More examples with 3+ chunks
2. ✅ **Avoid contradictory reasoning** - Clear KEEP logic for co-founders
3. ✅ **Handle full names correctly** - No truncation issues
4. ✅ **Find all co-founders** - Better at scanning entire contexts
5. ✅ **Correctly DISCARD non-co-founders** - Clear distinction between roles

## Training

The improved dataset is ready:
- **File**: `rag_cot_toggle_training_dataset.json`
- **Size**: 280 examples (140 RAG + 140 conversational)
- **Training Time**: ~1-1.5 hours (vs 5 hours with 2135 examples)

## Next Steps

1. Retrain the model with the improved dataset
2. Test again with the LedgerAI scenario
3. Expected improvement: 50% → 75-100% accuracy on complex contexts

## Files Modified

- `add_complex_extraction_examples.py` - Script to add complex examples
- `rag_cot_training_dataset.json` - Updated with 5 new complex examples
- `merge_cot_toggle_dataset.py` - Updated to prioritize complex examples
- `rag_cot_toggle_training_dataset.json` - Regenerated with 280 examples

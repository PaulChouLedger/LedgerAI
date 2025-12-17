# Chunk Display Feature

## What Was Added

The training monitor now displays **actual chunk text** so you can verify exactly what the model is seeing during training.

## Feature Details

### New Parameter: `show_chunks`
- **Default**: `True` (chunks are shown by default)
- **Purpose**: Display the actual chunk text that the model processes
- **Location**: `ExampleMonitorCallback.__init__()` and `create_example_monitor()`

### What Gets Displayed

For each chunk, you'll see:
1. **Chunk number and score**
2. **Full chunk text** (truncated to 400 chars if longer)
3. **Character count** (to see chunk size)
4. **Entity/item analysis** (what's found in the chunk)
5. **Expected items check** (whether expected items are in this chunk)

## Example Output

During training, you'll now see:

```
📊 TRAINING EXAMPLE MONITOR - Step 200

   Example 1 (Dataset Index 328):
   ────────────────────────────────────────────────────────────────────────────
   📋 Query: who are the executives of SmartSystems?
   ✅ Expected: {"answer_type": "entities", "items": ["Emery Hernandez", "Quinn Williams", "Logan Miller", "Jordan Jackson"], ...}
   
   🔍 Chunk Understanding Analysis:
      Chunk 1 (Score: 0.85) - Full Text (718 chars):
         "Emery Hernandez serves as executive at SmartSystems, leading strategic initiatives and overseeing key operations. Quinn Williams is executive at SmartSystems, responsible for driving innovation. Logan Miller holds the position of executive at SmartSystems, where they focus on expanding market presence. Jordan Jackson is executive at SmartSystems, with extensive experience in technology leadership..."
      
      Chunk 1: Found 4 entities: Emery Hernandez, Jordan Jackson, Logan Miller, Quinn Williams
            ✅ Contains 4/4 expected items: Emery Hernandez, Jordan Jackson, Logan Miller, Quinn Williams
      📊 Summary: 1 chunks, 4 expected items
   
   🤖 Model Output: ```json
   {
     "answer_type": "entities",
     "items": ["Logan Miller", "Logan Miller", "Logan Miller"],
     ...
   }
   
   🤖 Model's Understanding:
      Answer type: entities
      Extracted 3 items: Logan Miller, Logan Miller, Logan Miller
      ⚠️  Found 2 duplicate(s) - model may be repeating same entity
      Chunks used: [1]
      ❌ Missing 1 item(s) - incomplete extraction
```

## Benefits

1. **Verify Input**: See exactly what text the model is processing
2. **Debug Issues**: If model misses entities, check if they're actually in the chunk text
3. **Format Verification**: Ensure chunk formatting is correct (no parsing issues)
4. **Content Validation**: Confirm chunks contain expected information

## How to Use

### Default (Chunks Shown)
```python
example_monitor = create_example_monitor(
    dataset=train_dataset,
    tokenizer=tokenizer,
    model=model,
    show_chunks=True  # Default - chunks are shown
)
```

### Disable Chunk Display (Less Verbose)
```python
example_monitor = create_example_monitor(
    dataset=train_dataset,
    tokenizer=tokenizer,
    model=model,
    show_chunks=False  # Hide chunk text (only show analysis)
)
```

## What You Can Diagnose

### Scenario 1: Chunk Text Shows All Entities
**Chunk shows**: "Emery Hernandez... Quinn Williams... Logan Miller... Jordan Jackson..."
**Model extracts**: Only "Logan Miller" (repeated 3 times)
**Diagnosis**: Model is reading chunk but not processing all sentences - may be stopping early or not tracking multiple entities

### Scenario 2: Chunk Text Missing Entities
**Chunk shows**: Only "Logan Miller" mentioned
**Expected**: 4 executives
**Diagnosis**: Dataset issue - chunk doesn't contain all expected entities

### Scenario 3: Chunk Format Issues
**Chunk shows**: Malformed text or parsing errors
**Diagnosis**: Dataset generation issue - chunks not formatted correctly

### Scenario 4: Model Sees Different Text
**Chunk analysis shows**: 4 entities in chunk
**Model extracts**: Wrong entities or duplicates
**Diagnosis**: Model may be hallucinating or not understanding chunk content correctly

## Implementation

The chunk text is extracted from the training example using regex:
- Pattern: `[Chunk X] Score: Y.YY ... FULL CHUNK TEXT: '...'`
- Handles both single and double quotes
- Removes escape sequences
- Shows full text (truncated if >400 chars for readability)

## Files Modified

- ✅ `training_example_monitor.py` - Added `show_chunks` parameter and chunk text display
- ✅ `train_rag_analysis_colab.py` - Updated to pass `show_chunks=True`

## Next Steps

When training, you'll automatically see:
1. **Chunk text** - What the model is reading
2. **Chunk analysis** - What entities/items are in each chunk
3. **Model output** - What the model extracted
4. **Model understanding** - Analysis of model's extraction

This will help you identify exactly why the model is extracting duplicates or missing entities!

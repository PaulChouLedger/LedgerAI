# Chunk Understanding Diagnostics

## Problem Identified

During training, the model shows:
- **Example 2**: Extracting "Logan Miller" 3 times instead of all 4 executives
- **Issue**: Model appears to stop after finding first entity, or doesn't read chunks completely

## Solution: Enhanced Training Monitor

Added chunk understanding diagnostics to `training_example_monitor.py` that shows:

### 1. **Chunk Analysis** (What's Actually in Chunks)
For each chunk, shows:
- How many entities/items are found in the chunk
- Which entities/items are present
- Whether expected items are in this chunk

**Example Output:**
```
🔍 Chunk Understanding Analysis:
   Chunk 1: Found 4 entities: Emery Hernandez, Jordan Jackson, Logan Miller, Quinn Williams
         ✅ Contains 4/4 expected items: Emery Hernandez, Jordan Jackson, Logan Miller, Quinn Williams
   📊 Summary: 1 chunks, 4 expected items
```

### 2. **Model's Understanding** (What Model Extracted)
Shows:
- How many items the model extracted
- Which items were extracted
- Whether duplicates were found
- Which chunks the model claims to have used
- Comparison with expected items

**Example Output:**
```
🤖 Model's Understanding:
   Answer type: entities
   Extracted 3 items: Logan Miller, Logan Miller, Logan Miller
   ⚠️  Found 2 duplicate(s) - model may be repeating same entity
   Chunks used: [1]
   ❌ Missing 1 item(s) - incomplete extraction
```

## How It Helps Diagnose

### Scenario 1: Model Not Reading Chunks Completely
**Symptoms:**
- Chunk analysis shows 4 entities in chunk
- Model extracts only 1-2 entities
- Model's chunks_used shows correct chunk

**Diagnosis:** Model is reading chunk but stopping early or not processing all sentences

### Scenario 2: Model Not Understanding Chunk Content
**Symptoms:**
- Chunk analysis shows entities clearly present
- Model extracts wrong entities or duplicates
- Model's understanding doesn't match chunk content

**Diagnosis:** Model isn't parsing/understanding chunk text correctly

### Scenario 3: Model Not Tracking Across Chunks
**Symptoms:**
- Multiple chunks contain different entities
- Model only extracts from first chunk
- Model's chunks_used shows only [1]

**Diagnosis:** Model isn't reading all chunks before responding

### Scenario 4: Duplicate Extraction
**Symptoms:**
- Model extracts same entity multiple times
- Chunk analysis shows entity appears once
- Model's understanding shows duplicates

**Diagnosis:** Model is repeating extraction instead of tracking unique items

## Implementation Details

### Chunk Analysis Method (`_analyze_chunks`)
- Parses chunks from training example text
- Extracts entities using regex patterns:
  - `"Name serves as ROLE"`
  - `"As ROLE, Name"`
  - `"Name is ROLE"`
  - `"Name holds the position"`
- Compares found entities with expected items
- Shows summary of what's in each chunk

### Model Understanding Analysis (`_analyze_model_output`)
- Parses model's JSON output
- Extracts items, answer_type, chunks_used
- Detects duplicates
- Compares with expected items
- Shows what model "thinks" it extracted

## Usage

The diagnostics are automatically enabled when:
- `show_predictions=True` in training monitor
- Model generates predictions (not in very early training)

**Output appears in training logs:**
```
📊 TRAINING EXAMPLE MONITOR - Step 200

📈 Training Metrics:
   Loss: 0.0456
   Learning Rate: 4.34e-07
   Epoch: 3.71

📝 Sample Examples Being Processed (3 examples):
────────────────────────────────────────────────────────────────────────────

   Example 1 (Dataset Index 328):
   ────────────────────────────────────────────────────────────────────────────
   📋 Query: who are the executives of SmartSystems?
   ✅ Expected: {"answer_type": "entities", "items": ["Emery Hernandez", "Quinn Williams", "Logan Miller", "Jordan Jackson"], ...}
   
   🔍 Chunk Understanding Analysis:
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

## Next Steps Based on Diagnostics

### If Model Shows Duplicates:
- **Issue**: Model is repeating same entity instead of finding all
- **Fix**: Add more examples with explicit "extract ALL" instructions
- **Fix**: Increase LoRA rank for better tracking capacity

### If Model Shows Missing Items:
- **Issue**: Model stops after first match
- **Fix**: Add examples emphasizing "read ALL chunks completely"
- **Fix**: Increase weight on completeness in loss function

### If Model Shows Wrong Chunks Used:
- **Issue**: Model isn't reading all chunks
- **Fix**: Add examples with entities scattered across multiple chunks
- **Fix**: Simplify system prompt to emphasize "read all chunks"

## Benefits

1. **Real-time Diagnosis**: See issues as they happen during training
2. **Specific Problem Identification**: Know exactly what's wrong (duplicates, missing items, wrong chunks)
3. **Data-driven Fixes**: Make targeted improvements based on actual failures
4. **Training Efficiency**: Stop early if model isn't learning correctly

## Files Modified

- ✅ `training_example_monitor.py` - Added chunk analysis and model understanding diagnostics

The diagnostics are automatically active - no configuration needed!

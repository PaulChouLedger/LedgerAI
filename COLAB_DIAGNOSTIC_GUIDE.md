# Multi-Entity Extraction Diagnostic Tool - Google Colab Guide

## Quick Start

### Step 1: Install Dependencies

```python
!pip install unsloth transformers accelerate bitsandbytes
```

### Step 2: Upload Your Fine-Tuned Model

Upload your `outputs_rag_analysis/` folder to Colab. You can:
- Use Colab's file upload UI (folder icon on left sidebar)
- Or use `files.upload()` in a cell

**Important:** The model folder should contain:
- `config.json`
- `tokenizer.json` or `tokenizer_config.json`
- Model weights (`.safetensors` or `.bin` files)
- `adapter_config.json` (if using LoRA)

### Step 3: Upload the Diagnostic Script

Upload `debug_multi_entity_extraction.py` to Colab, or paste it into a cell.

### Step 4: Run the Diagnostic

```python
# Load and run the script
exec(open('debug_multi_entity_extraction.py').read())
```

Or if you pasted it into a cell, just run the cell.

## Usage Modes

### Mode 1: Test Case (Quick Test)

The script will automatically run a test case with the LedgerAI co-founders example:
- **Expected**: 4 co-founders (Paul Chou, Bob Carella, David Lara, Jorge Guinovart)
- **Checks**: Whether model extracts all 4 or just 1

### Mode 2: Interactive Mode

Enter your own query and chunks:
1. Enter your query (e.g., "who are the co-founders of X?")
2. Enter chunks one by one
3. Type 'done' when finished
4. View diagnostic results

## What the Diagnostic Shows

### 1. Chunk Analysis
- Finds all expected entities in chunks using regex
- Shows which chunk contains which entities
- **Example Output:**
  ```
  Chunk 1: Entities found: ['David Lara', 'Jorge Guinovart']
  Chunk 2: Entities found: ['Paul Chou', 'Bob Carella']
  Expected: 4 entities total
  ```

### 2. Token Analysis
- Checks if input fits within 8192 token limit
- Warns if truncation occurs (chunks get cut off)
- **Example Output:**
  ```
  Input token count: 2450
  Max sequence length: 8192
  Token usage: 29.9%
  ✅ Input fits within token limit
  ```

### 3. Model Reasoning Analysis
- Checks if model followed 6-step process
- Shows which chunks model mentioned
- Shows which entities model extracted
- **Example Output:**
  ```
  Step 1 (Understand Query): ✅
  Step 2 (Read Chunks): ✅
  Step 3 (Analyze Meaning): ✅
  Step 4 (Extract Info): ✅
  Step 5 (Verify Completeness): ❌  <-- Problem!
  Step 6 (Synthesize): ✅
  
  Chunks mentioned: [1, 2]
  Entities mentioned: ['Bob Carella']  <-- Only 1 of 4!
  ```

### 4. Extraction Comparison
- Compares expected vs actual entities
- Lists missing entities
- **Example Output:**
  ```
  Expected entities: ['Paul Chou', 'Bob Carella', 'David Lara', 'Jorge Guinovart']
  Actual entities extracted: ['Bob Carella']
  ❌ Missing entities: ['Paul Chou', 'David Lara', 'Jorge Guinovart']
  ```

## Interpreting Results

### Issue #1: Model Not Reading All Chunks
**Symptoms:**
- Chunks mentioned: [1] (only first chunk)
- Missing entities from later chunks

**Solution:**
- Model may need more training on multi-chunk examples
- Check if Step 2 (Read Chunks) is ✅

### Issue #2: Stopping After First Match
**Symptoms:**
- All chunks mentioned ✅
- Step 5 (Verify Completeness) is ❌
- Only extracts first entity found

**Solution:**
- Model learned to stop after first match
- Dataset may need more examples emphasizing "extract ALL"
- Training may need more epochs

### Issue #3: Token Truncation
**Symptoms:**
- Token count > 8192
- Warning: "Input exceeds max_length! Will be truncated"
- Later chunks cut off

**Solution:**
- Reduce chunk size
- Use fewer chunks
- Increase max_length (if model supports it)

### Issue #4: Training Issue
**Symptoms:**
- All steps present ✅
- All chunks mentioned ✅
- Still only extracts 1 entity

**Solution:**
- Model needs more training
- Dataset may need more multi-entity examples
- Consider increasing LoRA rank

## Results File

Results are saved to `extraction_diagnostic_results.json` with:
- Expected vs actual entities
- Full chunk analysis
- Token usage
- Model reasoning breakdown
- Full model response

## Custom Model Path

If your model is in a different location:

```python
# Specify custom path
model, tokenizer, model_type = load_model("/content/my_custom_model_path/")
```

## Troubleshooting

### "Model not found"
- Check folder name matches: `outputs_rag_analysis/`
- Verify model files are uploaded
- Try specifying path explicitly

### "Unsloth not installed"
```python
!pip install unsloth
```

### "CUDA out of memory"
- Script uses 4-bit quantization by default
- If still fails, reduce max_seq_length in load_model()

### "Module not found"
```python
!pip install transformers accelerate bitsandbytes
```

## Next Steps After Diagnosis

1. **If token truncation** → Fix chunk size/quantity
2. **If model ignores chunks** → Need more training epochs
3. **If model stops early** → Dataset needs more "extract ALL" examples
4. **If all steps present but incomplete** → Model needs more training

## Example Colab Notebook Cell

```python
# Cell 1: Install dependencies
!pip install unsloth transformers accelerate bitsandbytes

# Cell 2: Upload model (use Colab file UI or):
from google.colab import files
uploaded = files.upload()  # Upload outputs_rag_analysis folder

# Cell 3: Upload diagnostic script (or paste code)
# Upload debug_multi_entity_extraction.py

# Cell 4: Run diagnostic
exec(open('debug_multi_entity_extraction.py').read())
```

## Quick Test Command

For a quick test without interactive mode:

```python
# After loading model
from debug_multi_entity_extraction import run_test_case
results = run_test_case(model, tokenizer, model_type)
```

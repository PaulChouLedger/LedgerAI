# RAG Chunk Analysis Training Dataset

This directory contains scripts and datasets for fine-tuning an LLM to read, analyze, and score extracted information from RAG (Retrieval-Augmented Generation) chunks.

## Overview

The training teaches the model to:
1. **Read RAG chunks completely** - Don't stop reading once relevant information is found
2. **Evaluate relevance** - Categorize chunks as HIGH (≥0.70), MEDIUM (0.50-0.69), or LOW (<0.50) based on scores
3. **Extract only HIGH relevance information** - Be precise about what exactly matches the query
4. **Synthesize answers** - Combine information from multiple chunks to answer queries
5. **Handle various query types** - Factual, analytical, list, and personal reflection queries

## Files

### 1. `generate_rag_analysis_dataset.py`
Generates a supervised fine-tuning dataset with 100+ examples covering:
- **Factual queries** (20 examples): "who are the co-founders?", "what is the mission?", etc.
- **List queries** (15 examples): "list all the executives", "name all the products", etc.
- **Analytical queries** (20 examples): "analyze the strategic direction", "identify key themes", etc.
- **Personal reflection queries** (45 examples): "map major turning points", "analyze emotional themes", etc.

**Usage:**
```bash
python generate_rag_analysis_dataset.py
```

**Output:** `rag_analysis_dataset.json`

### 2. `train_rag_analysis_colab.py`
Fine-tuning script for Google Colab (or local GPU) that:
- Uses Qwen2.5-1.5B-Instruct model
- Configures LoRA adapters (rank 256)
- Trains for 8 epochs optimized for RAG analysis patterns
- Saves model in both HuggingFace and GGUF formats

**Usage in Colab:**
```python
# 1. Upload rag_analysis_dataset.json to Colab
# 2. Install dependencies
!pip install unsloth trl peft accelerate bitsandbytes datasets

# 3. Run training
!python train_rag_analysis_colab.py
```

**Output:**
- `outputs_rag_analysis/` - HuggingFace format model
- `gguf_model_rag_analysis/` - GGUF format model (for deployment)

## Dataset Structure

Each training example contains:
```json
{
  "messages": [
    {
      "role": "system",
      "content": "System prompt explaining RAG analysis task..."
    },
    {
      "role": "user",
      "content": "Query: [query text]\n\nRAG Chunks:\n[1] Score: 0.85, File: doc.pdf, Preview: '...'\n[1] FULL CHUNK TEXT: '...'"
    },
    {
      "role": "assistant",
      "content": "ANALYSIS: Reading all chunks completely...\nRELEVANCE EVALUATION:\n- HIGH relevance: X chunks\n- MEDIUM relevance: Y chunks\n- LOW relevance: Z chunks\n\nEXTRACTING INFORMATION:\n...\n\nSYNTHESIS:\n[Final answer based on HIGH relevance chunks]\n\n[Follow-up question]"
    }
  ],
  "query_type": "factual|list|analytical|personal",
  "num_chunks": 6,
  "high_relevance_count": 4
}
```

## Training Configuration

- **Model:** Qwen2.5-1.5B-Instruct (4-bit quantization)
- **LoRA Rank:** 256 (~180M trainable parameters, ~13.6% of model)
- **LoRA Alpha:** 512 (2x rank for optimal scaling)
- **Epochs:** 8
- **Batch Size:** 2 per device
- **Gradient Accumulation:** 4 (effective batch size = 8)
- **Learning Rate:** 1.5e-4
- **Max Sequence Length:** 2048

## Expected Model Behavior After Training

The fine-tuned model should:
1. ✅ Read every chunk completely before extracting information
2. ✅ Categorize chunks by relevance (HIGH/MEDIUM/LOW) based on scores
3. ✅ Extract only information from HIGH relevance chunks (score ≥0.70)
4. ✅ For list questions: Find EVERY matching item in EVERY chunk
5. ✅ For analytical questions: Extract all relevant information before synthesizing
6. ✅ Format responses showing: RELEVANCE EVALUATION → EXTRACTING INFORMATION → SYNTHESIS → Final Answer
7. ✅ End with a brief, natural follow-up question

## Example Query Types

### Factual Queries
- "who are the co-founders of LedgerAI?"
- "what is the mission of LedgerAI?"
- "who is the CEO of LedgerAI?"

### List Queries
- "list all the co-founders of LedgerAI"
- "what are all the products mentioned?"
- "name all the executives and their roles"

### Analytical Queries
- "analyze the strategic direction of LedgerAI based on the documents"
- "what are the key challenges mentioned in these documents?"
- "identify the main themes across these documents"

### Personal Reflection Queries
- "help me map the major turning points of my life and how they shaped my identity"
- "identify recurring patterns in my past writing that show how my thinking evolved"
- "what emotional themes show up most often when I write about regret or responsibility"
- "analyze relationships documented in these files. What were the power imbalances"

## Integration with LLM Container

After training, the model can be integrated into `llm-container/container_rest.py` to improve RAG chunk analysis. The model will:
- Better understand when to use HIGH vs MEDIUM vs LOW relevance chunks
- More accurately extract information from multiple chunks
- Synthesize answers more effectively for complex queries

## Notes

- The dataset includes examples where chunks don't contain the answer (teaching the model to recognize missing information)
- Some examples have intentionally low-scoring chunks to teach relevance discrimination
- The model learns to read chunks completely, not stopping at the first relevant sentence
- Personal reflection queries use realistic journal/therapy note examples

## Next Steps

1. Review `rag_analysis_dataset.json` to verify examples
2. Run training in Colab or on local GPU
3. Test the fine-tuned model on real RAG queries
4. Integrate into production LLM container if results are satisfactory


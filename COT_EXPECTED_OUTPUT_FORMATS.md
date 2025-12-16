# CoT Expected Output Formats - Enhancement Needed

## Issue Identified

The user correctly identified that **CoT steps need to include both instructions AND expected output formats** so the LLM can learn what structure to follow for each step.

## Current State

Currently, the dataset shows CoT steps with examples, but the **system prompt doesn't explicitly state what the expected output format is** for each step. The model learns from examples, but would benefit from explicit format specifications.

## Solution: Enhanced System Prompt

The system prompt should be updated to include **"EXPECTED OUTPUT FORMAT FOR STEP X"** sections that explicitly show:

1. **What the output structure should be** (template/format)
2. **Example outputs** showing the format in action

## Enhanced System Prompt Structure

Each step should now include:

```
STEP X: [STEP NAME]
- [Instruction 1]
- [Instruction 2]
- [Instruction 3]

EXPECTED OUTPUT FORMAT FOR STEP X:
"[Template showing expected structure]"

Example outputs:
- "[Example 1]"
- "[Example 2]"
```

## Specific Expected Output Formats

### STEP 1: UNDERSTAND THE QUERY
**Expected Format:**
```
"The query asks for [type]: [query]. I need to [action]."
```

**Examples:**
- "The query asks for co-founders of TechCorp. I need to extract ONLY people explicitly labeled as 'Co-Founder' of TechCorp, not other roles like CEO, CTO, CFO, or VP."
- "The query asks for a list: what are the features of blockchain?. I need to extract all items that match this query from all chunks."
- "The query asks for reasoning or causation: why did the company expand?. I need to extract information explaining why something happened, including causation words like 'because', 'due to', 'led to', or 'caused'."

### STEP 2: READ EACH CHUNK COMPLETELY
**Expected Format:**
```
"Chunk X (Score: Y.YY, [HIGH/MEDIUM/LOW] relevance): [first 1-2 sentences of chunk]..."
```

**Example:**
```
"Chunk 1 (Score: 0.85, HIGH relevance): John Smith is Co-Founder of TechCorp. Sarah Jones is Co-Founder of DataSystems.
Chunk 2 (Score: 0.66, MEDIUM relevance): Partnership ecosystems have been developed to create mutually beneficial business relationships..."
```

### STEP 3: ANALYZE CHUNK MEANING
**Expected Format:**
```
"Chunk X: [Contains entities: ...] [Relevant concepts: ...] Score Y.YY indicates [high/medium/low] relevance."
```

**Examples:**
- "Chunk 1: Contains entities: John Smith, Mike Brown. Relevant concepts: co-founder information. Score 0.85 indicates high relevance."
- "Chunk 2: Relevant concepts: causation/reasoning. Score 0.75 indicates high relevance."
- "Chunk 3: Score 0.46 indicates low relevance."

### STEP 4: EVALUATE RELEVANCE
**Expected Format:**
```
"Chunk X (Score: Y.YY, [HIGH/MEDIUM/LOW] relevance): [Directly answers/Does not directly answer] the query. [Contains information that matches the query requirements/Information should be ignored]."
```

**Examples:**
- "Chunk 1 (Score: 0.85, HIGH relevance): Directly answers the query. Contains information that matches the query requirements."
- "Chunk 2 (Score: 0.46, LOW relevance): Does not directly answer the query. Information should be ignored."

### STEP 5: EXTRACT MATCHING INFORMATION
**Expected Format:**
```
"Found X matching item(s):
  1. [item1]
  2. [item2]
  ...

Information found in: Chunk X, Chunk Y"
```

**OR if no information found:**
```
"No matching information found in any chunk. The query cannot be answered from the provided documents."
```

**Examples:**
- "Found 2 matching item(s):
  1. John Smith
  2. Mike Brown

Information found in: Chunk 1"
- "No matching information found in any chunk. The query cannot be answered from the provided documents."

### STEP 6: VERIFY COMPLETENESS
**Expected Format:**
```
"Read all X chunk(s) completely.
Extracted Y matching item(s) across all chunks.
Extraction is complete - [all relevant information has been identified/query cannot be answered from the provided documents]."
```

**Examples:**
- "Read all 4 chunk(s) completely.
Extracted 2 matching item(s) across all chunks.
Extraction is complete - all relevant information has been identified."
- "Read all 3 chunk(s) completely.
No matching information found in any chunk.
Extraction is complete - query cannot be answered from the provided documents."

### STEP 7: SYNTHESIZE RESPONSE
**Expected Format:**
```
[Just the final answer - no prefix, no "STEP 7:" marker, just the answer itself]
```

**Examples:**
- "John Smith and Mike Brown"
- "cloud-based storage, real-time analytics dashboard, automated reporting system, and mobile application"
- "I don't have that information in the provided documents"

## Implementation

A script has been created: `update_cot_system_prompt_with_expected_outputs.py`

This script will:
1. Load the dataset
2. Update all system prompts to include explicit expected output formats
3. Save the enhanced dataset

## Why This Matters

**Without explicit expected output formats:**
- Model learns from examples but may not understand the structure
- Model may produce inconsistent formats
- Model may not know what "good" output looks like for each step

**With explicit expected output formats:**
- Model knows exactly what structure to follow
- Model can learn the pattern more effectively
- Model produces consistent, well-structured outputs
- Model understands what "good" output looks like for each step

## Next Steps

1. Run `update_cot_system_prompt_with_expected_outputs.py` to update the dataset
2. Verify the system prompts have been updated correctly
3. Retrain the model with the enhanced dataset
4. Test to see if model produces more consistent CoT outputs

## Files Created

- `update_cot_system_prompt_with_expected_outputs.py` - Script to update system prompts
- `COT_EXPECTED_OUTPUT_FORMATS.md` - This documentation

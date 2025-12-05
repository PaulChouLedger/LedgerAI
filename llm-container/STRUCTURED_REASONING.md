# Structured Reasoning Integration

## Overview

The LLM reasoning and RAG prompting guide has been integrated into `container_rest.py` to provide structured, step-by-step reasoning for all queries.

## Features Integrated

### 1. Core Reasoning Rules
- ✅ Think step-by-step
- ✅ Use only provided information
- ✅ State "unknown" instead of guessing
- ✅ Identify contradictions/conflicts

### 2. Structured Output Format
The LLM is instructed to structure responses with:
- **Known Facts**: Key facts extracted from context
- **Reasoning Steps**: Step-by-step analysis process
- **Conflicts/Missing Info**: Contradictions or missing information
- **Final Answer**: Complete answer based on analysis
- **Confidence**: Rating (high/medium/low) based on information quality

### 3. RAG Chunk Evaluation
When RAG context is present, the LLM evaluates each context section:
- **High relevance**: Direct evidence that answers the question
- **Medium relevance**: Related information but not directly answering
- **Low relevance**: Not related to the question

### 4. Debug Mode Enhancement
When `SHOW_REASONING_DEBUG=true`, the LLM shows:
1. Known Facts
2. Reasoning Steps (with section-by-section analysis)
3. Conflicts / Missing Info
4. Final Answer
5. Confidence

The reasoning is logged but not spoken (only the Final Answer is streamed to TTS).

## Implementation Details

### Code Locations

1. **Structured Reasoning Instructions** (Lines 782-795)
   - Added to all prompt paths (RAG context, memory-only, instruction requests)
   - Generic and applicable to any query type

2. **RAG Chunk Evaluation** (Lines 797-807)
   - Only added when RAG context is present
   - Helps LLM identify most relevant sections

3. **Debug Mode Format** (Lines 820-842)
   - Enhanced to use structured format
   - Includes section-by-section analysis
   - Extracts only Final Answer for TTS

### Generic Design

All instructions are **generic** and work for:
- Co-founder questions
- Employee lists
- Product features
- Any relationship extraction
- Any query type

### Benefits

1. **Better Reasoning**: Step-by-step analysis improves accuracy
2. **Transparency**: Can see LLM's reasoning process in debug mode
3. **Quality Assessment**: Confidence ratings help identify uncertain answers
4. **Conflict Detection**: Identifies contradictions in provided information
5. **RAG Optimization**: Chunk evaluation helps identify most relevant sections

## Usage

### Normal Mode
The LLM uses structured reasoning internally but only outputs the final answer.

### Debug Mode
Enable with `SHOW_REASONING_DEBUG=true` in docker-compose.yml:
```yaml
llm-generic:
  environment:
    - SHOW_REASONING_DEBUG=true
```

The reasoning will be logged to console, but only the Final Answer is spoken.

## Example Output (Debug Mode)

```
Known Facts:
- Section 1 mentions: Paul Chou (CEO and Co-Founder of LedgerAI)
- Section 2 mentions: David Lara (Co-Founder and COO of LedgerAI), Jorge Guinovart (Co-Founder and CMO of LedgerAI)
- Section 3 mentions: Bob Carella (Co-Founder and CFO of LedgerAI)

Reasoning Steps:
STEP 1 - User is asking about co-founders of Ledger AI
STEP 2 - Analyzing Section 1: Found Paul Chou (Co-Founder of LedgerAI) - High relevance
STEP 2 - Analyzing Section 2: Found David Lara and Jorge Guinovart (both Co-Founders of LedgerAI) - High relevance
STEP 2 - Analyzing Section 3: Found Bob Carella (Co-Founder of LedgerAI) - High relevance
STEP 3 - All sections have high relevance
STEP 4 - Extracted 4 co-founders total
STEP 5 - No conflicts found

Conflicts / Missing Info:
None - all information is consistent

Final Answer:
The co-founders of Ledger AI are: Paul Chou (CEO and Co-Founder), David Lara (COO and Co-Founder), Jorge Guinovart (CMO and Co-Founder), and Bob Carella (CFO and Co-Founder).

Confidence:
High - all information is clearly stated in the context sections
```


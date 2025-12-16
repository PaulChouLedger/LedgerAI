# How the Model Learns from the Dataset Format

## Dataset Structure

Each training example is a JSON object with 3 messages:

```json
{
  "messages": [
    {
      "role": "system",
      "content": "[7,141 characters of instructions with expected output formats]"
    },
    {
      "role": "user", 
      "content": "Query: [question]\n\nRAG Chunks:\n[Chunk 1] Score: X.XX, File: ...\nFULL CHUNK TEXT: '...'\n[Chunk 2] ..."
    },
    {
      "role": "assistant",
      "content": "STEP 1: UNDERSTAND THE QUERY\n[reasoning]\n\nSTEP 2: READ EACH CHUNK COMPLETELY\n[reasoning]\n\n...\n\nSTEP 6: SYNTHESIZE RESPONSE\n[final answer]"
    }
  ]
}
```

## What the Model Sees During Training

### 1. SYSTEM MESSAGE (7,141 characters)

**Purpose:** Provides instructions and expected output formats

**Contains:**
- ✅ Instructions for each step (STEP 1-6)
- ✅ Expected output format for each step
- ✅ Example outputs showing what good responses look like
- ✅ Rules and guidelines (never hallucinate, exact matching, etc.)

**Example from system prompt:**
```
STEP 1: UNDERSTAND THE QUERY
- Identify what information is being requested
- Note any specific filtering requirements

EXPECTED OUTPUT FORMAT FOR STEP 1:
"The query asks: [query]. I need to [action]."

Example outputs:
- "The query asks for co-founders of TechCorp. I need to extract ONLY people explicitly labeled as 'Co-Founder'..."
- "The query asks: what is the difference between FutureCapital and AICapital?. I need to find relevant information for both entities..."
```

### 2. USER MESSAGE (3,331 characters in example)

**Purpose:** Provides the input (query + RAG chunks)

**Contains:**
- The query/question
- Multiple RAG chunks with:
  - Relevance scores (0.00-1.00)
  - File names
  - Full chunk text

**Example:**
```
Query: what is the difference between FutureCapital and AICapital?

RAG Chunks:
[Chunk 1] Score: 0.79, File: document.pdf
FULL CHUNK TEXT: 'Product lifecycle management processes have been optimized...'

[Chunk 2] Score: 0.85, File: document.pdf
FULL CHUNK TEXT: 'Brand positioning strategies have been refined...'
...
```

### 3. ASSISTANT MESSAGE (2,260 characters in example)

**Purpose:** Shows the expected output (what model should learn to generate)

**Contains:**
- Complete 6-step CoT reasoning
- This is the **ground truth** the model learns to replicate

**Example:**
```
STEP 1: UNDERSTAND THE QUERY
The query asks: what is the difference between FutureCapital and AICapital?. 
I need to find relevant information for both FutureCapital and AICapital and 
determine how the entities differ.

STEP 2: READ EACH CHUNK COMPLETELY
Chunk 1 (Score: 0.79, HIGH relevance): Product lifecycle management processes...
Chunk 2 (Score: 0.85, HIGH relevance): Brand positioning strategies...

STEP 3: ANALYZE CHUNK MEANING
Chunk 1: provides descriptive information regarding both FutureCapital and 
AICapital sufficient for comparison.
Chunk 2: provides descriptive information regarding both FutureCapital and 
AICapital sufficient for comparison.

STEP 4: EXTRACT MATCHING INFORMATION
Extract information from Chunk 1 and Chunk 2 and Chunk 4

STEP 5: VERIFY COMPLETENESS
Ensuring all relevant information was extracted. Read all 4 chunk(s) completely. 
Extracted 2 matching item(s) across all chunks. All relevant information has 
been identified.

STEP 6: SYNTHESIZE RESPONSE
The primary distinction between FutureCapital and AICapital lies in their 
handling of innovation strategy. While FutureCapital excels in pricing strategy, 
AICapital takes a more comprehensive approach to the market...
```

## How Fine-Tuning Works

### Training Process

1. **Model receives:** System prompt + User query + RAG chunks
2. **Model must predict:** The assistant's CoT response
3. **Loss is calculated:** Difference between predicted and actual assistant response
4. **Model updates:** Weights adjusted to minimize loss

### What the Model Learns

#### From System Message:
- **Structure:** "I should output STEP 1, then STEP 2, etc."
- **Format:** "Each step should follow this format..."
- **Rules:** "Never hallucinate, use exact matching, etc."

#### From User Messages:
- **Input patterns:** "When I see a query like this with chunks like this..."
- **Chunk format:** "Chunks have scores, file names, and full text"
- **Query types:** "Co-founder queries need role filtering, comparison queries need entity matching"

#### From Assistant Messages (Ground Truth):
- **Reasoning patterns:** "For this type of query, I should reason like this..."
- **Step content:** "STEP 3 should analyze like this..."
- **Output format:** "Final answer should be formatted like this..."

### Learning Mechanism

**Pattern Recognition:**
- Model sees 6,250 examples
- Each example shows: `Input (query+chunks) → Output (CoT steps)`
- Model learns: "Given similar input, I should produce similar output structure"

**Example Learning:**
```
Example 1: Co-founder query → STEP 1 (understand), STEP 2 (read chunks), 
           STEP 3 (analyze), STEP 4 (extract), STEP 5 (verify), STEP 6 (answer)

Example 2: Comparison query → STEP 1 (understand), STEP 2 (read chunks), 
           STEP 3 (analyze), STEP 4 (extract), STEP 5 (verify), STEP 6 (answer)

... (6,250 examples)

Model learns: "Always follow STEP 1-6 structure, but adapt content based on query type"
```

## Why This Format Works

### 1. **Explicit Instructions (System Message)**
- Model knows WHAT to do (instructions)
- Model knows HOW to format it (expected output formats)
- Model sees EXAMPLES of good outputs

### 2. **Concrete Examples (Assistant Messages)**
- Model sees 6,250 real examples of CoT reasoning
- Each example shows the complete reasoning process
- Model learns patterns: "For co-founder queries, STEP 1 should mention role filtering"

### 3. **Consistent Structure**
- All examples follow the same 6-step structure
- Model learns: "Always output STEP 1, then STEP 2, etc."
- Consistency helps model learn the pattern

### 4. **Rich Context**
- Each example includes full chunk text (not just summaries)
- Model learns to read and analyze complete chunks
- Model learns to extract from multiple chunks

## Training Statistics

- **Total examples:** 6,250
- **System prompt:** 7,141 characters (includes all instructions + expected formats)
- **Average user message:** ~3,000 characters (query + chunks)
- **Average assistant message:** ~2,000 characters (6-step CoT)
- **Total dataset size:** 79.9 MB

## What Makes This Effective

1. **Explicit Expected Output Formats:** Model knows exactly what structure to follow
2. **6,250 Examples:** Enough variety to learn patterns while maintaining consistency
3. **Complete Reasoning:** Model sees full reasoning process, not just final answers
4. **Systematic Structure:** 6-step process is consistent across all examples
5. **Rich Context:** Full chunk text allows model to learn proper analysis

## Model Learning Outcomes

After training, the model will:

1. **Recognize query types** and apply appropriate reasoning
2. **Follow the 6-step structure** consistently
3. **Format each step** according to expected output formats
4. **Analyze chunks meaningfully** (STEP 3)
5. **Extract relevant information** (STEP 4)
6. **Verify completeness** (STEP 5)
7. **Synthesize final answers** (STEP 6)

The model learns this through:
- **Supervised learning:** Given input, predict the correct output
- **Pattern matching:** Recognize patterns across 6,250 examples
- **Structure learning:** Learn the consistent 6-step format
- **Content adaptation:** Adapt step content based on query type

## Conclusion

The dataset format is **highly detailed** and provides:
- ✅ Clear instructions (system message)
- ✅ Rich input context (user message with full chunks)
- ✅ Complete reasoning examples (assistant message with full CoT)
- ✅ Expected output formats for each step
- ✅ 6,250 diverse examples showing the pattern

This format is **sufficient for the model to learn** because:
1. It provides explicit structure and format guidance
2. It shows complete reasoning processes (not just answers)
3. It includes enough examples to learn patterns
4. It maintains consistency while showing variety

The model learns by seeing thousands of examples of: "Given this query and these chunks, reason through it using these 6 steps in this format."

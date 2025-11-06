# Why LLM Generates Generic Questions Instead of Using Provided Terms

## Root Cause Analysis

### Model Limitations
- **Model**: Llama-3.2-1B-Instruct (1B parameters) or Nemotron-Mini-4B (4B parameters)
- **Issue**: Very small models struggle with:
  1. Complex multi-step instructions
  2. Following detailed rules
  3. Pattern matching vs. instruction following
  4. Long prompts with multiple constraints

### Prompt Issues

#### 1. **Conflicting Pattern in Rules** (Line 107)
The rules say:
```
Format: "Can you be more specific? For example, is it [option1], [option2], [option3], or [option4]?"
```

**Problem**: Small models see "Can you be more specific?" and copy that pattern, ignoring the rest.

#### 2. **Too Much Context at Top**
The prompt structure is:
```
{chief_complaint_context}
{conversation_context}

The patient already said: "{patient_answer}"
...
```

**Problem**: By the time the model reaches the instructions, it may have already decided on a response pattern.

#### 3. **Example Comes Too Late**
The example format is shown AFTER all the rules and context, so the model may not prioritize it.

#### 4. **Multiple "DO NOT" Instructions**
Small models often focus on what NOT to do, which can backfire - they may generate the forbidden pattern anyway.

## Why It Happens

1. **Pattern Matching**: Small models are pattern matchers. They see "Can you be more specific?" in the rules and generate it.
2. **Instruction Overload**: Too many rules confuse small models - they can't prioritize which instruction to follow.
3. **Context Confusion**: The conversation context might contain previous generic questions, reinforcing the pattern.
4. **Temperature**: Even at 0.1, small models can still be inconsistent with complex instructions.

## Solutions

### Option 1: Simplify Prompt (Recommended for Small Models)
- Remove "Can you be more specific?" from rules entirely
- Put example FIRST, before rules
- Use template-based approach: "Is it {term1}, {term2}, {term3}, or {term4}?"
- Remove conversation context (it may confuse the model)

### Option 2: Use Template Instead of LLM
- For clarification questions, use a simple template
- Only use LLM for natural language variation if needed
- More reliable for small models

### Option 3: Stronger Prompt Structure
- Start with example
- Use few-shot examples (show 2-3 correct examples)
- Remove negative examples (they can reinforce bad patterns)
- Simplify rules to 2-3 key points

### Option 4: Use Larger Model
- Upgrade to 7B+ model for better instruction following
- Trade-off: Slower inference, more memory

## Current Workaround

The manual fallback (lines 4617-4645) correctly builds the question when validation fails, so the system still works. The LLM failure is just a warning - the manual fallback ensures correct output.


# Clinical Reasoning Test Guide

This guide explains how to test if your fine-tuned LLM is using clinical reasoning patterns learned from training.

## Overview

The test scripts detect and analyze clinical reasoning patterns in the model's responses, including:

- **Comparative thinking**: "more concerning for X than Y"
- **Rule-in/rule-out logic**: "supports diagnosis X", "excludes condition Y"
- **Probability rankings**: "X is 85% probable"
- **Differential diagnosis**: Ranked list of conditions
- **Associated symptom reasoning**: How associated symptoms support diagnosis
- **Progressive narrowing**: How the differential narrows with each answer

## Test Scripts

### 1. Automated Test (`test_clinical_reasoning_colab.py`)

Runs predefined test cases and automatically analyzes reasoning.

**Usage:**
```python
# In Colab
!pip install unsloth transformers accelerate

# Upload your fine-tuned model (outputs/ or gguf_model/)
# Upload test_clinical_reasoning_colab.py

# Run the script
!python test_clinical_reasoning_colab.py
```

**What it does:**
- Loads your fine-tuned model
- Runs test conversations (chest pain, abdominal pain, etc.)
- Analyzes each response for reasoning patterns
- Provides summary statistics

**Output:**
```
✅ CLINICAL REASONING DETECTED (75% match)
   Patterns found: 6/8
   ✓ comparative_thinking, rule_in, probability, differential, clinical_reasoning, progressive_narrowing
```

### 2. Interactive Test (`test_clinical_reasoning_interactive_colab.py`)

Interactive version for manual testing - you can have conversations and see reasoning in real-time.

**Usage:**
```python
# In Colab
!pip install unsloth transformers accelerate

# Upload your fine-tuned model
# Upload test_clinical_reasoning_interactive_colab.py

# Run the script
!python test_clinical_reasoning_interactive_colab.py
```

**Commands:**
- Type your message to chat with the model
- Type `quit` to exit
- Type `reset` to start a new conversation
- Type `show` to see conversation history

**Example Session:**
```
👤 You: I have chest pain
🤖 Assistant: I understand you're experiencing chest pain. I'm here to help.

🧠 CLINICAL REASONING DETECTED!
   Patterns: comparative, probability
```

## What to Look For

### ✅ Good Clinical Reasoning

The model should show:

1. **Comparative thinking** after each answer:
   ```
   "Patient reported heavy pressure as chest pain character. 
    This is more concerning for Acute Myocardial Infarction 
    than costochondritis or musculoskeletal causes."
   ```

2. **Rule-in/rule-out logic**:
   ```
   "RULED IN: Acute MI - heavy pressure matches clinical pattern"
   "RULED OUT: Costochondritis - typically presents as sharp pain"
   ```

3. **Probability updates**:
   ```
   "Acute MI: 85% probability (increased from 70%)"
   ```

4. **Ranked differential**:
   ```
   "1. Acute MI: 85% probability
    2. Unstable Angina: 30% probability
    3. Pulmonary Embolism: 15% probability"
   ```

5. **Progressive narrowing**:
   ```
   "Based on location and character, the differential narrows 
    to cardiac causes, with Acute MI most probable."
   ```

### ❌ Missing Clinical Reasoning

If the model is NOT using reasoning, you'll see:

- Direct questions without explanation
- No comparative thinking
- No probability rankings
- No differential diagnosis updates
- No rule-in/rule-out logic

Example of **bad** response:
```
🤖 Assistant: Where is the pain located?
⚠️  No clinical reasoning detected
```

## Interpreting Results

### Reasoning Rate

- **≥ 50%**: ✅ Model is using clinical reasoning effectively
- **30-49%**: ⚠️ Model shows some reasoning, but may need more training
- **< 30%**: ❌ Model is not showing clinical reasoning - consider retraining

### Pattern Detection

The test checks for 8 reasoning patterns:
1. Comparative thinking
2. Rule-in logic
3. Rule-out logic
4. Probability rankings
5. Differential diagnosis
6. Clinical reasoning markers
7. Associated symptom reasoning
8. Progressive narrowing

**Good model**: 6-8 patterns detected
**Needs improvement**: 3-5 patterns detected
**Poor**: 0-2 patterns detected

## Troubleshooting

### Model Not Loading

**Error**: "Could not load any model format"

**Solutions**:
1. Ensure model files exist in `outputs/` (HuggingFace) or `gguf_model/` (GGUF)
2. Install required packages: `!pip install unsloth transformers accelerate`
3. For GGUF: `!pip install llama-cpp-python`

### No Reasoning Detected

**Possible causes**:
1. Model wasn't trained with reasoning examples
2. System prompt doesn't encourage reasoning
3. Model needs more training epochs

**Solutions**:
1. Retrain with `medical_sft_dataset_high_quality.json` (includes reasoning)
2. Update system prompt to explicitly request reasoning
3. Increase training epochs or learning rate

### Reasoning Too Verbose

If reasoning is too long or repetitive:

1. Check training data - ensure reasoning is concise
2. Adjust temperature (lower = more focused)
3. Limit max_new_tokens in generation

## Example Test Cases

### Test Case 1: Chest Pain (Acute MI)

```
User: I have chest pain
Expected: Empathy + reasoning about cardiac vs non-cardiac causes

User: It's new, started an hour ago
Expected: Reasoning about acute onset supporting cardiac etiology

User: In the center of my chest
Expected: Reasoning about retrosternal location supporting MI

User: It feels like heavy pressure
Expected: Reasoning about heavy pressure being classic for MI
```

### Test Case 2: Abdominal Pain (Appendicitis)

```
User: I have abdominal pain
Expected: Empathy + reasoning about abdominal causes

User: Lower right side
Expected: Reasoning about RLQ location supporting appendicitis

User: It feels sharp
Expected: Reasoning about sharp pain supporting appendicitis
```

## Advanced: Custom Test Cases

You can modify the test cases in `test_clinical_reasoning_colab.py`:

```python
test_cases = [
    {
        "name": "Your Test Case",
        "turns": [
            {"user": "I have [symptom]"},
            {"user": "[answer]"},
            # ... more turns
        ]
    }
]
```

## Next Steps

1. **Run the tests** after fine-tuning
2. **Review the results** - check reasoning rate and patterns
3. **If reasoning is low**: Retrain with more examples or adjust training parameters
4. **If reasoning is good**: Model is ready for deployment!

## Questions?

- Check if your training dataset includes reasoning examples
- Verify system prompt encourages clinical thinking
- Ensure model was trained for sufficient epochs
- Review training logs for convergence


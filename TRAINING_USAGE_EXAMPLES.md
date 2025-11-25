# Training Usage Examples: Enhanced Smart Dataset

## Quick Reference: How the Dataset is Used

### 1. Dataset Loading

```python
# train_medical_bot_colab.py automatically detects the dataset
DATASET_PATH = "medical_sft_dataset_enhanced_smart.json"

with open(DATASET_PATH, 'r', encoding='utf-8') as f:
    data = json.load(f)  # 726 conversations loaded
```

**Output:**
```
✅ Loaded 726 conversations from medical_sft_dataset_enhanced_smart.json
   📚 This is the ENHANCED SMART dataset with:
      - Skip tags for irrelevant OLD CARTS elements
      - British slang variations for UK market
      - Relevance metadata for each conversation
```

---

## Example 1: Skip Tag Training

### What the Model Sees During Training

**Training Input (Formatted with Qwen Template):**
```
<|im_start|>user
I have elevated blood pressure<|im_end|>
<|im_start|>assistant
When did it start?<|im_end|>
<|im_start|>user
gradually over the past few months<|im_end|>
<|im_start|>assistant
CLINICAL REASONING: This is the Onset (O)... [reasoning]<|im_end|>
<|im_start|>assistant
[SKIP:L] This OLD CARTS element is not relevant for this chief complaint and should be skipped.<|im_end|>
<|im_start|>assistant
How long has it been present?<|im_end|>
```

### What the Model Learns

**Pattern Recognition:**
- Sees skip tag `[SKIP:L]` after Onset question
- Learns: "For elevated blood pressure, skip Location"
- After seeing this pattern 5 times (5 conversations), model internalizes the rule

**After Training:**
```
Patient: "I have high blood pressure"
Model: "When did it start?" ✅
Model: [Internally skips Location question] ❌
Model: "How long has it been present?" ✅
```

---

## Example 2: British Slang Training

### American vs British Training Examples

**American Variant (Training Example 1):**
```
User: "I have chest pain"
User: "it came on suddenly"
User: "really bad"
```

**British Variant (Training Example 2 - Same Medical Logic):**
```
User: "I've got chest pain"
User: "it came on out of the blue"
User: "quite bad"
```

### What the Model Learns

**Language Equivalence:**
- Both examples teach the same medical assessment
- Model learns these phrases are equivalent:
  - "I have" = "I've got"
  - "suddenly" = "out of the blue"
  - "really bad" = "quite bad"

**After Training:**
The model understands both dialects and treats them equivalently.

---

## Example 3: Complete Training Flow

### Step-by-Step Process

**1. Load Dataset:**
```python
data = json.load(f)  # 726 conversations
# Each conversation has:
# - messages: [system, user, assistant messages]
# - relevant_oldcarts: {O: true, L: false, ...}
# - variant: "american" or "british"
# - smart_features: true
```

**2. Prepare Messages:**
```python
for conversation in data:
    messages = conversation.get("messages", [])
    # Includes:
    # - System prompts
    # - User messages (with British slang if variant="british")
    # - Assistant questions
    # - Skip tags: "[SKIP:L] This OLD CARTS element is not relevant..."
    # - Clinical reasoning after each answer
    # - Final diagnostic reasoning
```

**3. Apply Chat Template:**
```python
text = tokenizer.apply_chat_template(messages, tokenize=False)
# Converts to Qwen 2.5 format:
# <|im_start|>role
# content<|im_end|>
```

**4. Train Model:**
```python
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,  # All 726 conversations
    ...
)
trainer.train()  # Model learns patterns
```

---

## Example 4: Real Training Scenarios

### Scenario A: Hypertension (Skip Location, Character, Radiation)

**Training Data:**
```
User: "I have elevated blood pressure"
Assistant: "When did it start?" ✅
User: "gradually"
Assistant: [CLINICAL REASONING about Onset...]
Assistant: "[SKIP:L] This OLD CARTS element is not relevant..." ❌
Assistant: "How long has it been present?" ✅
User: "for months"
Assistant: [CLINICAL REASONING about Duration...]
Assistant: "[SKIP:C] This OLD CARTS element is not relevant..." ❌
Assistant: "What makes it worse?" ✅
```

**Model Learns:**
- Skip Location (L) for hypertension
- Skip Character (C) for hypertension
- Skip Radiation (R) for hypertension
- Ask Onset, Duration, Aggravating, Alleviating, Timing, Severity

### Scenario B: Chest Pain (Ask All Elements)

**Training Data:**
```
User: "I have chest pain"
Assistant: "When did it start?" ✅
User: "suddenly"
Assistant: [CLINICAL REASONING...]
Assistant: "Where exactly is the chest pain located?" ✅ (No skip tag)
User: "center of chest"
Assistant: [CLINICAL REASONING...]
Assistant: "What does the chest pain feel like?" ✅ (No skip tag)
User: "pressure"
Assistant: [CLINICAL REASONING...]
Assistant: "Does it spread to other areas?" ✅ (No skip tag)
```

**Model Learns:**
- For pain conditions, ask ALL OLD CARTS elements
- Location, Character, and Radiation are relevant for pain

### Scenario C: British Variant (Language Learning)

**Training Data:**
```
User: "I've got chest pain"  (British)
Assistant: "When did the chest pain start?"
User: "it came on out of the blue"  (British)
Assistant: [CLINICAL REASONING...]
Assistant: "Where exactly is the chest pain located?"
User: "right over here in the center"  (British)
```

**Model Learns:**
- "I've got" = "I have"
- "out of the blue" = "suddenly"
- "right over here" = "right here"
- All mean the same thing medically

---

## Example 5: Training Output Format

### What Gets Fed to the Model

**Single Training Example (Hypertension):**
```
<|im_start|>system
You are a medical professional conducting a clinical history...
<|im_end|>
<|im_start|>user
I have elevated blood pressure<|im_end|>
<|im_start|>assistant
I understand you're experiencing elevated blood pressure. I'm here to help.
Is this a new issue that just started, or is this an ongoing problem you've had before with a prior diagnosis?<|im_end|>
<|im_start|>user
It's new, started recently<|im_end|>
<|im_start|>assistant
How old are you?<|im_end|>
<|im_start|>user
I'm 45 years old<|im_end|>
<|im_start|>assistant
What is your biological sex?<|im_end|>
<|im_start|>user
Male<|im_end|>
<|im_start|>assistant
When did it start?<|im_end|>
<|im_start|>user
gradually over the past few months<|im_end|>
<|im_start|>assistant
CLINICAL REASONING: This is the Onset (O) - when the symptom started element of OLD CARTS.
ELEMENT IDENTIFICATION: The patient's answer 'gradually over the past few months' is being evaluated for the Onset (O) element.
CRITICAL: This is Onset (O), NOT any other OLD CARTS element...
Patient reported 'gradually over the past few months' for ONSET (O). This onset pattern supports Essential Hypertension...

RULED IN (increased probability):
• Essential Hypertension: Onset (O) 'gradually over the past few months' matches clinical pattern. Likelihood increased to 35%.

CURRENT DIFFERENTIAL DIAGNOSIS (ranked by probability):
1. Essential Hypertension: 35% probability (MOST PROBABLE)
2. Secondary Hypertension: 30% probability (less likely)
3. White Coat Hypertension: 25% probability (less likely)

NEXT STEP: Continue OLD CARTS assessment to further narrow differential.<|im_end|>
<|im_start|>assistant
[SKIP:L] This OLD CARTS element is not relevant for this chief complaint and should be skipped.<|im_end|>
<|im_start|>assistant
How long has it been present?<|im_end|>
...
```

**Model Training:**
- Sees 726 such examples
- Learns skip patterns from skip tags
- Learns language variations from British variants
- Learns clinical reasoning from reasoning sections

---

## Example 6: Expected Model Behavior

### After Training - Real Usage

**Input:**
```
Patient: "I have high blood pressure"
```

**Model Output (Trained):**
```
Assistant: "I understand you're experiencing high blood pressure. I'm here to help."
Assistant: "Is this a new issue that just started, or is this an ongoing problem?"
Patient: "It's new"
Assistant: "How old are you?"
Patient: "45"
Assistant: "What is your biological sex?"
Patient: "Male"
Assistant: "When did it start?"  ✅ (Learned: Onset is relevant)
Patient: "gradually"
Assistant: [Provides clinical reasoning about Onset]
Assistant: "How long has it been present?"  ✅ (Learned: Duration is relevant)
// Model SKIPS Location question (learned from skip tags)
// Model SKIPS Character question (learned from skip tags)
Assistant: "What makes it worse?"  ✅ (Learned: Aggravating is relevant)
```

**Key Learning:**
- Model learned to skip Location, Character, Radiation for hypertension
- Model learned from skip tags in training data
- Model asks only relevant questions

---

## Example 7: British Market Usage

### After Training - UK Deployment

**Input (British Patient):**
```
Patient: "I've got chest pain"
```

**Model Output (Trained):**
```
Assistant: "I understand you're experiencing chest pain. I am here to help."
Assistant: "When did the chest pain start?"
Patient: "it came on out of the blue"
Assistant: [Clinical reasoning...]
Assistant: "Where exactly is the chest pain located?"
Patient: "right over here in the center"
Assistant: [Clinical reasoning...]
Assistant: "What does the chest pain feel like?"
Patient: "it feels a bit like pressure"
```

**Key Learning:**
- Model understands British slang from training
- Model responds naturally to British English
- Model provides same medical assessment regardless of dialect

---

## Training Statistics

### Dataset Breakdown

- **Total Conversations:** 726
  - American: 363 (50%)
  - British: 363 (50%)

- **Conversations with Skip Tags:** 42
- **Total Skip Tags:** 85
  - Location (L): 28 skips
  - Character (C): 28 skips
  - Radiation (R): 29 skips

- **Training Examples per Pattern:**
  - Skip Location: ~28 examples
  - Skip Character: ~28 examples
  - Skip Radiation: ~29 examples
  - British slang: 363 examples

---

## Summary

The enhanced smart dataset provides:

1. **726 Training Examples** covering:
   - Multiple diagnoses
   - American and British English
   - Skip patterns for irrelevant questions
   - Clinical reasoning examples

2. **Skip Tag Learning:**
   - Model sees skip tags in training
   - Learns which questions to skip
   - Generalizes to similar conditions

3. **Language Learning:**
   - Model sees both American and British variants
   - Learns language equivalence
   - Becomes dialect-agnostic

4. **Clinical Reasoning:**
   - Model sees reasoning after each answer
   - Learns to provide structured reasoning
   - Learns differential diagnosis building

After training, the model will automatically:
- ✅ Skip irrelevant OLD CARTS questions
- ✅ Handle British English naturally
- ✅ Provide clinical reasoning
- ✅ Build differential diagnoses
- ✅ Ask only appropriate questions


# How the Enhanced Smart Dataset is Used in Training

## Overview

The `medical_sft_dataset_enhanced_smart.json` dataset is used with `train_medical_bot_colab.py` to fine-tune the Qwen 2.5 model. Here are concrete examples of how the dataset features train the model.

## Example 1: Learning to Skip Irrelevant Questions

### Training Example: Hypertension Conversation

**Input to Model (formatted with Qwen chat template):**
```
<|im_start|>system
You are a medical professional conducting a clinical history...
<|im_end|>
<|im_start|>user
I have elevated blood pressure<|im_end|>
<|im_start|>assistant
I understand you're experiencing elevated blood pressure. I'm here to help.
Is this a new issue that just started, or is this an ongoing problem you've had before?<|im_end|>
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
[SKIP:L] This OLD CARTS element is not relevant for this chief complaint and should be skipped.<|im_end|>
<|im_start|>assistant
How long has it been present?<|im_end|>
...
```

**What the Model Learns:**
- ✅ After asking about onset, it should **NOT** ask "Where is your elevated blood pressure located?"
- ✅ The skip tag `[SKIP:L]` teaches the model that Location questions don't make sense for hypertension
- ✅ The model learns to skip Location, Character, and Radiation for systemic conditions

**After Training:**
When a patient says "I have high blood pressure", the model will:
- ✅ Ask: "When did it start?" (Onset - relevant)
- ✅ Ask: "How long has it been present?" (Duration - relevant)
- ❌ **Skip**: "Where is your high blood pressure located?" (Location - not relevant)
- ❌ **Skip**: "What does your high blood pressure feel like?" (Character - not relevant)
- ✅ Ask: "What makes it worse?" (Aggravating - relevant)

## Example 2: British Slang Handling

### Training Example: British Variant Conversation

**American Version:**
```
<|im_start|>user
I have chest pain<|im_end|>
<|im_start|>assistant
When did the chest pain start?<|im_end|>
<|im_start|>user
it came on suddenly<|im_end|>
```

**British Version (same conversation):**
```
<|im_start|>user
I've got chest pain<|im_end|>
<|im_start|>assistant
When did the chest pain start?<|im_end|>
<|im_start|>user
it came on out of the blue<|im_end|>
```

**What the Model Learns:**
- ✅ "I have" and "I've got" mean the same thing
- ✅ "came on suddenly" and "came on out of the blue" mean the same thing
- ✅ "really bad" and "quite bad" are equivalent
- ✅ The model becomes language-agnostic for UK market

**After Training:**
The model will understand both:
- American: "I have chest pain, it's really bad"
- British: "I've got chest pain, it's quite bad"

## Example 3: Smart Question Selection for Pain Conditions

### Training Example: Chest Pain Conversation

**Dataset Structure:**
```json
{
  "messages": [
    {"role": "user", "content": "I have chest pain"},
    {"role": "assistant", "content": "When did it start?"},
    {"role": "user", "content": "suddenly during exertion"},
    {"role": "assistant", "content": "CLINICAL REASONING: ..."},
    {"role": "assistant", "content": "Where exactly is the chest pain located?"},
    {"role": "user", "content": "center of chest"},
    {"role": "assistant", "content": "CLINICAL REASONING: ..."},
    {"role": "assistant", "content": "What does the chest pain feel like?"},
    ...
  ],
  "relevant_oldcarts": {
    "O": true,   // Onset - relevant for pain
    "L": true,   // Location - relevant for pain
    "D": true,   // Duration - relevant
    "C": true,   // Character - relevant for pain
    "A_aggravating": true,
    "A_alleviating": true,
    "R": true,   // Radiation - relevant for pain
    "T": true,
    "S": true
  }
}
```

**What the Model Learns:**
- ✅ For pain conditions, ALL OLD CARTS elements are relevant
- ✅ Location, Character, and Radiation are important for pain
- ✅ The model learns to ask comprehensive questions for pain

**After Training:**
For "chest pain", the model will ask:
- ✅ "When did it start?" (Onset)
- ✅ "Where exactly is it located?" (Location)
- ✅ "What does it feel like?" (Character)
- ✅ "Does it spread anywhere?" (Radiation)
- ✅ All other OLD CARTS elements

## Example 4: Clinical Reasoning Training

### Training Example: Progressive Differential Diagnosis

**Formatted Training Text:**
```
<|im_start|>assistant
CLINICAL REASONING: This is the Onset (O) - when the symptom started element of OLD CARTS.
ELEMENT IDENTIFICATION: The patient's answer 'started suddenly' is being evaluated for the Onset (O) element.
CRITICAL: This is Onset (O), NOT any other OLD CARTS element...

Patient reported 'started suddenly' for ONSET (O). This onset pattern is CLASSIC for Acute Upper GI Bleed.

RULED IN (increased probability):
• Acute Upper GI Bleed: Onset (O) 'started suddenly' matches clinical pattern. Likelihood increased to 35%.

CURRENT DIFFERENTIAL DIAGNOSIS (ranked by probability):
1. Acute Upper GI Bleed: 35% probability (MOST PROBABLE)
2. Acute Cholecystitis: 35% probability (less likely)
3. Bowel Obstruction: 20% probability (less likely)

NEXT STEP: Continue OLD CARTS assessment to further narrow differential.<|im_end|>
```

**What the Model Learns:**
- ✅ How to identify which OLD CARTS element an answer corresponds to
- ✅ How to reason about clinical patterns
- ✅ How to build and rank differential diagnoses
- ✅ How to progressively narrow the differential
- ✅ Rule-in/rule-out logic

**After Training:**
The model will:
- ✅ Recognize "started suddenly" as Onset (O) information
- ✅ Understand that sudden onset is classic for certain conditions
- ✅ Update probability rankings after each answer
- ✅ Provide clinical reasoning automatically

## Example 5: Complete Training Flow

### Step-by-Step Training Process

**1. Dataset Loading:**
```python
# train_medical_bot_colab.py loads the dataset
with open('medical_sft_dataset_enhanced_smart.json', 'r') as f:
    data = json.load(f)  # 726 conversations

# Detects smart features
if "enhanced" in DATASET_PATH.lower() and "smart" in DATASET_PATH.lower():
    print("✅ Smart features detected:")
    print("   - Skip tags for irrelevant OLD CARTS elements")
    print("   - British slang variations")
    print("   - Relevance metadata")
```

**2. Message Preparation:**
```python
# Each conversation's messages are prepared
for conversation in data:
    messages = conversation.get("messages", [])
    # Includes:
    # - System prompts
    # - User messages (with British slang if variant="british")
    # - Assistant questions
    # - Skip tags: "[SKIP:L] This OLD CARTS element is not relevant..."
    # - Clinical reasoning
    # - Final diagnostic reasoning
```

**3. Chat Template Formatting:**
```python
# Qwen 2.5 chat template is applied
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=False
)
# Result: Formatted text with <|im_start|> and <|im_end|> tags
```

**4. Training:**
```python
# Model learns from formatted conversations
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,  # Contains all 726 conversations
    ...
)
trainer.train()  # Model learns patterns
```

## Example 6: Real Training Examples

### Example A: Hypertension (American)

**Training Input:**
```
System: You are a medical professional...
User: I have elevated blood pressure
Assistant: I understand you're experiencing elevated blood pressure...
Assistant: When did it start?
User: gradually over the past few months
Assistant: CLINICAL REASONING: This is the Onset (O)... [reasoning]
Assistant: [SKIP:L] This OLD CARTS element is not relevant...
Assistant: How long has it been present?
User: for a few months
Assistant: CLINICAL REASONING: This is the Duration (D)... [reasoning]
Assistant: [SKIP:C] This OLD CARTS element is not relevant...
Assistant: What makes it worse?
...
```

**Model Learns:**
- Skip Location, Character, Radiation for hypertension
- Ask Onset, Duration, Aggravating, Alleviating, Timing, Severity

### Example B: Chest Pain (British)

**Training Input:**
```
System: You are a medical professional...
User: I've got chest pain
Assistant: I understand you're experiencing chest pain...
Assistant: When did it start?
User: it came on out of the blue
Assistant: CLINICAL REASONING: This is the Onset (O)... [reasoning]
Assistant: Where exactly is the chest pain located?
User: right over here in the center
Assistant: CLINICAL REASONING: This is the Location (L)... [reasoning]
Assistant: What does the chest pain feel like?
User: it feels a bit like pressure
Assistant: CLINICAL REASONING: This is the Character (C)... [reasoning]
...
```

**Model Learns:**
- Understand British slang: "I've got", "out of the blue", "feels a bit like", "right over here"
- Ask all OLD CARTS elements for pain (no skips)
- Provide clinical reasoning in British English context

### Example C: Palpitations (Skip Radiation)

**Training Input:**
```
User: I have palpitations
Assistant: When did they start?
User: suddenly
Assistant: CLINICAL REASONING: ... [reasoning]
Assistant: [SKIP:L] This OLD CARTS element is not relevant...
Assistant: How long have they been present?
User: for minutes
Assistant: CLINICAL REASONING: ... [reasoning]
Assistant: What do the palpitations feel like?
User: racing
Assistant: CLINICAL REASONING: ... [reasoning]
Assistant: [SKIP:R] This OLD CARTS element is not relevant...
Assistant: Is it constant or does it come and go?
...
```

**Model Learns:**
- Skip Location and Radiation for palpitations (systemic symptom)
- Ask Character (what they feel like - "racing", "fluttering")
- Understand that palpitations don't have a physical location or radiation

## Training Outcomes

### After Training, the Model Will:

1. **Skip Irrelevant Questions:**
   - ❌ Won't ask "Where is your hypertension located?"
   - ❌ Won't ask "What does your polyuria feel like?"
   - ✅ Will ask relevant questions based on chief complaint

2. **Handle British English:**
   - ✅ Understands "I've got" = "I have"
   - ✅ Understands "quite bad" = "really bad"
   - ✅ Understands "proper painful" = "hurting"
   - ✅ Works seamlessly in UK market

3. **Provide Clinical Reasoning:**
   - ✅ Identifies OLD CARTS elements automatically
   - ✅ Builds differential diagnoses
   - ✅ Updates probability rankings
   - ✅ Provides rule-in/rule-out logic

4. **Ask Appropriate Questions:**
   - ✅ For pain: asks Location, Character, Radiation
   - ✅ For systemic conditions: skips Location, Character, Radiation
   - ✅ Always asks Onset, Duration, Aggravating, Alleviating, Timing, Severity

## Usage in Training Script

The training script automatically:

1. **Detects the dataset:**
   ```python
   # Automatically selects best available dataset
   if os.path.exists("medical_sft_dataset_enhanced_smart.json"):
       DATASET_PATH = "medical_sft_dataset_enhanced_smart.json"
       print("✅ Using ENHANCED SMART dataset")
   ```

2. **Processes all conversations:**
   - 363 American variants
   - 363 British variants
   - All with smart features and skip tags

3. **Formats with Qwen template:**
   - Converts messages to Qwen 2.5 format
   - Preserves skip tags and clinical reasoning
   - Maintains British slang variations

4. **Trains the model:**
   - Model learns skip patterns
   - Model learns British slang
   - Model learns clinical reasoning
   - Model learns appropriate question selection

## Expected Model Behavior After Training

### Scenario 1: Patient with Hypertension
**Patient:** "I have high blood pressure"
**Model will:**
- ✅ Ask: "When did it start?"
- ✅ Ask: "How long has it been present?"
- ❌ **Skip**: "Where is your high blood pressure located?"
- ❌ **Skip**: "What does your high blood pressure feel like?"
- ✅ Ask: "What makes it worse?"
- ✅ Ask: "What makes it better?"
- ❌ **Skip**: "Does it spread to other areas?"
- ✅ Ask: "Is it constant or does it come and go?"
- ✅ Ask: "How severe is it?"

### Scenario 2: Patient with Chest Pain (UK)
**Patient:** "I've got chest pain"
**Model will:**
- ✅ Understand British slang
- ✅ Ask: "When did it start?"
- ✅ Ask: "Where exactly is the chest pain located?"
- ✅ Ask: "What does the chest pain feel like?"
- ✅ Ask: "Does it spread to other areas?"
- ✅ Ask all relevant OLD CARTS elements

### Scenario 3: Patient with Palpitations
**Patient:** "I have palpitations"
**Model will:**
- ✅ Ask: "When did they start?"
- ❌ **Skip**: "Where are the palpitations located?"
- ✅ Ask: "What do the palpitations feel like?" (racing, fluttering)
- ❌ **Skip**: "Do they spread to other areas?"
- ✅ Ask other relevant elements

## Key Training Benefits

1. **Efficiency:** Model learns to skip 3-4 irrelevant questions per conversation
2. **Accuracy:** Only asks questions that make medical sense
3. **Language Support:** Handles both American and British English
4. **Clinical Reasoning:** Provides automatic differential diagnosis reasoning
5. **User Experience:** Faster, more natural conversations

## Summary

The enhanced smart dataset trains the model to:
- ✅ **Skip irrelevant questions** (learns from skip tags)
- ✅ **Handle British slang** (learns from British variants)
- ✅ **Provide clinical reasoning** (learns from reasoning examples)
- ✅ **Ask appropriate questions** (learns from relevance metadata)
- ✅ **Build differential diagnoses** (learns from diagnostic reasoning)

All 726 conversations (363 American + 363 British) are used in training, teaching the model comprehensive medical conversation skills with smart question selection and multi-language support.


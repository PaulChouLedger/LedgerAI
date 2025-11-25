# Intelligent Follow-Up Questions: Summary

## What Was Added

The dataset now includes **intelligent, diagnosis-specific follow-up questions** that leverage medical knowledge to ask contextually relevant questions beyond the OLD CARTS framework.

## Key Features

### 1. **Context-Aware Questions**

Questions are automatically selected based on the probable diagnosis:

- **Hypertension** → Ask about BP medications, family history, salt intake, smoking, exercise
- **Chest Pain** → Ask about cardiac history, medications (aspirin, statins), risk factors, associated symptoms
- **GI Bleed** → Ask about NSAIDs, anticoagulants, alcohol use, previous GI history
- **GERD** → Ask about current medications, dietary triggers, postural symptoms, red flags
- **Palpitations** → Ask about medications, caffeine/alcohol, thyroid history, associated symptoms

### 2. **Medical Knowledge Categories**

Questions are organized by medical relevance:

- **Medications:** Current medications, medication history
- **Risk Factors:** Smoking, family history, comorbidities
- **Lifestyle:** Diet, exercise, alcohol use
- **Associated Symptoms:** Related symptoms that help refine diagnosis
- **Red Flags:** Warning signs requiring urgent evaluation
- **Medical History:** Previous conditions, procedures
- **Epidemiology:** Exposure history, contacts

### 3. **Clinical Reasoning**

Each follow-up question includes clinical reasoning explaining:
- Why the question is medically relevant
- How the answer affects the assessment
- Clinical significance of the information

## Example: Hypertension

**After OLD CARTS Assessment:**
```
Assistant: "FINAL DIAGNOSTIC REASONING: ... Essential Hypertension (99% probability) ..."
```

**Intelligent Follow-Up Questions:**
```
Assistant: "Are you currently taking any medications for blood pressure?"
User: "Yes, I take lisinopril daily"
Assistant: [Clinical reasoning about medications]

Assistant: "Do you have a family history of high blood pressure or heart disease?"
User: "Yes, my mother has high blood pressure"
Assistant: [Clinical reasoning about family history]
```

## Example: Chest Pain

**After OLD CARTS Assessment:**
```
Assistant: "FINAL DIAGNOSTIC REASONING: ... Chest Pain (Cardiac) (99% probability) ..."
```

**Intelligent Follow-Up Questions:**
```
Assistant: "Do you have any history of heart disease, heart attack, or cardiac procedures?"
User: "No, no history of heart problems"
Assistant: [Clinical reasoning]

Assistant: "Are you taking any medications like aspirin, clopidogrel, or blood thinners?"
User: "Yes, I take daily aspirin"
Assistant: [Clinical reasoning about medications]
```

## Example: GI Bleed

**After OLD CARTS Assessment:**
```
Assistant: "FINAL DIAGNOSTIC REASONING: ... Acute Upper GI Bleed (99% probability) ..."
```

**Intelligent Follow-Up Questions:**
```
Assistant: "Are you taking any medications like aspirin, ibuprofen, naproxen, or other NSAIDs?"
User: "Yes, I take aspirin daily"
Assistant: [Clinical reasoning about NSAIDs]

Assistant: "Do you drink alcohol, and if so, how much?"
User: "I drink daily, several drinks"
Assistant: [Clinical reasoning about alcohol]
```

## Training Benefits

### 1. **Beyond OLD CARTS**

The model learns to ask intelligent questions that go beyond the standard OLD CARTS framework.

### 2. **Context-Aware**

The model learns to ask different questions based on the probable diagnosis.

### 3. **Medical Knowledge**

The model learns to leverage medical knowledge:
- Medications relevant to the condition
- Risk factors
- Red flag symptoms
- Lifestyle and social history

### 4. **Clinical Reasoning**

The model learns to provide clinical reasoning for follow-up questions.

## Dataset Statistics

- **Total Conversations:** 726
- **Conversations with Follow-ups:** 22+ (varies by run)
- **Follow-up Questions per Conversation:** 2 (configurable)
- **Diagnoses Covered:** 7+ diagnoses with intelligent follow-ups

## Files

1. **`add_intelligent_followups.py`** - Script to add follow-up questions
2. **`medical_sft_dataset_enhanced_smart_intelligent.json`** - Enhanced dataset with follow-ups
3. **`INTELLIGENT_FOLLOWUPS_DOCUMENTATION.md`** - Full documentation

## Usage

### Add Follow-Ups to Dataset

```bash
python3 add_intelligent_followups.py \
    medical_sft_dataset_enhanced_smart.json \
    medical_sft_dataset_enhanced_smart_intelligent.json \
    2  # Number of follow-up questions per conversation
```

### Use in Training

The enhanced dataset (`medical_sft_dataset_enhanced_smart_intelligent.json`) can be used directly with `train_medical_bot_colab.py`:

```python
# train_medical_bot_colab.py will automatically:
# 1. Load conversations with intelligent follow-ups
# 2. Train model to ask context-aware questions
# 3. Learn clinical reasoning for follow-ups
```

## Expected Model Behavior

After training, the model will:

1. **Complete OLD CARTS assessment** first
2. **Identify probable diagnosis** from clinical reasoning
3. **Ask intelligent follow-up questions** based on diagnosis:
   - Hypertension → Medications, family history, lifestyle
   - Chest Pain → Cardiac history, medications, risk factors
   - GI Bleed → NSAIDs, anticoagulants, alcohol
4. **Provide clinical reasoning** for follow-up questions

## Summary

The intelligent follow-up questions feature:

- ✅ Adds context-aware, diagnosis-specific questions
- ✅ Leverages medical knowledge (medications, risk factors, etc.)
- ✅ Goes beyond the OLD CARTS framework
- ✅ Provides clinical reasoning for follow-up questions
- ✅ Trains the model to ask intelligent, relevant questions

This makes the model more clinically useful and better at gathering comprehensive patient information.


# High-Quality Medical Training Dataset

## Overview

`medical_sft_dataset_high_quality.json` is a comprehensive training dataset designed to teach LLMs how to conduct medical history-taking with interleaved differential diagnosis reasoning.

## Features

### ✅ Complete OLD CARTS Sequence
- Empathy → Chronicity → Age → Biological Sex → OLD CARTS elements
- Natural, conversational question flow
- One question at a time

### ✅ Interleaved Differential Diagnosis Reasoning
- Reasoning appears **after each OLD CARTS answer**
- Shows which conditions are **RULED IN** (increased likelihood)
- Shows which conditions are **RULED OUT** (contradicted)
- Updates differential diagnosis progressively
- Confidence scores capped at 100% for realism

### ✅ Final Diagnostic Conclusion
- Most probable diagnosis with confidence score
- Complete reasoning path showing how diagnosis was reached
- Key matching factors from OLD CARTS elements

### ✅ Multi-Organ System Support
- Currently includes GI conditions (2-3 per system)
- Structured to easily expand to other organ systems:
  - Cardiovascular
  - Respiratory
  - Neurological
  - Dermatologic
  - Genitourinary

### ✅ Sensory and Visual Complaints
- Handles sensory symptoms (pain, burning, pressure)
- Handles visual symptoms (blood, rash, appearance)
- Appropriate question phrasing for each type

## Dataset Structure

Each conversation follows this format:

```json
{
  "messages": [
    {
      "role": "system",
      "content": "System prompt with conditions and context..."
    },
    {
      "role": "user",
      "content": "Chief complaint..."
    },
    {
      "role": "assistant",
      "content": "Empathetic response..."
    },
    {
      "role": "assistant",
      "content": "Chronicity question..."
    },
    {
      "role": "user",
      "content": "Answer..."
    },
    {
      "role": "assistant",
      "content": "Age question..."
    },
    {
      "role": "user",
      "content": "Answer..."
    },
    {
      "role": "assistant",
      "content": "Sex question..."
    },
    {
      "role": "user",
      "content": "Answer..."
    },
    {
      "role": "assistant",
      "content": "OLD CARTS question..."
    },
    {
      "role": "user",
      "content": "Answer..."
    },
    {
      "role": "assistant",
      "content": "REASONING: Based on the answer about [element]:\n\nRULED IN:\n• [Condition]: [reason] (+XX%)\n\nCURRENT DIFFERENTIAL:\n1. [Condition]: XX% likelihood\n\nNEXT STEP: Ask about next element..."
    },
    // ... more OLD CARTS questions with reasoning ...
    {
      "role": "assistant",
      "content": "FINAL DIAGNOSTIC REASONING:\n\nTOP DIAGNOSIS: [Condition] (XX% confidence)\n\nREASONING PATH:\n1. Started with differential...\n2. Each answer progressively narrowed...\n3. Pattern recognition identified...\n\nCONCLUSION: The symptom pattern most strongly supports [Condition]."
    }
  ]
}
```

## Reasoning Format

### Interleaved Reasoning (After Each Answer)
```
REASONING: Based on the answer about [element]:

RULED IN (increased likelihood):
• [Condition]: [Element] '[pattern]' matches pattern (+XX%)

RULED OUT:
• [Condition]: Answer contradicts pattern (excludes '[pattern]')

CURRENT DIFFERENTIAL (top 3):
1. [Condition]: XX% likelihood
2. [Condition]: XX% likelihood
3. [Condition]: XX% likelihood

NEXT STEP: Ask about the next OLD CARTS element to further narrow the differential.
```

### Final Diagnostic Reasoning
```
FINAL DIAGNOSTIC REASONING:

Based on the complete OLD CARTS assessment:

TOP DIAGNOSIS: [Condition] (XX% confidence)

REASONING PATH:
1. Started with differential: [Condition1], [Condition2], [Condition3]...
2. Each answer progressively narrowed the differential:
   • [Condition]: XX% likelihood
   • [Condition]: XX% likelihood
   • [Condition]: XX% likelihood

3. Pattern recognition identified [Condition] as most consistent
4. Key matching factors from OLD CARTS:
   • Location, character, timing, and aggravating/alleviating factors all support [Condition]

CONCLUSION: The symptom pattern most strongly supports [Condition].
This diagnosis is based on the systematic collection and analysis of OLD CARTS elements,
with each answer helping to rule in or rule out conditions in the differential diagnosis.
```

## Current Coverage

### Gastrointestinal (GI) System
- ✅ Acute Cholecystitis
- ✅ Acute Lower GI Bleed
- ✅ Acute Upper GI Bleed

*Note: Script is ready to generate conversations for all 11 GI conditions when needed.*

## How It Trains the LLM

1. **Question Sequencing**: Teaches the model to follow the strict sequence (empathy → chronicity → age → sex → OLD CARTS)

2. **Differential Diagnosis Logic**: Shows how to:
   - Start with a broad differential
   - Evaluate each answer against condition patterns
   - Rule in conditions that match
   - Rule out conditions that contradict
   - Progressively narrow the differential

3. **Confidence Scoring**: Demonstrates how to:
   - Assign confidence based on pattern matches
   - Update scores as more information is gathered
   - Cap confidence at realistic levels (100%)

4. **Reasoning Transparency**: Teaches the model to:
   - Explain why conditions are ruled in/out
   - Show the reasoning path
   - Provide clear diagnostic conclusions

## Usage

### Training
```python
# Use in train_medical_bot_colab.py
DATASET_PATH = "medical_sft_dataset_high_quality.json"
```

### Expansion
To add more organ systems:
1. Add guideline JSON files to `llm-medical-container/medical/guidelines/[ORGAN_SYSTEM]/`
2. Run `create_high_quality_medical_dataset.py`
3. The script will automatically detect and include new conditions

## Technical Details

### Matching Logic
- **Word-based matching**: Checks if key descriptive words from guidelines appear in patient answers
- **Match scoring**: Full phrase match = 100%, partial word matches scaled proportionally
- **Threshold**: Requires at least 50% word match to rule in a condition

### Score Calculation
- Location matches: +30% confidence
- Character matches: +25% confidence
- Timing matches: +20% confidence
- Aggravating/Alleviating matches: +15% confidence
- Scores capped at 100% maximum

### Answer Generation
- Extracts patient-friendly answers from guideline patterns
- Falls back to generic answers if pattern not found
- Ensures natural, conversational responses

## Quality Assurance

✅ **Format Validation**: Matches `medical_sft_dataset.json` format exactly  
✅ **Sequence Validation**: Strict empathy → chronicity → age → sex → OLD CARTS  
✅ **Reasoning Validation**: Every OLD CARTS answer has corresponding reasoning  
✅ **Final Diagnosis**: Every conversation ends with diagnostic conclusion  
✅ **Score Capping**: All confidence scores ≤ 100%  

## Future Enhancements

- [ ] Add cardiovascular conditions (AHA guidelines)
- [ ] Add respiratory conditions (ATS/ERS guidelines)
- [ ] Add neurological conditions
- [ ] Add dermatologic conditions
- [ ] Add genitourinary conditions
- [ ] Expand to 5-10 conversations per organ system
- [ ] Add more nuanced rule-out logic
- [ ] Include associated symptoms in reasoning

## Files

- `create_high_quality_medical_dataset.py`: Script to generate the dataset
- `medical_sft_dataset_high_quality.json`: The generated dataset
- `HIGH_QUALITY_DATASET_README.md`: This documentation


# Smart Dataset Features - Relevant OLD CARTS Questions

## Overview

The updated dataset generator (`generate_primary_care_dataset.py`) now intelligently determines which OLD CARTS questions are relevant for each diagnosis, skipping nonsensical questions like "Where is your high blood pressure located?"

## Key Features

### 1. **Relevance Detection**

The generator uses heuristics to determine which OLD CARTS elements are relevant:

- **Location (L)**: Only relevant for symptoms with a physical location (pain, rash, etc.)
  - ❌ Skip for: hypertension, hyperlipidemia, diabetes, fatigue, polyuria, etc.
  - ✅ Ask for: pain, headache, rash, skin conditions, etc.

- **Character (C)**: Only relevant for symptoms with sensory qualities
  - ❌ Skip for: hypertension, hyperlipidemia, polyuria, constipation, insomnia, etc.
  - ✅ Ask for: pain, discomfort, burning, itching, etc.

- **Radiation (R)**: Only relevant for symptoms that can spread
  - ❌ Skip for: hypertension, diabetes, fatigue, constipation, insomnia, etc.
  - ✅ Ask for: pain (can radiate), chest pain, back pain, etc.

- **Other elements** (Onset, Duration, Aggravating, Alleviating, Timing, Severity): Usually relevant for most conditions

### 2. **Skip Tags in Dataset**

When an OLD CARTS element is not relevant, the dataset includes a skip message:

```json
{
  "role": "assistant",
  "content": "[SKIP:L] This OLD CARTS element (where the symptom is located) is not relevant for Essential Hypertension and should be skipped.",
  "metadata": {
    "skip": true,
    "element": "L",
    "reason": "Not relevant for Essential Hypertension - where the symptom is located does not apply"
  }
}
```

This teaches the fine-tuned model to skip irrelevant questions.

### 3. **Diagnosis-Specific Questions**

Questions are now generated using the actual symptom/chief complaint rather than the diagnosis name:

- ✅ Good: "When did the headaches start?" (for hypertension with headache)
- ❌ Bad: "When did your high blood pressure start?" (confusing)

### 4. **Relevance Metadata**

Each conversation includes `relevant_oldcarts` metadata:

```json
{
  "relevant_oldcarts": {
    "O": true,
    "L": false,  // Not relevant for hypertension
    "D": true,
    "C": false,  // Not relevant for hypertension
    "A_aggravating": true,
    "A_alleviating": true,
    "R": false,  // Not relevant for hypertension
    "T": true,
    "S": true
  }
}
```

## Advanced Medical Navigator Integration

The `advanced_medical_navigator.py` has been updated to:

1. **Check relevance before asking**: Uses heuristics to skip irrelevant questions
2. **Respect skip patterns**: The fine-tuned model should learn to skip from training data
3. **Fallback logic**: Provides heuristics as backup if model doesn't learn skip patterns

### Skip Logic in Navigator

```python
def _should_skip_oldcarts_element(self, session, element, chief_complaint):
    """Skip irrelevant OLD CARTS elements based on chief complaint."""
    # Location - skip for systemic conditions
    if element == 'location':
        if 'hypertension' in chief_complaint.lower():
            return True  # Skip location for hypertension
    
    # Character - skip for non-sensory symptoms
    if element == 'character':
        if 'polyuria' in chief_complaint.lower():
            return True  # Skip character for polyuria
    
    # Radiation - skip for non-radiating symptoms
    if element == 'radiation':
        if 'fatigue' in chief_complaint.lower():
            return True  # Skip radiation for fatigue
    
    return False
```

## Training Benefits

When the model is fine-tuned on this dataset:

1. **Learns to skip**: The skip messages teach the model which questions don't make sense
2. **Natural questions**: Diagnosis-specific questions improve conversation quality
3. **Efficiency**: Fewer irrelevant questions = faster, better patient experience
4. **Clinical accuracy**: Only asks questions that make medical sense

## Example: Hypertension Conversation

**Before (Bad)**:
- "Where is your high blood pressure located?" ❌
- "What does your elevated blood pressure feel like?" ❌
- "Does your hypertension spread to other areas?" ❌

**After (Good)**:
- "When did the headaches start?" ✅
- "What makes the headaches worse?" ✅
- "On a scale from 1 to 10, how severe are the headaches?" ✅
- Skips location, character, and radiation questions ✅

## Usage

1. **Generate dataset**: Run `python3 generate_primary_care_dataset.py`
2. **Train model**: Use `train_medical_bot_colab.py` with the generated dataset
3. **Deploy**: The fine-tuned model will automatically skip irrelevant questions

## Future Enhancements

- Use LLM API to dynamically determine relevance (if OpenAI API available)
- Expand skip patterns based on more diagnoses
- Add more nuanced relevance detection (e.g., partial relevance)


# Element Inference Training Examples

## Overview

The dataset now includes **two types of training examples**:

1. **Standard Format**: Question → Answer → Reasoning (with element identified from question)
2. **Inference Format**: Answer only → Reasoning (model must infer element from answer)

## Example 1: Standard Format (with Question)

```json
{
  "role": "assistant",
  "content": "When did the chest pain start?"
},
{
  "role": "user",
  "content": "It started suddenly"
},
{
  "role": "assistant",
  "content": "CLINICAL REASONING: This is the Onset (O) - when the symptom started element of OLD CARTS.
ELEMENT IDENTIFICATION: The patient's answer 'It started suddenly' is being evaluated for the Onset (O) element.
CRITICAL: This is Onset (O), NOT any other OLD CARTS element...
Patient reported 'It started suddenly' for ONSET (O)..."
}
```

**Key Point**: Element is identified from the question context ("When did it start?" → Onset)

---

## Example 2: Element Inference Format (Answer Only)

```json
{
  "role": "system",
  "content": "...CRITICAL: You must identify which OLD CARTS element the patient's answer corresponds to based on the answer content and conversation context."
},
{
  "role": "user",
  "content": "I have severe anemia"
},
{
  "role": "assistant",
  "content": "I understand you're experiencing severe anemia. I'm here to help."
},
{
  "role": "user",
  "content": "started suddenly"
},
{
  "role": "assistant",
  "content": "CLINICAL REASONING: Based on the patient's answer 'started suddenly', this corresponds to the Onset (O) - when the symptom started element of OLD CARTS.

ELEMENT INFERENCE: The answer 'started suddenly' indicates this is the Onset (O) element because:
- The answer describes when the symptom started (e.g., 'started suddenly' indicates timing of onset)
- Keywords like 'started', 'began', 'came on' indicate onset information

ELEMENT IDENTIFICATION: The patient's answer 'started suddenly' is being evaluated for the Onset (O) element.
CRITICAL: This is Onset (O), NOT any other OLD CARTS element.

Patient reported 'started suddenly' for ONSET (O). This onset pattern is CLASSIC for Severe Anemia..."
}
```

**Key Point**: 
- ❌ **NO QUESTION** provided
- ✅ Model must **INFER** the element from the answer alone
- ✅ Explicit **ELEMENT INFERENCE** section explains HOW the element was identified
- ✅ Teaches model to recognize keywords and patterns

---

## Example 3: Location Inference

```json
{
  "role": "user",
  "content": "I have severe anemia"
},
{
  "role": "assistant",
  "content": "I understand you're experiencing severe anemia. I'm here to help."
},
{
  "role": "user",
  "content": "in the center of my chest"
},
{
  "role": "assistant",
  "content": "CLINICAL REASONING: Based on the patient's answer 'in the center of my chest', this corresponds to the Location (L) - where the symptom is located element of OLD CARTS.

ELEMENT INFERENCE: The answer 'in the center of my chest' indicates this is the Location (L) element because:
- The answer describes where the symptom is located (e.g., 'in the center of my chest' indicates anatomical location)
- Location answers typically include anatomical terms, body parts, or directional descriptors

ELEMENT IDENTIFICATION: The patient's answer 'in the center of my chest' is being evaluated for the Location (L) element.
CRITICAL: This is Location (L), NOT any other OLD CARTS element.

Patient reported 'in the center of my chest' for LOCATION (L)..."
}
```

**Key Point**: Model learns to identify Location from anatomical terms ("center of my chest", "right side", "lower abdomen", etc.)

---

## Example 4: Character Inference

```json
{
  "role": "user",
  "content": "I have severe anemia"
},
{
  "role": "assistant",
  "content": "I understand you're experiencing severe anemia. I'm here to help."
},
{
  "role": "user",
  "content": "sharp"
},
{
  "role": "assistant",
  "content": "CLINICAL REASONING: Based on the patient's answer 'sharp', this corresponds to the Character (C) - what the symptom feels like (NOT Distribution, NOT Intensity) element of OLD CARTS.

ELEMENT INFERENCE: The answer 'sharp' indicates this is the Character (C) element because:
- The answer describes what the symptom feels like (e.g., 'sharp' indicates sensory quality)
- Character answers describe sensations like sharp, dull, burning, pressure, etc.

ELEMENT IDENTIFICATION: The patient's answer 'sharp' is being evaluated for the Character (C) element.
CRITICAL: This is Character (C), NOT any other OLD CARTS element.

Patient reported 'sharp' for CHARACTER (C). CRITICAL: This is CHARACTER (what it feels like), NOT Distribution, NOT Intensity, NOT Location..."
}
```

**Key Point**: Model learns to identify Character from sensory descriptors ("sharp", "dull", "burning", "pressure", etc.)

---

## Training Statistics

- **Standard conversations**: 117 (with questions)
- **Inference examples**: 351 (3 per condition: onset, location, character)
- **Total**: 468 conversations

## Benefits

1. ✅ **Robustness**: Model works with or without explicit questions
2. ✅ **Context awareness**: Model learns to infer from answer content
3. ✅ **Keyword recognition**: Model learns patterns ("started" → Onset, "center" → Location, "sharp" → Character)
4. ✅ **Real-world applicability**: Handles cases where questions aren't explicitly stated

## System Prompt Difference

**Standard Format**:
```
"You are a medical professional... Clinical reasoning is provided ONLY for OLD CARTS elements..."
```

**Inference Format**:
```
"You are a medical professional... Clinical reasoning is provided ONLY for OLD CARTS elements...
CRITICAL: You must identify which OLD CARTS element the patient's answer corresponds to based on the answer content and conversation context."
```

The inference format adds explicit instruction to identify the element from the answer content.


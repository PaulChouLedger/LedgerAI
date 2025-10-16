# Rolling Top-5 Differential Diagnosis System

## Overview

This system implements a **rolling differential diagnosis** approach that maintains a dynamic list of the top 5 most likely diagnoses, continuously updating as new information is gathered.

## Core Concept: Dynamic Top-5 List

Unlike rigid decision trees OR overwhelming LLM context with all guidelines, this maintains exactly **5 active differentials** at all times through rolling replacement.

### How It Works:

```
Chief Complaint: "abdominal pain"
↓
Match ALL guidelines with "abdominal pain" trigger
↓
Found: 20 GI conditions

Initial Ranking:
┌─────────────────────────────────────────┐
│ ACTIVE (Top 5):                         │
│  1. Acute Appendicitis      (0.60)      │
│  2. Gastroenteritis         (0.55)      │
│  3. Cholecystitis           (0.52)      │
│  4. Acute Pancreatitis      (0.50)      │
│  5. Peptic Ulcer            (0.48)      │
├─────────────────────────────────────────┤
│ RESERVE POOL:                           │
│  6. Bowel Obstruction       (0.45)      │
│  7. Diverticulitis          (0.42)      │
│  8. Ovarian Torsion         (0.40)      │
│  ...                                    │
│ 20. Mesenteric Ischemia     (0.25)      │
└─────────────────────────────────────────┘

Question 1: "When did it start?"
Answer: "2 days ago" (acute)
↓
Re-score ALL 5 active:
  Appendicitis: 0.60 → 0.70 ✅
  Cholecystitis: 0.52 → 0.68 ✅
  Pancreatitis: 0.50 → 0.65 ✅
  Gastroenteritis: 0.55 → 0.25 ❌ RULED OUT (< 0.3)
  Peptic Ulcer: 0.48 → 0.28 ❌ RULED OUT (< 0.3)

Rolling Update:
┌─────────────────────────────────────────┐
│ ACTIVE (Top 5):                         │
│  1. Appendicitis            (0.70)      │
│  2. Cholecystitis           (0.68)      │
│  3. Pancreatitis            (0.65)      │
│  4. Bowel Obstruction       (0.45) ← NEW│
│  5. Diverticulitis          (0.42) ← NEW│
├─────────────────────────────────────────┤
│ RULED OUT:                              │
│  - Gastroenteritis          (0.25)      │
│  - Peptic Ulcer             (0.28)      │
└─────────────────────────────────────────┘

Question 2: "Where is the pain?"
Answer: "right lower side"
↓
Re-score:
  Appendicitis: 0.70 → 0.95 ✅✅ (RLQ!)
  Diverticulitis: 0.42 → 0.58 ✅
  Ovarian Torsion: 0.40 → 0.55 ← From reserve
  Cholecystitis: 0.68 → 0.28 ❌ (RUQ, not RLQ)
  Pancreatitis: 0.65 → 0.25 ❌ (epigastric)

Continue until diagnosis clear...
```

## Key Advantages

### 1. **Scalable**
- Works with 1 guideline or 200 guidelines
- Always processes exactly 5 at a time
- O(5) complexity regardless of total guidelines

### 2. **Progressive Narrowing**
- Systematically rules out conditions
- Brings in new possibilities as others eliminated
- Mimics real clinical reasoning

### 3. **Intelligent Question Selection**
- Picks questions that appear in MULTIPLE top differentials
- Prioritizes high-value questions
- Focuses on top diagnosis when clear leader emerges

### 4. **Data-Driven**
- All logic from JSON guidelines
- No hardcoded decision trees
- Easy to add new conditions

## Question Selection Algorithm

**How the system picks the next question:**

```python
For each unanswered question:
  score = 0
  
  # How many top differentials ask this? (breadth)
  score += num_differentials_asking * 0.3
  
  # What's the diagnostic value? (importance)
  score += avg_diagnostic_value_weight
  
  # Is it critical for #1 diagnosis? (priority)
  if top_diagnosis_considers_it_critical:
    score += 0.5
  
→ Ask highest-scoring question
```

**Example:**

```
Active differentials:
  1. Appendicitis
  2. Cholecystitis  
  3. Pancreatitis

Potential questions:
  "Fever?" 
    - 3 guidelines ask (0.9)
    - avg_value: moderate (0.10)
    - Score: 1.0
  
  "Pain location?"
    - 3 guidelines ask (0.9)
    - avg_value: critical (0.30)
    - top diagnosis: critical (+0.5)
    - Score: 1.7 ← WINNER!

→ Asks about location (most discriminating)
```

## Example Conversation

```
You: "I have abdominal pain"

[Matched: 1 guideline]
[Active: Appendicitis (0.50)]

System: "When did this pain start?"

You: "two days ago"

[Onset: acute_days]
[Appendicitis: 0.60]
[Questions: 1/4]

System: "Did the pain start in one place and move to another?"

You: "yes, from center to right lower side"

[Migration: periumbilical_to_RLQ] ← Matches guideline!
[Appendicitis: 0.90]
[Questions: 2/4]

System: "How would you describe the pain?"

You: "sharp and constant"

[Quality: sharp] ← Matches guideline!
[Appendicitis: 0.95]
[Questions: 3/4]

System: "Have you had any nausea or loss of appetite?"

You: "yes, can't eat"

[Appetite: loss] ← Matches guideline!
[Appendicitis: 0.98]
[Questions: 4/4] ← Minimum met!
[Score > 0.90] ← High confidence!

System: "Based on your symptoms - pain started two days ago, 
         yes from center to right lower side, sharp and constant,
         with nausea/loss of appetite - this is likely 
         Acute Appendicitis.
         
         ⚠️ This requires prompt medical attention. 
         Go to the emergency room immediately."
```

## Configuration

```python
MAX_ACTIVE = 5                # Keep top 5 differentials
RULE_OUT_THRESHOLD = 0.3      # Score < 0.3 → ruled out
MIN_QUESTIONS = 4             # Minimum questions before diagnosis
HIGH_CONFIDENCE = 0.90        # Score for diagnosis with min questions
GOOD_CONFIDENCE = 0.80        # Score for diagnosis after min questions
```

## Scaling Example

### With 1 Guideline:
```
Active: [Appendicitis]
Reserve: []
→ Works, asks 4 questions, diagnoses
```

### With 20 GI Guidelines:
```
Active: [Top 5]
Reserve: [Next 15]
→ As #4 ruled out, #6 promoted
→ Always 5 active until exhausted
```

### With 160 Guidelines (All Systems):
```
Chief complaint "abdominal pain" matches 18 GI conditions
Active: [Top 5]
Reserve: [Next 13]
→ Progressive narrowing
→ Scalable!
```

## Clinical Accuracy

### The "Left to Right" Problem:

**User said:** "started mostly on the left side and it went to the right side"

**Guideline expects:** "periumbilical to RLQ" (center → right lower)

**Current system:** Accepts as match (too loose!)

**Better approach needed:** 
- LLM should detect this mismatch
- Ask clarifying question
- Distinguish "left→right" from "center→right lower"

### Solution: Hybrid Approach

1. **Use rolling top-5** for scalability ✅
2. **Use fuzzy matching** for small variations ✅
3. **Add LLM validation** for critical findings:

```python
# After user answers critical question:
if diagnostic_value == 'critical':
    # Have LLM check if answer truly matches expected
    llm_validation = f"""
    Expected response: "periumbilical to RLQ"
    User said: "left side to right side"
    
    Does this match the expected migration pattern?
    Answer: yes/no + reason
    """
    
    if not validated:
        ask_clarifying_question()
```

## Summary

**This rolling top-5 system is:**
- ✅ Scalable (constant 5 active)
- ✅ Progressive (narrows systematically)
- ✅ Data-driven (from JSON)
- ✅ Flexible (fuzzy matching)

**To make it more "conversational":**
- Add LLM validation for critical findings
- Allow LLM to ask follow-up clarifications
- Use LLM to detect inconsistencies

**Best of both worlds:**
- Structured differential management (scalable)
- LLM intelligence (conversational)


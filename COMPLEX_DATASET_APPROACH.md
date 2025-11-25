# Complex Medical Dataset - Generation Approach

## Overview

This dataset focuses on **complex presentations** that require cross-organ system differentiation, clarification questions, and progressive scoring/ranking systems.

## Key Features

### 1. Cross-Organ System Differentiation
- **Chest Pain**: Differentiates between GERD (GI) vs Cardiac conditions
- **Abdominal Pain**: Differentiates between RUQ (cholecystitis) vs RLQ (appendicitis)
- **Flank Pain**: Differentiates between Renal vs GI/musculoskeletal

### 2. Clarification Questions
- When location answers are ambiguous (e.g., "right side"), the system asks:
  - "Is the pain in your right upper abdomen (near your ribs/liver area) or right lower abdomen (near your hip/appendix area)?"
- Trains the LLM to recognize when answers need clarification

### 3. Progressive Scoring & Ranking System
- **Initial State**: All conditions start at balanced probability (e.g., 5 conditions = 20% each)
- **After Each Answer**: LLM evaluates how the answer affects each condition
- **Score Updates**: Conditions receive deltas (+/-) based on answer characteristics
- **Rankings Update**: After each OLD CARTS element, rankings are updated and displayed
- **Rolling System**: Each answer progressively narrows the differential diagnosis

### 4. Associated Symptoms Questions
- **After OLD CARTS**: System asks associated symptom questions based on top 3 conditions
- **Differentiation Focus**: Questions designed to differentiate between the most likely diagnoses
- **LLM Knowledge-Based**: Examples include:
  - Acute MI: sweating, nausea, shortness of breath
  - GERD: sour taste, regurgitation
  - Cholecystitis: nausea, vomiting, fever
  - Appendicitis: nausea, loss of appetite, fever
  - Nephrolithiasis: hematuria, frequent urination
- **Scoring Updates**: Each associated symptom answer updates condition probabilities

### 5. LLM Internal Reasoning
Each clinical reasoning section includes:
- Element identification
- Answer sufficiency assessment
- Scoring analysis (which conditions increased/decreased)
- Current ranked differential diagnosis with probabilities
- Next steps

## Dataset Structure

### Complex Cases Defined

1. **Chest Pain - GERD vs Cardiac**
   - Differentiates: GERD, MI, Angina, Pericarditis, Aortic Dissection
   - Demonstrates how burning character, worse with lying down, better with antacids → GERD
   - Demonstrates how pressure, worse with exertion → Cardiac

2. **Chest Pain - Cardiac**
   - Shows cardiac presentation with pressure, exertion-aggravated
   - Demonstrates differentiation from GERD

3. **Abdominal Pain - RUQ vs RLQ Clarification**
   - Initial ambiguous answer: "on my right side"
   - Clarification question asked
   - Clarified answer narrows to RUQ → Cholecystitis

4. **Abdominal Pain - RLQ**
   - Initial ambiguous answer: "on my right side"
   - Clarification question asked
   - Clarified answer narrows to RLQ → Appendicitis

5. **Flank Pain - Renal vs GI**
   - Demonstrates renal vs other systems differentiation

## Scoring System

### Score Deltas by Pattern

**Chest Pain Patterns:**
- `burning` → GERD +0.2, Cardiac conditions -0.1 to -0.2
- `pressure/heaviness` → Cardiac +0.2, GERD -0.2
- `worse lying down` → GERD +0.3, Cardiac -0.2
- `worse exertion` → Cardiac +0.2, GERD -0.2

**Abdominal Pain Patterns:**
- `RUQ location` → Cholecystitis +0.3, Appendicitis -0.3
- `RLQ location` → Appendicitis +0.3, Cholecystitis -0.3

### Progressive Ranking Example

```
Initial: All conditions at 20%
  ↓ After Onset answer
GERD: 36.8% (+16.8%), Others: 15.8% each
  ↓ After Character answer (burning)
GERD: 73.2% (+36.4%), Others: 6.7% each
  ↓ After Aggravating answer (lying down)
GERD: 89.5% (+16.3%), Others: 2.6% each
  ↓ After Alleviating answer (antacids)
GERD: 95.2% (+5.7%), Others: 1.2% each
Final: GERD: 97.5%, Others: 0.6% each
```

## Conversation Flow

1. **Chief Complaint** (symptom, not diagnosis)
2. **Acknowledgment**
3. **Chronicity Question**
4. **Demographics** (age, sex)
5. **OLD CARTS Elements** (only relevant ones):
   - Question
   - Patient Answer (may be ambiguous)
   - Clarification Question (if needed)
   - Clarified Answer (if clarification occurred)
   - Clinical Reasoning with:
     - Element identification
     - Scoring analysis
     - **Updated ranked differential diagnosis**
     - Next steps
6. **Associated Symptom Questions** (1-3 questions based on top 3 conditions)
   - Question
   - Patient answer
   - Clinical reasoning with updated rankings
7. **Final Diagnostic Reasoning** with complete ranked differential

## Training Goals

This dataset trains the LLM to:

1. **Recognize complex presentations** requiring cross-system thinking
2. **Ask clarifying questions** when answers are ambiguous
3. **Progressive scoring** - update rankings after each answer
4. **Build differential diagnosis** through systematic OLD CARTS collection
5. **Use internal reasoning** to evaluate how each answer affects condition probabilities
6. **Skip irrelevant OLD CARTS elements** based on chief complaint

## Usage

```bash
# Generate with 3 conversations per case (default)
python3 generate_complex_dataset.py

# Generate with custom number
python3 generate_complex_dataset.py 5
```

## Output

Each conversation includes:
- Complete OLD CARTS assessment
- Progressive score updates after each answer
- Ranked differential diagnosis at each step
- Clarification questions when needed
- Final ranked differential diagnosis

The dataset is designed to train the LLM to think like a clinician, building and refining the differential diagnosis through systematic history-taking.


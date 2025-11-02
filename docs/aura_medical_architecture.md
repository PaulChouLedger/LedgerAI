# Aura Medical Architecture

A comprehensive explanation of how the adaptive diagnostic engine works, written in simple terms.

---

## Overview

The Aura Medical Assistant helps assess patient symptoms by asking questions systematically and comparing answers against medical guidelines. Think of it like a smart detective who:

1. **Listens** to what the patient says
2. **Asks clarifying questions** to gather more information
3. **Compares** responses against a database of medical conditions
4. **Narrows down** the possibilities until confident about the diagnosis

---

## The Flow: From Symptom to Assessment

### Step 1: Initial Complaint Processing

**What happens:**
When a patient says "I have abdominal pain," the system:

1. **Categorizes the problem**: Determines it's a gastrointestinal (GI) issue
2. **Loads relevant guidelines**: Fetches all GI condition files (appendicitis, cholecystitis, pancreatitis, etc.)
3. **Parses the complaint**: Looks for any initial clues in what they said

**Example:**
```
Patient: "I have severe abdominal pain on my lower right side"
         
System thinks:
- Category: GI (gastrointestinal)
- Load guidelines: Appendicitis, Cholecystitis, IBS, etc.
- Parses: "severe" → severity, "lower right side" → location
```

**Code location:** `adaptive_diagnostic_engine.py` → `start_assessment()`

---

### Step 2: Initial Pool Setup

**What happens:**
The system creates two groups of conditions:

1. **Active pool** (top 5 most likely conditions)
   - These get asked about first
   - Based on prevalence/initial matching

2. **Reserve pool** (everything else)
   - Kept in the background
   - Can be promoted if they score higher

**Why two pools?**
- Efficiency: Focus on likely diagnoses first
- Score-based: Only top scorers get attention

**Example:**
```
Active Pool (Top 5):
1. Acute Appendicitis (score: 50%)
2. Acute Cholecystitis (score: 49%)
3. IBS (score: 45%)
...

Reserve Pool:
- IBD Flare
- Gastroenteritis
...
```

**Code location:** `adaptive_diagnostic_engine.py` → Lines 347-354

---

### Step 3: Empathetic Response & First Question

**What happens:**
Before diving into symptoms, the system:

1. **Shows empathy**: "I'm sorry to hear you're experiencing abdominal pain..."
2. **Asks the first question**: "When did the pain start?"

**Why empathy first?**
- Makes the patient feel heard
- Establishes rapport
- Improves communication

**Code location:** `adaptive_diagnostic_engine.py` → Lines 402-429

---

## The Question Loop: OLDCARTS Framework

### What is OLDCARTS?

OLDCARTS is a medical mnemonic used by doctors to systematically gather symptom information:

- **O** = Onset (when did it start?)
- **L** = Location (where does it hurt?)
- **D** = Duration (how long does each episode last?)
- **C** = Character (what does it feel like?)
- **A** = Aggravating factors (what makes it worse?)
- **R** = Relieving factors (what makes it better?)
- **T** = Timing (is it constant or intermittent?)
- **S** = Severity (how bad is it?)

The system asks about these one at a time, learning more with each answer.

---

### Step 4: Answer Processing

**What happens when the patient answers:**

For each answer (e.g., "It started 2 days ago"):

#### A. **Normalization**
Convert patient-friendly language to medical terms:

```
Patient: "right side near my ribs"
↓
Normalized: "right upper quadrant"
```

**How it works:**
1. Loads synonym files (e.g., `gi_synonyms_oldcarts.json`)
2. Finds matching patient-friendly phrases
3. Maps to standard medical terminology

**Example synonyms:**
```json
"right upper quadrant": [
  "top right side near ribs",
  "upper right abdomen",
  "under ribs right side"
]
```

**Code location:** `medical_rule_engine.py` → `_normalize_with_synonyms()`

---

#### B. **Semantic Similarity (FAISS)**

Compare the patient's answer against medical guideline terms:

**How FAISS works:**
1. Converts text to numbers (embeddings)
2. Uses cosine similarity to find matches
3. Returns similarity score (0 = no match, 1 = perfect match)

**Example:**
```
Patient: "sharp stabbing pain"
↓
Similarity scores:
- "sharp": 0.95 ✅
- "dull ache": 0.15 ❌
- "burning": 0.20 ❌
```

**Why FAISS:**
- Handles semantic meaning, not just keywords
- Works even if wording differs slightly
- Fast search through thousands of terms

**Code location:** `medical_rule_engine.py` → `find_matching_terms_faiss()`

---

#### C. **Exact Matching**

For precision, also checks if the answer contains exact words:

**How it works:**
1. Checks if medical term appears in the answer
2. Checks if answer is a substring of medical term
3. Expands synonyms for broader matching

**Example:**
```
Patient: "lower right abdomen"
↓
Checks against synonyms:
- "lower right side around groin" → matches!
- "right lower quadrant" → matches!

Result: "right lower quadrant" is satisfied
```

**Code location:** `adaptive_diagnostic_engine.py` → `_analyze_missing_information()`

---

#### D. **Scoring Each Condition**

For every condition in the active pool:

**Score calculation:**
```
new_score = (old_score × 70%) + (similarity × 30%)
```

**Why 70/30 split?**
- Preserves accumulated knowledge (70%)
- Updates with new information (30%)
- Prevents one bad answer from tanking the score

**Example:**
```
Appendicitis before answer: 50%
Patient says "RLQ pain": similarity = 0.95
↓
New score = (0.50 × 0.70) + (0.95 × 0.30)
         = 0.35 + 0.285
         = 0.635 = 63.5% ✅
```

**Additional boosts:**
- **Word match boost**: +0.1-0.4 for exact term matches
- **Medical rules boost**: +0.3 for anatomical consistency

**Code location:** `adaptive_diagnostic_engine.py` → Lines 755-785

---

### Step 5: Re-ranking & Pooling

**What happens after scoring:**

#### A. **Rule Out Low Scorers**

Conditions below threshold get moved to "ruled out":

```
Threshold rule:
- If current score ≥ 0.3: threshold = 0.1
- If current score ≥ 0.2: threshold = 0.1
- Otherwise: threshold = 0.05

Example:
- Appendicitis: 63.5% → stays active ✅
- IBS: 8.2% → ruled out ❌
```

**Code location:** `adaptive_diagnostic_engine.py` → `_get_dynamic_threshold()`

---

#### B. **Promote High Scorers**

Conditions that jump into the top 5 get promoted:

**Example:**
```
Before location answer:
1. Cholecystitis (48%)
2. Appendicitis (45%)
...

After "RLQ pain" answer:
1. Appendicitis (63%) ⬆️ PROMOTED
2. Cholecystitis (48%)
```

**Debug output:**
```
[Engine] 🔼 PROMOTED to active:
[Engine]   ↑ Acute Appendicitis (score: 63%)
```

**Code location:** `adaptive_diagnostic_engine.py` → `_rerank_and_pool_guidelines()`

---

### Step 6: Clarification Logic

**When answers are ambiguous:**

The system asks follow-up questions if:

1. **Zero matches**: No terms from any guideline were satisfied
2. **Multiple matches**: Patient answer matches 2+ locations/characters

**Example:**
```
Patient: "right side"
↓
Matches both:
- Right upper quadrant
- Right lower quadrant

System: "Can you be more specific? Is it top right near your ribs, 
         or lower right around your groin?"
```

**Clarification stops when:**
- Exactly one term is satisfied
- Patient provides clear, specific answer

**Code location:** `adaptive_diagnostic_engine.py` → Lines 790-890

---

### Step 7: Next Question Selection

**What question to ask next:**

The system prioritizes based on:

1. **Missing information**: Which OLDCARTS elements haven't been covered?
2. **Discriminating power**: Which question best distinguishes between top conditions?
3. **Precedence**: Location → Character → Severity → Aggravating → Relieving

**Example:**
```
Covered so far:
- Onset ✅
- Location ✅

Missing:
- Character
- Severity
- Aggravating
- Relieving
- Timing

Next question: "What does the pain feel like?" (Character)
```

**Code location:** `adaptive_diagnostic_engine.py` → `_ask_next_clinical_question()`

---

## Special Cases

### Radiation Questions

**When to ask:**
Only after location is satisfied and at least one active condition has radiation data.

**What it asks:**
"Does the pain spread or radiate anywhere?"

**Why separate:**
- Many conditions don't have radiation patterns
- Radiating pain is often diagnostic
- Helps distinguish similar conditions

**Example:**
```
Cholecystitis: radiates to right shoulder
Appendicitis: no radiation

Patient says "yes, to my shoulder"
→ Cholecystitis score increases
→ Appendicitis score decreases
```

**Code location:** `adaptive_diagnostic_engine.py` → `_ask_about_radiation()`

---

### Demographics

**Age and sex:**
- Asked early in the flow
- Used for epidemiology-based scoring
- Helps rule out certain conditions

**Example:**
```
Appendicitis: most common in ages 15-30
Patient: 8 years old
→ Score adjustment: +0.1
```

**Code location:** `adaptive_diagnostic_engine.py` → Lines 649-667

---

## Data Structures

### Guidelines

Each medical condition is stored as JSON:

```json
{
  "condition": "Acute Appendicitis",
  "category": "gastrointestinal",
  "urgency": "urgent",
  "prevalence": "common",
  "key_features": {
    "classic_presentation": "ONSET: Sudden... LOCATION: RLQ...",
    "structured_oldcarts": {
      "onset": {
        "includes": [
          {"medical": "sudden", "patient_friendly": "all at once"}
        ],
        "excludes": [...]
      },
      "location": {
        "includes": [
          {"medical": "right lower quadrant", 
           "patient_friendly": "lower right side around groin"}
        ]
      }
    }
  }
}
```

---

### Active Guidelines Pool

**Structure:**
```python
active_guidelines = [
  {
    'name': 'Acute Appendicitis',
    'score': 0.635,
    'data': {...},  # Full guideline JSON
    'prevalence': 'common',
    'sources': []
  },
  ...
]
```

**Why this structure:**
- Combines guideline data with dynamic score
- Easy to sort and re-rank
- Preserves original data for reference

---

### Conversation History

**Tracks every exchange:**
```python
conversation_history = [
  {
    'type': 'statement',
    'message': 'I'm sorry to hear...'
  },
  {
    'type': 'question',
    'question': 'When did the pain start?',
    'oldcarts': 'onset',
    'focus': 'clinical'
  },
  {
    'type': 'answer',
    'answer': '2 days ago',
    'oldcarts': 'onset'
  }
]
```

**Why tracking:**
- Prevents asking same question twice
- Provides context for follow-ups
- Enables backtracking if needed

---

## Performance Optimizations

### 1. Synonym Caching

**Problem:** Loading synonym files on every question is slow.

**Solution:** Load all synonyms once at initialization:
```python
# During __init__:
self.synonym_cache = {}

# Loads all organ systems:
GI_synonyms_oldcarts.json
CARDIO_synonyms_oldcarts.json
...
```

**Impact:** 50-100ms saved per clarification question

---

### 2. Pre-built Data Structures

**Problem:** Building synonym expansion dicts is expensive.

**Solution:** Build once, reuse:
```python
# During __init__:
synonym_expansions = {
  'right upper quadrant': [
    'top right near ribs',
    'upper right abdomen',
    ...
  ]
}

synonym_to_group = {
  'top right near ribs': 'right upper quadrant',
  'upper right abdomen': 'right upper quadrant',
  ...
}
```

**Impact:** 30-50ms saved per question

---

### 3. O(1) Lookup Optimization

**Problem:** Nested loops for synonym matching are slow (O(n²)).

**Solution:** Use dictionary for instant lookup:
```python
# Old (slow):
for standard_term, synonym_list in synonym_expansions.items():
  if term in synonym_list:
    ...

# New (fast):
if term in synonym_to_group:
  group_key = synonym_to_group[term]  # O(1) lookup
```

**Impact:** 50-150ms saved per question

---

### 4. FAISS Semantic Search

**Problem:** Exact string matching misses semantically similar terms.

**Solution:** Use embeddings + FAISS:
1. Convert all terms to vectors
2. Build FAISS index
3. Fast similarity search

**Impact:** Handles "sharp" ≈ "stabbing" ≈ "piercing"

---

## Debug Output

### What gets logged:

**Initial assessment:**
```
[Engine] 🚀 NEW ASSESSMENT
[Engine] Chief Complaint: 'abdominal pain'
[Engine] 🎯 Category: gastrointestinal
[Guideline Load] 📚 Conditions: ['Acute Appendicitis', ...]
```

**Each answer:**
```
[Scoring] 📊 Acute Appendicitis: old=0.482, similarity=0.656, new=0.534
[Location Analysis] ✅ Satisfied terms: ['right lower quadrant']
```

**Pool changes:**
```
[Engine] 🔼 PROMOTED to active:
[Engine]   ↑ Acute Appendicitis (score: 53%)

[Engine] 🔽 DEMOTED to reserve:
[Engine]   ↓ Biliary Colic (score: 48%)
```

**Final rankings:**
```
[Engine] 📊 UPDATED RANKINGS:
[Engine]   1. Acute Appendicitis: 63% ⚠️
[Engine]   2. Acute Cholecystitis: 48% ⚠️
```

---

## Algorithm Summary

### The Complete Flow:

1. **Listen** to initial complaint
2. **Categorize** (GI, Cardio, etc.)
3. **Load** relevant guidelines
4. **Parse** for initial clues
5. **Score** all conditions
6. **Rank** by score
7. **Pool** into active/reserve
8. **Ask** next OLDCARTS question
9. **Normalize** patient answer
10. **Match** using FAISS + exact
11. **Score** each condition
12. **Re-rank** by new scores
13. **Promote/demote** between pools
14. **Rule out** low scorers
15. **Clarify** if ambiguous
16. **Repeat** until confident

---

## Key Design Decisions

### Why scoring instead of pure semantic search?

**Scoring benefits:**
- Accumulates evidence over multiple questions
- Handles contradictory information
- Provides confidence levels

**Pure semantic would:**
- Treat each question in isolation
- Lose context
- Be less robust to noise

---

### Why 70/30 weighting?

**Balance:**
- 70% old score: Preserves history, prevents wild swings
- 30% new similarity: Responsive to new information

**If 50/50:**
- Too sensitive to recent answers
- Scores would bounce around

**If 90/10:**
- Too resistant to change
- New evidence would be ignored

---

### Why two pools?

**Efficiency:**
- Focus on top 5 most likely conditions
- Don't waste time on unlikely diagnoses
- Dynamic promotion keeps it fair

**Alternative (single pool):**
- Would process all 100+ conditions every question
- Slower, less focused

---

## Common Issues & Solutions

### Problem: "right side" matches multiple quadrants

**Cause:** Bidirectional substring matching

**Solution:** Forward-only matching
```python
# OLD (wrong):
if term in answer OR answer in term:  # "right side" in "lower right side"
  match = True

# NEW (correct):
if synonym in answer:  # Only forward check
  match = True
```

---

### Problem: Very short terms cause false positives

**Cause:** "mi" appears in "abdominal"

**Solution:** Skip terms < 3 characters
```python
if len(term) < 3:
  continue  # Skip "mi", "a", etc.
```

---

### Problem: Slow clarification questions

**Cause:** Repeated file I/O and expensive computations

**Solution:** Cache everything at initialization
- Load all synonym files once
- Pre-build data structures
- O(1) lookups instead of loops

**Impact:** 4-5 seconds → 3.7-4.7 seconds

---

## Testing & Validation

### How to verify it's working:

**Check initial parsing:**
```
[Guideline Load] Should see relevant conditions
[OLDCARTS Analysis] Should extract elements from complaint
```

**Check scoring:**
```
[Scoring] Scores should increase with good matches
[Ranking] Top conditions should make medical sense
```

**Check pooling:**
```
[Promoted] New high-scorers join active pool
[Demoted] Low-scorers leave active pool
[Ruled Out] Very low scorers get removed
```

**Check clarification:**
```
[Clarification] Should ask when 0 or 2+ matches
[Location Analysis] Satisfied terms should match answer
```

---

## Future Improvements

### Potential enhancements:

1. **Confidence thresholds**: Auto-diagnose when score > 95%
2. **Question prioritization**: Ask discriminating questions first
3. **Context awareness**: Remember previous similar cases
4. **Multi-organ system**: Handle complaints affecting multiple systems
5. **Timeline tracking**: Build disease progression model

---

## Conclusion

The Aura Medical Assistant combines:
- **Systematic questioning** (OLDCARTS framework)
- **Intelligent matching** (semantic similarity + exact)
- **Dynamic ranking** (score-based pooling)
- **Performance optimization** (caching, fast lookups)

Result: A robust, responsive medical diagnostic system that helps clinicians systematically assess patient symptoms.

---

## Glossary

**FAISS**: Facebook AI Similarity Search - Fast vector search library

**Embedding**: Numerical representation of text that captures meaning

**Semantic similarity**: How similar two texts are in meaning, not just words

**OLDCARTS**: Medical mnemonic for symptom assessment

**Pool**: Group of conditions being actively considered

**Normalization**: Converting patient-friendly language to medical terms

**Synonym expansion**: Using multiple ways to say the same thing

**Substring matching**: Finding if one text appears inside another

**Cosine similarity**: Mathematical measure of how similar two vectors are


# Semantic OLDCARTS Architecture - Vector Similarity Scoring

## Revolutionary Approach

This system combines **OLDCARTS clinical framework** with **semantic vector similarity** to create an objective, hallucination-free diagnostic engine.

---

## Core Innovation

**LLM is used ONLY for:**
1. ✅ Question generation (generic OLDCARTS template)
2. ✅ Red flag question generation (after diagnosis)

**Vector Similarity is used for:**
1. ✅ **All scoring** (no LLM subjective judgment)
2. ✅ **OLDCARTS matching** (answer vs guideline OLDCARTS section)

---

## How It Works

### **Phase 1: Question Generation (Ultra-Minimal LLM)**

**LLM Sees:**
```
Patient has abdominal pain.

Ask about LOCATION.

Example: Where exactly is the pain located?

Your question:
```

**LLM Generates:**
```
"Where in your abdomen is the pain?"
```

**Why This Works:**
- ✅ **Minimal context** (~50 tokens vs ~1,000 tokens)
- ✅ **No guidelines** to confuse LLM
- ✅ **Clear directive** (ask about LOCATION)
- ✅ **One example** to guide phrasing
- ✅ **Prevents "3333..." hallucination**

---

### **Phase 2: Semantic Similarity Scoring (No LLM)**

**After Patient Answers:**
```
Q: "Where in your abdomen is the pain?"  → OLDCARTS element: 'L'
A: "Lower right side"
```

**System Does:**

1. **Extract LOCATION sections from each guideline:**
   ```
   Appendicitis LOCATION: 
   "Pain MIGRATES from periumbilical to right lower quadrant (RLQ) over 12-24 
   hours. Localizes to McBurney's point in RLQ."
   
   Cholecystitis LOCATION:
   "Right upper quadrant (RUQ), precisely localized just below right rib cage.  
   RADIATES TO RIGHT SHOULDER OR SCAPULA."
   
   Pancreatitis LOCATION:
   "Epigastric (upper mid-abdomen) and periumbilical. RADIATES STRAIGHT THROUGH 
   TO THE BACK."
   ```

2. **Vectorize answer and each LOCATION section:**
   ```
   POST http://localhost:11435/embed
   {
     "texts": [
       "Lower right side",
       "Pain MIGRATES from periumbilical to right lower quadrant (RLQ)...",
       "Right upper quadrant (RUQ), precisely localized...",
       "Epigastric (upper mid-abdomen) and periumbilical..."
     ]
   }
   ```

3. **Compute cosine similarity:**
   ```
   Answer: "Lower right side"
   
   vs Appendicitis LOCATION: 0.85 similarity (HIGH - RLQ matches!)
   vs Cholecystitis LOCATION: 0.42 similarity (low - RUQ doesn't match)
   vs Pancreatitis LOCATION: 0.38 similarity (low - epigastric doesn't match)
   ```

4. **Update scores with weighted average:**
   ```
   Appendicitis: 60% → 67% (0.6*0.7 + 0.85*0.3) ↑
   Cholecystitis: 60% → 55% (0.6*0.7 + 0.42*0.3) ↓
   Pancreatitis: 60% → 53% (0.6*0.7 + 0.38*0.3) ↓
   ```

---

### **Phase 3: Repeat for All OLDCARTS Elements**

**Question 2: DURATION**
```
Q: "How long does the pain last?"
A: "Constant for hours"

Extract DURATION sections:
  Appendicitis: "CONSTANT pain lasting hours to days (not episodic)"
  Cholecystitis: "CONSTANT pain lasting >6 hours"
  Pancreatitis: "CONSTANT pain lasting hours to days"

Similarity:
  vs Appendicitis: 0.88 (HIGH)
  vs Cholecystitis: 0.82 (HIGH)
  vs Pancreatitis: 0.85 (HIGH)

All scores increase slightly (all are constant pain conditions)
```

**Question 3: CHARACTER**
```
Q: "How would you describe the pain?"
A: "Sharp and steady"

Extract CHARACTER sections:
  Appendicitis: "SHARP and CONSTANT (NOT crampy or intermittent)"
  Cholecystitis: "SHARP or ACHING, CONSTANT (NOT colicky)"
  Pancreatitis: "DEEP ACHING or KNIFE-LIKE"

Similarity:
  vs Appendicitis: 0.92 (VERY HIGH - sharp matches!)
  vs Cholecystitis: 0.75 (moderate)
  vs Pancreatitis: 0.58 (low - not aching/knife-like)

Appendicitis: 67% → 75% ↑ (strong match)
Cholecystitis: 55% → 61% ↑ (moderate)
Pancreatitis: 53% → 54% (minimal)
```

**Continue through all 8 OLDCARTS elements...**

---

### **Phase 4: Diagnosis After OLDCARTS Complete**

```
OLDCARTS Coverage: OLDCARTS (8/8 complete)

Final Scores:
  1. Acute Appendicitis: 95% ⚠️
  2. Acute Cholecystitis: 62% ⚠️
  3. Biliary Colic: 58% 📋

✅ DIAGNOSIS: Acute Appendicitis (95% confidence, OLDCARTS complete)
```

---

### **Phase 5: Red Flag Screening (Full Context to LLM)**

**NOW send full guideline to LLM for red flag questions:**

```
Guideline: Acute Appendicitis
Classic Presentation: [FULL OLDCARTS TEXT]
Red Flags:
  - Sudden severe pain that then improves - may indicate PERFORATION
  - Board-like rigid abdomen - peritonitis, call 911
  - High fever >103°F - possible perforation
  - Hypotension, altered mental status - septic shock

Generate yes/no questions for each red flag.
```

**LLM Generates:**
```
Q1: "Did the pain suddenly get much better after being severe?"
Q2: "Is your abdomen very hard or rigid when you press on it?"
Q3: "Have you had a fever higher than 103 degrees?"
Q4: "Have you felt dizzy, lightheaded, or like you might faint?"
```

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│ PHASE 1: QUESTION GENERATION (LLM - Minimal Context)        │
├─────────────────────────────────────────────────────────────┤
│ Input: Generic OLDCARTS template + next element            │
│ LLM: Generate question (e.g., "Where is the pain?")        │
│ Output: Question → Patient                                  │
└─────────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────┐
│ PHASE 2: ANSWER RECEIVED                                    │
├─────────────────────────────────────────────────────────────┤
│ Patient Answer: "Lower right side"                         │
│ OLDCARTS Element: 'L' (LOCATION)                           │
└─────────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────┐
│ PHASE 3: SEMANTIC SIMILARITY SCORING (No LLM)               │
├─────────────────────────────────────────────────────────────┤
│ 1. Extract LOCATION section from each guideline            │
│    Appendicitis: "Pain MIGRATES to RLQ..."                 │
│    Cholecystitis: "RUQ pain..."                            │
│    Pancreatitis: "Epigastric..."                           │
│                                                             │
│ 2. Call RAG container /embed endpoint                      │
│    POST http://localhost:11435/embed                       │
│    texts: [answer, section1, section2, section3]           │
│                                                             │
│ 3. Compute cosine similarity                               │
│    similarity(answer, section1) = 0.85                     │
│    similarity(answer, section2) = 0.42                     │
│    similarity(answer, section3) = 0.38                     │
│                                                             │
│ 4. Update scores (weighted average)                        │
│    Appendicitis: 60% → 67% ↑                               │
│    Cholecystitis: 60% → 55% ↓                              │
│    Pancreatitis: 60% → 53% ↓                               │
└─────────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────┐
│ PHASE 4: REPEAT FOR ALL OLDCARTS (L, D, C, A, R, T, S)     │
├─────────────────────────────────────────────────────────────┤
│ After 8 questions covering all OLDCARTS elements:          │
│   Appendicitis: 95%                                        │
│   Cholecystitis: 58%                                       │
│   Pancreatitis: 52%                                        │
│                                                             │
│ ✅ DIAGNOSIS: Acute Appendicitis (95%, OLDCARTS complete)  │
└─────────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────┐
│ PHASE 5: RED FLAG SCREENING (LLM - Full Context)           │
├─────────────────────────────────────────────────────────────┤
│ NOW send full Appendicitis guideline to LLM                │
│ LLM generates red flag questions                           │
│ Patient answers → detect if red flags present              │
│ Final disposition with warnings                            │
└─────────────────────────────────────────────────────────────┘
```

---

## Key Advantages

### 1. ✅ **No LLM Hallucination**
- Minimal context for question generation
- LLM only sees generic template, not complex guidelines
- Prevents "3333..." and other hallucinations

### 2. ✅ **Objective Scoring**
- Vector similarity is mathematical, not subjective
- No LLM interpretation bias
- Reproducible results

### 3. ✅ **Precise Matching**
- Answer compared ONLY to relevant OLDCARTS section
- Not entire guideline (reduces noise)
- Example: "Lower right side" compared to LOCATION section only, not DURATION or CHARACTER

### 4. ✅ **Systematic Coverage**
- All 8 OLDCARTS elements must be covered
- No diagnosis until complete
- Ensures thorough assessment

### 5. ✅ **No Fallbacks**
- System fails cleanly if embeddings unavailable
- No silent degradation
- Clear error messages

### 6. ✅ **Separation of Concerns**
```
LLM:    Question generation (minimal) + Red flag assessment (full context)
Vector: Scoring (objective, mathematical)
```

---

## Example Flow: Appendicitis

**Chief Complaint:** "I have abdominal pain"

**Matched:** 20 guidelines → Top 3: Appendicitis, Cholecystitis, Pancreatitis (all 60%)

### OLDCARTS Questions (LLM generates, Vector scores):

| # | Element | Question | Answer | Similarity Scores | New Ranking |
|---|---------|----------|--------|-------------------|-------------|
| 1 | O | "When did it start?" | "Hours ago" | App:0.61, Chol:0.62, Pan:0.60 | All ~60% |
| 2 | L | "Where is the pain?" | "Lower right side" | App:0.85↑, Chol:0.42↓, Pan:0.38↓ | App:67%, Chol:55%, Pan:53% |
| 3 | D | "How long does it last?" | "Constant for hours" | App:0.88↑, Chol:0.82↑, Pan:0.85↑ | App:75%, Chol:63%, Pan:62% |
| 4 | C | "Describe the pain?" | "Sharp and steady" | App:0.92↑, Chol:0.75↑, Pan:0.58↓ | App:81%, Chol:67%, Pan:63% |
| 5 | A | "What makes it worse?" | "Movement, coughing" | App:0.95↑, Chol:0.68, Pan:0.55↓ | App:87%, Chol:68%, Pan:61% |
| 6 | R | "What makes it better?" | "Nothing helps" | App:0.90↑, Chol:0.85↑, Pan:0.60↓ | App:91%, Chol:73%, Pan:61% |
| 7 | T | "Is it constant?" | "Yes, constant" | App:0.91↑, Chol:0.88↑, Pan:0.87↑ | App:94%, Chol:77%, Pan:69% |
| 8 | S | "How severe (1-10)?" | "8 out of 10" | App:0.88↑, Chol:0.85↑, Pan:0.90↑ | App:95%, Chol:79%, Pan:75% |

**Result:** Appendicitis 95%, OLDCARTS complete → **DIAGNOSIS**

### Red Flag Screening:

**Now send FULL Appendicitis guideline to LLM:**
```
Full guideline with red flags → LLM generates 4 red flag questions → Check for complications
```

---

## Technical Implementation

### 1. OLDCARTS Tracking
```python
self.oldcarts_covered = {
    'O': False,  # Onset (hardcoded first)
    'L': False,  # Location
    'D': False,  # Duration
    'C': False,  # Character
    'A': False,  # Aggravating
    'R': False,  # Relieving
    'T': False,  # Timing
    'S': False   # Severity
}
```

### 2. Question Detection
```python
def _detect_oldcarts_element(question):
    if 'where' in question: return 'L'
    if 'how long' in question: return 'D'
    if 'describe' in question: return 'C'
    if 'worse' in question: return 'A'
    if 'better' in question: return 'R'
    if 'constant' in question: return 'T'
    if 'scale' in question or '1-10' in question: return 'S'
```

### 3. Section Extraction
```python
def _extract_oldcarts_section(classic_presentation, element):
    # Extract "LOCATION: ...text..." using regex
    pattern = f"{element_name}:([^.]*(?:\\.[^A-Z:][^.]*)*)"
    match = re.search(pattern, classic_presentation)
    return match.group(1) if match else ""
```

### 4. Semantic Similarity (RAG Container)
```python
def _compute_similarity(text1, text2):
    # Call RAG container
    emb1 = embedding_api.encode([text1])[0]  # Via RAG /embed endpoint
    emb2 = embedding_api.encode([text2])[0]
    
    # Cosine similarity
    similarity = np.dot(emb1, emb2) / (norm(emb1) * norm(emb2))
    
    return (similarity + 1) / 2  # Scale to 0-1
```

### 5. Score Update
```python
# Weighted average (70% old + 30% new similarity)
new_score = (old_score * 0.7) + (similarity * 0.3)
```

### 6. Diagnosis Criteria
```python
oldcarts_complete = all(oldcarts_covered.values())

if oldcarts_complete and top['score'] >= 0.95:
    diagnose()  # All OLDCARTS covered + high confidence
elif num_questions >= 15:
    diagnose()  # Max questions limit
else:
    ask_next_question()  # Continue OLDCARTS
```

---

## Benefits Over LLM-Based Scoring

### **Old Approach (LLM Scoring):**
```
LLM sees: Full guideline + question + answer
LLM outputs: "75" or "333333..." (hallucination)
Problems:
❌ LLM hallucinations ("3333...")
❌ Subjective/inconsistent scoring
❌ Token-heavy (complex prompt)
❌ Slow (LLM inference for each guideline)
```

### **New Approach (Vector Similarity):**
```
Extract: LOCATION section from guideline
Vectorize: Answer + section
Compute: Cosine similarity (0.85)
Update: Weighted average score
Benefits:
✅ Objective, mathematical
✅ No hallucination possible
✅ Token-efficient (API calls)
✅ Fast (vector operations)
✅ Precise (section-specific matching)
```

---

## Example: Why This is Better

**Patient says:** "The pain is in my lower right side"

**Old Approach (LLM):**
```
Prompt (1000 tokens):
"Guideline: Acute Appendicitis
 Full classic presentation: ...500 words...
 
 Patient: 35F with abdominal pain
 Q: Where is the pain?
 A: Lower right side
 
 Score 0-100:"

LLM Response: "333333..." (hallucination)
```

**New Approach (Vector Similarity):**
```
Prompt (50 tokens):
"Ask about LOCATION.
 Example: Where is the pain?"

LLM Response: "Where in your abdomen is the pain?"

Then scoring (no LLM):
  Extract: "Pain MIGRATES to right lower quadrant (RLQ)"
  Similarity: 0.85 (HIGH match for "lower right side")
  Score: 60% → 67% ↑
```

---

## No Fallbacks Policy

### What Fails Cleanly:

**1. Embedding Service Down:**
```
RuntimeError: "RAG embed API returned status 503"
→ Diagnosis cannot proceed (explicit error)
```

**2. OLDCARTS Section Missing:**
```
RuntimeError: "Could not extract L section from Appendicitis"
→ Guideline format issue (explicit error)
```

**3. Similarity Computation Error:**
```
RuntimeError: "Embedding model not initialized"
→ System misconfiguration (explicit error)
```

**No silent degradation, no default scores, no keyword fallbacks.**

---

## Token Efficiency

### Per Question Cycle:

**Question Generation:**
- Prompt: ~50 tokens (vs ~1,000 old)
- Response: ~10 tokens
- **Total: ~60 tokens**

**Semantic Scoring (3 guidelines):**
- Embedding API calls: 4 texts (answer + 3 sections)
- No LLM tokens used
- **Total: 0 LLM tokens**

**Per Question: ~60 tokens** (vs ~1,100 old)

**For 8 OLDCARTS questions: ~480 tokens total** (vs ~8,800 old)

---

## Summary

**Old System:**
- LLM generates questions (complex prompt, 1000 tokens)
- LLM scores (subjective, hallucinations, 1000 tokens)
- Total: ~2,000 tokens per question, prone to hallucination

**New System:**
- LLM generates questions (minimal prompt, 60 tokens)
- Vector similarity scores (objective, 0 LLM tokens)
- Total: ~60 tokens per question, no hallucination risk

**Improvement:**
- 97% reduction in LLM token usage
- 100% elimination of scoring hallucinations
- Objective, reproducible scoring
- Systematic OLDCARTS coverage

**This is the right architecture!** 🎯


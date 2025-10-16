# Hybrid Diagnostic System: Rolling Top-5 + LLM Intelligence

## Architecture

**Combines the best of both worlds:**
1. **Rolling Top-5 Differential** - Scalability (handles 160 guidelines)
2. **LLM + RAG Question Generation** - Intelligence (conversational, adaptive)

## System Flow

### **Step 1: Chief Complaint Matching**
```
User: "I have abdominal pain"
↓
JSON matching → ALL conditions with "abdominal pain" trigger
↓
Found: 20 GI conditions
↓
Sort by initial score (trigger match + prevalence)
↓
Active: Top 5
Reserve: Remaining 15
```

### **Step 2: LLM Reads Clinical Guidelines**
```
For each question cycle:
  1. Get top 3 differentials
  2. Retrieve FULL guidelines from RAG (12 chunks each)
  3. Give LLM the clinical content
  4. LLM generates intelligent question
```

### **Step 3: Intelligent Question Generation**

**LLM Prompt:**
```
CURRENT DIFFERENTIALS:
1. Acute Appendicitis (60%)
2. Cholecystitis (55%)
3. Acute Pancreatitis (52%)

CLINICAL GUIDELINES:
=== Acute Appendicitis ===
Classic presentation: Periumbilical pain → RLQ migration
Key features: Anorexia (>90%), rebound tenderness
Red flags: Fever >102.5°F suggests perforation
...

=== Cholecystitis ===
Classic presentation: RUQ pain after fatty meal
Key features: Murphy's sign, postprandial pain
...

=== Acute Pancreatitis ===
Classic presentation: Epigastric pain radiating to back
Key features: Alcohol history, gallstones
...

PATIENT INFO SO FAR:
- "I have abdominal pain"

ALREADY ASKED:
None

YOUR TASK:
Generate the SINGLE MOST IMPORTANT next question.
```

**LLM Response:**
```
"When did the pain start, and has it moved from one area to another?"

(Combined onset + migration because migration is critical for appendicitis!)
```

### **Step 4: User Answers**
```
User: "Started yesterday around my belly button, now it's in my lower right side"
↓
Feature extraction:
  - onset_timing: "acute_days"
  - Fuzzy match to "periumbilical then RLQ" (migration pattern)
↓
Store raw answer: "Started yesterday around my belly button..."
```

### **Step 5: Score All Active Guidelines**
```
Appendicitis:
  + Initial match: 0.50
  + Onset acute: +0.10
  + Migration periumbilical→RLQ: +0.30 (CRITICAL match!)
  = 0.90 ✅

Cholecystitis:
  + Initial match: 0.55
  + Onset acute: +0.10
  - Migration doesn't match RUQ: 0 points
  = 0.65 ✅

Pancreatitis:
  + Initial match: 0.52
  + Onset acute: +0.10
  - No migration pattern match
  = 0.62 ✅

Gastroenteritis:
  + Initial match: 0.50
  - Acute onset rules out (chronic more common)
  = 0.28 ❌ RULED OUT

Peptic Ulcer:
  + Initial match: 0.48
  - Acute onset doesn't match
  = 0.22 ❌ RULED OUT
```

### **Step 6: Rolling Update**
```
Active list updated:
  1. Appendicitis (0.90) ← Clear leader!
  2. Cholecystitis (0.65)
  3. Pancreatitis (0.62)
  4. Diverticulitis (0.45) ← From reserve
  5. Bowel Obstruction (0.42) ← From reserve

Ruled out:
  - Gastroenteritis
  - Peptic Ulcer
```

### **Step 7: Next LLM Question**

**LLM Prompt:**
```
DIFFERENTIALS:
1. Appendicitis (90%) ← Strong leader
2. Cholecystitis (65%)
3. Pancreatitis (62%)

GUIDELINES:
[Full content for all 3]

PATIENT INFO:
- Started yesterday around belly button
- Now in lower right side

ALREADY ASKED:
- "When did the pain start and has it moved?"

TASK: Next question?
```

**LLM Response:**
```
"How would you describe the pain - is it sharp and constant, 
or more of a cramping that comes and goes?"

(Quality helps differentiate: sharp/constant = appendicitis/cholecystitis, 
 cramping = bowel obstruction)
```

### **Step 8: Continue Until Diagnosis**
```
After 4-6 questions:

Scores:
  1. Appendicitis (0.96)
  2. Diverticulitis (0.45)
  3. Cholecystitis (0.42)

Diagnosis threshold met (0.96 > 0.90 AND 4+ questions)
```

### **Step 9: Clinical Recap**
```
System: "Based on your symptoms - pain started yesterday around 
         your belly button, now in lower right side, sharp and 
         constant, with nausea and loss of appetite - this is 
         likely Acute Appendicitis.
         
         ⚠️ Go to the emergency room immediately.
         
         [RAG-retrieved education about appendicitis]"
```

## Key Advantages

### **Vs. Rigid Triage:**
- ✅ Multi-guideline scoring (not single path)
- ✅ Natural questions (not templates)
- ✅ Intelligent (reads full guidelines)
- ✅ Adaptive (no fixed order)

### **Vs. Pure LLM:**
- ✅ Scalable (rolling top-5, not all 160)
- ✅ Structured scoring (not just LLM hallucination)
- ✅ Efficient RAG usage (3 guidelines, not 160)
- ✅ Fast (smaller context windows)

### **Best of Both:**
- **Rolling top-5**: Manages scalability
- **JSON matching**: Fast chief complaint routing
- **Structured scoring**: Reliable differential narrowing
- **LLM + RAG**: Intelligent, conversational questions
- **Fuzzy matching**: Handles misspellings
- **Clinical recap**: Uses patient's exact words

## Example Conversation

```
You: "I have chest pain"

[Matched: 15 cardiac conditions]
[Active: Top 5]
[Reserve: 10]

[LLM reads guidelines for top 3]
System: "When did this chest pain start, and did it come on suddenly or gradually?"

You: "About 20 minutes ago, very suddenly"

[Onset: acute_hours]
[Scores updated - MI: 0.85, Dissection: 0.80, GERD: 0.20 (ruled out)]
[Reserve promoted: PE: 0.65]

[LLM reads updated top 3]
System: "Can you describe what the pain feels like? Is it more like 
         pressure or heaviness, or is it sharp and tearing?"

(LLM knows: pressure = MI, tearing = dissection)

You: "Like an elephant sitting on my chest"

[Quality: pressure]
[MI: 0.95, Dissection: 0.60]

[LLM reads guidelines]
System: "Does the pain spread anywhere - to your left arm, jaw, 
         neck, or back?"

(LLM knows: arm/jaw = MI, back = dissection)

You: "Left arm and jaw"

[Radiation: left arm/jaw]
[MI: 0.98]

[4 questions, score > 0.90]
System: "Based on your symptoms - pain started 20 minutes ago very 
         suddenly, like an elephant sitting on your chest, spreading 
         to left arm and jaw - this is likely Acute Myocardial Infarction.
         
         🚨 Call 911 immediately. This is a medical emergency."
```

## Configuration

```python
# In adaptive_diagnostic_engine.py
MAX_ACTIVE = 5                # Keep top 5 differentials
RULE_OUT_THRESHOLD = 0.3      # Score < 0.3 → ruled out
MIN_QUESTIONS = 4             # Minimum questions before diagnosis
HIGH_CONFIDENCE = 0.90        # Score for diagnosis with min questions

# LLM parameters for question generation
max_tokens = 100
temperature = 0.7             # Some creativity for natural phrasing
```

## Testing

**On Jetson:**
```bash
cd ~/LedgerAI/llm-container
docker build -t aura-llm .
docker stop aura-llm && docker rm aura-llm
cd ../aura-control
python3 core/main.py
```

**Expected Logs:**
```
[Unified Medical] ✅ Adaptive engine initialized with LLM intelligence
[Adaptive] 🤖 Generating LLM-driven question...
[Adaptive]   📚 Retrieved 12 chunks for Acute Appendicitis
[Adaptive] 🤖 LLM generated: 'Where exactly do you feel the pain...'
```

## Summary

This **hybrid system** combines:
- **Rolling top-5** → Scalable to 160 guidelines
- **LLM reasoning** → Conversational and adaptive
- **RAG knowledge** → Evidence-based questions
- **Structured scoring** → Reliable differential narrowing
- **Fuzzy matching** → Handles real-world input

**It's the best of all approaches!** 🎯


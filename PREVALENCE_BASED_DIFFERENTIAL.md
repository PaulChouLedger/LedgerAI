# Prevalence-Based Rolling Differential System

## Overview

The Adaptive Diagnostic Engine now uses **evidence-based prevalence data** to intelligently prioritize conditions throughout the diagnostic process. This mimics how experienced clinicians think: **"Common things are common"**.

---

## How It Works

### 1. **Initial Matching & Sorting**

When a patient reports "I have abdominal pain":

```
Step 1: Match all relevant guidelines (trigger words match)
Step 2: Sort by URGENCY, then by PREVALENCE within urgency tier
  
  Urgency Priority:
  - Emergent > Urgent > Routine
  
  Prevalence Priority (within same urgency):
  - Common (60% initial score) > Uncommon (50%) > Rare (40%)
```

**Example Output:**
```
SORTED BY URGENCY + PREVALENCE:
1. Acute Appendicitis (common, urgent, 60%)
2. Acute Cholecystitis (common, urgent, 60%)
3. Acute Pancreatitis (common, urgent, 60%)
4. Acute Diverticulitis (uncommon, urgent, 50%)
5. Bowel Obstruction (uncommon, urgent, 50%)
6. Perforated Viscus (rare, emergent, 40%)    ← Emergent BUT rare, so scores lower
7. Mesenteric Ischemia (rare, emergent, 40%)
8. Ectopic Pregnancy (rare, emergent, 40%)
```

---

### 2. **Active Differentials (Top 3)**

The **top 3** conditions become the active differential:
- These are presented to the LLM for question generation
- LLM reads their full `classic_presentation` texts
- LLM generates the most discriminating question

**Example:**
```
ACTIVE DIFFERENTIALS:
1. Acute Appendicitis (common, 60%) ⚠️
2. Acute Cholecystitis (common, 60%) ⚠️
3. Acute Pancreatitis (common, 60%) ⚠️
```

---

### 3. **Reserve Pool (Rest)**

All other matched conditions go to the **reserve pool**, **sorted by prevalence**:
- Common conditions appear first
- Rare conditions appear last
- These are candidates for promotion if active conditions are ruled out

**Example:**
```
RESERVE POOL (5 conditions, prioritized by prevalence):
1. Acute Diverticulitis (uncommon, urgent, 50%)
2. Bowel Obstruction (uncommon, urgent, 50%)
3. Biliary Colic (common, routine, 60%)      ← Common but routine urgency
4. Perforated Viscus (rare, emergent, 40%)
5. Mesenteric Ischemia (rare, emergent, 40%)
```

---

### 4. **Question → Scoring → Rolling Replacement**

After each question:

```
Step 1: LLM scores each active guideline (0-100%)
  - "No" to key features → Low score
  - "Yes" to classic presentation → High score

Step 2: Rule out <30%
  - Any guideline scoring <30% is ruled out
  - Moved to "ruled_out" list

Step 3: Promote from reserve (PREVALENCE-PRIORITIZED)
  - Reserve pool is RE-SORTED by prevalence (common first)
  - Highest-prevalence condition promoted to active
  - This ensures COMMON conditions are always considered before RARE
```

**Example Flow:**

```
After Question 1: "Where is the pain?"
Answer: "Upper right abdomen"

SCORING:
  Appendicitis: 60% → 25% (pain wrong location) ❌ RULED OUT
  Cholecystitis: 60% → 85% (RUQ matches!) ↑
  Pancreatitis: 60% → 70% (could be epigastric) ↑

ROLLING REPLACEMENT:
  ❌ RULING OUT: Appendicitis (25% < 30%)
  
  Reserve pool re-sorted by prevalence:
    1. Biliary Colic (common, 60%) ← HIGHEST PREVALENCE
    2. Diverticulitis (uncommon, 50%)
    3. Perforated Viscus (rare, 40%)
  
  🔼 PROMOTING: Biliary Colic (common, score: 60%) from reserve

UPDATED RANKINGS:
  1. Cholecystitis: 85% ⚠️
  2. Pancreatitis: 70% ⚠️
  3. Biliary Colic: 60% 📋
```

---

## Clinical Rationale

### Why This Matters

**Traditional approach (no prevalence):**
- All conditions weighted equally
- Rare conditions (ectopic, mesenteric ischemia) compete equally with common ones
- May waste time ruling out zebras before considering horses

**Prevalence-based approach:**
- **Common conditions evaluated first** (appendicitis, cholecystitis, UTI)
- **Rare conditions only considered after common ones ruled out**
- Efficient: Most patients get diagnosed quickly
- Safe: Rare emergent conditions still captured via urgency priority

---

## Evidence-Based Prevalence Classification

Based on **PMC5075866** (5,340 ED cases) and **PMC4535107**:

### **COMMON** (>3% prevalence, initial score 0.60)
- Acute Appendicitis (10-23%)
- Acute Cholecystitis (7-10%)
- Biliary Colic (7-10%)
- Kidney Stone (3-16%)
- UTI/Pyelonephritis (5-12%)
- Acute Pancreatitis (3-11%)
- Acute Gastroenteritis (5-10%)
- Peptic Ulcer Disease (~4%)

### **UNCOMMON** (1-3% prevalence, initial score 0.50)
- Acute Diverticulitis (2-7%)
- Small Bowel Obstruction (0.7-2.3%)
- Acute Gastritis (2-4%)
- GERD (2-3%)
- Ruptured Ovarian Cyst (2-4%)
- IBD Flare (1-2%)
- Severe Constipation (1-3%)
- IBS (<1% acute)

### **RARE** (<1% prevalence, initial score 0.40)
- Perforated Viscus (<1%)
- Acute Mesenteric Ischemia (<0.5%)
- Ovarian Torsion (<1%)
- Ectopic Pregnancy (<1%)
- Acute Hepatitis (<1%)

---

## Example: Full Case Flow

**Chief Complaint:** "I have abdominal pain"

**Initial Match:** 20 guidelines matched

**Sorted & Split:**
```
ACTIVE (Top 3 by urgency + prevalence):
1. Appendicitis (common, urgent, 60%)
2. Cholecystitis (common, urgent, 60%)
3. Pancreatitis (common, urgent, 60%)

RESERVE (17 conditions, sorted by prevalence):
1. Gastroenteritis (common, routine, 60%)
2. UTI (common, urgent, 60%)
3. Kidney Stone (common, urgent, 60%)
4. Diverticulitis (uncommon, urgent, 50%)
5. Biliary Colic (common, routine, 60%)
...
17. Mesenteric Ischemia (rare, emergent, 40%)
```

**Question 1:** "How old are you?" → Age: 35

**Question 2:** "Are you male or female?" → Sex: Female

**Question 3:** "When did the pain start?"
- Answer: "Yesterday"
- Scores stay similar (all acute conditions)

**Question 4:** "Where in your abdomen is the pain?"
- Answer: "Right upper area"
- Cholecystitis: 60% → 85% (RUQ!)
- Appendicitis: 60% → 25% (wrong location) → RULED OUT
- Pancreatitis: 60% → 65% (could be epigastric)
- **Promoted:** UTI (common, 60%) - next highest prevalence

**Question 5:** "Does the pain get worse after eating fatty foods?"
- Answer: "Yes"
- Cholecystitis: 85% → 95% (classic trigger!)
- Pancreatitis: 65% → 40% (not typical) → might be ruled out next
- UTI: 60% → 30% (not related) → RULED OUT
- **Promoted:** Biliary Colic (common, 60%)

**Question 6:** "Do you have fever?"
- Answer: "Yes, 101°F"
- Cholecystitis: 95% → 98% (fever confirms!)
- Biliary Colic: 60% → 25% (fever unusual) → RULED OUT
- Pancreatitis: 40% → 35% (could have fever but low overall)
- **Promoted:** Gastroenteritis (common, 60%)

**Question 7:** "Do you have nausea or vomiting?"
- Answer: "Yes, nausea"
- Cholecystitis: 98% → 99% (very common!)
- Pancreatitis: 35% → 28% → RULED OUT
- **DIAGNOSIS:** Acute Cholecystitis (99% confidence after 7 questions)

---

## Key Benefits

1. ✅ **Clinically Realistic**: Mirrors how doctors think
2. ✅ **Efficient**: Most patients diagnosed quickly (common conditions considered first)
3. ✅ **Safe**: Emergent conditions still prioritized via urgency tier
4. ✅ **Evidence-Based**: Prevalence from published medical literature
5. ✅ **Adaptive**: Rare conditions still captured if common ones ruled out
6. ✅ **Transparent**: Clear logging shows prevalence-based decisions

---

## Future Enhancements

1. **Age-Stratified Prevalence**: Adjust scores based on patient age
   - Diverticulitis: uncommon <50, common >65
   - Appendicitis: common young adults, uncommon >50

2. **Gender-Stratified Prevalence**: Adjust for biological sex
   - Cholecystitis: higher in females
   - Ectopic/Ovarian: only in reproductive-age females

3. **Risk Factor Adjustment**: Boost scores based on history
   - Pancreatitis: +20% if alcohol use or gallstones
   - Diverticulitis: +20% if >60 years old

4. **Bayesian Update**: Continuously refine prevalence as evidence emerges

---

**Last Updated:** October 2025  
**Based On:** PMC5075866, PMC4535107, UpToDate, NEJM


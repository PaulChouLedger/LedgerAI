# Red Flag Integration - Full Guideline Context for LLM

## Overview

The LLM now receives the **entire guideline** (not just `classic_presentation`) for each active condition. This includes red flags, urgency, prevalence, and full clinical context.

---

## What Changed

### Before: Classic Presentation Only
```
Guideline 1: Acute Appendicitis (Current Score: 60%, Urgency: urgent)
Classic Presentation: Acute appendicitis typically presents with periumbilical pain...
```

### After: Full Guideline with Red Flags
```
Guideline 1: Acute Appendicitis
  Current Score: 60%
  Urgency: urgent
  Prevalence: common
  
  Classic Presentation:
  Acute appendicitis typically presents with periumbilical pain that MIGRATES to the 
  right lower quadrant (RLQ) over 12-24 hours...
  
  RED FLAGS (immediately escalate if present):
  - Sudden severe pain that then improves briefly - may indicate PERFORATION
  - Severe pain with abdominal rigidity (board-like abdomen) - perforation with peritonitis, call 911
  - High fever >103°F with severe pain - possible perforation
  - Hypotension, tachycardia, altered mental status - septic shock, call 911
```

---

## Benefits

### 1. ✅ **Red Flag Detection**
The LLM can now:
- Screen for life-threatening symptoms
- Ask targeted questions about red flags
- Escalate scores (80-95%) when red flags are present
- Prioritize dangerous conditions over benign ones

**Example:**
```
Patient: "The pain was terrible, then suddenly got better about an hour ago."

LLM recognizes: "sudden improvement after severe pain" = RED FLAG for perforation
LLM scores Appendicitis → 90% (was 60%)
LLM asks: "Are you having any fever or feeling dizzy?" (screening for sepsis)
```

### 2. ✅ **Urgency-Aware Questioning**
The LLM knows which conditions are emergent vs urgent vs routine and can:
- Prioritize screening for emergent conditions
- Ask more direct/rapid questions for emergent diagnoses
- Take a more thorough approach for routine conditions

### 3. ✅ **Better Scoring**
With full context, the LLM can:
- Give higher scores when red flags are present
- Lower scores when key features are absent
- Make more nuanced assessments

**Scoring Prompt Example:**
```
If answer is "no" to a key feature, give LOW score.
If answer matches classic presentation, give HIGH score.
If RED FLAG present, give VERY HIGH score (80-95).
```

### 4. ✅ **Final Diagnosis with Red Flags**
The diagnosis message now includes red flags for urgent/emergent conditions:
```
Based on your symptoms, this is most likely Acute Appendicitis (confidence: 92%).

⚠️ This requires prompt medical attention. Go to urgent care or ER today.

⚠️ Watch for these warning signs:
• Sudden severe pain that then improves briefly - may indicate PERFORATION
• Severe pain with abdominal rigidity (board-like abdomen) - call 911
• High fever >103°F with severe pain - possible perforation
```

---

## Token Usage

### Per Guideline Estimate:
- Classic presentation: ~250 tokens
- Red flags (4 items): ~75 tokens
- Metadata (name, urgency, prevalence): ~20 tokens
- **Total per guideline: ~345 tokens**

### For 3 Active Guidelines:
- **3 × 345 = ~1,035 tokens**

### Total Context for Question Generation:
- 3 guidelines: ~1,035 tokens
- Patient info + history: ~200 tokens
- System prompt + instructions: ~150 tokens
- **Total: ~1,385 tokens**

✅ **Well within limits** even for small models (1B-3B with 4096 context)

---

## Updated Sections

### 1. Question Generation (Lines 500-585)
- Now includes full guideline with red flags
- Instructs LLM to screen for red flags first
- Examples include red flag questions

### 2. Scoring (Lines 710-737)
- Full guideline + red flags sent to LLM
- Scoring prompt explicitly mentions red flags
- Higher scores (80-95%) for red flag presence

### 3. Final Diagnosis (Lines 842-888)
- Red flags included in diagnosis message
- Top 3 red flags shown for urgent/emergent conditions
- Red flag count logged

---

## Example Clinical Flow

**Chief Complaint:** "I have abdominal pain"

**Active Differentials:**
1. Acute Appendicitis (common, urgent, 60%)
2. Acute Cholecystitis (common, urgent, 60%)
3. Acute Pancreatitis (common, urgent, 60%)

**Question 1:** "When did the pain start?"
- Answer: "Yesterday"
- All 3 conditions still plausible (acute onset)

**Question 2:** "Did the pain move from one location to another?"
- Answer: "Yes, started around my belly button, now it's in my lower right side"
- **LLM recognizes:** "MIGRATES to RLQ" = KEY FEATURE of appendicitis
- Appendicitis: 60% → 85%
- Cholecystitis: 60% → 30% (wrong location)
- Pancreatitis: 60% → 35% (wrong location)

**Question 3:** "Have you had any fever?"
- Answer: "Yes, about 101 degrees"
- Appendicitis: 85% → 90% (low-grade fever expected)

**Question 4:** "Are you hungry or have you lost your appetite?"
- Answer: "I can't even think about food"
- **LLM recognizes:** "ANOREXIA" = KEY FEATURE in >90% of appendicitis
- Appendicitis: 90% → 95%

**Question 5:** "Did the pain suddenly get much better at any point?"
- Answer: "No, it's been constant and getting worse"
- **LLM screens:** No perforation red flag
- Appendicitis: 95% (confirmed)

**DIAGNOSIS:** Acute Appendicitis (95% confidence)
```
⚠️ This requires prompt medical attention. Go to urgent care or ER today.

⚠️ Watch for these warning signs:
• Sudden severe pain that then improves briefly - may indicate PERFORATION
• Severe pain with abdominal rigidity (board-like abdomen) - call 911
• High fever >103°F with severe pain - possible perforation
```

---

## Safety Features

### 1. **Red Flag Screening**
Questions explicitly ask about dangerous symptoms:
- "Have you noticed any blood in your stool?"
- "Are you having severe pain that won't go away?"
- "Have you felt dizzy or fainted?"

### 2. **Automatic Escalation**
If red flags detected:
- Score boosted to 80-95%
- Diagnosis message includes specific red flags
- Urgency reinforced in message

### 3. **Patient Education**
Final message includes:
- Top 3 red flags to watch for
- Specific action if red flag occurs (call 911, go to ER)
- Clear urgency level

---

## Future Enhancements

1. **Dynamic Red Flag Prioritization**
   - If emergent condition in active list, prioritize red flag screening
   - Ask red flag questions earlier in assessment

2. **Red Flag-Triggered Diagnosis**
   - If patient reports severe red flag (e.g., "board-like abdomen"), immediately diagnose without 7 questions
   - Fast-track to diagnosis + 911 recommendation

3. **Red Flag History Tracking**
   - Track which red flags were asked about
   - Don't repeat red flag questions

4. **Condition-Specific Red Flag Questions**
   - Appendicitis: "Did the pain suddenly improve?" (perforation)
   - Ectopic: "Have you had any dizziness or fainting?" (rupture)
   - Mesenteric Ischemia: "Is the pain much worse than your exam?" (pain out of proportion)

---

**Last Updated:** October 2025  
**Token Budget:** ~1,400 tokens per question (well within 4K context)  
**Safety Impact:** ✅ High - Red flags now actively screened and communicated


# Clinical Interview Flow - Exact Specification

## Complete 21-Guideline GI/GYN Library

✅ **All guidelines created and ready:**

### Gastrointestinal (16):
1. Acute Appendicitis
2. Acute Cholecystitis
3. Acute Pancreatitis
4. Acute Gastroenteritis
5. Acute Diverticulitis
6. Biliary Colic
7. Bowel Obstruction
8. Peptic Ulcer Disease
9. GERD
10. Gastritis
11. Constipation
12. IBD Flare
13. IBS
14. Mesenteric Ischemia
15. Perforated Viscus
16. Hepatitis

### Genitourinary (2):
17. Kidney Stone
18. UTI/Pyelonephritis

### Gynecologic (3):
19. Ectopic Pregnancy
20. Ovarian Torsion
21. Ruptured Ovarian Cyst

## Exact Interview Flow

### **Phase 1: Chief Complaint**
```
User: "I have abdominal pain"

[System matches 20 guidelines]
[Active top 5, Reserve 15]
```

### **Phase 2: Empathy + Demographics**
```
System: "I'm sorry you're experiencing abdominal pain. Let me ask 
         you some questions to help determine what's causing it. 
         How old are you?"

User: "35"
[Age: 35 stored]

System: "Are you male or female?"

User: "female"
[Sex: female stored]
[Scores updated - female conditions get bonus]
```

### **Phase 3: Location (PRIORITY for GI)**
```
System: "Where in your abdomen is the pain located?"

User: "right side"
[Ambiguous! Need clarification]

System: "Is it in the upper right (below your ribs) or 
         lower right side of your abdomen?"

User: "lower right"
[RLQ identified!]
[Scores updated:]
  Appendicitis: 0.50 → 0.85 (RLQ match!)
  Cholecystitis: 0.50 → 0.30 (RUQ expected, not RLQ)
  Diverticulitis: 0.50 → 0.35 (LLQ expected)
[Rolling update: Cholecystitis ruled out, new condition promoted]
```

### **Phase 4: High-Yield Questions from Classical Presentation**
```
[Top 3 now: Appendicitis (0.85), Ovarian Torsion (0.60), Kidney Stone (0.55)]

System: "Have you had any fever?"

User: "yes, I feel warm"
[Fever: positive]
[Scores:]
  Appendicitis: 0.85 → 0.95 (fever expected!)
  Ovarian Torsion: 0.60 → 0.65
  Kidney Stone: 0.55 → 0.50

[Threshold met: Appendicitis > 0.90!]
[Load FULL Appendicitis guideline from RAG]
```

### **Phase 5: Detailed Questions to Fill Gaps**
```
[System checks: Location ✓, Fever ✓, Missing: migration, quality]

System: "Did the pain start in one place and move to another?"
(NOT "Where is the pain?" - already asked!)

User: "yes, from center to right lower"
[Migration: periumbilical→RLQ!]
[Appendicitis: 0.95 → 0.98]

System: "Can you describe the pain - is it sharp, dull, or cramping?"

User: "sharp"
[Quality: sharp]
[Appendicitis: 0.98 → 0.99]

[4 questions answered, confidence > 0.95]
```

### **Phase 6: Diagnosis + Recap**
```
System: "Based on your symptoms - pain started yesterday, located 
         lower right side of your abdomen, yes from center to right 
         lower, sharp, with fever - this is likely Acute Appendicitis.
         
         ⚠️ This requires prompt medical attention. Go to the 
         emergency room immediately.
         
         Red flags to watch for:
         - Severe worsening pain
         - Unable to walk
         - Vomiting
         - Fever >101°F
         
         [More education from RAG...]"
```

## Key Features Implemented

### ✅ **Empathy First**
- "I'm sorry you're experiencing..."
- Sets compassionate tone

### ✅ **Demographics Early**  
- Age and sex critical for differential (e.g., ectopic pregnancy)
- Modifies scores appropriately

### ✅ **Location PRIORITY for GI**
- First clinical question always location
- Most discriminating for abdominal pain

### ✅ **Clarifies Ambiguity**
- "right side" → "upper right or lower right?"
- "left side" → "upper left or lower left?"
- Ensures precise localization

### ✅ **Never Combines Questions**
- LLM prompt explicitly forbids combining
- One question at a time for clarity

### ✅ **Doesn't Re-ask**
- Tracks all questions asked
- Asks complementary questions to fill gaps

### ✅ **Not a New Complaint**
- "I have pain on my right side" = ANSWER (more detail)
- NOT new complaint (same symptom type)

### ✅ **Rolling Top-5**
- Handles 20 guidelines efficiently
- Rules out, promotes from reserve
- Scalable to 160

## Testing on Jetson

**The system will auto-convert guidelines on startup!**

```bash
cd ~/LedgerAI/aura-control
python3 core/main.py
```

**Expected startup logs:**
```
[Aura] 🔄 Found 21 new medical guidelines - auto-converting...
[Aura] ✅ Guidelines converted to RAG format
[Aura] 🔄 Rebuilding embeddings with new guidelines...
[Aura] ✅ Embeddings rebuilt - new guidelines ready for RAG!
[Adaptive] ✅ Loaded 21 guidelines
```

**Then test:**
```
You: "I have abdominal pain"

System: "I'm sorry you're experiencing abdominal pain. Let me ask 
         you some questions to help determine what's causing it. 
         How old are you?"

You: "30"

System: "Are you male or female?"

You: "male"

System: "Where in your abdomen is the pain located?"

You: "right side"

System: "Is it in the upper right (below your ribs) or lower 
         right side of your abdomen?"

You: "lower right"

[Appendicitis score jumps to 0.85]

System: "Have you had any fever?"

...continues with intelligent questions...
```

**This is exactly the flow you specified!** 🎯


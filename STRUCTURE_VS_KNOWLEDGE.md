# Structure vs Knowledge: What We're Actually Training

## Key Insight

**The base LLM has enough medical knowledge to diagnose, but it lacks the STRUCTURE to:**
1. Ask questions systematically (OLD CARTS framework)
2. Reason methodically through differential diagnosis
3. Build diagnosis step-by-step using collected information

## What the Model Already Knows ✅

The base model (Qwen 2.5-1.5B-Instruct) already understands:
- **Medical facts**: Anatomy, physiology, pathology
- **Clinical patterns**: "RUQ pain + fatty meal" → gallbladder
- **Terminology**: "Pleuritic" → pulmonary/pleural
- **Anatomical relationships**: "Epigastric" → stomach/pancreas

**Example**: If you ask the base model "What causes right upper quadrant pain?", it knows:
- Liver disease
- Gallbladder issues
- Biliary system problems
- Hepatitis
- etc.

## What the Model Lacks ❌

The base model doesn't know how to:
- **Structure a medical interview**: What questions to ask, in what order
- **Reason systematically**: How to build a differential step-by-step
- **Apply knowledge methodically**: How to use medical facts in a structured way
- **Progressive narrowing**: How each answer should refine the differential

**Example**: The model knows RUQ pain = gallbladder, but doesn't know:
- How to ask about RUQ pain systematically
- How to reason through: "If onset is sudden + location is RUQ + character is colicky → gallbladder"
- How to build this diagnosis step-by-step

## What We're Training

### 1. Systematic Question Structure ✅

**Teaching**: How to ask questions in a structured way
- Empathy → Chronicity → Demographics → OLD CARTS
- One OLD CARTS element at a time
- Follow the framework systematically

**Not Teaching**: Medical facts (already knows)

### 2. Methodical Reasoning Process ✅

**Teaching**: How to reason after each answer
- Step 1: Collect Onset → Reason: How does this affect differential?
- Step 2: Collect Location → Reason: How does this narrow differential?
- Step 3: Collect Character → Reason: How does this refine diagnosis?
- Continue systematically

**Not Teaching**: What conditions exist (already knows)

### 3. Progressive Differential Building ✅

**Teaching**: How to build diagnosis step-by-step
- After each answer: Rule IN conditions that match
- After each answer: Rule OUT conditions that don't match
- Progressively narrow: Each element refines the differential
- Update rankings systematically

**Not Teaching**: Medical knowledge (already has)

### 4. Structured Application of Knowledge ✅

**Teaching**: How to apply medical knowledge in a structured way
- Use anatomical knowledge → but apply it systematically
- Use clinical patterns → but reason through them methodically
- Use terminology → but structure the conversation properly

**Not Teaching**: The knowledge itself (already has)

## Training Examples

### Example 1: Systematic Reasoning Structure

**What We Teach:**
```
Step 1: Collect Onset (O) = "started suddenly"
→ Reason: Sudden onset rules IN acute conditions, rules OUT chronic
→ Update differential: Acute MI ↑, Stable Angina ↓

Step 2: Collect Location (L) = "center chest"
→ Reason: Center chest + sudden onset = cardiac pattern
→ Update differential: Acute MI ↑↑, Pulmonary Embolism ↓

Step 3: Collect Character (C) = "pressure"
→ Reason: Center + sudden + pressure = CLASSIC MI pattern
→ Update differential: Acute MI ↑↑↑, Aortic Dissection ↓
```

**What We Don't Teach:**
- That MI causes chest pain (already knows)
- That center chest = cardiac (already knows)
- That pressure = cardiac (already knows)

### Example 2: Progressive Narrowing

**What We Teach:**
```
Initial: Broad differential (many possibilities)
After Onset: Narrow to acute conditions
After Location: Narrow to cardiac/pulmonary
After Character: Narrow to cardiac
After Duration: Narrow to MI vs Angina
Final: Most likely diagnosis
```

**What We Don't Teach:**
- What conditions exist (already knows)
- What symptoms mean (already knows)

## Dataset Structure

### Systematic Reasoning Examples (2 new examples)

Added examples that explicitly show:
1. **Step-by-step reasoning process**
   - "SYSTEMATIC REASONING - Step 1 (Onset)"
   - "SYSTEMATIC REASONING - Step 2 (Location + Character)"
   - "SYSTEMATIC REASONING - Step 3 (Duration + Timing)"

2. **Methodical differential building**
   - "REASONING PROCESS: 1. This rules IN... 2. This rules OUT..."
   - "UPDATED DIFFERENTIAL: Based on collected elements..."
   - "FINAL DIFFERENTIAL: Based on systematic collection..."

3. **Progressive narrowing**
   - Each step shows how differential narrows
   - Each step shows updated probabilities
   - Each step shows reasoning for changes

## System Prompt Updates

Updated to emphasize:
- "You have medical knowledge. Your job is to STRUCTURE the conversation"
- "Ask questions SYSTEMATICALLY"
- "Reason METHODICALLY"
- "Build diagnosis STEP-BY-STEP"
- "Apply knowledge in a STRUCTURED way"

## Expected Behavior

### Before Training:
- Model knows: RUQ pain = gallbladder ✅
- Model doesn't know: How to ask about RUQ pain systematically ❌
- Model doesn't know: How to reason through RUQ pain step-by-step ❌

### After Training:
- Model knows: RUQ pain = gallbladder ✅ (preserved)
- Model knows: How to ask about RUQ pain systematically ✅ (learned)
- Model knows: How to reason through RUQ pain step-by-step ✅ (learned)

## Summary

✅ **Training Focus**: STRUCTURE and SYSTEMATIC REASONING
- How to ask questions (framework)
- How to reason methodically (process)
- How to build diagnosis step-by-step (systematic approach)

❌ **Not Training**: Medical knowledge
- Medical facts (already in base model)
- Clinical patterns (already in base model)
- Terminology (already in base model)

**Key Insight**: The model has the knowledge, we're teaching it how to USE that knowledge systematically.


# Red Flag Screening System

## Overview

After reaching a diagnosis, the system **systematically screens for all red flags** to ensure no critical warning signs are missed. This is a crucial safety feature that happens automatically.

---

## How It Works

### Phase 1: Diagnostic Questioning
- System asks 7-12 clinical questions
- Reaches diagnosis when confidence ≥90% or 12 questions asked

### Phase 2: Red Flag Screening (NEW)
- **Automatically triggered** after diagnosis
- Asks simple yes/no questions for **each red flag**
- Tracks which red flags are present
- Escalates urgency if red flags detected

### Phase 3: Final Disposition
- Comprehensive message with diagnosis + detected red flags
- Clear urgency level (potentially escalated)
- Specific action items

---

## Example Flow

**Diagnosis Reached:**
```
[Engine] ✅ DIAGNOSIS REACHED: Acute Appendicitis (92% confidence)
[Engine] 🚩 Starting RED FLAG screening...
[Engine] 🚩 Screening 4 red flags for Acute Appendicitis
```

**Red Flag Questions:**
```
[Engine] 🚩 Red flag 1/4: Sudden severe pain that then improves briefly

Q: "Did the pain suddenly get much better after being severe?"
A: "No"
[Engine] ✓ Red flag not present

---

[Engine] 🚩 Red flag 2/4: Severe pain with abdominal rigidity (board-like abdomen)

Q: "Is your abdomen very hard or rigid when you press on it?"
A: "No"
[Engine] ✓ Red flag not present

---

[Engine] 🚩 Red flag 3/4: High fever >103°F with severe pain

Q: "Have you had a fever higher than 103 degrees?"
A: "Yes"
[Engine] ⚠️ RED FLAG PRESENT: High fever >103°F with severe pain - possible perforation

---

[Engine] 🚩 Red flag 4/4: Hypotension, tachycardia, altered mental status

Q: "Have you felt dizzy, lightheaded, or like you might faint?"
A: "No"
[Engine] ✓ Red flag not present

---

[Engine] ✅ Red flag screening complete (1 flags present)
```

**Final Diagnosis:**
```
================================================================================
[Engine] 🎯 FINAL DIAGNOSIS
================================================================================
[Engine] Condition: Acute Appendicitis
[Engine] Confidence: 92%
[Engine] Urgency: emergent (escalated from urgent)
[Engine] 🚨 Red Flags Detected: 1
[Engine]   - High fever >103°F with severe pain - possible perforation
================================================================================

Based on your symptoms, this is most likely Acute Appendicitis (confidence: 92%).

🚨 This is a medical emergency. Call 911 or go to the ER immediately.

🚨 WARNING SIGNS DETECTED:
• High fever >103°F with severe pain - possible perforation

Seek immediate medical attention.
```

---

## Red Flag Question Conversion

The system automatically converts red flag statements into simple yes/no questions:

| Red Flag Statement | Generated Question |
|-------------------|-------------------|
| "High fever >103°F with severe pain" | "Have you had a fever higher than 103 degrees?" |
| "Sudden severe pain that then improves briefly" | "Did the pain suddenly get much better after being severe?" |
| "Severe pain with abdominal rigidity (board-like abdomen)" | "Is your abdomen very hard or rigid when you press on it?" |
| "Hypotension, tachycardia, altered mental status" | "Have you felt dizzy, lightheaded, or like you might faint?" |
| "Bloody diarrhea with high fever" | "Have you seen any blood in your stool or diarrhea?" |
| "Jaundice (yellow eyes/skin)" | "Have your eyes or skin turned yellow?" |

---

## Urgency Escalation

If **any red flag is detected**, urgency is automatically escalated:

| Original Urgency | Escalated To | Reason |
|-----------------|-------------|--------|
| Routine | Urgent | Red flag detected |
| Urgent | **Emergent** | Red flag detected → Call 911 |
| Emergent | Emergent | Already maximum urgency |

---

## Safety Benefits

### 1. ✅ **Comprehensive Screening**
- **Every single red flag** is asked about
- No dangerous symptoms missed
- Systematic approach (not LLM-dependent)

### 2. ✅ **Automatic Escalation**
- Urgency escalated if red flags present
- Message changes to "Call 911" if emergent
- Patient receives clear action items

### 3. ✅ **Simple Yes/No Questions**
- No LLM hallucination risk (simple template-based questions)
- Easy for patient to answer
- Fast (no complex reasoning needed)

### 4. ✅ **Explicit Warnings**
- Detected red flags shown in final message
- Specific symptom highlighted (not generic)
- "Seek immediate medical attention" directive

---

## Implementation Details

### New State: `red_flag_screening`
```python
self.status = "red_flag_screening"
self.red_flags_present = []  # Track which are present
self.red_flag_index = 0      # Track which one we're asking about
```

### Red Flag Screening Function
```python
def _screen_red_flags(self, diagnosis_obj):
    # Get all red flags for diagnosed condition
    red_flags = diagnosis_obj['data'].get('red_flags', [])
    
    # Ask about each one sequentially
    if self.red_flag_index < len(red_flags):
        question = self._red_flag_to_question(red_flags[self.red_flag_index])
        return {'question': question}
    
    # When done, finalize diagnosis
    return self._finalize_diagnosis(diagnosis_obj)
```

### Answer Processing
```python
if self.status == 'red_flag_screening':
    is_yes = any(word in answer_lower for word in ['yes', 'yeah', 'yep', 'sure'])
    
    if is_yes:
        self.red_flags_present.append(red_flag_text)
        print(f"⚠️ RED FLAG PRESENT")
    
    self.red_flag_index += 1
    return self._screen_red_flags(diagnosis_obj)
```

---

## Example: Appendicitis with Perforation

**Scenario:** Patient has appendicitis with high fever (possible perforation)

**Diagnosis Phase:**
- 7 questions asked
- Diagnosis: Acute Appendicitis (92% confidence)
- Initial urgency: **Urgent** (needs ER today)

**Red Flag Screening:**
```
Q1: "Did the pain suddenly get much better?" → No
Q2: "Is your abdomen very hard or rigid?" → No
Q3: "Have you had a fever higher than 103?" → YES ✓
Q4: "Have you felt dizzy or faint?" → No
```

**Result:**
- 1 red flag detected: High fever >103°F
- Urgency escalated: Urgent → **Emergent**
- Message: "🚨 Call 911 or go to the ER immediately"
- Specific warning: "High fever >103°F - possible perforation"

**Impact:**
- Patient receives appropriate urgency (911, not just "go to ER today")
- Specific symptom highlighted (perforation risk)
- Clear action: Seek immediate emergency care

---

## Future Enhancements

1. **Adaptive Red Flag Ordering**
   - Ask most critical red flags first
   - Skip remaining if emergent red flag detected

2. **Red Flag-Specific Follow-Up**
   - If perforation suspected, ask about timing
   - If sepsis suspected, ask about vitals

3. **Red Flag History**
   - Don't re-ask red flags already covered in diagnostic questions
   - Track which were already asked during Q&A phase

4. **Smart Escalation**
   - Some red flags → Emergent + 911
   - Other red flags → Urgent + ER today
   - Not all red flags require same escalation

---

**Last Updated:** October 2025  
**Safety Impact:** ✅ **Critical** - Ensures no dangerous symptoms are missed  
**Implementation Status:** ✅ Active and operational


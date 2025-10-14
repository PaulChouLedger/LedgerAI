# Triage Context Bug Fix - Mid-Session Condition Switching

## Problem Description

The triage system was experiencing a critical bug where it would switch to a completely different medical condition mid-triage session. For example:

**User Conversation:**
1. User: "Hey aura my name is Liam, I have found a lump"
2. Aura: "Where is it located and how long have you noticed it?"
3. User: "It's on my leg, I noticed it 2 days ago"
4. Aura: "Can you tell me more about what's on your leg? Is it red, swollen, or painful?"
5. User: "It's not red. It is swollen and is not painful"
6. **BUG**: Aura: "How long have you had pain or burning with urination?" ← **WRONG CONDITION (GU/dysuria)**

## Root Cause

The system was detecting a NEW medical condition mid-triage because:

1. **Missing Conversation Context**: The `phrasing_history` state variable (which tracks questions asked) was NEVER being updated when questions were asked to the user
2. **LLM Intent Classifier Confusion**: When the user answered "It's not red. It is swollen and is not painful", the LLM intent classifier received:
   - User's answer: "It's not red. It is swollen and is not painful"
   - **NO context about what question was asked**
   - Result: LLM detected "swollen" as a NEW medical symptom → triggered `leg_swelling` condition
3. **Cascading Detection**: Then when asked the first `leg_swelling` question, the user said "I haven't had pain", and the LLM detected "pain" → switched to `dysuria` (painful urination) condition

## The Fix

### 1. Update `phrasing_history` when questions are asked

**Files Modified:**
- `llm-container/triage.py` (lines 1008-1013, 1041-1048)
- `llm-container/container_rest.py` (lines 268-285)

**Changes:**
```python
# After generating each triage question, update phrasing_history
if "phrasing_history" not in state:
    state["phrasing_history"] = []
state["phrasing_history"].append(final_question)
state["phrasing_history"] = state["phrasing_history"][-10:]  # Keep last 10
```

This ensures the LLM intent classifier knows what question was just asked, so it can recognize answers as clarifications rather than new conditions.

### 2. Improve Intent Classifier Instructions

**File Modified:**
- `llm-container/intent_classifier.py` (lines 52-64)

**Changes:**
Added explicit context-aware rules:
```
CRITICAL RULES FOR CONTEXT AWARENESS:
- If the LAST ASSISTANT MESSAGE was a triage question (e.g., "Where is the pain?", "When did it start?"), 
  then the user's response is an ANSWER to that question, NOT a new medical condition
- "I haven't had pain" in response to "Do you have pain?" is a clarification = is_medical: false
- "It is swollen" in response to "Tell me more" during active triage is a clarification = is_medical: false
- ONLY classify as a NEW medical condition if the user introduces COMPLETELY NEW symptoms unprompted
```

## Impact

This fix prevents:
- Mid-triage condition switching
- False detection of new conditions from user answers
- Confusion and poor user experience

The triage system will now properly:
- Maintain the active condition until triage completion
- Recognize user answers as clarifications rather than new symptoms
- Only detect new conditions when truly appropriate (e.g., initial complaint)

## Testing Recommendation

Test the original scenario:
1. "I have found a lump" → Should detect a condition or ask clarifying question
2. Follow-up answers → Should continue with same condition
3. Verify no mid-session switching occurs

Also test edge cases:
- User mentions new symptom mid-triage (e.g., "Actually I also have chest pain") → Should this start new triage?
- Ambiguous answers that contain medical keywords → Should be treated as clarifications

## Files Changed

1. `/Users/rcabello/Documents/GitHub/LedgerAI/llm-container/triage.py`
2. `/Users/rcabello/Documents/GitHub/LedgerAI/llm-container/container_rest.py`
3. `/Users/rcabello/Documents/GitHub/LedgerAI/llm-container/intent_classifier.py`


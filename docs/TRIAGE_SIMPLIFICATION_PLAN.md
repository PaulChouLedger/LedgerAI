# Triage System Simplification Plan

## Current Problem

The triage completion generates garbled output like:
```
"You reported chest pain with associated You described the pain as reported Yes, 
You My arm. that pain worsens with exertion, and Yes. pain radiation starting 
Onset was yesterday..."
```

## Root Cause

The `build_recap()` function (200+ lines) tries to:
1. Parse all answers
2. Categorize into positives/negatives/priority
3. Extract timing information
4. Assemble fragments using templates
5. Deduplicate and clean

**Result:** Fragile, breaks easily, produces garbled text

## Simplified Approach

### Option 1: LLM-Generated Summary (Recommended)

Instead of template assembly, have the LLM summarize the triage conversation:

```python
def generate_triage_completion(state, session_id):
    """Use LLM to generate clean summary"""
    
    # Build context from conversation
    qa_pairs = []
    for step, answer in zip(steps, answers):
        qa_pairs.append(f"Q: {step['question']}\nA: {answer}")
    
    conversation = "\n".join(qa_pairs)
    
    # Ask LLM to summarize
    prompt = f"""Based on this triage conversation, write a concise medical summary:

{conversation}

Generate a professional 2-3 sentence summary of the patient's symptoms and recommended action.
Format: Patient reported [symptoms]. [Assessment]. [Recommendation]."""
    
    summary = call_llm(prompt)  # Clean, natural summary
    return summary
```

**Pros:**
- ✅ Natural language output
- ✅ No template fragmentation
- ✅ Handles edge cases gracefully
- ✅ Much simpler code (~20 lines vs 200+)

**Cons:**
- ⚠️ Slight latency increase (~500ms)
- ⚠️ Less deterministic (but more natural)

### Option 2: Ultra-Simple Template

Replace complex logic with single template:

```python
def generate_triage_completion(state, session_id):
    """Simple template-based summary"""
    condition = state.get("condition")
    severity = classify_response(condition, state.get("flags", {}))
    
    # Get clinical summary from JSON
    clinical_summary = TRIAGE_DEFS[condition].get("clinical_summary", "")
    outcome = get_outcome(condition, severity)
    
    # Simple, clean summary
    summary = f"{clinical_summary} This is classified as {severity}. {outcome}"
    return summary
```

**Pros:**
- ✅ Very simple (~10 lines)
- ✅ Fast, deterministic
- ✅ Clean output

**Cons:**
- ⚠️ Less personalized
- ⚠️ Doesn't recap specific answers

### Option 3: Hybrid Approach

Use templates for structure, LLM for content:

```python
def generate_triage_completion(state, session_id):
    """Template structure, LLM for symptom description"""
    
    # Get key findings
    main_symptoms = extract_key_symptoms(state)
    severity = classify_response(...)
    
    # LLM generates just the symptom description
    symptom_text = call_llm(f"Summarize in one sentence: {main_symptoms}")
    
    # Template provides structure
    clinical_summary = TRIAGE_DEFS[condition]["clinical_summary"]
    outcome = get_outcome(severity)
    
    return f"{symptom_text} {clinical_summary} {outcome}"
```

**Pros:**
- ✅ Balanced approach
- ✅ Clean, structured output
- ✅ Natural symptom descriptions

## Recommendation

**Go with Option 1 (LLM-Generated)** because:
1. The LLM is already running
2. It's much simpler code
3. Produces natural language
4. Handles edge cases automatically
5. Already proven to work well in other modes

## Implementation Steps

1. **Backup current triage.py**
2. **Replace build_recap()** with LLM call
3. **Simplify generate_triage_completion()**
4. **Test with chest pain scenario**
5. **Compare outputs**

## Expected Results

### Before (Current):
```
You reported chest pain with associated You described the pain as reported 
Yes, You My arm. that pain worsens with exertion...
```

### After (LLM Summary):
```
You reported chest pain radiating to your left arm that worsens with exertion, 
starting yesterday. Your symptoms are concerning for a heart attack or major 
cardiac event. This is classified as emergency. Please seek immediate care or call 911.
```

## Code Reduction

- **Current:** ~400 lines in triage.py for recap logic
- **Simplified:** ~50 lines total
- **Maintenance:** Much easier

Would you like me to implement Option 1 (LLM-generated summaries)?


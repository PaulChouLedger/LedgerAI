# LLM Parameter Optimization for Medical Navigator

## Current Settings Analysis

### Current Values
```
LLM_TOP_P=0.7
LLM_TOP_K=40
LLM_REPEAT_PENALTY=1.15
LLM_PRESENCE_PENALTY=0.0
LLM_FREQUENCY_PENALTY=0.2
LLM_NUM_PREDICT=40
LLM_TEMPERATURE_SIMPLE=0.2
LLM_STOP=n/n
```

### Use Case Breakdown

The medical navigator has **three distinct LLM tasks**:

1. **JSON Scoring/Matching** (deterministic, structured output)
   - Chief complaint → category matching
   - Patient response → term scoring
   - **Requires**: Temperature=0.0, high precision
   - **Current**: Hardcoded to 0.0 ✅ (correct)

2. **Question Generation** (natural, varied language)
   - OLDCARTS questions
   - Empathetic statements
   - Clarifying questions
   - **Requires**: Temperature=0.4-0.5, natural variation
   - **Current**: Hardcoded to 0.3-0.5 (should use env)

3. **Summary Generation** (concise, accurate)
   - History summary
   - **Requires**: Temperature=0.2-0.3, balanced
   - **Current**: Hardcoded to 0.2 (should use env)

## Optimized Settings

### For JSON Scoring Tasks (Deterministic)
```
LLM_TEMPERATURE_JSON=0.0        # Deterministic for structured output
LLM_TOP_P_JSON=0.9              # Focused sampling for JSON
LLM_TOP_K_JSON=40               # Reasonable token diversity
LLM_REPEAT_PENALTY_JSON=1.1     # Low penalty (JSON structure needs repetition)
LLM_PRESENCE_PENALTY_JSON=0.0   # No presence penalty
LLM_FREQUENCY_PENALTY_JSON=0.0  # No frequency penalty (JSON keys may repeat)
LLM_NUM_PREDICT_JSON=2000       # Enough for large JSON objects
```

### For Question Generation (Natural Language)
```
LLM_TEMPERATURE_QUESTIONS=0.4   # Balanced for natural but consistent questions
LLM_TOP_P_QUESTIONS=0.8         # Slightly higher for variety
LLM_TOP_K_QUESTIONS=50          # More token diversity for natural language
LLM_REPEAT_PENALTY_QUESTIONS=1.2 # Higher penalty to avoid repetitive questions
LLM_PRESENCE_PENALTY_QUESTIONS=0.1 # Encourage diverse question phrasing
LLM_FREQUENCY_PENALTY_QUESTIONS=0.3 # Higher penalty to avoid word repetition
LLM_NUM_PREDICT_QUESTIONS=120   # Enough for complete questions with options
```

### For Summary Generation
```
LLM_TEMPERATURE_SUMMARY=0.25    # Slightly higher than JSON, lower than questions
LLM_TOP_P_SUMMARY=0.75          # Balanced sampling
LLM_TOP_K_SUMMARY=45            # Moderate diversity
LLM_REPEAT_PENALTY_SUMMARY=1.15 # Moderate penalty
LLM_PRESENCE_PENALTY_SUMMARY=0.05 # Low presence penalty
LLM_FREQUENCY_PENALTY_SUMMARY=0.2 # Moderate frequency penalty
LLM_NUM_PREDICT_SUMMARY=300     # Enough for concise summaries
```

## Recommended Universal Settings (Backward Compatible)

For simplicity, we'll optimize the existing environment variables for the **most common use case** (question generation), while allowing task-specific overrides:

```
# Primary settings (used for question generation and general tasks)
LLM_TEMPERATURE_SIMPLE=0.4      # Increased from 0.2 for more natural questions
LLM_TOP_P=0.8                   # Increased from 0.7 for better variety
LLM_TOP_K=50                    # Increased from 40 for more diversity
LLM_REPEAT_PENALTY=1.2          # Increased from 1.15 to avoid repetitive questions
LLM_PRESENCE_PENALTY=0.1        # Increased from 0.0 to encourage diversity
LLM_FREQUENCY_PENALTY=0.3       # Increased from 0.2 to avoid word repetition
LLM_NUM_PREDICT=120             # Increased from 40 for complete questions
LLM_STOP=n/n                    # Keep as is
```

## Rationale

### Temperature (0.2 → 0.4)
- **0.2**: Too deterministic, produces robotic questions
- **0.4**: Balanced - natural but consistent questions
- **0.0**: Reserved for JSON scoring (deterministic)

### TOP_P (0.7 → 0.8)
- **0.7**: Too focused, limited variety in questions
- **0.8**: Better balance - natural variation without being too random
- **0.9**: Too random for medical questions (reserved for JSON)

### TOP_K (40 → 50)
- **40**: Adequate for 1B model
- **50**: Better diversity for natural language generation
- **60+**: Too many tokens for 1B model (may cause incoherence)

### Repeat Penalty (1.15 → 1.2)
- **1.15**: Too low, allows repetitive phrasing
- **1.2**: Better for avoiding repetitive questions
- **1.3+**: Too high, may cause unnatural breaks

### Presence Penalty (0.0 → 0.1)
- **0.0**: No diversity incentive
- **0.1**: Encourages diverse question phrasing
- **0.2+**: Too high, may cause incoherence

### Frequency Penalty (0.2 → 0.3)
- **0.2**: Too low, allows word repetition
- **0.3**: Better for avoiding repetitive words
- **0.4+**: Too high, may cause unnatural breaks

### NUM_PREDICT (40 → 120)
- **40**: Too short for complete questions with options
- **120**: Enough for questions like "Where is your pain? You can mention things like: behind your breastbone, upper middle part of your belly."
- **200+**: Too long, may cause rambling

## Implementation

The navigator will:
1. Use **temperature=0.0** for JSON scoring tasks (hardcoded, correct)
2. Use **environment variables** for question generation (currently hardcoded)
3. Allow **task-specific overrides** via function parameters
4. Maintain **backward compatibility** with existing environment variables

## Testing

After optimization, test:
1. Question variety and naturalness
2. JSON scoring accuracy (should remain deterministic)
3. Summary quality and conciseness
4. Overall conversation flow


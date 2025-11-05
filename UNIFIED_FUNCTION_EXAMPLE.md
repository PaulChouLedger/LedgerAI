# Unified Function Step-by-Step Example

## Scenario: Patient says "right side" → Checking Diverticulitis (left_lower)

### Input:
- **Patient text**: "right side"
- **Guideline**: Diverticulitis
- **Location data**: `{"medical": "left lower quadrant", "anatomical_type": "left_lower"}`
- **Organ system**: "GI"

---

## Step 1: Raw Semantic Similarity (Embeddings)

```python
patient_embedding = embedding_model.encode("right side")
guideline_embedding = embedding_model.encode("left lower quadrant")

raw_similarity = cosine_similarity(patient_embedding, guideline_embedding)
# Result: ~0.45 (moderate semantic similarity - both are location terms)
```

**Why**: "right side" and "left lower quadrant" are semantically similar (both locations), so embeddings show some similarity even though directions are opposite.

---

## Step 2: Normalization (Synonym Expansion)

```python
# Load synonyms from gi_synonyms_oldcarts.json
synonyms = {
    "location": {
        "right_side": ["right side", "right", "right-sided", "on the right", ...],
        "left_lower_quadrant": ["left lower quadrant", "lower left", "llq", ...]
    }
}

# Normalize "right side" using synonyms
normalized_text = "right side"  # Already normalized
```

**Why**: Synonym expansion helps match variations, but in this case the patient text is already clear.

---

## Step 3: Word Match Boost (WHERE ANATOMICAL MISMATCH IS DETECTED)

### 3.1: Extract Anatomical Components

```python
# Patient components
patient_components = _extract_anatomical_components("right side")
# Uses medical_rules.json["anatomical_components"]["directional_keywords"]["horizontal"]["right"]
# Result: {"horizontal": "right"}

# Guideline location terms
includes_terms = ["left lower quadrant"]
condition_components = _extract_anatomical_components("left lower quadrant")
# Uses medical_rules.json["anatomical_components"]["quadrant_patterns"]["left_lower"]
# Result: {"horizontal": "left", "vertical": "lower"}
```

### 3.2: Check Anatomical Opposites

```python
# Check using medical_rules.json["anatomical_opposites"]
patient_components = {"horizontal": "right"}
condition_components = {"horizontal": "left", "vertical": "lower"}

# Query medical_rules.json["anatomical_opposites"]["horizontal"]
opposites = {
    "horizontal": {
        "left": ["right"],  # ← left is opposite of right
        "right": ["left"]   # ← right is opposite of left
    },
    "vertical": {
        "upper": ["lower"],
        "lower": ["upper"]
    }
}

# Check: Is "left" in the opposite list of "right"?
is_opposite = _are_anatomical_opposites(patient_components, condition_components)
# Logic:
#   1. Extract horizontal: "right" vs "left"
#   2. Check opposites["horizontal"]["right"] → contains "left"
#   3. Result: True ✅ MISMATCH DETECTED!
```

### 3.3: Apply Penalty

```python
# Since all anatomically-specific terms are mismatched:
if anatomically_specific_terms and all_specific_mismatched and not has_matching_term:
    return -0.3  # ⚠️ PENALTY FOR ANATOMICAL MISMATCH
```

**Result**: `word_match_boost = -0.3`

---

## Step 4: Combine Results

```python
final_similarity = raw_similarity + word_match_boost
final_similarity = 0.45 + (-0.3)
final_similarity = 0.15

# Clamp to [0.0, 1.0]
final_similarity = max(0.0, min(1.0, 0.15))
# Result: 0.15 (low score due to penalty)
```

---

## Step 5: Filtering Decision

```python
# In filter_guidelines_by_location():
if final_score < 0.2:  # Threshold: 0.2 (handles penalties properly)
    continue  # Filter out - no meaningful match or mismatch

# Since 0.15 < 0.2, guideline is FILTERED OUT ✅
```

**Result**: Diverticulitis is correctly filtered out because:
- Raw similarity: 0.45 (semantic match)
- Word match boost: -0.3 (anatomical mismatch penalty)
- Final score: 0.15 < 0.2 threshold → **FILTERED OUT**

---

## Example 2: Matching Case (Right Side → Right Lower Quadrant)

### Input:
- **Patient**: "right side"
- **Guideline**: Appendicitis (right_lower)

### Step 1: Raw Semantic Similarity
```python
raw_similarity = 0.52  # High similarity - both are right-sided locations
```

### Step 2: Normalization
```python
normalized_text = "right side"
```

### Step 3: Word Match Boost

```python
patient_components = {"horizontal": "right"}
condition_components = {"horizontal": "right", "vertical": "lower"}

# Check opposites
is_opposite = _are_anatomical_opposites(patient_components, condition_components)
# "right" vs "right" → NOT opposites → False

# Check for substring match
if "right lower quadrant" in "right side" or "right side" in "right lower quadrant":
    # Partial match found
    return 0.3  # ✅ BOOST FOR MATCH
```

### Step 4: Final Score
```python
final_similarity = 0.52 + 0.3 = 0.82  # High score!
```

### Step 5: Filtering
```python
if 0.82 < 0.2:  # False
    continue  # Not filtered

# Guideline is KEPT ✅
```

---

## Example 3: More Severe Mismatch (Both Horizontal + Vertical)

### Input:
- **Patient**: "right upper quadrant"
- **Guideline**: Diverticulitis (left_lower)

### Step 3.2: Check Opposites

```python
patient_components = {"horizontal": "right", "vertical": "upper"}
condition_components = {"horizontal": "left", "vertical": "lower"}

# Check horizontal mismatch
horizontal_mismatch = "left" in opposites["horizontal"]["right"]  # True

# Check vertical mismatch  
vertical_mismatch = "lower" in opposites["vertical"]["upper"]  # True

# Both mismatched → severe penalty
# Result: -0.3 penalty
```

### Step 4: Final Score

```python
raw_similarity = 0.50  # Similar location terms
word_match_boost = -0.3  # Both horizontal AND vertical mismatch
final_similarity = 0.50 + (-0.3) = 0.20
```

### Step 5: Filtering
```python
if 0.20 < 0.2:  # False (exactly at threshold)
    continue  # Not filtered (but close!)

# Actually, with threshold 0.2, this would pass
# But score is very low, so it won't rank highly
```

**Note**: Double mismatch (both horizontal and vertical) still gets -0.3 penalty. We might want a more severe penalty for multiple mismatches.

---

## Key Points:

1. **Anatomical mismatch detection happens in `_compute_word_match_boost`** (Step 3)
2. **Uses `_are_anatomical_opposites`** which reads from `medical_rules.json["anatomical_opposites"]`
3. **Applies -0.3 penalty** when mismatch detected
4. **Final score = raw_similarity + word_match_boost**
5. **Filtering threshold is 0.2** - scores below this are filtered out
6. **Penalty is applied even if semantic similarity is high** - this ensures anatomical correctness

---

## Code Flow:

```
compute_unified_similarity()
  ├─ Step 1: Raw semantic similarity (embeddings)
  ├─ Step 2: Normalization (synonyms)
  └─ Step 3: Word match boost
       ├─ _extract_anatomical_components() → Extract patient & condition components
       ├─ _are_anatomical_opposites() → Check if opposites (using medical_rules.json)
       └─ Return -0.3 if mismatch, +0.3/+0.5 if match
  └─ Step 4: Combine (raw_similarity + word_match_boost)
```

---

## Why This Works:

1. **Semantic similarity alone is not enough** - "right side" and "left lower quadrant" are semantically similar (both locations), but anatomically opposite
2. **Penalty ensures correctness** - Even if embeddings show similarity, the -0.3 penalty reduces the score below threshold
3. **Unified function handles everything** - No need for separate filtering logic
4. **Works for all organ systems** - Uses universal `medical_rules.json` definitions

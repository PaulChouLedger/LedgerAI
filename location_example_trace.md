# Location Processing Example: "right side"

## Scenario
**Patient Answer:** "right side"  
**Category:** Gastrointestinal  
**OLDCARTS Element:** location

---

## STEP 1: Filter Guidelines by Anatomical Opposites (medical_rules.json)

**Function:** `filter_guidelines_by_location()` in `medical_rule_engine.py`

### 1.1 Extract Patient Direction
- **Input:** Patient answer "right side"
- **Process:** 
  - FAISS finds location matches across all guidelines
  - Extracts directional component: `patient_direction = "right"`
  - Uses `medical_rules.json` to identify anatomical components:
    ```json
    {
      "horizontal": "right",
      "vertical": null,
      "bilateral": false,
      "vague": false
    }
    ```

### 1.2 Filter Guidelines
- **Before Filtering:** 10 guidelines (e.g., Appendicitis, Cholecystitis, Left-sided Diverticulitis, etc.)
- **Process:**
  - For each guideline, check `anatomical_type` from guideline JSON
  - Apply filtering rules from `medical_rules.json`:
    - **Right-only conditions** (e.g., Cholecystitis): ✅ KEEP (matches "right")
    - **Left-only conditions** (e.g., Left Diverticulitis): ❌ REMOVE (opposite of "right")
    - **Bilateral conditions** (e.g., Irritable Bowel Syndrome): ✅ KEEP (compatible with any direction)
    - **Vague conditions** (e.g., Diffuse Abdominal Pain): ✅ KEEP (compatible with any direction)

- **After Filtering:** 7 guidelines (removed 3 left-only conditions)

**Debug Output:**
```
[Engine] 🏥 Filtering guidelines using medical_rules.json
[Engine] 🔍 Before filtering (first 3): 
  - Appendicitis (score: 0.65)
  - Cholecystitis (score: 0.72)
  - Left Diverticulitis (score: 0.58)
  
[Engine] 🔍 After filtering (first 3):
  - Appendicitis (score: 0.65)
  - Cholecystitis (score: 0.72)
  [Left Diverticulitis REMOVED - anatomical mismatch]
```

---

## STEP 2: Score Guidelines with Patient-Friendly Semantic Matching

**Function:** `_process_clinical_answer()` → `_match_to_patient_friendly_terms()`

### 2.1 Collect Patient-Friendly Terms
For each remaining guideline, collect all `patient_friendly` terms from the location element:

**Example Guideline: Appendicitis**
```json
{
  "location": {
    "includes": [
      {
        "medical": "right lower quadrant",
        "patient_friendly": "right lower part of abdomen"
      },
      {
        "medical": "RLQ",
        "patient_friendly": "right lower part"
      },
      {
        "medical": "periumbilical",
        "patient_friendly": "around belly button"
      }
    ]
  }
}
```

### 2.2 Semantic Matching
- **Input:** Patient answer "right side"
- **Process:**
  - Encode patient answer: `[0.12, -0.45, 0.78, ...]` (embedding vector)
  - Encode all patient_friendly terms:
    - "right lower part of abdomen": `[0.15, -0.42, 0.81, ...]`
    - "right lower part": `[0.14, -0.41, 0.79, ...]`
    - "around belly button": `[0.08, -0.38, 0.65, ...]`
  - Calculate cosine similarity for each:
    - "right lower part of abdomen": **0.85** (highest)
    - "right lower part": **0.82**
    - "around belly button": **0.45**

- **Result:** Similarity = **0.85** (highest match)

### 2.3 Update Guideline Scores
- **Appendicitis:**
  - Old score: 0.65
  - Location similarity: 0.85
  - New score: `(0.65 * 0.7) + (0.85 * 0.3) = 0.71`

- **Cholecystitis:**
  - Old score: 0.72
  - Location similarity: 0.78 (matches "right upper part")
  - New score: `(0.72 * 0.7) + (0.78 * 0.3) = 0.74`

**Debug Output:**
```
[Scoring] 🔍 Scoring 7 guidelines for element: location
[Scoring] 📝 Patient answer: 'right side'
[Scoring] 📊 Appendicitis: old=0.650, location=0.850, new=0.710
[Scoring] 📊 Cholecystitis: old=0.720, location=0.780, new=0.740
```

---

## STEP 3: Use FAISS to Find Missing Terms for Clarifying Questions

**Function:** `_analyze_missing_information()` → `find_matching_terms_faiss()`

### 3.1 FAISS Semantic Search
- **Input:** Patient answer "right side", element="location"
- **Process:**
  - Query FAISS index with patient answer
  - Search across ALL location terms from ALL guidelines
  - Find semantically similar terms above threshold (0.6):
    - "right side": **0.95** ✅
    - "right lower quadrant": **0.82** ✅
    - "right upper quadrant": **0.79** ✅
    - "left side": **0.15** ❌ (below threshold)
    - "left upper quadrant": **0.12** ❌ (below threshold)

- **Semantic Matches:** `{"right side": 0.95, "right lower quadrant": 0.82, "right upper quadrant": 0.79}`

### 3.2 Compute Patient-Friendly Similarity Scores
For each FAISS-matched term, compute similarity using patient_friendly matching:

- **"right side"** → similarity: **0.95** (exact match)
- **"right lower quadrant"** → similarity: **0.85** (semantically similar)
- **"right upper quadrant"** → similarity: **0.78** (semantically similar)

**Debug Output:**
```
[Location Analysis] 🔍 FAISS found 3 matches above threshold: ['right side', 'right lower quadrant', 'right upper quadrant']
[Location Analysis] 🔍 Patient-friendly similarity scores: 
  {
    'right side': 0.95,
    'right lower quadrant': 0.85,
    'right upper quadrant': 0.78
  }
```

---

## STEP 4: Apply Anatomical Filtering to Missing Terms

**Function:** `_analyze_missing_information()` → anatomical filtering logic

### 4.1 Extract Patient Components
- **Input:** Normalized answer "right side"
- **Extracted Components:**
  ```python
  {
    "horizontal": "right",
    "vertical": null,
    "bilateral": false,
    "vague": false
  }
  ```

### 4.2 Check Each Potential Missing Term

**All Location Terms from Active Guidelines:**
- "right side" ✅ (already satisfied)
- "right lower quadrant"
- "right upper quadrant"
- "left side" ❌ (anatomical opposite)
- "left lower quadrant" ❌ (anatomical opposite)
- "left upper quadrant" ❌ (anatomical opposite)
- "periumbilical" ✅ (no horizontal direction - compatible)
- "epigastric" ✅ (no horizontal direction - compatible)

### 4.3 Filter by Anatomical Compatibility

For each term, check if it's anatomically compatible:

1. **"right lower quadrant"**
   - Components: `{"horizontal": "right", "vertical": "lower"}`
   - ✅ Compatible: same horizontal ("right")
   - ✅ **INCLUDED in missing**

2. **"right upper quadrant"**
   - Components: `{"horizontal": "right", "vertical": "upper"}`
   - ✅ Compatible: same horizontal ("right")
   - ✅ **INCLUDED in missing**

3. **"left side"**
   - Components: `{"horizontal": "left", "vertical": null}`
   - ❌ Incompatible: opposite horizontal ("left" vs "right")
   - ❌ **EXCLUDED from missing**

4. **"periumbilical"**
   - Components: `{"horizontal": null, "vertical": null}`
   - ✅ Compatible: no horizontal direction (vague/midline)
   - ✅ **INCLUDED in missing**

5. **"epigastric"**
   - Components: `{"horizontal": null, "vertical": "upper"}`
   - ✅ Compatible: no horizontal direction
   - ✅ **INCLUDED in missing**

**Final Missing Terms:** `["right lower quadrant", "right upper quadrant", "periumbilical", "epigastric"]`

**Debug Output:**
```
[Location Analysis] 🔍 Anatomical filtering: Patient components = {'horizontal': 'right', 'vertical': None}
[Location Analysis] 🔍 Checking 'right lower quadrant': components = {'horizontal': 'right', 'vertical': 'lower'}
[Location Analysis] ✅ 'right lower quadrant' INCLUDED in missing (compatible: patient right = term right)
[Location Analysis] 🔍 Checking 'left side': components = {'horizontal': 'left', 'vertical': None}
[Location Analysis] ❌ 'left side' EXCLUDED from missing (opposite horizontal: patient right ≠ term left)
[Location Analysis] ✅ 'periumbilical' INCLUDED in missing (vague term, no horizontal direction, compatible with any)
```

---

## STEP 5: Generate Clarifying Question

**Function:** `_generate_location_clarification_question()`

### 5.1 Build Question Options
- **Missing Terms:** `["right lower quadrant", "right upper quadrant", "periumbilical", "epigastric"]`
- **Patient-Friendly Conversion:**
  - "right lower quadrant" → "right lower part of your abdomen"
  - "right upper quadrant" → "right upper part of your abdomen"
  - "periumbilical" → "around your belly button"
  - "epigastric" → "upper middle part, below your breastbone"

### 5.2 Generate Question
**Question:** "Can you be more specific? For example, is it located at the right lower part of your abdomen, the right upper part of your abdomen, around your belly button, or the upper middle part below your breastbone?"

---

## Summary of New Logic Flow

1. **✅ STEP 1: Anatomical Filtering** - Removes incompatible guidelines using `medical_rules.json`
   - Patient says "right" → removes all "left-only" conditions
   - Keeps: right-only, bilateral, vague conditions

2. **✅ STEP 2: Patient-Friendly Matching** - Scores remaining guidelines using semantic similarity
   - Matches "right side" to patient_friendly terms like "right lower part"
   - Updates guideline scores based on similarity (0.0-1.0)

3. **✅ STEP 3: FAISS Missing Terms** - Finds semantically related terms for clarifying questions
   - Uses FAISS to find all location terms similar to "right side"
   - Computes patient_friendly similarity scores for each match

4. **✅ STEP 4: Anatomical Filtering of Missing Terms** - Filters out incompatible locations
   - Only includes terms compatible with patient's "right" direction
   - Excludes "left side", "left upper quadrant", etc.

5. **✅ STEP 5: Generate Clarifying Question** - Creates question with compatible options only

---

## Key Improvements

- **Simplified:** No more unified function complexity for location
- **Consistent:** Uses same patient_friendly matching as other OLDCARTS elements
- **Accurate:** Anatomical filtering prevents asking about incompatible locations
- **Efficient:** FAISS quickly finds relevant terms across all guidelines


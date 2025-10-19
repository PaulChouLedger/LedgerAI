# Hybrid Similarity Matching Examples

## How the Hybrid System Works

The hybrid system combines **Jaccard similarity** (primary) with **semantic similarity** (fallback) to provide the best of both worlds:

1. **Jaccard Similarity**: Fast, precise matching on normalized medical terms
2. **Semantic Similarity**: Handles edge cases and complex patient language
3. **Anatomical Filtering**: Prevents impossible matches (RUQ vs RLQ)

## Example 1: Perfect Jaccard Match (High Confidence)

### Patient Input:
```
"my tummy hurts in upper right"
```

### Processing:
1. **OLDCARTS Normalization**: `"abdominal pain right upper quadrant"`
2. **Anatomical Filtering**: ✅ No conflicts
3. **Jaccard Similarity**: `0.75` (high match)
4. **Semantic Similarity**: `0.68` (good match)
5. **Final Score**: `0.75` (Jaccard wins - high confidence)

### Result:
```
✅ Acute Cholecystitis: 0.750 (jaccard, high confidence)
   Jaccard: 0.750, Semantic: 0.680
   Location: "Right upper quadrant (RUQ), precisely localized just below right rib cage"
```

## Example 2: Semantic Fallback (Medium Confidence)

### Patient Input:
```
"I have a sharp pain in my side that gets worse when I breathe"
```

### Processing:
1. **OLDCARTS Normalization**: `"sharp pain side worse breathing"`
2. **Anatomical Filtering**: ✅ No conflicts
3. **Jaccard Similarity**: `0.25` (low match - no word overlap)
4. **Semantic Similarity**: `0.72` (high match - understands "side" + "breathing" = pleuritic)
5. **Final Score**: `0.504` (semantic fallback with 0.7 weight)

### Result:
```
✅ Pleurisy: 0.504 (semantic_fallback, medium confidence)
   Jaccard: 0.250, Semantic: 0.720
   Location: "Chest pain, worse with inspiration, may radiate to shoulder"
```

## Example 3: Anatomical Mismatch (Rejected)

### Patient Input:
```
"my tummy hurts in upper right"
```

### Processing:
1. **OLDCARTS Normalization**: `"abdominal pain right upper quadrant"`
2. **Anatomical Filtering**: ❌ **RUQ vs RLQ conflict**
3. **Result**: **REJECTED** - No similarity calculation needed

### Result:
```
❌ Acute Appendicitis: ANATOMICAL MISMATCH - skipping
   🚫 ANATOMICAL CONFLICT: RUQ vs RLQ
```

## Example 4: Hybrid Consensus (High Confidence)

### Patient Input:
```
"severe abdominal pain in my belly"
```

### Processing:
1. **OLDCARTS Normalization**: `"severe abdominal pain belly"`
2. **Anatomical Filtering**: ✅ No conflicts
3. **Jaccard Similarity**: `0.67` (good match)
4. **Semantic Similarity**: `0.71` (good match)
5. **Final Score**: `0.67` (both agree - high confidence)

### Result:
```
✅ Acute Gastroenteritis: 0.670 (jaccard, high confidence)
   Jaccard: 0.670, Semantic: 0.710
   Location: "PERIUMBILICAL or DIFFUSE throughout abdomen. NOT localized to one quadrant"
```

## Example 5: Low Confidence (Both Methods Fail)

### Patient Input:
```
"my head hurts really bad"
```

### Processing:
1. **OLDCARTS Normalization**: `"head pain severe"`
2. **Anatomical Filtering**: ✅ No conflicts
3. **Jaccard Similarity**: `0.15` (low match)
4. **Semantic Similarity**: `0.23` (low match)
5. **Final Score**: `0.15` (both low - low confidence)

### Result:
```
❌ Acute Cholecystitis: 0.150 (jaccard, low confidence)
   Jaccard: 0.150, Semantic: 0.230
   Location: "Right upper quadrant (RUQ), precisely localized just below right rib cage"
   REJECTED: final: 0.150 < 0.200
```

## Configuration Parameters

```python
hybrid_config = {
    'jaccard_threshold': 0.3,      # Primary threshold for Jaccard similarity
    'semantic_threshold': 0.5,     # Threshold for semantic similarity fallback
    'semantic_boost_threshold': 0.3,  # When semantic is significantly better than Jaccard
    'semantic_weight': 0.7,        # Weight for semantic similarity when used as fallback
    'confidence_threshold': 0.1    # Max difference for high confidence
}
```

## Decision Logic

```python
if jaccard_score > 0.3:
    # High Jaccard confidence - use it as primary
    final_score = jaccard_score
    confidence = "high"
    method_used = "jaccard"
    
elif semantic_score > jaccard_score + 0.3:
    # Semantic significantly better - use it with lower weight
    final_score = semantic_score * 0.7
    confidence = "medium"
    method_used = "semantic_fallback"
    
elif semantic_score > 0.5:
    # Semantic above threshold but not significantly better
    final_score = max(jaccard_score, semantic_score * 0.8)
    confidence = "medium"
    method_used = "hybrid"
    
else:
    # Use Jaccard as primary
    final_score = jaccard_score
    confidence = "low" if jaccard_score < 0.2 else "medium"
    method_used = "jaccard"
```

## Benefits of Hybrid Approach

1. **Reliability**: Jaccard similarity provides consistent, interpretable results
2. **Edge Case Handling**: Semantic similarity catches complex patient language
3. **Anatomical Accuracy**: Prevents impossible matches (RUQ vs RLQ)
4. **Confidence Scoring**: Know when to trust the results
5. **Fallback Safety**: System works even if semantic model fails
6. **Performance**: Jaccard is fast, semantic only used when needed

## Real-World Scenarios

### Scenario 1: Clear Medical Terms
- **Patient**: "I have right upper quadrant pain"
- **Result**: High Jaccard similarity, high confidence
- **Method**: Jaccard (primary)

### Scenario 2: Lay Language
- **Patient**: "My tummy hurts in the upper right"
- **Result**: Good Jaccard similarity after normalization, high confidence
- **Method**: Jaccard (primary)

### Scenario 3: Complex Description
- **Patient**: "I have this weird pain in my side that gets worse when I take deep breaths"
- **Result**: Low Jaccard, high semantic similarity, medium confidence
- **Method**: Semantic fallback

### Scenario 4: Anatomical Mismatch
- **Patient**: "Upper right pain"
- **Guideline**: "Right lower quadrant (RLQ)"
- **Result**: Rejected before similarity calculation
- **Method**: Anatomical filtering

This hybrid approach ensures both accuracy and reliability in medical diagnosis matching.

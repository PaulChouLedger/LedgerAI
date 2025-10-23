# Condition Scoring Analysis - Diagnostic Engine

## Overview
This analysis extracts all conditions from the diagnostic engine log output and organizes their scoring data to help develop a better scoring mechanism.

## Scoring Methods Identified

### 1. **Hybrid Balanced** (`hybrid_balanced`)
- **Formula**: 60% Jaccard + 40% Semantic similarity
- **Use Case**: When both Jaccard and semantic similarities are available
- **Confidence**: High
- **Example**: Acute Pancreatitis, Acute Diverticulitis, Sigmoid Volvulus

### 2. **Semantic Boosted** (`semantic_boosted`)
- **Formula**: `min(semantic_score * 1.2, 0.8)`
- **Use Case**: When only semantic similarity is available
- **Confidence**: Medium
- **Cap**: 0.8 maximum

### 3. **Anatomical Exclusion** (`anatomical_exclusion`)
- **Formula**: Returns 0.0 (hard mismatch)
- **Use Case**: When anatomical mismatch is detected
- **Criteria**: 
  - Jaccard similarity = 0.0 (no word overlap)
  - OR both semantic < 0.15 AND jaccard < 0.1
- **Penalty**: 0.6x reduction in score

### 4. **Similarity Weighted** (`similarity-weighted`)
- **Formula**: Score adjusted based on similarity score
- **Use Case**: General similarity-based adjustments
- **Confidence**: Variable

## Condition Scoring Data

### 📊 **Summary Statistics**
- **Total Conditions**: 25
- **Anatomical Exclusions**: 22 (88%)
- **Hybrid Balanced**: 3 (12%)
- **Average Score Reduction**: 28.8%

---

### 🔴 **ANATOMICAL EXCLUSIONS** (Hard Mismatches)
*All conditions excluded due to Jaccard similarity = 0.0 (no word overlap)*

#### **High Priority Conditions** (60% → 24%)
| Condition | Score Change | Semantic | Penalty | Reason |
|-----------|--------------|----------|---------|---------|
| **Acute Appendicitis** | ❌ -36% | 0.097 | 0.6x | Right vs Left |
| **Acute Cholecystitis** | ❌ -36% | 0.234 | 0.6x | Right vs Left |
| **Biliary Colic** | ❌ -36% | 0.271 | 0.6x | Right vs Left |
| **Peptic Ulcer Disease** | ❌ -36% | 0.102 | 0.6x | Right vs Left |
| **Kidney Stone** | ❌ -36% | 0.115 | 0.6x | Right vs Left |
| **UTI/Pyelonephritis** | ❌ -36% | 0.055 | 0.6x | Right vs Left |

#### **Medium Priority Conditions** (50% → 20%)
| Condition | Score Change | Semantic | Penalty | Reason |
|-----------|--------------|----------|---------|---------|
| **Small Bowel Obstruction** | ❌ -30% | 0.042 | 0.6x | Right vs Left |
| **Acute Cholangitis** | ❌ -30% | 0.238 | 0.6x | Right vs Left |
| **Severe Constipation** | ❌ -30% | 0.082 | 0.6x | Right vs Left |
| **Gastric Outlet Obstruction** | ❌ -30% | 0.136 | 0.6x | Right vs Left |
| **Acute Gastritis** | ❌ -30% | 0.089 | 0.6x | Right vs Left |
| **IBD Flare (Crohn's/UC)** | ❌ -30% | 0.092 | 0.6x | Right vs Left |
| **Irritable Bowel Syndrome** | ❌ -30% | 0.095 | 0.6x | Right vs Left |
| **Incarcerated Hernia** | ❌ -30% | 0.083 | 0.6x | Right vs Left |
| **Ruptured Ovarian Cyst** | ❌ -30% | 0.131 | 0.6x | Right vs Left |

#### **Low Priority Conditions** (40% → 16%)
| Condition | Score Change | Semantic | Penalty | Reason |
|-----------|--------------|----------|---------|---------|
| **Cecal Volvulus** | ❌ -24% | 0.171 | 0.6x | Right vs Left |
| **Acute Hepatitis** | ❌ -24% | 0.011 | 0.6x | Right vs Left |
| **Acute Mesenteric Ischemia** | ❌ -24% | 0.048 | 0.6x | Right vs Left |
| **Perforated Viscus** | ❌ -24% | 0.035 | 0.6x | Right vs Left |
| **Ectopic Pregnancy** | ❌ -24% | 0.138 | 0.6x | Right vs Left |
| **Ovarian Torsion** | ❌ -24% | 0.203 | 0.6x | Right vs Left |

#### **Special Cases**
| Condition | Score Change | Semantic | Penalty | Reason |
|-----------|--------------|----------|---------|---------|
| **Acute Gastroenteritis** | ❌ -36% | 0.043 | 0.6x | Diffuse vs Localized |

---

### 🟡 **HYBRID BALANCED** (Partial Matches)
*Conditions with both Jaccard and Semantic similarity*

| Condition | Initial | Final | Change | Semantic | Jaccard | Method | Status |
|-----------|---------|-------|---------|----------|---------|---------|---------|
| **Acute Pancreatitis** | 60% | 30% | ↓ -30% | 0.155 | 0.056 | Hybrid Balanced | Similarity Weighted |
| **Acute Diverticulitis** | 50% | 28% | ↓ -22% | 0.225 | 0.062 | Hybrid Balanced | Similarity Weighted |
| **Sigmoid Volvulus** | 40% | 29% | ↓ -11% | 0.277 | 0.167 | Hybrid Balanced | Similarity Weighted |

---

### 📈 **Scoring Method Distribution**

```
Anatomical Exclusion: ████████████████████████ 88% (22/25)
Hybrid Balanced:     ████ 12% (3/25)
Semantic Boosted:    ░░░░ 0% (0/25)
```

---

### 🎯 **Key Insights**

#### **Problem Areas**
- **88% Exclusion Rate**: Overly aggressive anatomical filtering
- **Binary Jaccard**: No partial credit for related terms
- **Fixed Penalties**: Same 0.6x penalty regardless of semantic similarity
- **Semantic Underutilization**: High semantic scores ignored when Jaccard = 0

#### **Success Cases**
- **Sigmoid Volvulus**: Best performer (29% final score)
- **Acute Diverticulitis**: Good semantic match (0.225)
- **Acute Pancreatitis**: Moderate match (0.155)

#### **Improvement Opportunities**
- **Anatomical Mapping**: Left ↔ Right equivalents
- **Fuzzy Jaccard**: Word embedding similarity
- **Dynamic Penalties**: Based on semantic score
- **Semantic-First**: Use semantic when Jaccard fails

## Key Insights

### 1. **Anatomical Exclusion Dominance**
- **22 out of 25 conditions** (88%) triggered anatomical exclusion
- All exclusions due to **Jaccard similarity = 0.0** (no word overlap)
- Patient answer: "left side" vs. conditions expecting "right" locations

### 2. **Scoring Method Distribution**
- **Anatomical Exclusion**: 22 conditions (88%)
- **Hybrid Balanced**: 3 conditions (12%)
- **Semantic Boosted**: 0 conditions (0%)

### 3. **Score Reduction Patterns**
- **Hard Mismatch**: 0.6x penalty (most common)
- **Similarity Weighted**: 0.5x-0.725x penalty
- **Average Score Reduction**: 28.8%

### 4. **Semantic vs Jaccard Analysis**
- **Semantic Range**: 0.011 - 0.277
- **Jaccard Range**: 0.000 - 0.167
- **Correlation**: Weak (many high semantic, zero Jaccard)

## Scoring Mechanism Issues

### 1. **Over-Aggressive Anatomical Exclusion**
- **Problem**: 88% of conditions excluded due to single word mismatch
- **Impact**: "left side" excludes all "right" conditions
- **Solution**: Implement anatomical mapping (left ↔ right equivalents)

### 2. **Jaccard Similarity Limitations**
- **Problem**: Binary word matching (0 or 1)
- **Impact**: No partial credit for related terms
- **Solution**: Implement fuzzy word matching or semantic word similarity

### 3. **Semantic Similarity Underutilization**
- **Problem**: Semantic scores ignored when Jaccard = 0
- **Impact**: Misses semantic relationships
- **Solution**: Use semantic similarity as primary when Jaccard fails

### 4. **Penalty System Issues**
- **Problem**: Fixed 0.6x penalty regardless of semantic similarity
- **Impact**: High semantic similarity still gets harsh penalty
- **Solution**: Dynamic penalty based on semantic score

## Recommended Improvements

### 1. **Anatomical Mapping System**
```python
anatomical_mappings = {
    'left': ['right', 'contralateral'],
    'right': ['left', 'contralateral'],
    'upper': ['lower', 'inferior'],
    'lower': ['upper', 'superior']
}
```

### 2. **Hybrid Scoring with Anatomical Awareness**
```python
def enhanced_scoring(patient_text, guideline_text, condition_name):
    # Check for anatomical opposites
    if is_anatomical_opposite(patient_text, guideline_text):
        # Use semantic similarity with reduced penalty
        semantic_score = compute_semantic_similarity(patient_text, guideline_text)
        return semantic_score * 0.8  # Reduced penalty for opposites
    
    # Standard hybrid scoring
    return compute_hybrid_similarity(patient_text, guideline_text)
```

### 3. **Fuzzy Jaccard Similarity**
```python
def fuzzy_jaccard_similarity(text1, text2):
    # Use word embeddings for fuzzy matching
    words1 = tokenize(text1)
    words2 = tokenize(text2)
    
    fuzzy_intersection = 0
    for w1 in words1:
        for w2 in words2:
            if compute_word_similarity(w1, w2) > 0.7:
                fuzzy_intersection += 1
                break
    
    return fuzzy_intersection / len(set(words1) | set(words2))
```

### 4. **Dynamic Penalty System**
```python
def dynamic_penalty(semantic_score, jaccard_score):
    if jaccard_score == 0:
        # Use semantic score to determine penalty
        if semantic_score > 0.3:
            return 0.8  # Light penalty
        elif semantic_score > 0.15:
            return 0.6  # Medium penalty
        else:
            return 0.4  # Heavy penalty
    else:
        return 1.0  # No penalty
```

## Conclusion

The current scoring system is overly conservative, excluding 88% of conditions due to anatomical mismatches. The system needs:

1. **Anatomical mapping** for opposite-side conditions
2. **Fuzzy word matching** instead of exact Jaccard similarity
3. **Dynamic penalties** based on semantic similarity
4. **Semantic-first approach** when Jaccard fails

These improvements would create a more nuanced and clinically realistic scoring mechanism.

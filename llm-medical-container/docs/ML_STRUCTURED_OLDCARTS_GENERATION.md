# ML-Based Structured OLDCARTS Generation

## 🎯 Goal
Automatically generate `structured_oldcarts` (includes/excludes terms) from new guideline text using LLM, eliminating manual JSON expansion.

## 📊 Current vs. Future Flow

### **Current (Manual)**
```
New Guideline Text
    ↓
Doctor reads text
    ↓
Manually extracts includes/excludes
    ↓
Writes structured_oldcarts JSON
    ↓
System uses it for scoring
```

### **Future (LLM-Assisted)**
```
New Guideline Text
    ↓
LLM extracts includes/excludes
    ↓
Auto-generates structured_oldcarts JSON
    ↓
System uses it for scoring
```

## 🔧 How It Works

### **1. Input: Classic Presentation Text**
```json
{
  "classic_presentation": "ONSET: SUDDEN onset within 6-48 hours (acute, NOT chronic). LOCATION: Pain MIGRATES from periumbilical to right lower quadrant (RLQ) over 12-24 hours..."
}
```

### **2. LLM Prompt**
```python
prompt = """Extract structured OLDCARTS from this medical guideline:

ONSET: SUDDEN onset within 6-48 hours (acute, NOT chronic).
LOCATION: Pain MIGRATES from periumbilical to right lower quadrant (RLQ)...

Return JSON with includes/excludes for each OLDCARTS element."""
```

### **3. LLM Output**
```json
{
  "onset": {
    "includes": ["sudden", "acute", "hours", "6-48 hours"],
    "excludes": ["chronic", "months", "years", "gradual"]
  },
  "location": {
    "includes": ["right lower quadrant", "RLQ", "McBurney point"],
    "excludes": ["left side", "LLQ", "upper abdomen"]
  }
}
```

### **4. System Integration**
- Generated data seamlessly fits existing architecture
- Word-match boost uses includes/excludes immediately
- No code changes needed in scoring logic

## 📋 Example Workflow

### **Adding a New Guideline**

**Step 1:** Write basic guideline
```json
{
  "condition": "Acute Pancreatitis",
  "key_features": {
    "classic_presentation": "ONSET: SUDDEN. LOCATION: Epigastric pain radiating to back..."
  }
}
```

**Step 2:** Run auto-generator
```bash
python scripts/auto_generate_structured_oldcarts.py \
  --guideline-file medical/guidelines/GI/GI_Acute_Pancreatitis.json \
  --apply
```

**Step 3:** Generated result
```json
{
  "condition": "Acute Pancreatitis",
  "key_features": {
    "classic_presentation": "...",
    "structured_oldcarts": {
      "location": {
        "includes": ["epigastric", "radiates to back"],
        "excludes": ["right side only", "isolated LLQ"]
      }
    }
  }
}
```

## 🎯 Benefits

1. **Speed**: Minutes vs. hours for manual extraction
2. **Consistency**: LLM follows same pattern every time
3. **Scaling**: Generate for hundreds of guidelines quickly
4. **Integration**: Zero code changes - uses existing architecture
5. **Quality**: LLM understands medical terminology

## 🔄 Integration with Current System

```
LLM Generation → structured_oldcarts.json
                       ↓
Word-Match Boost (includes/excludes)
                       ↓
Semantic Similarity + Boost
                       ↓
Scoring & Ranking
```

No changes needed to:
- Embeddings
- Synonyms
- Word-match logic
- Scoring algorithm

LLM just generates the **structured data** the system already expects!


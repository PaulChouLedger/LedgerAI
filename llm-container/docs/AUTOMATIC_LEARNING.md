# Automatic Learning with Manual Review

## 🎯 Overview
**System automatically learns from every patient interaction** - no user input required for data collection. User only needs to review and approve suggested updates.

## 🔄 How It Works

### **Automatic Recording (Zero User Input)**
```
Patient Interaction
    ↓
System Scores Answer
    ↓
Automatically Records Prediction
    ↓
Stored in ml/learning_data/corrections.jsonl
```

**What gets recorded:**
- Condition name
- OLDCARTS element (location, onset, etc.)
- User's answer
- Similarity score
- Guideline text

**Example record:**
```json
{
  "timestamp": "2025-10-28T12:34:56",
  "condition": "Acute Appendicitis",
  "oldcarts_element": "location",
  "user_answer": "hurts near my hip bone",
  "similarity_score": 0.28,
  "guideline_text": "Pain MIGRATES from periumbilical to right lower quadrant...",
  "context": {"method": "semantic_similarity"}
}
```

### **Pattern Detection (Automatic)**
System analyzes stored predictions to find patterns:

1. **Low-scoring answers** → Potential missing synonyms
2. **Repeated failures** → Same condition/element low scores
3. **Common patient terms** → Suggested additions to includes

### **Generate Suggestions (Manual Trigger)**
```bash
# Run analysis to generate suggestions
python ml/learning_suggestions.py --analyze
```

**Detects:**
- Low similarity scores (< 0.4) appearing frequently
- Patterns where patients use terms not in guidelines
- Suggestions for improving structured_oldcarts and synonyms

### **Review & Apply (User Input Required)**
```bash
# View suggestions
python ml/learning_suggestions.py --show

# Review and apply interactively
python ml/apply_suggestions.py --review
```

## 📊 Example Flow

### **Week 1-2: System Collects Data**
- 47 patient interactions recorded
- Low scores detected for "hurts near my hip bone" + Appendicitis

### **Week 3: Generate Suggestions**
```bash
$ python ml/learning_suggestions.py --analyze --min-occurrences 10

🔍 Analyzing learning data...
✅ Analysis complete!
   Generated 3 structured updates
   Generated 2 synonym updates
```

### **Week 3: Review Suggestions**
```bash
$ python ml/learning_suggestions.py --show

LEARNING SUGGESTIONS
================================================================================

📊 Summary:
   Total Corrections: 47
   Total Synonym Expansions: 32
   Structured Updates Suggested: 3
   Synonym Updates Suggested: 2
   Highest Confidence: 75%

📋 STRUCTURED OLDCARTS UPDATES:
--------------------------------------------------------------------------------

🎯 Acute Appendicitis (location)
   Frequency: 12 occurrences
   Avg Similarity Score: 0.28
   Detection Method: automatic
   Reason: Low similarity scores (12 occurrences)
   Suggested Adds to Includes:
      • hurts near my hip bone
      • pain by my hip
      • near right hip
   Examples:
      • hurts near my hip bone
      • pain by my hip
      • near right hip
   Confidence: 60%
```

### **Week 3: User Applies Update**
```bash
$ python scripts/apply_suggestions.py --review

Apply this change? (y/n/skip): y
   ✅ Applied!
   📝 Updated: medical/guidelines/GI/GI_Acute_Appendicitis.json
      Added 3 items to location includes
```

## 🎯 Key Benefits

1. **Zero Overhead**: No user input during normal operation
2. **Automatic Detection**: Finds patterns without manual analysis
3. **Smart Suggestions**: Only shows high-confidence recommendations
4. **Safety**: All changes require explicit approval
5. **Continuous Improvement**: System gets better over time

## 🔧 Configuration

### **Adjust Low Score Threshold**
```python
# In analyze_and_suggest()
analyze_and_suggest(
    min_occurrences=10,          # Require 10+ occurrences
    low_score_threshold=0.4      # Scores < 0.4 are "low"
)
```

### **Enable/Disable Learning**
```python
# In adaptive_diagnostic_engine.py
self.learner = LearningSuggestions()  # Enable
self.learner = None                    # Disable
```

## 📈 What Gets Learnt

### **Automatic Detection Finds:**
1. **Low-scoring patient terms** → Add to synonym files
2. **Common patient language** → Add to includes terms
3. **Patterns across conditions** → Improve guidelines
4. **Missing synonyms** → Expand synonym coverage

### **Example: Hip Pain Pattern**
```
Detected: 12 low scores for "hurts near my hip" + Appendicitis
Suggested: Add "hip bone", "near hip", "by hip" to RLQ synonyms
User Approves: Adds to gi_synonyms_oldcarts.json
Result: Future "hip" mentions get better scores
```

## 🚀 Usage

### **Daily Operation**
- System records predictions automatically
- No action required from user

### **Weekly/Monthly Review**
```bash
# Generate suggestions
python ml/learning_suggestions.py --analyze

# Review what changed
python ml/learning_suggestions.py --show

# Apply approved changes
python scripts/apply_suggestions.py --review
```

### **Monitoring**
```bash
# Check learning data size
du -sh ml/learning_data/

# View recent records
tail -n 20 ml/learning_data/corrections.jsonl | jq '.'

# Count predictions by condition
cat ml/learning_data/corrections.jsonl | jq -r '.condition' | sort | uniq -c
```

## ⚡ Performance Impact

**Recording is lightweight:**
- File append operation (< 1ms)
- No blocking on main thread
- Exception handling prevents failures

**Analysis is offline:**
- Only runs when user triggers it
- No impact on real-time operation

## ✅ Summary

**System automatically learns from every interaction** while you sleep, and presents suggestions for your approval. Zero daily overhead, continuous improvement!


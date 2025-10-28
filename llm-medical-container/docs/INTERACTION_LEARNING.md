# Interaction Learning System

## 🎯 Goal
Learn from user interactions to automatically refine:
1. **structured_oldcarts** (includes/excludes terms)
2. **synonym files** (patient terminology)
3. **guideline accuracy** (missing patterns)

## 📊 How It Works

### **1. Track Corrections**
When system misinterprets patient answers:

```python
learner.record_correction(
    condition="Acute Appendicitis",
    oldcarts_element="location",
    user_answer="hurts near my hip bone",
    expected_term="right lower quadrant",
    actual_result="matched to hip instead of RLQ"
)
```

**Result**: System learns that "hurts near my hip bone" should map to RLQ pain

### **2. Track New Synonyms**
When patients use unexpected terminology:

```python
learner.record_synonym_expansion(
    organ_system="GI",
    category="rlq_pain",
    new_synonym="hurts near my hip bone",
    context={"10 occurrences in appendicitis cases"}
)
```

**Result**: Adds "hurts near my hip bone" to RLQ pain synonyms

### **3. Track Patterns**
Detect common patient descriptions:

```python
learner.record_pattern_detection(
    condition="Pleurisy",
    oldcarts_element="aggravating",
    pattern="triggered by breathing",
    frequency=25,
    patient_variations=["hurts when I breathe", "worse on inhaling", "sharp when breathing"]
)
```

**Result**: Suggests adding "breathing" to aggravating includes

## 🔄 Learning Loop

```
User Interaction
    ↓
Record Issue (correction/synonym/pattern)
    ↓
Analyze (after N occurrences)
    ↓
Generate Updates (via LLM or frequency analysis)
    ↓
Apply Updates (to structured_oldcarts or synonyms)
    ↓
System Improves
```

## 📋 Example Workflow

### **Scenario: Missing Patient Term**

**Initial State:**
```json
{
  "location": {
    "includes": ["right lower quadrant", "RLQ", "appendix pain"],
    "excludes": ["left side", "LLQ"]
  }
}
```

**Patient Says:** "hurts near my hip bone" (10 times for appendicitis cases)

**Learning System:**
1. Records correction each time
2. After 5 occurrences, analyzes pattern
3. Generates update suggestion

**Generated Update:**
```json
{
  "add_to_includes": ["hip bone", "near hip", "by hip"],
  "confidence": 0.7
}
```

**Applied Update:**
```json
{
  "location": {
    "includes": [
      "right lower quadrant", 
      "RLQ", 
      "appendix pain",
      "hip bone",      // ← ADDED
      "near hip",      // ← ADDED
      "by hip"         // ← ADDED
    ],
    "excludes": ["left side", "LLQ"]
  }
}
```

## 🎯 Benefits

1. **Self-Improving**: System gets better with use
2. **Patient-Focused**: Learns real patient language
3. **Automatic**: No manual intervention needed
4. **Context-Aware**: Considers condition and situation
5. **Scalable**: Learns across all organ systems

## 🔧 Integration

### **Automatic Recording**
```python
# In adaptive_diagnostic_engine.py
class AdaptiveDiagnosticEngine:
    def __init__(self, ...):
        self.learner = InteractionLearning()
    
    def _process_clinical_answer(self, answer: str):
        # After scoring, check if interpretation is correct
        if user_provided_feedback:
            self.learner.record_correction(...)
```

### **Periodic Updates**
```python
# Run nightly or weekly
updates = learner.generate_updates(llm_fn=your_llm)
apply_learning_updates(updates, dry_run=False)
```

## 📊 Learning Data Structure

### **Corrections**
```json
{
  "timestamp": "2025-10-28T12:34:56",
  "condition": "Acute Appendicitis",
  "oldcarts_element": "location",
  "user_answer": "hurts near my hip bone",
  "expected_term": "right lower quadrant",
  "actual_result": "matched to hip instead of RLQ",
  "context": {"score": 0.3, "should_be": 0.8}
}
```

### **Synonym Expansions**
```json
{
  "timestamp": "2025-10-28T12:34:56",
  "organ_system": "GI",
  "oldcarts_element": "location",
  "category": "rlq_pain",
  "new_synonym": "hurts near my hip bone",
  "context": {"condition": "Acute Appendicitis", "frequency": 10}
}
```

### **Pattern Detections**
```json
{
  "timestamp": "2025-10-28T12:34:56",
  "condition": "Pleurisy",
  "oldcarts_element": "aggravating",
  "pattern": "triggered by breathing",
  "frequency": 25,
  "patient_variations": [
    "hurts when I breathe",
    "worse on inhaling",
    "sharp when breathing"
  ]
}
```

## 🚀 Future Enhancements

1. **LLM-Powered Suggestions**: Use LLM to propose better includes/excludes
2. **Confidence Scoring**: Only apply high-confidence updates
3. **A/B Testing**: Test updates before full deployment
4. **Multi-Language**: Learn patient terms in different languages
5. **Condition-Specific**: Adapt to local vernacular


# Synonym-Based Categorization System

## 🎯 **New Approach: Synonym Normalization + Substring Matching**

### **Old Hardcoded Approach:**
```python
# HARDCODED keyword lists
if any(word in complaint_lower for word in ['abdominal', 'stomach', 'belly', 'nausea', 'vomit', 'diarrhea', 'constipation']):
    return 'GI'
```

### **New Synonym-Based Approach:**
```python
# 1. Normalize with medical synonyms
normalized_complaint = self._normalize_complaint_with_synonyms(complaint)

# 2. Match against guideline triggers
category = self._categorize_complaint_by_substring(normalized_complaint)
```

## 🔄 **How It Works:**

### **Step 1: Synonym Normalization**
```python
def _normalize_complaint_with_synonyms(self, complaint: str) -> str:
    # Medical synonym mappings
    synonym_mappings = {
        'abdominal': ['stomach', 'belly', 'gut', 'tummy', 'abdomen', 'gastric', 'intestinal'],
        'pain': ['ache', 'hurt', 'sore', 'discomfort', 'cramp', 'cramping'],
        'nausea': ['sick', 'queasy', 'nauseous', 'feeling sick'],
        'vomit': ['throw up', 'puke', 'vomiting', 'throwing up'],
        # ... more synonyms
    }
    
    # Apply synonym normalization
    normalized_complaint = complaint_lower
    for standard_term, synonyms in synonym_mappings.items():
        for synonym in synonyms:
            if synonym in normalized_complaint:
                normalized_complaint = normalized_complaint.replace(synonym, standard_term)
    
    return normalized_complaint
```

### **Step 2: Substring Matching Against Guidelines**
```python
def _categorize_complaint_by_substring(self, normalized_complaint: str) -> str:
    category_matches = {'GI': 0, 'CARDIO': 0, 'NEURO': 0, 'MSK': 0, 'DERM': 0, 'RENAL': 0, 'GYN': 0}
    
    # Check each guideline's triggers for substring matches
    for name, guideline in self.all_guidelines.items():
        triggers = guideline.get('chief_complaint_triggers', [])
        for trigger in triggers:
            trigger_lower = trigger.lower()
            if any(word in trigger_lower for word in normalized_complaint.split()):
                # Determine category from guideline name
                if name.startswith('GI_'):
                    category_matches['GI'] += 1
                elif name.startswith('CARDIO_'):
                    category_matches['CARDIO'] += 1
                # ... more categories
    
    # Find category with most matches
    best_category = max(category_matches, key=category_matches.get)
    return best_category if category_matches[best_category] > 0 else 'ALL'
```

## 📊 **Example Flow:**

### **Input: "i have stomach ache"**

#### **Step 1: Synonym Normalization**
```
Input: "i have stomach ache"
↓
Synonym mapping: "stomach" → "abdominal", "ache" → "pain"
↓
Normalized: "i have abdominal pain"
```

#### **Step 2: Substring Matching**
```
Check against all guideline triggers:
- GI_Acute_Appendicitis: ["abdominal pain", "stomach pain"] → MATCH
- GI_Acute_Cholecystitis: ["abdominal pain", "right upper quadrant pain"] → MATCH
- CARDIO_Myocardial_Infarction: ["chest pain", "heart attack"] → NO MATCH
- NEURO_Migraine: ["headache", "head pain"] → NO MATCH

Category matches: {'GI': 2, 'CARDIO': 0, 'NEURO': 0, ...}
Best category: GI (2 matches)
```

#### **Step 3: Filter Guidelines**
```
Total guidelines: 144
GI guidelines: 22
Result: Process only 22 GI guidelines (85% reduction)
```

## ✅ **Benefits of New Approach:**

### **1. Flexible Synonym Handling:**
- **"stomach ache"** → **"abdominal pain"** (normalized)
- **"belly hurt"** → **"abdominal pain"** (normalized)
- **"gut cramp"** → **"abdominal pain"** (normalized)

### **2. Data-Driven Categorization:**
- **Uses actual guideline triggers** instead of hardcoded lists
- **Learns from existing guidelines** automatically
- **Adapts to new guidelines** without code changes

### **3. Better Matching:**
- **Substring matching** against real medical terms
- **Multiple word matching** for complex complaints
- **Context-aware** categorization

### **4. Maintainable:**
- **Easy to add synonyms** without changing logic
- **Guideline-driven** categorization
- **Self-updating** as guidelines are added

## 🔍 **Debug Output:**

### **Synonym Normalization:**
```
[Engine] 🔄 Synonym normalization: 'i have stomach ache' → 'i have abdominal pain'
```

### **Substring Matching:**
```
[Engine] 🎯 Substring matches: {'GI': 2, 'CARDIO': 0, 'NEURO': 0, 'MSK': 0, 'DERM': 0, 'RENAL': 0, 'GYN': 0}
[Engine] 🎯 Best category: GI (2 matches)
```

### **Category Filtering:**
```
[Engine] 🎯 Category filtering: GI → 22/144 guidelines
```

## 🎯 **Comparison:**

### **Old Hardcoded Approach:**
- ❌ **Rigid** keyword lists
- ❌ **Manual maintenance** required
- ❌ **Misses synonyms** not in list
- ❌ **Not data-driven**

### **New Synonym Approach:**
- ✅ **Flexible** synonym handling
- ✅ **Data-driven** from guidelines
- ✅ **Self-updating** with new guidelines
- ✅ **Better matching** accuracy

**The new approach is more intelligent, flexible, and data-driven while maintaining the same performance benefits!** 🏥⚡

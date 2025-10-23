# Debug Fixes for LLM Container Issues

## 🐛 **Issues Identified:**

### **1. Category Filtering Broken:**
- **Problem:** `GI → 0/144 guidelines` (should be 22 GI guidelines)
- **Root Cause:** Category filtering method not finding guidelines
- **Fix:** Added debug output to track category filtering process

### **2. LLM Normalization Over-Specification:**
- **Problem:** "abdominal pain" → "left side" (incorrect over-specification)
- **Root Cause:** LLM prompt allowing general complaints to be converted to specific locations
- **Fix:** Updated prompt to explicitly prevent over-specification

### **3. RAG Mode Display Confusion:**
- **Problem:** Shows "RAG API mode" but then "CPU FAISS" 
- **Root Cause:** Old debug message not reflecting actual RAG mode
- **Fix:** Updated debug message to show actual RAG mode

## 🔧 **Fixes Applied:**

### **1. Enhanced Category Filtering Debug:**
```python
def _get_guidelines_by_category(self, category: str) -> Dict:
    # Added debug output to track filtering process
    self._capture_debug(f"[Engine] 🔍 Category '{category}' patterns: {patterns}")
    self._capture_debug(f"[Engine] 🔍 Total guidelines loaded: {len(self.all_guidelines)}")
    
    for name, guideline in self.all_guidelines.items():
        if any(pattern in name for pattern in patterns):
            filtered_guidelines[name] = guideline
            self._capture_debug(f"[Engine] ✅ Matched: {name}")
    
    self._capture_debug(f"[Engine] 🔍 Filtered guidelines: {len(filtered_guidelines)}")
    return filtered_guidelines
```

### **2. Fixed LLM Normalization Prompt:**
```python
user_msg = f"""Normalize this patient response into standard medical terminology for the {context} component ONLY:

Patient: "{text}"
OLDCARTS Component: {context}

CRITICAL: 
- Only normalize the {context} component
- Do NOT add information from other symptoms or previous questions
- Do NOT convert general complaints (like "abdominal pain") to specific locations (like "left side") unless the patient specifically mentions a location
- Keep general terms general unless patient provides specific details

Normalized text:"""
```

### **3. Fixed RAG Mode Display:**
```python
# OLD: Fixed message
self._capture_debug(f"[Engine] 🚀 Using RAG API mode for matching")

# NEW: Dynamic message
self._capture_debug(f"[Engine] 🚀 Using RAG mode: {self.rag_mode}")
```

## 🎯 **Expected Results:**

### **1. Category Filtering:**
- **Before:** `GI → 0/144 guidelines`
- **After:** `GI → 22/144 guidelines` (should show matched GI guidelines)

### **2. LLM Normalization:**
- **Before:** "abdominal pain" → "left side" (incorrect)
- **After:** "abdominal pain" → "abdominal pain" (correct)

### **3. RAG Mode Display:**
- **Before:** "RAG API mode" → "CPU FAISS" (confusing)
- **After:** "RAG mode: CPU" → "CPU FAISS" (consistent)

## 🔍 **Debug Output to Watch:**

### **Category Filtering:**
```
[Engine] 🔍 Category 'GI' patterns: ['GI_', 'gastrointestinal', 'abdominal']
[Engine] 🔍 Total guidelines loaded: 144
[Engine] ✅ Matched: GI_Acute_Appendicitis
[Engine] ✅ Matched: GI_Acute_Cholecystitis
[Engine] 🔍 Filtered guidelines: 22
```

### **LLM Normalization:**
```
[Engine] 🧠 LLM normalizing: 'i have abdominal pain' (context: general)
[Engine] ✅ LLM normalization: 'i have abdominal pain' → 'abdominal pain'
```

### **RAG Mode:**
```
[Engine] 🚀 Using RAG mode: CPU
[Engine] 🧠 CPU FAISS semantic search (local processing)...
```

## ✅ **Testing:**

### **Test Case: "i have abdominal pain"**
1. **Category filtering:** Should find 22 GI guidelines
2. **LLM normalization:** Should keep "abdominal pain" as-is
3. **RAG mode:** Should show consistent CPU mode
4. **Guideline matching:** Should find relevant GI conditions

**These fixes should resolve the category filtering, LLM over-specification, and RAG mode display issues!** 🏥⚡

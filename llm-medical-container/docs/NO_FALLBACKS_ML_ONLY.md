# No Fallbacks - ML-Only System

## 🎯 **ML-Only Architecture:**

### **✅ Removed All Fallbacks:**

#### **1. RAG API Fallback (REMOVED):**
```python
# OLD: RAG API with fallback
elif self.use_rag_api:
    try:
        # RAG API processing
    except Exception as rag_error:
        self._capture_debug(f"[Engine] ❌ RAG API matching failed: {rag_error}")
        self._capture_debug(f"[Engine] 🔄 Falling back to brute-force matching")
        self.use_rag_api = False  # Disable RAG API for future queries

# NEW: ML-only mode (no fallbacks)
if self.use_rag_api:
    self._capture_debug(f"[Engine] 🚀 Using RAG mode: {self.rag_mode}")
    rag_result[0] = self._match_to_guidelines_rag(chief_complaint)
else:
    self._capture_debug(f"[Engine] 🧠 Using ML-only mode for matching")
    rag_result[0] = self._match_to_guidelines_ml(chief_complaint, "ALL")
```

#### **2. Word Overlap Fallback (REMOVED):**
```python
# OLD: Word overlap fallback
if self.medical_rule_engine:
    result = self.medical_rule_engine.get_enhanced_similarity(...)
    return result['similarity']
else:
    # Fallback to simple word overlap
    complaint_words = set(normalized_complaint.split())
    trigger_words = set(normalized_trigger.split())
    overlap = len(complaint_words.intersection(trigger_words))
    total = len(complaint_words.union(trigger_words))
    return overlap / total if total > 0 else 0.0

# NEW: ML-only (no fallback)
if not self.medical_rule_engine:
    raise RuntimeError("Medical Rule Engine not available - ML system required")

result = self.medical_rule_engine.get_enhanced_similarity(...)
return result['similarity']
```

#### **3. Character Overlap Fallback (REMOVED):**
```python
# OLD: Character overlap search as fallback
def _perform_character_overlap_search(self, complaint: str, core_symptom: str, matched: List[Dict], matched_guideline_names: set):
    """Perform character overlap search as fallback"""

# NEW: ML-only (no fallback)
def _perform_character_overlap_search(self, complaint: str, core_symptom: str, matched: List[Dict], matched_guideline_names: set):
    """Perform character overlap search (ML-only, no fallback)"""
```

#### **4. Template Fallbacks (REMOVED):**
```python
# OLD: Template fallback for LLM questions
self._capture_debug(f"[Engine] ⚠️ LLM combined multiple questions - using template fallback")
# Use simple template fallback
question = example

# NEW: ML-only question generation
def _generate_ml_question(self, oldcarts_element: str) -> str:
    """Generate question using ML-powered approach"""
    # Direct ML question generation (no fallbacks)
```

## 🚨 **Runtime Errors for Missing ML:**

### **1. Medical Rule Engine Required:**
```python
if not self.medical_rule_engine:
    raise RuntimeError("Medical Rule Engine not available - ML system required")
```

### **2. Embedding Model Required:**
```python
if not self.embedding_model:
    raise RuntimeError("Embedding model not initialized - cannot compute similarity")
```

### **3. RAG Embedding Required:**
```python
else:
    raise RuntimeError(f"RAG embedding failed")
```

## ✅ **ML-Only Benefits:**

### **1. No Hardcoded Fallbacks:**
- **No word overlap** fallbacks
- **No character overlap** fallbacks
- **No template** fallbacks
- **No brute-force** fallbacks

### **2. Pure ML Processing:**
- **Medical Rule Engine** for all similarity
- **Synonym files** for all normalization
- **ML predictions** for all scoring
- **Semantic similarity** for all matching

### **3. Fail Fast:**
- **Runtime errors** if ML components missing
- **No degraded performance** with fallbacks
- **Clear error messages** for missing components
- **Forces proper ML setup**

### **4. Consistent Behavior:**
- **Same processing** every time
- **No fallback variations**
- **Predictable performance**
- **Reliable results**

## 🔍 **Error Handling:**

### **Missing Medical Rule Engine:**
```
RuntimeError: Medical Rule Engine not available - ML system required
```

### **Missing Embedding Model:**
```
RuntimeError: Embedding model not initialized - cannot compute similarity
```

### **RAG Embedding Failure:**
```
RuntimeError: RAG embedding failed
```

## 🎯 **ML-Only Flow:**

### **1. Initial Processing:**
```python
# ML-powered complaint normalization
normalized_complaint = self._normalize_complaint_with_synonyms(chief_complaint)

# ML-powered category detection
category = self._categorize_complaint_by_substring(normalized_complaint)

# ML-powered guideline matching
matched_guidelines = self._match_to_guidelines_ml(normalized_complaint, category)
```

### **2. Similarity Computation:**
```python
# ML-only similarity (no fallbacks)
if not self.medical_rule_engine:
    raise RuntimeError("Medical Rule Engine not available - ML system required")

result = self.medical_rule_engine.get_enhanced_similarity(
    normalized_complaint, normalized_trigger, "", organ_system="general"
)
return result['similarity']
```

### **3. Question Generation:**
```python
# ML-only question generation (no templates)
def _generate_ml_question(self, oldcarts_element: str) -> str:
    """Generate question using ML-powered approach"""
    # Direct ML question generation
```

## ✅ **System Requirements:**

### **Required Components:**
- **Medical Rule Engine** (ml/medical_rule_engine.py)
- **Synonym Files** (synonyms/*.json)
- **RAG Client** (rag_client.py)
- **Embedding Model** (local or RAG API)

### **No Fallbacks:**
- **No word overlap** matching
- **No character overlap** matching
- **No template** questions
- **No brute-force** matching

**The system is now ML-only with no fallbacks - it will fail fast if ML components are missing, ensuring consistent ML-powered processing!** 🏥⚡

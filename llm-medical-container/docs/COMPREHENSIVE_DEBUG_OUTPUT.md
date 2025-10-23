# Comprehensive Debug Output - ML Guideline Matching

## 🔍 **Debug Output Added:**

### **1. Category Detection Debug:**
```python
def _categorize_complaint_by_substring(self, normalized_complaint: str) -> str:
    """Categorize complaint by substring matching against guideline triggers"""
    self._capture_debug(f"[Engine] 🔍 CATEGORY DETECTION DEBUG:")
    self._capture_debug(f"[Engine] 🔍 Input: '{normalized_complaint}'")
    self._capture_debug(f"[Engine] 🔍 Complaint words: {set(normalized_complaint.split())}")
    
    # Check each guideline's triggers for substring matches
    for name, guideline in self.all_guidelines.items():
        total_guidelines_checked += 1
        triggers = guideline.get('chief_complaint_triggers', [])
        self._capture_debug(f"[Engine] 🔍 Checking guideline: {name} ({len(triggers)} triggers)")
        
        for trigger in triggers:
            total_triggers_checked += 1
            trigger_lower = trigger.lower()
            # Check for word overlap between complaint and trigger
            complaint_words = set(normalized_complaint.split())
            trigger_words = set(trigger_lower.split())
            overlap = len(complaint_words.intersection(trigger_words))
            
            self._capture_debug(f"[Engine] 🔍   Trigger: '{trigger}' → '{trigger_lower}'")
            self._capture_debug(f"[Engine] 🔍   Trigger words: {trigger_words}")
            self._capture_debug(f"[Engine] 🔍   Overlap: {overlap} words")
            
            if overlap > 0:  # Any word overlap
                matches_found += 1
                # Determine category from guideline name
                if name.startswith('GI_'):
                    category_matches['GI'] += 1
                    self._capture_debug(f"[Engine] 🔍   ✓ GI match: {name} (overlap: {overlap})")
                # ... etc for other categories
            else:
                self._capture_debug(f"[Engine] 🔍   ✗ No overlap: {name}")
    
    self._capture_debug(f"[Engine] 🔍 CATEGORY DETECTION SUMMARY:")
    self._capture_debug(f"[Engine] 🔍 Total guidelines checked: {total_guidelines_checked}")
    self._capture_debug(f"[Engine] 🔍 Total triggers checked: {total_triggers_checked}")
    self._capture_debug(f"[Engine] 🔍 Total matches found: {matches_found}")
    self._capture_debug(f"[Engine] 🔍 Category matches: {category_matches}")
```

### **2. ML Guideline Matching Debug:**
```python
def _match_to_guidelines_ml(self, normalized_complaint: str, category: str) -> List[Dict]:
    """ML-powered guideline matching using comprehensive synonym files"""
    self._capture_debug(f"[Engine] 🧠 ML-POWERED GUIDELINE MATCHING DEBUG:")
    self._capture_debug(f"[Engine] 🧠 Input: '{normalized_complaint}'")
    self._capture_debug(f"[Engine] 🧠 Category: {category}")
    
    # Get relevant guidelines by category
    relevant_guidelines = self._get_guidelines_by_category(category)
    self._capture_debug(f"[Engine] 🧠 Relevant guidelines: {len(relevant_guidelines)}")
    
    # ML-powered matching using synonym files
    for name, guideline in relevant_guidelines.items():
        total_guidelines_checked += 1
        triggers = guideline.get('chief_complaint_triggers', [])
        self._capture_debug(f"[Engine] 🧠 Checking guideline: {name} ({len(triggers)} triggers)")
        
        # Check each trigger for ML similarity
        best_similarity = 0.0
        best_trigger = ""
        
        for trigger in triggers:
            total_triggers_checked += 1
            self._capture_debug(f"[Engine] 🧠   Trigger: '{trigger}'")
            
            # Use ML similarity with synonym normalization
            similarity = self._compute_ml_trigger_similarity(normalized_complaint, trigger)
            total_similarities_computed += 1
            
            self._capture_debug(f"[Engine] 🧠   ML similarity: {similarity:.3f}")
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_trigger = trigger
                self._capture_debug(f"[Engine] 🧠   New best similarity: {similarity:.3f}")
        
        self._capture_debug(f"[Engine] 🧠   Best similarity: {best_similarity:.3f} (trigger: '{best_trigger}')")
        
        # Add if similarity meets threshold
        if best_similarity > 0.3:  # ML threshold
            matched_guidelines.append({...})
            self._capture_debug(f"[Engine] 🧠   ✓ MATCHED: {name} (ML similarity: {best_similarity:.3f}, trigger: '{best_trigger}')")
        else:
            self._capture_debug(f"[Engine] 🧠   ✗ REJECTED: {name} (similarity: {best_similarity:.3f} < 0.3)")
    
    self._capture_debug(f"[Engine] 🧠 ML MATCHING SUMMARY:")
    self._capture_debug(f"[Engine] 🧠 Total guidelines checked: {total_guidelines_checked}")
    self._capture_debug(f"[Engine] 🧠 Total triggers checked: {total_triggers_checked}")
    self._capture_debug(f"[Engine] 🧠 Total similarities computed: {total_similarities_computed}")
    self._capture_debug(f"[Engine] 🧠 Guidelines matched: {len(matched_guidelines)}")
```

### **3. ML Similarity Computation Debug:**
```python
def _compute_ml_trigger_similarity(self, complaint: str, trigger: str) -> float:
    """Compute ML similarity between complaint and trigger using synonym files"""
    self._capture_debug(f"[Engine] 🧠 ML SIMILARITY COMPUTATION DEBUG:")
    self._capture_debug(f"[Engine] 🧠   Complaint: '{complaint}'")
    self._capture_debug(f"[Engine] 🧠   Trigger: '{trigger}'")
    
    # Normalize both complaint and trigger using synonym files
    normalized_complaint = self._normalize_complaint_with_synonyms(complaint)
    normalized_trigger = self._normalize_complaint_with_synonyms(trigger)
    
    self._capture_debug(f"[Engine] 🧠   Normalized complaint: '{normalized_complaint}'")
    self._capture_debug(f"[Engine] 🧠   Normalized trigger: '{normalized_trigger}'")
    
    # Use Medical Rule Engine for similarity
    if not self.medical_rule_engine:
        raise RuntimeError("Medical Rule Engine not available - ML system required")
    
    self._capture_debug(f"[Engine] 🧠   Computing ML similarity...")
    result = self.medical_rule_engine.get_enhanced_similarity(
        normalized_complaint, normalized_trigger, "", organ_system="general"
    )
    
    similarity = result['similarity']
    self._capture_debug(f"[Engine] 🧠   ML similarity result: {similarity:.3f}")
    return similarity
```

## 📊 **Expected Debug Output:**

### **Category Detection:**
```
[Engine] 🔍 CATEGORY DETECTION DEBUG:
[Engine] 🔍 Input: 'i have abdominal pain'
[Engine] 🔍 Complaint words: {'i', 'have', 'abdominal', 'pain'}
[Engine] 🔍 Checking guideline: GI_Acute_Appendicitis (2 triggers)
[Engine] 🔍   Trigger: 'abdominal pain' → 'abdominal pain'
[Engine] 🔍   Trigger words: {'abdominal', 'pain'}
[Engine] 🔍   Overlap: 2 words
[Engine] 🔍   ✓ GI match: GI_Acute_Appendicitis (overlap: 2)
[Engine] 🔍   Trigger: 'lower right side pain' → 'lower right side pain'
[Engine] 🔍   Trigger words: {'lower', 'right', 'side', 'pain'}
[Engine] 🔍   Overlap: 1 words
[Engine] 🔍   ✓ GI match: GI_Acute_Appendicitis (overlap: 1)
[Engine] 🔍 Checking guideline: GI_Acute_Cholecystitis (2 triggers)
[Engine] 🔍   Trigger: 'abdominal pain' → 'abdominal pain'
[Engine] 🔍   Trigger words: {'abdominal', 'pain'}
[Engine] 🔍   Overlap: 2 words
[Engine] 🔍   ✓ GI match: GI_Acute_Cholecystitis (overlap: 2)
[Engine] 🔍   Trigger: 'right upper quadrant pain' → 'right upper quadrant pain'
[Engine] 🔍   Trigger words: {'right', 'upper', 'quadrant', 'pain'}
[Engine] 🔍   Overlap: 1 words
[Engine] 🔍   ✓ GI match: GI_Acute_Cholecystitis (overlap: 1)
[Engine] 🔍 CATEGORY DETECTION SUMMARY:
[Engine] 🔍 Total guidelines checked: 144
[Engine] 🔍 Total triggers checked: 500+
[Engine] 🔍 Total matches found: 44
[Engine] 🔍 Category matches: {'GI': 44, 'CARDIO': 0, 'NEURO': 0, 'MSK': 0, 'DERM': 0, 'RENAL': 0, 'GYN': 0}
```

### **ML Guideline Matching:**
```
[Engine] 🧠 ML-POWERED GUIDELINE MATCHING DEBUG:
[Engine] 🧠 Input: 'i have abdominal pain'
[Engine] 🧠 Category: GI
[Engine] 🧠 Relevant guidelines: 22
[Engine] 🧠 Checking guideline: GI_Acute_Appendicitis (2 triggers)
[Engine] 🧠   Trigger: 'abdominal pain'
[Engine] 🧠 ML SIMILARITY COMPUTATION DEBUG:
[Engine] 🧠   Complaint: 'i have abdominal pain'
[Engine] 🧠   Trigger: 'abdominal pain'
[Engine] 🧠   Normalized complaint: 'i have abdominal pain'
[Engine] 🧠   Normalized trigger: 'abdominal pain'
[Engine] 🧠   Computing ML similarity...
[Engine] 🧠   ML similarity result: 0.850
[Engine] 🧠   ML similarity: 0.850
[Engine] 🧠   New best similarity: 0.850
[Engine] 🧠   Trigger: 'lower right side pain'
[Engine] 🧠 ML SIMILARITY COMPUTATION DEBUG:
[Engine] 🧠   Complaint: 'i have abdominal pain'
[Engine] 🧠   Trigger: 'lower right side pain'
[Engine] 🧠   Normalized complaint: 'i have abdominal pain'
[Engine] 🧠   Normalized trigger: 'lower right side pain'
[Engine] 🧠   Computing ML similarity...
[Engine] 🧠   ML similarity result: 0.650
[Engine] 🧠   ML similarity: 0.650
[Engine] 🧠   Best similarity: 0.850 (trigger: 'abdominal pain')
[Engine] 🧠   ✓ MATCHED: GI_Acute_Appendicitis (ML similarity: 0.850, trigger: 'abdominal pain')
[Engine] 🧠 Checking guideline: GI_Acute_Cholecystitis (2 triggers)
[Engine] 🧠   Trigger: 'abdominal pain'
[Engine] 🧠 ML SIMILARITY COMPUTATION DEBUG:
[Engine] 🧠   Complaint: 'i have abdominal pain'
[Engine] 🧠   Trigger: 'abdominal pain'
[Engine] 🧠   Normalized complaint: 'i have abdominal pain'
[Engine] 🧠   Normalized trigger: 'abdominal pain'
[Engine] 🧠   Computing ML similarity...
[Engine] 🧠   ML similarity result: 0.820
[Engine] 🧠   ML similarity: 0.820
[Engine] 🧠   New best similarity: 0.820
[Engine] 🧠   Trigger: 'right upper quadrant pain'
[Engine] 🧠 ML SIMILARITY COMPUTATION DEBUG:
[Engine] 🧠   Complaint: 'i have abdominal pain'
[Engine] 🧠   Trigger: 'right upper quadrant pain'
[Engine] 🧠   Normalized complaint: 'i have abdominal pain'
[Engine] 🧠   Normalized trigger: 'right upper quadrant pain'
[Engine] 🧠   Computing ML similarity...
[Engine] 🧠   ML similarity result: 0.580
[Engine] 🧠   ML similarity: 0.580
[Engine] 🧠   Best similarity: 0.820 (trigger: 'abdominal pain')
[Engine] 🧠   ✓ MATCHED: GI_Acute_Cholecystitis (ML similarity: 0.820, trigger: 'abdominal pain')
[Engine] 🧠 ML MATCHING SUMMARY:
[Engine] 🧠 Total guidelines checked: 22
[Engine] 🧠 Total triggers checked: 44
[Engine] 🧠 Total similarities computed: 44
[Engine] 🧠 Guidelines matched: 5
[Engine] 🧠 ML matching complete: 5 guidelines matched
```

## ✅ **Debug Benefits:**

### **1. Complete Visibility:**
- **Every step** of the matching process
- **Every similarity** computation
- **Every decision** made by the system

### **2. Performance Analysis:**
- **Total guidelines** checked
- **Total triggers** checked
- **Total similarities** computed
- **Processing time** for each step

### **3. Troubleshooting:**
- **Identify bottlenecks** in the process
- **Debug similarity** computation issues
- **Track category** detection problems
- **Monitor ML** performance

### **4. System Optimization:**
- **Identify redundant** computations
- **Optimize similarity** thresholds
- **Improve category** detection
- **Enhance ML** matching

**The comprehensive debug output provides complete visibility into every step of the ML guideline matching process!** 🏥⚡

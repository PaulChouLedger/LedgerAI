# Top 5 Active Guidelines - ML System Update

## 🎯 **Updated ML Configuration:**

### **Active Guidelines: Top 5 (vs Top 3)**
```python
# Use ML-matched guidelines
self.active_guidelines = matched_guidelines[:5]  # Top 5
self.reserve_pool = matched_guidelines[5:]  # Rest

self._capture_debug(f"[Engine] 🎯 ML-powered guidelines: Active={len(self.active_guidelines)}, Reserve={len(self.reserve_pool)}")
```

## 📊 **Benefits of Top 5 Active Guidelines:**

### **1. Better Coverage:**
- **More conditions** considered in active questioning
- **Reduced risk** of missing important differentials
- **Better diagnostic** accuracy

### **2. Improved Question Quality:**
- **More comprehensive** OLDCARTS coverage
- **Better discrimination** between similar conditions
- **More targeted** follow-up questions

### **3. Enhanced ML Learning:**
- **More data points** for ML training
- **Better pattern recognition** across conditions
- **Improved similarity** scoring

## 🔍 **Example: "I have abdominal pain"**

### **ML Similarity Results:**
```
1. GI_Acute_Appendicitis (ML similarity: 0.95, prevalence: common, score: 0.60)
2. GI_Acute_Cholecystitis (ML similarity: 0.92, prevalence: common, score: 0.60)
3. GI_Acute_Pancreatitis (ML similarity: 0.88, prevalence: uncommon, score: 0.50)
4. GI_Acute_Diverticulitis (ML similarity: 0.85, prevalence: uncommon, score: 0.50)
5. GI_Acute_Hepatitis (ML similarity: 0.82, prevalence: uncommon, score: 0.50)
```

### **Active Guidelines: 5**
- **Appendicitis** - Right lower quadrant focus
- **Cholecystitis** - Right upper quadrant focus
- **Pancreatitis** - Epigastric focus
- **Diverticulitis** - Left lower quadrant focus
- **Hepatitis** - Right upper quadrant focus

### **Reserve Pool: 5**
- **Cholangitis** - Biliary focus
- **Colitis** - Inflammatory focus
- **Gastritis** - Gastric focus
- **Enteritis** - Small bowel focus
- **Colitis** - Colonic focus

## 🎯 **OLDCARTS Questioning Strategy:**

### **Location (L) Questions:**
```
"Where exactly is the pain located?"
- Appendicitis: Right lower quadrant
- Cholecystitis: Right upper quadrant
- Pancreatitis: Epigastric
- Diverticulitis: Left lower quadrant
- Hepatitis: Right upper quadrant
```

### **Character (C) Questions:**
```
"How would you describe the pain?"
- Appendicitis: Sharp, stabbing
- Cholecystitis: Colicky, cramping
- Pancreatitis: Severe, burning
- Diverticulitis: Cramping, pressure
- Hepatitis: Dull, aching
```

### **Onset (O) Questions:**
```
"When did the pain start?"
- Appendicitis: Gradual onset
- Cholecystitis: Sudden onset
- Pancreatitis: Sudden onset
- Diverticulitis: Gradual onset
- Hepatitis: Gradual onset
```

## ✅ **ML System Benefits:**

### **1. Comprehensive Coverage:**
- **5 active conditions** for focused questioning
- **5 reserve conditions** for backup
- **Better diagnostic** accuracy

### **2. Intelligent Ranking:**
- **ML similarity** as primary ranking
- **Prevalence scores** as secondary ranking
- **Dynamic threshold** (0.7) for inclusion

### **3. Flexible Questioning:**
- **OLDCARTS elements** tailored to active conditions
- **Context-aware** question generation
- **Progressive** narrowing down

### **4. ML Learning:**
- **More training data** from 5 active conditions
- **Better pattern recognition** across conditions
- **Improved similarity** scoring over time

## 🔍 **Debug Output:**

```
[Engine] 🚀 NEW ASSESSMENT (ML-POWERED)
[Engine] 🧠 ML normalization: 'i have abdominal pain' → 'i have abdominal pain'
[Engine] 🎯 ML category: GI
[Engine] 🧠 ML-powered guideline matching for: 'i have abdominal pain'
[Engine]   ✓ GI_Acute_Appendicitis (ML similarity: 0.950, trigger: 'abdominal pain')
[Engine]   ✓ GI_Acute_Cholecystitis (ML similarity: 0.920, trigger: 'abdominal pain')
[Engine]   ✓ GI_Acute_Pancreatitis (ML similarity: 0.880, trigger: 'abdominal pain')
[Engine]   ✓ GI_Acute_Diverticulitis (ML similarity: 0.850, trigger: 'abdominal pain')
[Engine]   ✓ GI_Acute_Hepatitis (ML similarity: 0.820, trigger: 'abdominal pain')
[Engine] 📊 ML matching complete: 10 guidelines matched
[Engine] 🎯 ML-powered guidelines: Active=5, Reserve=5
[Engine] 🧠 Generating ML-powered first question...
[Engine] ✅ ML question generated: 'Where exactly is the pain located?' (element: L)
```

## 🎯 **System Configuration:**

### **Active Guidelines: 5**
- **Top 5** ML similarity scores
- **Focused questioning** on most relevant conditions
- **Better diagnostic** accuracy

### **Reserve Pool: 5**
- **Backup conditions** for additional questioning
- **Fallback options** if active conditions ruled out
- **Comprehensive coverage** of differentials

### **ML Threshold: 0.7**
- **Quality control** for guideline inclusion
- **Prevents low-quality** matches
- **Ensures relevant** conditions only

**The top 5 active guidelines provide better coverage, improved question quality, and enhanced ML learning for more accurate medical diagnosis!** 🏥⚡

# Dynamic Medical Assessment System - Setup Guide

## Overview

This system replaces rigid JSON-based triage with an intelligent, RAG-powered medical assessment that:
- ✅ **Scrapes** real medical guidelines from authoritative sources (CDC, NIH, WHO)
- ✅ **Uses RAG** to retrieve relevant guidelines based on patient symptoms
- ✅ **Asks dynamic questions** generated from guidelines (not hardcoded)
- ✅ **Generates diagnosis** with confidence scores and urgency ratings
- ✅ **Provides disposition** (call 911, ER, urgent care, routine appointment)

---

## Installation & Setup

### Step 1: Install Scraper Dependencies

```bash
cd ~/LedgerAI/medical
pip3 install -r requirements.txt
```

### Step 2: Scrape Medical Guidelines

```bash
# Scrape common medical conditions from CDC, NIH, MedlinePlus
python3 medical/guideline_scraper.py
```

This will:
- Scrape ~12 common conditions (chest pain, pancreatitis, diabetes, asthma, etc.)
- Save JSON backups to `data/input/medical_guidelines/` (for reference)
- Save .txt files DIRECTLY to `data/input/` (for RAG ingestion)

Expected output:
```
[MedlinePlus] ✅ Scraped Chest Pain: 15,243 chars
[MedlinePlus] ✅ Scraped Pancreatitis: 12,567 chars
...
[Export] ✅ Exported 12 guidelines directly to data/input/
✅ SCRAPING COMPLETE!
  Success: 12
```

### Step 3: Build Embeddings and Index

```bash
# Trigger RAG to process guidelines and build embeddings
python3 medical/ingest_guidelines.py
```

This will:
- Extract text from all .txt files in `data/input/`
- Create chunks (1000 chars each, 200 char overlap)
- Generate embeddings using sentence transformers
- Build FAISS index
- Reload RAG with new index

Expected output:
```
[Ingest] 📂 Found 12 guideline files
[Ingest] 🔄 Step 1: Extracting text...
[Ingest] ✅ Text extraction complete: Processed 12 files
[Ingest] 🔄 Step 2: Building embeddings...
📦 Created 450 text chunks
✅ FAISS index created with 450 vectors
[Ingest] ✅ RAG reloaded: 450 total chunks
```

### Step 4: Rebuild LLM Container

```bash
# Rebuild to include shared medical_terms.json
cd ~/LedgerAI
docker-compose build llm-container
docker-compose up -d
```

This ensures the LLM container has access to the centralized medical terms.

### Step 5: Enable Dynamic Assessment

In `llm-container/unified_medical_mode.py`, ensure:
```python
self.use_dynamic_assessment = True  # Line 102
```

This is already set to `True` by default.

---

## How It Works

### 1. **Patient Reports Symptom**
```
User: "I have chest pain"
```

### 2. **System Retrieves Guidelines**
```
[Dynamic] 🏥 Starting dynamic guideline-based assessment
[Dynamic]    Category: cardiovascular
[Dynamic] 📚 Retrieved 5 guideline chunks from RAG
```

RAG searches for:
- "chest pain medical guidelines"
- CDC/NIH chest pain protocols
- Emergency indicators for chest pain

### 3. **LLM Generates Contextual Question**
```
Based on AHA chest pain guidelines:
"Is the pain crushing or squeezing in nature?"
```

Questions are:
- ✅ **Generated from real guidelines** (not hardcoded)
- ✅ **Contextual** to previous answers
- ✅ **Prioritize red flags** (life-threatening symptoms first)

### 4. **Patient Responds**
```
User: "Yes, it's crushing and radiating to my left arm"
```

### 5. **System Analyzes Response**
```
[Dynamic] 📊 Updated state:
   Red flags: crushing, radiating
   Urgency: 7.5/10
```

System automatically:
- Extracts symptoms
- Detects red flags
- Updates urgency score

### 6. **More Questions (If Needed)**
```
"Are you experiencing shortness of breath or sweating?"
```

Continues until:
- ⚠️ Emergency detected (urgency ≥ 8)
- ✅ Sufficient information gathered (5-8 questions)
- 🚨 Multiple red flags (≥ 3)

### 7. **Final Diagnosis & Disposition**
```
ASSESSMENT:
  Primary Diagnosis: Acute Myocardial Infarction (MI)
  Urgency: 9/10
  
🚨 EMERGENCY - Call 911 immediately
  
Rationale: Crushing chest pain radiating to left arm with 
diaphoresis suggests acute MI. Requires immediate emergency evaluation.
```

---

## Comparison: Rigid Triage vs Dynamic Assessment

| Feature | Rigid Triage (Old) | Dynamic Assessment (New) |
|---------|-------------------|-------------------------|
| **Questions** | Hardcoded in JSON | Generated from guidelines |
| **Flexibility** | Fixed decision tree | Adapts to responses |
| **Coverage** | ~8 conditions | Unlimited (any with guidelines) |
| **Updates** | Manual JSON edits | Scrape new guidelines |
| **Intelligence** | Pre-programmed paths | LLM analyzes guidelines |
| **Accuracy** | Limited by programmer | Based on CDC/NIH evidence |
| **Maintenance** | High (manual updates) | Low (auto-scrape) |

---

## Usage Examples

### Example 1: Chest Pain (Emergency)

```
User: "I have chest pain"
Aura: "Is the pain crushing or squeezing in nature?"
User: "Yes, crushing"
Aura: "Is it radiating to your left arm or jaw?"
User: "Yes, my left arm"
Aura: "Are you experiencing shortness of breath or sweating?"
User: "Yes, both"

🚨 EMERGENCY DETECTED - Urgency: 9/10

ASSESSMENT:
  Primary Diagnosis: Acute Myocardial Infarction
  Disposition: Call 911 immediately
  Rationale: Classic MI presentation with crushing chest pain,
  radiation, and diaphoresis. Requires immediate emergency care.
```

### Example 2: Abdominal Pain (Urgent)

```
User: "I have stomach pain"
Aura: "Where exactly is the pain located?"
User: "Upper right side, below my ribs"
Aura: "When did the pain start?"
User: "About 6 hours ago, after eating"
Aura: "Is the pain constant or does it come and go?"
User: "Constant and getting worse"
Aura: "Have you had any nausea, vomiting, or fever?"
User: "Yes, I vomited twice and feel feverish"

ASSESSMENT:
  Primary Diagnosis: Acute Cholecystitis (70% confidence)
  Alternative: Pancreatitis (20%), Gastritis (10%)
  Urgency: 7/10
  
⚠️  URGENT - Visit emergency room within 1-2 hours
  
Rationale: Right upper quadrant pain after meals with fever
and vomiting suggests acute cholecystitis. Requires urgent
evaluation and possible imaging.
```

---

## Configuration

### Adjust Urgency Thresholds

In `unified_medical_mode.py`:
```python
# Urgency scoring (0-10 scale)
URGENCY_EMERGENCY = 8.0   # Call 911
URGENCY_URGENT = 6.0      # ER within 1-2 hours  
URGENCY_SEMI_URGENT = 4.0 # See doctor within 24 hours
# Below 4.0 = routine (schedule appointment)
```

### Toggle Assessment Mode

```python
# In UnifiedMedicalSession.__init__():
self.use_dynamic_assessment = True   # Dynamic RAG-powered (NEW)
self.use_dynamic_assessment = False  # Rigid triage (OLD fallback)
```

### Maximum Questions

```python
# In _should_complete_assessment():
if len(state.questions_asked) >= 8:  # Adjust this number
    return True
```

---

## Adding More Conditions

### Option 1: Scrape More Guidelines

Edit `guideline_scraper.py` to add more URLs:

```python
conditions = [
    ("MedlinePlus", "Your Condition", "https://medlineplus.gov/yourcondition.html"),
    # ... add more
]
```

### Option 2: Manual Addition

1. Create a text file in `data/input/medical_guidelines/rag_ready/`
2. Format:
   ```
   MEDICAL GUIDELINE: [Condition Name]
   Source: [CDC/NIH/WHO]
   Category: [cardiovascular/respiratory/etc.]
   
   COMMON SYMPTOMS:
     - symptom 1
     - symptom 2
   
   EMERGENCY WARNING SIGNS:
     ⚠️  red flag 1
     ⚠️  red flag 2
   
   FULL GUIDELINE CONTENT:
   [Full text from official source]
   ```

3. Copy to `data/input/`:
   ```bash
   cp data/input/medical_guidelines/rag_ready/*.txt data/input/
   ```

4. Trigger RAG ingest:
   ```bash
   python3 medical/ingest_guidelines.py
   ```

---

## Maintenance

### Daily Auto-Scrape (Optional)

Create a cron job:
```bash
# Add to crontab
0 2 * * * cd /home/aura/LedgerAI && python3 medical/guideline_scraper.py && python3 medical/ingest_guidelines.py
```

This will:
- Scrape guidelines daily at 2 AM
- Auto-ingest into RAG
- Keep system up-to-date with latest medical information

### Manual Update

```bash
# Re-scrape all guidelines
python3 medical/guideline_scraper.py

# Re-ingest into RAG
python3 medical/ingest_guidelines.py

# Restart containers
docker-compose restart
```

---

## Troubleshooting

### "No guidelines found in RAG"

**Problem:** System can't find medical guidelines

**Solution:**
1. Check files exist: `ls data/input/medical_guidelines/rag_ready/`
2. Re-run ingestion: `python3 medical/ingest_guidelines.py`
3. Verify RAG index: Check RAG container logs for "Total chunks: XXX"

### "LLM generating generic questions"

**Problem:** Questions aren't specific to condition

**Solution:**
1. Check guidelines were scraped: `ls data/input/*.txt | grep -i pancreatitis`
2. Verify RAG retrieval: Check logs for "[Dynamic] 📚 Retrieved X guideline chunks"
3. Add more specific guidelines for the condition

### "Assessment completes too quickly"

**Problem:** Only asks 1-2 questions

**Solution:**
1. Increase max questions in `_should_complete_assessment()`:
   ```python
   if len(state.questions_asked) >= 12:  # Increased from 8
   ```

2. Lower urgency thresholds:
   ```python
   if state.urgency_score >= 9.0:  # Increased from 8.0
   ```

---

## Next Steps

1. **Test with common scenarios:**
   - "I have chest pain"
   - "I have abdominal pain"
   - "I have a headache"
   - "What is pancreatitis?"

2. **Monitor RAG retrieval:**
   - Check logs for guideline chunks retrieved
   - Verify questions are guideline-based

3. **Expand guideline coverage:**
   - Add more conditions to scraper
   - Include specialist guidelines (cardiology, neurology, etc.)

4. **Fine-tune urgency scoring:**
   - Adjust red flag keywords
   - Tune urgency thresholds based on real use

---

## Architecture Summary

```
User Reports Symptom
        ↓
Categorize (cardiovascular, GI, neuro, etc.)
        ↓
RAG: Retrieve Medical Guidelines (CDC, NIH, WHO)
        ↓
LLM: Generate Question from Guidelines
        ↓
User Responds
        ↓
Analyze for Red Flags & Urgency
        ↓
Repeat Until Sufficient Information
        ↓
Generate Diagnosis + Urgency + Disposition
        ↓
Return Clinical Assessment
```

**Key Advantage:** System improves automatically as more guidelines are added to RAG!

---

## Files Created

- `/medical/guideline_scraper.py` - Web scraper for medical guidelines
- `/medical/ingest_guidelines.py` - RAG ingestion pipeline
- `/medical/dynamic_medical_assistant.py` - Core dynamic assessment engine
- `/medical/requirements.txt` - Python dependencies
- `/llm-container/unified_medical_mode.py` - Updated with dynamic assessment
- `MEDICAL_GUIDELINE_SYSTEM.md` - Architecture documentation
- `DYNAMIC_MEDICAL_SETUP.md` - This setup guide

---

✅ **System Ready!** Run the scraper and start using dynamic, guideline-based medical assessment!


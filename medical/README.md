# Medical System Components

This directory contains all medical-related functionality for the LedgerAI/Aura medical assistant.

## 📁 Directory Structure

```
medical/
├── README.md                           ← This file
│
├── SETUP & ARCHITECTURE DOCS
├── DYNAMIC_MEDICAL_SETUP.md            ← Setup guide for RAG-powered assessment
├── MEDICAL_GUIDELINE_SYSTEM.md         ← Architecture documentation
├── MEDICAL_DATA_SYSTEM_GUIDE.md        ← Data ingestion guide
├── MEDICAL_VOCABULARY_SYSTEM.md        ← Vocabulary/terminology guide
├── ENHANCED_CLINICIAN_INTEGRATION.md   ← Legacy clinician docs
│
├── GUIDELINE SCRAPING & INGESTION
├── guideline_scraper.py                ← Scrapes CDC, NIH, WHO guidelines
├── ingest_guidelines.py                ← Ingests guidelines into RAG
├── medical_data_ingestion.py           ← General medical data ingestion
├── medical_update_scheduler.py         ← Auto-update scheduler
│
├── DYNAMIC ASSESSMENT ENGINE
├── dynamic_medical_assistant.py        ← RAG-powered dynamic assessment
├── clinician_rag.py                    ← Legacy RAG integration (consider moving to llm-container)
│
└── DEPENDENCIES
    ├── requirements.txt                ← Python dependencies for scrapers
    └── requirements_medical.txt        ← Legacy requirements file
```

---

## 🚀 Quick Start - Dynamic Medical Assessment

### 1. Scrape Medical Guidelines
```bash
python3 medical/guideline_scraper.py
```
Scrapes common conditions from CDC, NIH, MedlinePlus (~12 conditions)

### 2. Ingest into RAG
```bash
python3 medical/ingest_guidelines.py
```
Adds ~500-1000 medical guideline chunks to RAG

### 3. Rebuild & Test
```bash
docker-compose build llm-container
docker-compose up -d
```

### 4. Try It
Say: **"I have chest pain"**  
Watch it generate intelligent questions from real medical guidelines!

---

## 📚 System Modes

### Dynamic Assessment (NEW ⭐)
- **Location:** `llm-container/unified_medical_mode.py`
- **Enabled by default:** `use_dynamic_assessment = True`
- **How it works:**
  1. User reports symptom → RAG retrieves guidelines
  2. LLM generates contextual questions from guidelines
  3. Tracks urgency, red flags, symptoms
  4. Generates diagnosis + disposition

### Rigid Triage (OLD - Fallback)
- **Location:** `llm-container/triage.py`
- **JSON-based:** Hardcoded questions in `triage_defs/*.json`
- **Use case:** Fallback if dynamic assessment fails

---

## 🔧 Maintenance

### Add New Medical Conditions

**Option 1: Automatic Scraping**
1. Edit `guideline_scraper.py` → add URL to `conditions` list
2. Run scraper
3. Run ingestion pipeline

**Option 2: Manual Addition**
1. Add .txt file to `data/input/medical_guidelines/rag_ready/`
2. Run `python3 medical/ingest_guidelines.py`

### Update Existing Guidelines

```bash
# Re-scrape (gets latest versions)
python3 medical/guideline_scraper.py

# Re-ingest
python3 medical/ingest_guidelines.py
```

### Schedule Auto-Updates

Use `medical_update_scheduler.py` to automatically scrape and ingest daily

---

## 🎯 Key Advantages

| Feature | Rigid Triage | Dynamic Assessment |
|---------|-------------|-------------------|
| Questions | Hardcoded | Generated from real guidelines |
| Flexibility | Fixed tree | Adapts to responses |
| Coverage | ~8 conditions | Unlimited |
| Updates | Manual edits | Auto-scrape from web |
| Evidence | Programmer knowledge | CDC/NIH/WHO guidelines |
| Maintenance | High | Low |

---

## 🔗 Integration Points

### With LLM Container
- `unified_medical_mode.py` imports dynamic assessment
- Uses shared `medical_terms.json` for keyword detection
- Routes symptom queries → dynamic assessment

### With RAG Container
- Guidelines stored in `data/input/` (auto-ingested)
- Retrieved during assessment for question generation
- Used for final diagnosis generation

### With Whisper Container
- Shares `medical_terms.json` for transcription accuracy
- Helps Whisper recognize medical terminology

---

## 📝 TODO / Future Enhancements

- [ ] Add more guideline sources (AHA, ADA, specialty societies)
- [ ] Implement confidence scoring for diagnosis
- [ ] Add multi-language support for international guidelines
- [ ] Build guideline version tracking (detect when CDC updates)
- [ ] Create guideline quality scoring (peer-reviewed vs general)
- [ ] Add guideline citation in responses ("According to CDC...")
- [ ] Implement guideline conflict resolution (when sources disagree)

---

## 🐛 Known Issues

1. **clinician_rag.py location** - Should be in `llm-container/` not `medical/`
2. **Multiple requirements files** - Consolidate `requirements.txt` and `requirements_medical.txt`

---

For detailed setup instructions, see **DYNAMIC_MEDICAL_SETUP.md**
For architecture details, see **MEDICAL_GUIDELINE_SYSTEM.md**


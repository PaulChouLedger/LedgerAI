# Dynamic Medical Guideline System - Architecture Design

## Overview
Replace rigid triage mode with a dynamic, RAG-powered medical assessment system that uses real, up-to-date clinical guidelines to ask intelligent questions and arrive at diagnosis, urgency, and disposition recommendations.

## Core Components

### 1. Guideline Scraping & Ingestion Pipeline

**Authoritative Free Sources:**
- **PubMed/NCBI** - Research articles, clinical guidelines
- **CDC** - Disease-specific guidelines, public health recommendations  
- **WHO** - Global health guidelines, pandemic protocols
- **NHS UK** - Evidence-based clinical pathways
- **Mayo Clinic** - Patient education, symptom checkers (free content)
- **American Heart Association** - Cardiovascular guidelines
- **American Diabetes Association** - Diabetes management
- **NIH/MedlinePlus** - Patient-friendly medical information

**Data Structure:**
```json
{
  "guideline_id": "cdc_chest_pain_2024",
  "source": "CDC",
  "title": "Chest Pain Evaluation Guidelines",
  "url": "https://...",
  "last_updated": "2024-10-15",
  "category": "cardiovascular",
  "urgency_indicators": ["crushing", "radiating", "shortness of breath"],
  "red_flags": ["unstable vitals", "diaphoresis", "syncope"],
  "questions_to_ask": [
    "Is the pain crushing or squeezing?",
    "Does it radiate to your arm or jaw?",
    "Do you have shortness of breath?",
    "Are you sweating profusely?"
  ],
  "differential_diagnosis": ["MI", "Unstable angina", "PE", "Aortic dissection"],
  "disposition": {
    "emergency": ["STEMI criteria", "unstable vitals"],
    "urgent": ["troponin positive", "abnormal EKG"],
    "routine": ["stable vitals", "musculoskeletal pain"]
  },
  "content": "Full guideline text..."
}
```

### 2. RAG Integration

**Storage:**
- **Location:** `data/input/medical_guidelines/` (auto-ingested into main RAG)
- **Format:** Structured JSON → converted to text chunks with metadata
- **Embeddings:** Same FAISS index as general documents

**Metadata Tags:**
```python
chunk_metadata = {
    "type": "medical_guideline",
    "category": "cardiovascular",
    "urgency_level": "high",
    "source": "CDC",
    "last_updated": "2024-10-15"
}
```

### 3. Dynamic Questioning Engine

**Algorithm:**
```
1. User reports symptom → Extract chief complaint
2. RAG search for relevant guidelines
3. LLM analyzes guidelines → generates next question
4. User responds → RAG searches for refined guidelines
5. Repeat until sufficient information gathered
6. Generate diagnosis, urgency, disposition
```

**Example Flow:**
```
User: "I have chest pain"
  ↓
RAG: Retrieve chest pain guidelines (CDC, AHA, Mayo)
  ↓
LLM: Analyze → Ask: "Is it crushing or squeezing?"
  ↓
User: "Yes, crushing"
  ↓
RAG: Retrieve MI-specific guidelines
  ↓
LLM: Ask: "Does it radiate to your left arm or jaw?"
  ↓
User: "Yes, left arm"
  ↓
LLM: High urgency → Ask critical questions (SOB, diaphoresis, syncope)
  ↓
Generate: Diagnosis (likely MI), Urgency (EMERGENCY), Disposition (Call 911)
```

### 4. Diagnosis & Disposition System

**Scoring Algorithm:**
```python
class DynamicDiagnosis:
    def __init__(self):
        self.symptoms_collected = []
        self.red_flags = []
        self.differential = {}  # diagnosis -> probability score
        self.urgency_score = 0  # 0-10 scale
    
    def update_from_rag(self, guideline_chunks):
        """Update differential and urgency based on RAG-retrieved guidelines"""
        # Extract diagnosis probabilities from guidelines
        # Weight by symptom matches
        # Calculate urgency based on red flags
    
    def get_disposition(self):
        """Return disposition recommendation"""
        if self.urgency_score >= 8:
            return "EMERGENCY - Call 911 immediately"
        elif self.urgency_score >= 6:
            return "URGENT - Visit ER within 1-2 hours"
        elif self.urgency_score >= 4:
            return "SEMI-URGENT - See doctor within 24 hours"
        else:
            return "ROUTINE - Schedule appointment within 1 week"
```

### 5. Implementation Phases

**Phase 1: Guideline Scraper (Week 1)**
- Build web scrapers for CDC, WHO, NIH
- Store as structured JSON
- Auto-update daily/weekly

**Phase 2: RAG Integration (Week 1-2)**
- Ingest guidelines into main RAG
- Add metadata support for guideline type
- Implement category-based filtering

**Phase 3: Dynamic Questioning (Week 2-3)**
- Build question generation from guidelines
- Implement conversation flow management
- Track symptoms and responses

**Phase 4: Diagnosis Engine (Week 3-4)**
- Implement differential diagnosis scoring
- Build urgency calculation algorithm
- Generate disposition recommendations

**Phase 5: Testing & Refinement (Week 4+)**
- Test with real medical scenarios
- Compare against rigid triage mode
- Refine question selection and urgency algorithms

## Technical Stack

**Scraping:**
- `requests` + `BeautifulSoup4` - HTML parsing
- `selenium` - For JavaScript-heavy sites
- `scrapy` - For large-scale scraping
- `schedule` - Auto-update cron

**Storage:**
- JSON files in `data/input/medical_guidelines/`
- Auto-ingested into FAISS RAG
- Metadata stored alongside embeddings

**Question Generation:**
- RAG retrieval → relevant guidelines
- LLM analyzes guidelines → generates contextual questions
- Tracks conversation state for follow-ups

## Advantages Over Rigid Triage

| Rigid Triage (JSON) | Dynamic Guidelines (RAG + LLM) |
|---------------------|-------------------------------|
| ❌ Hardcoded questions | ✅ Contextual questions from guidelines |
| ❌ Fixed decision tree | ✅ Adaptive questioning based on responses |
| ❌ Limited conditions (~8) | ✅ Unlimited (any condition with guidelines) |
| ❌ Manual updates required | ✅ Auto-updates from web |
| ❌ Outdated within months | ✅ Always current (daily/weekly scrapes) |
| ❌ Binary outcomes | ✅ Probabilistic diagnosis with confidence |

## Example Guideline Sources (Free & Public)

### High Priority:
1. **CDC Guidelines** - https://www.cdc.gov/
   - Disease-specific protocols
   - Public health recommendations

2. **NIH/MedlinePlus** - https://medlineplus.gov/
   - Consumer-friendly medical info
   - Evidence-based guidelines

3. **WHO** - https://www.who.int/
   - Global health guidelines
   - Pandemic protocols

### Medium Priority:
4. **NHS UK** - https://www.nhs.uk/conditions/
   - Comprehensive symptom guides
   - Treatment pathways

5. **Mayo Clinic** - https://www.mayoclinic.org/ (free articles)
   - Patient education
   - Symptom checkers

### Low Priority (Advanced):
6. **PubMed Central** - https://www.ncbi.nlm.nih.gov/pmc/
   - Research articles
   - Clinical studies

## Next Steps

**Immediate:**
1. Build proof-of-concept scraper for CDC chest pain guidelines
2. Test RAG ingestion and retrieval
3. Create dynamic questioning prototype

**Future:**
1. Expand to more conditions and sources
2. Implement auto-update scheduling
3. Add diagnosis confidence scoring
4. Build urgency/disposition algorithm

---

**Ready to start building?**


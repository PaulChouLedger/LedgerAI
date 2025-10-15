# Medical Vocabulary System for Whisper

## Overview

The Whisper transcription container now uses a **dynamic medical vocabulary system** that:
- Loads medical terms from a comprehensive JSON database
- Builds intelligent prompts to guide Whisper transcription
- Can learn new terms over time via API
- Supports 600+ medical terms across 15+ organ systems

## What Was Changed

### 1. Medical Terms Database
**File:** `whisper-container/medical_terms.json`

A comprehensive JSON file containing medical terminology organized by:
- **Organ Systems:** Cardiovascular, Respiratory, GI, Renal, Endocrine, Neurological, Musculoskeletal, Dermatology, Hematology
- **Specialties:** Infectious Disease, Psychiatric, Genitourinary, Ophthalmology, ENT
- **General:** Common symptoms, medications

**Total:** 600+ medical terms

### 2. Dynamic Prompt Generation
**File:** `whisper-container/container_rest.py`

The container now:
- Loads terms from JSON at startup
- Samples ~40 high-priority terms for the initial prompt
- Includes terms like: "pancreatitis", "diabetes", "hypertension", "myocardial infarction", etc.

### 3. Learning API
New endpoints for vocabulary management:

#### Add Medical Term
```bash
POST http://localhost:5000/add_medical_term
{
  "term": "rhabdomyolysis",
  "category": "musculoskeletal"  # optional, defaults to "learned"
}
```

#### View All Terms
```bash
GET http://localhost:5000/medical_terms
```

Returns:
- Total term count
- Terms per category
- Current prompt being used
- Full term dictionary

## How It Works

### Initial Prompt Construction

1. **Load JSON:** Container reads `medical_terms.json` at startup
2. **Sample Terms:** Takes first 5 terms from priority organ systems:
   - Cardiovascular
   - Respiratory
   - Gastrointestinal
   - Endocrine
   - Neurological
   - Common symptoms
   - Medications
3. **Build Prompt:** Creates a 40-term prompt like:
   ```
   "This is a medical conversation. Common terms include: 
   myocardial infarction, MI, heart attack, angina, pectoris, 
   pneumonia, bronchitis, asthma, COPD, emphysema, 
   gastroesophageal reflux disease, GERD, peptic ulcer, gastritis, 
   pancreatitis, ...
   Proper names and technical medical terms are important."
   ```

### Vocabulary Expansion (Future)

The system supports adding new terms:

**Manual Addition:**
```bash
curl -X POST http://localhost:5000/add_medical_term \
  -H "Content-Type: application/json" \
  -d '{"term": "cholecystitis", "category": "gastrointestinal"}'
```

**Programmatic Learning (future enhancement):**
- RAG could suggest terms based on document content
- LLM could identify medical terms in responses
- Manual correction system could add frequently missed terms

## Deployment

### Rebuild Container
```bash
cd /Users/rcabello/Documents/GitHub/LedgerAI
docker-compose build whisper-container
docker-compose up -d whisper-container
```

### Verify Medical Terms Loaded
```bash
# Check container logs for:
docker logs whisper-container 2>&1 | grep "Loaded.*medical terms"
# Should show: "[Whisper] 📚 Loaded 600+ medical terms from 15 categories"

# Or query the API:
curl http://localhost:5000/medical_terms | jq '.total_terms'
```

## Medical Terms by Category

### Current Coverage (600+ terms)

| Category | Sample Terms | Count |
|----------|-------------|-------|
| **Cardiovascular** | MI, angina, hypertension, CHF, arrhythmia | 32 |
| **Respiratory** | Pneumonia, asthma, COPD, dyspnea, hypoxia | 29 |
| **Gastrointestinal** | GERD, pancreatitis, hepatitis, cirrhosis | 31 |
| **Renal** | AKI, CKD, UTI, nephrotic syndrome | 28 |
| **Endocrine** | Diabetes, DKA, hypothyroidism, Cushing's | 30 |
| **Neurological** | Stroke, seizure, migraine, Parkinson's, MS | 35 |
| **Musculoskeletal** | Osteoarthritis, RA, gout, fracture | 28 |
| **Dermatology** | Eczema, psoriasis, cellulitis, melanoma | 26 |
| **Hematology** | Anemia, leukemia, lymphoma, thrombocytopenia | 24 |
| **Infectious** | Sepsis, COVID-19, meningitis, TB, HIV | 28 |
| **Psychiatric** | Depression, bipolar, anxiety, PTSD, OCD | 25 |
| **Genitourinary** | BPH, UTI, PCOS, endometriosis, PID | 24 |
| **Ophthalmology** | Glaucoma, cataracts, macular degeneration | 24 |
| **ENT** | Otitis media, sinusitis, vertigo, BPPV | 24 |
| **Symptoms** | Fever, dyspnea, nausea, chest pain | 28 |
| **Medications** | Metformin, lisinopril, atorvastatin, insulin | 28 |

## Examples

### Before Enhancement
**User says:** "What is pancreatitis?"
**Whisper hears:** "What is Bankercitis?"
❌ **Reason:** Model doesn't recognize medical term

### After Enhancement
**User says:** "What is pancreatitis?"
**Whisper hears:** "What is pancreatitis?"
✅ **Reason:** Term in initial prompt vocabulary

### Test Cases
```bash
# Test cardiovascular terms
"What causes myocardial infarction?"
"Explain atrial fibrillation"

# Test GI terms  
"What is pancreatitis?"
"Describe Crohn's disease"

# Test endocrine terms
"What is diabetic ketoacidosis?"
"Explain hypothyroidism"
```

## Future Enhancements

### 1. Adaptive Learning
- Monitor transcription corrections
- Automatically add frequently corrected terms
- RAG-based term suggestions from uploaded documents

### 2. Context-Aware Prompts
- Different prompts for different medical specialties
- Dynamic prompt based on conversation topic
- Patient-specific medical history terms

### 3. Pronunciation Dictionary
- Map common mispronunciations to correct terms
- Handle medical abbreviations
- Support drug brand/generic name variations

### 4. Quality Metrics
- Track terms successfully transcribed
- Identify problematic terms needing model upgrade
- A/B test different prompt strategies

## Troubleshooting

### Terms Not Being Recognized
1. **Check if term is in database:**
   ```bash
   curl http://localhost:5000/medical_terms | jq '.all_terms.cardiovascular'
   ```

2. **Verify prompt includes term:**
   ```bash
   curl http://localhost:5000/medical_terms | jq '.current_prompt'
   ```

3. **Add missing term:**
   ```bash
   curl -X POST http://localhost:5000/add_medical_term \
     -H "Content-Type: application/json" \
     -d '{"term": "rhabdomyolysis", "category": "musculoskeletal"}'
   ```

4. **Consider model upgrade:**
   - Current: `distil-small.en` (fast, lower accuracy)
   - Better: `distil-large-v3` (excellent accuracy, still fast)
   - Change in docker-compose.yml:
     ```yaml
     environment:
       - WHISPER_MODEL=distil-large-v3
     ```

### Container Won't Start
- Ensure `medical_terms.json` exists in `whisper-container/`
- Rebuild container: `docker-compose build whisper-container`
- Check logs: `docker logs whisper-container`

## API Reference

### GET /medical_terms
Returns complete medical vocabulary database

**Response:**
```json
{
  "total_terms": 600,
  "categories": 15,
  "terms_by_category": {
    "cardiovascular": 32,
    "respiratory": 29,
    ...
  },
  "current_prompt": "This is a medical conversation...",
  "all_terms": {
    "cardiovascular": ["MI", "angina", ...],
    ...
  }
}
```

### POST /add_medical_term
Add new term to vocabulary

**Request:**
```json
{
  "term": "cholecystitis",
  "category": "gastrointestinal"  // optional
}
```

**Response:**
```json
{
  "success": true,
  "message": "Added term 'cholecystitis' to category 'gastrointestinal'",
  "updated_prompt": "This is a medical conversation..."
}
```

## Integration with RAG

The RAG system can enhance vocabulary by:

1. **Document Analysis:**
   - Extract medical terms from uploaded PDFs
   - Add specialty-specific terms automatically

2. **Conversation Learning:**
   - Track which terms cause transcription errors
   - Suggest additions based on user corrections

3. **Smart Prompting:**
   - Include patient-specific conditions in prompt
   - Add medications from patient history
   - Customize based on medical specialty

Example integration in `listener.py`:
```python
# Get patient-specific terms from RAG
patient_terms = get_patient_medical_terms(user_name)

# Add to Whisper prompt
custom_prompt = f"{INITIAL_PROMPT} Patient history: {', '.join(patient_terms)}"

# Send to Whisper with enhanced prompt
response = requests.post(
    "http://localhost:5000/transcribe",
    files={"audio": wav_io},
    data={"initial_prompt": custom_prompt}
)
```

## Credits

Medical terminology sourced from:
- Common clinical diagnoses by organ system
- ICD-10 disease classifications  
- Standard medical abbreviations
- Commonly prescribed medications


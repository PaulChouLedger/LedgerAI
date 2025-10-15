# Medical Terminology Update System

## Overview

Automated monthly update system to keep medical terminology current using SNOMED CT standards.

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│           SNOMED CT (Standard Medical Codes)            │
│   - 350,000+ clinical concepts                          │
│   - Used by hospitals worldwide                         │
│   - Updated twice per year by SNOMED International      │
└─────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────┐
│              Monthly Update Pipeline                     │
│  snomed_updater.py (runs 1st of each month)            │
└─────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────┐
│         shared/medical_terms.json                        │
│  - Medical keywords (symptoms, conditions)               │
│  - Proper names (learned from Whisper)                  │
│  - Version: 2025.10                                     │
│  - Last updated: 2025-10-15                             │
└─────────────────────────────────────────────────────────┘
                         ↓
                   ┌─────┴─────┐
                   ↓           ↓
        ┌──────────────┐  ┌──────────────┐
        │ Whisper      │  │ LLM          │
        │ Container    │  │ Container    │
        │              │  │              │
        │ - Transcribe │  │ - Detect     │
        │   medical    │  │   medical    │
        │   terms      │  │   queries    │
        └──────────────┘  └──────────────┘
```

## Current Implementation (MVP)

### Phase 1: Curated Clinical Terms
**Status:** ✅ Ready to use (no license needed)

- **Source:** Hand-curated from clinical frequency data
- **Coverage:** ~300 most common terms across 8 specialties
- **Update method:** Manual review + community feedback
- **Cost:** $0
- **Maintenance:** Quarterly updates based on usage patterns

### Phase 2: UMLS Integration (Future)
**Status:** 🔄 Planned for Q1 2026

- **Source:** UMLS Metathesaurus (includes SNOMED CT)
- **Coverage:** 350,000+ clinical concepts
- **Update method:** Automated API calls
- **Cost:** Free (requires NIH UTS account)
- **Maintenance:** Automated monthly updates

### Phase 3: Full SNOMED CT (Enterprise)
**Status:** 💰 When revenue allows

- **Source:** SNOMED International license
- **Coverage:** Complete SNOMED CT with all relationships
- **Update method:** Biannual official releases
- **Cost:** ~$5,000-10,000/year
- **Maintenance:** Automated

---

## Usage

### Current (Curated Terms):

```bash
# Update medical_terms.json with curated terms
cd /home/aura/LedgerAI
python3 medical/snomed_updater.py

# Restart containers to apply
docker-compose restart aura-llm aura-whisper
```

### Future (UMLS API):

```bash
# Set your UMLS API key
export UMLS_API_KEY="your-key-here"

# Update from UMLS
python3 medical/snomed_updater.py --source umls

# Or run monthly via cron
0 2 1 * * /home/aura/LedgerAI/medical/monthly_update.sh
```

---

## File Structure

```
medical/
├── snomed_updater.py              # Main updater script
├── monthly_update.sh              # Cron job script (auto-generated)
├── update_log.txt                 # Update history
└── TERMINOLOGY_UPDATE_SYSTEM.md   # This file

shared/
└── medical_terms.json             # Live terminology database
    ├── medical_keywords: [...]    # Auto-updated monthly
    ├── proper_names: [...]        # Learned from usage
    ├── version: "2025.10"         # Update cycle
    └── last_updated: "2025-10-15" # Timestamp
```

---

## Curated Term Categories

### 1. **Gastrointestinal** (~30 terms)
- Symptoms: abdominal pain, nausea, vomiting, diarrhea, constipation
- Conditions: appendicitis, pancreatitis, cholecystitis, GERD, IBS
- Anatomical: RUQ, RLQ, LUQ, LLQ, epigastric, periumbilical

### 2. **Cardiovascular** (~30 terms)
- Symptoms: chest pain, palpitations, dyspnea, syncope, edema
- Conditions: MI, heart failure, AFib, DVT, PE, hypertension
- Descriptors: substernal, radiating, crushing, pressure

### 3. **Respiratory** (~25 terms)
- Symptoms: cough, SOB, wheezing, hemoptysis, pleuritic pain
- Conditions: pneumonia, asthma, COPD, pneumothorax, URI

### 4. **Neurological** (~25 terms)
- Symptoms: headache, dizziness, weakness, numbness, seizure
- Conditions: stroke, TIA, migraine, meningitis, concussion

### 5. **Musculoskeletal** (~20 terms)
- Symptoms: joint pain, back pain, stiffness, swelling
- Conditions: arthritis, gout, fracture, herniated disc

### 6. **Renal/Urological** (~20 terms)
- Symptoms: dysuria, hematuria, frequency, urgency, flank pain
- Conditions: UTI, kidney stone, pyelonephritis, renal failure

### 7. **Endocrine** (~20 terms)
- Symptoms: polyuria, polydipsia, weight changes, tremor
- Conditions: diabetes, thyroid disorders, DKA

### 8. **Infectious Disease** (~20 terms)
- Symptoms: fever, chills, night sweats, malaise
- Conditions: sepsis, COVID, influenza, hepatitis

---

## Update Schedule

### Monthly (Automated)
- Review usage logs
- Add frequently used terms
- Remove unused terms
- Increment version number

### Quarterly (Manual Review)
- Review medical literature for new terminology
- Align with ICD-10/CPT code updates
- Community feedback integration

### Annually (Major Update)
- Align with SNOMED CT official releases (Jan/July)
- Major terminology review
- Clinical validation

---

## Expansion Roadmap

### Q4 2025 (Current)
- ✅ Curated common terms (~300 total)
- ✅ 8 specialty categories
- ✅ Manual updates as needed

### Q1 2026
- 🔄 UMLS API integration
- 🔄 Automated monthly updates
- 🔄 Expand to 1,000+ terms

### Q2 2026
- 💰 Consider SNOMED CT license
- 💰 Full terminology coverage (350,000+ concepts)
- 💰 Automated biannual updates

---

## Benefits of This Approach

### For Users
- ✅ Recognition of natural language ("belly ache" = "abdominal pain")
- ✅ Medical synonyms understood
- ✅ Always current terminology
- ✅ Better transcription accuracy

### For Developers
- ✅ **No code changes** for new terms
- ✅ **Data-driven** - edit JSON, not Python
- ✅ **Version controlled** - Track changes over time
- ✅ **Automated** - Set and forget monthly updates

### For Medical Staff
- ✅ Can add terms without programming
- ✅ Specialty-specific organization
- ✅ Clinical validation workflow possible

---

## Adding New Terms (Manual)

### Quick Add:
```bash
# Edit shared/medical_terms.json
{
  "medical_keywords": [
    ...,
    "new_symptom_here",
    "new_condition_here"
  ]
}

# Restart containers
docker-compose restart aura-llm aura-whisper
```

### Organized Add (via updater):
```bash
# Edit medical/snomed_updater.py
# Add to appropriate specialty in get_common_clinical_terms()

# Run updater
python3 medical/snomed_updater.py

# Containers auto-restart (if cron enabled)
```

---

## UMLS API Integration (Future)

### Setup:
1. Create free account: https://uts.nlm.nih.gov/uts/signup-login
2. Get API key from UTS profile
3. Set environment variable:
   ```bash
   export UMLS_API_KEY="your-key-here"
   ```

### Usage:
```bash
python3 medical/snomed_updater.py --source umls
```

### API Endpoints:
- Get SNOMED CT concepts: `/rest/search/current?string={term}`
- Get synonyms: `/rest/content/current/CUI/{cui}/atoms`
- Get relationships: `/rest/content/current/CUI/{cui}/relations`

---

## Monitoring & Logs

### View Update History:
```bash
cat medical/update_log.txt
```

### Sample Log Entry:
```
✅ Medical terminology updated: 2025-10-01 02:00:01
   Version: 2025.10
   Keywords: 318
   Source: curated_clinical_terms
```

---

## Quality Assurance

### Before Each Update:
1. ✅ Validate JSON syntax
2. ✅ Check for duplicates
3. ✅ Verify specialty categorization
4. ✅ Test with sample queries

### After Each Update:
1. ✅ Restart containers
2. ✅ Run test queries
3. ✅ Monitor transcription accuracy
4. ✅ Review Aura logs for missed terms

---

## Future Enhancements

1. **Usage Analytics**
   - Track which terms are searched most
   - Auto-prioritize frequently used terms
   - Remove rarely used terms

2. **Community Contributions**
   - Medical staff can suggest new terms
   - Review/approval workflow
   - Version control for accountability

3. **Multi-Language Support**
   - Spanish medical terms
   - Other languages as needed
   - Synonym mapping across languages

4. **Automatic Learning**
   - Whisper corrections → new term suggestions
   - RAG search failures → identify gaps
   - LLM feedback → refine terminology

---

**Next Steps:**

1. Run `python3 medical/snomed_updater.py` to populate initial terms
2. Test with various medical queries
3. Add specialty-specific terms as needed
4. Set up monthly cron job when stable


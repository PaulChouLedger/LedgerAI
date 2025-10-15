# Aura Clinician Mode - Development Roadmap

## Vision

Aura will evolve from a **hardcoded triage system** to an **intelligent clinician** that thinks like a doctor, using RAG to access comprehensive medical guidelines and reason through diagnoses.

---

## 4-Mode Architecture

### 1. **CASUAL Mode** 💬
**Triggers:** Simple greetings, small talk
```
Examples: "Hello", "How are you?", "Good morning"
```
**Behavior:** Friendly, brief responses
**RAG:** Not used

### 2. **THINKER Mode** 🧠
**Triggers:** Knowledge/information queries
```
Examples: "What is myocardial infarction?", "Who is Rafael Cabello?", "Explain diabetes"
```
**Behavior:** Comprehensive, detailed answers
**RAG:** ✅ Searches knowledge base, returns thorough information

### 3. **TRIAGE Mode** 🏥 (Current System - Keep Intact)
**Triggers:** Medical symptoms (for now)
```
Examples: "I have chest pain", "My head hurts"
```
**Behavior:** Structured, hardcoded questions from JSON definitions
**RAG:** Not used (relies on triage_defs/)
**Status:** ✅ Working, keep as baseline while developing Clinician

### 4. **CLINICIAN Mode** 👨‍⚕️ (NEW - In Development)
**Triggers:** Medical symptoms (future)
```
Examples: "I have chest pain", "I'm experiencing dizziness"
```
**Behavior:** 
- Thinks like a real doctor
- Asks intelligent, context-aware questions
- Uses RAG to search medical guidelines
- Builds differential diagnosis
- Adapts based on findings

**RAG:** ✅ Heavily uses multi-organ medical guidelines
**Status:** 🚧 Framework created, needs medical guideline database

---

## Migration Path

### Phase 1: Build Foundation (Current)
- [x] Create `clinician.py` framework
- [x] Add feature flag (`use_clinician_mode = False`)
- [x] Integrate with container_rest.py
- [ ] Test basic flow without medical data

### Phase 2: Build Medical Guideline Database
**Goal:** Create comprehensive, RAG-searchable medical knowledge

**Structure:**
```
llm-container/medical_guidelines/
├── cardiovascular/
│   ├── chest_pain.md
│   ├── palpitations.md
│   ├── hypertension.md
│   └── myocardial_infarction.md
├── respiratory/
│   ├── shortness_of_breath.md
│   ├── cough.md
│   ├── pneumonia.md
│   └── asthma.md
├── neurology/
│   ├── headache.md
│   ├── dizziness.md
│   ├── stroke.md
│   └── seizures.md
├── gastroenterology/
│   ├── abdominal_pain.md
│   ├── nausea_vomiting.md
│   └── gi_bleeding.md
└── ... (more organ systems)
```

**Content for Each File:**
```markdown
# Chest Pain - Clinical Guidelines

## Differential Diagnosis
- Acute Coronary Syndrome (ACS)
- Pulmonary Embolism (PE)
- Aortic Dissection
- Pneumothorax
- GERD
- Musculoskeletal

## Critical Red Flags
- Radiation to jaw/arm
- Diaphoresis
- Dyspnea
- Syncope

## Key History Questions
1. Character: Sharp, dull, crushing, burning?
2. Location: Substernal, lateral, diffuse?
3. Radiation: To arm, jaw, back?
4. Onset: Sudden, gradual?
5. Duration: Seconds, minutes, hours?
6. Aggravating factors: Exertion, eating, deep breath?
7. Relieving factors: Rest, position change?
8. Associated symptoms: SOB, sweating, nausea?
9. Risk factors: Smoking, HTN, diabetes, family history?

## Physical Exam Findings
- Vital signs (BP, HR, RR, O2 sat)
- Cardiac auscultation
- Lung sounds
- Chest wall tenderness

## Workup Recommendations
- ECG (STAT if ACS suspected)
- Troponin
- Chest X-ray
- D-dimer (if PE suspected)

## Disposition Guidelines
- **EMERGENCY:** ACS, PE, dissection → 911
- **URGENT:** Unclear etiology, risk factors → ED within 2-4 hours
- **NON-URGENT:** Likely musculoskeletal → PCP within 24-48 hours
```

**How to Populate:**
1. Use UpToDate, medical textbooks, clinical guidelines
2. Extract key diagnostic reasoning for each chief complaint
3. Convert to markdown for easy RAG ingestion
4. Process through auto-ingest system

### Phase 3: Enable Clinician Mode (Beta)
- [ ] Generate medical guidelines (20-30 common complaints)
- [ ] Ingest into RAG system
- [ ] Set `use_clinician_mode = True`
- [ ] Test with real queries
- [ ] Compare quality vs Triage mode

### Phase 4: Parallel Operation
- [ ] Run both TRIAGE (fallback) and CLINICIAN (primary) in parallel
- [ ] User can choose mode or default to CLINICIAN
- [ ] Monitor quality and accuracy
- [ ] Iterate on clinician logic

### Phase 5: Full Migration
- [ ] Clinician mode handles all medical queries
- [ ] Deprecate triage_defs/ (keep as reference)
- [ ] TRIAGE mode becomes legacy fallback only
- [ ] Clinician becomes primary diagnostic engine

---

## Implementation Details

### Current State Files

**Triage State:**
```python
{
    "condition": "chest_pain",      # From triage_defs/
    "step_index": 3,
    "answers": ["severe", "yes", "no"],
    "flags": {...}
}
```

**Clinician State:**
```python
{
    "mode": "clinician",
    "chief_complaint": "I have chest pain",
    "conversation_history": [...],
    "findings": {
        "severity": "severe",
        "onset": "acute",
        "radiation": "left arm"
    },
    "differential_diagnoses": ["ACS", "PE", "Dissection"],
    "current_focus": "cardiovascular"
}
```

### Routing Logic

```python
# Priority order:
1. Check for active clinician session → Continue CLINICIAN
2. Check for active triage session → Continue TRIAGE
3. Check if use_clinician_mode=True and is_clinician_trigger() → Start CLINICIAN
4. Check detect_condition() → Start TRIAGE (fallback)
5. Check is_knowledge_query() → THINKER mode
6. Default → CASUAL mode
```

### Feature Flags

**Current:**
```python
use_clinician_mode = False  # Disabled - still building
```

**Beta Testing:**
```python
use_clinician_mode = True   # Enable for testing
clinician_fallback_to_triage = True  # Fall back if clinician fails
```

**Production:**
```python
use_clinician_mode = True   # Primary mode
clinician_fallback_to_triage = False  # Clinician handles everything
```

---

## Example Clinician Conversation Flow

**User:** "I have chest pain"

**Clinician Response 1:**
```
I understand you're experiencing chest pain. This is important to evaluate carefully.
Can you describe the pain - is it sharp, dull, crushing, or burning?
```
*[RAG searched: "chest pain differential diagnosis"]*

**User:** "It's a crushing pain in the center of my chest"

**Clinician Response 2:**
```
Crushing central chest pain can be concerning. Does this pain radiate anywhere, 
such as to your left arm, jaw, or back?
```
*[RAG searched: "chest pain crushing radiation acute coronary syndrome"]*

**User:** "Yes, it goes down my left arm"

**Clinician Response 3:**
```
This is very concerning for a possible heart problem. Do you have any 
shortness of breath, sweating, or nausea with this pain?
```
*[RAG searched: "acute coronary syndrome associated symptoms"]*

**User:** "Yes, I'm sweating and short of breath"

**Clinician Final Response:**
```
Based on your symptoms - crushing chest pain radiating to the left arm with 
sweating and shortness of breath - this is highly concerning for a heart attack.

EMERGENCY RECOMMENDATION: Call 911 immediately or have someone drive you to the 
nearest emergency room. Do not drive yourself. This requires immediate evaluation 
with an EKG and blood tests.

While you wait for help, sit down and try to stay calm. If you have aspirin and 
are not allergic, chewing one aspirin may be beneficial.
```
*[RAG searched: "acute coronary syndrome emergency management"]*

---

## Advantages Over Current Triage

| Feature | Current TRIAGE | Future CLINICIAN |
|---------|---------------|------------------|
| **Question Logic** | Hardcoded in JSON | RAG-powered, adaptive |
| **Medical Knowledge** | Limited to triage_defs/ | Entire medical guidelines database |
| **Flexibility** | Fixed question order | Adapts based on responses |
| **Comprehensiveness** | 8 organ systems, ~40 conditions | Unlimited, expandable |
| **Updates** | Manual JSON editing | Add new guidelines to RAG |
| **Reasoning** | Rule-based | LLM reasoning + medical evidence |
| **Follow-up Questions** | Not supported | Natural conversation flow |

---

## Next Steps

### Immediate (This Session)
1. ✅ Created `clinician.py` framework
2. ✅ Added integration hooks in `container_rest.py`
3. ✅ Added feature flag (disabled by default)
4. 📝 Created this roadmap

### Short Term (Next Development Sessions)
1. Create medical_guidelines directory structure
2. Start with 5-10 common chief complaints
3. Write comprehensive guideline docs
4. Ingest into RAG system
5. Enable feature flag and test

### Long Term (Production)
1. Expand to 50+ chief complaints across all organ systems
2. Add overnight conversation analysis
3. Clinician learns from past interactions
4. Proactive health insights
5. Full replacement of TRIAGE mode

---

## Notes

- **Don't delete triage_defs/** - Keep as reference and fallback
- **Feature flag approach** - Easy A/B testing
- **Gradual rollout** - Test thoroughly before full migration
- **Session state compatibility** - Both modes can coexist
- **RAG is key** - Quality of medical guidelines determines quality of diagnosis


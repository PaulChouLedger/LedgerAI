# Curated Diagnostic Guidelines System

## Overview

Hand-curated, evidence-based diagnostic guidelines designed for **RAG-powered dynamic medical assessment**.

Unlike web scraping or generic medical APIs, these guidelines are:
- ✅ **Structured for diagnosis** - Includes differential diagnoses, red flags, and targeted questions
- ✅ **Optimized for LLM consumption** - Clear, comprehensive text format
- ✅ **Evidence-based** - Sourced from AAFP, ACS, ACEP, and standard clinical references
- ✅ **Maintainable** - JSON format, version controlled, updatable

## Architecture

```
Curated JSON Guidelines
         ↓
   Converter Script
         ↓
   RAG Text Format
         ↓
   Embedding Generation
         ↓
    RAG Retrieval
         ↓
 LLM Dynamic Questioning
```

## Directory Structure

```
medical/
├── README_CURATED_GUIDELINES.md          # This file
├── CURATED_GUIDELINES_PLAN.md            # Roadmap (8 systems, 160 diagnoses)
├── diagnostic_guideline_schema.json      # JSON schema definition
│
├── guidelines/                            # Source guidelines (JSON)
│   ├── GI_Acute_Appendicitis.json
│   ├── GI_Acute_Pancreatitis.json
│   ├── CV_Myocardial_Infarction.json
│   └── ... (160 total when complete)
│
├── convert_guidelines_to_rag.py          # Converter script
└── ingest_guidelines.py                  # RAG ingestion pipeline
```

## Workflow

### 1. Create Guideline (JSON)

```bash
# Create new guideline following the schema
vim medical/guidelines/GI_Acute_Appendicitis.json
```

See `diagnostic_guideline_schema.json` for the complete structure.

### 2. Convert to RAG Format

```bash
cd /home/aura/LedgerAI
python3 medical/convert_guidelines_to_rag.py
```

This creates RAG-optimized text files in `data/input/`.

### 3. Ingest into RAG

```bash
python3 medical/ingest_guidelines.py
```

This:
- Extracts text
- Builds embeddings (on host, due to FAISS issues in container)
- Makes guidelines available to RAG system

### 4. Test Dynamic Questioning

Start Aura and say: **"I have abdominal pain"**

The system will:
1. RAG retrieves relevant guidelines (appendicitis, pancreatitis, etc.)
2. LLM reads diagnostic questions from guidelines
3. LLM asks targeted questions in natural language
4. LLM analyzes responses and narrows differential
5. LLM arrives at most likely diagnosis

## Guideline Coverage Plan

### Phase 1: Gastrointestinal (Priority)
- [x] Acute Appendicitis
- [ ] Acute Pancreatitis
- [ ] Cholecystitis
- [ ] Gastroenteritis
- [ ] Peptic Ulcer Disease
- [ ] ... (15 more)

**Status:** 1/20 complete

### Phase 2: Cardiovascular
- [ ] Acute Myocardial Infarction
- [ ] Unstable Angina
- [ ] Pulmonary Embolism
- [ ] ... (17 more)

**Status:** 0/20 complete

### Phase 3-8: Other Systems
See `CURATED_GUIDELINES_PLAN.md` for full roadmap.

## Guideline Structure

Each guideline includes:

### Essential Components
- **Chief Complaint Triggers** - Keywords that suggest this diagnosis
- **Classic Presentation** - Typical clinical picture
- **Diagnostic Questions** - Ordered questions with expected responses
- **Red Flags** - Emergency warning signs
- **Differential Diagnoses** - How to distinguish from similar conditions

### Additional Components
- Physical exam findings
- Diagnostic tests
- Treatment summary
- Evidence sources

## Example: Acute Appendicitis

```json
{
  "condition": "Acute Appendicitis",
  "chief_complaint_triggers": ["abdominal pain", "belly pain", "RLQ pain"],
  "diagnostic_questions": [
    {
      "question_focus": "pain onset and migration pattern",
      "diagnostic_value": "critical",
      "expected_positive_responses": [
        "started around belly button",
        "moved to right lower side"
      ],
      "context": "Classic migration from periumbilical to RLQ is highly specific"
    }
    // ... more questions
  ]
}
```

When converted to RAG text, the LLM reads this and generates questions like:
- "Where did the pain start, and has it moved?"
- "Did it begin around your belly button and then shift to the lower right side?"

## Future: UpToDate API Integration

When revenue allows, the system is designed to integrate with UpToDate API:

```python
# medical/uptodate_api_client.py (future)

def get_uptodate_guideline(condition):
    """Fetch latest evidence-based guideline from UpToDate"""
    # API call to UpToDate
    # Convert to our JSON schema
    # Auto-update guidelines monthly
```

This will provide:
- Real-time updates
- Thousands of conditions
- Cutting-edge clinical evidence

But for now, curated guidelines provide:
- Zero cost
- High quality for common conditions
- Full control over content
- Perfect for MVP and initial deployment

## Quality Assurance

All guidelines are:
1. **Evidence-based** - Sourced from AAFP, ACS, ACEP guidelines
2. **Peer-reviewed** - Validated against standard clinical references
3. **Version controlled** - Git tracks all changes
4. **Dated** - `last_reviewed` field for maintenance tracking

## Maintenance Schedule

- **Quarterly:** Review and update existing guidelines
- **Annually:** Major review against latest CPGs
- **As needed:** Add new diagnoses based on usage patterns

## Contributing Guidelines

When adding new diagnoses:

1. Follow `diagnostic_guideline_schema.json` structure
2. Include at least 5 diagnostic questions
3. List 3-5 differential diagnoses
4. Include red flags
5. Cite evidence sources
6. Run converter and test with RAG

## Current Statistics

- **Total Guidelines:** 1
- **Coverage:** Gastrointestinal (1/20)
- **RAG-Ready Files:** 1
- **Last Updated:** 2025-10-15

## Next Steps

1. Complete GI system (19 more diagnoses)
2. Build Cardiovascular system (20 diagnoses)
3. Build Respiratory system (20 diagnoses)
4. Continue with remaining 5 organ systems

---

**Questions?** See `CURATED_GUIDELINES_PLAN.md` for the full development roadmap.


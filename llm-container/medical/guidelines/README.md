# Medical Guidelines Organization

This directory contains curated medical guidelines organized by organ system for the Adaptive Diagnostic Engine.

## Structure

```
guidelines/
├── GI/          # Gastrointestinal (22 guidelines)
├── GU/          # Genitourinary (4 guidelines)
└── GYN/         # Gynecologic (4 guidelines)
```

## Organ Systems

### GI (Gastrointestinal) - 22 Guidelines
- Acute Appendicitis
- Acute Cholecystitis
- Acute Gastroenteritis
- Acute Pancreatitis
- Biliary Colic
- Bowel Obstruction
- Cecal Volvulus
- Cholangitis
- Constipation
- Diverticulitis
- GERD
- Gastric Outlet Obstruction
- Gastritis
- Hepatitis
- IBD Flare
- IBS
- Incarcerated Hernia
- Mallory-Weiss Tear
- Mesenteric Ischemia
- Peptic Ulcer Disease
- Perforated Viscus
- Sigmoid Volvulus

### GU (Genitourinary) - 4 Guidelines
- Kidney Stone
- Prostatitis
- Testicular Torsion
- UTI/Pyelonephritis

### GYN (Gynecologic) - 4 Guidelines
- Ectopic Pregnancy
- Ovarian Torsion
- Pelvic Inflammatory Disease (PID)
- Ruptured Ovarian Cyst

## Guideline Format

Each guideline is a JSON file with:
- `condition`: Full condition name
- `icd10`: ICD-10 code
- `snomed`: SNOMED CT code
- `sex`: male / female / both (for filtering by patient's biological sex)
- `prevalence`: common / uncommon / rare
- `chief_complaint_triggers`: Keywords that match this condition
- `urgency`: emergent / urgent / routine
- `key_features.classic_presentation`: OLDCARTS-based description
- `red_flags`: Critical warning signs

## Adding New Guidelines

1. Create JSON file following the template in `OLDCARTS_GUIDELINE_TEMPLATE.txt`
2. Place in appropriate organ system subdirectory
3. Ensure all OLDCARTS elements are covered in `classic_presentation`
4. Include evidence-based prevalence classification
5. Add critical red flags

## Total: 30 Guidelines

The system uses these guidelines for:
- Chief complaint matching
- Rolling top-3 differential diagnosis
- LLM-driven question generation
- Semantic similarity scoring
- Urgency classification
- Red flag screening


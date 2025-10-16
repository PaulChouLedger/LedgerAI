# Medical Guideline Creation Reference

## Evidence-Based Prevalence Classification

This document provides the **evidence-based prevalence classification system** for creating new medical guidelines. Use this to ensure consistency across all conditions.

### Data Sources
- **PMC5075866**: Large urban ED study (5,340 cases of acute abdominal pain)
- **PMC4535107**: Adult non-traumatic acute abdominal pain study
- **UpToDate, NEJM**: Clinical practice guidelines

---

## Prevalence Categories

### **COMMON** (>3% prevalence or routinely considered in differential)
Conditions that are frequently encountered in emergency/acute care settings and should be prioritized in differential diagnosis.

**Initial Score: 0.60**

| Condition | Prevalence | Age Considerations |
|-----------|------------|-------------------|
| Acute Appendicitis | 10-23% | Most common in young adults |
| Acute Cholecystitis | 7-10% | Increases with age (13% in >65) |
| Biliary Colic | 7-10% | Common in adults with gallstones |
| Kidney Stone (Renal Colic) | 3-16% | Peak 30-50 years |
| UTI/Pyelonephritis | 5-12% | More common in females |
| Acute Pancreatitis | 3-11% | Alcohol, gallstones |
| Acute Gastroenteritis | 5-10% | All ages |
| Peptic Ulcer Disease | ~4% | NSAIDs, H. pylori |

---

### **UNCOMMON** (1-3% prevalence, clinically important)
Conditions that are less frequent but must still be considered in appropriate clinical contexts.

**Initial Score: 0.50**

| Condition | Prevalence | Age Considerations |
|-----------|------------|-------------------|
| Acute Diverticulitis | 2-7% | Much higher in >50 years (7.3% in >65) |
| Small Bowel Obstruction | 0.7-2.3% | Higher in elderly (2.3% in >65) |
| Acute Gastritis | 2-4% | Common outpatient, less acute |
| GERD | 2-3% | Very common outpatient, rare acute ED |
| Ruptured Ovarian Cyst | 2-4% | Reproductive age females only |
| IBD Flare (Crohn's/UC) | 1-2% | Known IBD history |
| Severe Constipation | 1-3% | Mostly outpatient management |
| IBS | <1% acute | Chronic diagnosis, rare acute presentation |

---

### **RARE** (<1% prevalence, but can't miss)
Life-threatening or time-sensitive conditions with low prevalence but high morbidity/mortality.

**Initial Score: 0.40**

| Condition | Prevalence | Clinical Significance |
|-----------|------------|----------------------|
| Perforated Viscus | <1% | SURGICAL EMERGENCY - peritonitis |
| Acute Mesenteric Ischemia | <0.5% | SURGICAL EMERGENCY - high mortality |
| Ovarian Torsion | <1% | SURGICAL EMERGENCY - reproductive age |
| Ectopic Pregnancy | <1% | LIFE-THREATENING - reproductive age |
| Acute Hepatitis | <1% | Rare acute presentation |

---

## Guideline JSON Template

When creating new guidelines, use this exact format:

```json
{
  "condition": "Condition Name",
  "icd10": "XXX.XX",
  "snomed": "XXXXXXXX",
  "prevalence": "common|uncommon|rare",
  "chief_complaint_triggers": [
    "trigger phrase 1",
    "trigger phrase 2"
  ],
  "urgency": "emergent|urgent|routine",
  "key_features": {
    "classic_presentation": "Detailed clinical presentation with KEY FEATURES in ALL CAPS. Include: pain characteristics (onset, location, quality, severity, duration), associated symptoms, key history, key positives, key negatives, red flags, and discriminating features."
  },
  "red_flags": [
    "Red flag 1 - consequence/action needed",
    "Red flag 2 - consequence/action needed"
  ]
}
```

---

## Field Definitions

### `prevalence`
- **common**: >3% prevalence OR routinely considered in differential
- **uncommon**: 1-3% prevalence, clinically important
- **rare**: <1% prevalence but can't miss

### `urgency`
- **emergent**: Immediate life-threat, needs 911/ER now (shock, perforation, torsion, ectopic)
- **urgent**: Needs ER evaluation within hours (appendicitis, cholecystitis, pancreatitis)
- **routine**: Can wait for PCP/urgent care (GERD, IBS, constipation)

### `classic_presentation`
**Must include ALL of these elements:**
1. **Pain characteristics**: Onset (acute vs gradual), Location, Quality (sharp/dull/cramping), Severity (1-10), Duration
2. **Associated symptoms**: Nausea, vomiting, fever, diarrhea, etc.
3. **Key history**: Risk factors, triggers, prior episodes
4. **Key positives**: Findings that support diagnosis
5. **Key negatives**: Findings that argue against diagnosis
6. **Discriminating features**: What makes this unique vs similar conditions

**Use ALL CAPS for:**
- Key discriminating features
- Most important clinical findings
- Critical red flags
- Migration patterns
- Temporal relationships

---

## Prevalence Assignment Rules

### Age-Dependent Conditions
For conditions with significant age variation (e.g., diverticulitis):
- Classify based on **overall adult population** prevalence
- Note age-specific data in `classic_presentation`
- Example: Diverticulitis is "uncommon" overall (2-3%), but note "7.3% in >65 years" in presentation

### Gender-Specific Conditions
For conditions specific to one gender (ectopic, ovarian torsion):
- Classify based on prevalence **within the at-risk population**
- Ectopic: <1% of all abd pain, but ~5-10% of reproductive-age females with abd pain → **rare** overall
- Note population specificity in `classic_presentation`

### Outpatient vs Acute Conditions
- **GERD, IBS, Constipation**: Very common outpatient, rare acute ED → **uncommon**
- **Gastroenteritis, UTI**: Common both outpatient and acute → **common**

---

## Quality Control Checklist

Before finalizing a new guideline:
- [ ] Prevalence matches evidence-based data or clinical consensus
- [ ] ICD-10 and SNOMED codes verified
- [ ] `classic_presentation` includes ALL required elements
- [ ] Key discriminating features in ALL CAPS
- [ ] Red flags are specific and actionable
- [ ] Triggers cover common patient phrasings
- [ ] Urgency appropriate for time-sensitivity
- [ ] JSON format validated (no trailing commas, proper escaping)

---

## References

1. **PMC5075866**: "Causes of acute abdominal pain in emergency department" - 5,340 cases, large urban ED
2. **PMC4535107**: "Non-traumatic acute abdominal pain in adults" - surgical outcomes study
3. **UpToDate**: Clinical decision support for prevalence and management
4. **NEJM**: Evidence-based clinical reviews

---

## Update Schedule

- **Review prevalence data**: Annually
- **Update guidelines**: When new evidence emerges or prevalence patterns shift
- **Add new conditions**: As clinically relevant conditions are identified

---

**Last Updated**: October 2025  
**Next Review**: October 2026


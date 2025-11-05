# ML Learning System for Synonyms and Guidelines

## Overview

This ML learning system automatically learns new synonym expansions and guideline terms from patient interactions, facilitating continuous improvement of the diagnostic system.

## Components

### 1. SynonymLearner (`learn_synonyms.py`)

Tracks patient responses that successfully match medical terms and suggests new synonyms when patterns are detected.

**Features:**
- Records successful term matches with confidence scores
- Analyzes patterns to find frequently used patient phrases
- Suggests new synonyms that aren't already in the synonym files
- Filters out semantically redundant synonyms

**Usage:**
```python
from ml.learn_synonyms import SynonymLearner

learner = SynonymLearner()

# Record a successful match
learner.record_interaction(
    user_input="I feel nauseous",
    matched_term="nausea",
    oldcarts_element="associated",
    organ_system="GI",
    confidence=0.85
)

# Generate suggestions (after collecting enough data)
suggestions = learner.generate_suggestions(min_occurrences=3)

# Apply a suggestion
learner.apply_suggestion("GI", "associated.nausea", "feeling nauseous")
```

### 2. GuidelineLearner (`learn_guidelines.py`)

Tracks patient responses that don't match existing guideline terms and suggests new terms to add.

**Features:**
- Records unmatched or low-confidence responses
- Analyzes patterns to find frequently used patient phrases
- Suggests new terms with medical term proposals
- Adds terms to guideline JSON files

**Usage:**
```python
from ml.learn_guidelines import GuidelineLearner

learner = GuidelineLearner()

# Record an unmatched response
learner.record_unmatched_response(
    user_input="it feels like my stomach is on fire",
    oldcarts_element="character",
    organ_system="GI",
    condition="Acute Gastritis",
    matched_confidence=0.3
)

# Generate suggestions
suggestions = learner.generate_suggestions(min_occurrences=3)

# Apply a suggestion
learner.apply_suggestion(
    "GI", "Acute Gastritis", "character",
    "it feels like my stomach is on fire",
    medical_term="burning sensation"
)
```

### 3. Review Script (`review_suggestions.py`)

Interactive script to review and apply ML-generated suggestions.

**Usage:**
```bash
cd llm-medical-container
python3 ml/review_suggestions.py
```

**Options:**
- `[1]` Review Synonym Suggestions
- `[2]` Review Guideline Suggestions  
- `[3]` Review Both
- `[q]` Quit

## Integration

The ML learning system is integrated into `adaptive_diagnostic_engine.py`:

1. **Synonym Learning**: Automatically records successful term matches when FAISS finds matches with confidence ≥ 0.75
2. **Guideline Learning**: Automatically records unmatched responses when similarity < 0.6

**Enable Learning:**
Set environment variable:
```bash
export ENABLE_ML_LEARNING=true
```

Or in `.env`:
```
ENABLE_ML_LEARNING=true
```

## Data Storage

- **Interaction History**: `data/learning/synonym_interactions.jsonl`
- **Unmatched Responses**: `data/learning/guideline_unmatched.jsonl`
- **Suggestions**: 
  - `data/learning/synonym_suggestions.json`
  - `data/learning/guideline_suggestions.json`

## Workflow

1. **Collection Phase**: System runs with `ENABLE_ML_LEARNING=true`, collecting interactions
2. **Analysis Phase**: Run `review_suggestions.py` to generate suggestions
3. **Review Phase**: Review suggestions interactively
4. **Application Phase**: Apply approved suggestions to synonym/guideline files
5. **Verification**: System automatically uses new synonyms/terms in future interactions

## Configuration

**Synonym Learning:**
- `min_occurrences`: Minimum times a pattern must occur (default: 3)
- `min_confidence`: Minimum confidence to consider (default: 0.7)

**Guideline Learning:**
- `min_occurrences`: Minimum times a pattern must occur (default: 3)
- `max_confidence`: Maximum confidence to consider (default: 0.5)

## Notes

- ML learning is **opt-in** via `ENABLE_ML_LEARNING` environment variable
- Suggestions require manual review before application
- Semantic matching handles similarity, so suggestions focus on truly distinct phrases
- Empty synonym lists are handled gracefully
- All file paths updated to use `medical/synonyms/` directory


# Urgency-Based Assessment System

## Overview

The medical guidelines have been restructured to include urgency designations within each OLDCARTS element. This replaces the separate `red_flags` section and allows for more granular determination of care level needed.

**Phase 1 (Current)**: Simple binary system - only `emergent` flags for red flag symptoms.  
**Phase 2 (Future)**: Expand to include `urgent` and `routine` urgency levels for comprehensive triage.

## New Structure

Items in `structured_oldcarts` that correspond to red flags include an `urgency` field:
- **emergent**: Only items matching red flag statements have this field. Triggers elevation of care level.
- Items without `urgency` field: Normal symptoms that don't elevate care level.

### Example

```json
{
  "associated": {
    "includes": [
      {
        "medical": "pyrexia",
        "patient_friendly": "fever",
        "urgency": "emergent"
      },
      {
        "medical": "nausea",
        "patient_friendly": "nausea"
      }
    ]
  }
}
```

Only items matching red flags have the `urgency: "emergent"` field. Other items are normal symptoms.

## Benefits

1. **Simplified Care Level Determination**: Only red flag symptoms trigger emergent care elevation
2. **Integrated Assessment**: Urgency is part of the normal questioning flow, not a separate phase
3. **Dynamic Prioritization**: Emergent items are prioritized in questioning
4. **No Red Flag Phase**: Removes the need for a separate red flag screening section

## Migration

Use the migration script to update existing guidelines:

```bash
python3 scripts/migrate_guidelines_to_urgency.py [guidelines_directory]
```

The script will:
1. Add `urgency: "emergent"` field only to items that match red flag statements
2. Leave other items without urgency field (normal symptoms)
3. Remove the `red_flags` section from guideline files

## Engine Updates Needed

1. Remove red_flag phase logic
2. Prioritize questions with `urgency: "emergent"` when collecting missing OLDCARTS components
3. If any emergent items are present, elevate care level recommendation
4. Update question generation to highlight emergent symptoms when appropriate

## Future Enhancement (Phase 2)

When ready to scale up, the system can be extended to support:
- `urgency: "urgent"` - Requires urgent care (same-day clinic)
- `urgency: "routine"` - Can be managed with routine care (scheduled appointment)

The current structure is designed to be backward compatible - items without urgency field will be treated as routine/urgent based on context.

